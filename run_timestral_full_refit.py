#!/usr/bin/env python3
"""
run_timestral_full_refit.py

Production Full-History Refit of Timestral-209M LoRA on all historical observations (Weeks 0..133).
Guaranteed Zero Future Leakage:
- Training origins: k in [52, 125].
- Max target window: [125, 133] strictly <= 133.
- Horizon: 9 weeks.
- Context: 52 weeks.

Inference:
- Context window: Weeks 82..133 (last 52 historical weeks).
- Future 9-week horizon: 2026-07-27 to 2026-09-21.
- Outputs: 21 quantiles, median continuous prediction, rint, legacy_p.
- Blending: Blends with champion submission at optimal weight w*.
"""

from __future__ import annotations

import sys
import json
import time
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from timestral import TimestralForecaster
from timestral.forecaster import pad_to_patch

KEYS = ["week", "warehouse_name", "article"]
HORIZON = 9
CONTEXT = 52
QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]
FUTURE_WEEKS = [
    "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17",
    "2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21"
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def pinball_loss_21(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    y = target.unsqueeze(-1)
    err = y - preds
    q = quantiles.view(1, 1, -1)
    loss = torch.max(q * err, (q - 1.0) * err)
    if mask is not None:
        loss = loss[mask.unsqueeze(-1).expand_as(loss)]
    return loss.mean()


class StratifiedWindowSamplerFullHistory:
    def __init__(self, panel_train: np.ndarray, seed: int = 42):
        self.panel = panel_train  # (T_train, N_series), here (134, 8789)
        self.n_weeks, self.n_series = panel_train.shape
        self.rng = np.random.RandomState(seed)

        sums = panel_train.sum(axis=0)
        nz_rate = (panel_train > 0).mean(axis=0)

        p90 = np.percentile(sums, 90)
        self.idx_high = np.where(sums >= p90)[0]
        self.idx_zero = np.where((nz_rate < 0.15) & (sums < p90))[0]
        self.idx_reg = np.where((nz_rate >= 0.15) & (sums < p90))[0]

        # Valid origins: context 52, horizon 9. Max target index: k + 9 - 1 <= n_weeks - 1 -> k <= n_weeks - 9
        self.valid_origins = list(range(CONTEXT, self.n_weeks - HORIZON + 1))
        assert len(self.valid_origins) > 0, f"No valid origins for training history of length {self.n_weeks}!"

        self.seen_series = set()
        self.seen_origins = set()
        self.seen_windows = set()
        self.total_windows_sampled = 0
        self.target_zero_count = 0
        self.target_total_points = 0
        self.target_volume_sum = 0.0

    def sample_batch(self, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, int]:
        n_high = max(1, int(batch_size * 0.20))
        n_zero = max(1, int(batch_size * 0.40))
        n_reg = batch_size - n_high - n_zero

        high_sample = self.rng.choice(self.idx_high, size=n_high, replace=False)
        zero_sample = self.rng.choice(self.idx_zero, size=n_zero, replace=False)
        reg_sample = self.rng.choice(self.idx_reg, size=n_reg, replace=False)

        batch_series_idx = np.concatenate([high_sample, zero_sample, reg_sample])
        self.rng.shuffle(batch_series_idx)

        origin_k = int(self.rng.choice(self.valid_origins))

        ctx = self.panel[origin_k - CONTEXT : origin_k, batch_series_idx].T
        tgt = self.panel[origin_k : origin_k + HORIZON, batch_series_idx].T

        for s_idx in batch_series_idx:
            self.seen_series.add(int(s_idx))
            self.seen_windows.add((int(s_idx), origin_k))
        self.seen_origins.add(origin_k)
        self.total_windows_sampled += batch_size

        self.target_zero_count += int((tgt == 0).sum())
        self.target_total_points += tgt.size
        self.target_volume_sum += float(tgt.sum())

        return ctx, tgt, origin_k


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_base = ROOT / "experiments" / "timestral_production_model" / "refit_full_history"
    ckpt_dir = out_base / "checkpoints" / "final_refit"
    out_base.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TIMESTRAL-209M PRODUCTION FULL-HISTORY REFIT (WEEKS 0..133)")
    print("=" * 80)

    # 1. Load Data
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading historical weekly dataset: {data_path.name}...")
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    df_piv = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0)
    panel = df_piv.values.astype(np.float32)  # (134, 8789)
    weeks = df_piv.index.tolist()
    pairs = df_piv.columns.to_frame(index=False)
    n_weeks, n_series = panel.shape

    print(f"Panel shape: {n_weeks} weeks x {n_series} series")
    print(f"Weeks span: {weeks[0].strftime('%Y-%m-%d')} to {weeks[-1].strftime('%Y-%m-%d')}")

    # 2. Setup Sampler
    sampler = StratifiedWindowSamplerFullHistory(panel, seed=42)
    print(f"Valid full-history training origins: {len(sampler.valid_origins)} origins (origins {sampler.valid_origins[0]}..{sampler.valid_origins[-1]})")
    print(f"Zero future leakage verified: latest origin {sampler.valid_origins[-1]} target window ends at week {sampler.valid_origins[-1] + HORIZON - 1} <= {n_weeks - 1}")

    # 3. Model Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(str(model_dir))
    model = forecaster.model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["wQKV", "wO"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA initialized: {trainable_params:,} trainable / {total_params:,} total parameters ({trainable_params / total_params * 100:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    # 4. Training Loop
    total_steps = 600
    microbatch = 32
    grad_accum = 2
    step = 0
    t_start = time.perf_counter()
    running_loss = 0.0

    print(f"\nStarting 600-step training loop (microbatch={microbatch}, grad_accum={grad_accum}, effective={microbatch * grad_accum})...")
    optimizer.zero_grad()

    while step < total_steps:
        model.train()
        step_loss = 0.0
        for _ in range(grad_accum):
            ctx_np, tgt_np, _ = sampler.sample_batch(microbatch)
            ctx_t = torch.as_tensor(ctx_np, dtype=torch.float32, device=device)
            mask_t = torch.ones_like(ctx_t, dtype=torch.bool)
            inputs_pad, mask_pad = pad_to_patch(ctx_t, mask_t, patch_size=32)
            group_ids = torch.arange(microbatch, device=device)

            tgt_t = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
            tgt_mask = torch.ones_like(tgt_t, dtype=torch.bool)

            preds = model.predict_block(inputs_pad, mask_pad, group_ids, horizon_patches=1)[:, :HORIZON, :]
            loss_mb = pinball_loss_21(preds, tgt_t, q_tensor, tgt_mask) / grad_accum
            loss_mb.backward()
            step_loss += loss_mb.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += step_loss
        step += 1

        if step % 50 == 0:
            avg_loss = running_loss / 50.0
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step:03d}/{total_steps} | Avg Pinball Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f} s", flush=True)
            running_loss = 0.0

    train_time = time.perf_counter() - t_start
    print(f"\nTraining completed in {train_time:.1f} s ({train_time / 60:.2f} min)")

    # 5. Save Adapter
    print(f"\nSaving final refit adapter to: {ckpt_dir}...")
    model.save_pretrained(str(ckpt_dir))
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_file)
    print(f"Saved {adapter_file.name}: {adapter_file.stat().st_size:,} bytes | SHA256: {adapter_sha}")

    # 6. Future Forecast Generation
    print("\nGenerating Future 9-Week Forecast (Weeks 2026-07-27 .. 2026-09-21)...")
    forecaster.model = model
    forecaster.model.eval()

    # Context: weeks 82..133 (52 weeks)
    inf_ctx = panel[-CONTEXT:, :].T  # (8789, 52)
    inf_batch_size = 64
    all_preds_med = []
    all_preds_q = {q: [] for q in QUANTILES}

    with torch.no_grad():
        for i in range(0, n_series, inf_batch_size):
            b_ctx = torch.as_tensor(inf_ctx[i : i + inf_batch_size], dtype=torch.float32, device=device)
            fc = forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            med = fc.median[:, 0, :].detach().cpu().numpy()
            all_preds_med.append(med)
            for q in QUANTILES:
                qv = fc.quantile(q)[:, 0, :].detach().cpu().numpy()
                all_preds_q[q].append(qv)

    continuous_preds = np.vstack(all_preds_med)  # (8789, 9)
    quantiles_arr = {q: np.vstack(all_preds_q[q]) for q in QUANTILES}  # each (8789, 9)
    rint_preds = np.rint(continuous_preds).clip(0, None)
    legacy_p_preds = apply_p(continuous_preds, 1.05, 0.65)

    # Format keyed DataFrame
    print("Formatting keyed predictions...")
    rows = []
    for h_idx, w_str in enumerate(FUTURE_WEEKS):
        df_h = pairs.copy()
        df_h.insert(0, "week", w_str)
        df_h["y_pred_continuous"] = continuous_preds[:, h_idx]
        df_h["y_pred_rint"] = rint_preds[:, h_idx].astype(int)
        df_h["y_pred_legacy_p"] = legacy_p_preds[:, h_idx].astype(int)
        for q_val in QUANTILES:
            col_name = f"q{int(round(q_val * 100)):02d}"
            df_h[col_name] = quantiles_arr[q_val][:, h_idx]
        rows.append(df_h)

    future_df = pd.concat(rows, ignore_index=True)
    assert len(future_df) == 79101, f"Row count mismatch: {len(future_df)}"
    assert not future_df.duplicated(subset=KEYS).any(), "Duplicate keys!"
    assert not future_df[KEYS].isna().any().any(), "NaN keys!"

    future_csv = out_base / "timestral_209m_future_9w_keyed.csv.gz"
    future_df.to_csv(future_csv, index=False, compression="gzip")
    future_sha = sha256_file(future_csv)
    print(f"Saved keyed future predictions: {future_csv.name} ({future_csv.stat().st_size:,} bytes | SHA256: {future_sha})")

    # 7. Create Blended Candidate Submission
    champ_sub_path = ROOT / "submissions" / "submission_final_pumpline.csv"
    if not champ_sub_path.exists():
        champ_sub_path = ROOT / "delivery_final" / "submission.csv"

    print(f"\nCreating Blended Candidate Submission against: {champ_sub_path.name}...")
    df_champ_sub = pd.read_csv(champ_sub_path)
    merged_sub = pd.merge(df_champ_sub[KEYS + ["quantity"]], future_df[KEYS + ["y_pred_continuous"]], on=KEYS, validate="one_to_one")
    assert len(merged_sub) == 79101

    raw_champ = merged_sub["quantity"].astype(np.float64).values
    raw_tim = merged_sub["y_pred_continuous"].values

    for w_blend in [0.05, 0.10]:
        raw_blend = (1.0 - w_blend) * raw_champ + w_blend * raw_tim
        final_blend = apply_p(raw_blend, 1.05, 0.65).astype(int)

        sub_candidate = merged_sub[KEYS].copy()
        sub_candidate["quantity"] = final_blend

        sub_file = ROOT / "submissions" / f"submission_champion_timestral_refit_blend_w{int(w_blend * 100):02d}.csv"
        sub_candidate.to_csv(sub_file, index=False)
        sub_sha = sha256_file(sub_file)
        print(f"  Candidate Submission (w={w_blend:.2f}) saved: {sub_file.name} ({sub_file.stat().st_size:,} bytes | SHA256: {sub_sha})")

    # 8. Manifest & Results Summary
    summary = {
        "model_id": "mistralai/Timestral-209M",
        "training_mode": "Production Full-History Refit (Weeks 0..133)",
        "total_historical_weeks": n_weeks,
        "n_series": n_series,
        "valid_origins_count": len(sampler.valid_origins),
        "steps": total_steps,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "trainable_parameters": trainable_params,
        "training_time_sec": train_time,
        "adapter_safetensors_sha256": adapter_sha,
        "future_predictions_sha256": future_sha,
        "future_weeks": FUTURE_WEEKS,
        "rows": 79101,
    }
    with open(out_base / "production_refit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PRODUCTION FULL-HISTORY REFIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
