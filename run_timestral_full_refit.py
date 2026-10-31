#!/usr/bin/env python3
"""
run_timestral_full_refit.py

Production Full-History Refit of Timestral-209M LoRA on all historical observations (Weeks 0..133)
and mathematically exact raw-scale blending with the Champion pipeline.

Aligned 100% with Cross-Validation Recipe:
- Pretrained base model: experiments/urgent_models/timestral_209m (untouched original weights)
- LoRA config: r=8, alpha=16, target_modules=["wQKV", "wO"], dropout=0.05, bias="none"
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-4)
- Schedule: 30-step linear warmup + cosine decay to 0.01 * lr
- Training budget: 600 steps predetermined a priori (no future facts used)
- History: Full panel Weeks 0..133 (134 weeks), valid origins k in [52, 125], context=52, horizon=9
- Sampling: StratifiedWindowSamplerFullHistory (seed=42)
- Zero future leakage: latest target index <= 133

Champion & Blending Protocol:
- Control submission: delivery_final/submission.csv (verified SHA256, strictly zero fallback)
- Champion raw continuous predictions recovered via the verified inference pipeline
  (0.45*TFT + 0.37*LightGBM_Dual + 0.09*CatBoost + 0.09*PerCategory)
- Zero-weight test: blend(w=0) MUST match delivery_final/submission.csv 100% with max delta == 0
- Honest CV blend formula:
  final = apply_p((1 - w) * champion_raw + w * timestral_raw, alpha=1.05, thresh=0.65)
  for w = 0.05 and w = 0.10.
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
sys.path.insert(0, str(ROOT / "forecast-container"))

from timestral import TimestralForecaster
from timestral.forecaster import pad_to_patch
from app.predictor import predict_from_history

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
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    return np.rint(gated).clip(0, None).astype(int)


def load_or_generate_champion_raw() -> pd.DataFrame:
    raw_parquet = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.parquet"
    raw_path = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    if raw_parquet.exists():
        print(f"Loading cached verified champion raw components: {raw_parquet.name}...")
        df_comp = pd.read_parquet(raw_parquet)
        assert len(df_comp) == 79101
        return df_comp
    if raw_path.exists():
        print(f"Loading cached verified champion raw components: {raw_path.name}...")
        df_comp = pd.read_csv(raw_path)
        assert len(df_comp) == 79101
        return df_comp

    print("Cached champion raw not found. Running full inference pipeline to recover raw components...")
    sales_file = ROOT / "sales_dataset_hakaton_train.csv"
    stock_file = ROOT / "stock_dataset_hakaton_train.csv"
    bundle_dir = ROOT / "forecast-container" / "model"

    sales_df = pd.read_csv(sales_file)
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = sales_df[sales_df["date"] <= pd.Timestamp("2026-07-26")].copy()

    stock_df = pd.read_csv(stock_file)
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    stock_df = stock_df[stock_df["date"] <= pd.Timestamp("2026-07-26")].copy()

    df_comp = predict_from_history(sales_df, stock_df, bundle_dir=bundle_dir, return_raw=True)
    assert len(df_comp) == 79101

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(raw_path, index=False, compression="gzip")
    return df_comp


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_base = ROOT / "experiments" / "timestral_production_model" / "refit_full_history"
    ckpt_dir = out_base / "checkpoints" / "final_refit"
    out_base.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("TIMESTRAL-209M PRODUCTION REFIT (WEEKS 0..133) & EXACT RAW BLENDING")
    print("=" * 85)

    # 1. Verify Control Champion Submission (STRICTLY NO FALLBACK)
    champ_sub_path = ROOT / "delivery_final" / "submission.csv"
    assert champ_sub_path.exists(), f"Primary champion submission missing: {champ_sub_path}"
    champ_bytes = champ_sub_path.read_bytes()
    champ_crlf_sha = sha256_bytes(champ_bytes)
    champ_lf_sha = sha256_bytes(champ_bytes.replace(b"\r\n", b"\n"))
    print(f"\n1. Verifying Control Champion Submission: {champ_sub_path.relative_to(ROOT)}")
    print(f"   Size: {len(champ_bytes):,} bytes | CRLF SHA256: {champ_crlf_sha} | LF SHA256: {champ_lf_sha}")
    assert champ_crlf_sha == CONTROL_SHA256_CRLF or champ_lf_sha == CONTROL_SHA256_LF, (
        f"Control submission hash mismatch! Got CRLF={champ_crlf_sha}, LF={champ_lf_sha}"
    )
    df_champ_sub = pd.read_csv(champ_sub_path)
    assert len(df_champ_sub) == 79101, f"Expected 79,101 rows, got {len(df_champ_sub)}"
    assert not df_champ_sub.duplicated(subset=KEYS).any(), "Duplicate keys in control submission!"
    print("   [PASS] Control champion submission verified 100% against official SHA256 manifest.")

    # 2. Recover Champion Raw Components via Inference Pipeline
    print("\n2. Recovering Continuous Champion Raw Components...")
    df_champ_components = load_or_generate_champion_raw()
    comp_tft = df_champ_components["raw_tft"].to_numpy(dtype=np.float64)
    comp_lgb = df_champ_components["raw_lgb"].to_numpy(dtype=np.float64)
    comp_cb = df_champ_components["raw_cb"].to_numpy(dtype=np.float64)
    comp_pcat = df_champ_components["raw_pcat"].to_numpy(dtype=np.float64)
    raw_champion = df_champ_components["raw_champion"].to_numpy(dtype=np.float64)

    expected_raw = 0.45 * comp_tft + 0.37 * comp_lgb + 0.09 * comp_cb + 0.09 * comp_pcat
    comp_diff = np.max(np.abs(raw_champion - expected_raw))
    print(f"   Component formula check: max |raw_champion - sum(w_i * comp_i)| = {comp_diff:.8e}")
    assert comp_diff < 1e-4, f"Component formula discrepancy: {comp_diff}"

    # 3. Assert Zero-Weight Test (w = 0)
    print("\n3. Executing Mandatory Zero-Weight Test (w = 0.0)...")
    reproduced_q_w0 = apply_p(raw_champion, 1.05, 0.65)
    merged_ctrl = pd.merge(
        df_champ_sub[KEYS + ["quantity"]],
        df_champ_components[KEYS].assign(quantity_w0=reproduced_q_w0),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged_ctrl) == 79101, "Merge key alignment failed!"
    abs_delta_w0 = np.abs(merged_ctrl["quantity"].values - merged_ctrl["quantity_w0"].values)
    max_delta_w0 = int(abs_delta_w0.max())
    diff_count_w0 = int((abs_delta_w0 != 0).sum())

    print(f"   Zero-weight test results:")
    print(f"     Rows evaluated: {len(merged_ctrl):,}")
    print(f"     Different rows: {diff_count_w0}")
    print(f"     Max absolute difference: {max_delta_w0}")
    assert max_delta_w0 == 0, f"MANDATORY TEST FAILURE: blend(w=0) differs from control by {max_delta_w0}!"
    print("   [PASS] Zero-weight test PASSED: 100% identity to delivery_final/submission.csv (diff = 0).")

    # 4. Load Data & Prepare Sampler
    print("\n4. Loading Historical Panel (Weeks 0..133)...")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    df_piv = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0)
    panel = df_piv.values.astype(np.float32)  # (134, 8789)
    weeks = df_piv.index.tolist()
    pairs = df_piv.columns.to_frame(index=False)
    n_weeks, n_series = panel.shape

    print(f"   Panel shape: {n_weeks} weeks x {n_series} series")
    print(f"   Historical weeks span: {weeks[0].strftime('%Y-%m-%d')} to {weeks[-1].strftime('%Y-%m-%d')}")

    sampler = StratifiedWindowSamplerFullHistory(panel, seed=42)
    print(f"   Valid origins count: {len(sampler.valid_origins)} (origins {sampler.valid_origins[0]}..{sampler.valid_origins[-1]})")
    print(f"   Zero future leakage verified: latest origin {sampler.valid_origins[-1]} target ends at week {sampler.valid_origins[-1] + HORIZON - 1} <= {n_weeks - 1}")

    # 5. Initialize Model from Original Untouched Pretrained Weights
    print("\n5. Initializing Timestral-209M from Pretrained Weights...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(str(model_dir))
    model = forecaster.model.to(device)

    # EXACT CV HYPERPARAMETERS
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["wQKV", "wO"],
        lora_dropout=0.05,  # Exactly aligned with CV
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   LoRA Config: r=8, alpha=16, modules=['wQKV', 'wO'], dropout=0.05")
    print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # Optimizer & LR Scheduler: AdamW 1e-4, weight_decay 1e-4, 30 warmup steps + cosine
    total_steps = 600
    warmup_steps = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    # 6. Training Loop
    microbatch = 32
    grad_accum = 2
    step = 0
    t_start = time.perf_counter()
    running_loss = 0.0

    print(f"\n6. Starting {total_steps}-step LoRA Training Loop (microbatch={microbatch}, grad_accum={grad_accum}, effective={microbatch * grad_accum})...")
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
        scheduler.step()
        optimizer.zero_grad()

        running_loss += step_loss
        step += 1

        if step % 50 == 0:
            avg_loss = running_loss / 50.0
            elapsed = time.perf_counter() - t_start
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Step {step:03d}/{total_steps} | Pinball Loss: {avg_loss:.4f} | LR: {current_lr:.6e} | Elapsed: {elapsed:.1f} s", flush=True)
            running_loss = 0.0

    train_time = time.perf_counter() - t_start
    print(f"\n   Training completed in {train_time:.1f} s ({train_time / 60:.2f} min)")

    # 7. Save Refit Adapter
    print(f"\n7. Saving final refit adapter to: {ckpt_dir}...")
    model.save_pretrained(str(ckpt_dir))
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_file)
    print(f"   Saved {adapter_file.name}: {adapter_file.stat().st_size:,} bytes | SHA256: {adapter_sha}")

    # 8. Future Forecast Generation (Weeks 2026-07-27 .. 2026-09-21)
    print("\n8. Generating Future 9-Week Forecast (Weeks 2026-07-27 .. 2026-09-21)...")
    forecaster.model = model
    forecaster.model.eval()

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
    quantiles_arr = {q: np.vstack(all_preds_q[q]) for q in QUANTILES}
    rint_preds = np.rint(continuous_preds).clip(0, None)
    legacy_p_preds = apply_p(continuous_preds, 1.05, 0.65)

    print("   Formatting keyed future predictions...")
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
    print(f"   Saved keyed future predictions: {future_csv.name} ({future_csv.stat().st_size:,} bytes | SHA256: {future_sha})")

    # 9. Blending with Champion Raw (Honest CV Recipe)
    print("\n9. Generating Production Candidate Submissions (w=0.05, w=0.10)...")
    merged_comp = pd.merge(
        df_champ_components[KEYS + ["raw_champion"]],
        future_df[KEYS + ["y_pred_continuous"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged_comp) == 79101, "Merge mismatch during blending!"

    raw_champ = merged_comp["raw_champion"].values.astype(np.float64)
    raw_tim = merged_comp["y_pred_continuous"].values.astype(np.float64)

    candidate_paths = {}
    candidate_meta = {}

    for w_blend in [0.05, 0.10]:
        # Exact CV formula: final = apply_p((1 - w)*raw_champion + w*raw_timestral, 1.05, 0.65)
        raw_blend = (1.0 - w_blend) * raw_champ + w_blend * raw_tim
        final_blend = apply_p(raw_blend, 1.05, 0.65)

        sub_candidate = merged_comp[KEYS].copy()
        sub_candidate["quantity"] = final_blend

        sub_file = ROOT / "submissions" / f"submission_champion_timestral_refit_blend_w{int(w_blend * 100):02d}.csv"
        # Write with CRLF to match submission standard on Windows
        sub_candidate.to_csv(sub_file, index=False, lineterminator="\r\n")
        raw_bytes = sub_file.read_bytes()
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        crlf_sha = sha256_bytes(raw_bytes)
        lf_sha = sha256_bytes(lf_bytes)

        candidate_paths[f"w{int(w_blend * 100):02d}"] = str(sub_file)
        candidate_meta[f"w{int(w_blend * 100):02d}"] = {
            "path": str(sub_file.relative_to(ROOT)),
            "weight_timestral": w_blend,
            "weight_champion": 1.0 - w_blend,
            "rows": len(sub_candidate),
            "quantity_sum": int(sub_candidate["quantity"].sum()),
            "crlf_bytes": len(raw_bytes),
            "crlf_sha256": crlf_sha,
            "lf_bytes": len(lf_bytes),
            "lf_sha256": lf_sha,
        }
        print(f"   Candidate (w={w_blend:.2f}) saved: {sub_file.name}")
        print(f"     CRLF: {len(raw_bytes):,} bytes | SHA256: {crlf_sha}")
        print(f"     LF:   {len(lf_bytes):,} bytes | SHA256: {lf_sha}")
        print(f"     Quantity sum: {int(sub_candidate['quantity'].sum()):,}")

    # 10. Write Complete Production Summary & Manifest
    summary = {
        "model_id": "mistralai/Timestral-209M",
        "training_mode": "Production Full-History Refit (Weeks 0..133)",
        "recipe_alignment": "100% Exact Cross-Validation Recipe Match",
        "hyperparameters": {
            "lora_rank": 8,
            "lora_alpha": 16,
            "target_modules": ["wQKV", "wO"],
            "lora_dropout": 0.05,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 30,
            "lr_schedule": "warmup_cosine_decay",
            "total_steps": total_steps,
            "microbatch_size": microbatch,
            "grad_accum_steps": grad_accum,
            "effective_batch_size": microbatch * grad_accum,
            "loss": "masked_pinball_loss_21_quantiles",
            "grad_clip_norm": 1.0,
            "seed": 42,
        },
        "training_scope": {
            "total_historical_weeks": n_weeks,
            "n_series": n_series,
            "valid_origins_count": len(sampler.valid_origins),
            "valid_origins": [sampler.valid_origins[0], sampler.valid_origins[-1]],
            "future_leakage_points": 0,
        },
        "champion_control": {
            "path": "delivery_final/submission.csv",
            "crlf_sha256": champ_crlf_sha,
            "lf_sha256": champ_lf_sha,
            "zero_weight_test_status": "PASSED_BIT_EXACT_ZERO_DIFF",
            "zero_weight_max_abs_diff": max_delta_w0,
            "zero_weight_different_rows": diff_count_w0,
        },
        "adapter_safetensors_sha256": adapter_sha,
        "future_predictions_sha256": future_sha,
        "adapter_artifacts": {
            "adapter_model_safetensors_path": str(adapter_file.relative_to(ROOT)),
            "adapter_model_safetensors_bytes": adapter_file.stat().st_size,
            "adapter_model_safetensors_sha256": adapter_sha,
        },
        "future_predictions": {
            "keyed_csv_path": str(future_csv.relative_to(ROOT)),
            "keyed_csv_bytes": future_csv.stat().st_size,
            "keyed_csv_sha256": future_sha,
            "future_weeks": FUTURE_WEEKS,
            "rows": 79101,
        },
        "candidates": candidate_meta,
        "training_time_sec": train_time,
    }

    summary_file = out_base / "production_refit_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved production summary: {summary_file.name}")

    print("\n" + "=" * 85)
    print("PRODUCTION FULL-HISTORY REFIT & BLENDING COMPLETED WITH ZERO ERRORS!")
    print("=" * 85)


if __name__ == "__main__":
    main()
