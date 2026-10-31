#!/usr/bin/env python3
"""
run_timestral_lora_fold.py

Controlled Timestral-209M LoRA Fine-Tuning and Evaluation on any competition fold (Fold 0 to 4).

Temporal Splitting Protocol:
- Fold 4: origin = 89  (external test: 89..97, internal val: 80..88, train targets <= 79)
- Fold 3: origin = 98  (external test: 98..106, internal val: 89..97, train targets <= 88)
- Fold 2: origin = 107 (external test: 107..115, internal val: 98..106, train targets <= 97)
- Fold 1: origin = 116 (external test: 116..124, internal val: 107..115, train targets <= 106)
- Fold 0: origin = 125 (external test: 125..133, internal val: 116..124, train targets <= 115)

Zero Leakage Guarantee:
- Training origins: k in [CONTEXT, val_start - HORIZON].
- Target window: [k, k + HORIZON - 1] strictly <= val_start - 1.
- Internal validation window: [val_start, origin - 1].
- External test window: [origin, origin + HORIZON - 1].
- Overlap across all boundaries: 0.0%.
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
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
HORIZON = 9
CONTEXT = 52
QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def calc_scores(m_pred: dict, m_base: dict) -> tuple[float, float]:
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def pinball_loss_21(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    y = target.unsqueeze(-1)
    err = y - preds
    q = quantiles.view(1, 1, -1)
    loss = torch.max(q * err, (q - 1.0) * err)
    if mask is not None:
        loss = loss[mask.unsqueeze(-1).expand_as(loss)]
    return loss.mean()


class StratifiedWindowSampler:
    def __init__(self, panel_train: np.ndarray, seed: int = 42):
        self.panel = panel_train  # (T_train, N_series)
        self.n_weeks, self.n_series = panel_train.shape
        self.rng = np.random.RandomState(seed)

        sums = panel_train.sum(axis=0)
        nz_rate = (panel_train > 0).mean(axis=0)

        p90 = np.percentile(sums, 90)
        self.idx_high = np.where(sums >= p90)[0]
        self.idx_zero = np.where((nz_rate < 0.15) & (sums < p90))[0]
        self.idx_reg = np.where((nz_rate >= 0.15) & (sums < p90))[0]

        # Valid origins: context 52, horizon 9. Max target index: k + 9 - 1 <= n_weeks - 1 -> k <= n_weeks - 9.
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

    def get_coverage_stats(self) -> dict:
        total_unique_possible_windows = len(self.valid_origins) * self.n_series
        repeat_rate = (
            (self.total_windows_sampled - len(self.seen_windows)) / self.total_windows_sampled
            if self.total_windows_sampled > 0 else 0.0
        )
        zero_rate = (
            self.target_zero_count / self.target_total_points
            if self.target_total_points > 0 else 0.0
        )
        mean_volume = (
            self.target_volume_sum / self.target_total_points
            if self.target_total_points > 0 else 0.0
        )

        return {
            "total_windows_sampled": self.total_windows_sampled,
            "unique_windows_seen": len(self.seen_windows),
            "unique_possible_windows": total_unique_possible_windows,
            "window_coverage_pct": 100.0 * len(self.seen_windows) / total_unique_possible_windows,
            "unique_series_seen": len(self.seen_series),
            "series_coverage_pct": 100.0 * len(self.seen_series) / self.n_series,
            "unique_origins_seen": len(self.seen_origins),
            "origins_coverage_pct": 100.0 * len(self.seen_origins) / len(self.valid_origins),
            "repeat_rate_pct": 100.0 * repeat_rate,
            "target_zero_rate_pct": 100.0 * zero_rate,
            "target_mean_volume": mean_volume,
        }


def evaluate_forecast(
    forecaster: TimestralForecaster,
    ctx_tensor: torch.Tensor,
    truth_df: pd.DataFrame,
    base_df: pd.DataFrame,
    keys_df: pd.DataFrame,
    batch_size: int = 128,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    n_series = len(ctx_tensor)
    all_preds_med = []
    all_preds_q = {q: [] for q in QUANTILES}

    forecaster.model.eval()
    with torch.inference_mode():
        for i in range(0, n_series, batch_size):
            b_ctx = ctx_tensor[i : i + batch_size]
            fc = forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            med = fc.median[:, 0, :].detach().cpu().numpy()
            all_preds_med.append(med)
            for q in QUANTILES:
                qv = fc.quantile(q)[:, 0, :].detach().cpu().numpy()
                all_preds_q[q].append(qv)

    arr_med = np.vstack(all_preds_med)  # (8789, 9)

    records_med = [arr_med[:, h] for h in range(HORIZON)]
    records_q = {q: [np.vstack(all_preds_q[q])[:, h] for h in range(HORIZON)] for q in QUANTILES}

    y_med_raw = np.maximum(0.0, np.concatenate(records_med))
    y_rint = np.rint(y_med_raw)
    y_leg = apply_p(y_med_raw, 1.05, 0.65)

    m_base = compute_all_metrics_fast(truth_df, base_df)

    # 1. Rint
    df_rint = keys_df.copy()
    df_rint["quantity"] = y_rint
    m_rint = compute_all_metrics_fast(truth_df, df_rint)
    c_rint, u_rint = calc_scores(m_rint, m_base)
    eval_rint = {"clipped_score": c_rint, "unclipped_ratio": u_rint, **m_rint}

    # 2. Legacy P
    df_leg = keys_df.copy()
    df_leg["quantity"] = y_leg
    m_leg = compute_all_metrics_fast(truth_df, df_leg)
    c_leg, u_leg = calc_scores(m_leg, m_base)
    eval_leg = {"clipped_score": c_leg, "unclipped_ratio": u_leg, **m_leg}

    q_dict = {f"q_{q:g}": np.concatenate(records_q[q]) for q in QUANTILES}

    return eval_rint, eval_leg, y_med_raw, y_rint, y_leg, q_dict


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Timestral-209M LoRA Fine-Tuning across Folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 4, default: 0)")
    parser.add_argument("--steps", type=int, default=600, help="Training steps (default: 600)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Microbatch size (default: 32)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--eval-steps", type=int, default=150, help="Evaluation frequency (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--run-id", type=str, default="pilot", help="Unique run identifier (default: 'pilot')")
    args = parser.parse_args()

    fold = args.fold
    origin = FOLD_ORIGINS[fold]
    run_id = args.run_id

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print(f"TIMESTRAL-209M LORA FINE-TUNING [FOLD {fold}, ORIGIN {origin}, RUN_ID: {run_id}]")
    print(f"Configuration: steps={args.steps}, lr={args.lr}, microbatch={args.batch_size}, grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Weights Verification
    weights_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    safetensors_file = weights_dir / "model.safetensors"
    EXPECTED_SIZE = 837271648
    EXPECTED_SHA256 = "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"
    actual_size = safetensors_file.stat().st_size
    actual_sha = sha256_file(safetensors_file)
    assert actual_size == EXPECTED_SIZE and actual_sha.lower() == EXPECTED_SHA256.lower(), "Weights verification failed!"
    print("  [PASS] Authentic pretrained Timestral-209M weights verified bit-exact!")

    # 2. Output Directories
    out_dir = ROOT / "experiments" / f"timestral_lora_fold{fold}"
    ckpts_dir = out_dir / "checkpoints" / run_id
    keyed_dir = out_dir / "keyed_predictions" / run_id
    logs_dir = out_dir / "logs" / run_id
    for d in [out_dir, ckpts_dir, keyed_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Dataset Loading & Temporal Splitting
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    dataset_sha = sha256_file(data_path)
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    panel = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0).sort_index()
    n_wks, n_pairs = panel.shape
    pairs_df = panel.columns.to_frame(index=False)
    all_weeks = panel.index

    # Temporal isolation definition:
    val_start = origin - HORIZON
    train_history_end = val_start  # train history strictly up to val_start

    print(f"\nTemporal Isolation Architecture for Fold {fold}:")
    print(f"  Training History Window:  Weeks 0..{train_history_end - 1} ({train_history_end} weeks, max target <= {train_history_end - 1})")
    print(f"  Internal Val Horizon:     Weeks {val_start}..{origin - 1} ({all_weeks[val_start].strftime('%Y-%m-%d')} to {all_weeks[origin - 1].strftime('%Y-%m-%d')})")
    print(f"  External Test Horizon:    Weeks {origin}..{origin + HORIZON - 1} ({all_weeks[origin].strftime('%Y-%m-%d')} to {all_weeks[origin + HORIZON - 1].strftime('%Y-%m-%d')})")
    assert train_history_end <= val_start < origin < origin + HORIZON <= n_wks, "Temporal isolation failed!"
    print("  [PASS] Temporal isolation verified: 0.0% overlap across train, internal val, external test!")

    panel_train_np = panel.iloc[:train_history_end].values

    # Internal Val Context & Truth
    val_int_weeks_str = [all_weeks[val_start + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    val_int_keys = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    val_int_keys.insert(0, "week", np.repeat(val_int_weeks_str, n_pairs))

    val_int_truth_list = [panel.iloc[val_start + h].values for h in range(HORIZON)]
    val_int_m13_list = [panel.iloc[val_start - 13 : val_start].mean().values for _ in range(HORIZON)]

    val_int_truth_df = val_int_keys.copy()
    val_int_truth_df["quantity"] = np.concatenate(val_int_truth_list)
    val_int_base_df = val_int_keys.copy()
    val_int_base_df["quantity"] = np.concatenate(val_int_m13_list)

    val_int_ctx_np = panel.iloc[val_start - CONTEXT : val_start].values.T
    val_int_ctx_t = torch.as_tensor(val_int_ctx_np, dtype=torch.float32, device=device)

    # External Test Context & Truth
    ext_weeks_str = [all_weeks[origin + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    ext_keys = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    ext_keys.insert(0, "week", np.repeat(ext_weeks_str, n_pairs))

    ext_truth_list = [panel.iloc[origin + h].values for h in range(HORIZON)]
    ext_m13_list = [panel.iloc[origin - 13 : origin].mean().values for _ in range(HORIZON)]

    ext_truth_df = ext_keys.copy()
    ext_truth_df["quantity"] = np.concatenate(ext_truth_list)
    ext_base_df = ext_keys.copy()
    ext_base_df["quantity"] = np.concatenate(ext_m13_list)

    ext_ctx_np = panel.iloc[origin - CONTEXT : origin].values.T
    ext_ctx_t = torch.as_tensor(ext_ctx_np, dtype=torch.float32, device=device)

    # 4. Zero-Shot Baseline on External Test Horizon
    print(f"\n--- Evaluating Pretrained Timestral-209M Zero-Shot on Fold {fold} ---")
    forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    t0_zs = time.perf_counter()
    eval_zs_rint, eval_zs_leg, _, _, _, _ = evaluate_forecast(
        forecaster, ext_ctx_t, ext_truth_df, ext_base_df, ext_keys
    )
    print(f"Zero-Shot Fold {fold} Evaluation in {time.perf_counter() - t0_zs:.2f} s")
    print(f"  Zero-Shot Rint:     Clipped = {eval_zs_rint['clipped_score']:.6f}, Unclipped = {eval_zs_rint['unclipped_ratio']:.6f}, SMAPE = {eval_zs_rint['smape']:.3f}%, WAPE = {eval_zs_rint['wape']:.3f}%, RMSE = {eval_zs_rint['rmse']:.4f}")
    print(f"  Zero-Shot Legacy P: Clipped = {eval_zs_leg['clipped_score']:.6f}, Unclipped = {eval_zs_leg['unclipped_ratio']:.6f}, SMAPE = {eval_zs_leg['smape']:.3f}%, WAPE = {eval_zs_leg['wape']:.3f}%, RMSE = {eval_zs_leg['rmse']:.4f}")

    # 5. Initialize Fresh LoRA Model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["wQKV", "wO"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(forecaster.model, lora_config)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    print(f"\nTrainable LoRA parameters: {trainable_params:,} ({100.0 * trainable_params / all_params:.4f}%)")

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_steps = 30
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    quantiles_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    # 6. Training Loop with Stratified Sampler & Diagnostic Internal Validation
    print(f"\n--- Starting Controlled LoRA Fine-Tuning for Fold {fold} ---")
    sampler = StratifiedWindowSampler(panel_train_np, seed=args.seed)

    training_logs = []
    checkpoint_records = []
    best_val_unclipped = float("inf")
    best_checkpoint_path = None
    best_step = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_train_start = time.perf_counter()
    optimizer.zero_grad()

    for step in range(1, args.steps + 1):
        peft_model.train()
        step_loss = 0.0

        for _ in range(args.grad_accum):
            b_ctx_np, b_tgt_np, origin_k = sampler.sample_batch(args.batch_size)
            b_ctx_t = torch.as_tensor(b_ctx_np, dtype=torch.float32, device=device)
            b_mask_t = torch.ones_like(b_ctx_t, dtype=torch.bool)
            b_inputs_pad, b_mask_pad = pad_to_patch(b_ctx_t, b_mask_t, patch_size=32)
            b_group_ids = torch.arange(args.batch_size, device=device)

            b_tgt_t = torch.as_tensor(b_tgt_np, dtype=torch.float32, device=device)
            b_tgt_mask = torch.ones_like(b_tgt_t, dtype=torch.bool)

            preds = peft_model.predict_block(b_inputs_pad, b_mask_pad, b_group_ids, horizon_patches=1)[:, :HORIZON, :]
            loss_mb = pinball_loss_21(preds, b_tgt_t, quantiles_t, b_tgt_mask) / args.grad_accum

            loss_mb.backward()
            step_loss += loss_mb.item()

        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        curr_lr = scheduler.get_last_lr()[0]
        training_logs.append({"step": step, "train_loss": step_loss, "lr": curr_lr})

        if step % 25 == 0 or step == args.steps:
            print(f"Step {step:4d}/{args.steps} | Train Loss: {step_loss:.6f} | LR: {curr_lr:.3e}")

        # Checkpoint evaluation
        if step % args.eval_steps == 0 or step == args.steps:
            print(f"\n--- [Evaluation at Step {step}] Internal Validation (Weeks {val_start}..{origin - 1}) ---")
            forecaster.model = peft_model
            t0_eval = time.perf_counter()
            val_rint, val_leg, _, _, _, _ = evaluate_forecast(
                forecaster, val_int_ctx_t, val_int_truth_df, val_int_base_df, val_int_keys
            )
            t_eval_elapsed = time.perf_counter() - t0_eval

            ckpt_dir = ckpts_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(ckpt_dir)

            record = {
                "step": step,
                "train_loss": step_loss,
                "lr": curr_lr,
                "internal_val_eval_sec": t_eval_elapsed,
                "internal_val_rint": val_rint,
                "internal_val_legacy_p": val_leg,
                "checkpoint_path": str(ckpt_dir),
            }
            checkpoint_records.append(record)

            print(f"  Step {step} Val Rint:     Clipped = {val_rint['clipped_score']:.6f}, Unclipped = {val_rint['unclipped_ratio']:.6f}")
            print(f"  Step {step} Val Legacy P: Clipped = {val_leg['clipped_score']:.6f}, Unclipped = {val_leg['unclipped_ratio']:.6f}")

            score_to_track = val_leg["unclipped_ratio"]
            if score_to_track < best_val_unclipped:
                best_val_unclipped = score_to_track
                best_checkpoint_path = ckpt_dir
                best_step = step
                print(f"  >>> NEW BEST CHECKPOINT: Step {step} (Unclipped Ratio = {score_to_track:.6f})")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train_total = time.perf_counter() - t_train_start
    train_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    print(f"\nFine-tuning completed in {t_train_total:.2f} s ({t_train_total / 60:.2f} min)")
    print(f"Peak VRAM during training: {train_peak_vram_mb:.2f} MB")
    print(f"Selected BEST checkpoint strictly by internal validation: Step {best_step} ({best_checkpoint_path})")

    # 7. Evaluate Selected Checkpoint on External Fold Test Horizon
    print(f"\n--- Evaluating Best Checkpoint (Step {best_step}) on External Fold {fold} ---")
    fresh_forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    fresh_forecaster.model = PeftModel.from_pretrained(fresh_forecaster.model, str(best_checkpoint_path))
    fresh_forecaster.model.eval()

    t0_ext = time.perf_counter()
    ext_ft_rint, ext_ft_leg, ext_raw, ext_rint, ext_leg, ext_q = evaluate_forecast(
        fresh_forecaster, ext_ctx_t, ext_truth_df, ext_base_df, ext_keys
    )
    print(f"External Fold {fold} Evaluation completed in {time.perf_counter() - t0_ext:.2f} s")

    df_raw_ext = ext_keys.copy()
    df_raw_ext["quantity"] = ext_raw
    m_raw = compute_all_metrics_fast(ext_truth_df, df_raw_ext)
    m_base_ext = compute_all_metrics_fast(ext_truth_df, ext_base_df)
    c_raw, u_raw = calc_scores(m_raw, m_base_ext)
    ext_ft_raw = {"clipped_score": c_raw, "unclipped_ratio": u_raw, **m_raw}

    print(f"  Fine-Tuned Raw:      Clipped = {ext_ft_raw['clipped_score']:.6f}, Unclipped = {ext_ft_raw['unclipped_ratio']:.6f}")
    print(f"  Fine-Tuned Rint:     Clipped = {ext_ft_rint['clipped_score']:.6f}, Unclipped = {ext_ft_rint['unclipped_ratio']:.6f}, SMAPE = {ext_ft_rint['smape']:.3f}%, WAPE = {ext_ft_rint['wape']:.3f}%, RMSE = {ext_ft_rint['rmse']:.4f}")
    print(f"  Fine-Tuned Legacy P: Clipped = {ext_ft_leg['clipped_score']:.6f}, Unclipped = {ext_ft_leg['unclipped_ratio']:.6f}, SMAPE = {ext_ft_leg['smape']:.3f}%, WAPE = {ext_ft_leg['wape']:.3f}%, RMSE = {ext_ft_leg['rmse']:.4f}")

    # 8. Save Keyed Predictions
    keyed_df = ext_keys.copy()
    keyed_df["y_true"] = ext_truth_df["quantity"].values
    keyed_df["unconstrained_demand_13"] = ext_base_df["quantity"].values
    keyed_df["y_pred_continuous"] = ext_raw
    keyed_df["y_pred_rint"] = ext_rint
    keyed_df["y_pred_legacy_p"] = ext_leg
    for col_name, arr_vals in ext_q.items():
        keyed_df[col_name] = arr_vals

    assert len(keyed_df) == 79101, f"Expected 79,101 rows, got {len(keyed_df)}"
    assert not keyed_df.duplicated(subset=KEYS).any(), "Duplicate keys!"
    assert not keyed_df[KEYS].isna().any().any(), "NaN keys!"

    df_raw_verify = df_raw.copy()
    if pd.api.types.is_datetime64_any_dtype(df_raw_verify["week"]):
        df_raw_verify["week"] = df_raw_verify["week"].dt.strftime("%Y-%m-%d")
    raw_truth_merged = pd.merge(keyed_df[KEYS + ["y_true"]], df_raw_verify[KEYS + ["quantity"]], on=KEYS, how="left", validate="one_to_one")
    raw_diff = float((raw_truth_merged["y_true"] - raw_truth_merged["quantity"].fillna(0.0)).abs().max())
    assert raw_diff == 0.0, f"Discrepancy with raw dataset: {raw_diff}"
    print(f"  [PASS] Keyed predictions verified: 79,101 rows, 0.0 discrepancy with raw dataset!")

    keyed_filename = f"timestral_209m_lora_fold{fold}_{run_id}_keyed.csv.gz"
    keyed_filepath = keyed_dir / keyed_filename
    keyed_df.to_csv(keyed_filepath, index=False, compression="gzip")
    keyed_sha = sha256_file(keyed_filepath)
    print(f"Keyed predictions saved: {keyed_filepath.name} ({keyed_filepath.stat().st_size:,} bytes, SHA256={keyed_sha})")

    # 9. Blending Evaluation with Baseline Ensemble
    print(f"\n--- [Blending Evaluation on Fold {fold}] ---")
    blend_path = ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
    blending_results = []
    if blend_path.exists():
        df_b = pd.read_csv(blend_path)
        m_blend = pd.merge(
            df_b[KEYS + ["raw_blend"]],
            keyed_df[KEYS + ["y_true", "unconstrained_demand_13", "y_pred_continuous"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        assert len(m_blend) == 79101, "Merge mismatch with blend file!"
        raw_ens = m_blend["raw_blend"].values
        raw_tim = m_blend["y_pred_continuous"].values

        b_truth = m_blend[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
        b_base = m_blend[KEYS + ["unconstrained_demand_13"]].rename(columns={"unconstrained_demand_13": "quantity"})
        b_m_base = compute_all_metrics_fast(b_truth, b_base)

        print(f"{'Weight Timestral':<18} {'Clipped':<12} {'Unclipped':<12} {'SMAPE (%)':<12} {'WAPE (%)':<12} {'RMSE':<10}")
        print("-" * 78)
        for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
            r_mix = (1.0 - w) * raw_ens + w * raw_tim
            p_mix = apply_p(r_mix, 1.05, 0.65)
            mix_df = m_blend[KEYS].copy()
            mix_df["quantity"] = p_mix
            m_mix = compute_all_metrics_fast(b_truth, mix_df)
            c_mix, u_mix = calc_scores(m_mix, b_m_base)
            print(f"{w:<18.2f} {c_mix:<12.6f} {u_mix:<12.6f} {m_mix['smape']:<12.3f} {m_mix['wape']:<12.3f} {m_mix['rmse']:<10.4f}")
            blending_results.append({"weight_timestral": w, "clipped": c_mix, "unclipped": u_mix, **m_mix})

    # Save Experiment JSON
    experiment_results = {
        "experiment_name": f"timestral_209m_lora_fold{fold}_{run_id}",
        "fold": fold,
        "origin": origin,
        "run_id": run_id,
        "weights_sha256": actual_sha,
        "dataset_sha256": dataset_sha,
        "keyed_predictions_sha256": keyed_sha,
        "training_time_sec": t_train_total,
        "peak_vram_mb": train_peak_vram_mb,
        "best_checkpoint": {
            "step": best_step,
            "path": str(Path(f"experiments/timestral_lora_fold{fold}/checkpoints") / run_id / f"step_{best_step}"),
            "internal_val_unclipped": best_val_unclipped,
        },
        "coverage_statistics": sampler.get_coverage_stats(),
        "internal_validation_checkpoints": checkpoint_records,
        "external_evaluation": {
            "zero_shot_rint": eval_zs_rint,
            "zero_shot_legacy_p": eval_zs_leg,
            "fine_tuned_raw": ext_ft_raw,
            "fine_tuned_rint": ext_ft_rint,
            "fine_tuned_legacy_p": ext_ft_leg,
        },
        "blending_with_ensemble": blending_results,
    }

    results_json_path = logs_dir / f"timestral_lora_fold{fold}_{run_id}_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nExperiment JSON saved to: {results_json_path}")

    # Summary CSV
    summary_rows = [
        {"model": "Timestral_209M_ZeroShot", "variant": "rint", **eval_zs_rint, "step": 0},
        {"model": "Timestral_209M_ZeroShot", "variant": "legacy_p", **eval_zs_leg, "step": 0},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "raw_continuous", **ext_ft_raw, "step": best_step},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "rint", **ext_ft_rint, "step": best_step},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "legacy_p", **ext_ft_leg, "step": best_step},
    ]
    summary_csv_path = logs_dir / f"timestral_lora_fold{fold}_{run_id}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
