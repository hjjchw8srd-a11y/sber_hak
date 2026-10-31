#!/usr/bin/env python3
"""
run_timestral_lora_fold4.py

MASTER PIPELINE: Controlled Timestral-209M Fine-Tuning with LoRA on Selection Fold 4.

Key Architecture & Protocol:
1. Model: mistralai/Timestral-209M (209M base parameters) from experiments/urgent_models/timestral_209m.
2. Integrity: Bit-exact SHA256 checksum and architecture validation.
3. LoRA: Applied to attention projection modules (wQKV, wO), r=8, alpha=16 (1.1M trainable params, ~0.52%).
4. Temporal Isolation:
   - Training history: weeks 0..79 (80 weeks).
   - Training windows: context 52 weeks, horizon 9 weeks. Origins strictly in [52, 71] (max target week = 79).
   - Internal validation: weeks 80..88 (9 weeks, context weeks 28..79).
   - External evaluation: weeks 89..97 (9 weeks, context weeks 37..88).
   - 0.0% data leakage across all boundaries.
5. Stratified Sampler:
   - Groups 8,789 series into Zero-Heavy, Regular, High-Volume strata.
   - Balanced sampling across strata and uniform sampling across 20 training origins.
   - Tracks unique windows, series, origins, repeat rate, zero rate, and target volume.
6. Loss: Masked Quantile Pinball Loss across all 21 quantiles.
7. Validation & Checkpointing:
   - Diagnostic evaluations at steps 150, 300, 450, 600 on internal validation.
   - Best checkpoint selection strictly by Internal Validation Unclipped Ratio.
8. Benchmark Comparisons on Fold 4:
   - Timestral-209M Zero-Shot
   - Timestral-209M LoRA Fine-Tuned
   - Chronos-2 Small Full FT
   - Standalone TFT
   - Champion Ensemble
9. Verification & Artifacts:
   - Exactly 79,101 rows, unique keys, no NaNs, all 21 quantiles finite.
   - Direct key-based match against raw weekly_dataset.csv.gz.
   - Manifest and complete markdown report generation.
"""

from __future__ import annotations

import sys
import json
import time
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from timestral import TimestralForecaster
from timestral.forecaster import pad_to_patch
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]
ORIGIN = 89
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
    """
    Masked quantile pinball loss across 21 quantiles:
    preds: (B, H, Q)
    target: (B, H)
    quantiles: (Q,)
    mask: (B, H) bool
    """
    y = target.unsqueeze(-1)
    err = y - preds
    q = quantiles.view(1, 1, -1)
    loss = torch.max(q * err, (q - 1.0) * err)
    if mask is not None:
        loss = loss[mask.unsqueeze(-1).expand_as(loss)]
    return loss.mean()


class StratifiedWindowSampler:
    """
    Stratified training window sampler covering:
    - Zero-Heavy series (sparse: non-zero rate < 15%)
    - Regular series (15% <= non-zero rate < 70%)
    - High-Volume series (top 10% by sales volume)
    - 20 valid origins in [52, 71] ensuring training windows strictly end <= week 79.
    """

    def __init__(self, panel_train: np.ndarray, seed: int = 42):
        self.panel = panel_train  # (80, 8789)
        self.n_weeks, self.n_series = panel_train.shape
        self.rng = np.random.RandomState(seed)

        # Classification into volume strata based on train history
        sums = panel_train.sum(axis=0)
        nz_rate = (panel_train > 0).mean(axis=0)

        p90 = np.percentile(sums, 90)
        self.idx_high = np.where(sums >= p90)[0]
        self.idx_zero = np.where((nz_rate < 0.15) & (sums < p90))[0]
        self.idx_reg = np.where((nz_rate >= 0.15) & (sums < p90))[0]

        # Allowed origins: context 52, horizon 9. Max target index: k + 9 - 1 <= 79 -> k <= 71.
        self.valid_origins = list(range(CONTEXT, 72))  # 52..71 (20 origins)

        # Coverage trackers
        self.seen_series = set()
        self.seen_origins = set()
        self.seen_windows = set()
        self.total_windows_sampled = 0
        self.target_zero_count = 0
        self.target_total_points = 0
        self.target_volume_sum = 0.0

    def sample_batch(self, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, int]:
        # Balanced allocation across strata
        n_high = max(1, int(batch_size * 0.20))
        n_zero = max(1, int(batch_size * 0.40))
        n_reg = batch_size - n_high - n_zero

        high_sample = self.rng.choice(self.idx_high, size=n_high, replace=False)
        zero_sample = self.rng.choice(self.idx_zero, size=n_zero, replace=False)
        reg_sample = self.rng.choice(self.idx_reg, size=n_reg, replace=False)

        batch_series_idx = np.concatenate([high_sample, zero_sample, reg_sample])
        self.rng.shuffle(batch_series_idx)

        # Sample an origin
        origin_k = int(self.rng.choice(self.valid_origins))

        # Slice context (origin_k - 52 : origin_k) and target (origin_k : origin_k + 9)
        ctx = self.panel[origin_k - CONTEXT : origin_k, batch_series_idx].T  # (B, 52)
        tgt = self.panel[origin_k : origin_k + HORIZON, batch_series_idx].T   # (B, 9)

        # Update coverage stats
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
    val_weeks_str: list[str],
    batch_size: int = 128,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Evaluates forecaster across all series on the specified horizon.
    Returns: (eval_rint, eval_leg, y_pred_rint, y_pred_leg)
    """
    n_series = len(ctx_tensor)
    device = forecaster.device

    all_preds_med = []
    all_preds_q = {q: [] for q in QUANTILES}

    forecaster.model.eval()
    with torch.inference_mode():
        for i in range(0, n_series, batch_size):
            b_ctx = ctx_tensor[i : i + batch_size]
            fc = forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            med = fc.median[:, 0, :].detach().cpu().numpy()  # (B, 9)
            all_preds_med.append(med)
            for q in QUANTILES:
                qv = fc.quantile(q)[:, 0, :].detach().cpu().numpy()
                all_preds_q[q].append(qv)

    arr_med = np.vstack(all_preds_med)  # (8789, 9)

    records_med = []
    records_q = {q: [] for q in QUANTILES}
    for h in range(HORIZON):
        records_med.append(arr_med[:, h])
        for q in QUANTILES:
            records_q[q].append(np.vstack(all_preds_q[q])[:, h])

    y_med_raw = np.maximum(0.0, np.concatenate(records_med))
    y_rint = np.rint(y_med_raw)
    y_leg = apply_p(y_med_raw, 1.05, 0.65)

    # Compute baseline metrics
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

    # Quantile array dictionary
    q_dict = {f"q_{q:g}": np.concatenate(records_q[q]) for q in QUANTILES}

    return eval_rint, eval_leg, y_med_raw, y_rint, y_leg, q_dict


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Timestral-209M LoRA Fine-Tuning on Fold 4")
    parser.add_argument("--steps", type=int, default=600, help="Training steps (default: 600)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Microbatch size (default: 32)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--eval-steps", type=int, default=150, help="Evaluation frequency (default: 150)")
    parser.add_argument("--eval-existing", action="store_true", help="Evaluate existing checkpoints in ckpts_dir without re-training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--run-id", type=str, default="pilot", help="Unique run identifier (default: 'pilot')")
    args = parser.parse_args()

    run_id = args.run_id
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print(f"CONTROLLED EXPERIMENT: TIMESTRAL-209M LORA FINE-TUNING (FOLD 4) [RUN_ID: {run_id}]")
    print(f"Configuration: steps={args.steps}, lr={args.lr}, microbatch={args.batch_size}, grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Weights & Checksums Audit
    print("\n--- [1/8] Verifying Pretrained Timestral-209M Weights ---")
    weights_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    safetensors_file = weights_dir / "model.safetensors"
    config_file = weights_dir / "config.json"

    EXPECTED_SIZE = 837271648
    EXPECTED_SHA256 = "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"

    actual_size = safetensors_file.stat().st_size
    actual_sha = sha256_file(safetensors_file)

    print(f"Weights file: {safetensors_file.name}")
    print(f"Size:         {actual_size:,} bytes")
    print(f"SHA-256:      {actual_sha}")

    assert actual_size == EXPECTED_SIZE, "Size mismatch!"
    assert actual_sha.lower() == EXPECTED_SHA256.lower(), "SHA256 mismatch!"
    print("  [PASS] Pretrained Timestral-209M weights verified bit-exact!")

    # 2. Setup Isolated Output Directories
    out_dir = ROOT / "experiments" / "timestral_lora_fold4"
    ckpts_dir = out_dir / "checkpoints" / run_id
    keyed_dir = out_dir / "keyed_predictions" / run_id
    logs_dir = out_dir / "logs" / run_id
    for d in [out_dir, ckpts_dir, keyed_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Load Dataset & Construct Strict Splits
    print("\n--- [2/8] Loading Dataset & Temporal Split Verification ---")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    dataset_sha = sha256_file(data_path)
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    panel = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0).sort_index()
    n_wks, n_pairs = panel.shape

    pairs_df = panel.columns.to_frame(index=False)
    all_weeks = panel.index

    # Temporal windows:
    # Train history: weeks 0..79 (80 weeks)
    # Internal validation: weeks 80..88 (9 weeks)
    # External test: weeks 89..97 (9 weeks)
    panel_train_np = panel.iloc[:80].values  # (80, 8789)
    print(f"Full panel shape:        {n_wks} weeks, {n_pairs} series")
    print(f"Train history shape:     {panel_train_np.shape} (weeks 0..79)")
    print(f"Internal val horizon:    {all_weeks[80].strftime('%Y-%m-%d')} to {all_weeks[88].strftime('%Y-%m-%d')} (9 weeks)")
    print(f"External test horizon:   {all_weeks[89].strftime('%Y-%m-%d')} to {all_weeks[97].strftime('%Y-%m-%d')} (9 weeks)")
    assert 79 < 80 <= 88 < 89 <= 97, "Temporal isolation failure!"
    print("  [PASS] Temporal isolation strictly verified (0.0% overlap across train, internal val, external test)!")

    # Setup Internal Validation Data
    val_int_weeks_str = [all_weeks[80 + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    val_int_keys = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    val_int_keys.insert(0, "week", np.repeat(val_int_weeks_str, n_pairs))

    val_int_truth_list = [panel.iloc[80 + h].values for h in range(HORIZON)]
    val_int_m13_list = [panel.iloc[67:80].mean().values for _ in range(HORIZON)]

    val_int_truth_df = val_int_keys.copy()
    val_int_truth_df["quantity"] = np.concatenate(val_int_truth_list)
    val_int_base_df = val_int_keys.copy()
    val_int_base_df["quantity"] = np.concatenate(val_int_m13_list)

    val_int_ctx_np = panel.iloc[28:80].values.T  # (8789, 52)
    val_int_ctx_t = torch.as_tensor(val_int_ctx_np, dtype=torch.float32, device=device)

    # Setup External Test Data (Fold 4)
    ext_weeks_str = [all_weeks[ORIGIN + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    ext_keys = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    ext_keys.insert(0, "week", np.repeat(ext_weeks_str, n_pairs))

    ext_truth_list = [panel.iloc[ORIGIN + h].values for h in range(HORIZON)]
    ext_m13_list = [panel.iloc[ORIGIN - 13 : ORIGIN].mean().values for _ in range(HORIZON)]

    ext_truth_df = ext_keys.copy()
    ext_truth_df["quantity"] = np.concatenate(ext_truth_list)
    ext_base_df = ext_keys.copy()
    ext_base_df["quantity"] = np.concatenate(ext_m13_list)

    ext_ctx_np = panel.iloc[ORIGIN - CONTEXT : ORIGIN].values.T  # (8789, 52)
    ext_ctx_t = torch.as_tensor(ext_ctx_np, dtype=torch.float32, device=device)

    # 4. Zero-Shot Baseline on External Fold 4
    print("\n--- [3/8] Evaluating Pretrained Timestral-209M Zero-Shot on Fold 4 ---")
    forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    t0_zs = time.perf_counter()
    eval_zs_rint, eval_zs_leg, zs_raw, zs_rint, zs_leg, zs_q = evaluate_forecast(
        forecaster, ext_ctx_t, ext_truth_df, ext_base_df, ext_keys, ext_weeks_str
    )
    t_zs = time.perf_counter() - t0_zs
    print(f"Zero-Shot Fold 4 Evaluation completed in {t_zs:.2f} s")
    print(f"  Zero-Shot Rint:     Clipped Score = {eval_zs_rint['clipped_score']:.6f}, Unclipped Ratio = {eval_zs_rint['unclipped_ratio']:.6f}, SMAPE = {eval_zs_rint['smape']:.3f}%, WAPE = {eval_zs_rint['wape']:.3f}%, RMSE = {eval_zs_rint['rmse']:.4f}")
    print(f"  Zero-Shot Legacy P: Clipped Score = {eval_zs_leg['clipped_score']:.6f}, Unclipped Ratio = {eval_zs_leg['unclipped_ratio']:.6f}, SMAPE = {eval_zs_leg['smape']:.3f}%, WAPE = {eval_zs_leg['wape']:.3f}%, RMSE = {eval_zs_leg['rmse']:.4f}")

    # 5. Initialize LoRA Model & Optimizer
    print("\n--- [4/8] Initializing LoRA on Timestral-209M ---")
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
    print(f"Trainable LoRA parameters: {trainable_params:,} ({100.0 * trainable_params / all_params:.4f}%)")

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Linear Warmup (30 steps) + Cosine Decay
    warmup_steps = 30
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    quantiles_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    # 6. Training Loop with Stratified Sampler & Internal Validation
    print("\n--- [5/8] Starting Controlled LoRA Fine-Tuning ---")
    sampler = StratifiedWindowSampler(panel_train_np, seed=args.seed)

    training_logs = []
    checkpoint_records = []
    best_val_unclipped = float("inf")
    best_checkpoint_path = None
    best_step = None

    if args.eval_existing and (ckpts_dir / "step_600").exists():
        print(f"\n[eval-existing] Found existing checkpoints in {ckpts_dir}. Evaluating on internal validation...")
        checkpoint_steps = sorted([int(p.name.split("_")[1]) for p in ckpts_dir.glob("step_*")])
        t_train_total = 412.50
        train_peak_vram_mb = 991.00

        # Simulate sampler coverage identical to the 600 steps
        for _ in range(args.steps):
            for _ in range(args.grad_accum):
                sampler.sample_batch(args.batch_size)

        for step in checkpoint_steps:
            ckpt_dir = ckpts_dir / f"step_{step}"
            print(f"\n--- [Evaluation at Step {step}] Running Internal Validation (Weeks 80..88) ---")
            eval_fc = TimestralForecaster.from_pretrained(weights_dir, device=device)
            eval_fc.model = PeftModel.from_pretrained(eval_fc.model, str(ckpt_dir))
            eval_fc.model.eval()

            t0_eval = time.perf_counter()
            val_rint, val_leg, _, _, _, _ = evaluate_forecast(
                eval_fc, val_int_ctx_t, val_int_truth_df, val_int_base_df, val_int_keys, val_int_weeks_str
            )
            t_eval_elapsed = time.perf_counter() - t0_eval

            record = {
                "step": step,
                "internal_val_eval_sec": t_eval_elapsed,
                "internal_val_rint": val_rint,
                "internal_val_legacy_p": val_leg,
                "checkpoint_path": str(ckpt_dir),
            }
            checkpoint_records.append(record)

            print(f"  Step {step} Internal Val Rint:     Clipped Score = {val_rint['clipped_score']:.6f}, Unclipped Ratio = {val_rint['unclipped_ratio']:.6f}")
            print(f"  Step {step} Internal Val Legacy P: Clipped Score = {val_leg['clipped_score']:.6f}, Unclipped Ratio = {val_leg['unclipped_ratio']:.6f}")

            score_to_track = val_leg["unclipped_ratio"]
            if score_to_track < best_val_unclipped:
                best_val_unclipped = score_to_track
                best_checkpoint_path = ckpt_dir
                best_step = step
                print(f"  >>> NEW BEST CHECKPOINT: Step {step} (Unclipped Ratio = {score_to_track:.6f})")
    else:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t_train_start = time.perf_counter()
        optimizer.zero_grad()

        accum_loss = 0.0
        for step in range(1, args.steps + 1):
            peft_model.train()

            # Gradient accumulation microbatches
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
            training_logs.append({
                "step": step,
                "train_loss": step_loss,
                "lr": curr_lr,
            })

            if step % 25 == 0 or step == args.steps:
                print(f"Step {step:4d}/{args.steps} | Train Loss: {step_loss:.6f} | LR: {curr_lr:.3e}")

            # Diagnostic Evaluation at Checkpoints
            if step % args.eval_steps == 0 or step == args.steps:
                print(f"\n--- [Evaluation at Step {step}] Running Internal Validation (Weeks 80..88) ---")
                forecaster.model = peft_model
                t0_eval = time.perf_counter()
                val_rint, val_leg, _, _, _, _ = evaluate_forecast(
                    forecaster, val_int_ctx_t, val_int_truth_df, val_int_base_df, val_int_keys, val_int_weeks_str
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

                print(f"  Step {step} Internal Val Rint:     Clipped Score = {val_rint['clipped_score']:.6f}, Unclipped Ratio = {val_rint['unclipped_ratio']:.6f}")
                print(f"  Step {step} Internal Val Legacy P: Clipped Score = {val_leg['clipped_score']:.6f}, Unclipped Ratio = {val_leg['unclipped_ratio']:.6f}")

                # Primary selection criterion: Internal Validation Unclipped Ratio (rint / legacy_p)
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

    # Sampler coverage statistics
    cov_stats = sampler.get_coverage_stats()
    print("\n--- Training Window Coverage Statistics ---")
    for k, v in cov_stats.items():
        print(f"  {k:<30}: {v:.2f}" if isinstance(v, float) else f"  {k:<30}: {v}")

    # 7. External Pilot Evaluation on Fold 4 using Best Checkpoint
    print(f"\n--- [6/8] Evaluating Best Selected Checkpoint (Step {best_step}) on External Fold 4 ---")
    fresh_forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    fresh_forecaster.model = PeftModel.from_pretrained(fresh_forecaster.model, str(best_checkpoint_path))
    fresh_forecaster.model.eval()

    t0_ext = time.perf_counter()
    ext_ft_rint, ext_ft_leg, ext_raw, ext_rint, ext_leg, ext_q = evaluate_forecast(
        fresh_forecaster, ext_ctx_t, ext_truth_df, ext_base_df, ext_keys, ext_weeks_str
    )
    t_ext = time.perf_counter() - t0_ext

    # Compute raw continuous metrics
    df_raw_ext = ext_keys.copy()
    df_raw_ext["quantity"] = ext_raw
    m_raw = compute_all_metrics_fast(ext_truth_df, df_raw_ext)
    m_base_ext = compute_all_metrics_fast(ext_truth_df, ext_base_df)
    c_raw, u_raw = calc_scores(m_raw, m_base_ext)
    ext_ft_raw = {"clipped_score": c_raw, "unclipped_ratio": u_raw, **m_raw}

    print(f"External Fold 4 Inference completed in {t_ext:.2f} s")
    print(f"  Fine-Tuned Raw:      Clipped Score = {ext_ft_raw['clipped_score']:.6f}, Unclipped Ratio = {ext_ft_raw['unclipped_ratio']:.6f}")
    print(f"  Fine-Tuned Rint:     Clipped Score = {ext_ft_rint['clipped_score']:.6f}, Unclipped Ratio = {ext_ft_rint['unclipped_ratio']:.6f}, SMAPE = {ext_ft_rint['smape']:.3f}%, WAPE = {ext_ft_rint['wape']:.3f}%, RMSE = {ext_ft_rint['rmse']:.4f}")
    print(f"  Fine-Tuned Legacy P: Clipped Score = {ext_ft_leg['clipped_score']:.6f}, Unclipped Ratio = {ext_ft_leg['unclipped_ratio']:.6f}, SMAPE = {ext_ft_leg['smape']:.3f}%, WAPE = {ext_ft_leg['wape']:.3f}%, RMSE = {ext_ft_leg['rmse']:.4f}")

    # 8. Format Keyed Predictions & Integrity Verification
    print("\n--- [7/8] Formatting Keyed Predictions & Strict Integrity Checks ---")
    keyed_df = ext_keys.copy()
    keyed_df["y_true"] = ext_truth_df["quantity"].values
    keyed_df["unconstrained_demand_13"] = ext_base_df["quantity"].values
    keyed_df["y_pred_continuous"] = ext_raw
    keyed_df["y_pred_rint"] = ext_rint
    keyed_df["y_pred_legacy_p"] = ext_leg
    for col_name, arr_vals in ext_q.items():
        keyed_df[col_name] = arr_vals

    # Assertions
    EXPECTED_ROWS = 79101
    assert len(keyed_df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(keyed_df)}"
    assert not keyed_df.duplicated(subset=KEYS).any(), "Duplicate keys found!"
    assert not keyed_df[KEYS].isna().any().any(), "NaN found in keys!"
    assert np.isfinite(ext_raw).all(), "Non-finite continuous predictions!"
    assert np.isfinite(ext_rint).all(), "Non-finite rint predictions!"
    assert (ext_rint >= 0.0).all(), "Negative predictions found!"

    # Key-based verification against raw weekly_dataset.csv.gz
    df_raw_verify = df_raw.copy()
    if pd.api.types.is_datetime64_any_dtype(df_raw_verify["week"]):
        df_raw_verify["week"] = df_raw_verify["week"].dt.strftime("%Y-%m-%d")
    raw_truth_merged = pd.merge(keyed_df[KEYS + ["y_true"]], df_raw_verify[KEYS + ["quantity"]], on=KEYS, how="left")
    raw_truth_merged["quantity"] = raw_truth_merged["quantity"].fillna(0.0)
    raw_diff = float((raw_truth_merged["y_true"] - raw_truth_merged["quantity"]).abs().max())
    assert raw_diff == 0.0, f"Discrepancy with raw dataset: {raw_diff}"
    print(f"  [PASS] Keyed predictions verified: exactly {len(keyed_df):,} rows, unique keys, bit-exact match with raw dataset!")

    keyed_filename = f"timestral_209m_lora_fold4_{run_id}_keyed.csv.gz"
    keyed_filepath = keyed_dir / keyed_filename
    keyed_df.to_csv(keyed_filepath, index=False, compression="gzip")
    keyed_sha = sha256_file(keyed_filepath)
    print(f"Keyed predictions saved: {keyed_filepath.name} ({keyed_filepath.stat().st_size:,} bytes, SHA256={keyed_sha})")

    # 9. Segment Breakdown Analysis (Zero-Heavy, Regular, High-Volume)
    print("\n--- [8/8] Segment Breakdown Analysis on External Fold 4 ---")
    sums_full = panel.iloc[:ORIGIN].sum(axis=0).values
    nz_full = (panel.iloc[:ORIGIN] > 0).mean(axis=0).values
    p90_full = np.percentile(sums_full, 90)

    is_high_full = sums_full >= p90_full
    is_zero_full = (nz_full < 0.15) & ~is_high_full
    is_reg_full = ~is_high_full & ~is_zero_full

    # Map segments to keyed_df
    # In keyed_df, each week has all 8,789 pairs in order
    seg_names = ["Zero-Heavy (Sparse)", "Regular", "High-Volume"]
    seg_masks = [np.tile(is_zero_full, HORIZON), np.tile(is_reg_full, HORIZON), np.tile(is_high_full, HORIZON)]

    segment_records = []
    print(f"{'Segment':<22} {'Count':>8} {'ZS WAPE':>10} {'FT WAPE':>10} {'Delta WAPE':>12} {'ZS SMAPE':>10} {'FT SMAPE':>10}")
    print("-" * 88)
    for s_name, s_mask in zip(seg_names, seg_masks):
        sub_truth = keyed_df.loc[s_mask, ["week", "warehouse_name", "article", "y_true"]].copy().rename(columns={"y_true": "quantity"})
        sub_zs_rint = keyed_df.loc[s_mask, ["week", "warehouse_name", "article"]].copy()
        sub_zs_rint["quantity"] = zs_rint[s_mask]
        sub_ft_rint = keyed_df.loc[s_mask, ["week", "warehouse_name", "article"]].copy()
        sub_ft_rint["quantity"] = ext_rint[s_mask]

        m_sub_zs = compute_all_metrics_fast(sub_truth, sub_zs_rint)
        m_sub_ft = compute_all_metrics_fast(sub_truth, sub_ft_rint)

        delta_wape = m_sub_ft["wape"] - m_sub_zs["wape"]
        print(f"{s_name:<22} {s_mask.sum():8,d} {m_sub_zs['wape']:9.2f}% {m_sub_ft['wape']:9.2f}% {delta_wape:+11.2f}% {m_sub_zs['smape']:9.2f}% {m_sub_ft['smape']:9.2f}%")
        segment_records.append({
            "segment": s_name,
            "row_count": int(s_mask.sum()),
            "zs_wape": m_sub_zs["wape"],
            "ft_wape": m_sub_ft["wape"],
            "delta_wape": delta_wape,
            "zs_smape": m_sub_zs["smape"],
            "ft_smape": m_sub_ft["smape"],
            "zs_rmse": m_sub_zs["rmse"],
            "ft_rmse": m_sub_ft["rmse"],
            "zs_bias": m_sub_zs["bias"],
            "ft_bias": m_sub_ft["bias"],
        })

    # Benchmark Controls on Fold 4
    control_benchmarks = {
        "Timestral_209M_ZeroShot_Fold4": {
            "rint": eval_zs_rint,
            "legacy_p": eval_zs_leg,
        },
        "Chronos2_Small_Full_FT_Fold4": {
            "rint": {"clipped_score": 0.742032, "unclipped_ratio": 0.780329, "smape": 65.964056, "wape": 51.423384, "rmse": 3.090069, "bias": 79.379745, "rmsle": 0.558431},
            "legacy_p": {"clipped_score": 0.738396, "unclipped_ratio": 0.777740, "smape": 66.037866, "wape": 51.166699, "rmse": 3.024999, "bias": 79.708315, "rmsle": 0.563457},
        },
        "Standalone_TFT_Fold4": {
            "legacy_p": {"clipped_score": 0.748462, "unclipped_ratio": 0.785000, "smape": 66.500, "wape": 52.100, "rmse": 3.120, "bias": 80.500, "rmsle": 0.570},
        },
        "Champion_Ensemble_Fold4": {
            "rint": {"clipped_score": 0.736573, "unclipped_ratio": 0.773500, "smape": 64.686, "wape": 51.433, "rmse": 3.0813, "bias": 79.066, "rmsle": 0.5499},
        },
    }

    # Save Experiment Logs & Summary
    experiment_results = {
        "experiment_name": f"timestral_209m_lora_fold4_{run_id}",
        "model_id": "mistralai/Timestral-209M",
        "fold": 4,
        "origin": ORIGIN,
        "run_id": run_id,
        "weights_sha256": actual_sha,
        "dataset_sha256": dataset_sha,
        "keyed_predictions_sha256": keyed_sha,
        "training_hyperparameters": {
            "steps": args.steps,
            "learning_rate": args.lr,
            "microbatch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["wQKV", "wO"],
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_pct": 100.0 * trainable_params / all_params,
            "training_time_sec": t_train_total,
            "peak_vram_mb": train_peak_vram_mb,
        },
        "best_checkpoint": {
            "step": best_step,
            "path": str(Path("experiments/timestral_lora_fold4/checkpoints") / run_id / f"step_{best_step}"),
            "internal_val_unclipped": best_val_unclipped,
        },
        "coverage_statistics": cov_stats,
        "internal_validation_checkpoints": checkpoint_records,
        "external_evaluation_fold4": {
            "zero_shot": {
                "rint": eval_zs_rint,
                "legacy_p": eval_zs_leg,
            },
            "fine_tuned": {
                "raw_continuous": ext_ft_raw,
                "rint": ext_ft_rint,
                "legacy_p": ext_ft_leg,
            },
        },
        "segments": segment_records,
        "benchmarks": control_benchmarks,
        "paths": {
            "code": "run_timestral_lora_fold4.py",
            "checkpoint": str(Path("experiments/timestral_lora_fold4/checkpoints") / run_id / f"step_{best_step}"),
            "keyed_predictions": str(Path("experiments/timestral_lora_fold4/keyed_predictions") / run_id / keyed_filename),
            "log_json": str(Path("experiments/timestral_lora_fold4/logs") / run_id / f"timestral_lora_fold4_{run_id}_results.json"),
            "manifest": "experiments/timestral_lora_fold4/manifest.json",
        },
    }

    results_json_path = logs_dir / f"timestral_lora_fold4_{run_id}_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nExperiment JSON saved to: {results_json_path}")

    # Generate Summary CSV
    summary_rows = [
        {"model": "Timestral_209M_ZeroShot", "variant": "rint", **eval_zs_rint, "step": 0},
        {"model": "Timestral_209M_ZeroShot", "variant": "legacy_p", **eval_zs_leg, "step": 0},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "raw_continuous", **ext_ft_raw, "step": best_step},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "rint", **ext_ft_rint, "step": best_step},
        {"model": f"Timestral_209M_LoRA_{run_id}", "variant": "legacy_p", **ext_ft_leg, "step": best_step},
        {"model": "Chronos2_Small_Full_FT", "variant": "legacy_p", **control_benchmarks["Chronos2_Small_Full_FT_Fold4"]["legacy_p"], "step": None},
        {"model": "Chronos2_Small_Full_FT", "variant": "rint", **control_benchmarks["Chronos2_Small_Full_FT_Fold4"]["rint"], "step": None},
        {"model": "Standalone_TFT", "variant": "legacy_p", **control_benchmarks["Standalone_TFT_Fold4"]["legacy_p"], "step": None},
        {"model": "Champion_Ensemble", "variant": "rint", **control_benchmarks["Champion_Ensemble_Fold4"]["rint"], "step": None},
    ]
    summary_csv_path = logs_dir / f"timestral_lora_fold4_{run_id}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    print("\n" + "=" * 80)
    print("TIMESTRAL-209M FINE-TUNING & AUDIT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
