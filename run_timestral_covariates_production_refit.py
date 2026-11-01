#!/usr/bin/env python3
"""
run_timestral_covariates_production_refit.py

Full-History Production Refit of Timestral-209M LoRA with Selected Native Covariates:
Groups: ['D_category', 'C_network'] (6 past covariate channels)

Guarantees:
1. Pure native covariate conditioning via Timestral transformer architecture.
2. Full history panel (Weeks 0..133), valid training origins k in [52, 125].
3. Zero future leakage: maximum training target index <= 133.
4. Exact CV hyperparameter parity: LoRA r=8, alpha=16, modules=['wQKV', 'wO'], dropout=0.05,
   AdamW(lr=1e-4, weight_decay=1e-4), warmup 20 steps + cosine decay.
5. Control submission parity: delivery_final/submission.csv verified SHA256.
6. Mandatory Zero-Weight Test: blend(w=0) produces 100% bit-exact identity to delivery_final/submission.csv (diff = 0).
7. Outputs:
   - Final LoRA checkpoint: experiments/timestral_production_model/refit_covariates/checkpoints/final_refit/
   - Future 9-week continuous prediction: experiments/timestral_production_model/refit_covariates/timestral_covariates_future_9w_keyed.csv.gz
   - Candidate submission: submissions/submission_timestral_selected_features.csv (79,101 rows)
   - Manifest: experiments/timestral_production_model/refit_covariates/production_refit_summary.json
"""

from __future__ import annotations

import sys
import os
import time
import math
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from timestral import TimestralForecaster
from experiments.timestral_feature_research.timestral_feature_engine import (
    TimestralFeatureEngine, CONTEXT, HORIZON
)

KEYS = ["week", "warehouse_name", "article"]
SELECTED_GROUPS = ["D_category", "C_network"]
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"
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


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def pinball_loss_21(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    y = target.unsqueeze(-1)
    err = y - preds
    q = quantiles.view(1, 1, -1)
    loss = torch.max(q * err, (q - 1.0) * err)
    return loss.mean()


class StratifiedSeriesSampler:
    def __init__(self, quantity_matrix: np.ndarray, origin: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        sums = quantity_matrix[origin - 13 : origin, :].sum(axis=0)
        nz = (quantity_matrix[origin - 13 : origin, :] > 0).mean(axis=0)
        p90 = np.percentile(sums, 90)

        self.idx_high = np.where(sums >= p90)[0]
        self.idx_zero = np.where((nz < 0.15) & (sums < p90))[0]
        self.idx_reg = np.where((nz >= 0.15) & (sums < p90))[0]

    def sample_indices(self, batch_size: int = 32) -> np.ndarray:
        n_high = max(1, int(batch_size * 0.25))
        n_zero = max(1, int(batch_size * 0.35))
        n_reg = batch_size - n_high - n_zero

        high_s = self.rng.choice(self.idx_high, size=n_high, replace=False)
        zero_s = self.rng.choice(self.idx_zero, size=n_zero, replace=False)
        reg_s = self.rng.choice(self.idx_reg, size=n_reg, replace=False)
        return np.concatenate([high_s, zero_s, reg_s])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    print("=" * 105)
    print("TIMESTRAL-209M LoRA PRODUCTION REFIT WITH SELECTED NATIVE COVARIATES")
    print(f"Selected Groups: {SELECTED_GROUPS}")
    print("=" * 105)

    # 1. Verify Control Submission
    print("\n1. Verifying Control Champion Submission...")
    ctrl_path = ROOT / "delivery_final" / "submission.csv"
    assert ctrl_path.exists(), f"Control submission missing at {ctrl_path}"
    ctrl_sha = sha256_file(ctrl_path)
    print(f"   Control SHA256: {ctrl_sha}")
    assert ctrl_sha in (CONTROL_SHA256_CRLF, CONTROL_SHA256_LF), f"Control SHA mismatch: {ctrl_sha}"

    df_ctrl = pd.read_csv(ctrl_path)
    assert len(df_ctrl) == 79101, f"Expected 79,101 rows, got {len(df_ctrl)}"
    assert not df_ctrl.duplicated(subset=KEYS).any(), "Duplicate keys in control!"
    print("   [PASS] Control champion submission verified 100%.")

    # 2. Recover Champion Raw Components
    print("\n2. Loading Champion Raw Components...")
    raw_parquet = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.parquet"
    raw_csv = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    if raw_parquet.exists():
        df_champ_raw = pd.read_parquet(raw_parquet)
    else:
        df_champ_raw = pd.read_csv(raw_csv)
    assert len(df_champ_raw) == 79101
    raw_champion = df_champ_raw["raw_champion"].to_numpy(dtype=np.float64)

    # 3. Mandatory Zero-Weight Test
    print("\n3. Running Mandatory Zero-Weight Test (w = 0)...")
    q_w0 = apply_p(raw_champion, 1.05, 0.65)
    merged_ctrl = pd.merge(
        df_ctrl[KEYS + ["quantity"]],
        df_champ_raw[KEYS].assign(q_w0=q_w0),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    diff_w0 = np.abs(merged_ctrl["quantity"].values - merged_ctrl["q_w0"].values)
    max_delta = int(diff_w0.max())
    diff_count = int((diff_w0 != 0).sum())
    print(f"   Zero-weight test: differing rows = {diff_count}, max delta = {max_delta}")
    assert max_delta == 0, f"MANDATORY TEST FAILURE: w=0 differs from control by {max_delta}!"
    print("   [PASS] Zero-weight test PASSED: bit-exact identity to delivery_final/submission.csv (delta = 0).")

    # 4. Prepare Feature Engine & Sampling
    print("\n4. Loading Feature Engine on Full History (Weeks 0..133)...")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    engine = TimestralFeatureEngine(data_path)
    n_series = engine.n_series
    production_origin = engine.n_weeks  # origin = 134
    print(f"   Total panel length: {engine.n_weeks} weeks, series count: {n_series:,}")
    print(f"   Production forecast origin: {production_origin} (weeks {production_origin}..{production_origin+HORIZON-1})")

    valid_train_origins = list(range(CONTEXT, production_origin - HORIZON + 1))
    print(f"   Valid train origins: {len(valid_train_origins)} (origins {valid_train_origins[0]}..{valid_train_origins[-1]})")
    print(f"   Latest train target: week {valid_train_origins[-1] + HORIZON - 1} <= {engine.n_weeks - 1} (Zero leakage)")

    # 5. Initialize Model from Original Untouched Pretrained Weights
    print("\n5. Initializing Timestral-209M from Pretrained Weights...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(str(model_dir), device=device)
    model = forecaster.model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["wQKV", "wO"],
        lora_dropout=0.05,
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.train()

    total_steps = 300
    microbatch = 4
    grad_accum = 8
    lr = 1e-4
    warmup_steps = 20

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-5)

    sampler = StratifiedSeriesSampler(engine.matrices.quantity, production_origin, seed=42)
    quantile_knots = torch.tensor(forecaster.quantile_levels, device=device, dtype=torch.float32)

    ps = forecaster.patch_size
    horizon_patches = math.ceil(HORIZON / ps)

    print(f"\n6. Training Full-History LoRA ({total_steps} steps, lr={lr}, weight_decay=1e-4)...")
    train_start = time.time()
    step = 0
    accum_count = 0
    running_loss = 0.0
    history_losses = []

    optimizer.zero_grad()

    while step < total_steps:
        tr_origin = int(np.random.choice(valid_train_origins))
        batch_indices = sampler.sample_indices(batch_size=microbatch)

        tgt_ctx, y_true_arr, past_arr, known_arr, _, _ = engine.assemble_batch_tensors(
            groups=SELECTED_GROUPS,
            origin=tr_origin,
            indices=batch_indices,
            context_len=CONTEXT,
            horizon_len=HORIZON,
        )

        tgt_t = torch.as_tensor(tgt_ctx, device=device, dtype=torch.float32)
        obs_mask = torch.ones_like(tgt_t, dtype=torch.bool)
        y_true_t = torch.as_tensor(y_true_arr, device=device, dtype=torch.float32)
        past_t = torch.as_tensor(past_arr, device=device, dtype=torch.float32) if past_arr is not None else None
        known_t = torch.as_tensor(known_arr, device=device, dtype=torch.float32) if known_arr is not None else None

        rows, mask, role_ids, kf, kfm, n_tgt = forecaster._assemble_rows(
            inputs=tgt_t,
            observation_mask=obs_mask,
            prediction_length=HORIZON,
            past_covariates=past_t,
            known_covariates=known_t,
        )

        b, v, t_pad = rows.shape
        c_flat = rows.reshape(b * v, t_pad)
        m_flat = mask.reshape(b * v, t_pad)
        g_ids = torch.arange(b, device=device).repeat_interleave(v)

        quantiles_raw = lora_model.predict_block(
            inputs=c_flat,
            observed=m_flat,
            group_ids=g_ids,
            horizon_patches=horizon_patches,
            role_ids=role_ids,
            known_future=kf,
            known_future_mask=kfm,
        )

        quantiles = quantiles_raw.reshape(b, v, horizon_patches * ps, -1)
        target_preds = quantiles[:, :n_tgt, :HORIZON, :].squeeze(1)

        loss = pinball_loss_21(target_preds, y_true_t, quantile_knots)
        loss_scaled = loss / grad_accum
        loss_scaled.backward()

        running_loss += loss.item()
        accum_count += 1

        if accum_count == grad_accum:
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if step >= warmup_steps:
                scheduler.step()

            step += 1
            accum_count = 0
            avg_step_loss = running_loss / grad_accum
            history_losses.append(avg_step_loss)
            running_loss = 0.0

            if step % 50 == 0 or step == total_steps:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"   Step {step:03d}/{total_steps} | Loss: {avg_step_loss:.4f} | LR: {cur_lr:.2e}", flush=True)

    train_duration = time.time() - train_start
    print(f"   Training completed in {train_duration:.1f}s.")

    # 7. Save LoRA Checkpoint
    refit_dir = ROOT / "experiments" / "timestral_production_model" / "refit_covariates"
    ckpt_dir = refit_dir / "checkpoints" / "final_refit"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(ckpt_dir))
    print(f"\n7. Saved LoRA adapter to {ckpt_dir}")

    # 8. Full Production Inference at Origin 134
    print(f"\n8. Running Production Inference at Origin {production_origin} on all {n_series:,} series...")
    forecaster.model.eval()
    eval_batch_size = 64
    all_median_preds = []
    eval_start = time.time()
    all_indices = np.arange(n_series)

    with torch.inference_mode():
        for b_start in range(0, n_series, eval_batch_size):
            b_idx = all_indices[b_start : b_start + eval_batch_size]

            tgt_ctx, _, past_arr, known_arr, _, _ = engine.assemble_batch_tensors(
                groups=SELECTED_GROUPS,
                origin=production_origin,
                indices=b_idx,
                context_len=CONTEXT,
                horizon_len=HORIZON,
            )

            tgt_t = torch.as_tensor(tgt_ctx, device=device, dtype=torch.float32)
            past_t = torch.as_tensor(past_arr, device=device, dtype=torch.float32) if past_arr is not None else None
            known_t = torch.as_tensor(known_arr, device=device, dtype=torch.float32) if known_arr is not None else None

            fc = forecaster.forecast(
                context=tgt_t,
                prediction_length=HORIZON,
                past_covariates=past_t,
                known_covariates=known_t,
                symmetric_averaging=True,
                clip_nonnegative=True,
            )
            med = fc.median.squeeze(1).cpu().numpy()
            all_median_preds.append(med)

    eval_duration = time.time() - eval_start
    print(f"   Inference completed in {eval_duration:.1f}s.")

    pred_matrix = np.concatenate(all_median_preds, axis=0)  # (8789, 9)
    timestral_raw = pred_matrix.T.reshape(-1)               # 79,101

    # Format into DataFrame with competition keys
    future_weeks_series = pd.date_range("2026-07-27", periods=9, freq="W-MON")
    rows_list = []
    for h_idx, w_dt in enumerate(future_weeks_series):
        w_str = w_dt.strftime("%Y-%m-%d")
        for s_idx in range(n_series):
            rows_list.append({
                "week": w_str,
                "warehouse_name": engine.pairs.iloc[s_idx]["warehouse_name"],
                "article": engine.pairs.iloc[s_idx]["article"],
                "raw_timestral_selected": float(pred_matrix[s_idx, h_idx]),
            })
    df_timestral_future = pd.DataFrame(rows_list)
    assert len(df_timestral_future) == 79101

    # Save raw future predictions
    future_raw_path = refit_dir / "timestral_covariates_future_9w_keyed.csv.gz"
    df_timestral_future.to_csv(future_raw_path, index=False, compression="gzip")
    print(f"   Saved raw future predictions to {future_raw_path.name}")

    # 9. Generate Candidates and Evaluate Blends
    print("\n9. Generating Production Submissions...")
    merged_prod = pd.merge(
        df_champ_raw[KEYS + ["raw_champion"]],
        df_timestral_future[KEYS + ["raw_timestral_selected"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    raw_champ_arr = merged_prod["raw_champion"].values.astype(float)
    raw_time_arr = merged_prod["raw_timestral_selected"].values.astype(float)

    # 1. Standalone candidate: apply_p(raw_timestral)
    sub_standalone = merged_prod[KEYS].copy()
    sub_standalone["quantity"] = apply_p(raw_time_arr, 1.05, 0.65)
    sub_standalone_path = ROOT / "submissions" / "submission_timestral_selected_features_standalone.csv"
    sub_standalone.to_csv(sub_standalone_path, index=False)
    print(f"   Saved Standalone submission: {sub_standalone_path.name} (sum={sub_standalone['quantity'].sum():,})")

    # 2. Blend candidate: 10% Selected Timestral + 90% Champion
    raw_blend_10 = 0.90 * raw_champ_arr + 0.10 * raw_time_arr
    sub_blend10 = merged_prod[KEYS].copy()
    sub_blend10["quantity"] = apply_p(raw_blend_10, 1.05, 0.65)

    # Save as main submission candidate
    sub_main_path = ROOT / "submissions" / "submission_timestral_selected_features.csv"
    sub_blend10.to_csv(sub_main_path, index=False)
    print(f"   Saved Candidate submission: {sub_main_path.name} (sum={sub_blend10['quantity'].sum():,})")

    # Calculate differences vs Champion
    champ_q = df_ctrl["quantity"].values
    diff_blend = np.abs(sub_blend10["quantity"].values - champ_q)
    changed_rows = int((diff_blend != 0).sum())
    max_diff = int(diff_blend.max())
    print(f"   Candidate vs Champion: {changed_rows:,} rows changed ({changed_rows/79101*100:.2f}%), max delta = {max_diff}")

    # 10. Summary manifest
    summary = {
        "model": "mistralai/Timestral-209M",
        "features": SELECTED_GROUPS,
        "n_features": 6,
        "training": {
            "history_weeks": engine.n_weeks,
            "steps": total_steps,
            "microbatch": microbatch,
            "grad_accum": grad_accum,
            "lr": lr,
            "weight_decay": 1e-4,
            "train_duration_sec": train_duration,
            "final_loss": history_losses[-1],
        },
        "inference": {
            "origin": production_origin,
            "eval_duration_sec": eval_duration,
        },
        "zero_weight_test": {
            "passed": True,
            "differing_rows": diff_count,
            "max_delta": max_delta,
        },
        "candidate_submission": {
            "path": str(sub_main_path),
            "rows": len(sub_blend10),
            "sum": int(sub_blend10["quantity"].sum()),
            "changed_rows_vs_champion": changed_rows,
            "max_delta_vs_champion": max_diff,
            "sha256": sha256_file(sub_main_path),
        },
    }

    manifest_path = refit_dir / "production_refit_summary.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n10. Saved summary manifest to {manifest_path.name}")
    print("=" * 105)
    print("PRODUCTION REFIT AND CANDIDATE GENERATION COMPLETE!")
    print("=" * 105)


if __name__ == "__main__":
    main()
