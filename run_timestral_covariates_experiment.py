#!/usr/bin/env python3
"""
run_timestral_covariates_experiment.py

Modular training and evaluation pipeline for Timestral-209M with Native Covariates.
Supports:
- Group selection: any combination of groups ['A_dynamics', 'B_stock', 'C_network', 'D_category', 'E_money', 'F_calendar', 'G_year', 'H_categories']
- Exact reproducibility: fixed seed, stratified window sampling, AdamW + cosine schedule
- Exact competition scoring: compute_all_metrics_fast against fold baseline
- Standalone evaluation: evaluates Timestral alone (both continuous rint and postprocessed)
- Ensemble evaluation: evaluates blend with Champion raw
"""

from __future__ import annotations

import sys
import os
import time
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))
sys.path.insert(0, str(ROOT / "experiments" / "urgent_models" / "timestral_repo" / "src"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from timestral.forecaster import TimestralForecaster
from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine, CONTEXT, HORIZON

KEYS = ["week", "warehouse_name", "article"]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
ALL_GROUPS = [
    "A_dynamics",
    "B_stock",
    "C_network",
    "D_category",
    "E_money",
    "F_calendar",
    "G_year",
    "H_categories",
]


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
    return np.rint(gated).clip(0, None).astype(int)


def pinball_loss_21(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """preds: (B, H, Q), target: (B, H), quantiles: (Q,)"""
    y = target.unsqueeze(-1)
    err = y - preds
    q = quantiles.view(1, 1, -1)
    loss = torch.max(q * err, (q - 1.0) * err)
    return loss.mean()


class StratifiedSeriesSampler:
    """Samples series stratified by historical volume and non-zero activity."""
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


def train_and_evaluate_timestral_run(
    engine: TimestralFeatureEngine,
    groups: List[str],
    fold: int = 4,
    steps: int = 200,
    microbatch: int = 4,
    grad_accum: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Runs a complete training and evaluation experiment on a given fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    origin = FOLD_ORIGINS[fold]
    print(f"\n{'='*95}")
    print(f"RUNNING TIMESTRAL EXPERIMENT: Fold {fold} (Origin {origin}) | Groups: {groups or ['None (Univariate)']}")
    print(f"Hyperparameters: steps={steps}, microbatch={microbatch}, grad_accum={grad_accum}, lr={lr}, seed={seed}")
    print(f"{'='*95}")

    # 1. Load base model & forecaster
    model_path = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(model_path, device=device)
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

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_steps = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps - warmup_steps, 1), eta_min=1e-5)

    sampler = StratifiedSeriesSampler(engine.matrices.quantity, origin, seed=seed)
    quantile_knots = torch.tensor(forecaster.quantile_levels, device=device, dtype=torch.float32)

    # Training origins: sliding historical origins before forecast origin
    # e.g. for fold origin 89, valid origins are range(52, 89 - 9 + 1) -> range(52, 81)
    valid_train_origins = list(range(CONTEXT, origin - HORIZON + 1))
    assert len(valid_train_origins) > 0, f"No train origins for origin {origin}!"

    ps = forecaster.patch_size
    horizon_patches = math.ceil(HORIZON / ps)

    train_start_time = time.time()
    step = 0
    accum_count = 0
    running_loss = 0.0
    history_losses = []

    print("Beginning training...")
    optimizer.zero_grad()

    while step < steps:
        # Sample training origin and series indices
        tr_origin = int(np.random.choice(valid_train_origins))
        batch_indices = sampler.sample_indices(batch_size=microbatch)

        # Assemble batch via feature engine
        tgt_ctx, y_true_arr, past_arr, known_arr, p_names, k_names = engine.assemble_batch_tensors(
            groups=groups,
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

        # Assemble rows using internal forecaster logic
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

        # Reshape and extract target channel only
        quantiles = quantiles_raw.reshape(b, v, horizon_patches * ps, -1)
        target_preds = quantiles[:, :n_tgt, :HORIZON, :].squeeze(1)  # (B, 9, 21)

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

            if step % 25 == 0 or step == steps:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  Step {step:03d}/{steps} | Loss: {avg_step_loss:.4f} | LR: {cur_lr:.2e}", flush=True)

    train_duration = time.time() - train_start_time
    print(f"Training completed in {train_duration:.1f}s.", flush=True)

    # -------------------------------------------------------------------------
    # Evaluation on all 8,789 series of Fold
    # -------------------------------------------------------------------------
    print(f"\nEvaluating on all {engine.n_series:,} series of Fold {fold}...", flush=True)
    forecaster.model.eval()
    all_median_preds = []
    eval_batch_size = 64

    eval_start_time = time.time()
    all_indices = np.arange(engine.n_series)

    with torch.inference_mode():
        for b_start in range(0, engine.n_series, eval_batch_size):
            b_idx = all_indices[b_start : b_start + eval_batch_size]

            tgt_ctx, _, past_arr, known_arr, _, _ = engine.assemble_batch_tensors(
                groups=groups,
                origin=origin,
                indices=b_idx,
                context_len=CONTEXT,
                horizon_len=HORIZON,
            )

            tgt_t = torch.as_tensor(tgt_ctx, device=device, dtype=torch.float32)
            obs_mask = torch.ones_like(tgt_t, dtype=torch.bool)
            past_t = torch.as_tensor(past_arr, device=device, dtype=torch.float32) if past_arr is not None else None
            known_t = torch.as_tensor(known_arr, device=device, dtype=torch.float32) if known_arr is not None else None

            # Standard forecaster.forecast() in eval mode
            fc = forecaster.forecast(
                context=tgt_t,
                prediction_length=HORIZON,
                past_covariates=past_t,
                known_covariates=known_t,
                symmetric_averaging=True,
                clip_nonnegative=True,
            )
            # Median forecast is quantile 0.5: shape (B, 1, 9)
            med = fc.median.squeeze(1).cpu().numpy()  # (B, 9)
            all_median_preds.append(med)

    eval_duration = time.time() - eval_start_time
    print(f"Evaluation completed in {eval_duration:.1f}s.", flush=True)

    # Matrix of shape (N_series, 9) -> transpose to (9, N_series) -> flatten to 79,101 rows
    pred_matrix = np.concatenate(all_median_preds, axis=0)  # (8789, 9)
    pred_continuous = pred_matrix.T.reshape(-1)             # 9 * 8789 = 79101

    # Load ground truth from champion OOF
    champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
    df_champ = pd.read_csv(champ_path)
    assert len(df_champ) == 79101

    truth = df_champ[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
    base = df_champ[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})
    m_base = compute_all_metrics_fast(truth, base)

    # 1. Standalone RINT
    pred_rint = np.rint(np.clip(pred_continuous, 0, None)).astype(int)
    df_p_rint = truth[KEYS].copy()
    df_p_rint["quantity"] = pred_rint
    m_rint = compute_all_metrics_fast(truth, df_p_rint)
    c_rint, u_rint = calc_scores(m_rint, m_base)

    # 2. Standalone Legacy Postprocessing (apply_p: 1.05, 0.65)
    pred_post = apply_p(pred_continuous, 1.05, 0.65)
    df_p_post = truth[KEYS].copy()
    df_p_post["quantity"] = pred_post
    m_post = compute_all_metrics_fast(truth, df_p_post)
    c_post, u_post = calc_scores(m_post, m_base)

    # 3. 10% Blend with Champion Raw
    raw_champ = df_champ["raw_champion"].values.astype(float)
    raw_blend = 0.90 * raw_champ + 0.10 * pred_continuous
    pred_blend = apply_p(raw_blend, 1.05, 0.65)
    df_p_blend = truth[KEYS].copy()
    df_p_blend["quantity"] = pred_blend
    m_blend = compute_all_metrics_fast(truth, df_p_blend)
    c_blend, u_blend = calc_scores(m_blend, m_base)

    print(f"\n--- Metrics Summary for Fold {fold} ---", flush=True)
    print(f"  Standalone RINT:        Clipped: {c_rint:.6f} | Unclipped: {u_rint:.6f} | WAPE: {m_rint['wape']:.3f}% | RMSE: {m_rint['rmse']:.4f}", flush=True)
    print(f"  Standalone Legacy P:    Clipped: {c_post:.6f} | Unclipped: {u_post:.6f} | WAPE: {m_post['wape']:.3f}% | RMSE: {m_post['rmse']:.4f}", flush=True)
    print(f"  Champion + 10% Blend:   Clipped: {c_blend:.6f} | Unclipped: {u_blend:.6f} | WAPE: {m_blend['wape']:.3f}% | RMSE: {m_blend['rmse']:.4f}", flush=True)

    return {
        "groups": groups,
        "fold": fold,
        "train_duration_sec": train_duration,
        "eval_duration_sec": eval_duration,
        "final_train_loss": history_losses[-1] if history_losses else None,
        "standalone_rint": {"clipped": c_rint, "unclipped": u_rint, **m_rint},
        "standalone_post": {"clipped": c_post, "unclipped": u_post, **m_post},
        "blend_w10": {"clipped": c_blend, "unclipped": u_blend, **m_blend},
        "pred_continuous": pred_continuous,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=[], help="Feature groups to include")
    parser.add_argument("--fold", type=int, default=4, help="Fold to run")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")
    res = train_and_evaluate_timestral_run(
        engine=engine,
        groups=args.groups,
        fold=args.fold,
        steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
