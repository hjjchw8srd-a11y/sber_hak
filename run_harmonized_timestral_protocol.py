#!/usr/bin/env python3
"""
run_harmonized_timestral_protocol.py

Strictly harmonized training and evaluation protocol for Timestral-209M LoRA:
- Identical temporal splitting:
  - Training origins: k in [CONTEXT, val_start - HORIZON], target <= val_start - 1
  - Internal validation: [val_start, origin - 1]
  - External test: [origin, origin + HORIZON - 1]
- Checkpoint selection: evaluates every eval_steps on internal validation,
  selects best checkpoint by internal validation unclipped_ratio (zero test leakage!)
- Evaluates selected checkpoint on external test horizon
- Identical LoRA config (r=8, alpha=16, dropout=0.05, modules=['wQKV', 'wO'])
- Identical optimizer & schedule (AdamW 1e-4, weight_decay 1e-4, warmup 30 + cosine decay)
- Identical seed (42) and sampling
"""

from __future__ import annotations

import sys
import os
import time
import math
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from timestral import TimestralForecaster
from experiments.timestral_feature_research.timestral_feature_engine import (
    TimestralFeatureEngine, CONTEXT, HORIZON
)

KEYS = ["week", "warehouse_name", "article"]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}


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


def run_harmonized_evaluation(
    forecaster: TimestralForecaster,
    engine: TimestralFeatureEngine,
    groups: List[str],
    eval_origin: int,
    truth_df: pd.DataFrame,
    base_df: pd.DataFrame,
    device: torch.device,
    eval_batch_size: int = 64,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Runs batch inference and computes exact competition metrics."""
    forecaster.model.eval()
    all_median_preds = []
    all_indices = np.arange(engine.n_series)

    with torch.inference_mode():
        for b_start in range(0, engine.n_series, eval_batch_size):
            b_idx = all_indices[b_start : b_start + eval_batch_size]

            tgt_ctx, _, past_arr, known_arr, _, _ = engine.assemble_batch_tensors(
                groups=groups,
                origin=eval_origin,
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

    pred_matrix = np.concatenate(all_median_preds, axis=0)  # (8789, 9)
    pred_continuous = pred_matrix.T.reshape(-1)             # 79,101

    m_base = compute_all_metrics_fast(truth_df, base_df)

    # RINT
    pred_rint = np.rint(np.clip(pred_continuous, 0, None)).astype(int)
    df_p_rint = truth_df[KEYS].copy()
    df_p_rint["quantity"] = pred_rint
    m_rint = compute_all_metrics_fast(truth_df, df_p_rint)
    c_rint, u_rint = calc_scores(m_rint, m_base)

    # Legacy P
    pred_post = apply_p(pred_continuous, 1.05, 0.65)
    df_p_post = truth_df[KEYS].copy()
    df_p_post["quantity"] = pred_post
    m_post = compute_all_metrics_fast(truth_df, df_p_post)
    c_post, u_post = calc_scores(m_post, m_base)

    results = {
        "rint": {"clipped": c_rint, "unclipped": u_rint, **m_rint},
        "post": {"clipped": c_post, "unclipped": u_post, **m_post},
    }
    return results, pred_continuous


def train_harmonized_model(
    engine: TimestralFeatureEngine,
    groups: List[str],
    fold: int = 4,
    total_steps: int = 600,
    eval_steps: int = 150,
    microbatch: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    device_str: str = "cuda",
    save_tag: str = "harmonized",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    origin = FOLD_ORIGINS[fold]
    val_start = origin - HORIZON
    train_history_end = val_start

    print(f"\n{'='*100}")
    print(f"HARMONIZED TIMESTRAL RUN: Fold {fold} (Origin {origin}) | Groups: {groups or ['None (Univariate)']}")
    print(f"Protocol: train history <= {train_history_end-1}, internal val [{val_start}..{origin-1}], test [{origin}..{origin+HORIZON-1}]")
    print(f"Hyperparameters: steps={total_steps}, eval_every={eval_steps}, microbatch={microbatch}, accum={grad_accum}, lr={lr}, seed={seed}")
    print(f"{'='*100}")

    # Valid training origins: target strictly <= train_history_end - 1
    # latest target: k + HORIZON - 1 <= train_history_end - 1  ==>  k <= train_history_end - HORIZON
    valid_train_origins = list(range(CONTEXT, train_history_end - HORIZON + 1))
    assert len(valid_train_origins) > 0, f"No train origins for train cutoff {train_history_end}!"
    print(f"Valid training origins count: {len(valid_train_origins)} (origins {valid_train_origins[0]}..{valid_train_origins[-1]})")

    # Internal Validation Dataset
    # y_true on weeks [val_start .. origin - 1]
    val_int_truth_series = engine.matrices.quantity[val_start : origin, :].reshape(-1)
    # base mean13 on weeks [val_start - 13 : val_start]
    val_m13_arr = engine.matrices.quantity[val_start - 13 : val_start, :].mean(axis=0)
    val_m13_series = np.tile(val_m13_arr, HORIZON)

    # Format internal val keys
    val_weeks_str = [engine.weeks[val_start + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    val_rows = []
    for h_idx, w_str in enumerate(val_weeks_str):
        for s_idx in range(engine.n_series):
            val_rows.append({
                "week": w_str,
                "warehouse_name": engine.pairs.iloc[s_idx]["warehouse_name"],
                "article": engine.pairs.iloc[s_idx]["article"],
            })
    val_keys_df = pd.DataFrame(val_rows)
    val_truth_df = val_keys_df.assign(quantity=val_int_truth_series)
    val_base_df = val_keys_df.assign(quantity=val_m13_series)

    # External Test Dataset (loaded from official champion OOF for exact parity)
    champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
    df_champ = pd.read_csv(champ_path)
    test_truth_df = df_champ[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
    test_base_df = df_champ[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})

    # Model & LoRA initialization from authentic pretrained weights
    weights_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
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
    warmup_steps = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-5)

    sampler = StratifiedSeriesSampler(engine.matrices.quantity, train_history_end, seed=seed)
    quantile_knots = torch.tensor(forecaster.quantile_levels, device=device, dtype=torch.float32)

    ps = forecaster.patch_size
    horizon_patches = math.ceil(HORIZON / ps)

    train_start_time = time.time()
    step = 0
    accum_count = 0
    running_loss = 0.0

    ckpt_save_dir = ROOT / "experiments" / "timestral_feature_research" / "checkpoints" / f"{save_tag}_fold{fold}"
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    best_val_unclipped = float("inf")
    best_step = None
    best_ckpt_path = None
    val_records = []

    optimizer.zero_grad()
    print("Beginning harmonized training loop...")

    while step < total_steps:
        tr_origin = int(np.random.choice(valid_train_origins))
        batch_indices = sampler.sample_indices(batch_size=microbatch)

        tgt_ctx, y_true_arr, past_arr, known_arr, _, _ = engine.assemble_batch_tensors(
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
            running_loss = 0.0

            if step % 50 == 0 or step == total_steps:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  Step {step:03d}/{total_steps} | Loss: {avg_step_loss:.4f} | LR: {cur_lr:.2e}", flush=True)

            # Checkpoint evaluation on INTERNAL VALIDATION
            if step % eval_steps == 0 or step == total_steps:
                print(f"  [Internal Val at Step {step}] Evaluating on weeks {val_start}..{origin-1}...", flush=True)
                val_res, _ = run_harmonized_evaluation(
                    forecaster=forecaster,
                    engine=engine,
                    groups=groups,
                    eval_origin=val_start,
                    truth_df=val_truth_df,
                    base_df=val_base_df,
                    device=device,
                )
                val_unc = val_res["post"]["unclipped"]
                val_clp = val_res["post"]["clipped"]
                print(f"    Val Post: Clipped = {val_clp:.6f}, Unclipped = {val_unc:.6f}, WAPE = {val_res['post']['wape']:.3f}%, RMSE = {val_res['post']['rmse']:.4f}", flush=True)

                step_ckpt_path = ckpt_save_dir / f"step_{step}"
                lora_model.save_pretrained(str(step_ckpt_path))

                val_records.append({
                    "step": step,
                    "val_clipped": val_clp,
                    "val_unclipped": val_unc,
                    "val_wape": val_res["post"]["wape"],
                    "val_rmse": val_res["post"]["rmse"],
                })

                if val_unc < best_val_unclipped:
                    best_val_unclipped = val_unc
                    best_step = step
                    best_ckpt_path = step_ckpt_path
                    print(f"    >>> NEW BEST CHECKPOINT: Step {step} (Val Unclipped = {val_unc:.6f})", flush=True)

                lora_model.train()

    train_duration = time.time() - train_start_time
    print(f"\nTraining completed in {train_duration:.1f}s. Best checkpoint: Step {best_step} (Val Unclipped: {best_val_unclipped:.6f})", flush=True)

    # Reload best checkpoint for EXTERNAL TEST evaluation
    print(f"Reloading best checkpoint from {best_ckpt_path} for honest external test evaluation...")
    forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    forecaster.model = PeftModel.from_pretrained(forecaster.model, str(best_ckpt_path))
    forecaster.model.eval()

    test_start = time.time()
    test_results, pred_continuous = run_harmonized_evaluation(
        forecaster=forecaster,
        engine=engine,
        groups=groups,
        eval_origin=origin,
        truth_df=test_truth_df,
        base_df=test_base_df,
        device=device,
    )
    test_duration = time.time() - test_start

    # Compute blend with champion raw
    raw_champ = df_champ["raw_champion"].values.astype(float)
    raw_blend = 0.90 * raw_champ + 0.10 * pred_continuous
    pred_blend = apply_p(raw_blend, 1.05, 0.65)
    df_p_blend = test_truth_df[KEYS].copy()
    df_p_blend["quantity"] = pred_blend
    m_base = compute_all_metrics_fast(test_truth_df, test_base_df)
    m_blend = compute_all_metrics_fast(test_truth_df, df_p_blend)
    c_blend, u_blend = calc_scores(m_blend, m_base)

    print(f"\n{'='*100}")
    print(f"HONEST EXTERNAL TEST RESULTS: Fold {fold} | Groups: {groups or ['None (Univariate)']}")
    print(f"  Selected Checkpoint: Step {best_step} (Internal Val Unclipped: {best_val_unclipped:.6f})")
    print(f"  Standalone RINT:      Clipped = {test_results['rint']['clipped']:.6f} | Unclipped = {test_results['rint']['unclipped']:.6f} | WAPE = {test_results['rint']['wape']:.3f}% | RMSE = {test_results['rint']['rmse']:.4f}")
    print(f"  Standalone Legacy P:  Clipped = {test_results['post']['clipped']:.6f} | Unclipped = {test_results['post']['unclipped']:.6f} | WAPE = {test_results['post']['wape']:.3f}% | RMSE = {test_results['post']['rmse']:.4f}")
    print(f"  Champion + 10% Blend: Clipped = {c_blend:.6f} | Unclipped = {u_blend:.6f} | WAPE = {m_blend['wape']:.3f}% | RMSE = {m_blend['rmse']:.4f}")
    print(f"{'='*100}\n", flush=True)

    out_json = ROOT / "experiments" / "timestral_feature_research" / f"harmonized_{save_tag}_fold{fold}_results.json"
    summary_save = {
        "model": f"Timestral LoRA ({save_tag})",
        "groups": groups,
        "fold": fold,
        "best_step": best_step,
        "val_unclipped": best_val_unclipped,
        "val_records": val_records,
        "train_duration_sec": train_duration,
        "test_eval_duration_sec": test_duration,
        "standalone_rint": test_results["rint"],
        "standalone_post": test_results["post"],
        "blend_w10": {"clipped": c_blend, "unclipped": u_blend, **m_blend},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_save, f, indent=2)
    print(f"Saved results summary to {out_json.name}", flush=True)

    return {
        "groups": groups,
        "fold": fold,
        "best_step": best_step,
        "best_val_unclipped": best_val_unclipped,
        "train_duration_sec": train_duration,
        "test_eval_duration_sec": test_duration,
        "val_records": val_records,
        "test_rint": test_results["rint"],
        "test_post": test_results["post"],
        "test_blend_w10": {"clipped": c_blend, "unclipped": u_blend, **m_blend},
        "pred_continuous": pred_continuous,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tag", type=str, default="harmonized")
    args = parser.parse_args()

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")
    res = train_harmonized_model(
        engine=engine,
        groups=args.groups,
        fold=args.fold,
        total_steps=args.steps,
        eval_steps=args.eval_steps,
        microbatch=args.microbatch,
        grad_accum=args.grad_accum,
        seed=args.seed,
        save_tag=args.save_tag,
    )


if __name__ == "__main__":
    main()
