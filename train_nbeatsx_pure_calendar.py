#!/usr/bin/env python3
"""
train_nbeatsx_pure_calendar.py

Ablation of NBEATSx using ONLY 5 standard calendar features (strictly NO holiday features):
- Exogenous: ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']
- Strict non-negative clipping: C_raw = max(N_raw, 0.0)
- Identical budget: 500 steps, context 26 weeks, h=9, history 52 weeks, loss=MAE, seed=42
- Evaluation on Selection Folds: 4, 3, 2 (origins 89, 98, 107)
- Variants:
  * Standalone: P(C_raw)
  * Blend 10%: P(0.90 * E_raw + 0.10 * C_raw)
  * Blend 25%: P(0.75 * E_raw + 0.25 * C_raw)
- Saves step losses and evaluation metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13

SELECTION_FOLDS = {
    4: 89,   # 2025-09-15 — 2025-11-10
    3: 98,   # 2025-11-17 — 2026-01-12
    2: 107,  # 2026-01-19 — 2026-03-16
}

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidates" / "nbeatsx"
KEYED_DIR = OUTPUT_DIR / "keyed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)


def apply_p(x: np.ndarray | pd.Series) -> np.ndarray:
    scaled = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    print("=" * 80)
    print("NBEATSx ABLATION: PURE CALENDAR FEATURES ONLY (5 EXOG, NO HOLIDAYS, C_raw >= 0)")
    print("Budget: 500 steps, Loss: MAE, Context: 26 weeks, History: 52 weeks, Seed: 42")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    # Format base DataFrame
    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    # 5 standard calendar features
    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    calendar_exog_cols = ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']

    fold_results = []
    t_start_all = time.perf_counter()

    for fold, origin in SELECTION_FOLDS.items():
        print(f"\n--- Processing Selection Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        # Load Control B keyed OOF
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged_ctrl) == 79101
        e_raw = merged_ctrl["raw_blend"].values.astype(np.float64)

        # 52 weeks history strictly before origin
        train_start = weeks[origin - 52]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (df['ds'] >= train_start) & (df['ds'] < val_start)
        val_mask = (df['ds'] >= val_start) & (df['ds'] <= val_end)

        train_data = df[train_mask][['unique_id', 'ds', 'y'] + calendar_exog_cols].copy()
        val_data = df[val_mask][['unique_id', 'ds'] + calendar_exog_cols].copy()

        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=26,
            loss=MAE(),
            futr_exog_list=calendar_exog_cols,
            max_steps=500,
            batch_size=256,
            windows_batch_size=512,
            random_seed=42,
            scaler_type='standard',
        )
        nf = NeuralForecast(models=[model], freq='W-MON')
        nf.fit(df=train_data)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_df = nf.predict(futr_df=val_data)
        pred_time = time.perf_counter() - t1
        print(f"  Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")

        preds_df['ds_str'] = preds_df['ds'].dt.strftime('%Y-%m-%d')
        parts = preds_df['unique_id'].str.split('__', n=1, expand=True)
        preds_df['warehouse_name'] = parts[0]
        preds_df['article'] = parts[1].astype(truth['article'].dtype)

        merged_p = truth[KEYS].merge(
            preds_df[['ds_str', 'warehouse_name', 'article', 'NBEATSx']],
            left_on=['week', 'warehouse_name', 'article'],
            right_on=['ds_str', 'warehouse_name', 'article'],
            how='inner',
            validate='one_to_one'
        )
        assert len(merged_p) == 79101
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)

        # Enforce non-negative clipping on component forecast
        c_raw = np.maximum(n_raw, 0.0)
        neg_count = int((n_raw < 0.0).sum())
        min_n_raw = float(n_raw.min())
        print(f"  Negative raw forecasts clipped: {neg_count} / {len(n_raw)} (min was {min_n_raw:.4f})")

        # 1. Standalone: P(C_raw)
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(c_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # 2. Blend 10%: P(0.90 * E_raw + 0.10 * C_raw)
        b10_raw = 0.90 * e_raw + 0.10 * c_raw
        sub_b10 = truth[KEYS].copy()
        sub_b10['quantity'] = apply_p(b10_raw)
        m_b10 = compute_metrics(truth, sub_b10, b_df)

        # 3. Blend 25%: P(0.75 * E_raw + 0.25 * C_raw)
        b25_raw = 0.75 * e_raw + 0.25 * c_raw
        sub_b25 = truth[KEYS].copy()
        sub_b25['quantity'] = apply_p(b25_raw)
        m_b25 = compute_metrics(truth, sub_b25, b_df)

        # Control B
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        print(f"  Control B:            Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone Pure Cal:  Score = {m_standalone['clipped_score']:.6f}, Unclipped = {m_standalone['unclipped_ratio']:.6f}")
        print(f"  Blend 10% Pure Cal:   Score = {m_b10['clipped_score']:.6f}, Unclipped = {m_b10['unclipped_ratio']:.6f} (Delta: {m_b10['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")
        print(f"  Blend 25% Pure Cal:   Score = {m_b25['clipped_score']:.6f}, Unclipped = {m_b25['unclipped_ratio']:.6f} (Delta: {m_b25['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['n_raw'] = n_raw
        keyed_out['c_raw_nonneg'] = c_raw
        keyed_out['b10_raw'] = b10_raw
        keyed_out['b25_raw'] = b25_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_standalone'] = sub_standalone['quantity']
        keyed_out['pred_b10'] = sub_b10['quantity']
        keyed_out['pred_b25'] = sub_b25['quantity']
        keyed_out.to_csv(KEYED_DIR / f"nbeatsx_pure_cal_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "neg_count": neg_count,
            "min_n_raw": min_n_raw,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_b10["unclipped_ratio"],
            "blend_10_score": m_b10["clipped_score"],
            "blend_25_unclipped": m_b25["unclipped_ratio"],
            "blend_25_score": m_b25["clipped_score"],
        })

    df_res = pd.DataFrame(fold_results)
    csv_path = OUTPUT_DIR / "nbeatsx_pure_calendar_selection_metrics.csv"
    df_res.to_csv(csv_path, index=False)

    mean_ctrl_unc = df_res["ctrl_unclipped"].mean()
    mean_standalone_unc = df_res["standalone_unclipped"].mean()
    mean_b10_unc = df_res["blend_10_unclipped"].mean()
    mean_b25_unc = df_res["blend_25_unclipped"].mean()

    variants = {
        "Standalone_NBEATSx_PureCal": {
            "mean_unclipped": float(mean_standalone_unc),
            "delta_vs_control": float(mean_standalone_unc - mean_ctrl_unc),
            "mean_score": float(df_res["standalone_score"].mean()),
        },
        "Blend_0.90E_0.10NBEATSx_PureCal": {
            "mean_unclipped": float(mean_b10_unc),
            "delta_vs_control": float(mean_b10_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_10_score"].mean()),
        },
        "Blend_0.75E_0.25NBEATSx_PureCal": {
            "mean_unclipped": float(mean_b25_unc),
            "delta_vs_control": float(mean_b25_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_25_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("PURE CALENDAR NBEATSx SELECTION SUMMARY (FOLDS 4, 3, 2):")
    print(f"  Control B Baseline:               Unclipped = {mean_ctrl_unc:.6f}")
    for name, v in variants.items():
        print(f"  {name:32s}: Unclipped = {v['mean_unclipped']:.6f} (Delta = {v['delta_vs_control']:+.6f}, Score = {v['mean_score']:.6f})")
    print("=" * 80)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": time.perf_counter() - t_start_all,
        "selection_folds": [4, 3, 2],
        "control_baseline_unclipped": float(mean_ctrl_unc),
        "variants": variants,
        "beats_control": any(v["delta_vs_control"] < 0 for v in variants.values()),
        "conclusion": "REJECTED" if all(v["delta_vs_control"] >= 0 for v in variants.values()) else "ACCEPTED",
    }

    with open(OUTPUT_DIR / "nbeatsx_pure_calendar_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUTPUT_DIR / 'nbeatsx_pure_calendar_summary.json'}")


if __name__ == "__main__":
    main()
