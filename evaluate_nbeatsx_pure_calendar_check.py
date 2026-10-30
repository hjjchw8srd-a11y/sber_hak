#!/usr/bin/env python3
"""
evaluate_nbeatsx_pure_calendar_check.py

Stage 2 evaluation of NBEATSx Pure Calendar (5 exog, no holidays, C_raw >= 0)
on Check Folds: [1, 0] (origins 116, 125).
Combines results with Selection Folds [4, 3, 2] to produce full 5-fold metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

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

CHECK_FOLDS = {
    1: 116,  # 2026-03-23 — 2026-05-18
    0: 125,  # 2026-05-25 — 2026-07-20
}

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidates" / "nbeatsx"
KEYED_DIR = OUTPUT_DIR / "keyed"
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
    print("EVALUATION OF NBEATSx PURE CALENDAR ON CHECK FOLDS: 1, 0")
    print("=" * 80)

    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    calendar_exog_cols = ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']

    check_results = []
    t_start_all = time.perf_counter()

    for fold, origin in CHECK_FOLDS.items():
        print(f"\n--- Processing Check Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

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
        c_raw = np.maximum(n_raw, 0.0)

        # Standalone
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(c_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # Blend 10%
        b10_raw = 0.90 * e_raw + 0.10 * c_raw
        sub_b10 = truth[KEYS].copy()
        sub_b10['quantity'] = apply_p(b10_raw)
        m_b10 = compute_metrics(truth, sub_b10, b_df)

        # Blend 25%
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

        check_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_b10["unclipped_ratio"],
            "blend_10_score": m_b10["clipped_score"],
            "blend_25_unclipped": m_b25["unclipped_ratio"],
            "blend_25_score": m_b25["clipped_score"],
        })

    # Combine with selection folds
    sel_df = pd.read_csv(OUTPUT_DIR / "nbeatsx_pure_calendar_selection_metrics.csv")
    chk_df = pd.DataFrame(check_results)
    all5_df = pd.concat([sel_df, chk_df], ignore_index=True)
    all5_df.to_csv(OUTPUT_DIR / "nbeatsx_pure_calendar_all5_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("FULL 5-FOLD SUMMARY FOR NBEATSx PURE CALENDAR:")
    print("=" * 80)
    for variant in ["standalone", "blend_10", "blend_25"]:
        sel_u = sel_df[f"{variant}_unclipped"].mean()
        chk_u = chk_df[f"{variant}_unclipped"].mean()
        all_u = all5_df[f"{variant}_unclipped"].mean()
        all_s = all5_df[f"{variant}_score"].mean()

        ctrl_sel = sel_df["ctrl_unclipped"].mean()
        ctrl_chk = chk_df["ctrl_unclipped"].mean()
        ctrl_all_u = all5_df["ctrl_unclipped"].mean()
        ctrl_all_s = all5_df["ctrl_score"].mean()

        print(f"Variant: {variant}")
        print(f"  Selection (4,3,2): Unclipped = {sel_u:.6f} vs Ctrl = {ctrl_sel:.6f} (Delta = {sel_u - ctrl_sel:+.6f})")
        print(f"  Check (1,0):       Unclipped = {chk_u:.6f} vs Ctrl = {ctrl_chk:.6f} (Delta = {chk_u - ctrl_chk:+.6f})")
        print(f"  All 5 Folds:       Unclipped = {all_u:.6f} vs Ctrl = {ctrl_all_u:.6f} (Delta = {all_u - ctrl_all_u:+.6f})")
        print(f"                     Score     = {all_s:.6f} vs Ctrl = {ctrl_all_s:.6f} (Delta = {all_s - ctrl_all_s:+.6f})")

    summary_5fold = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": time.perf_counter() - t_start_all,
        "selection_unclipped_ctrl": float(sel_df["ctrl_unclipped"].mean()),
        "selection_unclipped_b10": float(sel_df["blend_10_unclipped"].mean()),
        "check_unclipped_ctrl": float(chk_df["ctrl_unclipped"].mean()),
        "check_unclipped_b10": float(chk_df["blend_10_unclipped"].mean()),
        "all5_unclipped_ctrl": float(all5_df["ctrl_unclipped"].mean()),
        "all5_unclipped_b10": float(all5_df["blend_10_unclipped"].mean()),
        "all5_clipped_ctrl": float(all5_df["ctrl_score"].mean()),
        "all5_clipped_b10": float(all5_df["blend_10_score"].mean()),
        "all5_delta_unclipped_b10": float(all5_df["blend_10_unclipped"].mean() - all5_df["ctrl_unclipped"].mean()),
        "all5_delta_clipped_b10": float(all5_df["blend_10_score"].mean() - all5_df["ctrl_score"].mean()),
    }
    with open(OUTPUT_DIR / "nbeatsx_pure_calendar_5fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_5fold, f, indent=2)
    print(f"\nSaved 5-fold summary to {OUTPUT_DIR / 'nbeatsx_pure_calendar_5fold_summary.json'}")


if __name__ == "__main__":
    main()
