#!/usr/bin/env python3
"""
run_candidate_selection.py

Stage 1 of the strict Two-Stage Protocol:
- Evaluates Candidate models ONLY on Selection Folds: [4, 3, 2] (origins 89, 98, 107).
- Computes mean unclipped_ratio on Folds 4, 3, 2.
- Compares with Control B (mean unclipped_ratio = 0.7613139384).
- Selects the best performing configuration strictly on Selection Folds.
- Freezes all parameters and writes experiments/model_improvements/selected_config.json
  with status: 'PENDING_CHECK_EVALUATION'.
- Does NOT execute on Check Folds (1, 0) - that is reserved strictly for run_candidate_evaluation.py.
"""

from __future__ import annotations

import argparse
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
from holiday_calendar import build_holiday_feature_dataframe, STATUTORY_FEATURE_COLUMNS

SELECTION_FOLDS = {
    4: 89,   # 2025-09-15 — 2025-11-10
    3: 98,   # 2025-11-17 — 2026-01-12
    2: 107,  # 2026-01-19 — 2026-03-16
}

CONTROL_B_SELECTION_UNCLIPPED = 0.7613139384
CONTROL_B_SELECTION_CLIPPED = 0.7345392853

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements"
CONFIG_PATH = OUTPUT_DIR / "selected_config.json"


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


def prepare_nbeatsx_data(data: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    # Calendar exogenous features
    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    # Attach statutory holiday features (Block A, 7 features)
    df['week_str'] = df['ds'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['week'])
    df = df.merge(holiday_df, left_on='week_str', right_on='week', how='left')
    df = df.drop(columns=['week_str', 'week'])

    return df


def run_nbeatsx_selection(budget_steps: int = 500) -> dict[str, Any]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    print("=" * 80)
    print(f"RUNNING NBEATSx SELECTION ON FOLDS 4, 3, 2 (BUDGET: {budget_steps} STEPS, SEED: 42)")
    print("=" * 80)

    # Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("Building statutory holiday features...")
    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)

    print("Formatting NeuralForecast DataFrame with exogenous calendar + statutory features...")
    full_df = prepare_nbeatsx_data(data, holiday_df)

    exog_cols = [
        'month', 'quarter', 'week_of_year', 'week_sin', 'week_cos',
        'holiday_days', 'holiday_days_prev_week', 'holiday_days_next_week',
        'new_year_days', 'may_holiday_days', 'days_to_nearest_holiday', 'days_since_last_holiday'
    ]

    candidate_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    keyed_dir = candidate_dir / "keyed"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    keyed_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    t_start_all = time.perf_counter()

    for fold, origin in SELECTION_FOLDS.items():
        print(f"\n--- Processing Selection Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        # Ground truth and baseline mean13
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

        # Prepare temporal split: 52 weeks history strictly before origin
        train_start = weeks[origin - 52]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (full_df['ds'] >= train_start) & (full_df['ds'] < val_start)
        val_mask = (full_df['ds'] >= val_start) & (full_df['ds'] <= val_end)

        train_data = full_df[train_mask][['unique_id', 'ds', 'y'] + exog_cols].copy()
        val_data = full_df[val_mask][['unique_id', 'ds'] + exog_cols].copy()

        print(f"  Train: {train_start} to {weeks[origin-1]} ({len(train_data)} rows)")
        print(f"  Val Horizon: {val_start} to {val_end} ({len(val_data)} rows)")

        # Initialize NBEATSx
        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=26,
            loss=MAE(),
            futr_exog_list=exog_cols,
            max_steps=budget_steps,
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

        # Align predictions with truth keys
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
        assert len(merged_p) == 79101, f"Prediction length mismatch: {len(merged_p)}"
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)

        # 1. Standalone NBEATSx: P(N_raw)
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(n_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # 2. Blend 10%: P(0.90 * E_raw + 0.10 * N_raw)
        blend_10_raw = 0.90 * e_raw + 0.10 * n_raw
        sub_blend_10 = truth[KEYS].copy()
        sub_blend_10['quantity'] = apply_p(blend_10_raw)
        m_blend_10 = compute_metrics(truth, sub_blend_10, b_df)

        # 3. Blend 25%: P(0.75 * E_raw + 0.25 * N_raw)
        blend_25_raw = 0.75 * e_raw + 0.25 * n_raw
        sub_blend_25 = truth[KEYS].copy()
        sub_blend_25['quantity'] = apply_p(blend_25_raw)
        m_blend_25 = compute_metrics(truth, sub_blend_25, b_df)

        # Control B metrics
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        print(f"  Control B:      Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone N:   Score = {m_standalone['clipped_score']:.6f}, Unclipped = {m_standalone['unclipped_ratio']:.6f}")
        print(f"  Blend 10% (N):  Score = {m_blend_10['clipped_score']:.6f}, Unclipped = {m_blend_10['unclipped_ratio']:.6f} (Delta: {m_blend_10['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")
        print(f"  Blend 25% (N):  Score = {m_blend_25['clipped_score']:.6f}, Unclipped = {m_blend_25['unclipped_ratio']:.6f} (Delta: {m_blend_25['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['raw_nbeatsx'] = n_raw
        keyed_out['raw_blend_10'] = blend_10_raw
        keyed_out['raw_blend_25'] = blend_25_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_standalone'] = sub_standalone['quantity']
        keyed_out['pred_blend_10'] = sub_blend_10['quantity']
        keyed_out['pred_blend_25'] = sub_blend_25['quantity']
        keyed_out.to_csv(keyed_dir / f"nbeatsx_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_blend_10["unclipped_ratio"],
            "blend_10_score": m_blend_10["clipped_score"],
            "blend_25_unclipped": m_blend_25["unclipped_ratio"],
            "blend_25_score": m_blend_25["clipped_score"],
        })

    total_time = time.perf_counter() - t_start_all
    df_res = pd.DataFrame(fold_results)
    df_res.to_csv(candidate_dir / "nbeatsx_selection_metrics.csv", index=False)

    # Compute selection summary
    mean_ctrl_unc = df_res["ctrl_unclipped"].mean()
    mean_standalone_unc = df_res["standalone_unclipped"].mean()
    mean_blend10_unc = df_res["blend_10_unclipped"].mean()
    mean_blend25_unc = df_res["blend_25_unclipped"].mean()

    variants = {
        "Standalone_NBEATSx": {
            "mean_unclipped": float(mean_standalone_unc),
            "delta_vs_control": float(mean_standalone_unc - mean_ctrl_unc),
            "mean_score": float(df_res["standalone_score"].mean()),
        },
        "Blend_0.90E_0.10NBEATSx": {
            "mean_unclipped": float(mean_blend10_unc),
            "delta_vs_control": float(mean_blend10_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_10_score"].mean()),
        },
        "Blend_0.75E_0.25NBEATSx": {
            "mean_unclipped": float(mean_blend25_unc),
            "delta_vs_control": float(mean_blend25_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_25_score"].mean()),
        },
    }

    # Best variant by selection criterion (minimum unclipped ratio on folds 4, 3, 2)
    best_variant_name = min(variants.keys(), key=lambda k: variants[k]["mean_unclipped"])
    best_variant_info = variants[best_variant_name]

    print("\n" + "=" * 80)
    print("NBEATSx SELECTION SUMMARY ON FOLDS 4, 3, 2:")
    print(f"  Control B Baseline:          Unclipped = {mean_ctrl_unc:.6f}")
    for name, v in variants.items():
        print(f"  {name:28s}: Unclipped = {v['mean_unclipped']:.6f} (Delta = {v['delta_vs_control']:+.6f}, Score = {v['mean_score']:.6f})")
    # Compare with Control B explicitly
    beats_control = bool(best_variant_info["delta_vs_control"] < 0)
    selection_decision = "ADVANCE_TO_CHECK_EVALUATION" if beats_control else "RETAIN_CONTROL_REJECT_CANDIDATE"
    winner = best_variant_name if beats_control else "Blend_LightGBM_B (Retain Control)"

    print(f"\nSelection Decision: {selection_decision}")
    print(f"Winner on Selection Folds: {winner} (Delta vs Control = {best_variant_info['delta_vs_control']:+.6f})")
    print("=" * 80)

    # Write isolated candidate config
    cand_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_config_path = cand_dir / "selected_config.json"

    selected_payload = {
        "protocol_version": "2.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_decision": selection_decision,
        "winner_on_selection": winner,
        "evaluation_status": "PENDING_CHECK_EVALUATION" if beats_control else "SELECTION_REJECTED",
        "selection_folds": [4, 3, 2],
        "check_folds": [1, 0],
        "selection_criterion": "mean_unclipped_ratio_folds_4_3_2",
        "control_baseline": {
            "name": "Blend_LightGBM_B",
            "selection_unclipped_ratio": float(mean_ctrl_unc),
            "selection_clipped_score": float(df_res["ctrl_score"].mean()),
        },
        "evaluated_candidate": {
            "name": best_variant_name,
            "architecture": "NBEATSx",
            "parameters": {
                "h": 9,
                "input_size": 26,
                "loss": "MAE",
                "max_steps": budget_steps,
                "batch_size": 256,
                "windows_batch_size": 512,
                "scaler_type": "standard",
                "random_seed": 42,
                "exogenous_features": exog_cols,
                "history_weeks": 52,
            },
            "selection_metrics": {
                "mean_unclipped_ratio": best_variant_info["mean_unclipped"],
                "delta_vs_control": best_variant_info["delta_vs_control"],
                "mean_clipped_score": best_variant_info["mean_score"],
            },
            "beats_control_on_selection": beats_control,
        },
        "all_evaluated_selection_variants": variants,
        "runtime_selection_sec": float(total_time),
    }

    with open(cand_config_path, "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, indent=2)
    print(f"Wrote isolated candidate configuration to {cand_config_path}")
    print("Selection phase complete. Champion config remains untouched.")

    return selected_payload


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Candidate Selection Phase (Folds 4, 3, 2)")
    parser.add_argument("--candidate", type=str, default="nbeatsx", choices=["nbeatsx"], help="Candidate to evaluate")
    parser.add_argument("--steps", type=int, default=500, help="Training max steps budget")
    args = parser.parse_args()

    if args.candidate == "nbeatsx":
        run_nbeatsx_selection(budget_steps=args.steps)


if __name__ == "__main__":
    main()
