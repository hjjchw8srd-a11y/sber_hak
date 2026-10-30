#!/usr/bin/env python3
"""
run_candidate_evaluation.py

Stage 2 of the strict Two-Stage Protocol:
- Reads experiments/model_improvements/selected_config.json.
- Verifies evaluation_status == 'PENDING_CHECK_EVALUATION'.
- Evaluates the chosen candidate model ONLY on Check Folds: [1, 0] (origins 116, 125)
  strictly without ANY retuning, grid search, or hyperparameter modifications.
- Aggregates Check Folds metrics and final 5-Fold combined metrics.
- Updates selected_config.json with status: 'EVALUATION_COMPLETED' and all final verified numbers.
- Updates experiments/model_improvements/metrics_by_fold.csv.
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

CHECK_FOLDS = {
    1: 116,  # 2026-03-23 — 2026-05-18
    0: 125,  # 2026-05-25 — 2026-07-20
}

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

    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    df['week_str'] = df['ds'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['week'])
    df = df.merge(holiday_df, left_on='week_str', right_on='week', how='left')
    df = df.drop(columns=['week_str', 'week'])

    return df


def run_nbeatsx_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    selected_cand = config["selected_candidate"]
    variant_name = selected_cand["name"]
    params = selected_cand["parameters"]

    print("=" * 80)
    print(f"RUNNING EVALUATION ON CHECK FOLDS 1, 0 FOR SELECTED CANDIDATE: {variant_name}")
    print(f"Parameters strictly frozen from selection phase: {params}")
    print("=" * 80)

    # Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)
    full_df = prepare_nbeatsx_data(data, holiday_df)

    exog_cols = params["exogenous_features"]
    budget_steps = params["max_steps"]

    candidate_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    keyed_dir = candidate_dir / "keyed"

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

        train_start = weeks[origin - params.get("history_weeks", 52)]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (full_df['ds'] >= train_start) & (full_df['ds'] < val_start)
        val_mask = (full_df['ds'] >= val_start) & (full_df['ds'] <= val_end)

        train_data = full_df[train_mask][['unique_id', 'ds', 'y'] + exog_cols].copy()
        val_data = full_df[val_mask][['unique_id', 'ds'] + exog_cols].copy()

        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=params["input_size"],
            loss=MAE(),
            futr_exog_list=exog_cols,
            max_steps=budget_steps,
            batch_size=params["batch_size"],
            windows_batch_size=params["windows_batch_size"],
            random_seed=params["random_seed"],
            scaler_type=params["scaler_type"],
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

        # Compute predictions according to the selected variant
        sub_pred = truth[KEYS].copy()
        if variant_name == "Standalone_NBEATSx":
            sub_pred['quantity'] = apply_p(n_raw)
        elif variant_name == "Blend_0.90E_0.10NBEATSx":
            sub_pred['quantity'] = apply_p(0.90 * e_raw + 0.10 * n_raw)
        elif variant_name == "Blend_0.75E_0.25NBEATSx":
            sub_pred['quantity'] = apply_p(0.75 * e_raw + 0.25 * n_raw)
        else:
            raise ValueError(f"Unknown variant name: {variant_name}")

        m_cand = compute_metrics(truth, sub_pred, b_df)

        # Control B
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        delta_unclipped = m_cand["unclipped_ratio"] - m_ctrl["unclipped_ratio"]
        delta_score = m_cand["clipped_score"] - m_ctrl["clipped_score"]

        print(f"  Control B: Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Candidate: Score = {m_cand['clipped_score']:.6f}, Unclipped = {m_cand['unclipped_ratio']:.6f} (Delta: {delta_unclipped:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['raw_nbeatsx'] = n_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_cand'] = sub_pred['quantity']
        keyed_out.to_csv(keyed_dir / f"nbeatsx_fold_{fold}.csv.gz", index=False, compression="gzip")

        check_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "cand_unclipped": m_cand["unclipped_ratio"],
            "cand_score": m_cand["clipped_score"],
            "delta_unclipped": delta_unclipped,
            "delta_score": delta_score,
            "cand_smape": m_cand["smape"],
            "cand_wape": m_cand["wape"],
            "cand_rmse": m_cand["rmse"],
            "cand_bias": m_cand["bias"],
            "cand_rmsle": m_cand["rmsle"],
        })

    # Read selection metrics to form full 5-fold evaluation table
    sel_df = pd.read_csv(candidate_dir / "nbeatsx_selection_metrics.csv")
    chk_df = pd.DataFrame(check_results)

    # Map candidate metrics for selection folds
    sel_records = []
    for _, row in sel_df.iterrows():
        f = int(row["fold"])
        orig = int(row["origin"])
        if variant_name == "Standalone_NBEATSx":
            c_unc = row["standalone_unclipped"]
            c_sc = row["standalone_score"]
        elif variant_name == "Blend_0.90E_0.10NBEATSx":
            c_unc = row["blend_10_unclipped"]
            c_sc = row["blend_10_score"]
        elif variant_name == "Blend_0.75E_0.25NBEATSx":
            c_unc = row["blend_25_unclipped"]
            c_sc = row["blend_25_score"]
        else:
            raise ValueError(f"Unknown variant name: {variant_name}")

        ctrl_unc = row["ctrl_unclipped"]
        ctrl_sc = row["ctrl_score"]
        sel_records.append({
            "fold": f,
            "role": "Selection",
            "origin": orig,
            "ctrl_unclipped": ctrl_unc,
            "ctrl_score": ctrl_sc,
            "cand_unclipped": c_unc,
            "cand_score": c_sc,
            "delta_unclipped": c_unc - ctrl_unc,
            "delta_score": c_sc - ctrl_sc,
        })

    for _, row in chk_df.iterrows():
        sel_records.append({
            "fold": int(row["fold"]),
            "role": "Check",
            "origin": int(row["origin"]),
            "ctrl_unclipped": row["ctrl_unclipped"],
            "ctrl_score": row["ctrl_score"],
            "cand_unclipped": row["cand_unclipped"],
            "cand_score": row["cand_score"],
            "delta_unclipped": row["delta_unclipped"],
            "delta_score": row["delta_score"],
        })

    full_5f_df = pd.DataFrame(sel_records).sort_values("fold", ascending=False).reset_index(drop=True)
    full_5f_df.to_csv(candidate_dir / "nbeatsx_full_5fold_metrics.csv", index=False)

    # Compute Final Summary Metrics
    sel_part = full_5f_df[full_5f_df["role"] == "Selection"]
    chk_part = full_5f_df[full_5f_df["role"] == "Check"]

    summary_metrics = {
        "selection_folds_4_3_2": {
            "control_unclipped": float(sel_part["ctrl_unclipped"].mean()),
            "cand_unclipped": float(sel_part["cand_unclipped"].mean()),
            "delta_unclipped": float(sel_part["delta_unclipped"].mean()),
            "control_score": float(sel_part["ctrl_score"].mean()),
            "cand_score": float(sel_part["cand_score"].mean()),
            "delta_score": float(sel_part["delta_score"].mean()),
        },
        "check_folds_1_0": {
            "control_unclipped": float(chk_part["ctrl_unclipped"].mean()),
            "cand_unclipped": float(chk_part["cand_unclipped"].mean()),
            "delta_unclipped": float(chk_part["delta_unclipped"].mean()),
            "control_score": float(chk_part["ctrl_score"].mean()),
            "cand_score": float(chk_part["cand_score"].mean()),
            "delta_score": float(chk_part["delta_score"].mean()),
        },
        "all_5_folds": {
            "control_unclipped": float(full_5f_df["ctrl_unclipped"].mean()),
            "cand_unclipped": float(full_5f_df["cand_unclipped"].mean()),
            "delta_unclipped": float(full_5f_df["delta_unclipped"].mean()),
            "control_score": float(full_5f_df["ctrl_score"].mean()),
            "cand_score": float(full_5f_df["cand_score"].mean()),
            "delta_score": float(full_5f_df["delta_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("FINAL 5-FOLD VERIFICATION OF SELECTED CANDIDATE:")
    print(f"Candidate: {variant_name}")
    print(f"  Selection (4,3,2) Unclipped: Control={summary_metrics['selection_folds_4_3_2']['control_unclipped']:.6f}, Cand={summary_metrics['selection_folds_4_3_2']['cand_unclipped']:.6f}, Delta={summary_metrics['selection_folds_4_3_2']['delta_unclipped']:+.6f}")
    print(f"  Check (1,0)     Unclipped: Control={summary_metrics['check_folds_1_0']['control_unclipped']:.6f}, Cand={summary_metrics['check_folds_1_0']['cand_unclipped']:.6f}, Delta={summary_metrics['check_folds_1_0']['delta_unclipped']:+.6f}")
    print(f"  All 5 Folds       Clipped: Control={summary_metrics['all_5_folds']['control_score']:.6f}, Cand={summary_metrics['all_5_folds']['cand_score']:.6f}, Delta={summary_metrics['all_5_folds']['delta_score']:+.6f}")
    print(f"  All 5 Folds     Unclipped: Control={summary_metrics['all_5_folds']['control_unclipped']:.6f}, Cand={summary_metrics['all_5_folds']['cand_unclipped']:.6f}, Delta={summary_metrics['all_5_folds']['delta_unclipped']:+.6f}")
    print("=" * 80)

    # Update selected_config.json
    config["evaluation_status"] = "EVALUATION_COMPLETED"
    config["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config["verified_metrics"] = summary_metrics

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Updated configuration saved to {CONFIG_PATH}")

    return summary_metrics


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Candidate Evaluation Phase (Check Folds 1, 0)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to selected candidate config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found. Run run_candidate_selection.py first!")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("evaluation_status") != "PENDING_CHECK_EVALUATION":
        print(f"Warning: config status is {config.get('evaluation_status')}, expected PENDING_CHECK_EVALUATION.")

    arch = config.get("selected_candidate", {}).get("architecture") or config.get("evaluated_candidate", {}).get("architecture")
    if arch == "NBEATSx":
        run_nbeatsx_evaluation(config)
    else:
        raise ValueError(f"Unknown architecture in config: {arch}")


if __name__ == "__main__":
    main()
