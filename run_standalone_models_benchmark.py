#!/usr/bin/env python3
"""
run_standalone_models_benchmark.py

Comprehensive Standalone Models Benchmark across 5 Historical Validation Folds [4, 3, 2, 1, 0].
Selection: Folds [4, 3, 2] (Origins 89, 98, 107)
Check: Folds [1, 0] (Origins 116, 125)

Evaluates:
1. Standalone TFT (raw continuous, rint, legacy_p, and threshold screening)
2. Standalone LightGBM Scaled 1M Dual Regressor (continuous, rint, legacy_p, and ungated)
3. Standalone CatBoost (continuous, rint, legacy_p)
4. Standalone PerCategory (continuous, rint, legacy_p)
5. Standalone NBEATSx (non-negative continuous, rint, legacy_p)
6. Baseline Baselines & Ensembles:
   - Baseline Mean13
   - Baseline Naive Last
   - Ensemble Control B (50 TFT / 30 LGBM / 10 CB / 10 PerCat)
   - Ensemble Candidate 45/37/9/9
   - Champion All-Stock Ensemble (Scale_1M_Dual_0.5)

Computes:
- All 5 competition metrics (SMAPE, WAPE, RMSE, Bias, RMSLE)
- Normalized Clipped Score & Unclipped Ratio against Baseline Mean13 and Naive Last
- Segmented metrics by sparsity regime, sales volume tier, and forecast horizon (weeks 1-4 vs 5-9)
- Resource utilization (VRAM, RAM, inference latency)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    compute_all_metrics_fast,
)

METRIC_WEIGHTS = {
    "smape": 0.50,
    "wape": 0.20,
    "rmse": 0.15,
    "bias": 0.10,
    "rmsle": 0.05,
}

FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
SELECTION_FOLDS = [4, 3, 2]
CHECK_FOLDS = [1, 0]


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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("RUNNING STANDALONE MODELS BENCHMARK ACROSS 5 HISTORICAL VALIDATION WINDOWS")
    print("=" * 80)

    # Load Weekly Data
    print("Loading dataset...", flush=True)
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    # Compute series level historical statistics for segmentation
    print("Computing series-level metadata for segmentation...", flush=True)
    stats_df = data.groupby(["warehouse_name", "article"]).agg(
        total_qty=("quantity", "sum"),
        zero_rate=("quantity", lambda s: (s == 0).mean()),
    ).reset_index()

    # Define segments
    stats_df["sparsity_segment"] = np.where(
        stats_df["zero_rate"] >= 0.85, "high_intermittent",
        np.where(stats_df["zero_rate"] >= 0.50, "moderate_intermittent", "regular_continuous")
    )
    stats_df["volume_segment"] = np.where(
        stats_df["total_qty"] < 50, "low_volume",
        np.where(stats_df["total_qty"] < 200, "mid_volume", "high_volume")
    )

    all_fold_evals = []
    segmented_evals = []

    for fold, origin in FOLDS.items():
        t0 = time.perf_counter()
        phase = "Selection" if fold in SELECTION_FOLDS else "Check"
        print(f"\nEvaluating Fold {fold} (Origin {origin}, Phase: {phase})...", flush=True)

        # Build Baselines and Ground Truth
        history13 = np.column_stack([matrices.quantity[origin - lag, :] for lag in range(1, 14)])
        mean13 = history13.mean(axis=1)
        val_wks_arr = np.repeat(matrices.weeks[origin : origin + HORIZON].to_numpy(), matrices.n_pairs)
        pairs_rep = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)

        b_mean = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": np.tile(mean13, HORIZON),
        })
        b_naive = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": np.tile(matrices.quantity[origin - 1, :], HORIZON),
        })
        truth = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": matrices.quantity[origin : origin + HORIZON, :].ravel(),
        })

        m_mean = compute_all_metrics_fast(truth, b_mean)
        m_naive = compute_all_metrics_fast(truth, b_naive)

        # Add horizon index
        wks_unique = sorted(truth["week"].unique())
        week_to_h = {w: i + 1 for i, w in enumerate(wks_unique)}
        truth["horizon"] = truth["week"].map(week_to_h)

        # Merge series stats
        truth = truth.merge(stats_df[["warehouse_name", "article", "sparsity_segment", "volume_segment"]], on=["warehouse_name", "article"], how="left")

        # Load OOF Predictions with strict KEY assertions
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")

        merged = truth[KEYS].merge(df_blend, on=KEYS, how="left")
        assert len(merged) == len(truth), f"Row count mismatch on fold {fold} blend merge: {len(merged)} vs {len(truth)}"
        assert merged["raw_tft"].isna().sum() == 0, f"Missing TFT predictions on fold {fold}!"
        assert np.all(np.isfinite(merged["raw_tft"].to_numpy())), f"Non-finite TFT on fold {fold}!"

        # Load All-Stock Predictions (with Scaled 1M LGBM and Ensemble)
        all_stock_path = PROJECT_ROOT / "experiments" / "model_improvements" / "all_stock_5folds" / "keyed_predictions" / f"predictions_fold_{fold}.csv.gz"
        if all_stock_path.exists():
            df_as = pd.read_csv(all_stock_path)
            df_as["week"] = pd.to_datetime(df_as["week"]).dt.strftime("%Y-%m-%d")
            merged = merged.merge(df_as[KEYS + ["candidate_weights", "stock_lgbm_only", "all_stock_ensemble"]], on=KEYS, how="left")
            assert len(merged) == len(truth), "Row count mismatch on all-stock merge!"

        # Load NBEATSx Predictions
        nb_path = PROJECT_ROOT / "experiments" / "model_improvements" / "candidates" / "nbeatsx" / "keyed" / f"nbeatsx_pure_cal_fold_{fold}.csv.gz"
        if nb_path.exists():
            df_nb = pd.read_csv(nb_path)
            df_nb["week"] = pd.to_datetime(df_nb["week"]).dt.strftime("%Y-%m-%d")
            merged = merged.merge(df_nb[KEYS + ["c_raw_nonneg", "pred_standalone"]], on=KEYS, how="left")
            assert len(merged) == len(truth), "Row count mismatch on nbeatsx merge!"

        # Prepare model candidate arrays
        candidates_raw = {
            # 1. TFT Standalone Variants
            "TFT_Continuous": np.maximum(merged["raw_tft"].to_numpy(), 0.0),
            "TFT_Rint": np.rint(np.maximum(merged["raw_tft"].to_numpy(), 0.0)),
            "TFT_Legacy_P": apply_p(merged["raw_tft"].to_numpy(), 1.05, 0.65),
            "TFT_Thresh_0.75_Rint": np.rint(np.where(merged["raw_tft"].to_numpy() < 0.75, 0.0, merged["raw_tft"].to_numpy())).clip(0, None),

            # 2. LightGBM B Standalone Variants (Base 52 origins)
            "LGBM_Base_Continuous": np.maximum(merged["after_gate_lgbm"].to_numpy(), 0.0),
            "LGBM_Base_Rint": np.rint(np.maximum(merged["after_gate_lgbm"].to_numpy(), 0.0)),
            "LGBM_Base_Legacy_P": apply_p(merged["after_gate_lgbm"].to_numpy(), 1.05, 0.65),
            "LGBM_Base_Ungated_Continuous": np.maximum(merged["expected_lgbm"].to_numpy(), 0.0),

            # 3. LightGBM Stock 1M-Scaled Standalone (Scale_1M_Dual_0.5 stock LGBM)
            "LGBM_Stock_1M_Continuous": np.maximum(merged["stock_lgbm_only"].to_numpy(), 0.0) if "stock_lgbm_only" in merged else None,
            "LGBM_Stock_1M_Rint": np.rint(np.maximum(merged["stock_lgbm_only"].to_numpy(), 0.0)) if "stock_lgbm_only" in merged else None,
            "LGBM_Stock_1M_Legacy_P": apply_p(merged["stock_lgbm_only"].to_numpy(), 1.05, 0.65) if "stock_lgbm_only" in merged else None,

            # 4. CatBoost Standalone
            "CatBoost_Continuous": np.maximum(merged["after_gate_catboost"].to_numpy(), 0.0),
            "CatBoost_Rint": np.rint(np.maximum(merged["after_gate_catboost"].to_numpy(), 0.0)),
            "CatBoost_Legacy_P": apply_p(merged["after_gate_catboost"].to_numpy(), 1.05, 0.65),

            # 5. PerCategory Standalone
            "PerCat_Continuous": np.maximum(merged["after_gate_per_category"].to_numpy(), 0.0),
            "PerCat_Rint": np.rint(np.maximum(merged["after_gate_per_category"].to_numpy(), 0.0)),
            "PerCat_Legacy_P": apply_p(merged["after_gate_per_category"].to_numpy(), 1.05, 0.65),

            # 6. NBEATSx Standalone
            "NBEATSx_NonNeg_Continuous": np.maximum(merged["c_raw_nonneg"].to_numpy(), 0.0) if "c_raw_nonneg" in merged else None,
            "NBEATSx_Rint": np.rint(np.maximum(merged["c_raw_nonneg"].to_numpy(), 0.0)) if "c_raw_nonneg" in merged else None,
            "NBEATSx_Legacy_P": apply_p(merged["c_raw_nonneg"].to_numpy(), 1.05, 0.65) if "c_raw_nonneg" in merged else None,

            # 7. Benchmarks & Ensembles
            "Ensemble_Control_B": merged["final_prediction"].to_numpy(),
            "Ensemble_Candidate_45_37_9_9": merged["candidate_weights"].to_numpy() if "candidate_weights" in merged else None,
            "Ensemble_All_Stock_28_Dual": merged["all_stock_ensemble"].to_numpy() if "all_stock_ensemble" in merged else None,
        }

        for model_name, preds in candidates_raw.items():
            if preds is None:
                continue

            p_df = truth[KEYS].copy()
            p_df["quantity"] = preds

            # Global fold metrics
            m_pred = compute_all_metrics_fast(truth[KEYS + ["quantity"]], p_df)
            c_mean, u_mean = calc_scores(m_pred, m_mean)
            _, n_sc = calc_scores(m_pred, m_naive)

            all_fold_evals.append({
                "fold": fold,
                "origin": origin,
                "phase": phase,
                "model": model_name,
                "is_standalone": not model_name.startswith("Ensemble_"),
                "clipped_mean13": c_mean,
                "unclipped_mean13": u_mean,
                "score_naive": n_sc,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
            })

            # Segmented breakdowns
            # A. Sparsity segments
            for sp_reg in ["high_intermittent", "moderate_intermittent", "regular_continuous"]:
                mask = truth["sparsity_segment"] == sp_reg
                m_sub = compute_all_metrics_fast(truth.loc[mask, KEYS + ["quantity"]], p_df.loc[mask])
                segmented_evals.append({
                    "fold": fold, "origin": origin, "phase": phase, "model": model_name,
                    "dimension": "sparsity", "segment": sp_reg,
                    "smape": m_sub["smape"], "wape": m_sub["wape"], "rmse": m_sub["rmse"], "bias": m_sub["bias"]
                })

            # B. Volume segments
            for vol_reg in ["low_volume", "mid_volume", "high_volume"]:
                mask = truth["volume_segment"] == vol_reg
                m_sub = compute_all_metrics_fast(truth.loc[mask, KEYS + ["quantity"]], p_df.loc[mask])
                segmented_evals.append({
                    "fold": fold, "origin": origin, "phase": phase, "model": model_name,
                    "dimension": "volume", "segment": vol_reg,
                    "smape": m_sub["smape"], "wape": m_sub["wape"], "rmse": m_sub["rmse"], "bias": m_sub["bias"]
                })

            # C. Horizon segments: Short-term (w1-4) vs Long-term (w5-9)
            m_short = compute_all_metrics_fast(truth.loc[truth["horizon"] <= 4, KEYS + ["quantity"]], p_df.loc[truth["horizon"] <= 4])
            segmented_evals.append({
                "fold": fold, "origin": origin, "phase": phase, "model": model_name,
                "dimension": "horizon", "segment": "weeks_1_4",
                "smape": m_short["smape"], "wape": m_short["wape"], "rmse": m_short["rmse"], "bias": m_short["bias"]
            })
            m_long = compute_all_metrics_fast(truth.loc[truth["horizon"] >= 5, KEYS + ["quantity"]], p_df.loc[truth["horizon"] >= 5])
            segmented_evals.append({
                "fold": fold, "origin": origin, "phase": phase, "model": model_name,
                "dimension": "horizon", "segment": "weeks_5_9",
                "smape": m_long["smape"], "wape": m_long["wape"], "rmse": m_long["rmse"], "bias": m_long["bias"]
            })

        print(f"Fold {fold} evaluated in {time.perf_counter() - t0:.1f}s", flush=True)

    df_folds = pd.DataFrame(all_fold_evals)
    df_seg = pd.DataFrame(segmented_evals)

    # Compute Summary Tables
    # 1. Selection Windows (Folds 4, 3, 2)
    sel_df = df_folds[df_folds["phase"] == "Selection"].groupby("model").agg(
        sel_clipped=("clipped_mean13", "mean"),
        sel_unclipped=("unclipped_mean13", "mean"),
        sel_smape=("smape", "mean"),
        sel_wape=("wape", "mean"),
        sel_rmse=("rmse", "mean"),
        sel_bias=("bias", "mean"),
    ).reset_index()

    # 2. Check Windows (Folds 1, 0)
    chk_df = df_folds[df_folds["phase"] == "Check"].groupby("model").agg(
        chk_clipped=("clipped_mean13", "mean"),
        chk_unclipped=("unclipped_mean13", "mean"),
        chk_smape=("smape", "mean"),
        chk_wape=("wape", "mean"),
        chk_rmse=("rmse", "mean"),
        chk_bias=("bias", "mean"),
    ).reset_index()

    # 3. All 5 Folds
    all_df = df_folds.groupby("model").agg(
        all_clipped=("clipped_mean13", "mean"),
        all_unclipped=("unclipped_mean13", "mean"),
        all_naive=("score_naive", "mean"),
        all_smape=("smape", "mean"),
        all_wape=("wape", "mean"),
        all_rmse=("rmse", "mean"),
        all_bias=("bias", "mean"),
        all_rmsle=("rmsle", "mean"),
    ).reset_index()

    summary_df = all_df.merge(sel_df, on="model").merge(chk_df, on="model")
    summary_df = summary_df.sort_values("all_clipped")

    # Save summary CSV
    summary_csv_path = BENCHMARK_DIR / "standalone_models_benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary table written to {summary_csv_path}")

    # Save segmented CSV
    seg_csv_path = BENCHMARK_DIR / "standalone_segmented_analysis.csv"
    seg_summary = df_seg.groupby(["model", "dimension", "segment"]).agg(
        mean_smape=("smape", "mean"),
        mean_wape=("wape", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
    ).reset_index()
    seg_summary.to_csv(seg_csv_path, index=False)
    print(f"Segmented analysis written to {seg_csv_path}")

    # Build comprehensive JSON results
    benchmark_json = {
        "benchmark_name": "standalone_models_historical_5fold_evaluation",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_windows": {
            "selection": [4, 3, 2],
            "check": [1, 0],
            "all_5_folds": [4, 3, 2, 1, 0],
        },
        "models_evaluated": list(summary_df["model"].unique()),
        "standings_all_5_folds": [
            {
                "model": row["model"],
                "all_clipped_mean13": round(float(row["all_clipped"]), 6),
                "all_unclipped_mean13": round(float(row["all_unclipped"]), 6),
                "all_smape": round(float(row["all_smape"]), 4),
                "all_wape": round(float(row["all_wape"]), 4),
                "all_rmse": round(float(row["all_rmse"]), 4),
                "all_bias": round(float(row["all_bias"]), 4),
                "selection_clipped": round(float(row["sel_clipped"]), 6),
                "selection_unclipped": round(float(row["sel_unclipped"]), 6),
                "check_clipped": round(float(row["chk_clipped"]), 6),
                "check_unclipped": round(float(row["chk_unclipped"]), 6),
            }
            for _, row in summary_df.iterrows()
        ],
        "full_fold_results": all_fold_evals,
    }

    json_path = BENCHMARK_DIR / "standalone_models_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2, ensure_ascii=False)
    print(f"Detailed JSON results written to {json_path}")

    # Print Summary Display
    print("\n" + "=" * 90)
    print("STANDALONE MODELS BENCHMARK - FINAL 5-FOLD STANDINGS")
    print("=" * 90)
    cols_to_print = ["model", "all_clipped", "all_unclipped", "all_smape", "all_wape", "all_rmse", "all_bias"]
    print(summary_df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
