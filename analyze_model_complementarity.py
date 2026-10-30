#!/usr/bin/env python3
"""
analyze_model_complementarity.py

Reproducible diagnostic script to analyze error complementarity among TFT, LightGBM B,
CatBoost, and Per-Category models across all 5 evaluation folds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from score import normalized_score
from run_weekly_feature_experiments import KEYS, HORIZON
from train_tft_weekly import baseline_mean13

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "complementarity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_ORIGINS = {
    0: 125,
    1: 116,
    2: 107,
    3: 98,
    4: 89,
}

def apply_postprocess(x: np.ndarray | pd.Series) -> np.ndarray:
    """Standard postprocessing P(x)."""
    z = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    return np.maximum(np.rint(np.where(z < 0.65, 0.0, z)), 0.0)

def compute_fold_metrics(df_truth: pd.DataFrame, df_pred: pd.DataFrame, df_baseline: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(df_truth, df_pred)
    bm = compute_all_metrics_fast(df_truth, df_baseline)
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
    print("=" * 80)
    print("ANALYZING MODEL COMPLEMENTARITY ACROSS 5 EXPANDING FOLDS")
    print("=" * 80)

    # 1. Load weekly dataset to compute identical baseline_mean13 for all folds
    print("Loading weekly dataset for mean13 baselines...")
    weekly_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(weekly_path, parse_dates=["week"])
    weeks = sorted(data["week"].unique())

    fold_truth = {}
    fold_baseline = {}
    fold_dfs = {}

    for fold, origin in FOLD_ORIGINS.items():
        val_weeks = weeks[origin : origin + HORIZON]
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
        fold_baseline[fold] = b_df

    correlation_records = []
    model_metric_records = []
    segment_records = []

    # Iterate over folds
    for fold in range(5):
        print(f"\nProcessing Fold {fold} (origin {FOLD_ORIGINS[fold]})...")
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        meta_path = PROJECT_ROOT / "experiments" / "stacking_honest" / "fold_meta_features" / f"fold_{fold}_meta.csv.gz"

        df_blend = pd.read_csv(blend_path)
        df_meta = pd.read_csv(meta_path)

        # Standardize week string
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        df_meta["week"] = pd.to_datetime(df_meta["week"]).dt.strftime("%Y-%m-%d")

        # Merge strictly 1-to-1
        merged = df_blend.merge(df_meta, on=KEYS, how="inner", suffixes=("", "_meta"), validate="one_to_one")
        assert len(merged) == 79101, f"Expected 79101 rows on fold {fold}, got {len(merged)}"
        assert not merged["target"].isna().any(), f"NaN in target on fold {fold}"

        # Build truth df
        truth_df = merged[KEYS + ["target"]].rename(columns={"target": "quantity"}).copy()
        truth_df["week"] = truth_df["week"].astype(str)
        fold_truth[fold] = truth_df

        y_true = merged["target"].values.astype(float)
        mean13 = merged["quantity_mean_13"].values.astype(float)
        safe_mean13 = np.maximum(mean13, 0.5)

        # Components before final alpha/rounding:
        # after_comp_tft, after_comp_lgbm, after_comp_catboost, after_comp_per_category
        tft_pred = np.maximum(merged["after_comp_tft"].values.astype(float), 0.0)
        lgbm_pred = np.maximum(merged["after_comp_lgbm"].values.astype(float), 0.0)
        cb_pred = np.maximum(merged["after_comp_catboost"].values.astype(float), 0.0)
        pc_pred = np.maximum(merged["after_comp_per_category"].values.astype(float), 0.0)
        raw_blend = merged["raw_blend"].values.astype(float)

        e_tft = tft_pred - y_true
        e_lgbm = lgbm_pred - y_true
        e_cb = cb_pred - y_true
        e_pc = pc_pred - y_true

        e_tft_norm = e_tft / safe_mean13
        e_lgbm_norm = e_lgbm / safe_mean13
        e_cb_norm = e_cb / safe_mean13
        e_pc_norm = e_pc / safe_mean13

        pos_mask = y_true > 0

        # Compute Pearson correlations
        def get_corr(x, y):
            c, _ = pearsonr(x, y)
            return float(c)

        corr_row = {
            "fold": fold,
            "corr_all_lgbm": get_corr(e_tft, e_lgbm),
            "corr_pos_lgbm": get_corr(e_tft[pos_mask], e_lgbm[pos_mask]),
            "corr_norm_all_lgbm": get_corr(e_tft_norm, e_lgbm_norm),
            "corr_norm_pos_lgbm": get_corr(e_tft_norm[pos_mask], e_lgbm_norm[pos_mask]),

            "corr_all_cb": get_corr(e_tft, e_cb),
            "corr_pos_cb": get_corr(e_tft[pos_mask], e_cb[pos_mask]),
            "corr_norm_all_cb": get_corr(e_tft_norm, e_cb_norm),
            "corr_norm_pos_cb": get_corr(e_tft_norm[pos_mask], e_cb_norm[pos_mask]),

            "corr_all_pc": get_corr(e_tft, e_pc),
            "corr_pos_pc": get_corr(e_tft[pos_mask], e_pc[pos_mask]),
            "corr_norm_all_pc": get_corr(e_tft_norm, e_pc_norm),
            "corr_norm_pos_pc": get_corr(e_tft_norm[pos_mask], e_pc_norm[pos_mask]),
        }
        correlation_records.append(corr_row)

        print(f"  Error corr with TFT: LGBM all={corr_row['corr_all_lgbm']:.4f}, pos={corr_row['corr_pos_lgbm']:.4f} | "
              f"CB all={corr_row['corr_all_cb']:.4f}, pos={corr_row['corr_pos_cb']:.4f} | "
              f"PC all={corr_row['corr_all_pc']:.4f}, pos={corr_row['corr_pos_pc']:.4f}")

        # Evaluate candidate models and blends
        candidates = {
            "Control_Blend_B": apply_postprocess(raw_blend),
            "Standalone_TFT": apply_postprocess(tft_pred),
            "Standalone_LightGBM": apply_postprocess(lgbm_pred),
            "Standalone_CatBoost": apply_postprocess(cb_pred),
            "Standalone_PerCategory": apply_postprocess(pc_pred),
            "Blend_0.90E_0.10LGBM": apply_postprocess(0.90 * raw_blend + 0.10 * lgbm_pred),
            "Blend_0.75E_0.25LGBM": apply_postprocess(0.75 * raw_blend + 0.25 * lgbm_pred),
            "Blend_0.90T_0.10LGBM": apply_postprocess(0.90 * tft_pred + 0.10 * lgbm_pred),
            "Blend_0.75T_0.25LGBM": apply_postprocess(0.75 * tft_pred + 0.25 * lgbm_pred),
            "Blend_0.90E_0.10CB": apply_postprocess(0.90 * raw_blend + 0.10 * cb_pred),
            "Blend_0.75E_0.25CB": apply_postprocess(0.75 * raw_blend + 0.25 * cb_pred),
            "Blend_0.90T_0.10CB": apply_postprocess(0.90 * tft_pred + 0.10 * cb_pred),
            "Blend_0.75T_0.25CB": apply_postprocess(0.75 * tft_pred + 0.25 * cb_pred),
        }

        # Oracle unreachable diagnostic
        best_of_all = np.where(np.abs(tft_pred - y_true) <= np.abs(lgbm_pred - y_true), tft_pred, lgbm_pred)
        best_of_all = np.where(np.abs(best_of_all - y_true) <= np.abs(cb_pred - y_true), best_of_all, cb_pred)
        candidates["Oracle_TFT_LGBM_CB_diagnostic"] = apply_postprocess(best_of_all)

        b_df = fold_baseline[fold]
        for name, pred_arr in candidates.items():
            pred_df = merged[KEYS].copy()
            pred_df["quantity"] = pred_arr
            m = compute_fold_metrics(truth_df, pred_df, b_df)
            model_metric_records.append({
                "fold": fold,
                "model": name,
                **m,
            })

        # Segment diagnostics (strictly using pre-cutoff historical features)
        merged["tft_post"] = candidates["Standalone_TFT"]
        merged["lgbm_post"] = candidates["Standalone_LightGBM"]
        merged["blend_post"] = candidates["Control_Blend_B"]

        # Define segments
        # 1. Regularity
        rare_mask = merged["quantity_nonzero_rate_13"] <= 0.10
        reg_low_mask = (merged["quantity_nonzero_rate_13"] > 0.10) & (merged["quantity_mean_13"] < 5.0)
        reg_high_mask = (merged["quantity_nonzero_rate_13"] > 0.10) & (merged["quantity_mean_13"] >= 5.0)

        # 2. Historical Trend
        growth_mask = (merged["quantity_mean_13"] > 0.1) & (merged["quantity_mean_4"] > 1.15 * merged["quantity_mean_13"])
        decline_mask = (merged["quantity_mean_13"] > 0.1) & (merged["quantity_mean_4"] < 0.85 * merged["quantity_mean_13"])
        stable_mask = ~(growth_mask | decline_mask)

        # 3. Horizon
        horizons = sorted(merged["horizon"].unique())

        # 4. Error classification
        # False sales on zero target:
        zero_target_mask = (y_true == 0)
        # Missed sales:
        nonzero_target_mask = (y_true > 0)

        def eval_subsegment(mask, seg_type, seg_name):
            sub_truth = y_true[mask]
            sub_tft = merged["tft_post"].values[mask]
            sub_lgbm = merged["lgbm_post"].values[mask]
            sub_blend = merged["blend_post"].values[mask]
            count = int(mask.sum())
            if count == 0:
                return
            fact_sum = float(sub_truth.sum())
            
            def get_sub_wape(y_t, y_p):
                denom = np.sum(np.abs(y_t))
                return float(np.sum(np.abs(y_t - y_p)) / denom * 100.0) if denom > 0 else 0.0

            def get_sub_rmse(y_t, y_p):
                return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

            segment_records.append({
                "fold": fold,
                "segment_type": seg_type,
                "segment_name": seg_name,
                "row_count": count,
                "fact_sum": fact_sum,
                "wape_tft": get_sub_wape(sub_truth, sub_tft),
                "wape_lgbm": get_sub_wape(sub_truth, sub_lgbm),
                "wape_blend": get_sub_wape(sub_truth, sub_blend),
                "rmse_tft": get_sub_rmse(sub_truth, sub_tft),
                "rmse_lgbm": get_sub_rmse(sub_truth, sub_lgbm),
                "rmse_blend": get_sub_rmse(sub_truth, sub_blend),
                "false_sales_tft": int((zero_target_mask[mask] & (sub_tft > 0)).sum()),
                "false_sales_lgbm": int((zero_target_mask[mask] & (sub_lgbm > 0)).sum()),
                "false_sales_blend": int((zero_target_mask[mask] & (sub_blend > 0)).sum()),
                "missed_sales_tft": int((nonzero_target_mask[mask] & (sub_tft == 0)).sum()),
                "missed_sales_lgbm": int((nonzero_target_mask[mask] & (sub_lgbm == 0)).sum()),
                "missed_sales_blend": int((nonzero_target_mask[mask] & (sub_blend == 0)).sum()),
                "large_under_tft": int(((sub_truth - sub_tft) >= 5).sum()),
                "large_under_lgbm": int(((sub_truth - sub_lgbm) >= 5).sum()),
                "large_under_blend": int(((sub_truth - sub_blend) >= 5).sum()),
                "large_over_tft": int(((sub_tft - sub_truth) >= 5).sum()),
                "large_over_lgbm": int(((sub_lgbm - sub_truth) >= 5).sum()),
                "large_over_blend": int(((sub_blend - sub_truth) >= 5).sum()),
            })

        eval_subsegment(rare_mask, "regularity", "Rare (<=10%)")
        eval_subsegment(reg_low_mask, "regularity", "Regular Low-Volume")
        eval_subsegment(reg_high_mask, "regularity", "Regular High-Volume")

        eval_subsegment(growth_mask, "trend", "Growth (mean4 > 1.15*mean13)")
        eval_subsegment(decline_mask, "trend", "Decline (mean4 < 0.85*mean13)")
        eval_subsegment(stable_mask, "trend", "Stable")

        for h in horizons:
            eval_subsegment(merged["horizon"] == h, "horizon", f"Horizon_{h}")

        for cat in sorted(merged["category_lvl_1"].unique()):
            eval_subsegment(merged["category_lvl_1"] == cat, "category", f"Cat_{cat}")

        fold_dfs[fold] = merged

    # Save summary tables
    df_corr = pd.DataFrame(correlation_records)
    df_metrics = pd.DataFrame(model_metric_records)
    df_segs = pd.DataFrame(segment_records)

    df_corr.to_csv(OUTPUT_DIR / "error_correlations_by_fold.csv", index=False)
    df_metrics.to_csv(OUTPUT_DIR / "model_metrics_by_fold.csv", index=False)
    df_segs.to_csv(OUTPUT_DIR / "segment_metrics_by_fold.csv", index=False)

    # Print mean correlations
    print("\n" + "=" * 80)
    print("MEAN ERROR CORRELATIONS WITH TFT ACROSS ALL 5 FOLDS:")
    print("=" * 80)
    mean_corr = df_corr.mean(numeric_only=True)
    print(f"LightGBM B:     All rows = {mean_corr['corr_all_lgbm']:.4f}, Target > 0 = {mean_corr['corr_pos_lgbm']:.4f}, Norm All = {mean_corr['corr_norm_all_lgbm']:.4f}, Norm Pos = {mean_corr['corr_norm_pos_lgbm']:.4f}")
    print(f"CatBoost:       All rows = {mean_corr['corr_all_cb']:.4f}, Target > 0 = {mean_corr['corr_pos_cb']:.4f}, Norm All = {mean_corr['corr_norm_all_cb']:.4f}, Norm Pos = {mean_corr['corr_norm_pos_cb']:.4f}")
    print(f"Per-Category:   All rows = {mean_corr['corr_all_pc']:.4f}, Target > 0 = {mean_corr['corr_pos_pc']:.4f}, Norm All = {mean_corr['corr_norm_all_pc']:.4f}, Norm Pos = {mean_corr['corr_norm_pos_pc']:.4f}")

    # Generate plots
    print("\nGenerating fact vs prediction plots...")
    np.random.seed(42)
    # Use Fold 0 for visualization
    f0_df = fold_dfs[0].copy()
    
    # 1. Random sample of 4 series
    unique_series = f0_df[["warehouse_name", "article"]].drop_duplicates().sample(4, random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (_, s_row) in enumerate(unique_series.iterrows()):
        wh, art = s_row["warehouse_name"], s_row["article"]
        sub = f0_df[(f0_df["warehouse_name"] == wh) & (f0_df["article"] == art)].sort_values("week")
        ax = axes[idx]
        w_labels = [w[-5:] for w in sub["week"]]
        ax.plot(w_labels, sub["target"], "o-", label="Ground Truth", color="black", linewidth=2)
        ax.plot(w_labels, sub["tft_post"], "s--", label="TFT", color="blue", alpha=0.7)
        ax.plot(w_labels, sub["lgbm_post"], "^--", label="LightGBM B", color="green", alpha=0.7)
        ax.plot(w_labels, sub["blend_post"], "d-", label="Control Blend B", color="red", linewidth=1.5)
        ax.set_title(f"Random Series: {wh} / {art}\n(mean13={sub['quantity_mean_13'].iloc[0]:.1f})", fontsize=10)
        ax.set_ylabel("Quantity")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "random_sample_series_comparison.png", dpi=150)
    plt.close()

    # 2. Large error series (where TFT had large error)
    # Find series with largest sum of |tft - target|
    f0_df["tft_abs_err"] = np.abs(f0_df["tft_post"] - f0_df["target"])
    top_err_series = (
        f0_df.groupby(["warehouse_name", "article"])["tft_abs_err"]
        .sum()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (wh, art) in enumerate(top_err_series):
        sub = f0_df[(f0_df["warehouse_name"] == wh) & (f0_df["article"] == art)].sort_values("week")
        ax = axes[idx]
        w_labels = [w[-5:] for w in sub["week"]]
        ax.plot(w_labels, sub["target"], "o-", label="Ground Truth", color="black", linewidth=2)
        ax.plot(w_labels, sub["tft_post"], "s--", label="TFT", color="blue", alpha=0.7)
        ax.plot(w_labels, sub["lgbm_post"], "^--", label="LightGBM B", color="green", alpha=0.7)
        ax.plot(w_labels, sub["blend_post"], "d-", label="Control Blend B", color="red", linewidth=1.5)
        ax.set_title(f"Large Error Case: {wh} / {art}\n(TFT Err Sum={sub['tft_abs_err'].sum():.1f}, mean13={sub['quantity_mean_13'].iloc[0]:.1f})", fontsize=10)
        ax.set_ylabel("Quantity")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "large_error_cases_comparison.png", dpi=150)
    plt.close()

    print("Plots saved successfully.")
    print("Complementarity analysis completed.")

if __name__ == "__main__":
    main()
