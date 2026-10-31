#!/usr/bin/env python3
"""
test_postprocessing_innovations.py

Comprehensive ablation of:
1. Hybrid Rounding (rint vs float vs hybrid thresholds)
2. Adaptive Thresholding by history nonzero_rate_13
3. Horizon-Dependent Blend Weights (dynamic h=1..9)

Evaluated on Selection Folds [4, 3, 2] and Check Folds [1, 0]
against baseline_mean13 and baseline_naive_last.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    compute_all_metrics_fast,
    columns_for_experiment,
)
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    train_predict_stock_lgbm,
    train_predict_stock_catboost,
    train_predict_stock_per_category,
)

METRIC_WEIGHTS = {
    "smape": 0.50,
    "wape": 0.20,
    "rmse": 0.15,
    "bias": 0.10,
    "rmsle": 0.05,
}

def calc_scores(m_pred, m_base):
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped

ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("TESTING POST-PROCESSING INNOVATIONS ON SELECTION FOLDS [4, 3, 2] & CHECK FOLDS [1, 0]")
    print("=" * 100)

    # Dictionary to collect results per configuration across folds
    configs = [
        "Base_Uniform_Rint",               # Current champion: threshold 0.65, full rint
        "Rounding_Pure_Float",             # Pure float: threshold 0.65, no rint
        "Rounding_Hybrid_1_35",            # z in [0.65, 1.35] -> 1.0, z > 1.35 -> float
        "Rounding_Hybrid_1_50",            # z in [0.65, 1.50] -> 1.0, z > 1.50 -> float
        "Rounding_Hybrid_HalfInt",         # z in [0.65, 1.35] -> 1.0, z > 1.35 -> round to 0.5
        "Adaptive_Thresh_v1",              # active(>=0.5): 0.58, mid(0.2..0.5): 0.65, low(<0.2): 0.72
        "Adaptive_Thresh_v2",              # active(>=0.4): 0.60, low(<0.4): 0.68
        "Adaptive_Thresh_v3",              # active(>=0.5): 0.55, mid(0.2..0.5): 0.65, low(<0.2): 0.75
        "Horizon_Dynamic_Weights",         # Near h<=3: boostings 0.65, Far h>=7: TFT 0.55
        "Horizon_Dyn_Adaptive_Thresh",     # Both Horizon Dynamic + Adaptive Thresh v1
    ]

    results = {c: {"sel_clipped": [], "sel_unclipped": [], "sel_naive": [], "sel_smape": [], "sel_bias": [], "sel_rmse": [],
                   "chk_clipped": [], "chk_unclipped": [], "chk_naive": [], "chk_smape": [], "chk_bias": [], "chk_rmse": []}
               for c in configs}

    for fold, origin in ALL_FOLDS.items():
        phase = "Selection" if fold in [4, 3, 2] else "Check"
        print(f"\nComputing Fold {fold} (Origin {origin}, {phase})...")

        # Baselines
        history13 = np.column_stack([matrices.quantity[origin - lag, :] for lag in range(1, 14)])
        mean13 = history13.mean(axis=1)
        nonzero_rate_13 = (history13 > 0).mean(axis=1)

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

        # Load raw components
        df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
        df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

        latest_train = origin - HORIZON
        lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
        cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

        X_train_lgb, y_train_lgb, X_val, val_df, val_pos = build_feature_matrices(matrices, origin, lgbm_origins, mode="29")
        X_train_cb, y_train_cb, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, mode="29")

        val_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]
        val_df["article"] = val_df["article"].astype(df_f["article"].dtype)

        feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

        pred_lgb = train_predict_stock_lgbm(X_train_lgb, y_train_lgb, X_val, feature_cols)
        pred_cb = train_predict_stock_catboost(X_train_cb, y_train_cb, X_val, feature_cols)
        pred_pcat = train_predict_stock_per_category(X_train_lgb, y_train_lgb, X_val, feature_cols + ["category_lvl_1"])

        val_df["stock_lgbm_gate"] = pred_lgb
        val_df["stock_cb_gate"] = pred_cb
        val_df["stock_pcat_gate"] = pred_pcat

        merged = df_f.merge(val_df[["week", "warehouse_name", "article", "stock_lgbm_gate", "stock_cb_gate", "stock_pcat_gate"]], on=KEYS, how="left")

        q_tft = merged["raw_tft"].to_numpy(dtype=np.float64)
        q_lgb = merged["stock_lgbm_gate"].to_numpy(dtype=np.float64)
        q_cb = merged["stock_cb_gate"].to_numpy(dtype=np.float64)
        q_pcat = merged["stock_pcat_gate"].to_numpy(dtype=np.float64)

        horizon_arr = np.tile(np.arange(1, HORIZON + 1), matrices.n_pairs)
        # Note: val_wks_arr is ordered week-by-week: week 0 for all pairs, week 1 for all pairs...
        # Let's verify horizon order:
        h_idx = np.repeat(np.arange(1, HORIZON + 1), matrices.n_pairs)
        nz_rep = np.tile(nonzero_rate_13, HORIZON)

        # Baseline raw blend (45/37/9/9)
        raw_blend_base = 0.45 * q_tft + 0.37 * q_lgb + 0.09 * q_cb + 0.09 * q_pcat
        scaled_base = raw_blend_base * 1.05

        # Horizon Dynamic raw blend:
        # Near (h <= 3): 0.35 TFT, 0.43 LGBM, 0.11 CB, 0.11 PerCat
        # Mid (4 <= h <= 6): 0.45 TFT, 0.37 LGBM, 0.09 CB, 0.09 PerCat
        # Far (h >= 7): 0.55 TFT, 0.29 LGBM, 0.08 CB, 0.08 PerCat
        w_tft_h = np.where(h_idx <= 3, 0.35, np.where(h_idx <= 6, 0.45, 0.55))
        w_lgb_h = np.where(h_idx <= 3, 0.43, np.where(h_idx <= 6, 0.37, 0.29))
        w_cb_h = np.where(h_idx <= 3, 0.11, np.where(h_idx <= 6, 0.09, 0.08))
        w_pcat_h = np.where(h_idx <= 3, 0.11, np.where(h_idx <= 6, 0.09, 0.08))

        raw_blend_dyn = w_tft_h * q_tft + w_lgb_h * q_lgb + w_cb_h * q_cb + w_pcat_h * q_pcat
        scaled_dyn = raw_blend_dyn * 1.05

        # Evaluate each configuration
        for cfg in configs:
            if "Horizon" in cfg:
                z = scaled_dyn.copy()
            else:
                z = scaled_base.copy()

            if cfg == "Base_Uniform_Rint":
                g = np.where(z < 0.65, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            elif cfg == "Rounding_Pure_Float":
                pred_final = np.where(z < 0.65, 0.0, z).clip(0, None)

            elif cfg == "Rounding_Hybrid_1_35":
                g = np.where(z < 0.65, 0.0, z)
                pred_final = np.where(g == 0, 0.0, np.where(g <= 1.35, 1.0, g)).clip(0, None)

            elif cfg == "Rounding_Hybrid_1_50":
                g = np.where(z < 0.65, 0.0, z)
                pred_final = np.where(g == 0, 0.0, np.where(g <= 1.50, 1.0, g)).clip(0, None)

            elif cfg == "Rounding_Hybrid_HalfInt":
                g = np.where(z < 0.65, 0.0, z)
                pred_final = np.where(g == 0, 0.0, np.where(g <= 1.35, 1.0, np.round(g * 2) / 2)).clip(0, None)

            elif cfg == "Adaptive_Thresh_v1":
                thresh = np.where(nz_rep >= 0.5, 0.58, np.where(nz_rep >= 0.2, 0.65, 0.72))
                g = np.where(z < thresh, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            elif cfg == "Adaptive_Thresh_v2":
                thresh = np.where(nz_rep >= 0.4, 0.60, 0.68)
                g = np.where(z < thresh, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            elif cfg == "Adaptive_Thresh_v3":
                thresh = np.where(nz_rep >= 0.5, 0.55, np.where(nz_rep >= 0.2, 0.65, 0.75))
                g = np.where(z < thresh, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            elif cfg == "Horizon_Dynamic_Weights":
                g = np.where(z < 0.65, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            elif cfg == "Horizon_Dyn_Adaptive_Thresh":
                thresh = np.where(nz_rep >= 0.5, 0.58, np.where(nz_rep >= 0.2, 0.65, 0.72))
                g = np.where(z < thresh, 0.0, z)
                pred_final = np.rint(g).clip(0, None)

            pred_df = pd.DataFrame({
                "week": truth["week"],
                "warehouse_name": truth["warehouse_name"],
                "article": truth["article"],
                "quantity": pred_final,
            })

            m_pred = compute_all_metrics_fast(truth, pred_df)
            clip_sc, unclip_sc = calc_scores(m_pred, m_mean)
            _, naive_sc = calc_scores(m_pred, m_naive)

            p_key = "sel_" if phase == "Selection" else "chk_"
            results[cfg][p_key + "clipped"].append(clip_sc)
            results[cfg][p_key + "unclipped"].append(unclip_sc)
            results[cfg][p_key + "naive"].append(naive_sc)
            results[cfg][p_key + "smape"].append(m_pred["smape"])
            results[cfg][p_key + "bias"].append(m_pred["bias"])
            results[cfg][p_key + "rmse"].append(m_pred["rmse"])

    print("\n" + "=" * 120)
    print("SELECTION FOLDS [4, 3, 2] ABLATION SUMMARY")
    print("=" * 120)
    fmt = "{:<28} | {:<12} | {:<14} | {:<14} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("Configuration", "Sel Clipped", "Sel Unclipped", "Score vs Naive", "SMAPE (%)", "Bias (%)", "RMSE"))
    print("-" * 120)

    for cfg in configs:
        sc_c = np.mean(results[cfg]["sel_clipped"])
        sc_u = np.mean(results[cfg]["sel_unclipped"])
        sc_n = np.mean(results[cfg]["sel_naive"])
        sm = np.mean(results[cfg]["sel_smape"])
        bi = np.mean(results[cfg]["sel_bias"])
        rm = np.mean(results[cfg]["sel_rmse"])
        print(fmt.format(cfg, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.2f}%", f"{bi:.2f}%", f"{rm:.4f}"))

    print("\n" + "=" * 120)
    print("CHECK FOLDS [1, 0] VALIDATION SUMMARY")
    print("=" * 120)
    print(fmt.format("Configuration", "Chk Clipped", "Chk Unclipped", "Score vs Naive", "SMAPE (%)", "Bias (%)", "RMSE"))
    print("-" * 120)

    for cfg in configs:
        sc_c = np.mean(results[cfg]["chk_clipped"])
        sc_u = np.mean(results[cfg]["chk_unclipped"])
        sc_n = np.mean(results[cfg]["chk_naive"])
        sm = np.mean(results[cfg]["chk_smape"])
        bi = np.mean(results[cfg]["chk_bias"])
        rm = np.mean(results[cfg]["chk_rmse"])
        print(fmt.format(cfg, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.2f}%", f"{bi:.2f}%", f"{rm:.4f}"))

    print("\n" + "=" * 120)
    print("OVERALL 5-FOLD (SELECTION + CHECK) COMPLETE SUMMARY")
    print("=" * 120)
    print(fmt.format("Configuration", "5F Clipped", "5F Unclipped", "5F vs Naive", "SMAPE (%)", "Bias (%)", "RMSE"))
    print("-" * 120)

    for cfg in configs:
        sc_c = np.mean(results[cfg]["sel_clipped"] + results[cfg]["chk_clipped"])
        sc_u = np.mean(results[cfg]["sel_unclipped"] + results[cfg]["chk_unclipped"])
        sc_n = np.mean(results[cfg]["sel_naive"] + results[cfg]["chk_naive"])
        sm = np.mean(results[cfg]["sel_smape"] + results[cfg]["chk_smape"])
        bi = np.mean(results[cfg]["sel_bias"] + results[cfg]["chk_bias"])
        rm = np.mean(results[cfg]["sel_rmse"] + results[cfg]["chk_rmse"])
        print(fmt.format(cfg, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.2f}%", f"{bi:.2f}%", f"{rm:.4f}"))

if __name__ == "__main__":
    main()
