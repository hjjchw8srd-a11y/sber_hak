#!/usr/bin/env python3
"""
test_combined_super_ensemble.py

Evaluates the synergy of:
- TFT: 50/50 Blend (Base TFT + Stock TFT) vs 100% Base TFT
- LightGBM: Bagging 5-Seeds Dual L1/L2 vs Base Single-Seed L2
- CatBoost (29 stock features)
- Per-Category (29 stock features)
- Post-processing: Uniform Rint vs Adaptive Thresh v2

Evaluated on Folds [2, 1, 0].
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
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
)
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    train_predict_stock_catboost,
    train_predict_stock_per_category,
    LGBM_PARAMS,
)
from evaluate_check_and_all_folds_enhancements import fit_lgbm_hurdle, apply_uniform_p, apply_adaptive_p, calc_scores

ALL_FOLDS = {2: 107, 1: 116, 0: 125}
SEEDS = [42, 52, 62, 72, 82]

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Loading dataset for super-ensemble test...", flush=True)
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    configs = [
        "Baseline_Old_Champion",           # Base TFT + Single-Seed L2 LGBM + Uniform
        "TFT_Blend_Only",                  # 50/50 TFT + Single-Seed L2 LGBM + Uniform
        "LGBM_BagDual_Only",               # Base TFT + Bagging Dual LGBM + Uniform
        "SuperEnsemble_Uniform",           # 50/50 TFT + Bagging Dual LGBM + Uniform
        "SuperEnsemble_Adaptive_v2",       # 50/50 TFT + Bagging Dual LGBM + Adaptive v2
        "SuperEnsemble_1M_Dual_Uniform",   # 50/50 TFT + Scale 1M Dual LGBM + Uniform
        "SuperEnsemble_1M_Dual_Adaptive",  # 50/50 TFT + Scale 1M Dual LGBM + Adaptive v2
    ]

    records = {c: {f: {} for f in ALL_FOLDS} for c in configs}

    for fold, origin in ALL_FOLDS.items():
        print(f"\n>>> Running Fold {fold} (Origin {origin})...", flush=True)
        # Baselines
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

        # TFT models
        base_tft_df = pd.read_csv(PROJECT_ROOT / "experiments" / "tft" / f"tft_predictions_fold_{fold}.csv.gz")
        q_base_tft = base_tft_df["quantity"].to_numpy(dtype=np.float64)

        stock_tft_df = pd.read_csv(PROJECT_ROOT / "experiments" / "tft" / f"tft_stock_predictions_fold_{fold}.csv.gz")
        q_stock_tft = stock_tft_df["quantity"].to_numpy(dtype=np.float64)

        q_tft_blend = 0.5 * q_base_tft + 0.5 * q_stock_tft

        # Boostings
        latest_train = origin - HORIZON
        lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
        cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

        X_train_lgb_42, y_train_lgb_42, X_val, val_df, val_pos = build_feature_matrices(matrices, origin, lgbm_origins, seed=42, mode="29")
        X_train_cb_42, y_train_cb_42, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, seed=42, mode="29")
        feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

        pred_cb = train_predict_stock_catboost(X_train_cb_42, y_train_cb_42, X_val, feature_cols)
        pred_pcat = train_predict_stock_per_category(X_train_lgb_42, y_train_lgb_42, X_val, feature_cols + ["category_lvl_1"])

        # Single seed 42 L2
        p_42, cond_l2_42, cond_l1_42 = fit_lgbm_hurdle(X_train_lgb_42, y_train_lgb_42, X_val, feature_cols, LGBM_PARAMS, fit_l1=True)
        pred_lgb_single = np.where(p_42 < 0.5, 0.0, p_42 * cond_l2_42)

        # Bagging 5 seeds Dual
        bag_p = [p_42]
        bag_q_dual = [0.5 * cond_l2_42 + 0.5 * cond_l1_42]
        for s in [52, 62, 72, 82]:
            X_tr_s, y_tr_s, _, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, seed=s, mode="29")
            params_s = LGBM_PARAMS.copy()
            params_s["random_state"] = s
            p_s, q_l2_s, q_l1_s = fit_lgbm_hurdle(X_tr_s, y_tr_s, X_val, feature_cols, params_s, fit_l1=True)
            bag_p.append(p_s)
            bag_q_dual.append(0.5 * q_l2_s + 0.5 * q_l1_s)

        p_avg = np.mean(bag_p, axis=0)
        q_avg_dual = np.mean(bag_q_dual, axis=0)
        pred_lgb_bag_dual = np.where(p_avg < 0.5, 0.0, p_avg * q_avg_dual)

        # Scale 1M Dual
        X_tr_1m, y_tr_1m, _, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, max_rows=1_000_000, seed=42, mode="29")
        scale_params = LGBM_PARAMS.copy()
        scale_params["n_estimators"] = 300
        scale_params["learning_rate"] = 0.035
        p_1m, q_l2_1m, q_l1_1m = fit_lgbm_hurdle(X_tr_1m, y_tr_1m, X_val, feature_cols, scale_params, fit_l1=True)
        pred_lgb_1m_dual = np.where(p_1m < 0.5, 0.0, p_1m * (0.5 * q_l2_1m + 0.5 * q_l1_1m))

        # Active mask
        history13_raw = matrices.quantity[origin - 13 : origin, :]
        nz_rate = (history13_raw > 0).mean(axis=0)
        nz_rate_repeated = np.tile(nz_rate, HORIZON)
        active_mask = nz_rate_repeated >= 0.4

        # Assemble configs
        variants = {
            "Baseline_Old_Champion": (0.45 * q_base_tft + 0.37 * pred_lgb_single + 0.09 * pred_cb + 0.09 * pred_pcat, "uniform"),
            "TFT_Blend_Only": (0.45 * q_tft_blend + 0.37 * pred_lgb_single + 0.09 * pred_cb + 0.09 * pred_pcat, "uniform"),
            "LGBM_BagDual_Only": (0.45 * q_base_tft + 0.37 * pred_lgb_bag_dual + 0.09 * pred_cb + 0.09 * pred_pcat, "uniform"),
            "SuperEnsemble_Uniform": (0.45 * q_tft_blend + 0.37 * pred_lgb_bag_dual + 0.09 * pred_cb + 0.09 * pred_pcat, "uniform"),
            "SuperEnsemble_Adaptive_v2": (0.45 * q_tft_blend + 0.37 * pred_lgb_bag_dual + 0.09 * pred_cb + 0.09 * pred_pcat, "adaptive"),
            "SuperEnsemble_1M_Dual_Uniform": (0.45 * q_tft_blend + 0.37 * pred_lgb_1m_dual + 0.09 * pred_cb + 0.09 * pred_pcat, "uniform"),
            "SuperEnsemble_1M_Dual_Adaptive": (0.45 * q_tft_blend + 0.37 * pred_lgb_1m_dual + 0.09 * pred_cb + 0.09 * pred_pcat, "adaptive"),
        }

        for c_name, (raw_blend, p_type) in variants.items():
            if p_type == "adaptive":
                final_p = apply_adaptive_p(raw_blend, active_mask, 1.05, 0.60, 0.68)
            else:
                final_p = apply_uniform_p(raw_blend, 1.05, 0.65)

            p_df = pd.DataFrame({
                "week": truth["week"],
                "warehouse_name": truth["warehouse_name"],
                "article": truth["article"],
                "quantity": final_p,
            })

            m_pred = compute_all_metrics_fast(truth, p_df)
            c_sc, u_sc = calc_scores(m_pred, m_mean)
            _, n_sc = calc_scores(m_pred, m_naive)

            records[c_name][fold] = {
                "clipped": c_sc,
                "unclipped": u_sc,
                "naive": n_sc,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
            }

    print("\n" + "=" * 125, flush=True)
    print("FOLDS [2, 1, 0] SUPER-ENSEMBLE SYNERGY RESULTS", flush=True)
    print("=" * 125, flush=True)
    fmt = "{:<32} | {:<12} | {:<14} | {:<14} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("Configuration", "Clipped", "Unclipped", "Score vs Naive", "SMAPE (%)", "WAPE (%)", "Bias (%)", "RMSE"), flush=True)
    print("-" * 125, flush=True)

    for cfg_name in configs:
        sc_c = np.mean([records[cfg_name][f]["clipped"] for f in ALL_FOLDS])
        sc_u = np.mean([records[cfg_name][f]["unclipped"] for f in ALL_FOLDS])
        sc_n = np.mean([records[cfg_name][f]["naive"] for f in ALL_FOLDS])
        sm = np.mean([records[cfg_name][f]["smape"] for f in ALL_FOLDS])
        wa = np.mean([records[cfg_name][f]["wape"] for f in ALL_FOLDS])
        bi = np.mean([records[cfg_name][f]["bias"] for f in ALL_FOLDS])
        rm = np.mean([records[cfg_name][f]["rmse"] for f in ALL_FOLDS])
        print(fmt.format(cfg_name, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.3f}%", f"{wa:.3f}%", f"{bi:.2f}%", f"{rm:.4f}"), flush=True)

if __name__ == "__main__":
    main()
