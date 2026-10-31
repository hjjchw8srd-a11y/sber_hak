#!/usr/bin/env python3
"""
evaluate_check_and_all_folds_enhancements.py

Evaluates winning model architecture enhancements on Check Folds [1, 0]
and computes full 5-fold metrics [4, 3, 2, 1, 0].
Evaluates against BOTH baseline_mean13 and baseline_naive_last.
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
    CAT_COLUMNS,
    WeeklyMatrices,
    compute_all_metrics_fast,
)
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    train_predict_stock_catboost,
    train_predict_stock_per_category,
    LGBM_PARAMS,
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

def apply_uniform_p(raw_arr, alpha=1.05, thresh=0.65):
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)

def apply_adaptive_p(raw_arr, active_mask, alpha=1.05, thresh_active=0.60, thresh_inactive=0.68):
    scaled = raw_arr * alpha
    thresh_arr = np.where(active_mask, thresh_active, thresh_inactive)
    gated = np.where(scaled < thresh_arr, 0.0, scaled)
    return np.rint(gated).clip(0, None)

ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}

def fit_lgbm_hurdle(X_tr, y_tr, X_v, feature_cols, params, fit_l1=True):
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]
    X_train_c = X_tr[feature_cols].copy()
    X_val_c = X_v[feature_cols].copy()
    for c in cat_cols:
        X_train_c[c] = X_train_c[c].astype("category")
        X_val_c[c] = X_val_c[c].astype("category")

    clf_params = params.copy()
    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X_train_c, (y_tr > 0).astype(int))
    p_sale = clf.predict_proba(X_val_c)[:, 1]

    pos_mask = y_tr > 0
    X_pos = X_train_c.loc[pos_mask]
    y_pos = np.log1p(y_tr[pos_mask])

    reg_l2_params = params.copy()
    reg_l2_params["objective"] = "regression"
    reg_l2 = lgb.LGBMRegressor(**reg_l2_params)
    reg_l2.fit(X_pos, y_pos)
    cond_l2 = np.maximum(np.expm1(reg_l2.predict(X_val_c)), 0.0)

    cond_l1 = None
    if fit_l1:
        reg_l1_params = params.copy()
        reg_l1_params["objective"] = "regression_l1"
        reg_l1 = lgb.LGBMRegressor(**reg_l1_params)
        reg_l1.fit(X_pos, y_pos)
        cond_l1 = np.maximum(np.expm1(reg_l1.predict(X_val_c)), 0.0)

    return p_sale, cond_l2, cond_l1


def run_fold(matrices, fold, origin, configs, SEEDS):
    t0 = time.perf_counter()
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

    df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
    df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

    latest_train = origin - HORIZON
    lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
    cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

    # Baseline single-seed matrices (seed=42)
    X_train_lgb_42, y_train_lgb_42, X_val, val_df, val_pos = build_feature_matrices(matrices, origin, lgbm_origins, seed=42, mode="29")
    X_train_cb_42, y_train_cb_42, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, seed=42, mode="29")

    feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

    # Fixed components (CB and PerCat on 29 features)
    pred_cb = train_predict_stock_catboost(X_train_cb_42, y_train_cb_42, X_val, feature_cols)
    pred_pcat = train_predict_stock_per_category(X_train_lgb_42, y_train_lgb_42, X_val, feature_cols + ["category_lvl_1"])
    q_tft = df_f["raw_tft"].to_numpy(dtype=np.float64)

    # Active mask for adaptive thresholding
    # nonzero_rate_13 >= 0.4
    history13_raw = matrices.quantity[origin - 13 : origin, :]
    nz_rate = (history13_raw > 0).mean(axis=0)
    nz_rate_repeated = np.tile(nz_rate, HORIZON)
    active_mask = nz_rate_repeated >= 0.4

    # Train LGBM Seed 42 (Classifier, L2, L1)
    p_42, cond_l2_42, cond_l1_42 = fit_lgbm_hurdle(X_train_lgb_42, y_train_lgb_42, X_val, feature_cols, LGBM_PARAMS, fit_l1=True)

    lgbm_preds = {}
    lgbm_preds["Base_L2_Seed42"] = np.where(p_42 < 0.5, 0.0, p_42 * cond_l2_42)
    lgbm_preds["Dual_Gamma_0.5"] = np.where(p_42 < 0.5, 0.0, p_42 * (0.5 * cond_l2_42 + 0.5 * cond_l1_42))
    lgbm_preds["Pure_L1"] = np.where(p_42 < 0.5, 0.0, p_42 * cond_l1_42)

    # Bagging seeds: 42, 52, 62, 72, 82
    bag_p = [p_42]
    bag_q_l2 = [cond_l2_42]
    bag_q_dual = [0.5 * cond_l2_42 + 0.5 * cond_l1_42]

    for s in [52, 62, 72, 82]:
        X_tr_s, y_tr_s, _, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, seed=s, mode="29")
        params_s = LGBM_PARAMS.copy()
        params_s["random_state"] = s
        p_s, q_l2_s, q_l1_s = fit_lgbm_hurdle(X_tr_s, y_tr_s, X_val, feature_cols, params_s, fit_l1=True)
        bag_p.append(p_s)
        bag_q_l2.append(q_l2_s)
        bag_q_dual.append(0.5 * q_l2_s + 0.5 * q_l1_s)

    p_avg = np.mean(bag_p, axis=0)
    q_avg_l2 = np.mean(bag_q_l2, axis=0)
    q_avg_dual = np.mean(bag_q_dual, axis=0)

    lgbm_preds["Bagging_5Seeds_L2"] = np.where(p_avg < 0.5, 0.0, p_avg * q_avg_l2)
    lgbm_preds["Bagging_5Seeds_Dual_0.5"] = np.where(p_avg < 0.5, 0.0, p_avg * q_avg_dual)

    # Scale 1M rows
    X_tr_1m, y_tr_1m, _, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, max_rows=1_000_000, seed=42, mode="29")
    scale_params = LGBM_PARAMS.copy()
    scale_params["n_estimators"] = 300
    scale_params["learning_rate"] = 0.035
    p_1m, q_l2_1m, q_l1_1m = fit_lgbm_hurdle(X_tr_1m, y_tr_1m, X_val, feature_cols, scale_params, fit_l1=True)

    lgbm_preds["Scale_1M_Dual_0.5"] = np.where(p_1m < 0.5, 0.0, p_1m * (0.5 * q_l2_1m + 0.5 * q_l1_1m))

    fold_metrics = {}
    for cfg_name in configs:
        if cfg_name == "Bagging_Dual_Adaptive_v2":
            lgb_p = lgbm_preds["Bagging_5Seeds_Dual_0.5"]
            raw_blend = 0.45 * q_tft + 0.37 * lgb_p + 0.09 * pred_cb + 0.09 * pred_pcat
            final_p = apply_adaptive_p(raw_blend, active_mask, 1.05, 0.60, 0.68)
        else:
            lgb_p = lgbm_preds[cfg_name]
            raw_blend = 0.45 * q_tft + 0.37 * lgb_p + 0.09 * pred_cb + 0.09 * pred_pcat
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

        fold_metrics[cfg_name] = {
            "clipped": c_sc,
            "unclipped": u_sc,
            "naive": n_sc,
            "smape": m_pred["smape"],
            "wape": m_pred["wape"],
            "rmse": m_pred["rmse"],
            "bias": m_pred["bias"],
            "rmsle": m_pred["rmsle"],
        }

    print(f"Fold {fold} finished in {time.perf_counter() - t0:.1f}s", flush=True)
    return fold_metrics


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Loading dataset...", flush=True)
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    configs = [
        "Base_L2_Seed42",
        "Dual_Gamma_0.5",
        "Pure_L1",
        "Bagging_5Seeds_L2",
        "Bagging_5Seeds_Dual_0.5",
        "Scale_1M_Dual_0.5",
        "Bagging_Dual_Adaptive_v2",
    ]

    SEEDS = [42, 52, 62, 72, 82]

    # Run on all 5 folds: Selection [4, 3, 2] -> Check [1, 0]
    all_results = {c: {f: {} for f in ALL_FOLDS} for c in configs}

    print("=" * 115, flush=True)
    print("RUNNING 5-FOLD HONEST EVALUATION: SELECTION [4, 3, 2] -> CHECK [1, 0]", flush=True)
    print("=" * 115, flush=True)

    for fold, origin in ALL_FOLDS.items():
        stage = "Selection" if fold in [4, 3, 2] else "Check"
        print(f"\n>>> Running {stage} Fold {fold} (Origin {origin})...", flush=True)
        f_res = run_fold(matrices, fold, origin, configs, SEEDS)
        for c in configs:
            all_results[c][fold] = f_res[c]

    # Print summary tables
    for split_name, folds in [("SELECTION FOLDS [4, 3, 2]", [4, 3, 2]), ("CHECK FOLDS [1, 0]", [1, 0]), ("FULL 5-FOLD AVERAGE", [4, 3, 2, 1, 0])]:
        print("\n" + "=" * 125, flush=True)
        print(f"{split_name} SUMMARY", flush=True)
        print("=" * 125, flush=True)
        fmt = "{:<26} | {:<12} | {:<14} | {:<14} | {:<10} | {:<10} | {:<10} | {:<10}"
        print(fmt.format("Configuration", "Clipped", "Unclipped", "Score vs Naive", "SMAPE (%)", "WAPE (%)", "Bias (%)", "RMSE"), flush=True)
        print("-" * 125, flush=True)

        for cfg_name in configs:
            sc_c = np.mean([all_results[cfg_name][f]["clipped"] for f in folds])
            sc_u = np.mean([all_results[cfg_name][f]["unclipped"] for f in folds])
            sc_n = np.mean([all_results[cfg_name][f]["naive"] for f in folds])
            sm = np.mean([all_results[cfg_name][f]["smape"] for f in folds])
            wa = np.mean([all_results[cfg_name][f]["wape"] for f in folds])
            bi = np.mean([all_results[cfg_name][f]["bias"] for f in folds])
            rm = np.mean([all_results[cfg_name][f]["rmse"] for f in folds])
            print(fmt.format(cfg_name, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.3f}%", f"{wa:.3f}%", f"{bi:.2f}%", f"{rm:.4f}"), flush=True)

    # Save to JSON
    import json
    save_path = PROJECT_ROOT / "experiments" / "model_improvements" / "five_folds_enhancements_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nSaved detailed 5-fold results to {save_path}", flush=True)

if __name__ == "__main__":
    main()
