#!/usr/bin/env python3
"""
train_lgbm_task3_stock_yearly.py

TASK 3: Feature Expansion for LightGBM B (Stock Dynamics & Yearly Seasonality)
Strict two-stage validation: Selection Folds (4, 3, 2) -> Check Folds (1, 0).

Features added:
1. Stock Dynamics:
   - stock_end_lag_1
   - stock_mean_lag_1
   - stock_nonpositive_rate_lag_1
   - weeks_of_supply: log1p(max(stock_end_lag_1, 0) / (quantity_mean_13 + 0.1))
   - unconstrained_demand_13: mean quantity when stock_nonpositive_rate < 0.5 (fallback to quantity_mean_13)
2. Yearly Seasonality:
   - quantity_lag_52 (log1p)
   - quantity_mean_lag_48_52 (log1p)
   - Safe NaN handling for origins < 52 weeks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
    signed_log1p,
)
from train_tft_weekly import baseline_mean13

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "lgbm_task3_stock_yearly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLDS = {4: 89, 3: 98, 2: 107}
CHECK_FOLDS = {1: 116, 0: 125}
ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}

PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.0495,
    "num_leaves": 29,
    "min_child_samples": 51,
    "subsample": 0.727,
    "colsample_bytree": 0.728,
    "reg_alpha": 0.0164,
    "reg_lambda": 0.0057,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


def apply_p(x: np.ndarray, alpha: float = 1.05, threshold: float = 0.65) -> np.ndarray:
    scaled = alpha * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < threshold, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_extra_features_for_origins(matrices: WeeklyMatrices, origin_pos: np.ndarray, pair_idx: np.ndarray) -> dict[str, np.ndarray]:
    """
    Computes Task 3 stock dynamics and yearly seasonality features.
    Vectorized and highly efficient.
    """
    history_pos = origin_pos - 1
    stock_end = matrices.stock_end[history_pos, pair_idx]
    stock_mean = matrices.stock_mean[history_pos, pair_idx]
    stock_nonpos = matrices.stock_nonpositive_rate[history_pos, pair_idx]
    
    # 1. Direct stock features
    stock_end_lag_1 = signed_log1p(stock_end)
    stock_mean_lag_1 = signed_log1p(stock_mean)
    stock_nonpos_lag_1 = stock_nonpos.copy()
    
    # Precompute per-origin table for unconstrained demand and yearly lags
    unique_origins = np.unique(origin_pos)
    n_pairs = matrices.n_pairs
    
    unconstrained_table = {}
    q52_table = {}
    q48_52_table = {}
    
    for O in unique_origins:
        # Unconstrained demand 13
        Q13 = matrices.quantity[O - 13 : O, :]
        SNR13 = matrices.stock_nonpositive_rate[O - 13 : O, :]
        valid = SNR13 < 0.5
        count = valid.sum(axis=0)
        q_sum = (Q13 * valid).sum(axis=0)
        unconstrained_table[O] = np.where(count > 0, q_sum / np.maximum(count, 1), Q13.mean(axis=0))
        
        # Yearly lags
        if O >= 52:
            q52_table[O] = np.log1p(np.maximum(matrices.quantity[O - 52, :], 0.0))
            q48_52_table[O] = np.log1p(np.maximum(matrices.quantity[O - 52 : O - 47, :].mean(axis=0), 0.0))
        else:
            q52_table[O] = np.full(n_pairs, np.nan, dtype=np.float32)
            q48_52_table[O] = np.full(n_pairs, np.nan, dtype=np.float32)

    # Broadcast back to row samples
    unconstrained_arr = np.empty(len(origin_pos), dtype=np.float32)
    q52_arr = np.empty(len(origin_pos), dtype=np.float32)
    q48_52_arr = np.empty(len(origin_pos), dtype=np.float32)
    
    for O in unique_origins:
        mask = origin_pos == O
        pairs_O = pair_idx[mask]
        unconstrained_arr[mask] = unconstrained_table[O][pairs_O]
        q52_arr[mask] = q52_table[O][pairs_O]
        q48_52_arr[mask] = q48_52_table[O][pairs_O]

    return {
        "stock_end_lag_1": stock_end_lag_1,
        "stock_mean_lag_1": stock_mean_lag_1,
        "stock_nonpositive_rate_lag_1": stock_nonpos_lag_1,
        "unconstrained_demand_13": np.log1p(np.maximum(unconstrained_arr, 0.0)),
        "quantity_lag_52": q52_arr,
        "quantity_mean_lag_48_52": q48_52_arr,
    }


def train_eval_fold(
    matrices: WeeklyMatrices,
    origin: int,
    data: pd.DataFrame,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    feature_mode: 'baseline', 'stock', or 'stock_yearly'
    """
    latest_train_origin = origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - 51)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)

    train_df, target_arr, target_pos = matrices.make_rows(train_origins, 350_000, seed=42)
    val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)

    # Recover origin_pos and pair_idx for extra features
    # train_df contains horizon and time_idx: origin_pos = time_idx - horizon + 1
    train_origin_pos = train_df["time_idx"].to_numpy() - train_df["horizon"].to_numpy() + 1
    val_origin_pos = val_df["time_idx"].to_numpy() - val_df["horizon"].to_numpy() + 1

    train_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([train_df["warehouse_name"], train_df["article"]])
    )
    val_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([val_df["warehouse_name"], val_df["article"]])
    )

    base_cols = columns_for_experiment("base_calendar_network")
    cat_columns = [c for c in CAT_COLUMNS if c in base_cols]

    X_train = train_df[base_cols].copy()
    X_val = val_df[base_cols].copy()

    # Weeks of supply
    train_mean13 = train_df["quantity_mean_13"].to_numpy()
    val_mean13 = val_df["quantity_mean_13"].to_numpy()

    if feature_mode in ["stock", "stock_yearly"]:
        train_extra = compute_extra_features_for_origins(matrices, train_origin_pos, train_pair_idx)
        val_extra = compute_extra_features_for_origins(matrices, val_origin_pos, val_pair_idx)

        # Stock features
        X_train["stock_end_lag_1"] = train_extra["stock_end_lag_1"]
        X_val["stock_end_lag_1"] = val_extra["stock_end_lag_1"]

        X_train["stock_mean_lag_1"] = train_extra["stock_mean_lag_1"]
        X_val["stock_mean_lag_1"] = val_extra["stock_mean_lag_1"]

        X_train["stock_nonpositive_rate_lag_1"] = train_extra["stock_nonpositive_rate_lag_1"]
        X_val["stock_nonpositive_rate_lag_1"] = val_extra["stock_nonpositive_rate_lag_1"]

        raw_train_stock = np.maximum(matrices.stock_end[train_origin_pos - 1, train_pair_idx], 0.0)
        raw_val_stock = np.maximum(matrices.stock_end[val_origin_pos - 1, val_pair_idx], 0.0)
        X_train["weeks_of_supply"] = np.log1p(raw_train_stock / (train_mean13 + 0.1))
        X_val["weeks_of_supply"] = np.log1p(raw_val_stock / (val_mean13 + 0.1))

        X_train["unconstrained_demand_13"] = train_extra["unconstrained_demand_13"]
        X_val["unconstrained_demand_13"] = val_extra["unconstrained_demand_13"]

    if feature_mode == "stock_yearly":
        X_train["quantity_lag_52"] = train_extra["quantity_lag_52"]
        X_val["quantity_lag_52"] = val_extra["quantity_lag_52"]

        X_train["quantity_mean_lag_48_52"] = train_extra["quantity_mean_lag_48_52"]
        X_val["quantity_mean_lag_48_52"] = val_extra["quantity_mean_lag_48_52"]

    for col in cat_columns:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(**PARAMS)
    clf.fit(X_train, (target_arr > 0).astype(int))
    p_sale = clf.predict_proba(X_val)[:, 1]

    pos_idx = target_arr > 0
    reg = lgb.LGBMRegressor(**PARAMS)
    reg.fit(X_train.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_val = np.maximum(np.expm1(reg.predict(X_val)), 0.0)
    exp_val = p_sale * cond_val
    after_gate = np.where(p_sale < 0.5, 0.0, exp_val)
    fit_time = time.perf_counter() - t0

    return p_sale, exp_val, after_gate, {"features": list(X_train.columns), "fit_time": fit_time}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("TASK 3: LIGHTGBM B FEATURE EXPANSION (STOCK DYNAMICS & YEARLY SEASONALITY)")
    print("Strict Selection Phase (Folds 4, 3, 2)")
    print("=" * 100)

    modes = ["baseline", "stock", "stock_yearly"]
    summary_results = {}

    for mode in modes:
        print(f"\n--- Testing Feature Mode: {mode.upper()} ---")
        unc_list = []
        score_list = []
        lgbm_unc_list = []
        lgbm_score_list = []

        for fold, origin in SELECTION_FOLDS.items():
            val_weeks = weeks[origin : origin + HORIZON]
            truth = make_truth(matrices, origin)
            truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")
            b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
            b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
            bm = compute_all_metrics_fast(truth, b_df)

            df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
            df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

            # Train LGBM with specified mode
            p_sale, exp_val, after_gate, info = train_eval_fold(matrices, origin, data, mode)
            print(f"Fold {fold} (Origin {origin}): {len(info['features'])} features fitted in {info['fit_time']:.1f}s")

            # Val DataFrame with week and keys
            val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)
            val_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]
            val_df["article"] = val_df["article"].astype(df_f["article"].dtype)

            # Evaluate Standalone LightGBM
            sub_lgbm = truth[KEYS].copy()
            sub_lgbm["quantity"] = apply_p(after_gate, 1.05, 0.65)

            m_lgbm = compute_all_metrics_fast(truth, sub_lgbm)
            lgbm_unc = sum(w * (m_lgbm[k] / bm[k]) for k, w in METRIC_WEIGHTS.items())
            lgbm_score = sum(w * min(max(m_lgbm[k] / bm[k], 0), 1) for k, w in METRIC_WEIGHTS.items())
            lgbm_unc_list.append(lgbm_unc)
            lgbm_score_list.append(lgbm_score)

            # Evaluate Candidate Ensemble (45 TFT + 37 new LGBM + 9 CB + 9 PerCat)
            val_df["new_lgbm_gate"] = after_gate
            merged = df_f.merge(val_df[["week", "warehouse_name", "article", "new_lgbm_gate"]], on=["week", "warehouse_name", "article"], how="left")

            raw_ens = (
                0.45 * merged["raw_tft"]
                + 0.37 * merged["new_lgbm_gate"]
                + 0.09 * merged["after_gate_catboost"]
                + 0.09 * merged["after_comp_per_category"]
            )
            sub_ens = truth[KEYS].copy()
            sub_ens["warehouse_name"] = merged["warehouse_name"]
            sub_ens["article"] = merged["article"]
            sub_ens["quantity"] = apply_p(raw_ens, 1.05, 0.65)

            m_ens = compute_all_metrics_fast(truth, sub_ens)
            ens_unc = sum(w * (m_ens[k] / bm[k]) for k, w in METRIC_WEIGHTS.items())
            ens_score = sum(w * min(max(m_ens[k] / bm[k], 0), 1) for k, w in METRIC_WEIGHTS.items())
            unc_list.append(ens_unc)
            score_list.append(ens_score)

        summary_results[mode] = {
            "lgbm_unclipped": float(np.mean(lgbm_unc_list)),
            "lgbm_clipped": float(np.mean(lgbm_score_list)),
            "ensemble_unclipped": float(np.mean(unc_list)),
            "ensemble_clipped": float(np.mean(score_list)),
            "delta_ens_unc_vs_ctrl": float(np.mean(unc_list) - 0.761314),
            "delta_ens_unc_vs_cand": float(np.mean(unc_list) - 0.760155),
        }
        print(f"Summary for {mode.upper()}:")
        print(f"  Standalone LGBM: Unclipped={summary_results[mode]['lgbm_unclipped']:.6f}, Clipped={summary_results[mode]['lgbm_clipped']:.6f}")
        print(f"  Ensemble 45/37/9/9: Unclipped={summary_results[mode]['ensemble_unclipped']:.6f} (Delta vs Cand: {summary_results[mode]['delta_ens_unc_vs_cand']:+.6f})")

    res_df = pd.DataFrame(summary_results).T
    out_csv = OUTPUT_DIR / "selection_summary.csv"
    res_df.to_csv(out_csv)
    print("\n" + "=" * 100)
    print("FINAL SELECTION COMPARISON:")
    print(res_df.to_string())
    print("=" * 100)


if __name__ == "__main__":
    main()
