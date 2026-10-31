#!/usr/bin/env python3
"""
experiment_all_stock_ensemble.py

TASK 1, 2, 3: Full-Ensemble Stock Arming (29 & 33 features) across LightGBM, CatBoost, and Per-Category.
Two-stage protocol: Selection Folds [4, 3, 2] -> Check Folds [1, 0].
Evaluates against BOTH baseline_mean13 and baseline_naive_last (TZ clause 5.3).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

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

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "all_stock_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLDS = {4: 89, 3: 98, 2: 107}
CHECK_FOLDS = {1: 116, 0: 125}
ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}

LGBM_PARAMS = {
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

CATBOOST_PARAMS = {
    "iterations": 160,
    "learning_rate": 0.06,
    "depth": 6,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}

PER_CAT_PARAMS = {
    "n_estimators": 140,
    "learning_rate": 0.06,
    "num_leaves": 25,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


def apply_p(x: np.ndarray, alpha: float = 1.05, threshold: float = 0.65) -> np.ndarray:
    scaled = alpha * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < threshold, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_extended_features(
    matrices: WeeklyMatrices,
    origin_pos: np.ndarray,
    pair_idx: np.ndarray,
    horizon_idx: np.ndarray,
    mode: str = "33",  # '29' or '33'
) -> dict[str, np.ndarray]:
    n_samples = len(origin_pos)
    history_pos = origin_pos - 1
    target_pos = origin_pos + horizon_idx

    stock_end = matrices.stock_end[history_pos, pair_idx]
    stock_mean = matrices.stock_mean[history_pos, pair_idx]
    stock_nonpos = matrices.stock_nonpositive_rate[history_pos, pair_idx]

    stock_end_lag_1 = signed_log1p(stock_end)
    stock_mean_lag_1 = signed_log1p(stock_mean)
    stock_nonpos_lag_1 = stock_nonpos.copy()

    unique_origins = np.unique(origin_pos)
    n_pairs = matrices.n_pairs

    unconstrained_table = {}
    trend_table = {}
    streak_table = {}

    for O in unique_origins:
        # 1. Unconstrained demand 13
        Q13 = matrices.quantity[O - 13 : O, :]
        SNR13 = matrices.stock_nonpositive_rate[O - 13 : O, :]
        valid = SNR13 < 0.5
        count = valid.sum(axis=0)
        q_sum = (Q13 * valid).sum(axis=0)
        unconstrained_table[O] = np.where(count > 0, q_sum / np.maximum(count, 1), Q13.mean(axis=0))

        if mode == "33":
            # 2. stock_trend_4_vs_13
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sm4 = np.nanmean(matrices.stock_mean[O - 4 : O, :], axis=0)
                sm13 = np.nanmean(matrices.stock_mean[O - 13 : O, :], axis=0)
            trend_table[O] = signed_log1p(np.nan_to_num(sm4, nan=0.0)) - signed_log1p(np.nan_to_num(sm13, nan=0.0))

            # 3. zero_stock_streak
            rev_snr = (matrices.stock_nonpositive_rate[:O, :] >= 0.5)[::-1, :]
            first_non_oos = np.argmax(~rev_snr, axis=0)
            all_oos = rev_snr.all(axis=0)
            streak_table[O] = np.where(all_oos, O, first_non_oos).astype(np.float32)

    unconstrained_arr = np.empty(n_samples, dtype=np.float32)
    for O in unique_origins:
        mask = origin_pos == O
        unconstrained_arr[mask] = unconstrained_table[O][pair_idx[mask]]

    features = {
        "stock_end_lag_1": stock_end_lag_1,
        "stock_mean_lag_1": stock_mean_lag_1,
        "stock_nonpositive_rate_lag_1": stock_nonpos_lag_1,
        "unconstrained_demand_13": np.log1p(np.maximum(unconstrained_arr, 0.0)),
    }

    if mode == "33":
        stock_trend_arr = np.empty(n_samples, dtype=np.float32)
        zero_streak_arr = np.empty(n_samples, dtype=np.float32)
        for O in unique_origins:
            mask = origin_pos == O
            stock_trend_arr[mask] = trend_table[O][pair_idx[mask]]
            zero_streak_arr[mask] = streak_table[O][pair_idx[mask]]

        valid_lag52 = target_pos >= 52
        lag52_arr = np.full(n_samples, np.nan, dtype=np.float32)
        mean_year_arr = np.full(n_samples, np.nan, dtype=np.float32)

        if valid_lag52.any():
            idx_valid = np.where(valid_lag52)[0]
            t_valid = target_pos[idx_valid]
            p_valid = pair_idx[idx_valid]

            lag52_raw = matrices.quantity[t_valid - 52, p_valid]
            lag52_arr[idx_valid] = signed_log1p(lag52_raw)

            w_51 = matrices.quantity[t_valid - 51, p_valid]
            w_52 = lag52_raw
            w_53 = np.where(t_valid >= 53, matrices.quantity[t_valid - 53, p_valid], lag52_raw)
            mean_year_arr[idx_valid] = signed_log1p((w_51 + w_52 + w_53) / 3.0)

        features["stock_trend_4_vs_13"] = stock_trend_arr
        features["zero_stock_streak"] = zero_streak_arr
        features["quantity_lag_52"] = lag52_arr
        features["quantity_mean_year_window"] = mean_year_arr

    return features


def build_feature_matrices(matrices: WeeklyMatrices, origin: int, train_origins: np.ndarray, max_rows: int = 350_000, seed: int = 42, mode: str = "29"):
    train_df, target_arr, _ = matrices.make_rows(train_origins, max_rows, seed=seed)
    val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)

    train_origin_pos = train_df["time_idx"].to_numpy() - train_df["horizon"].to_numpy() + 1
    val_origin_pos = val_df["time_idx"].to_numpy() - val_df["horizon"].to_numpy() + 1

    train_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([train_df["warehouse_name"], train_df["article"]])
    )
    val_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([val_df["warehouse_name"], val_df["article"]])
    )

    train_h_idx = train_df["horizon"].to_numpy() - 1
    val_h_idx = val_df["horizon"].to_numpy() - 1

    base_cols = columns_for_experiment("base_calendar_network")
    X_train = train_df[base_cols].copy()
    X_val = val_df[base_cols].copy()

    # Weeks of supply
    train_mean13 = train_df["quantity_mean_13"].to_numpy()
    val_mean13 = val_df["quantity_mean_13"].to_numpy()

    extra_train = compute_extended_features(matrices, train_origin_pos, train_pair_idx, train_h_idx, mode=mode)
    extra_val = compute_extended_features(matrices, val_origin_pos, val_pair_idx, val_h_idx, mode=mode)

    for k in extra_train:
        X_train[k] = extra_train[k]
        X_val[k] = extra_val[k]

    raw_train_stock = np.maximum(matrices.stock_end[train_origin_pos - 1, train_pair_idx], 0.0)
    raw_val_stock = np.maximum(matrices.stock_end[val_origin_pos - 1, val_pair_idx], 0.0)
    X_train["weeks_of_supply"] = np.log1p(raw_train_stock / (train_mean13 + 0.1))
    X_val["weeks_of_supply"] = np.log1p(raw_val_stock / (val_mean13 + 0.1))

    # Add category_lvl_1 for per-category splitting
    X_train["category_lvl_1"] = train_df["category_lvl_1"].values
    X_val["category_lvl_1"] = val_df["category_lvl_1"].values

    return X_train, target_arr, X_val, val_df, val_pos


def train_predict_stock_lgbm(X_train: pd.DataFrame, target_arr: np.ndarray, X_val: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]
    X_tr = X_train[feature_cols].copy()
    X_v = X_val[feature_cols].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_v[c] = X_v[c].astype("category")

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, (target_arr > 0).astype(int))
    p_sale = clf.predict_proba(X_v)[:, 1]

    pos_mask = target_arr > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS)
    reg.fit(X_tr.loc[pos_mask], np.log1p(target_arr[pos_mask]))
    cond = np.maximum(np.expm1(reg.predict(X_v)), 0.0)
    exp_val = p_sale * cond
    return np.where(p_sale < 0.5, 0.0, exp_val)


def train_predict_stock_catboost(X_train: pd.DataFrame, target_arr: np.ndarray, X_val: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]
    X_tr = X_train[feature_cols].copy()
    X_v = X_val[feature_cols].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype(str).fillna("unknown")
        X_v[c] = X_v[c].astype(str).fillna("unknown")

    cb = CatBoostRegressor(**CATBOOST_PARAMS, cat_features=cat_cols)
    cb.fit(X_tr, target_arr)
    return np.maximum(cb.predict(X_v), 0.0)


def train_predict_stock_per_category(X_train: pd.DataFrame, target_arr: np.ndarray, X_val: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols and c != "category_lvl_1"]
    cols_without_cat1 = [c for c in feature_cols if c != "category_lvl_1"]

    preds = np.zeros(len(X_val), dtype=np.float32)
    categories = X_val["category_lvl_1"].unique()

    for cat in categories:
        val_mask = X_val["category_lvl_1"] == cat
        tr_mask = X_train["category_lvl_1"] == cat

        X_tr_cat = X_train.loc[tr_mask, cols_without_cat1].copy()
        y_tr_cat = target_arr[tr_mask]
        X_v_cat = X_val.loc[val_mask, cols_without_cat1].copy()

        if len(X_v_cat) == 0:
            continue

        pos_mask = y_tr_cat > 0
        if pos_mask.sum() < 20:
            preds[val_mask] = max(0.0, float(np.mean(y_tr_cat))) if len(y_tr_cat) > 0 else 0.0
            continue

        for c in cat_cols:
            X_tr_cat[c] = X_tr_cat[c].astype("category")
            X_v_cat[c] = X_v_cat[c].astype("category")

        clf = lgb.LGBMClassifier(**PER_CAT_PARAMS)
        clf.fit(X_tr_cat, (y_tr_cat > 0).astype(int))
        p_sale = clf.predict_proba(X_v_cat)[:, 1]

        reg = lgb.LGBMRegressor(**PER_CAT_PARAMS)
        reg.fit(X_tr_cat.loc[pos_mask], np.log1p(y_tr_cat[pos_mask]))
        cond = np.maximum(np.expm1(reg.predict(X_v_cat)), 0.0)
        exp_val = p_sale * cond
        preds[val_mask] = np.where(p_sale < 0.5, 0.0, exp_val)

    return preds


def evaluate_predictions(truth_df, pred_df, b_mean13_df, b_naive_df):
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm_13 = compute_all_metrics_fast(truth_df, b_mean13_df)
    bm_nv = compute_all_metrics_fast(truth_df, b_naive_df)

    # Against mean13
    unc_13 = sum(w * (m[k] / bm_13[k]) for k, w in METRIC_WEIGHTS.items())
    clip_13 = sum(w * min(max(m[k] / bm_13[k], 0), 1) for k, w in METRIC_WEIGHTS.items())

    # Against naive_last
    unc_nv = sum(w * (m[k] / bm_nv[k]) for k, w in METRIC_WEIGHTS.items())
    clip_nv = sum(w * min(max(m[k] / bm_nv[k], 0), 1) for k, w in METRIC_WEIGHTS.items())

    return {
        "unc_mean13": unc_13,
        "clip_mean13": clip_13,
        "unc_naive": unc_nv,
        "clip_naive": clip_nv,
        "smape": m["smape"],
        "bias": m["bias"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "rmsle": m["rmsle"],
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("TASK 1 & TASK 2: FULL-ENSEMBLE STOCK ARMING EXPERIMENT")
    print("Testing 29 vs 33 Features across LGBM, CatBoost, and Per-Category")
    print("Selection Folds: [4, 3, 2]")
    print("=" * 100)

    for feature_mode in ["29", "33"]:
        print(f"\n=======================================================")
        print(f"RUNNING FEATURE MODE: {feature_mode} FEATURES")
        print(f"=======================================================")

        records = []

        for fold, origin in SELECTION_FOLDS.items():
            t0 = time.perf_counter()
            print(f"\n--- FOLD {fold} (Origin {origin}) ---")

            val_weeks = weeks[origin : origin + HORIZON]
            truth = make_truth(matrices, origin)
            truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

            b_mean13 = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
            b_mean13["week"] = pd.to_datetime(b_mean13["week"]).dt.strftime("%Y-%m-%d")

            last_sales = matrices.quantity[origin - 1, :]
            pairs = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
            val_wks_arr = np.repeat(matrices.weeks[origin : origin + HORIZON].to_numpy(), matrices.n_pairs)
            b_naive = pd.DataFrame({
                "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
                "warehouse_name": pairs["warehouse_name"],
                "article": pairs["article"],
                "quantity": np.tile(last_sales, HORIZON),
            })

            # Read historical blend_b for TFT and baseline components
            df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
            df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

            # Origins
            latest_train = origin - HORIZON
            lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
            cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

            # 1. Build features
            X_train_lgb, y_train_lgb, X_val, val_df, val_pos = build_feature_matrices(matrices, origin, lgbm_origins, mode=feature_mode)
            X_train_cb, y_train_cb, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, mode=feature_mode)

            val_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]
            val_df["article"] = val_df["article"].astype(df_f["article"].dtype)

            feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

            # Model 1: Stock LGBM (52w)
            t_m0 = time.perf_counter()
            pred_lgb = train_predict_stock_lgbm(X_train_lgb, y_train_lgb, X_val, feature_cols)
            print(f"  LGBM-{feature_mode} fitted in {time.perf_counter() - t_m0:.1f}s")

            # Model 2: Stock CatBoost (18w)
            t_m1 = time.perf_counter()
            pred_cb = train_predict_stock_catboost(X_train_cb, y_train_cb, X_val, feature_cols)
            print(f"  CatBoost-{feature_mode} fitted in {time.perf_counter() - t_m1:.1f}s")

            # Model 3: Stock Per-Category (14 models)
            t_m2 = time.perf_counter()
            pred_pcat = train_predict_stock_per_category(X_train_lgb, y_train_lgb, X_val, feature_cols + ["category_lvl_1"])
            print(f"  Per-Category-{feature_mode} fitted in {time.perf_counter() - t_m2:.1f}s")

            val_df["new_lgbm"] = pred_lgb
            val_df["new_cb"] = pred_cb
            val_df["new_pcat"] = pred_pcat

            merged = df_f.merge(val_df[["week", "warehouse_name", "article", "new_lgbm", "new_cb", "new_pcat"]], on=KEYS, how="left")

            # Models to evaluate:
            # 1. Control B (Baseline)
            c_ctrl = merged["final_prediction"].to_numpy()

            # 2. Candidate 45/37/9/9 (Previous Champion Base LGBM)
            c_cand = apply_p(
                0.45 * merged["raw_tft"]
                + 0.37 * merged["after_gate_lgbm"]
                + 0.09 * merged["after_gate_catboost"]
                + 0.09 * merged["after_comp_per_category"],
                1.05, 0.65
            )

            # 3. Only LGBM Armed with Stock (Previous step)
            c_lgbm_stock = apply_p(
                0.45 * merged["raw_tft"]
                + 0.37 * merged["new_lgbm"]
                + 0.09 * merged["after_gate_catboost"]
                + 0.09 * merged["after_comp_per_category"],
                1.05, 0.65
            )

            # 4. ALL BOOSTINGS ARMED WITH STOCK (LGBM + CatBoost + PerCat)
            c_all_stock = apply_p(
                0.45 * merged["raw_tft"]
                + 0.37 * merged["new_lgbm"]
                + 0.09 * merged["new_cb"]
                + 0.09 * merged["new_pcat"],
                1.05, 0.65
            )

            models = {
                "0. Control B": c_ctrl,
                "1. Candidate 45/37/9/9": c_cand,
                f"2. LGBM-{feature_mode} Only": c_lgbm_stock,
                f"3. All-Stock-{feature_mode} (LGBM+CB+PerCat)": c_all_stock,
            }

            for name, preds in models.items():
                sub = truth[KEYS].copy()
                sub["warehouse_name"] = merged["warehouse_name"]
                sub["article"] = merged["article"]
                sub["quantity"] = preds

                res = evaluate_predictions(truth, sub, b_mean13, b_naive)
                records.append({
                    "mode": feature_mode,
                    "fold": fold,
                    "origin": origin,
                    "model": name,
                    **res,
                })
                print(f"    {name:36s} | Unc13={res['unc_mean13']:.6f} | Clip13={res['clip_mean13']:.6f} | UncNv={res['unc_naive']:.6f} | SMAPE={res['smape']:.2f}% | Bias={res['bias']:.2f}% | RMSE={res['rmse']:.4f}")

        df_mode = pd.DataFrame(records)
        out_csv = OUTPUT_DIR / f"selection_results_{feature_mode}_features.csv"
        df_mode.to_csv(out_csv, index=False)

        print(f"\n--- SUMMARY FOR {feature_mode} FEATURES (Selection Folds 4, 3, 2 Mean) ---")
        piv = df_mode.groupby("model").agg({
            "unc_mean13": "mean",
            "clip_mean13": "mean",
            "unc_naive": "mean",
            "clip_naive": "mean",
            "smape": "mean",
            "bias": "mean",
            "wape": "mean",
            "rmse": "mean",
            "rmsle": "mean",
        })
        print(piv.to_string())


if __name__ == "__main__":
    main()
