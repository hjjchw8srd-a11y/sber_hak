#!/usr/bin/env python3
"""
generate_candidate_stock_lgbm_submission.py

Builds verified candidate submission 'submissions/candidate_stock_lgbm_reconciled.csv':
- Model: 45% TFT + 37% Stock Dynamics LightGBM (29 features) + 9% CatBoost + 9% Per-Category
- Post-processing: alpha = 1.05, threshold = 0.65, rounding = 'rint'
- Series Reconciliation: Activity-conditioned recovery (tau = 2.0, hist_nonzero >= 0.20, hist_mean >= 0.10)
- Validation: Exactly 79,101 rows, non-negative integers, unique keys
- Manifest: 'submissions/candidate_stock_lgbm_reconciled_manifest.json'
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from score import validate_submission
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    train_and_predict as train_and_predict_cb,
)
from train_lgbm_task3_stock_yearly import compute_extra_features_for_origins, apply_p
from evaluate_task3_check_and_all_folds import reconcile_series

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = SUBMISSIONS_DIR / "candidate_stock_lgbm_reconciled.csv"
MANIFEST_PATH = SUBMISSIONS_DIR / "candidate_stock_lgbm_reconciled_manifest.json"

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


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("GENERATING CANDIDATE SUBMISSION: candidate_stock_lgbm_reconciled.csv")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    first_week = data[data["week"] == data["week"].min()].sort_values(["warehouse_name", "article"])
    pairs_df = first_week[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()

    future_weeks = pd.date_range("2026-07-27", periods=HORIZON, freq="W-MON")
    future_rows = []
    for w in future_weeks:
        chunk = pairs_df.copy()
        chunk["week"] = w
        chunk["quantity"] = 0.0
        chunk["sales_sum"] = 0.0
        chunk["discount_sum"] = 0.0
        chunk["stock_week_end"] = np.nan
        chunk["stock_mean"] = np.nan
        chunk["stock_nonpositive_rate"] = np.nan
        chunk["stock_observed_days"] = 0.0
        future_rows.append(chunk)

    data_extended = pd.concat([data] + future_rows, ignore_index=True)
    matrices = WeeklyMatrices(data_extended)
    test_origin = 134

    # 2. Train Stock Dynamics LightGBM on origin 134
    latest_train_origin = test_origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - 51)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)
    print(f"Training Stock LightGBM on {len(train_origins)} origins (origins {earliest_train_origin}..{latest_train_origin})...")

    t0 = time.perf_counter()
    train_df, target_arr, _ = matrices.make_rows(train_origins, 350_000, seed=42)
    test_df, _, test_pos = matrices.make_rows(np.array([test_origin]), None)

    train_origin_pos = train_df["time_idx"].to_numpy() - train_df["horizon"].to_numpy() + 1
    test_origin_pos = test_df["time_idx"].to_numpy() - test_df["horizon"].to_numpy() + 1

    train_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([train_df["warehouse_name"], train_df["article"]])
    )
    test_pair_idx = matrices.pairs.set_index(["warehouse_name", "article"]).index.get_indexer(
        pd.MultiIndex.from_arrays([test_df["warehouse_name"], test_df["article"]])
    )

    base_cols = columns_for_experiment("base_calendar_network")
    cat_columns = [c for c in CAT_COLUMNS if c in base_cols]

    X_train = train_df[base_cols].copy()
    X_test = test_df[base_cols].copy()

    train_mean13 = train_df["quantity_mean_13"].to_numpy()
    test_mean13 = test_df["quantity_mean_13"].to_numpy()

    train_extra = compute_extra_features_for_origins(matrices, train_origin_pos, train_pair_idx)
    test_extra = compute_extra_features_for_origins(matrices, test_origin_pos, test_pair_idx)

    X_train["stock_end_lag_1"] = train_extra["stock_end_lag_1"]
    X_test["stock_end_lag_1"] = test_extra["stock_end_lag_1"]

    X_train["stock_mean_lag_1"] = train_extra["stock_mean_lag_1"]
    X_test["stock_mean_lag_1"] = test_extra["stock_mean_lag_1"]

    X_train["stock_nonpositive_rate_lag_1"] = train_extra["stock_nonpositive_rate_lag_1"]
    X_test["stock_nonpositive_rate_lag_1"] = test_extra["stock_nonpositive_rate_lag_1"]

    raw_train_stock = np.maximum(matrices.stock_end[train_origin_pos - 1, train_pair_idx], 0.0)
    raw_test_stock = np.maximum(matrices.stock_end[test_origin_pos - 1, test_pair_idx], 0.0)
    X_train["weeks_of_supply"] = np.log1p(raw_train_stock / (train_mean13 + 0.1))
    X_test["weeks_of_supply"] = np.log1p(raw_test_stock / (test_mean13 + 0.1))

    X_train["unconstrained_demand_13"] = train_extra["unconstrained_demand_13"]
    X_test["unconstrained_demand_13"] = test_extra["unconstrained_demand_13"]

    for col in cat_columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    clf = lgb.LGBMClassifier(**PARAMS)
    clf.fit(X_train, (target_arr > 0).astype(int))
    p_sale_test = clf.predict_proba(X_test)[:, 1]

    pos_idx = target_arr > 0
    reg = lgb.LGBMRegressor(**PARAMS)
    reg.fit(X_train.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_test = np.maximum(np.expm1(reg.predict(X_test)), 0.0)
    exp_test = p_sale_test * cond_test
    after_gate_lgbm = np.where(p_sale_test < 0.5, 0.0, exp_test)
    print(f"Stock LightGBM trained and predicted in {time.perf_counter() - t0:.1f}s")

    # 3. Load Remaining Test Components (TFT, CatBoost, Per-Category)
    tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_prediction_raw.csv.gz"
    if not tft_path.exists():
        tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_submission.csv"
    print(f"Loading TFT test predictions from {tft_path}...")
    df_tft = pd.read_csv(tft_path)
    df_tft["week"] = pd.to_datetime(df_tft["week"]).dt.strftime("%Y-%m-%d")
    df_tft = df_tft.rename(columns={"quantity": "raw_tft"})

    per_cat_path = PROJECT_ROOT / "experiments" / "benchmarks" / "per_category" / "per_category_final_submission.csv"
    print(f"Loading Per-Category test predictions from {per_cat_path}...")
    df_per_cat = pd.read_csv(per_cat_path)
    df_per_cat["week"] = pd.to_datetime(df_per_cat["week"]).dt.strftime("%Y-%m-%d")
    df_per_cat = df_per_cat.rename(columns={"quantity": "raw_per_cat"})

    # CatBoost test predictions
    cb_latest_train = test_origin - HORIZON
    cb_earliest_train = max(26, cb_latest_train - 17)
    cb_train_origins = np.arange(cb_earliest_train, cb_latest_train + 1)
    columns_cb = columns_for_experiment("base_calendar_network")

    cb_train_df, cb_target_arr, _ = matrices.make_rows(cb_train_origins, 350_000, seed=42)
    cb_test_df, _, _ = matrices.make_rows(np.array([test_origin]), None)
    cb_vals, _ = train_and_predict_cb(cb_train_df, cb_target_arr, cb_test_df, columns_cb, iterations=160)

    test_weeks_cb = np.repeat(matrices.weeks[test_origin : test_origin + HORIZON].to_numpy(), matrices.n_pairs)
    test_pairs_cb = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
    df_cb = pd.DataFrame({
        "week": pd.to_datetime(test_weeks_cb).strftime("%Y-%m-%d"),
        "warehouse_name": test_pairs_cb["warehouse_name"],
        "article": test_pairs_cb["article"],
        "raw_catboost": np.maximum(cb_vals, 0.0),
    })

    test_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in test_pos]
    test_df["article"] = test_df["article"].astype(df_tft["article"].dtype)
    test_df["stock_lgbm_gate"] = after_gate_lgbm

    # Merge all 4 components
    merged = test_df[["week", "warehouse_name", "article", "stock_lgbm_gate"]].merge(
        df_tft[["week", "warehouse_name", "article", "raw_tft"]], on=KEYS, how="left"
    ).merge(
        df_cb[["week", "warehouse_name", "article", "raw_catboost"]], on=KEYS, how="left"
    ).merge(
        df_per_cat[["week", "warehouse_name", "article", "raw_per_cat"]], on=KEYS, how="left"
    )

    # 4. Raw Ensemble Blend
    raw_blend = (
        0.45 * merged["raw_tft"]
        + 0.37 * merged["stock_lgbm_gate"]
        + 0.09 * merged["raw_catboost"]
        + 0.09 * merged["raw_per_cat"]
    )
    pred_base = apply_p(raw_blend, 1.05, 0.65)

    # 5. Activity-Conditioned Series Reconciliation
    hist_mean = matrices.quantity[test_origin-13:test_origin].mean(axis=0)
    hist_nonzero = (matrices.quantity[test_origin-13:test_origin] > 0).mean(axis=0)
    pair_df = matrices.pairs.copy()
    pair_df["hist_mean"] = hist_mean
    pair_df["hist_nonzero"] = hist_nonzero
    merged = merged.merge(pair_df, on=["warehouse_name", "article"], how="left")
    merged["series_id"] = merged["warehouse_name"] + "__" + merged["article"].astype(str)
    merged["pred_base"] = pred_base
    merged["raw_blend"] = raw_blend

    final_quantity = reconcile_series(merged, "pred_base", "raw_blend", tau=2.0, min_nz=0.20, min_m=0.10)

    # 6. Build and validate submission
    sub = merged[KEYS].copy()
    sub["quantity"] = final_quantity
    validate_submission(sub)

    sub.to_csv(SUBMISSION_PATH, index=False)
    sha = compute_sha256(SUBMISSION_PATH)

    manifest = {
        "submission_file": SUBMISSION_PATH.name,
        "sha256": sha,
        "rows": len(sub),
        "total_volume": float(sub["quantity"].sum()),
        "non_zero_rows": int((sub["quantity"] > 0).sum()),
        "non_zero_rate": float((sub["quantity"] > 0).mean()),
        "weights": {
            "tft": 0.45,
            "stock_lgbm_52": 0.37,
            "catboost_18": 0.09,
            "per_category": 0.09,
        },
        "stock_features": [
            "stock_end_lag_1",
            "stock_mean_lag_1",
            "stock_nonpositive_rate_lag_1",
            "weeks_of_supply",
            "unconstrained_demand_13",
        ],
        "series_reconciliation": {
            "tau": 2.0,
            "min_hist_nonzero": 0.20,
            "min_hist_mean": 0.10,
        },
        "postprocessing": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0.0,
        },
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("CANDIDATE SUBMISSION GENERATED SUCCESSFULLY:")
    print(f"Path: {SUBMISSION_PATH}")
    print(f"SHA-256: {sha}")
    print(f"Rows: {len(sub):,} | Total Volume: {sub['quantity'].sum():,.0f} | Non-zero Rate: {(sub['quantity'] > 0).mean():.2%}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
