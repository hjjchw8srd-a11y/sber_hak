#!/usr/bin/env python3
"""
build_champion_bundle_and_submissions.py

Trains production models on test origin 134 for All-Stock Ensemble (29 features):
1. LightGBM B (29 features, 52 origins 74..125)
2. CatBoost (29 features, 18 origins 108..125)
3. Per-Category (29 features, 14 models, 52 origins 74..125)
4. Exports bundle to experiments/model_improvements/champion_bundle/
5. Generates two submissions:
   - submissions/champion_clipped_score.csv (Pure Gated Blend)
   - submissions/champion_unclipped_rank.csv (With Activity-Conditioned Series Reconciler)
6. Generates JSON manifests and validates both files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

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
)
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    apply_p,
    LGBM_PARAMS,
    CATBOOST_PARAMS,
    PER_CAT_PARAMS,
)
from evaluate_task3_check_and_all_folds import reconcile_series

warnings.filterwarnings("ignore", category=RuntimeWarning)

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "champion_bundle"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


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
    print("BUILDING CHAMPION BUNDLE AND FINAL SUBMISSIONS (29 STOCK FEATURES)")
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

    latest_train = test_origin - HORIZON  # 125
    lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)  # 74..125
    cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)   # 108..125

    print(f"Building 29-feature matrices for test origin {test_origin}...")
    X_train_lgb, y_train_lgb, X_test, test_df, test_pos = build_feature_matrices(matrices, test_origin, lgbm_origins, mode="29")
    X_train_cb, y_train_cb, _, _, _ = build_feature_matrices(matrices, test_origin, cb_origins, mode="29")

    feature_cols = [c for c in X_test.columns if c not in ["category_lvl_1"]]
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]
    cols_without_cat1 = [c for c in feature_cols if c != "category_lvl_1"]

    # --- MODEL 1: LightGBM B (29 features) ---
    print("\n--- Training LightGBM B (29 features, 52 origins) ---")
    t0 = time.perf_counter()
    X_tr_lgb = X_train_lgb[feature_cols].copy()
    X_te_lgb = X_test[feature_cols].copy()
    for c in cat_cols:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")

    clf_lgb = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf_lgb.fit(X_tr_lgb, (y_train_lgb > 0).astype(int))
    p_sale_lgb = clf_lgb.predict_proba(X_te_lgb)[:, 1]

    pos_mask_lgb = y_train_lgb > 0
    reg_lgb = lgb.LGBMRegressor(**LGBM_PARAMS)
    reg_lgb.fit(X_tr_lgb.loc[pos_mask_lgb], np.log1p(y_train_lgb[pos_mask_lgb]))
    cond_lgb = np.maximum(np.expm1(reg_lgb.predict(X_te_lgb)), 0.0)
    exp_lgb = p_sale_lgb * cond_lgb
    gated_lgb = np.where(p_sale_lgb < 0.5, 0.0, exp_lgb)
    print(f"LGBM B trained in {time.perf_counter() - t0:.1f}s")

    # --- MODEL 2: CatBoost (29 features) ---
    print("\n--- Training CatBoost (29 features, 18 origins) ---")
    t1 = time.perf_counter()
    X_tr_cb = X_train_cb[feature_cols].copy()
    X_te_cb = X_test[feature_cols].copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("unknown")
        X_te_cb[c] = X_te_cb[c].astype(str).fillna("unknown")

    cb_reg = CatBoostRegressor(**CATBOOST_PARAMS, cat_features=cat_cols)
    cb_reg.fit(X_tr_cb, y_train_cb)
    pred_cb = np.maximum(cb_reg.predict(X_te_cb), 0.0)
    print(f"CatBoost trained in {time.perf_counter() - t1:.1f}s")

    # --- MODEL 3: Per-Category (29 features, 14 models) ---
    print("\n--- Training Per-Category (29 features, 14 models) ---")
    t2 = time.perf_counter()
    pcat_preds = np.zeros(len(X_test), dtype=np.float32)
    cat_models = {}

    for cat in X_test["category_lvl_1"].unique():
        tr_mask = X_train_lgb["category_lvl_1"] == cat
        val_mask = X_test["category_lvl_1"] == cat

        X_tr_cat = X_train_lgb.loc[tr_mask, cols_without_cat1].copy()
        y_tr_cat = y_train_lgb[tr_mask]
        X_v_cat = X_test.loc[val_mask, cols_without_cat1].copy()

        if len(X_v_cat) == 0:
            continue

        pos_mask = y_tr_cat > 0
        if pos_mask.sum() < 20:
            fallback_val = max(0.0, float(np.mean(y_tr_cat))) if len(y_tr_cat) > 0 else 0.0
            pcat_preds[val_mask] = fallback_val
            cat_models[str(cat)] = {"fallback": fallback_val}
            continue

        for c in cat_cols:
            if c != "category_lvl_1":
                X_tr_cat[c] = X_tr_cat[c].astype("category")
                X_v_cat[c] = X_v_cat[c].astype("category")

        cat_clf = lgb.LGBMClassifier(**PER_CAT_PARAMS)
        cat_clf.fit(X_tr_cat, (y_tr_cat > 0).astype(int))
        p_sale_cat = cat_clf.predict_proba(X_v_cat)[:, 1]

        cat_reg = lgb.LGBMRegressor(**PER_CAT_PARAMS)
        cat_reg.fit(X_tr_cat.loc[pos_mask], np.log1p(y_tr_cat[pos_mask]))
        cond_cat = np.maximum(np.expm1(cat_reg.predict(X_v_cat)), 0.0)
        exp_cat = p_sale_cat * cond_cat
        pcat_preds[val_mask] = np.where(p_sale_cat < 0.5, 0.0, exp_cat)
        cat_models[str(cat)] = {"clf": cat_clf, "reg": cat_reg}

    print(f"Per-Category trained in {time.perf_counter() - t2:.1f}s")

    # --- LOAD TFT PREDICTIONS ---
    tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_prediction_raw.csv.gz"
    if not tft_path.exists():
        tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_submission.csv"
    print(f"\nLoading TFT test predictions from {tft_path}...")
    df_tft = pd.read_csv(tft_path)
    df_tft["week"] = pd.to_datetime(df_tft["week"]).dt.strftime("%Y-%m-%d")
    df_tft = df_tft.rename(columns={"quantity": "raw_tft"})

    # Grid keys
    test_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in test_pos]
    test_df["article"] = test_df["article"].astype(df_tft["article"].dtype)
    test_df["stock_lgbm_gate"] = gated_lgb
    test_df["stock_cb_gate"] = pred_cb
    test_df["stock_pcat_gate"] = pcat_preds

    merged = test_df[["week", "warehouse_name", "article", "stock_lgbm_gate", "stock_cb_gate", "stock_pcat_gate"]].merge(
        df_tft[["week", "warehouse_name", "article", "raw_tft"]], on=KEYS, how="left"
    )

    # 4. RAW BLEND
    raw_blend = (
        0.45 * merged["raw_tft"]
        + 0.37 * merged["stock_lgbm_gate"]
        + 0.09 * merged["stock_cb_gate"]
        + 0.09 * merged["stock_pcat_gate"]
    )
    pred_clipped = apply_p(raw_blend, 1.05, 0.65)

    # 5. ACTIVITY-CONDITIONED RECONCILER
    hist_mean = matrices.quantity[test_origin-13:test_origin].mean(axis=0)
    hist_nonzero = (matrices.quantity[test_origin-13:test_origin] > 0).mean(axis=0)
    pair_df = matrices.pairs.copy()
    pair_df["hist_mean"] = hist_mean
    pair_df["hist_nonzero"] = hist_nonzero
    merged_rec = merged.merge(pair_df, on=["warehouse_name", "article"], how="left")
    merged_rec["series_id"] = merged_rec["warehouse_name"] + "__" + merged_rec["article"].astype(str)
    merged_rec["pred_base"] = pred_clipped
    merged_rec["raw_blend"] = raw_blend

    pred_unclipped = reconcile_series(merged_rec, "pred_base", "raw_blend", tau=2.0, min_nz=0.20, min_m=0.10)

    # --- SAVE SUBMISSIONS ---
    # Submission 1: Champion Clipped Score
    sub_clip = merged[KEYS].copy()
    sub_clip["quantity"] = pred_clipped
    validate_submission(sub_clip)
    clip_csv = SUBMISSIONS_DIR / "champion_clipped_score.csv"
    sub_clip.to_csv(clip_csv, index=False)
    sha_clip = compute_sha256(clip_csv)

    manifest_clip = {
        "candidate_name": "champion_clipped_score",
        "description": "All-Stock 29-feature Ensemble (45 TFT, 37 LGBM, 9 CB, 9 PerCat). Absolute best Clipped Score (0.729252) and lowest SMAPE (61.92%).",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "destination_file": clip_csv.name,
        "sha256": sha_clip,
        "rows": len(sub_clip),
        "total_volume": float(sub_clip["quantity"].sum()),
        "non_zero_rows": int((sub_clip["quantity"] > 0).sum()),
        "non_zero_rate": float((sub_clip["quantity"] > 0).mean()),
        "weights": {"tft": 0.45, "lightgbm_b": 0.37, "catboost": 0.09, "per_category": 0.09},
        "features_count": len(feature_cols),
        "feature_columns": feature_cols,
        "cv_performance_5_folds": {
            "clipped_score_mean13": 0.729252,
            "unclipped_ratio_mean13": 0.754471,
            "score_naive_last": 0.779110,
            "smape": 61.918,
            "bias": 75.642,
            "wape": 49.467,
            "rmse": 2.8877,
        },
    }
    with open(SUBMISSIONS_DIR / "champion_clipped_score_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_clip, f, indent=2, ensure_ascii=False)

    # Submission 2: Champion Unclipped Rank
    sub_unc = merged[KEYS].copy()
    sub_unc["quantity"] = pred_unclipped
    validate_submission(sub_unc)
    unc_csv = SUBMISSIONS_DIR / "champion_unclipped_rank.csv"
    sub_unc.to_csv(unc_csv, index=False)
    sha_unc = compute_sha256(unc_csv)

    manifest_unc = {
        "candidate_name": "champion_unclipped_rank",
        "description": "All-Stock 29-feature Ensemble + Activity-Conditioned Reconciler. Best Unclipped Ratio (0.752658), lowest Naive Last score (0.778697), lowest Bias (74.30%), and lowest RMSE (2.8868).",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "destination_file": unc_csv.name,
        "sha256": sha_unc,
        "rows": len(sub_unc),
        "total_volume": float(sub_unc["quantity"].sum()),
        "non_zero_rows": int((sub_unc["quantity"] > 0).sum()),
        "non_zero_rate": float((sub_unc["quantity"] > 0).mean()),
        "weights": {"tft": 0.45, "lightgbm_b": 0.37, "catboost": 0.09, "per_category": 0.09},
        "features_count": len(feature_cols),
        "feature_columns": feature_cols,
        "series_reconciliation": {"tau": 2.0, "min_hist_nonzero": 0.20, "min_hist_mean": 0.10},
        "cv_performance_5_folds": {
            "clipped_score_mean13": 0.729715,
            "unclipped_ratio_mean13": 0.752658,
            "score_naive_last": 0.778697,
            "smape": 62.042,
            "bias": 74.301,
            "wape": 49.468,
            "rmse": 2.8868,
        },
    }
    with open(SUBMISSIONS_DIR / "champion_unclipped_rank_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_unc, f, indent=2, ensure_ascii=False)

    # --- EXPORT PRODUCTION BUNDLE ---
    print("\n--- Exporting Production Model Bundle ---")
    lgbm_dir = BUNDLE_DIR / "lgbm_b"
    cb_dir = BUNDLE_DIR / "catboost"
    pcat_dir = BUNDLE_DIR / "per_category"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    cb_dir.mkdir(parents=True, exist_ok=True)
    pcat_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM B
    (lgbm_dir / "clf.txt").write_text(clf_lgb.booster_.model_to_string(), encoding="utf-8")
    (lgbm_dir / "reg.txt").write_text(reg_lgb.booster_.model_to_string(), encoding="utf-8")

    # Save CatBoost
    cb_reg.save_model(str(cb_dir / "reg.cbm"))

    # Save Per-Category
    pcat_meta = {}
    for cat_name, m_dict in cat_models.items():
        cat_slug = "".join(c if c.isalnum() else "_" for c in str(cat_name))
        if "fallback" in m_dict:
            pcat_meta[str(cat_name)] = {"fallback": m_dict["fallback"]}
        else:
            clf_name = f"{cat_slug}_clf.txt"
            reg_name = f"{cat_slug}_reg.txt"
            (pcat_dir / clf_name).write_text(m_dict["clf"].booster_.model_to_string(), encoding="utf-8")
            (pcat_dir / reg_name).write_text(m_dict["reg"].booster_.model_to_string(), encoding="utf-8")
            pcat_meta[str(cat_name)] = {"clf_file": clf_name, "reg_file": reg_name}

    with open(pcat_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(pcat_meta, f, indent=2, ensure_ascii=False)

    # Save Valid Pairs
    pairs_df.to_parquet(BUNDLE_DIR / "valid_pairs.parquet", index=False)

    # Copy TFT from existing bundle
    existing_tft = PROJECT_ROOT / "forecast-container" / "model" / "tft"
    tft_dest = BUNDLE_DIR / "tft"
    if tft_dest.exists():
        shutil.rmtree(tft_dest)
    shutil.copytree(existing_tft, tft_dest)

    # Manifest and Postprocessing
    bundle_manifest = {
        "model_name": "champion_all_stock_ensemble",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features_count": len(feature_cols),
        "feature_columns": feature_cols,
        "cat_columns": cat_cols,
        "models": {
            "tft": {"type": "NeuralForecast"},
            "lgbm_b": {"type": "LightGBM", "origins": 52},
            "catboost": {"type": "CatBoost", "origins": 18},
            "per_category": {"type": "PerCategory_LightGBM", "categories": len(pcat_meta)},
        },
    }
    with open(BUNDLE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(bundle_manifest, f, indent=2, ensure_ascii=False)

    bundle_postprocessing = {
        "components": {
            "tft": {"weight": 0.45},
            "lightgbm_b": {"weight": 0.37},
            "catboost": {"weight": 0.09},
            "per_category": {"weight": 0.09},
        },
        "blend": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0,
        },
        "series_reconciliation": {
            "enabled": False,
            "tau": 2.0,
            "min_hist_nonzero": 0.20,
            "min_hist_mean": 0.10,
        },
    }
    with open(BUNDLE_DIR / "postprocessing.json", "w", encoding="utf-8") as f:
        json.dump(bundle_postprocessing, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("CHAMPION SUBMISSIONS AND BUNDLE EXPORT COMPLETE:")
    print(f"1. Champion Clipped Score: {clip_csv}")
    print(f"   SHA-256: {sha_clip}")
    print(f"   Total Volume: {sub_clip['quantity'].sum():,.0f} | Non-zero: {(sub_clip['quantity'] > 0).mean():.2%}")
    print(f"2. Champion Unclipped Rank: {unc_csv}")
    print(f"   SHA-256: {sha_unc}")
    print(f"   Total Volume: {sub_unc['quantity'].sum():,.0f} | Non-zero: {(sub_unc['quantity'] > 0).mean():.2%}")
    print(f"3. Bundle exported to: {BUNDLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
