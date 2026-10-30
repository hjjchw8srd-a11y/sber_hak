#!/usr/bin/env python3
"""
run_real_fold0_ablation.py

Honest, empirical 180-day history ablation on Fold 0:
1. Builds a dedicated Fold 0 bundle trained strictly on data prior to Fold 0 cutoff (2026-05-24, origin 125).
   - TFT trained on temporal data up to 2026-05-24 (origin 125)
   - LightGBM B trained on 52 origins (65..116)
   - CatBoost trained on 18 origins (99..116)
   - Per-Category trained on 18 origins (99..116)
2. Prepares two real inputs from raw daily files:
   - Input Full: 2024-01-01 .. 2026-05-24 (875 days)
   - Input 180d: 2025-11-26 .. 2026-05-24 (exactly 180 calendar days)
3. Independently runs the real production Predictor on both inputs with NO caching.
4. Calculates all 5 official metrics, clipped score, unclipped ratio, differencing rows,
   sums, zero rate, and execution times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import TFT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))
sys.path.insert(0, str(PROJECT_ROOT))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
)
from train_tft_weekly import (
    BASE_HIST_EXOG,
    FUTR_EXOG,
    baseline_mean13,
    make_model,
    prepare_frames,
)
from app.predictor import Predictor
from app.validators import validate_and_load_sales, validate_and_load_stock

FOLD_0_ORIGIN = 125
FOLD_0_CUTOFF = pd.Timestamp("2026-05-24")
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "reviews" / "fold_0_bundle"


def prepare_fold0_bundle(matrices: WeeklyMatrices, data: pd.DataFrame) -> Path:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "tft").mkdir(exist_ok=True)
    (BUNDLE_DIR / "lgbm_b").mkdir(exist_ok=True)
    (BUNDLE_DIR / "catboost").mkdir(exist_ok=True)
    (BUNDLE_DIR / "per_category").mkdir(exist_ok=True)

    manifest_path = BUNDLE_DIR / "manifest.json"
    if (BUNDLE_DIR / "lgbm_b" / "clf.txt").exists() and (BUNDLE_DIR / "tft" / "TFT_0.safetensors").exists():
        print(f"Fold 0 bundle already exists at {BUNDLE_DIR}, reusing...")
        return BUNDLE_DIR

    print(f"\n--- Building Fold 0 Model Bundle (strictly trained up to {FOLD_0_CUTOFF.date()}) ---")

    # 1. Canonical pairs
    first_week = data[data["week"] == data["week"].min()].sort_values(["warehouse_name", "article"])
    pairs_df = first_week[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()
    pairs_df.to_parquet(BUNDLE_DIR / "valid_pairs.parquet", index=False)

    # 2. Train and export Fold 0 TFT (origins < 125)
    print("Training Fold 0 TFT (temporal data < 2026-05-25)...")
    data_f0 = data[data["week"] < pd.Timestamp("2026-05-25")].copy()
    temporal, static_df, static_columns = prepare_frames(data_f0)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    tft_args = argparse.Namespace(
        input_size=26,
        hidden_size=32,
        max_steps=300,
        batch_size=128,
        windows_batch_size=512,
        learning_rate=1e-3,
        val_check_steps=50,
        patience=-1,
        seed=42,
    )
    tft = make_model(tft_args, static_columns, device, hist_exog_list=BASE_HIST_EXOG)
    nf = NeuralForecast(models=[tft], freq="W-MON")
    nf.fit(df=temporal, static_df=static_df, val_size=0)
    nf.save(str(BUNDLE_DIR / "tft"), save_dataset=False, overwrite=True)
    print("Fold 0 TFT saved!")

    # 3. LightGBM B (origins 65..116)
    print("Training Fold 0 LightGBM B (origins 65..116)...")
    feature_cols = columns_for_experiment("base_calendar_network")
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]

    latest_train_origin = FOLD_0_ORIGIN - HORIZON  # 116
    earliest_train_origin_b = max(26, latest_train_origin - 51)  # 65
    train_origins_b = np.arange(earliest_train_origin_b, latest_train_origin + 1)

    train_df_b, target_arr_b, _ = matrices.make_rows(train_origins_b, 350_000, seed=42)
    X_tr_b = train_df_b[feature_cols].copy()
    for col in cat_cols:
        X_tr_b[col] = X_tr_b[col].astype("category")

    lgb_params = {
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
    clf_b = lgb.LGBMClassifier(**lgb_params)
    clf_b.fit(X_tr_b, (target_arr_b > 0).astype(int))
    (BUNDLE_DIR / "lgbm_b" / "clf.txt").write_text(clf_b.booster_.model_to_string(), encoding="utf-8")

    pos_idx_b = target_arr_b > 0
    reg_b = lgb.LGBMRegressor(**lgb_params)
    reg_b.fit(X_tr_b.loc[pos_idx_b], np.log1p(target_arr_b[pos_idx_b]))
    (BUNDLE_DIR / "lgbm_b" / "reg.txt").write_text(reg_b.booster_.model_to_string(), encoding="utf-8")

    # 4. CatBoost (origins 99..116)
    print("Training Fold 0 CatBoost (origins 99..116)...")
    earliest_train_origin_cb = max(26, latest_train_origin - 17)  # 99
    train_origins_cb = np.arange(earliest_train_origin_cb, latest_train_origin + 1)

    train_df_cb, target_arr_cb, _ = matrices.make_rows(train_origins_cb, 350_000, seed=42)
    cb_common = {
        "iterations": 160,
        "depth": 7,
        "learning_rate": 0.08,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    clf_cb = CatBoostClassifier(loss_function="Logloss", **cb_common)
    clf_cb.fit(train_df_cb[feature_cols], (target_arr_cb > 0).astype(np.int8), cat_features=cat_cols)
    clf_cb.save_model(str(BUNDLE_DIR / "catboost" / "clf.cbm"))

    pos_cb = target_arr_cb > 0
    reg_cb = CatBoostRegressor(loss_function="RMSE", **cb_common)
    reg_cb.fit(train_df_cb.loc[pos_cb, feature_cols], np.log1p(target_arr_cb[pos_cb]), cat_features=cat_cols)
    reg_cb.save_model(str(BUNDLE_DIR / "catboost" / "reg.cbm"))

    # 5. Per-Category (origins 99..116)
    print("Training Fold 0 Per-Category Models...")
    cat_feature_cols = [c for c in feature_cols if c != "category_lvl_1"]
    cat_cat_cols = [c for c in CAT_COLUMNS if c in cat_feature_cols]

    train_df_pcat = train_df_cb.copy()
    for col in cat_cat_cols:
        train_df_pcat[col] = train_df_pcat[col].astype("category")
    y_tr_pcat = target_arr_cb

    unique_categories = sorted(train_df_pcat["category_lvl_1"].unique())
    cat_meta = {}

    for cat in unique_categories:
        train_mask = train_df_pcat["category_lvl_1"] == cat
        X_tr = train_df_pcat.loc[train_mask]
        y_tr = y_tr_pcat[train_mask]
        pos_mask = y_tr > 0
        cat_safe = "".join(c if c.isalnum() else "_" for c in str(cat))

        if pos_mask.sum() < 20:
            fallback_val = max(0.0, float(np.mean(y_tr)))
            cat_meta[cat] = {"fallback": fallback_val}
        else:
            cat_params = {
                "n_estimators": 140,
                "learning_rate": 0.06,
                "num_leaves": 25,
                "min_child_samples": 20,
                "random_state": 42,
                "verbosity": -1,
                "n_jobs": -1,
            }
            c_clf = lgb.LGBMClassifier(**cat_params)
            c_clf.fit(X_tr[cat_feature_cols], (y_tr > 0).astype(int))
            (BUNDLE_DIR / "per_category" / f"{cat_safe}_clf.txt").write_text(c_clf.booster_.model_to_string(), encoding="utf-8")

            c_reg = lgb.LGBMRegressor(**cat_params)
            c_reg.fit(X_tr.loc[pos_mask, cat_feature_cols], np.log1p(y_tr[pos_mask]))
            (BUNDLE_DIR / "per_category" / f"{cat_safe}_reg.txt").write_text(c_reg.booster_.model_to_string(), encoding="utf-8")

            cat_meta[cat] = {
                "clf_file": f"{cat_safe}_clf.txt",
                "reg_file": f"{cat_safe}_reg.txt",
            }

    with open(BUNDLE_DIR / "per_category" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(cat_meta, f, indent=2, ensure_ascii=False)

    postprocessing_config = {
        "components": {
            "tft": {"weight": 0.50, "type": "raw_continuous"},
            "lightgbm_b": {"weight": 0.30, "type": "two_stage_hurdle", "prob_gate": 0.50},
            "catboost": {"weight": 0.10, "type": "raw_two_stage", "prob_gate": 0.50},
            "per_category": {"weight": 0.10, "type": "category_hurdle_postprocessed", "threshold": 0.75, "rounding": "rint"},
        },
        "blend": {"alpha": 1.05, "threshold": 0.65, "rounding": "rint", "clip_lower": 0.0},
        "calendar_anchor": "2024-01-01",
        "horizon_weeks": 9,
    }
    with open(BUNDLE_DIR / "postprocessing.json", "w", encoding="utf-8") as f:
        json.dump(postprocessing_config, f, indent=2, ensure_ascii=False)

    manifest = {
        "bundle_name": "fold_0_bundle",
        "training_cutoff_date": "2026-05-24",
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
    }
    with open(BUNDLE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Fold 0 bundle ready!")
    return BUNDLE_DIR


def compute_metrics_diagnostics(df_truth: pd.DataFrame, df_pred: pd.DataFrame, df_baseline: pd.DataFrame):
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
        "clipped_score": float(clipped_score),
        "unclipped_ratio": float(unclipped_ratio),
    }


def main():
    print("=" * 80)
    print("HONEST EMPIRICAL EXPERIMENT: FULL HISTORY vs EXACT 180 DAYS ON FOLD 0")
    print(f"Cutoff Date: {FOLD_0_CUTOFF.date()} | Target: 2026-05-25 .. 2026-07-20 (9 weeks)")
    print("=" * 80)

    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    val_start_week = pd.Timestamp(weeks[FOLD_0_ORIGIN])
    val_weeks = weeks[FOLD_0_ORIGIN : FOLD_0_ORIGIN + HORIZON]

    truth = make_truth(matrices, FOLD_0_ORIGIN)
    truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")
    baseline = baseline_mean13(data, val_start_week, pd.Index(val_weeks))
    baseline["week"] = pd.to_datetime(baseline["week"]).dt.strftime("%Y-%m-%d")

    # Step 1: Prepare Fold 0 bundle
    bundle_path = prepare_fold0_bundle(matrices, data)
    predictor = Predictor.load(bundle_path)

    # Step 2: Load raw sales and stock files
    print("\nLoading raw sales and stock datasets...")
    sales_raw = pd.read_csv(PROJECT_ROOT / "sales_dataset_hakaton_train.csv", parse_dates=["date"])
    stock_raw = pd.read_csv(PROJECT_ROOT / "stock_dataset_hakaton_train.csv", parse_dates=["date"])

    # Input 1: Full History up to cutoff 2026-05-24
    s_full = sales_raw[sales_raw["date"] <= FOLD_0_CUTOFF].copy()
    st_full = stock_raw[stock_raw["date"] <= FOLD_0_CUTOFF].copy()
    full_days = (s_full["date"].max() - s_full["date"].min()).days + 1
    print(f"\n[Case 1: Full History] Date range: {s_full['date'].min().date()} .. {s_full['date'].max().date()} ({full_days} days)")

    # Input 2: Exactly 180 Calendar Days up to cutoff 2026-05-24
    # (2026-05-24 - 179 days = 2025-11-26)
    start_180d = FOLD_0_CUTOFF - pd.Timedelta(days=179)
    s_180d = s_full[(s_full["date"] >= start_180d) & (s_full["date"] <= FOLD_0_CUTOFF)].copy()
    st_180d = st_full[(st_full["date"] >= start_180d) & (st_full["date"] <= FOLD_0_CUTOFF)].copy()
    d180_days = (s_180d["date"].max() - s_180d["date"].min()).days + 1
    print(f"[Case 2: Exact 180 Days] Date range: {s_180d['date'].min().date()} .. {s_180d['date'].max().date()} ({d180_days} days)")
    assert d180_days == 180, f"Expected exactly 180 days, got {d180_days}"

    # Step 3: Run real predictor on Full History
    print("\nRunning real predictor on Case 1 (Full History)...")
    t0_full = time.perf_counter()
    pred_full = predictor.predict(validate_and_load_sales(s_full), validate_and_load_stock(st_full))
    t_full = time.perf_counter() - t0_full
    print(f"Full history prediction finished in {t_full:.2f}s! Total qty: {pred_full['quantity'].sum()}")

    # Step 4: Run real predictor on Exact 180 Days
    print("\nRunning real predictor on Case 2 (Exact 180 Days)...")
    t0_180 = time.perf_counter()
    pred_180d = predictor.predict(validate_and_load_sales(s_180d), validate_and_load_stock(st_180d))
    t_180d = time.perf_counter() - t0_180
    print(f"180-day prediction finished in {t_180d:.2f}s! Total qty: {pred_180d['quantity'].sum()}")

    # Compare predictions
    diff = np.abs(pred_full["quantity"] - pred_180d["quantity"])
    diff_rows = int((diff > 0).sum())
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())

    m_full = compute_metrics_diagnostics(truth, pred_full, baseline)
    m_180d = compute_metrics_diagnostics(truth, pred_180d, baseline)

    print("\n" + "=" * 80)
    print("REAL EMPIRICAL FOLD 0 COMPARISON TABLE (Measured without caching or OOF mocks)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Full History':<18} | {'Exact 180 Days':<18} | {'Delta (180d - Full)':<20}")
    print("-" * 80)
    print(f"{'Score (Clipped)':<25} | {m_full['clipped_score']:<18.6f} | {m_180d['clipped_score']:<18.6f} | {m_180d['clipped_score'] - m_full['clipped_score']:+.6f}")
    print(f"{'Unclipped Ratio':<25} | {m_full['unclipped_ratio']:<18.6f} | {m_180d['unclipped_ratio']:<18.6f} | {m_180d['unclipped_ratio'] - m_full['unclipped_ratio']:+.6f}")
    print(f"{'SMAPE (%)':<25} | {m_full['smape']:<18.4f} | {m_180d['smape']:<18.4f} | {m_180d['smape'] - m_full['smape']:+.4f}")
    print(f"{'WAPE (%)':<25} | {m_full['wape']:<18.4f} | {m_180d['wape']:<18.4f} | {m_180d['wape'] - m_full['wape']:+.4f}")
    print(f"{'RMSE':<25} | {m_full['rmse']:<18.4f} | {m_180d['rmse']:<18.4f} | {m_180d['rmse'] - m_full['rmse']:+.4f}")
    print(f"{'Bias (%)':<25} | {m_full['bias']:<18.4f} | {m_180d['bias']:<18.4f} | {m_180d['bias'] - m_full['bias']:+.4f}")
    print(f"{'RMSLE':<25} | {m_full['rmsle']:<18.4f} | {m_180d['rmsle']:<18.4f} | {m_180d['rmsle'] - m_full['rmsle']:+.4f}")
    print(f"{'Total Predicted Units':<25} | {int(pred_full['quantity'].sum()):<18} | {int(pred_180d['quantity'].sum()):<18} | {int(pred_180d['quantity'].sum()) - int(pred_full['quantity'].sum()):+d}")
    print(f"{'Zero Rate (%)':<25} | {(pred_full['quantity'] == 0).mean() * 100:<18.2f} | {(pred_180d['quantity'] == 0).mean() * 100:<18.2f} | {((pred_180d['quantity'] == 0).mean() - (pred_full['quantity'] == 0).mean()) * 100:+.2f}")
    print(f"{'Differing Rows':<25} | {'-':<18} | {diff_rows:<18} | {diff_rows} ({diff_rows / len(pred_full) * 100:.2f}%)")
    print(f"{'Max Abs Difference':<25} | {'-':<18} | {max_diff:<18} | {max_diff}")
    print(f"{'Duration (sec)':<25} | {t_full:<18.2f} | {t_180d:<18.2f} | {t_180d - t_full:+.2f}")
    print("=" * 80)

    # Save artifact
    artifact_path = PROJECT_ROOT / "experiments" / "reviews" / "fold_0_real_180d_ablation.json"
    artifact = {
        "cutoff_date": str(FOLD_0_CUTOFF.date()),
        "validation_period": "2026-05-25 .. 2026-07-20",
        "history_days_full": full_days,
        "history_days_180d": d180_days,
        "full_history": {
            "score": m_full["clipped_score"],
            "unclipped_ratio": m_full["unclipped_ratio"],
            "smape": m_full["smape"],
            "wape": m_full["wape"],
            "rmse": m_full["rmse"],
            "bias": m_full["bias"],
            "rmsle": m_full["rmsle"],
            "total_quantity": int(pred_full["quantity"].sum()),
            "zero_rate": float((pred_full["quantity"] == 0).mean()),
            "duration_sec": t_full,
        },
        "exact_180d": {
            "score": m_180d["clipped_score"],
            "unclipped_ratio": m_180d["unclipped_ratio"],
            "smape": m_180d["smape"],
            "wape": m_180d["wape"],
            "rmse": m_180d["rmse"],
            "bias": m_180d["bias"],
            "rmsle": m_180d["rmsle"],
            "total_quantity": int(pred_180d["quantity"].sum()),
            "zero_rate": float((pred_180d["quantity"] == 0).mean()),
            "duration_sec": t_180d,
        },
        "comparison": {
            "differing_rows": diff_rows,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "delta_score": m_180d["clipped_score"] - m_full["clipped_score"],
            "delta_unclipped_ratio": m_180d["unclipped_ratio"] - m_full["unclipped_ratio"],
        },
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nReal ablation artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
