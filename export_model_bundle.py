#!/usr/bin/env python3
"""
export_model_bundle.py

Offline tool to export all trained models and static artifacts for the
candidate_lgbm52 ensemble into a self-contained bundle for runtime inference.

Outputs:
- model/tft/ (NeuralForecast TFT checkpoint without historical dataset)
- model/lgbm_b/ (LightGBM B classifier and regressor boosters)
- model/catboost/ (CatBoost classifier and regressor native models)
- model/per_category/ (Category-level LightGBM boosters and routing metadata)
- model/valid_pairs.parquet (Canonical 8,789 warehouse-article pairs with metadata)
- model/postprocessing.json (Thresholds, scaling, blending parameters)
- model/manifest.json (SHA-256 hashes, source modules, training parameters)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from neuralforecast import NeuralForecast

PROJECT_ROOT = Path(__file__).resolve().parent

# Canonical feature definitions
from sber_hak.run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(
    data_path: Path,
    tft_model_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    start_time = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tft").mkdir(exist_ok=True)
    (output_dir / "lgbm_b").mkdir(exist_ok=True)
    (output_dir / "catboost").mkdir(exist_ok=True)
    (output_dir / "per_category").mkdir(exist_ok=True)

    print(f"Exporting model bundle to: {output_dir}")

    # 1. Export TFT without historical dataset
    print("\n[1/6] Exporting TFT model without dataset...")
    nf = NeuralForecast.load(str(tft_model_path))
    nf.save(str(output_dir / "tft"), save_dataset=False, overwrite=True)

    # 2. Prepare dataset and canonical pairs
    print("\n[2/6] Loading weekly dataset and extracting canonical pairs...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    first_week = data[data["week"] == data["week"].min()].sort_values(["warehouse_name", "article"])
    pairs_df = first_week[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()
    pairs_df.to_parquet(output_dir / "valid_pairs.parquet", index=False)
    print(f"  Canonical pairs: {len(pairs_df)} (11 warehouses x 799 articles)")

    # Build extended data for WeeklyMatrices
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

    extended_data = pd.concat([data, *future_rows], ignore_index=True)
    matrices = WeeklyMatrices(extended_data)
    test_origin = len(data["week"].unique())  # 134

    feature_cols = columns_for_experiment("base_calendar_network")
    cat_cols = [c for c in CAT_COLUMNS if c in feature_cols]

    # 3. Train and export LightGBM B (52 origins: 74..125)
    print("\n[3/6] Training and exporting LightGBM B (52 origins)...")
    latest_train_origin = test_origin - HORIZON
    earliest_train_origin_b = max(26, latest_train_origin - 51)
    train_origins_b = np.arange(earliest_train_origin_b, latest_train_origin + 1)
    print(f"  Origins: {train_origins_b[0]}..{train_origins_b[-1]} ({len(train_origins_b)} origins)")

    train_df_b, target_arr_b, _ = matrices.make_rows(train_origins_b, 350_000, seed=seed)
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
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }

    clf_b = lgb.LGBMClassifier(**lgb_params)
    clf_b.fit(X_tr_b, (target_arr_b > 0).astype(int))
    (output_dir / "lgbm_b" / "clf.txt").write_text(clf_b.booster_.model_to_string(), encoding="utf-8")

    pos_idx_b = target_arr_b > 0
    reg_b = lgb.LGBMRegressor(**lgb_params)
    reg_b.fit(X_tr_b.loc[pos_idx_b], np.log1p(target_arr_b[pos_idx_b]))
    (output_dir / "lgbm_b" / "reg.txt").write_text(reg_b.booster_.model_to_string(), encoding="utf-8")

    # 4. Train and export CatBoost (18 origins: 108..125)
    print("\n[4/6] Training and exporting CatBoost (18 origins)...")
    earliest_train_origin_cb = max(26, latest_train_origin - 17)
    train_origins_cb = np.arange(earliest_train_origin_cb, latest_train_origin + 1)
    print(f"  Origins: {train_origins_cb[0]}..{train_origins_cb[-1]} ({len(train_origins_cb)} origins)")

    train_df_cb, target_arr_cb, _ = matrices.make_rows(train_origins_cb, 350_000, seed=seed)
    cb_common = {
        "iterations": 160,
        "depth": 7,
        "learning_rate": 0.08,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    clf_cb = CatBoostClassifier(loss_function="Logloss", **cb_common)
    clf_cb.fit(train_df_cb[feature_cols], (target_arr_cb > 0).astype(np.int8), cat_features=cat_cols)
    clf_cb.save_model(str(output_dir / "catboost" / "clf.cbm"))

    pos_cb = target_arr_cb > 0
    reg_cb = CatBoostRegressor(loss_function="RMSE", **cb_common)
    reg_cb.fit(train_df_cb.loc[pos_cb, feature_cols], np.log1p(target_arr_cb[pos_cb]), cat_features=cat_cols)
    reg_cb.save_model(str(output_dir / "catboost" / "reg.cbm"))

    # 5. Train and export Per-Category Models
    print("\n[5/6] Training and exporting Per-Category Models...")
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
            print(f"  Category '{cat}': fallback constant = {fallback_val}")
        else:
            cat_params = {
                "n_estimators": 140,
                "learning_rate": 0.06,
                "num_leaves": 25,
                "min_child_samples": 20,
                "random_state": seed,
                "verbosity": -1,
                "n_jobs": -1,
            }
            c_clf = lgb.LGBMClassifier(**cat_params)
            c_clf.fit(X_tr[cat_feature_cols], (y_tr > 0).astype(int))
            (output_dir / "per_category" / f"{cat_safe}_clf.txt").write_text(c_clf.booster_.model_to_string(), encoding="utf-8")

            c_reg = lgb.LGBMRegressor(**cat_params)
            c_reg.fit(X_tr.loc[pos_mask, cat_feature_cols], np.log1p(y_tr[pos_mask]))
            (output_dir / "per_category" / f"{cat_safe}_reg.txt").write_text(c_reg.booster_.model_to_string(), encoding="utf-8")

            cat_meta[cat] = {
                "clf_file": f"{cat_safe}_clf.txt",
                "reg_file": f"{cat_safe}_reg.txt",
            }

    with open(output_dir / "per_category" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(cat_meta, f, indent=2, ensure_ascii=False)

    # 6. Postprocessing configuration
    print("\n[6/6] Writing postprocessing and manifest...")
    postprocessing_config = {
        "components": {
            "tft": {
                "weight": 0.50,
                "type": "raw_continuous",
                "description": "Temporal Fusion Transformer continuous prediction",
            },
            "lightgbm_b": {
                "weight": 0.30,
                "type": "two_stage_hurdle",
                "prob_gate": 0.50,
                "description": "LightGBM B trained on 52 origins with prob_gate 0.5",
            },
            "catboost": {
                "weight": 0.10,
                "type": "raw_two_stage",
                "prob_gate": 0.50,
                "description": "CatBoost trained on 18 origins, self-gated, treated as raw in blend",
            },
            "per_category": {
                "weight": 0.10,
                "type": "category_hurdle_postprocessed",
                "threshold": 0.75,
                "rounding": "rint",
                "description": "14 Category-level models with comp threshold 0.75 and rint",
            },
        },
        "blend": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0.0,
        },
        "calendar_anchor": "2024-01-01",
        "horizon_weeks": 9,
    }
    with open(output_dir / "postprocessing.json", "w", encoding="utf-8") as f:
        json.dump(postprocessing_config, f, indent=2, ensure_ascii=False)

    # Compute bundle file checksums
    file_hashes = {}
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = str(p.relative_to(output_dir)).replace("\\", "/")
            file_hashes[rel] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    manifest = {
        "bundle_name": "candidate_lgbm52_bundle",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_cutoff_date": "2026-07-26",
        "reference_candidate": {
            "name": "candidate_lgbm52",
            "destination_file": "submissions/candidate_lgbm52.csv",
            "sha256": "d8ff734c962bc782987f750962e6110fb6204ed4cacc7874048d160bd7ad1f0f",
            "row_count": 79101,
            "total_quantity": 210453,
        },
        "components": {
            "tft": {
                "framework": "NeuralForecast",
                "model": "TFT",
                "input_size": 26,
                "horizon": 9,
                "dataset_stripped": True,
            },
            "lightgbm_b": {
                "framework": "LightGBM",
                "version": lgb.__version__,
                "origins": 52,
                "origin_range": [int(train_origins_b[0]), int(train_origins_b[-1])],
            },
            "catboost": {
                "framework": "CatBoost",
                "origins": 18,
                "origin_range": [int(train_origins_cb[0]), int(train_origins_cb[-1])],
            },
            "per_category": {
                "framework": "LightGBM",
                "num_categories": len(unique_categories),
                "categories": unique_categories,
            },
        },
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
        "files": file_hashes,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - start_time
    print(f"\nBundle export complete in {elapsed:.2f} seconds!")
    print(f"Total files in bundle: {len(file_hashes)}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz")
    parser.add_argument("--tft-model", type=Path, default=PROJECT_ROOT / "experiments" / "tft" / "final_model")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "forecast-container" / "model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_bundle(args.data, args.tft_model, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
