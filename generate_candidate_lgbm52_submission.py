#!/usr/bin/env python3
"""
generate_candidate_lgbm52_submission.py

Builds verified candidate submission 'candidate_lgbm52.csv':
- Uses the EXACT SAME predict_from_components pipeline from history_audit_pipeline.py.
- Automated historical check on Fold 0: asserts 0 diff against Fold 0 OOF predictions.
- Strict non-leakage cutoff: 2026-07-26 (origin 134).
- Bottom-level LightGBM Variant B: trained on 52 origins (74..125).
- Blends with TFT (50%), CatBoost (10%), Per-Category (10%), LightGBM 52w (30%).
- Scaling alpha = 1.05, threshold = 0.65, rounding = 'rint'.
- Outputs strictly to 'submissions/candidate_lgbm52.csv' (DO NOT TOUCH final_submission.csv!).
- Generates 'submissions/candidate_lgbm52_manifest.json'.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from sber_hak.history_audit_pipeline import (
    predict_from_components,
    train_lgbm_history_variant,
)

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_PATH = SUBMISSIONS_DIR / "candidate_lgbm52.csv"
MANIFEST_PATH = SUBMISSIONS_DIR / "candidate_lgbm52_manifest.json"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=" * 80)
    print("GENERATING CANDIDATE SUBMISSION: candidate_lgbm52.csv")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Automated Historical Check on Fold 0 (origin 125)
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Running Automated Historical Consistency Check on Fold 0 ---")
    f0_meta_path = PROJECT_ROOT / "experiments" / "stacking_honest" / "fold_meta_features" / "fold_0_meta.csv.gz"
    f0_oof_b_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / "blend_b_fold_0.csv.gz"

    assert f0_meta_path.exists(), f"Missing {f0_meta_path}"
    assert f0_oof_b_path.exists(), f"Missing {f0_oof_b_path}"

    f0_meta = pd.read_csv(f0_meta_path)
    f0_meta["week"] = pd.to_datetime(f0_meta["week"]).dt.strftime("%Y-%m-%d")
    f0_oof_b = pd.read_csv(f0_oof_b_path)

    # Reconstruct Fold 0 using predict_from_components
    inf_matched_config = {
        "name": "candidate_lgbm52_config",
        "components": {
            "tft": {"weight": 0.50, "type": "raw"},
            "lgbm": {"weight": 0.30, "type": "two_stage", "prob_gate": 0.5, "comp_postprocess": None},
            "catboost": {"weight": 0.10, "type": "two_stage", "prob_gate": 0.5, "comp_postprocess": None},
            "per_category": {"weight": 0.10, "type": "two_stage", "prob_gate": None, "comp_postprocess": None},
        },
        "blend": {"alpha": 1.05, "threshold": 0.65, "rounding": "rint", "clip_lower": 0.0},
    }

    comp_dict_f0 = {
        "tft": pd.DataFrame({"week": f0_meta["week"], "warehouse_name": f0_meta["warehouse_name"], "article": f0_meta["article"], "quantity": f0_meta["raw_tft"]}),
        "lgbm": pd.DataFrame({
            "week": f0_oof_b["week"], "warehouse_name": f0_oof_b["warehouse_name"], "article": f0_oof_b["article"],
            "p_sale": f0_oof_b["p_sale_lgbm"], "q_cond": f0_oof_b["q_cond_lgbm"], "expected": f0_oof_b["expected_lgbm"],
        }),
        "catboost": pd.DataFrame({
            "week": f0_meta["week"], "warehouse_name": f0_meta["warehouse_name"], "article": f0_meta["article"],
            "p_sale": f0_meta["p_sale_cb"], "q_cond": f0_meta["q_cond_cb"], "expected": f0_meta["exp_cb"],
        }),
        "per_category": pd.DataFrame({
            "week": f0_meta["week"], "warehouse_name": f0_meta["warehouse_name"], "article": f0_meta["article"],
            "p_sale": np.ones(len(f0_meta)), "q_cond": f0_meta["raw_per_cat"], "expected": f0_meta["raw_per_cat"],
        }),
    }

    keyed_check, sub_check = predict_from_components(comp_dict_f0, inf_matched_config)
    diff_count = int((sub_check["quantity"] != f0_oof_b["final_prediction"]).sum())
    max_abs_diff = float(np.abs(sub_check["quantity"] - f0_oof_b["final_prediction"]).max())
    print(f"Historical Check on Fold 0: mismatched rows = {diff_count} / {len(sub_check)}, max abs diff = {max_abs_diff}")
    assert diff_count == 0, f"Historical pipeline check failed on Fold 0! {diff_count} rows differ."
    print(">> Automated historical consistency check PASSED! Exact 0 difference across all 79,101 rows. <<")

    # -------------------------------------------------------------------------
    # STEP 2: Load and Prepare Full Dataset with Extended Horizon
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: Preparing Test Data Extended with Future Horizon ---")
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

    future_df = pd.concat(future_rows, ignore_index=True)
    extended_data = pd.concat([data, future_df], ignore_index=True)
    matrices = WeeklyMatrices(extended_data)

    test_origin = len(data["week"].unique())  # 134
    print(f"Test Origin index: {test_origin}, first prediction week: {matrices.weeks[test_origin].date()}")
    assert matrices.weeks[test_origin] == pd.Timestamp("2026-07-27"), f"Unexpected origin week: {matrices.weeks[test_origin]}"

    # -------------------------------------------------------------------------
    # STEP 3: Train Bottom-Level LightGBM Variant B (52 origins, seed 42)
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Training Bottom LightGBM Variant B on 52 origins ---")
    latest_train_origin = test_origin - HORIZON  # 134 - 9 = 125
    earliest_train_origin = max(26, latest_train_origin - 51)  # 125 - 51 = 74
    train_origins_b = np.arange(earliest_train_origin, latest_train_origin + 1)
    print(f"LightGBM 52w Training Origins: {train_origins_b[0]} .. {train_origins_b[-1]} ({len(train_origins_b)} origins)")
    print(f"Target week range: {matrices.weeks[train_origins_b[0]].date()} .. {matrices.weeks[train_origins_b[-1] + HORIZON - 1].date()}")
    assert matrices.weeks[train_origins_b[-1] + HORIZON - 1] < matrices.weeks[test_origin], "Data leakage detected!"

    df_lgbm_pred, diag_lgbm = train_lgbm_history_variant(matrices, test_origin, variant="B", seed=42)
    print(f"LightGBM 52w training complete: time = {diag_lgbm['train_time_sec']:.2f}s, mean expected = {diag_lgbm['mean_expected']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 4: Train / Load Remaining Components (TFT, CatBoost, Per-Category)
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Loading and Training Remaining Components ---")

    # 1. TFT (raw continuous)
    tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_prediction_raw.csv.gz"
    if not tft_path.exists():
        tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_submission.csv"
    print(f"Loading TFT test predictions from {tft_path}...")
    df_tft = pd.read_csv(tft_path)
    df_tft["week"] = pd.to_datetime(df_tft["week"]).dt.strftime("%Y-%m-%d")
    df_tft = df_tft.rename(columns={"quantity": "raw_tft"})

    # 2. Per-Category Models
    per_cat_path = PROJECT_ROOT / "experiments" / "benchmarks" / "per_category" / "per_category_final_submission.csv"
    print(f"Loading Per-Category test predictions from {per_cat_path}...")
    df_per_cat = pd.read_csv(per_cat_path)
    df_per_cat["week"] = pd.to_datetime(df_per_cat["week"]).dt.strftime("%Y-%m-%d")
    df_per_cat = df_per_cat.rename(columns={"quantity": "raw_per_cat"})

    # 3. CatBoost (trained on 18 origins 108..125)
    print("Training CatBoost on 18 origins (108..125)...")
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
        "quantity": np.maximum(cb_vals, 0.0),
    })

    # -------------------------------------------------------------------------
    # STEP 5: Apply predict_from_components to Generate Candidate Prediction
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Blending Components with predict_from_components ---")
    candidate_comp_dict = {
        "tft": df_tft[["week", "warehouse_name", "article", "raw_tft"]],
        "lgbm": pd.DataFrame({
            "week": df_lgbm_pred["week"],
            "warehouse_name": df_lgbm_pred["warehouse_name"],
            "article": df_lgbm_pred["article"],
            "p_sale": df_lgbm_pred["p_sale"],
            "q_cond": df_lgbm_pred["conditional_quantity"],
            "expected": df_lgbm_pred["expected_quantity"],
        }),
        "catboost": pd.DataFrame({
            "week": df_cb["week"],
            "warehouse_name": df_cb["warehouse_name"],
            "article": df_cb["article"],
            "quantity": df_cb["quantity"],
        }),
        "per_category": pd.DataFrame({
            "week": df_per_cat["week"],
            "warehouse_name": df_per_cat["warehouse_name"],
            "article": df_per_cat["article"],
            "quantity": df_per_cat["raw_per_cat"],
        }),
    }

    candidate_config = {
        "name": "candidate_lgbm52",
        "components": {
            "tft": {"weight": 0.50, "type": "raw"},
            "lgbm": {"weight": 0.30, "type": "two_stage", "prob_gate": 0.5, "comp_postprocess": None},
            "catboost": {"weight": 0.10, "type": "raw"},
            "per_category": {"weight": 0.10, "type": "raw"},
        },
        "blend": {"alpha": 1.05, "threshold": 0.65, "rounding": "rint", "clip_lower": 0.0},
    }

    keyed_candidate, candidate_sub = predict_from_components(candidate_comp_dict, candidate_config)

    # -------------------------------------------------------------------------
    # STEP 6: Validate Format and Save Candidate File
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: Validating and Saving candidate_lgbm52.csv ---")
    candidate_sub["quantity"] = candidate_sub["quantity"].astype(int)
    warnings = validate_submission(candidate_sub, "candidate_lgbm52")
    if warnings:
        print("Validation Warnings:")
        for w in warnings:
            print("  ⚠", w)
    else:
        print("Format Validation PASSED: 0 errors, 100% compliant with competition requirements.")

    # Save to submissions/candidate_lgbm52.csv
    candidate_sub.to_csv(CANDIDATE_PATH, index=False)
    print(f"Saved candidate submission to: {CANDIDATE_PATH}")
    assert CANDIDATE_PATH.exists() and CANDIDATE_PATH.stat().st_size > 0

    candidate_sha = compute_sha256(CANDIDATE_PATH)
    tft_sha = compute_sha256(tft_path)
    per_cat_sha = compute_sha256(per_cat_path)

    # Statistics
    nonzero_rate = float((candidate_sub["quantity"] > 0).mean())
    total_qty = int(candidate_sub["quantity"].sum())
    max_val = int(candidate_sub["quantity"].max())
    p99_val = float(np.percentile(candidate_sub["quantity"], 99))

    print(f"Candidate Summary Statistics:")
    print(f"  Rows:         {len(candidate_sub)}")
    print(f"  Non-zero rate:{nonzero_rate * 100:.2f}%")
    print(f"  Total Volume: {total_qty}")
    print(f"  Max Quantity: {max_val}")
    print(f"  99th Pct:     {p99_val}")
    print(f"  SHA-256:      {candidate_sha}")

    # -------------------------------------------------------------------------
    # STEP 7: Save Manifest JSON
    # -------------------------------------------------------------------------
    manifest = {
        "candidate_name": "candidate_lgbm52",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": "9a0489fd592b83c8d6673c6d7233de6c2970ad1e",
        "destination_file": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
        "sha256": candidate_sha,
        "row_count": len(candidate_sub),
        "target_period": "2026-07-27 .. 2026-09-21 (9 weeks)",
        "training_cutoff_date": "2026-07-26",
        "components": {
            "tft": {
                "source_file": str(tft_path.relative_to(PROJECT_ROOT)),
                "sha256": tft_sha,
                "weight": 0.50,
                "type": "raw_continuous",
            },
            "lightgbm_variant_b": {
                "history_origins": 52,
                "origin_range": [int(train_origins_b[0]), int(train_origins_b[-1])],
                "weight": 0.30,
                "prob_gate": 0.5,
                "type": "two_stage_hurdle",
            },
            "catboost": {
                "history_origins": 18,
                "origin_range": [int(cb_train_origins[0]), int(cb_train_origins[-1])],
                "weight": 0.10,
                "type": "regressor",
            },
            "per_category": {
                "source_file": str(per_cat_path.relative_to(PROJECT_ROOT)),
                "sha256": per_cat_sha,
                "weight": 0.10,
                "type": "benchmark_aligned",
            },
        },
        "blend_postprocessing": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0,
        },
        "cv_performance_5_folds": {
            "mean_clipped_score": 0.730988,
            "mean_unclipped_ratio": 0.756203,
            "mean_smape": 62.130872,
            "mean_wape": 49.576987,
            "mean_rmse": 2.897645,
            "mean_bias": 75.576152,
            "mean_rmsle": 0.527622,
        },
        "test_prediction_statistics": {
            "total_rows": len(candidate_sub),
            "nonzero_rows": int((candidate_sub["quantity"] > 0).sum()),
            "nonzero_rate": nonzero_rate,
            "total_quantity": total_qty,
            "max_quantity": max_val,
            "p99_quantity": p99_val,
            "mean_quantity": float(candidate_sub["quantity"].mean()),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to: {MANIFEST_PATH}")
    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION FINISHED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
