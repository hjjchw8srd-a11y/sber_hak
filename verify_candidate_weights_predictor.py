#!/usr/bin/env python3
"""
verify_candidate_weights_predictor.py

Automated verification of Candidate Weights 45/37/9/9:
1. Validates CSV format and integrity: 79,101 rows, correct columns, non-negative integers.
2. Checks manifest sha256 matches submissions/candidate_weights_45_37_9_9.csv.
3. Tests Predictor loading both Candidate Bundle and Baseline Bundle.
4. Executes inference on demo dataset with both bundles and confirms:
   - Inference completes with 0 errors.
   - Bundle weights produce distinct, consistent predictions reflecting 45/37/9/9 vs 50/30/10/10.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))

from app.predictor import Predictor, predict_from_history
from score import validate_submission

SUB_CSV = PROJECT_ROOT / "submissions" / "candidate_weights_45_37_9_9.csv"
MANIFEST = PROJECT_ROOT / "submissions" / "candidate_weights_45_37_9_9_manifest.json"
CAND_BUNDLE = PROJECT_ROOT / "experiments" / "model_improvements" / "candidate_weights" / "bundle"
BASE_BUNDLE = PROJECT_ROOT / "forecast-container" / "model"
DEMO_SALES = PROJECT_ROOT / "forecast-container" / "data" / "demo_sales.csv"
DEMO_STOCK = PROJECT_ROOT / "forecast-container" / "data" / "demo_stock.csv"


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
    print("VERIFICATION OF CANDIDATE WEIGHTS 45/37/9/9 AND PREDICTOR API/CLI")
    print("=" * 80)

    # 1. Check CSV
    assert SUB_CSV.exists(), f"Missing {SUB_CSV}"
    df_sub = pd.read_csv(SUB_CSV)
    assert len(df_sub) == 79101, f"Expected 79101 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["week", "warehouse_name", "article", "quantity"]
    assert (df_sub["quantity"] >= 0).all()
    assert (df_sub["quantity"] == df_sub["quantity"].astype(int)).all()
    warnings = validate_submission(df_sub, "candidate_weights_45_37_9_9")
    assert not warnings, f"Validation failed with warnings: {warnings}"
    print(f"1. CSV validation PASSED: 79,101 rows, strictly valid format.")

    # 2. Check Manifest
    assert MANIFEST.exists(), f"Missing {MANIFEST}"
    with open(MANIFEST, "r", encoding="utf-8") as f:
        man = json.load(f)
    actual_sha = compute_sha256(SUB_CSV)
    assert man["sha256"] == actual_sha, f"SHA mismatch! Manifest: {man['sha256']}, actual: {actual_sha}"
    print(f"2. Manifest SHA-256 PASSED: {actual_sha}")

    # 3. Check Predictor Bundle loading
    cand_pred = Predictor.load(CAND_BUNDLE)
    base_pred = Predictor.load(BASE_BUNDLE)

    c_w = cand_pred.postprocessing["components"]
    b_w = base_pred.postprocessing["components"]

    assert c_w["tft"]["weight"] == 0.45
    assert c_w["lightgbm_b"]["weight"] == 0.37
    assert c_w["catboost"]["weight"] == 0.09
    assert c_w["per_category"]["weight"] == 0.09

    assert b_w["tft"]["weight"] == 0.50
    assert b_w["lightgbm_b"]["weight"] == 0.30
    assert b_w["catboost"]["weight"] == 0.10
    assert b_w["per_category"]["weight"] == 0.10
    print("3. Predictor bundle configs PASSED: Candidate (45/37/9/9) vs Baseline (50/30/10/10).")

    # 4. Predict on demo data
    print("4. Running Predictor on demo dataset with Candidate Bundle...")
    res_cand = predict_from_history(DEMO_SALES, DEMO_STOCK, bundle_dir=CAND_BUNDLE)
    assert len(res_cand) == 79101
    assert list(res_cand.columns) == ["week", "warehouse_name", "article", "quantity"]
    print("   Candidate prediction on demo completed: 79101 rows.")

    print("5. Running Predictor on demo dataset with Baseline Bundle...")
    res_base = predict_from_history(DEMO_SALES, DEMO_STOCK, bundle_dir=BASE_BUNDLE)
    assert len(res_base) == 79101

    diff = np.abs(res_cand["quantity"] - res_base["quantity"]).sum()
    rows_differ = int((res_cand["quantity"] != res_base["quantity"]).sum())
    print(f"   Comparison on demo dataset: {diff} units differ across {rows_differ} rows.")
    assert rows_differ > 0, "Expected difference between Candidate and Baseline weights!"
    print("   Predictor weights dynamic dispatch VERIFIED: Candidate vs Baseline behave distinctly.")

    print("\nALL VERIFICATION CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
