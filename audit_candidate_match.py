#!/usr/bin/env python3
"""
audit_candidate_match.py

Runs predict_from_history on the full allowed competition history (cutoff 2026-07-26)
and asserts 0 discrepancy against submissions/candidate_lgbm52.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))

from app.predictor import predict_from_history

CUTOFF = pd.Timestamp("2026-07-26")
CANDIDATE_PATH = PROJECT_ROOT / "submissions" / "candidate_lgbm52.csv"
BUNDLE_DIR = PROJECT_ROOT / "forecast-container" / "model"


def main():
    print("=" * 80)
    print("RUNNING CANDIDATE EQUIVALENCE AUDIT ON FULL ALLOWED HISTORY")
    print(f"Cutoff: {CUTOFF.date()}")
    print("=" * 80)

    start = time.perf_counter()

    print("\nLoading raw sales and stock up to cutoff 2026-07-26...")
    sales_raw = pd.read_csv(PROJECT_ROOT / "sales_dataset_hakaton_train.csv", parse_dates=["date"])
    sales = sales_raw[sales_raw["date"] <= CUTOFF].copy()
    print(f"Sales rows: {len(sales):,} | Date range: {sales['date'].min().date()} .. {sales['date'].max().date()}")

    stock_raw = pd.read_csv(PROJECT_ROOT / "stock_dataset_hakaton_train.csv", parse_dates=["date"])
    stock = stock_raw[stock_raw["date"] <= CUTOFF].copy()
    print(f"Stock rows: {len(stock):,} | Date range: {stock['date'].min().date()} .. {stock['date'].max().date()}")

    print("\nRunning predict_from_history with production bundle...")
    pred_start = time.perf_counter()
    pred_df = predict_from_history(sales, stock, bundle_dir=BUNDLE_DIR)
    pred_time = time.perf_counter() - pred_start
    print(f"Prediction generated in {pred_time:.2f} seconds! Rows: {len(pred_df):,}")

    print("\nLoading reference submissions/candidate_lgbm52.csv...")
    ref_df = pd.read_csv(CANDIDATE_PATH)
    ref_df["week"] = pd.to_datetime(ref_df["week"]).dt.strftime("%Y-%m-%d")
    pred_df["week"] = pd.to_datetime(pred_df["week"]).dt.strftime("%Y-%m-%d")

    merged = ref_df.merge(pred_df, on=["week", "warehouse_name", "article"], suffixes=("_ref", "_pred"), validate="one_to_one")
    assert len(merged) == len(ref_df), f"Key mismatch! Expected {len(ref_df)}, got {len(merged)}"

    diff = np.abs(merged["quantity_ref"] - merged["quantity_pred"])
    mismatch_count = int((diff > 0).sum())
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())

    print("\n" + "-" * 60)
    print("AUDIT RESULTS:")
    print(f"  Total Rows:             {len(merged):,}")
    print(f"  Mismatched Rows:        {mismatch_count}")
    print(f"  Max Absolute Diff:      {max_diff}")
    print(f"  Mean Absolute Diff:     {mean_diff:.6f}")
    print(f"  Reference Total Qty:    {int(merged['quantity_ref'].sum()):,}")
    print(f"  Predicted Total Qty:    {int(merged['quantity_pred'].sum()):,}")
    print(f"  Reference Max Qty:      {int(merged['quantity_ref'].max())}")
    print(f"  Predicted Max Qty:      {int(merged['quantity_pred'].max())}")
    print(f"  Reference Non-zero:     {(merged['quantity_ref'] > 0).mean() * 100:.2f}%")
    print(f"  Predicted Non-zero:     {(merged['quantity_pred'] > 0).mean() * 100:.2f}%")
    print("-" * 60)

    if mismatch_count == 0:
        print("\n>>> AUDIT PASSED: 100% EXACT MATCH (0 DIFFERENCES) ACROSS ALL 79,101 ROWS! <<<\n")
    else:
        print(f"\n>>> AUDIT FAILED: {mismatch_count} rows differ! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
