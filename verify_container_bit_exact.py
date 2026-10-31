#!/usr/bin/env python3
"""
verify_container_bit_exact.py

Verifies that the forecast-container's Predictor on the full historical dataset
produces a 100% bit-exact match against submissions/champion_clipped_score.csv (diff.sum() == 0.0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))

from app.predictor import Predictor, predict_from_history

def main():
    print("=" * 80)
    print("VERIFYING CONTAINER BIT-EXACT REPRODUCIBILITY")
    print("=" * 80)

    champion_csv = PROJECT_ROOT / "submissions" / "champion_clipped_score.csv"
    print(f"Loading champion submission: {champion_csv}...")
    sub_df = pd.read_csv(champion_csv)
    print(f"Champion rows: {len(sub_df):,}, sum: {sub_df['quantity'].sum():,.0f}")

    sales_path = PROJECT_ROOT / "sales_dataset_hakaton_train.csv"
    stock_path = PROJECT_ROOT / "stock_dataset_hakaton_train.csv"
    container_model_dir = PROJECT_ROOT / "forecast-container" / "model"

    print("\nExecuting Predictor.predict() from container model bundle on cutoff <= 2026-07-26...")
    sales_df = pd.read_csv(sales_path)
    sales_df = sales_df[sales_df["date"] <= "2026-07-26"]
    stock_df = pd.read_csv(stock_path)
    stock_df = stock_df[stock_df["date"] <= "2026-07-26"]

    t0 = time.perf_counter()
    container_pred = predict_from_history(sales_df, stock_df, bundle_dir=container_model_dir)
    elapsed = time.perf_counter() - t0
    print(f"Inference completed in {elapsed:.2f}s!")
    print(f"Container pred rows: {len(container_pred):,}, sum: {container_pred['quantity'].sum():,.0f}")

    # Merge and compare
    merged = sub_df.merge(container_pred, on=["week", "warehouse_name", "article"], suffixes=("_sub", "_cont"))
    assert len(merged) == len(sub_df), f"Row count mismatch: {len(merged)} vs {len(sub_df)}"

    diff = np.abs(merged["quantity_sub"].to_numpy() - merged["quantity_cont"].to_numpy())
    print(f"Total absolute diff: {diff.sum():.6f}")
    print(f"Max absolute diff: {diff.max():.6f}")
    print(f"Differing rows: {(diff > 0).sum():,} / {len(diff):,}")

    if diff.sum() == 0.0:
        print("\n" + "*" * 80)
        print("SUCCESS: 100% BIT-EXACT MATCH CONFIRMED (diff.sum() == 0.0)!")
        print("*" * 80)
    else:
        print("\n" + "!" * 80)
        print("FAILURE: Discrepancy detected!")
        print("!" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()
