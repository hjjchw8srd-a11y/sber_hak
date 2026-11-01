#!/usr/bin/env python3
"""
rerun_univariate_check.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_timestral_covariates_experiment import train_and_evaluate_timestral_run

engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

res = train_and_evaluate_timestral_run(
    engine=engine,
    groups=[],
    fold=4,
    steps=120,
    seed=42,
)

print("\n--- RESULTS OF NEW UNIVARIATE RUN ---")
print("Standalone RINT:", res["standalone_rint"])
print("Standalone Legacy P:", res["standalone_post"])
print("Champion + 10% Blend:", res["blend_w10"])

pred = res["pred_continuous"]
print(f"Pred min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}")
print(f"Pred 99th percentile: {np.percentile(pred, 99):.4f}, 99.9th: {np.percentile(pred, 99.9):.4f}")
