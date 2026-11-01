#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))
from fast_score import compute_all_metrics_fast

champ_path = ROOT / "experiments" / "champion_oof" / "champion_scale_1m_fold_4.csv.gz"
df_champ = pd.read_csv(champ_path)
truth = df_champ[["week", "warehouse_name", "article", "y_true"]].rename(columns={"y_true": "quantity"})

# Take normal prediction with RMSE = 3.0357
# Suppose just one outlier of magnitude 3000 is present:
df_test = truth[["week", "warehouse_name", "article"]].copy()
df_test["quantity"] = truth["quantity"].copy() # perfect prediction, RMSE=0

# Add a single outlier of 3000 at row 0:
df_test.loc[0, "quantity"] = 3000
m = compute_all_metrics_fast(truth, df_test)
print(f"Single outlier of 3,000 -> RMSE: {m['rmse']:.4f}, WAPE: {m['wape']:.3f}%, SMAPE: {m['smape']:.4f}")

# What magnitude gives RMSE = 11.08?
# RMSE^2 = 11.0827^2 = 122.826
# N = 79101
# Sum of squared errors = 122.826 * 79101 = 9,715,698
# If base MSE is ~9.2 (RMSE=3.03, MSE=9.18), excess MSE is 122.826 - 9.18 = 113.646
# Excess sum of squares = 113.646 * 79101 = 8,989,451
# sqrt(8,989,451) = 2998.24!
print(f"An error of exactly 2,998 on a SINGLE row produces an RMSE jump from 3.03 to 11.08!")
