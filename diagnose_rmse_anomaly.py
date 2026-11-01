#!/usr/bin/env python3
"""
diagnose_rmse_anomaly.py

Investigates why Univariate Baseline on Fold 4 had RMSE 11.0854 in FEATURE_ABLATION.csv,
while Task 7 Univariate LoRA had RMSE 3.0056, and covariate models had ~3.0026.
Checks:
- Truth alignment
- Distribution of predictions and errors
- Max predictions
- Top-100 squared error contributions
- Extreme outliers
- Exact computation of compute_all_metrics_fast
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]

# 1. Load ground truth from champion OOF for Fold 4
champ_path = ROOT / "experiments" / "champion_oof" / "champion_scale_1m_fold_4.csv.gz"
df_champ = pd.read_csv(champ_path)
truth = df_champ[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
base = df_champ[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})

# 2. Check Task 7 Univariate LoRA predictions on Fold 4
t7_path = ROOT / "experiments" / "timestral_lora_fold4" / "keyed_predictions" / "timestral_209m_predictions_keyed.csv.gz"
if t7_path.exists():
    df_t7 = pd.read_csv(t7_path)
    merged_t7 = pd.merge(truth, df_t7, on=KEYS)
    # Task 7 has raw_timestral
    col_t7 = [c for c in df_t7.columns if "raw" in c or "pred" in c or "quantity" in c]
    print(f"Task 7 columns: {df_t7.columns.tolist()}")
    if "raw_timestral" in df_t7.columns:
        m_t7 = compute_all_metrics_fast(truth, df_t7.rename(columns={"raw_timestral": "quantity"}))
        print(f"Task 7 raw_timestral metrics: RMSE={m_t7['rmse']:.4f}, WAPE={m_t7['wape']:.3f}%")
        q_t7_rint = np.rint(np.clip(df_t7["raw_timestral"].values, 0, None)).astype(int)
        df_t7_rint = df_t7[KEYS].assign(quantity=q_t7_rint)
        m_t7_rint = compute_all_metrics_fast(truth, df_t7_rint)
        print(f"Task 7 rint metrics: RMSE={m_t7_rint['rmse']:.4f}, WAPE={m_t7_rint['wape']:.3f}%")

# 3. Check Task 11 Selected Covariates on Fold 4
t11_oof_path = ROOT / "experiments" / "timestral_feature_research" / "oof_predictions" / "selected_covariates_raw_fold_4.csv.gz"
if t11_oof_path.exists():
    df_t11 = pd.read_csv(t11_oof_path)
    merged_t11 = pd.merge(truth, df_t11, on=KEYS)
    raw_t11 = df_t11["raw_timestral_selected"].values
    m_t11 = compute_all_metrics_fast(truth, df_t11.rename(columns={"raw_timestral_selected": "quantity"}))
    print(f"\nTask 11 raw_timestral_selected metrics: RMSE={m_t11['rmse']:.4f}, WAPE={m_t11['wape']:.3f}%")
    q_t11_rint = np.rint(np.clip(raw_t11, 0, None)).astype(int)
    df_t11_rint = df_t11[KEYS].assign(quantity=q_t11_rint)
    m_t11_rint = compute_all_metrics_fast(truth, df_t11_rint)
    print(f"Task 11 rint metrics: RMSE={m_t11_rint['rmse']:.4f}, WAPE={m_t11_rint['wape']:.3f}%")
    print(f"Task 11 max raw: {raw_t11.max():.2f}, 99.9th percentile: {np.percentile(raw_t11, 99.9):.2f}")

# 4. Now run a clean 120-step Univariate run using EXACTLY the current codebase to see where 11.08 came from!
print("\nInvestigating current Univariate run...")
