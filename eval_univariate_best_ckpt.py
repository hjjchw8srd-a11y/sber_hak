#!/usr/bin/env python3
"""
eval_univariate_best_ckpt.py
Evaluates best checkpoint step_600 on External Test (origin 89).
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import PeftModel

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from timestral import TimestralForecaster
from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_harmonized_timestral_protocol import run_harmonized_evaluation, apply_p, calc_scores, KEYS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

weights_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)

ckpt_path = ROOT / "experiments" / "timestral_feature_research" / "checkpoints" / "univariate_fold4" / "step_600"
forecaster.model = PeftModel.from_pretrained(forecaster.model, str(ckpt_path))
forecaster.model.eval()

# External Test Dataset
champ_path = ROOT / "experiments" / "champion_oof" / "champion_scale_1m_fold_4.csv.gz"
df_champ = pd.read_csv(champ_path)
test_truth_df = df_champ[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
test_base_df = df_champ[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})

test_results, pred_continuous = run_harmonized_evaluation(
    forecaster=forecaster,
    engine=engine,
    groups=[],
    eval_origin=89,
    truth_df=test_truth_df,
    base_df=test_base_df,
    device=device,
)

# Blend with champion
raw_champ = df_champ["raw_champion"].values.astype(float)
raw_blend = 0.90 * raw_champ + 0.10 * pred_continuous
pred_blend = apply_p(raw_blend, 1.05, 0.65)
df_p_blend = test_truth_df[KEYS].copy()
df_p_blend["quantity"] = pred_blend
m_base = compute_all_metrics_fast(test_truth_df, test_base_df)
m_blend = compute_all_metrics_fast(test_truth_df, df_p_blend)
c_blend, u_blend = calc_scores(m_blend, m_base)

print("\n" + "=" * 80)
print("HARMONIZED UNIVARIATE (600 STEPS, STEP 600 CKPT) EXTERNAL TEST RESULTS:")
print(f"  Standalone RINT:      Clipped = {test_results['rint']['clipped']:.6f} | Unclipped = {test_results['rint']['unclipped']:.6f} | WAPE = {test_results['rint']['wape']:.3f}% | RMSE = {test_results['rint']['rmse']:.4f}")
print(f"  Standalone Legacy P:  Clipped = {test_results['post']['clipped']:.6f} | Unclipped = {test_results['post']['unclipped']:.6f} | WAPE = {test_results['post']['wape']:.3f}% | RMSE = {test_results['post']['rmse']:.4f}")
print(f"  Champion + 10% Blend: Clipped = {c_blend:.6f} | Unclipped = {u_blend:.6f} | WAPE = {m_blend['wape']:.3f}% | RMSE = {m_blend['rmse']:.4f}")
print("=" * 80)

# Save test metrics JSON
out_json = ROOT / "experiments" / "timestral_feature_research" / "harmonized_univariate_fold4_results.json"
results_dict = {
    "model": "Univariate Timestral LoRA (Harmonized 600 steps)",
    "fold": 4,
    "best_step": 600,
    "val_unclipped": 0.743792,
    "standalone_rint": test_results["rint"],
    "standalone_post": test_results["post"],
    "blend_w10": {"clipped": c_blend, "unclipped": u_blend, **m_blend},
}
with open(out_json, "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"Saved results to {out_json.name}")
