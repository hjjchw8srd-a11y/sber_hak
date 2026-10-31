#!/usr/bin/env python3
"""
blend_timestral_fold4.py

Evaluates blending weights of Timestral-209M LoRA with the baseline ensemble on Fold 4.
Blends un-postprocessed continuous forecasts directly in the same scale,
then applies final post-processing (alpha=1.05, thresh=0.65, np.rint) exactly once.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

def apply_p(raw: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)

def calc_scores(m_pred: dict, m_base: dict) -> tuple[float, float]:
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped

def main():
    # 1. Load Blend B Fold 4
    df_b = pd.read_csv(ROOT / "experiments" / "history_window_audit" / "keyed_oof" / "blend_b_fold_4.csv.gz")
    
    # 2. Load Timestral LoRA Fold 4
    df_timestral = pd.read_csv(ROOT / "experiments" / "timestral_lora_fold4" / "keyed_predictions" / "pilot" / "timestral_209m_lora_fold4_pilot_keyed.csv.gz")

    keys = ["week", "warehouse_name", "article"]
    merged = pd.merge(
        df_b[keys + ["raw_blend"]],
        df_timestral[keys + ["y_true", "unconstrained_demand_13", "y_pred_continuous"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all() and len(merged) == 79101, "Merge failed!"
    print(f"Strict one-to-one merge verified: exactly {len(merged):,} rows.")

    truth = merged[keys + ["y_true"]].rename(columns={"y_true": "quantity"})
    base = merged[keys + ["unconstrained_demand_13"]].rename(columns={"unconstrained_demand_13": "quantity"})
    m_base = compute_all_metrics_fast(truth, base)

    raw_ens = merged["raw_blend"].values
    raw_tim = merged["y_pred_continuous"].values

    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.0]

    print("\n" + "=" * 95)
    print(f"{'Weight Timestral':<18} {'Clipped':<12} {'Unclipped':<12} {'SMAPE (%)':<12} {'WAPE (%)':<12} {'RMSE':<10} {'Bias (%)':<10} {'RMSLE':<10}")
    print("=" * 95)

    results = []
    for w in weights:
        raw_mix = (1.0 - w) * raw_ens + w * raw_tim
        pred_mix = apply_p(raw_mix, 1.05, 0.65)
        p_df = merged[keys].copy()
        p_df["quantity"] = pred_mix
        m = compute_all_metrics_fast(truth, p_df)
        c, u = calc_scores(m, m_base)
        print(f"{w:<18.2f} {c:<12.6f} {u:<12.6f} {m['smape']:<12.3f} {m['wape']:<12.3f} {m['rmse']:<10.4f} {m['bias']:<10.3f} {m['rmsle']:<10.4f}")
        results.append({"weight": w, "clipped": c, "unclipped": u, **m})

    df_res = pd.DataFrame(results)
    out_path = ROOT / "experiments" / "timestral_lora_fold4" / "logs" / "pilot" / "blend_b_timestral_fold4_grid.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nGrid search saved to {out_path}")

if __name__ == "__main__":
    main()
