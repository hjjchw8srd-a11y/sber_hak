#!/usr/bin/env python3
"""
experiment_champion_timestral_blending.py

Performs honest, verified ensembling of the true champion 'Scale_1M_Dual_0.5'
with 'Timestral-209M LoRA' across all 5 competition folds:
- Selection Folds: [4, 3, 2]
- Check Folds: [1, 0]

Blending Protocol:
- Continuous forecasts blended in un-postprocessed scale:
  raw_blend = (1.0 - w) * raw_champion + w * raw_timestral
- Post-processing applied exactly once:
  final_blend = apply_p(raw_blend, 1.05, 0.65)
- Weights evaluated: w in [0.00, 0.05, 0.10, 0.20, 0.30] and w = 1.00 (Standalone Timestral).
- Strict one-to-one key-based merge (week, warehouse_name, article).
- Selection folds [4, 3, 2] used to identify optimal weight w*.
- Check folds [1, 0] evaluated out-of-sample on fixed w*.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]
FOLDS = [4, 3, 2, 1, 0]
WEIGHTS = [0.00, 0.05, 0.10, 0.20, 0.30, 1.00]


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


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 105)
    print("HONEST 5-FOLD ENSEMBLING EXPERIMENT: Scale_1M_Dual_0.5 (Champion) + Timestral-209M LoRA")
    print("=" * 105)

    all_rows = []

    for fold in FOLDS:
        champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
        tim_path = ROOT / "experiments" / f"timestral_lora_fold{fold}" / "keyed_predictions" / "pilot" / f"timestral_209m_lora_fold{fold}_pilot_keyed.csv.gz"

        if not champ_path.exists():
            raise FileNotFoundError(f"Missing champion OOF for fold {fold}: {champ_path}")
        if not tim_path.exists():
            raise FileNotFoundError(f"Missing Timestral predictions for fold {fold}: {tim_path}")

        df_champ = pd.read_csv(champ_path)
        df_tim = pd.read_csv(tim_path)

        merged = pd.merge(
            df_champ[KEYS + ["raw_champion", "y_true", "mean13"]],
            df_tim[KEYS + ["y_pred_continuous"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        assert len(merged) == 79101, f"Fold {fold} row count mismatch: {len(merged)}"

        truth = merged[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
        base = merged[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})
        m_base = compute_all_metrics_fast(truth, base)

        raw_champ = merged["raw_champion"].values
        raw_tim = merged["y_pred_continuous"].values

        # Baseline champion score (w = 0.0)
        p_champ_base = apply_p(raw_champ, 1.05, 0.65)
        df_cb = truth[KEYS].copy()
        df_cb["quantity"] = p_champ_base
        m_cb = compute_all_metrics_fast(truth, df_cb)
        c_base, u_base = calc_scores(m_cb, m_base)

        for w in WEIGHTS:
            if w == 1.00:
                raw_mix = raw_tim
            elif w == 0.00:
                raw_mix = raw_champ
            else:
                raw_mix = (1.0 - w) * raw_champ + w * raw_tim

            p_mix = apply_p(raw_mix, 1.05, 0.65)
            df_mix = truth[KEYS].copy()
            df_mix["quantity"] = p_mix
            m_mix = compute_all_metrics_fast(truth, df_mix)
            c_mix, u_mix = calc_scores(m_mix, m_base)

            delta_c = c_mix - c_base
            delta_u = u_mix - u_base

            all_rows.append({
                "fold": fold,
                "weight_timestral": w,
                "stage": "Selection" if fold in [4, 3, 2] else "Check",
                "clipped_score": c_mix,
                "unclipped_ratio": u_mix,
                "delta_clipped_vs_champ": delta_c,
                "delta_unclipped_vs_champ": delta_u,
                "smape": m_mix["smape"],
                "wape": m_mix["wape"],
                "rmse": m_mix["rmse"],
                "bias": m_mix["bias"],
                "rmsle": m_mix["rmsle"],
            })

    df_res = pd.DataFrame(all_rows)

    # Detailed per-fold results table
    print("\n" + "=" * 115)
    print(f"{'Fold':<6} {'Stage':<10} {'Weight Tim':<12} {'Clipped':<12} {'Unclipped':<12} {'Delta Clip':<12} {'SMAPE (%)':<10} {'WAPE (%)':<10} {'RMSE':<10}")
    print("=" * 115)
    for r in all_rows:
        print(f"{r['fold']:<6} {r['stage']:<10} {r['weight_timestral']:<12.2f} {r['clipped_score']:<12.6f} {r['unclipped_ratio']:<12.6f} {r['delta_clipped_vs_champ']:<+12.6f} {r['smape']:<10.3f} {r['wape']:<10.3f} {r['rmse']:<10.4f}")

    # Aggregations
    print("\n" + "=" * 105)
    print("AGGREGATED RESULTS ACROSS FOLDS")
    print("=" * 105)

    for split_name, fold_subset in [("SELECTION FOLDS [4, 3, 2]", [4, 3, 2]), ("CHECK FOLDS [1, 0]", [1, 0]), ("FULL 5-FOLD AVERAGE", [4, 3, 2, 1, 0])]:
        print(f"\n--- {split_name} ---")
        print(f"{'Weight Tim':<12} {'Mean Clipped':<14} {'Mean Unclipped':<16} {'Mean Delta Clip':<16} {'Mean SMAPE (%)':<14} {'Mean WAPE (%)':<14} {'Mean RMSE':<12}")
        print("-" * 100)
        df_sub = df_res[df_res["fold"].isin(fold_subset)]
        for w in WEIGHTS:
            sub_w = df_sub[df_sub["weight_timestral"] == w]
            mean_c = sub_w["clipped_score"].mean()
            mean_u = sub_w["unclipped_ratio"].mean()
            mean_dc = sub_w["delta_clipped_vs_champ"].mean()
            mean_sm = sub_w["smape"].mean()
            mean_wa = sub_w["wape"].mean()
            mean_rm = sub_w["rmse"].mean()
            print(f"{w:<12.2f} {mean_c:<14.6f} {mean_u:<16.6f} {mean_dc:<+16.6f} {mean_sm:<14.3f} {mean_wa:<14.3f} {mean_rm:<12.4f}")

    out_csv = ROOT / "experiments" / "champion_oof" / "champion_timestral_5folds_blending_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nFull 5-fold blending results saved to: {out_csv}")


if __name__ == "__main__":
    main()
