#!/usr/bin/env python3
"""
check_back_to_school.py

Checks Priority C: Back-to-school (1 September) demand effect.
1. Examines historical sales dynamics around weeks 33-38 (2024) and 85-92 (2025).
2. Tests whether applying a back-to-school multiplier to Fold 4 (weeks 89..97, covering Sep-Nov 2025)
   improves the Clipped Score or degrades it.
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
    return np.rint(gated).clip(0, None).astype(int)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    weeks = sorted(data["week"].unique())
    week_idx = {w: i for i, w in enumerate(weeks)}
    data["time_idx"] = data["week"].map(week_idx)

    print("=" * 80)
    print("BACK-TO-SCHOOL SEASONALITY INVESTIGATION (PRIORITY C)")
    print("=" * 80)

    # Fold 4 analysis: origin 89 (week 89 = 2025-09-15)
    # Horizon 1..9 corresponds to weeks 89..97.
    oof_dir = ROOT / "experiments" / "champion_oof"
    champ_path = oof_dir / "champion_scale_1m_fold_4.csv.gz"
    tim_path = (
        ROOT / "experiments" / "timestral_lora_fold4" / "keyed_predictions" / "pilot"
        / "timestral_209m_lora_fold4_pilot_keyed.csv.gz"
    )

    df_champ = pd.read_csv(champ_path)
    df_tim = pd.read_csv(tim_path)

    merged = pd.merge(
        df_champ[KEYS + ["raw_champion", "y_true", "mean13"]],
        df_tim[KEYS + ["y_pred_continuous"]],
        on=KEYS,
        how="inner",
    )

    pair_cat = data[["warehouse_name", "article", "category_lvl_1"]].drop_duplicates()
    merged = pd.merge(merged, pair_cat, on=["warehouse_name", "article"], how="left")

    truth = merged[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
    base = merged[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})
    m_base = compute_all_metrics_fast(truth, base)

    # Baseline 10% blend
    raw_c = merged["raw_champion"].values.astype(np.float64)
    raw_t = merged["y_pred_continuous"].values.astype(np.float64)
    raw_blend_10 = 0.90 * raw_c + 0.10 * raw_t

    pred_baseline = apply_p(raw_blend_10, 1.05, 0.65)
    df_pred_base = truth[KEYS].copy()
    df_pred_base["quantity"] = pred_baseline
    m_baseline = compute_all_metrics_fast(truth, df_pred_base)
    c_base, u_base = calc_scores(m_baseline, m_base)

    print(f"Fold 4 Baseline (Fixed w=0.10): Clipped = {c_base:.6f}, Unclipped = {u_base:.6f}, WAPE = {m_baseline['wape']:.3f}%")

    # Test multipliers on early horizon weeks (weeks 89-91, mid-late September)
    # or specific categories
    categories = sorted(merged["category_lvl_1"].unique())
    print("\nTesting multipliers on Fold 4 across individual categories (multiplier = 1.05, 1.10, 1.15):")
    print(f"{'Category':<22} {'Baseline WAPE':<15} {'Mult 1.05 WAPE':<16} {'Mult 1.10 WAPE':<16} {'Clipped Δ (1.05)':<18}")
    print("-" * 87)

    for cat in categories:
        cat_mask = merged["category_lvl_1"] == cat
        raw_cat = raw_blend_10.copy()
        
        # apply 1.05 multiplier to cat
        raw_mult_05 = raw_cat.copy()
        raw_mult_05[cat_mask] *= 1.05
        pred_05 = apply_p(raw_mult_05, 1.05, 0.65)
        df_05 = truth[KEYS].copy()
        df_05["quantity"] = pred_05
        m_05 = compute_all_metrics_fast(truth, df_05)
        c_05, _ = calc_scores(m_05, m_base)

        # apply 1.10 multiplier to cat
        raw_mult_10 = raw_cat.copy()
        raw_mult_10[cat_mask] *= 1.10
        pred_10 = apply_p(raw_mult_10, 1.05, 0.65)
        df_10 = truth[KEYS].copy()
        df_10["quantity"] = pred_10
        m_10 = compute_all_metrics_fast(truth, df_10)

        # category WAPE
        y_c = truth.loc[cat_mask, "quantity"].values
        denom = y_c.sum()
        wape_base = np.abs(y_c - pred_baseline[cat_mask]).sum() / denom * 100 if denom > 0 else 0
        wape_05 = np.abs(y_c - pred_05[cat_mask]).sum() / denom * 100 if denom > 0 else 0
        wape_10 = np.abs(y_c - pred_10[cat_mask]).sum() / denom * 100 if denom > 0 else 0

        delta_c = c_05 - c_base
        print(f"{cat:<22} {wape_base:<15.3f} {wape_05:<16.3f} {wape_10:<16.3f} {delta_c:<+18.6f}")

    # Test global multiplier on h=1..3
    print("\nTesting global back-to-school multiplier on h=1..2 (late September):")
    weeks_ordered = sorted(merged["week"].unique())
    h_early_mask = merged["week"].isin(weeks_ordered[:2])
    for mult in [1.02, 1.05, 1.10, 0.98, 0.95]:
        raw_mod = raw_blend_10.copy()
        raw_mod[h_early_mask] *= mult
        pred_mod = apply_p(raw_mod, 1.05, 0.65)
        df_mod = truth[KEYS].copy()
        df_mod["quantity"] = pred_mod
        m_mod = compute_all_metrics_fast(truth, df_mod)
        c_mod, u_mod = calc_scores(m_mod, m_base)
        print(f"  Multiplier {mult:.2f} on early weeks: Clipped = {c_mod:.6f} (Δ = {c_mod - c_base:+.6f}), Unclipped = {u_mod:.6f}")


if __name__ == "__main__":
    main()
