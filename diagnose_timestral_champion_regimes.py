#!/usr/bin/env python3
"""
diagnose_timestral_champion_regimes.py

Detailed diagnostic analysis of where Timestral reduces Champion errors across folds,
using ONLY features known strictly prior to each forecast cutoff:
- Demand regimes: zero-heavy (nz13 < 0.15), sparse (0.15 <= nz13 < 0.50), regular (nz13 >= 0.50)
- Volume regimes: high-volume (top 10% mean13), medium (50-90%), low (<50%)
- Stock regimes: out-of-stock (SNR >= 0.50), positive stock (SNR < 0.50)
- Horizon breakdown: near (h=1..3), mid (h=4..6), far (h=7..9)
- Category level 1 breakdown

Computes WAPE, RMSE, SMAPE for:
1) Pure Champion (w=0.00)
2) Pure Timestral (w=1.00)
3) Fixed Blend (w=0.10)
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
from run_weekly_feature_experiments import WeeklyMatrices

KEYS = ["week", "warehouse_name", "article"]
FOLDS = [4, 3, 2, 1, 0]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"count": len(y_true), "volume": float(denom), "wape": wape, "rmse": rmse, "mae": mae}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 100)
    print("TIMESTRAL VS CHAMPION REGIME DIAGNOSIS (STRICT PRE-CUTOFF SEGMENTS)")
    print("=" * 100)

    # Load dataset & matrices
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    all_fold_dfs = []

    for fold in FOLDS:
        origin = FOLD_ORIGINS[fold]
        history_pos = origin - 1

        # 1. Historical pre-cutoff features per series (8789 series)
        Q13 = matrices.quantity[origin - 13 : origin, :]
        mean13 = Q13.mean(axis=0)
        nz13 = (Q13 > 0).mean(axis=0)

        p90_vol = np.percentile(mean13, 90)
        p50_vol = np.percentile(mean13, 50)

        stock_snr = matrices.stock_nonpositive_rate[history_pos, :]
        stock_end = matrices.stock_end[history_pos, :]

        # Demand regime
        demand_regime = np.where(nz13 < 0.15, "Zero-Heavy (<15%)", np.where(nz13 >= 0.50, "Regular (>=50%)", "Sparse (15-50%)"))
        # Volume regime
        volume_regime = np.where(mean13 >= p90_vol, "High-Volume (Top 10%)", np.where(mean13 >= p50_vol, "Mid-Volume (50-90%)", "Low-Volume (<50%)"))
        # Stock regime
        stock_regime = np.where(stock_snr >= 0.50, "Stock Depleted (SNR>=0.5)", "Stock Healthy (SNR<0.5)")

        series_df = pd.merge(matrices.pairs, pair_meta, on=["warehouse_name", "article"], how="left")
        series_df["demand_regime"] = demand_regime
        series_df["volume_regime"] = volume_regime
        series_df["stock_regime"] = stock_regime
        series_df["hist_mean13"] = mean13
        series_df["hist_nz13"] = nz13

        # 2. Load fold predictions
        champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
        tim_path = (
            ROOT / "experiments" / f"timestral_lora_fold{fold}" / "keyed_predictions" / "pilot"
            / f"timestral_209m_lora_fold{fold}_pilot_keyed.csv.gz"
        )

        df_champ = pd.read_csv(champ_path)
        df_tim = pd.read_csv(tim_path)

        merged = pd.merge(
            df_champ[KEYS + ["raw_champion", "y_true"]],
            df_tim[KEYS + ["y_pred_continuous"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        assert len(merged) == 79101

        # Add horizon index 1..9
        weeks_ordered = sorted(merged["week"].unique())
        week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
        merged["horizon"] = merged["week"].map(week_to_h)
        merged["horizon_group"] = np.where(merged["horizon"] <= 3, "Near (h=1..3)", np.where(merged["horizon"] <= 6, "Mid (h=4..6)", "Far (h=7..9)"))

        # Merge pre-cutoff series metadata
        merged = pd.merge(merged, series_df, on=["warehouse_name", "article"], how="left")
        merged["fold"] = fold

        raw_c = merged["raw_champion"].values.astype(np.float64)
        raw_t = merged["y_pred_continuous"].values.astype(np.float64)

        merged["pred_champ"] = apply_p(raw_c, 1.05, 0.65)
        merged["pred_tim"] = apply_p(raw_t, 1.05, 0.65)
        merged["pred_w05"] = apply_p(0.95 * raw_c + 0.05 * raw_t, 1.05, 0.65)
        merged["pred_w10"] = apply_p(0.90 * raw_c + 0.10 * raw_t, 1.05, 0.65)
        merged["pred_w20"] = apply_p(0.80 * raw_c + 0.20 * raw_t, 1.05, 0.65)

        all_fold_dfs.append(merged)

    full_df = pd.concat(all_fold_dfs, ignore_index=True)
    print(f"Total rows analyzed across 5 folds: {len(full_df):,}")

    # Evaluate across dimensions
    dimensions = ["demand_regime", "volume_regime", "stock_regime", "horizon_group"]

    for dim in dimensions:
        print("\n" + "=" * 105)
        print(f"SEGMENTATION BY: {dim.upper()}")
        print(f"{'Segment':<28} {'Rows':<8} {'Vol Share':<10} {'WAPE Champ':<11} {'WAPE Tim':<10} {'WAPE w10':<10} {'WAPE w20':<10} {'Delta w10':<10}")
        print("=" * 105)

        total_vol = full_df["y_true"].sum()
        for seg_name, grp in full_df.groupby(dim, observed=True):
            y_t = grp["y_true"].values
            m_c = calc_metrics(y_t, grp["pred_champ"].values)
            m_tim = calc_metrics(y_t, grp["pred_tim"].values)
            m_w05 = calc_metrics(y_t, grp["pred_w05"].values)
            m_w10 = calc_metrics(y_t, grp["pred_w10"].values)
            m_w20 = calc_metrics(y_t, grp["pred_w20"].values)

            vol_share = m_c["volume"] / total_vol * 100.0
            delta_w10 = m_w10["wape"] - m_c["wape"]

            print(
                f"{seg_name:<28} {len(grp):<8} {vol_share:<9.1f}% "
                f"{m_c['wape']:<11.3f} {m_tim['wape']:<10.3f} {m_w10['wape']:<10.3f} {m_w20['wape']:<10.3f} "
                f"{delta_w10:<+10.3f}"
            )

    # Category analysis
    print("\n" + "=" * 105)
    print("SEGMENTATION BY CATEGORY LEVEL 1")
    print(f"{'Category Lvl 1':<28} {'Rows':<8} {'Vol Share':<10} {'WAPE Champ':<11} {'WAPE Tim':<10} {'WAPE w10':<10} {'Delta w10':<10}")
    print("=" * 105)
    for cat_name, grp in full_df.groupby("category_lvl_1", observed=True):
        y_t = grp["y_true"].values
        m_c = calc_metrics(y_t, grp["pred_champ"].values)
        m_tim = calc_metrics(y_t, grp["pred_tim"].values)
        m_w10 = calc_metrics(y_t, grp["pred_w10"].values)
        vol_share = m_c["volume"] / total_vol * 100.0
        delta_w10 = m_w10["wape"] - m_c["wape"]
        print(f"{str(cat_name):<28} {len(grp):<8} {vol_share:<9.1f}% {m_c['wape']:<11.3f} {m_tim['wape']:<10.3f} {m_w10['wape']:<10.3f} {delta_w10:<+10.3f}")

    print("\n" + "=" * 100)
    print("DIAGNOSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
