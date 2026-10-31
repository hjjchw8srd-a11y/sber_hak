#!/usr/bin/env python3
"""
experiment_adaptive_timestral_blend.py

Priority A & B Research:
Evaluates adaptive blending schemes between Champion continuous raw predictions
and Timestral continuous predictions across the 5 competition folds.

Protocol:
- Selection Folds: [4, 3, 2] (Origins 89, 98, 107)
- Check Folds: [1, 0] (Origins 116, 125)
- Full 5-Folds: [4, 3, 2, 1, 0]

All segmentation rules and weights depend strictly on pre-cutoff features.
Outputs full metrics: Clipped Score, Unclipped Ratio, SMAPE, WAPE, RMSE, Bias, RMSLE.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Callable

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
SELECTION_FOLDS = [4, 3, 2]
CHECK_FOLDS = [1, 0]


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


def compute_pre_cutoff_features(matrices: WeeklyMatrices, origin: int, pair_meta: pd.DataFrame) -> pd.DataFrame:
    """Computes strictly pre-cutoff features for all 8789 series."""
    history_pos = origin - 1

    # 13-week & 4-week window
    Q13 = matrices.quantity[origin - 13 : origin, :]
    Q4 = matrices.quantity[origin - 4 : origin, :]

    mean13 = Q13.mean(axis=0)
    nz13 = (Q13 > 0).mean(axis=0)
    mean4 = Q4.mean(axis=0)
    growth_4_13 = np.where(mean13 > 0, mean4 / (mean13 + 1e-4), 1.0)

    # Consecutive zero streak immediately preceding cutoff
    n_series = matrices.quantity.shape[1]
    zero_streaks = np.zeros(n_series, dtype=int)
    for i in range(n_series):
        streak = 0
        for t in range(origin - 1, -1, -1):
            if matrices.quantity[t, i] == 0:
                streak += 1
            else:
                break
        zero_streaks[i] = streak

    # Stock features
    stock_snr = matrices.stock_nonpositive_rate[history_pos, :]
    stock_end = matrices.stock_end[history_pos, :]
    weeks_of_supply = np.where(mean13 > 0, stock_end / (mean13 + 1e-4), 999.0)

    # Percentiles for volume
    p90 = np.percentile(mean13, 90)
    p75 = np.percentile(mean13, 75)
    p50 = np.percentile(mean13, 50)
    p25 = np.percentile(mean13, 25)

    df_feats = pd.merge(matrices.pairs, pair_meta, on=["warehouse_name", "article"], how="left")
    df_feats["mean13"] = mean13
    df_feats["nz13"] = nz13
    df_feats["mean4"] = mean4
    df_feats["growth_4_13"] = growth_4_13
    df_feats["zero_streak"] = zero_streaks
    df_feats["stock_snr"] = stock_snr
    df_feats["stock_end"] = stock_end
    df_feats["weeks_of_supply"] = weeks_of_supply
    df_feats["p90_vol"] = p90
    df_feats["p50_vol"] = p50

    return df_feats


def load_fold_data(matrices: WeeklyMatrices, pair_meta: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Loads and aligns Champion OOF, Timestral OOF, pre-cutoff series features, and horizon."""
    origin = FOLD_ORIGINS[fold]
    oof_dir = ROOT / "experiments" / "champion_oof"
    champ_path = oof_dir / f"champion_scale_1m_fold_{fold}.csv.gz"
    tim_path = (
        ROOT / "experiments" / f"timestral_lora_fold{fold}" / "keyed_predictions" / "pilot"
        / f"timestral_209m_lora_fold{fold}_pilot_keyed.csv.gz"
    )

    df_champ = pd.read_csv(champ_path)
    df_tim = pd.read_csv(tim_path)

    merged = pd.merge(
        df_champ[KEYS + ["raw_champion", "y_true", "mean13"]],
        df_tim[KEYS + ["y_pred_continuous"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 79101

    # Horizon mapping (1..9)
    weeks_ordered = sorted(merged["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged["horizon"] = merged["week"].map(week_to_h)

    # Series pre-cutoff features
    df_feats = compute_pre_cutoff_features(matrices, origin, pair_meta)
    merged = pd.merge(merged, df_feats, on=["warehouse_name", "article"], how="left")

    return merged


# ==============================================================================
# Weighting Schemes
# ==============================================================================

def weight_fixed(w: float) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda df: np.full(len(df), w, dtype=np.float64)


def weight_volume_gated(df: pd.DataFrame) -> np.ndarray:
    """
    Volume gating:
    - Low-volume (<p50): w = 0.02
    - Mid-volume (p50-p90): w = 0.10
    - High-volume (>=p90): w = 0.18
    """
    mean13 = df["mean13_y"].values  # from df_feats
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values

    w = np.where(mean13 >= p90, 0.18, np.where(mean13 >= p50, 0.10, 0.02))
    return w.astype(np.float64)


def weight_volume_gated_conservative(df: pd.DataFrame) -> np.ndarray:
    """
    Volume gating conservative:
    - Low-volume (<p50): w = 0.00
    - Mid-volume (p50-p90): w = 0.10
    - High-volume (>=p90): w = 0.15
    """
    mean13 = df["mean13_y"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values

    w = np.where(mean13 >= p90, 0.15, np.where(mean13 >= p50, 0.10, 0.00))
    return w.astype(np.float64)


def weight_sparsity_and_volume(df: pd.DataFrame) -> np.ndarray:
    """
    Sparsity & volume rule:
    - Zero-heavy (nz13 < 0.15 or zero_streak >= 6): w = 0.00
    - Sparse (0.15 <= nz13 < 0.50): w = 0.05
    - Regular (nz13 >= 0.50):
        - if mean13 >= p90: w = 0.15
        - else: w = 0.10
    """
    nz13 = df["nz13"].values
    streak = df["zero_streak"].values
    mean13 = df["mean13_y"].values
    p90 = df["p90_vol"].values

    is_zero = (nz13 < 0.15) | (streak >= 6)
    is_high = (nz13 >= 0.50) & (mean13 >= p90)
    is_sparse = (nz13 < 0.50) & (~is_zero)

    w = np.where(is_zero, 0.00, np.where(is_high, 0.15, np.where(is_sparse, 0.05, 0.10)))
    return w.astype(np.float64)


def weight_horizon_decay(df: pd.DataFrame) -> np.ndarray:
    """
    Horizon decay:
    - Near (h=1..3): w = 0.12
    - Mid (h=4..6): w = 0.12
    - Far (h=7..9): w = 0.06
    """
    h = df["horizon"].values
    w = np.where(h <= 6, 0.12, 0.06)
    return w.astype(np.float64)


def weight_full_adaptive(df: pd.DataFrame) -> np.ndarray:
    """
    Comprehensive rule combining volume, sparsity, and horizon:
    - Inactive / dead series (streak >= 8 or nz13 < 0.10): w = 0.00
    - Low-volume (<p50): w = 0.02
    - High-volume regular (nz13 >= 0.50 and mean13 >= p90):
        - Near/Mid (h <= 6): w = 0.18
        - Far (h > 6): w = 0.12
    - Regular mid-volume:
        - Near/Mid (h <= 6): w = 0.10
        - Far (h > 6): w = 0.08
    - Sparse: w = 0.05
    """
    nz13 = df["nz13"].values
    streak = df["zero_streak"].values
    mean13 = df["mean13_y"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    h = df["horizon"].values

    is_dead = (streak >= 8) | (nz13 < 0.10)
    is_low = mean13 < p50
    is_high_reg = (nz13 >= 0.50) & (mean13 >= p90)
    is_sparse = (nz13 < 0.50)

    # Base allocation
    w = np.where(is_dead, 0.00,
        np.where(is_low, 0.02,
            np.where(is_high_reg, np.where(h <= 6, 0.18, 0.12),
                np.where(is_sparse, 0.05, np.where(h <= 6, 0.10, 0.08))
            )
        )
    )
    return w.astype(np.float64)


def weight_full_adaptive_tuned(df: pd.DataFrame) -> np.ndarray:
    """
    Refined adaptive weighting:
    - Dead series (streak >= 6 or nz13 < 0.15): w = 0.00
    - Low volume (<p50): w = 0.00
    - High volume (>=p90): w = 0.15
    - Mid volume (p50-p90): w = 0.10
    - Horizon adjustment: for h >= 7, reduce weight by 0.03 (min 0)
    """
    nz13 = df["nz13"].values
    streak = df["zero_streak"].values
    mean13 = df["mean13_y"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    h = df["horizon"].values

    base_w = np.where((nz13 < 0.15) | (streak >= 6) | (mean13 < p50), 0.00,
                np.where(mean13 >= p90, 0.15, 0.10))
    adj_w = np.where(h >= 7, np.maximum(0.0, base_w - 0.03), base_w)
    return adj_w.astype(np.float64)


SCHEMES = {
    "Fixed w=0.00 (Champion)": weight_fixed(0.00),
    "Fixed w=0.05": weight_fixed(0.05),
    "Fixed w=0.08": weight_fixed(0.08),
    "Fixed w=0.10 (Current Best)": weight_fixed(0.10),
    "Fixed w=0.12": weight_fixed(0.12),
    "Fixed w=0.15": weight_fixed(0.15),
    "Fixed w=0.20": weight_fixed(0.20),
    "Adaptive: Volume Gated (0.02 / 0.10 / 0.18)": weight_volume_gated,
    "Adaptive: Volume Conservative (0.00 / 0.10 / 0.15)": weight_volume_gated_conservative,
    "Adaptive: Sparsity + Volume (0.00 / 0.05 / 0.10 / 0.15)": weight_sparsity_and_volume,
    "Adaptive: Horizon Decay (0.12 near/mid, 0.06 far)": weight_horizon_decay,
    "Adaptive: Full Multi-Regime (Dead 0, Low 2, RegMid 10/8, RegHigh 18/12)": weight_full_adaptive,
    "Adaptive: Full Tuned (Zero 0, Low 0, Mid 10/7, High 15/12)": weight_full_adaptive_tuned,
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 115)
    print("TIMESTRAL ADAPTIVE BLENDING EXPERIMENT (SELECTION FOLDS [4,3,2] VS CHECK FOLDS [1,0])")
    print("=" * 115)

    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    print("Loading fold data...")
    fold_data = {}
    for fold in FOLDS:
        print(f"  Loading fold {fold}...")
        fold_data[fold] = load_fold_data(matrices, pair_meta, fold)

    results = []

    for name, scheme_fn in SCHEMES.items():
        for fold in FOLDS:
            df = fold_data[fold]
            w = scheme_fn(df)

            raw_c = df["raw_champion"].values.astype(np.float64)
            raw_t = df["y_pred_continuous"].values.astype(np.float64)

            # Continuous blend
            raw_blend = (1.0 - w) * raw_c + w * raw_t
            pred = apply_p(raw_blend, 1.05, 0.65)

            truth = df[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
            base = df[KEYS + ["mean13_x"]].rename(columns={"mean13_x": "quantity"})
            m_base = compute_all_metrics_fast(truth, base)

            df_pred = truth[KEYS].copy()
            df_pred["quantity"] = pred
            m_pred = compute_all_metrics_fast(truth, df_pred)

            c_score, u_ratio = calc_scores(m_pred, m_base)

            results.append({
                "scheme": name,
                "fold": fold,
                "stage": "Selection" if fold in SELECTION_FOLDS else "Check",
                "clipped_score": c_score,
                "unclipped_ratio": u_ratio,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
                "mean_weight": float(w.mean()),
            })

    res_df = pd.DataFrame(results)

    # Summaries
    print("\n" + "=" * 115)
    print("EXPERIMENT SUMMARY TABLE (SORTED BY 5-FOLD CLIPPED SCORE)")
    print("=" * 115)
    print(
        f"{'Scheme':<45} {'Mean W':<7} "
        f"{'Sel Clipped [4,3,2]':<20} {'Chk Clipped [1,0]':<18} "
        f"{'Full 5F Clipped':<16} {'Full 5F Unclipped':<17} {'WAPE (%)':<9}"
    )
    print("-" * 115)

    summary_rows = []
    for name in SCHEMES.keys():
        s_df = res_df[res_df["scheme"] == name]
        sel_c = s_df[s_df["stage"] == "Selection"]["clipped_score"].mean()
        chk_c = s_df[s_df["stage"] == "Check"]["clipped_score"].mean()
        full_c = s_df["clipped_score"].mean()
        full_u = s_df["unclipped_ratio"].mean()
        full_wape = s_df["wape"].mean()
        full_rmse = s_df["rmse"].mean()
        full_smape = s_df["smape"].mean()
        mean_w = s_df["mean_weight"].mean()

        summary_rows.append({
            "scheme": name,
            "mean_w": mean_w,
            "sel_clipped": sel_c,
            "chk_clipped": chk_c,
            "full_clipped": full_c,
            "full_unclipped": full_u,
            "wape": full_wape,
            "rmse": full_rmse,
            "smape": full_smape,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("full_clipped")

    for _, row in summary_df.iterrows():
        print(
            f"{row['scheme']:<45} {row['mean_w']:<7.3f} "
            f"{row['sel_clipped']:<20.6f} {row['chk_clipped']:<18.6f} "
            f"{row['full_clipped']:<16.6f} {row['full_unclipped']:<17.6f} {row['wape']:<9.3f}"
        )

    # Save results to json
    out_dir = ROOT / "experiments" / "timestral_feature_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "adaptive_blend_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)
    print(f"\nDetailed summary saved to {summary_path}")


if __name__ == "__main__":
    main()
