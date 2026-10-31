"""
adaptive_blending_rules.py

Centralized, unified module for pre-cutoff feature extraction and adaptive weighting rules.
Shared identically by:
- experiment_adaptive_timestral_blend.py
- generate_adaptive_timestral_candidate.py
- test_adaptive_candidate_roundtrip.py
- test_strict_roundtrip_audit.py

Guarantees 100% mathematical and code parity between historical experiments and production inference.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

from run_weekly_feature_experiments import WeeklyMatrices

KEYS = ["week", "warehouse_name", "article"]
FOLDS = [4, 3, 2, 1, 0]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
SELECTION_FOLDS = [4, 3, 2]
CHECK_FOLDS = [1, 0]


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    """Standard competition postprocessing: single application."""
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def compute_pre_cutoff_features(matrices: WeeklyMatrices, origin: int, pair_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Computes strictly pre-cutoff features for all 8,789 series based on history < origin.
    Origin 89 (Fold 4), 98 (Fold 3), 107 (Fold 2), 116 (Fold 1), 125 (Fold 0), 134 (Production).
    """
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

    # Stock features (available at history_pos)
    stock_snr = matrices.stock_nonpositive_rate[history_pos, :]
    stock_end = matrices.stock_end[history_pos, :]
    weeks_of_supply = np.where(mean13 > 0, stock_end / (mean13 + 1e-4), 999.0)

    # Percentiles for volume
    p90 = np.percentile(mean13, 90)
    p50 = np.percentile(mean13, 50)

    df_feats = matrices.pairs.copy()
    if pair_meta is not None:
        df_feats = pd.merge(df_feats, pair_meta, on=["warehouse_name", "article"], how="left")

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


# ==============================================================================
# Weighting Schemes
# ==============================================================================

def weight_fixed(w: float) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda df: np.full(len(df), w, dtype=np.float64)


def weight_volume_gated(df: pd.DataFrame) -> np.ndarray:
    """Volume gating: Low (<p50) 0.02, Mid (p50-p90) 0.10, High (>=p90) 0.18"""
    mean13 = df["mean13_feat"].values if "mean13_feat" in df.columns else df["mean13"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    w = np.where(mean13 >= p90, 0.18, np.where(mean13 >= p50, 0.10, 0.02))
    return w.astype(np.float64)


def weight_volume_gated_conservative(df: pd.DataFrame) -> np.ndarray:
    """Volume conservative: Low (<p50) 0.00, Mid (p50-p90) 0.10, High (>=p90) 0.15"""
    mean13 = df["mean13_feat"].values if "mean13_feat" in df.columns else df["mean13"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    w = np.where(mean13 >= p90, 0.15, np.where(mean13 >= p50, 0.10, 0.00))
    return w.astype(np.float64)


def weight_sparsity_and_volume(df: pd.DataFrame) -> np.ndarray:
    """Sparsity + Volume: Zero 0.00, Sparse 0.05, Mid 0.10, High 0.15"""
    nz13 = df["nz13"].values
    streak = df["zero_streak"].values
    mean13 = df["mean13_feat"].values if "mean13_feat" in df.columns else df["mean13"].values
    p90 = df["p90_vol"].values
    is_zero = (nz13 < 0.15) | (streak >= 6)
    is_high = (nz13 >= 0.50) & (mean13 >= p90)
    is_sparse = (nz13 < 0.50) & (~is_zero)
    w = np.where(is_zero, 0.00, np.where(is_high, 0.15, np.where(is_sparse, 0.05, 0.10)))
    return w.astype(np.float64)


def weight_horizon_decay(df: pd.DataFrame) -> np.ndarray:
    """Horizon decay: Near/Mid (h<=6) 0.12, Far (h>6) 0.06"""
    h = df["horizon"].values
    w = np.where(h <= 6, 0.12, 0.06)
    return w.astype(np.float64)


def weight_full_adaptive_multi_regime(df: pd.DataFrame) -> np.ndarray:
    """
    Selection Winner: Multi-Regime
    - Dead series (streak >= 8 or nz13 < 0.10): w = 0.00
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
    mean13 = df["mean13_feat"].values if "mean13_feat" in df.columns else df["mean13"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    h = df["horizon"].values

    is_dead = (streak >= 8) | (nz13 < 0.10)
    is_low = mean13 < p50
    is_high_reg = (nz13 >= 0.50) & (mean13 >= p90)
    is_sparse = (nz13 < 0.50)

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
    Candidate: Full Tuned
    - Dead series (streak >= 6 or nz13 < 0.15) OR low volume (<p50): w = 0.00
    - High volume (>=p90): w = 0.15 (h<=6), w = 0.12 (h>=7)
    - Mid volume (p50-p90): w = 0.10 (h<=6), w = 0.07 (h>=7)
    """
    nz13 = df["nz13"].values
    streak = df["zero_streak"].values
    mean13 = df["mean13_feat"].values if "mean13_feat" in df.columns else df["mean13"].values
    p90 = df["p90_vol"].values
    p50 = df["p50_vol"].values
    h = df["horizon"].values

    base_w = np.where((nz13 < 0.15) | (streak >= 6) | (mean13 < p50), 0.00,
                np.where(mean13 >= p90, 0.15, 0.10))
    adj_w = np.where(h >= 7, np.maximum(0.0, base_w - 0.03), base_w)
    return adj_w.astype(np.float64)


SCHEMES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
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
    "Adaptive: Full Multi-Regime (Dead 0, Low 2, RegMid 10/8, RegHigh 18/12)": weight_full_adaptive_multi_regime,
    "Adaptive: Full Tuned (Zero 0, Low 0, Mid 10/7, High 15/12)": weight_full_adaptive_tuned,
}
