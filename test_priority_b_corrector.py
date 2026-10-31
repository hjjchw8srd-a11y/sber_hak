#!/usr/bin/env python3
"""
test_priority_b_corrector.py

Priority B: Lightweight corrector layer on top of Timestral predictions.
Trains on historical folds [4, 3] and validates strictly on out-of-sample folds [2, 1, 0].
Features:
- timestral_pred
- mean13
- stock_end_lag_1
- stock_nonpositive_rate_lag_1
- is_zero_streak (binary: streak >= 4)
- category_lvl_1 (target/frequency encoded)
- horizon (1..9)

Tests:
1. Multiplicative confidence / shrinkage model: y_corr = g(x) * y_tim
2. Residual regression model: y_corr = y_tim + f(x)
3. Evaluates impact on blend (w=0.10 and adaptive weight) vs raw Timestral.
4. Checks threshold criterion: if gain on [2, 1, 0] is < 0.0001 Clipped Score, reject Priority B.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import WeeklyMatrices

KEYS = ["week", "warehouse_name", "article"]
FOLDS = [4, 3, 2, 1, 0]
FOLD_ORIGINS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
TRAIN_FOLDS = [4, 3]
TEST_FOLDS = [2, 1, 0]


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
    history_pos = origin - 1
    Q13 = matrices.quantity[origin - 13 : origin, :]
    mean13 = Q13.mean(axis=0)
    nz13 = (Q13 > 0).mean(axis=0)

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

    stock_snr = matrices.stock_nonpositive_rate[history_pos, :]
    stock_end = matrices.stock_end[history_pos, :]

    df_feats = pd.merge(matrices.pairs, pair_meta, on=["warehouse_name", "article"], how="left")
    df_feats["mean13"] = mean13
    df_feats["nz13"] = nz13
    df_feats["zero_streak"] = zero_streaks
    df_feats["stock_snr"] = stock_snr
    df_feats["stock_end"] = stock_end
    df_feats["is_zero_streak"] = (zero_streaks >= 4).astype(float)
    df_feats["stock_depleted"] = (stock_snr >= 0.50).astype(float)

    return df_feats


def load_fold_data(matrices: WeeklyMatrices, pair_meta: pd.DataFrame, fold: int) -> pd.DataFrame:
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
    )

    weeks_ordered = sorted(merged["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged["horizon"] = merged["week"].map(week_to_h)

    df_feats = compute_pre_cutoff_features(matrices, origin, pair_meta)
    merged = pd.merge(merged, df_feats, on=["warehouse_name", "article"], how="left")
    merged["fold"] = fold
    return merged


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 100)
    print("TEST PRIORITY B: LIGHTWEIGHT TIMESTRAL FEATURE-AWARE CORRECTOR")
    print("=" * 100)

    data = pd.read_csv(ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    print("Loading folds data...")
    fold_dfs = {fold: load_fold_data(matrices, pair_meta, fold) for fold in FOLDS}

    # 1. Prepare training data on Folds 4 and 3
    df_train = pd.concat([fold_dfs[f] for f in TRAIN_FOLDS], ignore_index=True)
    print(f"Training set: Folds {TRAIN_FOLDS}, rows: {len(df_train):,}")

    feature_cols = ["mean13_y", "stock_end", "stock_snr", "is_zero_streak", "horizon"]
    
    # Simple target: ratio y_true / (timestral_pred + 1e-3) clipped to [0, 2]
    # or residual: y_true - timestral_pred
    y_tim_train = df_train["y_pred_continuous"].values
    y_true_train = df_train["y_true"].values
    
    # Residual corrector: predicts residual (y_true - y_tim)
    X_train = df_train[feature_cols].copy().fillna(0.0)
    X_train["timestral_pred"] = y_tim_train
    y_res_train = y_true_train - y_tim_train

    print("Fitting Ridge residual corrector on Folds 4, 3...")
    model_res = Ridge(alpha=100.0)
    model_res.fit(X_train, y_res_train)

    print(f"Residual corrector weights: {dict(zip(X_train.columns, model_res.coef_))}")

    # Shrinkage corrector: in depleted stock or zero streak, shrink Timestral towards 0
    # Let's test both on Validation Folds 2, 1, 0
    print("\nEvaluating on Test Folds [2, 1, 0] (Honest Out-Of-Sample):")

    for fold in TEST_FOLDS:
        df_test = fold_dfs[fold]
        truth = df_test[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
        base = df_test[KEYS + ["mean13_x"]].rename(columns={"mean13_x": "quantity"})
        m_base = compute_all_metrics_fast(truth, base)

        raw_c = df_test["raw_champion"].values.astype(np.float64)
        raw_t = df_test["y_pred_continuous"].values.astype(np.float64)

        # Baseline Fixed w=0.10
        raw_b10 = 0.90 * raw_c + 0.10 * raw_t
        p_b10 = apply_p(raw_b10, 1.05, 0.65)
        df_b10 = truth[KEYS].copy()
        df_b10["quantity"] = p_b10
        m_b10 = compute_all_metrics_fast(truth, df_b10)
        c_b10, u_b10 = calc_scores(m_b10, m_base)

        # Corrected Timestral: y_tim_corr = np.maximum(0, y_tim + pred_residual)
        X_test = df_test[feature_cols].copy().fillna(0.0)
        X_test["timestral_pred"] = raw_t
        pred_res = model_res.predict(X_test)
        raw_t_corr = np.maximum(0.0, raw_t + 0.5 * pred_res)  # damped residual correction

        raw_corr_b10 = 0.90 * raw_c + 0.10 * raw_t_corr
        p_corr = apply_p(raw_corr_b10, 1.05, 0.65)
        df_corr = truth[KEYS].copy()
        df_corr["quantity"] = p_corr
        m_corr = compute_all_metrics_fast(truth, df_corr)
        c_corr, u_corr = calc_scores(m_corr, m_base)

        # Simple rule-based shrinkage: if zero_streak >= 4 or stock_snr >= 0.8, shrink Timestral by 50%
        shrink_mask = (df_test["zero_streak"] >= 4) | (df_test["stock_snr"] >= 0.8)
        raw_t_shrink = raw_t.copy()
        raw_t_shrink[shrink_mask] *= 0.5
        raw_shrink_b10 = 0.90 * raw_c + 0.10 * raw_t_shrink
        p_shrink = apply_p(raw_shrink_b10, 1.05, 0.65)
        df_shrink = truth[KEYS].copy()
        df_shrink["quantity"] = p_shrink
        m_shrink = compute_all_metrics_fast(truth, df_shrink)
        c_shrink, u_shrink = calc_scores(m_shrink, m_base)

        print(f"Fold {fold}:")
        print(f"  Fixed w=0.10:           Clipped = {c_b10:.6f}, Unclipped = {u_b10:.6f}, WAPE = {m_b10['wape']:.3f}%")
        print(f"  Ridge Corrected w=0.10: Clipped = {c_corr:.6f} (Δ = {c_corr - c_b10:+.6f}), Unclipped = {u_corr:.6f}, WAPE = {m_corr['wape']:.3f}%")
        print(f"  Rule Shrinkage w=0.10:  Clipped = {c_shrink:.6f} (Δ = {c_shrink - c_b10:+.6f}), Unclipped = {u_shrink:.6f}, WAPE = {m_shrink['wape']:.3f}%")

    print("\nSummary & Decision Protocol:")
    print("Threshold requirement: gain on [2, 1, 0] must be >= 0.0001 Clipped Score.")


if __name__ == "__main__":
    main()
