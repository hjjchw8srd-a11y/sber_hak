#!/usr/bin/env python3
"""
train_lgbm_holiday_ablation.py

Strict ablation of Russian Federal Holiday features on LightGBM B across 5 evaluation folds.
Uses identical 52 origins, 350,000 sampled rows, seed 42, identical hyperparameters,
and the prescribed blending rule:
    E_holidays_raw = E_raw + 0.30 * (L_holidays - L_control)
    final_prediction = P(E_holidays_raw)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from holiday_calendar import build_holiday_feature_dataframe, HOLIDAY_FEATURE_COLUMNS

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "holidays"
KEYED_DIR = OUTPUT_DIR / "keyed_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

FOLD_ORIGINS = {
    4: 89,   # 2025-09-15 — 2025-11-10 (Selection)
    3: 98,   # 2025-11-17 — 2026-01-12 (Selection)
    2: 107,  # 2026-01-19 — 2026-03-16 (Selection)
    1: 116,  # 2026-03-23 — 2026-05-18 (Check)
    0: 125,  # 2026-05-25 — 2026-07-20 (Check)
}

PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.0495,
    "num_leaves": 29,
    "min_child_samples": 51,
    "subsample": 0.727,
    "colsample_bytree": 0.728,
    "reg_alpha": 0.0164,
    "reg_lambda": 0.0057,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

def apply_postprocess(x: np.ndarray | pd.Series) -> np.ndarray:
    z = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    return np.maximum(np.rint(np.where(z < 0.65, 0.0, z)), 0.0)

def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }

def train_lgbm_pair(matrices: WeeklyMatrices, origin: int, holiday_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    val_start = matrices.weeks[origin]
    latest_train_origin = origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - 51)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)

    train_df, target_arr, target_pos = matrices.make_rows(train_origins, 350_000, seed=42)
    val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)

    # Attach holiday features based on target week dates
    train_weeks = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in target_pos]
    val_weeks = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]

    train_holidays = pd.DataFrame({"week": train_weeks}).merge(holiday_df, on="week", how="left")
    val_holidays = pd.DataFrame({"week": val_weeks}).merge(holiday_df, on="week", how="left")

    base_cols = columns_for_experiment("base_calendar_network")
    cat_columns = [c for c in CAT_COLUMNS if c in base_cols]

    # Model 1: Control (24 base features)
    X_train_ctrl = train_df[base_cols].copy()
    X_val_ctrl = val_df[base_cols].copy()
    for col in cat_columns:
        X_train_ctrl[col] = X_train_ctrl[col].astype("category")
        X_val_ctrl[col] = X_val_ctrl[col].astype("category")

    t0 = time.perf_counter()
    clf_ctrl = lgb.LGBMClassifier(**PARAMS)
    clf_ctrl.fit(X_train_ctrl, (target_arr > 0).astype(int))
    p_sale_ctrl = clf_ctrl.predict_proba(X_val_ctrl)[:, 1]

    pos_idx = target_arr > 0
    reg_ctrl = lgb.LGBMRegressor(**PARAMS)
    reg_ctrl.fit(X_train_ctrl.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_ctrl = np.maximum(np.expm1(reg_ctrl.predict(X_val_ctrl)), 0.0)
    exp_ctrl = p_sale_ctrl * cond_ctrl
    l_ctrl = np.where(p_sale_ctrl < 0.5, 0.0, exp_ctrl)
    t_ctrl = time.perf_counter() - t0

    # Model 2: Holidays (24 base features + 9 holiday features)
    holiday_cols = [c for c in HOLIDAY_FEATURE_COLUMNS if c in holiday_df.columns]
    X_train_hol = X_train_ctrl.copy()
    X_val_hol = X_val_ctrl.copy()
    for hc in holiday_cols:
        X_train_hol[hc] = train_holidays[hc].values.astype(float)
        X_val_hol[hc] = val_holidays[hc].values.astype(float)

    t1 = time.perf_counter()
    clf_hol = lgb.LGBMClassifier(**PARAMS)
    clf_hol.fit(X_train_hol, (target_arr > 0).astype(int))
    p_sale_hol = clf_hol.predict_proba(X_val_hol)[:, 1]

    reg_hol = lgb.LGBMRegressor(**PARAMS)
    reg_hol.fit(X_train_hol.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_hol = np.maximum(np.expm1(reg_hol.predict(X_val_hol)), 0.0)
    exp_hol = p_sale_hol * cond_hol
    l_hol = np.where(p_sale_hol < 0.5, 0.0, exp_hol)
    t_hol = time.perf_counter() - t1

    info = {
        "origin": origin,
        "train_rows": len(train_df),
        "num_origins": len(train_origins),
        "ctrl_time_sec": t_ctrl,
        "hol_time_sec": t_hol,
        "holiday_features_count": len(holiday_cols),
    }
    return l_ctrl, l_hol, info

def main():
    print("=" * 80)
    print("LIGHTGBM B HOLIDAYS ABLATION (5 FOLDS)")
    print("=" * 80)

    print("Loading weekly dataset...")
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("Building Russian holiday feature table...")
    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays)

    fold_metrics = []

    # Process all folds (4, 3, 2 first, then 1, 0)
    for fold, origin in FOLD_ORIGINS.items():
        role = "Selection" if fold in (4, 3, 2) else "Check"
        print(f"\nProcessing Fold {fold} ({role}, origin {origin})...")

        # Load ground truth & baseline
        val_weeks = weeks[origin : origin + HORIZON]
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        # Load control Blend B keyed oof
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged) == 79101

        e_raw = merged["raw_blend"].values.astype(float)
        t_raw = merged["after_comp_tft"].values.astype(float)
        l_saved_ctrl = merged["after_comp_lgbm"].values.astype(float)

        # Train models
        l_ctrl, l_hol, info = train_lgbm_pair(matrices, origin, holiday_df)

        # Verify reproduction
        diff_ctrl = np.abs(l_ctrl - l_saved_ctrl).max()
        assert diff_ctrl < 1e-10, f"Fold {fold}: Control reproduction failed! Max diff: {diff_ctrl}"

        # Evaluate candidate models
        # 1. Primary Control: P(E_raw)
        pred_ctrl_blend = apply_postprocess(e_raw)
        # 2. Standalone Control LightGBM: P(L_ctrl)
        pred_ctrl_lgbm = apply_postprocess(l_ctrl)
        # 3. Standalone Holidays LightGBM: P(L_hol)
        pred_hol_lgbm = apply_postprocess(l_hol)
        # 4. Holidays Blend: E_hol_raw = E_raw + 0.30 * (L_hol - L_ctrl)
        e_hol_raw = e_raw + 0.30 * (l_hol - l_ctrl)
        pred_hol_blend = apply_postprocess(e_hol_raw)

        # Helper to compute and format
        def eval_variant(var_name, pred_arr):
            p_df = merged[KEYS].copy()
            p_df["quantity"] = pred_arr
            m = compute_metrics(truth, p_df, b_df)
            fold_metrics.append({
                "fold": fold,
                "role": role,
                "variant": var_name,
                **m,
            })
            return m

        m_ctrl_b = eval_variant("Control_Blend_B", pred_ctrl_blend)
        m_hol_b = eval_variant("Holidays_Blend_B", pred_hol_blend)
        m_ctrl_l = eval_variant("Standalone_LGBM_Control", pred_ctrl_lgbm)
        m_hol_l = eval_variant("Standalone_LGBM_Holidays", pred_hol_lgbm)

        delta_unclipped = m_hol_b["unclipped_ratio"] - m_ctrl_b["unclipped_ratio"]
        delta_clipped = m_hol_b["clipped_score"] - m_ctrl_b["clipped_score"]

        print(f"  Control Blend B:  Unclipped = {m_ctrl_b['unclipped_ratio']:.6f}, Clipped = {m_ctrl_b['clipped_score']:.6f}")
        print(f"  Holidays Blend B: Unclipped = {m_hol_b['unclipped_ratio']:.6f}, Clipped = {m_hol_b['clipped_score']:.6f}")
        print(f"  Delta (Hol - Ctrl): Unclipped = {delta_unclipped:+.6f}, Clipped = {delta_clipped:+.6f}")
        print(f"  Standalone LGBM: Control Unclipped = {m_ctrl_l['unclipped_ratio']:.6f} vs Holidays Unclipped = {m_hol_l['unclipped_ratio']:.6f}")

        # Save keyed predictions
        save_df = merged[KEYS].copy()
        save_df["lgbm_ctrl"] = l_ctrl
        save_df["lgbm_holidays"] = l_hol
        save_df["e_raw_ctrl"] = e_raw
        save_df["e_raw_holidays"] = e_hol_raw
        save_df["pred_blend_ctrl"] = pred_ctrl_blend
        save_df["pred_blend_holidays"] = pred_hol_blend
        save_df.to_csv(KEYED_DIR / f"lgbm_holidays_fold_{fold}.csv.gz", index=False, compression="gzip")

    # Summary
    df_res = pd.DataFrame(fold_metrics)
    df_res.to_csv(OUTPUT_DIR / "holiday_metrics_by_fold.csv", index=False)

    print("\n" + "=" * 80)
    print("SELECTION FOLDS (4, 3, 2) SUMMARY:")
    print("=" * 80)
    df_sel = df_res[df_res["fold"].isin([4, 3, 2])]
    pvt_sel = df_sel.pivot_table(index="variant", values=["unclipped_ratio", "clipped_score", "smape", "wape", "rmse"], aggfunc="mean")
    print(pvt_sel.round(6))

    print("\n" + "=" * 80)
    print("ALL 5 FOLDS SUMMARY:")
    print("=" * 80)
    pvt_all = df_res.pivot_table(index="variant", values=["unclipped_ratio", "clipped_score", "smape", "wape", "rmse"], aggfunc="mean")
    print(pvt_all.round(6))

if __name__ == "__main__":
    main()
