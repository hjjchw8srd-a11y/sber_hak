#!/usr/bin/env python3
"""
train_lgbm_statutory_ablation.py

Strict ablation of Block A Statutory Russian Federal Holiday features (Article 112 ТК РФ)
on LightGBM B across all 5 evaluation folds.
Uses identical 52 origins, 350,000 sampled rows, seed 42, identical hyperparameters,
and the prescribed blending rule:
    E_statutory_raw = E_raw + 0.30 * (L_statutory - L_control)
    final_prediction = P(E_statutory_raw)

Features in Block A (7 statutory features, zero decree dependency):
- holiday_days
- holiday_days_prev_week
- holiday_days_next_week
- new_year_days
- may_holiday_days
- days_to_nearest_holiday
- days_since_last_holiday

Saves results to experiments/holidays_statutory/ (preserves experiments/holidays/ as historical record).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from holiday_calendar import build_holiday_feature_dataframe, STATUTORY_FEATURE_COLUMNS

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "holidays_statutory"
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


def train_lgbm_statutory_pair(
    matrices: WeeklyMatrices,
    origin: int,
    holiday_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
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

    # Model 2: Statutory Holidays (24 base features + 7 statutory holiday features)
    statutory_cols = [c for c in STATUTORY_FEATURE_COLUMNS if c in holiday_df.columns]
    X_train_stat = X_train_ctrl.copy()
    X_val_stat = X_val_ctrl.copy()
    for hc in statutory_cols:
        X_train_stat[hc] = train_holidays[hc].values.astype(float)
        X_val_stat[hc] = val_holidays[hc].values.astype(float)

    t1 = time.perf_counter()
    clf_stat = lgb.LGBMClassifier(**PARAMS)
    clf_stat.fit(X_train_stat, (target_arr > 0).astype(int))
    p_sale_stat = clf_stat.predict_proba(X_val_stat)[:, 1]

    reg_stat = lgb.LGBMRegressor(**PARAMS)
    reg_stat.fit(X_train_stat.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_stat = np.maximum(np.expm1(reg_stat.predict(X_val_stat)), 0.0)
    exp_stat = p_sale_stat * cond_stat
    l_stat = np.where(p_sale_stat < 0.5, 0.0, exp_stat)
    t_stat = time.perf_counter() - t1

    info = {
        "origin": origin,
        "train_rows": len(train_df),
        "num_origins": len(train_origins),
        "ctrl_time_sec": t_ctrl,
        "stat_time_sec": t_stat,
        "statutory_features_count": len(statutory_cols),
    }
    return l_ctrl, l_stat, info


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("LIGHTGBM B STATUTORY HOLIDAYS (BLOCK A, 7 FEATURES) ABLATION (5 FOLDS)")
    print("=" * 80)

    print("Loading weekly dataset...")
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("Building statutory holiday feature table (Art. 112 Labor Code, statutory_only=True)...")
    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)
    print(f"Generated {len(holiday_df)} weeks with {len(STATUTORY_FEATURE_COLUMNS)} statutory features.")

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
        l_saved_ctrl = merged["after_comp_lgbm"].values.astype(float)

        # Train models
        l_ctrl, l_stat, info = train_lgbm_statutory_pair(matrices, origin, holiday_df)

        ctrl_diff = np.max(np.abs(l_ctrl - l_saved_ctrl))
        print(f"  Max absolute diff with saved LightGBM B: {ctrl_diff:.2e}")
        assert ctrl_diff < 1e-4, f"Control mismatch on fold {fold}: {ctrl_diff}"

        # Evaluate Standalone Models (postprocessed)
        df_pred_ctrl = truth[KEYS].copy()
        df_pred_ctrl["quantity"] = apply_postprocess(l_ctrl)
        m_lgbm_ctrl = compute_metrics(truth, df_pred_ctrl, b_df)

        df_pred_stat = truth[KEYS].copy()
        df_pred_stat["quantity"] = apply_postprocess(l_stat)
        m_lgbm_stat = compute_metrics(truth, df_pred_stat, b_df)

        # Evaluate Blends
        # Control Blend B: P(e_raw)
        df_blend_ctrl = truth[KEYS].copy()
        df_blend_ctrl["quantity"] = apply_postprocess(e_raw)
        m_blend_ctrl = compute_metrics(truth, df_blend_ctrl, b_df)

        # Statutory Blend B: E_stat_raw = E_raw + 0.30 * (L_stat - L_ctrl)
        delta_stat = l_stat - l_ctrl
        e_stat_raw = e_raw + 0.30 * delta_stat
        df_blend_stat = truth[KEYS].copy()
        df_blend_stat["quantity"] = apply_postprocess(e_stat_raw)
        m_blend_stat = compute_metrics(truth, df_blend_stat, b_df)

        delta_unclipped = m_blend_stat["unclipped_ratio"] - m_blend_ctrl["unclipped_ratio"]
        delta_clipped = m_blend_stat["clipped_score"] - m_blend_ctrl["clipped_score"]

        print(f"  Standalone LGBM Ctrl: Score={m_lgbm_ctrl['clipped_score']:.6f}, Unclipped={m_lgbm_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone LGBM Stat: Score={m_lgbm_stat['clipped_score']:.6f}, Unclipped={m_lgbm_stat['unclipped_ratio']:.6f}")
        print(f"  Blend Ctrl B:         Score={m_blend_ctrl['clipped_score']:.6f}, Unclipped={m_blend_ctrl['unclipped_ratio']:.6f}")
        print(f"  Blend Statutory:      Score={m_blend_stat['clipped_score']:.6f}, Unclipped={m_blend_stat['unclipped_ratio']:.6f} (Delta unclipped={delta_unclipped:+.6f})")

        # Save keyed predictions
        df_keyed = truth[KEYS].copy()
        df_keyed["origin"] = origin
        df_keyed["e_raw_ctrl"] = e_raw
        df_keyed["l_ctrl"] = l_ctrl
        df_keyed["l_stat"] = l_stat
        df_keyed["e_stat_raw"] = e_stat_raw
        df_keyed["final_pred_ctrl"] = df_blend_ctrl["quantity"]
        df_keyed["final_pred_stat"] = df_blend_stat["quantity"]
        df_keyed.to_csv(KEYED_DIR / f"statutory_blend_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_metrics.append({
            "fold": fold,
            "role": role,
            "origin": origin,
            "ctrl_time_sec": info["ctrl_time_sec"],
            "stat_time_sec": info["stat_time_sec"],
            "lgbm_ctrl_unclipped": m_lgbm_ctrl["unclipped_ratio"],
            "lgbm_ctrl_score": m_lgbm_ctrl["clipped_score"],
            "lgbm_stat_unclipped": m_lgbm_stat["unclipped_ratio"],
            "lgbm_stat_score": m_lgbm_stat["clipped_score"],
            "blend_ctrl_unclipped": m_blend_ctrl["unclipped_ratio"],
            "blend_ctrl_score": m_blend_ctrl["clipped_score"],
            "blend_stat_unclipped": m_blend_stat["unclipped_ratio"],
            "blend_stat_score": m_blend_stat["clipped_score"],
            "delta_unclipped": delta_unclipped,
            "delta_score": delta_clipped,
            "stat_smape": m_blend_stat["smape"],
            "stat_wape": m_blend_stat["wape"],
            "stat_rmse": m_blend_stat["rmse"],
            "stat_bias": m_blend_stat["bias"],
            "stat_rmsle": m_blend_stat["rmsle"],
        })

    df_res = pd.DataFrame(fold_metrics)
    csv_path = OUTPUT_DIR / "statutory_holiday_metrics_by_fold.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved fold metrics to {csv_path}")

    # Summary
    sel = df_res[df_res["role"] == "Selection"]
    chk = df_res[df_res["role"] == "Check"]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_folds_4_3_2": {
            "blend_ctrl_mean_unclipped": float(sel["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(sel["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(sel["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(sel["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(sel["blend_stat_score"].mean()),
            "delta_score": float(sel["delta_score"].mean()),
        },
        "check_folds_1_0": {
            "blend_ctrl_mean_unclipped": float(chk["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(chk["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(chk["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(chk["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(chk["blend_stat_score"].mean()),
            "delta_score": float(chk["delta_score"].mean()),
        },
        "all_5_folds": {
            "blend_ctrl_mean_unclipped": float(df_res["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(df_res["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(df_res["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(df_res["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(df_res["blend_stat_score"].mean()),
            "delta_score": float(df_res["delta_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("STATUTORY HOLIDAY ABLATION SUMMARY:")
    print(f"Selection Folds (4, 3, 2) Unclipped: Ctrl={summary['selection_folds_4_3_2']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['selection_folds_4_3_2']['blend_stat_mean_unclipped']:.6f}, Delta={summary['selection_folds_4_3_2']['delta_unclipped']:+.6f}")
    print(f"Check Folds (1, 0)     Unclipped: Ctrl={summary['check_folds_1_0']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['check_folds_1_0']['blend_stat_mean_unclipped']:.6f}, Delta={summary['check_folds_1_0']['delta_unclipped']:+.6f}")
    print(f"All 5 Folds              Clipped: Ctrl={summary['all_5_folds']['blend_ctrl_mean_score']:.6f}, Stat={summary['all_5_folds']['blend_stat_mean_score']:.6f}, Delta={summary['all_5_folds']['delta_score']:+.6f}")
    print(f"All 5 Folds            Unclipped: Ctrl={summary['all_5_folds']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['all_5_folds']['blend_stat_mean_unclipped']:.6f}, Delta={summary['all_5_folds']['delta_unclipped']:+.6f}")
    print("=" * 80)

    with open(OUTPUT_DIR / "statutory_holiday_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
