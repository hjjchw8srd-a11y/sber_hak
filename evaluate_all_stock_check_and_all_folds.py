#!/usr/bin/env python3
"""
evaluate_all_stock_check_and_all_folds.py

Strict 5-Fold Evaluation (Selection Folds [4, 3, 2] + Check Folds [1, 0])
of the All-Stock Boosting Ensemble (29 features) across LightGBM, CatBoost, and Per-Category.
Evaluated against BOTH baseline_mean13 and baseline_naive_last.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    train_predict_stock_lgbm,
    train_predict_stock_catboost,
    train_predict_stock_per_category,
    evaluate_predictions,
    apply_p,
)
from evaluate_task3_check_and_all_folds import reconcile_series

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "all_stock_5folds"
KEYED_DIR = OUTPUT_DIR / "keyed_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("STRICT 5-FOLD EVALUATION OF ALL-STOCK ENSEMBLE (29 FEATURES)")
    print("Selection Folds: [4, 3, 2] | Check Folds: [1, 0]")
    print("=" * 100)

    records = []

    for fold, origin in ALL_FOLDS.items():
        phase = "Selection" if fold in [4, 3, 2] else "Check"
        t0 = time.perf_counter()
        print(f"\nProcessing Fold {fold} (Origin {origin}, {phase} fold)...")

        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_mean13 = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_mean13["week"] = pd.to_datetime(b_mean13["week"]).dt.strftime("%Y-%m-%d")

        last_sales = matrices.quantity[origin - 1, :]
        pairs = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
        val_wks_arr = np.repeat(matrices.weeks[origin : origin + HORIZON].to_numpy(), matrices.n_pairs)
        b_naive = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs["warehouse_name"],
            "article": pairs["article"],
            "quantity": np.tile(last_sales, HORIZON),
        })

        df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
        df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

        latest_train = origin - HORIZON
        lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
        cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

        # 1. Build features (29 features)
        X_train_lgb, y_train_lgb, X_val, val_df, val_pos = build_feature_matrices(matrices, origin, lgbm_origins, mode="29")
        X_train_cb, y_train_cb, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, mode="29")

        val_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]
        val_df["article"] = val_df["article"].astype(df_f["article"].dtype)

        feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

        # Train models
        pred_lgb = train_predict_stock_lgbm(X_train_lgb, y_train_lgb, X_val, feature_cols)
        pred_cb = train_predict_stock_catboost(X_train_cb, y_train_cb, X_val, feature_cols)
        pred_pcat = train_predict_stock_per_category(X_train_lgb, y_train_lgb, X_val, feature_cols + ["category_lvl_1"])

        val_df["stock_lgbm_gate"] = pred_lgb
        val_df["stock_cb_gate"] = pred_cb
        val_df["stock_pcat_gate"] = pred_pcat

        merged = df_f.merge(val_df[["week", "warehouse_name", "article", "stock_lgbm_gate", "stock_cb_gate", "stock_pcat_gate"]], on=KEYS, how="left")

        # Models:
        # 1. Control B
        c_ctrl = merged["final_prediction"].to_numpy()

        # 2. Candidate 45/37/9/9 (base models)
        raw_cand = (
            0.45 * merged["raw_tft"]
            + 0.37 * merged["after_gate_lgbm"]
            + 0.09 * merged["after_gate_catboost"]
            + 0.09 * merged["after_comp_per_category"]
        )
        c_cand = apply_p(raw_cand, 1.05, 0.65)

        # 3. Stock LGBM Only (our previous champion)
        raw_lgbm_stock = (
            0.45 * merged["raw_tft"]
            + 0.37 * merged["stock_lgbm_gate"]
            + 0.09 * merged["after_gate_catboost"]
            + 0.09 * merged["after_comp_per_category"]
        )
        c_lgbm_stock = apply_p(raw_lgbm_stock, 1.05, 0.65)

        # 4. ALL-STOCK ENSEMBLE (LGBM + CatBoost + Per-Category)
        raw_all_stock = (
            0.45 * merged["raw_tft"]
            + 0.37 * merged["stock_lgbm_gate"]
            + 0.09 * merged["stock_cb_gate"]
            + 0.09 * merged["stock_pcat_gate"]
        )
        c_all_stock = apply_p(raw_all_stock, 1.05, 0.65)

        # 5. ALL-STOCK + SERIES RECONCILER (Activity-Conditioned)
        hist_mean = matrices.quantity[origin-13:origin].mean(axis=0)
        hist_nonzero = (matrices.quantity[origin-13:origin] > 0).mean(axis=0)
        pair_df = matrices.pairs.copy()
        pair_df["hist_mean"] = hist_mean
        pair_df["hist_nonzero"] = hist_nonzero
        merged_rec = merged.merge(pair_df, on=["warehouse_name", "article"], how="left")
        merged_rec["series_id"] = merged_rec["warehouse_name"] + "__" + merged_rec["article"].astype(str)
        merged_rec["pred_all_stock"] = c_all_stock
        merged_rec["raw_all_stock"] = raw_all_stock
        c_all_stock_rec = reconcile_series(merged_rec, "pred_all_stock", "raw_all_stock", tau=2.0, min_nz=0.20, min_m=0.10)

        # Save predictions for this fold
        save_df = merged[KEYS].copy()
        save_df["control_b"] = c_ctrl
        save_df["candidate_weights"] = c_cand
        save_df["stock_lgbm_only"] = c_lgbm_stock
        save_df["all_stock_ensemble"] = c_all_stock
        save_df["all_stock_reconciled"] = c_all_stock_rec
        save_df.to_csv(KEYED_DIR / f"predictions_fold_{fold}.csv.gz", index=False, compression="gzip")

        models = {
            "Control B": c_ctrl,
            "Candidate 45/37/9/9": c_cand,
            "Stock LGBM Only": c_lgbm_stock,
            "All-Stock Ensemble (29)": c_all_stock,
            "All-Stock + Reconciled": c_all_stock_rec,
        }

        for model_name, preds in models.items():
            sub = truth[KEYS].copy()
            sub["warehouse_name"] = merged["warehouse_name"]
            sub["article"] = merged["article"]
            sub["quantity"] = preds

            res = evaluate_predictions(truth, sub, b_mean13, b_naive)
            records.append({
                "fold": fold,
                "origin": origin,
                "phase": phase,
                "model": model_name,
                **res,
            })
            print(f"    {model_name:28s} | Clip13={res['clip_mean13']:.6f} | Unc13={res['unc_mean13']:.6f} | ClipNv={res['clip_naive']:.6f} | SMAPE={res['smape']:.2f}% | Bias={res['bias']:.2f}% | RMSE={res['rmse']:.4f}")

    df_all = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "all_stock_five_folds_results.csv"
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS SUMMARY:")
    print("=" * 100)

    for ph, folds_filter in [("SELECTION (4, 3, 2)", [4, 3, 2]), ("CHECK (1, 0)", [1, 0]), ("ALL FIVE FOLDS (4, 3, 2, 1, 0)", [4, 3, 2, 1, 0])]:
        print(f"\n--- {ph} ---")
        sub_df = df_all[df_all["fold"].isin(folds_filter)]
        piv = sub_df.groupby("model").agg({
            "clip_mean13": "mean",
            "unc_mean13": "mean",
            "clip_naive": "mean",
            "unc_naive": "mean",
            "smape": "mean",
            "bias": "mean",
            "wape": "mean",
            "rmse": "mean",
            "rmsle": "mean",
        }).loc[["Control B", "Candidate 45/37/9/9", "Stock LGBM Only", "All-Stock Ensemble (29)", "All-Stock + Reconciled"]]
        print(piv.to_string())

    print("\n" + "=" * 100)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
