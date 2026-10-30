#!/usr/bin/env python3
"""
evaluate_task3_check_and_all_folds.py

Strict Check Phase (Folds 1, 0) and Full 5-Fold Evaluation (Folds 4, 3, 2, 1, 0)
for:
1. Control B (Baseline Gated 50/30/10/10)
2. Candidate Weights (45/37/9/9 Gated with base LGBM)
3. New Stock LGBM Ensemble (45 TFT + 37 Stock LGBM + 9 CB + 9 PerCat)
4. Combined: New Stock LGBM Ensemble + Activity-Filtered Series Reconciler
"""

from __future__ import annotations

import json
import time
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
from train_lgbm_task3_stock_yearly import train_eval_fold, apply_p

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "task3_check_evaluation"
KEYED_DIR = OUTPUT_DIR / "keyed_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}


def reconcile_series(
    df: pd.DataFrame,
    pred_col: str,
    raw_col: str,
    tau: float = 2.0,
    min_nz: float = 0.20,
    min_m: float = 0.10,
) -> np.ndarray:
    df = df.copy()
    grp = df.groupby("series_id")
    pred_sum = grp[pred_col].transform("sum")
    raw_sum = grp[raw_col].transform("sum")
    raw_max = grp[raw_col].transform("max")
    is_max = (df[raw_col] == raw_max) & (raw_max > 0)
    df["_is_max"] = is_max.astype(int)
    rank = df.groupby("series_id")["_is_max"].cumsum()
    is_single_max = is_max & (rank == 1)

    mask = (pred_sum == 0) & (raw_sum >= tau) & (df["hist_nonzero"] >= min_nz) & (df["hist_mean"] >= min_m) & is_single_max
    reconciled = df[pred_col].to_numpy().copy()
    reconciled[mask] = 1.0
    return reconciled


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("STRICT 5-FOLD EVALUATION OF TASK 3 & SERIES RECONCILER")
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
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
        bm = compute_all_metrics_fast(truth, b_df)

        # 1. Read existing OOF components
        df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
        df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")

        # 2. Train new Stock LGBM on this fold
        p_sale, exp_val, after_gate, info = train_eval_fold(matrices, origin, data, "stock")
        print(f"  Fitted Stock LGBM ({len(info['features'])} features) in {info['fit_time']:.1f}s")

        val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)
        val_df["week"] = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]
        val_df["article"] = val_df["article"].astype(df_f["article"].dtype)
        val_df["new_lgbm_gate"] = after_gate

        merged = df_f.merge(val_df[["week", "warehouse_name", "article", "new_lgbm_gate"]], on=["week", "warehouse_name", "article"], how="left")

        # Historical activity for series reconciliation
        hist_mean = matrices.quantity[origin-13:origin].mean(axis=0)
        hist_nonzero = (matrices.quantity[origin-13:origin] > 0).mean(axis=0)
        pair_df = matrices.pairs.copy()
        pair_df["hist_mean"] = hist_mean
        pair_df["hist_nonzero"] = hist_nonzero
        merged = merged.merge(pair_df, on=["warehouse_name", "article"], how="left")
        merged["series_id"] = merged["warehouse_name"] + "__" + merged["article"].astype(str)

        # Definitions of models:
        # A: Control B (final_prediction)
        pred_ctrl = merged["final_prediction"].to_numpy()

        # B: Candidate 45/37/9/9 (base LGBM)
        raw_cand = (
            0.45 * merged["raw_tft"]
            + 0.37 * merged["after_gate_lgbm"]
            + 0.09 * merged["after_gate_catboost"]
            + 0.09 * merged["after_comp_per_category"]
        )
        pred_cand = apply_p(raw_cand, 1.05, 0.65)

        # C: Stock LGBM Ensemble 45/37/9/9
        raw_stock = (
            0.45 * merged["raw_tft"]
            + 0.37 * merged["new_lgbm_gate"]
            + 0.09 * merged["after_gate_catboost"]
            + 0.09 * merged["after_comp_per_category"]
        )
        pred_stock = apply_p(raw_stock, 1.05, 0.65)

        # D: Combined: Stock LGBM + Activity-Filtered Series Reconciler
        merged["pred_stock"] = pred_stock
        merged["raw_stock"] = raw_stock
        pred_combined = reconcile_series(merged, "pred_stock", "raw_stock", tau=2.0, min_nz=0.20, min_m=0.10)

        # Save keyed predictions for this fold
        save_df = merged[KEYS].copy()
        save_df["control_b"] = pred_ctrl
        save_df["candidate_weights"] = pred_cand
        save_df["stock_lgbm_ensemble"] = pred_stock
        save_df["combined_stock_reconciled"] = pred_combined
        save_df.to_csv(KEYED_DIR / f"predictions_fold_{fold}.csv.gz", index=False, compression="gzip")

        models = {
            "Control B": pred_ctrl,
            "Candidate 45/37/9/9": pred_cand,
            "Stock LGBM Ens": pred_stock,
            "Combined Stock+Reconcile": pred_combined,
        }

        for model_name, preds in models.items():
            sub = truth[KEYS].copy()
            sub["warehouse_name"] = merged["warehouse_name"]
            sub["article"] = merged["article"]
            sub["quantity"] = preds

            m = compute_all_metrics_fast(truth, sub)
            unc = sum(w * (m[k] / bm[k]) for k, w in METRIC_WEIGHTS.items())
            score = sum(w * min(max(m[k] / bm[k], 0), 1) for k, w in METRIC_WEIGHTS.items())

            records.append({
                "fold": fold,
                "origin": origin,
                "phase": phase,
                "model": model_name,
                "unclipped": unc,
                "clipped": score,
                "smape": m["smape"],
                "bias": m["bias"],
                "wape": m["wape"],
                "rmse": m["rmse"],
                "rmsle": m["rmsle"],
            })
            print(f"    {model_name:24s} | Unclipped={unc:.6f} | Clipped={score:.6f} | SMAPE={m['smape']:.2f}% | Bias={m['bias']:.2f}% | WAPE={m['wape']:.2f}% | RMSE={m['rmse']:.4f}")

    df_all = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "five_folds_detailed_results.csv"
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS SUMMARY:")
    print("=" * 100)

    for ph, folds_filter in [("SELECTION (4, 3, 2)", [4, 3, 2]), ("CHECK (1, 0)", [1, 0]), ("ALL FIVE FOLDS (4, 3, 2, 1, 0)", [4, 3, 2, 1, 0])]:
        print(f"\n--- {ph} ---")
        sub_df = df_all[df_all["fold"].isin(folds_filter)]
        piv = sub_df.groupby("model").agg({
            "unclipped": "mean",
            "clipped": "mean",
            "smape": "mean",
            "bias": "mean",
            "wape": "mean",
            "rmse": "mean",
            "rmsle": "mean",
        }).loc[["Control B", "Candidate 45/37/9/9", "Stock LGBM Ens", "Combined Stock+Reconcile"]]
        print(piv.to_string())

    print("\n" + "=" * 100)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
