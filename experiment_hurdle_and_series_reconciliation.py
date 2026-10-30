#!/usr/bin/env python3
"""
experiment_hurdle_and_series_reconciliation.py

Comprehensive ablation of:
1. Double gating decoupling:
   - Gated blend (current baseline) vs Ungated blend (E_raw using continuous expectations)
2. Hybrid probability-conditional recovery:
   - when p_blend >= p_gate: beta * E_raw + (1 - beta) * Q_cond
3. Series-aware reconciliation:
   - Zero-series recovery conditioned on S_series >= tau and recent history
   - Series-level bias damping for large series
Evaluated strictly on Selection Folds: 4, 3, 2.
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
from run_weekly_feature_experiments import WeeklyMatrices, make_truth, KEYS, HORIZON
from train_tft_weekly import baseline_mean13

SELECTION_FOLDS = {4: 89, 3: 98, 2: 107}


def apply_p(x: np.ndarray, alpha: float = 1.05, threshold: float = 0.65) -> np.ndarray:
    scaled = alpha * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < threshold, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def series_reconcile_smart(
    df: pd.DataFrame,
    pred_col: str,
    raw_col: str,
    p_col: str | None = None,
    tau_series: float = 1.3,
    min_p_max: float = 0.30,
    damp_large_series: bool = False,
) -> np.ndarray:
    """
    Applies series-aware reconciliation:
    1. Zero-series recovery: if series has 0 predictions across all 9 weeks,
       BUT 9-week sum of raw forecasts >= tau_series
       AND max weekly probability of sale >= min_p_max:
       put 1.0 (or rint(S)) at the week of maximum raw demand.
    2. Large-series bias damping (optional).
    """
    df = df.copy()
    df["series_id"] = df["warehouse_name"] + "__" + df["article"].astype(str)

    grouped = df.groupby("series_id")
    raw_sums = grouped[raw_col].transform("sum")
    pred_sums = grouped[pred_col].transform("sum")
    raw_max = grouped[raw_col].transform("max")
    
    # Track first occurrence of max week to avoid multi-week duplicate assignment on ties
    is_max_week = (df[raw_col] == raw_max) & (raw_max > 0)
    
    # Cumulative count of max weeks per series to break ties
    df["_is_max"] = is_max_week.astype(int)
    max_rank = df.groupby("series_id")["_is_max"].cumsum()
    is_single_max_week = is_max_week & (max_rank == 1)

    reconciled = df[pred_col].to_numpy().copy()

    # Recovery mask
    if p_col is not None and p_col in df.columns:
        p_max = grouped[p_col].transform("max")
        recover_mask = (pred_sums == 0) & (raw_sums >= tau_series) & (p_max >= min_p_max) & is_single_max_week
    else:
        recover_mask = (pred_sums == 0) & (raw_sums >= tau_series) & is_single_max_week

    reconciled[recover_mask] = 1.0

    # Optional Series-Level Bias Damping for active series
    if damp_large_series:
        large_mask = (raw_sums >= 5.0) & (pred_sums > 0)
        # Ratio of raw sum to rounded prediction sum
        ratio = np.where(pred_sums > 0, raw_sums / pred_sums, 1.0)
        # Apply gentle correction if divergence > 15%
        divergent = large_mask & (np.abs(ratio - 1.0) > 0.15)
        # Multiply week predictions by ratio and round
        # Only damp slightly to avoid destabilizing RMSE
        damped_vals = np.maximum(np.rint(reconciled * (0.5 + 0.5 * ratio)), 0.0)
        reconciled = np.where(divergent, damped_vals, reconciled)

    return reconciled


def evaluate_variant(matrices, weeks, data, variant_fn, name: str) -> dict:
    scores = []
    uncs = []
    smapes = []
    wapes = []
    rmses = []
    biases = []
    rmsles = []

    for fold, origin in SELECTION_FOLDS.items():
        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        df_f = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
        df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")
        merged = truth[KEYS].merge(df_f, on=KEYS)

        sub = truth[KEYS].copy()
        sub["warehouse_name"] = merged["warehouse_name"]
        sub["article"] = merged["article"]
        sub["quantity"] = variant_fn(merged)
        sub = sub[KEYS + ["quantity"]]

        m = compute_all_metrics_fast(truth, sub)
        bm = compute_all_metrics_fast(truth, b_df)

        score = sum(w * min(max(m[k] / bm[k], 0), 1) for k, w in METRIC_WEIGHTS.items())
        unc = sum(w * (m[k] / bm[k]) for k, w in METRIC_WEIGHTS.items())

        scores.append(score)
        uncs.append(unc)
        smapes.append(m["smape"])
        wapes.append(m["wape"])
        rmses.append(m["rmse"])
        biases.append(m["bias"])
        rmsles.append(m["rmsle"])

    return {
        "name": name,
        "mean_score": float(np.mean(scores)),
        "mean_unclipped": float(np.mean(uncs)),
        "delta_unclipped_vs_ctrl": float(np.mean(uncs) - 0.761314),
        "delta_score_vs_ctrl": float(np.mean(scores) - 0.734539),
        "mean_smape": float(np.mean(smapes)),
        "mean_wape": float(np.mean(wapes)),
        "mean_rmse": float(np.mean(rmses)),
        "mean_bias": float(np.mean(biases)),
        "mean_rmsle": float(np.mean(rmsles)),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("=" * 100)
    print("TASK 1 & TASK 2 COMPREHENSIVE EXPERIMENT ON SELECTION FOLDS (4, 3, 2)")
    print("Baseline Control B Selection Unclipped = 0.761314, Clipped = 0.734539")
    print("=" * 100)

    results = []

    # 0. Baseline Control B
    res_ctrl = evaluate_variant(
        matrices, weeks, data,
        lambda df: df["final_prediction"],
        "0. Control B (Baseline Gated 50/30/10/10)"
    )
    results.append(res_ctrl)

    # 1. Candidate Weights 45/37/9/9 (Gated)
    def fn_cand_gated(df):
        raw = 0.45 * df["raw_tft"] + 0.37 * df["after_gate_lgbm"] + 0.09 * df["after_gate_catboost"] + 0.09 * df["after_comp_per_category"]
        return apply_p(raw, 1.05, 0.65)
    res_cand = evaluate_variant(matrices, weeks, data, fn_cand_gated, "1. Candidate 45/37/9/9 Gated")
    results.append(res_cand)

    # 2. Decoupled Ungated Blend (Continuous Expectation without p < 0.5 gate)
    def fn_ctrl_ungated(df):
        raw = 0.50 * df["raw_tft"] + 0.30 * df["expected_lgbm"] + 0.10 * df["expected_catboost"] + 0.10 * df["expected_per_category"]
        return apply_p(raw, 1.05, 0.65)
    res_ctrl_ungated = evaluate_variant(matrices, weeks, data, fn_ctrl_ungated, "2. Control Ungated Blend (50/30/10/10)")
    results.append(res_ctrl_ungated)

    def fn_cand_ungated(df):
        raw = 0.45 * df["raw_tft"] + 0.37 * df["expected_lgbm"] + 0.09 * df["expected_catboost"] + 0.09 * df["expected_per_category"]
        return apply_p(raw, 1.05, 0.65)
    res_cand_ungated = evaluate_variant(matrices, weeks, data, fn_cand_ungated, "3. Candidate Ungated Blend (45/37/9/9)")
    results.append(res_cand_ungated)

    # 3. Hybrid Probability-Conditional Formulation
    # p_blend = 0.60 * p_lgb + 0.20 * p_cb + 0.20 * p_pcat
    for p_gate in [0.40, 0.45, 0.50]:
        for beta in [0.70, 0.85]:
            def make_hybrid_fn(pg, b):
                def fn(df):
                    p_blend = 0.60 * df["p_sale_lgbm"] + 0.20 * df["p_sale_catboost"] + 0.20 * df["p_sale_per_category"]
                    e_raw = 0.45 * df["raw_tft"] + 0.37 * df["expected_lgbm"] + 0.09 * df["expected_catboost"] + 0.09 * df["expected_per_category"]
                    q_cond = e_raw / (p_blend + 1e-4)
                    hybrid = np.where(p_blend >= pg, b * e_raw + (1.0 - b) * q_cond, e_raw)
                    return apply_p(hybrid, 1.05, 0.65)
                return fn
            name = f"4. Hybrid p_gate={p_gate:.2f}, beta={beta:.2f}"
            res_hyb = evaluate_variant(matrices, weeks, data, make_hybrid_fn(p_gate, beta), name)
            results.append(res_hyb)

    # 4. Series-Aware Reconciliation on top of Candidate Ungated
    for tau in [1.0, 1.2, 1.3, 1.4, 1.5, 1.8]:
        for min_p in [0.0, 0.25, 0.35]:
            def make_reconcile_fn(t, mp):
                def fn(df):
                    df_work = df.copy()
                    df_work["raw_ungated"] = 0.45 * df["raw_tft"] + 0.37 * df["expected_lgbm"] + 0.09 * df["expected_catboost"] + 0.09 * df["expected_per_category"]
                    df_work["p_blend"] = 0.60 * df["p_sale_lgbm"] + 0.20 * df["p_sale_catboost"] + 0.20 * df["p_sale_per_category"]
                    df_work["pred_base"] = apply_p(df_work["raw_ungated"], 1.05, 0.65)
                    return series_reconcile_smart(
                        df_work, "pred_base", "raw_ungated", p_col="p_blend",
                        tau_series=t, min_p_max=mp, damp_large_series=False
                    )
                return fn
            name = f"5. Series_Reconcile tau={tau:.1f}, min_p={min_p:.2f}"
            res_rec = evaluate_variant(matrices, weeks, data, make_reconcile_fn(tau, min_p), name)
            results.append(res_rec)

    # 5. Combined: Series Reconcile + Series Bias Damping
    for tau in [1.3, 1.5]:
        def make_reconcile_damp_fn(t):
            def fn(df):
                df_work = df.copy()
                df_work["raw_ungated"] = 0.45 * df["raw_tft"] + 0.37 * df["expected_lgbm"] + 0.09 * df["expected_catboost"] + 0.09 * df["expected_per_category"]
                df_work["p_blend"] = 0.60 * df["p_sale_lgbm"] + 0.20 * df["p_sale_catboost"] + 0.20 * df["p_sale_per_category"]
                df_work["pred_base"] = apply_p(df_work["raw_ungated"], 1.05, 0.65)
                return series_reconcile_smart(
                    df_work, "pred_base", "raw_ungated", p_col="p_blend",
                    tau_series=t, min_p_max=0.30, damp_large_series=True
                )
            return fn
        name = f"6. Reconcile+Damp tau={tau:.1f}"
        res_damp = evaluate_variant(matrices, weeks, data, make_reconcile_damp_fn(tau), name)
        results.append(res_damp)

    df_res = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "experiments" / "model_improvements" / "hurdle_and_series_reconciliation_selection_results.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("SELECTION RESULTS SUMMARY (SORTED BY UNCLIPPED RATIO):")
    print("=" * 100)
    df_res_sorted = df_res.sort_values("mean_unclipped")
    for _, r in df_res_sorted.iterrows():
        print(f"{r['name']:48s} | Unclipped={r['mean_unclipped']:.6f} (Delta={r['delta_unclipped_vs_ctrl']:+.6f}) | Score={r['mean_score']:.6f} (Delta={r['delta_score_vs_ctrl']:+.6f}) | SMAPE={r['mean_smape']:.2f}% | Bias={r['mean_bias']:.2f}% | WAPE={r['mean_wape']:.2f}% | RMSE={r['mean_rmse']:.4f}")
    print("=" * 100)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
