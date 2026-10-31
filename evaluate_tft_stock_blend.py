#!/usr/bin/env python3
"""
evaluate_tft_stock_blend.py

Evaluates TFT-Stock in the ensemble on Folds [2, 1, 0]:
1. 100% Base TFT (0.45 Base TFT)
2. 50/50 Blend TFT (0.225 Base TFT + 0.225 Stock TFT)
3. 30/15 Blend TFT (0.30 Base TFT + 0.15 Stock TFT)
4. Regime-Switching TFT (Stock TFT if stock_nonpositive_rate_lag_1 > 0, else Base TFT)
5. Standalone Base TFT vs Standalone Stock TFT

Evaluates on Folds 2, 1, 0 with standard post-processing (alpha=1.05, thresh=0.65, rint).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    compute_all_metrics_fast,
)
from fast_score import METRIC_WEIGHTS

def calc_scores(m_pred, m_base):
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped

def apply_p(raw_arr, alpha=1.05, thresh=0.65):
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)

ALL_FOLDS = {2: 107, 1: 116, 0: 125}

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Loading data for TFT Stock evaluation...", flush=True)
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    configs = [
        "100% Base TFT",
        "50/50 Blend TFT",
        "30/15 Blend TFT",
        "Regime-Switching TFT",
        "Standalone Base TFT",
        "Standalone Stock TFT",
    ]

    records = {c: {f: {} for f in ALL_FOLDS} for c in configs}

    for fold, origin in ALL_FOLDS.items():
        print(f"\n>>> Evaluating Fold {fold} (Origin {origin})...", flush=True)

        # Baselines
        history13 = np.column_stack([matrices.quantity[origin - lag, :] for lag in range(1, 14)])
        mean13 = history13.mean(axis=1)

        val_wks_arr = np.repeat(matrices.weeks[origin : origin + HORIZON].to_numpy(), matrices.n_pairs)
        pairs_rep = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)

        b_mean = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": np.tile(mean13, HORIZON),
        })
        b_naive = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": np.tile(matrices.quantity[origin - 1, :], HORIZON),
        })
        truth = pd.DataFrame({
            "week": pd.to_datetime(val_wks_arr).strftime("%Y-%m-%d"),
            "warehouse_name": pairs_rep["warehouse_name"],
            "article": pairs_rep["article"],
            "quantity": matrices.quantity[origin : origin + HORIZON, :].ravel(),
        })

        m_mean = compute_all_metrics_fast(truth, b_mean)
        m_naive = compute_all_metrics_fast(truth, b_naive)

        # Load Base TFT predictions
        base_tft_df = pd.read_csv(PROJECT_ROOT / "experiments" / "tft" / f"tft_predictions_fold_{fold}.csv.gz")
        q_base_tft = base_tft_df["quantity"].to_numpy(dtype=np.float64)

        # Load Stock TFT predictions
        stock_tft_path = PROJECT_ROOT / "experiments" / "tft" / f"tft_stock_predictions_fold_{fold}.csv.gz"
        if not stock_tft_path.exists():
            print(f"Error: {stock_tft_path} not found! Run train_save_stock_tft_folds.py first.")
            return
        stock_tft_df = pd.read_csv(stock_tft_path)
        q_stock_tft = stock_tft_df["quantity"].to_numpy(dtype=np.float64)

        # Load boostings from keyed_oof
        df_oof = pd.read_csv(PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz")
        pred_lgb = df_oof["after_gate_lgbm"].to_numpy(dtype=np.float64)
        pred_cb = df_oof["after_gate_catboost"].to_numpy(dtype=np.float64)
        pred_pcat = df_oof["after_gate_per_category"].to_numpy(dtype=np.float64)

        # Stock deficit mask
        snr_hist = matrices.stock_nonpositive_rate[origin - 1, :]
        snr_rep = np.tile(snr_hist, HORIZON)
        deficit_mask = snr_rep > 0

        # Construct TFT variants
        tft_variants = {
            "100% Base TFT": q_base_tft,
            "50/50 Blend TFT": 0.5 * q_base_tft + 0.5 * q_stock_tft,
            "30/15 Blend TFT": (0.30 / 0.45) * q_base_tft + (0.15 / 0.45) * q_stock_tft,
            "Regime-Switching TFT": np.where(deficit_mask, q_stock_tft, q_base_tft),
        }

        # Evaluate ensemble with each TFT variant
        for v_name, tft_pred in tft_variants.items():
            raw_blend = 0.45 * tft_pred + 0.37 * pred_lgb + 0.09 * pred_cb + 0.09 * pred_pcat
            final_p = apply_p(raw_blend, 1.05, 0.65)

            p_df = pd.DataFrame({
                "week": truth["week"],
                "warehouse_name": truth["warehouse_name"],
                "article": truth["article"],
                "quantity": final_p,
            })

            m_pred = compute_all_metrics_fast(truth, p_df)
            c_sc, u_sc = calc_scores(m_pred, m_mean)
            _, n_sc = calc_scores(m_pred, m_naive)

            records[v_name][fold] = {
                "clipped": c_sc,
                "unclipped": u_sc,
                "naive": n_sc,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
            }

        # Standalone evaluations
        p_base_standalone = apply_p(q_base_tft, 1.0, 0.75)
        p_stock_standalone = apply_p(q_stock_tft, 1.0, 0.75)

        for s_name, s_p in [("Standalone Base TFT", p_base_standalone), ("Standalone Stock TFT", p_stock_standalone)]:
            p_df = pd.DataFrame({
                "week": truth["week"],
                "warehouse_name": truth["warehouse_name"],
                "article": truth["article"],
                "quantity": s_p,
            })
            m_pred = compute_all_metrics_fast(truth, p_df)
            c_sc, u_sc = calc_scores(m_pred, m_mean)
            _, n_sc = calc_scores(m_pred, m_naive)
            records[s_name][fold] = {
                "clipped": c_sc,
                "unclipped": u_sc,
                "naive": n_sc,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
            }

    # Summary table across Folds [2, 1, 0]
    print("\n" + "=" * 125, flush=True)
    print("FOLDS [2, 1, 0] TFT-STOCK EVALUATION SUMMARY", flush=True)
    print("=" * 125, flush=True)
    fmt = "{:<24} | {:<12} | {:<14} | {:<14} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("TFT Strategy", "Clipped", "Unclipped", "Score vs Naive", "SMAPE (%)", "WAPE (%)", "Bias (%)", "RMSE"), flush=True)
    print("-" * 125, flush=True)

    for cfg_name in configs:
        sc_c = np.mean([records[cfg_name][f]["clipped"] for f in ALL_FOLDS])
        sc_u = np.mean([records[cfg_name][f]["unclipped"] for f in ALL_FOLDS])
        sc_n = np.mean([records[cfg_name][f]["naive"] for f in ALL_FOLDS])
        sm = np.mean([records[cfg_name][f]["smape"] for f in ALL_FOLDS])
        wa = np.mean([records[cfg_name][f]["wape"] for f in ALL_FOLDS])
        bi = np.mean([records[cfg_name][f]["bias"] for f in ALL_FOLDS])
        rm = np.mean([records[cfg_name][f]["rmse"] for f in ALL_FOLDS])
        print(fmt.format(cfg_name, f"{sc_c:.6f}", f"{sc_u:.6f}", f"{sc_n:.6f}", f"{sm:.3f}%", f"{wa:.3f}%", f"{bi:.2f}%", f"{rm:.4f}"), flush=True)

if __name__ == "__main__":
    main()
