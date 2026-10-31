#!/usr/bin/env python3
"""
train_nhits_benchmark.py

Executes NHITS experimentation protocol according to task instructions:
- h = 9
- input_size = 52 (primary), 26 (ablation)
- Loss: MAE() (primary), MSE() (ablation)
- Scaler: robust
- Global model across all 8,789 warehouse-article pairs
- Known future calendar features: ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']
- Internal validation with early stopping (val_size=9)
- Keyed predictions saved before rounding: fold, week, warehouse_name, article, y_true, y_pred
- Evaluated on:
  * Recent period: Fold 0 (Origin 125, May-July 2026)
  * Autumn-winter period: Fold 3 (Origin 98, Nov 2025 - Jan 2026)
- Evaluated postprocessing:
  * continuous: clip(prediction, 0, None)
  * rint: np.rint(clip(prediction, 0, None))
- Direct comparison with frozen Ensemble Control B and Scale_1M_Dual_0.5
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "nhits_benchmark"
KEYED_DIR = OUTPUT_DIR / "keyed_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = {
    0: 125,  # 2026-05-25 — 2026-07-20 (Recent Summer)
    3: 98,   # 2025-11-17 — 2026-01-12 (Autumn-Winter)
}


def calc_normalized_scores(m_pred: dict, m_base: dict) -> tuple[float, float]:
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        ratio = m_pred[k] / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped


def run_nhits_fold(
    df: pd.DataFrame,
    weeks: list[str],
    fold: int,
    origin: int,
    input_size: int,
    loss_name: str,
    max_steps: int = 500,
    seed: int = 42,
    history_weeks: int = 78,
) -> dict:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE, MSE

    loss_fn = MAE() if loss_name == "MAE" else MSE()
    futr_exog = ["month", "quarter", "week_of_year", "week_sin", "week_cos"]

    val_weeks = weeks[origin : origin + HORIZON]
    val_start = weeks[origin]
    val_end = weeks[origin + HORIZON - 1]
    train_start = weeks[max(0, origin - history_weeks)]

    train_mask = (df["ds"] >= train_start) & (df["ds"] < val_start)
    val_mask = (df["ds"] >= val_start) & (df["ds"] <= val_end)

    train_data = df[train_mask][["unique_id", "ds", "y"] + futr_exog].copy()
    val_data = df[val_mask][["unique_id", "ds"] + futr_exog].copy()

    # Model definition
    model = NHITS(
        h=HORIZON,
        input_size=input_size,
        loss=loss_fn,
        futr_exog_list=futr_exog,
        max_steps=max_steps,
        scaler_type="robust",
        val_check_steps=25,
        early_stop_patience_steps=5,
        random_seed=seed,
        batch_size=256,
        windows_batch_size=512,
    )
    nf = NeuralForecast(models=[model], freq="W-MON")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nf.fit(df=train_data, val_size=HORIZON)
    torch.cuda.synchronize()
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds_df = nf.predict(futr_df=val_data)
    torch.cuda.synchronize()
    pred_time = time.perf_counter() - t1
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    preds_df["ds_str"] = preds_df["ds"].dt.strftime("%Y-%m-%d")
    parts = preds_df["unique_id"].str.split("__", n=1, expand=True)
    preds_df["warehouse_name"] = parts[0]
    preds_df["article"] = parts[1]

    return {
        "preds_df": preds_df,
        "fit_time": fit_time,
        "pred_time": pred_time,
        "peak_vram_mb": peak_vram_mb,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("EXECUTING STANDALONE NHITS BENCHMARK (FOLDS 0 & 3)")
    print("=" * 80)

    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    # Format base DataFrame
    df = data[["week", "warehouse_name", "article", "quantity"]].copy()
    df["ds"] = pd.to_datetime(df["week"])
    df["unique_id"] = df["warehouse_name"] + "__" + df["article"].astype(str)
    df = df.rename(columns={"quantity": "y"})

    # Calendar features
    w = df["ds"].dt.isocalendar().week.astype(float)
    df["month"] = df["ds"].dt.month.astype(float)
    df["quarter"] = df["ds"].dt.quarter.astype(float)
    df["week_of_year"] = w
    df["week_sin"] = np.sin(2.0 * np.pi * w / 52.1775)
    df["week_cos"] = np.cos(2.0 * np.pi * w / 52.1775)

    experiments = [
        {"name": "NHITS_Input52_MAE_Robust", "input_size": 52, "loss": "MAE"},
        {"name": "NHITS_Input26_MAE_Robust", "input_size": 26, "loss": "MAE"},
        {"name": "NHITS_Input52_MSE_Robust", "input_size": 52, "loss": "MSE"},
    ]

    all_records = []

    for exp in experiments:
        exp_name = exp["name"]
        inp_size = exp["input_size"]
        loss_n = exp["loss"]

        print(f"\n=======================================================")
        print(f"EXPERIMENT: {exp_name} (input_size={inp_size}, loss={loss_n})")
        print(f"=======================================================")

        for fold, origin in FOLDS.items():
            print(f"\n--- Running Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")
            val_weeks = weeks[origin : origin + HORIZON]
            truth = make_truth(matrices, origin)
            truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")
            truth["article"] = truth["article"].astype(str)

            b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
            b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
            b_df["article"] = b_df["article"].astype(str)
            m_base = compute_all_metrics_fast(truth, b_df)

            # Load Control B and tests champion predictions for benchmark comparison
            blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
            df_blend = pd.read_csv(blend_path)
            df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
            df_blend["article"] = df_blend["article"].astype(str)

            merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
            assert len(merged_ctrl) == 79101
            p_ctrl = merged_ctrl["final_prediction"].to_numpy()
            sub_ctrl = truth[KEYS].copy()
            sub_ctrl["quantity"] = p_ctrl
            m_ctrl = compute_all_metrics_fast(truth, sub_ctrl)
            c_ctrl, u_ctrl = calc_normalized_scores(m_ctrl, m_base)

            # Fit and Predict NHITS
            run_res = run_nhits_fold(
                df=df,
                weeks=weeks,
                fold=fold,
                origin=origin,
                input_size=inp_size,
                loss_name=loss_n,
                max_steps=500,
                seed=42,
                history_weeks=78,
            )

            preds_df = run_res["preds_df"]
            fit_t = run_res["fit_time"]
            pred_t = run_res["pred_time"]
            vram_mb = run_res["peak_vram_mb"]

            merged_p = truth[KEYS + ["quantity"]].rename(columns={"quantity": "y_true"}).merge(
                preds_df[["ds_str", "warehouse_name", "article", "NHITS"]],
                left_on=["week", "warehouse_name", "article"],
                right_on=["ds_str", "warehouse_name", "article"],
                how="inner",
                validate="one_to_one"
            )
            assert len(merged_p) == 79101, f"Mismatch in merged rows: {len(merged_p)}"
            merged_p = merged_p.rename(columns={"NHITS": "y_pred_raw"})

            # Save keyed predictions
            keyed_save_path = KEYED_DIR / f"{exp_name}_fold_{fold}_keyed.csv.gz"
            merged_p[["week", "warehouse_name", "article", "y_true", "y_pred_raw"]].to_csv(keyed_save_path, index=False)
            print(f"  Keyed predictions saved to {keyed_save_path}")

            raw_preds = merged_p["y_pred_raw"].to_numpy(dtype=float)
            neg_count = int((raw_preds < 0.0).sum())
            min_raw = float(raw_preds.min())
            max_raw = float(raw_preds.max())
            print(f"  Raw predictions: min={min_raw:.4f}, max={max_raw:.4f}, negative_count={neg_count}/{len(raw_preds)}")

            # Evaluate Postprocessing Variants
            # 1. Continuous (clip to 0 only)
            p_cont = np.maximum(raw_preds, 0.0)
            sub_cont = truth[KEYS].copy()
            sub_cont["quantity"] = p_cont
            m_cont = compute_all_metrics_fast(truth, sub_cont)
            c_cont, u_cont = calc_normalized_scores(m_cont, m_base)

            # 2. Integer Rounded (rint of non-negative)
            p_rint = np.rint(p_cont)
            sub_rint = truth[KEYS].copy()
            sub_rint["quantity"] = p_rint
            m_rint = compute_all_metrics_fast(truth, sub_rint)
            c_rint, u_rint = calc_normalized_scores(m_rint, m_base)

            print(f"  [Continuous] Clipped: {c_cont:.6f}, Unclipped: {u_cont:.6f}, SMAPE: {m_cont['smape']:.2f}%, WAPE: {m_cont['wape']:.2f}%, RMSE: {m_cont['rmse']:.4f}")
            print(f"  [Rint]       Clipped: {c_rint:.6f}, Unclipped: {u_rint:.6f}, SMAPE: {m_rint['smape']:.2f}%, WAPE: {m_rint['wape']:.2f}%, RMSE: {m_rint['rmse']:.4f}")
            print(f"  [Control B]  Clipped: {c_ctrl:.6f}, Unclipped: {u_ctrl:.6f}, SMAPE: {m_ctrl['smape']:.2f}%, WAPE: {m_ctrl['wape']:.2f}%, RMSE: {m_ctrl['rmse']:.4f}")

            # Record metrics
            for mode, m_val, c_val, u_val in [("continuous", m_cont, c_cont, u_cont), ("rint", m_rint, c_rint, u_rint)]:
                all_records.append({
                    "experiment": exp_name,
                    "input_size": inp_size,
                    "loss": loss_n,
                    "fold": fold,
                    "origin": origin,
                    "mode": mode,
                    "clipped_score": c_val,
                    "unclipped_ratio": u_val,
                    "smape": m_val["smape"],
                    "wape": m_val["wape"],
                    "rmse": m_val["rmse"],
                    "bias": m_val["bias"],
                    "rmsle": m_val["rmsle"],
                    "ctrl_clipped_score": c_ctrl,
                    "ctrl_unclipped_ratio": u_ctrl,
                    "ctrl_smape": m_ctrl["smape"],
                    "ctrl_wape": m_ctrl["wape"],
                    "ctrl_rmse": m_ctrl["rmse"],
                    "delta_clipped_vs_ctrl": c_val - c_ctrl,
                    "fit_time_sec": fit_t,
                    "pred_time_sec": pred_t,
                    "peak_vram_mb": vram_mb,
                    "keyed_file": str(keyed_save_path.name),
                })

    df_results = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "nhits_comparison_summary.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    # Summary table by experiment and mode across the 2 screening folds
    summary = df_results.groupby(["experiment", "mode"]).agg(
        mean_clipped=("clipped_score", "mean"),
        mean_unclipped=("unclipped_ratio", "mean"),
        mean_smape=("smape", "mean"),
        mean_wape=("wape", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_fit_time=("fit_time_sec", "mean"),
    ).reset_index()

    print("\n" + "=" * 90)
    print("NHITS BENCHMARK SUMMARY (AVERAGE OVER FOLD 0 & FOLD 3)")
    print("=" * 90)
    print(summary.to_string(index=False))

    with open(OUTPUT_DIR / "nhits_experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
