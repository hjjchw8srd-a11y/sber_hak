#!/usr/bin/env python3
"""
train_save_stock_tft_folds.py

Trains NeuralForecast TFT with STOCK_FULL_EXOG (16 historical features)
on Folds [2, 1, 0] and saves raw predictions to:
experiments/tft/tft_stock_predictions_fold_{fold}.csv.gz
Matching the exact training protocol and hyperparameters of train_tft_weekly.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_tft_weekly import (
    BASE_HIST_EXOG,
    FUTR_EXOG,
    HORIZON,
    STOCK_FULL_EXOG,
    make_model,
    prediction_to_submission,
    prepare_frames,
    truth_frame,
    baseline_mean13,
)
from run_tft_feature_experiments import unique_columns
from fast_score import compute_all_metrics_fast
from score import normalized_score


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    output_dir = PROJECT_ROOT / "experiments" / "tft"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for Stock TFT training...", flush=True)
    raw = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    temporal, static_df, static_columns = prepare_frames(raw)
    weeks = pd.Index(sorted(raw["week"].unique()))
    latest_origin = len(weeks) - HORIZON
    
    # Folds: 0: origin 125, 1: origin 116, 2: origin 107
    origins = [latest_origin - fold * HORIZON for fold in range(3)]
    device = "gpu" if torch.cuda.is_available() else "cpu"

    hist_columns = unique_columns(BASE_HIST_EXOG, STOCK_FULL_EXOG)
    print(f"Historic columns ({len(hist_columns)}): {hist_columns}", flush=True)

    # Use exact args matching train_tft_weekly.py
    class Args:
        input_size = 26
        hidden_size = 32
        max_steps = 300
        batch_size = 128
        windows_batch_size = 512
        learning_rate = 1e-3
        val_check_steps = 50
        patience = 3
        seed = 42
        precision = "16-mixed" if device == "gpu" else "32-true"

    args = Args()

    for fold in [0, 1, 2]:
        out_pred_path = output_dir / f"tft_stock_predictions_fold_{fold}.csv.gz"
        if out_pred_path.exists():
            print(f"Fold {fold} prediction already exists at {out_pred_path}, skipping training.", flush=True)
            continue

        origin = origins[fold]
        validation_weeks = weeks[origin : origin + HORIZON]
        cutoff = pd.Timestamp(validation_weeks[0])
        print(f"\n>>> Training Stock TFT on Fold {fold} (origin {origin}, cutoff {cutoff.date()})...", flush=True)

        train_df = temporal[temporal["ds"] < cutoff].copy()
        future_df = temporal[temporal["ds"].isin(validation_weeks)][
            ["unique_id", "ds", *FUTR_EXOG]
        ].copy()

        t0 = time.perf_counter()
        model = make_model(args, static_columns, device, hist_exog_list=hist_columns)
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=train_df, static_df=static_df, val_size=HORIZON)
        
        raw_pred = prediction_to_submission(nf.predict(futr_df=future_df), raw)
        elapsed = time.perf_counter() - t0
        print(f"Fold {fold} finished training and predicting in {elapsed:.1f}s", flush=True)

        raw_pred.to_csv(out_pred_path, index=False, compression="gzip")
        print(f"Saved raw Stock TFT predictions to {out_pred_path}", flush=True)

        # Baseline check
        truth = truth_frame(raw, validation_weeks)
        baseline = baseline_mean13(raw, cutoff, validation_weeks)
        m_truth = compute_all_metrics_fast(truth, baseline)
        m_stock = compute_all_metrics_fast(truth, raw_pred)
        sc = normalized_score(m_stock, m_truth)
        print(f"Fold {fold} Raw Stock TFT Normalized Score: {sc:.6f}", flush=True)

    print("\nAll Stock TFT fold predictions ready!", flush=True)


if __name__ == "__main__":
    main()
