"""Финальное обучение TFT на всей истории и прогноз следующих девяти недель."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

from analyze_tft_postprocessing import apply_rule
from score import validate_submission
from train_tft_weekly import FUTR_EXOG, HORIZON, make_model, prediction_to_submission, prepare_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tft"))
    parser.add_argument("--input-size", type=int, default=26)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--windows-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-check-steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.75)
    return parser.parse_args()


def make_future_frame(temporal: pd.DataFrame) -> pd.DataFrame:
    ids = temporal["unique_id"].drop_duplicates().sort_values()
    last_week = pd.Timestamp(temporal["ds"].max())
    future_weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-MON")
    frame = pd.MultiIndex.from_product([ids, future_weeks], names=["unique_id", "ds"]).to_frame(index=False)
    time_start = int(temporal["time_idx"].max()) + 1
    week_position = pd.Series(np.arange(HORIZON, dtype=np.float32), index=future_weeks)
    frame["time_idx"] = frame["ds"].map(week_position).astype(np.float32) + time_start
    iso_week = frame["ds"].dt.isocalendar().week.astype(np.float32)
    frame["month"] = frame["ds"].dt.month.astype(np.float32)
    frame["quarter"] = frame["ds"].dt.quarter.astype(np.float32)
    frame["week_of_year"] = iso_week
    frame["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0).astype(np.float32)
    frame["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0).astype(np.float32)
    return frame[["unique_id", "ds", *FUTR_EXOG]]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    raw = pd.read_csv(args.data, parse_dates=["week"])
    temporal, static_df, static_columns = prepare_frames(raw)
    future_df = make_future_frame(temporal)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    model = make_model(args, static_columns, device)
    nf = NeuralForecast(models=[model], freq="W-MON")

    started = time.perf_counter()
    nf.fit(df=temporal, static_df=static_df, val_size=0)
    raw_prediction = prediction_to_submission(nf.predict(futr_df=future_df), raw)
    elapsed = time.perf_counter() - started
    final_prediction = raw_prediction.copy()
    final_prediction["quantity"] = apply_rule(
        final_prediction["quantity"], args.threshold, round_to_int=True
    )
    warnings = validate_submission(final_prediction, "tft_submission")

    raw_prediction.to_csv(args.output_dir / "tft_final_prediction_raw.csv.gz", index=False)
    final_prediction.to_csv(args.output_dir / "tft_final_submission.csv", index=False)
    nf.save(path=str(args.output_dir / "final_model"), overwrite=True)
    info = {
        "train_seconds": elapsed,
        "train_rows": len(temporal),
        "series": int(temporal["unique_id"].nunique()),
        "forecast_start": str(final_prediction["week"].min().date()),
        "forecast_end": str(final_prediction["week"].max().date()),
        "forecast_rows": len(final_prediction),
        "threshold": args.threshold,
        "round_to_int": True,
        "warnings": warnings,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output_dir / "tft_final_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
