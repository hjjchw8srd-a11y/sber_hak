"""Сравнение групп исторических признаков для недельного TFT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

from fast_score import compute_all_metrics_fast
from score import normalized_score
from train_tft_weekly import (
    BASE_HIST_EXOG,
    DEMAND_DYNAMICS_EXOG,
    FUTR_EXOG,
    HORIZON,
    METRICS,
    MONEY_EXOG,
    STOCK_BASIC_EXOG,
    STOCK_DERIVED_EXOG,
    STOCK_FULL_EXOG,
    baseline_mean13,
    make_model,
    prediction_to_submission,
    prepare_frames,
    truth_frame,
)


def unique_columns(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(column for group in groups for column in group))


EXPERIMENTS = {
    "base": BASE_HIST_EXOG,
    "stock_basic": unique_columns(BASE_HIST_EXOG, STOCK_BASIC_EXOG),
    "stock_full": unique_columns(BASE_HIST_EXOG, STOCK_FULL_EXOG),
    "stock_derived": unique_columns(BASE_HIST_EXOG, STOCK_DERIVED_EXOG),
    "money": unique_columns(BASE_HIST_EXOG, MONEY_EXOG),
    "money_raw": unique_columns(BASE_HIST_EXOG, MONEY_EXOG[:2]),
    "money_derived": unique_columns(BASE_HIST_EXOG, MONEY_EXOG[2:]),
    "dynamics": unique_columns(BASE_HIST_EXOG, DEMAND_DYNAMICS_EXOG),
    "dynamics_level": unique_columns(BASE_HIST_EXOG, DEMAND_DYNAMICS_EXOG[:5]),
    "dynamics_growth": unique_columns(BASE_HIST_EXOG, DEMAND_DYNAMICS_EXOG[5:7]),
    "dynamics_sparse": unique_columns(BASE_HIST_EXOG, DEMAND_DYNAMICS_EXOG[7:9]),
    "dynamics_autocorr": unique_columns(BASE_HIST_EXOG, DEMAND_DYNAMICS_EXOG[9:]),
    "stock_basic_money": unique_columns(BASE_HIST_EXOG, STOCK_BASIC_EXOG, MONEY_EXOG),
    "stock_basic_dynamics": unique_columns(
        BASE_HIST_EXOG, STOCK_BASIC_EXOG, DEMAND_DYNAMICS_EXOG
    ),
    "stock_derived_money": unique_columns(
        BASE_HIST_EXOG, STOCK_DERIVED_EXOG, MONEY_EXOG
    ),
    "stock_full_money_raw": unique_columns(
        BASE_HIST_EXOG, STOCK_FULL_EXOG, MONEY_EXOG[:2]
    ),
    "stock_full_dynamics_level": unique_columns(
        BASE_HIST_EXOG, STOCK_FULL_EXOG, DEMAND_DYNAMICS_EXOG[:5]
    ),
    "all_groups": unique_columns(
        BASE_HIST_EXOG,
        STOCK_FULL_EXOG,
        STOCK_DERIVED_EXOG,
        MONEY_EXOG,
        DEMAND_DYNAMICS_EXOG,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tft_features_screen"))
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--fold-step", type=int, default=HORIZON)
    parser.add_argument("--input-size", type=int, default=26)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--windows-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-check-steps", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--precision", choices=["16-mixed", "32-true"], default=None)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def processed_prediction(prediction: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = prediction.copy()
    values = result["quantity"].to_numpy(dtype=float)
    values[values < threshold] = 0
    result["quantity"] = np.rint(np.clip(values, 0, None))
    return result


def refresh_report(output_dir: Path, results: pd.DataFrame, max_steps: int) -> None:
    summary = (
        results.groupby("experiment", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            hist_features=("hist_features", "first"),
            mean_score=("normalized_score", "mean"),
            std_score=("normalized_score", "std"),
            mean_raw_score=("raw_normalized_score", "mean"),
            mean_smape=("smape", "mean"),
            mean_wape=("wape", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            mean_rmsle=("rmsle", "mean"),
            mean_seconds=("run_seconds", "mean"),
        )
        .sort_values("mean_score")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "tft_feature_summary.csv", index=False)
    shown = summary.copy()
    for column in shown.columns:
        if column not in {"experiment", "folds", "hist_features"}:
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = shown.columns.tolist()
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    table = "\n".join(table_lines)
    report = f"""# Перебор признаков TFT

Все варианты используют одну архитектуру, одинаковые окна и постобработку:
значения меньше 0.75 обнуляются, затем прогноз округляется. На этапе screening
каждая модель обучается {max_steps} шагов. Меньше — лучше.

{table}

`raw_score` показывает выход без порога и округления. Финальные выводы делаются
только после перепроверки лучших вариантов на трёх окнах с 300 шагами.
"""
    (output_dir / "TFT_FEATURE_EXPERIMENTS.md").write_text(report, encoding="utf-8")

    plot = summary.dropna(subset=["mean_score"]).sort_values("mean_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot) * 0.55)))
    positions = np.arange(len(plot))
    ax.barh(positions, plot["mean_score"], color="#4C78A8")
    ax.set_yticks(positions, plot["experiment"])
    ax.axvline(0.7433, color="#2F855A", linestyle="--", label="CatBoost 0.7433")
    ax.axvline(1.0, color="#C53030", linestyle=":", label="mean13 = 1.0")
    for position, value in zip(positions, plot["mean_score"]):
        ax.text(value + 0.004, position, f"{value:.4f}", va="center")
    ax.set_xlabel("Normalized score (меньше — лучше)")
    ax.set_title("TFT: сравнение групп признаков")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tft_feature_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "tft_feature_results.csv"
    results = pd.read_csv(result_path) if result_path.exists() and not args.restart else pd.DataFrame()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    raw = pd.read_csv(args.data, parse_dates=["week"])
    temporal, static_df, static_columns = prepare_frames(raw)
    weeks = pd.Index(sorted(raw["week"].unique()))
    latest_origin = len(weeks) - HORIZON
    origins = [latest_origin - fold * args.fold_step for fold in range(args.folds)]
    device = "gpu" if torch.cuda.is_available() else "cpu"

    for fold, origin in enumerate(origins):
        validation_weeks = weeks[origin : origin + HORIZON]
        cutoff = pd.Timestamp(validation_weeks[0])
        train_df = temporal[temporal["ds"] < cutoff]
        future_df = temporal[temporal["ds"].isin(validation_weeks)][
            ["unique_id", "ds", *FUTR_EXOG]
        ]
        truth = truth_frame(raw, validation_weeks)
        baseline = baseline_mean13(raw, cutoff, validation_weeks)
        baseline_metrics = compute_all_metrics_fast(truth, baseline)

        for experiment in args.experiments:
            if not results.empty and (
                (results["fold"] == fold) & (results["experiment"] == experiment)
            ).any():
                print(f"skip fold={fold} experiment={experiment}")
                continue
            columns = EXPERIMENTS[experiment]
            started = time.perf_counter()
            model = make_model(args, static_columns, device, hist_exog_list=columns)
            nf = NeuralForecast(models=[model], freq="W-MON")
            try:
                nf.fit(df=train_df, static_df=static_df, val_size=HORIZON)
                raw_prediction = prediction_to_submission(nf.predict(futr_df=future_df), raw)
                elapsed = time.perf_counter() - started
                raw_metrics = compute_all_metrics_fast(truth, raw_prediction)
                metrics = compute_all_metrics_fast(
                    truth, processed_prediction(raw_prediction, args.threshold)
                )
                row: dict[str, object] = {
                    "fold": fold,
                    "validation_start": str(pd.Timestamp(validation_weeks[0]).date()),
                    "validation_end": str(pd.Timestamp(validation_weeks[-1]).date()),
                    "experiment": experiment,
                    "hist_features": len(columns),
                    "status": "ok",
                    **metrics,
                    "normalized_score": normalized_score(metrics, baseline_metrics),
                    "raw_normalized_score": normalized_score(raw_metrics, baseline_metrics),
                    "run_seconds": elapsed,
                }
            except Exception as error:
                row = {
                    "fold": fold,
                    "validation_start": str(pd.Timestamp(validation_weeks[0]).date()),
                    "validation_end": str(pd.Timestamp(validation_weeks[-1]).date()),
                    "experiment": experiment,
                    "hist_features": len(columns),
                    "status": f"failed: {type(error).__name__}: {error}",
                    **{metric: np.nan for metric in METRICS},
                    "normalized_score": np.nan,
                    "raw_normalized_score": np.nan,
                    "run_seconds": time.perf_counter() - started,
                }
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(result_path, index=False)
            refresh_report(args.output_dir, results, args.max_steps)
            print(json.dumps(row, ensure_ascii=False))
            del nf, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    refresh_report(args.output_dir, results, args.max_steps)


if __name__ == "__main__":
    main()
