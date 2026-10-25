"""Rolling validation групп признаков на нескольких окнах по 9 недель."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fast_score import compute_all_metrics_fast
from run_weekly_feature_experiments import (
    EXPERIMENTS,
    HORIZON,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
    prediction_frame,
    train_and_predict,
)
from score import normalized_score


DEFAULT_EXPERIMENTS = list(EXPERIMENTS)
METRICS = ["smape", "wape", "rmse", "bias", "rmsle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/rolling"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-step", type=int, default=HORIZON)
    parser.add_argument("--train-origins", type=int, default=18)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=DEFAULT_EXPERIMENTS,
    )
    parser.add_argument("--restart", action="store_true", help="Не использовать сохранённый прогресс")
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def refresh_reports(output_dir: Path, results: pd.DataFrame) -> None:
    model_rows = results[results["experiment"] != "baseline_mean13"].copy()
    aggregate = (
        model_rows.groupby("experiment", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_score=("normalized_score", "mean"),
            std_score=("normalized_score", "std"),
            worst_score=("normalized_score", "max"),
            mean_smape=("smape", "mean"),
            mean_wape=("wape", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            mean_rmsle=("rmsle", "mean"),
        )
        .sort_values(["mean_score", "std_score"])
        .reset_index(drop=True)
    )
    aggregate.insert(0, "rank", np.arange(1, len(aggregate) + 1))
    aggregate.to_csv(output_dir / "rolling_feature_summary.csv", index=False)

    complete = aggregate[aggregate["folds"] == results["fold"].nunique()].copy()
    if not complete.empty:
        plot = complete.sort_values("mean_score", ascending=False)
        fig, ax = plt.subplots(figsize=(11, max(5, 0.65 * len(plot) + 2)))
        errors = plot["std_score"].fillna(0).to_numpy()
        positions = np.arange(len(plot))
        ax.errorbar(
            plot["mean_score"],
            positions,
            xerr=errors,
            fmt="o",
            markersize=8,
            color="#2B6CB0",
            ecolor="#4A5568",
            capsize=4,
        )
        ax.set_yticks(positions, plot["experiment"])
        left = float((plot["mean_score"] - errors).min() - 0.002)
        right = float((plot["mean_score"] + errors).max() + 0.002)
        ax.set_xlim(left, right)
        for position, score in zip(positions, plot["mean_score"]):
            ax.text(score, position + 0.23, f"{score:.4f}", ha="center", fontsize=8)
        ax.set_xlabel("Средний normalized score (меньше — лучше)")
        ax.set_title(
            "Rolling validation: среднее и разброс по окнам\n"
            "Ось X увеличена; baseline mean13 = 1.0",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "rolling_feature_comparison.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    aggregate_show = aggregate.copy()
    for column in aggregate_show.columns:
        if column not in {"rank", "experiment", "folds"}:
            aggregate_show[column] = aggregate_show[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    detail = results[
        ["fold", "validation_start", "validation_end", "experiment", "normalized_score", *METRICS]
    ].sort_values(["fold", "normalized_score"])
    detail_show = detail.copy()
    for column in ["normalized_score", *METRICS]:
        detail_show[column] = detail_show[column].map(lambda value: f"{value:.4f}")

    report = f"""# Rolling validation признаков

## Как проводится проверка

- Каждое окно содержит 9 недель, окна не пересекаются.
- Для каждого окна модель обучается заново только на более ранних данных.
- Настройки CatBoost, seed и число обучающих строк одинаковы.
- `normalized_score` считается относительно `mean13` соответствующего окна.
- Меньше — лучше. Разброс важен: стабильный набор предпочтительнее случайной победы.

## Средние результаты

{markdown_table(aggregate_show)}

## Результаты по окнам

{markdown_table(detail_show)}
"""
    (output_dir / "ROLLING_VALIDATION.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "rolling_feature_results.csv"
    if result_path.exists() and not args.restart:
        results = pd.read_csv(result_path)
    else:
        results = pd.DataFrame()

    data = pd.read_csv(args.data, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    latest_validation_origin = len(matrices.weeks) - HORIZON
    validation_origins = [
        latest_validation_origin - fold * args.fold_step for fold in range(args.folds)
    ]

    for fold, validation_origin in enumerate(validation_origins):
        validation_start = str(matrices.weeks[validation_origin].date())
        validation_end = str(matrices.weeks[validation_origin + HORIZON - 1].date())
        latest_train_origin = validation_origin - HORIZON
        earliest_train_origin = max(13, latest_train_origin - args.train_origins + 1)
        train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)

        train, target, _ = matrices.make_rows(
            train_origins, args.max_train_rows, seed=42 + fold
        )
        validation, _, _ = matrices.make_rows(np.array([validation_origin]), None)
        truth = make_truth(matrices, validation_origin)
        baseline_prediction = prediction_frame(
            truth, validation["quantity_mean_13"].to_numpy()
        )
        baseline_metrics = compute_all_metrics_fast(truth, baseline_prediction)

        completed = set()
        if not results.empty:
            completed = set(
                results.loc[results["fold"] == fold, "experiment"].astype(str)
            )

        if "baseline_mean13" not in completed:
            baseline_row: dict[str, object] = {
                "fold": fold,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "experiment": "baseline_mean13",
                "feature_count": 1,
                **baseline_metrics,
                "normalized_score": 1.0,
                "run_seconds": 0.0,
            }
            results = pd.concat([results, pd.DataFrame([baseline_row])], ignore_index=True)
            results.to_csv(result_path, index=False)

        for name in args.experiments:
            if name in completed:
                print(f"skip fold={fold} experiment={name}")
                continue
            started = time.perf_counter()
            columns = columns_for_experiment(name)
            prediction, diagnostics = train_and_predict(
                train, target, validation, columns, args.iterations
            )
            metrics = compute_all_metrics_fast(
                truth, prediction_frame(truth, prediction)
            )
            row: dict[str, object] = {
                "fold": fold,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "experiment": name,
                "feature_count": len(columns),
                **metrics,
                "normalized_score": normalized_score(metrics, baseline_metrics),
                "run_seconds": time.perf_counter() - started,
                **diagnostics,
            }
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(result_path, index=False)
            refresh_reports(args.output_dir, results)
            print(json.dumps(row, ensure_ascii=True))

    refresh_reports(args.output_dir, results)


if __name__ == "__main__":
    main()
