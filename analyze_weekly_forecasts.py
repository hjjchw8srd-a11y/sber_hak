"""Диагностика текущей модели на трёх характерных временных рядах."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fast_score import compute_all_metrics_fast
from run_weekly_feature_experiments import (
    HORIZON,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
    prediction_frame,
    train_and_predict,
)


DATA_PATH = Path("weekly_data/weekly_dataset.csv.gz")
OUTPUT_DIR = Path("experiments/forecast_diagnostics")
EXPERIMENT = "base_calendar_network"


def smape_array(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = (np.abs(actual) + np.abs(prediction)) / 2.0
    values = np.divide(
        np.abs(actual - prediction),
        denominator,
        out=np.zeros_like(actual, dtype=float),
        where=denominator > 0,
    )
    return float(values.mean() * 100)


def choose_series(quantity: np.ndarray, validation_origin: int) -> dict[str, int]:
    history13 = quantity[validation_origin - 13 : validation_origin]
    history26 = quantity[validation_origin - 26 : validation_origin]
    previous9 = quantity[validation_origin - 13 : validation_origin - 4]
    recent4 = quantity[validation_origin - 4 : validation_origin]

    mean13 = history13.mean(axis=0)
    std13 = history13.std(axis=0)
    nonzero13 = (history13 > 0).mean(axis=0)
    cv13 = np.divide(
        std13,
        mean13,
        out=np.full_like(std13, np.inf),
        where=mean13 > 0,
    )
    history_nonzero26 = (history26 > 0).mean(axis=0)

    stable_candidates = np.where((nonzero13 >= 0.85) & (mean13 >= 2))[0]
    stable = stable_candidates[np.argmin(cv13[stable_candidates])]

    sparse_candidates = np.where(
        (history_nonzero26 >= 0.08)
        & (history_nonzero26 <= 0.30)
        & (history26.sum(axis=0) >= 3)
    )[0]
    sparse_activity = history26[:, sparse_candidates].sum(axis=0)
    sparse = sparse_candidates[np.argsort(sparse_activity)[len(sparse_candidates) // 2]]

    recent_mean = recent4.mean(axis=0)
    previous_mean = previous9.mean(axis=0)
    growth = np.divide(
        recent_mean + 0.25,
        previous_mean + 0.25,
    )
    growth_candidates = np.where(
        (recent_mean >= 2) & (previous_mean >= 0.25) & (nonzero13 >= 0.4)
    )[0]
    growing = growth_candidates[np.argmax(growth[growth_candidates])]

    return {
        "Стабильный ряд": int(stable),
        "Разреженный ряд": int(sparse),
        "Растущий ряд": int(growing),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    validation_origin = len(matrices.weeks) - HORIZON
    latest_train_origin = validation_origin - HORIZON
    train_origins = np.arange(latest_train_origin - 17, latest_train_origin + 1)

    train, target, _ = matrices.make_rows(train_origins, 350_000)
    validation, _, _ = matrices.make_rows(np.array([validation_origin]), None)
    columns = columns_for_experiment(EXPERIMENT)
    prediction, diagnostics = train_and_predict(
        train, target, validation, columns, iterations=160
    )
    prediction_matrix = prediction.reshape(HORIZON, matrices.n_pairs)
    actual_matrix = matrices.quantity[
        validation_origin : validation_origin + HORIZON
    ]
    selected = choose_series(matrices.quantity, validation_origin)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    rows: list[dict[str, object]] = []
    history_start = validation_origin - 26
    history_weeks = matrices.weeks[history_start:validation_origin]
    validation_weeks = matrices.weeks[
        validation_origin : validation_origin + HORIZON
    ]

    for ax, (series_type, pair_idx) in zip(axes, selected.items()):
        actual = actual_matrix[:, pair_idx].astype(float)
        forecast = prediction_matrix[:, pair_idx].astype(float)
        history = matrices.quantity[history_start:validation_origin, pair_idx]
        pair = matrices.pairs.iloc[pair_idx]
        title = f"{series_type}: {pair['warehouse_name']} × {pair['article']}"

        ax.plot(history_weeks, history, marker="o", markersize=3, label="История")
        ax.plot(
            validation_weeks,
            actual,
            marker="o",
            linewidth=2,
            color="#2F855A",
            label="Факт holdout",
        )
        ax.plot(
            validation_weeks,
            forecast,
            marker="o",
            linewidth=2,
            linestyle="--",
            color="#C53030",
            label="Прогноз",
        )
        ax.axvline(validation_weeks[0], color="#4A5568", linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Продажи за неделю")
        ax.legend(loc="upper left", ncol=3)
        ax.grid(alpha=0.25)

        rows.append(
            {
                "series_type": series_type,
                "warehouse_name": pair["warehouse_name"],
                "article": pair["article"],
                "history_nonzero_rate_26": float((history > 0).mean()),
                "history_mean_13": float(history[-13:].mean()),
                "actual_sum_9": float(actual.sum()),
                "forecast_sum_9": float(forecast.sum()),
                "smape_9": smape_array(actual, forecast),
                "mae_9": float(np.abs(actual - forecast).mean()),
            }
        )

    axes[-1].set_xlabel("Понедельник недели")
    fig.suptitle(
        "Диагностика base_calendar_network на последних 9 неделях",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        OUTPUT_DIR / "selected_series_forecasts.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    selected_frame = pd.DataFrame(rows)
    selected_frame.to_csv(OUTPUT_DIR / "selected_series_metrics.csv", index=False)
    truth = make_truth(matrices, validation_origin)
    overall_metrics = compute_all_metrics_fast(
        truth, prediction_frame(truth, prediction)
    )
    summary = {
        "experiment": EXPERIMENT,
        "validation_start": str(validation_weeks[0].date()),
        "validation_end": str(validation_weeks[-1].date()),
        "overall_metrics": overall_metrics,
        "model_diagnostics": diagnostics,
        "selected_series": rows,
    }
    (OUTPUT_DIR / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
