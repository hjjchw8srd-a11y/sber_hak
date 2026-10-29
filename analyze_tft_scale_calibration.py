"""Проверка простой масштабной калибровки сохранённых rolling-прогнозов TFT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fast_score import compute_all_metrics_fast
from score import normalized_score
from train_tft_weekly import baseline_mean13, truth_frame


SCALES = [0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]
THRESHOLDS = [0.00, 0.50, 0.75, 1.00]


def postprocess(values: pd.Series, scale: float, threshold: float) -> np.ndarray:
    prediction = values.to_numpy(dtype=float) * scale
    prediction[prediction < threshold] = 0.0
    return np.rint(np.clip(prediction, 0, None))


def main() -> None:
    output_dir = Path("experiments/model_weaknesses")
    output_dir.mkdir(parents=True, exist_ok=True)
    tft_dir = Path("experiments/tft")
    data = pd.read_csv("weekly_data/weekly_dataset.csv.gz", parse_dates=["week"])
    folds = pd.read_csv(tft_dir / "tft_rolling_results.csv")
    results: list[dict[str, float | int]] = []

    for fold_row in folds.itertuples(index=False):
        fold = int(fold_row.fold)
        start = pd.Timestamp(fold_row.validation_start)
        end = pd.Timestamp(fold_row.validation_end)
        weeks = pd.Index(pd.date_range(start, end, freq="W-MON"))
        truth = truth_frame(data, weeks)
        baseline = baseline_mean13(data, start, weeks)
        baseline_metrics = compute_all_metrics_fast(truth, baseline)
        raw = pd.read_csv(
            tft_dir / f"tft_predictions_fold_{fold}.csv.gz", parse_dates=["week"]
        )

        for scale in SCALES:
            for threshold in THRESHOLDS:
                prediction = raw.copy()
                prediction["quantity"] = postprocess(
                    prediction["quantity"], scale, threshold
                )
                metrics = compute_all_metrics_fast(truth, prediction)
                results.append(
                    {
                        "fold": fold,
                        "scale": scale,
                        "threshold": threshold,
                        **metrics,
                        "normalized_score": normalized_score(metrics, baseline_metrics),
                        "actual_sum": float(truth["quantity"].sum()),
                        "prediction_sum": float(prediction["quantity"].sum()),
                    }
                )

    detail = pd.DataFrame(results)
    detail.to_csv(output_dir / "tft_scale_calibration_detail.csv", index=False)
    summary = (
        detail.groupby(["scale", "threshold"], as_index=False)
        .agg(
            mean_score=("normalized_score", "mean"),
            std_score=("normalized_score", "std"),
            mean_smape=("smape", "mean"),
            mean_wape=("wape", "mean"),
            mean_bias=("bias", "mean"),
            actual_sum=("actual_sum", "sum"),
            prediction_sum=("prediction_sum", "sum"),
        )
        .sort_values("mean_score")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "tft_scale_calibration_summary.csv", index=False)

    cross_rows: list[dict[str, float | int]] = []
    for held_out in sorted(detail["fold"].unique()):
        train = (
            detail[detail["fold"] != held_out]
            .groupby(["scale", "threshold"], as_index=False)["normalized_score"]
            .mean()
            .sort_values("normalized_score")
        )
        selected = train.iloc[0]
        test = detail[
            (detail["fold"] == held_out)
            & (detail["scale"] == selected["scale"])
            & (detail["threshold"] == selected["threshold"])
        ].iloc[0]
        cross_rows.append(
            {
                "held_out_fold": int(held_out),
                "selected_scale": float(selected["scale"]),
                "selected_threshold": float(selected["threshold"]),
                "held_out_score": float(test["normalized_score"]),
            }
        )
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output_dir / "tft_scale_calibration_cross_fitted.csv", index=False)

    base = summary[(summary["scale"] == 1.0) & (summary["threshold"] == 0.75)].iloc[0]
    best = summary.iloc[0]
    report = f"""# Калибровка уровня прогноза TFT

Проверена простая постобработка сохранённых прогнозов: умножение выхода TFT на
коэффициент от 0.80 до 1.50, затем порог и округление. Нового обучения здесь нет.

- исходная схема (scale 1.00, threshold 0.75): **{base['mean_score']:.6f}**;
- лучший исследовательский вариант: scale **{best['scale']:.2f}**, threshold
  **{best['threshold']:.2f}**, score **{best['mean_score']:.6f}**;
- cross-fitted score: **{cross['held_out_score'].mean():.6f}**;
- коэффициенты, выбранные без удерживаемого окна:
  {', '.join(f"{row.selected_scale:.2f}" for row in cross.itertuples(index=False))}.

Результат нельзя считать финальной независимой оценкой: коэффициент выбирался
по тем же трём rolling-окнам. Но cross-fitted проверка показывает, является ли
недооценка уровня достаточно устойчивой для следующего полноценного эксперимента.
"""
    (output_dir / "TFT_SCALE_CALIBRATION.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
