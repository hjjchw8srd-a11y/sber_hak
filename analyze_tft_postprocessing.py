"""Проверка порога обнуления и округления уже обученных прогнозов TFT."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fast_score import compute_all_metrics_fast
from score import normalized_score
from train_tft_weekly import METRICS, baseline_mean13, truth_frame


THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]


def apply_rule(values: pd.Series, threshold: float, round_to_int: bool) -> np.ndarray:
    prediction = values.to_numpy(dtype=float).copy()
    prediction[prediction < threshold] = 0.0
    if round_to_int:
        prediction = np.rint(prediction)
    return np.clip(prediction, 0, None)


def main() -> None:
    data_path = Path("weekly_data/weekly_dataset.csv.gz")
    output_dir = Path("experiments/tft")
    raw = pd.read_csv(data_path, parse_dates=["week"])
    folds = pd.read_csv(output_dir / "tft_rolling_results.csv")
    detail_rows: list[dict[str, object]] = []

    for fold_row in folds.itertuples(index=False):
        fold = int(fold_row.fold)
        start = pd.Timestamp(fold_row.validation_start)
        end = pd.Timestamp(fold_row.validation_end)
        weeks = pd.Index(pd.date_range(start, end, freq="W-MON"))
        truth = truth_frame(raw, weeks)
        baseline = baseline_mean13(raw, start, weeks)
        baseline_metrics = compute_all_metrics_fast(truth, baseline)
        prediction = pd.read_csv(
            output_dir / f"tft_predictions_fold_{fold}.csv.gz", parse_dates=["week"]
        )

        for threshold in THRESHOLDS:
            for round_to_int in [False, True]:
                processed = prediction.copy()
                processed["quantity"] = apply_rule(
                    processed["quantity"], threshold, round_to_int
                )
                metrics = compute_all_metrics_fast(truth, processed)
                detail_rows.append(
                    {
                        "fold": fold,
                        "threshold": threshold,
                        "round_to_int": round_to_int,
                        **metrics,
                        "normalized_score": normalized_score(metrics, baseline_metrics),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(output_dir / "tft_postprocessing_details.csv", index=False)
    summary = (
        detail.groupby(["threshold", "round_to_int"], as_index=False)
        .agg(
            mean_score=("normalized_score", "mean"),
            std_score=("normalized_score", "std"),
            mean_smape=("smape", "mean"),
            mean_wape=("wape", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            mean_rmsle=("rmsle", "mean"),
        )
        .sort_values("mean_score")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "tft_postprocessing_summary.csv", index=False)

    cross_fitted_rows: list[dict[str, object]] = []
    for held_out_fold in sorted(detail["fold"].unique()):
        train_scores = (
            detail[detail["fold"] != held_out_fold]
            .groupby(["threshold", "round_to_int"], as_index=False)["normalized_score"]
            .mean()
            .sort_values("normalized_score")
        )
        selected = train_scores.iloc[0]
        test = detail[
            (detail["fold"] == held_out_fold)
            & (detail["threshold"] == selected["threshold"])
            & (detail["round_to_int"] == selected["round_to_int"])
        ].iloc[0]
        cross_fitted_rows.append(
            {
                "held_out_fold": int(held_out_fold),
                "selected_threshold": float(selected["threshold"]),
                "selected_round_to_int": bool(selected["round_to_int"]),
                "held_out_score": float(test["normalized_score"]),
                **{metric: float(test[metric]) for metric in METRICS},
            }
        )
    cross_fitted = pd.DataFrame(cross_fitted_rows)
    cross_fitted.to_csv(output_dir / "tft_postprocessing_cross_fitted.csv", index=False)

    best = summary.iloc[0]
    raw_score = float(
        summary[(summary["threshold"] == 0) & (~summary["round_to_int"])]["mean_score"].iloc[0]
    )
    cross_score = float(cross_fitted["held_out_score"].mean())
    for fold in sorted(detail["fold"].unique()):
        prediction_path = output_dir / f"tft_predictions_fold_{int(fold)}.csv.gz"
        prediction = pd.read_csv(prediction_path)
        prediction["quantity"] = apply_rule(
            prediction["quantity"],
            float(best["threshold"]),
            bool(best["round_to_int"]),
        )
        prediction.to_csv(
            output_dir / f"tft_predictions_postprocessed_fold_{int(fold)}.csv.gz",
            index=False,
        )
    report = f"""# Постобработка прогнозов TFT

Проверены пороги обнуления от 0 до 2 и два режима: оставить вещественный
прогноз или округлить его до целого. Это не новое обучение, а обработка уже
готового выхода TFT.

- исходный средний score: **{raw_score:.4f}**;
- лучший порог на всех трёх окнах: **{float(best['threshold']):.2f}**;
- округление в лучшем варианте: **{'да' if bool(best['round_to_int']) else 'нет'}**;
- лучший исследовательский score: **{float(best['mean_score']):.4f}**;
- cross-fitted score, когда каждое окно не участвует в выборе своего правила:
  **{cross_score:.4f}**;
- CatBoost на этих окнах: **0.7433**.

Лучший результат на тех же окнах нельзя считать независимой оценкой: правило
выбрано по этим данным. Для более честного ориентира приведён cross-fitted
вариант. Меньше — лучше.
"""
    (output_dir / "TFT_POSTPROCESSING.md").write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    for rounded, group in summary.groupby("round_to_int"):
        ordered = group.sort_values("threshold")
        ax.plot(
            ordered["threshold"],
            ordered["mean_score"],
            marker="o",
            label="с округлением" if rounded else "без округления",
        )
    ax.axhline(0.7433, color="#2F855A", linestyle="--", label="CatBoost 0.7433")
    ax.axhline(1.0, color="#C53030", linestyle=":", label="mean13 = 1.0")
    ax.set_xlabel("Обнулять прогнозы меньше порога")
    ax.set_ylabel("Средний normalized score")
    ax.set_title("TFT: влияние порога и округления")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tft_postprocessing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "raw_score": raw_score,
                "best_threshold": float(best["threshold"]),
                "best_round_to_int": bool(best["round_to_int"]),
                "best_exploratory_score": float(best["mean_score"]),
                "cross_fitted_score": cross_score,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
