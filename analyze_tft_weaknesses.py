"""Сегментная диагностика сохранённых rolling-прогнозов базового TFT."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["week", "warehouse_name", "article"]
OUTPUT_DIR = Path("experiments/model_weaknesses")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def history_features(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = data[data["week"] < cutoff].sort_values(KEYS)
    pair = ["warehouse_name", "article"]
    last13 = history.groupby(pair, observed=True).tail(13)
    last4 = history.groupby(pair, observed=True).tail(4)
    stats13 = (
        last13.groupby(pair, as_index=False, observed=True)
        .agg(
            mean13=("quantity", "mean"),
            nonzero_rate13=("quantity", lambda values: float((values > 0).mean())),
            std13=("quantity", "std"),
        )
        .fillna({"std13": 0})
    )
    stats4 = last4.groupby(pair, as_index=False, observed=True).agg(mean4=("quantity", "mean"))
    last_known = history.groupby(pair, observed=True).tail(1)[
        pair + ["stock_week_end", "stock_observed_days", "week"]
    ].rename(columns={"week": "last_history_week"})
    positive = history[history["quantity"] > 0]
    last_sale = (
        positive.groupby(pair, as_index=False, observed=True)["week"]
        .max()
        .rename(columns={"week": "last_sale_week"})
    )
    result = stats13.merge(stats4, on=pair, how="left").merge(last_known, on=pair, how="left")
    result = result.merge(last_sale, on=pair, how="left")
    result["growth_4_vs_13"] = np.log1p(result["mean4"]) - np.log1p(result["mean13"])
    result["weeks_since_sale"] = (
        (cutoff - result["last_sale_week"]).dt.days.div(7).fillna(26).clip(upper=26)
    )
    result["demand_regime"] = pd.cut(
        result["nonzero_rate13"],
        bins=[-0.001, 0, 0.25, 0.5, 1.001],
        labels=["нет продаж", "редкий", "прерывистый", "регулярный"],
    ).astype(str)
    result["history_scale"] = pd.cut(
        result["mean13"],
        bins=[-0.001, 0, 1, 3, 10, np.inf],
        labels=["0", "0–1", "1–3", "3–10", ">10"],
    ).astype(str)
    result["trend_regime"] = pd.cut(
        result["growth_4_vs_13"],
        bins=[-np.inf, -0.25, 0.25, np.inf],
        labels=["снижение", "стабильно", "рост"],
    ).astype(str)
    result["stock_state"] = np.select(
        [
            result["stock_week_end"].isna(),
            result["stock_week_end"].le(0),
            result["stock_week_end"].between(1, 5),
            result["stock_week_end"].gt(5),
        ],
        ["нет данных", "≤0", "1–5", ">5"],
        default="нет данных",
    )
    return result


def grouped_metrics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, part in frame.groupby(group, observed=True, dropna=False):
        actual = part["quantity_true"].to_numpy(dtype=float)
        prediction = part["quantity_pred"].to_numpy(dtype=float)
        baseline = part["quantity_baseline"].to_numpy(dtype=float)
        actual_positive = actual > 0
        pred_positive = prediction > 0
        tp = int(np.sum(actual_positive & pred_positive))
        fp = int(np.sum(~actual_positive & pred_positive))
        fn = int(np.sum(actual_positive & ~pred_positive))
        denominator = (np.abs(actual) + np.abs(prediction)) / 2
        smape = np.divide(
            np.abs(actual - prediction),
            denominator,
            out=np.zeros_like(actual),
            where=denominator > 0,
        ).mean() * 100
        baseline_denominator = (np.abs(actual) + np.abs(baseline)) / 2
        baseline_smape = np.divide(
            np.abs(actual - baseline),
            baseline_denominator,
            out=np.zeros_like(actual),
            where=baseline_denominator > 0,
        ).mean() * 100
        model_wape = safe_ratio(
            np.abs(actual - prediction).sum(), np.abs(actual).sum()
        ) * 100
        baseline_wape = safe_ratio(
            np.abs(actual - baseline).sum(), np.abs(actual).sum()
        ) * 100
        rows.append(
            {
                group: value,
                "rows": len(part),
                "actual_sum": actual.sum(),
                "prediction_sum": prediction.sum(),
                "smape_points": smape,
                "wape": model_wape,
                "mean13_wape": baseline_wape,
                "wape_delta_vs_mean13": model_wape - baseline_wape,
                "mean13_smape_points": baseline_smape,
                "smape_delta_vs_mean13": smape - baseline_smape,
                "signed_bias": safe_ratio(prediction.sum() - actual.sum(), actual.sum()) * 100,
                "actual_positive_rate": actual_positive.mean() * 100,
                "pred_positive_rate": pred_positive.mean() * 100,
                "precision": safe_ratio(tp, tp + fp) * 100,
                "recall": safe_ratio(tp, tp + fn) * 100,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.2f}")
    headers = shown.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("weekly_data/weekly_dataset.csv.gz", parse_dates=["week"])
    folds = pd.read_csv("experiments/tft/tft_rolling_results.csv")
    all_rows = []

    for fold_row in folds.itertuples(index=False):
        fold = int(fold_row.fold)
        cutoff = pd.Timestamp(fold_row.validation_start)
        end = pd.Timestamp(fold_row.validation_end)
        truth = data[data["week"].between(cutoff, end)][
            KEYS + ["quantity", "category_lvl_1", "category_lvl_2"]
        ].rename(columns={"quantity": "quantity_true"})
        prediction = pd.read_csv(
            f"experiments/tft/tft_predictions_postprocessed_fold_{fold}.csv.gz",
            parse_dates=["week"],
        ).rename(columns={"quantity": "quantity_pred"})
        raw_prediction = pd.read_csv(
            f"experiments/tft/tft_predictions_fold_{fold}.csv.gz", parse_dates=["week"]
        ).rename(columns={"quantity": "quantity_raw"})
        frame = truth.merge(prediction, on=KEYS, validate="one_to_one").merge(
            raw_prediction, on=KEYS, validate="one_to_one"
        )
        history = data[data["week"] < cutoff].sort_values(KEYS)
        baseline = (
            history.groupby(["warehouse_name", "article"], observed=True)
            .tail(13)
            .groupby(["warehouse_name", "article"], as_index=False, observed=True)["quantity"]
            .mean()
            .rename(columns={"quantity": "quantity_baseline"})
        )
        frame = frame.merge(
            baseline, on=["warehouse_name", "article"], how="left", validate="many_to_one"
        )
        frame["quantity_baseline"] = frame["quantity_baseline"].fillna(0)
        frame["fold"] = fold
        frame["horizon"] = ((frame["week"] - cutoff).dt.days // 7 + 1).astype(int)
        frame = frame.merge(
            history_features(data, cutoff), on=["warehouse_name", "article"], how="left"
        )
        all_rows.append(frame)

    frame = pd.concat(all_rows, ignore_index=True)
    actual_positive = frame["quantity_true"] > 0
    pred_positive = frame["quantity_pred"] > 0
    tp = int(np.sum(actual_positive & pred_positive))
    fp = int(np.sum(~actual_positive & pred_positive))
    fn = int(np.sum(actual_positive & ~pred_positive))
    tn = int(np.sum(~actual_positive & ~pred_positive))
    overview = {
        "rows": len(frame),
        "zero_target_rate": float((~actual_positive).mean()),
        "predicted_zero_rate": float((~pred_positive).mean()),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "actual_sum": float(frame["quantity_true"].sum()),
        "prediction_sum": float(frame["quantity_pred"].sum()),
        "stock_missing_rate": float(frame["stock_week_end"].isna().mean()),
    }
    pd.Series(overview).to_json(OUTPUT_DIR / "overview.json", force_ascii=False, indent=2)

    groups = [
        "horizon",
        "demand_regime",
        "history_scale",
        "trend_regime",
        "stock_state",
        "warehouse_name",
        "category_lvl_1",
    ]
    summaries = {}
    for group in groups:
        summary = grouped_metrics(frame, group)
        summary.to_csv(OUTPUT_DIR / f"errors_by_{group}.csv", index=False)
        summaries[group] = summary

    raw_bins = [-0.001, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, np.inf]
    raw_labels = ["0–0.1", "0.1–0.25", "0.25–0.5", "0.5–0.75", "0.75–1", "1–2", "2–5", "5–10", ">10"]
    frame["raw_prediction_bin"] = pd.cut(
        frame["quantity_raw"], bins=raw_bins, labels=raw_labels, include_lowest=True
    )
    calibration = (
        frame.groupby("raw_prediction_bin", observed=True)
        .agg(
            rows=("quantity_true", "size"),
            mean_prediction=("quantity_raw", "mean"),
            mean_actual=("quantity_true", "mean"),
            actual_positive_rate=("quantity_true", lambda values: float((values > 0).mean() * 100)),
        )
        .reset_index()
    )
    calibration.to_csv(OUTPUT_DIR / "raw_prediction_calibration.csv", index=False)

    horizon = summaries["horizon"]
    regime = summaries["demand_regime"]
    stock = summaries["stock_state"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(horizon["horizon"], horizon["wape"], marker="o", label="WAPE")
    axes[0, 0].plot(horizon["horizon"], horizon["smape_points"], marker="o", label="SMAPE")
    axes[0, 0].set_title("Ошибка по горизонту")
    axes[0, 0].set_xlabel("Неделя прогноза")
    axes[0, 0].legend()
    x = np.arange(len(regime))
    axes[0, 1].bar(x - 0.18, regime["actual_positive_rate"], 0.36, label="факт")
    axes[0, 1].bar(x + 0.18, regime["pred_positive_rate"], 0.36, label="прогноз")
    axes[0, 1].set_xticks(x, regime["demand_regime"], rotation=15)
    axes[0, 1].set_title("Частота продаж по типу спроса")
    axes[0, 1].legend()
    axes[1, 0].bar(stock["stock_state"], stock["wape"], color="#F58518")
    axes[1, 0].set_title("WAPE по последнему известному остатку")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 1].plot(
        calibration["mean_prediction"], calibration["actual_positive_rate"], marker="o"
    )
    axes[1, 1].set_title("Калибровка сырого TFT")
    axes[1, 1].set_xlabel("Средний сырой прогноз")
    axes[1, 1].set_ylabel("Доля ненулевого факта, %")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tft_weaknesses.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    worst_horizon = horizon.sort_values("wape", ascending=False).head(3)
    worst_regime = regime.sort_values("wape", ascending=False)
    worst_stock = stock.sort_values("wape", ascending=False)
    report = f"""# Слабые места недельного TFT

Диагностика построена по {len(frame):,} точкам трёх rolling-окон.

## Общая картина

- нулевых фактов: **{overview['zero_target_rate']:.1%}**;
- нулевых прогнозов после порога и округления: **{overview['predicted_zero_rate']:.1%}**;
- precision факта продажи: **{overview['precision']:.1%}**;
- recall факта продажи: **{overview['recall']:.1%}**;
- ложных продаж: **{fp:,}**, пропущенных продаж: **{fn:,}**;
- сумма факта: **{overview['actual_sum']:,.0f}**, сумма прогноза:
  **{overview['prediction_sum']:,.0f}**;
- для **{overview['stock_missing_rate']:.1%}** точек последний остаток неизвестен.

## Худшие горизонты

{markdown_table(worst_horizon[['horizon', 'wape', 'mean13_wape', 'wape_delta_vs_mean13', 'signed_bias', 'precision', 'recall']])}

## Ошибка по типу спроса

{markdown_table(worst_regime[['demand_regime', 'rows', 'wape', 'mean13_wape', 'wape_delta_vs_mean13', 'actual_positive_rate', 'pred_positive_rate', 'precision', 'recall']])}

## Ошибка по состоянию остатка

{markdown_table(worst_stock[['stock_state', 'rows', 'wape', 'signed_bias', 'precision', 'recall']])}

Полные таблицы по горизонту, масштабу спроса, тренду, складам и категориям
сохранены рядом с отчётом.
"""
    (OUTPUT_DIR / "TFT_WEAKNESSES.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
