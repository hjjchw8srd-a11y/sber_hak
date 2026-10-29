"""Обучение и rolling validation недельного TFT из NeuralForecast.

TFT получает сам временной ряд quantity, известный будущий календарь,
исторический сетевой контекст и статические one-hot признаки склада/категорий.
Все исторические признаки на неделе t рассчитаны только по неделям до t.
"""

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
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import TFT

from fast_score import compute_all_metrics_fast
from score import normalized_score


HORIZON = 9
KEYS = ["week", "warehouse_name", "article"]
FUTR_EXOG = ["time_idx", "month", "quarter", "week_of_year", "week_sin", "week_cos"]
BASE_HIST_EXOG = [
    "article_network_sales_mean_13_log",
    "article_network_growth_4_vs_13",
    "warehouse_share_article_13",
    "active_warehouses_article_13",
]
STOCK_BASIC_EXOG = [
    "stock_end_log",
    "stock_mean_log",
    "stock_nonpositive_rate_hist",
    "stock_observed_days_hist",
    "stock_missing",
]
STOCK_FULL_EXOG = [
    *STOCK_BASIC_EXOG,
    "stock_start_log",
    "stock_min_log",
    "stock_max_log",
    "stock_median_log",
    "stock_change_log",
    "stock_nonpositive_days_hist",
    "stock_negative_days_hist",
]
STOCK_DERIVED_EXOG = [
    *STOCK_BASIC_EXOG,
    "stock_to_sales_weeks_log",
    "stockout_flag",
    "probable_replenishment_log",
]
MONEY_EXOG = [
    "sales_sum_log_hist",
    "discount_sum_log_hist",
    "discount_share_hist",
    "paid_value_per_unit_log",
    "full_value_per_unit_log",
]
DEMAND_DYNAMICS_EXOG = [
    "quantity_min_13_hist",
    "quantity_max_13_hist",
    "quantity_median_13_hist",
    "quantity_cv_13_hist",
    "quantity_mean_26_hist",
    "growth_log_4_vs_previous_4_hist",
    "growth_log_13_vs_previous_13_hist",
    "weeks_since_last_sale_hist",
    "zero_streak_hist",
    "autocorrelation_lag_1_hist",
    "autocorrelation_lag_4_hist",
]
# Обратная совместимость для первого базового эксперимента и тестов.
HIST_EXOG = BASE_HIST_EXOG
METRICS = ["smape", "wape", "rmse", "bias", "rmsle"]


def signed_log1p(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=np.float32)
    return pd.Series(np.sign(array) * np.log1p(np.abs(array)), index=values.index)


def weeks_since_previous_sale(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=np.float32)
    result = np.empty(len(array), dtype=np.float32)
    last_sale = -1
    for position, value in enumerate(array):
        result[position] = position - last_sale if last_sale >= 0 else 26.0
        if value > 0:
            last_sale = position
    return pd.Series(result, index=values.index)


def previous_zero_streak(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=np.float32)
    result = np.empty(len(array), dtype=np.float32)
    streak = 0
    for position, value in enumerate(array):
        result[position] = streak
        streak = 0 if value > 0 else streak + 1
    return pd.Series(result, index=values.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tft"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-step", type=int, default=HORIZON)
    parser.add_argument("--input-size", type=int, default=26)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--windows-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-check-steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--series-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() else "_" for char in text)


def prepare_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Строит формат NeuralForecast без утечки будущего таргета."""
    data = data.copy()
    optional_stock = [
        "stock_week_start",
        "stock_week_end",
        "stock_min",
        "stock_max",
        "stock_mean",
        "stock_median",
        "stock_observed_days",
        "stock_nonpositive_days",
        "stock_negative_days",
        "stock_nonpositive_rate",
        "stock_change",
    ]
    for column in optional_stock:
        if column not in data:
            data[column] = np.nan
    for column in ["sales_sum", "discount_sum", "discount_rows"]:
        if column not in data:
            data[column] = 0.0
    data["week"] = pd.to_datetime(data["week"])
    data = data.sort_values(KEYS).reset_index(drop=True)
    data["unique_id"] = data["warehouse_name"].astype(str) + "|" + data["article"].astype(str)

    weeks = pd.Index(sorted(data["week"].unique()))
    week_to_idx = pd.Series(np.arange(len(weeks), dtype=np.float32), index=weeks)
    data["time_idx"] = data["week"].map(week_to_idx).astype(np.float32)
    iso_week = data["week"].dt.isocalendar().week.astype(np.float32)
    data["month"] = data["week"].dt.month.astype(np.float32)
    data["quarter"] = data["week"].dt.quarter.astype(np.float32)
    data["week_of_year"] = iso_week
    data["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0).astype(np.float32)
    data["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0).astype(np.float32)

    pair_group = data.groupby(["warehouse_name", "article"], sort=False, observed=True)
    pair_sum_13 = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).sum()
    )
    article_week = (
        data.groupby(["article", "week"], as_index=False, observed=True)["quantity"]
        .sum()
        .sort_values(["article", "week"])
    )
    article_group = article_week.groupby("article", sort=False, observed=True)["quantity"]
    article_week["network_sum_13"] = article_group.transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).sum()
    )
    article_week["network_sum_4"] = article_group.transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).sum()
    )
    article_week["network_mean_13"] = article_week["network_sum_13"] / 13.0
    article_week["network_mean_4"] = article_week["network_sum_4"] / 4.0
    data = data.merge(
        article_week[
            ["article", "week", "network_sum_13", "network_mean_13", "network_mean_4"]
        ],
        on=["article", "week"],
        how="left",
        validate="many_to_one",
    )
    data["article_network_sales_mean_13_log"] = np.log1p(
        data["network_mean_13"].fillna(0)
    ).astype(np.float32)
    data["article_network_growth_4_vs_13"] = (
        np.log1p(data["network_mean_4"].fillna(0))
        - np.log1p(data["network_mean_13"].fillna(0))
    ).astype(np.float32)
    data["warehouse_share_article_13"] = np.divide(
        pair_sum_13.to_numpy(dtype=np.float32),
        data["network_sum_13"].to_numpy(dtype=np.float32),
        out=np.zeros(len(data), dtype=np.float32),
        where=data["network_sum_13"].fillna(0).to_numpy() > 0,
    )
    active = (
        data.assign(_active=(pair_sum_13 > 0).astype(np.float32))
        .groupby(["week", "article"], observed=True)["_active"]
        .transform("sum")
    )
    data["active_warehouses_article_13"] = active.astype(np.float32)

    # Исторические остатки. В TFT они подаются только в encoder, поэтому для
    # будущих девяти недель их знать не требуется. Пропуск и реальный ноль
    # разделены отдельным флагом.
    data["stock_missing"] = data["stock_week_end"].isna().astype(np.float32)
    stock_log_mapping = {
        "stock_week_end": "stock_end_log",
        "stock_mean": "stock_mean_log",
        "stock_week_start": "stock_start_log",
        "stock_min": "stock_min_log",
        "stock_max": "stock_max_log",
        "stock_median": "stock_median_log",
        "stock_change": "stock_change_log",
    }
    for source, target in stock_log_mapping.items():
        data[target] = signed_log1p(data[source].fillna(0)).astype(np.float32)
    data["stock_nonpositive_rate_hist"] = data["stock_nonpositive_rate"].fillna(0).astype(np.float32)
    data["stock_observed_days_hist"] = data["stock_observed_days"].fillna(0).astype(np.float32)
    data["stock_nonpositive_days_hist"] = data["stock_nonpositive_days"].fillna(0).astype(np.float32)
    data["stock_negative_days_hist"] = data["stock_negative_days"].fillna(0).astype(np.float32)
    data["stockout_flag"] = (
        data["stock_week_end"].notna() & data["stock_week_end"].le(0)
    ).astype(np.float32)

    previous_mean_13 = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).mean()
    )
    stock_to_sales = np.divide(
        data["stock_week_end"].fillna(0).to_numpy(dtype=np.float32),
        previous_mean_13.fillna(0).to_numpy(dtype=np.float32) + 1.0,
    )
    stock_to_sales = np.clip(stock_to_sales, -10, 100)
    data["stock_to_sales_weeks_log"] = (
        np.sign(stock_to_sales) * np.log1p(np.abs(stock_to_sales))
    ).astype(np.float32)
    probable_replenishment = (
        data["stock_change"].fillna(0).to_numpy(dtype=np.float32)
        + data["quantity"].to_numpy(dtype=np.float32)
    )
    data["probable_replenishment_log"] = np.log1p(
        np.maximum(probable_replenishment, 0)
    ).astype(np.float32)

    # Денежные показатели известны только для прошлых недель и поэтому также
    # используются как historic exogenous variables.
    sales = data["sales_sum"].fillna(0).to_numpy(dtype=np.float32)
    discount = data["discount_sum"].fillna(0).to_numpy(dtype=np.float32)
    quantity = data["quantity"].to_numpy(dtype=np.float32)
    full_value = sales + discount
    data["sales_sum_log_hist"] = np.log1p(np.maximum(sales, 0)).astype(np.float32)
    data["discount_sum_log_hist"] = np.log1p(np.maximum(discount, 0)).astype(np.float32)
    data["discount_share_hist"] = np.divide(
        discount,
        full_value,
        out=np.zeros_like(discount),
        where=full_value > 0,
    ).astype(np.float32)
    data["paid_value_per_unit_log"] = np.log1p(
        np.maximum(np.divide(sales, quantity, out=np.zeros_like(sales), where=quantity > 0), 0)
    ).astype(np.float32)
    data["full_value_per_unit_log"] = np.log1p(
        np.maximum(
            np.divide(full_value, quantity, out=np.zeros_like(full_value), where=quantity > 0),
            0,
        )
    ).astype(np.float32)
    data["discount_rows_hist"] = data["discount_rows"].fillna(0).astype(np.float32)

    # Ручные признаки формы спроса. Они всегда сдвинуты минимум на одну неделю.
    data["quantity_min_13_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).min()
    ).fillna(0).astype(np.float32)
    data["quantity_max_13_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).max()
    ).fillna(0).astype(np.float32)
    data["quantity_median_13_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=1).median()
    ).fillna(0).astype(np.float32)
    std_13 = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=2).std()
    ).fillna(0)
    data["quantity_cv_13_hist"] = np.divide(
        std_13.to_numpy(dtype=np.float32),
        previous_mean_13.fillna(0).to_numpy(dtype=np.float32),
        out=np.zeros(len(data), dtype=np.float32),
        where=previous_mean_13.fillna(0).to_numpy() > 0,
    )
    data["quantity_mean_26_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(26, min_periods=1).mean()
    ).fillna(0).astype(np.float32)
    mean_4 = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    ).fillna(0)
    previous_mean_4 = pair_group["quantity"].transform(
        lambda values: values.shift(5).rolling(4, min_periods=1).mean()
    ).fillna(0)
    previous_previous_mean_13 = pair_group["quantity"].transform(
        lambda values: values.shift(14).rolling(13, min_periods=1).mean()
    ).fillna(0)
    data["growth_log_4_vs_previous_4_hist"] = (
        np.log1p(mean_4) - np.log1p(previous_mean_4)
    ).astype(np.float32)
    data["growth_log_13_vs_previous_13_hist"] = (
        np.log1p(previous_mean_13.fillna(0)) - np.log1p(previous_previous_mean_13)
    ).astype(np.float32)
    data["weeks_since_last_sale_hist"] = pair_group["quantity"].transform(
        weeks_since_previous_sale
    ).clip(upper=26).astype(np.float32)
    data["zero_streak_hist"] = pair_group["quantity"].transform(
        previous_zero_streak
    ).clip(upper=26).astype(np.float32)
    data["autocorrelation_lag_1_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=4).corr(values.shift(2))
    ).replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    data["autocorrelation_lag_4_hist"] = pair_group["quantity"].transform(
        lambda values: values.shift(1).rolling(13, min_periods=4).corr(values.shift(5))
    ).replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    metadata = (
        data[["unique_id", "warehouse_name", "category_lvl_1", "category_lvl_2", "category_lvl_3"]]
        .drop_duplicates("unique_id")
        .sort_values("unique_id")
        .reset_index(drop=True)
    )
    dummy_parts = []
    for column in ["warehouse_name", "category_lvl_1", "category_lvl_2", "category_lvl_3"]:
        dummies = pd.get_dummies(metadata[column].fillna("unknown"), prefix=column, dtype=np.float32)
        dummies.columns = [safe_name(value) for value in dummies.columns]
        dummy_parts.append(dummies)
    static_df = pd.concat([metadata[["unique_id"]], *dummy_parts], axis=1)
    static_columns = [column for column in static_df.columns if column != "unique_id"]

    all_hist_exog = list(
        dict.fromkeys(
            [
                *BASE_HIST_EXOG,
                *STOCK_FULL_EXOG,
                *STOCK_DERIVED_EXOG,
                *MONEY_EXOG,
                *DEMAND_DYNAMICS_EXOG,
            ]
        )
    )
    temporal = data.rename(columns={"week": "ds", "quantity": "y"})[
        ["unique_id", "ds", "y", *FUTR_EXOG, *all_hist_exog]
    ].copy()
    temporal["y"] = temporal["y"].astype(np.float32)
    temporal[all_hist_exog] = temporal[all_hist_exog].fillna(0).astype(np.float32)
    temporal[FUTR_EXOG] = temporal[FUTR_EXOG].astype(np.float32)
    return temporal.sort_values(["unique_id", "ds"]), static_df, static_columns


def truth_frame(data: pd.DataFrame, weeks: pd.Index) -> pd.DataFrame:
    truth = data[data["week"].isin(weeks)][KEYS + ["quantity"]].copy()
    return truth.sort_values(KEYS).reset_index(drop=True)


def baseline_mean13(data: pd.DataFrame, cutoff: pd.Timestamp, weeks: pd.Index) -> pd.DataFrame:
    history = data[data["week"] < cutoff].sort_values(KEYS)
    means = (
        history.groupby(["warehouse_name", "article"], observed=True)
        .tail(13)
        .groupby(["warehouse_name", "article"], as_index=False, observed=True)["quantity"]
        .mean()
    )
    grid = pd.MultiIndex.from_product(
        [weeks, data["warehouse_name"].unique(), data["article"].unique()],
        names=KEYS,
    ).to_frame(index=False)
    return grid.merge(means, on=["warehouse_name", "article"], how="left").fillna({"quantity": 0})


def prediction_to_submission(prediction: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in prediction.columns if column not in {"unique_id", "ds", "cutoff"}]
    if len(value_columns) != 1:
        raise ValueError(f"Не удалось однозначно определить колонку прогноза: {value_columns}")
    mapping = data[["warehouse_name", "article"]].drop_duplicates().copy()
    mapping["unique_id"] = (
        mapping["warehouse_name"].astype(str) + "|" + mapping["article"].astype(str)
    )
    result = prediction[["unique_id", "ds", value_columns[0]]].merge(
        mapping, on="unique_id", how="left", validate="many_to_one"
    )
    result = result.rename(columns={"ds": "week", value_columns[0]: "quantity"})
    result["quantity"] = result["quantity"].fillna(0).clip(lower=0)
    return result[KEYS + ["quantity"]].sort_values(KEYS).reset_index(drop=True)


def make_model(
    args: argparse.Namespace,
    static_columns: list[str],
    device: str,
    hist_exog_list: list[str] | None = None,
) -> TFT:
    selected_hist_exog = hist_exog_list or BASE_HIST_EXOG
    return TFT(
        h=HORIZON,
        input_size=args.input_size,
        stat_exog_list=static_columns,
        hist_exog_list=selected_hist_exog,
        futr_exog_list=FUTR_EXOG,
        hidden_size=args.hidden_size,
        n_head=4,
        dropout=0.1,
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=args.patience,
        batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        windows_batch_size=args.windows_batch_size,
        inference_windows_batch_size=args.windows_batch_size,
        scaler_type="robust",
        random_seed=args.seed,
        alias="TFT",
        accelerator=device,
        devices=1,
        precision=getattr(args, "precision", None) or ("16-mixed" if device == "gpu" else "32-true"),
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )


def save_report(output_dir: Path, results: pd.DataFrame, config: dict[str, object]) -> None:
    complete = results.sort_values("fold")
    means = complete[METRICS + ["normalized_score", "run_seconds"]].mean()
    rows = "\n".join(
        f"| {int(row.fold)} | {row.validation_start} | {row.validation_end} | "
        f"{row.smape:.4f} | {row.wape:.4f} | {row.rmse:.4f} | {row.bias:.4f} | "
        f"{row.rmsle:.4f} | {row.normalized_score:.4f} | {row.run_seconds:.1f} |"
        for row in complete.itertuples()
    )
    report = f"""# Первый недельный TFT

TFT обучается как глобальная модель сразу на всех рядах «склад–товар» и выдаёт
9 недель прогноза одним проходом. Календарь известен на всём горизонте, сетевой
контекст используется только из прошлого. Отрицательные выходы обрезаются до 0.

## Настройки

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

## Результаты

| fold | начало | конец | SMAPE | WAPE | RMSE | Bias | RMSLE | score | секунды |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

Средний normalized score: **{means['normalized_score']:.4f}**. Для сравнения,
зафиксированный CatBoost `base_calendar_network` имеет **0.7433** на этих же
трёх окнах. Меньше — лучше.
"""
    (output_dir / "TFT_REPORT.md").write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"fold {int(value)}" for value in complete["fold"]] + ["среднее"]
    scores = complete["normalized_score"].tolist() + [float(means["normalized_score"])]
    colors = ["#4C78A8"] * len(complete) + ["#F58518"]
    ax.bar(labels, scores, color=colors)
    ax.axhline(0.7433, color="#2F855A", linestyle="--", label="CatBoost 0.7433")
    ax.axhline(1.0, color="#C53030", linestyle=":", label="mean13 = 1.0")
    for position, value in enumerate(scores):
        ax.text(position, value + 0.015, f"{value:.4f}", ha="center")
    ax.set_ylabel("Normalized score (меньше — лучше)")
    ax.set_title("TFT: rolling validation")
    ax.set_ylim(0, max(1.08, max(scores) + 0.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tft_rolling_scores.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "tft_rolling_results.csv"
    if result_path.exists() and not args.restart:
        results = pd.read_csv(result_path)
    else:
        results = pd.DataFrame()

    raw = pd.read_csv(args.data, parse_dates=["week"])
    if args.series_limit:
        series = (
            raw[["warehouse_name", "article"]]
            .drop_duplicates()
            .sort_values(["warehouse_name", "article"])
            .head(args.series_limit)
        )
        raw = raw.merge(series, on=["warehouse_name", "article"], how="inner")
    temporal, static_df, static_columns = prepare_frames(raw)
    weeks = pd.Index(sorted(raw["week"].unique()))
    latest_origin = len(weeks) - HORIZON
    origins = [latest_origin - fold * args.fold_step for fold in range(args.folds)]
    device = "gpu" if torch.cuda.is_available() else "cpu"

    config: dict[str, object] = {
        "horizon": HORIZON,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "windows_batch_size": args.windows_batch_size,
        "learning_rate": args.learning_rate,
        "folds": args.folds,
        "series": int(temporal["unique_id"].nunique()),
        "static_features": len(static_columns),
        "historic_features": HIST_EXOG,
        "future_features": FUTR_EXOG,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output_dir / "tft_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False))

    for fold, origin in enumerate(origins):
        if not results.empty and fold in set(results["fold"].astype(int)):
            print(f"skip fold={fold}")
            continue
        validation_weeks = weeks[origin : origin + HORIZON]
        cutoff = pd.Timestamp(validation_weeks[0])
        train_df = temporal[temporal["ds"] < cutoff].copy()
        future_df = temporal[temporal["ds"].isin(validation_weeks)][
            ["unique_id", "ds", *FUTR_EXOG]
        ].copy()
        truth = truth_frame(raw, validation_weeks)
        baseline = baseline_mean13(raw, cutoff, validation_weeks)
        baseline_metrics = compute_all_metrics_fast(truth, baseline)

        model = make_model(args, static_columns, device)
        nf = NeuralForecast(models=[model], freq="W-MON")
        started = time.perf_counter()
        nf.fit(df=train_df, static_df=static_df, val_size=HORIZON)
        prediction = prediction_to_submission(nf.predict(futr_df=future_df), raw)
        elapsed = time.perf_counter() - started
        metrics = compute_all_metrics_fast(truth, prediction)
        row: dict[str, object] = {
            "fold": fold,
            "validation_start": str(pd.Timestamp(validation_weeks[0]).date()),
            "validation_end": str(pd.Timestamp(validation_weeks[-1]).date()),
            **metrics,
            "normalized_score": normalized_score(metrics, baseline_metrics),
            "run_seconds": elapsed,
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(result_path, index=False)
        prediction.to_csv(args.output_dir / f"tft_predictions_fold_{fold}.csv.gz", index=False)
        save_report(args.output_dir, results, config)
        print(json.dumps(row, ensure_ascii=False))
        del nf, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_report(args.output_dir, results, config)


if __name__ == "__main__":
    main()
