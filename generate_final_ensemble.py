"""Генерация финального прогноза ансамбля (TFT 0.5 + CatBoost 0.5) на тестовые 9 недель.

Период: 2026-07-27 .. 2026-09-21 (9 недель).
Число строк: 79 101.
Формат: week, warehouse_name, article, quantity.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sber_hak.run_weekly_feature_experiments import (
    HORIZON,
    WeeklyMatrices,
    columns_for_experiment,
    train_and_predict,
)
from sber_hak.score import validate_submission


def main() -> None:
    print("1. Загрузка данных и подготовка признаков...")
    data = pd.read_csv("weekly_data/weekly_dataset.csv.gz", parse_dates=["week"])
    first_week = data[data["week"] == data["week"].min()].sort_values(["warehouse_name", "article"])
    pairs_df = first_week[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()

    future_weeks = pd.date_range("2026-07-27", periods=HORIZON, freq="W-MON")

    future_rows = []
    for w in future_weeks:
        chunk = pairs_df.copy()
        chunk["week"] = w
        chunk["quantity"] = 0.0
        chunk["sales_sum"] = 0.0
        chunk["discount_sum"] = 0.0
        chunk["stock_week_end"] = np.nan
        chunk["stock_mean"] = np.nan
        chunk["stock_nonpositive_rate"] = np.nan
        chunk["stock_observed_days"] = 0.0
        future_rows.append(chunk)

    future_df = pd.concat(future_rows, ignore_index=True)
    extended_data = pd.concat([data, future_df], ignore_index=True)

    matrices = WeeklyMatrices(extended_data)
    origin = len(data["week"].unique())  # 134 (первая прогнозная неделя - 134)

    # Обучаем CatBoost на последних 18 точках истории, предшествующих тесту
    latest_train_origin = origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - 17)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)
    print(f"Обучающие ориджины CatBoost: {train_origins}")

    columns = columns_for_experiment("base_calendar_network")
    train_df, target_arr, _ = matrices.make_rows(train_origins, 350_000, seed=42)
    test_df, _, _ = matrices.make_rows(np.array([origin]), None)

    print("2. Обучение CatBoost и построение прогноза на 9 недель...")
    cb_prediction_values, _ = train_and_predict(
        train_df, target_arr, test_df, columns, iterations=160
    )

    test_weeks = np.repeat(matrices.weeks[origin : origin + HORIZON].to_numpy(), matrices.n_pairs)
    test_pairs = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)

    cb_pred = pd.DataFrame({
        "week": pd.to_datetime(test_weeks).strftime("%Y-%m-%d"),
        "warehouse_name": test_pairs["warehouse_name"],
        "article": test_pairs["article"],
        "quantity_cb": np.maximum(cb_prediction_values, 0),
    })

    print("3. Загрузка прогноза TFT...")
    tft_path = Path("experiments/tft/tft_final_submission.csv")
    if not tft_path.exists():
        tft_path = Path("sber_hak/experiments/tft/tft_final_submission.csv")
    tft_pred = pd.read_csv(tft_path)
    tft_pred["week"] = pd.to_datetime(tft_pred["week"]).dt.strftime("%Y-%m-%d")
    tft_pred = tft_pred.rename(columns={"quantity": "quantity_tft"})

    print("4. Объединение в ансамбль (0.5 TFT + 0.5 CatBoost) и округление...")
    keys = ["week", "warehouse_name", "article"]
    merged = tft_pred.merge(cb_pred, on=keys, how="inner")

    if len(merged) != 79101:
        raise ValueError(f"Ожидалось 79101 строк, получено {len(merged)}")

    ensemble_qty = 0.5 * merged["quantity_tft"].values + 0.5 * merged["quantity_cb"].values
    merged["quantity"] = np.rint(ensemble_qty).clip(0, None).astype(int)

    final_sub = merged[keys + ["quantity"]]
    out_file = Path("final_submission.csv")
    final_sub.to_csv(out_file, index=False)
    print(f"Файл {out_file} сохранен, строк: {len(final_sub)}")

    print("5. Валидация формата сабмита...")
    warnings = validate_submission(final_sub, "final_submission")
    if warnings:
        print("Предупреждения валидации:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print(" Успешно! Файл полностью соответствует ТЗ (0 предупреждений).")

    print("\nПример первых строк final_submission.csv:")
    print(final_sub.head())


if __name__ == "__main__":
    main()
