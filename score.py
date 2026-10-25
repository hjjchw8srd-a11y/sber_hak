#!/usr/bin/env python3
"""
Автоматический скоринг для хакатона «Прогноз продаж по складам и товарам».

Метрики и веса:
    SMAPE  — 50%   (считается по сериям, усредняется)
    WAPE   — 20%   (глобально)
    RMSE   — 15%   (глобально)
    Bias   — 10%   (|MPE| по сериям, усредняется)
    RMSLE  — 5%    (глобально, log1p)

Чем ниже итоговый скор — тем лучше.

Режимы нормализации:
    minmax — 0 = идеальный прогноз, 1 = наивный бейзлайн
    rank   — ранги участников от 0 (лучший) до 1 (худший)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape":  0.20,
    "rmse":  0.15,
    "bias":  0.10,
    "rmsle": 0.05,
}

REQUIRED_COLUMNS = ["week", "warehouse_name", "article", "quantity"]
NUM_WAREHOUSES = 11
NUM_ARTICLES = 799
FORECAST_HORIZON = 9
EXPECTED_ROWS = NUM_WAREHOUSES * NUM_ARTICLES * FORECAST_HORIZON  # 79 101
FORECAST_START = pd.Timestamp("2026-07-27")


def smape_series(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE для одной серии. Возвращает значение в процентах (0..200)."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    # Актуальный FAQ требует учитывать 0/0 как точку с ошибкой 0.
    return float(np.mean(values) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, series_ids: np.ndarray) -> float:
    """SMAPE по сериям (склад × товар), усреднённый."""
    unique = np.unique(series_ids)
    values = []
    for sid in unique:
        mask = series_ids == sid
        values.append(smape_series(y_true[mask], y_pred[mask]))
    return float(np.mean(values))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE — глобально. Возвращает значение в процентах."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — глобально."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias_abs_series(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """|Bias| для одной серии. None если mean(y_true) == 0."""
    mean_y = np.mean(y_true)
    if mean_y == 0:
        return None
    return float(abs(np.mean(y_pred) - mean_y) / mean_y * 100.0)


def bias_abs(y_true: np.ndarray, y_pred: np.ndarray, series_ids: np.ndarray) -> float:
    """|Bias| по сериям, усреднённый. Серии с mean(y)=0 исключаются."""
    unique = np.unique(series_ids)
    values = []
    for sid in unique:
        mask = series_ids == sid
        val = bias_abs_series(y_true[mask], y_pred[mask])
        if val is not None:
            values.append(val)
    if not values:
        return 0.0
    return float(np.mean(values))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE — глобально, с log1p для безопасности."""
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def compute_all_metrics(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame
) -> Dict[str, float]:
    """Считает все 5 метрик для пары (truth, prediction)."""
    df = df_true.merge(
        df_pred, on=["week", "warehouse_name", "article"], how="left",
        suffixes=("_true", "_pred")
    )
    nan_count = df["quantity_pred"].isna().sum()
    if nan_count > 0:
        print(f"  ⚠  Пропуски в submission: {nan_count} строк. Заполнены нулями.")
    df["quantity_pred"] = df["quantity_pred"].fillna(0).clip(lower=0)

    y_true = df["quantity_true"].values.astype(float)
    y_pred = df["quantity_pred"].values.astype(float)
    series_ids = (
        df["warehouse_name"].astype(str) + "|" + df["article"].astype(str)
    ).values

    return {
        "smape": smape(y_true, y_pred, series_ids),
        "wape":  wape(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "bias":  bias_abs(y_true, y_pred, series_ids),
        "rmsle": rmsle(y_true, y_pred),
    }


def validate_submission(df: pd.DataFrame, name: str = "submission") -> List[str]:
    """Проверяет формат submission и возвращает список предупреждений."""
    warnings_list = []

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        warnings_list.append(f"Отсутствуют колонки: {missing_cols}")
        return warnings_list

    duplicate_count = df.duplicated(["week", "warehouse_name", "article"]).sum()
    if duplicate_count > 0:
        warnings_list.append(f"Дубли ключа week × склад × товар: {duplicate_count}")

    nan_count = df["quantity"].isna().sum()
    if nan_count > 0:
        warnings_list.append(f"NaN в quantity: {nan_count} строк")

    neg_count = (df["quantity"] < 0).sum()
    if neg_count > 0:
        warnings_list.append(f"Отрицательные значения в quantity: {neg_count} строк")

    if len(df) != EXPECTED_ROWS:
        warnings_list.append(f"Ожидается {EXPECTED_ROWS} строк, получено {len(df)}")

    weeks = pd.to_datetime(df["week"], errors="coerce")
    invalid_week_count = int(weeks.isna().sum())
    if invalid_week_count:
        warnings_list.append(f"Некорректные значения week: {invalid_week_count}")
    unique_weeks = weeks.dropna().unique()
    if len(unique_weeks) != FORECAST_HORIZON:
        warnings_list.append(
            f"Ожидается {FORECAST_HORIZON} уникальных недель, получено {len(unique_weeks)}"
        )

    if len(unique_weeks) == FORECAST_HORIZON:
        sorted_weeks = pd.to_datetime(np.sort(unique_weeks))
        if (sorted_weeks.dayofweek != 0).any():
            warnings_list.append("Поле week должно содержать понедельники")
        if sorted_weeks.min() != FORECAST_START:
            warnings_list.append(
                f"Первая неделя должна быть {FORECAST_START.date()}, "
                f"получено {sorted_weeks.min().date()}"
            )
        gaps = sorted_weeks.to_series().diff().dropna()
        if (gaps != pd.Timedelta(days=7)).any():
            warnings_list.append("Недели не идут подряд с шагом 7 дней")

    combos = df.groupby(["warehouse_name", "article"]).ngroups
    expected_combos = NUM_WAREHOUSES * NUM_ARTICLES
    if combos != expected_combos:
        warnings_list.append(
            f"Ожидается {expected_combos} уникальных комбинаций склад × товар, получено {combos}"
        )

    return warnings_list


def normalized_score(
    metrics: Dict[str, float], baseline_metrics: Dict[str, float]
) -> float:
    """Взвешенный minmax-score: 0 — идеально, 1 — не лучше бейзлайна."""
    score = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        baseline_value = baseline_metrics[metric]
        ratio = metrics[metric] / baseline_value if baseline_value > 0 else 0.0
        score += weight * min(max(ratio, 0.0), 1.0)
    return float(score)


def main() -> None:
    # Векторизованная реализация даёт те же значения, но не перебирает
    # 8 789 временных рядов Python-циклом.
    from fast_score import compute_all_metrics_fast

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--submissions", required=True, nargs="+", type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    baseline = pd.read_csv(args.baseline)
    baseline_metrics = compute_all_metrics_fast(truth, baseline)

    rows = []
    for submission_path in args.submissions:
        submission = pd.read_csv(submission_path)
        for warning in validate_submission(submission, submission_path.name):
            print(f"{submission_path.name}: {warning}")
        metrics = compute_all_metrics_fast(truth, submission)
        rows.append({
            "model": submission_path.stem,
            **metrics,
            "score_vs_baseline": normalized_score(metrics, baseline_metrics),
        })

    leaderboard = pd.DataFrame(rows).sort_values("score_vs_baseline")
    print(leaderboard.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(args.output, index=False)
        print(f"\nСохранено: {args.output}")


if __name__ == "__main__":
    main()
