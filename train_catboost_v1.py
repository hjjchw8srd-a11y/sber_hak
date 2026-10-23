"""Первая воспроизводимая CatBoost-модель для чек-пойнта.

Валидация: последние 64 календарных дня.
Модель двухэтапная:
1) классификатор вероятности ненулевой продажи;
2) регрессор log1p(quantity) при условии quantity > 0.

Все лаги и скользящие статистики сдвинуты минимум на один день, поэтому
таргет текущего/будущего дня в признаки не попадает. Прогноз на 64 дня
строится рекурсивно. Stock в v1 намеренно не используется: будущие остатки
неизвестны, а их сырые строки требуют отдельной корректной агрегации.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from fast_score import compute_all_metrics_fast


HORIZON = 64
MAX_LAG = 56
CAT_COLUMNS = ["warehouse", "article", "dow", "cat1", "cat2", "cat3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_dataset_hakaton_train.csv")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--zero-keep-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="catboost_v1")
    return parser.parse_args()


def category_codes(sales: pd.DataFrame, articles: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    article_meta = sales.drop_duplicates("article", keep="last").set_index("article")
    for source, output in [
        ("category_lvl_1", "cat1"),
        ("category_lvl_2", "cat2"),
        ("category_lvl_3", "cat3"),
    ]:
        values = article_meta[source].reindex(articles).fillna("unknown").astype(str)
        result[output] = pd.factorize(values, sort=True)[0].astype(np.int32)
    return result


def make_features(
    matrix: np.ndarray,
    date_positions: np.ndarray,
    series_positions: np.ndarray,
    dates: pd.DatetimeIndex,
    warehouse_ids: np.ndarray,
    article_ids: np.ndarray,
    category_ids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Создаёт признаки для произвольных пар (позиция даты, позиция серии)."""
    selected_dates = dates[date_positions]
    data: dict[str, np.ndarray] = {
        "warehouse": warehouse_ids[series_positions],
        "article": article_ids[series_positions],
        "dow": selected_dates.dayofweek.astype(np.int32),
        "month": selected_dates.month.astype(np.int32),
        "day": selected_dates.day.astype(np.int32),
        "dayofyear_sin": np.sin(2 * np.pi * selected_dates.dayofyear / 365.25).astype(
            np.float32
        ),
        "dayofyear_cos": np.cos(2 * np.pi * selected_dates.dayofyear / 365.25).astype(
            np.float32
        ),
        "time_idx": (date_positions / len(dates)).astype(np.float32),
    }
    for name, values in category_ids.items():
        data[name] = values[series_positions]

    for lag in (1, 7, 14, 28, 56):
        data[f"lag_{lag}"] = matrix[date_positions - lag, series_positions]

    # На выборке порядка миллиона строк прямое индексирование нескольких
    # прошлых дней экономнее, чем материализация полной таблицы rolling.
    for window in (7, 28, 56):
        offsets = np.arange(1, window + 1, dtype=np.int32)
        history = matrix[
            date_positions[:, None] - offsets[None, :], series_positions[:, None]
        ]
        data[f"mean_{window}"] = history.mean(axis=1, dtype=np.float32)
        data[f"nonzero_rate_{window}"] = (history > 0).mean(axis=1).astype(np.float32)
        if window in (28, 56):
            data[f"max_{window}"] = history.max(axis=1)

    return pd.DataFrame(data)


def correct_sampled_probability(prob_sample: np.ndarray, negative_rate: float) -> np.ndarray:
    """Возвращает вероятность с поправкой на downsampling отрицательного класса."""
    numerator = negative_rate * prob_sample
    denominator = 1.0 - prob_sample + numerator
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(prob_sample),
        where=denominator > 0,
    )


def forecast_recursive(
    classifier: CatBoostClassifier,
    regressor: CatBoostRegressor,
    history: np.ndarray,
    all_dates: pd.DatetimeIndex,
    cutoff_position: int,
    warehouse_ids: np.ndarray,
    article_ids: np.ndarray,
    category_ids: dict[str, np.ndarray],
    negative_rate: float,
    gate: float,
    prediction_cap: float,
) -> np.ndarray:
    dynamic = history.copy()
    n_series = dynamic.shape[1]
    series_positions = np.arange(n_series, dtype=np.int32)
    result = np.zeros((HORIZON, n_series), dtype=np.float32)

    for step in range(HORIZON):
        date_position = cutoff_position + 1 + step
        date_positions = np.full(n_series, date_position, dtype=np.int32)
        features = make_features(
            dynamic,
            date_positions,
            series_positions,
            all_dates,
            warehouse_ids,
            article_ids,
            category_ids,
        )
        prob_sample = classifier.predict_proba(features)[:, 1].astype(np.float32)
        probability = correct_sampled_probability(prob_sample, negative_rate)
        conditional = np.expm1(regressor.predict(features)).astype(np.float32)
        conditional = np.clip(conditional, 0, prediction_cap)
        prediction = np.where(probability >= gate, conditional, 0).astype(np.float32)
        dynamic[date_position] = prediction
        result[step] = prediction
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "date",
        "warehouse_name",
        "article",
        "category_lvl_1",
        "category_lvl_2",
        "category_lvl_3",
        "quantity",
    ]
    sales = pd.read_csv(args.sales, usecols=usecols, parse_dates=["date"])
    daily = sales.groupby(
        ["date", "warehouse_name", "article"], as_index=False, sort=False
    )["quantity"].sum()

    warehouses = sorted(sales["warehouse_name"].unique())
    articles = sorted(sales["article"].unique())
    pairs = pd.MultiIndex.from_product(
        [warehouses, articles], names=["warehouse_name", "article"]
    )
    observed_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    future_dates = pd.date_range(
        observed_dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D"
    )
    all_dates = observed_dates.append(future_dates)

    pivot = daily.pivot(
        index="date", columns=["warehouse_name", "article"], values="quantity"
    ).reindex(index=observed_dates, columns=pairs, fill_value=0)
    pivot = pivot.fillna(0)
    observed_matrix = pivot.to_numpy(dtype=np.float32)
    matrix = np.zeros((len(all_dates), len(pairs)), dtype=np.float32)
    matrix[: len(observed_dates)] = observed_matrix

    warehouse_lookup = {value: i for i, value in enumerate(warehouses)}
    article_lookup = {value: i for i, value in enumerate(articles)}
    warehouse_ids = np.array(
        [warehouse_lookup[w] for w, _ in pairs], dtype=np.int32
    )
    article_ids = np.array([article_lookup[a] for _, a in pairs], dtype=np.int32)
    article_categories = category_codes(sales, articles)
    category_ids = {
        name: values[article_ids] for name, values in article_categories.items()
    }

    validation_start = len(observed_dates) - HORIZON
    cutoff_position = validation_start - 1
    first_train = max(MAX_LAG, validation_start - args.history_days)
    train_dates = np.arange(first_train, validation_start, dtype=np.int32)

    target_block = matrix[train_dates].reshape(-1)
    rng = np.random.default_rng(args.seed)
    keep = (target_block > 0) | (rng.random(len(target_block)) < args.zero_keep_rate)
    kept_rows = np.flatnonzero(keep)
    date_positions = train_dates[kept_rows // len(pairs)]
    series_positions = (kept_rows % len(pairs)).astype(np.int32)
    target = target_block[kept_rows]

    print(
        f"Train: {all_dates[first_train].date()} — "
        f"{all_dates[validation_start - 1].date()}, строк после sampling: {len(target):,}"
    )
    features = make_features(
        matrix,
        date_positions,
        series_positions,
        all_dates,
        warehouse_ids,
        article_ids,
        category_ids,
    )

    common = dict(
        iterations=args.iterations,
        depth=7,
        learning_rate=0.08,
        random_seed=args.seed,
        verbose=50,
        allow_writing_files=False,
        thread_count=-1,
    )
    classifier = CatBoostClassifier(loss_function="Logloss", **common)
    classifier.fit(features, (target > 0).astype(np.int8), cat_features=CAT_COLUMNS)

    positive = target > 0
    regressor = CatBoostRegressor(loss_function="RMSE", **common)
    regressor.fit(
        features.loc[positive],
        np.log1p(target[positive]),
        cat_features=CAT_COLUMNS,
    )

    classifier.save_model(output_dir / "classifier.cbm")
    regressor.save_model(output_dir / "regressor.cbm")

    validation_dates = observed_dates[validation_start:]
    truth = pd.DataFrame(
        {
            "date": np.repeat(validation_dates.values, len(pairs)),
            "warehouse_name": np.tile(pairs.get_level_values(0), HORIZON),
            "article": np.tile(pairs.get_level_values(1), HORIZON),
            "quantity": matrix[validation_start : validation_start + HORIZON].reshape(-1),
        }
    )
    prediction_cap = float(np.quantile(target[target > 0], 0.999))
    rows = []
    best: tuple[float, pd.DataFrame] | None = None
    # Более высокий gate обычно нужен из-за 50%-го веса SMAPE и редкого спроса.
    for gate in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        prediction = forecast_recursive(
            classifier,
            regressor,
            matrix.copy(),
            all_dates,
            cutoff_position,
            warehouse_ids,
            article_ids,
            category_ids,
            args.zero_keep_rate,
            gate,
            prediction_cap,
        )
        submission = truth[["date", "warehouse_name", "article"]].copy()
        submission["quantity"] = prediction.reshape(-1)
        metrics = compute_all_metrics_fast(truth, submission)
        rows.append({"gate": gate, **metrics})
        # Локальный proxy без неизвестной нормализации лидерборда.
        proxy = (
            0.50 * metrics["smape"] / 200
            + 0.20 * metrics["wape"] / 100
            + 0.15 * metrics["rmse"]
            + 0.10 * metrics["bias"] / 100
            + 0.05 * metrics["rmsle"]
        )
        if best is None or proxy < best[0]:
            best = (proxy, submission)
            best_gate = gate

    metrics_df = pd.DataFrame(rows).sort_values("gate")
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    assert best is not None
    best[1].to_csv(output_dir / "validation_forecast.csv", index=False)

    config = {
        "validation_start": str(validation_dates[0].date()),
        "validation_end": str(validation_dates[-1].date()),
        "history_days": args.history_days,
        "iterations": args.iterations,
        "zero_keep_rate": args.zero_keep_rate,
        "best_gate": best_gate,
        "prediction_cap": prediction_cap,
        "features": features.columns.tolist(),
        "stock_used": False,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nМетрики по порогам:")
    print(metrics_df.to_string(index=False))
    print(f"\nЛучший gate по локальному proxy: {best_gate}")
    print(f"Результаты: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
