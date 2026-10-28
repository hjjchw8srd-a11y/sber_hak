"""Сравнение групп признаков на честном недельном holdout из 9 недель.

Модель строит direct multi-horizon прогноз: для каждой точки происхождения
используются только недели, завершившиеся до неё, а горизонт 1..9 является
отдельным признаком. Поэтому фактические значения внутри прогнозного окна не
попадают в лаги следующих недель.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from fast_score import compute_all_metrics_fast
from score import normalized_score


HORIZON = 9
KEYS = ["week", "warehouse_name", "article"]
CAT_COLUMNS = [
    "warehouse_name",
    "article",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
]

EXPERIMENTS: dict[str, dict[str, object]] = {
    "base": {
        "groups": [],
        "description": "Категории, горизонт, тренд и история продаж",
    },
    "base_stock": {
        "groups": ["stock"],
        "description": "Base + остатки предыдущей известной недели",
    },
    "base_calendar": {
        "groups": ["calendar"],
        "description": "Base + календарь прогнозируемой недели",
    },
    "base_money_raw": {
        "groups": ["money_raw"],
        "description": "Base + исходные sales_sum и discount_sum (lag 1)",
    },
    "base_money_derived": {
        "groups": ["money_derived"],
        "description": "Base + доля скидки и суммы на единицу (lag 1)",
    },
    "base_stock_calendar": {
        "groups": ["stock", "calendar"],
        "description": "Base + остатки + календарь, без денежных признаков",
    },
    "base_category_context": {
        "groups": ["category_context"],
        "description": "Base + положение товара внутри категории на складе",
    },
    "base_network_context": {
        "groups": ["network_context"],
        "description": "Base + положение товара среди складов всей сети",
    },
    "base_calendar_category": {
        "groups": ["calendar", "category_context"],
        "description": "Base + календарь + контекст категории",
    },
    "base_calendar_network": {
        "groups": ["calendar", "network_context"],
        "description": "Base + календарь + контекст сети",
    },
    "base_calendar_all_context": {
        "groups": ["calendar", "category_context", "network_context"],
        "description": "Base + календарь + контекст категории и сети",
    },
    "base_dynamics": {
        "groups": ["demand_dynamics"],
        "description": "Base + форма и направление временного ряда",
    },
    "base_calendar_dynamics": {
        "groups": ["calendar", "demand_dynamics"],
        "description": "Base + календарь + динамика спроса",
    },
    "base_calendar_network_dynamics": {
        "groups": ["calendar", "network_context", "demand_dynamics"],
        "description": "Текущий лидер + динамика спроса",
    },
    "all_raw": {
        "groups": ["stock", "calendar", "money_raw"],
        "description": "Base + остатки + календарь + исходные денежные признаки",
    },
    "all_derived": {
        "groups": ["stock", "calendar", "money_derived"],
        "description": "Base + остатки + календарь + производные денежные признаки",
    },
    "all_both": {
        "groups": ["stock", "calendar", "money_raw", "money_derived"],
        "description": "Все группы вместе: проверка совместного эффекта",
    },
}

BASE_COLUMNS = [
    *CAT_COLUMNS,
    "horizon",
    "time_idx",
    "quantity_lag_1",
    "quantity_lag_2",
    "quantity_lag_4",
    "quantity_lag_13",
    "quantity_mean_4",
    "quantity_mean_13",
    "quantity_std_13",
    "quantity_nonzero_rate_13",
]
GROUP_COLUMNS = {
    "stock": [
        "stock_end_lag_1",
        "stock_mean_lag_1",
        "stock_nonpositive_rate_lag_1",
        "stock_observed_days_lag_1",
        "stock_missing_lag_1",
    ],
    "calendar": [
        "month",
        "quarter",
        "week_of_year",
        "week_sin",
        "week_cos",
    ],
    "money_raw": ["sales_sum_log_lag_1", "discount_sum_log_lag_1"],
    "money_derived": [
        "discount_share_lag_1",
        "paid_value_per_unit_lag_1",
        "full_value_per_unit_lag_1",
    ],
    "category_context": [
        "category_sales_mean_13_log",
        "category_growth_4_vs_13",
        "article_share_category_13",
        "article_rank_category_13",
        "active_articles_category_13",
        "discount_share_vs_category_lag_1",
    ],
    "network_context": [
        "article_network_sales_mean_13_log",
        "article_network_growth_4_vs_13",
        "warehouse_share_article_13",
        "active_warehouses_article_13",
    ],
    "demand_dynamics": [
        "quantity_min_13",
        "quantity_max_13",
        "quantity_median_13",
        "quantity_cv_13",
        "quantity_mean_26",
        "growth_log_4_vs_previous_4",
        "growth_log_13_vs_previous_13",
        "quantity_slope_13",
        "quantity_slope_26",
        "weeks_since_last_sale",
        "current_zero_streak",
        "autocorrelation_lag_1",
        "autocorrelation_lag_4",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--train-origins", type=int, default=18)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    return parser.parse_args()


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt(
        (left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )


class WeeklyMatrices:
    def __init__(self, data: pd.DataFrame) -> None:
        data = data.sort_values(KEYS).reset_index(drop=True)
        self.weeks = pd.DatetimeIndex(sorted(data["week"].unique()))
        first = data[data["week"] == self.weeks[0]].sort_values(KEYS[1:])
        self.pairs = first[["warehouse_name", "article"]].reset_index(drop=True)
        self.n_pairs = len(self.pairs)
        expected = len(self.weeks) * self.n_pairs
        if len(data) != expected:
            raise ValueError(f"Недельная сетка неполная: {len(data):,} строк вместо {expected:,}")

        ordered = data.sort_values(["week", "warehouse_name", "article"]).reset_index(drop=True)
        self.static = first[
            ["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]
        ].fillna("unknown").astype(str).reset_index(drop=True)
        self.warehouse_codes = pd.factorize(
            self.static["warehouse_name"], sort=True
        )[0].astype(np.int32)
        self.article_codes = pd.factorize(
            self.static["article"], sort=True
        )[0].astype(np.int32)
        self.category_codes = pd.factorize(
            self.static["category_lvl_2"], sort=True
        )[0].astype(np.int32)
        warehouse_category = pd.MultiIndex.from_arrays(
            [self.warehouse_codes, self.category_codes]
        )
        self.warehouse_category_codes = pd.factorize(
            warehouse_category, sort=True
        )[0].astype(np.int32)
        self.n_warehouse_categories = int(self.warehouse_category_codes.max() + 1)
        self.n_articles = int(self.article_codes.max() + 1)

        def matrix(column: str) -> np.ndarray:
            return ordered[column].to_numpy(dtype=np.float32).reshape(len(self.weeks), self.n_pairs)

        self.quantity = matrix("quantity")
        self.sales_sum = matrix("sales_sum")
        self.discount_sum = matrix("discount_sum")
        self.stock_end = matrix("stock_week_end")
        self.stock_mean = matrix("stock_mean")
        self.stock_nonpositive_rate = matrix("stock_nonpositive_rate")
        self.stock_observed_days = matrix("stock_observed_days")

    def make_rows(
        self,
        origins: np.ndarray,
        max_rows: int | None,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        total = len(origins) * HORIZON * self.n_pairs
        if max_rows is not None and total > max_rows:
            rng = np.random.default_rng(seed)
            flat = np.sort(rng.choice(total, size=max_rows, replace=False))
        else:
            flat = np.arange(total, dtype=np.int64)

        block = HORIZON * self.n_pairs
        origin_idx = flat // block
        remainder = flat % block
        horizon_idx = remainder // self.n_pairs
        pair_idx = remainder % self.n_pairs
        origin_pos = origins[origin_idx]
        target_pos = origin_pos + horizon_idx
        history_pos = origin_pos - 1

        history26 = np.column_stack(
            [self.quantity[origin_pos - lag, pair_idx] for lag in range(1, 27)]
        )
        history13 = history26[:, :13]
        history4 = history26[:, :4]
        sales = self.sales_sum[history_pos, pair_idx]
        discount = self.discount_sum[history_pos, pair_idx]
        quantity_lag1 = self.quantity[history_pos, pair_idx]

        # sales_sum — сумма продаж, discount_sum — сумма скидки. Поэтому доля
        # скидки считается от суммы до скидки: sales + discount. Денежные
        # признаки относятся только к последней известной неделе.
        full_value = sales + discount
        discount_share = np.divide(
            discount,
            full_value,
            out=np.full_like(discount, np.nan),
            where=full_value > 0,
        )
        paid_per_unit = np.divide(
            sales,
            quantity_lag1,
            out=np.full_like(sales, np.nan),
            where=quantity_lag1 > 0,
        )
        full_per_unit = np.divide(
            full_value,
            quantity_lag1,
            out=np.full_like(full_value, np.nan),
            where=quantity_lag1 > 0,
        )

        target_weeks = self.weeks[target_pos]
        week_number = target_weeks.isocalendar().week.to_numpy(dtype=np.float32)
        static = self.static.iloc[pair_idx].reset_index(drop=True)
        frame = static.copy()
        frame["horizon"] = horizon_idx.astype(np.int16) + 1
        frame["time_idx"] = target_pos.astype(np.int16)
        frame["quantity_lag_1"] = quantity_lag1
        frame["quantity_lag_2"] = self.quantity[origin_pos - 2, pair_idx]
        frame["quantity_lag_4"] = self.quantity[origin_pos - 4, pair_idx]
        frame["quantity_lag_13"] = self.quantity[origin_pos - 13, pair_idx]
        frame["quantity_mean_4"] = history4.mean(axis=1)
        frame["quantity_mean_13"] = history13.mean(axis=1)
        frame["quantity_std_13"] = history13.std(axis=1)
        frame["quantity_nonzero_rate_13"] = (history13 > 0).mean(axis=1)

        mean13 = history13.mean(axis=1)
        std13 = history13.std(axis=1)
        frame["quantity_min_13"] = history13.min(axis=1)
        frame["quantity_max_13"] = history13.max(axis=1)
        frame["quantity_median_13"] = np.median(history13, axis=1)
        frame["quantity_cv_13"] = np.divide(
            std13,
            mean13,
            out=np.zeros_like(std13),
            where=mean13 > 0,
        )
        frame["quantity_mean_26"] = history26.mean(axis=1)
        frame["growth_log_4_vs_previous_4"] = np.log1p(
            history26[:, :4].sum(axis=1)
        ) - np.log1p(history26[:, 4:8].sum(axis=1))
        frame["growth_log_13_vs_previous_13"] = np.log1p(
            history26[:, :13].sum(axis=1)
        ) - np.log1p(history26[:, 13:26].sum(axis=1))

        chronological13 = history13[:, ::-1]
        chronological26 = history26[:, ::-1]
        x13 = np.arange(13, dtype=np.float32) - 6.0
        x26 = np.arange(26, dtype=np.float32) - 12.5
        frame["quantity_slope_13"] = (chronological13 @ x13) / (x13 @ x13)
        frame["quantity_slope_26"] = (chronological26 @ x26) / (x26 @ x26)

        nonzero_lag_order = history26 > 0
        has_previous_sale = nonzero_lag_order.any(axis=1)
        first_nonzero = np.argmax(nonzero_lag_order, axis=1)
        frame["weeks_since_last_sale"] = np.where(
            has_previous_sale, first_nonzero + 1, 27
        )
        frame["current_zero_streak"] = np.where(
            has_previous_sale, first_nonzero, 26
        )
        frame["autocorrelation_lag_1"] = rowwise_correlation(
            chronological13[:, 1:], chronological13[:, :-1]
        )
        frame["autocorrelation_lag_4"] = rowwise_correlation(
            chronological26[:, 4:], chronological26[:, :-4]
        )

        stock_end = self.stock_end[history_pos, pair_idx]
        frame["stock_end_lag_1"] = signed_log1p(stock_end)
        frame["stock_mean_lag_1"] = signed_log1p(self.stock_mean[history_pos, pair_idx])
        frame["stock_nonpositive_rate_lag_1"] = self.stock_nonpositive_rate[history_pos, pair_idx]
        frame["stock_observed_days_lag_1"] = self.stock_observed_days[history_pos, pair_idx]
        frame["stock_missing_lag_1"] = np.isnan(stock_end).astype(np.int8)

        frame["month"] = target_weeks.month.astype(np.int8)
        frame["quarter"] = target_weeks.quarter.astype(np.int8)
        frame["week_of_year"] = week_number
        frame["week_sin"] = np.sin(2 * np.pi * week_number / 52.18)
        frame["week_cos"] = np.cos(2 * np.pi * week_number / 52.18)

        frame["sales_sum_log_lag_1"] = np.log1p(np.maximum(sales, 0))
        frame["discount_sum_log_lag_1"] = np.log1p(np.maximum(discount, 0))
        frame["discount_share_lag_1"] = discount_share
        frame["paid_value_per_unit_lag_1"] = np.log1p(np.maximum(paid_per_unit, 0))
        frame["full_value_per_unit_lag_1"] = np.log1p(np.maximum(full_per_unit, 0))

        context = {
            column: np.full(len(frame), np.nan, dtype=np.float32)
            for column in GROUP_COLUMNS["category_context"]
            + GROUP_COLUMNS["network_context"]
        }
        for origin in np.unique(origin_pos):
            row_mask = origin_pos == origin
            selected_pairs = pair_idx[row_mask]
            pair_sum_13 = self.quantity[origin - 13 : origin].sum(axis=0)
            pair_sum_4 = self.quantity[origin - 4 : origin].sum(axis=0)

            category_sum_13 = np.bincount(
                self.warehouse_category_codes,
                weights=pair_sum_13,
                minlength=self.n_warehouse_categories,
            )
            category_sum_4 = np.bincount(
                self.warehouse_category_codes,
                weights=pair_sum_4,
                minlength=self.n_warehouse_categories,
            )
            category_mean_13 = category_sum_13 / 13.0
            category_mean_4 = category_sum_4 / 4.0
            category_growth = np.log1p(category_mean_4) - np.log1p(category_mean_13)
            article_share_category = np.divide(
                pair_sum_13,
                category_sum_13[self.warehouse_category_codes],
                out=np.zeros_like(pair_sum_13),
                where=category_sum_13[self.warehouse_category_codes] > 0,
            )
            article_rank_category = (
                pd.Series(pair_sum_13)
                .groupby(self.warehouse_category_codes)
                .rank(method="average", pct=True)
                .to_numpy(dtype=np.float32)
            )
            active_articles = np.bincount(
                self.warehouse_category_codes,
                weights=(pair_sum_13 > 0).astype(np.float32),
                minlength=self.n_warehouse_categories,
            )

            last_sales = self.sales_sum[origin - 1]
            last_discount = self.discount_sum[origin - 1]
            last_full_value = last_sales + last_discount
            own_discount_share = np.divide(
                last_discount,
                last_full_value,
                out=np.zeros_like(last_discount),
                where=last_full_value > 0,
            )
            category_discount = np.bincount(
                self.warehouse_category_codes,
                weights=last_discount,
                minlength=self.n_warehouse_categories,
            )
            category_full_value = np.bincount(
                self.warehouse_category_codes,
                weights=last_full_value,
                minlength=self.n_warehouse_categories,
            )
            category_discount_share = np.divide(
                category_discount,
                category_full_value,
                out=np.zeros_like(category_discount),
                where=category_full_value > 0,
            )

            network_sum_13 = np.bincount(
                self.article_codes,
                weights=pair_sum_13,
                minlength=self.n_articles,
            )
            network_sum_4 = np.bincount(
                self.article_codes,
                weights=pair_sum_4,
                minlength=self.n_articles,
            )
            network_mean_13 = network_sum_13 / 13.0
            network_mean_4 = network_sum_4 / 4.0
            network_growth = np.log1p(network_mean_4) - np.log1p(network_mean_13)
            warehouse_share_article = np.divide(
                pair_sum_13,
                network_sum_13[self.article_codes],
                out=np.zeros_like(pair_sum_13),
                where=network_sum_13[self.article_codes] > 0,
            )
            active_warehouses = np.bincount(
                self.article_codes,
                weights=(pair_sum_13 > 0).astype(np.float32),
                minlength=self.n_articles,
            )

            selected_category = self.warehouse_category_codes[selected_pairs]
            selected_article = self.article_codes[selected_pairs]
            context["category_sales_mean_13_log"][row_mask] = np.log1p(
                category_mean_13[selected_category]
            )
            context["category_growth_4_vs_13"][row_mask] = category_growth[
                selected_category
            ]
            context["article_share_category_13"][row_mask] = article_share_category[
                selected_pairs
            ]
            context["article_rank_category_13"][row_mask] = article_rank_category[
                selected_pairs
            ]
            context["active_articles_category_13"][row_mask] = active_articles[
                selected_category
            ]
            context["discount_share_vs_category_lag_1"][row_mask] = (
                own_discount_share[selected_pairs]
                - category_discount_share[selected_category]
            )
            context["article_network_sales_mean_13_log"][row_mask] = np.log1p(
                network_mean_13[selected_article]
            )
            context["article_network_growth_4_vs_13"][row_mask] = network_growth[
                selected_article
            ]
            context["warehouse_share_article_13"][row_mask] = warehouse_share_article[
                selected_pairs
            ]
            context["active_warehouses_article_13"][row_mask] = active_warehouses[
                selected_article
            ]

        for column, values in context.items():
            frame[column] = values

        target = self.quantity[target_pos, pair_idx]
        return frame, target, target_pos


def columns_for_experiment(name: str) -> list[str]:
    columns = BASE_COLUMNS.copy()
    for group in EXPERIMENTS[name]["groups"]:
        columns.extend(GROUP_COLUMNS[str(group)])
    return columns


def make_truth(matrices: WeeklyMatrices, validation_origin: int) -> pd.DataFrame:
    validation_end = validation_origin + HORIZON
    weeks = np.repeat(
        matrices.weeks[validation_origin:validation_end].to_numpy(), matrices.n_pairs
    )
    pairs = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
    return pd.DataFrame(
        {
            "week": weeks,
            "warehouse_name": pairs["warehouse_name"],
            "article": pairs["article"],
            "quantity": matrices.quantity[validation_origin:validation_end].reshape(-1),
        }
    )


def prediction_frame(truth: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    result = truth[KEYS].copy()
    result["quantity"] = np.maximum(values, 0)
    return result


def train_and_predict(
    train: pd.DataFrame,
    target: np.ndarray,
    validation: pd.DataFrame,
    columns: list[str],
    iterations: int,
    model_params: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    cat_columns = [column for column in CAT_COLUMNS if column in columns]
    common = {
        "iterations": iterations,
        "depth": 7,
        "learning_rate": 0.08,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if model_params:
        common.update(model_params)
    classifier = CatBoostClassifier(loss_function="Logloss", **common)
    classifier.fit(train[columns], (target > 0).astype(np.int8), cat_features=cat_columns)

    positive = target > 0
    regressor = CatBoostRegressor(loss_function="RMSE", **common)
    regressor.fit(
        train.loc[positive, columns],
        np.log1p(target[positive]),
        cat_features=cat_columns,
    )

    probability = classifier.predict_proba(validation[columns])[:, 1]
    conditional = np.expm1(regressor.predict(validation[columns]))
    prediction = probability * np.maximum(conditional, 0)
    prediction[probability < 0.5] = 0.0
    diagnostics = {
        "mean_positive_probability": float(probability.mean()),
        "predicted_positive_rate": float((probability >= 0.5).mean()),
    }
    return prediction, diagnostics


def update_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    baseline_metrics: dict[str, float],
) -> None:
    result_path = output_dir / "feature_test_results.csv"
    previous = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    current = pd.DataFrame(rows)
    if not previous.empty:
        previous = previous[~previous["experiment"].isin(current["experiment"])]
        current = pd.concat([previous, current], ignore_index=True)
    current = current.sort_values("normalized_score").reset_index(drop=True)
    current.to_csv(result_path, index=False)

    metric_names = ["smape", "wape", "rmse", "bias", "rmsle"]
    ratios = current[metric_names].copy()
    for metric in metric_names:
        ratios[metric] /= baseline_metrics[metric]
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.55 * len(current) + 2)))
    image = ax.imshow(ratios.to_numpy(), cmap="RdYlGn_r", vmin=0.65, vmax=1.15, aspect="auto")
    ax.set_xticks(range(len(metric_names)), [name.upper() for name in metric_names])
    ax.set_yticks(range(len(current)), current["experiment"])
    for row in range(len(current)):
        for col in range(len(metric_names)):
            ax.text(col, row, f"{ratios.iloc[row, col]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Метрики относительно baseline mean13 (меньше 1 — лучше)", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Отношение к метрике baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_test_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    table_columns = ["experiment", "feature_count", *metric_names, "normalized_score"]
    table = current[table_columns].copy()
    for column in metric_names + ["normalized_score"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    headers = table.columns.tolist()
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join(markdown_lines)
    plan_rows = "\n".join(
        f"- `{name}` — {settings['description']}"
        for name, settings in EXPERIMENTS.items()
    )
    text = f"""# Эксперименты с недельными признаками

## Протокол

- Holdout: последние 9 известных недель.
- Direct multi-horizon: одна точка старта и горизонты 1–9 без подглядывания внутрь holdout.
- Во всех тестах одинаковые строки обучения, seed и параметры CatBoost.
- Денежные и складские признаки сдвинуты минимум на одну неделю.
- Gate классификатора фиксирован на 0.5; метрики на holdout не используются для его подбора.
- `normalized_score`: взвешенное отношение к baseline `mean13`; меньше — лучше.

## План тестов

{plan_rows}

## Текущие результаты

Baseline mean13: {json.dumps(baseline_metrics, ensure_ascii=False)}

{markdown_table}

## Денежные признаки

`sales_sum` уже возвращён из логарифма до недельной агрегации. Для последней
известной недели проверяются две версии:

- raw: `log1p(sales_sum)` и `log1p(discount_sum)`;
- derived: `discount_sum / (sales_sum + discount_sum)`, `sales_sum / quantity`
  и `(sales_sum + discount_sum) / quantity`.

Умножение доли скидки на 100 меняет только единицы измерения. Вычитать процент
из денежной суммы нельзя: это величины разных размерностей.
"""
    (output_dir / "FEATURE_EXPERIMENTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    validation_origin = len(matrices.weeks) - HORIZON
    latest_train_origin = validation_origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - args.train_origins + 1)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)

    train, target, _ = matrices.make_rows(train_origins, args.max_train_rows)
    validation, _, _ = matrices.make_rows(np.array([validation_origin]), None)
    truth = make_truth(matrices, validation_origin)

    mean13 = validation["quantity_mean_13"].to_numpy()
    baseline_prediction = prediction_frame(truth, mean13)
    baseline_metrics = compute_all_metrics_fast(truth, baseline_prediction)

    rows: list[dict[str, object]] = []
    for name in args.experiments:
        started = time.perf_counter()
        columns = columns_for_experiment(name)
        prediction, diagnostics = train_and_predict(
            train, target, validation, columns, args.iterations
        )
        pred_frame = prediction_frame(truth, prediction)
        metrics = compute_all_metrics_fast(truth, pred_frame)
        score = normalized_score(metrics, baseline_metrics)
        duration = time.perf_counter() - started
        row: dict[str, object] = {
            "experiment": name,
            "description": EXPERIMENTS[name]["description"],
            "feature_count": len(columns),
            "model": "CatBoost two-stage",
            "train_rows": len(train),
            "validation_start": str(matrices.weeks[validation_origin].date()),
            "validation_end": str(matrices.weeks[-1].date()),
            "gate": 0.5,
            **metrics,
            "normalized_score": score,
            "run_seconds": duration,
            **diagnostics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        update_outputs(args.output_dir, rows, baseline_metrics)

    metadata = {
        "train_origins": [str(matrices.weeks[pos].date()) for pos in train_origins],
        "validation_origin": str(matrices.weeks[validation_origin].date()),
        "training_rows": len(train),
        "iterations": args.iterations,
        "seed": 42,
        "baseline_metrics": baseline_metrics,
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
