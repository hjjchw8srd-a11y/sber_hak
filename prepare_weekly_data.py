"""Подготовка недельного датасета для second_version.

Правила актуального ТЗ:
- таргет: сумма quantity за календарную неделю (понедельник-воскресенье);
- для основного соревнования используются данные не позднее 2026-07-26;
- sales_sum и непустой discount_sum возвращаются через np.exp до агрегации;
- null в discount_sum означает отсутствие скидки и превращается в 0;
- дубликаты stock внутри одного дня суммируются;
- stock между разными днями не суммируется: из дней строятся недельные статистики.

Получившиеся stock-статистики описывают историческую неделю. При построении
модельных признаков их необходимо сдвигать минимум на одну неделю.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CUTOFF = "2026-07-26"
KEYS = ["week", "warehouse_name", "article"]
CATEGORY_COLUMNS = ["category_lvl_1", "category_lvl_2", "category_lvl_3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_dataset_hakaton_train.csv")
    parser.add_argument("--stock", default="stock_dataset_hakaton_train.csv")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", default="weekly_data")
    return parser.parse_args()


def monday_of_week(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates, errors="raise")
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.dayofweek, unit="D")


def prepare_sales(
    sales_path: Path, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    usecols = [
        "date",
        "warehouse_name",
        "article",
        "quantity",
        "sales_sum",
        "discount_sum",
        *CATEGORY_COLUMNS,
    ]
    sales = pd.read_csv(sales_path, usecols=usecols, parse_dates=["date"])
    original_rows = len(sales)
    rows_after_cutoff = int((sales["date"] > cutoff).sum())
    sales = sales.loc[sales["date"] <= cutoff].copy()

    sales["week"] = monday_of_week(sales["date"])

    # sales_sum логарифмирован через np.log. Три известных пропуска сохраняем
    # отдельным счётчиком, а их числовой вклад при агрегации считаем нулевым.
    sales_sum_missing = sales["sales_sum"].isna()
    sales["sales_sum_original"] = np.where(
        sales_sum_missing,
        0.0,
        np.exp(sales["sales_sum"].to_numpy(dtype=np.float64)),
    )
    sales["sales_sum_missing_rows"] = sales_sum_missing.astype(np.int16)

    # Важно: np.exp(0) == 1, поэтому сначала отделяем null, а не делаем
    # np.exp(discount_sum.fillna(0)).
    discount_present = sales["discount_sum"].notna()
    sales["discount_sum_original"] = np.where(
        discount_present,
        np.exp(sales["discount_sum"].fillna(0).to_numpy(dtype=np.float64)),
        0.0,
    )
    sales["discount_rows"] = discount_present.astype(np.int16)

    weekly_sales = (
        sales.groupby(KEYS, as_index=False, observed=True)
        .agg(
            quantity=("quantity", "sum"),
            sales_sum=("sales_sum_original", "sum"),
            discount_sum=("discount_sum_original", "sum"),
            sale_rows=("quantity", "size"),
            discount_rows=("discount_rows", "sum"),
            sales_sum_missing_rows=("sales_sum_missing_rows", "sum"),
        )
    )

    # Категории считаем метаданными артикула. Если категория менялась, берём
    # последнее известное значение и отражаем число нестабильных артикулов в summary.
    ordered_sales = sales.sort_values("date")
    article_meta = (
        ordered_sales.drop_duplicates("article", keep="last")
        .set_index("article")[CATEGORY_COLUMNS]
        .fillna("unknown")
        .astype(str)
    )
    category_variants = sales.groupby("article", observed=True)[CATEGORY_COLUMNS].nunique()
    unstable_articles = int((category_variants > 1).any(axis=1).sum())

    sales_summary: dict[str, object] = {
        "sales_source_rows": original_rows,
        "sales_rows_excluded_after_cutoff": rows_after_cutoff,
        "sales_rows_used": len(sales),
        "sales_sum_missing_rows": int(sales_sum_missing.sum()),
        "discount_null_rows_replaced_with_zero": int((~discount_present).sum()),
        "articles_with_category_changes": unstable_articles,
        "quantity_source_total": float(sales["quantity"].sum()),
        "quantity_aggregated_total": float(weekly_sales["quantity"].sum()),
    }
    return weekly_sales, article_meta, sales_summary


def prepare_stock(
    stock_path: Path, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    stock = pd.read_csv(stock_path, parse_dates=["date"])
    if "article" not in stock.columns and "char_article" in stock.columns:
        stock = stock.rename(columns={"char_article": "article"})

    required = {"date", "warehouse_name", "article", "stock"}
    missing = required - set(stock.columns)
    if missing:
        raise ValueError(f"В stock отсутствуют колонки: {sorted(missing)}")

    stock = stock[["date", "warehouse_name", "article", "stock"]]
    original_rows = len(stock)
    rows_after_cutoff = int((stock["date"] > cutoff).sum())
    stock = stock.loc[stock["date"] <= cutoff].copy()

    # Первый уровень: записи одного дня — части общего остатка, поэтому суммируем.
    daily_stock = (
        stock.groupby(["date", "warehouse_name", "article"], as_index=False, observed=True)[
            "stock"
        ]
        .sum()
        .sort_values(["warehouse_name", "article", "date"])
    )
    daily_stock["week"] = monday_of_week(daily_stock["date"])
    daily_stock["is_nonpositive"] = (daily_stock["stock"] <= 0).astype(np.int8)
    daily_stock["is_negative"] = (daily_stock["stock"] < 0).astype(np.int8)

    # Второй уровень: остатки разных дней не являются потоком. Вместо суммы
    # сохраняем состояние и характеристики наличия внутри недели.
    weekly_stock = (
        daily_stock.groupby(KEYS, as_index=False, observed=True)
        .agg(
            stock_week_start=("stock", "first"),
            stock_week_end=("stock", "last"),
            stock_min=("stock", "min"),
            stock_max=("stock", "max"),
            stock_mean=("stock", "mean"),
            stock_median=("stock", "median"),
            stock_observed_days=("stock", "size"),
            stock_nonpositive_days=("is_nonpositive", "sum"),
            stock_negative_days=("is_negative", "sum"),
        )
    )
    weekly_stock["stock_nonpositive_rate"] = (
        weekly_stock["stock_nonpositive_days"] / weekly_stock["stock_observed_days"]
    )
    weekly_stock["stock_change"] = (
        weekly_stock["stock_week_end"] - weekly_stock["stock_week_start"]
    )

    stock_summary: dict[str, object] = {
        "stock_source_rows": original_rows,
        "stock_rows_excluded_after_cutoff": rows_after_cutoff,
        "stock_rows_used": len(stock),
        "stock_daily_rows_after_duplicate_sum": len(daily_stock),
        "stock_duplicate_rows_collapsed": len(stock) - len(daily_stock),
        "negative_stock_source_rows": int((stock["stock"] < 0).sum()),
        "negative_daily_stock_after_sum": int((daily_stock["stock"] < 0).sum()),
    }
    return weekly_stock, stock_summary


def build_full_weekly_dataset(
    weekly_sales: pd.DataFrame,
    weekly_stock: pd.DataFrame,
    article_meta: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    warehouses = sorted(weekly_sales["warehouse_name"].unique())
    articles = sorted(weekly_sales["article"].unique())
    first_week = weekly_sales["week"].min()
    last_week = cutoff.normalize() - pd.Timedelta(days=cutoff.weekday())
    weeks = pd.date_range(first_week, last_week, freq="W-MON")

    grid = pd.MultiIndex.from_product(
        [weeks, warehouses, articles], names=KEYS
    ).to_frame(index=False)
    dataset = grid.merge(weekly_sales, on=KEYS, how="left", validate="one_to_one")

    zero_fill_columns = [
        "quantity",
        "sales_sum",
        "discount_sum",
        "sale_rows",
        "discount_rows",
        "sales_sum_missing_rows",
    ]
    dataset[zero_fill_columns] = dataset[zero_fill_columns].fillna(0)
    dataset = dataset.merge(weekly_stock, on=KEYS, how="left", validate="one_to_one")
    dataset = dataset.merge(
        article_meta.reset_index(), on="article", how="left", validate="many_to_one"
    )
    return dataset.sort_values(KEYS).reset_index(drop=True)


def validate_and_summarize(
    dataset: pd.DataFrame,
    weekly_sales: pd.DataFrame,
    weekly_stock: pd.DataFrame,
    cutoff: pd.Timestamp,
    sales_summary: dict[str, object],
    stock_summary: dict[str, object],
) -> dict[str, object]:
    duplicate_count = int(dataset.duplicated(KEYS).sum())
    if duplicate_count:
        raise AssertionError(f"В недельном датасете {duplicate_count} дублей ключа")

    source_total = float(sales_summary["quantity_source_total"])
    dataset_total = float(dataset["quantity"].sum())
    if not np.isclose(source_total, dataset_total):
        raise AssertionError(
            f"Потеря quantity при агрегации: source={source_total}, dataset={dataset_total}"
        )

    expected_step = dataset["week"].drop_duplicates().sort_values().diff().dropna()
    if not (expected_step == pd.Timedelta(days=7)).all():
        raise AssertionError("Недели идут не подряд с шагом 7 дней")

    summary: dict[str, object] = {
        "cutoff": str(cutoff.date()),
        "first_week": str(dataset["week"].min().date()),
        "last_week": str(dataset["week"].max().date()),
        "weeks": int(dataset["week"].nunique()),
        "warehouses": int(dataset["warehouse_name"].nunique()),
        "articles": int(dataset["article"].nunique()),
        "pairs": int(dataset.groupby(["warehouse_name", "article"]).ngroups),
        "rows": len(dataset),
        "duplicate_keys": duplicate_count,
        "zero_weekly_target_rows": int((dataset["quantity"] == 0).sum()),
        "zero_weekly_target_rate": float((dataset["quantity"] == 0).mean()),
        "positive_weekly_target_rows": int((dataset["quantity"] > 0).sum()),
        "weekly_quantity_mean": float(dataset["quantity"].mean()),
        "weekly_quantity_median": float(dataset["quantity"].median()),
        "weekly_quantity_max": float(dataset["quantity"].max()),
        "rows_without_weekly_stock": int(dataset["stock_observed_days"].isna().sum()),
        "rows_without_weekly_stock_rate": float(
            dataset["stock_observed_days"].isna().mean()
        ),
        "weekly_sales_groups": len(weekly_sales),
        "weekly_stock_groups": len(weekly_stock),
        **sales_summary,
        **stock_summary,
    }
    return summary


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff).normalize()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/4 Подготавливаем недельные продажи...")
    weekly_sales, article_meta, sales_summary = prepare_sales(Path(args.sales), cutoff)
    print("2/4 Агрегируем дневные остатки и считаем недельные характеристики...")
    weekly_stock, stock_summary = prepare_stock(Path(args.stock), cutoff)
    print("3/4 Создаём полную сетку неделя x склад x товар...")
    dataset = build_full_weekly_dataset(
        weekly_sales, weekly_stock, article_meta, cutoff
    )
    summary = validate_and_summarize(
        dataset,
        weekly_sales,
        weekly_stock,
        cutoff,
        sales_summary,
        stock_summary,
    )

    output_path = output_dir / "weekly_dataset.csv.gz"
    preview_path = output_dir / "weekly_dataset_preview.csv"
    summary_path = output_dir / "weekly_summary.json"

    print("4/4 Сохраняем датасет и проверки...")
    dataset.to_csv(output_path, index=False, compression="gzip")
    dataset.head(200).to_csv(preview_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Недельный датасет: {output_path.resolve()}")
    print(f"Пример 200 строк: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
