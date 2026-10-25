"""Шесть EDA-графиков для недельной постановки second_version."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("weekly_data/weekly_dataset.csv.gz")
OUTPUT_DIR = Path("weekly_plots")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    }
)


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_trend(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    network = data.groupby("week", as_index=True)["quantity"].sum().sort_index()
    ma4 = network.rolling(4, min_periods=1).mean()
    ma13 = network.rolling(13, min_periods=1).mean()
    validation_start = network.index[-9]
    validation_end = network.index[-1] + pd.Timedelta(days=6)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(network.index, network, color="#90CDF4", linewidth=1.2, alpha=0.65, label="Продажи за неделю")
    ax.plot(ma4.index, ma4, color="#2B6CB0", linewidth=2.0, label="Среднее за 4 недели")
    ax.plot(ma13.index, ma13, color="#C53030", linewidth=2.3, label="Среднее за 13 недель")
    ax.axvspan(
        validation_start,
        validation_end,
        color="#F6AD55",
        alpha=0.18,
        label="Пример валидации: последние 9 недель",
    )
    ax.set_title("Общий недельный объём продаж по всей сети", fontweight="bold")
    ax.set_xlabel("Понедельник прогнозной недели")
    ax.set_ylabel("Продано единиц за неделю")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "plot_1_weekly_trend.png")

    analysis["plot_1"] = {
        "first_week_sales": float(network.iloc[0]),
        "last_week_sales": float(network.iloc[-1]),
        "first_13_week_mean": float(network.iloc[:13].mean()),
        "last_13_week_mean": float(network.iloc[-13:].mean()),
        "minimum_week": str(network.idxmin().date()),
        "minimum_sales": float(network.min()),
        "maximum_week": str(network.idxmax().date()),
        "maximum_sales": float(network.max()),
        "validation_start": str(validation_start.date()),
    }


def plot_lagged_stock_effect(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    frame = data.sort_values(["warehouse_name", "article", "week"]).copy()
    pair_group = frame.groupby(["warehouse_name", "article"], sort=False)
    frame["stock_end_lag_1"] = pair_group["stock_week_end"].shift(1)
    frame = frame.dropna(subset=["stock_end_lag_1"])

    bins = [-np.inf, 0, 5, 20, 100, np.inf]
    labels = ["≤0", "1–5", "6–20", "21–100", ">100"]
    frame["stock_bucket"] = pd.cut(
        frame["stock_end_lag_1"], bins=bins, labels=labels, include_lowest=True
    )
    grouped = frame.groupby("stock_bucket", observed=False)["quantity"].agg(
        mean="mean", positive_rate=lambda x: (x > 0).mean(), count="size"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    bars = axes[0].bar(labels, grouped["mean"], color="#3182CE", width=0.65)
    for bar, value in zip(bars, grouped["mean"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )
    axes[0].set_title("Средние продажи следующей недели", fontweight="bold")
    axes[0].set_xlabel("Остаток в конце предыдущей недели")
    axes[0].set_ylabel("Продано единиц")

    positive_percent = grouped["positive_rate"] * 100
    bars = axes[1].bar(labels, positive_percent, color="#DD6B20", width=0.65)
    for bar, value, count in zip(bars, positive_percent, grouped["count"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{value:.1f}%\nn={count:,}",
            ha="center",
            fontsize=8,
        )
    axes[1].set_title("Вероятность ненулевой продажи", fontweight="bold")
    axes[1].set_xlabel("Остаток в конце предыдущей недели")
    axes[1].set_ylabel("Недель с продажами, %")
    axes[1].set_ylim(0, max(positive_percent.max() * 1.18, 10))
    fig.suptitle(
        "Связь прошлого остатка с продажами следующей недели\n"
        "Группа ≤0 включает нулевые и отрицательные остатки",
        fontsize=15,
        fontweight="bold",
    )
    save(fig, "plot_2_lagged_stock_effect.png")

    analysis["plot_2"] = {
        label: {
            "mean_next_week_quantity": float(grouped.loc[label, "mean"]),
            "positive_week_rate": float(grouped.loc[label, "positive_rate"]),
            "observations": int(grouped.loc[label, "count"]),
        }
        for label in labels
    }


def plot_monthly_seasonality(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    network = data.groupby("week", as_index=False)["quantity"].sum().sort_values("week")
    # Убираем сильный общий рост: сравниваем каждую неделю с локальным
    # центрированным 13-недельным уровнем. Иначе поздние месяцы получили бы
    # преимущество только потому, что относятся к более свежей части истории.
    network["local_trend"] = network["quantity"].rolling(
        13, center=True, min_periods=7
    ).mean()
    network["seasonal_index"] = network["quantity"] / network["local_trend"] * 100
    network["month"] = network["week"].dt.month
    grouped = network.groupby("month")["seasonal_index"].agg(["mean", "std", "count"])
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])
    months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(
        months,
        grouped.reindex(range(1, 13))["mean"],
        yerr=grouped.reindex(range(1, 13))["ci95"],
        capsize=3,
        color="#4A90E2",
        alpha=0.85,
    )
    for bar, value in zip(bars, grouped.reindex(range(1, 13))["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}%",
            ha="center",
            fontsize=8,
            rotation=0,
        )
    ax.set_title(
        "Сезонный индекс продаж по месяцу начала недели\n"
        "100% — локальный 13-недельный тренд; планки — 95% интервал",
        fontweight="bold",
    )
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Продажи относительно локального тренда, %")
    ax.axhline(100, color="#C53030", linewidth=1.5, linestyle="--", label="Локальный тренд")
    ax.legend(loc="upper left")
    save(fig, "plot_3_monthly_seasonality.png")

    month_means = grouped["mean"]
    analysis["plot_3"] = {
        "highest_month": int(month_means.idxmax()),
        "highest_month_seasonal_index": float(month_means.max()),
        "lowest_month": int(month_means.idxmin()),
        "lowest_month_seasonal_index": float(month_means.min()),
        "month_seasonal_indices": {str(int(k)): float(v) for k, v in month_means.items()},
    }


def plot_warehouse_distribution(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    weekly = data.groupby(["week", "warehouse_name"], as_index=False)["quantity"].sum()
    order = (
        weekly.groupby("warehouse_name")["quantity"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=weekly,
        x="warehouse_name",
        y="quantity",
        order=order,
        color="#63B3ED",
        showfliers=False,
        ax=ax,
    )
    means = weekly.groupby("warehouse_name")["quantity"].mean().reindex(order)
    ax.scatter(range(len(order)), means, marker="D", color="#C53030", s=25, label="Среднее")
    ax.set_title("Распределение недельных продаж по складам", fontweight="bold")
    ax.set_xlabel("Склад")
    ax.set_ylabel("Продано единиц за неделю")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(loc="upper right")
    save(fig, "plot_4_warehouse_weekly_distribution.png")

    stats = weekly.groupby("warehouse_name")["quantity"].agg(["mean", "median", "std"])
    analysis["plot_4"] = {
        "highest_mean_warehouse": str(stats["mean"].idxmax()),
        "highest_mean": float(stats["mean"].max()),
        "lowest_mean_warehouse": str(stats["mean"].idxmin()),
        "lowest_mean": float(stats["mean"].min()),
        "warehouse_stats": {
            name: {key: float(value) for key, value in row.items()}
            for name, row in stats.iterrows()
        },
    }


def plot_sparsity_and_stock(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    zero_share = float((data["quantity"] == 0).mean())
    stock_end = data["stock_week_end"]
    stock_labels = ["Нет данных", "≤0", "1–5", "6–20", "21–100", ">100"]
    stock_masks = [
        stock_end.isna(),
        stock_end.le(0),
        stock_end.between(1, 5),
        stock_end.between(6, 20),
        stock_end.between(21, 100),
        stock_end.gt(100),
    ]
    stock_shares = np.array([float(mask.mean() * 100) for mask in stock_masks])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    target_values = np.array([zero_share, 1 - zero_share]) * 100
    bars = axes[0].bar(
        ["Нулевая неделя", "Есть продажи"],
        target_values,
        color=["#A0AEC0", "#3182CE"],
        width=0.62,
    )
    for bar, value in zip(bars, target_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
        )
    axes[0].set_title("Разреженность недельного таргета", fontweight="bold")
    axes[0].set_ylabel("Доля недель × складов × товаров, %")
    axes[0].set_ylim(0, max(target_values) * 1.16)

    bars = axes[1].bar(stock_labels, stock_shares, color="#DD6B20", alpha=0.85)
    for bar, value in zip(bars, stock_shares):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
            fontsize=8,
        )
    axes[1].set_title("Остаток в конце исторической недели", fontweight="bold")
    axes[1].set_xlabel("Агрегированный дневной stock")
    axes[1].set_ylabel("Доля недельных строк, %")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Нули недельных продаж и покрытие остатков", fontsize=15, fontweight="bold")
    save(fig, "plot_5_weekly_sparsity_stock_coverage.png")

    analysis["plot_5"] = {
        "zero_target_rate": zero_share,
        "positive_target_rate": 1 - zero_share,
        "stock_bucket_rates": {
            label: float(value / 100) for label, value in zip(stock_labels, stock_shares)
        },
    }


def plot_feature_correlations(data: pd.DataFrame, analysis: dict[str, object]) -> None:
    weeks = pd.Index(sorted(data["week"].unique()))
    pairs = pd.MultiIndex.from_product(
        [
            sorted(data["warehouse_name"].unique()),
            sorted(data["article"].unique()),
        ],
        names=["warehouse_name", "article"],
    )

    def matrix(column: str) -> np.ndarray:
        return (
            data.pivot(index="week", columns=["warehouse_name", "article"], values=column)
            .reindex(index=weeks, columns=pairs)
            .to_numpy(dtype=np.float32)
        )

    quantity = matrix("quantity")
    sales_sum = matrix("sales_sum")
    discount_sum = matrix("discount_sum")
    stock_end = matrix("stock_week_end")
    stock_nonpositive = matrix("stock_nonpositive_rate")

    rng = np.random.default_rng(42)
    n_weeks, n_pairs = quantity.shape
    available = (n_weeks - 13) * n_pairs
    sample_size = min(250_000, available)
    flat_positions = rng.choice(available, sample_size, replace=False)
    week_pos = flat_positions // n_pairs + 13
    pair_pos = flat_positions % n_pairs

    past4 = np.column_stack(
        [quantity[week_pos - offset, pair_pos] for offset in range(1, 5)]
    )
    past13 = np.column_stack(
        [quantity[week_pos - offset, pair_pos] for offset in range(1, 14)]
    )
    sample = pd.DataFrame(
        {
            "Таргет недели": quantity[week_pos, pair_pos],
            "Продажи lag 1": quantity[week_pos - 1, pair_pos],
            "Продажи lag 4": quantity[week_pos - 4, pair_pos],
            "Среднее прошлых 4": past4.mean(axis=1),
            "Доля продаж за 13": (past13 > 0).mean(axis=1),
            "sales_sum lag 1": sales_sum[week_pos - 1, pair_pos],
            "discount_sum lag 1": discount_sum[week_pos - 1, pair_pos],
            "stock_end lag 1": stock_end[week_pos - 1, pair_pos],
            "stock≤0 rate lag 1": stock_nonpositive[week_pos - 1, pair_pos],
        }
    )
    corr = sample.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.75, "label": "Корреляция Спирмена"},
        ax=ax,
    )
    ax.set_title(
        "Корреляции недельного таргета с доступными прошлыми признаками\n"
        "Все признаки сдвинуты минимум на одну неделю",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "plot_6_weekly_feature_correlations.png")

    target_corr = corr["Таргет недели"].drop("Таргет недели").sort_values(ascending=False)
    analysis["plot_6"] = {
        "target_spearman": {name: float(value) for name, value in target_corr.items()},
        "sample_size": sample_size,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["week"])
    analysis: dict[str, object] = {}

    plot_weekly_trend(data, analysis)
    plot_lagged_stock_effect(data, analysis)
    plot_monthly_seasonality(data, analysis)
    plot_warehouse_distribution(data, analysis)
    plot_sparsity_and_stock(data, analysis)
    plot_feature_correlations(data, analysis)

    (OUTPUT_DIR / "weekly_plot_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # ASCII-вывод совместим со стандартной Windows-консолью; сам JSON выше
    # сохраняется в UTF-8 с русскими подписями.
    print(json.dumps(analysis, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
