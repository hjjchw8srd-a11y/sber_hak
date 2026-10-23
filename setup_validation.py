import pandas as pd
import numpy as np
from datetime import timedelta

print("="*60)
print("1. ПОДГОТОВКА СТРУКТУРЫ ВАЛИДАЦИИ (ПОСЛЕДНИЕ 64 ДНЯ)")
print("="*60)

# Загружаем sales
print("Загрузка sales...")
df_sales = pd.read_csv("sales_dataset_hakaton_train.csv")
df_sales["date"] = pd.to_datetime(df_sales["date"])

# Агрегируем по департаментам (сумма quantity)
print("Агрегация департаментов (sum quantity)...")
sales_agg = (
    df_sales.groupby(["date", "warehouse_name", "article"], as_index=False)
    .agg({"quantity": "sum"})
)

# Определяем горизонт валидации: последние 64 дня
max_date = sales_agg["date"].max()
val_dates = pd.date_range(end=max_date, periods=64, freq="D")
val_start_date = val_dates.min()

print(f"Период валидации (64 дня): с {val_start_date.date()} по {max_date.date()}")
print(f"Период обучения: до {val_start_date.date() - timedelta(days=1)}")

# Фиксируем все уникальные склады и артикулы
warehouses = sorted(df_sales["warehouse_name"].unique())
articles = sorted(df_sales["article"].unique())
print(f"Складов: {len(warehouses)}, Артикулов: {len(articles)}, Серий: {len(warehouses) * len(articles)}")

# 1. Формируем полную сетку валидации (Ground Truth)
print("Создание полной сетки Ground Truth (562 496 строк)...")
grid_val = pd.MultiIndex.from_product(
    [val_dates, warehouses, articles],
    names=["date", "warehouse_name", "article"]
).to_frame().reset_index(drop=True)

# Мерджим с реальными продажами
df_val_truth = grid_val.merge(
    sales_agg, on=["date", "warehouse_name", "article"], how="left"
)
df_val_truth["quantity"] = df_val_truth["quantity"].fillna(0.0)

# Сохраняем ground truth для валидации
df_val_truth.to_csv("val_ground_truth.csv", index=False)
print(f"val_ground_truth.csv сохранён! Строк: {len(df_val_truth):,}")
print(f"Фактических продаж > 0 на валидации: {(df_val_truth['quantity'] > 0).sum():,} из {len(df_val_truth):,}")

# 2. Отрезаем данные для обучения (до даты старта валидации)
train_sales = sales_agg[sales_agg["date"] < val_start_date].copy()
print(f"Строк в обучающих продажах (только дни с продажами): {len(train_sales):,}")

# ─────────────────────────────────────────────────────────────
# 3. СОЗДАНИЕ БЕЙЗЛАЙНОВ
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. ГЕНЕРАЦИЯ БЕЙЗЛАЙНОВ ДЛЯ СРАВНЕНИЯ")
print("="*60)

# Бейзлайн 1: Все нули (Zero Baseline)
print("Генерация Baseline 1: Zero (все нули)...")
df_b_zero = grid_val.copy()
df_b_zero["quantity"] = 0.0
df_b_zero.to_csv("baseline_zero.csv", index=False)

# Бейзлайн 2: Наивный повтор последнего известного дня (Naive Last Day)
print("Генерация Baseline 2: Последнее известное значение...")
last_day_train = train_sales["date"].max()
last_day_sales = train_sales[train_sales["date"] == last_day_train]
df_b_last = grid_val.merge(
    last_day_sales[["warehouse_name", "article", "quantity"]],
    on=["warehouse_name", "article"],
    how="left"
)
df_b_last["quantity"] = df_b_last["quantity"].fillna(0.0)
df_b_last = df_b_last[["date", "warehouse_name", "article", "quantity"]]
df_b_last.to_csv("baseline_naive_last.csv", index=False)

# Бейзлайн 3: Среднее по дню недели за последние 4 недели перед валидацией (Day-of-Week Mean)
print("Генерация Baseline 3: Среднее по дням недели за последние 28 дней...")
four_weeks_ago = val_start_date - timedelta(days=28)
recent_train = train_sales[
    (train_sales["date"] >= four_weeks_ago) & (train_sales["date"] < val_start_date)
].copy()

# Полная сетка за последние 28 дней для честного учета нулей!
recent_dates = pd.date_range(start=four_weeks_ago, end=val_start_date - timedelta(days=1), freq="D")
grid_recent = pd.MultiIndex.from_product(
    [recent_dates, warehouses, articles],
    names=["date", "warehouse_name", "article"]
).to_frame().reset_index(drop=True)

grid_recent = grid_recent.merge(recent_train, on=["date", "warehouse_name", "article"], how="left")
grid_recent["quantity"] = grid_recent["quantity"].fillna(0.0)
grid_recent["dayofweek"] = grid_recent["date"].dt.dayofweek

# Считаем среднее по каждому дню недели для каждой пары (склад, артикул)
dow_means = (
    grid_recent.groupby(["warehouse_name", "article", "dayofweek"])["quantity"]
    .mean()
    .reset_index()
    .rename(columns={"quantity": "dow_mean"})
)

df_b_dow = grid_val.copy()
df_b_dow["dayofweek"] = df_b_dow["date"].dt.dayofweek
df_b_dow = df_b_dow.merge(dow_means, on=["warehouse_name", "article", "dayofweek"], how="left")
df_b_dow["quantity"] = df_b_dow["dow_mean"].fillna(0.0)
df_b_dow = df_b_dow[["date", "warehouse_name", "article", "quantity"]]
df_b_dow.to_csv("baseline_dow_mean.csv", index=False)

print("\nВсе 3 бейзлайна успешно сохранены!")
