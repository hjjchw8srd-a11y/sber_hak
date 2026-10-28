# Первый недельный TFT

TFT обучается как глобальная модель сразу на всех рядах «склад–товар» и выдаёт
9 недель прогноза одним проходом. Календарь известен на всём горизонте, сетевой
контекст используется только из прошлого. Отрицательные выходы обрезаются до 0.

## Настройки

```json
{
  "horizon": 9,
  "input_size": 26,
  "hidden_size": 32,
  "max_steps": 300,
  "batch_size": 128,
  "windows_batch_size": 512,
  "learning_rate": 0.001,
  "folds": 3,
  "series": 8789,
  "static_features": 78,
  "historic_features": [
    "article_network_sales_mean_13_log",
    "article_network_growth_4_vs_13",
    "warehouse_share_article_13",
    "active_warehouses_article_13"
  ],
  "future_features": [
    "time_idx",
    "month",
    "quarter",
    "week_of_year",
    "week_sin",
    "week_cos"
  ],
  "device": "gpu",
  "torch": "2.11.0+cu128",
  "cuda": "12.8"
}
```

## Результаты

| fold | начало | конец | SMAPE | WAPE | RMSE | Bias | RMSLE | score | секунды |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2026-05-25 | 2026-07-20 | 139.3049 | 47.2945 | 3.1958 | 74.9579 | 0.5108 | 0.9805 | 120.8 |
| 1 | 2026-03-23 | 2026-05-18 | 88.1854 | 51.2053 | 2.9592 | 75.9842 | 0.5221 | 0.8496 | 184.7 |
| 2 | 2026-01-19 | 2026-03-16 | 103.7383 | 49.9102 | 2.5411 | 75.0405 | 0.5180 | 0.9025 | 176.4 |

Средний normalized score: **0.9109**. Для сравнения,
зафиксированный CatBoost `base_calendar_network` имеет **0.7433** на этих же
трёх окнах. Меньше — лучше.
