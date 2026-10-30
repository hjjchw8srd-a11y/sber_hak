# Forecast Container — Сервис недельного прогнозирования спроса (Candidate LGBM52)

Production-ready контейнер и HTTP API для еженедельного прогнозирования спроса (`quantity`) по комбинациям «склад × артикул» на горизонт 9 недель в соответствии с разделом 10 ТЗ хакатона.

---

## 1. Описание решения

В сервисе развернут верифицированный ансамбль **Candidate LGBM52**, точно воспроизводящий эталонный сабмит `submissions/candidate_lgbm52.csv` (0 расхождений на всех 79 101 строках):

1. **Temporal Fusion Transformer (TFT)** — вес **50%**:
   - Глубокая нейросетевая модель из библиотеки `NeuralForecast` (`TFT`, 32 скрытых нейрона, 300 шагов).
   - Принимает исторические продажи `quantity`, динамический календарь (месяц, квартал, номер недели, sin/cos), сетевые признаки (`article_network_sales_mean_13_log`, `article_network_growth_4_vs_13`, `warehouse_share_article_13`, `active_warehouses_article_13`) и one-hot статические признаки складов и категорий 1–3 уровней.
   - Подается в финальный ансамбль в виде непрерывного сырого прогноза.
   - **Очищен от обучающей истории:** из артефактов исключены `dataset.safetensors` и `dataset.json`. История передается динамически при каждом запросе.

2. **LightGBM Variant B (52 origins)** — вес **30%**:
   - Двухэтапная модель (Hurdle Model): классификатор вероятности ненулевой продажи (`p_sale`) + регрессор логарифма объема при наличии продажи (`q_cond`).
   - Обучен на расширенном окне в 52 исторических точки происхождения (origins 74..125, 350 000 сэмплов, seed 42).
   - Фильтрация: при `p_sale < 0.5` прогноз зануляется; при `p_sale >= 0.5` используется математическое ожидание `p_sale * q_cond`.

3. **CatBoost Regressor** — вес **10%**:
   - Двухэтапная модель на 18 точках происхождения (origins 108..125, depth 7, 160 итераций).
   - Внутренний порог вероятности `p < 0.5` зануляет редкие продажи; в ансамбль подается как готовая двухэтапная оценка объема.

4. **14 категориальных моделей (Per-Category LightGBM)** — вес **10%**:
   - Отдельные пары классификатор + регрессор для каждого `category_lvl_1` (14 групп, полученных автоматически из данных, с константным fallback для малых групп при положительных сэмплах < 20).
   - Собственная постобработка: порог 0.75 и целочисленное округление `np.rint`.

### Финальный блендинг и постобработка:
- Взвешенная сумма: `raw_blend = 0.50 * TFT + 0.30 * LGBM_B + 0.10 * CatBoost + 0.10 * PerCategory`.
- Масштабирование: `scaled_blend = raw_blend * 1.05`.
- Финальный порог: `threshold = 0.65` (значения ниже 0.65 обращаются в 0).
- Округление: `np.rint`, нижняя граница 0.

---

## 2. Используемые признаки

В архитектуре сервиса строго разделены:
1. **Канонический набор признаков моделей ансамбля (`feature_columns` в `manifest.json`)**: признаки, которые реально подаются на вход обученным моделям кандидата.
2. **Расширенный препроцессор признаков**: пайплайн динамической подготовки матриц признаков.

### Активный вектор признаков для LightGBM B и CatBoost (24 признака):
- **Идентификаторы и категории (5)**: `warehouse_name`, `article`, `category_lvl_1`, `category_lvl_2`, `category_lvl_3`.
- **Горизонт и абсолютное время (2)**: `horizon` (1..9), `time_idx` (недели от базового якоря `2024-01-01`).
- **Лаги и статистики продаж (8)**: `quantity_lag_1`, `quantity_lag_2`, `quantity_lag_4`, `quantity_lag_13`, `mean_4`, `mean_13`, `std_13`, `nonzero_rate_13`.
- **Календарная сезонность (5)**: `month`, `quarter`, `week_of_year`, `week_sin`, `week_cos`.
- **Сетевой контекст артикула (4)**: `article_network_sales_mean_13_log`, `article_network_growth_4_vs_13`, `warehouse_share_article_13`, `active_warehouses_article_13`.

### Признаки для 14 категориальных моделей Per-Category LightGBM (23 признака):
- Используются те же признаки, за исключением `category_lvl_1`, так как обучение и инференс маршрутизируются отдельно для каждой категории.

### Входные данные для Temporal Fusion Transformer (TFT):
- **Основной временной ряд**: `quantity` (исторические продажи).
- **Временные экзогенные признаки (`futr_exog`, 6)**: `time_idx`, `month`, `quarter`, `week_of_year`, `week_sin`, `week_cos`.
- **Исторические сетевые признаки (`hist_exog`, 4)**: `article_network_sales_mean_13_log`, `article_network_growth_4_vs_13`, `warehouse_share_article_13`, `active_warehouses_article_13`.
- **Статические категориальные признаки (`stat_exog`, 78)**: One-hot кодирование складов и иерархии категорий 1–3 уровней.

### Дополнительные признаки, рассчитываемые препроцессором:
Пайплайн предобработки также вычисляет остатки (`stock_end_lag_1`, `stock_mean_lag_1`, индикаторы дефицита), денежные показатели (`sales_sum_log_lag_1`, скидки) и расширенную форму ряда (`min_13`, `max_13`, `cv_13`, тренды `slope_13`, `slope_26`). В конфигурации **Candidate LGBM52** эти признаки не включены в `feature_columns` финальных моделей, так как в абляциях они приводили к небольшому смещению на тесте. Они рассчитываются динамически и доступны для расширенных конфигураций.

> **Важно:** Все признаки рассчитываются строго ретроспективно до даты отсечки (`cutoff`). Никакие будущие значения не используются.

---

## 3. Инструкция по сборке

Контейнер полностью автономен и не требует доступа в интернет во время инференса. Все веса моделей (37 файлов, 17.5 MB) и статические преобразования упакованы в папку `model/`.

### Сборка Docker-образа:
```bash
cd forecast-container
docker build -t forecast-service:latest .
```

- Базовый образ: `python:3.11-slim`.
- Поддерживается как CPU, так и GPU инференс (PyTorch 2.11, LightGBM 4.7.0, CatBoost 1.2.10).

---

## 4. Инструкция по запуску

### Запуск в Docker на GPU (рекомендуется):
```bash
docker run -d --gpus all -p 8000:8000 --name forecast-app forecast-service:latest
```

### Запуск в Docker на CPU:
```bash
docker run -d -p 8000:8000 --name forecast-app forecast-service:latest
```

### Локальный запуск без Docker (Windows / Linux):
```bash
python -m uvicorn app.main:app --app-dir forecast-container --host 127.0.0.1 --port 8000
```

---

## 5. Описание API и примеры curl

Сервис предоставляет 4 эндпоинта по спецификации раздела 10 ТЗ:

### 1. `GET /health` — проверка готовности сервиса
Проверяет загрузку моделей в память, контрольную сумму бандла и счетчик инференсов.

```bash
curl -sf http://localhost:8000/health
```

Пример ответа:
```json
{
  "status": "healthy",
  "model_bundle": "candidate_lgbm52_bundle",
  "bundle_digest": "4308ad2c385c7df7...",
  "components": ["tft", "lightgbm_b", "catboost", "per_category"],
  "has_cached_forecast": true,
  "inference_call_count": 3
}
```

### 2. `POST /predict` — расчет прогноза по загруженным данным
Принимает два multipart CSV-файла: `sales` и `stock`. Глубина истории должна быть не менее 180 дней.

```bash
curl -X POST http://localhost:8000/predict \
  -F "sales=@data/sample_sales.csv" \
  -F "stock=@data/sample_stock.csv" \
  -o forecast.csv
```

#### Кеширование и принудительный инференс:
- Сервис кеширует последний результат прогноза по криптографическому ключу `sha256(sales_bytes + b"::" + stock_bytes + b"::" + bundle_digest.encode())`.
- Повторный запрос с идентичными файлами отдается из памяти за <0.05 с без увеличения `inference_call_count`.
- Для явного обхода кеша и принудительного повторного запуска моделей передайте параметр `no_cache=true` или HTTP-заголовок `X-Bypass-Cache: 1`:

```bash
curl -X POST "http://localhost:8000/predict?no_cache=true" \
  -F "sales=@data/sample_sales.csv" \
  -F "stock=@data/sample_stock.csv" \
  -o forecast.csv
```

Ответ (`text/csv`):
```csv
week,warehouse_name,article,quantity
2026-08-03,warehouse_name_1,char_article_1,84
2026-08-03,warehouse_name_1,char_article_2,21
...
```

#### Подготовка входного среза данных (180 дней):
Для извлечения ровно 180 календарных дней из исходных датасетов соревнования:
```python
import pandas as pd
cutoff = pd.Timestamp("2026-07-26")
start_date = cutoff - pd.Timedelta(days=179)  # ровно 180 календарных дней включительно

sales = pd.read_csv("sales_dataset_hakaton_train.csv", parse_dates=["date"])
sales_180 = sales[(sales["date"] >= start_date) & (sales["date"] <= cutoff)]
sales_180.to_csv("sample_sales_180d.csv", index=False)

stock = pd.read_csv("stock_dataset_hakaton_train.csv", parse_dates=["date"])
stock_180 = stock[(stock["date"] >= start_date) & (stock["date"] <= cutoff)]
stock_180.to_csv("sample_stock_180d.csv", index=False)
```

### 3. `GET /predict` — срез прогноза по артикулу и/или складу
Возвращает срез последнего рассчитанного через POST прогноза. Должен быть указан хотя бы один query-параметр: `article` или `warehouse`. При указании обоих возвращается их пересечение.

```bash
# Прогноз по товару на всех складах
curl "http://localhost:8000/predict?article=char_article_1" -o article_forecast.csv

# Прогноз по складу для всех товаров
curl "http://localhost:8000/predict?warehouse=warehouse_name_1" -o warehouse_forecast.csv

# Прогноз по конкретной паре
curl "http://localhost:8000/predict?article=char_article_1&warehouse=warehouse_name_1"
```

### 4. `GET /predict/all` — полный последний прогноз
Возвращает полный прогноз в формате CSV или JSON.

```bash
# В формате CSV (по умолчанию)
curl "http://localhost:8000/predict/all?format=csv" -o full_forecast.csv

# В формате JSON
curl "http://localhost:8000/predict/all?format=json" -o full_forecast.json
```

---

## 6. Структура директории

```
forecast-container/
├── Dockerfile              # Рецепт легковесного образа на базе python:3.11-slim
├── .dockerignore           # Исключение лишних файлов (датасеты, git, oof, кэши)
├── requirements.txt        # Зафиксированные версии Python-зависимостей
├── README.md               # Документация и руководство пользователя
├── app/
│   ├── __init__.py
│   ├── main.py             # Сервер FastAPI, маршрутизация и кэширование
│   ├── predictor.py        # Загрузка моделей, подготовка признаков, инференс
│   └── validators.py       # Валидация входных файлов и кодов ошибок
├── model/
│   ├── manifest.json       # Контрольные суммы SHA-256, метаданные и параметры
│   ├── valid_pairs.parquet # 8 789 канонических пар (11 складов x 799 артикулов)
│   ├── postprocessing.json # Веса компонентов, alpha, threshold, округление
│   ├── tft/                # NeuralForecast TFT (без обучающей истории)
│   ├── lgbm_b/             # LightGBM Variant B (52 origins)
│   ├── catboost/           # CatBoost 18 origins (двухэтапный)
│   └── per_category/       # 14 моделей категорий и маршрутизация
├── data/
│   ├── sample_sales.csv    # Пример входного CSV продаж (100 строк)
│   └── sample_stock.csv    # Пример входного CSV остатков (100 строк)
└── tests/
    ├── test_api.py         # Автоматизированный тестовый набор (12 проверок)
    └── test_api.sh         # Скрипт проверки эндпоинтов через curl
```

---

## 7. Особенности и ограничения

1. **Динамический горизонт прогноза:** Дата первой недели рассчитывается автоматически: следующий понедельник после максимальной даты продаж.
   - Для воскресенья `2026-07-26` старт `2026-07-27`.
   - Для пятницы `2026-07-31` старт `2026-08-03`.
   - Для понедельника `2026-08-03` старт `2026-08-10`.
2. **Абсолютный календарный anchor:** Параметр `time_idx` вычисляется строго от `2024-01-01`, благодаря чему при передаче произвольного 180-дневного окна временные координаты не сдвигаются.
3. **Отрицательные остатки:** Значения `stock < 0` валидны (означают зарезервированный товар) и корректно обрабатываются.
4. **Пропуски скидки:** Пропуск (`null`) в `discount_sum` означает отсутствие скидки и переводится в 0 до экспоненты.
5. **Последовательный инференс:** POST-запросы защищены асинхронным семафором (`asyncio.Lock`), предотвращая дублирование тяжелых моделей в памяти GPU/RAM.
6. **Кэш прогноза:** Результат последнего успешного POST кэшируется атомарно по SHA-256 содержимого обоих файлов; при повторном запросе тех же данных ответ возвращается мгновенно без повторного пересчета.
