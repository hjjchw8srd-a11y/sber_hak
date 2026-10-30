# Версия для защиты: Candidate LGBM52 и Forecast Container

**Дата фиксации:** 2026-10-30  
**Приоритет:** Воспроизводимая версия для защиты и задание со звёздочкой (production container + HTTP API).

---

## 1. Параметры зафиксированной версии

- **Ветка Git:** `codex/defense_api_ready` (создана от `per_category_models`).
- **Текущий коммит ветки:** `bb5aa29` (`bb5aa29b9ae997f7260293a4fbf7e6dbd1b8f4c3`).
- **Базовый коммит Git (HEAD upstream):** `9a0489f` (*Implement OOF Hurdle Stacking + Tweedie Peak Restoration for final submission*).
- **Статус Docker build/run:** **NOT RUN** (Docker CLI / daemon отсутствует на текущем хосте Windows; сервис, все 13 тестов и валидаторы проверены на 100% локально через Uvicorn и TestClient. Dockerfile и контейнерная сборка в этом окружении не выполнялись).
- **Вложенный репозиторий `sber_hak`:** gitlink на коммите `e3f48ae` (история и `.git` не изменялись).
- **Выбранный эталонный сабмит:** `submissions/candidate_lgbm52.csv`
  - **SHA-256:** `d8ff734c962bc782987f750962e6110fb6204ed4cacc7874048d160bd7ad1f0f`
  - **Число строк:** 79 101
  - **Суммарный объем продаж (`quantity`):** 210 453
  - **Максимальное значение (`max`):** 161
  - **99-й перцентиль (`p99`):** 36.0
  - **Доля ненулевых строк:** 43.39% (34 323 строки)
  - **Период прогноза основного соревнования:** 2026-07-27 .. 2026-09-21 (9 понедельников)
  - **Дата отсечки обучения:** 2026-07-26 (origin 134)
- **Конфигурационный манифест кандидата:** `submissions/candidate_lgbm52_manifest.json`  
  (SHA-256: `01063102888b0945e982551922dfc95eac6758c966f041d304ebd9a93f0da576`)
- **Неприкосновенность архива:** `final_submission.csv` (SHA-256: `7bf5e15ed2bdfa22c0cb13b9d0bf62cc5048273b0a1e82035f55579cc83158b7`) и исторические файлы OOF не перезаписывались.

---

## 2. Перечень канонических модулей и контрольные суммы SHA-256

Для устранения неоднозначности между корневой папкой и вложенным репозиторием `sber_hak` зафиксированы канонические модули, соответствующие проверенному кандидату:

| Канонический модуль | Абсолютный путь | SHA-256 | Роль в решении |
|---|---|---|---|
| `candidate_lgbm52.csv` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/submissions/candidate_lgbm52.csv` | `d8ff734c962bc782987f750962e6110fb6204ed4cacc7874048d160bd7ad1f0f` | Эталонный сабмит кандидата для защиты |
| `candidate_lgbm52_manifest.json` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/submissions/candidate_lgbm52_manifest.json` | `01063102888b0945e982551922dfc95eac6758c966f041d304ebd9a93f0da576` | Манифест параметров и метрик кандидата |
| `sber_hak/run_weekly_feature_experiments.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/sber_hak/run_weekly_feature_experiments.py` | `445b8c48ab9ee4e88011b8ceacdb315b14dc2a76e9ff33b06e1c2dddaf7a6285` | Канонический генератор признаков `WeeklyMatrices` |
| `sber_hak/history_audit_pipeline.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/sber_hak/history_audit_pipeline.py` | `f7a7bf029e91ea8b6f1356a01ba1504fc5e82c69f21db696533c61e1bdb2e898` | Пайплайн обучения LightGBM Variant B (52 origins) |
| `sber_hak/score.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/sber_hak/score.py` | `15fe3380a5c981126d4882ed9962db3a7f41684c68a927e1a9fedfaf7021445b` | Валидатор формата и официальный скоринг |
| `sber_hak/fast_score.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/sber_hak/fast_score.py` | `9b190fe96c884f0054c9d19dc264f114038d1fddc9dc4682607dfee48987b3c0` | Быстрый расчет 5 официальных метрик |
| `train_tft_final.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/train_tft_final.py` | `a31895934de5aed04a780e06103c03ef32bf218238ef721c405289be77909959` | Логика финального обучения и экспорта TFT |
| `train_tft_weekly.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/train_tft_weekly.py` | `2a2d6a01ab8358b52692273aa245489a3a6ea9f74283e01944ecb4033167e241` | Подготовка temporal и static фреймов TFT |
| `export_model_bundle.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/export_model_bundle.py` | `a388506e00b8480374e541fc11f970ea450ee0b93838421c64ec39c636f328f4` | Офлайн-скрипт сборки продакшн-бандла |
| `forecast-container/app/predictor.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/forecast-container/app/predictor.py` | `7cf0ddf8a29a2862d25087fa12fe4ffb234479e0a0d42125bb76997b69bb155e` | Самодостаточный модуль инференса ансамбля |
| `forecast-container/app/main.py` | `C:/Users/lenya/OneDrive/Рабочий стол/hak/forecast-container/app/main.py` | `da7e844f2ffbc777e48b894101e13e01c9006e885d58fbc8d3132e4d07b4d1b7` | FastAPI HTTP сервис раздела 10 |

---

## 3. Архитектура разделения: Offline Export vs Runtime Inference

1. **Команда офлайн-экспорта моделей (Offline Training & Bundle Packaging):**
   ```bash
   python export_model_bundle.py --data weekly_data/weekly_dataset.csv.gz --output-dir forecast-container/model
   ```
   - Занимает ~69 секунд.
   - Обучает LightGBM B (52 origins), CatBoost (18 origins) и 14 моделей категорий.
   - Экспортирует TFT без обучающего датасета (`save_dataset=False`).
   - Записывает канонический список 8 789 пар, конфигурацию постобработки и `manifest.json`.

2. **Команда запуска сервиса (Runtime Service Startup):**
   ```bash
   python -m uvicorn app.main:app --app-dir forecast-container --host 127.0.0.1 --port 8000
   ```
   - Модели загружаются в память один раз при старте.
   - При обработке POST `/predict` **не запускаются** `fit`, Optuna или поиск признаков; используются только готовые сериализованные модели.
   - Запросы детерминированно обслуживаются через семафор, кешируются по хэшу `sha256(sales + stock + bundle_digest)` и при необходимости принудительно рассчитываются заново через `?no_cache=true` / `X-Bypass-Cache: 1`.

---

## 4. Входы, артефакты и ограничения

- **Входы сервиса:**
  - `sales`: CSV с колонками `date, composite_division, cities_name, warehouse_name, category_lvl_1, category_lvl_2, category_lvl_3, article, characteristic, quantity, sales_sum, discount_sum` (глубина >= 180 дней).
  - `stock`: CSV с колонками `date, warehouse_name, stock` (глубина >= 180 дней).
- **Выход сервиса:**
  - CSV с заголовком `week,warehouse_name,article,quantity` ровно на 9 понедельников (79 101 строка, целые неотрицательные числа).
- **Ограничения и правила предобработки:**
  - Дата отсечки определяется динамически: `cutoff_date = sales["date"].max()`.
  - Дата первой недели прогноза: понедельник следующей календарной недели (`max_date.normalize() - Timedelta(days=max_date.weekday()) + Timedelta(days=7)`).
  - Если отсечка падает на понедельник, этот понедельник считается историческим, а прогноз начинается со следующего понедельника (+7 дней).
  - Отрицательные продажи запрещены (`422 VALIDATION_ERROR`).
  - Отрицательные остатки разрешены и трактуются как подтвержденная клиентская бронь.
  - Пропуски скидок (`null`) означают 0 скидки и корректно отделяются от логарифмированных значений.
  - Абсолютный календарный anchor: `2024-01-01` (индекс `time_idx` не зависит от длины переданного файла).
  - Вектор признаков: модели LightGBM B и CatBoost строго принимают 24 канонических признака (`base_calendar_network`). Дополнительные признаки остатков и денег рассчитываются препроцессором динамически, но не подаются в финальные модели кандидата.
  - Короткая история (180 дней vs вся история): честно верифицировано на одном ранее использованном историческом фолде (Fold 0, cutoff 2026-05-24, origin 125). Это проверка чувствительности реализации к длине окна истории, а не новый holdout. 180 дней дают 4.79% отличающихся строк (из-за нехватки 13 недель предыстории для сетевых средних на первых шагах 26-недельного энкодера TFT), дельта официального скора минимальна: -0.000551 (0.717154 против 0.717704).
  - Входные файлы для демонстрации: `forecast-container/data/demo_sales.csv` и `forecast-container/data/demo_stock.csv` (ровно 180 дней, 22 пары: 11 складов и 2 товара, 3 960 строк, полное соответствие 12 колонкам Section 2). Это демонстрационный срез для проверки интерфейсов и дымовых тестов (инференс ~12.7 с), он не является заменой полного входа соревнования.

---

## 5. Дистрибутивный архив и контрольные суммы

- **Команда сборки пакета для защиты:**
  ```bash
  python create_defense_package.py
  ```
- **Сформированный архив:** `defense_package_candidate_lgbm52.zip` (автономный runtime-комплект для инференса обученного ансамбля).
- **Контрольная сумма архива SHA-256:** сохраняется в отдельном sidecar-файле `defense_package_candidate_lgbm52.zip.sha256` при каждой сборке пакета во избежание циклической зависимости хэшей.
- **Проверка совпадения эталонного сабмита:**
  ```bash
  python audit_candidate_match.py
  ```
  - Результат: 79 101 строка, 0 расхождений, sum=210453, max=161.

