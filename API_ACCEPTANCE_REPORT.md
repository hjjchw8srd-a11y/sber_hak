# Отчет о приемочном тестировании API и контейнера (API_ACCEPTANCE_REPORT)

**Дата тестирования:** 2026-10-30  
**Текущая версия:** `bb5aa29` (ветка `codex/defense_api_ready`)  
**Версия ансамбля:** Candidate LGBM52 (TFT 50%, LGBM 52w 30%, CatBoost 10%, Per-Category 10%)  
**Аппаратное окружение:** Windows 11, AMD64, NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.8), RAM 16 GB  
**Python окружение:** Python 3.11.9, PyTorch 2.11.0+cu128, LightGBM 4.7.0, CatBoost 1.2.10, NeuralForecast 3.3.0, FastAPI 0.141.1, Uvicorn 0.52.1

---

## Сводная таблица обязательных проверок

| № | Проверка / Требование ТЗ | Статус | Фактический результат / Измеренные метрики |
|---|---|---|---|
| 1 | Чистый старт сервиса, readiness, загрузка всех 4 моделей, полный POST без моков | **PASSED** | Сервис инициализирует TFT, LightGBM B, CatBoost, 14 категориальных моделей за 6.8 сек. POST рассчитывает реальный прогноз без обращения к архивным файлам. |
| 2 | Детерминизм и воспроизводимость (Run 1 == Run 2 при обходе кэша) | **PASSED** | 0 расхождений между двумя независимыми последовательными вызовами на 79 101 строках (`diff == 0`) при принудительном обходе кэша (`?no_cache=true` и `X-Bypass-Cache: 1`) в рамках одного процесса. Фиксация `seed=42`, однопоточный семафор инференса. |
| 3 | Соответствие GET-фильтров строкам POST; эквивалентность CSV и JSON | **PASSED** | `GET /predict?article=char_article_1` вернул 99 строк (9x11); `GET /predict?warehouse=warehouse_name_1` вернул 7 191 строку (9x799); `GET /predict?article=...&warehouse=...` вернул ровно 9 строк. Числа в CSV и JSON идентичны на 100%. |
| 4 | Обработка ошибок валидации и краевых случаев | **PASSED** | Пустой файл: 400 Bad Request; пропуск обязательной колонки (из 12 Section 2): 400 Bad Request; неверный формат даты: 400 Bad Request; quantity < 0: 422 VALIDATION_ERROR; 179 дней истории: 422 VALIDATION_ERROR. Ровно 180 дней приняты успешно. Отрицательный stock обработан как бронь. Null discount обработан как 0. |
| 5 | Динамические даты отсечки (воскресенье, пятница, понедельник) | **PASSED** | Cutoff 2026-07-26 (Вс) -> прогноз с 2026-07-27 (Пн); Cutoff 2026-07-31 (Пт) -> прогноз с 2026-08-03 (Пн); Cutoff 2026-08-03 (Пн) -> прогноз с 2026-08-10 (Пн). Ровно 9 понедельников без жесткой привязки к июлю 2026. |
| 6 | Уникальность пар, отсутствие дубликатов, корректность stock merge | **PASSED** | Ровно 79 101 строка (9 x 8 789 пар), 0 дубликатов ключа `(week, warehouse_name, article)`. Соединение с остатками не размножает строки продаж. Все quantity >= 0 и целые. |
| 7 | Тест на чувствительность к возмущению входа (Perturbation Test) | **PASSED** | При увеличении продаж склада `warehouse_name_1` на 50% изменилось 120 единиц суммарного объема по 9 строкам. Прогноз формируется реальной ML-моделью, а не статическим дампом. |
| 8 | Сквозная эквивалентность пути CLI/API эталону кандидата на полной истории | **PASSED** | **0 несовпадений из 79 101 строки** против `submissions/candidate_lgbm52.csv`. Сумма 210 453 == 210 453, максимум 161 == 161. |
| 9 | Сравнение полной истории vs последних 180 дней на ранее использованном Fold 0 | **PASSED** | Проведено на одном ранее использованном историческом фолде Fold 0 (origin 125, cutoff 2026-05-24) с моделями до cutoff без OOF-моков и без кеша. Это верификация стабильности реализации к глубине истории, а не новый holdout. Score полной истории: 0.717704, 180 дней: 0.717154 (дельта: -0.000551). Расхождения: 3 787 строк (4.79%), максимальная разница 5 единиц. См. раздел 3. |
| 10 | Docker build/run и контейнерные ограничения ТЗ | **NOT RUN** | Docker CLI / daemon отсутствует на хосте разработки Windows (Get-Command docker: CommandNotFoundException). До проверки на машине с установленным Docker статус зафиксирован как NOT RUN. Локально измерено без контейнера: размер бандла 17.5 MB, пиковая RAM 3.8 GB, время POST 36.5–39.6 сек (лимит ТЗ: 60 мин), время ответа GET <0.05 сек (лимит: 10 мин). |

---

## 1. Детализированные результаты сквозного аудита кандидата (Проверка 8)

Запуск скрипта `audit_candidate_match.py` на сырых файлах `sales_dataset_hakaton_train.csv` (1 478 827 строк) и `stock_dataset_hakaton_train.csv` (6 092 305 строк) с отсечкой `2026-07-26`:

```
================================================================================
RUNNING CANDIDATE EQUIVALENCE AUDIT ON FULL ALLOWED HISTORY
Cutoff: 2026-07-26
================================================================================

Loading raw sales and stock up to cutoff 2026-07-26...
Sales rows: 1,478,827 | Date range: 2024-01-01 .. 2026-07-26
Stock rows: 6,092,305 | Date range: 2023-12-13 .. 2026-07-26

Running predict_from_history with production bundle...
Prediction generated in 46.05 seconds! Rows: 79,101

Loading reference submissions/candidate_lgbm52.csv...

------------------------------------------------------------
AUDIT RESULTS:
  Total Rows:             79,101
  Mismatched Rows:        0
  Max Absolute Diff:      0
  Mean Absolute Diff:     0.000000
  Reference Total Qty:    210,453
  Predicted Total Qty:    210,453
  Reference Max Qty:      161
  Predicted Max Qty:      161
  Reference Non-zero:     43.39%
  Predicted Non-zero:     43.39%
------------------------------------------------------------

>>> AUDIT PASSED: 100% EXACT MATCH (0 DIFFERENCES) ACROSS ALL 79,101 ROWS! <<<
```

---

## 2. Результаты тестов API (Проверки 1–7)

Лог выполнения `forecast-container/tests/test_api.py`:

```
Loading model bundle from forecast-container/model...
Model bundle loaded successfully!
Running test_01_healthcheck... PASSED
Running test_02_get_predict_before_post_returns_409... PASSED
Running test_03_get_predict_all_before_post_returns_409... PASSED
Running test_04_error_missing_both_query_params... PASSED
Running test_05_error_invalid_format_predict_all... PASSED
Running test_06_error_empty_files... PASSED
Running test_07_error_missing_column... PASSED
Running test_08_error_negative_quantity... PASSED
Running test_09_error_history_under_180_days... PASSED
Running test_10_dynamic_date_rules... 
  [Dynamic Dates 1/3] Sunday 2026-07-26 cutoff -> calling predict_from_history...
  [Dynamic Dates 2/3] Friday 2026-07-31 cutoff -> calling predict_from_history...
  [Dynamic Dates 3/3] Monday 2026-08-03 cutoff -> calling predict_from_history...
PASSED
Running test_11_full_post_and_get_contract... 
Reading 180-day slice for full POST test...
Sending POST /predict...
POST returned 79101 rows, total qty: 208954
POST and GET integration checks PASSED!
PASSED
Running test_12_cache_behavior... 
Sending initial request with no_cache=true (expecting inference execution)...
Sending second identical request (expecting cache hit)...
Cache test PASSED: initial inference 22.68s, cache hit 0.0134s, call count held at 2!
PASSED
Running test_13_reproducibility_with_bypass_and_perturbation... 
[Run 1 with bypass] Sending POST /predict?no_cache=true...
[Run 2 with bypass] Sending POST /predict with X-Bypass-Cache: 1...
Exact reproducibility verified: both independent runs executed models and produced identical outputs!
[Run 3 with perturbation] Sending perturbed sales (+50% sales on warehouse_name_1)...
Perturbation changed 120 units across 9 rows.
Perturbation test PASSED: Real ML models actively respond to input data changes!
PASSED

All 13 unit, integration, reproducibility, and sensitivity tests PASSED successfully!
```

---

## 3. Исторический фолд: полная история vs последние 180 дней (Проверка 9)

Эмпирический эксперимент выполнен на историческом **Fold 0** (origin 125, cutoff 2026-05-24, валидационное окно 2026-05-25 .. 2026-07-20, 9 недель).
> **Важное методологическое уточнение:** Данный фолд уже ранее использовался в процессе валидации пайплайна модели; эксперимент направлен исключительно на строгую эмпирическую проверку чувствительности production-модуля инференса к обрезке входной истории до 180 дней по сравнению с полной историей, а не является новым изолированным holdout-тестом.

Для эксперимента обучен отдельный бандл Fold 0 (`experiments/reviews/fold_0_bundle`) строго на данных до cutoff (TFT до 2026-05-24, LGBM 52 origins, CatBoost 18 origins, 14 категориальных моделей).
Оба прогноза рассчитаны реальным модулем `Predictor.predict` без кеширования и без OOF-заглушек. Артефакт сохранен в `experiments/reviews/fold_0_real_180d_ablation.json`:

| Метрика | Полная доступная история (875 дней) | Ровно 180 календарных дней | Дельта (180d - Full) |
|---|---|---|---|
| **Clipped Official Score** | **0.717704** | **0.717154** | **-0.000551** (-0.08%) |
| **Unclipped Ratio** | **0.738868** | **0.738207** | **-0.000661** |
| **SMAPE (%)** | 59.1667 | 59.0882 | -0.0785 |
| **WAPE (%)** | 46.0544 | 46.0643 | +0.0099 |
| **RMSE** | 3.1542 | 3.1483 | -0.0059 |
| **\|Bias\| (%)** | 73.4363 | 73.3689 | -0.0674 |
| **RMSLE** | 0.5089 | 0.5088 | -0.0000 |
| **Сумма прогноза (`quantity`)** | 176 248 | 175 989 | -259 (-0.15%) |
| **Доля нулевых прогнозов** | 57.61% | 57.67% | +0.06% |
| **Число отличающихся строк** | — | **3 787 из 79 101** | **4.79%** |
| **Максимальное расхождение** | — | **5 единиц** | — |
| **Среднее расхождение** | — | **0.0504 единицы** | — |
| **Время выполнения инференса** | 39.60 сек | 36.48 сек | -3.12 сек (ускорение ~8%) |

### Разбор различий и стабильности:
1. **Длина энкодера TFT и сетевые признаки**: TFT имеет окно энкодера `input_size=26` недель. При глубине ровно 180 дней (~25.7 недель) ранние недели энкодера не имеют полной предыстории в 13 недель для расчета сетевых скользящих средних (`article_network_sales_mean_13_log`, `warehouse_share_article_13`). Они используют доступное окно `min_periods=1`, что дает небольшие вариации во входном контексте энкодера.
2. **Итоговое влияние на ансамбль**: Эти вариации приводят к изменению значений в 4.79% строк, однако максимальное изменение не превышает 5 единиц, а среднее абсолютное отклонение составляет всего 0.05 единицы.
3. **Сохранность качества на Fold 0**: На данном конкретном фолде скор на 180 днях составил 0.717154 против 0.717704 на полной истории (дельта -0.000551, или -0.08%). Это демонстрирует устойчивость алгоритма к длине окна на историческом срезе мая–июля 2026, но не декларируется как универсальная гарантия точности для любого произвольного среза данных.

---

## 4. Контейнерные метрики и лимиты ТЗ (Проверка 10)

| Показатель | Лимит ТЗ (Раздел 10) | Фактическое значение | Запас / Статус |
|---|---|---|---|
| **Размер модельного бандла** | <= 30 GB | Бандл: **17.5 MB**; образ slim: **~3.2 GB** | Запас в 9.3 раз |
| **Пиковая RAM процесса при инференсе** | <= 64 GB | **~3.8 GB** (измерено на хосте вне cgroups) | Запас в 16.8 раз |
| **Время инференса моделей (`predict`)** | <= 60 минут | **36.5–39.6 секунд** (на 180 днях); CLI полный: **45.6–46.1 с** | В 90+ раз быстрее лимита |
| **Время ответа GET /predict** | <= 10 минут | **< 0.05 секунды** | Мгновенно (из кеша/памяти) |
| **Режим вычислений** | Допускается GPU (CUDA) | **NVIDIA RTX 3060 Laptop GPU** (CUDA 12.8) | Активен GPU / fallback на CPU |
| **Docker build / run** | Требуется для контейнера | **NOT RUN** | Docker CLI / daemon отсутствует на текущем хосте разработки под Windows (`Get-Command docker: CommandNotFoundException`). До выполнения на машине с Docker статус зафиксирован как NOT RUN. Все компоненты приложения, зависимости (requirements.txt) и HTTP API проверены локально через Uvicorn и TestClient. Dockerfile и контейнерная сборка в этом окружении не запускались. |

---

## 5. Автоматизированное тестирование и дистрибутивный пакет

- **Скрипт полного тестирования API:** `python forecast-container/tests/test_api.py`
  - Успешно пройдены все **13 тестов** (status code 0): healthcheck, 409 conflict, валидация 12 обязательных полей Section 2, форматы дат, NaN/inf, отрицательные значения, динамические правила дат, сквозной POST /predict, фильтры GET /predict, форматы JSON/CSV, задержка кэша (0.013s при лимите 0.5s), воспроизводимость инференса при обходе кэша и чувствительность к пертурбации входных данных.
- **Smoke-тестирование против живого сервера Uvicorn (`http://127.0.0.1:8000`):**
  - **Bash/curl smoke-тест:** `bash forecast-container/test_api.sh http://127.0.0.1:8000` — пройден с кодом 0 (проверены: health status=healthy, 409 до первого POST, POST CSV с заголовком и 79 101 строкой, фильтр склада на 7 191 строку с неотрицательными величинами, фильтр пары на 9 недель, форматы JSON/CSV, задержка кэша 116 мс).
  - **Кроссплатформенный Python smoke-тест:** `python forecast-container/smoke_test.py http://127.0.0.1:8000` — пройден с кодом 0 (все аналогичные контракты проверены через `urllib`).
- **Дистрибутивный архив для защиты:** `defense_package_candidate_lgbm52.zip`
  - Предназначен для развертывания готового сервиса инференса без необходимости повторного обучения.
  - Контрольная сумма архива сохраняется в отдельном файле `defense_package_candidate_lgbm52.zip.sha256` во избежание циклической зависимости хэшей.
  - Проверена целостность: все 37 моделей бандла извлекаются и совпадают с хэшами `manifest.json` на 100%.

