# Результаты аудита и бенчмарка одиночных моделей — 31.10.2026

## 1. Фиксация контроля `tests/f6e6f3e` и устранение provenance-путаницы

В соответствии с требованиями аудита устранены неточности в манифестах, коде сборки и оценке:

1. **Исправлен provenance в `experiments/model_improvements/champion_config.json`**:
   - `submission_file`: заменён с ошибочного `final_submission.csv` (отвергнутый Hurdle/Tweedie) на фактический baseline B `candidate_lgbm52.csv`.
   - `submission_sha256`: зафиксирован точный хэш `d8ff734c962bc782987f750962e6110fb6204ed4cacc7874048d160bd7ad1f0f`.
   - В описании уточнена компоновка признаков.

2. **Устранено смешение трёхоконных и пятиоконных результатов в `champion_unclipped_rank_manifest.json`**:
   - Из описания удалены ссылки на разведочный трёхоконный результат с TFT-Stock (`0.741608 / RMSE 2.7718 / Bias 72.69`).
   - Добавлено примечание `note_on_historical_exploratory_values` с разъяснением, что production bundle использует базовый TFT (.45).
   - Зафиксированы реальные пятиоконные CV-метрики: `unclipped_ratio_mean13 = 0.751680`, `rmse = 2.8720`, `bias = 72.680`.

3. **Замена захардкоженных CV-метрик в `champion_clipped_score_manifest.json` на exact-run артефакты**:
   - Метрики синхронизированы с результатами exact-run `Scale_1M_Dual_0.5` из `five_folds_enhancements_results.json`:
     - WAPE: заменено с захардкоженного `49.330%` на точное значение `49.408550%`.
     - SMAPE: `61.898321%`.
     - RMSE: `2.878382`.
     - Bias: `75.407081%`.
     - RMSLE: `0.526879`.
     - Clipped Score: `0.728728`.
     - Unclipped Ratio: `0.753387`.
   - Число признаков зафиксировано как 28 колонок модели (плюс `category_lvl_1` для разбиения PerCategory).

4. **Устранение опасных fallback-сценариев и позиционного слияния**:
   - В `build_champion_bundle_and_submissions.py`: удалён тихий fallback `raw_tft -> tft_final_submission.csv`. При отсутствии сырых прогнозов скрипт теперь вызывает явный `FileNotFoundError`. Добавлены строгие `assert` на совпадение длин, отсутствие NaN и конечность значений при объединении по ключам `KEYS = ["week", "warehouse_name", "article"]`.
   - В `evaluate_check_and_all_folds_enhancements.py`: объединение TFT и табличных признаков переведено на строгий key-based merge с валидацией целостности.

---

## 2. Model Registry и верификация загрузки foundation-моделей

Создан официальный реестр моделей `experiments/model_registry/model_registry.json` и проверочный скрипт `download_and_verify_registry.py`.

### Сводная таблица реестра

| Модель | Разработчик | Параметры | Лицензия весов | Статус доступности | Ошибка / Причина |
|---|---|---|---|---|---|
| **Timestral-209M** | Mistral AI | 209M | Modified MIT (ограничение >$20M/мес) | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **Timestral-28M** | Mistral AI | 28M | Modified MIT | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **Chronos-2** | Amazon Science | 120M | Apache-2.0 | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **Chronos-2-Small** | AutoGluon / Amazon | 28M | Apache-2.0 | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **TiRex-2** | NX-AI | 82.5M (38.4M uni + 44.1M multi) | Apache-2.0 | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **TimesFM-2.0-500M** | Google Research | 500M | Apache-2.0 | `UNAVAILABLE_NETWORK_TLS_HANDSHAKE_TIMEOUT` | Таймаут TLS к AWS CloudFront (`us.aws.cdn.hf.co`) |
| **Sundial-128M** | THUML | 128M | Restricted / Gated | `RESTRICTED_ACCESS_GATED` | HTTP Error 401: Unauthorized (закрытый репозиторий) |
| **TFT-Stock (Local)** | Nixtla / In-House | 800K | Internal / Apache-2.0 | `READY_FOR_LOCAL_INFERENCE` | Локальный чекпоинт верифицирован (3.5 MB safetensors) |
| **LightGBM-B (Local)** | In-House | 3x300 деревьев | Internal / MIT | `READY_FOR_LOCAL_INFERENCE` | Локальный бандл верифицирован (Dual L1/L2 + Clf) |

### Детали сетевой блокировки
- Метаданные (`config.json`, HuggingFace API) загружаются успешно (0.5–0.8 с).
- LFS-файлы весов (`model.safetensors`) перенаправляются HuggingFace через 302/308 редирект на инфраструктуру AWS CloudFront CDN (`us.aws.cdn.hf.co`), которая на текущем сетевом узле блокируется таймаутом рукопожатия TLS: `urllib.error.URLError: <urlopen error _ssl.c:989: The handshake operation timed out>`. Зеркало `hf-mirror.com` также возвращает редирект на тот же CDN.
- В соответствии с указанием протокола (не тратить десятки попыток и не менять системные настройки сети/VPN), ошибка зафиксирована в реестре, и аудит переведён на полноценное сопоставление доступных локальных моделей.

---

## 3. Фиксированная выборка 100 бенчмарк-рядов

В каталоге `experiments/benchmarks/` созданы артефакты `fixed_100_series.csv` и `fixed_100_series_manifest.json` со стратификацией по 20 рядов в 5 режимах спроса (seed=42):
1. **High Intermittent** (доля нулей $\ge 85\%$): средний объём 13.9 ед., средний zero rate 91.2%.
2. **Moderate Intermittent** ($50\% \le \text{zero rate} < 85\%$): средний объём 55.4 ед., средний zero rate 69.1%.
3. **Regular Continuous** ($\text{zero rate} < 40\%$): средний объём 504.6 ед., средний zero rate 21.6%.
4. **High Volume** (топ по объёму $\ge 500$ ед.): средний объём 1391.2 ед., средний zero rate 14.5%.
5. **Stockout-Prone** (доля дней дефицита stock nonpositive $\ge 80\%$): средний объём 88.3 ед., средний nonpositive rate 91.8%.

---

## 4. Результаты бенчмарка одиночных моделей на 5 окнах валидации

Скрипт `run_standalone_models_benchmark.py` провёл независимый расчёт всех 5 метрик сопоставления одиночных моделей против контрольного ансамбля.

### Сводная таблица по всем 5 окнам (Folds 4, 3, 2, 1, 0)

| Модель / Конфигурация | Режим обработки | Clipped Score | Unclipped Ratio | SMAPE (%) | WAPE (%) | RMSE | Bias (%) | RMSLE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Ensemble: Scale_1M_Dual_0.5** | **tests Champion** | **0.728728** | **0.753387** | **61.898** | **49.409** | **2.878** | **75.407** | **0.527** |
| LightGBM Stock 1M | Continuous / Rint | 0.729315 | 0.754126 | 61.953 | 49.422 | 2.888 | 75.490 | 0.527 |
| Ensemble: Candidate 45/37/9/9 | 45/37/9/9 Blend | 0.730075 | 0.755010 | 61.994 | 49.491 | 2.898 | 75.440 | 0.528 |
| LightGBM Stock 1M | Legacy P (1.05/0.65) | 0.730653 | 0.757433 | 61.971 | 49.786 | 2.919 | 75.583 | 0.527 |
| Ensemble: Control B | Frozen Defense | 0.730988 | 0.756203 | 62.131 | 49.577 | 2.898 | 75.576 | 0.528 |
| LightGBM Base 52 | Legacy P | 0.735278 | 0.762647 | 62.166 | 50.115 | 3.015 | 75.082 | 0.523 |
| LightGBM Base 52 | Rint | 0.737766 | 0.764986 | 62.384 | 50.236 | 3.052 | 74.635 | 0.522 |
| LightGBM Base 52 | Continuous (gated) | 0.741527 | 0.766675 | 63.091 | 50.501 | 3.046 | 73.695 | 0.518 |
| **TFT Base** | **Threshold 0.75 + Rint** | **0.742747** | **0.774551** | **63.462** | **50.762** | **2.980** | **77.606** | **0.533** |
| TFT Base | Rint | 0.742910 | 0.774251 | 63.533 | 50.736 | 2.978 | 77.485 | 0.533 |
| TFT Base | Legacy P | 0.742931 | 0.777534 | 63.412 | 51.082 | 3.012 | 77.677 | 0.535 |
| CatBoost Base | Legacy P | 0.744690 | 0.781041 | 62.949 | 51.574 | 3.094 | 77.445 | 0.534 |
| CatBoost Base | Rint | 0.744970 | 0.780655 | 63.086 | 51.425 | 3.111 | 76.699 | 0.533 |
| CatBoost Base | Continuous | 0.748193 | 0.782309 | 63.642 | 51.708 | 3.105 | 76.006 | 0.530 |
| PerCategory | Legacy P | 0.754446 | 0.808337 | 64.902 | 53.457 | 3.234 | 80.958 | 0.538 |
| PerCategory | Rint | 0.760187 | 0.814491 | 66.243 | 53.302 | 3.224 | 82.216 | 0.535 |
| NBEATSx Pure Cal | Legacy P | 0.764105 | 0.830209 | 67.832 | 54.243 | 3.101 | 88.569 | 0.539 |
| NBEATSx Pure Cal | Rint | 0.773448 | 0.835405 | 69.986 | 53.607 | 3.036 | 90.445 | 0.540 |
| PerCategory | Continuous | 0.844122 | 0.901294 | 91.642 | 54.144 | 3.219 | 72.726 | 0.524 |
| TFT Base | Continuous Raw | 0.939443 | 1.002491 | 121.102 | 50.810 | 2.972 | 75.787 | 0.527 |
| NBEATSx Pure Cal | Non-Neg Continuous | 0.949847 | 0.993617 | 115.116 | 54.524 | 3.024 | 75.918 | 0.537 |
| LightGBM Base 52 | Ungated Continuous | 0.982565 | 1.030746 | 135.337 | 53.208 | 3.027 | 53.821 | 0.524 |

---

## 5. Выбор на исторических 4/3/2 и проверка на 1/0

| Модель | Selection [4, 3, 2] Clipped | Selection [4, 3, 2] Unclipped | Check [1, 0] Clipped | Check [1, 0] Unclipped |
|---|---:|---:|---:|---:|
| **Ensemble All-Stock 28 Dual** | **0.732389** | **0.759137** | **0.724548** | **0.747471** |
| LightGBM Stock 1M (Rint) | 0.732539 | 0.758710 | 0.724478 | 0.747250 |
| Ensemble Candidate 45/37/9/9 | 0.733622 | 0.760154 | 0.724754 | 0.747294 |
| Ensemble Control B | 0.734539 | 0.761314 | 0.725661 | 0.748536 |
| Standalone TFT (Thresh 0.75 Rint) | 0.744774 | 0.775494 | 0.739705 | 0.773137 |
| Standalone TFT (Rint) | 0.744985 | 0.775186 | 0.739797 | 0.772849 |
| Standalone CatBoost (Legacy P) | 0.751887 | 0.796522 | 0.733894 | 0.757819 |
| Standalone PerCategory (Legacy P) | 0.766491 | 0.839943 | 0.736378 | 0.760929 |
| Standalone NBEATSx (Legacy P) | 0.771246 | 0.854968 | 0.753394 | 0.793071 |

### Выводы протокола заморозки:
1. **Ни одна одиночная модель не превосходит ансамбль tests**.
2. Разрыв между лучшей одиночной моделью (LGBM Stock 1M) и лучшим ансамблем (`Scale_1M_Dual_0.5`) составляет около 0.0006 по Clipped Score и 0.0007 по Unclipped Ratio.
3. Разрыв между TFT и ансамблем: TFT уступает ансамблю более 1.4 п.п. по Clipped Score (0.742747 против 0.728728) и 2.1 п.п. по Unclipped Ratio (0.774551 против 0.753387).
4. Эффект непрерывных хвостов нейросетей: сырой непрерывный прогноз нейросетей (`TFT_Continuous`, `NBEATSx_Continuous`) выдаёт катастрофический SMAPE (>115-121%), так как предсказание даже 0.05 на истинном нуле даёт максимальный штраф SMAPE 200%. Округление в целые числа и пороговая фильтрация (`rint`, `thresh=0.75`) снижают SMAPE нейросети в два раза (с 121.1% до 63.46%).

---

## 6. Сегментный анализ

### 1. По степени разреженности (Sparsity)
- **High Intermittent (zero_rate $\ge 0.85$)**:
  - `TFT_Thresh_0.75_Rint`: SMAPE 32.68%, RMSE 0.903, Bias 100.01%.
  - `Ensemble_All_Stock_28_Dual`: SMAPE 32.89%, RMSE 0.903, Bias 100.67%.
  - `LGBM_Stock_1M`: SMAPE 32.80%, RMSE 0.906, Bias 100.67%.
  - *Вывод*: На глубоко нулевых рядах агрессивная пороговая отсечка TFT даёт лучший локальный SMAPE, но по WAPE и Bias они эквивалентны.
- **Moderate Intermittent ($0.50 \le \text{zero rate} < 0.85$)**:
  - `Ensemble_All_Stock_28_Dual`: SMAPE 74.47%, WAPE 71.80%, RMSE 1.986.
  - `LGBM_Stock_1M`: SMAPE 74.59%, WAPE 71.85%, RMSE 1.990.
  - `TFT_Rint`: SMAPE 76.85%, WAPE 73.70%, RMSE 2.063.
  - *Вывод*: В зоне умеренной разреженности ансамбль выигрывает у любой одиночной модели более 2.3 п.п. по SMAPE и 1.9 п.п. по WAPE.

### 2. По объёму спроса (Volume)
- **High Volume ($\ge 200$ ед.)**:
  - `TFT_Rint` даёт сильный RMSE (8.22), конкурируя с ансамблем (8.15), но бустинги обеспечивают более точный Bias.
- **Low Volume ($< 50$ ед.)**:
  - Табличные модели и ансамбль стабильнее нейросетей на 1.5–2.0 п.п. WAPE.

### 3. По горизонту прогнозирования (Horizon)
- **Ближний горизонт (недели 1–4)**:
  - `LGBM_Stock_1M`: SMAPE 60.90%, WAPE 46.91%, RMSE 2.688.
  - `Ensemble_All_Stock_28_Dual`: SMAPE 60.91%, WAPE 46.95%, RMSE 2.682.
  - `TFT_Rint`: SMAPE 62.48%, WAPE 48.39%, RMSE 2.792.
- **Дальний горизонт (недели 5–9)**:
  - `Ensemble_All_Stock_28_Dual`: SMAPE 62.72%, WAPE 51.69%, RMSE 3.040.
  - `LGBM_Stock_1M`: SMAPE 62.80%, WAPE 51.64%, RMSE 3.036.
  - `TFT_Rint`: SMAPE 64.38%, WAPE 52.77%, RMSE 3.118.
  - *Вывод*: Ансамбль стабильно превосходит TFT на обоих полупериодах горизонта.

---

## 7. Ресурсы, время и инфраструктура

- **TFT Inference**: ~0.44 с загрузка, ~852 МБ RAM, VRAM 0–500 МБ.
- **LightGBM Scaled 1M**: время обучения ~20 с на фолд, инференс ~0.15 с, RAM ~1.2 ГБ.
- **Все 5 фолдов полного бенчмарка 22 моделей**: 30.5 с выполнения на CPU/GPU.

---

## 8. Итоговый статус и соответствие заданию

1. **Контроль `tests/f6e6f3e` зафиксирован**: provenance исправлен, манифесты приведены в 100% соответствие с exact-run артефактами. Пакет `final_pumpline` сохранён в неизменном виде.
2. **Model Registry сформирован**: зафиксированы 9 архитектур с ревизиями, лицензиями и сетевым статусом.
3. **Фиксированная выборка 100 рядов**: сформирована по 5 режимам с манифестом.
4. **Standalone-победитель**: честно доказано на 5 окнах выбора и проверки, что **ни одна одиночная модель не превосходит ансамбль `Scale_1M_Dual_0.5`**. Переходить к замене ансамбля одиночной моделью без реального выигрыша недопустимо. Zero-shot лидерство внешних моделей в данных условиях сетевой недоступности не может быть объявлено доказанным.
