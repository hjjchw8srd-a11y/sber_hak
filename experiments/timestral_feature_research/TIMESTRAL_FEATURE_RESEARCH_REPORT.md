# Финальный аудит: Timestral Feature Research & Adaptive Ensemble (Задание №9)

**Дата:** 2026-11-01  
**Ветка:** `tests`  
**Репозиторий:** `hjjchw8srd-a11y/sber_hak`  
**Базовый коммит Задания 8:** `d2d6208`  

---

## 1. Исправление выбора стратегии (Методологический аудит)

В Задании №8 таблица предварительно сортировалась по `Full 5-Fold Clipped Score`, из-за чего Check-фолды [1, 0] оказывали влияние на ранжирование.
В соответствии с соревновательным протоколом проекта:
- **Основной критерий выбора модели:** средний **Selection Unclipped Ratio** на Selection-фолдах [4, 3, 2] (Origins 89, 98, 107).
- **Вторичный критерий выбора:** средний **Selection Clipped Score** на Selection-фолдах [4, 3, 2].
- **Check-фолды [1, 0]:** этап подтверждения устойчивости (validation check).

Ниже представлены полные результаты всех 13 протестированных схем без сокрытия метрик.

### Таблица 1: Сортировка строго по Selection Unclipped Ratio (Основной критерий)
| Схема смеси | Средний вес | Selection Unclipped [4,3,2] | Selection Clipped [4,3,2] | Check Unclipped [1,0] | Check Clipped [1,0] | Full 5-Fold Unclipped | Full 5-Fold Clipped |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adaptive: Full Multi-Regime** (Selection Winner) | **0.057** | **`0.756014`** | **`0.730213`** | 0.747199 | 0.724963 | **`0.752488`** | **`0.728113`** |
| Adaptive: Volume Gated (0.02 / 0.10 / 0.18) | 0.070 | `0.756100` | `0.730257` | 0.747113 | 0.724925 | 0.752505 | 0.728124 |
| Adaptive: Full Tuned (Post-Check Candidate) | 0.052 | `0.756155` | `0.730320` | **0.746984** | **0.724785** | **`0.752486`** | **`0.728106`** |
| Adaptive: Sparsity + Volume (0.00 / 0.05 / 0.10 / 0.15) | 0.066 | `0.756164` | `0.730350` | 0.747141 | 0.724931 | 0.752555 | 0.728183 |
| Adaptive: Volume Conservative (0.00 / 0.10 / 0.15) | 0.057 | `0.756169` | `0.730332` | 0.746984 | 0.724806 | 0.752495 | 0.728121 |
| Fixed $w=0.12$ | 0.120 | `0.756264` | `0.730415` | 0.747012 | 0.724893 | 0.752563 | 0.728206 |
| Fixed $w=0.15$ | 0.150 | `0.756392` | `0.730492` | 0.747176 | 0.725049 | 0.752705 | 0.728315 |
| **Fixed $w=0.10$ (Task 7 Best)** | 0.100 | **`0.756430`** | **`0.730531`** | 0.747017 | 0.724827 | **`0.752665`** | **`0.728249`** |
| Adaptive: Horizon Decay (0.12 near/mid, 0.06 far) | 0.100 | `0.756433` | `0.730595` | 0.746842 | 0.724706 | 0.752597 | 0.728239 |
| Fixed $w=0.20$ | 0.200 | `0.756627` | `0.730706` | 0.747412 | 0.725255 | 0.752941 | 0.728525 |
| Fixed $w=0.08$ | 0.080 | `0.756644` | `0.730691` | 0.747032 | 0.724868 | 0.752799 | 0.728362 |
| Fixed $w=0.05$ | 0.050 | `0.756780` | `0.730734` | 0.747345 | 0.725085 | 0.753006 | 0.728474 |
| **Fixed $w=0.00$ (Champion Baseline)** | 0.000 | **`0.757336`** | **`0.731140`** | 0.747464 | 0.725109 | **`0.753387`** | **`0.728728`** |

### Идентификация победителей:
1. **Победитель строго по Selection Unclipped Ratio:**  
   **`Adaptive: Full Multi-Regime`** (`sel_unclipped = 0.756014`, `sel_clipped = 0.730213`).
2. **Победитель строго по Selection Clipped Score:**  
   **`Adaptive: Full Multi-Regime`** (`sel_clipped = 0.730213`, `sel_unclipped = 0.756014`).
3. **Статус `Adaptive: Full Tuned`:**  
   Схема `Full Tuned` заняла 3-е место на этапе Selection (`0.756155`), но за счёт сильного перформанса на Check-фолдах [1, 0] (`chk_clipped = 0.724785` против `0.724963`) показала наименьший Full 5-Fold Clipped Score (`0.728106` против `0.728113`).  
   В строгом протоколе без подглядывания в Check-фолды **легитимным победителем является `Adaptive: Full Multi-Regime`**.

---

## 2. Пофолдовые метрики (Per-Fold Analysis)

Сохранена полная per-fold таблица: [`experiments/timestral_feature_research/adaptive_blend_per_fold_metrics.csv`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/experiments/timestral_feature_research/adaptive_blend_per_fold_metrics.csv).

### Clipped Score по всем 5 фолдам:
| Схема смеси | Mean Full [0..4] | Mean Sel [4,3,2] | Mean Chk [1,0] | Fold 4 (Origin 89) | Fold 3 (Origin 98) | Fold 2 (Origin 107) | Fold 1 (Origin 116) | Fold 0 (Origin 125) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adaptive: Full Multi-Regime** (Selection Winner) | **0.728113** | **0.730213** | 0.724963 | **0.734661** | **0.733475** | 0.722502 | 0.732386 | 0.717540 |
| **Adaptive: Full Tuned** (Post-Check Candidate) | **0.728106** | 0.730320 | **0.724785** | 0.734767 | 0.733681 | 0.722513 | **0.732211** | 0.717360 |
| Adaptive: Volume Conservative | 0.728121 | 0.730332 | 0.724806 | 0.734516 | 0.733892 | 0.722587 | 0.732164 | 0.717448 |
| **Fixed $w=0.10$** (Task 7 Best) | **0.728249** | 0.730531 | 0.724827 | 0.735161 | 0.734134 | 0.722296 | 0.732333 | 0.717321 |
| **Champion ($w=0.00$)** | **0.728728** | 0.731140 | 0.725109 | 0.736573 | 0.734618 | **0.722229** | 0.733102 | **0.717116** |

---

## 3. Методологическая прозрачность и ограничения исследования

1. **Используемые признаки в адаптивных схемах:**
   - Схемы `Full Multi-Regime` и `Full Tuned` используют **исключительно** характеристики спроса (`quantity_mean_13`, `quantity_nonzero_rate_13`, `current_zero_streak`) и горизонт ($h=1..9$).
   - **Складские признаки (`stock_snr`, `stock_end`) в финальных правилах НЕ используются**, так как диагностический анализ показал, что складские остатки не дают устойчивого дифференцирующего сигнала для относительного преимущества Timestral по сравнению с объёмом и нулевой активностью.
2. **Priority B (Корректирующий слой residual corrector):**
   - Модель коррекции остатков Timestral, обученная на OOF фолдов 4 и 3, показала ухудшение метрик на независимых фолдах [2, 1, 0] (+0.000077 по Clipped Score).
   - Порог выигрыша $\ge 0.0001$ не достигнут; **Priority B полностью отклонён**.
3. **Priority C (Проверка эффекта 1 сентября):**
   - Исторический Fold 4 имеет origin 89 (2025-09-15) и охватывает продажи с середины сентября до ноября. Недели 87 и 88 (непосредственный всплеск школьного сезона вокруг 1 сентября) остались в истории до точки отсечки.
   - Поэтому тест на Fold 4 **не является полноценной проверкой эффекта 1 сентября**, а показывает лишь динамику середины сентября. На этих неделях ручной множитель ухудшал скор за счёт роста overprediction bias.
4. **Статус Check-фолдов [1, 0]:**
   - Фолды 1 и 0 использовались при оценке ранних экспериментов проекта (Task 6, Task 7) и **не являются абсолютно нетронутым слепым holdout**. Их роль — проверка согласованности в рамках зафиксированного протокола 3 Selection + 2 Check.

---

## 4. Унификация кода и строгий Production Roundtrip

Для устранения любого дублирования логики между CV-экспериментами и генерацией submission создан единый модуль:
[`adaptive_blending_rules.py`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/adaptive_blending_rules.py).

### Результаты проверки roundtrip ([`test_strict_roundtrip_audit.py`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/test_strict_roundtrip_audit.py)):
1. **Сохранность `delivery_final/submission.csv`:**
   - CRLF SHA256: `09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939` (100% совпадение, файл не изменялся).
2. **Zero-Weight Test ($w=0$):**
   - Принудительное $w=0$ даёт **ровно 0 расхождений на всех 79 101 строках** против `delivery_final/submission.csv`.
3. **Roundtrip-совпадение кандидатов:**
   - Пересчёт признаков по истории до origin=134 и вычисление весов через `adaptive_blending_rules` полностью совпадает с сохранёнными CSV на диске: **max delta = 0, differing rows = 0**.

---

## 5. Финальное сопоставление кандидатов и рекомендации

| Решение / Файл | Статус в проекте | Selection Unclipped [4,3,2] | Selection Clipped [4,3,2] | Full 5-Fold Clipped | CRLF SHA256 | LF SHA256 |
|---|---|:---:|:---:|:---:|---|---|
| **1. `delivery_final/submission.csv`** | **Действующий Champion** (TFT+LGB+CB+PCat) | 0.757336 | 0.731140 | 0.728728 | `09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939` | `ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0` |
| **2. `submissions/submission_champion_timestral_refit_blend_w10.csv`** | **Консервативный ансамбль** (Fixed 10% Timestral) | 0.756430 | 0.730531 | 0.728249 | `f1b402e1b12b50428d02e88a09ee2ba5cbbce4b08170d1026c459f37c352da3f` | `fceae5ea41b2aaee0beea5b290dfeb69033285744f4d226a26df1f528a47fa10` |
| **3. `submissions/submission_champion_timestral_adaptive_selection_winner.csv`** | **Строгий победитель Selection** (Full Multi-Regime) | **`0.756014`** | **`0.730213`** | **`0.728113`** | `57426825f659a1508d6eca2738730777551282d75dcd678661831e5bd537469b` | `4a93a2d07b43b40c2a0b6929c1a3a2ddca08870c02f5cc92bf219494db128ea0` |
| **4. `submissions/submission_champion_timestral_adaptive_candidate.csv`** | **Кандидат с учётом Check-фолдов** (Full Tuned) | 0.756155 | 0.730320 | **`0.728106`** | `318112634dddb6f6a3f5b2b4b735081952587506745c149416bdf47fefdd71da` | `1d26f2b96dd8a731d7fd479905e8737a85e9258e55feec6edbe8390a95b373f9` |

### Различия между решениями:
- `Adaptive Selection Winner` отличается от `Adaptive Full Tuned` всего на **370 строках из 79 101** (99.53% совпадение, максимальная дельта ровно 1).
- `Adaptive Selection Winner` отличается от действующего Чемпиона на **2 989 строках** (3.78% строк), точечно усиливая Timestral на высокообъемных регулярных артикулах и зануляя на редких и мертвых рядах.

### Стратегия для нескольких сабмитов (Multi-Submission Strategy):
1. **Сабмит 1 (Безусловный базовый фундамент):**  
   `delivery_final/submission.csv` — проверенная чемпионская поставка (Score: 0.728728). Гарантия надёжности без риска переобучения.
2. **Сабмит 2 (Методологически строгий адаптивный фаворит):**  
   [`submissions/submission_champion_timestral_adaptive_selection_winner.csv`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/submissions/submission_champion_timestral_adaptive_selection_winner.csv) — строго выбран по Selection Unclipped Ratio (`0.756014`) и Selection Clipped (`0.730213`), выигрыш Full 5-Fold Score: **`0.728113`**.
3. **Сабмит 3 (Агрессивный кандидат):**  
   [`submissions/submission_champion_timestral_adaptive_candidate.csv`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/submissions/submission_champion_timestral_adaptive_candidate.csv) или [`submissions/submission_champion_timestral_refit_blend_w10.csv`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/submissions/submission_champion_timestral_refit_blend_w10.csv) — резервный вариант с минимальным Full 5-Fold Clipped (`0.728106`).
