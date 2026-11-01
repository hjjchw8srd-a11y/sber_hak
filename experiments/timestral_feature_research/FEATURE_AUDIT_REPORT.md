# Полный аудит признаков для Timestral-209M (Feature Audit Report)

**Дата:** 2026-11-01  
**Ветка:** `tests`  
**Цель:** Сверка реализованных признаков в `timestral_feature_engine.py` с `FEATURE_REGISTRY.md`, моделями TFT/LightGBM/CatBoost и исходными недельными таблицами.

---

## 1. Сводная таблица статуса признаков по группам

| Группа | Исходный признак | Источник | Статус | Реализация в `timestral_feature_engine.py` | Временная валидность на Cutoff |
|---|---|:---:|:---:|---|:---:|
| **A. Динамика спроса** | `quantity_lag_1` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=1))` | $t \le \text{origin}-1$ |
| | `quantity_lag_2` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=2))` | $t \le \text{origin}-2$ |
| | `quantity_lag_4` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=4))` | $t \le \text{origin}-4$ |
| | `quantity_lag_13` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=13))` | $t \le \text{origin}-13$ |
| | `quantity_mean_4` | Rolling | **Реализован** | `signed_log1p(rolling_mean(Q, 4))` | $t \le \text{origin}-1$ |
| | `quantity_mean_13` | Rolling | **Реализован** | `signed_log1p(rolling_mean(Q, 13))` | $t \le \text{origin}-1$ |
| | `quantity_nonzero_rate_13`| Rolling | **Реализован** | `rolling_mean(Q > 0, 13)` | $t \le \text{origin}-1$ |
| | `growth_4_vs_13` | Rolling | **Реализован** | $\text{clip}(\text{mean}_4 / (\text{mean}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `quantity_cv_13` | Rolling | **Реализован** | $\text{clip}(\text{std}_{13} / (\text{mean}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `quantity_min_13` / `max_13` | Rolling | *Не включён в baseline A* | Доступен как расширенный канал; не приравнивается автоматически к внутренним патчам | $t \le \text{origin}-1$ |
| | `quantity_slope_13` | Rolling | *Не включён в baseline A* | Линейный наклон 13 недель требует отдельной проверки, не приравнивается к attention | $t \le \text{origin}-1$ |
| **B. Складские остатки** | `stock_week_end` | Исходный | **Реализован** | `signed_log1p(stock_end)` | $t \le \text{origin}-1$ |
| | `stock_week_start` | Исходный | **Реализован** | `signed_log1p(stock_week_start)` | $t \le \text{origin}-1$ |
| | `stock_change` | Производный | **Реализован** | `signed_log1p(stock_change)` | $t \le \text{origin}-1$ |
| | `stock_min` | Исходный | **Реализован** | `signed_log1p(stock_min)` | $t \le \text{origin}-1$ |
| | `stock_max` | Исходный | **Реализован** | `signed_log1p(stock_max)` | $t \le \text{origin}-1$ |
| | `stock_mean` | Исходный | **Реализован** | `signed_log1p(stock_mean)` | $t \le \text{origin}-1$ |
| | `stock_nonpositive_rate` | Исходный | **Реализован** | `stock_nonpositive_rate` $[0, 1]$ | $t \le \text{origin}-1$ |
| | `stock_missing` | Производный | **Реализован** | Бинарный индикатор `isnan(stock_end)` | $t \le \text{origin}-1$ |
| | `stockout_flag` | Производный | **Реализован** | $(\text{stock\_end} \le 0 \land \text{observed} > 0)$ | $t \le \text{origin}-1$ |
| | `weeks_of_supply` | Производный | **Реализован** | $\text{clip}(\text{stock\_end} / (\text{mean}_{13} + 10^{-4}), 0, 20)$ | $t \le \text{origin}-1$ |
| **C. Сетевой контекст** | `article_network_sales_mean_13_log` | Network | **Реализован** | `signed_log1p(mean_13(network_sales))` | $t \le \text{origin}-1$ |
| | `article_network_growth_4_vs_13` | Network | **Реализован** | $\text{clip}(\text{net}_4 / (\text{net}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `warehouse_share_article_13` | Network | **Реализован** | $\text{clip}(\text{own}_{13} / (\text{net}_{13} + 10^{-4}), 0, 1)$ | $t \le \text{origin}-1$ |
| | `active_warehouses_article_13` | Network | **Реализован** | Число складов сети с активными продажами артикула за 13 нед. | $t \le \text{origin}-1$ |
| **D. Контекст категории**| `category_sales_mean_13_log` | Category | **Реализован** | `signed_log1p(mean_13(category_sales))` | $t \le \text{origin}-1$ |
| | `category_growth_4_vs_13` | Category | **Реализован** | $\text{clip}(\text{cat}_4 / (\text{cat}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `article_share_category_13` | Category | **Реализован** | $\text{clip}(\text{own}_{13} / (\text{cat}_{13} + 10^{-4}), 0, 1)$ | $t \le \text{origin}-1$ |
| | `active_articles_category_13` | Category | **Реализован** | Число активных артикулов в категории за 13 нед. | $t \le \text{origin}-1$ |
| **E. Денежные признаки** | `sales_sum_log` | Исходный | **Реализован** | `signed_log1p(sales_sum)` | $t \le \text{origin}-1$ |
| | `discount_sum_log` | Исходный | **Реализован** | `signed_log1p(discount_sum)` | $t \le \text{origin}-1$ |
| | `discount_share` | Производный | **Реализован** | $\text{discount} / (\text{sales} + \text{discount} + 10^{-4})$ | $t \le \text{origin}-1$ |
| | `paid_value_per_unit` | Производный | **Реализован** | $\text{sales} / (\text{quantity} + 10^{-4})$ | $t \le \text{origin}-1$ |
| **F. Календарь и события**| `cal_month` | Календарь | **Реализован** | $\text{month} / 12$ (`known_covariate`) | Известен на $52+9$ недель |
| | `cal_week_sin` / `cos` | Календарь | **Реализован** | $\sin / \cos (2\pi \cdot \text{week} / 52.18)$ | Известен на $52+9$ недель |
| | `cal_holiday_share` | Праздники | **Реализован** | Доля нерабочих/праздничных дней РФ | Известен на $52+9$ недель |
| | `cal_is_holiday` | Праздники | **Реализован** | Индикатор наличия праздничного дня | Известен на $52+9$ недель |
| | `cal_is_school` | События | **Реализован** | Флаг сезона «Снова в школу» (нед. 34–36) | Известен на $52+9$ недель |
| **G. Годовая сезонность**| `quantity_lag_51` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=51))` | $t \le \text{origin}-51$ |
| | `quantity_lag_52` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=52))` | $t \le \text{origin}-52$ |
| | `quantity_lag_53` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=53))` | $t \le \text{origin}-53$ |
| **H. Категориальные коды**| `target_enc_category_lvl_1` | Статический | **Реализован** | Сглаженный pre-cutoff таргетинг категории 1 | Строго до $\text{origin}$ |
| | `target_enc_category_lvl_2` | Статический | **Реализован** | Сглаженный pre-cutoff таргетинг категории 2 | Строго до $\text{origin}$ |
| | `target_enc_warehouse` | Статический | **Реализован** | Сглаженный pre-cutoff таргетинг склада | Строго до $\text{origin}$ |

---

## 2. Верификация и автоматическое тестирование

1. **Автоматический тест соответствия:** Реализован в [`test_timestral_feature_registry_parity.py`](file:///c:/Users/lenya/OneDrive/Рабочий%20стол/hak/test_timestral_feature_registry_parity.py).
   - Тест выполняет сверку на нескольких временных срезах ($\text{origins} = 89, 107, 125$).
   - Проверяются 100% каналов на корректность размерностей, соответствие именований и полное отсутствие `NaN` и `Inf`.
2. **Временная валидность (Zero Leakage):** Ни один признак не использует данные из будущего (будущие промо, поставки или объемы продаж). В `known_covariates` входят исключительно календарные маркеры и утвержденный производственный календарь РФ.
3. **Статус гипотезы о RMSE 11.08:** В раннем скрининге Univariate Baseline (120 шагов) сырые построчные прогнозы не сохранялись на диск. Математическое моделирование выброса 3000 в `check_outlier_impact.py` является иллюстративным доказательством чувствительности метрики $L_2$, а не верификацией конкретного значения в том запуске. В гармонизированном 600-шаговом протоколе Univariate RMSE составляет **3.0167**, а D+C — **2.9636**.
