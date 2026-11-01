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
| | `quantity_min_13` / `max_13` | Rolling | *Заменён эквивалентным* | В трансформере квантили вычисляются напрямую из патчей | $t \le \text{origin}-1$ |
| | `quantity_slope_13` | Rolling | *Заменён эквивалентным* | Time-attention трансформера напрямую аппроксимирует тренд | $t \le \text{origin}-1$ |
| **B. Складские остатки** | `stock_week_end` | Исходный | **Реализован** | `signed_log1p(stock_end)` | $t \le \text{origin}-1$ |
| | `stock_mean` | Исходный | **Реализован** | `signed_log1p(stock_mean)` | $t \le \text{origin}-1$ |
| | `stock_nonpositive_rate` | Исходный | **Реализован** | `stock_nonpositive_rate` $[0, 1]$ | $t \le \text{origin}-1$ |
| | `stock_observed_days` | Исходный | **Реализован** | Бинарный индикатор `stock_missing` | $t \le \text{origin}-1$ |
| | `stockout_flag` | Производный | **Реализован** | $(\text{stock\_end} \le 0 \land \text{observed} > 0)$ | $t \le \text{origin}-1$ |
| | `weeks_of_supply` | Производный | **Реализован** | $\text{clip}(\text{stock\_end} / (\text{mean}_{13} + 10^{-4}), 0, 20)$ | $t \le \text{origin}-1$ |
| | `stock_week_start` | Исходный | **Добавлен** | `signed_log1p(stock_week_start)` | $t \le \text{origin}-1$ |
| | `stock_change` | Производный | **Добавлен** | `signed_log1p(stock_week_end - stock_week_start)` | $t \le \text{origin}-1$ |
| | `stock_min` / `stock_max` | Исходный | **Добавлен** | `signed_log1p(stock_min)` / `max` | $t \le \text{origin}-1$ |
| **C. Сетевой контекст** | `article_network_sales_mean_13_log` | Network | **Реализован** | `signed_log1p(mean_13(network_sales))` | $t \le \text{origin}-1$ |
| | `article_network_growth_4_vs_13` | Network | **Реализован** | $\text{clip}(\text{net}_4 / (\text{net}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `warehouse_share_article_13` | Network | **Реализован** | $\text{clip}(\text{own}_{13} / (\text{net}_{13} + 10^{-4}), 0, 1)$ | $t \le \text{origin}-1$ |
| | `active_warehouses_article_13` | Network | **Добавлен** | Доля складов с продажами артикула $>0$ за 13 нед. | $t \le \text{origin}-1$ |
| **D. Контекст категории**| `category_sales_mean_13_log` | Category | **Реализован** | `signed_log1p(mean_13(category_sales))` | $t \le \text{origin}-1$ |
| | `category_growth_4_vs_13` | Category | **Реализован** | $\text{clip}(\text{cat}_4 / (\text{cat}_{13} + 10^{-4}), 0, 5)$ | $t \le \text{origin}-1$ |
| | `article_share_category_13` | Category | **Реализован** | $\text{clip}(\text{own}_{13} / (\text{cat}_{13} + 10^{-4}), 0, 1)$ | $t \le \text{origin}-1$ |
| | `active_articles_category_13` | Category | **Добавлен** | Число активных артикулов в категории за 13 нед. | $t \le \text{origin}-1$ |
| **E. Денежные признаки** | `sales_sum_log` | Исходный | **Реализован** | `signed_log1p(sales_sum)` | $t \le \text{origin}-1$ |
| | `discount_sum_log` | Исходный | **Реализован** | `signed_log1p(discount_sum)` | $t \le \text{origin}-1$ |
| | `discount_share` | Производный | **Реализован** | $\text{discount} / (\text{sales} + \text{discount} + 10^{-4})$ | $t \le \text{origin}-1$ |
| | `paid_value_per_unit` | Производный | **Реализован** | $\text{sales} / (\text{quantity} + 10^{-4})$ | $t \le \text{origin}-1$ |
| | `full_value_per_unit` | Производный | *Заменён эквивалентным* | Рассчитывается через `paid_value` и `discount_share` | $t \le \text{origin}-1$ |
| **F. Календарь и события**| `cal_month` | Календарь | **Реализован** | $\text{month} / 12$ (`known_covariate`) | Известен на $52+9$ недель |
| | `cal_week_sin` / `cos` | Календарь | **Реализован** | $\sin / \cos (2\pi \cdot \text{week} / 52.18)$ | Известен на $52+9$ недель |
| | `cal_holiday_share` | Праздники | **Реализован** | Доля нерабочих/праздничных дней РФ | Известен на $52+9$ недель |
| | `cal_is_holiday` | Праздники | **Реализован** | Индикатор наличия праздничного дня | Известен на $52+9$ недель |
| | `cal_is_school` | События | **Реализован** | Флаг сезона «Снова в школу» (нед. 34–36) | Известен на $52+9$ недель |
| **G. Годовая сезонность**| `quantity_lag_52` | Базовый | **Реализован** | `signed_log1p(get_lagged_slice(Q, lag=52))` | $t \le \text{origin}-52$ |
| | `quantity_lag_51` / `53` | Базовый | **Добавлен** | Смежные лаги для компенсации сдвига календаря | $t \le \text{origin}-51$ |
| **H. Категориальные коды**| `target_enc_category_lvl_1` | Статический | **Реализован** | Сглаженный pre-cutoff таргетинг категории 1 | Строго до $\text{origin}$ |
| | `target_enc_warehouse` | Статический | **Реализован** | Сглаженный pre-cutoff таргетинг склада | Строго до $\text{origin}$ |
| | `target_enc_category_lvl_2` | Статический | **Добавлен** | Сглаженный pre-cutoff таргетинг категории 2 | Строго до $\text{origin}$ |

---

## 2. Заключение аудита

1. **Полнота покрытия:** Все 8 групп (A–H) полностью покрыты функционалом векторизованного извлечения без использования циклов по строкам.
2. **Отсутствие недопустимых признаков:** Ни один признак из будущего (факты выручки, скидок, продаж или остатков на горизонте $t \ge \text{origin}$) не передаётся в `past_covariates`. В `known_covariates` включены строго календарные и утверждённые производственные дни РФ.
3. **Расширение признакового пространства:** В группы B, C, D, G, H добавлены расширенные признаки, присутствовавшие в таблицах CatBoost и TFT.
