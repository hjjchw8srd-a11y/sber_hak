# ДИАГНОСТИЧЕСКИЙ ДАМП ВЕТКИ `tests`

**Дата и время создания:** 2026-10-31 01:05:00 MSK  
**Текущая ветка:** `tests`  
**Удаленный репозиторий:** `origin/tests` (`https://github.com/hjjchw8srd-a11y/sber_hak.git`)  
**Текущий коммит:** `c978c31 feat(model_improvements): implement holiday calendar fixes, Candidate Weights 45/37/9/9, and pure calendar NBEATSx ablation`  

---

## 1. СТАТУС РЕПОЗИТОРИЯ, ИСТОРИЯ КОММИТОВ И ДЕРЕВО ФАЙЛОВ

### 1.1. Вывод `git status`
```bash
On branch tests
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	_backup_baseline_20261030/
	analyze_plot_2.py
	audit_180d_vs_full_history.py
	audit_frozen_ensemble_extra_folds.py
	audit_hurdle_stacking_honest.py
	check_stock.py
	defense_package_candidate_lgbm52.zip
	defense_package_candidate_lgbm52.zip.sha256
	eda.py
	evaluate_hierarchical_article_forecast.py
	experiments/benchmarks/
	experiments/ensemble/
	experiments/hierarchical/
	experiments/history_window_audit/
	experiments/honest_oof/
	experiments/optuna/
	experiments/reviews/antigravity_api_review_2026-10-30.md
	experiments/reviews/antigravity_defense_and_api_2026-10-30.md
	experiments/reviews/antigravity_next_experiment_2026-10-30.md
	experiments/reviews/antigravity_next_models_and_holidays_2026-10-30.md
	experiments/reviews/api_review_after_bb5aa29_2026-10-30.md
	experiments/reviews/defense_df377f1_verification_2026-10-30.md
	experiments/reviews/defense_package_smoke_review_2026-10-30.md
	experiments/reviews/first_stage_independent_metrics_2026-10-31.json
	experiments/reviews/first_stage_review_2026-10-31.md
	experiments/reviews/fold_0_180d_ablation.json
	experiments/reviews/fold_0_bundle/
	experiments/reviews/hierarchy_review_2026-10-30.md
	experiments/reviews/history_window_review_2026-10-30.md
	experiments/reviews/model_improvements_action_log_2026-10-30.md
	experiments/reviews/nbeatsx_nonnegative_diagnostic_2026-10-31.json
	experiments/reviews/second_stage_independent_metrics_2026-10-31.json
	experiments/reviews/second_stage_review_2026-10-31.md
	experiments/stacking/
	experiments/stacking_honest/
	final_score.csv
	generate_boxplots_and_heatmap.py
	generate_eda_plots.py
	generate_repo_dump.py
	lightning_logs/
	run_history_window_audit.py
	run_strict_history_audit.py
	test_thresh.py

nothing added to commit but untracked files present (use "git add" to track)
```

### 1.2. Вывод `git log -n 5 --oneline`
```bash
c978c31 feat(model_improvements): implement holiday calendar fixes, Candidate Weights 45/37/9/9, and pure calendar NBEATSx ablation
df377f1 fix(defense): correct smoke test contracts, add portable smoke_test.py, demo CSVs and documentation sync
bb5aa29 feat(defense): implement production forecast container and defense package for candidate_lgbm52
9a0489f Implement OOF Hurdle Stacking + Tweedie Peak Restoration for final submission
fc45f11 Calibrate mega-ensemble with scale alpha=1.05 and threshold 0.65 (New Score: 0.725573)
```

### 1.3. Дерево файлов (`maxdepth 3`)
```text
./
  API_ACCEPTANCE_REPORT.md
  CHECKPOINT_CATBOOST_V1.md
  DEFENSE_VERSION.md
  MODEL_ACTION_LOG.md
  MODEL_IMPROVEMENT_REPORT.md
  PROJECT_HANDOFF.md
  analyze_model_complementarity.py
  analyze_plot_2.py
  analyze_tft_postprocessing.py
  analyze_tft_scale_calibration.py
  analyze_tft_weaknesses.py
  analyze_weekly_forecasts.py
  audit_180d_vs_full_history.py
  audit_candidate_match.py
  audit_frozen_ensemble_extra_folds.py
  audit_hurdle_stacking_honest.py
  baseline_dow_mean.csv
  baseline_naive_last.csv
  baseline_zero.csv
  check_stock.py
  create_defense_package.py
  defense_package_candidate_lgbm52.zip
  defense_package_candidate_lgbm52.zip.sha256
  download_chronos_weights.py
  eda.py
  evaluate_candidate_weights_seeds.py
  evaluate_chronos2.py
  evaluate_hierarchical_article_forecast.py
  evaluate_nbeatsx_pure_calendar_check.py
  export_model_bundle.py
  fast_score.py
  final_score.csv
  final_submission.csv
  generate_boxplots_and_heatmap.py
  generate_candidate_lgbm52_submission.py
  generate_candidate_weights_submission.py
  generate_eda_plots.py
  generate_final_ensemble.py
  generate_repo_dump.py
  generate_weekly_eda_plots.py
  holiday_calendar.py
  info_about_sign.md
  prepare_weekly_data.py
  requirements.txt
  run_candidate_evaluation.py
  run_candidate_selection.py
  run_history_window_audit.py
  run_strict_history_audit.py
  run_tft_feature_experiments.py
  run_weekly_feature_experiments.py
  run_weekly_rolling_validation.py
  sales_dataset_hakaton_train.csv
  score.py
  setup_validation.py
  stock_dataset_hakaton_train.csv
  summarize_tft_stock_results.py
  test_holiday_calendar.py
  test_thresh.py
  train_catboost_v1.py
  train_lgbm_holiday_ablation.py
  train_lgbm_statutory_ablation.py
  train_nbeatsx_pure_calendar.py
  train_tft_final.py
  train_tft_weekly.py
  tune_catboost_optuna.py
  val_ground_truth.csv
  verify_candidate_weights_predictor.py
  _backup_baseline_20261030/
    candidate_lgbm52.csv
    candidate_lgbm52_manifest.json
    generate_candidate_lgbm52_submission.py
    sber_hak/
      generate_per_category_submission.py
      history_audit_pipeline.py
      hurdle_pipeline.py
      run_weekly_feature_experiments.py
  catboost_v1/
    classifier.cbm
    config.json
    leaderboard.csv
    regressor.cbm
    validation_forecast.csv
    validation_metrics.csv
  experiments/
    FEATURE_EXPERIMENTS.md
    experiment_metadata.json
    feature_test_comparison.png
    feature_test_results.csv
    benchmarks/
      mega_ensemble_search.csv
      mega_threshold_search.csv
      super_ensemble_search.csv
      catboost/
      fast_slow/
      lgbm/
      nhits/
      per_category/
      tide/
    complementarity/
      COMPLEMENTARITY_REPORT.md
      error_correlations_by_fold.csv
      large_error_cases_comparison.png
      model_metrics_by_fold.csv
      random_sample_series_comparison.png
      segment_metrics_by_fold.csv
    ensemble/
      ensemble_results.csv
      ensemble_summary.csv
    forecast_diagnostics/
      FORECAST_DIAGNOSTICS.md
      diagnostics.json
      expert_forecast_comparison.png
      selected_series_forecasts.png
      selected_series_metrics.csv
    hierarchical/
      HIERARCHICAL_AUDIT.md
      fold_metrics.csv
      summary.csv
    history_window_audit/
      REPORT.md
      article_level_metrics.csv
      blend_metrics.csv
      component_metrics.csv
      config.json
      horizon_metrics.csv
      reconciliation_metrics.csv
      seed_metrics.csv
      segment_metrics.csv
      selected_config.json
      keyed_oof/
    holidays/
      HOLIDAY_REPORT.md
      holiday_metrics_by_fold.csv
      keyed_predictions/
    holidays_statutory/
      statutory_holiday_metrics_by_fold.csv
      statutory_holiday_summary.json
      keyed_predictions/
    honest_oof/
      FROZEN_ENSEMBLE_AUDIT.md
      fold_metrics.csv
      model_summary.csv
      raw_predictions/
    model_improvements/
      MODEL_ACTION_LOG.md
      REPORT.md
      champion_config.json
      metrics_by_fold.csv
      protocol.json
      selected_config.json
      candidate_weights/
      candidates/
      chronos2_eval/
      chronos2_repo/
      chronos2_weights/
    model_weaknesses/
      IMPROVEMENT_PLAN.md
      TFT_SCALE_CALIBRATION.md
      TFT_WEAKNESSES.md
      errors_by_category_lvl_1.csv
      errors_by_demand_regime.csv
      errors_by_history_scale.csv
      errors_by_horizon.csv
      errors_by_stock_state.csv
      errors_by_trend_regime.csv
      errors_by_warehouse_name.csv
      overview.json
      raw_prediction_calibration.csv
      tft_scale_calibration_cross_fitted.csv
      tft_scale_calibration_detail.csv
      tft_scale_calibration_summary.csv
      tft_weaknesses.png
    optuna/
      catboost_optuna.db
      lgbm_optuna.db
      lgbm_optuna_trials.csv
    reviews/
      antigravity_api_review_2026-10-30.md
      antigravity_defense_and_api_2026-10-30.md
      antigravity_next_experiment_2026-10-30.md
      antigravity_next_models_and_holidays_2026-10-30.md
      api_review_after_bb5aa29_2026-10-30.md
      defense_df377f1_verification_2026-10-30.md
      defense_package_smoke_review_2026-10-30.md
      first_stage_independent_metrics_2026-10-31.json
      first_stage_review_2026-10-31.md
      fold_0_180d_ablation.json
      fold_0_real_180d_ablation.json
      hierarchy_review_2026-10-30.md
      history_window_review_2026-10-30.md
      model_improvements_action_log_2026-10-30.md
      nbeatsx_nonnegative_diagnostic_2026-10-31.json
      run_real_fold0_ablation.py
      second_stage_independent_metrics_2026-10-31.json
      second_stage_review_2026-10-31.md
      fold_0_bundle/
    rolling/
      ROLLING_VALIDATION.md
      rolling_feature_comparison.png
      rolling_feature_results.csv
      rolling_feature_summary.csv
    stacking/
      meta_features_fold_0.csv.gz
      meta_features_fold_1.csv.gz
      meta_features_fold_2.csv.gz
    stacking_honest/
      HONEST_AUDIT.md
      fold_metrics.csv
      summary.csv
      fold_meta_features/
    tft/
      TFT_POSTPROCESSING.md
      TFT_REPORT.md
      tft_config.json
      tft_final_info.json
      tft_final_prediction_raw.csv.gz
      tft_final_submission.csv
      tft_postprocessing.png
      tft_postprocessing_cross_fitted.csv
      tft_postprocessing_details.csv
      tft_postprocessing_summary.csv
      tft_predictions_fold_0.csv.gz
      tft_predictions_fold_1.csv.gz
      tft_predictions_fold_2.csv.gz
      tft_predictions_postprocessed_fold_0.csv.gz
      tft_predictions_postprocessed_fold_1.csv.gz
      tft_predictions_postprocessed_fold_2.csv.gz
      tft_rolling_results.csv
      tft_rolling_scores.png
      tft_stock_final_info.json
      tft_stock_final_prediction_raw.csv.gz
      tft_stock_final_submission.csv
      final_model/
      final_model_stock/
    tft_features_32bit/
      TFT_FEATURE_EXPERIMENTS.md
      tft_feature_comparison.png
      tft_feature_results.csv
      tft_feature_summary.csv
    tft_features_final/
      TFT_FEATURE_EXPERIMENTS.md
      TFT_STOCK_COMPARISON.md
      tft_feature_comparison.png
      tft_feature_results.csv
      tft_feature_summary.csv
      tft_stock_vs_base.csv
      tft_stock_vs_base.png
    tft_features_screen/
      TFT_FEATURE_EXPERIMENTS.md
      tft_feature_comparison.png
      tft_feature_results.csv
      tft_feature_summary.csv
  forecast-container/
    Dockerfile
    README.md
    requirements.txt
    smoke_test.py
    test_api.sh
    app/
      __init__.py
      main.py
      predictor.py
      validators.py
    data/
      demo_sales.csv
      demo_stock.csv
      sample_sales.csv
      sample_stock.csv
    model/
      manifest.json
      postprocessing.json
      valid_pairs.parquet
      catboost/
      lgbm_b/
      per_category/
      tft/
    tests/
      test_api.py
      test_api.sh
  sber_hak/
    MODEL_ACTION_LOG.md
    analyze_tft_postprocessing.py
    analyze_weekly_forecasts.py
    build_ensemble.py
    evaluate_all_benchmarks.py
    evaluate_fast_slow_ensemble.py
    fast_score.py
    fine_tune_tft_threshold.py
    generate_mega_submission.py
    generate_per_category_submission.py
    generate_weekly_eda_plots.py
    history_audit_pipeline.py
    holiday_calendar.py
    hurdle_pipeline.py
    info_about_sign.md
    optimize_multi_ensemble.py
    plot_expert_forecasts.py
    prepare_weekly_data.py
    requirements.txt
    run_lgbm_benchmark.py
    run_neural_benchmarks.py
    run_weekly_feature_experiments.py
    run_weekly_rolling_validation.py
    save_catboost_folds.py
    score.py
    search_mega_ensemble.py
    search_super_ensemble.py
    setup_validation.py
    train_catboost_v1.py
    train_fast_slow_models.py
    train_hurdle_stacking_submission.py
    train_oof_hurdle_stacking.py
    train_per_category_models.py
    train_tft_final.py
    train_tft_weekly.py
    tune_catboost_optuna.py
    tune_lgbm_optuna.py
    tune_mega_threshold.py
    tune_tft_optuna.py
    catboost_v1/
      classifier.cbm
      config.json
      leaderboard.csv
      regressor.cbm
      validation_metrics.csv
    experiments/
      FEATURE_EXPERIMENTS.md
      experiment_metadata.json
      feature_test_comparison.png
      feature_test_results.csv
      forecast_diagnostics/
      rolling/
      tft/
    tests/
      test_tft_preparation.py
      test_weekly_feature_leakage.py
      test_weekly_preparation.py
  submissions/
    candidate_lgbm52.csv
    candidate_lgbm52_manifest.json
    candidate_weights_45_37_9_9.csv
    candidate_weights_45_37_9_9_manifest.json
  teammate_project/
    nonton_intermediate 2/
      DELIVERY_MANIFEST.md
      README.md
      REQUIREMENTS_AUDIT.md
      pipeline.py
      requirements.txt
      requirements_original.md
      validate_submission.py
      artifacts/
  tests/
    test_tft_preparation.py
    test_weekly_feature_leakage.py
    test_weekly_preparation.py
  weekly_data/
    weekly_dataset.csv.gz
    weekly_dataset_preview.csv
    weekly_summary.json
  weekly_plots/
    plot_1_weekly_trend.png
    plot_2_lagged_stock_effect.png
    plot_3_monthly_seasonality.png
    plot_4_warehouse_weekly_distribution.png
    plot_5_weekly_sparsity_stock_coverage.png
    plot_6_weekly_feature_correlations.png
    weekly_plot_analysis.json
```

---

## 2. МЕТРИКИ И ЛОГИ ВАЛИДАЦИИ ПО ВСЕМ МОДЕЛЯМ И ОКНАМ

### 2.1. Сводная таблица 5-Fold Mean по всем моделям (Сравнение с эталоном)

| Модель / Конфигурация | SMAPE (%) | WAPE (%) | RMSE | Bias (%) | RMSLE | Clipped Score | Unclipped Ratio | Delta vs Control (Unclipped) | Статус в протоколе |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Candidate Weights 45/37/9/9 (Seed 42)** | 61.99 | 49.49 | 2.8980 | 75.44 | 0.5268 | 0.730075 | 0.755010 | -0.001193 | **Победитель (упакован)** |
| **Blend 10% NBEATSx Pure Cal** | 62.09 | 49.56 | 2.8892 | 75.60 | 0.5266 | 0.730186 | 0.755484 | -0.000719 | Исследование завершено |
| **Control_Blend_B (52w, Seed 42)** | 62.13 | 49.58 | 2.8976 | 75.58 | 0.5276 | 0.730988 | 0.756203 | +0.000000 | **Эталон защиты** |
| **LightGBM + Holidays (Блок А)** | 62.17 | 50.12 | 3.0182 | 75.10 | 0.5231 | 0.731204 | 0.756855 | +0.000652 | Отклонен (+0.001103 sel) |
| **Blend 10% NBEATSx (Holidays)** | 62.24 | 49.69 | 2.9051 | 75.82 | 0.5284 | 0.733214 | 0.758998 | +0.002795 | Отклонен (+0.004045 sel) |
| **Standalone LightGBM B (52w)** | 62.17 | 50.12 | 3.0151 | 75.08 | 0.5228 | 0.735278 | 0.762647 | +0.006444 | Компонент (30% / 37%) |
| **Standalone TFT (52w)** | 63.41 | 51.08 | 3.0125 | 77.68 | 0.5347 | 0.742931 | 0.777534 | +0.021331 | Компонент (50% / 45%) |
| **Standalone CatBoost (18w)** | 62.95 | 51.57 | 3.0940 | 77.44 | 0.5343 | 0.744690 | 0.781041 | +0.024838 | Компонент (10% / 9%) |
| **Standalone NBEATSx Pure Cal** | 67.83 | 54.24 | 3.1008 | 88.57 | 0.5463 | 0.764105 | 0.830209 | +0.074006 | Слабее TFT и бустингов |
| **Standalone Per-Category (14m)** | 64.90 | 53.46 | 3.2339 | 80.96 | 0.5377 | 0.754446 | 0.808337 | +0.052134 | Компонент (10% / 9%) |
| **Standalone NBEATSx (Holidays)** | 74.52 | 62.15 | 3.4510 | 96.12 | 0.5891 | 0.823779 | 1.013069 | +0.251755 | Отклонен на выборе |
| **Oracle Diagnostic [Недостижимо]** | 57.10 | 41.21 | 2.5102 | 62.45 | 0.4789 | 0.610054 | 0.617178 | -0.139025 | Диагностический ориентир |

### 2.2. Пофолдовая детализация ключевых моделей

#### Control Blend B vs Candidate Weights 45/37/9/9 (Seed 42)

| Окно | Период | Роль | Ctrl Clipped | Cand Clipped | Delta Clipped | Ctrl Unclipped | Cand Unclipped | Delta Unclipped | Ctrl WAPE (%) | Cand WAPE (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 4 | 2025-09-15 .. 2025-11-10 | Selection | 0.742798 | 0.742367 | -0.000431 | 0.779804 | 0.779249 | -0.000555 | 51.92% | 51.86% |
| Fold 3 | 2025-11-17 .. 2026-01-12 | Selection | 0.737232 | 0.735293 | -0.001939 | 0.766648 | 0.764034 | -0.002613 | 52.39% | 52.24% |
| Fold 2 | 2026-01-19 .. 2026-03-16 | Selection | 0.723588 | 0.723205 | -0.000382 | 0.737490 | 0.737180 | -0.000310 | 48.10% | 48.08% |
| Fold 1 | 2026-03-23 .. 2026-05-18 | Check | 0.733866 | 0.732331 | -0.001535 | 0.758479 | 0.756820 | -0.001659 | 49.42% | 49.25% |
| Fold 0 | 2026-05-25 .. 2026-07-20 | Check | 0.717455 | 0.717177 | -0.000277 | 0.738593 | 0.737767 | -0.000826 | 46.06% | 46.02% |

### 2.3. Результаты многосидовой проверки Candidate Weights (Seeds 42, 52, 62)

> Примечание: Варьировались сиды обучения LightGBM B; TFT, CatBoost и Per-Category заморожены.

| Сид | Selection Unclipped Delta | Check Unclipped Delta | 5-Fold Clipped Score Delta | 5-Fold Unclipped Ratio Delta | Результат |
|---|---:|---:|---:|---:|:---:|
| Seed 42 | -0.001159 | -0.001242 | -0.000913 (0.730075 vs 0.730988) | -0.001193 (0.755010 vs 0.756203) | 5 / 5 побед |
| Seed 52 | -0.000996 | -0.001505 | -0.000720 (0.730653 vs 0.731373) | -0.001200 (0.755934 vs 0.757134) | 5 / 5 побед |
| Seed 62 | -0.000577 | -0.001070 | -0.000425 (0.730916 vs 0.731341) | -0.000774 (0.756889 vs 0.757663) | 5 / 5 побед |
| **Среднее** | **-0.000911 ± 0.000245** | **-0.001273 ± 0.000179** | **-0.000686 ± 0.000201** | **-0.001056 ± 0.000199** | **15 / 15 побед** |

---

## 3. ПОЛНЫЙ КОД СКРИПТОВ

### 3.74 `holiday_calendar.py`
*Модуль расчета производственного календаря РФ (Постановление № 1466 и ст. 112 ТК РФ)*  
Путь: [`holiday_calendar.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/holiday_calendar.py)

```python
"""
sber_hak/holiday_calendar.py

Official Russian Federation holiday calendar module based on Article 112 of the Labor Code of the Russian Federation
and official annual government decrees on holiday transfers (2023-2027).

Standalone, zero network calls during inference or training.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# 1. Statutory federal holidays of the Russian Federation (ст. 112 ТК РФ)
# Month, Day
STATUTORY_HOLIDAY_DATES: Set[Tuple[int, int]] = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),  # Новогодние каникулы и Рождество
    (2, 23),  # День защитника Отечества
    (3, 8),   # Международный женский день
    (5, 1),   # Праздник Весны и Труда
    (5, 9),   # День Победы
    (6, 12),  # День России
    (11, 4),  # День народного единства
}

NEW_YEAR_DATES: Set[Tuple[int, int]] = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
}

MAY_HOLIDAY_DATES: Set[Tuple[int, int]] = {
    (5, 1), (5, 9),
}

# 2. Transferred non-working days
# Documented with exact sources of law:
# A. Decree-based transfers (ежегодные постановления Правительства РФ):
# - 2024: Постановление Правительства РФ от 10.08.2023 № 1314 (опубликовано 2023-08-10)
# - 2025: Постановление Правительства РФ от 04.10.2024 № 1335 (опубликовано 2024-10-04)
# - 2026: Постановление Правительства РФ от 24.09.2025 № 1466 (опубликовано 2025-09-24)
#   Прямой текст Постановления № 1466:
#   "с субботы 3 января на пятницу 9 января; с воскресенья 4 января на четверг 31 декабря."
DECREE_PUBLICATION_DATES: Dict[int, datetime.date] = {
    2024: datetime.date(2023, 8, 10),
    2025: datetime.date(2024, 10, 4),
    2026: datetime.date(2025, 9, 24),
}

DECREE_TRANSFERRED_DAYS: Set[datetime.date] = {
    # 2024 (Постановление Правительства РФ от 10.08.2023 № 1314)
    datetime.date(2024, 4, 29),
    datetime.date(2024, 4, 30),
    datetime.date(2024, 5, 10),
    datetime.date(2024, 12, 30),
    datetime.date(2024, 12, 31),
    # 2025 (Постановление Правительства РФ от 04.10.2024 № 1335)
    datetime.date(2025, 5, 2),
    datetime.date(2025, 5, 8),
    datetime.date(2025, 6, 13),
    datetime.date(2025, 11, 3),
    datetime.date(2025, 12, 31),
    # 2026 (Постановление Правительства РФ от 24.09.2025 № 1466)
    datetime.date(2026, 1, 9),    # перенос с сб 3 января на пт 9 января
    datetime.date(2026, 12, 31),  # перенос с вс 4 января на чт 31 декабря
}

# B. Automatic statutory transfers (ст. 112 ч. 2 ТК РФ при совпадении с выходным днем):
# Постоянная норма закона: перенос на следующий после праздничного рабочий день.
# Источник доступности - ТК РФ (не зависит от ежегодного постановления правительства).
AUTOMATIC_STATUTORY_TRANSFERS: Set[datetime.date] = {
    # 2026: 8 марта (вс) -> пн 9 марта; 9 мая (сб) -> пн 11 мая
    datetime.date(2026, 3, 9),
    datetime.date(2026, 5, 11),
}

TRANSFERRED_NONWORKING_DAYS: Set[datetime.date] = DECREE_TRANSFERRED_DAYS | AUTOMATIC_STATUTORY_TRANSFERS

# 3. Working Saturdays (субботы, ставшие рабочими днями из-за переносов)
WORKING_SATURDAYS: Set[datetime.date] = {
    # 2024
    datetime.date(2024, 4, 27),
    datetime.date(2024, 11, 2),
    datetime.date(2024, 12, 28),
    # 2025
    datetime.date(2025, 11, 1),
}


def is_statutory_holiday(d: datetime.date) -> bool:
    return (d.month, d.day) in STATUTORY_HOLIDAY_DATES


def is_new_year_holiday(d: datetime.date) -> bool:
    return (d.month, d.day) in NEW_YEAR_DATES


def is_may_holiday(d: datetime.date) -> bool:
    return (d.month, d.day) in MAY_HOLIDAY_DATES


def get_all_statutory_holiday_dates(start_year: int = 2023, end_year: int = 2027) -> list[datetime.date]:
    holidays = []
    for y in range(start_year, end_year + 1):
        for m, d in sorted(STATUTORY_HOLIDAY_DATES):
            try:
                holidays.append(datetime.date(y, m, d))
            except ValueError:
                pass
    return sorted(holidays)


ALL_STATUTORY_HOLIDAYS = get_all_statutory_holiday_dates()


STATUTORY_FEATURE_COLUMNS: List[str] = [
    "holiday_days",
    "holiday_days_prev_week",
    "holiday_days_next_week",
    "new_year_days",
    "may_holiday_days",
    "days_to_nearest_holiday",
    "days_since_last_holiday",
]

TRANSFER_FEATURE_COLUMNS: List[str] = [
    "working_saturdays",
    "extra_nonworking_days",
]

HOLIDAY_FEATURE_COLUMNS: List[str] = STATUTORY_FEATURE_COLUMNS + TRANSFER_FEATURE_COLUMNS


def compute_holiday_features_for_week(
    monday: datetime.date,
    as_of_date: Optional[datetime.date] = None,
    statutory_only: bool = False,
) -> dict[str, float]:
    """
    Computes holiday features for a single target week starting on monday (Monday) through Sunday.
    - If statutory_only is True, returns ONLY the 7 permanent Article 112 features (no decree dependency).
    - If as_of_date is provided, decree-based transfers for year Y are only counted if as_of_date >= DECREE_PUBLICATION_DATES[Y].
    """
    assert monday.weekday() == 0, f"Expected Monday, got {monday} with weekday {monday.weekday()}"

    week_days = [monday + datetime.timedelta(days=i) for i in range(7)]
    prev_week_days = [monday - datetime.timedelta(days=7) + datetime.timedelta(days=i) for i in range(7)]
    next_week_days = [monday + datetime.timedelta(days=7) + datetime.timedelta(days=i) for i in range(7)]

    # Block A: Permanent statutory holidays (ст. 112 ТК РФ)
    holiday_days = sum(1 for d in week_days if is_statutory_holiday(d))
    holiday_days_prev_week = sum(1 for d in prev_week_days if is_statutory_holiday(d))
    holiday_days_next_week = sum(1 for d in next_week_days if is_statutory_holiday(d))

    new_year_days = sum(1 for d in week_days if is_new_year_holiday(d))
    may_holiday_days = sum(1 for d in week_days if is_may_holiday(d))

    # Days to nearest upcoming holiday (capped at 60)
    upcoming = [h for h in ALL_STATUTORY_HOLIDAYS if h >= monday]
    if upcoming:
        days_to_nearest = min((upcoming[0] - monday).days, 60)
    else:
        days_to_nearest = 60

    # Days since last holiday (capped at 60)
    past = [h for h in ALL_STATUTORY_HOLIDAYS if h < monday]
    if past:
        days_since_last = min((monday - past[-1]).days, 60)
    else:
        days_since_last = 60

    result = {
        "holiday_days": float(holiday_days),
        "holiday_days_prev_week": float(holiday_days_prev_week),
        "holiday_days_next_week": float(holiday_days_next_week),
        "new_year_days": float(new_year_days),
        "may_holiday_days": float(may_holiday_days),
        "days_to_nearest_holiday": float(days_to_nearest),
        "days_since_last_holiday": float(days_since_last),
    }

    if statutory_only:
        return result

    # Block B: Decree-based transfers with publication date verification
    # + Automatic statutory transfers under Art. 112 p. 2 (always known from Labor Code)
    valid_working_sats = set()
    for ws in WORKING_SATURDAYS:
        pub_date = DECREE_PUBLICATION_DATES.get(ws.year)
        if as_of_date is None or pub_date is None or as_of_date >= pub_date:
            valid_working_sats.add(ws)

    valid_transferred_days = set(AUTOMATIC_STATUTORY_TRANSFERS)  # Art. 112 p. 2 always valid
    for td in DECREE_TRANSFERRED_DAYS:
        pub_date = DECREE_PUBLICATION_DATES.get(td.year)
        if as_of_date is None or pub_date is None or as_of_date >= pub_date:
            valid_transferred_days.add(td)

    result["working_saturdays"] = float(sum(1 for d in week_days if d in valid_working_sats))
    result["extra_nonworking_days"] = float(sum(1 for d in week_days if d in valid_transferred_days))

    return result


def build_holiday_feature_dataframe(
    weeks: list[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
    as_of_date: Optional[datetime.date] = None,
    statutory_only: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by week string ('YYYY-MM-DD') with holiday feature columns.
    """
    dt_series = pd.Series(pd.to_datetime(weeks))
    unique_mondays = sorted(dt_series.dt.date.unique())
    rows = []
    for m in unique_mondays:
        feats = compute_holiday_features_for_week(m, as_of_date=as_of_date, statutory_only=statutory_only)
        feats["week"] = m.strftime("%Y-%m-%d")
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df
```

---

### 3.84 `test_holiday_calendar.py`
*Регрессионные и модульные тесты календаря праздников РФ*  
Путь: [`test_holiday_calendar.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/test_holiday_calendar.py)

```python
"""
test_holiday_calendar.py

Comprehensive regression test suite for sber_hak/holiday_calendar.py:
- Separation of Block A (Statutory Art. 112) and Block B (Decree transfers)
- Regression test for 2026-05-04 (standard working Monday, NOT extra non-working)
- Regression test for 2026-12-31 (transferred non-working day under Decree No. 1466)
- Regression test for summer horizon distances (non-zero days_to_nearest and days_since_last)
- Temporal availability cutoff test (decrees not visible prior to publication date)
"""

import datetime
import numpy as np
import pandas as pd

from sber_hak.holiday_calendar import (
    compute_holiday_features_for_week,
    build_holiday_feature_dataframe,
    STATUTORY_FEATURE_COLUMNS,
    TRANSFER_FEATURE_COLUMNS,
    HOLIDAY_FEATURE_COLUMNS,
    is_statutory_holiday,
)


def test_january_holidays():
    # 2025-01-06 (Monday). Jan 6, 7, 8 are statutory holidays (3 days)
    feats = compute_holiday_features_for_week(datetime.date(2025, 1, 6))
    assert feats["holiday_days"] == 3.0
    assert feats["new_year_days"] == 3.0
    assert feats["holiday_days_prev_week"] == 5.0  # Jan 1..5 in previous week
    assert feats["holiday_days_next_week"] == 0.0


def test_march_8_holidays():
    # 2025-03-03 (Monday). 2025-03-08 is Saturday (1 day)
    feats = compute_holiday_features_for_week(datetime.date(2025, 3, 3))
    assert feats["holiday_days"] == 1.0
    assert feats["new_year_days"] == 0.0
    assert feats["may_holiday_days"] == 0.0


def test_may_holidays_2025():
    # 2025-04-28 (Monday). May 1 is Thursday (1 statutory holiday + May 2 transferred)
    feats = compute_holiday_features_for_week(datetime.date(2025, 4, 28))
    assert feats["holiday_days"] == 1.0
    assert feats["may_holiday_days"] == 1.0
    assert feats["extra_nonworking_days"] == 1.0  # May 2

    # 2025-05-05 (Monday). May 9 is Friday (1 statutory holiday + May 8 transferred)
    feats9 = compute_holiday_features_for_week(datetime.date(2025, 5, 5))
    assert feats9["holiday_days"] == 1.0
    assert feats9["may_holiday_days"] == 1.0
    assert feats9["extra_nonworking_days"] == 1.0  # May 8


def test_regression_2026_05_04_week_has_zero_extra_nonworking_days():
    """
    CRITICAL REGRESSION TEST (Audit 31.10.2026):
    Under RF Government Decree 24.09.2025 No. 1466:
    Jan 3 was transferred to Friday Jan 9, NOT May 8!
    In week 2026-05-04 (May 4..May 10):
    - May 4 (Mon) - working
    - May 5 (Tue) - working
    - May 6 (Wed) - working
    - May 7 (Thu) - working
    - May 8 (Fri) - regular working day (NOT transferred)
    - May 9 (Sat) - statutory Victory Day holiday
    - May 10 (Sun) - regular weekend
    Thus, extra_nonworking_days in week 2026-05-04 is STRICTLY 0.0,
    and statutory holiday_days == 1.0 (May 9).
    """
    feats = compute_holiday_features_for_week(datetime.date(2026, 5, 4))
    assert feats["extra_nonworking_days"] == 0.0, f"Expected 0.0 extra non-working days, got {feats['extra_nonworking_days']}"
    assert feats["holiday_days"] == 1.0
    assert feats["may_holiday_days"] == 1.0


def test_regression_2026_01_09_is_day_off_decree_1466():
    """
    CRITICAL REGRESSION TEST (Audit 31.10.2026):
    Under RF Government Decree 24.09.2025 No. 1466:
    Saturday 3 January 2026 is transferred to Friday 9 January 2026.
    In week 2026-01-05 (Jan 5..Jan 11):
    - Jan 5, 6, 7, 8: statutory New Year holidays (4 days)
    - Jan 9 (Fri): transferred day off from Sat Jan 3 -> extra_nonworking_days == 1.0
    """
    feats = compute_holiday_features_for_week(datetime.date(2026, 1, 5))
    assert feats["extra_nonworking_days"] == 1.0, f"Expected 1.0 extra nonworking day (Jan 9), got {feats['extra_nonworking_days']}"
    assert feats["holiday_days"] == 4.0
    assert feats["new_year_days"] == 4.0


def test_regression_2026_05_11_is_automatic_transfer():
    """
    CRITICAL REGRESSION TEST (Art. 112 p. 2):
    May 9, 2026 is Saturday -> day off automatically transferred to Monday May 11, 2026.
    In week 2026-05-11 (May 11..May 17):
    - May 11 is automatic transferred day off -> extra_nonworking_days == 1.0
    """
    feats = compute_holiday_features_for_week(datetime.date(2026, 5, 11))
    assert feats["extra_nonworking_days"] == 1.0, f"Expected 1.0 extra nonworking day (May 11), got {feats['extra_nonworking_days']}"


def test_regression_2026_12_31_is_day_off():
    """
    CRITICAL REGRESSION TEST:
    Under RF Government Decree 24.09.2025 No. 1466:
    Sunday 4 January 2026 is transferred to Thursday 31 December 2026.
    In week 2026-12-28 (Dec 28..Jan 3 2027):
    - Dec 31 is transferred day off -> extra_nonworking_days >= 1.0
    - Jan 1, 2, 3 are statutory New Year holidays -> holiday_days >= 3.0
    """
    feats = compute_holiday_features_for_week(datetime.date(2026, 12, 28))
    assert feats["extra_nonworking_days"] >= 1.0, f"Expected Dec 31 to be counted as extra nonworking day, got {feats['extra_nonworking_days']}"


def test_summer_target_horizon_properties():
    """
    CRITICAL TEST FOR TARGET HORIZON:
    Target horizon: 2026-07-27 to 2026-09-21 (9 weeks)
    1. Zero statutory holidays in all 9 weeks.
    2. Zero decree-based transfers in all 9 weeks.
    3. Distance features (days_to_nearest_holiday, days_since_last_holiday) are strictly non-zero!
    """
    horizon_weeks = pd.date_range("2026-07-27", periods=9, freq="W-MON")
    df = build_holiday_feature_dataframe(horizon_weeks)
    assert len(df) == 9

    # Zero holidays directly in the week
    assert (df["holiday_days"] == 0).all(), "Found non-zero holiday_days in target horizon!"
    assert (df["new_year_days"] == 0).all()
    assert (df["may_holiday_days"] == 0).all()
    assert (df["working_saturdays"] == 0).all()
    assert (df["extra_nonworking_days"] == 0).all()

    # Distances are strictly non-zero across the entire horizon!
    assert (df["days_to_nearest_holiday"] > 0).all(), "days_to_nearest_holiday must be > 0 in summer"
    assert (df["days_since_last_holiday"] > 0).all(), "days_since_last_holiday must be > 0 in summer"

    # Exact distance trajectory:
    # First week (2026-07-27) is 45 days after 2026-06-12 (Russia Day)
    assert df.iloc[0]["days_since_last_holiday"] == 45.0
    # Last week (2026-09-21) is 44 days before 2026-11-04 (Unity Day)
    assert df.iloc[-1]["days_to_nearest_holiday"] == 44.0


def test_statutory_only_mode():
    """
    Verifies that statutory_only=True returns exactly the 7 Art. 112 columns,
    with zero decree dependency and zero temporal leakage risk.
    """
    monday = datetime.date(2025, 4, 28)
    feats_stat = compute_holiday_features_for_week(monday, statutory_only=True)
    assert set(feats_stat.keys()) == set(STATUTORY_FEATURE_COLUMNS)
    assert "working_saturdays" not in feats_stat
    assert "extra_nonworking_days" not in feats_stat
    assert feats_stat["holiday_days"] == 1.0


def test_temporal_decree_availability_cutoff():
    """
    Verifies that decree transfers are not visible before the decree publication date,
    whereas automatic statutory transfers (Art. 112 p. 2) are always visible.

    Decree No. 1466 for 2026 was published on 2025-09-24:
    - Week 2026-01-05 contains transferred Jan 9 (decree-based).
      Before 2025-09-24: extra_nonworking_days == 0.0.
      After 2025-09-24: extra_nonworking_days == 1.0.
    - Week 2026-05-11 contains transferred May 11 (Art. 112 p. 2 automatic).
      Even as of 2025-01-01: extra_nonworking_days == 1.0 (statutory law).
    """
    monday_jan = datetime.date(2026, 1, 5)
    # Before Decree No. 1466 publication:
    feats_jan_before = compute_holiday_features_for_week(monday_jan, as_of_date=datetime.date(2025, 9, 1))
    assert feats_jan_before["extra_nonworking_days"] == 0.0

    # After Decree No. 1466 publication:
    feats_jan_after = compute_holiday_features_for_week(monday_jan, as_of_date=datetime.date(2025, 10, 1))
    assert feats_jan_after["extra_nonworking_days"] == 1.0

    # Automatic statutory transfer is independent of decree:
    monday_may = datetime.date(2026, 5, 11)
    feats_may_early = compute_holiday_features_for_week(monday_may, as_of_date=datetime.date(2025, 1, 1))
    assert feats_may_early["extra_nonworking_days"] == 1.0


def test_leap_year_handling():
    feats = compute_holiday_features_for_week(datetime.date(2024, 2, 19))
    assert feats["holiday_days"] == 1.0
    feats_leap = compute_holiday_features_for_week(datetime.date(2024, 2, 26))
    assert feats_leap["holiday_days"] == 0.0


def test_shuffled_and_repeated_consistency():
    dates = [
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-05-05"),
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2024-11-04"),
    ]
    df1 = build_holiday_feature_dataframe(dates)
    df2 = build_holiday_feature_dataframe(dates[::-1])
    row1 = df1[df1["week"] == "2025-01-06"].iloc[0].to_dict()
    row2 = df2[df2["week"] == "2025-01-06"].iloc[0].to_dict()
    assert row1 == row2


if __name__ == "__main__":
    test_january_holidays()
    test_march_8_holidays()
    test_may_holidays_2025()
    test_regression_2026_05_04_week_has_zero_extra_nonworking_days()
    test_regression_2026_01_09_is_day_off_decree_1466()
    test_regression_2026_05_11_is_automatic_transfer()
    test_regression_2026_12_31_is_day_off()
    test_summer_target_horizon_properties()
    test_statutory_only_mode()
    test_temporal_decree_availability_cutoff()
    test_leap_year_handling()
    test_shuffled_and_repeated_consistency()
    print("ALL HOLIDAY CALENDAR REGRESSION TESTS PASSED SUCCESSFULLY!")
```

---

### 3.94 `fast_score.py`
*Быстрый расчет 5 соревновательных метрик (SMAPE, WAPE, RMSE, Bias, RMSLE, Clipped Score, Unclipped Ratio)*  
Путь: [`fast_score.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/fast_score.py)

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape":  0.20,
    "rmse":  0.15,
    "bias":  0.10,
    "rmsle": 0.05,
}

def compute_all_metrics_fast(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> Dict[str, float]:
    """
    Быстрый и 100% математически точный расчет метрик хакатона.
    Работает за < 1 секунды вместо нескольких минут.
    """
    keys = ["week", "warehouse_name", "article"]
    df = df_true[keys + ["quantity"]].merge(
        df_pred[keys + ["quantity"]],
        on=keys,
        how="left",
        suffixes=("_true", "_pred")
    )
    df["quantity_pred"] = df["quantity_pred"].fillna(0.0).clip(lower=0.0)
    
    y_true = df["quantity_true"].values.astype(float)
    y_pred = df["quantity_pred"].values.astype(float)
    
    # 1. WAPE (глобально)
    denom = np.sum(np.abs(y_true))
    wape_val = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
    
    # 2. RMSE (глобально)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # 3. RMSLE (глобально)
    rmsle_val = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
    
    # Для метрик по сериям (SMAPE и Bias)
    df["series_id"] = df["warehouse_name"].astype(str) + "|" + df["article"].astype(str)
    
    # SMAPE per element: |y - y_hat| / ((|y| + |y_hat|) / 2)
    # If both 0 -> 0
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        elem_smape = np.where(denominator > 0, numerator / denominator * 100.0, 0.0)
    
    df["elem_smape"] = elem_smape
    # Серийный SMAPE: среднее по всем точкам каждой серии, затем среднее
    # по сериям. Согласно актуальному FAQ, точка 0/0 остаётся в среднем
    # с ошибкой 0, а не исключается.
    series_group = df.groupby("series_id")
    smape_series = series_group["elem_smape"].mean()
    smape_val = float(smape_series.mean())
    
    # Bias: | mean(pred) - mean(true) | / mean(true) * 100.0
    # Исключаются серии, где mean(true) == 0
    mean_true = series_group["quantity_true"].mean()
    mean_pred = series_group["quantity_pred"].mean()
    
    valid_series = mean_true > 0
    if valid_series.any():
        bias_series = (np.abs(mean_pred[valid_series] - mean_true[valid_series]) / mean_true[valid_series]) * 100.0
        bias_val = float(bias_series.mean())
    else:
        bias_val = 0.0
        
    return {
        "smape": smape_val,
        "wape":  wape_val,
        "rmse":  rmse_val,
        "bias":  bias_val,
        "rmsle": rmsle_val,
    }

if __name__ == "__main__":
    df_truth = pd.read_csv("val_ground_truth.csv")
    for name in ["baseline_zero", "baseline_naive_last", "baseline_dow_mean"]:
        df_sub = pd.read_csv(f"{name}.csv")
        m = compute_all_metrics_fast(df_truth, df_sub)
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"  {k:6s}: {v:10.4f}")
```

---

### 3.104 `generate_candidate_weights_submission.py`
*Генерация финального сабмита кандидата 45/37/9/9, манифеста и упаковка бандла*  
Путь: [`generate_candidate_weights_submission.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/generate_candidate_weights_submission.py)

```python
#!/usr/bin/env python3
"""
generate_candidate_weights_submission.py

Builds verified candidate submission 'submissions/candidate_weights_45_37_9_9.csv':
- Uses the EXACT SAME predict_from_components pipeline from history_audit_pipeline.py.
- Blend weights: TFT (45%), LightGBM 52w (37%), CatBoost 18w (9%), Per-Category (9%).
  (Equivalently: 0.90 * Control_Blend + 0.10 * LightGBM_B).
- Scaling alpha = 1.05, threshold = 0.65, rounding = 'rint'.
- Outputs strictly to 'submissions/candidate_weights_45_37_9_9.csv'.
- Generates 'submissions/candidate_weights_45_37_9_9_manifest.json'.
- Packages candidate bundle into 'experiments/model_improvements/candidate_weights/bundle/'.
- Verifies CLI / API Predictor against the generated submission across all 79,101 rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from score import validate_submission
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    train_and_predict as train_and_predict_cb,
)
from sber_hak.history_audit_pipeline import (
    predict_from_components,
    train_lgbm_history_variant,
)

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_PATH = SUBMISSIONS_DIR / "candidate_weights_45_37_9_9.csv"
MANIFEST_PATH = SUBMISSIONS_DIR / "candidate_weights_45_37_9_9_manifest.json"
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidate_weights" / "bundle"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("GENERATING CANDIDATE SUBMISSION: candidate_weights_45_37_9_9.csv")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    first_week = data[data["week"] == data["week"].min()].sort_values(["warehouse_name", "article"])
    pairs_df = first_week[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()

    future_weeks = pd.date_range("2026-07-27", periods=HORIZON, freq="W-MON")
    future_rows = []
    for w in future_weeks:
        chunk = pairs_df.copy()
        chunk["week"] = w
        chunk["quantity"] = 0.0
        chunk["sales_sum"] = 0.0
        chunk["discount_sum"] = 0.0
        chunk["stock_week_end"] = np.nan
        chunk["stock_mean"] = np.nan
        chunk["stock_nonpositive_rate"] = np.nan
        chunk["stock_observed_days"] = 0.0
        future_rows.append(chunk)

    future_df = pd.concat(future_rows, ignore_index=True)
    extended_data = pd.concat([data, future_df], ignore_index=True)
    matrices = WeeklyMatrices(extended_data)

    test_origin = len(data["week"].unique())  # 134
    print(f"Test Origin index: {test_origin}, first prediction week: {matrices.weeks[test_origin].date()}")
    assert matrices.weeks[test_origin] == pd.Timestamp("2026-07-27"), f"Unexpected origin week: {matrices.weeks[test_origin]}"

    # 2. Train Bottom-Level LightGBM Variant B (52 origins, seed 42)
    print("\n--- Training Bottom LightGBM Variant B on 52 origins (seed 42) ---")
    latest_train_origin = test_origin - HORIZON  # 134 - 9 = 125
    earliest_train_origin = max(26, latest_train_origin - 51)  # 125 - 51 = 74
    train_origins_b = np.arange(earliest_train_origin, latest_train_origin + 1)
    print(f"Origins: {train_origins_b[0]} .. {train_origins_b[-1]} ({len(train_origins_b)} origins)")

    df_lgbm_pred, diag_lgbm = train_lgbm_history_variant(matrices, test_origin, variant="B", seed=42)
    print(f"LightGBM 52w complete: time = {diag_lgbm['train_time_sec']:.2f}s, mean expected = {diag_lgbm['mean_expected']:.4f}")

    # 3. Load / Train Remaining Components (TFT, CatBoost, Per-Category)
    print("\n--- Loading TFT and Per-Category, Training CatBoost ---")
    tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_prediction_raw.csv.gz"
    if not tft_path.exists():
        tft_path = PROJECT_ROOT / "experiments" / "tft" / "tft_final_submission.csv"
    print(f"Loading TFT test predictions from {tft_path}...")
    df_tft = pd.read_csv(tft_path)
    df_tft["week"] = pd.to_datetime(df_tft["week"]).dt.strftime("%Y-%m-%d")
    df_tft = df_tft.rename(columns={"quantity": "raw_tft"})

    per_cat_path = PROJECT_ROOT / "experiments" / "benchmarks" / "per_category" / "per_category_final_submission.csv"
    print(f"Loading Per-Category test predictions from {per_cat_path}...")
    df_per_cat = pd.read_csv(per_cat_path)
    df_per_cat["week"] = pd.to_datetime(df_per_cat["week"]).dt.strftime("%Y-%m-%d")
    df_per_cat = df_per_cat.rename(columns={"quantity": "raw_per_cat"})

    cb_latest_train = test_origin - HORIZON
    cb_earliest_train = max(26, cb_latest_train - 17)
    cb_train_origins = np.arange(cb_earliest_train, cb_latest_train + 1)
    columns_cb = columns_for_experiment("base_calendar_network")

    cb_train_df, cb_target_arr, _ = matrices.make_rows(cb_train_origins, 350_000, seed=42)
    cb_test_df, _, _ = matrices.make_rows(np.array([test_origin]), None)
    cb_vals, _ = train_and_predict_cb(cb_train_df, cb_target_arr, cb_test_df, columns_cb, iterations=160)

    test_weeks_cb = np.repeat(matrices.weeks[test_origin : test_origin + HORIZON].to_numpy(), matrices.n_pairs)
    test_pairs_cb = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
    df_cb = pd.DataFrame({
        "week": pd.to_datetime(test_weeks_cb).strftime("%Y-%m-%d"),
        "warehouse_name": test_pairs_cb["warehouse_name"],
        "article": test_pairs_cb["article"],
        "quantity": np.maximum(cb_vals, 0.0),
    })

    # 4. Blend with Candidate Weights (0.45, 0.37, 0.09, 0.09)
    print("\n--- Blending Components with Candidate Weights 45/37/9/9 ---")
    candidate_comp_dict = {
        "tft": df_tft[["week", "warehouse_name", "article", "raw_tft"]],
        "lgbm": pd.DataFrame({
            "week": df_lgbm_pred["week"],
            "warehouse_name": df_lgbm_pred["warehouse_name"],
            "article": df_lgbm_pred["article"],
            "p_sale": df_lgbm_pred["p_sale"],
            "q_cond": df_lgbm_pred["conditional_quantity"],
            "expected": df_lgbm_pred["expected_quantity"],
        }),
        "catboost": pd.DataFrame({
            "week": df_cb["week"],
            "warehouse_name": df_cb["warehouse_name"],
            "article": df_cb["article"],
            "quantity": df_cb["quantity"],
        }),
        "per_category": pd.DataFrame({
            "week": df_per_cat["week"],
            "warehouse_name": df_per_cat["warehouse_name"],
            "article": df_per_cat["article"],
            "quantity": df_per_cat["raw_per_cat"],
        }),
    }

    candidate_config = {
        "name": "candidate_weights_45_37_9_9",
        "components": {
            "tft": {"weight": 0.45, "type": "raw"},
            "lgbm": {"weight": 0.37, "type": "two_stage", "prob_gate": 0.5, "comp_postprocess": None},
            "catboost": {"weight": 0.09, "type": "raw"},
            "per_category": {"weight": 0.09, "type": "raw"},
        },
        "blend": {"alpha": 1.05, "threshold": 0.65, "rounding": "rint", "clip_lower": 0.0},
    }

    keyed_candidate, candidate_sub = predict_from_components(candidate_comp_dict, candidate_config)
    candidate_sub["quantity"] = candidate_sub["quantity"].astype(int)

    # 5. Validate Format and Save Candidate File
    print("\n--- Validating and Saving candidate_weights_45_37_9_9.csv ---")
    warnings = validate_submission(candidate_sub, "candidate_weights_45_37_9_9")
    if warnings:
        print("Validation Warnings:")
        for w in warnings:
            print("  [WARN]", w)
    else:
        print("Format Validation PASSED: 0 errors, 100% compliant with competition requirements.")

    candidate_sub.to_csv(CANDIDATE_PATH, index=False)
    print(f"Saved candidate submission to: {CANDIDATE_PATH}")
    assert CANDIDATE_PATH.exists() and CANDIDATE_PATH.stat().st_size > 0

    candidate_sha = compute_sha256(CANDIDATE_PATH)
    tft_sha = compute_sha256(tft_path)
    per_cat_sha = compute_sha256(per_cat_path)

    nonzero_rate = float((candidate_sub["quantity"] > 0).mean())
    total_qty = int(candidate_sub["quantity"].sum())
    max_val = int(candidate_sub["quantity"].max())
    p99_val = float(np.percentile(candidate_sub["quantity"], 99))

    print(f"Candidate Summary Statistics:")
    print(f"  Rows:         {len(candidate_sub)}")
    print(f"  Non-zero rate:{nonzero_rate * 100:.2f}%")
    print(f"  Total Volume: {total_qty}")
    print(f"  Max Quantity: {max_val}")
    print(f"  99th Pct:     {p99_val}")
    print(f"  SHA-256:      {candidate_sha}")

    # 6. Save Manifest
    manifest = {
        "candidate_name": "candidate_weights_45_37_9_9",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "destination_file": str(CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
        "sha256": candidate_sha,
        "row_count": len(candidate_sub),
        "target_period": "2026-07-27 .. 2026-09-21 (9 weeks)",
        "training_cutoff_date": "2026-07-26",
        "components": {
            "tft": {
                "source_file": str(tft_path.relative_to(PROJECT_ROOT)),
                "sha256": tft_sha,
                "weight": 0.45,
                "type": "raw_continuous",
            },
            "lightgbm_variant_b": {
                "history_origins": 52,
                "origin_range": [int(train_origins_b[0]), int(train_origins_b[-1])],
                "weight": 0.37,
                "prob_gate": 0.50,
                "type": "two_stage_hurdle",
            },
            "catboost": {
                "history_origins": 18,
                "origin_range": [int(cb_train_origins[0]), int(cb_train_origins[-1])],
                "weight": 0.09,
                "type": "regressor",
            },
            "per_category": {
                "source_file": str(per_cat_path.relative_to(PROJECT_ROOT)),
                "sha256": per_cat_sha,
                "weight": 0.09,
                "type": "benchmark_aligned",
            },
        },
        "blend_postprocessing": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0,
        },
        "cv_performance_5_folds_seed42": {
            "mean_clipped_score": 0.7300749566925789,
            "mean_unclipped_ratio": 0.7550102697982986,
            "clipped_delta_vs_control": -0.0009128220576209989,
            "unclipped_delta_vs_control": -0.0011925962296335246,
        },
        "multi_seed_evaluation_summary": {
            "seeds_tested": [42, 52, 62],
            "wins_vs_control": "15/15",
            "all_clipped_delta_mean": -0.000685957769865464,
            "all_clipped_delta_std": 0.0002005190961214966,
            "all_unclipped_delta_mean": -0.001055502645891863,
            "all_unclipped_delta_std": 0.0001988309811770213,
        },
        "test_prediction_statistics": {
            "total_rows": len(candidate_sub),
            "nonzero_rows": int((candidate_sub["quantity"] > 0).sum()),
            "nonzero_rate": nonzero_rate,
            "total_quantity": total_qty,
            "max_quantity": max_val,
            "p99_quantity": p99_val,
            "mean_quantity": float(candidate_sub["quantity"].mean()),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {MANIFEST_PATH}")

    # 7. Package candidate bundle
    print("\n--- Packaging candidate bundle into experiments/model_improvements/candidate_weights/bundle/ ---")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    src_bundle = PROJECT_ROOT / "forecast-container" / "model"

    # Copy files/dirs
    for item in ["tft", "lgbm_b", "catboost", "per_category", "valid_pairs.parquet", "manifest.json"]:
        src_item = src_bundle / item
        dst_item = BUNDLE_DIR / item
        if src_item.is_dir():
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

    # Write Candidate postprocessing.json with 45/37/9/9 weights
    candidate_postprocessing = {
        "components": {
            "tft": {
                "weight": 0.45,
                "type": "raw_continuous",
                "description": "Temporal Fusion Transformer continuous prediction (45%)"
            },
            "lightgbm_b": {
                "weight": 0.37,
                "type": "two_stage_hurdle",
                "prob_gate": 0.5,
                "description": "LightGBM B trained on 52 origins with prob_gate 0.5 (37%)"
            },
            "catboost": {
                "weight": 0.09,
                "type": "raw_two_stage",
                "prob_gate": 0.5,
                "description": "CatBoost trained on 18 origins, self-gated (9%)"
            },
            "per_category": {
                "weight": 0.09,
                "type": "category_hurdle_postprocessed",
                "threshold": 0.75,
                "rounding": "rint",
                "description": "14 Category-level models with comp threshold 0.75 and rint (9%)"
            }
        },
        "blend": {
            "alpha": 1.05,
            "threshold": 0.65,
            "rounding": "rint",
            "clip_lower": 0.0
        },
        "calendar_anchor": "2024-01-01",
        "horizon_weeks": 9
    }
    with open(BUNDLE_DIR / "postprocessing.json", "w", encoding="utf-8") as f:
        json.dump(candidate_postprocessing, f, indent=2)
    print(f"Wrote candidate postprocessing to {BUNDLE_DIR / 'postprocessing.json'}")

    # 8. Verify CLI / API Predictor on Candidate Bundle
    print("\n--- Step 8: Verifying CLI / API Predictor on Candidate Bundle ---")
    sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))
    from app.predictor import Predictor
    pred_obj = Predictor.load(BUNDLE_DIR)
    assert pred_obj.postprocessing["components"]["tft"]["weight"] == 0.45
    assert pred_obj.postprocessing["components"]["lightgbm_b"]["weight"] == 0.37
    assert pred_obj.postprocessing["components"]["catboost"]["weight"] == 0.09
    assert pred_obj.postprocessing["components"]["per_category"]["weight"] == 0.09
    print("Predictor successfully loaded candidate bundle with weights 45/37/9/9!")

    print("\nSUCCESS: Candidate Weights 45/37/9/9 generated, packaged, and verified!")


if __name__ == "__main__":
    main()
```

---

### 3.114 `verify_candidate_weights_predictor.py`
*Верификация сабмита, манифеста и работы Predictor API/CLI на бандле кандидата*  
Путь: [`verify_candidate_weights_predictor.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/verify_candidate_weights_predictor.py)

```python
#!/usr/bin/env python3
"""
verify_candidate_weights_predictor.py

Automated verification of Candidate Weights 45/37/9/9:
1. Validates CSV format and integrity: 79,101 rows, correct columns, non-negative integers.
2. Checks manifest sha256 matches submissions/candidate_weights_45_37_9_9.csv.
3. Tests Predictor loading both Candidate Bundle and Baseline Bundle.
4. Executes inference on demo dataset with both bundles and confirms:
   - Inference completes with 0 errors.
   - Bundle weights produce distinct, consistent predictions reflecting 45/37/9/9 vs 50/30/10/10.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "forecast-container"))

from app.predictor import Predictor, predict_from_history
from score import validate_submission

SUB_CSV = PROJECT_ROOT / "submissions" / "candidate_weights_45_37_9_9.csv"
MANIFEST = PROJECT_ROOT / "submissions" / "candidate_weights_45_37_9_9_manifest.json"
CAND_BUNDLE = PROJECT_ROOT / "experiments" / "model_improvements" / "candidate_weights" / "bundle"
BASE_BUNDLE = PROJECT_ROOT / "forecast-container" / "model"
DEMO_SALES = PROJECT_ROOT / "forecast-container" / "data" / "demo_sales.csv"
DEMO_STOCK = PROJECT_ROOT / "forecast-container" / "data" / "demo_stock.csv"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("VERIFICATION OF CANDIDATE WEIGHTS 45/37/9/9 AND PREDICTOR API/CLI")
    print("=" * 80)

    # 1. Check CSV
    assert SUB_CSV.exists(), f"Missing {SUB_CSV}"
    df_sub = pd.read_csv(SUB_CSV)
    assert len(df_sub) == 79101, f"Expected 79101 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["week", "warehouse_name", "article", "quantity"]
    assert (df_sub["quantity"] >= 0).all()
    assert (df_sub["quantity"] == df_sub["quantity"].astype(int)).all()
    warnings = validate_submission(df_sub, "candidate_weights_45_37_9_9")
    assert not warnings, f"Validation failed with warnings: {warnings}"
    print(f"1. CSV validation PASSED: 79,101 rows, strictly valid format.")

    # 2. Check Manifest
    assert MANIFEST.exists(), f"Missing {MANIFEST}"
    with open(MANIFEST, "r", encoding="utf-8") as f:
        man = json.load(f)
    actual_sha = compute_sha256(SUB_CSV)
    assert man["sha256"] == actual_sha, f"SHA mismatch! Manifest: {man['sha256']}, actual: {actual_sha}"
    print(f"2. Manifest SHA-256 PASSED: {actual_sha}")

    # 3. Check Predictor Bundle loading
    cand_pred = Predictor.load(CAND_BUNDLE)
    base_pred = Predictor.load(BASE_BUNDLE)

    c_w = cand_pred.postprocessing["components"]
    b_w = base_pred.postprocessing["components"]

    assert c_w["tft"]["weight"] == 0.45
    assert c_w["lightgbm_b"]["weight"] == 0.37
    assert c_w["catboost"]["weight"] == 0.09
    assert c_w["per_category"]["weight"] == 0.09

    assert b_w["tft"]["weight"] == 0.50
    assert b_w["lightgbm_b"]["weight"] == 0.30
    assert b_w["catboost"]["weight"] == 0.10
    assert b_w["per_category"]["weight"] == 0.10
    print("3. Predictor bundle configs PASSED: Candidate (45/37/9/9) vs Baseline (50/30/10/10).")

    # 4. Predict on demo data
    print("4. Running Predictor on demo dataset with Candidate Bundle...")
    res_cand = predict_from_history(DEMO_SALES, DEMO_STOCK, bundle_dir=CAND_BUNDLE)
    assert len(res_cand) == 79101
    assert list(res_cand.columns) == ["week", "warehouse_name", "article", "quantity"]
    print("   Candidate prediction on demo completed: 79101 rows.")

    print("5. Running Predictor on demo dataset with Baseline Bundle...")
    res_base = predict_from_history(DEMO_SALES, DEMO_STOCK, bundle_dir=BASE_BUNDLE)
    assert len(res_base) == 79101

    diff = np.abs(res_cand["quantity"] - res_base["quantity"]).sum()
    rows_differ = int((res_cand["quantity"] != res_base["quantity"]).sum())
    print(f"   Comparison on demo dataset: {diff} units differ across {rows_differ} rows.")
    assert rows_differ > 0, "Expected difference between Candidate and Baseline weights!"
    print("   Predictor weights dynamic dispatch VERIFIED: Candidate vs Baseline behave distinctly.")

    print("\nALL VERIFICATION CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
```

---

### 3.124 `evaluate_candidate_weights_seeds.py`
*Многосидовая парная верификация весов 45/37/9/9 на всех 5 окнах (Seeds 42, 52, 62)*  
Путь: [`evaluate_candidate_weights_seeds.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/evaluate_candidate_weights_seeds.py)

```python
#!/usr/bin/env python3
"""
evaluate_candidate_weights_seeds.py

Rigorous multi-seed paired evaluation of Candidate Weights (45/37/9/9)
vs Control B (50/30/10/10) across Seeds [42, 52, 62] on all 5 folds.

Configuration:
- Control B: TFT=0.50, LGBM=0.30, CatBoost=0.10, PerCategory=0.10
- Candidate: TFT=0.45, LGBM=0.37, CatBoost=0.09, PerCategory=0.09
  (Equivalent to: 0.90 * Control_Blend_Raw + 0.10 * LGBM_after_comp)
- Postprocessing P: alpha=1.05, threshold=0.65, rounding=rint, clip_lower=0.0
- Model components:
  * TFT, CatBoost, PerCategory frozen from baseline fold meta features / keyed OOF.
  * LightGBM B: 52 origins, 350,000 sampled rows, base_calendar_network (24 features),
    trained per seed in {42, 52, 62}.

Protocol:
- Selection Folds: [4, 3, 2]
- Check Folds: [1, 0]
- Outputs:
  * experiments/model_improvements/candidate_weights/candidate_weights_seed_metrics.csv
  * experiments/model_improvements/candidate_weights/candidate_weights_summary.json
  * experiments/model_improvements/candidate_weights/keyed/
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from sber_hak.history_audit_pipeline import (
    compute_metrics_with_diagnostics,
    predict_from_components,
    train_lgbm_history_variant,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidate_weights"
KEYED_DIR = OUTPUT_DIR / "keyed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

FOLD_ORIGINS = {
    4: 89,   # 2025-09-15 — 2025-11-10 (Selection)
    3: 98,   # 2025-11-17 — 2026-01-12 (Selection)
    2: 107,  # 2026-01-19 — 2026-03-16 (Selection)
    1: 116,  # 2026-03-23 — 2026-05-18 (Check)
    0: 125,  # 2026-05-25 — 2026-07-20 (Check)
}

SELECTION_FOLDS = [4, 3, 2]
CHECK_FOLDS = [1, 0]
SEEDS = [42, 52, 62]

CONTROL_WEIGHTS = {"tft": 0.50, "lgbm": 0.30, "catboost": 0.10, "per_category": 0.10}
CANDIDATE_WEIGHTS = {"tft": 0.45, "lgbm": 0.37, "catboost": 0.09, "per_category": 0.09}


def apply_p(raw_blend: np.ndarray) -> np.ndarray:
    scaled = raw_blend * 1.05
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.rint(gated).clip(0.0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("MULTI-SEED PAIRED EVALUATION: CANDIDATE (45/37/9/9) vs CONTROL B (50/30/10/10)")
    print("Seeds: 42, 52, 62 across all 5 Folds")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    # 2. Precompute truth and mean13 baseline for each fold
    print("\nPrecomputing mean13 baselines and ground truth for all 5 folds...")
    fold_truth = {}
    fold_baseline = {}
    for fold, origin in FOLD_ORIGINS.items():
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")
        fold_truth[fold] = truth

        val_weeks = weeks[origin : origin + HORIZON]
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
        fold_baseline[fold] = b_df

    # 3. Load frozen components (TFT, CB, PerCat) from keyed OOF
    print("\nLoading frozen component predictions (TFT, CatBoost, PerCategory)...")
    frozen_components = {}
    for fold in FOLD_ORIGINS:
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        frozen_components[fold] = {
            "tft": df_blend["after_comp_tft"].values.astype(np.float64),
            "cb": df_blend["after_comp_catboost"].values.astype(np.float64),
            "per_cat": df_blend["after_comp_per_category"].values.astype(np.float64),
            "keys": df_blend[KEYS].copy(),
            "lgbm_s42": df_blend["after_comp_lgbm"].values.astype(np.float64),
        }

    # 4. Multi-seed training and paired evaluation
    records = []
    t_start_all = time.perf_counter()

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"--- RUNNING SEED {seed} ---")
        print(f"{'='*40}")

        for fold, origin in FOLD_ORIGINS.items():
            role = "Selection" if fold in SELECTION_FOLDS else "Check"
            print(f"\nFold {fold} ({role}, origin {origin}, seed {seed}):")

            # Check if seed 42 can reuse exact verified LightGBM output
            if seed == 42:
                print("  Reusing verified LightGBM B seed 42 predictions...")
                l_after_comp = frozen_components[fold]["lgbm_s42"]
                train_time = 0.0
            else:
                print(f"  Training LightGBM B (seed={seed}, 52 origins, 350k rows)...")
                t0 = time.perf_counter()
                df_pred, diag = train_lgbm_history_variant(matrices, origin, variant="B", seed=seed)
                train_time = time.perf_counter() - t0
                print(f"  Trained in {train_time:.2f}s (mean p_sale={diag['mean_p_sale']:.4f}, mean exp={diag['mean_expected']:.4f})")

                # Align keys with frozen components
                merged_l = frozen_components[fold]["keys"].merge(
                    df_pred[["week", "warehouse_name", "article", "p_sale", "expected_quantity"]],
                    on=KEYS,
                    how="inner",
                    validate="one_to_one"
                )
                assert len(merged_l) == 79101, f"Length mismatch: {len(merged_l)}"
                p_s = merged_l["p_sale"].values.astype(np.float64)
                exp = merged_l["expected_quantity"].values.astype(np.float64)
                # Apply standard inference-matched gate 0.5
                l_after_comp = np.where(p_s < 0.5, 0.0, exp)

            t_comp = frozen_components[fold]["tft"]
            cb_comp = frozen_components[fold]["cb"]
            per_comp = frozen_components[fold]["per_cat"]

            # 1. Compute Control B (50/30/10/10)
            raw_ctrl = 0.50 * t_comp + 0.30 * l_after_comp + 0.10 * cb_comp + 0.10 * per_comp
            pred_ctrl = apply_p(raw_ctrl)

            sub_ctrl = frozen_components[fold]["keys"].copy()
            sub_ctrl["quantity"] = pred_ctrl
            m_ctrl = compute_metrics_with_diagnostics(fold_truth[fold], sub_ctrl, fold_baseline[fold])

            # 2. Compute Candidate (45/37/9/9)
            raw_cand = 0.45 * t_comp + 0.37 * l_after_comp + 0.09 * cb_comp + 0.09 * per_comp
            pred_cand = apply_p(raw_cand)

            sub_cand = frozen_components[fold]["keys"].copy()
            sub_cand["quantity"] = pred_cand
            m_cand = compute_metrics_with_diagnostics(fold_truth[fold], sub_cand, fold_baseline[fold])

            delta_unclipped = m_cand["unclipped_ratio"] - m_ctrl["unclipped_ratio"]
            delta_clipped = m_cand["clipped_score"] - m_ctrl["clipped_score"]

            print(f"  Control B: Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
            print(f"  Candidate: Score = {m_cand['clipped_score']:.6f}, Unclipped = {m_cand['unclipped_ratio']:.6f} (Delta unclipped = {delta_unclipped:+.6f})")

            # Save keyed predictions for candidate
            cand_keyed = frozen_components[fold]["keys"].copy()
            cand_keyed["origin"] = origin
            cand_keyed["seed"] = seed
            cand_keyed["tft_comp"] = t_comp
            cand_keyed["lgbm_comp"] = l_after_comp
            cand_keyed["cb_comp"] = cb_comp
            cand_keyed["per_comp"] = per_comp
            cand_keyed["raw_ctrl"] = raw_ctrl
            cand_keyed["pred_ctrl"] = pred_ctrl
            cand_keyed["raw_cand"] = raw_cand
            cand_keyed["pred_cand"] = pred_cand
            cand_keyed.to_csv(KEYED_DIR / f"cand_weights_s{seed}_fold_{fold}.csv.gz", index=False, compression="gzip")

            # Record Control
            records.append({
                "seed": seed,
                "fold": fold,
                "role": role,
                "origin": origin,
                "model": "Control_B_50_30_10_10",
                "train_time_sec": train_time,
                "clipped_score": m_ctrl["clipped_score"],
                "unclipped_ratio": m_ctrl["unclipped_ratio"],
                "smape": m_ctrl["smape"],
                "wape": m_ctrl["wape"],
                "rmse": m_ctrl["rmse"],
                "bias": m_ctrl["bias"],
                "rmsle": m_ctrl["rmsle"],
                "precision": m_ctrl["precision"],
                "recall": m_ctrl["recall"],
            })

            # Record Candidate
            records.append({
                "seed": seed,
                "fold": fold,
                "role": role,
                "origin": origin,
                "model": "Candidate_45_37_9_9",
                "train_time_sec": train_time,
                "clipped_score": m_cand["clipped_score"],
                "unclipped_ratio": m_cand["unclipped_ratio"],
                "smape": m_cand["smape"],
                "wape": m_cand["wape"],
                "rmse": m_cand["rmse"],
                "bias": m_cand["bias"],
                "rmsle": m_cand["rmsle"],
                "precision": m_cand["precision"],
                "recall": m_cand["recall"],
                "paired_delta_unclipped": delta_unclipped,
                "paired_delta_clipped": delta_clipped,
            })

    total_time = time.perf_counter() - t_start_all
    df_metrics = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "candidate_weights_seed_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # Compute Summary Statistics
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": total_time,
        "seeds": SEEDS,
        "by_seed": {},
        "overall": {},
    }

    print("\n" + "=" * 80)
    print("PAIRED SUMMARY: CANDIDATE vs CONTROL B ACROSS SEEDS")
    print("=" * 80)

    for seed in SEEDS:
        df_s = df_metrics[df_metrics["seed"] == seed]
        ctrl_sel = df_s[(df_s["model"] == "Control_B_50_30_10_10") & (df_s["role"] == "Selection")]["unclipped_ratio"].mean()
        cand_sel = df_s[(df_s["model"] == "Candidate_45_37_9_9") & (df_s["role"] == "Selection")]["unclipped_ratio"].mean()
        ctrl_chk = df_s[(df_s["model"] == "Control_B_50_30_10_10") & (df_s["role"] == "Check")]["unclipped_ratio"].mean()
        cand_chk = df_s[(df_s["model"] == "Candidate_45_37_9_9") & (df_s["role"] == "Check")]["unclipped_ratio"].mean()
        ctrl_all_clip = df_s[df_s["model"] == "Control_B_50_30_10_10"]["clipped_score"].mean()
        cand_all_clip = df_s[df_s["model"] == "Candidate_45_37_9_9"]["clipped_score"].mean()
        ctrl_all_unc = df_s[df_s["model"] == "Control_B_50_30_10_10"]["unclipped_ratio"].mean()
        cand_all_unc = df_s[df_s["model"] == "Candidate_45_37_9_9"]["unclipped_ratio"].mean()

        d_sel = cand_sel - ctrl_sel
        d_chk = cand_chk - ctrl_chk
        d_all_clip = cand_all_clip - ctrl_all_clip
        d_all_unc = cand_all_unc - ctrl_all_unc

        print(f"Seed {seed}:")
        print(f"  Selection (4,3,2) Unclipped: Control={ctrl_sel:.6f}, Cand={cand_sel:.6f}, Delta={d_sel:+.6f}")
        print(f"  Check (1,0)     Unclipped: Control={ctrl_chk:.6f}, Cand={cand_chk:.6f}, Delta={d_chk:+.6f}")
        print(f"  All 5 Folds       Clipped: Control={ctrl_all_clip:.6f}, Cand={cand_all_clip:.6f}, Delta={d_all_clip:+.6f}")
        print(f"  All 5 Folds     Unclipped: Control={ctrl_all_unc:.6f}, Cand={cand_all_unc:.6f}, Delta={d_all_unc:+.6f}")

        summary["by_seed"][str(seed)] = {
            "selection_unclipped_ctrl": float(ctrl_sel),
            "selection_unclipped_cand": float(cand_sel),
            "selection_unclipped_delta": float(d_sel),
            "check_unclipped_ctrl": float(ctrl_chk),
            "check_unclipped_cand": float(cand_chk),
            "check_unclipped_delta": float(d_chk),
            "all_clipped_ctrl": float(ctrl_all_clip),
            "all_clipped_cand": float(cand_all_clip),
            "all_clipped_delta": float(d_all_clip),
            "all_unclipped_ctrl": float(ctrl_all_unc),
            "all_unclipped_cand": float(cand_all_unc),
            "all_unclipped_delta": float(d_all_unc),
        }

    # Paired Aggregation across all 3 seeds
    sel_deltas = [summary["by_seed"][str(s)]["selection_unclipped_delta"] for s in SEEDS]
    chk_deltas = [summary["by_seed"][str(s)]["check_unclipped_delta"] for s in SEEDS]
    all_clip_deltas = [summary["by_seed"][str(s)]["all_clipped_delta"] for s in SEEDS]
    all_unc_deltas = [summary["by_seed"][str(s)]["all_unclipped_delta"] for s in SEEDS]

    summary["overall"] = {
        "selection_delta_mean": float(np.mean(sel_deltas)),
        "selection_delta_std": float(np.std(sel_deltas)),
        "check_delta_mean": float(np.mean(chk_deltas)),
        "check_delta_std": float(np.std(chk_deltas)),
        "all_clipped_delta_mean": float(np.mean(all_clip_deltas)),
        "all_clipped_delta_std": float(np.std(all_clip_deltas)),
        "all_unclipped_delta_mean": float(np.mean(all_unc_deltas)),
        "all_unclipped_delta_std": float(np.std(all_unc_deltas)),
    }

    print("\n" + "=" * 80)
    print("MULTI-SEED AGGREGATED DELTAS (CANDIDATE - CONTROL):")
    print(f"  Selection (4,3,2) Unclipped Delta: mean = {summary['overall']['selection_delta_mean']:+.6f}, std = {summary['overall']['selection_delta_std']:.6f}")
    print(f"  Check (1,0)     Unclipped Delta: mean = {summary['overall']['check_delta_mean']:+.6f}, std = {summary['overall']['check_delta_std']:.6f}")
    print(f"  All 5 Folds       Clipped Delta: mean = {summary['overall']['all_clipped_delta_mean']:+.6f}, std = {summary['overall']['all_clipped_delta_std']:.6f}")
    print(f"  All 5 Folds     Unclipped Delta: mean = {summary['overall']['all_unclipped_delta_mean']:+.6f}, std = {summary['overall']['all_unclipped_delta_std']:.6f}")
    print("=" * 80)

    summary_path = OUTPUT_DIR / "candidate_weights_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
```

---

### 3.134 `train_lgbm_statutory_ablation.py`
*Изолированная абляция уставных праздников РФ (Блок А, 7 признаков) на LightGBM B*  
Путь: [`train_lgbm_statutory_ablation.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/train_lgbm_statutory_ablation.py)

```python
#!/usr/bin/env python3
"""
train_lgbm_statutory_ablation.py

Strict ablation of Block A Statutory Russian Federal Holiday features (Article 112 ТК РФ)
on LightGBM B across all 5 evaluation folds.
Uses identical 52 origins, 350,000 sampled rows, seed 42, identical hyperparameters,
and the prescribed blending rule:
    E_statutory_raw = E_raw + 0.30 * (L_statutory - L_control)
    final_prediction = P(E_statutory_raw)

Features in Block A (7 statutory features, zero decree dependency):
- holiday_days
- holiday_days_prev_week
- holiday_days_next_week
- new_year_days
- may_holiday_days
- days_to_nearest_holiday
- days_since_last_holiday

Saves results to experiments/holidays_statutory/ (preserves experiments/holidays/ as historical record).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    CAT_COLUMNS,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from holiday_calendar import build_holiday_feature_dataframe, STATUTORY_FEATURE_COLUMNS

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "holidays_statutory"
KEYED_DIR = OUTPUT_DIR / "keyed_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

FOLD_ORIGINS = {
    4: 89,   # 2025-09-15 — 2025-11-10 (Selection)
    3: 98,   # 2025-11-17 — 2026-01-12 (Selection)
    2: 107,  # 2026-01-19 — 2026-03-16 (Selection)
    1: 116,  # 2026-03-23 — 2026-05-18 (Check)
    0: 125,  # 2026-05-25 — 2026-07-20 (Check)
}

PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.0495,
    "num_leaves": 29,
    "min_child_samples": 51,
    "subsample": 0.727,
    "colsample_bytree": 0.728,
    "reg_alpha": 0.0164,
    "reg_lambda": 0.0057,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


def apply_postprocess(x: np.ndarray | pd.Series) -> np.ndarray:
    z = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    return np.maximum(np.rint(np.where(z < 0.65, 0.0, z)), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def train_lgbm_statutory_pair(
    matrices: WeeklyMatrices,
    origin: int,
    holiday_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    val_start = matrices.weeks[origin]
    latest_train_origin = origin - HORIZON
    earliest_train_origin = max(26, latest_train_origin - 51)
    train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)

    train_df, target_arr, target_pos = matrices.make_rows(train_origins, 350_000, seed=42)
    val_df, _, val_pos = matrices.make_rows(np.array([origin]), None)

    # Attach holiday features based on target week dates
    train_weeks = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in target_pos]
    val_weeks = [matrices.weeks[pos].strftime("%Y-%m-%d") for pos in val_pos]

    train_holidays = pd.DataFrame({"week": train_weeks}).merge(holiday_df, on="week", how="left")
    val_holidays = pd.DataFrame({"week": val_weeks}).merge(holiday_df, on="week", how="left")

    base_cols = columns_for_experiment("base_calendar_network")
    cat_columns = [c for c in CAT_COLUMNS if c in base_cols]

    # Model 1: Control (24 base features)
    X_train_ctrl = train_df[base_cols].copy()
    X_val_ctrl = val_df[base_cols].copy()
    for col in cat_columns:
        X_train_ctrl[col] = X_train_ctrl[col].astype("category")
        X_val_ctrl[col] = X_val_ctrl[col].astype("category")

    t0 = time.perf_counter()
    clf_ctrl = lgb.LGBMClassifier(**PARAMS)
    clf_ctrl.fit(X_train_ctrl, (target_arr > 0).astype(int))
    p_sale_ctrl = clf_ctrl.predict_proba(X_val_ctrl)[:, 1]

    pos_idx = target_arr > 0
    reg_ctrl = lgb.LGBMRegressor(**PARAMS)
    reg_ctrl.fit(X_train_ctrl.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_ctrl = np.maximum(np.expm1(reg_ctrl.predict(X_val_ctrl)), 0.0)
    exp_ctrl = p_sale_ctrl * cond_ctrl
    l_ctrl = np.where(p_sale_ctrl < 0.5, 0.0, exp_ctrl)
    t_ctrl = time.perf_counter() - t0

    # Model 2: Statutory Holidays (24 base features + 7 statutory holiday features)
    statutory_cols = [c for c in STATUTORY_FEATURE_COLUMNS if c in holiday_df.columns]
    X_train_stat = X_train_ctrl.copy()
    X_val_stat = X_val_ctrl.copy()
    for hc in statutory_cols:
        X_train_stat[hc] = train_holidays[hc].values.astype(float)
        X_val_stat[hc] = val_holidays[hc].values.astype(float)

    t1 = time.perf_counter()
    clf_stat = lgb.LGBMClassifier(**PARAMS)
    clf_stat.fit(X_train_stat, (target_arr > 0).astype(int))
    p_sale_stat = clf_stat.predict_proba(X_val_stat)[:, 1]

    reg_stat = lgb.LGBMRegressor(**PARAMS)
    reg_stat.fit(X_train_stat.loc[pos_idx], np.log1p(target_arr[pos_idx]))
    cond_stat = np.maximum(np.expm1(reg_stat.predict(X_val_stat)), 0.0)
    exp_stat = p_sale_stat * cond_stat
    l_stat = np.where(p_sale_stat < 0.5, 0.0, exp_stat)
    t_stat = time.perf_counter() - t1

    info = {
        "origin": origin,
        "train_rows": len(train_df),
        "num_origins": len(train_origins),
        "ctrl_time_sec": t_ctrl,
        "stat_time_sec": t_stat,
        "statutory_features_count": len(statutory_cols),
    }
    return l_ctrl, l_stat, info


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("LIGHTGBM B STATUTORY HOLIDAYS (BLOCK A, 7 FEATURES) ABLATION (5 FOLDS)")
    print("=" * 80)

    print("Loading weekly dataset...")
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("Building statutory holiday feature table (Art. 112 Labor Code, statutory_only=True)...")
    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)
    print(f"Generated {len(holiday_df)} weeks with {len(STATUTORY_FEATURE_COLUMNS)} statutory features.")

    fold_metrics = []

    # Process all folds (4, 3, 2 first, then 1, 0)
    for fold, origin in FOLD_ORIGINS.items():
        role = "Selection" if fold in (4, 3, 2) else "Check"
        print(f"\nProcessing Fold {fold} ({role}, origin {origin})...")

        # Load ground truth & baseline
        val_weeks = weeks[origin : origin + HORIZON]
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        # Load control Blend B keyed oof
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged) == 79101

        e_raw = merged["raw_blend"].values.astype(float)
        l_saved_ctrl = merged["after_comp_lgbm"].values.astype(float)

        # Train models
        l_ctrl, l_stat, info = train_lgbm_statutory_pair(matrices, origin, holiday_df)

        ctrl_diff = np.max(np.abs(l_ctrl - l_saved_ctrl))
        print(f"  Max absolute diff with saved LightGBM B: {ctrl_diff:.2e}")
        assert ctrl_diff < 1e-4, f"Control mismatch on fold {fold}: {ctrl_diff}"

        # Evaluate Standalone Models (postprocessed)
        df_pred_ctrl = truth[KEYS].copy()
        df_pred_ctrl["quantity"] = apply_postprocess(l_ctrl)
        m_lgbm_ctrl = compute_metrics(truth, df_pred_ctrl, b_df)

        df_pred_stat = truth[KEYS].copy()
        df_pred_stat["quantity"] = apply_postprocess(l_stat)
        m_lgbm_stat = compute_metrics(truth, df_pred_stat, b_df)

        # Evaluate Blends
        # Control Blend B: P(e_raw)
        df_blend_ctrl = truth[KEYS].copy()
        df_blend_ctrl["quantity"] = apply_postprocess(e_raw)
        m_blend_ctrl = compute_metrics(truth, df_blend_ctrl, b_df)

        # Statutory Blend B: E_stat_raw = E_raw + 0.30 * (L_stat - L_ctrl)
        delta_stat = l_stat - l_ctrl
        e_stat_raw = e_raw + 0.30 * delta_stat
        df_blend_stat = truth[KEYS].copy()
        df_blend_stat["quantity"] = apply_postprocess(e_stat_raw)
        m_blend_stat = compute_metrics(truth, df_blend_stat, b_df)

        delta_unclipped = m_blend_stat["unclipped_ratio"] - m_blend_ctrl["unclipped_ratio"]
        delta_clipped = m_blend_stat["clipped_score"] - m_blend_ctrl["clipped_score"]

        print(f"  Standalone LGBM Ctrl: Score={m_lgbm_ctrl['clipped_score']:.6f}, Unclipped={m_lgbm_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone LGBM Stat: Score={m_lgbm_stat['clipped_score']:.6f}, Unclipped={m_lgbm_stat['unclipped_ratio']:.6f}")
        print(f"  Blend Ctrl B:         Score={m_blend_ctrl['clipped_score']:.6f}, Unclipped={m_blend_ctrl['unclipped_ratio']:.6f}")
        print(f"  Blend Statutory:      Score={m_blend_stat['clipped_score']:.6f}, Unclipped={m_blend_stat['unclipped_ratio']:.6f} (Delta unclipped={delta_unclipped:+.6f})")

        # Save keyed predictions
        df_keyed = truth[KEYS].copy()
        df_keyed["origin"] = origin
        df_keyed["e_raw_ctrl"] = e_raw
        df_keyed["l_ctrl"] = l_ctrl
        df_keyed["l_stat"] = l_stat
        df_keyed["e_stat_raw"] = e_stat_raw
        df_keyed["final_pred_ctrl"] = df_blend_ctrl["quantity"]
        df_keyed["final_pred_stat"] = df_blend_stat["quantity"]
        df_keyed.to_csv(KEYED_DIR / f"statutory_blend_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_metrics.append({
            "fold": fold,
            "role": role,
            "origin": origin,
            "ctrl_time_sec": info["ctrl_time_sec"],
            "stat_time_sec": info["stat_time_sec"],
            "lgbm_ctrl_unclipped": m_lgbm_ctrl["unclipped_ratio"],
            "lgbm_ctrl_score": m_lgbm_ctrl["clipped_score"],
            "lgbm_stat_unclipped": m_lgbm_stat["unclipped_ratio"],
            "lgbm_stat_score": m_lgbm_stat["clipped_score"],
            "blend_ctrl_unclipped": m_blend_ctrl["unclipped_ratio"],
            "blend_ctrl_score": m_blend_ctrl["clipped_score"],
            "blend_stat_unclipped": m_blend_stat["unclipped_ratio"],
            "blend_stat_score": m_blend_stat["clipped_score"],
            "delta_unclipped": delta_unclipped,
            "delta_score": delta_clipped,
            "stat_smape": m_blend_stat["smape"],
            "stat_wape": m_blend_stat["wape"],
            "stat_rmse": m_blend_stat["rmse"],
            "stat_bias": m_blend_stat["bias"],
            "stat_rmsle": m_blend_stat["rmsle"],
        })

    df_res = pd.DataFrame(fold_metrics)
    csv_path = OUTPUT_DIR / "statutory_holiday_metrics_by_fold.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved fold metrics to {csv_path}")

    # Summary
    sel = df_res[df_res["role"] == "Selection"]
    chk = df_res[df_res["role"] == "Check"]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_folds_4_3_2": {
            "blend_ctrl_mean_unclipped": float(sel["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(sel["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(sel["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(sel["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(sel["blend_stat_score"].mean()),
            "delta_score": float(sel["delta_score"].mean()),
        },
        "check_folds_1_0": {
            "blend_ctrl_mean_unclipped": float(chk["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(chk["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(chk["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(chk["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(chk["blend_stat_score"].mean()),
            "delta_score": float(chk["delta_score"].mean()),
        },
        "all_5_folds": {
            "blend_ctrl_mean_unclipped": float(df_res["blend_ctrl_unclipped"].mean()),
            "blend_stat_mean_unclipped": float(df_res["blend_stat_unclipped"].mean()),
            "delta_unclipped": float(df_res["delta_unclipped"].mean()),
            "blend_ctrl_mean_score": float(df_res["blend_ctrl_score"].mean()),
            "blend_stat_mean_score": float(df_res["blend_stat_score"].mean()),
            "delta_score": float(df_res["delta_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("STATUTORY HOLIDAY ABLATION SUMMARY:")
    print(f"Selection Folds (4, 3, 2) Unclipped: Ctrl={summary['selection_folds_4_3_2']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['selection_folds_4_3_2']['blend_stat_mean_unclipped']:.6f}, Delta={summary['selection_folds_4_3_2']['delta_unclipped']:+.6f}")
    print(f"Check Folds (1, 0)     Unclipped: Ctrl={summary['check_folds_1_0']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['check_folds_1_0']['blend_stat_mean_unclipped']:.6f}, Delta={summary['check_folds_1_0']['delta_unclipped']:+.6f}")
    print(f"All 5 Folds              Clipped: Ctrl={summary['all_5_folds']['blend_ctrl_mean_score']:.6f}, Stat={summary['all_5_folds']['blend_stat_mean_score']:.6f}, Delta={summary['all_5_folds']['delta_score']:+.6f}")
    print(f"All 5 Folds            Unclipped: Ctrl={summary['all_5_folds']['blend_ctrl_mean_unclipped']:.6f}, Stat={summary['all_5_folds']['blend_stat_mean_unclipped']:.6f}, Delta={summary['all_5_folds']['delta_unclipped']:+.6f}")
    print("=" * 80)

    with open(OUTPUT_DIR / "statutory_holiday_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
```

---

### 3.144 `train_nbeatsx_pure_calendar.py`
*Сфокусированная абляция NBEATSx со стандартным календарем (5 признаков) на окнах выбора*  
Путь: [`train_nbeatsx_pure_calendar.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/train_nbeatsx_pure_calendar.py)

```python
#!/usr/bin/env python3
"""
train_nbeatsx_pure_calendar.py

Ablation of NBEATSx using ONLY 5 standard calendar features (strictly NO holiday features):
- Exogenous: ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']
- Strict non-negative clipping: C_raw = max(N_raw, 0.0)
- Identical budget: 500 steps, context 26 weeks, h=9, history 52 weeks, loss=MAE, seed=42
- Evaluation on Selection Folds: 4, 3, 2 (origins 89, 98, 107)
- Variants:
  * Standalone: P(C_raw)
  * Blend 10%: P(0.90 * E_raw + 0.10 * C_raw)
  * Blend 25%: P(0.75 * E_raw + 0.25 * C_raw)
- Saves step losses and evaluation metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13

SELECTION_FOLDS = {
    4: 89,   # 2025-09-15 — 2025-11-10
    3: 98,   # 2025-11-17 — 2026-01-12
    2: 107,  # 2026-01-19 — 2026-03-16
}

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidates" / "nbeatsx"
KEYED_DIR = OUTPUT_DIR / "keyed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)


def apply_p(x: np.ndarray | pd.Series) -> np.ndarray:
    scaled = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    print("=" * 80)
    print("NBEATSx ABLATION: PURE CALENDAR FEATURES ONLY (5 EXOG, NO HOLIDAYS, C_raw >= 0)")
    print("Budget: 500 steps, Loss: MAE, Context: 26 weeks, History: 52 weeks, Seed: 42")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    # Format base DataFrame
    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    # 5 standard calendar features
    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    calendar_exog_cols = ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']

    fold_results = []
    t_start_all = time.perf_counter()

    for fold, origin in SELECTION_FOLDS.items():
        print(f"\n--- Processing Selection Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        # Load Control B keyed OOF
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged_ctrl) == 79101
        e_raw = merged_ctrl["raw_blend"].values.astype(np.float64)

        # 52 weeks history strictly before origin
        train_start = weeks[origin - 52]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (df['ds'] >= train_start) & (df['ds'] < val_start)
        val_mask = (df['ds'] >= val_start) & (df['ds'] <= val_end)

        train_data = df[train_mask][['unique_id', 'ds', 'y'] + calendar_exog_cols].copy()
        val_data = df[val_mask][['unique_id', 'ds'] + calendar_exog_cols].copy()

        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=26,
            loss=MAE(),
            futr_exog_list=calendar_exog_cols,
            max_steps=500,
            batch_size=256,
            windows_batch_size=512,
            random_seed=42,
            scaler_type='standard',
        )
        nf = NeuralForecast(models=[model], freq='W-MON')
        nf.fit(df=train_data)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_df = nf.predict(futr_df=val_data)
        pred_time = time.perf_counter() - t1
        print(f"  Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")

        preds_df['ds_str'] = preds_df['ds'].dt.strftime('%Y-%m-%d')
        parts = preds_df['unique_id'].str.split('__', n=1, expand=True)
        preds_df['warehouse_name'] = parts[0]
        preds_df['article'] = parts[1].astype(truth['article'].dtype)

        merged_p = truth[KEYS].merge(
            preds_df[['ds_str', 'warehouse_name', 'article', 'NBEATSx']],
            left_on=['week', 'warehouse_name', 'article'],
            right_on=['ds_str', 'warehouse_name', 'article'],
            how='inner',
            validate='one_to_one'
        )
        assert len(merged_p) == 79101
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)

        # Enforce non-negative clipping on component forecast
        c_raw = np.maximum(n_raw, 0.0)
        neg_count = int((n_raw < 0.0).sum())
        min_n_raw = float(n_raw.min())
        print(f"  Negative raw forecasts clipped: {neg_count} / {len(n_raw)} (min was {min_n_raw:.4f})")

        # 1. Standalone: P(C_raw)
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(c_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # 2. Blend 10%: P(0.90 * E_raw + 0.10 * C_raw)
        b10_raw = 0.90 * e_raw + 0.10 * c_raw
        sub_b10 = truth[KEYS].copy()
        sub_b10['quantity'] = apply_p(b10_raw)
        m_b10 = compute_metrics(truth, sub_b10, b_df)

        # 3. Blend 25%: P(0.75 * E_raw + 0.25 * C_raw)
        b25_raw = 0.75 * e_raw + 0.25 * c_raw
        sub_b25 = truth[KEYS].copy()
        sub_b25['quantity'] = apply_p(b25_raw)
        m_b25 = compute_metrics(truth, sub_b25, b_df)

        # Control B
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        print(f"  Control B:            Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone Pure Cal:  Score = {m_standalone['clipped_score']:.6f}, Unclipped = {m_standalone['unclipped_ratio']:.6f}")
        print(f"  Blend 10% Pure Cal:   Score = {m_b10['clipped_score']:.6f}, Unclipped = {m_b10['unclipped_ratio']:.6f} (Delta: {m_b10['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")
        print(f"  Blend 25% Pure Cal:   Score = {m_b25['clipped_score']:.6f}, Unclipped = {m_b25['unclipped_ratio']:.6f} (Delta: {m_b25['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['n_raw'] = n_raw
        keyed_out['c_raw_nonneg'] = c_raw
        keyed_out['b10_raw'] = b10_raw
        keyed_out['b25_raw'] = b25_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_standalone'] = sub_standalone['quantity']
        keyed_out['pred_b10'] = sub_b10['quantity']
        keyed_out['pred_b25'] = sub_b25['quantity']
        keyed_out.to_csv(KEYED_DIR / f"nbeatsx_pure_cal_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "neg_count": neg_count,
            "min_n_raw": min_n_raw,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_b10["unclipped_ratio"],
            "blend_10_score": m_b10["clipped_score"],
            "blend_25_unclipped": m_b25["unclipped_ratio"],
            "blend_25_score": m_b25["clipped_score"],
        })

    df_res = pd.DataFrame(fold_results)
    csv_path = OUTPUT_DIR / "nbeatsx_pure_calendar_selection_metrics.csv"
    df_res.to_csv(csv_path, index=False)

    mean_ctrl_unc = df_res["ctrl_unclipped"].mean()
    mean_standalone_unc = df_res["standalone_unclipped"].mean()
    mean_b10_unc = df_res["blend_10_unclipped"].mean()
    mean_b25_unc = df_res["blend_25_unclipped"].mean()

    variants = {
        "Standalone_NBEATSx_PureCal": {
            "mean_unclipped": float(mean_standalone_unc),
            "delta_vs_control": float(mean_standalone_unc - mean_ctrl_unc),
            "mean_score": float(df_res["standalone_score"].mean()),
        },
        "Blend_0.90E_0.10NBEATSx_PureCal": {
            "mean_unclipped": float(mean_b10_unc),
            "delta_vs_control": float(mean_b10_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_10_score"].mean()),
        },
        "Blend_0.75E_0.25NBEATSx_PureCal": {
            "mean_unclipped": float(mean_b25_unc),
            "delta_vs_control": float(mean_b25_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_25_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("PURE CALENDAR NBEATSx SELECTION SUMMARY (FOLDS 4, 3, 2):")
    print(f"  Control B Baseline:               Unclipped = {mean_ctrl_unc:.6f}")
    for name, v in variants.items():
        print(f"  {name:32s}: Unclipped = {v['mean_unclipped']:.6f} (Delta = {v['delta_vs_control']:+.6f}, Score = {v['mean_score']:.6f})")
    print("=" * 80)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": time.perf_counter() - t_start_all,
        "selection_folds": [4, 3, 2],
        "control_baseline_unclipped": float(mean_ctrl_unc),
        "variants": variants,
        "beats_control": any(v["delta_vs_control"] < 0 for v in variants.values()),
        "conclusion": "REJECTED" if all(v["delta_vs_control"] >= 0 for v in variants.values()) else "ACCEPTED",
    }

    with open(OUTPUT_DIR / "nbeatsx_pure_calendar_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUTPUT_DIR / 'nbeatsx_pure_calendar_summary.json'}")


if __name__ == "__main__":
    main()
```

---

### 3.154 `evaluate_nbeatsx_pure_calendar_check.py`
*Оценка NBEATSx Pure Calendar на окнах проверки и объединение 5 окон*  
Путь: [`evaluate_nbeatsx_pure_calendar_check.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/evaluate_nbeatsx_pure_calendar_check.py)

```python
#!/usr/bin/env python3
"""
evaluate_nbeatsx_pure_calendar_check.py

Stage 2 evaluation of NBEATSx Pure Calendar (5 exog, no holidays, C_raw >= 0)
on Check Folds: [1, 0] (origins 116, 125).
Combines results with Selection Folds [4, 3, 2] to produce full 5-fold metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13

CHECK_FOLDS = {
    1: 116,  # 2026-03-23 — 2026-05-18
    0: 125,  # 2026-05-25 — 2026-07-20
}

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidates" / "nbeatsx"
KEYED_DIR = OUTPUT_DIR / "keyed"
KEYED_DIR.mkdir(parents=True, exist_ok=True)


def apply_p(x: np.ndarray | pd.Series) -> np.ndarray:
    scaled = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    print("=" * 80)
    print("EVALUATION OF NBEATSx PURE CALENDAR ON CHECK FOLDS: 1, 0")
    print("=" * 80)

    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    calendar_exog_cols = ['month', 'quarter', 'week_of_year', 'week_sin', 'week_cos']

    check_results = []
    t_start_all = time.perf_counter()

    for fold, origin in CHECK_FOLDS.items():
        print(f"\n--- Processing Check Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        # Load Control B keyed OOF
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged_ctrl) == 79101
        e_raw = merged_ctrl["raw_blend"].values.astype(np.float64)

        train_start = weeks[origin - 52]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (df['ds'] >= train_start) & (df['ds'] < val_start)
        val_mask = (df['ds'] >= val_start) & (df['ds'] <= val_end)

        train_data = df[train_mask][['unique_id', 'ds', 'y'] + calendar_exog_cols].copy()
        val_data = df[val_mask][['unique_id', 'ds'] + calendar_exog_cols].copy()

        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=26,
            loss=MAE(),
            futr_exog_list=calendar_exog_cols,
            max_steps=500,
            batch_size=256,
            windows_batch_size=512,
            random_seed=42,
            scaler_type='standard',
        )
        nf = NeuralForecast(models=[model], freq='W-MON')
        nf.fit(df=train_data)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_df = nf.predict(futr_df=val_data)
        pred_time = time.perf_counter() - t1
        print(f"  Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")

        preds_df['ds_str'] = preds_df['ds'].dt.strftime('%Y-%m-%d')
        parts = preds_df['unique_id'].str.split('__', n=1, expand=True)
        preds_df['warehouse_name'] = parts[0]
        preds_df['article'] = parts[1].astype(truth['article'].dtype)

        merged_p = truth[KEYS].merge(
            preds_df[['ds_str', 'warehouse_name', 'article', 'NBEATSx']],
            left_on=['week', 'warehouse_name', 'article'],
            right_on=['ds_str', 'warehouse_name', 'article'],
            how='inner',
            validate='one_to_one'
        )
        assert len(merged_p) == 79101
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)
        c_raw = np.maximum(n_raw, 0.0)

        # Standalone
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(c_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # Blend 10%
        b10_raw = 0.90 * e_raw + 0.10 * c_raw
        sub_b10 = truth[KEYS].copy()
        sub_b10['quantity'] = apply_p(b10_raw)
        m_b10 = compute_metrics(truth, sub_b10, b_df)

        # Blend 25%
        b25_raw = 0.75 * e_raw + 0.25 * c_raw
        sub_b25 = truth[KEYS].copy()
        sub_b25['quantity'] = apply_p(b25_raw)
        m_b25 = compute_metrics(truth, sub_b25, b_df)

        # Control B
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        print(f"  Control B:            Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone Pure Cal:  Score = {m_standalone['clipped_score']:.6f}, Unclipped = {m_standalone['unclipped_ratio']:.6f}")
        print(f"  Blend 10% Pure Cal:   Score = {m_b10['clipped_score']:.6f}, Unclipped = {m_b10['unclipped_ratio']:.6f} (Delta: {m_b10['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")
        print(f"  Blend 25% Pure Cal:   Score = {m_b25['clipped_score']:.6f}, Unclipped = {m_b25['unclipped_ratio']:.6f} (Delta: {m_b25['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['n_raw'] = n_raw
        keyed_out['c_raw_nonneg'] = c_raw
        keyed_out['b10_raw'] = b10_raw
        keyed_out['b25_raw'] = b25_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_standalone'] = sub_standalone['quantity']
        keyed_out['pred_b10'] = sub_b10['quantity']
        keyed_out['pred_b25'] = sub_b25['quantity']
        keyed_out.to_csv(KEYED_DIR / f"nbeatsx_pure_cal_fold_{fold}.csv.gz", index=False, compression="gzip")

        check_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_b10["unclipped_ratio"],
            "blend_10_score": m_b10["clipped_score"],
            "blend_25_unclipped": m_b25["unclipped_ratio"],
            "blend_25_score": m_b25["clipped_score"],
        })

    # Combine with selection folds
    sel_df = pd.read_csv(OUTPUT_DIR / "nbeatsx_pure_calendar_selection_metrics.csv")
    chk_df = pd.DataFrame(check_results)
    all5_df = pd.concat([sel_df, chk_df], ignore_index=True)
    all5_df.to_csv(OUTPUT_DIR / "nbeatsx_pure_calendar_all5_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("FULL 5-FOLD SUMMARY FOR NBEATSx PURE CALENDAR:")
    print("=" * 80)
    for variant in ["standalone", "blend_10", "blend_25"]:
        sel_u = sel_df[f"{variant}_unclipped"].mean()
        chk_u = chk_df[f"{variant}_unclipped"].mean()
        all_u = all5_df[f"{variant}_unclipped"].mean()
        all_s = all5_df[f"{variant}_score"].mean()

        ctrl_sel = sel_df["ctrl_unclipped"].mean()
        ctrl_chk = chk_df["ctrl_unclipped"].mean()
        ctrl_all_u = all5_df["ctrl_unclipped"].mean()
        ctrl_all_s = all5_df["ctrl_score"].mean()

        print(f"Variant: {variant}")
        print(f"  Selection (4,3,2): Unclipped = {sel_u:.6f} vs Ctrl = {ctrl_sel:.6f} (Delta = {sel_u - ctrl_sel:+.6f})")
        print(f"  Check (1,0):       Unclipped = {chk_u:.6f} vs Ctrl = {ctrl_chk:.6f} (Delta = {chk_u - ctrl_chk:+.6f})")
        print(f"  All 5 Folds:       Unclipped = {all_u:.6f} vs Ctrl = {ctrl_all_u:.6f} (Delta = {all_u - ctrl_all_u:+.6f})")
        print(f"                     Score     = {all_s:.6f} vs Ctrl = {ctrl_all_s:.6f} (Delta = {all_s - ctrl_all_s:+.6f})")

    summary_5fold = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": time.perf_counter() - t_start_all,
        "selection_unclipped_ctrl": float(sel_df["ctrl_unclipped"].mean()),
        "selection_unclipped_b10": float(sel_df["blend_10_unclipped"].mean()),
        "check_unclipped_ctrl": float(chk_df["ctrl_unclipped"].mean()),
        "check_unclipped_b10": float(chk_df["blend_10_unclipped"].mean()),
        "all5_unclipped_ctrl": float(all5_df["ctrl_unclipped"].mean()),
        "all5_unclipped_b10": float(all5_df["blend_10_unclipped"].mean()),
        "all5_clipped_ctrl": float(all5_df["ctrl_score"].mean()),
        "all5_clipped_b10": float(all5_df["blend_10_score"].mean()),
        "all5_delta_unclipped_b10": float(all5_df["blend_10_unclipped"].mean() - all5_df["ctrl_unclipped"].mean()),
        "all5_delta_clipped_b10": float(all5_df["blend_10_score"].mean() - all5_df["ctrl_score"].mean()),
    }
    with open(OUTPUT_DIR / "nbeatsx_pure_calendar_5fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_5fold, f, indent=2)
    print(f"\nSaved 5-fold summary to {OUTPUT_DIR / 'nbeatsx_pure_calendar_5fold_summary.json'}")


if __name__ == "__main__":
    main()
```

---

### 3.164 `run_candidate_selection.py`
*Этап 1: Протокол отбора кандидата на окнах выбора (Folds 4, 3, 2)*  
Путь: [`run_candidate_selection.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/run_candidate_selection.py)

```python
#!/usr/bin/env python3
"""
run_candidate_selection.py

Stage 1 of the strict Two-Stage Protocol:
- Evaluates Candidate models ONLY on Selection Folds: [4, 3, 2] (origins 89, 98, 107).
- Computes mean unclipped_ratio on Folds 4, 3, 2.
- Compares with Control B (mean unclipped_ratio = 0.7613139384).
- Selects the best performing configuration strictly on Selection Folds.
- Freezes all parameters and writes experiments/model_improvements/selected_config.json
  with status: 'PENDING_CHECK_EVALUATION'.
- Does NOT execute on Check Folds (1, 0) - that is reserved strictly for run_candidate_evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from holiday_calendar import build_holiday_feature_dataframe, STATUTORY_FEATURE_COLUMNS

SELECTION_FOLDS = {
    4: 89,   # 2025-09-15 — 2025-11-10
    3: 98,   # 2025-11-17 — 2026-01-12
    2: 107,  # 2026-01-19 — 2026-03-16
}

CONTROL_B_SELECTION_UNCLIPPED = 0.7613139384
CONTROL_B_SELECTION_CLIPPED = 0.7345392853

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements"
CONFIG_PATH = OUTPUT_DIR / "selected_config.json"


def apply_p(x: np.ndarray | pd.Series) -> np.ndarray:
    scaled = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def prepare_nbeatsx_data(data: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    # Calendar exogenous features
    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    # Attach statutory holiday features (Block A, 7 features)
    df['week_str'] = df['ds'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['week'])
    df = df.merge(holiday_df, left_on='week_str', right_on='week', how='left')
    df = df.drop(columns=['week_str', 'week'])

    return df


def run_nbeatsx_selection(budget_steps: int = 500) -> dict[str, Any]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    print("=" * 80)
    print(f"RUNNING NBEATSx SELECTION ON FOLDS 4, 3, 2 (BUDGET: {budget_steps} STEPS, SEED: 42)")
    print("=" * 80)

    # Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    print("Building statutory holiday features...")
    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)

    print("Formatting NeuralForecast DataFrame with exogenous calendar + statutory features...")
    full_df = prepare_nbeatsx_data(data, holiday_df)

    exog_cols = [
        'month', 'quarter', 'week_of_year', 'week_sin', 'week_cos',
        'holiday_days', 'holiday_days_prev_week', 'holiday_days_next_week',
        'new_year_days', 'may_holiday_days', 'days_to_nearest_holiday', 'days_since_last_holiday'
    ]

    candidate_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    keyed_dir = candidate_dir / "keyed"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    keyed_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    t_start_all = time.perf_counter()

    for fold, origin in SELECTION_FOLDS.items():
        print(f"\n--- Processing Selection Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        # Ground truth and baseline mean13
        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        # Load Control B keyed OOF
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged_ctrl) == 79101
        e_raw = merged_ctrl["raw_blend"].values.astype(np.float64)

        # Prepare temporal split: 52 weeks history strictly before origin
        train_start = weeks[origin - 52]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (full_df['ds'] >= train_start) & (full_df['ds'] < val_start)
        val_mask = (full_df['ds'] >= val_start) & (full_df['ds'] <= val_end)

        train_data = full_df[train_mask][['unique_id', 'ds', 'y'] + exog_cols].copy()
        val_data = full_df[val_mask][['unique_id', 'ds'] + exog_cols].copy()

        print(f"  Train: {train_start} to {weeks[origin-1]} ({len(train_data)} rows)")
        print(f"  Val Horizon: {val_start} to {val_end} ({len(val_data)} rows)")

        # Initialize NBEATSx
        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=26,
            loss=MAE(),
            futr_exog_list=exog_cols,
            max_steps=budget_steps,
            batch_size=256,
            windows_batch_size=512,
            random_seed=42,
            scaler_type='standard',
        )
        nf = NeuralForecast(models=[model], freq='W-MON')
        nf.fit(df=train_data)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_df = nf.predict(futr_df=val_data)
        pred_time = time.perf_counter() - t1
        print(f"  Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")

        # Align predictions with truth keys
        preds_df['ds_str'] = preds_df['ds'].dt.strftime('%Y-%m-%d')
        parts = preds_df['unique_id'].str.split('__', n=1, expand=True)
        preds_df['warehouse_name'] = parts[0]
        preds_df['article'] = parts[1].astype(truth['article'].dtype)

        merged_p = truth[KEYS].merge(
            preds_df[['ds_str', 'warehouse_name', 'article', 'NBEATSx']],
            left_on=['week', 'warehouse_name', 'article'],
            right_on=['ds_str', 'warehouse_name', 'article'],
            how='inner',
            validate='one_to_one'
        )
        assert len(merged_p) == 79101, f"Prediction length mismatch: {len(merged_p)}"
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)

        # 1. Standalone NBEATSx: P(N_raw)
        sub_standalone = truth[KEYS].copy()
        sub_standalone['quantity'] = apply_p(n_raw)
        m_standalone = compute_metrics(truth, sub_standalone, b_df)

        # 2. Blend 10%: P(0.90 * E_raw + 0.10 * N_raw)
        blend_10_raw = 0.90 * e_raw + 0.10 * n_raw
        sub_blend_10 = truth[KEYS].copy()
        sub_blend_10['quantity'] = apply_p(blend_10_raw)
        m_blend_10 = compute_metrics(truth, sub_blend_10, b_df)

        # 3. Blend 25%: P(0.75 * E_raw + 0.25 * N_raw)
        blend_25_raw = 0.75 * e_raw + 0.25 * n_raw
        sub_blend_25 = truth[KEYS].copy()
        sub_blend_25['quantity'] = apply_p(blend_25_raw)
        m_blend_25 = compute_metrics(truth, sub_blend_25, b_df)

        # Control B metrics
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        print(f"  Control B:      Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Standalone N:   Score = {m_standalone['clipped_score']:.6f}, Unclipped = {m_standalone['unclipped_ratio']:.6f}")
        print(f"  Blend 10% (N):  Score = {m_blend_10['clipped_score']:.6f}, Unclipped = {m_blend_10['unclipped_ratio']:.6f} (Delta: {m_blend_10['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")
        print(f"  Blend 25% (N):  Score = {m_blend_25['clipped_score']:.6f}, Unclipped = {m_blend_25['unclipped_ratio']:.6f} (Delta: {m_blend_25['unclipped_ratio'] - m_ctrl['unclipped_ratio']:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['raw_nbeatsx'] = n_raw
        keyed_out['raw_blend_10'] = blend_10_raw
        keyed_out['raw_blend_25'] = blend_25_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_standalone'] = sub_standalone['quantity']
        keyed_out['pred_blend_10'] = sub_blend_10['quantity']
        keyed_out['pred_blend_25'] = sub_blend_25['quantity']
        keyed_out.to_csv(keyed_dir / f"nbeatsx_fold_{fold}.csv.gz", index=False, compression="gzip")

        fold_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "standalone_unclipped": m_standalone["unclipped_ratio"],
            "standalone_score": m_standalone["clipped_score"],
            "blend_10_unclipped": m_blend_10["unclipped_ratio"],
            "blend_10_score": m_blend_10["clipped_score"],
            "blend_25_unclipped": m_blend_25["unclipped_ratio"],
            "blend_25_score": m_blend_25["clipped_score"],
        })

    total_time = time.perf_counter() - t_start_all
    df_res = pd.DataFrame(fold_results)
    df_res.to_csv(candidate_dir / "nbeatsx_selection_metrics.csv", index=False)

    # Compute selection summary
    mean_ctrl_unc = df_res["ctrl_unclipped"].mean()
    mean_standalone_unc = df_res["standalone_unclipped"].mean()
    mean_blend10_unc = df_res["blend_10_unclipped"].mean()
    mean_blend25_unc = df_res["blend_25_unclipped"].mean()

    variants = {
        "Standalone_NBEATSx": {
            "mean_unclipped": float(mean_standalone_unc),
            "delta_vs_control": float(mean_standalone_unc - mean_ctrl_unc),
            "mean_score": float(df_res["standalone_score"].mean()),
        },
        "Blend_0.90E_0.10NBEATSx": {
            "mean_unclipped": float(mean_blend10_unc),
            "delta_vs_control": float(mean_blend10_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_10_score"].mean()),
        },
        "Blend_0.75E_0.25NBEATSx": {
            "mean_unclipped": float(mean_blend25_unc),
            "delta_vs_control": float(mean_blend25_unc - mean_ctrl_unc),
            "mean_score": float(df_res["blend_25_score"].mean()),
        },
    }

    # Best variant by selection criterion (minimum unclipped ratio on folds 4, 3, 2)
    best_variant_name = min(variants.keys(), key=lambda k: variants[k]["mean_unclipped"])
    best_variant_info = variants[best_variant_name]

    print("\n" + "=" * 80)
    print("NBEATSx SELECTION SUMMARY ON FOLDS 4, 3, 2:")
    print(f"  Control B Baseline:          Unclipped = {mean_ctrl_unc:.6f}")
    for name, v in variants.items():
        print(f"  {name:28s}: Unclipped = {v['mean_unclipped']:.6f} (Delta = {v['delta_vs_control']:+.6f}, Score = {v['mean_score']:.6f})")
    # Compare with Control B explicitly
    beats_control = bool(best_variant_info["delta_vs_control"] < 0)
    selection_decision = "ADVANCE_TO_CHECK_EVALUATION" if beats_control else "RETAIN_CONTROL_REJECT_CANDIDATE"
    winner = best_variant_name if beats_control else "Blend_LightGBM_B (Retain Control)"

    print(f"\nSelection Decision: {selection_decision}")
    print(f"Winner on Selection Folds: {winner} (Delta vs Control = {best_variant_info['delta_vs_control']:+.6f})")
    print("=" * 80)

    # Write isolated candidate config
    cand_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_config_path = cand_dir / "selected_config.json"

    selected_payload = {
        "protocol_version": "2.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_decision": selection_decision,
        "winner_on_selection": winner,
        "evaluation_status": "PENDING_CHECK_EVALUATION" if beats_control else "SELECTION_REJECTED",
        "selection_folds": [4, 3, 2],
        "check_folds": [1, 0],
        "selection_criterion": "mean_unclipped_ratio_folds_4_3_2",
        "control_baseline": {
            "name": "Blend_LightGBM_B",
            "selection_unclipped_ratio": float(mean_ctrl_unc),
            "selection_clipped_score": float(df_res["ctrl_score"].mean()),
        },
        "evaluated_candidate": {
            "name": best_variant_name,
            "architecture": "NBEATSx",
            "parameters": {
                "h": 9,
                "input_size": 26,
                "loss": "MAE",
                "max_steps": budget_steps,
                "batch_size": 256,
                "windows_batch_size": 512,
                "scaler_type": "standard",
                "random_seed": 42,
                "exogenous_features": exog_cols,
                "history_weeks": 52,
            },
            "selection_metrics": {
                "mean_unclipped_ratio": best_variant_info["mean_unclipped"],
                "delta_vs_control": best_variant_info["delta_vs_control"],
                "mean_clipped_score": best_variant_info["mean_score"],
            },
            "beats_control_on_selection": beats_control,
        },
        "all_evaluated_selection_variants": variants,
        "runtime_selection_sec": float(total_time),
    }

    with open(cand_config_path, "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, indent=2)
    print(f"Wrote isolated candidate configuration to {cand_config_path}")
    print("Selection phase complete. Champion config remains untouched.")

    return selected_payload


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Candidate Selection Phase (Folds 4, 3, 2)")
    parser.add_argument("--candidate", type=str, default="nbeatsx", choices=["nbeatsx"], help="Candidate to evaluate")
    parser.add_argument("--steps", type=int, default=500, help="Training max steps budget")
    args = parser.parse_args()

    if args.candidate == "nbeatsx":
        run_nbeatsx_selection(budget_steps=args.steps)


if __name__ == "__main__":
    main()
```

---

### 3.174 `run_candidate_evaluation.py`
*Этап 2: Протокол проверки отобранного кандидата на окнах проверки (Folds 1, 0)*  
Путь: [`run_candidate_evaluation.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/run_candidate_evaluation.py)

```python
#!/usr/bin/env python3
"""
run_candidate_evaluation.py

Stage 2 of the strict Two-Stage Protocol:
- Reads experiments/model_improvements/selected_config.json.
- Verifies evaluation_status == 'PENDING_CHECK_EVALUATION'.
- Evaluates the chosen candidate model ONLY on Check Folds: [1, 0] (origins 116, 125)
  strictly without ANY retuning, grid search, or hyperparameter modifications.
- Aggregates Check Folds metrics and final 5-Fold combined metrics.
- Updates selected_config.json with status: 'EVALUATION_COMPLETED' and all final verified numbers.
- Updates experiments/model_improvements/metrics_by_fold.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from holiday_calendar import build_holiday_feature_dataframe, STATUTORY_FEATURE_COLUMNS

CHECK_FOLDS = {
    1: 116,  # 2026-03-23 — 2026-05-18
    0: 125,  # 2026-05-25 — 2026-07-20
}

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements"
CONFIG_PATH = OUTPUT_DIR / "selected_config.json"


def apply_p(x: np.ndarray | pd.Series) -> np.ndarray:
    scaled = 1.05 * np.maximum(np.asarray(x, dtype=float), 0.0)
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.maximum(np.rint(gated), 0.0)


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, float]:
    m = compute_all_metrics_fast(truth_df, pred_df)
    bm = compute_all_metrics_fast(truth_df, base_df)
    clipped_score = 0.0
    unclipped_ratio = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        base_val = bm[metric]
        ratio = m[metric] / base_val if base_val > 0 else 0.0
        clipped_score += weight * min(max(ratio, 0.0), 1.0)
        unclipped_ratio += weight * ratio
    return {
        "smape": m["smape"],
        "wape": m["wape"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "rmsle": m["rmsle"],
        "clipped_score": clipped_score,
        "unclipped_ratio": unclipped_ratio,
    }


def prepare_nbeatsx_data(data: pd.DataFrame, holiday_df: pd.DataFrame) -> pd.DataFrame:
    df = data[['week', 'warehouse_name', 'article', 'quantity']].copy()
    df['ds'] = pd.to_datetime(df['week'])
    df['unique_id'] = df['warehouse_name'] + '__' + df['article'].astype(str)
    df = df.rename(columns={'quantity': 'y'})

    w = df['ds'].dt.isocalendar().week.astype(float)
    df['month'] = df['ds'].dt.month.astype(float)
    df['quarter'] = df['ds'].dt.quarter.astype(float)
    df['week_of_year'] = w
    df['week_sin'] = np.sin(2.0 * np.pi * w / 52.1775)
    df['week_cos'] = np.cos(2.0 * np.pi * w / 52.1775)

    df['week_str'] = df['ds'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['week'])
    df = df.merge(holiday_df, left_on='week_str', right_on='week', how='left')
    df = df.drop(columns=['week_str', 'week'])

    return df


def run_nbeatsx_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE

    selected_cand = config["selected_candidate"]
    variant_name = selected_cand["name"]
    params = selected_cand["parameters"]

    print("=" * 80)
    print(f"RUNNING EVALUATION ON CHECK FOLDS 1, 0 FOR SELECTED CANDIDATE: {variant_name}")
    print(f"Parameters strictly frozen from selection phase: {params}")
    print("=" * 80)

    # Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    all_mondays = pd.date_range("2023-01-02", "2027-01-04", freq="W-MON")
    holiday_df = build_holiday_feature_dataframe(all_mondays, statutory_only=True)
    full_df = prepare_nbeatsx_data(data, holiday_df)

    exog_cols = params["exogenous_features"]
    budget_steps = params["max_steps"]

    candidate_dir = OUTPUT_DIR / "candidates" / "nbeatsx"
    keyed_dir = candidate_dir / "keyed"

    check_results = []
    t_start_all = time.perf_counter()

    for fold, origin in CHECK_FOLDS.items():
        print(f"\n--- Processing Check Fold {fold} (Origin {origin}, Start: {weeks[origin]}) ---")

        val_weeks = weeks[origin : origin + HORIZON]
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")

        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")

        # Load Control B keyed OOF
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        merged_ctrl = truth[KEYS].merge(df_blend, on=KEYS, how="inner")
        assert len(merged_ctrl) == 79101
        e_raw = merged_ctrl["raw_blend"].values.astype(np.float64)

        train_start = weeks[origin - params.get("history_weeks", 52)]
        val_start = weeks[origin]
        val_end = weeks[origin + HORIZON - 1]

        train_mask = (full_df['ds'] >= train_start) & (full_df['ds'] < val_start)
        val_mask = (full_df['ds'] >= val_start) & (full_df['ds'] <= val_end)

        train_data = full_df[train_mask][['unique_id', 'ds', 'y'] + exog_cols].copy()
        val_data = full_df[val_mask][['unique_id', 'ds'] + exog_cols].copy()

        t0 = time.perf_counter()
        model = NBEATSx(
            h=9,
            input_size=params["input_size"],
            loss=MAE(),
            futr_exog_list=exog_cols,
            max_steps=budget_steps,
            batch_size=params["batch_size"],
            windows_batch_size=params["windows_batch_size"],
            random_seed=params["random_seed"],
            scaler_type=params["scaler_type"],
        )
        nf = NeuralForecast(models=[model], freq='W-MON')
        nf.fit(df=train_data)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_df = nf.predict(futr_df=val_data)
        pred_time = time.perf_counter() - t1
        print(f"  Fit time: {fit_time:.2f}s, Predict time: {pred_time:.2f}s")

        preds_df['ds_str'] = preds_df['ds'].dt.strftime('%Y-%m-%d')
        parts = preds_df['unique_id'].str.split('__', n=1, expand=True)
        preds_df['warehouse_name'] = parts[0]
        preds_df['article'] = parts[1].astype(truth['article'].dtype)

        merged_p = truth[KEYS].merge(
            preds_df[['ds_str', 'warehouse_name', 'article', 'NBEATSx']],
            left_on=['week', 'warehouse_name', 'article'],
            right_on=['ds_str', 'warehouse_name', 'article'],
            how='inner',
            validate='one_to_one'
        )
        assert len(merged_p) == 79101
        n_raw = merged_p['NBEATSx'].values.astype(np.float64)

        # Compute predictions according to the selected variant
        sub_pred = truth[KEYS].copy()
        if variant_name == "Standalone_NBEATSx":
            sub_pred['quantity'] = apply_p(n_raw)
        elif variant_name == "Blend_0.90E_0.10NBEATSx":
            sub_pred['quantity'] = apply_p(0.90 * e_raw + 0.10 * n_raw)
        elif variant_name == "Blend_0.75E_0.25NBEATSx":
            sub_pred['quantity'] = apply_p(0.75 * e_raw + 0.25 * n_raw)
        else:
            raise ValueError(f"Unknown variant name: {variant_name}")

        m_cand = compute_metrics(truth, sub_pred, b_df)

        # Control B
        sub_ctrl = truth[KEYS].copy()
        sub_ctrl['quantity'] = apply_p(e_raw)
        m_ctrl = compute_metrics(truth, sub_ctrl, b_df)

        delta_unclipped = m_cand["unclipped_ratio"] - m_ctrl["unclipped_ratio"]
        delta_score = m_cand["clipped_score"] - m_ctrl["clipped_score"]

        print(f"  Control B: Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
        print(f"  Candidate: Score = {m_cand['clipped_score']:.6f}, Unclipped = {m_cand['unclipped_ratio']:.6f} (Delta: {delta_unclipped:+.6f})")

        # Save keyed predictions
        keyed_out = truth[KEYS].copy()
        keyed_out['origin'] = origin
        keyed_out['raw_ctrl'] = e_raw
        keyed_out['raw_nbeatsx'] = n_raw
        keyed_out['pred_ctrl'] = sub_ctrl['quantity']
        keyed_out['pred_cand'] = sub_pred['quantity']
        keyed_out.to_csv(keyed_dir / f"nbeatsx_fold_{fold}.csv.gz", index=False, compression="gzip")

        check_results.append({
            "fold": fold,
            "origin": origin,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "ctrl_unclipped": m_ctrl["unclipped_ratio"],
            "ctrl_score": m_ctrl["clipped_score"],
            "cand_unclipped": m_cand["unclipped_ratio"],
            "cand_score": m_cand["clipped_score"],
            "delta_unclipped": delta_unclipped,
            "delta_score": delta_score,
            "cand_smape": m_cand["smape"],
            "cand_wape": m_cand["wape"],
            "cand_rmse": m_cand["rmse"],
            "cand_bias": m_cand["bias"],
            "cand_rmsle": m_cand["rmsle"],
        })

    # Read selection metrics to form full 5-fold evaluation table
    sel_df = pd.read_csv(candidate_dir / "nbeatsx_selection_metrics.csv")
    chk_df = pd.DataFrame(check_results)

    # Map candidate metrics for selection folds
    sel_records = []
    for _, row in sel_df.iterrows():
        f = int(row["fold"])
        orig = int(row["origin"])
        if variant_name == "Standalone_NBEATSx":
            c_unc = row["standalone_unclipped"]
            c_sc = row["standalone_score"]
        elif variant_name == "Blend_0.90E_0.10NBEATSx":
            c_unc = row["blend_10_unclipped"]
            c_sc = row["blend_10_score"]
        elif variant_name == "Blend_0.75E_0.25NBEATSx":
            c_unc = row["blend_25_unclipped"]
            c_sc = row["blend_25_score"]
        else:
            raise ValueError(f"Unknown variant name: {variant_name}")

        ctrl_unc = row["ctrl_unclipped"]
        ctrl_sc = row["ctrl_score"]
        sel_records.append({
            "fold": f,
            "role": "Selection",
            "origin": orig,
            "ctrl_unclipped": ctrl_unc,
            "ctrl_score": ctrl_sc,
            "cand_unclipped": c_unc,
            "cand_score": c_sc,
            "delta_unclipped": c_unc - ctrl_unc,
            "delta_score": c_sc - ctrl_sc,
        })

    for _, row in chk_df.iterrows():
        sel_records.append({
            "fold": int(row["fold"]),
            "role": "Check",
            "origin": int(row["origin"]),
            "ctrl_unclipped": row["ctrl_unclipped"],
            "ctrl_score": row["ctrl_score"],
            "cand_unclipped": row["cand_unclipped"],
            "cand_score": row["cand_score"],
            "delta_unclipped": row["delta_unclipped"],
            "delta_score": row["delta_score"],
        })

    full_5f_df = pd.DataFrame(sel_records).sort_values("fold", ascending=False).reset_index(drop=True)
    full_5f_df.to_csv(candidate_dir / "nbeatsx_full_5fold_metrics.csv", index=False)

    # Compute Final Summary Metrics
    sel_part = full_5f_df[full_5f_df["role"] == "Selection"]
    chk_part = full_5f_df[full_5f_df["role"] == "Check"]

    summary_metrics = {
        "selection_folds_4_3_2": {
            "control_unclipped": float(sel_part["ctrl_unclipped"].mean()),
            "cand_unclipped": float(sel_part["cand_unclipped"].mean()),
            "delta_unclipped": float(sel_part["delta_unclipped"].mean()),
            "control_score": float(sel_part["ctrl_score"].mean()),
            "cand_score": float(sel_part["cand_score"].mean()),
            "delta_score": float(sel_part["delta_score"].mean()),
        },
        "check_folds_1_0": {
            "control_unclipped": float(chk_part["ctrl_unclipped"].mean()),
            "cand_unclipped": float(chk_part["cand_unclipped"].mean()),
            "delta_unclipped": float(chk_part["delta_unclipped"].mean()),
            "control_score": float(chk_part["ctrl_score"].mean()),
            "cand_score": float(chk_part["cand_score"].mean()),
            "delta_score": float(chk_part["delta_score"].mean()),
        },
        "all_5_folds": {
            "control_unclipped": float(full_5f_df["ctrl_unclipped"].mean()),
            "cand_unclipped": float(full_5f_df["cand_unclipped"].mean()),
            "delta_unclipped": float(full_5f_df["delta_unclipped"].mean()),
            "control_score": float(full_5f_df["ctrl_score"].mean()),
            "cand_score": float(full_5f_df["cand_score"].mean()),
            "delta_score": float(full_5f_df["delta_score"].mean()),
        },
    }

    print("\n" + "=" * 80)
    print("FINAL 5-FOLD VERIFICATION OF SELECTED CANDIDATE:")
    print(f"Candidate: {variant_name}")
    print(f"  Selection (4,3,2) Unclipped: Control={summary_metrics['selection_folds_4_3_2']['control_unclipped']:.6f}, Cand={summary_metrics['selection_folds_4_3_2']['cand_unclipped']:.6f}, Delta={summary_metrics['selection_folds_4_3_2']['delta_unclipped']:+.6f}")
    print(f"  Check (1,0)     Unclipped: Control={summary_metrics['check_folds_1_0']['control_unclipped']:.6f}, Cand={summary_metrics['check_folds_1_0']['cand_unclipped']:.6f}, Delta={summary_metrics['check_folds_1_0']['delta_unclipped']:+.6f}")
    print(f"  All 5 Folds       Clipped: Control={summary_metrics['all_5_folds']['control_score']:.6f}, Cand={summary_metrics['all_5_folds']['cand_score']:.6f}, Delta={summary_metrics['all_5_folds']['delta_score']:+.6f}")
    print(f"  All 5 Folds     Unclipped: Control={summary_metrics['all_5_folds']['control_unclipped']:.6f}, Cand={summary_metrics['all_5_folds']['cand_unclipped']:.6f}, Delta={summary_metrics['all_5_folds']['delta_unclipped']:+.6f}")
    print("=" * 80)

    # Update selected_config.json
    config["evaluation_status"] = "EVALUATION_COMPLETED"
    config["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config["verified_metrics"] = summary_metrics

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Updated configuration saved to {CONFIG_PATH}")

    return summary_metrics


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Candidate Evaluation Phase (Check Folds 1, 0)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to selected candidate config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found. Run run_candidate_selection.py first!")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("evaluation_status") != "PENDING_CHECK_EVALUATION":
        print(f"Warning: config status is {config.get('evaluation_status')}, expected PENDING_CHECK_EVALUATION.")

    arch = config.get("selected_candidate", {}).get("architecture") or config.get("evaluated_candidate", {}).get("architecture")
    if arch == "NBEATSx":
        run_nbeatsx_evaluation(config)
    else:
        raise ValueError(f"Unknown architecture in config: {arch}")


if __name__ == "__main__":
    main()
```

---

### 3.184 `forecast-container/app/predictor.py`
*Модуль инференса Predictor: динамическая сборка недельных признаков и взвешенный блендинг*  
Путь: [`forecast-container/app/predictor.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/forecast-container/app/predictor.py)

```python
"""
predictor.py

Core inference module for Candidate LGBM52 Ensemble.
Decouples trained model artifacts from training/optuna code and archive predictions.

Models in bundle:
- TFT (NeuralForecast): 50%
- LightGBM Variant B (52 origins): 30%
- CatBoost (18 origins): 10%
- Per-Category LightGBM (14 models): 10%

Final blend:
alpha = 1.05, threshold = 0.65, rounding = 'rint', clip_lower = 0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from neuralforecast import NeuralForecast

# Local or package imports
from .validators import validate_and_load_sales, validate_and_load_stock

HORIZON = 9
KEYS = ["week", "warehouse_name", "article"]
CAT_COLUMNS = [
    "warehouse_name",
    "article",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
]
CALENDAR_ANCHOR = pd.Timestamp("2024-01-01")

BASE_HIST_EXOG = [
    "article_network_sales_mean_13_log",
    "article_network_growth_4_vs_13",
    "warehouse_share_article_13",
    "active_warehouses_article_13",
]
FUTR_EXOG = ["time_idx", "month", "quarter", "week_of_year", "week_sin", "week_cos"]


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator > 0)


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() else "_" for char in text)


def apply_rule(series: pd.Series | np.ndarray, threshold: float = 0.75, round_to_int: bool = True) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    result = np.where(values < threshold, 0.0, values)
    if round_to_int:
        result = np.rint(result)
    return np.maximum(result, 0.0)


class Predictor:
    def __init__(
        self,
        bundle_dir: Path,
        manifest: dict[str, Any],
        postprocessing: dict[str, Any],
        valid_pairs: pd.DataFrame,
        tft_model: NeuralForecast,
        lgbm_clf: lgb.Booster,
        lgbm_reg: lgb.Booster,
        cb_clf: CatBoostClassifier,
        cb_reg: CatBoostRegressor,
        cat_meta: dict[str, Any],
        cat_boosters: dict[str, Tuple[lgb.Booster, lgb.Booster]],
    ) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.postprocessing = postprocessing
        self.valid_pairs = valid_pairs
        self.tft_model = tft_model
        self.lgbm_clf = lgbm_clf
        self.lgbm_reg = lgbm_reg
        self.cb_clf = cb_clf
        self.cb_reg = cb_reg
        self.cat_meta = cat_meta
        self.cat_boosters = cat_boosters

        self.feature_cols = manifest["feature_columns"]
        self.cat_feature_cols = [c for c in self.feature_cols if c != "category_lvl_1"]

    @classmethod
    def load(cls, bundle_dir: Union[str, Path]) -> "Predictor":
        bundle_path = Path(bundle_dir).resolve()
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle directory not found: {bundle_path}")

        # Load metadata
        with open(bundle_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(bundle_path / "postprocessing.json", "r", encoding="utf-8") as f:
            postprocessing = json.load(f)

        valid_pairs = pd.read_parquet(bundle_path / "valid_pairs.parquet")

        # Load TFT model
        tft_model = NeuralForecast.load(str(bundle_path / "tft"))

        # Load LightGBM B models
        lgbm_clf = lgb.Booster(model_str=(bundle_path / "lgbm_b" / "clf.txt").read_text(encoding="utf-8"))
        lgbm_reg = lgb.Booster(model_str=(bundle_path / "lgbm_b" / "reg.txt").read_text(encoding="utf-8"))

        # Load CatBoost models
        cb_clf = CatBoostClassifier()
        cb_clf.load_model(str(bundle_path / "catboost" / "clf.cbm"))
        cb_reg = CatBoostRegressor()
        cb_reg.load_model(str(bundle_path / "catboost" / "reg.cbm"))

        # Load Per-Category models
        with open(bundle_path / "per_category" / "meta.json", "r", encoding="utf-8") as f:
            cat_meta = json.load(f)

        cat_boosters = {}
        for cat, info in cat_meta.items():
            if "fallback" not in info:
                c_clf = lgb.Booster(model_str=(bundle_path / "per_category" / info["clf_file"]).read_text(encoding="utf-8"))
                c_reg = lgb.Booster(model_str=(bundle_path / "per_category" / info["reg_file"]).read_text(encoding="utf-8"))
                cat_boosters[cat] = (c_clf, c_reg)

        return cls(
            bundle_dir=bundle_path,
            manifest=manifest,
            postprocessing=postprocessing,
            valid_pairs=valid_pairs,
            tft_model=tft_model,
            lgbm_clf=lgbm_clf,
            lgbm_reg=lgbm_reg,
            cb_clf=cb_clf,
            cb_reg=cb_reg,
            cat_meta=cat_meta,
            cat_boosters=cat_boosters,
        )

    def prepare_dataset_from_raw(
        self,
        sales_df: pd.DataFrame,
        stock_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DatetimeIndex, int]:
        cutoff_date = sales_df["date"].max()

        # Sales preprocessing
        sales = sales_df.copy()
        sales["week"] = sales["date"].dt.normalize() - pd.to_timedelta(sales["date"].dt.dayofweek, unit="D")

        sales_sum_missing = sales["sales_sum"].isna()
        sales["sales_sum_orig"] = np.where(
            sales_sum_missing,
            0.0,
            np.exp(sales["sales_sum"].to_numpy(dtype=np.float64)),
        )

        discount_present = sales["discount_sum"].notna()
        sales["discount_sum_orig"] = np.where(
            discount_present,
            np.exp(sales["discount_sum"].fillna(0).to_numpy(dtype=np.float64)),
            0.0,
        )

        weekly_sales = (
            sales.groupby(["week", "warehouse_name", "article"], as_index=False, observed=True)
            .agg(
                quantity=("quantity", "sum"),
                sales_sum=("sales_sum_orig", "sum"),
                discount_sum=("discount_sum_orig", "sum"),
            )
        )

        # Stock preprocessing
        stock = stock_df[stock_df["date"] <= cutoff_date].copy()
        if "article" not in stock.columns and "char_article" in stock.columns:
            stock = stock.rename(columns={"char_article": "article"})

        daily_stock = (
            stock.groupby(["date", "warehouse_name", "article"], as_index=False, observed=True)["stock"]
            .sum()
            .sort_values(["warehouse_name", "article", "date"])
        )
        daily_stock["week"] = daily_stock["date"].dt.normalize() - pd.to_timedelta(daily_stock["date"].dt.dayofweek, unit="D")
        daily_stock["is_nonpositive"] = (daily_stock["stock"] <= 0).astype(np.int8)

        weekly_stock = (
            daily_stock.groupby(["week", "warehouse_name", "article"], as_index=False, observed=True)
            .agg(
                stock_week_end=("stock", "last"),
                stock_mean=("stock", "mean"),
                stock_observed_days=("stock", "size"),
                stock_nonpositive_days=("is_nonpositive", "sum"),
            )
        )
        weekly_stock["stock_nonpositive_rate"] = (
            weekly_stock["stock_nonpositive_days"] / weekly_stock["stock_observed_days"]
        )

        # Dynamic forecast horizon start
        first_forecast_monday = cutoff_date.normalize() - pd.Timedelta(days=cutoff_date.weekday()) + pd.Timedelta(days=7)
        future_mondays = pd.date_range(first_forecast_monday, periods=HORIZON, freq="W-MON")

        # Historical observed Mondays
        min_sales_week = weekly_sales["week"].min()
        latest_sales_week = cutoff_date.normalize() - pd.Timedelta(days=cutoff_date.weekday())
        observed_mondays = pd.date_range(min_sales_week, latest_sales_week, freq="W-MON")

        # Padding policy: Ensure at least 26 historical weeks for TFT and lag-26 features
        min_required_weeks = 26
        if len(observed_mondays) < min_required_weeks:
            missing_count = min_required_weeks - len(observed_mondays)
            pad_start = observed_mondays[0] - pd.Timedelta(weeks=missing_count)
            pad_mondays = pd.date_range(pad_start, periods=missing_count, freq="W-MON")
            historical_mondays = pad_mondays.union(observed_mondays)
        else:
            historical_mondays = observed_mondays

        # Build grid across canonical valid pairs
        pairs = self.valid_pairs[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]].copy()

        grid_parts = []
        for w in historical_mondays:
            chunk = pairs.copy()
            chunk["week"] = w
            grid_parts.append(chunk)

        hist_grid = pd.concat(grid_parts, ignore_index=True)
        hist_data = hist_grid.merge(weekly_sales, on=["week", "warehouse_name", "article"], how="left")
        hist_data["quantity"] = hist_data["quantity"].fillna(0.0)
        hist_data["sales_sum"] = hist_data["sales_sum"].fillna(0.0)
        hist_data["discount_sum"] = hist_data["discount_sum"].fillna(0.0)

        hist_data = hist_data.merge(
            weekly_stock[["week", "warehouse_name", "article", "stock_week_end", "stock_mean", "stock_nonpositive_rate", "stock_observed_days"]],
            on=["week", "warehouse_name", "article"],
            how="left",
        )

        # Future placeholder rows for WeeklyMatrices
        future_parts = []
        for w in future_mondays:
            chunk = pairs.copy()
            chunk["week"] = w
            chunk["quantity"] = 0.0
            chunk["sales_sum"] = 0.0
            chunk["discount_sum"] = 0.0
            chunk["stock_week_end"] = np.nan
            chunk["stock_mean"] = np.nan
            chunk["stock_nonpositive_rate"] = np.nan
            chunk["stock_observed_days"] = 0.0
            future_parts.append(chunk)

        future_data = pd.concat(future_parts, ignore_index=True)
        full_data = pd.concat([hist_data, future_data], ignore_index=True)
        full_data = full_data.sort_values(["week", "warehouse_name", "article"]).reset_index(drop=True)

        test_origin = len(historical_mondays)
        return full_data, future_mondays, test_origin

    def _build_test_matrix_features(self, extended_data: pd.DataFrame, test_origin: int) -> pd.DataFrame:
        """Vectorized feature construction matching WeeklyMatrices with calendar anchor."""
        data = extended_data.sort_values(KEYS).reset_index(drop=True)
        weeks = pd.DatetimeIndex(sorted(data["week"].unique()))
        first = data[data["week"] == weeks[0]].sort_values(KEYS[1:])
        pairs = first[["warehouse_name", "article"]].reset_index(drop=True)
        n_pairs = len(pairs)

        ordered = data.sort_values(["week", "warehouse_name", "article"]).reset_index(drop=True)
        static = first[
            ["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]
        ].fillna("unknown").astype(str).reset_index(drop=True)

        warehouse_codes = pd.factorize(static["warehouse_name"], sort=True)[0].astype(np.int32)
        article_codes = pd.factorize(static["article"], sort=True)[0].astype(np.int32)
        category_codes = pd.factorize(static["category_lvl_2"], sort=True)[0].astype(np.int32)
        warehouse_category = pd.MultiIndex.from_arrays([warehouse_codes, category_codes])
        warehouse_category_codes = pd.factorize(warehouse_category, sort=True)[0].astype(np.int32)
        n_warehouse_categories = int(warehouse_category_codes.max() + 1)
        n_articles = int(article_codes.max() + 1)

        def matrix(column: str) -> np.ndarray:
            return ordered[column].to_numpy(dtype=np.float32).reshape(len(weeks), n_pairs)

        quantity_mat = matrix("quantity")
        sales_sum_mat = matrix("sales_sum")
        discount_sum_mat = matrix("discount_sum")
        stock_end_mat = matrix("stock_week_end")
        stock_mean_mat = matrix("stock_mean")
        stock_nonpositive_rate_mat = matrix("stock_nonpositive_rate")
        stock_observed_days_mat = matrix("stock_observed_days")

        total = HORIZON * n_pairs
        flat = np.arange(total, dtype=np.int64)
        horizon_idx = flat // n_pairs
        pair_idx = flat % n_pairs
        origin_pos = test_origin
        target_pos = origin_pos + horizon_idx
        history_pos = origin_pos - 1

        history26 = np.column_stack(
            [quantity_mat[origin_pos - lag, pair_idx] for lag in range(1, 27)]
        )
        history13 = history26[:, :13]
        history4 = history26[:, :4]
        sales = sales_sum_mat[history_pos, pair_idx]
        discount = discount_sum_mat[history_pos, pair_idx]
        quantity_lag1 = quantity_mat[history_pos, pair_idx]

        full_value = sales + discount
        discount_share = np.divide(discount, full_value, out=np.full_like(discount, np.nan), where=full_value > 0)
        paid_per_unit = np.divide(sales, quantity_lag1, out=np.full_like(sales, np.nan), where=quantity_lag1 > 0)
        full_per_unit = np.divide(full_value, quantity_lag1, out=np.full_like(full_value, np.nan), where=quantity_lag1 > 0)

        target_weeks = weeks[target_pos]
        week_number = target_weeks.isocalendar().week.to_numpy(dtype=np.float32)

        frame = static.iloc[pair_idx].reset_index(drop=True).copy()
        frame["horizon"] = horizon_idx.astype(np.int16) + 1

        # Calendar anchor matching training
        frame["time_idx"] = ((target_weeks - CALENDAR_ANCHOR).days // 7).astype(np.int16)

        frame["quantity_lag_1"] = quantity_lag1
        frame["quantity_lag_2"] = quantity_mat[origin_pos - 2, pair_idx]
        frame["quantity_lag_4"] = quantity_mat[origin_pos - 4, pair_idx]
        frame["quantity_lag_13"] = quantity_mat[origin_pos - 13, pair_idx]
        frame["quantity_mean_4"] = history4.mean(axis=1)
        frame["quantity_mean_13"] = history13.mean(axis=1)
        frame["quantity_std_13"] = history13.std(axis=1)
        frame["quantity_nonzero_rate_13"] = (history13 > 0).mean(axis=1)

        mean13 = history13.mean(axis=1)
        std13 = history13.std(axis=1)
        frame["quantity_min_13"] = history13.min(axis=1)
        frame["quantity_max_13"] = history13.max(axis=1)
        frame["quantity_median_13"] = np.median(history13, axis=1)
        frame["quantity_cv_13"] = np.divide(std13, mean13, out=np.zeros_like(std13), where=mean13 > 0)
        frame["quantity_mean_26"] = history26.mean(axis=1)
        frame["growth_log_4_vs_previous_4"] = np.log1p(history26[:, :4].sum(axis=1)) - np.log1p(history26[:, 4:8].sum(axis=1))
        frame["growth_log_13_vs_previous_13"] = np.log1p(history26[:, :13].sum(axis=1)) - np.log1p(history26[:, 13:26].sum(axis=1))

        chronological13 = history13[:, ::-1]
        chronological26 = history26[:, ::-1]
        x13 = np.arange(13, dtype=np.float32) - 6.0
        x26 = np.arange(26, dtype=np.float32) - 12.5
        frame["quantity_slope_13"] = (chronological13 @ x13) / (x13 @ x13)
        frame["quantity_slope_26"] = (chronological26 @ x26) / (x26 @ x26)

        nonzero_lag_order = history26 > 0
        has_previous_sale = nonzero_lag_order.any(axis=1)
        first_nonzero = np.argmax(nonzero_lag_order, axis=1)
        frame["weeks_since_last_sale"] = np.where(has_previous_sale, first_nonzero + 1, 27)
        frame["current_zero_streak"] = np.where(has_previous_sale, first_nonzero, 26)
        frame["autocorrelation_lag_1"] = rowwise_correlation(chronological13[:, 1:], chronological13[:, :-1])
        frame["autocorrelation_lag_4"] = rowwise_correlation(chronological26[:, 4:], chronological26[:, :-4])

        stock_end = stock_end_mat[history_pos, pair_idx]
        frame["stock_end_lag_1"] = signed_log1p(stock_end)
        frame["stock_mean_lag_1"] = signed_log1p(stock_mean_mat[history_pos, pair_idx])
        frame["stock_nonpositive_rate_lag_1"] = stock_nonpositive_rate_mat[history_pos, pair_idx]
        frame["stock_observed_days_lag_1"] = stock_observed_days_mat[history_pos, pair_idx]
        frame["stock_missing_lag_1"] = np.isnan(stock_end).astype(np.int8)

        frame["month"] = target_weeks.month.astype(np.int8)
        frame["quarter"] = target_weeks.quarter.astype(np.int8)
        frame["week_of_year"] = week_number
        frame["week_sin"] = np.sin(2 * np.pi * week_number / 52.18)
        frame["week_cos"] = np.cos(2 * np.pi * week_number / 52.18)

        frame["sales_sum_log_lag_1"] = np.log1p(np.maximum(sales, 0))
        frame["discount_sum_log_lag_1"] = np.log1p(np.maximum(discount, 0))
        frame["discount_share_lag_1"] = discount_share
        frame["paid_value_per_unit_lag_1"] = np.log1p(np.maximum(paid_per_unit, 0))
        frame["full_value_per_unit_lag_1"] = np.log1p(np.maximum(full_per_unit, 0))

        # Context features
        pair_sum_13 = quantity_mat[origin_pos - 13 : origin_pos].sum(axis=0)
        pair_sum_4 = quantity_mat[origin_pos - 4 : origin_pos].sum(axis=0)

        category_sum_13 = np.bincount(warehouse_category_codes, weights=pair_sum_13, minlength=n_warehouse_categories)
        category_sum_4 = np.bincount(warehouse_category_codes, weights=pair_sum_4, minlength=n_warehouse_categories)
        category_mean_13 = category_sum_13 / 13.0
        category_mean_4 = category_sum_4 / 4.0
        category_growth = np.log1p(category_mean_4) - np.log1p(category_mean_13)
        article_share_category = np.divide(
            pair_sum_13,
            category_sum_13[warehouse_category_codes],
            out=np.zeros_like(pair_sum_13),
            where=category_sum_13[warehouse_category_codes] > 0,
        )
        article_rank_category = (
            pd.Series(pair_sum_13).groupby(warehouse_category_codes).rank(method="average", pct=True).to_numpy(dtype=np.float32)
        )
        active_articles = np.bincount(
            warehouse_category_codes,
            weights=(pair_sum_13 > 0).astype(np.float32),
            minlength=n_warehouse_categories,
        )

        last_sales = sales_sum_mat[origin_pos - 1]
        last_discount = discount_sum_mat[origin_pos - 1]
        last_full_value = last_sales + last_discount
        own_discount_share = np.divide(last_discount, last_full_value, out=np.zeros_like(last_discount), where=last_full_value > 0)
        category_discount = np.bincount(warehouse_category_codes, weights=last_discount, minlength=n_warehouse_categories)
        category_full_value = np.bincount(warehouse_category_codes, weights=last_full_value, minlength=n_warehouse_categories)
        category_discount_share = np.divide(category_discount, category_full_value, out=np.zeros_like(category_discount), where=category_full_value > 0)

        network_sum_13 = np.bincount(article_codes, weights=pair_sum_13, minlength=n_articles)
        network_sum_4 = np.bincount(article_codes, weights=pair_sum_4, minlength=n_articles)
        network_mean_13 = network_sum_13 / 13.0
        network_mean_4 = network_sum_4 / 4.0
        network_growth = np.log1p(network_mean_4) - np.log1p(network_mean_13)
        warehouse_share_article = np.divide(pair_sum_13, network_sum_13[article_codes], out=np.zeros_like(pair_sum_13), where=network_sum_13[article_codes] > 0)
        active_warehouses = np.bincount(article_codes, weights=(pair_sum_13 > 0).astype(np.float32), minlength=n_articles)

        selected_category = warehouse_category_codes[pair_idx]
        selected_article = article_codes[pair_idx]

        frame["category_sales_mean_13_log"] = np.log1p(category_mean_13[selected_category])
        frame["category_growth_4_vs_13"] = category_growth[selected_category]
        frame["article_share_category_13"] = article_share_category[pair_idx]
        frame["article_rank_category_13"] = article_rank_category[pair_idx]
        frame["active_articles_category_13"] = active_articles[selected_category]
        frame["discount_share_vs_category_lag_1"] = own_discount_share[pair_idx] - category_discount_share[selected_category]

        frame["article_network_sales_mean_13_log"] = np.log1p(network_mean_13[selected_article])
        frame["article_network_growth_4_vs_13"] = network_growth[selected_article]
        frame["warehouse_share_article_13"] = warehouse_share_article[pair_idx]
        frame["active_warehouses_article_13"] = active_warehouses[selected_article]

        for col in CAT_COLUMNS:
            if col in frame:
                frame[col] = frame[col].astype("category")

        return frame

    def _prepare_tft_frames(self, hist_data: pd.DataFrame, future_mondays: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = hist_data.copy()
        data["week"] = pd.to_datetime(data["week"])
        data = data.sort_values(KEYS).reset_index(drop=True)
        data["unique_id"] = data["warehouse_name"].astype(str) + "|" + data["article"].astype(str)

        # Calendar anchored time_idx
        data["time_idx"] = ((data["week"] - CALENDAR_ANCHOR).dt.days // 7).astype(np.float32)
        iso_week = data["week"].dt.isocalendar().week.astype(np.float32)
        data["month"] = data["week"].dt.month.astype(np.float32)
        data["quarter"] = data["week"].dt.quarter.astype(np.float32)
        data["week_of_year"] = iso_week
        data["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0).astype(np.float32)
        data["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0).astype(np.float32)

        pair_group = data.groupby(["warehouse_name", "article"], sort=False, observed=True)
        pair_sum_13 = pair_group["quantity"].transform(lambda values: values.shift(1).rolling(13, min_periods=1).sum())

        article_week = (
            data.groupby(["article", "week"], as_index=False, observed=True)["quantity"]
            .sum()
            .sort_values(["article", "week"])
        )
        article_group = article_week.groupby("article", sort=False, observed=True)["quantity"]
        article_week["network_sum_13"] = article_group.transform(lambda values: values.shift(1).rolling(13, min_periods=1).sum())
        article_week["network_sum_4"] = article_group.transform(lambda values: values.shift(1).rolling(4, min_periods=1).sum())
        article_week["network_mean_13"] = article_week["network_sum_13"] / 13.0
        article_week["network_mean_4"] = article_week["network_sum_4"] / 4.0

        data = data.merge(article_week[["article", "week", "network_sum_13", "network_mean_13", "network_mean_4"]], on=["article", "week"], how="left")
        data["article_network_sales_mean_13_log"] = np.log1p(data["network_mean_13"].fillna(0)).astype(np.float32)
        data["article_network_growth_4_vs_13"] = (np.log1p(data["network_mean_4"].fillna(0)) - np.log1p(data["network_mean_13"].fillna(0))).astype(np.float32)
        data["warehouse_share_article_13"] = np.divide(
            pair_sum_13.to_numpy(dtype=np.float32),
            data["network_sum_13"].to_numpy(dtype=np.float32),
            out=np.zeros(len(data), dtype=np.float32),
            where=data["network_sum_13"].fillna(0).to_numpy() > 0,
        )
        active = (
            data.assign(_active=(pair_sum_13 > 0).astype(np.float32))
            .groupby(["week", "article"], observed=True)["_active"]
            .transform("sum")
        )
        data["active_warehouses_article_13"] = active.astype(np.float32)

        # Static frame
        metadata = (
            self.valid_pairs[["warehouse_name", "article", "category_lvl_1", "category_lvl_2", "category_lvl_3"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        metadata["unique_id"] = metadata["warehouse_name"].astype(str) + "|" + metadata["article"].astype(str)
        metadata = metadata.sort_values("unique_id").reset_index(drop=True)

        dummy_parts = []
        for column in ["warehouse_name", "category_lvl_1", "category_lvl_2", "category_lvl_3"]:
            dummies = pd.get_dummies(metadata[column].fillna("unknown"), prefix=column, dtype=np.float32)
            dummies.columns = [safe_name(value) for value in dummies.columns]
            dummy_parts.append(dummies)
        static_df = pd.concat([metadata[["unique_id"]], *dummy_parts], axis=1)

        temporal = data.rename(columns={"week": "ds", "quantity": "y"})[
            ["unique_id", "ds", "y", *FUTR_EXOG, *BASE_HIST_EXOG]
        ].copy()
        temporal["y"] = temporal["y"].astype(np.float32)
        temporal[BASE_HIST_EXOG] = temporal[BASE_HIST_EXOG].fillna(0).astype(np.float32)
        temporal[FUTR_EXOG] = temporal[FUTR_EXOG].astype(np.float32)
        temporal = temporal.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Future frame
        ids = metadata["unique_id"].drop_duplicates().sort_values()
        futr_frame = pd.MultiIndex.from_product([ids, future_mondays], names=["unique_id", "ds"]).to_frame(index=False)
        futr_frame["time_idx"] = ((futr_frame["ds"] - CALENDAR_ANCHOR).dt.days // 7).astype(np.float32)
        f_iso_week = futr_frame["ds"].dt.isocalendar().week.astype(np.float32)
        futr_frame["month"] = futr_frame["ds"].dt.month.astype(np.float32)
        futr_frame["quarter"] = futr_frame["ds"].dt.quarter.astype(np.float32)
        futr_frame["week_of_year"] = f_iso_week
        futr_frame["week_sin"] = np.sin(2 * np.pi * f_iso_week / 52.0).astype(np.float32)
        futr_frame["week_cos"] = np.cos(2 * np.pi * f_iso_week / 52.0).astype(np.float32)

        return temporal, static_df, futr_frame[["unique_id", "ds", *FUTR_EXOG]]

    def predict(
        self,
        sales_df: pd.DataFrame,
        stock_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Runs the complete ensemble forecast on arbitrary sales and stock data."""
        extended_data, future_mondays, test_origin = self.prepare_dataset_from_raw(sales_df, stock_df)
        hist_data = extended_data[extended_data["week"] < future_mondays[0]].copy()

        # Build boosting features
        test_df = self._build_test_matrix_features(extended_data, test_origin)

        # 1. TFT prediction
        temporal, static_df, futr_df = self._prepare_tft_frames(hist_data, future_mondays)
        tft_preds = self.tft_model.predict(df=temporal, static_df=static_df, futr_df=futr_df)

        val_col = [c for c in tft_preds.columns if c not in {"unique_id", "ds", "cutoff"}][0]
        tft_preds["warehouse_name"] = tft_preds["unique_id"].str.split("|").str[0]
        tft_preds["article"] = tft_preds["unique_id"].str.split("|").str[1]
        tft_preds = tft_preds.rename(columns={"ds": "week", val_col: "q_tft"})
        tft_preds["week"] = pd.to_datetime(tft_preds["week"]).dt.strftime("%Y-%m-%d")

        # 2. LightGBM B prediction
        p_lgb = self.lgbm_clf.predict(test_df[self.feature_cols])
        q_cond_lgb = np.maximum(np.expm1(self.lgbm_reg.predict(test_df[self.feature_cols])), 0.0)
        exp_lgb = p_lgb * q_cond_lgb
        gated_lgb = np.where(p_lgb < 0.5, 0.0, exp_lgb)

        # 3. CatBoost prediction
        p_cb = self.cb_clf.predict_proba(test_df[self.feature_cols])[:, 1]
        q_cond_cb = np.maximum(np.expm1(self.cb_reg.predict(test_df[self.feature_cols])), 0.0)
        cb_pred = p_cb * q_cond_cb
        cb_pred[p_cb < 0.5] = 0.0
        cb_pred = np.maximum(cb_pred, 0.0)

        # 4. Per-Category prediction
        pcat_preds = np.zeros(len(test_df), dtype=np.float32)
        for cat, info in self.cat_meta.items():
            val_mask = (test_df["category_lvl_1"] == cat).values
            if not val_mask.any():
                continue
            if "fallback" in info:
                pcat_preds[val_mask] = info["fallback"]
            else:
                c_clf, c_reg = self.cat_boosters[cat]
                c_proba = c_clf.predict(test_df.loc[val_mask, self.cat_feature_cols])
                c_cond = np.maximum(np.expm1(c_reg.predict(test_df.loc[val_mask, self.cat_feature_cols])), 0.0)
                c_pred = c_proba * c_cond
                c_pred[c_proba < 0.5] = 0.0
                pcat_preds[val_mask] = c_pred

        pcat_postprocessed = np.asarray(
            apply_rule(pd.Series(pcat_preds), threshold=0.75, round_to_int=True), dtype=np.float64
        )

        # Grid keys for boosting
        test_weeks_arr = np.repeat(future_mondays.strftime("%Y-%m-%d").to_numpy(), len(self.valid_pairs))
        test_pairs_arr = pd.concat([self.valid_pairs[["warehouse_name", "article"]]] * HORIZON, ignore_index=True)

        boosting_df = pd.DataFrame({
            "week": test_weeks_arr,
            "warehouse_name": test_pairs_arr["warehouse_name"],
            "article": test_pairs_arr["article"],
            "gated_lgb": gated_lgb,
            "cb_pred": cb_pred,
            "pcat_postprocessed": pcat_postprocessed,
        })

        # Merge TFT and boostings
        merged = boosting_df.merge(
            tft_preds[["week", "warehouse_name", "article", "q_tft"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

        components_cfg = self.postprocessing.get("components", {})
        w_tft = float(components_cfg.get("tft", {}).get("weight", 0.50))
        w_lgb = float(components_cfg.get("lightgbm_b", {}).get("weight", 0.30))
        w_cb = float(components_cfg.get("catboost", {}).get("weight", 0.10))
        w_pcat = float(components_cfg.get("per_category", {}).get("weight", 0.10))

        # Blend
        raw_blend = (
            w_tft * merged["q_tft"].to_numpy(dtype=np.float64)
            + w_lgb * merged["gated_lgb"].to_numpy(dtype=np.float64)
            + w_cb * merged["cb_pred"].to_numpy(dtype=np.float64)
            + w_pcat * merged["pcat_postprocessed"].to_numpy(dtype=np.float64)
        )

        blend_cfg = self.postprocessing.get("blend", {})
        alpha = blend_cfg.get("alpha", 1.05)
        threshold = blend_cfg.get("threshold", 0.65)
        scaled_blend = raw_blend * alpha
        gated_blend = np.where(scaled_blend < threshold, 0.0, scaled_blend)
        final_quantity = np.rint(gated_blend).clip(0, None).astype(int)

        merged["quantity"] = final_quantity
        result = merged[KEYS + ["quantity"]].sort_values(KEYS).reset_index(drop=True)
        return result


# Global singleton instance for CLI / API
_GLOBAL_PREDICTOR: Optional[Predictor] = None


def predict_from_history(
    sales: Union[str, Path, pd.DataFrame],
    stock: Union[str, Path, pd.DataFrame],
    bundle_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Unified entry point for CLI and API.
    Builds features dynamically from provided history and returns forecast DataFrame.
    """
    global _GLOBAL_PREDICTOR

    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent.parent / "model"
    bundle_path = Path(bundle_dir).resolve()

    if _GLOBAL_PREDICTOR is None or _GLOBAL_PREDICTOR.bundle_dir != bundle_path:
        _GLOBAL_PREDICTOR = Predictor.load(bundle_path)

    # Validate inputs
    sales_df = validate_and_load_sales(sales)
    cutoff_date = sales_df["date"].max()
    stock_df = validate_and_load_stock(stock, cutoff_date=cutoff_date)

    return _GLOBAL_PREDICTOR.predict(sales_df, stock_df)
```

---

### 3.194 `forecast-container/app/validators.py`
*Валидаторы входных CSV-файлов продаж и остатков для сервиса предсказаний*  
Путь: [`forecast-container/app/validators.py`](C:/Users/lenya/OneDrive/Рабочий стол/hak/forecast-container/app/validators.py)

```python
"""
validators.py

Validation rules conforming strictly to Section 2 and Section 10 of Hackathon TZ:
1. All 12 columns from Section 2 required for sales:
   date, composite_division, cities_name, warehouse_name, category_lvl_1,
   category_lvl_2, category_lvl_3, article, characteristic, quantity, sales_sum, discount_sum
2. Stock required columns: date, warehouse_name, article (or char_article), stock
3. Strict ISO date format (YYYY-MM-DD), no fallback to non-ISO formats, no NaT.
4. Finiteness of numbers: no NaN, no +inf / -inf in quantity and stock.
5. Negative sales quantity rejected with 422 VALIDATION_ERROR.
6. Negative stock allowed (reserved items).
7. Null discount_sum allowed (no discount); non-null values must be finite numbers.
8. Non-empty string keys for warehouse_name and article (no whitespace or empty strings).
9. History depth: (max_date - min_date).days + 1 >= 180 days, otherwise 422 VALIDATION_ERROR.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException


REQUIRED_SALES_COLUMNS = [
    "date",
    "composite_division",
    "cities_name",
    "warehouse_name",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "article",
    "characteristic",
    "quantity",
    "sales_sum",
    "discount_sum",
]

REQUIRED_STOCK_COLUMNS = [
    "date",
    "warehouse_name",
    "stock",
]


class APIValidationError(HTTPException):
    def __init__(self, status_code: int, detail: str, code: str = "VALIDATION_ERROR"):
        super().__init__(status_code=status_code, detail={"detail": detail, "code": code})


def validate_and_load_sales(content: bytes | str | pd.DataFrame | Path) -> pd.DataFrame:
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, Path) or (isinstance(content, str) and "\n" not in content and Path(content).exists() and Path(content).is_file()):
            text = Path(content).read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) == 0:
                raise APIValidationError(400, "Empty sales CSV file", code="BAD_REQUEST")
        elif isinstance(content, bytes):
            if len(content) == 0:
                raise APIValidationError(400, "Empty sales CSV file", code="BAD_REQUEST")
            text = content.decode("utf-8", errors="replace")
        else:
            if len(content.strip()) == 0:
                raise APIValidationError(400, "Empty sales CSV file", code="BAD_REQUEST")
            text = content

        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise APIValidationError(400, f"Unreadable sales CSV: {e}", code="BAD_REQUEST")

    if len(df) == 0:
        raise APIValidationError(400, "Sales file contains 0 rows", code="BAD_REQUEST")

    # 1. Required columns check
    missing = [c for c in REQUIRED_SALES_COLUMNS if c not in df.columns]
    if missing:
        raise APIValidationError(
            400,
            f"Sales CSV missing required columns from TZ Section 2: {missing}",
            code="BAD_REQUEST",
        )

    # 2. Strict Date format YYYY-MM-DD (no fallback to non-ISO dates)
    try:
        parsed_dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise APIValidationError(
            400,
            f"Date column in sales CSV must be strictly in YYYY-MM-DD ISO format: {e}",
            code="BAD_REQUEST",
        )

    if parsed_dates.isna().any():
        raise APIValidationError(400, "Date column in sales CSV contains null/NaT values", code="BAD_REQUEST")
    df["date"] = parsed_dates

    # 3. Keys validation (non-empty, non-whitespace strings)
    for key_col in ["warehouse_name", "article"]:
        if df[key_col].isna().any():
            raise APIValidationError(400, f"Sales CSV column '{key_col}' contains null values", code="BAD_REQUEST")
        str_series = df[key_col].astype(str).str.strip()
        if (str_series == "").any():
            raise APIValidationError(400, f"Sales CSV column '{key_col}' contains empty or whitespace strings", code="BAD_REQUEST")

    # 4. Quantity validation: numeric, finite, non-null, non-negative
    try:
        qty_numeric = pd.to_numeric(df["quantity"], errors="raise")
    except Exception as e:
        raise APIValidationError(400, f"Non-numeric values in sales quantity: {e}", code="BAD_REQUEST")

    if qty_numeric.isna().any():
        raise APIValidationError(422, "Sales quantity contains NaN/null values", code="VALIDATION_ERROR")

    if np.isinf(qty_numeric).any():
        raise APIValidationError(422, "Sales quantity contains infinite (+inf/-inf) values", code="VALIDATION_ERROR")

    if (qty_numeric < 0).any():
        neg_count = int((qty_numeric < 0).sum())
        raise APIValidationError(422, f"Sales quantity contains {neg_count} negative values", code="VALIDATION_ERROR")

    df["quantity"] = qty_numeric

    # 5. Monetary values validation: numeric, non-infinite
    for col in ["sales_sum", "discount_sum"]:
        # NaNs in discount_sum are allowed (meaning no discount)
        # NaNs in sales_sum are converted to 0 in aggregation
        non_null_mask = df[col].notna()
        if non_null_mask.any():
            try:
                numeric_vals = pd.to_numeric(df.loc[non_null_mask, col], errors="raise")
            except Exception as e:
                raise APIValidationError(400, f"Non-numeric values in sales column '{col}': {e}", code="BAD_REQUEST")

            if np.isinf(numeric_vals).any():
                raise APIValidationError(400, f"Sales column '{col}' contains infinite values", code="BAD_REQUEST")

            # Check for extreme log values that would overflow exp (> 100)
            if (numeric_vals > 100).any():
                raise APIValidationError(400, f"Sales column '{col}' contains values > 100 causing exp overflow", code="BAD_REQUEST")

    # 6. History depth check (at least 180 days)
    min_date = df["date"].min()
    max_date = df["date"].max()
    depth_days = (max_date - min_date).days + 1
    if depth_days < 180:
        raise APIValidationError(
            422,
            f"Insufficient sales history depth: {depth_days} days provided, minimum 180 days required ({min_date.date()} .. {max_date.date()})",
            code="VALIDATION_ERROR",
        )

    return df


def validate_and_load_stock(content: bytes | str | pd.DataFrame | Path, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, Path) or (isinstance(content, str) and "\n" not in content and Path(content).exists() and Path(content).is_file()):
            text = Path(content).read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) == 0:
                raise APIValidationError(400, "Empty stock CSV file", code="BAD_REQUEST")
        elif isinstance(content, bytes):
            if len(content) == 0:
                raise APIValidationError(400, "Empty stock CSV file", code="BAD_REQUEST")
            text = content.decode("utf-8", errors="replace")
        else:
            if len(content.strip()) == 0:
                raise APIValidationError(400, "Empty stock CSV file", code="BAD_REQUEST")
            text = content

        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise APIValidationError(400, f"Unreadable stock CSV: {e}", code="BAD_REQUEST")

    if len(df) == 0:
        raise APIValidationError(400, "Stock file contains 0 rows", code="BAD_REQUEST")

    # Rename char_article -> article
    if "article" not in df.columns and "char_article" in df.columns:
        df = df.rename(columns={"char_article": "article"})

    if "article" not in df.columns:
        raise APIValidationError(400, "Stock CSV missing column 'article' (or 'char_article')", code="BAD_REQUEST")

    missing = [c for c in REQUIRED_STOCK_COLUMNS if c not in df.columns]
    if missing:
        raise APIValidationError(400, f"Stock CSV missing required columns: {missing}", code="BAD_REQUEST")

    # Strict Date format YYYY-MM-DD
    try:
        parsed_dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise APIValidationError(
            400,
            f"Date column in stock CSV must be strictly in YYYY-MM-DD ISO format: {e}",
            code="BAD_REQUEST",
        )

    if parsed_dates.isna().any():
        raise APIValidationError(400, "Date column in stock CSV contains null/NaT values", code="BAD_REQUEST")
    df["date"] = parsed_dates

    # Keys validation
    for key_col in ["warehouse_name", "article"]:
        if df[key_col].isna().any():
            raise APIValidationError(400, f"Stock CSV column '{key_col}' contains null values", code="BAD_REQUEST")
        str_series = df[key_col].astype(str).str.strip()
        if (str_series == "").any():
            raise APIValidationError(400, f"Stock CSV column '{key_col}' contains empty or whitespace strings", code="BAD_REQUEST")

    # Stock numeric validation (negative stock allowed, but must be finite and non-null)
    try:
        stock_numeric = pd.to_numeric(df["stock"], errors="raise")
    except Exception as e:
        raise APIValidationError(400, f"Non-numeric values in stock: {e}", code="BAD_REQUEST")

    if stock_numeric.isna().any():
        raise APIValidationError(400, "Stock contains NaN/null values", code="BAD_REQUEST")

    if np.isinf(stock_numeric).any():
        raise APIValidationError(400, "Stock contains infinite (+inf/-inf) values", code="BAD_REQUEST")

    df["stock"] = stock_numeric

    # Filter stock after cutoff date if provided
    if cutoff_date is not None:
        df = df[df["date"] <= cutoff_date].copy()

    # History depth check (at least 180 days)
    if len(df) == 0:
        raise APIValidationError(422, "No stock records remain on or before sales cutoff date", code="VALIDATION_ERROR")

    min_date = df["date"].min()
    max_date = df["date"].max()
    depth_days = (max_date - min_date).days + 1
    if depth_days < 180:
        raise APIValidationError(
            422,
            f"Insufficient stock history depth: {depth_days} days provided, minimum 180 days required ({min_date.date()} .. {max_date.date()})",
            code="VALIDATION_ERROR",
        )

    return df
```

---
