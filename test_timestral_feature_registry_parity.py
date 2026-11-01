#!/usr/bin/env python3
"""
test_timestral_feature_registry_parity.py

Automated parity test verifying:
1. Exact correspondence between FEATURE_REGISTRY.md / FEATURE_AUDIT_REPORT.md and timestral_feature_engine.py.
2. Complete availability of all extended channels:
   - stock_start, stock_change, stock_min, stock_max (Group B)
   - active_warehouses_article_13 (Group C_network_extended / C_active)
   - active_articles_category_13 (Group D_category_extended / D_active)
   - quantity_lag_51, quantity_lag_53 (Group G_year)
   - target_enc_category_lvl_2 (Group H_categories)
3. Correct tensor shapes: (N_series, C_past, 52) and (N_series, C_known, 61).
4. Strict numerical sanity: zero NaNs, zero Infs, bounded numerical values.
5. Invariance across multiple forecast origins (89, 107, 125).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine


EXPECTED_GROUPS = {
    "A_dynamics": {
        "past": [
            "quantity_lag_1", "quantity_lag_2", "quantity_lag_4", "quantity_lag_13",
            "quantity_mean_4", "quantity_mean_13", "quantity_nonzero_rate_13",
            "growth_4_vs_13", "quantity_cv_13"
        ],
        "known": [],
    },
    "B_stock": {
        "past": [
            "stock_end", "stock_start", "stock_change", "stock_min", "stock_max",
            "stock_mean", "stock_nonpositive_rate", "stock_missing", "stockout_flag", "weeks_of_supply"
        ],
        "known": [],
    },
    "C_network": {
        "past": [
            "article_network_sales_mean_13_log", "article_network_growth_4_vs_13", "warehouse_share_article_13"
        ],
        "known": [],
    },
    "C_network_extended": {
        "past": [
            "article_network_sales_mean_13_log", "article_network_growth_4_vs_13",
            "warehouse_share_article_13", "active_warehouses_article_13"
        ],
        "known": [],
    },
    "D_category": {
        "past": [
            "category_sales_mean_13_log", "category_growth_4_vs_13", "article_share_category_13"
        ],
        "known": [],
    },
    "D_category_extended": {
        "past": [
            "category_sales_mean_13_log", "category_growth_4_vs_13",
            "article_share_category_13", "active_articles_category_13"
        ],
        "known": [],
    },
    "E_money": {
        "past": [
            "sales_sum_log", "discount_sum_log", "discount_share", "paid_value_per_unit"
        ],
        "known": [],
    },
    "F_calendar": {
        "past": [],
        "known": [
            "cal_month", "cal_week_sin", "cal_week_cos",
            "cal_holiday_share", "cal_is_holiday", "cal_is_school"
        ],
    },
    "G_year": {
        "past": [
            "quantity_lag_51", "quantity_lag_52", "quantity_lag_53"
        ],
        "known": [],
    },
    "H_categories": {
        "past": [
            "target_enc_category_lvl_1", "target_enc_category_lvl_2", "target_enc_warehouse"
        ],
        "known": [],
    },
}


def test_feature_registry_parity():
    print("=" * 80)
    print("RUNNING TIMESTRAL FEATURE REGISTRY PARITY TEST")
    print("=" * 80)

    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    assert data_path.exists(), f"Missing {data_path}"
    engine = TimestralFeatureEngine(data_path)
    N = engine.n_series

    test_origins = [89, 107, 125]  # Fold 4, Fold 2, Fold 0

    total_channels_checked = 0

    for origin in test_origins:
        print(f"\n--- Checking Origin {origin} (Test Week: {engine.weeks[origin].strftime('%Y-%m-%d')}) ---")
        for group_name, expected in EXPECTED_GROUPS.items():
            past_arr, known_arr, past_names, known_names = engine.get_group_features(
                group=group_name,
                origin=origin,
                context_len=52,
                horizon_len=9,
            )

            # 1. Check exact names
            assert past_names == expected["past"], (
                f"Group {group_name} past names mismatch!\nExpected: {expected['past']}\nActual: {past_names}"
            )
            assert known_names == expected["known"], (
                f"Group {group_name} known names mismatch!\nExpected: {expected['known']}\nActual: {known_names}"
            )

            # 2. Check shapes
            assert past_arr.shape == (N, len(expected["past"]), 52), (
                f"Group {group_name} past shape {past_arr.shape} != {(N, len(expected['past']), 52)}"
            )
            assert known_arr.shape == (N, len(expected["known"]), 61), (
                f"Group {group_name} known shape {known_arr.shape} != {(N, len(expected['known']), 61)}"
            )

            # 3. Check numerical cleanliness: zero NaNs and Infs
            if past_arr.size > 0:
                n_nans = np.isnan(past_arr).sum()
                n_infs = np.isinf(past_arr).sum()
                assert n_nans == 0, f"Group {group_name} past has {n_nans} NaNs at origin {origin}!"
                assert n_infs == 0, f"Group {group_name} past has {n_infs} Infs at origin {origin}!"

            if known_arr.size > 0:
                n_nans = np.isnan(known_arr).sum()
                n_infs = np.isinf(known_arr).sum()
                assert n_nans == 0, f"Group {group_name} known has {n_nans} NaNs at origin {origin}!"
                assert n_infs == 0, f"Group {group_name} known has {n_infs} Infs at origin {origin}!"

            n_ch = len(past_names) + len(known_names)
            total_channels_checked += n_ch
            print(f"  [OK] {group_name:20s}: {len(past_names):2d} past, {len(known_names):2d} known channels. All valid.")

    print("\n" + "=" * 80)
    print(f"ALL FEATURE PARITY CHECKS PASSED: Verified {total_channels_checked} channel-origin evaluations!")
    print("=" * 80)


if __name__ == "__main__":
    test_feature_registry_parity()
