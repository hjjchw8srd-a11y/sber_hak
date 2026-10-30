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
