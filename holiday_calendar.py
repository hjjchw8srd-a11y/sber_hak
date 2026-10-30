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
