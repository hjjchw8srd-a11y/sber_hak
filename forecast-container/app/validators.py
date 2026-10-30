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


def validate_and_load_sales(content: bytes | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
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


def validate_and_load_stock(content: bytes | str | pd.DataFrame, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if isinstance(content, pd.DataFrame):
        df = content.copy()
    else:
        if isinstance(content, bytes):
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
