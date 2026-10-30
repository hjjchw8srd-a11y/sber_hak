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
