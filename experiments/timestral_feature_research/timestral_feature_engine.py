"""
timestral_feature_engine.py

Vectorized, leakage-free feature computation engine for Native Timestral Covariates.
Computes dense temporal tensors for all 8 feature groups (A to H) relative to any forecast origin:

- Group A: Demand Dynamics (lags, rolling means, slope, nonzero rate, zero streak, growth)
- Group B: Stock State (stock_end, stock_mean, nonpositive_rate, observed_days, missing, weeks_of_supply, stockout_flag)
- Group C: Network Demand Context (article network sales mean 13 log, growth 4 vs 13, warehouse share)
- Group D: Category Context (category sales mean 13 log, growth 4 vs 13, article share in category)
- Group E: Money & Discounts (sales_sum_log, discount_sum_log, discount_share, paid_per_unit)
- Group F: Calendar & Events (month, week_sin, week_cos, holiday_share, is_holiday_week, is_back_to_school) [KNOWN COVARIATE: extends into horizon]
- Group G: Year Seasonality (lag 52, year window mean)
- Group H: Categorical Encodings (target-encoded category_lvl_1, category_lvl_2, warehouse_name)

Strict time safety:
- past_covariates: only historical values for t in [origin - context_len, origin - 1].
- known_covariates: extends into the horizon [origin, origin + horizon_len - 1] ONLY for calendar/events.
- Zero future target leakage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch

from run_weekly_feature_experiments import WeeklyMatrices
from holiday_calendar import STATUTORY_HOLIDAY_DATES, TRANSFERRED_NONWORKING_DAYS, WORKING_SATURDAYS

KEYS = ["week", "warehouse_name", "article"]
CONTEXT = 52
HORIZON = 9


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def get_lagged_slice(matrix: np.ndarray, origin: int, context_len: int, lag: int) -> np.ndarray:
    """Returns (N_series, context_len) ending at origin - lag, left-padded with zeros if needed."""
    end = origin - lag
    start = end - context_len
    N = matrix.shape[1]
    if start >= 0:
        return matrix[start:end, :].T
    elif end > 0:
        valid = matrix[0:end, :].T  # (N, end)
        pad_len = context_len - end
        pad = np.zeros((N, pad_len), dtype=np.float32)
        return np.concatenate([pad, valid], axis=1)
    else:
        return np.zeros((N, context_len), dtype=np.float32)


class TimestralFeatureEngine:
    def __init__(self, data_path: Path):
        data = pd.read_csv(data_path, parse_dates=["week"])
        self.matrices = WeeklyMatrices(data)
        self.weeks = self.matrices.weeks
        self.n_weeks = len(self.weeks)
        self.n_series = self.matrices.n_pairs
        self.pairs = self.matrices.pairs.copy()

        # Precompute calendar arrays across all 134+ weeks
        self._precompute_calendar()
        # Precompute category and network aggregations across time
        self._precompute_context_aggregates()
        self._group_cache = {}

    def _precompute_calendar(self):
        """Precomputes calendar indicators for all weeks in the dataset."""
        week_nums = self.weeks.isocalendar().week.to_numpy(dtype=np.float32)
        months = self.weeks.month.to_numpy(dtype=np.float32)
        quarters = self.weeks.quarter.to_numpy(dtype=np.float32)

        self.cal_month = months / 12.0
        self.cal_quarter = quarters / 4.0
        self.cal_week_of_year = week_nums / 52.0
        self.cal_week_sin = np.sin(2.0 * np.pi * week_nums / 52.18).astype(np.float32)
        self.cal_week_cos = np.cos(2.0 * np.pi * week_nums / 52.18).astype(np.float32)
        self.cal_time_idx = (np.arange(len(self.weeks), dtype=np.float32) / 134.0).astype(np.float32)

        # Holiday & event calendar
        holiday_shares = []
        is_holiday = []
        is_working_sat = []
        is_school = []

        for w_start in self.weeks:
            w_days = [w_start + pd.Timedelta(days=d) for d in range(7)]
            n_holidays = sum(
                1 for day in w_days
                if (day.month, day.day) in STATUTORY_HOLIDAY_DATES
                or day.date() in TRANSFERRED_NONWORKING_DAYS
            )
            n_work_sat = sum(1 for day in w_days if day.date() in WORKING_SATURDAYS)
            
            # School rush: weeks 34, 35, 36 (late August - early September)
            is_school_week = 1.0 if w_start.week in [34, 35, 36] else 0.0

            holiday_shares.append(n_holidays / 7.0)
            is_holiday.append(1.0 if n_holidays > 0 else 0.0)
            is_working_sat.append(1.0 if n_work_sat > 0 else 0.0)
            is_school.append(is_school_week)

        self.cal_holiday_share = np.array(holiday_shares, dtype=np.float32)
        self.cal_is_holiday = np.array(is_holiday, dtype=np.float32)
        self.cal_is_working_sat = np.array(is_working_sat, dtype=np.float32)
        self.cal_is_school = np.array(is_school, dtype=np.float32)

    def _precompute_context_aggregates(self):
        """Precomputes category and network time series."""
        # Article total network sales over time: (T, N_articles)
        # Quantity is (T, N_series)
        T, N = self.matrices.quantity.shape
        n_art = self.matrices.n_articles
        n_wcat = self.matrices.n_warehouse_categories

        self.network_quantity = np.zeros((T, n_art), dtype=np.float32)
        for t in range(T):
            self.network_quantity[t] = np.bincount(
                self.matrices.article_codes,
                weights=self.matrices.quantity[t],
                minlength=n_art,
            )

        # Warehouse-category sales over time: (T, N_wcat)
        self.cat_quantity = np.zeros((T, n_wcat), dtype=np.float32)
        for t in range(T):
            self.cat_quantity[t] = np.bincount(
                self.matrices.warehouse_category_codes,
                weights=self.matrices.quantity[t],
                minlength=n_wcat,
            )

    def get_group_features(
        self,
        group: str,
        origin: int,
        context_len: int = CONTEXT,
        horizon_len: int = HORIZON,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extracts features for a specific group at a given origin.
        Returns:
            past_feats: (N_series, C_past, context_len)
            known_feats: (N_series, C_known, context_len + horizon_len)
            past_names: List of past feature names
            known_names: List of known feature names
        """
        cache_key = (group, origin, context_len, horizon_len)
        if cache_key in self._group_cache:
            return self._group_cache[cache_key]

        ctx_start = origin - context_len
        ctx_end = origin
        total_len = context_len + horizon_len
        N = self.n_series

        past_list: List[np.ndarray] = []
        known_list: List[np.ndarray] = []
        past_names: List[str] = []
        known_names: List[str] = []

        # ----------------------------------------------------------------------
        # Group A: Demand Dynamics (Past Covariates across context window)
        # ----------------------------------------------------------------------
        if group == "A_dynamics":
            lag1 = get_lagged_slice(self.matrices.quantity, origin, context_len, lag=1)
            lag2 = get_lagged_slice(self.matrices.quantity, origin, context_len, lag=2)
            lag4 = get_lagged_slice(self.matrices.quantity, origin, context_len, lag=4)
            lag13 = get_lagged_slice(self.matrices.quantity, origin, context_len, lag=13)

            # Rolling stats across the context window
            Q = self.matrices.quantity
            mean4_seq = np.zeros((N, context_len), dtype=np.float32)
            mean13_seq = np.zeros((N, context_len), dtype=np.float32)
            nz13_seq = np.zeros((N, context_len), dtype=np.float32)
            std13_seq = np.zeros((N, context_len), dtype=np.float32)

            for step_idx, t in enumerate(range(ctx_start, ctx_end)):
                s4 = max(0, t - 4)
                s13 = max(0, t - 13)
                if t > s4:
                    mean4_seq[:, step_idx] = Q[s4:t, :].mean(axis=0)
                if t > s13:
                    w13 = Q[s13:t, :]
                    mean13_seq[:, step_idx] = w13.mean(axis=0)
                    nz13_seq[:, step_idx] = (w13 > 0).mean(axis=0)
                    std13_seq[:, step_idx] = w13.std(axis=0)

            growth_4_13 = np.divide(mean4_seq, mean13_seq + 1e-4, out=np.ones_like(mean4_seq), where=mean13_seq > 0)
            cv13 = np.divide(std13_seq, mean13_seq + 1e-4, out=np.zeros_like(std13_seq), where=mean13_seq > 0)

            past_list.extend([
                signed_log1p(lag1),
                signed_log1p(lag2),
                signed_log1p(lag4),
                signed_log1p(lag13),
                signed_log1p(mean4_seq),
                signed_log1p(mean13_seq),
                nz13_seq,
                np.clip(growth_4_13, 0.0, 5.0),
                np.clip(cv13, 0.0, 5.0),
            ])
            past_names.extend([
                "quantity_lag_1", "quantity_lag_2", "quantity_lag_4", "quantity_lag_13",
                "quantity_mean_4", "quantity_mean_13", "quantity_nonzero_rate_13",
                "growth_4_vs_13", "quantity_cv_13"
            ])

        # ----------------------------------------------------------------------
        # Group B: Stock State (Past Covariates across context window)
        # ----------------------------------------------------------------------
        elif group == "B_stock":
            st_end = get_lagged_slice(self.matrices.stock_end, origin, context_len, lag=0)
            st_mean = get_lagged_slice(self.matrices.stock_mean, origin, context_len, lag=0)
            st_snr = get_lagged_slice(self.matrices.stock_nonpositive_rate, origin, context_len, lag=0)
            st_obs = get_lagged_slice(self.matrices.stock_observed_days, origin, context_len, lag=0)

            st_missing = np.isnan(st_end).astype(np.float32)
            st_end_clean = np.nan_to_num(st_end, nan=0.0)
            st_mean_clean = np.nan_to_num(st_mean, nan=0.0)
            st_snr_clean = np.nan_to_num(st_snr, nan=1.0)
            st_out = ((st_end_clean <= 0) & (st_obs > 0)).astype(np.float32)

            # Weeks of supply using recent rolling mean
            Q = self.matrices.quantity
            wos_seq = np.zeros((N, context_len), dtype=np.float32)
            for step_idx, t in enumerate(range(ctx_start, ctx_end)):
                s13 = max(0, t - 13)
                m13 = Q[s13:t, :].mean(axis=0) if t > s13 else np.zeros(N, dtype=np.float32)
                wos_seq[:, step_idx] = np.clip(st_end_clean[:, step_idx] / (m13 + 1e-4), 0.0, 20.0)

            past_list.extend([
                signed_log1p(st_end_clean),
                signed_log1p(st_mean_clean),
                st_snr_clean,
                st_missing,
                st_out,
                wos_seq,
            ])
            past_names.extend([
                "stock_end", "stock_mean", "stock_nonpositive_rate",
                "stock_missing", "stockout_flag", "weeks_of_supply"
            ])

        # ----------------------------------------------------------------------
        # Group C: Network Demand Context (Past Covariates across context window)
        # ----------------------------------------------------------------------
        elif group == "C_network":
            # Network sales of article over past 13w and 4w
            net_sales_log = np.zeros((N, context_len), dtype=np.float32)
            net_growth = np.zeros((N, context_len), dtype=np.float32)
            wh_share = np.zeros((N, context_len), dtype=np.float32)

            art_codes = self.matrices.article_codes
            Q = self.matrices.quantity
            NQ = self.network_quantity

            for step_idx, t in enumerate(range(ctx_start, ctx_end)):
                s13 = max(0, t - 13)
                s4 = max(0, t - 4)
                net13 = NQ[s13:t, :].mean(axis=0) if t > s13 else np.zeros(NQ.shape[1], dtype=np.float32)
                net4 = NQ[s4:t, :].mean(axis=0) if t > s4 else np.zeros(NQ.shape[1], dtype=np.float32)
                own13 = Q[s13:t, :].mean(axis=0) if t > s13 else np.zeros(Q.shape[1], dtype=np.float32)

                net_art = net13[art_codes]
                net_art4 = net4[art_codes]

                net_sales_log[:, step_idx] = signed_log1p(net_art)
                net_growth[:, step_idx] = np.clip(np.divide(net_art4, net_art + 1e-4, out=np.ones_like(net_art), where=net_art > 0), 0.0, 5.0)
                wh_share[:, step_idx] = np.clip(np.divide(own13, net_art + 1e-4, out=np.zeros_like(own13), where=net_art > 0), 0.0, 1.0)

            past_list.extend([net_sales_log, net_growth, wh_share])
            past_names.extend(["article_network_sales_mean_13_log", "article_network_growth_4_vs_13", "warehouse_share_article_13"])

        # ----------------------------------------------------------------------
        # Group D: Category Context (Past Covariates across context window)
        # ----------------------------------------------------------------------
        elif group == "D_category":
            cat_sales_log = np.zeros((N, context_len), dtype=np.float32)
            cat_growth = np.zeros((N, context_len), dtype=np.float32)
            art_cat_share = np.zeros((N, context_len), dtype=np.float32)

            wcat_codes = self.matrices.warehouse_category_codes
            Q = self.matrices.quantity
            CQ = self.cat_quantity

            for step_idx, t in enumerate(range(ctx_start, ctx_end)):
                s13 = max(0, t - 13)
                s4 = max(0, t - 4)
                cat13 = CQ[s13:t, :].mean(axis=0) if t > s13 else np.zeros(CQ.shape[1], dtype=np.float32)
                cat4 = CQ[s4:t, :].mean(axis=0) if t > s4 else np.zeros(CQ.shape[1], dtype=np.float32)
                own13 = Q[s13:t, :].mean(axis=0) if t > s13 else np.zeros(Q.shape[1], dtype=np.float32)

                cat_w = cat13[wcat_codes]
                cat_w4 = cat4[wcat_codes]

                cat_sales_log[:, step_idx] = signed_log1p(cat_w)
                cat_growth[:, step_idx] = np.clip(np.divide(cat_w4, cat_w + 1e-4, out=np.ones_like(cat_w), where=cat_w > 0), 0.0, 5.0)
                art_cat_share[:, step_idx] = np.clip(np.divide(own13, cat_w + 1e-4, out=np.zeros_like(own13), where=cat_w > 0), 0.0, 1.0)

            past_list.extend([cat_sales_log, cat_growth, art_cat_share])
            past_names.extend(["category_sales_mean_13_log", "category_growth_4_vs_13", "article_share_category_13"])

        # ----------------------------------------------------------------------
        # Group E: Money & Discounts (Past Covariates across context window)
        # ----------------------------------------------------------------------
        elif group == "E_money":
            sales = np.nan_to_num(get_lagged_slice(self.matrices.sales_sum, origin, context_len, lag=0), nan=0.0)
            disc = np.nan_to_num(get_lagged_slice(self.matrices.discount_sum, origin, context_len, lag=0), nan=0.0)
            qty = np.maximum(get_lagged_slice(self.matrices.quantity, origin, context_len, lag=0), 0.0)

            full_val = sales + disc
            disc_share = np.divide(disc, full_val, out=np.zeros_like(disc), where=full_val > 0)
            paid_per_u = np.divide(sales, qty, out=np.zeros_like(sales), where=qty > 0)

            past_list.extend([
                signed_log1p(sales),
                signed_log1p(disc),
                disc_share,
                signed_log1p(paid_per_u),
            ])
            past_names.extend(["sales_sum_log", "discount_sum_log", "discount_share", "paid_value_per_unit"])

        # ----------------------------------------------------------------------
        # Group F: Calendar & Events (KNOWN COVARIATES: context + horizon)
        # ----------------------------------------------------------------------
        elif group == "F_calendar":
            # Slice across total window [ctx_start, ctx_start + total_len]
            # Broadcast to (N_series, total_len)
            c_month = np.tile(self.cal_month[ctx_start : ctx_start + total_len], (N, 1))
            c_sin = np.tile(self.cal_week_sin[ctx_start : ctx_start + total_len], (N, 1))
            c_cos = np.tile(self.cal_week_cos[ctx_start : ctx_start + total_len], (N, 1))
            c_hol_share = np.tile(self.cal_holiday_share[ctx_start : ctx_start + total_len], (N, 1))
            c_is_hol = np.tile(self.cal_is_holiday[ctx_start : ctx_start + total_len], (N, 1))
            c_is_school = np.tile(self.cal_is_school[ctx_start : ctx_start + total_len], (N, 1))

            known_list.extend([c_month, c_sin, c_cos, c_hol_share, c_is_hol, c_is_school])
            known_names.extend(["cal_month", "cal_week_sin", "cal_week_cos", "cal_holiday_share", "cal_is_holiday", "cal_is_school"])

        # ----------------------------------------------------------------------
        # Group G: Year Seasonality (Past Covariates across context window)
        # ----------------------------------------------------------------------
        elif group == "G_year":
            lag52 = get_lagged_slice(self.matrices.quantity, origin, context_len, lag=52)
            past_list.append(signed_log1p(lag52))
            past_names.append("quantity_lag_52")

        # ----------------------------------------------------------------------
        # Group H: Categorical Encodings (Past Covariates: smoothed target encoding)
        # ----------------------------------------------------------------------
        elif group == "H_categories":
            # Per-series pre-cutoff target mean for category_lvl_1, category_lvl_2, warehouse_name
            # Constant along time context
            Q_ctx = self.matrices.quantity[ctx_start : ctx_end, :]
            mean_series = Q_ctx.mean(axis=0)  # (N,)

            # Mean by category_lvl_1
            cat1_codes = pd.factorize(self.matrices.static["category_lvl_1"], sort=True)[0]
            cat1_means = np.bincount(cat1_codes, weights=mean_series, minlength=int(cat1_codes.max()+1))
            cat1_counts = np.bincount(cat1_codes, minlength=int(cat1_codes.max()+1))
            cat1_te = (cat1_means / np.maximum(cat1_counts, 1))[cat1_codes]

            # Mean by warehouse
            wh_codes = self.matrices.warehouse_codes
            wh_means = np.bincount(wh_codes, weights=mean_series, minlength=int(wh_codes.max()+1))
            wh_counts = np.bincount(wh_codes, minlength=int(wh_codes.max()+1))
            wh_te = (wh_means / np.maximum(wh_counts, 1))[wh_codes]

            te_cat1 = np.tile(signed_log1p(cat1_te)[:, None], (1, context_len))
            te_wh = np.tile(signed_log1p(wh_te)[:, None], (1, context_len))

            past_list.extend([te_cat1, te_wh])
            past_names.extend(["target_enc_category_lvl_1", "target_enc_warehouse"])

        # Stack into numpy arrays
        if len(past_list) > 0:
            past_arr = np.stack(past_list, axis=1)  # (N, C_past, context_len)
        else:
            past_arr = np.empty((N, 0, context_len), dtype=np.float32)

        if len(known_list) > 0:
            known_arr = np.stack(known_list, axis=1)  # (N, C_known, total_len)
        else:
            known_arr = np.empty((N, 0, total_len), dtype=np.float32)

        result = (past_arr, known_arr, past_names, known_names)
        self._group_cache[cache_key] = result
        return result

    def assemble_batch_tensors(
        self,
        groups: List[str],
        origin: int,
        indices: np.ndarray,
        context_len: int = CONTEXT,
        horizon_len: int = HORIZON,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, List[str], List[str]]:
        """
        Assembles target, y_true, past_covs, and known_covs for a given subset of series indices.
        Returns:
            target_ctx: (B, 1, context_len)
            y_true: (B, horizon_len)
            past_covs: (B, C_past, context_len) or None
            known_covs: (B, C_known, context_len + horizon_len) or None
            all_past_names: List[str]
            all_known_names: List[str]
        """
        ctx_start = origin - context_len
        ctx_end = origin
        horiz_end = origin + horizon_len

        # Target context and true future
        target_ctx = self.matrices.quantity[ctx_start : ctx_end, indices].T[:, None, :]  # (B, 1, context_len)
        y_true = self.matrices.quantity[ctx_end : horiz_end, indices].T                  # (B, horizon_len)

        all_past: List[np.ndarray] = []
        all_known: List[np.ndarray] = []
        all_past_names: List[str] = []
        all_known_names: List[str] = []

        for g in groups:
            p_arr, k_arr, p_names, k_names = self.get_group_features(g, origin, context_len, horizon_len)
            if p_arr.shape[1] > 0:
                all_past.append(p_arr[indices])
                all_past_names.extend(p_names)
            if k_arr.shape[1] > 0:
                all_known.append(k_arr[indices])
                all_known_names.extend(k_names)

        past_covs = np.concatenate(all_past, axis=1) if len(all_past) > 0 else None
        known_covs = np.concatenate(all_known, axis=1) if len(all_known) > 0 else None

        return target_ctx, y_true, past_covs, known_covs, all_past_names, all_known_names
