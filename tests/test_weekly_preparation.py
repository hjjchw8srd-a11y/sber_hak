from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fast_score import compute_all_metrics_fast
from prepare_weekly_data import prepare_sales, prepare_stock


class WeeklyPreparationTests(unittest.TestCase):
    def test_sales_inverse_transform_and_weekly_sum(self) -> None:
        sales = pd.DataFrame(
            {
                "date": ["2026-07-20", "2026-07-21"],
                "warehouse_name": ["w1", "w1"],
                "article": ["a1", "a1"],
                "quantity": [2.0, 3.0],
                "sales_sum": np.log([10.0, 20.0]),
                "discount_sum": [np.nan, np.log(5.0)],
                "category_lvl_1": ["c1", "c1"],
                "category_lvl_2": ["c2", "c2"],
                "category_lvl_3": [np.nan, np.nan],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sales.csv"
            sales.to_csv(path, index=False)
            weekly, meta, _ = prepare_sales(path, pd.Timestamp("2026-07-26"))

        row = weekly.iloc[0]
        self.assertEqual(row["quantity"], 5.0)
        self.assertAlmostEqual(row["sales_sum"], 30.0)
        self.assertAlmostEqual(row["discount_sum"], 5.0)
        self.assertEqual(row["discount_rows"], 1)
        self.assertEqual(meta.loc["a1", "category_lvl_3"], "unknown")

    def test_stock_duplicates_sum_within_day_not_across_days(self) -> None:
        stock = pd.DataFrame(
            {
                "date": ["2026-07-20", "2026-07-20", "2026-07-21"],
                "warehouse_name": ["w1", "w1", "w1"],
                "article": ["a1", "a1", "a1"],
                "stock": [2.0, 3.0, 7.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stock.csv"
            stock.to_csv(path, index=False)
            weekly, summary = prepare_stock(path, pd.Timestamp("2026-07-26"))

        row = weekly.iloc[0]
        self.assertEqual(summary["stock_duplicate_rows_collapsed"], 1)
        self.assertEqual(row["stock_week_start"], 5.0)
        self.assertEqual(row["stock_week_end"], 7.0)
        self.assertEqual(row["stock_min"], 5.0)
        self.assertEqual(row["stock_max"], 7.0)
        self.assertEqual(row["stock_mean"], 6.0)
        self.assertEqual(row["stock_observed_days"], 2)

    def test_smape_keeps_zero_zero_point(self) -> None:
        truth = pd.DataFrame(
            {
                "week": ["2026-07-20", "2026-07-27"],
                "warehouse_name": ["w1", "w1"],
                "article": ["a1", "a1"],
                "quantity": [0.0, 1.0],
            }
        )
        prediction = truth.copy()
        prediction["quantity"] = 0.0
        metrics = compute_all_metrics_fast(truth, prediction)
        self.assertAlmostEqual(metrics["smape"], 100.0)


if __name__ == "__main__":
    unittest.main()
