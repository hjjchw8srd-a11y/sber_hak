from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_weekly_feature_experiments import (
    GROUP_COLUMNS,
    WeeklyMatrices,
)


class WeeklyFeatureLeakageTests(unittest.TestCase):
    @staticmethod
    def make_data() -> pd.DataFrame:
        weeks = pd.date_range("2025-01-06", periods=23, freq="W-MON")
        rows = []
        for week_idx, week in enumerate(weeks):
            for warehouse_idx in range(2):
                for article_idx in range(2):
                    quantity = float((week_idx + warehouse_idx + article_idx) % 5)
                    rows.append(
                        {
                            "week": week,
                            "warehouse_name": f"warehouse_{warehouse_idx}",
                            "article": f"article_{article_idx}",
                            "category_lvl_1": "category_1",
                            "category_lvl_2": f"category_2_{article_idx}",
                            "category_lvl_3": "category_3",
                            "quantity": quantity,
                            "sales_sum": quantity * 100.0,
                            "discount_sum": quantity * 10.0,
                            "stock_week_end": 5.0,
                            "stock_mean": 6.0,
                            "stock_nonpositive_rate": 0.0,
                            "stock_observed_days": 7.0,
                        }
                    )
        return pd.DataFrame(rows)

    def test_future_targets_do_not_change_context_features(self) -> None:
        data = self.make_data()
        origin = 14
        original_features, original_target, _ = WeeklyMatrices(data).make_rows(
            np.array([origin]), None
        )

        changed = data.copy()
        future_weeks = sorted(changed["week"].unique())[origin : origin + 9]
        changed.loc[changed["week"].isin(future_weeks), "quantity"] += 10_000.0
        changed_features, changed_target, _ = WeeklyMatrices(changed).make_rows(
            np.array([origin]), None
        )

        context_columns = (
            GROUP_COLUMNS["category_context"] + GROUP_COLUMNS["network_context"]
        )
        pd.testing.assert_frame_equal(
            original_features[context_columns],
            changed_features[context_columns],
        )
        self.assertFalse(np.array_equal(original_target, changed_target))


if __name__ == "__main__":
    unittest.main()
