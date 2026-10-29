import unittest

import numpy as np
import pandas as pd

from train_tft_weekly import (
    BASE_HIST_EXOG,
    DEMAND_DYNAMICS_EXOG,
    MONEY_EXOG,
    STOCK_DERIVED_EXOG,
    STOCK_FULL_EXOG,
    prepare_frames,
)


ALL_HIST_EXOG = list(
    dict.fromkeys(
        [
            *BASE_HIST_EXOG,
            *STOCK_FULL_EXOG,
            *STOCK_DERIVED_EXOG,
            *MONEY_EXOG,
            *DEMAND_DYNAMICS_EXOG,
        ]
    )
)


class TftPreparationTest(unittest.TestCase):
    @staticmethod
    def sample_data() -> pd.DataFrame:
        rows = []
        weeks = pd.date_range("2025-01-06", periods=40, freq="W-MON")
        for week_index, week in enumerate(weeks):
            for warehouse_index, warehouse in enumerate(["A", "B"]):
                rows.append(
                    {
                        "week": week,
                        "warehouse_name": warehouse,
                        "article": "item_1",
                        "category_lvl_1": "food",
                        "category_lvl_2": "snacks",
                        "category_lvl_3": "chips",
                        "quantity": float((week_index + warehouse_index) % 4),
                    }
                )
        return pd.DataFrame(rows)

    def test_future_target_does_not_change_features_at_origin(self) -> None:
        data = self.sample_data()
        origin = pd.Timestamp("2025-07-07")
        original, _, _ = prepare_frames(data)
        changed = data.copy()
        changed.loc[changed["week"] >= origin, "quantity"] = 10_000.0
        changed_frame, _, _ = prepare_frames(changed)

        # Historic exogenous variables передаются TFT только до недели,
        # предшествующей origin. Строки самого прогнозного горизонта содержат
        # только FUTR_EXOG и в encoder не попадают.
        last_history_week = origin - pd.Timedelta(weeks=1)
        original_origin = (
            original[original["ds"] == last_history_week]
            .sort_values("unique_id")
            .reset_index(drop=True)
        )
        changed_origin = (
            changed_frame[changed_frame["ds"] == last_history_week]
            .sort_values("unique_id")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            original_origin[ALL_HIST_EXOG].to_numpy(),
            changed_origin[ALL_HIST_EXOG].to_numpy(),
        )

    def test_prepared_frames_have_expected_shape(self) -> None:
        data = self.sample_data()
        temporal, static, static_columns = prepare_frames(data)
        self.assertEqual(len(temporal), len(data))
        self.assertEqual(temporal["unique_id"].nunique(), 2)
        self.assertEqual(len(static), 2)
        self.assertTrue(static_columns)
        self.assertFalse(temporal[ALL_HIST_EXOG].isna().any().any())


if __name__ == "__main__":
    unittest.main()
