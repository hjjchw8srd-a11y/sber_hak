import unittest

import numpy as np
import pandas as pd

from train_tft_weekly import HIST_EXOG, prepare_frames


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

        original_origin = (
            original[original["ds"] == origin]
            .sort_values("unique_id")
            .reset_index(drop=True)
        )
        changed_origin = (
            changed_frame[changed_frame["ds"] == origin]
            .sort_values("unique_id")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            original_origin[HIST_EXOG].to_numpy(),
            changed_origin[HIST_EXOG].to_numpy(),
        )

    def test_prepared_frames_have_expected_shape(self) -> None:
        data = self.sample_data()
        temporal, static, static_columns = prepare_frames(data)
        self.assertEqual(len(temporal), len(data))
        self.assertEqual(temporal["unique_id"].nunique(), 2)
        self.assertEqual(len(static), 2)
        self.assertTrue(static_columns)
        self.assertFalse(temporal[HIST_EXOG].isna().any().any())


if __name__ == "__main__":
    unittest.main()
