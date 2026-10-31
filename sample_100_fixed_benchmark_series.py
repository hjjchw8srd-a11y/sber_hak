#!/usr/bin/env python3
"""
sample_100_fixed_benchmark_series.py

Samples 100 fixed representative series (warehouse_name, article pairs)
across 5 distinct historical demand regimes for foundation model benchmarking:
1. Low intermittent (zero_rate >= 0.85)
2. Moderate intermittent (0.50 <= zero_rate < 0.85)
3. Regular continuous demand (zero_rate < 0.40)
4. High volume (top sales volume, total_qty >= 500)
5. Stockout-prone (stock_nonpositive_rate >= 0.80, total_qty > 0)

Saves:
- experiments/benchmarks/fixed_100_series.csv
- experiments/benchmarks/fixed_100_series_manifest.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading weekly dataset to construct fixed 100 benchmark series...")
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    df = pd.read_csv(data_path)

    stats = df.groupby(["warehouse_name", "article"]).agg(
        total_qty=("quantity", "sum"),
        mean_qty=("quantity", "mean"),
        zero_rate=("quantity", lambda s: (s == 0).mean()),
        nonpos_stock_rate=("stock_nonpositive_rate", "mean"),
        category_lvl_1=("category_lvl_1", "first"),
        category_lvl_2=("category_lvl_2", "first"),
        category_lvl_3=("category_lvl_3", "first"),
    ).reset_index()

    stats["nonpos_stock_rate"] = stats["nonpos_stock_rate"].fillna(0.0)

    # 1. Stratified sampling with seed 42
    rng = np.random.RandomState(42)

    strata = {}
    # Stratum 1: High Intermittent (zero_rate >= 0.85, total_qty >= 5)
    s1 = stats[(stats["zero_rate"] >= 0.85) & (stats["total_qty"] >= 5)]
    strata["high_intermittent"] = s1.sample(n=20, random_state=42)

    # Stratum 2: Moderate Intermittent (0.50 <= zero_rate < 0.85)
    s2 = stats[(stats["zero_rate"] >= 0.50) & (stats["zero_rate"] < 0.85) & ~stats.index.isin(strata["high_intermittent"].index)]
    strata["moderate_intermittent"] = s2.sample(n=20, random_state=42)

    # Stratum 3: Regular Continuous (zero_rate < 0.40)
    s3 = stats[(stats["zero_rate"] < 0.40) & ~stats.index.isin(strata["high_intermittent"].index) & ~stats.index.isin(strata["moderate_intermittent"].index)]
    strata["regular_continuous"] = s3.sample(n=20, random_state=42)

    # Stratum 4: High Volume (total_qty >= 500)
    already_selected = set(strata["high_intermittent"].index) | set(strata["moderate_intermittent"].index) | set(strata["regular_continuous"].index)
    s4 = stats[(stats["total_qty"] >= 500) & ~stats.index.isin(already_selected)]
    strata["high_volume"] = s4.sample(n=20, random_state=42)

    # Stratum 5: Stockout Prone (nonpos_stock_rate >= 0.80, total_qty >= 20)
    already_selected |= set(strata["high_volume"].index)
    s5 = stats[(stats["nonpos_stock_rate"] >= 0.80) & (stats["total_qty"] >= 20) & ~stats.index.isin(already_selected)]
    strata["stockout_prone"] = s5.sample(n=20, random_state=42)

    combined_rows = []
    for regime_name, regime_df in strata.items():
        regime_df = regime_df.copy()
        regime_df["demand_regime"] = regime_name
        combined_rows.append(regime_df)

    benchmark_df = pd.concat(combined_rows, ignore_index=True)
    assert len(benchmark_df) == 100
    assert len(benchmark_df[["warehouse_name", "article"]].drop_duplicates()) == 100

    csv_path = BENCHMARK_DIR / "fixed_100_series.csv"
    benchmark_df.to_csv(csv_path, index=False)
    print(f"Saved 100 fixed series to {csv_path}")

    # Generate manifest
    manifest = {
        "dataset_name": "fixed_100_benchmark_series",
        "total_series": 100,
        "seed": 42,
        "regimes": {
            regime: {
                "count": len(df_sub),
                "mean_total_qty": float(df_sub["total_qty"].mean()),
                "mean_zero_rate": float(df_sub["zero_rate"].mean()),
                "mean_nonpos_stock_rate": float(df_sub["nonpos_stock_rate"].mean()),
            }
            for regime, df_sub in benchmark_df.groupby("demand_regime")
        },
        "series": [
            {
                "warehouse_name": str(row["warehouse_name"]),
                "article": str(row["article"]),
                "demand_regime": str(row["demand_regime"]),
                "zero_rate": round(float(row["zero_rate"]), 4),
                "total_qty": float(row["total_qty"]),
                "nonpos_stock_rate": round(float(row["nonpos_stock_rate"]), 4),
                "category_lvl_1": str(row["category_lvl_1"]),
            }
            for _, row in benchmark_df.iterrows()
        ]
    }

    manifest_path = BENCHMARK_DIR / "fixed_100_series_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
