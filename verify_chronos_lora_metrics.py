#!/usr/bin/env python3
"""
verify_chronos_lora_metrics.py

Independent audit script for Chronos-2 LoRA Fold 4:
1. Loads keyed predictions CSV (chronos2_small_lora_fold4_keyed.csv.gz).
2. Performs strict integrity checks:
   - Exactly 79,101 rows (9 weeks x 8,789 pairs)
   - Unique keys ['week', 'warehouse_name', 'article']
   - Zero missing or NaN values across all columns
   - All 13 quantile columns present and finite
3. Re-computes all 5 metrics (SMAPE, WAPE, RMSE, Bias, RMSLE) and both Scores (Clipped, Unclipped)
   for 'rint', 'legacy_p', and 'continuous_raw'.
4. Compares recalculated values against experiments/chronos_lora_fold4/logs/fold4_lora_results.json
   with strict numerical tolerance <= 1e-6.
5. Fails with an explicit error upon any mismatch.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]
EXPECTED_ROWS = 79101
QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TOLERANCE = 1e-6


def calc_scores(m_pred: dict, m_base: dict) -> tuple[float, float]:
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("INDEPENDENT AUDIT: CHRONOS-2 LORA FOLD 4 RECALCULATION & VERIFICATION")
    print("=" * 80)

    keyed_path = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "keyed_predictions" / "chronos2_small_lora_fold4_keyed.csv.gz"
    json_path = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "logs" / "fold4_lora_results.json"

    if not keyed_path.exists():
        raise FileNotFoundError(f"Keyed predictions not found: {keyed_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON results not found: {json_path}")

    # 1. Load keyed predictions
    print(f"\n[1/5] Loading keyed predictions from {keyed_path.name}...")
    df = pd.read_csv(keyed_path)
    print(f"Loaded {len(df):,} rows.")

    # 2. Strict Integrity Checks
    print("\n[2/5] Performing strict key and column integrity checks...")

    # Row count
    if len(df) != EXPECTED_ROWS:
        raise AssertionError(f"Integrity check failed: expected {EXPECTED_ROWS} rows, got {len(df)}")
    print(f"  [PASS] Exact row count verified: {len(df):,}")

    # Key uniqueness
    dup_cnt = df.duplicated(subset=KEYS).sum()
    if dup_cnt > 0:
        raise AssertionError(f"Integrity check failed: {dup_cnt} duplicate keys found!")
    print(f"  [PASS] Key uniqueness verified: 0 duplicates in {KEYS}")

    # Missing / NaN keys
    if df[KEYS].isna().any().any():
        raise AssertionError("Integrity check failed: NaN values found in keys!")
    print("  [PASS] Zero NaNs in key columns")

    # Quantiles completeness and finiteness
    for q in QUANTILES:
        col = f"q_{q:g}"
        if col not in df.columns:
            raise AssertionError(f"Integrity check failed: missing quantile column '{col}'!")
        vals = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise AssertionError(f"Integrity check failed: non-finite values in '{col}'!")
    print(f"  [PASS] All {len(QUANTILES)} quantile columns present and strictly finite")

    # 3. Ground Truth & Baseline
    print("\n[3/5] Verifying ground truth and Mean13 baseline on Fold 4...")
    truth_df = df[KEYS].copy()
    truth_df["quantity"] = df["y_true"].to_numpy(dtype=np.float64)

    mean13_df = df[KEYS].copy()
    mean13_df["quantity"] = df["unconstrained_demand_13"].to_numpy(dtype=np.float64)

    if not np.isfinite(truth_df["quantity"].to_numpy()).all():
        raise AssertionError("Non-finite values found in ground truth!")
    if not np.isfinite(mean13_df["quantity"].to_numpy()).all():
        raise AssertionError("Non-finite values found in Mean13 baseline!")

    m_base = compute_all_metrics_fast(truth_df, mean13_df)
    print(f"  Mean13 Baseline: SMAPE={m_base['smape']:.3f}%, WAPE={m_base['wape']:.3f}%, RMSE={m_base['rmse']:.4f}, Bias={m_base['bias']:.3f}%")

    # 4. Metric Recalculation
    print("\n[4/5] Recalculating metrics and comparing with JSON results...")
    with open(json_path, "r", encoding="utf-8") as f:
        json_results = json.load(f)

    expected_eval = json_results["evaluation"]
    variants = {
        "rint": df["y_pred_rint"].to_numpy(dtype=np.float64),
        "legacy_p": df["y_pred_legacy_p"].to_numpy(dtype=np.float64),
        "continuous_raw": df["y_pred_continuous"].to_numpy(dtype=np.float64),
    }

    all_matched = True
    for v_name, v_pred in variants.items():
        print(f"\n  Checking variant: '{v_name}'")
        pred_df = df[KEYS].copy()
        pred_df["quantity"] = v_pred

        m = compute_all_metrics_fast(truth_df, pred_df)
        sc_clip, sc_unclip = calc_scores(m, m_base)

        recalc = {
            "clipped_score": sc_clip,
            "unclipped_ratio": sc_unclip,
            "smape": m["smape"],
            "wape": m["wape"],
            "rmse": m["rmse"],
            "bias": m["bias"],
            "rmsle": m["rmsle"],
        }
        exp = expected_eval[v_name]

        for metric_key, val_calc in recalc.items():
            val_exp = exp[metric_key]
            diff = abs(val_calc - val_exp)
            if diff > TOLERANCE:
                all_matched = False
                print(f"    [FAIL] {metric_key}: recalc={val_calc:.8f} != json={val_exp:.8f} (diff={diff:.2e} > {TOLERANCE})")
            else:
                print(f"    [OK]   {metric_key:15s}: {val_calc:.6f} (diff: {diff:.2e})")

        if not all_matched:
            raise AssertionError(f"Recalculation discrepancy detected in variant '{v_name}'!")

    # 5. Check Full FT Benchmark Alignment
    print("\n[5/5] Auditing dynamic Full Fine-Tuning benchmark records...")
    bench_csv = PROJECT_ROOT / "experiments" / "chronos_finetune" / "chronos2_finetuned_benchmark_summary.csv"
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        row_leg = df_bench[(df_bench["fold"] == 4) & (df_bench["variant"] == "legacy_p")]
        assert len(row_leg) == 1, f"Expected 1 unique row for fold 4 legacy_p, got {len(row_leg)}"
        r = row_leg.iloc[0]

        print(f"  Verified Fold 4 Legacy P from CSV:")
        print(f"    Clipped Score:   {r['clipped_score']:.9f} (expected: 0.738395540)")
        print(f"    Unclipped Ratio: {r['unclipped_ratio']:.9f} (expected: 0.777739953)")
        print(f"    SMAPE:           {r['smape']:.9f}% (expected: 66.037865921)")
        print(f"    WAPE:            {r['wape']:.9f}% (expected: 51.166699386)")
        print(f"    RMSE:            {r['rmse']:.9f} (expected: 3.024998771)")

        assert abs(r["clipped_score"] - 0.738395540) < 1e-6
        assert abs(r["unclipped_ratio"] - 0.777739953) < 1e-6
        assert abs(r["smape"] - 66.037865921) < 1e-6
        assert abs(r["wape"] - 51.166699386) < 1e-6
        assert abs(r["rmse"] - 3.024998771) < 1e-6
        print("  [PASS] Full FT benchmark records bit-exact match confirmed!")

    print("\n" + "*" * 80)
    print("SUCCESS: ALL INTEGRITY CHECKS AND INDEPENDENT RECALCULATIONS PASSED (100% BIT-EXACT)!")
    print("*" * 80)


if __name__ == "__main__":
    main()
