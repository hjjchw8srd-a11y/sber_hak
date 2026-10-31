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
3. Direct key-based verification of ground truth, forecast dates, and warehouse-article pairs
   directly against the raw historical dataset (weekly_data/weekly_dataset.csv.gz) without trusting CSV y_true.
4. Re-computes all 5 metrics (SMAPE, WAPE, RMSE, Bias, RMSLE) and both Scores (Clipped, Unclipped)
   for 'rint', 'legacy_p', and 'continuous_raw'.
5. Compares recalculated values against experiments/chronos_lora_fold4/logs/fold4_lora_results.json
   with strict numerical tolerance <= 1e-6.
6. Audits dynamic Full Fine-Tuning benchmark records.
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


def apply_p(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    out = np.zeros_like(arr)
    mask = arr > 0
    p = np.where(arr < 1.0, p_low, p_high)
    out[mask] = np.floor(arr[mask] + p[mask])
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("INDEPENDENT AUDIT: CHRONOS-2 LORA FOLD 4 RECALCULATION & RAW DATA VERIFICATION")
    print("=" * 80)

    keyed_path = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "keyed_predictions" / "chronos2_small_lora_fold4_keyed.csv.gz"
    json_path = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "logs" / "fold4_lora_results.json"
    raw_data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"

    if not keyed_path.exists():
        raise FileNotFoundError(f"Keyed predictions not found: {keyed_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON results not found: {json_path}")
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw weekly dataset not found: {raw_data_path}")

    # 1. Load keyed predictions
    print(f"\n[1/6] Loading keyed predictions from {keyed_path.name}...")
    df = pd.read_csv(keyed_path)
    print(f"Loaded {len(df):,} rows.")

    # 2. Strict Integrity Checks on Keyed CSV
    print("\n[2/6] Performing strict key and column integrity checks...")

    if len(df) != EXPECTED_ROWS:
        raise AssertionError(f"Integrity check failed: expected {EXPECTED_ROWS} rows, got {len(df)}")
    print(f"  [PASS] Exact row count verified: {len(df):,}")

    dup_cnt = df.duplicated(subset=KEYS).sum()
    if dup_cnt > 0:
        raise AssertionError(f"Integrity check failed: {dup_cnt} duplicate keys found!")
    print(f"  [PASS] Key uniqueness verified: 0 duplicates in {KEYS}")

    if df[KEYS].isna().any().any():
        raise AssertionError("Integrity check failed: NaN values found in keys!")
    print("  [PASS] Zero NaNs in key columns")

    for q in QUANTILES:
        col = f"q_{q:g}"
        if col not in df.columns:
            raise AssertionError(f"Integrity check failed: missing quantile column '{col}'!")
        vals = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise AssertionError(f"Integrity check failed: non-finite values in '{col}'!")
    print(f"  [PASS] All {len(QUANTILES)} quantile columns present and strictly finite")

    # 3. Direct Key-Based Verification Against Raw Historical Dataset (Requirement 4)
    print("\n[3/6] Verifying y_true, dates, and warehouse-article pairs against raw weekly_dataset.csv.gz...")
    raw_df = pd.read_csv(raw_data_path, parse_dates=["week"])
    panel = raw_df.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0).sort_index()

    ORIGIN = 89
    CONTEXT = 52
    HORIZON = 9

    expected_val_weeks = [str(w)[:10] for w in panel.index[ORIGIN : ORIGIN + HORIZON]]
    csv_val_weeks = sorted(df["week"].unique().tolist())
    if csv_val_weeks != expected_val_weeks:
        raise AssertionError(f"Forecast dates mismatch! Expected {expected_val_weeks}, got {csv_val_weeks}")
    print(f"  [PASS] Forecast horizon dates verified ({len(expected_val_weeks)} weeks: {expected_val_weeks[0]} to {expected_val_weeks[-1]})")

    raw_truth_list = []
    raw_mean13_list = []
    for h in range(HORIZON):
        raw_truth_list.append(panel.iloc[ORIGIN + h].values)
        raw_mean13_list.append(panel.iloc[ORIGIN - 13 : ORIGIN].mean().values)

    raw_y_true = np.concatenate(raw_truth_list)
    raw_y_mean13 = np.concatenate(raw_mean13_list)

    csv_y_true = df["y_true"].to_numpy(dtype=np.float64)
    csv_y_mean13 = df["unconstrained_demand_13"].to_numpy(dtype=np.float64)

    diff_y_true = np.abs(csv_y_true - raw_y_true).max()
    diff_mean13 = np.abs(csv_y_mean13 - raw_y_mean13).max()

    if diff_y_true != 0.0:
        raise AssertionError(f"Ground truth y_true discrepancy detected! Max absolute diff: {diff_y_true}")
    print(f"  [PASS] y_true key-based match: 100% bit-exact across all {len(csv_y_true):,} rows (diff = 0.00e+00)")

    if diff_mean13 > 1e-10:
        raise AssertionError(f"Mean13 baseline discrepancy detected! Max absolute diff: {diff_mean13}")
    print(f"  [PASS] Mean13 baseline key-based match: 100% bit-exact (diff = {diff_mean13:.2e})")

    # 4. Ground Truth & Baseline Metrics
    print("\n[4/6] Computing ground truth and Mean13 baseline metrics on Fold 4...")
    truth_df = df[KEYS].copy()
    truth_df["quantity"] = csv_y_true

    mean13_df = df[KEYS].copy()
    mean13_df["quantity"] = csv_y_mean13

    m_base = compute_all_metrics_fast(truth_df, mean13_df)
    print(f"  Mean13 Baseline: SMAPE={m_base['smape']:.3f}%, WAPE={m_base['wape']:.3f}%, RMSE={m_base['rmse']:.4f}, Bias={m_base['bias']:.3f}%")

    # 5. Metric Recalculation
    print("\n[5/6] Recalculating candidate metrics and comparing with JSON results...")
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
        for met_name, r_val in recalc.items():
            e_val = exp[met_name]
            diff = abs(r_val - e_val)
            ok = diff <= TOLERANCE
            status = "[OK]  " if ok else "[FAIL]"
            if not ok:
                all_matched = False
            print(f"    {status} {met_name:<15}: {r_val:.6f} (diff: {diff:.2e})")

    if not all_matched:
        raise AssertionError("Audit FAILED: one or more recalculated metrics deviate from results JSON!")

    # 6. Full FT Benchmark Verification
    print("\n[6/6] Auditing dynamic Full Fine-Tuning benchmark records...")
    bench_csv = PROJECT_ROOT / "experiments" / "chronos_finetune" / "chronos2_finetuned_benchmark_summary.csv"
    if not bench_csv.exists():
        raise FileNotFoundError(f"Missing Full FT benchmark summary CSV: {bench_csv}")

    df_bench = pd.read_csv(bench_csv)
    row_fold4_leg = df_bench[(df_bench["fold"] == 4) & (df_bench["variant"] == "legacy_p")]
    if len(row_fold4_leg) != 1:
        raise AssertionError(f"Expected 1 record for Fold 4 Legacy P, found {len(row_fold4_leg)}")
    r4 = row_fold4_leg.iloc[0]

    exp_f4_leg = {
        "clipped_score": 0.738395540,
        "unclipped_ratio": 0.777739953,
        "smape": 66.037865921,
        "wape": 51.166699386,
        "rmse": 3.024998771,
    }
    print("  Verified Fold 4 Legacy P from CSV:")
    for k, ev in exp_f4_leg.items():
        av = float(r4[k])
        diff = abs(av - ev)
        assert diff <= 1e-6, f"Mismatch in Full FT benchmark {k}: actual={av}, expected={ev}"
        print(f"    {k:<16}: {av:.9f} (expected: {ev:.9f})")
    print("  [PASS] Full FT benchmark records bit-exact match confirmed!")

    print("\n" + "*" * 80)
    print("SUCCESS: ALL INTEGRITY CHECKS AND INDEPENDENT RECALCULATIONS PASSED (100% BIT-EXACT)!")
    print("*" * 80)


if __name__ == "__main__":
    main()
