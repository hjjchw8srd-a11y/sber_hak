#!/usr/bin/env python3
"""
generate_adaptive_timestral_candidate.py

Generates the adaptive candidate submission:
`submissions/submission_champion_timestral_adaptive_candidate.csv`
based on the best validated adaptive scheme:
Adaptive Full Tuned (5-Fold Clipped Score = 0.728106).

Strict Safeguards:
1. Verifies delivery_final/submission.csv integrity.
2. Mandatory Zero-Weight Test: blend(w=0) must equal delivery_final/submission.csv on 100% of rows (max abs diff = 0).
3. Evaluates strictly pre-cutoff features at origin 134 (weeks <= 133).
4. Verifies 79,101 rows, keys, format, CRLF/LF SHA256.
5. Saves manifest to experiments/timestral_feature_research/adaptive_candidate_manifest.json.
"""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from run_weekly_feature_experiments import WeeklyMatrices

KEYS = ["week", "warehouse_name", "article"]
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 95)
    print("GENERATING ADAPTIVE TIMESTRAL BLEND PRODUCTION CANDIDATE")
    print("=" * 95)

    # 1. Verify control champion submission
    control_sub_path = ROOT / "delivery_final" / "submission.csv"
    assert control_sub_path.exists(), f"Control submission missing: {control_sub_path}"
    control_bytes = control_sub_path.read_bytes()
    control_crlf_sha = sha256_bytes(control_bytes)
    control_lf_sha = sha256_bytes(control_bytes.replace(b"\r\n", b"\n"))
    print(f"1. Verified control submission: {control_sub_path}")
    print(f"   CRLF SHA256: {control_crlf_sha}")
    print(f"   LF   SHA256: {control_lf_sha}")
    assert control_crlf_sha == CONTROL_SHA256_CRLF or control_lf_sha == CONTROL_SHA256_LF, "Control submission corrupted!"

    # 2. Load Champion continuous raw predictions
    champ_raw_path = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    assert champ_raw_path.exists(), f"Missing champion raw file: {champ_raw_path}"
    df_champ_raw = pd.read_csv(champ_raw_path)
    assert len(df_champ_raw) == 79101
    assert "raw_champion" in df_champ_raw.columns

    # 3. Mandatory Zero-Weight Test
    print("\n2. Executing Zero-Weight Test (w=0 vs delivery_final/submission.csv)...")
    df_control = pd.read_csv(control_sub_path)
    merged_ctrl = pd.merge(df_champ_raw[KEYS + ["raw_champion"]], df_control[KEYS + ["quantity"]], on=KEYS, how="inner")
    assert len(merged_ctrl) == 79101

    pred_w0 = apply_p(merged_ctrl["raw_champion"].values.astype(np.float64), 1.05, 0.65)
    diff_w0 = np.abs(pred_w0 - merged_ctrl["quantity"].values)
    max_diff_w0 = int(np.max(diff_w0))
    n_diff_w0 = int(np.sum(diff_w0 != 0))

    print(f"   Max absolute difference (w=0 vs control): {max_diff_w0}")
    print(f"   Number of differing rows:                 {n_diff_w0}")
    assert max_diff_w0 == 0 and n_diff_w0 == 0, f"ZERO-WEIGHT TEST FAILED! Differing rows: {n_diff_w0}, max diff: {max_diff_w0}"
    print("   -> ZERO-WEIGHT TEST PASSED WITH 100% BIT-EXACT IDENTITY!")

    # 4. Load Timestral continuous future predictions
    tim_future_path = ROOT / "experiments" / "timestral_production_model" / "refit_full_history" / "timestral_209m_future_9w_keyed.csv.gz"
    assert tim_future_path.exists(), f"Missing Timestral future predictions: {tim_future_path}"
    df_tim = pd.read_csv(tim_future_path)
    assert len(df_tim) == 79101
    assert "y_pred_continuous" in df_tim.columns
    tim_future_sha = sha256_file(tim_future_path)
    print(f"\n3. Loaded Timestral future predictions: {tim_future_path.name} (SHA256: {tim_future_sha})")

    # 5. Compute pre-cutoff series features at origin 134 (cutoff = week 133)
    print("\n4. Computing pre-cutoff series features (Cutoff: week 133)...")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    origin = 134  # Test horizon begins at week index 134
    Q13 = matrices.quantity[origin - 13 : origin, :]
    mean13 = Q13.mean(axis=0)
    nz13 = (Q13 > 0).mean(axis=0)

    n_series = matrices.quantity.shape[1]
    zero_streaks = np.zeros(n_series, dtype=int)
    for i in range(n_series):
        streak = 0
        for t in range(origin - 1, -1, -1):
            if matrices.quantity[t, i] == 0:
                streak += 1
            else:
                break
        zero_streaks[i] = streak

    p90_vol = np.percentile(mean13, 90)
    p50_vol = np.percentile(mean13, 50)

    df_feats = matrices.pairs.copy()
    df_feats["mean13"] = mean13
    df_feats["nz13"] = nz13
    df_feats["zero_streak"] = zero_streaks
    df_feats["p90_vol"] = p90_vol
    df_feats["p50_vol"] = p50_vol

    # 6. Merge and assign adaptive weights
    merged = pd.merge(df_champ_raw[KEYS + ["raw_champion"]], df_tim[KEYS + ["y_pred_continuous"]], on=KEYS, how="inner")
    assert len(merged) == 79101

    weeks_ordered = sorted(merged["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged["horizon"] = merged["week"].map(week_to_h)

    merged = pd.merge(merged, df_feats, on=["warehouse_name", "article"], how="left")

    # Adaptive Full Tuned Rule:
    # - Dead series (streak >= 6 or nz13 < 0.15) OR low-volume (< p50): w = 0.00
    # - High-volume (>= p90): w = 0.15 (for h <= 6), w = 0.12 (for h >= 7)
    # - Mid-volume (p50..p90): w = 0.10 (for h <= 6), w = 0.07 (for h >= 7)
    nz13_arr = merged["nz13"].values
    streak_arr = merged["zero_streak"].values
    mean13_arr = merged["mean13"].values
    h_arr = merged["horizon"].values

    base_w = np.where((nz13_arr < 0.15) | (streak_arr >= 6) | (mean13_arr < p50_vol), 0.00,
                np.where(mean13_arr >= p90_vol, 0.15, 0.10))
    adaptive_w = np.where(h_arr >= 7, np.maximum(0.0, base_w - 0.03), base_w).astype(np.float64)

    merged["weight_timestral"] = adaptive_w
    mean_w = float(adaptive_w.mean())
    print(f"\n5. Adaptive Weight Allocation:")
    print(f"   Mean Timestral weight:    {mean_w:.4f}")
    print(f"   Zero weight rows (w=0):   {int((adaptive_w == 0.0).sum()):,} ({(adaptive_w == 0.0).mean()*100:.1f}%)")
    print(f"   Mid weight rows (w>0,<0.12): {int(((adaptive_w > 0.0) & (adaptive_w < 0.12)).sum()):,} ({( (adaptive_w > 0.0) & (adaptive_w < 0.12)).mean()*100:.1f}%)")
    print(f"   High weight rows (w>=0.12):  {int((adaptive_w >= 0.12).sum()):,} ({(adaptive_w >= 0.12).mean()*100:.1f}%)")

    # 7. Compute Blend and apply single postprocessing
    raw_c = merged["raw_champion"].values.astype(np.float64)
    raw_t = merged["y_pred_continuous"].values.astype(np.float64)
    raw_blend = (1.0 - adaptive_w) * raw_c + adaptive_w * raw_t
    pred_final = apply_p(raw_blend, 1.05, 0.65)

    sub_candidate = merged[KEYS].copy()
    sub_candidate["quantity"] = pred_final

    # 8. Integrity and Format Validation
    assert len(sub_candidate) == 79101, f"Row count mismatch: {len(sub_candidate)}"
    assert not sub_candidate[KEYS].isna().any().any(), "NaN keys in submission!"
    assert not sub_candidate.duplicated(subset=KEYS).any(), "Duplicate keys in submission!"
    assert (sub_candidate["quantity"] >= 0).all(), "Negative quantities found!"
    assert np.issubdtype(sub_candidate["quantity"].dtype, np.integer), "Quantity column must be integer!"

    out_csv = ROOT / "submissions" / "submission_champion_timestral_adaptive_candidate.csv"
    sub_candidate.to_csv(out_csv, index=False, lineterminator="\r\n")

    raw_bytes = out_csv.read_bytes()
    lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
    crlf_sha = sha256_bytes(raw_bytes)
    lf_sha = sha256_bytes(lf_bytes)

    print(f"\n6. Candidate Created: {out_csv.name}")
    print(f"   Path:        {out_csv}")
    print(f"   Total rows:  {len(sub_candidate):,}")
    print(f"   Total sum:   {int(sub_candidate['quantity'].sum()):,}")
    print(f"   CRLF Size:   {len(raw_bytes):,} bytes | SHA256: {crlf_sha}")
    print(f"   LF   Size:   {len(lf_bytes):,} bytes | SHA256: {lf_sha}")

    # 9. Reload Roundtrip Verification
    df_reloaded = pd.read_csv(out_csv)
    assert len(df_reloaded) == 79101
    assert (df_reloaded["quantity"] == sub_candidate["quantity"]).all(), "Reload values mismatch!"
    print("   -> Reload roundtrip verification: 100% IDENTICAL!")

    # 10. Write Manifest
    manifest = {
        "candidate_file": "submissions/submission_champion_timestral_adaptive_candidate.csv",
        "recipe": "Adaptive Full Tuned Blend (Zero 0, Low 0, Mid 10/7, High 15/12)",
        "cv_validation": {
            "selection_folds_4_3_2_clipped": 0.730320,
            "check_folds_1_0_clipped": 0.724785,
            "full_5_fold_clipped": 0.728106,
            "full_5_fold_unclipped": 0.752486,
            "full_5_fold_wape": 49.298,
            "champion_5_fold_clipped": 0.728728,
            "fixed_w10_5_fold_clipped": 0.728249,
            "improvement_vs_champion": -0.000622,
            "improvement_vs_fixed_w10": -0.000143,
        },
        "zero_weight_test": {
            "status": "PASSED_BIT_EXACT",
            "differing_rows": 0,
            "max_abs_diff": 0,
        },
        "file_integrity": {
            "rows": 79101,
            "quantity_sum": int(sub_candidate["quantity"].sum()),
            "crlf_bytes": len(raw_bytes),
            "crlf_sha256": crlf_sha,
            "lf_bytes": len(lf_bytes),
            "lf_sha256": lf_sha,
        },
        "mean_timestral_weight": mean_w,
    }

    manifest_path = ROOT / "experiments" / "timestral_feature_research" / "adaptive_candidate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n7. Manifest written to {manifest_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
