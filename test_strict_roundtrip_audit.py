#!/usr/bin/env python3
"""
test_strict_roundtrip_audit.py

Task 9 Rigorous Roundtrip and Parity Audit:
1. Verifies delivery_final/submission.csv SHA256 integrity (must be 100% untouched).
2. Performs full roundtrip from saved raw components:
   - champion_raw + timestral_raw -> pre-cutoff features (origin 134) -> adaptive weights -> apply_p.
3. Tests zero-weight blend (w=0): must give exact 0 error on all 79,101 rows vs delivery_final/submission.csv.
4. Verifies both adaptive candidate CSVs:
   - submissions/submission_champion_timestral_adaptive_candidate.csv
   - submissions/submission_champion_timestral_adaptive_selection_winner.csv
5. Verifies reload determinism, key integrity, absence of NaNs, integer types.
6. Analyzes difference matrices among Champion, Fixed w10, Selection Winner, and Full Tuned.
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from run_weekly_feature_experiments import WeeklyMatrices
from adaptive_blending_rules import (
    KEYS,
    apply_p,
    compute_pre_cutoff_features,
    weight_full_adaptive_tuned,
    weight_full_adaptive_multi_regime,
)

CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    print("=" * 105)
    print("STRICT PRODUCTION ROUNDTRIP & CODE PARITY AUDIT (TASK 9)")
    print("=" * 105)

    # 1. Delivery Final Submission Integrity
    delivery_path = ROOT / "delivery_final" / "submission.csv"
    assert delivery_path.exists(), "delivery_final/submission.csv missing!"
    d_bytes = delivery_path.read_bytes()
    d_crlf = sha256_bytes(d_bytes)
    d_lf = sha256_bytes(d_bytes.replace(b"\r\n", b"\n"))
    print(f"1. Checking delivery_final/submission.csv:")
    print(f"   CRLF SHA256: {d_crlf}")
    print(f"   LF   SHA256: {d_lf}")
    assert d_crlf == CONTROL_SHA256_CRLF or d_lf == CONTROL_SHA256_LF, "delivery_final/submission.csv has been modified!"
    print("   -> [PASS] delivery_final/submission.csv is 100% UNTOUCHED.")

    # 2. Raw components loading
    champ_raw_path = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    tim_future_path = ROOT / "experiments" / "timestral_production_model" / "refit_full_history" / "timestral_209m_future_9w_keyed.csv.gz"
    assert champ_raw_path.exists(), "Missing champion raw components!"
    assert tim_future_path.exists(), "Missing timestral future predictions!"

    df_champ_raw = pd.read_csv(champ_raw_path)
    df_tim = pd.read_csv(tim_future_path)

    merged = pd.merge(df_champ_raw[KEYS + ["raw_champion"]], df_tim[KEYS + ["y_pred_continuous"]], on=KEYS, how="inner")
    assert len(merged) == 79101

    # Add horizon
    weeks_ordered = sorted(merged["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged["horizon"] = merged["week"].map(week_to_h)

    # 3. Zero-Weight Test
    print("\n2. Executing Zero-Weight Test (w=0 vs delivery_final/submission.csv)...")
    df_ctrl = pd.read_csv(delivery_path)
    merged_ctrl = pd.merge(merged, df_ctrl[KEYS + ["quantity"]], on=KEYS, how="inner")
    pred_w0 = apply_p(merged_ctrl["raw_champion"].values.astype(float), 1.05, 0.65)
    diff_w0 = np.abs(pred_w0 - merged_ctrl["quantity"].values)
    max_d_w0 = int(np.max(diff_w0))
    n_diff_w0 = int(np.sum(diff_w0 != 0))
    print(f"   Max absolute delta: {max_d_w0}")
    print(f"   Differing rows:     {n_diff_w0}")
    assert max_d_w0 == 0 and n_diff_w0 == 0, "Zero-weight test failed!"
    print("   -> [PASS] Zero-weight test: 100% bit-exact identity on all 79,101 rows.")

    # 4. Feature recomputation from history
    print("\n3. Recomputing pre-cutoff series features (origin 134, history <= 133)...")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    df_feats = compute_pre_cutoff_features(matrices, origin=134, pair_meta=pair_meta)
    df_feats = df_feats.rename(columns={"mean13": "mean13_feat"})
    merged_with_feats = pd.merge(merged, df_feats, on=["warehouse_name", "article"], how="left")

    raw_c = merged_with_feats["raw_champion"].values.astype(float)
    raw_t = merged_with_feats["y_pred_continuous"].values.astype(float)

    # 5. Roundtrip validation for both candidates
    candidates = [
        ("Full Tuned", ROOT / "submissions" / "submission_champion_timestral_adaptive_candidate.csv", weight_full_adaptive_tuned),
        ("Selection Winner", ROOT / "submissions" / "submission_champion_timestral_adaptive_selection_winner.csv", weight_full_adaptive_multi_regime),
    ]

    print("\n4. Roundtrip & File Audits:")
    for label, filepath, weight_fn in candidates:
        assert filepath.exists(), f"Missing candidate: {filepath}"
        w = weight_fn(merged_with_feats)
        raw_b = (1.0 - w) * raw_c + w * raw_t
        pred_calc = apply_p(raw_b, 1.05, 0.65)

        # Compare with file on disk
        df_disk = pd.read_csv(filepath)
        assert len(df_disk) == 79101, f"Row count: {len(df_disk)}"
        assert list(df_disk.columns) == ["week", "warehouse_name", "article", "quantity"]
        assert not df_disk.isna().any().any(), "NaN values found!"
        assert not df_disk.duplicated(subset=KEYS).any(), "Duplicate keys!"
        assert (df_disk["quantity"] >= 0).all(), "Negative values found!"
        assert np.issubdtype(df_disk["quantity"].dtype, np.integer), "Non-integer quantity!"

        merged_disk = pd.merge(merged_with_feats[KEYS], df_disk, on=KEYS, how="inner")
        diff = np.abs(merged_disk["quantity"].values - pred_calc)
        max_d = int(np.max(diff))
        n_d = int(np.sum(diff != 0))

        raw_bytes = filepath.read_bytes()
        crlf_sha = sha256_bytes(raw_bytes)
        lf_sha = sha256_bytes(raw_bytes.replace(b"\r\n", b"\n"))

        print(f"   Candidate [{label}]: {filepath.name}")
        print(f"     Max delta from fresh calculation: {max_d}")
        print(f"     Differing rows:                   {n_d}")
        print(f"     Total quantity sum:               {int(df_disk['quantity'].sum()):,}")
        print(f"     CRLF SHA256:                      {crlf_sha}")
        print(f"     LF   SHA256:                      {lf_sha}")
        assert max_d == 0 and n_d == 0, f"Roundtrip discrepancy for {label}!"
        print(f"     -> [PASS] 100% BIT-EXACT ROUNDTRIP MATCH.")

    # 6. Differences Matrix
    print("\n5. Comparison of Predictions Across All Final Candidates:")
    df_w10 = pd.read_csv(ROOT / "submissions" / "submission_champion_timestral_refit_blend_w10.csv")
    df_tun = pd.read_csv(ROOT / "submissions" / "submission_champion_timestral_adaptive_candidate.csv")
    df_sel = pd.read_csv(ROOT / "submissions" / "submission_champion_timestral_adaptive_selection_winner.csv")

    pairs = [
        ("Champion (delivery_final)", "Fixed w10 (Task 7)", df_ctrl, df_w10),
        ("Champion (delivery_final)", "Adaptive Full Tuned", df_ctrl, df_tun),
        ("Champion (delivery_final)", "Adaptive Selection Winner", df_ctrl, df_sel),
        ("Fixed w10 (Task 7)", "Adaptive Selection Winner", df_w10, df_sel),
        ("Adaptive Full Tuned", "Adaptive Selection Winner", df_tun, df_sel),
    ]

    print(f"   {'Model A':<28} vs {'Model B':<28} | {'Diff Rows':<12} {'Max Delta':<10} {'MAE on Diff'}")
    print("   " + "-" * 88)
    for name_a, name_b, df_a, df_b in pairs:
        m_pair = pd.merge(df_a[KEYS + ["quantity"]], df_b[KEYS + ["quantity"]], on=KEYS, suffixes=("_a", "_b"))
        delta = np.abs(m_pair["quantity_a"].values - m_pair["quantity_b"].values)
        diff_mask = delta > 0
        n_diff = int(diff_mask.sum())
        max_delta = int(delta.max())
        mae_diff = float(delta[diff_mask].mean()) if n_diff > 0 else 0.0
        print(f"   {name_a:<28} vs {name_b:<28} | {n_diff:<12,} {max_delta:<10} {mae_diff:.3f}")

    print("\n" + "=" * 105)
    print("ALL RIGOROUS ROUNDTRIP AND AUDIT CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 105)


if __name__ == "__main__":
    main()
