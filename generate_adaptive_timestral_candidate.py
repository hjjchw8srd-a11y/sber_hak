#!/usr/bin/env python3
"""
generate_adaptive_timestral_candidate.py

Unified production candidate generator & verification for Adaptive Timestral blends.
Uses centralized `adaptive_blending_rules.py` to ensure 100% parity with CV experiments.

Generates and audits:
1. `submissions/submission_champion_timestral_adaptive_candidate.csv`
   (Full Tuned rule: w=0 for low/dead, w=0.10/0.07 mid, w=0.15/0.12 high)
   CV: 5-Fold Clipped 0.728106, Full Unclipped 0.752486

2. `submissions/submission_champion_timestral_adaptive_selection_winner.csv`
   (Selection Winner rule: Full Multi-Regime chosen strictly on Folds [4,3,2] without Check fold influence)
   CV: Selection Clipped 0.730213, Selection Unclipped 0.756014, 5-Fold Clipped 0.728113, Full Unclipped 0.752488

Strict Safeguards:
- delivery_final/submission.csv checked against control SHA256 (must be untouched).
- Mandatory Zero-Weight Test: blend(w=0) must equal delivery_final/submission.csv on 100% of rows (max abs diff = 0).
- Features calculated strictly pre-cutoff (origin 134, weeks <= 133).
- Verification of 79,101 rows, keys, format, CRLF/LF SHA256.
- Reload roundtrip test for both candidates.
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
from adaptive_blending_rules import (
    KEYS,
    apply_p,
    compute_pre_cutoff_features,
    weight_full_adaptive_tuned,
    weight_full_adaptive_multi_regime,
)

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


def build_candidate_dataframe(
    merged_base: pd.DataFrame,
    df_feats: pd.DataFrame,
    weight_fn,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Applies pre-cutoff features and weighting function, computes continuous blend and postprocessing."""
    df_merged = pd.merge(merged_base, df_feats, on=["warehouse_name", "article"], how="left")
    weights = weight_fn(df_merged).astype(np.float64)

    raw_c = df_merged["raw_champion"].values.astype(np.float64)
    raw_t = df_merged["y_pred_continuous"].values.astype(np.float64)

    raw_blend = (1.0 - weights) * raw_c + weights * raw_t
    pred = apply_p(raw_blend, 1.05, 0.65)

    sub = df_merged[KEYS].copy()
    sub["quantity"] = pred
    return sub, weights


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 105)
    print("PRODUCTION ADAPTIVE CANDIDATES GENERATION & AUDIT (TASK 9)")
    print("=" * 105)

    # 1. Verify control champion submission
    control_sub_path = ROOT / "delivery_final" / "submission.csv"
    assert control_sub_path.exists(), f"Control submission missing: {control_sub_path}"
    control_bytes = control_sub_path.read_bytes()
    control_crlf_sha = sha256_bytes(control_bytes)
    control_lf_sha = sha256_bytes(control_bytes.replace(b"\r\n", b"\n"))
    print(f"1. Verified control submission: {control_sub_path.name}")
    print(f"   CRLF SHA256: {control_crlf_sha}")
    print(f"   LF   SHA256: {control_lf_sha}")
    assert control_crlf_sha == CONTROL_SHA256_CRLF or control_lf_sha == CONTROL_SHA256_LF, "Control submission corrupted!"
    print("   -> CONTROL DELIVERY_FINAL/SUBMISSION.CSV IS 100% UNTOUCHED.")

    # 2. Load Champion continuous raw predictions
    champ_raw_path = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    assert champ_raw_path.exists(), f"Missing champion raw file: {champ_raw_path}"
    df_champ_raw = pd.read_csv(champ_raw_path)
    assert len(df_champ_raw) == 79101
    assert "raw_champion" in df_champ_raw.columns

    # 3. Mandatory Zero-Weight Test
    print("\n2. Executing Zero-Weight Test (blend(w=0) vs delivery_final/submission.csv)...")
    df_control = pd.read_csv(control_sub_path)
    merged_ctrl = pd.merge(df_champ_raw[KEYS + ["raw_champion"]], df_control[KEYS + ["quantity"]], on=KEYS, how="inner")
    assert len(merged_ctrl) == 79101

    pred_w0 = apply_p(merged_ctrl["raw_champion"].values.astype(np.float64), 1.05, 0.65)
    diff_w0 = np.abs(pred_w0 - merged_ctrl["quantity"].values)
    max_diff_w0 = int(np.max(diff_w0))
    n_diff_w0 = int(np.sum(diff_w0 != 0))

    print(f"   Max absolute difference: {max_diff_w0}")
    print(f"   Number of differing rows: {n_diff_w0}")
    assert max_diff_w0 == 0 and n_diff_w0 == 0, f"Zero-weight test failed! Differing: {n_diff_w0}, max diff: {max_diff_w0}"
    print("   -> ZERO-WEIGHT TEST PASSED WITH EXACT 0 ERROR ACROSS ALL 79,101 ROWS!")

    # 4. Load Timestral continuous future predictions
    tim_future_path = ROOT / "experiments" / "timestral_production_model" / "refit_full_history" / "timestral_209m_future_9w_keyed.csv.gz"
    assert tim_future_path.exists(), f"Missing Timestral future predictions: {tim_future_path}"
    df_tim = pd.read_csv(tim_future_path)
    assert len(df_tim) == 79101
    assert "y_pred_continuous" in df_tim.columns
    tim_future_sha = sha256_file(tim_future_path)
    print(f"\n3. Loaded Timestral future predictions: {tim_future_path.name} (SHA256: {tim_future_sha})")

    # 5. Base merge with Horizon mapping
    merged_base = pd.merge(df_champ_raw[KEYS + ["raw_champion"]], df_tim[KEYS + ["y_pred_continuous"]], on=KEYS, how="inner")
    assert len(merged_base) == 79101
    weeks_ordered = sorted(merged_base["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged_base["horizon"] = merged_base["week"].map(week_to_h)

    # 6. Compute pre-cutoff series features at origin 134 (cutoff: week 133)
    print("\n4. Computing pre-cutoff features via adaptive_blending_rules (Cutoff: week 133, origin 134)...")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    df_feats = compute_pre_cutoff_features(matrices, origin=134, pair_meta=pair_meta)
    df_feats = df_feats.rename(columns={"mean13": "mean13_feat"})

    candidates_to_build = [
        {
            "name": "Adaptive: Full Tuned Candidate",
            "file": ROOT / "submissions" / "submission_champion_timestral_adaptive_candidate.csv",
            "fn": weight_full_adaptive_tuned,
            "cv_metrics": {
                "selection_unclipped": 0.756155,
                "selection_clipped": 0.730320,
                "check_unclipped": 0.746984,
                "check_clipped": 0.724785,
                "full_5f_unclipped": 0.752486,
                "full_5f_clipped": 0.728106,
            },
            "rule_description": "Full Tuned: Zero 0.00, Low 0.00, Mid 0.10/0.07, High 0.15/0.12",
        },
        {
            "name": "Adaptive: Selection Winner Candidate",
            "file": ROOT / "submissions" / "submission_champion_timestral_adaptive_selection_winner.csv",
            "fn": weight_full_adaptive_multi_regime,
            "cv_metrics": {
                "selection_unclipped": 0.756014,
                "selection_clipped": 0.730213,
                "check_unclipped": 0.747199,
                "check_clipped": 0.724963,
                "full_5f_unclipped": 0.752488,
                "full_5f_clipped": 0.728113,
            },
            "rule_description": "Full Multi-Regime: Dead 0.00, Low 0.02, Sparse 0.05, RegMid 0.10/0.08, RegHigh 0.18/0.12",
        },
    ]

    manifests = {}

    for cand_info in candidates_to_build:
        c_name = cand_info["name"]
        c_file = cand_info["file"]
        c_fn = cand_info["fn"]
        c_desc = cand_info["rule_description"]
        c_cv = cand_info["cv_metrics"]

        print(f"\n--- Processing {c_name} ---")
        sub_df, weights = build_candidate_dataframe(merged_base, df_feats, c_fn)

        # Integrity checks
        assert len(sub_df) == 79101, f"Row count: {len(sub_df)}"
        assert not sub_df[KEYS].isna().any().any(), "NaN keys!"
        assert not sub_df.duplicated(subset=KEYS).any(), "Duplicate keys!"
        assert (sub_df["quantity"] >= 0).all(), "Negative quantities!"
        assert np.issubdtype(sub_df["quantity"].dtype, np.integer), "Non-integer quantities!"

        # If file already exists, check if new calculation matches existing file bit-exactly
        if c_file.exists():
            existing_bytes = c_file.read_bytes()
            existing_crlf_sha = sha256_bytes(existing_bytes)
            existing_df = pd.read_csv(c_file)
            diff_from_existing = np.abs(existing_df["quantity"].values - sub_df["quantity"].values)
            max_delta_existing = int(np.max(diff_from_existing))
            differing_existing = int(np.sum(diff_from_existing != 0))
            print(f"   Roundtrip check against existing {c_file.name}:")
            print(f"     Max delta from existing on disk: {max_delta_existing}")
            print(f"     Differing rows from existing:    {differing_existing}")
            assert max_delta_existing == 0 and differing_existing == 0, "Roundtrip logic mismatch with existing candidate!"
            print(f"     -> VERIFIED: Existing candidate matches unified Python logic 100% bit-exactly!")

        # Write candidate with CRLF line endings
        sub_df.to_csv(c_file, index=False, lineterminator="\r\n")

        raw_bytes = c_file.read_bytes()
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        crlf_sha = sha256_bytes(raw_bytes)
        lf_sha = sha256_bytes(lf_bytes)

        # Reload roundtrip verification
        df_reloaded = pd.read_csv(c_file)
        assert len(df_reloaded) == 79101
        assert (df_reloaded["quantity"] == sub_df["quantity"]).all(), "Reload roundtrip failed!"

        print(f"   Candidate file: {c_file.name}")
        print(f"     Total rows:   {len(sub_df):,}")
        print(f"     Total sum:    {int(sub_df['quantity'].sum()):,}")
        print(f"     Mean weight:  {float(weights.mean()):.4f}")
        print(f"     CRLF SHA256:  {crlf_sha}")
        print(f"     LF   SHA256:  {lf_sha}")
        print(f"     Reload test:  100% BIT-EXACT PASS")

        manifests[c_file.name] = {
            "path": str(c_file.relative_to(ROOT)),
            "scheme_name": c_name,
            "rule_description": c_desc,
            "cv_metrics": c_cv,
            "mean_weight": float(weights.mean()),
            "rows": len(sub_df),
            "quantity_sum": int(sub_df["quantity"].sum()),
            "crlf_bytes": len(raw_bytes),
            "crlf_sha256": crlf_sha,
            "lf_bytes": len(lf_bytes),
            "lf_sha256": lf_sha,
            "zero_weight_test": "PASSED_BIT_EXACT",
            "reload_test": "PASSED_BIT_EXACT",
        }

    # Save comprehensive production manifest
    manifest_file = ROOT / "experiments" / "timestral_feature_research" / "adaptive_candidates_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)
    print(f"\nManifest saved to {manifest_file.name}")
    print("=" * 105)


if __name__ == "__main__":
    main()
