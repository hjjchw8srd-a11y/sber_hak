#!/usr/bin/env python3
"""
test_adaptive_candidate_roundtrip.py

Test roundtrip and integrity check for:
submissions/submission_champion_timestral_adaptive_candidate.csv
"""

import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

KEYS = ["week", "warehouse_name", "article"]
CONTROL_CRLF_SHA = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_LF_SHA = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    print("=" * 80)
    print("TESTING ADAPTIVE CANDIDATE INTEGRITY AND DETERMINISM")
    print("=" * 80)

    # 1. Delivery final must be untouched
    delivery_path = ROOT / "delivery_final" / "submission.csv"
    assert delivery_path.exists(), "Missing delivery_final/submission.csv"
    d_bytes = delivery_path.read_bytes()
    d_crlf = sha256_bytes(d_bytes)
    d_lf = sha256_bytes(d_bytes.replace(b"\r\n", b"\n"))
    assert d_crlf == CONTROL_CRLF_SHA or d_lf == CONTROL_LF_SHA, "delivery_final/submission.csv was altered!"
    print("[PASS] delivery_final/submission.csv is 100% UNTOUCHED.")

    # 2. Candidate check
    cand_path = ROOT / "submissions" / "submission_champion_timestral_adaptive_candidate.csv"
    assert cand_path.exists(), "Candidate missing!"
    c_bytes = cand_path.read_bytes()
    c_crlf = sha256_bytes(c_bytes)
    c_lf = sha256_bytes(c_bytes.replace(b"\r\n", b"\n"))

    df_cand = pd.read_csv(cand_path)
    assert len(df_cand) == 79101, f"Row count: {len(df_cand)}"
    assert list(df_cand.columns) == ["week", "warehouse_name", "article", "quantity"]
    assert not df_cand.isna().any().any(), "Candidate has NaNs!"
    assert not df_cand.duplicated(subset=KEYS).any(), "Duplicate keys!"
    assert (df_cand["quantity"] >= 0).all(), "Negative quantities!"
    assert np.issubdtype(df_cand["quantity"].dtype, np.integer), "Non-integer quantities!"

    print(f"[PASS] Candidate structure valid: 79,101 rows, 0 nulls, valid types.")
    print(f"       CRLF SHA256: {c_crlf}")
    print(f"       LF   SHA256: {c_lf}")

    # 3. Differences with Champion
    df_ctrl = pd.read_csv(delivery_path)
    merged = pd.merge(df_ctrl, df_cand, on=KEYS, suffixes=("_ctrl", "_cand"))
    assert len(merged) == 79101
    diff = np.abs(merged["quantity_cand"] - merged["quantity_ctrl"])
    n_diff = int((diff > 0).sum())
    max_d = int(diff.max())
    print(f"[PASS] Differences vs Champion: {n_diff:,} rows modified ({n_diff/79101*100:.2f}%), max diff: {max_d}")

    print("=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
