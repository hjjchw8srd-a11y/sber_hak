#!/usr/bin/env python3
"""
test_timestral_selected_features_submission.py

Verification test for Task 11 submission candidate:
submissions/submission_timestral_selected_features.csv
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

KEYS = ["week", "warehouse_name", "article"]
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_submission():
    sub_path = ROOT / "submissions" / "submission_timestral_selected_features.csv"
    ctrl_path = ROOT / "delivery_final" / "submission.csv"

    assert sub_path.exists(), f"Missing candidate submission: {sub_path}"
    assert ctrl_path.exists(), f"Missing control submission: {ctrl_path}"

    ctrl_sha = sha256_file(ctrl_path)
    assert ctrl_sha in (CONTROL_SHA256_CRLF, CONTROL_SHA256_LF), f"Control corrupted! SHA: {ctrl_sha}"

    df_sub = pd.read_csv(sub_path)
    df_ctrl = pd.read_csv(ctrl_path)

    assert len(df_sub) == 79101, f"Expected 79,101 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["week", "warehouse_name", "article", "quantity"]

    assert not df_sub.isna().any().any(), "Found NaNs in submission!"
    assert (df_sub["quantity"] >= 0).all(), "Found negative quantities!"
    assert np.issubdtype(df_sub["quantity"].dtype, np.integer), "Quantities are not integers!"

    # Key alignment
    merged = pd.merge(df_ctrl, df_sub, on=KEYS, suffixes=("_ctrl", "_sub"), how="inner", validate="one_to_one")
    assert len(merged) == 79101, "Key mismatch with control!"

    diff = np.abs(merged["quantity_sub"] - merged["quantity_ctrl"])
    n_changed = int((diff > 0).sum())
    max_diff = int(diff.max())

    print("=" * 80)
    print("TASK 11 SUBMISSION CANDIDATE AUDIT")
    print(f"File: {sub_path.name}")
    print(f"Rows: {len(df_sub):,}")
    print(f"Total quantity sum: {df_sub['quantity'].sum():,}")
    print(f"Changed rows vs Champion: {n_changed:,} ({n_changed / 79101 * 100:.2f}%)")
    print(f"Max absolute delta vs Champion: {max_diff}")
    print(f"Candidate SHA256: {sha256_file(sub_path)}")
    print("=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_submission()
