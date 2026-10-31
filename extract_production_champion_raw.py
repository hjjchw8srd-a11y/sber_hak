#!/usr/bin/env python3
"""
extract_production_champion_raw.py

Runs the full inference pipeline (TFT, LightGBM Dual, CatBoost, PerCategory)
to recover the exact continuous champion raw predictions.
Verifies:
1. raw_champion == 0.45*raw_tft + 0.37*raw_lgb + 0.09*raw_cb + 0.09*raw_pcat
2. apply_p(raw_champion, 1.05, 0.65) == delivery_final/submission.csv (diff = 0)
3. Saves to experiments/champion_production_raw/champion_raw_components.csv.gz
"""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "forecast-container"))

from app.predictor import predict_from_history

KEYS = ["week", "warehouse_name", "article"]
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("RECOVERING EXACT CHAMPION RAW VIA INFERENCE PIPELINE")
    print("=" * 80)

    # 1. Verify control delivery_final/submission.csv
    control_file = ROOT / "delivery_final" / "submission.csv"
    assert control_file.exists(), f"Control submission missing: {control_file}"
    raw_bytes = control_file.read_bytes()
    raw_sha = sha256_bytes(raw_bytes)
    norm_sha = sha256_bytes(raw_bytes.replace(b"\r\n", b"\n"))
    print(f"Control file: {control_file}")
    print(f"  Size: {len(raw_bytes):,} bytes | CRLF SHA256: {raw_sha} | LF SHA256: {norm_sha}")
    assert raw_sha == CONTROL_SHA256_CRLF or norm_sha == CONTROL_SHA256_LF, "Control file hash mismatch!"

    df_control = pd.read_csv(control_file)
    assert len(df_control) == 79101, f"Row count mismatch: {len(df_control)}"
    assert not df_control.duplicated(subset=KEYS).any(), "Duplicate keys in control!"

    # 2. Run Inference Pipeline
    sales_file = ROOT / "sales_dataset_hakaton_train.csv"
    stock_file = ROOT / "stock_dataset_hakaton_train.csv"
    bundle_dir = ROOT / "forecast-container" / "model"

    print("\nRunning predict_from_history with return_raw=True...")
    sales_df = pd.read_csv(sales_file)
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = sales_df[sales_df["date"] <= pd.Timestamp("2026-07-26")].copy()

    stock_df = pd.read_csv(stock_file)
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    stock_df = stock_df[stock_df["date"] <= pd.Timestamp("2026-07-26")].copy()

    pred_components = predict_from_history(sales_df, stock_df, bundle_dir=bundle_dir, return_raw=True)
    assert len(pred_components) == 79101, f"Inference row count mismatch: {len(pred_components)}"

    # 3. Mathematical Verification of Components Blend
    comp_tft = pred_components["raw_tft"].to_numpy(dtype=np.float64)
    comp_lgb = pred_components["raw_lgb"].to_numpy(dtype=np.float64)
    comp_cb = pred_components["raw_cb"].to_numpy(dtype=np.float64)
    comp_pcat = pred_components["raw_pcat"].to_numpy(dtype=np.float64)
    champ_raw = pred_components["raw_champion"].to_numpy(dtype=np.float64)

    expected_raw = 0.45 * comp_tft + 0.37 * comp_lgb + 0.09 * comp_cb + 0.09 * comp_pcat
    comp_diff = np.max(np.abs(champ_raw - expected_raw))
    print(f"\nComponent formulation check: max |raw_champion - (0.45*TFT + 0.37*LGB + 0.09*CB + 0.09*PCat)| = {comp_diff:.8e}")
    assert comp_diff < 1e-12, f"Component formula mismatch: {comp_diff}"

    # 4. Zero-Weight / Identity Test against Control
    reproduced_q = apply_p(champ_raw, 1.05, 0.65)
    merged_ctrl = pd.merge(
        df_control[KEYS + ["quantity"]],
        pred_components[KEYS].assign(reproduced_quantity=reproduced_q),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged_ctrl) == 79101, "Merge key alignment failed!"
    abs_delta = np.abs(merged_ctrl["quantity"].values - merged_ctrl["reproduced_quantity"].values)
    max_delta = int(abs_delta.max())
    diff_count = int((abs_delta != 0).sum())

    print(f"\nZero-weight / Control identity test:")
    print(f"  Rows compared: {len(merged_ctrl):,}")
    print(f"  Different rows: {diff_count}")
    print(f"  Max absolute difference: {max_delta}")
    assert max_delta == 0, f"Control mismatch! Max diff = {max_delta}, diff count = {diff_count}"
    print("  [PASS] 100% IDENTICAL to delivery_final/submission.csv with MAX DELTA = 0!")

    # 5. Export Champion Raw Components
    out_dir = ROOT / "experiments" / "champion_production_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "champion_raw_components.csv.gz"
    out_parquet = out_dir / "champion_raw_components.parquet"
    pred_components.to_csv(out_file, index=False, compression="gzip", float_format="%.15g")
    pred_components.to_parquet(out_parquet, index=False)
    sha_out = hashlib.sha256(out_file.read_bytes()).hexdigest()
    sha_parquet = hashlib.sha256(out_parquet.read_bytes()).hexdigest()
    print(f"\nSaved raw components: {out_file.name} ({out_file.stat().st_size:,} bytes | SHA256: {sha_out})")
    print(f"Saved raw parquet:    {out_parquet.name} ({out_parquet.stat().st_size:,} bytes | SHA256: {sha_parquet})")

    meta = {
        "status": "VERIFIED_EXACT",
        "rows": 79101,
        "max_abs_difference_vs_control": 0,
        "different_rows": 0,
        "control_crlf_sha256": raw_sha,
        "control_lf_sha256": norm_sha,
        "champion_raw_components_sha256": sha_out,
        "champion_raw_parquet_sha256": sha_parquet,
        "component_weights": {"tft": 0.45, "lightgbm": 0.37, "catboost": 0.09, "per_category": 0.09},
        "postprocessing": {"alpha": 1.05, "threshold": 0.65, "rounding": "rint", "clip_lower": 0},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 80)
    print("CHAMPION RAW EXTRACTION AND ZERO-WEIGHT VERIFICATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
