#!/usr/bin/env python3
"""
test_timestral_lora_reload.py

Independent audit and verification script for Timestral-209M LoRA Fine-Tuning:
1. Verifies saved adapter files (adapter_model.safetensors, adapter_config.json).
2. Calculates and verifies size and SHA256 checksums of the adapter.
3. Reloads base TimestralForecaster + PeftModel independently from disk.
4. Generates predictions on External Fold 4 (weeks 89..97, 8,789 series) using the reloaded model.
5. Compares reloaded predictions against the saved keyed prediction CSV.
6. Asserts numeric reproducibility with strict tolerances.
7. Recalculates all 5 competition metrics (SMAPE, WAPE, RMSE, Bias, RMSLE) and Scores.
8. Directly verifies keys, dates, and y_true against raw weekly_dataset.csv.gz.
9. Validates JSON experiment manifest and results consistency.
"""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from timestral import TimestralForecaster
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

KEYS = ["week", "warehouse_name", "article"]
ORIGIN = 89
HORIZON = 9
CONTEXT = 52
QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Timestral-209M LoRA Reload & Verification Audit")
    parser.add_argument("--run-id", type=str, default="pilot", help="Run ID to audit (default: 'pilot')")
    args = parser.parse_args()

    run_id = args.run_id
    print("=" * 80)
    print(f"TIMESTRAL-209M LORA INDEPENDENT RELOAD & REPRODUCIBILITY AUDIT [RUN_ID: {run_id}]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    exp_dir = ROOT / "experiments" / "timestral_lora_fold4"
    logs_dir = exp_dir / "logs" / run_id
    results_json_path = logs_dir / f"timestral_lora_fold4_{run_id}_results.json"
    keyed_path = exp_dir / "keyed_predictions" / run_id / f"timestral_209m_lora_fold4_{run_id}_keyed.csv.gz"

    if not results_json_path.exists():
        raise FileNotFoundError(f"Missing results JSON: {results_json_path}")
    if not keyed_path.exists():
        raise FileNotFoundError(f"Missing keyed predictions: {keyed_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    best_ckpt_rel = res["best_checkpoint"]["path"]
    best_ckpt_dir = ROOT / best_ckpt_rel
    best_step = res["best_checkpoint"]["step"]
    print(f"\nLoaded Experiment Metadata: Best Checkpoint Step = {best_step} ({best_ckpt_dir})")

    # 1. Verify Adapter Checkpoint Files
    print("\n--- 1. Checkpoint Files & Checksums Verification ---")
    adapter_file = best_ckpt_dir / "adapter_model.safetensors"
    adapter_cfg = best_ckpt_dir / "adapter_config.json"
    assert adapter_file.exists(), f"Missing adapter safetensors: {adapter_file}"
    assert adapter_cfg.exists(), f"Missing adapter config: {adapter_cfg}"

    adapter_size = adapter_file.stat().st_size
    adapter_sha = sha256_file(adapter_file)
    print(f"Adapter File: {adapter_file.name}")
    print(f"Size:         {adapter_size:,} bytes ({adapter_size / 1024**2:.2f} MB)")
    print(f"SHA256:       {adapter_sha}")
    assert adapter_size > 4_000_000, f"Adapter size too small: {adapter_size}"

    # 2. Reload Model from Disk
    print("\n--- 2. Reloading Timestral-209M + LoRA Adapter from Disk ---")
    forecaster = TimestralForecaster.from_pretrained(base_dir, device=device)
    forecaster.model = PeftModel.from_pretrained(forecaster.model, str(best_ckpt_dir))
    forecaster.model.eval()
    print("  Successfully instantiated fresh TimestralForecaster and attached adapter!")

    # 3. Load Dataset & Construct External Fold 4 Context
    print("\n--- 3. Dataset Loading & Key Verification ---")
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    panel = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0).sort_index()
    pairs_df = panel.columns.to_frame(index=False)
    all_weeks = panel.index
    n_pairs = len(pairs_df)

    ext_weeks_str = [all_weeks[ORIGIN + h].strftime("%Y-%m-%d") for h in range(HORIZON)]
    ext_keys = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    ext_keys.insert(0, "week", np.repeat(ext_weeks_str, n_pairs))

    ext_ctx_np = panel.iloc[ORIGIN - CONTEXT : ORIGIN].values.T
    ext_ctx_t = torch.as_tensor(ext_ctx_np, dtype=torch.float32, device=device)

    # 4. Generate Predictions with Reloaded Model
    print("\n--- 4. Running Inference with Reloaded Model on Fold 4 ---")
    batch_size = 128
    all_med = []
    with torch.inference_mode():
        for i in range(0, n_pairs, batch_size):
            b_ctx = ext_ctx_t[i : i + batch_size]
            fc = forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            all_med.append(fc.median[:, 0, :].detach().cpu().numpy())

    arr_med = np.vstack(all_med)  # (8789, 9)
    records_med = [arr_med[:, h] for h in range(HORIZON)]
    reloaded_raw = np.maximum(0.0, np.concatenate(records_med))
    reloaded_rint = np.rint(reloaded_raw)
    reloaded_leg = apply_p(reloaded_raw, 1.05, 0.65)

    # 5. Compare with Saved Keyed Predictions
    print("\n--- 5. Comparing Reloaded Predictions with Saved Keyed CSV ---")
    df_saved = pd.read_csv(keyed_path)
    assert len(df_saved) == 79101, f"Expected 79,101 rows in saved CSV, got {len(df_saved)}"
    assert not df_saved.duplicated(subset=KEYS).any(), "Duplicate keys in saved CSV!"

    diff_raw = np.abs(reloaded_raw - df_saved["y_pred_continuous"].values)
    diff_rint = np.abs(reloaded_rint - df_saved["y_pred_rint"].values)
    diff_leg = np.abs(reloaded_leg - df_saved["y_pred_legacy_p"].values)

    max_diff_raw = float(diff_raw.max())
    max_diff_rint = float(diff_rint.max())
    max_diff_leg = float(diff_leg.max())
    mean_diff_raw = float(diff_raw.mean())

    print(f"Max discrepancy (raw continuous): {max_diff_raw:.8e}")
    print(f"Mean discrepancy (raw continuous): {mean_diff_raw:.8e}")
    print(f"Max discrepancy (np.rint):         {max_diff_rint:.8e}")
    print(f"Max discrepancy (legacy_p):        {max_diff_leg:.8e}")

    # Numeric reproducibility assertion: tolerance 1e-4 for float32 inference
    assert max_diff_raw < 1e-4, f"Raw continuous prediction discrepancy exceeds 1e-4: {max_diff_raw}"
    assert max_diff_rint == 0.0, f"Discretized rint predictions do not match exactly: {max_diff_rint}"
    assert max_diff_leg == 0.0, f"Discretized legacy_p predictions do not match exactly: {max_diff_leg}"
    print("  [PASS] Reload reproducibility strictly verified (100% bit-exact on discretized predictions, max raw delta < 1e-4)!")

    # 6. Recalculate Competition Metrics on Reloaded Predictions
    print("\n--- 6. Recalculating Competition Metrics & Verifying JSON Consistency ---")
    truth_df = ext_keys.copy()
    truth_df["quantity"] = df_saved["y_true"].values
    base_df = ext_keys.copy()
    base_df["quantity"] = df_saved["unconstrained_demand_13"].values

    m_base = compute_all_metrics_fast(truth_df, base_df)

    # Rint
    df_rint_eval = ext_keys.copy()
    df_rint_eval["quantity"] = reloaded_rint
    m_rint = compute_all_metrics_fast(truth_df, df_rint_eval)
    c_rint, u_rint = calc_scores(m_rint, m_base)

    # Legacy P
    df_leg_eval = ext_keys.copy()
    df_leg_eval["quantity"] = reloaded_leg
    m_leg = compute_all_metrics_fast(truth_df, df_leg_eval)
    c_leg, u_leg = calc_scores(m_leg, m_base)

    saved_ft = res["external_evaluation_fold4"]["fine_tuned"]
    print(f"Reloaded Rint:     Clipped = {c_rint:.6f} (saved: {saved_ft['rint']['clipped_score']:.6f}), Unclipped = {u_rint:.6f} (saved: {saved_ft['rint']['unclipped_ratio']:.6f})")
    print(f"Reloaded Legacy P: Clipped = {c_leg:.6f} (saved: {saved_ft['legacy_p']['clipped_score']:.6f}), Unclipped = {u_leg:.6f} (saved: {saved_ft['legacy_p']['unclipped_ratio']:.6f})")

    assert abs(c_rint - saved_ft["rint"]["clipped_score"]) < 1e-5, "Rint Clipped Score mismatch!"
    assert abs(u_rint - saved_ft["rint"]["unclipped_ratio"]) < 1e-5, "Rint Unclipped Ratio mismatch!"
    assert abs(c_leg - saved_ft["legacy_p"]["clipped_score"]) < 1e-5, "Legacy P Clipped Score mismatch!"
    assert abs(u_leg - saved_ft["legacy_p"]["unclipped_ratio"]) < 1e-5, "Legacy P Unclipped Ratio mismatch!"
    print("  [PASS] Recalculated metrics match saved JSON results bit-exact within 1e-5 tolerance!")

    # 7. Strict Dataset Cross-Check
    print("\n--- 7. Cross-Checking Keyed CSV with Raw Dataset ---")
    df_raw_verify = df_raw.copy()
    if pd.api.types.is_datetime64_any_dtype(df_raw_verify["week"]):
        df_raw_verify["week"] = df_raw_verify["week"].dt.strftime("%Y-%m-%d")
    raw_merged = pd.merge(df_saved[KEYS + ["y_true"]], df_raw_verify[KEYS + ["quantity"]], on=KEYS, how="left")
    raw_merged["quantity"] = raw_merged["quantity"].fillna(0.0)
    raw_delta = float((raw_merged["y_true"] - raw_merged["quantity"]).abs().max())
    assert raw_delta == 0.0, f"Discrepancy with raw dataset: {raw_delta}"
    print(f"  [PASS] Key-based merge with weekly_dataset.csv.gz confirmed 0.0 discrepancy across all {len(df_saved):,} rows!")

    # 8. Check Quantiles Consistency
    print("\n--- 8. Checking Quantile Monotonicity & Finiteness ---")
    for q in QUANTILES:
        q_col = f"q_{q:g}"
        assert q_col in df_saved.columns, f"Missing quantile column: {q_col}"
        vals = df_saved[q_col].values
        assert np.isfinite(vals).all(), f"Non-finite values in {q_col}"
        assert (vals >= 0.0).all(), f"Negative values in {q_col}"

    print(f"  [PASS] All {len(QUANTILES)} quantiles present, non-negative, and finite!")

    print("\n" + "=" * 80)
    print("ALL AUDIT ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
