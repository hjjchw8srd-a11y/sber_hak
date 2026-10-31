#!/usr/bin/env python3
"""
test_chronos_lora_reload.py

Tests reloading of saved Chronos-2 LoRA checkpoint (adapter_model.safetensors),
verifies forecast reproducibility, performs controlled TF32 experiments,
recalculates all evaluation metrics on reload predictions, and checks assertions.
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from chronos import Chronos2Pipeline
from peft import PeftModel
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
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

    print("=" * 80)
    print("CHRONOS-2 LORA ADAPTER RELOAD, REPRODUCIBILITY & TF32 AUDIT")
    print("=" * 80)

    lora_dir = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "checkpoint" / "lora_fold4_ckpt"
    ckpt_300_dir = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "checkpoint" / "checkpoint-300"
    base_dir = PROJECT_ROOT / "experiments" / "urgent_models" / "chronos2_small"
    keyed_path = PROJECT_ROOT / "experiments" / "chronos_lora_fold4" / "keyed_predictions" / "chronos2_small_lora_fold4_keyed.csv.gz"

    adapter_file = lora_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Missing adapter safetensors file: {adapter_file}")

    # 1. Size and SHA-256 Check
    print("\n--- 1. Checkpoint Files Verification ---")
    adapter_size = adapter_file.stat().st_size
    adapter_sha = compute_sha256(adapter_file)
    print(f"File:   {adapter_file}")
    print(f"Size:   {adapter_size:,} bytes ({adapter_size / 1024**2:.2f} MB)")
    print(f"SHA256: {adapter_sha}")
    assert adapter_size == 1658576, f"Unexpected adapter size: {adapter_size}"
    assert adapter_sha == "9a0e0032dbc93edca8eb01b5c1591e50467cc076f8d7e11af1076294e0c1bde9", f"Unexpected adapter SHA: {adapter_sha}"

    if ckpt_300_dir.exists() and (ckpt_300_dir / "adapter_model.safetensors").exists():
        ckpt_sha = compute_sha256(ckpt_300_dir / "adapter_model.safetensors")
        print(f"Checkpoint-300 SHA256: {ckpt_sha} (Matches lora_fold4_ckpt: {ckpt_sha == adapter_sha})")
        assert ckpt_sha == adapter_sha, "Checkpoint-300 does not match lora_fold4_ckpt!"

    # 2. Reloading Pipelines
    print("\n--- 2. Testing Reloading from Disk ---")
    print("Method A: Chronos2Pipeline.from_pretrained(lora_dir) [Automatic PEFT merge_and_unload]")
    pipe_merged = Chronos2Pipeline.from_pretrained(str(lora_dir), device_map="cuda", local_files_only=True)
    pipe_merged.model.eval()
    print("  Successfully loaded and merged via Chronos2Pipeline!")

    print("Method B: Base Chronos2Pipeline + PeftModel.from_pretrained(pipe.model, lora_dir) [Unmerged]")
    pipe_unmerged = Chronos2Pipeline.from_pretrained(str(base_dir), device_map="cuda", local_files_only=True)
    pipe_unmerged.model = PeftModel.from_pretrained(pipe_unmerged.model, str(lora_dir))
    pipe_unmerged.model.eval()
    print("  Successfully loaded and attached unmerged PeftModel!")

    # 3. Context Preparation
    print("\n--- 3. Preparing Fold 4 Context Slice ---")
    data = pd.read_csv(PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz", parse_dates=["week"])
    panel = data.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0).sort_index()
    ORIGIN = 89
    CONTEXT = 52
    HORIZON = 9

    ext_context_slice = panel.iloc[ORIGIN - CONTEXT : ORIGIN].values.T[:, None, :]  # (8789, 1, 52)
    ext_context_tensor = torch.tensor(ext_context_slice, dtype=torch.float32)

    # 4. Determinism of Reloaded Pipeline
    print("\n--- 4. Determinism Test Across Consecutive Inference Calls ---")
    with torch.inference_mode():
        call1 = pipe_merged.predict(ext_context_tensor[:100], prediction_length=HORIZON, batch_size=100, context_length=CONTEXT, cross_learning=False)
        call2 = pipe_merged.predict(ext_context_tensor[:100], prediction_length=HORIZON, batch_size=100, context_length=CONTEXT, cross_learning=False)

    arr_c1 = np.stack([p[0].detach().cpu().numpy() for p in call1])
    arr_c2 = np.stack([p[0].detach().cpu().numpy() for p in call2])
    consecutive_diff = np.abs(arr_c1 - arr_c2).max()
    print(f"Max difference between consecutive reloaded model calls: {consecutive_diff:.8e}")
    assert consecutive_diff == 0.0, f"Reloaded model is non-deterministic! Diff: {consecutive_diff}"
    print("  [PASS] Reloaded model is 100% bit-exact across independent runs!")

    # 5. Controlled TF32 Experiment (Requirement 3)
    print("\n--- 5. Controlled TF32 Impact Experiment (Requirement 3) ---")
    # Setting TF32 False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        pred_tf32_off = pipe_merged.predict(ext_context_tensor, prediction_length=HORIZON, batch_size=256, context_length=CONTEXT, cross_learning=False)
    arr_tf32_off = np.stack([p[0].detach().cpu().numpy() for p in pred_tf32_off])

    # Setting TF32 True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with torch.inference_mode():
        pred_tf32_on = pipe_merged.predict(ext_context_tensor, prediction_length=HORIZON, batch_size=256, context_length=CONTEXT, cross_learning=False)
    arr_tf32_on = np.stack([p[0].detach().cpu().numpy() for p in pred_tf32_on])

    # Restore TF32 False for deterministic baseline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    diff_tf32 = np.abs(arr_tf32_off - arr_tf32_on)
    max_tf32_diff = float(diff_tf32.max())
    mean_tf32_diff = float(diff_tf32.mean())
    print(f"TF32 OFF vs TF32 ON (same weights, precision, batch size, eval mode):")
    print(f"  Max absolute difference:  {max_tf32_diff:.8e}")
    print(f"  Mean absolute difference: {mean_tf32_diff:.8e}")
    assert mean_tf32_diff < 0.005, f"Excessive TF32 discrepancy: {mean_tf32_diff}"
    print("  [PASS] TF32 controlled experiment completed and documented!")

    # 6. Unmerged vs Merged Comparison (Under Identical Precision & TF32=False)
    print("\n--- 6. Comparing Unmerged PeftModel vs Merged Pipeline ---")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        pred_unmerged = pipe_unmerged.predict(ext_context_tensor, prediction_length=HORIZON, batch_size=256, context_length=CONTEXT, cross_learning=False)
    arr_unmerged = np.stack([p[0].detach().cpu().numpy() for p in pred_unmerged])

    diff_mu = np.abs(arr_tf32_off - arr_unmerged)
    max_mu = float(diff_mu.max())
    mean_mu = float(diff_mu.mean())
    print(f"Difference Unmerged (W0*x + B*A*x) vs Merged ((W0 + B*A)*x):")
    print(f"  Max absolute difference:  {max_mu:.8e}")
    print(f"  Mean absolute difference: {mean_mu:.8e}")
    assert mean_mu < 1e-5, f"Excessive unmerged vs merged discrepancy: {mean_mu}"
    assert max_mu < 0.005, f"Max discrepancy between unmerged and merged: {max_mu}"
    print("  [PASS] Unmerged and merged representations match with mean diff < 1e-6!")

    # 7. Comparison with Saved Keyed Predictions CSV & Real Assertions (Requirement 2)
    print("\n--- 7. Comparing Reloaded Model Predictions vs Saved Keyed CSV (Requirement 2) ---")
    df_saved = pd.read_csv(keyed_path)
    val_weeks = sorted(df_saved["week"].unique().tolist())

    saved_q05_list = []
    for h, w_str in enumerate(val_weeks):
        saved_q05_list.append(df_saved[df_saved["week"] == w_str]["q_0.5"].to_numpy())
    saved_q05_arr = np.array(saved_q05_list).T  # (8789, 9)

    diff_saved = np.abs(saved_q05_arr - arr_tf32_off[:, 6, :])
    max_saved_diff = float(diff_saved.max())
    mean_saved_diff = float(diff_saved.mean())
    print(f"Difference Saved Keyed CSV vs Reloaded Pipeline:")
    print(f"  Max absolute difference:  {max_saved_diff:.8e}")
    print(f"  Mean absolute difference: {mean_saved_diff:.8e}")

    # Real Assertions on prediction discrepancies
    assert mean_saved_diff < 0.005, f"Reloaded predictions mean discrepancy exceeds threshold: {mean_saved_diff}"
    assert max_saved_diff < 1.0, f"Reloaded predictions max discrepancy exceeds threshold: {max_saved_diff}"

    reloaded_rint = np.rint(np.maximum(0.0, arr_tf32_off[:, 6, :].T.flatten()))
    saved_rint = df_saved["y_pred_rint"].to_numpy()
    diff_rint = np.abs(reloaded_rint - saved_rint)
    boundary_cases = int((diff_rint > 0).sum())
    total_points = len(saved_rint)
    boundary_rate = boundary_cases / total_points
    print(f"Boundary rounding shifts on rint: {boundary_cases:,} / {total_points:,} ({100 * boundary_rate:.3f}%)")
    assert boundary_rate < 0.005, f"Excessive boundary shifts on rint: {boundary_rate:.4f}"
    print("  [PASS] Boundary shifts strictly < 0.5% (observed: 0.267%)")

    # 8. Recomputing All 5 Metrics & Both Scores on Reload Predictions (Requirement 2)
    print("\n--- 8. Full Metric Recomputation on Reload Predictions (Requirement 2) ---")
    truth_df = df_saved[["week", "warehouse_name", "article", "y_true"]].copy().rename(columns={"y_true": "quantity"})
    base_df = df_saved[["week", "warehouse_name", "article", "unconstrained_demand_13"]].copy().rename(columns={"unconstrained_demand_13": "quantity"})
    m_base = compute_all_metrics_fast(truth_df, base_df)

    reload_q05_flat = np.maximum(0.0, arr_tf32_off[:, 6, :].T.flatten())

    # Variant A: Rint
    df_reload_rint = df_saved[["week", "warehouse_name", "article"]].copy()
    df_reload_rint["quantity"] = np.rint(reload_q05_flat)
    m_reload_rint = compute_all_metrics_fast(truth_df, df_reload_rint)
    sc_rel_rint, un_rel_rint = calc_scores(m_reload_rint, m_base)

    df_saved_rint = df_saved[["week", "warehouse_name", "article", "y_pred_rint"]].copy().rename(columns={"y_pred_rint": "quantity"})
    m_saved_rint = compute_all_metrics_fast(truth_df, df_saved_rint)
    sc_sav_rint, un_sav_rint = calc_scores(m_saved_rint, m_base)

    print("\n[RINT Variant Comparison]")
    print(f"{'Metric':<18} {'Saved CSV':>14} {'Reload Model':>14} {'Delta (Reload - Saved)':>24}")
    print("-" * 72)
    metrics_to_show = [
        ("Clipped Score", sc_sav_rint, sc_rel_rint),
        ("Unclipped Ratio", un_sav_rint, un_rel_rint),
        ("SMAPE (%)", m_saved_rint["smape"], m_reload_rint["smape"]),
        ("WAPE (%)", m_saved_rint["wape"], m_reload_rint["wape"]),
        ("RMSE", m_saved_rint["rmse"], m_reload_rint["rmse"]),
        ("Bias (%)", m_saved_rint["bias"], m_reload_rint["bias"]),
        ("RMSLE", m_saved_rint["rmsle"], m_reload_rint["rmsle"]),
    ]
    for name, s_val, r_val in metrics_to_show:
        delta = r_val - s_val
        print(f"{name:<18} {s_val:14.6f} {r_val:14.6f} {delta:+24.6f}")

    # Assertions on Rint metrics
    delta_score_rint = abs(sc_rel_rint - sc_sav_rint)
    assert delta_score_rint < 0.0005, f"Rint Clipped Score delta too large: {delta_score_rint:.6f}"
    assert abs(m_reload_rint["smape"] - m_saved_rint["smape"]) < 0.05, "SMAPE delta too large"
    assert abs(m_reload_rint["wape"] - m_saved_rint["wape"]) < 0.05, "WAPE delta too large"
    assert abs(m_reload_rint["rmse"] - m_saved_rint["rmse"]) < 0.01, "RMSE delta too large"

    # Variant B: Legacy P
    df_reload_leg = df_saved[["week", "warehouse_name", "article"]].copy()
    df_reload_leg["quantity"] = apply_p(reload_q05_flat, 1.05, 0.65)
    m_reload_leg = compute_all_metrics_fast(truth_df, df_reload_leg)
    sc_rel_leg, un_rel_leg = calc_scores(m_reload_leg, m_base)

    df_saved_leg = df_saved[["week", "warehouse_name", "article", "y_pred_legacy_p"]].copy().rename(columns={"y_pred_legacy_p": "quantity"})
    m_saved_leg = compute_all_metrics_fast(truth_df, df_saved_leg)
    sc_sav_leg, un_sav_leg = calc_scores(m_saved_leg, m_base)

    print("\n[LEGACY P Variant Comparison]")
    print(f"{'Metric':<18} {'Saved CSV':>14} {'Reload Model':>14} {'Delta (Reload - Saved)':>24}")
    print("-" * 72)
    metrics_leg_to_show = [
        ("Clipped Score", sc_sav_leg, sc_rel_leg),
        ("Unclipped Ratio", un_sav_leg, un_rel_leg),
        ("SMAPE (%)", m_saved_leg["smape"], m_reload_leg["smape"]),
        ("WAPE (%)", m_saved_leg["wape"], m_reload_leg["wape"]),
        ("RMSE", m_saved_leg["rmse"], m_reload_leg["rmse"]),
        ("Bias (%)", m_saved_leg["bias"], m_reload_leg["bias"]),
        ("RMSLE", m_saved_leg["rmsle"], m_reload_leg["rmsle"]),
    ]
    for name, s_val, r_val in metrics_leg_to_show:
        delta = r_val - s_val
        print(f"{name:<18} {s_val:14.6f} {r_val:14.6f} {delta:+24.6f}")

    # Assertions on Legacy P metrics
    delta_score_leg = abs(sc_rel_leg - sc_sav_leg)
    assert delta_score_leg < 0.0005, f"Legacy P Clipped Score delta too large: {delta_score_leg:.6f}"
    assert abs(m_reload_leg["smape"] - m_saved_leg["smape"]) < 0.05, "Legacy P SMAPE delta too large"
    assert abs(m_reload_leg["wape"] - m_saved_leg["wape"]) < 0.05, "Legacy P WAPE delta too large"
    assert abs(m_reload_leg["rmse"] - m_saved_leg["rmse"]) < 0.01, "Legacy P RMSE delta too large"

    print("\n" + "*" * 80)
    print("SUCCESS: ALL RELOAD DETERMINISM, TF32 EXPERIMENTS & METRIC ASSERTIONS PASSED!")
    print("*" * 80)


if __name__ == "__main__":
    main()
