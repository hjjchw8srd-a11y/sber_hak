#!/usr/bin/env python3
"""
test_production_roundtrip_blend.py

Comprehensive production roundtrip and reload verification test:
1. Champion components -> raw_champion -> postprocessing -> control submission (max delta == 0)
2. Timestral pretrained + LoRA -> raw_timestral forecast parity
3. Raw blending for w=0.00, w=0.05, w=0.10:
   - w=0.00 matches delivery_final/submission.csv with max delta == 0
   - w=0.05 matches submission_champion_timestral_refit_blend_w05.csv with max delta == 0
   - w=0.10 matches submission_champion_timestral_refit_blend_w10.csv with max delta == 0
4. Reload determinism: fresh base model reload + adapter re-attach -> exact candidate match (0 different rows)
"""

from __future__ import annotations

import sys
import json
import math
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

KEYS = ["week", "warehouse_name", "article"]
HORIZON = 9
CONTEXT = 52
CONTROL_SHA256_CRLF = "09cd97d4dc930a5280bb93ae249f01ded35d755fff9b276a35e7473359022939"
CONTROL_SHA256_LF = "ef658d22e8f8f685c6d9d0041c18854a06a900f9ef014368c0fddd70006647d0"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None).astype(int)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 90)
    print("PRODUCTION TIMESTRAL BLEND ROUNDTRIP & RELOAD DETERMINISM TEST")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # STAGE 1: Champion Components -> Raw Champion -> Postprocessing -> Control
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Testing Champion Components -> Raw Champion -> Control Submission...")
    control_path = ROOT / "delivery_final" / "submission.csv"
    assert control_path.exists(), f"Control submission missing: {control_path}"
    ctrl_bytes = control_path.read_bytes()
    crlf_sha = sha256_bytes(ctrl_bytes)
    lf_sha = sha256_bytes(ctrl_bytes.replace(b"\r\n", b"\n"))
    assert crlf_sha == CONTROL_SHA256_CRLF or lf_sha == CONTROL_SHA256_LF, "Control file SHA256 mismatch!"
    print(f"  Control submission verified: {control_path.name} (CRLF SHA: {crlf_sha[:16]}..., LF SHA: {lf_sha[:16]}...)")

    raw_comp_path = ROOT / "experiments" / "champion_production_raw" / "champion_raw_components.csv.gz"
    assert raw_comp_path.exists(), f"Champion raw components file missing: {raw_comp_path}"
    df_comp = pd.read_csv(raw_comp_path)
    assert len(df_comp) == 79101, f"Expected 79,101 rows, got {len(df_comp)}"

    comp_tft = df_comp["raw_tft"].to_numpy(dtype=np.float64)
    comp_lgb = df_comp["raw_lgb"].to_numpy(dtype=np.float64)
    comp_cb = df_comp["raw_cb"].to_numpy(dtype=np.float64)
    comp_pcat = df_comp["raw_pcat"].to_numpy(dtype=np.float64)
    raw_champ = df_comp["raw_champion"].to_numpy(dtype=np.float64)

    # Check component linear combination
    expected_champ = 0.45 * comp_tft + 0.37 * comp_lgb + 0.09 * comp_cb + 0.09 * comp_pcat
    comp_formula_delta = np.max(np.abs(raw_champ - expected_champ))
    print(f"  Component combination check: max |raw_champion - sum(w_i * comp_i)| = {comp_formula_delta:.8e}")
    assert comp_formula_delta < 1e-4, f"Component formula delta {comp_formula_delta} exceeds threshold!"

    # Check single postprocessing vs control submission
    df_ctrl = pd.read_csv(control_path)
    reproduced_ctrl_q = apply_p(raw_champ, 1.05, 0.65)
    merged_ctrl = pd.merge(
        df_ctrl[KEYS + ["quantity"]],
        df_comp[KEYS].assign(reproduced_quantity=reproduced_ctrl_q),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    diff_ctrl = np.abs(merged_ctrl["quantity"].values - merged_ctrl["reproduced_quantity"].values)
    max_ctrl_delta = int(diff_ctrl.max())
    diff_ctrl_count = int((diff_ctrl != 0).sum())
    print(f"  Stage 1 Control Match: different rows = {diff_ctrl_count}, max abs delta = {max_ctrl_delta}")
    assert max_ctrl_delta == 0, f"Stage 1 failed: discrepancy vs control submission! Max delta = {max_ctrl_delta}"
    print("  [PASS] Stage 1 PASSED with exact zero discrepancy (max delta = 0).")

    # -------------------------------------------------------------------------
    # STAGE 2: Timestral Pretrained + LoRA -> Raw Timestral Parity
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Testing Timestral LoRA Forecast Parity against Saved Future Predictions...")
    prod_dir = ROOT / "experiments" / "timestral_production_model" / "refit_full_history"
    ckpt_dir = prod_dir / "checkpoints" / "final_refit"
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    future_file = prod_dir / "timestral_209m_future_9w_keyed.csv.gz"
    assert adapter_file.exists(), f"Adapter missing: {adapter_file}"
    assert future_file.exists(), f"Future keyed file missing: {future_file}"

    df_future = pd.read_csv(future_file)
    assert len(df_future) == 79101, f"Expected 79,101 rows, got {len(df_future)}"
    print(f"  Saved future forecast: {future_file.name} (SHA: {sha256_file(future_file)[:16]}...)")
    print("  [PASS] Stage 2 future prediction file exists and has 79,101 rows.")

    # -------------------------------------------------------------------------
    # STAGE 3: Raw Blending & Postprocessing for w=0.00, w=0.05, w=0.10
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Testing Raw Blending & Candidate Files...")
    merged_comp = pd.merge(
        df_comp[KEYS + ["raw_champion"]],
        df_future[KEYS + ["y_pred_continuous"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged_comp) == 79101

    raw_c = merged_comp["raw_champion"].values.astype(np.float64)
    raw_t = merged_comp["y_pred_continuous"].values.astype(np.float64)

    # 3.1: Zero weight w = 0.00
    q_w0 = apply_p((1.0 - 0.0) * raw_c + 0.0 * raw_t, 1.05, 0.65)
    delta_w0 = np.abs(df_ctrl["quantity"].values - q_w0)
    print(f"  Zero-weight test (w=0.00): max delta = {int(delta_w0.max())}, different rows = {int((delta_w0 != 0).sum())}")
    assert int(delta_w0.max()) == 0, f"Zero-weight test failed! Max delta = {int(delta_w0.max())}"
    print("  [PASS] Zero-weight test passed with 100% bit-exact match to delivery_final/submission.csv!")

    # 3.2: w = 0.05
    cand_w05_path = ROOT / "submissions" / "submission_champion_timestral_refit_blend_w05.csv"
    assert cand_w05_path.exists(), f"Candidate w05 missing: {cand_w05_path}"
    df_w05 = pd.read_csv(cand_w05_path)
    assert len(df_w05) == 79101
    q_w05_calc = apply_p(0.95 * raw_c + 0.05 * raw_t, 1.05, 0.65)
    delta_w05 = np.abs(df_w05["quantity"].values - q_w05_calc)
    print(f"  Candidate w05 match: max delta = {int(delta_w05.max())}, different rows = {int((delta_w05 != 0).sum())}")
    assert int(delta_w05.max()) == 0, f"Candidate w05 mismatch! Max delta = {int(delta_w05.max())}"
    print("  [PASS] Candidate w05 bit-exact match verified!")

    # 3.3: w = 0.10
    cand_w10_path = ROOT / "submissions" / "submission_champion_timestral_refit_blend_w10.csv"
    assert cand_w10_path.exists(), f"Candidate w10 missing: {cand_w10_path}"
    df_w10 = pd.read_csv(cand_w10_path)
    assert len(df_w10) == 79101
    q_w10_calc = apply_p(0.90 * raw_c + 0.10 * raw_t, 1.05, 0.65)
    delta_w10 = np.abs(df_w10["quantity"].values - q_w10_calc)
    print(f"  Candidate w10 match: max delta = {int(delta_w10.max())}, different rows = {int((delta_w10 != 0).sum())}")
    assert int(delta_w10.max()) == 0, f"Candidate w10 mismatch! Max delta = {int(delta_w10.max())}"
    print("  [PASS] Candidate w10 bit-exact match verified!")

    # -------------------------------------------------------------------------
    # STAGE 4: Reload Determinism & Independence Test
    # -------------------------------------------------------------------------
    print("\n[STAGE 4] Testing Reload Determinism with Fresh Model Instance...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"

    fresh_forecaster = TimestralForecaster.from_pretrained(str(model_dir))
    fresh_forecaster.model = PeftModel.from_pretrained(fresh_forecaster.model, str(ckpt_dir))
    fresh_forecaster.model.to(device)
    fresh_forecaster.model.eval()

    # Load historical panel context (weeks 82..133)
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    df_piv = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0)
    panel = df_piv.values.astype(np.float32)
    n_series = panel.shape[1]

    inf_ctx = panel[-CONTEXT:, :].T  # (8789, 52)
    inf_batch_size = 64
    reloaded_preds_list = []

    print(f"  Running inference with freshly reloaded adapter on {n_series} series...")
    with torch.no_grad():
        for i in range(0, n_series, inf_batch_size):
            b_ctx = torch.as_tensor(inf_ctx[i : i + inf_batch_size], dtype=torch.float32, device=device)
            fc = fresh_forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            med = fc.median[:, 0, :].detach().cpu().numpy()
            reloaded_preds_list.append(med)

    reloaded_continuous = np.vstack(reloaded_preds_list)  # (8789, 9)

    # Compare reloaded continuous forecast with saved future continuous forecast
    saved_pivot = df_future.pivot(index=["warehouse_name", "article"], columns="week", values="y_pred_continuous")
    saved_arr = saved_pivot.values
    reload_diff = np.max(np.abs(saved_arr - reloaded_continuous))
    print(f"  Reload forecast discrepancy vs saved CSV: max diff = {reload_diff:.8e}")
    assert reload_diff <= 1e-4, f"Reload parity failure: {reload_diff} > 1e-4"

    # Verify that reloaded predictions blended with raw_champion reproduce candidates 100% bit-exact
    reloaded_raw_tim = reloaded_continuous.T.ravel()  # Order: week-first, then warehouse/article
    reloaded_q_w05 = apply_p(0.95 * raw_c + 0.05 * reloaded_raw_tim, 1.05, 0.65)
    reloaded_q_w10 = apply_p(0.90 * raw_c + 0.10 * reloaded_raw_tim, 1.05, 0.65)

    delta_reload_w05 = np.abs(df_w05["quantity"].values - reloaded_q_w05)
    delta_reload_w10 = np.abs(df_w10["quantity"].values - reloaded_q_w10)
    print(f"  Reloaded Candidate w05 vs disk: different rows = {int((delta_reload_w05 != 0).sum())}, max delta = {int(delta_reload_w05.max())}")
    print(f"  Reloaded Candidate w10 vs disk: different rows = {int((delta_reload_w10 != 0).sum())}, max delta = {int(delta_reload_w10.max())}")
    assert int(delta_reload_w05.max()) == 0, "Reloaded w05 discrepancy!"
    assert int(delta_reload_w10.max()) == 0, "Reloaded w10 discrepancy!"
    print("  [PASS] Full roundtrip reload determinism verified: 100% bit-exact equality (0 different rows)!")

    print("\n" + "=" * 90)
    print("ALL ROUNDTRIP, BLENDING AND RELOAD VERIFICATIONS PASSED WITH ZERO ERRORS!")
    print("=" * 90)


if __name__ == "__main__":
    main()
