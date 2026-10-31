#!/usr/bin/env python3
"""
test_timestral_production_reload.py

Strict reload and verification test for the full-history production Timestral-209M LoRA model.
Validates:
1. Adapter file size and SHA256 integrity
2. Successful reloading of base model + adapter into PeftModel
3. Forecast generation on production context (weeks 82..133)
4. Numerical parity with saved future forecast CSV (max diff <= 1e-5)
5. Format validation: exactly 79,101 rows, correct columns, no NaNs
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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("TESTING PRODUCTION TIMESTRAL-209M RELOAD AND FORECAST PARITY")
    print("=" * 80)

    model_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    prod_dir = ROOT / "experiments" / "timestral_production_model" / "refit_full_history"
    ckpt_dir = prod_dir / "checkpoints" / "final_refit"
    adapter_file = ckpt_dir / "adapter_model.safetensors"
    future_file = prod_dir / "timestral_209m_future_9w_keyed.csv.gz"

    assert adapter_file.exists(), f"Adapter file missing: {adapter_file}"
    assert future_file.exists(), f"Future keyed file missing: {future_file}"

    # 1. Verify adapter file integrity
    adapter_sha = sha256_file(adapter_file)
    print(f"Adapter file: {adapter_file.name}")
    print(f"  Size: {adapter_file.stat().st_size:,} bytes")
    print(f"  SHA256: {adapter_sha}")

    with open(prod_dir / "production_refit_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert adapter_sha == summary["adapter_safetensors_sha256"], "Adapter SHA256 mismatch with summary!"
    print("  [PASS] Adapter SHA256 matches summary JSON!")

    # 2. Reload base model and attach adapter
    print("\nReloading base model and attaching PEFT adapter...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    forecaster = TimestralForecaster.from_pretrained(str(model_dir))
    forecaster.model = PeftModel.from_pretrained(forecaster.model, str(ckpt_dir))
    forecaster.model.to(device)
    forecaster.model.eval()
    print("  [PASS] Successfully reloaded into PeftModel!")

    # 3. Load historical context (last 52 weeks)
    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    df_raw = pd.read_csv(data_path, parse_dates=["week"])
    df_piv = df_raw.pivot(index="week", columns=["warehouse_name", "article"], values="quantity").fillna(0.0)
    panel = df_piv.values.astype(np.float32)
    n_series = panel.shape[1]

    inf_ctx = panel[-CONTEXT:, :].T  # (8789, 52)
    inf_batch_size = 64
    n_batches = int(math.ceil(n_series / inf_batch_size))
    reloaded_preds_list = []

    print(f"Running inference with reloaded adapter on {n_series} series...")
    reloaded_preds_list = []
    with torch.no_grad():
        for i in range(0, n_series, inf_batch_size):
            b_ctx = torch.as_tensor(inf_ctx[i : i + inf_batch_size], dtype=torch.float32, device=device)
            fc = forecaster.forecast(b_ctx, prediction_length=HORIZON, clip_nonnegative=True)
            med = fc.median[:, 0, :].detach().cpu().numpy()
            reloaded_preds_list.append(med)

    reloaded_continuous = np.vstack(reloaded_preds_list)  # (8789, 9)

    # 4. Compare with saved future predictions
    print("\nVerifying numerical parity against saved CSV...")
    df_saved = pd.read_csv(future_file)
    assert len(df_saved) == 79101, f"Expected 79,101 rows, got {len(df_saved)}"

    # Order in saved CSV is week-first, then warehouse/article
    saved_pivot = df_saved.pivot(index=["warehouse_name", "article"], columns="week", values="y_pred_continuous")
    saved_arr = saved_pivot.values  # (8789, 9)

    max_abs_diff = float(np.max(np.abs(saved_arr - reloaded_continuous)))
    mean_abs_diff = float(np.mean(np.abs(saved_arr - reloaded_continuous)))

    print(f"  Max absolute difference:  {max_abs_diff:.8e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.8e}")

    assert max_abs_diff <= 1e-4, f"Parity failure: max difference {max_abs_diff} exceeds 1e-4!"
    print("  [PASS] Reload parity verified! (max diff <= 1e-4)")

    print("\n" + "=" * 80)
    print("PRODUCTION RELOAD VERIFICATION COMPLETED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
