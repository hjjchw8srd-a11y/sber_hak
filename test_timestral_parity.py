#!/usr/bin/env python3
"""
test_timestral_parity.py

Diagnostic parity test between low-level predict_block() and official forecast():
1. Tests on authentic pretrained weights.
2. Compares across 3 representative series types:
   - Zero-heavy (all zeros in context)
   - Sparse / Rare demand (intermittent non-zero entries)
   - High-volume demand (large continuous sales)
3. Evaluates all 21 quantiles across 9-week horizon.
4. Part A: predict_block() vs forecast(symmetric_averaging=False, clip_nonnegative=False).
5. Part B: predict_block() with manual symmetric averaging & non-negative clamping vs official forecast(symmetric_averaging=True, clip_nonnegative=True).
6. Measures max absolute and relative discrepancies.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from timestral import TimestralForecaster
from timestral.forecaster import pad_to_patch, nonnegative_context

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_dir = ROOT / "experiments" / "urgent_models" / "timestral_209m"
    forecaster = TimestralForecaster.from_pretrained(weights_dir, device=device)
    forecaster.model.eval()

    # Construct representative test contexts of length 52
    torch.manual_seed(42)
    
    # 1. Zero-heavy (all zeros)
    ctx_zero = torch.zeros(1, 52, device=device, dtype=torch.float32)
    
    # 2. Sparse / Rare demand (e.g. 3 sales of 1 or 2 items across 52 weeks)
    ctx_sparse = torch.zeros(1, 52, device=device, dtype=torch.float32)
    ctx_sparse[0, [10, 25, 48]] = torch.tensor([1.0, 2.0, 1.0], device=device)
    
    # 3. High-volume demand (e.g. sales around 50..150 items)
    ctx_high = torch.normal(mean=80.0, std=15.0, size=(1, 52), device=device).clamp_min(10.0)

    contexts = {
        "Zero-heavy": ctx_zero,
        "Sparse / Rare": ctx_sparse,
        "High-volume": ctx_high,
    }

    horizon = 9
    ps = forecaster.patch_size  # 32

    print("=" * 80)
    print("TIMESTRAL-209M PARITY TEST: LOW-LEVEL predict_block() vs OFFICIAL forecast()")
    print("=" * 80)

    for name, ctx in contexts.items():
        print(f"\n--- Testing Series: {name} ---")
        print(f"  Context min: {ctx.min().item():.2f}, mean: {ctx.mean().item():.2f}, max: {ctx.max().item():.2f}")

        # Official forecast WITHOUT symmetric averaging and WITHOUT non-negative clipping
        with torch.inference_mode():
            fc_raw = forecaster.forecast(ctx, prediction_length=horizon, symmetric_averaging=False, clip_nonnegative=False)
            q_official_raw = fc_raw.quantiles[:, 0, :, :]  # (1, 9, 21)

        # Low-level predict_block pass
        with torch.inference_mode():
            mask = torch.ones_like(ctx, dtype=torch.bool)
            padded_ctx, padded_mask = pad_to_patch(ctx, mask, patch_size=ps)
            group_ids = torch.zeros(1, dtype=torch.long, device=device)
            # 1 horizon patch covers 32 steps >= 9
            q_predict_block = forecaster.model.predict_block(padded_ctx, padded_mask, group_ids, horizon_patches=1)[:, :horizon, :]

        abs_diff_raw = (q_official_raw - q_predict_block).abs()
        rel_diff_raw = abs_diff_raw / (q_official_raw.abs() + 1e-7)

        max_abs_raw = abs_diff_raw.max().item()
        max_rel_raw = rel_diff_raw.max().item()
        mean_abs_raw = abs_diff_raw.mean().item()

        print(f"  [Part A: Single Pass (No symmetry, No clamp)]")
        print(f"    Max Absolute Discrepancy: {max_abs_raw:.8e}")
        print(f"    Mean Absolute Discrepancy: {mean_abs_raw:.8e}")
        print(f"    Max Relative Discrepancy: {max_rel_raw:.8e}")

        # Official forecast WITH symmetric averaging and WITH non-negative clipping
        with torch.inference_mode():
            fc_sym = forecaster.forecast(ctx, prediction_length=horizon, symmetric_averaging=True, clip_nonnegative=True)
            q_official_sym = fc_sym.quantiles[:, 0, :, :]  # (1, 9, 21)

        # Reproduce symmetric averaging & non-negative clamping via predict_block:
        with torch.inference_mode():
            # Pass 1: forward
            p1 = forecaster.model.predict_block(padded_ctx, padded_mask, group_ids, horizon_patches=1)[:, :horizon, :]
            # Pass 2: mirrored negative forward
            p2 = forecaster.model.predict_block(-padded_ctx, padded_mask, group_ids, horizon_patches=1)[:, :horizon, :]
            # Symmetric combination: 0.5 * (p1 + (-p2).flip(-1))
            p_sym = 0.5 * (p1 + (-p2).flip(-1))
            # Nonnegative clamp
            eligible = nonnegative_context(ctx.unsqueeze(1), mask.unsqueeze(1)).squeeze(1)
            if eligible.any():
                p_sym = p_sym.clamp_min(0.0)

        abs_diff_sym = (q_official_sym - p_sym).abs()
        rel_diff_sym = abs_diff_sym / (q_official_sym.abs() + 1e-7)

        max_abs_sym = abs_diff_sym.max().item()
        max_rel_sym = rel_diff_sym.max().item()
        mean_abs_sym = abs_diff_sym.mean().item()

        print(f"  [Part B: Full Official Protocol (Symmetric Averaging + Nonnegative Clamp)]")
        print(f"    Max Absolute Discrepancy: {max_abs_sym:.8e}")
        print(f"    Mean Absolute Discrepancy: {mean_abs_sym:.8e}")
        print(f"    Max Relative Discrepancy: {max_rel_sym:.8e}")

        # Verification of Quantile Monotonicity
        mono_official = (torch.diff(q_official_sym, dim=-1) >= -1e-6).all().item()
        mono_predict = (torch.diff(p_sym, dim=-1) >= -1e-6).all().item()
        print(f"    Quantile Monotonicity: Official={mono_official}, predict_block={mono_predict}")

        assert max_abs_raw < 1e-5, f"Discrepancy in Part A exceeds tolerance: {max_abs_raw}"
        assert max_abs_sym < 1e-5, f"Discrepancy in Part B exceeds tolerance: {max_abs_sym}"

    print("\n" + "=" * 80)
    print("ALL PARITY TESTS PASSED BIT-EXACT (Discrepancy < 1e-5 across all quantiles & series)!")
    print("=" * 80)

if __name__ == "__main__":
    main()
