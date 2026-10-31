#!/usr/bin/env python3
"""
diagnose_demand_segments.py

Evaluates error metrics across demand segments for:
1) Standalone Champion (Scale_1M_Dual_0.5)
2) Standalone Timestral-209M LoRA
3) Optimal Ensemble Blend (e.g., w=0.05 / w=0.10)

Segments:
- Zero-Heavy: series where >= 80% of test weeks have zero demand
- Regular Demand: series where >= 50% of test weeks have non-zero demand
- High-Volume: top 10% series by total actual sales volume across the test horizon
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

KEYS = ["week", "warehouse_name", "article"]
SERIES_KEYS = ["warehouse_name", "article"]
FOLDS = [4, 3, 2, 1, 0]


def calc_segment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes SMAPE, WAPE, RMSE, and Bias for a specific segment."""
    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero_denom = denom > 0
    smape_vals = np.zeros_like(y_true, dtype=np.float64)
    smape_vals[nonzero_denom] = 200.0 * np.abs(y_true[nonzero_denom] - y_pred[nonzero_denom]) / denom[nonzero_denom]
    smape = float(np.mean(smape_vals))

    total_actual = float(np.sum(np.abs(y_true)))
    total_abs_err = float(np.sum(np.abs(y_true - y_pred)))
    wape = (total_abs_err / total_actual * 100.0) if total_actual > 0 else 0.0

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float(np.sum(y_pred - y_true) / total_actual * 100.0) if total_actual > 0 else 0.0

    return {
        "count": len(y_true),
        "total_actual": total_actual,
        "smape": smape,
        "wape": wape,
        "rmse": rmse,
        "bias": bias,
    }


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def run_segment_diagnosis(optimal_w: float = 0.05) -> dict:
    all_fold_results = {}

    for fold in FOLDS:
        champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
        tim_path = ROOT / "experiments" / f"timestral_lora_fold{fold}" / "keyed_predictions" / "pilot" / f"timestral_209m_lora_fold{fold}_pilot_keyed.csv.gz"

        if not champ_path.exists() or not tim_path.exists():
            print(f"[SKIP] Fold {fold} files not fully available yet.")
            continue

        df_champ = pd.read_csv(champ_path)
        df_tim = pd.read_csv(tim_path)

        merged = pd.merge(
            df_champ[KEYS + ["raw_champion", "final_champion", "y_true"]],
            df_tim[KEYS + ["y_pred_continuous"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        assert len(merged) == 79101

        # Continuous blend
        raw_champ = merged["raw_champion"].values
        raw_tim = merged["y_pred_continuous"].values
        raw_blend = (1.0 - optimal_w) * raw_champ + optimal_w * raw_tim

        pred_champ = merged["final_champion"].values
        pred_tim = apply_p(raw_tim, 1.05, 0.65)
        pred_blend = apply_p(raw_blend, 1.05, 0.65)
        y_true = merged["y_true"].values

        merged["pred_champ"] = pred_champ
        merged["pred_tim"] = pred_tim
        merged["pred_blend"] = pred_blend

        # Series-level statistics
        series_stats = (
            merged.groupby(SERIES_KEYS)
            .agg(
                zero_rate=("y_true", lambda s: (s == 0).mean()),
                total_volume=("y_true", "sum"),
            )
            .reset_index()
        )

        vol_90 = np.quantile(series_stats["total_volume"].values, 0.90)

        series_stats["is_zero_heavy"] = series_stats["zero_rate"] >= 0.80
        series_stats["is_regular"] = (1.0 - series_stats["zero_rate"]) >= 0.50
        series_stats["is_high_volume"] = series_stats["total_volume"] >= vol_90

        merged = merged.merge(series_stats[SERIES_KEYS + ["is_zero_heavy", "is_regular", "is_high_volume"]], on=SERIES_KEYS)

        fold_diag = {}
        for seg_name, mask_col in [
            ("All Series", None),
            ("Zero-Heavy (>=80% zeros)", "is_zero_heavy"),
            ("Regular (>=50% non-zero)", "is_regular"),
            ("High-Volume (Top 10%)", "is_high_volume"),
        ]:
            sub = merged if mask_col is None else merged[merged[mask_col]]
            yt = sub["y_true"].values
            fold_diag[seg_name] = {
                "n_series": int(sub[SERIES_KEYS].drop_duplicates().shape[0]),
                "n_rows": len(yt),
                "actual_sum": float(yt.sum()),
                "actual_mean": float(yt.mean()),
                "champion": calc_segment_metrics(yt, sub["pred_champ"].values),
                "timestral": calc_segment_metrics(yt, sub["pred_tim"].values),
                "blend": calc_segment_metrics(yt, sub["pred_blend"].values),
            }

        all_fold_results[fold] = fold_diag

    return all_fold_results


def print_and_save_diagnosis(results: dict, optimal_w: float = 0.05):
    if not results:
        print("No completed folds to report.")
        return

    print("=" * 110)
    print(f"DEMAND SEGMENT ERROR DIAGNOSIS (Optimal Blend w = {optimal_w:.2f})")
    print("=" * 110)

    segments = ["All Series", "Zero-Heavy (>=80% zeros)", "Regular (>=50% non-zero)", "High-Volume (Top 10%)"]

    # Average across available folds
    avg_results = {}
    for seg in segments:
        avg_results[seg] = {
            "champion": {"smape": [], "wape": [], "rmse": [], "bias": []},
            "timestral": {"smape": [], "wape": [], "rmse": [], "bias": []},
            "blend": {"smape": [], "wape": [], "rmse": [], "bias": []},
        }
        for f, f_dict in results.items():
            for model_key in ["champion", "timestral", "blend"]:
                for metric in ["smape", "wape", "rmse", "bias"]:
                    avg_results[seg][model_key][metric].append(f_dict[seg][model_key][metric])

    print(f"\n{'Segment':<28} {'Model':<12} {'SMAPE (%)':<12} {'WAPE (%)':<12} {'RMSE':<10} {'Bias (%)':<10}")
    print("-" * 88)
    for seg in segments:
        for model_key, display_name in [("champion", "Champion"), ("timestral", "Timestral"), ("blend", f"Blend({optimal_w:.2f})")]:
            m = {k: np.mean(v) for k, v in avg_results[seg][model_key].items()}
            print(f"{seg:<28} {display_name:<12} {m['smape']:<12.3f} {m['wape']:<12.3f} {m['rmse']:<10.4f} {m['bias']:<+10.2f}")
        print("-" * 88)

    out_file = ROOT / "experiments" / "champion_oof" / "demand_segment_diagnosis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnosis results saved to: {out_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=0.10, help="Blend weight for Timestral")
    args = parser.parse_args()

    res = run_segment_diagnosis(optimal_w=args.weight)
    print_and_save_diagnosis(res, optimal_w=args.weight)


if __name__ == "__main__":
    main()
