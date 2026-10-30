#!/usr/bin/env python3
"""
evaluate_candidate_weights_seeds.py

Rigorous multi-seed paired evaluation of Candidate Weights (45/37/9/9)
vs Control B (50/30/10/10) across Seeds [42, 52, 62] on all 5 folds.

Configuration:
- Control B: TFT=0.50, LGBM=0.30, CatBoost=0.10, PerCategory=0.10
- Candidate: TFT=0.45, LGBM=0.37, CatBoost=0.09, PerCategory=0.09
  (Equivalent to: 0.90 * Control_Blend_Raw + 0.10 * LGBM_after_comp)
- Postprocessing P: alpha=1.05, threshold=0.65, rounding=rint, clip_lower=0.0
- Model components:
  * TFT, CatBoost, PerCategory frozen from baseline fold meta features / keyed OOF.
  * LightGBM B: 52 origins, 350,000 sampled rows, base_calendar_network (24 features),
    trained per seed in {42, 52, 62}.

Protocol:
- Selection Folds: [4, 3, 2]
- Check Folds: [1, 0]
- Outputs:
  * experiments/model_improvements/candidate_weights/candidate_weights_seed_metrics.csv
  * experiments/model_improvements/candidate_weights/candidate_weights_summary.json
  * experiments/model_improvements/candidate_weights/keyed/
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import (
    HORIZON,
    KEYS,
    WeeklyMatrices,
    make_truth,
)
from train_tft_weekly import baseline_mean13
from sber_hak.history_audit_pipeline import (
    compute_metrics_with_diagnostics,
    predict_from_components,
    train_lgbm_history_variant,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_improvements" / "candidate_weights"
KEYED_DIR = OUTPUT_DIR / "keyed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYED_DIR.mkdir(parents=True, exist_ok=True)

FOLD_ORIGINS = {
    4: 89,   # 2025-09-15 — 2025-11-10 (Selection)
    3: 98,   # 2025-11-17 — 2026-01-12 (Selection)
    2: 107,  # 2026-01-19 — 2026-03-16 (Selection)
    1: 116,  # 2026-03-23 — 2026-05-18 (Check)
    0: 125,  # 2026-05-25 — 2026-07-20 (Check)
}

SELECTION_FOLDS = [4, 3, 2]
CHECK_FOLDS = [1, 0]
SEEDS = [42, 52, 62]

CONTROL_WEIGHTS = {"tft": 0.50, "lgbm": 0.30, "catboost": 0.10, "per_category": 0.10}
CANDIDATE_WEIGHTS = {"tft": 0.45, "lgbm": 0.37, "catboost": 0.09, "per_category": 0.09}


def apply_p(raw_blend: np.ndarray) -> np.ndarray:
    scaled = raw_blend * 1.05
    gated = np.where(scaled < 0.65, 0.0, scaled)
    return np.rint(gated).clip(0.0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("MULTI-SEED PAIRED EVALUATION: CANDIDATE (45/37/9/9) vs CONTROL B (50/30/10/10)")
    print("Seeds: 42, 52, 62 across all 5 Folds")
    print("=" * 80)

    # 1. Load data
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    print(f"Loading weekly dataset from {data_path}...")
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    weeks = sorted(data["week"].unique())

    # 2. Precompute truth and mean13 baseline for each fold
    print("\nPrecomputing mean13 baselines and ground truth for all 5 folds...")
    fold_truth = {}
    fold_baseline = {}
    for fold, origin in FOLD_ORIGINS.items():
        truth = make_truth(matrices, origin)
        truth["week"] = pd.to_datetime(truth["week"]).dt.strftime("%Y-%m-%d")
        fold_truth[fold] = truth

        val_weeks = weeks[origin : origin + HORIZON]
        b_df = baseline_mean13(data, pd.Timestamp(val_weeks[0]), pd.Index(val_weeks))
        b_df["week"] = pd.to_datetime(b_df["week"]).dt.strftime("%Y-%m-%d")
        fold_baseline[fold] = b_df

    # 3. Load frozen components (TFT, CB, PerCat) from keyed OOF
    print("\nLoading frozen component predictions (TFT, CatBoost, PerCategory)...")
    frozen_components = {}
    for fold in FOLD_ORIGINS:
        blend_path = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
        df_blend = pd.read_csv(blend_path)
        df_blend["week"] = pd.to_datetime(df_blend["week"]).dt.strftime("%Y-%m-%d")
        frozen_components[fold] = {
            "tft": df_blend["after_comp_tft"].values.astype(np.float64),
            "cb": df_blend["after_comp_catboost"].values.astype(np.float64),
            "per_cat": df_blend["after_comp_per_category"].values.astype(np.float64),
            "keys": df_blend[KEYS].copy(),
            "lgbm_s42": df_blend["after_comp_lgbm"].values.astype(np.float64),
        }

    # 4. Multi-seed training and paired evaluation
    records = []
    t_start_all = time.perf_counter()

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"--- RUNNING SEED {seed} ---")
        print(f"{'='*40}")

        for fold, origin in FOLD_ORIGINS.items():
            role = "Selection" if fold in SELECTION_FOLDS else "Check"
            print(f"\nFold {fold} ({role}, origin {origin}, seed {seed}):")

            # Check if seed 42 can reuse exact verified LightGBM output
            if seed == 42:
                print("  Reusing verified LightGBM B seed 42 predictions...")
                l_after_comp = frozen_components[fold]["lgbm_s42"]
                train_time = 0.0
            else:
                print(f"  Training LightGBM B (seed={seed}, 52 origins, 350k rows)...")
                t0 = time.perf_counter()
                df_pred, diag = train_lgbm_history_variant(matrices, origin, variant="B", seed=seed)
                train_time = time.perf_counter() - t0
                print(f"  Trained in {train_time:.2f}s (mean p_sale={diag['mean_p_sale']:.4f}, mean exp={diag['mean_expected']:.4f})")

                # Align keys with frozen components
                merged_l = frozen_components[fold]["keys"].merge(
                    df_pred[["week", "warehouse_name", "article", "p_sale", "expected_quantity"]],
                    on=KEYS,
                    how="inner",
                    validate="one_to_one"
                )
                assert len(merged_l) == 79101, f"Length mismatch: {len(merged_l)}"
                p_s = merged_l["p_sale"].values.astype(np.float64)
                exp = merged_l["expected_quantity"].values.astype(np.float64)
                # Apply standard inference-matched gate 0.5
                l_after_comp = np.where(p_s < 0.5, 0.0, exp)

            t_comp = frozen_components[fold]["tft"]
            cb_comp = frozen_components[fold]["cb"]
            per_comp = frozen_components[fold]["per_cat"]

            # 1. Compute Control B (50/30/10/10)
            raw_ctrl = 0.50 * t_comp + 0.30 * l_after_comp + 0.10 * cb_comp + 0.10 * per_comp
            pred_ctrl = apply_p(raw_ctrl)

            sub_ctrl = frozen_components[fold]["keys"].copy()
            sub_ctrl["quantity"] = pred_ctrl
            m_ctrl = compute_metrics_with_diagnostics(fold_truth[fold], sub_ctrl, fold_baseline[fold])

            # 2. Compute Candidate (45/37/9/9)
            raw_cand = 0.45 * t_comp + 0.37 * l_after_comp + 0.09 * cb_comp + 0.09 * per_comp
            pred_cand = apply_p(raw_cand)

            sub_cand = frozen_components[fold]["keys"].copy()
            sub_cand["quantity"] = pred_cand
            m_cand = compute_metrics_with_diagnostics(fold_truth[fold], sub_cand, fold_baseline[fold])

            delta_unclipped = m_cand["unclipped_ratio"] - m_ctrl["unclipped_ratio"]
            delta_clipped = m_cand["clipped_score"] - m_ctrl["clipped_score"]

            print(f"  Control B: Score = {m_ctrl['clipped_score']:.6f}, Unclipped = {m_ctrl['unclipped_ratio']:.6f}")
            print(f"  Candidate: Score = {m_cand['clipped_score']:.6f}, Unclipped = {m_cand['unclipped_ratio']:.6f} (Delta unclipped = {delta_unclipped:+.6f})")

            # Save keyed predictions for candidate
            cand_keyed = frozen_components[fold]["keys"].copy()
            cand_keyed["origin"] = origin
            cand_keyed["seed"] = seed
            cand_keyed["tft_comp"] = t_comp
            cand_keyed["lgbm_comp"] = l_after_comp
            cand_keyed["cb_comp"] = cb_comp
            cand_keyed["per_comp"] = per_comp
            cand_keyed["raw_ctrl"] = raw_ctrl
            cand_keyed["pred_ctrl"] = pred_ctrl
            cand_keyed["raw_cand"] = raw_cand
            cand_keyed["pred_cand"] = pred_cand
            cand_keyed.to_csv(KEYED_DIR / f"cand_weights_s{seed}_fold_{fold}.csv.gz", index=False, compression="gzip")

            # Record Control
            records.append({
                "seed": seed,
                "fold": fold,
                "role": role,
                "origin": origin,
                "model": "Control_B_50_30_10_10",
                "train_time_sec": train_time,
                "clipped_score": m_ctrl["clipped_score"],
                "unclipped_ratio": m_ctrl["unclipped_ratio"],
                "smape": m_ctrl["smape"],
                "wape": m_ctrl["wape"],
                "rmse": m_ctrl["rmse"],
                "bias": m_ctrl["bias"],
                "rmsle": m_ctrl["rmsle"],
                "precision": m_ctrl["precision"],
                "recall": m_ctrl["recall"],
            })

            # Record Candidate
            records.append({
                "seed": seed,
                "fold": fold,
                "role": role,
                "origin": origin,
                "model": "Candidate_45_37_9_9",
                "train_time_sec": train_time,
                "clipped_score": m_cand["clipped_score"],
                "unclipped_ratio": m_cand["unclipped_ratio"],
                "smape": m_cand["smape"],
                "wape": m_cand["wape"],
                "rmse": m_cand["rmse"],
                "bias": m_cand["bias"],
                "rmsle": m_cand["rmsle"],
                "precision": m_cand["precision"],
                "recall": m_cand["recall"],
                "paired_delta_unclipped": delta_unclipped,
                "paired_delta_clipped": delta_clipped,
            })

    total_time = time.perf_counter() - t_start_all
    df_metrics = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "candidate_weights_seed_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # Compute Summary Statistics
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runtime_sec": total_time,
        "seeds": SEEDS,
        "by_seed": {},
        "overall": {},
    }

    print("\n" + "=" * 80)
    print("PAIRED SUMMARY: CANDIDATE vs CONTROL B ACROSS SEEDS")
    print("=" * 80)

    for seed in SEEDS:
        df_s = df_metrics[df_metrics["seed"] == seed]
        ctrl_sel = df_s[(df_s["model"] == "Control_B_50_30_10_10") & (df_s["role"] == "Selection")]["unclipped_ratio"].mean()
        cand_sel = df_s[(df_s["model"] == "Candidate_45_37_9_9") & (df_s["role"] == "Selection")]["unclipped_ratio"].mean()
        ctrl_chk = df_s[(df_s["model"] == "Control_B_50_30_10_10") & (df_s["role"] == "Check")]["unclipped_ratio"].mean()
        cand_chk = df_s[(df_s["model"] == "Candidate_45_37_9_9") & (df_s["role"] == "Check")]["unclipped_ratio"].mean()
        ctrl_all_clip = df_s[df_s["model"] == "Control_B_50_30_10_10"]["clipped_score"].mean()
        cand_all_clip = df_s[df_s["model"] == "Candidate_45_37_9_9"]["clipped_score"].mean()
        ctrl_all_unc = df_s[df_s["model"] == "Control_B_50_30_10_10"]["unclipped_ratio"].mean()
        cand_all_unc = df_s[df_s["model"] == "Candidate_45_37_9_9"]["unclipped_ratio"].mean()

        d_sel = cand_sel - ctrl_sel
        d_chk = cand_chk - ctrl_chk
        d_all_clip = cand_all_clip - ctrl_all_clip
        d_all_unc = cand_all_unc - ctrl_all_unc

        print(f"Seed {seed}:")
        print(f"  Selection (4,3,2) Unclipped: Control={ctrl_sel:.6f}, Cand={cand_sel:.6f}, Delta={d_sel:+.6f}")
        print(f"  Check (1,0)     Unclipped: Control={ctrl_chk:.6f}, Cand={cand_chk:.6f}, Delta={d_chk:+.6f}")
        print(f"  All 5 Folds       Clipped: Control={ctrl_all_clip:.6f}, Cand={cand_all_clip:.6f}, Delta={d_all_clip:+.6f}")
        print(f"  All 5 Folds     Unclipped: Control={ctrl_all_unc:.6f}, Cand={cand_all_unc:.6f}, Delta={d_all_unc:+.6f}")

        summary["by_seed"][str(seed)] = {
            "selection_unclipped_ctrl": float(ctrl_sel),
            "selection_unclipped_cand": float(cand_sel),
            "selection_unclipped_delta": float(d_sel),
            "check_unclipped_ctrl": float(ctrl_chk),
            "check_unclipped_cand": float(cand_chk),
            "check_unclipped_delta": float(d_chk),
            "all_clipped_ctrl": float(ctrl_all_clip),
            "all_clipped_cand": float(cand_all_clip),
            "all_clipped_delta": float(d_all_clip),
            "all_unclipped_ctrl": float(ctrl_all_unc),
            "all_unclipped_cand": float(cand_all_unc),
            "all_unclipped_delta": float(d_all_unc),
        }

    # Paired Aggregation across all 3 seeds
    sel_deltas = [summary["by_seed"][str(s)]["selection_unclipped_delta"] for s in SEEDS]
    chk_deltas = [summary["by_seed"][str(s)]["check_unclipped_delta"] for s in SEEDS]
    all_clip_deltas = [summary["by_seed"][str(s)]["all_clipped_delta"] for s in SEEDS]
    all_unc_deltas = [summary["by_seed"][str(s)]["all_unclipped_delta"] for s in SEEDS]

    summary["overall"] = {
        "selection_delta_mean": float(np.mean(sel_deltas)),
        "selection_delta_std": float(np.std(sel_deltas)),
        "check_delta_mean": float(np.mean(chk_deltas)),
        "check_delta_std": float(np.std(chk_deltas)),
        "all_clipped_delta_mean": float(np.mean(all_clip_deltas)),
        "all_clipped_delta_std": float(np.std(all_clip_deltas)),
        "all_unclipped_delta_mean": float(np.mean(all_unc_deltas)),
        "all_unclipped_delta_std": float(np.std(all_unc_deltas)),
    }

    print("\n" + "=" * 80)
    print("MULTI-SEED AGGREGATED DELTAS (CANDIDATE - CONTROL):")
    print(f"  Selection (4,3,2) Unclipped Delta: mean = {summary['overall']['selection_delta_mean']:+.6f}, std = {summary['overall']['selection_delta_std']:.6f}")
    print(f"  Check (1,0)     Unclipped Delta: mean = {summary['overall']['check_delta_mean']:+.6f}, std = {summary['overall']['check_delta_std']:.6f}")
    print(f"  All 5 Folds       Clipped Delta: mean = {summary['overall']['all_clipped_delta_mean']:+.6f}, std = {summary['overall']['all_clipped_delta_std']:.6f}")
    print(f"  All 5 Folds     Unclipped Delta: mean = {summary['overall']['all_unclipped_delta_mean']:+.6f}, std = {summary['overall']['all_unclipped_delta_std']:.6f}")
    print("=" * 80)

    summary_path = OUTPUT_DIR / "candidate_weights_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
