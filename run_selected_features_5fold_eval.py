#!/usr/bin/env python3
"""
run_selected_features_5fold_eval.py

Full 5-Fold Cross-Validation for Timestral-209M LoRA with Selected Native Covariates:
Groups: ['D_category', 'C_network'] (Category Macro Context + Network Demand Context)

Evaluates:
- Folds: 4, 3, 2, 1, 0
- Metrics: Standalone RINT, Standalone Legacy P, Champion + 10% Blend
- Aggregates: Selection Folds (4, 3, 2), Check Folds (1, 0), Full 5-Fold CV
- Saves:
  - Raw OOF predictions: experiments/timestral_feature_research/oof_predictions/
  - Per-fold metrics table: experiments/timestral_feature_research/selected_features_5fold_metrics.csv
  - Summary JSON: experiments/timestral_feature_research/selected_features_5fold_summary.json
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_timestral_covariates_experiment import train_and_evaluate_timestral_run

SELECTED_GROUPS = ["D_category", "C_network"]
FOLDS = [4, 3, 2, 1, 0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out_dir = ROOT / "experiments" / "timestral_feature_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = out_dir / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "selected_features_5fold_metrics.csv"
    summary_json = out_dir / "selected_features_5fold_summary.json"

    print("=" * 105)
    print("TIMESTRAL-209M WITH SELECTED NATIVE COVARIATES: FULL 5-FOLD CV EVALUATION")
    print(f"Selected Groups: {SELECTED_GROUPS}")
    print(f"Evaluation Folds: {FOLDS}")
    print("=" * 105)

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

    records: List[Dict[str, Any]] = []
    completed_folds = set()
    if metrics_csv.exists():
        df_exist = pd.read_csv(metrics_csv)
        records = df_exist.to_dict(orient="records")
        completed_folds = set(df_exist["fold"].tolist())
        print(f"Loaded {len(records)} existing fold evaluations from {metrics_csv.name}.")

    start_total_time = time.time()

    for fold in FOLDS:
        if fold in completed_folds:
            print(f">>> Fold {fold} already evaluated. Skipping. <<<", flush=True)
            continue

        stage = "Selection" if fold in [4, 3, 2] else "Check"
        print(f"\n{'='*95}")
        print(f">>> STARTING EVALUATION ON FOLD {fold} ({stage} Fold) <<<")
        print(f"{'='*95}", flush=True)

        res = train_and_evaluate_timestral_run(
            engine=engine,
            groups=SELECTED_GROUPS,
            fold=fold,
            steps=120,
            seed=42,
        )

        # Save continuous OOF predictions
        oof_path = oof_dir / f"selected_covariates_raw_fold_{fold}.csv.gz"
        champ_path = ROOT / "experiments" / "champion_oof" / f"champion_scale_1m_fold_{fold}.csv.gz"
        df_champ = pd.read_csv(champ_path)
        df_save = df_champ[["week", "warehouse_name", "article"]].copy()
        df_save["raw_timestral_selected"] = res["pred_continuous"]
        df_save.to_csv(oof_path, index=False, compression="gzip")
        print(f"Saved raw OOF predictions to {oof_path.name} ({len(df_save):,} rows)", flush=True)

        row = {
            "fold": fold,
            "stage": stage,
            "groups": "+".join(SELECTED_GROUPS),
            "standalone_rint_clipped": res["standalone_rint"]["clipped"],
            "standalone_rint_unclipped": res["standalone_rint"]["unclipped"],
            "standalone_rint_wape": res["standalone_rint"]["wape"],
            "standalone_rint_rmse": res["standalone_rint"]["rmse"],
            "standalone_post_clipped": res["standalone_post"]["clipped"],
            "standalone_post_unclipped": res["standalone_post"]["unclipped"],
            "standalone_post_wape": res["standalone_post"]["wape"],
            "standalone_post_rmse": res["standalone_post"]["rmse"],
            "blend_w10_clipped": res["blend_w10"]["clipped"],
            "blend_w10_unclipped": res["blend_w10"]["unclipped"],
            "blend_w10_wape": res["blend_w10"]["wape"],
            "blend_w10_rmse": res["blend_w10"]["rmse"],
            "train_duration_sec": res["train_duration_sec"],
            "eval_duration_sec": res["eval_duration_sec"],
        }
        records.append(row)
        completed_folds.add(fold)
        pd.DataFrame(records).to_csv(metrics_csv, index=False)
        print(f"Progress saved to {metrics_csv.name}", flush=True)

    # -------------------------------------------------------------------------
    # Aggregate Summary
    # -------------------------------------------------------------------------
    df_all = pd.DataFrame(records).sort_values("fold", ascending=False)
    df_sel = df_all[df_all["stage"] == "Selection"]
    df_chk = df_all[df_all["stage"] == "Check"]

    summary = {
        "groups": SELECTED_GROUPS,
        "n_folds": len(df_all),
        "full_5fold": {
            "standalone_post_clipped": float(df_all["standalone_post_clipped"].mean()),
            "standalone_post_unclipped": float(df_all["standalone_post_unclipped"].mean()),
            "standalone_post_wape": float(df_all["standalone_post_wape"].mean()),
            "standalone_post_rmse": float(df_all["standalone_post_rmse"].mean()),
            "blend_w10_clipped": float(df_all["blend_w10_clipped"].mean()),
            "blend_w10_unclipped": float(df_all["blend_w10_unclipped"].mean()),
            "blend_w10_wape": float(df_all["blend_w10_wape"].mean()),
            "blend_w10_rmse": float(df_all["blend_w10_rmse"].mean()),
        },
        "selection_folds_4_3_2": {
            "standalone_post_clipped": float(df_sel["standalone_post_clipped"].mean()),
            "standalone_post_unclipped": float(df_sel["standalone_post_unclipped"].mean()),
            "standalone_post_wape": float(df_sel["standalone_post_wape"].mean()),
            "standalone_post_rmse": float(df_sel["standalone_post_rmse"].mean()),
            "blend_w10_clipped": float(df_sel["blend_w10_clipped"].mean()),
            "blend_w10_unclipped": float(df_sel["blend_w10_unclipped"].mean()),
            "blend_w10_wape": float(df_sel["blend_w10_wape"].mean()),
            "blend_w10_rmse": float(df_sel["blend_w10_rmse"].mean()),
        },
        "check_folds_1_0": {
            "standalone_post_clipped": float(df_chk["standalone_post_clipped"].mean()),
            "standalone_post_unclipped": float(df_chk["standalone_post_unclipped"].mean()),
            "standalone_post_wape": float(df_chk["standalone_post_wape"].mean()),
            "standalone_post_rmse": float(df_chk["standalone_post_rmse"].mean()),
            "blend_w10_clipped": float(df_chk["blend_w10_clipped"].mean()),
            "blend_w10_unclipped": float(df_chk["blend_w10_unclipped"].mean()),
            "blend_w10_wape": float(df_chk["blend_w10_wape"].mean()),
            "blend_w10_rmse": float(df_chk["blend_w10_rmse"].mean()),
        },
        "per_fold": records,
        "total_duration_sec": time.time() - start_total_time,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 105)
    print("5-FOLD EVALUATION COMPLETE!")
    print(f"Full 5-Fold Standalone Post Clipped:  {summary['full_5fold']['standalone_post_clipped']:.6f}")
    print(f"Full 5-Fold Standalone Post Unclipped:{summary['full_5fold']['standalone_post_unclipped']:.6f}")
    print(f"Full 5-Fold Standalone WAPE:          {summary['full_5fold']['standalone_post_wape']:.3f}%")
    print(f"Full 5-Fold Standalone RMSE:          {summary['full_5fold']['standalone_post_rmse']:.4f}")
    print(f"Full 5-Fold Blend w=0.10 Clipped:     {summary['full_5fold']['blend_w10_clipped']:.6f}")
    print(f"Full 5-Fold Blend w=0.10 Unclipped:   {summary['full_5fold']['blend_w10_unclipped']:.6f}")
    print(f"Full 5-Fold Blend w=0.10 WAPE:        {summary['full_5fold']['blend_w10_wape']:.3f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
