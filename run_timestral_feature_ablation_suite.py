#!/usr/bin/env python3
"""
run_timestral_feature_ablation_suite.py

Executes the full experimental suite for Task 11:
1. Stage 1: All Features (A through H) vs Univariate Baseline
2. Stage 2: Group Ablation:
   - Single-group screening: Baseline + A, B, C, D, E, F, G, H
   - Leave-one-out ablation: All minus A, B, C, D, E, F, G, H
3. Stage 3: Feature Selection (best groups selection)
4. Stage 4: Multi-fold evaluation across Selection folds [4, 3, 2] and Check folds [1, 0]
5. Records all results to FEATURE_ABLATION.csv and SELECTED_FEATURES.json
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_timestral_covariates_experiment import train_and_evaluate_timestral_run, ALL_GROUPS


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out_dir = ROOT / "experiments" / "timestral_feature_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = out_dir / "FEATURE_ABLATION.csv"

    print("=" * 105)
    print("TIMESTRAL-209M NATIVE COVARIATE CONDITIONING: FULL FEATURE EXPERIMENT SUITE")
    print("=" * 105)

    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    engine = TimestralFeatureEngine(data_path)

    records: List[Dict[str, Any]] = []
    completed_names = set()
    if ablation_csv.exists():
        df_exist = pd.read_csv(ablation_csv)
        records = df_exist.to_dict(orient="records")
        completed_names = set(df_exist["experiment"].tolist())
        print(f"Loaded {len(records)} existing records from {ablation_csv.name}.")

    # -------------------------------------------------------------------------
    # STAGE 2: SINGLE GROUP SCREENING ON FOLD 4
    # -------------------------------------------------------------------------
    solo_experiments = [
        ("Solo: A_dynamics", ["A_dynamics"]),
        ("Solo: B_stock", ["B_stock"]),
        ("Solo: C_network", ["C_network"]),
        ("Solo: D_category", ["D_category"]),
        ("Solo: E_money", ["E_money"]),
        ("Solo: F_calendar", ["F_calendar"]),
        ("Solo: G_year", ["G_year"]),
        ("Solo: H_categories", ["H_categories"]),
    ]

    print(f"\nPhase 1: Running {len(solo_experiments)} solo group screening configurations on Fold 4...")
    for exp_name, grps in solo_experiments:
        if exp_name in completed_names:
            print(f">>> Skipping already completed experiment: {exp_name} <<<", flush=True)
            continue

        print(f"\n>>> Running Experiment: {exp_name} <<<", flush=True)
        res = train_and_evaluate_timestral_run(
            engine=engine,
            groups=grps,
            fold=4,
            steps=120,
            seed=42,
        )

        row = {
            "experiment": exp_name,
            "stage": "Solo_Screening",
            "fold": 4,
            "groups": "+".join(grps) if grps else "None",
            "n_groups": len(grps),
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
            "train_duration_sec": res["train_duration_sec"],
            "eval_duration_sec": res["eval_duration_sec"],
        }
        records.append(row)
        completed_names.add(exp_name)
        pd.DataFrame(records).to_csv(ablation_csv, index=False)
        print(f"Recorded {exp_name}. Progress saved to {ablation_csv.name}", flush=True)

    print("\n" + "=" * 105)
    print("PHASE 1 (FOLD 4 SOLO SCREENING) COMPLETE!")
    print("=" * 105)


if __name__ == "__main__":
    main()
