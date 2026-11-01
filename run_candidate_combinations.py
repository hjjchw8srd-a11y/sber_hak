#!/usr/bin/env python3
"""
run_candidate_combinations.py

Evaluates candidate combinations of top-performing feature groups on Fold 4:
1. D_category + C_network (Macro Context)
2. D_category + A_dynamics (Category + SKU Dynamics)
3. D_category + C_network + A_dynamics (Triad)
4. D_category + C_network + H_categories (Context + Encodings)
5. D_category + C_network + G_year (Context + Annual Lag)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_timestral_covariates_experiment import train_and_evaluate_timestral_run


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    out_dir = ROOT / "experiments" / "timestral_feature_research"
    ablation_csv = out_dir / "FEATURE_ABLATION.csv"

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

    records: List[Dict[str, Any]] = []
    completed_names = set()
    if ablation_csv.exists():
        df_exist = pd.read_csv(ablation_csv)
        records = df_exist.to_dict(orient="records")
        completed_names = set(df_exist["experiment"].tolist())

    combos = [
        ("Combo: D_category + C_network", ["D_category", "C_network"]),
        ("Combo: D_category + A_dynamics", ["D_category", "A_dynamics"]),
        ("Combo: D_category + C_network + A_dynamics", ["D_category", "C_network", "A_dynamics"]),
        ("Combo: D_category + C_network + H_categories", ["D_category", "C_network", "H_categories"]),
        ("Combo: D_category + C_network + G_year", ["D_category", "C_network", "G_year"]),
    ]

    print("\n" + "=" * 95)
    print("RUNNING CANDIDATE FEATURE COMBINATIONS ON FOLD 4")
    print("=" * 95)

    for exp_name, grps in combos:
        if exp_name in completed_names:
            print(f">>> Skipping already completed: {exp_name} <<<", flush=True)
            continue

        print(f"\n>>> Running: {exp_name} <<<", flush=True)
        res = train_and_evaluate_timestral_run(
            engine=engine,
            groups=grps,
            fold=4,
            steps=120,
            seed=42,
        )

        row = {
            "experiment": exp_name,
            "stage": "Combo_Selection",
            "fold": 4,
            "groups": "+".join(grps),
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

    print("\n" + "=" * 95)
    print("COMBINATIONS COMPLETE!")
    print("=" * 95)


if __name__ == "__main__":
    main()
