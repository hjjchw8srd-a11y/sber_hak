#!/usr/bin/env python3
"""
run_feature_expansion_screening.py

Evaluates adding remaining feature groups to the D_category + C_network baseline:
- D+C + B_stock
- D+C + F_calendar
- D+C + A_dynamics
- D+C + G_year
- D+C + H_categories
- D+C + E_money

Runs short screening (150 steps with internal validation checkpointing)
on Selection Fold 2 (independent selection fold, origin 107) to avoid overfitting Fold 4.
Records all results to feature_expansion_screening.csv.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from experiments.timestral_feature_research.timestral_feature_engine import TimestralFeatureEngine
from run_harmonized_timestral_protocol import train_harmonized_model


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

    candidates = {
        "D+C (Baseline)": ["D_category", "C_network"],
        "D+C + B_stock": ["D_category", "C_network", "B_stock"],
        "D+C + F_calendar": ["D_category", "C_network", "F_calendar"],
        "D+C + A_dynamics": ["D_category", "C_network", "A_dynamics"],
        "D+C + G_year": ["D_category", "C_network", "G_year"],
        "D+C + H_categories": ["D_category", "C_network", "H_categories"],
        "D+C + E_money": ["D_category", "C_network", "E_money"],
    }

    fold = 2  # Unseen Selection Fold (Origin 107)
    steps = 150
    eval_steps = 75

    records = []
    print("=" * 90)
    print(f"RUNNING FEATURE EXPANSION SCREENING ON SELECTION FOLD {fold} (Origin 107)")
    print(f"Budget: {steps} steps | Eval every {eval_steps} steps | Seed: 42")
    print("=" * 90, flush=True)

    for name, groups in candidates.items():
        tag = name.replace(" ", "_").replace("+", "_").lower()
        print(f"\n--- Testing Candidate: {name} (Groups: {groups}) ---", flush=True)
        t0 = time.time()
        torch.cuda.empty_cache()

        res = train_harmonized_model(
            engine=engine,
            groups=groups,
            fold=fold,
            total_steps=steps,
            eval_steps=eval_steps,
            microbatch=8,
            grad_accum=4,
            seed=42,
            save_tag=f"screen_f2_{tag}",
        )
        elapsed = time.time() - t0

        records.append({
            "candidate": name,
            "fold": fold,
            "groups": "+".join(groups),
            "n_groups": len(groups),
            "best_step": res["best_step"],
            "val_unclipped": res["best_val_unclipped"],
            "test_post_clipped": res["test_post"]["clipped"],
            "test_post_unclipped": res["test_post"]["unclipped"],
            "test_post_wape": res["test_post"]["wape"],
            "test_post_rmse": res["test_post"]["rmse"],
            "test_rint_clipped": res["test_rint"]["clipped"],
            "test_blend_clipped": res["test_blend_w10"]["clipped"],
            "duration_sec": elapsed,
        })
        print(f"  >>> {name}: Post Clp = {res['test_post']['clipped']:.6f} | Post Unc = {res['test_post']['unclipped']:.6f} | WAPE = {res['test_post']['wape']:.3f}%\n", flush=True)

    df = pd.DataFrame(records)
    out_csv = ROOT / "experiments" / "timestral_feature_research" / "feature_expansion_screening_fold2.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print("FEATURE EXPANSION SCREENING RESULTS (Sorted by Test Post Clipped Score):")
    print("=" * 90)
    df_sorted = df.sort_values("test_post_clipped")
    print(df_sorted[["candidate", "test_post_clipped", "test_post_unclipped", "test_post_wape", "test_post_rmse", "test_blend_clipped", "duration_sec"]].to_string(index=False))
    print(f"\nSaved screening results to {out_csv.name}")


if __name__ == "__main__":
    main()
