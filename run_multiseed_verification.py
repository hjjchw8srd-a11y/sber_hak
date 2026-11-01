#!/usr/bin/env python3
"""
run_multiseed_verification.py

Evaluates multi-seed stability (Seeds 42, 43, 44) for:
1) Timestral Univariate LoRA
2) Timestral D_category + C_network

On Fold 4 (Origin 89) under the exact 600-step harmonized protocol.
Checks whether the substantial covariate gain (~0.0056 Clipped) exceeds seed variance (sigma_seed).
Saves results to experiments/timestral_feature_research/multiseed_verification_fold4.csv.
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

    fold = 4  # Origin 89
    total_steps = 600
    eval_steps = 150
    seeds = [43, 44]  # Seed 42 is already completed in the primary 5-fold evaluation

    experiments = [
        ("univariate", []),
        ("selected_dc", ["D_category", "C_network"]),
    ]

    out_csv = ROOT / "experiments" / "timestral_feature_research" / "multiseed_verification_fold4.csv"
    records = []

    # Load existing Seed 42 results if available
    for m_type, groups in experiments:
        json_path = ROOT / "experiments" / "timestral_feature_research" / f"harmonized_{m_type}_fold{fold}_results.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                res42 = json.load(f)
            p_post = res42.get("standalone_post") or res42.get("test_post", {})
            v_unc = res42.get("val_unclipped") or res42.get("best_val_unclipped")
            records.append({
                "model": m_type,
                "seed": 42,
                "fold": fold,
                "best_step": res42["best_step"],
                "val_unclipped": v_unc,
                "test_clipped": p_post["clipped"],
                "test_unclipped": p_post["unclipped"],
                "test_wape": p_post["wape"],
                "test_rmse": p_post["rmse"],
                "duration_sec": res42.get("train_duration_sec", res42.get("duration_sec", 0.0)),
            })
            print(f"Loaded existing Seed 42 for {m_type}: Clipped = {p_post['clipped']:.6f}")

    print("\n" + "=" * 90)
    print(f"RUNNING MULTI-SEED VERIFICATION (SEEDS 43, 44) ON FOLD {fold}")
    print("=" * 90, flush=True)

    for m_type, groups in experiments:
        for seed in seeds:
            print(f"\n>>> Running Model: {m_type} | Seed: {seed} | Fold: {fold} | Steps: {total_steps} <<<", flush=True)
            t0 = time.time()
            torch.cuda.empty_cache()

            res = train_harmonized_model(
                engine=engine,
                groups=groups,
                fold=fold,
                total_steps=total_steps,
                eval_steps=eval_steps,
                microbatch=8,
                grad_accum=4,
                seed=seed,
                save_tag=f"multiseed_{m_type}_s{seed}",
            )
            elapsed = time.time() - t0

            records.append({
                "model": m_type,
                "seed": seed,
                "fold": fold,
                "best_step": res["best_step"],
                "val_unclipped": res["best_val_unclipped"],
                "test_clipped": res["test_post"]["clipped"],
                "test_unclipped": res["test_post"]["unclipped"],
                "test_wape": res["test_post"]["wape"],
                "test_rmse": res["test_post"]["rmse"],
                "duration_sec": elapsed,
            })
            print(f"  >>> {m_type} Seed {seed}: Test Clipped = {res['test_post']['clipped']:.6f} | Val Unc = {res['best_val_unclipped']:.6f}\n", flush=True)

            # Checkpoint save dataframe after each run
            df_temp = pd.DataFrame(records)
            df_temp.to_csv(out_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("MULTI-SEED VERIFICATION SUMMARY:")
    print("=" * 90)
    print(df.to_string(index=False))

    # Compute summary stats
    print("\n" + "-" * 50)
    print("AGGREGATED SEED STATISTICS:")
    print("-" * 50)
    for m_type in ["univariate", "selected_dc"]:
        sub = df[df["model"] == m_type]
        mean_clp = sub["test_clipped"].mean()
        std_clp = sub["test_clipped"].std()
        mean_wape = sub["test_wape"].mean()
        std_wape = sub["test_wape"].std()
        print(f"{m_type.upper():15s} | Clipped: {mean_clp:.6f} +/- {std_clp:.6f} | WAPE: {mean_wape:.3f}% +/- {std_wape:.3f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
