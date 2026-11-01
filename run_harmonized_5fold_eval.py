#!/usr/bin/env python3
"""
run_harmonized_5fold_eval.py

Executes the harmonized 600-step training protocol for:
- Univariate LoRA
- Selected Covariates (D_category + C_network)
across folds 3, 2, 1, 0 (Fold 4 already completed and verified).

Ensures:
- Identical train/val/test splits with zero leakage
- 600-step budget, validation every 150 steps
- Selection of best checkpoint strictly by internal validation
- Raw OOF export with SHA256 calculation
- Standalone RINT and Standalone Legacy P reporting
"""

from __future__ import annotations

import sys
import time
import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="*", type=int, default=[3, 2, 1, 0])
    parser.add_argument("--models", nargs="*", default=["univariate", "selected_dc"])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = TimestralFeatureEngine(ROOT / "weekly_data" / "weekly_dataset.csv.gz")

    model_configs = {
        "univariate": {
            "groups": [],
            "tag": "univariate",
        },
        "selected_dc": {
            "groups": ["D_category", "C_network"],
            "tag": "selected_dc",
        },
    }

    print("=" * 100)
    print(f"STARTING HARMONIZED 5-FOLD RUNS: Folds {args.folds} | Models: {args.models}")
    print("=" * 100, flush=True)

    for fold in args.folds:
        for model_key in args.models:
            cfg = model_configs[model_key]
            print(f"\n>>> Running Fold {fold} | Model: {model_key} | Groups: {cfg['groups']} ...", flush=True)
            t0 = time.time()
            torch.cuda.empty_cache()

            res = train_harmonized_model(
                engine=engine,
                groups=cfg["groups"],
                fold=fold,
                total_steps=args.steps,
                eval_steps=args.eval_steps,
                microbatch=8,
                grad_accum=4,
                seed=args.seed,
                save_tag=f"{cfg['tag']}",
            )
            elapsed = time.time() - t0
            print(f">>> Completed Fold {fold} | {model_key} in {elapsed:.1f}s\n", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
