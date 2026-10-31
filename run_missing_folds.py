#!/usr/bin/env python3
"""
run_missing_folds.py

Executes Timestral-209M LoRA training and evaluation across the missing folds:
- Fold 3 (origin 98)
- Fold 2 (origin 107)
- Fold 1 (origin 116)
(Fold 4 and Fold 0 were already trained and verified).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

def run_fold(fold: int):
    print("=" * 80)
    print(f"STARTING TIMESTRAL-209M LORA RUN FOR FOLD {fold} AT {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    t0 = time.perf_counter()
    cmd = [
        PYTHON,
        str(ROOT / "run_timestral_lora_fold.py"),
        "--fold", str(fold),
        "--run-id", "pilot",
        "--steps", "600",
        "--eval-steps", "150",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Fold {fold} failed with exit code {res.returncode}")
    elapsed = time.perf_counter() - t0
    print(f"\n[SUCCESS] Fold {fold} completed in {elapsed:.1f} s ({elapsed / 60:.2f} min)\n")

def main():
    start_total = time.perf_counter()
    for f in [3, 2, 1]:
        run_fold(f)
    total_elapsed = time.perf_counter() - start_total
    print("=" * 80)
    print(f"ALL 3 MISSING FOLDS (3, 2, 1) COMPLETED IN {total_elapsed / 60:.2f} MINUTES")
    print("=" * 80)

if __name__ == "__main__":
    main()
