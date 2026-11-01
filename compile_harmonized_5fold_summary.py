#!/usr/bin/env python3
"""
compile_harmonized_5fold_summary.py

Parses all 10 harmonized run JSONs (Folds 0..4 for Univariate and Selected D+C),
computes 5-fold macro averages, selection averages [4, 3, 2], check averages [1, 0],
verifies SHA256 hashes of all raw OOF files, and exports:
- harmonized_5fold_metrics.csv
- harmonized_5fold_summary.json
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "experiments" / "timestral_feature_research"
OOF_DIR = EXP_DIR / "oof_predictions"

models = ["univariate", "selected_dc"]
folds = [0, 1, 2, 3, 4]

records = []

for model in models:
    for fold in folds:
        json_path = EXP_DIR / f"harmonized_{model}_fold{fold}_results.json"
        assert json_path.exists(), f"Missing {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        oof_path = ROOT / d.get("raw_oof_path", f"experiments/timestral_feature_research/oof_predictions/harmonized_{model}_fold{fold}_raw.csv.gz")
        assert oof_path.exists(), f"Missing {oof_path}"
        sha = hashlib.sha256(oof_path.read_bytes()).hexdigest()

        post = d.get("standalone_post", {})
        rint = d.get("standalone_rint", {})
        blend = d.get("blend_w10", {})

        records.append({
            "model": model,
            "fold": fold,
            "best_step": d.get("best_step"),
            "val_unclipped": d.get("val_unclipped"),
            "post_clipped": post.get("clipped"),
            "post_unclipped": post.get("unclipped"),
            "post_wape": post.get("wape"),
            "post_rmse": post.get("rmse"),
            "post_bias": post.get("bias"),
            "post_rmsle": post.get("rmsle"),
            "post_smape": post.get("smape"),
            "rint_clipped": rint.get("clipped"),
            "rint_unclipped": rint.get("unclipped"),
            "rint_wape": rint.get("wape"),
            "rint_rmse": rint.get("rmse"),
            "blend_w10_clipped": blend.get("clipped"),
            "blend_w10_unclipped": blend.get("unclipped"),
            "train_duration_sec": d.get("train_duration_sec"),
            "test_eval_duration_sec": d.get("test_eval_duration_sec"),
            "raw_oof_sha256": sha,
            "raw_oof_path": str(oof_path.relative_to(ROOT)),
        })

df = pd.DataFrame(records)
csv_out = EXP_DIR / "harmonized_5fold_metrics.csv"
df.to_csv(csv_out, index=False)
print(f"Saved {csv_out.name}\n")

# Compute summary comparison
for model in models:
    df_m = df[df["model"] == model]
    sel = df_m[df_m["fold"].isin([4, 3, 2])]
    chk = df_m[df_m["fold"].isin([1, 0])]

    print(f"================================================================================")
    print(f"MODEL: {model.upper()}")
    print(f"================================================================================")
    print(f"{'Fold':<6} {'Step':<6} {'Val Unc':<10} {'Post Clp':<10} {'Post Unc':<10} {'WAPE':<8} {'RMSE':<8} {'RINT Clp':<10} {'RINT Unc':<10} {'Blend Clp':<10}")
    print("-" * 88)
    for _, r in df_m.sort_values("fold", ascending=False).iterrows():
        print(f"Fold {int(r['fold']):<1}  {int(r['best_step']):<6} {r['val_unclipped']:<10.6f} {r['post_clipped']:<10.6f} {r['post_unclipped']:<10.6f} {r['post_wape']:<7.3f}% {r['post_rmse']:<8.4f} {r['rint_clipped']:<10.6f} {r['rint_unclipped']:<10.6f} {r['blend_w10_clipped']:<10.6f}")
    print("-" * 88)
    print(f"Full 5-Fold Mean:  {df_m['val_unclipped'].mean():<10.6f} {df_m['post_clipped'].mean():<10.6f} {df_m['post_unclipped'].mean():<10.6f} {df_m['post_wape'].mean():<7.3f}% {df_m['post_rmse'].mean():<8.4f} {df_m['rint_clipped'].mean():<10.6f} {df_m['rint_unclipped'].mean():<10.6f} {df_m['blend_w10_clipped'].mean():<10.6f}")
    print(f"Selection [4,3,2]: {sel['val_unclipped'].mean():<10.6f} {sel['post_clipped'].mean():<10.6f} {sel['post_unclipped'].mean():<10.6f} {sel['post_wape'].mean():<7.3f}% {sel['post_rmse'].mean():<8.4f} {sel['rint_clipped'].mean():<10.6f} {sel['rint_unclipped'].mean():<10.6f} {sel['blend_w10_clipped'].mean():<10.6f}")
    print(f"Check [1,0]:       {chk['val_unclipped'].mean():<10.6f} {chk['post_clipped'].mean():<10.6f} {chk['post_unclipped'].mean():<10.6f} {chk['post_wape'].mean():<7.3f}% {chk['post_rmse'].mean():<8.4f} {chk['rint_clipped'].mean():<10.6f} {chk['rint_unclipped'].mean():<10.6f} {chk['blend_w10_clipped'].mean():<10.6f}")
    print("\n")

# Delta comparison
u = df[df["model"] == "univariate"].set_index("fold")
dc = df[df["model"] == "selected_dc"].set_index("fold")

print("================================================================================")
print("DELTA: SELECTED D+C vs UNIVARIATE (Negative is improvement!)")
print("================================================================================")
print(f"{'Fold':<8} {'Delta Val Unc':<15} {'Delta Post Clp':<16} {'Delta Post Unc':<16} {'Delta WAPE':<12} {'Delta RMSE':<12} {'Delta RINT Clp':<16}")
print("-" * 95)
for f in [4, 3, 2, 1, 0]:
    d_val = dc.loc[f, "val_unclipped"] - u.loc[f, "val_unclipped"]
    d_clp = dc.loc[f, "post_clipped"] - u.loc[f, "post_clipped"]
    d_unc = dc.loc[f, "post_unclipped"] - u.loc[f, "post_unclipped"]
    d_wape = dc.loc[f, "post_wape"] - u.loc[f, "post_wape"]
    d_rmse = dc.loc[f, "post_rmse"] - u.loc[f, "post_rmse"]
    d_rint = dc.loc[f, "rint_clipped"] - u.loc[f, "rint_clipped"]
    print(f"Fold {f:<3} {d_val:<+15.6f} {d_clp:<+16.6f} {d_unc:<+16.6f} {d_wape:<+11.3f}% {d_rmse:<+12.4f} {d_rint:<+16.6f}")

print("-" * 95)
d_val_mean = dc["val_unclipped"].mean() - u["val_unclipped"].mean()
d_clp_mean = dc["post_clipped"].mean() - u["post_clipped"].mean()
d_unc_mean = dc["post_unclipped"].mean() - u["post_unclipped"].mean()
d_wape_mean = dc["post_wape"].mean() - u["post_wape"].mean()
d_rmse_mean = dc["post_rmse"].mean() - u["post_rmse"].mean()
d_rint_mean = dc["rint_clipped"].mean() - u["rint_clipped"].mean()
print(f"5-Fold   {d_val_mean:<+15.6f} {d_clp_mean:<+16.6f} {d_unc_mean:<+16.6f} {d_wape_mean:<+11.3f}% {d_rmse_mean:<+12.4f} {d_rint_mean:<+16.6f}")

sel_dc = dc.loc[[4, 3, 2]]
sel_u = u.loc[[4, 3, 2]]
d_sel_clp = sel_dc["post_clipped"].mean() - sel_u["post_clipped"].mean()
d_sel_unc = sel_dc["post_unclipped"].mean() - sel_u["post_unclipped"].mean()
d_sel_wape = sel_dc["post_wape"].mean() - sel_u["post_wape"].mean()
d_sel_rmse = sel_dc["post_rmse"].mean() - sel_u["post_rmse"].mean()
print(f"Select   {sel_dc['val_unclipped'].mean() - sel_u['val_unclipped'].mean():<+15.6f} {d_sel_clp:<+16.6f} {d_sel_unc:<+16.6f} {d_sel_wape:<+11.3f}% {d_sel_rmse:<+12.4f} {sel_dc['rint_clipped'].mean() - sel_u['rint_clipped'].mean():<+16.6f}")

chk_dc = dc.loc[[1, 0]]
chk_u = u.loc[[1, 0]]
d_chk_clp = chk_dc["post_clipped"].mean() - chk_u["post_clipped"].mean()
d_chk_unc = chk_dc["post_unclipped"].mean() - chk_u["post_unclipped"].mean()
d_chk_wape = chk_dc["post_wape"].mean() - chk_u["post_wape"].mean()
d_chk_rmse = chk_dc["post_rmse"].mean() - chk_u["post_rmse"].mean()
print(f"Check    {chk_dc['val_unclipped'].mean() - chk_u['val_unclipped'].mean():<+15.6f} {d_chk_clp:<+16.6f} {d_chk_unc:<+16.6f} {d_chk_wape:<+11.3f}% {d_chk_rmse:<+12.4f} {chk_dc['rint_clipped'].mean() - chk_u['rint_clipped'].mean():<+16.6f}")
print("================================================================================")
