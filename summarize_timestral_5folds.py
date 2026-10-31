import json
from pathlib import Path
import numpy as np
import pandas as pd

rows = []
for f in [4, 3, 2, 1, 0]:
    p = Path(f"experiments/timestral_lora_fold{f}/logs/pilot/timestral_lora_fold{f}_pilot_results.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        ext = d.get("external_evaluation") or d.get("external_evaluation_fold4") or {}
        if "fine_tuned" in ext:
            zs_rint = ext.get("zero_shot", {}).get("rint", {})
            zs_leg = ext.get("zero_shot", {}).get("legacy_p", {})
            ft_rint = ext.get("fine_tuned", {}).get("rint", {})
            ft_leg = ext.get("fine_tuned", {}).get("legacy_p", {})
        else:
            zs_rint = ext.get("zero_shot_rint", {})
            zs_leg = ext.get("zero_shot_legacy_p", {})
            ft_rint = ext.get("fine_tuned_rint", {})
            ft_leg = ext.get("fine_tuned_legacy_p", {})
        best_step = d.get("best_checkpoint", {}).get("step")

        rows.append({
            "fold": f,
            "origin": d["origin"],
            "best_step": best_step,
            "zs_rint_clipped": zs_rint.get("clipped_score", np.nan),
            "zs_rint_unclipped": zs_rint.get("unclipped_ratio", np.nan),
            "zs_rint_wape": zs_rint.get("wape", np.nan),
            "zs_leg_clipped": zs_leg.get("clipped_score", np.nan),
            "zs_leg_unclipped": zs_leg.get("unclipped_ratio", np.nan),
            "zs_leg_wape": zs_leg.get("wape", np.nan),
            "ft_rint_clipped": ft_rint["clipped_score"],
            "ft_rint_unclipped": ft_rint["unclipped_ratio"],
            "ft_rint_smape": ft_rint["smape"],
            "ft_rint_wape": ft_rint["wape"],
            "ft_rint_rmse": ft_rint["rmse"],
            "ft_rint_bias": ft_rint["bias"],
            "ft_rint_rmsle": ft_rint["rmsle"],
            "ft_leg_clipped": ft_leg["clipped_score"],
            "ft_leg_unclipped": ft_leg["unclipped_ratio"],
            "ft_leg_smape": ft_leg["smape"],
            "ft_leg_wape": ft_leg["wape"],
            "ft_leg_rmse": ft_leg["rmse"],
            "ft_leg_bias": ft_leg["bias"],
            "ft_leg_rmsle": ft_leg["rmsle"],
        })

df = pd.DataFrame(rows)
print("=" * 110)
print("TIMESTRAL-209M LORA 5-FOLD RESULTS SUMMARY")
print("=" * 110)
print(df[["fold", "best_step", "ft_rint_clipped", "ft_rint_unclipped", "ft_rint_smape", "ft_rint_wape", "ft_rint_rmse", "ft_rint_bias", "ft_rint_rmsle"]].to_string(index=False))

print("\nLEGACY P POST-PROCESSING:")
print(df[["fold", "best_step", "ft_leg_clipped", "ft_leg_unclipped", "ft_leg_smape", "ft_leg_wape", "ft_leg_rmse", "ft_leg_bias", "ft_leg_rmsle"]].to_string(index=False))

print("\nMEANS ACROSS 5 FOLDS:")
print("Fine-Tuned Rint:")
print(f"  Clipped:   {df['ft_rint_clipped'].mean():.6f}")
print(f"  Unclipped: {df['ft_rint_unclipped'].mean():.6f}")
print(f"  SMAPE:     {df['ft_rint_smape'].mean():.3f}%")
print(f"  WAPE:      {df['ft_rint_wape'].mean():.3f}%")
print(f"  RMSE:      {df['ft_rint_rmse'].mean():.4f}")
print(f"  Bias:      {df['ft_rint_bias'].mean():.2f}%")
print(f"  RMSLE:     {df['ft_rint_rmsle'].mean():.4f}")

print("\nFine-Tuned Legacy P:")
print(f"  Clipped:   {df['ft_leg_clipped'].mean():.6f}")
print(f"  Unclipped: {df['ft_leg_unclipped'].mean():.6f}")
print(f"  SMAPE:     {df['ft_leg_smape'].mean():.3f}%")
print(f"  WAPE:      {df['ft_leg_wape'].mean():.3f}%")
print(f"  RMSE:      {df['ft_leg_rmse'].mean():.4f}")
print(f"  Bias:      {df['ft_leg_bias'].mean():.2f}%")
print(f"  RMSLE:     {df['ft_leg_rmsle'].mean():.4f}")

print("\nZero-Shot Comparison:")
print(df[["fold", "zs_rint_clipped", "zs_rint_unclipped", "zs_rint_wape", "ft_rint_clipped", "ft_rint_unclipped", "ft_rint_wape"]].to_string(index=False))
