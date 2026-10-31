#!/usr/bin/env python3
"""
experiment_adaptive_timestral_blend.py

Evaluates adaptive blending schemes between Champion continuous raw predictions
and Timestral continuous predictions across the 5 competition folds.

Evaluation Protocol:
- Selection Folds: [4, 3, 2] (Origins 89, 98, 107) -> Primary model selection criterion: Selection Unclipped Ratio
- Check Folds: [1, 0] (Origins 116, 125) -> Validation confirmation
- Full 5-Folds: [4, 3, 2, 1, 0] -> Overall reference

Outputs:
1. Summary table sorted by Selection Unclipped Ratio (primary criterion)
2. Summary table sorted by Selection Clipped Score
3. Full per-fold metrics table for all 13 schemes
4. Saves summary JSON and per-fold CSV to experiments/timestral_feature_research/
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import WeeklyMatrices
from adaptive_blending_rules import (
    KEYS,
    FOLDS,
    FOLD_ORIGINS,
    SELECTION_FOLDS,
    CHECK_FOLDS,
    apply_p,
    compute_pre_cutoff_features,
    SCHEMES,
)


def calc_scores(m_pred: dict, m_base: dict) -> tuple[float, float]:
    clipped = 0.0
    unclipped = 0.0
    for k, w in METRIC_WEIGHTS.items():
        base_val = m_base[k]
        pred_val = m_pred[k]
        ratio = pred_val / base_val if base_val > 0 else 1.0
        clipped += w * min(max(ratio, 0.0), 1.0)
        unclipped += w * ratio
    return clipped, unclipped


def load_fold_data(matrices: WeeklyMatrices, pair_meta: pd.DataFrame, fold: int) -> pd.DataFrame:
    origin = FOLD_ORIGINS[fold]
    oof_dir = ROOT / "experiments" / "champion_oof"
    champ_path = oof_dir / f"champion_scale_1m_fold_{fold}.csv.gz"
    tim_path = (
        ROOT / "experiments" / f"timestral_lora_fold{fold}" / "keyed_predictions" / "pilot"
        / f"timestral_209m_lora_fold{fold}_pilot_keyed.csv.gz"
    )

    df_champ = pd.read_csv(champ_path)
    df_tim = pd.read_csv(tim_path)

    merged = pd.merge(
        df_champ[KEYS + ["raw_champion", "y_true", "mean13"]],
        df_tim[KEYS + ["y_pred_continuous"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 79101

    # Horizon mapping (1..9)
    weeks_ordered = sorted(merged["week"].unique())
    week_to_h = {w: i + 1 for i, w in enumerate(weeks_ordered)}
    merged["horizon"] = merged["week"].map(week_to_h)

    # Series pre-cutoff features
    df_feats = compute_pre_cutoff_features(matrices, origin, pair_meta)
    df_feats = df_feats.rename(columns={"mean13": "mean13_feat"})
    merged = pd.merge(merged, df_feats, on=["warehouse_name", "article"], how="left")

    return merged


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 125)
    print("STRICT TIMESTRAL ADAPTIVE BLENDING EXPERIMENT & AUDIT")
    print("Primary Selection Criterion: Average Unclipped Ratio on Selection Folds [4, 3, 2]")
    print("Validation Check: Folds [1, 0]")
    print("=" * 125)

    data_path = ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    pair_meta = data[["warehouse_name", "article", "category_lvl_1", "category_lvl_2"]].drop_duplicates(subset=["warehouse_name", "article"])
    matrices = WeeklyMatrices(data)

    print("\nLoading fold data...")
    fold_data = {}
    for fold in FOLDS:
        print(f"  Fold {fold} (Origin {FOLD_ORIGINS[fold]})...")
        fold_data[fold] = load_fold_data(matrices, pair_meta, fold)

    results = []

    print("\nEvaluating all 13 schemes across 5 folds...")
    for name, scheme_fn in SCHEMES.items():
        for fold in FOLDS:
            df = fold_data[fold]
            w = scheme_fn(df)

            raw_c = df["raw_champion"].values.astype(np.float64)
            raw_t = df["y_pred_continuous"].values.astype(np.float64)

            # Continuous blend
            raw_blend = (1.0 - w) * raw_c + w * raw_t
            pred = apply_p(raw_blend, 1.05, 0.65)

            truth = df[KEYS + ["y_true"]].rename(columns={"y_true": "quantity"})
            base = df[KEYS + ["mean13"]].rename(columns={"mean13": "quantity"})
            m_base = compute_all_metrics_fast(truth, base)

            df_pred = truth[KEYS].copy()
            df_pred["quantity"] = pred
            m_pred = compute_all_metrics_fast(truth, df_pred)

            c_score, u_ratio = calc_scores(m_pred, m_base)

            results.append({
                "scheme": name,
                "fold": fold,
                "stage": "Selection" if fold in SELECTION_FOLDS else "Check",
                "clipped_score": c_score,
                "unclipped_ratio": u_ratio,
                "smape": m_pred["smape"],
                "wape": m_pred["wape"],
                "rmse": m_pred["rmse"],
                "bias": m_pred["bias"],
                "rmsle": m_pred["rmsle"],
                "mean_weight": float(w.mean()),
            })

    res_df = pd.DataFrame(results)

    # 1. Compute Aggregates per Scheme
    summary_rows = []
    for name in SCHEMES.keys():
        s_df = res_df[res_df["scheme"] == name]
        sel_df = s_df[s_df["stage"] == "Selection"]
        chk_df = s_df[s_df["stage"] == "Check"]

        summary_rows.append({
            "scheme": name,
            "mean_w": float(s_df["mean_weight"].mean()),
            "sel_unclipped": float(sel_df["unclipped_ratio"].mean()),
            "sel_clipped": float(sel_df["clipped_score"].mean()),
            "chk_unclipped": float(chk_df["unclipped_ratio"].mean()),
            "chk_clipped": float(chk_df["clipped_score"].mean()),
            "full_unclipped": float(s_df["unclipped_ratio"].mean()),
            "full_clipped": float(s_df["clipped_score"].mean()),
            "full_wape": float(s_df["wape"].mean()),
            "full_rmse": float(s_df["rmse"].mean()),
            "full_smape": float(s_df["smape"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Table 1: Sorted by Selection Unclipped Ratio
    print("\n" + "=" * 125)
    print("TABLE 1: SCHEMES SORTED STRICTLY BY SELECTION UNCLIPPED RATIO (PRIMARY PROTOCOL CRITERION)")
    print("=" * 125)
    df_by_sel_unclipped = summary_df.sort_values("sel_unclipped")
    print(f"{'Scheme':<45} {'Mean W':<7} {'Sel Unclipped':<14} {'Sel Clipped':<12} {'Chk Unclipped':<14} {'Chk Clipped':<12} {'Full Unclipped':<15} {'Full Clipped':<13}")
    print("-" * 125)
    for _, r in df_by_sel_unclipped.iterrows():
        print(f"{r['scheme']:<45} {r['mean_w']:<7.3f} {r['sel_unclipped']:<14.6f} {r['sel_clipped']:<12.6f} {r['chk_unclipped']:<14.6f} {r['chk_clipped']:<12.6f} {r['full_unclipped']:<15.6f} {r['full_clipped']:<13.6f}")

    # Table 2: Sorted by Selection Clipped Score
    print("\n" + "=" * 125)
    print("TABLE 2: SCHEMES SORTED STRICTLY BY SELECTION CLIPPED SCORE")
    print("=" * 125)
    df_by_sel_clipped = summary_df.sort_values("sel_clipped")
    print(f"{'Scheme':<45} {'Mean W':<7} {'Sel Clipped':<12} {'Sel Unclipped':<14} {'Chk Clipped':<12} {'Chk Unclipped':<14} {'Full Clipped':<13} {'Full Unclipped':<15}")
    print("-" * 125)
    for _, r in df_by_sel_clipped.iterrows():
        print(f"{r['scheme']:<45} {r['mean_w']:<7.3f} {r['sel_clipped']:<12.6f} {r['sel_unclipped']:<14.6f} {r['chk_clipped']:<12.6f} {r['chk_unclipped']:<14.6f} {r['full_clipped']:<13.6f} {r['full_unclipped']:<15.6f}")

    # Table 3: Full Per-Fold Pivot Tables
    print("\n" + "=" * 125)
    print("TABLE 3: PER-FOLD CLIPPED SCORE (ALL 5 FOLDS)")
    print("=" * 125)
    pivot_clipped = res_df.pivot(index="fold", columns="scheme", values="clipped_score")
    pivot_clipped.loc["Mean_Sel [4,3,2]"] = pivot_clipped.loc[[4, 3, 2]].mean()
    pivot_clipped.loc["Mean_Chk [1,0]"] = pivot_clipped.loc[[1, 0]].mean()
    pivot_clipped.loc["Mean_Full [0..4]"] = pivot_clipped.loc[[4, 3, 2, 1, 0]].mean()
    print(pivot_clipped.T[["Mean_Sel [4,3,2]", "Mean_Chk [1,0]", "Mean_Full [0..4]", 4, 3, 2, 1, 0]].to_string())

    print("\n" + "=" * 125)
    print("TABLE 4: PER-FOLD UNCLIPPED RATIO (ALL 5 FOLDS)")
    print("=" * 125)
    pivot_unclipped = res_df.pivot(index="fold", columns="scheme", values="unclipped_ratio")
    pivot_unclipped.loc["Mean_Sel [4,3,2]"] = pivot_unclipped.loc[[4, 3, 2]].mean()
    pivot_unclipped.loc["Mean_Chk [1,0]"] = pivot_unclipped.loc[[1, 0]].mean()
    pivot_unclipped.loc["Mean_Full [0..4]"] = pivot_unclipped.loc[[4, 3, 2, 1, 0]].mean()
    print(pivot_unclipped.T[["Mean_Sel [4,3,2]", "Mean_Chk [1,0]", "Mean_Full [0..4]", 4, 3, 2, 1, 0]].to_string())

    # Winners identification
    winner_by_unclipped = df_by_sel_unclipped.iloc[0]["scheme"]
    winner_by_clipped = df_by_sel_clipped.iloc[0]["scheme"]

    print("\n" + "=" * 90)
    print("SELECTION CRITERIA WINNERS (STRICT SELECTION FOLDS [4, 3, 2] ONLY):")
    print(f"  1. Winner strictly by Selection Unclipped Ratio: {winner_by_unclipped}")
    print(f"     -> Sel Unclipped: {df_by_sel_unclipped.iloc[0]['sel_unclipped']:.6f} | Sel Clipped: {df_by_sel_unclipped.iloc[0]['sel_clipped']:.6f}")
    print(f"  2. Winner strictly by Selection Clipped Score:   {winner_by_clipped}")
    print(f"     -> Sel Clipped: {df_by_sel_clipped.iloc[0]['sel_clipped']:.6f} | Sel Unclipped: {df_by_sel_clipped.iloc[0]['sel_unclipped']:.6f}")
    print("=" * 90)

    # Save outputs
    out_dir = ROOT / "experiments" / "timestral_feature_research"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "adaptive_blend_experiment_summary.json"
    per_fold_csv = out_dir / "adaptive_blend_per_fold_metrics.csv"
    per_fold_json = out_dir / "adaptive_blend_per_fold_metrics.json"

    export_payload = {
        "selection_winner_unclipped": winner_by_unclipped,
        "selection_winner_clipped": winner_by_clipped,
        "sorted_by_selection_unclipped": df_by_sel_unclipped.to_dict(orient="records"),
        "sorted_by_selection_clipped": df_by_sel_clipped.to_dict(orient="records"),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    res_df.to_csv(per_fold_csv, index=False)
    with open(per_fold_json, "w", encoding="utf-8") as f:
        json.dump(res_df.to_dict(orient="records"), f, indent=2)

    print(f"\nSaved summary to: {summary_file.name}")
    print(f"Saved per-fold metrics to: {per_fold_csv.name} and {per_fold_json.name}")


if __name__ == "__main__":
    main()
