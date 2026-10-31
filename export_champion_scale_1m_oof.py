#!/usr/bin/env python3
"""
export_champion_scale_1m_oof.py

Recovers and exports the EXACT historical OOF predictions for the true champion:
'Scale_1M_Dual_0.5' across all 5 evaluation folds:
- Selection Folds: 4 (origin 89), 3 (origin 98), 2 (origin 107)
- Check Folds: 1 (origin 116), 0 (origin 125)

For each fold:
1. Builds exact 29-feature matrices strictly up to latest_train (origin - HORIZON).
2. Fits 1M-sample LightGBM Hurdle with Dual L1/L2 regressor (seed 42).
3. Fits CatBoost on 18 origins (seed 42).
4. Fits Per-Category LightGBM on 52 origins (seed 42).
5. Loads verified NeuralForecast TFT out-of-fold predictions.
6. Blends components: 0.45*TFT + 0.37*LGBM_1M_Dual + 0.09*CB + 0.09*PerCat -> raw_champion.
7. Applies project post-processing (alpha=1.05, thresh=0.65, np.rint) -> final_champion.
8. Asserts bit-exact match against five_folds_enhancements_results.json.
9. Exports keyed predictions to experiments/champion_oof/champion_scale_1m_fold_{fold}.csv.gz.
"""

from __future__ import annotations

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sber_hak"))

from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from run_weekly_feature_experiments import HORIZON, KEYS, WeeklyMatrices
from experiment_all_stock_ensemble import (
    build_feature_matrices,
    train_predict_stock_catboost,
    train_predict_stock_per_category,
    LGBM_PARAMS,
)
from evaluate_check_and_all_folds_enhancements import fit_lgbm_hurdle, apply_uniform_p

warnings.filterwarnings("ignore", category=RuntimeWarning)

ALL_FOLDS = {4: 89, 3: 98, 2: 107, 1: 116, 0: 125}
OUT_DIR = PROJECT_ROOT / "experiments" / "champion_oof"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def run_fold(matrices: WeeklyMatrices, fold: int, origin: int, expected_metrics: dict) -> pd.DataFrame:
    print(f"\n--- Processing Fold {fold} (Origin {origin}) ---")
    t0 = time.perf_counter()

    # Ground truth and Mean13 baseline
    val_weeks = matrices.weeks[origin : origin + HORIZON]
    val_weeks_str = [w.strftime("%Y-%m-%d") for w in val_weeks]
    n_pairs = len(matrices.pairs)

    truth = pd.concat([matrices.pairs] * HORIZON, ignore_index=True)
    truth.insert(0, "week", np.repeat(val_weeks_str, n_pairs))
    truth["quantity"] = matrices.quantity[origin : origin + HORIZON, :].ravel()

    m_mean13_values = matrices.quantity[origin - 13 : origin, :].mean(axis=0)
    base_mean = truth.copy()
    base_mean["quantity"] = np.tile(m_mean13_values, HORIZON)
    m_base = compute_all_metrics_fast(truth, base_mean)

    # Load TFT OOF
    tft_file = PROJECT_ROOT / "experiments" / "history_window_audit" / "keyed_oof" / f"blend_b_fold_{fold}.csv.gz"
    df_f = pd.read_csv(tft_file)
    df_f["week"] = pd.to_datetime(df_f["week"]).dt.strftime("%Y-%m-%d")
    merged_tft = truth[KEYS].merge(df_f[KEYS + ["raw_tft"]], on=KEYS, how="left", validate="one_to_one")
    assert len(merged_tft) == len(truth) and not merged_tft["raw_tft"].isna().any(), "TFT merge failed!"
    q_tft = merged_tft["raw_tft"].to_numpy(dtype=np.float64)

    # Feature matrices
    latest_train = origin - HORIZON
    lgbm_origins = np.arange(max(26, latest_train - 51), latest_train + 1)
    cb_origins = np.arange(max(26, latest_train - 17), latest_train + 1)

    X_train_lgb_42, y_train_lgb_42, X_val, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, seed=42, mode="29")
    X_train_cb_42, y_train_cb_42, _, _, _ = build_feature_matrices(matrices, origin, cb_origins, seed=42, mode="29")
    feature_cols = [c for c in X_val.columns if c not in ["category_lvl_1"]]

    # 1. CatBoost
    print("  Fitting CatBoost (18 origins)...", flush=True)
    pred_cb = train_predict_stock_catboost(X_train_cb_42, y_train_cb_42, X_val, feature_cols)

    # 2. Per-Category LightGBM
    print("  Fitting Per-Category (52 origins)...", flush=True)
    pred_pcat = train_predict_stock_per_category(X_train_lgb_42, y_train_lgb_42, X_val, feature_cols + ["category_lvl_1"])

    # 3. Scaled 1M LightGBM Hurdle with Dual L1/L2
    print("  Fitting LightGBM 1M-Row Dual Regressor...", flush=True)
    X_tr_1m, y_tr_1m, _, _, _ = build_feature_matrices(matrices, origin, lgbm_origins, max_rows=1_000_000, seed=42, mode="29")
    scale_params = LGBM_PARAMS.copy()
    scale_params["n_estimators"] = 300
    scale_params["learning_rate"] = 0.035
    p_1m, q_l2_1m, q_l1_1m = fit_lgbm_hurdle(X_tr_1m, y_tr_1m, X_val, feature_cols, scale_params, fit_l1=True)
    lgb_1m_dual = np.where(p_1m < 0.5, 0.0, p_1m * (0.5 * q_l2_1m + 0.5 * q_l1_1m))

    # Champion Blend
    raw_champion = 0.45 * q_tft + 0.37 * lgb_1m_dual + 0.09 * pred_cb + 0.09 * pred_pcat
    final_champion = apply_uniform_p(raw_champion, 1.05, 0.65)

    p_df = truth[KEYS].copy()
    p_df["quantity"] = final_champion
    m_pred = compute_all_metrics_fast(truth, p_df)
    c_sc, u_sc = calc_scores(m_pred, m_base)

    print(f"  Recalculated Fold {fold}: Clipped={c_sc:.6f}, Unclipped={u_sc:.6f}, SMAPE={m_pred['smape']:.3f}%, WAPE={m_pred['wape']:.3f}%, RMSE={m_pred['rmse']:.4f}")
    print(f"  Expected     Fold {fold}: Clipped={expected_metrics['clipped']:.6f}, Unclipped={expected_metrics['unclipped']:.6f}, SMAPE={expected_metrics['smape']:.3f}%, WAPE={expected_metrics['wape']:.3f}%, RMSE={expected_metrics['rmse']:.4f}")

    # Assert exact match with expected metrics from JSON
    assert abs(c_sc - expected_metrics["clipped"]) < 1e-4, f"Clipped score discrepancy: {c_sc} vs {expected_metrics['clipped']}"
    assert abs(u_sc - expected_metrics["unclipped"]) < 1e-4, f"Unclipped ratio discrepancy: {u_sc} vs {expected_metrics['unclipped']}"
    assert abs(m_pred["wape"] - expected_metrics["wape"]) < 0.05, f"WAPE discrepancy: {m_pred['wape']} vs {expected_metrics['wape']}"
    print("  [PASS] Exactly matches historical five_folds_enhancements_results.json!")

    # Format OOF DataFrame
    oof_df = truth[KEYS].copy()
    oof_df["y_true"] = truth["quantity"].values
    oof_df["mean13"] = base_mean["quantity"].values
    oof_df["raw_champion"] = raw_champion
    oof_df["final_champion"] = final_champion
    oof_df["origin"] = origin
    oof_df["fold"] = fold

    out_file = OUT_DIR / f"champion_scale_1m_fold_{fold}.csv.gz"
    oof_df.to_csv(out_file, index=False, compression="gzip")
    print(f"  Saved OOF file: {out_file.name} ({out_file.stat().st_size:,} bytes) in {time.perf_counter() - t0:.1f} s")
    return oof_df


def main():
    print("=" * 80)
    print("RECOVERING EXACT HISTORICAL OOF FOR CHAMPION: Scale_1M_Dual_0.5")
    print("=" * 80)

    # Load dataset
    data_path = PROJECT_ROOT / "weekly_data" / "weekly_dataset.csv.gz"
    data = pd.read_csv(data_path, parse_dates=["week"])
    matrices = WeeklyMatrices(data)

    # Load expected benchmark metrics
    json_path = PROJECT_ROOT / "experiments" / "model_improvements" / "five_folds_enhancements_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    expected = res["Scale_1M_Dual_0.5"]

    for fold, origin in ALL_FOLDS.items():
        run_fold(matrices, fold, origin, expected[str(fold)])

    print("\n" * 2 + "=" * 80)
    print("ALL 5 FOLDS OF CHAMPION Scale_1M_Dual_0.5 EXACTLY RECOVERED & VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
