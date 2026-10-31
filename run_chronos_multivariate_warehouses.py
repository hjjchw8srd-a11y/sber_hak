"""
Multivariate Cross-Warehouse Chronos-2 Small Benchmark
Grouping: 11 warehouses of the same article as 11 variates of a multivariate time series.
Model: Unmodified pretrained Chronos-2 Small (no fine-tuning, no extra features).
Folds: 0 (origin 125) and 3 (origin 98), Horizon 9 weeks.
Contexts: 52 and 104 weeks.
"""

import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sber_hak'))
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from chronos import BaseChronosPipeline

KEYS = ['week', 'warehouse_name', 'article']
ORIGINS = {0: 125, 3: 98}
HORIZON = 9


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


def apply_p(raw_arr: np.ndarray, alpha: float = 1.05, thresh: float = 0.65) -> np.ndarray:
    scaled = raw_arr * alpha
    gated = np.where(scaled < thresh, 0.0, scaled)
    return np.rint(gated).clip(0, None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("MULTIVARIATE CROSS-WAREHOUSE CHRONOS-2 SMALL BENCHMARK (FOLDS 0 & 3)")
    print("Grouping: 11 warehouses of the same article as 11 variates")
    print("=" * 80)

    weights_dir = ROOT / 'experiments' / 'urgent_models' / 'chronos2_small'
    out_dir = ROOT / 'experiments' / 'chronos_multivariate_benchmark'
    keyed_dir = out_dir / 'keyed_predictions'
    keyed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained Pipeline
    print("\nLoading Chronos-2 Small via BaseChronosPipeline...")
    torch.manual_seed(42)
    t0_load = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(
        str(weights_dir),
        device_map='cuda',
        local_files_only=True
    )
    load_time = time.perf_counter() - t0_load
    print(f"Loaded pipeline in {load_time:.2f} s")

    quantiles = list(pipe.model.chronos_config.quantiles)
    qi_med = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    print(f"Available quantiles ({len(quantiles)}): {quantiles}")
    print(f"Median quantile index: {qi_med} (value: {quantiles[qi_med]})")

    # 2. Load Weekly Dataset
    print("\nLoading weekly dataset...")
    data = pd.read_csv(ROOT / 'weekly_data' / 'weekly_dataset.csv.gz', parse_dates=['week'])
    
    # Pivot so columns are MultiIndex: (article, warehouse_name)
    panel = data.pivot(index='week', columns=['article', 'warehouse_name'], values='quantity').sort_index()
    n_wks, total_series = panel.shape
    articles = panel.columns.levels[0].unique()
    warehouses = panel.columns.levels[1].unique()
    n_articles = len(articles)
    n_warehouses = len(warehouses)
    print(f"Panel: {n_wks} weeks, {n_articles} unique articles, {n_warehouses} warehouses per article = {total_series} series")
    assert n_warehouses == 11, f"Expected 11 warehouses, got {n_warehouses}"
    assert n_articles == 799, f"Expected 799 articles, got {n_articles}"

    all_records = []
    contexts = [52, 104]

    for ctx in contexts:
        print(f"\n" + "=" * 60)
        print(f"MULTIVARIATE: EVALUATING CONTEXT LENGTH: {ctx} WEEKS")
        print("=" * 60)

        for fold, origin in ORIGINS.items():
            print(f"\n--- Fold {fold} (Origin {origin}) ---")
            val_weeks = panel.index[origin : origin + HORIZON].strftime('%Y-%m-%d')
            ctx_actual = min(origin, ctx)

            # Build 3D array: (n_articles=799, n_warehouses=11, ctx_actual)
            hist = panel.iloc[origin - ctx_actual : origin].values # (ctx_actual, 8789)
            # Reshape to (ctx_actual, 799, 11) and transpose to (799, 11, ctx_actual)
            x_3d = hist.reshape(ctx_actual, n_articles, n_warehouses).transpose(1, 2, 0)
            x_tensor = torch.tensor(x_3d, dtype=torch.float32)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.inference_mode():
                # Cross-learning between the 11 variates for each article!
                predictions = pipe.predict(
                    x_tensor,
                    prediction_length=HORIZON,
                    batch_size=64,
                    context_length=ctx_actual,
                    cross_learning=False # multivariate across variates inside each article
                )

            torch.cuda.synchronize()
            t_elapsed = time.perf_counter() - t_start
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Multivariate inference in {t_elapsed:.2f} s. Peak VRAM: {peak_vram_mb:.1f} MB")

            # Stack predictions: shape (799, 11, len(quantiles), 9)
            pred_arr = np.stack([p.detach().cpu().numpy() for p in predictions])
            assert pred_arr.shape == (n_articles, n_warehouses, len(quantiles), HORIZON)

            # Flatten to long DataFrame matching ground truth
            # We iterate week by week, and for each week, all articles and warehouses
            rows = []
            for h, w_str in enumerate(val_weeks):
                for a_idx, art in enumerate(articles):
                    for w_idx, wh in enumerate(warehouses):
                        q_med = pred_arr[a_idx, w_idx, qi_med, h]
                        rows.append({
                            'week': w_str,
                            'article': art,
                            'warehouse_name': wh,
                            'y_pred_continuous': float(np.maximum(0.0, q_med)),
                        })

            pred_df = pd.DataFrame(rows)
            pred_df['y_pred_rint'] = np.rint(pred_df['y_pred_continuous'])
            pred_df['y_pred_legacy_p'] = apply_p(pred_df['y_pred_continuous'].values, 1.05, 0.65)
            pred_df['y_pred_thresh75_rint'] = np.rint(np.where(pred_df['y_pred_continuous'].values < 0.75, 0.0, pred_df['y_pred_continuous'].values))

            # Ground truth and baseline Mean13
            val_df = data[(data['week'] >= panel.index[origin]) & (data['week'] <= panel.index[origin + 8])][KEYS + ['quantity']]
            mean13_data = data[(data['week'] >= panel.index[origin - 13]) & (data['week'] < panel.index[origin])]
            m13_calc = mean13_data.groupby(['warehouse_name', 'article'])['quantity'].mean().reset_index().rename(columns={'quantity': 'unconstrained_demand_13'})

            merged = val_df.merge(m13_calc, on=['warehouse_name', 'article'], how='left')
            merged['week'] = merged['week'].dt.strftime('%Y-%m-%d')
            merged = merged.merge(pred_df, on=KEYS, how='left')

            assert len(merged) == len(val_df) == 79101, f"Row mismatch: {len(merged)}"

            truth_df = merged[KEYS + ['quantity']].copy()
            base_df = merged[KEYS].copy(); base_df['quantity'] = merged['unconstrained_demand_13'].values
            m_mean = compute_all_metrics_fast(truth_df, base_df)

            # Save keyed predictions
            keyed_filename = f"chronos2_multivariate_ctx{ctx}_fold_{fold}_keyed.csv.gz"
            keyed_path = keyed_dir / keyed_filename
            save_cols = KEYS + ['quantity', 'unconstrained_demand_13', 'y_pred_continuous', 'y_pred_rint', 'y_pred_legacy_p', 'y_pred_thresh75_rint']
            merged[save_cols].rename(columns={'quantity': 'y_true'}).to_csv(keyed_path, index=False, compression='gzip')
            print(f"Saved keyed predictions: {keyed_filename} ({len(merged)} rows)")

            # Compute metrics for variants
            variants = {
                'raw_continuous': merged['y_pred_continuous'].values,
                'rint': merged['y_pred_rint'].values,
                'legacy_p': merged['y_pred_legacy_p'].values,
                'thresh_0.75_rint': merged['y_pred_thresh75_rint'].values,
            }

            for v_name, v_pred in variants.items():
                p_eval = truth_df.copy(); p_eval['quantity'] = v_pred
                m = compute_all_metrics_fast(truth_df, p_eval)
                sc_clip, sc_unclip = calc_scores(m, m_mean)
                row = {
                    'model': 'Chronos2_Multivariate_11Warehouses',
                    'context': ctx,
                    'fold': fold,
                    'variant': v_name,
                    'clipped_score': sc_clip,
                    'unclipped_ratio': sc_unclip,
                    'smape': m['smape'],
                    'wape': m['wape'],
                    'rmse': m['rmse'],
                    'bias': m['bias'],
                    'rmsle': m['rmsle'],
                    'inference_sec': t_elapsed,
                    'peak_vram_mb': peak_vram_mb,
                    'keyed_file': keyed_filename,
                }
                all_records.append(row)
                print(f"  [{v_name:18s}] Clipped: {sc_clip:.6f} | Unclipped: {sc_unclip:.6f} | SMAPE: {m['smape']:.2f}% | WAPE: {m['wape']:.2f}% | RMSE: {m['rmse']:.4f}")

    df_res = pd.DataFrame(all_records)
    df_res.to_csv(out_dir / 'chronos2_multivariate_summary.csv', index=False)

    print("\n" + "=" * 90)
    print("MULTIVARIATE CHRONOS-2 SUMMARY (AVERAGE OVER FOLDS 0 & 3)")
    print("=" * 90)
    summary_avg = df_res.groupby(['context', 'variant']).agg({
        'clipped_score': 'mean',
        'unclipped_ratio': 'mean',
        'smape': 'mean',
        'wape': 'mean',
        'rmse': 'mean',
        'bias': 'mean',
        'rmsle': 'mean',
        'inference_sec': 'mean',
        'peak_vram_mb': 'mean',
    }).reset_index().sort_values('clipped_score')
    print(summary_avg.to_string(index=False))


if __name__ == '__main__':
    main()
