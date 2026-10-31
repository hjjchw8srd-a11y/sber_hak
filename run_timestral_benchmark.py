"""
Timestral-209M Benchmark from Local Checkpoint.
Weights in: experiments/urgent_models/timestral_209m (config.json, model.safetensors)
Evaluates Contexts 52 and 104 weeks, Horizon 9 weeks, on Folds 0 and 3.
Batch size: 64 (memory safe).
Saves keyed predictions, quantiles, continuous and rint metrics.
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

from timestral import TimestralForecaster

KEYS = ['week', 'warehouse_name', 'article']
ORIGINS = {0: 125, 3: 98}
HORIZON = 9
BATCH_SIZE = 64


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
    print("TIMESTRAL-209M STANDALONE BENCHMARK (FOLDS 0 & 3)")
    print(f"Batch size: {BATCH_SIZE} (safe memory budget)")
    print("=" * 80)

    weights_dir = ROOT / 'experiments' / 'urgent_models' / 'timestral_209m'
    safetensors_file = weights_dir / 'model.safetensors'
    config_file = weights_dir / 'config.json'

    if not safetensors_file.exists() or not config_file.exists():
        print(f"\n[AWAITING WEIGHTS] Timestral-209M files not found in {weights_dir}")
        sys.exit(0)

    file_bytes = safetensors_file.read_bytes()
    weights_sha256 = hashlib.sha256(file_bytes).hexdigest()
    weights_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"Loaded weights sha256: {weights_sha256}")
    print(f"Weights size: {weights_size_mb:.2f} MB")

    out_dir = ROOT / 'experiments' / 'timestral_benchmark'
    keyed_dir = out_dir / 'keyed_predictions'
    keyed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print("\nLoading TimestralForecaster on CUDA...")
    torch.manual_seed(42)
    t0_load = time.perf_counter()
    model = TimestralForecaster.from_pretrained(
        str(weights_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    load_time = time.perf_counter() - t0_load
    print(f"Loaded model in {load_time:.2f} s")
    print(f"Quantile levels ({len(model.quantile_levels)}): {model.quantile_levels}")

    # 2. Load Dataset
    print("\nLoading weekly dataset...")
    data = pd.read_csv(ROOT / 'weekly_data' / 'weekly_dataset.csv.gz', parse_dates=['week'])
    panel = data.pivot(index='week', columns=['warehouse_name', 'article'], values='quantity').sort_index()
    n_wks, n_pairs = panel.shape
    print(f"Panel shape: {n_wks} weeks, {n_pairs} series")
    pairs_list = panel.columns.tolist() # list of (warehouse_name, article)

    all_records = []
    contexts = [52, 104]

    for ctx in contexts:
        print(f"\n" + "=" * 60)
        print(f"TIMESTRAL: EVALUATING CONTEXT LENGTH: {ctx} WEEKS")
        print("=" * 60)

        for fold, origin in ORIGINS.items():
            print(f"\n--- Fold {fold} (Origin {origin}) ---")
            val_weeks = panel.index[origin : origin + HORIZON]
            val_weeks_str = [w.strftime('%Y-%m-%d') for w in val_weeks]

            ctx_actual = min(origin, ctx)
            hist = panel.iloc[origin - ctx_actual : origin].values.T # (8789, ctx_actual)

            # Predict in conservative batches (batch_size=64)
            preds_med = []
            preds_q10 = []
            preds_q40 = []
            preds_q90 = []

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            for i in range(0, n_pairs, BATCH_SIZE):
                b_hist = hist[i : i + BATCH_SIZE]
                with torch.inference_mode():
                    fc = model.forecast(b_hist, prediction_length=HORIZON, clip_nonnegative=True)
                    med = fc.median.detach().cpu().numpy().reshape(len(b_hist), HORIZON)
                    q10 = fc.quantile(0.1).detach().cpu().numpy().reshape(len(b_hist), HORIZON)
                    q40 = fc.quantile(0.4).detach().cpu().numpy().reshape(len(b_hist), HORIZON)
                    q90 = fc.quantile(0.9).detach().cpu().numpy().reshape(len(b_hist), HORIZON)
                    preds_med.append(med)
                    preds_q10.append(q10)
                    preds_q40.append(q40)
                    preds_q90.append(q90)

            torch.cuda.synchronize()
            t_elapsed = time.perf_counter() - t_start
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Inference ({n_pairs} series) completed in {t_elapsed:.2f} s. Peak VRAM: {peak_vram_mb:.1f} MB")

            arr_med = np.vstack(preds_med) # (8789, 9)
            arr_q10 = np.vstack(preds_q10)
            arr_q40 = np.vstack(preds_q40)
            arr_q90 = np.vstack(preds_q90)

            # Build long prediction dataframe strictly aligned by key
            rows = []
            for p_idx, (wh, art) in enumerate(pairs_list):
                for h in range(HORIZON):
                    rows.append({
                        'week': val_weeks_str[h],
                        'warehouse_name': wh,
                        'article': art,
                        'y_pred_continuous': float(np.maximum(0.0, arr_med[p_idx, h])),
                        'y_pred_q10': float(np.maximum(0.0, arr_q10[p_idx, h])),
                        'y_pred_q40': float(np.maximum(0.0, arr_q40[p_idx, h])),
                        'y_pred_q90': float(np.maximum(0.0, arr_q90[p_idx, h])),
                    })

            pred_df = pd.DataFrame(rows)
            pred_df['y_pred_rint'] = np.rint(pred_df['y_pred_continuous'])
            pred_df['y_pred_legacy_p'] = apply_p(pred_df['y_pred_continuous'].values, 1.05, 0.65)
            pred_df['y_pred_thresh75_rint'] = np.rint(np.where(pred_df['y_pred_continuous'].values < 0.75, 0.0, pred_df['y_pred_continuous'].values))

            # Ground truth and baseline Mean13 from weekly_data
            val_df = data[(data['week'] >= val_weeks[0]) & (data['week'] <= val_weeks[-1])][KEYS + ['quantity']]
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
            keyed_filename = f"timestral_209m_ctx{ctx}_fold_{fold}_keyed.csv.gz"
            keyed_path = keyed_dir / keyed_filename
            save_cols = KEYS + ['quantity', 'unconstrained_demand_13', 'y_pred_continuous', 'y_pred_rint', 'y_pred_legacy_p', 'y_pred_thresh75_rint', 'y_pred_q10', 'y_pred_q40', 'y_pred_q90']
            merged[save_cols].rename(columns={'quantity': 'y_true'}).to_csv(keyed_path, index=False, compression='gzip')
            print(f"Saved keyed predictions to {keyed_filename} ({len(merged)} rows)")

            variants = {
                'raw_continuous': merged['y_pred_continuous'].values,
                'rint': merged['y_pred_rint'].values,
                'legacy_p': merged['y_pred_legacy_p'].values,
                'thresh_0.75_rint': merged['y_pred_thresh75_rint'].values,
                'q40_rint': np.rint(merged['y_pred_q40'].values),
            }

            for v_name, v_pred in variants.items():
                p_eval = truth_df.copy(); p_eval['quantity'] = v_pred
                m = compute_all_metrics_fast(truth_df, p_eval)
                sc_clip, sc_unclip = calc_scores(m, m_mean)
                row = {
                    'model': 'Timestral-209M',
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
    df_res.to_csv(out_dir / 'timestral_benchmark_summary.csv', index=False)

    print("\n" + "=" * 90)
    print("TIMESTRAL-209M BENCHMARK SUMMARY (AVERAGE OVER FOLDS 0 & 3)")
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
