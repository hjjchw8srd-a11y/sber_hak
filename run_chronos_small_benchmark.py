"""
Benchmark Chronos-2 Small (pretrained weights in experiments/urgent_models/chronos2_small)
Strictly adheres to:
- BaseChronosPipeline.from_pretrained(local_files_only=True, device_map='cuda')
- Folds 0 (origin 125) and 3 (origin 98), horizon=9 weeks
- Keyed predictions saved: week, warehouse_name, article, y_true, y_pred_*
- Standard metrics: SMAPE, WAPE, RMSE, Bias, RMSLE, clipped_score, unclipped_ratio vs Mean13
- Head-to-head comparison with Control B, Champion Scale_1M_Dual_0.5, LGBM, TFT, and NHITS
- Blend analysis with champion ensemble
"""

import os
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
    print("CHRONOS-2 SMALL STANDALONE BENCHMARK (FOLDS 0 & 3)")
    print("=" * 80)

    weights_dir = ROOT / 'experiments' / 'urgent_models' / 'chronos2_small'
    safetensors_file = weights_dir / 'model.safetensors'
    if not safetensors_file.exists():
        raise FileNotFoundError(f"Missing weights file at {safetensors_file}")

    file_bytes = safetensors_file.read_bytes()
    weights_sha256 = hashlib.sha256(file_bytes).hexdigest()
    weights_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"Loaded weights sha256: {weights_sha256}")
    print(f"Weights size: {weights_size_mb:.2f} MB")

    out_dir = ROOT / 'experiments' / 'chronos_benchmark'
    keyed_dir = out_dir / 'keyed_predictions'
    keyed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print("\nLoading Chronos-2 Small via BaseChronosPipeline...")
    torch.manual_seed(42)
    t0_load = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(
        str(weights_dir),
        device_map='cuda',
        local_files_only=True
    )
    load_time = time.perf_counter() - t0_load
    print(f"Loaded pipeline class: {type(pipe).__name__} in {load_time:.2f} s")
    
    quantiles = list(pipe.model.chronos_config.quantiles)
    print(f"Available quantiles ({len(quantiles)}): {quantiles}")
    qi_med = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    assert abs(quantiles[qi_med] - 0.5) < 1e-5, f"Quantile 0.5 not found: {quantiles}"
    print(f"Median quantile index: {qi_med} (quantile {quantiles[qi_med]})")

    # 2. Load Dataset
    print("\nLoading weekly dataset...")
    ds_path = ROOT / 'weekly_data' / 'weekly_dataset.csv.gz'
    data = pd.read_csv(ds_path, parse_dates=['week'])
    panel = data.pivot(index='week', columns=['warehouse_name', 'article'], values='quantity').sort_index()
    n_wks, n_pairs = panel.shape
    print(f"Panel shape: {n_wks} weeks, {n_pairs} warehouse-article series")
    assert n_pairs == 8789, f"Expected 8789 pairs, got {n_pairs}"

    # Load baseline blend files for direct fold comparison
    ctrl_oof_dir = ROOT / 'experiments' / 'history_window_audit' / 'keyed_oof'
    all_stock_oof_dir = ROOT / 'experiments' / 'model_improvements' / 'all_stock_5folds' / 'keyed_predictions'

    all_records = []
    blending_records = []

    # 3. Evaluate Folds 0 and 3
    # We test both context_length=52 and context_length=104
    contexts = [52, 104]

    for ctx in contexts:
        print(f"\n" + "=" * 60)
        print(f"EVALUATING CONTEXT LENGTH: {ctx} WEEKS")
        print("=" * 60)

        for fold, origin in ORIGINS.items():
            print(f"\n--- Fold {fold} (Origin {origin}, Target: {panel.index[origin].strftime('%Y-%m-%d')} to {panel.index[origin + 8].strftime('%Y-%m-%d')}) ---")
            
            # Ground truth and Mean13
            val_weeks = panel.index[origin : origin + HORIZON].strftime('%Y-%m-%d')
            pairs_df = panel.columns.to_frame(index=False)
            
            # Extract historical matrix: (n_pairs, 1, ctx)
            ctx_actual = min(origin, ctx)
            hist_slice = panel.iloc[origin - ctx_actual : origin].values.T # (8789, ctx_actual)
            x_tensor = torch.tensor(hist_slice[:, None, :], dtype=torch.float32)
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_inf_start = time.perf_counter()
            with torch.inference_mode():
                predictions = pipe.predict(
                    x_tensor,
                    prediction_length=HORIZON,
                    batch_size=256,
                    context_length=ctx_actual,
                    cross_learning=False
                )
            torch.cuda.synchronize()
            t_inf_elapsed = time.perf_counter() - t_inf_start
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Inference completed in {t_inf_elapsed:.2f} s. Peak VRAM: {peak_vram_mb:.1f} MB")

            # Stack predictions: shape (n_pairs, len(quantiles), 9)
            pred_arr = np.stack([p[0].detach().cpu().numpy() for p in predictions])
            assert pred_arr.shape == (n_pairs, len(quantiles), HORIZON), f"Unexpected shape {pred_arr.shape}"

            # Build long dataframe for evaluation
            # Order: for each week h in 0..8, all pairs
            records_truth = []
            records_mean13 = []
            records_preds_q = {q: [] for q in quantiles}

            for h, w_str in enumerate(val_weeks):
                actual_y = panel.iloc[origin + h].values
                # mean13 is mean of previous 13 weeks before origin
                m13 = panel.iloc[origin - 13 : origin].mean().values
                records_truth.append(actual_y)
                records_mean13.append(m13)
                for q_idx, q in enumerate(quantiles):
                    records_preds_q[q].append(pred_arr[:, q_idx, h])

            # Flatten to 1D arrays aligned with keys
            keys_expanded = pd.concat([pairs_df] * HORIZON, ignore_index=True)
            keys_expanded.insert(0, 'week', np.repeat(val_weeks, n_pairs))

            y_true = np.concatenate(records_truth)
            y_mean13 = np.concatenate(records_mean13)

            truth_df = keys_expanded.copy()
            truth_df['quantity'] = y_true

            mean13_df = keys_expanded.copy()
            mean13_df['quantity'] = y_mean13

            m_mean = compute_all_metrics_fast(truth_df, mean13_df)

            # Extract median
            y_med_raw = np.maximum(0.0, np.concatenate(records_preds_q[quantiles[qi_med]]))
            
            # Quantiles q0.3 and q0.4
            qi_30 = int(np.argmin(np.abs(np.asarray(quantiles) - 0.3)))
            qi_40 = int(np.argmin(np.abs(np.asarray(quantiles) - 0.4)))
            y_q30_raw = np.maximum(0.0, np.concatenate(records_preds_q[quantiles[qi_30]]))
            y_q40_raw = np.maximum(0.0, np.concatenate(records_preds_q[quantiles[qi_40]]))

            # Candidate variants
            variants = {
                'raw_continuous': y_med_raw,
                'rint': np.rint(y_med_raw),
                'legacy_p': apply_p(y_med_raw, 1.05, 0.65),
                'thresh_0.75_rint': np.rint(np.where(y_med_raw < 0.75, 0.0, y_med_raw)),
                'thresh_0.50_rint': np.rint(np.where(y_med_raw < 0.50, 0.0, y_med_raw)),
                'q30_raw': y_q30_raw,
                'q30_rint': np.rint(y_q30_raw),
                'q40_raw': y_q40_raw,
                'q40_rint': np.rint(y_q40_raw),
            }

            # Save keyed predictions
            keyed_df = keys_expanded.copy()
            keyed_df['y_true'] = y_true
            keyed_df['unconstrained_demand_13'] = y_mean13
            keyed_df['y_pred_continuous'] = y_med_raw
            keyed_df['y_pred_rint'] = np.rint(y_med_raw)
            keyed_df['y_pred_legacy_p'] = apply_p(y_med_raw, 1.05, 0.65)
            keyed_df['y_pred_thresh75_rint'] = np.rint(np.where(y_med_raw < 0.75, 0.0, y_med_raw))
            keyed_df['y_pred_q30_rint'] = np.rint(y_q30_raw)
            keyed_df['y_pred_q40_rint'] = np.rint(y_q40_raw)

            keyed_filename = f"chronos2_small_ctx{ctx}_fold_{fold}_keyed.csv.gz"
            keyed_path = keyed_dir / keyed_filename
            keyed_df.to_csv(keyed_path, index=False, compression='gzip')
            print(f"Saved keyed predictions to {keyed_path.name} ({len(keyed_df)} rows)")

            # Compute metrics for each variant
            for v_name, v_pred in variants.items():
                pred_df = keys_expanded.copy()
                pred_df['quantity'] = v_pred
                m = compute_all_metrics_fast(truth_df, pred_df)
                sc_clip, sc_unclip = calc_scores(m, m_mean)
                row = {
                    'model': 'Chronos2_Small',
                    'context': ctx,
                    'fold': fold,
                    'origin': origin,
                    'variant': v_name,
                    'clipped_score': sc_clip,
                    'unclipped_ratio': sc_unclip,
                    'smape': m['smape'],
                    'wape': m['wape'],
                    'rmse': m['rmse'],
                    'bias': m['bias'],
                    'rmsle': m['rmsle'],
                    'inference_sec': t_inf_elapsed,
                    'peak_vram_mb': peak_vram_mb,
                    'keyed_file': keyed_filename,
                }
                all_records.append(row)
                if v_name in ['raw_continuous', 'rint', 'legacy_p', 'thresh_0.75_rint', 'q40_rint']:
                    print(f"  [{v_name:18s}] Clipped: {sc_clip:.6f} | Unclipped: {sc_unclip:.6f} | SMAPE: {m['smape']:.2f}% | WAPE: {m['wape']:.2f}% | RMSE: {m['rmse']:.4f} | Bias: {m['bias']:.2f}%")

            # 4. Direct Blending with Control B and Scale_1M_Dual_0.5
            ctrl_file = ctrl_oof_dir / f"blend_b_fold_{fold}.csv.gz"
            if ctrl_file.exists():
                df_ctrl = pd.read_csv(ctrl_file)
                df_ctrl['week'] = pd.to_datetime(df_ctrl['week']).dt.strftime('%Y-%m-%d')
                ctrl_col = 'final_prediction' if 'final_prediction' in df_ctrl.columns else 'blend_b'
                merged_ctrl = keyed_df.merge(df_ctrl[KEYS + [ctrl_col]].rename(columns={ctrl_col: 'blend_b'}), on=KEYS, how='left')
                
                y_ctrl = merged_ctrl['blend_b'].values
                pred_ctrl_df = keys_expanded.copy(); pred_ctrl_df['quantity'] = y_ctrl
                m_ctrl = compute_all_metrics_fast(truth_df, pred_ctrl_df)
                sc_ctrl_clip, sc_ctrl_unclip = calc_scores(m_ctrl, m_mean)

                # Also load All-Stock / Scale_1M_Dual_0.5 if available
                all_stock_file = all_stock_oof_dir / f"predictions_fold_{fold}.csv.gz"
                y_as = None
                sc_as_clip, sc_as_unclip = None, None
                if all_stock_file.exists():
                    df_as = pd.read_csv(all_stock_file)
                    df_as['week'] = pd.to_datetime(df_as['week']).dt.strftime('%Y-%m-%d')
                    merged_as = keyed_df.merge(df_as[KEYS + ['candidate_weights', 'all_stock_ensemble']], on=KEYS, how='left')
                    y_as = merged_as['all_stock_ensemble'].values
                    pred_as_df = keys_expanded.copy(); pred_as_df['quantity'] = y_as
                    m_as = compute_all_metrics_fast(truth_df, pred_as_df)
                    sc_as_clip, sc_as_unclip = calc_scores(m_as, m_mean)

                for w in [0.02, 0.05, 0.10, 0.15, 0.20]:
                    # Blend with Control B
                    blend_raw = (1.0 - w) * y_ctrl + w * y_med_raw
                    blend_rint = np.rint(blend_raw)
                    pred_b_df = keys_expanded.copy(); pred_b_df['quantity'] = blend_raw
                    pred_br_df = keys_expanded.copy(); pred_br_df['quantity'] = blend_rint
                    m_b = compute_all_metrics_fast(truth_df, pred_b_df)
                    m_br = compute_all_metrics_fast(truth_df, pred_br_df)
                    sc_b_clip, sc_b_unclip = calc_scores(m_b, m_mean)
                    sc_br_clip, sc_br_unclip = calc_scores(m_br, m_mean)

                    b_row = {
                        'base': 'Control_B',
                        'context': ctx,
                        'fold': fold,
                        'weight_chronos': w,
                        'blend_clipped_raw': sc_b_clip,
                        'blend_unclipped_raw': sc_b_unclip,
                        'blend_clipped_rint': sc_br_clip,
                        'blend_unclipped_rint': sc_br_unclip,
                        'smape_rint': m_br['smape'],
                        'wape_rint': m_br['wape'],
                        'rmse_rint': m_br['rmse'],
                        'base_clipped': sc_ctrl_clip,
                        'delta_clipped_rint': sc_br_clip - sc_ctrl_clip,
                    }
                    blending_records.append(b_row)

                    # Blend with Champion All-Stock Ensemble if available
                    if y_as is not None:
                        as_raw = (1.0 - w) * y_as + w * y_med_raw
                        as_rint = np.rint(as_raw)
                        pred_asb_df = keys_expanded.copy(); pred_asb_df['quantity'] = as_raw
                        pred_asbr_df = keys_expanded.copy(); pred_asbr_df['quantity'] = as_rint
                        m_asb = compute_all_metrics_fast(truth_df, pred_asb_df)
                        m_asbr = compute_all_metrics_fast(truth_df, pred_asbr_df)
                        sc_asb_clip, sc_asb_unclip = calc_scores(m_asb, m_mean)
                        sc_asbr_clip, sc_asbr_unclip = calc_scores(m_asbr, m_mean)

                        as_b_row = {
                            'base': 'Scale_1M_Dual_0.5',
                            'context': ctx,
                            'fold': fold,
                            'weight_chronos': w,
                            'blend_clipped_raw': sc_asb_clip,
                            'blend_unclipped_raw': sc_asb_unclip,
                            'blend_clipped_rint': sc_asbr_clip,
                            'blend_unclipped_rint': sc_asbr_unclip,
                            'smape_rint': m_asbr['smape'],
                            'wape_rint': m_asbr['wape'],
                            'rmse_rint': m_asbr['rmse'],
                            'base_clipped': sc_as_clip,
                            'delta_clipped_rint': sc_asbr_clip - sc_as_clip,
                        }
                        blending_records.append(as_b_row)

    # 5. Summary and Comparison
    df_results = pd.DataFrame(all_records)
    df_results.to_csv(out_dir / 'chronos2_small_benchmark_summary.csv', index=False)

    df_blends = pd.DataFrame(blending_records)
    df_blends.to_csv(out_dir / 'chronos2_small_blending_results.csv', index=False)

    # Calculate 2-fold averages
    avg_df = df_results.groupby(['model', 'context', 'variant']).agg({
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

    print("\n" + "=" * 90)
    print("CHRONOS-2 SMALL BENCHMARK SUMMARY (AVERAGE ACROSS FOLDS 0 & 3)")
    print("=" * 90)
    print(avg_df.to_string(index=False))

    # Save manifest
    manifest = {
        'model_name': 'Chronos-2 Small',
        'source_repo': 'experiments/urgent_models/chronos2_small',
        'pipeline_class': 'BaseChronosPipeline (Chronos2Pipeline)',
        'weights_sha256': weights_sha256,
        'weights_size_mb': weights_size_mb,
        'evaluated_folds': [0, 3],
        'horizon': 9,
        'summary_results': avg_df.to_dict(orient='records'),
    }
    with open(out_dir / 'chronos2_small_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll results saved to {out_dir}")


if __name__ == '__main__':
    main()
