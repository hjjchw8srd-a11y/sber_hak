"""
Chronos-2 Small Fine-Tuning Benchmark.
Strict Protocol:
- Initial checkpoint: experiments/urgent_models/chronos2_small (fresh pretrained weights).
- prediction_length = 9, context_length = 52.
- learning_rate = 1e-6, AdamW, 300 steps budget.
- For each fold (Folds 0 & 3): strictly reload fresh original pretrained weights.
- All training sliding windows and internal validation strictly BEFORE evaluation origin.
- Evaluates on Folds 0 & 3, saves keyed predictions, all quantiles, continuous and rint metrics.
"""

import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sber_hak'))
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from chronos import BaseChronosPipeline

KEYS = ['week', 'warehouse_name', 'article']
ORIGINS = {0: 125, 3: 98}
HORIZON = 9
CONTEXT = 52
BATCH_SIZE = 64
LR = 1e-6
STEPS = 300


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
    print("CHRONOS-2 SMALL FINE-TUNING BENCHMARK (FOLDS 0 & 3)")
    print(f"Settings: lr={LR}, steps={STEPS}, context={CONTEXT}, horizon={HORIZON}, batch_size={BATCH_SIZE}")
    print("=" * 80)

    weights_dir = ROOT / 'experiments' / 'urgent_models' / 'chronos2_small'
    out_dir = ROOT / 'experiments' / 'chronos_finetune'
    keyed_dir = out_dir / 'keyed_predictions'
    keyed_dir.mkdir(parents=True, exist_ok=True)

    # Load Weekly Dataset
    print("\nLoading weekly dataset...")
    data = pd.read_csv(ROOT / 'weekly_data' / 'weekly_dataset.csv.gz', parse_dates=['week'])
    panel = data.pivot(index='week', columns=['warehouse_name', 'article'], values='quantity').sort_index()
    n_wks, n_pairs = panel.shape
    print(f"Panel: {n_wks} weeks, {n_pairs} series")
    pairs_df = panel.columns.to_frame(index=False)

    all_records = []

    for fold, origin in ORIGINS.items():
        print(f"\n" + "=" * 60)
        print(f"FOLD {fold} (Origin {origin}): RELOADING FRESH PRETRAINED CHECKPOINT")
        print("=" * 60)

        # 1. Strictly load fresh pretrained pipeline for this fold
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        pipe = BaseChronosPipeline.from_pretrained(
            str(weights_dir),
            device_map='cuda',
            local_files_only=True
        )
        model = pipe.model
        quantiles = list(model.chronos_config.quantiles)
        qi_med = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
        print(f"Pretrained weights loaded fresh. Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 2. Strict Training / Validation Split BEFORE origin
        # External test: [origin, origin + HORIZON)
        # Internal validation window: [origin - HORIZON, origin)
        # Training windows: origins w in [CONTEXT, origin - HORIZON]
        train_max_origin = origin - HORIZON
        assert train_max_origin >= CONTEXT, f"Not enough history for fold {fold}: {train_max_origin} < {CONTEXT}"
        train_origins = list(range(CONTEXT, train_max_origin + 1))
        print(f"Train origins range: {train_origins[0]} to {train_origins[-1]} ({len(train_origins)} window positions)")
        print(f"Evaluation window strictly at origin {origin} ({panel.index[origin].strftime('%Y-%m-%d')} to {panel.index[origin + HORIZON - 1].strftime('%Y-%m-%d')})")

        # 3. Fine-tuning Loop (300 steps)
        model.train()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        history_matrix = panel.iloc[:origin].values # shape (origin, n_pairs)

        t0_train = time.perf_counter()
        running_loss = 0.0

        for step in range(1, STEPS + 1):
            # Sample random origins and random series
            sampled_w = np.random.choice(train_origins, size=BATCH_SIZE)
            sampled_s = np.random.choice(n_pairs, size=BATCH_SIZE)

            # Build batch context and future_target
            batch_ctx = np.zeros((BATCH_SIZE, CONTEXT), dtype=np.float32)
            batch_tgt = np.zeros((BATCH_SIZE, 16), dtype=np.float32) # patch size 16

            for b in range(BATCH_SIZE):
                w = sampled_w[b]
                s = sampled_s[b]
                batch_ctx[b] = history_matrix[w - CONTEXT : w, s]
                batch_tgt[b, :HORIZON] = history_matrix[w : w + HORIZON, s]

            ctx_t = torch.tensor(batch_ctx, device='cuda', dtype=torch.float32)
            tgt_t = torch.tensor(batch_tgt, device='cuda', dtype=torch.float32)
            mask_t = torch.zeros((BATCH_SIZE, 16), device='cuda', dtype=torch.float32)
            mask_t[:, :HORIZON] = 1.0

            optimizer.zero_grad()
            out = model(context=ctx_t, future_target=tgt_t, future_target_mask=mask_t, num_output_patches=1)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0 or step == STEPS:
                avg_l = running_loss / (50 if step % 50 == 0 else (step % 50))
                running_loss = 0.0
                print(f"  Step {step:3d}/{STEPS} | Train Quantile Loss: {avg_l:.4f}")

        train_elapsed = time.perf_counter() - t0_train
        print(f"Fine-tuning completed in {train_elapsed:.2f} s")

        # 4. Evaluation on External Validation Period
        model.eval()
        val_weeks = panel.index[origin : origin + HORIZON].strftime('%Y-%m-%d')
        ctx_actual = CONTEXT
        hist_slice = panel.iloc[origin - ctx_actual : origin].values.T # (8789, 52)
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

        # pred_arr: (n_pairs, len(quantiles), 9)
        pred_arr = np.stack([p[0].detach().cpu().numpy() for p in predictions])

        # Ground truth and Mean13
        keys_expanded = pd.concat([pairs_df] * HORIZON, ignore_index=True)
        keys_expanded.insert(0, 'week', np.repeat(val_weeks, n_pairs))

        records_truth = []
        records_mean13 = []
        records_preds_q = {q: [] for q in quantiles}

        for h, w_str in enumerate(val_weeks):
            actual_y = panel.iloc[origin + h].values
            m13 = panel.iloc[origin - 13 : origin].mean().values
            records_truth.append(actual_y)
            records_mean13.append(m13)
            for q_idx, q in enumerate(quantiles):
                records_preds_q[q].append(pred_arr[:, q_idx, h])

        y_true = np.concatenate(records_truth)
        y_mean13 = np.concatenate(records_mean13)

        truth_df = keys_expanded.copy(); truth_df['quantity'] = y_true
        mean13_df = keys_expanded.copy(); mean13_df['quantity'] = y_mean13
        m_mean = compute_all_metrics_fast(truth_df, mean13_df)

        y_med_raw = np.maximum(0.0, np.concatenate(records_preds_q[quantiles[qi_med]]))
        y_rint = np.rint(y_med_raw)
        y_leg = apply_p(y_med_raw, 1.05, 0.65)
        y_thresh75 = np.rint(np.where(y_med_raw < 0.75, 0.0, y_med_raw))

        # Save keyed predictions
        keyed_df = keys_expanded.copy()
        keyed_df['y_true'] = y_true
        keyed_df['unconstrained_demand_13'] = y_mean13
        keyed_df['y_pred_continuous'] = y_med_raw
        keyed_df['y_pred_rint'] = y_rint
        keyed_df['y_pred_legacy_p'] = y_leg
        keyed_df['y_pred_thresh75_rint'] = y_thresh75

        # Also save all quantiles
        for q in quantiles:
            keyed_df[f'q_{q:g}'] = np.concatenate(records_preds_q[q])

        keyed_filename = f"chronos2_finetuned_fold_{fold}_keyed.csv.gz"
        keyed_path = keyed_dir / keyed_filename
        keyed_df.to_csv(keyed_path, index=False, compression='gzip')
        print(f"Saved keyed predictions: {keyed_filename} ({len(keyed_df)} rows)")

        variants = {
            'raw_continuous': y_med_raw,
            'rint': y_rint,
            'legacy_p': y_leg,
            'thresh_0.75_rint': y_thresh75,
        }

        for v_name, v_pred in variants.items():
            pred_df = keys_expanded.copy(); pred_df['quantity'] = v_pred
            m = compute_all_metrics_fast(truth_df, pred_df)
            sc_clip, sc_unclip = calc_scores(m, m_mean)
            row = {
                'model': 'Chronos2_Small_FineTuned_300Steps',
                'context': CONTEXT,
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
                'train_sec': train_elapsed,
                'inference_sec': t_inf_elapsed,
                'peak_vram_mb': peak_vram_mb,
                'keyed_file': keyed_filename,
            }
            all_records.append(row)
            print(f"  [{v_name:18s}] Clipped: {sc_clip:.6f} | Unclipped: {sc_unclip:.6f} | SMAPE: {m['smape']:.2f}% | WAPE: {m['wape']:.2f}% | RMSE: {m['rmse']:.4f}")

    df_res = pd.DataFrame(all_records)
    df_res.to_csv(out_dir / 'chronos2_finetuned_benchmark_summary.csv', index=False)

    print("\n" + "=" * 90)
    print("FINE-TUNED CHRONOS-2 SMALL SUMMARY (AVERAGE OVER FOLDS 0 & 3)")
    print("=" * 90)
    summary_avg = df_res.groupby('variant').agg({
        'clipped_score': 'mean',
        'unclipped_ratio': 'mean',
        'smape': 'mean',
        'wape': 'mean',
        'rmse': 'mean',
        'bias': 'mean',
        'rmsle': 'mean',
        'train_sec': 'mean',
        'inference_sec': 'mean',
    }).reset_index().sort_values('clipped_score')
    print(summary_avg.to_string(index=False))


if __name__ == '__main__':
    main()
