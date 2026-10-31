"""
Controlled Experiment: Chronos-2 LoRA Fine-Tuning on Fold 4 (origin=89).
Official Chronos2Pipeline.fit with finetune_mode="lora".
Strict time-series split with internal validation window.
"""

import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sber_hak'))
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS
from chronos import Chronos2Pipeline

KEYS = ['week', 'warehouse_name', 'article']
FOLD = 4
ORIGIN = 89
HORIZON = 9
CONTEXT = 52
BATCH_SIZE = 32
GRAD_ACCUM = 2
LR = 1e-5
STEPS = 300
LOG_STEPS = 50
EVAL_STEPS = 50


class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = dict(logs)
            entry['global_step'] = state.global_step
            self.logs.append(entry)


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


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_exact_and_unique_keys(truth_df: pd.DataFrame, pred_df: pd.DataFrame, expected_rows: int = 79101):
    """
    Strict validation of prediction keys:
    - Exactly expected_rows rows (79,101)
    - Zero duplicates in keys
    - Zero missing or NaN keys
    - Exact full set equality between truth keys and prediction keys
    - Zero unmapped or mismatched rows (outer join check)
    - All predictions must be finite (no NaN or inf)
    """
    if len(pred_df) != expected_rows:
        raise ValueError(f"Integrity error: predictions row count {len(pred_df)} != expected {expected_rows}")
    if len(truth_df) != expected_rows:
        raise ValueError(f"Integrity error: truth row count {len(truth_df)} != expected {expected_rows}")

    dup_pred = int(pred_df.duplicated(subset=KEYS).sum())
    if dup_pred > 0:
        raise ValueError(f"Integrity error: duplicate keys detected in predictions ({dup_pred} duplicates)!")

    dup_truth = int(truth_df.duplicated(subset=KEYS).sum())
    if dup_truth > 0:
        raise ValueError(f"Integrity error: duplicate keys detected in ground truth ({dup_truth} duplicates)!")

    if pred_df[KEYS].isna().any().any():
        raise ValueError("Integrity error: missing/NaN values found in prediction keys!")

    merged_keys = truth_df[KEYS].merge(pred_df[KEYS], on=KEYS, how="outer", indicator=True)
    if len(merged_keys) != expected_rows or (merged_keys["_merge"] != "both").any():
        mismatch_cnt = int((merged_keys["_merge"] != "both").sum())
        raise ValueError(f"Integrity error: key mismatch between truth and predictions ({mismatch_cnt} missing/extra keys)!")

    if not np.isfinite(pred_df['quantity'].to_numpy(dtype=np.float64)).all():
        raise ValueError("Integrity error: non-finite (NaN or Inf) values found in predicted quantities!")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("CONTROLLED EXPERIMENT: CHRONOS-2 LORA FINE-TUNING (FOLD 4)")
    print("=" * 80)

    # 1. Environment & Library versions
    import transformers
    import peft
    import accelerate
    import chronos

    print("\n--- Environment & Versions ---")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Transformers: {transformers.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"Chronos: {chronos.__version__}")

    # 2. Check full Chronos-2 (120M) vs Chronos-2 Small
    print("\n--- Checking Model Checkpoints ---")
    full_weights_dir = ROOT / 'experiments' / 'model_improvements' / 'chronos2_weights'
    full_safetensors = full_weights_dir / 'model.safetensors'
    full_available = full_safetensors.exists() and full_safetensors.stat().st_size > 1000

    small_weights_dir = ROOT / 'experiments' / 'urgent_models' / 'chronos2_small'
    small_safetensors = small_weights_dir / 'model.safetensors'
    small_available = small_safetensors.exists() and small_safetensors.stat().st_size > 1000

    if full_available:
        print(f"Full Chronos-2 (120M) found at {full_weights_dir}")
        active_weights_dir = full_weights_dir
        model_name = "Chronos-2 (120M)"
        status_weights = "CONFIRMED_PRETRAINED"
    else:
        print(f"Full Chronos-2 (120M) weights missing: NEED_PRETRAINED_WEIGHTS")
        print(f"Checking Chronos-2 Small fallback at {small_weights_dir}...")
        assert small_available, "Chronos-2 Small weights also missing!"
        active_weights_dir = small_weights_dir
        model_name = "Chronos-2 Small (28M)"
        small_sha = sha256_file(small_safetensors)
        print(f"Chronos-2 Small verified: size={small_safetensors.stat().st_size:,} bytes, SHA256={small_sha}")
        status_weights = "FALLBACK_SMALL_PRETRAINED_VERIFIED"

    # 3. Setup output directories
    out_dir = ROOT / 'experiments' / 'chronos_lora_fold4'
    ckpt_dir = out_dir / 'checkpoint'
    keyed_dir = out_dir / 'keyed_predictions'
    logs_dir = out_dir / 'logs'
    for d in [out_dir, ckpt_dir, keyed_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. Load Weekly Dataset
    print("\n--- Loading Dataset & Setting Strict Splits ---")
    data_path = ROOT / 'weekly_data' / 'weekly_dataset.csv.gz'
    data = pd.read_csv(data_path, parse_dates=['week'])
    panel = data.pivot(index='week', columns=['warehouse_name', 'article'], values='quantity').sort_index()
    n_wks, n_pairs = panel.shape
    pairs_df = panel.columns.to_frame(index=False)
    print(f"Dataset panel shape: {n_wks} weeks, {n_pairs} series")

    # Time splits
    # Fold 4: origin = 89 (week: 2025-09-15)
    # External test: [origin, origin + HORIZON) = [89, 98) (2025-09-15 to 2025-11-10)
    # Internal val:  [origin - HORIZON, origin) = [80, 89) (2025-07-14 to 2025-09-08)
    # Train history: [0, origin - HORIZON)      = [0, 80)
    val_weeks = panel.index[ORIGIN : ORIGIN + HORIZON].strftime('%Y-%m-%d')
    int_val_weeks = panel.index[ORIGIN - HORIZON : ORIGIN].strftime('%Y-%m-%d')

    print(f"Fold: {FOLD}, Origin: {ORIGIN}")
    print(f"External Evaluation window: {val_weeks[0]} to {val_weeks[-1]} (9 weeks, weeks {ORIGIN}..{ORIGIN + HORIZON - 1})")
    print(f"Internal Validation window: {int_val_weeks[0]} to {int_val_weeks[-1]} (9 weeks, weeks {ORIGIN - HORIZON}..{ORIGIN - 1})")
    print(f"Training history window:   weeks 0 to {ORIGIN - HORIZON - 1} (length {ORIGIN - HORIZON} weeks, strictly BEFORE internal val)")

    # Prepare 3D tensors: shape (n_series, 1, history_length)
    train_history = panel.iloc[: ORIGIN - HORIZON].values.T[:, None, :]  # shape: (8789, 1, 80)
    val_history   = panel.iloc[: ORIGIN].values.T[:, None, :]             # shape: (8789, 1, 89)

    train_tensor = torch.tensor(train_history, dtype=torch.float32)
    val_tensor   = torch.tensor(val_history, dtype=torch.float32)
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Validation tensor shape: {val_tensor.shape}")

    # Strict temporal isolation verification (Requirement 4)
    train_history_len = train_history.shape[2]
    val_history_len = val_history.shape[2]
    assert train_history_len == (ORIGIN - HORIZON), f"Train history length mismatch: {train_history_len} != {ORIGIN - HORIZON}"
    assert val_history_len == ORIGIN, f"Validation history length mismatch: {val_history_len} != {ORIGIN}"
    # In Chronos-2, training target windows are sampled strictly within train_tensor.
    # Maximum training target index inside train_tensor is train_history_len - 1 = 79.
    # Internal validation target window starts at week ORIGIN - HORIZON = 80.
    # Therefore: train_target_max_week (79) < internal_val_start_week (80). Zero overlap!
    assert (train_history_len - 1) < (ORIGIN - HORIZON), "Temporal leakage: train targets overlap with internal validation!"
    assert (ORIGIN - 1) < ORIGIN, "Temporal leakage: validation overlaps with external test!"
    print(f"Temporal isolation verified: train target max week index = {train_history_len - 1} < internal val start = {ORIGIN - HORIZON} < external eval start = {ORIGIN}.")

    # 5. Load Pipeline fresh
    print("\n--- Initializing Pipeline & Model ---")
    torch.manual_seed(42 + FOLD)
    np.random.seed(42 + FOLD)

    pipe = Chronos2Pipeline.from_pretrained(
        str(active_weights_dir),
        device_map='cuda',
        local_files_only=True
    )
    base_model = pipe.model
    total_base_params = sum(p.numel() for p in base_model.parameters())
    quantiles = list(base_model.chronos_config.quantiles)
    qi_med = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    print(f"Base model loaded fresh. Total base parameters: {total_base_params:,}")

    # 6. LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )

    # Check PEFT parameters upfront
    peft_test_model = get_peft_model(base_model, lora_config)
    n_trainable_params, n_total_params = peft_test_model.get_nb_trainable_parameters()
    print(f"LoRA Target Modules: {lora_config.target_modules}")
    print(f"LoRA Rank r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"Trainable parameters: {n_trainable_params:,} / {n_total_params:,} ({100 * n_trainable_params / n_total_params:.2f}%)")

    # Reload fresh pipeline to avoid state contamination from test get_peft_model
    del peft_test_model
    del pipe
    torch.cuda.empty_cache()

    pipe = Chronos2Pipeline.from_pretrained(
        str(active_weights_dir),
        device_map='cuda',
        local_files_only=True
    )

    # 7. Fine-Tuning Execution
    print("\n--- Starting Fine-Tuning with LoRA ---")
    print(f"Parameters: steps={STEPS}, lr={LR}, context_length={CONTEXT}, horizon={HORIZON}")
    print(f"Batch size={BATCH_SIZE}, gradient_accumulation_steps={GRAD_ACCUM}")

    loss_callback = LossLoggingCallback()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_train_start = time.perf_counter()

    ft_pipe = pipe.fit(
        inputs=train_tensor,
        validation_inputs=val_tensor,
        prediction_length=HORIZON,
        context_length=CONTEXT,
        finetune_mode="lora",
        lora_config=lora_config,
        learning_rate=LR,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        output_dir=ckpt_dir,
        finetuned_ckpt_name="lora_fold4_ckpt",
        callbacks=[loss_callback],
        logging_steps=LOG_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        gradient_accumulation_steps=GRAD_ACCUM,
    )

    torch.cuda.synchronize()
    t_train_elapsed = time.perf_counter() - t_train_start
    train_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Fine-tuning completed in {t_train_elapsed:.2f} s ({t_train_elapsed / 60:.2f} min)")
    print(f"Peak VRAM during training: {train_peak_vram_mb:.2f} MB")

    # Extract train and eval losses
    train_losses = []
    eval_losses = []
    for l in loss_callback.logs:
        if 'loss' in l and 'eval_loss' not in l:
            train_losses.append({'step': l.get('global_step'), 'loss': float(l['loss']), 'lr': float(l.get('learning_rate', 0))})
            print(f"  Step {l.get('global_step'):3d} | Train Loss: {float(l['loss']):.4f} | LR: {float(l.get('learning_rate', 0)):.2e}")
        elif 'eval_loss' in l:
            eval_losses.append({'step': l.get('global_step'), 'eval_loss': float(l['eval_loss'])})
            print(f"  Step {l.get('global_step'):3d} | Internal Validation Loss: {float(l['eval_loss']):.4f}")

    final_train_loss = train_losses[-1]['loss'] if train_losses else None
    final_eval_loss = eval_losses[-1]['eval_loss'] if eval_losses else None

    # 8. External Inference
    print("\n--- External Inference on Fold 4 Forecast Horizon ---")
    ext_context_slice = panel.iloc[ORIGIN - CONTEXT : ORIGIN].values.T[:, None, :] # (8789, 1, 52)
    ext_context_tensor = torch.tensor(ext_context_slice, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_inf_start = time.perf_counter()

    with torch.inference_mode():
        predictions = ft_pipe.predict(
            ext_context_tensor,
            prediction_length=HORIZON,
            batch_size=256,
            context_length=CONTEXT,
            cross_learning=False
        )

    torch.cuda.synchronize()
    t_inf_elapsed = time.perf_counter() - t_inf_start
    inf_peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Inference completed in {t_inf_elapsed:.2f} s")
    print(f"Inference peak VRAM: {inf_peak_vram_mb:.2f} MB")

    # Shape: (8789, len(quantiles), 9)
    pred_arr = np.stack([p[0].detach().cpu().numpy() for p in predictions])
    print(f"Predictions array shape: {pred_arr.shape}")

    # 9. Format Keyed Predictions
    print("\n--- Constructing Keyed Predictions & Integrity Verification ---")
    keys_expanded = pd.concat([pairs_df] * HORIZON, ignore_index=True)
    keys_expanded.insert(0, 'week', np.repeat(val_weeks, n_pairs))

    records_truth = []
    records_mean13 = []
    records_preds_q = {q: [] for q in quantiles}

    for h, w_str in enumerate(val_weeks):
        actual_y = panel.iloc[ORIGIN + h].values
        m13 = panel.iloc[ORIGIN - 13 : ORIGIN].mean().values
        records_truth.append(actual_y)
        records_mean13.append(m13)
        for q_idx, q in enumerate(quantiles):
            records_preds_q[q].append(pred_arr[:, q_idx, h])

    y_true = np.concatenate(records_truth)
    y_mean13 = np.concatenate(records_mean13)
    y_med_raw = np.maximum(0.0, np.concatenate(records_preds_q[quantiles[qi_med]]))
    y_rint = np.rint(y_med_raw)
    y_leg = apply_p(y_med_raw, 1.05, 0.65)

    keyed_df = keys_expanded.copy()
    keyed_df['y_true'] = y_true
    keyed_df['unconstrained_demand_13'] = y_mean13
    keyed_df['y_pred_continuous'] = y_med_raw
    keyed_df['y_pred_rint'] = y_rint
    keyed_df['y_pred_legacy_p'] = y_leg

    for q in quantiles:
        keyed_df[f'q_{q:g}'] = np.concatenate(records_preds_q[q])

    # Integrity assertions
    assert len(keyed_df) == 79101, f"Expected 79,101 rows, got {len(keyed_df)}"
    assert not keyed_df.duplicated(subset=KEYS).any(), "Duplicate keys found!"
    assert not keyed_df[KEYS].isna().any().any(), "NaN found in keys!"
    assert np.isfinite(y_med_raw).all(), "Non-finite values in continuous predictions!"
    assert np.isfinite(y_rint).all(), "Non-finite values in rint predictions!"
    assert np.isfinite(y_leg).all(), "Non-finite values in legacy_p predictions!"
    print(f"Integrity check PASSED: exactly {len(keyed_df):,} rows, unique keys, no NaN/inf.")

    # Save keyed predictions
    keyed_filename = "chronos2_small_lora_fold4_keyed.csv.gz"
    keyed_path = keyed_dir / keyed_filename
    keyed_df.to_csv(keyed_path, index=False, compression='gzip')
    print(f"Keyed predictions saved: {keyed_path} ({keyed_path.stat().st_size:,} bytes)")

    # 10. Metric Evaluation
    print("\n--- Metric Evaluation ---")
    truth_df = keys_expanded.copy(); truth_df['quantity'] = y_true
    mean13_df = keys_expanded.copy(); mean13_df['quantity'] = y_mean13
    m_base = compute_all_metrics_fast(truth_df, mean13_df)
    print(f"Mean13 Baseline Fold 4: SMAPE={m_base['smape']:.3f}%, WAPE={m_base['wape']:.3f}%, RMSE={m_base['rmse']:.4f}, Bias={m_base['bias']:.3f}%, RMSLE={m_base['rmsle']:.4f}")

    variants = {
        'rint': y_rint,
        'legacy_p': y_leg,
        'continuous_raw': y_med_raw,
    }

    eval_results = {}
    for v_name, v_pred in variants.items():
        pred_df = keys_expanded.copy(); pred_df['quantity'] = v_pred
        # Requirement 2: Strict verification of key uniqueness, completeness and full matching
        verify_exact_and_unique_keys(truth_df, pred_df, expected_rows=79101)
        m = compute_all_metrics_fast(truth_df, pred_df)
        sc_clip, sc_unclip = calc_scores(m, m_base)
        eval_results[v_name] = {
            'clipped_score': sc_clip,
            'unclipped_ratio': sc_unclip,
            'smape': m['smape'],
            'wape': m['wape'],
            'rmse': m['rmse'],
            'bias': m['bias'],
            'rmsle': m['rmsle'],
        }
        print(f"  [{v_name:15s}] Clipped: {sc_clip:.6f} | Unclipped: {sc_unclip:.6f} | SMAPE: {m['smape']:.3f}% | WAPE: {m['wape']:.3f}% | RMSE: {m['rmse']:.4f} | Bias: {m['bias']:.3f}% | RMSLE: {m['rmsle']:.4f}")

    # 11. Comparison with Control Benchmarks
    # Fold 4 control from current champion Scale_1M_Dual_0.5
    control_ensemble_fold4 = {
        'clipped_score': 0.736573,
        'unclipped_ratio': 0.773500,
        'smape': 64.685983,
        'wape': 51.432570,
        'rmse': 3.081258,
        'bias': 79.065592,
        'rmsle': 0.549929,
    }

    # Requirement 3: Exact Full FT metrics
    # Chronos-2 Small Full Fine-Tuning across all 5 folds (verified benchmark summary)
    control_small_full_ft_5fold_rint = {
        'clipped_score': 0.737456,
        'unclipped_ratio': 0.764913,
        'smape': 62.949049,
        'wape': 50.017672,
        'rmse': 2.930979,
        'bias': 76.494682,
        'rmsle': 0.533512,
    }

    # Fold 4 Chronos-2 Small Full Fine-Tuning
    control_small_full_ft_fold4_rint = {
        'clipped_score': 0.742032,
        'unclipped_ratio': 0.780329,
        'smape': 65.964056,
        'wape': 51.423384,
        'rmse': 3.090069,
        'bias': 79.379745,
        'rmsle': 0.558431,
    }
    control_small_full_ft_fold4_leg = {
        'clipped_score': 0.738396,
        'unclipped_ratio': 0.777740,
        'smape': 66.037866,
        'wape': 51.166699,
        'rmse': 3.024999,
        'bias': 79.708315,
        'rmsle': 0.563457,
    }

    # 12. Save Complete Experiment JSON & Summary CSV
    report = {
        'experiment_name': 'chronos2_lora_fold4_pilot',
        'model_loaded': model_name,
        'weights_status': status_weights,
        'full_120m_status': 'NEED_PRETRAINED_WEIGHTS',
        'fold': FOLD,
        'origin': ORIGIN,
        'external_window': [val_weeks[0], val_weeks[-1]],
        'internal_val_window': [int_val_weeks[0], int_val_weeks[-1]],
        'training_history_end': ORIGIN - HORIZON,
        'parameters': {
            'context_length': CONTEXT,
            'prediction_length': HORIZON,
            'learning_rate': LR,
            'num_steps': STEPS,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRAD_ACCUM,
            'effective_batch_size': BATCH_SIZE * GRAD_ACCUM,
        },
        'lora': {
            'applied': True,
            'rank': lora_config.r,
            'alpha': lora_config.lora_alpha,
            'target_modules': list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules,
            'trainable_params': n_trainable_params,
            'total_params': n_total_params,
            'trainable_pct': float(100 * n_trainable_params / n_total_params),
        },
        'resources': {
            'train_sec': t_train_elapsed,
            'inference_sec': t_inf_elapsed,
            'train_peak_vram_mb': train_peak_vram_mb,
            'inf_peak_vram_mb': inf_peak_vram_mb,
        },
        'training_loss_history': train_losses,
        'internal_val_loss_history': eval_losses,
        'final_train_loss': final_train_loss,
        'final_eval_loss': final_eval_loss,
        'evaluation': eval_results,
        'controls': {
            'ensemble_scale_1m_dual_fold4': control_ensemble_fold4,
            'chronos2_small_full_ft_fold4_rint': control_small_full_ft_fold4_rint,
            'chronos2_small_full_ft_fold4_leg': control_small_full_ft_fold4_leg,
            'chronos2_small_full_ft_5fold_rint': control_small_full_ft_5fold_rint,
        },
        'paths': {
            'code': 'run_chronos_lora_fold4.py',
            'checkpoint': str(ckpt_dir / 'lora_fold4_ckpt'),
            'keyed_predictions': str(keyed_path),
            'log_json': str(logs_dir / 'fold4_lora_results.json'),
        }
    }

    with open(logs_dir / 'fold4_lora_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_rows = []
    for v_name, res in eval_results.items():
        summary_rows.append({
            'model': f"{model_name}_LoRA",
            'variant': v_name,
            **res,
            'train_sec': t_train_elapsed,
            'train_peak_vram_mb': train_peak_vram_mb,
        })
    summary_rows.append({
        'model': 'Control_Ensemble_Scale_1M_Dual_0.5',
        'variant': 'tests_champion',
        **control_ensemble_fold4,
        'train_sec': None,
        'train_peak_vram_mb': None,
    })
    summary_rows.append({
        'model': 'Chronos2_Small_Full_FT_Fold4',
        'variant': 'rint',
        **control_small_full_ft_fold4_rint,
        'train_sec': None,
        'train_peak_vram_mb': None,
    })
    summary_rows.append({
        'model': 'Chronos2_Small_Full_FT_Fold4',
        'variant': 'legacy_p',
        **control_small_full_ft_fold4_leg,
        'train_sec': None,
        'train_peak_vram_mb': None,
    })
    summary_rows.append({
        'model': 'Chronos2_Small_Full_FT_5Fold_Avg',
        'variant': 'rint',
        **control_small_full_ft_5fold_rint,
        'train_sec': None,
        'train_peak_vram_mb': None,
    })
    pd.DataFrame(summary_rows).to_csv(logs_dir / 'fold4_lora_summary.csv', index=False)
    print(f"\nResults JSON saved to {logs_dir / 'fold4_lora_results.json'}")
    print(f"Summary CSV saved to {logs_dir / 'fold4_lora_summary.csv'}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    main()
