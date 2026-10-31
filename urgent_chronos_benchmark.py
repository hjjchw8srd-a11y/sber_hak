"""Real pretrained Chronos evaluation. No random-weight fallback, no ensemble."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'sber_hak'))
from fast_score import compute_all_metrics_fast, METRIC_WEIGHTS

K = ['week', 'warehouse_name', 'article']
ORIGINS = {4:89, 3:98, 2:107, 1:116, 0:125}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='experiments/urgent_models/chronos2_small')
    p.add_argument('--folds', nargs='+', type=int, default=[4,3,2])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', default='experiments/urgent_models/chronos2_small_results')
    args = p.parse_args()
    weights = ROOT / args.weights
    file = weights / 'model.safetensors'
    if not file.exists():
        raise FileNotFoundError('Pretrained weights missing; evaluation NOT performed')
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    pipe = Chronos2Pipeline.from_pretrained(str(weights), device_map='cuda', local_files_only=True)
    levels = list(pipe.model.chronos_config.quantiles)
    qi = int(np.argmin(np.abs(np.asarray(levels)-.5)))
    assert abs(levels[qi]-.5)<1e-6
    data = pd.read_csv(ROOT/'weekly_data/weekly_dataset.csv.gz', usecols=K+['quantity'],parse_dates=['week'])
    assert not data.duplicated(K).any()
    panel = data.pivot(index='week',columns=['warehouse_name','article'],values='quantity').sort_index()
    assert panel.shape[1]==8789 and panel.notna().all().all()
    records=[]
    config={'weights_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'folds':args.folds,'context_max':104,'horizon':9,'batch_size':args.batch_size,
            'quantiles':levels,'torch':torch.__version__}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for fold in args.folds:
        origin=ORIGINS[fold]
        x=torch.tensor(panel.iloc[max(0,origin-104):origin].values.T.copy(),dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            predictions=pipe.predict(x,prediction_length=9,batch_size=args.batch_size,context_length=min(origin,104),cross_learning=False)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        arr=np.stack([v.detach().cpu().numpy().reshape(-1,len(levels),9)[0] for v in predictions])
        assert arr.shape==(8789,len(levels),9) and np.isfinite(arr).all()
        pairs=panel.columns.to_frame(index=False)
        keys=pd.concat([pairs]*9,ignore_index=True)
        keys.insert(0,'week',np.repeat(panel.index[origin:origin+9].strftime('%Y-%m-%d'),8789))
        truth=keys.copy();truth['quantity']=panel.iloc[origin:origin+9].values.ravel()
        base=keys.copy();base['quantity']=np.tile(panel.iloc[origin-13:origin].mean().values,9)
        bm=compute_all_metrics_fast(truth,base)
        saved=keys.copy()
        for j,lev in enumerate(levels):saved[f'q_{lev:g}']=arr[:,j,:].T.ravel()
        saved.to_csv(out/f'fold_{fold}_quantiles.csv.gz',index=False)
        med=np.maximum(arr[:,qi,:].T.ravel(),0)
        for name,pred in {'median':med,'median_rint':np.rint(med),'median_legacy':np.rint(np.where(1.05*med<.65,0,1.05*med))}.items():
            df=keys.copy();df['quantity']=pred
            m=compute_all_metrics_fast(truth,df)
            ratios={k:m[k]/bm[k] for k in METRIC_WEIGHTS}
            row=dict(fold=fold,variant=name,**m,clipped=sum(w*min(ratios[k],1) for k,w in METRIC_WEIGHTS.items()),unclipped=sum(w*ratios[k] for k,w in METRIC_WEIGHTS.items()),inference_sec=elapsed,peak_vram_mb=torch.cuda.max_memory_allocated()/2**20)
            records.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)

if __name__=='__main__':main()
