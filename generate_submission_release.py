"""Reproduce the final competition CSV from daily history and saved models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'forecast-container'))
from app.predictor import predict_from_history
from score import validate_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sales', required=True)
    p.add_argument('--stock', required=True)
    p.add_argument('--output', default='reproduced_submission.csv')
    p.add_argument('--reference', default=None)
    a = p.parse_args()
    frames = []
    for filename in (a.sales, a.stock):
        df = pd.read_csv(filename)
        dates = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='raise')
        df = df.loc[dates <= pd.Timestamp('2026-07-26')].copy()
        frames.append(df)
    pred = predict_from_history(*frames, bundle_dir=ROOT/'forecast-container'/'model')
    warnings = validate_submission(pred)
    assert not warnings, warnings
    assert np.isfinite(pred.quantity.to_numpy()).all()
    keys = ['week', 'warehouse_name', 'article']
    pairs = pd.read_parquet(ROOT/'forecast-container'/'model'/'valid_pairs.parquet')[keys[1:]]
    expected = pd.DataFrame({'week': pd.date_range('2026-07-27', periods=9, freq='W-MON').strftime('%Y-%m-%d')}).merge(pairs, how='cross')
    pred['week'] = pd.to_datetime(pred.week).dt.strftime('%Y-%m-%d')
    matched = expected.merge(pred, on=keys, how='outer', validate='one_to_one', indicator=True)
    assert matched['_merge'].eq('both').all(), 'Prediction keys differ from expected grid'
    result = {'rows': len(pred), 'quantity_sum': float(pred.quantity.sum())}
    if a.reference:
        ref = pd.read_csv(a.reference)
        joined = ref.merge(pred, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('_ref', '_new'))
        assert joined['_merge'].eq('both').all()
        delta = np.abs(joined.quantity_ref - joined.quantity_new)
        result.update(max_abs_difference=float(delta.max()), different_rows=int((delta != 0).sum()))
        assert delta.max() == 0, result
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    result['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
