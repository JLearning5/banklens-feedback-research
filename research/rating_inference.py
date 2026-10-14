import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_scores(folder, frame):
    recipe = json.loads((folder / 'model.json').read_text())
    if recipe['family'] == 'pipeline':
        return joblib.load(folder / recipe['file']).predict_proba(frame['review_text'])
    saved = pd.read_parquet(folder / recipe['file']).set_index('review_id')
    assert set(frame['review_id']).issubset(set(saved.index)), 'UNKNOWN_CORPUS_ID'
    return saved.loc[frame['review_id'], recipe['classes']].to_numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    args, _ = parser.parse_known_args()
    output = Path(args.output)
    frame = pd.read_parquet(output / 'metrics' / 'reload_input.parquet')
    values = load_scores(output / 'models' / args.model, frame)
    assert values.shape == (len(frame), 3) and np.isfinite(values).all(), 'INVALID_RELOAD_SCORES'
    np.save(output / 'metrics' / 'reload_scores.npy', values)
    (output / 'metrics' / 'reload_check.json').write_text(json.dumps({'model': args.model, 'rows': len(frame), 'finite_scores': True, 'fresh_process': True}))
