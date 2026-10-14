import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import load_selected


def execute(folder, model_id):
    folder = Path(folder)
    config = json.loads((folder / 'run_config.json').read_text())
    recipe = next(item for item in config['models'] if item['id'] == model_id)
    model = load_selected(recipe, config, folder / 'models' / model_id)
    inputs = pd.read_parquet(folder / 'metrics' / 'reload_smoke_input.parquet')
    scores = model.predict(inputs)
    assert scores.shape == (len(inputs), len(config['classes']))
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(1), 1, atol=1e-4)
    np.save(folder / 'metrics' / 'reload_smoke_scores.npy', scores)
    summary = {'model': model_id, 'rows': len(inputs), 'classes': config['classes'], 'fresh_process': True, 'status': 'passed'}
    (folder / 'inference_smoke_test.json').write_text(json.dumps(summary, indent=2))
    print('INFERENCE_SMOKE_TEST', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-output', required=True)
    parser.add_argument('--model', required=True)
    args, _ = parser.parse_known_args()
    execute(args.run_output, args.model)
