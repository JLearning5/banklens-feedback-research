import hashlib
import json
import os
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import create
from rating_evaluation import CLASSES, prepare
from run_experiments import Monitor
from source_audit import source_audit


def main():
    started = time.monotonic()
    assets = Path(os.environ['EVALUATION_ASSET_ROOT'])
    output = Path('/kaggle/working/baseline_pretraining')
    output.mkdir(exist_ok=True)
    prepared = output / 'prepared'
    prepared.mkdir(exist_ok=True)
    try:
        context = runpy.run_path(str(assets / 'feedback_pipeline.py'))
        context['prepare_data'].__globals__['asset_root'] = lambda: assets
        context['prepare_data'](prepared)
        pilot = assets / 'pilot_unique_labels.parquet'
        assert len(pd.read_parquet(pilot)) == 224, 'PILOT_ROWS_MISMATCH'
        frame, manifest, checks = prepare(prepared / 'prepared_rows.parquet', pilot, 20261014)
        manifest.to_csv(output / 'split_manifest.csv', index=False)
        print('SPLIT_SUMMARY', frame.groupby(['split', 'target']).size().to_json(), flush=True)
        train = frame[frame['split'].eq('train')].reset_index(drop=True)
        validation = frame[frame['split'].eq('validation')].reset_index(drop=True)
        config = {'seed': 20261014, 'classes': CLASSES}
        models = {}
        for family in ['logistic', 'svm']:
            folder = output / 'models' / family
            folder.mkdir(parents=True, exist_ok=True)
            at = time.monotonic()
            with Monitor() as monitor:
                model = create({'id': family, 'family': family}, config, folder)
                model.fit(train, validation)
                scores = model.predict(validation)
                model.save()
            assert scores.shape == (len(validation), 3) and np.isfinite(scores).all() and np.allclose(scores.sum(1), 1), 'INVALID_VALIDATION_SCORES'
            np.save(folder / 'validation_scores.npy', scores)
            validation[['review_id']].to_parquet(folder / 'validation_ids.parquet', index=False)
            pd.DataFrame(monitor.samples).to_csv(folder / 'resources.csv', index=False)
            models[family] = {'training_and_validation_seconds': time.monotonic() - at, 'peak_rss_mb': max(row['rss_mb'] for row in monitor.samples), 'training_rows': len(train), 'validation_rows': len(validation), 'artifact_sha256': hashlib.sha256((folder / 'pipeline.joblib').read_bytes()).hexdigest()}
            print('BASELINE_PREPARED', family, json.dumps(models[family]), flush=True)
            del model
        summary = {'status': 'complete', 'seed': 20261014, 'classes': CLASSES, 'data': checks, 'models': models, 'test_predictions_generated': False, 'runtime_seconds': time.monotonic() - started}
        (output / 'pretraining.json').write_text(json.dumps(summary, indent=2))
        (output / 'source_audits.json').write_text(json.dumps([source_audit(path) for path in assets.glob('*.py')], indent=2))
        print('PRETRAINING_COMPLETE', json.dumps(summary), flush=True)
    finally:
        print('TOTAL_RUNTIME_SECONDS', round(time.monotonic() - started, 3), flush=True)


if __name__ == '__main__':
    main()
