import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import runpy
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from evaluation import comparison_curves, resources_figure, save_figure
from models import create
from run_experiments import Monitor
from source_audit import source_audit


CLASSES = ['positive', 'negative', 'neutral']


def decisions(scores):
    result = scores.argmax(1)
    result[scores.max(1) <= 0] = -1
    return result


def summarize(y, scores, weights=None):
    predicted = decisions(scores)
    report = classification_report(y, predicted, labels=np.arange(3), target_names=CLASSES, zero_division=0, sample_weight=weights, output_dict=True)
    return {'accuracy': float(np.average(predicted == y, weights=weights)), 'macro_f1': float(f1_score(y, predicted, labels=np.arange(3), average='macro', zero_division=0, sample_weight=weights)), 'coverage': float(np.average(predicted >= 0, weights=weights)), 'report': report}


def display_report(frame, scores, model_id, output, dpi):
    y = frame['target'].to_numpy()
    predicted = decisions(scores)
    result = summarize(y, scores)
    result['group_balanced'] = summarize(y, scores, 1 / frame.groupby('duplicate_group')['review_id'].transform('size').to_numpy())
    result['rows'] = len(frame)
    result['abstained'] = int((predicted < 0).sum())
    text = classification_report(y, predicted, labels=np.arange(3), target_names=CLASSES, zero_division=0)
    print('CLASSIFICATION_REPORT', model_id, flush=True)
    print(text, flush=True)
    (output / 'metrics' / f'{model_id}_classification_report.txt').write_text(text)
    (output / 'metrics' / f'{model_id}_evaluation.json').write_text(json.dumps(result, indent=2))
    matrix = np.zeros((3, 4), dtype=int)
    np.add.at(matrix, (y, np.where(predicted < 0, 3, predicted)), 1)
    assert matrix.sum() == len(frame), 'CONFUSION_ROWS_LOST'
    image = np.ones((3, 4, 3))
    for i in range(3):
        for j in range(4):
            if matrix[i, j]:
                image[i, j] = np.array([47, 107, 255] if i == j else [242, 142, 43]) / 255
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if i == j and matrix[i, j] else 'black')
    ax.set_xticks(range(4), CLASSES + ['referred'], rotation=25, ha='right')
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel('Output sentiment')
    ax.set_ylabel('Rating reference')
    ax.set_title(model_id + ' validation counts')
    save_figure(fig, [ax], output / 'figures' / f'{model_id}_confusion.png', model_id + '_confusion', dpi)
    return result


def frozen_scores(frame, family):
    if family == 'qwen4b_pipeline':
        result = np.zeros((len(frame), 3), dtype=float)
        mapping = {name: i for i, name in enumerate(CLASSES)}
        for i, label in enumerate(frame['sentiment']):
            if label in mapping:
                result[i, mapping[label]] = 1
        return result
    values = frame[['probability_positive', 'probability_negative', 'probability_neutral']].to_numpy(dtype=float)
    return np.nan_to_num(values)


def prepare(data_path, pilot_path, seed):
    data = pd.read_parquet(data_path)
    assert data['review_id'].is_unique, 'DUPLICATE_SOURCE_ID'
    rating = pd.to_numeric(data['rating'], errors='coerce')
    eligible = rating.isin([1, 2, 3, 4, 5]) & data['review_text'].str.strip().str.len().gt(0)
    frame = data[eligible].copy().reset_index(drop=True)
    frame['target'] = np.where(frame['rating'].astype(int).ge(4), 0, np.where(frame['rating'].astype(int).le(2), 1, 2))
    forced = set()
    if pilot_path:
        forced = set(pd.read_parquet(pilot_path)['duplicate_group'])
    frame['split'] = 'train'
    selected = frame[~frame['duplicate_group'].isin(forced)]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(selected['review_text'], selected['target'], selected['duplicate_group'])):
        frame.loc[selected.index[indices], 'split'] = 'test' if fold == 0 else 'validation' if fold == 1 else 'train'
    assert frame.groupby('duplicate_group')['split'].nunique().max() == 1, 'SPLIT_LEAKAGE'
    assert not set(frame[frame['split'].ne('train')]['duplicate_group']).intersection(forced), 'PILOT_LEAKAGE'
    manifest = frame[['review_id', 'duplicate_group', 'target', 'split']].sort_values('review_id')
    checks = {'source_rows': len(data), 'eligible_rows': len(frame), 'excluded_rows': int((~eligible).sum()), 'pilot_groups_reserved_for_training': len(forced.intersection(set(frame['duplicate_group']))), 'rating_conflict_groups': int(frame.groupby('duplicate_group')['target'].nunique().gt(1).sum()), 'duplicate_groups_disjoint': True, 'split_hash': hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()}
    return frame, manifest, checks


def reuse_baseline(root, family, folder, validation, checks):
    root = Path(root)
    summary = json.loads((root / 'pretraining.json').read_text())
    assert summary['status'] == 'complete' and summary['test_predictions_generated'] is False, 'INVALID_PRETRAINING_STATUS'
    assert summary['seed'] == 20261014 and summary['classes'] == CLASSES, 'PRETRAINING_CONFIG_MISMATCH'
    assert summary['data'] == checks, 'PRETRAINING_DATA_MISMATCH'
    previous = root / 'models' / family
    details = summary['models'][family]
    assert hashlib.sha256((previous / 'pipeline.joblib').read_bytes()).hexdigest() == details['artifact_sha256'], 'PRETRAINED_ARTIFACT_MISMATCH'
    ids = pd.read_parquet(previous / 'validation_ids.parquet')['review_id'].tolist()
    assert ids == validation['review_id'].tolist(), 'PRETRAINED_VALIDATION_IDS_MISMATCH'
    scores = np.load(previous / 'validation_scores.npy')
    assert scores.shape == (len(validation), 3) and np.isfinite(scores).all() and np.allclose(scores.sum(1), 1), 'INVALID_PRETRAINED_SCORES'
    model = joblib.load(previous / 'pipeline.joblib')
    assert np.allclose(model.predict_proba(validation['review_text'].iloc[:5]), scores[:5], atol=1e-6), 'PRETRAINED_RELOAD_MISMATCH'
    shutil.copy2(previous / 'pipeline.joblib', folder / 'pipeline.joblib')
    samples = pd.read_csv(previous / 'resources.csv').to_dict('records')
    assert samples, 'MISSING_PRETRAINING_RESOURCES'
    return scores, samples, details


def run(data_path, pilot_path, output, dpi=1200, baseline_root=None):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ['figures', 'metrics', 'models', 'reviewer_only']:
        (output / name).mkdir(exist_ok=True)
    assert not (output / 'final_test.json').exists(), 'FINAL_TEST_ALREADY_EVALUATED'
    seed = 20261014
    try:
        frame, manifest, checks = prepare(data_path, pilot_path, seed)
        manifest.to_csv(output / 'split_manifest.csv', index=False)
        (output / 'reviewer_only' / 'data_checks.json').write_text(json.dumps(checks, indent=2))
        print('DATA_CHECKS', json.dumps(checks), flush=True)
        print('SPLIT_SUMMARY', frame.groupby(['split', 'target']).size().to_json(), flush=True)
        train = frame[frame['split'].eq('train')].reset_index(drop=True)
        validation = frame[frame['split'].eq('validation')].reset_index(drop=True)
        test = frame[frame['split'].eq('test')].reset_index(drop=True)
        config = {'seed': seed, 'classes': CLASSES}
        scores, metrics, resource_samples = {}, {}, {}
        for family in ['logistic', 'svm', 'qwen4b_pipeline', 'roberta_sentiment_check']:
            at = time.monotonic()
            folder = output / 'models' / family
            folder.mkdir(exist_ok=True)
            reused = None
            with Monitor() as monitor:
                if family in ['logistic', 'svm']:
                    if baseline_root:
                        predicted, previous_samples, reused = reuse_baseline(baseline_root, family, folder, validation, checks)
                    else:
                        model = create({'id': family, 'family': family}, config, folder)
                        model.fit(train, validation)
                        predicted = model.predict(validation)
                        model.save()
                    recipe = {'family': 'pipeline', 'file': 'pipeline.joblib'}
                else:
                    predicted = frozen_scores(validation, family)
                    catalog = frame[['review_id']].copy()
                    all_scores = frozen_scores(frame, family)
                    for i, name in enumerate(CLASSES):
                        catalog[name] = all_scores[:, i]
                    catalog.to_parquet(folder / 'catalog_predictions.parquet', index=False)
                    recipe = {'family': 'corpus_catalog', 'file': 'catalog_predictions.parquet'}
            resource_samples[family] = previous_samples if reused else monitor.samples
            stage_seconds = reused['training_and_validation_seconds'] if reused else time.monotonic() - at
            if family in ['logistic', 'svm'] and not reused:
                del model
                gc.collect()
            recipe.update(classes=CLASSES, source_family=family, score_kind='fixed_class_indicator' if family == 'qwen4b_pipeline' else 'class_probability')
            (folder / 'model.json').write_text(json.dumps(recipe, indent=2))
            scores[family] = predicted
            metrics[family] = display_report(validation, predicted, family, output, dpi)
            metrics[family]['benchmark_stage_seconds'] = stage_seconds
            metrics[family]['peak_rss_mb'] = max(v['rss_mb'] for v in resource_samples[family])
            metrics[family]['reused_verified_pretraining'] = reused is not None
            pd.DataFrame(resource_samples[family]).to_csv(output / 'metrics' / f'{family}_resources.csv', index=False)
            rows = validation[['review_id', 'duplicate_group', 'target']].copy()
            rows['prediction'] = decisions(predicted)
            for i, name in enumerate(CLASSES):
                rows['score_' + name] = predicted[:, i]
            rows.to_parquet(output / 'metrics' / f'{family}_validation_predictions.parquet', index=False)
        comparison_curves(validation['target'].to_numpy(), scores, CLASSES, output / 'figures', dpi)
        resources_figure(resource_samples, output / 'figures', dpi)
        order = ['logistic', 'svm', 'qwen4b_pipeline', 'roberta_sentiment_check']
        selected = max(metrics, key=lambda name: (metrics[name]['macro_f1'], -order.index(name)))
        selection = {'model': selected, 'metric': 'validation_macro_f1', 'value': metrics[selected]['macro_f1'], 'split_hash': checks['split_hash']}
        (output / 'selection.json').write_text(json.dumps(selection, indent=2))
        validation.iloc[:5][['review_id', 'review_text']].to_parquet(output / 'metrics' / 'reload_input.parquet', index=False)
        subprocess.run([sys.executable, str(Path(__file__).parent / 'rating_inference.py'), '--output', str(output), '--model', selected], check=True)
        reloaded = np.load(output / 'metrics' / 'reload_scores.npy')
        assert np.allclose(reloaded, scores[selected][:5], atol=1e-6), 'RELOAD_MISMATCH'
        from rating_inference import load_scores
        test_scores = load_scores(output / 'models' / selected, test)
        final = summarize(test['target'].to_numpy(), test_scores)
        final['group_balanced'] = summarize(test['target'].to_numpy(), test_scores, 1 / test.groupby('duplicate_group')['review_id'].transform('size').to_numpy())
        final.update(model=selected, rows=len(test), split='test', split_hash=checks['split_hash'])
        final_rows = test[['review_id', 'duplicate_group', 'target']].copy()
        final_rows['prediction'] = decisions(test_scores)
        for i, name in enumerate(CLASSES):
            final_rows['score_' + name] = test_scores[:, i]
        final_rows.to_parquet(output / 'final_test_predictions.parquet', index=False)
        (output / 'final_test.json').write_text(json.dumps(final, indent=2))
        print('FINAL_TEST_EVALUATION', json.dumps(final), flush=True)
        result = {'status': 'complete', 'reference': 'source_rating', 'rating_mapping': {'1,2': 'negative', '3': 'neutral', '4,5': 'positive'}, 'classes': CLASSES, 'data': checks, 'validation': metrics, 'selection': selection, 'final_test': final, 'runtime_seconds': time.monotonic() - started}
        (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
        audits = [source_audit(path) for path in Path(__file__).parent.glob('*.py')]
        (output / 'reviewer_only' / 'source_audits.json').write_text(json.dumps(audits, indent=2))
        packages = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib', 'pyarrow', 'torch', 'transformers']}
        (output / 'runtime.json').write_text(json.dumps({'python': sys.version, 'platform': platform.platform(), 'packages': packages}, indent=2))
        (output / 'requirements.txt').write_text('\n'.join(name + '==' + version for name, version in packages.items()) + '\n')
        ctx = runpy.run_path(str(Path(__file__).parent / 'feedback_pipeline.py'))
        ctx['audit_notebook'].__globals__['asset_root'] = lambda: Path(__file__).parent
        ctx['audit_notebook'].__globals__['source_audit'] = source_audit
        ctx['audit_notebook'](output, 'rating_notebook_source.json')
        submitted = output / 'reviewer_only' / 'submitted_notebook.json'
        shutil.copy2(Path(__file__).parent / 'rating_notebook_source.json', submitted)
        inventory = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'checksums.json']
        (output / 'checksums.json').write_text(json.dumps(inventory, indent=2))
        archive = output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in output.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(output).as_posix())
        print('ARTIFACT_ZIP', str(archive), archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
        return result
    finally:
        print('TOTAL_RUNTIME_SECONDS', round(time.monotonic() - started, 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pilot')
    parser.add_argument('--baselines')
    parser.add_argument('--output', default='/kaggle/working/rating_evaluation')
    args, _ = parser.parse_known_args()
    run(args.data, args.pilot, args.output, baseline_root=args.baselines)
