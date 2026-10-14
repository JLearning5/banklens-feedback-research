import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from huggingface_hub import model_info, snapshot_download
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation import choose_threshold, comparison_curves, confusion_figure, report_model, resources_figure
from models import CompactTransformer, Generative, create, parallel
from protocol import prepare_gold


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.process = psutil.Process()

    def __enter__(self):
        self.started = time.monotonic()
        self.process.cpu_percent()
        self.samples.append({'seconds': 0.0, 'rss_mb': self.process.memory_info().rss / 1e6, 'cpu_percent': 0.0})

        def sample():
            while not self.stop.is_set():
                self.samples.append({'seconds': time.monotonic() - self.started, 'rss_mb': self.process.memory_info().rss / 1e6, 'cpu_percent': self.process.cpu_percent() / (os.cpu_count() or 1)})
                self.stop.wait(0.5)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def load_selected(recipe, config, folder, train=None):
    metadata = json.loads((folder / 'model.json').read_text())
    if recipe['family'] in ['logistic', 'svm']:
        model = create(recipe, config, folder)
        model.model = joblib.load(folder / 'pipeline.joblib')
        return model
    if recipe['family'] == 'transformer':
        model = CompactTransformer.__new__(CompactTransformer)
        model.recipe, model.config, model.output = recipe, config, folder
        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.tokenizer = AutoTokenizer.from_pretrained(folder / 'checkpoint')
        model.model = parallel(AutoModelForSequenceClassification.from_pretrained(folder / 'checkpoint').to(model.device))
        return model
    recipe = {**recipe, 'revision': metadata['revision']}
    model = create(recipe, config, folder)
    if recipe['family'] == 'agent':
        model.train = pd.read_parquet(folder / 'retrieval_examples.parquet') if train is None else train.reset_index(drop=True)
        saved = joblib.load(folder / 'retriever.joblib')
        assert saved['training_ids'] == model.train['review_id'].tolist(), 'RETRIEVER_TRAINING_MISMATCH'
        model.vectorizer, model.retrieval_matrix = saved['vectorizer'], saved['matrix']
    return model


def execute(data_path, reviews_path, output, config_path=None):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'final_test_evaluation.json').exists(), 'FINAL_TEST_ALREADY_EVALUATED'
    config = json.loads(Path(config_path or Path(__file__).parent / 'experiment_config.json').read_text())
    gold, manifest, split_hash = prepare_gold(data_path, reviews_path, config)
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    config['split_hash'] = split_hash
    config['review_file_sha256'] = hashlib.sha256(Path(reviews_path).read_bytes()).hexdigest()
    print('SPLIT_SUMMARY', gold.groupby(['split', 'reviewer_sentiment']).size().to_json(), flush=True)
    train = gold[gold['split'].eq('train')].reset_index(drop=True)
    validation = gold[gold['split'].eq('validation')].reset_index(drop=True)
    test = gold[gold['split'].eq('test')].reset_index(drop=True)
    for recipe in config['models']:
        if 'checkpoint' in recipe:
            recipe['revision'] = model_info(recipe['checkpoint']).sha
            snapshot_download(recipe['checkpoint'], revision=recipe['revision'], allow_patterns=['*.json', '*.txt', '*.safetensors'])
    (output / 'run_config.json').write_text(json.dumps(config, indent=2))
    figures, metrics = output / 'figures', output / 'metrics'
    figures.mkdir(exist_ok=True)
    metrics.mkdir(exist_ok=True)
    predictions, records, samples = {}, [], {}
    try:
        for recipe in config['models']:
            folder = output / 'models' / recipe['id']
            folder.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            at = time.monotonic()
            with Monitor() as monitor:
                model = create(recipe, config, folder)
                model.fit(train, validation)
                scores = model.predict(validation)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                metadata = model.save()
            metadata['classes'] = config['classes']
            (folder / 'model.json').write_text(json.dumps(metadata, indent=2))
            assert scores.shape == (len(validation), len(config['classes'])), 'SCORE_SHAPE'
            assert np.isfinite(scores).all() and np.allclose(scores.sum(1), 1, atol=1e-4), 'INVALID_SCORES'
            predictions[recipe['id']] = scores
            report = report_model(validation['target'].to_numpy(), scores, config['classes'], recipe['id'], metrics)
            samples[recipe['id']] = monitor.samples
            rows = validation[['review_id', 'duplicate_group', 'target', 'split']].copy()
            rows['prediction'] = scores.argmax(1)
            for index, name in enumerate(config['classes']):
                rows['score_' + name] = scores[:, index]
            rows.to_parquet(metrics / f'{recipe["id"]}_validation_predictions.parquet', index=False)
            result = {'model': recipe['id'], 'primary_metric': {'name': 'macro_f1', 'split': 'validation', 'value': report['macro_f1'], 'higher_is_better': True}, 'validation_accuracy': report['accuracy'], 'runtime_seconds': time.monotonic() - at, 'peak_rss_mb': max(v['rss_mb'] for v in monitor.samples), 'mean_cpu_percent': float(np.mean([v['cpu_percent'] for v in monitor.samples])), 'gpu_peak_allocated_mb': [torch.cuda.max_memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())], 'artifact_bytes': sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())}
            records.append(result)
            pd.DataFrame(monitor.samples).to_csv(metrics / f'{recipe["id"]}_resources.csv', index=False)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            confusion_figure(validation['target'].to_numpy(), scores, config['classes'], recipe['id'], figures, config['figure_dpi'])
        comparison_curves(validation['target'].to_numpy(), predictions, config['classes'], figures, config['figure_dpi'])
        resources_figure(samples, figures, config['figure_dpi'])
        best = max(records, key=lambda item: (item['primary_metric']['value'], -item['runtime_seconds']))
        selected = next(recipe for recipe in config['models'] if recipe['id'] == best['model'])
        threshold = choose_threshold(validation['target'].to_numpy(), predictions[best['model']], config['target_selective_precision'])
        selection = {'model': best['model'], 'split_hash': split_hash, 'validation_macro_f1': best['primary_metric']['value'], 'operating_rule': 'argmax', 'selective_threshold': threshold}
        (output / 'selection.json').write_text(json.dumps(selection, indent=2))
        validation.iloc[:3].to_parquet(metrics / 'reload_smoke_input.parquet', index=False)
        subprocess.run([sys.executable, str(Path(__file__).parent / 'inference_smoke.py'), '--run-output', str(output), '--model', best['model']], check=True)
        reload_scores = np.load(metrics / 'reload_smoke_scores.npy')
        assert np.allclose(reload_scores, predictions[best['model']][:3], atol=0.002), 'RELOAD_SCORE_MISMATCH'
        winner = load_selected(selected, config, output / 'models' / best['model'], train)
        test_scores = winner.predict(test)
        accepted = test_scores.max(1) >= threshold['threshold']
        predicted = test_scores.argmax(1)
        final = {'model': best['model'], 'split': 'test', 'macro_f1': float(f1_score(test['target'], predicted, average='macro', zero_division=0)), 'accuracy': float(np.mean(predicted == test['target'].to_numpy())), 'coverage': float(accepted.mean()), 'accepted_accuracy': float(np.mean(predicted[accepted] == test['target'].to_numpy()[accepted])) if accepted.any() else None, 'selective_threshold': threshold['threshold'], 'split_hash': split_hash}
        (output / 'final_test_evaluation.json').write_text(json.dumps(final, indent=2))
        print('FINAL_TEST_EVALUATION', json.dumps(final), flush=True)
        test_rows = test[['review_id', 'duplicate_group', 'target', 'split']].copy()
        test_rows['prediction'], test_rows['accepted'] = predicted, accepted
        for index, name in enumerate(config['classes']):
            test_rows['score_' + name] = test_scores[:, index]
        test_rows.to_parquet(metrics / 'final_test_predictions.parquet', index=False)
        from source_audit import source_audit
        audits = [source_audit(path) for path in Path(__file__).parent.glob('*.py')]
        (output / 'python_sources_audit.json').write_text(json.dumps(audits, indent=2))
        run = {'models': records, 'selection': selection, 'final_test': final, 'split_hash': split_hash, 'seed': config['seed'], 'runtime_seconds': time.monotonic() - started, 'status': 'complete'}
        (output / 'metrics.json').write_text(json.dumps(run, indent=2))
        packages = ['torch', 'transformers', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'joblib', 'psutil']
        (output / 'requirements.txt').write_text('\n'.join(name + '==' + importlib.metadata.version(name) for name in packages) + '\n')
        checksums = [{'path': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'checksums.json']
        (output / 'checksums.json').write_text(json.dumps(checksums, indent=2))
        archive = output.with_name(output.name + '_artifacts.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
            for path in output.rglob('*'):
                if path.is_file():
                    handle.write(path, path.relative_to(output).as_posix())
        print('ARTIFACT_ZIP', str(archive), archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
        return run
    finally:
        print('TOTAL_RUNTIME_SECONDS', round(time.monotonic() - started, 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--output', default='/kaggle/working/banklens_experiments')
    parser.add_argument('--config')
    args, _ = parser.parse_known_args()
    execute(args.data, args.reviews, args.output, args.config)
