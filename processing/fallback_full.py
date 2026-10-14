import concurrent.futures
import hashlib
import json
import runpy
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer


def main():
    assets = next(Path('/kaggle/input').rglob('feedback_pipeline.py')).parent
    ctx = runpy.run_path(str(assets / 'feedback_pipeline.py'))
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    out = Path('/kaggle/working/sarah_feedback_full')
    out.mkdir(exist_ok=True)
    data, unique = ctx['prepare_data'](out)
    ident = 'Qwen/Qwen3-4B-Instruct-2507'
    revision = model_info(ident).sha
    prompt = (assets / 'annotation_prompt.txt').read_text()
    metadata = {'id': ident, 'revision': revision, 'engine': 'transformers', 'dtype': 'float16', 'max_new_tokens': 96, 'retry_max_new_tokens': 160, 'seed': 20261014, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    ctx['save_json'](out / 'generative_model.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(ident, revision=revision)
    shortened = {r.duplicate_group: len(r.review_text) > 1400 or len(tokenizer.encode(r.review_text, add_special_tokens=False)) > 650 for r in unique.itertuples()}
    prior = {}
    for path in Path('/kaggle/input').rglob('generative_checkpoint.parquet'):
        if 'sarah_feedback_full' not in str(path):
            continue
        config = path.parent / 'fallback_config.json'
        if not config.exists():
            continue
        previous = json.loads(config.read_text())
        assert all(previous[k] == metadata[k] for k in ['id', 'revision', 'prompt_sha256', 'seed']), 'RESUME_CONFIG_MISMATCH'
        for row in pd.read_parquet(path).to_dict('records'):
            if row['valid']:
                row['a'] = json.loads(row['a'])
                prior[row['duplicate_group']] = row
    unique = unique[~unique['duplicate_group'].isin(prior)].reset_index(drop=True)
    devices = min(2, torch.cuda.device_count())
    assert devices > 0, 'CUDA_REQUIRED'
    started = time.monotonic()

    def run(index):
        target = out / f'gpu_{index}'
        target.mkdir(exist_ok=True)
        ctx['save_json'](target / 'fallback_config.json', metadata)
        part = unique.iloc[index::devices].reset_index(drop=True)
        if not len(part):
            return []
        torch.cuda.set_device(index)
        with torch.cuda.device(index):
            return ctx['infer_generative'](part, target, batch_size=32, device=f'cuda:{index}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=devices) as pool:
        future = [pool.submit(run, index) for index in range(devices)]
        records = list(prior.values()) + [record for item in future for record in item.result()]
    for record in records:
        record['input_shortened'] = shortened[record['duplicate_group']]
    assert len(records) == len(shortened), 'INCOMPLETE_LABEL_SET'
    (out / 'generative_records.jsonl').write_text('\n'.join(json.dumps(record, ensure_ascii=False) for record in records), encoding='utf-8')
    seconds = time.monotonic() - started
    summary = {'records': len(records), 'seconds': seconds, 'reviews_per_second': len(unique) / seconds, 'resumed_records': len(prior), 'invalid_after_retry': sum(not r['valid'] for r in records), 'engine': 'transformers', 'devices': devices}
    ctx['save_json'](out / 'generation_timing.json', summary)
    print('FULL_GENERATION_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
