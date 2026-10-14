import gc
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    from huggingface_hub import model_info
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    assets = next(Path('/kaggle/input').rglob('feedback_pipeline.py')).parent
    ctx = runpy.run_path(str(assets / 'feedback_pipeline.py'))
    out = Path('/kaggle/working/sarah_feedback_full')
    out.mkdir(exist_ok=True)
    data, unique = ctx['prepare_data'](out)
    ident = 'Qwen/Qwen3-4B-Instruct-2507'
    revision = model_info(ident).sha
    tokenizer = AutoTokenizer.from_pretrained(ident, revision=revision)
    prompt = (assets / 'annotation_prompt.txt').read_text()
    sentiments = ['positive', 'negative', 'neutral', 'mixed']
    aspect_keys = ['payments', 'login', 'app', 'fees', 'account', 'cards', 'support', 'other']
    schema = {'type': 'object', 'properties': {'s': {'type': 'string', 'enum': sentiments}, 'a': {'type': 'object', 'properties': {k: {'type': 'string', 'enum': sentiments} for k in aspect_keys}, 'additionalProperties': False, 'minProperties': 1}, 'p': {'type': 'integer', 'enum': [0, 1, 2, 3]}}, 'required': ['s', 'a', 'p'], 'additionalProperties': False}
    metadata = {'id': ident, 'revision': revision, 'dtype': 'float16', 'engine': 'vllm', 'tensor_parallel_size': 2, 'schema': schema, 'seed': 20261014, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    ctx['save_json'](out / 'generative_model.json', metadata)
    model = LLM(model=ident, revision=revision, dtype='float16', tensor_parallel_size=2, gpu_memory_utilization=0.80, max_model_len=1536, max_num_seqs=128, max_num_batched_tokens=4096, enable_prefix_caching=True, disable_custom_all_reduce=True, attention_config={'backend': 'TRITON_ATTN'}, enforce_eager=True, seed=20261014)
    params = SamplingParams(temperature=0, max_tokens=192, structured_outputs=StructuredOutputsParams(json=schema))
    checkpoint = out / 'generative_records.jsonl'
    completed = {}
    for path in Path('/kaggle/input').rglob('generative_records.jsonl'):
        if 'sarah_feedback_full' not in str(path):
            continue
        previous = json.loads((path.parent / 'generative_model.json').read_text())
        assert all(previous[k] == metadata[k] for k in ['id', 'revision', 'prompt_sha256', 'schema', 'seed']), 'RESUME_CONFIG_MISMATCH'
        for line in path.read_text().splitlines():
            record = json.loads(line)
            completed[record['duplicate_group']] = record
    allowed = set(unique['duplicate_group'])
    completed = {k: v for k, v in completed.items() if k in allowed}
    selected = unique[~unique['duplicate_group'].isin(completed)].reset_index(drop=True)
    start = time.monotonic()
    initial_count = len(completed)
    checkpoint.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in completed.values()) + ('\n' if completed else ''), encoding='utf-8')
    with checkpoint.open('a', encoding='utf-8') as handle:
        for offset in range(0, len(selected), 4096):
            chunk = selected.iloc[offset:offset + 4096]
            prompts = ctx['prompts_for'](tokenizer, chunk['review_text'].tolist(), prompt)
            outputs = model.generate(prompts, params, use_tqdm=False)
            for row, response in zip(chunk.itertuples(), outputs):
                raw = response.outputs[0].text
                record = ctx['parse_result'](raw)
                record['duplicate_group'] = row.duplicate_group
                record['input_shortened'] = len(row.review_text) > 1400 or len(tokenizer.encode(row.review_text, add_special_tokens=False)) > 650
                completed[row.duplicate_group] = record
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            elapsed = time.monotonic() - start
            done = len(completed)
            print('FULL_GENERATION_PROGRESS', json.dumps({'done': done, 'total': len(unique), 'seconds': round(elapsed, 2), 'reviews_per_second': round((done - initial_count) / elapsed, 3), 'invalid': sum(not r['valid'] for r in completed.values())}), flush=True)
    invalid = [k for k, value in completed.items() if not value['valid']]
    lookup = unique.set_index('duplicate_group')['review_text'].to_dict()
    for offset in range(0, len(invalid), 256):
        keys = invalid[offset:offset + 256]
        outputs = model.generate(ctx['prompts_for'](tokenizer, [lookup[k] for k in keys], prompt), params, use_tqdm=False)
        for key, response in zip(keys, outputs):
            record = ctx['parse_result'](response.outputs[0].text)
            record['duplicate_group'] = key
            record['input_shortened'] = completed[key]['input_shortened']
            record['retried'] = True
            completed[key] = record
    checkpoint.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in completed.values()), encoding='utf-8')
    seconds = time.monotonic() - start
    assert len(completed) == len(unique)
    summary = {'records': len(completed), 'seconds': seconds, 'reviews_per_second': (len(completed) - initial_count) / seconds, 'resumed_records': initial_count, 'invalid_after_retry': sum(not r['valid'] for r in completed.values())}
    ctx['save_json'](out / 'generation_timing.json', summary)
    print('FULL_GENERATION_COMPLETE', json.dumps(summary), flush=True)
    del model
    gc.collect()


if __name__ == '__main__':
    main()
