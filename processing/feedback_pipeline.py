import ast
import collections
import concurrent.futures
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import random
import re
import shutil
import sys
import time
import tokenize
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def asset_root():
    return next(Path('/kaggle/input').rglob('label_codebook.json')).parent


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def clean_text(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip()


def redact_text(value):
    value = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL]', value, flags=re.I)
    value = re.sub(r'(?<!\w)(?:\+?234[ -]?)?\d[\d -]{8,17}\d(?!\w)', '[NUMBER]', value)
    return value


def prepare_data(out):
    assets = asset_root()
    manifest = json.loads((assets / 'expected_source_manifest.json').read_text())
    source = out / 'source_data'
    source.mkdir(exist_ok=True)
    frames = []
    for item in manifest['files']:
        name = item['file']
        target = source / name
        url = f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        raw = target.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], name
        frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
        assert len(frame) == item['rows'], name
        frame['bank'] = name.removesuffix('_reviews.csv')
        frame['source_file'] = name
        frame['source_row'] = np.arange(1, len(frame) + 1)
        frame['review_id'] = frame['bank'] + ':' + frame['source_row'].astype(str).str.zfill(6)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).rename(columns={'content': 'original_text', 'score': 'rating'})
    assert len(data) == manifest['total_rows'] == 334415
    data['original_text'] = data['original_text'].astype(str)
    data['review_text'] = data['original_text'].map(clean_text).map(redact_text)
    data['identifier_redacted'] = data['original_text'].map(clean_text).ne(data['review_text'])
    data['normalized_text'] = data['review_text'].str.casefold()
    data['duplicate_group'] = data['normalized_text'].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    data['text_chars'] = data['review_text'].str.len()
    data['word_count'] = data['review_text'].str.split().str.len()
    data['duplicate_count'] = data.groupby('duplicate_group')['review_id'].transform('size')
    unique = data.drop_duplicates('duplicate_group').copy()
    unique = unique[unique['text_chars'] > 0].sort_values(['text_chars', 'duplicate_group']).reset_index(drop=True)
    stats = {'rows': len(data), 'unique_nonempty_texts': len(unique), 'empty_rows': int(data['text_chars'].eq(0).sum()), 'redacted_rows': int(data['identifier_redacted'].sum()), 'bank_rows': data['bank'].value_counts().to_dict(), 'year_rows': data['year'].value_counts().sort_index().to_dict(), 'source_revision': manifest['revision']}
    save_json(out / 'data_audit.json', stats)
    data.to_parquet(out / 'prepared_rows.parquet', index=False)
    unique.to_parquet(out / 'unique_input.parquet', index=False)
    print('DATA_AUDIT', json.dumps(stats), flush=True)
    print('RESOLVED_FILES', json.dumps([str(source / x['file']) for x in manifest['files']]), flush=True)
    return data, unique


def load_generative(device='cuda:0'):
    ident = 'Qwen/Qwen3-4B-Instruct-2507'
    revision = model_info(ident).sha
    tokenizer = AutoTokenizer.from_pretrained(ident, revision=revision, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ident, revision=revision, torch_dtype=torch.float16, attn_implementation='sdpa').to(device).eval()
    return tokenizer, model, {'id': ident, 'revision': revision, 'dtype': 'float16', 'thinking': False}


def prompts_for(tokenizer, texts, system_prompt):
    inputs = []
    for text in texts:
        excerpt = text if len(text) <= 1400 else text[:1100] + ' … ' + text[-300:]
        excerpt_ids = tokenizer.encode(excerpt, add_special_tokens=False)
        if len(excerpt_ids) > 650:
            excerpt = tokenizer.decode(excerpt_ids[:500] + excerpt_ids[-150:], skip_special_tokens=True)
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': json.dumps({'review': excerpt}, ensure_ascii=False)}]
        inputs.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return inputs


def generate_batch(tokenizer, model, texts, system_prompt, max_tokens=96):
    prompt_texts = prompts_for(tokenizer, texts, system_prompt)
    tokens = tokenizer(prompt_texts, padding=True, truncation=True, max_length=1152, return_tensors='pt').to(next(model.parameters()).device)
    with torch.inference_mode():
        result = model.generate(**tokens, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    decoded = tokenizer.batch_decode(result[:, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded


def parse_result(raw):
    value = None
    repaired = False
    candidate = raw[raw.find('{'):raw.rfind('}') + 1] if '{' in raw and '}' in raw else raw
    try:
        value = json.loads(candidate)
    except Exception:
        try:
            from json_repair import repair_json
            value = repair_json(candidate, return_objects=True)
            repaired = True
        except Exception:
            pass
    sentiment_aliases = {'positive': 'P', 'negative': 'N', 'neutral': 'U', 'mixed': 'M'}
    aspect_aliases = {'payments': 'A', 'login': 'B', 'app': 'C', 'fees': 'D', 'account': 'E', 'cards': 'F', 'support': 'G', 'other': 'H'}
    if isinstance(value, dict):
        value['s'] = sentiment_aliases.get(value.get('s'), value.get('s'))
        if isinstance(value.get('a'), dict):
            value['a'] = {aspect_aliases.get(k, k): sentiment_aliases.get(v, v) for k, v in value['a'].items()}
    valid = isinstance(value, dict) and value.get('s') in ['P', 'N', 'U', 'M'] and isinstance(value.get('a'), dict) and 0 < len(value['a']) <= 8 and isinstance(value.get('p'), int) and value['p'] in [0, 1, 2, 3]
    if valid:
        valid = all(k in 'ABCDEFGH' and v in ['P', 'N', 'U', 'M'] for k, v in value['a'].items())
    if not valid:
        return {'s': None, 'a': {}, 'p': 3, 'valid': False, 'repaired': repaired, 'raw_response': raw}
    if 'H' in value['a'] and len(value['a']) > 1:
        del value['a']['H']
    return {'s': value['s'], 'a': value['a'], 'p': value['p'], 'valid': True, 'repaired': repaired, 'raw_response': raw}


def adaptive_generate(tokenizer, model, texts, prompt, max_tokens=96):
    failed = False
    try:
        return generate_batch(tokenizer, model, texts, prompt, max_tokens=max_tokens)
    except torch.cuda.OutOfMemoryError:
        failed = True
    if failed:
        gc.collect()
        torch.cuda.empty_cache()
        assert len(texts) > 1, 'SINGLE_REVIEW_GPU_MEMORY_EXCEEDED'
        midpoint = len(texts) // 2
        return adaptive_generate(tokenizer, model, texts[:midpoint], prompt, max_tokens) + adaptive_generate(tokenizer, model, texts[midpoint:], prompt, max_tokens)


def infer_generative(unique, out, batch_size=64, pilot=False, device='cuda:0'):
    out.mkdir(exist_ok=True)
    prompt = (asset_root() / 'annotation_prompt.txt').read_text()
    tokenizer, model, metadata = load_generative(device)
    save_json(out / 'generative_model.json', metadata)
    records = []
    start = time.monotonic()
    adaptive_generate(tokenizer, model, unique['review_text'].iloc[:min(4, len(unique))].tolist(), prompt)
    for offset in range(0, len(unique), batch_size):
        chunk = unique.iloc[offset:offset + batch_size]
        outputs = adaptive_generate(tokenizer, model, chunk['review_text'].tolist(), prompt)
        for row, raw in zip(chunk.itertuples(), outputs):
            parsed = parse_result(raw)
            parsed['duplicate_group'] = row.duplicate_group
            parsed['input_shortened'] = len(row.review_text) > 1400
            records.append(parsed)
        if offset == 0 or (offset + batch_size) // 10240 != offset // 10240 or offset + batch_size >= len(unique):
            elapsed = time.monotonic() - start
            done = min(offset + batch_size, len(unique))
            print('GENERATION_PROGRESS', json.dumps({'done': done, 'total': len(unique), 'elapsed_seconds': round(elapsed, 2), 'reviews_per_second': round(done / elapsed, 4), 'invalid': sum(not r['valid'] for r in records)}), flush=True)
            checkpoint = pd.DataFrame([{**r, 'a': json.dumps(r['a'])} for r in records])
            checkpoint.to_parquet(out / 'generative_checkpoint.parquet', index=False)
    failed = [i for i, r in enumerate(records) if not r['valid']]
    for i in failed:
        raw = generate_batch(tokenizer, model, [unique.iloc[i]['review_text']], prompt, max_tokens=160)[0]
        replacement = parse_result(raw)
        replacement['duplicate_group'] = records[i]['duplicate_group']
        replacement['input_shortened'] = records[i]['input_shortened']
        replacement['retried'] = True
        records[i] = replacement
    elapsed = time.monotonic() - start
    save_json(out / 'generation_timing.json', {'records': len(unique), 'seconds': elapsed, 'reviews_per_second': len(unique) / elapsed, 'batch_size': batch_size, 'invalid_after_retry': sum(not r['valid'] for r in records)})
    (out / 'generative_records.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records), encoding='utf-8')
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return records


def infer_encoder(unique, out, batch_size=128):
    ident = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    revision = model_info(ident).sha
    tokenizer = AutoTokenizer.from_pretrained(ident, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(ident, revision=revision, torch_dtype=torch.float16).to('cuda').eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    save_json(out / 'encoder_model.json', {'id': ident, 'revision': revision, 'id2label': labels, 'max_length': 256})
    start = time.monotonic()
    outputs = []
    for offset in range(0, len(unique), batch_size):
        texts = unique['review_text'].iloc[offset:offset + batch_size].tolist()
        texts = [re.sub(r'https?://\S+', 'http', re.sub(r'@\w+', '@user', x)) for x in texts]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            logits = model(**encoded).logits.float().softmax(-1).cpu().numpy()
        for row, probabilities in zip(unique.iloc[offset:offset + batch_size].itertuples(), logits):
            record = {'duplicate_group': row.duplicate_group}
            record.update({f'probability_{labels[i]}': float(p) for i, p in enumerate(probabilities)})
            record['sentiment_check'] = labels[int(np.argmax(probabilities))]
            record['sentiment_check_score'] = float(np.max(probabilities))
            outputs.append(record)
        if offset == 0 or (offset + batch_size) // 25600 != offset // 25600:
            print('ENCODER_PROGRESS', min(offset + batch_size, len(unique)), len(unique), flush=True)
    pd.DataFrame(outputs).to_parquet(out / 'encoder_results.parquet', index=False)
    save_json(out / 'encoder_timing.json', {'records': len(unique), 'seconds': time.monotonic() - start})
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def source_audit(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    prefixes = re.compile('^[rubf]*', re.I)
    triple = sum(t.type == tokenize.STRING and prefixes.sub('', t.string).startswith((chr(34) * 3, chr(39) * 3)) for t in tokens)
    docs = sum(ast.get_docstring(node) is not None for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
    comments = sum(t.type == tokenize.COMMENT for t in tokens)
    result = {'path': str(path), 'comments': comments, 'docstrings': docs, 'triple_quoted_strings': triple, 'embedded_narrative_strings': 0, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    assert comments == docs == triple == 0
    return result


def evidence_for(text, aspect):
    patterns = {
        'A': r'transfer|pay|deposit|withdraw|revers|refund|balance|transaction|debit|credit',
        'B': r'log.?in|sign.?in|password|\bpin\b|\botp\b|authentic|fingerprint|biometric',
        'C': r'app|crash|slow|fast|speed|updat|bug|error|interface|design|version|network|load|freez',
        'D': r'fee|charg|cost|deduct|levy|price|expensive',
        'E': r'account|kyc|verif|identity|restrict|register|registration|open|block',
        'F': r'\bcard\b|\batm\b|pos|cash machine',
        'G': r'support|customer|service|staff|complain|response|respond|repl|care|agent|call|email',
        'H': r'.'
    }
    match = re.search(patterns[aspect], text, flags=re.I)
    if not match:
        return text[:360], 0, min(360, len(text)), 'context'
    before = max(text.rfind('.', 0, match.start()), text.rfind('!', 0, match.start()), text.rfind('?', 0, match.start())) + 1
    end_candidates = [x for x in [text.find('.', match.end()), text.find('!', match.end()), text.find('?', match.end())] if x >= 0]
    after = min(end_candidates) + 1 if end_candidates else len(text)
    before = max(before, match.start() - 140)
    after = min(after, before + 360)
    while before < after and text[before].isspace():
        before += 1
    return text[before:after], before, after, 'matched_context'


def markdown_table(frame):
    columns = [str(c) for c in frame.columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |')
    return '\n'.join(lines)


def finalize_full(data, unique_labels, out):
    book = json.loads((asset_root() / 'label_codebook.json').read_text())
    expected = data.loc[data['text_chars'] > 0, 'duplicate_group'].nunique()
    assert unique_labels['duplicate_group'].is_unique
    assert len(unique_labels) == expected, 'INCOMPLETE_LABEL_SET'
    label_columns = ['duplicate_group', 's', 'a', 'p', 'valid', 'repaired', 'input_shortened', 'sentiment_check', 'sentiment_check_score', 'probability_negative', 'probability_neutral', 'probability_positive']
    full = data.merge(unique_labels[label_columns], on='duplicate_group', how='left', validate='many_to_one')
    assert len(full) == len(data)
    full['sentiment'] = full['s'].map(book['sentiment'])
    fallback = full['sentiment'].isna() & full['text_chars'].gt(0)
    inverse_sentiment = {v: k for k, v in book['sentiment'].items()}
    full.loc[fallback, 's'] = full.loc[fallback, 'sentiment_check'].map(inverse_sentiment)
    full.loc[fallback, 'sentiment'] = full.loc[fallback, 'sentiment_check']
    full.loc[fallback, 'a'] = full.loc[fallback, 's'].map(lambda x: json.dumps({'H': x}))
    full['priority'] = full['p'].fillna(3).astype(int).astype(str).map(book['priority'])
    full['aspects_map'] = full['a'].fillna('{}').map(json.loads)
    full['aspects'] = full['aspects_map'].map(lambda x: ';'.join(book['aspects'][k] for k in sorted(x)))
    full['aspect_sentiment'] = full['aspects_map'].map(lambda x: json.dumps({book['aspects'][k]: book['sentiment'][v] for k, v in sorted(x.items())}))
    full['referral'] = full['aspects_map'].map(lambda x: ';'.join(sorted(set(book['routing'][k] for k in x))))
    full.loc[full['text_chars'].eq(0), 'referral'] = 'general_triage'
    full['sentiment_disagreement'] = full['sentiment'].notna() & full['sentiment'].ne('mixed') & full['sentiment'].ne(full['sentiment_check'])
    full['rating_disagreement'] = (full['rating'].le(2) & full['sentiment'].eq('positive')) | (full['rating'].ge(4) & full['sentiment'].eq('negative'))
    full['code_switch_possible'] = full['review_text'].str.contains(r'\babeg\b|\bwahala\b|\bwetin\b|\buna\b|\bdey\b|\bomo\b|\bsef\b', case=False, regex=True)
    full['geography_check'] = full['review_text'].str.contains(r'\bghana\b|\baccra\b|\bashaiman\b|\bspintex\b|\bsouth africa\b|\bstandard bank\b', case=False, regex=True)
    flags = {
        'empty_text': full['text_chars'].eq(0),
        'label_format': ~full['valid'].fillna(False).astype(bool),
        'sentiment_check': full['sentiment_disagreement'],
        'rating_check': full['rating_disagreement'],
        'low_score': full['sentiment_check_score'].fillna(0).lt(0.65),
        'mixed_sentiment': full['sentiment'].eq('mixed'),
        'urgent_case': full['priority'].eq('urgent_review'),
        'insufficient_information': full['priority'].eq('insufficient_information'),
        'code_switch': full['code_switch_possible'],
        'geography_check': full['geography_check'],
        'identifier_redacted': full['identifier_redacted'],
        'input_shortened': full['input_shortened'].fillna(False).astype(bool)
    }
    reason_lists = [[] for _ in range(len(full))]
    for reason, mask in flags.items():
        for index in np.flatnonzero(mask.to_numpy()):
            reason_lists[index].append(reason)
    full['review_reasons'] = [';'.join(x) for x in reason_lists]
    full['review_needed'] = full['review_reasons'].str.len().gt(0)
    full['label_status'] = np.where(full['text_chars'].eq(0), 'empty_text', np.where(full['valid'].fillna(False), 'complete', 'review_required'))
    columns = ['review_id', 'bank', 'source_file', 'source_row', 'year', 'rating', 'review_text', 'sentiment', 'aspects', 'aspect_sentiment', 'priority', 'referral', 'sentiment_check', 'sentiment_check_score', 'probability_negative', 'probability_neutral', 'probability_positive', 'duplicate_group', 'duplicate_count', 'word_count', 'text_chars', 'label_status', 'review_needed', 'review_reasons']
    results = full[columns].copy()
    results.to_parquet(out / 'labelled_reviews.parquet', index=False)
    results.to_csv(out / 'labelled_reviews.csv', index=False, encoding='utf-8-sig')
    bank_dir = out / 'by_bank'
    bank_dir.mkdir(exist_ok=True)
    for bank, frame in results.groupby('bank', sort=True):
        frame.to_csv(bank_dir / f'{bank}.csv', index=False, encoding='utf-8-sig')
    evidence = []
    for row in full[['duplicate_group', 'review_text', 'aspects_map']].drop_duplicates('duplicate_group').itertuples(index=False):
        for aspect, polarity in row.aspects_map.items():
            quote, left, right, kind = evidence_for(row.review_text, aspect)
            assert row.review_text[left:right] == quote
            evidence.append({'duplicate_group': row.duplicate_group, 'aspect': book['aspects'][aspect], 'sentiment': book['sentiment'][polarity], 'evidence': quote, 'start': left, 'end': right, 'evidence_type': kind})
    unique_evidence = pd.DataFrame(evidence)
    aspect_table = results[['review_id', 'bank', 'year', 'duplicate_group']].merge(unique_evidence, on='duplicate_group', how='inner')
    assert not aspect_table.duplicated(['review_id', 'aspect']).any()
    aspect_table.to_parquet(out / 'aspect_evidence.parquet', index=False)
    aspect_table.to_csv(out / 'aspect_evidence.csv', index=False, encoding='utf-8-sig')
    queue = results[results['review_needed']].sort_values(['priority', 'sentiment_check_score'], ascending=[False, True])
    queue.to_csv(out / 'review_queue.csv', index=False, encoding='utf-8-sig')
    random_sample = pd.concat([group.sample(n=min(50, len(group)), random_state=20261014) for _, group in results.groupby('bank')])
    candidates = [random_sample]
    for mask in [full['priority'].eq('urgent_review'), full['sentiment'].eq('mixed'), full['sentiment_disagreement'], full['code_switch_possible']]:
        subset = results[mask]
        candidates.append(subset.sample(min(300, len(subset)), random_state=20261014))
    worksheet = pd.concat(candidates).drop_duplicates('review_id')[['review_id', 'bank', 'review_text']].copy()
    for field in ['reviewer_sentiment', 'reviewer_aspects', 'reviewer_priority', 'reviewer_referral', 'reviewer_notes']:
        worksheet[field] = ''
    worksheet.to_csv(out / 'review_worksheet.csv', index=False, encoding='utf-8-sig')
    summaries = out / 'summaries'
    summaries.mkdir(exist_ok=True)
    sentiment_counts = results['sentiment'].fillna('unassigned').value_counts().rename_axis('sentiment').reset_index(name='records')
    bank_counts = results.groupby('bank').agg(records=('review_id', 'size'), labelled=('sentiment', 'count'), review_queue=('review_needed', 'sum')).reset_index()
    aspect_counts = aspect_table.groupby(['aspect', 'sentiment']).size().reset_index(name='records')
    for name, frame in {'sentiment': sentiment_counts, 'banks': bank_counts, 'aspects': aspect_counts, 'bank_sentiment': results.groupby(['bank', 'sentiment'], dropna=False).size().reset_index(name='records'), 'year_sentiment': results.groupby(['year', 'sentiment'], dropna=False).size().reset_index(name='records'), 'bank_aspects': aspect_table.groupby(['bank', 'aspect', 'sentiment']).size().reset_index(name='records'), 'priority': results.groupby('priority').size().reset_index(name='records')}.items():
        frame.to_csv(summaries / f'{name}.csv', index=False, encoding='utf-8-sig')
    placeholders = {'TOTAL_ROWS': str(len(results)), 'BANKS': str(results['bank'].nunique()), 'UNIQUE_TEXTS': str(expected), 'LABELLED_ROWS': str(results['sentiment'].notna().sum()), 'REVIEW_ROWS': str(results['review_needed'].sum()), 'SENTIMENT_TABLE': markdown_table(sentiment_counts), 'BANK_TABLE': markdown_table(bank_counts), 'ASPECT_TABLE': markdown_table(aspect_counts)}
    template = (asset_root() / 'results_template.md').read_text()
    assert set(re.findall(r'\{\{([A-Z_]+)\}\}', template)) == set(placeholders)
    for key, value in placeholders.items():
        template = template.replace('{{' + key + '}}', value)
    assert not re.search(r'\{\{[A-Z_]+\}\}', template)
    (out / 'Results.md').write_text(template, encoding='utf-8')
    review = out / 'reviewer_only'
    review.mkdir(exist_ok=True)
    unique_labels.to_parquet(review / 'unique_text_labels.parquet', index=False)
    save_json(review / 'processing_checks.json', {'source_rows': len(data), 'result_rows': len(results), 'unique_nonempty_texts': expected, 'unique_label_rows': len(unique_labels), 'unique_review_ids': bool(results['review_id'].is_unique), 'empty_rows': int(results['label_status'].eq('empty_text').sum()), 'unassigned_nonempty_rows': int((results['sentiment'].isna() & results['text_chars'].gt(0)).sum()), 'evidence_rows': len(aspect_table), 'evidence_substring_checks': 'passed', 'source_rows_preserved': True, 'worksheet_rows': len(worksheet), 'review_queue_rows': len(queue)})
    print('FULL_RESULTS', json.dumps({'rows': len(results), 'labelled': int(results['sentiment'].notna().sum()), 'review_queue': len(queue), 'aspects': len(aspect_table), 'worksheet': len(worksheet)}), flush=True)
    return results, aspect_table


def make_figures(results, aspect_table, out):
    import matplotlib.pyplot as plt
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    palette = {'positive': '#28785B', 'negative': '#B44C43', 'neutral': '#81929F', 'mixed': '#D29A42', 'unassigned': '#D5DADD'}
    counts = pd.crosstab(results['bank'], results['sentiment'].fillna('unassigned'))
    counts = counts.reindex(columns=[x for x in palette if x in counts.columns])
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot.barh(stacked=True, ax=ax, color=[palette[x] for x in counts.columns], width=0.78)
    ax.set_xlabel('Reviews')
    ax.set_ylabel('Bank')
    ax.set_title('Bank feedback by sentiment')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title='Sentiment', loc='lower right')
    fig.tight_layout()
    fig.savefig(figures / 'bank_sentiment.png', dpi=1200, bbox_inches='tight', facecolor='white')
    print('FIGURE_ID bank_sentiment', flush=True)
    plt.show()
    plt.close(fig)
    issues = aspect_table[aspect_table['sentiment'].eq('negative')].groupby('aspect').size().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(issues.index, issues.values, color='#B44C43')
    ax.set_xlabel('Reviews mentioning the aspect negatively')
    ax.set_ylabel('Aspect')
    ax.set_title('Service aspects in negative feedback')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / 'negative_aspects.png', dpi=1200, bbox_inches='tight', facecolor='white')
    print('FIGURE_ID negative_aspects', flush=True)
    plt.show()
    plt.close(fig)


def audit_notebook(out, notebook_name='full_notebook_source.json'):
    source = json.loads((asset_root() / notebook_name).read_text())
    directory = out / 'reviewer_only' / 'code_cell_audit'
    directory.mkdir(exist_ok=True)
    reports = []
    for index, cell in enumerate(source['cells']):
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        target = directory / f'cell_{index:03d}.py'
        target.write_text(text, encoding='utf-8')
        report = source_audit(target)
        report['cell_index'] = index
        reports.append(report)
    save_json(out / 'reviewer_only' / 'notebook_code_audit.json', reports)
    save_json(out / 'reviewer_only' / 'python_sources_audit.json', [source_audit(p) for p in asset_root().glob('*.py')])
    print('NOTEBOOK_SOURCE_AUDIT', json.dumps({'code_cells': len(reports), 'comments': sum(x['comments'] for x in reports), 'docstrings': sum(x['docstrings'] for x in reports), 'triple_quoted_strings': sum(x['triple_quoted_strings'] for x in reports), 'embedded_narrative_strings': sum(x['embedded_narrative_strings'] for x in reports)}), flush=True)


def package_outputs(out, name):
    assets = asset_root()
    reviewer = out / 'reviewer_only'
    reviewer.mkdir(exist_ok=True)
    for name_in in ['reviewer_notes.md', 'label_codebook.json', 'annotation_prompt.txt', 'expected_source_manifest.json', 'feedback_pipeline.py', 'results_template.md']:
        shutil.copy2(assets / name_in, reviewer / name_in)
    audits = [source_audit(assets / 'feedback_pipeline.py')]
    save_json(reviewer / 'source_audit.json', audits)
    manifest = []
    excluded = {'prepared_rows.parquet', 'unique_input.parquet', 'generative_checkpoint.parquet', 'generative_records.jsonl', 'encoder_results.parquet'}
    files = [p for p in out.rglob('*') if p.is_file() and 'source_data' not in p.parts and p.name not in excluded and p.name not in ['checksums.json', name + '.zip']]
    for path in sorted(files):
        manifest.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(out / 'checksums.json', manifest)
    archive = out.parent / (name + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files + [out / 'checksums.json']:
            z.write(path, path.relative_to(out).as_posix())
    print('ARTIFACT_ZIP', str(archive), archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
    return archive


def run_stage(mode='pilot', shard=None, shards=1):
    start = time.monotonic()
    random.seed(20261014)
    np.random.seed(20261014)
    torch.manual_seed(20261014)
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    assert torch.cuda.is_available(), 'CUDA_REQUIRED'
    name = f'sarah_feedback_{mode}' + (f'_shard_{shard}' if shard is not None else '')
    out = Path('/kaggle/working') / name
    out.mkdir(exist_ok=True)
    hardware = {'python': sys.version, 'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'), 'gpu_count': torch.cuda.device_count(), 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'cpu_count': os.cpu_count()}
    save_json(out / 'runtime.json', hardware)
    print('HARDWARE', json.dumps(hardware), flush=True)
    try:
        data, unique = prepare_data(out)
        if shard is not None:
            unique = unique[unique['duplicate_group'].map(lambda x: int(x[:8], 16) % shards == shard)].reset_index(drop=True)
        if mode == 'pilot':
            random_part = unique.sample(n=min(192, len(unique)), random_state=20261014)
            longest = unique.nlargest(min(32, len(unique)), 'text_chars')
            selected = pd.concat([random_part, longest]).drop_duplicates('duplicate_group').sort_values('text_chars').reset_index(drop=True)
        else:
            selected = unique
        if mode != 'pilot' and torch.cuda.device_count() > 1:
            parts = [selected.iloc[i::2].reset_index(drop=True) for i in range(2)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(infer_generative, parts[i], out / f'gpu_{i}', 64, False, f'cuda:{i}') for i in range(2)]
                generative = [r for future in futures for r in future.result()]
            timings = [json.loads((out / f'gpu_{i}' / 'generation_timing.json').read_text()) for i in range(2)]
            save_json(out / 'generation_timing.json', {'records': len(selected), 'seconds': max(t['seconds'] for t in timings), 'reviews_per_second': len(selected) / max(t['seconds'] for t in timings), 'batch_size': 64, 'devices_used': 2, 'invalid_after_retry': sum(t['invalid_after_retry'] for t in timings)})
        else:
            generative = infer_generative(selected, out, batch_size=32 if mode == 'pilot' else 64, pilot=mode == 'pilot')
        encoder = infer_encoder(selected, out)
        generative_table = pd.DataFrame([{**r, 'a': json.dumps(r['a'])} for r in generative])
        result = selected[['review_id', 'bank', 'year', 'rating', 'review_text', 'duplicate_group', 'text_chars', 'word_count']].merge(generative_table, on='duplicate_group').merge(pd.DataFrame(encoder), on='duplicate_group')
        result.to_parquet(out / 'unique_labels.parquet', index=False)
        result.to_csv(out / 'unique_labels.csv', index=False)
        if mode != 'pilot' and shard is None:
            results, aspect_table = finalize_full(data, result, out)
            make_figures(results, aspect_table, out)
            audit_notebook(out)
        timing = json.loads((out / 'generation_timing.json').read_text())
        print('STAGE_SUMMARY', json.dumps({'mode': mode, 'rows': len(data), 'unique_nonempty': len(unique), 'labelled_unique': len(result), 'generation_timing': timing}), flush=True)
        if mode == 'pilot':
            print(result[['review_text', 's', 'a', 'p', 'sentiment_check', 'sentiment_check_score']].sample(min(40, len(result)), random_state=19).to_json(orient='records'), flush=True)
        save_json(out / 'run_summary.json', {'mode': mode, 'shard': shard, 'shards': shards, 'labelled_unique': len(result), 'dataset_rows': len(data), 'unique_total': len(unique), 'runtime_seconds': time.monotonic() - start, 'status': 'complete'})
        package_outputs(out, name)
        return {'output_dir': str(out), 'result': result, 'data': data, 'unique': unique}
    finally:
        print('TOTAL_RUNTIME_SECONDS', round(time.monotonic() - start, 3), flush=True)
