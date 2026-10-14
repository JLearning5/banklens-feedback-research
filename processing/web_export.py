import gzip
import hashlib
import json


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    else:
        path.write_bytes(raw)


def export_web(results, out, assets):
    root = out / 'web_data'
    banks = sorted(results['bank'].unique().tolist())
    sentiments = ['positive', 'negative', 'neutral', 'mixed', 'unassigned']
    priorities = ['routine', 'prompt_attention', 'urgent_review', 'insufficient_information']
    codebook = json.loads((assets / 'label_codebook.json').read_text())
    aspects = list(codebook['aspects'].values())
    catalog = []
    search_texts = []
    shard = []
    cube = {}
    for seq, row in enumerate(results.itertuples(index=False)):
        bank = banks.index(row.bank)
        year = int(row.year) if row.year == row.year else 0
        rating = int(row.rating) if row.rating == row.rating else 0
        sentiment = sentiments.index(row.sentiment) if isinstance(row.sentiment, str) else 4
        priority = priorities.index(row.priority)
        pairs = json.loads(row.aspect_sentiment)
        mask = sum(1 << aspects.index(a) for a in pairs)
        needed = int(row.review_needed)
        catalog.append([bank, year, rating, sentiment, mask, priority, needed, int(row.duplicate_count)])
        search_texts.append(row.review_text.casefold())
        details = {'i': seq, 'id': row.review_id, 'text': row.review_text, 'aspects': pairs, 'referral': row.referral.split(';'), 'reasons': row.review_reasons.split(';') if row.review_reasons else [], 'group': row.duplicate_group, 'source': row.source_file, 'row': int(row.source_row), 'status': row.label_status}
        shard.append(details)
        if len(shard) == 2000 or seq == len(results) - 1:
            dump(root / 'rows' / f'{seq // 2000:03d}.json', shard)
            shard = []
        key = f'{bank},{year},{sentiment},{priority},{mask},{needed}'
        cube[key] = cube.get(key, 0) + 1
    dump(root / 'catalog.json.gz', catalog)
    dump(root / 'search.json.gz', search_texts)
    dump(root / 'cube.json', [[*[int(v) for v in key.split(',')], count] for key, count in sorted(cube.items())])
    metadata = {'rows': len(results), 'banks': banks, 'sentiments': sentiments, 'priorities': priorities, 'aspects': aspects, 'years': sorted(set(v[1] for v in catalog)), 'shard_size': 2000, 'version': '2026-10-14', 'catalog_fields': ['bank', 'year', 'rating', 'sentiment', 'aspects', 'priority', 'review_needed', 'duplicate_count']}
    dump(root / 'metadata.json', metadata)
    assert len(catalog) == len(search_texts) == len(results)
    assert sum(cube.values()) == len(results)
    assert len({row.review_id for row in results.itertuples(index=False)}) == len(results)
    inventory = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]
    dump(root / 'manifest.json', inventory)
    summary = {'rows': len(results), 'bank_count': len(banks), 'files': len(inventory), 'bytes': sum(p['bytes'] for p in inventory), 'largest_file_bytes': max(p['bytes'] for p in inventory), 'cube_count': len(cube), 'checks': 'passed'}
    dump(out / 'reviewer_only' / 'web_export_checks.json', summary)
    print('WEB_EXPORT', json.dumps(summary), flush=True)
    return summary
