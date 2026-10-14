import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def prepare_gold(data_path, review_path, config):
    data = pd.read_parquet(data_path)
    assert data['review_id'].is_unique, 'DUPLICATE_SOURCE_ID'
    review_path = Path(review_path)
    if review_path.suffix == '.json':
        payload = json.loads(review_path.read_text(encoding='utf-8-sig'))
        assert payload.get('schema_version') == 1, 'REVIEW_SCHEMA'
        review = pd.DataFrame(payload['decisions']).rename(columns={'sentiment': 'reviewer_sentiment'})
    else:
        review = pd.read_csv(review_path, keep_default_na=False)
    assert review['review_id'].is_unique, 'DUPLICATE_REVIEW_ID'
    assert set(review['review_id']).issubset(set(data['review_id'])), 'UNKNOWN_REVIEW_ID'
    assert 'reviewer_sentiment' in review, 'REVIEWER_LABELS_REQUIRED'
    review = review[review['reviewer_sentiment'].isin(config['classes'])].copy()
    gold = data[['review_id', 'review_text', 'duplicate_group', 'bank', 'year']].merge(review[['review_id', 'reviewer_sentiment']], on='review_id', validate='one_to_one')
    gold = gold[gold['review_text'].str.strip().str.len().gt(0)].reset_index(drop=True)
    assert not gold.empty, 'EMPTY_REVIEW_SET'
    assert gold.groupby('duplicate_group')['reviewer_sentiment'].nunique().max() == 1, 'GROUP_LABEL_CONFLICT'
    counts = gold.drop_duplicates('duplicate_group')['reviewer_sentiment'].value_counts()
    assert set(counts.index) == set(config['classes']), 'MISSING_SENTIMENT_CLASS'
    assert counts.min() >= config['min_groups_per_class'], 'INSUFFICIENT_CLASS_GROUPS'
    mapping = {name: index for index, name in enumerate(config['classes'])}
    gold['target'] = gold['reviewer_sentiment'].map(mapping)
    splitter = StratifiedGroupKFold(n_splits=config['folds'], shuffle=True, random_state=config['seed'])
    gold['fold'] = -1
    for fold, (_, validation) in enumerate(splitter.split(gold['review_text'], gold['target'], gold['duplicate_group'])):
        gold.loc[validation, 'fold'] = fold
    gold['split'] = gold['fold'].map(lambda fold: 'test' if fold == 0 else 'validation' if fold == 1 else 'train')
    assert gold.groupby('duplicate_group')['split'].nunique().max() == 1, 'SPLIT_LEAKAGE'
    assert all(set(group['target']) == set(mapping.values()) for _, group in gold.groupby('split')), 'SPLIT_CLASS_MISSING'
    manifest = gold[['review_id', 'duplicate_group', 'target', 'fold', 'split']].sort_values('review_id')
    raw = manifest.to_csv(index=False).encode()
    return gold, manifest, hashlib.sha256(raw).hexdigest()
