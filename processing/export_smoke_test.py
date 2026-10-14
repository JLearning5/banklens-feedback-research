import json
import runpy
from pathlib import Path

import pandas as pd


def main():
    assets = next(Path('/kaggle/input').rglob('feedback_pipeline.py')).parent
    ctx = runpy.run_path(str(assets / 'feedback_pipeline.py'))
    out = Path('/kaggle/working/export_smoke')
    out.mkdir(exist_ok=True)
    rows = []
    for i, (text, group) in enumerate([('Good app', 'one'), ('Failed transfer and unfair charges', 'two'), ('Good app', 'one'), ('', 'empty')]):
        rows.append({'review_id': str(i), 'bank': 'example', 'source_file': 'fixture.csv', 'source_row': i + 1, 'year': 2023, 'rating': 5 if group == 'one' else 1, 'review_text': text, 'duplicate_group': group, 'duplicate_count': 2 if group == 'one' else 1, 'word_count': len(text.split()), 'text_chars': len(text), 'identifier_redacted': False})
    data = pd.DataFrame(rows)
    labels = []
    for group, sentiment, aspects, priority in [('one', 'P', {'C': 'P'}, 0), ('two', 'N', {'A': 'N', 'D': 'N'}, 1)]:
        labels.append({'duplicate_group': group, 's': sentiment, 'a': json.dumps(aspects), 'p': priority, 'valid': True, 'repaired': False, 'input_shortened': False, 'sentiment_check': 'positive' if sentiment == 'P' else 'negative', 'sentiment_check_score': 0.95, 'probability_positive': 0.95 if sentiment == 'P' else 0.02, 'probability_negative': 0.95 if sentiment == 'N' else 0.02, 'probability_neutral': 0.03})
    results, evidence = ctx['finalize_full'](data, pd.DataFrame(labels), out)
    assert len(results) == 4
    assert results['review_id'].is_unique
    assert results.loc[0, 'sentiment'] == results.loc[2, 'sentiment'] == 'positive'
    assert pd.isna(results.loc[3, 'sentiment'])
    assert results.loc[3, 'label_status'] == 'empty_text'
    assert len(evidence) == 4
    check = json.loads((out / 'reviewer_only' / 'processing_checks.json').read_text())
    assert check['source_rows_preserved'] and check['evidence_substring_checks'] == 'passed'
    audits = [ctx['source_audit'](p) for p in assets.glob('*.py')]
    ctx['save_json'](out / 'source_audits.json', audits)
    web_source = assets / 'web_export.py'
    if web_source.exists():
        web = runpy.run_path(str(web_source))
        summary = web['export_web'](results, out, assets)
        assert summary['rows'] == 4
        assert summary['bank_count'] == 1
        assert summary['largest_file_bytes'] < 25 * 1024 * 1024
    ctx['save_json'](out / 'smoke_test.json', {'status': 'passed', 'rows': len(results), 'evidence_rows': len(evidence), 'source_row_preservation': True, 'duplicate_inheritance': True, 'empty_text_handling': True})
    print('EXPORT_SMOKE_TEST', json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
