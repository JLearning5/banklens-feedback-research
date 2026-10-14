import asyncio
import csv
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from playwright.async_api import async_playwright


async def main():
    started = time.monotonic()
    assets = Path(os.environ['LIVE_ASSET_ROOT'])
    config = json.loads((assets / 'live_app_cases.json').read_text())
    output = Path('/kaggle/working/hosted_app_checks')
    output.mkdir(exist_ok=True)
    base = config['base_url']
    session = requests.Session()

    def fetch(name):
        response = session.get(base + name, timeout=120)
        response.raise_for_status()
        return response.content

    metadata = json.loads(fetch('data/metadata.json'))
    raw = fetch('data/catalog.json.gz')
    catalog = json.loads(gzip.decompress(raw) if raw[:2] == bytes([31, 139]) else raw)
    assert len(catalog) == metadata['rows'] == config['expected_rows'], 'CORPUS_ROW_MISMATCH'
    expected_hashes = json.loads((assets / 'hosted_source_hashes.json').read_text())
    checked_hashes = {}
    for name, expected in expected_hashes.items():
        checked_hashes[name] = hashlib.sha256(fetch(name)).hexdigest()
        assert checked_hashes[name] == expected, 'HOSTED_SOURCE_MISMATCH'

    def selection(filters):
        values = []
        for row in catalog:
            if 'bank' in filters and metadata['banks'][row[0]] != filters['bank']:
                continue
            if 'year' in filters and row[1] != filters['year']:
                continue
            if 'sentiment' in filters and metadata['sentiments'][row[3]] != filters['sentiment']:
                continue
            values.append(row)
        return values

    def aggregate_valid(answer, filters):
        rows = selection(filters)
        actual = answer.get('aggregate')
        if not actual or actual['total'] != len(rows):
            return False
        counts = [sum(row[3] == i for row in rows) for i in range(len(metadata['sentiments']))]
        banks = {name: sum(row[0] == i for row in rows) for i, name in enumerate(metadata['banks'])}
        return actual['sentiment'] == counts and all(item['total'] == banks[item['bank']] for item in actual['banks'])

    shard_cache = {}

    def exact_sources(answer):
        evidence = answer.get('evidence', [])
        for record in evidence:
            key = record['i'] // metadata['shard_size']
            if key not in shard_cache:
                shard_cache[key] = json.loads(fetch(f'data/rows/{key:03d}.json'))
            original = next(row for row in shard_cache[key] if row['i'] == record['i'])
            if original['id'] != record['id'] or original['text'] != record['text']:
                return False
        if answer.get('summary'):
            cited = [row for row in evidence if row['id'] in answer.get('citations', [])]
            return answer.get('extractive') is True and len(cited) == 1 and answer['summary'] in cited[0]['text']
        return True

    report = {'status': 'running', 'base_url': base, 'cases_version': config['version'], 'corpus_rows': len(catalog), 'hosted_source_hashes': checked_hashes, 'interface_checks': [], 'tasks': []}
    errors = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = await browser.new_context(viewport={'width': 1440, 'height': 960}, accept_downloads=True)
            page = await context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.goto(base + 'UI-fixed/', wait_until='domcontentloaded')
            await page.locator('.metric-value').first.wait_for(timeout=180000)
            actual = int((await page.locator('.metric-value').first.inner_text()).replace(',', ''))
            assert actual == len(catalog), 'VISIBLE_TOTAL_MISMATCH'
            report['interface_checks'].append('full_collection_total')
            bank = str(metadata['banks'].index('access'))
            negative = str(metadata['sentiments'].index('negative'))
            await page.locator('#bank-filter').select_option(bank)
            await page.locator('#year-filter').select_option('2023')
            await page.locator('#sentiment-filter').select_option(negative)
            expected = len(selection({'bank': 'access', 'year': 2023, 'sentiment': 'negative'}))
            await page.wait_for_function("n=>document.querySelector('.metric-value').textContent.replaceAll(',','')===String(n)", arg=expected)
            report['interface_checks'].append('compound_filter_counts')
            await page.locator('#clear-filters').click()
            await page.wait_for_function("n=>document.querySelector('.metric-value').textContent.replaceAll(',','')===String(n)", arg=len(catalog))
            async with page.expect_download(timeout=600000) as capture:
                await page.locator('#export-view').click()
            download = await capture.value
            exported = output / 'verification_full_export.csv'
            await download.save_as(exported)
            seen = set()
            with exported.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    assert row['review_id'] not in seen, 'DUPLICATE_EXPORTED_ID'
                    seen.add(row['review_id'])
            assert len(seen) == len(catalog), 'FULL_CSV_ROW_MISMATCH'
            report['full_export'] = {'rows': len(seen), 'bytes': exported.stat().st_size, 'sha256': hashlib.sha256(exported.read_bytes()).hexdigest()}
            report['interface_checks'].append('complete_csv_export')
            for name, width, height in [('desktop', 1440, 960), ('tablet', 820, 1180), ('mobile', 390, 844)]:
                await page.set_viewport_size({'width': width, 'height': height})
                dimensions = await page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                assert dimensions['width'] == width and dimensions['scroll'] <= width, 'HORIZONTAL_OVERFLOW'
                await page.screenshot(path=str(output / f'UI-fixed-{name}.png'), full_page=True)
                original = await context.new_page()
                await original.set_viewport_size({'width': width, 'height': height})
                await original.goto(base + 'Original/')
                await original.locator('.metric-value').first.wait_for(timeout=120000)
                await original.screenshot(path=str(output / f'Original-{name}.png'), full_page=True)
                await original.close()
                report['interface_checks'].append('viewport_' + name)
            await page.set_viewport_size({'width': 1440, 'height': 960})
            await page.locator('[data-route="explorer"]').click()
            raw = fetch('data/search.json.gz')
            texts = json.loads(gzip.decompress(raw) if raw[:2] == bytes([31, 139]) else raw)
            expected = sum('transfer' in value for value in texts)
            await page.locator('#search-input').fill('transfer')
            await page.wait_for_function("n=>document.querySelector('#filter-count').textContent.replaceAll(',','').startsWith(String(n)+' records')", arg=expected, timeout=120000)
            report['interface_checks'].append('full_text_search')
            await page.locator('#clear-filters').click()
            await page.locator('[data-route="review"]').click()
            await page.locator('.feedback-table tbody [data-action="record"]').first.click()
            await page.locator('#review-dialog').wait_for(state='visible')
            assert await page.locator('#drawer-content .record-labels').count() == 0, 'BLIND_LABELS_VISIBLE'
            assert 'stars' not in await page.locator('.record-meta').inner_text(), 'BLIND_RATING_VISIBLE'
            await page.locator('#decision-form select[name="sentiment"]').select_option('negative')
            await page.locator('#decision-form select[name="priority"]').select_option('prompt_attention')
            await page.locator('#decision-notes').fill('Verification fixture')
            await page.locator('#decision-form button[type="submit"]').click()
            await page.locator('#save-status').get_by_text('Decision saved', exact=True).wait_for()
            await page.keyboard.press('Escape')
            await page.reload()
            await page.get_by_text('1 decisions saved', exact=True).wait_for(timeout=120000)
            async with page.expect_download() as capture:
                await page.locator('[data-action="export-decisions"]').click()
            download = await capture.value
            file = output / 'verification_decisions.json'
            await download.save_as(file)
            assert len(json.loads(file.read_text())['decisions']) == 1, 'DECISION_EXPORT_MISMATCH'
            await page.locator('#review-import').set_input_files(file)
            await page.get_by_text('1 decisions imported', exact=True).wait_for()
            report['interface_checks'].extend(['blind_review', 'decision_persistence', 'decision_export_import'])
            await page.locator('#clear-filters').click()
            await page.locator('[data-route="analysis"]').click()
            assert await page.locator('#analysis-form button[type="submit"]').is_disabled(), 'MODEL_NOT_PREPARED'
            assert await page.locator('[data-action="cancel-analysis"]').is_hidden(), 'IDLE_CANCEL_VISIBLE'
            at = time.monotonic()
            await page.locator('#prepare-model').click()
            await page.wait_for_function("document.querySelector('#model-status')?.textContent==='Analysis is ready.'", timeout=1800000)
            report['model_preparation_seconds'] = time.monotonic() - at
            assert await page.locator('[data-action="cancel-analysis"]').is_hidden(), 'READY_CANCEL_VISIBLE'
            for task in config['tasks']:
                await page.locator('[data-mode="' + task['mode'] + '"]').click()
                await page.locator('#analysis-input').fill(task['question'])
                at = time.monotonic()
                await page.locator('#analysis-form button[type="submit"]').click()
                await page.locator('[data-action="export-analysis"]').wait_for(timeout=600000)
                async with page.expect_download() as capture:
                    await page.locator('[data-action="export-analysis"]').click()
                download = await capture.value
                file = output / (task['id'] + '.json')
                await download.save_as(file)
                bundle = json.loads(file.read_text())
                answer = bundle['result']
                sources_valid = exact_sources(answer)
                assert sources_valid, 'HOSTED_SOURCE_VALIDATION_FAILED'
                expected = task['expect']
                if expected == 'aggregate':
                    accepted = aggregate_valid(answer, task['filters'])
                elif expected == 'evidence':
                    accepted = bool(answer.get('summary')) and any(any(term in record['text'].lower() for term in task['terms']) for record in answer.get('evidence', []))
                    if 'expected_sentiment' in task:
                        accepted = accepted and all(record['sentiment'] == task['expected_sentiment'] for record in answer.get('evidence', []))
                elif expected == 'labels':
                    accepted = answer['sentiment'] == task['sentiment'] and set(answer['aspects']) == set(task['aspects']) and ('priority' not in task or answer['priority'] == task['priority'])
                else:
                    accepted = answer['status'] == expected
                report['tasks'].append({'id': task['id'], 'accepted': bool(accepted), 'sources_valid': sources_valid, 'wall_seconds': time.monotonic() - at, 'response': bundle})
                print('HOSTED_TASK', task['id'], bool(accepted), flush=True)
            await page.screenshot(path=str(output / 'UI-fixed-analysis.png'), full_page=True)
            assert await page.locator('[data-action="cancel-analysis"]').is_hidden(), 'COMPLETED_CANCEL_VISIBLE'
            report['interface_checks'].append('analysis_control_states')
            report['task_successes'] = sum(row['accepted'] for row in report['tasks'])
            report['task_total'] = len(report['tasks'])
            report['page_errors'] = errors
            assert not errors, 'HOSTED_PAGE_ERRORS'
            assert report['tasks'][0]['accepted'] and report['tasks'][4]['accepted'] and report['tasks'][8]['accepted'], 'PRIMARY_HOSTED_TASKS_FAILED'
            report['status'] = 'complete'
            await browser.close()
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        (output / 'hosted_checks.json').write_text(json.dumps(report, indent=2))
        print('HOSTED_CHECKS', json.dumps(report), flush=True)
        print('TOTAL_RUNTIME_SECONDS', round(time.monotonic() - started, 3), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
