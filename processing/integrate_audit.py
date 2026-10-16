import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def encoded(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')


def write_json(path,value,compressed=False):
    raw=encoded(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0) if compressed else raw)


def text(value):
    return '' if value is None or pd.isna(value) else str(value)


def labels(value):
    if not text(value):
        return {}
    parsed=json.loads(value)
    if not isinstance(parsed,dict):
        raise ValueError('Aspect labels must be an object')
    return parsed


def main(archive,audit_directory,project,verification):
    if sha256(archive)!='86c5ee2ebbbb1e172816a53197faa1a476caa9bc1376be93b2cb5dff724ce5f5':
        raise ValueError('Authoritative source archive differs')
    audit_file=audit_directory/'Second_Pass_Audit.parquet'
    evidence_file=audit_directory/'Source_Evidence.parquet'
    audit=pd.read_parquet(audit_file)
    evidence=pd.read_parquet(evidence_file)
    if len(audit)!=83753 or not audit.review_id.is_unique or len(evidence)!=92988:
        raise ValueError('Audit inventory differs')
    by_id={row.review_id:row for row in audit.itertuples(index=False)}
    quotes=defaultdict(list)
    for row in evidence.itertuples(index=False):
        quotes[row.review_id].append({'aspect':row.aspect,'sentiment':row.sentiment,'quote':row.quote,
                                     'start':int(row.start),'end':int(row.end)})
    seen=set()
    changes=Counter()
    audit_shards=defaultdict(list)
    audit_summaries=defaultdict(list)
    detail_shards=[]
    source_ids=[]
    shard_span=128
    data=project/'data'
    with zipfile.ZipFile(archive) as source:
        metadata=json.loads(source.read('web_data/metadata.json'))
        original_catalog=json.loads(gzip.decompress(source.read('web_data/catalog.json.gz')))
        catalog=[list(row)+[0] for row in original_catalog]
        sentiments=set(metadata['sentiments'])-{'unassigned'}
        priorities=set(metadata['priorities'])
        aspects=set(metadata['aspects'])
        for member in sorted(n for n in source.namelist() if n.startswith('web_data/rows/') and n.endswith('.json')):
            rows=json.loads(source.read(member))
            for record in rows:
                i=record['i'];identifier=record['id'];source_ids.append(identifier)
                if identifier not in by_id:
                    if catalog[i][:8]!=original_catalog[i]:
                        raise ValueError('Outside-audit label changed')
                    continue
                a=by_id[identifier]
                if identifier in seen or record['text']!=text(a.review_text):
                    raise ValueError('Audit source text or ID mismatch')
                seen.add(identifier)
                previous={'sentiment':metadata['sentiments'][catalog[i][3]],'priority':metadata['priorities'][catalog[i][5]],
                          'aspects':dict(record['aspects']),'referral':list(record['referral'])}
                candidate={'sentiment':text(a.candidate_sentiment) or None,'priority':text(a.candidate_priority) or None,
                           'aspects':labels(a.candidate_aspect_sentiment),
                           'referral':[item for item in text(a.candidate_referral).split(';') if item]}
                active={'sentiment':previous['sentiment'],'priority':previous['priority'],
                        'aspects':dict(previous['aspects']),'referral':list(previous['referral'])}
                if candidate['sentiment'] in sentiments:
                    active['sentiment']=candidate['sentiment']
                if candidate['priority'] in priorities:
                    active['priority']=candidate['priority']
                if candidate['aspects']:
                    if not all(key in aspects and value in sentiments for key,value in candidate['aspects'].items()):
                        raise ValueError('Candidate aspect vocabulary differs')
                    active['aspects']=candidate['aspects']
                if candidate['referral']:
                    active['referral']=candidate['referral']
                differences=[field for field in active if active[field]!=previous[field]]
                changes.update(differences)
                changes['records_changed']+=bool(differences)
                findings=[item for item in text(a.audit_findings).split(';') if item]
                excerpts=quotes.get(identifier,[])
                for quote in excerpts:
                    if record['text'][quote['start']:quote['end']]!=quote['quote']:
                        raise ValueError('Source excerpt position differs: '+identifier)
                payload={'i':i,'id':identifier,'findings':findings,'recommendation':a.audit_recommendation,
                         'candidate':candidate,'earlier':previous,'evidence':excerpts,
                         'status':'awaiting_review','order':int(a.review_order)}
                key=str((int(a.review_order)-1)//shard_span).zfill(4)
                audit_shards[key].append(payload)
                record['aspects']=active['aspects'];record['referral']=active['referral']
                record['audit_recommendation']=a.audit_recommendation
                record['audit_findings_count']=len(findings)
                audit_summaries[key].append(dict(record))
                catalog[i][3]=metadata['sentiments'].index(active['sentiment'])
                catalog[i][4]=sum(1<<metadata['aspects'].index(key) for key in active['aspects'])
                catalog[i][5]=metadata['priorities'].index(active['priority'])
                catalog[i][8]=int(a.review_order)
            destination=data/member.removeprefix('web_data/')
            write_json(destination,rows)
            detail_shards.append(destination)
        if len(source_ids)!=334415 or len(set(source_ids))!=334415 or seen!=set(by_id):
            raise ValueError('Full application/audit coverage mismatch')
        if sum(row[6] for row in catalog)!=83753:
            raise ValueError('Review queue coverage differs')
        if sha256(data/'search.json.gz')!=hashlib.sha256(source.read('web_data/search.json.gz')).hexdigest():
            raise ValueError('Full text search source differs')
    previous_manifest=data/'audit/manifest.json'
    if previous_manifest.exists():
        previous=json.loads(previous_manifest.read_text(encoding='utf-8'))
        expected={prefix+key+'.json.gz' for key in audit_shards for prefix in ['records-','summary-']}
        for item in previous.get('shards',[])+previous.get('summary_shards',[]):
            old=data/item['path']
            if old.parent.resolve()!=(data/'audit').resolve():
                raise ValueError('Generated audit path leaves its directory')
            if old.name not in expected:
                old.unlink(missing_ok=True)
    manifest=[]
    summary_manifest=[]
    for key,rows in sorted(audit_shards.items()):
        rows.sort(key=lambda row:row['i'])
        destination=data/'audit'/('records-'+key+'.json.gz')
        write_json(destination,rows,True)
        manifest.append({'path':str(destination.relative_to(data)).replace('\\','/'),'rows':len(rows),
                         'evidence':sum(len(row['evidence']) for row in rows),'bytes':destination.stat().st_size,
                         'sha256':sha256(destination)})
        summaries=sorted(audit_summaries[key],key=lambda row:next(a['order'] for a in rows if a['i']==row['i']))
        summary_path=data/'audit'/('summary-'+key+'.json.gz')
        write_json(summary_path,summaries,True)
        summary_manifest.append({'path':str(summary_path.relative_to(data)).replace('\\','/'),'rows':len(summaries),
                                 'bytes':summary_path.stat().st_size,'sha256':sha256(summary_path)})
    audit_manifest={'version':'20261016-audit1','records':83753,'evidence':92988,'shard_size':shard_span,
                    'recommendations':audit.audit_recommendation.value_counts().to_dict(),'shards':manifest,'summary_shards':summary_manifest}
    write_json(data/'audit/manifest.json',audit_manifest)
    write_json(data/'catalog.json.gz',catalog,True)
    metadata['audit']={'records':83753,'evidence':92988,'shard_size':shard_span,'version':'20261016-audit1'}
    metadata['catalog_fields']=metadata['catalog_fields'][:8]+['audit_order']
    write_json(data/'metadata.json',metadata)
    cube=Counter(tuple(row[index] for index in [0,1,3,5,4,6]) for row in catalog)
    write_json(data/'cube.json',[list(key)+[count] for key,count in sorted(cube.items())])
    files=[]
    for path in sorted(data.rglob('*')):
        if path.is_file() and path.name!='manifest.json':
            files.append({'path':str(path.relative_to(data)).replace('\\','/'),'bytes':path.stat().st_size,'sha256':sha256(path)})
    write_json(data/'manifest.json',files)
    result={'status':'complete','source_records':334415,'audit_records':len(seen),'source_excerpts':len(evidence),
            'audit_shards':len(manifest),'audit_bytes':sum(item['bytes'] for item in manifest),
            'maximum_audit_shard_bytes':max(item['bytes'] for item in manifest),'label_changes':dict(changes),
            'outside_audit_records_changed':0,'source_text_and_ids_preserved':True,
            'source_archive_sha256':sha256(archive),'audit_table_sha256':sha256(audit_file),
            'evidence_table_sha256':sha256(evidence_file),
            'sentiment_counts':dict(Counter(metadata['sentiments'][row[3]] for row in catalog)),
            'priority_counts':dict(Counter(metadata['priorities'][row[5]] for row in catalog))}
    verification.mkdir(parents=True,exist_ok=True)
    write_json(verification/'data_build_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--audit-directory',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True)
    args=parser.parse_args()
    main(args.archive,args.audit_directory,args.project,args.verification)
