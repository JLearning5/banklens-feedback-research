import {matches,summarize,exportCSV} from './core.js';
let metadata,catalog,texts,stopwords;
const base=new URL('../data/',import.meta.url);
const cache=new Map();
async function json(name){const r=await fetch(new URL(name,base));if(!r.ok)throw Error('DATA_FETCH_'+r.status);if(name.endsWith('.gz')){const bytes=new Uint8Array(await r.arrayBuffer());if(bytes[0]===31&&bytes[1]===139){const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));return JSON.parse(await new Response(stream).text());}return JSON.parse(new TextDecoder().decode(bytes));}return r.json();}
async function ensureText(){if(!texts){self.postMessage({event:'status',text:'Preparing full-text search…'});texts=await json('search.json.gz');if(texts.length!==catalog.length)throw Error('TEXT_COUNT_MISMATCH');}return texts;}
async function detail(i){const key=String(Math.floor(i/metadata.shard_size)).padStart(3,'0');if(!cache.has(key)){cache.set(key,json('rows/'+key+'.json').catch(e=>{cache.delete(key);throw e;}));if(cache.size>12)cache.delete(cache.keys().next().value);}const row=(await cache.get(key)).find(r=>r.i===i);if(!row)throw Error('RECORD_NOT_FOUND');const c=catalog[i];return {...row,bank:metadata.banks[c[0]],year:c[1],rating:c[2],sentiment:metadata.sentiments[c[3]],priority:metadata.priorities[c[5]],review_needed:Boolean(c[6]),duplicate_count:c[7]};}
function filter(f,query=''){const list=[];for(let i=0;i<catalog.length;i++){if(matches(catalog[i],f)&&(!query||texts[i].includes(query)))list.push(i);}return list;}
async function retrieve(question,f){
 await ensureText();stopwords??=new Set(await (await fetch(new URL('./search_stopwords.json',import.meta.url))).json());
 const terms=[...new Set((question.toLowerCase().match(/[a-z]{3,}/g)||[]).filter(t=>!stopwords.has(t)&&!['review','reviews','example','examples','complaint','complaints','negative','positive','neutral','mixed','bank','banks','show','tell','find'].includes(t)).map(t=>t.slice(0,6)))].slice(0,12);
 const candidates=[],df=Object.fromEntries(terms.map(t=>[t,0]));
 for(const i of filter(f)){const text=texts[i];const hit=terms.filter(t=>text.includes(t));if(hit.length){hit.forEach(t=>df[t]++);candidates.push({i,hit,length:text.length});}}
 candidates.forEach(c=>c.score=c.hit.reduce((n,t)=>n+Math.log(1+catalog.length/(1+df[t])),0)/(1+c.length/1500));candidates.sort((a,b)=>b.score-a.score||a.i-b.i);
 const selected=[],seen=new Set();for(const c of candidates){if(seen.has(texts[c.i]))continue;seen.add(texts[c.i]);selected.push(c.i);if(selected.length===6)break;}
 return {records:await Promise.all(selected.map(detail)),matched_records:candidates.length,query_terms:terms};
}
self.onmessage=async({data})=>{const {id,type,payload}=data;try{
 if(type==='init'){metadata=await json('metadata.json');catalog=await json('catalog.json.gz');if(catalog.length!==metadata.rows)throw Error('CATALOG_COUNT_MISMATCH');self.postMessage({id,result:metadata});return;}
 if(!catalog)throw Error('DATA_NOT_READY');
 let result;
 if(type==='query'){const f=payload.filters;const q=(f.query||'').toLowerCase().trim();if(q)await ensureText();const list=filter(f,q);const start=Math.max(0,payload.page||0)*25;result={summary:summarize(catalog,list,metadata),rows:await Promise.all(list.slice(start,start+25).map(detail)),page:payload.page||0,pages:Math.ceil(list.length/25)};}
 else if(type==='detail')result=await detail(payload.index);
 else if(type==='aggregate'){const list=filter(payload.filters);result=summarize(catalog,list,metadata);}
 else if(type==='retrieve')result=await retrieve(payload.question,payload.filters);
 else if(type==='export'){const f=payload.filters;const q=(f.query||'').toLowerCase().trim();if(q)await ensureText();const list=filter(f,q),chunks=[],encoder=new TextEncoder();for(let offset=0;offset<list.length;offset+=2000){const rows=await Promise.all(list.slice(offset,offset+2000).map(detail));let csv=exportCSV(rows.map(r=>({review_id:r.id,bank:r.bank,year:r.year,rating:r.rating,review_text:r.text,sentiment:r.sentiment,aspect_sentiment:r.aspects,priority:r.priority,referral:r.referral,review_needed:r.review_needed,review_reasons:r.reasons,duplicate_group:r.group,duplicate_count:r.duplicate_count})));if(offset)csv='\r\n'+csv.slice(csv.indexOf('\r\n')+2);chunks.push(encoder.encode(csv));self.postMessage({event:'status',text:'Preparing export '+Math.min(list.length,offset+2000).toLocaleString()+' / '+list.length.toLocaleString()});}result={blob:new Blob(chunks,{type:'text/csv;charset=utf-8'}),rows:list.length};}
 else throw Error('UNKNOWN_DATA_ACTION');
 self.postMessage({id,result});
}catch(e){self.postMessage({id,error:String(e.message||e)});}};
