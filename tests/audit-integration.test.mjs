import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {defaultFilters} from '../shared/core.js';
import {renderAudit} from '../shared/audit-view.js';

const requests=[];
const pending=new Map();
globalThis.self={postMessage(message){if(message.id){const task=pending.get(message.id);pending.delete(message.id);message.error?task.reject(Error(message.error)):task.resolve(message.result);}}};
globalThis.fetch=async url=>{
 const path=fileURLToPath(url);requests.push(path.replaceAll('\\','/'));
 try{return new Response(await fs.readFile(path));}catch{return new Response('',{status:404});}
};
await import('../shared/data-worker.js');
let serial=0;
function call(type,payload={}){const id=++serial;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});self.onmessage({data:{id,type,payload}});});}
const metadata=await call('init');

test('complete audit queue is paginated and details remain on demand',async()=>{
 assert.equal(metadata.rows,334415);assert.equal(metadata.audit.records,83753);
 const queue={...defaultFilters(),queue:true};
 const first=await call('query',{filters:queue,page:0});
 assert.equal(first.summary.total,83753);assert.equal(first.pages,3351);assert.equal(first.rows.length,25);
 assert.equal(first.rows[0].id,'fbn:026507');assert.equal(first.rows[0].sentiment,'neutral');assert.equal(first.rows[0].priority,'routine');
 assert.equal(requests.filter(url=>/audit\/summary-/.test(url)).length,1);
 assert.equal(requests.filter(url=>/audit\/records-/.test(url)).length,0);
 assert.equal(requests.filter(url=>/search.json/.test(url)).length,0);
 const detail=await call('detail',{index:first.rows[0].i});
 assert.equal(detail.audit.earlier.sentiment,'negative');assert.equal(detail.audit.candidate.sentiment,'neutral');
 assert.equal(detail.audit.status,'awaiting_review');assert.equal(detail.audit.recommendation,'resolve_ambiguity');
 for(const quote of detail.audit.evidence)assert.equal(detail.text.slice(quote.start,quote.end),quote.quote);
 assert.equal(requests.filter(url=>/audit\/records-/.test(url)).length,1);
 const last=await call('query',{filters:queue,page:3350});
 assert.equal(last.rows.length,3);assert.equal(last.summary.total,83753);
 const totals=await call('aggregate',{filters:defaultFilters()});
 assert.equal(totals.total,334415);assert.deepEqual(totals.sentiment,[183959,110919,28634,10807,96]);
 const negative=await call('query',{filters:{...queue,sentiment:'1'},page:0});
 assert.ok(negative.rows.every(row=>row.sentiment==='negative'));
});

test('audit comparison escapes source content and blind mode hides findings',()=>{
 const audit={findings:['sentiment_comparison'],recommendation:'compare_proposal',earlier:{sentiment:'negative',priority:'routine',aspects:{},referral:[]},candidate:{sentiment:'neutral',priority:null,aspects:{},referral:[]},evidence:[{aspect:'other_unspecified',sentiment:'neutral',quote:'<script>alert(1)</script>'}]};
 const rendered=renderAudit(audit);
 assert.ok(rendered.includes('Earlier labels'));assert.ok(rendered.includes('Candidate labels'));
 assert.ok(rendered.includes('&lt;script&gt;'));assert.ok(!rendered.includes('<script>'));
 const blind=renderAudit(audit,true);assert.ok(blind.includes('Awaiting review'));
 assert.ok(!blind.includes('Source evidence'));assert.ok(!blind.includes('Candidate labels'));assert.ok(!blind.includes('Sentiment comparison'));
 assert.equal(renderAudit(null),'');
});
