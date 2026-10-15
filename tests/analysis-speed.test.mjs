import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {runAgent,analyseFeedback,quickPriority} from '../shared/agent.js';
import {defaultFilters,routePolicy} from '../shared/core.js';

const metadata={banks:['access'],sentiments:['positive','negative','neutral','mixed','unassigned'],aspects:['transfers_payments','login_authentication','app_reliability_usability','fees_charges','account_access_kyc','cards_atm','customer_support','other_unspecified'],years:[2023]};
const prompts=JSON.parse(fs.readFileSync(new URL('../shared/prompts.json',import.meta.url)));
const forbid=async()=>{throw Error('Unexpected model inference');};

test('cold collection count returns without downloading or calling a model',async()=>{
 const result=await runAgent('How many negative reviews are there?',defaultFilters(),{quick:true,metadata,prompts,choose:forbid,generate:forbid,tools:{aggregate:async scope=>{assert.equal(scope.sentiment,'1');return {total:7};}}});
 assert.equal(result.aggregate.total,7);assert.equal(result.planner,'catalog_tools');assert.equal(result.status,'complete');
});
test('cold retrieval returns a source-checked quote without language-model selection',async()=>{
 const text='My transfer failed and no response was received.';
 const result=await runAgent('Show examples of failed transfers.',defaultFilters(),{quick:true,metadata,prompts,choose:forbid,generate:forbid,tools:{retrieve:async(q,scope)=>{assert.equal(scope.sentiment,'1');return {records:[{id:'access:000001',text,i:0,bank:'access',year:2023},{id:'access:000002',text:'A second complaint',i:1,bank:'access',year:2023}]};}}});
 assert.equal(result.summary,text);assert.deepEqual(result.citations,['access:000001']);assert.equal(result.extractive,true);
});
test('the screenshot account-opening request is handled immediately outside collection analysis',async()=>{
 const question='I need to get a firstAsset Management account';assert.equal(routePolicy(question),'refer');
 const result=await runAgent(question,defaultFilters(),{quick:true,metadata,prompts,choose:forbid,generate:forbid,tools:{aggregate:forbid,retrieve:forbid}});
 assert.equal(result.status,'referred');assert.equal(routePolicy('Show customer reviews about opening an account.'),'retrieve');
});
test('standard feedback analysis batches unique fragments and preserves opposing aspects',async()=>{
 let calls=0;const result=await analyseFeedback('The app is easy to use, but the fees are too high.',{prompts,sentimentModel:forbid,sentimentBatch:async texts=>{calls++;assert.equal(texts.length,new Set(texts).size);return texts.map(text=>text.includes('fees')?'negative':'positive');}});
 assert.equal(calls,1);assert.equal(result.sentiment,'mixed');assert.deepEqual(result.aspects,{app_reliability_usability:'positive',fees_charges:'negative'});assert.equal(result.analysis_method,'sentiment_with_triage_rules');
});
test('standard priority separates an explicit incident from insults and negation',()=>{
 assert.equal(quickPriority('My card was stolen.'),'urgent_review');assert.equal(quickPriority('No unauthorized transactions have occurred.'),'routine');assert.equal(quickPriority('Scam bank, terrible service.'),'routine');assert.equal(quickPriority('My transfer failed.'),'prompt_attention');assert.equal(quickPriority('I was debited but my money has not been reversed.'),'urgent_review');assert.equal(quickPriority('12345'),'insufficient_information');
});
