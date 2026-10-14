import {parseObject,validateAnswer,redact,effectiveScope,routePolicy} from './core.js';
export async function runAgent(question,filters,{generate,choose,tools,metadata,prompts,onEvent=()=>{},signal}){
 const started=performance.now(),trace=[],scope=effectiveScope(question,filters,metadata),policy=routePolicy(question);let aggregate=null,evidence=[];
 const mentioned=Object.entries(prompts.aspectPatterns||{}).filter(([key,pattern])=>key!=='other'&&new RegExp(pattern,'i').test(question)).map(([key])=>aliases[key]);
 if(scope.aspect==='all'&&mentioned.length===1&&metadata.aspects.includes(mentioned[0]))scope.aspect=String(metadata.aspects.indexOf(mentioned[0]));
 if(policy==='retrieve'&&scope.sentiment==='all'&&/\b(?:fail(?:ed|ure|ing)?|problems?|complaints?|broken|crash(?:es|ing)?)\b/i.test(question))scope.sentiment=String(metadata.sentiments.indexOf('negative'));
 const event=(tool,detail)=>{const item={tool,detail,elapsed_ms:Math.round(performance.now()-started)};trace.push(item);onEvent(item);};
 const check=()=>{if(signal?.aborted)throw new DOMException('Cancelled','AbortError');};
 const pick=raw=>{const found=String(raw).toLowerCase().match(/\b(aggregate|retrieve|refer|finalize)\b/g)||[];const unique=[...new Set(found)];return unique.length===1?unique[0]:'refer';};
 check();event('plan',policy?'Checking the requested analysis':'Choosing an analysis tool');
 const select=async(messages,choices)=>{const selected=choose?await choose(messages,choices):pick(await generate(messages,8));return choices.includes(selected)?selected:'refer';};
 let action=policy||await select([{role:'system',content:prompts.route},{role:'user',content:redact(question).slice(0,600)}],['aggregate','retrieve','refer']);
 for(let step=0;step<3;step++){
  check();
  if(action==='refer'){event('refer','Request referred for review');return {status:'referred',summary:null,evidence,aggregate,scope,planner:policy?'request_policy':'language_model',trace,runtime_seconds:(performance.now()-started)/1000};}
  if(action==='aggregate'){aggregate=await tools.aggregate(scope);event('aggregate',aggregate.total.toLocaleString()+' records in scope');check();action=policy==='aggregate'?'finalize':await select([{role:'system',content:prompts.next},{role:'user',content:JSON.stringify({question,available:'counts',records:aggregate.total})}],['retrieve','finalize']);continue;}
  if(action==='retrieve'){const found=await tools.retrieve(question,scope);evidence=found.records.slice(0,4);event('retrieve',evidence.length+' distinct reviews linked');action='finalize';}
  if(action==='finalize')break;
 }
 check();
 if(!aggregate&&!evidence.length){event('refer','No supporting records were found');return {status:'no_evidence',summary:null,evidence,aggregate,scope,planner:policy?'request_policy':'language_model',trace,runtime_seconds:(performance.now()-started)/1000};}
 let summary=null,citations=[],extractive=false;
 if(evidence.length){event('summarize','Reading linked feedback');const excerpts=evidence.map((r,i)=>({n:i,text:r.text.slice(0,360)}));let value;if(choose){const options=excerpts.map(r=>r.text);const selected=options.length===1?options[0]:await choose([{role:'system',content:prompts.extract},{role:'user',content:question.slice(0,400)}],options);value={summary:selected,citations:[options.indexOf(selected)],extractive:true};}else{const raw=await generate([{role:'system',content:prompts.summary},{role:'user',content:JSON.stringify({question:question.slice(0,400),excerpts})}],96);value=parseObject(raw);}check();try{const valid=validateAnswer(value,evidence);summary=valid.summary;citations=valid.citations;extractive=Boolean(valid.extractive);event('validate_sources',citations.length+' source links checked');}catch(e){event('validate_sources','Open the linked reviews to continue');}}
 else event('validate_sources','Counts returned directly from the catalog');
 return {status:summary||aggregate?'complete':'evidence',summary,citations,extractive,evidence,aggregate,scope,planner:policy?'request_policy':'language_model',trace,runtime_seconds:(performance.now()-started)/1000};
}
const aliases={payments:'transfers_payments',login:'login_authentication',app:'app_reliability_usability',fees:'fees_charges',account:'account_access_kyc',cards:'cards_atm',support:'customer_support',other:'other_unspecified'};
export function validateAnalysis(value){
 const sentiments=['positive','negative','neutral','mixed'];if(!sentiments.includes(value.sentiment))throw Error('INVALID_SENTIMENT');
 if(!value.aspects||Array.isArray(value.aspects)||typeof value.aspects!=='object'||!Object.keys(value.aspects).length||Object.keys(value.aspects).length>8)throw Error('INVALID_ASPECTS');
 if(!Number.isInteger(value.priority)||value.priority<0||value.priority>3)throw Error('INVALID_PRIORITY');
 const aspects={};for(const [key,polarity] of Object.entries(value.aspects)){if(!Object.hasOwn(aliases,key)||!sentiments.includes(polarity))throw Error('INVALID_ASPECT');aspects[aliases[key]]=polarity;}
 if(Object.keys(aspects).length>1)delete aspects.other_unspecified;
 return {sentiment:value.sentiment,aspects,priority:['routine','prompt_attention','urgent_review','insufficient_information'][value.priority]};
}
export async function analyseFeedback(text,options){
 const {choose,sentimentModel,prompts}=options;
 if(!sentimentModel)return legacyAnalysis(text,options);
 const input=redact(text);if(!input)throw Error('EMPTY_FEEDBACK');if(input.length>1800)throw Error('FEEDBACK_TOO_LONG');
 const started=performance.now(),cache=new Map();
 const classify=async value=>{if(!cache.has(value))cache.set(value,await sentimentModel(value));return cache.get(value);};
 const fragments=input.split(/(?<=[.!?;])\s+|\s+(?:but|however|although)\s+/i).map(s=>s.trim()).filter(Boolean).slice(0,8);
 let sentiment=await classify(input);const fragmentSentiments=[];
 for(const fragment of fragments)fragmentSentiments.push(await classify(fragment));
 if(fragmentSentiments.includes('positive')&&fragmentSentiments.includes('negative'))sentiment='mixed';
 let selected=Object.keys(aliases).filter(key=>key!=='other'&&typeof prompts.aspectPatterns[key]==='string'&&new RegExp(prompts.aspectPatterns[key],'i').test(input));
 if(selected.length>3){const ranked=[];while(ranked.length<3){const remaining=selected.filter(key=>!ranked.includes(key));const key=await choose([{role:'system',content:prompts.aspectChoice},{role:'user',content:JSON.stringify({review:input,selected:ranked})}],remaining);if(!remaining.includes(key))throw Error('INVALID_ASPECT');ranked.push(key);}selected=ranked;}
 if(!selected.length)selected=['other'];
 const aspects={};
 for(const aspect of selected){const pattern=new RegExp(prompts.aspectPatterns[aspect],'i'),polarities=fragments.map((part,i)=>pattern.test(part)?fragmentSentiments[i]:null).filter(Boolean);aspects[aspect]=polarities.includes('positive')&&polarities.includes('negative')?'mixed':polarities.includes('negative')?'negative':polarities.includes('positive')?'positive':polarities.length?'neutral':sentiment;}
 const priority=await choose([{role:'system',content:prompts.priorityChoice},{role:'user',content:JSON.stringify({review:input})}],['routine','prompt_attention','urgent_review','insufficient_information']);
 const value=validateAnalysis({sentiment,aspects,priority:['routine','prompt_attention','urgent_review','insufficient_information'].indexOf(priority)});
 return {...value,text:input,analysis_method:'hybrid',runtime_seconds:(performance.now()-started)/1000};
}
async function legacyAnalysis(text,{generate,choose,sentimentModel,prompts}){const input=redact(text);if(!input)throw Error('EMPTY_FEEDBACK');if(input.length>1800)throw Error('FEEDBACK_TOO_LONG');const started=performance.now();let value;if(choose){const pick=(prompt,choices,extra={})=>choose([{role:'system',content:prompt},{role:'user',content:JSON.stringify({review:input,...extra})}],choices);const sentiments=['positive','negative','neutral','mixed'];let sentiment=sentimentModel?await sentimentModel(input):await pick(prompts.sentimentChoice,sentiments);const aspects={};let remaining=Object.keys(aliases);for(let i=0;i<3;i++){const choices=i===0?remaining:['none',...remaining.filter(k=>k!=='other')];const aspect=await pick(prompts.aspectChoice,choices,{selected:Object.keys(aspects)});if(aspect==='none')break;if(sentimentModel){const fragments=input.split(/(?<=[.!?;])\s+|\s+(?:but|however|although)\s+/i),pattern=new RegExp(prompts.aspectPatterns[aspect],'i'),matched=fragments.filter(part=>pattern.test(part)).slice(0,6),polarities=[];for(const fragment of matched.length?matched:[input])polarities.push(await sentimentModel(fragment));aspects[aspect]=polarities.includes('positive')&&polarities.includes('negative')?'mixed':polarities.includes('negative')?'negative':polarities.includes('positive')?'positive':'neutral';}else aspects[aspect]=await pick(prompts.polarityChoice,sentiments,{aspect});remaining=remaining.filter(k=>k!==aspect);if(aspect==='other')break;}const polarities=Object.values(aspects);if(polarities.includes('mixed')||(polarities.includes('positive')&&polarities.includes('negative')))sentiment='mixed';const priority=await pick(prompts.priorityChoice,['routine','prompt_attention','urgent_review','insufficient_information']);value={sentiment,aspects,priority:['routine','prompt_attention','urgent_review','insufficient_information'].indexOf(priority)};}else{const raw=await generate([{role:'system',content:prompts.classify},{role:'user',content:JSON.stringify({review:input})}],160);value=parseObject(raw);}return {...validateAnalysis(value),text:input,runtime_seconds:(performance.now()-started)/1000};}
