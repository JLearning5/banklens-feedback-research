export const bankNames={access:'Access Bank',ecoBank:'Ecobank',fbn:'First Bank',fcmb:'FCMB',fidelityBank:'Fidelity Bank',gtb:'GTBank',jaizBank:'Jaiz Bank',keyStoneBank:'Keystone Bank',polarisBank:'Polaris Bank',stanbicIbtc:'Stanbic IBTC',sterlingBank:'Sterling Bank',uba:'UBA',unionBank:'Union Bank',unityBank:'Unity Bank',wemaBank:'Wema Bank',zenithBank:'Zenith Bank'};
export const aspectNames={transfers_payments:'Transfers & payments',login_authentication:'Login & authentication',app_reliability_usability:'App experience',fees_charges:'Fees & charges',account_access_kyc:'Account access & KYC',cards_atm:'Cards & ATM',customer_support:'Customer support',other_unspecified:'Other feedback'};
export const title=s=>String(s??'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
export const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const defaultFilters=()=>({bank:'all',year:'all',sentiment:'all',aspect:'all',priority:'all',queue:false,query:''});
export function routePolicy(question){
 const q=String(question).trim().toLowerCase();
 if(/^(?:please\s+)?(?:send|transfer|withdraw|deposit|pay)\b/.test(q)&&/\b(?:money|funds?|cash|payments?|account)\b/.test(q)&&! /\b(?:reviews?|feedback|examples?|customers?)\b/.test(q))return 'refer';
 if(/^(?:please\s+)?(?:change|reset|block|unblock|open|close)\b/.test(q)&&/\b(?:my|our)\b/.test(q)&&/\b(?:password|pin|account|card)\b/.test(q))return 'refer';
 if(/\bhow many\b|\b(?:count|total|number|percentage|percent|compare|comparison)\b/.test(q))return 'aggregate';
 if(/\b(?:show|find|examples?|retrieve)\b|\bwhat (?:do|are|have) (?:the )?customers\b|\b(?:feedback|reviews?)\s+(?:about|on|containing)\b/.test(q))return 'retrieve';
 return null;
}
export function matches(row,f){
  return (f.bank==='all'||row[0]===Number(f.bank))&&(f.year==='all'||row[1]===Number(f.year))&&(f.sentiment==='all'||row[3]===Number(f.sentiment))&&(f.aspect==='all'||Boolean(row[4]&(1<<Number(f.aspect))))&&(f.priority==='all'||row[5]===Number(f.priority))&&(!f.queue||row[6]===1);
}
export function summarize(catalog,indices,metadata){
  const sentiment=Array(5).fill(0),aspects=Array(metadata.aspects.length).fill(0),priority=Array(4).fill(0),banks=metadata.banks.map((b,i)=>({bank:b,index:i,total:0,sentiment:Array(5).fill(0),queue:0})),years={};
  let queue=0,ratingTotal=0,rated=0;
  for(const i of indices){const r=catalog[i];sentiment[r[3]]++;priority[r[5]]++;banks[r[0]].total++;banks[r[0]].sentiment[r[3]]++;banks[r[0]].queue+=r[6];queue+=r[6];if(r[2]>0){ratingTotal+=r[2];rated++;}for(let a=0;a<aspects.length;a++)if(r[4]&(1<<a))aspects[a]++;years[r[1]]??=Array(5).fill(0);years[r[1]][r[3]]++;}
  return {total:indices.length,labelled:indices.length-sentiment[4],sentiment,aspects,priority,queue,banks,years,averageRating:rated?ratingTotal/rated:null};
}
export function parseObject(raw){
  const text=String(raw).replace(/<think>[\s\S]*?<\/think>/g,'').trim();
  const start=text.indexOf('{'),end=text.lastIndexOf('}');
  if(start<0||end<start)throw Error('INVALID_JSON');
  const value=JSON.parse(text.slice(start,end+1));
  if(!value||Array.isArray(value)||typeof value!=='object')throw Error('INVALID_OBJECT');
  return value;
}
export function validateAnswer(value,evidence){
  if(typeof value.summary!=='string'||!value.summary.trim()||value.summary.length>1400)throw Error('INVALID_SUMMARY');
  if(!Array.isArray(value.citations)||!value.citations.length||value.citations.some(i=>!Number.isInteger(i)||i<0||i>=evidence.length))throw Error('INVALID_CITATION');
  if(value.extractive===true){if(value.citations.length!==1||!evidence[value.citations[0]].text.includes(value.summary.trim()))throw Error('UNSUPPORTED_EXCERPT');}
  else if(/\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|million|percent|percentage)\b|there (?:are|is) no/i.test(value.summary))throw Error('NUMERIC_CLAIM');
  return {summary:value.summary.trim(),citations:[...new Set(value.citations)].map(i=>evidence[i].id),...(value.extractive===true?{extractive:true}:{})};
}
export function validateDecision(value){
  if(!value||typeof value!=='object'||typeof value.review_id!=='string'||!value.review_id||value.review_id.length>100)throw Error('INVALID_REVIEW_ID');
  if(!['positive','negative','neutral','mixed','unassigned'].includes(value.sentiment))throw Error('INVALID_SENTIMENT');
  if(!['routine','prompt_attention','urgent_review','insufficient_information'].includes(value.priority))throw Error('INVALID_PRIORITY');
  if(!Array.isArray(value.aspects)||value.aspects.some(a=>!Object.hasOwn(aspectNames,a)))throw Error('INVALID_ASPECT');
  if(typeof value.notes!=='string'||value.notes.length>5000)throw Error('INVALID_NOTES');
  return {review_id:value.review_id,sentiment:value.sentiment,priority:value.priority,aspects:[...new Set(value.aspects)],notes:value.notes,updated_at:typeof value.updated_at==='string'?value.updated_at:new Date().toISOString()};
}
export function exportCSV(rows){
  if(!rows.length)return '';
  const columns=Object.keys(rows[0]);
  const cell=v=>{let s=Array.isArray(v)?v.join(';'):typeof v==='object'&&v!==null?JSON.stringify(v):String(v??'');if(/^[=+\-@\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  return '\uFEFF'+[columns.map(cell).join(','),...rows.map(r=>columns.map(c=>cell(r[c])).join(','))].join('\r\n');
}
export function redact(text){return String(text).normalize('NFKC').replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,'[EMAIL]').replace(/(?<!\w)(?:\+?234[ -]?)?\d[\d -]{8,17}\d(?!\w)/g,'[NUMBER]').replace(/\s+/g,' ').trim();}
export function effectiveScope(question,filters,metadata){
  const f={...filters,query:''};const q=question.toLowerCase();
  if(f.bank==='all'){const found=metadata.banks.map((bank,i)=>({i,names:[bank,bankNames[bank]]})).filter(b=>b.names.some(n=>n&&new RegExp('\\b'+n.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\b','i').test(question)));if(found.length===1)f.bank=String(found[0].i);}
  if(f.sentiment==='all'){for(const s of ['negative','positive','neutral','mixed'])if(new RegExp('\\b'+s+'\\b').test(q)){f.sentiment=String(metadata.sentiments.indexOf(s));break;}}
  const year=q.match(/\b20\d{2}\b/);if(f.year==='all'&&year&&metadata.years.includes(Number(year[0])))f.year=year[0];
  return f;
}
