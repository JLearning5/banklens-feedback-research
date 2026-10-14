import {AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification,LogitsProcessor,env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
env.allowLocalModels=false;
env.backends.onnx.wasm.numThreads=1;
let tokenizer,model,busy=false,config,sentimentTokenizer,sentimentModel;
class ChoiceProcessor extends LogitsProcessor{constructor(ids){super();this.ids=ids;} _call(inputIds,scores){const width=scores.dims.at(-1);for(let row=0;row<inputIds.length;row++){const offset=row*width,values=this.ids.map(id=>scores.data[offset+id]);scores.data.fill(-Infinity,offset,offset+width);this.ids.forEach((id,i)=>scores.data[offset+id]=values[i]);}return scores;}}
self.onmessage=async({data})=>{const {id,type,payload}=data;if(busy){self.postMessage({id,error:'MODEL_BUSY'});return;}busy=true;try{
 if(type==='load'){
  if(model)await model.dispose();model=null;tokenizer=null;config=payload;
  const progress_callback=p=>self.postMessage({event:'progress',file:p.file,status:p.status,progress:p.progress,loaded:p.loaded,total:p.total});
  let device='wasm',dtype=config.cpu_dtype;
  if(navigator.gpu){try{const adapter=await navigator.gpu.requestAdapter();if(adapter?.features.has('shader-f16')){device='webgpu';dtype=config.gpu_dtype;}}catch{}}
  tokenizer=await AutoTokenizer.from_pretrained(config.id,{revision:config.revision,progress_callback});
  try{model=await AutoModelForCausalLM.from_pretrained(config.id,{revision:config.revision,dtype,device,progress_callback});}catch(error){if(device!=='webgpu')throw error;device='wasm';dtype=config.cpu_dtype;self.postMessage({event:'progress',status:'CPU preparation'});model=await AutoModelForCausalLM.from_pretrained(config.id,{revision:config.revision,dtype,device,progress_callback});}
  const warm=tokenizer('Ready');await model.generate({...warm,max_new_tokens:1,do_sample:false});
  self.postMessage({id,result:{model:config.id,revision:config.revision,device,dtype}});
 }else if(type==='choose'){
  if(!model||!tokenizer)throw Error('MODEL_NOT_READY');
  if(!Array.isArray(payload.choices)||payload.choices.length<2||payload.choices.length>16)throw Error('INVALID_CHOICES');
  const codes=payload.choices.map((value,i)=>({code:String.fromCharCode(65+i),value}));
  const wordTokens=payload.choices.map(value=>/^[a-z_]+$/.test(value)?tokenizer.encode(value,{add_special_tokens:false}):[]);
  const direct=!payload.choices.some(value=>['aggregate','retrieve','refer','finalize'].includes(value))&&wordTokens.every(t=>t.length>0)&&new Set(wordTokens.map(t=>t[0])).size===wordTokens.length;
  const last=payload.messages.at(-1),messages=[...payload.messages.slice(0,-1),{role:'user',content:JSON.stringify({input:last.content,options:direct?payload.choices:codes})}];
  messages[0]={...messages[0],content:messages[0].content+'\n'+(direct?payload.wordInstruction:payload.instruction)};
  const inputs=tokenizer.apply_chat_template(messages,{add_generation_prompt:true,return_dict:true,enable_thinking:false});
  if(inputs.input_ids.dims.at(-1)>1600)throw Error('CONTEXT_TOO_LONG');
  const tokens=direct?wordTokens.map(t=>[t[0]]):codes.map(c=>tokenizer.encode(c.code,{add_special_tokens:false}));
  if(tokens.some(t=>t.length!==1))throw Error('CHOICE_TOKEN_LENGTH');
  const ids=tokens.map(t=>t[0]);
  const output=await model.generate({...inputs,max_new_tokens:1,do_sample:false,logits_processor:[new ChoiceProcessor(ids)]});
  const picked=Number(output.tolist()[0].at(-1)),index=ids.indexOf(picked);
  if(index<0)throw Error('INVALID_CHOICE_TOKEN');
  self.postMessage({id,result:codes[index].value});
 }else if(type==='sentiment'){
  if(!config?.sentiment)throw Error('SENTIMENT_MODEL_NOT_CONFIGURED');
  const cfg=config.sentiment;
  if(!sentimentModel){const progress_callback=p=>self.postMessage({event:'progress',file:p.file,status:p.status,progress:p.progress});sentimentTokenizer=await AutoTokenizer.from_pretrained(cfg.id,{revision:cfg.revision,progress_callback});sentimentModel=await AutoModelForSequenceClassification.from_pretrained(cfg.id,{revision:cfg.revision,dtype:'q8',device:'wasm',progress_callback});}
  const inputs=sentimentTokenizer(payload.text,{truncation:true,max_length:256});
  const output=await sentimentModel(inputs),values=Array.from(output.logits.data),maximum=Math.max(...values),exponentials=values.map(v=>Math.exp(v-maximum)),sum=exponentials.reduce((a,b)=>a+b,0);
  const scores=Object.fromEntries(exponentials.map((v,i)=>[String(sentimentModel.config.id2label[i]).toLowerCase(),v/sum]));
  if(!['positive','negative','neutral'].every(k=>Object.hasOwn(scores,k)))throw Error('INVALID_SENTIMENT_LABELS');
  const sentiment=Object.entries(scores).sort((a,b)=>b[1]-a[1])[0][0];
  self.postMessage({id,result:{sentiment,scores,model:cfg.id,revision:cfg.revision}});
 }else if(type==='generate'){
  if(!model||!tokenizer)throw Error('MODEL_NOT_READY');
  const inputs=tokenizer.apply_chat_template(payload.messages,{add_generation_prompt:true,return_dict:true,enable_thinking:false});
  if(inputs.input_ids.dims.at(-1)>1600)throw Error('CONTEXT_TOO_LONG');
  const output=await model.generate({...inputs,max_new_tokens:Math.min(192,payload.maxTokens),do_sample:false});
  const ids=output.tolist()[0].slice(inputs.input_ids.dims.at(-1));
  self.postMessage({id,result:tokenizer.decode(ids,{skip_special_tokens:true})});
 }else throw Error('UNKNOWN_MODEL_ACTION');
}catch(e){self.postMessage({id,error:String(e.message||e)});}finally{busy=false;}};
