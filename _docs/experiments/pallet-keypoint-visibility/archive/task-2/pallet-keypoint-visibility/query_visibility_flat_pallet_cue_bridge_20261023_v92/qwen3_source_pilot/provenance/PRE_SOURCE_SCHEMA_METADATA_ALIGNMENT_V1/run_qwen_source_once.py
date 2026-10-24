"""Fixed public20 frozen Qwen3 source observations; never semantic labels or current inference."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback,random
import numpy as np
ROOT=Path(__file__).resolve().parent
V92=ROOT.parent
P=V92.parent
MODEL_DIR=Path('/dev/shm/pallet_v92_qwen3vl2b_89644892_v1')
SMALL_DIR=Path('/dev/shm/pallet_qwen3_vl2b_v92')
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
WEIGHT_SHA='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
SAM_PY=P/'foundation_runtime_sam21_20261023/venv/bin/python'
OUT=ROOT/'SOURCE20_ONE_SHOT'
PROMPT=('Image 1 is the full-frame context with the saved predicted target box in cyan and the current initial query in yellow. '
        'Image 2 is the unmarked original query crop, enlarged for viewing. Identify the visible content near the crop center. '
        'Distinguish flat wooden or plastic pallet structure from a box, shelf, bin, cart, drum, cone, other object, or floor. '
        'The expected native part is not supplied. Do not infer that an expected part is covered merely because a foreign surface is visible. '
        'Do not assign native keypoint identities or semantic D/E/S states. If target identity or visible evidence is ambiguous, say so. '
        'Return only one JSON object with exactly these fields: '
        '"local_object": "PALLET_STRUCTURE", "OTHER_OBJECT", "BACKGROUND", or "UNCERTAIN"; '
        '"visible_distinct_pallet_cue": "YES", "NO", or "UNCERTAIN"; '
        '"target_binding": "CLEAR" or "AMBIGUOUS"; '
        '"note": one short sentence describing only the visible evidence.')
ENUMS={'local_object':{'PALLET_STRUCTURE','OTHER_OBJECT','BACKGROUND','UNCERTAIN'},'visible_distinct_pallet_cue':{'YES','NO','UNCERTAIN'},'target_binding':{'CLEAR','AMBIGUOUS'}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,o):
 assert not Path(p).exists(),str(p)
 Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(d):
 assert isinstance(d,dict) and set(d)>={'path','sha256'}
 assert sha(d['path'])==d['sha256'],d['path']
 return Path(d['path'])
def js(p):return json.loads(Path(p).read_text())
def mem_available():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024

def parsed_output(text):
 """No semantic mapping or repair. Malformed text is retained as uncertain, never retried."""
 try:
  o=json.loads(text)
  assert isinstance(o,dict) and set(o)==set(ENUMS)|{'note'}
  assert all(isinstance(o[k],str) and o[k] in v for k,v in ENUMS.items())
  assert isinstance(o['note'],str) and 0<len(o['note'])<=400
  return o,None
 except Exception as e:return None,type(e).__name__

def make_context(rgb,box,q):
 from PIL import Image,ImageDraw
 assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
 image=Image.fromarray(rgb.copy());d=ImageDraw.Draw(image)
 b=np.asarray(box,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
 assert b.shape==(4,) and q.shape==(2,) and np.isfinite(b).all() and np.isfinite(q).all()
 w=max(2,int(round(max(image.size)/600)))
 d.rectangle(tuple(b.tolist()),outline=(0,255,255),width=w)
 x,y=q;rad=max(6,int(round(max(image.size)/160)))
 d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=(255,255,0),width=w)
 d.line((x-rad*2,y,x+rad*2,y),fill=(255,255,0),width=w)
 d.line((x,y-rad*2,x,y+rad*2),fill=(255,255,0),width=w)
 return image

def calls_for_flags(flags):
 f=np.asarray(flags);assert f.shape==(160,) and f.dtype==bool
 return list(np.flatnonzero(f))
def generated():
 from PIL import Image
 valid=np.zeros(160,bool);valid[np.arange(122)]=True
 assert calls_for_flags(valid)==list(range(122))
 assert len(calls_for_flags(np.zeros(160,bool)))==0
 good={'local_object':'PALLET_STRUCTURE','visible_distinct_pallet_cue':'YES','target_binding':'CLEAR','note':'A wooden edge is directly visible.'}
 assert parsed_output(json.dumps(good))[0]==good
 for bad in [dict(good,semantic_state='D'),dict(good,local_object='E'),dict(good,target_binding=''),{'note':'x'},good['note'],'```json\n'+json.dumps(good)+'\n```',json.dumps(good)+' trailing']:
  s=bad if isinstance(bad,str) else json.dumps(bad);assert parsed_output(s)[0] is None
 rgb=np.zeros((40,70,3),np.uint8);before=rgb.copy();im=make_context(rgb,[-3,1,60,35],[32,20]);assert np.array_equal(rgb,before)
 assert im.size==(70,40)
 crop=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);old=crop.copy();vlm=Image.fromarray(crop).resize((256,256),Image.Resampling.BICUBIC)
 assert vlm.size==(256,256) and np.array_equal(old,crop)
 assert 'native part is not supplied' in PROMPT and all(k in PROMPT for k in ENUMS)
 assert 'semantic D/E/S' in PROMPT
 result={'completed':True,'PASS':True,'code':bind(__file__),'generated_only':True,'all160_preserved_valid122_only_call_count_fixture':True,'invalid38_no_call_fixture':True,'strict_JSON_enum_no_semantic_mapping_no_repair':True,'parse_failure_is_explicit_UNPARSED_UNCERTAIN_no_retry':True,'inputRGB_unchanged_overlay_separate':True,'unmarked_RGB96_preserved_separate_256_view':True,'no_source_RGB_GT_weights_model_current_reads':True,'model_forward_fit_PnP':0,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()}
 write(ROOT/'GENERATED_QWEN_SOURCE_TESTS.json',result);print(json.dumps(result,indent=2))

def freeze():
 readiness=ROOT/'RUNTIME_PROCESSOR_READINESS_V2.json';assert js(readiness)['PASS']
 source_seal=V92/'rgb96_review/RGB96_REVIEW_SEAL.json';s=js(source_seal);assert s['completed'] and s['PASS']
 plan=V92/'model_feasibility/QWEN3VL2B_NATIVE_ALLOWLIST_PLAN.json';p=js(plan);assert p['revision']==REVISION and p['total_bytes']==4266647442
 protocol={'schema':'V92_QWEN3_SOURCE_PILOT_BEFORE_WEIGHTS_AND_FORWARD_V1','code':bind(__file__),'generated':bind(ROOT/'GENERATED_QWEN_SOURCE_TESTS.json'),'runtime_readiness':bind(readiness),'runtime_original_readiness':bind(ROOT/'RUNTIME_PROCESSOR_READINESS.json'),'runtime_preparation_code':bind(ROOT/'prepare_runtime_processor.py'),'processor_readiness_stdout':bind(ROOT/'PROCESSOR_READINESS_V2_STDOUT.json'),'asset_allowlist':bind(plan),'source_input_seal':bind(source_seal),'fixed20_policy':bind(V92/'FIXED20_SOURCE_QUERY_RGB_POLICY_BEFORE_PIXELS.json'),'model_id':'Qwen/Qwen3-VL-2B-Instruct','revision':REVISION,'weight_sha256':WEIGHT_SHA,'model_dir':str(MODEL_DIR),'processor_assets':str(SMALL_DIR),'runtime_python':str(SAM_PY),'runtime_overlay':str(ROOT/'runtime_overlay'),'prompt':PROMPT,'frames':20,'slots':160,'predicted_query_slots':144,'no_detection_slots':16,'valid_inference_slots':122,'no_call_U':38,'native8_semantic_mapping':'UNKNOWN','q_L_source_order':'exact saved original, no recenter/clip/L change or GT choice','views':{'context':'canonical fullRGB copy; only saved modelbbox cyan + current q yellow; native rawRGB unchanged','query':'original saved unmarked RGB96 preserved; separate PILbicubic256x256 VLM view. Upscaling adds no detail.','official_processor':'AutoProcessor local_only trust_remote_codeFalse use_fastTrue; images_kwargs size shortest_edge65536 longest_edge262144 pixel-area bounds. Context aspect-preserving official32multiple resize; query256fixed grid16x16patch/64merged tokens.','valid_mask':'review metadata only; no extra modeltensor'},'model':{'dtype':'bfloat16','device':'cuda:0','attn_implementation':'sdpa','low_cpu_mem_usage':True,'local_files_only':True,'trust_remote_code':False,'use_safetensors':True,'batch_size':1,'eval_inference_mode':True,'TF32':False,'seed':20261023,'do_sample':False,'num_beams':1,'max_new_tokens':96},'resource_guards':{'RAM_before_cache_GiB':12,'RAM_before_load_GiB':8,'GPU_free_before_load_GiB':7,'disk_available_min_bytes':500000000,'tmpfs_one_copy_only':True},'runtime_exceptions':'halt, preserve partial output/log; no automatic retry','model_format_failures':'retain raw text and UNPARSED_UNCERTAIN, continue fixed remainingcalls without repair/retry','data_interpretation':'Exploratory visible-content observations only; no D/E/S teacher/gold, no current admission or fully-solved claim','output':str(OUT),'current_or_GT_reads':0,'fit_PnP':0,'repeat_forward':0}
 write(ROOT/'SOURCE_PILOT_PROTOCOL_BEFORE_MODEL_WEIGHTS.json',protocol)
 template={'schema':'V92_ROOT_APPROVED_QWEN3_SOURCE_FORWARD_ENTRY_V1','approved':False,'source_only':True,'protocol':bind(ROOT/'SOURCE_PILOT_PROTOCOL_BEFORE_MODEL_WEIGHTS.json'),'code':bind(__file__),'generated':bind(ROOT/'GENERATED_QWEN_SOURCE_TESTS.json'),'independent_review':{'path':'ROOT_FILL','sha256':'ROOT_FILL'},'root_weight_receipt':{'path':'ROOT_FILL','sha256':'ROOT_FILL'},'source_input_seal':bind(source_seal),'runtime_readiness':bind(readiness),'allowlist':bind(plan),'current':False,'fit':False,'PnP':False,'output_directory':str(OUT)}
 write(ROOT/'ROOT_FORWARD_ENTRYLOCK_TEMPLATE.json',template)
 print(json.dumps({'protocol':bind(ROOT/'SOURCE_PILOT_PROTOCOL_BEFORE_MODEL_WEIGHTS.json'),'template':bind(ROOT/'ROOT_FORWARD_ENTRYLOCK_TEMPLATE.json')},indent=2))

def authorize(entry_path):
 e=js(entry_path);assert e['schema']=='V92_ROOT_APPROVED_QWEN3_SOURCE_FORWARD_ENTRY_V1' and e['approved'] is True and e['source_only'] is True
 assert all(e[x] is False for x in ['current','fit','PnP'])
 # Authorization and code/protocol/review proof before source artifacts/model parameters.
 assert e['code']['sha256']==sha(__file__);check(e['code']);p=js(check(e['protocol']))
 assert p['code']['sha256']==sha(__file__) and p['prompt']==PROMPT and p['revision']==REVISION
 for k in ['generated','runtime_readiness','allowlist','source_input_seal']:
  assert e[k]['sha256']==p['asset_allowlist' if k=='allowlist' else k]['sha256'];check(e[k])
 assert js(check(e['independent_review']))['PASS'] is True
 assert js(check(e['generated']))['PASS'] and js(check(e['runtime_readiness']))['PASS']
 receipt=js(check(e['root_weight_receipt']));assert receipt.get('completed') is True and receipt.get('PASS') is True
 assert Path(p['model_dir'])==MODEL_DIR and str(MODEL_DIR).startswith('/dev/shm/') and not (MODEL_DIR/'model.safetensors').is_symlink()
 assert sha(MODEL_DIR/'model.safetensors')==WEIGHT_SHA
 plan=js(check(e['allowlist']));assert plan['revision']==REVISION
 for f in plan['files']:
  q=MODEL_DIR/f['name'];assert q.is_file() and q.stat().st_size==f['bytes']
  if f['name']!='model.safetensors':assert hashlib.sha1(b'blob '+str(f['bytes']).encode()+b'\0'+q.read_bytes()).hexdigest()==f['git_blob_sha1']
 assert Path(sys.executable).resolve()==SAM_PY.resolve()
 assert e['output_directory']==str(OUT) and not OUT.exists()
 return e,p

def run(entry_path):
 e,p=authorize(entry_path);OUT.mkdir();start=time.time()
 # No source-pixel reads or model parameters before above gate.
 import torch,transformers
 from PIL import Image
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 assert torch.__version__=='2.5.1+cu121' and transformers.__version__=='4.57.6'
 assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1'
 assert mem_available()>=8*2**30
 assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=500_000_000
 gpu_free,gpu_total=torch.cuda.mem_get_info();assert gpu_free>=7*2**30
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(20261023);np.random.seed(20261023);random.seed(20261023)
 processor=AutoProcessor.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,use_fast=True)
 model,loading=Qwen3VLForConditionalGeneration.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True,device_map={'':'cuda:0'},output_loading_info=True)
 assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
 model.eval();write(OUT/'MODEL_LOAD_INFO.json',loading)
 # Whitelisted original input artifacts only, never initializer GT or semantic annotations.
 s=js(check(e['source_input_seal']));assert s['completed'] and s['PASS']
 files=s['outputs'];descs={Path(f['path']).name:f for f in files}
 npz=check(descs['INPUT_RGB96_ONLY.npz']);rows_path=check(descs['ROWS_INPUT_ONLY.jsonl'])
 manifest=js(check(s['protocol']))['input_manifest'];frame_path=check(manifest)
 frames=[json.loads(x) for x in frame_path.read_text().splitlines()];rows=[json.loads(x) for x in rows_path.read_text().splitlines()]
 assert len(frames)==20 and len(rows)==160
 z=np.load(npz,allow_pickle=False);valid=z['query_center_valid'];assert valid.shape==(160,) and valid.dtype==bool and int(valid.sum())==122
 assert z['rgb96'].shape==(160,3,96,96) and z['rgb96'].dtype==np.uint8
 assert z['query_xy'].shape==(160,2) and z['L'].shape==(160,)
 calls=0;raw_rgb=None;lastframe=-1
 with (OUT/'ALL160_SOURCE_OBSERVATIONS.jsonl').open('x') as out:
  for i,row in enumerate(rows):
   fi=i//8;k=i%8;frame=frames[fi]
   result={'slot_index':i,'frame_index':fi,'native_slot_logical_only':k,'input_id':frame['id'],'image_sha256':frame['image_sha256'],'q':z['query_xy'][i].tolist() if z['query_exists'][i] else None,'L':float(z['L'][i]) if z['query_exists'][i] else None,'query_center_valid':bool(valid[i]),'semantic_native8':'UNKNOWN','model_output_is_gold':False,'model_calls_for_slot':0,'raw_text':None,'parsed_observation':None}
   if not valid[i]:result['status']='U_NO_MODEL_CALL';result['reason']='NO_SAVED_QUERY' if not z['query_exists'][i] else 'INVALID_QUERY_CENTRE'
   else:
    if fi!=lastframe:
     rawpath=frame['decoded_RGB_input'];rp=check(rawpath);raw_rgb=np.load(rp,allow_pickle=False)
     assert raw_rgb.dtype==np.uint8 and list(raw_rgb.shape[:2])==frame['raw_hw']
     assert hashlib.sha256(raw_rgb.tobytes()).hexdigest()==frame['RGB_pixels_sha256']
     lastframe=fi
    q=z['query_xy'][i];assert np.array_equal(q,np.asarray(frame['query8_xy'])[k]) and float(z['L'][i])==float(frame['L'])
    context=make_context(raw_rgb,frame['selected_model_bbox_xyxy'],q)
    crop=Image.fromarray(z['rgb96'][i].transpose(1,2,0)).resize((256,256),Image.Resampling.BICUBIC)
    msgs=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':PROMPT}]}]
    text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=processor(text=[text],images=[context,crop],return_tensors='pt',images_kwargs={'size':{'shortest_edge':65536,'longest_edge':262144}})
    grid=inp['image_grid_thw'].tolist();assert len(grid)==2 and grid[1]==[1,16,16] and grid[0][1]*grid[0][2]*256<=262144
    inp=inp.to('cuda:0')
    with torch.inference_mode():generated=model.generate(**inp,do_sample=False,num_beams=1,max_new_tokens=96)
    textout=processor.batch_decode(generated[:,inp['input_ids'].shape[-1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    parsed,error=parsed_output(textout);calls+=1
    result.update(status='PARSED_MODEL_OBSERVATION' if parsed else 'UNPARSED_UNCERTAIN',reason=error,raw_text=textout,parsed_observation=parsed,model_calls_for_slot=1,image_grid_thw=grid,generated_tokens=int(generated.shape[-1]-inp['input_ids'].shape[-1]),crop_rgb96_sha256=hashlib.sha256(z['rgb96'][i].tobytes()).hexdigest(),context_overlay_pixels_sha256=hashlib.sha256(np.asarray(context).tobytes()).hexdigest())
   out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
   print(json.dumps({'completed_slots':i+1,'model_calls':calls,'status':result['status']}),flush=True)
 assert calls==122
 assert sha(MODEL_DIR/'model.safetensors')==WEIGHT_SHA
 write(OUT/'SOURCE_PILOT_SEAL.json',{'completed':True,'PASS':True,'entry':bind(entry_path),'code':bind(__file__),'protocol':e['protocol'],'runtime_readiness':e['runtime_readiness'],'root_weight_receipt':e['root_weight_receipt'],'frames':20,'slots':160,'inference_slots':122,'U_without_model':38,'outputs':[bind(OUT/'ALL160_SOURCE_OBSERVATIONS.jsonl'),bind(OUT/'MODEL_LOAD_INFO.json')],'model_weight_file_unchanged':True,'model_training':False,'model_state_optimizer_updates':0,'model_output_is_gold':False,'semantic_native8':'UNKNOWN','current223':0,'PnP':0,'seconds':time.time()-start,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generated','freeze','run']);ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.stage=='generated':generated()
 elif a.stage=='freeze':freeze()
 else:
  assert a.entry
  try:run(a.entry)
  except BaseException as ex:
   if OUT.exists():
    q=OUT/'HALTED_FAILURE.json'
    if not q.exists():write(q,{'completed':False,'error':repr(ex),'traceback':traceback.format_exc(),'silent_retry':False,'partial_outputs_preserved':True})
   raise
if __name__=='__main__':main()
