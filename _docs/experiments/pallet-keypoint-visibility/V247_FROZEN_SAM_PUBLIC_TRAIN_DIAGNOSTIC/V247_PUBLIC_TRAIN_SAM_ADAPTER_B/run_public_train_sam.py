"""Frozen SAM2.1 on exact public TRAIN44/352. Root true entry only.
No teacher, GT, old source input, training, PnP or evaluation reader.
"""
import argparse,ast,csv,hashlib,importlib.metadata,io,json,math,os,stat,sys,time,traceback
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';H=R/'V247_PUBLIC_TRAIN_SAM_ADAPTER_B'
RUNTIME=P/'foundation_runtime_sam21_20261023'
ROOT_PROTOCOL={'path':str(R/'V247_ROOT_PUBLIC_TRAIN_SAM/PROTOCOL_BEFORE_NEW_SOURCE_SAM_OUTPUTS.json'),'sha256':'8569a11e7b77ae5123a86660efb288a3853bd12a014113d9595c73408895cb0e','size_bytes':6148}
INPUTS={'frames':{'path':str(R/'V246_ROOT_PALLOX_TRAIN/train_input_metadata_v1/TRAIN44_FRAMES_INPUT_ONLY.jsonl'),'sha256':'ec53a055ab7de0ef7e5d3414b39a634dd0a0069534682820d62654dc2ff4341d','size_bytes':21294},'queries':{'path':str(R/'V246_ROOT_PALLOX_TRAIN/train_input_metadata_v1/TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl'),'sha256':'636df349195319949f81b4ad172f3b41161efe1753bcc39095f1a69e240c1284','size_bytes':184369}}
CHECKPOINT={'path':str(RUNTIME/'checkpoints/sam2.1_hiera_base_plus.pt'),'sha256':'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5','size_bytes':323606802}
SEAL_SHA='e5655309c2c99ee18e76fa0ebe85541194445895673d9b64f915384130bcbf82'
MANIFEST_SHA='36d685db5d72d4e60db2919278b720940e03576e0c2ee268a7d50fc759d85750'
SOURCE_MANIFEST_SHA='cd12433757525dbcf3757383b27674bdab502a5b7c19e6b81476959491ac71fb'
UNUSED_ALIASES={'sam2/sam2_hiera_b+.yaml','sam2/sam2_hiera_l.yaml','sam2/sam2_hiera_s.yaml','sam2/sam2_hiera_t.yaml'}
LOADER_SHA='856d64d71e44407401297b551884fb4cd1aba8082de0ff5fc60fac3dd42f094d'
CONFIG='configs/sam2.1/sam2.1_hiera_b+.yaml'
REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4'
FRAME_FIELDS={'frame_input_id','image','image_sha256','pair_id','raw_hw','seed_V84_reuse','session','source_family','split','view'}
QUERY_FIELDS={'query_id','frame_index','frame_input_id','pair_id','view','split','native_slot','query_xy','roi_side','original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid','reason','native_semantic_state','human_gold'}

def require(condition,message):
    if not condition:raise ValueError(message)

def native_pixel(query,hw):
    q=np.asarray(query,dtype=np.float64);require(q.shape==(2,),'A native query must be xy')
    h,w=map(int,hw);require(h>0 and w>0,'Positive native H/W required')
    if not np.isfinite(q).all():return False,(-1,-1),'NONFINITE_QUERY'
    x,y=map(int,np.floor(q+.5))
    if not(0<=q[0]<w and 0<=q[1]<h):return False,(x,y),'CONTINUOUS_OUTSIDE_NATIVE'
    if not(0<=x<w and 0<=y<h):return False,(x,y),'ROUNDED_OUTSIDE_NATIVE'
    return True,(x,y),'VALID'

def select_mask(masks,quality,hw):
    masks=np.asarray(masks);quality=np.asarray(quality)
    require(masks.shape==(3,int(hw[0]),int(hw[1])) and quality.shape==(3,),'Three full-native mask logits/quality scores required')
    require(masks.dtype==np.float32,'Official returned full-native logits must remain float32, with no half-storage roundtrip')
    require(np.isfinite(masks).all() and np.isfinite(quality).all(),'Finite predicted masks/quality required')
    selected=int(np.argmax(quality))
    return np.ascontiguousarray(masks[selected],dtype=np.float32),np.ascontiguousarray(quality,dtype=np.float32),selected

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        a=np.ascontiguousarray(tensor.detach().cpu().numpy())
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()

def prompt_box(q8):
 try:a=np.asarray(q8,dtype=np.float64)
 except (TypeError,ValueError):return None,'Q8_MISSING_OR_NONNUMERIC'
 if a.shape!=(8,2) or not np.isfinite(a).all():return None,'Q8_MISSING_OR_NONFINITE'
 b=np.array([a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()],dtype=np.float64)
 if not(b[2]>b[0] and b[3]>b[1]):return None,'Q8_DEGENERATE_BOX'
 return b,None

def lexical(path):
 text=str(path)
 require(not any(x in text.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')),'Forbidden before filesystem')
 p=Path(text);require(p.is_absolute() and '..' not in p.parts,'Absolute literal path required');return p

def guard(path):
 p=lexical(path)
 for ancestor in reversed((p,*p.parents)):
  try:mode=os.lstat(ancestor).st_mode
  except FileNotFoundError:continue
  require(not stat.S_ISLNK(mode),'Symlink bound artifact forbidden')
 return p

def identity(b):
 p=lexical(b['path']);d=b['sha256'];require(isinstance(d,str) and len(d)==64 and all(c in '0123456789abcdef' for c in d),'SHA256 required');return str(p),d

def read(path):
 with os.fdopen(os.open(guard(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def checked(b):
 identity(b);p=guard(b['path']);digest=hashlib.sha256();size=0
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block);size+=len(block)
 require(digest.hexdigest()==b['sha256'],'Changed bound bytes')
 for key in ('bytes','size_bytes'):
  if key in b:require(type(b[key]) is int and b[key]==size,'Bound size changed')
 return p

def bind(path):
 p=guard(path);raw=read(p);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def json_bound(b):return json.loads(read(checked(b)))

def write_json(path,obj):
 p=guard(path)
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return bind(p)

def validate_inputs(frames,queries,RGB_white):
 require(len(frames)==44 and len(queries)==352 and len(RGB_white)==44,'Fixed44frames/352queries/RGB44')
 fm={};groups={};white={identity(b):b for b in RGB_white};require(len(white)==44,'Unique exactRGB44 whitelist')
 for f in frames:
  require(set(f)==FRAME_FIELDS and f['split']=='TRAIN_LAB' and f['session'].endswith('_lab') and f['raw_hw']==[480,640],'Exact public TRAIN native frame schema')
  require(f['frame_input_id'] not in fm,'Unique frame ID');fm[f['frame_input_id']]=f;groups[f['frame_input_id']]=[]
  require(identity({'path':f['image'],'sha256':f['image_sha256']}) in white,'Frame RGB absent from exact whitelist')
 require({identity({'path':f['image'],'sha256':f['image_sha256']}) for f in frames}==set(white),'RGB whitelist contains no extra images')
 last=-1;seen=set()
 for q in queries:
  require(set(q)==QUERY_FIELDS and q['query_id'] not in seen,'Original15field query schema/uniqueID');seen.add(q['query_id'])
  f=fm[q['frame_input_id']];require(all(q[k]==f[k] for k in ('pair_id','view','split')),'Original perframe joins')
  require(type(q['frame_index']) is int and type(q['native_slot']) is int and 0<=q['native_slot']<8 and q['query_id']==q['frame_input_id']+':native'+str(q['native_slot']),'Original source ordinal/nativeID')
  ordinal=q['frame_index']*8+q['native_slot'];require(ordinal>last,'Original source global order retained');last=ordinal
  require(q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False,'No native authority upgrade')
  for k in ('original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid'):require(type(q[k]) is bool,'Original support flags')
  if q['query_xy'] is None:
   require(q['roi_side'] is None and q['reason']=='NO_DETECTION_U' and not any(q[k] for k in ('original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid')),'Original absentU')
  else:
   xy=np.asarray(q['query_xy'],np.float64);require(xy.shape==(2,) and np.isfinite(xy).all() and isinstance(q['roi_side'],(int,float)) and math.isfinite(q['roi_side']) and q['roi_side']>0,'Original finite q/L')
   x,y=xy;h,w=f['raw_hw'];continuous=bool(0<=x<w and 0<=y<h);nearest=bool(0<=np.floor(x+.5)<w and 0<=np.floor(y+.5)<h);centre=bool(-.5<=x<w-.5 and -.5<=y<h-.5)
   require((continuous,nearest,centre)==(q['original_query_continuous_valid'],q['query_nearest_pixel_valid'],q['query_rgb96_centre_domain_valid']),'Original continuous/rounded support separate')
   require(q['reason']==('VALID_QUERY_NOT_A_VISIBILITY_LABEL' if continuous else 'OOB_OR_INVALID_U'),'Original U reason')
  groups[q['frame_input_id']].append(q)
 require(sum(q['original_query_continuous_valid'] for q in queries)==297 and len({f['pair_id'] for f in frames})==22,'Fixed297available/22pairs')
 for fid,rows in groups.items():require([q['native_slot'] for q in rows]==list(range(8)) and len({q['frame_index'] for q in rows})==1,'Original eight slots without renumbering')
 return groups,white

def proposal_readout(chosen,q,prompt_available,runtime_available):
 invalid={'prediction_available':False,'state':'U','native_query_valid':False,'pixel':None,'exact_logit':None,'radius2_logit':None,'exact_mask_support':None,'radius2_mask_support':None,'radius2_window_pixels':0,'radius2_clipped':False}
 if q['query_xy'] is None:return dict(invalid,U_reason='ORIGINAL_QUERY_ABSENT')
 valid,pixel,reason=native_pixel(q['query_xy'],[480,640]);require(valid==(q['original_query_continuous_valid'] and q['query_nearest_pixel_valid']),'Original native support unchanged')
 invalid.update(native_query_valid=valid,pixel=list(pixel))
 if not valid:return dict(invalid,U_reason='ORIGINAL_'+reason)
 if not prompt_available:return dict(invalid,U_reason='PROMPT_UNAVAILABLE')
 if not runtime_available:return dict(invalid,U_reason='RUNTIME_UNAVAILABLE')
 require(chosen.shape==(480,640) and chosen.dtype==np.float32 and np.isfinite(chosen).all(),'Native F32 chosen mask')
 x,y=pixel;x0,x1=max(0,x-2),min(640,x+3);y0,y1=max(0,y-2),min(480,y+3)
 exact=float(chosen[y,x]);near=float(chosen[y0:y1,x0:x1].max());inside=bool(exact>0)
 return {'prediction_available':True,'state':'SAM_VISIBLE_MASK_SUPPORT' if inside else 'OUTSIDE_PROPOSAL','native_query_valid':True,'pixel':[x,y],'exact_logit':exact,'radius2_logit':near,'exact_mask_support':inside,'radius2_mask_support':bool(near>0),'radius2_window_pixels':(x1-x0)*(y1-y0),'radius2_clipped':(x1-x0)*(y1-y0)!=25,'U_reason':None}

def runtime_preflight(e):
 rt=e['runtime'];require(rt['python']==str(RUNTIME/'venv/bin/python') and Path(sys.executable).absolute()==Path(rt['python']),'Sealed runtime interpreter required')
 require(rt['repo_root']==str(RUNTIME/'sam2_repo') and rt['repo_revision']==REVISION and rt['config']==CONFIG,'Fixed official vendor/config')
 require(identity(rt['runtime_seal'])==(str(RUNTIME/'RUNTIME_SEAL.json'),SEAL_SHA),'Exact runtime seal')
 require(identity(rt['code_manifest'])==(str(R/'V247_ROOT_PUBLIC_TRAIN_SAM/RUNTIME_CODE_REGULAR83_MANIFEST.json'),MANIFEST_SHA),'Exact regular83 official source manifest')
 seal=json_bound(rt['runtime_seal']);manifest=json_bound(rt['code_manifest'])
 require(seal['READY'] is True and seal['repo_revision']==REVISION and rt['pinned_versions']==seal['pinned_versions'],'Existing sealed runtime metadata')
 for package,version in rt['pinned_versions'].items():require(importlib.metadata.version(package)==version,'Runtime version drift: '+package)
 # Only exact official source/config files; never seal.bindings/wheels/smoke data.
 require(identity(manifest['source_manifest'])==(str(RUNTIME/'provenance/SAM2_REPO_MANIFEST.json'),SOURCE_MANIFEST_SHA),'Preserved old87 source manifest metadata')
 original=json_bound(manifest['source_manifest']);excluded=manifest['excluded_legacy_symlink_aliases']
 require({x['relative_path'] for x in excluded}==UNUSED_ALIASES and len(excluded)==4,'Exactly four unused inert aliases')
 require(all(x['target_not_read_or_followed'] is True and x['recorded_original_sha256']==original['code_files_sha256'][x['relative_path']] for x in excluded),'Alias metadata only; no filesystem access')
 repo=guard(rt['repo_root']);files=manifest['code_files_sha256'];require(isinstance(files,dict) and len(files)==83 and files=={k:v for k,v in original['code_files_sha256'].items() if k not in UNUSED_ALIASES} and files.get('sam2/build_sam.py')==LOADER_SHA and 'sam2/'+CONFIG in files,'Pinned regular loader/config closure')
 for rel,digest in files.items():
  rp=Path(rel);require(not rp.is_absolute() and '..' not in rp.parts and rp.suffix in ('','.md','.py','.toml','.yaml','.yml'), 'CODE/config/notice-only relative manifest')
  checked({'path':str(repo/rp),'sha256':digest})
 loader=repo/'sam2/build_sam.py';tree=ast.parse(read(loader));functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_load_checkpoint'];require(len(functions)==1,'Official checkpoint loader')
 calls=[n for n in ast.walk(functions[0]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
 require(len(calls)==1 and {k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}=={'map_location':'cpu','weights_only':True},'CPU weights_only=True official loader')
 for call in ast.walk(functions[0]):
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='load_state_dict':require(not any(k.arg=='strict' and ast.literal_eval(k.value) is False for k in call.keywords),'Strict official load')
 require(e['checkpoint']==CHECKPOINT,'Only official base-plus checkpoint');checked(CHECKPOINT)
 sys.path.insert(0,str(repo))
 import torch
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 require(torch.__version__=='2.5.1+cu121' and torch.cuda.is_available(),'Frozen CUDA runtime')
 import inspect
 require(Path(inspect.getfile(build_sam2)).absolute()==loader and Path(inspect.getfile(SAM2ImagePredictor)).absolute()==repo/'sam2/sam2_image_predictor.py','Pinned imported API paths')
 torch.manual_seed(20261023);np.random.seed(20261023);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 return torch,build_sam2,SAM2ImagePredictor,{'repo_revision':REVISION,'official_code_files_verified':len(files),'loader_weights_only_CPU_strict':True,'versions':rt['pinned_versions'],'oldseal_payload_or_wheel_follow':0}

def run(entry_path):
 eb=read(entry_path);e=json.loads(eb)
 require(e.get('approved') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Root true SOURCE-only entry required')
 require(identity(e['code'])==identity(bind(__file__)) and identity(e['root_protocol'])==identity(ROOT_PROTOCOL),'Exact adapter/root protocol')
 protocol=json_bound(e['protocol']);root=json_bound(e['root_protocol']);scope=json_bound(e['read_scope'])
 require(identity(protocol['code'])==identity(e['code']),'Implementation code pin')
 require(scope.get('approved_SOURCE_TRAIN_frozen_SAM_diagnostic') is True and scope.get('no_GT_teacher_current_or_training') is True,'Root bounded read scope')
 for key in ('code','protocol','root_protocol'):require(identity(scope[key])==identity(e[key]),'Scope exact code/protocol '+key)
 for key in ('inputs','RGB_whitelist44','runtime','checkpoint'):require(scope[key]==e[key],'Scope exact '+key)
 require(e['inputs']==INPUTS and root['model_fixed_before_new_outputs']['checkpoint_sha256']==CHECKPOINT['sha256'],'Frozen cohort/checkpoint')
 frames=[json.loads(x) for x in read(checked(INPUTS['frames'])).splitlines()];queries=[json.loads(x) for x in read(checked(INPUTS['queries'])).splitlines()];groups,white=validate_inputs(frames,queries,e['RGB_whitelist44'])
 out=guard(e['output_directory']);require(out==R/'V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1' and not out.exists(),'Fixed fresh output')
 out.mkdir();(out/'masks').mkdir();start=time.monotonic();statuses=[];answers={};fatal=None;model=None;torch=None;state_before=None;state_after=None;runtime_info=None;compile_calls=0;calls={'set_image':0,'predict':0}
 for i,f in enumerate(frames):
  box,why=prompt_box([q['query_xy'] for q in groups[f['frame_input_id']]])
  statuses.append({'frame_position':i,'original_frame_input':f,'prompt_available':box is not None,'prompt_reason':why,'box_xyxy':None if box is None else box.tolist(),'runtime_available':False,'runtime_reason':'NOT_ATTEMPTED','output':None})
 write_json(out/'STARTED.json',{'entry':bind(entry_path),'code':e['code'],'protocol':e['protocol'],'root_protocol':e['root_protocol'],'inputs':INPUTS,'frames':44,'queries':352,'actual_payload_access_before_entry':0,'optimizer_updates':0})
 old_compile=None
 try:
  torch,build_sam2,Predictor,runtime_info=runtime_preflight(e)
  def no_compile(*args,**kwargs):
   nonlocal compile_calls
   compile_calls+=1;raise AssertionError('torch.compile forbidden')
  old_compile=torch.compile;torch.compile=no_compile
  model=build_sam2(CONFIG,str(CHECKPOINT['path']),device='cuda',mode='eval',apply_postprocessing=False,hydra_overrides_extra=['++model.compile_image_encoder=false'])
  model.requires_grad_(False);require(not model.training and all(not p.requires_grad and p.dtype==torch.float32 for p in model.parameters()),'Frozen FP32 official model')
  state_before=state_digest(model);predictor=Predictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
  from PIL import Image,__version__ as PIL_VERSION
  for s in statuses:
   if not s['prompt_available']:s['runtime_reason']='PROMPT_UNAVAILABLE';continue
   f=s['original_frame_input'];rows=groups[f['frame_input_id']];b=white[identity({'path':f['image'],'sha256':f['image_sha256']})]
   try:
    encoded=read(checked(b))
    with Image.open(io.BytesIO(encoded)) as im:rgb=np.asarray(im.convert('RGB'),np.uint8).copy()
    require(rgb.shape==(480,640,3),'Original native RGB shape');decoded=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    if 'decoded_RGB_sha256' in b:require(decoded==b['decoded_RGB_sha256'],'Declared decoded RGB bytes')
    box=np.asarray(s['box_xyxy'],np.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     calls['set_image']+=1;predictor.set_image(rgb)
     calls['predict']+=1;masks,quality,_=predictor.predict(point_coords=None,point_labels=None,box=box,multimask_output=True,return_logits=True)
    chosen,quality,selected=select_mask(masks,quality,[480,640]);s.update(runtime_available=True,runtime_reason=None,quality3=quality.tolist(),selected_index=selected,decoded_RGB_sha256=decoded,decoder={'Pillow':PIL_VERSION,'operation':'Image.open(BytesIO).convert(RGB)'})
    readouts=[proposal_readout(chosen,q,True,True) for q in rows]
    mask_path=out/'masks'/('FRAME_%03d.npz'%s['frame_position'])
    with os.fdopen(os.open(mask_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as stream:
     np.savez_compressed(stream,chosen_logit=chosen,quality_scores=quality,selected_index=np.int64(selected),original_prompt_box=np.asarray(s['box_xyxy'],np.float64),predictor_prompt_box=box,raw_hw=np.array([480,640],np.int64),original_q8=np.asarray([[np.nan,np.nan] if q['query_xy'] is None else q['query_xy'] for q in rows],np.float64),original_L=np.asarray([np.nan if q['roi_side'] is None else q['roi_side'] for q in rows],np.float64),native_slot=np.asarray([q['native_slot'] for q in rows],np.int8),original_query_index=np.asarray([q['frame_index']*8+q['native_slot'] for q in rows],np.int64),query_id=np.asarray([q['query_id'] for q in rows]),original_continuous_valid=np.asarray([q['original_query_continuous_valid'] for q in rows],bool),original_nearest_valid=np.asarray([q['query_nearest_pixel_valid'] for q in rows],bool),prediction_available=np.asarray([x['prediction_available'] for x in readouts],bool),exact_logit=np.asarray([np.nan if x['exact_logit'] is None else x['exact_logit'] for x in readouts],np.float32),radius2_logit=np.asarray([np.nan if x['radius2_logit'] is None else x['radius2_logit'] for x in readouts],np.float32))
    s['output']=bind(mask_path)
    for q,z in zip(rows,readouts):answers[q['query_id']]=z
    print(json.dumps({'frame_position':s['frame_position'],'completed_frames':sum(x['runtime_available'] for x in statuses),'prompt_valid_frames':sum(x['prompt_available'] for x in statuses)}),flush=True)
   except BaseException as exc:s['runtime_reason']=type(exc).__name__+': '+str(exc);raise
 except BaseException as exc:fatal={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 finally:
  if model is not None and state_before is not None:
   try:state_after=state_digest(model)
   except BaseException as exc:
    if fatal is None:fatal={'type':type(exc).__name__,'message':'Final model state SHA: '+str(exc)}
  if torch is not None and old_compile is not None:torch.compile=old_compile
 rows=[];status_by_id={x['original_frame_input']['frame_input_id']:x for x in statuses}
 for q in queries:
  s=status_by_id[q['frame_input_id']];z=answers.get(q['query_id'],proposal_readout(None,q,s['prompt_available'],False))
  rows.append({'original_input_unchanged':q,'original_query_index':q['frame_index']*8+q['native_slot'],'prompt_available':s['prompt_available'],'runtime_available':s['runtime_available'],'selected_index':s.get('selected_index'),'quality3':s.get('quality3'),'box_xyxy':s['box_xyxy'],'scene_output':s['output'],**z,'native_visibility_label':None,'semantic_SELF':None,'training_admission':False})
 rp=out/'ALL352_SAM_PROPOSAL_READOUTS.jsonl'
 with os.fdopen(os.open(rp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 cp=out/'QUERY352.csv'
 with os.fdopen(os.open(cp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
  names=['query_id','original_query_index','frame_index','frame_input_id','pair_id','view','native_slot','query_x','query_y','L','original_continuous_valid','original_nearest_valid','prompt_available','runtime_available','prediction_available','state','U_reason','exact_logit','radius2_logit','exact_mask_support','radius2_mask_support','selected_index','quality0','quality1','quality2'];writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
  for row in rows:
   q=row['original_input_unchanged'];xy=q['query_xy'];quality=row['quality3'];writer.writerow({'query_id':q['query_id'],'original_query_index':row['original_query_index'],'frame_index':q['frame_index'],'frame_input_id':q['frame_input_id'],'pair_id':q['pair_id'],'view':q['view'],'native_slot':q['native_slot'],'query_x':None if xy is None else xy[0],'query_y':None if xy is None else xy[1],'L':q['roi_side'],'original_continuous_valid':q['original_query_continuous_valid'],'original_nearest_valid':q['query_nearest_pixel_valid'],**{k:row[k] for k in ('prompt_available','runtime_available','prediction_available','state','U_reason','exact_logit','radius2_logit','exact_mask_support','radius2_mask_support','selected_index')},**{'quality'+str(i):None if quality is None else quality[i] for i in range(3)}})
 checkpoint_unchanged=False
 if runtime_info is not None:
  try:checked(CHECKPOINT);checkpoint_unchanged=True
  except BaseException as exc:
   if fatal is None:fatal={'type':type(exc).__name__,'message':'Checkpoint final SHA: '+str(exc)}
 passed=fatal is None and state_before is not None and state_before==state_after and checkpoint_unchanged and compile_calls==0 and all(s['runtime_available'] or not s['prompt_available'] for s in statuses)
 if fatal is not None:write_json(out/'FAILURE.json',{'completed':False,'PASS':False,'fatal':fatal,'automatic_retry':False})
 complete={'completed':True,'PASS':passed,'entry':bind(entry_path),'code':e['code'],'protocol':e['protocol'],'root_protocol':e['root_protocol'],'read_scope':e['read_scope'],'inputs':INPUTS,'RGB_whitelist44':e['RGB_whitelist44'],'runtime':runtime_info,'checkpoint':CHECKPOINT,'frames':44,'queries':352,'frame_outputs':statuses,'all_rows':bind(rp),'query_CSV':bind(cp),'model_state_before_sha256':state_before,'model_state_after_sha256':state_after,'model_state_byte_unchanged':state_before is not None and state_before==state_after,'checkpoint_file_byte_unchanged':checkpoint_unchanged,'calls':calls,'torch_compile_calls':compile_calls,'original_continuous_available':297,'original_input_U':55,'states':{key:sum(x['state']==key for x in rows) for key in ('SAM_VISIBLE_MASK_SUPPORT','OUTSIDE_PROPOSAL','U')},'threshold_strict_gt0':True,'canonical_prediction_semantics':'Segmentation proposal support only, no native D/E/SELF','GT_teacher_pose_depth_model_inputs':False,'new_training_updates':0,'PnP':0,'current_or_held_payloads':0,'native_labels_created':0,'training_admission':False,'model_adopted':False,'automatic_retry':False,'fatal':fatal,'seconds':time.monotonic()-start}
 cb=write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':passed,'COMPLETE':cb}),flush=True)
 if not passed:raise SystemExit('Source run failure preserved; no automatic retry')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
