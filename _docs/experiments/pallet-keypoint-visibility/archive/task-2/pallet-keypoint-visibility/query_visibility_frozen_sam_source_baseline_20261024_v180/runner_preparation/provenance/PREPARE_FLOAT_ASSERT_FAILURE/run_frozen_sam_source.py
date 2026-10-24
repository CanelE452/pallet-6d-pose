"""V180 exact official SAM1 ViT-B source-TRAIN proposal baseline.
--prepare: metadata, source hashes, import probe, invented geometry tests ONLY.
--entry: separately approved immutable entry, future source RGB/weights/forward.
"""
import argparse,hashlib,json,math,os,sys,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
V=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;P=V.parent
VENDOR=V/'vendor/segment-anything';COMMIT='dca509fe793f601edb92606367a655c15ac00fdf'
SCOPE=V/'ROOT_SCOPE_BEFORE_SAM_CODE_WEIGHTS_OR_SOURCE_RGB.json'
MANIFEST=P/'query_visibility_public_source_target_definition_audit_20261024_v179/source_train_goal_alignment_a/SAM_SMOKE_INPUTS128_METADATA_ONLY.jsonl'
OUT=V/'frozen_sam_source_predictions_v1'
FULL_MANIFEST=V/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl'
EXTENSION=V/'ROOT_ALL_TRAIN_EXTENSION_BEFORE_ANY_SAM_SOURCE_RGB_OR_SCORES.json'
CHECKPOINT=V/'checkpoints/sam_vit_b_01ec64.pth'
CHECKPOINT_SHA='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
RUNTIME={'python':'3.10.20','torch':'2.1.1+cu118','torchvision':'0.16.1+cu118','numpy':'1.26.4','Pillow':'12.2.0'}
PROTO=D/'PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json';PIN=D/'OFFICIAL_SOURCE_AND_IMPORT_PROBE.json';TESTS=D/'GENERATED_MEANINGFUL_CHECKS.json'

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def put(p,x,exclusive=True):
 with Path(p).open('x' if exclusive else 'w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(VENDOR),*args],text=True).strip()

def import_runtime():
 import torch,torchvision,PIL,inspect
 sys.path.insert(0,str(VENDOR))
 import segment_anything
 from segment_anything import SamPredictor,sam_model_registry
 actual={'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'Pillow':PIL.__version__}
 assert actual==RUNTIME,(actual,RUNTIME)
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
 assert Path(segment_anything.__file__).resolve()==VENDOR/'segment_anything/__init__.py'
 assert 'weights_only' in inspect.signature(torch.load).parameters
 return torch,SamPredictor,sam_model_registry,actual

def prompt_box(q8):
 try:a=np.asarray(q8,dtype=np.float64)
 except (TypeError,ValueError):return None,'Q8_MISSING_OR_NONNUMERIC'
 if a.shape!=(8,2) or not np.isfinite(a).all():return None,'Q8_MISSING_OR_NONFINITE'
 b=np.array([a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()],dtype=np.float64)
 if not(b[2]>b[0] and b[3]>b[1]):return None,'Q8_DEGENERATE_BOX'
 return b,None

def raw96_grid(q,L):
 q=np.asarray(q,dtype=np.float64)
 if q.shape!=(2,) or not np.isfinite(q).all() or L is None or not math.isfinite(L) or L<=0:return None
 offsets=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
 return np.stack(np.meshgrid(q[0]+offsets,q[1]+offsets),axis=-1)

def numeric_meta(rows):
 q=[];length=[]
 for r in rows:
  q.append([np.nan,np.nan] if r['original_query_xy'] is None else [np.nan if x is None else x for x in r['original_query_xy']])
  length.append(np.nan if r['original_L'] is None else r['original_L'])
 return {'original_query_xy':np.asarray(q,dtype=np.float64),'original_L':np.asarray(length,dtype=np.float64),'native_slot':np.asarray([r['native_slot'] for r in rows],dtype=np.int8),'original_global_index':np.asarray([r['original_global_index'] for r in rows],dtype=np.int64),'original_occurrence_id':np.asarray([r['original_occurrence_id'] for r in rows],dtype=np.str_),'original_query_available':np.asarray([r['original_input_flags']['query_available'] for r in rows],dtype=np.bool_)}

def grouped(rows,expected_rows,expected_scenes):
 assert len(rows)==expected_rows==8*expected_scenes
 assert len({r['original_global_index'] for r in rows})==expected_rows
 assert len({r['original_occurrence_id'] for r in rows})==expected_rows
 groups={}
 for r in rows:
  assert r['split']=='TRAIN' and r['training_admission'] is False
  assert isinstance(r['native_slot'],int) and 0<=r['native_slot']<8
  groups.setdefault(r['scene_index'],[]).append(r)
 assert len(groups)==expected_scenes
 for scene,rr in groups.items():
  rr.sort(key=lambda r:r['native_slot']);assert [r['native_slot'] for r in rr]==list(range(8))
  for r in rr:
   assert r['scene_index']==scene and r['original_global_index']==scene*8+r['native_slot']
   for k in ['base_index','variant','original_initial_points8','original_q8_context_box_xyxy','raw_RGB_PNG','decoded_RGB_sha256_inherited','raw_hw']:
    assert r[k]==rr[0][k],(scene,k)
   assert r['raw_hw']==[512,512]
   qs=r['original_initial_points8']
   if qs is not None:
    assert len(qs)==8
    assert r['original_query_xy']==qs[r['native_slot']]
  box,reason=prompt_box(rr[0]['original_initial_points8'])
  if box is not None:assert np.array_equal(box,np.asarray(rr[0]['original_q8_context_box_xyxy'],dtype=np.float64))
 return groups

def predicted_status(original_available,prompt_available,runtime_available):
 if not original_available:return False,'ORIGINAL_INPUT_U'
 if not prompt_available:return False,'PROMPT_UNAVAILABLE'
 if not runtime_available:return False,'RUNTIME_UNAVAILABLE'
 return True,None

def invented_checks():
 from segment_anything.utils.transforms import ResizeLongestSide
 checks=[]
 def ok(name,value):assert bool(value),name;checks.append({'name':name,'PASS':True})
 q=np.array([[7,21],[-2,15],[9,-1],[4,30],[7,21],[0,2],[9,25],[1,5]],dtype=np.float64)
 b,e=prompt_box(q);ok('unsorted_repeated_q8_exact_minmax_XYXY',e is None and np.array_equal(b,[-2,-1,9,30]))
 ok('q8_input_unchanged',np.array_equal(q,[[7,21],[-2,15],[9,-1],[4,30],[7,21],[0,2],[9,25],[1,5]]))
 ok('no_box_expansion_clipping_or_recenter',b[0]==-2 and b[1]==-1)
 for name,x in [('missing_q8',None),('partial_null_q8',[[None,None]]*8),('nan_q8',np.full((8,2),np.nan)),('wrong_count',q[:7]),('flatX',np.zeros((8,2)))]:ok(name+'_prompt_U',prompt_box(x)[0] is None)
 transformed=ResizeLongestSide(1024).apply_boxes(b,(512,512));ok('official_box_transform_exact_512_to1024',np.array_equal(transformed,b[None]*2))
 states=[predicted_status(a,b,c) for a in [False,True] for b in [False,True] for c in [False,True]];ok('all8_availability_combinations_only_allTrue_available',sum(v[0] for v in states)==1)
 ok('original_U_precedes_prompt_runtime_U',predicted_status(False,False,False)==(False,'ORIGINAL_INPUT_U'))
 ok('prompt_U_precedes_runtime_U',predicted_status(True,False,False)==(False,'PROMPT_UNAVAILABLE'))
 grid=raw96_grid([23.,41.],192.);ok('96pixel_centers_not_linspace_endpoints',np.array_equal(grid[0,0],[-72.,-54.]) and np.array_equal(grid[-1,-1],[118.,136.]))
 ok('central4_straddles_exact_unchanged_query',np.array_equal(grid[47,47],[22.,40.]) and np.array_equal(grid[48,48],[24.,42.]))
 ok('central32_is_exact_32_64_slice',grid[32:64,32:64].shape==(32,32,2) and np.array_equal(grid[32,32],[-8.,10.]))
 ok('null_qL_no_invented_grid',raw96_grid([np.nan,1.],192.) is None and raw96_grid([1.,2.],None) is None)
 toy=[{'original_query_xy':None,'original_L':None,'native_slot':7,'original_global_index':9,'original_occurrence_id':'invented7','original_input_flags':{'query_available':False}},{'original_query_xy':[1.,2.],'original_L':3.,'native_slot':2,'original_global_index':8,'original_occurrence_id':'invented2','original_input_flags':{'query_available':True}}]
 m=numeric_meta(toy);ok('null_metadata_nan_only_explicit_originalU_kept',np.isnan(m['original_query_xy'][0]).all() and np.isnan(m['original_L'][0]) and not m['original_query_available'][0]);ok('native_slot_not_inferred_from_row_order',m['native_slot'].tolist()==[7,2])
 return checks

def prepare():
 assert not OUT.exists(),'Existing actual output is never read or overwritten by prepare'
 assert git('rev-parse','HEAD')==COMMIT and git('remote','get-url','origin')=='https://github.com/facebookresearch/segment-anything.git'
 assert git('status','--porcelain','--untracked-files=no')==''
 torch,_,_,actual=import_runtime();assert not torch.cuda.is_initialized()
 rows=[json.loads(s) for s in MANIFEST.read_text().splitlines()];g=grouped(rows,128,16)
 assert set(r['base_index'] for r in rows)=={0,1,16,17} and set(r['variant'] for r in rows)=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
 full_rows=[json.loads(s) for s in FULL_MANIFEST.read_text().splitlines()];full_groups=grouped(full_rows,4096,512)
 assert sha(FULL_MANIFEST)=='86e016c7a528f0b631609c0088081e8033a715b35e971c4c28972cd683b44704' and sha(EXTENSION)=='511741e29620f0044e2a1d745183d773de33d07dcc8dd8800176711283b5e0c1'
 assert sum(not r['original_input_flags']['query_available'] for r in full_rows)==80
 sourcefiles=sorted((VENDOR/'segment_anything').rglob('*.py'))+[VENDOR/'README.md',VENDOR/'LICENSE']
 pin={'official_origin':git('remote','get-url','origin'),'exact_commit':COMMIT,'tracked_worktree_clean':True,'source_files':[bind(p) for p in sourcefiles],'runtime_import_probe':actual,'SAM_module_path':str(VENDOR/'segment_anything/__init__.py'),'model_instances_created':0,'checkpoint_image_feature_GT_bodies_read':0,'actual_forward_calls':0,'CUDA_initialized':torch.cuda.is_initialized(),'environment_modified_or_pip_calls':0}
 put(PIN,pin)
 checks=invented_checks();put(TESTS,{'completed':True,'PASS':True,'checks':checks,'count':len(checks),'invented_geometry_and_input_metadata_only':True,'actual_metadata_validation':{'rows':128,'scenes':16,'inputU':sum(not r['original_input_flags']['query_available'] for r in rows),'all16_boxes_equal_exact_original_q8_minmax':True,'full_TRAIN_metadata_rows':4096,'full_TRAIN_scenes':512,'full_TRAIN_inputU':80,'full_TRAIN_null_q8_scenes':sum(prompt_box(rr[0]['original_initial_points8'])[0] is None for rr in full_groups.values())},'actual_weights_RGB_forward_calls':0})
 protocol={'schema':'V180_FROZEN_SAM1_SOURCE_RUN_PROTOCOL_v1','scope':bind(SCOPE),'code':bind(Path(__file__)),'source_pin':bind(PIN),'checks':bind(TESTS),'default_manifest':bind(MANIFEST),'full_TRAIN_manifest':bind(FULL_MANIFEST),'extension_scope':bind(EXTENSION),'default_expected_rows':128,'default_expected_scenes':16,'default_output_dir':str(OUT),'model':'Official SAM1 vit_b only, build(checkpoint=None), torch.load(map_location=cpu,weights_only=True), load_state_dict(strict=True).','checkpoint':{'path':str(CHECKPOINT),'sha256':CHECKPOINT_SHA,'expected_bytes':375042383,'inherited_only_not_read_during_preparation':True},'runtime':actual,'model_input_whitelist':['RGB decoded uint8[512,512,3] from original raw_RGB_PNG only','one box exact float64 minmax of original_initial_points8, no clipping/expansion; official predictor casts transformed box to float32'],'prompt_no_GT_or_teacher_fields':True,'API':{'set_image':'image_format=RGB','predict':{'point_coords':None,'point_labels':None,'box':'original q8 XYXY','mask_input':None,'multimask_output':False,'return_logits':True}},'state':'FP32 CUDA eval,requires_gradFalse,inference_mode,TF32False,deterministic_algorithmsTrue,cudnn_benchmarkFalse,threads1; full state hash before/after, checkpoint SHA before/after.','scene_order':'First appearance in immutable manifest, native rows sorted0..7 only;16 unique set_image+predict attempts then one independent first-scene repeat; no quality-based masks/selection or retries.','output_schema':{'per_scene_NPZ':'SCENE_000004_OUTPUTS.npz: image_logits f32[512,512],lowres_logits f32[256,256],predicted_quality f32[1],mask_logits_gt0 bool[512,512],box_xyxy f64[4],original_query_xy f64[8,2],original_L f64[8],native_slot i8[8],original_global_index i64[8],original_occurrence_id str[8],original_query_available bool[8],prediction_available bool[8]','COMPLETE':'completed/PASS,scenes exact count,each scene outputs binding/null+prompt/runtime availability/reasons+original_input_metadata8;first_scene_repeat,state/input/checkpoint hashes','rows':'ALL128_INPUTS_AND_PREDICTION_STATUS.jsonl (generic ALL{N}): all original128 metadata+separate original/prompt/runtime/finalavailability; null q/L stay JSONnull,NPZ numericmeta NaN with original metadata preserved.'},'U_failure_policy':'OriginalinputU first; any missing/nonfinite/degenerate q8 retains8promptU. Runtime/RGB verification failure retains all scene/row statuses, outputs null; no fakezero masks/no retry. Global failure retains all records and fails terminal.','overwrite_policy':'Entry-bound output path under V180 must not exist; mkdir(exist_ok=False).','generic_expansion':'Optional --manifest/--expected-rows/--expected-scenes/--output-dir must exactly match separately approved entry. SourceTRAIN only. Full4096 expansion only after root review, decided on runtime/integrity not SAM scores.','evaluation_separation':'No source GT/part/depth or labels model input; future saved-output evaluation has separate root entry after prediction seal. A class-agnostic proposal is not selected-instance/native E/S/corner proof. Nonmask does not mean external.','training_admission':False,'actual_forward_completed_during_prepare':False}
 put(PROTO,protocol)
 template={'approved':False,'source_TRAIN_only':True,'code':bind(Path(__file__)),'protocol':bind(PROTO),'source_pin':bind(PIN),'generated_checks':bind(TESTS),'scope':bind(SCOPE),'extension_scope':bind(EXTENSION),'manifest':bind(MANIFEST),'expected_rows':128,'expected_scenes':16,'output_dir':str(OUT),'checkpoint':{'path':str(CHECKPOINT),'sha256':CHECKPOINT_SHA},'no_model_prompt_threshold_or_data_replacement_from_outcomes':True}
 put(D/'ROOT_ACTUAL_RUN_ENTRY_TEMPLATE.json',template)
 put(D/'PREPARATION_COMPLETE.json',{'completed':True,'PASS':True,'prepared_only':True,'code':bind(Path(__file__)),'protocol':bind(PROTO),'source_pin':bind(PIN),'checks':bind(TESTS),'entry_template':bind(D/'ROOT_ACTUAL_RUN_ENTRY_TEMPLATE.json'),'model_RGB_weight_feature_bodies_or_forward':0,'environment_modifications':0,'training_admission':False})
 print(json.dumps({'prepared_only':True,'code':bind(Path(__file__)),'protocol':bind(PROTO),'checks':bind(TESTS),'source_commit':COMMIT}))

def state_sha(model):
 h=hashlib.sha256()
 for name,t in sorted(model.state_dict().items()):
  a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def load_RGB(row):
 p=Path(row['raw_RGB_PNG']['path']).resolve();allowed=P/'query_visibility_registered_native_diversity_20261023_v129/actual_corpus_v1/bases'
 assert p.is_relative_to(allowed) and p.name==row['variant']+'_RGB.png'
 check(row['raw_RGB_PNG'])
 with Image.open(p) as im:assert im.mode=='RGB' and im.size==(512,512);rgb=np.array(im,dtype=np.uint8,copy=True)
 assert rgb.shape==(512,512,3) and arrsha(rgb)==row['decoded_RGB_sha256_inherited']
 return np.ascontiguousarray(rgb)

def infer(predictor,row,box,torch,counters):
 rgb=load_RGB(row)
 with torch.inference_mode():
  counters['set_image_calls']+=1
  predictor.set_image(rgb,image_format='RGB')
  counters['predict_calls']+=1
  a,q,l=predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True)
 assert a.shape==(1,512,512) and q.shape==(1,) and l.shape==(1,256,256)
 assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in [a,q,l])
 return {'image_logits':a[0].copy(),'lowres_logits':l[0].copy(),'predicted_quality':q.copy(),'mask_logits_gt0':(a[0]>0)}

def save_npz(p,arrays):
 with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)
 with np.load(p,allow_pickle=False) as f:
  assert set(f.files)==set(arrays)
  for k,v in arrays.items():
   assert f[k].dtype==v.dtype and f[k].shape==v.shape
   assert (np.array_equal(f[k],v) if v.dtype.kind=='U' else np.array_equal(f[k],v,equal_nan=True))
 return bind(p)

def run(entry_path,args):
 entry=read(entry_path);assert entry['approved'] is True and entry['source_TRAIN_only'] is True
 for k in ['code','protocol','source_pin','generated_checks','scope','manifest']:check(entry[k])
 assert entry['code']==bind(Path(__file__)) and entry['protocol']==bind(PROTO) and entry['scope']==bind(SCOPE)
 protocol=read(PROTO);pin=read(entry['source_pin']['path']);assert pin['exact_commit']==git('rev-parse','HEAD')==COMMIT
 assert git('status','--porcelain','--untracked-files=no')==''
 for b in pin['source_files']:check(b)
 manifest=Path(args.manifest or entry['manifest']['path']).resolve();n=args.expected_rows or entry['expected_rows'];ns=args.expected_scenes or entry['expected_scenes'];out=Path(args.output_dir or entry['output_dir']).resolve()
 assert (n,ns) in [(128,16),(4096,512)]
 if n==4096:
  check(entry['extension_scope']);assert entry['extension_scope']==bind(EXTENSION) and sha(EXTENSION)=='511741e29620f0044e2a1d745183d773de33d07dcc8dd8800176711283b5e0c1'
 assert manifest==Path(entry['manifest']['path']).resolve() and n==entry['expected_rows'] and ns==entry['expected_scenes'] and out==Path(entry['output_dir']).resolve()
 assert out.is_relative_to(V) and not out.exists();assert entry['checkpoint']=={'path':str(CHECKPOINT),'sha256':CHECKPOINT_SHA}
 rows=[json.loads(s) for s in manifest.read_text().splitlines()];groups=grouped(rows,n,ns)
 out.mkdir(exist_ok=False);start=time.monotonic();scenes=[]
 for scene,rr in groups.items():
  box,reason=prompt_box(rr[0]['original_initial_points8']);scenes.append({'scene_index':scene,'base_index':rr[0]['base_index'],'variant':rr[0]['variant'],'prompt_available':box is not None,'prompt_reason':reason,'box_xyxy':None if box is None else box.tolist(),'runtime_available':False,'runtime_reason':'NOT_ATTEMPTED','original_input_metadata8':rr,'outputs':None})
 put(out/'STARTED.json',{'UTC':utc(),'PID':os.getpid(),'entry':bind(entry_path),'code':bind(Path(__file__)),'protocol':bind(PROTO),'manifest':entry['manifest'],'rows':n,'scenes':ns})
 model=None;state_before=None;fatal=None;counters={'set_image_calls':0,'predict_calls':0};repeat={'attempted':False,'PASS':False,'reason':'NOT_ATTEMPTED'};torch=None
 try:
  check(entry['checkpoint']);assert CHECKPOINT.stat().st_size==375042383
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch,Predictor,registry,runtime=import_runtime()
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available()
  state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True);assert isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values())
  model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state
  model.requires_grad_(False);model.eval();model.float().to('cuda');assert all(not p.requires_grad and p.dtype==torch.float32 for p in model.parameters()) and not model.training and model.mask_threshold==0.0
  state_before=state_sha(model);predictor=Predictor(model)
  for idx,s in enumerate(scenes):
   if not s['prompt_available']:s['runtime_reason']='PROMPT_UNAVAILABLE'
   else:
    try:
     arrays=infer(predictor,s['original_input_metadata8'][0],np.array(s['box_xyxy'],dtype=np.float64),torch,counters)
     arrays.update(numeric_meta(s['original_input_metadata8']));arrays['box_xyxy']=np.array(s['box_xyxy'],dtype=np.float64);arrays['prediction_available']=arrays['original_query_available'].copy()
     s['outputs']=save_npz(out/f"SCENE_{s['scene_index']:06d}_OUTPUTS.npz",arrays);s['runtime_available']=True;s['runtime_reason']=None
     s['array_sha256']={k:arrsha(v) for k,v in arrays.items()}
    except Exception as e:s['runtime_reason']=type(e).__name__+': '+str(e);s['failure_traceback']=traceback.format_exc()
   progress={'scene_index':s['scene_index'],'completed':idx+1,'total':ns,'runtime_available':s['runtime_available'],'output_sha256':None if s['outputs'] is None else s['outputs']['sha256'],'elapsed_seconds':time.monotonic()-start};put(out/'PROGRESS.json',progress,False);print(json.dumps(progress),flush=True)
  first=scenes[0]
  if first['runtime_available']:
   ra=infer(predictor,first['original_input_metadata8'][0],np.array(first['box_xyxy'],dtype=np.float64),torch,counters)
   repeat={'attempted':True,'scene_index':first['scene_index'],'outputs':save_npz(out/'REPEAT_FIRST_SCENE_OUTPUTS.npz',ra),'PASS':all(arrsha(v)==first['array_sha256'][k] for k,v in ra.items()),'arrays_byte_equal_to_first':{k:arrsha(v)==first['array_sha256'][k] for k,v in ra.items()}}
  else:repeat={'attempted':False,'PASS':False,'reason':'FIRST_SCENE_UNAVAILABLE_NO_RETRY'}
 except BaseException as e:fatal={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 state_after=None
 try:state_after=state_sha(model) if model is not None and state_before is not None else None
 except Exception as e:
  if fatal is None:fatal={'type':type(e).__name__,'message':'Final state hash failure: '+str(e)}
 finalrows=[]
 for s in scenes:
  for r in s['original_input_metadata8']:
   available,reason=predicted_status(r['original_input_flags']['query_available'],s['prompt_available'],s['runtime_available'])
   finalrows.append({**r,'prompt_available':s['prompt_available'],'prompt_reason':s['prompt_reason'],'runtime_available':s['runtime_available'],'runtime_reason':s['runtime_reason'],'prediction_available':available,'prediction_U_reason':reason,'scene_output':s['outputs'],'new_labels':False,'training_admission':False})
 rp=out/f'ALL{n}_INPUTS_AND_PREDICTION_STATUS.jsonl'
 with rp.open('x') as f:
  for r in finalrows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 try:manifest_unchanged=sha(manifest)==entry['manifest']['sha256']
 except OSError:manifest_unchanged=False
 try:weights_unchanged=sha(CHECKPOINT)==CHECKPOINT_SHA
 except OSError:weights_unchanged=False
 passed=fatal is None and all(s['runtime_available'] or not s['prompt_available'] for s in scenes) and repeat['PASS'] and state_before==state_after and manifest_unchanged and weights_unchanged
 complete={'completed':True,'PASS':passed,'UTC':utc(),'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,'entry':bind(entry_path),'code':bind(Path(__file__)),'protocol':bind(PROTO),'source_pin':entry['source_pin'],'checkpoint':entry['checkpoint'],'source_commit':COMMIT,'rows':n,'scene_count':ns,'scenes':scenes,'all_rows':bind(rp),'first_scene_repeat':repeat,'model_state_before_sha256':state_before,'model_state_after_sha256':state_after,'model_state_byte_unchanged':state_before is not None and state_before==state_after,'manifest_byte_unchanged':manifest_unchanged,'checkpoint_file_byte_unchanged':weights_unchanged,'actual_forward_call_counters':counters,'actual_unique_scene_forward_calls':sum(s['runtime_available'] for s in scenes),'first_scene_repeat_forward':repeat['attempted'],'original_inputU':sum(not r['original_input_flags']['query_available'] for r in rows),'final_prediction_U':sum(not r['prediction_available'] for r in finalrows),'fatal':fatal,'no_auto_retry':True,'GT_part_mask_depth_teacher_model_inputs':False,'new_training_updates':0,'training_admission':False,'selected_model':None,'fully_solved':False}
 put(out/'COMPLETE.json',complete)
 if not passed:raise SystemExit('Frozen run failed; outputs/failures preserved, no automatic retry.')
 print(json.dumps({'completed':True,'PASS':True,'COMPLETE':bind(out/'COMPLETE.json')}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--entry');p.add_argument('--manifest');p.add_argument('--expected-rows',type=int);p.add_argument('--expected-scenes',type=int);p.add_argument('--output-dir');args=p.parse_args()
 assert args.prepare != bool(args.entry),'Choose prepare OR separately approved entry'
 if args.prepare:prepare()
 else:run(Path(args.entry).resolve(),args)
