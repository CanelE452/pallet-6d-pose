"""Evaluation-only synthetic VAL2048 SAM transport. No fit or source GT I/O.
Preparation: metadata and invented arrays only. Actual GPU requires separate root entry.
"""
from pathlib import Path
import argparse,ast,contextlib,hashlib,json,math,os,sys,subprocess,time,traceback
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
MAN=V/'synthetic_val_preparation/VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'
OLD=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
LEGACY=OLD/'runner_preparation/run_frozen_sam_source.py';OLD_PROTOCOL=OLD/'runner_preparation/PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json'
PROTO=H/'PROTOCOL_BEFORE_SYNTHETIC_VAL_SAM_RGB_OR_WEIGHTS.json';TESTS=H/'GENERATED_MEANINGFUL_CHECKS.json';OUT=V/'frozen_sam_synthetic_VAL2048_predictions_v1'
SCOPE_SHA='8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc';MAN_SHA='b460c33dc49ed9b7eb68bc5a6df919f516e1d55c0992fc22963f790ee9942944';LEGACY_SHA='b5d7f5591f83b231887270759de5059a27380ed58419acc3eef01ebdde0f8fa6'
NAMES={'prompt_box','numeric_meta','predicted_status','predict_warn_only_call','actual_cuda_cumsum_guard','state_sha','import_runtime','git','load_RGB','infer','save_npz'}
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','_v169','_v177','_v178','/private_evaluation/'))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d,exclusive=True):
 with Path(p).open('x' if exclusive else 'w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def helpers():
 assert sha(LEGACY)==LEGACY_SHA;p=read(OLD_PROTOCOL);pin=read(check(p['source_pin']))
 nodes=[n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in NAMES];assert {n.name for n in nodes}==NAMES
 ns=dict(np=np,Image=Image,Path=Path,P=P,sys=sys,math=math,hashlib=hashlib,subprocess=subprocess,bind=bind,check=check,arrsha=ash,VENDOR=OLD/'vendor/segment-anything',COMMIT=pin['exact_commit'],RUNTIME=p['runtime'])
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LEGACY)+'::EXACT_V180_FUNCTIONS_ONLY','exec'),ns);return ns,p,pin

def scene_rows_ok(rr,scene):
 assert len(rr)==8 and [r['native_slot'] for r in rr]==list(range(8))
 for r in rr:
  assert r['split']=='VALIDATION' and r['evaluation_only'] is True and r['training_admission'] is False
  assert r['scene_index']==scene and r['original_global_index']==scene*8+r['native_slot'] and r['raw_hw']==[512,512]
  for key in ('base_index','variant','original_initial_points8','original_q8_context_box_xyxy','raw_RGB_PNG','full_RGB_packet','decoded_RGB_sha256_inherited'):assert r[key]==rr[0][key]
  if r['original_initial_points8'] is not None:assert r['original_query_xy']==r['original_initial_points8'][r['native_slot']]

def groups(rows,helper):
 assert len(rows)==2048 and len({r['original_occurrence_id'] for r in rows})==2048
 assert [r['original_global_index'] for r in rows]==list(range(4096,6144))
 assert sum(not r['original_input_flags']['query_available'] for r in rows)==8
 out=[]
 for offset,scene in enumerate(range(512,768)):
  rr=rows[offset*8:(offset+1)*8];scene_rows_ok(rr,scene)
  box,reason=helper['prompt_box'](rr[0]['original_initial_points8'])
  if box is not None:assert box.tolist()==rr[0]['original_q8_context_box_xyxy']
  out.append(dict(group_id='SYNTH_VAL_SCENE_%06d'%scene,domain='SYNTHETIC_VALIDATION',scene_index=scene,base_index=rr[0]['base_index'],variant=rr[0]['variant'],split='VALIDATION',evaluation_only=True,prompt_available=box is not None,prompt_reason=reason,box_xyxy=None if box is None else box.tolist(),runtime_available=False,runtime_reason='NOT_ATTEMPTED',original_input_metadata8=rr,outputs=None))
 assert sum(not s['prompt_available'] for s in out)==1
 return out

def invented():
 helper,_,_=helpers();checks={}
 class Torch:
  strict=True;warn=False
  def are_deterministic_algorithms_enabled(self):return self.strict
  def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
  def use_deterministic_algorithms(self,v,warn_only=False):self.strict=v;self.warn=warn_only
  def inference_mode(self):return contextlib.nullcontext()
 class Predictor:
  def set_image(self,rgb,image_format):assert rgb.dtype==np.uint8 and rgb.shape==(512,512,3) and image_format=='RGB'
  def predict(self,**kw):
   assert kw['point_coords'] is kw['point_labels'] is kw['mask_input'] is None and kw['multimask_output'] is False and kw['return_logits'] is True
   return np.zeros((1,512,512),np.float32),np.ones(1,np.float32),np.zeros((1,256,256),np.float32)
 # Only the pure-helper namespace used by this invented call gets a fake loader.
 helper['load_RGB']=lambda row:np.zeros((512,512,3),np.uint8)
 counters={'set_image_calls':0,'predict_calls':0,'encoder_strict_state_checks':0,'prediction_warn_only_scopes':[]};t=Torch();box=np.array([-1.,2.,513.,600.],np.float64)
 a=helper['infer'](Predictor(),{},box,t,counters)
 checks['exact_old_infer_API_shape_and_strict_scope_exercised']=counters['set_image_calls']==counters['predict_calls']==1 and t.strict and not t.warn and a['image_logits'].shape==(512,512)
 checks['threshold0_strict_and_box_unclipped']=not a['mask_logits_gt0'].any() and box.tolist()==[-1.,2.,513.,600.]
 class Bad(Predictor):
  def predict(self,**kw):raise RuntimeError('invented only')
 try:helper['infer'](Bad(),{},box,t,counters)
 except RuntimeError:checks['predict_exception_propagates_and_restores_strict']=t.strict and not t.warn
 else:raise AssertionError('must propagate')
 checks['Ufirst_all8_boolean_combinations']=sum(helper['predicted_status'](x,y,z)[0] for x in (False,True) for y in (False,True) for z in (False,True))==1
 checks['missing_q8_preserved_promptU']=helper['prompt_box'](None)[0] is None and helper['predicted_status'](False,False,False)==(False,'ORIGINAL_INPUT_U')
 rr=[dict(original_query_xy=None if i==0 else [1.,2.],original_L=None if i==0 else 10.,native_slot=7-i,original_global_index=4096+i,original_occurrence_id=f'INVENTED:{i}',original_input_flags=dict(query_available=i!=0)) for i in range(8)]
 m=helper['numeric_meta'](rr);checks['null_numeric_metadata_NaN_and_native_input_order']=np.isnan(m['original_query_xy'][0]).all() and np.isnan(m['original_L'][0]) and m['native_slot'].tolist()==list(range(7,-1,-1)) and not m['original_query_available'][0]
 toy=[]
 for k in range(8):toy.append(dict(native_slot=k,split='VALIDATION',evaluation_only=True,training_admission=False,scene_index=512,original_global_index=4096+k,raw_hw=[512,512],base_index=128,variant='BEFORE',original_initial_points8=[[1.,2.]]*8,original_q8_context_box_xyxy=[1,2,1,2],raw_RGB_PNG={'path':'INVENTED'},full_RGB_packet={'path':'INVENTED'},decoded_RGB_sha256_inherited='INVENTED',original_query_xy=[1.,2.]))
 scene_rows_ok(toy,512);checks['VALIDATION_spelling_preserved_by_runtime_identity_guard']=all(r['split']=='VALIDATION' for r in toy)
 bad=[dict(r) for r in toy];bad[0]['split']='TRAIN'
 try:scene_rows_ok(bad,512)
 except AssertionError:checks['TRAIN_split_rejected_failclosed']=True
 else:raise AssertionError('TRAIN cannot be VAL')
 bad=[dict(r) for r in toy];bad[0]['original_global_index']=1
 try:scene_rows_ok(bad,512)
 except AssertionError:checks['misjoined_original_ID_rejected_failclosed']=True
 else:raise AssertionError('ID cannot drift')
 assert all(checks.values());return checks

def prepare():
 assert sha(SCOPE)==SCOPE_SHA and sha(MAN)==MAN_SHA and not PROTO.exists()
 helper,old,pin=helpers();rows=lines(MAN);scenes=groups(rows,helper);checks=invented()
 write(TESTS,dict(completed=True,PASS=True,checks=checks,count=len(checks),actual_source_RGB_NPZ_weight_body_header_hash_reads_or_forward=0,code=bind(__file__)))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),manifest=bind(MAN),legacy_code=bind(LEGACY),legacy_protocol=bind(OLD_PROTOCOL),legacy_AST_function_names=sorted(NAMES),source_pin=old['source_pin'],checkpoint=old['checkpoint'],runtime=old['runtime'],tests=bind(TESTS),output=str(OUT),counts=dict(rows2048=2048,scenes256=256,original_inputU8=8,prompt_missing_scenes1=1),
  inputs='Only root bound VAL2048 original input-only metadata; originalsplit VALIDATION,global4096..6143,scene512..767,native8,allq/L/inputflags preserved. All training_admission=false/evaluation_only=true.',
  exact_calculation='V180 exact official SAM1ViTB code/checkpoint/FP32/defaultthreshold0/q8 minmaxbox. Exact old PNGloader/infer/numericmeta/status/saver and strict-set_image/predict-warn_only-finallystrict/cumsumguard/state_sha AST reuse. No public/CF producer edits, no GT/model/threshold/data selection or fitting.',
  asset_loading='Original raw_RGB_PNG boundSHA, modeRGB512x512uint8 and decodedRGBSHA match inherited source transport. Original full_RGB_packet NPZ bindings remain metadata provenance and are never opened/hashed by this SAM runner. No GT/part/depth/labels/source feature reads. Checkpoint only loaded after separate rootentry.',
  output_schema='COMPLETE.scenes[256] and groups alias, each group_id SYNTH_VAL_SCENE_%06d/domainSYNTHETIC_VALIDATION/scene_index/prompt/runtime/box/original_input_metadata8/outputs binding|null. V180exact12 NPZ keys unchanged[512,512] f32logits; originalmetadata globalIDs and queryavailability intact. ALL2048_INPUTS_AND_PREDICTION_STATUS.jsonl keeps missing/U/reasons. Firstscene repeated once; no quality-based retry/mask choice.',
  restriction='Evaluation-only historically reused synthetic validation. No fit/draw/normalizer/checkpoint/architecture/arm/threshold selection, no private/current/publicheld/V112/PnP/qL changes. No new labels. GPU execution must not overlap root public/CF job; root schedules separate entry.',actual_prepare_asset_or_RGB_NPZ_body_hash_reads=0,actual_forwards=0,training_admission=False,selected_model=None)
 write(PROTO,p)
 entry=dict(approved=False,synthetic_VALIDATION_only=True,evaluation_only=True,code=p['code'],protocol=bind(PROTO),scope=p['scope'],manifest=p['manifest'],source_pin=p['source_pin'],tests=p['tests'],checkpoint=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']),output_dir=str(OUT),no_fit_or_model_prompt_threshold_data_selection=True)
 write(H/'ROOT_ACTUAL_VAL_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry)
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTO),checks=p['tests'],entry_template=bind(H/'ROOT_ACTUAL_VAL_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_RGB_weight_NPZ_body_reads_or_forwards=0))
 print(json.dumps(dict(prepared=True,code=p['code'],protocol=bind(PROTO),checks=p['tests'],preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

def run(entry_path):
 e=read(entry_path);assert e['approved'] is True and e['synthetic_VALIDATION_only'] is True and e['evaluation_only'] is True and e['no_fit_or_model_prompt_threshold_data_selection'] is True
 for key in ('code','protocol','scope','manifest','source_pin','tests'):check(e[key])
 assert e['code']==bind(__file__) and e['protocol']==bind(PROTO) and sha(SCOPE)==SCOPE_SHA and sha(MAN)==MAN_SHA
 p=read(PROTO)
 for key in ('code','scope','manifest','source_pin','tests'):assert p[key]==e[key]
 assert read(check(p['tests']))['PASS'];helper,old,pin=helpers();assert p['legacy_code']==bind(LEGACY) and p['legacy_protocol']==bind(OLD_PROTOCOL)
 assert helper['git']('rev-parse','HEAD')==pin['exact_commit'] and helper['git']('status','--porcelain','--untracked-files=no')==''
 for b in pin['source_files']:check(b)
 rows=lines(MAN);scenes=groups(rows,helper);out=Path(e['output_dir']).resolve();assert str(out)==p['output'] and out.is_relative_to(V) and not out.exists()
 ck=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']);assert e['checkpoint']==ck
 out.mkdir();start=time.monotonic();write(out/'STARTED.json',dict(entry=bind(entry_path),protocol=bind(PROTO),PID=os.getpid(),evaluation_only=True))
 counters={'set_image_calls':0,'predict_calls':0,'encoder_strict_state_checks':0,'prediction_warn_only_scopes':[]};model=None;state_before=None;state_after=None;fatal=None;guard=None;repeat=dict(attempted=False,PASS=False,reason='NOT_ATTEMPTED')
 try:
  check(ck);assert Path(ck['path']).stat().st_size==p['checkpoint']['expected_bytes'];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  torch,Predictor,registry,actual=helper['import_runtime']();assert actual==p['runtime'];torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available();guard=helper['actual_cuda_cumsum_guard'](torch);write(out/'CUDA_CUMSUM_PARITY_GUARD.json',guard);assert guard['PASS']
  state=torch.load(ck['path'],map_location='cpu',weights_only=True);assert isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values());model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state
  model.requires_grad_(False);model.eval();model.float().to('cuda');assert all(not x.requires_grad and x.dtype==torch.float32 for x in model.parameters()) and model.mask_threshold==0.
  state_before=helper['state_sha'](model);predictor=Predictor(model)
  for j,s in enumerate(scenes):
   if not s['prompt_available']:s['runtime_reason']='PROMPT_UNAVAILABLE'
   else:
    try:
     a=helper['infer'](predictor,s['original_input_metadata8'][0],np.asarray(s['box_xyxy'],np.float64),torch,counters);a.update(helper['numeric_meta'](s['original_input_metadata8']));a['box_xyxy']=np.asarray(s['box_xyxy'],np.float64);a['prediction_available']=a['original_query_available'].copy()
     s['outputs']=helper['save_npz'](out/f'SCENE_{s["scene_index"]:06d}_OUTPUTS.npz',a);s['array_sha256']={k:ash(x) for k,x in a.items()};s['runtime_available']=True;s['runtime_reason']=None
    except Exception as exc:s['runtime_reason']=type(exc).__name__+': '+str(exc);s['failure_traceback']=traceback.format_exc()
   if (j+1)%8==0 or j+1==256:
    pr=dict(completed_scenes=j+1,total=256,forward_calls=counters['set_image_calls'],elapsed_seconds=time.monotonic()-start);write(out/'PROGRESS.json',pr,False);print(json.dumps(pr),flush=True)
  first=scenes[0]
  if first['runtime_available']:
   a=helper['infer'](predictor,first['original_input_metadata8'][0],np.asarray(first['box_xyxy'],np.float64),torch,counters);comparison={k:ash(x)==first['array_sha256'][k] for k,x in a.items()};repeat=dict(attempted=True,scene_index=first['scene_index'],PASS=all(comparison.values()),array_byte_equal=comparison,outputs=helper['save_npz'](out/'REPEAT_FIRST_SCENE_OUTPUTS.npz',a))
  else:repeat=dict(attempted=False,PASS=False,reason='FIRST_SCENE_UNAVAILABLE_NO_RETRY')
 except BaseException as exc:fatal=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 try:state_after=helper['state_sha'](model) if model is not None and state_before is not None else None
 except Exception as exc:fatal=fatal or dict(type=type(exc).__name__,message='Final state SHA: '+str(exc))
 output_rows=[]
 for s in scenes:
  for r in s['original_input_metadata8']:
   available,reason=helper['predicted_status'](r['original_input_flags']['query_available'],s['prompt_available'],s['runtime_available']);row=dict(r);row.update(group_id=s['group_id'],domain=s['domain'],prompt_available=s['prompt_available'],runtime_available=s['runtime_available'],prediction_available=available,prediction_U_reason=reason,scene_output=s['outputs']);output_rows.append(row)
 rp=out/'ALL2048_INPUTS_AND_PREDICTION_STATUS.jsonl'
 with rp.open('x') as f:
  for r in output_rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 unchanged=True
 try:
  for b in (e['code'],e['manifest'],e['source_pin'],e['checkpoint'],e['scope']):check(b)
  for s in scenes:
   if s['runtime_available']:check(s['original_input_metadata8'][0]['raw_RGB_PNG'])
  for b in pin['source_files']:check(b)
 except Exception as exc:unchanged=False;fatal=fatal or dict(type=type(exc).__name__,message='Final immutable binding: '+str(exc))
 passed=fatal is None and all(s['runtime_available'] or not s['prompt_available'] for s in scenes) and repeat['PASS'] and state_before is not None and state_before==state_after and unchanged
 complete=dict(completed=True,PASS=passed,code=bind(__file__),protocol=bind(PROTO),scope=bind(SCOPE),entry=bind(entry_path),manifest=bind(MAN),source_pin=e['source_pin'],checkpoint=ck,source_commit=pin['exact_commit'],scenes=scenes,groups=scenes,scene_count=256,rows=2048,all_rows=bind(rp),first_scene_repeat=repeat,cuda_cumsum_guard=None if guard is None else bind(out/'CUDA_CUMSUM_PARITY_GUARD.json'),actual_forward_call_counters=counters,model_state_before_sha256=state_before,model_state_after_sha256=state_after,model_state_byte_unchanged=state_before is not None and state_before==state_after,all_input_source_checkpoint_hashes_unchanged=unchanged,original_inputU=8,final_prediction_U=sum(not r['prediction_available'] for r in output_rows),split='VALIDATION',evaluation_only=True,training_admission=False,no_model_threshold_data_selection_or_new_labels=True,new_training_updates=0,no_automatic_retry=True,selected_model=None,fully_solved=False,fatal=fatal,seconds=time.monotonic()-start)
 write(out/'COMPLETE.json',complete)
 if not passed:raise SystemExit('Frozen VAL cache failed; all outputs/statuses preserved, no automatic retry.')
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json'),scenes=256,rows=2048)),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare and a.entry is None:prepare()
 elif a.entry is not None and not a.prepare:run(a.entry)
 else:raise SystemExit('Use --prepare, or a separately approved evaluation-only --entry; no automatic GPU execution')
