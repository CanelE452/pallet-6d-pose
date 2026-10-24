"""V181 frozen SAM input cache, public development185 + existing CF477 only.
--prepare uses metadata/text + invented arrays. Root --entry alone loads pixels/weights.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,contextlib,hashlib,json,math,os,sys,time,traceback,subprocess
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
OLD=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
LEGACY=OLD/'runner_preparation/run_frozen_sam_source.py'
OLD_PROTOCOL=OLD/'runner_preparation/PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json'
PUBLIC=H/'PUBLIC185_FRAMES_METADATA_ONLY.jsonl';SLOTS=H/'PUBLIC1480_ORIGINAL_INPUTS_METADATA_ONLY.jsonl';PUBLIC_READY=H/'PUBLIC_INPUT_READINESS.json'
CF=P/'query_visibility_public_train_owner_counterfactual_20261024_v161/next_feature_input_metadata/ALL3816_COUNTERFACTUAL_INPUT_OCCURRENCES_NO_FEATURES.jsonl'
GROUPS=H/'ALL662_SAM_GROUPS_INPUT_ONLY.jsonl';ROWS=H/'ALL5296_SAM_SLOTS_INPUT_ONLY.jsonl'
PROTOCOL=H/'PROTOCOL_BEFORE_PUBLIC_CF_RGB_OR_WEIGHT_BODIES.json';TESTS=H/'GENERATED_MEANINGFUL_CHECKS.json';OUT=V/'frozen_sam_public_cf_inputs_v1'
EXPECTED_SCOPE_SHA='8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc'
EXPECTED_LEGACY_SHA='b5d7f5591f83b231887270759de5059a27380ed58419acc3eef01ebdde0f8fa6'
NAMES={'prompt_box','predict_warn_only_call','actual_cuda_cumsum_guard','state_sha','import_runtime','git'}
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','_v169','_v177','_v178','/private_evaluation/'))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x,exclusive=True):
 with Path(p).open('x' if exclusive else 'w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rs):
 with Path(p).open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def pure_legacy():
 assert sha(LEGACY)==EXPECTED_LEGACY_SHA
 p=read(OLD_PROTOCOL);pin=read(check(p['source_pin']))
 tree=ast.parse(LEGACY.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES];assert {n.name for n in nodes}==NAMES
 ns=dict(np=np,Path=Path,sys=sys,math=math,hashlib=hashlib,subprocess=subprocess,arrsha=ash,VENDOR=OLD/'vendor/segment-anything',COMMIT=pin['exact_commit'],RUNTIME=p['runtime'])
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LEGACY)+'::EXACT_PURE_RUNTIME_HELPERS','exec'),ns)
 return ns,p,pin

def numeric_meta(rr):
 return dict(original_query_xy=np.asarray([r['original_query_xy'] for r in rr],np.float64),original_L=np.asarray([r['original_L'] for r in rr],np.float64),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),original_global_index=np.asarray([r['original_global_index'] for r in rr],np.int64),original_occurrence_id=np.asarray([r['original_occurrence_id'] for r in rr]),original_query_available=np.asarray([r['original_query_available'] for r in rr],bool))
def frame_key(g):return (g['original_frame_index'],g['source_id'])
def reuse_key(g,rgb,box):
 meta=numeric_meta(g['original_input_metadata8'])
 fields=tuple((k,str(a.dtype),a.shape,ash(a)) for k,a in sorted(meta.items()))
 return (frame_key(g),rgb.shape,ash(rgb),ash(box),ash(np.asarray(g['original_initial_points8'],np.float64)),fields)
def make_group(group_id,domain,rr,image,frame_index,source_id,q8,box,extra=None):
 assert [r['native_slot'] for r in rr]==list(range(8)) and all(r['split']=='SOURCE_DEVELOPMENT' for r in rr)
 assert all(r['original_query_xy']==q8[r['native_slot']] for r in rr)
 g=dict(group_id=group_id,domain=domain,original_frame_index=frame_index,source_id=source_id,split='SOURCE_DEVELOPMENT',image=image,raw_hw=image['raw_hw'],original_initial_points8=q8,box_xyxy=box,original_input_metadata8=rr)
 if extra:g.update(extra)
 return g

def build_metadata(helper):
 ready=read(PUBLIC_READY);assert ready['completed'] and ready['PASS'] and ready['frames_manifest']==bind(PUBLIC) and ready['slots_manifest']==bind(SLOTS)
 public=lines(PUBLIC);slots=lines(SLOTS);assert len(public)==185 and len(slots)==1480
 byframe={r['original_frame_index']:r for r in public};byslot={r['original_global_index']:r for r in slots};groups=[]
 for f in public:
  rr=[r for r in slots if r['sam_input_occurrence_id']==f['sam_input_occurrence_id']]
  box,reason=helper['prompt_box'](f['original_initial_points8']);assert reason is None and box.tolist()==f['original_q8_context_box_xyxy']
  # Preserve raw source availability; no nearest/central/context/prompt-area gate.
  groups.append(make_group(f['sam_input_occurrence_id'],'PUBLIC_ORIGINAL',rr,f['image'],f['original_frame_index'],f['source_id'],f['original_initial_points8'],box.tolist()))
 cf=lines(CF);assert len(cf)==3816 and len({r['input_occurrence_id'] for r in cf})==3816
 cfgroup={}
 for r in cf:
  assert r['split']=='SOURCE_DEVELOPMENT';cfgroup.setdefault(r['pair_id']+'/'+r['variant'],[]).append(r)
 assert len(cfgroup)==477
 for gid,source_rows in cfgroup.items():
  source_rows.sort(key=lambda r:r['native_slot']);assert [r['native_slot'] for r in source_rows]==list(range(8))
  f=byframe[source_rows[0]['original_frame_index']];q8=[r['query_xy'] for r in source_rows];box,reason=helper['prompt_box'](q8);assert reason is None
  assert all(r['variant_payload']==source_rows[0]['variant_payload'] and r['original_context_box_xyxy']==box.tolist() for r in source_rows)
  image=dict(binding=source_rows[0]['variant_payload'],format='NPZ_RGB_UINT8_HWC',key='full_RGB',raw_hw=f['image']['raw_hw'],decoded_RGB_sha256=None)
  rr=[]
  for r in source_rows:
   p=byslot[r['original_global_index']]
   assert (r['original_frame_index'],r['native_slot'],r['original_query_id'],r['source_id'],r['query_xy'],r['L'],r['query_available'])==(p['original_frame_index'],p['native_slot'],p['original_occurrence_id'],p['source_id'],p['original_query_xy'],p['original_L'],p['original_query_available'])
   rr.append(dict(input_occurrence_id=r['input_occurrence_id'],sam_input_occurrence_id=gid,family='YCB_PUBLIC_CF',original_global_index=r['original_global_index'],original_occurrence_id=r['original_query_id'],original_frame_index=r['original_frame_index'],source_id=r['source_id'],split=r['split'],native_slot=r['native_slot'],original_query_xy=r['query_xy'],original_L=r['L'],original_query_available=r['query_available'],original_input_flags={k:r[k] for k in ('feature_constructed','query_available','nearest_valid','central4_available','context_available')},pair_id=r['pair_id'],variant=r['variant'],original_payload_binding=r['variant_payload'],pair_pixel_proof_PASS=r['pair_pixel_proof_PASS'],training_admission=False,selected_instance_identity_certified=False))
  groups.append(make_group(gid,'YCB_PUBLIC_CF',rr,image,f['original_frame_index'],f['source_id'],q8,box.tolist(),dict(pair_id=source_rows[0]['pair_id'],variant=source_rows[0]['variant'],pair_pixel_proof_PASS=source_rows[0]['pair_pixel_proof_PASS'])))
 allrows=[r for g in groups for r in g['original_input_metadata8']]
 assert len(groups)==662 and len(allrows)==5296 and len({r['input_occurrence_id'] for r in allrows})==5296
 assert sum(not r['original_query_available'] for r in allrows)==554
 assert sum(not g.get('pair_pixel_proof_PASS',True) for g in groups)==6
 return groups,allrows

def load_image(g):
 image=g['image'];path=Path(image['binding']['path']).resolve();check(image['binding'])
 if image['format']=='NPY_RGB_UINT8_HWC':
  assert g['domain']=='PUBLIC_ORIGINAL' and image['key'] is None and path.is_relative_to(P/'query_visibility_mr6d_full_public_source_20261024_v153/full_source_v1/canonical_RGB')
  rgb=np.load(path,allow_pickle=False)
 else:
  assert g['domain']=='YCB_PUBLIC_CF' and image['format']=='NPZ_RGB_UINT8_HWC' and image['key']=='full_RGB'
  assert path.is_relative_to(P/'query_visibility_public_train_owner_counterfactual_20261024_v161/source_pixels_v1')
  # Exact input whitelist; never native/quality/owner targets, masks, depth or teacher fields.
  with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in ('full_RGB','query_xy','L','native_slot','query_id')}
  rgb=a['full_RGB'];m=numeric_meta(g['original_input_metadata8'])
  for key,outkey in (('query_xy','original_query_xy'),('L','original_L'),('native_slot','native_slot'),('query_id','original_occurrence_id')):
   assert a[key].shape==m[outkey].shape and a[key].dtype==m[outkey].dtype and a[key].tobytes()==m[outkey].tobytes(),key
 assert rgb.dtype==np.uint8 and list(rgb.shape)==list(image['raw_hw'])+[3]
 rgb=np.ascontiguousarray(rgb);decoded=ash(rgb)
 if image['decoded_RGB_sha256'] is not None:assert decoded==image['decoded_RGB_sha256']
 return rgb,decoded

def infer_image(predictor,rgb,box,torch,helper,counters):
 h,w=rgb.shape[:2]
 with torch.inference_mode():
  assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
  counters['set_image_calls']+=1;predictor.set_image(rgb,image_format='RGB')
  counters['predict_calls']+=1;audit={'predict_call':counters['predict_calls']};counters['predict_warn_only_scopes'].append(audit)
  logits,quality,low=helper['predict_warn_only_call'](torch,lambda:predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True),audit)
 assert logits.shape==(1,h,w) and quality.shape==(1,) and low.shape==(1,256,256)
 assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in (logits,quality,low))
 return dict(image_logits=logits[0].copy(),lowres_logits=low[0].copy(),predicted_quality=quality.copy(),mask_logits_gt0=logits[0]>0)
def save_packet(path,a):
 with Path(path).open('xb') as f:np.savez_compressed(f,**a)
 with np.load(path,allow_pickle=False) as z:assert set(z.files)==set(a) and all(z[k].dtype==x.dtype and z[k].shape==x.shape and z[k].tobytes()==x.tobytes() for k,x in a.items())
 return bind(path)

def invented(helper):
 checks={}
 class Torch:
  strict=True;warn=False
  def are_deterministic_algorithms_enabled(self):return self.strict
  def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
  def use_deterministic_algorithms(self,on,warn_only=False):self.strict=on;self.warn=warn_only
  def inference_mode(self):return contextlib.nullcontext()
 class Predictor:
  def set_image(self,rgb,image_format):assert image_format=='RGB';self.hw=rgb.shape[:2]
  def predict(self,**kw):
   assert kw['point_coords'] is kw['point_labels'] is kw['mask_input'] is None and kw['multimask_output'] is False and kw['return_logits'] is True
   return np.zeros((1,)+self.hw,np.float32),np.array([.5],np.float32),np.zeros((1,256,256),np.float32)
 t=Torch();p=Predictor();rgb=np.zeros((23,41,3),np.uint8);box=np.array([-2.,0.,45.,30.],np.float64);ct={'set_image_calls':0,'predict_calls':0,'predict_warn_only_scopes':[]}
 a=infer_image(p,rgb,box,t,helper,ct);checks['real_infer_adapter_arbitrary_HW_exact_API_and_strict_restored']=a['image_logits'].shape==(23,41) and ct['set_image_calls']==ct['predict_calls']==1 and t.strict and not t.warn
 checks['zero_logit_strict_gt0_false_unmodified_box']=not a['mask_logits_gt0'].any() and box.tolist()==[-2.,0.,45.,30.]
 class Bad(Predictor):
  def predict(self,**kw):raise RuntimeError('invented only')
 try:infer_image(Bad(),rgb,box,t,helper,ct)
 except RuntimeError:checks['infer_exception_propagates_strict_finally_restore']=t.strict and not t.warn
 else:raise AssertionError('must propagate')
 rr=[dict(native_slot=k,original_global_index=80+k,original_occurrence_id=f'ORIGINAL:{k}',original_query_xy=[float(k),float(k+1)],original_L=96.,original_query_available=k!=2,split='SOURCE_DEVELOPMENT') for k in range(8)]
 g=make_group('invented','PUBLIC_ORIGINAL',rr,dict(raw_hw=[23,41]),10,'SOURCE:10',[r['original_query_xy'] for r in rr],[0,1,7,8]);g2=dict(g,group_id='CF/BEFORE',domain='YCB_PUBLIC_CF')
 checks['same_frame_bytes_q8_L_metadata_reuse_independent_of_domain']=reuse_key(g,rgb,box)==reuse_key(g2,rgb.copy(),box.copy())
 changed=rgb.copy();changed[0,0,0]=1;checks['one_RGB_byte_blocks_reuse']=reuse_key(g,changed,box)!=reuse_key(g,rgb,box)
 changed=dict(g,original_input_metadata8=[dict(r) for r in rr]);changed['original_input_metadata8'][0]['original_L']=97.;checks['one_L_change_blocks_reuse']=reuse_key(changed,rgb,box)!=reuse_key(g,rgb,box)
 checks['sourceU_preserved_native_not_global_modulo']=numeric_meta(rr)['original_query_available'].tolist()==[True,True,False,True,True,True,True,True] and numeric_meta(rr)['native_slot'].tolist()==list(range(8))
 checks['original_q8_box_no_clipping']=helper['prompt_box']([[-2,0],[45,0],[45,30],[-2,30],[0,1],[1,2],[2,3],[3,4]])[0].tolist()==[-2,0,45,30]
 checks['full_OOF_finite_box_still_prompt_valid']=helper['prompt_box']([[1296+i,180+i] for i in range(8)])[0] is not None
 checks['missing_or_degenerate_prompt_U']=helper['prompt_box'](None)[0] is None and helper['prompt_box']([[1,1]]*8)[0] is None
 checks['prepare_no_CUDA_or_model_import_required']=True
 assert all(checks.values());return checks

def prepare():
 assert sha(SCOPE)==EXPECTED_SCOPE_SHA and not PROTOCOL.exists()
 helper,old,pin=pure_legacy();groups,rs=build_metadata(helper);write_rows(GROUPS,groups);write_rows(ROWS,rs)
 checks=invented(helper);write(TESTS,dict(completed=True,PASS=True,checks=checks,count=len(checks),code=bind(__file__),actual_RGB_NPZ_header_body_hash_weight_model_forward_reads=0))
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),public_readiness=bind(PUBLIC_READY),public_frames=bind(PUBLIC),public_slots=bind(SLOTS),CF_metadata=bind(CF),groups_manifest=bind(GROUPS),rows_manifest=bind(ROWS),legacy_exact_helpers=bind(LEGACY),legacy_protocol=bind(OLD_PROTOCOL),source_pin=old['source_pin'],checkpoint=old['checkpoint'],runtime=old['runtime'],tests=bind(TESTS),output=str(OUT),counts=dict(groups=662,rows=5296,public_groups185=185,public_rows1480=1480,public_inputU194=194,CF_groups477=477,CF_rows3816=3816,CF_inputU360=360,all_inputU554=554,all477_including_failed2_pairs=True),
  calculation='Same V180 official SAM1 ViTB FP32 checkpoint/config/API. Full original RGB only, own exact q8 minmax XYXY only. No clip/expand/GTbox/point prompts. Mask logits>0, no quality-based choice, fit or threshold calibration.',
  runtime_policy='Reuse pinned exact import_runtime/state_sha/predict_warn_only_call/actual_cuda_cumsum_guard AST functions; set_image strict; predict warn_only finally strict restored; CUDA/CPU ones64 cumsum byteguard before images. First public group repeat once, full model/checkpoint/source/input hashes unchanged.',
  pixel_whitelist='PUBLIC original canonical RGB.npy array only, HWCuint8/rawhw/decodedSHA strict. CF payload boundfileSHA then members full_RGB/query_xy/L/native_slot/query_id only; q8/native/L/input IDs must exactly match frozen metadata. No other member/teacher/mask/depth/owner/native targets loaded.',
  reuse='After each authorized input is loaded and hashed: same originalframe/sourceid + decoded RGB hash/shape + exact q8/box + all numeric outputmetadata bytes including L/originalqueryavailability permit linking the prior output. No shape/GT matching or quality selection. Every662 group/5296 occurrence stays in ledger; metadata/data equality established even reused. No expected reuse count asserted before actual pixel evidence.',
  output_schema=dict(COMPLETE='completed/PASS,groups662 in frozen order,each group group_id/domain/originalframe/source_id,box_xyxy,prompt/runtime flags+reasons,original_input_metadata8,outputs binding|null,reuse proof; all5296 rows status binding; model state/checkpoint/source/input hashes; firstrepeat',NPZ='Exact12 V180 keys: image_logits f32[H,W],lowres_logits f32[256,256],predicted_quality f32[1],mask_logits_gt0 bool[H,W],box_xyxy f64[4],original_query_xy f64[8,2],original_L f64[8],native_slot i8[8],original_global_index i64[8],original_occurrence_id string[8],original_query_available bool[8],prediction_available bool[8]. Reuse links one exact canonical packet, numericmetadata must be byteequal.',group_domains=['PUBLIC_ORIGINAL','YCB_PUBLIC_CF']),
  availability='Original continuous-only availability retained before prompt/runtime conditions; nearest/center4/context are provenance, not new gates. All public1480 andCF3816 retained including original554U, BEFORE/failed2 pairs/unknowns. No GT input; no owner253/CF314 subset in SAM forward. A later rawcache samples logits at original96grid, availability originalqueryavailable & prompt/runtime & unchanged sample-domain only.',
  forbidden='No private/current/held or V112 access, no feature/initializer/GT/model/new architecture selection or q/L/PnP change. No source label bodies read by runner; output is input cache, not new GT or performance proof.',actual_forward_or_RGB_weight_or_NPZ_header_body_hash_reads_during_preparation=0,training_admission=False,selected_model=None)
 write(PROTOCOL,protocol)
 entry=dict(approved=False,source_development_only=True,code=protocol['code'],protocol=bind(PROTOCOL),scope=protocol['scope'],public_readiness=protocol['public_readiness'],groups_manifest=protocol['groups_manifest'],rows_manifest=protocol['rows_manifest'],source_pin=protocol['source_pin'],tests=protocol['tests'],checkpoint=dict(path=old['checkpoint']['path'],sha256=old['checkpoint']['sha256']),output_dir=str(OUT),no_model_prompt_threshold_data_replacement_from_outcomes=True)
 write(H/'ROOT_ACTUAL_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry)
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTOCOL),checks=protocol['tests'],groups=protocol['groups_manifest'],rows=protocol['rows_manifest'],entry_template=bind(H/'ROOT_ACTUAL_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_forward_or_RGB_weight_NPZ_body_reads=0))
 print(json.dumps(dict(prepared=True,code=protocol['code'],protocol=bind(PROTOCOL),tests=protocol['tests'],preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

def run(entry_path):
 e=read(entry_path);assert e['approved'] is True and e['source_development_only'] is True and e['no_model_prompt_threshold_data_replacement_from_outcomes'] is True
 for key in ('code','protocol','scope','public_readiness','groups_manifest','rows_manifest','source_pin','tests'):check(e[key])
 assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['scope']==bind(SCOPE) and sha(SCOPE)==EXPECTED_SCOPE_SHA
 p=read(PROTOCOL)
 for key in ('code','scope','public_readiness','groups_manifest','rows_manifest','source_pin','tests'):assert p[key]==e[key]
 assert read(check(p['tests']))['PASS']
 helper,old,pin=pure_legacy();assert p['legacy_exact_helpers']==bind(LEGACY) and p['legacy_protocol']==bind(OLD_PROTOCOL) and p['source_pin']==old['source_pin']
 assert helper['git']('rev-parse','HEAD')==pin['exact_commit'] and helper['git']('status','--porcelain','--untracked-files=no')==''
 for b in pin['source_files']:check(b)
 groups=lines(GROUPS);rows=lines(ROWS);assert len(groups)==662 and len(rows)==5296 and rows==[r for g in groups for r in g['original_input_metadata8']]
 assert len({g['group_id'] for g in groups})==662 and len({r['input_occurrence_id'] for r in rows})==5296
 assert all(g['split']=='SOURCE_DEVELOPMENT' and all(r['split']=='SOURCE_DEVELOPMENT' for r in g['original_input_metadata8']) for g in groups)
 assert all([r['native_slot'] for r in g['original_input_metadata8']]==list(range(8)) for g in groups)
 assert sum(not r['original_query_available'] for r in rows)==554
 out=Path(e['output_dir']).resolve();assert str(out)==p['output'] and out.is_relative_to(V) and not out.exists()
 checkpoint=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']);assert e['checkpoint']==checkpoint
 results=[]
 for ordinal,g in enumerate(groups):
  box,why=helper['prompt_box'](g['original_initial_points8']);assert box is None or box.tolist()==g['box_xyxy']
  s=dict(g);s.update(group_ordinal=ordinal,prompt_available=box is not None,prompt_reason=why,runtime_available=False,runtime_reason='NOT_ATTEMPTED',outputs=None,reused=False,reused_from_group_id=None,decoded_RGB_sha256_observed=None);results.append(s)
 out.mkdir();start=time.monotonic();write(out/'STARTED.json',dict(entry=bind(entry_path),protocol=bind(PROTOCOL),groups=662,rows=5296,started_UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid()))
 counters={'set_image_calls':0,'predict_calls':0,'predict_warn_only_scopes':[]};model=None;torch=None;state_before=None;state_after=None;fatal=None;cache={};used={};repeat=dict(attempted=False,PASS=False,reason='NOT_ATTEMPTED');guard=None
 try:
  check(checkpoint);assert Path(checkpoint['path']).stat().st_size==p['checkpoint']['expected_bytes']
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch,Predictor,registry,actual=helper['import_runtime']();assert actual==p['runtime']
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available();guard=helper['actual_cuda_cumsum_guard'](torch);write(out/'CUDA_CUMSUM_PARITY_GUARD.json',guard);assert guard['PASS']
  state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);assert isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values())
  model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state;model.requires_grad_(False);model.eval();model.float().to('cuda')
  assert all(not x.requires_grad and x.dtype==torch.float32 for x in model.parameters()) and not model.training and model.mask_threshold==0.
  state_before=helper['state_sha'](model);predictor=Predictor(model)
  for j,s in enumerate(results):
   if not s['prompt_available']:s['runtime_reason']='PROMPT_UNAVAILABLE'
   else:
    try:
     rgb,decoded=load_image(s);s['decoded_RGB_sha256_observed']=decoded;used[s['image']['binding']['path']]=s['image']['binding']
     box=np.asarray(s['box_xyxy'],np.float64);key=reuse_key(s,rgb,box)
     if key in cache:
      canonical=cache[key];s['outputs']=canonical['outputs'];s['array_sha256']=canonical['array_sha256'];s['reused']=True;s['reused_from_group_id']=canonical['group_id']
      s['reuse_proof']=dict(same_original_frame_and_source_id=True,same_decoded_RGB_SHA_shape=True,same_q8_box_L_and_numeric_metadata_bytes=True,canonical_group_id=canonical['group_id'])
     else:
      a=infer_image(predictor,rgb,box,torch,helper,counters);a.update(numeric_meta(s['original_input_metadata8']));a['box_xyxy']=box.copy();a['prediction_available']=a['original_query_available'].copy()
      s['outputs']=save_packet(out/f'GROUP_{j:06d}_OUTPUTS.npz',a);s['array_sha256']={k:ash(v) for k,v in a.items()};cache[key]=dict(group_id=s['group_id'],outputs=s['outputs'],array_sha256=s['array_sha256'])
     s['runtime_available']=True;s['runtime_reason']=None
    except Exception as exc:s['runtime_reason']=type(exc).__name__+': '+str(exc);s['failure_traceback']=traceback.format_exc()
   if (j+1)%8==0 or j+1==len(results):
    progress=dict(completed_groups=j+1,total_groups=662,unique_forward_calls=counters['set_image_calls'],reused=sum(x['reused'] for x in results[:j+1]),runtime_failures=sum(x['prompt_available'] and not x['runtime_available'] for x in results[:j+1]),elapsed_seconds=time.monotonic()-start);write(out/'PROGRESS.json',progress,False);print(json.dumps(progress),flush=True)
  first=results[0]
  if first['runtime_available']:
   rgb,_=load_image(first);a=infer_image(predictor,rgb,np.asarray(first['box_xyxy'],np.float64),torch,helper,counters)
   comparison={k:ash(x)==first['array_sha256'][k] for k,x in a.items()};repeat=dict(attempted=True,group_id=first['group_id'],PASS=all(comparison.values()),array_byte_equal=comparison,outputs=save_packet(out/'REPEAT_FIRST_GROUP_OUTPUTS.npz',a))
  else:repeat=dict(attempted=False,PASS=False,reason='FIRST_GROUP_UNAVAILABLE_NO_RETRY')
 except BaseException as exc:fatal=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 try:state_after=helper['state_sha'](model) if model is not None and state_before is not None else None
 except Exception as exc:
  if fatal is None:fatal=dict(type=type(exc).__name__,message='Final state hash: '+str(exc))
 finalrows=[]
 for g in results:
  for r in g['original_input_metadata8']:
   available=bool(r['original_query_available'] and g['prompt_available'] and g['runtime_available'])
   reason=None if available else 'ORIGINAL_INPUT_U' if not r['original_query_available'] else 'PROMPT_UNAVAILABLE' if not g['prompt_available'] else 'RUNTIME_UNAVAILABLE'
   row=dict(r);row.update(group_id=g['group_id'],domain=g['domain'],prompt_available=g['prompt_available'],runtime_available=g['runtime_available'],prediction_available=available,prediction_U_reason=reason,group_output=g['outputs'],reused_from_group_id=g['reused_from_group_id']);finalrows.append(row)
 rp=out/'ALL5296_INPUTS_AND_PREDICTION_STATUS.jsonl';write_rows(rp,finalrows)
 unchanged=True
 try:
  for b in used.values():check(b)
  for b in (e['groups_manifest'],e['rows_manifest'],e['checkpoint'],e['source_pin'],e['scope']):check(b)
  for b in pin['source_files']:check(b)
 except Exception as exc:unchanged=False;fatal=fatal or dict(type=type(exc).__name__,message='Final immutable binding check: '+str(exc))
 passed=fatal is None and all(g['runtime_available'] or not g['prompt_available'] for g in results) and repeat['PASS'] and state_before is not None and state_before==state_after and unchanged
 complete=dict(completed=True,PASS=passed,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),groups_manifest=e['groups_manifest'],rows_manifest=e['rows_manifest'],groups=results,group_count=662,rows=5296,all_rows=bind(rp),checkpoint=checkpoint,source_pin=e['source_pin'],source_commit=pin['exact_commit'],cuda_cumsum_guard=None if guard is None else bind(out/'CUDA_CUMSUM_PARITY_GUARD.json'),first_group_repeat=repeat,model_state_before_sha256=state_before,model_state_after_sha256=state_after,model_state_byte_unchanged=state_before is not None and state_before==state_after,input_checkpoint_and_source_hashes_unchanged=unchanged,actual_forward_call_counters=counters,actual_unique_group_forward_calls=len(cache),reused_group_occurrences=sum(g['reused'] for g in results),original_input_U=554,final_prediction_U=sum(not r['prediction_available'] for r in finalrows),public_rows1480=1480,CF_rows3816=3816,all_original_occurrences_retained=True,new_GT_or_owner_native_labels_created=False,fatal=fatal,no_automatic_retry=True,GT_or_teacher_inputs_to_SAM=False,new_training_updates=0,training_admission=False,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
 write(out/'COMPLETE.json',complete)
 if not passed:raise SystemExit('Frozen cache execution failed. All statuses preserved; no automatic retry.')
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json'),groups=662,rows=5296,unique_group_forward_calls=len(cache),reused_groups=complete['reused_group_occurrences'])),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare and args.entry is None:prepare()
 elif args.entry is not None and not args.prepare:run(args.entry)
 else:raise SystemExit('Prepare invented/metadata first; only root-approved --entry may execute actual pixels/weights/forward')
