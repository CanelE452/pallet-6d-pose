"""V182 fixed original-COVER-query diagnostic SAM inputs, sourceTRAIN only.
Prepare reads metadata/code + invented arrays; only root --entry reads RGB/weights/logits.
New diagnostic IDs are never aliases for original production-query metadata.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,contextlib,hashlib,json,math,os,subprocess,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
A=V/'common_query_preparation_a'
A_GROUPS=A/'COMMON512_SAM_GROUPS_INPUT_ONLY.jsonl';A_ROWS=A/'COMMON4096_SLOTS_INPUT_ONLY.jsonl'
OLD=P/'query_visibility_frozen_sam_source_baseline_20261024_v180'
LEGACY=OLD/'runner_preparation/run_frozen_sam_source.py'
OLD_PROTOCOL=OLD/'runner_preparation/PROTOCOL_BEFORE_WEIGHTS_OR_SOURCE_RGB_BODIES.json'
OLD_COMPLETE=OLD/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_COMMON_QUERY_RGB_WEIGHTS_OR_SAM_OUTPUTS.json'
GROUPS=H/'ALL512_COMMON_QUERY_SAM_GROUPS_INPUT_ONLY.jsonl';ROWS=H/'ALL4096_COMMON_QUERY_SAM_ROWS_INPUT_ONLY.jsonl'
TESTS=H/'GENERATED_MEANINGFUL_CHECKS.json';OUT=V/'frozen_common_query_sam_v1'
NAMES={'prompt_box','predict_warn_only_call','actual_cuda_cumsum_guard','state_sha','import_runtime','git'}
RAW=('image_logits','lowres_logits','predicted_quality','mask_logits_gt0')
LEGACY_SHA='b5d7f5591f83b231887270759de5059a27380ed58419acc3eef01ebdde0f8fa6'
SCOPE_SHA='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
OLD_COMPLETE_SHA='dc926768be71a62a94889c724c1fa0a9218c4ab2b9f2935ea42d4fd786a31c3b'
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','_v169','_v177','_v178','private','synthetic_val'))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x,exclusive=True):
 with Path(p).open('x' if exclusive else 'w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rs):
 with Path(p).open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def helpers():
 assert sha(LEGACY)==LEGACY_SHA
 old=read(OLD_PROTOCOL);pin=read(check(old['source_pin']))
 nodes=[n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in NAMES];assert {n.name for n in nodes}==NAMES
 ns=dict(np=np,Path=Path,sys=sys,math=math,hashlib=hashlib,subprocess=subprocess,arrsha=ash,VENDOR=OLD/'vendor/segment-anything',COMMIT=pin['exact_commit'],RUNTIME=old['runtime'])
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(LEGACY)+'::EXACT_SAM_HELPERS','exec'),ns)
 return ns,old,pin

def qarray(q):
 if q is None:return np.full((8,2),np.nan,np.float64)
 a=np.asarray([[np.nan,np.nan] if r is None else [np.nan if v is None else v for v in r] for r in q],np.float64);assert a.shape==(8,2);return a

def larray(x):
 if x is None:return np.full(8,np.nan,np.float64)
 a=np.asarray([np.nan if v is None else v for v in x],np.float64);assert a.shape==(8,);return a

def packet_meta(g):
 rr=g['diagnostic_input_metadata8'];assert len(rr)==8
 return dict(original_query_xy=qarray(g['common_query_xy8']),original_L=larray(g['common_L8']),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),original_global_index=np.asarray([r['diagnostic_global_index'] for r in rr],np.int64),original_occurrence_id=np.asarray([r['row_id'] for r in rr],np.str_),original_query_available=np.asarray(g['common_original_query_available8'],bool),prediction_available=np.asarray(g['common_original_query_available8'],bool))

def raw_validate(a):
 expected={'image_logits':(np.dtype('float32'),(512,512)),'lowres_logits':(np.dtype('float32'),(256,256)),'predicted_quality':(np.dtype('float32'),(1,)),'mask_logits_gt0':(np.dtype('bool'),(512,512))}
 assert set(a)==set(RAW)
 for k,(dt,sh) in expected.items():assert a[k].dtype==dt and a[k].shape==sh,k
 assert all(np.isfinite(a[k]).all() for k in RAW[:3]) and np.array_equal(a['mask_logits_gt0'],a['image_logits']>0)

def old_cover_matches(g,old_scene,rgb,box):
 """Fail closed; old production IDs checked against anchor, not diagnostic IDs."""
 assert g['variant']=='COVER' and old_scene['variant']=='COVER' and old_scene['base_index']==g['base_index']
 assert old_scene['prompt_available'] and old_scene['runtime_available'] and old_scene['outputs'] is not None
 rr=old_scene['original_input_metadata8'];m=packet_meta(g)
 assert [r['native_slot'] for r in rr]==m['native_slot'].tolist()
 assert same(qarray([r['original_query_xy'] for r in rr]),m['original_query_xy'])
 assert same(larray([r['original_L'] for r in rr]),m['original_L'])
 assert np.array_equal(np.asarray([r['original_global_index'] for r in rr],np.int64),np.asarray(g['anchor_original_global_index8'],np.int64))
 assert [r['original_occurrence_id'] for r in rr]==g['anchor_original_occurrence_id8']
 assert [r['original_input_flags'] for r in rr]==g['anchor_original_flags8']
 assert same(np.asarray([r['original_input_flags']['query_available'] for r in rr],bool),m['original_query_available'])
 assert same(np.asarray(old_scene['box_xyxy'],np.float64),box)
 assert all(r['raw_RGB_PNG']==g['source_RGB_PNG_binding'] and r['decoded_RGB_sha256_inherited']==ash(rgb) for r in rr)
 return True

def copy_old_raw(g,old_scene,rgb,box):
 assert old_cover_matches(g,old_scene,rgb,box)
 # Only four immutable model-output arrays are copied. Fresh diagnostic metadata is added separately.
 with np.load(check(old_scene['outputs']),allow_pickle=False) as z:
  a={k:z[k].copy() for k in RAW}
  m=packet_meta(g)
  for k,ek in (('original_query_xy','original_query_xy'),('original_L','original_L'),('native_slot','native_slot'),('original_query_available','original_query_available')):assert same(z[k],m[ek]),k
  assert same(z['original_global_index'],np.asarray(g['anchor_original_global_index8'],np.int64));assert z['original_occurrence_id'].tolist()==g['anchor_original_occurrence_id8'] and same(z['box_xyxy'],box)
  assert same(z['prediction_available'],m['original_query_available'])
 raw_validate(a);return a

def load_RGB(g):
 from PIL import Image
 path=check(g['source_RGB_PNG_binding']).resolve();allowed=P/'query_visibility_registered_native_diversity_20261023_v129/actual_corpus_v1/bases'
 assert path.is_relative_to(allowed) and path.name==g['variant']+'_RGB.png' and path.parent.name==f"V129_BASE_{g['base_index']:03d}"
 with Image.open(path) as im:assert im.mode=='RGB' and im.size==(512,512);a=np.array(im,dtype=np.uint8,copy=True)
 assert a.shape==(512,512,3) and ash(a)==g['decoded_RGB_sha256'];return np.ascontiguousarray(a)

def infer(predictor,rgb,box,torch,h,counters):
 with torch.inference_mode():
  assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
  counters['set_image_calls']+=1;predictor.set_image(rgb,image_format='RGB')
  audit={'predict_call':counters['predict_calls']+1};counters['predict_calls']+=1;counters['predict_warn_only_scopes'].append(audit)
  a,q,l=h['predict_warn_only_call'](torch,lambda:predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True),audit)
 assert a.shape==(1,512,512) and q.shape==(1,) and l.shape==(1,256,256)
 out=dict(image_logits=a[0].copy(),lowres_logits=l[0].copy(),predicted_quality=q.copy(),mask_logits_gt0=a[0]>0);raw_validate(out);return out

def save_packet(p,a):
 with Path(p).open('xb') as f:np.savez_compressed(f,**a)
 with np.load(p,allow_pickle=False) as z:assert set(z.files)==set(a) and all(same(z[k],v) for k,v in a.items())
 return bind(p)

# A-manifest adapter is added after the metadata-only schema is frozen. No body fallback.
def build_metadata(h):
 assert sha(A_GROUPS)=='b6a256150e33713448fed45818bab080b2b68af9cc1ceb98e7f90f062ccf93f9'
 assert sha(A_ROWS)=='17fdddce08d0da37609e850963ab2ec55cc3a37a1a836c7d852cd59e08aeb75a'
 ag=lines(A_GROUPS);ar=lines(A_ROWS);prior=read(OLD_COMPLETE)
 assert len(ag)==512 and len(ar)==4096 and prior['completed'] and prior['PASS'] and prior['model_state_byte_unchanged']
 assert prior['source_commit']=='dca509fe793f601edb92606367a655c15ac00fdf' and prior['checkpoint']['sha256']=='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
 scenes={s['scene_index']:s for s in prior['scenes']};covers={s['base_index']:s for s in prior['scenes'] if s['variant']=='COVER'}
 assert len(scenes)==512 and set(covers)==set(range(128))
 groups=[];rows=[]
 for j,a in enumerate(ag):
  rr=ar[j*8:j*8+8];assert a['original_input_metadata8']==rr
  assert a['split']=='TRAIN' and not a['fit_admission'] and not a['production_query_replay'] and a['raw_hw']==[512,512]
  assert a['diagnostic_group_index']==j and a['group_id']==f"V182_BASE{a['base_index']:03d}/{a['variant']}"
  assert a['variant'] in ('BEFORE','COVER','NEAR_SHAM','FAR_SHAM') and 0<=a['base_index']<128
  image=scenes[a['scene_index']];anchor=covers[a['base_index']];im=image['original_input_metadata8'];an=anchor['original_input_metadata8']
  assert (image['base_index'],image['variant'])==(a['base_index'],a['variant'])
  assert a['source_RGB_binding']==im[0]['full_RGB_packet'] and a['source_RGB_key']=='RGB_uint8' and a['decoded_RGB_SHA']==im[0]['decoded_RGB_sha256_inherited']
  assert a['anchor_original_global_indices8']==[r['original_global_index'] for r in an] and a['anchor_original_ids8']==[r['original_occurrence_id'] for r in an]
  assert a['anchor_original_flags8']==[r['original_input_flags'] for r in an]
  assert a['image_original_global_indices8']==[r['original_global_index'] for r in im] and a['image_original_ids8']==[r['original_occurrence_id'] for r in im]
  assert a['image_original_flags8']==[r['original_input_flags'] for r in im]
  assert a['image_original_query_xy8']==[r['original_query_xy'] for r in im] and a['image_original_L8']==[r['original_L'] for r in im]
  assert same(qarray(a['common_query_xy8']),qarray([r['original_query_xy'] for r in an])) and same(larray(a['common_L8']),larray([r['original_L'] for r in an]))
  assert a['common_original_query_available8']==[r['original_input_flags']['query_available'] for r in an]
  assert same(qarray(a['common_initial_points8']),qarray(a['common_query_xy8']))
  box,why=h['prompt_box'](a['common_query_xy8']);assert (None if box is None else box.tolist())==a['common_q8_context_box_xyxy']
  assert a['old_COVER_SAM_scene_binding']==anchor['outputs'] and a['old_COVER_SAM_scene_metadata']==anchor
  diagnostic=[]
  for k,r in enumerate(rr):
   assert r['native_slot']==k and r['split']=='TRAIN' and r['group_id']==a['group_id'] and r['row_id']==a['group_id']+f'/native{k}'
   assert r['original_global_index']==r['diagnostic_row_index']==j*8+k and r['original_occurrence_id']==r['row_id']
   assert not r['fit_admission'] and not r['production_query_replay'] and not r['GT_model_input']
   assert r['source_image_original_metadata']==im[k] and r['anchor_original_metadata']==an[k]
   assert r['query_xy']==r['original_query_xy']==r['anchor_original_query_xy']==a['common_query_xy8'][k]
   assert r['L']==r['original_L']==r['anchor_original_L']==a['common_L8'][k]
   assert r['original_query_available']==r['common_original_query_available']==a['common_original_query_available8'][k]
   assert r['original_input_flags']==r['anchor_original_flags']==an[k]['original_input_flags']
   assert r['image_original_query_xy']==im[k]['original_query_xy'] and r['image_original_L']==im[k]['original_L'] and r['image_original_flags']==im[k]['original_input_flags']
   assert r['anchor_original_global_index']==an[k]['original_global_index'] and r['source_image_original_global_index']==im[k]['original_global_index']
   row=dict(row_id=r['row_id'],original_occurrence_id=r['row_id'],diagnostic_global_index=r['diagnostic_row_index'],original_global_index=r['diagnostic_row_index'],group_id=r['group_id'],base_index=r['base_index'],scene_index=r['scene_index'],variant=r['variant'],split='TRAIN',native_slot=k,common_query_xy=r['query_xy'],common_L=r['L'],common_query_available=r['original_query_available'],original_query_xy=r['query_xy'],original_L=r['L'],original_query_available=r['original_query_available'],original_input_flags=r['original_input_flags'],anchor_original_global_index=r['anchor_original_global_index'],anchor_original_occurrence_id=an[k]['original_occurrence_id'],anchor_original_query_xy=r['anchor_original_query_xy'],anchor_original_L=r['anchor_original_L'],anchor_original_flags=r['anchor_original_flags'],image_original_global_index=r['source_image_original_global_index'],image_original_occurrence_id=im[k]['original_occurrence_id'],image_original_query_xy=r['image_original_query_xy'],image_original_L=r['image_original_L'],image_original_flags=r['image_original_flags'],training_admission=False,production_query_replay=False,packet_original_prefix_means='New V182 diagnostic input, not either original production origin')
   diagnostic.append(row);rows.append(row)
  g=dict(group_id=a['group_id'],diagnostic_group_index=j,base_index=a['base_index'],scene_index=a['scene_index'],variant=a['variant'],split='TRAIN',raw_hw=[512,512],source_RGB_PNG_binding=im[0]['raw_RGB_PNG'],source_RGB_buffer_binding=a['source_RGB_binding'],decoded_RGB_sha256=a['decoded_RGB_SHA'],common_query_xy8=a['common_query_xy8'],common_L8=a['common_L8'],common_original_query_available8=a['common_original_query_available8'],common_q8_context_box_xyxy=a['common_q8_context_box_xyxy'],anchor_scene_index=anchor['scene_index'],anchor_original_global_index8=a['anchor_original_global_indices8'],anchor_original_occurrence_id8=a['anchor_original_ids8'],anchor_original_flags8=a['anchor_original_flags8'],image_original_global_index8=a['image_original_global_indices8'],image_original_occurrence_id8=a['image_original_ids8'],image_original_query_xy8=a['image_original_query_xy8'],image_original_L8=a['image_original_L8'],image_original_flags8=a['image_original_flags8'],diagnostic_input_metadata8=diagnostic,original_input_metadata8=diagnostic,old_COVER_SAM_output_binding=anchor['outputs'],packet_original_prefix_means='New V182 diagnostic input only; both original production origins are separately preserved',training_admission=False,production_query_replay=False)
  assert packet_meta(g)['original_query_xy'].shape==(8,2)
  groups.append(g)
 assert len({g['group_id'] for g in groups})==512 and len({r['row_id'] for r in rows})==4096
 assert len({g['base_index'] for g in groups})==128 and all(sum(g['base_index']==b for g in groups)==4 for b in range(128))
 assert sum(not r['common_query_available'] for r in rows)==96
 assert sum(h['prompt_box'](g['common_query_xy8'])[0] is None for g in groups)==8
 return groups,rows

def invented(h):
 checks={}
 def ok(k,b):assert b,k;checks[k]=True
 q=[[10.+k,20.+2*k] for k in range(8)];box,why=h['prompt_box'](q)
 ok('exact_q8_minmax_no_clipping',why is None and box.tolist()==[10.,20.,17.,34.])
 ok('fully_outside_finite_box_stays_prompt_available',h['prompt_box']([[900.+k,1000.+k] for k in range(8)])[0] is not None)
 ok('null_or_partial_q8_has_no_fallback_prompt',h['prompt_box'](None)[0] is None and h['prompt_box']([[None,None]]*8)[0] is None)
 rr=[dict(native_slot=k,diagnostic_global_index=1000+k,row_id=f'NEW/native{k}') for k in range(8)]
 g=dict(group_id='NEW',base_index=0,variant='COVER',diagnostic_input_metadata8=rr,common_query_xy8=q,common_L8=[96.]*8,common_original_query_available8=[True,False]+[True]*6,anchor_original_global_index8=list(range(8)),anchor_original_occurrence_id8=[f'OLD/native{k}' for k in range(8)],anchor_original_flags8=[dict(query_available=k!=1) for k in range(8)],image_original_global_index8=list(range(8)),source_RGB_PNG_binding=dict(path='invented',sha256='invented'))
 rgb=np.zeros((512,512,3),np.uint8)
 old=dict(base_index=0,variant='COVER',prompt_available=True,runtime_available=True,outputs=dict(path='invented'),box_xyxy=box.tolist(),original_input_metadata8=[dict(native_slot=k,original_global_index=k,original_occurrence_id=f'OLD/native{k}',original_query_xy=q[k],original_L=96.,original_input_flags=dict(query_available=k!=1),raw_RGB_PNG=g['source_RGB_PNG_binding'],decoded_RGB_sha256_inherited=ash(rgb)) for k in range(8)])
 ok('COVER_raw_reuse_accepts_exact_anchor_despite_new_diagnostic_ID',old_cover_matches(g,old,rgb,box))
 for name,key,value in [('q_drift','common_query_xy8',[[11.,20.]]+q[1:]),('L_drift','common_L8',[97.]+[96.]*7),('availability_drift','common_original_query_available8',[True]*8),('anchor_ID_drift','anchor_original_occurrence_id8',['BAD']+g['anchor_original_occurrence_id8'][1:])]:
  try:old_cover_matches(dict(g,**{key:value}),old,rgb,box)
  except AssertionError:ok(name+'_blocks_COVER_reuse',True)
  else:raise AssertionError(name)
 changed=rgb.copy();changed[0,0,0]=1
 try:old_cover_matches(g,old,changed,box)
 except AssertionError:ok('one_RGB_byte_blocks_COVER_reuse',True)
 else:raise AssertionError('RGB')
 m=packet_meta(g);ok('fresh_diagnostic_metadata_never_old_packet_alias',m['original_occurrence_id'][0]=='NEW/native0' and m['original_global_index'][0]==1000 and g['anchor_original_global_index8'][0]==0)
 missing=dict(g,common_query_xy8=None,common_L8=[None]*8,common_original_query_available8=[False]*8);mm=packet_meta(missing)
 ok('null_metadata_serializes_nan_without_fake_coordinates',np.isnan(mm['original_query_xy']).all() and np.isnan(mm['original_L']).all() and not mm['original_query_available'].any())
 ok('sourceU_preserved_separate_from_prompt_validity',m['prediction_available'].tolist()==[True,False]+[True]*6)
 class Torch:
  strict=True;warn=False
  def are_deterministic_algorithms_enabled(self):return self.strict
  def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
  def use_deterministic_algorithms(self,on,warn_only=False):self.strict=on;self.warn=warn_only
  def inference_mode(self):return contextlib.nullcontext()
 class Predictor:
  def set_image(self,rgb,image_format):assert image_format=='RGB'
  def predict(self,**kw):
   assert kw['point_coords'] is kw['point_labels'] is kw['mask_input'] is None and kw['multimask_output'] is False and kw['return_logits'] is True
   assert same(kw['box'],box)
   return np.zeros((1,512,512),np.float32),np.array([.1],np.float32),np.zeros((1,256,256),np.float32)
 t=Torch();c=dict(set_image_calls=0,predict_calls=0,predict_warn_only_scopes=[]);raw=infer(Predictor(),rgb,box,t,h,c)
 ok('actual_infer_adapter_API_FP32_zero_logit_gt0_and_strict_restore',not raw['mask_logits_gt0'].any() and t.strict and not t.warn and c['set_image_calls']==c['predict_calls']==1)
 class Bad(Predictor):
  def predict(self,**kw):raise RuntimeError('invented')
 try:infer(Bad(),rgb,box,t,h,c)
 except RuntimeError:ok('predict_exception_finally_strict_restored',t.strict and not t.warn)
 else:raise AssertionError('exception')
 ok('four_raw_arrays_are_copies_before_fresh_metadata',set(raw)==set(RAW) and all(a.flags.owndata for a in raw.values()))
 import tempfile
 with tempfile.TemporaryDirectory(prefix='V182_INVENTED_ONLY_',dir=H) as td:
  td=Path(td);prior=dict(raw,box_xyxy=box.copy(),original_query_xy=qarray(q),original_L=larray([96.]*8),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.arange(8,dtype=np.int64),original_occurrence_id=np.asarray(g['anchor_original_occurrence_id8']),original_query_available=np.asarray(g['common_original_query_available8'],bool),prediction_available=np.asarray(g['common_original_query_available8'],bool))
  prior_binding=save_packet(td/'INVENTED_OLD_OUTPUT.npz',prior);toy_scene=dict(old,outputs=prior_binding)
  copied=copy_old_raw(g,toy_scene,rgb,box);fresh=dict(copied,**packet_meta(g),box_xyxy=box.copy());fresh_binding=save_packet(td/'INVENTED_NEW_OUTPUT.npz',fresh)
  ok('actual_copy_and_save_helpers_raw4_equal_fresh_IDs_and_U',all(same(copied[k],prior[k]) for k in RAW) and fresh['original_occurrence_id'].tolist()!=prior['original_occurrence_id'].tolist() and fresh['prediction_available'].tolist()==prior['prediction_available'].tolist())
  wrong=dict(prior,original_global_index=np.arange(8,dtype=np.int64)+1);wrong_binding=save_packet(td/'INVENTED_BAD_OLD_OUTPUT.npz',wrong)
  try:copy_old_raw(g,dict(old,outputs=wrong_binding),rgb,box)
  except AssertionError:ok('saved_old_numeric_metadata_drift_blocks_reuse_even_when_JSON_matches',True)
  else:raise AssertionError('old NPZ metadata')
  missing_fresh=dict(raw,**packet_meta(missing),box_xyxy=box.copy());save_packet(td/'INVENTED_NULL_META.npz',missing_fresh)
  ok('same_save_helper_roundtrips_nan_null_metadata_exactbytes',np.isnan(missing_fresh['original_query_xy']).all())
 return checks

def prepare():
 assert sha(SCOPE)==SCOPE_SHA and sha(OLD_COMPLETE)==OLD_COMPLETE_SHA and not PROTOCOL.exists()
 h,old,pin=helpers();groups,rows=build_metadata(h)
 write_rows(GROUPS,groups);write_rows(ROWS,rows);tests=invented(h)
 write(TESTS,dict(completed=True,PASS=True,count=len(tests),checks=tests,code=bind(__file__),actual_new_RGB_or_weight_or_SAM_output_body_reads_or_forward=0))
 counts=dict(groups=len(groups),rows=len(rows),common_input_U=sum(not r['common_query_available'] for r in rows),prompt_unavailable_groups=sum(h['prompt_box'](g['common_query_xy8'])[0] is None for g in groups),COVER_groups=128)
 protocol=dict(prepared_only=True,code=bind(__file__),scope=bind(SCOPE),A_groups=bind(A_GROUPS),A_rows=bind(A_ROWS),groups=bind(GROUPS),rows=bind(ROWS),old_complete=bind(OLD_COMPLETE),legacy_code=bind(LEGACY),legacy_protocol=bind(OLD_PROTOCOL),source_pin=old['source_pin'],checkpoint=old['checkpoint'],runtime=old['runtime'],tests=bind(TESTS),counts=counts,output_dir=str(OUT),model='Exact official SAM1 ViTB V180 config/state/import/API. Full original512RGB; one exact common original-COVERq8 minmax XYXY. No clip/expansion/GTprompt/quality choice.',runtime_policy='FP32 eval/no-grad/TF32false deterministic strict set_image; predict-only warn_only scope finally strictrestore; CUDA/CPU ones64cumsum byteguard beforeimages; first usable group independent repeat; fullstate/source/checkpoint/inputfile SHA unchanged.',reuse='COVER only after sourcePNG binding+decodedbytes/commonq8/L/native/anchororiginalID/flags/box exact match frozen old production COVER. Read/copy only4rawoutputarrays, verify saved old numericmeta, then write a fresh packet with new diagnosticIDs/commonq and both original originindices. Metadata never silently aliases old production packet. All nonCOVER promptavailable groups freshforward. Reuse mismatch is preserved failure, no fallback/retry.',output_schema=dict(raw4='image_logits f32[512,512],lowres_logits f32[256,256],predicted_quality f32[1],mask_logits_gt0 bool[512,512], thresholdstrict>0',metadata='Exact12 V180/V181 packet: raw4 plus box_xyxy f64[4],original_query_xy f64[8,2],original_L f64[8],native_slot i8[8],original_global_index i64[8],original_occurrence_id string[8],original_query_available bool[8],prediction_available bool[8]. All original-prefix fields here explicitly describe the NEW V182 diagnostic input (IDs0..4095 and V182 row IDs); BOTH original production origins separately retained in COMPLETE.groups metadata.',COMPLETE='groups in frozen512 order, each full A/input provenance+prompt/runtime/reasons/outputbinding|null/reuseproof, all4096statusrows; firstusable repeat/modelstate/source/input immutability'),U_policy='Original commonCOVER inputU first, then prompt/runtime availability; anchor missing and failures retained; nofakezero/logit imputation/no image replacement. All sameq diagnostic q/L/IDs/flags preserve origin. Original per-variant production input not changed.',training_admission=False,selected_model=None,fully_solved=False,actual_new_RGB_weight_SAM_output_body_reads_or_forward_during_preparation=0)
 write(PROTOCOL,protocol)
 entry=dict(approved=False,source_TRAIN_only=True,new_diagnostic_not_original_production_replay=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),A_groups=protocol['A_groups'],A_rows=protocol['A_rows'],groups=protocol['groups'],rows=protocol['rows'],old_complete=protocol['old_complete'],source_pin=protocol['source_pin'],tests=protocol['tests'],checkpoint=dict(path=old['checkpoint']['path'],sha256=old['checkpoint']['sha256']),output_dir=str(OUT),no_model_prompt_threshold_data_replacement_from_outcomes=True)
 write(H/'ROOT_ACTUAL_COMMON_QUERY_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry)
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=entry['code'],protocol=entry['protocol'],tests=entry['tests'],counts=counts,entry_template=bind(H/'ROOT_ACTUAL_COMMON_QUERY_SAM_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_new_RGB_weight_SAM_output_body_reads_or_forward=0))
 print(json.dumps(dict(prepared=True,code=entry['code'],protocol=entry['protocol'],preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

def run(entry_path):
 e=read(entry_path);assert e['approved'] is True and e['source_TRAIN_only'] is True and e['new_diagnostic_not_original_production_replay'] is True and e['no_model_prompt_threshold_data_replacement_from_outcomes'] is True
 for k in ('code','protocol','scope','A_groups','A_rows','groups','rows','old_complete','source_pin','tests'):check(e[k])
 assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and sha(SCOPE)==SCOPE_SHA and sha(OLD_COMPLETE)==OLD_COMPLETE_SHA
 p=read(PROTOCOL)
 for k in ('code','scope','A_groups','A_rows','groups','rows','old_complete','source_pin','tests'):assert e[k]==p[k]
 assert read(check(p['tests']))['PASS'];h,old,pin=helpers();assert p['legacy_code']==bind(LEGACY) and p['legacy_protocol']==bind(OLD_PROTOCOL) and p['source_pin']==old['source_pin']
 assert h['git']('rev-parse','HEAD')==pin['exact_commit'] and h['git']('status','--porcelain','--untracked-files=no')==''
 for b in pin['source_files']:check(b)
 groups,rows=build_metadata(h);assert groups==lines(GROUPS) and rows==lines(ROWS)
 prior=read(OLD_COMPLETE);prior_by={s['scene_index']:s for s in prior['scenes']}
 out=Path(e['output_dir']).resolve();assert str(out)==p['output_dir'] and out.is_relative_to(V) and not out.exists()
 checkpoint=dict(path=p['checkpoint']['path'],sha256=p['checkpoint']['sha256']);assert e['checkpoint']==checkpoint
 results=[]
 for g in groups:
  box,why=h['prompt_box'](g['common_query_xy8']);s=dict(g);s.update(box_xyxy=None if box is None else box.tolist(),prompt_available=box is not None,prompt_reason=why,runtime_available=False,runtime_reason='NOT_ATTEMPTED',RGB_verified=False,outputs=None,reused=False,reuse_proof=None);results.append(s)
 out.mkdir();start=time.monotonic();write(out/'STARTED.json',dict(PID=os.getpid(),UTC=datetime.now(timezone.utc).isoformat(),entry=bind(entry_path),protocol=bind(PROTOCOL),groups=512,rows=4096))
 c=dict(set_image_calls=0,predict_calls=0,predict_warn_only_scopes=[]);model=None;before=None;after=None;fatal=None;used={};guard=None;repeat=dict(attempted=False,PASS=False,reason='NOT_ATTEMPTED')
 try:
  check(checkpoint);assert Path(checkpoint['path']).stat().st_size==p['checkpoint']['expected_bytes'];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  torch,Predictor,registry,actual=h['import_runtime']();assert actual==p['runtime'];torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  assert torch.cuda.is_available();guard=h['actual_cuda_cumsum_guard'](torch);write(out/'CUDA_CUMSUM_PARITY_GUARD.json',guard);assert guard['PASS']
  state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);assert isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values())
  model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state;model.requires_grad_(False).eval().float().to('cuda');assert all(not x.requires_grad and x.dtype==torch.float32 for x in model.parameters()) and not model.training and model.mask_threshold==0.
  before=h['state_sha'](model);predictor=Predictor(model)
  for j,g in enumerate(results):
   try:
    rgb=load_RGB(g);used[g['source_RGB_PNG_binding']['path']]=g['source_RGB_PNG_binding'];g['decoded_RGB_sha256_observed']=ash(rgb);g['RGB_verified']=True
    if not g['prompt_available']:g['runtime_reason']='PROMPT_UNAVAILABLE'
    else:
     box=np.asarray(g['box_xyxy'],np.float64)
     if g['variant']=='COVER':
      old_scene=prior_by[g['anchor_scene_index']];a=copy_old_raw(g,old_scene,rgb,box);used[old_scene['outputs']['path']]=old_scene['outputs'];g['reused']=True
      g['reuse_proof']=dict(old_output=old_scene['outputs'],old_scene_index=old_scene['scene_index'],raw4_byte_copy_only=True,RGB_q8_box_L_native_anchor_ID_flags_exact=True,fresh_diagnostic_metadata=True,raw_array_SHA={k:ash(a[k]) for k in RAW})
     else:a=infer(predictor,rgb,box,torch,h,c)
     a.update(packet_meta(g));a['box_xyxy']=box.copy();g['outputs']=save_packet(out/f'GROUP_{j:06d}_OUTPUTS.npz',a);g['array_sha256']={k:ash(x) for k,x in a.items()};g['runtime_available']=True;g['runtime_reason']=None
   except Exception as exc:g['runtime_reason']=type(exc).__name__+': '+str(exc);g['failure_traceback']=traceback.format_exc()
   if (j+1)%16==0 or j+1==512:
    progress=dict(completed_groups=j+1,groups512=512,fresh_forward_calls=c['set_image_calls'],reused=sum(x['reused'] for x in results[:j+1]),runtime_failures=sum(x['prompt_available'] and not x['runtime_available'] for x in results[:j+1]),seconds=time.monotonic()-start);write(out/'PROGRESS.json',progress,False);print(json.dumps(progress),flush=True)
  first=next((s for s in results if s['runtime_available']),None)
  if first is not None:
   a=infer(predictor,load_RGB(first),np.asarray(first['box_xyxy'],np.float64),torch,h,c);eq={k:ash(x)==first['array_sha256'][k] for k,x in a.items()};repeat=dict(attempted=True,group_id=first['group_id'],PASS=all(eq.values()),raw_array_byte_equal=eq,outputs=save_packet(out/'REPEAT_FIRST_USABLE_GROUP_OUTPUTS.npz',a))
  else:repeat=dict(attempted=False,PASS=False,reason='NO_USABLE_GROUP_NO_RETRY')
 except BaseException as exc:fatal=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 try:after=h['state_sha'](model) if model is not None and before is not None else None
 except Exception as exc:fatal=fatal or dict(type=type(exc).__name__,message='finalstate: '+str(exc))
 finalrows=[]
 for g in results:
  for r in g['diagnostic_input_metadata8']:
   avail=bool(r['common_query_available'] and g['prompt_available'] and g['runtime_available']);reason=None if avail else 'ORIGINAL_COVER_INPUT_U' if not r['common_query_available'] else 'PROMPT_UNAVAILABLE' if not g['prompt_available'] else 'RUNTIME_UNAVAILABLE'
   finalrows.append(dict(r,prompt_available=g['prompt_available'],runtime_available=g['runtime_available'],prediction_available=avail,prediction_U_reason=reason,group_output=g['outputs'],reused_COVER_raw4=g['reused']))
 rp=out/'ALL4096_DIAGNOSTIC_INPUTS_AND_PREDICTION_STATUS.jsonl';write_rows(rp,finalrows)
 unchanged=True
 try:
  for b in used.values():check(b)
  for k in ('code','protocol','scope','A_groups','A_rows','groups','rows','old_complete','source_pin','tests','checkpoint'):check(e[k])
  for b in pin['source_files']:check(b)
 except Exception as exc:unchanged=False;fatal=fatal or dict(type=type(exc).__name__,message='finalimmutable: '+str(exc))
 passed=fatal is None and all(g['RGB_verified'] and (g['runtime_available'] or not g['prompt_available']) for g in results) and repeat['PASS'] and before is not None and before==after and unchanged
 complete=dict(completed=True,PASS=passed,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),groups_manifest=e['groups'],rows_manifest=e['rows'],groups=results,group_count=512,rows=4096,all_rows=bind(rp),checkpoint=checkpoint,source_pin=e['source_pin'],source_commit=pin['exact_commit'],old_complete=e['old_complete'],cuda_cumsum_guard=None if guard is None else bind(out/'CUDA_CUMSUM_PARITY_GUARD.json'),first_usable_group_repeat=repeat,model_state_before_sha256=before,model_state_after_sha256=after,model_state_byte_unchanged=before is not None and before==after,input_checkpoint_source_hashes_unchanged=unchanged,fresh_forward_counters=c,reused_COVER_groups=sum(g['reused'] for g in results),original_common_COVER_input_U=sum(not r['common_query_available'] for r in finalrows),final_prediction_U=sum(not r['prediction_available'] for r in finalrows),all512groups4096rows_retained=True,new_diagnostic_not_original_production_replay=True,GT_or_teacher_prompt=False,new_training_updates=0,training_admission=False,selected_model=None,fully_solved=False,fatal=fatal,no_automatic_retry=True,seconds=time.monotonic()-start)
 write(out/'COMPLETE.json',complete)
 if not passed:raise SystemExit('Execution failed; all statuses preserved. No automatic retry.')
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json'),fresh_forward_calls=c['set_image_calls'],reused_COVER_groups=complete['reused_COVER_groups'])),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare and args.entry is None:prepare()
 elif args.entry is not None and not args.prepare:run(args.entry)
 else:raise SystemExit('Prepare first; actual run requires separate root-approved --entry')
