"""Independent V182 saved-SAM audit. No producer import, RGB, weights or forward.
--prepare: invented arrays and metadata only. --complete: after root terminal notification.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
ENTRY=V/'ROOT_ACTUAL_COMMON_QUERY_SAM_ENTRY.json'
RAW=('image_logits','lowres_logits','predicted_quality','mask_logits_gt0')
META=('box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available')
COMMIT='dca509fe793f601edb92606367a655c15ac00fdf'
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','private','_v169','_v177','_v178','synthetic_val','final_synth_val'))
 assert p.suffix not in ('.pth','.pt','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def qarray(q):
 if q is None:return np.full((8,2),np.nan,np.float64)
 a=np.array([[np.nan,np.nan] if x is None else [np.nan if v is None else v for v in x] for x in q],np.float64);assert a.shape==(8,2);return a
def lengths(x):
 a=np.array([np.nan if v is None else v for v in ([None]*8 if x is None else x)],np.float64);assert a.shape==(8,);return a
def box_of(q):
 a=qarray(q)
 if not np.isfinite(a).all():return None
 b=np.array([min(a[:,0]),min(a[:,1]),max(a[:,0]),max(a[:,1])],np.float64)
 return b if b[2]>b[0] and b[3]>b[1] else None
def expected_packet_metadata(g):
 rr=g['diagnostic_input_metadata8']
 return dict(box_xyxy=box_of(g['common_query_xy8']),original_query_xy=qarray(g['common_query_xy8']),original_L=lengths(g['common_L8']),native_slot=np.array([r['native_slot'] for r in rr],np.int8),original_global_index=np.array([r['diagnostic_global_index'] for r in rr],np.int64),original_occurrence_id=np.array([r['row_id'] for r in rr],np.str_),original_query_available=np.array(g['common_original_query_available8'],bool),prediction_available=np.array(g['common_original_query_available8'],bool))
def load(b):
 with np.load(check(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def check_raw(a):
 expected={'image_logits':(np.dtype('float32'),(512,512)),'lowres_logits':(np.dtype('float32'),(256,256)),'predicted_quality':(np.dtype('float32'),(1,)),'mask_logits_gt0':(np.dtype('bool'),(512,512))}
 for k,(dt,shape) in expected.items():assert a[k].dtype==dt and a[k].shape==shape,k
 assert all(np.isfinite(a[k]).all() for k in RAW[:3]) and np.array_equal(a['mask_logits_gt0'],a['image_logits']>0)
def invented():
 c={}
 def ok(k,v):assert v,k;c[k]=True
 q=[[i+.25,2*i-.5] for i in range(8)]
 ok('independent_exact_box_unclipped',box_of(q).tolist()==[.25,-.5,7.25,13.5])
 ok('null_q8_has_no_prompt_and_no_imputation',box_of(None) is None and np.isnan(qarray(None)).all())
 ok('partial_nonfinite_q8_has_no_prompt',box_of([[None,None]]+q[1:]) is None)
 a=np.array([-1.,0.,np.nextafter(np.float32(0),np.float32(1))],np.float32)
 ok('strict_positive_mask_keeps_zero_background',(a>0).tolist()==[False,False,True])
 ok('bytes_not_allclose_or_equal_nan',not same(np.array([0.],np.float32),np.array([-0.],np.float32)) and not same(np.array([1.],np.float32),np.array([1.],np.float64)))
 ok('U_priority_original_before_prompt_runtime',all((not original or not prompt or not runtime)==(not(original and prompt and runtime)) for original in (False,True) for prompt in (False,True) for runtime in (False,True)))
 rr=[dict(native_slot=k,diagnostic_global_index=100+k,row_id=f'NEW/native{k}') for k in range(8)]
 g=dict(diagnostic_input_metadata8=rr,common_query_xy8=q,common_L8=[96.]*8,common_original_query_available8=[True,False]+[True]*6)
 m=expected_packet_metadata(g)
 ok('diagnostic_IDs_remain_new_not_origin_alias',m['original_occurrence_id'][0]=='NEW/native0' and m['original_global_index'][0]==100)
 ok('U_boolean_and_float64_qL_exact_schema',m['original_query_available'].dtype==bool and m['original_query_xy'].dtype==m['original_L'].dtype==np.float64 and not m['prediction_available'][1])
 return c

def audit(complete_path):
 assert Path(complete_path).resolve()==V/'frozen_common_query_sam_v1/COMPLETE.json'
 e=read(ENTRY);assert e['approved'] and e['source_TRAIN_only'] and e['new_diagnostic_not_original_production_replay']
 for k in ('code','protocol','scope','A_groups','A_rows','groups','rows','old_complete','source_pin','tests'):check(e[k])
 p=read(e['protocol']['path']);c=read(complete_path);old=read(e['old_complete']['path']);pin=read(e['source_pin']['path'])
 assert c['completed'] and c['PASS'] and c['entry']==bind(ENTRY) and c['code']==e['code'] and c['protocol']==e['protocol'] and c['scope']==e['scope']
 assert c['groups_manifest']==e['groups'] and c['rows_manifest']==e['rows'] and c['old_complete']==e['old_complete'] and c['checkpoint']==e['checkpoint'] and c['source_pin']==e['source_pin']
 assert c['source_commit']==old['source_commit']==pin['exact_commit']==COMMIT and c['checkpoint']==old['checkpoint']
 assert c['checkpoint']['sha256']=='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
 assert c['new_diagnostic_not_original_production_replay'] and c['all512groups4096rows_retained'] and c['training_admission'] is False and c['selected_model'] is None and c['new_training_updates']==0 and c['GT_or_teacher_prompt'] is False and c['fatal'] is None and c['no_automatic_retry']
 for b in pin['source_files']:check(b)
 frozen=lines(e['groups']['path']);frows=lines(e['rows']['path']);ag=lines(e['A_groups']['path']);ar=lines(e['A_rows']['path']);final=lines(check(c['all_rows']))
 assert len(c['groups'])==len(frozen)==len(ag)==512 and len(frows)==len(final)==len(ar)==4096
 assert c['group_count']==512 and c['rows']==4096
 old_scenes={s['scene_index']:s for s in old['scenes']};counts=Counter();perbase={};first=None;sampled=[]
 for j,(s,f,a) in enumerate(zip(c['groups'],frozen,ag)):
  for k,v in f.items():assert s[k]==v,(j,k)
  assert s['group_id']==a['group_id'] and s['scene_index']==a['scene_index'] and s['base_index']==a['base_index'] and s['variant']==a['variant'] and s['split']=='TRAIN'
  assert s['decoded_RGB_sha256_observed']==s['decoded_RGB_sha256']==a['decoded_RGB_SHA'] and s['RGB_verified']
  rr=s['diagnostic_input_metadata8'];assert len(rr)==8 and rr==s['original_input_metadata8'] and rr==frows[j*8:j*8+8]
  assert [r['native_slot'] for r in rr]==list(range(8))
  anchor=old_scenes[s['anchor_scene_index']];origin=old_scenes[s['scene_index']];an=anchor['original_input_metadata8'];im=origin['original_input_metadata8']
  assert anchor['variant']=='COVER' and anchor['base_index']==s['base_index'] and origin['variant']==s['variant'] and origin['base_index']==s['base_index']
  assert s['anchor_original_global_index8']==[r['original_global_index'] for r in an] and s['anchor_original_occurrence_id8']==[r['original_occurrence_id'] for r in an] and s['anchor_original_flags8']==[r['original_input_flags'] for r in an]
  assert s['image_original_global_index8']==[r['original_global_index'] for r in im] and s['image_original_occurrence_id8']==[r['original_occurrence_id'] for r in im] and s['image_original_flags8']==[r['original_input_flags'] for r in im]
  assert s['image_original_query_xy8']==[r['original_query_xy'] for r in im] and s['image_original_L8']==[r['original_L'] for r in im]
  assert same(qarray(s['common_query_xy8']),qarray([r['original_query_xy'] for r in an])) and same(lengths(s['common_L8']),lengths([r['original_L'] for r in an]))
  assert s['common_original_query_available8']==[r['original_input_flags']['query_available'] for r in an]
  assert s['source_RGB_PNG_binding']==im[0]['raw_RGB_PNG'] and s['source_RGB_buffer_binding']==im[0]['full_RGB_packet']
  expected=expected_packet_metadata(s);box=expected['box_xyxy'];assert s['prompt_available']==(box is not None)
  assert (None if box is None else box.tolist())==s['box_xyxy']==s['common_q8_context_box_xyxy']
  perbase.setdefault(s['base_index'],[]).append(s)
  for k,(r,original,fin) in enumerate(zip(rr,ar[j*8:j*8+8],final[j*8:j*8+8])):
   assert r['row_id']==original['row_id']==r['original_occurrence_id']==s['group_id']+f'/native{k}'
   assert r['diagnostic_global_index']==r['original_global_index']==original['diagnostic_row_index']==j*8+k
   assert r['anchor_original_global_index']==an[k]['original_global_index'] and r['image_original_global_index']==im[k]['original_global_index']
   assert r['anchor_original_occurrence_id']==an[k]['original_occurrence_id'] and r['image_original_occurrence_id']==im[k]['original_occurrence_id']
   assert r['anchor_original_query_xy']==an[k]['original_query_xy'] and r['anchor_original_L']==an[k]['original_L'] and r['anchor_original_flags']==an[k]['original_input_flags']
   assert r['image_original_query_xy']==im[k]['original_query_xy'] and r['image_original_L']==im[k]['original_L'] and r['image_original_flags']==im[k]['original_input_flags']
   for key in ('common_query_xy','original_query_xy'):assert r[key]==original['query_xy']
   for key in ('common_L','original_L'):assert r[key]==original['L']
   assert r['common_query_available']==original['original_query_available'] and r['original_input_flags']==original['anchor_original_flags']
   for key,value in r.items():assert fin[key]==value
   avail=bool(r['common_query_available'] and box is not None and s['runtime_available']);reason=None if avail else 'ORIGINAL_COVER_INPUT_U' if not r['common_query_available'] else 'PROMPT_UNAVAILABLE' if box is None else 'RUNTIME_UNAVAILABLE'
   assert fin['prediction_available']==avail and fin['prediction_U_reason']==reason and fin['group_output']==s['outputs'] and fin['reused_COVER_raw4']==s['reused']
   counts['original_U']+=not r['common_query_available'];counts['prediction_U']+=not avail
  if box is None:
   assert not s['runtime_available'] and s['runtime_reason']=='PROMPT_UNAVAILABLE' and s['outputs'] is None and not s['reused'];counts['prompt_missing_groups']+=1;continue
  assert s['runtime_available'] and s['runtime_reason'] is None
  arr=load(s['outputs']);assert set(arr)==set(RAW+META);check_raw(arr)
  for k,v in expected.items():assert same(arr[k],v),(s['group_id'],k)
  for k,v in arr.items():assert ash(v)==s['array_sha256'][k]
  if first is None:first=(s,{k:arr[k].copy() for k in RAW})
  counts['saved_packets']+=1;sampled.append(s['outputs'])
  if s['variant']=='COVER':
   assert s['reused'] and s['reuse_proof']['old_output']==anchor['outputs'] and s['reuse_proof']['raw4_byte_copy_only'] and s['reuse_proof']['fresh_diagnostic_metadata']
   oldarr=load(anchor['outputs']);assert set(oldarr)==set(RAW+META)
   for k in RAW:assert same(arr[k],oldarr[k]);assert ash(arr[k])==s['reuse_proof']['raw_array_SHA'][k]
   assert same(oldarr['original_query_xy'],arr['original_query_xy']) and same(oldarr['original_L'],arr['original_L']) and same(oldarr['native_slot'],arr['native_slot']) and same(oldarr['original_query_available'],arr['original_query_available']) and same(oldarr['box_xyxy'],arr['box_xyxy'])
   assert oldarr['original_occurrence_id'].tolist()==s['anchor_original_occurrence_id8'] and oldarr['original_global_index'].tolist()==s['anchor_original_global_index8']
   assert arr['original_occurrence_id'].tolist()!=oldarr['original_occurrence_id'].tolist()
   assert Path(s['outputs']['path']).resolve()!=Path(anchor['outputs']['path']).resolve();counts['COVER_raw4_exact_copy']+=1
  else:assert not s['reused'] and s['reuse_proof'] is None;counts['fresh_groups']+=1
 assert set(perbase)==set(range(128))
 for b,gg in perbase.items():
  assert len(gg)==4 and {g['variant'] for g in gg}=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
  q=expected_packet_metadata(gg[0]);flags=gg[0]['anchor_original_flags8']
  for g in gg[1:]:
   m=expected_packet_metadata(g)
   for k in ('box_xyxy','original_query_xy','original_L','native_slot','original_query_available'):
    assert (q[k] is None and m[k] is None) or (q[k] is not None and m[k] is not None and same(q[k],m[k]))
   assert g['anchor_original_flags8']==flags
 assert counts==Counter(original_U=96,prediction_U=96,prompt_missing_groups=8,saved_packets=504,COVER_raw4_exact_copy=126,fresh_groups=378)
 repeat=c['first_usable_group_repeat'];assert repeat['attempted'] and repeat['PASS'] and repeat['group_id']==first[0]['group_id'] and all(repeat['raw_array_byte_equal'].values())
 repeatarr=load(repeat['outputs']);assert set(repeatarr)==set(RAW)
 for k in RAW:assert same(repeatarr[k],first[1][k])
 counter=c['fresh_forward_counters'];assert counter['set_image_calls']==counter['predict_calls']==counts['fresh_groups']+1==379 and len(counter['predict_warn_only_scopes'])==379
 for n,audit in enumerate(counter['predict_warn_only_scopes']):assert audit==dict(predict_call=n+1,strict_before=True,warn_only_during=True,strict_restored_after=True)
 assert c['reused_COVER_groups']==126 and c['original_common_COVER_input_U']==c['final_prediction_U']==96
 assert c['model_state_byte_unchanged'] and c['model_state_before_sha256']==c['model_state_after_sha256']==old['model_state_after_sha256'] and c['input_checkpoint_source_hashes_unchanged']
 guard=read(check(c['cuda_cumsum_guard']));assert guard['completed'] and guard['PASS']
 for z in guard['dimensions']:assert z['byte_equal'] and z['CPU_sha256']==z['CUDA_sha256'] and z['shape']==[64,64] and z['dtype']=='float32'
 for b in sampled:check(b)
 # Checkpoint/source RGB immutable claims are checked against their saved bindings/receipts;
 # no new checkpoint or original RGB body read is performed by this independent audit.
 return dict(completed=True,PASS=True,code=bind(__file__),scope=e['scope'],root_entry=bind(ENTRY),actual_SAM_complete=bind(complete_path),A_inputs=[e['A_groups'],e['A_rows']],frozen_SAM_inputs=[e['groups'],e['rows']],old_SAM_complete=e['old_complete'],counts=dict(counts),all128bases_same_COVER_q_box_L_flags_across4=True,all4096_fresh_diagnostic_IDs_and_both_production_origins_verified=True,all504_saved_NPZ_schema_dtype_shape_raw_gt0_and_metadata_exact=True,COVER126_raw4_reloaded_old_and_new_byte_equal=True,repeated_first_usable_raw4_reloaded_byte_equal=True,fresh_calls_including_repeat=379,official_commit=COMMIT,official_source_text_hashes_verified=True,fullstate_same_as_frozen_old_SAM_and_after_via_saved_hash_receipts=True,all_predict_scopes_strict_restored=True,no_producer_import_or_helper_reuse=True,actual_RGB_weight_feature_model_body_reads_or_forward=0,scope_limit='Saved arrays/JSON/text only: no independent checkpoint state deserialization, RGB re-decode, or model forward. Runtime/state/input immutability claims are receipt-bound; physical target/native truth is not certified.',training_admission=False,selected_model=None,fully_solved=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--complete',type=Path);a=ap.parse_args()
 if a.prepare and a.complete is None:
  tests=invented();write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),invented_checks=tests,actual_output_bodies_read=0,command=f'{sys.executable} {Path(__file__).resolve()} --complete {V}/frozen_common_query_sam_v1/COMPLETE.json',root_terminal_notification_required=True));print(json.dumps({'prepared':True,'receipt':bind(H/'PREPARATION_COMPLETE.json')}))
 elif a.complete is not None and not a.prepare:
  assert not (H/'REVIEW.json').exists();report=audit(a.complete);write(H/'REVIEW.json',report);print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json')}))
 else:raise SystemExit('Use --prepare now or --complete only after root completion notification.')
if __name__=='__main__':main()
