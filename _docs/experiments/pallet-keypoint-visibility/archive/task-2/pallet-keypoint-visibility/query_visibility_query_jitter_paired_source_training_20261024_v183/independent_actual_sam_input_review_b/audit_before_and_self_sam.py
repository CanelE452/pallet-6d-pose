"""Independent V183 saved BEFORE-SAM + SELF source mapping audit. No producer import.
Run --complete only after root reports terminal; no model/RGB/weights/forward.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;ENTRY=V/'ROOT_ACTUAL_BEFORE_ANCHOR_SAM_ENTRY.json'
RAW=('image_logits','lowres_logits','predicted_quality','mask_logits_gt0');META=('box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available')
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','private','_v169','_v177','_v178','_v182','synthetic_val','final_synth_val')) and p.suffix not in ('.pth','.pt','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(b):return [json.loads(s) for s in check(b).read_text().splitlines() if s]
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def q8(q):return np.asarray([[np.nan,np.nan] if x is None else x for x in ([None]*8 if q is None else q)],np.float64)
def L8(L):return np.asarray([np.nan if x is None else x for x in ([None]*8 if L is None else L)],np.float64)
def box(q):
 a=q8(q)
 if a.shape!=(8,2) or not np.isfinite(a).all():return None
 b=np.array([a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()],np.float64)
 return b if b[2]>b[0] and b[3]>b[1] else None
def load(b,keys=None):
 with np.load(check(b),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def rawcheck(a):
 for k,dt,shape in [('image_logits',np.float32,(512,512)),('lowres_logits',np.float32,(256,256)),('predicted_quality',np.float32,(1,)),('mask_logits_gt0',bool,(512,512))]:assert a[k].dtype==dt and a[k].shape==shape
 assert all(np.isfinite(a[k]).all() for k in RAW[:3]) and np.array_equal(a['mask_logits_gt0'],a['image_logits']>0)
def oldmeta(rr,b):
 return dict(box_xyxy=np.asarray(b,np.float64),original_query_xy=q8([r['original_query_xy'] for r in rr]),original_L=L8([r['original_L'] for r in rr]),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),original_global_index=np.asarray([r['original_global_index'] for r in rr],np.int64),original_occurrence_id=np.asarray([r['original_occurrence_id'] for r in rr]),original_query_available=np.asarray([r['original_input_flags']['query_available'] for r in rr],bool),prediction_available=np.asarray([r['original_input_flags']['query_available'] for r in rr],bool))
def audit(cp):
 assert Path(cp).resolve()==V/'frozen_before_anchor_sam_v1/COMPLETE.json'
 e=read(ENTRY);assert e['approved'] and e['source_TRAIN_only'] and e['new_augmentation_not_original_production_replay']
 for k in ('code','protocol','scope','A_scenes','A_rows','SELF_groups','groups','rows','old_complete','source_pin','tests'):check(e[k])
 c=read(cp);old=read(e['old_complete']['path']);scope=read(e['scope']['path']);pin=read(e['source_pin']['path'])
 assert c['completed'] and c['PASS'] and c['entry']==bind(ENTRY) and c['code']==e['code'] and c['protocol']==e['protocol'] and c['scope']==e['scope']
 assert c['old_complete']==e['old_complete'] and c['SELF_original_SAM_groups_manifest']==e['SELF_groups'] and c['SELF_new_SAM_forwards']==0
 assert c['checkpoint']==old['checkpoint']==e['checkpoint'] and c['source_commit']==old['source_commit']==pin['exact_commit']=='dca509fe793f601edb92606367a655c15ac00fdf'
 assert c['checkpoint']['sha256']=='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
 assert not c['fatal'] and c['new_training_updates']==0 and not c['training_admission'] and c['selected_model'] is None and c['GT_or_teacher_prompt'] is False and c['no_automatic_retry']
 frozen=lines(e['groups']);frows=lines(e['rows']);scenes=lines(e['A_scenes']);refs=lines(e['A_rows']);selfgroups=lines(e['SELF_groups']);final=lines(c['all_rows'])
 assert len(frozen)==len(c['groups'])==len(scenes)==len(selfgroups)==512 and len(frows)==len(final)==4096 and len(refs)==65536
 oldgroups={s['scene_index']:s for s in old['scenes']};assert set(oldgroups)==set(range(512))
 refby={r['original_global_index']:r for r in refs};assert set(refby)==set(range(65536));counts=Counter();basegroups={};first=None
 for j,(s,f) in enumerate(zip(c['groups'],frozen)):
  for k,v in f.items():assert s[k]==v,(j,k)
  assert s['split']=='TRAIN' and s['family']=='BEFORE_ANCHOR4' and s['scene_index']==j and s['group_id']==f"BEFORE_ANCHOR_BASE{s['base_index']:03d}/{s['variant']}"
  anchor=oldgroups[s['anchor_scene_index']];im=oldgroups[s['scene_index']]['original_input_metadata8'];an=anchor['original_input_metadata8'];assert anchor['variant']=='BEFORE' and anchor['base_index']==s['base_index']
  assert s['source_RGB_PNG_binding']==im[0]['raw_RGB_PNG'] and s['source_RGB_buffer_binding']==im[0]['full_RGB_packet'] and s['RGB_verified'] and s['decoded_RGB_sha256_observed']==s['decoded_RGB_sha256']==im[0]['decoded_RGB_sha256_inherited']
  assert s['anchor_original_global_index8']==[r['original_global_index'] for r in an] and s['anchor_original_occurrence_id8']==[r['original_occurrence_id'] for r in an] and s['anchor_original_flags8']==[r['original_input_flags'] for r in an]
  assert s['image_original_global_index8']==[r['original_global_index'] for r in im] and s['image_original_occurrence_id8']==[r['original_occurrence_id'] for r in im] and s['image_original_flags8']==[r['original_input_flags'] for r in im]
  assert same(q8(s['common_query_xy8']),q8([r['original_query_xy'] for r in an])) and same(L8(s['common_L8']),L8([r['original_L'] for r in an]))
  assert s['common_original_query_available8']==[r['original_input_flags']['query_available'] for r in an]
  rr=s['augmentation_input_metadata8'];assert rr==s['original_input_metadata8']==frows[j*8:j*8+8] and [r['native_slot'] for r in rr]==list(range(8))
  b=box(s['common_query_xy8']);assert s['prompt_available']==(b is not None) and s['box_xyxy']==(None if b is None else b.tolist());basegroups.setdefault(s['base_index'],[]).append(s)
  for k,(r,fin) in enumerate(zip(rr,final[j*8:j*8+8])):
   ref=refby[r['original_global_index']];assert 61440<=r['original_global_index']<65536 and ref['family']=='BEFORE_ANCHOR4' and r['augmentation_global_index']==r['original_global_index']
   assert r['row_id']==r['original_occurrence_id']==ref['row_id'] and r['A_augmentation_row_ref']==ref and r['native_slot']==ref['native_slot']==k
   assert r['common_query_xy']==ref['query_xy']==an[k]['original_query_xy'] and r['common_L']==ref['L']==an[k]['original_L'] and r['common_query_available']==ref['original_query_available']==an[k]['original_input_flags']['query_available']
   assert r['anchor_original_global_index']==an[k]['original_global_index'] and r['anchor_original_occurrence_id']==an[k]['original_occurrence_id'] and r['anchor_original_flags']==an[k]['original_input_flags']
   assert r['image_original_global_index']==im[k]['original_global_index'] and r['image_original_occurrence_id']==im[k]['original_occurrence_id'] and r['image_original_flags']==im[k]['original_input_flags'] and r['image_original_query_xy']==im[k]['original_query_xy'] and r['image_original_L']==im[k]['original_L']
   for key,v in r.items():assert fin[key]==v
   av=bool(r['common_query_available'] and s['prompt_available'] and s['runtime_available']);reason=None if av else 'ORIGINAL_BEFORE_INPUT_U' if not r['common_query_available'] else 'PROMPT_UNAVAILABLE' if b is None else 'RUNTIME_UNAVAILABLE'
   assert fin['prediction_available']==av and fin['prediction_U_reason']==reason and fin['group_output']==s['outputs'] and fin['reused_BEFORE_raw4']==s['reused'];counts['BEFORE_inputU']+=not r['common_query_available'];counts['BEFORE_predictionU']+=not av
  if b is None:assert not s['runtime_available'] and s['outputs'] is None and not s['reused'] and s['runtime_reason']=='PROMPT_UNAVAILABLE';counts['missing_groups']+=1;continue
  assert s['runtime_available'] and s['runtime_reason'] is None;a=load(s['outputs']);assert set(a)==set(RAW+META);rawcheck(a)
  expected=dict(box_xyxy=b,original_query_xy=q8(s['common_query_xy8']),original_L=L8(s['common_L8']),native_slot=np.asarray([r['native_slot'] for r in rr],np.int8),original_global_index=np.asarray([r['original_global_index'] for r in rr],np.int64),original_occurrence_id=np.asarray([r['row_id'] for r in rr]),original_query_available=np.asarray(s['common_original_query_available8'],bool),prediction_available=np.asarray(s['common_original_query_available8'],bool))
  for k,v in expected.items():assert same(a[k],v),(s['group_id'],k)
  for k,v in a.items():assert ash(v)==s['array_sha256'][k]
  if first is None:first=(s,{k:a[k].copy() for k in RAW})
  counts['new_packets']+=1
  if s['variant']=='BEFORE':
   assert s['reused'] and s['reuse_proof']['old_output']==anchor['outputs'] and s['reuse_proof']['raw4_byte_copy_only'] and s['reuse_proof']['fresh_augmentation_metadata'];oa=load(anchor['outputs']);rawcheck(oa)
   for k in RAW:assert same(a[k],oa[k]) and ash(a[k])==s['reuse_proof']['raw_array_SHA'][k]
   for k,v in oldmeta(an,anchor['box_xyxy']).items():assert same(oa[k],v)
   assert a['original_occurrence_id'].tolist()!=oa['original_occurrence_id'].tolist() and a['original_global_index'].tolist()!=oa['original_global_index'].tolist() and s['outputs']!=anchor['outputs'];counts['old_BEFORE_raw4_copies']+=1
  else:assert not s['reused'] and s['reuse_proof'] is None;counts['fresh_groups']+=1
 assert set(basegroups)==set(range(128))
 for gg in basegroups.values():
  assert len(gg)==4 and {s['variant'] for s in gg}=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
  for s in gg[1:]:assert s['common_query_xy8']==gg[0]['common_query_xy8'] and s['common_L8']==gg[0]['common_L8'] and s['anchor_original_flags8']==gg[0]['anchor_original_flags8'] and s['box_xyxy']==gg[0]['box_xyxy']
 assert counts==Counter(BEFORE_inputU=64,BEFORE_predictionU=64,missing_groups=4,new_packets=508,old_BEFORE_raw4_copies=127,fresh_groups=381)
 states=scope['training_synthetic']['states15'];selfU=0;oldmetadata_packets=0
 for si,(m,s) in enumerate(zip(selfgroups,scenes)):
  og=oldgroups[si];im=og['original_input_metadata8'];assert m['group_id']==f'SELF_SCENE_{si:04d}' and m['family']=='SELF_SCENE15' and m['scene_index']==si and m['split']=='TRAIN' and m['new_SAM_forward'] is False
  assert m['original_input_metadata8']==s['original_input_metadata8']==im and m['original_SAM_output_binding']==s['old_SAM_output']==og['outputs'] and m['original_SAM_box_xyxy']==og['box_xyxy'] and m['original_SAM_prompt_available']==og['prompt_available'] and m['original_SAM_runtime_available']==og['runtime_available']
  expectedindices=[r['original_global_index'] for r in refs[:61440] if r['scene_index']==si];assert m['augmented_row_indices']==expectedindices and len(expectedindices)==120
  for i in expectedindices:
   r=refby[i];k=r['native_slot'];state=states[r['state_index']];q=im[k]['original_query_xy'];L=im[k]['original_L'];assert r['source_image_original_global_index']==r['anchor_original_global_index']==im[k]['original_global_index'] and r['sam_group_id']==m['group_id'] and r['family']=='SELF_SCENE15'
   assert r['query_xy']==(None if q is None else [q[0]+state['dx'],q[1]+state['dy']]) and r['L']==(None if L is None else L*state['scale']) and r['original_query_available']==im[k]['original_input_flags']['query_available'];selfU+=not r['original_query_available']
  if og['outputs'] is not None:
   a=load(og['outputs'],META)
   for k,v in oldmeta(im,og['box_xyxy']).items():assert same(a[k],v),(si,k)
   oldmetadata_packets+=1
  else:assert not og['prompt_available'] and not og['runtime_available']
 assert selfU==1200 and oldmetadata_packets==506 and selfU+counts['BEFORE_inputU']==1264
 repeat=c['first_usable_group_repeat'];assert repeat['attempted'] and repeat['PASS'] and repeat['group_id']==first[0]['group_id'] and all(repeat['raw_array_byte_equal'].values());ra=load(repeat['outputs']);assert set(ra)==set(RAW)
 for k in RAW:assert same(ra[k],first[1][k])
 counter=c['fresh_forward_counters'];assert counter['set_image_calls']==counter['predict_calls']==382 and len(counter['predict_warn_only_scopes'])==382
 for i,a in enumerate(counter['predict_warn_only_scopes']):assert a==dict(predict_call=i+1,strict_before=True,warn_only_during=True,strict_restored_after=True)
 assert c['model_state_byte_unchanged'] and c['model_state_before_sha256']==c['model_state_after_sha256']==old['model_state_after_sha256'] and c['input_checkpoint_source_hashes_unchanged'] and c['original_common_BEFORE_input_U']==c['final_prediction_U']==64
 guard=read(check(c['cuda_cumsum_guard']));assert guard['PASS']
 for a in guard['dimensions']:assert a['byte_equal'] and a['CPU_sha256']==a['CUDA_sha256']
 for b in pin['source_files']:check(b)
 return dict(completed=True,PASS=True,code=bind(__file__),actual_complete=bind(cp),root_entry=bind(ENTRY),counts=dict(counts),SELF512_original_metadata_groups=True,SELF506_saved_metadata_packets_reloaded=True,SELF_inputU1200=True,total_aug_inputU1264=True,all_BEFORE508_packets_schema_and_metadata_exact=True,old_BEFORE127_raw4_byte_equal=True,fresh_augmentationIDs61440to65535=True,all128_same_BEFORE_box_qL_flags4variants=True,first_repeat_raw4_byte_equal=True,official_source_text_hashes_state_and_strict_runtime_receipts_verified=True,no_producer_import=True,scope_limit='Saved SAM/metadata only. No new RGB decode, checkpoint/model state deserialize, features, sampling or forward. State and RGB immutability are receipt-bound.',new_forward_fit_or_selection=0,selected_model=None,fully_solved=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--complete',type=Path);a=ap.parse_args()
 if a.prepare:
  assert a.complete is None;out=H/'BEFORE_SAM_PREPARATION_COMPLETE.json';x=dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),actual_SAM_or_RGB_weight_bodies_read=0,requires_root_terminal_notification=True)
 else:assert a.complete is not None;out=H/'BEFORE_SAM_REVIEW.json';x=audit(a.complete)
 with out.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(receipt=bind(out),PASS=True)))
if __name__=='__main__':main()
