"""Bounded V3 integration fixtures; invented pixels/metadata, no model or source."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
V3=P/'query_visibility_matched_query_scalar_20261023_v94/paired_source_preparation/v3_integrated_same_q_builder/integrated_same_q_v3.py'

def load():
 s=importlib.util.spec_from_file_location('independent_v95_v3',V3);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def decision(r,state):
 return dict(query_key=[r['input_id'],r['native_slot_logical_only']],input_id=r['input_id'],native_slot=r['native_slot_logical_only'],frame_index=r['frame_index'],split=r['split'],image_sha256=r['image_sha256'],original_q=r['query_xy'],original_L=r['L'],before_aux_state=state,paired_construction_eligible=state=='D',support_bbox_xyxy=[1.5,29.5,29.5,32.5]if state=='D'else None,uncertainty_bbox_xyxy=[1.5,29.5,29.5,32.5]if state=='D'else None,native_expected_part_state='UNKNOWN',human_gold=False)

def main():
 m=load()
 row=dict(slot_index=0,frame_index=0,input_id='GENERATED',native_slot_logical_only=0,split='TRAIN_LAB',image_sha256='a'*64,raw_hw=[96,96],query_xy=[32.,32.],L=64.,selected_box_xyxy=[0.,0.,70.,70.],original_input_valid=True)
 cropfile=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
 crop,_=m.v1.frozen_rgb96_operator(cropfile,'802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66')
 target=np.full((96,96,3),13,np.uint8);donor=np.full((32,60,3),219,np.uint8);visible=np.zeros((32,60),bool);visible[4:26,4:56]=True
 metadata=dict(id='ASSIGNED',split='TRAIN',object_id=3,image_sha256='b'*64,foreign_opaque_cutout_admitted=True)
 before=np.zeros((65,9,9),np.float16);before[64]=1
 def cachekey(rgb,q,L):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),tuple(q),float(L)
 cache={cachekey(target,row['query_xy'],row['L']):before.copy()};hits=[];misses=[]
 def bound_callback(rgb,q,L):
  k=cachekey(rgb,q,L)
  if k in cache:hits.append(k);return cache[k].copy()
  misses.append(k);value=int(k[0][:4],16)%1000+1
  roi=np.full((65,9,9),value,np.float16);roi[64]=1;cache[k]=roi
  return roi.copy()
 good=m.materialize_row(target,row,decision(row,'D'),donor,visible,metadata,crop,before,bound_callback)
 assert good['pair_ready']and len(misses)==2 and not hits
 assert [o['gt_only']['candidate_aux_hidden_label']for o in good['occurrences']]==[0,1,0]
 for o in good['occurrences'][1:]:
  assert cachekey(o['full_rgb'],row['query_xy'],row['L'])in misses
  assert not np.array_equal(o['head_inputs_only']['p3_roi'],before)
  assert set(o['head_inputs_only'])=={'p3_roi','local_rgb96','local_valid96'}
 badmeta=dict(metadata,object_id=2)
 failed=m.materialize_row(target,row,decision(row,'D'),donor,visible,badmeta,crop,before,bound_callback)
 assert not failed['pair_ready']and len(hits)==2 and len(misses)==2
 assert [o['gt_only']['candidate_aux_hidden_label']for o in failed['occurrences']]==[0,-1,-1]
 assert all(np.array_equal(o['full_rgb'],target)and np.array_equal(o['head_inputs_only']['p3_roi'],before)for o in failed['occurrences'])
 # Natural E is a canonical source field outside this D/U construction API.
 try:m.decision_identity(row,decision(row,'E'))
 except AssertionError:pass
 else:raise AssertionError('D/U construction API unexpectedly admitted natural E')
 # Invented full960 cohort with exact prospective D/E/U totals and original splits.
 rr=[];dd=[];canonical=[]
 for i in range(960):
  if i<160:split='TRAIN'if i<96 else'VALIDATION'
  else:split='TRAIN_LAB'if i<512 else'VAL_HALL'
  r=dict(row,slot_index=i,frame_index=i//8,input_id='GENERATED_FRAME_'+str(i//8),native_slot_logical_only=i%8,split=split)
  y=0 if i in [0,1,2,96,97,98]or 160<=i<216 or 512<=i<593 else 1 if 216<=i<225 or 593<=i<604 else -1
  rr.append(r);canonical.append(y);dd.append(decision(r,'D'if y==0 else'U'))
 plan=m.all_row_occurrence_plan(rr,dd);assert len(plan)==2880
 assert sum(y==0 for y in canonical)==143 and sum(y==1 for y in canonical)==20 and sum(y==-1 for y in canonical)==797
 assert [x['slot_index']for x in plan[::3]]==list(range(960))
 assert all(x['original_query_xy']==rr[x['slot_index']]['query_xy']and x['original_L']==rr[x['slot_index']]['L']and x['original_split']==rr[x['slot_index']]['split']for x in plan)
 assert all(plan[3*i]['root_before_aux_state_GT_ONLY']=='U'for i,y in enumerate(canonical)if y==1)
 # Assignment is metadata-first and invariant to candidate input order.
 donors=[dict(id=str(i),split='TRAIN',object_id=i,image_sha256=str(i)*64)for i in [1,3,5,7]]
 assert m.choose_one_donor(row,donors)==m.choose_one_donor(row,donors[::-1])
 result=dict(completed=True,PASS=True,scope='Generated integration only; no actual source/private/model reads.',
  changed_fullRGB_callback_misses=2,changed_fullRGB_never_used_before_cached_ROI=True,
  failed_pair_sameRGB_sameqL_cache_hits=2,unchanged_only_cache_reuse_allowed=True,
  both_afterU_and_beforeD_preserved=True,natural_E_construction_API_rejected_and_canonical_separate=True,
  invented_original_rows=960,occurrences=2880,canonical_before_D=143,canonical_before_E=20,canonical_before_U=797,
  original_q_L_split_order_exact=True,one_donor_minSHA_order_invariant=True,
  new_source_pixels_model_forward_fit_current_PnP=0)
 out=HERE/'INDEPENDENT_V3_INTEGRATION_FIXTURES.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
