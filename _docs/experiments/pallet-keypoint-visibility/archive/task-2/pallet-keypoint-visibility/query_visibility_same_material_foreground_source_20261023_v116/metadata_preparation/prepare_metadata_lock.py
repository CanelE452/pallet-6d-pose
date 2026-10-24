"""Root-authorized V116 metadata selection only. No payload/array/model imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from copy import deepcopy
import json,hashlib
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
POLICY=V/'ROOT_SOURCE_POLICY_BEFORE_DONORS_AND_PIXELS.json';POLICY_SHA='c2bb6ab955bb9d726281397a0f3df281192cd9acde964fe12cf843d24af0902c'
W=P/'query_visibility_ycb_wood_foam_source_20261023_v113';R=P/'query_visibility_query_anchored_cue_20261023_v96'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(name,rows):
 p=HERE/name
 with p.open('x') as f:
  if isinstance(rows,list):
   for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
  else:f.write(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 return bind(p)
def main():
 assert sha(POLICY)==POLICY_SHA
 F={'policy':POLICY,'V113_inputs400':W/'query_preparation/INPUTS400_QUERY_EMULATION_ONLY.jsonl','V113_reference400':W/'query_preparation/REFERENCE400_GT_ONLY.jsonl','V113_lineage400':W/'material_teacher_feasibility/INSTANCE400_LINEAGE_GT_ONLY.jsonl','V95_rows960':P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/ALL960_NORMALIZED_INPUT_ONLY.jsonl','V96_source960':R/'actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl','V96_targets960':R/'query_anchored_pair_preparation/MATCHED_TARGET960_GT_ONLY.jsonl','old_assignments960':R/'query_anchored_pair_preparation/DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl','V96_kernel':R/'query_anchored_pair_preparation/query_anchored_pair_kernel.py','feasibility_basis':P/'query_visibility_material_augmented_query_head_20261023_v114/next_wood_foreground_source_feasibility/RESULT.json'}
 B={k:bind(p) for k,p in F.items()};ins=lines(F['V113_inputs400']);refs=lines(F['V113_reference400']);lin=lines(F['V113_lineage400']);rows=lines(F['V95_rows960']);src=lines(F['V96_source960']);tar=lines(F['V96_targets960']);oldass=lines(F['old_assignments960'])
 assert len(ins)==len(refs)==len(lin)==400 and len(rows)==len(src)==len(tar)==len(oldass)==960
 candidates=[]
 for i,(x,y,z) in enumerate(zip(ins,refs,lin)):
  assert x['input_id']==y['input_id']==z['input_id'] and x['image']['sha256']==z['image']['sha256']==y['source_instance']['image_sha256']
  if z['object_id']==16 and z['original_split']=='train':
   assert i in range(100,300) and z['scene_id'] in [2,4] and x['split']=='TRAIN'
   owners=z['all_visible_owners'];assert [o['instance_index'] for o in owners]==list(range(z['complete_scene_GT_owner_count'])) and owners[z['instance_id']]['object_id']==16
   candidates.append({'candidate_index':len(candidates),'original_plan_row_index':i,'input_id':x['input_id'],'id':z['original_source_instance_id'],'original_source_instance_id':z['original_source_instance_id'],'original_source_row_index':z['original_source_row_index'],'scene_id':z['scene_id'],'frame_id':z['frame_id'],'instance_id':z['instance_id'],'object_id':16,'split':'TRAIN','original_split':'train','image':deepcopy(z['image']),'image_sha256':z['image']['sha256'],'raw_hw':x['raw_hw'],'own_visible_mask':deepcopy(z['own_visible_mask']),'own_full_mask':deepcopy(z['own_full_mask']),'complete_scene_GT_owner_count':z['complete_scene_GT_owner_count'],'all_visible_owners':deepcopy(owners),'all_visible_owner_files_available_recorded':z['all_visible_owner_files_available'],'controls':deepcopy(z['controls']),'original_input_reference':{'file':B['V113_inputs400'],'row_index':i},'original_reference_GT_only':{'file':B['V113_reference400'],'row_index':i},'original_instance_lineage_GT_only':{'file':B['V113_lineage400'],'row_index':i},'native_mapping':'UNKNOWN','human_gold':False,'payloads_read_here':False})
 assert len(candidates)==200 and Counter(x['scene_id'] for x in candidates)==Counter({2:100,4:100})
 assert len({x['image_sha256'] for x in candidates})==len({x['id'] for x in candidates})==200
 ordered=sorted(candidates,key=lambda x:(x['image_sha256'],x['original_source_instance_id']));chosen=[];seen_rgb=set();seen_id=set()
 for x in ordered:
  if x['image_sha256'] in seen_rgb or x['id'] in seen_id:continue
  y=deepcopy(x);y['donor_rank']=len(chosen);chosen.append(y);seen_rgb.add(x['image_sha256']);seen_id.add(x['id'])
  if len(chosen)==16:break
 assert len(chosen)==16
 oldflat=[d for a in oldass for d in a['ranks']];oldval={d['image_sha256'] for d in oldflat if d['split']=='VALIDATION'};oldtrain={d['image_sha256'] for d in oldflat if d['split']=='TRAIN'}
 assert seen_rgb.isdisjoint(oldval) and {x['image_sha256'] for x in candidates}<=oldtrain
 candidate_binding=write('ALL200_DONOR_METADATA_ONLY.jsonl',candidates);donor_binding=write('SELECTED16_DONORS_METADATA_ONLY.jsonl',chosen)
 bases=[];targets=[]
 for i,(r,s,t) in enumerate(zip(rows,src,tar)):
  assert r['slot_index']==s['base_row_index']==t['base_row_index']==i and r['query_xy']==s['query_xy']==t['original_q'] and r['L']==s['L']==t['original_L']
  if r['split'] in ['TRAIN','TRAIN_LAB'] and t['query_anchored_label']==0:
   assert t['construction_eligible'] and r['original_input_valid'] and t['matched_certified_group'] and t['original_regions']
   j=len(bases);bases.append({'base_index':j,'original_base_row_index':i,'source':s['source'],'source_family':s['source_family'],'split':'TRAIN','raw_split':r['split'],'original_row':deepcopy(r),'source_lineage':deepcopy(s),'original_normalized_input_reference':{'file':B['V95_rows960'],'row_index':i},'original_source_reference':{'file':B['V96_source960'],'row_index':i}})
   targets.append({'base_index':j,'original_base_row_index':i,'original_target':deepcopy(t),'original_target_reference':{'file':B['V96_targets960'],'row_index':i},'target_state_unchanged':'D_AUX','native_mapping':'UNKNOWN','human_gold':False,'training_admission':False})
 assert len(bases)==28 and Counter(x['source'] for x in bases)==Counter({'LOCO':3,'PALLOX':25})
 assert [b['original_base_row_index'] for b in bases]==sorted(b['original_base_row_index'] for b in bases)
 base_binding=write('BASES28_INPUT_ONLY.jsonl',bases);target_binding=write('MATCHED_TARGETS28_GT_ONLY.jsonl',targets)
 attempts=[];plan=[]
 for b in bases:
  j=b['base_index'];r=b['original_row'];s=b['source_lineage']
  for d in chosen:
   assert r['image_sha256']!=d['image_sha256']
   a=16*j+d['donor_rank'];attempts.append({'attempt_index':a,'base_index':j,'original_base_row_index':b['original_base_row_index'],'slot_index':r['slot_index'],'donor_rank':d['donor_rank'],'donor_id':d['id'],'donor_image_sha256':d['image_sha256'],'base_image_sha256':r['image_sha256'],'split':'TRAIN','raw_split':b['raw_split'],'source':b['source'],'base_reference':{'file':base_binding,'row_index':j},'target_GT_only_reference':{'file':target_binding,'row_index':j},'donor_reference':{'file':donor_binding,'row_index':d['donor_rank']},'replacement_allowed':False,'appearance_or_geometry_success_filter_allowed':False})
  for o in range(33):
   rank=None if o==0 else (o-1)//2;variant='BEFORE' if o==0 else 'OPAQUE_COVER' if o%2 else 'CUE_PRESERVING_SHAM'
   oid=f"{r['input_id']}::slot{r['native_slot_logical_only']}::V116::"+('BEFORE' if rank is None else f'RANK_{rank:02d}::{variant}')
   plan.append({'occurrence_index':33*j+o,'occurrence_id':oid,'base_index':j,'original_base_row_index':b['original_base_row_index'],'slot_index':r['slot_index'],'attempt_index':None if rank is None else 16*j+rank,'donor_rank':rank,'variant':variant,'input_id':r['input_id'],'native_slot_logical_only':r['native_slot_logical_only'],'source':b['source'],'source_family':b['source_family'],'split':'TRAIN','raw_split':b['raw_split'],'query_xy':deepcopy(r['query_xy']),'L':r['L'],'raw_hw':r['raw_hw'],'original_input_valid':r['original_input_valid'],'selected_box_xyxy':deepcopy(r['selected_box_xyxy']),'image_sha256_before':r['image_sha256'],'before_fullRGB_path':s['fullRGB_path'],'before_fullRGB_format':s['fullRGB_format'],'before_fullRGB_file_binding':deepcopy(s['fullRGB_file_binding']),'before_decoded_RGB_sha256':s['before_decoded_sha256'],'base_reference':{'file':base_binding,'row_index':j}})
 assert len(attempts)==448 and [x['attempt_index'] for x in attempts]==list(range(448))
 assert len(plan)==924 and [x['occurrence_index'] for x in plan]==list(range(924)) and len({x['occurrence_id'] for x in plan})==924
 assert Counter(x['variant'] for x in plan)==Counter({'BEFORE':28,'OPAQUE_COVER':448,'CUE_PRESERVING_SHAM':448})
 attempt_binding=write('ASSIGNMENTS448_METADATA_ONLY.jsonl',attempts);plan_binding=write('INPUT_PLAN924_INPUT_ONLY.jsonl',plan)
 outputs={'all200':candidate_binding,'selected16':donor_binding,'bases28':base_binding,'targets28_GT_only':target_binding,'assignments448':attempt_binding,'input_plan924':plan_binding}
 receipt={'schema':'V116_EXACT_METADATA_SELECTION_ASSIGNMENT_LOCK_V1','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'policy':B['policy'],'code':bind(Path(__file__)),'bindings':B,'outputs':outputs,'counts':{'donor_candidates':200,'candidate_scenes':dict(Counter(x['scene_id'] for x in candidates)),'selected_donors':16,'selected_scenes':dict(Counter(x['scene_id'] for x in chosen)),'public_TRAIN_bases':28,'bases_by_source':dict(Counter(x['source'] for x in bases)),'base_unique_original_RGB_SHA':len({x['original_row']['image_sha256'] for x in bases}),'assignments':448,'occurrences':924,'BEFORE':28,'COVER':448,'SHAM':448},'selection_rule_exact':'Sort originalRGBSHA256 then originalsourceinstanceID; first16 distinct imageSHA/instanceIDs. No payload/success/appearance filter.','selected_original_candidate_indices':[x['candidate_index'] for x in chosen],'all_base_q_L_target_group_original_regions_stencil_order_exact':True,'all448_source_donor_RGB_differ':True,'overlap':{'all200_vs_oldVALdonor_RGB_SHA':len({x['image_sha256'] for x in candidates}&oldval),'selected16_vs_oldVALdonor_RGB_SHA':len(seen_rgb&oldval),'all200_already_present_in_oldTRAIN_scene_RGB_pool':len({x['image_sha256'] for x in candidates}&oldtrain),'selected16_already_present_in_oldTRAIN_scene_RGB_pool':len(seen_rgb&oldtrain),'hashes_are_sealed_metadata_values_not_payload_rehashes':True},'future_after_states':'PENDING actual certificates: accepted coverE/shamD; failed bothU, never force success. No new after labels generated here.','payload_status':'Original RGB/ownmask/all-owner/control hashes copied as recorded in sealed lineages. No payload bytes/stat/arrays decoded here. Prior feasibility stat present is bound, not re-certified pixels.','eligibility_wrapper':'Exact selected16 object16 originalTRAIN scenes2/4 explicit allowlist, not sealedV96 odd-ID eligibility; lower-level geometry unchanged.','producer_API':'For V96.protected_mask pass BASES28.original_row and MATCHED_TARGETS28.original_target. Do not pass augmented wrappers into strict input schema.','training_admission':False,'native_mapping':'UNKNOWN','human_gold':False,'operations':{'metadata_code_reads':True,'donor_metadata_selection':True,'RGB_mask_depth_feature_payload_reads':0,'array_imports':0,'model_checkpoint_import_or_load':0,'forward_fit_private_current_V112_SAM_PnP':0,'old_artifact_edits':0,'new_root_entry':False}}
 rec=write('SELECTION_AND_ASSIGNMENT_RECEIPT.json',receipt)
 seal={'completed':True,'PASS':True,'receipt':rec,'policy':B['policy'],'code':bind(Path(__file__)),'outputs':outputs,'only_metadata_selected':True,'no_actual_pixels_model_or_fit':True,'training_admission':False}
 se=write('SEAL.json',seal)
 print(json.dumps({'receipt':rec,'seal':se,'outputs':outputs,'counts':receipt['counts'],'overlap':receipt['overlap']}))
if __name__=='__main__':main()
