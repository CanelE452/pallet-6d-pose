"""All37 existing natural cue pairs; metadata only, no pixel/model reader."""
from pathlib import Path
import collections,hashlib,json,math,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PUBLIC37_INTAKE_SCOPE_BEFORE_PIXELS.json';SCOPE_SHA='09cf91dac301efafe7de6e38929751bcecb284f6f5636180551fa8d8742be93e'
V91=P/'query_visibility_physical_cue_aux_20261023_v91/teacher_preparation'
CUES=V91/'corpus_build_v1/ALL302_CUES_SOURCE_IDS_GT_ONLY.jsonl'
NORMAL=P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/ALL960_NORMALIZED_INPUT_ONLY.jsonl'
LINEAGE=P/'query_visibility_query_anchored_cue_20261023_v96/actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl'
MANIFEST=H/'PAIRS37_SOURCE_INTAKE_INPUT_AND_WEAK_EVIDENCE.jsonl'
def bind(p):
 p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,r):
 with Path(p).open('x')as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def notes(node,pointer=''):
 out=[]
 if isinstance(node,dict):
  for k,v in node.items():
   key=pointer+'/'+k
   if 'registration'in k.lower()or'motion'in k.lower()or'movement'in k.lower():out.append(dict(json_pointer=key,value_unchanged=v))
   out.extend(notes(v,key))
 elif isinstance(node,list):
  for i,v in enumerate(node):out.extend(notes(v,pointer+'/'+str(i)))
 return out

def build():
 assert bind(SCOPE)['sha256']==SCOPE_SHA and not MANIFEST.exists()
 scope=read(SCOPE);teacher=scope['source_teacher'];assert bind(teacher['path'])==teacher
 views=rows(teacher['path']);cues=rows(CUES);normal=rows(NORMAL);lineage=rows(LINEAGE)
 assert len(views)==604 and len(cues)==302 and len(normal)==len(lineage)==960
 bycue=collections.defaultdict(dict)
 for i,r in enumerate(views):assert r['cue_view_index']==i;bycue[r['cue_id']][r['view']]=r
 selected=sorted([v for v in bycue.values()if v['before']['teacher_label']==0 and v['after']['teacher_label']==1],key=lambda v:v['before']['cue_view_index'])
 assert len(selected)==37 and collections.Counter(v['before']['split']for v in selected)==dict(TRAIN_LAB=15,VAL_HALL=22)
 byframe=collections.defaultdict(list)
 for i,(a,b)in enumerate(zip(normal,lineage)):
  assert a['slot_index']==b['slot_index']==i and all(a[k]==b[k]for k in('input_id','native_slot_logical_only','query_xy','L','image_sha256','split','raw_hw'))
  byframe[a['input_id']].append((i,a,b))
 cueindex={r['cue_id']:i for i,r in enumerate(cues)};assert len(cueindex)==302
 metadata_bindings=dict(root_scope=bind(SCOPE),teacher604=teacher,cues302=bind(CUES),V91_teacher_contract=bind(V91/'TEACHER_CONTRACT_BEFORE_BUILD.json'),V91_teacher_seal=bind(V91/'corpus_build_v1/GT_CORPUS_SEAL.json'),original_normalized960=bind(NORMAL),raw_RGB_and_initializer_lineage960=bind(LINEAGE),code=bind(__file__))
 result=[];rawfiles={};missinginit=0
 for pair_index,vs in enumerate(selected):
  before,after=vs['before'],vs['after'];cue=cues[cueindex[before['cue_id']]]
  assert cue['cue_id']==before['cue_id']==after['cue_id']and cue['pair_id']==before['pair_id']==after['pair_id']
  assert before['cue_index']==after['cue_index']==cueindex[cue['cue_id']]
  for k in('split','reference_xy_unchanged','uncertainty_bbox_xyxy_half_open_unchanged','uncertainty9_xy_GT_only','duplicate_group_link_only','duplicate_group_size','duplicate_weight'):assert before[k]==after[k],(pair_index,k)
  ref=before['reference_xy_unchanged'];unc=before['uncertainty_bbox_xyxy_half_open_unchanged'];assert len(ref)==2 and len(unc)==4 and all(math.isfinite(x)for x in ref+unc)
  assert cue['reference_xy']==ref and cue['uncertainty_bbox_xyxy_half_open']==unc
  rec=dict(pair_index=pair_index,pair_id=before['pair_id'],cue_id=before['cue_id'],cue_index=before['cue_index'],split=before['split'],source_family='ONE_PALLOX_PHYSICAL_BIN',slot_or_proposal_id=before['slot_or_proposal_id'],before_cue_view_index=before['cue_view_index'],after_cue_view_index=after['cue_view_index'],reference_xy_unchanged=ref,uncertainty_bbox_xyxy_half_open_unchanged=unc,uncertainty9_xy_GT_only_unchanged=before['uncertainty9_xy_GT_only'],duplicate_group_link_only=before['duplicate_group_link_only'],duplicate_group_size=before['duplicate_group_size'],duplicate_weight=before['duplicate_weight'],source_corpus=cue['corpus'],source_cue_provenance=dict(file=bind(CUES),row_index=cueindex[cue['cue_id']]),source_cue_record_unchanged=cue,registration_evidence_unchanged=notes(cue),native_mapping='UNKNOWN',native8_semantic_id=None,human_gold=False,official_gold=False,pixel_material_membership=False,training_admission=False,new_visibility_labels_created=False,diagnostic_reference_crop_not_model_input=True,actual_initializer_slot_selected_by_GT=False)
  for view,r in(('before',before),('after',after)):
   frame=byframe.get(r['frame_input_id'],[]);assert len(frame)==8 and[a['native_slot_logical_only']for _,a,_ in frame]==list(range(8))
   _,a,base=frame[0];rgb=base['fullRGB_file_binding'];decoded=base.get('decoded_RGB_sha256')or base['before_decoded_sha256'];assert rgb['sha256']==r['image_sha256']and a['raw_hw']==r['raw_hw']and Path(rgb['path']).is_file()
   for _,aa,bb in frame:
    assert aa['split']==r['split']and aa['raw_hw']==r['raw_hw']and bb['fullRGB_file_binding']==rgb and(bb.get('decoded_RGB_sha256')or bb['before_decoded_sha256'])==decoded and aa['selected_box_xyxy']==a['selected_box_xyxy']
   initial=[aa['query_xy']for _,aa,_ in frame];sides=[aa['L']for _,aa,_ in frame];no_detection=all(q is None for q in initial);missinginit+=int(no_detection)
   rec[view]=dict(view=view,frame_input_id=r['frame_input_id'],rawRGB=rgb,format=base['fullRGB_format'],decoded_RGB_sha256=decoded,raw_hw=r['raw_hw'],reference_xy_unchanged=r['reference_xy_unchanged'],uncertainty_bbox_xyxy_half_open_unchanged=r['uncertainty_bbox_xyxy_half_open_unchanged'],weak_teacher_label_unchanged=r['teacher_label'],weak_teacher_label_meaning_unchanged=r['label_meaning'],weak_teacher_label_reason_unchanged=r['label_reason'],source_observation_unchanged=r['source_observation_unchanged'],teacher_record_provenance=dict(file=teacher,row_index=r['cue_view_index']),teacher_record_unchanged=r,actual_initializer=dict(metadata_available=True,no_detection=no_detection,initial_points=initial,L8=sides,selected_bbox_xyxy=a['selected_box_xyxy'],original_native_slots=list(range(8)),original_normalized_row_indices=[j for j,_,_ in frame],original_query_continuous_valid=[bb['original_query_continuous_valid']for _,_,bb in frame],query_nearest_valid=[bb['query_nearest_valid']for _,_,bb in frame],original_input_valid=[aa['original_input_valid']for _,aa,_ in frame],original_input_status=[bb['original_input_status']for _,_,bb in frame],q_and_L_unchanged=True,no_GT_best_slot_or_nearest_reference_join=True),native_mapping='UNKNOWN',human_gold=False,training_admission=False)
   rawfiles[rgb['path']]=dict(file=rgb,format=base['fullRGB_format'],decoded_RGB_sha256=decoded,raw_hw=r['raw_hw'])
  assert rec['registration_evidence_unchanged'],'Original registration caveat must not disappear'
  result.append(rec)
 with MANIFEST.open('x')as f:
  for r in result:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 summary=dict(completed=True,PASS=True,manifest=bind(MANIFEST),bindings=metadata_bindings,selection='All existing before0/after1 pairs, sorted original before_cue_view_index; no score/pixel filtering.',pairs37=len(result),original_physical_framepairs=len({r['pair_id']for r in result}),fold_counts=dict(collections.Counter(r['split']for r in result)),source_corpus_counts=dict(collections.Counter(r['source_corpus']for r in result)),unique_raw_RGB_files=len(rawfiles),raw_RGB_references74=74,all_actual_initializer_frames_metadata_present=True,no_detection_frame_occurrences=missinginit,original_teacher_reference_uncertainty_all_exact=True,all_registration_notes_preserved=True,all37_rows_and_all8_actual_queries_per_view_retained=True,actual_initializer_GT_best_slot_selection=False,raw_file_SHA_inherited_from_saved_source_metadata=True,raw_RGB_payloads_or_header_reads=0,native_mapping='UNKNOWN',human_gold=False,pixel_material_membership=False,new_labels_or_training_admission=False,encoder_head_fit_current_V112_private_PnP=0)
 write(H/'SUMMARY.json',summary)
 write(H/'RAW_RGB_REFERENCES44_INPUT_ONLY.json',dict(completed=True,files=list(rawfiles.values()),count=len(rawfiles),payload_hash_revalidation_deferred_to_renderer=True))
 lock=dict(completed=True,locked_before_pixels=True,manifest=bind(MANIFEST),summary=bind(H/'SUMMARY.json'),bindings=metadata_bindings,ordered_cue_ids=[r['cue_id']for r in result],ordered_before_cue_view_indices=[r['before_cue_view_index']for r in result],ordered_after_cue_view_indices=[r['after_cue_view_index']for r in result],native_mapping='UNKNOWN',human_gold=False,training_admission=False,all_actual_RGB_payload_reads=0)
 write(H/'PAIR37_IDENTITY_LOCK_BEFORE_PIXELS.json',lock)
 write(H/'SEAL.json',dict(completed=True,PASS=True,metadata_only=True,outputs={n:bind(H/n)for n in('PAIRS37_SOURCE_INTAKE_INPUT_AND_WEAK_EVIDENCE.jsonl','SUMMARY.json','RAW_RGB_REFERENCES44_INPUT_ONLY.json','PAIR37_IDENTITY_LOCK_BEFORE_PIXELS.json')},code=bind(__file__),root_scope=bind(SCOPE)))
 print(json.dumps(dict(manifest=bind(MANIFEST),lock=bind(H/'PAIR37_IDENTITY_LOCK_BEFORE_PIXELS.json'),seal=bind(H/'SEAL.json'),summary=summary),ensure_ascii=False),flush=True)
if __name__=='__main__':build()
