"""Serialize completed visual review decisions; no model or nearest-GT matching.

Manual decisions below were made after viewing all20 original-query contact
pages, then12 paired structure checks. They are weak source proposals only.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
S=P/'query_visibility_mr6d_full_public_source_20261024_v153'
POLICY=V/'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,x):
 if Path(p).exists():
  assert read(p)==x,'Preserved existing review content differs';return
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def seq(a,b):return set(range(a,b+1))

# All original frames are accounted for. Correct entity does not assert exact
# coordinates, and a visually ambiguous stack never becomes an entity negative.
SUPPORTED=seq(0,7)|seq(31,53)|seq(74,117)-{76,82,114}|seq(124,156)|seq(162,165)|seq(199,217)|{220}
WRONG=seq(60,73)|seq(118,123)|seq(157,161)|seq(180,192)|{218,219}|seq(221,229)
AMBIGUOUS=seq(8,30)|seq(54,59)|seq(166,179)|seq(193,198)
NOQUERY={76,82,114}
# Same pallet entity, but the named native face/channel arrangement is not
# supported by the visual side-by-side source rule. No alternative order chosen.
FACE_UNCERTAIN={112,117,138,211,212}|seq(140,147)
# Rear partial roles are separately clear on GT0 upper pallet in these mixed
# stacks; per-row entity=1 is justified only for those4roles, frame entity staysU.
PARTIAL_UPPER_REAR=seq(8,12)|seq(25,30)

def reason(i):
 if i in NOQUERY:return 'NO_DETECTION_ALL8_RETAINED'
 if i==73:return 'FIRSTARGMAX_BLACK_METAL_STAND_AT_LEFT_NOT_WOODEN_PALLET'
 if i in seq(60,72):return 'FIRSTARGMAX_BLACK_STAND_GRAY_KLT_TAG_OR_BACKGROUND_NOT_GT_PALLET'
 if i in seq(118,123):return 'FIRSTARGMAX_SMALL_WOODEN_CARGO_CRATE_NOT_PALLET'
 if i in seq(157,161):return 'FIRSTARGMAX_CARGO_CRATE_OR_ADJACENT_STAND_NOT_WHOLE_PALLET'
 if i in seq(180,192):return 'FIRSTARGMAX_FOREIGN_WOODEN_CRATE_OR_BLUE_CONTAINER_NOT_UPRIGHT_PALLET'
 if i in {218,219}|seq(221,229):return 'FIRSTARGMAX_SMALL_STAND_OR_GRAY_CONTAINER_FEATURE_NOT_WHOLE_PALLET'
 if i in seq(8,30):return 'STACKED_PALLETS_QUERY_FRONT_REAR_OR_FACE_ASSIGNMENT_AMBIGUOUS'
 if i in seq(54,59):return 'MOSTLY_TRUNCATED_STACK_INSUFFICIENT_WHOLE_ENTITY_IDENTITY'
 if i in seq(166,179)|seq(193,198):return 'UPRIGHT_PALLET_AND_FOREIGN_CRATES_MIXED_VOLUME_NATIVE_DECK_AXES_UNCERTIFIED'
 return 'VISIBLE_WHOLE_PALLET_STRUCTURE_SUPPORTS_SOURCE_ENTITY_NOT_EXACT_POSITION'

def run():
 policy=read(POLICY);assert policy['new_encoder_or_R0_calls_allowed']is False
 assert not(H/'ASSOCIATION_REVIEW_COMPLETE.json').exists()
 assert len(SUPPORTED)==129 and len(WRONG)==49 and len(AMBIGUOUS)==49 and len(NOQUERY)==3
 assert set.union(SUPPORTED,WRONG,AMBIGUOUS,NOQUERY)==set(range(230))and sum(map(len,(SUPPORTED,WRONG,AMBIGUOUS,NOQUERY)))==230
 assert FACE_UNCERTAIN<=SUPPORTED and len(FACE_UNCERTAIN)==13 and PARTIAL_UPPER_REAR<=AMBIGUOUS and len(PARTIAL_UPPER_REAR)==11
 es=read(check(policy['source_query_evidence']));em=read(check(es['evidence_manifest']))['records'];assert len(em)==230
 qs=read(S/'actual_full230_initializer_v1/PREDICTION_SEAL.json');assert qs['completed']and qs['PASS']and qs['rows']==1840
 qfiles={Path(b['path']).name:check(b)for b in qs['outputs']};qr=lines(qfiles['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'])
 with np.load(qfiles['ORIGINAL_NATIVE_QUERY1840_AND_VALID9X9_INPUT_ONLY.npz'],allow_pickle=False)as f:q={k:f[k].copy()for k in f.files}
 refseal=read(check(policy['source_reference_seal']));refpath=check(refseal['references']);refs=lines(refpath)
 assert len(refs)==2424
 byref={(r['frame_index'],r['gt_instance_index'],r['native_slot']):r for r in refs};assert len(byref)==2424
 refindex=read(S/'native_reference_preparation/source_reference_contact_v1/ALL230_SOURCE_REFERENCE_CONTACT_INDEX.json')['frames']
 contacts=read(H/'REVIEW_CONTACT_REGISTRY_BEFORE_JUDGMENT.json')['pages'];paired=read(H/'PAIRED_STRUCTURE_REGISTRY.json')['pages']
 assert len(contacts)==20 and sorted(i for page in contacts for i in page['frame_indices'])==list(range(230))and len(paired)==12
 for b in contacts+paired:assert bind(b['path'])['sha256']==b['sha256']
 allrows=[];frames=[]
 for i,e in enumerate(em):
  frame_entity=1 if i in SUPPORTED else 0 if i in WRONG else -1
  state='SUPPORTED'if i in SUPPORTED else'WRONG_ENTITY'if i in WRONG else'NOQUERY'if i in NOQUERY else'AMBIGUOUS'
  instance=0 if i in SUPPORTED else None
  pages=[dict(path=p['path'],sha256=p['sha256'])for p in contacts+paired if i in p['frame_indices']]
  frame=dict(frame_index=i,id=e['frame_id'],source_id=e['source_id'],split=e['split'],frame_entity_target=frame_entity,frame_association_state=state,selected_gt_instance_index=instance,reason=reason(i),confidence='HIGH'if i in WRONG|NOQUERY else'MEDIUM',original_query_evidence=e['PNG'],provided_reference_overlay=refindex[i]['all_instance_overlay'],actual_viewed_review_pages=pages,exact3px_point_quality='UNKNOWN_NOT_GOLD')
  frames.append(frame)
  for k in range(8):
   g=i*8+k;r=qr[g];assert r['frame_index']==i and r['native_slot']==k and r['source_id']==e['source_id']and str(q['query_id'][g])==r['query_id']
   assert np.array_equal(np.asarray(r['query_xy'],np.float64),q['query_xy'][g],equal_nan=True)and np.array_equal(np.asarray(r['L'],np.float64),q['L'][g],equal_nan=True)
   partial=i in PARTIAL_UPPER_REAR and k>=4
   supported=(i in SUPPORTED and i not in FACE_UNCERTAIN)or partial
   entity=1 if partial else frame_entity
   gt=0 if entity==1 else None
   ref=byref.get((i,gt,k))if gt is not None else None
   assert not supported or ref is not None
   why='PARTIAL_REAR4_7_VISUALLY_SAME_UPPER_PALLET_GT0_FRAME_REMAINS_AMBIGUOUS'if partial else'CORRECT_ENTITY_SAME_NATIVE_FACE_ASSIGNMENT_UNCERTIFIED_NO_PERMUTATION'if i in FACE_UNCERTAIN else'SAME_NATIVE_CORNER_ROLE_SUPPORTED_BY_WHOLE_PALLET_STRUCTURE_COORDINATE_ACCURACY_UNKNOWN'if supported else reason(i)
   row=dict(global_index=g,frame_index=i,native_slot=k,id=e['frame_id'],query_id=r['query_id'],source_id=r['source_id'],split=r['split'],query_xy=r['query_xy'],L=r['L'],original_input_valid=bool(q['continuous_valid'][g]),nearest_valid=bool(q['nearest_valid'][g]),no_detection=bool(q['no_detection'][g]),finite_non_sentinel=bool(q['finite_non_sentinel'][g]),native_grid_any_valid=bool(q['native_grid_valid'][g].any()),frame_entity_target=frame_entity,frame_association_state=state,entity_target=entity,selected_gt_instance_index=gt,selected_GT_instance_id=None if ref is None else ref['instance_id'],role_correspondence_supported=supported,role_correspondence='SUPPORTED'if supported else'UNKNOWN',source_same_native_reference_id=None if ref is None else ref['instance_id']+':native'+str(k),native_name=None if ref is None else ref['native_name'],partial_clear_role_with_frame_entity_U=partial,reason=why,confidence='MEDIUM_STRUCTURAL_WEAK_PROPOSAL'if supported else'HIGH_WRONG_ENTITY'if entity==0 else'UNCERTIFIED',point_quality_target=-1,exact3px_point_quality_known=False,coordinate_error_used_for_decision=False,GT_nearest_or_permutation_used=False,semantic_D_E_S_U_not_copied=True,human_gold=False,original_query_evidence=e['PNG'],provided_reference_overlay=refindex[i]['all_instance_overlay'])
   allrows.append(row)
 assert len(allrows)==1840 and [r['global_index']for r in allrows]==list(range(1840))
 assert sum(r['partial_clear_role_with_frame_entity_U']for r in allrows)==44 and all(r['entity_target']==1 and r['selected_gt_instance_index']==0 for r in allrows if r['partial_clear_role_with_frame_entity_U'])
 assert all(not r['role_correspondence_supported']or r['entity_target']==1 for r in allrows)
 fp=H/'FRAME230_ENTITY_ASSOCIATION_PROPOSALS.json';write(fp,dict(completed=True,frames=frames))
 rp=H/'ALL1840_ORIGINAL_QUERY_ENTITY_AND_NATIVE_ROLE_PROPOSALS.jsonl'
 if rp.exists():assert lines(rp)==allrows,'Preserved original query proposals changed'
 else:
  with rp.open('x')as f:
   for r in allrows:f.write(json.dumps(r,allow_nan=False)+'\n')
 summary=dict(frames=230,rows=1840,frame_entity_states=dict(Counter(r['frame_association_state']for r in frames)),row_entity_targets=dict(Counter(str(r['entity_target'])for r in allrows)),role_supported=sum(r['role_correspondence_supported']for r in allrows),role_unknown=sum(not r['role_correspondence_supported']for r in allrows),partial_upper_rear_rows=44,entity_supported_but_face_uncertain_frames=sorted(FACE_UNCERTAIN),byfold={fold:dict(frames=sum(r['split']==fold for r in frames),frame_states=dict(Counter(r['frame_association_state']for r in frames if r['split']==fold)),rows=sum(r['split']==fold for r in allrows),role_supported=sum(r['role_correspondence_supported']for r in allrows if r['split']==fold),role_supported_available=sum(r['role_correspondence_supported']and r['original_input_valid']for r in allrows if r['split']==fold))for fold in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION')})
 sp=H/'REVIEW_SUMMARY.json';write(sp,dict(completed=True,PASS=True,**summary))
 viewlog=H/'ACTUAL_VISUAL_REVIEW_LOG.json';write(viewlog,dict(completed=True,review_order='Validation45 first then all185 development; whole structure and original crops, then targeted native face/stack checks.',all230_query_boards_actually_seen_in_contacts=True,query_contact_pages=[{**p,'actually_viewed':True}for p in contacts],paired_structure_pages=[{**p,'actually_viewed':True}for p in paired],additional_source_reference_contacts_actually_viewed=['SOURCE_VALIDATION_CONTACT_01.png','SOURCE_VALIDATION_CONTACT_05.png','SOURCE_VALIDATION_CONTACT_06.png'],additional_full_reference_frames_actually_viewed=[0,5,31,45,95,117,166],additional_original_full_boards_actually_viewed=[5,31,73,95,109,113,115,117,166],source_only_independent_instance_note='Other source-only reviewer confirmed GT0upper/GT1lower while scene0pallets visible; no D/E proposal read.',semantic_head_scores_or_other_agent_semantic_proposals_read=False))
 complete=H/'ASSOCIATION_REVIEW_COMPLETE.json';write(complete,dict(completed=True,PASS=True,policy=bind(POLICY),code=bind(__file__),source_query_evidence=policy['source_query_evidence'],initializer_prediction_seal=bind(S/'actual_full230_initializer_v1/PREDICTION_SEAL.json'),source_reference_seal=policy['source_reference_seal'],source_reference_rows=bind(refpath),frame_proposals=bind(fp),query_proposals=bind(rp),summary=bind(sp),actual_visual_log=bind(viewlog),all230_frames1840_original_q_L_flags_and_order_retained=True,**summary,semantic_labels_copied=0,exact_real_coordinate_quality_labels=0,model_encoder_initializer_fit_PnP_or_private_access=0,nearest_owner_or_new_permutation_or_q_L_correction=0,limits=['Assistant structural weak proposals, not human gold or training admission.','Entity support does not certify qpixel target membership or exact3px point location.','Correct entity but native face choice uncertainty keeps same-slot visibility unavailable; no remapping performed.','Per-row partial44 supports only upper GT0rear4..7; their11whole frames remain ambiguous.','No source D/E/S/OOF semantic state was copied; root must perform a separate same-native role join.']))
 print(json.dumps(dict(review=bind(complete),**summary)),flush=True)
if __name__=='__main__':run()
