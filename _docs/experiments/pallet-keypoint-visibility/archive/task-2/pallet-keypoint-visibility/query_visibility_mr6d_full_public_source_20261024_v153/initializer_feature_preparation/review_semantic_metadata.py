"""Semantic proposal inventory metadata only; no image/query/model truth review."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
H=Path(__file__).resolve().parent;V=H.parent;D=V/'native_reference_preparation/source_validation_visual_review_agent'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
status=V/'ROOT_STATUS_INDEPENDENT_REVIEW.json';status_before=sha(status);assert status_before=='c077e13f2e8d17b3960ac4ce96698461dca089915d3db86feace5bf9de6c30d8'
delivery_path=D/'FINAL_REVIEW_DELIVERY.json';delivery=read(delivery_path);seal_path=ck(delivery['original_review_seal']);seal=read(seal_path)
assert delivery['completed']and delivery['PASS']and seal['completed']and seal['PASS']
for k in ('human_gold','train_ready'):assert delivery[k]is False and seal[k]is False
assert delivery['root_independent_adjudication_pending']and seal['root_adjudication_pending']
assert delivery['visibility_head_score_read']is False and delivery['head_fit']==0
assert seal['query_outputs_read']is False and seal['visibility_model_scores_read']is False and seal['new_model_fit_PnP_private_access']==0
val_binding=delivery['validation360_role_proposals'];full_binding=delivery['all2424_partial_inventory'];assert full_binding==seal['full2424_partial_inventory']
val=lines(ck(val_binding));full=lines(ck(full_binding));assert len(val)==360 and len(full)==2424
projection=read(ck(seal['source_projection']));original_binding=projection['references'];original=lines(ck(original_binding));assert len(original)==2424
selection=read(ck(read(ck(delivery['source_intake']))['selection']));sf=selection['frames'];assert len(sf)==230
validation={r['frame_index']:r for r in sf if r['split']=='SOURCE_VALIDATION'};development={r['frame_index']:r for r in sf if r['split']=='SOURCE_DEVELOPMENT'}
assert len(validation)==45 and len(development)==185 and all(r['scene']=='000001'for r in validation.values())
def key(r):return(r['frame_index'],r['gt_instance_index'],r['native_slot'])
assert len({key(r)for r in original})==2424 and [key(r)for r in original]==[key(r)for r in full]
for o,a in zip(original,full):
 assert {k:a[k]for k in o}==o # includes reference coordinates/order/role/semantic U
 assert a['human_gold']is False and a['train_ready']is False
 assert a['semantic_state']=='U_UNREVIEWED'
 assert a['split']==sf[a['frame_index']]['split']and a['id']==sf[a['frame_index']]['id']and a['source_id']==sf[a['frame_index']]['source_id']
 assert a['gt_instance_index']in sf[a['frame_index']]['obj1_gt_instance_indices']and a['native_slot']in range(8)
FV=[a for a in full if a['split']=='SOURCE_VALIDATION'];FD=[a for a in full if a['split']=='SOURCE_DEVELOPMENT'];assert len(FV)==360 and len(FD)==2064
assert FV==val and {r['frame_index']for r in val}==set(validation)
assert Counter(r['proposal_state']for r in val)==Counter(D=116,E=61,S=9,OOF=93,U=81)
assert all(r['proposal_state']=='U_UNREVIEWED'and r['actual_visual_review']is False for r in FD)
assert all(r['actual_query_entity_correspondence']=='NOT_READ_NOT_ASSUMED'and r['actual_query_availability']=='NOT_READ'for r in val)
assert all(r['human_gold']is False and r['train_ready']is False and r['coordinate_supervision_ready']is False and r['exact_reference_pixel_gold']is False and r['native_order_or_provided_pose_or_reference_changed']is False for r in val)
assert all(r['input_or_head_score_used_for_label']is False and r['mask_used_to_autofill_label']is False and r['convex_self_proxy_used_to_autofill_label']is False for r in val)
assert all(not any(k in r for k in ('query_xy','original_query_xy','selected_index','query_available','raw_head_logit_before_U'))for r in val)
assert sum(r['proposal_state']in ('D','E')for r in val)==delivery['counts']['D_E_role_proposals_before_actual_query_association']==seal['D_E_weak_candidate_roles_before_actual_query_association']==177
assert Counter(len([r for r in val if r['frame_index']==i])for i in validation)==Counter({8:45})
rear_pending=[dict(frame_index=r['frame_index'],source_id=r['source_id'],GT_instance_index=r['gt_instance_index'],native_slot=r['native_slot'],proposal_state=r['proposal_state'])for r in val if r['proposal_state']=='D'and((r['native_slot']==5 and 34<=int(r['frame_key'])<=38)or(r['native_slot']==6 and 36<=int(r['frame_key'])<=38))]
assert sha(status)==status_before and sha(original_binding['path'])==original_binding['sha256']
report=dict(completed=True,PASS=True,created_UTC=datetime.now(timezone.utc).isoformat(),review_code=bind(__file__),unchanged_original_status=bind(status),delivery=bind(delivery_path),semantic_review_seal=bind(seal_path),validation360=val_binding,all2424_partial=full_binding,original2424_reference=original_binding,
 metadata_checks=dict(fixed_validation_frames=45,validation_scene='000001',validation_GT_role_rows=360,validation_counts=dict(Counter(r['proposal_state']for r in val)),DE_GT_role_proposals=177,remaining_development_frames=185,remaining_development_GT_roles_U_UNREVIEWED=2064,total_original_GT_roles=2424,all_ID_instance_slot_split_order_exact=True,all_original_reference_fields_coordinates_semantic_U_unchanged=True,original_reference_file_SHA_unchanged=True,all_human_gold_false=True,all_train_ready_false=True,coordinate_supervision_not_ready=True,proposal_labeler_declares_query_and_head_outputs_unread=True,no_actual_query_fields_copied=True),
 interpretation_KO=['177개는 제공 GT 개체의 native 역할 D/E 약한 제안 수다. 실제 초기 query177개나 학습 가능한 crop177개를 뜻하지 않는다.','실제 first-argmax 개체·native 채널 대응과 입력 유효성 연결을 아직 검증·봉인하지 않았으므로 다른 물체의 crop에 이 역할 라벨을 복사하면 안 된다.','S9도 구조 검토 제안이며 독립 확정 gold가 아니다. 아래 후방 D 제안8개를 포함한 root 독립 판정은 미완료다.','이 부록은 ID/split/count/reference 보존을 검산했으며 실제 시각 라벨의 정답성을 독립 검증하지 않았다. 새 학습이나 성능 향상 증거가 추가된 것은 아니다.'],
 rear_D_proposals_specifically_pending_root_adjudication=rear_pending,all_semantic_proposals_pending_root_adjudication=True,semantic_truth_independent_visual_validation_claim=False,
 access=dict(proposal_and_reference_metadata_rows_read=True,actual_RGB_or_overlay_images_read=False,actual_R0_query_or_candidate_outputs_read=False,feature_or_model_or_checkpoint_body_read=False,private_current_V112_access=False,new_encoder_or_model_calls=0),goal_solved=False,selected_model=None)
o=V/'SEMANTIC_PROPOSAL_METADATA_APPENDIX.json'
with o.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,appendix=bind(o),counts=report['metadata_checks'],rear_D_pending=len(rear_pending)),ensure_ascii=False))
