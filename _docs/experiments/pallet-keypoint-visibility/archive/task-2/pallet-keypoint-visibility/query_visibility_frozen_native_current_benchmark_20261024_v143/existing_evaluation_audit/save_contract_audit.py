"""Read code/contract/summary/seal metadata only; never opens current bodies."""
from pathlib import Path
import hashlib
import json

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_frozen_native_current_benchmark_20261024_v143'
H=Path(__file__).resolve().parent
OLD=P/'query_visibility_training_duration_control_20261023_v124'
POLICY=V/'ROOT_FROZEN_BENCHMARK_POLICY_BEFORE_PRIVATE_BODY_READS.json'
POLICY_SHA='33856e89ab3a563e7ee80d2d2daf846d64199122340c02e7299f66fc9f51cd1b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
evidence=[]
def meta(p):
    p=Path(p);assert p.suffix=='.json'and '_v112'not in str(p).lower()
    evidence.append(dict(kind='existing_contract_summary_or_seal_metadata',**bind(p)))
    return json.loads(p.read_text())
def code(p):
    p=Path(p);assert p.suffix=='.py'and '_v112'not in str(p).lower()
    evidence.append(dict(kind='code',**bind(p)))
assert sha(POLICY)==POLICY_SHA
policy=meta(POLICY)
contract=meta(P/'evaluation_preparation_20261022/EVALUATION_CONTRACT.json')
inventory=meta(P/'evaluation_preparation_20261022/PREPARATION_AUDIT.json')
source_bindings=meta(P/'evaluation_preparation_20261022/SOURCE_BINDINGS.json')
operator=meta(OLD/'current_evaluation_preparation/CURRENT_OPERATOR_REUSE_PROTOCOL.json')
prep=meta(OLD/'current_evaluation_preparation/PREPARATION.json')
infer_entry=meta(OLD/'ROOT_ACTUAL_C16_FROZEN_CURRENT_INFER_ENTRY.json')
eval_entry=meta(OLD/'ROOT_ACTUAL_C16_SAVED_CURRENT_EVALUATION_ENTRY.json')
c16seal=meta(OLD/'private_evaluation/C16_FROZEN_CURRENT223/INFERENCE_SEAL.json')
c11seal=meta(P/'query_visibility_spatial_target_query_features_20261023_v111/private_evaluation/C11_FROZEN_CURRENT223/INFERENCE_SEAL.json')
review=meta(OLD/'saved_current_review/REVIEW.json')
for path in [OLD/'current_evaluation_preparation/frozen_head_current.py',
             P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py',
             P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/infer_same_query_once.py',
             P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py',
             P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py',
             P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/current_adapter.py',
             P/'query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py',
             P/'evaluation_preparation_20261022/prepare_saved_evaluation.py',
             P/'near_error_diagnosis_20261022/real/audit_real_near.py']:
    code(path)

body_bindings=dict(
    current_feature_packet= infer_entry['C11_current_packet'],
    saved_C16_outputs=c16seal['outputs'],
    saved_C16_inference_seal=infer_entry['C11_inference_seal'] if False else eval_entry['inference_seal'],
    evaluation_manifest=eval_entry['bindings']['evaluation_manifest'],
    original_prior_rows=eval_entry['bindings']['prior_input_rows'],
    manual_coordinate_provenance=eval_entry['bindings']['manual_provenance'],
    original_v61_prediction=source_bindings['v61_outputs'],
    v61_legacy_same_query_saved_outputs=source_bindings['saved_outputs'],
    original_baseline_metric_replay=eval_entry['bindings']['original_baseline_metrics'],
    v61_freeze=source_bindings['v61_freeze'],
    legacy_descriptor=source_bindings['legacy_descriptor'])
assert c16seal['slots']==c11seal['slots']==1784 and c16seal['U']==c11seal['U']==45
assert inventory['frames']==223 and inventory['native_slots']==1784
assert review['PASS']and review['all_ID_qL_native_order_original_image_SHA_exact']

result=dict(schema='V143_EXISTING_CURRENT_OPERATOR_MAPPING_AND_BASELINE_METADATA_AUDIT_V1',completed=True,PASS=True,
 code=bind(__file__),prospective_policy=bind(POLICY),
 audit_scope=dict(code_contract_phase_summary_seal_metadata_only=True,current_RGB_feature_annotation_prediction_body_reads=0,
                 current_body_hash_reads=0,checkpoint_tensor_body_reads=0,new_model_forward_fit_PnP_calls=0,
                 forbidden_phase_access=0,private_data_open_authorized_by_this_audit=False),
 conclusion_KO='원래 V124의 input loader·전체1784 동시 head 계산·raw0/U-first decoder·매핑/분모/저장 baseline 평가 산술은 그대로 재사용할 수 있습니다. 후보 모델의 source authorization은 V124 C16 전용 guard와 분리해야 합니다. 실제 특징 호환성은 metadata만으로 확정할 수 없으며, 별도 root entry 후 원래 C16 selected4500 가중치로 저장 raw/safe/p-visible와 metadata를 byte parity 검증한 다음에만 C29/C32를 평가해야 합니다.',
 reusable_operator=dict(
  code=prep['code'],operator=prep['operator'],unchanged_function_names=['load_saved_inputs','predict_generated_or_actual','state_sha','pure_core'],
  input_packet_binding=infer_entry['C11_current_packet'],input_binding_inherited_only_not_read_or_hashed=True,
  features=dict(shape=[1784,1928],dtype='float32',finite_all=True,block_order='T/Q/F/C/P each384 followed8 descriptors; final1927=context_Q_inside',
                learned_input_fields=['features1928'],no_GT_ID_native_geometry_reference_input=True),
  metadata_whitelist=['query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside','query_xy','side_L','original_query_xy','original_side_L','id','native_corner'],
  original_context_box='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX; derived from initial native8, not selected detection/GT bbox',
  availability='valid=query_available=prior_valid AND context_available AND local_central4_available; Q_inside is descriptor only, no extra U gate',
  expected_original_valid=1739,expected_original_input_U=45,expected_extra_operator_U=0,
  q_L='original F64 metadata; feature q/L F32 casts checked equal; no correction/recentering/resizing',
  batch_math='ONE ALL1784 batch, repeat identical second call; never compact available rows or reuse source compact512 batching',
  runtime='Torch2.1.1+cu118 FP32, cuda, one thread, deterministic, TF32false, inference_mode/eval/no gradients',
  C16_original_head_parameters=221825,C16_selected_update=4500,C16_checkpoint=infer_entry['checkpoint'],
  candidate_head_parameters=224513,candidate_G42='all zeros, original C29/C32 source control, no new native geometry build',
  decoder='all raw logits finite; safe z=where(valid,raw,0) F32; p_visible=sigmoid(-z) display only; valid D iff z<0, E iff z>=0; invalid U first',
  exact_minimal_CSV_fields=['id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision']),
 source_authorization=dict(reuse_C16_source_guard_for_candidates=False,
  why='V124 source_guard is tied to C16 design, selected0..8000/100, C15 exact1000+7000 certificate and C16 freeze. C29/C32 retain their original failed source freezes/current_allowedfalse.',
  new_frozen_benchmark_policy_only=True,models_fixed=['C29','C32'],all_fixed_update8000_raw0=True,
  both_reported_even_if_worse=True,source_gate_waivers=0,selected_model=None,adopted_model=None,
  compatibility_gate='Original source-selected C16 checkpoint ee285b2b... at4500, not C16 FINAL8000, must byte-reproduce original C16 all1784 raw-before-U/safe-z/pv and metadata before either candidate call.'),
 annotation_and_mapping=dict(
  original_frames=223,WOOD105_slots=840,SQUARE118_slots=944,native_slots=1784,centroid8_excluded=True,
  frozen_mapping='Existing N3 branch/permutation native->human reference only for evaluation; original qFinal parity certificate. Never new GT-best permutation or physical native certificate.',
  original_mapping_status=inventory['frame_mapping_status'],unresolved_mapping_ids=inventory['unresolved_mapping_ids'],
  native_order=contract['native_slot_order'],group_original_inventory={k:inventory['groups'][k]['mapped_native_state_inventory']for k in inventory['groups']},
  groups=inventory['groups'],
  binary_target='Human DIRECT_VISIBLE vs EXTERNAL_OCCLUDED only. Human SELF_OCCLUDED,OUT_OF_FRAME,UNANNOTATED,UNKNOWN_MAPPING stay outside binary E and remain in full1784 ledger.',
  no_pixel_segmentation_gold=contract['pixel_presence_reference'],
  prior_source_missing='Wood13 NO_REFERENCE and square346 pnp_projected review-only slots; mapped square343 UNANNOTATED+8 UNKNOWN_MAPPING differs from source-only346 count. No conversion toD/E.'),
 cohort_contract=dict(
  paired_same_query_RGB=contract['cohorts']['paired_same_query_RGB'],
  same_slot_pipeline=contract['cohorts']['same_slot_pipeline'],
  primary_paired_from_existing_metadata=dict(WOOD=dict(D=616,E=29,N=645),SQUARE=dict(D=531,E=61,N=592),total=1237),
  same_slot_from_existing_review=dict(WOOD=dict(D=626,E=30,N=656),SQUARE=dict(D=531,E=61,N=592),total=1248),
  note='Full1784 inputU45 and primary paired D/E1237 inputU0 are different denominators. Preserve all original cohorts rather than filtering by new predictions/GTerror.',
  paired_transitions='Use identical original eligible row keys. Keep baseline-unresolved/new-unresolved and damage/recovery counts, no success-only subset.',
  full_denominator_reporting='In addition to original cohort CM2x3, report all1784/group inventory and original input U/mapping/annotation exclusions. Do not treat all1784 as labelled D/E.'),
 stored_baselines=dict(
  v61=dict(original_binding=source_bindings['v61_outputs'],saved_decision='present_at_fixed_0_5 unchanged',
           fixed_threshold=.5,meaning=contract['v61']['target'],new_weight_or_forward=False,
           full_inventory_guard='V72 call(v61) itself returnsD/E without invalid guard. Original cohorts are already valid-only; all1784 inventory must separately preserve original query_valid/missing asU rather than turning invalid False intoE.'),
  C16=dict(saved_outputs=c16seal['outputs'],inference_seal=eval_entry['inference_seal'],
           fixed_raw0=True,metadata_from_same_original_packet=True),
  real_CAL_legacy=dict(available_matched_stored_comparison=True,model_name='legacy',training=contract['legacy']['training'],
     fixed_historical_rules=contract['legacy'],state_decode='B/I=D;O=binaryhidden;U=ABSTAIN;S=OUT_OF_TASK_GEOMETRY_S inD_vsE;T/missing=OTHER_UNRESOLVED',
     geometry_S='Out-of-taskS is kept separately and aggregated into predictionU only for CM2x3. It is not externalE. No new geometric fitting or learnedself claim.',
     weight_or_fit_or_teacher_read=False),
  exact_historical_metric_replay='Reuse historical_replay on original v61/legacy and C16 counts before reporting candidates; do not recompute old threshold decisions from rounded probabilities.'),
 coordinate_diagnostics=dict(
  general_bins=['ALL_ORIGINAL_ELIGIBLE','LE3','GT3','MISSING_REFERENCE_ERROR'],
  general_error_scope='Original initial query to existing registered reference, not verified query ownership/miss truth. E reference coordinates remain estimated/unspecified.',
  manual_near_D='annotation_coordinate_source=manual_click AND humanD AND existingerror<=3 AND originalqueryvalid; historicalN153wood/223square. Supplemental diagnostic only, not evaluation admission/model selection.',
  manual_provenance_binding=eval_entry['bindings']['manual_provenance'],
  forbidden_uses=['GTerror primary filtering','q/L correction','GT-centered features/crops','GT permutation choice','threshold/checkpoint/model/data-source selection']),
 limitations_KO=[
  '현재223은 과거 반복 개발에 노출된 자료입니다. 평가 전용 사용을 유지해도 독립 blind test로 표현할 수 없습니다.',
  '기존 사람 라벨은 원래 reference 의미 가림이며 실사 단일픽셀 target mask gold가 아닙니다.',
  '등록 native 명칭/이전 GT평가 branch와 독립적인 물리적 대응 인증을 구분해야 합니다.',
  'C29/C32의 source 실패와 current_allowedfalse는 불변입니다. 이 별도 benchmark는 source 개선의 실사 전이를 측정하는 진단이며 모델 채택이나 새로운 허가의 소급 적용이 아닙니다.',
  '호환성은 실제 C16 저장출력 byte parity를 통과하기 전에는 미확정입니다. 실패하면 feature 재계산이나 숫자 보정 없이 종속 benchmark를 중단해야 합니다.'
 ],
 actual_body_bindings_inherited_only_not_opened_or_hashed=body_bindings,evidence=evidence)
path=H/'CONTRACT_AUDIT.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=True,PASS=True,path=str(path),sha256=sha(path),evidence=len(evidence),current_bodies_read_or_hashed=0)))
