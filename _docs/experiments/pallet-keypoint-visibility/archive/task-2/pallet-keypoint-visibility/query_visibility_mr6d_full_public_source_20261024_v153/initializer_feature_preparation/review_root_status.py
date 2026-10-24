"""Final status review from completed source receipts and prior aggregate only.
No private originals/prediction rows, features, pixels, weights or model calls.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
intake_path=V/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json';intake=read(intake_path)
r0_path=V/'actual_full230_initializer_v1/RESULT.json';r0=read(r0_path)
r0_seal_path=V/'actual_full230_initializer_v1/PREDICTION_SEAL.json';r0_seal=read(r0_seal_path)
feature_path=V/'full230_native_DINO_v1/FEATURE_COMPLETE.json';feature=read(feature_path)
qa_path=H/'FULL230_SAVED_FEATURE_REVIEW.json';qa=read(qa_path)
ind_r0_path=V/'independent_saved_pipeline_review/INITIALIZER_REVIEW.json';ind_r0=read(ind_r0_path)
quality_path=V/'head_preparation/saved_synthetic_point_quality_v1/RESULTS.json';quality=read(quality_path)
quality_review_path=H/'SEPARATED_HEAD_AND_QUALITY_REVIEW.json';quality_review=read(quality_review_path)
prototype_path=V/'head_preparation/SEPARATED_HEAD_PROTOTYPE_CHECKS.json';prototype=read(prototype_path)
old_path=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149/ROOT_RESULTS_KO.json';old=read(old_path)
for x in (intake,r0,r0_seal,feature,qa,ind_r0,quality_review):assert x['completed']and x['PASS']
assert quality['completed']and prototype['completed']and old['completed']
assert intake['frames']==230 and intake['rows']==1840 and intake['development_frames']==185 and intake['validation_frames']==45 and intake['GT_instance_observations']==303
assert intake['original_payload_files']==1067 and intake['canonical_RGB_arrays']==230
assert r0['frames']==230 and r0['native_slots']==1840 and r0['candidate_count']==532 and r0['no_detection_frames']==3
assert r0['finite_non_sentinel_slots']==1816 and r0['continuous_valid_slots']==1538 and r0['nearest_valid_slots']==1535 and r0['explicit_predict_calls']==232
assert feature['frames']==230 and feature['rows']==1840 and feature['features_shape']==[1840,1928]and feature['total_encoder_calls']==240
assert feature['feature_constructed']==1816 and feature['available']==1538 and qa['nearest_available']==1535
assert quality['total_rows']==26624 and quality['all_counts']==dict(unlabeled=21119,accurate1=5245,inaccurate0=260)
assert quality['all_original_queries_L_order_roles_and_visibility_masks_preserved']and quality['middle_error_rows_retained_no_visibility_loss_filter']
assert quality['real_MR6D_coordinate_accuracy_labels_created']==0
assert prototype['prototype_only']and prototype['parameters']==224643 and prototype['actual_training_updates']==0 and not prototype['parent_checkpoint_or_actual_features_read']
assert old['source_native35_pass_each']==22 and old['source_native35_total']==35 and old['source_qualification_unchanged']and old['selected_model']is None and not old['fully_solved']
models=[]
for model in ('v61','legacy','C16','C33','C34'):
 row=next(x for x in old['aggregate_primary']if x['model']==model);cm=row['CM_D_E_by_pred_D_E_U'];D,E=map(sum,cm);N=D+E
 assert(D,E,N)==(1147,90,1237)
 drec=cm[0][0]/D;erec=cm[1][1]/E;U=cm[0][2]+cm[1][2];ur=U/N
 assert math.isclose(drec,row['D_recall'])and math.isclose(erec,row['E_recall'])and math.isclose(ur,row['U'])
 models.append(dict(model=model,N=N,D_support=D,E_support=E,CM_D_E_by_pred_D_E_U=cm,D_recall=drec,E_recall=erec,U_count=U,U_rate=ur,D_recall_percent=100*drec,E_recall_percent=100*erec,U_percent=100*ur))
assert any('CAL46'in s and '중복'in s for s in old['limitations_KO'])
assert not old['current_used_for_training_or_threshold_checkpoint_model_choice']
report=dict(completed=True,PASS=True,created_UTC=datetime.now(timezone.utc).isoformat(),review_code=bind(__file__),goal_solved=False,selected_model=None,
 review_scope_KO='완료된 V153 공개 source 산출물·독립 검산 receipt와 V149의 기존 집계 JSON만 검토. 새 성능평가나 학습이 아니다.',
 source_data_progress=dict(frames230=True,original_payload_files=1067,provided_GT_instance_observations=303,development_frames=185,validation_frames=45,same_physical_object_possible_not_identity_holdout=True,official_blind_benchmark_claim=False,R0_original_queries=1840,raw_candidates=532,no_detection_frames=3,no_detection_query_rows=24,finite_constructed_queries=1816,continuous_valid=1538,nearest_valid=1535,all_queries_including_failures_retained=True,features_shape=[1840,1928],DINO_primary_calls=230,DINO_fixed10_replay_calls=10,DINO_total_calls=240,original_q_L_preserved=True,old10_R0_input_and_new_DINO_repeat_parity=True,additional_visibility_head_training_on_new230=0,new230_performance_improvement_demonstrated=False),
 separate_quality_and_prototype=dict(exact_synthetic_rows_preserved=26624,accurate1_LE3=5245,inaccurate0_GT20=260,unlabeled=21119,middle_errors_unknown_quality_only=True,visibility_training_not_filtered_by_GT_error=True,MR6D_pixel_accuracy_gold_created=0,prototype_parameters=224643,prototype_new_training_updates=0,outputs=['external_occlusion','target_entity','coarse_point_quality'],inference_fusion_or_abstention_policy_defined=False,prototype_performance_claim=False),
 latest_existing_current_aggregate=dict(source=bind(old_path),cohort='paired_same_query_RGB known D/E only',N=1237,D_support=1147,E_support=90,models=models,source_native35=dict(C33_pass=22,C34_pass=22,total=35,fail_each=13,source_current_allowed_false_unchanged=True),
 legacy_overlap_warning_KO='기존 legacy/CAL46은 평가 일부와 학습 중복이 있는 과거 참고다. 독립 일반화 성능의 공정한 상한이나 새 공개 source 모델의 성공 인증으로 쓰지 않는다.',development_exposed_not_blind=True,GT_error_only_diagnostic_not_filter=True,
 C34_vs_C33_KO='보임 25개 회복과 외부 가림 7개 손상이 함께 발생했다. 두 능력의 동시 개선 근거가 아니다.',C34_vs_C16_KO='보임 40개 손실, 외부 가림 1개 추가 검출이다. C16을 전반적으로 넘어선 모델로 채택할 수 없다.'),
 semantic_label_review_status=dict(status_at_review='ACTUAL_QUERY_RENDER_AND_SOURCE_VALIDATION_WEAK_LABEL_SEAL_IN_PROGRESS',semantic_truth_independent_visual_validation_claim=False,assistant_weak_labels_not_human_gold=True,planned_metadata_review_only='완료 seal 뒤 ID/split/count만 별도 확인 가능; 이 상태 검토는 새230개 시각 라벨의 정답성을 독립 검증했다고 주장하지 않는다.',wrong_selected_entity_never_automatic_E=True,nominal_pose_centimeter_uncertainty_not_3px_gold=True),
 conclusion_KO='원인 분해와 공개 source 데이터·실제 초기 query 연결에는 검증된 진전이 있다. 배포할 가림 모델의 성능 향상은 아직 증명되지 않았다. 새230장은 이제 입력과 feature가 완성된 단계이며, 새 학습·source 검증 전에는 성공을 주장할 수 없다. 목표는 아직 미해결이고 성공 확률을 수치로 보장할 근거도 없다.',
 safe_short_user_status_KO=['진전은 있습니다. 보임·외부가림·대상 개체 오류·좌표 품질을 섞어 판단하던 문제를 분리했고, 공개 실사230장과 원래1840개 query·feature를 실제로 준비했습니다.','다만 새230장으로 개선된 가림 모델을 학습·검증한 단계는 아닙니다. 기존 C34도 보임95.6%·외부가림40.0%로, C16의99.1%·38.9%를 동시에 개선하지 못했습니다.','따라서 방법을 좁히고 데이터 근거를 보강한 상태이며, 아직 해결이나 배포 성공을 확정할 수준은 아닙니다.'],
 access=dict(V149_existing_aggregate_summary_read=True,V149_original_current_labels_predictions_features_RGB_or_model_body_reads_or_hashes=0,V153_new_private_forward_training_or_evaluation_runs=0,V112_any_access=0,new_encoder_model_calls_during_status_review=0),
 bindings=dict(intake=bind(intake_path),R0_result=bind(r0_path),R0_prediction_seal=bind(r0_seal_path),independent_R0_review=bind(ind_r0_path),feature_complete=bind(feature_path),independent_feature_review=bind(qa_path),quality_result=bind(quality_path),independent_quality_review=bind(quality_review_path),prototype_checks=bind(prototype_path)))
out=V/'ROOT_STATUS_INDEPENDENT_REVIEW.json'
with out.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(out),models=models),ensure_ascii=False))
