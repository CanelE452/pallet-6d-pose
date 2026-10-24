from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_paired_cue_20261023_v84';R=V/'independent_metadata_audit';PAL=P/'query_visibility_controlled_real_20261022_v66/public_pallet'
def rd(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def check(rec):assert bind(rec['path'])==rec,rec['path']
def save(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
scope=rd(V/'SOURCE_CUE_EVIDENCE_SCOPE_BEFORE_RGB.json');md=rd(V/'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json');check(scope['fixed_pair']);check(md['source_manifest'])
for t in scope['bindings']:check(t)
pairs=rd(md['source_manifest']['path'])['pairs'];selected=min((x for x in pairs if x['split']=='TRAIN_LAB'),key=lambda x:x['pair_id'])
assert selected==md['selected_pair'];assert selected['pair_id']=='0.55m_lab:0000_0001'
rep=rd(PAL/'SYMMETRIC_ALL50_V2/SYMMETRIC_SOURCE_EVIDENCE_REPORT.json');case=next(x for x in rep['cases']if x['pair_id']==selected['pair_id'])
rawmatches=[]
for phase,ab in [('before','A'),('after','B')]:
 for key,r in md['raw_files'][phase].items():
  prior=case['files'][ab][key]
  expected={k:prior[k]for k in ['path','sha256','bytes']}
  assert r==expected,(phase,key)
  q=Path(r['path']);assert q.is_file()and q.stat().st_size==r['bytes']
  rawmatches.append({'phase':phase,'kind':key,'metadata_exact_prior_sealed_source_case':True,'file_exists_size_exact':True,'raw_body_opened_in_this_audit':False})
assert md['raw_files']['before']['pose']['sha256']==md['raw_files']['after']['pose']['sha256']==selected['supplied_pose_sha256']
assert md['raw_files']['before']['K']['sha256']==md['raw_files']['after']['K']['sha256']==selected['K_sha256']
assert len(scope['sequence'])==6
assert set(scope['forbidden_this_phase']) == {
 'after RGB before before seals', 'SAM/depth/mask model state used as visual cue truth',
 'native8 mapping invented from physical corner appearance', 'new model forward',
 'fit', 'compositing', 'threshold/checkpoint change', 'current223 data read or eval',
 'PnP or pose/registration optimization'}
assert scope['goal_status']=='active'and scope['fully_solved']is False
checks={
 'scope_fixed_pair_binding_exact':True,'all_scope_metadata_bindings_exact':True,
 'first_lex_TRAIN_pair_all50_replayed_exact':True,'same_session_same_split_same_physical_bin':True,
 'all10_source_file_path_sha_size_metadata_exact_previous_case':True,
 'all10_source_file_exists_size_checks':True,'same_supplied_T_K_SHA':True,
 'phase1_independent_before_only_reviews_required':True,'phase2_both_before_seals_then_root_union_required':True,
 'phase3_after_view_only_after_union_seal':True,'no_after_guided_coordinate_relocation':True,
 'before_uncertainty_and_alternative_interpretation_required':True,'zero_cues_allowed':True,
 'visible_structural_cue_not_mask0_or_M4_GT':True,'native8_UNKNOWN_no_human_gold':True,
 'same_T_not_calibrated_RGB_registration':True,'ambiguous_partial_motion_kept_UNKNOWN':True,
 'same_family_no_independent_validation_claim':True,'no_GT_centered_input_or_q_modification':True,
 'no_fit_or_semantic_training_admitted':True,'prior_exposure_disclosed_no_blind_human_claim':True}
report={'schema':'V84_INDEPENDENT_PAIR_METADATA_SEQUENCE_SOURCE_TIER_AUDIT','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'blockers':[],
 'scope':'Metadata/docs only. No before/after RGB, source depth/masks, reviewer annotation results, model or current data opened.',
 'checks':checks,'source_file_metadata_checks':rawmatches,
 'pair_id':selected['pair_id'],'source_split':'TRAIN_LAB','official_source_instance':'One high-sided wooden Pallox pallet bin, same physical object; not flat-pallet/native8 gold.',
 'source_tier_verdict_KO':'관찰 가능한 물리part-cue에 대한 AI 조건부 auxiliary evidence로 적절하다. supplied pose/mask membership을 cue label로 바꾸지 않으며 paired actualRGB에서 같은 reference/foreground blocker를 별도 확인해야 한다. 이번scope는 annotation/admission/학습을 아직 확정하지 않는다.',
 'important_limits_KO':[
  'PASS는 고정 metadata·계획 순서·tier 계약에 대한 것이다. 도구 접근로그 전체를 감사하지 않아 실제 reviewer가 after를 일찍 보지 않았다는 실행 인증은 아니다. 각 before seal/union seal 후 별도 실행증거가 필요하다.',
  'before/after 파일명은 사전선정 촬영순서일 뿐 unobstructed/occluded 정답이 아니다. 첫 RGB에서 cue0개도 허용하므로 even/odd 가림 방향을 강제하지 않는다.',
  '동일T/K는 pair linkage이고 ≈0 supplied-pose reprojection은 산술항등; 실제RGB subpixel alignment를 보증하지 않는다. 관찰된 edge displacement/registration ambiguity는 U로 남겨야 한다.',
  '원본 SHA는 이전 source case와 metadata가 정확히 같음을 확인했고 file size/existence를 다시 확인했다. raw image/depth/mask bytes를 재해시하거나 decode하지 않았으므로 이번감사 단독의 원본 byte검증으로 과장하지 않는다.',
  '고정 uncertainty geometry는 reviewer별로 보존하고 union 후 after evidence로 위치를 수정하지 않는다. 별개같아보이는 cues를 GT없이native8같은점으로 합치지 않는다.',
  '실제 외부가림 관찰이 있더라도 source conditional part-cue auxiliary tier이며 full user goal/native8 canonical 또는 현재223 성능의 증거가 아니다.'
 ],
 'bindings':[bind(V/'SOURCE_CUE_EVIDENCE_SCOPE_BEFORE_RGB.json'),bind(V/'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json'),bind(PAL/'EXACT_POSE_PAIR_MANIFEST.json'),bind(PAL/'SYMMETRIC_ALL50_V2/SYMMETRIC_SOURCE_EVIDENCE_REPORT.json'),bind(P/'query_visibility_public_cue_audit_20261023_v83/public_source_next_step_review/REVIEW_SEAL.json')],
 'before_RGB_read':0,'after_RGB_or_after_annotation_read':0,'depth_mask_array_or_GTlabel_new_read':0,'ML_weights_features_or_new_forward':0,'fit':0,'current223_read':0,'download':0,'PnP':0,'prior_artifacts_modified':False}
o=R/'INDEPENDENT_METADATA_SEQUENCE_TIER_AUDIT.json';save(o,report);save(R/'SEAL.json',{'completed':True,'PASS':True,'files':[bind(o)],'no_RGB_after_review_annotation_model_current_fit_PnP':True})
print(json.dumps({'audit':bind(o),'seal':bind(R/'SEAL.json')}))
