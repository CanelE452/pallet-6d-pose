"""Reviewer's sealed after observations from the fixed pair and eight regions."""
from pathlib import Path
import hashlib,json,collections
R=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def write(p,o):
    with p.open('x') as f:json.dump(o,f,indent=2,ensure_ascii=False);f.write('\n')
policy=read(R/'AFTER_POLICY_BEFORE_RGB.json')
assert bind(Path(policy['bindings']['root_union']['path']))==policy['bindings']['root_union']
union=read(Path(policy['bindings']['root_union']['path']))
display=read(R/'PAIR_REGION_DISPLAY_RECEIPT.json')
VISIBLE='APPARENT_SAME_CUE_VISIBLE_AI'
COVER='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI'
findings={
 'V84_A_CUE_00':(VISIBLE,'최상단 목재 plank의 오른쪽 윗 윤곽/세로 끝이 after에도 같은 고정 영역에서 보입니다. 초록 장비와의 경계 및 인접 목재 면을 계속 구분할 수 있습니다. Cardboard는 이 상부 오른쪽 영역을 덮지 않습니다.'),
 'V84_A_CUE_01':(VISIBLE,'최상단 plank의 왼쪽 윗선과 세로 목재 끝이 검은 배경 앞에 계속 보입니다. 새 cardboard의 위쪽 끝은 이 cue보다 훨씬 아래에 있어 고정 영역의 목재 junction을 덮지 않습니다.'),
 'V84_A_CUE_02':(COVER,'before의 왼쪽 base-runner 돌출 끝/상부 part-line은 after에서 보이지 않고, 고정 불확실 영역 전체가 새 cardboard의 갈색 면 안에 놓입니다. 전체 after 장면에서 큰 별도 cardboard가 목재 앞을 가로막고, 목재 윤곽이 그 경계에서 끊기는 맥락을 확인했습니다. 단순한 cue 부재나 mask0 판정이 아닙니다.'),
 'V84_A_CUE_03':(VISIBLE,'오른쪽 하단 support/base-runner의 노출 끝과 바닥 쪽 윤곽이 after에도 계속 보입니다. Cardboard는 반대쪽 왼쪽에 있으며 이 고정 범위에 foreign blocker가 관찰되지 않습니다.'),
 'B01':(VISIBLE,'A의 상부 오른쪽 제안과 연결된 중복 영역입니다. B의 원래 y182와 별도 bbox를 움직이지 않고 보아도 top-plank 윗선/오른쪽 끝이 after에 남습니다. 두 리뷰를 독립 capture로 세지 않습니다.'),
 'B02':(VISIBLE,'A의 상부 왼쪽 제안과 연결된 중복 영역입니다. B의 원래 y185/bbox를 그대로 유지했을 때도 목재 상부 외곽과 왼쪽 끝이 after에 관찰됩니다. 새 cardboard가 이 상단을 덮지 않습니다.'),
 'B03':(COVER,'before의 가장 낮은 넓은 front plank 왼쪽 끝은 after에서 cardboard의 면 뒤로 가려집니다. B의 고정 x76..84/y415..423 불확실 영역 전체에 갈색 cardboard가 보이고, 이 별도 물체가 목재 앞을 차단하는 full-scene 맥락이 있습니다. A의 더 낮은 foot cue와는 다른 목재 part-line입니다.'),
 'B04':(VISIBLE,'가장 낮은 넓은 front plank의 오른쪽 끝과 아래쪽 경계가 after에도 보입니다. 아래 support와의 구조 관계도 유지되며 cardboard는 이 오른쪽 끝을 덮지 않습니다.')}
records=[]
for cue,shown in zip(union['proposals'],display['records']):
    ident=cue['union_id'];state,evidence=findings[ident]
    assert ident==shown['union_id']
    assert cue['reference_xy_unchanged']==shown['r_unchanged']
    assert cue['uncertainty_bbox_xyxy_half_open_unchanged']==shown['uncertainty_unchanged']
    records.append(dict(union_id=ident,before_reviewer=cue['reviewer'],reference_xy_unchanged=cue['reference_xy_unchanged'],uncertainty_bbox_unchanged=cue['uncertainty_bbox_xyxy_half_open_unchanged'],before_cue_description_unchanged=cue['cue_description_unchanged'],after_state=state,actual_RGB_evidence_KO=evidence,
        foreign_opaque_blocker=('LARGE_CARDBOARD_IN_FRONT_OF_WOODEN_STRUCTURE' if state==COVER else 'NONE_OBSERVED_AT_THIS_CUE'),
        apparent_entire_fixed_uncertainty_region_covered=(True if state==COVER else False),
        foreground_cover_evidence_is_AI_visual_not_alpha_pixel_certificate=True,
        reference_definition_uncertainty_preserved=True,coordinate_relocalized=False,region_resized=False,
        semantic_native8_id=None,native8_status='UNKNOWN',human_gold=False,official_gold=False,training_admitted=False,
        local_registration_note_KO='같은 좌표 crop에서 목재/장면의 대응은 시각적으로 그럴듯합니다. 작은 camera/target 움직임이나 sensor/exposure 차이를 정량 배제하지 못하며, 원 r가 exact material pixel이라는 주장은 하지 않습니다.',
        fixed_region_display=shown['paired_figure'],unmarked_pair_crops=shown['unmarked_crops']))
counts=dict(collections.Counter(x['after_state'] for x in records))
assert counts=={VISIBLE:6,COVER:2}
report=dict(schema='V84_REVIEWER_A_AFTER_FIXED_REGIONS_EVIDENCE_V1',completed=True,
 status='SOURCE_PAIRED_PART_CUE_WEAK_EVIDENCE_ONLY_NO_TRAINING_ADMISSION',pair_id=union['pair_id'],records=records,
 row_counts=counts,retained_proposal_rows=8,distinct_physical_regions_from_root_before_union=6,
 duplicate_groups_link_only=union['possible_duplicate_groups_before_only'],
 qualitative_distinct_region_counts=dict(apparent_visible=4,foreign_blocker_covered=2,partial_or_unclear=0,registration_or_movement_U=0),
 full_scene_observation_KO='After에는 큰 별도 cardboard가 영상 왼쪽 foreground에 추가되어 목재 구조의 왼쪽 중하단을 가립니다. 위/오른쪽 목재 영역과 배경의 큰 구조는 before와 대체로 같은 구성으로 보입니다. 왼쪽 두 낮은 단서의 사라짐에는 명시적인 foreign foreground blocker 맥락이 있습니다.',
 registration_assessment=dict(gross_camera_or_target_movement_observed=False,pixel_level_registration_certified=False,small_shift_or_capture_difference_not_excluded=True,numerical_registration_or_warp_calls=0,
    explanation_KO='Full-scene와 고정 영역을 눈으로 비교했을 때 같은 목재의 윗부분/오른쪽 contour와 배경의 큰 배치는 대체로 유지되어 apparent same-cue 비교가 가능합니다. 이 판단은 정량 정렬 검증이 아니며 동일 supplied T를 subpixel registration 증명으로 쓰지 않았습니다. occluded 부분의 동일 pixel correspondence는 직접 볼 수 없어 source weak evidence의 한계로 남깁니다.'),
 source_identity_scope_KO='이 한 pair의 동일 front wooden structure 내 physical part-cue 증거입니다. 원래 virtual native8 corner/사람 pallet semantic D/E/S/실제 입력 query association의 정답으로 바꾸지 않습니다.',
 limitation_KO='불확실 bbox는 점/선 정의의 불확실 영역이며 모든 pixel의 목재 가시성 주장이 아닙니다. Cover 두 영역은 after의 별도 cardboard 면이 전체 범위를 차지하는 AI 시각 증거이나 정확한 alpha mask/scene-depth/human certificate가 아닙니다. 자연 semantic S는 평가하지 않았습니다. 한 pair와 중복 annotation을 독립 VAL/추가 capture로 세지 않습니다.',
 all8_full_context_and_native_pair_crops_directly_viewed=True,root_union_coordinates_and_regions_unchanged=True,
 B_after_judgments_read_before_this_seal=False,after_only_same_fixed_frame=True,
 bindings=dict(policy=bind(R/'AFTER_POLICY_BEFORE_RGB.json'),root_union=policy['bindings']['root_union'],own_before_seal=policy['bindings']['own_before_seal'],pair_RGB_verification=bind(R/'PAIR_RGB_VERIFICATION.json'),display=bind(R/'PAIR_REGION_DISPLAY_RECEIPT.json'),code=bind(Path(__file__))),
 mask_depth_pose_K_model_otherreview_current_reads=0,new_model_forward_fit_composites_PnP=0,
 stop_for_root_cross_review=True,training_admitted=False)
write(R/'AFTER_CUE_EVIDENCE_REVIEW.json',report)
write(R/'AFTER_VISUAL_QA.json',dict(completed=True,PASS=True,full_after_raw_and_pair_overview_directly_viewed=True,all8_individual_fourpanel_paired_region_images_directly_viewed=True,unmarked_before_after_and_fixed_r_uncertainty_both_shown=True,all8_order_retained_and_no_coordinate_or_region_changes=True,foreign_cardboard_vs_remaining_wood_evidence_not_mask0=True,pixel_registration_not_certified=True,independent_B_after_judgments_unread=True,report=bind(R/'AFTER_CUE_EVIDENCE_REVIEW.json'),new_forward_fit_current_PnP=0))
files=sorted(x for x in R.iterdir() if x.is_file())
write(R/'AFTER_REVIEW_SEAL.json',dict(completed=True,PASS=True,phase='ALL8_AFTER_EVIDENCE_SEALED_STOP_NO_LABEL_OR_FIT_ADMISSION',files=[bind(p) for p in files],report=bind(R/'AFTER_CUE_EVIDENCE_REVIEW.json'),policy=bind(R/'AFTER_POLICY_BEFORE_RGB.json'),root_union=policy['bindings']['root_union'],own_before_seal=policy['bindings']['own_before_seal'],raw_RGB_unique_pair_files=2,mask_depth_pose_otherreview_model_current_reads=0,new_model_forward_fit_composites_PnP=0,stop_now=True))
print(json.dumps(dict(report=bind(R/'AFTER_CUE_EVIDENCE_REVIEW.json'),seal=bind(R/'AFTER_REVIEW_SEAL.json'))))
