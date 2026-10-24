"""Record independent actual fullscene viewing; no labels/queries are mutated."""
from pathlib import Path
import hashlib
import json

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_native_initializer_association_20261024_v141')
H=Path(__file__).resolve().parent
S=V/'saved_public_association_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
seal=json.loads((S/'ASSOCIATION_COMPLETE.json').read_text())
assert seal['completed']and seal['PASS']and sha(S/'ASSOCIATION_COMPLETE.json')=='e54c1cfe8a533fe9c3d38c7db58a70f8ffb08b585cda7340b10915d13b01af71'
assert sha(seal['results']['path'])==seal['results']['sha256']
source=json.loads(Path(seal['results']['path']).read_text())
selected={r['frame_index']:r for r in source['frame_diagnostics']if r['frame_index']in(10,12,13,14,15)}
assert len(selected)==5

frame_findings={
10:dict(entity='DIFFERENT_ENTITY',extent='NOT_APPLICABLE_UNMATCHED_ENTITY',
 evidence_KO='왼쪽 원본과 오른쪽 표시를 실제 확인했습니다. R1/R2:D는 앞쪽 높은 종이상자 아래 팔레트 ann371955에 붙어 있습니다. 첫 argmax의 작은 주황 박스와 Q0..7은 화면 뒤쪽 작은 팔레트/받침 구조를 가리킵니다. 같은 슬롯 번호와 큰 좌표 차이를 대응 근거로 쓸 수 없습니다.',
 coordinate_interpretation_KO='216.6/236.6px는 서로 다른 대상의 같은 슬롯 차이입니다. 해당 팔레트의 순수 좌표 오차로 해석하거나 D를 E로 바꾸지 않습니다.'),
12:dict(entity='SAME_INTEGRATED_WOODEN_CRATE_BASE_PLAUSIBLE',extent='SELECTED_LOWER_BASE_DIFFERS_FROM_FULL_HIGH_CRATE',
 evidence_KO='목재 상자와 그 일체형 하부 받침이 보이고, 선택 박스/Q8은 상자 전체보다 낮은 베이스 범위를 일관되게 형성합니다. Q0/Q1은 하부 전면 상단, Q4/Q5는 하부의 뒤쪽 상단에 놓이며, R0/R1은 상자 윗테두리입니다. Q2/Q3은 전면 하단 모서리 역할로 보여 기존 R2/R3와 공유할 가능성이 있습니다.',
 coordinate_interpretation_KO='상단의 약78px 차이는 단순 잡음보다 높이/구조 범위 정의 차이와 일치합니다. 하단2의10.7px, 하단3의4.1px 차이는 공유 바닥 역할을 가정할 때 좌표 오차와 source virtual-reference/registration 편차가 함께 포함될 수 있습니다. 거리 임계값으로 D/E를 바꾸거나 제외하지 않습니다.'),
13:dict(entity='SAME_INTEGRATED_WOODEN_CRATE_BASE_PLAUSIBLE',extent='SELECTED_LOWER_BASE_DIFFERS_FROM_FULL_HIGH_CRATE',
 evidence_KO='선택 박스/Q8은 윗테두리보다 아래의 베이스 높이 구간에 놓입니다. R0/R1:D는 높은 목재 상자의 윗테두리 역할이고 Q0/Q1은 하부 전면 상단입니다. 오른쪽 하단 Q2와 R2는 같은 하부 외곽 역할이 가능하고 왼쪽 하단 Q3/R3는 모두 종이상자에 가려진 전면 하단 역할로 해석할 수 있습니다.',
 coordinate_interpretation_KO='상단의 약130px 차이는 베이스/전체 상자 높이 역할의 불일치와 일관됩니다. 하단2의19.3px 차이를 이유로 D를 E/S로 바꾸지 않습니다. 하단3의4.0px 차이도 가림 라벨의 자동 증명은 아닙니다.'),
14:dict(entity='DIFFERENT_ENTITY_OR_BACKGROUND_STRUCTURE',extent='SELECTED_TINY_BACKGROUND_REGION_NOT_CRATE',
 evidence_KO='선택 주황 박스와 Q8이 목재 상자 왼쪽의 작은 배경/지지 구조 부근에 몰려 있습니다. 오른쪽 큰 목재 상자의 R0/R4 윗테두리와 종이상자가 가리는 R3 하단을 선택 대상으로 볼 근거가 없습니다. 정확히 어떤 작은 물체인지는 확정하지 않습니다.',
 coordinate_interpretation_KO='큰 q-reference 차이는 먼저 대상 연결 실패로 분류합니다. 목재 상자 위의 같은 물리 모서리를 크게 잘못 찍은 사례라고 확정하지 않습니다.'),
15:dict(entity='SAME_CRATE_LOWER_BASE_PLAUSIBLE_BUT_NOT_CERTIFIED',extent='LOWER_PARTIAL_BASE_NOT_FULL_CRATE',
 evidence_KO='선택 박스와 Q8은 큰 상자의 낮은 부분 및 종이상자가 가린 베이스를 모델링하는 모양이며 일부 q는 화면 밖입니다. R0:D는 높은 상자 윗테두리라 Q0의 하부 전면 상단과 역할이 다릅니다. R3:E와 Q3은 전면 왼쪽 하단을 가리킬 가능성이 있고 모두 종이상자 쪽에 있지만, 가려진 바닥 footprint와 역할 연결은 이 영상만으로 강하게 확정하기 어렵습니다.',
 coordinate_interpretation_KO='상단0의244.4px는 먼저 구조 범위 차이로 봅니다. 하단3의40.9px는 같은 바닥 모서리라는 조건을 유지하면 큰 위치 오차/등록 편차 후보이며, 그 크기 때문에 E를 D/S로 바꾸거나 행을 제거하지 않습니다. 다른 모서리를 잡았을 가능성도 남습니다.')
}

decisions=[]
frames=[]
for i,f in selected.items():
    assert sha(f['figure']['path'])==f['figure']['sha256']
    frames.append(dict(frame_index=i,id=f['id'],source_id=f['source_id'],figure=f['figure'],
        personally_viewed_raw_and_overlay_fullscene=True,selected_index_unchanged=f['selected_index'],
        selected_bbox_xyxy=f['selected_bbox_xyxy'],**frame_findings[i]))
    for q in f['queries']:
        r=q['frozen_source_proposal']
        if r is None or r['source_reference_state']=='U':continue
        k=q['native_slot'];state=r['source_reference_state']
        if i in (10,14):
            category='REJECT_DIFFERENT_ENTITY'
            candidate=None
            why='기존 source 라벨 대상과 실제 첫 argmax 대상이 다릅니다. source 의미 라벨은 원래 대상에 유지하며 actual q의 다른 대상에 복사하지 않습니다.'
        elif k in (0,1):
            category='REJECT_FULLCRATE_UPPER_ROLE_TO_BASE_UPPER_ROLE'
            candidate=None
            why='같은 일체형 물체일 수 있어도 source는 전체 높은 상자 윗테두리, 실제 q는 낮은 베이스 상단 역할입니다. 둘의 보임 상태를 같은 native 라벨로 이전하지 않습니다. 큰 좌표 차이만 보고 라벨을 바꾸는 판단과 구분합니다.'
        elif i in (12,13)and k in(2,3):
            category='CONDITIONAL_SHARED_BOTTOM_ROLE_CANDIDATE'
            candidate=state
            why=('전면 오른쪽 하단은 보이는 같은 일체형 베이스 외곽 역할이 가능하므로 source D를 조건부 후보로 유지합니다.'if k==2 else'전면 왼쪽 하단이 같은 일체형 베이스 footprint 모서리라는 조건에서는 종이상자의 외부 가림 E가 q 주변에서도 의미상 유지됩니다.')+' 다만 전체 상자와 base의 정확한 공통 footprint/native 역할 계약과 provisional source 등록을 승인하기 전에는 학습 라벨로 직접 채택하지 않습니다.'
        else:
            assert i==15 and k==3 and state=='E'
            category='UNRESOLVED_SHARED_BOTTOM_ROLE_WITH_LARGE_QUERY_DISPLACEMENT'
            candidate=state
            why='같은 전면 왼쪽 바닥 역할을 가정하면 외부 종이상자 가림 E는 유지되고40.9px의 위치 차이가 별도로 존재합니다. 하지만 가려진 footprint와 R0의 부분 base 대상 범위가 불확실하므로 즉시 이전하지 않습니다. 거리로 E를 D/S로 바꾸거나 이 행을 버리지 않습니다.'
        decisions.append(dict(frame_index=i,source_id=f['source_id'],native_slot=k,
            original_source_semantic_state=state,source_reference_xy=r['source_reference_xy'],
            actual_query_xy_unchanged=q['query_xy'],actual_input_L_unchanged=q['L'],
            actual_input_continuous_valid=q['native_continuous_valid'],
            same_slot_displacement_px_diagnostic=q['same_native_slot_distance_px_diagnostic'],
            classification=category,conditional_candidate_state_if_native_role_confirmed=candidate,
            direct_transfer_to_training_now=False,actual_query_semantic_state_remains='U',
            reason_KO=why,semantic_relabel_due_to_coordinate_error=False,
            query_or_reference_correction=False,drop_from_original128=False,
            human_gold=False,trainready=False))

assert len(decisions)==15
counts={}
for r in decisions:counts[r['classification']]=counts.get(r['classification'],0)+1
assert counts==dict(REJECT_DIFFERENT_ENTITY=5,REJECT_FULLCRATE_UPPER_ROLE_TO_BASE_UPPER_ROLE=5,
    CONDITIONAL_SHARED_BOTTOM_ROLE_CANDIDATE=4,UNRESOLVED_SHARED_BOTTOM_ROLE_WITH_LARGE_QUERY_DISPLACEMENT=1)
result=dict(schema='INDEPENDENT_SOURCE_ENTITY_EXTENT_AND_NATIVE_ROLE_VISUAL_REVIEW_V1',completed=True,
 source_only=True,code=bind(__file__),association_seal=bind(S/'ASSOCIATION_COMPLETE.json'),
 source_diagnostics=seal['results'],viewed_frames=frames,reviewed_existing_D_E_proposals=15,
 original_source_proposal_states=dict(D=11,E=4),classification_counts=counts,
 conditional_shared_bottom_candidates=dict(LAB_source_D=2,LAB_source_E=2,HALL_unresolved_E=1),
 direct_trainready_transfers=0,all_original_queries_preserved=128,original_query_semantic_states_unchanged=dict(U=128),
 decisions=decisions,
 conclusion_KO='현재15개 source D/E 제안을 actual q에 일괄 이전할 수 없습니다.5개는 다른 대상,5개는 높은 상자 윗테두리와 베이스 상단의 역할 차이입니다. Lab 하단4개는 공통 바닥 역할을 명시적으로 확인하면 D2/E2 조건부 후보가 됩니다. Hall 하단1개는 같은 역할 가정 아래 E를 유지한 큰 위치 차이 후보지만 연결 불확실성이 더 큽니다. 이 판단은 source 의미를 좌표 거리로 바꾸거나 실패 행을 제거한 결과가 아닙니다.',
 interpretation_limits_KO=[
 '선택 박스와 q8의 배치에서 낮은 base를 대상으로 본 흔적은 강하지만, 모델의 내부 의도를 직접 증명한 것은 아닙니다.',
 '원래 source32의 조건부 축 등록/virtual reference 의미와 surface·픽셀 지지 차이는 계속 유지됩니다. 동일 물체라는 사실만으로 전체 상자와 base의 모든 native 역할이 같아지지 않습니다.',
 '기존 source D/E는 원래 대상으로 유지했습니다. 거리만으로 D/E/S를 재정의하지 않았고 S를 새로 만들지 않았습니다.',
 'Lab bottom 역할의 조건부 호환성은 원래 source 제안과 물체 구조를 근거로 한 검토입니다. 단순히 가까운 q를 골라 대응시키거나 GT 순열을 선택하지 않았습니다.',
 '공개 실사5장 검토는 현재223 평가 개선이나 학습 성공의 증거가 아닙니다. 별도 prospective source 계약 전에는 trainready=false입니다.'
 ],no_nearest_mapping_or_GT_permutation=True,thresholds_selected=0,
 original_labels_queries_figures_edited=False,new_model_head_fit_private_V112_PnP_calls_or_reads=0,
 actual_source_fullscene_images_viewed=5)
path=H/'SOURCE_LABEL_TRANSFER_VISUAL_REVIEW.json'
assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=True,path=str(path),sha256=sha(path),reviewed=15,classification_counts=counts)))
