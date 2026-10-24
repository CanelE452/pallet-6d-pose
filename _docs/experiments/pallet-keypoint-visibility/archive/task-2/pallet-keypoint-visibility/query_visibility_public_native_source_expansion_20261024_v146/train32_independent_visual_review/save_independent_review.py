"""Serialize independently viewed TRAIN32 weak proposals; no model or image reads.

States below were entered after viewing all 32 sealed whole/crop sheets. This
script only joins immutable contact metadata and hashes the already viewed PNGs.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CONTACT = ROOT / 'saved_source64_contact_v1'

def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def save(name, obj):
    path = OUT / name
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return binding(path)

# frame: entity association, lower2, lower3, visual evidence.
# MATCHED_PARTIAL means the selected amodal box belongs to the cut-off bin;
# it does not assert accurate query coordinates or an exact physical vertex.
NOTES = {
 0: ('MATCHED', 'D', 'D', '오른쪽 먼 목재 bin의 하부를 선택했다. 양쪽 하부 끝과 바닥 접촉부가 드러나며 전경 물체가 가로막지 않는다.'),
 1: ('MATCHED', 'D', 'D', '중앙 목재 bin의 하부를 선택했다. 양쪽 전면 하단 지지부와 바닥판 윤곽이 분리되어 보인다.'),
 2: ('WRONG_SELECTED_ENTITY', 'U', 'U', '선택 box와 원래 crop은 왼쪽 노란 안전봉의 검정 받침대를 가리킨다. 제공 mesh에 해당하는 목재 bin은 오른쪽에 별도로 있다.'),
 3: ('MATCHED_PARTIAL', 'U', 'E', '크게 잘린 목재 bin 하부의 amodal 선택으로 보인다. 오른쪽 하단 역할은 화면 밖이다. 왼쪽 하단 역할은 bin 앞의 큰 cardboard가 가로막는다. q3의 cardboard 색만으로 정한 E가 아니다.'),
 4: ('MATCHED', 'D', 'E', '먼 목재 bin 하부 선택. 오른쪽 하단 지지부는 드러나고 왼쪽 하단 위치까지 이어지는 구조는 bin 앞 cardboard 뒤에 가려진다.'),
 5: ('MATCHED', 'U', 'U', '시각적으로는 목재 bin을 선택했지만 제공 T의 SE3 조건 실패로 lower-role 연속성 UNRESOLVED이다. 외관만으로 native2/3 D/E를 강제하지 않는다.'),
 6: ('MATCHED', 'D', 'E', '목재 bin 하부 선택. 오른쪽 하단 지지부/바닥판은 보이고 왼쪽은 앞쪽 cardboard가 가로막는다.'),
 7: ('MATCHED', 'D', 'E', '작은 목재 bin 하부 선택. 오른쪽 하단은 드러나고 왼쪽 전면 하단은 앞 cardboard에 가려진다.'),
 8: ('MATCHED_PARTIAL', 'U', 'D', '오른쪽으로 잘린 큰 목재 bin 하부를 선택한 것으로 보인다. 오른쪽 역할은 화면 밖이나 왼쪽 하단 목재 지지부와 바닥판 끝은 보인다. 원래 q3 crop은 바닥으로 빗나가며, 이 좌표 오차를 E로 바꾸지 않는다.'),
 9: ('MATCHED_PARTIAL', 'U', 'D', '오른쪽으로 잘린 목재 bin 하부 선택. 오른쪽 하단 역할은 화면 밖이고 왼쪽 하단 지지부는 장애물 없이 보인다.'),
10: ('WRONG_SELECTED_ENTITY', 'U', 'U', '선택 box와 crop은 노란 안전봉 받침대를 가리킨다. 참조 목재 bin/cardboard 묶음은 장면 오른쪽에 별도로 있다.'),
11: ('MATCHED', 'D', 'D', '중앙 먼 목재 bin 하부 선택. 두 전면 하단 끝과 바닥판이 보이며 외부 가림이 없다.'),
12: ('MATCHED', 'D', 'D', '중앙 목재 bin 하부 선택. 양쪽 지지부와 바닥판 외곽이 드러난다.'),
13: ('MATCHED', 'D', 'D', '중앙 작은 목재 bin 하부 선택. 양쪽 전면 하단 지지부가 드러나고 앞쪽 장애물이 없다.'),
14: ('MATCHED_PARTIAL', 'U', 'D', '오른쪽으로 잘린 목재 bin 하부를 선택한 것으로 보인다. 오른쪽 하단 역할은 화면 밖이고 왼쪽 지지부/바닥판 끝은 드러난다. q3는 왼쪽 바닥으로 빗나갔지만 구조적 역할 보임과 분리한다.'),
15: ('MATCHED', 'D', 'E', '목재 bin 하부 선택. 오른쪽 하단은 보이며 왼쪽 하단을 전경 cardboard가 가로막는다.'),
16: ('MATCHED', 'D', 'D', '먼 목재 bin 하부 선택. 양쪽 하부 끝이 드러나며 앞쪽 장애물이 없다. 작은 원해상도 때문에 정확한 pixel/vertex gold로 보지 않는다.'),
17: ('MATCHED_PARTIAL', 'U', 'D', '오른쪽으로 잘린 큰 목재 bin 하부의 amodal 선택으로 보인다. 오른쪽 하단 역할은 화면 밖이고 왼쪽 하단 지지부와 바닥판은 드러난다. q3 crop의 바닥 위치를 teacher로 사용하지 않았다.'),
18: ('MATCHED_PARTIAL', 'U', 'U', '큰 목재 bin을 선택했으나 하부가 아래/오른쪽 영상 경계에 잘려 두 lower endpoint의 구조적 보임을 확인할 수 없다. truncation U이며 외부 가림 E가 아니다.'),
19: ('MATCHED', 'D', 'E', '목재 bin 하부 선택. 오른쪽 하단 지지부는 드러나고 왼쪽 전면 하단은 앞 cardboard 뒤에 가려진다.'),
20: ('WRONG_SELECTED_ENTITY', 'U', 'U', '최초 선택은 큰 전경 목재 bin이 아니라 뒤쪽 흰 원통/핸드파렛트가 놓인 별도 작은 pallet이다. 그 별도 개체에 참조 bin의 lower 역할을 옮기지 않는다.'),
21: ('MATCHED', 'D', 'D', '목재 bin 하부 선택. 두 전면 하단 지지부와 바닥판 윤곽이 외부 가림 없이 보인다.'),
22: ('MATCHED', 'D', 'E', '목재 bin 하부 선택. 오른쪽 하단은 노출되어 있고 왼쪽 하단은 전경 cardboard가 가로막는다.'),
23: ('MATCHED', 'D', 'E', '작은 목재 bin 하부 선택. 오른쪽 전면 하단은 보이고 왼쪽 하단은 앞 cardboard 뒤에 가려진다.'),
24: ('MATCHED', 'D', 'D', '오른쪽 목재 bin 하부 선택. 양쪽 하단 지지부와 바닥판 끝이 영상 안에 보이며 외부 장애물이 없다.'),
25: ('MATCHED', 'D', 'D', '먼 오른쪽 목재 bin 하부 선택. 두 하단 지지부/바닥판 끝이 드러나며 외부 장애물이 없다.'),
26: ('WRONG_SELECTED_ENTITY', 'U', 'U', '최초 선택은 전경 큰 목재 bin이 아니라 그 뒤 흰 원통을 싣고 있는 별도 pallet이다. 참조 bin의 역할을 이 개체로 바꾸지 않는다.'),
27: ('WRONG_SELECTED_ENTITY', 'U', 'U', '최초 선택과 crop은 왼쪽 노란 안전봉의 검정 받침대이며 참조 목재 bin은 오른쪽에 따로 있다.'),
28: ('MATCHED', 'D', 'D', '목재 bin 하부 선택. 기울어진 두 전면 하단 지지부와 바닥판 윤곽이 모두 보이고 앞쪽 가림 물체가 없다.'),
29: ('MATCHED', 'D', 'D', '중앙 목재 bin 하부 선택. 두 전면 하단 끝이 장애물 없이 보인다.'),
30: ('UNRESOLVED_SELECTED_ENTITY', 'U', 'U', '최초 box가 전경 cardboard를 중심으로 잡고 오른쪽 목재 bin과 섞여 있다. 완전히 잘린 하부에서 cardboard 오검출인지 bin amodal 추론인지 확증할 수 없어 개체 대응과 lower teacher 모두 U로 보류한다.'),
31: ('MATCHED', 'D', 'D', '먼 중앙 목재 bin 하부 선택. 양쪽 하부 끝이 노출되어 있고 앞쪽 장애물이 없다.'),
}

def main():
    seal_path = CONTACT / 'CONTACT_RENDER_COMPLETE.json'
    assert binding(seal_path)['sha256'] == 'ee5884b2c5beec495ef0de2d0c11b1dad7f9dd37d987964ee3192ba863c8d961'
    seal = json.loads(seal_path.read_text())
    assert seal['completed'] and seal['PASS']
    facts_path = Path(seal['all512_crop_facts']['path'])
    assert binding(facts_path) == seal['all512_crop_facts']
    facts = [json.loads(line) for line in facts_path.read_text().splitlines()]
    assert len(facts) == 512
    facts = [row for row in facts if row['frame_index'] < 32]
    assert len(facts) == 256
    assert {row['split'] for row in facts} == {'TRAIN_LAB'}
    assert [(r['frame_index'], r['native_slot']) for r in facts] == [(f,s) for f in range(32) for s in range(8)]
    sheets = {r['frame_index']: r for r in seal['sheets']}
    frame_rows, proposal_rows = [], []
    for frame in range(32):
        entity, n2, n3, reason = NOTES[frame]
        s = sheets[frame]
        assert binding(Path(s['sheet']['path'])) == s['sheet']
        frame_rows.append({
            'frame_index': frame, 'id': s['id'], 'split': 'TRAIN_LAB',
            'sheet': s['sheet'], 'actual_full_scene_and_all8_crops_viewed': True,
            'selected_entity_association': entity, 'native2_proposal': n2,
            'native3_proposal': n3, 'reason_KO': reason,
            'root_adjudication_needed': True,
            'not_physical_pixel_gold': True,
        })
        for row in facts[frame * 8:(frame + 1) * 8]:
            slot = row['native_slot']
            state = n2 if slot == 2 else n3 if slot == 3 else 'U'
            if slot not in (2,3):
                code = 'UPPER_ROLE_UNCERTIFIED' if slot < 2 else 'REAR_ROLE_OUT_OF_SCOPE'
                explanation = 'base upper0/1 경계가 인증되지 않아 U.' if slot < 2 else 'rear4..7 역할/자기 가림을 이번 lower partial bridge에서 판정하지 않아 U.'
            elif entity == 'WRONG_SELECTED_ENTITY':
                code, explanation = 'WRONG_SELECTED_ENTITY', reason
            elif entity == 'UNRESOLVED_SELECTED_ENTITY':
                code, explanation = 'UNRESOLVED_SELECTED_ENTITY', reason
            elif row['lower_role_continuity_status'] == 'UNRESOLVED':
                assert state == 'U'
                code, explanation = 'UNRESOLVED_PROVIDED_POSE_LOWER_ROLE', reason
            elif state == 'U':
                code, explanation = 'LOWER_ROLE_IMAGE_TRUNCATED_OR_UNCONFIRMED', reason
            elif state == 'E':
                code, explanation = 'EXTERNAL_CARDBOARD_STRUCTURAL_OCCLUSION', reason
            else:
                code, explanation = 'EXPOSED_LOWER_WHOLEASSEMBLY_ROLE', reason
            result = dict(row)
            result.update({
                'semantic_state': state,
                'semantic_reason_KO': explanation,
                'semantic_reason_code': code,
                'assistant_reviewed': True,
                'reviewer': 'initializer_source_readiness_independent_TRAIN32',
                'proposal_scope': 'ASSISTANT_WEAK_WHOLEASSEMBLY_LOWER_ROLE',
                'selected_entity_association': entity,
                'review_sheet': s['sheet'],
                'actual_full_scene_and_all8_crops_viewed': True,
                'root_adjudicated': False,
                'root_adjudication_needed': True,
                'not_query_pixel_membership_teacher': True,
                'q_error_used_for_teacher': False,
                'input_availability_and_teacher_separate': True,
                'human_gold': False, 'precise_physical_vertex_gold': False,
                'trainready': False, 'training_now': False,
            })
            assert result['query_xy'] == row['query_xy'] and result['L'] == row['L']
            assert result['original_input_valid'] == row['original_input_valid']
            proposal_rows.append(result)
    assert len(proposal_rows) == 256
    assert all(r['semantic_state'] == 'U' for r in proposal_rows if r['native_slot'] not in (2,3))
    assert all(r['semantic_state'] == 'U' for r in proposal_rows if r['frame_index'] == 5)
    counts = dict(collections.Counter(r['semantic_state'] for r in proposal_rows))
    lower_counts = dict(collections.Counter(r['semantic_state'] for r in proposal_rows if r['native_slot'] in (2,3)))
    assert counts == {'U':213, 'D':35, 'E':8}, counts
    assert lower_counts == {'D':35, 'U':21, 'E':8}, lower_counts
    rows_path = OUT / 'INDEPENDENT_TRAIN256_WEAK_PROPOSALS.jsonl'
    assert not rows_path.exists()
    rows_path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in proposal_rows))
    frame_binding = save('ALL32_FRAME_OBSERVATIONS.json', {'frames': frame_rows})
    receipt = {
        'completed': True, 'PASS': True,
        'reviewer': 'initializer_source_readiness_independent_TRAIN32',
        'contact_complete': binding(seal_path),
        'policy': binding(ROOT/'ROOT_PUBLIC64_NATIVE_EXPANSION_POLICY_BEFORE_NEW_BODIES.json'),
        'all_source_facts': binding(facts_path),
        'code': binding(Path(__file__)),
        'frame_observations': frame_binding,
        'proposals': binding(rows_path),
        'frames_actually_viewed': list(range(32)),
        'overview_sheets_actually_viewed': [f'OVERVIEW_{n:02d}_FRAMES_{n*8:02d}_TO_{n*8+7:02d}.png' for n in range(4)],
        'whole_scene_and_all8_crop_sheets_actually_viewed': 32,
        'all256_rows_retained': True, 'split': 'TRAIN_LAB',
        'counts_all256': counts, 'counts_lower64': lower_counts,
        'selected_entity_counts': dict(collections.Counter(NOTES[i][0] for i in range(32))),
        'wrong_selected_entity_frames': [2,10,20,26,27],
        'unresolved_selected_entity_frames': [30],
        'lower_camera_proof_unresolved_frame': 5,
        'external_occlusion_frames_native3': [3,4,6,7,15,19,22,23],
        'image_truncated_lower_roles': [[3,2],[8,2],[9,2],[14,2],[17,2],[18,2],[18,3]],
        'q_off_role_but_structurally_visible_lower3_frames': [8,14,17],
        'all_upper_and_rear_U': True,
        'other_agent_V146_proposals_read': False,
        'q_error_computed_or_used_as_teacher': False,
        'input_availability_controls_teacher': False,
        'new_RGB_recrop': False,
        'model_or_fit_or_PnP': False,
        'current223_or_forbidden_phase_access': False,
        'root_adjudication_needed': True,
        'training_authorized_by_this_receipt': False,
        'limits_KO': [
            '이 결과는 독립 assistant 구조적 weak proposal이며 root의 실제 시각 확인/채택이 필요하다.',
            '기존 UP=-OBJY 잠정성, 원본 전체 mesh AABB와 고립 scan outlier 한계는 유지한다. 좌표나 pose를 보정하지 않았다.',
            'native2/3만 wholeassembly lower virtual footprint 역할로 판정했다. pallet base upper0/1 또는 full8 의미 인증이 아니다.',
            'D/E는 대상 구조 역할의 노출/외부물체 가림이다. query pixel의 material membership이나 정확한 물리 vertex label이 아니다.',
            '원 q3가 바닥으로 빗나가도 보이는 대상 lower3를 E로 만들지 않았다. 특히 frame8/14/17은 root가 선택 개체 연관을 재확인할 수 있다.',
            '가까운 잘린 장면의 MATCHED_PARTIAL은 원래 amodal 선택이 해당 bin에 속한다는 약한 시각 판단이다. 새로운 bbox/candidate 선택을 하지 않았다.',
        ],
    }
    print(json.dumps(save('REVIEW_COMPLETE.json', receipt), ensure_ascii=False))
    print(json.dumps({'counts_all256':counts,'counts_lower64':lower_counts},ensure_ascii=False))

if __name__ == '__main__':
    main()
