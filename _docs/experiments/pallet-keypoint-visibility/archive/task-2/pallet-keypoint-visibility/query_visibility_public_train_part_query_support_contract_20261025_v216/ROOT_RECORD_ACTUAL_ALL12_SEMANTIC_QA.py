"""Record root's actual visual observations. No image-derived automatic labels or training."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).parent


def read(rel):
    p = V / rel
    assert 'v112' not in str(p).lower() and not p.is_symlink()
    return json.loads(p.read_text())


def binding(p):
    p = Path(p)
    assert 'v112' not in str(p).lower() and not p.is_symlink()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(name, obj):
    p = V / name
    assert not p.exists()
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


# Root actually viewed all twelve 2304x1360 saved overlays, in fixed order
# 0..2, 3..5, 6..8, 9..11, through view_image(detail='original'). These notes
# are the root's visual judgements; this program only binds and records them.
NOTES = [
    '팔레트 오른쪽 상판 끝 n6의 붉은/파란 ROI는 같은 수직 끝 윤곽이다. n4는 바닥/케이블과 padding, n5는 A가 판 끝/C가 판 면으로 서로 다르다. n7은 같은 목재 주변이지만 native 대응 합의는 없다. 원 U0..3 유지.',
    'q8이 회색 플라스틱 상자에 배치됐다. 상자 모서리/리브가 선명해도 팔레트 부위는 아니다. n0/n3는 검은 받침과 상자 가장자리로 두 ROI가 다른 구조를 고른다. native D/E는 UNKNOWN.',
    '목재 팔레트 가까운 왼쪽 끝 n0와 먼쪽 왼쪽 상판 끝 n4는 같은 부위의 보이는 윤곽이다. n3는 두 관찰자가 D라고 했으나 ROI 공간 합의가 없어 기계 join도 UNKNOWN이다. n5/n7 ROI는 더 넓은 판 면/끝 주변을 달리 고른다. U1/2 유지.',
    '팔레트는 아래에 있으나 q8은 위의 목재 상자 쪽에 몰렸다. 목재 상자의 이음선/손잡이/상단 변을 팔레트 native로 바꿀 수 없다. n0 ROI도 한쪽은 배경을 길게 포함한다. 겹침만으로 같은 native 부위를 확정하지 않는다.',
    'n0는 전경 팔레트 왼쪽 상판 끝의 같은 모서리로 약한 부위 근거가 있다. n3는 A ROI가 EPAL 글자가 있는 블록 면의 위쪽, C ROI가 아래 왼쪽 끝/바닥 경계를 주로 고른다. 겹친 작은 구역만으로 동일한 끝 feature를 확정하지 않아 n3 native 약후보는 보류한다. n1/2 ROI invalid를 수정하지 않는다.',
    '팔레트 왼쪽 상판 끝 n4와 그 아래 옆면 경계 n7은 같은 팔레트/판의 연속성으로 약한 부위 근거가 있다. n5는 위의 목재 상자, n6는 파란 상자와 목재 상자의 경계여서 다른 물체를 native로 해석하지 않는다. n1/2는 원 q 밖의 아래쪽 팔레트 끝 주변을 골랐으며 OFF를 E로 바꾸지 않는다. U0/3 유지.',
    '뒤에 세운 팔레트와 앞의 목재 상자가 섞인 q8이다. n4/5/6은 같은 위쪽 팔레트 판 양 끝의 보이는 근거다. n7 A ROI는 위 판 끝면을 넓게 포함하고 C ROI는 아래 변/틈과 다음 판 시작까지 포함해 동일 feature는 불확실하다. n7 native 약후보 보류. n0..3의 목재 상자 feature는 다른 물체다. n5 q 중심의 배경/padding은 가림 근거가 아니다.',
    '실제 팔레트는 배경에 있고 초기 q는 앞쪽 목재 상자/패널로 갔다. 상자 모서리/리브/구멍이 잘 보여도 native 팔레트 visibility 근거가 아니다. 바닥에 놓인 n3도 E로 승격하지 않는다.',
    '초기 q8이 앞쪽 목재 상자의 변/손잡이 주변에 몰렸다. n0/3 상자 아래 코너, n4/5/7 상자 판 면, n6 금속/밝은 손잡이 등은 ROI-only 관찰이다. 같은 목재라는 이유로 배경 팔레트의 부위라 하지 않는다.',
    'q8이 뒤쪽 운반 카트의 검은 손잡이로 잘못 모였다. 손잡이 좌우 모서리가 실제 보이고 q ON 합의도 있지만 팔레트의 존재/가림 라벨이 아니다. 회색 상자 아래 목재 판과 해당 손잡이는 별개의 구조다.',
    '고정 설명 사례: 모든 q가 카트 손잡이 주변에 배치됐다. 선명한 local edge와 q ON 합의가 native 정답이라는 뜻이 아님을 보여준다. 원래 팔레트 부위가 보이지 않는다고 E로 만들지 않는다.',
    '회색 상자와 아래 긴 목재 판, 흰 패널이 함께 있다. n5/6은 목재 판 끝/옆면이 보이지만 q8 전체와 의도 native 부위의 대응은 두 관찰자가 합의하지 않았다. n4/7은 플라스틱 상자다. U1/2 보존; 다른 물체/흰 면/패딩을 native E로 하지 않는다.',
]
WITHHELD = {(4, 3): 'A: EPAL 블록 면 위쪽, C: 아래 바깥 코너. 같은 국소 feature가 확실하지 않음.',
            (6, 7): 'A: 위 판 끝면, C: 아래 변/틈과 인접 판. 넓은 ROI 겹침만으로 동일 feature 확정 불가.'}
rows = read('joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json')['rows96']
assert len(rows) == 96
frames = []
out = []
for fi in range(12):
    rr = rows[8 * fi:8 * fi + 8]
    assert [r['input']['native_slot'] for r in rr] == list(range(8))
    frames.append({'frame_position': fi, 'frame_key': rr[0]['input']['frame_key'],
                   'internal_source_role': rr[0]['input']['internal_source_role'],
                   'actual_overlay_view': binding(V / 'joined_part_query_support_v2' / f'FIXED_FRAME_{fi:02d}_TWO_READER_OVERLAY.png'),
                   'viewer': 'tools.view_image(detail=original)', 'actually_viewed': True,
                   'manual_root_observation_ko': NOTES[fi]})
    for n, r in enumerate(rr):
        inp = r['input']; raw = r['raw_helper_candidate']
        original_u = not inp['original_query_available']
        candidate = raw['native_weak_visibility_agreement_candidate']
        if original_u:
            status = 'ORIGINAL_INPUT_U_RETAINED'; reason = '원 U: 전축 UNKNOWN/ROI 없음 유지'
        elif (fi, n) in WITHHELD:
            assert candidate == 'D'
            status = 'RAW_D_WITHHELD_SAME_FEATURE_UNRESOLVED'; reason = WITHHELD[(fi, n)]
        elif candidate == 'D':
            status = 'SAME_PALLET_PART_VISIBLE_WEAK'; reason = '같은 팔레트의 의도 부위에 대한 약한 보임 근거. 정확한 native 점 인증/좌표 정답 아님.'
        elif raw['spatial_witness_agreement_candidate']:
            status = 'LOCAL_FEATURE_ONLY_NATIVE_UNRESOLVED'; reason = 'ROI 공간 합의만 보존. 물체/부위/feature 대응은 native 학습 정답으로 승격하지 않음.'
        else:
            status = 'NO_NATIVE_CANDIDATE_RETAIN_UNKNOWN'; reason = '두 관찰자의 공간/부위/주소화 합의가 부족함. 억지로 D/E를 채우지 않음.'
        out.append({'frame_position': fi, 'row_id': inp['row_id'], 'frame_key': inp['frame_key'],
                    'native_slot': n, 'original_role': inp['role'], 'internal_source_role': inp['internal_source_role'],
                    'original_inputU': original_u, 'raw_native_weak_candidate': candidate,
                    'raw_spatial_witness_candidate': raw['spatial_witness_agreement_candidate'],
                    'raw_query_support_agreement': raw['query_support_agreement'],
                    'root_semantic_review_status': status, 'root_weak_part_state': 'D' if status == 'SAME_PALLET_PART_VISIBLE_WEAK' else 'UNKNOWN',
                    'reason_ko': reason, 'frame_context_note_index': fi,
                    'native_coordinate_correctness_GT': 'UNKNOWN', 'physical_native_point_visibility_gold': 'UNKNOWN',
                    'primary_DE_training_admission': False, 'auxiliary_training_admission': False})
assert Counter(r['root_weak_part_state'] for r in out) == {'D': 9, 'UNKNOWN': 87}
assert sum(r['original_inputU'] for r in out) == 11
counts = {}
for role in ['ALL', 'SOURCE_FIT', 'SOURCE_DEV']:
    q = [r for r in out if role == 'ALL' or r['internal_source_role'] == role]
    counts[role] = {'N': len(q), 'original_inputU': sum(r['original_inputU'] for r in q),
                    'root_weak_part_state': dict(Counter(r['root_weak_part_state'] for r in q)),
                    'review_status': dict(Counter(r['root_semantic_review_status'] for r in q))}
extra = []
for fi, n in WITHHELD:
    extra.append({'frame_position': fi, 'native_slot': n, 'binding': rows[8 * fi + n]['display_bindings']['clean96'],
                  'viewer': 'tools.view_image(detail=original)', 'actually_viewed': True})
write('ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json', {
    'completed': True, 'PASS': True, 'UTC': datetime.now(timezone.utc).isoformat(),
    'scope': binding(V / 'ROOT_PROSPECTIVE_JOIN_SEMANTIC_AND_FIXED_IMAGE_QA_SCOPE.json'),
    'sealed_raw_join': binding(V / 'joined_part_query_support_v2/COMPLETE.json'),
    'sealed_raw96': binding(V / 'joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json'),
    'independent_saved_numeric_validity_QA': binding(V / 'independent_saved_join_review_c/REVIEW.json'),
    'all12_actual_viewed': frames, 'additional_two_actual_clean96_views': extra,
    'counts': counts, 'rows96': out,
    'interpretation': [
        'Raw D11 / local witness62 remain unchanged in the sealed join. Root withholds two ambiguous same-feature candidates, leaving nine weak same-pallet-part visible observations.',
        'These are agent weak observations, not human gold or physical native-point labels. Both original reader values remain immutable.',
        'Root review is conservative qualitative source evidence, not a scored independent test. V215 views and shared historical notes were already exposed.',
        'ON/OFF refers to the annotated local feature and is not native coordinate correctness. Foreign crate/handle examples demonstrate the distinction.',
        'No negative label from missing candidate, mask, background, query OFF, or another object. E/SELF still zero confirmed in this SOURCE12 review.',
        'No perfect mesh or exact-corner certification blanket gate for legitimate weak part/ROI supervision. Future input/loss contract and controls must match the weak question actually supervised.',
        'All training admissions remain false. SOURCE_DEV and all descendants excluded from every future training branch. Physical component independence not established.',
        'Original U11, q/L/native order and RAW96 sampling preserved. No GT, arrays, new inference, fit, protected evaluation, threshold selection, or PnP.']})
with (V / 'ROOT_ALL96_SEMANTIC_REVIEW.csv').open('x', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
write('ROOT_METADATA_OBSERVATION_FAILURES_AND_BOUNDED_FIXES.json', {
    'completed': True,
    'failures': [
        {'stage': 'root saved metadata observation', 'error': 'Large JSON stdout was truncated by exec output cap; functions JSON.parse saw Warning prefix and failed.', 'science_mutation': 0, 'fix': 'Read only bounded frame bindings/counts; no join rerun or producer edits.'},
        {'stage': 'root readiness report observation', 'error': 'Guessed REVIEW_KO.md basename absent; read failed.', 'science_mutation': 0, 'fix': 'Read actual REVIEW.json and exact owned directory metadata; actual report is KO.md.'},
        {'stage': 'root canonical handoff observation', 'error': 'Full historical handoff output truncated.', 'science_mutation': 0, 'fix': 'Future handoff readers must select canonical fields before printing; no truncated text treated as complete evidence.'}],
    'no_actual_join_failure_or_retry': True, 'protected_or_model_or_GT_reads': 0})
print(json.dumps({'completed': True, 'actual_overlay_views': 12, 'extra_clean96_views': 2, 'root_counts': counts}, ensure_ascii=False))
