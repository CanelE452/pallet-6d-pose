"""Seal the actual paired public cue observations, without model/current reads."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json

OUT = Path(__file__).resolve().parent
P = OUT.parent


def bound(path):
    f = Path(path)
    return {'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes': f.stat().st_size}


def read(path):
    return json.loads((OUT / path).read_text())


def save(path, value):
    with (OUT / path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    u = read('ROOT_BEFORE_REFERENCE_UNION_LOCK.json')
    a = read('reviewer_a/after_review/AFTER_CUE_EVIDENCE_REVIEW.json')
    b = read('reviewer_b/after_review/AFTER_CUE_EVIDENCE_REVIEW.json')
    m = read('independent_metadata_audit/UNION_METADATA_PARITY.json')
    for f in ['reviewer_a/after_review/AFTER_REVIEW_SEAL.json', 'reviewer_b/after_review/AFTER_REVIEW_SEAL.json']:
        assert (OUT / f).is_file()
    assert a['B_after_judgments_read_before_this_seal'] is False
    assert b['scientific_scope']['other_reviewer_after_judgments_read_or_received'] is False
    ar = {r['union_id']: r for r in a['records']}
    br = {r['union_id']: r for r in b['observations']}
    coverage = {r['cue_id']: r for r in m['per_original_cue_all8']}
    expected_ids = {r['union_id'] for r in u['proposals']}
    assert set(ar) == set(br) == set(coverage) == expected_ids
    result_rows = []
    for r in u['proposals']:
        k = r['union_id']
        assert ar[k]['reference_xy_unchanged'] == br[k]['reference_xy_unchanged'] == r['reference_xy_unchanged']
        assert ar[k]['uncertainty_bbox_unchanged'] == br[k]['uncertainty_bbox_xyxy_half_open_unchanged'] == r['uncertainty_bbox_xyxy_half_open_unchanged']
        assert ar[k]['after_state'] == br[k]['after_state']
        assert ar[k]['native8_status'] == br[k]['native8_status'] == 'UNKNOWN'
        result_rows.append({
            'id': k, 'reference_xy_unchanged': r['reference_xy_unchanged'],
            'uncertainty_bbox_unchanged': r['uncertainty_bbox_xyxy_half_open_unchanged'],
            'cue_description': r['cue_description_unchanged'],
            'before': 'APPARENT_VISIBLE_PHYSICAL_PART_CUE_AI',
            'after_A': ar[k]['after_state'], 'after_B': br[k]['after_state'],
            'review_agreement': True, 'human_gold': False, 'native8': 'UNKNOWN',
            'saved_source_crop_states_all8': coverage[k]['all8_crop_states'],
            'saved_source_has_FULL': coverage[k]['any_FULL'],
            'native_query_or_training_label_admitted': False,
        })
    counts = Counter(r['after_A'] for r in result_rows)
    assert counts == {'APPARENT_SAME_CUE_VISIBLE_AI': 6, 'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI': 2}
    assert m['q8_L_box_rawHW_A_B_exact'] and m['all8_full_original_dict_and_source_before_bindings_exact']
    assert m['summary']['combinations_FULL_PARTIAL_MISS'] == {'MISS': 58, 'FULL': 4, 'PARTIAL': 2}
    rows_by_id = {r['id']: r for r in result_rows}
    # Duplicated top-corner proposals are linked only; original coordinates remain.
    groups = [['V84_A_CUE_00', 'B01'], ['V84_A_CUE_01', 'B02'], ['V84_A_CUE_02'], ['V84_A_CUE_03'], ['B03'], ['B04']]
    unique_counts = Counter(rows_by_id[g[0]]['after_A'] for g in groups)
    assert unique_counts == {'APPARENT_SAME_CUE_VISIBLE_AI': 4, 'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI': 2}
    files = [
        'SOURCE_CUE_EVIDENCE_SCOPE_BEFORE_RGB.json', 'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json',
        'ROOT_BEFORE_REFERENCE_UNION_LOCK.json', 'reviewer_a/BEFORE_REVIEW_SEAL.json',
        'reviewer_b/BEFORE_REVIEW_SEAL.json', 'reviewer_a/after_review/AFTER_REVIEW_SEAL.json',
        'reviewer_b/after_review/AFTER_REVIEW_SEAL.json',
        'reviewer_a/after_review/AFTER_CUE_EVIDENCE_REVIEW.json',
        'reviewer_b/after_review/AFTER_CUE_EVIDENCE_REVIEW.json',
        'independent_metadata_audit/INDEPENDENT_METADATA_SEQUENCE_TIER_AUDIT.json',
        'independent_metadata_audit/UNION_METADATA_PARITY.json',
        'independent_metadata_audit/UNION_METADATA_PARITY_SEAL.json',
        'independent_metadata_audit/UNION_COUNT_UNIT_CLARIFICATION.json',
        'reviewer_b/after_review/FULL_SCENE_BEFORE_AFTER_UNREGISTERED.png',
        'reviewer_b/after_review/ALL8_BEFORE_AFTER_FIXED_REGIONS_CONTACT.png',
        'reviewer_b/after_review/V84_A_CUE_02_BEFORE_AFTER_FIXED_PANEL.png',
        'reviewer_b/after_review/B03_BEFORE_AFTER_FIXED_PANEL.png',
    ]
    result = {
        'schema': 'V84_ROOT_PUBLIC_REAL_PAIRED_PART_CUE_EVIDENCE_KO_V1',
        'completed_phase': True, 'goal_status': 'active', 'fully_solved': False,
        'main_finding_KO': '공개 실사 첫 고정 TRAIN pair에서 같은 목재 cue의 실제 외부 가림 근거를 확보했다. 중복 연결 후6개 물리 단서 중4개는 after에도 보이고2개는 별도 cardboard에 전체 불확실 영역이 덮였다. 두 AI 검토는 모든 원제안8개에서 일치했다.',
        'pair_id': '0.55m_lab:0000_0001',
        'selection': 'First lexicographic existing TRAIN_LAB exact-pose pair; not chosen by crop/model/occlusion outcome.',
        'review_order': 'Two independent before-only inspections/seals -> root unchanged-reference union -> two independent after inspections/seals -> root cross-review.',
        'count_units': {'raw_RGB': 2, 'paired_units': 1, 'physical_target_instances': 1, 'original_proposals': 8, 'distinct_physical_cue_regions_qualitative': 6, 'human_gold': 0, 'native8_correspondences': 0},
        'rows': result_rows,
        'duplicate_links_without_averaging': u['possible_duplicate_groups_before_only'],
        'distinct_region_after_counts': dict(unique_counts),
        'root_actual_visual_review': {
            'full_pair_unregistered': True, 'all8_paired_contact': True,
            'both_covered_region_panels_individually': True,
            'observation_KO': 'before의 왼쪽 base-runner와 최하단 넓은 판재 왼쪽 끝은 after에서 별도 판지 면 안에 놓이고 원래 목재 단서는 보이지 않는다. 상단과 오른쪽 목재 윤곽은 유지된다. 큰 이동은 관찰되지 않았으나 정밀 pixel 등록/정답 위치는 인증하지 않았다.'
        },
        'saved_source_input_coverage': {
            'all64_per_geometry_states': m['summary']['combinations_FULL_PARTIAL_MISS'],
            'proposals_any_FULL': 4, 'distinct_regions_any_FULL': 3,
            'physically_covered_regions_with_FULL': ['V84_A_CUE_02', 'B03'],
            'source_query_lineage_KO': m['original_input_lineage_KO'],
            'before_after_q8_L_box_HW_exact': True,
            'no_GT_crop_recentering_or_candidate_selection': True,
            'interpretation_KO': '불변 source crop으로 관찰할 수 있는 보임유지1지역과 보임→외부가림2지역이 있다. Source sampler의 slot4/6 등을 native8 의미로 바꾸지 않는다. 모든64조합을 남긴 가용성 진단이며 훈련용 crop을 GT로 새로 고른 것이 아니다.'
        },
        'decision': {
            'weak_physical_cue_evidence_pack_ready': True,
            'can_support_next_spatial_cue_auxiliary_data_preparation': True,
            'human_or_native8_semantic_gold': False,
            'scalar_native8_training_label_admission': False,
            'training_or_current_evaluation_admitted_by_this_phase': False,
            'next_work_KO': '기존50쌍의 source split(TRAIN lab22/VAL hall28)을 유지하며 같은 before-cue 고정→after 외부 blocker 확인 절차로 spatial cue 보조감독 자료를 확대한다. 새 fit 전에 미어노테이션 cue/U, 중복·촬영묶음 가중치, 입력 분포와 native8 주과제의 관계를 명시한다. 현재223을 선택에 쓰지 않는다.',
            'not_waiting_for_human_permission': True,
            'why_distinct_from_V66_V72_KO': '실제 자연 촬영 전후에서 이름 붙인 동일 목재 부분선/코너와 foreign blocker를 RGB로 확인했다. 기존 V66은 mesh/depth/SAM의 조건부 pixel-area teacher, V72는 소수 generic cue/합성 cover pilot이었다. 이번 한 쌍의 근거만으로 새 모델 전이를 증명하거나 과거 실패를 해소했다고 하지 않는다.'
        },
        'limits_KO': '같은 tall wooden bin 한 개의 한 pair이고 flat pallet/정사각형의 독립 성능 증거가 아니다. AI 조건부 관찰이며 native8 대응과 사람 gold는0, self S는UNKNOWN이다. 원 불확실 범위는 검교정된 오차 보증이 아니며 정밀 정합은 인증하지 않았다.',
        'current_capture_provenance_verbatim_KO': '직접 촬영했고 KU Leuven 자료와 무관합니다',
        'execution': {'new_public_RGB_evidence_review': True, 'new_model_forward': 0, 'fit': 0, 'compositing': 0, 'current223_read_or_eval': 0, 'PnP_or_alignment_optimization': 0, 'native_coordinates_changed': 0},
        'bindings': [bound(OUT / f) for f in files],
    }
    save('ROOT_PAIRED_CUE_EVIDENCE_AND_NEXT_DECISION_KO.json', result)
    text_rows = []
    names = ['상단 오른쪽', '상단 왼쪽', '왼쪽 base-runner', '오른쪽 base-runner', '최하단 넓은 판재 왼쪽', '최하단 넓은 판재 오른쪽']
    for g, name in zip(groups, names):
        r = rows_by_id[g[0]]
        state = '판지에 가림' if 'EXTERNAL_' in r['after_A'] else '계속 보임'
        avail = 'FULL' if r['saved_source_has_FULL'] else ('PARTIAL' if 'PARTIAL' in r['saved_source_crop_states_all8'] else 'MISS')
        text_rows.append(f'<tr><td>{name}</td><td>{html.escape(" / ".join(g))}</td><td>보임</td><td>{state}</td><td>{avail}</td></tr>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개 실사 같은 코너의 전후 가림 V84</title>
<style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.7 'Nanum Gothic',sans-serif;color:#172536}h1{font-size:29px}.lead{padding:18px;background:#eaf4ed;border-radius:8px}img{display:block;max-width:100%;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border:1px solid #c5d1dd;text-align:left}.pair{width:800px}</style>
<h1>같은 코너가 실제로 가려지는 공개 실사: V84</h1>
<p class="lead">가림 전 영상에서 먼저 고정한 6개 목재 단서 중 4개는 가림 후에도 보이고, 2개는 앞쪽 판지에 가려집니다. 두 AI 검토가 일치했으며, 같은 위치의 실제 crop으로 확인했습니다.</p>
<p>KU Leuven Pallox의 기존 TRAIN 첫 촬영 쌍(0.55m_lab:0000/0001)입니다. 사용자는 현재 평가223장이 직접 촬영한 독립 자료라고 확인했습니다.</p>
<img src="reviewer_b/after_review/FULL_SCENE_BEFORE_AFTER_UNREGISTERED.png" alt="정렬이나 좌표 수정 없는 공개 실사 전후">
<p>두 검토자가 before만 보고 좌표와 불확실 범위를 봉인했습니다. 8개 제안 중 상단 두 곳이 중복되어 물리 단서는 6곳입니다. 이후 after를 열어 같은 범위를 대조했습니다.</p>
<table><tr><th>물리 단서</th><th>원 제안 ID</th><th>Before</th><th>After: 두 AI 일치</th><th>저장 source crop 가용성</th></tr>''' + ''.join(text_rows) + '''</table>
<p>아래 두 crop에서 노란 범위와 좌표는 그대로입니다. 목재 단서가 사라진 자리와 그 앞을 덮는 별도 판지가 함께 확인됩니다.</p>
<img class="pair" src="reviewer_b/after_review/V84_A_CUE_02_BEFORE_AFTER_FIXED_PANEL.png" alt="왼쪽 base-runner 동일 위치: before 보임 after 판지 가림">
<img class="pair" src="reviewer_b/after_review/B03_BEFORE_AFTER_FIXED_PANEL.png" alt="왼쪽 판재 끝 동일 위치: before 보임 after 판지 가림">
<p>이는 AI가 관찰한 물리 단서의 조건부 source 증거입니다. 사람 정답이나 기존 virtual8 키포인트의 대응 라벨은 아니며, self 가림도 주석하지 않았습니다. 두 사진은 같은 물체의 한 촬영 쌍입니다.</p>
<p>저장 source q8·crop 크기·box는 두 사진에서 같습니다. 전체64 cue/crop 조합은 FULL4·PARTIAL2·MISS58입니다. 이 q는 teacher로 만든 source sampling 위치이며 실제 초기 detector의 native8 출력은 아닙니다. 입력 가용성과 단서의 가시성을 분리했습니다.</p>
<p>다음은 같은 방식으로 공개 실사의 보조감독 자료를 확대하는 작업입니다. 현재 평가셋은 계속 학습·문턱·모델 선택에서 제외합니다. 이번 단계의 새 학습·모델 추론·현재 평가·PnP는 모두0입니다. 전체 문제는 아직 해결되지 않았습니다.</p>
<p><a href="ROOT_PAIRED_CUE_EVIDENCE_AND_NEXT_DECISION_KO.json">원 좌표·교차검산·한계·다음 결정</a> · <a href="reviewer_b/after_review/ALL8_BEFORE_AFTER_FIXED_REGIONS_CONTACT.png">원 제안8개 실제 paired crop</a> · <a href="../query_visibility_public_cue_audit_20261023_v83/ROOT_READOUT_KO.html">직전 초기 query 대응 감사 V83</a></p>
<p>출처: <a href="https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/DJLDFS">KU Leuven Pallox 공개 자료, DOI10.48804/DJLDFS</a>. 원본·출처·이전 부정 결과를 보존했습니다.</p></html>'''
    with (OUT / 'ROOT_READOUT_KO.html').open('x') as f:
        f.write(page)
    save('ROOT_PHASE_RESULT_SEAL.json', {
        'completed_phase': True, 'goal_status': 'active', 'fully_solved': False,
        'code': bound(__file__),
        'outputs': [bound(OUT / f) for f in ['ROOT_PAIRED_CUE_EVIDENCE_AND_NEXT_DECISION_KO.json', 'ROOT_READOUT_KO.html']],
        'new_model_fit_current_PnP': 0,
    })
    print(json.dumps(bound(OUT / 'ROOT_PHASE_RESULT_SEAL.json')))


if __name__ == '__main__':
    main()
