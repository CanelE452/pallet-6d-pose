"""Summarize sealed source evidence without models or current evaluation reads."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np

OUT = Path(__file__).resolve().parent
P = OUT.parent


def binding(path):
    f = Path(path)
    return dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)


def read(rel):
    return json.loads((OUT / rel).read_text())


def write(rel, data):
    with (OUT / rel).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    a = read('reviewer_a/CUE_EVIDENCE_REVIEW.json')
    b = read('reviewer_b/REVIEWER_B_SOURCE_CUE_EVIDENCE.json')
    q = read('reviewer_b/PUBLIC_INITIALIZER_METADATA_DIAGNOSTIC.json')
    allq = read('next_contract_review/ALL_EXISTING_CANDIDATE_AVAILABILITY.json')
    verification = []
    # Recompute geometric availability from original saved q metadata, without RGB.
    for row in q['cases']:
        if row['selected_raw9'] is None:
            assert row['input_native_grid_L'] is None
            verification.append(dict(case_id=row['case_id'], no_detection=True, full=0, center=0))
            continue
        xy = np.asarray(row['selected_raw9'], dtype=np.float64)[:8]
        finite = np.isfinite(xy).all(1) & ~np.all(xy == -1, axis=1)
        good = xy[finite]
        H, W = row['raw_hw']
        L = float(np.clip(.25 * np.ptp(good, axis=0).max(), 16, max(H, W))) if len(good) >= 2 else 16.
        assert L == row['input_native_grid_L']
        r = np.asarray(row['reference_xy_GT_only'], dtype=np.float64)
        offset = (r - xy) / L
        np.testing.assert_array_equal(offset, row['reference_minus_q_over_L_GT_only'])
        center = np.flatnonzero(finite & (offset >= -.5).all(1) & (offset < .5).all(1)).tolist()
        full = np.flatnonzero(finite & (offset - 4 / L >= -.5).all(1) & (offset + 4 / L < .5).all(1)).tolist()
        assert center == row['selected_query_local_squares_containing_reference_center'] == []
        assert full == row['selected_query_local_squares_fully_containing_prior_uncertainty_square'] == []
        verification.append(dict(case_id=row['case_id'], no_detection=False, L=L, full=len(full), center=len(center)))
    assert a['counts']['apparent_local_visible_cue_weak_evidence'] == 3
    assert b['counts']['observed_visible_operational_cue_regions_conditional_AI'] == 3
    assert all(x['native8_semantic_id'] is None for x in a['records'])
    assert all(x['native8_assignment'] == 'UNKNOWN' for x in b['findings'])
    cases = allq['cases']
    assert sum(x['all_candidate_q8_rows'] for x in cases) == 88
    assert sum(x['all_candidate_availability']['FULL'] for x in cases) == 7
    result = {
        'schema': 'V83_ROOT_SEALED_PUBLIC_CUE_REVIEW_KO_V1',
        'completed_phase': True, 'goal_status': 'active', 'fully_solved': False,
        'main_finding_KO': '세 고정 공개 팔레트 reference 주변에 실제 목재 코너/부분선 cue의 약한 AI 관찰 증거가 있다. 그러나 저장된 selected top1 q8에는 FULL cue crop이 0이며 native8 대응도 UNKNOWN이다. 관측 가능한 cue, query 가용성, 대상/코너 대응을 분리해야 한다.',
        'capture_provenance_human_confirmation': {
            'statement_verbatim_KO': '직접 촬영했고 KU Leuven 자료와 무관합니다',
            'meaning': 'Current223 captured privately and does not share source with KU Leuven Pallox according to direct user confirmation.',
            'current223_use': 'Evaluation only; no new current223 reads/evaluation in V83 or V84 review.'
        },
        'independent_AI_reviews': {'A': a['counts'], 'B': b['counts'], 'agreement_is_human_gold': False},
        'root_actual_visual_review': 'Viewed reviewer_a/THREE_FIXED_PUBLIC_CUES_CONTACT.png showing full scenes, unmarked old detail and native96 crops with fixed points. Root agrees only with region-level weak observed-cue interpretation, not pixel material/native8 correctness.',
        'selected_query_availability_replayed_by_root': verification,
        'all_saved_candidate_availability': {
            'all_q_rows': 88, 'FULL': 7, 'PARTIAL': 1, 'MISS': 80,
            'selected_top1_FULL': 0,
            'rows': cases,
            'interpretation': 'Other saved candidates contain physical cues, but no candidate is selected using GT reference distance. FULL is geometric coverage only; intended-target/native8 association remains unproved.'
        },
        'actual_new_labels_inputs_forward_fit_current_PnP': 0,
        'source_tiers_preserved': ['query-pixel target membership', 'reference-local M0/M4 support', 'observed physical corner/part-line evidence', 'canonical virtual native8 semantics', 'convex self proxy'],
        'decision': {
            'native8_scalar_label_admission': False,
            'generic_spatial_cue_auxiliary_potential': 'Plausible weak source auxiliary, provided unlabeled cues remain unknown and target association/input distribution are separately explicit. Does not alone solve native8 D/E/S.',
            'GT_free_grid_scalar_cure': False,
            'why_grid_not_sufficient_KO': '새 grid는 cue가 crop에 들어오는 수를 늘리지만 어느 reference를 묻는지의 대응을 만들어 주지 않는다.',
            'next_executed_scope': binding(P / 'query_visibility_paired_cue_20261023_v84/SOURCE_CUE_EVIDENCE_SCOPE_BEFORE_RGB.json'),
            'next_step_KO': '기존 Pallox TRAIN exact-pose 첫 pair에서 before 실제 목재 cue를 먼저 독립 주석·동결하고, 이후 after에서 같은 cue의 외부 blocker/불확실성을 확인한다. 새 mask 모델이나 fit부터 반복하지 않는다.',
            'no_automatic_training_or_current_eval_admission': True
        },
        'bindings': [binding(OUT / x) for x in [
            'SOURCE_VISUAL_REVIEW_SCOPE_BEFORE_RGB.json',
            'reviewer_a/REVIEW_SEAL.json', 'reviewer_a/CUE_EVIDENCE_REVIEW.json',
            'reviewer_a/additive_erratum_and_teacher_inventory/ERRATUM_KO.json',
            'reviewer_b/REVIEW_SEAL.json', 'reviewer_b/REVIEWER_B_SOURCE_CUE_EVIDENCE.json',
            'reviewer_b/PUBLIC_INITIALIZER_METADATA_DIAGNOSTIC.json',
            'next_contract_review/ALL_EXISTING_CANDIDATE_AVAILABILITY.json',
            'next_contract_review/GRID_CUE_AUXILIARY_IDENTIFIABILITY_REVIEW.json',
            'public_source_next_step_review/PUBLIC_SOURCE_REFERENCE_EVIDENCE_REVIEW_KO.json',
            'public_source_next_step_review/REVIEW_SEAL.json',
            'reviewer_a/additive_erratum_and_teacher_inventory/SYNTH_NATIVE8_TEACHER_INVENTORY_KO.json',
            'reviewer_a/THREE_FIXED_PUBLIC_CUES_CONTACT.png',
        ]],
        'limitations_KO': '같은 촬영 계열 세 장의 AI 검토이며 사람 gold는 0이다. 원 ±4px는 검교정된 좌표 오차 보증이 아니다. 실제 자연 E/S 주석은 만들지 않았다. 원래 제안/좌표/모델/부정 결과는 보존한다.'
    }
    write('ROOT_EVIDENCE_AND_NEXT_DECISION_KO.json', result)
    h = html.escape
    rows = ''.join('<tr><td>'+h(x['case_id'])+'</td><td>'+str(x['saved_candidate_count'])+'</td><td>'+str(x['all_candidate_availability']['FULL'])+'</td><td>0</td><td>UNKNOWN</td></tr>' for x in cases)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개 팔레트 cue 검토 V83</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.7 'Nanum Gothic',sans-serif;color:#172536}}h1{{font-size:29px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #c6d2df;text-align:left}}img{{width:100%;margin:20px 0}}.lead{{background:#edf4fa;padding:18px;border-radius:8px}}</style>
<h1>공개 실사 cue와 초기 query 대응: V83</h1>
<p class="lead">실제 목재 코너·부분선 cue는 세 곳 모두 관찰됩니다. 저장된 selected top1 query에는 FULL cue crop이 없고, canonical native8 대응은 세 곳 모두 UNKNOWN입니다.</p>
<p>사용자는 현재 평가 223장을 직접 촬영했고 KU Leuven 자료와 무관하다고 확인했습니다. 평가셋의 학습·문턱 선택 제외 원칙을 유지했습니다.</p>
<table><tr><th>고정 cue</th><th>저장 후보 수</th><th>모든 후보의 FULL q crop</th><th>selected top1 FULL</th><th>native8 대응</th></tr>{rows}</table>
<p>P0의 selected crop은 cue 중심이 경계 바로 밖이고 원 ±4px 범위 일부만 겹칩니다. P8의 selected 후보는 뒤쪽 다른 대상입니다. P15에는 검출이 없습니다. 다른 후보의 FULL 7개는 가용성 진단이며 GT로 후보를 바꾸지 않았습니다.</p>
<img src="reviewer_a/THREE_FIXED_PUBLIC_CUES_CONTACT.png" alt="고정 공개 source 세 장과 실제 코너 crop">
<p>두 AI 검토자의 관찰은 약한 source 증거이며 사람 gold가 아닙니다. 원래 ±4px 범위도 정밀 좌표 보증은 아닙니다. cue가 crop 밖인 경우를 가림 음성으로 바꾸지 않았습니다.</p>
<p>다음 작업은 Pallox의 같은 자세 촬영 쌍입니다. 가림 전 영상에서 실제 cue를 먼저 고정한 뒤, 가림 후 영상의 같은 cue와 외부 blocker를 확인합니다. 이 단계에서는 새 학습·모델 추론·현재 평가·PnP를 실행하지 않습니다.</p>
<p><a href="ROOT_EVIDENCE_AND_NEXT_DECISION_KO.json">판단·검산·원본 봉인 연결</a> · <a href="reviewer_b/REVIEWER_B_SOURCE_CUE_EVIDENCE.json">B 독립 검토</a> · <a href="reviewer_a/CUE_EVIDENCE_REVIEW.json">A 독립 검토</a></p></html>'''
    with (OUT / 'ROOT_READOUT_KO.html').open('x') as f:
        f.write(page)
    write('ROOT_PHASE_RESULT_SEAL.json', {
        'completed_phase': True, 'goal_status': 'active', 'fully_solved': False,
        'code': binding(__file__),
        'outputs': [binding(OUT / x) for x in ['ROOT_EVIDENCE_AND_NEXT_DECISION_KO.json', 'ROOT_READOUT_KO.html']],
        'new_model_fit_current_PnP': 0
    })
    print(json.dumps(binding(OUT / 'ROOT_PHASE_RESULT_SEAL.json')))


if __name__ == '__main__':
    main()
