"""Append-only v22/v23 accounting after complete prediction, audit and score.

Source preparation does not decode score archives.  Run only after root's
post-audit/score GO. Each method is uniform over all238; CAL remains separate
from primary192. No inference, fitting, reference decode or method routing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from . import build_repair_campaign_progress_v3_v18 as P
from . import build_repair_campaign_progress_v4_v19 as PAIR
from . import evaluate_complete_bi_bounded_priors_v4_23 as V


OLD = V.OLD
OUTPUT = P.OUTPUT
NAMES = ('PROGRESS_V6_V23.json', 'RESULTS_V6_V23.html', 'PROGRESS_V6_V23_SEAL.json')
LABELS = {
    P.BASELINE: '기본 e (변경 없음)',
    'O_SOFT_NATIVE': 'v22 기존 B/I · O 벌점',
    'O_SOFT_S_NATIVE': 'v22 기존 B/I · O/S 벌점',
    'O_SOFT_PLUS_MASK': 'v22 기존 B/I · O 벌점 + 마스크',
    'O_SOFT_S_PLUS_MASK': 'v22 기존 B/I · O/S 벌점 + 마스크',
    'COMPLETE_BI_O_SOFT_NATIVE': 'v23 B/I 영상선 · O 벌점',
    'COMPLETE_BI_O_SOFT_S_NATIVE': 'v23 B/I 영상선 · O/S 벌점',
    'COMPLETE_BI_O_SOFT_PLUS_MASK': 'v23 B/I 영상선 · O 벌점 + 마스크',
    'COMPLETE_BI_O_SOFT_S_PLUS_MASK': 'v23 B/I 영상선 · O/S 벌점 + 마스크',
}


def _profile_pairs(old_profiles, new_profiles):
    """Match profile AND selector position; the two runs have four arms each."""
    if set(old_profiles) != {'O', 'OS'} or set(new_profiles) != set(old_profiles):
        raise ValueError('Exact O/OS profiles required')
    pairs = []
    for profile in ('O', 'OS'):
        before, after = old_profiles[profile], new_profiles[profile]
        if len(before) != 2 or len(after) != 2:
            raise ValueError('Each profile must declare native, then fixed-mask selectors')
        for index, selector in enumerate(('native', 'fixed_mask')):
            pairs.append(dict(profile=profile, selector=selector, selector_index=index,
                before=before[index], after=after[index]))
    if len({r['before'] for r in pairs}) != 4 or len({r['after'] for r in pairs}) != 4:
        raise ValueError('Four distinct uniform arms required in each run')
    return pairs


def _require_audit(folder, *, complete_bi=False):
    C = P.C
    path = folder / 'independent_root_audit_v1/AUDIT.json'
    sealpath = path.with_name('AUDIT_SEAL.json')
    audit, seal = C.read(path), C.read(sealpath)
    if (audit['rows'] != 238 or audit['failure_count'] or audit['failures']
            or C.sha(path) != seal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(folder / 'PREDICTION_SEAL.json')):
        raise ValueError('Independent zero-failure all238 pose audit required')
    paths = [path, sealpath]
    if complete_bi:
        receiptpath = path.with_name('COMPLETE_BI_READOUT_AND_KERNEL_RECEIPT.json')
        receipt = C.read(receiptpath)
        if (receipt['failure_count'] or receipt['failures']
                or receipt['selected_pose_audit']['sha256'] != C.sha(path)
                or receipt['core_problem_factory_independent_of_experiment_driver'] is not True
                or receipt['new_image_pose_fits_NN_or_reference_decode'] != 0):
            raise ValueError('Independent complete-BI readout/kernel closure required')
        paths.append(receiptpath)
    return paths


def _require_declared_arms(folder, arms):
    seal = P.C.read(folder / 'PREDICTION_SEAL.json')
    if (seal.get('arms') != list(arms) or seal.get('arm_count') != len(arms)
            or seal.get('all_declared_arms_complete') is not True):
        raise ValueError('Exact five declared arms must be completely sealed')


def _same_e(old, new):
    for part in P.COUNTS:
        if [r['id'] for r in old[part]['rows']] != [r['id'] for r in new[part]['rows']]:
            raise ValueError('Original ordered paired cohort changed')
        before = [(r['metrics'][P.BASELINE], r['accepted'][P.BASELINE]) for r in old[part]['rows']]
        after = [(r['metrics'][P.BASELINE], r['accepted'][P.BASELINE]) for r in new[part]['rows']]
        if before != after:
            raise ValueError('Immutable e metric or fallback changed')


def _population_summaries(reports, arm):
    result = {}
    for part, count in P.COUNTS.items():
        rows = reports[part]['rows']
        if len(rows) != count:
            raise ValueError('No frame, fallback, or native-flag exclusion allowed')
        result[part] = dict(count=count, **P._summary(rows, arm))
        P._same_summary(result[part], reports[part]['summary'][arm])
    result['PRIMARY192'] = dict(count=192, **P._summary(
        reports['TEST']['rows'] + reports['REMAINDER']['rows'], arm))
    return result


def _render(data):
    content = ['<!doctype html><html lang="ko"><meta charset="utf-8">',
        '<title>v22·v23 전체 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1440px;margin:32px auto;padding:0 20px;color:#192536}'
        'table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{border-bottom:1px solid #ccd4df;'
        'padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef4fa}'
        'p{max-width:1150px}.scroll{overflow-x:auto}</style>',
        '<p>기본 방법 e는 변경하지 않았습니다. 각 행은 기존 TEST49와 추가143장 모두에 동일하게 적용한 하나의 방법입니다. '
        '사진별·집합별 최선 방법을 섞지 않았고 모든 실패와 fallback을 포함했습니다. CAL46은 보정용 집합으로 별도 표시합니다.</p>',
        '<p>v22는 기존 B/I 관측과 제한된 O/S 기하 벌점을 사용합니다. v23은 같은 영상선 기록에서 적용 가능한 B/I의 '
        '완전한 2D 비용을 영상선·불확실한 접선 비용으로 교체하고 별도 CAL 잡음 모델을 사용합니다. '
        '두 세대의 O/OS 프로필과 native/고정 마스크 선택기를 각각 대응 비교했습니다. 이는 단일 요소만 바꾼 실험이 아닙니다.</p>',
        '<p>S는 계속 자세 투영으로 출력하며 S 영상·2D·영역 관측은 없습니다. S 벌점은 상관된 초기 모델 가시성 prior입니다. '
        'U는 영역으로만 사용합니다. O 타원은 hard 조건 대신 제한된 벌점이며 닫힌 영상 범위 조건을 유지합니다. '
        '기존 hard guard와 v19 시작점 이름 정책을 유지했습니다.</p>']
    for part, title in (('PRIMARY192', '전체192 = TEST49 + 추가143'), ('TEST', '기존 TEST49'),
            ('REMAINDER', '추가143'), ('CAL', '별도 CAL46')):
        content.append('<p><b>' + title + '</b> — T≤5cm는 위치 오차, 공동은 5cm·5° 모두 만족한 수입니다.</p>')
        content.append('<div class="scroll"><table><tr><th>방법</th><th>전체 수</th><th>T≤5cm</th><th>공동</th>'
            '<th>평균 T(cm)</th><th>최대 T(cm)</th><th>평균 R(°)</th><th>최대 R(°)</th>'
            '<th>R 미확정</th><th>fallback</th></tr>')
        for method in data['methods']:
            q = method['populations'][part]
            rmean = '미확정' if q['rotation_deg']['mean'] is None else f"{q['rotation_deg']['mean']:.3f}"
            rmax = '미확정' if q['rotation_deg']['max'] is None else f"{q['rotation_deg']['max']:.3f}"
            content.append(f'<tr><td>{html.escape(method["label_KO"])}</td><td>{q["count"]}</td>'
                f'<td>{q["translation_le5"]}</td><td>{q["successes_5cm_5deg"]}</td>'
                f'<td>{q["translation_cm"]["mean"]:.3f}</td><td>{q["translation_cm"]["max"]:.3f}</td>'
                f'<td>{rmean}</td><td>{rmax}</td><td>{q["rotation_unresolved"]}</td><td>{q["fallback_count"]}</td></tr>')
        content.append('</table></div>')
    content.extend(['<p>severe는 제외했습니다. 추가143장의 참조는 annotation/PnP 유도값이며 독립적인 물리 계측이 아닙니다. '
        '반복 사용한 TEST의 탐색 결과이며 새로운 holdout 일반화나 기본 방법 채택을 주장하지 않습니다. '
        '분포·최대 사례·대응별 증가/감소·회전·검산·봉인은 JSON에 보존했습니다.</p>',
        '<p><a href="PROGRESS_V6_V23.json">전체 수치와 검산 근거</a></p></html>'])
    return '\n'.join(content)


def build(*, root_postscore_GO=False):
    if root_postscore_GO is not True:
        raise ValueError('Wait for complete prediction, root audit, score and explicit aggregation GO')
    C = P.C
    paths = [OUTPUT / name for name in NAMES]
    if any(path.exists() for path in paths):
        raise FileExistsError('Append-only comparison: preserve all previous documents')
    pairs = _profile_pairs(OLD.PROFILE_ARMS, V.PROFILE_ARMS)
    oldprotocol, old = P._load_run(OLD.OUTPUT)
    protocol, new = P._load_run(V.OUTPUT)
    _require_declared_arms(OLD.OUTPUT, OLD.ARMS)
    _require_declared_arms(V.OUTPUT, V.ARMS)
    auditpaths = _require_audit(OLD.OUTPUT) + _require_audit(V.OUTPUT, complete_bi=True)
    if oldprotocol['ids'] != protocol['ids']:
        raise ValueError('Same ordered all238 required')
    _same_e(old, new)
    methods = []
    for reports, folder, arms in ((old, OLD.OUTPUT, OLD.ARMS), (new, V.OUTPUT, V.ARMS[1:])):
        for arm in arms:
            populations = _population_summaries(reports, arm)
            methods.append(dict(method=arm, label_KO=LABELS[arm], populations=populations,
                source_output=str(folder.relative_to(C.ROOT)), current_default=arm == P.BASELINE,
                original_TEST49_all_translation_le5=populations['TEST']['translation_le5'] == 49,
                numerical_primary192_target_met=populations['PRIMARY192']['translation_le5'] == 192))
    if len(methods) != 9 or len({method['method'] for method in methods}) != 9:
        raise ValueError('Exactly e + four v22 + four v23 uniform methods required')
    paired = {}
    for pair in pairs:
        previous, arm = pair['before'], pair['after']
        populations = {part: PAIR._paired(old[part]['rows'], new[part]['rows'], previous, arm)
            for part in P.COUNTS}
        populations['PRIMARY192'] = PAIR._paired(old['TEST']['rows'] + old['REMAINDER']['rows'],
            new['TEST']['rows'] + new['REMAINDER']['rows'], previous, arm)
        paired[arm] = dict(**pair, populations=populations)
    evidencepaths = [Path(__file__), Path(P.__file__), Path(PAIR.__file__), Path(OLD.__file__), Path(V.__file__)]
    evidencepaths.extend(auditpaths)
    for folder in (OLD.OUTPUT, V.OUTPUT):
        evidencepaths.extend(folder / name for name in ('PROTOCOL.json', 'INPUT_SEAL.json',
            'PREDICTION_SEAL.json', 'SCORE_SEAL.json', 'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json'))
    for folder in (OLD.OCAL.OUTPUT, OLD.SCAL.OUTPUT, V.CAL.OUTPUT):
        evidencepaths.extend(folder / name for name in ('MODEL.json', 'MODEL_SEAL.json', 'CAL_LOGO_REPORT.json'))
    evidencepaths.append(V.PLAN)
    active = C.read(OUTPUT / 'ACTIVE_GOAL_PROTOCOL_V1.json')
    P._verify(active['protected_user_sha256'])
    bindings = [P._binding(path) for path in evidencepaths]
    data = dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V6_COMPLETE_BI_V23',
        generated_UTC=datetime.now(timezone.utc).isoformat(), goal_status='active', goal_completion_declared=False,
        default_changed=False, default_method=P.BASELINE, methods=methods, method_count=9,
        paired_v22_to_v23=paired, exact_baseline_consistency_verified=True,
        primary=dict(TEST49=49, REMAINDER143=143, total=192, severe=0), separate_CAL=46,
        no_per_frame_best_or_GT_ID_partition_routing=True, all_methods_uniform_full_population=True,
        all238_no_frame_fallback_or_native_flag_filter=True,
        numerical_target_met_any_new_method=any(method['numerical_primary192_target_met'] for method in methods[5:]),
        change='complete-BI edge/tangent likelihood replacement with separately sealed CAL noise; same O/OS profile and selector pairing',
        S_image_2D_or_regions=0, U_regions_only=True, original_hard_guards_unchanged=True,
        original_v19_start_names_policy_unchanged=True, independent_new_image_information_claimed=False,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,
        reused_exploratory_TEST_not_new_holdout=True, no_new_inference_pose_fit_images_or_reference_decode=True,
        independent_root_audits=[P._binding(path) for path in auditpaths], evidence=bindings,
        protected_user_sha256=active['protected_user_sha256'])
    # Finish checking all bytes before the first append-only document write.
    for binding in bindings:
        if C.sha(C.ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Evidence changed during aggregation')
    C.write(paths[0], data)
    paths[1].write_text(_render(data), encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2], dict(source=P._binding(Path(__file__)), json=P._binding(paths[0]),
        html=P._binding(paths[1]), evidence=bindings, exact_e_preserved=True,
        method_count=9, all_declared_arms_and_zero_failure_audits_required=True,
        no_new_inference_pose_fit_images_or_reference_decode=True))
    print(dict(methods=9, json_sha256=C.sha(paths[0]), html_sha256=C.sha(paths[1])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-v23-score-and-root-audit-go', action='store_true', required=True)
    args = parser.parse_args()
    build(root_postscore_GO=args.after_v23_score_and_root_audit_go)
