"""Append-only post-seal O-localization-abstention supplement.

Read only cached, sealed summaries after the owner score and independent root
audit are complete. No inference, pose fitting, image or reference decoding.
Every method is uniform across the original CAL46/TEST49/remaining143 rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import build_repair_campaign_progress_v3_v18 as PREVIOUS

C = PREVIOUS.C
OUTPUT = PREVIOUS.OUTPUT
PARENT = PREVIOUS.BOUNDARY/'refined_grazing_image_chain_mixtures_v4_18c'
CONTROL = PREVIOUS.BOUNDARY/'o_localization_abstention_v4_19'
BASELINE = PREVIOUS.BASELINE
PAIR_ARMS = (
    ('NATIVE', 'REFINED_GRAZING_NMS_NATIVE', 'O_LOCALIZATION_ABSTAINED_NATIVE'),
    ('PLUS_MASK', 'REFINED_GRAZING_NMS_PLUS_MASK', 'O_LOCALIZATION_ABSTAINED_PLUS_MASK'),
)
LABELS = {
    BASELINE: '기존 e',
    'REFINED_GRAZING_NMS_NATIVE': '부모 v18c 연속·NMS',
    'REFINED_GRAZING_NMS_PLUS_MASK': '부모 v18c 연속·NMS+마스크',
    'O_LOCALIZATION_ABSTAINED_NATIVE': 'v19 O 위치 범위 보류',
    'O_LOCALIZATION_ABSTAINED_PLUS_MASK': 'v19 O 위치 범위 보류+마스크',
}


def _paired(before, after, old_arm, new_arm):
    if [r['id']for r in before] != [r['id']for r in after]:
        raise ValueError('Paired original population/order required')
    old = np.array([r['metrics'][old_arm]['translation_cm']for r in before])
    new = np.array([r['metrics'][new_arm]['translation_cm']for r in after])
    delta = new-old
    tolerance = 1e-8  # Reporting-only numerical equality, never a fit parameter.
    old_joint = [bool(r['metrics'][old_arm]['success_5cm_5deg'])for r in before]
    new_joint = [bool(r['metrics'][new_arm]['success_5cm_5deg'])for r in after]
    return dict(count=len(before), mean_translation_delta_cm=float(delta.mean()),
        translation_decreased=int((delta < -tolerance).sum()),
        translation_increased=int((delta > tolerance).sum()),
        numerically_equal=int((np.abs(delta) <= tolerance).sum()),
        reporting_equality_tolerance_cm=tolerance,
        translation_le5_gains=[r['id']for r, a, b in zip(before, old, new)if a > 5 and b <= 5],
        translation_le5_losses=[r['id']for r, a, b in zip(before, old, new)if a <= 5 and b > 5],
        joint_gains=[r['id']for r, a, b in zip(before, old_joint, new_joint)if not a and b],
        joint_losses=[r['id']for r, a, b in zip(before, old_joint, new_joint)if a and not b],
        fallback_count_before=sum(not r['accepted'][old_arm]for r in before),
        fallback_count_after=sum(not r['accepted'][new_arm]for r in after))


def _root_audit(protocol):
    folder = CONTROL/'independent_root_audit_v1'
    audit = C.read(folder/'AUDIT.json')
    invariant = C.read(folder/'CONTROL_INVARIANTS_V1.json')
    seal = C.read(folder/'CONTROL_INVARIANTS_SEAL_V1.json')
    no_o = 238-protocol['changed_O_frames']
    if (audit['rows'] != 238 or audit['primary192'] != 192 or audit['severe'] != 0
            or audit['failure_count'] != 0 or audit['failures']
            or invariant['failure_count'] != 0 or invariant['failures']
            or C.sha(folder/'CONTROL_INVARIANTS_V1.json') != seal['receipt_sha256']
            or invariant['no_O_arm_pose_parity_count'] != 2*no_o
            or invariant['maximum_fixed_pose_likelihood_difference'] >= 1e-8
            or not invariant['all238_kept']
            or audit['prediction_seal']['sha256'] != C.sha(CONTROL/'PREDICTION_SEAL.json')
            or invariant['prediction_seal']['sha256'] != C.sha(CONTROL/'PREDICTION_SEAL.json')
            or invariant['selected_pose_audit']['sha256'] != C.sha(folder/'AUDIT.json')):
        raise ValueError('Complete current no-O parity and independent fresh-guard audit required')
    for binding in (audit['source'], invariant['source'], invariant['kernel_audit_source']):
        if C.sha(C.ROOT/binding['path']) != binding['sha256']:
            raise ValueError('Independent audit source bytes changed')
    return dict(selected_pose_audit=audit, control_invariants=invariant,
        no_O_frames=no_o, no_O_arms=2*no_o,
        expected_no_O_pose_parity_verified=True)


def _markdown(data, output):
    lines = [
        '기존 e를 유지합니다. 이번 v19는 O 위치 원만 보류한 통제이며 새 기본 방법으로 채택하지 않았습니다. V3와 모든 이전 예측·점수·모델은 그대로 보존했습니다.', '',
        'O 상태는 그대로 두고 raw N3 위치 예측 주위의 타원 hard bound만 제거했습니다. 기존 닫힌 이미지 영역과 추가 half-plane은 유지합니다. 실제 가림 물체 마스크를 새로 얻거나 O를 정확한 2D 점으로 맞춘 실험은 아닙니다. S·B/I 관측과 공분산·접선·U/T 영역·NMS/CAL 모델·물리 크기/Q·후보은행·선택식은 같습니다.', '',
        '수치는 cm이며 `공동`은 5cm·5° 모두 만족한 수입니다. Primary192는 기존 TEST49와 추가143장의 합입니다. CAL46은 별도이고, 모든 원래 행·실패·fallback·축 관례의 미확정 사항을 유지했습니다.', '',
    ]
    for key, title in (('PRIMARY192', 'Primary192'), ('TEST', '기존 TEST49'),
                       ('REMAINDER', '추가143'), ('CAL', '별도 CAL46')):
        lines.extend([title, '', '| 방법 | T≤5cm | 공동 | 평균 T | 최대 T | fallback |',
                      '|---|---:|---:|---:|---:|---:|'])
        for method in data['methods']:
            s = method['populations'][key]
            lines.append(f"| {method['label_KO']} | {s['translation_le5']} | {s['successes_5cm_5deg']} | "
                f"{s['translation_cm']['mean']:.3f} | {s['translation_cm']['max']:.3f} | {s['fallback_count']} |")
        lines.append('')
    audit = data['independent_root_audit']
    lines.extend([
        f"O 범위가 없던 {audit['no_O_frames']}장은 두 선택식 모두 부모 v18c와 자세·family·feasibility가 정확히 같았습니다({audit['no_O_arms']}개 arm 확인). O가 있던 경우에도 부모의 feasible 자세는 완화한 hard constraint를 만족하고 같은 고정 자세에서 likelihood는 바뀌지 않았습니다. 다만 완화 후 feasible 후보가 늘면 동일한 최대 6개 시작점 정책의 실제 부분집합과 최적화 경로가 달라질 수 있습니다. 부모 연속 endpoint가 모두 새 후보은행에 들어간다거나 결과의 정확도·목적함수가 단조 개선된다고 주장하지 않습니다.", '',
        '기존 O 범위는 plastic CAL 5장·7점의 raw 위치 오차로 만든 경험적 원이며 wood CAL O 사례는 없습니다. 정확한 명목 참조가 원 밖이라는 진단은 5cm 근방의 모든 자세가 불가능하다는 증거가 아닙니다. 이번에는 GT에 맞춰 반경을 키우거나 정답 자세·family를 선택하지 않고, 같은 규칙으로 해당 위치 제한만 보류했습니다.', '',
        '표의 각 행은 전체 집합에 적용한 단일 방법입니다. 사진별 최선 방법을 섞지 않았으며, native-only와 고정 마스크 가중 선택도 분리했습니다. 관측/목적함수의 개선과 실제 위치 오차 개선은 다르고, 넓은 O 허용 영역이나 accepted 상태는 정확한 코너·유일한 자세·5cm 성공을 뜻하지 않습니다.', '',
        '참조는 기존 고정95 및 추가143의 저장된 annotation/PnP·canonical/C2 정책입니다. 독립 물리 계측이나 새 holdout 일반화를 주장하지 않습니다. 원래 TEST를 반복 사용한 탐색 통제이며, 전체192 목표의 달성·기본 방법 교체는 선언하지 않습니다.', '',
        '비교 전후 gain/loss 사례, 회전·최대 오차, 독립 guard/pose parity 검산과 모든 봉인은 [PROGRESS_V4_V19.json](PROGRESS_V4_V19.json)에 있습니다. 이전 전체 캠페인은 [RESULTS_V3_V18.md](RESULTS_V3_V18.md)에 보존되어 있습니다.', '',
        '[부모 v18c 실제 사진·저장 단계 보고서]('
        + PREVIOUS._link(C.ROOT/data['parent_actual_report']['path'], output)
        + ')는 부모 과정의 시각화이며 새 v19 결과 사진으로 표시하지 않았습니다.',
        '', '[v19 O 위치 범위 보류 실제 과정]('
        + PREVIOUS._link(C.ROOT/data['v19_actual_report']['path'], output)+')',
    ])
    return '\n'.join(lines)+'\n'


def build(output=OUTPUT):
    output = Path(output).resolve()
    paths = [output/name for name in ('PROGRESS_V4_V19.json', 'RESULTS_V4_V19.md', 'PROGRESS_V4_V19_SEAL.json')]
    if any(p.exists()for p in paths):
        raise FileExistsError('Append-only V4: preserve every prior campaign document')
    # Do not call before the root's explicit score-and-independent-audit GO.
    old_protocol, before = PREVIOUS._load_run(PARENT)
    protocol, after = PREVIOUS._load_run(CONTROL)
    if old_protocol['ids'] != protocol['ids']:
        raise ValueError('All original ordered238 rows must remain')
    root_audit = _root_audit(protocol)
    previous_seal = C.read(output/'PROGRESS_V3_V18_SEAL.json')
    for key in ('source', 'json', 'markdown'):
        binding = previous_seal[key]
        if C.sha(C.ROOT/binding['path']) != binding['sha256']:
            raise ValueError('V3 immutable source/document binding changed')
    active = C.read(output/'ACTIVE_GOAL_PROTOCOL_V1.json')
    PREVIOUS._verify(active['protected_user_sha256'])
    for part in PREVIOUS.COUNTS:
        if [r['id']for r in before[part]['rows']] != [r['id']for r in after[part]['rows']]:
            raise ValueError('Partition rows changed')
        if [(r['metrics'][BASELINE], r['accepted'][BASELINE])for r in before[part]['rows']] != [
                (r['metrics'][BASELINE], r['accepted'][BASELINE])for r in after[part]['rows']]:
            raise ValueError('Frozen e baseline changed')
    methods = []
    for reports, folder, arms in ((before, PARENT, (BASELINE,)+tuple(p[1]for p in PAIR_ARMS)),
                                  (after, CONTROL, tuple(p[2]for p in PAIR_ARMS))):
        for arm in arms:
            populations = {}
            for part in PREVIOUS.COUNTS:
                populations[part] = PREVIOUS._summary(reports[part]['rows'], arm)
                PREVIOUS._same_summary(populations[part], reports[part]['summary'][arm])
            populations['PRIMARY192'] = PREVIOUS._summary(reports['TEST']['rows']+reports['REMAINDER']['rows'], arm)
            methods.append(dict(method=arm, label_KO=LABELS[arm], populations=populations,
                source_output=str(folder.relative_to(C.ROOT)), current_default=arm == BASELINE, new_default=False,
                original_TEST49_preserved=populations['TEST']['translation_le5'] == 49,
                numerical_primary192_target_met=populations['PRIMARY192']['translation_le5'] == 192))
    paired = {}
    for name, old, new in PAIR_ARMS:
        paired[name] = {part: _paired(before[part]['rows'], after[part]['rows'], old, new)
                       for part in PREVIOUS.COUNTS}
        paired[name]['PRIMARY192'] = _paired(before['TEST']['rows']+before['REMAINDER']['rows'],
            after['TEST']['rows']+after['REMAINDER']['rows'], old, new)
    evidence_paths = [Path(PREVIOUS.__file__), output/'PROGRESS_V3_V18.json',
        output/'RESULTS_V3_V18.md', output/'PROGRESS_V3_V18_SEAL.json',
        output/'PROGRESS_V3_V18_QA_RECEIPT.json',
        PREVIOUS.BOUNDARY/'o_localization_abstention_v4_19_checks/COMPLETION_AND_PAIRED_RECEIPT_V1.json',
        PREVIOUS.REMAINDER/'cached_coverage_failure_diagnosis_v4_18/O_DOMAIN_AND_EMPTY_BANK_DETAILS_V1.json']
    for folder in (PARENT, CONTROL):
        evidence_paths.extend(folder/name for name in ('PROTOCOL.json', 'INPUT_SEAL.json',
            'PREDICTION_SEAL.json', 'SCORE_SEAL.json', 'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json'))
        evidence_paths.extend(folder.glob('*RECEIPT*.json'))
        evidence_paths.extend(folder.glob('*COMPATIBILITY_SEAL*.json'))
        evidence_paths.extend((folder/'independent_root_audit_v1').glob('*.json'))
    actual = PREVIOUS.BOUNDARY/'refined_grazing_mixture_process_v4_18c'
    evidence_paths.extend([actual/'index.html', actual/'RENDER_SEAL.json'])
    new_actual = PREVIOUS.BOUNDARY/'o_localization_abstention_process_v4_19'
    if not all((new_actual/name).is_file()for name in ('index.html', 'RENDER_SEAL.json')):
        raise ValueError('Wait for the sealed actual v19 process report before final V4 output')
    evidence_paths.extend([new_actual/'index.html', new_actual/'RENDER_SEAL.json'])
    evidence = [PREVIOUS._binding(p)for p in sorted(set(evidence_paths))]
    data = dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V4_V19_O_LOCALIZATION_ABLATION',
        generated_UTC=datetime.now(timezone.utc).isoformat(), generation_source=PREVIOUS._binding(Path(__file__)),
        preparation_before_new_v19_scores=True, execution_after_owner_scores_and_root_audit_GO=True,
        goal_status='active', default_method=BASELINE, default_changed=False, no_control_adoption=True,
        primary=dict(TEST49=49, REMAINDER143=143, total=192, severe=0), CAL46_separate=True,
        methods=methods, paired_parent18c_to_v19=paired,
        numerical_target_met_any_new_control=any(m['numerical_primary192_target_met']for m in methods[3:]),
        intervention=dict(only_O_ellipse_hard_localization_abstained=True, O_state_preserved=True,
            O_closed_canvas_and_extra_halfplanes_retained=True, O_point_or_soft_costs=0,
            exact_old_NMS_CAL_q_covariance_reliability_BI_partial_T4_U_T_S_models_Q_and_bank=True,
            runtime_raw_N3_error_circle_not_observed_occluder_mask=True,
            changed_O_points=protocol['changed_O_points'], changed_O_frames=protocol['changed_O_frames'],
            no_GT_adjusted_radius_or_runtime_router=True,
            same_start_policy_but_feasible_start_subset_can_change=True),
        independent_root_audit=root_audit, full238_no_drop=True, baseline_exact=True,
        parent_original_score_entrypoint_success=False,
        parent_metadata_only_score_adapter_preserved=True,
        new_v19_tangent_metadata_supplied_before_fixed_scorer=True,
        source_binding_scope='conservatively inherited dependency unions, not per-frame execution traces',
        evidence=evidence, parent_actual_report=PREVIOUS._binding(actual/'index.html'),
        v19_actual_report=PREVIOUS._binding(new_actual/'index.html'),
        parent_actual_report_is_not_new_v19_visualization=True,
        reference_caveats=['Fixed95 and unchanged derived143 canonical/C2 reader',
            'All original native flags and unconfirmed signed axes/camera conventions kept',
            'Derived annotation/PnP reference is not independent physical metrology',
            'Repeated TEST exploratory control, not new holdout evidence'],
        new_inference_pose_fits_images_or_reference_decodes=0,
        no_per_frame_best_or_GT_based_method_family_routing=True)
    for bound_protocol in (old_protocol, protocol):
        for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
            PREVIOUS._verify(bound_protocol[key])
    for binding in evidence:
        if C.sha(C.ROOT/binding['path']) != binding['sha256']:
            raise ValueError('Supplement evidence changed during aggregation')
    C.write(paths[0], data)
    paths[1].write_text(_markdown(data, output), encoding='utf-8')
    PREVIOUS._verify(active['protected_user_sha256'])
    C.write(paths[2], dict(source=PREVIOUS._binding(Path(__file__)),
        json=PREVIOUS._binding(paths[0]), markdown=PREVIOUS._binding(paths[1]),
        evidence=evidence, protected_user_sha256=active['protected_user_sha256'],
        all_source_input_score_and_audit_bindings_checked=True,
        no_new_inference_pose_fits_or_reference_decodes=True))
    print(dict(methods=len(methods), json_sha256=C.sha(paths[0]), markdown_sha256=C.sha(paths[1]),
        no_O_frames=root_audit['no_O_frames'], no_O_arm_pose_parity=root_audit['no_O_arms']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--after-v19-score-and-root-audit-go', action='store_true', required=True)
    args = parser.parse_args()
    build(args.output)
