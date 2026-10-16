"""Postseal fixed-input paired effect audit. No fitting, RGB or target reads."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np

HERE = Path(__file__).resolve().parent
FIX = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'scripts/research/pallet_visibility_curriculum_20261013_v1').is_dir())
FINAL = FIX / 'evaluation_bias_v4_1'
BASE = FIX / 'evaluation_v1'
MEAN = FIX / 'coarse_mean_control_v4_1'
ALL, NO_OT, NO_PRIOR = 'SELECTED_V4_ALL_OT', 'SELECTED_V4_NO_OT', 'SELECTED_V4_NO_PRIOR'
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm', 'success_5cm_5deg')
REFERENCES = ('ORIGINAL_N3_PNP', 'FROZEN_STATEWISE_V3', 'OLD_RGB_SINGLE_FAMILY',
              'BOUNDARY_PROGRESSIVE', 'BASE_V4_ZERO_MEAN', 'NO_OT', 'NO_PRIOR', 'COARSE_MEAN_ONLY')
FIXED_CASES = ('eval_pallet07:1778652170735118080', 'wood_183705:000706',
               'eval_pallet09:1778653661653195264')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing audit artifact: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify_map(values):
    for name, digest in values.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Binding changed: ' + name)


def verify_binding(value):
    if sha(ROOT / value['path']) != value['sha256']:
        raise ValueError('Binding changed: ' + value['path'])


def prepare():
    selected = FIX / 'bias_selection_v4_1/SELECTED_CONFIG.json'
    plan = {'schema': 'PREDECLARED_POSTSEAL_EFFECT_AUDIT_V4_1',
        'no_parameter_or_hypothesis_selection': True, 'no_GT_target_or_RGB_reads': True,
        'trigger': 'Final complete98/all3arms prediction seal AND POSTSEAL_SCORING exist and validate before metric reads',
        'primary': {'CAL': 46, 'TEST': 49, 'severe': 0}, 'crops_separate': 3,
        'same49_references': list(REFERENCES), 'metrics': list(METRICS),
        'cluster_interval': {'method': 'Exact ordered session resampling with replacement',
            'draws': 'n_sessions ** n_sessions; expected TEST4^4=256',
            'quantiles': [.025, .975], 'quantile_method': 'linear',
            'weights': 'Each draw duplicates all paired frames of each sampled session; report frame-weighted mean',
            'scope': 'Exploratory interval from four reused sessions; no external generalization or significance claim'},
        'fixed_case_ids': list(FIXED_CASES), 'case_addition': 'All original-success to final-failure primary cases; both CAL and TEST',
        'OT_active_slack_threshold': 1e-5, 'OT_activity_scope': 'Selected accepted pose only; inequalities normalized by frozen solver',
        'pose_change_threshold': {'translation_m': 1e-7, 'rotation_matrix_norm': 1e-6},
        'attribution': {'NO_PRIOR_vs_ORIGINAL': 'Compound B/I correction/state/soft fitting/search/OT; not isolated BI effect',
            'BASE_V4_vs_NO_PRIOR': 'Zero-mean prior effect conditional on same observations',
            'FINAL_vs_BASE_V4': 'CAL median bias effect conditional on verified fresh-observation parity',
            'FINAL_vs_NO_OT': 'All OT constraints effect with same frozen measurements/calibrated covariance/prior/bias',
            'MEAN_ONLY_vs_ORIGINAL': 'CAL global mean shift without B/I fit or O/T',
            'FINAL_vs_MEAN_ONLY': 'Statewise fit/constraints relative to mean-only baseline; not isolated component'},
        'selected_config': binding(selected), 'selection_seal': binding(selected.parent / 'SELECTION_SEAL.json'),
        'preexisting_final_protocol': binding(FINAL / 'PROTOCOL.json'),
        'target_failure_policy': 'If any fixed target fails, report target_achieved false; hypotheses only, no new TEST tuning',
        'limitations': ['Historical annotation-derived physical C2 reference', 'TEST is reused exploratory',
            'Native-incompatible and fallback frames remain included', 'CAL full-model audit is not the conditional LOGO selection score',
            'Initial S is geometry-conditional, so human SELF disagreements are diagnosed separately',
            'Accepted ambiguous pose families do not prove unique pose or exact O/T coordinates']}
    write(HERE / 'AUDIT_PLAN.json', plan)
    write(HERE / 'PREPARATION_SEAL.json', {'schema': 'EFFECT_AUDIT_SOURCE_PLAN_SEAL_V1',
        'source': binding(__file__), 'plan': binding(HERE / 'AUDIT_PLAN.json'),
        'metric_or_target_files_opened': False, 'fits': 0})


def verify_prepared():
    seal = read(HERE / 'PREPARATION_SEAL.json')
    verify_binding(seal['source']); verify_binding(seal['plan'])
    plan = read(HERE / 'AUDIT_PLAN.json')
    for name in ('selected_config', 'selection_seal', 'preexisting_final_protocol'):
        verify_binding(plan[name])
    return plan


def load_sealed(directory, mean_only=False):
    # This function is never called until BOTH final seal and final score exist.
    seal, protocol = read(directory / 'PREDICTION_SEAL.json'), read(directory / 'PROTOCOL.json')
    if (not seal.get('before_own_GT_scoring' if mean_only else 'before_scoring') or seal['rows'] != 98
            or sha(directory / 'PREDICTIONS.jsonl.gz') != seal['predictions_sha256']
            or sha(directory / 'PROTOCOL.json') != seal['protocol_sha256']):
        raise ValueError('Incomplete/broken prediction seal: ' + str(directory))
    if not mean_only:
        if not seal['all3arms_complete'] or seal['annotation_targets_accessed']:
            raise ValueError('Incomplete/tainted prediction seal')
        if sha(directory / 'CONTRACT_CHECKS.json') != seal['contract_checks_sha256']:
            raise ValueError('Contract seal changed')
        for name in ('source_sha256',): verify_map(seal[name])
        verify_map(protocol['input_and_model_sha256']); verify_map(protocol['protected_user_sha256'])
    else:
        verify_map(protocol['source_sha256'])
    score = read(directory / 'POSTSEAL_SCORING.json')
    expected = score['prediction_seal_sha256'] if mean_only else score['prediction_seal']['sha256']
    if sha(directory / 'PREDICTION_SEAL.json') != expected:
        raise ValueError('Score/prediction seal mismatch')
    with gzip.open(directory / 'PREDICTIONS.jsonl.gz', 'rt') as stream:
        predictions = list(map(json.loads, stream))
    if len(predictions) != 98 or len({r['id'] for r in predictions}) != 98:
        raise ValueError('Duplicate/incomplete prediction population')
    if not mean_only and [r['id'] for r in predictions] != protocol['ids']:
        raise ValueError('Prediction ordered population changed')
    return predictions, score, protocol


def stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                P90=float(np.quantile(a, .9)), max=float(a.max()))


def paired(rows, reference, candidate='FINAL'):
    sessions = sorted({r['session'] for r in rows})
    groups = [[r for r in rows if r['session'] == s] for s in sessions]
    draws = np.asarray(list(itertools.product(range(len(sessions)), repeat=len(sessions))), int)
    output = {'frames': len(rows), 'sessions': sessions, 'exact_cluster_draws': len(draws),
              'sign_convention': 'candidate-reference; negative error and positive success are improvements',
              'by_metric': {}, 'by_session': {}}
    for metric in METRICS:
        chunks = [np.array([float(r['metrics'][candidate][metric])-float(r['metrics'][reference][metric]) for r in g]) for g in groups]
        sums, counts = np.array([x.sum() for x in chunks]), np.array([len(x) for x in chunks])
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        flat = np.concatenate(chunks)
        output['by_metric'][metric] = {'mean_difference': float(flat.mean()),
            'exact_session_resampling_interval95': np.quantile(boot, [.025, .975]).tolist(),
            'improved_frames': int((flat > 0).sum()) if metric == 'success_5cm_5deg' else int((flat < 0).sum()),
            'worsened_frames': int((flat < 0).sum()) if metric == 'success_5cm_5deg' else int((flat > 0).sum()),
            'equal_frames': int((flat == 0).sum())}
    for session, g in zip(sessions, groups):
        output['by_session'][session] = {'frames': len(g), 'candidate_success': sum(r['metrics'][candidate]['success_5cm_5deg'] for r in g),
            'reference_success': sum(r['metrics'][reference]['success_5cm_5deg'] for r in g),
            'mean_differences': {m: float(np.mean([float(r['metrics'][candidate][m])-float(r['metrics'][reference][m]) for r in g])) for m in METRICS}}
    output['success_gain_ids'] = [r['id'] for r in rows if r['metrics'][candidate]['success_5cm_5deg'] and not r['metrics'][reference]['success_5cm_5deg']]
    output['success_loss_ids'] = [r['id'] for r in rows if not r['metrics'][candidate]['success_5cm_5deg'] and r['metrics'][reference]['success_5cm_5deg']]
    return output


def domain_diagnostics(runtime, scored):
    chosen = {r['domain_index']: r for r in runtime['joint_pose']['joint_constraint_audit']}
    records = {r['native_id']: r for r in runtime['state_observation']['records']}
    result = []
    for index, domain in enumerate(runtime['ot_domains']['domains']):
        k = domain['native_corner']; state = domain['state']; human = scored['human_states_native'][k]
        audit = chosen.get(index)
        physical_T = bool(records.get(k, {}).get('physical_truncation_certified', False))
        category = ('SOURCE_EXTERNAL_OCCLUSION_O' if state == 'O' and human == 'EXTERNAL_OCCLUDED'
                    else 'O_ON_SOURCE_DIRECT' if state == 'O' and human == 'DIRECT_VISIBLE'
                    else 'O_ON_SOURCE_SELF' if state == 'O' and human == 'SELF_OCCLUDED'
                    else 'O_UNANNOTATED_OR_UNRESOLVED' if state == 'O'
                    else 'INDEPENDENTLY_CERTIFIED_T' if physical_T else 'INITIAL_GEOMETRY_CONDITIONAL_T')
        active = [] if audit is None else [q for q in audit['inequalities'] if q['slack'] <= 1e-5]
        result.append({'native_corner': k, 'state': state, 'shape': domain['shape'], 'source_human_state': human,
            'support_category': category, 'physical_truncation_certified': physical_T,
            'outside_canvas_side': domain.get('outside_canvas_side'), 'accepted_pose_constraint_audit': audit,
            'active_inequalities': active, 'selected_constraint_active': bool(active),
            'region_provenance': deepcopy(domain.get('provenance', {}))})
    return result


def observation_comparison(a, b):
    am, bm = a['frozen_visible_measurements'], b['frozen_visible_measurements']
    if [m['native_corner'] for m in am] != [m['native_corner'] for m in bm]:
        raise ValueError('Final/base BI native IDs differ')
    def maxdiff(name):
        return max((float(np.max(np.abs(np.array(x[name])-np.array(y[name])))) for x, y in zip(am, bm)), default=0.)
    return {'states_exact': a['states'] == b['states'],
        'raw_and_initial_pose_exact': a['original_raw_points'] == b['original_raw_points'] and a['original_coarse_pose'] == b['original_coarse_pose'],
        'observed_BI_xy_max_abs_px': maxdiff('xy'), 'observed_BI_covariance_max_abs_px2': maxdiff('covariance_px2'),
        'observed_BI_records_exact': am == bm, 'calibrated_BI_records_exact': a['calibrated_visible_measurements'] == b['calibrated_visible_measurements'],
        'OT_domains_exact': a['ot_domains']['domains'] == b['ot_domains']['domains'],
        'nonS_scale_exact': a['ot_domains']['nonself_prediction_scale_px'] == b['ot_domains']['nonself_prediction_scale_px']}


def runtime_record(prediction, scored, base):
    rt = prediction['runtime']; initial_S = {k for k, h in enumerate(rt['initial_geometry']['initial_self_hidden']) if h}
    human_self = {k for k, h in enumerate(scored['human_states_native']) if h == 'SELF_OCCLUDED'}
    queried = set(rt['state_observation'].get('rgb_queried_ids', [])); corrected = set(rt['visible_correction'].get('queried_native_ids', []))
    measured = {m['native_corner'] for m in rt['frozen_visible_measurements']}
    if initial_S & (queried | corrected | measured): raise ValueError('Initial S observation contract broken')
    domains = domain_diagnostics(rt, scored)
    return {k: deepcopy(scored[k]) for k in ('states', 'human_states_native', 'physical_native_compatible', 'physical_native_gap_m',
        'BI_count', 'OT_count', 'joint_accepted', 'material_ambiguity', 'actual_pose_source', 'actual_pose_fallback',
        'OT_changes_pose_vs_no_OT', 'pose_delta_all_minus_no_OT')} | {
        'BI_state_counts': dict(Counter(m['state'] for m in rt['frozen_visible_measurements'])),
        'BI_source_counts': dict(Counter(str(m.get('source')) for m in rt['frozen_visible_measurements'])),
        'initial_S_ids': sorted(initial_S), 'source_human_SELF_ids': sorted(human_self),
        'source_human_SELF_not_initial_S_ids': sorted(human_self-initial_S),
        'source_human_SELF_state_RGB_queried_ids': sorted(human_self & queried),
        'source_human_SELF_BI_correction_queried_ids': sorted(human_self & corrected),
        'source_human_SELF_BI_measurement_ids': sorted(human_self & measured),
        'initial_S_image_query_contract_passed': True, 'human_SELF_disagreement_does_not_redefine_initial_S': True,
        'OT_domains': domains, 'OT_active_support_categories': dict(Counter(d['support_category'] for d in domains if d['selected_constraint_active'])),
        'globally_unique_pose_claimed': False, 'pose_family_status': rt['joint_pose']['status'],
        'visible_residual_audit': rt['joint_pose'].get('visible_protection_audit', []),
        'objective_components': rt['joint_pose'].get('objective_components'),
        'final_model_visibility_audit': rt['joint_pose'].get('final_model_visibility_audit'),
        'BI_frozen_records': rt['frozen_visible_measurements'], 'calibrated_BI_records': rt['calibrated_visible_measurements'],
        'final_vs_base_observation_parity': observation_comparison(rt, base['runtime'])}


def run():
    plan = verify_prepared()
    # Before this gate, only source/protocol/config metadata may be inspected.
    if not all((FINAL / name).is_file() for name in ('PREDICTION_SEAL.json', 'POSTSEAL_SCORING.json')):
        raise RuntimeError('Wait for final all98/all3arms prediction seal AND postseal score')
    inputs = {str((d/n).relative_to(ROOT)):sha(d/n) for d in (FINAL,BASE,MEAN) for n in ('PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json','PROTOCOL.json','POSTSEAL_SCORING.json')}
    final, score, final_protocol = load_sealed(FINAL)
    baseline, base_score, _ = load_sealed(BASE)
    mean, mean_score, _ = load_sealed(MEAN, True)
    if final_protocol['solver_configuration'] != read(ROOT / plan['selected_config']['path'])['solver_configuration']:
        raise ValueError('Final protocol differs from the frozen selected bias solver')
    by_id, base_id, mean_id = ({r['id']: r for r in x} for x in (final, baseline, mean))
    if set(by_id) != set(base_id) or set(by_id) != set(mean_id): raise ValueError('Complete98 matched IDs differ')
    scored_id = {r['id']: r for r in score['rows']}; base_scored = {r['id']: r for r in base_score['rows']}; mean_scored = {r['id']: r for r in mean_score['rows']}
    if set(scored_id) != set(by_id) or set(base_scored) != set(by_id) or set(mean_scored) != set(by_id): raise ValueError('Scored population differs')
    for part in ('CAL', 'TEST'):
        if score['label_bindings'][part] != base_score['label_bindings'][part] or score['label_bindings'][part]['sha256'] != mean_score['label_sha256'][part]:
            raise ValueError('Physical reference bindings differ')
    rows = []
    for fid, r in by_id.items():
        s = scored_id[fid]; metrics = {'FINAL': s['metrics'][ALL],
            **{arm: s['metrics'][arm] for arm in REFERENCES[:4] if arm in s['metrics']},
            'BASE_V4_ZERO_MEAN': base_scored[fid]['metrics'][ALL], 'NO_OT': s['metrics'][NO_OT],
            'NO_PRIOR': s['metrics'][NO_PRIOR], 'COARSE_MEAN_ONLY': mean_scored[fid]['metrics']}
        if any(s['metrics'][arm] != base_scored[fid]['metrics'][arm] for arm in REFERENCES[:4] if arm in s['metrics']):
            raise ValueError('Historical same-pose comparator metrics differ across scorer versions')
        # Historical comparator arms are absent for the three crop controls.
        if r['cohort'] != 'PRIMARY_NONSEVERE':
            metrics = {'FINAL': s['metrics'][ALL], 'ORIGINAL_N3_PNP': s['metrics']['ORIGINAL_N3_PNP'],
                'BASE_V4_ZERO_MEAN': base_scored[fid]['metrics'][ALL], 'NO_OT': s['metrics'][NO_OT],
                'NO_PRIOR': s['metrics'][NO_PRIOR], 'COARSE_MEAN_ONLY': mean_scored[fid]['metrics']}
        rows.append({k: r[k] for k in ('id', 'source_record_id', 'partition', 'session', 'cohort')} |
                    {'metrics': metrics, 'diagnostics': runtime_record(r, s, base_id[fid])})
    primary = [r for r in rows if r['cohort'] == 'PRIMARY_NONSEVERE']; crops = [r for r in rows if r['cohort'] != 'PRIMARY_NONSEVERE']
    if len(primary) != 95 or len(crops) != 3: raise ValueError('95primary+3crop contract changed')
    summaries, comparisons, attribution = {}, {}, {}
    for part, n in (('CAL', 46), ('TEST', 49)):
        group = [r for r in primary if r['partition'] == part]
        if len(group) != n: raise ValueError('Primary partition changed')
        if part == 'TEST' and len({r['session'] for r in group}) != 4: raise ValueError('Expected four reused TEST sessions')
        arms = ['FINAL', *REFERENCES]
        summaries[part] = {a: {'frames': n, 'success_count': sum(r['metrics'][a]['success_5cm_5deg'] for r in group),
            'metrics': {m: stats([r['metrics'][a][m] for r in group]) for m in METRICS[:3]}} for a in arms}
        comparisons[part] = {a: paired(group, a) for a in REFERENCES}
        attribution[part] = {'compound_BI_state_fit_search_OT_vs_original': paired(group, 'ORIGINAL_N3_PNP', 'NO_PRIOR'),
            'zero_mean_prior_given_observations': paired(group, 'NO_PRIOR', 'BASE_V4_ZERO_MEAN'),
            'median_bias_given_observation_parity': paired(group, 'BASE_V4_ZERO_MEAN', 'FINAL'),
            'OT_given_same_measurements_and_prior_bias': paired(group, 'NO_OT', 'FINAL'),
            'mean_only_vs_original': paired(group, 'ORIGINAL_N3_PNP', 'COARSE_MEAN_ONLY'),
            'statewise_vs_mean_only': paired(group, 'COARSE_MEAN_ONLY', 'FINAL')}
    regressions = [r['id'] for r in primary if r['metrics']['ORIGINAL_N3_PNP']['success_5cm_5deg'] and not r['metrics']['FINAL']['success_5cm_5deg']]
    selected_case_ids = list(dict.fromkeys([*FIXED_CASES, *regressions]))
    lookup = {r['id']: r for r in rows}
    detail = {'rows': rows, 'fixed_and_original_success_regression_cases': [lookup[fid] for fid in selected_case_ids],
        'all_original_success_to_final_failure_ids': regressions,
        'all_OT_pose_changed_cases': [r for r in rows if r['diagnostics']['OT_changes_pose_vs_no_OT']],
        'all_OT_selected_active_constraint_cases': [r for r in rows if any(d['selected_constraint_active'] for d in r['diagnostics']['OT_domains'])],
        'source_human_SELF_BI_cases': [r for r in rows if r['diagnostics']['source_human_SELF_BI_measurement_ids']]}
    contract = {'primary95_included': True, 'primary_partition': {'CAL': 46, 'TEST': 49, 'severe': 0}, 'crops_excluded_from_primary': 3,
        'fallbacks': {p: [r['id'] for r in primary if r['partition']==p and r['diagnostics']['actual_pose_fallback']] for p in ('CAL', 'TEST')},
        'native_incompatible': {p: [r['id'] for r in primary if r['partition']==p and not r['diagnostics']['physical_native_compatible']] for p in ('CAL', 'TEST')},
        'ambiguity': {p: [r['id'] for r in primary if r['partition']==p and r['diagnostics']['material_ambiguity']] for p in ('CAL', 'TEST')},
        'initial_S_queries': 0, 'GT_runtime_inputs': False, 'source_human_SELF_BI_corners': {p: sum(len(r['diagnostics']['source_human_SELF_BI_measurement_ids']) for r in primary if r['partition']==p) for p in ('CAL','TEST')},
        'OT_information': {p: {'BI_ge3_changed': sum(r['diagnostics']['BI_count']>=3 and r['diagnostics']['OT_changes_pose_vs_no_OT'] for r in primary if r['partition']==p),
            'BI2_accepted_families': sum(r['diagnostics']['BI_count']==2 and r['diagnostics']['joint_accepted'] for r in primary if r['partition']==p),
            'selected_active_support_categories': dict(Counter(d['support_category'] for r in primary if r['partition']==p for d in r['diagnostics']['OT_domains'] if d['selected_constraint_active']))} for p in ('CAL','TEST')}}
    parity = [r['diagnostics']['final_vs_base_observation_parity'] for r in rows]
    contract['final_vs_base_observations'] = {'states_and_raw_initial_exact': all(p['states_exact'] and p['raw_and_initial_pose_exact'] for p in parity),
        'all_frozen_BI_records_exact': all(p['observed_BI_records_exact'] for p in parity), 'all_effective_BI_records_exact': all(p['calibrated_BI_records_exact'] for p in parity),
        'all_OT_domains_exact': all(p['OT_domains_exact'] for p in parity),
        'max_BI_xy_difference_px': max(p['observed_BI_xy_max_abs_px'] for p in parity),
        'max_BI_covariance_difference_px2': max(p['observed_BI_covariance_max_abs_px2'] for p in parity)}
    contract['NO_PRIOR_base_vs_final_metric_max_abs_difference'] = max(abs(float(scored_id[fid]['metrics'][NO_PRIOR][m])-float(base_scored[fid]['metrics'][NO_PRIOR][m])) for fid in by_id for m in METRICS)
    target = {'required': score['required_target'], 'checks': score['target_checks'], 'target_achieved': score['target_achieved'],
        'overall_goal_complete_claim_allowed': bool(score['target_achieved']),
        'no_postscore_parameter_selection': True}
    report = {'schema': 'POSTSEAL_PAIRED_EFFECT_AUDIT_V4_1', 'summary': summaries, 'paired': comparisons,
        'attribution': attribution, 'population_and_semantic_audit': contract, 'fixed_numeric_target_status': target,
        'limits': plan['limitations'], 'statistics': plan['cluster_interval'],
        'CAL_grounded_unselected_hypotheses_if_target_failed': ['Correct fixed native model-family/correspondence inconsistencies before fitting poses',
            'Test CAL-only uncertainty in O/T and coarse self-visibility gates; retain initial S query exclusion',
            'Distinguish global CAL coarse mean correction from image-statewise improvements with predeclared same-input controls'] if not score['target_achieved'] else []}
    verify_map(inputs); verify_prepared()
    write(HERE / 'EFFECT_AUDIT.json', report); write(HERE / 'FRAME_DETAILS.json', detail)
    lines = ['# V4.1 봉인 후 동일 프레임 효과 감사', '', '선택 설정은 바꾸지 않았습니다. 모든 비교는 같은 TEST49와 물리 C2 기준을 사용하며, fallback·native 불일치 프레임을 포함합니다. CAL46은 fullCAL 모델 점수이며 LOGO 선택 점수와 구분합니다.', '', '| 방법 | TEST 성공/49 | 평균 T(cm) | 평균 R(deg) |', '|---|---:|---:|---:|']
    for arm, value in summaries['TEST'].items():
        lines.append(f'| {arm} | {value["success_count"]} | {value["metrics"]["translation_cm"]["mean"]:.6f} | {value["metrics"]["rotation_deg"]["mean"]:.6f} |')
    lines += ['', '고정 수치 목표 충족: **' + str(target['target_achieved']) + '**. 이 결과만으로 통계적 우월성이나 새 환경 일반화를 주장하지 않습니다.', '',
        'B/I 관측 교정의 순수 효과는 현재 대조군으로 분리할 수 없습니다. NO_PRIOR 대비 초기값 차이는 상태·soft fitting·검색·O/T가 섞인 효과입니다. zero-mean prior, CAL median bias, O/T, mean-only 차이는 JSON의 조건부 paired 비교에 따로 기록했습니다.', '',
        'O/T 효과는 동일 B/I·유효 공분산·prior·bias를 유지한 NO_OT 비교로 판단했습니다. source EXTERNAL O와 초기 기하에 따른 conditional T를 구분했고, BI2의 수락은 모호한 pose family입니다.', '',
        '초기 S의 이미지 질의는 0입니다. annotation SELF가 B/I로 들어간 사례는 초기 coarse self-visibility와 과거 annotation 간 불일치로 별도 기록했습니다.', '',
        '4개 세션을 복원추출한 256개 모든 ordered draw에서 구간을 계산했습니다. 재사용 TEST의 탐색적 구간이며 독립 표본의 유의성 검정이 아닙니다.', '',
        '전체 프레임·고정 사진 사례·초기 성공→최종 실패·활성 O/T는 FRAME_DETAILS.json에 있습니다. 원본 봉인 소스·예측·선택은 수정하지 않았습니다.']
    (HERE / 'AUDIT_KO.md').write_text('\n'.join(lines)+'\n')
    verify_map(inputs); verify_prepared()
    write(HERE / 'COMPLETION_RECEIPT.json', {'schema':'POSTSEAL_EFFECT_AUDIT_COMPLETION_V1',
        'source':binding(__file__),'plan':binding(HERE/'AUDIT_PLAN.json'),'input_sha256':inputs,
        'outputs':[binding(HERE/n) for n in ('EFFECT_AUDIT.json','FRAME_DETAILS.json','AUDIT_KO.md')],
        'GT_target_files_opened':False,'RGB_reads':0,'new_fits':0,'parameter_changes':0,
        'target_achieved':score['target_achieved'],'Python':platform.python_version(),'numpy':np.__version__})
    print(json.dumps({'TEST':summaries['TEST'],'target':target,'OT':contract['OT_information']},ensure_ascii=False))


def test():
    rows=[]
    for session,n,delta in [('A',2,1),('B',1,-2),('C',1,0),('D',1,3)]:
        for k in range(n):
            base={'translation_cm':5.,'rotation_deg':6.,'ADDsym_cm':7.,'success_5cm_5deg':False}
            final=base|{'translation_cm':5.+delta,'success_5cm_5deg':delta<0}
            rows.append({'id':session+str(k),'session':session,'metrics':{'FINAL':final,'REF':base}})
    result=paired(rows,'REF')
    assert result['exact_cluster_draws']==256 and result['by_metric']['translation_cm']['mean_difference']==.6
    assert result['success_gain_ids']==['B0'] and not result['success_loss_ids']
    assert result==paired(list(reversed(rows)),'REF') or result['by_metric']==paired(list(reversed(rows)),'REF')['by_metric']
    tiny=[{'id':'x','session':'only','metrics':{'FINAL':{'translation_cm':1.,'rotation_deg':2.,'ADDsym_cm':3.,'success_5cm_5deg':True},'REF':{'translation_cm':2.,'rotation_deg':3.,'ADDsym_cm':4.,'success_5cm_5deg':False}}}]
    assert paired(tiny,'REF')['by_metric']['translation_cm']['exact_session_resampling_interval95']==[-1.,-1.]
    rt={'joint_pose':{'joint_constraint_audit':[{'domain_index':0,'inequalities':[{'slack':0.,'name':'outside_right'}]}]},
        'state_observation':{'records':[{'native_id':0,'physical_truncation_certified':False},{'native_id':1}]},
        'ot_domains':{'domains':[{'native_corner':0,'state':'T','shape':'halfplane'},
            {'native_corner':1,'state':'O','shape':'ellipse'}]}}
    dd=domain_diagnostics(rt,{'human_states_native':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']})
    assert dd[0]['selected_constraint_active'] and dd[0]['support_category']=='INITIAL_GEOMETRY_CONDITIONAL_T'
    assert not dd[1]['selected_constraint_active'] and dd[1]['support_category']=='SOURCE_EXTERNAL_OCCLUSION_O'
    print('4 focused weighted-session/success/OT-support tests PASS; no dataset reads')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','test'));a=p.parse_args()
    {'prepare':prepare,'run':run,'test':test}[a.phase]()
