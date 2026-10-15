"""Predeclared seven-method actual process; render only after explicit root GO.

Saved poses, global samples, RGB, masks, state/observation records and sealed
scores only. No inference, observation search, pose fit or reference geometry.
The first coarse wire is the original N3 pose, never e's numerical pose.
"""
from pathlib import Path
import argparse
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from . import common as C
from . import evaluate_coarse_prior_containment_v4_25 as EXP
from . import render_source_preserving_tangent_mask_process_v4_24 as PREVIOUS

OLD, DRAW, CONTEXT, ACCOUNT = PREVIOUS.OLD, PREVIOUS.DRAW, PREVIOUS.CONTEXT, PREVIOUS.ACCOUNT
OUTPUT = EXP.OUTPUT.parent / 'coarse_prior_containment_process_v4_25'
PLAN = EXP.OUTPUT.parent / 'coarse_prior_containment_report_plan_v4_25'
AUDIT_SUBDIR = 'independent_root_audit_v1'
CASES = PREVIOUS.CASES
BASELINE, TP_NATIVE, TP_C, TP_LOCK, FULL_NATIVE, FULL_C, FULL_LOCK = (EXP.ARMS[i] for i in (0, 1, 3, 4, 5, 7, 8))
METHODS = (BASELINE, TP_NATIVE, TP_C, TP_LOCK, FULL_NATIVE, FULL_C, FULL_LOCK)
CONTAINMENT_METHODS = (TP_C, TP_LOCK, FULL_C, FULL_LOCK)
LOCKED_METHODS = (TP_LOCK, FULL_LOCK)
NAMES = {BASELINE:'기본 e', TP_NATIVE:'이동 prior · native', TP_C:'이동 prior · containment',
    TP_LOCK:'같은 이동+C · e family 고정', FULL_NATIVE:'6D prior · native',
    FULL_C:'6D prior · containment', FULL_LOCK:'같은 6D+C · e family 고정'}
COLORS = dict(zip(METHODS, ('#2879c3', '#b48a16', '#169d7a', '#e37136', '#ae64bc', '#5d52b9', '#d84b7d')))
TITLES = ('① 초기 모델·고정 B/I', '② 실제 마스크·보관 NMS 대안', '③ O 캔버스·U 범위·기존 부분 제약',
    '④ 별도 bounded O/S prior', '⑤ SOURCE 비용 계획', '⑥ 원래 첫 N3 pose prior', '⑦ soft containment: 내부/외부',
    '⑧ 기본 e 저장 자세', '⑨ 이동 prior native 저장 자세', '⑩ 이동 prior containment 저장 자세',
    '⑪ 같은 이동+C · e family 고정', '⑫ 6D prior native 저장 자세', '⑬ 6D prior containment 저장 자세',
    '⑭ 같은 6D+C · e family 고정')
CAPTIONS = (
    '회색은 초기 모델 wire. 분홍 ×는 고정 complete B/I, 회색 +는 기존 부분관측의 보관 q. S/U 입력 점은 조회하거나 표시하지 않음.',
    '녹색은 실제 저장 마스크. 회색은 보관 NMS 대안, 청록은 이동+C 선택 family의 활성 변에 속한 보관 대안이며 모든 방법 공통 활성 계획이 아님. ⑤는 두 native 방법의 실제 비용 계획이고 각 C 방법의 성분은 JSON에 보존. 같은 지원점의 여러 fit은 독립 관측이 아님.',
    'O hard domain은 닫힌 영상 캔버스, U는 경험 범위만 사용. 기존 부분 B/I의 법선·접선 구간을 유지하며 정밀 KP를 새로 만들지 않음.',
    'O 타원은 coarse raw 위치의 bounded soft 가정이며 실제 occluder 외곽이 아님. S는 초기 모델 가시성 prior만 사용하고 영상·2D·범위 관측을 만들지 않음.',
    '각 native 방법의 선택 family에서 SOURCE 비용 계획을 읽음. RGB/unknown complete q는 유지하고 정확히 backbone fallback만 대체 가능. 접선 σCAL=√(1+γa), 원래 guide·유효 Σ·신뢰도는 보존.',
    '저장된 ORIGINAL 첫 N3 pose의 wire와 모델 중심만 표시. e 수치자세가 아님. 이동 prior는 원래 3개 unwhitened 이동 잔차와 marginal Σ6[3:,3:]를 사용; 6D는 전체 Σ6. CAL μ를 whitening 전에 한 번만 빼며 λ=β=1.',
    '저장 모델의 full 미클립 convex hull과 고정 관측 boundary sample만 표시. 녹색은 내부/경계(비용0), 주황은 외부. 외부만 기존 reverse .5 평균 robust 비용; 모델→관측 비용0. 표본은 전역 마스크 증거이며 S/U KP 조회가 아님.',
    '기존 e 저장 모델 wire. 현재 N/O/S/P/C 비용을 e에 재평가하거나 재사용하지 않으며 GT wire가 아님.',
    'SOURCE+CAL 접선·bounded O/S·이동 CAL prior. 기존 whole-mask 비용이나 수치 동률 mask tie-break를 사용하지 않음.',
    '같은 이동 prior bank에 같은 모드 native endpoint를 유지하고 soft C를 최적화와 선택에 한 번 더함. 더 낮은 비용이 더 정확한 포즈를 보장하지 않음.',
    '같은 이동+C 전체 후보에서 e의 예측 family 이름만 인증 전에 고정. e 수치 pose를 복사하지 않으며 요청 family에 수용해가 없으면 own coarse fallback/실패로 남김.',
    '이동만이 아닌 전체 6D CAL prior를 사용한 같은 SOURCE native 방법. backbone과 상관된 regularizer이며 새로운 영상 정보·독립성·포함률 인증이 아님.',
    '같은 6D native bank+endpoint와 soft C. C는 내부0/외부만이며 SAM 오염이나 모델 확대·깊이 모호성을 스스로 해결하지 못함.',
    '같은 6D+C를 e 예측 family 이름으로만 제한. 모든 실패·fallback·원래 인구집합을 유지하고 사례별 최적 방법을 섞지 않음.')


def _binding(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)), C.sha(path)


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(), OUTPUT), safe='/')


def _dependency_paths():
    modules = (EXP.V.TCAL, EXP.V.SOURCE_CAL, EXP.V.V.CAL, EXP.OLD.OCAL, EXP.OLD.SCAL)
    paths = [Path(EXP.__file__), Path(EXP.P.__file__), Path(EXP.P.FACTOR.__file__), Path(EXP.PRIOR.__file__),
        Path(PREVIOUS.__file__), Path(OLD.__file__), Path(DRAW.__file__), Path(CONTEXT.__file__), Path(ACCOUNT.__file__),
        EXP.PLAN, EXP.PLAN.with_name('METADATA_CLARIFICATION_V1.json'),
        EXP.POSE_MODEL.SELECTION, EXP.POSE_MODEL.SELECTION.with_name('SELECTION_SEAL.json'),
        EXP.POSE_MODEL.E.FIX / 'calibration/final_adapter_CAL46/MODEL.json']
    paths += [module.OUTPUT / name for module in modules for name in ('MODEL.json', 'MODEL_SEAL.json')]
    return paths


def prepare_plan():
    if PLAN.exists():
        raise FileExistsError('Preserve each predeclaration')
    if (EXP.OUTPUT / 'SCORE_SEAL.json').exists():
        raise ValueError('Declare before new25 scores')
    PLAN.mkdir(parents=True)
    source = Path(__file__).resolve()
    (PLAN / source.name).write_bytes(source.read_bytes())
    C.write(PLAN / 'PREDECLARATION.json', dict(schema='COARSE_PRIOR_CONTAINMENT_ACTUAL_PROCESS_PLAN_V25',
        case_ids=list(CASES), same_cases_as_v23_v24=True, uniform_methods=list(METHODS),
        declared_all9_arms=list(EXP.ARMS), stages_per_case=len(TITLES), renderer_source_sha256=C.sha(source),
        declared_before_new25_metric_read=True, requires_all238_all9_prediction_score_and_zero_failure_independent_audit=True,
        requires_explicit_root_postscore_audit_memory_GO=True, case_best_method_or_reference_routing=False,
        original_first_N3_prior_not_e_numeric_pose=True, TP_uses_marginal_covariance=True, bias_before_whitening_once=True,
        global_containment_stored_hull_and_samples_only=True, old_full_mask_or_native_mask_tie_used=False,
        S_U_point_markers=False, U_regions_only=True, no_GT_wire_or_reference_pose_corner_reads=True,
        no_new_image_model_or_pose_fit=True, default_e_unchanged=True,
        dependencies=dict(_binding(path) for path in _dependency_paths())))
    print(dict(predeclared_cases=list(CASES), methods=list(METHODS), stages=len(TITLES), new25_metrics_read=False), flush=True)


def _load_plan():
    plan = C.read(PLAN / 'PREDECLARATION.json')
    if (tuple(plan['case_ids']) != CASES or tuple(plan['uniform_methods']) != METHODS
            or plan['declared_all9_arms'] != list(EXP.ARMS) or not plan['declared_before_new25_metric_read']
            or C.sha(__file__) != plan['renderer_source_sha256']):
        raise ValueError('Exact predeclared cases/methods/source required')
    for path, digest in plan['dependencies'].items():
        if C.sha(C.ROOT / path) != digest:
            raise ValueError('Predeclared source/model dependency changed')
    return plan


def _completed():
    folder = EXP.OUTPUT
    protocol, pred, score = (C.read(folder / name) for name in ('PROTOCOL.json', 'PREDICTION_SEAL.json', 'SCORE_SEAL.json'))
    if (pred['rows'] != 238 or not pred.get('all9_arms_complete') or not pred.get('all_declared_arms_complete')
            or pred['arms'] != list(EXP.ARMS) or pred['arm_count'] != 9
            or C.sha(folder / 'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(folder / 'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(folder / 'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(folder / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']):
        raise ValueError('Complete9arm/238 prediction+score required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        EXP.E._verify_sources(protocol[key])
    auditpath = folder / AUDIT_SUBDIR / 'AUDIT.json'
    sealpath = auditpath.with_name('AUDIT_SEAL.json')
    audit, aseal = C.read(auditpath), C.read(sealpath)
    if (audit['rows'] != 238 or audit['failure_count'] or audit['failures']
            or C.sha(auditpath) != aseal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(folder / 'PREDICTION_SEAL.json')
            or not audit['no_old_full_outline_factor_or_numeric_e_fallback']):
        raise ValueError('Independent current zero-failure full238 audit required')
    selected, ordered = {}, []
    for row in C.rows(folder / 'PREDICTIONS.jsonl.gz'):
        if set(row['arms']) != set(EXP.ARMS):
            raise ValueError('Every row must contain all nine arms')
        ordered.append(row['id'])
        if row['id'] in CASES:
            selected[row['id']] = row
    if ordered != protocol['ids'] or len(set(ordered)) != 238 or set(selected) != set(CASES):
        raise ValueError('All238 and four fixed cases required')
    scores, reports = {}, {}
    for part, count in ACCOUNT.COUNTS.items():
        path = folder / (part + '_SCORE.json')
        if C.sha(path) != score['score_sha256'][part]:
            raise ValueError('Sealed score changed')
        report = C.read(path)
        if report['count'] != count or len(report['rows']) != count:
            raise ValueError('No frame/fallback/native exclusion')
        reports[part] = report
        scores.update({r['id']:r for r in report['rows']})
    if set(scores) != set(ordered):
        raise ValueError('All238 score inventory required')
    return protocol, selected, scores, reports, [auditpath, sealpath]


def _energies(arm, key):
    if key == BASELINE:
        return dict(applied=False, native=None, O=None, S=None, P=None, C=None, weighted_C=None, total=None)
    if not arm['accepted']:
        return dict(applied=True, native=None, O=None, S=None, P=None, C=None, weighted_C=None, total=None, failed_joint=True)
    solution = arm['solution']
    priors = solution['regularizer_costs']
    if [r['kind'] for r in priors] != list(EXP.P.REGULARIZER_KINDS):
        raise ValueError('Exactly separate O/S/CAL coarse priors required')
    native = float(solution['native_observation_objective'])
    ov, sv, pv = [float(r['weighted_cost']) for r in priors]
    if (not np.isclose(solution['total_regularized_objective'], native + ov + sv + pv, atol=1e-9, rtol=1e-10)
            or not np.isclose(solution['CAL_coarse_pose_prior_objective'], pv, atol=1e-9, rtol=1e-10)
            or solution['coarse_pose_prior_count'] != 1):
        raise ValueError('Native and three-prior ledger changed')
    selection = arm['candidate_selection']
    selected = next(r for r in selection['candidates'] if r['name'] == selection['selected_name'])
    containment = key in CONTAINMENT_METHODS
    shape, weighted, count = 0., 0., 0
    if containment:
        shape, weighted = float(solution['global_containment_cost']), float(solution['weighted_containment_objective'])
        count = solution['containment_count_in_optimizer']
        if (count not in (0, 1) or not np.isclose(weighted, solution['global_containment_weight'] * shape, atol=1e-9, rtol=1e-10)
                or not np.isclose(selected['global_containment_cost'], shape, atol=1e-9, rtol=1e-10)):
            raise ValueError('Containment ledger must enter once')
    elif selected['global_containment_cost'] != 0.:
        raise ValueError('Native-only arm must not reuse an old shape cost')
    if (not np.isclose(solution['objective'], native + ov + sv + pv + weighted, atol=1e-9, rtol=1e-10)
            or not np.isclose(selected['total_energy'], solution['objective'], atol=1e-9, rtol=1e-10)
            or selection['old_full_outline_factor_or_tie_break_used'] is not False):
        raise ValueError('Same optimizer/selector N+O+S+P+lambdaC required')
    expected_mode = 'TRANSLATION' if key in (TP_NATIVE, TP_C, TP_LOCK) else 'FULL'
    prior_audit = solution['CAL_coarse_pose_prior_audit']
    if (prior_audit['mode'] != expected_mode or prior_audit['mean_bias_subtracted_once'] is not True
            or prior_audit['translation_covariance_is_marginal_block'] != (expected_mode == 'TRANSLATION')):
        raise ValueError('Exact prior mode/marginal covariance/mean-once audit required')
    return dict(applied=True, native=native, O=ov, S=sv, P=pv, C=shape, weighted_C=weighted,
        total=float(solution['objective']), coarse_mode=expected_mode, coarse_prior_count=1,
        containment_count_in_optimizer=count, old_full_outline_or_numeric_mask_tie_used=False)


def _energy_text(energy):
    if not energy['applied']:
        return '현재 N/O/S/P/C 에너지 비교 미적용'
    if energy['native'] is None:
        return 'fallback: 수용된 joint 비용 없음'
    return (f"N {energy['native']:.3f} · O {energy['O']:.3f} · S {energy['S']:.3f} · "
            f"P {energy['P']:.3f} · λC {energy['weighted_C']:.3f} = {energy['total']:.3f}")


def _containment_display_geometry(projected, observation):
    """Classification for drawing saved geometry only; no factor/fit/pixels/GT."""
    if projected is None or not observation.get('supported'):
        return dict(available=False, reason='saved_pose_or_global_observation_unavailable')
    projected = np.asarray(projected, float)
    points = np.asarray(observation['observed_boundary_samples_xy'], float)
    try:
        hull = projected[ConvexHull(projected).vertices]
    except (QhullError, ValueError):
        return dict(available=False, reason='degenerate_saved_projection')
    edge = np.roll(hull, -1, axis=0) - hull
    relative = points[:, None, :] - hull[None, :, :]
    outside = np.any(edge[None, :, 0] * relative[:, :, 1] - edge[None, :, 1] * relative[:, :, 0] < 0., axis=1)
    return dict(available=True, projected_full_hull_xy=hull.tolist(), observed_boundary_samples_xy=points.tolist(),
        outside_indices=np.flatnonzero(outside).tolist(), inside_or_boundary_indices=np.flatnonzero(~outside).tolist(),
        model_hull_canvas_clipped=False, model_to_observed_term=0., marker_source='stored_global_mask_samples_not_S_U_keypoints',
        new_RGB_queries=0, new_factor_or_pose_evaluations=0)


def _containment_panel(ax, display, key, arm, energy):
    if not display['available']:
        ax.text(.02, .97, NAMES[key] + '\n전역 관측/저장 모델 없음', transform=ax.transAxes, va='top')
        return
    hull = np.asarray(display['projected_full_hull_xy'])
    points = np.asarray(display['observed_boundary_samples_xy'])
    closed = np.vstack([hull, hull[0]])
    ax.plot(closed[:, 0], closed[:, 1], c=COLORS[key], lw=1.5)
    for indices, color in ((display['inside_or_boundary_indices'], '#2ca25f'), (display['outside_indices'], '#e97c20')):
        if indices:
            ax.scatter(points[indices, 0], points[indices, 1], s=3, c=color, alpha=.65)
    text = (NAMES[key] + ' / ' + str(arm['WD_family'])
        + f"\n내부/경계 {len(display['inside_or_boundary_indices'])} · 외부 {len(display['outside_indices'])}"
        + '\n모델→관측 비용0 · ' + _energy_text(energy))
    if not arm['accepted']:
        text += '\nfallback 기하 표시: joint 비용 미수용'
    ax.text(.02, .97, text, transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(facecolor='white', alpha=.9))


def _original_coarse_geometry(prior_inputs, K):
    tp, full = prior_inputs['TRANSLATION'], prior_inputs['FULL']
    for key in ('original_native_xyz', 'original_R_cf', 'original_t', 'original_first_R_physical',
                'original_first_center_camera_xyz', 'prior_covariance6', 'prior_bias6'):
        if not np.array_equal(np.asarray(tp[key]), np.asarray(full[key])):
            raise ValueError('Both modes must use the identical original first N3 prior target')
    if any(record['factor_weight'] != 1. or record['bias_strength'] != 1. for record in (tp, full)):
        raise ValueError('Fixed CAL lambda=beta=1 required')
    projection = C.project(np.asarray(tp['original_native_xyz']), np.asarray(tp['original_R_cf']), np.asarray(tp['original_t']), np.asarray(K))
    center = np.asarray(tp['original_first_center_camera_xyz'])
    homogeneous = np.asarray(K) @ center
    center_xy = homogeneous[:2] / homogeneous[2]
    return dict(original_first_model_projection_xy=projection.tolist(), original_first_model_center_projection_xy=center_xy.tolist(),
        TP_marginal_covariance3=np.asarray(tp['prior_covariance6'])[3:, 3:].tolist(),
        FULL_covariance6=full['prior_covariance6'], original_bias6=tp['prior_bias6'],
        e_numeric_pose_used=False, target_wire_is_original_before_bias=True,
        CAL_bias_is_subtracted_once_in_residual_not_a_new_observed_KP=True)


def _pose_panel(ax, q, arm, key, metric, energy, bounds):
    if q is None:
        ax.text(.02, .97, '저장 자세 없음', transform=ax.transAxes, va='top')
        return
    DRAW._wire(ax, q, COLORS[key], width=1.8, ls='-' if arm['accepted'] else '--')
    text = OLD._metric_text(metric) + ' · ' + ('joint 수용' if arm['accepted'] else 'fallback') + '\n' + _energy_text(energy)
    outside = np.any((q < [bounds[0], bounds[3]]) | (q > [bounds[1], bounds[2]]), axis=1)
    if outside.any():
        text += '\n확대창 밖 모델 꼭짓점 ' + str(int(outside.sum())) + '개 · 전체 좌표 JSON'
    if key in LOCKED_METHODS:
        a = arm['early_family_audit']
        text += '\n요청 ' + str(a['requested_predicted_family']) + ' / 실제 ' + str(arm['WD_family']) + ' · e 수치자세 미사용'
        if not a['locked_family_constraint_satisfied']:
            text += '\n요청 family 해 미수용'
    ax.text(.02, .97, text, transform=ax.transAxes, va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.9))


def _summary(reports):
    rows = []
    for part, label in (('PRIMARY192', '전체192'), ('TEST', '기존 TEST49'), ('REMAINDER', '추가143'), ('CAL', '별도 CAL46')):
        records = reports['TEST']['rows'] + reports['REMAINDER']['rows'] if part == 'PRIMARY192' else reports[part]['rows']
        for key in METHODS:
            summary = ACCOUNT._summary(records, key)
            if part != 'PRIMARY192':
                ACCOUNT._same_summary(summary, reports[part]['summary'][key])
            rows.append(f'<tr><td>{label}</td><td>{NAMES[key]}</td><td>{len(records)}</td><td>{summary["translation_le5"]}</td>'
                f'<td>{summary["successes_5cm_5deg"]}</td><td>{summary["translation_cm"]["mean"]:.3f}</td>'
                f'<td>{summary["translation_cm"]["max"]:.3f}</td><td>{summary["fallback_count"]}</td></tr>')
    return '<table><tr><th>집합</th><th>고정 방법</th><th>전체</th><th>T≤5cm</th><th>5cm·5°</th><th>평균 cm</th><th>최대 cm</th><th>fallback</th></tr>' + ''.join(rows) + '</table>'


def render(*, root_postscore_audit_memory_GO=False):
    if not root_postscore_audit_memory_GO:
        raise ValueError('Wait for explicit root score/audit/memory GO')
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed and partial reports')
    _load_plan()
    protocol, new, scores, reports, auditpaths = _completed()
    tasks = {r['id']:r for r in C.rows(EXP.OUTPUT / 'INPUTS.jsonl.gz') if r['id'] in CASES}
    observations = {r['id']:r for r in C.rows(OLD.OBS.OUTPUT / 'OBSERVATIONS.jsonl.gz') if r['id'] in CASES}
    if set(tasks) != set(CASES) or set(observations) != set(CASES):
        raise ValueError('Four unchanged actual inputs required')
    prior_models = dict(O=C.read(EXP.OLD.OCAL.OUTPUT / 'MODEL.json'), S=C.read(EXP.OLD.SCAL.OUTPUT / 'MODEL.json'))
    paths = [Path(__file__), PLAN / 'PREDECLARATION.json', OLD.OBS.OUTPUT / 'OBSERVATIONS.jsonl.gz',
             OLD.OBS.OUTPUT / 'OBSERVATION_SEAL.json'] + _dependency_paths() + auditpaths
    paths += [EXP.OUTPUT / name for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'INPUTS.jsonl.gz', 'PREDICTION_SEAL.json',
        'PREDICTIONS.jsonl.gz', 'SCORE_SEAL.json', 'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json')]
    bindings = dict(_binding(path) for path in paths)
    font = DRAW._font()
    if font.is_file():
        bindings[str(font)] = C.sha(font)
    OUTPUT.mkdir(parents=True)
    assets = OUTPUT / 'assets'
    assets.mkdir()
    cases, pictures = [], []
    for index, fid in enumerate(CASES):
        task, observed = tasks[fid], observations[fid]
        if (task['states'] != observed['states'] or task['spec']['visible'] != observed['visible_measurements']
                or task['new_chain_observation'] != observed['chain_observation'] or task['baseline'] != new[fid]['arms'][BASELINE]):
            raise ValueError('Exact unchanged observations and e required')
        for measurement in task['spec']['visible']:
            k = measurement['native_corner']
            if task['initial_self_hidden'][k] or task['states'][k] not in ('B', 'I'):
                raise ValueError('S/U gate must precede displayed measurement coordinates')
        original_domains = [r['original_domain'] for r in task['O_localization_abstention']['records']]
        if original_domains != [d for d in observed['domains'] if d['state'] == 'O']:
            raise ValueError('Original O prior provenance differs')
        image_path, mask_path = C.ROOT / task['image'], C.ROOT / observed['mask_binding']['path']
        if C.sha(image_path) != task['image_sha256'] or C.sha(mask_path) != observed['mask_binding']['sha256']:
            raise ValueError('Actual RGB/mask bytes differ')
        bindings[task['image']] = task['image_sha256']
        bindings[observed['mask_binding']['path']] = observed['mask_binding']['sha256']
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError('Actual image decode failed')
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(mask_path, allow_pickle=False) as arrays:
            mask = arrays['pallet_mask_clean'].copy()
        geometry, K = observed['initial_geometry'], observed['initial_geometry']['K']
        _, _, prepared_S = EXP.OLD._prior_inputs(task, prior_models)
        eligible_S = [r['native_corner'] for r in prepared_S['eligible_initial_S']]
        arms = {key:new[fid]['arms'][key] for key in METHODS}
        projected = {key:OLD._model(arm, K) for key, arm in arms.items()}
        metrics = {key:scores[fid]['metrics'][key] for key in METHODS}
        energies = {key:_energies(arms[key], key) for key in METHODS}
        plans = {key:PREVIOUS._saved_plan(arms[key]) for key in METHODS[1:]}
        active_bundle = dict(groups=plans[TP_C]['active_groups'])
        coarse = _original_coarse_geometry(new[fid]['original_first_pose_prior_inputs'], K)
        displays = {key:_containment_display_geometry(projected[key], task['outline']) for key in CONTAINMENT_METHODS}
        support = [np.asarray(geometry['initial_projection_xy']),
            np.asarray([m['xy'] for m in task['spec']['visible']]).reshape(-1, 2),
            np.asarray([d['center_xy'] for d in original_domains]).reshape(-1, 2)]
        xy = np.vstack(support)
        h, w = image.shape[:2]
        lo, hi = np.maximum(xy.min(0) - 35, 0), np.minimum(xy.max(0) + 35, [w, h])
        bounds = tuple(map(float, (lo[0], hi[0], hi[1], lo[1])))
        case = dict(id=fid, partition=task['partition'], image=task['image'], image_sha256=task['image_sha256'],
            states=task['states'], initial_geometry=geometry, frozen_BI=task['spec']['visible'], original_O_domains=original_domains,
            hard_O_T_domains=task['spec']['domains'], U_regions=task['u_regions'], partial_boundary=task['boundary_merged'],
            actual_NMS_observation=task['new_chain_observation'], mask_binding=observed['mask_binding'],
            eligible_initial_S_ids=eligible_S, original_first_pose_prior_inputs=new[fid]['original_first_pose_prior_inputs'],
            original_first_coarse_display=coarse, stored_global_containment_observation=task['outline'],
            S_U_point_markers=False, crop_bounds=bounds, crop_not_based_on_reference_or_final_pose=True,
            methods={key:dict(actual_pose=arm['actual_pose'], accepted=arm['accepted'], WD_family=arm.get('WD_family'),
                model_projection_xy=None if projected[key] is None else projected[key].tolist(), metric=metrics[key],
                energy_ledger=energies[key], early_family_audit=arm.get('early_family_audit'), fallback_audit=arm.get('fallback_audit'),
                runtime_complete_BI_replacement_plan=plans.get(key), containment_display=displays.get(key),
                plan_is_accepted_joint_solution=arm['accepted'] if key != BASELINE else None) for key, arm in arms.items()}, stages=[])
        for stage, title in enumerate(TITLES):
            if stage == 4:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                for ax, key in zip(axes, (TP_NATIVE, FULL_NATIVE)):
                    ax.imshow(image)
                    ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.set_aspect('equal'); ax.set_axis_off()
                    counts = PREVIOUS._replacement_panel(ax, task, plans[key], NAMES[key] + ' / ' + str(arms[key]['WD_family'])
                        + (' / 미수용 계획' if not arms[key]['accepted'] else ''))
                    case['methods'][key]['displayed_plan_counts'] = counts
            elif stage == 6:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                for ax, key in zip(axes.ravel(), CONTAINMENT_METHODS):
                    ax.imshow(image); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect('equal'); ax.set_axis_off()
                    _containment_panel(ax, displays[key], key, arms[key], energies[key])
            else:
                fig, ax = DRAW._base(image, title)
                ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
                if stage == 0:
                    DRAW._wire(ax, np.asarray(geometry['initial_projection_xy']), '#7e8790', ls='--')
                    OLD._frozen_points(ax, task)
                elif stage == 1:
                    ax.imshow(np.dstack([np.zeros_like(mask), mask, np.zeros_like(mask), mask * .15]))
                    case['NMS_display_audit'] = OLD._nms(ax, task['new_chain_observation'], active_bundle)
                elif stage == 2:
                    if original_domains:
                        CONTEXT._new_canvas(ax, image, task['spec']['domains'], bounds)
                    CONTEXT._context(ax, task)
                elif stage == 3:
                    CONTEXT._old_o(ax, original_domains)
                    ax.text(.02, .97, '초기 S 모델 prior 대상: ' + str(eligible_S) + '\nS/U 점 표시·S 영상 관측 없음',
                        transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.9))
                elif stage == 5:
                    q0 = np.asarray(coarse['original_first_model_projection_xy'])
                    center_xy = np.asarray(coarse['original_first_model_center_projection_xy'])
                    DRAW._wire(ax, q0, '#576779', ls='--')
                    ax.scatter([center_xy[0]], [center_xy[1]], marker='o', facecolors='none', edgecolors='#576779', s=70)
                    DRAW._label(ax, center_xy, '첫 N3 모델 중심 · KP 아님', '#576779', (7, 9))
                    ax.text(.02, .97, '원래 첫 N3 physical pose · e 수치자세 아님\nTP: 이동3개 + marginal Σtt\nFULL: 전체6개 + Σ6\nCAL μ whitening 전1회 · λ=β=1',
                        transform=ax.transAxes, va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.9))
                else:
                    key = METHODS[stage - 7]
                    _pose_panel(ax, projected[key], arms[key], key, metrics[key], energies[key], bounds)
            path = assets / f'case{index}_stage{stage + 1}.png'
            DRAW._save(fig, path, CAPTIONS[stage])
            case['stages'].append(dict(title=title, caption=CAPTIONS[stage], asset=str(path.relative_to(OUTPUT))))
        cases.append(case)
        pictures.append((image, bounds, projected, metrics, energies))
    fig, axes = plt.subplots(len(METHODS), len(CASES), figsize=(18, 24), squeeze=False)
    for col, (image, bounds, projected, metrics, energies) in enumerate(pictures):
        for row, key in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(image)
            if projected[key] is not None:
                DRAW._wire(ax, projected[key], COLORS[key], width=1.4)
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.set_aspect('equal'); ax.set_axis_off()
            ax.set_title((CASES[col] + '\n' if row == 0 else '') + NAMES[key] + '\n' + OLD._metric_text(metrics[key])
                         + '\n' + _energy_text(energies[key]), fontsize=7)
    fig.suptitle('실제 사진 · 같은 네 사례 / 일곱 고정 방법\n원래 첫 pose prior·soft containment·family 고정 구분 · GT/S/U KP 표시 없음', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUTPUT / 'overview_actual.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    C.write(OUTPUT / 'PROCESS_DATA.json', dict(schema='COARSE_PRIOR_CONTAINMENT_ACTUAL_PROCESS_V25',
        predeclared_cases=list(CASES), uniform_methods=list(METHODS), cases=cases, source_bindings=bindings,
        S_U_input_or_projection_point_markers=False, global_sample_markers_not_KP_observations=True,
        no_GT_wire_or_reference_pose_corner_reads=True, new_inference_or_fit=0,
        old_full_outline_cost_or_numeric_mask_tie_used=False, coarse_priors_not_independent_image_evidence=True,
        first_prior_wire_not_e_numeric_pose=True, default_e_unchanged=True))
    sections = []
    for case in cases:
        rows = ''.join('<tr><td>' + NAMES[key] + '</td><td>' + OLD._metric_text(case['methods'][key]['metric'])
            + '</td><td>' + html.escape(str(case['methods'][key]['WD_family'])) + '</td><td>'
            + ('joint 수용' if case['methods'][key]['accepted'] else 'fallback') + '</td><td>'
            + _energy_text(case['methods'][key]['energy_ledger']) + '</td></tr>' for key in METHODS)
        stages = ''.join('<h3>' + s['title'] + '</h3><img src="' + s['asset']
            + '" alt="실제 저장 처리 단계"><p>' + s['caption'] + '</p>' for s in case['stages'])
        sections.append('<details open><summary>' + html.escape(case['id']) + '</summary><p>고정 상태: '
            + ' '.join(case['states']) + '</p><div class="scroll"><table><tr><th>고정 방법</th><th>명목 참조 오차</th>'
            + '<th>W/D</th><th>수용</th><th>분리 비용</th></tr>' + rows + '</table></div>'
            + '<p>각 SOURCE 비용 계획은 해당 저장 family의 기록입니다. fallback 계획은 joint 해가 아닙니다. 모든 투영·범위·표본·실패 기록은 JSON에 보존합니다.</p>' + stages + '</details>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAL 첫 pose prior와 soft containment</title><style>body{background:#edf2f6;color:#173044;font-family:system-ui,sans-serif;margin:0}main{max-width:1160px;margin:auto;padding:18px}img{max-width:100%;height:auto;display:block}details{background:white;margin:18px 0;padding:16px;border-radius:10px}summary{font-weight:bold}p{line-height:1.6}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #cbd6df;text-align:left}@media(max-width:500px){main{padding:8px}details{padding:9px}}</style><main><h1>원래 첫 CAL pose prior · soft containment: 실제 저장 과정</h1><p>기존 네 사례를 새25 점수 전에 그대로 고정했습니다. 모든 사진에서 e, 이동 prior native/C/C+family고정, 6D prior native/C/C+family고정의 일곱 방법을 같은 순서로 비교합니다. 사례별 최적 방법을 섞지 않습니다.</p><p>기존 SOURCE 정책·CAL 접선·원 q/Σ/신뢰도·hard guard·bounded O/S는 유지합니다. 접선 분산은 1+γa이고 기존 부분관측 T4는 그대로입니다. 새 Gaussian prior는 원래 첫 N3 physical R와 모델 중심을 고정하며 e의 숫자 자세가 아닙니다. 이동 prior는 원 unwhitened 이동 잔차와 marginal Σtt만 사용하고 6D prior는 전체 Σ6을 사용합니다. CAL μ는 whitening 전에 한 번만 빼며 λ=β=1입니다. backbone과 상관된 regularizer이고 독립 영상 정보·검증된 posterior는 아닙니다.</p><p>새 C는 관측 foreground가 full projected convex cuboid의 subset이라는 조건부 soft cue입니다. model hull 내부 표본은0, 외부 표본만 .5 평균 robust 비용이며 모델→관측 비용은0입니다. 관측은 저장된 clip된 boundary sample이고 모델 hull은 canvas로 clip하지 않습니다. falsepositive SAM과 실제 모형 오차는 남고 모델 확대·깊이 모호성·유일성은 C만으로 해결되지 않습니다. 기존 대칭 whole-mask 비용이나 native numeric mask tie-break는 사용하지 않습니다.</p><p>선택과 최적화의 총비용은 N+O+S+P+λC이며 각 항은 한 번만 계산합니다. native 방법은 C=0입니다. e에는 현재 비용을 재사용하지 않습니다. family 고정은 e의 예측 이름만 모든 후보 인증 전에 제한하고 e의 수치 pose를 복사하지 않습니다. 요청 family에 해가 없으면 own coarse fallback/실패를 남깁니다. S/U KP 점·GT wire는 없으며 C 패널의 표본은 전역 마스크 경계입니다.</p><p>전체238와 실패·fallback·native 플래그를 유지합니다. primary192와 CAL46을 분리하고, 추가143 참조는 annotation/PnP 유도값으로 독립 계측이 아닙니다. 반복 사용한 자료의 탐색 결과이며 기존 e 기본값은 유지합니다.</p><div class="scroll">''' + _summary(reports) + '''</div><img src="overview_actual.png" alt="같은 네 사례 일곱 고정 방법"><p><a href="PROCESS_DATA.json">전체 표시값과 출처</a> · <a href="''' + _url(PREVIOUS.OUTPUT / 'index.html') + '''">이전24 실제 과정</a></p>''' + ''.join(sections) + '</main></html>'
    (OUTPUT / 'index.html').write_text(page, encoding='utf-8')
    for relative, digest in bindings.items():
        path = Path(relative) if Path(relative).is_absolute() else C.ROOT / relative
        if C.sha(path) != digest:
            raise ValueError('Bound source/evidence changed while rendering')
    C.write(OUTPUT / 'RENDER_SEAL.json', dict(source_sha256=C.sha(__file__), input_sha256=bindings,
        assets={str(path.relative_to(OUTPUT)):C.sha(path) for path in OUTPUT.rglob('*') if path.is_file()},
        cases=4, stages_per_case=len(TITLES), predeclared_case_order=list(CASES), uniform_methods=list(METHODS),
        actual_RGB_only=True, no_GT_wire_or_S_U_KP_point_markers=True, global_sample_markers_not_KP_queries=True,
        first_prior_wire_not_e_numeric_pose=True, new_inference_or_fit=0))
    print(dict(output=str(OUTPUT), cases=4, methods=7, stages=len(TITLES)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'render'))
    parser.add_argument('--after-root-score-audit-memory-go', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_plan()
    else:
        render(root_postscore_audit_memory_GO=args.after_root_score_audit_memory_go)
