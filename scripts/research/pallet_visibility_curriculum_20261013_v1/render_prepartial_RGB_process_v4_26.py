"""Predeclared seven-method actual process; render only after explicit root GO.

Saved poses, global samples, RGB, masks, state/observation records and sealed
scores only. No inference, observation search, pose fit or reference geometry; old q restoration is not a new measurement.
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
from . import evaluate_prepartial_RGB_preservation_v4_26 as EXP
from . import render_coarse_prior_containment_process_v4_25 as V25
from . import render_source_preserving_tangent_mask_process_v4_24 as PREVIOUS

OLD, DRAW, CONTEXT, ACCOUNT = PREVIOUS.OLD, PREVIOUS.DRAW, PREVIOUS.CONTEXT, PREVIOUS.ACCOUNT
OUTPUT = EXP.OUTPUT.parent / 'prepartial_RGB_process_v4_26'
PLAN = EXP.OUTPUT.parent / 'prepartial_RGB_process_plan_v4_26'
CASE_SELECTION = PLAN / 'CASE_SELECTION_V1.json'
AUDIT_SUBDIR = 'independent_root_audit_v1'
CASES = ('eval_noapril:1775201414728017664', 'plastic_day_01:005516', 'wood_night_01:031426')
METHODS = tuple(EXP.ARMS)
BASELINE, ZERO_NATIVE, ZERO_LOCK, TP_NATIVE, TP_NATIVE_LOCK, TP_C, TP_LOCK = METHODS
CONTAINMENT_METHODS = (TP_C, TP_LOCK)
LOCKED_METHODS = (ZERO_LOCK, TP_NATIVE_LOCK, TP_LOCK)
NAMES = {BASELINE:'기본 e', ZERO_NATIVE:'RGB q 복구 · coarse prior 0', ZERO_LOCK:'같은 prior0 · e family 고정',
    TP_NATIVE:'RGB q 복구 · 이동 prior', TP_NATIVE_LOCK:'같은 이동 prior · e family 고정',
    TP_C:'RGB q 복구 · 이동 prior+C', TP_LOCK:'같은 이동+C · e family 고정'}
COLORS = dict(zip(METHODS, ('#2879c3','#b48a16','#e37136','#169d7a','#ae64bc','#5d52b9','#d84b7d')))
TITLES = ('① 초기 모델·복구 전 고정 B/I','② 실제 마스크·보관 NMS 대안','③ 원래 partial 관측과 유지되는 guard',
    '④ 원 q 복구·중복 비용 owner 제거','⑤ O/U 범위·bounded O/S·S 모델', '⑥ 첫 N3 target·prior0과 이동 prior',
    '⑦ 저장 모델의 soft containment','⑧ 기본 e 저장 자세','⑨ RGB q 복구·prior0 저장 자세',
    '⑩ 같은 prior0·e family 고정','⑪ RGB q 복구·이동 prior 저장 자세','⑫ 같은 이동 prior·e family 고정',
    '⑬ RGB q 복구·이동+C 저장 자세','⑭ 같은 이동+C·e family 고정')
CAPTIONS = (
    '회색은 첫 N3 모델 wire. 분홍 ×는 현재 complete B/I, 회색 +는 이전 partial B/I의 보관 q. S/U 입력 점은 조회하거나 표시하지 않음.',
    '실제 저장 마스크와 NMS 대안. 청록은 prior0 선택 family 활성 변에 속한 보관 대안이며 모든 방법 공통 활성 계획이 아님. 동일 지원점 fit 대안은 독립 관측이 아님.',
    '복구 전 partial의 법선·접선 구간과 O/U 범위를 표시. 이 원래 hard guard는 복구 후에도 동일하며, 그림은 optimizer 중간 자세가 아님.',
    'RGB/unknown prepartial은 정확히 원 q/관측Σ/유효Σ/신뢰도의 complete 2D 비용으로 복구. 자기 oldnormal·T4·edge owner는 비용에서만 제거. 다른 유효 fallback owner가 있으면 공유 변은 한 번 남음. 새 KP 추론이 아님.',
    'O hard domain은 닫힌 캔버스, 원 O 타원은 bounded 위치 가정이며 실제 occluder가 아님. U는 범위만, S는 모델 wire·coarse 가시성 prior만 쓰며 영상/2D 관측 없음.',
    '첫 N3 physical R와 모델 중심은 동일. noP는 FULL schema지만 weight0이라 비용·gradient0; 이동 prior는 weight1·marginal Σtt. 기존 CAL μ를 whitening 전에 한 번 빼며 새 coarse prior fit은 없음.',
    '고정된 global 마스크 표본과 full 미클립 모델 hull. 내부/경계0·외부만 soft 비용, 모델→관측0. S/U KP 관측이 아니며 모델 확대·깊이·SAM 오염을 해결한다고 주장하지 않음.',
    '기존 e 최종 저장 자세. 현재 N/O/S/P/C 비용을 재평가하거나 재사용하지 않음. GT wire가 아님.',
    '복구된 원 RGB/unknown q와 남은 fallback partial·새 CAL edge noise. bounded O/S는 유지하지만 추가 coarse pose prior는 정확히0. 이전 whole-mask 비용·tie는 없음.',
    '같은 prior0 후보에서 e 예측 family 이름만 AUTH 전에 제한. e의 수치 pose를 target·fallback으로 사용하지 않음.',
    '동일한 q 복구 관측에 기존 CAL 이동 prior만 추가. 옛 Σ/μ·λ1/β1을 재사용하며 새 물리 포함률이나 독립 정보 주장은 없음.',
    '같은 이동-prior 후보에서 e 예측 family를 고정. 해가 없으면 own ORIGINAL coarse rejected fallback과 실패 표기를 유지.',
    '동일 이동-prior의 native endpoint를 남기고 soft C를 최적화·선택에 한 번 더함. 낮은 목적함수가 정확도 개선을 보장하지 않음.',
    '같은 이동+C 후보의 e family 고정 최종 결과. 모든 사례에 동일한 7방법을 적용하며 사례별 최적 방법을 섞지 않음.')


def _binding(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)), C.sha(path)


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(), OUTPUT), safe='/')


def _dependency_paths():
    modules = (EXP.RESTORE_CAL, EXP.V.TCAL, EXP.OLD.OCAL, EXP.OLD.SCAL)
    paths = [Path(EXP.__file__),Path(EXP.RESTORE.__file__),Path(EXP.P.__file__),Path(EXP.P.FACTOR.__file__),
        Path(EXP.PRIOR.__file__),Path(V25.__file__),Path(PREVIOUS.__file__),Path(OLD.__file__),
        Path(DRAW.__file__),Path(CONTEXT.__file__),Path(ACCOUNT.__file__),EXP.PLAN,CASE_SELECTION,
        EXP.POSE_MODEL.SELECTION,EXP.POSE_MODEL.SELECTION.with_name('SELECTION_SEAL.json'),
        EXP.POSE_MODEL.E.FIX/'calibration/final_adapter_CAL46/MODEL.json']
    paths += [m.OUTPUT/name for m in modules for name in ('MODEL.json','MODEL_SEAL.json')]
    return paths


def prepare_plan():
    target=PLAN/'PREDECLARATION.json'
    if target.exists():raise FileExistsError('Preserve every report predeclaration')
    if (EXP.OUTPUT/'SCORE_SEAL.json').exists():raise ValueError('Declare before new26 metrics')
    selected=C.read(CASE_SELECTION)
    if tuple(selected['case_ids'])!=CASES or tuple(selected['methods'])!=METHODS:
        raise ValueError('Root-fixed metadata-only cases and seven uniform methods required')
    source=Path(__file__).resolve();PLAN.mkdir(parents=True,exist_ok=True)
    (PLAN/source.name).write_bytes(source.read_bytes())
    C.write(target,dict(schema='PREPARTIAL_RGB_ACTUAL_PROCESS_PLAN_V26',case_ids=list(CASES),
        case_selection_binding=dict(path=str(CASE_SELECTION.relative_to(C.ROOT)),sha256=C.sha(CASE_SELECTION)),
        uniform_methods=list(METHODS),declared_all7_arms=list(EXP.ARMS),stages_per_case=len(TITLES),
        renderer_source_sha256=C.sha(source),declared_before_new26_metric_read=True,
        requires_all238_all7_prediction_score_and_zero_failure_independent_audit=True,
        requires_explicit_root_postscore_audit_memory_GO=True,case_best_method_or_reference_routing=False,
        restored_RGB_q_is_original_not_new_inference=True,restored_own_normal_tangent_owner_costs_removed=True,
        original_hard_guards_unchanged=True,other_surviving_shared_owner_once=True,new_CAL_noise_population=True,
        noP_FULL_schema_weight0_TP_weight1=True,first_N3_physical_target_identical=True,
        S_U_point_markers=False,U_regions_only=True,no_GT_wire_or_reference_pose_corner_reads=True,
        no_new_image_model_or_pose_fit=True,default_e_unchanged=True,
        dependencies=dict(_binding(path)for path in _dependency_paths())))
    print(dict(predeclared_cases=list(CASES),methods=list(METHODS),stages=len(TITLES),new26_metrics_read=False),flush=True)


def _load_plan():
    plan = C.read(PLAN / 'PREDECLARATION.json')
    if (tuple(plan['case_ids']) != CASES or tuple(plan['uniform_methods']) != METHODS
            or plan['declared_all7_arms'] != list(EXP.ARMS) or not plan['declared_before_new26_metric_read']
            or C.sha(__file__) != plan['renderer_source_sha256']):
        raise ValueError('Exact predeclared cases/methods/source required')
    for path, digest in plan['dependencies'].items():
        if C.sha(C.ROOT / path) != digest:
            raise ValueError('Predeclared source/model dependency changed')
    return plan


def _completed():
    folder = EXP.OUTPUT
    protocol, pred, score = (C.read(folder / name) for name in ('PROTOCOL.json', 'PREDICTION_SEAL.json', 'SCORE_SEAL.json'))
    if (pred['rows'] != 238 or not pred.get('all7_arms_complete') or not pred.get('all_declared_arms_complete')
            or pred['arms'] != list(EXP.ARMS) or pred['arm_count'] != 7
            or C.sha(folder / 'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(folder / 'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(folder / 'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(folder / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']):
        raise ValueError('Complete7arm/238 prediction+score required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        EXP.E._verify_sources(protocol[key])
    auditpath = folder / AUDIT_SUBDIR / 'AUDIT.json'
    sealpath = auditpath.with_name('AUDIT_SEAL.json')
    audit, aseal = C.read(auditpath), C.read(sealpath)
    if (audit['rows'] != 238 or audit['failure_count'] or audit['failures']
            or C.sha(auditpath) != aseal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(folder / 'PREDICTION_SEAL.json')
            or not audit.get('passed',audit.get('failure_count')==0)):
        raise ValueError('Independent current zero-failure full238 audit required')
    selected, ordered = {}, []
    for row in C.rows(folder / 'PREDICTIONS.jsonl.gz'):
        if set(row['arms']) != set(EXP.ARMS):
            raise ValueError('Every row must contain all seven arms')
        ordered.append(row['id'])
        if row['id'] in CASES:
            selected[row['id']] = row
    if ordered != protocol['ids'] or len(set(ordered)) != 238 or set(selected) != set(CASES):
        raise ValueError('All238 and three fixed cases required')
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


def _energies(arm,key):
    if key==BASELINE:return dict(applied=False,native=None,O=None,S=None,P=None,C=None,weighted_C=None,total=None)
    if not arm['accepted']:return dict(applied=True,native=None,O=None,S=None,P=None,C=None,weighted_C=None,total=None,failed_joint=True)
    sol=arm['solution'];costs=sol['regularizer_costs']
    if [v['kind']for v in costs]!=list(EXP.P.REGULARIZER_KINDS):raise ValueError('Separate native/O/S/coarse ledger required')
    native=float(sol['native_observation_objective']);ov,sv,pv=[float(v['weighted_cost'])for v in costs]
    zero=key in (ZERO_NATIVE,ZERO_LOCK);audit=sol['CAL_coarse_pose_prior_audit']
    if (audit['mode']!=('FULL'if zero else'TRANSLATION') or audit['factor_weight']!=(0. if zero else 1.)
            or not audit['mean_bias_subtracted_once'] or audit['translation_covariance_is_marginal_block']!= (not zero)):
        raise ValueError('Exact zero-schema versus active marginal TP required')
    if zero and pv!=0.:raise ValueError('noP factor must be exactly zero')
    shape=float(sol.get('global_containment_cost',0.));weighted=float(sol.get('weighted_containment_objective',0.))
    if key not in CONTAINMENT_METHODS and (shape!=0. or weighted!=0.):raise ValueError('No shape in native-only profile')
    if key in CONTAINMENT_METHODS and not np.isclose(weighted,shape*sol['global_containment_weight'],atol=1e-9,rtol=1e-10):raise ValueError('C weighted once')
    chosen=next(c for c in arm['candidate_selection']['candidates']if c['name']==arm['candidate_selection']['selected_name'])
    if (not np.isclose(native+ov+sv+pv+weighted,sol['objective'],atol=1e-9,rtol=1e-10)
            or not np.isclose(chosen['total_energy'],sol['objective'],atol=1e-9,rtol=1e-10)
            or arm['candidate_selection']['old_full_outline_factor_or_tie_break_used']is not False):raise ValueError('Same optimizer/selector sum required')
    return dict(applied=True,native=native,O=ov,S=sv,P=pv,C=shape,weighted_C=weighted,total=float(sol['objective']),
        coarse_mode=audit['mode'],coarse_prior_weight=audit['factor_weight'],coarse_prior_count=sol['coarse_pose_prior_count'],
        containment_count_in_optimizer=sol.get('containment_count_in_optimizer',0),old_full_outline_or_numeric_mask_tie_used=False)


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


def _original_coarse_geometry(prior_inputs,K):
    tp,zero=prior_inputs['TRANSLATION'],prior_inputs['FULL']
    for key in ('original_native_xyz','original_R_cf','original_t','original_first_R_physical',
            'original_first_center_camera_xyz','prior_covariance6','prior_bias6'):
        if not np.array_equal(np.asarray(tp[key]),np.asarray(zero[key])):raise ValueError('Same original physical target for both modes')
    if zero['factor_weight']!=0. or tp['factor_weight']!=1. or tp['bias_strength']!=1. or zero['bias_strength']!=1.:
        raise ValueError('FULL placeholder weight0 and TP weight1, beta1 required')
    q=C.project(np.asarray(tp['original_native_xyz']),np.asarray(tp['original_R_cf']),np.asarray(tp['original_t']),np.asarray(K))
    center=np.asarray(tp['original_first_center_camera_xyz']);h=np.asarray(K)@center
    return dict(original_first_model_projection_xy=q.tolist(),original_first_model_center_projection_xy=(h[:2]/h[2]).tolist(),
        TP_marginal_covariance3=np.asarray(tp['prior_covariance6'])[3:,3:].tolist(),original_bias6=tp['prior_bias6'],
        noP_schema='FULL',noP_weight=0.,TP_weight=1.,same_physical_R_and_center=True,e_numeric_pose_used=False,
        CAL_bias_in_residual_once_not_new_KP=True)


def _restoration_panel(ax,task,plan,title):
    for k in plan['restored_prepartial_complete_ids']:
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I'):
            raise ValueError('Restored BI state gate precedes archived point display')
    counts=PREVIOUS._replacement_panel(ax,task,plan,title)
    restored=plan['restored_prepartial_complete_ids'];partial={b['native_corner']for b in task['boundary_merged']}
    for k in restored:
        a=plan['restored_prepartial_source_archives'][str(k)];xy=np.asarray(a['original_observed_xy'])
        ax.scatter([xy[0]],[xy[1]],marker='x',s=55,c='#bd33a5')
        DRAW._label(ax,xy,'복구 old q'+str(k),'#982487',(5,-10))
    counts.update(restored_prepartial=len(restored),remaining_fallback_partial=len(partial-set(restored)),
        removed_owner_count=len(plan['prepartial_owner_removals']),
        removed_ownerless_edges=len(plan['prepartial_ownerless_withheld_physical_edges']),
        surviving_physical_edges=len(plan['active_groups']))
    ax.text(.02,.02,'복구 '+str(restored)+' / 남은 partial '+str(sorted(partial-set(restored)))
        +'\n자기 normal/T/owner 비용 제거 · 원 guard 유지\n다른 owner의 공유 변은 한 번 유지',
        transform=ax.transAxes,va='bottom',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
    return counts


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
        raise ValueError('Three unchanged actual inputs required')
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
        active_bundle = dict(groups=plans[ZERO_NATIVE]['active_groups'])
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
        case['prepartial_restoration_plans']={k:plans[k]for k in METHODS[1:]}
        for stage,title in enumerate(TITLES):
            if stage==6:
                fig,axes=plt.subplots(1,2,figsize=(15,6))
                for ax,key in zip(axes,CONTAINMENT_METHODS):
                    ax.imshow(image);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_axis_off()
                    _containment_panel(ax,displays[key],key,arms[key],energies[key])
            else:
                fig,ax=DRAW._base(image,title);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
                if stage==0:
                    DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#7e8790',ls='--');OLD._frozen_points(ax,task)
                elif stage==1:
                    ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.15]))
                    case['NMS_display_audit']=OLD._nms(ax,task['new_chain_observation'],active_bundle)
                elif stage==2:
                    CONTEXT._context(ax,task)
                    ax.text(.02,.97,'복구 전 partial / 원래 guard는 복구 후에도 유지',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
                elif stage==3:
                    case['methods'][ZERO_NATIVE]['displayed_plan_counts']=_restoration_panel(ax,task,plans[ZERO_NATIVE],NAMES[ZERO_NATIVE]+' / '+str(arms[ZERO_NATIVE]['WD_family']))
                elif stage==4:
                    if original_domains:CONTEXT._new_canvas(ax,image,task['spec']['domains'],bounds)
                    CONTEXT._context(ax,task);CONTEXT._old_o(ax,original_domains)
                    ax.text(.02,.97,'S 모델 prior IDs '+str(eligible_S)+' / S·U 점 없음',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
                elif stage==5:
                    q0=np.asarray(coarse['original_first_model_projection_xy']);center=np.asarray(coarse['original_first_model_center_projection_xy'])
                    DRAW._wire(ax,q0,'#576779',ls='--');ax.scatter([center[0]],[center[1]],marker='o',facecolors='none',edgecolors='#576779',s=65)
                    DRAW._label(ax,center,'첫 모델 중심 · KP 아님','#576779',(5,8))
                    ax.text(.02,.97,'동일 첫 N3 target · e 수치pose 아님\nnoP: FULL schema weight0 / TP: marginal Σtt weight1\nCAL μ1회 · bounded O/S는 두 모드에 유지',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
                else:
                    key=METHODS[stage-7];_pose_panel(ax,projected[key],arms[key],key,metrics[key],energies[key],bounds)
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
    fig.suptitle('실제 사진 · 같은 세 사례 / 일곱 고정 방법\n원 RGB q 복구·중복 비용 제거·prior0/TP/C 구분 · GT/S/U KP 표시 없음', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUTPUT / 'overview_actual.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    C.write(OUTPUT / 'PROCESS_DATA.json', dict(schema='PREPARTIAL_RGB_RESTORATION_ACTUAL_PROCESS_V26',
        predeclared_cases=list(CASES), uniform_methods=list(METHODS), cases=cases, source_bindings=bindings,
        S_U_input_or_projection_point_markers=False, global_sample_markers_not_KP_observations=True,
        no_GT_wire_or_reference_pose_corner_reads=True, new_inference_or_fit=0,
        old_full_outline_cost_or_numeric_mask_tie_used=False, coarse_priors_not_independent_image_evidence=True,restored_original_q_not_new_inferred_point=True,
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
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>원 RGB q 복구 · v26 실제 과정</title><style>body{background:#edf2f6;color:#173044;font-family:system-ui,sans-serif;margin:0}main{max-width:1160px;margin:auto;padding:18px}img{max-width:100%;height:auto;display:block}details{background:white;margin:18px 0;padding:16px;border-radius:10px}summary{font-weight:bold}p{line-height:1.65}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #cbd6df;text-align:left}</style><main><h1>기존 RGB q 복구와 비용 owner 정리: 실제 저장 과정</h1><p>점수가 아닌 metadata로 먼저 고정한 세 사례이며 모든 사진에서 같은 7방법을 같은 순서로 비교합니다. 첫 영향 TEST·첫 영향 추가143·이전 사용자 요청031426입니다. 사례별 최적 방법을 섞지 않습니다.</p><p>RGB/unknown prepartial의 원 q·관측Σ·유효Σ·신뢰도를 complete 2D 비용으로 복구하고 자기 oldnormal·old tangent·edge owner는 비용에서만 제거합니다. 다른 유효 fallback owner가 있으면 공유 변은 한 번 남으며 복구 q와의 상관은 남을 수 있습니다. 정확히 VISIBLE_BACKBONE_FALLBACK만 기존 partial/revocation을 유지합니다. 원 hard guard와 v19 시작점 정책은 바꾸지 않았습니다. 새 active edge population의 CAL 모델은 별도 적합됐고 이는 새 q를 추론한 것이 아닙니다.</p><p>noP는 FULL schema weight0이므로 추가 coarse pose 비용·gradient가0이고 bounded O/S는 계속 활성입니다. 이동 prior는 같은 첫 N3 physical R·중심을 target으로 기존 marginal Σtt·μ·λ1/β1을 재사용합니다. 새 coarse prior 적합은 없습니다. C는 full 미클립 hull 내부0·외부만 soft 비용이며 옛 대칭 whole-mask나 native mask tie는 없습니다. e family 고정은 예측 이름만 AUTH 전에 제한하고 e 수치pose를 복사하지 않습니다. 실패면 own ORIGINAL coarse rejected fallback입니다.</p><p>S는 모델 wire·coarse 가시성 prior만, U는 범위만 사용합니다. S/U 점·GT wire·새 RGB/NN·새 pose fit은 이 보고서에 없습니다. 원 RGB 밝기는 유지합니다. 표본·모델 투영·원 q는 서로 다른 정보입니다. e에는 현재 비용을 재사용하지 않습니다. 추가143 참조는 annotation/PnP 유도값이며 독립 계측이 아닙니다. 반복 사용한 자료의 탐색 결과이고 e 기본값과 미완료 목표를 유지합니다.</p><div class="scroll">'+_summary(reports)+'</div><img src="overview_actual.png" alt="실제 세 사례 일곱 고정 방법"><p><a href="PROCESS_DATA.json">표시값·원장·출처</a> · <a href="'+_url(V25.OUTPUT/'index.html')+'">이전 v25 과정</a></p>'+''.join(sections)+'</main></html>'
    (OUTPUT / 'index.html').write_text(page, encoding='utf-8')
    for relative, digest in bindings.items():
        path = Path(relative) if Path(relative).is_absolute() else C.ROOT / relative
        if C.sha(path) != digest:
            raise ValueError('Bound source/evidence changed while rendering')
    C.write(OUTPUT / 'RENDER_SEAL.json', dict(source_sha256=C.sha(__file__), input_sha256=bindings,
        assets={str(path.relative_to(OUTPUT)):C.sha(path) for path in OUTPUT.rglob('*') if path.is_file()},
        cases=len(CASES), stages_per_case=len(TITLES), predeclared_case_order=list(CASES), uniform_methods=list(METHODS),
        actual_RGB_only=True, no_GT_wire_or_S_U_KP_point_markers=True, global_sample_markers_not_KP_queries=True,
        first_prior_wire_not_e_numeric_pose=True, new_inference_or_fit=0))
    print(dict(output=str(OUTPUT), cases=len(CASES), methods=7, stages=len(TITLES)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'render'))
    parser.add_argument('--after-root-score-audit-memory-go', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_plan()
    else:
        render(root_postscore_audit_memory_GO=args.after_root_score_audit_memory_go)
