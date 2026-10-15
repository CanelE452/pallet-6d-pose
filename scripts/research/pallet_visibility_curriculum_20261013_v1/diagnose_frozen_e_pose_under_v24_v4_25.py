"""TEST49 postseal diagnosis: frozen e pose under the current v24 objective.

Diagnostic poses are never inserted into a proposal bank or used as runtime
fallbacks. No reference coordinates/poses, images, raw S/U coordinates, fitting
or optimization are read/run. Prepare freezes the complete49 recipe and bytes
before reading TEST journals or already-sealed posthoc scalar metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import islice, zip_longest
from pathlib import Path

import numpy as np

from . import audit_source_preserving_tangent_mask_v4_24 as A

V, C, E = A.V, A.V.C, A.V.E
OUTPUT = V.OUTPUT.parent / 'frozen_e_current_objective_diagnosis_v4_25'
ARM = V.ARMS[6]
FOCUS_SUFFIXES = ('1430551447040', '2140531310080', '2142480077056')


def _proper(matrix):
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-7, rtol=0)):
        raise ValueError('Finite proper physical/registered rotation required')
    return matrix


def transport_physical_pose_to_family(pose, family):
    """R_cf_trial = R_physical_saved Q_trial.T; unchanged physical centroid."""
    physical = _proper(pose['R_physical'])
    Q = _proper(family['physical_registration_Q'])
    R = _proper(physical @ Q.T)
    t = np.asarray(pose['centroid'], float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Finite saved physical centroid required')
    return R, t, dict(physical_rotation_max_absolute_difference=float(np.max(np.abs(R @ Q - physical))),
        transport='R_cf_trial=R_physical_saved@Q_trial.T; same centered physical centroid',
        original_e_pose_not_a_new_proposal=True)


def _physical_body_gap(first, second):
    # Native corner order may differ; this is the registered unordered body.
    a = np.asarray(first['native_xyz']) @ np.asarray(first['physical_registration_Q'])
    b = np.asarray(second['native_xyz']) @ np.asarray(second['physical_registration_Q'])
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def classify_comparison(e_feasible, selected_accepted, e_total, selected_total, tolerance=1e-8):
    if not selected_accepted:
        return 'selected_is_own_fallback_not_a_solved_pose'
    if not e_feasible:
        return 'diagnostic_e_excluded_by_current_hard_or_numeric_guard'
    if e_total < selected_total - tolerance:
        return 'feasible_diagnostic_e_has_lower_energy_bounded_search_coverage_gap'
    if e_total > selected_total + tolerance:
        return 'current_objective_prefers_selected_over_diagnostic_e'
    return 'same_current_energy_within_reporting_tolerance'


def _costs(records):
    kinds, factors = defaultdict(float), defaultdict(float)
    for record in records:
        value = float(record['weighted_robust_cost'])
        kinds[record['kind']] += value
        key = (record['kind'], record.get('native_corner'), record.get('native_neighbor'))
        factors[key] += value
    return dict(kinds), dict(factors)


def _verify_parent():
    protocol = C.read(V.OUTPUT / 'PROTOCOL.json')
    pred = C.read(V.OUTPUT / 'PREDICTION_SEAL.json')
    inputs = C.read(V.OUTPUT / 'INPUT_SEAL.json')
    score = C.read(V.OUTPUT / 'SCORE_SEAL.json')
    auditpath = V.OUTPUT / 'independent_root_audit_v1/AUDIT.json'
    audit, aseal = C.read(auditpath), C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if (pred['rows'] != 238 or pred['TEST'] != 49 or pred['arms'] != list(V.ARMS)
            or not pred['all_declared_arms_complete'] or not pred['all7_arms_complete']
            or C.sha(V.OUTPUT / 'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(V.OUTPUT / 'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(V.OUTPUT / 'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(V.OUTPUT / 'INPUTS.jsonl.gz') != inputs['input_sha256']
            or C.sha(V.OUTPUT / 'PROTOCOL.json') != inputs['protocol_sha256']
            or C.sha(V.OUTPUT / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']
            or C.sha(V.OUTPUT / 'TEST_SCORE.json') != score['score_sha256']['TEST']
            or audit['rows'] != 238 or audit['failure_count'] or audit['failures']
            or C.sha(auditpath) != aseal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(V.OUTPUT / 'PREDICTION_SEAL.json')):
        raise ValueError('Complete immutable all238 prediction, score and zero-failure audit required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    return protocol


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Append-only diagnostic generation')
    parent = _verify_parent()
    sources = dict(parent['source_sha256'])
    for module in (A, A.NATIVE, A.SOURCE, A.SOURCE_CAL, A.TCAL, A.O, A.S, A.MASK_CORE):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    paths = [V.OUTPUT / name for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'PREDICTION_SEAL.json',
        'SCORE_SEAL.json', 'INPUTS.jsonl.gz', 'PREDICTIONS.jsonl.gz', 'TEST_SCORE.json',
        'independent_root_audit_v1/AUDIT.json', 'independent_root_audit_v1/AUDIT_SEAL.json')]
    for module in (A.SOURCE_CAL, A.TCAL, A.OFIT, A.SFIT):
        paths.extend(module.OUTPUT / name for name in ('MODEL.json', 'MODEL_SEAL.json'))
    bindings = {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}
    output.mkdir(parents=True)
    plan = dict(schema='FROZEN_E_POSE_CURRENT_V24_OBJECTIVE_DIAGNOSIS_PLAN_V25',
        UTC=datetime.now(timezone.utc).isoformat(), population='ALL original TEST49; no filtering',
        TEST_ids=parent['ids'][46:95], source_arm=ARM, source_profile='SOURCE with continuous wholemask',
        focus_suffixes_posthoc_display_only=list(FOCUS_SUFFIXES), fixed_all49_same_analysis=True,
        operation='transport frozen e physical pose into current selected WD; fresh constraints and N+O+S+lambda*M',
        original_e_pose_runtime_candidate_or_fallback_insertion=False, new_pose_optimization_or_model_fit=False,
        model_or_runtime_threshold_selection=False, new_image_or_NN_or_GT_reference_decode=False,
        S_U_raw_or_local_queries=0, posthoc_already_sealed_scalar_score_only=True,
        translation_threshold_for_posthoc_reporting_cm=5.,
        read_scope='stream first95 lines, decode TEST49 metadata only; do not decode remaining143',
        source_sha256=sources, input_sha256=bindings, protected_user_sha256=parent['protected_user_sha256'],
        source_and_plan_frozen_before_TEST_journal_or_scalar_score_decode=True,
        current_default_e_unchanged=True)
    C.write(output / 'PLAN.json', plan)
    C.write(output / 'PREPARATION_SEAL.json', dict(plan_sha256=C.sha(output / 'PLAN.json'),
        source_sha256=sources, input_sha256=bindings, before_diagnostic_TEST_decode=True))
    print(dict(prepared_TEST=49, plan_sha256=C.sha(output / 'PLAN.json')), flush=True)


def _TEST_prefix(path, expected_ids):
    """Read bounded gzip prefix; JSON-decode only the TEST49 lines."""
    import gzip
    import json
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(islice(stream, 95)):
            if index < 46:
                continue
            row = json.loads(line)
            if row['partition'] != 'TEST' or row['id'] != expected_ids[index - 46]:
                raise ValueError('Exact original ordered TEST prefix required')
            yield row


def _evaluate(problem, task, R, t, regions, oparameters, sprepared):
    fresh = problem.evaluate(R, t)
    p = A.NATIVE.V._pose_vector(R, t, problem.scale)
    native, _, parts = problem.objective(p)
    ov, _, oa = A.O.bounded_o_localization_value_and_gradient(p, problem.xyz, problem.K,
        problem.scale, regions, **oparameters)
    rebound = A.S.rebind_bounded_visibility_trial_geometry(sprepared, problem.xyz)
    sv, _, sa = A.S.coarse_visibility_prior_value_and_gradient(p, problem.xyz, problem.scale, rebound)
    cfg = A.MASK_CORE._outline_config(task['outline']['configuration'])
    supported = bool(task['outline'].get('supported'))
    weight = float(cfg['outline_factor_weight']) if supported else 0.
    mask, mask_audit = 0., dict(valid_projection=True, enabled=False)
    if supported:
        mask, _, mask_audit = A.MASK_CORE.outline_value_and_gradient(p, problem.xyz,
            problem.K, problem.scale, task['outline'], cfg)
    valid_mask = bool(mask_audit.get('valid_projection') and np.isfinite(mask) and mask >= 0)
    finite = bool(np.isfinite([native, ov, sv, mask]).all())
    total = float(native + ov + sv + weight * mask) if finite and valid_mask else None
    kinds, factors = _costs(parts['native_costs'])
    return dict(accepted_current_hard_and_numeric_guards=bool(fresh['accepted'] and valid_mask and finite),
        original_hard_guard_accepted=bool(fresh['accepted']), mask_numeric_valid=valid_mask,
        native=float(native), O_prior=float(ov), S_prior=float(sv), mask=float(mask),
        mask_weight=weight, weighted_mask=float(weight * mask), total=total,
        native_by_kind=kinds, native_by_factor=factors,
        minimum_joint_slack=fresh.get('minimum_joint_slack'),
        guard_audits={key: fresh.get(key) for key in ('visible_protection_audit', 'joint_constraint_audit',
            'U_region_constraint_audit', 'boundary_observation_audit')},
        O_point_terms=oa['O_point_terms'], S_image_or_2D_terms=sa['S_image_or_2D_terms'])


def run(output=OUTPUT):
    output = Path(output).resolve()
    if (output / 'DIAGNOSIS.json').exists():
        raise FileExistsError('Preserve completed diagnostic')
    plan, seal = C.read(output / 'PLAN.json'), C.read(output / 'PREPARATION_SEAL.json')
    if C.sha(output / 'PLAN.json') != seal['plan_sha256']:
        raise ValueError('Frozen diagnostic plan changed')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(plan[key])
    _verify_parent()
    edge = C.read(A.SOURCE_CAL.OUTPUT / 'MODEL.json')
    tangent = C.read(A.TCAL.OUTPUT / 'MODEL.json')
    omodel = C.read(A.OFIT.OUTPUT / 'MODEL.json')
    oparameters = A.OFIT.prior_parameters(omodel)
    smodel = C.read(A.SFIT.OUTPUT / 'MODEL.json')
    A.S.validate_model(smodel)
    # The numerical values below are copied from sealed scalar score reports,
    # after all diagnostic sources/recipes were pinned. No reference is read.
    scores = C.read(V.OUTPUT / 'TEST_SCORE.json')
    if scores['count'] != 49 or [r['id'] for r in scores['rows']] != plan['TEST_ids']:
        raise ValueError('Keep complete scored TEST49 including every fallback')
    scored = {r['id']: r for r in scores['rows']}
    rows, classifications, regressions = [], Counter(), []
    maximum_decomposition_difference = 0.
    for task, row in zip_longest(_TEST_prefix(V.OUTPUT / 'INPUTS.jsonl.gz', plan['TEST_ids']),
            _TEST_prefix(V.OUTPUT / 'PREDICTIONS.jsonl.gz', plan['TEST_ids'])):
        if task is None or row is None or task['id'] != row['id']:
            raise ValueError('Exact all49 matched prediction inputs required')
        ident = task['id']
        if row['arms'][V.ARMS[0]] != task['baseline']:
            raise ValueError('Frozen e baseline changed')
        result = row['arms'][ARM]
        family = task['families'][result['WD_family']]
        e_family = task['families'][task['baseline']['WD_family']]
        gap = _physical_body_gap(e_family, family)
        if gap > 1e-12:
            raise ValueError('Same registered unordered physical body required')
        eR, et, transport = transport_physical_pose_to_family(task['baseline']['actual_pose'], family)
        selected_pose = result['actual_pose']
        newR, newt = np.asarray(selected_pose['R_cf']), np.asarray(selected_pose['centroid'])
        problem, _ = A._reconstruct_native(task, family, edge, tangent, source_preserving=True)
        original = task['families']['ORIGINAL_WD']
        coarse = original['coarse_pose']
        sprepared = A.S.prepare_bounded_coarse_visibility_prior(original['native_xyz'], coarse['R'], coarse['t'],
            np.asarray(task['initial_self_hidden'], bool), smodel)
        archives = []
        for record in task['O_localization_abstention']['records']:
            d = record['original_domain']
            if (task['initial_self_hidden'][d['native_corner']] or task['states'][d['native_corner']] != 'O'
                    or d['binding']['calibration_model_sha256'] != omodel['source_O_model_digest']):
                raise ValueError('Exact current O state/source model required')
            archives.append(d)
        regions = A.O.prepare_o_localization_regions(archives, task['states'], task['initial_self_hidden'])
        old = _evaluate(problem, task, eR, et, regions, oparameters, sprepared)
        new = _evaluate(problem, task, newR, newt, regions, oparameters, sprepared)
        if result['accepted']:
            delta = abs(new['total'] - result['solution']['objective'])
            maximum_decomposition_difference = max(maximum_decomposition_difference, delta)
            if delta > 1e-8 * max(1., abs(new['total'])):
                raise ValueError('Fresh selected current objective differs from sealed solution')
        classification = classify_comparison(old['accepted_current_hard_and_numeric_guards'],
            result['accepted'], old['total'], new['total'])
        classifications[classification] += 1
        metric = scored[ident]['metrics']
        prior_metric, current_metric = metric[V.ARMS[0]], metric[ARM]
        regression = bool(prior_metric['translation_cm'] <= 5. < current_metric['translation_cm'])
        if regression:
            regressions.append(ident)
        kinds = sorted(set(old['native_by_kind']) | set(new['native_by_kind']))
        factor_keys = sorted(set(old['native_by_factor']) | set(new['native_by_factor']), key=str)
        factors = [dict(kind=key[0], native_corner=key[1], native_neighbor=key[2],
            e_pose_cost=old['native_by_factor'].get(key, 0.), selected_cost=new['native_by_factor'].get(key, 0.),
            e_minus_selected=old['native_by_factor'].get(key, 0.) - new['native_by_factor'].get(key, 0.))
            for key in factor_keys]
        for evaluation in (old, new):
            del evaluation['native_by_factor']
        rows.append(dict(id=ident, partition='TEST', focus_posthoc_display=any(ident.endswith(s) for s in FOCUS_SUFFIXES),
            states=task['states'], current_selected_family=result['WD_family'],
            saved_e_predicted_family=task['baseline']['WD_family'], same_unordered_physical_body_gap_m=gap,
            physical_Q_transport=transport, diagnostic_e_R_cf=eR.tolist(), diagnostic_e_centroid=et.tolist(),
            selected_R_cf=newR.tolist(), selected_centroid=newt.tolist(),
            diagnostic_e_current_evaluation=old, selected_current_evaluation=new,
            current_selected_was_accepted=result['accepted'], classification=classification,
            e_minus_selected_by_kind={key: old['native_by_kind'].get(key, 0.) - new['native_by_kind'].get(key, 0.) for key in kinds},
            e_minus_selected_OS_mask={key: old[key] - new[key] for key in ('O_prior', 'S_prior', 'weighted_mask')},
            native_factors=factors,
            sealed_scalar_metrics_posthoc_only=dict(e=prior_metric, current=current_metric,
                translation_threshold_regression=regression),
            runtime_candidate_or_fallback_insertions=0))
    if len(rows) != 49 or [r['id'] for r in rows] != plan['TEST_ids']:
        raise ValueError('No frame or fallback dropout')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(plan[key])
    data = dict(schema='FROZEN_E_POSE_CURRENT_V24_OBJECTIVE_DIAGNOSIS_V25', rows=rows,
        count=49, source_arm=ARM, all_TEST49_included=True, summary=dict(classifications=dict(classifications),
            e_pose_currently_feasible=sum(r['diagnostic_e_current_evaluation']['accepted_current_hard_and_numeric_guards'] for r in rows),
            selected_accepted=sum(r['current_selected_was_accepted'] for r in rows),
            posthoc_translation_threshold_regressions=regressions,
            maximum_selected_objective_decomposition_difference=maximum_decomposition_difference),
        plan=E._binding(output / 'PLAN.json'), preparation_seal=E._binding(output / 'PREPARATION_SEAL.json'),
        no_e_pose_runtime_candidate_or_fallback_addition=True, no_GT_reference_pose_or_2D_decode=True,
        S_U_raw_or_local_image_queries=0, new_image_NN_model_fit_or_pose_optimizations=0,
        bounded_search_gap_is_not_global_optimum_or_accuracy_proof=True,
        objective_preference_is_not_identified_causal_image_error=True,
        scalar_metrics_are_reused_derived_reference_posthoc_not_independent_metrology=True,
        all_original_sources_predictions_models_default_and_protected_files_unchanged=True)
    C.write(output / 'DIAGNOSIS.json', C.finite(data))
    C.write(output / 'DIAGNOSIS_SEAL.json', dict(diagnosis_sha256=C.sha(output / 'DIAGNOSIS.json'),
        plan_sha256=C.sha(output / 'PLAN.json'), source_sha256=plan['source_sha256'],
        input_sha256=plan['input_sha256'], protected_user_sha256=plan['protected_user_sha256'], no_new_fits_or_reference_decodes=True))
    print(data['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args.output)
