"""Independent selected-pose audit for the seven uniform v24 controls.

No experiment problem factory or regularizer/mask wrapper certifies its own
objective. Rebuild pure native/authentication kernels, replace complete
tangent variance algebraically, and add pure O/S/outline values separately.
Inputs/predictions are streamed; no image/reference decode or optimization.
Run the full audit only after root's memory handoff and a complete seal.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import math

import numpy as np

from . import evaluate_source_preserving_tangent_mask_v4_24 as V
from . import native_edge_problem_v4_17 as NATIVE
from . import complete_bi_image_chain_evidence_v4_20d as ALL
from . import source_preserving_complete_bi_evidence_v4_24 as SOURCE
from . import calibrate_complete_bi_edge_mixture_v4_20d as ALL_CAL
from . import calibrate_source_preserving_bi_edge_mixture_v4_24 as SOURCE_CAL
from . import calibrate_archived_complete_tangent_v4_24 as TCAL
from . import bounded_o_localization_prior_v4_21 as O
from . import calibrate_bounded_o_localization_v4_21 as OFIT
from . import bounded_coarse_visibility_prior_v4_20 as S
from . import calibrate_coarse_visibility_prior_v4_20 as SFIT
from . import statewise_partial_BI_outline_solver_v4_13 as MASK_CORE


def _reconstruct_native(task, family, edge_model, tangent_model, *, source_preserving):
    """Pure shared cores, independently composed; no V._problem call."""
    spec = task['spec']
    raw = NATIVE.prepare_native_edge_problem(family['native_xyz'], spec['K'], spec['states'],
        spec['visible'], spec['domains'], spec['image_hw'], family['coarse_pose'],
        config={**spec['config'], 'prior_weight': 0.}, boundary_observations=task['boundary_merged'],
        u_regions=task['u_regions'], edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    auth, calibration = (SOURCE, SOURCE_CAL) if source_preserving else (ALL, ALL_CAL)
    bundle = auth.authenticate_complete_bi_image_chain_groups(raw, task['new_chain_observation'],
        junction_proposals=task['complete_BI_evidence']['junction_proposals'],
        visible_measurements=spec['visible'], initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'], image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],
        **calibration.mixture_noise_parameters(edge_model))
    problem = auth.CompleteBIImageChainProblem(raw, bundle, overlap_policy=ALL.PROTOTYPE_OVERLAP_POLICY)
    parameters = TCAL.tangent_variance_parameters(tangent_model)
    for record in problem.plan['corner_plans']:
        if record.get('tangent_kind') != 'ARCHIVED_COMPLETE_BI_TANGENT':
            continue
        k = record['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B', 'I'):
            raise ValueError('TCAL requires an authoritative nonS BI tangent')
        tangent = np.asarray(record['tangent_unit_xy'], float)
        covariance = np.asarray(record['original_measurement_archive']['original_effective_fit_covariance_px2'], float)
        axis = float(tangent @ covariance @ tangent)
        if (not np.isclose(tangent @ tangent, 1., atol=1e-10, rtol=0)
                or axis <= 0 or not np.isclose(record['tangent_variance_px2'], 4. * axis, atol=1e-10, rtol=1e-10)):
            raise ValueError('Exact original tangent axis/covariance and saved x4 variance required')
        record['tangent_variance_px2'] = parameters['pixel_floor_variance_px2'] + parameters['gamma'] * axis
    return problem, raw


def _plan_signature(plan):
    """Only actual likelihood/owner archives, independent of extra prose tags."""
    corners = []
    for record in plan['corner_plans']:
        item = {key: record[key] for key in ('native_corner', 'state', 'active_physical_edges',
            'point_cost_revoked', 'original_measurement_archive')}
        for key in ('tangent_kind', 'tangent_unit_xy', 'tangent_variance_px2'):
            if key in record:
                item[key] = record[key]
        corners.append(item)
    groups = [{key: group[key] for key in ('physical_native_edge', 'components', 'component_weights',
        'reliability_weight', 'replaces_normal_factor_refs', 'activation_owners')}
        for group in plan['active_groups']]
    return dict(corner_plans=corners, active_groups=groups,
        revoked_complete_ids=plan['revoked_complete_ids'],
        replaces_normal_factor_refs=plan['replaces_normal_factor_refs'])


def _native_cost_buckets(records):
    result = defaultdict(float)
    for record in records:
        key = (record['kind'], record.get('native_corner'), record.get('native_neighbor'))
        result[key] += float(record['weighted_robust_cost'])
    return dict(result)


def _selection_energies(native, prior, mask, weight, *, continuous_mask, mask_selector):
    """Mask belongs once in optimizer C and once in the same selector total."""
    regularized = math.fsum((native, prior))
    weighted = float(weight * mask)
    return dict(solution_objective=regularized + (weighted if continuous_mask else 0.),
        selector_objective=regularized + (weighted if mask_selector else 0.),
        regularized_native_objective=regularized, weighted_mask_objective=weighted)


def _model_readout_checks(task, result):
    """S/U/oldpartial semantics also apply when the pose is a fallback."""
    family = task['families'][result['WD_family']]
    pose, readout = result['actual_pose'], result['keypoint_readout']
    projected = V.C.project(np.asarray(family['native_xyz']), np.asarray(pose['R_cf']),
        np.asarray(pose['centroid']), np.asarray(task['spec']['K']))
    partial = {record['native_corner'] for record in task['boundary_merged']}
    checks = []
    for k, state in enumerate(task['states']):
        if state == 'S':
            checks.append((np.allclose(readout['reported_points'][k], projected[k], atol=1e-8, rtol=0),
                'S pose-projection-only readout including fallback'))
        elif state == 'U' or k in partial:
            checks.append((readout['reported_points'][k] is None,
                'U/oldpartial is not a precise KP including fallback'))
        elif state in ('O', 'T') and not result['accepted']:
            checks.append((readout['reported_points'][k] is None, 'fallback does not solve O/T'))
    return checks


def main(output=None):
    C, E = V.C, V.E
    output = Path(V.OUTPUT if output is None else output)
    dest = output / 'independent_root_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve previous independent v24 audit')
    protocol = C.read(output / 'PROTOCOL.json')
    seal = C.read(output / 'PREDICTION_SEAL.json')
    inputseal = C.read(output / 'INPUT_SEAL.json')
    parent = V.OLD.V.OUTPUT  # Frozen v19 start names, not a selected old pose.
    parentseal = C.read(parent / 'PREDICTION_SEAL.json')
    if (seal['rows'] != 238 or seal.get('all_declared_arms_complete') is not True
            or seal.get('all7_arms_complete') is not True or seal.get('arms') != list(V.ARMS)
            or seal.get('arm_count') != 7 or protocol['arms'] != list(V.ARMS)
            or seal['predictions_sha256'] != C.sha(output / 'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or seal['input_seal_sha256'] != C.sha(output / 'INPUT_SEAL.json')
            or inputseal['input_sha256'] != C.sha(output / 'INPUTS.jsonl.gz')
            or inputseal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or parentseal['rows'] != 238
            or parentseal['predictions_sha256'] != C.sha(parent / 'PREDICTIONS.jsonl.gz')):
        raise ValueError('Complete sealed current238 and frozen start-policy parent required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    allmodel = C.read(ALL_CAL.OUTPUT / 'MODEL.json')
    sourcemodel = C.read(SOURCE_CAL.OUTPUT / 'MODEL.json')
    tangentmodel = C.read(TCAL.OUTPUT / 'MODEL.json')
    ALL_CAL.mixture_noise_parameters(allmodel)
    SOURCE_CAL.mixture_noise_parameters(sourcemodel)
    TCAL.tangent_variance_parameters(tangentmodel)
    omodel = C.read(OFIT.OUTPUT / 'MODEL.json')
    oparameters = OFIT.prior_parameters(omodel)
    smodel = C.read(SFIT.OUTPUT / 'MODEL.json')
    S.validate_model(smodel)
    checks, failures, maximum = 0, [], 0.
    counts, selected_count, fallback_count = Counter(), 0, 0

    def check(condition, what, ident, arm=None):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(dict(id=ident, arm=arm, check=what))

    def near(first, second):
        nonlocal maximum
        delta = abs(float(first) - float(second))
        maximum = max(maximum, delta)
        return delta <= 1e-8 * max(1., abs(float(first)), abs(float(second)))

    rows = 0
    for task, row, previous in zip_longest(C.rows(output / 'INPUTS.jsonl.gz'),
            C.rows(output / 'PREDICTIONS.jsonl.gz'), C.rows(parent / 'PREDICTIONS.jsonl.gz')):
        if task is None or row is None or previous is None:
            raise ValueError('All three ordered populations must retain every frame')
        rows += 1
        ident = task['id']
        counts[task['partition']] += 1
        check(ident == row['id'] == previous['id'] == protocol['ids'][rows-1], 'ordered all238', ident)
        check(row['arms'][V.ARMS[0]] == task['baseline'], 'exact e baseline', ident)
        check(set(row['arms']) == set(V.ARMS), 'all seven arms', ident)
        check(row['states'] == task['states'] == task['spec']['states'], 'authoritative states', ident)
        original = task['families']['ORIGINAL_WD']
        coarse = original['coarse_pose']
        sprepared = S.prepare_bounded_coarse_visibility_prior(original['native_xyz'], coarse['R'], coarse['t'],
            np.asarray(task['initial_self_hidden'], bool), smodel)
        archives = []
        for record in task['O_localization_abstention']['records']:
            domain = record['original_domain']
            k = domain['native_corner']
            if task['initial_self_hidden'][k] or task['states'][k] != 'O':
                raise ValueError('O state gate precedes localization archive reads')
            check(domain['binding']['calibration_model_sha256'] == omodel['source_O_model_digest'],
                'source O-radius model binding', ident)
            archives.append(domain)
        regions = O.prepare_o_localization_regions(archives, task['states'], task['initial_self_hidden'])
        cfg = MASK_CORE._outline_config(task['outline']['configuration'])
        supported = bool(task['outline'].get('supported'))
        weight = float(cfg['outline_factor_weight']) if supported else 0.
        problems = {}
        oldsearches = previous['native_refinement_audit'][V.OLD.V.ARMS[1]]['family_refinements']
        for profile, arms in V.PROFILE_ARMS.items():
            if profile not in ('ALL', 'SOURCE', 'MASK_FIT') or len(arms) != 2:
                raise ValueError('Exact declared source-defined profile mapping required')
            newsearches = row['bounded_prior_refinement_audit'][arms[0]]['family_refinements']
            check(set(newsearches) == set(task['families']) == set(oldsearches), 'same search families', ident, arms[0])
            for family_name in newsearches:
                expected = oldsearches[family_name]['start_selection']['selected_start_names']
                check(newsearches[family_name]['start_selection']['selected_start_names'] == expected,
                    'exact v19 original-native start names', ident, arms[0])
            for index, arm in enumerate(arms):
                result = row['arms'][arm]
                family = task['families'][result['WD_family']]
                pose, solution = result['actual_pose'], result['solution']
                R, t = np.asarray(pose['R_cf']), np.asarray(pose['centroid'])
                check(np.allclose(R @ np.asarray(family['physical_registration_Q']),
                    pose['R_physical'], atol=1e-12, rtol=0), 'exact physical Q', ident, arm)
                locked = profile == 'MASK_FIT' and index == 1
                lockaudit = result['early_family_audit']
                check(lockaudit['enabled'] is locked and lockaudit['e_numeric_pose_used'] is False,
                    'only source-defined seventh arm locks predicted WD label', ident, arm)
                for condition, explanation in _model_readout_checks(task, result):
                    check(condition, explanation, ident, arm)
                if locked:
                    check(lockaudit['requested_predicted_family'] == task['baseline']['WD_family'],
                        'stored predicted e family only', ident, arm)
                    check(result['candidate_selection']['e_numeric_pose_used'] is False,
                        'no e numeric pose fallback in lock selector', ident, arm)
                if not result['accepted']:
                    fallback_count += 1
                    check(pose['fallback'] and np.array_equal(R, np.asarray(coarse['R']))
                        and np.array_equal(t, np.asarray(coarse['t'])), 'own initial coarse fallback only', ident, arm)
                    check(not result['fallback_audit']['old_best_pose_fallback'], 'no old-best pose fallback', ident, arm)
                    if locked:
                        check(result['WD_family'] == 'ORIGINAL_WD'
                            and lockaudit['locked_family_constraint_satisfied'] is False
                            and result['fallback_audit']['lock_requested_but_unsatisfied'] is True
                            and result['candidate_selection']['selected_name'] is None,
                            'unsatisfied lock has explicit own-original fallback, no e pose substitution', ident, arm)
                    continue
                selected_count += 1
                key = (profile != 'ALL', result['WD_family'])
                if key not in problems:
                    problems[key] = _reconstruct_native(task, family, sourcemodel if key[0] else allmodel,
                        tangentmodel, source_preserving=key[0])
                problem, raw = problems[key]
                fresh = problem.evaluate(R, t)
                check(fresh['accepted'], 'fresh original all-guard acceptance', ident, arm)
                p = NATIVE.V._pose_vector(R, t, problem.scale)
                actual_v, actual_j = problem.constraints(p, problem.sides[0])
                old_v, old_j = raw.constraints(p, raw.sides[0])
                check(np.array_equal(actual_v, old_v) and np.array_equal(actual_j, old_j),
                    'unchanged hardguard values/Jacobian', ident, arm)
                check(E._digest(_plan_signature(problem.plan)) == E._digest(
                    _plan_signature(solution['complete_BI_replacement_audit'])),
                    'exact source/owner/normalref/q/noise/tangent plan', ident, arm)
                check(solution['replaced_old_normal_factor_count'] == len(problem.plan['replaces_normal_factor_refs']),
                    'normal replacement count derives from exact active refs', ident, arm)
                if key[0]:
                    for corner in problem.plan['corner_plans']:
                        if corner['point_cost_revoked']:
                            check(corner['original_measurement_archive']['original_measurement_source'] == SOURCE.FALLBACK_SOURCE,
                                'only exact fallback current-complete q revoked', ident, arm)
                for corner in problem.plan['corner_plans']:
                    if corner.get('tangent_kind'):
                        stored = next(r for r in solution['complete_BI_replacement_audit']['corner_plans']
                            if r['native_corner'] == corner['native_corner'])
                        check(near(corner['tangent_variance_px2'], stored['tangent_variance_px2']),
                            'variance 1+gamma*t_original*Sigma_original*t_original', ident, arm)
                native = float(fresh['objective'])
                check(near(native, solution['native_observation_objective']), 'independent complete native energy', ident, arm)
                first, second = _native_cost_buckets(fresh['native_costs']), _native_cost_buckets(solution['native_costs'])
                check(first.keys() == second.keys() and all(near(first[k], second[k]) for k in first),
                    'native factor kinds/vertex association/reliability once', ident, arm)
                ov, _, oa = O.bounded_o_localization_value_and_gradient(p, problem.xyz, problem.K,
                    problem.scale, regions, **oparameters)
                rebound = S.rebind_bounded_visibility_trial_geometry(sprepared, problem.xyz)
                sv, _, sa = S.coarse_visibility_prior_value_and_gradient(p, problem.xyz, problem.scale, rebound)
                prior = math.fsum((ov, sv))
                actual_priors = {r['kind']: r['weighted_cost'] for r in solution['regularizer_costs']}
                check(set(actual_priors) == {'O_COARSE_LOCALIZATION_PRIOR', 'S_COARSE_VISIBILITY_PRIOR'},
                    'mask excluded from two named frame regularizers', ident, arm)
                check(near(actual_priors['O_COARSE_LOCALIZATION_PRIOR'], ov)
                    and near(actual_priors['S_COARSE_VISIBILITY_PRIOR'], sv)
                    and near(solution['regularizer_objective'], prior), 'independent O/S energy', ident, arm)
                check(ov <= -2*math.log(oparameters['epsilon']) + 1e-10
                    and sv <= -2*math.log(smodel['epsilon']) + 1e-10,
                    'bounded frame prior caps', ident, arm)
                check(oa['O_point_terms'] == 0 and sa['S_image_or_2D_terms'] == 0,
                    'O/S no hidden-point image factor', ident, arm)
                mv, ma = 0., None
                if supported:
                    mv, _, ma = MASK_CORE.outline_value_and_gradient(p, problem.xyz, problem.K,
                        problem.scale, task['outline'], cfg)
                    check(ma.get('valid_projection') and np.isfinite(mv) and mv >= 0,
                        'pure frozen whole-mask projection validity', ident, arm)
                is_continuous = profile == 'MASK_FIT'
                mask_selector = is_continuous or index == 1
                energies = _selection_energies(native, prior, mv, weight,
                    continuous_mask=is_continuous, mask_selector=mask_selector)
                check(near(solution['objective'], energies['solution_objective']),
                    'solution objective has mask exactly once only for C', ident, arm)
                if is_continuous:
                    check(near(solution['weighted_mask_objective'], energies['weighted_mask_objective'])
                        and near(solution['global_mask_cost'], mv)
                        and near(solution['native_plus_coarse_regularized_objective'], native + prior)
                        and solution['mask_count_in_optimizer'] == int(supported and weight > 0),
                        'continuous-mask separate ledger/no duplicate objective', ident, arm)
                selection = result['candidate_selection']
                candidates = selection['candidates']
                chosen = [candidate for candidate in candidates if candidate['name'] == selection['selected_name']]
                check(len(chosen) == 1, 'one selected authenticated representative', ident, arm)
                if chosen:
                    chosen = chosen[0]
                    check(near(chosen['regularized_native_energy'], native + prior)
                        and near(chosen['total_energy'], energies['selector_objective']),
                        'same selected energy in optimizer and declared selector', ident, arm)
                    check(near(chosen['global_mask_cost'], mv) if supported else chosen['global_mask_cost'] == 0.,
                        'same frozen global mask score', ident, arm)
                    minimum = min(c['total_energy'] for c in candidates)
                    tol = 32*np.finfo(float).eps * max(1., abs(minimum), abs(chosen['total_energy']))
                    check(chosen['total_energy'] <= minimum + tol, 'minimum declared candidate energy', ident, arm)
                if locked:
                    check(result['WD_family'] == task['baseline']['WD_family']
                        and lockaudit['locked_family_constraint_satisfied'], 'predicted e family constraint', ident, arm)
                readout = result['keypoint_readout']
                for k, state in enumerate(task['states']):
                    if k in problem.revoked:
                        check(readout['reported_points'][k] is None, 'revoked complete is not a precise KP', ident, arm)
        unlocked = row['arms'][V.PROFILE_ARMS['MASK_FIT'][0]]['candidate_selection']
        locked = row['arms'][V.PROFILE_ARMS['MASK_FIT'][1]]['candidate_selection']
        expected = {c['name']: c for c in unlocked['candidates'] if c['family'] == task['baseline']['WD_family']}
        observed = {c['name']: c for c in locked['candidates']}
        check(expected.keys() == observed.keys(), 'all feasible C-bank poses filtered only by predicted e WD label', ident)
        check(all(c['family'] == task['baseline']['WD_family'] for c in observed.values()),
            'no other-WD contender in locked bank', ident)
        check(all(all(near(expected[name][key], observed[name][key]) for key in
            ('native_energy', 'regularizer_energy', 'regularized_native_energy', 'global_mask_cost', 'total_energy'))
            for name in observed), 'lock changes eligibility only, no numeric pose or cost substitution', ident)
    check(rows == 238 and counts == Counter(CAL=46, TEST=49, REMAINDER=143), 'all238 severe0 retained', None)
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    modelpaths = [module.OUTPUT / 'MODEL.json' for module in (ALL_CAL, SOURCE_CAL, TCAL, OFIT, SFIT)]
    receipt = dict(schema='INDEPENDENT_SOURCE_PRESERVING_TCAL_CONTINUOUS_MASK_AUDIT_V24',
        rows=rows, partitions=dict(counts), primary192=192, severe=0, checks=checks,
        failure_count=len(failures), failures=failures, selected_arm_poses=selected_count,
        fallback_arm_poses=fallback_count, maximum_energy_difference=maximum,
        source=E._binding(Path(__file__)), prediction_seal=E._binding(output / 'PREDICTION_SEAL.json'),
        frozen_start_policy_prediction_seal=E._binding(parent / 'PREDICTION_SEAL.json'),
        model_bindings=[E._binding(path) for path in modelpaths],
        experiment_problem_and_regularizer_and_mask_wrappers_not_used_to_certify_objective=True,
        hardguard_comparison='exact first-branch values/Jacobian and independent all-branch scalar evaluate',
        complete_source_owner_plan_and_original_tangent_covariance_reconstructed=True,
        continuous_mask_factor_count_one=True, e_family_lock_uses_label_not_numeric_pose=True,
        audit_scope='independent selected-pose energy/guards/plan; stored nonselected energies for minimum check',
        nonselected_candidate_numeric_energies_recomputed=False,
        new_pose_fits_RGB_NN_or_reference_decodes=0, default_adoption=False)
    C.write(dest / 'AUDIT.json', receipt)
    C.write(dest / 'AUDIT_SEAL.json', dict(receipt_sha256=C.sha(dest / 'AUDIT.json')))
    print(dict(rows=rows, checks=checks, failures=len(failures), max_energy_difference=maximum), flush=True)
    if failures:
        raise ValueError('Independent v24 selected-pose audit failed')


if __name__ == '__main__':
    main()
