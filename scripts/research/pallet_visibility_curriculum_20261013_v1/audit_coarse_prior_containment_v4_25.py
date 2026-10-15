"""Selected-pose v25 audit, composed independently of new prior wrappers.

Frozen v24 native/authentication reconstruction is reused. O/S use their pure
kernels, the extra coarse prior calls frozen BIAS directly with CAL Sigma/mu,
and optional C calls the new pure outside-only containment kernel. No new
wrapper/factory certifies its own objective. Stream only after root memory GO.
"""
from __future__ import annotations

from collections import Counter
from itertools import zip_longest
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from . import audit_source_preserving_tangent_mask_v4_24 as A
from . import evaluate_coarse_prior_containment_v4_25 as V
from . import statewise_uncertain_solver_v4_bias as BIAS
from . import global_foreground_containment_v4_25 as CONTAINMENT


def _original_prior_fields(task, model, model_sha256, mode):
    """Independent original target, fixed before consulting a trial family."""
    original = task['families']['ORIGINAL_WD']
    R0, t0 = np.asarray(original['coarse_pose']['R']), np.asarray(original['coarse_pose']['t'])
    xyz0, Q0 = np.asarray(original['native_xyz']), np.asarray(original['physical_registration_Q'])
    return dict(schema='FIXED_ORIGINAL_PHYSICAL_CAL_COARSE_POSE_PRIOR_V25', mode=mode,
        factor_weight=1., bias_strength=1., calibration_model_digest_sha256=model_sha256,
        prior_covariance6=model['prior_covariance6'], prior_bias6=model['prior_fit']['median_bias6'],
        original_native_xyz=xyz0.tolist(), original_R_cf=R0.tolist(), original_t=t0.tolist(),
        original_physical_Q=Q0.tolist(), original_first_R_physical=(R0 @ Q0).tolist(),
        original_first_center_camera_xyz=(R0 @ xyz0.mean(axis=0) + t0).tolist())


def direct_old_P1_prior(p, xyz, scale, original, trial_Q):
    """Do not call v25 prepare/rebind/value/wrapper; mu is subtracted once."""
    covariance = np.asarray(original['prior_covariance6'], float)
    bias = np.asarray(original['prior_bias6'], float)
    Q = np.asarray(trial_Q, float)
    if (covariance.shape != (6, 6) or bias.shape != (6,)
            or not np.isfinite(covariance).all() or not np.isfinite(bias).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)):
        raise ValueError('Finite symmetric CAL6 covariance and mean required')
    np.linalg.cholesky(covariance)
    R0 = (np.asarray(original['original_R_cf']) if np.array_equal(Q, original['original_physical_Q'])
        else np.asarray(original['original_first_R_physical']) @ Q.T)
    center0 = np.asarray(original['original_first_center_camera_xyz'])
    center_xyz = np.asarray(xyz).mean(axis=0)
    if original['mode'] == 'FULL':
        residual, jacobian = BIAS._prior_residual_and_jacobian(p, scale, center_xyz, R0,
            center0, np.linalg.inv(np.linalg.cholesky(covariance)), bias)
    elif original['mode'] == 'TRANSLATION':
        raw, full_jacobian = BIAS._prior_residual_and_jacobian(p, scale, center_xyz, R0,
            center0, np.eye(6), None)
        whitening = np.linalg.inv(np.linalg.cholesky(covariance[3:, 3:]))
        residual, jacobian = whitening @ (raw[3:] - bias[3:]), whitening @ full_jacobian[3:]
    else:
        raise ValueError('Exact source-defined FULL or TRANSLATION mode required')
    return float(residual @ residual), 2. * residual @ jacobian, dict(
        trial_coarse_R_cf=R0.tolist(), original_center_camera_xyz=center0.tolist(),
        residual=residual.tolist(), marginal_translation=original['mode'] == 'TRANSLATION',
        bias_subtracted_once=True)


def _source_binding(fields):
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _energies(native, O, S, prior, containment, weight, enabled):
    regularizer = math.fsum((O, S, prior))
    shape = weight * containment if enabled else 0.
    return dict(regularizer=regularizer, regularized_native=native + regularizer,
        weighted_containment=shape, total=native + regularizer + shape)


def _native_endpoint_bank(task, row, mode):
    profiles = [arms for _, candidate_mode, enabled, arms in V.PROFILES
        if candidate_mode == mode and not enabled]
    if len(profiles) != 1:
        raise ValueError('One native endpoint profile per fixed prior mode required')
    searches = row['bounded_prior_refinement_audit'][profiles[0][0]]['family_refinements']
    records = []
    for family_name in task['families']:
        for endpoint in searches[family_name]['feasible_endpoints']:
            index = len(records)
            records.append(dict(name=family_name + '/NATIVE_ENDPOINT_' + str(index), family=family_name,
                R=endpoint['R'], t=endpoint['t']))
    return records


def main(output=None):
    C, E = V.C, V.E
    output = Path(V.OUTPUT if output is None else output)
    dest = output / 'independent_root_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve all previous audit evidence')
    protocol, seal = C.read(output / 'PROTOCOL.json'), C.read(output / 'PREDICTION_SEAL.json')
    inputseal = C.read(output / 'INPUT_SEAL.json')
    parent = A.V.OLD.V.OUTPUT
    parentseal = C.read(parent / 'PREDICTION_SEAL.json')
    if (seal['rows'] != 238 or seal['arms'] != list(V.ARMS) or seal['arm_count'] != 9
            or not seal['all9_arms_complete'] or not seal['all_declared_arms_complete']
            or protocol['arms'] != list(V.ARMS)
            or seal['predictions_sha256'] != C.sha(output / 'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or seal['input_seal_sha256'] != C.sha(output / 'INPUT_SEAL.json')
            or inputseal['input_sha256'] != C.sha(output / 'INPUTS.jsonl.gz')
            or inputseal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or parentseal['rows'] != 238
            or parentseal['predictions_sha256'] != C.sha(parent / 'PREDICTIONS.jsonl.gz')
            or protocol['coarse_prior_weights'] != dict(lambda_prior=1., beta_bias=1.)
            or protocol['old_full_outline_factor_or_tie_break_used'] is not False):
        raise ValueError('Exact all238 nine-arm seal and fixed lambda1/beta1 required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    binding = protocol['coarse_prior_model_binding']
    if C.sha(C.ROOT / binding['path']) != binding['sha256']:
        raise ValueError('Frozen CAL coarse covariance/mean binding changed')
    coarsemodel = C.read(C.ROOT / binding['path'])
    if coarsemodel['fit_partition'] != 'CAL':
        raise ValueError('Only the existing frozen CAL prior model is allowed')
    edge = C.read(A.SOURCE_CAL.OUTPUT / 'MODEL.json')
    tangent = C.read(A.TCAL.OUTPUT / 'MODEL.json')
    A.SOURCE_CAL.mixture_noise_parameters(edge)
    A.TCAL.tangent_variance_parameters(tangent)
    omodel, smodel = C.read(A.OFIT.OUTPUT / 'MODEL.json'), C.read(A.SFIT.OUTPUT / 'MODEL.json')
    op = A.OFIT.prior_parameters(omodel)
    A.S.validate_model(smodel)
    checks, failures, maximum, rows = 0, [], 0., 0
    counts, accepted, fallback = Counter(), 0, 0

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

    for task, row, previous in zip_longest(C.rows(output / 'INPUTS.jsonl.gz'),
            C.rows(output / 'PREDICTIONS.jsonl.gz'), C.rows(parent / 'PREDICTIONS.jsonl.gz')):
        if task is None or row is None or previous is None:
            raise ValueError('All ordered frames including fallbacks must remain')
        rows += 1
        ident = task['id']
        counts[task['partition']] += 1
        check(ident == row['id'] == previous['id'] == protocol['ids'][rows-1], 'ordered238', ident)
        check(row['arms'][V.ARMS[0]] == task['baseline'], 'exact frozen e baseline', ident)
        check(set(row['arms']) == set(V.ARMS), 'all nine declared arms', ident)
        check(task['states'] == row['states'] == task['spec']['states'], 'unchanged authoritative states', ident)
        first = task['families']['ORIGINAL_WD']
        coarse = first['coarse_pose']
        sprepared = A.S.prepare_bounded_coarse_visibility_prior(first['native_xyz'], coarse['R'], coarse['t'],
            np.asarray(task['initial_self_hidden'], bool), smodel)
        regions = []
        for r in task['O_localization_abstention']['records']:
            d = r['original_domain']
            k = d['native_corner']
            if task['initial_self_hidden'][k] or task['states'][k] != 'O':
                raise ValueError('O state gate precedes localization fields')
            check(d['binding']['calibration_model_sha256'] == omodel['source_O_model_digest'], 'O source-radius model', ident)
            regions.append(d)
        regions = A.O.prepare_o_localization_regions(regions, task['states'], task['initial_self_hidden'])
        cfg = A.MASK_CORE._outline_config(task['outline']['configuration'])  # Config only; never old mask cost.
        supported = bool(task['outline'].get('supported'))
        weight = cfg['outline_factor_weight'] if supported else 0.
        originals = {mode: _original_prior_fields(task, coarsemodel, binding['sha256'], mode)
            for mode in ('TRANSLATION', 'FULL')}
        for mode, fields in originals.items():
            stored = row['original_first_pose_prior_inputs'][mode]
            check(all(E._digest(stored[k]) == E._digest(value) for k, value in fields.items())
                and stored['source_binding_sha256'] == _source_binding(fields),
                'original first physical pose/CAL parameters fixed before every WD trial', ident)
        oldsearches = previous['native_refinement_audit'][A.V.OLD.V.ARMS[1]]['family_refinements']
        problems = {}
        for profile, mode, containment, arms in V.PROFILES:
            if mode not in ('TRANSLATION', 'FULL') or V.PROFILE_ARMS[profile] != arms or len(arms) != 2:
                raise ValueError('Exact four source-defined optimization profiles required')
            searches = row['bounded_prior_refinement_audit'][arms[0]]['family_refinements']
            check(set(searches) == set(oldsearches) == set(task['families']), 'same two WD searches', ident, arms[0])
            for name, search in searches.items():
                check(search['start_selection']['selected_start_names'] == oldsearches[name]['start_selection']['selected_start_names'],
                    'exact frozen v19-native start names', ident, arms[0])
                check(search['search']['old_full_outline_factor_used'] is False
                    and search['search']['mask_objective'] is False,
                    'no old bidirectional mask optimization', ident, arms[0])
            for index, arm in enumerate(arms):
                result = row['arms'][arm]
                family = task['families'][result['WD_family']]
                pose, solution = result['actual_pose'], result['solution']
                R, t = np.asarray(pose['R_cf']), np.asarray(pose['centroid'])
                locked = index == 1
                lock = result['early_family_audit']
                check(lock['enabled'] is locked and lock['e_numeric_pose_used'] is False,
                    'only declared locked arm uses predicted WD label', ident, arm)
                check(np.allclose(R @ family['physical_registration_Q'], pose['R_physical'], atol=1e-12, rtol=0),
                    'selected physical Q unchanged', ident, arm)
                for condition, reason in A._model_readout_checks(task, result):
                    check(condition, reason, ident, arm)
                check(result['keypoint_readout']['old_full_outline_factor_used'] is False,
                    'no old full-mask readout claim', ident, arm)
                if locked:
                    check(lock['requested_predicted_family'] == task['baseline']['WD_family'], 'predicted e label only', ident, arm)
                if not result['accepted']:
                    fallback += 1
                    check(pose['fallback'] and result['WD_family'] == 'ORIGINAL_WD'
                        and np.array_equal(R, coarse['R']) and np.array_equal(t, coarse['t'])
                        and result['fallback_audit']['old_best_pose_fallback'] is False,
                        'own original coarse fallback only', ident, arm)
                    if locked:
                        check(lock['locked_family_constraint_satisfied'] is False
                            and result['fallback_audit']['lock_requested_but_unsatisfied']
                            and result['candidate_selection']['selected_name'] is None,
                            'failed family constraint is explicit', ident, arm)
                    continue
                accepted += 1
                if result['WD_family'] not in problems:
                    problems[result['WD_family']] = A._reconstruct_native(task, family, edge, tangent, source_preserving=True)
                native_problem, raw = problems[result['WD_family']]
                fresh = native_problem.evaluate(R, t)
                check(fresh['accepted'], 'independent original scalar all-guard acceptance', ident, arm)
                p = A.NATIVE.V._pose_vector(R, t, native_problem.scale)
                values, jac = native_problem.constraints(p, native_problem.sides[0])
                oldvalues, oldjac = raw.constraints(p, raw.sides[0])
                check(np.array_equal(values, oldvalues) and np.array_equal(jac, oldjac), 'hardguard values/Jacobian exact', ident, arm)
                check(E._digest(A._plan_signature(native_problem.plan)) == E._digest(A._plan_signature(solution['complete_BI_replacement_audit'])),
                    'exact SOURCE/TCAL ownership and original q/noise archives', ident, arm)
                check(solution['replaced_old_normal_factor_count'] == len(native_problem.plan['replaces_normal_factor_refs']),
                    'old normals replaced once', ident, arm)
                native = float(fresh['objective'])
                check(near(native, solution['native_observation_objective']), 'independent native image cost', ident, arm)
                left, right = A._native_cost_buckets(fresh['native_costs']), A._native_cost_buckets(solution['native_costs'])
                check(left.keys() == right.keys() and all(near(left[k], right[k]) for k in left), 'native factor identity and cost once', ident, arm)
                ov, _, oa = A.O.bounded_o_localization_value_and_gradient(p, native_problem.xyz, native_problem.K,
                    native_problem.scale, regions, **op)
                sr = A.S.rebind_bounded_visibility_trial_geometry(sprepared, native_problem.xyz)
                sv, _, sa = A.S.coarse_visibility_prior_value_and_gradient(p, native_problem.xyz, native_problem.scale, sr)
                pv, _, pa = direct_old_P1_prior(p, native_problem.xyz, native_problem.scale, originals[mode], family['physical_registration_Q'])
                parts = solution['regularizer_costs']
                kinds = ['O_COARSE_LOCALIZATION_PRIOR', 'S_COARSE_VISIBILITY_PRIOR', 'CAL_COARSE_POSE_PRIOR']
                check([part['kind'] for part in parts] == kinds, 'exact three separate priors; C excluded', ident, arm)
                actual = {part['kind']: part['weighted_cost'] for part in parts}
                check(all(near(actual[k], value) for k, value in zip(kinds, (ov, sv, pv))), 'independent O/S/direct-old-P1 prior costs', ident, arm)
                check(oa['O_point_terms'] == 0 and sa['S_image_or_2D_terms'] == 0, 'no hidden-point image prior', ident, arm)
                check(ov <= -2*math.log(op['epsilon']) + 1e-10 and sv <= -2*math.log(smodel['epsilon']) + 1e-10,
                    'O/S one-frame caps; coarse Gaussian not falsely capped', ident, arm)
                prior_audit = solution['CAL_coarse_pose_prior_audit']
                check(prior_audit['mode'] == mode and prior_audit['factor_weight'] == 1.
                    and prior_audit['bias_strength'] == 1. and prior_audit['calibration_model_digest_sha256'] == binding['sha256']
                    and prior_audit['mean_bias_subtracted_once'] is True and prior_audit['S_U_raw_image_2D_or_regions'] == 0,
                    'fixed CAL model/lambda1/beta1/no S/U image inputs', ident, arm)
                check(np.allclose(prior_audit['whitened_residual'], pa['residual'], atol=1e-9, rtol=1e-10)
                    and prior_audit['original_source_binding_sha256'] == _source_binding(originals[mode]),
                    'direct residual vector and original physical/CAL binding, not only squared norm', ident, arm)
                shape, ca = 0., None
                if containment and supported and weight > 0:
                    shape, _, ca = CONTAINMENT.outline_value_and_gradient(p, native_problem.xyz, native_problem.K,
                        native_problem.scale, task['outline'], cfg)
                    check(ca['valid_projection'] and ca['model_to_observed_factor_weight'] == 0.
                        and ca['inside_model_hull_cost_zero'] is True, 'pure outside-only C, no complete silhouette matching', ident, arm)
                energies = _energies(native, ov, sv, pv, shape, weight, containment)
                check(near(solution['regularizer_objective'], energies['regularizer'])
                    and near(solution['objective'], energies['total']), 'optimizer total N+O+S+P+C once', ident, arm)
                check(solution['coarse_pose_prior_count'] == 1 and near(solution['CAL_coarse_pose_prior_objective'], pv),
                    'coarse prior ledger once', ident, arm)
                if containment:
                    check(near(solution['global_containment_cost'], shape)
                        and near(solution['weighted_containment_objective'], energies['weighted_containment'])
                        and near(solution['native_plus_coarse_regularized_objective'], energies['regularized_native'])
                        and solution['containment_count_in_optimizer'] == int(supported and weight > 0)
                        and solution['old_bidirectional_full_outline_factor_used'] is False,
                        'independent C ledger and no old M', ident, arm)
                selection = result['candidate_selection']
                check(selection['old_full_outline_factor_or_tie_break_used'] is False
                    and selection['e_numeric_pose_used'] is False,
                    'no old mask tie-break or numeric e fallback', ident, arm)
                chosen = [c for c in selection['candidates'] if c['name'] == selection['selected_name']]
                check(len(chosen) == 1, 'one selected authenticated representative', ident, arm)
                if chosen:
                    chosen = chosen[0]
                    check(near(chosen['regularized_native_energy'], energies['regularized_native'])
                        and near(chosen['global_containment_cost'], shape)
                        and near(chosen['weighted_containment_objective'], energies['weighted_containment'])
                        and near(chosen['total_energy'], energies['total']), 'selector uses identical independent components once', ident, arm)
                    minimum = min(c['total_energy'] for c in selection['candidates'])
                    tol = 32*np.finfo(float).eps * max(1., abs(minimum), abs(chosen['total_energy']))
                    check(chosen['total_energy'] <= minimum + tol, 'minimum stored contender total', ident, arm)
                if locked:
                    check(result['WD_family'] == task['baseline']['WD_family'] and lock['locked_family_constraint_satisfied'],
                        'predicted family constraint satisfied', ident, arm)
                for k in native_problem.revoked:
                    check(result['keypoint_readout']['reported_points'][k] is None, 'revoked q not precise new KP', ident, arm)
            unlocked, locked = (row['arms'][arm]['candidate_selection'] for arm in arms)
            expected = {c['name']: c for c in unlocked['candidates'] if c['family'] == task['baseline']['WD_family']}
            observed = {c['name']: c for c in locked['candidates']}
            check(expected.keys() == observed.keys(), 'same feasible bank, predicted-family-only filter', ident, arms[1])
            check(all(c['family'] == task['baseline']['WD_family'] for c in observed.values()), 'no other family in lock bank', ident, arms[1])
            fields = ('native_energy', 'regularizer_energy', 'regularized_native_energy', 'global_containment_cost',
                'weighted_containment_objective', 'total_energy')
            check(all(all(near(expected[name][key], observed[name][key]) for key in fields) for name in observed),
                'lock changes eligibility only, no pose/cost substitution', ident, arms[1])
            if containment:
                check(row['bounded_prior_refinement_audit'][arms[0]]['same_mode_native_endpoints_retained'] is True,
                    'containment bank declares same-mode native endpoint retention', ident, arms[0])
                names = {c['name'] for c in unlocked['candidates']}
                for endpoint in _native_endpoint_bank(task, row, mode):
                    if endpoint['name'] in names:
                        check(True, 'same-mode native endpoint retained in accepted C bank', ident, arms[0])
                        continue
                    # A missing native endpoint is allowed only if the new C
                    # kernel itself is numerically invalid. No pose fit occurs.
                    valid = True
                    if supported and weight > 0:
                        f = task['families'][endpoint['family']]
                        xyz = np.asarray(f['native_xyz'])
                        scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
                        p = A.NATIVE.V._pose_vector(np.asarray(endpoint['R']), np.asarray(endpoint['t']), scale)
                        cv, cg, ca = CONTAINMENT.outline_value_and_gradient(p, xyz, np.asarray(task['spec']['K']),
                            scale, task['outline'], cfg)
                        valid = bool(ca['valid_projection'] and np.isfinite(cv) and cv >= 0 and np.isfinite(cg).all())
                    check(not valid, 'native endpoint absent only for independent C numeric invalidity', ident, arms[0])
    check(rows == 238 and counts == Counter(CAL=46, TEST=49, REMAINDER=143), 'all238 severe0 retained', None)
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    receipt = dict(schema='INDEPENDENT_COARSE_PRIOR_CONTAINMENT_SELECTED_POSE_AUDIT_V25', rows=rows,
        partitions=dict(counts), primary192=192, severe=0, checks=checks, failure_count=len(failures), failures=failures,
        selected_arm_poses=accepted, fallback_arm_poses=fallback, maximum_energy_difference=maximum,
        source=E._binding(Path(__file__)), prediction_seal=E._binding(output / 'PREDICTION_SEAL.json'),
        frozen_start_policy_prediction_seal=E._binding(parent / 'PREDICTION_SEAL.json'),
        coarse_model_binding=binding, prior_and_containment_wrappers_not_used_to_certify_own_objective=True,
        direct_old_BIAS_API_with_full_or_marginal_whitening_and_mean_once=True,
        no_old_full_outline_factor_or_numeric_e_fallback=True,
        same_mode_native_endpoint_retention_checked=True,
        missing_native_endpoint_containment_numeric_validity_only_recomputed_if_needed=True,
        nonselected_candidate_numeric_energies_recomputed=False,
        scope='independent selected-pose energy/guards/archives; stored nonselected energies for minimum check',
        new_pose_fits_RGB_NN_or_reference_decodes=0, default_adoption=False)
    C.write(dest / 'AUDIT.json', receipt)
    C.write(dest / 'AUDIT_SEAL.json', dict(receipt_sha256=C.sha(dest / 'AUDIT.json')))
    print(dict(rows=rows, checks=checks, failures=len(failures), max_energy_difference=maximum), flush=True)
    if failures:
        raise ValueError('Independent v25 selected-pose audit failed')


if __name__ == '__main__':
    main()
