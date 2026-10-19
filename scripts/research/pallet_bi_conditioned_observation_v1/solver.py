"""Common fresh point / partial-normal pose backend for the six scorer fits.

This is a narrow adapter, not a pose-bank selector.  It reuses the immutable
v4.13 typed parser, revocation validator and objective, v4 bias geometry /
canvas / depth constraints, and v4.9 two-WD representation conversion.  The
only search start in either family is the actual saved N0 pose.  Original
partial support intervals remain audit data: without physical endpoint
certification they supply no cuboid tangent constraint.

No RGB, reference, annotation, risk policy, or final-pose cache is read here.
Scorer conditions are deliberately absent from the public solve signature.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as V3
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_jacobian_v2 as J
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_u_solver_v4_13 as PART
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as WD
from scripts.research.pallet_visibility_curriculum_20261013_v1 import visible_bi_correction_v2 as CUE


SCHEMA = 'BI_CONDITIONED_OBSERVATION_COMMON_SOLVER_V1'
SOLVER_CONFIG = PART._configuration(dict(prior_weight=0., analytic_jacobian=True,
    max_p3p_triples=0, global_rotation_starts=0))
POLICY = dict(initializations_per_family=1, maximum_pose_optimizer_calls=2,
    initialization='actual saved N0, transformed by existing WD.make_wd_families',
    family_selection='existing WD minimum same image objective; exact tie ORIGINAL_WD',
    rank_policy='image information rank below 6 rejects to original N0',
    optimizer_acceptance='optimizer success AND finite pose/objective AND existing guards',
    physical_source_ownership='first native corner in numeric order; later correlated observations suppressed',
    partial_terms='existing typed normal objective only; complete likelihood revoked',
    partial_tangent_constraints=0, O_T_U_region_constraints=0,
    partial_raw_points_used_for_initialization=False,
    partial_original_xy_role='revocation archive only; no raw tangent cost, seed, prior, or Mahalanobis guard',
    source_identity_scope='known identical RGB sample-source reuse; material edge identity and unknown correlations uncertified',
    backend_states='immutable automatic states; scorer condition never accepted',
    prior_weight=0., H1=False, R_CONT=False, global_outline_cost=False)


def _json(value):
    return RT._json(value)


def contract():
    modules = [RT, V3, J, PART, PART.BASE, PART.N, WD, CUE]
    return dict(schema=SCHEMA, configuration=deepcopy(SOLVER_CONFIG), policy=deepcopy(POLICY),
        adapter_source_sha256=RT.sha256(__file__),
        source_sha256={str(Path(m.__file__).name): RT.sha256(m.__file__) for m in modules},
        source_functions=['runtime.registered_geometry', 'PART._boundary_inputs',
            'PART.validate_complete_BI_likelihood_revocations', 'PART._joint_objective_pair',
            'PART._normal_residual_and_jacobian', 'PART.BASE._constraints',
            'WD.make_wd_families', 'WD.solve_two_wd_families'])


def observation_digest(selection):
    """Exact observations + fixed backend; arm names / scorer tags cannot key it."""
    payload = dict(selection=selection, configuration=SOLVER_CONFIG, policy=POLICY)
    return hashlib.sha256(json.dumps(_json(payload), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def raw_selection(frame, states, *, original_cal_fit=None):
    """Original N0 points, no correction, with the existing v2 fallback noise.

    An original CAL fit is inherited only for an exactly identical observation,
    including conditional covariance.  A previous corrected winner's noise
    cannot be transplanted onto q0.
    """
    raw = frame.get('initial_points', frame.get('initial_points9x2'))
    if raw is None:
        raise ValueError('Actual N0 initial_points required')
    canonical = {V3._id(r): r for r in original_cal_fit or []}
    records = []
    for k, state in enumerate(states):
        if state not in ('B', 'I'):
            continue
        sigma = CUE.DEFAULT_CONFIG['fallback_boundary_sigma_px' if state == 'B'
            else 'fallback_interior_sigma_px']
        record = dict(native_corner=k, state=state, xy=deepcopy(raw[k]),
            covariance_px2=(np.eye(2) * sigma**2).tolist(),
            protection_px=CUE.DEFAULT_CONFIG['visible_protection_px'],
            source='VISIBLE_BACKBONE_FALLBACK', physical_source_ids=['N0_POINT:' + str(k)],
            provenance=dict(raw_N0=True, previous_winner_covariance_transplanted=False))
        old = canonical.get(k)
        if (old and old.get('state') == state and old.get('xy') == record['xy']
                and old.get('covariance_px2') == record['covariance_px2']):
            for key in ('fit_covariance_px2', 'fit_reliability_weight'):
                if key in old:
                    record[key] = deepcopy(old[key])
            record['provenance']['original_CAL_fit_inherited_exact_observation'] = True
        records.append(record)
    return dict(states=list(states), full_points=records, partial_lines=[])


def legacy_selection(source, *, partial_lines=None):
    """Convert canonical existing observations without recomputing winners.

    The caller chooses the documented canonical stage (normally the original
    minimal runtime visible_correction, or D1 original source).  The converter
    cannot read old pose outputs and makes no calibration choices.
    """
    states = list(source['states'])
    records = source.get('frozen_visible_measurements', source.get('visible_measurements'))
    if records is None and 'visible_correction' in source:
        records = source['visible_correction']['visible_measurements']
    if records is None:
        raise ValueError('Canonical legacy visible_measurements required')
    return dict(states=states, full_points=deepcopy(records), partial_lines=deepcopy(partial_lines or []))


def selection_from_candidates(bank, selected_ids):
    """Bind chosen IDs to applied coordinates and their own uncertainty.

    ``selected_ids`` maps native integer corner to an existing candidate ID.
    A shared typed partial replaces its archived original 2D measurement.
    """
    selection = dict(states=list(bank['states']), full_points=[], partial_lines=[],
        selected_candidates=[])
    for corner in sorted(bank['corners'], key=lambda c: c['native_corner']):
        k = int(corner['native_corner'])
        if bank['states'][k] not in ('B', 'I'):
            raise ValueError('Bank corners must be automatic visible B/I')
        partial = corner.get('partial_line')
        if partial is not None:
            original = deepcopy(corner['partial_original_visible'])
            selection['full_points'].append(original)
            selection['partial_lines'].append(deepcopy(partial))
            selection['selected_candidates'].append(dict(native_corner=k, kind='partial_normal',
                candidate_id=None, actual_full_xy=None))
            continue
        wanted = selected_ids.get(k, selected_ids.get(str(k)))
        matches = [c for c in corner['candidates'] if c['candidate_id'] == wanted]
        if len(matches) != 1:
            raise ValueError('Exactly one actual candidate ID required for each full corner')
        candidate = matches[0]
        record = dict(native_corner=k, state=bank['states'][k],
            xy=deepcopy(candidate['applied_xy']), covariance_px2=deepcopy(candidate['covariance_px2']),
            protection_px=CUE.DEFAULT_CONFIG['visible_protection_px'],
            source='VISIBLE_BACKBONE_FALLBACK' if candidate['kind'] == 'keep' else 'SHARED_LOCAL_IMAGE_INTERSECTION',
            physical_source_ids=deepcopy(candidate.get('physical_source_ids', [])),
            provenance=dict(candidate_id=wanted, source_line_ids=deepcopy(candidate.get('source_line_ids', [])),
                q0=deepcopy(candidate['q0']), raw_intersection_xy=deepcopy(candidate.get('raw_intersection_xy')),
                blend_alpha=candidate['blend_alpha'], actual_applied_coordinate=True))
        if candidate['kind'] == 'keep':
            for field in ('fit_covariance_px2', 'fit_reliability_weight'):
                if field in candidate:
                    record[field] = deepcopy(candidate[field])
        elif 'fit_covariance_px2' in candidate or 'fit_reliability_weight' in candidate:
            raise ValueError('A new intersection must carry only its own conditional covariance; old CAL fit cannot be transplanted')
        selection['full_points'].append(record)
        selection['selected_candidates'].append(dict(native_corner=k, kind=candidate['kind'],
            candidate_id=wanted, actual_full_xy=deepcopy(candidate['applied_xy'])))
    return selection


def _physical_ids(line):
    provenance = line.get('provenance', {})
    explicit = line.get('physical_source_ids', provenance.get('physical_source_ids'))
    if explicit:
        return [str(s) for s in explicit]
    explicit = line.get('physical_source_id', provenance.get('physical_source_id'))
    if explicit:
        return [str(explicit)]
    # Existing FACE records retain actual observed pixels.  Hash those, not
    # merely the undirected native edge (opposite local samples may differ).
    actual = provenance.get('line_record', {})
    observed = actual.get('observed_points')
    if observed is None:
        raise ValueError('Partial line needs a physical source ID or actual observed RGB pixels')
    payload = dict(points=observed, mask=actual.get('inlier_mask'),
        confidence=actual.get('observation_confidence'), normal=actual.get('normal'), offset=actual.get('offset'))
    return ['RGB_LINE:' + hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()]


def _deduplicate(visible, boundary):
    """Never count one physical image source at two native corners."""
    partial = {V3._id(r): deepcopy(r) for r in boundary}
    originals = {V3._id(r): deepcopy(r) for r in visible}
    owners = {}; kept = []; kept_partial = []; suppressed = []
    for k in sorted(originals):
        record = originals[k]
        if k in partial:
            group = partial[k]; lines = []
            for index, line in enumerate(group['lines']):
                ids = _physical_ids(line)
                overlap = [s for s in ids if s in owners]
                if overlap:
                    suppressed.append(dict(native_corner=k, kind='partial_normal', line_index=index,
                        physical_source_ids=ids, existing_owner_native_ids=sorted({owners[s] for s in overlap})))
                    continue
                for source in ids:
                    owners[source] = k
                lines.append(deepcopy(line))
            if lines:
                group['lines'] = lines; kept_partial.append(group); kept.append(record)
        else:
            ids = [str(s) for s in record.get('physical_source_ids', ['POINT:' + str(k)])]
            overlap = [s for s in ids if s in owners]
            if overlap:
                suppressed.append(dict(native_corner=k, kind='full_point', physical_source_ids=ids,
                    existing_owner_native_ids=sorted({owners[s] for s in overlap})))
                continue
            for source in ids:
                owners[source] = k
            kept.append(record)
    return kept, kept_partial, dict(physical_source_owners=owners, suppressed=suppressed,
        independent_reuse_of_duplicate_source=False, physical_material_edge_identity_certified=False,
        unknown_source_correlations_certified_independent=False)


def _prepare(xyz, K, states, visible, image_hw, boundary, cfg):
    # The existing parser requires all its B/I slots to carry a record.  Slots
    # without accepted evidence are U only in this structural parse; automatic
    # backend states remain immutable and are restored for the typed parser.
    ids = {V3._id(r) for r in visible}
    structural_states = [s if k in ids else 'S' if s == 'S' else 'U' for k, s in enumerate(states)]
    xyz, K, _, measurements, _, image_hw = V3._inputs(xyz, K, structural_states,
        visible, [], image_hw, PART._helper_config(cfg))
    original = {V3._id(r): r for r in visible}
    for m in measurements:
        source = original[m['native_corner']]; state = states[m['native_corner']]
        observed = m['covariance_px2'].copy()
        covariance = V3._finite_array(source.get('fit_covariance_px2',
            cfg['covariance_scale'] * observed + np.asarray(cfg['state_BI_total_error_floor_px2'][state])),
            (2, 2), 'candidate effective fit covariance')
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError('Candidate fit covariance must be symmetric')
        whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        weight = float(source.get('fit_reliability_weight', cfg['state_loss_weights'][state]))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('Candidate fit reliability must be finite and positive')
        m.update(observed_covariance_px2=observed, observed_whitening=m['whitening'].copy(),
            covariance_px2=covariance, whitening=whitening, fit_reliability_weight=weight,
            fit_covariance_source='OWN_CANDIDATE_OR_EXACT_OBSERVATION_CAL_OVERRIDE')
    parsed = PART._boundary_inputs(boundary, states, image_hw, xyz)
    PART.validate_complete_BI_likelihood_revocations(parsed, visible, cfg)
    return xyz, K, measurements, image_hw, parsed


def _image_jacobian(p, xyz, K, measurements, boundary, scale):
    partial_ids = {r['native_corner'] for r in boundary}
    complete = [m for m in measurements if m['native_corner'] not in partial_ids]
    point = J.visible_residual_and_jacobian(p, xyz, K, complete, scale)[1]
    if complete:
        point *= np.repeat(np.sqrt([m['fit_reliability_weight'] for m in complete]), 2)[:, None]
    normal = PART._normal_residual_and_jacobian(p, xyz, K, boundary, scale)[1]
    if boundary:
        normal *= np.repeat(np.sqrt([b['fit_reliability_weight'] for b in boundary]),
            [len(b['lines']) for b in boundary])[:, None]
    return np.vstack([point, normal])


def _family_solver(xyz, K, states, visible, domains, image_hw, coarse_pose,
                   initializations=None, config=None, *, boundary_observations=None, u_regions=None):
    """One SLSQP call using existing objective, gradient and guard primitives."""
    if domains or u_regions or initializations:
        raise ValueError('Common visible adapter accepts no region terms or extra starts')
    cfg = PART._configuration(config)
    xyz, K, measurements, image_hw, boundary = _prepare(xyz, K, states, visible,
        image_hw, boundary_observations or [], cfg)
    R0 = RT._rotation(coarse_pose['R'], 'N0 family R')
    t0 = RT._array(coarse_pose['t'], (3,), 'N0 family t')
    Q = RT._rotation(coarse_pose['physical_registration_Q'], 'family Q')
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    p0 = V3._pose_vector(R0, t0, scale)
    deviations = np.r_[np.repeat(np.deg2rad(cfg['prior_rotation_sigma_deg']), 3),
        np.repeat(cfg['prior_ray_sigma'], 2), cfg['prior_log_depth_sigma']]
    covariance6 = np.diag(deviations**2) if cfg['prior_covariance6'] is None else np.asarray(cfg['prior_covariance6'])
    prior = dict(R=R0, center=R0 @ xyz.mean(axis=0) + t0,
        whitening=np.linalg.inv(np.linalg.cholesky(covariance6)))
    rank0 = V3._rank(_image_jacobian(p0, xyz, K, measurements, boundary, scale))
    base = dict(schema=SCHEMA + '_FAMILY', accepted=False, status='unresolved', R=None, t=None,
        R_physical=None, objective=None, rank=dict(image=rank0), candidates=[],
        optimizer_calls=0, fallback_reason=None, boundary_support_intervals_are_archive_only=True,
        removed_complete_BI_likelihood_ids=sorted(r['native_corner'] for r in boundary),
        complete_point_count=len(measurements) - len(boundary), partial_normal_count=sum(len(b['lines']) for b in boundary),
        initial_N0_R=R0.tolist(), initial_N0_t=t0.tolist(), automatic_backend_states=list(states),
        ambiguity=dict(detected=rank0['rank'] < 6, global_uniqueness_proved=False))
    if rank0['rank'] < 6:
        base.update(status='empty_observation' if not measurements else 'rank_deficient',
            fallback_reason='no_image_information' if not measurements else 'image_information_rank_below_6')
        return base
    objective = V3._cached_value_jacobian(lambda p: PART._joint_objective_pair(
        p, xyz, K, measurements, scale, prior, cfg, boundary, 1.))
    guards = V3._cached_value_jacobian(lambda p: PART.BASE._constraints(
        p, xyz, K, measurements, [], (), image_hw, scale, cfg))
    base['optimizer_calls'] = 1
    try:
        answer = minimize(lambda p: objective(p)[0], p0, method='SLSQP',
            jac=lambda p: objective(p)[1].copy(),
            constraints=[dict(type='ineq', fun=lambda p: guards(p)[0].copy(),
                jac=lambda p: guards(p)[1].copy())],
            options=dict(maxiter=int(cfg['max_joint_iterations']), ftol=cfg['optimizer_ftol'], disp=False))
        R, t = V3._pose(answer.x, scale); projection, depth = V3.project_native(xyz, R, t, K)
        values, _ = guards(answer.x); value, _, components = objective(answer.x)
        finite = bool(np.isfinite(answer.x).all() and np.isfinite(value) and
            np.isfinite(projection).all() and np.isfinite(values).all())
        feasible = bool(finite and V3._valid_rotation(R) and
            np.min(depth) >= cfg['positive_depth_m'] - 1e-10 and
            np.min(values) >= -cfg['feasibility_tolerance'])
        rank = V3._rank(_image_jacobian(answer.x, xyz, K, measurements, boundary, scale)) if finite else rank0
        record = dict(optimizer_success=bool(answer.success), optimizer_message=str(answer.message),
            optimizer_status=int(answer.status), iterations=int(answer.nit), objective=float(value) if np.isfinite(value) else None,
            finite=finite, existing_guards_pass=feasible,
            minimum_constraint_slack=float(np.min(values)) if finite else None, rank=rank)
        base['candidates'] = [record]; base['rank'] = dict(image=rank, initial_image=rank0)
        if not finite:
            base.update(status='numerical_failure', fallback_reason='nonfinite_optimizer_result')
        elif not answer.success:
            base.update(status='optimizer_failed', fallback_reason='optimizer_did_not_converge')
        elif not feasible:
            base.update(status='algorithm_rejected', fallback_reason='existing_geometry_canvas_or_depth_guard')
        elif rank['rank'] < 6:
            base.update(status='rank_deficient', fallback_reason='final_image_information_rank_below_6')
        else:
            audit = PART.BASE._visible_audit(projection, measurements, -R.T @ t, image_hw, cfg)
            partial_ids = {r['native_corner'] for r in boundary}
            complete_audit = [a for a in audit if a['native_corner'] not in partial_ids]
            archive_audit = [dict(a, archived_original_only=True, full_2D_likelihood_active=False,
                measurement_Mahalanobis_is_constraint=False, archived_xy_used_as_seed_or_prior=False)
                for a in audit if a['native_corner'] in partial_ids]
            base.update(accepted=True, status='joint_feasible', R=R.tolist(), t=t.tolist(),
                R_physical=(R @ Q).tolist(), projected_native=projection.tolist(), objective=float(value),
                objective_components=components,
                ambiguity=dict(detected=False, global_uniqueness_proved=False),
                visible_audit=complete_audit, partial_original_archive_audit=archive_audit)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        base.update(status='numerical_failure', fallback_reason=type(exc).__name__ + ': ' + str(exc))
    return base


def solve_frame(frame, selection):
    """Recalculate an actual pose from actual selected observations.

    Invalid observations retain an explicit failure and the original N0 pose.
    Invalid N0 / geometry raises: a fabricated successful fallback is forbidden.
    """
    start = time.perf_counter()
    coarse = deepcopy(frame['initial_pose'])
    geometry = RT.registered_geometry(frame['raw_hw'], frame['K'], frame['native_xyz'], coarse)
    fallback = deepcopy(coarse); fallback['source'] = 'ORIGINAL_N0_FALLBACK'
    output = dict(schema=SCHEMA, accepted=False, fallback=True, status='unresolved',
        actual_pose=fallback, observed_states_preserved=deepcopy(selection['states']),
        solver_configuration=deepcopy(SOLVER_CONFIG), policy=deepcopy(POLICY),
        selected_observations=deepcopy(selection), solution=None, optimizer_calls=0,
        GT_inputs=False, old_pose_bank_inputs=False, physical_endpoint_constraints=0)
    try:
        states = [V3._state(s) for s in selection['states']]
        if len(states) != 8:
            raise ValueError('Eight fixed automatic states required')
        if any(states[k] != 'S' for k in np.flatnonzero(geometry['initial_self_hidden'])):
            raise ValueError('Initial N0 S gate must remain fixed')
        visible = deepcopy(selection['full_points']); boundary = deepcopy(selection.get('partial_lines', []))
        ids = [V3._id(r) for r in visible]
        if len(ids) != len(set(ids)) or any(states[k] not in ('B', 'I') for k in ids):
            raise ValueError('Unique automatic B/I observation IDs required')
        for record in visible:
            k = V3._id(record)
            if record.get('state', states[k]) != states[k]:
                raise ValueError('Observation state must equal immutable automatic backend state')
        partial_ids = [V3._id(r) for r in boundary]
        if len(partial_ids) != len(set(partial_ids)) or not set(partial_ids) <= set(ids):
            raise ValueError('Partial requires one unique original visible archive')
        # Validate every submitted covariance before suppression so malformed
        # correlated inputs cannot disappear silently behind de-duplication.
        _prepare(frame['native_xyz'], frame['K'], states, visible, frame['raw_hw'], boundary, SOLVER_CONFIG)
        visible, boundary, dedup = _deduplicate(visible, boundary)
        output['source_deduplication'] = dedup
        output['effective_full_points'] = deepcopy(visible)
        output['effective_partial_lines'] = deepcopy(boundary)
        output['observation_digest'] = observation_digest(dict(states=states, full_points=visible, partial_lines=boundary))
        registered = dict(R=coarse['R_cf'], t=coarse['centroid'], R_physical=coarse['R_physical'],
            physical_registration_Q=geometry['physical_registration_Q'].tolist())
        packet = WD.solve_two_wd_families(frame['native_xyz'], frame['K'], states, visible, [],
            frame['raw_hw'], registered, config=deepcopy(SOLVER_CONFIG),
            boundary_observations=boundary, u_regions=[], solver_callable=_family_solver)
        output['solution'] = packet
        output['optimizer_calls'] = sum(f['solution']['optimizer_calls'] for f in packet['families'])
        output['family_statuses'] = {f['name']: f['solution']['status'] for f in packet['families']}
        solution = packet['solution']; output['status'] = solution['status']
        if packet['accepted']:
            model = packet['selected_model_geometry']
            output.update(accepted=True, fallback=False, actual_pose=dict(available=True,
                R_cf=solution['R'], R_physical=solution['R_physical'], centroid=solution['t'],
                cf_extents=model['cf_extents'], source='FRESH_COMMON_POINT_PARTIAL_NORMAL_PNP',
                family=packet['selected_family'], physical_registration_Q=model['physical_registration_Q']))
        else:
            output['fallback_reason'] = solution['fallback_reason']
    except (ValueError, np.linalg.LinAlgError) as exc:
        output.update(status='invalid_observation', fallback_reason=type(exc).__name__ + ': ' + str(exc))
    output['model_projection_xy'] = V3.project_native(RT.C.cuboid(*output['actual_pose']['cf_extents']),
        np.asarray(output['actual_pose']['R_cf']), np.asarray(output['actual_pose']['centroid']), np.asarray(frame['K']))[0].tolist()
    output['elapsed_seconds'] = time.perf_counter() - start
    return _json(output)
