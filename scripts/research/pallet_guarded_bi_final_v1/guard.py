"""Two fixed B/I replacement rules, with no fitting, image or reference IO.

Rule A requires the existing solver's feasible, locally identified proposal
and a real current-image correction. Rule B additionally verifies its model
edge projections against the same archived observed line samples. These are
consistency checks, not independent evidence or a correctness guarantee.
The caller must authenticate the original observation/source seal; strings
inside a dictionary cannot authenticate its bytes or target identity.
"""
from __future__ import annotations

from copy import deepcopy
import re

import numpy as np


SCHEMA = 'GUARDED_BI_FINAL_V1'
MODES = ('A', 'B')
MOVEMENT_ATOL_PX = 1e-8
COST_ATOL = 1e-9
HUBER_DELTA_PX = 1.25
MINIMUM_BI_COUNT = 4
_SOURCES = {
    'B_TARGET_BACKGROUND_LINES': ('B', 'OBSERVED_TARGET_BACKGROUND_TRANSITION'),
    'I_INTERNAL_RGB_JUNCTION': ('I', 'OBSERVED_INTERNAL_RGB_GRADIENT'),
    'B_ACTUAL_BOUNDARY_RECOVERY': ('B', 'OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION'),
}
_STATES = {'S', 'B', 'I', 'O', 'T', 'U'}


def _array(value, shape=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if shape is not None and result.shape != shape:
        return None
    return result if np.isfinite(result).all() else None


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _false(value):
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _rotation(value):
    R = _array(value, (3, 3))
    if R is None or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0):
        return None
    return R if np.isclose(np.linalg.det(R), 1., atol=1e-7, rtol=0) else None


def _cuboid(extents):
    a, b, c = np.asarray(extents) / 2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def _project(xyz, pose, K):
    camera = xyz @ np.asarray(pose['R_cf'], float).T + np.asarray(pose['centroid'], float)
    homogeneous = camera @ K.T
    if not np.isfinite(homogeneous).all() or np.any(camera[:, 2] <= 0) or np.any(homogeneous[:, 2] <= 0):
        return None
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _huber(distance):
    absolute = np.abs(distance)
    return np.where(absolute <= HUBER_DELTA_PX, .5 * absolute**2,
                    HUBER_DELTA_PX * (absolute - .5 * HUBER_DELTA_PX))


def _line_cost(points, weights, projected, corner, neighbor):
    direction = projected[neighbor] - projected[corner]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        return None
    normal = np.array([-direction[1], direction[0]]) / norm
    distances = (points - projected[corner]) @ normal
    numerator = float(weights @ _huber(distances))
    return dict(cost=numerator / float(weights.sum()), numerator=numerator,
                normal=normal.tolist(), perpendicular_distances_px=distances.tolist())


def apply_guard(observation, solved, coarse_pose, K, mode):
    """Select an unchanged own baseline or its own B/I proposal.

    Input dictionaries are immutable. A rejection deep-copies the supplied
    ``coarse_pose`` verbatim, including unavailable flags and all metadata.
    Guard ``accepted`` describes replacement, not the underlying solver flag.
    S/O/T/U never become observations. S may be the other model endpoint of
    an observed edge; its projected coordinate is never a measured 2D point.
    Every supported ray is evaluated on all stored observed samples, with
    the stored confidence weights. Unsupported/malformed/foreign sources
    cannot supply evidence. All their exclusion reasons remain in the audit.
    """
    if mode not in MODES:
        raise ValueError('mode must be exactly A or B')
    baseline = deepcopy(coarse_pose)
    audit = dict(schema=SCHEMA, thresholds=dict(minimum_unique_BI=MINIMUM_BI_COUNT,
        movement_atol_px=MOVEMENT_ATOL_PX, ray_cost_atol=COST_ATOL,
        total_strict_improvement_atol=COST_ATOL, huber_delta_px=HUBER_DELTA_PX),
        base_checks=[], measurement_records=[], ray_records=[],
        GT_inputs=False, fit_calls=0, image_queries=0, new_PnP_calls=0,
        same_observation_consistency_not_independent_evidence=True,
        model_projection_is_not_image_measurement=True,
        caller_must_verify_observation_and_source_seals=True,
        pose_projection_basis='CF_NATIVE_CUBOID_R_CF_CENTROID',
        physical_registration_Q_preserved=True)

    def check(name, passed, detail=None):
        row = dict(check=name, passed=bool(passed))
        if detail is not None:
            row['detail'] = detail
        audit['base_checks'].append(row)
        return bool(passed)

    def finish(accepted, reason):
        audit['replacement_selected'] = bool(accepted)
        audit['decision_reason'] = reason
        return dict(schema=SCHEMA, mode=mode, method='GUARDED_BI_' + mode,
                    status='guarded_BI_replacement' if accepted else 'exact_own_baseline_retained',
                    actual_pose=deepcopy(solved['actual_pose']) if accepted else baseline,
                    accepted=bool(accepted), fallback=not accepted, reason=reason,
                    decision_audit=audit)

    if not isinstance(observation, dict) or not isinstance(solved, dict) or not isinstance(coarse_pose, dict):
        check('dictionary_inputs', False)
        return finish(False, 'invalid_dictionary_input')
    states = observation.get('states')
    if not check('native_eight_states', isinstance(states, list) and len(states) == 8
                 and all(s in _STATES for s in states)):
        return finish(False, 'invalid_native_states')
    geometry = observation.get('initial_geometry') or {}
    correction = observation.get('visible_correction') or {}
    provenance = observation.get('provenance') or {}
    if not all(isinstance(v, dict) for v in (geometry, correction, provenance)):
        return finish(False, 'invalid_observation_metadata')
    cp = correction.get('provenance') or {}
    if not isinstance(cp, dict):
        return finish(False, 'invalid_correction_provenance')
    valid_provenance = (observation.get('schema') == 'STATEWISE_RGB_OBSERVATIONS_V4'
        and correction.get('schema') == 'visible_BI_measurements_boundary_recovery_v4_3b'
        and _false(provenance.get('GT_inputs'))
        and _true(provenance.get('current_RGB_state_inference'))
        and provenance.get('S_image_queries') == 0
        and _false(cp.get('S_queried'))
        and _false(cp.get('projected_coordinates_used_as_measurements'))
        and _true(cp.get('image_coordinates_frozen_before_joint_OT'))
        and isinstance(correction.get('image_pixel_sha256'), str)
        and re.fullmatch('[0-9a-f]{64}', correction['image_pixel_sha256']) is not None)
    if not check('current_RGB_observation_provenance', valid_provenance):
        return finish(False, 'invalid_current_RGB_observation_provenance')
    candidate = solved.get('actual_pose') or {}
    solution = solved.get('solution') or {}
    if not isinstance(candidate, dict) or not isinstance(solution, dict):
        return finish(False, 'invalid_solver_metadata')
    if not check('own_BI_solver_accepted', _true(solved.get('accepted')) and _true(solution.get('accepted'))):
        return finish(False, 'BI_solver_not_accepted')
    extents = _array(coarse_pose.get('cf_extents'), (3,))
    matrix = _array(K, (3, 3))
    R0, R1 = _rotation(coarse_pose.get('R_cf')), _rotation(candidate.get('R_cf'))
    Rp0, Rp1 = _rotation(coarse_pose.get('R_physical')), _rotation(candidate.get('R_physical'))
    t0, t1 = _array(coarse_pose.get('centroid'), (3,)), _array(candidate.get('centroid'), (3,))
    valid_geometry = (extents is not None and np.all(extents > 0) and matrix is not None
        and R0 is not None and R1 is not None and Rp0 is not None and Rp1 is not None
        and t0 is not None and t1 is not None and candidate.get('available', True) is not False
        and coarse_pose.get('available', True) is not False)
    if not check('finite_available_registered_poses', valid_geometry):
        return finish(False, 'invalid_registered_pose')
    xyz = _cuboid(extents)
    candidate_extents = _array(candidate.get('cf_extents'), (3,))
    observed_xyz = _array(geometry.get('native_xyz'), (8, 3))
    gR, gt = _array(geometry.get('R_cf'), (3, 3)), _array(geometry.get('centroid'), (3,))
    gQ = _array(geometry.get('physical_registration_Q'), (3, 3))
    gK = _array(geometry.get('K'), (3, 3))
    model_matches = (candidate_extents is not None and np.array_equal(candidate_extents, extents)
        and observed_xyz is not None and np.array_equal(observed_xyz, xyz)
        and gR is not None and np.array_equal(gR, R0)
        and gt is not None and np.array_equal(gt, t0)
        and gK is not None and np.array_equal(gK, matrix)
        and gQ is not None and np.allclose(gQ, R0.T @ Rp0, atol=1e-9, rtol=0)
        and np.allclose(R0.T @ Rp0, R1.T @ Rp1, atol=1e-9, rtol=0))
    if not check('own_native_geometry_and_registration', model_matches):
        return finish(False, 'foreign_geometry_or_changed_registration')
    original = _array(observation.get('original_raw_points'))
    measured = observation.get('frozen_visible_measurements')
    records = correction.get('records')
    if (original is None or original.shape not in ((8, 2), (9, 2))
            or not isinstance(measured, list) or not isinstance(records, list)):
        check('native_measurement_arrays', False)
        return finish(False, 'invalid_native_measurement_arrays')
    ids, by_record = [], {}
    for record in records:
        k = record.get('native_corner') if isinstance(record, dict) else None
        if not _integer(k) or not 0 <= k < 8 or k in by_record or record.get('state') != states[k]:
            check('record_native_state_binding', False)
            return finish(False, 'invalid_record_native_state_binding')
        by_record[int(k)] = record
    if set(by_record) != set(range(8)):
        check('record_native_state_binding', False)
        return finish(False, 'incomplete_native_records')
    moved = []
    for measurement in measured:
        k = measurement.get('native_corner') if isinstance(measurement, dict) else None
        if not _integer(k) or not 0 <= k < 8 or k in ids or states[k] not in ('B', 'I'):
            check('unique_BI_measurement_inputs', False)
            return finish(False, 'nonBI_duplicate_or_invalid_measurement')
        k = int(k)
        ids.append(k)
        xy = _array(measurement.get('xy'), (2,))
        cov = _array(measurement.get('covariance_px2'), (2, 2))
        valid = (measurement.get('state') == states[k] and xy is not None and cov is not None
            and np.allclose(cov, cov.T, atol=1e-9, rtol=0)
            and np.linalg.eigvalsh(cov).min() > 0
            and _true(measurement.get('image_measurement'))
            and _false(measurement.get('measurement_is_pose_projection')))
        if not valid:
            check('finite_authentic_BI_measurements', False)
            return finish(False, 'invalid_BI_measurement')
        record = by_record[k]
        source = measurement.get('source')
        spec = _SOURCES.get(source)
        raw_xy, measured_xy = _array(record.get('raw_xy'), (2,)), _array(record.get('measured_xy'), (2,))
        genuine = (spec is not None and spec[0] == states[k]
            and record.get('method') == source and _true(record.get('corrected'))
            and _true(record.get('pixel_queried')) and raw_xy is not None and measured_xy is not None
            and np.array_equal(raw_xy, original[k]) and np.array_equal(measured_xy, xy))
        movement = float(np.linalg.norm(xy - original[k]))
        actual_moved = bool(genuine and movement > MOVEMENT_ATOL_PX)
        audit['measurement_records'].append(dict(native_corner=k, state=states[k], source=source,
            genuine_current_RGB_correction=bool(genuine), movement_px=movement,
            moved=actual_moved, reason='genuine_moved_current_RGB_correction' if actual_moved else
            'genuine_correction_below_movement_tolerance' if genuine else 'not_genuine_current_RGB_correction'))
        if actual_moved:
            moved.append(k)
    check('unique_BI_measurement_inputs', set(ids) == {k for k, s in enumerate(states) if s in ('B', 'I')}, ids)
    check('solver_input_ids_identical', solved.get('input_native_ids') == ids)
    check('solver_observed_states_identical', solved.get('observed_states_preserved') == states)
    check('solver_frozen_measurements_identical', solved.get('frozen_visible_measurements') == measured)
    check('solver_excludes_hidden_2D_terms', solved.get('S_image_terms') == 0
          and _true(solved.get('O_T_U_are_not_PnP_points')))
    check('minimum_four_unique_BI', len(ids) >= MINIMUM_BI_COUNT, len(ids))
    rank = solution.get('rank') or {}
    visible_rank = rank.get('visible') if isinstance(rank, dict) else None
    check('visible_rank_six', isinstance(visible_rank, dict) and visible_rank.get('rank') == 6)
    ambiguity = solution.get('ambiguity') or {}
    for flag in ('detected', 'local_underdetermined', 'pose_family_from_two_BI'):
        check('ambiguity_' + flag + '_false', isinstance(ambiguity, dict) and _false(ambiguity.get(flag)))
    check('actual_moved_current_RGB_BI', bool(moved), moved)
    if not all(row['passed'] for row in audit['base_checks']):
        return finish(False, 'rule_A_requirements_failed')
    if mode == 'A':
        return finish(True, 'rule_A_requirements_passed')
    p0, p1 = _project(xyz, coarse_pose, matrix), _project(xyz, candidate, matrix)
    if p0 is None or p1 is None:
        return finish(False, 'invalid_positive_depth_projection_for_line_check')
    total0 = total1 = total_weight = 0.
    all_nonworsening = True
    for k in ids:
        record = by_record[k]
        method = record.get('method')
        spec = _SOURCES.get(method)
        rays = record.get('rays', [])
        if not isinstance(rays, list):
            audit['ray_records'].append(dict(native_corner=k, ray_index=None,
                status='excluded', reason='invalid_ray_collection'))
            continue
        for index, ray in enumerate(rays):
            row = dict(native_corner=k, ray_index=index, status='excluded')
            audit['ray_records'].append(row)
            if not isinstance(ray, dict):
                row['reason'] = 'invalid_ray_dictionary'
                continue
            row.update(native_neighbor=ray.get('native_neighbor'), source=ray.get('measurement_source'))
            if not _true(ray.get('supported')):
                row['reason'] = 'unsupported_ray'
                continue
            mate = ray.get('native_neighbor')
            source_matches = (spec is not None and spec[0] == states[k]
                and ray.get('measurement_source') == spec[1]
                and ray.get('guide_source') == 'initial_registered_model_direction_only'
                and _true(record.get('pixel_queried')))
            if not source_matches:
                row['reason'] = 'unrecognized_or_nonimage_line_provenance'
                continue
            if (not _integer(mate) or not 0 <= mate < 8 or mate == k
                    or ray.get('native_corner') != k or ray.get('endpoints') != [k, mate]
                    or ray.get('edge') != k * 8 + mate
                    or np.count_nonzero(xyz[k] != xyz[mate]) != 1):
                row['reason'] = 'invalid_native_incident_edge_binding'
                continue
            if method == 'B_ACTUAL_BOUNDARY_RECOVERY' and (
                    not _true(cp.get('recovery_RGB_support_required'))
                    or ray.get('shared_native_edge') != sorted([k, mate])
                    or not _integer(ray.get('rgb_supported_sample_count'))
                    or ray['rgb_supported_sample_count'] < 1):
                row['reason'] = 'missing_actual_RGB_recovery_support'
                continue
            points = _array(ray.get('observed_points'))
            weights = _array(ray.get('observation_confidence'))
            normal = _array(ray.get('normal'), (2,))
            offset = _array(ray.get('offset'), ())
            if (points is None or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or weights is None or weights.shape != (len(points),)
                    or np.any(weights < 0) or weights.sum() <= 0
                    or normal is None or np.linalg.norm(normal) <= 0 or offset is None):
                row['reason'] = 'invalid_finite_observed_line_samples_or_weights'
                continue
            old, new = _line_cost(points, weights, p0, k, mate), _line_cost(points, weights, p1, k, mate)
            if old is None or new is None:
                row['reason'] = 'degenerate_projected_native_edge'
                continue
            nonworse = new['cost'] <= old['cost'] + COST_ATOL
            row.update(status='valid', reason='same_archived_observed_ray_evaluated',
                sample_count=len(points), weight_sum=float(weights.sum()),
                baseline_cost=old['cost'], candidate_cost=new['cost'], nonworsening=bool(nonworse),
                baseline_model_normal=old['normal'], candidate_model_normal=new['normal'])
            all_nonworsening = all_nonworsening and nonworse
            total0 += old['numerator']
            total1 += new['numerator']
            total_weight += float(weights.sum())
    valid_count = sum(r['status'] == 'valid' for r in audit['ray_records'])
    mean0, mean1 = (total0 / total_weight, total1 / total_weight) if total_weight > 0 else (None, None)
    improves = bool(valid_count and mean0 - mean1 > COST_ATOL)
    audit['line_cost'] = dict(valid_ray_count=valid_count, weight_sum=total_weight,
        baseline_weighted_mean=mean0, candidate_weighted_mean=mean1,
        strict_improvement=improves, all_rays_nonworsening=bool(all_nonworsening),
        aggregation='sum_all_valid_ray_confidence_times_huber / sum_all_valid_ray_confidence',
        observations='all_archived_observed_points_not_only_fitted_inliers',
        infinite_model_lines=True, finite_segment_or_heldout_validation=False)
    if not valid_count:
        return finish(False, 'rule_B_no_valid_observed_rays')
    if not all_nonworsening:
        return finish(False, 'rule_B_observed_ray_worsened')
    if not improves:
        return finish(False, 'rule_B_no_strict_aggregate_improvement')
    return finish(True, 'rule_B_requirements_passed')
