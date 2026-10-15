"""Fresh geometric BI PnP seeds, with no cached-pose/baseline/bank input.

Only state-admitted B/I xy enters unweighted SQPnP. Covariances, reliability
and source are archived unchanged for the downstream native objective; this
seed generator does not claim to use them as PnP weights. It enumerates the
full BI set and fixed 3/4-point subsets (at most127 calls for eight vertices),
retains all finite proper-rotation, all-vertex-front-facing solutions, and
deduplicates only numerically equal physical poses. It does not rank by any
reference, identity, partition, reprojection error or previous method pose.

The independently produced initial coarse pose is a separate, explicitly
attested source. Its independence must be established by the caller's input
binding; an attestation here cannot prove upstream ancestry. No files,
images, S/U/O/T measurements, models or references are fetched by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import combinations
import hashlib
import json

import cv2
import numpy as np


SCHEMA = 'FRESH_OBSERVED_BI_SQPNP_PROPOSALS_V32'
COARSE_SOURCE = 'INDEPENDENT_INITIAL_COARSE_PREDICTOR'
FRESH_SOURCE = 'FRESH_OBSERVED_BI_SQPNP_SUBSET'
MAXIMUM_SUBSET_CALLS = 127
ROTATION_EQUAL_ATOL = 1e-8
CENTER_EQUAL_RELATIVE_BODY_ATOL = 1e-8
STATES = frozenset(('B', 'I', 'O', 'T', 'U', 'S'))


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + ' requires finite shape ' + str(shape))
    return result.copy()


def _rotation(value, name):
    result = _array(value, (3, 3), name)
    if (not np.allclose(result.T @ result, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(result), 1., atol=1e-7, rtol=0)):
        raise ValueError(name + ' must be a proper rotation')
    return result


def _id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 8:
        raise ValueError('Strict native integer ID in0..7 required')
    return int(value)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _covariance(value, name):
    covariance = _array(value, (2, 2), name)
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError(name + ' must be symmetric')
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(name + ' must be positive definite') from error


def _measurements(records, states, initial_self_hidden):
    admitted = {}
    # Only the native ID may be read before the state/initial-S gate. In
    # particular, excluded record payloads are never copied or serialized.
    for record in records:
        k = _id(record['native_corner'])
        if initial_self_hidden[k] or states[k] not in ('B', 'I'):
            continue
        if k in admitted:
            raise ValueError('One immutable B/I measurement per native ID required')
        xy = _array(record['xy'], (2,), 'B/I xy')
        source = record['source']
        if not isinstance(source, str) or not source:
            raise ValueError('Explicit original B/I source required')
        archive = dict(native_corner=k, state=states[k], xy=xy.tolist(), source=source)
        covariance_count = 0
        for key in ('covariance_px2', 'fit_covariance_px2'):
            if key in record:
                _covariance(record[key], key)
                archive[key] = _plain(deepcopy(record[key]))
                covariance_count += 1
        if not covariance_count:
            raise ValueError('Original B/I covariance archive required')
        if 'fit_reliability_weight' in record:
            weight = record['fit_reliability_weight']
            if isinstance(weight, (bool, np.bool_)) or not np.isfinite(weight) or weight <= 0:
                raise ValueError('Positive original B/I reliability required')
            archive['fit_reliability_weight'] = _plain(deepcopy(weight))
        admitted[k] = archive
    return [admitted[k] for k in sorted(admitted)]


def _rank_at_least_two(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return len(singular) >= 2 and singular[1] > 1e-10 * max(float(singular[0]), 1e-12)


def _subsets(ids):
    if len(ids) < 3:
        return []
    full = tuple(ids)
    result = [full]
    for count in (3, 4):
        if count <= len(ids):
            result.extend(group for group in combinations(ids, count) if group != full)
    if len(result) > MAXIMUM_SUBSET_CALLS or len(set(result)) != len(result):
        raise AssertionError('Fixed unique subset budget exceeded')
    return result


def _proposal(R, t, xyz, Q, name, source, provenance):
    R = _rotation(R, source + ' R_native')
    t = _array(t, (3,), source + ' t_native')
    camera = xyz @ R.T + t
    if np.any(camera[:, 2] <= 1e-9 * max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)):
        raise ValueError('Every original 3D vertex must be in front of the camera')
    center = R @ xyz.mean(axis=0) + t
    return dict(name=name, source=source, R=R.tolist(), t=t.tolist(),
        R_physical=(R @ Q).tolist(), centroid=center.tolist(),
        physical_registration_Q=Q.tolist(), native_mean=xyz.mean(axis=0).tolist(),
        centroid_is_native_t=bool(np.array_equal(xyz.mean(axis=0), np.zeros(3))),
        all_native_vertices_positive_depth=True, minimum_native_depth_m=float(camera[:, 2].min()),
        covariance_weighted_PnP=False, provenance=provenance, duplicate_generation_provenance=[])


def generate_fresh_observed_BI_pnp_proposals(native_xyz, K, states, initial_self_hidden,
        visible_measurements, initial_coarse_pose, physical_registration_Q):
    """Return independent coarse plus fresh SQPnP seeds, never cached poses.

    Coarse requires native-basis ``R``/``t``, ``source=COARSE_SOURCE`` and
    ``independent_of_cached_pose_bank=True``. Q is explicit and is validated
    against optional coarse Q/R_physical. All returned candidates still need
    the unchanged downstream native objective and full current hard guards.
    """
    xyz = _array(native_xyz, (8, 3), 'native_xyz')
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-10) < 3:
        raise ValueError('Nondegenerate registered 3D body required')
    camera = _array(K, (3, 3), 'K')
    if (camera[2, 2] <= 0 or not np.allclose(camera[2, :2], 0., atol=1e-12, rtol=0)):
        raise ValueError('Finite pinhole K with positive homogeneous scale required')
    camera /= camera[2, 2]
    if camera[0, 0] <= 0 or camera[1, 1] <= 0 or abs(np.linalg.det(camera)) <= 1e-12:
        raise ValueError('Invertible positive-focal pinhole K required')
    Q = _rotation(physical_registration_Q, 'physical_registration_Q')
    if len(states) != 8 or any(not isinstance(s, str) or s not in STATES for s in states):
        raise ValueError('Eight exact native states required')
    states = list(states)
    if (len(initial_self_hidden) != 8
            or any(not isinstance(v, (bool, np.bool_)) for v in initial_self_hidden)):
        raise ValueError('Eight original boolean self-hidden gates required')
    hidden = [bool(v) for v in initial_self_hidden]
    if (not isinstance(initial_coarse_pose, Mapping)
            or initial_coarse_pose.get('source') != COARSE_SOURCE
            or initial_coarse_pose.get('independent_of_cached_pose_bank') is not True):
        raise ValueError('Explicit caller-attested independent initial coarse source required')
    R0 = _rotation(initial_coarse_pose['R'], 'initial coarse R')
    t0 = _array(initial_coarse_pose['t'], (3,), 'initial coarse native t')
    if ('physical_registration_Q' in initial_coarse_pose and not np.allclose(
            _rotation(initial_coarse_pose['physical_registration_Q'], 'coarse Q'), Q, atol=1e-12, rtol=0)):
        raise ValueError('Initial coarse and requested physical Q must agree')
    if ('R_physical' in initial_coarse_pose and not np.allclose(
            _rotation(initial_coarse_pose['R_physical'], 'coarse physical R'), R0 @ Q, atol=1e-7, rtol=0)):
        raise ValueError('Initial coarse physical R must equal R_native @ Q')
    measurements = _measurements(visible_measurements, states, hidden)
    source_input = dict(native_xyz=xyz.tolist(), K=camera.tolist(), states=states,
        initial_self_hidden=hidden, admitted_BI=measurements,
        initial_coarse_R=R0.tolist(), initial_coarse_t=t0.tolist(), physical_Q=Q.tolist())
    binding = _digest(source_input)
    coarse = _proposal(R0, t0, xyz, Q, 'OWN_INDEPENDENT_INITIAL_COARSE', COARSE_SOURCE,
        dict(input_binding_sha256=binding, caller_attestation_not_upstream_ancestry_proof=True))
    proposals = [coarse]
    by_id = {m['native_corner']: m for m in measurements}
    ids = list(by_id)
    subsets = _subsets(ids)
    rejected, calls, fresh_outputs = [], 0, 0
    inverse_K = np.linalg.inv(camera)
    body_scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    for subset in subsets:
        image = np.asarray([by_id[k]['xy'] for k in subset], dtype=np.float64)
        rays = np.column_stack((image, np.ones(len(image)))) @ inverse_K.T
        normalized = rays[:, :2] / rays[:, 2, None]
        object_points = np.ascontiguousarray(xyz[list(subset)])
        normalized = np.ascontiguousarray(normalized)
        if not _rank_at_least_two(object_points) or not _rank_at_least_two(normalized):
            rejected.append(dict(subset_native_ids=list(subset), reason='structurally_degenerate_subset'))
            continue
        calls += 1
        try:
            solved = cv2.solvePnPGeneric(object_points, normalized, np.eye(3), None,
                flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error as error:
            rejected.append(dict(subset_native_ids=list(subset), reason='SQPNP_failed', detail=str(error)))
            continue
        if not solved[0] or len(solved[1]) != len(solved[2]):
            rejected.append(dict(subset_native_ids=list(subset), reason='SQPNP_no_solution'))
            continue
        for index, (rotvec, translation) in enumerate(zip(solved[1], solved[2])):
            provenance = dict(input_binding_sha256=binding, subset_native_ids=list(subset),
                SQPNP_solution_index=index, normalized_full_K=True, extrinsic_guess_used=False)
            try:
                rotvec = _array(np.asarray(rotvec).reshape(-1), (3,), 'SQPNP rotvec')
                R = cv2.Rodrigues(rotvec)[0]
                t = _array(np.asarray(translation).reshape(-1), (3,), 'SQPNP native t')
                candidate = _proposal(R, t, xyz, Q, 'FRESH_SQPNP_' + str(fresh_outputs), FRESH_SOURCE, provenance)
            except (ValueError, cv2.error) as error:
                rejected.append(dict(**provenance, reason='invalid_rotation_translation_or_cheirality', detail=str(error)))
                continue
            fresh_outputs += 1
            duplicate = next((p for p in proposals if np.allclose(p['R_physical'], candidate['R_physical'],
                atol=ROTATION_EQUAL_ATOL, rtol=0) and np.allclose(p['centroid'], candidate['centroid'],
                atol=CENTER_EQUAL_RELATIVE_BODY_ATOL * body_scale, rtol=0)), None)
            if duplicate is None:
                proposals.append(candidate)
            else:
                duplicate['duplicate_generation_provenance'].append(provenance)
    fresh_count = sum(p['source'] == FRESH_SOURCE for p in proposals)
    return dict(schema=SCHEMA, proposals=proposals, fresh_seed_count=fresh_count,
        valid_fresh_solver_outputs_before_numeric_dedup=fresh_outputs, admitted_BI_ids=ids,
        admitted_BI_archive=measurements, input_binding_sha256=binding,
        subset_count=len(subsets), SQPNP_calls=calls, maximum_subset_calls=MAXIMUM_SUBSET_CALLS,
        rejected_subsets_or_solutions=rejected,
        abstained_from_fresh_seeds=fresh_count == 0,
        abstention_reason=('fewer_than_three_BI' if len(ids) < 3 else
            'no_distinct_fresh_valid_seed_after_structural_checks_or_dedup' if fresh_count == 0 else None),
        coarse_independence_is_caller_attestation_not_inferred=True,
        covariances_sources_and_reliabilities_archived_unchanged=True,
        covariance_weighted_PnP=False, all_current_hardguards_still_required=True,
        reference_error_ID_partition_or_cached_pose_ranking=False,
        new_S_U_O_T_measurement_payload_reads=0, new_images_models_files_or_reference_reads=0,
        downstream_diverse_start_selection_performed=False)
