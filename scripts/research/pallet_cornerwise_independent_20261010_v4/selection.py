"""Select actual boundary coordinates using a corner-held-out N3 pose.

The model, query decoder, source calibration and v2 geometric admission remain
unchanged. A held-out pose is a validation prediction, never a new observation.
Its numeric pose solve excludes the candidate corner and actual initial self-hidden
IDs with no initial pose/projection/dimension prior. H and proposal features still
come from the original fixed estimators: this is conditional numeric independence,
not complete statistical independence or physical ownership.

No truth, frame scores, images, model, calibration fitting or tuning interface
is available here. Final self-hidden reprojection belongs to the final solver
and driver, after selection; its projections must not be re-fit.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import observation_points
from .pose import PoseBank


POLICY = dict(
    schema='observation_only_cornerwise_selection_policy_v4',
    candidate_admission='unchanged v2 final-line support and corner uncertainty cap8px',
    native_basin='unchanged v2 distance<=8px for an available native N3 coordinate',
    missing_native_exception='admitted actual boundary may be used if LOO supports it; native-distance comparison unavailable',
    heldout_fit='actual initial H union candidate corner; one robust solve per eligible corner',
    heldout_is_self_occlusion=False,
    candidate_residual_cap_px=8.0,
    squared_residual_tie_px2=1e-8,
    selection='LOO new pose, excluded k/H, candidate residual<=8px and native squared residual minus candidate squared residual>1e-8',
    ties_or_unavailable_or_ambiguous='preserve native N3 RGB coordinate; never fail the frame for this selection',
    coordinate='accepted actual boundary intersection, never LOO projected coordinate',
    initial_prior='none: validation pose dimensions/ranking/fit use retained observations only',
    conditional_mask='H comes from unchanged initial N3, fixed during each heldout solve',
    proposal_features='fixed Base-derived query and role features; not independent physical evidence',
    fully_independent_validation=False,
    same_N3_bank_reused_across_heldouts=True,
        conditional_H_inherited_from_initial_N3=True,
        proposal_Base_feature_dependence_retained=True,
    maximum_heldout_solves=8,
    new_threshold_fitting=False,
    truth_input=False,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _native_available(point):
    return bool(np.isfinite(point).all() and not np.all(point == -1))


def _counts(bank):
    return {} if bank is None else copy.deepcopy(bank.ledger)


def _same_bank(bank, native, K, xyz, image_size):
    """A reused bank must describe these exact immutable N3 observations."""
    _require(isinstance(bank, PoseBank), 'Expected the observation-only v4 PoseBank')
    expected = hashlib.sha256(b''.join(a.tobytes() for a in (native, K, xyz)) +
                              str(image_size).encode()).hexdigest()
    _require(bank.digest == expected, 'LOO bank coordinates/K/dimensions/image-size differ')
    _require(np.array_equal(bank.points, native, equal_nan=True), 'LOO bank is not native N3')


def select_corners(native_n3, observation, hidden, K, xyz,
                   image_size, *, bank=None):
    """Return selected observation coordinates and an explicit selection ledger.

    ``native_n3`` is [9,2] in the original image; center8 is preserved. ``xyz``
    is physical (width,height,depth) in meters, image_size=(width,height).
    A supplied bank is the observation-only v4 PoseBank for these N3 coordinates.
    All available accepted coordinates are real decoded boundary intersections.
    H coordinates are unchanged here and are excluded/reprojected by final fit.
    """
    native = np.asarray(native_n3, dtype=np.float64).copy()
    camera = np.asarray(K, dtype=np.float64)
    dimensions = np.asarray(xyz, dtype=np.float64)
    size = tuple(image_size)
    _require(native.shape == (9, 2), 'Expected native N3 corners and center [9,2]')
    _require(camera.shape == (3, 3) and np.isfinite(camera).all(), 'Invalid native camera matrix')
    _require(dimensions.shape == (3,) and np.isfinite(dimensions).all() and
             (dimensions > 0).all(), 'Invalid physical width/height/depth')
    _require(len(size) == 2 and all(np.isfinite(v) and v > 0 for v in size), 'Invalid image size')
    H = tuple(sorted(set(int(k) for k in hidden)))
    _require(all(0 <= k < 8 for k in H), 'Only corner IDs0..7 may be self-hidden')
    if bank is not None:
        _same_bank(bank, native, camera, dimensions, size)

    sparse, _, admitted = observation_points(native, observation, H)
    contract = copy.deepcopy(admitted)
    contract['v2_admission_before_cornerwise_selection'] = copy.deepcopy(admitted)
    before = _counts(bank)
    selected = native.copy()
    records = []
    accepted = []
    solves = 0
    v2 = {r['id']: r for r in admitted['per_validated_corner']}

    for k in admitted['validated_boundary_corner_ids']:
        candidate = sparse[k].copy()
        available_native = _native_available(native[k])
        distance = float(np.linalg.norm(candidate - native[k])) if available_native else None
        record = dict(id=int(k), actual_self_hidden_ids=list(H),
            heldout_corner_id=int(k), heldout_is_self_occlusion=False,
            candidate_xy=candidate.tolist(), native_N3_xy=native[k].tolist(),
            native_N3_available=available_native, native_distance_px=distance,
            native_distance_check='available' if available_native else 'unavailable',
            v2_admission=copy.deepcopy(v2[k]), heldout_fit_requested_excluded_ids=sorted(set(H) | {k}),
            heldout_pose_calls=0, fit_input_ids=[], final_inlier_ids=[],
            candidate_LOO_residual_px=None, native_N3_LOO_residual_px=None,
            native_minus_candidate_squared_residual_px2=None,
            accepted=False, missing_native_boundary_used=False,
            initial_prior_includes_heldout_influence=False,
            projected_coordinate_used_as_observation=False,
            numeric_validation_pose_prior_used=False,
            reason='INITIAL_SELF_HIDDEN' if k in H else
                   'OUTSIDE_FIXED8PX_NATIVE_BASIN' if available_native and distance > 8.0 else
                   'PENDING_HELDOUT_FIT')
        records.append(record)
        if record['reason'] != 'PENDING_HELDOUT_FIT':
            continue
        if bank is None:
            bank = PoseBank(native, camera, dimensions, image_size=size)
            _same_bank(bank, native, camera, dimensions, size)
        # H is the actual mask and k is temporary exclusion. Both are absent
        # from fit/branch selection; their diagnostic projections are never
        # observations, including the corner currently withheld.
        solved = bank.solve(excluded=(k,), hidden=H, robust=True)
        solves += 1
        record.update(heldout_pose_calls=1, loo_solver=copy.deepcopy(solved),
                      loo_solver_temporary_exclusion_is_not_self_hidden=True,
                      loo_state=solved.get('state'), loo_reason=solved.get('reason'),
                      fit_input_ids=list(solved.get('fit_input_ids', [])),
                      final_inlier_ids=list(solved.get('final_inliers', [])),
                      numeric_validation_pose_prior_used=bool(solved.get('prior_used',False)),
                      operation_counts=copy.deepcopy(solved.get('operation_counts', {})))
        exclusion_ok = (not set(record['fit_input_ids']) & (set(H) | {k}) and not solved.get('prior_used',False))
        record['fit_exclusion_verified'] = exclusion_ok
        if not exclusion_ok:
            record['reason'] = 'HELDOUT_FIT_EXCLUSION_VIOLATION'
            continue
        if solved.get('unresolved_ambiguity') or solved.get('state') == 'AMBIGUOUS_PNP':
            record['reason'] = 'HELDOUT_POSE_AMBIGUOUS'
            continue
        if not solved.get('available') or solved.get('state') != 'NEW_POSE':
            record['reason'] = 'HELDOUT_POSE_UNAVAILABLE'
            continue
        projected = np.asarray(solved.get('projected'), dtype=np.float64)
        if projected.shape != (8, 2) or not np.isfinite(projected[k]).all():
            record['reason'] = 'HELDOUT_PROJECTION_UNAVAILABLE'
            continue
        candidate_residual = float(np.linalg.norm(candidate - projected[k]))
        native_residual = float(np.linalg.norm(native[k] - projected[k])) if available_native else None
        gain = native_residual ** 2 - candidate_residual ** 2 if available_native else None
        record.update(heldout_predicted_corner_xy=projected[k].tolist(),
                      candidate_LOO_residual_px=candidate_residual,
                      native_N3_LOO_residual_px=native_residual,
                      native_minus_candidate_squared_residual_px2=gain)
        if candidate_residual > POLICY['candidate_residual_cap_px']:
            record['reason'] = 'BOUNDARY_NOT_SUPPORTED_BY_HELDOUT_POSE'
        elif available_native and abs(gain) <= POLICY['squared_residual_tie_px2']:
            record['reason'] = 'NUMERICAL_RESIDUAL_TIE_PRESERVE_NATIVE_N3'
        elif available_native and gain <= POLICY['squared_residual_tie_px2']:
            record['reason'] = 'HELDOUT_POSE_PREFERS_NATIVE_N3'
        else:
            selected[k] = candidate
            accepted.append(int(k))
            record.update(accepted=True, missing_native_boundary_used=not available_native,
                          reason='SUPPORTED_ACTUAL_BOUNDARY_WITH_MISSING_NATIVE_N3' if not available_native
                          else 'HELDOUT_POSE_PREFERS_ACTUAL_BOUNDARY')

    after = _counts(bank)
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(before) | set(after))}
    _require(solves <= POLICY['maximum_heldout_solves'], 'More than one solve per held-out corner')
    _require(np.array_equal(selected[8], native[8], equal_nan=True), 'Detector center changed')
    _require(np.array_equal(selected[list(H)], native[list(H)], equal_nan=True), 'Self-hidden input was replaced before final fit')
    accepted_set = set(accepted)
    per_corner = {r['id']: r for r in records}
    for entry in contract['per_validated_corner']:
        decision = per_corner[entry['id']]
        entry.update(selected_for_hybrid=decision['accepted'], reason=decision['reason'],
                     cornerwise_selection=copy.deepcopy(decision))
    contract.update(schema='actual_observation_only_cornerwise_contract_v4',
        cornerwise_policy=copy.deepcopy(POLICY), cornerwise_records=records,
        hybrid_boundary_corner_ids=accepted,
        hybrid_rejected_boundary_corner_ids=[k for k in admitted['validated_boundary_corner_ids'] if k not in accepted_set],
        native_N3_RGB_corner_ids=[k for k in range(8) if k not in accepted_set],
        cornerwise_boundary_corner_ids=accepted,
        missing_native_boundary_ids=[r['id'] for r in records if r['missing_native_boundary_used']],
        heldout_fit_exclusion_only_ids=[r['id'] for r in records if r['heldout_pose_calls']],
        actual_self_hidden_ids=list(H), heldout_is_self_occlusion=False,
        LOO_initial_prior_includes_heldout_influence=False,
        fully_independent_validation=False, heldout_projection_is_observation=False,
        selected_coordinate_source='actual decoded boundary or native N3 RGB; no LOO projections',
        maximum_heldout_solves=8, heldout_pose_calls=solves,
        operation_counts=delta, hypothesis_bank_counts=after,
        selection_bank_before_counts=before, selection_bank_after_counts=after,
        same_N3_bank_reused_across_heldouts=True,
        conditional_H_inherited_from_initial_N3=True,
        proposal_Base_feature_dependence_retained=True,
        selection_does_not_fail_frame_for_mask_disagreement=True,
        final_self_hidden_reprojection_deferred=True,
        final_reprojection_must_not_be_refit=True)
    contract['cornerwise_selection'] = dict(
        schema='actual_observation_only_cornerwise_summary_v4',
        LOO_solve_calls=solves,
        per_corner_decisions=[copy.deepcopy({key: value for key, value in record.items()
                                            if key != 'loo_solver'}) for record in records],
        full_LOO_solver_records_location='observation_contract.cornerwise_records',
        selected_boundary_corner_ids=list(accepted),
        actual_self_hidden_ids=list(H),
        heldout_fit_exclusion_only_ids=list(contract['heldout_fit_exclusion_only_ids']),
        missing_native_boundary_ids=list(contract['missing_native_boundary_ids']),
        operation_counts=copy.deepcopy(delta),
        hypothesis_bank_counts=copy.deepcopy(after),
        selection_bank_before_counts=copy.deepcopy(before),
        selection_bank_after_counts=copy.deepcopy(after),
        same_N3_bank_reused_across_heldouts=True,
        conditional_H_inherited_from_initial_N3=True,
        proposal_Base_feature_dependence_retained=True,
        initial_prior_includes_heldout_influence=False,
        fully_independent_validation=False,
        heldout_is_self_occlusion=False,
        heldout_projection_is_observation=False,
        final_self_hidden_reprojection_deferred=True,
        final_reprojection_must_not_be_refit=True,
        policy=copy.deepcopy(POLICY))
    return selected, contract
