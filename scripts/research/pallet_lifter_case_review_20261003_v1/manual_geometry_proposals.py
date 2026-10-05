"""Proposal-only visibility assistance from this frame's actual manual clicks.

No evaluated prediction, previous frame, repeated annotation or model is read.
The fixed cuboid is an approximation: a proposed hidden corner still requires
the person's C confirmation.  This module never fills reference coordinates or
labels, and deliberately never proposes a visible corner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from scripts.annotate import annotate_pnp as pnp
from scripts.research.pallet_static_registry_review_20261003_v1 import geometry_visibility_proposals as visibility_geometry


FIXED_DIMS_WDH = (1.1, 1.1, 0.15)
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _result(reason, **values):
    result = dict(proposals={}, projected=[], projected_by_candidate=[], candidates=[],
                  candidate_count=0, reason=reason, helper_sha256=_hash(__file__),
                  source='current_frame_confirmed_manual_clicks_only',
                  human_confirmation_required=True, independent_reference_claim=False,
                  input_modified=False, model_predictions_read=False)
    result.update(values)
    return result


def _collect_candidates(kps, K, points3d, size):
    """Retain the existing solver and *both* valid planar IPPE branches.

    solve_pose_candidates returns one preferred pose for each W/D hypothesis;
    it does not retain the second planar branch.  Reuse the same IPPE operation
    already used by annotate_pnp, without picking the branch with lower error.
    """
    width, height = size
    candidates = []
    for pose in pnp.solve_pose_candidates(kps, K, dims=FIXED_DIMS_WDH,
            img_shape=(height, width), auto_swap_dims=False):
        candidates.append(dict(R=pose['R'], t=pose['t'], source='existing_annotate_pose_candidate'))
    ids = [i for i, xy in enumerate(kps[:8]) if xy is not None]
    obj = points3d[ids]
    if np.linalg.matrix_rank(obj - obj.mean(axis=0)) == 2:
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj, np.asarray([kps[i] for i in ids], dtype=np.float64), K, None,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            ok, rvecs, tvecs = False, [], []
        if ok:
            for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                rotation, _ = cv2.Rodrigues(rvec)
                candidates.append(dict(R=rotation, t=np.asarray(tvec).reshape(3),
                                       source='existing_planar_IPPE_branch_' + str(index)))
    return candidates


def _valid_candidate(candidate, kps, K, points3d, size):
    """Apply existing annotation geometry guards, with no new fit threshold."""
    try:
        rotation = np.asarray(candidate['R'], dtype=float)
        translation = np.asarray(candidate['t'], dtype=float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0.)):
        return None
    camera_points = points3d @ rotation.T + translation
    if (camera_points[:, 2] <= 0.).any():
        return None
    projection = np.asarray(pnp.project_3d(points3d, rotation, translation, K))
    if not np.isfinite(projection).all():
        return None
    lr, tb, fr, _, _ = pnp._eval_pair_invariants(
        rotation, translation, K, pnp.make_pallet_keypoints_3d(*FIXED_DIMS_WDH))
    if lr + tb + fr != 0 or pnp._eval_v8_tilt(rotation) < pnp.V8_TILT_HARD_THR:
        return None
    width, height = size
    # Existing annotate_pnp v9 projected footprint guard, not a tuned cutoff.
    if np.prod(np.ptp(projection, axis=0)) < 0.005 * width * height:
        return None
    pose = dict(projected_all=projection.tolist())
    ratio = pnp._projection_to_raw_area_ratio(pose, kps)
    if ratio is not None and ratio < 0.75:  # solve_pose_safe default contract.
        return None
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    hidden = visibility_geometry._hidden(transform, np.asarray([1.1, .15, 1.1]))
    if hidden is None:
        return None
    ids = [i for i, xy in enumerate(kps[:8]) if xy is not None]
    errors = np.linalg.norm(projection[ids] - np.asarray([kps[i] for i in ids]), axis=1)
    return dict(R=rotation, t=translation, transform=transform, projected=projection,
                hidden=hidden, source=candidate.get('source', 'manual_pnp_candidate'),
                reprojection_mean_px=float(errors.mean()), reprojection_max_px=float(errors.max()),
                projection_to_raw_area_ratio=ratio)


def proposal_for_manual_record(record, contract, K, image_size):
    """Return SELF/OUT proposals, never labels, from four or more real clicks.

    ``image_size`` is the original (width,height), and ``K`` must be the frozen
    original-resolution camera intrinsics.  Previously filled slots, including
    uncertain and occlusion states, are untouched.  All returned coordinates
    are overlay-only projections; hidden reference x/y must remain null.
    """
    try:
        width, height = image_size
        if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
            return _result('invalid_image_size')
        matrix = np.asarray(K, dtype=float)
    except (TypeError, ValueError):
        return _result('invalid_intrinsics')
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0. or matrix[1, 1] <= 0.
            or not np.allclose(matrix[2], [0., 0., 1.], atol=1e-12, rtol=0.)
            or abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12
            or not (0. <= matrix[0, 2] < width and 0. <= matrix[1, 2] < height)):
        return _result('invalid_intrinsics')
    try:
        rows = contract['corners']
        if (len(rows) != 8 or {c['id'] for c in rows} != set(range(8))
                or any(type(c['id']) is not int for c in rows)):
            return _result('invalid_frozen_corner_contract')
        fixed = np.asarray([next(c['xyz_m'] for c in rows if c['id'] == i) for i in range(8)], dtype=float)
        expected = pnp.make_pallet_keypoints_3d(*FIXED_DIMS_WDH)[:8]
    except (KeyError, TypeError, ValueError, StopIteration):
        return _result('invalid_frozen_corner_contract')
    if fixed.shape != (8, 3) or not np.allclose(fixed, expected, atol=1e-12, rtol=0.):
        return _result('invalid_frozen_corner_contract')
    try:
        corners = record['corners']
        if (len(corners) != 8 or {c['id'] for c in corners} != set(range(8))
                or any(type(c['id']) is not int for c in corners)):
            return _result('invalid_manual_record')
        ordered = [next(c for c in corners if c['id'] == i) for i in range(8)]
    except (KeyError, TypeError, StopIteration):
        return _result('invalid_manual_record')
    kps = [None] * 9
    clicked_ids = []
    for index, corner in enumerate(ordered):
        if corner.get('visibility') != 'direct_visible':
            continue
        try:
            xy = np.asarray([corner.get('x'), corner.get('y')], dtype=float)
        except (TypeError, ValueError):
            return _result('invalid_direct_click')
        if (corner.get('definition_confirmed') is not True or not np.isfinite(xy).all()
                or any(corner.get(axis, False) for axis in AXES)
                or not (0. <= xy[0] < width and 0. <= xy[1] < height)):
            return _result('invalid_direct_click')
        kps[index] = xy.tolist()
        clicked_ids.append(index)
    evidence = dict(manual_clicked_ids=clicked_ids, manual_clicked_xy=[kps[i] for i in clicked_ids],
        actual_image_size=[width, height], intrinsics=matrix.tolist(),
        fixed_dims_width_depth_height_m=list(FIXED_DIMS_WDH),
        corner_contract_version=contract.get('version'),
        helper_sha256=_hash(__file__), annotation_pnp_sha256=_hash(pnp.__file__),
        visibility_geometry_sha256=_hash(visibility_geometry.__file__),
        prediction_input_used=False, hidden_reference_coordinates_created=False,
        solid_cuboid_approximation=True, human_confirmation_required=True,
        independent_reference_claim=False, reprojection_cutoff_px=None,
        ambiguity_rule='all geometrically valid planar branches; no reprojection ranking')
    if len(clicked_ids) < 4:
        return _result('need_four_confirmed_manual_clicks', evidence=evidence)
    image_points = np.asarray([kps[i] for i in clicked_ids])
    if (np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2
            or np.linalg.matrix_rank(fixed[clicked_ids] - fixed[clicked_ids].mean(axis=0)) < 2):
        return _result('degenerate_manual_clicks', evidence=evidence)
    if pnp._eval_click_lr_viol(kps) or pnp._eval_click_tb_viol(kps):
        return _result('manual_corner_order_conflict', evidence=evidence)
    try:
        raw = _collect_candidates(kps, matrix, fixed, (width, height))
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return _result('manual_pnp_failed', evidence=evidence)
    candidates = []
    for pose in raw:
        valid = _valid_candidate(pose, kps, matrix, fixed, (width, height))
        if valid is not None and not any(
                np.allclose(valid['R'], other['R'], atol=1e-7, rtol=0.)
                and np.allclose(valid['t'], other['t'], atol=1e-7, rtol=0.) for other in candidates):
            candidates.append(valid)
    if not candidates:
        return _result('no_geometrically_valid_manual_pose', evidence=evidence)
    details = [dict(source=c['source'], pose_transform=c['transform'].tolist(),
        reprojection_mean_px=c['reprojection_mean_px'], reprojection_max_px=c['reprojection_max_px'],
        projection_to_raw_area_ratio=c['projection_to_raw_area_ratio']) for c in candidates]
    evidence.update(candidate_count=len(candidates), candidate_pose_details=details,
                    primary_projection_is_overlay_only=True)
    proposals = {}
    for index, corner in enumerate(ordered):
        if corner.get('visibility') is not None:
            continue
        projections = [c['projected'][index] for c in candidates]
        outside = all(not (0. <= xy[0] < width and 0. <= xy[1] < height) for xy in projections)
        hidden = all(c['hidden'][index] for c in candidates)
        if not outside and not hidden:
            continue
        axis = 'out_of_frame' if outside else 'self_occlusion'
        proposals[index] = dict(axis=axis, projected_xy=projections[0].tolist(),
            evidence=dict(**evidence, corner_id=index,
                          projected_xy_by_candidate=[xy.tolist() for xy in projections],
                          consensus_axis=axis, machine_proposal_only=True))
    return _result('ok' if proposals else 'no_consensus_hidden_or_outside_proposals',
        proposals=proposals, projected=candidates[0]['projected'].tolist(),
        projected_by_candidate=[c['projected'].tolist() for c in candidates],
        candidate_count=len(candidates), candidates=details, pose_details=details,
        evidence=evidence)
