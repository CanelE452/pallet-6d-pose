"""Read-only geometry suggestions for the native reference visibility viewer.

These are machine proposals for explicit human confirmation, never reviewed
labels or independent ground truth.  A finite reference outside the real image
is exact for that stored coordinate.  Self occlusion uses a *solid cuboid*
approximation, so it cannot establish visibility through pallet openings or
external occlusion.  It deliberately never proposes DIRECT_VISIBLE.
"""
from __future__ import annotations

import numpy as np


# The existing camera_dynamic_0123_v4 convention: near 0..3, far 4..7;
# top 0,1,4,5; bottom 2,3,6,7; left 0,3,4,7; right 1,2,5,6.
FACES = (
    ((0, 1, 2, 3), (0., 0., -1.)),
    ((4, 5, 6, 7), (0., 0., 1.)),
    ((0, 1, 4, 5), (0., -1., 0.)),
    ((2, 3, 6, 7), (0., 1., 0.)),
    ((0, 3, 4, 7), (-1., 0., 0.)),
    ((1, 2, 5, 6), (1., 0., 0.)),
)
# Numerical grazing exclusion only; no fitted visibility/accuracy threshold.
GRAZING_COSINE_EPS = 1e-6


def _dimensions(value, *, physical=False):
    try:
        names = ('x', 'y', 'z') if physical else ('width', 'height', 'depth')
        result = np.asarray([float(value[name]) for name in names])
    except (TypeError, KeyError, ValueError):
        return None
    return result if np.isfinite(result).all() and (result > 0).all() else None


def _points(dimensions):
    w, h, d = dimensions / 2.
    return np.asarray([[-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
                       [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]])


def _hidden(pose, dimensions):
    """Return eight conservative back-face states, or None for invalid geometry."""
    try:
        transform = np.asarray(pose, dtype=float)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-6, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0.)):
        return None
    camera_points = _points(dimensions) @ rotation.T + transform[:3, 3]
    if (camera_points[:, 2] <= 0.).any():
        return None
    scores = []
    for members, normal in FACES:
        centre = camera_points[list(members)].mean(axis=0)
        norm = float(np.linalg.norm(centre))
        if norm == 0.:
            return None
        scores.append(float(np.dot(rotation @ np.asarray(normal), centre)) / norm)
    # A vertex is hidden only if every incident face confidently faces away.
    # Near-grazing faces therefore cannot produce a SELF_OCCLUDED proposal.
    return np.asarray([all(scores[j] > GRAZING_COSINE_EPS
                           for j, (members, _) in enumerate(FACES) if i in members)
                       for i in range(8)])


def _xy_for_corner(obj, index):
    annotations = obj.get('keypoint_annotations') or []
    point = annotations[index] if index < len(annotations) else {}
    if not isinstance(point, dict):
        point = {}
    xy = point.get('xy')
    fallback = obj.get('manual_kps') or obj.get('projected_cuboid') or []
    if xy is None and index < len(fallback):
        xy = fallback[index]
    try:
        coordinate = np.asarray(xy, dtype=float)
    except (TypeError, ValueError):
        return None, point
    if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
        return None, point
    return coordinate.tolist(), point


def proposal_for_object(obj, camera_data, image_size=None):
    """Suggest out-of-frame/self-hidden slots using original annotation only.

    image_size is the actual image's (width, height), never a prediction or a
    resized UI canvas.  Invalid/ambiguous geometry is omitted.  Existing manual
    labels must take precedence in the caller, including previously confirmed
    sidecar records.  Returned dictionary keys are integer corner IDs 0..7.
    """
    if not isinstance(obj, dict) or not isinstance(camera_data, dict):
        return {}
    try:
        width, height = image_size if image_size is not None else (
            camera_data['width'], camera_data['height'])
        width, height = int(width), int(height)
    except (TypeError, ValueError, KeyError):
        return {}
    if width <= 0 or height <= 0:
        return {}
    proposals, coordinates = {}, {}
    for index in range(8):
        xy, point = _xy_for_corner(obj, index)
        coordinates[index] = (xy, point)
        if xy is None or (0. <= xy[0] < width and 0. <= xy[1] < height):
            continue
        proposals[index] = dict(
            status='OUT_OF_FRAME', source='machine_reference_coordinate_bounds',
            machine_proposal_only=True, human_confirmation_required=True,
            independent_reference_claim=False,
            evidence=dict(reference_xy=xy, actual_image_size=[width, height],
                image_boundary_rule='0 <= x < width and 0 <= y < height',
                existing_annotation_in_frame=point.get('in_frame'),
                coordinate_source=point.get('source', 'legacy'),
                bound_applies_to_existing_reference_coordinate=True))

    # The pose and corner convention must describe the same existing reference.
    if obj.get('keypoint_frame') != 'camera_dynamic_0123_v4':
        return proposals
    facing = obj.get('camera_facing_pnp') or {}
    dimensions = _dimensions(facing.get('dimensions_m') or obj.get('dimensions_m'))
    if dimensions is None:
        return proposals
    pose = facing.get('pose_transform')
    base_hidden = _hidden(pose, dimensions)
    if base_hidden is None:
        return proposals
    hypotheses = [base_hidden]
    axis_candidates = []
    candidates = obj.get('canonical_pose_candidates') or []
    canonical_dimensions = _dimensions(obj.get('physical_dimensions_m'), physical=True)
    if candidates and canonical_dimensions is None:
        return proposals
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return proposals
        permutation = candidate.get('canonical_to_camera_facing_keypoint_permutation')
        if (not isinstance(permutation, (list, tuple)) or len(permutation) < 8
                or any(type(i) is not int for i in permutation[:8])
                or sorted(permutation[:8]) != list(range(8))):
            return proposals
        hidden = _hidden(candidate.get('pose_transform'), canonical_dimensions)
        if hidden is None:
            return proposals
        hypotheses.append(hidden[np.asarray(permutation[:8], dtype=int)])
        axis_candidates.append(candidate.get('axis_assignment'))
    if not candidates and not facing.get('axis_assignment_confirmed', False):
        # No explicit axis confirmation and no hypotheses to establish agreement.
        return proposals
    for index in range(8):
        xy, point = coordinates[index]
        if index in proposals or xy is None or not all(h[index] for h in hypotheses):
            continue
        if point.get('reason') == 'visible' or point.get('visibility') == 2:
            # A contradictory previous direct-visible claim needs individual review.
            continue
        proposals[index] = dict(
            status='SELF_OCCLUDED', source='machine_existing_reference_cuboid_consensus',
            machine_proposal_only=True, human_confirmation_required=True,
            independent_reference_claim=False,
            evidence=dict(reference_xy=xy, coordinate_source=point.get('source', 'legacy'),
                corner_order='camera_dynamic_0123_v4', method='incident_face_back_face_culling',
                pose_source='existing_annotation.camera_facing_pnp_and_canonical_pose_candidates',
                camera_facing_dimensions_m=dict(width=float(dimensions[0]),
                    height=float(dimensions[1]), depth=float(dimensions[2])),
                hypotheses_compared=len(hypotheses), canonical_axis_candidates=axis_candidates,
                signed_axis_confirmed=bool(facing.get('axis_assignment_confirmed', False)),
                all_mapped_hypotheses_hidden=True, grazing_cosine_exclusion=GRAZING_COSINE_EPS,
                geometry_assumption='solid_cuboid_approximation_not_actual_pallet_mesh',
                limitations=['pallet_openings_and_actual_material_not_modelled',
                            'external_occlusion_not_determined',
                            'existing_annotation_pose_is_not_independent_reference']))
    return proposals
