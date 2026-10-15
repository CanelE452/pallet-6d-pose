"""GT-free whole-pallet mask evidence for comparing rigid pose candidates.

This never queries pixels at S/U/O/T or invents keypoint observations. A cube's
projected convex silhouette is compared with the observed mask's convex outer
envelope. Frame-border segments are excluded from contour distance: a clipped
canvas edge is not an observed pallet edge. Hull filling is an explicit shape
approximation; it is not a reconstructed mask or a visibility certificate.
"""
from __future__ import annotations

import cv2
import numpy as np


def _finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite with shape ' + str(shape))
    return result


def _hull_mask(points, image_hw):
    h, w = image_hw
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(xy) < 3 or not np.isfinite(xy).all() or np.max(np.abs(xy)) > 1e7:
        return None
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or abs(cv2.contourArea(hull)) < 1.:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    return mask


def _interior_contour(mask, margin):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), float)
    points = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    h, w = mask.shape
    eligible = ((points[:, 0] >= margin) & (points[:, 0] < w-margin)
                & (points[:, 1] >= margin) & (points[:, 1] < h-margin))
    return points[eligible]


def _distance_to_contour(points, contour, image_hw):
    if not len(points) or not len(contour):
        return None
    canvas = np.ones(image_hw, np.uint8)
    xy = np.asarray(contour, np.int32)
    canvas[xy[:, 1], xy[:, 0]] = 0
    distance = cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    p = np.asarray(points, np.int32)
    return distance[p[:, 1], p[:, 0]].astype(float)


def observe_mask_outline(pallet_mask, image_hw, *, frame_margin_px=7.):
    """Whole-image mask envelope; no detector or model coordinates are read."""
    h, w = map(int, image_hw)
    mask = np.asarray(pallet_mask, dtype=float)
    if (mask.shape != (h, w) or not np.isfinite(mask).all()
            or np.any(mask < 0) or np.any(mask > 1)
            or not np.isfinite(frame_margin_px) or frame_margin_px < 0):
        raise ValueError('Bound finite mask/current canvas/nonnegative frame margin required')
    y, x = np.nonzero(mask >= .5)
    if len(x) < 3:
        return dict(supported=False, reason='insufficient_actual_mask_foreground')
    points = np.column_stack([x, y]); hull = _hull_mask(points, (h, w))
    if hull is None:
        return dict(supported=False, reason='degenerate_actual_mask_outer_envelope')
    contour = _interior_contour(hull, frame_margin_px)
    if len(contour) < 3:
        return dict(supported=False, reason='no_actual_non_canvas_outer_contour')
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.)
    return dict(supported=True, image_hw=[h, w], hull_mask=hull, interior_contour_xy=contour,
        mask_area_px=int((mask >= .5).sum()), hull_area_px=int(hull.sum()),
        reference_diagonal_px=diagonal, frame_margin_px=float(frame_margin_px),
        source='whole observed pallet-mask convex outer envelope',
        hull_is_surface_or_visibility_certificate=False, S_pixel_queries=0, U_pixel_queries=0)


def score_projected_outline(observation, xyz, K, R, t):
    """Compare a pose using an existing global mask; no KP positions are read.

The fixed dimensionless score is (1-IoU) plus symmetric median contour
distance divided by observed diagonal. It is an engineering fit criterion,
not a calibrated likelihood or a proof of physical pose accuracy.
"""
    if not observation.get('supported'):
        return dict(supported=False, reason=observation.get('reason'))
    xyz = _finite(xyz, (8, 3), 'registered cuboid')
    K = _finite(K, (3, 3), 'K'); R = _finite(R, (3, 3), 'R'); t = _finite(t, (3,), 't')
    if not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-7) or not np.isclose(np.linalg.det(R), 1.):
        raise ValueError('Proper rigid rotation required')
    camera = xyz @ R.T + t
    if np.min(camera[:, 2]) <= 1e-5:
        return dict(supported=False, reason='nonpositive_model_depth')
    homogeneous = camera @ K.T
    xy = homogeneous[:, :2] / homogeneous[:, 2:]
    canvas = tuple(observation['image_hw']); model_mask = _hull_mask(xy, canvas)
    if model_mask is None or not model_mask.any():
        return dict(supported=False, reason='model_silhouette_has_no_canvas_area')
    # Apply the same raster-envelope construction to both sides. Without
    # this, digital staircase boundaries enlarge only the observed hull.
    yy, xx = np.nonzero(model_mask)
    model_mask = _hull_mask(np.column_stack([xx, yy]), canvas)
    if model_mask is None:
        return dict(supported=False, reason='degenerate_model_raster_envelope')
    actual = observation['hull_mask']
    union = int(np.count_nonzero(actual | model_mask))
    intersection = int(np.count_nonzero(actual & model_mask))
    iou = intersection / union
    model_contour = _interior_contour(model_mask, observation['frame_margin_px'])
    actual_contour = observation['interior_contour_xy']
    model_to_actual = _distance_to_contour(model_contour, actual_contour, canvas)
    actual_to_model = _distance_to_contour(actual_contour, model_contour, canvas)
    if model_to_actual is None or actual_to_model is None:
        return dict(supported=False, reason='no_eligible_two_way_outline_support')
    def stats(values):
        return dict(count=len(values), mean_px=float(values.mean()), median_px=float(np.median(values)),
                    P90_px=float(np.quantile(values, .9)), max_px=float(values.max()))
    first, second = stats(model_to_actual), stats(actual_to_model)
    distance = .5 * (first['median_px'] + second['median_px'])
    normalized = distance / observation['reference_diagonal_px']
    return dict(supported=True, hull_IoU=float(iou), symmetric_median_distance_px=distance,
        normalized_symmetric_median_distance=normalized, outline_loss=float(1-iou+normalized),
        actual_to_model=second, model_to_actual=first,
        observed_hull_area_px=observation['hull_area_px'], model_hull_area_px=int(model_mask.sum()),
        canvas_border_is_not_actual_edge=True, model_corners_are_pose_geometry_not_KP_observations=True,
        shape_fit_score_is_calibrated_probability=False, GT_inputs=False,
        S_local_queries=0, U_local_queries=0)


def choose_by_outline(observation, candidates):
    """Select the smallest supported global score, retaining original tie order."""
    scored = []
    for index, candidate in enumerate(candidates):
        evidence = score_projected_outline(observation, candidate['xyz'], candidate['K'],
                                           candidate['R'], candidate['t'])
        scored.append(dict(index=index, name=candidate['name'], outline=evidence))
    valid = [r for r in scored if r['outline'].get('supported')]
    chosen = min(valid, key=lambda r: (r['outline']['outline_loss'], r['index'])) if valid else None
    return dict(selected_index=None if chosen is None else chosen['index'],
                selected_name=None if chosen is None else chosen['name'], candidates=scored,
                selection_source='observed global silhouette only; stable original order ties',
                GT_inputs=False, per_record_thresholds=False)
