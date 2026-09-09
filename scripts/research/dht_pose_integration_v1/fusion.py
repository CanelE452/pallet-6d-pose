"""Fixed 2D corner/side-line fusion under camera-facing 0123.

Inference takes predicted corners and predicted structural lines only. All
coordinates must already be in ORIGINAL image pixels. This module performs no
resize, padding correction, corner permutation, instance selection or PnP.
The evaluation functions below are separate and may receive ground truth.

For each observed corner p, minimize in image-diagonal-normalized coordinates
    ||q-p||^2 + lam * sum_incident_lines (n @ q - rho)^2.
The unit anchor gives a unique solution even if the two incident lines are
parallel. It does not make inaccurate DHT lines reliable or enforce a coherent
3D cuboid. Missing corners remain missing; an optional ninth center is unchanged.
"""
from __future__ import annotations

import numpy as np


SIDE_EDGES = ((1, 2), (3, 0), (5, 6), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))
SIDE_ROLE_IDS = (1, 3, 5, 7, 8, 9, 10, 11)
INCIDENT_ROLES = tuple(tuple(j for j, edge in enumerate(SIDE_EDGES) if k in edge)
                       for k in range(8))
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FIXED_SECONDARY_LAMBDA = 1.0


def _image_geometry(width, height):
    width, height = float(width), float(height)
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("width and height must be finite positive image sizes")
    return np.array([(width - 1) / 2, (height - 1) / 2]), float(np.hypot(width, height))


def _points_and_valid(points, valid):
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.shape not in ((8, 2), (9, 2)):
        raise ValueError("points must have shape [8,2] or [9,2]")
    if valid.shape not in ((8,), (len(points),)):
        raise ValueError("valid must have 8 entries or one entry per point")
    # A valid explicit prediction at [-1,-1] is allowed: points may be off-frame.
    # Missingness is provided by the detector adapter, never inferred from GT.
    valid = valid.copy() & np.isfinite(points[:len(valid)]).all(-1)
    return points, valid


def _line_equations(lines, origin, diagonal):
    lines = np.asarray(lines, dtype=np.float64)
    if lines.shape != (8, 2, 2):
        raise ValueError("lines must have shape [8,2,2] in SIDE_EDGES order")
    finite = np.isfinite(lines).all(axis=(1, 2))
    delta = (lines[:, 1] - lines[:, 0]) / diagonal
    length = np.linalg.norm(delta, axis=-1)
    valid = finite & (length > 1e-12)
    normals, offsets = np.zeros((8, 2)), np.zeros(8)
    normals[valid] = np.stack([-delta[valid, 1], delta[valid, 0]], -1) / length[valid, None]
    offsets[valid] = np.sum(normals[valid] * ((lines[valid, 0] - origin) / diagonal), -1)
    return normals, offsets, valid


def fuse_corners(points, valid, lines, lam, width, height):
    """Return (fused_points, effective_valid) for ONE frame and ONE DHT seed.

    ``points`` is [8/9,2], ``valid`` [8] or [8/9], ``lines`` [8,2,2]. The
    returned shapes match these inputs. Invalid entries are left unchanged;
    finite checks can only remove validity. The center, if present, is copied
    exactly. Lines get no confidence, GT-support or visibility weighting. Only
    numerically undefined predicted lines are ignored. ``lam=0`` is exact
    identity, including off-frame and missing values. No coordinate clipping.
    """
    points, effective_valid = _points_and_valid(points, valid)
    origin, diagonal = _image_geometry(width, height)
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    normals, offsets, line_valid = _line_equations(lines, origin, diagonal)
    result = points.copy()
    if lam == 0:
        return result, effective_valid
    for corner, roles in enumerate(INCIDENT_ROLES):
        if not effective_valid[corner]:
            continue
        roles = [j for j in roles if line_valid[j]]
        if not roles:
            continue
        n, rho = normals[roles], offsets[roles]
        p = (points[corner] - origin) / diagonal
        system = np.eye(2) + lam * (n.T @ n)
        # Solve for displacement to avoid round-trip noise on unconstrained p.
        displacement = np.linalg.solve(system, -lam * n.T @ (n @ p - rho))
        result[corner] = points[corner] + diagonal * displacement
    return result, effective_valid


def corners_to_lines(points, valid):
    """Corresponding eight structural lines from predicted endpoint pairs.

    The result is an infinite-line representation. Both predicted endpoints
    must be finite, observed and distinct; missing corners are not reconstructed.
    """
    points, valid = _points_and_valid(points, valid)
    lines = points[np.asarray(SIDE_EDGES)].copy()
    line_valid = np.array([valid[a] and valid[b] for a, b in SIDE_EDGES])
    line_valid &= np.isfinite(lines).all(axis=(1, 2))
    line_valid &= np.linalg.norm(lines[:, 1] - lines[:, 0], axis=-1) > 1e-9
    return lines, line_valid


def _mean_or_nan(values, valid):
    return float(np.mean(np.asarray(values)[valid])) if np.any(valid) else float("nan")


def corner_metrics(points, pred_valid, gt_points, gt_valid, width, height):
    """Strict corner-ID 0..7 evaluation; no Hungarian or GT-selected symmetry.

    Missing predictions cost one ORIGINAL image diagonal. The frozen CONFIG
    selects using ``mean_capped_diagonal_normalized_error``: divide each error
    by that diagonal and cap at one, then average valid GT corners. Callers
    average frames and DHT seeds equally. Other explicitly named pixel-error
    fields remain uncapped so large finite failures are visible. Missing
    predictions fail PCK. Real DEV must not select parameters with these metrics.
    """
    points, pred_valid = _points_and_valid(points, pred_valid)
    gt_points, gt_valid = _points_and_valid(gt_points, gt_valid)
    _, diagonal = _image_geometry(width, height)
    truth_valid, observed = gt_valid[:8], pred_valid[:8] & gt_valid[:8]
    error = np.full(8, np.nan)
    error[observed] = np.linalg.norm(points[:8][observed] - gt_points[:8][observed], axis=-1)
    penalized = np.full(8, np.nan)
    penalized[truth_valid] = diagonal
    penalized[observed] = error[observed]
    return {
        "error_px": error,
        "penalized_error_px": penalized,
        "gt_valid": truth_valid.copy(),
        "predicted_on_gt": observed,
        "n_gt": int(truth_valid.sum()),
        "n_pred_on_gt": int(observed.sum()),
        "coverage": float(observed.sum() / truth_valid.sum()) if truth_valid.any() else float("nan"),
        "mean_error_px": _mean_or_nan(error, observed),
        "mean_error_missing_diagonal_px": _mean_or_nan(penalized, truth_valid),
        "mean_error_missing_diagonal_normalized": _mean_or_nan(penalized / diagonal, truth_valid),
        "mean_capped_diagonal_normalized_error": _mean_or_nan(np.minimum(penalized / diagonal, 1), truth_valid),
        "pck_5px": _mean_or_nan(observed & (error <= 5), truth_valid),
        "pck_10px": _mean_or_nan(observed & (error <= 10), truth_valid),
        "pck_20px": _mean_or_nan(observed & (error <= 20), truth_valid),
        "all8_gt_valid": bool(truth_valid.all()),
        "all8_at_10px": bool((observed & (error <= 10)).all()) if truth_valid.all() else None,
    }


def _segment_intersects(p0, p1, width, height):
    # Same original-frame support rectangle as deep_hough_side_v1/targets.py.
    delta, lo, hi = p1 - p0, 0.0, 1.0
    for p, q in ((-delta[0], p0[0]), (delta[0], width - p0[0]),
                 (-delta[1], p0[1]), (delta[1], height - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
    return lo <= hi


def gt_line_support(gt_points, gt_valid, width, height):
    """EVALUATION ONLY: annotated >=2px segments intersecting original frame.

    This is structural support, including amodal lines, not physical visibility.
    Its definition matches the frozen DHT experiment's target support.
    """
    points, valid = _points_and_valid(gt_points, gt_valid)
    _image_geometry(width, height)
    return np.array([bool(valid[a] and valid[b]
                          and np.linalg.norm(points[a] - points[b]) >= 2.0
                          and _segment_intersects(points[a], points[b], width, height))
                     for a, b in SIDE_EDGES])


def line_metrics(lines, pred_line_valid, gt_points, gt_valid, width, height,
                 gt_support=None):
    """EVALUATION ONLY: undirected angle and GT-endpoint to infinite-line px.

    For paired baseline/fused line evaluation pass ``corners_to_lines`` output.
    DHT-only line error may also be evaluated with all eight predicted roles.
    Missing lines incur diagonal distance and 90 degrees in the explicit
    penalized fields. Raw means report coverage-conditioned observed lines.
    Optional GT support can only SUBSET the fixed structural support definition.
    """
    origin, diagonal = _image_geometry(width, height)
    normals, _, finite = _line_equations(lines, origin, diagonal)
    lines = np.asarray(lines, dtype=np.float64)
    pred_line_valid = np.asarray(pred_line_valid, dtype=bool)
    if pred_line_valid.shape != (8,):
        raise ValueError("pred_line_valid must have shape [8]")
    points, _ = _points_and_valid(gt_points, gt_valid)
    support = gt_line_support(gt_points, gt_valid, width, height)
    if gt_support is not None:
        subset = np.asarray(gt_support, dtype=bool)
        if subset.shape != (8,):
            raise ValueError("gt_support must have shape [8]")
        support &= subset
    observed = support & finite & pred_line_valid
    truth = points[np.asarray(SIDE_EDGES)]
    angle, distance = np.full(8, np.nan), np.full(8, np.nan)
    delta = truth[observed, 1] - truth[observed, 0]
    truth_direction = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
    pred_direction = np.stack([normals[observed, 1], -normals[observed, 0]], -1)
    # atan2 is stable near parallel lines, where arccos loses precision.
    cross = truth_direction[:, 0] * pred_direction[:, 1] - truth_direction[:, 1] * pred_direction[:, 0]
    dot = np.sum(truth_direction * pred_direction, -1)
    angle[observed] = np.rad2deg(np.arctan2(np.abs(cross), np.abs(dot)))
    distance[observed] = np.abs(np.sum((truth[observed] - lines[observed, :1])
                                      * normals[observed, None], -1)).mean(-1)
    penalized_distance, penalized_angle = np.full(8, np.nan), np.full(8, np.nan)
    penalized_distance[support], penalized_angle[support] = diagonal, 90.0
    penalized_distance[observed], penalized_angle[observed] = distance[observed], angle[observed]
    return {
        "distance_px": distance, "angle_deg": angle,
        "penalized_distance_px": penalized_distance,
        "penalized_angle_deg": penalized_angle,
        "gt_support": support, "predicted_on_gt": observed,
        "n_gt": int(support.sum()), "n_pred_on_gt": int(observed.sum()),
        "coverage": float(observed.sum() / support.sum()) if support.any() else float("nan"),
        "mean_distance_px": _mean_or_nan(distance, observed),
        "mean_angle_deg": _mean_or_nan(angle, observed),
        "mean_distance_missing_diagonal_px": _mean_or_nan(penalized_distance, support),
        "mean_angle_missing_90deg": _mean_or_nan(penalized_angle, support),
    }
