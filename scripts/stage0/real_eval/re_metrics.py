"""Evaluators for REAL IN-HOUSE evaluation: ADD / ADD-S, exact 3D IoU, AP.

`mh_fusion.run_eval` records that oriented-box 3D IoU was NOT_COMPUTED because
"an approximation would be a wrong number under a right name".  This module
supplies the exact one instead of the approximation: two oriented boxes are
intersected as convex polyhedra (12 half-spaces, an interior point from a
Chebyshev-centre LP, then the hull volume).  Disjoint boxes make the LP
infeasible, which is the correct answer 0 rather than a fallback.

Nothing here selects a threshold.  Detection metrics take the operating point
as an argument, because the final one comes from REAL_DEV.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

EPS = 1e-9


# --------------------------------------------------------------------------
# pose error


def add(model_points, R_pred, t_pred, R_gt, t_gt):
    """Mean 3D distance between corresponding transformed model points."""
    p = (R_pred @ model_points.T).T + t_pred
    q = (R_gt @ model_points.T).T + t_gt
    return float(np.linalg.norm(p - q, axis=1).mean())


def add_s(model_points, R_pred, t_pred, R_gt, t_gt):
    """Symmetric variant: nearest-neighbour instead of corresponding point.

    A pallet is not fully symmetric, so ADD is the primary number; ADD-S is
    reported beside it because a 180-degree yaw flip is the failure this task
    actually produces, and ADD alone hides how close such a pose is.
    """
    p = (R_pred @ model_points.T).T + t_pred
    q = (R_gt @ model_points.T).T + t_gt
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def pose_error(R_pred, t_pred, R_gt, t_gt):
    """(rotation degrees, translation metres)."""
    cos = (np.trace(R_pred.T @ R_gt) - 1.0) / 2.0
    degrees = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return degrees, float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt)))


def success_5cm5deg(R_pred, t_pred, R_gt, t_gt):
    degrees, metres = pose_error(R_pred, t_pred, R_gt, t_gt)
    return bool(degrees <= 5.0 and metres <= 0.05)


# --------------------------------------------------------------------------
# exact oriented-box 3D IoU


def box_corners(R, t, extents):
    """8 corners of an oriented box with full side lengths `extents`."""
    half = np.asarray(extents, float) / 2.0
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     float)
    return (R @ (signs * half).T).T + np.asarray(t, float)


def box_halfspaces(R, t, extents):
    """(A, b) with A x + b <= 0 for the six faces, as scipy wants them."""
    half = np.asarray(extents, float) / 2.0
    centre = np.asarray(t, float)
    rows, offs = [], []
    for axis in range(3):
        n = R[:, axis]
        rows.append(n)
        offs.append(-(n @ centre + half[axis]))
        rows.append(-n)
        offs.append(-(-n @ centre + half[axis]))
    return np.asarray(rows), np.asarray(offs)


def _interior_point(A, b):
    """Chebyshev centre.  Returns None when the intersection has no volume."""
    norms = np.linalg.norm(A, axis=1)
    # maximise r subject to  A x + norm*r <= -b
    cost = np.zeros(A.shape[1] + 1)
    cost[-1] = -1.0
    ub_A = np.hstack([A, norms[:, None]])
    result = linprog(cost, A_ub=ub_A, b_ub=-b,
                     bounds=[(None, None)] * A.shape[1] + [(None, None)],
                     method="highs")
    if not result.success or result.x[-1] <= EPS:
        return None
    return result.x[:-1]


def intersection_volume(R_a, t_a, ext_a, R_b, t_b, ext_b):
    A1, b1 = box_halfspaces(R_a, t_a, ext_a)
    A2, b2 = box_halfspaces(R_b, t_b, ext_b)
    A, b = np.vstack([A1, A2]), np.concatenate([b1, b2])
    inside = _interior_point(A, b)
    if inside is None:
        return 0.0
    from scipy.spatial import HalfspaceIntersection
    halfspaces = np.hstack([A, b[:, None]])
    try:
        intersection = HalfspaceIntersection(halfspaces, inside)
        return float(ConvexHull(intersection.intersections).volume)
    except Exception:
        return 0.0


def iou_3d(R_pred, t_pred, ext_pred, R_gt, t_gt, ext_gt):
    """Exact IoU of two oriented boxes.  No axis-aligned approximation."""
    inter = intersection_volume(R_pred, t_pred, ext_pred, R_gt, t_gt, ext_gt)
    v_a = float(np.prod(ext_pred))
    v_b = float(np.prod(ext_gt))
    union = v_a + v_b - inter
    return float(inter / union) if union > EPS else 0.0


# --------------------------------------------------------------------------
# detection


def precision_recall(scores, labels, threshold):
    """At one operating point.  `labels` is 1 for a frame containing a pallet."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "tn": tn, "precision": float(precision), "recall": float(recall),
            "fp_per_image": float(fp / max(fp + tn, 1)),
            "f1": float(2 * precision * recall / max(precision + recall, EPS))}


def average_precision(scores, labels):
    """Area under the precision-recall curve, by exact rank interpolation."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    recall = tp / max(int((labels == 1).sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))
