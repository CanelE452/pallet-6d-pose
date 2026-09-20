"""Bounded, GT-free corner-pair gates for frozen PoseFix predictions.

This module reads no datasets, checkpoints, or annotations.  It only evaluates
the supplied model predictions, camera calibration, and registry dimensions.
The existing pseudo-label PnP helpers are deliberately reused without changing
their solver, camera convention, or projected-diagonal definition.
"""

from __future__ import annotations

import cv2
import numpy as np

from scripts.self_training_yolo import pseudo_label_filters as F


PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6))
FLIP_PERM = np.array([1, 0, 3, 2, 5, 4, 7, 6, 8], dtype=np.int64)
GEOMETRY_THRESHOLD = 0.05
FLIP_THRESHOLD = 0.05
MODE_THRESHOLD = 0.05
BOX_CONFIDENCE_THRESHOLD = 0.85
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
MIN_VALID_CORNERS = 6
POLICIES = ("geometry_only", "full")


def _points(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (9, 2):
        raise ValueError(f"{name} must have shape (9, 2), got {result.shape}")
    return result


def _valid(value, name):
    result = np.asarray(value, dtype=bool)
    if result.shape != (9,):
        raise ValueError(f"{name} must have shape (9,), got {result.shape}")
    return result


def _eight(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (8,):
        raise ValueError(f"{name} must have shape (8,), got {result.shape}")
    return result


def unflip_points(q, width):
    """Undo horizontal image reflection AND native keypoint permutation."""
    q = _points(q, "q")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be finite and positive")
    result = q[FLIP_PERM].copy()
    result[:, 0] = float(width) - 1.0 - result[:, 0]
    return result


def normalized_distances(q, other, valid, other_valid, diagonal):
    """Per-corner correspondence distances; missing/invalid evidence is inf."""
    q, other = _points(q, "q"), _points(other, "other")
    valid, other_valid = _valid(valid, "valid"), _valid(other_valid, "other_valid")
    result = np.full(8, np.inf, dtype=np.float64)
    if not np.isfinite(diagonal) or diagonal <= 0:
        return result
    both = (valid & other_valid & np.isfinite(q).all(1) & np.isfinite(other).all(1))[:8]
    result[both] = np.linalg.norm(q[:8][both] - other[:8][both], axis=1) / diagonal
    return result


def _hypothesis_detail(name, points_3d, q, valid, K, normalization_px):
    """Expose the old helper's aggregate score and its individual residuals."""
    try:
        s = F._hypothesis_scores(name, points_3d, q, valid, K, None, None)
    except cv2.error as error:
        # The legacy solver can throw for five valid corners when a removal
        # leaves four noncoplanar points.  This is a failed geometry check,
        # never permission to apply a correction.  Preserve the error text.
        return {
            "name": name,
            "s_reproj": float("inf"),
            "s_remove": float("inf"),
            "projected_diagonal_px": 0.0,
            "normalization_px": 0.0 if normalization_px is None else float(normalization_px),
            "per_corner_remove": [float("inf")] * 8,
            "per_corner_reprojection": [float("inf")] * 8,
            "solver_error": str(error),
        }
    denominator = s.projected_diagonal_px if normalization_px is None else normalization_px
    remove = np.full(8, np.inf, dtype=np.float64)
    reprojection = np.full(8, np.inf, dtype=np.float64)
    indices = np.flatnonzero(valid[:8])
    if np.isfinite(denominator) and denominator > 1e-6 and s.rvec is not None:
        projected = F._project(points_3d[:8], s.rvec, s.tvec, K)
        reprojection[indices] = np.linalg.norm(projected[indices] - q[indices], axis=1) / denominator
        if len(indices) >= F.MIN_REMOVAL_POINTS:
            for position, left_out in enumerate(indices):
                remaining = np.delete(indices, position)
                try:
                    partial = F._solve(points_3d[remaining], q[remaining], K)
                    if partial is None:
                        continue
                    prediction = F._project(points_3d[left_out:left_out + 1], *partial, K)[0]
                    error = np.linalg.norm(prediction - q[left_out]) / denominator
                    if np.isfinite(error):
                        remove[left_out] = float(error)
                except cv2.error:
                    # An unsolved corner cannot be accepted.  Other corners
                    # remain inspectable; no failing residual becomes zero.
                    continue
    return {
        "name": name,
        "s_reproj": float(s.reprojection),
        "s_remove": float(s.keypoint_removal),
        "projected_diagonal_px": float(s.projected_diagonal_px),
        "normalization_px": float(denominator),
        "per_corner_remove": remove.tolist(),
        "per_corner_reprojection": reprojection.tolist(),
    }


def geometry_details(q, valid, K, dimensions_xyz, *, hypothesis_name=None, normalization_px=None):
    """Return one coherent, no-GT W/D hypothesis plus exact legacy scores.

    The default hypothesis minimizes the FINITE legacy median-removal score;
    ties retain registry order.  No per-corner minimum across hypotheses is
    allowed.  ``hypothesis_name`` and ``normalization_px`` pin the original
    hypothesis and scale during the mixed-output structural recheck.
    """
    q, valid = _points(q, "q"), _valid(valid, "valid")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must be finite (3, 3) with positive focal lengths")
    dimensions_xyz = {key: float(dimensions_xyz[key]) for key in ("x", "y", "z")}
    if not all(np.isfinite(x) and x > 0 for x in dimensions_xyz.values()):
        raise ValueError("registry dimensions must be finite and positive")
    if normalization_px is not None and (not np.isfinite(normalization_px) or normalization_px <= 1e-6):
        raise ValueError("normalization_px must be finite and greater than 1e-6")
    effective_valid = valid & np.isfinite(q).all(axis=1)
    hypotheses = F.registry_hypotheses(dimensions_xyz)
    if hypothesis_name is not None and hypothesis_name not in {name for name, _ in hypotheses}:
        raise ValueError(f"unknown registry hypothesis: {hypothesis_name}")

    # The old scores are kept separately for exact legacy-rule comparisons.
    # They minimize each aggregate score independently, unlike the new gate.
    try:
        legacy = F.geometry_scores(q, effective_valid, K, dimensions_xyz)
    except cv2.error as error:
        legacy = {
            "s_reproj": float("inf"), "s_remove": float("inf"), "s_flip": None,
            "projected_diagonal_px": 0.0, "hypotheses": [], "solver_error": str(error),
        }
    details = [
        _hypothesis_detail(name, points_3d, q, effective_valid, K, normalization_px)
        for name, points_3d in hypotheses
    ]
    if hypothesis_name is not None:
        chosen = next(item for item in details if item["name"] == hypothesis_name)
    else:
        finite = [item for item in details if np.isfinite(item["s_remove"])
                  and np.isfinite(item["projected_diagonal_px"])
                  and item["projected_diagonal_px"] > 1e-6]
        chosen = min(finite, key=lambda item: item["s_remove"]) if finite else None
    return {
        "chosen_hypothesis": chosen["name"] if chosen else None,
        "projected_diagonal_px": chosen["projected_diagonal_px"] if chosen else 0.0,
        "normalization_px": chosen["normalization_px"] if chosen else 0.0,
        "s_reproj": chosen["s_reproj"] if chosen else float("inf"),
        "s_remove": chosen["s_remove"] if chosen else float("inf"),
        "per_corner_remove": chosen["per_corner_remove"] if chosen else [float("inf")] * 8,
        "per_corner_reprojection": chosen["per_corner_reprojection"] if chosen else [float("inf")] * 8,
        "hypotheses": details,
        "legacy_geometry_scores": legacy,
        "effective_valid": effective_valid.tolist(),
    }


def decide_pairs(policy, geometry, valid, box_conf, kp_conf, *, flip_residuals=None, mode_separation=None):
    """Decide corner evidence first, then require BOTH vertical-pair endpoints.

    ``flip_residuals`` and ``mode_separation`` must already be normalized by
    the ORIGINAL candidate's chosen-hypothesis diagonal.  Invalid flip points
    must have infinite residuals (``normalized_distances`` enforces this).
    There is deliberately no global median-reprojection veto.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    valid = _valid(valid, "valid")
    kp_conf = np.asarray(kp_conf, dtype=np.float64)
    if kp_conf.shape != (9,):
        raise ValueError("kp_conf must have shape (9,)")
    effective = _valid(geometry["effective_valid"], "geometry.effective_valid")
    eligible = (valid & effective & np.isfinite(kp_conf) & (kp_conf >= KEYPOINT_CONFIDENCE_THRESHOLD))[:8]
    removal = _eight(geometry["per_corner_remove"], "per_corner_remove")
    frame_reasons = []
    if not np.isfinite(box_conf) or box_conf < BOX_CONFIDENCE_THRESHOLD:
        frame_reasons.append("box_confidence")
    if int(eligible.sum()) < MIN_VALID_CORNERS:
        frame_reasons.append("too_few_valid_corners")
    if geometry["chosen_hypothesis"] is None:
        frame_reasons.append("no_finite_geometry_hypothesis")
    evidence = {"geometry": (removal, GEOMETRY_THRESHOLD)}
    if policy == "full":
        flip = np.full(8, np.inf) if flip_residuals is None else _eight(flip_residuals, "flip_residuals")
        mode = np.full(8, np.inf) if mode_separation is None else _eight(mode_separation, "mode_separation")
        evidence.update(flip=(flip, FLIP_THRESHOLD), mode=(mode, MODE_THRESHOLD))
    corner_reasons = []
    corner_pass = []
    for i in range(8):
        reasons = list(frame_reasons)
        if not eligible[i]:
            reasons.append("invalid_or_low_corner_confidence")
        for name, (values, threshold) in evidence.items():
            if not np.isfinite(values[i]) or values[i] < 0 or values[i] > threshold:
                reasons.append(name)
        corner_reasons.append(reasons)
        corner_pass.append(not reasons)
    pair_accept = [bool(corner_pass[a] and corner_pass[b]) for a, b in PAIRS]
    corner_accept = [False] * 8
    for accepted, (a, b) in zip(pair_accept, PAIRS):
        corner_accept[a] = corner_accept[b] = accepted
    return {
        "policy": policy,
        "frame_reasons": frame_reasons,
        "corner_reasons": corner_reasons,
        "corner_evidence_pass": corner_pass,
        "pair_accept": pair_accept,
        "corner_accept": corner_accept,
    }


def apply_pairs(n2_xy, candidate_xy, pair_accept):
    """Copy accepted pairs only; fallback and centroid retain N2 bit patterns."""
    base = np.asarray(n2_xy)
    if base.shape != (9, 2) or not np.issubdtype(base.dtype, np.floating):
        raise ValueError("n2_xy must be a floating array of shape (9, 2)")
    candidate = _points(candidate_xy, "candidate_xy")
    pair_accept = np.asarray(pair_accept, dtype=bool)
    if pair_accept.shape != (4,):
        raise ValueError("pair_accept must have shape (4,)")
    result = base.copy()
    for accepted, pair in zip(pair_accept, PAIRS):
        if accepted:
            indices = list(pair)
            if not np.isfinite(candidate[indices]).all():
                raise ValueError("accepted candidate corners must be finite")
            result[indices] = candidate[indices]
    return result


def gate_with_recheck(n2_xy, candidate_xy, valid, K, dimensions_xyz, box_conf, kp_conf,
                     *, policy="full", flip_residuals=None, mode_separation=None, geometry=None):
    """Apply a gate and monotonically reject structurally invalid mixed pairs.

    Rechecks pin the candidate's selected registry hypothesis and original
    projected diagonal.  Only ACCEPTED corners can cause further rejection;
    unchanged fallback corners are not required to become perfect.  No pair
    is ever re-enabled, so four removal rounds suffice for four pairs.
    """
    if geometry is None:
        geometry = geometry_details(candidate_xy, valid, K, dimensions_xyz)
    decision = decide_pairs(policy, geometry, valid, box_conf, kp_conf,
                            flip_residuals=flip_residuals, mode_separation=mode_separation)
    accepted = np.asarray(decision["pair_accept"], dtype=bool)
    history = []
    original_diagonal = float(geometry["projected_diagonal_px"])
    hypothesis = geometry["chosen_hypothesis"]
    if accepted.any() and (hypothesis is None or not np.isfinite(original_diagonal) or original_diagonal <= 1e-6):
        raise ValueError("accepted pairs require a finite original geometry hypothesis")
    for round_index in range(4):
        if not accepted.any():
            break
        mixed = apply_pairs(n2_xy, candidate_xy, accepted)
        checked = geometry_details(mixed, valid, K, dimensions_xyz,
                                   hypothesis_name=hypothesis, normalization_px=original_diagonal)
        residuals = _eight(checked["per_corner_remove"], "per_corner_remove")
        reject = np.array([bool(accepted[i] and any(
            not np.isfinite(residuals[j]) or residuals[j] < 0 or residuals[j] > GEOMETRY_THRESHOLD
            for j in pair)) for i, pair in enumerate(PAIRS)])
        history.append({
            "round": round_index + 1,
            "chosen_hypothesis": hypothesis,
            "normalization_px": original_diagonal,
            "per_corner_remove": residuals.tolist(),
            "pair_accept_before": accepted.tolist(),
            "pair_rejected": reject.tolist(),
        })
        if not reject.any():
            break
        accepted &= ~reject
    corners = np.zeros(8, dtype=bool)
    for enabled, pair in zip(accepted, PAIRS):
        if enabled:
            corners[list(pair)] = True
    return {
        "points": apply_pairs(n2_xy, candidate_xy, accepted),
        "decision": decision,
        "pair_accept": accepted.tolist(),
        "corner_accept": corners.tolist(),
        "recheck_history": history,
    }
