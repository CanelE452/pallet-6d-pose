"""Inference-only easy-to-hard refinement of independently predicted corners.

The RGB classifier supplies evidence, never pose-created correspondences.  The
caller supplies the initial pose and its exact ordered object model; no dimension
swap, corner permutation, or oracle visibility is performed here.  Planar easy
sets retain the branch of that initial pose through a conservative local prior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CurriculumConfig:
    easy_visibility: float = 0.65
    visible_threshold: float = 0.55
    easy_boundary: float = 0.55
    easy_accuracy: float = 0.45
    minimum_accuracy: float = 0.10
    interior_weight: float = 0.60
    unknown_weight: float = 0.10
    robust_px: float = 8.0
    minimum_spread_ratio: float = 0.01
    minimum_spread_px: float = 3.0
    # Thin front-face sets can be noncollinear yet poorly constrain depth/tilt.
    # Defer these sets until more spatially diverse independent corners arrive.
    maximum_jacobian_condition: float = 15.0
    maximum_rotation_deg: float = 15.0
    maximum_translation_fraction: float = 0.15
    maximum_projection_shift_px: float = 64.0
    rotation_prior_deg: float = 8.0
    translation_prior_fraction: float = 0.08
    prior_strength_px: float = 1.0
    validation_tolerance_px2: float = 0.10
    minimum_gain_px2: float = 1e-7
    unknown_gate_min_px: float = 6.0
    unknown_gate_max_px: float = 18.0
    max_nfev: int = 80
    irls_steps: int = 4
    enable_unknown: bool = True


def project(xyz, R, t, K):
    camera = (np.asarray(R) @ np.asarray(xyz).T).T + np.asarray(t).reshape(3)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1., 1.))))


def _array_field(appearance, name, default, length=8):
    value = np.asarray(appearance.get(name, np.full(length, default)), dtype=float)
    if value.shape != (length,):
        raise ValueError(f"appearance_result.{name} must have shape ({length},)")
    return np.clip(np.nan_to_num(value, nan=0., posinf=1., neginf=0.), 0., 1.)


def _shape(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    ratio = singular[1] / max(singular[0], 1e-12) if len(singular) > 1 else 0.
    return {"singular_values": singular.tolist(), "spread_ratio": float(ratio)}


def _geometry(xyz, points, ids, R, t, K, cfg):
    ids = np.asarray(ids, dtype=int)
    if len(ids) < 4:
        return {"usable": False, "reason": "fewer_than_four_independent_observations"}
    image = _shape(points[ids])
    obj = _shape(xyz[ids])
    if image["spread_ratio"] < cfg.minimum_spread_ratio or image["singular_values"][1] < cfg.minimum_spread_px:
        return {"usable": False, "reason": "insufficient_image_spread", "image": image, "object": obj}
    if obj["spread_ratio"] < 1e-8:
        return {"usable": False, "reason": "collinear_object_points", "image": image, "object": obj}
    rv = cv2.Rodrigues(R)[0]
    _, jac = cv2.projectPoints(np.ascontiguousarray(xyz[ids]), rv, np.asarray(t), K, None)
    jac = jac[:, :6]
    jac = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-12)[None, :]
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    rank = int(np.sum(singular > singular[0] * 1e-9))
    return {"usable": bool(rank == 6 and condition <= cfg.maximum_jacobian_condition),
            "reason": "usable" if rank == 6 and condition <= cfg.maximum_jacobian_condition else "unstable_pose_geometry",
            "image": image, "object": obj, "jacobian_rank": rank,
            "jacobian_condition": condition, "jacobian_singular_values": singular.tolist()}


def _huber_score(residuals, weights, radius):
    residuals = np.asarray(residuals)
    losses = np.where(residuals <= radius, residuals ** 2, 2 * radius * residuals - radius ** 2)
    return float(np.sum(losses * weights) / max(np.sum(weights), 1e-12))


def _finite_list(value):
    value = np.asarray(value)
    if value.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in value]
    return [_finite_list(v) for v in value]


def _fit(xyz, points, K, initial_R, initial_t, start_R, start_t, ids, weights, cfg):
    """IRLS weighted reprojection with an initial-pose prior, without PnP reseeding."""
    ids = np.asarray(ids, dtype=int)
    X, target = xyz[ids], points[ids]
    depth_scale = max(np.linalg.norm(initial_t), np.linalg.norm(np.ptp(xyz, axis=0)), 1e-6)
    initial_rotvec = cv2.Rodrigues(start_R @ initial_R.T)[0].reshape(3)
    x = np.r_[initial_rotvec, (start_t - initial_t) / depth_scale]
    robust_weights = np.asarray(weights, dtype=float).copy()

    def unpack(params):
        return cv2.Rodrigues(params[:3])[0] @ initial_R, initial_t + params[3:] * depth_scale

    def fun(params):
        R, t = unpack(params)
        residual = (project(X, R, t, K) - target) * np.sqrt(robust_weights[:, None])
        rotation_prior = params[:3] / np.deg2rad(cfg.rotation_prior_deg)
        translation_prior = params[3:] / cfg.translation_prior_fraction
        return np.r_[residual.reshape(-1), cfg.prior_strength_px * rotation_prior,
                     cfg.prior_strength_px * translation_prior]

    nfev = 0
    solved = None
    for _ in range(cfg.irls_steps):
        solved = least_squares(fun, x, method="trf", max_nfev=cfg.max_nfev,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        x = solved.x
        nfev += solved.nfev
        R, t = unpack(x)
        residual = np.linalg.norm(project(X, R, t, K) - target, axis=1)
        updated = weights * np.minimum(1., cfg.robust_px / np.maximum(residual, 1e-12))
        converged = np.max(np.abs(updated - robust_weights)) < 1e-4
        robust_weights = updated
        if converged:
            break
    return (*unpack(x), {"nfev": int(nfev), "optimizer_success": bool(solved.success),
                         "optimizer_message": str(solved.message),
                         "robust_weights": robust_weights.tolist()})


def solve_curriculum(points, xyz, K, initial_R, initial_t, self_hidden, appearance_result, config=None):
    """Return JSON-safe pose and per-stage evidence; preserve initial on rejection.

    Inputs are exactly eight detector observations and their corresponding object
    points, with a supplied valid object-to-camera pose.  ``self_hidden`` must be
    inferred from this initial pose by the caller.  RGB states/scores are evidence
    at the supplied observations, not calibrated probabilities.  Configuration
    may be a ``CurriculumConfig`` or a dict with those field names.

    A boundary stage may be deferred when fewer than four corners are visible.
    The visible stage then adds interior corners before fitting.  Unknowns can
    only augment an already independently supported visible solution; they never
    establish the pose by themselves.  All validation uses unchanged detector
    observations. Hidden projections are output separately and never refitted.
    """
    cfg = CurriculumConfig(**config) if isinstance(config, dict) else (config or CurriculumConfig())
    points = np.array(points, dtype=float, copy=True)
    xyz = np.array(xyz, dtype=float, copy=True)
    K = np.array(K, dtype=float, copy=True)
    initial_R = np.array(initial_R, dtype=float, copy=True)
    initial_t = np.array(initial_t, dtype=float, copy=True).reshape(3)
    hidden = np.array(self_hidden, dtype=bool, copy=True)
    if points.shape != (8, 2) or xyz.shape != (8, 3) or K.shape != (3, 3) or initial_R.shape != (3, 3) or hidden.shape != (8,):
        raise ValueError("expected points (8,2), xyz (8,3), K/R (3,3), t (3,), self_hidden (8,)")
    if not all(np.isfinite(v).all() for v in (xyz, K, initial_R, initial_t)):
        raise ValueError("object points, camera and initial pose must be finite")
    if not np.allclose(initial_R.T @ initial_R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(initial_R), 1., atol=1e-6):
        raise ValueError("initial_R must be a proper rotation matrix")
    if (((initial_R @ xyz.T).T + initial_t)[:, 2] <= 1e-6).any():
        raise ValueError("initial pose must place every model corner in front of the camera")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    if min(cfg.robust_px, cfg.rotation_prior_deg, cfg.translation_prior_fraction, cfg.max_nfev, cfg.irls_steps) <= 0:
        raise ValueError("robust radius, prior scales and optimizer limits must be positive")
    appearance = appearance_result or {}
    states = np.asarray([str(s).upper() for s in appearance.get("states", ["UNKNOWN"] * 8)])
    if states.shape != (8,):
        raise ValueError("appearance_result.states must have eight entries")
    visibility = _array_field(appearance, "visibility_score", 0.)
    boundary = _array_field(appearance, "boundary_score", 0.)
    accuracy = _array_field(appearance, "local_accuracy_score", 1.)
    independent = np.asarray(appearance.get("independent", np.ones(8, dtype=bool)), dtype=bool)
    if independent.shape != (8,):
        raise ValueError("appearance_result.independent must have eight entries")
    finite = np.isfinite(points).all(axis=1) & ~np.all(points == -1., axis=1)
    excluded_state = np.isin(states, ["SELF_OCCLUDED", "SELF_HIDDEN", "OUT_OF_FRAME", "TRUNCATED", "OCCLUSION_SUSPECTED", "EXTERNAL_OCCLUDED"])
    eligible = finite & independent & ~hidden & ~excluded_state
    visible_state = np.isin(states, ["VISIBLE_BOUNDARY", "VISIBLE_INTERIOR"])
    visible = eligible & visible_state & (visibility >= cfg.visible_threshold) & (accuracy >= cfg.minimum_accuracy)
    easy = visible & (states == "VISIBLE_BOUNDARY") & (visibility >= cfg.easy_visibility) & (boundary >= cfg.easy_boundary) & (accuracy >= cfg.easy_accuracy)
    unknown = eligible & (states == "UNKNOWN")
    weights = np.zeros(8)
    weights[visible] = np.maximum(visibility[visible], .05) * np.maximum(accuracy[visible], .05)
    weights[visible & (states == "VISIBLE_INTERIOR")] *= cfg.interior_weight
    weights[unknown] = cfg.unknown_weight * np.maximum(accuracy[unknown], .05)
    visible_ids = np.flatnonzero(visible).tolist()
    initial_projected = project(xyz, initial_R, initial_t, K)
    R, t = initial_R.copy(), initial_t.copy()
    stage0 = {"name": "initial", "accepted": True, "reason": "supplied_initial_pose",
              "used_ids": [], "R": R.tolist(), "t": t.tolist(), "projected": initial_projected.tolist()}
    stages = [stage0]
    accepted_count = 0
    final_used = []
    visible_geometry = _geometry(xyz, points, visible_ids, R, t, K, cfg)

    def residual_for(pose_R, pose_t, ids):
        return np.linalg.norm(project(xyz, pose_R, pose_t, K)[ids] - points[ids], axis=1)

    def attempt(name, ids):
        nonlocal R, t, accepted_count, final_used
        ids = sorted(ids)
        heldout = sorted(set(visible_ids) - set(ids))
        stage = {"name": name, "used_ids": ids, "added_ids": sorted(set(ids) - set(final_used)),
                 "heldout_ids": heldout, "weights": weights[ids].tolist(), "accepted": False,
                 "pose_before_R": R.tolist(), "pose_before_t": t.tolist()}
        geom = _geometry(xyz, points, ids, R, t, K, cfg)
        stage["geometry"] = geom
        if not geom["usable"]:
            stage["reason"] = geom["reason"]
            stage.update(R=R.tolist(), t=t.tolist(), projected=project(xyz, R, t, K).tolist())
            stages.append(stage)
            return
        before = residual_for(R, t, visible_ids)
        score_before = _huber_score(before, weights[visible_ids], cfg.robust_px)
        stage.update(visible_score_before=score_before, residual_fit_before=residual_for(R, t, ids).tolist(),
                     residual_visible_before=before.tolist(), residual_heldout_before=residual_for(R, t, heldout).tolist())
        try:
            candidate_R, candidate_t, optimizer = _fit(xyz, points, K, initial_R, initial_t, R, t,
                                                        ids, weights[ids], cfg)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, cv2.error) as exc:
            stage["reason"] = "optimizer_failure"
            stage["error"] = str(exc)
            stage.update(R=R.tolist(), t=t.tolist(), projected=project(xyz, R, t, K).tolist())
            stages.append(stage)
            return
        projected = project(xyz, candidate_R, candidate_t, K)
        after = residual_for(candidate_R, candidate_t, visible_ids)
        score_after = _huber_score(after, weights[visible_ids], cfg.robust_px)
        rot_change = _rotation_angle(candidate_R @ initial_R.T)
        translation_change = float(np.linalg.norm(candidate_t - initial_t) / max(np.linalg.norm(initial_t), 1e-6))
        shift = float(np.max(np.linalg.norm(projected - initial_projected, axis=1)))
        candidate_geometry = _geometry(xyz, points, ids, candidate_R, candidate_t, K, cfg)
        depth = ((candidate_R @ xyz.T).T + candidate_t)[:, 2]
        reason = "accepted"
        if not np.isfinite(projected).all() or not np.isfinite(candidate_t).all() or (depth <= 1e-6).any():
            reason = "invalid_camera_pose"
        elif rot_change > cfg.maximum_rotation_deg or translation_change > cfg.maximum_translation_fraction or shift > cfg.maximum_projection_shift_px:
            reason = "exceeds_initial_pose_motion_limit"
        elif not candidate_geometry["usable"]:
            reason = "unstable_candidate_geometry"
        elif score_after > score_before + cfg.validation_tolerance_px2:
            reason = "independent_visible_evidence_worsened"
        elif score_before - score_after <= cfg.minimum_gain_px2:
            reason = "no_independent_visible_gain"
        # Protect independently held-out visible corners when a stage has any.
        elif heldout and _huber_score(residual_for(candidate_R, candidate_t, heldout), weights[heldout], cfg.robust_px) > _huber_score(residual_for(R, t, heldout), weights[heldout], cfg.robust_px) + cfg.robust_px ** 2 / 4:
            reason = "heldout_visible_evidence_worsened"
        stage.update(reason=reason, optimizer=optimizer, visible_score_after=score_after,
                     residual_fit_after=residual_for(candidate_R, candidate_t, ids).tolist(),
                     residual_visible_after=after.tolist(), residual_heldout_after=residual_for(candidate_R, candidate_t, heldout).tolist(),
                     rotation_change_deg=rot_change, translation_fraction=translation_change,
                     maximum_projection_shift_px=shift, candidate_geometry=candidate_geometry,
                     candidate_R=candidate_R.tolist(), candidate_t=candidate_t.tolist())
        if reason == "accepted":
            R, t = candidate_R, candidate_t
            accepted_count += 1
            final_used = ids
            stage["accepted"] = True
        stage.update(R=R.tolist(), t=t.tolist(), projected=project(xyz, R, t, K).tolist())
        stages.append(stage)

    attempt("easy_visible_boundary", np.flatnonzero(easy).tolist())
    attempt("add_visible_interior", visible_ids)
    # Only independent visible support makes the unknown augmentation eligible.
    gate = None
    unknown_ids = []
    if cfg.enable_unknown and visible_geometry["usable"] and accepted_count:
        residual_visible = residual_for(R, t, visible_ids)
        median = float(np.median(residual_visible))
        mad = float(np.median(np.abs(residual_visible - median)))
        gate = float(np.clip(median + 3 * 1.4826 * mad, cfg.unknown_gate_min_px, cfg.unknown_gate_max_px))
        unknown_residual = np.linalg.norm(project(xyz, R, t, K) - points, axis=1)
        unknown_ids = np.flatnonzero(unknown & (unknown_residual <= gate)).tolist()
        if unknown_ids:
            attempt("add_consistent_unknown", sorted(set(visible_ids) | set(unknown_ids)))
    projected = project(xyz, R, t, K)
    diagnostic_ids = visible_ids or np.flatnonzero(eligible).tolist()
    before = residual_for(initial_R, initial_t, diagnostic_ids)
    after = residual_for(R, t, diagnostic_ids)
    fallback_reason = None if accepted_count else stages[-1]["reason"]
    return {"available": True, "accepted": bool(accepted_count), "new_pose_estimated": bool(accepted_count),
            "R": R.tolist(), "t": t.tolist(), "R_cf": R.tolist(), "centroid": t.tolist(),
            "projected": projected.tolist(), "observed_points": _finite_list(points),
            "used_ids": final_used, "visible_ids": visible_ids, "easy_ids": np.flatnonzero(easy).tolist(),
            "unknown_added_ids": unknown_ids if stages[-1]["name"] == "add_consistent_unknown" and stages[-1]["accepted"] else [],
            "self_hidden_ids": np.flatnonzero(hidden).tolist(), "excluded_ids": np.flatnonzero(~eligible).tolist(),
            "stages": stages, "fallback_reason": fallback_reason, "reason": "refined" if accepted_count else "initial_pose_preserved",
            "residual_before": float(before.mean()) if len(before) else None,
            "residual_after": float(after.mean()) if len(after) else None,
            "residual_ids": diagnostic_ids, "residuals_before_px": before.tolist(), "residuals_after_px": after.tolist(),
            "unknown_gate_px": gate, "hidden_projected": projected[hidden].tolist(),
            "configuration": asdict(cfg), "independent_observations_only": True,
            "reprojected_points_reused_as_observations": False, "corner_order_and_model_preserved": True}
