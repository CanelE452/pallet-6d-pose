"""Inference-only easy-to-hard refinement of independently predicted corners.

The RGB classifier supplies evidence, never pose-created correspondences.  The
caller supplies the initial pose and its exact ordered object model; no dimension
swap, corner permutation, or oracle visibility is performed here.  Planar easy
sets retain the branch of that initial pose through a conservative local prior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CurriculumConfig:
    easy_visibility: float = 0.45
    visible_threshold: float = 0.55
    easy_boundary: float = 0.35
    easy_accuracy: float = 0.25
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
    staged_visible: bool = True
    # Exact derivative of the same residual; the frozen default uses SciPy's
    # original finite differences for reproducibility with cached first runs.
    analytic_jacobian: bool = False
    # Optional image-silhouette seed from the supplied initial cuboid. RGB
    # evidence, observations and loss weights remain unchanged.
    geometric_outline_priority: bool = False
    # Validate progressive stages on their active independent support, then
    # protect the final immutable visible pool against the supplied initial.
    progressive_active_support: bool = False


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


def _initial_prior_cost(R, t, xyz, initial_R, initial_t, cfg):
    """Same squared fixed-initial prior used by _fit, in pixel-loss units."""
    depth_scale = max(np.linalg.norm(initial_t), np.linalg.norm(np.ptp(xyz, axis=0)), 1e-6)
    delta_rotation = cv2.Rodrigues(R @ initial_R.T)[0].reshape(3)
    delta_translation = (t-initial_t)/depth_scale
    return float(cfg.prior_strength_px**2 * (
        np.sum((delta_rotation/np.deg2rad(cfg.rotation_prior_deg))**2)
        + np.sum((delta_translation/cfg.translation_prior_fraction)**2)))


def _aggregate_objective(xyz, points, K, R, t, initial_R, initial_t, ids, weights, cfg):
    """Frozen-weight SUM Huber plus fixed prior, normalised by weight SUM.

    IRLS's temporary optimiser weights are never validation weights. The same
    prior and depth scale apply to every intermediate and final checkpoint.
    """
    ids = np.asarray(ids, dtype=int)
    residuals = np.linalg.norm(project(xyz, R, t, K)[ids]-points[ids],axis=1)
    losses = np.where(residuals <= cfg.robust_px,residuals**2,
                      2*cfg.robust_px*residuals-cfg.robust_px**2)
    data_sum = float(np.sum(losses*weights[ids]))
    prior = _initial_prior_cost(R,t,xyz,initial_R,initial_t,cfg)
    total_weight = max(float(np.sum(weights[ids])),1e-12)
    return dict(objective=(data_sum+prior)/total_weight,
                data_objective=data_sum/total_weight,prior_cost_sum=prior,
                frozen_weight_sum=total_weight)


def _finite_list(value):
    value = np.asarray(value)
    if value.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in value]
    return [_finite_list(v) for v in value]


def _projection_jacobian(params, X0, K, initial_t, depth_scale):
    """Derivative of uv for R(delta) @ R_initial and depth-normalized delta-t.

    X0 is xyz @ R_initial.T. Rows follow the point-major residual order. Use
    the full supplied camera matrix, including skew, in the quotient rule.
    """
    rotation, derivative = cv2.Rodrigues(np.asarray(params[:3], dtype=float))
    d_rotation = derivative.reshape(3, 3, 3)
    camera = X0 @ rotation.T + initial_t + params[3:] * depth_scale
    homogeneous = camera @ K.T
    duv_dcamera = (
        K[None, :2, :] * homogeneous[:, 2, None, None]
        - homogeneous[:, :2, None] * K[None, 2:3, :]
    ) / homogeneous[:, 2, None, None] ** 2
    dcamera_drotation = np.einsum("pj,ikj->pki", X0, d_rotation)
    jacobian = np.empty((len(X0), 2, 6), dtype=float)
    jacobian[:, :, :3] = np.einsum("puk,pki->pui", duv_dcamera, dcamera_drotation)
    jacobian[:, :, 3:] = duv_dcamera * depth_scale
    return jacobian.reshape(-1, 6)


def _weighted_residual_jacobian(params, X0, K, initial_t, depth_scale, weights, cfg):
    derivative = _projection_jacobian(params, X0, K, initial_t, depth_scale)
    derivative *= np.repeat(np.sqrt(weights), 2)[:, None]
    prior = np.diag(np.r_[
        np.full(3, cfg.prior_strength_px / np.deg2rad(cfg.rotation_prior_deg)),
        np.full(3, cfg.prior_strength_px / cfg.translation_prior_fraction),
    ])
    return np.vstack([derivative, prior])


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

    X0 = X @ initial_R.T if cfg.analytic_jacobian else None

    def jac(params):
        return _weighted_residual_jacobian(params, X0, K, initial_t, depth_scale, robust_weights, cfg)

    nfev = 0
    solved = None
    for _ in range(cfg.irls_steps):
        solved = least_squares(fun, x, jac=jac if cfg.analytic_jacobian else "2-point",
                               method="trf", max_nfev=cfg.max_nfev,
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
    Remaining visible corners are added in evidence/difficulty order until four
    spatially diverse independent corners can establish a stable seed. Further
    visible corners, then gated low-weight unknowns, are attempted one at a time.
    Unknowns never establish a pose by themselves. All validation uses unchanged
    detector observations. Hidden projections are separate and never refitted.
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
    checkpoints = []
    visible_geometry = _geometry(xyz, points, visible_ids, R, t, K, cfg)

    def residual_for(pose_R, pose_t, ids):
        return np.linalg.norm(project(xyz, pose_R, pose_t, K)[ids] - points[ids], axis=1)

    def attempt(name, ids):
        nonlocal R, t, accepted_count, final_used
        ids = sorted(ids)
        heldout = sorted(set(visible_ids) - set(ids))
        protected = sorted(set(final_used) & set(visible_ids))
        active_validation = cfg.progressive_active_support and not name.startswith("add_consistent_unknown_")
        stage = {"name": name, "used_ids": ids, "added_ids": sorted(set(ids) - set(final_used)),
                 "heldout_ids": heldout, "protected_visible_ids": protected,
                 "weights": weights[ids].tolist(), "accepted": False,
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
        fit_before = residual_for(R, t, ids)
        fit_score_before = _huber_score(fit_before, weights[ids], cfg.robust_px)
        protected_before = _huber_score(residual_for(R, t, protected), weights[protected], cfg.robust_px) if protected else None
        stage.update(visible_score_before=score_before, residual_fit_before=residual_for(R, t, ids).tolist(),
                     residual_visible_before=before.tolist(), residual_heldout_before=residual_for(R, t, heldout).tolist(),
                     fit_score_before=fit_score_before, protected_score_before=protected_before)
        if cfg.progressive_active_support:
            stage["validation_policy"] = "progressive_active_support" if active_validation else "legacy_unknown_visible_guards"
            stage["active_objective_before"] = _aggregate_objective(
                xyz,points,K,R,t,initial_R,initial_t,ids,weights,cfg)
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
        fit_after = residual_for(candidate_R, candidate_t, ids)
        fit_score_after = _huber_score(fit_after, weights[ids], cfg.robust_px)
        protected_after = _huber_score(residual_for(candidate_R, candidate_t, protected), weights[protected], cfg.robust_px) if protected else None
        rot_change = _rotation_angle(candidate_R @ initial_R.T)
        translation_change = float(np.linalg.norm(candidate_t - initial_t) / max(np.linalg.norm(initial_t), 1e-6))
        shift = float(np.max(np.linalg.norm(projected - initial_projected, axis=1)))
        candidate_geometry = _geometry(xyz, points, ids, candidate_R, candidate_t, K, cfg)
        depth = ((candidate_R @ xyz.T).T + candidate_t)[:, 2]
        reason = "accepted"
        active_after = _aggregate_objective(xyz,points,K,candidate_R,candidate_t,
            initial_R,initial_t,ids,weights,cfg) if cfg.progressive_active_support else None
        if not np.isfinite(projected).all() or not np.isfinite(candidate_t).all() or (depth <= 1e-6).any():
            reason = "invalid_camera_pose"
        elif rot_change > cfg.maximum_rotation_deg or translation_change > cfg.maximum_translation_fraction or shift > cfg.maximum_projection_shift_px:
            reason = "exceeds_initial_pose_motion_limit"
        elif not candidate_geometry["usable"]:
            reason = "unstable_candidate_geometry"
        elif active_validation:
            if stage["active_objective_before"]["objective"]-active_after["objective"] <= cfg.minimum_gain_px2:
                reason = "no_active_aggregate_objective_gain"
        elif score_after > score_before + cfg.validation_tolerance_px2:
            reason = "independent_visible_evidence_worsened"
        elif protected and protected_after > protected_before + cfg.validation_tolerance_px2:
            reason = "previous_visible_support_worsened"
        elif fit_score_before - fit_score_after <= cfg.minimum_gain_px2:
            reason = "no_independent_fit_pool_gain"
        # Protect independently held-out visible corners when a stage has any.
        elif heldout and _huber_score(residual_for(candidate_R, candidate_t, heldout), weights[heldout], cfg.robust_px) > _huber_score(residual_for(R, t, heldout), weights[heldout], cfg.robust_px) + cfg.robust_px ** 2 / 4:
            reason = "heldout_visible_evidence_worsened"
        stage.update(reason=reason, optimizer=optimizer, visible_score_after=score_after,
                     residual_fit_after=fit_after.tolist(), fit_score_after=fit_score_after,
                     protected_score_after=protected_after,
                     residual_visible_after=after.tolist(), residual_heldout_after=residual_for(candidate_R, candidate_t, heldout).tolist(),
                     rotation_change_deg=rot_change, translation_fraction=translation_change,
                     maximum_projection_shift_px=shift, candidate_geometry=candidate_geometry,
                     candidate_R=candidate_R.tolist(), candidate_t=candidate_t.tolist())
        if cfg.progressive_active_support:
            stage["active_objective_after"] = active_after
            stage["heldout_and_protected_used_as_visible_veto"] = not active_validation
        if reason == "accepted":
            R, t = candidate_R, candidate_t
            accepted_count += 1
            final_used = ids
            stage["accepted"] = True
            if cfg.progressive_active_support:
                checkpoints.append(dict(stage_index=len(stages),name=name,R=R.copy(),t=t.copy(),used_ids=ids.copy()))
        stage.update(R=R.tolist(), t=t.tolist(), projected=project(xyz, R, t, K).tolist())
        stages.append(stage)

    easy_ids = np.flatnonzero(easy).tolist()
    # Difficulty is determined from RGB evidence, not residuals against truth.
    # Keep boundary seeds first; otherwise prefer spatially supported, locally
    # accurate visible observations before featureless interior observations.
    def priority(index):
        category = 0 if easy[index] else (1 if states[index] == "VISIBLE_BOUNDARY" else 2)
        return (category, -float(visibility[index] * accuracy[index]), index)
    if cfg.geometric_outline_priority:
        # The initial projection selects a difficulty category only. Image
        # samples and fit coordinates remain independent measured points.
        outline_ids = cv2.convexHull(initial_projected.astype(np.float32), returnPoints=False).reshape(-1).tolist()
        outline = np.isin(np.arange(8), outline_ids)
        outline_visible = sorted(k for k in visible_ids if outline[k])
        def priority(index):
            return (0 if outline[index] else 1, -float(visibility[index] * accuracy[index]), index)
        choices = []
        if len(visible_ids) >= 4:
            _, jacobian = cv2.projectPoints(np.ascontiguousarray(xyz[visible_ids]), cv2.Rodrigues(initial_R)[0], initial_t, K, None)
            jacobian = jacobian[:, :6]
            jacobian /= np.maximum(np.linalg.norm(jacobian, axis=0), 1e-12)[None]
            positions = {corner: index for index, corner in enumerate(visible_ids)}
            for subset in combinations(outline_visible, 4):
                geometry = _geometry(xyz, points, subset, initial_R, initial_t, K, cfg)
                if not geometry["usable"]:
                    continue
                rows = np.asarray([[2 * positions[k], 2 * positions[k] + 1] for k in subset]).reshape(-1)
                weighted = jacobian[rows] * np.repeat(np.sqrt(weights[list(subset)]), 2)[:, None]
                eigenvalues = np.linalg.eigvalsh(weighted.T @ weighted)
                choices.append((float(eigenvalues[0]), float(np.sum(weights[list(subset)])), subset, geometry))
        if choices:
            best = max(choices, key=lambda x: (x[0], x[1], tuple(-k for k in x[2])))
            easy_ids = list(best[2])
            seed_evidence = dict(seed=easy_ids, minimum_information_eigenvalue=best[0], geometry=best[3],
                                 valid_outline_four_point_subsets=len(choices))
        else:
            easy_ids = sorted(outline_visible, key=priority)
            seed_evidence = dict(seed=easy_ids, reason="no_usable_outline_four_point_subset_grow_outline_then_interior")
        stages[0]["initial_geometric_outline"] = outline.tolist()
        stages[0]["geometric_outline_seed"] = seed_evidence
        stages[0]["RGB_states_and_weights_preserved"] = True
    remaining = sorted(set(visible_ids) - set(easy_ids), key=priority)
    if cfg.staged_visible:
        attempt("easy_visible_boundary", easy_ids)
        seed = set(easy_ids)
        for index, corner in enumerate(remaining, 1):
            if accepted_count:
                attempt(f"add_visible_{index}", sorted(set(final_used) | {corner}))
            else:
                seed.add(corner)
                if len(seed) >= 4:
                    attempt(f"seed_easiest_supported_{len(seed)}", sorted(seed))
        # Record an insufficient support result even when the seed never reaches
        # four, so the fallback reason describes the full visible evidence.
        if not accepted_count and len(seed) < 4:
            attempt("seed_easiest_supported_insufficient", sorted(seed))
    else:
        attempt("add_visible_interior", visible_ids)
    # Only independent visible support makes the unknown augmentation eligible.
    gate = None
    unknown_ids = []
    if cfg.enable_unknown and visible_geometry["usable"] and accepted_count:
        supported_visible = sorted(set(final_used) & set(visible_ids))
        residual_visible = residual_for(R, t, supported_visible)
        median = float(np.median(residual_visible))
        mad = float(np.median(np.abs(residual_visible - median)))
        gate = float(np.clip(median + 3 * 1.4826 * mad, cfg.unknown_gate_min_px, cfg.unknown_gate_max_px))
        unknown_residual = np.linalg.norm(project(xyz, R, t, K) - points, axis=1)
        unknown_ids = np.flatnonzero(unknown & (unknown_residual <= gate)).tolist()
        for index, corner in enumerate(sorted(unknown_ids, key=lambda i:(unknown_residual[i], -accuracy[i], i)), 1):
            if np.linalg.norm(project(xyz, R, t, K)[corner] - points[corner]) <= gate:
                attempt(f"add_consistent_unknown_{index}", sorted(set(final_used) | {corner}))
    intermediate_accepted_count = accepted_count
    final_guard = None
    if cfg.progressive_active_support:
        reference = _aggregate_objective(xyz,points,K,initial_R,initial_t,
            initial_R,initial_t,visible_ids,weights,cfg)
        checkpoint_records = []
        selected = None
        for checkpoint in checkpoints:
            measured = _aggregate_objective(xyz,points,K,checkpoint["R"],checkpoint["t"],
                initial_R,initial_t,visible_ids,weights,cfg)
            gain = reference["objective"]-measured["objective"]
            passes = gain > cfg.minimum_gain_px2
            checkpoint_records.append(dict(stage_index=checkpoint["stage_index"],name=checkpoint["name"],
                active_ids=checkpoint["used_ids"],full_visible=measured,initial_objective_gain=gain,
                passes_initial_all_visible_guard=passes))
            if passes:
                selected = checkpoint  # Latest temporally passing, never minimum/GT-selected.
        rejected_ids = sorted({k for stage in stages[1:] if not stage["accepted"]
            for k in stage.get("added_ids",[]) if k in visible_ids})
        rejected_single_add_ids = sorted({k for stage in stages[1:]
            if not stage["accepted"] and stage["name"].startswith("add_visible_")
            for k in stage.get("added_ids",[]) if k in visible_ids})
        if selected is None:
            R,t = initial_R.copy(),initial_t.copy()
            final_used = []
            accepted_count = 0
            reason = "no_accepted_independent_checkpoint" if not checkpoints else "final_all_visible_initial_objective_no_gain"
            fallback_source = "supplied_initial_pose"
        else:
            R,t = selected["R"].copy(),selected["t"].copy()
            final_used = selected["used_ids"].copy()
            reason = "latest_checkpoint_passes_initial_all_visible_guard"
            fallback_source = "latest_accepted_checkpoint" if selected is checkpoints[-1] else "earlier_passing_checkpoint"
        chosen_index = selected["stage_index"] if selected is not None else None
        for stage_index,stage in enumerate(stages):
            stage["retained_pose_checkpoint"] = stage_index==chosen_index
            stage["intermediate_accept_discarded_by_final_guard"] = bool(stage_index and stage["accepted"]
                and (chosen_index is None or stage_index>chosen_index))
        final_guard = dict(policy="immutable_all_visible_initial_objective",all_visible_ids=visible_ids,
            initial=reference,checkpoints=checkpoint_records,retained_checkpoint_index=chosen_index,
            retained_active_ids=final_used,all_omitted_visible_ids=sorted(set(visible_ids)-set(final_used)),
            attempted_rejected_visible_ids=rejected_ids,rejected_single_add_ids=rejected_single_add_ids,
            rejected_additions_still_omitted_ids=sorted(set(rejected_single_add_ids)-set(final_used)),
            intermediate_accepted_count=intermediate_accepted_count,
            discarded_accepted_checkpoint_indices=[s["stage_index"] for s in checkpoint_records
                if chosen_index is None or s["stage_index"]>chosen_index],fallback_source=fallback_source,
            output_objective=_aggregate_objective(xyz,points,K,R,t,initial_R,initial_t,visible_ids,weights,cfg),
            final_batch_refit_performed=False,GT_stage_selection=False)
        stages.append(dict(name="final_all_visible_initial_guard",accepted=selected is not None,reason=reason,
            used_ids=final_used,fit_performed=False,R=R.tolist(),t=t.tolist(),projected=project(xyz,R,t,K).tolist(),
            fallback_source=fallback_source,full_visible_ids=visible_ids,retained_checkpoint_index=chosen_index))
    projected = project(xyz, R, t, K)
    diagnostic_ids = visible_ids or np.flatnonzero(eligible).tolist()
    before = residual_for(initial_R, initial_t, diagnostic_ids)
    after = residual_for(R, t, diagnostic_ids)
    fallback_reason = None if accepted_count else stages[-1]["reason"]
    output = {"available": True, "accepted": bool(accepted_count), "new_pose_estimated": bool(accepted_count),
            "R": R.tolist(), "t": t.tolist(), "R_cf": R.tolist(), "centroid": t.tolist(),
            "projected": projected.tolist(), "observed_points": _finite_list(points),
            "used_ids": final_used, "visible_ids": visible_ids, "easy_ids": np.flatnonzero(easy).tolist(),
            "unknown_added_ids": sorted(set(final_used) & set(unknown_ids)),
            "unknown_considered_ids": unknown_ids,
            "self_hidden_ids": np.flatnonzero(hidden).tolist(), "excluded_ids": np.flatnonzero(~eligible).tolist(),
            "stages": stages, "fallback_reason": fallback_reason, "reason": "refined" if accepted_count else "initial_pose_preserved",
            "residual_before": float(before.mean()) if len(before) else None,
            "residual_after": float(after.mean()) if len(after) else None,
            "residual_ids": diagnostic_ids, "residuals_before_px": before.tolist(), "residuals_after_px": after.tolist(),
            "unknown_gate_px": gate, "hidden_projected": projected[hidden].tolist(),
            "configuration": asdict(cfg), "independent_observations_only": True,
            "reprojected_points_reused_as_observations": False, "corner_order_and_model_preserved": True}
    if cfg.progressive_active_support:
        output.update(final_guard=final_guard,nominal_visible_order=list(easy_ids)+remaining,
            frozen_weights_by_id=weights.tolist(),intermediate_accepted_count=intermediate_accepted_count,
            final_omitted_visible_ids=final_guard["all_omitted_visible_ids"])
    return output
