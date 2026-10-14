"""Pure continuous S/B/I/O/T pose inference with uncertain B/I observations.

Observed B/I coordinates and their conditional covariance are never changed.
A separately supplied CAL total-error covariance is used for a radial robust
loss. There is no hard B/I residual or seed-projection budget. A coarse pose
prior regularizes camera-tangent rotation, center-ray and log-depth residuals.
All O/T regions remain simultaneous individual hard constraints. S/U provide
no image observations, residuals, regions, scale estimates or initialization.

This new module imports frozen numerical helpers without modifying them. It
performs no image, model, annotation, GT, network or filesystem reads.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import joint_ot_pnp as V3
from . import joint_ot_jacobian_v2 as J


_DEFAULTS = {
    "max_p3p_triples": 56, "global_rotation_starts": 24,
    "max_seed_nfev": 160, "max_joint_iterations": 250,
    "maximum_union_branches": 256, "positive_depth_m": 1e-5,
    "outside_canvas_margin_px": 0.0, "feasibility_tolerance": 1e-7,
    "optimizer_ftol": 1e-10, "analytic_jacobian": True,
    "enforce_visible_geometry": True, "geometry_tolerance_deg": 2.0,
    "enforce_visible_canvas": True, "huber_scale": 3.0,
    "covariance_scale": 1.0,
    "state_BI_total_error_floor_px2": {"B": [[0., 0.], [0., 0.]],
                                       "I": [[0., 0.], [0., 0.]]},
    "state_loss_weights": {"B": 1.0, "I": 1.0},
    "prior_weight": 1.0, "prior_covariance6": None,
    "prior_rotation_sigma_deg": 10.0, "prior_ray_sigma": 0.03,
    "prior_log_depth_sigma": 0.15,
    "ambiguity_objective_gap": 0.05, "ambiguity_rotation_deg": 0.25,
    "ambiguity_translation_fraction": 1e-3,
}


def _configuration(config: Mapping | None) -> dict:
    supplied = dict(config or {})
    unknown = set(supplied) - set(_DEFAULTS)
    if unknown:
        raise ValueError("Unknown v4 pose config fields: " + ", ".join(sorted(unknown)))
    cfg = {**_DEFAULTS, **supplied}
    for key in ("analytic_jacobian", "enforce_visible_geometry", "enforce_visible_canvas"):
        if not isinstance(cfg[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
    for key in ("max_p3p_triples", "global_rotation_starts", "max_seed_nfev",
                "max_joint_iterations", "maximum_union_branches"):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["max_seed_nfev"] == 0 or cfg["max_joint_iterations"] == 0:
        raise ValueError("Optimization iteration limits must be positive")
    for key in ("positive_depth_m", "feasibility_tolerance", "optimizer_ftol",
                "huber_scale", "covariance_scale", "prior_rotation_sigma_deg",
                "prior_ray_sigma", "prior_log_depth_sigma"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("outside_canvas_margin_px", "prior_weight", "ambiguity_objective_gap",
                "ambiguity_rotation_deg", "ambiguity_translation_fraction"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not np.isfinite(cfg["geometry_tolerance_deg"]) or not 0 <= cfg["geometry_tolerance_deg"] < 90:
        raise ValueError("geometry_tolerance_deg must lie in [0,90)")
    for name in ("state_BI_total_error_floor_px2", "state_loss_weights"):
        if set(cfg[name]) != {"B", "I"}:
            raise ValueError(f"{name} must declare exactly B and I")
    for state in ("B", "I"):
        floor = V3._finite_array(cfg["state_BI_total_error_floor_px2"][state], (2, 2), "CAL covariance floor")
        if not np.allclose(floor, floor.T, atol=1e-12, rtol=0) or np.min(np.linalg.eigvalsh(floor)) < -1e-12:
            raise ValueError("CAL covariance floors must be symmetric positive semidefinite")
        weight = cfg["state_loss_weights"][state]
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("B/I reliability weights must be positive")
    return cfg


def _helper_config(cfg: Mapping) -> dict:
    # These two legacy values are required by frozen input/guard helpers. Their
    # residual-budget rows are explicitly removed below, never sent to SLSQP.
    return {**V3._DEFAULTS, **{k: v for k, v in cfg.items() if k in V3._DEFAULTS},
            "protect_seed_projection": False, "maximum_visible_mahalanobis": 3.0}


def _without_residual_budget_rows(measurements: Sequence, cfg: Mapping,
                                  row_count: int) -> np.ndarray:
    remove = []
    index = 8
    for _ in measurements:
        remove.append(index)
        index += 1 + int(cfg["enforce_visible_geometry"]) + 4 * int(cfg["enforce_visible_canvas"])
    keep = np.ones(row_count, dtype=bool)
    keep[remove] = False
    return keep


def _constraints(p: np.ndarray, xyz: np.ndarray, K: np.ndarray, measurements: Sequence,
                 domains: Sequence, sides: Sequence, image_hw: Sequence,
                 scale: float, cfg: Mapping, analytic: bool = True):
    helper = _helper_config(cfg)
    if analytic:
        values, jacobian = J.constraint_values_and_jacobian(
            p, xyz, K, measurements, domains, sides, np.zeros((8, 2)), image_hw, scale, helper)
        keep = _without_residual_budget_rows(measurements, cfg, len(values))
        return values[keep], jacobian[keep]
    values = V3._constraint_values(p, xyz, K, measurements, domains, sides,
                                 np.zeros((8, 2)), image_hw, scale, helper)
    return values[_without_residual_budget_rows(measurements, cfg, len(values))]


def _right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    squared = float(phi @ phi)
    omega = J.skew(phi)
    if squared < 1e-6:
        coefficient = 1 / 12 + squared / 720 + squared**2 / 30240
    else:
        angle = np.sqrt(squared)
        coefficient = (1 - .5 * angle / np.tan(.5 * angle)) / squared
    return np.eye(3) + .5 * omega + coefficient * (omega @ omega)


def _prior_residual_and_jacobian(p: np.ndarray, scale: float, center_xyz: np.ndarray,
                                coarse_R: np.ndarray, coarse_center: np.ndarray,
                                whitening: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R, t = V3._pose(p, scale)
    tangent = Rotation.from_matrix(R @ coarse_R.T).as_rotvec()
    rotation_jacobian = _right_jacobian_inverse(tangent) @ coarse_R @ J.so3_right_jacobian(p[:3])
    center = R @ center_xyz + t
    center_jacobian = np.column_stack([-R @ J.skew(center_xyz) @ J.so3_right_jacobian(p[:3]),
                                       np.eye(3) * scale])
    # A finite positive extension permits a transient invalid iterate. The
    # independent all-vertex depth guard rejects such an iterate as output.
    z = max(float(center[2]), 1e-10)
    dz = 1.0 if center[2] > 1e-10 else 0.0
    relative = np.r_[tangent, center[:2] / z - coarse_center[:2] / coarse_center[2],
                     np.log(z / coarse_center[2])]
    ray_jacobian = np.array([[1 / z, 0, -center[0] * dz / z**2],
                             [0, 1 / z, -center[1] * dz / z**2],
                             [0, 0, dz / z]])
    jacobian = np.vstack([np.column_stack([rotation_jacobian, np.zeros((3, 3))]),
                          ray_jacobian @ center_jacobian])
    return whitening @ relative, whitening @ jacobian


def _objective_pair(p: np.ndarray, xyz: np.ndarray, K: np.ndarray, measurements: Sequence,
                    scale: float, prior: Mapping, cfg: Mapping) -> tuple[float, np.ndarray, dict]:
    residual, jacobian = J.visible_residual_and_jacobian(p, xyz, K, measurements, scale)
    radial = residual.reshape((-1, 2))
    squared = np.einsum("ij,ij->i", radial, radial)
    weights = np.asarray([m["fit_reliability_weight"] for m in measurements])
    delta = cfg["huber_scale"]
    denominator = np.sqrt(1 + squared / delta**2)
    data_value = float(np.sum(weights * (2 * delta**2 * (denominator - 1))))
    data_gradient = (2 * weights[:, None] * radial / denominator[:, None]).reshape(-1) @ jacobian
    prior_residual, prior_jacobian = _prior_residual_and_jacobian(
        p, scale, xyz.mean(axis=0), prior["R"], prior["center"], prior["whitening"])
    prior_value = float(cfg["prior_weight"] * (prior_residual @ prior_residual))
    gradient = data_gradient + 2 * cfg["prior_weight"] * prior_residual @ prior_jacobian
    return data_value + prior_value, gradient, {
        "soft_BI_robust_loss": data_value, "coarse_pose_prior_loss": prior_value,
        "prior_whitened_residual": prior_residual.tolist(),
        "objective_uses_any_S_or_U": False}


def _visible_audit(q: np.ndarray, measurements: Sequence, camera: np.ndarray,
                   image_hw: Sequence, cfg: Mapping) -> list[dict]:
    height, width = image_hw
    result = []
    for m in measurements:
        native_id = m["native_corner"]
        residual = q[native_id] - m["xy"]
        cosine = V3._maximum_incident_cosine(camera, m)
        geometry = not cfg["enforce_visible_geometry"] or cosine + np.sin(np.deg2rad(cfg["geometry_tolerance_deg"])) >= -cfg["feasibility_tolerance"]
        inside = min(q[native_id, 0] / width, q[native_id, 1] / height,
                     (width - q[native_id, 0]) / width,
                     (height - q[native_id, 1]) / height) >= -cfg["feasibility_tolerance"]
        canvas = not cfg["enforce_visible_canvas"] or inside
        result.append({"native_corner": native_id, "measurement_xy": m["xy"].tolist(),
                       "observed_covariance_px2": m["observed_covariance_px2"].tolist(),
                       "fit_covariance_px2": m["covariance_px2"].tolist(),
                       "fit_covariance_source": m["fit_covariance_source"],
                       "fit_reliability_weight": m["fit_reliability_weight"],
                       "projected_xy": q[native_id].tolist(),
                       "measurement_residual_px": float(np.linalg.norm(residual)),
                       "measurement_mahalanobis": float(np.linalg.norm(m["whitening"] @ residual)),
                       "observed_conditional_mahalanobis": float(np.linalg.norm(m["observed_whitening"] @ residual)),
                       "hard_measurement_budget_active": False,
                       "seed_projection_guard_active": False,
                       "visible_geometry_pass": bool(geometry), "visible_canvas_pass": bool(canvas),
                       "visible_support_guards_pass": bool(geometry and canvas),
                       "maximum_incident_outward_ray_cosine": cosine,
                       "covariance_is_calibrated_coverage_probability": False})
    return result


def solve_uncertain_joint_ot_pnp(xyz: Any, K: Any, states: Sequence, visible: Any,
                                 domains: Any, image_hw: Sequence, coarse_pose: Mapping,
                                 initializations: Any = None, config: Mapping | None = None) -> dict:
    """Fit full SE(3) using soft B/I, a coarse prior and every O/T inequality.

    Inputs match v3's native observations/domains. ``coarse_pose`` requires R/t
    in the supplied native basis and optionally physical_registration_Q. Each
    visible record may supply CAL-only ``fit_covariance_px2`` and positive
    ``fit_reliability_weight``; neither changes the observed xy/covariance.
    Otherwise effective Sigma = covariance_scale*observed Sigma + the B/I
    state_BI_total_error_floor_px2. No calibration or GT access occurs here.

    Prior covariance ordering is camera rotation tangent radians[3], model
    center x/z and y/z, log center depth. Its default scales are declared
    generic assumptions, exposed for CAL-only selection, not fitted claims.
    accepted means a searched joint-feasible hypothesis, never proven unique.
    Two B/I with O/T is always reported as an ambiguous pose family.
    """
    cfg = _configuration(config)
    source_records = V3._records(visible)
    original = {V3._id(r): r for r in source_records}
    xyz, K, states, measurements, domains, image_hw = V3._inputs(
        xyz, K, states, source_records, domains, image_hw, _helper_config(cfg))
    normals = V3.registered_incident_normals(xyz)
    if any(n.shape != (3, 3) or not np.allclose(n @ n.T, np.eye(3), atol=1e-6, rtol=0) for n in normals):
        raise ValueError("Registered native geometry must be the original eight cuboid vertices")
    for m in measurements:
        record = original[m["native_corner"]]
        state = states[m["native_corner"]]
        observed_covariance = m["covariance_px2"].copy()
        if "fit_covariance_px2" in record:
            covariance = V3._finite_array(record["fit_covariance_px2"], (2, 2), "fit_covariance_px2")
            source = "CALLER_CAL_TOTAL_ERROR_COVARIANCE_OVERRIDE"
        else:
            covariance = (cfg["covariance_scale"] * observed_covariance
                          + np.asarray(cfg["state_BI_total_error_floor_px2"][state], dtype=float))
            source = "OBSERVED_CONDITIONAL_COVARIANCE_PLUS_CAL_STATE_FLOOR"
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError("fit covariance must be symmetric")
        try:
            whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        except np.linalg.LinAlgError as exc:
            raise ValueError("fit covariance must be positive definite") from exc
        weight = float(record.get("fit_reliability_weight", cfg["state_loss_weights"][state]))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("fit_reliability_weight must be positive")
        m.update(observed_covariance_px2=observed_covariance, observed_whitening=m["whitening"].copy(),
                 covariance_px2=covariance.copy(), whitening=whitening,
                 fit_reliability_weight=weight, fit_covariance_source=source)
    coarse_R = V3._finite_array(coarse_pose["R"], (3, 3), "coarse R")
    coarse_t = V3._finite_array(coarse_pose["t"], (3,), "coarse t")
    Q = V3._finite_array(coarse_pose.get("physical_registration_Q", np.eye(3)), (3, 3), "physical registration Q")
    if not V3._valid_rotation(coarse_R) or not V3._valid_rotation(Q):
        raise ValueError("Coarse rotation and physical registration Q must be proper rotations")
    if "R_physical" in coarse_pose and not np.allclose(coarse_R @ Q, coarse_pose["R_physical"], atol=1e-7, rtol=0):
        raise ValueError("Coarse physical rotation must equal R_native @ Q")
    coarse_center = coarse_R @ xyz.mean(axis=0) + coarse_t
    if coarse_center[2] <= 0:
        raise ValueError("The initial registered model center must have positive depth")
    covariance6 = cfg["prior_covariance6"]
    if covariance6 is None:
        deviations = np.r_[np.repeat(np.deg2rad(cfg["prior_rotation_sigma_deg"]), 3),
                           np.repeat(cfg["prior_ray_sigma"], 2), cfg["prior_log_depth_sigma"]]
        covariance6 = np.diag(deviations**2)
    covariance6 = V3._finite_array(covariance6, (6, 6), "prior_covariance6")
    if not np.allclose(covariance6, covariance6.T, atol=1e-12, rtol=0):
        raise ValueError("Prior covariance must be symmetric")
    try:
        prior_whitening = np.linalg.inv(np.linalg.cholesky(covariance6))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Prior covariance must be positive definite") from exc
    prior = {"R": coarse_R, "center": coarse_center, "whitening": prior_whitening}
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    observations = [{"native_corner": m["native_corner"], "state": states[m["native_corner"]],
                     "xy": m["xy"].tolist(), "covariance_px2": m["observed_covariance_px2"].tolist(),
                     "protection_px": m["protection_px"], "provenance": m["provenance"],
                     "fit_covariance_px2": m["covariance_px2"].tolist(),
                     "fit_reliability_weight": m["fit_reliability_weight"]} for m in measurements]
    coarse_q, _ = V3.project_native(xyz, coarse_R, coarse_t, K)
    result = {"schema": "statewise_uncertain_solver_v4", "accepted": False, "status": "unresolved",
              "R": None, "t": None, "R_physical": None, "states": list(states),
              "projected_native": None, "rigid_projection8": None, "visible_observations": observations,
              "reported_BI_coordinates_preserved": True, "reported_native": [None] * 8,
              "fallback_reason": None, "BI_prior_seed": None, "corrected_BI_seed": None,
              "candidates": [], "feasible_pose_samples": [], "pose_family_candidates": [],
              "joint_constraint_audit": [], "rank": None, "ambiguity": None,
              "OT_constraints_satisfied": False, "OT_feasible_hypothesis_available": False,
              "OT_unique_solution_claimed": False, "OT_solved": False,
              "prior_audit": {"parameter_order": ["camera_rotation_x_rad", "camera_rotation_y_rad",
                               "camera_rotation_z_rad", "center_x_over_z", "center_y_over_z", "log_center_depth"],
                              "covariance6": covariance6.tolist(), "prior_weight": cfg["prior_weight"],
                              "enabled": bool(cfg["prior_weight"] > 0),
                              "coarse_R": coarse_R.tolist(), "coarse_t": coarse_t.tolist(),
                              "physical_registration_Q": Q.tolist(), "default_scales_are_calibration_claims": False},
              "provenance": {"GT_inputs": False, "image_reads": 0, "S_image_terms": 0, "S_region_terms": 0,
                             "S_or_U_used_for_scale_or_initialization": False, "all_OT_simultaneous": True,
                             "OT_used_as_precise_correspondences": False,
                             "initial_pose_prior": bool(cfg["prior_weight"] > 0),
                             "coarse_pose_used_for_initialization": True,
                             "B_I_measurements_and_covariance_frozen": True,
                             "B_I_absolute_covariance_residual_constraints": False,
                             "seed_projection_protection_enabled": False,
                             "final_BI_model_geometry_enforced": cfg["enforce_visible_geometry"],
                             "final_BI_closed_canvas_enforced": cfg["enforce_visible_canvas"],
                             "rotation_fixed_for_three_points": False, "native_order_and_basis_preserved": True}}
    result["fallback_pose"] = {"R": coarse_R.tolist(), "t": coarse_t.tolist(),
                               "projected_native": coarse_q.tolist(), "source": "INITIAL_REGISTERED_COARSE_PNP",
                               "valid_as_joint_OT_solution": False, "S_projection_only": True}
    for m in measurements:
        result["reported_native"][m["native_corner"]] = m["xy"].tolist()
    for native_id, state in enumerate(states):
        if state == "S":
            result["reported_native"][native_id] = coarse_q[native_id].tolist()
    if len(measurements) < 2 or (len(measurements) == 2 and not domains):
        result["fallback_reason"] = "two_BI_requires_OT_ranges" if len(measurements) == 2 else "fewer_than_two_BI"
        return result
    pose_family = len(measurements) == 2
    supplied = [{"name": "REGISTERED_COARSE_PRIOR", "R": coarse_R, "t": coarse_t},
                *list(initializations or [])]
    starts, search = V3._initial_poses(xyz, K, measurements, supplied, cfg)
    result["search"] = {**search, "supplied_initializations_are_priors": False,
                        "registered_coarse_pose_is_prior": bool(cfg["prior_weight"] > 0), "analytic_jacobian_used": cfg["analytic_jacobian"],
                        "joint_method": "SLSQP_full_SE3_soft_BI_coarse_prior_all_OT_inequalities"}
    objective_pair = V3._cached_value_jacobian(lambda p: _objective_pair(p, xyz, K, measurements, scale, prior, cfg))
    objective = lambda p: objective_pair(p)[0]
    gradient = lambda p: objective_pair(p)[1].copy()
    analytic = cfg["analytic_jacobian"]

    def fit(start, subset_domains, sides, limit):
        pair = V3._cached_value_jacobian(lambda p: _constraints(p, xyz, K, measurements, subset_domains,
                                                               sides, image_hw, scale, cfg))
        inequality = {"type": "ineq", "fun": lambda p: pair(p)[0].copy()}
        if analytic:
            inequality["jac"] = lambda p: pair(p)[1].copy()
        answer = minimize(objective, V3._pose_vector(start["R"], start["t"], scale),
                          method="SLSQP", jac=gradient if analytic else None, constraints=[inequality],
                          options={"maxiter": int(limit), "ftol": cfg["optimizer_ftol"], "disp": False})
        R, t = V3._pose(answer.x, scale)
        q, depth = V3.project_native(xyz, R, t, K)
        values = _constraints(answer.x, xyz, K, measurements, subset_domains, sides, image_hw, scale, cfg, analytic=False)
        feasible = bool(np.isfinite(q).all() and np.isfinite(values).all()
                        and np.min(values) >= -cfg["feasibility_tolerance"]
                        and V3._valid_rotation(R) and np.min(depth) >= cfg["positive_depth_m"] - 1e-10)
        return answer, {"R": R, "t": t, "q": q, "p": answer.x, "name": start["name"],
                        "objective": objective(answer.x), "feasible": feasible, "sides": sides,
                        "minimum_constraint_slack": float(np.min(values))}

    seeds = []
    for start in starts:
        try:
            _, seed = fit(start, [], (), cfg["max_seed_nfev"])
            if seed["feasible"] and not any(V3._same_pose((seed["R"], seed["t"]), (s["R"], s["t"]), scale) for s in seeds):
                seeds.append(seed)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    seeds.sort(key=lambda s: (s["objective"], s["name"]))
    if seeds:
        seed = seeds[0]
        seed_rank = V3._rank(J.visible_residual_and_jacobian(seed["p"], xyz, K, measurements, scale)[1])
        underconstrained_seed = pose_family or seed_rank["rank"] < 6
        seed_record = {"R": seed["R"].tolist(), "t": seed["t"].tolist(), "projected_native": seed["q"].tolist(),
                       "objective": seed["objective"], "name": seed["name"],
                       "fitted_only_corrected_BI": bool(cfg["prior_weight"] == 0),
                       "fitted_BI_and_coarse_prior": bool(cfg["prior_weight"] > 0),
                       "coarse_prior_weight": cfg["prior_weight"], "visible_support_guards_pass": True,
                       "contains_underconstrained_pose_family": underconstrained_seed,
                       "admissible_as_joint_fallback": False, "visible_rank": seed_rank,
                       "visible_measurement_audit": _visible_audit(seed["q"], measurements, -seed["R"].T @ seed["t"], image_hw, cfg)}
        result["BI_prior_seed"] = seed_record
        result["corrected_BI_seed"] = seed_record
        if not underconstrained_seed:
            seed_source = ("SOFT_BI_AND_REGISTERED_COARSE_PRIOR_SEED" if cfg["prior_weight"] > 0
                           else "SOFT_BI_SEED_WITH_REGISTERED_COARSE_INITIALIZATION")
            result["fallback_pose"] = {**seed_record, "source": seed_source,
                                       "valid_as_joint_OT_solution": False, "S_projection_only": True}
            for i, state in enumerate(states):
                if state == "S":
                    result["reported_native"][i] = seed["q"][i].tolist()
    choices = [(list(d["outside_canvas_side"]) if isinstance(d["outside_canvas_side"], tuple)
                else [d["outside_canvas_side"]] if d["outside_canvas_side"] else list(V3._SIDES))
               if d["state"] == "T" else [None] for d in domains]
    branch_count = int(np.prod([len(c) for c in choices], dtype=int))
    result["search"].update(T_union_branches_required=branch_count, BI_prior_local_minima=len(seeds))
    if branch_count > cfg["maximum_union_branches"]:
        result["fallback_reason"] = "T_union_branch_limit_exceeded_no_partial_joint_claim"
        return result
    branches = list(product(*choices))
    joint_starts = [{"name": "BI_PRIOR_SEED_" + s["name"], "R": s["R"], "t": s["t"]} for s in seeds]
    for start in starts:
        if not any(V3._same_pose((start["R"], start["t"]), (s["R"], s["t"]), scale) for s in joint_starts):
            joint_starts.append(start)
    result["search"].update(T_union_branches_searched=len(branches), joint_initializations_per_branch=len(joint_starts))
    solutions = []
    for branch_index, sides in enumerate(branches):
        for start in joint_starts:
            try:
                answer, solved = fit(start, domains, sides, cfg["max_joint_iterations"])
                record = {"initialization": start["name"], "branch": branch_index,
                          "optimizer_success": bool(answer.success), "optimizer_message": str(answer.message),
                          "iterations": int(answer.nit), "joint_feasible": solved["feasible"],
                          "objective": solved["objective"], "objective_components": objective_pair(answer.x)[2],
                          "minimum_constraint_slack": solved["minimum_constraint_slack"],
                          "visible_measurement_audit": _visible_audit(solved["q"], measurements, -solved["R"].T @ solved["t"], image_hw, cfg),
                          "domain_audit": V3._domain_audit(solved["q"], domains, sides, image_hw, _helper_config(cfg))}
                result["candidates"].append(record)
                if solved["feasible"]:
                    record.update(R=solved["R"].tolist(), t=solved["t"].tolist())
                    if not any(V3._same_pose((solved["R"], solved["t"]), (s["R"], s["t"]), scale) for s in solutions):
                        solutions.append({**solved, "record": record})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                result["candidates"].append({"initialization": start["name"], "branch": branch_index,
                                             "joint_feasible": False, "error": str(exc)})
    result["search"].update(feasible_distinct_local_poses=len(solutions), global_feasibility_proved=False)
    if not solutions:
        result.update(status="no_joint_feasible_pose_found", fallback_reason="all_searched_joint_branches_infeasible_or_not_converged")
        result["fallback_diagnostics"] = {"no_global_infeasibility_proof": True, "hard_BI_residual_caps_used": False,
                                           "OT_output_unresolved": True, "fallback_source": result["fallback_pose"]["source"]}
        return result
    solutions.sort(key=lambda s: (s["objective"], s["name"]))
    chosen = solutions[0]
    R, t, q, p = chosen["R"], chosen["t"], chosen["q"], chosen["p"]
    alternatives = []
    for other in solutions[1:]:
        rotation = float(np.rad2deg(Rotation.from_matrix(other["R"] @ R.T).magnitude()))
        translation = float(np.linalg.norm(other["t"] - t) / scale)
        gap = other["objective"] - chosen["objective"]
        if (rotation > cfg["ambiguity_rotation_deg"] or translation > cfg["ambiguity_translation_fraction"]) and gap <= cfg["ambiguity_objective_gap"]:
            alternatives.append({"R": other["R"].tolist(), "t": other["t"].tolist(),
                                 "objective_gap": gap, "rotation_difference_deg": rotation,
                                 "translation_difference_fraction": translation})
    visible_jacobian = J.visible_residual_and_jacobian(p, xyz, K, measurements, scale)[1]
    values, guards_jacobian = _constraints(p, xyz, K, measurements, domains, chosen["sides"], image_hw, scale, cfg)
    active = np.flatnonzero(values <= 1e-5)
    joint_jacobian = np.vstack([visible_jacobian, guards_jacobian[active]])
    visible_rank, joint_rank = V3._rank(visible_jacobian), V3._rank(joint_jacobian)
    ambiguous = bool(pose_family or alternatives or joint_rank["rank"] < 6)
    result.update(accepted=True, status="joint_feasible_pose_family" if pose_family else "joint_feasible_ambiguous" if ambiguous else "joint_feasible",
                  R=R.tolist(), t=t.tolist(), R_physical=(R @ Q).tolist(), projected_native=q.tolist(), rigid_projection8=q.tolist(),
                  objective=chosen["objective"], objective_components=objective_pair(p)[2],
                  visible_protection_audit=chosen["record"]["visible_measurement_audit"],
                  joint_constraint_audit=chosen["record"]["domain_audit"], minimum_joint_slack=chosen["minimum_constraint_slack"],
                  rank={"visible": visible_rank, "visible_and_active_joint": joint_rank,
                        "active_constraint_count": len(active), "prior_not_counted_as_image_information": True},
                  ambiguity={"detected": ambiguous, "alternatives": alternatives, "pose_family_from_two_BI": pose_family,
                             "visible_data_underdetermined": visible_rank["rank"] < 6,
                             "local_underdetermined": joint_rank["rank"] < 6, "global_uniqueness_proved": False,
                             "prior_regularized_MAP_is_not_observation_uniqueness_proof": True},
                  OT_constraints_satisfied=True, OT_feasible_hypothesis_available=bool(domains),
                  OT_solved=bool(domains) and not ambiguous)
    result["feasible_pose_samples"] = [{"R": s["R"].tolist(), "t": s["t"].tolist(),
                                       "projected_native": s["q"].tolist(), "objective": s["objective"],
                                       "all_OT_individual_constraints_pass": True,
                                       "initialization": s["name"]} for s in solutions]
    result["pose_family_candidates"] = result["feasible_pose_samples"] if ambiguous else []
    for i, state in enumerate(states):
        if state in {"O", "T", "S"}:
            result["reported_native"][i] = q[i].tolist()
    result["output_corner_sources"] = ["IMMUTABLE_BI_OBSERVATION" if s in {"B", "I"}
        else "S_MODEL_PROJECTION_ONLY" if s == "S" else "UNRESOLVED" if s == "U"
        else "AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION" if ambiguous else "JOINT_FEASIBLE_REPRESENTATIVE_PROJECTION" for s in states]
    result["mixed_readout_is_rigid_pose"] = False
    camera = -R.T @ t
    self_ids = [i for i, normal in enumerate(normals) if np.max(normal @ (camera - xyz[i]) / np.linalg.norm(camera - xyz[i]))
                < -np.sin(np.deg2rad(cfg["geometry_tolerance_deg"]))]
    input_s = [i for i, state in enumerate(states) if state == "S"]
    result["final_model_visibility_audit"] = {"initial_S_ids": input_s, "final_model_self_ids": self_ids,
                                              "S_set_changed": set(input_s) != set(self_ids),
                                              "S_state_is_initial_geometry_conditional": True}
    return result
