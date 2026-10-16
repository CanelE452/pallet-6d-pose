"""Pure, opt-in joint S/B/I/O/T pose solver.

B/I coordinates and their covariance are immutable observations. O/T domains
are simultaneous hard inequalities, never point correspondences. S (and U)
has no image term at all. This module performs no image, model, dataset or GT
reads and does not depend on the older completion solvers.

Pixel halfplanes use ``a*x + b*y <= c``. A T domain is intersected with its
outside-canvas halfplane; an unknown side is a four-way union enumerated before
continuous optimization. Continuous canvas safety closure uses x<=0,
x>=width, y<=0 or y>=height; an optional margin moves those boundaries
outwards. A successful local solve is not a global feasibility proof.
"""
from __future__ import annotations

from itertools import combinations, permutations, product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


STATES = frozenset({"S", "B", "I", "O", "T", "U"})
_SIDES = ("left", "right", "top", "bottom")
_DEFAULTS = {
    "max_p3p_triples": 56,
    "global_rotation_starts": 24,
    "max_seed_nfev": 160,
    "max_joint_iterations": 250,
    "maximum_union_branches": 256,
    "positive_depth_m": 1e-5,
    "outside_canvas_margin_px": 0.0,
    "feasibility_tolerance": 1e-7,
    "optimizer_ftol": 1e-10,
    "maximum_visible_mahalanobis": 3.0,
    "analytic_jacobian": False,
    "huber_scale": 3.0,
    "ambiguity_objective_gap": 0.05,
    "ambiguity_rotation_deg": 0.25,
    "ambiguity_translation_fraction": 1e-3,
}


def project_native(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the supplied native ordering; no reindexing or image inference."""
    camera = np.asarray(xyz, dtype=float) @ np.asarray(R, dtype=float).T + t
    homogeneous = camera @ np.asarray(K, dtype=float).T
    denominator = homogeneous[:, 2]
    # Optimization can transiently cross z=0; a finite extension lets the
    # independent positive-depth inequality reject that iterate.
    safe = np.where(np.abs(denominator) < 1e-10,
                    np.where(denominator < 0, -1e-10, 1e-10), denominator)
    return homogeneous[:, :2] / safe[:, None], camera[:, 2]


def _pose_vector(R: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    return np.r_[Rotation.from_matrix(R).as_rotvec(), np.asarray(t) / scale]


def _pose(p: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    return Rotation.from_rotvec(p[:3]).as_matrix(), p[3:] * scale


def _records(records: Any) -> list[dict]:
    if isinstance(records, Mapping):
        if "native_corner" in records or "native_id" in records:
            return [dict(records)]
        out = []
        for native_id, value in records.items():
            record = dict(value)
            record.setdefault("native_corner", int(native_id))
            out.append(record)
        return out
    return [dict(record) for record in records]


def _id(record: Mapping) -> int:
    value = record.get("native_corner", record.get("native_id"))
    if isinstance(value, bool) or value is None or int(value) != value:
        raise ValueError("Every record needs an integer native_corner in [0, 7]")
    value = int(value)
    if not 0 <= value < 8:
        raise ValueError("native_corner must be in [0, 7]")
    return value


def _state(value: Any) -> str:
    value = str(getattr(value, "value", value))
    if value not in STATES:
        raise ValueError(f"Unknown final state {value!r}; expected S/B/I/O/T/U")
    return value


def _finite_array(value: Any, shape: tuple, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _inputs(xyz: Any, K: Any, states: Sequence, visible: Any, domains: Any,
            image_hw: Sequence, cfg: Mapping) -> tuple:
    xyz = _finite_array(xyz, (8, 3), "xyz_native")
    K = _finite_array(K, (3, 3), "K")
    if np.linalg.matrix_rank(K) != 3 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must be nonsingular with positive focal lengths")
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("K must be a pinhole intrinsic matrix")
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) != 3:
        raise ValueError("The original native cuboid must have three-dimensional extent")
    if len(states) != 8:
        raise ValueError("Exactly eight final native states are required")
    states = tuple(_state(s) for s in states)
    if len(image_hw) != 2 or any(int(x) != x or int(x) <= 0 for x in image_hw):
        raise ValueError("image_hw must contain positive integer height and width")
    image_hw = tuple(int(x) for x in image_hw)
    measurements = []
    seen = set()
    for record in _records(visible):
        native_id = _id(record)
        if native_id in seen or states[native_id] not in {"B", "I"}:
            raise ValueError("Only unique final B/I corners may be visible measurements")
        seen.add(native_id)
        q = _finite_array(record["xy"], (2,), "visible xy")
        if not (0 <= q[0] < image_hw[1] and 0 <= q[1] < image_hw[0]):
            raise ValueError("Final B/I observations must lie inside the image canvas")
        covariance = _finite_array(record["covariance_px2"], (2, 2), "covariance_px2")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError("covariance_px2 must be symmetric")
        try:
            whitening = np.linalg.inv(np.linalg.cholesky(covariance))
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance_px2 must be positive definite") from exc
        protection = float(record["protection_px"])
        if not np.isfinite(protection) or protection <= 0:
            raise ValueError("protection_px must be a declared positive noisy-pixel budget")
        measurements.append({"native_corner": native_id, "xy": q,
                             "covariance_px2": covariance, "whitening": whitening,
                             "protection_px": protection,
                             "provenance": dict(record.get("provenance", {}))})
    expected_visible = {i for i, state in enumerate(states) if state in {"B", "I"}}
    if seen != expected_visible:
        raise ValueError("Every final B/I corner needs its corrected measurement")
    measurements.sort(key=lambda record: record["native_corner"])
    normalized_domains = []
    domain_ids = set()
    for index, record in enumerate(_records(domains)):
        native_id = _id(record)
        state = _state(record.get("state", states[native_id]))
        if state != states[native_id] or state not in {"O", "T"}:
            raise ValueError("Only final O/T corners may carry spatial domains")
        domain_ids.add(native_id)
        domain = {"native_corner": native_id, "state": state,
                  "domain_index": index, "shape": record.get("shape", "ellipse"),
                  "provenance": dict(record.get("provenance", {}))}
        shape = domain["shape"]
        if shape == "ellipse":
            center = record.get("center_xy", record.get("ellipse_center"))
            axes = record.get("half_axes_px", record.get("half_axes"))
            domain["center_xy"] = _finite_array(center, (2,), "ellipse center")
            domain["half_axes_px"] = _finite_array(axes, (2,), "ellipse half axes")
            if np.any(domain["half_axes_px"] <= 0):
                raise ValueError("ellipse half axes must be positive")
            angle = float(record.get("angle_deg", record.get("angle", 0)))
            if not np.isfinite(angle):
                raise ValueError("ellipse angle must be finite degrees")
            a = np.deg2rad(angle)
            domain["basis"] = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            domain["angle_deg"] = angle
        elif shape == "box":
            bounds = record.get("bounds_xyxy", record.get("box_xyxy"))
            domain["bounds_xyxy"] = _finite_array(bounds, (4,), "box bounds")
            if np.any(domain["bounds_xyxy"][2:] <= domain["bounds_xyxy"][:2]):
                raise ValueError("box upper bounds must exceed lower bounds")
        elif shape != "halfplane":
            raise ValueError("Spatial domain shape must be ellipse, box or halfplane")
        halfplanes = record.get("halfplanes", [])
        if "normal" in record:
            halfplanes = [*halfplanes, [*record["normal"], record["offset"]]]
        halfplanes = np.asarray(halfplanes, dtype=float).reshape((-1, 3))
        if not np.isfinite(halfplanes).all() or (len(halfplanes) and
                                                np.any(np.linalg.norm(halfplanes[:, :2], axis=1) <= 0)):
            raise ValueError("halfplanes must be finite nonzero [a,b,c] for a*x+b*y<=c")
        domain["halfplanes"] = halfplanes.copy()
        side = record.get("outside_canvas_side")
        if isinstance(side, str) and side in {"unknown", "any"}:
            side = None
        if isinstance(side, (list, tuple)):
            side = tuple(dict.fromkeys(side))
            if not side or any(value not in _SIDES for value in side):
                raise ValueError("outside_canvas_side choices must be nonempty canvas sides")
        elif side is not None and side not in _SIDES:
            raise ValueError("outside_canvas_side must be left/right/top/bottom or None")
        if state != "T" and side is not None:
            raise ValueError("outside-canvas side belongs only to T")
        if shape == "halfplane" and state == "O" and not len(halfplanes):
            raise ValueError("An O halfplane domain needs at least one spatial inequality")
        domain["outside_canvas_side"] = side
        margin = float(record.get("outside_margin_px", cfg["outside_canvas_margin_px"]))
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("outside_margin_px must be finite and nonnegative")
        domain["outside_margin_px"] = margin
        if "canvas_hw" in record and tuple(record["canvas_hw"]) != image_hw:
            raise ValueError("T canvas_hw must agree with image_hw")
        normalized_domains.append(domain)
    expected_domains = {i for i, state in enumerate(states) if state in {"O", "T"}}
    if domain_ids != expected_domains:
        raise ValueError("Every final O/T corner needs a spatial domain")
    for key in ("positive_depth_m", "huber_scale", "maximum_visible_mahalanobis",
                "feasibility_tolerance", "optimizer_ftol"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not np.isfinite(cfg["outside_canvas_margin_px"]) or cfg["outside_canvas_margin_px"] < 0:
        raise ValueError("outside_canvas_margin_px must be nonnegative")
    return xyz, K, states, measurements, normalized_domains, image_hw


def _visible_residual(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                      measurements: list[dict], scale: float) -> np.ndarray:
    R, t = _pose(p, scale)
    q, _ = project_native(xyz, R, t, K)
    return np.concatenate([record["whitening"] @ (q[record["native_corner"]] - record["xy"])
                           for record in measurements])


def _objective(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
               measurements: list[dict], scale: float, huber_scale: float) -> float:
    residual = _visible_residual(p, xyz, K, measurements, scale).reshape((-1, 2))
    # Radial pseudo-Huber in frozen covariance coordinates. There is no O/T
    # averaged score, no initial-pose prior and no term involving S/U.
    squared = np.einsum("ij,ij->i", residual, residual)
    return float(np.sum(2 * huber_scale**2 * (np.sqrt(1 + squared / huber_scale**2) - 1)))


def _valid_rotation(R: np.ndarray) -> bool:
    return (np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            and abs(np.linalg.det(R) - 1) <= 1e-7)


def _same_pose(first: tuple, second: tuple, scale: float, rotation_tol: float = 1e-5,
               translation_tol: float = 1e-5) -> bool:
    relative = Rotation.from_matrix(first[0] @ second[0].T).magnitude()
    return bool(relative < rotation_tol and np.linalg.norm(first[1] - second[1]) / scale < translation_tol)


def _initial_poses(xyz: np.ndarray, K: np.ndarray, measurements: list[dict],
                   initializations: Any, cfg: Mapping) -> tuple[list[dict], dict]:
    """Enumerate finite corrected-only P3P/PnP roots, plus explicit/global starts."""
    import cv2  # Optional runtime dependency, only needed when this solver runs.

    ids = np.array([record["native_corner"] for record in measurements], dtype=int)
    pixels = np.array([record["xy"] for record in measurements], dtype=float)
    homogeneous = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    normalized = homogeneous[:, :2] / homogeneous[:, 2:]
    poses = []
    triples = list(combinations(range(len(ids)), 3))
    limit = int(cfg["max_p3p_triples"])
    searched = 0

    def add(name: str, R: Any, t: Any) -> None:
        R = _finite_array(R, (3, 3), "initial R")
        t = _finite_array(t, (3,), "initial t")
        if not _valid_rotation(R):
            raise ValueError("initial R must be a proper rotation in the original native basis")
        if not any(_same_pose((R, t), (old["R"], old["t"]), 1.0) for old in poses):
            poses.append({"name": name, "R": R, "t": t})

    for triple in triples[:max(0, limit)]:
        subset = np.array(triple)
        object_points = np.ascontiguousarray(xyz[ids[subset]], dtype=float)
        image_points = np.ascontiguousarray(normalized[subset], dtype=float)
        if (np.linalg.matrix_rank(object_points - object_points.mean(axis=0)) < 2
                or np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2):
            continue
        searched += 1
        try:
            count, rvecs, tvecs = cv2.solveP3P(object_points, image_points, np.eye(3), None,
                                            flags=cv2.SOLVEPNP_P3P)
            for root in range(count):
                R = cv2.Rodrigues(rvecs[root])[0]
                t = np.asarray(tvecs[root]).reshape(3)
                add(f"P3P_{','.join(str(ids[i]) for i in triple)}_ROOT_{root}", R, t)
        except cv2.error:
            continue
    if len(ids) >= 3:
        try:
            answer = cv2.solvePnPGeneric(np.ascontiguousarray(xyz[ids]), normalized, np.eye(3),
                                         None, flags=cv2.SOLVEPNP_SQPNP)
            for root, (rvec, tvec) in enumerate(zip(answer[1], answer[2])):
                add(f"SQPNP_ROOT_{root}", cv2.Rodrigues(rvec)[0], np.asarray(tvec).reshape(3))
        except cv2.error:
            pass
    for index, pose in enumerate(initializations or []):
        add(str(pose.get("name", f"SUPPLIED_START_{index}")), pose["R"], pose["t"])
    # Coarse global rotations are initializations only, without any trust veto.
    extent = np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2))
    normalized_span = max(float(np.linalg.norm(np.ptp(normalized, axis=0))), 0.02)
    depth = max(extent / normalized_span, 2 * extent)
    center_ray = np.r_[normalized.mean(axis=0), 1.0]
    rotations = []
    for order in permutations(range(3)):
        for signs in product([-1, 1], repeat=3):
            matrix = np.eye(3)[:, order] * np.array(signs)
            if np.linalg.det(matrix) > 0:
                rotations.append(matrix)
    for index, R in enumerate(rotations[:max(0, int(cfg["global_rotation_starts"]))]):
        add(f"GLOBAL_ROTATION_{index}", R, depth * center_ray - R @ xyz[ids].mean(axis=0))
    return poses, {"p3p_triples_possible": len(triples), "p3p_triples_searched": searched,
                   "p3p_enumeration_complete": limit >= len(triples),
                   "initialization_count": len(poses), "global_feasibility_proved": False,
                   "uses_only_corrected_BI_for_PnP": True,
                   "supplied_initializations_are_priors": False}


def _domain_slacks(domain: dict, xy: np.ndarray, side: str | None,
                   image_hw: tuple, margin: float) -> tuple[list[float], list[str]]:
    slacks, names = [], []
    if domain["shape"] == "ellipse":
        delta = domain["basis"].T @ (xy - domain["center_xy"])
        slacks.append(float(1 - np.sum((delta / domain["half_axes_px"])**2)))
        names.append("ellipse_inside")
    elif domain["shape"] == "box":
        bounds = domain["bounds_xyxy"]
        span = bounds[2:] - bounds[:2]
        slacks.extend(((xy - bounds[:2]) / span).tolist())
        slacks.extend(((bounds[2:] - xy) / span).tolist())
        names.extend(["box_left", "box_top", "box_right", "box_bottom"])
    for index, (a, b, c) in enumerate(domain["halfplanes"]):
        slacks.append(float((c - a * xy[0] - b * xy[1]) / np.hypot(a, b)))
        names.append(f"halfplane_{index}")
    if domain["state"] == "T":
        height, width = image_hw
        margin = domain["outside_margin_px"]
        slack = {"left": -margin - xy[0], "right": xy[0] - (width + margin),
                 "top": -margin - xy[1], "bottom": xy[1] - (height + margin)}[side]
        slacks.append(float(slack))
        names.append(f"T_outside_{side}")
    return slacks, names


def _constraint_values(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                       measurements: list[dict], domains: list[dict], sides: tuple,
                       reference_projection: np.ndarray, image_hw: tuple,
                       scale: float, cfg: Mapping) -> np.ndarray:
    R, t = _pose(p, scale)
    q, depth = project_native(xyz, R, t, K)
    values = ((depth - cfg["positive_depth_m"]) / scale).tolist()
    for record in measurements:
        native_id = record["native_corner"]
        values.append(float(1 - np.sum((q[native_id] - reference_projection[native_id])**2)
                            / record["protection_px"]**2))
        whitened = record["whitening"] @ (q[native_id] - record["xy"])
        values.append(float(1 - np.sum(whitened**2) / cfg["maximum_visible_mahalanobis"]**2))
    for domain, side in zip(domains, sides):
        slacks, _ = _domain_slacks(domain, q[domain["native_corner"]], side,
                                  image_hw, cfg["outside_canvas_margin_px"])
        values.extend(slacks)
    return np.asarray(values)


def _domain_audit(q: np.ndarray, domains: list[dict], sides: tuple, image_hw: tuple,
                  cfg: Mapping) -> list[dict]:
    records = []
    for domain, side in zip(domains, sides):
        xy = q[domain["native_corner"]]
        slacks, names = _domain_slacks(domain, xy, side, image_hw,
                                      cfg["outside_canvas_margin_px"])
        records.append({"native_corner": domain["native_corner"], "state": domain["state"],
                        "domain_index": domain["domain_index"], "shape": domain["shape"],
                        "projected_xy": xy.tolist(), "outside_canvas_side": side,
                        "inequalities": [{"name": name, "slack": slack,
                                          "passes": slack >= -cfg["feasibility_tolerance"]}
                                         for name, slack in zip(names, slacks)],
                        "feasible": bool(min(slacks) >= -cfg["feasibility_tolerance"]),
                        "minimum_slack": min(slacks)})
    return records


def _measurement_audit(q: np.ndarray, measurements: list[dict], reference: np.ndarray,
                       cfg: Mapping) -> list[dict]:
    records = []
    tolerance_factor = float(np.sqrt(1 + cfg["feasibility_tolerance"]))
    for measurement in measurements:
        native_id = measurement["native_corner"]
        residual = q[native_id] - measurement["xy"]
        shift = float(np.linalg.norm(q[native_id] - reference[native_id]))
        mahalanobis = float(np.linalg.norm(measurement["whitening"] @ residual))
        records.append({"native_corner": native_id, "measurement_xy": measurement["xy"].tolist(),
                        "projected_xy": q[native_id].tolist(),
                        "measurement_residual_px": float(np.linalg.norm(residual)),
                        "measurement_mahalanobis": mahalanobis,
                        "maximum_visible_mahalanobis": cfg["maximum_visible_mahalanobis"],
                        "measurement_budget_pass": mahalanobis <= cfg["maximum_visible_mahalanobis"] * tolerance_factor,
                        "seed_projection_shift_px": shift, "protection_px": measurement["protection_px"],
                        "protection_pass": shift <= measurement["protection_px"] * tolerance_factor,
                        "covariance_budget_is_calibrated_probability": False})
    return records


def _jacobian(function: Any, p: np.ndarray) -> np.ndarray:
    columns = []
    for index in range(len(p)):
        step = 1e-6 * max(1, abs(p[index]))
        delta = np.zeros(len(p)); delta[index] = step
        columns.append((function(p + delta) - function(p - delta)) / (2 * step))
    return np.stack(columns, axis=1)


def _rank(jacobian: np.ndarray) -> dict:
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = max(jacobian.shape) * max(float(singular[0]) if len(singular) else 0, 1) * 1e-8
    rank = int(np.sum(singular > threshold))
    return {"rank": rank, "parameter_count": 6, "singular_values": singular.tolist(),
            "condition": float(singular[0] / singular[-1]) if rank == 6 else None,
            "rank_proves_global_uniqueness": False}


def _cached_value_jacobian(evaluate: Any) -> Any:
    """Share one exact evaluation between SciPy's fun/jac callbacks."""
    cache = [None, None]

    def paired(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[1] = evaluate(p)
            cache[0] = np.array(p, copy=True)
        return cache[1]

    return paired


def solve_joint_ot_pnp(xyz: Any, K: Any, states: Sequence, visible: Any, domains: Any,
                       image_hw: Sequence, initializations: Any = None,
                       config: Mapping | None = None) -> dict:
    """Fit one rigid SE(3) pose to immutable B/I and *all* O/T domains.

    ``visible`` records require native_corner (native_id is accepted), xy,
    covariance_px2, protection_px. ``domains`` records require native_corner,
    state, shape and its spatial parameters. Unknown-side T is a union whose
    side branches are searched explicitly. Every result is JSON serializable.

    ``accepted=False`` exposes the corrected-only B/I seed as an explicitly
    infeasible fallback; it never silently claims that seed satisfies O/T.
    Caller-provided poses seed the search but impose no rotation/motion prior.
    Each B/I residual must also lie inside its declared covariance budget
    (default Mahalanobis norm<=3). This is a heuristic assumed-noise tolerance,
    not a calibrated probability or coverage assertion.
    """
    cfg = {**_DEFAULTS, **dict(config or {})}
    if not isinstance(cfg["analytic_jacobian"], (bool, np.bool_)):
        raise ValueError("analytic_jacobian must be a boolean")
    xyz, K, states, measurements, domains, image_hw = _inputs(
        xyz, K, states, visible, domains, image_hw, cfg)
    scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
    observed = [{"native_corner": record["native_corner"], "state": states[record["native_corner"]],
                 "xy": record["xy"].tolist(), "covariance_px2": record["covariance_px2"].tolist(),
                 "protection_px": record["protection_px"], "provenance": record["provenance"]}
                for record in measurements]
    result = {"schema": "joint_ot_pnp_v1", "accepted": False, "status": "unresolved",
              "R": None, "t": None, "projected_native": None, "rigid_projection8": None,
              "states": list(states), "visible_observations": observed,
              "reported_BI_coordinates_preserved": True, "fallback_reason": None,
              "corrected_BI_seed": None, "joint_constraint_audit": [], "candidates": [],
              "rank": None, "ambiguity": None,
              "provenance": {"GT_inputs": False, "image_reads": 0,
                             "S_image_terms": 0, "S_region_terms": 0,
                             "OT_used_as_precise_correspondences": False,
                             "all_OT_simultaneous": True, "initial_pose_prior": False,
                             "rotation_fixed_for_three_points": False,
                             "native_order_and_basis_preserved": True,
                             "B_I_measurements_and_covariance_frozen": True,
                             "B_I_absolute_covariance_residual_constraints": True,
                             "maximum_visible_mahalanobis": cfg["maximum_visible_mahalanobis"],
                             "covariance_budget_is_calibrated_probability": False,
                             "protection_reference": "corrected_BI_only_seed"}}
    # Even an unresolved result retains actual corrected B/I readouts, never a
    # prior pose's guessed S/O/T coordinates.
    result["reported_native"] = [None] * 8
    for record in measurements:
        result["reported_native"][record["native_corner"]] = record["xy"].tolist()
    if len(measurements) < 3:
        result["fallback_reason"] = "fewer_than_three_corrected_BI_for_pose_seed"
        return result
    starts, initialization_audit = _initial_poses(xyz, K, measurements, initializations, cfg)
    result["search"] = initialization_audit
    analytic = bool(cfg["analytic_jacobian"])
    if analytic:
        from . import joint_ot_jacobian_v2 as A
        residual_pair = _cached_value_jacobian(lambda p: A.visible_residual_and_jacobian(
            p, xyz, K, measurements, scale))
        objective_pair = _cached_value_jacobian(lambda p: A.visible_objective_and_gradient(
            p, xyz, K, measurements, scale, cfg["huber_scale"]))
        # SciPy's robust least-squares loss scales residual/Jacobian buffers
        # in place. Keep cached exact pairs intact across its fun/jac calls.
        residual = lambda p: residual_pair(p)[0].copy()
        residual_jacobian = lambda p: residual_pair(p)[1].copy()
        objective = lambda p: objective_pair(p)[0]
        objective_jacobian = lambda p: objective_pair(p)[1].copy()
    else:
        residual = lambda p: _visible_residual(p, xyz, K, measurements, scale)
        objective = lambda p: _objective(p, xyz, K, measurements, scale, cfg["huber_scale"])
    seeds = []
    for start in starts:
        p0 = _pose_vector(start["R"], start["t"], scale)
        try:
            answer = least_squares(residual, p0, loss="soft_l1", f_scale=cfg["huber_scale"],
                                   jac=residual_jacobian if analytic else "2-point",
                                   max_nfev=int(cfg["max_seed_nfev"]),
                                   xtol=1e-11, ftol=1e-11, gtol=1e-11)
            R, t = _pose(answer.x, scale)
            q, depth = project_native(xyz, R, t, K)
            if np.min(depth) < cfg["positive_depth_m"] or not np.isfinite(q).all():
                continue
            candidate = {"name": start["name"], "p": answer.x, "R": R, "t": t,
                         "q": q, "objective": objective(answer.x)}
            if not any(_same_pose((R, t), (old["R"], old["t"]), scale) for old in seeds):
                seeds.append(candidate)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not seeds:
        result["fallback_reason"] = "no_positive_depth_corrected_BI_seed_found"
        return result
    seeds.sort(key=lambda seed: (round(seed["objective"], 9), seed["name"]))
    seed = seeds[0]
    reference = seed["q"].copy()
    result["corrected_BI_seed"] = {"R": seed["R"].tolist(), "t": seed["t"].tolist(),
                                    "projected_native": reference.tolist(),
                                    "objective": seed["objective"], "name": seed["name"],
                                    "fitted_only_corrected_BI": True,
                                    "admissible_as_joint_fallback": False}
    seed_measurement_audit = _measurement_audit(reference, measurements, reference, cfg)
    result["corrected_BI_seed"]["visible_measurement_audit"] = seed_measurement_audit
    result["corrected_BI_seed"]["visible_measurement_budgets_pass"] = all(
        record["measurement_budget_pass"] for record in seed_measurement_audit)
    result["search"]["corrected_BI_local_minima"] = len(seeds)
    side_choices = [(list(domain["outside_canvas_side"]) if isinstance(domain["outside_canvas_side"], tuple)
                     else [domain["outside_canvas_side"]] if domain["outside_canvas_side"] else list(_SIDES))
                    if domain["state"] == "T" else [None] for domain in domains]
    branch_count = int(np.prod([len(choice) for choice in side_choices], dtype=int))
    result["search"]["T_union_branches_required"] = branch_count
    if branch_count > int(cfg["maximum_union_branches"]):
        result["fallback_reason"] = "T_union_branch_limit_exceeded_no_partial_joint_claim"
        return result
    branches = list(product(*side_choices))
    result["search"]["T_union_branches_searched"] = len(branches)
    solutions = []
    # Raw finite roots/global supplied poses also enter the joint optimizer.
    # Unconstrained B/I refinement must not erase a legitimate joint branch.
    joint_starts = [{"name": f"BI_SEED_{s['name']}", "R": s["R"], "t": s["t"]} for s in seeds]
    for start in starts:
        if not any(_same_pose((start["R"], start["t"]), (old["R"], old["t"]), scale)
                   for old in joint_starts):
            joint_starts.append(start)
    result["search"]["joint_initializations_per_branch"] = len(joint_starts)
    result["search"]["analytic_jacobian_used"] = analytic
    for branch_index, sides in enumerate(branches):
        if analytic:
            constraint_pair = _cached_value_jacobian(lambda p, sides=sides: A.constraint_values_and_jacobian(
                p, xyz, K, measurements, domains, sides, reference, image_hw, scale, cfg))
            constraints = lambda p: constraint_pair(p)[0].copy()
            constraint_jacobian = lambda p: constraint_pair(p)[1].copy()
            inequality = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
        else:
            constraints = lambda p: _constraint_values(p, xyz, K, measurements, domains, sides,
                                                       reference, image_hw, scale, cfg)
            inequality = {"type": "ineq", "fun": constraints}
        for start in joint_starts:
            p0 = _pose_vector(start["R"], start["t"], scale)
            try:
                answer = minimize(objective, p0, method="SLSQP",
                                  jac=objective_jacobian if analytic else None,
                                  constraints=[inequality],
                                  options={"maxiter": int(cfg["max_joint_iterations"]),
                                           "ftol": cfg["optimizer_ftol"], "disp": False})
                R, t = _pose(answer.x, scale)
                q, depth = project_native(xyz, R, t, K)
                # Independent final verification retains the original scalar
                # inequality implementation even with analytic callbacks.
                values = _constraint_values(answer.x, xyz, K, measurements, domains, sides,
                                            reference, image_hw, scale, cfg)
                feasible = bool(np.isfinite(values).all() and np.isfinite(q).all()
                                and np.min(values) >= -cfg["feasibility_tolerance"]
                                and _valid_rotation(R)
                                and np.min(depth) >= cfg["positive_depth_m"] - 1e-10)
                record = {"initialization": start["name"], "branch": branch_index,
                          "optimizer_success": bool(answer.success), "optimizer_message": str(answer.message),
                          "iterations": int(answer.nit), "joint_feasible": feasible,
                          "objective": objective(answer.x), "minimum_constraint_slack": float(np.min(values)),
                          "visible_measurement_audit": _measurement_audit(q, measurements, reference, cfg),
                          "domain_audit": _domain_audit(q, domains, sides, image_hw, cfg)}
                result["candidates"].append(record)
                if feasible:
                    record["R"] = R.tolist(); record["t"] = t.tolist()
                    if not any(_same_pose((R, t), (old["R"], old["t"]), scale) for old in solutions):
                        solutions.append({"R": R, "t": t, "q": q, "p": answer.x,
                                          "objective": record["objective"], "record": record,
                                          "sides": sides})
            except (ValueError, np.linalg.LinAlgError) as exc:
                result["candidates"].append({"initialization": start["name"], "branch": branch_index,
                                             "joint_feasible": False, "error": str(exc)})
    result["search"]["feasible_distinct_local_poses"] = len(solutions)
    result["search"]["joint_method"] = "SLSQP_continuous_SE3_all_OT_inequalities"
    result["search"]["global_feasibility_proved"] = False
    if not solutions:
        result["status"] = "no_joint_feasible_pose_found"
        result["fallback_reason"] = "all_searched_joint_branches_infeasible_or_not_converged"
        result["fallback_diagnostics"] = dict(
            corrected_BI_seed_failed_measurement_ids=[record["native_corner"] for record in seed_measurement_audit
                                                       if not record["measurement_budget_pass"]],
            absolute_BI_measurement_budgets_are_required=True,
            no_global_infeasibility_proof=True)
        # Expose seed violations for diagnosis, without using the seed as output.
        result["joint_constraint_audit"] = [
            {"T_branch": i, "domains": _domain_audit(reference, domains, sides, image_hw, cfg)}
            for i, sides in enumerate(branches)]
        return result
    solutions.sort(key=lambda solution: solution["objective"])
    chosen = solutions[0]
    R, t, q, p = chosen["R"], chosen["t"], chosen["q"], chosen["p"]
    audit = chosen["record"]["domain_audit"]
    visible_audit = _measurement_audit(q, measurements, reference, cfg)
    alternatives = []
    for other in solutions[1:]:
        rotation_deg = float(np.rad2deg(Rotation.from_matrix(other["R"] @ R.T).magnitude()))
        translation_fraction = float(np.linalg.norm(other["t"] - t) / scale)
        gap = float(other["objective"] - chosen["objective"])
        distinct = (rotation_deg > cfg["ambiguity_rotation_deg"]
                    or translation_fraction > cfg["ambiguity_translation_fraction"])
        if distinct and gap <= cfg["ambiguity_objective_gap"]:
            alternatives.append({"rotation_difference_deg": rotation_deg,
                                 "translation_difference_fraction": translation_fraction,
                                 "objective_gap": gap, "R": other["R"].tolist(), "t": other["t"].tolist()})
    visible_jacobian = residual_jacobian(p) if analytic else _jacobian(residual, p)
    constraints = lambda x: _constraint_values(x, xyz, K, measurements, domains, chosen["sides"],
                                               reference, image_hw, scale, cfg)
    values = constraints(p)
    active = np.flatnonzero(values <= 1e-5)
    joint_jacobian = visible_jacobian
    if len(active):
        active_jacobian = A.constraint_values_and_jacobian(p, xyz, K, measurements, domains,
            chosen["sides"], reference, image_hw, scale, cfg)[1] if analytic else _jacobian(constraints, p)
        joint_jacobian = np.vstack([visible_jacobian, active_jacobian[active]])
    visible_rank = _rank(visible_jacobian)
    joint_rank = _rank(joint_jacobian)
    underdetermined = joint_rank["rank"] < 6
    ambiguous = bool(alternatives or underdetermined)
    result.update({"accepted": True, "status": "joint_feasible_ambiguous" if ambiguous else "joint_feasible",
                   "R": R.tolist(), "t": t.tolist(), "projected_native": q.tolist(),
                   "rigid_projection8": q.tolist(), "joint_constraint_audit": audit,
                   "visible_protection_audit": visible_audit, "objective": chosen["objective"],
                   "minimum_joint_slack": chosen["record"]["minimum_constraint_slack"],
                   "rank": {"visible": visible_rank, "visible_and_active_joint": joint_rank,
                            "active_constraint_count": len(active)},
                   "ambiguity": {"detected": ambiguous, "alternatives": alternatives,
                                 "local_underdetermined": underdetermined,
                                 "global_uniqueness_proved": False,
                                 "finite_multistart_absence_is_not_uniqueness_proof": True},
                   "output_corner_sources": ["IMMUTABLE_BI_OBSERVATION_AND_RIGID_PROJECTION" if state in {"B", "I"}
                                             else "RIGID_MODEL_PROJECTION_ONLY" for state in states]})
    reported = q.copy()
    for record in measurements:
        reported[record["native_corner"]] = record["xy"]
    result["reported_native"] = reported.tolist()
    result["mixed_readout_is_rigid_pose"] = False
    return result
