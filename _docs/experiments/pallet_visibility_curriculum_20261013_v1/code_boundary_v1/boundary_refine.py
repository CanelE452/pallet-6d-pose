"""Conservative independent RGB boundary observations for cuboid corners.

Two fitted material/background outline edges must independently support a
convex-hull corner. No pose reprojection is used as a coordinate observation.
An intersection can still be a virtual bbox corner, especially for rounded
or non-convex pallets; the method does not prove physical surface ownership.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

try:
    from .appearance import build_appearance_mask
except ImportError:
    from appearance import build_appearance_mask


DEFAULT_CONFIG = {
    "minimum_visibility": .70,
    "minimum_boundary_score": .30,
    "minimum_accuracy": .10,
    "normal_search_radius_px": 12,
    "normal_step_px": .5,
    "transition_probe_px": 2.5,
    "context_probe_px": 7.,
    "minimum_material_inside": .65,
    "maximum_material_outside": .35,
    "minimum_transition": .45,
    "maximum_context_outside": .45,
    "minimum_context_inside": .50,
    "sample_spacing_px": 4.,
    "minimum_samples": 9,
    "maximum_samples": 49,
    "along_interval": [.15, .85],
    "minimum_edge_length_px": 20.,
    "minimum_inlier_count": 6,
    "minimum_inlier_fraction": .60,
    "minimum_inlier_span_fraction": .55,
    "huber_delta_px": 1.25,
    "maximum_line_rms_px": 1.,
    "maximum_angle_change_deg": 10.,
    "minimum_intersection_sine": .25,
    "maximum_corner_shift_px": 6.,
    "appearance_config": None,
}


def _configuration(config):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown boundary-refine fields: {sorted(unknown)}")
        cfg.update(config)
    for key in ("normal_search_radius_px", "normal_step_px", "transition_probe_px",
                "context_probe_px", "sample_spacing_px", "maximum_corner_shift_px",
                "huber_delta_px", "maximum_line_rms_px"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    interval = np.asarray(cfg["along_interval"], float)
    if interval.shape != (2,) or not 0 < interval[0] < interval[1] < 1:
        raise ValueError("along_interval must be an increasing interior interval")
    return cfg


def _remap(mask, positions):
    return cv2.remap(mask, positions[..., 0].astype(np.float32),
                     positions[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(float) / 255.


def _fit_line(observations, confidence, t, seed_normal, total_samples, cfg):
    n = seed_normal.copy()
    b = -float(np.median(observations @ n))
    base = np.clip(confidence, .05, 1.)
    angle = 0.
    for _ in range(10):
        residual = observations @ n + b
        weights = base * np.minimum(1., cfg["huber_delta_px"] / np.maximum(np.abs(residual), 1e-9))
        center = np.average(observations, axis=0, weights=weights)
        delta = observations - center
        covariance = (delta * weights[:, None]).T @ delta / weights.sum()
        _, vectors = np.linalg.eigh(covariance)
        candidate = vectors[:, 0]
        if candidate @ seed_normal < 0:
            candidate = -candidate
        angle = float(np.degrees(np.arccos(np.clip(candidate @ seed_normal, -1, 1))))
        if angle > cfg["maximum_angle_change_deg"]:
            return None, {"reason": "image_line_direction_inconsistent", "angle_change_deg": angle}
        n, b = candidate, -float(candidate @ center)
    residual = np.abs(observations @ n + b)
    inlier = residual <= cfg["huber_delta_px"]
    count = int(inlier.sum())
    span = float(np.ptp(t[inlier])) / (cfg["along_interval"][1] - cfg["along_interval"][0]) if count else 0.
    rms = float(np.sqrt(np.mean(residual[inlier] ** 2))) if count else None
    diag = {"normal": n.tolist(), "offset": b, "inlier_count": count,
            "inlier_fraction": count / total_samples, "inlier_span_fraction": span,
            "inlier_rms_px": rms, "angle_change_deg": angle,
            "observed_points": observations.tolist(), "observation_confidence": confidence.tolist(),
            "inlier_mask": inlier.tolist()}
    if count < cfg["minimum_inlier_count"]:
        reason = "insufficient_image_transition_inliers"
    elif count / total_samples < cfg["minimum_inlier_fraction"]:
        reason = "insufficient_along_edge_coverage"
    elif span < cfg["minimum_inlier_span_fraction"]:
        reason = "insufficient_along_edge_span"
    elif rms is None or rms > cfg["maximum_line_rms_px"]:
        reason = "large_image_line_residual"
    else:
        reason = "supported_rgb_transition_line"
    diag["reason"] = reason
    if reason != "supported_rgb_transition_line":
        return None, diag
    return {"normal": n, "offset": b}, diag


def _observe_edge(mask, a, b, inward_normal, cfg):
    length = float(np.linalg.norm(b - a))
    if length < cfg["minimum_edge_length_px"]:
        return None, {"reason": "short_predicted_outline_edge", "length_px": length}
    samples = int(np.clip(math.ceil(length / cfg["sample_spacing_px"]) + 1,
                          cfg["minimum_samples"], cfg["maximum_samples"]))
    t = np.linspace(*cfg["along_interval"], samples)
    centers = a + t[:, None] * (b - a)
    offsets = np.arange(-cfg["normal_search_radius_px"],
                        cfg["normal_search_radius_px"] + cfg["normal_step_px"] / 2,
                        cfg["normal_step_px"])
    positions = centers[:, None, :] + offsets[None, :, None] * inward_normal
    # A positively oriented transition has target-like material inward and
    # background outward. Context probes reject many internal texture edges.
    probe = cfg["transition_probe_px"]
    inside = sum(_remap(mask, positions + inward_normal * probe * scale)
                 for scale in (.5, 1., 1.5)) / 3
    outside = sum(_remap(mask, positions - inward_normal * probe * scale)
                  for scale in (.5, 1., 1.5)) / 3
    context = cfg["context_probe_px"]
    inner_context = _remap(mask, positions + inward_normal * context)
    outer_context = _remap(mask, positions - inward_normal * context)
    fine_transition = (_remap(mask, positions + inward_normal * .5)
                       - _remap(mask, positions - inward_normal * .5))
    h, w = mask.shape
    # Image-border padding must not become artificial background evidence.
    margin = max(context, 1.5 * probe)
    pi, po = positions + inward_normal * margin, positions - inward_normal * margin
    valid = ((pi[..., 0] >= 0) & (pi[..., 0] < w - 1) & (pi[..., 1] >= 0) & (pi[..., 1] < h - 1)
             & (po[..., 0] >= 0) & (po[..., 0] < w - 1) & (po[..., 1] >= 0) & (po[..., 1] < h - 1))
    transition = inside - outside
    available = (valid & (inside >= cfg["minimum_material_inside"])
                 & (outside <= cfg["maximum_material_outside"])
                 & (transition >= cfg["minimum_transition"])
                 & (inner_context >= cfg["minimum_context_inside"])
                 & (outer_context <= cfg["maximum_context_outside"]))
    # The small distance penalty resolves equal plateaus and nearby textures;
    # the image transition remains the dominant term.
    score = (.65 * transition + .50 * fine_transition
             + .15 * (inner_context - outer_context) - .015 * np.abs(offsets)[None, :])
    score = np.where(available, score, -np.inf)
    choice = np.argmax(score, axis=1)
    best = score[np.arange(samples), choice]
    usable = np.isfinite(best)
    diag = {"length_px": length, "samples": samples,
            "transition_sample_count": int(usable.sum()),
            "inward_normal": inward_normal.tolist()}
    if usable.sum() < cfg["minimum_inlier_count"]:
        return None, {**diag, "reason": "insufficient_material_background_transitions"}
    observations = positions[np.arange(samples)[usable], choice[usable]]
    confidence = transition[np.arange(samples)[usable], choice[usable]]
    line, fit = _fit_line(observations, confidence, t[usable], inward_normal, samples, cfg)
    return line, {**diag, **fit}


def infer_rgb_boundary_points(image_bgr, points, self_hidden, appearance_result, config=None):
    """Refine eligible outer corners from two independent RGB line families.

    Output points preserve the original 8/9 slot contract. Only IDs listed in
    ``observed_refined_ids`` moved; missing and out-of-frame inputs are retained.
    Multiple corner intersections can share a fitted edge and are correlated.
    """
    cfg = _configuration(config)
    image = np.asarray(image_bgr)
    original = np.asarray(points, dtype=np.float64)
    hidden = np.asarray(self_hidden, bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image [H,W,3]")
    if original.shape not in ((8, 2), (9, 2)) or hidden.shape != (8,):
        raise ValueError("Expected points [8,2] or [9,2] and self_hidden [8]")
    q, result = original[:8].copy(), original.copy()
    states = appearance_result.get("states", [])
    vis = np.asarray(appearance_result.get("visibility_score", []), float)
    bound = np.asarray(appearance_result.get("boundary_score", []), float)
    accuracy = np.asarray(appearance_result.get("local_accuracy_score", []), float)
    if len(states) != 8 or vis.shape != (8,) or bound.shape != (8,) or accuracy.shape != (8,):
        raise ValueError("Expected eight-corner appearance states and score arrays")
    h, w = image.shape[:2]
    finite = np.isfinite(q).all(1) & ~np.all(q == -1, axis=1)
    inside = finite & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    eligible = (inside & ~hidden & np.isfinite(vis) & np.isfinite(bound) & np.isfinite(accuracy)
                & (vis >= cfg["minimum_visibility"]) & (bound >= cfg["minimum_boundary_score"])
                & (accuracy >= cfg["minimum_accuracy"])
                & np.array([s == "VISIBLE_BOUNDARY" for s in states]))
    records = [{"corner": k, "eligible": bool(eligible[k]), "refined": False,
                "reason": "no_two_independent_outline_edges" if eligible[k] else "not_eligible_rgb_boundary"}
               for k in range(8)]
    ids = np.flatnonzero(finite)
    hull_ids = ids[cv2.convexHull(q[ids].astype(np.float32), returnPoints=False).reshape(-1)].tolist() if len(ids) >= 3 else []
    edges, lines, refined = [], [], []
    if len(hull_ids) >= 3 and eligible.any():
        mask = build_appearance_mask(image, q, hidden, cfg["appearance_config"])
        center = q[hull_ids].mean(axis=0)
        for edge, first in enumerate(hull_ids):
            second = hull_ids[(edge + 1) % len(hull_ids)]
            row = {"edge": edge, "endpoints": [first, second], "supported": False}
            if not (eligible[first] and eligible[second]):
                line = None
                row["reason"] = "ineligible_endpoint"
            else:
                a, b = q[first], q[second]
                delta = b - a
                norm = float(np.linalg.norm(delta))
                if norm <= 1e-8:
                    line = None
                    row["reason"] = "degenerate_predicted_edge"
                else:
                    inward = np.array([-delta[1], delta[0]]) / norm
                    if inward @ (center - (a + b) / 2) < 0:
                        inward = -inward
                    line, observation = _observe_edge(mask, a, b, inward, cfg)
                    row.update(observation)
                    row["supported"] = line is not None
            edges.append(row)
            lines.append(line)
        for position, k in enumerate(hull_ids):
            if not eligible[k]:
                continue
            incident = [(position - 1) % len(hull_ids), position]
            first, second = [lines[e] for e in incident]
            records[k]["incident_shared_edges"] = incident
            if first is None or second is None:
                continue
            matrix = np.stack([first["normal"], second["normal"]])
            sine = abs(float(np.linalg.det(matrix)))
            records[k]["intersection_sine"] = sine
            if sine < cfg["minimum_intersection_sine"]:
                records[k]["reason"] = "poor_image_line_intersection_condition"
                continue
            candidate = np.linalg.solve(matrix, -np.array([first["offset"], second["offset"]]))
            distance = float(np.linalg.norm(candidate - q[k]))
            records[k].update(candidate=candidate.tolist(), shift_px=distance)
            if not np.isfinite(candidate).all() or not 0 <= candidate[0] < w or not 0 <= candidate[1] < h:
                records[k]["reason"] = "invalid_rgb_intersection"
            elif distance > cfg["maximum_corner_shift_px"]:
                records[k]["reason"] = "rgb_intersection_exceeds_shift_limit"
            else:
                result[k] = candidate
                refined.append(k)
                records[k].update(refined=True, reason="observed_rgb_line_intersection")
    return {"points": result.tolist(), "observed_refined_ids": refined,
            "edge_records": edges, "corner_records": records,
            "hull_corner_ids": hull_ids,
            "observation_source": "rgb_material_background_line_intersection",
            "independent_image_evidence": True,
            "reprojected_points_used": False,
            "diagnostics": {"config": cfg, "GT_inputs": False,
                "fitted_edge_count": sum(row["supported"] for row in edges),
                "shared_edge_correlations": [row["endpoints"] for row in edges if row["supported"]],
                "limits": ["same_coloured_occluder_or_floor", "incorrect_material_palette",
                           "virtual_bbox_or_rounded_corner", "non_convex_outline",
                           "shared_line_correlated_corners"]}}
