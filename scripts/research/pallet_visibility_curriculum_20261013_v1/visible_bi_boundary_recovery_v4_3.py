"""Image-supported recovery of failed B junctions; the frozen v2 stays intact.

Only original fallback B measurements are eligible. A SAM exterior contour
sets search reach, then an observed line supplies an orthogonal search anchor.
Neither a projected vertex nor a region center becomes an image measurement.
This module contains no annotations, pose fitting, or data-dependent tuning.
"""
from copy import deepcopy

import cv2
import numpy as np

from . import common as C
from . import boundary_refine as F
from . import visible_bi_correction_v2 as B


DEFAULT_CONFIG = dict(recenter_on_observed_line=True, maximum_alpha=1.,
                      maximum_corner_shift_px=None, require_rgb_support=True)


def _configuration(config):
    cfg = dict(DEFAULT_CONFIG)
    if set(config or {}) - set(cfg):
        raise ValueError("Unknown boundary recovery configuration field")
    cfg.update(config or {})
    for name in ("recenter_on_observed_line", "require_rgb_support"):
        if not isinstance(cfg[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if not np.isfinite(cfg["maximum_alpha"]) or not 0 < cfg["maximum_alpha"] <= 1:
        raise ValueError("maximum_alpha must be finite in (0,1]")
    cap = cfg["maximum_corner_shift_px"]
    if cap is not None and (not np.isfinite(cap) or cap <= 0):
        raise ValueError("Optional maximum_corner_shift_px must be finite and positive")
    return cfg


def _external_mask_and_contours(mask):
    """Keep exterior contours; mask holes cannot supply incident silhouette lines."""
    contours, _ = cv2.findContours((mask >= .5).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    exterior = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(exterior, contours, -1, 1, cv2.FILLED)
    return exterior.astype(float), contours


def _exterior_distance(point, contours):
    if not contours:
        return None
    return min(abs(float(cv2.pointPolygonTest(c, tuple(map(float, point)), True)))
               for c in contours)


def _observed_line(image, mask, lab, gradients, q, k, mate, guide, center,
                   legacy_cfg, radius, require_rgb):
    cfg = dict(legacy_cfg, normal_search_radius_px=float(radius))
    row = B._line(image, mask, lab, gradients, q, k, mate, guide, center, "B", cfg)
    row.update(search_anchor_xy=q[k].tolist(), normal_search_radius_px=float(radius),
               shared_native_edge=sorted([k, mate]))
    if not row.get("supported") or not require_rgb:
        return row
    # Mask geometry alone is insufficient: retain actual RGB transition samples
    # and rerun exactly the same direction/count/fraction/span/RMS line guards.
    xy = np.asarray(row["observed_points"], float)
    normal = np.asarray(row["normal"], float)
    gx, gy = (B._sample(g, xy[:, None, :])[:, 0] for g in gradients)
    energy = np.linalg.norm(gx * normal[0] + gy * normal[1], axis=-1)
    magnitude = np.sqrt(np.sum(gx * gx + gy * gy, axis=-1))
    alignment = energy / np.maximum(magnitude, 1e-12)
    contrast = np.linalg.norm(B._sample(lab, (xy + normal * 1.5)[:, None, :])[:, 0]
                              - B._sample(lab, (xy - normal * 1.5)[:, None, :])[:, 0], axis=-1)
    keep = ((energy >= cfg["minimum_gradient"])
            & (alignment >= cfg["minimum_gradient_alignment"])
            & (contrast >= cfg["minimum_color_transition"]))
    row["rgb_supported_sample_count"] = int(keep.sum())
    row["mask_supported_sample_count"] = len(xy)
    row["rgb_support_thresholds"] = {name: cfg[name] for name in
        ("minimum_gradient", "minimum_gradient_alignment", "minimum_color_transition")}
    if keep.sum() < cfg["minimum_inlier_count"]:
        return dict(row, supported=False, reason="insufficient_actual_RGB_boundary_samples")
    direction = np.asarray(guide, float) / np.linalg.norm(guide)
    seed_normal = np.array([-direction[1], direction[0]])
    if seed_normal @ (center - q[k]) < 0:
        seed_normal = -seed_normal
    end = min(cfg["along_end_px"], .8 * np.linalg.norm(guide))
    along = (xy - q[k]) @ direction
    positions = .15 + .7 * (along - cfg["along_start_px"]) / (end - cfg["along_start_px"])
    line_cfg = dict(along_interval=[.15, .85], huber_delta_px=1.25,
        maximum_angle_change_deg=cfg["maximum_direction_change_deg"],
        minimum_inlier_count=cfg["minimum_inlier_count"],
        minimum_inlier_fraction=cfg["minimum_inlier_fraction"],
        minimum_inlier_span_fraction=cfg["minimum_inlier_span_fraction"],
        maximum_line_rms_px=cfg["maximum_line_rms_px"])
    fitted, diagnostics = F._fit_line(xy[keep], np.asarray(row["observation_confidence"])[keep],
                                     positions[keep], seed_normal, row["samples"], line_cfg)
    row.update(diagnostics, supported=fitted is not None,
               measurement_source="OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION")
    return row


def _intersection(first, second, minimum_sine):
    matrix = np.asarray([first["normal"], second["normal"]], float)
    sine = abs(float(np.linalg.det(matrix)))
    if sine < minimum_sine:
        return None
    point = np.linalg.solve(matrix, -np.asarray([first["offset"], second["offset"]]))
    return dict(sine=sine, first=first, second=second, point=point)


def _shared_edge_consistency(proposals, maximum_direction_change_deg):
    """Independently observed endpoint rays must agree on their shared edge."""
    audit = []
    invalid = set()
    for k, value in sorted(proposals.items()):
        for ray in (value["first"], value["second"]):
            mate = int(ray["native_neighbor"])
            if mate <= k or mate not in proposals:
                continue
            opposite = [r for r in (proposals[mate]["first"], proposals[mate]["second"])
                        if int(r["native_neighbor"]) == k]
            if not opposite:
                continue
            other = opposite[0]
            n1, n2 = np.asarray(ray["normal"]), np.asarray(other["normal"])
            angle = float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1))))
            points = np.asarray([value["point"], proposals[mate]["point"]])
            gap = float(max(np.abs(points @ n1 + ray["offset"]).max(),
                            np.abs(points @ n2 + other["offset"]).max()))
            valid = angle <= maximum_direction_change_deg and gap <= 3.
            audit.append(dict(native_edge=[k, mate], angle_difference_deg=angle,
                              endpoint_line_disagreement_px=gap, supported=bool(valid)))
            if not valid:
                invalid.update((k, mate))
    return audit, invalid


def recover_visible_boundary(image_bgr, raw_points, K, initial_pose, states, *,
                             pallet_mask, base_correction, native_xyz=None, config=None):
    """Return a v2-compatible measurement dictionary with recovery audit fields.

``base_correction`` must be the matching frozen ``correct_visible_bi`` output.
Its successful B/I measurements are copied exactly. Recovery alpha is a fixed
observation readout factor, not a fitted detector confidence. At alpha=1 the
conditional covariance is the measured line-intersection covariance itself.
"""
    cfg = _configuration(config)
    image, q, mask = np.asarray(image_bgr), np.asarray(raw_points, float), np.asarray(pallet_mask, float)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or q.shape not in ((8, 2), (9, 2)) or len(states) != 8
            or any(s not in ("B", "I", "S", "O", "T", "U") for s in states)):
        raise ValueError("Actual BGR image, native q8/9 and eight valid states required")
    if (mask.shape != image.shape[:2] or not np.isfinite(mask).all()
            or np.any((mask < 0) | (mask > 1))):
        raise ValueError("Bound actual SAM target mask in [0,1] required")
    base = deepcopy(base_correction)
    if base.get("image_pixel_sha256") != B.image_sha256(image):
        raise ValueError("Base correction image binding mismatch")
    if not np.array_equal(np.asarray(base.get("raw_points"), float), q, equal_nan=True):
        raise ValueError("Base correction raw native observations mismatch")
    rows = {int(r["native_corner"]): r for r in base.get("records", [])}
    measurements = {int(r["native_corner"]): r for r in base.get("visible_measurements", [])}
    # The interval adapter acts after frozen B/I correction. Its sole state
    # transition is unqueried T->U; this cannot change any B/I measurement.
    excluded_T_to_U = [k for k in rows if rows[k].get("state") == "T"
                       and k < len(states) and states[k] == "U"
                       and not rows[k].get("pixel_queried") and not rows[k].get("corrected")]
    if (set(rows) != set(range(8))
            or any(rows[k].get("state") != states[k] and k not in excluded_T_to_U for k in rows)
            or set(measurements) != {k for k, s in enumerate(states) if s in ("B", "I")}
            or any(measurements[k].get("state") != states[k] for k in measurements)):
        raise ValueError("Base correction state/native measurement binding mismatch")
    for k in excluded_T_to_U:
        rows[k].update(state="U", original_correction_state="T",
                       state_transition="post_correction_truncation_interval_T_to_U")
    legacy = dict(base.get("configuration", {}))
    if set(legacy) != set(B.DEFAULT_CONFIG):
        raise ValueError("Complete original B/I correction configuration required")
    X = C.cuboid(*initial_pose["cf_extents"])
    if native_xyz is not None:
        registered = np.asarray(native_xyz, float)
        if registered.shape != (8, 3) or not np.isfinite(registered).all() or not np.allclose(registered, X, rtol=0, atol=1e-9):
            raise ValueError("Registered native cuboid/order must match initial cf_extents")
        X = registered.copy()
    P = C.project(X, initial_pose["R_cf"], initial_pose["centroid"], np.asarray(K, float))
    if not np.isfinite(P).all():
        raise ValueError("Projectable initial registered model required")
    hull = cv2.convexHull(P.astype(np.float32), returnPoints=False).ravel().tolist()
    neighbors = {k: (hull[(i-1) % len(hull)], hull[(i+1) % len(hull)]) for i, k in enumerate(hull)}
    exterior, contours = _external_mask_and_contours(mask)
    lab = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), .8)
    gradients = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)/8., cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)/8.
    h, w = image.shape[:2]
    recovery, proposals = [], {}
    for k, state in enumerate(states):
        row = dict(native_corner=k, state=state, pixel_queried=False, recovered=False)
        recovery.append(row)
        if state != "B":
            row["reason"] = "excluded_before_recovery_image_query"
            continue
        if rows[k].get("corrected"):
            row["reason"] = "existing_corrected_B_preserved"
            continue
        if k not in neighbors:
            row["reason"] = "B_not_on_initial_model_hull"
            continue
        if not np.isfinite(q[k]).all() or not (0 <= q[k, 0] < w and 0 <= q[k, 1] < h):
            raise ValueError("Visible B requires an original in-frame image coordinate")
        distance = _exterior_distance(q[k], contours)
        if distance is None:
            row["reason"] = "no_actual_target_exterior_contour"
            continue
        radius = max(legacy["normal_search_radius_px"], distance + legacy["boundary_context_px"])
        row.update(pixel_queried=True, raw_xy=q[k].tolist(), exterior_distance_px=distance,
                   image_derived_reach_px=radius, reach_rule="max(original_radius, exterior_distance+original_context)")
        rays = [_observed_line(image, exterior, lab, gradients, q, k, mate, P[mate]-P[k],
                               P.mean(0), legacy, radius, cfg["require_rgb_support"]) for mate in neighbors[k]]
        row["expanded_rays"] = rays
        supported = [r for r in rays if r.get("supported")]
        candidates, anchors = [], []
        if cfg["recenter_on_observed_line"]:
            for seed in supported:
                normal = np.asarray(seed["normal"], float)
                anchor = q[k] - (normal @ q[k] + seed["offset"]) * normal
                local_q = q.copy(); local_q[k] = anchor
                anchor_distance = _exterior_distance(anchor, contours)
                local_radius = max(legacy["normal_search_radius_px"], anchor_distance + legacy["boundary_context_px"])
                for mate in neighbors[k]:
                    if mate == seed["native_neighbor"]:
                        continue
                    other = _observed_line(image, exterior, lab, gradients, local_q, k, mate,
                        P[mate]-P[k], P.mean(0), legacy, local_radius, cfg["require_rgb_support"])
                    anchors.append(dict(seed_native_neighbor=seed["native_neighbor"], observed_anchor_xy=anchor.tolist(),
                                        anchor_source="orthogonal_projection_on_observed_RGB_mask_line", remeasured_ray=other))
                    if other.get("supported"):
                        candidate = _intersection(seed, other, legacy["minimum_intersection_sine"])
                        if candidate is not None:
                            candidates.append(candidate)
        elif len(supported) == 2:
            candidate = _intersection(*supported, legacy["minimum_intersection_sine"])
            if candidate is not None:
                candidates.append(candidate)
        row["observed_line_recentering"] = anchors
        if not candidates:
            row["reason"] = "no_two_supported_nonparallel_actual_boundary_lines"
            continue
        disagreement = max(float(np.linalg.norm(a["point"]-b["point"])) for a in candidates for b in candidates)
        chosen = max(candidates, key=lambda c: c["sine"])
        shift = float(np.linalg.norm(chosen["point"]-q[k]))
        row.update(intersection_xy=chosen["point"].tolist(), intersection_shift_px=shift,
                   intersection_sine=chosen["sine"], candidate_count=len(candidates),
                   candidate_disagreement_px=disagreement,
                   original_12px_trust_would_pass=bool(shift <= legacy["maximum_corner_shift_px"]))
        point = chosen["point"]
        if disagreement > 3.:
            row["reason"] = "ambiguous_observed_intersections"
        elif not (0 <= point[0] < w and 0 <= point[1] < h):
            row["reason"] = "observed_intersection_outside_image"
        elif cfg["maximum_corner_shift_px"] is not None and shift > cfg["maximum_corner_shift_px"]:
            row["reason"] = "explicit_recovery_corner_shift_cap"
        else:
            proposals[k] = chosen
            row["reason"] = "supported_observed_intersection_pending_covariance"
    shared, invalid = _shared_edge_consistency(proposals, legacy["maximum_direction_change_deg"])
    measured = np.asarray(base["measured_points"], float).copy()
    for k, chosen in sorted(proposals.items()):
        row = recovery[k]
        if k in invalid:
            row["reason"] = "inconsistent_observed_shared_edge"
            continue
        first, second, point = chosen["first"], chosen["second"], chosen["point"]
        correction = dict(edge_records=[first, second], corner_records=[dict(corner=k, refined=True,
            candidate=point.tolist(), incident_shared_edges=[first["edge"], second["edge"]])])
        covs, _ = B.intersection_uncertainty(correction, repetitions=int(legacy["line_resampling"]),
                                            block_length=3, normal_floor_px=1., seed=int(legacy["seed"]))
        cov = covs.get(k)
        if cov is None or cov["successful_pair_resamples"] < 16:
            row["reason"] = "insufficient_observed_line_resamples"
            continue
        conditional = np.asarray(cov["covariance_px2"], float)
        alpha = cfg["maximum_alpha"]
        measured[k] = q[k] + alpha*(point-q[k])
        # At alpha1 do not retain a raw detector term or add the same normal
        # floor twice. This remains conditional uncertainty, not total error.
        covariance = alpha**2 * conditional + (1-alpha)**2 * legacy["detector_sigma_px"]**2 * np.eye(2)
        row.update(recovered=True, reason="actual_exterior_RGB_boundary_intersection", blend_alpha=alpha,
                   applied_shift_px=float(np.linalg.norm(measured[k]-q[k])), measured_xy=measured[k].tolist(),
                   covariance_px2=covariance.tolist(), conditional_covariance=deepcopy(cov),
                   selected_rays=[first, second])
        rows[k].update(corrected=True, method="B_ACTUAL_BOUNDARY_RECOVERY", reason=row["reason"],
            rays=[first, second], raw_xy=q[k].tolist(), measured_xy=measured[k].tolist(),
            intersection_xy=point.tolist(), intersection_shift_px=row["intersection_shift_px"],
            intersection_sine=row["intersection_sine"], blend_alpha=alpha,
            applied_shift_px=row["applied_shift_px"], covariance_px2=covariance.tolist(),
            conditional_covariance=deepcopy(cov), boundary_recovery=True)
        measurements[k].update(xy=measured[k].tolist(), covariance_px2=covariance.tolist(),
            source="B_ACTUAL_BOUNDARY_RECOVERY", image_measurement=True,
            measurement_is_pose_projection=False)
    base.update(schema="visible_BI_measurements_boundary_recovery_v4_3", measured_points=measured.tolist(),
        boundary_recovery_configuration=cfg, boundary_recovery_records=recovery,
        observed_shared_edge_audit=shared,
        recovery_queried_native_ids=[r["native_corner"] for r in recovery if r["pixel_queried"]],
        recovered_native_ids=[r["native_corner"] for r in recovery if r["recovered"]])
    base["provenance"].update(recovery_only_original_fallback_B=True, recovery_GT_inputs=False,
        recovery_prior_projection_is_measurement=False, recovery_search_reach_source="actual_SAM_exterior_distance",
        recovery_observed_line_recentering=cfg["recenter_on_observed_line"],
        recovery_covariance_is_total_error_or_semantic_probability=False,
        recovery_exterior_holes_ignored=True, recovery_RGB_support_required=cfg["require_rgb_support"],
        recovery_initial_pose_supplies_directions_only=True,
        recovery_post_correction_excluded_T_to_U_native_ids=excluded_T_to_U,
        recovery_shared_edges_independently_observed=True)
    return C.finite(base)
