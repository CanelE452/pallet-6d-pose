"""Versioned fallback-B search reach around immutable recovery v4.3b.

Nearest-mask-boundary distance is not a corner-localization error bound: a
query can already lie on one incident edge and miss the other edge. The fixed
search floor is the existing along-end sampling length plus boundary context.
Observed support is checked before conditioning ranks candidate intersections.
All RGB, line, disagreement and observed finite-support guard values stay unchanged.
This is an opt-in image correction, with no pose fit or uncertainty calibration.
"""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from . import visible_bi_boundary_recovery_v4_3b as PARENT

DEFAULT_CONFIG = deepcopy(PARENT.DEFAULT_CONFIG)
ORIGINAL = PARENT.ORIGINAL
C, B = ORIGINAL.C, ORIGINAL.B
_configuration = ORIGINAL._configuration
_external_mask_and_contours = ORIGINAL._external_mask_and_contours
_exterior_distance = ORIGINAL._exterior_distance
_observed_line = ORIGINAL._observed_line
_intersection = ORIGINAL._intersection
_shared_edge_consistency = ORIGINAL._shared_edge_consistency


def _choose_supported_pair(candidates, maximum_extrapolation_px):
    """Conditioning ranks admissible observed pairs, not unsupported ones."""
    audits = []
    valid = []
    for index, candidate in enumerate(candidates):
        guard = PARENT._support_interval_guard(candidate["point"],
            [candidate["first"], candidate["second"]], maximum_extrapolation_px)
        audits.append(dict(candidate_index=index, intersection_xy=candidate["point"].tolist(),
            intersection_sine=candidate["sine"], support_guard=guard, admissible=guard["passed"]))
        if guard["passed"]:
            valid.append((index, candidate))
    chosen = max(valid, key=lambda v:v[1]["sine"]) if valid else None
    return (chosen[1] if chosen else None), dict(
        maximum_extrapolation_px=maximum_extrapolation_px,
        candidate_count=len(candidates), support_admissible_count=len(valid),
        selected_candidate_index=chosen[0] if chosen else None,
        candidate_records=audits, support_filter_before_conditioning_selection=True)


def _working_configuration(original):
    """Change only the search radius; derive its floor from existing fields."""
    config = deepcopy(dict(original))
    values = np.asarray([config["normal_search_radius_px"], config["along_end_px"],
                         config["boundary_context_px"]], float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Original search radius, along end and context must be finite nonnegative")
    floor = float(values[1] + values[2])
    config["normal_search_radius_px"] = max(float(values[0]), floor)
    return config, floor


def _recover_with_support_before_selection(image_bgr, raw_points, K, initial_pose, states, *,
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
        chosen, support_selection = _choose_supported_pair(
            candidates, float(legacy["along_start_px"] + legacy["boundary_context_px"]))
        row["support_before_selection"] = support_selection
        if chosen is None:
            row["reason"] = "all_observed_intersections_extrapolate_beyond_support"
            row["rejected_intersection_xy"] = max(candidates, key=lambda c:c["sine"])["point"].tolist()
            continue
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


def recover_visible_boundary(image_bgr, raw_points, K, initial_pose, states, *,
                             pallet_mask, base_correction, native_xyz=None, config=None):
    """Keep the parent signature and every already successful B/I measurement.

    The supplied correction can include previous successful recovery, such as
    v4.6's full-mask B0/B3 results. Only its still-fallback B corners are queried.
    Original measurement/configuration dictionaries are not mutated. Parent
    exterior distance + context may enlarge the working floor further.
    """
    if config is not None and not config.get("require_rgb_support", True):
        raise ValueError("Corner reach requires the unchanged actual RGB support guard")
    original = deepcopy(base_correction["configuration"])
    working, floor = _working_configuration(original)
    supplied = deepcopy(base_correction)
    supplied["configuration"] = working
    result = _recover_with_support_before_selection(image_bgr, raw_points, K, initial_pose, states,
        pallet_mask=pallet_mask, base_correction=supplied, native_xyz=native_xyz, config=config)
    previous_rows = {int(r["native_corner"]):r for r in base_correction["records"]}
    previous_measurements = {int(r["native_corner"]):r for r in base_correction["visible_measurements"]}
    rows = {int(r["native_corner"]):r for r in result["records"]}
    measurements = {int(r["native_corner"]):r for r in result["visible_measurements"]}
    preserved = []
    for k, state in enumerate(states):
        if state == "I" or (state == "B" and previous_rows[k].get("corrected")):
            if (rows[k] != previous_rows[k] or measurements[k] != previous_measurements[k]
                    or result["measured_points"][k] != base_correction["measured_points"][k]):
                raise RuntimeError("Parent changed a supplied successful B or existing I")
            preserved.append(k)
    support_margin = float(original["along_start_px"] + original["boundary_context_px"])
    support_audits = []
    for row in result["boundary_recovery_records"]:
        if not row["recovered"]:
            continue
        audit = PARENT._support_interval_guard(row["intersection_xy"], row["selected_rays"], support_margin)
        if not audit["passed"]:
            raise RuntimeError("Support-admissible selection produced an unsupported output")
        audit.update(native_corner=row["native_corner"], base_measurement_preserved=False)
        row["boundary_recovery_support_guard"] = deepcopy(audit)
        support_audits.append(audit)
    result["boundary_recovery_support_guard"] = dict(
        rule="each_ray_tangent_interval_extrapolation <= original_along_start_px + original_boundary_context_px",
        along_start_px=original["along_start_px"], boundary_context_px=original["boundary_context_px"],
        maximum_extrapolation_px=support_margin, corner_records=support_audits,
        rejected_native_ids=[r["native_corner"] for r in result["boundary_recovery_records"]
            if r["reason"] == "all_observed_intersections_extrapolate_beyond_support"],
        support_filter_before_maximum_sine=True, new_image_queries=0, GT_inputs=False,
        geometry_is_observed_inlier_segment=True)
    result["configuration"] = original
    result["schema"] = "visible_BI_measurements_corner_reach_v4_7"
    result["corner_reach_audit"] = dict(
        schema="FALLBACK_B_INCIDENT_CORNER_REACH_V4_7",
        rule="max(original_normal_search_radius, original_along_end + original_boundary_context, exterior_distance + original_boundary_context)",
        original_normal_search_radius_px=float(original["normal_search_radius_px"]),
        original_along_end_px=float(original["along_end_px"]),
        original_boundary_context_px=float(original["boundary_context_px"]),
        derived_incident_corner_reach_floor_px=floor,
        working_normal_search_radius_px=working["normal_search_radius_px"],
        original_configuration=deepcopy(original), working_configuration=deepcopy(working),
        top_level_original_configuration_restored=True,
        preserved_complete_B_and_I_ids=preserved,
        eligible_fallback_B_ids=[k for k,s in enumerate(states) if s == "B" and not previous_rows[k].get("corrected")],
        new_recovered_native_ids=list(result["recovered_native_ids"]),
        per_corner=[dict(native_corner=r["native_corner"],state=r["state"],
            pixel_queried=r["pixel_queried"],recovered=r["recovered"],reason=r["reason"],
            exterior_distance_px=r.get("exterior_distance_px"),
            actual_search_reach_px=r.get("image_derived_reach_px"),
            support_before_selection=deepcopy(r.get("support_before_selection"))) for r in result["boundary_recovery_records"]],
        RGB_support_guard_preserved=True, angular_RMS_count_span_sine_guards_preserved=True,
        observed_support_extrapolation_px=result["boundary_recovery_support_guard"]["maximum_extrapolation_px"],
        source_of_floor="existing_incident_edge_sampling_extent_and_boundary_context",
        floor_is_CAL_fitted_or_GT_tuned=False, nearest_boundary_distance_is_corner_error_bound=False,
        no_new_pose_fits=True, no_new_uncertainty_calibration=True, S_image_queries=0, GT_inputs=False)
    result["provenance"].update(corner_reach_immutable_parent="visible_bi_boundary_recovery_v4_3b",
        corner_reach_original_top_configuration_preserved=True,
        corner_reach_original_complete_B_and_I_preserved=True,
        corner_reach_changes_only_fallback_B_search_and_support_admissible_selection=True,
        corner_reach_support_filter_before_maximum_sine=True,
        corner_reach_original_all_candidate_disagreement_guard_preserved=True,
        corner_reach_recovery_body_copied_from_immutable_v4_3=True,
        corner_reach_floor_derived_from_original_along_end_and_context=True,
        corner_reach_GT_inputs=False,corner_reach_S_image_queries=0)
    return result

