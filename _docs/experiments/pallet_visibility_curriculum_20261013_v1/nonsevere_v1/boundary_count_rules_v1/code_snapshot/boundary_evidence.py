"""Pure boundary support/status contract; neither fitting nor semantic invention.

Tiny/RGB support and a class-agnostic SAM outline cannot identify an occluder
or prove which requested 3D corner generated a pixel. Strong boundary counts
refer to two locally observed lines, not pose rank or certified 3D visibility.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _flags(value, name):
    result = np.asarray(value, bool)
    if result.shape != (8,):
        raise ValueError(name + " must retain eight native corner slots")
    return result


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _indexed(records, key):
    result = {}
    for record in records:
        index = int(record[key])
        if index in result:
            raise ValueError("Duplicate " + key + " record")
        result[index] = record
    return result


def _line_evidence(edge, raw):
    """Validate actual retained samples; missing samples are not occlusion proof."""
    endpoints = list(edge.get("endpoints", []))
    points = np.asarray(edge.get("observed_points", []), float)
    confidence = np.asarray(edge.get("observation_confidence", []), float)
    inlier = np.asarray(edge.get("inlier_mask", []), bool)
    normal = np.asarray(edge.get("normal", []), float)
    n = len(points)
    samples = int(edge.get("samples", 0))
    payload = (len(endpoints) == 2 and len(set(endpoints)) == 2
               and all(isinstance(k, (int, np.integer)) and 0 <= k < 8 for k in endpoints)
               and points.shape == (n, 2) and n >= 5 and np.isfinite(points).all()
               and confidence.shape == (n,) and np.isfinite(confidence).all()
               and np.all((confidence >= 0) & (confidence <= 1))
               and inlier.shape == (n,) and normal.shape == (2,)
               and np.isfinite(normal).all() and np.isclose(np.linalg.norm(normal), 1., atol=1e-5)
               and samples >= n and _number(edge.get("offset")) is not None)
    reasons = []
    verified = False
    gap = None
    span = _number(edge.get("inlier_span_fraction"))
    rms = _number(edge.get("inlier_rms_px"))
    fraction = _number(edge.get("inlier_fraction"))
    angle = _number(edge.get("angle_change_deg"))
    if not edge.get("supported", False):
        reasons.append(str(edge.get("reason", "not_supported")))
    elif not payload:
        reasons.append("missing_or_invalid_actual_line_samples")
    else:
        count = int(inlier.sum())
        recomputed_rms = float(np.sqrt(np.mean((points[inlier] @ normal + float(edge["offset"])) ** 2))) if count else None
        a, b = raw[endpoints]
        delta = b - a
        length2 = float(delta @ delta)
        location = (points @ delta - a @ delta) / length2 if length2 > 1e-9 else np.zeros(n)
        recomputed_span = float(np.ptp(location[inlier])) / .7 if count else 0.
        verified = (count >= 5 and int(edge.get("inlier_count", -1)) == count
                    and fraction is not None and np.isclose(fraction, count / samples)
                    and fraction >= .4 and span is not None and span >= .5
                    and np.isclose(span, recomputed_span, atol=1e-5)
                    and rms is not None and rms <= 1.5 and recomputed_rms is not None
                    and np.isclose(rms, recomputed_rms, atol=1e-5)
                    and angle is not None and angle <= 15.)
        if not verified:
            reasons.append("line_quality_or_sample_binding_failed")
        # Gaps are along the sampled interior interval, not a verified physical
        # disappearance/reappearance, depth ordering, or occluder region.
        if length2 > 1e-9:
            bins = np.clip(np.rint((location - .15) / .7 * (samples - 1)).astype(int), 0, samples - 1)
            present = np.zeros(samples, bool)
            present[bins] = True
            longest = current = 0
            for value in present:
                current = 0 if value else current + 1
                longest = max(longest, current)
            gap = dict(supported_sample_bins=int(present.sum()), sampled_bins=samples,
                       longest_missing_run=longest, proves_occluder=False)
    return dict(edge=int(edge["edge"]), endpoints=endpoints, verified_image_line=bool(verified),
                reason=reasons or ["supported_actual_mask_transition_line"],
                transition_sample_count=int(edge.get("transition_sample_count", 0)), sampled_count=samples,
                inlier_count=int(edge.get("inlier_count", 0)), inlier_span_fraction=span,
                inlier_rms_px=rms, directionality="target_mask_inside_non_target_outside" if verified else "NOT_VERIFIED",
                directionality_proves_non_target_is_other_object=False,
                continuation="SUPPORTED_OVER_SAMPLED_SPAN" if verified else "NOT_VERIFIED",
                sampled_transition_gaps=gap)


def _occluder_evidence(item, corner, xy):
    """Accept explicit external evidence bindings, never target-mask negatives.

This validates the supplied contract, not the sensor/instance algorithm. A
caller must retain independently verified source records. Initial model depth
is explicitly insufficient to certify the requested physical corner depth.
"""
    if not item:
        return dict(independent_region_supported=False, foreground_order_verified=False,
                    confirmed=False, source_reference=None)
    query = np.asarray(item.get("query_xy", []), float)
    region = (item.get("schema") == "independent_occluding_region_v1"
              and item.get("native_corner") == corner and query.shape == (2,)
              and np.isfinite(query).all() and np.allclose(query, xy, atol=1e-6, rtol=0)
              and item.get("independent_of_target_palette") is True
              and item.get("different_instance_from_target") is True
              and item.get("region_contains_query") is True
              and item.get("region_observed_rgb") is True
              and isinstance(item.get("instance_id"), str) and bool(item["instance_id"])
              and isinstance(item.get("source_reference"), str) and bool(item["source_reference"])
              and item.get("source_type") in {"instance_segmentation", "depth_instance", "verified_foreground_tracker"})
    depth = item.get("verified_depth_order", {})
    foreground, target = _number(depth.get("foreground_depth_m")), _number(depth.get("requested_point_depth_m"))
    uncertainty = _number(depth.get("depth_margin_uncertainty_m"))
    ordered = (region and item.get("source_type") == "depth_instance"
               and depth.get("requested_point_binding") == "independently_verified_physical_corner"
               and depth.get("target_depth_origin") == "independently_verified_requested_point_depth"
               and depth.get("same_camera_ray") is True
               and isinstance(depth.get("verification_reference"), str) and bool(depth["verification_reference"])
               and foreground is not None and target is not None and uncertainty is not None
               and foreground > 0 and uncertainty >= 0 and target - foreground > uncertainty)
    return dict(independent_region_supported=bool(region), foreground_order_verified=bool(ordered),
                confirmed=bool(ordered), source_reference=item.get("source_reference") if region else None,
                source_facts_verified_by_this_helper=False)


def summarize_boundary_evidence(observation_result, *, image_hw=None,
                               strong_visibility_threshold=.65, occluding_evidence=None):
    """Return eight statuses and strong/native IDs for a separate count policy.

``observation_result`` is the unchanged BoundaryObservation output. An actual
canvas is required, either explicit or its BoundaryVisibility configuration.
No geometry/model projections become coordinates, nor do precision/usability
scores affect the semantic visibility classification. Optional external region
evidence is keyed by native corner and must refer to the original RGB query.
"""
    if not 0 <= strong_visibility_threshold <= 1:
        raise ValueError("Visibility threshold must be in [0,1]")
    raw = np.asarray(observation_result["raw_points"], float)
    if raw.shape not in ((8, 2), (9, 2)):
        raise ValueError("Expected unchanged raw native q8/9")
    boundary = observation_result["boundary_visibility"]
    if image_hw is None:
        image_hw = boundary.get("configuration", {}).get("actual_image_hw")
    canvas = np.asarray(image_hw)
    if canvas.shape != (2,) or not np.isfinite(canvas).all() or np.any(canvas <= 0):
        raise ValueError("Actual image_hw required; principal point is not canvas size")
    height, width = map(int, canvas)
    H = _flags(observation_result.get("self_hidden", observation_result.get("fixed_H")), "self_hidden")
    outline = _flags(boundary["expected_outline"], "expected_outline")
    truncated = _flags(boundary["conditional_truncation"], "conditional_truncation")
    claimed = _flags(boundary["geometric_boundary_candidates"], "geometric_boundary_candidates")
    valid = np.isfinite(raw[:8]).all(1) & ~np.all(raw[:8] == -1, axis=1)
    inside = valid & (raw[:8, 0] >= 0) & (raw[:8, 0] < width) & (raw[:8, 1] >= 0) & (raw[:8, 1] < height)
    eligible = outline & ~H & ~truncated & inside
    if not np.array_equal(claimed, eligible):
        raise ValueError("Geometric boundary population must use original H/outline/projection/canvas, not model scores")
    appearance = observation_result["visibility_appearance"]
    probability = np.asarray(appearance["visibility_score"], float)
    if (probability.shape != (8,) or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1)) or len(appearance["states"]) != 8):
        raise ValueError("Fresh independent RGB visibility appearance required")
    if appearance.get("diagnostics", {}).get("joint_observation"):
        raise ValueError("P_usable is a coordinate target, not independent RGB visibility")
    correction = observation_result["correction"]
    if correction.get("reprojected_points_used", False) or not correction.get("independent_image_evidence", False):
        raise ValueError("Line coordinates must be actual image observations, never completed projections")
    edges = _indexed(correction.get("edge_records", []), "edge")
    corners = _indexed(correction.get("corner_records", []), "corner")
    lines = {k: _line_evidence(edge, raw[:8]) for k, edge in edges.items()}
    covariance = observation_result.get("intersection_covariance", {})
    external = occluding_evidence or {}
    diagnostics = {int(d["corner"]): d for d in appearance.get("diagnostics", {}).get("corners", [])}
    records, strong_ids, material_ids, line_ids = [], [], [], []
    for k in range(8):
        corner = corners.get(k, {})
        incident = list(corner.get("incident_shared_edges", []))
        candidate = np.asarray(corner.get("candidate", []), float)
        two_lines = (len(incident) == 2 and len(set(incident)) == 2
                     and all(e in lines and lines[e]["verified_image_line"] and k in lines[e]["endpoints"] for e in incident))
        shift = float(np.linalg.norm(candidate - raw[k])) if candidate.shape == (2,) and np.isfinite(candidate).all() else None
        intersection_matches = False
        if two_lines:
            matrix = np.stack([edges[e]["normal"] for e in incident])
            determinant = abs(float(np.linalg.det(matrix)))
            if determinant >= .25 and candidate.shape == (2,) and np.isfinite(candidate).all():
                actual_intersection = np.linalg.solve(matrix, -np.array([edges[e]["offset"] for e in incident]))
                intersection_matches = bool(np.allclose(candidate, actual_intersection, atol=1e-5, rtol=0))
        point_supported = bool(two_lines and intersection_matches and corner.get("refined") is True and candidate.shape == (2,)
            and shift is not None and shift <= 12. + 1e-8
            and 0 <= candidate[0] < width and 0 <= candidate[1] < height
            and _number(corner.get("intersection_sine")) is not None and corner["intersection_sine"] >= .25)
        covariance_record = covariance.get(str(k), covariance.get(k))
        covariance_valid = None
        if covariance_record is not None:
            matrix = np.asarray(covariance_record.get("covariance_px2", []), float)
            covariance_valid = bool(matrix.shape == (2, 2) and np.isfinite(matrix).all()
                and np.allclose(matrix, matrix.T) and np.linalg.eigvalsh(matrix).min() >= -1e-8
                and int(covariance_record.get("successful_pair_resamples", 0)) >= 16
                and set(covariance_record.get("incident_shared_edges", [])) == set(incident))
        region = _occluder_evidence(external.get(k, external.get(str(k))), k, raw[k]) if inside[k] and ~H[k] and ~truncated[k] else _occluder_evidence(None, k, raw[k])
        material = bool(eligible[k] and probability[k] >= strong_visibility_threshold)
        # A separately observed different instance can match the pallet color.
        # It vetoes strong target ownership even while depth ordering remains a
        # suspicion. Retain high RGB support as a separate conflicting fact.
        strong = bool(material and point_supported and covariance_valid is not False
                      and not region["independent_region_supported"])
        if material:
            material_ids.append(k)
        if eligible[k] and point_supported:
            line_ids.append(k)
        if strong:
            strong_ids.append(k)
        RGB_state = appearance["states"][k]
        if H[k]:
            state = "SELF_OCCLUDED"
        elif truncated[k]:
            state = "TRUNCATED"
        elif not valid[k] or not inside[k]:
            state = "UNKNOWN"
        elif region["confirmed"]:
            state = "OTHER_OBJECT_OCCLUSION"
        elif region["independent_region_supported"]:
            state = "OTHER_OBJECT_SUSPECTED"
        elif probability[k] >= strong_visibility_threshold:
            state = "VISIBLE_BOUNDARY" if strong or RGB_state == "VISIBLE_BOUNDARY" else "VISIBLE_INTERIOR" if RGB_state == "VISIBLE_INTERIOR" else "VISIBLE_MATERIAL_SUPPORTED"
        else:
            state = "UNKNOWN_OR_OCCLUSION"
        local = diagnostics.get(k, {})
        directionality = {key: _number(local.get(key)) for key in
            ("inward_material_fraction", "outward_material_fraction", "patch_material_fraction",
             "nearby_inward_material_fraction", "soft_inward_material_support", "soft_nearby_inward_material_support",
             "segmentation_probability_inward_mean", "segmentation_probability_outward_mean")}
        inward, outward = directionality["inward_material_fraction"], directionality["outward_material_fraction"]
        directionality["inward_minus_outward"] = inward - outward if inward is not None and outward is not None else None
        records.append(dict(corner=k, state=state, expected_outline=bool(outline[k]),
            self_occluded_conditional=bool(H[k]), truncated_conditional=bool(truncated[k]),
            raw_prediction_missing=bool(not valid[k]), raw_prediction_out_of_frame=bool(valid[k] and not inside[k]),
            raw_query_xy=raw[k].tolist() if valid[k] else None, geometric_boundary_candidate=bool(eligible[k]),
            p_visible=float(probability[k]), RGB_state_at_raw_query=RGB_state,
            RGB_boundary_score=_number(appearance.get("boundary_score", [None] * 8)[k]),
            coordinate_accuracy_score=_number(appearance.get("local_accuracy_score", [None] * 8)[k]),
            coordinate_accuracy_changes_visibility=False, RGB_material_supported=material,
            two_actual_image_lines=point_supported, strong_measured_boundary=strong,
            boundary_evidence_level="RGB_SUPPORT_CONFLICTS_WITH_DIFFERENT_INSTANCE" if
                material and region["independent_region_supported"] else
                "TWO_IMAGE_LINES_PLUS_HIGH_RGB_SUPPORT" if strong else
                "HIGH_RGB_SUPPORT_GEOMETRIC_BOUNDARY_ONLY" if material else
                "HIGH_RGB_SUPPORT_MODEL_INTERIOR" if inside[k] and not H[k] and not truncated[k]
                and probability[k] >= strong_visibility_threshold else "NOT_STRONGLY_MEASURED",
            incident_image_edges=incident, independent_intersection_xy=candidate.tolist() if point_supported else None,
            intersection_shift_px=shift, conditional_covariance_valid=covariance_valid,
            covariance_is_calibrated_semantic_confidence=False, RGB_material_directionality=directionality,
            occluder_evidence=region, external_occlusion_confirmed=bool(region["confirmed"] and state == "OTHER_OBJECT_OCCLUSION"),
            visibility_is_requested_3D_corner_proof=False,
            external_occlusion_assessment="OTHER_OBJECT_OCCLUSION_VERIFIED_BY_EXTERNAL_SOURCE" if region["confirmed"] else
                "EXTERNAL_OCCLUSION_SUSPECTED" if state == "OTHER_OBJECT_SUSPECTED" else
                "UNKNOWN_OCCLUSION_OR_LOCALIZATION" if state == "UNKNOWN_OR_OCCLUSION" else "NOT_ESTABLISHED",
            external_occlusion_hypothesis="EXTERNAL_OCCLUSION_SUSPECTED_OR_LOCALIZATION_ERROR" if state == "UNKNOWN_OR_OCCLUSION" else None))
    return dict(schema="boundary_evidence_v1", strong_ids=strong_ids, strong_count=len(strong_ids),
        geometric_boundary_ids=np.flatnonzero(eligible).tolist(), geometric_boundary_count=int(eligible.sum()),
        geometry_visible_expected_outline_ids=np.flatnonzero(outline & ~H & ~truncated).tolist(),
        geometry_visible_expected_outline_count=int((outline & ~H & ~truncated).sum()),
        RGB_material_supported_boundary_ids=material_ids, RGB_material_supported_boundary_count=len(material_ids),
        two_image_line_boundary_ids=line_ids, two_image_line_boundary_count=len(line_ids),
        state_counts=dict(Counter(r["state"] for r in records)), records=records,
        line_records=list(lines.values()), strong_visibility_threshold=float(strong_visibility_threshold),
        count_is_pose_rank_proof=False, OTHER_OBJECT_is_palette_negative=False,
        provenance=dict(annotation_reads=0, model_forwards=0, pose_fits=0,
                        original_native_slots_and_center_unchanged=True, geometry_fixed_before_RGB_measurement=True,
                        projected_coordinates_used_as_measurements=False, coordinate_precision_is_semantic_visibility=False),
        limits=["A non-target pixel can be background, a hole, a wrong query, a virtual corner, or an occluder",
                "Class-agnostic SAM lines establish observed mask boundaries, not other-instance identity or foreground ordering",
                "Line support gaps are not a verified occlusion break; suppressed or gated samples may have other causes",
                "SELF_OCCLUDED and TRUNCATED remain conditional on the initial cuboid pose",
                "Strong count alone cannot establish pose rank, conditioning, or uniqueness",
                "Supplied external instance/depth facts require verification by their independent producer"])
