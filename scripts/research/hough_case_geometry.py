"""Fixed, GT-free finite-segment association and anchored corner refinement.

This is a single-case diagnostic operator, not a validated rejection gate.
GT appears only in the separate audit CLI, never in associate_and_refine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EDGES12 = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
           (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
EDGES8 = tuple(EDGES12[i] for i in (1, 3, 5, 7, 8, 9, 10, 11))
CONFIG = dict(angle_max_deg=12., normal_distance_bbox_diagonal_fraction=.03,
              minimum_projected_overlap_fraction=.20, point_sigma_bbox_diagonal_fraction=.025,
              line_sigma_bbox_diagonal_fraction=.01, sigma_floor_px=1., lam=1.,
              max_move_image_diagonal_fraction=.01, minimum_segment_length_px=2.,
              assignment="ascending cost, then role index, then original segment index; one-to-one greedy",
              cost="angle/12 + symmetric_midpoint_normal_distance/(.03*bbox_diagonal) + (1-overlap)")


def clip_segment(segment, width, height):
    """Liang-Barsky clip to the observed pixel-center rectangle."""
    segment = np.asarray(segment, dtype=float)
    if segment.shape != (2, 2) or not np.isfinite(segment).all():
        return None
    start, delta = segment[0], segment[1]-segment[0]
    lower, upper = 0., 1.
    for origin, change, bound in zip(start, delta, (width-1, height-1)):
        if abs(change) < 1e-12:
            if origin < 0 or origin > bound:
                return None
        else:
            enter, leave = sorted((-origin/change, (bound-origin)/change))
            lower, upper = max(lower, enter), min(upper, leave)
            if lower > upper:
                return None
    clipped = np.stack((start+lower*delta, start+upper*delta))
    return clipped if np.linalg.norm(clipped[1]-clipped[0]) >= CONFIG["minimum_segment_length_px"] else None


def line_of(segment):
    delta = segment[1]-segment[0]
    length = float(np.linalg.norm(delta))
    tangent = delta/length
    normal = np.array([-tangent[1], tangent[0]])
    midpoint = segment.mean(0)
    return dict(h=np.r_[normal, -normal @ midpoint], normal=normal,
                tangent=tangent, midpoint=midpoint, length=length)


def associate_and_refine(points9, bbox_xyxy, segments_Nx2x2, width, height, edges):
    """Return original-pixel points and evidence; no GT or visibility argument.

    Corners outside the image remain legitimate amodal points. Only finite
    segments inside the observed image supply image evidence. Missing NaN points
    and center8 are copied. No detection score or rejection decision is produced.
    """
    points = np.asarray(points9, dtype=float)
    bbox = np.asarray(bbox_xyxy, dtype=float)
    segments = np.asarray(segments_Nx2x2, dtype=float)
    if points.shape != (9, 2) or bbox.shape != (4,):
        raise ValueError("Expected points[9,2] and bbox[4] in original-image pixels")
    if segments.size == 0:
        segments = np.empty((0, 2, 2), dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("Expected Hough segments[N,2,2]")
    if width < 2 or height < 2:
        raise ValueError("Invalid image dimensions")
    edges = [tuple(map(int, edge)) for edge in edges]
    if any(len(e) != 2 or min(e) < 0 or max(e) > 7 or e[0] == e[1] for e in edges):
        raise ValueError("Only structural corner0..7 edges are allowed")
    if len({tuple(sorted(e)) for e in edges}) != len(edges):
        raise ValueError("Duplicate structural edges would double-count evidence")
    refined = points.copy()
    available = np.isfinite(points).all(1)
    move = np.where(available, 0., np.nan)
    cap = CONFIG["max_move_image_diagonal_fraction"]*math.hypot(width, height)
    associations = [dict(role=i, edge=list(e), matched=False, segment_index=None, candidate_index=None,
                         reason="no_eligible_unique_segment") for i, e in enumerate(edges)]
    diagnostics = dict(config=CONFIG.copy(), gt_input=False, coordinate_frame="original_image_pixels",
        predicted_point_available=available.tolist(), observed_rectangle_xyxy=[0, 0, width-1, height-1],
        raw_segments=len(segments), eligible_segment_pairs=0, matched_roles=0,
        displacement_cap_px=cap, correction_is_validated=False, detection_rejected=False,
        missing_edge_evidence_semantics="Unsupported; occlusion/truncation can remove a real structural edge. It is not evidence of an incorrect point.")
    if not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        diagnostics["status"] = "invalid_box_identity"
        return dict(points_xy=refined.tolist(), associations=associations,
                    displacement_px=move.tolist(), diagnostics=diagnostics)
    diagonal = float(np.linalg.norm(bbox[2:]-bbox[:2]))
    sigma_point = max(CONFIG["sigma_floor_px"], CONFIG["point_sigma_bbox_diagonal_fraction"]*diagonal)
    sigma_line = max(CONFIG["sigma_floor_px"], CONFIG["line_sigma_bbox_diagonal_fraction"]*diagonal)
    threshold_distance = CONFIG["normal_distance_bbox_diagonal_fraction"]*diagonal
    diagnostics.update(bbox_diagonal_px=diagonal, point_sigma_px=sigma_point,
                       line_sigma_px=sigma_line, normal_distance_limit_px=threshold_distance)
    observed = []
    for si, segment in enumerate(segments):
        clipped = clip_segment(segment, width, height)
        if clipped is not None:
            observed.append((si, clipped, line_of(clipped)))
    diagnostics["finite_in_frame_segments"] = len(observed)
    pairs = []
    for role, (a, b) in enumerate(edges):
        if not (available[a] and available[b]):
            associations[role]["reason"] = "missing_predicted_endpoint"
            continue
        clipped = clip_segment(points[[a, b]], width, height)
        if clipped is None:
            associations[role]["reason"] = "predicted_segment_has_no_observed_support"
            continue
        reference = line_of(clipped)
        associations[role]["predicted_segment_clipped_xy"] = clipped.tolist()
        for si, segment, candidate in observed:
            cosine = np.clip(abs(reference["tangent"] @ candidate["tangent"]), 0., 1.)
            angle = math.degrees(math.acos(cosine))
            distance = .5*(abs(candidate["h"][:2] @ reference["midpoint"]+candidate["h"][2])
                           +abs(reference["h"][:2] @ candidate["midpoint"]+reference["h"][2]))
            projected = np.sort((segment-clipped[0]) @ reference["tangent"])
            overlap_px = max(0., min(reference["length"], projected[1])-max(0., projected[0]))
            overlap = float(np.clip(overlap_px/reference["length"], 0., 1.))
            if (angle > CONFIG["angle_max_deg"] or distance > threshold_distance
                    or overlap < CONFIG["minimum_projected_overlap_fraction"]):
                continue
            cost = angle/CONFIG["angle_max_deg"]+distance/threshold_distance+1-overlap
            pairs.append((float(cost), role, si, dict(angle_deg=angle, normal_distance_px=float(distance),
                overlap_fraction=overlap, overlap_px=float(overlap_px), h_original_px=candidate["h"].tolist(),
                observed_segment_clipped_xy=segment.tolist())))
    diagnostics["eligible_segment_pairs"] = len(pairs)
    used_roles, used_segments = set(), set()
    for cost, role, si, evidence in sorted(pairs, key=lambda pair: pair[:3]):
        if role in used_roles or si in used_segments:
            continue
        used_roles.add(role)
        used_segments.add(si)
        associations[role].update(matched=True, segment_index=si, candidate_index=si, reason="fixed_criteria_match",
                                  cost=cost, **evidence)
    diagnostics["matched_roles"] = len(used_roles)
    diagnostics["status"] = "refined" if used_roles else "no_association_identity"
    point_precision, line_precision = 1/sigma_point**2, CONFIG["lam"]/sigma_line**2
    clipped_corners = []
    for corner in range(8):
        if not available[corner]:
            continue
        incident = [a for a in associations if a["matched"] and corner in a["edge"]]
        if not incident:
            continue
        matrix, rhs = np.eye(2)*point_precision, np.zeros(2)
        for evidence in incident:
            h = np.asarray(evidence["h_original_px"])
            normal = h[:2]
            matrix += line_precision*np.outer(normal, normal)
            rhs -= line_precision*normal*(normal @ points[corner]+h[2])
        delta = np.linalg.solve(matrix, rhs)
        norm = float(np.linalg.norm(delta))
        if norm > cap:
            delta *= cap/norm
            clipped_corners.append(corner)
        refined[corner] += delta
        move[corner] = float(np.linalg.norm(delta))
    diagnostics["clipped_corners"] = clipped_corners
    diagnostics["center_preserved_exact"] = bool(np.array_equal(refined[8], points[8], equal_nan=True))
    diagnostics["no_new_points_filled"] = bool(np.array_equal(np.isfinite(refined), np.isfinite(points)))
    return dict(points_xy=refined.tolist(), associations=associations,
                displacement_px=move.tolist(), diagnostics=diagnostics)


def self_test():
    points = np.array([[100., 100.], [200., 100.], [200., 200.], [100., 200.],
                       [120., 80.], [220., 80.], [220., 180.], [120., 180.], [160., 140.]])
    box = [80., 60., 240., 220.]
    empty = associate_and_refine(points, box, [], 640, 480, EDGES8)
    assert np.array_equal(empty["points_xy"], points)
    observed = np.array([[[102., 100.], [102., 200.]], [[100., 103.], [200., 103.]]])
    out = associate_and_refine(points, box, observed, 640, 480, [(3, 0), (0, 1)])
    assert out["diagnostics"]["matched_roles"] == 2
    anchor, line = out["diagnostics"]["point_sigma_px"], out["diagnostics"]["line_sigma_px"]
    weight = anchor**2/(anchor**2+line**2)
    assert np.allclose(np.array(out["points_xy"])[0], points[0]+weight*np.array([2., 3.]), atol=1e-10)
    assert np.array_equal(np.array(out["points_xy"])[8], points[8])
    # Null and out-of-view observations do not extrapolate into apparent evidence.
    invalid = associate_and_refine(points, box, [[[float("nan"), 0.], [1., 1.]],
        [[100., 500.], [200., 500.]]], 640, 480, EDGES8)
    assert np.array_equal(invalid["points_xy"], points)
    # Two accepted displaced orthogonal observations require the registered8px cap.
    far = np.array([[[114., 100.], [114., 200.]], [[100., 114.], [200., 114.]]])
    cap = associate_and_refine(points, [0., 0., 500., 400.], far, 640, 480, [(3, 0), (0, 1)])
    assert abs(cap["displacement_px"][0]-8.) < 1e-10
    missing = points.copy(); missing[0] = np.nan
    null = associate_and_refine(missing, box, observed, 640, 480, [(3, 0), (0, 1)])
    assert np.array_equal(null["points_xy"], missing, equal_nan=True)
    # Orientation of an undirected Hough segment cannot change the solution.
    reverse = associate_and_refine(points, box, observed[:, ::-1], 640, 480, [(3, 0), (0, 1)])
    assert np.allclose(reverse["points_xy"], out["points_xy"], atol=1e-10)
    # The same observed segment cannot independently support two roles.
    duplicate = points.copy(); duplicate[4:6] = points[:2]
    unique = associate_and_refine(duplicate, box, observed[1:], 640, 480, [(0, 1), (4, 5)])
    assert unique["diagnostics"]["matched_roles"] == 1
    return dict(PASS=True, checks=["null_identity", "center_exact", "analytic_two_line_anchor_solution",
        "finite_and_observed_segments_only", "8px_displacement_cap", "missing_preserved",
        "undirected_segment_orientation_invariance", "one_to_one_association"])


def geometry_audit(output_dir):
    """GT-oracle diagnosis only: never returns a deployable relabel rule."""
    closure = ROOT/"data/pallet/results/paper_pose_metric_closure_v1"
    fid = "eval_pallet07__1778652166837872128"
    prediction_path = closure/"predictions/R0.json"
    geometry_path = closure/"GEOMETRY_RESOLVED_POSE_GT.json"
    annotation_path = ROOT/"data/evaluation/pallet_eval_v1/dev_existing/annotations/eval_pallet07/1778652166837872128.json"
    raw_path = ROOT/"challenge/data/01_real/manual_gt/capturepallet07_manual_gt/1778652166837872128.png"
    reference_image = ROOT/"data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_pallet07/rgb/1778652166837872128.png"
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    read = lambda path: json.loads(path.read_text())
    protocol_path = output_dir/"PROTOCOL.json"
    provenance_path = output_dir/"provenance/provenance.json"
    protocol, provenance = read(protocol_path), read(provenance_path)
    assert digest(protocol_path) == "f17180efad3a5706f12b6366cfe52d885495a4820f630115bd03880b26485b3e"
    assert provenance["complete"] and provenance["PASS"]
    for key, section, field in (
        ("angle_max_deg", "association", "angle_max_deg"),
        ("normal_distance_bbox_diagonal_fraction", "association", "symmetric_midpoint_normal_distance_max_bbox_diagonal"),
        ("minimum_projected_overlap_fraction", "association", "projected_overlap_min_fraction"),
        ("point_sigma_bbox_diagonal_fraction", "fusion", "point_anchor_std_bbox_diagonal"),
        ("line_sigma_bbox_diagonal_fraction", "fusion", "line_std_bbox_diagonal"),
        ("sigma_floor_px", "fusion", "std_floor_px"), ("lam", "fusion", "lambda"),
        ("max_move_image_diagonal_fraction", "fusion", "max_displacement_image_diagonal")):
        assert CONFIG[key] == protocol[section][field]
    pred = read(prediction_path)["frames"][fid]
    annotation = read(annotation_path)
    labels = annotation["objects"][0]["keypoint_annotations"]
    points, gt = np.array(pred["keypoints_xy"]), np.array([row["xy"] for row in labels])
    visibility = np.array([row["visibility"] for row in labels])
    support = visibility > 0
    assert np.array_equal(points, provenance["M4_record"]["keypoints_xy"])
    assert np.array_equal(gt, provenance["M4_record"]["gt_xy"])
    assert np.array_equal(support, provenance["M4_record"]["gt_supervised"])
    permutation = np.array([4, 0, 3, 7, 5, 1, 2, 6, 8])
    inverse = np.argsort(permutation)
    geometry = read(geometry_path)["frames"][fid]
    w, h, d = [geometry["physical_dimensions_m"][key] for key in ("across", "height", "along")]
    x = .5*np.array([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]])
    q = x@np.array(geometry["R_gt_representative"]).T+np.array(geometry["t_gt"])
    intrinsics = annotation["camera_data"]["intrinsics"]
    projected = np.c_[intrinsics["fx"]*q[:,0]/q[:,2]+intrinsics["cx"],
                      intrinsics["fy"]*q[:,1]/q[:,2]+intrinsics["cy"]]
    lookup = {tuple(sorted(edge)):i for i, edge in enumerate(EDGES12)}
    mapped_edges = [lookup[tuple(sorted((permutation[a], permutation[b])))] for a, b in EDGES12]
    assert set(mapped_edges) == set(range(12))
    def stats(values):
        return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)), maximum=float(np.max(values)))
    value = dict(schema="hough_wrong_case_geometry_audit_v1", complete=True, PASS=True,
        frame_id=fid, coordinate_frame="original_image_pixels", image_shape_hw=[480,640],
        source_sha256={str(path):digest(path) for path in (Path(__file__), prediction_path, geometry_path,
            annotation_path, raw_path, reference_image, protocol_path, provenance_path)},
        protocol_constants_match=True, exact_screenshot_M4_points_GT_mask_parity=True,
        raw_and_paper_image_bytes_identical=digest(raw_path)==digest(reference_image),
        self_test=self_test(), correction_config=CONFIG, prediction_points=points.tolist(),
        gt_diagnostic_only=dict(points=gt.tolist(), visibility=visibility.tolist(), supervision_mask=support.tolist(),
            interpretation="Visibility2 visible corner,1 occluded corner,0 unavailable/truncated. Point visibility is not a physical-edge visibility mask. Corner3 is outside480px image;5,6,8 are occluded."),
        semantic_permutation_diagnostic=dict(gt_oracle=True, deployable_correction=False,
            prediction_index_to_gt_index=permutation.tolist(), oracle_reordered_prediction=points[inverse].tolist(),
            inverse_for_reordered_prediction=inverse.tolist(),
            original_supervised_9point_error_px=stats(np.linalg.norm(points-gt,axis=1)[support]),
            oracle_reordered_same_supervision_error_px=stats(np.linalg.norm(points[inverse]-gt,axis=1)[support]),
            original_all8corner_error_px=stats(np.linalg.norm(points[:8]-gt[:8],axis=1)),
            oracle_all8corner_error_px=stats(np.linalg.norm(points[:8]-gt[permutation[:8]],axis=1)),
            residual_per_prediction_index_px=np.linalg.norm(points[:8]-gt[permutation[:8]],axis=1).tolist()),
        geometry_reference_comparison=dict(not_manual_point_GT=True,
            manual_vs_reprojected_all8_px=stats(np.linalg.norm(gt[:8]-projected,axis=1)),
            prediction_indexed_all8_px=stats(np.linalg.norm(points[:8]-projected,axis=1)),
            prediction_oracle_permuted_all8_px=stats(np.linalg.norm(points[:8]-projected[permutation[:8]],axis=1))),
        graph_invariance=dict(edges12=[list(e) for e in EDGES12], original_edge_to_permuted_edge_index=mapped_edges,
            exact_unordered12_edge_set_invariant=True,
            side8_fullgraph_indices=[1,3,5,7,8,9,10,11],
            side8_after_permutation_fullgraph_indices=[mapped_edges[i] for i in (1,3,5,7,8,9,10,11)],
            height_roles_remain_height=True, depth_roles_become_width=True,
            meaning="A label permutation changes semantic corner/line roles while exactly preserving an unordered cuboid12-edge graph. This does not assert that the measured predicted coordinates coincide with GT or that a90-degree rotation is an allowed physical symmetry for unequal width/depth."),
        observations=["Both semantic index mismatch and residual local coordinate error exist; the earlier 'not a localisation failure' description is too strong.",
            "The screenshot267.929px is the median of7supervised corners plus center, not all8corners.",
            "A raw line support score can be high for a wrong semantic permutation. It cannot by itself identify width versus depth or camera-facing corner labels."],
        limits=["Oracle relabel uses GT solely to decompose this failure and is not applied to any stored model prediction.",
            "Canny/Hough evidence can come from grid slats, stools, the red barrier, truck, shadows and reflected contrast; a line has no pallet role by itself.",
            "An infinite supporting line may pass a corner without any observed segment there. Association explicitly requires in-frame projected overlap.",
            "Low edge evidence is inconclusive under occlusion/truncation; do not reject an amodal corner solely for lacking an image gradient.",
            "Any fixed8px local refinement cannot relocate a corner by the approximately270px required to correct a semantic permutation.",
            "This one requested case is descriptive. Calibrate future rejection/correction rules on separate synthetic calibration data and freeze them before paired multi-session evaluation, with missing and coverage denominators retained."])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/"GEOMETRY_AUDIT.json"
    if path.exists():
        raise FileExistsError("Preserve the existing geometry audit; do not silently overwrite")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(path=str(path), sha256=digest(path), self_test=value["self_test"],
                         semantic_errors=value["semantic_permutation_diagnostic"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.output_dir:
        geometry_audit(args.output_dir.resolve())
    else:
        parser.error("Use --self-test or --output-dir")
