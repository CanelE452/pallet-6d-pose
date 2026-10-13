"""Postseal visibility audit on a geometry-selected silhouette-corner subset.

Subset selection consumes only initial pose/model, camera, fixed H, detector
coordinates and actual canvas size. No RGB state, confidence or evaluation
label participates in selection. This runner changes no inference policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .scoring import distribution


def initial_geometric_boundary(points, K, initial_pose, self_hidden, image_hw):
    """Return initial projected-cuboid hull vertices and independent gates."""
    q = np.asarray(points, float)[:8]
    h, w = map(int, image_hw)
    xyz = C.cuboid(*initial_pose["cf_extents"])
    R, t = np.asarray(initial_pose["R_cf"]), np.asarray(initial_pose["centroid"])
    camera_points = xyz @ R.T + t
    projected = C.project(xyz, R, t, K)
    projection_valid = np.isfinite(projected).all(1) & (camera_points[:, 2] > 1e-6)
    if not projection_valid.all():
        raise ValueError("Require all eight initial cuboid corners projectable for hull selection")
    hull_ids = cv2.convexHull(projected.astype(np.float32), returnPoints=False).reshape(-1)
    hull = np.zeros(8, bool)
    hull[hull_ids] = True
    raw_valid = np.isfinite(q).all(1) & ~np.all(q == -1., axis=1)
    raw_inside = raw_valid & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    projected_inside = projection_valid & (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
    hidden = np.asarray(self_hidden, bool)
    selected = hull & ~hidden & raw_inside & projected_inside
    return dict(hull_ids=hull_ids.tolist(), selected_ids=np.flatnonzero(selected).tolist(),
                hull=hull, hidden=hidden, raw_valid=raw_valid, raw_inside=raw_inside,
                projection_valid=projection_valid, projected_inside=projected_inside,
                projected=projected)


def confusion(records, score_key, threshold):
    """DIRECT versus EXTERNAL only; absent class rates stay undefined."""
    known = [r for r in records if r["human_state"] in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED")]
    y = np.asarray([r["human_state"] == "DIRECT_VISIBLE" for r in known], bool)
    probability = np.asarray([r[score_key] for r in known], float)
    admit = probability >= threshold
    tp, fn = int((y & admit).sum()), int((y & ~admit).sum())
    fp, tn = int((~y & admit).sum()), int((~y & ~admit).sum())
    positive, negative = probability[y], probability[~y]
    auc = float(((positive[:, None] > negative[None, :]) + .5 *
                 (positive[:, None] == negative[None, :])).mean()) if len(positive) and len(negative) else None
    return dict(known_points=len(known), DIRECT=len(positive), EXTERNAL=len(negative),
                threshold=threshold, TP=tp, FP=fp, FN=fn, TN=tn,
                direct_recall=tp / (tp + fn) if tp + fn else None,
                external_recall=tn / (tn + fp) if tn + fp else None,
                visible_precision=tp / (tp + fp) if tp + fp else None,
                visibility_ranking_AUC=auc,
                DIRECT_score=distribution(positive), EXTERNAL_score=distribution(negative))


def audit(run, destination):
    if destination.exists():
        raise FileExistsError("Preserve completed analysis: " + str(destination))
    seal = C.read(run / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["complete"] and seal["GT_read_canary"] == "PASS"
    assert C.sha(run / "PREDICTIONS.jsonl.gz") == seal["sha256"]
    protocol = C.read(run / "PROTOCOL.json")
    assert protocol["partition"] == "test"
    namespace = run.parent
    cohort = C.read(namespace / "COHORT_PROTOCOL.json")
    model_path = namespace / "visibility_logistic_CAL46_v2.json"
    assert protocol["model_sha256"][str(model_path)] == C.sha(model_path)
    visibility_threshold = C.read(model_path)["visible_threshold"]
    frames = C.read(run / "ISOLATED_TEST_INPUTS.json")["frames"]
    assert len(frames) == 49 and {f["id"] for f in frames} == set(protocol["ids"])
    assert not {f["id"] for f in frames} & set(cohort["excluded_severe_ids"])
    # Evaluation truth is opened only after prediction-seal verification.
    scoring = C.read(run / "ISOLATED_TEST_SCORING.json")["rows"]
    assert set(scoring) == {f["id"] for f in frames}
    predictions = {(r["id"], r["method"]): r for r in C.rows(run / "PREDICTIONS.jsonl.gz")}
    previous_diagnostics = C.read(run / "OBSERVATION_DIAGNOSTICS.json")
    point_metadata = {(r["id"], r["corner"]): r for r in previous_diagnostics["records"]}
    points, frame_records = [], []
    gate_counts = Counter()
    for f in frames:
        fid = f["id"]
        initial = f["initial_poses"]["N3_SUBPIX"]
        xyz = C.cuboid(*initial["cf_extents"])
        H = C.self_hidden(xyz, initial["R_cf"], initial["centroid"])
        selection = initial_geometric_boundary(f["points"]["N3_SUBPIX"], f["K"], initial, H, f["raw_hw"])
        rgb = predictions[fid, "SEGMENTED_RGB_CURRICULUM"]
        joint = predictions[fid, "JOINT_WD_BRANCH"]
        assert rgb["self_hidden"] == H.tolist()
        assert joint["self_hidden"] == H.tolist()
        for k in range(8):
            reason = "NON_HULL_INTERIOR" if not selection["hull"][k] else "FIXED_SELF_HIDDEN" if H[k] else "RAW_PREDICTION_MISSING" if not selection["raw_valid"][k] else "RAW_PREDICTION_OUTSIDE" if not selection["raw_inside"][k] else "INITIAL_PROJECTION_OUTSIDE" if not selection["projected_inside"][k] else "SELECTED_GEOMETRIC_BOUNDARY"
            gate_counts[reason] += 1
        frame_records.append(dict(id=fid, session=f["session"], actual_image_hw=f["raw_hw"],
                                  initial_hull_native_ids=selection["hull_ids"], selected_native_ids=selection["selected_ids"],
                                  fixed_H=H.tolist(), original_q_inside=selection["raw_inside"].tolist(),
                                  initial_projection_inside=selection["projected_inside"].tolist()))
        for k in selection["selected_ids"]:
            truth = scoring[fid]["metadata"]
            previous = point_metadata[fid, k]
            points.append(dict(id=fid, session=f["session"], severity=previous["severity"], corner=k,
                human_state=truth["human_states_native"][k], native_reference_corner=truth["native_to_reference_corner_ids"][k],
                fixed_reference_correspondence_compatible=previous["fixed_reference_correspondence_compatible"],
                original_xy=f["points"]["N3_SUBPIX"][k], initial_projected_xy=selection["projected"][k].tolist(),
                RGB_p_visible=rgb["appearance"]["visibility_score"][k], RGB_state=rgb["appearance"]["states"][k],
                RGB_boundary_score=rgb["appearance"]["boundary_score"][k],
                joint_P_usable=joint["joint_observation"]["P_usable"][k],
                joint_state=joint["appearance"]["states"][k],
                raw_error_px=previous["raw_error_px"], corrected_error_px=previous["corrected_error_px"]))
    sessions = {}
    for session in sorted({f["session"] for f in frames}):
        selected = [r for r in points if r["session"] == session]
        sessions[session] = dict(frames=sum(f["session"] == session for f in frames), selected_points=len(selected),
            human_state_counts=dict(Counter(r["human_state"] for r in selected)),
            RGB_visibility=confusion(selected, "RGB_p_visible", visibility_threshold),
            joint_usable_as_visibility_proxy=confusion(selected, "joint_P_usable", .45))
    failure_cases = [r for r in points if (r["human_state"] == "DIRECT_VISIBLE" and r["RGB_p_visible"] < visibility_threshold)
                     or (r["human_state"] == "EXTERNAL_OCCLUDED" and r["RGB_p_visible"] >= visibility_threshold)]
    for r in failure_cases:
        r["failure_type"] = "FALSE_VISIBLE_EXTERNAL" if r["human_state"] == "EXTERNAL_OCCLUDED" else "FALSE_ABSENT_DIRECT"
    compatible = [r for r in points if r["fixed_reference_correspondence_compatible"]]
    report = dict(schema="NONSEVERE_TEST49_GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1", postseal_analysis=True,
        primary_predictions_unchanged=True, new_training_or_policy_selection=0,
        population_selection=dict(inputs_only=["original_native_points", "K", "initial_pose", "fixed_initial_H", "actual_image_hw"],
            algorithm="cv2.convexHull(initial_projected_cuboid8,returnPoints=False); vertex IDs intersect~fixedH and valid/inframe originalq and initialprojection",
            hull_precision="OpenCV float32 index-returning convex hull", appearance_or_probability_used=False,
            human_label_used=False, geometry_projection_used_for_pixel_sampling_or_pose_fit=False,
            interpretation="Predicted initial cuboid silhouette vertices, conditional on initial model/pose. This does not certify the actual pixel is an object/background boundary."),
        counts=dict(frames=49, original_corner_points=392, hull_vertex_points=sum(len(r["initial_hull_native_ids"]) for r in frame_records),
            selected_points=len(points), frames_with_selected_points=sum(bool(r["selected_native_ids"]) for r in frame_records),
            gate_reason_counts=dict(gate_counts), human_state_counts=dict(Counter(r["human_state"] for r in points)),
            severity_counts=dict(Counter(r["severity"] for r in points)),
            self_labels_remaining_despite_fixed_H_exclusion=sum(r["human_state"] == "SELF_OCCLUDED" for r in points),
            all_TEST49_EXTERNAL_labels=sum(s == "EXTERNAL_OCCLUDED" for r in scoring.values() for s in r["metadata"]["human_states_native"])),
        RGB_visibility=confusion(points, "RGB_p_visible", visibility_threshold),
        joint_usable_as_visibility_proxy=dict(target="DIRECT_VISIBLE and raw coordinate error<=8px; not pure external occlusion", **confusion(points, "joint_P_usable", .45)),
        by_session=sessions, failure_cases=failure_cases,
        reference_compatible42_posthoc_diagnostic=dict(primary49_unchanged=True, selected_points=len(compatible),
            human_state_counts=dict(Counter(r["human_state"] for r in compatible)),
            RGB_visibility=confusion(compatible, "RGB_p_visible", visibility_threshold)),
        bindings=dict(predictions_sha256=seal["sha256"], protocol_sha256=C.sha(run / "PROTOCOL.json"),
            TEST_inputs_sha256=C.sha(run / "ISOLATED_TEST_INPUTS.json"), TEST_scoring_sha256=C.sha(run / "ISOLATED_TEST_SCORING.json"),
            visibility_model_sha256=C.sha(model_path), joint_model_sha256=C.sha(namespace / "joint_model/MODEL.json"),
            observation_diagnostics_sha256=C.sha(run / "OBSERVATION_DIAGNOSTICS.json"), analysis_source_sha256=C.sha(__file__)),
        limitations=["TEST49 is reused historical evaluation with4sessions; only12EXTERNAL point labels. No new threshold or model was selected here.",
            "Only original-geometric self mask is excluded.6humanSELF and5UNANNOTATED labels remain in the boundary population; binary visibility metrics use DIRECT/EXTERNAL only, without silently redefining selection.",
            "All3RGB failure cases have incompatible historical reference/native phase; coordinate errors19.55/222.17/275.07px can confound requested-point visibility. Compatible42 is additional posthoc diagnosis, not a new primary population.",
            "Geometric hull membership does not establish rawquery localization or physical3D ownership. Visible mask support at an inaccurate query can be high while the requested corner is hidden.",
            "Low p_visible remains UNKNOWN/occlusion-suspected evidence; joint P_usable must not be treated as a pure external-occlusion probability.",
            "All12EXTERNAL labels happen to remain in this selected subset; the10/12external recall was recomputed here rather than transferred from whole49 metrics.",
            "No realCAL in-sample performance, sourceTEST or FINAL evaluation is included."],
        frame_selection_records=frame_records, records=points)
    C.write(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=C.DOC / "nonsevere_v1/test_reused_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.run / "GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1.json"
    report = audit(args.run.resolve(), destination.resolve())
    print("BOUNDARY_AUDIT", report["counts"], report["RGB_visibility"], destination, flush=True)


if __name__ == "__main__":
    main()
