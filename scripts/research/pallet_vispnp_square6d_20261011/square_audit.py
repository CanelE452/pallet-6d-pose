"""Read-only GREEN0918 B0/B1 audit; record the unresolved B2/B3 conflict.

No pose solver, model, prediction coordinates, or annotation writer is imported.
The two public outputs contain hashes/counts and an explicitly empty pose set.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "_docs/experiments/pallet_vispnp_square6d_20261011"
STATUS = "BLOCKED_REFERENCE_PROCEDURE_CONFLICT"
SOURCE_DIRECTORY = "outputs/annotations/0918_dataset_square"
SNAPSHOT = "_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json"
SYMMETRY = "_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
POSE_ROOT = "data/pallet/results/paper_pose_metric_closure_v1"
GENERATOR = "scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py"
MANIFEST_GENERATOR = "scripts/paper/pose_metric_closure_v1/build_axis_review_manifest.py"
SECONDARY_QA = "scripts/annotate/audit_gt_data.py"
AXIS_LOCK = POSE_ROOT + "/GT_AXIS_RESOLUTION_LOCK.json"
AXIS_MANIFEST = POSE_ROOT + "/AXIS_REVIEW_MANIFEST.json"
COORDINATE_RULE = (
    "Only corner channels 0..7 with source=manual_click, visibility!=0, "
    "finite non-sentinel xy; in frame means 0<=x<raw_width and 0<=y<raw_height. "
    "Center8 and PnP-derived/unknown points do not count."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def binding(path, public_path):
    return {"path": public_path, "sha256": sha(path), "bytes": Path(path).stat().st_size}


def evidence(path, public_path, snippets):
    text = Path(path).read_text()
    lines = text.splitlines()
    records = []
    for snippet in snippets:
        matches = [i + 1 for i, line in enumerate(lines) if snippet in line]
        if len(matches) != 1:
            raise RuntimeError(f"Existing source contract missing/ambiguous: {public_path}: {snippet}")
        records.append({"line": matches[0], "statement": lines[matches[0] - 1].strip()})
    return {**binding(path, public_path), "statements": records}


def manual_rows(source, snapshot, track):
    labels = sorted((source / SOURCE_DIRECTORY).glob("*.json"))
    if len(labels) != 119:
        raise RuntimeError(f"Expected 119 square annotations; found {len(labels)}")
    frozen = {str(row["id"]): row for row in snapshot["records"]}
    if len(frozen) != 119 or set(frozen) != {path.stem for path in labels}:
        raise RuntimeError("Square snapshot membership differs from actual annotations")
    rows = []
    for label in labels:
        image = label.with_suffix(".png")
        record = frozen[label.stem]
        for key, path in (("annotation", label), ("image", image)):
            actual = sha(path)
            if actual != record[key]["sha256"]:
                raise RuntimeError(f"Frozen square {key} differs: {label.stem}")
            track[path] = actual
        doc = read(label)
        if doc.get("schema_version") != "real_pallet_gt_v2" or len(doc.get("objects", [])) != 1:
            raise RuntimeError(f"Unexpected annotation schema: {label.stem}")
        obj = doc["objects"][0]
        if obj.get("object_type", doc.get("object_type")) != "plastic_standard_110x110x15":
            raise RuntimeError(f"Square object type drift: {label.stem}")
        dims = obj["physical_dimensions_m"]
        if [dims["x"], dims["z"], dims["y"]] != [1.1, 1.1, .15]:
            raise RuntimeError(f"Registered square dimensions drift: {label.stem}")
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        camera = doc["camera_data"]
        if [height, width] != [camera["height"], camera["width"]]:
            raise RuntimeError(f"Square image/camera size differs: {label.stem}")
        intrinsics = camera["intrinsics"]
        K = [[float(intrinsics["fx"]), 0., float(intrinsics["cx"])],
             [0., float(intrinsics["fy"]), float(intrinsics["cy"])], [0., 0., 1.]]
        if not all(math.isfinite(v) for line in K for v in line) or K[0][0] <= 0 or K[1][1] <= 0:
            raise RuntimeError(f"Invalid existing K: {label.stem}")
        entries = obj["keypoint_annotations"]
        if len(entries) != 9:
            raise RuntimeError(f"Expected nine corner/center records: {label.stem}")
        declared = []
        in_frame = []
        manual_payload = []
        for corner, entry in enumerate(entries[:8]):
            xy = entry.get("xy")
            valid = (entry.get("source") == "manual_click" and entry.get("visibility", 0) != 0
                     and isinstance(xy, list) and len(xy) == 2
                     and all(math.isfinite(float(v)) for v in xy) and xy != [-1, -1])
            if valid:
                declared.append(corner)
                manual_payload.append({"corner": corner, "xy": xy,
                                       "visibility": entry["visibility"], "source": "manual_click"})
                if 0 <= xy[0] < width and 0 <= xy[1] < height:
                    in_frame.append(corner)
        count = len(in_frame)
        rows.append({
            "id": label.stem, "session": doc.get("capture_session_id"),
            "image_sha256": track[image], "annotation_sha256": track[label],
            "manual_corners_sha256": canonical_sha(manual_payload),
            "camera_K_sha256": canonical_sha(K), "K": K, "raw_hw": [height, width],
            "manual_declared_count": len(declared), "manual_in_frame_count": count,
            "manual_declared_indices": declared, "manual_in_frame_indices": in_frame,
            "outside_manual_indices": sorted(set(declared) - set(in_frame)),
            "B1_eligible": count >= 4, "existing_reference_min6_satisfied": count >= 6,
            "existing_min6_shortfall_if_B1_eligible": max(0, 6 - count) if count >= 4 else None,
            "LOO_remaining_points_below4": count == 4,
            "canonical_pose_present": obj.get("canonical_pose") is not None,
            "pose_status": obj.get("pose_status"),
        })
    return rows


def audit(*, source_root, public_root, legacy_qa, doc):
    source = Path(source_root).resolve()
    public = Path(public_root).resolve()
    destination = Path(doc).resolve()
    if not destination.is_dir():
        raise RuntimeError("Root must create the new public output directory first")
    outputs = [destination / "SQUARE_INPUT_AUDIT.json", destination / "SQUARE_REFERENCE_POSES.json"]
    if any(path.exists() for path in outputs):
        raise RuntimeError("Square output already exists; preserve it and do not rerun")
    track = {}
    def bound(relative, base=public):
        path = base / relative
        track[path] = sha(path)
        return path
    snapshot_path = bound(SNAPSHOT)
    symmetry_path = bound(SYMMETRY)
    generator_path = bound(GENERATOR, source)
    manifest_generator_path = bound(MANIFEST_GENERATOR, source)
    axis_lock_path = bound(AXIS_LOCK, source)
    axis_manifest_path = bound(AXIS_MANIFEST, source)
    secondary_path = bound(SECONDARY_QA, source)
    legacy_path = Path(legacy_qa).resolve()
    track[legacy_path] = sha(legacy_path)
    rows = manual_rows(source, read(snapshot_path), track)
    square = next(row for row in read(symmetry_path)["objects"]
                  if row["object_type"] == "plastic_standard_110x110x15")
    if square["group_order"] != 4 or any(permutation[8] != 8 for permutation in square["permutations"]):
        raise RuntimeError("Existing square C4/center contract drift")
    rotation_determinants = []
    for rotation in square["rotations"]:
        R = rotation
        determinant = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
                       - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
                       + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
        orthogonality_error = max(abs(sum(R[k][i] * R[k][j] for k in range(3))
                                      - float(i == j)) for i in range(3) for j in range(3))
        if abs(determinant - 1) > 1e-12 or orthogonality_error > 1e-12:
            raise RuntimeError("Approved square rotation is not proper/orthogonal")
        rotation_determinants.append(determinant)
    minimum_snippet = "if usable.sum() < 6:"
    generator_proof = evidence(generator_path, GENERATOR, [
        "flags=cv2.SOLVEPNP_SQPNP)", "rvec, tvec = cv2.solvePnPRefineLM(model[usable]",
        "usable = np.isfinite(points).all(axis=1)", minimum_snippet,
        "chosen = min(solved, key=lambda k: solved[k][2])"])
    manifest_proof = evidence(manifest_generator_path, MANIFEST_GENERATOR, [
        'points = obj.get("keypoint_annotations") or []',
        'keypoints = [p.get("xy") for p in points[:9]]'])
    legacy_proof = evidence(legacy_path, "external_legacy_qa/qa_risk.py", [
        "flag=cv2.SOLVEPNP_SQPNP if len(obj_pts)>=6 else cv2.SOLVEPNP_ITERATIVE",
        'proj=np.asarray(o.get("projected_cuboid") or [],float)[:8]',
        "if len(rest)<4: loo.append(np.nan); continue", "s=1.4826*mad if mad>0 else np.nanstd(v)",
        'z_loo=rz("max_loo_norm"); z_rep=rz("reproj_median_norm")',
        'r["severity"]="RED" if (hard or zz>5) else ("AMBER" if zz>3 else "GREEN")'])
    # Parse and inspect code only; never import or call an existing QA/solver.
    ast.parse(legacy_path.read_text())
    secondary_proof = evidence(secondary_path, SECONDARY_QA, [
        "def audit_frame(path, tol_stored=1.0, tol_gross=10.0,",
        "tol_pose_deg=0.5, tol_pose_m=0.01):",
        'for key, zname in (("resolve_reproj_med", "z_resolve"),',
        "elif zz > 5:", 'elif zz > 3 or r.get("hard_flags"):'])
    axis_lock = read(axis_lock_path)
    if axis_lock["quality_condition"]["threshold_px"] != 5.0:
        raise RuntimeError("Existing rectangular quality threshold drift")
    rect_sources = Counter()
    for frame in read(axis_manifest_path)["frames_list"]:
        path = bound(frame["annotation"], source)
        rectangle = read(path)["objects"][0]
        rect_sources.update(entry.get("source", "missing")
                            for entry in rectangle.get("keypoint_annotations", [])[:8])
    declared_distribution = Counter(row["manual_declared_count"] for row in rows)
    inside_distribution = Counter(row["manual_in_frame_count"] for row in rows)
    eligible = [row for row in rows if row["B1_eligible"]]
    min6 = [row for row in eligible if row["existing_reference_min6_satisfied"]]
    incompatible = [row for row in eligible if not row["existing_reference_min6_satisfied"]]
    counts = {
        "frames": len(rows), "sessions": len({row["session"] for row in rows}),
        "session_counts": dict(Counter(row["session"] for row in rows)),
        "camera_K_unique_count": len({row["camera_K_sha256"] for row in rows}),
        "manual_declared_corners": sum(row["manual_declared_count"] for row in rows),
        "manual_in_frame_corners": sum(row["manual_in_frame_count"] for row in rows),
        "manual_declared_count_distribution": dict(sorted(declared_distribution.items())),
        "manual_in_frame_count_distribution": dict(sorted(inside_distribution.items())),
        "B1_eligible_manual_in_frame_ge4": len(eligible),
        "B1_ineligible_ids": [row["id"] for row in rows if not row["B1_eligible"]],
        "existing_reference_min6_satisfied": len(min6),
        "B1_eligible_but_existing_min6_incompatible": len(incompatible),
        "B1_eligible_existing_min6_shortfall_corners": sum(
            row["existing_min6_shortfall_if_B1_eligible"] for row in eligible),
        "shortfall_interpretation": "Arithmetic diagnostic only; no annotation request or new annotation performed.",
        "B1_eligible_with_unsupported_manual_only_LOO4": sum(
            row["LOO_remaining_points_below4"] for row in eligible),
        "canonical_pose_present": sum(row["canonical_pose_present"] for row in rows),
        "pose_status": dict(Counter(row["pose_status"] for row in rows)),
    }
    expected = (119, 602, 600, 118, 36, 82, 112, 30)
    actual = tuple(counts[key] for key in (
        "frames", "manual_declared_corners", "manual_in_frame_corners",
        "B1_eligible_manual_in_frame_ge4", "existing_reference_min6_satisfied",
        "B1_eligible_but_existing_min6_incompatible", "B1_eligible_existing_min6_shortfall_corners",
        "B1_eligible_with_unsupported_manual_only_LOO4"))
    if actual != expected:
        raise RuntimeError(f"Square manual-only audit differs from read-only discovery: {actual}")
    untouched = all(sha(path) == digest for path, digest in track.items())
    if not untouched:
        raise RuntimeError("An existing input changed during read-only square audit")
    conflicts = [
        "B1 permits four in-frame manual corners, but the actual rectangular reference lock/generator requires six finite corners. 82 of 118 B1-eligible frames fail that existing input condition.",
        "The actual rectangular manifest takes all keypoint_annotations.xy without a source filter; existing references include PnP-projected/extrapolated/unknown points. B2 allows only direct manual corners, so copying that input procedure would violate the new manual-only contract.",
        "Legacy qa_risk.py solves non-sentinel projected_cuboid rather than manual_click-only inputs. It uses ITERATIVE for four/five points; adapting it is not established as the same procedure.",
        "Removing one of four manual corners leaves three; the existing LOO code returns NaN. No pre-existing replacement QA/approval policy for these 30 frames was found or invented.",
        "The actual rectangular reference generator has no LOO/robust-z stage; the later audit_gt_data.py is a distinct QA procedure with different normalization and stored-pose checks.",
    ]
    audit_result = {
        "schema": "pallet_vispnp_square6d_square_input_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "status": STATUS,
        "B0_status": "PASS", "B1_status": "PASS", "B2_status": STATUS, "B3_status": STATUS,
        "coordinate_count_rule": COORDINATE_RULE, "counts": counts, "rows": rows,
        "source_bindings": {
            "square_snapshot": binding(snapshot_path, SNAPSHOT),
            "C4_contract": binding(symmetry_path, SYMMETRY),
            "rectangular_reference_generator": generator_proof,
            "rectangular_manifest_generator": manifest_proof,
            "rectangular_axis_lock": binding(axis_lock_path, AXIS_LOCK),
            "rectangular_axis_manifest": binding(axis_manifest_path, AXIS_MANIFEST),
            "legacy_QA": legacy_proof, "distinct_stored_pose_QA": secondary_proof,
        },
        "C4": {**{key: square[key] for key in ("object_type", "dimensions_m", "group_order", "rotations", "permutations")},
               "verified_rotation_determinants": rotation_determinants, "center8_fixed": True},
        "existing_rectangular_procedure": {
            "min_usable_finite_corners": 6, "solver": "SQPnP then RefineLM for each W/D hypothesis",
            "selection": "minimum mean Euclidean reprojection residual",
            "quality_bar_px": axis_lock["quality_condition"]["threshold_px"],
            "quality_bar_policy": "review only; no automatic residual/margin exclusion",
            "reference_generator_LOO_robust_z": False,
            "source_corner_counts_actual_319_manifest": dict(rect_sources),
        },
        "legacy_QA_contract": {
            "input": "non-sentinel projected_cuboid; manual_kps is counted but not the solver input",
            "solver_initial": "SQPnP if n>=6; ITERATIVE otherwise; RefineLM after both",
            "LOO_min_remaining_points": 4, "LOO_four_point_frame": "unavailable/NaN",
            "robust_z_inputs": ["max_loo_norm", "reproj_median_norm"],
            "normalization": "annotated non-sentinel corner bbox diagonal",
            "round_before_robust_z": {"max_loo_norm": 5, "reproj_median_norm": 5},
            "robust_scale": "1.4826*MAD; nanstd fallback if MAD==0",
            "RED": "hard flag or max(finite z_loo,z_reproj)>5", "AMBER": "max(finite z_loo,z_reproj)>3",
            "distinct_stored_pose_QA_defaults": {"excess_px": 1.0, "gross_median_px": 10.0,
                "stored_resolved_rotation_deg": .5, "stored_resolved_translation_m": .01,
                "robust_z_red": 5, "robust_z_amber": 3},
            "new_thresholds": False, "QA_executed": False,
        },
        "conflicts": conflicts, "existing_inputs_unchanged": untouched,
        "existing_input_files_hash_checked_before_after": len(track),
        "execution": {"prediction_coordinates_read_for_manual_counts": False,
            "model_inference_calls": 0, "PnP_calls": 0, "F_calls": 0, "new_reference_poses": 0,
            "new_manual_annotations": 0, "review_sheets_created": 0,
            "axis_approval": "NOT_REQUESTED_B2_B3_BLOCKED", "square_6D_evaluation": "NOT_RUN",
            "RGB_published": False},
    }
    blocked_reference = {
        "schema": "pallet_vispnp_square6d_square_reference_status_v1", "status": STATUS,
        "reference_poses_generated": False, "frames": {}, "reference_pose_count": 0,
        "B1_eligible_ids": [row["id"] for row in eligible],
        "existing_min6_satisfied_ids": [row["id"] for row in min6],
        "B1_eligible_but_existing_min6_incompatible_ids": [row["id"] for row in incompatible],
        "audit": "SQUARE_INPUT_AUDIT.json", "conflicts": conflicts,
        "registered_WDH_m": [1.1, 1.1, .15], "symmetry_group": "C4 proper rotations",
        "canonical_pose_approval": "NOT_REQUESTED_B2_B3_BLOCKED",
        "human_axis_approval_received": False, "square_6D_evaluation_executed": False,
        "new_QA_or_threshold_invented": False, "new_manual_annotation_requested": False,
    }
    for path, payload in zip(outputs, (audit_result, blocked_reference)):
        with path.open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": STATUS, "B0": "PASS", "B1": "PASS", "counts": counts,
                      "poses_generated": 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=os.environ.get("PALLET_SOURCE_ROOT"))
    parser.add_argument("--public-root", type=Path, default=ROOT)
    parser.add_argument("--legacy-qa", type=Path, default=os.environ.get("PALLET_LEGACY_QA_SOURCE"))
    parser.add_argument("--doc", type=Path, default=DOC)
    args = parser.parse_args()
    if args.source_root is None or args.legacy_qa is None:
        parser.error("--source-root and --legacy-qa (or their PALLET_* environment variables) are required")
    audit(source_root=args.source_root, public_root=args.public_root, legacy_qa=args.legacy_qa, doc=args.doc)


if __name__ == "__main__":
    main()
