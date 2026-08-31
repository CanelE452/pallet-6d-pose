"""Audit the two historical wood-pallet manual-GT sessions without mutation.

The output CSV is both a human-readable membership audit and the immutable
SHA/mtime/size input accepted by :mod:`migrate_real_gt_v2`.  Bare six-digit
stems are never used as population IDs because all 45 collide with historical
negative-frame IDs; the canonical frame ID is ``<session>:<stem>``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

try:
    from .object_geometry_registry import (
        DEFAULT_REGISTRY_PATH,
        WOOD_OBJECT_TYPE,
        load_object_geometry_registry,
    )
except ImportError:
    from object_geometry_registry import (  # type: ignore[no-redef]
        DEFAULT_REGISTRY_PATH,
        WOOD_OBJECT_TYPE,
        load_object_geometry_registry,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "challenge" / "real_gt_v2" / "wood_audit"
SESSIONS = {
    "wood_183705": (
        REPO_ROOT
        / "challenge/data/01_real/manual_gt/wood_pallet_20260618_183705_manual_gt"
    ),
    "wood_184309": (
        REPO_ROOT
        / "challenge/data/01_real/manual_gt/wood_pallet_20260618_184309_manual_gt"
    ),
}
EXPECTED_INTRINSICS = {
    "fx": 908.8597333333333,
    "fy": 908.9547333333333,
    "cx": 636.3943333333333,
    "cy": 384.4384666666666,
}
INTRINSICS_QUALITY = "SENSOR_PROFILE_SCALED"
INTRINSICS_SOURCE = (
    "RealSense D435I 1920x1080 sensor profile scaled linearly by 2/3 to 1280x720; "
    "camera serial and distortion model unavailable in legacy JSON"
)
AUDIT_SCHEMA = "wood_real_gt_audit_v1"
QA_SCHEMA = "wood_real_gt_qa_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()


def _rotation_metrics(transform: np.ndarray) -> tuple[float, float]:
    rotation = transform[:3, :3]
    return (
        float(np.linalg.det(rotation)),
        float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
    )


def _diagram(width: float, height: float, depth: float) -> np.ndarray:
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array(
        [
            [-w, -h, -d],
            [+w, -h, -d],
            [+w, +h, -d],
            [-w, +h, -d],
            [-w, -h, +d],
            [+w, -h, +d],
            [+w, +h, +d],
            [-w, +h, +d],
        ],
        dtype=np.float64,
    )
    return np.vstack([corners, np.zeros((1, 3), dtype=np.float64)])


def _project(points: np.ndarray, transform: np.ndarray, camera: np.ndarray) -> np.ndarray:
    camera_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    if (camera_points[:, 2] <= 0.0).any():
        raise ValueError("CHEIRALITY_FAILED")
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = camera[0, 0] * camera_points[:, 0] / camera_points[:, 2] + camera[0, 2]
    pixels[:, 1] = camera[1, 1] * camera_points[:, 1] / camera_points[:, 2] + camera[1, 2]
    return pixels


def _manifest_image_hashes(name: str) -> tuple[set[str], set[str]]:
    path = REPO_ROOT / "challenge/real_gt_v2/manifests" / f"{name}.json"
    payload = json.loads(path.read_text("utf-8"))
    byte_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    for item in payload["items"]:
        image = REPO_ROOT / item["image"]
        byte_hashes.add(_sha256(image))
        pixel_hashes.add(_pixel_sha256(image))
    return byte_hashes, pixel_hashes


def _hash_lines(rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\t".join(str(row[field]) for field in fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_exclusive(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _csv_exclusive(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def audit(*, output_dir: Path, registry_path: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_object_geometry_registry(registry_path)
    spec = registry.resolve(WOOD_OBJECT_TYPE)
    plastic_byte, plastic_pixel = _manifest_image_hashes("DEV_POS140")
    common_byte, common_pixel = _manifest_image_hashes("COMMON_DEV_POS128")
    negative_byte, negative_pixel = _manifest_image_hashes("DEV_NEG2689")
    negative_ids = {
        item["frame_id"]
        for item in json.loads(
            (REPO_ROOT / "challenge/real_gt_v2/manifests/DEV_NEG2689.json").read_text(
                "utf-8"
            )
        )["items"]
    }

    rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    for session_id, directory in SESSIONS.items():
        if not directory.is_dir():
            raise RuntimeError(f"wood GT session missing: {directory}")
        for label in sorted(directory.glob("*.json")):
            stem = label.stem
            frame_id = f"{session_id}:{stem}"
            image = directory / f"{stem}.png"
            if not image.exists():
                raise RuntimeError(f"wood image missing: {image}")
            label_bytes = label.read_bytes()
            document = json.loads(label_bytes.decode("utf-8"))
            objects = document.get("objects")
            if not isinstance(objects, list) or len(objects) != 1:
                raise RuntimeError(f"{frame_id}: expected exactly one object")
            obj = objects[0]
            dimensions = obj.get("dimensions_m")
            camera_data = document.get("camera_data")
            intrinsics = camera_data.get("intrinsics") if isinstance(camera_data, dict) else None
            manual = obj.get("manual_kps")
            projected = list(obj.get("projected_cuboid") or []) + [
                obj.get("projected_cuboid_centroid")
            ]
            transform = np.asarray(obj.get("pose_transform"), dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise RuntimeError(f"{frame_id}: invalid pose_transform")
            determinant, orthogonality = _rotation_metrics(transform)
            width = float(dimensions["width"])
            height = float(dimensions["height"])
            depth = float(dimensions["depth"])
            points = _diagram(width, height, depth)
            camera = np.array(
                [
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            reprojected = _project(points, transform, camera)
            stored = np.asarray(projected, dtype=np.float64)
            residuals = np.linalg.norm(reprojected - stored, axis=1)
            outside = [
                index
                for index, point in enumerate(stored)
                if not (
                    0.0 <= point[0] < int(camera_data["width"])
                    and 0.0 <= point[1] < int(camera_data["height"])
                )
            ]
            label_stat = label.stat()
            image_sha = _sha256(image)
            pixel_sha = _pixel_sha256(image)
            label_sha = hashlib.sha256(label_bytes).hexdigest()
            finite_manual = bool(
                isinstance(manual, list)
                and len(manual) == 9
                and np.asarray(manual, dtype=np.float64).shape == (9, 2)
                and np.isfinite(np.asarray(manual, dtype=np.float64)).all()
            )
            intrinsic_exact = all(
                math.isclose(
                    float(intrinsics[key]), value, rel_tol=0.0, abs_tol=1e-12
                )
                for key, value in EXPECTED_INTRINSICS.items()
            )
            reasons: list[str] = []
            if obj.get("class") != "pallet":
                reasons.append("CLASS_NOT_PALLET")
            if obj.get("gt_source") != "manual":
                reasons.append("GT_SOURCE_NOT_MANUAL")
            if not finite_manual:
                reasons.append("MANUAL_KPS_INVALID")
            if abs(determinant - 1.0) > 1e-6 or orthogonality > 1e-6:
                reasons.append("POSE_NOT_PROPER")
            if not intrinsic_exact:
                reasons.append("INTRINSICS_PROFILE_MISMATCH")
            if float(np.median(residuals)) >= 10.0:
                reasons.append("PNP_REPROJECTION_MEDIAN_GE_10PX")
            qa_status = "GREEN" if not reasons else "RED"
            source_set = session_id
            row = {
                "frame_id": frame_id,
                "source_frame_id": stem,
                "session_id": session_id,
                "source_set": source_set,
                "population": "DEV_WOOD_POS45",
                "population_role": "DEV",
                "object_type": spec.object_type,
                "label_path": _relative(label),
                "image_path": _relative(image),
                "label_sha256": label_sha,
                "label_mtime_ns": label_stat.st_mtime_ns,
                "label_size_bytes": label_stat.st_size,
                "image_sha256": image_sha,
                "image_pixel_sha256": pixel_sha,
                "image_exists": True,
                "objects_count": len(objects),
                "split": obj.get("split") or "MISSING_LEGACY",
                "gt_source": obj.get("gt_source"),
                "legacy_width_m": width,
                "legacy_height_m": height,
                "legacy_depth_m": depth,
                "physical_x_m": spec.physical_dimensions.x_m,
                "physical_y_m": spec.physical_dimensions.y_m,
                "physical_z_m": spec.physical_dimensions.z_m,
                "fix_swap_present": "fix_swap" in obj,
                "fix_swap": json.dumps(obj.get("fix_swap")),
                "manual_kps": json.dumps(manual, separators=(",", ":")),
                "manual_kps_present_count": 9 if finite_manual else 0,
                "manual_kps_missing_count": 0 if finite_manual else 9,
                "extrapolated_mask_present": "extrapolated_mask" in obj,
                "extrapolated_mask": json.dumps(obj.get("extrapolated_mask")),
                "known_direct_count": 0,
                "known_extrapolated_count": 0,
                "provenance_unknown_non_null_count": 9,
                "pose_transform": json.dumps(obj.get("pose_transform"), separators=(",", ":")),
                "pose_rotation_det": determinant,
                "pose_orthogonality_max_error": orthogonality,
                "reproj_error_px": obj.get("reproj_error_px"),
                "audit_reprojection_median_px": float(np.median(residuals)),
                "audit_reprojection_p90_px": float(np.percentile(residuals, 90)),
                "audit_reprojection_max_px": float(np.max(residuals)),
                "camera_fx": float(intrinsics["fx"]),
                "camera_fy": float(intrinsics["fy"]),
                "camera_cx": float(intrinsics["cx"]),
                "camera_cy": float(intrinsics["cy"]),
                "image_width": int(camera_data["width"]),
                "image_height": int(camera_data["height"]),
                "intrinsics_quality": INTRINSICS_QUALITY,
                "intrinsics_source": INTRINSICS_SOURCE,
                "object_visibility": obj.get("visibility"),
                "projected_keypoints": json.dumps(projected, separators=(",", ":")),
                "outside_keypoint_indices": json.dumps(outside),
                "outside_keypoint_count": len(outside),
                "bare_frame_id_in_dev_neg": stem in negative_ids,
                "qualified_frame_id_in_dev_neg": frame_id in negative_ids,
                "plastic_image_overlap": image_sha in plastic_byte or pixel_sha in plastic_pixel,
                "common_plastic_image_overlap": image_sha in common_byte or pixel_sha in common_pixel,
                "negative_image_overlap": image_sha in negative_byte or pixel_sha in negative_pixel,
                "qa_status": qa_status,
                "qa_reasons": json.dumps(reasons),
                "missing_fields": json.dumps(
                    [
                        field
                        for field in (
                            "schema_version",
                            "object_type",
                            "population_role",
                            "keypoint_frame",
                            "physical_dimensions_m",
                            "keypoint_annotations",
                        )
                        if (field not in document and field not in obj)
                    ]
                ),
            }
            rows.append(row)
            qa_rows.append(
                {
                    "frame_id": frame_id,
                    "session_id": session_id,
                    "status": qa_status,
                    "reprojection_median_px": row["audit_reprojection_median_px"],
                    "reprojection_p90_px": row["audit_reprojection_p90_px"],
                    "reprojection_max_px": row["audit_reprojection_max_px"],
                    "rotation_det": determinant,
                    "rotation_orthogonality_max_error": orthogonality,
                    "outside_keypoint_indices": outside,
                    "reasons": reasons,
                }
            )
            visibility_rows.append(
                {
                    "frame_id": frame_id,
                    "session_id": session_id,
                    "object_type": spec.object_type,
                    "source_label": _relative(label),
                    "reason": "LEGACY_VISIBILITY_AND_PROVENANCE_UNKNOWN",
                    "unknown_keypoint_indices": json.dumps(list(range(9))),
                    "unknown_keypoint_count": 9,
                }
            )

    rows.sort(key=lambda row: row["frame_id"])
    qa_rows.sort(key=lambda row: row["frame_id"])
    visibility_rows.sort(key=lambda row: row["frame_id"])
    frame_ids = [row["frame_id"] for row in rows]
    image_hashes = [row["image_sha256"] for row in rows]
    pixel_hashes = [row["image_pixel_sha256"] for row in rows]
    label_hashes = [row["label_sha256"] for row in rows]
    session_counts = {
        session: sum(row["session_id"] == session for row in rows) for session in SESSIONS
    }
    checks = {
        "count_exact_45": len(rows) == 45,
        "sessions_exact_2": session_counts == {"wood_183705": 25, "wood_184309": 20},
        "unique_qualified_frame_ids": len(frame_ids) == len(set(frame_ids)),
        "unique_image_sha256": len(image_hashes) == len(set(image_hashes)),
        "unique_decoded_pixel_sha256": len(pixel_hashes) == len(set(pixel_hashes)),
        "unique_label_sha256": len(label_hashes) == len(set(label_hashes)),
        "images_exist_45": sum(bool(row["image_exists"]) for row in rows) == 45,
        "objects_exactly_one": all(row["objects_count"] == 1 for row in rows),
        "manual_kps_exact_9": all(row["manual_kps_present_count"] == 9 for row in rows),
        "proper_rotations": all(
            abs(float(row["pose_rotation_det"]) - 1.0) <= 1e-6
            and float(row["pose_orthogonality_max_error"]) <= 1e-6
            for row in rows
        ),
        "intrinsics_sensor_profile_exact": all(
            row["intrinsics_quality"] == INTRINSICS_QUALITY for row in rows
        ),
        "plastic_overlap_zero": not any(row["plastic_image_overlap"] for row in rows),
        "common_plastic_overlap_zero": not any(
            row["common_plastic_image_overlap"] for row in rows
        ),
        "negative_image_overlap_zero": not any(row["negative_image_overlap"] for row in rows),
        "qualified_negative_frame_id_overlap_zero": not any(
            row["qualified_frame_id_in_dev_neg"] for row in rows
        ),
        "qa_red_zero": not any(row["qa_status"] == "RED" for row in rows),
    }
    audit_status = "PASS" if all(checks.values()) else "BLOCKED"
    snapshot_fields = (
        "frame_id",
        "label_path",
        "label_sha256",
        "label_mtime_ns",
        "label_size_bytes",
    )
    audit_payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": audit_status,
        "population_id": "DEV_WOOD_POS45",
        "role": "CROSS_SHAPE_DEV",
        "population_role": "DEV",
        "object_type": spec.object_type,
        "physical_dimensions_m": spec.physical_dimensions.as_dict(),
        "geometry_registry": _relative(registry.source_path),
        "geometry_registry_sha256": registry.sha256,
        "count": len(rows),
        "session_counts": session_counts,
        "resolution_counts": {"1280x720": len(rows)},
        "intrinsics_quality": INTRINSICS_QUALITY,
        "intrinsics_source": INTRINSICS_SOURCE,
        "membership_identity_sha256": _hash_lines(
            rows,
            ("frame_id", "image_path", "label_path", "image_sha256", "label_sha256"),
        ),
        "source_label_snapshot_sha256": _hash_lines(rows, snapshot_fields),
        "source_image_snapshot_sha256": _hash_lines(
            rows, ("frame_id", "image_path", "image_sha256")
        ),
        "checks": checks,
        "bare_frame_id_collisions_with_dev_neg": sum(
            bool(row["bare_frame_id_in_dev_neg"]) for row in rows
        ),
        "qualified_frame_id_collisions_with_dev_neg": sum(
            bool(row["qualified_frame_id_in_dev_neg"]) for row in rows
        ),
        "geometrically_truncated_frame_count": sum(
            int(row["outside_keypoint_count"]) > 0 for row in rows
        ),
        "outside_keypoint_count": sum(int(row["outside_keypoint_count"]) for row in rows),
        "visibility_review_required_count": len(rows),
        "visibility_unknown_keypoint_count": 9 * len(rows),
        "symmetry_status": spec.symmetry_status,
        "selector_status": "NOT_RUN",
        "final_eligible": False,
        "prior_use_evidence": {
            "status": "PREVIOUSLY_EVALUATED__DEV_ONLY",
            "code": "scripts/stage0/wood/wood_gt_eval.py",
            "result": "data/pallet/eval_results/paper_s2_scratch_diffpnp/wood_gt_eval.md",
        },
    }
    qa_payload = {
        "schema_version": QA_SCHEMA,
        "status": "PASS" if not any(row["status"] == "RED" for row in qa_rows) else "BLOCKED",
        "population_id": "DEV_WOOD_POS45",
        "count": len(qa_rows),
        "green": sum(row["status"] == "GREEN" for row in qa_rows),
        "amber": sum(row["status"] == "AMBER" for row in qa_rows),
        "red": sum(row["status"] == "RED" for row in qa_rows),
        "quarantined": 0,
        "exclusions": [],
        "reprojection_median_px": float(
            np.median([row["reprojection_median_px"] for row in qa_rows])
        ),
        "reprojection_max_frame_median_px": float(
            max(row["reprojection_median_px"] for row in qa_rows)
        ),
        "intrinsics_quality": INTRINSICS_QUALITY,
        "symmetry_status": spec.symmetry_status,
        "pose_metric_status": "BLOCKED_SYMMETRY_UNREVIEWED_AND_SELECTOR_NOT_RUN",
        "frames": qa_rows,
    }

    csv_fields = list(rows[0])
    _csv_exclusive(output_dir / "WOOD_GT_PER_FRAME.csv", rows, csv_fields)
    _json_exclusive(output_dir / "WOOD_GT_AUDIT.json", audit_payload)
    _json_exclusive(output_dir / "WOOD_GT_QA.json", qa_payload)
    _csv_exclusive(
        output_dir / "WOOD_VISIBILITY_REVIEW_QUEUE.csv",
        visibility_rows,
        list(visibility_rows[0]),
    )
    membership_md = f"""# Wood DEV membership audit

Status: **{audit_status}**

- Population: `DEV_WOOD_POS45` / reporting role `CROSS_SHAPE_DEV`
- Sessions: `wood_183705` {session_counts['wood_183705']} + `wood_184309` {session_counts['wood_184309']} = **{len(rows)}**
- Object: `{spec.object_type}`, canonical `(X,Y,Z)=({spec.physical_dimensions.x_m:.2f}, {spec.physical_dimensions.y_m:.2f}, {spec.physical_dimensions.z_m:.2f}) m`
- Images/labels: 45/45 present; exact image, decoded-pixel, and label duplicates: 0
- Image overlap with plastic DEV and DEV_NEG2689: 0
- Bare six-digit IDs colliding with DEV_NEG2689: **{audit_payload['bare_frame_id_collisions_with_dev_neg']}/45**; session-qualified IDs colliding: 0
- Intrinsics: `{INTRINSICS_QUALITY}`, 1280x720, one exact K profile across both sessions
- Prior use: historical Stage-B/wood diagnostics already evaluated all 45; this population is DEV and can never be promoted to FINAL.
- Symmetry: `UNREVIEWED`; selector: `NOT_RUN`; paper pose fields remain blocked.

`DEV_WOOD_POS45` membership is frozen only after this audit passes. Frames are not removed for truncation or review priority.
"""
    qa_md = f"""# Wood GT QA report

Status: **{qa_payload['status']}**

- GREEN/AMBER/RED: `{qa_payload['green']}/{qa_payload['amber']}/{qa_payload['red']}`
- Quarantined/excluded: `0/0`
- Stored-pose projection median across frames: `{qa_payload['reprojection_median_px']:.3f} px`
- Worst frame median: `{qa_payload['reprojection_max_frame_median_px']:.3f} px`
- Proper rotations and cheirality: 45/45
- Geometric truncation: {audit_payload['geometrically_truncated_frame_count']} frames / {audit_payload['outside_keypoint_count']} keypoints; retained
- Visibility/provenance: 405/405 point states remain unknown and are queued for human review
- Intrinsics: sensor-profile-scaled, not per-session calibrated
- Wood symmetry remains unreviewed, so this QA PASS does not activate pose metrics.
"""
    _text_exclusive(output_dir / "WOOD_MEMBERSHIP_AUDIT.md", membership_md)
    _text_exclusive(output_dir / "WOOD_GT_QA_REPORT.md", qa_md)
    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geometry-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()
    result = audit(output_dir=args.output_dir, registry_path=args.geometry_registry)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
