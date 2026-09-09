"""Cuboid side-outline targets under camera_dynamic_0123_v4.

These are eight *structural cuboid support lines*, including amodal boundaries.
Neither valid annotations nor camera-facing geometry establish physical edge
visibility: pallet gaps, cargo, and other occluders are not labelled per edge.
Incoming cached grids use the previous continuous 0..50 convention; they are
converted here to the VGG feature-pixel centers 0..49 before constructing lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ALL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
SIDE_ROLE_IDS = (1, 3, 5, 7, 8, 9, 10, 11)
SIDE_EDGES = tuple(ALL_EDGES[i] for i in SIDE_ROLE_IDS)
EDGES = SIDE_EDGES
ROLE_NAMES = ("right front vertical", "left front vertical",
              "right rear vertical", "left rear vertical",
              "left top depth", "right top depth",
              "right bottom depth", "left bottom depth")
# Outward winding, with negative signed area for a camera-facing face in the
# image's x-right/y-down coordinates. Verified against world geometry in audit.
SIDE_FACES = {"left": (0, 4, 7, 3), "right": (1, 2, 6, 5)}
SIDE_FACE_ROLE_INDICES = {"left": (1, 3, 4, 7), "right": (0, 2, 5, 6)}


def _valid_points(record):
    points = np.asarray(record["gt_points"], dtype=np.float64)[:8]
    valid = np.asarray(record.get("gt_valid", [True] * 8), dtype=bool)[:8]
    valid &= np.isfinite(points).all(-1)
    valid &= ~(points == -1).all(-1)
    return points, valid


def _segment_intersects(p0, p1, width, height):
    """Liang-Barsky intersection with the original, unpadded image rectangle."""
    delta = p1 - p0
    t_lo, t_hi = 0.0, 1.0
    for p, q in ((-delta[0], p0[0]), (delta[0], width - p0[0]),
                 (-delta[1], p0[1]), (delta[1], height - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            ratio = q / p
            if p < 0:
                t_lo = max(t_lo, ratio)
            else:
                t_hi = min(t_hi, ratio)
    return t_lo <= t_hi


def make_targets(records, grids, grid_size=50, center=24.5, min_length_px=2.0, pad=100):
    """Return numpy theta_deg, centered rho, support arrays, each [B, 8].

    ``grids`` is [B, >=8, 2] in the inherited continuous convention, namely
    ``(original_xy + pad) * grid_size / padded_image_size``. OpenCV resizing
    samples pixel centers; three stride-2 VGG pools have stride 8 and first
    receptive-field center 3.5 in the 400px input. The actual feature coordinate
    is therefore ``(original_xy + pad + .5)*grid_size/padded_image_size - .5``.
    Apply the half-pixel correction here; callers must not apply it twice.
    The undirected normal is canonicalized to theta in [0,180), flipping rho
    together with the normal. Unsupported entries are finite zero placeholders.
    Support means annotated, length >= min_length_px, intersects original image;
    it does not mean physically visible. No annotation is fabricated or moved.
    """
    grids = np.asarray(grids, dtype=np.float64)
    if grids.ndim != 3 or grids.shape[0] != len(records) or grids.shape[1] < 8 or grids.shape[2] != 2:
        raise ValueError("grids must have shape [len(records), >=8, 2]")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    shape = (len(records), len(SIDE_EDGES))
    theta, rho = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    support = np.zeros(shape, bool)
    for i, record in enumerate(records):
        points, valid = _valid_points(record)
        correction = .5 * grid_size / np.array(
            [record["width"] + 2 * pad, record["height"] + 2 * pad]) - .5
        for j, (a, b) in enumerate(SIDE_EDGES):
            p0, p1 = grids[i, [a, b]] + correction
            if not (valid[a] and valid[b] and np.isfinite([p0, p1]).all()):
                continue
            delta = p1 - p0
            length = np.linalg.norm(delta)
            if length < 1e-9:
                continue
            normal = np.array([-delta[1], delta[0]]) / length
            angle = np.degrees(np.arctan2(normal[1], normal[0]))
            offset = np.dot(normal, (p0 + p1) / 2 - center)
            if angle < 0 or angle >= 180:
                offset = -offset
            theta[i, j], rho[i, j] = angle % 180, offset
            support[i, j] = (
                np.linalg.norm(points[a] - points[b]) >= min_length_px
                and _segment_intersects(points[a], points[b], record["width"], record["height"])
            )
    return theta, rho, support


def side_face_geometry(record, min_area_px2=1.0):
    """Return signed projected area and facing flags, never physical visibility."""
    points, valid = _valid_points(record)
    result = {}
    for side, indices in SIDE_FACES.items():
        q = points[list(indices)]
        good = bool(valid[list(indices)].all())
        area = float(0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                - q[:, 1] * np.roll(q[:, 0], -1))) if good else None
        result[side] = {"annotated": good, "signed_area_px2": area,
                        "camera_facing": good and area < -min_area_px2}
    return result


def facing_side_support(records, min_area_px2=1.0):
    """[B,8] union of boundaries on a geometrically camera-facing side face.

    Intersect with ``make_targets(...)[2]`` when evaluating. A face requires all
    four valid corners; unknown or almost edge-on faces are conservatively out.
    This mask uses GT only for evaluation stratification, never for prediction.
    """
    mask = np.zeros((len(records), len(SIDE_EDGES)), bool)
    for i, record in enumerate(records):
        for side, data in side_face_geometry(record, min_area_px2).items():
            if data["camera_facing"]:
                mask[i, list(SIDE_FACE_ROLE_INDICES[side])] = True
    return mask


def line_pixels(theta_deg, rho, width, height, pad=100, grid_size=50, center=24.5):
    """Centered Hough parameters (...,) -> original-image lines (..., 2, 2)."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    rho = np.asarray(rho, dtype=np.float64)
    normal = np.stack([np.cos(theta), np.sin(theta)], -1)
    base = center + rho[..., None] * normal
    direction = np.stack([-normal[..., 1], normal[..., 0]], -1)
    points = np.stack([base - 2 * grid_size * direction,
                       base + 2 * grid_size * direction], -2)
    return (points + .5) * np.array([(width + 2 * pad) / grid_size,
                                     (height + 2 * pad) / grid_size]) - .5 - pad


def pixel_errors(lines, gt_points):
    """Original-pixel undirected angle and mean endpoint-to-infinite-line error.

    ``lines`` is [8,2,2], ``gt_points`` [>=8,2]; return two [8] arrays. Caller
    must apply support mask; unsupported geometry does not become valid here.
    """
    lines = np.asarray(lines, dtype=np.float64)
    truth = np.asarray(gt_points, dtype=np.float64)[np.asarray(SIDE_EDGES)]
    predicted_direction = lines[:, 1] - lines[:, 0]
    predicted_direction /= np.linalg.norm(predicted_direction, axis=-1, keepdims=True).clip(1e-12)
    truth_direction = truth[:, 1] - truth[:, 0]
    truth_direction /= np.linalg.norm(truth_direction, axis=-1, keepdims=True).clip(1e-12)
    angles = np.rad2deg(np.arccos(np.abs((predicted_direction * truth_direction).sum(-1)).clip(0, 1)))
    normal = np.stack([-predicted_direction[:, 1], predicted_direction[:, 0]], -1)
    distance = np.abs(((truth - lines[:, :1]) * normal[:, None]).sum(-1)).mean(-1)
    return angles, distance


def audit(manifest_path, output_dir, pad=100):
    """Audit target math, per-population support/facing, and draw four examples."""
    import cv2

    manifest = json.loads(Path(manifest_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"manifest": str(Path(manifest_path).resolve()),
              "side_role_ids": SIDE_ROLE_IDS, "side_edges": SIDE_EDGES,
              "role_names": ROLE_NAMES,
              "support_semantics": "Valid annotated cuboid segment >=2px intersecting original image, not physical visibility",
              "facing_semantics": "Projected outward face winding; geometry-only, ignores cargo and occluders",
              "real_label_limit": "Existing manual/projected cuboid corners; physical edge and per-corner visibility/provenance unavailable",
              "feature_coordinate_transform": "(original_xy + pad + 0.5) * 50 / padded_image_size - 0.5",
              "feature_coordinate_inverse": "(feature_xy + 0.5) * padded_image_size / 50 - 0.5 - pad",
              "coordinate_alignment": "OpenCV resize half-pixel centers and VGG stride8 first receptive-field center3.5; cached inherited grids corrected inside make_targets",
              "populations": {}}
    maximum_distance, maximum_angle = 0.0, 0.0
    maximum_center_error = 0.0
    for population, records in manifest["populations"].items():
        grids = [((np.asarray(r["gt_points"]) + pad)
                  * [50 / (r["width"] + 2 * pad), 50 / (r["height"] + 2 * pad)]) for r in records]
        theta, rho, supported = make_targets(records, grids, pad=pad)
        facing = facing_side_support(records) & supported
        counts = {"left": 0, "right": 0, "neither": 0, "both": 0}
        comparisons, mismatches = 0, 0
        lengths = []
        for i, record in enumerate(records):
            geometry = side_face_geometry(record)
            left, right = geometry["left"]["camera_facing"], geometry["right"]["camera_facing"]
            counts["both" if left and right else "left" if left else "right" if right else "neither"] += 1
            lines = line_pixels(theta[i], rho[i], record["width"], record["height"], pad)
            origin_line = line_pixels(0., 0., record["width"], record["height"], pad)
            image_center = np.array([(record["width"] - 1) / 2,
                                     (record["height"] - 1) / 2])
            maximum_center_error = max(maximum_center_error,
                                       float(np.abs(origin_line.mean(0) - image_center).max()))
            angles, distance = pixel_errors(lines, record["gt_points"])
            if supported[i].any():
                maximum_distance = max(maximum_distance, float(distance[supported[i]].max()))
                maximum_angle = max(maximum_angle, float(angles[supported[i]].max()))
            points = np.asarray(record["gt_points"])[np.asarray(SIDE_EDGES)]
            lengths.extend(np.linalg.norm(points[:, 1] - points[:, 0], axis=-1)[supported[i]].tolist())
            annotation = json.loads(Path(record["annotation"]).read_text())
            obj, cam = annotation["objects"][0], annotation["camera_data"]
            if "cuboid" in obj and "location_worldframe" in cam:
                world = np.asarray(obj["cuboid"], float)[:8]
                camera = np.asarray(cam["location_worldframe"], float)
                center_world = world.mean(0)
                for side, indices in SIDE_FACES.items():
                    q = world[list(indices)]
                    normal = np.cross(q[1] - q[0], q[3] - q[0])
                    if np.dot(normal, q.mean(0) - center_world) < 0:
                        normal = -normal
                    dot = np.dot(normal, camera - q.mean(0))
                    area = geometry[side]["signed_area_px2"]
                    if area is not None and abs(area) > 1:
                        comparisons += 1
                        mismatches += int((dot > 0) != (area < 0))
        report["populations"][population] = {
            "frames": len(records), "supported_lines": int(supported.sum()),
            "supported_per_role": supported.sum(0).tolist(), "camera_facing_frames": counts,
            "camera_facing_supported_lines": int(facing.sum()),
            "facing_comparisons_to_world_geometry": comparisons,
            "facing_mismatches_to_world_geometry": mismatches,
            "side_segment_length_px_p10_p50_p90": np.percentile(lengths, [10, 50, 90]).tolist(),
        }
    report["roundtrip_max_angle_deg"] = maximum_angle
    report["roundtrip_max_distance_px"] = maximum_distance
    report["feature_origin_to_image_center_max_error_px"] = maximum_center_error
    if maximum_distance > 1e-3 or maximum_angle > 1e-3:
        raise AssertionError("Hough target/pixel roundtrip failed")
    if maximum_center_error > 1e-9:
        raise AssertionError("Feature origin must map to exact original-image pixel center")
    if any(r["facing_mismatches_to_world_geometry"] for r in report["populations"].values()):
        raise AssertionError("Projected face winding disagrees with world geometry")
    palette = [(240, 170, 0), (0, 170, 250), (230, 100, 0), (0, 100, 230),
               (0, 220, 250), (250, 220, 0), (250, 100, 140), (130, 100, 250)]
    panels = []
    for population in ("synth_train", "real_dev"):
        records = manifest["populations"][population]
        selected = [next(r for r in records if side_face_geometry(r)["left"]["camera_facing"]),
                    next(r for r in records if side_face_geometry(r)["right"]["camera_facing"])]
        for record in selected:
            original = cv2.imread(record["image"])
            drawn = original.copy()
            points, valid = _valid_points(record)
            for j, (a, b) in enumerate(SIDE_EDGES):
                if valid[a] and valid[b]:
                    p0, p1 = tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int))
                    cv2.line(drawn, p0, p1, palette[j], 2, cv2.LINE_AA)
                    mid = tuple(np.rint((points[a] + points[b]) / 2).astype(int))
                    cv2.putText(drawn, str(SIDE_ROLE_IDS[j]), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(drawn, str(SIDE_ROLE_IDS[j]), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, palette[j], 1, cv2.LINE_AA)
            for j, point in enumerate(points):
                if valid[j]:
                    p = tuple(np.rint(point).astype(int))
                    cv2.circle(drawn, p, 3, (255, 255, 255), -1)
                    cv2.putText(drawn, f"v{j}", (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(drawn, f"v{j}", (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.concatenate([original, drawn], 1)
            canvas = np.full((panel.shape[0] + 82, panel.shape[1], 3), 245, np.uint8)
            canvas[82:] = panel
            face = next(s for s, g in side_face_geometry(record).items() if g["camera_facing"])
            cv2.putText(canvas, record["id"], (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"ALL 8 structural side lines | camera-facing side: {face} (geometry only)", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Includes hidden/cuboid boundaries. Labels: original role ID; v0..v7 corners.", (12, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
            path = output_dir / f"{record['id']}.png"
            cv2.imwrite(str(path), canvas)
            panels.append(str(path.resolve()))
    report["panels"] = panels
    (output_dir / "TARGET_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/pallet/results/hough_attention_transfer_v1/manifest.json")
    parser.add_argument("--output-dir", default="data/pallet/results/deep_hough_side_v1/target_audit")
    args = parser.parse_args()
    audit(args.manifest, args.output_dir)
