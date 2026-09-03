"""GATE 0B — 전수 sensor validity 와 pallet-local depth 구조를 측정한다.

    python3 scripts/self_training_yolo/depth_corrected/audit_pallet_local_depth.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0b

출력  FULL_DEPTH_VALIDITY.csv · FULL_DEPTH_VALIDITY_SUMMARY.json
      PALLET_LOCAL_DEPTH_PER_FRAME.csv · PALLET_LOCAL_DEPTH_SUMMARY.json

Gate 0 는 bbox·convex hull 전체에서만 depth 를 봤고 배경이 섞였다.  그것은
**국소 구조가 없다는 증거가 아니다**.  여기서는 예측된 cuboid 면의 내부를
따로 떼어 본다.

면 topology·shrink 비율·keypoint 샘플 반경·validity 임계값은 전부
`GATE0B_METHOD_LOCK.json` 에서 읽는다.  여기서 정하지 않는다.
평가 GT 를 읽지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def shrink_polygon(points: np.ndarray, ratio: float) -> np.ndarray:
    centroid = points.mean(axis=0)
    return centroid + (points - centroid) * (1.0 - ratio)


def mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def plane_residuals(points_3d: np.ndarray) -> tuple[float, float]:
    """PCA 최소자승 평면.  RANSAC 을 쓰지 않는다."""

    centred = points_3d - points_3d.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    distance = np.abs(centred @ normal)
    return float(np.median(distance)), float(np.percentile(distance, 90))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    lock = json.loads((out_dir / "GATE0B_METHOD_LOCK.json").read_text())
    conventions = lock["reused_repository_conventions"]
    faces = {name: tuple(indices) for name, indices in conventions["faces"].items()}
    shrink = float(lock["face_interior_shrink_ratio"])
    radius = int(conventions["keypoint_sampling_radius"])
    kp_conf_min = float(conventions["keypoint_validity_threshold"])
    scale = float(conventions["depth_scale_m_per_unit"])
    min_plane_points = 3          # 평면이 정의되려면 최소 3점.  튜닝값 아님.
    print(f"lock: shrink {shrink}  radius {radius}  kp_conf {kp_conf_min}  scale {scale}")

    roi = json.loads((out_dir / "R0_FULL_ADAPT_ROI_CACHE.json").read_text())["frames"]
    camera_cache: dict[str, np.ndarray] = {}

    sensor_rows, local_rows = [], []
    previous_face_median: dict[str, tuple[str, float]] = {}
    for order, (relative_path, entry) in enumerate(sorted(roi.items()), start=1):
        image_path = REPO_ROOT / relative_path
        depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.png"
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        dtype_max = np.iinfo(raw.dtype).max
        loose = raw > 0
        strict = loose & (raw != dtype_max)
        finite = raw[strict].astype(np.float64)
        session = entry["capture_session"]

        sensor_rows.append({
            "image_path": relative_path, "source_recording": session,
            "lighting": entry["lighting"],
            "loose_valid_fraction": float(loose.mean()),
            "strict_valid_fraction": float(strict.mean()),
            "zero_fraction": float((raw == 0).mean()),
            "dtype_max_fraction": float((raw == dtype_max).mean()),
            "raw_p95": float(np.percentile(finite, 95)) if finite.size else None,
            "raw_p99": float(np.percentile(finite, 99)) if finite.size else None,
            "raw_max": float(finite.max()) if finite.size else None,
            "metric_p99_m": float(np.percentile(finite, 99) * scale) if finite.size else None,
        })

        top = entry.get("top1")
        if top is None:
            continue
        if session not in camera_cache:
            camera_cache[session] = np.loadtxt(image_path.parent.parent / "cam_K.txt")
        camera = camera_cache[session]
        metric = raw.astype(np.float32) * scale
        metric[~strict] = 0.0
        height, width = metric.shape

        keypoints = np.asarray(top["keypoints_xy"], np.float32)
        confidences = np.nan_to_num(np.asarray(top["keypoints_conf"], float), nan=0.0)

        # ROI-A bbox, ROI-B hull
        x1, y1, x2, y2 = [int(round(v)) for v in top["box_xyxy"]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)
        bbox_values = metric[y1:y2, x1:x2][metric[y1:y2, x1:x2] > 0]
        bbox_spread = (float(np.percentile(bbox_values, 90) - np.percentile(bbox_values, 10))
                       if bbox_values.size else None)

        record = {
            "image_path": relative_path, "source_recording": session,
            "lighting": entry["lighting"],
            "bbox_pixels": int(max(x2 - x1, 0) * max(y2 - y1, 0)),
            "bbox_depth_spread_cm": bbox_spread * 100 if bbox_spread is not None else None,
            "n_faces_evaluated": 0, "best_face": None,
            "face_valid_points": None, "face_strict_fraction": None,
            "face_median_m": None, "face_iqr_cm": None, "face_mad_cm": None,
            "face_p10_p90_cm": None,
            "plane_residual_median_mm": None, "plane_residual_p90_mm": None,
            "ring_median_m": None, "ring_minus_face_cm": None, "ring_mad_cm": None,
            "kp_valid_samples": 0, "kp_local_mad_cm": None,
        }

        best = None
        for name, indices in faces.items():
            corners = keypoints[list(indices)]
            if (confidences[list(indices)] < kp_conf_min).any():
                continue
            if not np.isfinite(corners).all():
                continue
            # 다각형 순서를 볼록껍질로 정리한 뒤 안쪽으로 줄인다
            hull = cv2.convexHull(corners.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) < 3:
                continue
            inner = shrink_polygon(hull, shrink)
            mask = np.zeros(metric.shape, np.uint8)
            cv2.fillPoly(mask, [inner.astype(np.int32)], 1)
            interior = mask.astype(bool)
            if interior.sum() < min_plane_points:
                continue
            record["n_faces_evaluated"] += 1

            values = metric[interior & (metric > 0)]
            if values.size < min_plane_points:
                continue
            ys, xs = np.nonzero(interior & (metric > 0))
            z = metric[ys, xs].astype(np.float64)
            X = (xs - camera[0, 2]) * z / camera[0, 0]
            Y = (ys - camera[1, 2]) * z / camera[1, 1]
            residual_median, residual_p90 = plane_residuals(np.stack([X, Y, z], axis=1))

            outer = cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0
            ring = outer & ~interior
            ring_values = metric[ring & (metric > 0)]

            candidate = {
                "face": name,
                "valid_points": int(values.size),
                "strict_fraction": float(values.size / max(int(interior.sum()), 1)),
                "median_m": float(np.median(values)),
                "iqr_cm": float((np.percentile(values, 75) - np.percentile(values, 25)) * 100),
                "mad_cm": float(mad(values) * 100),
                "p10_p90_cm": float((np.percentile(values, 90) - np.percentile(values, 10)) * 100),
                "plane_median_mm": residual_median * 1000,
                "plane_p90_mm": residual_p90 * 1000,
                "ring_median_m": float(np.median(ring_values)) if ring_values.size else None,
                "ring_mad_cm": float(mad(ring_values) * 100) if ring_values.size else None,
            }
            # "가장 조밀한 면" 을 대표로 삼는다.  성능 기준이 아니다.
            if best is None or candidate["valid_points"] > best["valid_points"]:
                best = candidate

        if best is not None:
            record.update({
                "best_face": best["face"],
                "face_valid_points": best["valid_points"],
                "face_strict_fraction": best["strict_fraction"],
                "face_median_m": best["median_m"],
                "face_iqr_cm": best["iqr_cm"],
                "face_mad_cm": best["mad_cm"],
                "face_p10_p90_cm": best["p10_p90_cm"],
                "plane_residual_median_mm": best["plane_median_mm"],
                "plane_residual_p90_mm": best["plane_p90_mm"],
                "ring_median_m": best["ring_median_m"],
                "ring_minus_face_cm": ((best["ring_median_m"] - best["median_m"]) * 100
                                       if best["ring_median_m"] is not None else None),
                "ring_mad_cm": best["ring_mad_cm"],
            })
            key = f"{session}"
            stem = int(image_path.stem) if image_path.stem.isdigit() else None
            if key in previous_face_median and stem is not None:
                prev_stem, prev_value = previous_face_median[key]
                record["adjacent_face_median_delta_cm"] = abs(
                    best["median_m"] - prev_value) * 100
            previous_face_median[key] = (image_path.stem, best["median_m"])

        # keypoint 이웃 (기존 sample_depth 관례)
        samples = []
        for index in range(8):
            if confidences[index] < kp_conf_min:
                continue
            cx, cy = int(keypoints[index][0]), int(keypoints[index][1])
            patch = metric[max(cy - radius, 0):cy + radius + 1,
                           max(cx - radius, 0):cx + radius + 1]
            values = patch[patch > 0.05]
            if values.size:
                samples.append(float(np.median(values)))
        record["kp_valid_samples"] = len(samples)
        if len(samples) >= 2:
            record["kp_local_mad_cm"] = float(mad(np.array(samples)) * 100)
        local_rows.append(record)

        if order % 1000 == 0:
            print(f"  {order}/{len(roi)}", flush=True)

    # ── 저장
    with (out_dir / "FULL_DEPTH_VALIDITY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensor_rows[0]))
        writer.writeheader()
        writer.writerows(sensor_rows)
    fields = sorted({k for r in local_rows for k in r})
    with (out_dir / "PALLET_LOCAL_DEPTH_PER_FRAME.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in local_rows:
            writer.writerow({k: row.get(k) for k in fields})

    def group(rows, key, value):
        return [r for r in rows if r[key] == value]

    def sensor_summary(rows):
        return {
            "n": len(rows),
            "strict_valid_median": float(np.median([r["strict_valid_fraction"] for r in rows])),
            "loose_valid_median": float(np.median([r["loose_valid_fraction"] for r in rows])),
            "zero_median": float(np.median([r["zero_fraction"] for r in rows])),
            "dtype_max_median": float(np.median([r["dtype_max_fraction"] for r in rows])),
            "raw_p99_median": float(np.median([r["raw_p99"] for r in rows if r["raw_p99"]])),
            "metric_p99_median_m": float(np.median(
                [r["metric_p99_m"] for r in rows if r["metric_p99_m"]])),
        }

    def local_summary(rows):
        with_face = [r for r in rows if r["best_face"] is not None]
        block = {"detected_frames": len(rows), "frames_with_face_interior": len(with_face),
                 "face_rate": len(with_face) / len(rows) if rows else 0.0}
        if with_face:
            pick = lambda k: np.array([r[k] for r in with_face if r[k] is not None], float)
            block |= {
                "face_valid_points_median": float(np.median(pick("face_valid_points"))),
                "face_mad_cm_median": float(np.median(pick("face_mad_cm"))),
                "face_p10_p90_cm_median": float(np.median(pick("face_p10_p90_cm"))),
                "plane_residual_median_mm_median": float(np.median(pick("plane_residual_median_mm"))),
                "plane_residual_p90_mm_median": float(np.median(pick("plane_residual_p90_mm"))),
                "ring_minus_face_cm_median": float(np.median(pick("ring_minus_face_cm"))),
                "bbox_depth_spread_cm_median": float(np.median(pick("bbox_depth_spread_cm"))),
            }
            deltas = [r["adjacent_face_median_delta_cm"] for r in with_face
                      if r.get("adjacent_face_median_delta_cm") is not None]
            if deltas:
                block["adjacent_delta_cm_median"] = float(np.median(deltas))
                block["adjacent_delta_cm_p90"] = float(np.percentile(deltas, 90))
        return block

    sensor = {"ALL": sensor_summary(sensor_rows)}
    local = {"ALL": local_summary(local_rows)}
    for lighting in ("day", "night"):
        sensor[lighting.upper()] = sensor_summary(group(sensor_rows, "lighting", lighting))
        local[lighting.upper()] = local_summary(group(local_rows, "lighting", lighting))
    for name in sorted({r["source_recording"] for r in sensor_rows}):
        sensor[name] = sensor_summary(group(sensor_rows, "source_recording", name))
        local[name] = local_summary(group(local_rows, "source_recording", name))

    (out_dir / "FULL_DEPTH_VALIDITY_SUMMARY.json").write_text(json.dumps({
        "schema_version": "full_depth_validity_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "population": len(sensor_rows),
        "layers": {"loose": "depth > 0", "strict": "depth > 0 and depth != dtype max"},
        "no_new_far_clipping_threshold": True,
        "summary": sensor,
    }, indent=2) + "\n")
    (out_dir / "PALLET_LOCAL_DEPTH_SUMMARY.json").write_text(json.dumps({
        "schema_version": "pallet_local_depth_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "face_topology_source": conventions["face_topology_source"],
        "shrink_ratio": shrink,
        "plane_fit": "PCA least squares, no RANSAC, no inlier threshold",
        "min_points_for_a_plane": min_plane_points,
        "why_that_minimum": "a plane needs three points to be defined; this is geometry, not a tuned threshold",
        "representative_face_rule": "the face interior holding the most valid depth points; not chosen by any quality score",
        "summary": local,
    }, indent=2) + "\n")

    print(f"\n{'group':20}{'det':>7}{'face':>7}{'rate':>7}{'pts':>7}"
          f"{'MAD cm':>9}{'plane mm':>10}{'ring-face cm':>14}{'bbox spread cm':>16}")
    print("-" * 97)
    for name in ["ALL", "DAY", "NIGHT"] + sorted({r["source_recording"] for r in local_rows}):
        block = local.get(name)
        if not block or not block.get("frames_with_face_interior"):
            print(f"{name:20}{block['detected_frames']:7d}{0:7d}")
            continue
        print(f"{name:20}{block['detected_frames']:7d}{block['frames_with_face_interior']:7d}"
              f"{block['face_rate']:7.3f}{block['face_valid_points_median']:7.0f}"
              f"{block['face_mad_cm_median']:9.1f}{block['plane_residual_median_mm_median']:10.1f}"
              f"{block['ring_minus_face_cm_median']:14.1f}"
              f"{block['bbox_depth_spread_cm_median']:16.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
