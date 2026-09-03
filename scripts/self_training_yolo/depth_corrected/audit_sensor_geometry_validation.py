"""SENSOR VALIDATION — 저장된 depth 가 수동 어노 RGB 기하와 양립하는지 검증한다.

    python3 scripts/self_training_yolo/depth_corrected/audit_sensor_geometry_validation.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/sensor_validation_v1

출력  SENSOR_VALIDATION_POPULATION.csv · SENSOR_VALIDATION_PER_FRAME.csv
      SENSOR_VALIDATION_SUMMARY.json · contact_sheets/

Gate 0B 가 막힌 것은 depth 자체가 아니라 acquisition 기록의 부재였다.  수동
어노 키포인트 · 알려진 치수 · depth · cam_K 가 한 프레임에 다 있는 곳에서는
그 공백을 문서 대신 **실측**으로 시험할 수 있다.

모든 임계값·비율·면 정의는 `SENSOR_VALIDATION_LOCK.json` 에서 읽는다.
scale 은 고정값만 평가한다 — 적합·스윕·최적값 탐색 금지.
teacher 예측을 쓰지 않는다.  이건 성능 측정이 아니다.
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
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
RAW = REPO_ROOT / "data/pallet/raw_data"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
NIGHT_TOKENS = ("night",)


def sha_index_of_raw() -> dict[str, Path]:
    """raw 촬영본의 이미지 SHA -> 경로.  이름이 아니라 내용으로 되짚기 위해."""

    import hashlib
    index: dict[str, Path] = {}
    for root in (RAW / "outside", RAW / "night"):
        if not root.is_dir():
            continue
        for session in sorted(root.iterdir()):
            rgb, depth = session / "rgb", session / "depth"
            if not (rgb.is_dir() and depth.is_dir()):
                continue
            for image in rgb.iterdir():
                if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                index[digest] = image
    return index


def registry_dimensions() -> dict[str, dict]:
    return {e["object_type"]: e["physical_dimensions_m"]
            for e in json.loads(REGISTRY.read_text())["objects"]}


def shrink(points: np.ndarray, ratio: float) -> np.ndarray:
    centroid = points.mean(axis=0)
    return centroid + (points - centroid) * (1.0 - ratio)


def cuboid(across, height, along):
    ha, hh, hb = across / 2, height / 2, along / 2
    return np.array([[-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
                     [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb]],
                    dtype=np.float64)


def solve_reference(model, points, camera, usable):
    """수동 키포인트 + 알려진 치수로 기준 자세를 복원.  모델 예측이 아니다."""

    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    residual = float(np.linalg.norm(
        projected.reshape(-1, 2)[usable] - points[usable], axis=1).mean())
    return rotation, tvec.reshape(-1), residual


def ray_surface_z(pixels, camera, rotation, translation, extents):
    """각 픽셀의 시선이 기준 cuboid 와 처음 만나는 z (미터).  슬랩 교차."""

    directions = np.stack([(pixels[:, 0] - camera[0, 2]) / camera[0, 0],
                           (pixels[:, 1] - camera[1, 2]) / camera[1, 1],
                           np.ones(len(pixels))], axis=1)
    local_dir = directions @ rotation            # world->local 은 R^T, 행벡터라 @R
    local_origin = -translation @ rotation
    half = np.array(extents) / 2.0
    t_near = np.full(len(pixels), -np.inf)
    t_far = np.full(len(pixels), np.inf)
    for axis in range(3):
        d = local_dir[:, axis]
        o = local_origin[axis]
        parallel = np.abs(d) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half[axis] - o) / d
            t2 = (+half[axis] - o) / d
        lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
        # 시선이 이 축과 평행하면 슬랩은 원점이 안에 있느냐로만 갈린다 (o 는 스칼라)
        inside_slab = abs(o) <= half[axis]
        lo[parallel] = -np.inf if inside_slab else np.inf
        hi[parallel] = np.inf if inside_slab else -np.inf
        t_near = np.maximum(t_near, lo)
        t_far = np.minimum(t_far, hi)
    hit = (t_near <= t_far) & (t_far > 0)
    t = np.where(t_near > 0, t_near, t_far)
    return np.where(hit, t * directions[:, 2], np.nan), hit


def point_to_surface(points_3d, rotation, translation, extents):
    """점에서 cuboid 표면까지 거리 (미터).  안/밖 모두 처리."""

    local = (points_3d - translation) @ rotation
    half = np.array(extents) / 2.0
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sheets = out_dir / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    lock = json.loads((out_dir / "SENSOR_VALIDATION_LOCK.json").read_text())
    measure = lock["measurements_fixed_before_results"]
    ratio = float(measure["A_depth_at_manual_face"]["shrink_ratio"])
    scale = float(measure["G_scale"]["evaluate_only"])
    faces = {"near": (0, 1, 2, 3), "far": (4, 5, 6, 7), "top": (0, 1, 4, 5),
             "bottom": (2, 3, 6, 7), "left": (0, 3, 4, 7), "right": (1, 2, 5, 6)}
    print(f"lock: shrink {ratio}  scale {scale} (fixed, never fitted)")

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation"))
    from eval_workspace import load_frames, evaluation_population_views

    frames = load_frames(WORKSPACE)
    eligible = {r["frame_id"] for r in
                evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]}
    dimensions = registry_dimensions()
    print("hashing raw recordings to resolve provenance by content ...", flush=True)
    raw_index = sha_index_of_raw()
    print(f"  raw images indexed {len(raw_index)}")

    population, per_frame = [], []
    for frame in frames:
        if frame.get("is_annotated") != "true" or frame.get("is_positive") != "true":
            continue
        digest = frame.get("source_image_sha256") or ""
        raw_image = raw_index.get(digest)
        if raw_image is None:
            continue
        depth_path = raw_image.parent.parent / "depth" / f"{raw_image.stem}.png"
        cam_k = raw_image.parent.parent / "cam_K.txt"
        if not (depth_path.exists() and cam_k.exists()):
            continue
        annotation = WORKSPACE / frame["annotation_path"]
        if not annotation.exists():
            continue
        payload = json.loads(annotation.read_text())
        obj = payload["objects"][0]
        object_type = ("plastic_standard_110x130x11"
                       if frame.get("object_type") == "plastic"
                       else "wood_small_80x59x14")
        if object_type not in dimensions:
            continue
        session = raw_image.parent.parent.name
        population.append({
            "frame_id": frame["frame_id"], "source_recording": session,
            "rgb_path": str(raw_image.relative_to(REPO_ROOT)),
            "depth_path": str(depth_path.relative_to(REPO_ROOT)),
            "annotation_path": frame["annotation_path"],
            "object_type": object_type,
            "dimensions": json.dumps(dimensions[object_type]),
            "lighting": "night" if any(t in session for t in NIGHT_TOKENS) else "day",
            "current_evaluation_role": frame.get("usage_role", ""),
            "source_provenance": "resolved to raw recording by image sha256",
            "overlaps_existing_eval": frame["frame_id"] in eligible,
        })

    print(f"population {len(population)}")
    if not population:
        print("no frame satisfies all five requirements")
        return 1
    with (out_dir / "SENSOR_VALIDATION_POPULATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(population[0]))
        writer.writeheader()
        writer.writerows(population)

    camera_cache: dict[str, np.ndarray] = {}
    sheet_taken: dict[str, int] = {}
    for row in population:
        raw_image = REPO_ROOT / row["rgb_path"]
        session = row["source_recording"]
        if session not in camera_cache:
            camera_cache[session] = np.loadtxt(raw_image.parent.parent / "cam_K.txt")
        camera = camera_cache[session]
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        obj = payload["objects"][0]
        keypoints = np.array([p if p else [np.nan, np.nan]
                              for p in obj["projected_cuboid"]], np.float64)[:8]
        usable = np.isfinite(keypoints).all(axis=1)
        if usable.sum() < 6:
            continue

        dims = json.loads(row["dimensions"])
        long_m = max(dims["x"], dims["z"])
        short_m = min(dims["x"], dims["z"])
        height_m = dims["y"]
        best = None
        for across, along in ((long_m, short_m), (short_m, long_m)):
            model = cuboid(across, height_m, along)
            fit = solve_reference(model, keypoints, camera, usable)
            if fit and (best is None or fit[2] < best[0][2]):
                best = (fit, (across, height_m, along))
        if best is None:
            continue
        (rotation, translation, reproj), extents = best

        raw = cv2.imread(str(REPO_ROOT / row["depth_path"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(raw_image))
        if raw is None or rgb is None:
            continue
        metric = raw.astype(np.float32) * scale
        metric[raw == np.iinfo(raw.dtype).max] = 0.0
        height, width = metric.shape

        # 면 내부 마스크 (수동 키포인트 기반)
        face_mask = np.zeros(metric.shape, np.uint8)
        for indices in faces.values():
            corners = keypoints[list(indices)]
            if not np.isfinite(corners).all():
                continue
            hull = cv2.convexHull(corners.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) < 3:
                continue
            cv2.fillPoly(face_mask, [shrink(hull, ratio).astype(np.int32)], 1)
        interior = face_mask.astype(bool) & (metric > 0)
        ring = (cv2.dilate(face_mask, np.ones((25, 25), np.uint8)) > 0) \
            & ~face_mask.astype(bool) & (metric > 0)
        if interior.sum() < 3:
            continue

        record = {"frame_id": row["frame_id"], "source_recording": session,
                  "lighting": row["lighting"], "object_type": row["object_type"],
                  "reference_reproj_px": reproj,
                  "face_valid_points": int(interior.sum()),
                  "face_valid_fraction": float(interior.sum() / max(face_mask.sum(), 1))}

        def residuals(mask):
            ys, xs = np.nonzero(mask)
            z = metric[ys, xs].astype(np.float64)
            X = (xs - camera[0, 2]) * z / camera[0, 0]
            Y = (ys - camera[1, 2]) * z / camera[1, 1]
            surface = np.abs(point_to_surface(np.stack([X, Y, z], 1),
                                              rotation, translation, extents))
            expected, hit = ray_surface_z(np.stack([xs, ys], 1).astype(float),
                                          camera, rotation, translation, extents)
            depth_delta = np.abs(z - expected)[hit] if hit.any() else np.array([])
            return surface, depth_delta, z

        face_surface, face_delta, face_z = residuals(interior)
        record["face_median_m"] = float(np.median(face_z))
        record["face_mad_cm"] = float(np.median(np.abs(face_z - np.median(face_z))) * 100)
        record["face_p10_p90_cm"] = float(
            (np.percentile(face_z, 90) - np.percentile(face_z, 10)) * 100)
        record["surface_residual_median_cm"] = float(np.median(face_surface) * 100)
        record["surface_residual_p90_cm"] = float(np.percentile(face_surface, 90) * 100)
        if face_delta.size:
            record["depth_residual_median_cm"] = float(np.median(face_delta) * 100)
            record["depth_residual_p90_cm"] = float(np.percentile(face_delta, 90) * 100)
        if ring.sum() >= 3:
            ring_surface, ring_delta, _ = residuals(ring)
            record["ring_surface_residual_median_cm"] = float(np.median(ring_surface) * 100)
            record["face_better_than_ring"] = bool(
                np.median(face_surface) < np.median(ring_surface))

        # 경계 정렬: 수동 팔레트 윤곽 vs depth 불연속
        outline = cv2.convexHull(
            keypoints[usable].astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
        manual_edge = np.zeros(metric.shape, np.uint8)
        cv2.polylines(manual_edge, [outline.astype(np.int32)], True, 1, 1)
        valid = metric > 0
        gx = cv2.Sobel(np.where(valid, metric, 0), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(np.where(valid, metric, 0), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(gx, gy)
        finite = magnitude[valid]
        depth_edge = ((magnitude >= np.percentile(finite, 97)) & valid
                      if finite.size else np.zeros_like(valid))
        if depth_edge.any() and manual_edge.any():
            to_depth = cv2.distanceTransform((~depth_edge).astype(np.uint8), cv2.DIST_L2, 3)
            d = to_depth[manual_edge.astype(bool)]
            record["manual_edge_to_depth_edge_median_px"] = float(np.median(d))
            record["manual_edge_to_depth_edge_p90_px"] = float(np.percentile(d, 90))
            near = cv2.dilate(manual_edge, np.ones((21, 21), np.uint8)) > 0
            to_manual = cv2.distanceTransform((~manual_edge.astype(bool)).astype(np.uint8),
                                              cv2.DIST_L2, 3)
            reverse = to_manual[depth_edge & near]
            if reverse.size:
                record["depth_edge_to_manual_edge_median_px"] = float(np.median(reverse))
        per_frame.append(record)

        taken = sheet_taken.get(session, 0)
        if taken < 4:
            sheet_taken[session] = taken + 1
            expected_map = np.zeros_like(metric)
            ys, xs = np.nonzero(face_mask.astype(bool))
            ez, hit = ray_surface_z(np.stack([xs, ys], 1).astype(float),
                                    camera, rotation, translation, extents)
            expected_map[ys[hit], xs[hit]] = ez[hit]
            def colour(m):
                v = m > 0
                if not v.any():
                    return np.zeros((*m.shape, 3), np.uint8)
                lo, hi = np.percentile(m[v], [2, 98])
                out = cv2.applyColorMap(
                    (np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_TURBO)
                out[~v] = 0
                return out
            annotated = rgb.copy()
            cv2.polylines(annotated, [outline.astype(np.int32)], True, (80, 220, 255), 2)
            panel = np.hstack([cv2.resize(p, (240, 180)) for p in
                               (rgb, annotated, colour(metric),
                                colour(np.where(face_mask.astype(bool), metric, 0)),
                                colour(expected_map))])
            path = sheets / f"{session}.jpg"
            if path.exists():
                panel = np.vstack([cv2.imread(str(path)), panel])
            cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])

    fields = sorted({k for r in per_frame for k in r})
    with (out_dir / "SENSOR_VALIDATION_PER_FRAME.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_frame:
            writer.writerow({k: row.get(k) for k in fields})

    def summarize(rows):
        if not rows:
            return {"n": 0}
        pick = lambda k: np.array([r[k] for r in rows if r.get(k) is not None], float)
        block = {"n": len(rows)}
        for key in ("reference_reproj_px", "face_valid_fraction", "face_mad_cm",
                    "surface_residual_median_cm", "surface_residual_p90_cm",
                    "depth_residual_median_cm", "depth_residual_p90_cm",
                    "ring_surface_residual_median_cm",
                    "manual_edge_to_depth_edge_median_px",
                    "manual_edge_to_depth_edge_p90_px",
                    "depth_edge_to_manual_edge_median_px"):
            values = pick(key)
            if values.size:
                block[f"{key}_median"] = float(np.median(values))
        flags = [r["face_better_than_ring"] for r in rows if "face_better_than_ring" in r]
        if flags:
            block["fraction_face_better_than_ring"] = float(np.mean(flags))
            block["n_with_ring"] = len(flags)
        return block

    summary = {"ALL": summarize(per_frame)}
    for lighting in ("day", "night"):
        summary[lighting.upper()] = summarize(
            [r for r in per_frame if r["lighting"] == lighting])
    for session in sorted({r["source_recording"] for r in per_frame}):
        summary[session] = summarize(
            [r for r in per_frame if r["source_recording"] == session])

    (out_dir / "SENSOR_VALIDATION_SUMMARY.json").write_text(json.dumps({
        "schema_version": "sensor_validation_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock": "SENSOR_VALIDATION_LOCK.json",
        "teacher_predictions_used": False,
        "scale_fitted": False,
        "scale_evaluated": scale,
        "reference_geometry": "manual 2D keypoints plus registered dimensions, frozen SQPnP contract",
        "population": len(population),
        "measured": len(per_frame),
        "overlaps_existing_eval": sum(1 for r in population if r["overlaps_existing_eval"]),
        "summary": summary,
    }, indent=2) + "\n")

    print(f"\nmeasured {len(per_frame)} / population {len(population)}")
    print(f"{'group':22}{'n':>5}{'reproj':>8}{'valid':>8}{'surf med cm':>13}"
          f"{'depth med cm':>14}{'ring cm':>10}{'face<ring':>11}{'edge px':>9}")
    print("-" * 100)
    for name in ["ALL", "DAY", "NIGHT"] + sorted({r["source_recording"] for r in per_frame}):
        b = summary.get(name)
        if not b or not b.get("n"):
            continue
        print(f"{name:22}{b['n']:5d}{b.get('reference_reproj_px_median', float('nan')):8.2f}"
              f"{b.get('face_valid_fraction_median', float('nan')):8.3f}"
              f"{b.get('surface_residual_median_cm_median', float('nan')):13.1f}"
              f"{b.get('depth_residual_median_cm_median', float('nan')):14.1f}"
              f"{b.get('ring_surface_residual_median_cm_median', float('nan')):10.1f}"
              f"{b.get('fraction_face_better_than_ring', float('nan')):11.3f}"
              f"{b.get('manual_edge_to_depth_edge_median_px_median', float('nan')):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
