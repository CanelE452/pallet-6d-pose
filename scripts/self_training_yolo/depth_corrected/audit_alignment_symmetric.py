"""GATE 0B — 대칭 chamfer 정렬 진단과 거리대별 contact sheet.

    python3 scripts/self_training_yolo/depth_corrected/audit_alignment_symmetric.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0b

출력  ALIGNMENT_AUDIT_V2.json · contact_sheets/

Gate 0 의 비대칭 edge proxy 를 버리지 않고 옆에 둔다.  여기 더하는 것은 양방향
chamfer 거리다 — RGB 경계에서 depth 경계까지, 그리고 그 반대.  한쪽만 재면
경계가 많은 쪽이 유리해진다.

★ best shift 로 depth 를 옮기거나 offset 을 보정하지 않는다.  검증 전용이다.

contact sheet 프레임은 teacher bbox 면적의 close/mid/far 대역에서 결정론적으로
고른다.  GT·성능·오차 순위로 고르지 않는다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIFTS = range(-12, 13, 2)
PER_BAND = 4          # close/mid/far 각 4장 = recording 당 12장


def edges_of(rgb, metric):
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_edge = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 50, 150) > 0
    valid = metric > 0
    gx = cv2.Sobel(np.nan_to_num(np.where(valid, metric, 0)), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.nan_to_num(np.where(valid, metric, 0)), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    finite = magnitude[valid]
    if finite.size == 0:
        return rgb_edge, np.zeros_like(rgb_edge)
    return rgb_edge, (magnitude >= np.percentile(finite, 97)) & valid


def symmetric_chamfer(a, b, dx, dy):
    """양방향 평균 거리.  낮을수록 두 경계가 가깝다."""

    shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    if a.sum() == 0 or shifted.sum() == 0:
        return None
    dist_to_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_b = cv2.distanceTransform((~shifted).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dist_to_a[shifted].mean() + dist_to_b[a].mean()))


def colorise(metric):
    valid = metric > 0
    if not valid.any():
        return np.zeros((*metric.shape, 3), np.uint8)
    low, high = np.percentile(metric[valid], [2, 98])
    out = cv2.applyColorMap(
        (np.clip((metric - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO)
    out[~valid] = 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sheets = out_dir / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    lock = json.loads((out_dir / "GATE0B_METHOD_LOCK.json").read_text())
    conventions = lock["reused_repository_conventions"]
    faces = {k: tuple(v) for k, v in conventions["faces"].items()}
    shrink = float(lock["face_interior_shrink_ratio"])
    kp_conf_min = float(conventions["keypoint_validity_threshold"])
    scale = float(conventions["depth_scale_m_per_unit"])

    roi = json.loads((out_dir / "R0_FULL_ADAPT_ROI_CACHE.json").read_text())["frames"]
    by_recording: dict[str, list] = {}
    for relative_path, entry in roi.items():
        if entry.get("top1"):
            x1, y1, x2, y2 = entry["top1"]["box_xyxy"]
            by_recording.setdefault(entry["capture_session"], []).append(
                ((x2 - x1) * (y2 - y1), relative_path, entry))

    audit = {}
    for session, rows in sorted(by_recording.items()):
        rows.sort(key=lambda r: -r[0])
        bands = np.array_split(np.arange(len(rows)), 3)   # close / mid / far
        picked = []
        for band in bands:
            if band.size == 0:
                continue
            positions = np.linspace(0, band.size - 1,
                                    min(PER_BAND, band.size)).round().astype(int)
            picked += [rows[band[p]] for p in positions]

        camera = None
        zero_scores, best_scores, best_shifts, panels = [], [], [], []
        for order, (_, relative_path, entry) in enumerate(picked):
            image_path = REPO_ROOT / relative_path
            depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.png"
            rgb = cv2.imread(str(image_path))
            raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or raw is None:
                continue
            if camera is None:
                camera = np.loadtxt(image_path.parent.parent / "cam_K.txt")
            metric = raw.astype(np.float32) * scale
            metric[raw == np.iinfo(raw.dtype).max] = 0.0

            rgb_edge, depth_edge = edges_of(rgb, metric)
            scores = {}
            for dx in SHIFTS:
                for dy in SHIFTS:
                    value = symmetric_chamfer(rgb_edge, depth_edge, dx, dy)
                    if value is not None:
                        scores[(dx, dy)] = value
            if scores:
                best = min(scores, key=scores.get)
                zero_scores.append(scores.get((0, 0)))
                best_scores.append(scores[best])
                best_shifts.append(best)

            top = entry["top1"]
            keypoints = np.asarray(top["keypoints_xy"], np.float32)
            confidences = np.nan_to_num(np.asarray(top["keypoints_conf"], float), nan=0.0)
            x1, y1, x2, y2 = [int(round(v)) for v in top["box_xyxy"]]

            strict_panel = np.dstack([(metric > 0).astype(np.uint8) * 255] * 3)
            bbox_panel = rgb.copy()
            cv2.rectangle(bbox_panel, (x1, y1), (x2, y2), (90, 255, 120), 2)
            face_panel = rgb.copy()
            face_depth = np.zeros_like(metric)
            for name, indices in faces.items():
                if (confidences[list(indices)] < kp_conf_min).any():
                    continue
                hull = cv2.convexHull(
                    keypoints[list(indices)].reshape(-1, 1, 2)).reshape(-1, 2)
                if len(hull) < 3:
                    continue
                centroid = hull.mean(axis=0)
                inner = (centroid + (hull - centroid) * (1 - shrink)).astype(np.int32)
                cv2.polylines(face_panel, [inner], True, (80, 220, 255), 2)
                mask = np.zeros(metric.shape, np.uint8)
                cv2.fillPoly(mask, [inner], 1)
                face_depth[mask.astype(bool)] = metric[mask.astype(bool)]
            overlay = rgb.copy()
            overlay[depth_edge] = (0, 255, 255)
            if order < 12:
                panels.append(np.hstack([cv2.resize(p, (178, 134)) for p in (
                    rgb, colorise(metric), strict_panel, bbox_panel,
                    face_panel, colorise(face_depth), overlay)]))

        if panels:
            lines = [f"{session}   depth scale {scale}   alignment ASSUMED_ALIGNED_UNPROVEN",
                     "close/mid/far by teacher bbox area (not by GT, not by error)",
                     "RGB | depth | strict mask | bbox | face interiors | face depth | edges"]
            width = panels[0].shape[1]
            header = np.full((22 * len(lines) + 8, width, 3), 18, np.uint8)
            for i, line in enumerate(lines):
                cv2.putText(header, line, (8, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imwrite(str(sheets / f"{session}__GATE0B.jpg"),
                        np.vstack([header] + panels), [cv2.IMWRITE_JPEG_QUALITY, 85])

        counts: dict[str, int] = {}
        for shift in best_shifts:
            counts[str(shift)] = counts.get(str(shift), 0) + 1
        audit[session] = {
            "frames_checked": len(best_shifts),
            "symmetric_chamfer_zero_shift_median": (float(np.median(zero_scores))
                                                    if zero_scores else None),
            "symmetric_chamfer_best_median": (float(np.median(best_scores))
                                              if best_scores else None),
            "improvement_from_shifting_px": (
                float(np.median(zero_scores) - np.median(best_scores))
                if zero_scores and best_scores else None),
            "best_shift_counts": counts,
            "best_shift_is_zero_rate": float(np.mean(
                [1.0 if s == (0, 0) else 0.0 for s in best_shifts])) if best_shifts else None,
            "median_best_abs_shift_px": float(np.median(
                [abs(dx) + abs(dy) for dx, dy in best_shifts])) if best_shifts else None,
        }
        print(f"  {session:20} chamfer@0 {audit[session]['symmetric_chamfer_zero_shift_median']:6.2f}"
              f"  best {audit[session]['symmetric_chamfer_best_median']:6.2f}"
              f"  gain {audit[session]['improvement_from_shifting_px']:5.2f} px"
              f"  |shift| {audit[session]['median_best_abs_shift_px']:.1f}")

    (out_dir / "ALIGNMENT_AUDIT_V2.json").write_text(json.dumps({
        "schema_version": "alignment_audit_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "metric": "symmetric chamfer distance in pixels between RGB structural edges and depth discontinuity edges; lower is closer",
        "gate0_proxy_not_discarded": "data/pallet/results/paper_depth_selftrain_v1/gate0/ALIGNMENT_AUDIT.json",
        "depth_was_not_shifted": True,
        "best_shift_used_only_for_verification": True,
        "how_to_read": ("if the stored depth were misaligned by a fixed offset, shifting "
                        "it would cut the chamfer distance substantially. A small gain "
                        "means no large rigid offset is present; it does not prove "
                        "sub-pixel calibration."),
        "shift_range_px": [min(SHIFTS), max(SHIFTS)],
        "recordings": audit,
    }, indent=2) + "\n")
    print(f"\nwrote {(out_dir / 'ALIGNMENT_AUDIT_V2.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
