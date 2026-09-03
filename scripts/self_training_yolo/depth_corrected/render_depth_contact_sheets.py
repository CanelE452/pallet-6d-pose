"""GATE 0 — 정렬 감사(자동 proxy + 사람이 볼 contact sheet) 와 역투영 smoke.

    python3 scripts/self_training_yolo/depth_corrected/render_depth_contact_sheets.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0

출력  contact_sheets/<recording>.jpg · ALIGNMENT_AUDIT.json · pointcloud_smoke/

자동 proxy 로 PASS/FAIL 을 정하지 않는다(§12).  용도는 "수십 픽셀 밀린 depth"
같은 명백한 오류를 찾는 것이다.  그래서 shift sweep 을 같이 돈다 — RGB 경계와
depth 불연속 경계의 일치도가 (0,0) 에서 최대면 정렬 근거이고, 다른 곳에서
최대면 어긋남의 근거다.

contact sheet 에는 recording · timestamp · pair delta · depth scale · alignment
mode 만 적는다.  GT · pose 오차 · keypoint 오차 · 모델 점수는 표시하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data/pallet/raw_data"
RECORDINGS = [
    ("day", "outside/capturepallet01"), ("day", "outside/capturepallet10"),
    ("day", "outside/capturepallet11"),
    ("night", "night/capturenight01"), ("night", "night/capturenight02"),
    ("night", "night/capturenight03"), ("night", "night/capturenight04"),
    ("night", "night/capturenight10"),
]
CACHES = [
    "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json",
    "data/pallet/results/paper_selftrain_site_v1/preflight/SITE_A_TEACHER_CACHE.json",
]
N_SAMPLES = 10
DEPTH_SCALE = 0.001          # DEPTH_SCALE_AUDIT 의 consumer contract 값
SHIFTS = range(-12, 13, 2)   # 정렬 진단용 이동 범위.  임계값이 아니다.
SMOKE_FRAMES = 2             # recording 당 역투영 smoke


def load_roi_index() -> dict[str, dict]:
    index = {}
    for relative in CACHES:
        payload = json.loads((REPO_ROOT / relative).read_text())
        for entry in payload["entries"]:
            index.setdefault(entry["image_path"], entry)
    return index


def edge_maps(rgb, depth_metric):
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_edge = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 50, 150) > 0
    valid = depth_metric > 0
    filled = np.where(valid, depth_metric, np.nan)
    gx = cv2.Sobel(np.nan_to_num(filled), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.nan_to_num(filled), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    # 불연속 = 이 프레임 자체의 상위 분위.  전역 임계값을 만들지 않는다.
    finite = magnitude[valid]
    if finite.size == 0:
        return rgb_edge, np.zeros_like(rgb_edge)
    depth_edge = (magnitude >= np.percentile(finite, 97)) & valid
    return rgb_edge, depth_edge


def agreement(rgb_edge, depth_edge, dx, dy):
    shifted = np.roll(np.roll(depth_edge, dy, axis=0), dx, axis=1)
    dilated = cv2.dilate(rgb_edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if shifted.sum() == 0:
        return 0.0
    return float((shifted & dilated).sum() / shifted.sum())


def colorise(depth_metric):
    valid = depth_metric > 0
    if not valid.any():
        return np.zeros((*depth_metric.shape, 3), np.uint8)
    low, high = np.percentile(depth_metric[valid], [2, 98])
    scaled = np.clip((depth_metric - low) / max(high - low, 1e-6), 0, 1)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sheets = out_dir / "contact_sheets"
    smoke_dir = out_dir / "pointcloud_smoke"
    sheets.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    roi_index = load_roi_index()

    alignment, smoke = {}, {}
    for lighting, relative in RECORDINGS:
        base = RAW / relative
        camera = np.loadtxt(base / "cam_K.txt")
        rgb_paths = sorted((base / "rgb").iterdir())
        positions = np.linspace(0, len(rgb_paths) - 1,
                                min(N_SAMPLES, len(rgb_paths))).round().astype(int)

        panels, best_shifts, zero_scores, out_of_bounds = [], [], [], []
        for order, index in enumerate(positions):
            rgb_path = rgb_paths[index]
            depth_path = base / "depth" / f"{rgb_path.stem}.png"
            rgb = cv2.imread(str(rgb_path))
            raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or raw is None:
                continue
            metric = raw.astype(np.float32) * DEPTH_SCALE
            metric[raw == np.iinfo(raw.dtype).max] = 0.0     # 포화는 측정으로 안 쓴다

            rgb_edge, depth_edge = edge_maps(rgb, metric)
            scores = {(dx, dy): agreement(rgb_edge, depth_edge, dx, dy)
                      for dx in SHIFTS for dy in SHIFTS}
            best = max(scores, key=scores.get)
            best_shifts.append(best)
            zero_scores.append(scores[(0, 0)])

            # 역투영 smoke: depth pixel -> 3D -> 같은 K 로 재투영
            if order < SMOKE_FRAMES:
                ys, xs = np.nonzero(metric > 0)
                pick = np.linspace(0, len(xs) - 1, min(20000, len(xs))).astype(int)
                xs, ys = xs[pick], ys[pick]
                z = metric[ys, xs]
                X = (xs - camera[0, 2]) * z / camera[0, 0]
                Y = (ys - camera[1, 2]) * z / camera[1, 1]
                u = camera[0, 0] * X / z + camera[0, 2]
                v = camera[1, 1] * Y / z + camera[1, 2]
                residual = float(np.abs(u - xs).max() + np.abs(v - ys).max())
                inside = ((u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0]))
                out_of_bounds.append(float(1.0 - inside.mean()))
                smoke.setdefault(base.name, []).append({
                    "frame": rgb_path.stem,
                    "points": int(len(xs)),
                    "z_min_m": float(z.min()), "z_median_m": float(np.median(z)),
                    "z_max_m": float(z.max()),
                    "x_range_m": [float(X.min()), float(X.max())],
                    "y_range_m": [float(Y.min()), float(Y.max())],
                    "roundtrip_pixel_residual": residual,
                    "nan_or_inf": bool(~np.isfinite(np.concatenate([X, Y, z])).all()),
                    "reprojected_out_of_bounds_fraction": float(1.0 - inside.mean()),
                })
                if order == 0:
                    with (smoke_dir / f"{base.name}__{rgb_path.stem}.ply").open("w") as h:
                        h.write("ply\nformat ascii 1.0\n"
                                f"element vertex {min(len(xs), 5000)}\n"
                                "property float x\nproperty float y\nproperty float z\n"
                                "property uchar red\nproperty uchar green\n"
                                "property uchar blue\nend_header\n")
                        for i in range(min(len(xs), 5000)):
                            b, g, r = rgb[ys[i], xs[i]]
                            h.write(f"{X[i]:.4f} {Y[i]:.4f} {z[i]:.4f} {r} {g} {b}\n")

            if order < 6:
                entry = roi_index.get(str(rgb_path.relative_to(REPO_ROOT)))
                overlay = rgb.copy()
                overlay[depth_edge] = (0, 255, 255)
                roi_panel = rgb.copy()
                if entry and entry.get("top1"):
                    x1, y1, x2, y2 = [int(round(v)) for v in entry["top1"]["box_xyxy"]]
                    cv2.rectangle(roi_panel, (x1, y1), (x2, y2), (90, 255, 120), 2)
                    mask = np.zeros(metric.shape, bool)
                    mask[max(y1, 0):y2, max(x1, 0):x2] = True
                    roi_panel[mask & (metric > 0)] = (
                        0.5 * roi_panel[mask & (metric > 0)] + np.array([0, 0, 128]))
                mask_panel = np.dstack([(metric > 0).astype(np.uint8) * 255] * 3)
                panels.append(np.hstack([
                    cv2.resize(p, (213, 160)) for p in
                    (rgb, colorise(metric), mask_panel, overlay, roi_panel)]))

        if panels:
            header_lines = [
                f"{base.name}   {lighting}   depth scale {DEPTH_SCALE} (consumer contract)",
                "pair delta UNMEASURABLE (one shared timestamp)   "
                "alignment mode ASSUMED_ALIGNED_UNPROVEN",
                "panels: RGB | metric depth | valid mask | RGB+depth edges | R0 ROI",
            ]
            width = panels[0].shape[1]
            header = np.full((22 * len(header_lines) + 8, width, 3), 18, np.uint8)
            for i, line in enumerate(header_lines):
                cv2.putText(header, line, (8, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.44, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imwrite(str(sheets / f"{base.name}.jpg"),
                        np.vstack([header] + panels), [cv2.IMWRITE_JPEG_QUALITY, 85])

        shift_counts: dict[str, int] = {}
        for shift in best_shifts:
            shift_counts[str(shift)] = shift_counts.get(str(shift), 0) + 1
        alignment[base.name] = {
            "lighting": lighting,
            "frames_checked": len(best_shifts),
            "edge_agreement_at_zero_shift_median": float(np.median(zero_scores)),
            "best_shift_counts": shift_counts,
            "best_shift_is_zero_rate": float(np.mean(
                [1.0 if s == (0, 0) else 0.0 for s in best_shifts])),
            "median_best_abs_shift_px": float(np.median(
                [abs(dx) + abs(dy) for dx, dy in best_shifts])),
            "reprojection_out_of_bounds_rate": (float(np.mean(out_of_bounds))
                                                if out_of_bounds else None),
        }
        print(f"  {base.name:20} zero-shift agree {alignment[base.name]['edge_agreement_at_zero_shift_median']:.3f}"
              f"  best-shift==0 {alignment[base.name]['best_shift_is_zero_rate']:.2f}"
              f"  median |shift| {alignment[base.name]['median_best_abs_shift_px']:.1f}px")

    (out_dir / "ALIGNMENT_AUDIT.json").write_text(json.dumps({
        "schema_version": "rgbd_alignment_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "proxy_does_not_decide": ("the shift sweep is a diagnostic for gross "
                                  "misalignment, not a pass/fail rule; the visual sheets "
                                  "are the other half and a clear mismatch there fails "
                                  "the gate regardless of this number"),
        "shift_range_px": [min(SHIFTS), max(SHIFTS)],
        "depth_scale_used": DEPTH_SCALE,
        "saturated_pixels_excluded": True,
        "recordings": alignment,
        "pointcloud_smoke": smoke,
    }, indent=2) + "\n")
    print(f"\nwrote {(out_dir / 'ALIGNMENT_AUDIT.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
