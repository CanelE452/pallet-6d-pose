#!/usr/bin/env python3
"""Extract pixel/model evidence for human review of mixed incoming captures.

This script does not assign evaluation membership.  It writes one auditable
row per raw frame so material boundaries, no-pallet gaps and blur thresholds
can be reviewed before ``annotation.py`` exposes object-specific views.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


FIELDS = (
    "frame",
    "source_ordinal",
    "det_conf",
    "detection_count",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_area_fraction",
    "box_touches_border",
    "kp_mean_conf",
    "kp_min_conf",
    "kp_confident_count",
    "plastic_det_conf",
    "plastic_detection_count",
    "plastic_kp_mean_conf",
    "full_laplacian_var",
    "crop_laplacian_var",
    "crop_tenengrad",
    "crop_median_value",
    "crop_median_saturation",
    "wood_fraction_day",
    "wood_fraction_lowlight",
)


def _float(value: float) -> str:
    return f"{float(value):.8g}"


def _features(
    result,
    image: np.ndarray,
    frame: str,
    source_ordinal: int,
    pad: int,
    plastic_result=None,
) -> dict[str, str]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row = {field: "" for field in FIELDS}
    row.update({
        "frame": frame,
        "source_ordinal": str(source_ordinal),
        "det_conf": "0",
        "detection_count": str(len(result.boxes or [])),
        "full_laplacian_var": _float(cv2.Laplacian(
            gray, cv2.CV_64F).var()),
    })

    if plastic_result is not None:
        plastic_count = len(plastic_result.boxes or [])
        row["plastic_detection_count"] = str(plastic_count)
        if plastic_count:
            plastic_best = int(plastic_result.boxes.conf.argmax().item())
            row["plastic_det_conf"] = _float(
                plastic_result.boxes.conf[plastic_best].item())
            if (
                plastic_result.keypoints is not None
                and plastic_result.keypoints.conf is not None
                and len(plastic_result.keypoints.conf) > plastic_best
            ):
                conf = plastic_result.keypoints.conf[plastic_best]
                row["plastic_kp_mean_conf"] = _float(
                    conf.detach().float().mean().cpu().item())
    if result.boxes is None or len(result.boxes) == 0:
        return row

    best = int(result.boxes.conf.argmax().item())
    confidence = float(result.boxes.conf[best].item())
    x1f, y1f, x2f, y2f = (
        float(value) for value in result.boxes.xyxy[best].cpu().tolist())
    x1f -= pad
    y1f -= pad
    x2f -= pad
    y2f -= pad
    x1 = max(0, min(width - 1, int(np.floor(x1f))))
    y1 = max(0, min(height - 1, int(np.floor(y1f))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2f))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2f))))
    crop = image[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(crop_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_gray, cv2.CV_32F, 0, 1, ksize=3)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(crop.astype(np.float32))
    _hue, saturation, value = cv2.split(hsv)
    wood_day = (
        (red > 65.0)
        & (red > green * 1.05)
        & (red > blue * 1.18)
        & (green > blue * 1.05)
        & (saturation > 35)
    )
    wood_lowlight = (
        (red > 25.0)
        & (red > green * 1.05)
        & (red > blue * 1.18)
        & (green > blue * 1.03)
        & (saturation > 25)
    )
    row.update({
        "det_conf": _float(confidence),
        "x1": _float(x1f),
        "y1": _float(y1f),
        "x2": _float(x2f),
        "y2": _float(y2f),
        "box_area_fraction": _float(
            (x2f - x1f) * (y2f - y1f) / (width * height)),
        "box_touches_border": "true" if (
            x1f <= 2.0 or y1f <= 2.0
            or x2f >= width - 2.0 or y2f >= height - 2.0) else "false",
        "crop_laplacian_var": _float(cv2.Laplacian(
            crop_gray, cv2.CV_64F).var()),
        "crop_tenengrad": _float(np.mean(gx * gx + gy * gy)),
        "crop_median_value": _float(np.median(value)),
        "crop_median_saturation": _float(np.median(saturation)),
        "wood_fraction_day": _float(np.mean(wood_day)),
        "wood_fraction_lowlight": _float(np.mean(wood_lowlight)),
    })

    keypoint_conf = None
    if result.keypoints is not None and result.keypoints.conf is not None:
        conf = result.keypoints.conf
        if len(conf) > best:
            keypoint_conf = conf[best].detach().float().cpu().numpy()
    if keypoint_conf is not None and keypoint_conf.size:
        row.update({
            "kp_mean_conf": _float(np.mean(keypoint_conf)),
            "kp_min_conf": _float(np.min(keypoint_conf)),
            "kp_confident_count": str(int(np.sum(keypoint_conf >= 0.5))),
        })

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--plastic-model",
        type=Path,
        default=None,
        help=(
            "Optional plastic-specialist model. Its score is recorded separately; "
            "it is never treated as a wood/no-pallet classifier by itself."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--pad", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rgb_dir = args.session_dir / "rgb"
    paths = sorted(
        path for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"no images under {rgb_dir}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.pad < 0:
        parser.error("--pad must be non-negative")
    model = YOLO(str(args.model))
    plastic_model = (
        YOLO(str(args.plastic_model)) if args.plastic_model is not None else None)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        processed = 0
        for offset in range(0, len(paths), args.batch_size):
            batch = paths[offset:offset + args.batch_size]
            originals = []
            inputs = []
            for path in batch:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read image: {path}")
                originals.append(image)
                inputs.append(
                    cv2.copyMakeBorder(
                        image,
                        args.pad,
                        args.pad,
                        args.pad,
                        args.pad,
                        cv2.BORDER_REFLECT_101,
                    ) if args.pad else image
                )
            results = model.predict(
                inputs,
                imgsz=args.imgsz,
                conf=args.conf_floor,
                device=args.device,
                verbose=False,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"model returned {len(results)} results for {len(batch)} frames")
            plastic_results = None
            if plastic_model is not None:
                plastic_results = plastic_model.predict(
                    inputs,
                    imgsz=args.imgsz,
                    conf=args.conf_floor,
                    device=args.device,
                    verbose=False,
                )
                if len(plastic_results) != len(batch):
                    raise RuntimeError(
                        "plastic model returned "
                        f"{len(plastic_results)} results for {len(batch)} frames")
            if plastic_results is None:
                plastic_results = [None] * len(batch)
            for path, original, result, plastic_result in zip(
                batch, originals, results, plastic_results
            ):
                processed += 1
                writer.writerow(_features(
                    result,
                    original,
                    path.name,
                    processed,
                    args.pad,
                    plastic_result,
                ))
            if processed % 512 < len(batch) or processed == len(paths):
                handle.flush()
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[pixel audit] {processed}/{len(paths)} "
                    f"({processed / elapsed:.1f} frame/s)",
                    flush=True,
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(f"[pixel audit] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
