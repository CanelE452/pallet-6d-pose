"""Truncation-Aware Canvas Augmentation.

The training set audit found 96.7% of samples show the whole pallet, 3.3% have
any corner off-screen, and the centroid is off-screen in 0.0% of them, while the
population that fails is 76.9% truncated with the bounding box 32.8px past the
frame edge.  The existing `utils_dataset.apply_truncation_aug` is not a
counter-example: it crops the pallet at an edge and then calls `_trunc_pad_back`,
which reflect-pads and rescales until every one of the nine keypoints is back
inside a [0.20, 0.80] band.  It produces *cropped-looking* frames in which
nothing is actually off-screen, which is why the model has never been supervised
on the regime that breaks it.

TACA keeps the crop and drops the pad-back.  A corner that leaves the frame stays
out, and Phase C's target policy gives it no heatmap target and a loss mask of
zero rather than a clamped or sentinel one.

The scale-normalisation branch uses constant grey, not reflect: the padding audit
measured constant127 >= replicate > reflect on every recovery metric, so the
effect was canvas margin rather than context continuation, and mirrored texture
was actively worse.
"""
from __future__ import annotations

import cv2
import numpy as np

OUT_W, OUT_H = 640, 480
ASPECT = OUT_W / OUT_H
PAD_PIXELS = 100                      # dope_predict_mp4_pad.py:353
PAD_VALUE = (127, 127, 127)
MAX_ATTEMPTS = 10                     # Phase D1, no unbounded retry

# fixed before any result was seen
CLASS_LEGACY = "legacy"
CLASS_TRUNCATION = "frame_edge_truncation"
CLASS_SCALE = "constant_margin_scale"
SAMPLING = ((CLASS_LEGACY, 0.50), (CLASS_TRUNCATION, 0.25), (CLASS_SCALE, 0.25))

# Phase D1 acceptance envelope, derived from the D13 failure population
IN_FRAME_MIN, IN_FRAME_MAX = 4, 7
BORDER_MIN, BORDER_MAX = -64.0, 16.0
WIDTH_RATIO_MIN, WIDTH_RATIO_MAX = 0.65, 1.05


def pick_class(rng) -> str:
    draw = rng.random()
    total = 0.0
    for name, share in SAMPLING:
        total += share
        if draw < total:
            return name
    return CLASS_LEGACY


def _in_frame(points: np.ndarray) -> np.ndarray:
    return ((points[:, 0] >= 0) & (points[:, 0] < OUT_W)
            & (points[:, 1] >= 0) & (points[:, 1] < OUT_H))


def geometry_stats(points: np.ndarray) -> dict:
    """Border proximity is negative when the box runs past the frame edge."""
    corners = np.asarray(points, dtype=np.float64)[:8]
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    inside = _in_frame(np.asarray(points, dtype=np.float64))
    return {"in_frame_corners": int(inside[:8].sum()),
            "off_screen_corners": int((~inside[:8]).sum()),
            "centroid_in_frame": bool(inside[8]),
            "border_proximity_px": float(min(x0, y0, OUT_W - x1, OUT_H - y1)),
            "bbox_width_ratio": float((x1 - x0) / OUT_W),
            "bbox_height_ratio": float((y1 - y0) / OUT_H)}


def _crop_window(points: np.ndarray, width: int, height: int, side: str,
                 fraction: float, rng) -> tuple | None:
    """A 4:3 window that cuts the pallet on `side`; the port of the existing
    window builder, with the pad-back step deliberately absent."""
    corners = np.asarray(points, dtype=np.float64)
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    span_x, span_y = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    left = x0 - span_x * rng.uniform(0.05, 0.20)
    right = x1 + span_x * rng.uniform(0.05, 0.20)
    top = y0 - span_y * rng.uniform(0.05, 0.20)
    bottom = y1 + span_y * rng.uniform(0.05, 0.20)
    if "L" in side:
        left = x0 + fraction * span_x
    if "R" in side:
        right = x1 - fraction * span_x
    if "T" in side:
        top = y0 + fraction * span_y
    if "B" in side:
        bottom = y1 - fraction * span_y
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(width), right), min(float(height), bottom)
    window_w, window_h = right - left, bottom - top
    if window_w < 20 or window_h < 20:
        return None
    if window_w / window_h > ASPECT:
        need = window_w / ASPECT
        grow = need - window_h
        share = grow * (0.0 if "T" in side else (1.0 if "B" in side else 0.5))
        top -= share
        bottom += grow - share
    else:
        need = window_h * ASPECT
        grow = need - window_w
        share = grow * (0.0 if "L" in side else (1.0 if "R" in side else 0.5))
        left -= share
        right += grow - share
    left, top = max(0.0, left), max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    if right - left < 20 or bottom - top < 20:
        return None
    return left, top, right - left, bottom - top


def frame_edge_truncation(image: np.ndarray, points: np.ndarray, rng):
    """Crop so the pallet is cut by the output frame and leave it cut."""
    height, width = image.shape[:2]
    sides = ("L", "R", "T", "B", "LT", "LB", "RT", "RB")
    for _ in range(MAX_ATTEMPTS):
        side = sides[rng.randrange(len(sides))]
        fraction = rng.uniform(0.10, 0.45)
        window = _crop_window(points, width, height, side, fraction, rng)
        if window is None:
            continue
        left, top, window_w, window_h = window
        patch = image[int(round(top)):int(round(top + window_h)),
                      int(round(left)):int(round(left + window_w))]
        if patch.size == 0:
            continue
        out = cv2.resize(patch, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = OUT_W / window_w, OUT_H / window_h
        moved = np.asarray(points, dtype=np.float64).copy()
        moved[:, 0] = (moved[:, 0] - left) * scale_x
        moved[:, 1] = (moved[:, 1] - top) * scale_y
        stats = geometry_stats(moved)
        if not (IN_FRAME_MIN <= stats["in_frame_corners"] <= IN_FRAME_MAX):
            continue
        if stats["off_screen_corners"] < 1:
            continue
        if not (BORDER_MIN <= stats["border_proximity_px"] <= BORDER_MAX):
            continue
        if not (WIDTH_RATIO_MIN <= stats["bbox_width_ratio"] <= WIDTH_RATIO_MAX):
            continue
        return out, moved, stats
    return None


def constant_margin_scale(image: np.ndarray, points: np.ndarray):
    """Constant-grey canvas margin, the geometry of pad_frame with pad = 100."""
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(image, PAD_PIXELS, PAD_PIXELS, PAD_PIXELS,
                                PAD_PIXELS, cv2.BORDER_CONSTANT, value=PAD_VALUE)
    out = cv2.resize(padded, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas_w, canvas_h = width + 2 * PAD_PIXELS, height + 2 * PAD_PIXELS
    moved = np.asarray(points, dtype=np.float64).copy()
    moved[:, 0] = (moved[:, 0] + PAD_PIXELS) * (width / canvas_w)
    moved[:, 1] = (moved[:, 1] + PAD_PIXELS) * (height / canvas_h)
    return out, moved, geometry_stats(moved)


def apply(image: np.ndarray, points: np.ndarray, rng):
    """Returns (image, points, record).  Falls back to legacy on failure."""
    chosen = pick_class(rng)
    if chosen == CLASS_TRUNCATION:
        result = frame_edge_truncation(image, points, rng)
        if result is not None:
            out, moved, stats = result
            return out, moved, {"class": CLASS_TRUNCATION, "fallback": False, **stats}
        return image, np.asarray(points, dtype=np.float64), {
            "class": CLASS_LEGACY, "fallback": True,
            **geometry_stats(np.asarray(points, dtype=np.float64))}
    if chosen == CLASS_SCALE:
        out, moved, stats = constant_margin_scale(image, points)
        return out, moved, {"class": CLASS_SCALE, "fallback": False, **stats}
    moved = np.asarray(points, dtype=np.float64)
    return image, moved, {"class": CLASS_LEGACY, "fallback": False,
                          **geometry_stats(moved)}
