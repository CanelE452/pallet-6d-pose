"""Deterministic line hypotheses from a predicted edge field, and the geometry
needed to score them against ground truth.

The learned twelve-edge head places corners at 44% within 20px on synthetic and
2.5% on the canonical set, while the same decoder on ground-truth geometry
reaches 98.7%.  The open question is whether the correct physical line is
present in the predicted field and merely not selected, or absent altogether.
Answering it needs hypotheses, not a dense argmax, so this module turns a
probability map into a ranked candidate list three different ways.

Everything here is deterministic.  No RANSAC, no random sampling, no learned
component: repeated runs on the same input return bitwise-identical arrays,
which is asserted in the tests.  A line is (theta, rho) in normal form with
theta in [0, pi) and rho = n . x for n = (cos theta, sin theta), plus the
endpoints of the supported segment.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Frozen before any result was seen.
TOP_K = 5
MIN_COMPONENT_PIXELS = 3
COMPONENT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
HOUGH_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
TOP_MASS_PERCENTS = (1.0, 2.0, 5.0, 10.0)
HOUGH_THETA_BIN_DEG = 1.0
HOUGH_RHO_BIN_CELLS = 1.0
HOUGH_NMS_THETA_DEG = 5.0
HOUGH_NMS_RHO_CELLS = 3.0
HOUGH_SUPPORT_CELLS = 1.5        # half the NMS rho window; fixed, never tuned
E3_REMOVE_CELLS = 2.0
CONDITION_MAX = 1.0e3            # 3-line intersection guard

STRICT_ANGLE_DEG = 5.0
STRICT_OFFSET_CELLS = 3.0
LOOSE_ANGLE_DEG = 10.0
LOOSE_OFFSET_CELLS = 5.0
STRICT_OVERLAP = 0.50
LOOSE_OVERLAP = 0.25

GT_BAND_CELLS = 1.5
NEAR_BACKGROUND_CELLS = (5.0, 10.0)


# ---------------------------------------------------------------------------
# line algebra
# ---------------------------------------------------------------------------
def normalise_line(normal: np.ndarray, rho: float) -> tuple[float, float]:
    """(theta in [0, pi), rho) with the normal on a canonical half-plane."""
    theta = float(np.arctan2(normal[1], normal[0]))
    if theta < 0:
        theta += np.pi
        rho = -rho
    if theta >= np.pi:
        theta -= np.pi
        rho = -rho
    return theta, float(rho)


def line_from_segment(a: np.ndarray, b: np.ndarray) -> Optional[tuple[float, float]]:
    direction = np.asarray(b, float) - np.asarray(a, float)
    length = float(np.hypot(*direction))
    if length < 1e-9:
        return None
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    return normalise_line(normal, float(normal @ np.asarray(a, float)))


def weighted_total_least_squares(points: np.ndarray, weights: np.ndarray
                                 ) -> Optional[tuple[float, float]]:
    """Best line through weighted points; the normal is the minor eigenvector."""
    total = float(weights.sum())
    if points.shape[0] < 2 or total <= 0:
        return None
    centre = (weights[:, None] * points).sum(axis=0) / total
    centred = points - centre
    covariance = (weights[:, None] * centred).T @ centred
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmin(values))]
    return normalise_line(normal, float(normal @ centre))


def segment_from_support(points: np.ndarray, theta: float, rho: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    direction = np.array([-np.sin(theta), np.cos(theta)])
    normal = np.array([np.cos(theta), np.sin(theta)])
    projection = points @ direction
    base = rho * normal
    return (base + direction * float(projection.min()),
            base + direction * float(projection.max()))


def make_candidate(theta: float, rho: float, points: np.ndarray,
                   weights: np.ndarray) -> dict[str, Any]:
    start, end = segment_from_support(points, theta, rho)
    direction = np.array([-np.sin(theta), np.cos(theta)])
    projection = points @ direction
    return {
        "theta": float(theta), "rho": float(rho),
        "p0": [float(start[0]), float(start[1])],
        "p1": [float(end[0]), float(end[1])],
        "mass": float(weights.sum()),
        "support": int(points.shape[0]),
        "length": float(projection.max() - projection.min()),
        "thickness": float(np.abs(points @ np.array([np.cos(theta), np.sin(theta)])
                                  - rho).mean()),
    }


# ---------------------------------------------------------------------------
# E1  connected component + weighted TLS
# ---------------------------------------------------------------------------
def component_tls(probability: np.ndarray, threshold: float,
                  top_k: int = TOP_K) -> list[dict[str, Any]]:
    import cv2
    mask = (probability >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if xs.size < MIN_COMPONENT_PIXELS:
            continue
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        weights = probability[ys, xs].astype(np.float64)
        line = weighted_total_least_squares(points, weights)
        if line is None:
            continue
        candidate = make_candidate(line[0], line[1], points, weights)
        candidate["score"] = candidate["mass"]
        candidates.append(candidate)
    candidates.sort(key=lambda c: (-c["score"], c["theta"], c["rho"]))
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# E2  deterministic weighted Hough
# ---------------------------------------------------------------------------
def weighted_hough(probability: np.ndarray, threshold: float,
                   top_k: int = TOP_K) -> list[dict[str, Any]]:
    grid = probability.shape[0]
    ys, xs = np.nonzero(probability >= threshold)
    if xs.size == 0:
        return []
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    weights = probability[ys, xs].astype(np.float64)

    thetas = np.deg2rad(np.arange(0.0, 180.0, HOUGH_THETA_BIN_DEG))
    rho_max = float(np.ceil(np.hypot(grid, grid)))
    rho_edges = np.arange(-rho_max, rho_max + HOUGH_RHO_BIN_CELLS, HOUGH_RHO_BIN_CELLS)
    rho_values = points @ np.stack([np.cos(thetas), np.sin(thetas)])   # (N, T)
    bins = np.clip(np.digitize(rho_values, rho_edges) - 1, 0, len(rho_edges) - 2)

    accumulator = np.zeros((len(thetas), len(rho_edges) - 1))
    for index in range(len(thetas)):
        np.add.at(accumulator[index], bins[:, index], weights)

    candidates: list[dict[str, Any]] = []
    working = accumulator.copy()
    theta_window = int(round(HOUGH_NMS_THETA_DEG / HOUGH_THETA_BIN_DEG))
    rho_window = int(round(HOUGH_NMS_RHO_CELLS / HOUGH_RHO_BIN_CELLS))
    normals = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    for _ in range(top_k):
        if working.max() <= 0:
            break
        flat = int(np.argmax(working))
        ti, ri = np.unravel_index(flat, working.shape)
        theta = float(thetas[ti])
        rho = float(0.5 * (rho_edges[ri] + rho_edges[ri + 1]))
        distance = np.abs(points @ normals[ti] - rho)
        inside = distance <= HOUGH_SUPPORT_CELLS
        if inside.sum() >= 2:
            candidate = make_candidate(theta, rho, points[inside], weights[inside])
            candidate["score"] = float(accumulator[ti, ri])
            candidates.append(candidate)
        t0, t1 = max(0, ti - theta_window), min(working.shape[0], ti + theta_window + 1)
        r0, r1 = max(0, ri - rho_window), min(working.shape[1], ri + rho_window + 1)
        working[t0:t1, r0:r1] = 0.0
    return candidates


# ---------------------------------------------------------------------------
# E3  top-mass weighted TLS with iterative residual removal
# ---------------------------------------------------------------------------
def top_mass_tls(probability: np.ndarray, percent: float,
                 top_k: int = TOP_K) -> list[dict[str, Any]]:
    flat = probability.reshape(-1)
    count = max(MIN_COMPONENT_PIXELS, int(round(flat.size * percent / 100.0)))
    order = np.argsort(-flat, kind="stable")[:count]
    ys, xs = np.unravel_index(order, probability.shape)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    weights = flat[order].astype(np.float64)
    candidates = []
    alive = np.ones(points.shape[0], bool)
    for _ in range(top_k):
        if alive.sum() < MIN_COMPONENT_PIXELS:
            break
        line = weighted_total_least_squares(points[alive], weights[alive])
        if line is None:
            break
        theta, rho = line
        candidate = make_candidate(theta, rho, points[alive], weights[alive])
        candidate["score"] = candidate["mass"]
        candidates.append(candidate)
        normal = np.array([np.cos(theta), np.sin(theta)])
        close = np.abs(points @ normal - rho) <= E3_REMOVE_CELLS
        alive = alive & ~close
    return candidates


EXTRACTORS = {
    "E1_COMPONENT_TLS": (component_tls, COMPONENT_THRESHOLDS),
    "E2_WEIGHTED_HOUGH": (weighted_hough, HOUGH_THRESHOLDS),
    "E3_TOP_MASS_TLS": (top_mass_tls, TOP_MASS_PERCENTS),
}


def extract(name: str, parameter: float, probability: np.ndarray
            ) -> list[dict[str, Any]]:
    function, _ = EXTRACTORS[name]
    return function(probability, parameter)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def angular_error_deg(theta_a: float, theta_b: float) -> float:
    difference = abs(theta_a - theta_b) % np.pi
    return float(np.rad2deg(min(difference, np.pi - difference)))


def match_metrics(candidate: dict[str, Any], gt_a: np.ndarray, gt_b: np.ndarray
                  ) -> Optional[dict[str, Any]]:
    truth = line_from_segment(gt_a, gt_b)
    if truth is None:
        return None
    theta_gt, rho_gt = truth
    theta, rho = candidate["theta"], candidate["rho"]
    normal = np.array([np.cos(theta), np.sin(theta)])
    normal_gt = np.array([np.cos(theta_gt), np.sin(theta_gt)])
    aligned_rho = rho if float(normal @ normal_gt) >= 0 else -rho
    midpoint = 0.5 * (np.asarray(gt_a, float) + np.asarray(gt_b, float))
    angle = angular_error_deg(theta, theta_gt)
    offset = abs(aligned_rho - rho_gt)

    direction_gt = np.array([-np.sin(theta_gt), np.cos(theta_gt)])
    gt_projection = np.array([direction_gt @ np.asarray(gt_a, float),
                              direction_gt @ np.asarray(gt_b, float)])
    gt_low, gt_high = gt_projection.min(), gt_projection.max()
    predicted = np.array([direction_gt @ np.asarray(candidate["p0"], float),
                          direction_gt @ np.asarray(candidate["p1"], float)])
    low, high = predicted.min(), predicted.max()
    overlap = max(0.0, min(gt_high, high) - max(gt_low, low))
    gt_length = max(gt_high - gt_low, 1e-9)

    strict_line = angle <= STRICT_ANGLE_DEG and offset <= STRICT_OFFSET_CELLS
    loose_line = angle <= LOOSE_ANGLE_DEG and offset <= LOOSE_OFFSET_CELLS
    ratio = float(overlap / gt_length)
    return {
        "angle_err_deg": angle, "offset_cells": float(offset),
        "midpoint_distance": float(abs(normal @ midpoint - rho)),
        "overlap_ratio": ratio,
        "endpoint_distance": float(min(
            np.hypot(*(np.asarray(candidate["p0"], float) - np.asarray(gt_a, float))),
            np.hypot(*(np.asarray(candidate["p0"], float) - np.asarray(gt_b, float))))),
        "strict_line": bool(strict_line), "loose_line": bool(loose_line),
        "strict_segment": bool(strict_line and ratio >= STRICT_OVERLAP),
        "loose_segment": bool(loose_line and ratio >= LOOSE_OVERLAP),
    }


# ---------------------------------------------------------------------------
# corner from three lines
# ---------------------------------------------------------------------------
def intersect_lines(lines: list[tuple[float, float]]
                    ) -> Optional[dict[str, Any]]:
    """Least-squares intersection of three infinite lines in normal form."""
    matrix = np.array([[np.cos(theta), np.sin(theta)] for theta, _ in lines])
    vector = np.array([rho for _, rho in lines])
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > CONDITION_MAX:
        return None
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    residual = float(np.linalg.norm(matrix @ solution - vector))
    return {"point": [float(solution[0]), float(solution[1])],
            "residual": residual, "condition": condition}
