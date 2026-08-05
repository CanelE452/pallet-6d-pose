"""Stage-1 target policy.

Three things the existing pipeline does not do, and that the failure analysis
says it has to:

* corner and centroid channels get different Gaussian widths, because the
  deployment decoder needs a centroid target of at least 2.5 to survive its own
  sigma = 3 blur while corners want to stay narrow at 2.0;
* a corner whose projected centre leaves the frame gets **no** target and a loss
  mask of zero -- not a border clamp, not a sentinel, not a truncated Gaussian --
  because a clamped target teaches the network to put a peak at the edge and a
  sentinel teaches it to put one at a fixed pixel;
* validity is decided from the transformed coordinate, never from "did the
  Gaussian come out all-zero", which silently conflates an off-screen corner
  with a corner the renderer never produced.

Visibility is a separate three-state label so that an off-screen channel can be
excluded at assembly time instead of being trusted.
"""
from __future__ import annotations

import numpy as np

BELIEF = 50
CORNER_SIGMA = 2.0
CENTROID_SIGMA = 2.5
N_KP = 9

VIS_VISIBLE, VIS_OCCLUDED, VIS_OFF_SCREEN = 0, 1, 2


def channel_sigma(channel: int) -> float:
    return CENTROID_SIGMA if channel == N_KP - 1 else CORNER_SIGMA


def to_belief_grid(points: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.asarray(points, dtype=np.float64).copy()
    grid[:, 0] = grid[:, 0] * BELIEF / float(width)
    grid[:, 1] = grid[:, 1] * BELIEF / float(height)
    return grid


def in_frame_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return ((pts[:, 0] >= 0) & (pts[:, 0] < width)
            & (pts[:, 1] >= 0) & (pts[:, 1] < height))


def gaussian_channel(centre, sigma: float) -> np.ndarray:
    axis = np.arange(BELIEF, dtype=np.float64)
    gx = np.exp(-((axis - centre[0]) ** 2) / (2.0 * sigma ** 2))
    gy = np.exp(-((axis - centre[1]) ** 2) / (2.0 * sigma ** 2))
    return np.outer(gy, gx)


def build_targets(points: np.ndarray, width: int, height: int,
                  occluded: np.ndarray | None = None,
                  source_valid: np.ndarray | None = None):
    """Belief targets, per-channel loss mask and three-state visibility.

    `source_valid` is the loader's own channel mask: a channel the renderer never
    produced stays invalid regardless of where its coordinate would fall.
    """
    pts = np.asarray(points, dtype=np.float64)
    inside = in_frame_mask(pts, width, height)
    if source_valid is None:
        source_valid = np.ones(N_KP, dtype=bool)
    source_valid = np.asarray(source_valid, dtype=bool)
    if occluded is None:
        occluded = np.zeros(N_KP, dtype=bool)
    occluded = np.asarray(occluded, dtype=bool)

    grid = to_belief_grid(pts, width, height)
    belief = np.zeros((N_KP, BELIEF, BELIEF), dtype=np.float32)
    belief_mask = np.zeros(N_KP, dtype=np.float32)
    visibility = np.full(N_KP, VIS_OFF_SCREEN, dtype=np.int64)
    visibility_mask = np.zeros(N_KP, dtype=np.float32)

    for channel in range(N_KP):
        if not source_valid[channel]:
            continue                      # never rendered: no target, no label
        visibility_mask[channel] = 1.0
        if not inside[channel]:
            visibility[channel] = VIS_OFF_SCREEN
            continue                      # off-screen: mask 0, no Gaussian
        belief[channel] = gaussian_channel(grid[channel],
                                           channel_sigma(channel)).astype(np.float32)
        belief_mask[channel] = 1.0
        visibility[channel] = VIS_OCCLUDED if occluded[channel] else VIS_VISIBLE

    affinity_mask = np.zeros(16, dtype=np.float32)
    for corner in range(8):
        affinity_mask[2 * corner] = belief_mask[corner]
        affinity_mask[2 * corner + 1] = belief_mask[corner]

    truncated = bool(source_valid[:8].any()
                     and (~inside[:8] & source_valid[:8]).any())
    return {"belief": belief, "belief_mask": belief_mask,
            "affinity_mask": affinity_mask, "visibility": visibility,
            "visibility_mask": visibility_mask, "truncated": truncated,
            "in_frame": inside, "grid": grid}


def palletness_target(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Projected cuboid hull clipped to the frame, at belief resolution.

    Object-level extent, not the fork-slot RLE: the slots put real cuboid edges
    outside the instance mask, and using the mask removes structure that is
    genuinely part of the pallet.
    """
    import cv2

    grid = to_belief_grid(np.asarray(points, dtype=np.float64)[:8], width, height)
    canvas = np.zeros((BELIEF, BELIEF), dtype=np.uint8)
    finite = grid[np.isfinite(grid).all(axis=1)]
    if len(finite) >= 3:
        hull = cv2.convexHull(finite.astype(np.float32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 1)
    return canvas.astype(np.float32)
