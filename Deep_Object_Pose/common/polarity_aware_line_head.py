"""Polarity-aware 5-class semantic line head, target generator, and losses.

The oracle screen showed that a *top/base aware* line representation resolves
the vertical polarity ambiguity that unsigned width/depth/vertical lines cannot
(inversion 30/86 -> 3/86).  This module is the learned counterpart: it produces
exactly five raster maps

    top_width, top_depth, base_width, base_depth, vertical

and nothing else.  No vector, offset, endpoint, tangent, or voting output
exists here — those representations were tested and rejected in earlier tracks,
and a line is never converted back into a keypoint.

The head only *re-ranks* an existing SAI candidate set; it does not generate
candidates.  That keeps this a clean measurement of polarity-selection
capability rather than a joint change of generation and selection at once.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pallet_graph_geometry as PG
    import pallet_polarity_disambiguation as PPD
except ImportError:  # pragma: no cover
    from . import pallet_graph_geometry as PG  # type: ignore
    from . import pallet_polarity_disambiguation as PPD  # type: ignore

TARGET_GRID = 100
TARGET_SIGMA_CELLS = 1.5
MASK_DILATION_CELLS = 2
GATE_EPSILON = 0.15
GATE_POOL = 5
CONTRAST_MARGIN = 1.0
POSITIVE_THRESHOLD = 0.5      # frozen before training
OPPOSITE_THRESHOLD = 0.1
POLARITY_PAIRS = (("top_width", "base_width"), ("top_depth", "base_depth"))
CLASS_ORDER = PPD.POLARITY_CLASSES


# ============================================================================
# Target generation
# ============================================================================
def decode_mask_rle(entry: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """JSON mask_rle -> binary mask.  PNG masks are never read (gradient maps)."""
    import sys

    if "utils_pvnet" not in sys.modules:
        try:
            from utils_pvnet import decode_mask_rle as _decode
        except ImportError:  # pragma: no cover
            _decode = None
    else:  # pragma: no cover
        _decode = sys.modules["utils_pvnet"].decode_mask_rle
    from utils_pvnet import decode_mask_rle as _decode  # noqa: F811

    mask = _decode(entry).astype(np.uint8)
    if mask.shape[:2] != shape:
        import cv2

        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def build_polarity_targets(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], image_size: tuple[int, int],
    visible_mask: np.ndarray, grid: int = TARGET_GRID,
    sigma_cells: float = TARGET_SIGMA_CELLS,
    dilation_cells: int = MASK_DILATION_CELLS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Five soft distance-field targets on a ``grid x grid`` lattice.

    Only edge samples that are (a) self-visible and (b) inside the dilated
    mask_rle support become positives.  A projected line that falls outside the
    real mask is NOT forced to be a positive: doing so would teach the head to
    hallucinate occluded structure.
    """
    import cv2

    width, height = int(image_size[0]), int(image_size[1])
    edges = PPD.polarity_edge_classes(dims)
    visibility = PG.visible_edges(rotation, translation, dims)
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    projected, depth = PG.project_points(corners, rotation, translation, intrinsics)

    small = cv2.resize(visible_mask.astype(np.uint8), (grid, grid),
                       interpolation=cv2.INTER_NEAREST)
    if dilation_cells > 0:
        kernel = np.ones((2 * dilation_cells + 1, 2 * dilation_cells + 1), np.uint8)
        small = cv2.dilate(small, kernel)

    masks = {name: np.zeros((grid, grid), np.uint8) for name in CLASS_ORDER}
    stats = {name: 0 for name in CLASS_ORDER}
    dropped_outside = 0
    for (i, j), line_class in edges:
        if depth[i] <= 1e-6 or depth[j] <= 1e-6:
            continue
        if not visibility[(i, j)]:
            continue
        clipped = PG.clip_segment_to_image(projected[i], projected[j], width, height)
        if clipped is None:
            continue
        for q in PG.sample_along(clipped[0], clipped[1], pixels_per_sample=1.0):
            gx = int(round(float(q[0]) * (grid - 1) / max(width - 1, 1)))
            gy = int(round(float(q[1]) * (grid - 1) / max(height - 1, 1)))
            if not (0 <= gx < grid and 0 <= gy < grid):
                continue
            if not small[gy, gx]:
                dropped_outside += 1
                continue
            masks[line_class][gy, gx] = 1
            stats[line_class] += 1

    targets = np.zeros((len(CLASS_ORDER), grid, grid), dtype=np.float32)
    for index, name in enumerate(CLASS_ORDER):
        if masks[name].sum() == 0:
            continue
        distance = cv2.distanceTransform(1 - masks[name], cv2.DIST_L2, 3)
        targets[index] = np.exp(-(distance**2) / (2.0 * sigma_cells**2))
    info = {
        "per_class_samples": stats,
        "dropped_outside_mask": dropped_outside,
        "nonempty_classes": int(sum(1 for v in stats.values() if v > 0)),
        "has_top": bool(stats["top_width"] + stats["top_depth"] > 0),
        "has_base": bool(stats["base_width"] + stats["base_depth"] > 0),
        "mask_support_fraction": (
            1.0 - dropped_outside / max(sum(stats.values()) + dropped_outside, 1)),
    }
    return targets, info


# ============================================================================
# Model
# ============================================================================
def find_high_resolution_feature(
    backbone: nn.Module, sample: torch.Tensor, target_size: int = TARGET_GRID
) -> tuple[int, int]:
    """Locate the layer whose output is target_size x target_size at runtime.

    Deliberately not a hardcoded index: the VGG trunk has been edited in this
    repo, and guessing the layer is exactly the kind of silent mismatch that
    would invalidate every downstream number.
    """
    found: list[tuple[int, int]] = []
    handles = []

    def make_hook(index: int):
        def hook(module, inputs, output):
            if output.ndim == 4 and output.shape[-1] == target_size \
                    and output.shape[-2] == target_size:
                found.append((index, int(output.shape[1])))
        return hook

    for index, module in enumerate(backbone):
        handles.append(module.register_forward_hook(make_hook(index)))
    with torch.no_grad():
        backbone(sample)
    for handle in handles:
        handle.remove()
    if not found:
        raise RuntimeError(f"no {target_size}x{target_size} feature found in backbone")
    return found[-1]


class FreshMaskHead(nn.Module):
    """New mask head trained on mask_rle.  The ep57 seg head is never reused."""

    def __init__(self, in_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.a = nn.Conv2d(channels, channels, 3, padding=1)
        self.na = nn.GroupNorm(8, channels)
        self.b = nn.Conv2d(channels, channels, 3, padding=1)
        self.nb = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.na(self.a(x)), inplace=True)
        y = self.nb(self.b(y))
        return F.relu(x + y, inplace=True)


class PolarityLineHead(nn.Module):
    """Exactly five raster logits.  No vector/offset/endpoint output exists."""

    def __init__(self, feature_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2 * feature_channels + 1, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(_ResidualBlock(hidden), _ResidualBlock(hidden))
        self.out = nn.Conv2d(hidden, len(CLASS_ORDER), 1)

    def forward(self, feature: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([feature, gate, gate * feature], dim=1)
        return self.out(self.blocks(self.stem(stacked)))


def soft_gate(mask_logits: torch.Tensor) -> torch.Tensor:
    """Dilated soft gate in [GATE_EPSILON, 1]; never an exact hard zero.

    Detached so the line loss cannot train the mask head through the gate — the
    mask head must learn from mask supervision alone, otherwise 'mask helps'
    would be unfalsifiable.
    """
    probability = torch.sigmoid(mask_logits).detach()
    dilated = F.max_pool2d(probability, GATE_POOL, stride=1, padding=GATE_POOL // 2)
    return GATE_EPSILON + (1.0 - GATE_EPSILON) * dilated


# ============================================================================
# Losses
# ============================================================================
def dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3)) + eps
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - numerator / denominator).mean()


def mask_loss(mask_logits: torch.Tensor, mask_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(mask_logits, mask_target) + dice_loss(
        torch.sigmoid(mask_logits), mask_target)


def line_map_loss(
    line_logits: torch.Tensor, targets: torch.Tensor,
    positive_weight: torch.Tensor
) -> torch.Tensor:
    """Class-balanced weighted BCE over the five soft distance-field targets."""
    weight = positive_weight.view(1, -1, 1, 1).to(line_logits.device)
    return F.binary_cross_entropy_with_logits(
        line_logits, targets, pos_weight=weight)


def polarity_contrast_loss(
    line_logits: torch.Tensor, targets: torch.Tensor,
    margin: float = CONTRAST_MARGIN
) -> torch.Tensor:
    """Push top above base where the target is unambiguous, and vice versa.

    Pixels where both paired targets are high are excluded: at a viewing angle
    where top and base project onto each other, demanding a margin would ask
    the head to invent evidence that the image does not contain.
    """
    index = {name: i for i, name in enumerate(CLASS_ORDER)}
    losses = []
    for top_name, base_name in POLARITY_PAIRS:
        top_logit = line_logits[:, index[top_name]]
        base_logit = line_logits[:, index[base_name]]
        top_target = targets[:, index[top_name]]
        base_target = targets[:, index[base_name]]
        top_positive = (top_target >= POSITIVE_THRESHOLD) & (base_target <= OPPOSITE_THRESHOLD)
        base_positive = (base_target >= POSITIVE_THRESHOLD) & (top_target <= OPPOSITE_THRESHOLD)
        if bool(top_positive.any()):
            losses.append(F.softplus(
                margin - (top_logit - base_logit))[top_positive].mean())
        if bool(base_positive.any()):
            losses.append(F.softplus(
                margin - (base_logit - top_logit))[base_positive].mean())
    if not losses:
        return line_logits.sum() * 0.0
    return torch.stack(losses).mean()


def outside_mask_penalty(
    line_logits: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """One-sided: suppress line probability outside the mask, never require it."""
    probability = torch.sigmoid(line_logits).sum(dim=1, keepdim=True)
    return ((1.0 - gate) * probability).mean()


# ============================================================================
# Shared O0 gradient association (extracted verbatim; numeric output unchanged)
# ============================================================================
# The oracle O0 evidence used a Canny response with a distance-transform
# tolerance to keep only the projected edge stretches that a real image
# gradient actually supports.  Extracting it here (instead of re-deriving a
# threshold) is what makes the learned T2 target and the oracle O0 evidence the
# SAME definition; a fresh threshold would silently change what "observed"
# means and make the learned-vs-oracle comparison meaningless.
O0_CANNY = (100, 200)          # CANNY_SETTINGS[1] in the runner
O0_ASSOCIATION_RADIUS_PX = 4.0


def gradient_association_mask(
    image_bgr: np.ndarray, canny: tuple[int, int] = O0_CANNY,
    radius_px: float = O0_ASSOCIATION_RADIUS_PX
) -> np.ndarray:
    """Pixels within ``radius_px`` of a real image edge response.

    Byte-identical to the runner's ``association_keep_mask``; a regression test
    asserts the two agree on real frames.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny[0], canny[1])
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return distance <= float(radius_px)


# ============================================================================
# Target modes (roles fixed BEFORE any number was produced)
# ============================================================================
# T0 MASK_FILTERED_LEGACY  — failure control.  Never the main target.
# T1 SELF_VISIBLE_FULL     — geometry upper bound.  May include edges that have
#                            no contrast in the image.  Never the main target.
# T2 OBSERVED_FRAGMENT     — MAIN target, fixed in advance: self-visible
#                            projection intersected with the SAME gradient
#                            association the oracle O0 used, and NO mask
#                            filtering.  mask_rle is pallet foreground, not a
#                            definition of which cuboid lines are valid; a
#                            pallet's fork slots make mask/hull = 0.52..0.80,
#                            so mask filtering deletes legitimate top edges.
TARGET_MODES = ("mask_filtered", "self_visible_full", "observed_fragment")
MAIN_TARGET_MODE = "observed_fragment"


def build_polarity_targets_v2(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], image_size: tuple[int, int],
    target_mode: str = MAIN_TARGET_MODE,
    visible_mask: Optional[np.ndarray] = None,
    image_bgr: Optional[np.ndarray] = None,
    grid: int = TARGET_GRID, sigma_cells: float = TARGET_SIGMA_CELLS,
    dilation_cells: int = MASK_DILATION_CELLS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Five soft distance-field targets under one of the three fixed modes."""
    import cv2

    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target_mode {target_mode!r}")
    if target_mode == "mask_filtered" and visible_mask is None:
        raise ValueError("mask_filtered requires visible_mask")
    if target_mode == "observed_fragment" and image_bgr is None:
        raise ValueError("observed_fragment requires image_bgr")

    width, height = int(image_size[0]), int(image_size[1])
    edges = PPD.polarity_edge_classes(dims)
    visibility = PG.visible_edges(rotation, translation, dims)
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    projected, depth = PG.project_points(corners, rotation, translation, intrinsics)

    keep_mask = None
    if target_mode == "mask_filtered":
        small = cv2.resize(np.asarray(visible_mask, np.uint8), (grid, grid),
                           interpolation=cv2.INTER_NEAREST)
        if dilation_cells > 0:
            kernel = np.ones((2 * dilation_cells + 1,) * 2, np.uint8)
            small = cv2.dilate(small, kernel)
        keep_mask = small
    elif target_mode == "observed_fragment":
        keep_mask = gradient_association_mask(image_bgr)  # full image resolution

    masks = {name: np.zeros((grid, grid), np.uint8) for name in CLASS_ORDER}
    stats = {name: 0 for name in CLASS_ORDER}
    projected_len = {name: 0 for name in CLASS_ORDER}
    dropped = 0
    for (i, j), line_class in edges:
        if depth[i] <= 1e-6 or depth[j] <= 1e-6:
            continue
        if not visibility[(i, j)]:
            continue
        clipped = PG.clip_segment_to_image(projected[i], projected[j], width, height)
        if clipped is None:
            continue
        for q in PG.sample_along(clipped[0], clipped[1], pixels_per_sample=1.0):
            projected_len[line_class] += 1
            px, py = int(round(float(q[0]))), int(round(float(q[1])))
            gx = int(round(float(q[0]) * (grid - 1) / max(width - 1, 1)))
            gy = int(round(float(q[1]) * (grid - 1) / max(height - 1, 1)))
            if not (0 <= gx < grid and 0 <= gy < grid):
                continue
            if target_mode == "mask_filtered":
                if not keep_mask[gy, gx]:
                    dropped += 1
                    continue
            elif target_mode == "observed_fragment":
                if not (0 <= px < width and 0 <= py < height and keep_mask[py, px]):
                    dropped += 1
                    continue
            masks[line_class][gy, gx] = 1
            stats[line_class] += 1

    targets = np.zeros((len(CLASS_ORDER), grid, grid), dtype=np.float32)
    for index, name in enumerate(CLASS_ORDER):
        if masks[name].sum() == 0:
            continue
        distance = cv2.distanceTransform(1 - masks[name], cv2.DIST_L2, 3)
        targets[index] = np.exp(-(distance**2) / (2.0 * sigma_cells**2))

    total_projected = sum(projected_len.values())
    info = {
        "target_mode": target_mode,
        "per_class_samples": stats,
        "per_class_projected": projected_len,
        "dropped": dropped,
        "retained_fraction": (
            sum(stats.values()) / max(total_projected, 1)),
        "nonempty_classes": int(sum(1 for v in stats.values() if v > 0)),
        "has_top": bool(stats["top_width"] + stats["top_depth"] > 0),
        "has_base": bool(stats["base_width"] + stats["base_depth"] > 0),
    }
    return targets, info
