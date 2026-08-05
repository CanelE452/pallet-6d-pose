"""Twelve physical cuboid edges, their fixed corner incidence, and the O5/O12
line-only corner decoders.

The five-class semantic representation cannot identify a corner even from
perfect ground truth.  A corner's three incident edges span the classes
{top_width, top_depth, vertical} or {base_width, base_depth, vertical}, and
those classes hold 2 + 2 + 4 = eight edges rather than three.  Worse, the class
set is identical for all four corners of a face, so the decoder cannot produce
more than two distinct points for eight corners.  The twelve-edge instance
representation does not have that degeneracy.  Both decoders live here so the
learned arms are scored by exactly the same code that produced the oracle
reference.

Nothing in this module is hand-written per index.  The edge list, the semantic
class of each edge, the corner incidence and the top/base polarity pairing are
all derived from the canonical 3D keypoints, so a revision of the corner
convention propagates instead of silently disagreeing.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any, Optional

import numpy as np

_CHALLENGE = pathlib.Path(__file__).resolve().parents[2] / "challenge/scripts"
if str(_CHALLENGE) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE))

import annotate_pnp as APNP  # noqa: E402

N_CORNERS = 8
N_EDGES = 12
INCIDENT_PER_CORNER = 3
SEMANTIC_CLASSES = ("top_width", "top_depth", "base_width", "base_depth", "vertical")
EXPECTED_CLASS_COUNTS = (2, 2, 2, 2, 4)

# Frozen by the c190284 oracle screen; never tuned against a result.
TAU = 5.0                     # belief cells
TARGET_SIGMA_CELLS = 1.5      # identical to polarity_aware_line_head.TARGET_SIGMA_CELLS
PROB_FLOOR = 1.0e-6           # inversion floor -> distance cap sigma*sqrt(-2 ln floor)

# Reference dimensions for topology derivation.  Only pairwise distinctness
# matters: the edge list is a function of which axis separates two corners.
TOPOLOGY_DIMS = (1.1, 1.3, 0.11)


def _axis_class(points: np.ndarray, i: int, j: int) -> str:
    delta = np.abs(points[i] - points[j])
    axis = int(np.argmax(delta))
    if axis == 1:                       # local Y is the height axis
        return "vertical"
    # OpenCV convention: local +Y points down, so the top face has negative Y.
    top = bool(points[i][1] < 0 and points[j][1] < 0)
    span = "width" if axis == 0 else "depth"
    return f"{'top' if top else 'base'}_{span}"


def build_topology(dims: tuple[float, float, float] = TOPOLOGY_DIMS) -> dict[str, Any]:
    """Derive the twelve physical edges from the canonical 3D keypoints.

    A physical edge is a corner pair whose 3D coordinates differ along exactly
    one axis.  Edge ids are the lexicographic order of ``(min, max)`` corner
    ids, which is deterministic and independent of iteration order.
    """
    points = np.asarray(APNP.make_pallet_keypoints_3d(*dims), dtype=np.float64)
    if points.shape != (9, 3):
        raise RuntimeError(f"unexpected canonical keypoint shape {points.shape}")
    corners = points[:N_CORNERS]

    pairs: list[tuple[int, int]] = []
    for i in range(N_CORNERS):
        for j in range(i + 1, N_CORNERS):
            delta = np.abs(corners[i] - corners[j])
            if int(np.count_nonzero(delta > 1e-9)) == 1:
                pairs.append((min(i, j), max(i, j)))
    pairs.sort()

    classes = [_axis_class(corners, i, j) for i, j in pairs]
    incidence = {c: [k for k, (i, j) in enumerate(pairs) if c in (i, j)]
                 for c in range(N_CORNERS)}

    # Top/base polarity partners: the same edge translated along the height
    # axis.  Derived by matching the two non-height coordinates of both
    # endpoints, so no index is written by hand.
    def _key(edge: tuple[int, int]) -> tuple:
        flat = []
        for c in edge:
            xz = (round(float(corners[c][0]), 9), round(float(corners[c][2]), 9))
            flat.append(xz)
        return tuple(sorted(flat))

    by_key: dict[tuple, list[int]] = {}
    for k, edge in enumerate(pairs):
        if classes[k] == "vertical":
            continue
        by_key.setdefault(_key(edge), []).append(k)
    polarity_pairs = []
    for members in by_key.values():
        if len(members) != 2:
            raise RuntimeError(f"polarity partner group of size {len(members)}")
        a, b = members
        top = a if classes[a].startswith("top") else b
        base = b if top == a else a
        polarity_pairs.append((top, base))
    polarity_pairs.sort()

    topology = {
        "dims": list(dims),
        "source": "annotate_pnp.make_pallet_keypoints_3d",
        "rule": "corner pair differing along exactly one local axis",
        "edges": [list(p) for p in pairs],
        "edge_classes": classes,
        "semantic_classes": list(SEMANTIC_CLASSES),
        "class_counts": {c: classes.count(c) for c in SEMANTIC_CLASSES},
        "corner_edge_incidence": {str(c): incidence[c] for c in range(N_CORNERS)},
        "polarity_pairs": [list(p) for p in polarity_pairs],
        "tau": TAU,
        "target_sigma_cells": TARGET_SIGMA_CELLS,
    }
    topology["topology_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in topology.items() if k != "dims"},
                   sort_keys=True).encode()).hexdigest()
    assert_topology(topology)
    return topology


def assert_topology(topology: dict[str, Any]) -> None:
    edges = [tuple(e) for e in topology["edges"]]
    classes = topology["edge_classes"]
    incidence = {int(k): v for k, v in topology["corner_edge_incidence"].items()}
    if len(edges) != N_EDGES:
        raise RuntimeError(f"{len(edges)} physical edges, expected {N_EDGES}")
    if len(set(edges)) != N_EDGES:
        raise RuntimeError("duplicate physical edge")
    for edge in edges:
        if len(edge) != 2 or edge[0] == edge[1]:
            raise RuntimeError(f"edge {edge} does not have two distinct endpoints")
    if edges != sorted(edges):
        raise RuntimeError("edge ids are not in lexicographic order")
    for corner in range(N_CORNERS):
        inc = incidence[corner]
        if len(inc) != INCIDENT_PER_CORNER:
            raise RuntimeError(f"corner {corner} has {len(inc)} incident edges")
        if len(set(inc)) != INCIDENT_PER_CORNER:
            raise RuntimeError(f"corner {corner} incidence is not unique")
        for k in inc:
            if corner not in edges[k]:
                raise RuntimeError(f"corner {corner} not an endpoint of edge {k}")
    counts = tuple(classes.count(c) for c in SEMANTIC_CLASSES)
    if counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(f"semantic class counts {counts}, expected {EXPECTED_CLASS_COUNTS}")
    pairs = [tuple(p) for p in topology["polarity_pairs"]]
    if len(pairs) != 4:
        raise RuntimeError(f"{len(pairs)} polarity pairs, expected 4")
    for top, base in pairs:
        if not classes[top].startswith("top") or not classes[base].startswith("base"):
            raise RuntimeError(f"polarity pair ({top},{base}) is not top/base")


def incidence_lists(topology: dict[str, Any], mode: str,
                    permutation: Optional[list[int]] = None) -> list[list[int]]:
    """Channels each corner is decoded from.

    ``O12`` indexes the three physically incident edges of a twelve-channel
    field stack.  ``O5`` indexes the same twelve-channel stack but only knows
    the semantic class of those three, so every edge sharing a class is
    admitted -- eight channels, and the admitted set is identical for all four
    corners of a face.  ``O5C`` is the same statement against a five-channel
    class map, which is what a five-class head actually predicts: the three
    class channels of the incident edges.  Both five-class variants can produce
    at most two distinct points for eight corners.
    """
    incidence = {int(k): v for k, v in topology["corner_edge_incidence"].items()}
    classes = topology["edge_classes"]
    out: list[list[int]] = []
    for corner in range(N_CORNERS):
        if mode == "O12":
            inc = list(incidence[corner])
        elif mode == "O5":
            wanted = {classes[k] for k in incidence[corner]}
            inc = [k for k in range(N_EDGES) if classes[k] in wanted]
        elif mode == "O5C":
            wanted = sorted({classes[k] for k in incidence[corner]},
                            key=SEMANTIC_CLASSES.index)
            out.append([SEMANTIC_CLASSES.index(name) for name in wanted])
            continue
        else:
            raise ValueError(f"unknown decoder mode {mode}")
        if permutation is not None:
            inc = [permutation[k] for k in inc]
        out.append(inc)
    return out


def class_distance_fields(edge_fields: np.ndarray, topology: dict[str, Any]
                          ) -> np.ndarray:
    """Collapse twelve edge distance fields into five semantic class fields.

    A class channel is the distance to the *nearest* edge of that class, which
    is what a rasterised union of same-class edges produces.  That makes this
    the strongest five-class decoder available, not a weakened one.
    """
    classes = topology["edge_classes"]
    fields = np.stack([
        edge_fields[[k for k in range(N_EDGES) if classes[k] == name]].min(axis=0)
        for name in SEMANTIC_CLASSES])
    return fields


def shuffled_permutation(seed: int = 1) -> list[int]:
    """Fixed edge-channel permutation for the shuffled-incidence control."""
    rng = np.random.default_rng(seed)
    while True:
        candidate = list(rng.permutation(N_EDGES))
        if any(int(candidate[k]) != k for k in range(N_EDGES)):
            return [int(v) for v in candidate]


# ============================================================================
# geometry
# ============================================================================
def segment_distance_field(grid: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact distance in grid cells from every cell centre to segment ``ab``."""
    ys, xs = np.mgrid[0:grid, 0:grid]
    ab = np.asarray(b, float) - np.asarray(a, float)
    denominator = float((ab ** 2).sum())
    if denominator <= 1e-12:
        return np.hypot(xs - a[0], ys - a[1]).astype(np.float64)
    t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / denominator, 0.0, 1.0)
    return np.hypot(xs - (a[0] + t * ab[0]), ys - (a[1] + t * ab[1]))


def clip_segment(p: np.ndarray, q: np.ndarray, width: float, height: float
                 ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky clip of a segment to ``[0,width] x [0,height]``.

    Endpoints are never clamped to the border: a segment that leaves the frame
    keeps only the part that is inside, so no false edge is created along the
    image boundary.
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for numerator, denominator in ((-d[0], p[0] - 0.0), (d[0], width - p[0]),
                                   (-d[1], p[1] - 0.0), (d[1], height - p[1])):
        if abs(numerator) < 1e-12:
            if denominator < 0:
                return None
            continue
        r = denominator / numerator
        if numerator < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return p + t0 * d, p + t1 * d


def clipped_edges_in_grid(points_image: list[Optional[np.ndarray]],
                          topology: dict[str, Any], width: float, height: float,
                          grid: int) -> tuple[list[Optional[tuple]], list[bool]]:
    """Per-edge clipped endpoints expressed in grid coordinates.

    Returns ``(segments, in_frame)`` where a ``None`` segment means the edge is
    entirely outside the image (or has an unavailable endpoint).
    """
    edges = [tuple(e) for e in topology["edges"]]
    segments: list[Optional[tuple]] = []
    in_frame: list[bool] = []
    for i, j in edges:
        a, b = points_image[i], points_image[j]
        if a is None or b is None:
            segments.append(None)
            in_frame.append(False)
            continue
        clipped = clip_segment(np.asarray(a, float), np.asarray(b, float), width, height)
        if clipped is None:
            segments.append(None)
            in_frame.append(False)
            continue
        scale = np.array([grid / float(width), grid / float(height)])
        segments.append((clipped[0] * scale, clipped[1] * scale))
        in_frame.append(True)
    return segments, in_frame


def distance_fields_from_segments(segments: list[Optional[tuple]], grid: int
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Stack of per-edge distance fields plus the per-edge availability flag."""
    cap = TARGET_SIGMA_CELLS * float(np.sqrt(-2.0 * np.log(PROB_FLOOR)))
    fields = np.full((len(segments), grid, grid), cap, dtype=np.float64)
    available = np.zeros(len(segments), dtype=bool)
    for k, segment in enumerate(segments):
        if segment is None:
            continue
        fields[k] = segment_distance_field(grid, segment[0], segment[1])
        available[k] = True
    return fields, available


def decode_corners(distance_fields: np.ndarray, incidence: list[list[int]],
                   grid: int, width: float, height: float, tau: float = TAU
                   ) -> list[list[float]]:
    """Line-only corner placement.

    ``score_i(x, y) = -exp(mean distance to the edges corner i is decoded from
    / tau)`` followed by a global argmax over the whole grid.  No corner
    heatmap and no top-K candidate set is involved: the only input is the edge
    distance fields.
    """
    points: list[list[float]] = []
    for corner in range(len(incidence)):
        mean = distance_fields[incidence[corner]].mean(axis=0)
        score = -np.exp(mean / tau)
        iy, ix = np.unravel_index(int(score.argmax()), score.shape)
        points.append([float(ix) * width / grid, float(iy) * height / grid])
    return points


# ============================================================================
# targets
# ============================================================================
def build_edge_targets(segments: list[Optional[tuple]], grid: int,
                       sigma_cells: float = TARGET_SIGMA_CELLS) -> np.ndarray:
    """One soft distance-field channel per physical edge.

    The field is ``exp(-d^2 / 2 sigma^2)`` of the rasterised clipped segment --
    the same quantity the five-class PPD target uses, with the class axis
    replaced by the physical-edge instance axis.  Every edge keeps its target
    whether or not it is visible: the decoder needs all three incident edges of
    a corner, including occluded ones, so an amodal target is required.
    """
    import cv2

    targets = np.zeros((len(segments), grid, grid), dtype=np.float32)
    for k, segment in enumerate(segments):
        if segment is None:
            continue
        raster = np.zeros((grid, grid), np.uint8)
        a, b = segment
        cv2.line(raster, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), 1, 1)
        if raster.sum() == 0:
            continue
        distance = cv2.distanceTransform(1 - raster, cv2.DIST_L2, 3)
        targets[k] = np.exp(-(distance ** 2) / (2.0 * sigma_cells ** 2))
    return targets


def distance_from_probability(probability: np.ndarray,
                              sigma_cells: float = TARGET_SIGMA_CELLS,
                              floor: float = PROB_FLOOR) -> np.ndarray:
    """Invert the soft distance field back to a distance in grid cells.

    ``exp(-d^2/2s^2) = p`` gives ``d = s sqrt(-2 ln p)``.  The floor caps the
    distance far from any edge, where the target carries no information anyway;
    the cap is a constant, so it never changes which cell wins the argmax.
    """
    clipped = np.clip(probability.astype(np.float64), floor, 1.0)
    return sigma_cells * np.sqrt(np.maximum(-2.0 * np.log(clipped), 0.0))
