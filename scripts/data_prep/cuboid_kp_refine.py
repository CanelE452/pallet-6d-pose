"""Cuboid keypoint refinement — K-free, shape-prior plug-in for DOPE output.

Refine the 9 DOPE keypoints of a flat pallet cuboid using ONLY 2D projective
geometry (no camera intrinsics, no PnP, no known dimensions). The pallet is a
thin slab, so its three parallel-edge groups are NOT equally usable:

    convention: camera-facing v6 (annotate_pnp.make_pallet_keypoints_3d)
      faces: near {0,1,2,3}  far {4,5,6,7}
             top  {0,1,4,5}  bottom {2,3,6,7}
      edge groups (parallel in 3D -> one vanishing point each in 2D):
        WIDTH  : (0,1)(3,2)(4,5)(7,6)     large  -> stable VP
        DEPTH  : (0,4)(1,5)(2,6)(3,7)     large  -> stable VP
        HEIGHT : (0,3)(1,2)(4,7)(5,6)     THIN   -> VP degenerate, DO NOT fit

(verified on wood/selected real detections 2026-06-18: HEIGHT vanishing-point
spread was 600-11000 px even on clean frames because the slab thickness edges
are ~40 px and near-parallel; WIDTH/DEPTH VP spread was 90-500 px.)

Method (per face, then both faces share the corrected structure):
  1. Fit a robust vanishing point for WIDTH and for DEPTH (SVD of the edge
     lines + one IRLS reweight to down-weight a single noisy edge). Parallel
     edges yield a point-at-infinity, handled naturally in homogeneous coords.
  2. Each corner lies on exactly one WIDTH edge and one DEPTH edge. Re-derive
     its position as the intersection of those two edge LINES, where each edge
     line is anchored to the edge's detected MIDPOINT (keeps position from the
     data) but its DIRECTION is snapped to the consensus vanishing point (the
     shape constraint). HEIGHT is never used to move a corner, so the thin
     axis cannot inject error.
  3. Trust region: a corner moves toward its refined location only in
     proportion to how far it was (blend), and never more than `max_move_frac`
     of the cuboid diagonal. Points already consistent barely move.
  4. centroid (kp 8) = mean of the space-diagonal intersections (projective
     invariant), falling back to the detected kp8 / mean of corners.

Gate (return applied=False, original kept) when refinement is unreliable:
  - fewer than `min_corners` of the 8 cuboid corners detected
  - a group has fewer than 2 usable edges (cannot fit a VP)
  - degenerate thinness: slab thickness collapses (height edges ~0 vs span)
  - any refined move exceeds `reject_move_frac` * diagonal (detection broke)

Public API:
    refine_keypoints(kps, img_size=None, **cfg) -> RefineResult
        kps      : list/seq of 9 items, each (x, y) or None
        returns  : RefineResult(kps, applied, confidence, info)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# camera-facing v6 edge groups
WIDTH_EDGES = [(0, 1), (3, 2), (4, 5), (7, 6)]
DEPTH_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
HEIGHT_EDGES = [(0, 3), (1, 2), (4, 7), (5, 6)]
SPACE_DIAGS = [(0, 6), (1, 7), (2, 4), (3, 5)]

# each corner -> (its WIDTH edge, its DEPTH edge)
_W_OF = {0: (0, 1), 1: (0, 1), 3: (3, 2), 2: (3, 2),
         4: (4, 5), 5: (4, 5), 7: (7, 6), 6: (7, 6)}
_D_OF = {0: (0, 4), 4: (0, 4), 1: (1, 5), 5: (1, 5),
         2: (2, 6), 6: (2, 6), 3: (3, 7), 7: (3, 7)}


@dataclass
class RefineResult:
    kps: List[Optional[Tuple[float, float]]]   # 9 refined keypoints (None preserved)
    applied: bool                               # was any refinement applied?
    confidence: float                           # 0..1 shape-prior fit confidence
    info: dict = field(default_factory=dict)    # diagnostics (moves, vp resid, gate)


# ── homogeneous helpers ──────────────────────────────────────────────────
def _h(p):
    return np.array([p[0], p[1], 1.0])


def _line(a, b):
    return np.cross(_h(a), _h(b))


def _inter(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def _robust_vp(kp, edges):
    """Vanishing point of a parallel-edge group via SVD + one IRLS reweight.

    Returns (vp_homogeneous (3,), n_edges) or (None, n_edges<2). The result may
    be a point at infinity (w ~ 0); that is the correct limit for parallel
    edges and downstream intersection handles it."""
    segs = [(a, b) for a, b in edges if kp[a] is not None and kp[b] is not None]
    if len(segs) < 2:
        return None, len(segs)
    lines = np.array([_line(kp[a], kp[b]) for a, b in segs], dtype=np.float64)
    # normalize each line by its (a,b) direction magnitude for numeric balance
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12
    L = lines / norms
    _, _, Vt = np.linalg.svd(L)
    vp = Vt[-1]
    if len(segs) >= 3:
        # one IRLS step: down-weight the edge whose line is farthest from vp
        resid = np.abs(L @ vp)
        med = np.median(resid) + 1e-9
        w = 1.0 / (1.0 + (resid / med) ** 2)
        Lw = L * w[:, None]
        _, _, Vt = np.linalg.svd(Lw)
        vp = Vt[-1]
    return vp, len(segs)


def _edge_line_via_vp(kp, edge, vp):
    """Line through the edge's detected midpoint constrained to pass the VP.

    Anchors position to the data (midpoint) and direction to the shape (VP).
    If the VP is at infinity, the line keeps the edge's own 2D direction."""
    a, b = edge
    mid = (np.asarray(kp[a], float) + np.asarray(kp[b], float)) / 2.0
    if abs(vp[2]) < 1e-9:
        # VP at infinity: direction = (vp[0], vp[1]); line through mid with that dir
        return np.cross(_h(mid), np.array([vp[0], vp[1], 0.0]))
    vxy = np.array([vp[0] / vp[2], vp[1] / vp[2]])
    return _line(mid, vxy)


# ── main ───────────────────────────────────────────────────────────────────
def refine_keypoints(
    kps: Sequence[Optional[Tuple[float, float]]],
    img_size: Optional[Tuple[int, int]] = None,
    min_corners: int = 6,
    blend: float = 1.0,
    max_move_frac: float = 0.06,
    reject_move_frac: float = 0.25,
    min_thin_ratio: float = 0.04,
) -> RefineResult:
    """Refine 9 cuboid keypoints toward the flat-cuboid shape prior (K-free).

    Args:
        kps:        9 items, each (x, y) or None.
        img_size:   (W, H) optional, only for diagnostics.
        min_corners: min detected cuboid corners (of 8) to attempt refinement.
        blend:      0..1 fraction of the (clipped) refine move to apply.
                    1.0 = full clipped move; lower = gentler.
        max_move_frac:  a corner moves at most this * cuboid_diagonal (trust
                    region; well-fitting corners move far less than this).
        reject_move_frac: if any corner *wants* to move more than this *
                    diagonal, the detection is inconsistent with the prior ->
                    GATE OUT (keep original, applied=False).
        min_thin_ratio: min (mean height-edge len)/(cuboid diagonal). Below
                    this the slab has collapsed (edge-on) -> GATE OUT.

    Returns:
        RefineResult.
    """
    kp = [None if p is None else (float(p[0]), float(p[1])) for p in kps]
    corners_det = [i for i in range(8) if kp[i] is not None]
    info = {"n_corners": len(corners_det), "gate": None,
            "moves": {}, "max_move": 0.0}

    # diagonal scale (for normalization / trust region)
    pts = np.array([kp[i] for i in corners_det], float)
    if len(pts) >= 2:
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    else:
        diag = 0.0
    info["diag"] = diag

    def fail(reason):
        info["gate"] = reason
        return RefineResult([p for p in kp], False, 0.0, info)

    if len(corners_det) < min_corners:
        return fail(f"too_few_corners({len(corners_det)}<{min_corners})")
    if diag < 1e-3:
        return fail("degenerate_diag")

    # degenerate-thinness gate (edge-on slab collapse)
    h_lens = [np.linalg.norm(np.array(kp[a]) - np.array(kp[b]))
              for a, b in HEIGHT_EDGES if kp[a] is not None and kp[b] is not None]
    if h_lens:
        thin_ratio = float(np.mean(h_lens)) / diag
        info["thin_ratio"] = thin_ratio
        if thin_ratio < min_thin_ratio:
            return fail(f"edge_on_collapse(thin={thin_ratio:.3f})")

    # robust vanishing points for the two STABLE groups
    vpW, nW = _robust_vp(kp, WIDTH_EDGES)
    vpD, nD = _robust_vp(kp, DEPTH_EDGES)
    info["n_edges_W"], info["n_edges_D"] = nW, nD
    if vpW is None or vpD is None:
        return fail(f"insufficient_edges(W={nW},D={nD})")

    # refine each corner = intersection of its (VP-snapped) W and D edge lines
    refined = {}
    want_moves = []
    for i in range(8):
        if kp[i] is None:
            continue
        we, de = _W_OF[i], _D_OF[i]
        if (kp[we[0]] is None or kp[we[1]] is None or
                kp[de[0]] is None or kp[de[1]] is None):
            refined[i] = np.array(kp[i], float)   # can't reconstruct -> keep
            continue
        lw = _edge_line_via_vp(kp, we, vpW)
        ld = _edge_line_via_vp(kp, de, vpD)
        p = _inter(lw, ld)
        if p is None or not np.all(np.isfinite(p)):
            refined[i] = np.array(kp[i], float)
            continue
        refined[i] = p
        want_moves.append(np.linalg.norm(p - np.array(kp[i], float)))

    if not want_moves:
        return fail("no_reconstructable_corner")

    # GATE: a large *wanted* move means the detection contradicts the prior
    # (broken detection) -> do not snap it to a plausible rectangle.
    max_want = float(max(want_moves))
    info["max_want_move"] = max_want
    if max_want > reject_move_frac * diag:
        return fail(f"move_exceeds_reject({max_want:.0f}>"
                    f"{reject_move_frac * diag:.0f})")

    # apply with trust region (clip per-corner move, then blend)
    cap = max_move_frac * diag
    out = [None] * 9
    for i in range(8):
        if kp[i] is None:
            continue
        orig = np.array(kp[i], float)
        if i not in refined:
            out[i] = (orig[0], orig[1])
            continue
        delta = refined[i] - orig
        n = np.linalg.norm(delta)
        if n > cap:
            delta = delta * (cap / n)
        delta = delta * blend
        new = orig + delta
        out[i] = (float(new[0]), float(new[1]))
        m = float(np.linalg.norm(new - orig))
        info["moves"][i] = m
        info["max_move"] = max(info["max_move"], m)

    # centroid = mean of space-diagonal intersections (projective invariant)
    diag_segs = [(kp[a], kp[b]) for a, b in SPACE_DIAGS
                 if kp[a] is not None and kp[b] is not None]
    ctr = None
    if len(diag_segs) >= 2:
        inters = []
        for x in range(len(diag_segs)):
            for y in range(x + 1, len(diag_segs)):
                v = _inter(_line(*diag_segs[x]), _line(*diag_segs[y]))
                if v is not None and np.all(np.isfinite(v)):
                    inters.append(v)
        if inters:
            ctr = np.mean(inters, axis=0)
    if ctr is None:
        valid = [out[i] for i in range(8) if out[i] is not None]
        if valid:
            ctr = np.mean(np.array(valid, float), axis=0)
    if ctr is not None:
        if kp[8] is not None:
            o8 = np.array(kp[8], float)
            d8 = ctr - o8
            n8 = np.linalg.norm(d8)
            if n8 > cap:
                d8 = d8 * (cap / n8)
            new8 = o8 + d8 * blend
            out[8] = (float(new8[0]), float(new8[1]))
            info["moves"][8] = float(np.linalg.norm(new8 - o8))
            info["max_move"] = max(info["max_move"], info["moves"][8])
        else:
            out[8] = (float(ctr[0]), float(ctr[1]))

    # confidence: low max_want (already-consistent) + both groups well-supported
    fit = max(0.0, 1.0 - max_want / (reject_move_frac * diag))
    support = min(nW, nD) / 4.0
    conf = float(0.5 * fit + 0.5 * support)
    info["confidence_fit"] = fit
    info["confidence_support"] = support

    return RefineResult(out, True, conf, info)
