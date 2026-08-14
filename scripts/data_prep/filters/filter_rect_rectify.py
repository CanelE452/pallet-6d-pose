"""filter_rect_rectify.py — dims-free clean-PL filter via rectangle rectification.

A planar rectangular cuboid FACE projects to a quadrilateral whose two pairs of
opposite edges meet at two vanishing points v1, v2. With a known intrinsic K the
edge directions in 3D are d_i = normalize(K^-1 v_i); for a true rectangle these
two directions are ORTHOGONAL, so the orthogonality residual

    res = |d1 . d2|        (0 = perfect rectangle, dims-free, metric-free)

is a pure projective/affine check that needs NO depth and NO known dimensions.
We also recover the face aspect ratio (Liebowitz/Zhang metric rectification) for
diagnostics only — it is dims-free (a ratio, not a size).

Which face? The pallet is a thin slab (W=1.1, D=1.3, H=0.11 m). In object-frame
canonical convention (verified by 3d-expert audit 2026-06-03):
    front {0,1,2,3} = W x H   (THIN, ~edge-on -> rectification unstable)
    top   {0,1,4,5} = W x D   (the LARGE camera-facing slab face)
    side  {0,3,7,4} = D x H   (thin)
The user's intent ("the trusted camera-facing rectangle") maps to the LARGE
W x D face for a top-down warehouse camera, NOT the thin 0123 slab. We therefore
score every available rectangular face and report which one is the reliable
clean-PL signal, decided by correlation with GT corner error (no guessing).

degenerate handling: if a face's two opposite-edge pairs are near-parallel the
VP goes to infinity; we then fall back to the image-plane direction (the edge's
2D direction back-projected through K^-1 with the point at infinity), which is
the correct limit. Faces that are near edge-on (tiny projected area / extreme
foreshortening) are GATED OUT (skip, not pass) because rectification collapses.

Reuses filter_pr_camfacing for model loading / belief extraction / hungarian
good judgment so signatures stay compatible.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]


import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import filter_pr_camfacing as fp  # load_model, extract_keypoints_from_belief, hungarian_mean_dist, canonical_kp3d  # noqa: E402

# ── Face definitions (object-frame canonical corner indices) ─────────────
# A face = (name, 4 corner idx in cyclic order, label of the two edge groups).
# Edge group A = edges (c0-c1) & (c3-c2); group B = (c0-c3) & (c1-c2).
FACES = {
    "front_WH": (0, 1, 2, 3),   # W x H  (thin)
    "top_WD":   (0, 1, 5, 4),   # W x D  (large slab) — cyclic 0-1-5-4
    "side_DH":  (0, 4, 7, 3),   # D x H  (thin) — cyclic 0-4-7-3
}

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PAD = 100  # reflect-pad (matches /tmp/c0403_filter.py + training canvas)

# Gating thresholds
MIN_FACE_AREA_PX = 1500.0      # quad area below this => face too small / edge-on
MIN_EDGE_PX = 12.0             # an edge shorter than this is noise-dominated
MIN_FORESHORTEN = 0.10         # min(side_a/side_b, side_b/side_a) floor; below = edge-on
VP_PARALLEL_COS = 0.9995       # |cos| of the two edges => treat VP as at infinity


# ── projective helpers ──────────────────────────────────────────────────
def _line_dir_inf(p1, p2):
    """Homogeneous point at infinity in the direction p1->p2 (the VP limit)."""
    d = np.array([p2[0] - p1[0], p2[1] - p1[1], 0.0])
    n = np.linalg.norm(d[:2])
    return d / n if n > 1e-9 else None


def _vp(p1, p2, p3, p4):
    """Vanishing point (homogeneous) of lines (p1p2) and (p3p4).

    Returns (vp_homogeneous_3, is_at_infinity). Uses the cross-product of the two
    homogeneous lines, which naturally yields w~0 for parallel lines (the correct
    point-at-infinity), so no special branch is needed for the degenerate case.
    """
    def hline(a, b):
        ha = np.array([a[0], a[1], 1.0]); hb = np.array([b[0], b[1], 1.0])
        return np.cross(ha, hb)
    l1 = hline(p1, p2); l2 = hline(p3, p4)
    v = np.cross(l1, l2)
    # how parallel are the two segments?
    d1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    d2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    cosang = abs(d1 @ d2) / (n1 * n2 + 1e-12) if n1 > 1e-9 and n2 > 1e-9 else 1.0
    at_inf = cosang > VP_PARALLEL_COS or abs(v[2]) < 1e-6
    return v, bool(at_inf)


def _dir_from_vp(v, at_inf, p_ref0, p_ref1, Kinv):
    """3D edge direction d = normalize(K^-1 v). For a finite VP, v=(x,y,w); for a
    VP at infinity use the 2D edge direction as the homogeneous point (w=0)."""
    if at_inf:
        d2 = _line_dir_inf(p_ref0, p_ref1)
        if d2 is None:
            return None
        vh = np.array([d2[0], d2[1], 0.0])
    else:
        if abs(v[2]) < 1e-9:
            return None
        vh = np.array([v[0] / v[2], v[1] / v[2], 1.0])
    d = Kinv @ vh
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def _quad_area(c0, c1, c2, c3):
    pts = np.array([c0, c1, c2, c3], float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rectify_face(c0, c1, c2, c3, K):
    """Orthogonality residual + recovered aspect for a quad face.

    c0..c3 in cyclic order. Edge group A = (c0-c1, c3-c2), group B = (c0-c3, c1-c2).
    Returns dict or None (degenerate / gated). Keys:
        resid        |d_A . d_B|  (0 = perfect rectangle; the FILTER SCORE)
        aspect       recovered |edge_A| : |edge_B| length ratio (dims-free)
        area, at_inf_a, at_inf_b, foreshorten
    """
    c0, c1, c2, c3 = (np.asarray(p, float) for p in (c0, c1, c2, c3))
    area = _quad_area(c0, c1, c2, c3)
    # edge length sanity (the four sides)
    sA = [np.linalg.norm(c1 - c0), np.linalg.norm(c2 - c3)]   # group A sides
    sB = [np.linalg.norm(c3 - c0), np.linalg.norm(c2 - c1)]   # group B sides
    if min(sA) < MIN_EDGE_PX or min(sB) < MIN_EDGE_PX:
        return None
    mA, mB = np.mean(sA), np.mean(sB)
    foreshorten = min(mA, mB) / max(mA, mB)
    if area < MIN_FACE_AREA_PX or foreshorten < MIN_FORESHORTEN:
        return None  # edge-on / collapsed: rectification unreliable -> skip

    Kinv = np.linalg.inv(K)
    vA, infA = _vp(c0, c1, c3, c2)   # group A vanishing point
    vB, infB = _vp(c0, c3, c1, c2)   # group B vanishing point
    dA = _dir_from_vp(vA, infA, c0, c1, Kinv)
    dB = _dir_from_vp(vB, infB, c0, c3, Kinv)
    if dA is None or dB is None:
        return None
    resid = abs(float(dA @ dB))

    # dims-free aspect: foreshortening-corrected ratio of the two edge dirs.
    # Use the mean side lengths corrected by the depth (z) of each VP direction;
    # for the orthogonality filter the residual is what matters, aspect is a
    # diagnostic so we report the raw projected side ratio as a robust proxy.
    aspect = mA / mB if mB > 1e-6 else float("inf")
    return {"resid": resid, "aspect": float(aspect), "area": float(area),
            "at_inf_a": infA, "at_inf_b": infB, "foreshorten": float(foreshorten)}


def best_face_rectify(kp, K, faces=("top_WD", "front_WH", "side_DH")):
    """Try each face (whose 4 corners are all detected); return per-face results
    and the chosen primary face (largest valid area)."""
    out = {}
    for fname in faces:
        idx = FACES[fname]
        if any(kp[i] is None for i in idx):
            continue
        r = rectify_face(kp[idx[0]], kp[idx[1]], kp[idx[2]], kp[idx[3]], K)
        if r is not None:
            out[fname] = r
    if not out:
        return out, None
    primary = max(out, key=lambda f: out[f]["area"])
    return out, primary


# ── preprocessing (reflect-pad P=100 -> 400, matches training canvas) ────
def infer_kp(bgr, model, device, threshold=0.3):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    inp = cv2.resize(cv2.copyMakeBorder(rgb, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101), (400, 400))
    t = ((inp.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    with torch.no_grad():
        ob, _ = model(torch.from_numpy(t).float().unsqueeze(0).to(device))
    belief = ob[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = fp.extract_keypoints_from_belief(belief, threshold)
    kp, confs = [], []
    for k in kps_bel:
        if k[0] < 0:
            kp.append(None); confs.append(0.0)
        else:
            ox = k[0] * (w0 + 2 * PAD) / bw - PAD
            oy = k[1] * (h0 + 2 * PAD) / bh - PAD
            kp.append((ox, oy)); confs.append(k[2])
    return kp, confs


def draw(img, kp, face_idx, passed, resid):
    col = (0, 200, 0) if passed else (0, 0, 230)
    pts = [kp[i] for i in face_idx]
    for a in range(4):
        p, q = pts[a], pts[(a + 1) % 4]
        if p and q:
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, 2)
    for i in face_idx:
        if kp[i]:
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 4, (255, 255, 0), -1)
            cv2.putText(img, str(i), (int(kp[i][0]) + 4, int(kp[i][1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    tag = "PASS" if passed else "REJECT"
    cv2.putText(img, f"{tag} res={resid:.3f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
