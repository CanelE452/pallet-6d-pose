"""filter_pr_camfacing.py — Stage-1 P/R screening for self-training pseudo-label filters.

Compares, in ONE framework, on GT-annotated real eval sets:

  Controls:
    none    : no filter (PnP/detection succeeds)
    conf    : confidence gate only (min belief peak > 0.5)

  Existing PnP-based filters (SQPnP, correct dims, canonical 3D order):
    ransac     : RANSAC subset consensus (n_iter=50, subset=5, tau=5px, c>=6)
    ransac_loo : RANSAC consensus AND leave-one-out PnP stability (tau_LOO)
    cf_strict  : canonical B AND C AND D (all structural priors)

  New 2D-geometric filters (NO PnP — projective image-space invariants):
    diag    : 4 spatial diagonals (0-6,1-7,2-4,3-5) intersect near centroid(8)
    topbot  : image-y of {0,1,4,5} above {2,3,6,7}
    ratio   : opposite parallel edges have consistent length (perspective-tolerant)
    fullkp  : all 9 keypoints detected (strict pre-filter)
    combo   : diag AND topbot AND ratio AND (>=8 kp)

"good" judgment (order-free, convention-agnostic):
    Hungarian-match predicted 8 corners to GT projected_cuboid 8 corners by 2D
    distance; mean matched pixel distance < good_thresh_px  =>  good pseudo-label.

Corner index mapping (object-frame canonical, verified by 3d-expert audit
2026-06-03; HEIGHT edge shortest 98-100% on these very sets):
    0 LTF 1 RTF 2 RBF 3 LBF (front {0,1,2,3})
    4 LTK 5 RTK 6 RBK 7 LBK (rear  {4,5,6,7})   8 centroid
    Top {0,1,4,5}  Bottom {2,3,6,7}
    width  edges 0-1,3-2,4-5,7-6 ; depth 0-4,1-5,2-6,3-7 ; height 0-3,1-2,4-7,5-6
    spatial diagonals (opposite corners) 0-6,1-7,2-4,3-5
    face-diagonal midpoint pairs FACE_DIAG_PAIRS (canonical_filters)

Usage:
    python scripts/data_prep/eval/filter_pr_camfacing.py \
        --weights weights/dope_cropaug_ft_s2/net_epoch_0180.pth --tag s2

Output (data/pallet/eval_results/filter_pr_camfacing/):
    summary_{tag}.json / .csv   per-filter and per-dataset P/R/F1/n_pass
    per_frame_{tag}.json        per-frame keypoints / good / filter pass-fail
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PREP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "self_training"))
sys.path.insert(0, DATA_PREP)

from models import DopeNetwork  # noqa: E402
import canonical_filters as cf  # noqa: E402

# ── Corner index constants (object-frame canonical) ───────────────────
SPACE_DIAG = [(0, 6), (1, 7), (2, 4), (3, 5)]   # opposite cuboid corners
TOP_IDX = [0, 1, 4, 5]
BOT_IDX = [2, 3, 6, 7]
# opposite parallel edge pairs (same physical length in 3D)
WIDTH_EDGES = [(0, 1), (3, 2), (4, 5), (7, 6)]
DEPTH_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
HEIGHT_EDGES = [(0, 3), (1, 2), (4, 7), (5, 6)]
# Within each parallel group, edges that lie on the SAME physical face share a
# vanishing point, so their 2D-length ratio is tightly bounded by perspective
# (foreshortening is continuous along the face).  Cross-face edges (front vs
# rear) can differ much more.  filt_edgelen compares co-planar opposite edges.
#   W group:  front {0-1,3-2}, rear {4-5,7-6}
#   D group:  top  {0-4,3-7}, bottom {1-5,2-6}
#   H group:  left {0-3,4-7}, right {1-2,5-6}
_FACE_EDGE_PAIRS = {
    "W": [((0, 1), (3, 2)), ((4, 5), (7, 6))],
    "D": [((0, 4), (3, 7)), ((1, 5), (2, 6))],
    "H": [((0, 3), (4, 7)), ((1, 2), (5, 6))],
}

CONF_THRESHOLD = 0.5
DEFAULT_GOOD_PX = 10.0

# (gt_dir, image_dir or None for same-as-gt). forklift GT lives in gt_manual/
# while its images live in a sibling rgb/ folder.
DEFAULT_SETS = {
    "outside": (os.path.join(ROOT, "data", "_eval_sets", "outside_combined"), None),
    "night": (os.path.join(ROOT, "data", "_eval_sets", "night_combined"), None),
}
FORKLIFT_GT = os.path.join(ROOT, "data", "outside",
                           "forklift_raw_20260528_163408", "gt_manual")
FORKLIFT_IMG = os.path.join(ROOT, "data", "outside",
                            "forklift_raw_20260528_163408", "rgb")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "pallet", "eval_results",
                              "filter_pr_camfacing")


# ── 3D model points (canonical order, matches Isaac SDG / GT) ─────────
def canonical_kp3d(width, depth, height):
    W, H, D = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-W, -H, +D], [+W, -H, +D], [+W, +H, +D], [-W, +H, +D],   # front 0-3
        [-W, -H, -D], [+W, -H, -D], [+W, +H, -D], [-W, +H, -D],   # rear  4-7
    ], dtype=np.float64)
    return np.vstack([corners, corners.mean(0, keepdims=True)])


# ── DOPE inference ────────────────────────────────────────────────────
def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def extract_keypoints_from_belief(belief_maps, threshold=0.3):
    OFFSET, RAN = 0.4395, 5
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        if bmap.max() < threshold:
            keypoints.append((-1, -1, float(bmap.max())))
            continue
        sm = gaussian_filter(bmap, sigma=2)
        p = 1
        pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
        pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
        pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
        pd = np.zeros_like(sm); pd[:, :-p] = sm[:, p:]
        peaks = ((sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) &
                 (sm > threshold))
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            keypoints.append((-1, -1, float(bmap.max())))
            continue
        best = int(np.argmax([bmap[y, x] for y, x in zip(ys, xs)]))
        px, py = int(xs[best]), int(ys[best])
        y0, y1 = max(0, py - RAN), min(bmap.shape[0], py + RAN + 1)
        x0, x1 = max(0, px - RAN), min(bmap.shape[1], px + RAN + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() > 0:
            xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            wx = float(np.average(xg, weights=patch)) + OFFSET
            wy = float(np.average(yg, weights=patch)) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy, float(bmap.max())))
    return keypoints


# ── order-free good judgment ──────────────────────────────────────────
def hungarian_mean_dist(pred8, gt8):
    """Mean matched 2D distance between predicted and GT 8 corners.

    pred8: (8,2) with NaN for missing corners (skipped). gt8: (8,2).
    Returns (mean_dist, n_matched). Requires >=6 predicted corners.
    """
    valid = ~np.isnan(pred8[:, 0])
    if valid.sum() < 6:
        return float("inf"), int(valid.sum())
    P = pred8[valid]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].mean()), len(ri)


# ── 2D geometric filters (no PnP) ─────────────────────────────────────
def _line_intersect(p1, p2, p3, p4):
    A = np.array([[p2[0] - p1[0], -(p4[0] - p3[0])],
                  [p2[1] - p1[1], -(p4[1] - p3[1])]])
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(A, b)
    return p1 + t[0] * (p2 - p1)


def filt_diag(kp, tau=0.05):
    """Spatial diagonals intersect near centroid(8). Scale-free (norm by diag len)."""
    need = set(sum([list(d) for d in SPACE_DIAG], [])) | {8}
    if any(kp[i] is None for i in need):
        return False, float("inf")
    pts = {i: np.asarray(kp[i], float) for i in need}
    inter = []
    for a in range(len(SPACE_DIAG)):
        for b in range(a + 1, len(SPACE_DIAG)):
            i, j = SPACE_DIAG[a]; k, l = SPACE_DIAG[b]
            ip = _line_intersect(pts[i], pts[j], pts[k], pts[l])
            if ip is not None:
                inter.append(ip)
    if not inter:
        return False, float("inf")
    mp = np.mean(inter, axis=0)
    diag_len = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in SPACE_DIAG])
    if diag_len < 1e-6:
        return False, float("inf")
    score = np.linalg.norm(mp - pts[8]) / diag_len
    return score < tau, float(score)


def filt_topbot(kp, margin_frac=0.0):
    """Image-y of TOP corners above BOTTOM corners (mean), per detected pairs."""
    pairs = [(0, 3), (1, 2), (4, 7), (5, 6)]  # (top, bottom) of each height edge
    diffs = []
    for tp, bt in pairs:
        if kp[tp] is not None and kp[bt] is not None:
            diffs.append(kp[bt][1] - kp[tp][1])  # bottom_y - top_y > 0 expected
    if len(diffs) < 2:
        return False, float("inf")
    # all detected height-edges must respect ordering
    ok = all(d > 0 for d in diffs)
    score = float(np.mean(diffs))
    return ok, score


def filt_ratio(kp, tau=0.35):
    """Opposite parallel edges consistent length (perspective-tolerant).

    For width edges {0-1,3-2,4-5,7-6} and depth edges {0-4,...}, the max/min
    ratio of detected edge lengths within each parallel group should be modest.
    """
    def group_ratio(edges):
        lens = []
        for a, b in edges:
            if kp[a] is not None and kp[b] is not None:
                lens.append(np.linalg.norm(np.asarray(kp[a], float) -
                                           np.asarray(kp[b], float)))
        if len(lens) < 2:
            return None
        lens = np.array(lens)
        return float(lens.max() / max(lens.min(), 1e-6) - 1.0)  # 0 = identical

    rw = group_ratio(WIDTH_EDGES)
    rd = group_ratio(DEPTH_EDGES)
    vals = [v for v in (rw, rd) if v is not None]
    if not vals:
        return False, float("inf")
    score = float(max(vals))
    return score < tau, score


def _edge_len(kp, a, b):
    if kp[a] is None or kp[b] is None:
        return None
    return float(np.linalg.norm(np.asarray(kp[a], float) - np.asarray(kp[b], float)))


def filt_edgelen(kp, tau=0.55, tau_h=1.20, h_min_px=4.0):
    """PnP-free cuboid edge-length consistency (the 12 real edges only).

    The 12 edges are grouped into 3 parallel groups (W/H/D). For a true cuboid
    each group is 4 parallel 3D segments; their 2D projections are NOT equal
    (perspective foreshortening), but two edges lying on the SAME physical face
    share a vanishing point, so their length ratio is tightly bounded. We score
    each group by the worst co-planar opposite-edge pair using the symmetric
    log-ratio |log(L1/L2)|, then take the max over groups. depth-collapse (back
    or bottom corners squashed toward the front) blows up the D/H pairs while
    leaving the front face intact, so it is rejected; correct perspective boxes
    pass.

    The HEIGHT group is special: the pallet is very thin (H=0.11m), so H edges
    are short (a few px) and noise-dominated. We therefore (a) require a larger
    tolerance tau_h for H, and (b) skip any H pair whose edges are below
    h_min_px (pure noise — not informative either way). H never causes a reject
    on its own unless it is grossly inconsistent (tau_h).

    Args:
        kp: length-9 list of (x,y) or None (camera-facing 0123 convention).
        tau: max |log-ratio| allowed for a W/D co-planar opposite-edge pair.
        tau_h: looser tolerance for the (thin, noisy) H group.
        h_min_px: skip H pairs whose edges are shorter than this (noise).
    Returns (bool pass, float score). score = max normalized log-ratio
    (0 = perfectly consistent); inf if too few edges to judge.
    """
    worst = 0.0
    judged = False
    for grp, pairs in _FACE_EDGE_PAIRS.items():
        thr = tau_h if grp == "H" else tau
        for (e1, e2) in pairs:
            l1 = _edge_len(kp, *e1)
            l2 = _edge_len(kp, *e2)
            if l1 is None or l2 is None:
                continue
            if grp == "H" and (l1 < h_min_px or l2 < h_min_px):
                continue  # both endpoints near-collapsed: H too short to trust
            lo, hi = sorted((max(l1, 1e-6), max(l2, 1e-6)))
            lr = float(np.log(hi / lo))           # >= 0, scale-free, symmetric
            judged = True
            # normalize each group's log-ratio by its own tolerance so the max
            # over groups is comparable (score < 1.0  <=>  all pairs within tol)
            worst = max(worst, lr / thr)
    if not judged:
        return False, float("inf")
    return worst < 1.0, float(worst)


def filt_fullkp(kp):
    n = sum(1 for i in range(9) if kp[i] is not None)
    return n == 9, n


def filt_combo(kp):
    n_det = sum(1 for i in range(8) if kp[i] is not None)
    d_ok, _ = filt_diag(kp)
    t_ok, _ = filt_topbot(kp)
    r_ok, _ = filt_ratio(kp)
    return bool(d_ok and t_ok and r_ok and n_det >= 8)


# ── PnP-based filters ─────────────────────────────────────────────────
def sqpnp(kp_list, kp3d, K, dist, min_pts=6):
    obj, img = [], []
    for i in range(8):
        if kp_list[i] is not None:
            obj.append(kp3d[i]); img.append([float(kp_list[i][0]), float(kp_list[i][1])])
    if len(obj) < min_pts:
        return False, None, None, len(obj)
    obj = np.asarray(obj, float).reshape(-1, 1, 3)
    img = np.asarray(img, float).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return False, None, None, len(obj)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    except cv2.error:
        pass
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    if t[2] < 0:
        t, R = -t, -R
    return True, R, t, len(obj) // 1


def ransac_consensus(kp_list, kp3d, K, dist, n_iter=50, subset=5,
                     tau=5.0, seed=0):
    """Random-subset PnP, return best full-set inlier consensus count."""
    det = [i for i in range(8) if kp_list[i] is not None]
    if len(det) < subset:
        return 0, None, None
    d2 = np.array([[float(kp_list[i][0]), float(kp_list[i][1])] for i in det])
    d3 = kp3d[det].astype(np.float64)
    rng = np.random.default_rng(seed)
    best_c, best_R, best_t = -1, None, None
    for _ in range(n_iter):
        sel = (np.arange(len(det)) if len(det) == subset
               else rng.choice(len(det), subset, replace=False))
        try:
            ok, rvec, tvec = cv2.solvePnP(d3[sel], d2[sel], K, dist,
                                          flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            continue
        if not ok or float(tvec[2, 0]) < 0:
            continue
        proj, _ = cv2.projectPoints(d3, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - d2, axis=1)
        c = int((err < tau).sum())
        if c > best_c:
            best_c = c
            R, _ = cv2.Rodrigues(rvec)
            best_R, best_t = R, tvec.flatten()
    return max(best_c, 0), best_R, best_t


def loo_stability(kp_list, kp3d, K, dist, R, t, tau=0.05, min_pts=5):
    det = [i for i in range(8) if kp_list[i] is not None]
    if len(det) < min_pts:
        return False
    d2 = np.array([[float(kp_list[i][0]), float(kp_list[i][1])] for i in det])
    d3 = kp3d[det].astype(np.float64)
    rvec0, _ = cv2.Rodrigues(R)
    proj_all, _ = cv2.projectPoints(kp3d[:8], rvec0, t.reshape(3, 1), K, dist)
    diag = 0.0
    pa = proj_all.reshape(-1, 2)
    for i in range(8):
        for j in range(i + 1, 8):
            diag = max(diag, np.linalg.norm(pa[i] - pa[j]))
    if diag < 1e-6:
        return False
    errs = []
    for li in range(len(det)):
        mask = [m for m in range(len(det)) if m != li]
        if len(mask) < 4:
            continue
        ok, rv, tv = cv2.solvePnP(d3[mask], d2[mask], K, dist,
                                  flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        pr, _ = cv2.projectPoints(d3[li].reshape(1, 3), rv, tv, K, dist)
        errs.append(np.linalg.norm(pr.reshape(2) - d2[li]))
    if not errs:
        return False
    return (np.median(errs) / diag) < tau


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--good_thresh_px", type=float, default=DEFAULT_GOOD_PX)
    ap.add_argument("--ransac_consensus", type=int, default=6)
    ap.add_argument("--include_forklift", action="store_true",
                    help="add forklift gt_manual (rgb subdir) as a 3rd set")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sets = dict(DEFAULT_SETS)
    if args.include_forklift:
        sets["forklift"] = (FORKLIFT_GT, FORKLIFT_IMG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])

    FILTERS = ["none", "conf", "ransac", "ransac_loo", "cf_strict",
               "diag", "topbot", "ratio", "edgelen", "fullkp", "combo"]
    per_frame = []

    for ds_name, (ds_dir, img_dir) in sets.items():
        idir = img_dir or ds_dir
        files = sorted(glob.glob(os.path.join(ds_dir, "*.json")))
        print(f"[{ds_name}] {len(files)} GT frames")
        for fi, jp in enumerate(files):
            base = os.path.splitext(os.path.basename(jp))[0]
            ip = None
            for ext in (".png", ".jpg"):
                c = os.path.join(idir, base + ext)
                if os.path.exists(c):
                    ip = c; break
            if ip is None:
                continue
            gt = json.load(open(jp))
            obj = gt["objects"][0]
            gt8 = np.array(obj["projected_cuboid"], float)[:8]
            dm = obj.get("dimensions_m", {"width": 1.3, "depth": 1.1, "height": 0.11})
            kp3d = canonical_kp3d(dm["width"], dm["depth"], dm["height"])
            intr = gt["camera_data"]["intrinsics"]
            K = np.array([[intr["fx"], 0, intr["cx"]],
                          [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
            dist = np.zeros((5, 1))

            img = cv2.imread(ip)
            h0, w0 = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = ((cv2.resize(rgb, (448, 448)).astype(np.float32) / 255.0 - mean) / std)
            tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out_bel, _ = model(tensor)
            belief = out_bel[-1][0].cpu().numpy()
            kps_bel = extract_keypoints_from_belief(belief, args.threshold)
            bh, bw = belief.shape[1], belief.shape[2]
            sx, sy = bw / w0, bh / h0

            kp = []           # (u,v) in original image px, or None  (9 entries)
            confs = []
            for k in kps_bel:
                if k[0] < 0:
                    kp.append(None); confs.append(0.0)
                else:
                    kp.append((k[0] / sx, k[1] / sy)); confs.append(k[2])
            n_det = sum(1 for x in kp[:8] if x is not None)
            min_conf = (min([c for x, c in zip(kp[:8], confs) if x is not None])
                        if n_det else 0.0)

            # order-free good judgment
            pred8 = np.full((8, 2), np.nan)
            for i in range(8):
                if kp[i] is not None:
                    pred8[i] = kp[i]
            mean_match, n_match = hungarian_mean_dist(pred8, gt8)
            good = bool(mean_match < args.good_thresh_px)

            # PnP (SQPnP) for PnP-based filters
            ok_pnp, R, t_, _ = sqpnp(kp, kp3d, K, dist, min_pts=6)
            n_cons, R_rs, t_rs = ransac_consensus(kp, kp3d, K, dist)
            ransac_pass = n_cons >= args.ransac_consensus
            loo_pass = (loo_stability(kp, kp3d, K, dist, R_rs, t_rs)
                        if (ransac_pass and R_rs is not None) else False)

            # canonical structural filters (cf_strict = B AND C AND D)
            if ok_pnp:
                try:
                    pb, _ = cf.filter_B(kp, _PnPShim(kp3d, K, dist), R, t_,
                                        img_size=(w0, h0))
                except Exception:
                    pb = False
                try:
                    pc, _ = cf.filter_C(kp, _PnPShim(kp3d, K, dist), R, t_)
                except Exception:
                    pc = False
                try:
                    pd, _, _ = cf.filter_D(kp, _PnPShim(kp3d, K, dist), R, t_)
                except Exception:
                    pd = False
            else:
                pb = pc = pd = False

            res = {}
            res["none"] = bool(ok_pnp)
            res["conf"] = bool(ok_pnp and min_conf > CONF_THRESHOLD)
            res["ransac"] = bool(ransac_pass)
            res["ransac_loo"] = bool(ransac_pass and loo_pass)
            res["cf_strict"] = bool(pb and pc and pd)
            res["diag"] = filt_diag(kp)[0]
            res["topbot"] = filt_topbot(kp)[0]
            res["ratio"] = filt_ratio(kp)[0]
            res["edgelen"] = filt_edgelen(kp)[0]
            res["fullkp"] = filt_fullkp(kp)[0]
            res["combo"] = filt_combo(kp)

            edgelen_score = filt_edgelen(kp)[1]
            per_frame.append({
                "dataset": ds_name, "frame": base,
                "n_detected": n_det, "min_conf": round(float(min_conf), 4),
                "mean_match_px": round(mean_match, 2) if np.isfinite(mean_match) else None,
                "n_match": n_match, "good": good,
                "ransac_consensus": int(n_cons),
                "edgelen_score": round(edgelen_score, 4) if np.isfinite(edgelen_score) else None,
                "kp": [list(p) if p is not None else None for p in kp],
                "filters": {k: bool(v) for k, v in res.items()},
            })
            if (fi + 1) % 40 == 0:
                print(f"  [{ds_name} {fi+1}/{len(files)}]")

    # aggregate (overall + per dataset)
    def aggregate(rows):
        out = []
        for fid in FILTERS:
            TP = FP = TN = FN = 0
            for r in rows:
                passed = r["filters"][fid]; good = r["good"]
                if passed and good: TP += 1
                elif passed and not good: FP += 1
                elif (not passed) and good: FN += 1
                else: TN += 1
            n_pass = TP + FP
            P = TP / n_pass if n_pass else 0.0
            R = TP / (TP + FN) if (TP + FN) else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) else 0.0
            out.append({"filter": fid, "n_pass": n_pass, "TP": TP, "FP": FP,
                        "TN": TN, "FN": FN, "precision": round(P, 4),
                        "recall": round(R, 4), "f1": round(F1, 4)})
        return out

    overall = aggregate(per_frame)
    per_ds = {ds: aggregate([r for r in per_frame if r["dataset"] == ds])
              for ds in sets}
    n_good = sum(1 for r in per_frame if r["good"])

    print(f"\n=== OVERALL (n={len(per_frame)}, good={n_good}, "
          f"thr={args.good_thresh_px}px) ===")
    hdr = f"{'filter':<11}{'pass':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>7}{'R':>7}{'F1':>7}"
    print(hdr); print("-" * len(hdr))
    for r in overall:
        print(f"{r['filter']:<11}{r['n_pass']:>5}{r['TP']:>5}{r['FP']:>5}"
              f"{r['FN']:>5}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")

    out_json = os.path.join(args.output_dir, f"summary_{args.tag}.json")
    out_csv = os.path.join(args.output_dir, f"summary_{args.tag}.csv")
    out_pf = os.path.join(args.output_dir, f"per_frame_{args.tag}.json")
    json.dump({"weights": args.weights, "tag": args.tag,
               "n_frames": len(per_frame), "n_good": n_good,
               "good_thresh_px": args.good_thresh_px,
               "ransac_consensus": args.ransac_consensus,
               "overall": overall, "per_dataset": per_ds},
              open(out_json, "w"), indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "filter", "n_pass", "TP",
                           "FP", "TN", "FN", "precision", "recall", "f1"])
        w.writeheader()
        for r in overall:
            w.writerow({"scope": "overall", **r})
        for ds, rows in per_ds.items():
            for r in rows:
                w.writerow({"scope": ds, **r})
    json.dump(per_frame, open(out_pf, "w"), indent=2, default=str)
    print(f"\n[save] {out_json}\n[save] {out_csv}\n[save] {out_pf}")


class _PnPShim:
    """Minimal adapter so canonical_filters (expects pnp_solver) can be reused."""
    def __init__(self, kp3d, K, dist):
        self.keypoints_3d = kp3d
        self.camera_matrix = K
        self.dist_coeffs = dist

    def reproject(self, R, t):
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(self.keypoints_3d, rvec, t.reshape(3, 1),
                                    self.camera_matrix, self.dist_coeffs)
        return proj.reshape(-1, 2)


if __name__ == "__main__":
    main()
