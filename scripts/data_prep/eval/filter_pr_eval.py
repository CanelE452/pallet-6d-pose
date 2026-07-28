"""filter_pr_eval.py — Pseudo-label filter Precision/Recall on GT-annotated real data.

Goal: pick the best self-training filter by computing precision/recall against
ADD < 5cm ground truth on capture0403middle (440 frames).

Compares 14 filter candidates:
    F0  no filter
    F1  confidence (min peak > 0.5)
    F2  old reproj+cuboid+size (current geometric_filter.py)
    F3  A only (flip consistency, canonical_filters.filter_A)
    F4  B only (structural support)
    F5  C only (LOO PnP stability)
    F6  D only (diagonal incidence)
    F7  B AND C
    F8  A AND B AND C
    F9  B AND C AND D
    F10 A AND B AND C AND D
    F11 RANSAC subset consensus (n_iter=50, subset=5, min_consensus=6)
    F12 reproj-guided PnP (Huber + coverage)
    F13 (B AND C) on reproj-guided pose

For F0..F10 the pose used is the current PnP (solvePnPRansac).
For F11 the pose is the RANSAC-subset best pose.
For F12, F13 the pose is the reproj-guided pose.

Usage:
    python scripts/data_prep/eval/filter_pr_eval.py \\
        --weights weights/v9_ablation_A_coord/final_net_epoch_0065.pth \\
        --tag ep65

Output (under data/pallet/eval_results/filter_pr/):
    summary_{tag}.json    aggregated TP/FP/TN/FN/P/R/F1 per filter
    summary_{tag}.csv     same, CSV
    per_frame_{tag}.json  per-frame keypoints, poses, ADD, filter pass/fail
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# ── Path setup ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PREP = os.path.dirname(HERE)  # data_prep/ root (for canonical_filters)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "self_training"))
sys.path.insert(0, DATA_PREP)  # for canonical_filters

from models import DopeNetwork  # noqa: E402
from pnp_solver import (  # noqa: E402
    PalletPnPSolver,
    make_camera_matrix,
)


def make_pallet_keypoints_3d_canonical(width=1.10, depth=1.30, height=0.11):
    """Build 9 keypoints in the SAME object frame Isaac SDG uses for label generation.

    sdg_math._canonical_corners places corners in canonical bbox space:
        X = medium (width)
        Y = height (Y=UP convention)
        Z = long   (depth)
    with corner order:
        0=(mn_x, mx_y, mx_z)  1=(mx_x, mx_y, mx_z)  2=(mx_x, mn_y, mx_z)
        3=(mn_x, mn_y, mx_z)  4=(mn_x, mx_y, mn_z)  5=(mx_x, mx_y, mn_z)
        6=(mx_x, mn_y, mn_z)  7=(mn_x, mn_y, mn_z)

    These are the 3D points the DOPE model was trained against (label projection),
    so PnP at inference must use this exact frame to recover a meaningful pose.
    """
    W = width / 2.0
    H = height / 2.0
    D = depth / 2.0
    mn = (-W, -H, -D)
    mx = (+W, +H, +D)
    corners = np.array([
        [mn[0], mx[1], mx[2]],  # 0
        [mx[0], mx[1], mx[2]],  # 1
        [mx[0], mn[1], mx[2]],  # 2
        [mn[0], mn[1], mx[2]],  # 3
        [mn[0], mx[1], mn[2]],  # 4
        [mx[0], mx[1], mn[2]],  # 5
        [mx[0], mn[1], mn[2]],  # 6
        [mn[0], mn[1], mn[2]],  # 7
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])
from geometric_filter import GeometricFilter  # noqa: E402
import canonical_filters as cf  # noqa: E402


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA = os.path.join(ROOT, "data", "pallet", "raw_data", "capture0403middle")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "pallet", "eval_results", "filter_pr")

DEFAULT_REPROJ_GOOD_THRESHOLD_PX = 10.0  # default 2D mean corner reproj error vs GT
CONF_THRESHOLD = 0.5             # F1: confidence gate

# NOTE on "good" criterion:
# We use 2D reprojection error against GT projected_cuboid (pixel space)
# rather than 3D ADD because:
#   1. Model training (Isaac Y=UP) and GT generation (apriltag_gt_multitag.py
#      uses Isaac OpenCV-converted frame) have different object frames →
#      3D ADD systematically biased.
#   2. Pixel-space comparison is frame-agnostic and matches what reviewers
#      see in qualitative figures.
#   3. For self-training pseudo-label quality, "does the recovered pose
#      project to the right place in the image?" is the right question.

FILTER_IDS = [
    "F0", "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12", "F13",
    # ── B∧C threshold sweep (F7 base = default canonical) ──
    "F14",  # B∧C loose 2x
    "F15",  # B∧C loose 3x
    "F16",  # B∧C very loose
    # ── RANSAC consensus sweep (F11 base = consensus≥6) ──
    "F17",  # RANSAC consensus≥4
    "F18",  # RANSAC consensus≥5
    "F19",  # RANSAC consensus≥7
    "F20",  # RANSAC consensus≥8
    # ── Loose-B alone, loose-C alone (diagnostic) ──
    "F21",  # B loose 2x only
    "F22",  # C loose 2x only
]

# Threshold sweeps
B_DEFAULT = dict(tau_span=0.35, tau_end=0.10, tau_nc=0.02)
B_LOOSE_2X = dict(tau_span=0.20, tau_end=0.20, tau_nc=0.01)
B_LOOSE_3X = dict(tau_span=0.12, tau_end=0.30, tau_nc=0.005)
B_VERY_LOOSE = dict(tau_span=0.05, tau_end=0.50, tau_nc=0.001)

C_DEFAULT = 0.05
C_LOOSE_2X = 0.10
C_LOOSE_3X = 0.15
C_VERY_LOOSE = 0.30


# ── DOPE inference helpers ────────────────────────────────────────────
def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def extract_keypoints_from_belief(belief_maps, threshold=0.3):
    """Same logic as evaluate_real.py — returns list of (x, y, peak) in belief coords."""
    OFFSET = 0.4395
    RAN = 5
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap_ori = belief_maps[i]
        max_val = bmap_ori.max()
        if max_val < threshold:
            keypoints.append((-1, -1, float(max_val)))
            continue
        bmap_smooth = gaussian_filter(bmap_ori, sigma=2)
        p = 1
        pad_l = np.zeros_like(bmap_smooth); pad_l[p:, :] = bmap_smooth[:-p, :]
        pad_r = np.zeros_like(bmap_smooth); pad_r[:-p, :] = bmap_smooth[p:, :]
        pad_u = np.zeros_like(bmap_smooth); pad_u[:, p:] = bmap_smooth[:, :-p]
        pad_d = np.zeros_like(bmap_smooth); pad_d[:, :-p] = bmap_smooth[:, p:]
        peaks_binary = (
            (bmap_smooth >= pad_l) & (bmap_smooth >= pad_r) &
            (bmap_smooth >= pad_u) & (bmap_smooth >= pad_d) &
            (bmap_smooth > threshold)
        )
        peak_ys, peak_xs = np.nonzero(peaks_binary)
        if len(peak_xs) == 0:
            keypoints.append((-1, -1, float(max_val)))
            continue
        peak_vals = [bmap_ori[py, px] for py, px in zip(peak_ys, peak_xs)]
        best_idx = int(np.argmax(peak_vals))
        px, py = int(peak_xs[best_idx]), int(peak_ys[best_idx])
        y_lo = max(0, py - RAN); y_hi = min(bmap_ori.shape[0], py + RAN + 1)
        x_lo = max(0, px - RAN); x_hi = min(bmap_ori.shape[1], px + RAN + 1)
        patch = bmap_ori[y_lo:y_hi, x_lo:x_hi]
        if patch.sum() > 0:
            ys = np.arange(y_lo, y_hi)
            xs = np.arange(x_lo, x_hi)
            xg, yg = np.meshgrid(xs, ys)
            wx = float(np.average(xg, weights=patch)) + OFFSET
            wy = float(np.average(yg, weights=patch)) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy, float(max_val)))
    return keypoints


def compute_reproj_2d_error(reproj_8, gt_cuboid_2d_8):
    """Mean 2D pixel distance between predicted-reprojected 8 corners and GT 8 corners.

    Both inputs are (8, 2) in pixel coordinates of the original image.
    """
    diff = np.array(reproj_8, dtype=np.float64) - np.array(gt_cuboid_2d_8, dtype=np.float64)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


# ── F11: RANSAC subset PnP ────────────────────────────────────────────
def ransac_subset_pnp(kps_orig, pnp_solver, n_iter=50, subset_size=5,
                      reproj_thresh=5.0, min_consensus=6, seed=0):
    """Random subset PnP, vote by full reproj inlier count."""
    detected_idx = [i for i, p in enumerate(kps_orig[:8]) if p is not None]
    if len(detected_idx) < subset_size:
        return False, None, None, 0

    detected_2d = np.array(
        [[float(kps_orig[i][0]), float(kps_orig[i][1])] for i in detected_idx],
        dtype=np.float64,
    )
    kp3d = pnp_solver.keypoints_3d[:8]
    detected_3d = kp3d[detected_idx].astype(np.float64)

    best_consensus = -1
    best_rvec, best_tvec = None, None
    rng = np.random.default_rng(seed)
    n = len(detected_idx)

    for _ in range(n_iter):
        sel = (np.arange(n) if n == subset_size
               else rng.choice(n, size=subset_size, replace=False))
        try:
            ok, rvec, tvec = cv2.solvePnP(
                detected_3d[sel], detected_2d[sel],
                pnp_solver.camera_matrix, pnp_solver.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            continue
        if not ok or float(tvec[2, 0]) < 0:
            continue
        reproj, _ = cv2.projectPoints(
            detected_3d, rvec, tvec,
            pnp_solver.camera_matrix, pnp_solver.dist_coeffs,
        )
        errors = np.linalg.norm(reproj.reshape(-1, 2) - detected_2d, axis=1)
        consensus = int(np.sum(errors < reproj_thresh))
        if consensus > best_consensus:
            best_consensus = consensus
            best_rvec, best_tvec = rvec, tvec

    if best_rvec is None:
        return False, None, None, 0
    R, _ = cv2.Rodrigues(best_rvec)
    t = best_tvec.flatten()
    passed = best_consensus >= min_consensus
    return passed, R, t, int(best_consensus)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tag", required=True, help="output suffix (e.g. ep65)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="belief peak threshold for keypoint extraction")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="0 = all frames")
    parser.add_argument("--good_thresh_px", type=float,
                        default=DEFAULT_REPROJ_GOOD_THRESHOLD_PX,
                        help="2D mean reproj error (px) below which a pose is 'good'")
    args = parser.parse_args()
    REPROJ_GOOD_THRESHOLD_PX = args.good_thresh_px

    os.makedirs(args.output_dir, exist_ok=True)

    # Camera intrinsics from cam_K.txt
    cam_k_path = os.path.join(args.data_dir, "cam_K.txt")
    K_intr = np.loadtxt(cam_k_path)
    fx, fy = K_intr[0, 0], K_intr[1, 1]
    cx, cy = K_intr[0, 2], K_intr[1, 2]
    print(f"[cam] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    cam_matrix = make_camera_matrix(fx, fy, cx, cy)
    # Use Isaac canonical 3D points with real pallet dims (1.10×1.30×0.11)
    # to match both training data and GT generation. PnP recovers a pose
    # whose reproject() output we compare in pixel space.
    pnp_solver = PalletPnPSolver(cam_matrix)
    pnp_solver.keypoints_3d = make_pallet_keypoints_3d_canonical(
        width=1.10, depth=1.30, height=0.11)
    old_gf = GeometricFilter(pnp_solver)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights}  ({device})")
    model = load_model(args.weights, device)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    gt_dir = os.path.join(args.data_dir, "gt_final")
    rgb_dir = os.path.join(args.data_dir, "rgb")
    json_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    if args.max_frames > 0:
        json_files = json_files[: args.max_frames]
    print(f"[data] {len(json_files)} GT frames in {gt_dir}")

    per_frame = []
    t0 = time.time()

    for ji, json_path in enumerate(json_files):
        basename = os.path.splitext(os.path.basename(json_path))[0]
        img_path = None
        for ext in [".png", ".jpg"]:
            cand = os.path.join(rgb_dir, basename + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue

        with open(json_path) as f:
            gt_data = json.load(f)
        gt_obj = gt_data["objects"][0]
        gt_cuboid_2d = np.array(gt_obj["projected_cuboid"], dtype=np.float64)  # (8, 2)

        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (400, 400))  # 학습 입력(400×400)과 일치
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()
        pred_kps_belief = extract_keypoints_from_belief(belief, args.threshold)

        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = bw / w_orig, bh / h_orig

        kps_orig = []
        peak_confs = []
        for kp in pred_kps_belief:
            if kp[0] < 0:
                kps_orig.append(None)
                peak_confs.append(0.0)
            else:
                kps_orig.append((float(kp[0]) / sx, float(kp[1]) / sy))
                peak_confs.append(float(kp[2]))

        # belief-coord version for filter_A
        kps_belief = [(kp[0], kp[1]) if kp[0] >= 0 else None
                      for kp in pred_kps_belief]

        n_detected = sum(1 for k in kps_orig if k is not None)
        valid_confs = [c for k, c in zip(kps_orig, peak_confs) if k is not None]
        min_conf = float(min(valid_confs)) if valid_confs else 0.0

        # ── PnP variants ──
        ok_cur, R_cur, t_cur, _ = pnp_solver.solve(kps_orig)
        try:
            rg_out = pnp_solver.solve_reproj_guided(
                kps_orig, peak_confidences=peak_confs)
            ok_rg, R_rg, t_rg = rg_out[0], rg_out[1], rg_out[2]
        except Exception:
            ok_rg, R_rg, t_rg = False, None, None
        ok_rs, R_rs, t_rs, n_consensus = ransac_subset_pnp(kps_orig, pnp_solver)

        # Per-pose 2D reproj error vs GT projected_cuboid
        if ok_cur:
            reproj_cur = pnp_solver.reproject(R_cur, t_cur)[:8]
            err_cur = compute_reproj_2d_error(reproj_cur, gt_cuboid_2d)
        else:
            err_cur = float("inf")
        if ok_rg:
            reproj_rg = pnp_solver.reproject(R_rg, t_rg)[:8]
            err_rg = compute_reproj_2d_error(reproj_rg, gt_cuboid_2d)
        else:
            err_rg = float("inf")
        if ok_rs:
            reproj_rs = pnp_solver.reproject(R_rs, t_rs)[:8]
            err_rs = compute_reproj_2d_error(reproj_rs, gt_cuboid_2d)
        else:
            err_rs = float("inf")

        good_cur = bool(err_cur < REPROJ_GOOD_THRESHOLD_PX)
        good_rg = bool(err_rg < REPROJ_GOOD_THRESHOLD_PX)
        good_rs = bool(err_rs < REPROJ_GOOD_THRESHOLD_PX)

        # ── Filter pass/fail ──
        filter_results = {f: False for f in FILTER_IDS}

        # F0: no filter (just needs PnP success)
        filter_results["F0"] = bool(ok_cur)
        # F1: confidence
        filter_results["F1"] = bool(ok_cur and (min_conf > CONF_THRESHOLD))
        # F2: old reproj+cuboid+size
        if ok_cur:
            f2_pass, _ = old_gf.validate(kps_orig, R_cur, t_cur)
            filter_results["F2"] = bool(f2_pass)

        # F3-F6: A, B, C, D individually (using current PnP pose)
        if ok_cur:
            try:
                pa, _ = cf.filter_A(model, img, kps_belief, device,
                                    pnp_solver, R_cur, t_cur)
            except Exception:
                pa = False
            try:
                pb, _ = cf.filter_B(kps_orig, pnp_solver, R_cur, t_cur,
                                    img_size=(w_orig, h_orig))
            except Exception:
                pb = False
            try:
                pc, _ = cf.filter_C(kps_orig, pnp_solver, R_cur, t_cur)
            except Exception:
                pc = False
            try:
                pd, _, _ = cf.filter_D(kps_orig, pnp_solver, R_cur, t_cur)
            except Exception:
                pd = False
        else:
            pa = pb = pc = pd = False

        filter_results["F3"] = bool(pa)
        filter_results["F4"] = bool(pb)
        filter_results["F5"] = bool(pc)
        filter_results["F6"] = bool(pd)
        filter_results["F7"] = bool(pb and pc)
        filter_results["F8"] = bool(pa and pb and pc)
        filter_results["F9"] = bool(pb and pc and pd)
        filter_results["F10"] = bool(pa and pb and pc and pd)

        # F11: RANSAC subset (its own pose)
        filter_results["F11"] = bool(ok_rs)

        # F12: reproj-guided (its own pose)
        filter_results["F12"] = bool(ok_rg)

        # F13: B∧C on reproj-guided pose
        if ok_rg:
            try:
                pb_rg, _ = cf.filter_B(kps_orig, pnp_solver, R_rg, t_rg,
                                       img_size=(w_orig, h_orig))
                pc_rg, _ = cf.filter_C(kps_orig, pnp_solver, R_rg, t_rg)
                filter_results["F13"] = bool(pb_rg and pc_rg)
            except Exception:
                filter_results["F13"] = False

        # ── F14-F16: B∧C threshold sweep (loose 2x / 3x / very loose) ──
        if ok_cur:
            for fid, b_cfg, c_th in [
                ("F14", B_LOOSE_2X, C_LOOSE_2X),
                ("F15", B_LOOSE_3X, C_LOOSE_3X),
                ("F16", B_VERY_LOOSE, C_VERY_LOOSE),
            ]:
                try:
                    pb_l, _ = cf.filter_B(
                        kps_orig, pnp_solver, R_cur, t_cur,
                        tau_span=b_cfg["tau_span"],
                        tau_end=b_cfg["tau_end"],
                        tau_nc=b_cfg["tau_nc"],
                        img_size=(w_orig, h_orig),
                    )
                    pc_l, _ = cf.filter_C(
                        kps_orig, pnp_solver, R_cur, t_cur, tau_C=c_th)
                    filter_results[fid] = bool(pb_l and pc_l)
                except Exception:
                    filter_results[fid] = False
            # F21: B loose 2x only
            try:
                pb_l, _ = cf.filter_B(
                    kps_orig, pnp_solver, R_cur, t_cur,
                    tau_span=B_LOOSE_2X["tau_span"],
                    tau_end=B_LOOSE_2X["tau_end"],
                    tau_nc=B_LOOSE_2X["tau_nc"],
                    img_size=(w_orig, h_orig),
                )
                filter_results["F21"] = bool(pb_l)
            except Exception:
                filter_results["F21"] = False
            # F22: C loose 2x only
            try:
                pc_l, _ = cf.filter_C(
                    kps_orig, pnp_solver, R_cur, t_cur, tau_C=C_LOOSE_2X)
                filter_results["F22"] = bool(pc_l)
            except Exception:
                filter_results["F22"] = False

        # ── F17-F20: RANSAC consensus sweep (4/5/7/8) ──
        # n_consensus is the BEST consensus from F11's run with min_consensus=6.
        # We just compare to different thresholds; same pose is used.
        filter_results["F17"] = bool(R_rs is not None and n_consensus >= 4)
        filter_results["F18"] = bool(R_rs is not None and n_consensus >= 5)
        filter_results["F19"] = bool(R_rs is not None and n_consensus >= 7)
        filter_results["F20"] = bool(R_rs is not None and n_consensus >= 8)

        # ── Map filter → which pose's reproj it should be judged against ──
        good_for_filter = {fid: good_cur for fid in
                           ["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10",
                            "F14","F15","F16","F21","F22"]}
        good_for_filter["F11"] = good_rs
        good_for_filter["F12"] = good_rg
        good_for_filter["F13"] = good_rg
        good_for_filter["F17"] = good_rs
        good_for_filter["F18"] = good_rs
        good_for_filter["F19"] = good_rs
        good_for_filter["F20"] = good_rs

        per_frame.append({
            "frame": basename,
            "n_detected": n_detected,
            "min_conf": round(min_conf, 4),
            "ok_cur": bool(ok_cur),
            "err_cur_px": round(err_cur, 2) if np.isfinite(err_cur) else None,
            "good_cur": good_cur,
            "ok_rg": bool(ok_rg),
            "err_rg_px": round(err_rg, 2) if np.isfinite(err_rg) else None,
            "good_rg": good_rg,
            "ok_rs": bool(ok_rs),
            "err_rs_px": round(err_rs, 2) if np.isfinite(err_rs) else None,
            "good_rs": good_rs,
            "ransac_consensus": int(n_consensus),
            "filters": filter_results,
            "good_for_filter": good_for_filter,
        })

        if (ji + 1) % 50 == 0 or ji == 0:
            elapsed = time.time() - t0
            err_str = f"{err_cur:.1f}px" if np.isfinite(err_cur) else "FAIL"
            print(f"  [{ji+1}/{len(json_files)}] {basename}  "
                  f"det={n_detected}/9  cur={err_str}  ({elapsed:.0f}s)")

    # ── Aggregate ──
    print("\n[aggregate] computing TP/FP/TN/FN per filter...")
    summary_rows = []
    for fid in FILTER_IDS:
        TP = FP = TN = FN = 0
        for fr in per_frame:
            passed = fr["filters"][fid]
            good = fr["good_for_filter"][fid]
            if passed and good:
                TP += 1
            elif passed and not good:
                FP += 1
            elif not passed and good:
                FN += 1
            else:
                TN += 1
        n_pass = TP + FP
        precision = TP / n_pass if n_pass > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = ((2 * precision * recall) / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        summary_rows.append({
            "filter": fid,
            "n_pass": n_pass,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    # ── Print table ──
    print()
    header = f"{'ID':<5} {'pass':<6} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5} {'P':<8} {'R':<8} {'F1':<8}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['filter']:<5} {row['n_pass']:<6} "
              f"{row['TP']:<5} {row['FP']:<5} {row['TN']:<5} {row['FN']:<5} "
              f"{row['precision']:<8.3f} {row['recall']:<8.3f} {row['f1']:<8.3f}")

    n_good_cur = sum(1 for fr in per_frame if fr["good_cur"])
    n_good_rg = sum(1 for fr in per_frame if fr["good_rg"])
    n_good_rs = sum(1 for fr in per_frame if fr["good_rs"])
    print(f"\n[good count] cur={n_good_cur}/{len(per_frame)}  "
          f"rg={n_good_rg}/{len(per_frame)}  rs={n_good_rs}/{len(per_frame)}")

    # ── Save ──
    out_summary = os.path.join(args.output_dir, f"summary_{args.tag}.json")
    out_csv = os.path.join(args.output_dir, f"summary_{args.tag}.csv")
    out_perframe = os.path.join(args.output_dir, f"per_frame_{args.tag}.json")
    with open(out_summary, "w") as f:
        json.dump({
            "weights": args.weights,
            "tag": args.tag,
            "n_frames": len(per_frame),
            "reproj_threshold_px": REPROJ_GOOD_THRESHOLD_PX,
            "conf_threshold": CONF_THRESHOLD,
            "n_good_cur": n_good_cur,
            "n_good_rg": n_good_rg,
            "n_good_rs": n_good_rs,
            "filters": summary_rows,
        }, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filter","n_pass","TP","FP","TN","FN","precision","recall","f1"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(out_perframe, "w") as f:
        json.dump(per_frame, f, indent=2, default=str)

    print(f"\n[save] {out_summary}")
    print(f"[save] {out_csv}")
    print(f"[save] {out_perframe}")
    print(f"[time] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
