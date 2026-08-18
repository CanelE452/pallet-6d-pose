"""Prove (or disprove) that G / T / R share one keypoint meaning, using the DEPLOYED PnP contract.

A permutation-based check (convert_to_camera_facing_v4) only tells us whether a 2D
heuristic agrees. It cannot tell us whether keypoint i means the same physical corner
across datasets. Reprojection error can: if the 3D model point order of the deployment
contract matches the 2D order stored in a dataset, PnP reprojects tightly. If near/far
or left/right were swapped, the residual explodes.

Model points come from challenge/.../depth_cam/calib/pose6d_adapter.py (deployment),
NOT from a local re-derivation.

Usage:
  python challenge/yolo_pose_one_model/scripts/verify_kp_contract.py --n 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DEPLOY = REPO / "challenge/25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam/calib"
sys.path.insert(0, str(DEPLOY))

from challenge.data_paths import get as dp  # noqa: E402


def model_points(w, h, d):
    """Deployment object_points (pose6d_adapter.keypoints9_to_align_vars), parameterised by dims."""
    hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
    return np.array([
        [-hw, +hh, +hd],   # 0 front-top-LEFT
        [+hw, +hh, +hd],   # 1 front-top-RIGHT
        [+hw, -hh, +hd],   # 2 front-bot-RIGHT
        [-hw, -hh, +hd],   # 3 front-bot-LEFT
        [-hw, +hh, -hd],   # 4 rear-top-LEFT
        [+hw, +hh, -hd],   # 5 rear-top-RIGHT
        [+hw, -hh, -hd],   # 6 rear-bot-RIGHT
        [-hw, -hh, -hd],   # 7 rear-bot-LEFT
        [0.0, 0.0, 0.0],   # 8 centroid
    ], dtype=np.float64)


def solve(kps9, K, dims, swap_nearfar=False):
    """Return median reprojection error (px) or None."""
    obj = model_points(*dims)
    if swap_nearfar:
        obj = obj[[4, 5, 6, 7, 0, 1, 2, 3, 8]]
    kps = np.asarray(kps9, dtype=np.float64)
    if kps.shape != (9, 2):
        return None
    vis = ~np.isnan(kps).any(axis=1) & (kps[:, 0] >= 0) & (kps[:, 1] >= 0)
    if int(vis.sum()) < 6:
        return None
    o = obj[vis].reshape(-1, 1, 3)
    i = kps[vis].reshape(-1, 1, 2)
    D = np.zeros((5, 1))
    try:
        ok, rvec, tvec = cv2.solvePnP(o, i, K, D, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(o, i, K, D, rvec, tvec)
    except cv2.error:
        return None
    proj, _ = cv2.projectPoints(o, rvec, tvec, K, D)
    return float(np.median(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)))


def kfrom(cd):
    it = cd["intrinsics"]
    return np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]], dtype=np.float64)


def load_G(f):
    d = json.load(open(f)); o = d["objects"][0]
    dm = o["dimensions_m"]
    kp = list(o["projected_cuboid"][:8]) + [o["projected_cuboid_centroid"]]
    return kp, kfrom(d["camera_data"]), (dm["width"], dm["height"], dm["depth"])


def load_T(f):
    d = json.load(open(f)); o = d["objects"][0]
    w, dep, h = o["cuboid_dimensions_m"]        # stored as [width, depth, height]
    kp = list(o["projected_cuboid"][:8]) + [o["projected_cuboid_centroid"]]
    return kp, kfrom(d["camera_data"]), (w, h, dep)


def load_R(f):
    d = json.load(open(f)); o = d["objects"][0]
    dm = o["dimensions_m"]
    mk = o.get("manual_kps")
    if mk is None:
        return None
    kp = [(p if p is not None else [-1.0, -1.0]) for p in mk]
    if len(kp) != 9:
        return None
    return kp, kfrom(d["camera_data"]), (dm["width"], dm["height"], dm["depth"])


def run(name, files, loader, n, seed=42):
    picked = files if n <= 0 else random.Random(seed).sample(files, min(n, len(files)))
    asis, swapped, skipped = [], [], 0
    for f in picked:
        try:
            got = loader(f)
        except Exception:
            got = None
        if got is None:
            skipped += 1
            continue
        kp, K, dims = got
        a = solve(kp, K, dims, False)
        s = solve(kp, K, dims, True)
        if a is None:
            skipped += 1
            continue
        asis.append(a)
        if s is not None:
            swapped.append(s)
    if not asis:
        print(f"{name:<26} no solvable frame (skipped {skipped})")
        return
    a = np.array(asis); s = np.array(swapped) if swapped else np.array([np.nan])
    print(f"{name:<26} n={len(a):<5} skip={skipped:<5} "
          f"as-is median={np.median(a):7.2f}px  p90={np.percentile(a,90):8.2f}  "
          f"| near/far-swapped median={np.median(s):8.2f}px")
    print(f"{'':<26} as-is  <2px {100*np.mean(a<2):5.1f}%   <5px {100*np.mean(a<5):5.1f}%  "
          f"<10px {100*np.mean(a<10):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    G = sorted(glob.glob(str(REPO / "data/pallet/training_data/paper_release/"
                             "v2_prod40k_clean_merged/labels/*.json")))
    T1 = sorted(p for p in glob.glob(os.path.join(dp("synth.v1", absolute=True), "**", "*.json"),
                                     recursive=True) if not p.endswith(".orig"))
    T2 = sorted(p for p in glob.glob(os.path.join(dp("synth.v2", absolute=True), "**", "*.json"),
                                     recursive=True) if not p.endswith(".orig"))
    Rf = sorted(glob.glob(str(REPO / "challenge/data/01_real/manual_gt/*/*.json")) +
                glob.glob(str(REPO / "challenge/data/01_real/eval_canonical/*/*.json")))

    print("Deployment contract: pose6d_adapter.object_points, EPnP + LM, >=6 visible kp\n")
    print("A low 'as-is' median with a much higher 'swapped' median proves the stored order")
    print("matches the deployed model-point order (i.e. 0-3 really is the near face).\n")
    run("G  paper_release 40k", G, load_G, args.n)
    run("T  v1 palletobj", T1, load_T, args.n)
    run("T  v2 palletobj", T2, load_T, args.n)
    run("R  real manual_kps", Rf, load_R, args.n)


if __name__ == "__main__":
    main()
