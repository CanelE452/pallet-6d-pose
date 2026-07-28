"""Diagnose why PnP returns TB-flipped solution.

User click semantics:
  0=(294,296) FRONT-TOP-LEFT  click is upper-left
  1=(540,310) FRONT-TOP-RIGHT
  2=(544,335) FRONT-BOT-RIGHT click is lower-right
  3=(300,319) FRONT-BOT-LEFT
  4=(376,279) REAR-TOP-LEFT  click is upper (above 0)
  5=(551,288) REAR-TOP-RIGHT click is upper (above 1)

Cuboid local:
  0 = [-w/2, -h/2, +d/2]   Y=-h/2 = TOP (OpenCV Y=down)
  1 = [+w/2, -h/2, +d/2]
  2 = [+w/2, +h/2, +d/2]   Y=+h/2 = BOT
  3 = [-w/2, +h/2, +d/2]
  4 = [-w/2, -h/2, -d/2]   TOP, rear
  5 = [+w/2, -h/2, -d/2]
  6 = [+w/2, +h/2, -d/2]
  7 = [-w/2, +h/2, -d/2]

Expected pose: cuboid is roughly upright (Y_local ≈ cam_Y, gravity-down).
So R should be roughly identity (with maybe slight tilt for the perspective).
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import make_pallet_keypoints_3d, project_3d, PALLET_DIMS

_REPO = os.path.dirname(os.path.dirname(_HERE))
SRC_JSON = os.path.join(
    _REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.json")


def reproj_per_corner(R, t, K, kp3d, clicks):
    proj = project_3d(kp3d, R, t, K)
    out = []
    for i, p in enumerate(clicks[:9]):
        if p is None:
            out.append((i, proj[i][0], proj[i][1], None, None, None))
            continue
        du = proj[i][0] - p[0]
        dv = proj[i][1] - p[1]
        out.append((i, proj[i][0], proj[i][1], p[0], p[1], float(np.hypot(du, dv))))
    return out, proj


def main():
    with open(SRC_JSON) as f:
        d = json.load(f)
    cam = d["camera_data"]["intrinsics"]
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    manual = d["objects"][0]["manual_kps"]
    print("manual_kps:")
    for i, p in enumerate(manual[:8]):
        print(f"  {i}: ({p[0]:6.1f}, {p[1]:6.1f})")
    print()

    kps_6 = list(manual[:6]) + [None] * 3
    kp3d = make_pallet_keypoints_3d(*PALLET_DIMS)
    print(f"kp3d (local):")
    for i, p in enumerate(kp3d[:8]):
        print(f"  {i}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    print()

    # Try forcing the "correct" pose: identity rotation (cuboid aligned with cam axes)
    print("=" * 70)
    print("Test 1: R=I (cuboid axes aligned with cam axes)")
    print("=" * 70)
    R0 = np.eye(3)
    t0 = np.array([0.5, 0.0, 3.0])
    res, proj = reproj_per_corner(R0, t0, K, kp3d, kps_6)
    for (i, pu, pv, cu, cv, err) in res:
        err_s = f"err={err:.2f}" if err is not None else "no click"
        print(f"  {i}: proj=({pu:6.1f},{pv:6.1f}) click=({cu},{cv}) {err_s}")
    print()

    # PnP just from clicks 0-3 (4 coplanar pts → IPPE)
    print("=" * 70)
    print("Test 2: IPPE on 4 front clicks only - should give 2 solutions")
    print("=" * 70)
    obj = kp3d[:4].astype(np.float64)
    img = np.array(manual[:4], dtype=np.float64)
    ok_n, rvecs, tvecs, errs_ippe = cv2.solvePnPGeneric(
        obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    print(f"  n_solutions={len(rvecs) if ok_n else 0}")
    for k, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rv)
        t = tv.flatten()
        print(f"  solution {k}: t={t}")
        # Check TB on full cuboid projected
        proj_all = project_3d(kp3d, R, t, K)
        v_top = (proj_all[0][1] + proj_all[1][1]) / 2
        v_bot = (proj_all[2][1] + proj_all[3][1]) / 2
        print(f"    TOP_v_mean={v_top:.1f}  BOT_v_mean={v_bot:.1f}  "
              f"{'TB OK' if v_top < v_bot else 'TB FLIP'}")
        # cam-Y of front corners
        pts_cam = (R @ kp3d[:8].T).T + t
        print(f"    front cam Y: 0={pts_cam[0,1]:.3f} 3={pts_cam[3,1]:.3f}")
        # full reproj 8 corner (only 4 clicked but check error on all 4)
        errs = []
        for i in range(4):
            du = proj_all[i][0] - manual[i][0]
            dv = proj_all[i][1] - manual[i][1]
            errs.append(np.hypot(du, dv))
        print(f"    reproj on 4 front: {np.mean(errs):.2f}px")

    # PnP from 6 clicks (the actual GUI scenario) and dump candidates with full info
    print()
    print("=" * 70)
    print("Test 3: EPNP on 6 clicks - see what reproj-best gives")
    print("=" * 70)
    obj = kp3d[:6].astype(np.float64)
    img = np.array(manual[:6], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    res, proj = reproj_per_corner(R, t, K, kp3d, kps_6)
    print(f"  R=\n{R}")
    print(f"  t={t}")
    print(f"  per-click:")
    for (i, pu, pv, cu, cv, err) in res:
        if err is not None:
            print(f"    {i}: proj=({pu:6.1f},{pv:6.1f}) click=({cu},{cv}) err={err:.2f}px")
    # Check invariants on full 8 corner
    pts_cam = (R @ kp3d[:8].T).T + t
    print(f"\n  Full 8-corner projection:")
    for i in range(8):
        print(f"    {i}: proj=({proj[i][0]:6.1f},{proj[i][1]:6.1f}) "
              f"cam=({pts_cam[i,0]:+.3f},{pts_cam[i,1]:+.3f},{pts_cam[i,2]:.3f})")

    # Per-pair check
    print(f"\n  LR_PAIRS:")
    for (a, b) in [(0, 1), (3, 2), (4, 5), (7, 6)]:
        print(f"    ({a},{b}): proj.u {proj[a][0]:.1f} vs {proj[b][0]:.1f} "
              f"{'OK' if proj[a][0] < proj[b][0] else 'FAIL'}")
    print(f"  TB_PAIRS:")
    for (a, b) in [(0, 3), (1, 2), (4, 7), (5, 6)]:
        print(f"    ({a},{b}): proj.v {proj[a][1]:.1f} vs {proj[b][1]:.1f} "
              f"{'OK' if proj[a][1] < proj[b][1] else 'FAIL'}")
    print(f"  FR_PAIRS:")
    for (a, b) in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        print(f"    ({a},{b}): cam.z {pts_cam[a,2]:.3f} vs {pts_cam[b,2]:.3f} "
              f"{'OK' if pts_cam[a,2] < pts_cam[b,2] else 'FAIL'}")


if __name__ == "__main__":
    main()
