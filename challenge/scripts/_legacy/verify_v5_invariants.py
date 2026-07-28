"""verify_v5_invariants.py — v5 결과 자체 invariant 검증.

각 sample 에 v5 적용 후 다음 invariant 체크:
  1. TOP z 평균 > BOT z 평균 (origin frame, gravity-up)
  2. vertical edge: (0,4),(1,5),(2,6),(3,7) 각 페어의 xy 거리 매우 가까움
  3. FRONT 가 카메라 향함 (yaw_align > 0)
  4. FRONT 가 REAR 보다 가까움 (cam distance 작음)
  5. LR ordering: image x: new_0 ≤ new_1, new_3 ≤ new_2, new_4 ≤ new_5, new_7 ≤ new_6
  6. TB ordering: origin z: TOP > BOT (이미 1번)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_camera_facing_v5 import (
    compute_perm_v5, get_origin_3d, get_pts_cam, apply_perm
)


def check_invariants(data, obj, alpha=0.7, beta=0.3):
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    origin = get_origin_3d(obj)
    pts_cam = get_pts_cam(data, obj)
    if origin is None or pts_cam is None:
        return None
    perm = compute_perm_v5(origin, pts_cam, proj, alpha=alpha, beta=beta)
    if perm is None:
        return None

    proj_np = np.array(proj[:8], dtype=np.float64)
    # Reorder
    new_origin = origin[perm[:8]]
    new_cam = pts_cam[perm[:8]]
    new_proj = proj_np[perm[:8]]

    res = {}
    # 1. TOP z > BOT z
    top_z = new_origin[[0, 1, 5, 4], 2].mean()
    bot_z = new_origin[[3, 2, 6, 7], 2].mean()
    res["top_above_bot"] = top_z > bot_z

    # 2. vertical edge (TOP↔BOT pair) xy proximity (relative to body diag)
    # User convention: 0=FTL,1=FTR,2=FBR,3=FBL,4=RTL,5=RTR,6=RBR,7=RBL
    # Vertical pairs: (0,3),(1,2),(4,7),(5,6) — same vertical column
    diag = np.linalg.norm(new_origin.max(axis=0) - new_origin.min(axis=0))
    pairs = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vert_ok = True
    for a, b in pairs:
        d = np.linalg.norm(new_origin[a, :2] - new_origin[b, :2])
        if d > 0.05 * diag:
            vert_ok = False
            break
    res["vert_edge_ok"] = vert_ok

    # 3. FRONT yaw alignment ≥ REAR (FRONT 가 적어도 REAR 보다는 카메라 향함)
    front_center = new_cam[[0, 1, 2, 3]].mean(axis=0)
    rear_center = new_cam[[4, 5, 6, 7]].mean(axis=0)
    body_c = new_cam.mean(axis=0)

    def yaw_of(face_pts, fc):
        p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return -1.0
        n = n / nn
        if np.dot(n, fc - body_c) < 0:
            n = -n
        los = fc / max(np.linalg.norm(fc), 1e-9)
        return float(np.dot(n, -los))

    yaw_f = yaw_of(new_cam[[0, 1, 2, 3]], front_center)
    yaw_r = yaw_of(new_cam[[4, 5, 6, 7]], rear_center)
    res["front_yaw_ge_rear"] = yaw_f >= yaw_r
    res["front_yaw_pos"] = yaw_f > 0
    res["front_yaw_val"] = yaw_f
    res["rear_yaw_val"] = yaw_r

    # 4. FRONT closer than REAR
    res["front_closer"] = np.linalg.norm(front_center) < np.linalg.norm(rear_center)

    # 5. LR ordering
    lr_ok = (new_proj[0, 0] <= new_proj[1, 0] and
             new_proj[3, 0] <= new_proj[2, 0] and
             new_proj[4, 0] <= new_proj[5, 0] and
             new_proj[7, 0] <= new_proj[6, 0])
    res["lr_ok"] = bool(lr_ok)

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "data/pallet/training_data/mixed_v8_train",
        "challenge/data/training/v1",
        "challenge/data/training/v2",
    ])
    ap.add_argument("--n_sample_per_root", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]

    total_stats = {}
    for r in roots:
        files = sorted(p for p in glob.glob(os.path.join(r, "**", "*.json"), recursive=True)
                       if not p.endswith(".orig"))
        files = files[:args.n_sample_per_root]
        rel = os.path.basename(r.rstrip("/").rstrip("\\"))
        print(f"\n[{rel}] checking {len(files)} files")
        stats = {"total": 0, "top_above_bot": 0, "vert_edge_ok": 0,
                 "front_yaw_pos": 0, "front_yaw_ge_rear": 0,
                 "front_closer": 0, "lr_ok": 0, "all_ok": 0,
                 "skipped": 0}
        for fp in files:
            bak = fp + ".orig"
            src = bak if os.path.exists(bak) else fp
            try:
                d = json.load(open(src, "r", encoding="utf-8"))
            except Exception:
                stats["skipped"] += 1
                continue
            objs = d.get("objects", [])
            if not objs:
                stats["skipped"] += 1
                continue
            res = check_invariants(d, objs[0], alpha=args.alpha, beta=args.beta)
            if res is None:
                stats["skipped"] += 1
                continue
            stats["total"] += 1
            # all_ok uses front_yaw_ge_rear (more permissive than yaw>0)
            check_keys = ["top_above_bot", "vert_edge_ok", "front_yaw_ge_rear",
                          "front_closer", "lr_ok"]
            all_ok = True
            for k in check_keys:
                if res[k]:
                    stats[k] += 1
                else:
                    all_ok = False
            # also record strict yaw>0 separately
            if res["front_yaw_pos"]:
                stats["front_yaw_pos"] += 1
            if all_ok:
                stats["all_ok"] += 1
        total_stats[rel] = stats
        print(f"  Tested        : {stats['total']}  (skipped {stats['skipped']})")
        if stats["total"] > 0:
            for k in ["top_above_bot", "vert_edge_ok", "front_yaw_ge_rear",
                      "front_yaw_pos", "front_closer", "lr_ok", "all_ok"]:
                print(f"  {k:20s}: {stats[k]:4d}  ({100*stats[k]/stats['total']:5.1f}%)")


if __name__ == "__main__":
    main()
