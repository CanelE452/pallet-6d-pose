"""visualize_v5_preview.py — v5 PREVIEW (file 변경 X). 좌=원본(.orig 또는 현재), 우=v5 적용.

v4 와 같은 시각화 스타일. 사용자 컨벤션:
  TOP (red) / BOT (blue) / FRONT (bold yellow) / REAR (gray) / 0=FTL ... 7=RBL
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_camera_facing_v5 import (
    compute_perm_v5, get_origin_3d, get_pts_cam, apply_perm
)


KP_COLORS = {
    0: (0, 0, 255),
    1: (0, 165, 255),
    2: (0, 220, 255),
    3: (0, 255, 100),
    4: (255, 100, 0),
    5: (255, 50, 100),
    6: (200, 0, 200),
    7: (255, 0, 100),
    8: (255, 255, 255),
}
KP_NAMES = ["0 FTL", "1 FTR", "2 FBR", "3 FBL",
            "4 RTL", "5 RTR", "6 RBR", "7 RBL", "8 ctr"]

FACE_TOP = [0, 1, 5, 4]
FACE_BOT = [3, 2, 6, 7]
FACE_FRONT = [0, 1, 2, 3]
FACE_REAR = [4, 5, 6, 7]


def safe_int_pt(p):
    if p is None:
        return None
    x, y = p[0], p[1]
    if x is None or y is None:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (int(round(x)), int(round(y)))


def draw_face(vis, pts, idx_list, color, thickness=2):
    polypts = []
    for i in idx_list:
        if i < len(pts) and pts[i] is not None:
            polypts.append(pts[i])
    if len(polypts) < 2:
        return
    for k in range(len(polypts)):
        a = polypts[k]
        b = polypts[(k + 1) % len(polypts)]
        cv2.line(vis, a, b, color, thickness, cv2.LINE_AA)


def draw_overlay(img, proj_list, title=""):
    vis = img.copy()
    pts = [safe_int_pt(p) for p in proj_list]
    draw_face(vis, pts, FACE_REAR, (130, 130, 130), 2)
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (170, 170, 170), 2, cv2.LINE_AA)
    draw_face(vis, pts, FACE_FRONT, (0, 230, 230), 4)
    draw_face(vis, pts, FACE_TOP, (0, 0, 220), 2)
    draw_face(vis, pts, FACE_BOT, (220, 80, 0), 2)
    for i in range(min(9, len(pts))):
        p = pts[i]
        if p is None:
            continue
        col = KP_COLORS[i]
        r = 10 if i < 4 else (8 if i < 8 else 6)
        cv2.circle(vis, p, r + 2, (0, 0, 0), -1)
        cv2.circle(vis, p, r, col, -1)
        cv2.circle(vis, p, r, (255, 255, 255), 1)
        cv2.putText(vis, KP_NAMES[i], (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, KP_NAMES[i], (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    bar = np.full((50, vis.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(bar, title, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2, cv2.LINE_AA)
    return np.vstack([bar, vis])


def add_legend(panel):
    h, w = 60, panel.shape[1]
    bar = np.full((h, w, 3), 20, dtype=np.uint8)
    txts = [
        ("TOP={0,1,5,4} (red)  BOT={3,2,6,7} (blue)", (10, 22), (200, 220, 220)),
        ("FRONT={0,1,2,3} (yellow, near)  REAR={4,5,6,7} (gray)", (10, 46), (200, 220, 220)),
    ]
    for t, p, c in txts:
        cv2.putText(bar, t, p, cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def proj_to_list9(obj):
    proj = obj.get("projected_cuboid", [])
    out = [tuple(p) if p is not None else None for p in proj]
    if len(out) < 9:
        c = obj.get("projected_cuboid_centroid")
        if c is not None:
            out = list(out) + [tuple(c)]
        else:
            out = list(out) + [None]
    return out


def load_original(json_path):
    """Load .orig if exists else current file. Returns (data, src_path)."""
    bak = json_path + ".orig"
    src = bak if os.path.exists(bak) else json_path
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f), src


def simulate_v5(data, obj, alpha=0.7, beta=0.3):
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None, None
    origin = get_origin_3d(obj)
    if origin is None:
        return None, None
    pts_cam = get_pts_cam(data, obj)
    if pts_cam is None:
        return None, None
    perm = compute_perm_v5(origin, pts_cam, proj, alpha=alpha, beta=beta)
    if perm is None:
        return None, None
    if len(proj) >= 9:
        new_proj = apply_perm(proj, perm)
    else:
        new_proj = apply_perm(list(proj) + [obj.get("projected_cuboid_centroid")], perm)
    return new_proj, perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="debug/converted_gt_viz_v5")
    ap.add_argument("--samples", nargs="+", default=[
        # v4 와 같은 sample 사용 (직접 비교)
        "data/pallet/training_data/mixed_v8_train/000031.json",
        "data/pallet/training_data/mixed_v8_train/000110.json",
        "data/pallet/training_data/mixed_v8_train/000123.json",
        "data/pallet/training_data/mixed_v8_train/000131.json",
        "data/pallet/training_data/mixed_v8_train/000226.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000213.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000200.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000106.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000181.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000114.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000296.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000072.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000251.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000110.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000194.json",
    ])
    ap.add_argument("--upscale", type=float, default=1.8)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for json_rel in args.samples:
        jp = json_rel if os.path.isabs(json_rel) else os.path.join(repo_root, json_rel)
        if not os.path.isfile(jp):
            print(f"[SKIP] not found: {jp}")
            continue
        pp = jp.replace(".json", ".png")
        if not os.path.isfile(pp):
            print(f"[SKIP] no PNG: {pp}")
            continue
        img = cv2.imread(pp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] read fail: {pp}")
            continue
        # always start from original (.orig if exists)
        try:
            data, src = load_original(jp)
        except Exception as e:
            print(f"[SKIP] read fail {jp}: {e}")
            continue
        if not data.get("objects"):
            print(f"[SKIP] no objects: {jp}")
            continue
        obj = data["objects"][0]
        proj_orig = proj_to_list9(obj)
        new_proj, perm = simulate_v5(data, obj, alpha=args.alpha, beta=args.beta)
        if new_proj is None:
            print(f"[SKIP] simulate fail: {jp}")
            continue

        scale = args.upscale
        if scale != 1.0:
            new_h = int(img.shape[0] * scale)
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            proj_orig = [(p[0] * scale, p[1] * scale) if p is not None else None for p in proj_orig]
            new_proj = [(p[0] * scale, p[1] * scale) if p is not None else None for p in new_proj]

        parent = os.path.basename(os.path.dirname(jp))
        gp = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        if "training_data" in jp:
            tag = parent
        else:
            tag = f"{gp}_{parent}"

        src_tag = "orig" if src.endswith(".orig") else "current"
        title_old = f"BEFORE ({src_tag})  {tag}/{os.path.basename(jp)}"
        title_new = f"AFTER (v5)  perm={perm[:8]}  alpha={args.alpha} beta={args.beta}"

        vis_old = draw_overlay(img, proj_orig, title=title_old)
        vis_new = draw_overlay(img, new_proj, title=title_new)
        combined = np.hstack([vis_old, vis_new])
        combined = add_legend(combined)

        stem = os.path.splitext(os.path.basename(jp))[0]
        op = os.path.join(out_dir, f"{tag}_{stem}.png")
        cv2.imwrite(op, combined)
        saved.append(op)
        changed = "CHANGED" if perm[:8] != [0, 1, 2, 3, 4, 5, 6, 7] else "identity"
        print(f"  [{changed}] {tag}/{stem} -> {op}")

    print()
    print(f"[Done] {len(saved)} viz saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
