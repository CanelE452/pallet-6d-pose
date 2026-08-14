"""visualize_v4_preview.py — v4 PREVIEW (file 변경 X). 좌=원본, 우=v4 변환 후.

사용자 컨벤션 시각화 가이드:
  - TOP face (0,1,5,4) : warm color, 진한 빨강 wireframe
  - BOT face (3,2,6,7) : cool color, 진한 파랑 wireframe
  - FRONT face (0,1,2,3) : 굵은 노란 wireframe (camera 가까운 큰 면)
  - REAR face (4,5,6,7)  : 얇은 회색 wireframe
  - corner labels: "0 FTL" (Front-Top-Left), "1 FTR", "2 FBR", "3 FBL",
                   "4 RTL", "5 RTR", "6 RBR", "7 RBL", "8 ctr"

각 sample 에 대해 좌(BEFORE) + 우(AFTER) 패널 + 상단 정보 (perm, face area).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_camera_facing_v4 import compute_perm_v4, get_origin_3d, polyarea, apply_perm
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
from challenge.data_paths import get as _dp  # 경로 단일 출처


# 9 keypoint 색깔 (BGR)
# TOP=0,1,5,4 (warm)  BOT=3,2,6,7 (cool)
KP_COLORS = {
    0: (0, 0, 255),       # FTL red
    1: (0, 165, 255),     # FTR orange
    2: (0, 220, 255),     # FBR yellow-orange
    3: (0, 255, 100),     # FBL green
    4: (255, 100, 0),     # RTL light blue
    5: (255, 50, 100),    # RTR purple-blue
    6: (200, 0, 200),     # RBR magenta
    7: (255, 0, 100),     # RBL pink
    8: (255, 255, 255),   # centroid white
}
KP_NAMES = ["0 FTL", "1 FTR", "2 FBR", "3 FBL",
            "4 RTL", "5 RTR", "6 RBR", "7 RBL", "8 ctr"]

# Faces
FACE_TOP = [0, 1, 5, 4]      # user convention
FACE_BOT = [3, 2, 6, 7]
FACE_FRONT = [0, 1, 2, 3]
FACE_REAR = [4, 5, 6, 7]
FACE_LEFT = [0, 4, 7, 3]     # FTL, RTL, RBL, FBL
FACE_RIGHT = [1, 5, 6, 2]    # FTR, RTR, RBR, FBR


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
    """Draw closed quad with given face idx list."""
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


def draw_overlay(img, proj_list, title="", show_user_convention=True):
    """img + 9-keypoint overlay. proj_list: list of (x,y) or (x,y) None.
    show_user_convention: if True, color faces by FRONT/REAR/TOP/BOT semantics.
    """
    vis = img.copy()
    pts = [safe_int_pt(p) for p in proj_list]

    if show_user_convention:
        # REAR face: thin gray
        draw_face(vis, pts, FACE_REAR, (130, 130, 130), 2)
        # vertical edges
        for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(vis, pts[a], pts[b], (170, 170, 170), 2, cv2.LINE_AA)
        # FRONT face: bold yellow
        draw_face(vis, pts, FACE_FRONT, (0, 230, 230), 4)
        # TOP face: red overlay
        draw_face(vis, pts, FACE_TOP, (0, 0, 220), 2)
        # BOT face: blue overlay
        draw_face(vis, pts, FACE_BOT, (220, 80, 0), 2)

    # corners + labels
    for i in range(min(9, len(pts))):
        p = pts[i]
        if p is None:
            continue
        col = KP_COLORS[i]
        r = 10 if i < 4 else (8 if i < 8 else 6)
        cv2.circle(vis, p, r + 2, (0, 0, 0), -1)
        cv2.circle(vis, p, r, col, -1)
        cv2.circle(vis, p, r, (255, 255, 255), 1)
        # label
        cv2.putText(vis, KP_NAMES[i], (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, KP_NAMES[i], (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # title bar
    bar = np.full((50, vis.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(bar, title, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2, cv2.LINE_AA)
    out = np.vstack([bar, vis])
    return out


def add_legend(panel):
    """Top legend explaining user convention."""
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


def simulate_v4(obj):
    """원본 obj 에서 v4 적용 후 9 corner list 반환. (file 변경 안 함)"""
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None, None
    origin = get_origin_3d(obj)
    if origin is None:
        return None, None
    perm = compute_perm_v4(origin, proj)
    if perm is None:
        return None, None
    # apply
    if len(proj) >= 9:
        new_proj = apply_perm(proj, perm)
    else:
        new_proj = apply_perm(list(proj) + [obj.get("projected_cuboid_centroid")], perm)
    return new_proj, perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="debug/converted_gt_viz_v4")
    ap.add_argument("--samples", nargs="+", default=[
        # mixed_v8 (object-frame cuboid) — diverse view ratios
        "data/pallet/training_data/mixed_v8_train/000031.json",   # frontal (ratio~1)
        "data/pallet/training_data/mixed_v8_train/000110.json",   # slight oblique
        "data/pallet/training_data/mixed_v8_train/000123.json",   # mid oblique (identity)
        "data/pallet/training_data/mixed_v8_train/000131.json",   # oblique
        "data/pallet/training_data/mixed_v8_train/000226.json",   # strong oblique (identity)
        # v1 (world-frame keypoints)
        _dp("synth.v1") + "/part_000/train_palletobj_v1/000213.json",
        _dp("synth.v1") + "/part_000/train_palletobj_v1/000200.json",   # identity
        _dp("synth.v1") + "/part_000/train_palletobj_v1/000106.json",
        _dp("synth.v1") + "/part_000/train_palletobj_v1/000181.json",
        _dp("synth.v1") + "/part_000/train_palletobj_v1/000114.json",   # strong
        # v2
        _dp("synth.v2") + "/part_000/train_palletobj_v2/000296.json",
        _dp("synth.v2") + "/part_000/train_palletobj_v2/000072.json",   # identity
        _dp("synth.v2") + "/part_000/train_palletobj_v2/000251.json",
        _dp("synth.v2") + "/part_000/train_palletobj_v2/000110.json",
        _dp("synth.v2") + "/part_000/train_palletobj_v2/000194.json",   # strong
    ])
    ap.add_argument("--upscale", type=float, default=1.8, help="image upscale for readability")
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
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not d.get("objects"):
            print(f"[SKIP] no objects: {jp}")
            continue
        obj = d["objects"][0]
        proj_orig = proj_to_list9(obj)
        new_proj, perm = simulate_v4(obj)
        if new_proj is None:
            print(f"[SKIP] simulate fail: {jp}")
            continue

        # upscale image + projections for readability
        scale = args.upscale
        if scale != 1.0:
            new_h = int(img.shape[0] * scale)
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            proj_orig = [(p[0] * scale, p[1] * scale) if p is not None else None for p in proj_orig]
            new_proj = [(p[0] * scale, p[1] * scale) if p is not None else None for p in new_proj]

        # face areas for info bar
        po = np.array([p if p else (0, 0) for p in proj_orig[:8]])
        a03_old = polyarea(po[[0, 1, 2, 3]])
        a47_old = polyarea(po[[4, 5, 6, 7]])
        pn = np.array([p if p else (0, 0) for p in new_proj[:8]])
        a03_new = polyarea(pn[[0, 1, 2, 3]])
        a47_new = polyarea(pn[[4, 5, 6, 7]])

        # data root suffix
        parent = os.path.basename(os.path.dirname(jp))
        gp = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        if "training_data" in jp:
            tag = parent
        else:
            tag = f"{gp}_{parent}"

        title_old = f"BEFORE  {tag}/{os.path.basename(jp)}   area_0123={a03_old:.0f}  area_4567={a47_old:.0f}"
        title_new = f"AFTER (v4)  perm={perm[:8]}   area_0123={a03_new:.0f}  area_4567={a47_new:.0f}"

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
