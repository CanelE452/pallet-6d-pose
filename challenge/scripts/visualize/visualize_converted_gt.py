"""visualize_converted_gt.py — 변환된 GT JSON 의 9 keypoint overlay 시각화.

원본 PNG 위에 변환 후 projected_cuboid 9 corner 를 색깔 + idx 번호로 표시.
사용자가 0~7 위치 직접 확인 — "0 이 카메라 가까운 face 의 어디" 인지.

PNG/JSON pair 찾기:
  - 같은 폴더의 NNNNNN.png 와 NNNNNN.json
  - .orig 백업도 함께 비교 (변환 전 vs 변환 후)
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


# 9 keypoint 색깔 (BGR) — 0~3 = 가까운 face (warm color), 4~7 = 먼 face (cool)
KP_COLORS = [
    (0,   0, 255),   # 0  red
    (0, 128, 255),   # 1  orange
    (0, 255, 255),   # 2  yellow
    (0, 255,   0),   # 3  green
    (255, 255, 0),   # 4  cyan
    (255,  80, 0),   # 5  blue (lighter)
    (255,   0, 128), # 6  magenta
    (128,  0, 255),  # 7  purple
    (255, 255, 255), # 8  white centroid
]

KP_NAMES = [
    "0 NTL", "1 NTR", "2 NBR", "3 NBL",
    "4 FTL", "5 FTR", "6 FBR", "7 FBL",
    "8 ctr"
]

EDGES_NEAR = [(0, 1), (1, 2), (2, 3), (3, 0)]   # 가까운 face (0~3)
EDGES_FAR = [(4, 5), (5, 6), (6, 7), (7, 4)]    # 먼 face (4~7)
EDGES_VERT = [(0, 4), (1, 5), (2, 6), (3, 7)]   # near↔far edges


def draw_overlay(img, proj, label=""):
    """8 corner + centroid overlay + wireframe."""
    vis = img.copy()

    # wireframe: 먼 face → vertical → 가까운 face (가까운 면 강조)
    pts = []
    for i in range(8):
        if i < len(proj):
            pts.append((int(proj[i][0]), int(proj[i][1])))
        else:
            pts.append(None)
    for a, b in EDGES_FAR:
        if pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (100, 100, 100), 1, cv2.LINE_AA)
    for a, b in EDGES_VERT:
        if pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
    for a, b in EDGES_NEAR:
        if pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (0, 220, 0), 3, cv2.LINE_AA)   # 굵게

    # 점 + 번호
    for i in range(min(9, len(proj))):
        p = proj[i]
        if p[0] < 0 or p[1] < 0:
            continue
        px, py = int(p[0]), int(p[1])
        r = 8 if i < 4 else (6 if i < 8 else 5)
        cv2.circle(vis, (px, py), r, KP_COLORS[i], -1)
        cv2.circle(vis, (px, py), r + 2, (0, 0, 0), 1)
        cv2.putText(vis, KP_NAMES[i], (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)

    # title
    bar = np.full((30, vis.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(bar, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="debug/converted_gt_viz")
    ap.add_argument("--samples", nargs="+", default=[
        # 각 데이터셋 별 다양한 yaw / view sample
        "data/pallet/training_data/mixed_v8_train/000000.json",
        "data/pallet/training_data/mixed_v8_train/000100.json",
        "data/pallet/training_data/mixed_v8_train/000500.json",
        "challenge/data/02_synthetic/training/v1/part_000/train_palletobj_v1/000004.json",
        "challenge/data/02_synthetic/training/v1/part_000/train_palletobj_v1/000050.json",
        "challenge/data/02_synthetic/training/v2/part_000/train_palletobj_v2/000010.json",
    ])
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for json_rel in args.samples:
        json_path = json_rel if os.path.isabs(json_rel) else os.path.join(repo_root, json_rel)
        if not os.path.isfile(json_path):
            print(f"[SKIP] not found: {json_path}")
            continue
        png_path = json_path.replace(".json", ".png")
        if not os.path.isfile(png_path):
            print(f"[SKIP] no PNG: {png_path}")
            continue
        bak_path = json_path + ".orig"
        has_orig = os.path.isfile(bak_path)

        img = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] read fail: {png_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data_new = json.load(f)
        obj_new = data_new["objects"][0]
        proj_new = obj_new["projected_cuboid"]
        if len(proj_new) < 9 and obj_new.get("projected_cuboid_centroid"):
            proj_new = list(proj_new) + [obj_new["projected_cuboid_centroid"]]

        # 변환 후 overlay
        vis_new = draw_overlay(img, proj_new, label=f"AFTER  {os.path.basename(json_path)}")

        # 변환 전 (orig) 있으면 비교
        if has_orig:
            with open(bak_path, "r", encoding="utf-8") as f:
                data_old = json.load(f)
            obj_old = data_old["objects"][0]
            proj_old = obj_old["projected_cuboid"]
            if len(proj_old) < 9 and obj_old.get("projected_cuboid_centroid"):
                proj_old = list(proj_old) + [obj_old["projected_cuboid_centroid"]]
            vis_old = draw_overlay(img, proj_old, label=f"BEFORE {os.path.basename(json_path)}")
            combined = np.hstack([vis_old, vis_new])
            label_suffix = "_BEFORE_vs_AFTER"
        else:
            combined = vis_new
            label_suffix = "_unchanged"

        stem = os.path.splitext(os.path.basename(json_path))[0]
        parent = os.path.basename(os.path.dirname(json_path))
        out_path = os.path.join(out_dir, f"{parent}_{stem}{label_suffix}.png")
        cv2.imwrite(out_path, combined)
        saved.append(out_path)
        status = "SWAP" if has_orig else "no_swap"
        print(f"  [{status}] {parent}/{stem} -> {out_path}")

    print()
    print(f"[Done] {len(saved)} viz saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
