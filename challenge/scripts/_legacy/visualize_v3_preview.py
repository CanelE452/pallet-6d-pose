"""visualize_v3_preview.py — v3 (정확한 사용자 컨벤션) preview 시각화."""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from convert_to_camera_facing_v3 import (
    get_pts_cam, get_origin_3d, compute_perm_v3_full
)
from visualize_converted_gt import draw_overlay


def main():
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    out_dir = os.path.join(repo_root, "debug", "converted_gt_viz_v3")
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for i in [200, 1500, 3000, 4500, 6000, 8000]:
        samples.append(f"data/pallet/training_data/mixed_v8_train/{i:06d}.json")
    for part, idx in [(0, 200), (1, 500), (3, 800), (5, 100), (7, 600), (9, 300)]:
        samples.append(f"challenge/data/training/v1/part_{part:03d}/train_palletobj_v1/{idx:06d}.json")
    for part, idx in [(0, 150), (2, 400), (4, 700), (6, 250), (8, 550)]:
        samples.append(f"challenge/data/training/v2/part_{part:03d}/train_palletobj_v2/{idx:06d}.json")

    saved = []
    for rel in samples:
        json_path = os.path.join(repo_root, rel)
        if not os.path.isfile(json_path):
            parent = os.path.dirname(json_path)
            if os.path.isdir(parent):
                jsons = sorted([p for p in os.listdir(parent) if p.endswith(".json") and not p.endswith(".orig")])
                if jsons:
                    json_path = os.path.join(parent, jsons[0])
                else:
                    continue
            else:
                continue
        png_path = json_path.replace(".json", ".png")
        if not os.path.isfile(png_path):
            continue
        img = cv2.imread(png_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = data["objects"][0]
        proj_old = obj["projected_cuboid"]
        if len(proj_old) < 9 and obj.get("projected_cuboid_centroid"):
            proj_old = list(proj_old) + [obj["projected_cuboid_centroid"]]

        origin = get_origin_3d(obj)
        pts_cam = get_pts_cam(data, obj)
        if origin is None or pts_cam is None:
            print(f"[SKIP] no 3D pose {json_path}")
            continue

        try:
            perm = compute_perm_v3_full(proj_old, origin, pts_cam)
        except Exception as e:
            print(f"[SKIP] perm fail {json_path}: {e}")
            continue
        proj_new = [proj_old[perm[i]] if perm[i] < len(proj_old) else None for i in range(9)]

        vis_old = draw_overlay(img, proj_old, label=f"ORIG {os.path.basename(json_path)}")
        vis_new = draw_overlay(img, proj_new, label=f"v3  perm={perm[:8]}")
        combined = np.hstack([vis_old, vis_new])

        parent = os.path.basename(os.path.dirname(json_path))
        if parent in ("train_palletobj_v1", "train_palletobj_v2"):
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
            parent = f"{grandparent}_{parent}"
        stem = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(out_dir, f"{parent}_{stem}.png")
        cv2.imwrite(out_path, combined)
        saved.append(out_path)
        print(f"  {parent}/{stem}: perm={perm[:8]} → {os.path.basename(out_path)}")

    print()
    print(f"[Done] {len(saved)} v3 preview saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
