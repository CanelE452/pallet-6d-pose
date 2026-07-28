"""visualize_v2_diverse.py — 다양한 view (정면/옆/비스듬/멀리) v2 preview 시각화."""
from __future__ import annotations
import json
import os
import sys
import random

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from convert_to_camera_facing_v2 import compute_perm
from visualize_converted_gt import draw_overlay


def main():
    random.seed(42)
    np.random.seed(42)
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    out_dir = os.path.join(repo_root, "debug", "converted_gt_viz_v2_diverse")
    os.makedirs(out_dir, exist_ok=True)

    # 각 데이터셋에서 다양한 frame
    samples = []
    # mixed_v8: 9000 frame 균등 sample 6 개
    for i in [200, 1500, 3000, 4500, 6000, 8000]:
        samples.append(f"data/pallet/training_data/mixed_v8_train/{i:06d}.json")
    # v1: 각 part 마다 다른 frame (~10K, 10 part)
    for part, idx in [(0, 200), (1, 500), (3, 800), (5, 100), (7, 600), (9, 300)]:
        samples.append(f"challenge/data/training/v1/part_{part:03d}/train_palletobj_v1/{idx:06d}.json")
    # v2: 같은 방식
    for part, idx in [(0, 150), (2, 400), (4, 700), (6, 250), (8, 550)]:
        samples.append(f"challenge/data/training/v2/part_{part:03d}/train_palletobj_v2/{idx:06d}.json")

    saved = []
    for rel in samples:
        json_path = os.path.join(repo_root, rel)
        if not os.path.isfile(json_path):
            # fallback: 그 part 의 첫 file
            parent = os.path.dirname(json_path)
            if os.path.isdir(parent):
                jsons = sorted([p for p in os.listdir(parent) if p.endswith(".json") and not p.endswith(".orig")])
                if jsons:
                    json_path = os.path.join(parent, jsons[0])
                else:
                    print(f"[SKIP] no JSON in {parent}")
                    continue
            else:
                print(f"[SKIP] no dir {parent}")
                continue
        png_path = json_path.replace(".json", ".png")
        if not os.path.isfile(png_path):
            print(f"[SKIP] no PNG {png_path}")
            continue
        img = cv2.imread(png_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = data["objects"][0]
        proj_old = obj["projected_cuboid"]
        if len(proj_old) < 9 and obj.get("projected_cuboid_centroid"):
            proj_old = list(proj_old) + [obj["projected_cuboid_centroid"]]

        try:
            perm, swapped = compute_perm(proj_old)
        except Exception as e:
            print(f"[SKIP] perm fail {json_path}: {e}")
            continue
        proj_new = [proj_old[perm[i]] if perm[i] < len(proj_old) else None for i in range(9)]

        # 시각화: 좌(ORIGINAL) + 우(v2 PREVIEW)
        vis_old = draw_overlay(img, proj_old, label=f"ORIG {os.path.basename(json_path)}")
        vis_new = draw_overlay(img, proj_new, label=f"v2  swap={'Y' if swapped else 'N'}  perm={perm[:8]}")
        combined = np.hstack([vis_old, vis_new])

        parent = os.path.basename(os.path.dirname(json_path))
        if parent == "train_palletobj_v1" or parent == "train_palletobj_v2":
            # part 정보 포함
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
            parent = f"{grandparent}_{parent}"
        stem = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(out_dir, f"{parent}_{stem}.png")
        cv2.imwrite(out_path, combined)
        saved.append(out_path)
        print(f"  {parent}/{stem}: swap={swapped} perm={perm[:8]} → {os.path.basename(out_path)}")

    print()
    print(f"[Done] {len(saved)} preview saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
