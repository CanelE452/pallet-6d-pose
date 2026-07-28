"""visualize_v2_preview.py — v2 perm 을 file 안 바꾸고 sample 6 frame 에 시각적 preview.

기존 JSON file 은 그대로 두고, v2 의 image-plane perm 함수 호출 후 시각화.
사용자 OK 받으면 그 다음 convert_to_camera_facing_v2.py 로 실제 변환.
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from convert_to_camera_facing_v2 import compute_perm
from visualize_converted_gt import draw_overlay


def main():
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    out_dir = os.path.join(repo_root, "debug", "converted_gt_viz_v2")
    os.makedirs(out_dir, exist_ok=True)

    samples = [
        "data/pallet/training_data/mixed_v8_train/000000.json",
        "data/pallet/training_data/mixed_v8_train/000100.json",
        "data/pallet/training_data/mixed_v8_train/000500.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000004.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000050.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000010.json",
    ]
    saved = []
    for rel in samples:
        json_path = os.path.join(repo_root, rel)
        if not os.path.isfile(json_path):
            print(f"[SKIP] no JSON: {rel}")
            continue
        png_path = json_path.replace(".json", ".png")
        if not os.path.isfile(png_path):
            print(f"[SKIP] no PNG: {rel}")
            continue
        img = cv2.imread(png_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = data["objects"][0]
        proj_old = obj["projected_cuboid"]
        if len(proj_old) < 9 and obj.get("projected_cuboid_centroid"):
            proj_old = list(proj_old) + [obj["projected_cuboid_centroid"]]

        # v2 perm 적용 (file 변경 없음)
        perm, swapped = compute_perm(proj_old)
        proj_new = [proj_old[perm[i]] if perm[i] < len(proj_old) else None for i in range(9)]

        label_old = f"ORIGINAL (file 상태)  {os.path.basename(json_path)}"
        label_new = f"v2 PREVIEW  swap={'Y' if swapped else 'N'}  perm={perm[:8]}"
        vis_old = draw_overlay(img, proj_old, label=label_old)
        vis_new = draw_overlay(img, proj_new, label=label_new)
        combined = np.hstack([vis_old, vis_new])

        stem = os.path.splitext(os.path.basename(json_path))[0]
        parent = os.path.basename(os.path.dirname(json_path))
        out_path = os.path.join(out_dir, f"{parent}_{stem}_v2_PREVIEW.png")
        cv2.imwrite(out_path, combined)
        saved.append(out_path)
        print(f"  {parent}/{stem}: swap={swapped} perm={perm[:8]} → {out_path}")

    print()
    print(f"[Done] {len(saved)} v2 preview saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
