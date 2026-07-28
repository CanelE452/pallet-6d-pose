"""visualize_v4_vs_v5.py — v4 결과 vs v5 결과 나란히 비교 (file 변경 X).

disagree sample 만 골라서:  좌=v4 결과 | 우=v5 결과

원본 (.orig 또는 현재 file) 에서 양쪽 perm 다시 계산 후 적용.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_camera_facing_v4 import compute_perm_v4
from convert_to_camera_facing_v5 import (
    compute_perm_v5, get_origin_3d, get_pts_cam, apply_perm
)
from visualize_v5_preview import (
    draw_overlay, add_legend, proj_to_list9, load_original
)


def simulate(data, obj, mode, alpha=0.7, beta=0.3):
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None, None
    origin = get_origin_3d(obj)
    if origin is None:
        return None, None
    if mode == "v4":
        perm = compute_perm_v4(origin, proj)
    elif mode == "v5":
        pts_cam = get_pts_cam(data, obj)
        if pts_cam is None:
            return None, None
        perm = compute_perm_v5(origin, pts_cam, proj, alpha=alpha, beta=beta)
    else:
        return None, None
    if perm is None:
        return None, None
    if len(proj) >= 9:
        new_proj = apply_perm(proj, perm)
    else:
        new_proj = apply_perm(list(proj) + [obj.get("projected_cuboid_centroid")], perm)
    return new_proj, perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="debug/converted_gt_viz_v4_vs_v5")
    ap.add_argument("--samples", nargs="+", default=[
        # 비교 결과에서 발견된 disagree 의 대표 sample
        # mixed_v8 — 다양한 disagree 패턴
        "data/pallet/training_data/mixed_v8_train/000000.json",   # all 8 diff
        "data/pallet/training_data/mixed_v8_train/000001.json",   # different
        "data/pallet/training_data/mixed_v8_train/000002.json",   # different
        "data/pallet/training_data/mixed_v8_train/000005.json",   # different
        "data/pallet/training_data/mixed_v8_train/000009.json",   # v4 identity vs v5 perm
        "data/pallet/training_data/mixed_v8_train/000010.json",
        "data/pallet/training_data/mixed_v8_train/000031.json",   # 비교 baseline
        "data/pallet/training_data/mixed_v8_train/000110.json",
        # v1
        "challenge/data/training/v1/part_000/train_palletobj_v1/000367.json",   # v4 perm vs v5 identity
        "challenge/data/training/v1/part_000/train_palletobj_v1/000486.json",
        "challenge/data/training/v1/part_000/train_palletobj_v1/000661.json",
        # v2
        "challenge/data/training/v2/part_000/train_palletobj_v2/000751.json",
        "challenge/data/training/v2/part_000/train_palletobj_v2/000904.json",
        "challenge/data/training/v2/part_002/train_palletobj_v2/002401.json",
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
        try:
            data, src = load_original(jp)
        except Exception as e:
            print(f"[SKIP] read fail {jp}: {e}")
            continue
        if not data.get("objects"):
            continue
        obj = data["objects"][0]
        v4_proj, v4_perm = simulate(data, obj, "v4")
        v5_proj, v5_perm = simulate(data, obj, "v5", alpha=args.alpha, beta=args.beta)
        if v4_proj is None or v5_proj is None:
            print(f"[SKIP] sim fail: {jp}")
            continue

        scale = args.upscale
        if scale != 1.0:
            new_h = int(img.shape[0] * scale)
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            v4_proj = [(p[0] * scale, p[1] * scale) if p is not None else None for p in v4_proj]
            v5_proj = [(p[0] * scale, p[1] * scale) if p is not None else None for p in v5_proj]

        parent = os.path.basename(os.path.dirname(jp))
        gp = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        if "training_data" in jp:
            tag = parent
        else:
            tag = f"{gp}_{parent}"

        diff_marker = "DIFF" if v4_perm[:8] != v5_perm[:8] else "same"
        title_v4 = f"v4 (area)   perm={v4_perm[:8]}   [{diff_marker}]"
        title_v5 = f"v5 (yaw+dist) perm={v5_perm[:8]}   alpha={args.alpha} beta={args.beta}"

        vis_v4 = draw_overlay(img, v4_proj, title=title_v4)
        vis_v5 = draw_overlay(img, v5_proj, title=title_v5)
        combined = np.hstack([vis_v4, vis_v5])
        combined = add_legend(combined)

        stem = os.path.splitext(os.path.basename(jp))[0]
        op = os.path.join(out_dir, f"{tag}_{stem}.png")
        cv2.imwrite(op, combined)
        saved.append(op)
        print(f"  [{diff_marker}] {tag}/{stem} -> {op}")

    print()
    print(f"[Done] {len(saved)} viz saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
