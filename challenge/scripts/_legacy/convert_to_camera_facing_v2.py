"""convert_to_camera_facing_v2.py — JSON GT 를 image-plane 기준 camera-facing 으로.

v1 (face swap 만) 의 문제: cuboid frame 의 TL 이 image 좌상 보장 안 됨.
v2: 모든 frame 의 projected_cuboid 8 corner 를 다음으로 재배열.

  0: near 좌상   1: near 우상
  2: near 우하   3: near 좌하
  4: far  좌상   5: far  우상
  6: far  우하   7: far  좌하

Step:
  1. face area 비교 (polygon_area) → near 4 corner + far 4 corner 결정
  2. 각 face 의 4 corner 를 image y 정렬 → 위 2 (top) + 아래 2 (bottom)
  3. top 2 의 x 정렬 → 좌상=0/4, 우상=1/5
  4. bottom 2 의 x 정렬 → 우하=2/6, 좌하=3/7
  5. keypoint_in_frame 같이 permute

dataloader 의 학습 target = projected_cuboid + projected_cuboid_centroid 만 사용.
다른 필드 (pose_transform, cuboid, keypoints_3d_world) 는 그대로 — 학습 영향 X.

사용:
  python challenge/scripts/convert_to_camera_facing_v2.py --dry_run
  python challenge/scripts/convert_to_camera_facing_v2.py
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil

import numpy as np


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        x[0] * y[1] - x[1] * y[0] +
        x[1] * y[2] - x[2] * y[1] +
        x[2] * y[3] - x[3] * y[2] +
        x[3] * y[0] - x[0] * y[3]
    )


def compute_perm(projected_cuboid):
    """8 corner image position 으로 image-plane 기준 permutation 결정.

    return: (perm[0..8], swapped: bool) — perm[new_idx] = old_idx. perm[8]=8 (centroid).
    """
    pts = np.asarray(projected_cuboid[:8], dtype=np.float64)
    area_03 = polygon_area(pts[[0, 1, 2, 3]])
    area_47 = polygon_area(pts[[4, 5, 6, 7]])
    if area_03 >= area_47:
        near_old = [0, 1, 2, 3]
        far_old = [4, 5, 6, 7]
    else:
        near_old = [4, 5, 6, 7]
        far_old = [0, 1, 2, 3]

    def reorder(group):
        # image y 정렬 → 위 2 (top), 아래 2 (bottom)
        sorted_y = sorted(group, key=lambda i: pts[i][1])
        top = sorted_y[:2]
        bot = sorted_y[2:]
        # top 2: x 정렬 (좌, 우)
        top.sort(key=lambda i: pts[i][0])
        tl, tr = top[0], top[1]
        # bot 2: x 정렬 (좌, 우)
        bot.sort(key=lambda i: pts[i][0])
        bl, br = bot[0], bot[1]
        # 0=TL, 1=TR, 2=BR, 3=BL
        return [tl, tr, br, bl]

    near_perm = reorder(near_old)
    far_perm = reorder(far_old)
    perm = near_perm + far_perm + [8]
    # swapped: identity 가 아닐 때
    swapped = perm[:8] != [0, 1, 2, 3, 4, 5, 6, 7]
    return perm, swapped


def apply_perm(arr, perm):
    return [arr[perm[i]] if perm[i] < len(arr) else None for i in range(len(perm))]


def convert_json(json_path, dry_run=False):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"read_fail: {e}"

    objs = data.get("objects", [])
    if not objs:
        return False, "no_objects"

    n_swapped = 0
    for obj in objs:
        proj = obj.get("projected_cuboid")
        if proj is None or len(proj) < 8:
            continue
        try:
            perm, swapped = compute_perm(proj)
        except Exception as e:
            continue
        if not swapped:
            continue
        # 9 corner (+centroid 가 있으면) permute
        if len(proj) >= 9:
            obj["projected_cuboid"] = apply_perm(proj, perm)
        else:
            obj["projected_cuboid"] = apply_perm(proj + [None], perm)[:8]
        # keypoint_in_frame permute
        kif = obj.get("keypoint_in_frame")
        if kif and len(kif) >= 8:
            ext = list(kif) + ([True] * (9 - len(kif)))
            obj["keypoint_in_frame"] = apply_perm(ext, perm)[:len(kif)]
        n_swapped += 1

    if n_swapped == 0:
        return False, "no_perm_needed"
    if dry_run:
        return True, f"would_perm_{n_swapped}_obj"
    bak = json_path + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(json_path, bak)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return True, f"perm_{n_swapped}_obj"


def gather(roots):
    files = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*.json"), recursive=True):
            if p.endswith(".orig"):
                continue
            files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "data/pallet/training_data/mixed_v8_train",
        "challenge/data/training/v1",
        "challenge/data/training/v2",
    ])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_sample", type=int, default=0)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]
    files = gather(roots)
    print(f"[Found] {len(files)} JSON files. Mode={'DRY' if args.dry_run else 'WRITE'}")
    if args.n_sample > 0:
        files = files[:args.n_sample]

    stats = {"perm": 0, "no_perm": 0, "no_obj": 0, "read_fail": 0, "other": 0}
    for i, fp in enumerate(files):
        changed, reason = convert_json(fp, dry_run=args.dry_run)
        if changed:
            stats["perm"] += 1
        elif reason == "no_perm_needed":
            stats["no_perm"] += 1
        elif reason == "no_objects":
            stats["no_obj"] += 1
        elif reason.startswith("read_fail"):
            stats["read_fail"] += 1
        else:
            stats["other"] += 1
        if i % 3000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}] perm={stats['perm']} no_perm={stats['no_perm']}")

    total = len(files)
    print()
    print("─" * 60)
    print(f"  Total       : {total}")
    print(f"  Permuted    : {stats['perm']}  ({100*stats['perm']/max(total,1):.1f}%)")
    print(f"  Already OK  : {stats['no_perm']}  ({100*stats['no_perm']/max(total,1):.1f}%)")
    print(f"  No objects  : {stats['no_obj']}")
    print(f"  Read fail   : {stats['read_fail']}")


if __name__ == "__main__":
    main()
