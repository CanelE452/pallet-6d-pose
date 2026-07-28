"""convert_to_camera_facing.py — 학습 데이터 GT JSON 의 0~3 ↔ 4~7 swap (2026-05-22).

Camera-facing convention 적용: 0~3 = 카메라 가까운 near face. 4~7 = far face.

판정 기준: **projected_cuboid 의 face area 비교** (perspective 본질).
  - polygon_area(0,1,2,3) < polygon_area(4,5,6,7) → 0~3 이 멀리 (작음) → swap 필요
  - 이 방법은 데이터셋 형식 차이 무시 (mixed_v8 vs v1/v2 모두 적용)
  - 단순 + robust + 3D pose 의존 없음

Swap 매핑: [4, 5, 6, 7, 0, 1, 2, 3, 8]
  - 0↔4, 1↔5, 2↔6, 3↔7 (LR/top-bottom 유지, near↔far swap)
  - centroid (8) 변경 없음

변경 필드:
  projected_cuboid                 : 8 corner swap
  keypoint_in_frame                : per-keypoint visibility swap (v1/v2)

보존 필드 (학습 영향 X 또는 obsolete):
  pose_transform, cuboid, keypoints_3d_world, location, quaternion
  → 향후 forklift 응용에서 R 사용 시 별도 매핑

사용:
  python challenge/scripts/convert_to_camera_facing.py --dry_run                # 통계만
  python challenge/scripts/convert_to_camera_facing.py --dry_run --n_sample 5   # 5 file 만 검증
  python challenge/scripts/convert_to_camera_facing.py                          # 실제 변환 (.orig 백업)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np


SWAP_MAP = [4, 5, 6, 7, 0, 1, 2, 3, 8]   # 0↔4, 1↔5, 2↔6, 3↔7, centroid 그대로


def polygon_area(pts):
    """4-corner polygon Shoelace formula. pts: (4, 2). 순서: TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        x[0] * y[1] - x[1] * y[0] +
        x[1] * y[2] - x[2] * y[1] +
        x[2] * y[3] - x[3] * y[2] +
        x[3] * y[0] - x[0] * y[3]
    )


def needs_swap(projected_cuboid):
    """projected_cuboid 의 face 0~3 vs 4~7 area 비교.
    0~3 이 더 작으면 (멀리) swap 필요."""
    pts = np.asarray(projected_cuboid[:8], dtype=np.float64)
    area_03 = polygon_area(pts[[0, 1, 2, 3]])
    area_47 = polygon_area(pts[[4, 5, 6, 7]])
    return area_03 < area_47


def apply_swap(arr, swap_map=SWAP_MAP):
    """arr 의 element 를 swap_map 순서로 재배열. None 안전."""
    out = []
    for i in range(len(swap_map)):
        j = swap_map[i]
        out.append(arr[j] if j < len(arr) else None)
    return out


def convert_json(json_path, dry_run=False):
    """단일 JSON 변환. (swapped: bool, reason: str) 반환."""
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
        if not needs_swap(proj):
            continue

        # 8 corner + centroid (if exist) swap
        new_proj = apply_swap(proj)
        # centroid 가 8번 idx 로 있으면 그대로, 없으면 별도 처리
        if len(proj) >= 9:
            obj["projected_cuboid"] = new_proj[:9]
        else:
            obj["projected_cuboid"] = new_proj[:8]

        # keypoint_in_frame (per-keypoint visibility) swap
        kif = obj.get("keypoint_in_frame")
        if kif and len(kif) >= 8:
            new_kif = apply_swap(kif)
            obj["keypoint_in_frame"] = new_kif[:len(kif)]

        n_swapped += 1

    if n_swapped == 0:
        return False, "no_swap_needed"

    if dry_run:
        return True, f"would_swap_{n_swapped}_obj"

    # 백업
    bak_path = json_path + ".orig"
    if not os.path.exists(bak_path):
        shutil.copy2(json_path, bak_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return True, f"swapped_{n_swapped}_obj"


def gather_jsons(roots):
    """여러 root 디렉토리에서 *.json (재귀) 수집. .orig 제외."""
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
    ap.add_argument("--dry_run", action="store_true", help="실제 변환 안 함, 통계만")
    ap.add_argument("--n_sample", type=int, default=0,
                    help=">0 이면 처음 N file 만 처리 (검증용)")
    ap.add_argument("--repo_root", default=None, help="기본 = script 의 grandparent")
    args = ap.parse_args()

    repo_root = args.repo_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]

    print(f"[Roots]")
    for r in roots:
        exists = "OK" if os.path.isdir(r) else "MISSING"
        print(f"  {exists}: {r}")
    print(f"[Mode] {'DRY RUN' if args.dry_run else 'WRITE'}  swap_map={SWAP_MAP}")
    print()

    files = gather_jsons(roots)
    print(f"[Found] {len(files)} JSON files")
    if args.n_sample > 0:
        files = files[:args.n_sample]
        print(f"[Sample] processing first {args.n_sample}")
    print()

    stats = {"swapped": 0, "no_swap_needed": 0, "no_objects": 0, "read_fail": 0,
             "no_projected_cuboid": 0, "other": 0}
    sample_outputs = []
    for i, fp in enumerate(files):
        changed, reason = convert_json(fp, dry_run=args.dry_run)
        if changed:
            stats["swapped"] += 1
        elif reason == "no_swap_needed":
            stats["no_swap_needed"] += 1
        elif reason == "no_objects":
            stats["no_objects"] += 1
        elif reason.startswith("read_fail"):
            stats["read_fail"] += 1
        else:
            stats["other"] += 1

        if args.n_sample > 0 and i < 5:
            sample_outputs.append((fp, changed, reason))

        if i % 2000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}] swapped={stats['swapped']} "
                  f"no_swap={stats['no_swap_needed']}")

    print()
    print("─" * 60)
    print("[결과]")
    total = len(files)
    print(f"  Total           : {total}")
    print(f"  Swapped         : {stats['swapped']} ({100*stats['swapped']/max(total,1):.1f}%)")
    print(f"  No swap needed  : {stats['no_swap_needed']} ({100*stats['no_swap_needed']/max(total,1):.1f}%)")
    print(f"  No objects      : {stats['no_objects']}")
    print(f"  Read fail       : {stats['read_fail']}")
    print(f"  Other           : {stats['other']}")
    print()
    if sample_outputs:
        print("[Sample 5 file 처리 결과]")
        for fp, changed, reason in sample_outputs:
            rel = os.path.relpath(fp, repo_root)
            marker = "★ SWAP" if changed else "  skip"
            print(f"  {marker}  {reason:25s}  {rel}")
    print()
    if args.dry_run:
        print("[DRY RUN] No file written. .orig 백업도 안 만듦. 실제 변환은 --dry_run 빼고 재실행.")
    else:
        print(f"[WRITE] 백업: .orig 파일 함께 보존 (rollback: rename .orig → .json)")


if __name__ == "__main__":
    main()
