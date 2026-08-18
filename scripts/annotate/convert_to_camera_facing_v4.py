"""convert_to_camera_facing_v4.py — 사용자 컨벤션 정확 구현 (2026-05-22).

사용자 컨벤션:
  TOP    (윗면, 3D height gravity-up)  = {0, 1, 5, 4}   ← origin frame z 큰 4 corner
  BOTTOM (아래면)                       = {3, 2, 6, 7}   ← origin frame z 작은 4 corner
  FRONT  (camera 가까운 큰 면, image polygon area 큰) = {0, 1, 2, 3}
  REAR                                                = {4, 5, 6, 7}

  0 = front-top-LEFT    (image 좌상)   1 = front-top-RIGHT  (image 우상)
  2 = front-bot-RIGHT   (image 우하)   3 = front-bot-LEFT   (image 좌하)
  4 = rear-top-LEFT                    5 = rear-top-RIGHT
  6 = rear-bot-RIGHT                   7 = rear-bot-LEFT

v1/v2/v3 와 다른 점:
  v1: face area 비교로 swap 만 → LR/TB 매핑 없음
  v2: image y 정렬로 top/bot → 카메라 각도에 따라 잘못
  v3: camera-frame z (depth) 작은 = near → v1/v2 dataset 의 USD camera convention
      (z < 0 = forward) 에서 부호 뒤집힘 발생. mixed_v8 (OpenCV z>0=forward) 만 통과,
      v1/v2 에선 near/far 가 정반대로 매핑됨.
  v4: 사용자 명시 "image polygon area 큰 face = FRONT" 그대로 구현. cam-frame 의존 X.

알고리즘:
  1. origin frame z (height) 정렬 → top4 (z 큰) + bot4 (z 작)
  2. top4 ↔ bot4 vertical pairing: 각 top vertex 의 horizontal (xy) 거리 가장 가까운 bot vertex
  3. top4 를 두 평행 edge 로 split: 가능한 3 가지 split 중 평행한 2 가지 (parallel cuboid edges).
     diagonal split (대각선) 은 origin 3D 에서 두 edge 의 unit vector dot product 절대값
     ~1.0 (평행) 인 것만 채택.
  4. 두 평행 split 중에서 polygon area 차이가 가장 큰 split = FRONT/REAR split
     (LR side split 은 양쪽 face 가 모두 비슷한 area).
     더 큰 area face = FRONT, 작은 = REAR.
  5. FRONT-TOP edge 안에서 image x 작은 = new_0 (FRONT-TOP-LEFT), 큰 = new_1
  6. vertical pairing 으로 bot edge 의 LEFT = top edge LEFT 의 pair → new_3, RIGHT → new_2
  7. REAR 도 같은 방식 → new_4, 5, 6, 7
  8. centroid (idx 8) 그대로

데이터셋별 origin frame:
  mixed_v8:  cuboid (object frame, z=height, z=0.15 top / 0.0 bot)
  v1/v2:     keypoints_3d_world (world frame, z=height, z=0.078 top / -0.042 bot)

학습 영향:
  변경: projected_cuboid, keypoint_in_frame
  보존: pose_transform, cuboid, keypoints_3d_world, location, quaternion (학습 target 아님)

사용:
  python challenge/scripts/annotate/convert_to_camera_facing_v4.py --dry_run
  python challenge/scripts/annotate/convert_to_camera_facing_v4.py --dry_run --n_sample 100
  python challenge/scripts/annotate/convert_to_camera_facing_v4.py    # 실제 변환 + .orig 백업
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
import shutil

import numpy as np
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
from challenge.data_paths import get as _dp  # 경로 단일 출처


def polyarea(pts):
    """Shoelace formula. pts: (4,2)."""
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        x[0] * y[1] - x[1] * y[0] +
        x[1] * y[2] - x[2] * y[1] +
        x[2] * y[3] - x[3] * y[2] +
        x[3] * y[0] - x[0] * y[3]
    )


def get_origin_3d(obj):
    """원본 3D corner (height axis = z). (8,3) or None."""
    if obj.get("cuboid") and len(obj["cuboid"]) >= 8:
        arr = np.array(obj["cuboid"][:8], dtype=np.float64)
        return arr if arr.shape == (8, 3) else None
    kp = obj.get("keypoints_3d_world")
    if kp and len(kp) >= 8:
        arr = np.array(kp[:8], dtype=np.float64)
        return arr if arr.shape == (8, 3) else None
    return None


def compute_perm_v4(origin, proj):
    """사용자 컨벤션 permutation 계산. perm[new_idx] = old_idx.

    origin: (8,3) origin frame (object or world), z = height (gravity-up)
    proj:   (8,2) image
    return: perm[9] list of int, or None on degenerate input
    """
    origin = np.asarray(origin, dtype=np.float64)
    proj = np.asarray(proj[:8], dtype=np.float64)

    # 1. top/bot by 3D height
    z_order = np.argsort(origin[:, 2])[::-1]   # z desc
    top4 = [int(i) for i in z_order[:4]]
    bot4 = [int(i) for i in z_order[4:]]

    # 2. vertical pairing: top_i -> nearest bot in xy plane
    top_to_bot = {}
    used_bot = set()
    for ti in top4:
        candidates = sorted(bot4, key=lambda b: ((origin[ti, 0] - origin[b, 0]) ** 2 +
                                                  (origin[ti, 1] - origin[b, 1]) ** 2))
        # take nearest not yet used
        chosen = None
        for bj in candidates:
            if bj not in used_bot:
                chosen = bj
                break
        if chosen is None:
            return None
        top_to_bot[ti] = chosen
        used_bot.add(chosen)
    if len(used_bot) != 4:
        return None

    # 3. split top4 into 2 edges (3 possible ways)
    splits = [
        ((top4[0], top4[1]), (top4[2], top4[3])),
        ((top4[0], top4[2]), (top4[1], top4[3])),
        ((top4[0], top4[3]), (top4[1], top4[2])),
    ]
    # 4. parallel split + max-area-diff = FRONT/REAR split
    best_diff = -1.0
    best_front = None
    best_rear = None
    for (sA, sB) in splits:
        eA = origin[sA[1]] - origin[sA[0]]
        eB = origin[sB[1]] - origin[sB[0]]
        nA = np.linalg.norm(eA)
        nB = np.linalg.norm(eB)
        if nA < 1e-9 or nB < 1e-9:
            continue
        cos = abs(np.dot(eA, eB) / (nA * nB))
        if cos < 0.95:   # not parallel -> diagonal split
            continue
        # face = [top1, top2, bot of top2, bot of top1]  (going around)
        faceA = [sA[0], sA[1], top_to_bot[sA[1]], top_to_bot[sA[0]]]
        faceB = [sB[0], sB[1], top_to_bot[sB[1]], top_to_bot[sB[0]]]
        aA = polyarea(np.array([proj[i] for i in faceA]))
        aB = polyarea(np.array([proj[i] for i in faceB]))
        diff = abs(aA - aB)
        if diff > best_diff:
            best_diff = diff
            if aA >= aB:
                best_front = faceA
                best_rear = faceB
            else:
                best_front = faceB
                best_rear = faceA

    if best_front is None:
        return None

    # 5. FRONT-TOP edge LR by image x
    ft_t1, ft_t2 = best_front[0], best_front[1]
    if proj[ft_t1, 0] <= proj[ft_t2, 0]:
        ft_L, ft_R = ft_t1, ft_t2
    else:
        ft_L, ft_R = ft_t2, ft_t1

    rt_t1, rt_t2 = best_rear[0], best_rear[1]
    if proj[rt_t1, 0] <= proj[rt_t2, 0]:
        rt_L, rt_R = rt_t1, rt_t2
    else:
        rt_L, rt_R = rt_t2, rt_t1

    # 6. bot LR from vertical pairing
    fb_L = top_to_bot[ft_L]
    fb_R = top_to_bot[ft_R]
    rb_L = top_to_bot[rt_L]
    rb_R = top_to_bot[rt_R]

    # 7. construct perm: new 0=ft_L, 1=ft_R, 2=fb_R, 3=fb_L, 4=rt_L, 5=rt_R, 6=rb_R, 7=rb_L
    perm = [ft_L, ft_R, fb_R, fb_L, rt_L, rt_R, rb_R, rb_L, 8]
    return [int(x) for x in perm]


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

    n_perm = 0
    for obj in objs:
        proj = obj.get("projected_cuboid")
        if not proj or len(proj) < 8:
            continue
        origin = get_origin_3d(obj)
        if origin is None:
            return False, "no_3d_corner"
        perm = compute_perm_v4(origin, proj)
        if perm is None:
            return False, "degenerate"
        if perm[:8] == [0, 1, 2, 3, 4, 5, 6, 7]:
            continue   # identity

        # apply to projected_cuboid (8 + optional centroid)
        if len(proj) >= 9:
            obj["projected_cuboid"] = apply_perm(proj, perm)
        else:
            obj["projected_cuboid"] = apply_perm(list(proj) + [None], perm)[:8]
        # keypoint_in_frame
        kif = obj.get("keypoint_in_frame")
        if kif and len(kif) >= 8:
            ext = list(kif) + ([True] * (9 - len(kif)))
            obj["keypoint_in_frame"] = apply_perm(ext, perm)[:len(kif)]
        n_perm += 1

    if n_perm == 0:
        return False, "no_perm_needed"
    if dry_run:
        return True, f"would_perm_{n_perm}_obj"
    bak = json_path + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(json_path, bak)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return True, f"perm_{n_perm}_obj"


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
        _dp("synth.v1"),
        _dp("synth.v2"),
    ])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_sample", type=int, default=0)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]
    files = gather(roots)
    print(f"[Found] {len(files)} JSON. Mode={'DRY' if args.dry_run else 'WRITE'}")
    if args.n_sample > 0:
        files = files[:args.n_sample]
        print(f"[Sample] first {args.n_sample}")
    print()

    # per-root stats
    stats_per_root = {}
    for r in roots:
        stats_per_root[r] = {"perm": 0, "id": 0, "degenerate": 0, "no_3d": 0,
                              "no_obj": 0, "read_fail": 0, "total": 0}

    def which_root(fp):
        for r in roots:
            if fp.startswith(r):
                return r
        return roots[0]

    for i, fp in enumerate(files):
        rk = which_root(fp)
        stats_per_root[rk]["total"] += 1
        changed, reason = convert_json(fp, dry_run=args.dry_run)
        if changed:
            stats_per_root[rk]["perm"] += 1
        elif reason == "no_perm_needed":
            stats_per_root[rk]["id"] += 1
        elif reason == "degenerate":
            stats_per_root[rk]["degenerate"] += 1
        elif reason == "no_3d_corner":
            stats_per_root[rk]["no_3d"] += 1
        elif reason == "no_objects":
            stats_per_root[rk]["no_obj"] += 1
        elif reason.startswith("read_fail"):
            stats_per_root[rk]["read_fail"] += 1
        if i % 3000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}]")

    total = len(files)
    print()
    print("=" * 70)
    grand = {"perm": 0, "id": 0, "degenerate": 0, "no_3d": 0, "no_obj": 0,
             "read_fail": 0, "total": 0}
    for r, s in stats_per_root.items():
        rel = os.path.basename(r.rstrip("/").rstrip("\\"))
        if s["total"] == 0:
            continue
        print(f"\n[{rel}]")
        print(f"  Total          : {s['total']}")
        print(f"  Permuted       : {s['perm']:5d}  ({100*s['perm']/max(s['total'],1):.1f}%)")
        print(f"  Identity (OK)  : {s['id']:5d}  ({100*s['id']/max(s['total'],1):.1f}%)")
        print(f"  Degenerate     : {s['degenerate']:5d}")
        print(f"  No 3D corners  : {s['no_3d']:5d}")
        print(f"  No objects     : {s['no_obj']:5d}")
        print(f"  Read fail      : {s['read_fail']:5d}")
        for k in grand:
            grand[k] += s[k]

    print()
    print("-" * 70)
    print(f"[GRAND TOTAL]")
    print(f"  Total          : {grand['total']}")
    print(f"  Permuted       : {grand['perm']:5d}  ({100*grand['perm']/max(grand['total'],1):.1f}%)")
    print(f"  Identity (OK)  : {grand['id']:5d}  ({100*grand['id']/max(grand['total'],1):.1f}%)")
    print(f"  Degenerate     : {grand['degenerate']:5d}")
    print(f"  No 3D corners  : {grand['no_3d']:5d}")


if __name__ == "__main__":
    main()
