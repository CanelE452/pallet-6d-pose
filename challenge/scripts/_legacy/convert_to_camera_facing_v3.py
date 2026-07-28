"""convert_to_camera_facing_v3.py — 정확한 사용자 컨벤션 (2026-05-22).

사용자 컨벤션:
  TOP    (윗면)         = {0, 1, 5, 4}  ← 3D height (z) 큰 4 corner
  BOTTOM (아래면)        = {3, 2, 6, 7}  ← 3D height (z) 작은 4 corner
  FRONT  (camera 가까운) = {0, 1, 2, 3}  ← camera-frame z 작은 4 corner
  REAR   (먼)           = {4, 5, 6, 7}

  0 = front-top-LEFT   1 = front-top-RIGHT
  2 = front-bot-RIGHT  3 = front-bot-LEFT
  4 = rear-top-LEFT    5 = rear-top-RIGHT
  6 = rear-bot-RIGHT   7 = rear-bot-LEFT

Algorithm:
  1. 8 corner 3D 좌표 + camera-frame 변환
  2. 3D z (height) 정렬 → top 4 + bottom 4
  3. camera z (depth) 기준 → top 4 안에서 가까운 2 / 먼 2, bottom 도 같이
  4. image x 정렬 → LR 매핑

데이터셋별 3D pose 처리:
  mixed_v8: cuboid + pose_transform (object → camera 4x4)
  v1/v2:    keypoints_3d_world + camera_data.{location, quaternion}_worldframe
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil

import numpy as np


def quat_xyzw_to_R(q):
    """quaternion (x, y, z, w) → 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def get_pts_cam(data, obj):
    """카메라 frame 8 corner 좌표 반환. dataset 형식 자동 식별. (8,3) or None."""
    # case 1: mixed_v8 - cuboid + pose_transform
    if obj.get("cuboid") and obj.get("pose_transform"):
        cuboid = np.array(obj["cuboid"][:8], dtype=np.float64)
        if cuboid.shape != (8, 3):
            return None
        M = np.array(obj["pose_transform"], dtype=np.float64)
        if M.size != 16:
            return None
        M = M.reshape(4, 4)
        R = M[:3, :3]
        t = M[:3, 3]
        return (R @ cuboid.T).T + t

    # case 2: v1/v2 - keypoints_3d_world + camera worldframe pose
    kp_world_full = obj.get("keypoints_3d_world")
    if kp_world_full and len(kp_world_full) >= 8:
        kp_world = np.array(kp_world_full[:8], dtype=np.float64)
        cam = data.get("camera_data", {})
        cam_loc = cam.get("location_worldframe")
        cam_quat = cam.get("quaternion_xyzw_worldframe")
        if cam_loc is None or cam_quat is None:
            return None
        cam_t = np.array(cam_loc, dtype=np.float64)
        R_cam_to_world = quat_xyzw_to_R(cam_quat)
        # world → camera: R_world_to_cam @ (p_world - cam_t)
        return ((kp_world - cam_t) @ R_cam_to_world)

    return None


def compute_perm_v3(proj, pts_cam):
    """사용자 컨벤션 permutation 계산.

    return: (perm[9], ok: bool)
    """
    proj = np.asarray(proj[:8], dtype=np.float64)   # image
    pts_cam = np.asarray(pts_cam, dtype=np.float64)  # camera frame 3D (8, 3)
    # camera 좌표계: 일반적으로 z = depth (forward). 검증.
    # mixed_v8 pose_transform.t[2] = 3.62 > 0 → z=forward 가정 OK

    # 3D height axis 결정: 8 corner 중 z 값 분리 (또는 3D pose 의 원본 axis)
    # mixed_v8 cuboid: z=0.15/0.0 → 원본 z 가 height. 단 cuboid 좌표 그대로 사용.
    # v1/v2 kp_world: z=±height. 같음.
    # 즉 원본 3D 좌표의 z 가 height. 근데 camera frame 변환 후 z 는 depth 가 됨.
    # 해결: 원본 (object/world) 좌표에서 z height 분류 → top/bottom 결정. camera-frame
    # 은 near/far 만.

    # 원본 좌표 다시 가져옴 (top/bottom 결정용)
    # 함수 외부에서 origin_3d 도 전달받아야 정확. 여기서는 obj 통째 다시 받기 위해
    # 함수 signature 수정 필요. 일단 pts_cam 의 y axis 로 fallback (camera y = image 아래).
    # 더 robust: caller 에서 height_z 도 함께 전달.
    raise NotImplementedError("use compute_perm_v3_full instead")


def compute_perm_v3_full(proj, origin_3d, pts_cam):
    """proj: (8,2) image. origin_3d: (8,3) object/world frame (height axis = z).
       pts_cam: (8,3) camera frame (depth axis = z).

    return: (perm[9])
    """
    proj = np.asarray(proj[:8], dtype=np.float64)
    origin = np.asarray(origin_3d, dtype=np.float64)
    cam = np.asarray(pts_cam, dtype=np.float64)

    # 1. 3D height (origin z) 정렬 → top 4 (z 큰) + bottom 4 (z 작)
    z_sorted = np.argsort(origin[:, 2])[::-1]   # z 큰 순서
    top4 = list(z_sorted[:4])
    bot4 = list(z_sorted[4:])

    # 2. top 4 안에서 camera-frame z (depth) 작은 2 = front-top, 큰 2 = rear-top
    top4_sorted = sorted(top4, key=lambda i: cam[i, 2])
    front_top = top4_sorted[:2]   # 가까운 2
    rear_top = top4_sorted[2:]    # 먼 2

    # 3. bottom 4 도 같이
    bot4_sorted = sorted(bot4, key=lambda i: cam[i, 2])
    front_bot = bot4_sorted[:2]
    rear_bot = bot4_sorted[2:]

    # 4. image x 정렬 → LR
    def by_x(idx):
        return sorted(idx, key=lambda i: proj[i, 0])

    ft_sorted = by_x(front_top)   # 작=L=0, 큰=R=1
    fb_sorted = by_x(front_bot)   # 작=L=3, 큰=R=2
    rt_sorted = by_x(rear_top)    # 작=L=4, 큰=R=5
    rb_sorted = by_x(rear_bot)    # 작=L=7, 큰=R=6

    new_0 = ft_sorted[0]
    new_1 = ft_sorted[1]
    new_3 = fb_sorted[0]
    new_2 = fb_sorted[1]
    new_4 = rt_sorted[0]
    new_5 = rt_sorted[1]
    new_7 = rb_sorted[0]
    new_6 = rb_sorted[1]

    perm = [new_0, new_1, new_2, new_3, new_4, new_5, new_6, new_7, 8]
    return perm


def get_origin_3d(obj):
    """원본 (object/world) frame 8 corner 좌표 (height axis = z). None if fail."""
    if obj.get("cuboid") and len(obj["cuboid"]) >= 8:
        return np.array(obj["cuboid"][:8], dtype=np.float64)
    kp = obj.get("keypoints_3d_world")
    if kp and len(kp) >= 8:
        return np.array(kp[:8], dtype=np.float64)
    return None


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
        if origin is None or origin.shape != (8, 3):
            return False, "no_3d_corner"

        pts_cam = get_pts_cam(data, obj)
        if pts_cam is None or pts_cam.shape != (8, 3):
            return False, "no_cam_pose"

        perm = compute_perm_v3_full(proj, origin, pts_cam)
        if perm[:8] == [0, 1, 2, 3, 4, 5, 6, 7]:
            continue   # identity, no change

        # apply: projected_cuboid (8 + opt centroid)
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
        "challenge/data/training/v1",
        "challenge/data/training/v2",
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

    stats = {"perm": 0, "no_perm": 0, "no_obj": 0, "read_fail": 0, "no_3d": 0, "no_cam": 0, "other": 0}
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
        elif reason == "no_3d_corner":
            stats["no_3d"] += 1
        elif reason == "no_cam_pose":
            stats["no_cam"] += 1
        else:
            stats["other"] += 1
        if i % 3000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}] perm={stats['perm']} no_perm={stats['no_perm']}")

    total = len(files)
    print()
    print("─" * 60)
    print(f"  Total          : {total}")
    print(f"  Permuted       : {stats['perm']}  ({100*stats['perm']/max(total,1):.1f}%)")
    print(f"  Already OK     : {stats['no_perm']}  ({100*stats['no_perm']/max(total,1):.1f}%)")
    print(f"  No 3D corners  : {stats['no_3d']}")
    print(f"  No cam pose    : {stats['no_cam']}")
    print(f"  No objects     : {stats['no_obj']}")
    print(f"  Read fail      : {stats['read_fail']}")


if __name__ == "__main__":
    main()
