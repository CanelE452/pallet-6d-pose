"""convert_to_camera_facing_v5.py — 3D pose 기반 FRONT 식별 (2026-05-22).

사용자 컨벤션 (v4 와 동일):
  TOP    (윗면) = {0, 1, 5, 4}     ← origin frame z (gravity-up) 큰 4 corner
  BOTTOM       = {3, 2, 6, 7}     ← origin frame z 작은 4 corner
  FRONT  (camera 가까운 큰 면) = {0, 1, 2, 3}
  REAR                       = {4, 5, 6, 7}
  0=FTL  1=FTR  2=FBR  3=FBL  4=RTL  5=RTR  6=RBR  7=RBL  (LR=image x)

v4 vs v5:
  v4: 4 vertical face 의 **image polygon area 최대 = FRONT**
      - projection + truncation 영향 받음 (truncated face 가 area 작아 보이는 risk)
  v5: 4 vertical face 의 **face normal vs camera line-of-sight yaw alignment**
      + face center 카메라 거리. truncation 영향 X.
      - score = α * yaw_alignment + β * distance_proximity
      - yaw_alignment = dot(face_normal_outward, -los_unit)  ∈ [-1, 1]
      - distance_proximity = mean_face_dist 가 최소면 1, 최대면 0 (face 들 간 normalize)
      - α=0.7, β=0.3 (yaw 가 주, distance 는 보조)
      - face_normal outward 방향: face center - body center 방향과 같은 쪽

데이터셋별 camera-frame 변환 (v3 와 동일):
  mixed_v8:  pose_transform (object→camera 4x4) 로 cuboid 변환
  v1/v2:     keypoints_3d_world + camera_data.{loc, quat}_worldframe (world→camera)

camera convention:
  mixed_v8: OpenCV (+z = forward, line-of-sight = +z)
  v1/v2:    USD/OpenGL (-z = forward, line-of-sight = -z)
  → 데이터셋별 los direction 자동 판별: pose_t[2] > 0 → +z; v1/v2 는 일관되게 +z (확인 필요)

  실제로는 face_normal 의 부호를 body-center-outward 로 강제하고,
  los = face_center_camframe / |face_center_camframe| (origin→face 방향) 로 계산.
  yaw_alignment = dot(face_normal_camframe_outward, -los) 로 통일 (convention 무관).

학습 영향:
  변경: projected_cuboid, keypoint_in_frame
  보존: pose_transform, cuboid, keypoints_3d_world, location, quaternion

사용:
  python challenge/scripts/convert_to_camera_facing_v5.py --dry_run
  python challenge/scripts/convert_to_camera_facing_v5.py --dry_run --n_sample 100
  python challenge/scripts/convert_to_camera_facing_v5.py    # 실제 변환 + .orig 백업
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil

import numpy as np


# ─────────── camera-frame 변환 (v3 와 동일) ───────────

def quat_xyzw_to_R(q):
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


def get_origin_3d(obj):
    """원본 frame (object/world) 8 corner. z = height. (8,3) or None."""
    if obj.get("cuboid") and len(obj["cuboid"]) >= 8:
        arr = np.array(obj["cuboid"][:8], dtype=np.float64)
        return arr if arr.shape == (8, 3) else None
    kp = obj.get("keypoints_3d_world")
    if kp and len(kp) >= 8:
        arr = np.array(kp[:8], dtype=np.float64)
        return arr if arr.shape == (8, 3) else None
    return None


def get_pts_cam(data, obj):
    """카메라 frame 8 corner (3D). (8,3) or None."""
    # case 1: mixed_v8 — pose_transform (object → camera)
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

    # case 2: v1/v2 — keypoints_3d_world + camera worldframe pose
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
        # world → camera: (p_world - cam_t) @ R_cam_to_world
        return ((kp_world - cam_t) @ R_cam_to_world)

    return None


# ─────────── v5 core ───────────

def _plane_normal(pts4):
    """4 corner (~planar) 의 normal vector (단위) — first 3 corner cross product."""
    p0, p1, p2 = pts4[0], pts4[1], pts4[2]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    return n / nn


def compute_perm_v5(origin, pts_cam, proj, alpha=0.7, beta=0.3):
    """3D pose 기반 사용자 컨벤션 permutation.

    origin: (8,3) origin frame, z = height
    pts_cam: (8,3) camera frame 3D
    proj: (8,2) image (LR 결정용)
    alpha: yaw_alignment weight
    beta:  distance_proximity weight
    return: perm[9] list or None
    """
    origin = np.asarray(origin, dtype=np.float64)
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    proj = np.asarray(proj[:8], dtype=np.float64)

    # 1. top/bot by 3D height
    z_order = np.argsort(origin[:, 2])[::-1]   # z desc
    top4 = [int(i) for i in z_order[:4]]
    bot4 = [int(i) for i in z_order[4:]]

    # 2. vertical pairing (xy 거리)
    top_to_bot = {}
    used_bot = set()
    for ti in top4:
        candidates = sorted(bot4, key=lambda b: ((origin[ti, 0] - origin[b, 0]) ** 2 +
                                                  (origin[ti, 1] - origin[b, 1]) ** 2))
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

    # 3. top4 → 3 split → 평행한 2 split 만 채택 (cuboid edge)
    splits = [
        ((top4[0], top4[1]), (top4[2], top4[3])),
        ((top4[0], top4[2]), (top4[1], top4[3])),
        ((top4[0], top4[3]), (top4[1], top4[2])),
    ]
    body_center_cam = pts_cam.mean(axis=0)
    cam_origin = np.zeros(3, dtype=np.float64)

    # 4. 각 split 의 2 face → 4 vertical face 후보 (평행 split 2개 × 2 face)
    candidates = []   # list of (face_idx_list, score, info_dict)
    for (sA, sB) in splits:
        eA = origin[sA[1]] - origin[sA[0]]
        eB = origin[sB[1]] - origin[sB[0]]
        nA = np.linalg.norm(eA)
        nB = np.linalg.norm(eB)
        if nA < 1e-9 or nB < 1e-9:
            continue
        cos = abs(np.dot(eA, eB) / (nA * nB))
        if cos < 0.95:   # not parallel -> diagonal
            continue
        faceA = [sA[0], sA[1], top_to_bot[sA[1]], top_to_bot[sA[0]]]
        faceB = [sB[0], sB[1], top_to_bot[sB[1]], top_to_bot[sB[0]]]
        candidates.append(faceA)
        candidates.append(faceB)

    if len(candidates) != 4:
        return None   # 평행 split 2 개가 안 잡힘 (degenerate)

    # 5. 각 face 의 yaw_alignment + distance_proximity 계산
    face_infos = []   # list of dict
    face_dists = []
    for face in candidates:
        face_pts_cam = pts_cam[face]   # (4,3)
        face_center = face_pts_cam.mean(axis=0)
        normal = _plane_normal(face_pts_cam)
        if normal is None:
            return None
        # outward: body_center → face_center 방향과 같은 쪽
        out_dir = face_center - body_center_cam
        if np.dot(normal, out_dir) < 0:
            normal = -normal
        # line-of-sight (camera origin → face center) unit
        los = face_center - cam_origin
        d = np.linalg.norm(los)
        if d < 1e-6:
            return None
        los_unit = los / d
        # yaw_alignment: face_normal_outward 가 카메라 향함 = dot(normal, -los_unit) > 0
        yaw_align = float(np.dot(normal, -los_unit))   # ∈ [-1, 1]
        face_infos.append({
            "face": face,
            "yaw_align": yaw_align,
            "dist": float(d),
        })
        face_dists.append(d)

    # distance_proximity: face 들 중 min~max 를 [0,1] 로 inverse-normalize
    dmin, dmax = min(face_dists), max(face_dists)
    drange = max(dmax - dmin, 1e-9)
    for info in face_infos:
        info["prox"] = float(1.0 - (info["dist"] - dmin) / drange)   # 가까울수록 1
        info["score"] = alpha * info["yaw_align"] + beta * info["prox"]

    # 6. score 최대 face = FRONT, 가장 score 작은 face = REAR (보통 face 정반대)
    face_infos_sorted = sorted(face_infos, key=lambda x: x["score"], reverse=True)
    best_front = face_infos_sorted[0]["face"]
    # REAR: best_front 의 정반대 face. 평행 split 안에서 best_front 의 짝.
    # candidates 는 [splitA_face0, splitA_face1, splitB_face0, splitB_face1] 순서 (위에서 append)
    front_idx = candidates.index(best_front)
    pair_idx = front_idx ^ 1   # 같은 split 안의 짝 (0<->1, 2<->3)
    best_rear = candidates[pair_idx]

    # 7. FRONT-TOP edge LR (image x)
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

    fb_L = top_to_bot[ft_L]
    fb_R = top_to_bot[ft_R]
    rb_L = top_to_bot[rt_L]
    rb_R = top_to_bot[rt_R]

    perm = [ft_L, ft_R, fb_R, fb_L, rt_L, rt_R, rb_R, rb_L, 8]
    return [int(x) for x in perm]


def apply_perm(arr, perm):
    return [arr[perm[i]] if perm[i] < len(arr) else None for i in range(len(perm))]


# ─────────── file IO ───────────

def convert_json(json_path, dry_run=False, alpha=0.7, beta=0.3):
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
        pts_cam = get_pts_cam(data, obj)
        if pts_cam is None:
            return False, "no_cam_pose"
        perm = compute_perm_v5(origin, pts_cam, proj, alpha=alpha, beta=beta)
        if perm is None:
            return False, "degenerate"
        if perm[:8] == [0, 1, 2, 3, 4, 5, 6, 7]:
            continue

        if len(proj) >= 9:
            obj["projected_cuboid"] = apply_perm(proj, perm)
        else:
            obj["projected_cuboid"] = apply_perm(list(proj) + [None], perm)[:8]
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
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [r if os.path.isabs(r) else os.path.join(repo_root, r) for r in args.roots]
    files = gather(roots)
    print(f"[Found] {len(files)} JSON. Mode={'DRY' if args.dry_run else 'WRITE'}  alpha={args.alpha} beta={args.beta}")
    if args.n_sample > 0:
        files = files[:args.n_sample]
        print(f"[Sample] first {args.n_sample}")
    print()

    stats_per_root = {}
    for r in roots:
        stats_per_root[r] = {"perm": 0, "id": 0, "degenerate": 0, "no_3d": 0,
                              "no_cam": 0, "no_obj": 0, "read_fail": 0, "total": 0}

    def which_root(fp):
        for r in roots:
            if fp.startswith(r):
                return r
        return roots[0]

    for i, fp in enumerate(files):
        rk = which_root(fp)
        stats_per_root[rk]["total"] += 1
        changed, reason = convert_json(fp, dry_run=args.dry_run,
                                       alpha=args.alpha, beta=args.beta)
        if changed:
            stats_per_root[rk]["perm"] += 1
        elif reason == "no_perm_needed":
            stats_per_root[rk]["id"] += 1
        elif reason == "degenerate":
            stats_per_root[rk]["degenerate"] += 1
        elif reason == "no_3d_corner":
            stats_per_root[rk]["no_3d"] += 1
        elif reason == "no_cam_pose":
            stats_per_root[rk]["no_cam"] += 1
        elif reason == "no_objects":
            stats_per_root[rk]["no_obj"] += 1
        elif reason.startswith("read_fail"):
            stats_per_root[rk]["read_fail"] += 1
        if i % 3000 == 0 and i > 0:
            print(f"  [{i}/{len(files)}]")

    print()
    print("=" * 70)
    grand = {"perm": 0, "id": 0, "degenerate": 0, "no_3d": 0, "no_cam": 0,
             "no_obj": 0, "read_fail": 0, "total": 0}
    for r, s in stats_per_root.items():
        if s["total"] == 0:
            continue
        rel = os.path.basename(r.rstrip("/").rstrip("\\"))
        print(f"\n[{rel}]")
        print(f"  Total          : {s['total']}")
        print(f"  Permuted       : {s['perm']:5d}  ({100*s['perm']/max(s['total'],1):.1f}%)")
        print(f"  Identity (OK)  : {s['id']:5d}  ({100*s['id']/max(s['total'],1):.1f}%)")
        print(f"  Degenerate     : {s['degenerate']:5d}")
        print(f"  No 3D corners  : {s['no_3d']:5d}")
        print(f"  No cam pose    : {s['no_cam']:5d}")
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
    print(f"  No cam pose    : {grand['no_cam']:5d}")


if __name__ == "__main__":
    main()
