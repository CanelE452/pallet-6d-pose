"""verify_annotate_v4_fix.py — annotate_pnp.solve_pose 의 v4 강제 동작 검증.

V1: 임의 pose 합성 → 0~3 = camera-near face → solve_pose → projected_all 의 0~3 face
    가 image polygon area 최대 face 인지 확인 (v4 invariant)
V2: 의도적으로 0~3 = far face 로 클릭 → v4_warning=True 인지 + projected_all 이
    v4 로 강제되는지 확인
V3: 4 점만 vs 6 점 클릭 비교 → 4 점에서도 v4 강제가 정상 작동하는지

검증 PNG: data/pallet/results/annotate_v4_fix_verify/
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import (
    PALLET_DIMS, make_pallet_keypoints_3d, project_3d, solve_pose,
)
from convert_to_camera_facing_v4 import polyarea


_REPO = os.path.dirname(os.path.dirname(_HERE))
SEQ = os.path.join(_REPO, "data", "outside", "capturepallet09")
OUT_DIR = os.path.join(_REPO, "data", "pallet", "results", "annotate_v4_fix_verify")
os.makedirs(OUT_DIR, exist_ok=True)


def load_K():
    p = os.path.join(SEQ, "cam_K.txt")
    return np.loadtxt(p).reshape(3, 3)


def load_first_frame():
    import glob
    rgb = sorted(glob.glob(os.path.join(SEQ, "rgb", "*.png")))[0]
    return cv2.imread(rgb), rgb


def euler_R(yaw, pitch, roll):
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


# v4 invariant check: 0~3 face area >= other 3 vertical face areas
def v4_front_largest(proj_all):
    proj = np.array(proj_all[:8], dtype=np.float64)
    # 4 vertical faces (excluding TOP {0,1,5,4} and BOT {3,2,6,7})
    front = polyarea(proj[[0, 1, 2, 3]])     # 0,1,2,3 — FRONT
    rear  = polyarea(proj[[5, 4, 7, 6]])     # 4,5,6,7 — REAR (winding match)
    left  = polyarea(proj[[4, 0, 3, 7]])     # 0,4,7,3 — LEFT side
    right = polyarea(proj[[1, 5, 6, 2]])     # 1,5,6,2 — RIGHT side
    other_max = max(rear, left, right)
    return front, other_max, front >= other_max - 1.0


def draw_check(img, kps_2d, pose, title, status, save_path):
    vis = img.copy()
    h, w = vis.shape[:2]
    # cuboid wireframe
    if pose is not None:
        proj = pose["projected_all"]
        pts = [(int(p[0]), int(p[1])) if p[0] >= 0 else None for p in proj[:8]]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for k, (a, b) in enumerate(edges):
            if pts[a] and pts[b]:
                col = (0, 220, 0) if k < 4 else (0, 160, 0)
                cv2.line(vis, pts[a], pts[b], col, 3 if k < 4 else 1, cv2.LINE_AA)
        # corner labels (from projected_all, post v4 permute)
        for i, p in enumerate(proj[:8]):
            if p[0] < 0:
                continue
            c = (int(p[0]), int(p[1]))
            cv2.circle(vis, c, 5, (255, 255, 255), -1)
            cv2.circle(vis, c, 6, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # clicked points (red X for v2 case)
    for i, p in enumerate(kps_2d):
        if p is None:
            continue
        c = (int(p[0]), int(p[1]))
        cv2.drawMarker(vis, c, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(vis, f"clk{i}", (c[0] + 8, c[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    # title bar
    cv2.rectangle(vis, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    # status box
    col = (0, 255, 0) if "PASS" in status else (0, 0, 255)
    cv2.rectangle(vis, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                col, 2)
    cv2.imwrite(save_path, vis)


def _v4_largest_face_idx(proj_all):
    """4 vertical face 중 image area 최대 face 의 corner idx 4 개 (winding 순)."""
    proj = np.array(proj_all[:8], dtype=np.float64)
    faces = {
        'Zmax': [0, 1, 2, 3],
        'Zmin': [5, 4, 7, 6],
        'Xmin': [0, 3, 7, 4],
        'Xmax': [1, 5, 6, 2],
    }
    return max(faces.values(), key=lambda idx: polyarea(proj[idx]))


def _order_face_clicks(face_proj_4):
    """face 4 corner image 좌표 (4,2) → v4 click order (top-L, top-R, bot-R, bot-L)."""
    arr = np.array(face_proj_4)
    order_y = np.argsort(arr[:, 1])
    top2 = sorted(order_y[:2].tolist(), key=lambda i: arr[i, 0])  # L, R
    bot2 = sorted(order_y[2:].tolist(), key=lambda i: arr[i, 0])  # L, R
    return [list(face_proj_4[top2[0]]), list(face_proj_4[top2[1]]),
            list(face_proj_4[bot2[1]]), list(face_proj_4[bot2[0]])]


def run_v1(img, K):
    """V1: 합성 GT pose → 사용자가 v4 컨벤션에 맞게 (image-largest vertical face)
    를 0~3 으로 클릭 → solve_pose 결과 warning=False + invariant 통과 확인."""
    R = euler_R(yaw=35, pitch=-5, roll=0)
    t = np.array([0.0, 0.1, 2.2])
    kp3d = make_pallet_keypoints_3d(*PALLET_DIMS)
    proj_gt = project_3d(kp3d, R, t, K)

    # v4: image area 최대 vertical face → 그 face 의 corner 를 image LR/TB 순으로 0~3 클릭
    largest_idx = _v4_largest_face_idx(proj_gt)
    largest_proj = [proj_gt[i] for i in largest_idx]
    kps_2d = _order_face_clicks(largest_proj) + [None] * 5

    pose = solve_pose(kps_2d, K)
    assert pose is not None, "V1: PnP failed"

    front, other_max, inv_ok = v4_front_largest(pose["projected_all"])
    perm = pose.get("v4_perm")
    warn = pose.get("v4_warning")
    # PASS = invariant + no warning (사용자 v4 정합 클릭)
    ok = inv_ok and not warn
    status = (f"V1 {'PASS' if ok else 'FAIL'} | front_area={front:.0f} "
              f"max_other={other_max:.0f} | warning={warn} (expect False) | "
              f"perm={perm[:8] if perm else None}")
    print(status)
    draw_check(img, kps_2d, pose,
               "V1: user clicks v4-largest face (correct convention)",
               status,
               os.path.join(OUT_DIR, "V1_clicks_v4_largest_face.png"))
    return ok


def run_v2(img, K):
    """V2: v4-largest 가 아닌 작은 face 를 0~3 으로 클릭 → warning=True + v4 강제."""
    R = euler_R(yaw=35, pitch=-5, roll=0)
    t = np.array([0.0, 0.1, 2.2])
    kp3d = make_pallet_keypoints_3d(*PALLET_DIMS)
    proj_gt = project_3d(kp3d, R, t, K)

    # 4 vertical face 중 image area 최소 face 를 0~3 으로 클릭 (사용자 오류 시뮬)
    proj = np.array(proj_gt[:8])
    faces = {
        'Zmax': [0, 1, 2, 3],
        'Zmin': [5, 4, 7, 6],
        'Xmin': [0, 3, 7, 4],
        'Xmax': [1, 5, 6, 2],
    }
    smallest_idx = min(faces.values(), key=lambda idx: polyarea(proj[idx]))
    smallest_proj = [proj_gt[i] for i in smallest_idx]
    kps_2d = _order_face_clicks(smallest_proj) + [None] * 5

    pose = solve_pose(kps_2d, K)
    assert pose is not None, "V2: PnP failed"

    front, other_max, inv_ok = v4_front_largest(pose["projected_all"])
    perm = pose.get("v4_perm")
    warn = pose.get("v4_warning")
    # PASS = v4_warning=True AND post-permute invariant 통과
    ok = bool(warn) and inv_ok
    status = (f"V2 {'PASS' if ok else 'FAIL'} | warning={warn} (expect True) | "
              f"perm={perm[:8] if perm else None} | "
              f"front={front:.0f} max_other={other_max:.0f}")
    print(status)
    draw_check(img, kps_2d, pose,
               "V2: user clicks SMALL face (wrong) — expect warning",
               status,
               os.path.join(OUT_DIR, "V2_clicks_small_face_wrong.png"))
    return ok


def run_v3(img, K):
    """V3: 4 점 (face-only) vs 6 점 (face + 2 top-far) 비교 → 둘 다 v4 invariant 통과.
    사용자가 v4-largest face 를 0~3 으로 정확히 클릭한 시나리오."""
    R = euler_R(yaw=25, pitch=-3, roll=0)
    t = np.array([0.0, 0.05, 2.5])
    kp3d = make_pallet_keypoints_3d(*PALLET_DIMS)
    proj_gt = project_3d(kp3d, R, t, K)

    largest_idx = _v4_largest_face_idx(proj_gt)
    largest_proj = [proj_gt[i] for i in largest_idx]
    front_clicks = _order_face_clicks(largest_proj)

    # 4 점만 (0~3 = v4-largest face)
    kps_4 = front_clicks + [None] * 5
    pose_4 = solve_pose(kps_4, K)
    front_4, other_4, ok_4 = (v4_front_largest(pose_4["projected_all"])
                              if pose_4 else (0, 0, False))
    warn_4 = pose_4.get("v4_warning") if pose_4 else None

    # 6 점: 0~3 = v4-largest, 4/5 는 pose_4 의 projected_all 4/5 가져옴 (correct rear top)
    if pose_4 is not None:
        p45 = pose_4["projected_all"]
        kps_6 = front_clicks + [list(p45[4]), list(p45[5])] + [None] * 3
    else:
        kps_6 = front_clicks + [None] * 5
    pose_6 = solve_pose(kps_6, K)
    front_6, other_6, ok_6 = (v4_front_largest(pose_6["projected_all"])
                              if pose_6 else (0, 0, False))
    warn_6 = pose_6.get("v4_warning") if pose_6 else None
    # 6 점에서는 사용자 클릭 4,5 가 PnP 가 풀어준 새 해의 4,5 와 정확히 매칭 안 될
    # 수 있음 (R, t 가 미세 변동). warning 은 0~3 만 보므로 4,5 일치는 별도 관심.
    # V3 의 목적은 "4 점 vs 6 점 둘 다 v4 invariant 통과" 확인.

    ok = ok_4 and ok_6
    status = (f"V3 {'PASS' if ok else 'FAIL'} | "
              f"4pt: front={front_4:.0f} other={other_4:.0f} ok={ok_4} | "
              f"6pt: front={front_6:.0f} other={other_6:.0f} ok={ok_6}")
    print(status)
    # 두 결과를 side-by-side
    img_4 = img.copy()
    img_6 = img.copy()
    draw_check(img_4, kps_4, pose_4, "V3a: 4 clicks (0-3 only)",
               f"4pt front={front_4:.0f} max_other={other_4:.0f} "
               f"ok={ok_4}",
               os.path.join(OUT_DIR, "V3a_4_clicks.png"))
    draw_check(img_6, kps_6, pose_6, "V3b: 6 clicks (0-5)",
               f"6pt front={front_6:.0f} max_other={other_6:.0f} "
               f"ok={ok_6}",
               os.path.join(OUT_DIR, "V3b_6_clicks.png"))

    a = cv2.imread(os.path.join(OUT_DIR, "V3a_4_clicks.png"))
    b = cv2.imread(os.path.join(OUT_DIR, "V3b_6_clicks.png"))
    combined = np.hstack([a, b])
    cv2.putText(combined, status, (10, combined.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "V3_4vs6_compare.png"), combined)
    return ok


def main():
    K = load_K()
    img, src = load_first_frame()
    print(f"K shape: {K.shape}, image: {img.shape}, src: {src}")
    print(f"Output dir: {OUT_DIR}")
    print()
    r1 = run_v1(img, K)
    r2 = run_v2(img, K)
    r3 = run_v3(img, K)
    print()
    print("=" * 60)
    print(f"V1 (correct near click) : {'PASS' if r1 else 'FAIL'}")
    print(f"V2 (wrong far click)    : {'PASS' if r2 else 'FAIL'}")
    print(f"V3 (4pt vs 6pt)         : {'PASS' if r3 else 'FAIL'}")
    print(f"OVERALL                 : {'PASS' if (r1 and r2 and r3) else 'FAIL'}")


if __name__ == "__main__":
    main()
