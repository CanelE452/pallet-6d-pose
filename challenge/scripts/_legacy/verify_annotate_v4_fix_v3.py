"""verify_annotate_v4_fix_v3.py — Method C (strict v4 invariant in PnP) 검증.

저장된 frame: challenge/data/capturepallet03_manual_gt/1778651569891693056.json
사용자 클릭 manual_kps 8 점 (idx 0~5 = 실제 클릭, 6/7 = 이전 PnP 자동 fill).

Outputs (data/pallet/results/annotate_v4_fix_v3/):
  v3_user_actual_clicks.png    — 8 점 + solve_pose v3 → wireframe + reproj + diag
  v3_4points_only.png          — 0~3 만 (4~5 제외) → fix v3 가 어떻게 풀어주는지
  v3_3D_pose_diagnostic.png    — cam-frame Z 분포 + 4 vertical face area 분포 (bar)
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import solve_pose, make_pallet_keypoints_3d  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(_HERE))
SRC_JSON = os.path.join(
    _REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.json")
SRC_IMG = os.path.join(
    _REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.png")
OUT = os.path.join(_REPO, "data/pallet/results/annotate_v4_fix_v3")
os.makedirs(OUT, exist_ok=True)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # FRONT (thick)
    (4, 5), (5, 6), (6, 7), (7, 4),   # REAR
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical
]
KP_COLORS = [
    (0, 0, 255), (0, 255, 255), (0, 128, 255), (0, 255, 0),
    (255, 128, 0), (255, 0, 0), (255, 0, 128), (128, 0, 255), (255, 255, 255),
]


def _draw_pose_overlay(img, kps_2d, pose, title, out_path):
    vis = img.copy()
    h, w = vis.shape[:2]
    if pose is not None:
        proj = pose["projected_all"]
        pts = [(int(p[0]), int(p[1])) if p[0] >= 0 else None for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            if pts[a] and pts[b]:
                col = (0, 220, 0) if k < 4 else (0, 160, 0)
                thick = 3 if k < 4 else 1
                cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
        for i, p in enumerate(proj[:8]):
            if p[0] < 0:
                continue
            c = (int(p[0]), int(p[1]))
            cv2.circle(vis, c, 4, (255, 255, 255), -1)
            cv2.circle(vis, c, 6, (0, 0, 0), 1)
            cv2.putText(vis, f"P{i}", (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # user clicks
    for i, p in enumerate(kps_2d[:9]):
        if p is None:
            continue
        c = (int(p[0]), int(p[1]))
        cv2.drawMarker(vis, c, KP_COLORS[i], cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(vis, f"c{i}", (c[0] + 8, c[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, KP_COLORS[i], 2)
    # title bar
    cv2.rectangle(vis, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    # bottom info
    if pose is not None:
        z_front, z_rear = pose.get("_v4_pts_cam_z", (0, 0))
        nf = pose.get("_v4_near_far_viol", 0)
        av = pose.get("_v4_area_viol", 0)
        am = pose.get("_v4_area_margin", 0)
        areas = pose.get("_v4_areas", {})
        line1 = (f"dims={pose['dims']} reproj={pose['reproj_error_px']:.2f}px "
                 f"perm={pose.get('v4_perm', [-1] * 9)[:8]} "
                 f"warn={pose.get('v4_warning', False)}")
        line2 = (f"z_front={z_front:.3f} z_rear={z_rear:.3f} nf_viol={nf} "
                 f"area_viol={av} margin={am:.0f}")
        line3 = ("areas: " + " ".join(f"{k}={v:.0f}" for k, v in areas.items()))
        cv2.rectangle(vis, (0, h - 56), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, line1, (8, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)
        cv2.putText(vis, line2, (8, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)
        cv2.putText(vis, line3, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)
    cv2.imwrite(out_path, vis)
    print(f"[saved] {out_path}")


def _diagnostic_plot(pose, kps_2d, out_path):
    """3D cam-frame Z dist + 4 vertical face area bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    R, t = pose["R"], pose["t"]
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    pts_cam = (R @ kp3d[:8].T).T + t
    z = pts_cam[:, 2]
    colors_z = ["tab:red" if i < 4 else "tab:blue" for i in range(8)]
    axes[0].bar(range(8), z, color=colors_z)
    axes[0].axhline(z[:4].mean(), color="tab:red", linestyle="--",
                    label=f"front(0-3) mean={z[:4].mean():.3f}")
    axes[0].axhline(z[4:].mean(), color="tab:blue", linestyle="--",
                    label=f"rear(4-7) mean={z[4:].mean():.3f}")
    axes[0].set_xlabel("corner idx (red=FRONT 0-3, blue=REAR 4-7)")
    axes[0].set_ylabel("camera-frame Z (m)")
    nf_ok = "OK" if z[:4].mean() < z[4:].mean() else "VIOL"
    axes[0].set_title(f"Near-Far invariant: front_Z < rear_Z [{nf_ok}]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    areas = pose.get("_v4_areas", {})
    names = list(areas.keys())
    vals = [areas[n] for n in names]
    colors_a = ["tab:green" if n == "FRONT" else "tab:gray" for n in names]
    axes[1].bar(names, vals, color=colors_a)
    front_v = areas.get("FRONT", 0)
    other_max = max((v for n, v in areas.items() if n != "FRONT"), default=0)
    a_ok = "OK" if front_v >= other_max - 1 else "VIOL"
    axes[1].set_ylabel("image polygon area (px^2, image-clipped)")
    axes[1].set_title(f"Area invariant: FRONT >= others [{a_ok}]")
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"fix v3 diagnostic — dims={pose['dims']} "
        f"reproj={pose['reproj_error_px']:.2f}px "
        f"nf_viol={pose.get('_v4_near_far_viol', '?')} "
        f"area_viol={pose.get('_v4_area_viol', '?')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    with open(SRC_JSON) as f:
        d = json.load(f)
    cam = d["camera_data"]["intrinsics"]
    K = np.array([
        [cam["fx"], 0, cam["cx"]],
        [0, cam["fy"], cam["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    manual = d["objects"][0]["manual_kps"]
    img0 = cv2.imread(SRC_IMG)
    if img0 is None:
        raise SystemExit(f"image not found: {SRC_IMG}")

    print(f"K=\n{K}")
    print(f"img shape: {img0.shape}")
    print(f"manual_kps (9): {manual}")
    print()

    # Case 1: 사용자 실제 클릭 8 점 (0~7) — 저장된 manual_kps
    print("=" * 60)
    print("[Case 1] User actual 8 clicks (0~7 from manual_kps)")
    print("=" * 60)
    kps_8 = list(manual[:8]) + [None]
    pose_8 = solve_pose(kps_8, K)
    if pose_8 is None:
        print("PnP FAILED")
    else:
        print(f"  dims={pose_8['dims']} reproj={pose_8['reproj_error_px']:.3f}")
        print(f"  perm={pose_8.get('v4_perm')}  warn={pose_8.get('v4_warning')}")
        print(f"  near_far_viol={pose_8.get('_v4_near_far_viol')} "
              f"area_viol={pose_8.get('_v4_area_viol')} "
              f"area_margin={pose_8.get('_v4_area_margin'):.1f}")
        print(f"  pts_cam_z: front={pose_8.get('_v4_pts_cam_z')[0]:.3f} "
              f"rear={pose_8.get('_v4_pts_cam_z')[1]:.3f}")
        print(f"  areas={pose_8.get('_v4_areas')}")
    _draw_pose_overlay(
        img0, kps_8, pose_8,
        "Case 1: user actual 8 clicks (manual_kps 0-7)",
        os.path.join(OUT, "v3_user_actual_clicks.png"))

    # Case 2: 0~3 만 (4~5 제외) — face-only 클릭으로 v3 가 face-flip ambiguity 풀어주는지
    print()
    print("=" * 60)
    print("[Case 2] 4 clicks (0~3 only - front face)")
    print("=" * 60)
    kps_4 = list(manual[:4]) + [None] * 5
    pose_4 = solve_pose(kps_4, K)
    if pose_4 is None:
        print("PnP FAILED")
    else:
        print(f"  dims={pose_4['dims']} reproj={pose_4['reproj_error_px']:.3f}")
        print(f"  perm={pose_4.get('v4_perm')}  warn={pose_4.get('v4_warning')}")
        print(f"  near_far_viol={pose_4.get('_v4_near_far_viol')} "
              f"area_viol={pose_4.get('_v4_area_viol')} "
              f"area_margin={pose_4.get('_v4_area_margin'):.1f}")
    _draw_pose_overlay(
        img0, kps_4, pose_4,
        "Case 2: only 4 front clicks (0-3, no rear)",
        os.path.join(OUT, "v3_4points_only.png"))

    # Case 3: 3D diagnostic plot (Case 1 의 pose 기준)
    print()
    print("=" * 60)
    print("[Case 3] 3D pose diagnostic plot")
    print("=" * 60)
    if pose_8 is not None:
        _diagnostic_plot(pose_8, kps_8,
                         os.path.join(OUT, "v3_3D_pose_diagnostic.png"))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY (fix v3 - Method C)")
    print("=" * 60)
    for name, pose, allow_nf in [
        ("Case 1 (8 pts)", pose_8, False),    # 8 점: nf=0 강제
        ("Case 2 (4 pts, planar)", pose_4, True),  # 4 coplanar: nf ambiguous (geometric limit)
    ]:
        if pose is None:
            print(f"{name}: PnP FAILED")
            continue
        nf = pose.get("_v4_near_far_viol", 0)
        av = pose.get("_v4_area_viol", 0)
        perm = pose.get("v4_perm", [-1] * 9)[:8]
        identity = perm == [0, 1, 2, 3, 4, 5, 6, 7]
        ok = (av == 0 and identity and (nf == 0 or allow_nf))
        verdict = "PASS" if ok else (
            f"FAIL (nf={nf} av={av} perm_identity={identity})")
        print(f"{name}: reproj={pose['reproj_error_px']:.2f}px  "
              f"nf_viol={nf}  area_viol={av}  "
              f"perm_identity={identity}  [{verdict}]")
    print(f"\n[done] outputs: {OUT}")


if __name__ == "__main__":
    main()
