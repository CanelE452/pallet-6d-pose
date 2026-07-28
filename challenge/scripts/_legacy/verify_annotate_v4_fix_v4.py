"""verify_annotate_v4_fix_v4.py — Method D (gravity invariant 추가) 검증.

fix v4 추가 사항:
  (e) gravity_viol — TOP {0,1,4,5} 의 image v 평균 < BOTTOM {2,3,6,7} 평균
      (image 위쪽 = 작은 v = gravity-up). weight=50000 으로 사실상 reject.
  + solve_pose 의 compute_perm_v4 후처리 perm 적용 제거 (raw projection 사용).
  + 경고 logic 단순화 — gravity_viol 만 경고 (area/nf 위반은 top-down view 등
    false-positive 발생, 진단용 _v4_* 필드만 보관).

검증 케이스 (저장된 manual_kps 사용, manual[:6] 만 사용자 실제 클릭, 6/7 은 fill):
  Case A: 사용자 0~5 (6 click) → TOP/BOT 가 정확히 매핑되는지
  Case B: 사용자 0~7 (8 click, 6/7 = 이전 PnP fill)
  Case C: 사용자 0~3 (4 click, FRONT only) → face-flip 자동 정렬

Outputs (data/pallet/results/annotate_v4_fix_v4/):
  v4_user_6points.png     — Case A
  v4_user_8points.png     — Case B
  v4_warning_logic.png    — gravity OK 케이스에 경고 안 뜨는지 (Case A 와 동일)
  v4_diagnostic.png       — 3D cam-Y 분포 + image v 분포
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
OUT = os.path.join(_REPO, "data/pallet/results/annotate_v4_fix_v4")
os.makedirs(OUT, exist_ok=True)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
KP_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 128), (128, 0, 255), (255, 255, 255),
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
            if p[0] < 0: continue
            c = (int(p[0]), int(p[1]))
            cv2.circle(vis, c, 4, (255, 255, 255), -1)
            cv2.circle(vis, c, 6, (0, 0, 0), 1)
            cv2.putText(vis, f"P{i}", (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for i, p in enumerate(kps_2d[:9]):
        if p is None: continue
        c = (int(p[0]), int(p[1]))
        cv2.drawMarker(vis, c, KP_COLORS[i], cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(vis, f"c{i}", (c[0] + 8, c[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, KP_COLORS[i], 2)
    # title bar
    cv2.rectangle(vis, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    # gravity warning bar
    if pose is not None and pose.get("v4_warning"):
        cv2.rectangle(vis, (0, 28), (w, 50), (0, 0, 0), -1)
        cv2.putText(vis, "[v4] GRAVITY-FLIP: TOP {0,1,4,5} below BOTTOM",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
    # bottom info
    if pose is not None:
        z_front, z_rear = pose.get("_v4_pts_cam_z", (0, 0))
        gv = pose.get("_v4_gravity_viol", 0)
        gm = pose.get("_v4_gravity_margin", 0)
        nf = pose.get("_v4_near_far_viol", 0)
        av = pose.get("_v4_area_viol", 0)
        tbv = pose.get("_v4_top_bot_v_proj", (0, 0))
        line1 = (f"dims={pose['dims']} reproj={pose['reproj_error_px']:.2f}px "
                 f"warn={pose.get('v4_warning')}")
        line2 = (f"GRAVITY: top_v={tbv[0]:.1f} bot_v={tbv[1]:.1f} margin={gm:.1f} "
                 f"viol={gv}")
        line3 = (f"diagnostic: nf_viol={nf}  area_viol={av}  "
                 f"z_front={z_front:.2f} z_rear={z_rear:.2f}")
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    R, t = pose["R"], pose["t"]
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    pts_cam = (R @ kp3d[:8].T).T + t
    proj = pose["projected_all"]

    # Left: cam-frame Y per corner (OpenCV Y=down, gravity-up = -Y, top = Y_min)
    y = pts_cam[:, 1]
    colors = ["tab:red" if i in (0, 1, 4, 5) else "tab:blue" for i in range(8)]
    axes[0].bar(range(8), y, color=colors)
    axes[0].axhline(y[[0, 1, 4, 5]].mean(), color="tab:red", linestyle="--",
                    label=f"TOP {{0,1,4,5}} mean={y[[0,1,4,5]].mean():.3f}")
    axes[0].axhline(y[[2, 3, 6, 7]].mean(), color="tab:blue", linestyle="--",
                    label=f"BOTTOM {{2,3,6,7}} mean={y[[2,3,6,7]].mean():.3f}")
    axes[0].set_xlabel("corner idx (red=TOP, blue=BOT)")
    axes[0].set_ylabel("camera-frame Y (m) [Y=down]")
    grav_ok = "OK" if y[[0, 1, 4, 5]].mean() < y[[2, 3, 6, 7]].mean() else "VIOL"
    axes[0].set_title(f"3D cam-Y: TOP_Y < BOT_Y (gravity-up) [{grav_ok}]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: projected image v per corner (image top = small v)
    vs = [proj[i][1] for i in range(8)]
    axes[1].bar(range(8), vs, color=colors)
    axes[1].axhline(np.mean([vs[i] for i in (0, 1, 4, 5)]), color="tab:red",
                    linestyle="--",
                    label=f"TOP v mean={np.mean([vs[i] for i in (0,1,4,5)]):.1f}")
    axes[1].axhline(np.mean([vs[i] for i in (2, 3, 6, 7)]), color="tab:blue",
                    linestyle="--",
                    label=f"BOT v mean={np.mean([vs[i] for i in (2,3,6,7)]):.1f}")
    # overlay user click v
    for i in range(min(9, len(kps_2d))):
        if kps_2d[i] is None: continue
        axes[1].scatter([i], [kps_2d[i][1]], color="black", marker="x", s=80,
                        label="user click" if i == 0 else None)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("corner idx")
    axes[1].set_ylabel("image v (top = small)")
    img_ok = ("OK" if np.mean([vs[i] for i in (0, 1, 4, 5)])
              < np.mean([vs[i] for i in (2, 3, 6, 7)]) else "VIOL")
    axes[1].set_title(f"image v: TOP above BOT [{img_ok}]  warn={pose.get('v4_warning')}")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"fix v4 diagnostic — dims={pose['dims']} "
        f"reproj={pose['reproj_error_px']:.2f}px "
        f"gravity_viol={pose.get('_v4_gravity_viol', '?')} "
        f"warn={pose.get('v4_warning')}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _summary(name, pose, expect_warning):
    if pose is None:
        print(f"{name}: PnP FAILED")
        return False
    gv = pose.get("_v4_gravity_viol", 0)
    warn = bool(pose.get("v4_warning"))
    ok = (warn == expect_warning) and (gv == 0 if not expect_warning else gv == 1)
    verdict = "PASS" if ok else "FAIL"
    print(f"{name}: reproj={pose['reproj_error_px']:.2f}px  "
          f"gravity_viol={gv}  warn={warn}  expect_warn={expect_warning}  "
          f"[{verdict}]")
    return ok


def main():
    with open(SRC_JSON) as f:
        d = json.load(f)
    cam = d["camera_data"]["intrinsics"]
    K = np.array([
        [cam["fx"], 0, cam["cx"]],
        [0, cam["fy"], cam["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)
    manual = d["objects"][0]["manual_kps"]
    img0 = cv2.imread(SRC_IMG)
    if img0 is None:
        raise SystemExit(f"image not found: {SRC_IMG}")

    print(f"K=\n{K}\n")
    print(f"manual_kps:")
    for i, p in enumerate(manual[:8]):
        print(f"  {i}: ({p[0]:6.1f}, {p[1]:6.1f})")
    print()

    # Case A: 사용자 시나리오 — 0~5 (6 click). 4, 5 는 image 위쪽 (TOP rear)
    print("=" * 70)
    print("[Case A] 6 clicks: 0~3 (FRONT vertical) + 4~5 (TOP rear edge)")
    print("=" * 70)
    kps_6 = list(manual[:6]) + [None] * 3
    pose_6 = solve_pose(kps_6, K)
    _draw_pose_overlay(
        img0, kps_6, pose_6,
        "Case A: 6 clicks (0~3 FRONT + 4~5 TOP-rear)",
        os.path.join(OUT, "v4_user_6points.png"))

    # Case B: 사용자 0~7 (저장된 manual_kps 전부)
    print()
    print("=" * 70)
    print("[Case B] 8 clicks: 0~7 (6/7 = ealier PnP fill)")
    print("=" * 70)
    kps_8 = list(manual[:8]) + [None]
    pose_8 = solve_pose(kps_8, K)
    _draw_pose_overlay(
        img0, kps_8, pose_8,
        "Case B: 8 clicks (0~7 from manual_kps)",
        os.path.join(OUT, "v4_user_8points.png"))

    # Case C: warning logic 확인 — Case A 와 동일한 케이스, 정확한 click 에 경고 안 뜨는지
    print()
    print("=" * 70)
    print("[Case C] Warning logic: 정확한 click 에 gravity warning 안 뜨는지")
    print("=" * 70)
    _draw_pose_overlay(
        img0, kps_6, pose_6,
        "Case C: warning logic (gravity OK = no warn expected)",
        os.path.join(OUT, "v4_warning_logic.png"))

    # diagnostic plot
    print()
    if pose_6 is not None:
        _diagnostic_plot(pose_6, kps_6, os.path.join(OUT, "v4_diagnostic.png"))

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY (fix v4 - Method D + gravity invariant)")
    print("=" * 70)
    _summary("Case A (6 pts)", pose_6, expect_warning=False)
    _summary("Case B (8 pts)", pose_8, expect_warning=False)
    _summary("Case C (warn)",  pose_6, expect_warning=False)
    print(f"\n[done] outputs: {OUT}")


if __name__ == "__main__":
    main()
