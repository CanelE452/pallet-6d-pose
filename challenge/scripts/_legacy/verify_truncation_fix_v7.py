"""verify_truncation_fix_v7.py — 7 missing + 외삽 점 weighted PnP 검증.

사용자 시나리오 (capturepallet03/1778651573855324672):
  - 7 점 클릭: 0 (t 외삽, image 좌측 밖), 1, 2, 3 (x 외삽), 4, 5, 6
  - 7 = 미클릭 (image 좌측 더 밖, 외삽 어려움)
  - 8 (centroid) = 미클릭 (PnP projection 으로 fill)

v6 의 버그: `_reproj_err_dict` 가 projection 의 u<0 을 "behind camera" 로 오인
→ 1e6 error 채택 → 모든 candidate 가 잘못된 reproj 로 evaluated.
사용자가 클릭한 0 = (-16.7, 317.5) 가 image 좌측 밖이라, 정상 candidate 의 projection
도 u<0 으로 떨어짐 → 1e6 → selection 망가짐 → z=4m + 작은 cube 채택.

v7 fix:
  (1) `_reproj_err_dict`: u==-1 ∧ v==-1 sentinel 만 1e6, 그 외 u<0 은 valid.
  (2) `_solve_pose_single`: extrapolated_mask 받아 외삽 점 weight 0.3.
  (3) `_solve_pose_single`: cuboid bbox area < image area 1.5% candidate reject.

이 스크립트:
  [A] v6 (buggy) 재현: extrapolated_mask=None + 기존 logic 시뮬레이션
  [B] v7 patched: extrapolated_mask=[T,F,F,T,F,F,F,F,F] 로 호출
  [C] v7 + 7 도 외삽 (parallelogram BOTTOM(3,2,6,7) 에서 3,2,6 → 7 외삽)

비교:
  - z (m), reproj_clk (직접 click 만), cuboid bbox 크기
  - GT 와 비교 불가 (이 frame 은 manual GT 도 부정확) — 사용자 직접 click
    1,2,4,5,6 에 wireframe 가 잘 fit 하는지가 진짜 metric.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from annotate_pnp import (
    solve_pose, project_3d, make_pallet_keypoints_3d, PALLET_DIMS,
    parallelogram_extrapolate,
)
from annotate_draw import CUBOID_EDGES, KP_COLORS


REPO = os.path.dirname(os.path.dirname(_HERE))
SEQ_DIR = os.path.join(REPO, "data", "outside", "capturepallet03")
GT_DIR  = os.path.join(REPO, "challenge", "data", "capturepallet03_manual_gt")
FRAME   = "1778651573855324672"
OUT_DIR = os.path.join(REPO, "data", "pallet", "results", "annotate_truncation_v7")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Load ────────────────────────────────────────────────────────────────────
img = cv2.imread(os.path.join(SEQ_DIR, "rgb", f"{FRAME}.png"))
K = np.loadtxt(os.path.join(SEQ_DIR, "cam_K.txt"))
with open(os.path.join(GT_DIR, f"{FRAME}.json"), "r", encoding="utf-8") as f:
    saved = json.load(f)
saved_kps = saved["objects"][0]["manual_kps"]


# 사용자 시나리오 — 7 missing
clicks_7miss = [
    list(saved_kps[0]),   # 0  EXTRAP (t)
    list(saved_kps[1]),   # 1  CLICK
    list(saved_kps[2]),   # 2  CLICK
    list(saved_kps[3]),   # 3  EXTRAP (x)
    list(saved_kps[4]),   # 4  CLICK
    list(saved_kps[5]),   # 5  CLICK
    list(saved_kps[6]),   # 6  CLICK
    None,                 # 7  MISSING
    None,                 # 8  centroid PnP fill
]
extrap_mask = [True, False, False, True, False, False, False, False, False]

# (C) 7 도 외삽 — BOTTOM (3,2,6,7) 의 3 corner (3, 2, 6) 으로 7 외삽
pt7, fname, finds = parallelogram_extrapolate(clicks_7miss, 7)
clicks_7extrap = list(clicks_7miss)
extrap_mask_7e = list(extrap_mask)
if pt7 is not None:
    clicks_7extrap[7] = pt7
    extrap_mask_7e[7] = True
    print(f"[C] 7 외삽 face = {fname} {finds}  →  ({pt7[0]:.1f}, {pt7[1]:.1f})")
else:
    print(f"[C] 7 외삽 실패")
print()


def report_pose(label, kps, pose, K_arr, extrap):
    """pose 의 핵심 metric 출력 — z, click-only reproj, cuboid bbox."""
    if pose is None:
        print(f"  [{label}] PnP FAIL")
        return None
    R, t = pose["R"], pose["t"]
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    proj = project_3d(kp3d, R, t, K_arr)

    # click-only reproj (외삽 점 제외)
    click_idx = [i for i in range(8) if kps[i] is not None and not extrap[i]]
    errs_clk = []
    for i in click_idx:
        u, v = proj[i]
        if u == -1.0 and v == -1.0: continue
        errs_clk.append(float(np.hypot(u - kps[i][0], v - kps[i][1])))
    # extrap reproj
    extrap_idx = [i for i in range(8) if kps[i] is not None and extrap[i]]
    errs_ex = []
    for i in extrap_idx:
        u, v = proj[i]
        if u == -1.0 and v == -1.0: continue
        errs_ex.append(float(np.hypot(u - kps[i][0], v - kps[i][1])))

    proj8 = np.array(proj[:8], dtype=np.float64)
    bbox_w = float(proj8[:, 0].max() - proj8[:, 0].min())
    bbox_h = float(proj8[:, 1].max() - proj8[:, 1].min())

    print(f"  [{label}] z={t[2]:.2f}m  dims={pose['dims']}  "
          f"reproj_clk={np.mean(errs_clk):.2f}px ({len(errs_clk)} pts)  "
          f"reproj_ext={np.mean(errs_ex):.2f}px ({len(errs_ex)} pts)")
    print(f"        cuboid bbox = {bbox_w:.0f} × {bbox_h:.0f} px  "
          f"viol_sum={pose.get('_v6_viol_sum', -1)}  "
          f"strict={'PASS' if pose.get('_v6_strict_passed', False) else 'FAIL'}")
    return pose


print("=" * 92)
print("[A] saved JSON 결과 (v6 buggy 가 풀어 저장한 pose)")
print("=" * 92)
sav_pose_dim = (
    saved["objects"][0]["dimensions_m"]["width"],
    saved["objects"][0]["dimensions_m"]["depth"],
    saved["objects"][0]["dimensions_m"]["height"],
)
sav = {
    "R": np.array(saved["objects"][0]["pose_transform"])[:3, :3],
    "t": np.array(saved["objects"][0]["pose_transform"])[:3, 3],
    "dims": sav_pose_dim,
    "_v6_viol_sum": 0, "_v6_strict_passed": True,
}
report_pose("A v6 saved", clicks_7miss, sav, K, extrap_mask)
print()


print("=" * 92)
print("[B] v7 patched + extrapolated_mask=[T,F,F,T,F,F,F,F,F] (7 missing)")
print("=" * 92)
pose_B = solve_pose(clicks_7miss, K,
                    extrapolated_mask=extrap_mask,
                    img_shape=img.shape)
report_pose("B v7 7miss", clicks_7miss, pose_B, K, extrap_mask)
print()


print("=" * 92)
print("[C] v7 patched + 7 도 외삽 (parallelogram BOTTOM)")
print("=" * 92)
pose_C = solve_pose(clicks_7extrap, K,
                    extrapolated_mask=extrap_mask_7e,
                    img_shape=img.shape)
report_pose("C v7 7extr", clicks_7extrap, pose_C, K, extrap_mask_7e)
print()


# ─── Visualization ──────────────────────────────────────────────────────────
def draw_full(img_src, kps, pose, title, sub, extrap):
    vis = img_src.copy()
    if pose is not None:
        proj = pose["projected_all"] if "projected_all" in pose else \
               project_3d(make_pallet_keypoints_3d(*pose["dims"]),
                          pose["R"], pose["t"], K)
        pts = [(int(round(p[0])), int(round(p[1]))) for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            col = (0, 220, 0) if k < 4 else (0, 160, 0)
            thick = 3 if k < 4 else 1
            cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
        if len(proj) > 8 and not (proj[8][0] == -1.0 and proj[8][1] == -1.0):
            cv2.drawMarker(vis, (int(round(proj[8][0])), int(round(proj[8][1]))),
                           (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
    for i, p in enumerate(kps):
        if p is None:
            continue
        c = (int(round(p[0])), int(round(p[1])))
        if c[0] < -50 or c[0] >= vis.shape[1] + 50 or c[1] < -50 or c[1] >= vis.shape[0] + 50:
            continue
        # 외삽이면 점선 원, 직접 click 이면 채운 원
        if extrap[i]:
            cv2.circle(vis, c, 7, KP_COLORS[i], 1)   # 두꺼운 외곽선
            cv2.circle(vis, c, 9, (200, 200, 200), 1)
            cv2.putText(vis, f"{i}*", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)
        else:
            cv2.circle(vis, c, 6, KP_COLORS[i], -1)
            cv2.circle(vis, c, 8, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, sub, (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    # legend: * = extrapolated
    cv2.rectangle(vis, (0, vis.shape[0] - 22), (260, vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, "filled = click, * outlined = extrap",
                (5, vis.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return vis


def fmt_sub(pose, kps, extrap):
    if pose is None:
        return "PnP FAIL"
    R, t = pose["R"], pose["t"]
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    proj = project_3d(kp3d, R, t, K)
    click_idx = [i for i in range(8) if kps[i] is not None and not extrap[i]]
    errs_clk = []
    for i in click_idx:
        u, v = proj[i]
        if u == -1.0 and v == -1.0: continue
        errs_clk.append(float(np.hypot(u - kps[i][0], v - kps[i][1])))
    proj8 = np.array(proj[:8], dtype=np.float64)
    bw = float(proj8[:, 0].max() - proj8[:, 0].min())
    bh = float(proj8[:, 1].max() - proj8[:, 1].min())
    return (f"z={t[2]:.2f}m  reproj_click={np.mean(errs_clk):.1f}px  "
            f"bbox={bw:.0f}x{bh:.0f}")


vis_A = draw_full(img, clicks_7miss, sav,
                  "[A] v6 SAVED (BUGGY) — 7 missing",
                  fmt_sub(sav, clicks_7miss, extrap_mask),
                  extrap_mask)
vis_B = draw_full(img, clicks_7miss, pose_B,
                  "[B] v7 PATCHED — 7 missing (weighted PnP)",
                  fmt_sub(pose_B, clicks_7miss, extrap_mask),
                  extrap_mask)
vis_C = draw_full(img, clicks_7extrap, pose_C,
                  "[C] v7 PATCHED — 7 extrap (parallelogram BOTTOM)",
                  fmt_sub(pose_C, clicks_7extrap, extrap_mask_7e),
                  extrap_mask_7e)

cv2.imwrite(os.path.join(OUT_DIR, "v7_A_v6_saved.png"), vis_A)
cv2.imwrite(os.path.join(OUT_DIR, "v7_B_7missing.png"), vis_B)
cv2.imwrite(os.path.join(OUT_DIR, "v7_C_7extrap.png"), vis_C)

# 3-panel comparison
h, w = img.shape[:2]
grid = np.zeros((h, w * 3, 3), dtype=np.uint8)
grid[:, :w] = vis_A
grid[:, w:2*w] = vis_B
grid[:, 2*w:] = vis_C
cv2.imwrite(os.path.join(OUT_DIR, "v7_7missing.png"), grid)
print(f"saved: {os.path.join(OUT_DIR, 'v7_7missing.png')}  (3-panel A vs B vs C)")
print(f"saved: {os.path.join(OUT_DIR, 'v7_A_v6_saved.png')}")
print(f"saved: {os.path.join(OUT_DIR, 'v7_B_7missing.png')}")
print(f"saved: {os.path.join(OUT_DIR, 'v7_C_7extrap.png')}")


# ─── 회귀 테스트: 기존 truncation_fix frame (1778651569891693056) ──────────
print()
print("=" * 92)
print("REGRESSION TEST — capturepallet03/1778651569891693056 (012456 6pt, 3/7 missing)")
print("=" * 92)
FRAME_REG = "1778651569891693056"
img_reg = cv2.imread(os.path.join(SEQ_DIR, "rgb", f"{FRAME_REG}.png"))
with open(os.path.join(GT_DIR, f"{FRAME_REG}.json"), "r", encoding="utf-8") as f:
    gt_reg = json.load(f)
GT_KPS_REG = gt_reg["objects"][0]["manual_kps"]
GT_8_REG = np.array(GT_KPS_REG[:8], dtype=np.float64)

clicks_reg = [
    list(GT_8_REG[0]),    # 0 — 시뮬레이션에서는 화면 안. CLICK 으로 처리.
    list(GT_8_REG[1]),
    list(GT_8_REG[2]),
    None,                  # 3 missing
    list(GT_8_REG[4]),
    list(GT_8_REG[5]),
    list(GT_8_REG[6]),
    None,                  # 7 missing
    None,
]
extrap_reg = [False] * 9   # 회귀: 외삽 표시 없이 (기존 동작과 동일)
pose_reg = solve_pose(clicks_reg, K, img_shape=img_reg.shape)
if pose_reg:
    proj_reg = pose_reg["projected_all"]
    rms = float(np.sqrt(np.mean(np.sum(
        (np.array(proj_reg[:8]) - GT_8_REG) ** 2, axis=1))))
    print(f"  v7 (no extrap mask, img_shape 만): RMS vs GT 8 corner = {rms:.2f}px  "
          f"(reference: v6 = 2.24px)")
else:
    print(f"  v7 FAIL")
