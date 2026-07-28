"""verify_truncation_fix.py — annotate.py truncation case 검증.

테스트 대상 frame:
  capturepallet03 / 1778651569891693056 — 이미 라벨링된 manual GT 존재 (reproj 1.53px).
  GT 8 corner 사용해서 truncation 시뮬레이션 (사용자 케이스: 012456 6점).

시나리오 (사용자 보고 케이스 모사):
  - 사용자가 클릭 가능한 점: 1, 2, 4, 5, 6 (5 점).
  - 0 (NearTopLeft) = image 밖 → t (TWO-LINE) 외삽 가능 → 6점.
  - 3 (NearBottomLeft) = image 밖 → t 외삽 어려움 → 미클릭 (5점) 또는 parallelogram 외삽 (6점).
  - 7 (FarBottomLeft) = 미클릭.

이 frame 은 image 안에 다 들어가 있으므로 GT 좌표가 그대로 valid.
"image 밖" 시뮬레이션 == GT 좌표 그대로 두되 ' 사용자가 클릭 못 했다 ' 가정.
즉 0/3 의 GT 좌표는 reproj 계산용으로만 보관, kps_2d 에서 None 으로 처리.

테스트 (4 단계):
  [A] Legacy + 012456 6점     — FRONT/TOP IPPE seed 만
  [B] v6+6face + 012456 6점   — 6 face IPPE seed
  [C] PARALLELOGRAM(3 외삽)   — 012(extrap-3)456 7점
  [D] v6+6face + 7점 + auto-fill — 최종 결과

GT 와 비교한 RMS pixel 오차 보고 (8 corner 기준).

저장:
  data/pallet/results/annotate_truncation_fix/
    _raw.png
    truncation_012456_before.png   — [A] legacy
    truncation_012456_after.png    — [D] 최종
    truncation_3_extrapolation.png — [C] parallelogram 단계
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
    solve_pose, project_3d, make_pallet_keypoints_3d,
    parallelogram_extrapolate,
    PALLET_DIMS, _seed_from_ippe_face, _CUBE_FLIPS_DEG, _rot_axis_angle,
    _refine_with_init, _eval_pair_invariants, _reproj_err_dict,
)
from annotate_draw import CUBOID_EDGES, KP_COLORS


REPO = os.path.dirname(os.path.dirname(_HERE))
SEQ_DIR = os.path.join(REPO, "data", "outside", "capturepallet03")
GT_DIR  = os.path.join(REPO, "challenge", "data", "capturepallet03_manual_gt")
FRAME   = "1778651569891693056"
OUT_DIR = os.path.join(REPO, "data", "pallet", "results", "annotate_truncation_fix")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Load frame + GT ────────────────────────────────────────────────────────
img = cv2.imread(os.path.join(SEQ_DIR, "rgb", f"{FRAME}.png"))
K = np.loadtxt(os.path.join(SEQ_DIR, "cam_K.txt"))
with open(os.path.join(GT_DIR, f"{FRAME}.json"), "r", encoding="utf-8") as f:
    gt = json.load(f)
GT_KPS = gt["objects"][0]["manual_kps"]   # 9 corners (0..7 + centroid)
GT_8 = np.array(GT_KPS[:8], dtype=np.float64)

print(f"frame: {FRAME}")
print(f"image shape: {img.shape}, K fx={K[0,0]:.1f}")
print(f"GT (8 corners):")
for i, p in enumerate(GT_8):
    print(f"  {i}: ({p[0]:.1f}, {p[1]:.1f})")
print(f"GT reproj (saved): {gt['objects'][0]['reproj_error_px']:.2f}px")
print()


# ─── 사용자 케이스: 012456 6점 (3, 7 = None) ─────────────────────────────
# 이 frame 은 사실 trunc 가 아니라 다 보이지만, '사용자가 클릭 못 했다' 시뮬레이션.
def make_clicks(missing):
    """missing idx list 만 None, 나머지는 GT 좌표 그대로."""
    out = []
    for i in range(8):
        if i in missing:
            out.append(None)
        else:
            out.append([float(GT_8[i, 0]), float(GT_8[i, 1])])
    out.append(None)   # centroid
    return out


KPS_012456 = make_clicks(missing={3, 7})


def rms_8corner(pose):
    """pose 의 projected_all 과 GT 8 corner 의 RMS pixel 오차."""
    proj = np.array(pose["projected_all"][:8], dtype=np.float64)
    err = np.linalg.norm(proj - GT_8, axis=1)
    return float(np.sqrt((err ** 2).mean()))


# ─── Legacy solve (FRONT/TOP IPPE seed 만) ───────────────────────────────
def solve_pose_legacy_front_top_only(kps_2d, K, dims):
    """v6 truncation fix 이전 — FRONT(0,1,2,3) + TOP(0,1,5,4) IPPE seed 만."""
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    if len(valid_idx) < 4:
        return None
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    inits = []
    inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d, [0, 1, 2, 3]))  # FRONT
    inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d, [0, 1, 5, 4]))  # TOP
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
        except cv2.error:
            pass
    try:
        ok_n, rvec_list, tvec_list, _ = cv2.solvePnPGeneric(
            obj, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rvec_list, tvec_list):
                if tv[2, 0] > 0:
                    R_ippe, _ = cv2.Rodrigues(rv)
                    inits.append((R_ippe, tv.flatten()))
    except cv2.error:
        pass

    fx_K, cx_K, cy_K = K[0, 0], K[0, 2], K[1, 2]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    iw = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(iw, 50.0))
    t_manual = np.array([(mean_u - cx_K) * z_guess / fx_K,
                         (mean_v - cy_K) * z_guess / fx_K, z_guess], dtype=np.float64)
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append((Rx180.copy(), t_manual.copy()))
    inits.append((np.eye(3), t_manual.copy()))

    flips = []
    for ax_rot_deg in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1, 0, 0), ax_rot_deg[0])
        ry = _rot_axis_angle((0, 1, 0), ax_rot_deg[1])
        rz = _rot_axis_angle((0, 0, 1), ax_rot_deg[2])
        flips.append(rz @ ry @ rx)

    click_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    click_span = max(float(click_pts[:, 0].max() - click_pts[:, 0].min()),
                     float(click_pts[:, 1].max() - click_pts[:, 1].min()), 50.0)
    z_far_limit = 50.0 * fx_K * max(dims) / click_span

    candidates = []
    for R0, t0 in inits:
        for F in flips:
            res = _refine_with_init(obj, img_pts, K, R0 @ F, t0)
            if res is None:
                continue
            R, t = res
            if t[2] <= 0 or t[2] > z_far_limit:
                continue
            pts_cam_check = (R @ kp3d.T).T + t
            if (pts_cam_check[:, 2] <= 0).any():
                continue
            lrv, tbv, frv, proj_all, _ = _eval_pair_invariants(R, t, K, kp3d)
            err = _reproj_err_dict(proj_all, valid_idx, kps_2d)
            viol = lrv + tbv + frv
            candidates.append({"err": err, "viol_sum": viol, "R": R, "t": t,
                               "proj_all": proj_all, "lr": lrv, "tb": tbv, "fr": frv})
    if not candidates:
        return None
    strict = [c for c in candidates if c["viol_sum"] == 0]
    if strict:
        best = min(strict, key=lambda c: c["err"])
    else:
        best = min(candidates, key=lambda c: c["err"] + 100000.0 * c["viol_sum"])
    return {"R": best["R"], "t": best["t"], "reproj_error_px": best["err"],
            "projected_all": best["proj_all"], "dims": dims,
            "_v6_lr_viol": best["lr"], "_v6_tb_viol": best["tb"],
            "_v6_fr_viol": best["fr"], "_v6_viol_sum": best["viol_sum"],
            "_v6_strict_passed": best["viol_sum"] == 0,
            "_v6_n_candidates": len(candidates), "_v6_n_strict_ok": len(strict)}


def solve_pose_legacy(kps_2d, K):
    a = solve_pose_legacy_front_top_only(kps_2d, K, PALLET_DIMS)
    b = solve_pose_legacy_front_top_only(
        kps_2d, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))
    cands = [p for p in (a, b) if p is not None]
    if not cands:
        return None
    strict = [p for p in cands if p.get("_v6_strict_passed", False)]
    if strict:
        return min(strict, key=lambda p: p["reproj_error_px"])
    return min(cands, key=lambda p: (p.get("_v6_viol_sum", 0), p["reproj_error_px"]))


# ─── Visualize helper ───────────────────────────────────────────────────────
def draw_cuboid_and_kps(img, kps, pose, title="", warning_lines=None, gt_8=None):
    vis = img.copy()
    # GT 8 corner (light cyan dotted)
    if gt_8 is not None:
        for k, (a, b) in enumerate(CUBOID_EDGES):
            pa = (int(round(gt_8[a, 0])), int(round(gt_8[a, 1])))
            pb = (int(round(gt_8[b, 0])), int(round(gt_8[b, 1])))
            cv2.line(vis, pa, pb, (255, 200, 100), 1, cv2.LINE_AA)
        for i in range(8):
            cv2.circle(vis, (int(round(gt_8[i, 0])), int(round(gt_8[i, 1]))),
                       3, (255, 200, 100), 1)
    if pose is not None:
        proj = pose["projected_all"]
        pts = [(int(round(p[0])), int(round(p[1]))) for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            col = (0, 220, 0) if k < 4 else (0, 160, 0)
            thick = 3 if k < 4 else 1
            cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
        if proj[8][0] >= -1e5:
            cv2.drawMarker(vis, (int(round(proj[8][0])), int(round(proj[8][1]))),
                           (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    for i, p in enumerate(kps):
        if p is None:
            continue
        c = (int(round(p[0])), int(round(p[1])))
        cv2.circle(vis, c, 6, KP_COLORS[i], -1)
        cv2.circle(vis, c, 8, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, KP_COLORS[i], 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if warning_lines:
        y = 52
        for wl, col in warning_lines:
            cv2.rectangle(vis, (0, y - 18), (vis.shape[1], y + 4), (0, 0, 0), -1)
            cv2.putText(vis, wl, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            y += 22
    # legend GT
    cv2.rectangle(vis, (0, vis.shape[0] - 22), (220, vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, "GT cuboid = orange thin", (5, vis.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    return vis


def fmt(pose):
    if pose is None:
        return "FAIL"
    return (f"reproj={pose['reproj_error_px']:.2f}px  "
            f"z={pose['t'][2]:.2f}m  "
            f"viol={pose['_v6_viol_sum']} "
            f"strict={'PASS' if pose.get('_v6_strict_passed', False) else 'FAIL'}")


cv2.imwrite(os.path.join(OUT_DIR, "_raw.png"), img)


# ─── [A] Legacy + 012456 ─────────────────────────────────────────────────────
print("=" * 78)
print("[A] LEGACY (FRONT/TOP IPPE only) + 012456 (6 점)")
print("=" * 78)
pose_A = solve_pose_legacy(KPS_012456, K)
if pose_A is not None:
    rms_A = rms_8corner(pose_A)
    print(f"  {fmt(pose_A)}")
    print(f"  RMS vs GT 8 corner = {rms_A:.2f}px")
    print(f"  t = {pose_A['t']}")
else:
    rms_A = None
    print("  → PnP 실패")
vis_A = draw_cuboid_and_kps(
    img, KPS_012456, pose_A,
    "BEFORE [A]: 012456 (6 pts) + legacy FRONT/TOP IPPE only",
    [(f"{fmt(pose_A)}  | RMS vs GT 8c = {rms_A:.2f}px" if rms_A is not None
      else "PnP FAILED", (0, 0, 255) if rms_A and rms_A > 10 else (0, 255, 0))],
    gt_8=GT_8)
cv2.imwrite(os.path.join(OUT_DIR, "truncation_012456_before.png"), vis_A)
print(f"  saved: truncation_012456_before.png")
print()

# ─── [B] v6 + 6-face seed + 012456 ──────────────────────────────────────────
print("=" * 78)
print("[B] v6 patched (6 face IPPE) + 012456 (6 점)")
print("=" * 78)
pose_B = solve_pose(KPS_012456, K)
if pose_B is not None:
    rms_B = rms_8corner(pose_B)
    print(f"  {fmt(pose_B)}")
    print(f"  RMS vs GT 8 corner = {rms_B:.2f}px")
    print(f"  t = {pose_B['t']}")
else:
    rms_B = None
    print("  → PnP 실패")
print()


# ─── [C] PARALLELOGRAM 3 외삽 ─────────────────────────────────────────────────
print("=" * 78)
print("[C] PARALLELOGRAM: kp3 외삽 (012 → 3)")
print("=" * 78)
pt3, fname, finds = parallelogram_extrapolate(KPS_012456, 3)
if pt3 is not None:
    err3 = float(np.hypot(pt3[0] - GT_8[3, 0], pt3[1] - GT_8[3, 1]))
    print(f"  face = {fname} {finds}")
    print(f"  외삽 3   = ({pt3[0]:.1f}, {pt3[1]:.1f})")
    print(f"  GT     3 = ({GT_8[3, 0]:.1f}, {GT_8[3, 1]:.1f})")
    print(f"  외삽 오차 = {err3:.2f}px")
else:
    print("  → 외삽 실패 (face 3 corner 미클릭)")

# 시각화
vis_C = img.copy()
# GT cuboid (light)
for k, (a, b) in enumerate(CUBOID_EDGES):
    pa = (int(round(GT_8[a, 0])), int(round(GT_8[a, 1])))
    pb = (int(round(GT_8[b, 0])), int(round(GT_8[b, 1])))
    cv2.line(vis_C, pa, pb, (255, 200, 100), 1, cv2.LINE_AA)
if pt3 is not None and fname == "FRONT":
    a, b, c, d = finds
    pa = (int(round(KPS_012456[a][0])), int(round(KPS_012456[a][1])))
    pb = (int(round(KPS_012456[b][0])), int(round(KPS_012456[b][1])))
    pc = (int(round(KPS_012456[c][0])), int(round(KPS_012456[c][1])))
    pd = (int(round(pt3[0])), int(round(pt3[1])))
    cv2.line(vis_C, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(vis_C, pb, pc, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(vis_C, pc, pd, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(vis_C, pd, pa, (0, 255, 255), 2, cv2.LINE_AA)
    # parallelogram diagonal vectors
    cv2.arrowedLine(vis_C, pb, pc, (255, 100, 200), 1, tipLength=0.05)
    cv2.arrowedLine(vis_C, pa, pd, (255, 100, 200), 2, tipLength=0.05)
KPS_AFTER_EXT = list(KPS_012456)
if pt3 is not None:
    KPS_AFTER_EXT[3] = pt3
for i, p in enumerate(KPS_AFTER_EXT):
    if p is None:
        continue
    c = (int(round(p[0])), int(round(p[1])))
    col = KP_COLORS[i] if i != 3 else (0, 255, 255)
    cv2.circle(vis_C, c, 6, col, -1)
    cv2.circle(vis_C, c, 8, (0, 0, 0), 1)
    cv2.putText(vis_C, str(i) + (" (extrap)" if i == 3 else ""),
                (c[0] + 8, c[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
cv2.rectangle(vis_C, (0, 0), (vis_C.shape[1], 60), (0, 0, 0), -1)
cv2.putText(vis_C, "[C] PARALLELOGRAM EXTRAPOLATION: kp3 (FRONT face cyclic)",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
if pt3 is not None:
    cv2.putText(vis_C,
                f"3 = 0 + (2 - 1) = ({pt3[0]:.1f}, {pt3[1]:.1f})   "
                f"GT 3 = ({GT_8[3, 0]:.1f}, {GT_8[3, 1]:.1f})   "
                f"err = {err3:.2f}px",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
cv2.imwrite(os.path.join(OUT_DIR, "truncation_3_extrapolation.png"), vis_C)
print(f"  saved: truncation_3_extrapolation.png")
print()


# ─── [D] v6 + 6-face seed + 7 점 (extrap 3 포함) + g auto-fill ───────────────
print("=" * 78)
print("[D] AFTER: v6 patched + 7 점 (3=extrap) + g auto-fill")
print("=" * 78)
pose_D = solve_pose(KPS_AFTER_EXT, K)
if pose_D is not None:
    rms_D = rms_8corner(pose_D)
    print(f"  {fmt(pose_D)}")
    print(f"  RMS vs GT 8 corner = {rms_D:.2f}px")
    print(f"  t = {pose_D['t']}")
    # g 키 시뮬레이션 — 4/7/8 PnP projection 으로 채움
    KPS_AUTOFILL = list(KPS_AFTER_EXT)
    proj = pose_D["projected_all"]
    n_auto = 0
    for i in range(9):
        if KPS_AUTOFILL[i] is None and proj[i][0] >= -1e5:
            KPS_AUTOFILL[i] = list(proj[i])
            n_auto += 1
    print(f"  g auto-fill: {n_auto} corners filled (4/7/8 같은 미클릭 idx)")
else:
    rms_D = None
    KPS_AUTOFILL = list(KPS_AFTER_EXT)
    print("  → PnP 실패")
vis_D = draw_cuboid_and_kps(
    img, KPS_AUTOFILL, pose_D,
    "AFTER [D]: 012(extrap-3)456 -> 7pts + g auto-fill",
    [(f"{fmt(pose_D)}  | RMS vs GT 8c = {rms_D:.2f}px" if rms_D is not None
      else "PnP FAILED",
      (0, 255, 0) if rms_D is not None and rms_D < 10
      else (0, 200, 255) if rms_D is not None
      else (0, 0, 255))],
    gt_8=GT_8)
cv2.imwrite(os.path.join(OUT_DIR, "truncation_012456_after.png"), vis_D)
print(f"  saved: truncation_012456_after.png")
print()


# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("                         pose                                 RMS vs GT 8c")
print(f"  [A] legacy 012456  : {fmt(pose_A):<55s}  "
      f"{f'{rms_A:.2f}px' if rms_A is not None else 'FAIL':>10s}")
print(f"  [B] v6 6f  012456  : {fmt(pose_B):<55s}  "
      f"{f'{rms_B:.2f}px' if rms_B is not None else 'FAIL':>10s}")
print(f"  [D] v6 6f  7pt+aut : {fmt(pose_D):<55s}  "
      f"{f'{rms_D:.2f}px' if rms_D is not None else 'FAIL':>10s}")
print()
print(f"[C] parallelogram 3 외삽 오차: "
      f"{err3:.2f}px (사용자 click 대신 자동)" if pt3 is not None else "[C] 외삽 실패")
print()
print("[USAGE]  annotate.py 사용자 안내:")
print("  - 012 클릭 후 's 3' (active 를 3 으로) → 'x' → 3 자동 외삽 (parallelogram)")
print("  - 또는 4+ 점 클릭한 상태에서 'g' → 미클릭 idx 자동 채움 + 저장")
print("  - 단축 워크플로 (012456 케이스):")
print("    1) 1,2,4,5,6 클릭   (5점)")
print("    2) 0 = t (TWO-LINE) 외삽   (6점)")
print("    3) idx 3 select → x   (parallelogram, 7점)")
print("    4) g                (auto-fill 4/7 + 8 centroid + save)")
print("    5) 시각 확인 → n (다음 frame)")
