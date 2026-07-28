"""verify_annotate_v4_fix_v6.py - Method F: cuboid Z-flip + strict invariants.

근본 원인 (verify_v6_diagnose + GT pose analysis 로 확정):
  make_pallet_keypoints_3d_diagram 의 cuboid local frame 에서 0~3 가 Z_local=+d/2 에
  배치되어 있음. 그러나 OpenCV cam 좌표 +Z=forward 기준 R=I 가정 시 Z_local=+d/2 가
  cam.z 더 큼 (FAR). 즉 "0~3 = near" 라는 컨벤션 의도와 반대.

  GT pose 와 사용자 click 패턴이 일관성 있는 유일한 해:
    cuboid 0~3 at Z_local=-d/2, dims=(W=1.1, D=1.3, H=0.11)
    → IPPE front-4 reproj=2.20px, LR/TB/FR 모두 PASS.

  기존 fix v5 까지: 0~3 at Z_local=+d/2 라 PnP 의 reproj-best 해가 TB-flipped 또는
  FR-flipped 둘 중 하나가 되어 strict invariant 통과 불가능 (proper rotation 24 cube
  symmetry 만으로는 TB ∧ FR 동시 OK 가 안 됨, 반사 필요).

fix v6 (Method F):
  (1) cuboid 0~3 를 Z_local=-d/2 로 재정의 (kp3d 생성 함수 내부).
      compute_perm_v4 는 보존 (학습 데이터 변환 시 사용 중, 다른 데이터 의존).
  (2) Strict invariant 강제 — 모든 LR/TB/FR pair 부등호 위반 즉시 reject.
      IPPE front-4 + IPPE top-4 + EPNP/SQPNP + 24 cube symmetry seed.
  (3) 사용자 click LR/TB 부등호 위반 (click 정의상 모순) → GUI 빨간 경고,
      strict mode disable (click 일관성 우선).

  Invariants (cuboid local frame, kp3d v6-flipped index):
    LR_PAIRS = [(0,1), (3,2), (4,5), (7,6)]   proj.u: left < right (image)
    TB_PAIRS = [(0,3), (1,2), (4,7), (5,6)]   proj.v: top  < bot   (OpenCV y=down)
    FR_PAIRS = [(0,4), (1,5), (2,6), (3,7)]   cam.z:  near < far   (OpenCV z=forward)

검증 케이스 (저장된 manual_kps 사용):
  Case A: 6 clicks (0~5) - 사용자가 GUI 에서 실제 클릭한 그대로 (4,5 image 위쪽)
  Case B: 8 clicks (0~7) - manual_kps 전체

Outputs (data/pallet/results/annotate_v4_fix_v6/):
  v6_strict_6points.png      - Case A, wireframe LR/TB/FR 정상
  v6_strict_8points.png      - Case B
  v6_candidate_table.png     - 24 후보별 invariant 표 (PASS/FAIL)
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
from annotate_pnp import (  # noqa: E402
    project_3d, PALLET_DIMS,
    _CUBE_FLIPS_DEG, _rot_axis_angle, _refine_with_init,
)

_REPO = os.path.dirname(os.path.dirname(_HERE))
SRC_JSON = os.path.join(
    _REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.json")
SRC_IMG = os.path.join(
    _REPO, "challenge/data/capturepallet03_manual_gt/1778651569891693056.png")
OUT = os.path.join(_REPO, "data/pallet/results/annotate_v4_fix_v6")
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

LR_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]
TB_PAIRS = [(0, 3), (1, 2), (4, 7), (5, 6)]
FR_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def make_pallet_keypoints_3d_v6(width=1.1, depth=1.3, height=0.11):
    """fix v6: cuboid local frame Z-flip.

    기존 (annotate_pnp 의 make_pallet_keypoints_3d_diagram):
      0~3 at Z_local = +d/2  (docstring 'near' 라고 하나 R=I 가정 시 실제로 cam_z 큰 쪽)
    fix v6:
      0~3 at Z_local = -d/2  (R=I 가정 시 cam_z 작은 쪽 = 진짜 near).

    cuboid local axes: X=right (+), Y=down (OpenCV, +y=bottom), Z=forward (+).
    indices (camera-facing near = front face):
      0 = (-w/2, -h/2, -d/2)   near-top-LEFT
      1 = (+w/2, -h/2, -d/2)   near-top-RIGHT
      2 = (+w/2, +h/2, -d/2)   near-bot-RIGHT
      3 = (-w/2, +h/2, -d/2)   near-bot-LEFT
      4 = (-w/2, -h/2, +d/2)   far-top-LEFT
      5 = (+w/2, -h/2, +d/2)   far-top-RIGHT
      6 = (+w/2, +h/2, +d/2)   far-bot-RIGHT
      7 = (-w/2, +h/2, +d/2)   far-bot-LEFT
      8 = centroid
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, -d],   # 0
        [+w, -h, -d],   # 1
        [+w, +h, -d],   # 2
        [-w, +h, -d],   # 3
        [-w, -h, +d],   # 4
        [+w, -h, +d],   # 5
        [+w, +h, +d],   # 6
        [-w, +h, +d],   # 7
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])


def _eval_invariants(R, t, K, kp3d):
    pts_cam = (R @ kp3d[:8].T).T + t
    proj_all = project_3d(kp3d, R, t, K)
    proj = np.array(proj_all[:8], dtype=np.float64)

    lr_viol, tb_viol, fr_viol = 0, 0, 0
    per_pair = {"LR": [], "TB": [], "FR": []}
    for (a, b) in LR_PAIRS:
        ok = proj[a, 0] < proj[b, 0]
        per_pair["LR"].append((a, b, float(proj[a, 0]), float(proj[b, 0]), bool(ok)))
        if not ok:
            lr_viol += 1
    for (a, b) in TB_PAIRS:
        ok = proj[a, 1] < proj[b, 1]
        per_pair["TB"].append((a, b, float(proj[a, 1]), float(proj[b, 1]), bool(ok)))
        if not ok:
            tb_viol += 1
    for (a, b) in FR_PAIRS:
        ok = pts_cam[a, 2] < pts_cam[b, 2]
        per_pair["FR"].append((a, b, float(pts_cam[a, 2]), float(pts_cam[b, 2]), bool(ok)))
        if not ok:
            fr_viol += 1
    return lr_viol, tb_viol, fr_viol, proj_all, pts_cam, per_pair


def _reproj_err(proj_all, valid_idx, kps_2d):
    errs = []
    for i in valid_idx:
        u, v = proj_all[i]
        if u < 0:
            errs.append(1e6); continue
        du, dv = u - kps_2d[i][0], v - kps_2d[i][1]
        errs.append(float(np.hypot(du, dv)))
    return float(np.mean(errs)) if errs else 1e9


def _eval_click_lr_viol(kps_2d):
    n = 0
    for (a, b) in LR_PAIRS:
        if (a < len(kps_2d) and b < len(kps_2d)
                and kps_2d[a] is not None and kps_2d[b] is not None):
            if kps_2d[a][0] >= kps_2d[b][0] - 1.0:
                n += 1
    return n


def _eval_click_tb_viol(kps_2d):
    n = 0
    for (a, b) in TB_PAIRS:
        if (a < len(kps_2d) and b < len(kps_2d)
                and kps_2d[a] is not None and kps_2d[b] is not None):
            if kps_2d[a][1] >= kps_2d[b][1] - 1.0:
                n += 1
    return n


def _seed_from_ippe_face(kps_2d, K, kp3d, face_indices):
    """face_indices 4 점 IPPE → 평면 PnP 2 개 해 seed."""
    seeds = []
    if not all(i < len(kps_2d) and kps_2d[i] is not None for i in face_indices):
        return seeds
    obj = np.array([kp3d[i] for i in face_indices], dtype=np.float64)
    img = np.array([kps_2d[i] for i in face_indices], dtype=np.float64)
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            for rv, tv in zip(rvecs, tvecs):
                if tv[2, 0] > 0:
                    R, _ = cv2.Rodrigues(rv)
                    seeds.append((R, tv.flatten()))
    except cv2.error:
        pass
    return seeds


def solve_pose_v6(kps_2d, K, dims=PALLET_DIMS):
    """fix v6: Z-flipped cuboid + IPPE seed + strict invariant."""
    kp3d = make_pallet_keypoints_3d_v6(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    if len(valid_idx) < 4:
        return None
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    # ── 1) IPPE seeds on planar subsets (front 4, top 4) ──
    inits = []
    inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d, [0, 1, 2, 3]))   # FRONT
    inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d, [0, 1, 5, 4]))   # TOP
    # ── 2) PnP on all valid clicks ──
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
        except cv2.error:
            pass
    try:
        ok_n, rvec_list, tvec_list, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rvec_list, tvec_list):
                if tv[2, 0] > 0:
                    R_ippe, _ = cv2.Rodrigues(rv)
                    inits.append((R_ippe, tv.flatten()))
    except cv2.error:
        pass

    cx_K, cy_K = K[0, 2], K[1, 2]
    fx_K = K[0, 0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w, 50.0))
    t_manual = np.array([(mean_u - cx_K) * z_guess / fx_K,
                         (mean_v - cy_K) * z_guess / fx_K, z_guess], dtype=np.float64)
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append((Rx180.copy(), t_manual.copy()))
    inits.append((np.eye(3), t_manual.copy()))

    # ── 3) 24 cube symmetry flip ──
    flips = []
    for ax in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1, 0, 0), ax[0])
        ry = _rot_axis_angle((0, 1, 0), ax[1])
        rz = _rot_axis_angle((0, 0, 1), ax[2])
        flips.append(rz @ ry @ rx)

    click_lr_viol = _eval_click_lr_viol(kps_2d)
    click_tb_viol = _eval_click_tb_viol(kps_2d)

    # ── 4) Candidate evaluation ──
    candidates = []
    for R0, t0 in inits:
        for F in flips:
            R_init = R0 @ F
            res = _refine_with_init(obj, img, K, R_init, t0)
            if res is None:
                continue
            R, t = res
            if t[2] <= 0:
                continue
            pts_cam = (R @ kp3d.T).T + t
            if (pts_cam[:, 2] <= 0).any():
                continue
            lrv, tbv, frv, proj_all, _, per_pair = _eval_invariants(R, t, K, kp3d)
            err = _reproj_err(proj_all, valid_idx, kps_2d)
            candidates.append({
                "err": err,
                "lr_viol": lrv, "tb_viol": tbv, "fr_viol": frv,
                "viol_sum": lrv + tbv + frv,
                "R": R, "t": t, "proj_all": proj_all,
                "pts_cam": pts_cam, "per_pair": per_pair,
            })

    if not candidates:
        return None

    # ── 5) Strict mode selection ──
    strict_ok = [c for c in candidates if c["viol_sum"] == 0]
    if click_lr_viol >= 1 or click_tb_viol >= 1:
        best = min(candidates, key=lambda c: c["err"])
        strict_passed = False
    elif strict_ok:
        best = min(strict_ok, key=lambda c: c["err"])
        strict_passed = True
    else:
        def _sc(c):
            return c["err"] + 100000.0 * c["viol_sum"]
        best = min(candidates, key=_sc)
        strict_passed = False

    rvec, _ = cv2.Rodrigues(best["R"])
    return {
        "R": best["R"], "t": best["t"],
        "rvec": rvec, "tvec": best["t"].reshape(3, 1),
        "reproj_error_px": best["err"],
        "projected_all": best["proj_all"],
        "dims": dims,
        "_v6_lr_viol": best["lr_viol"],
        "_v6_tb_viol": best["tb_viol"],
        "_v6_fr_viol": best["fr_viol"],
        "_v6_viol_sum": best["viol_sum"],
        "_v6_per_pair": best["per_pair"],
        "_v6_strict_passed": strict_passed,
        "_v6_click_lr_viol": click_lr_viol,
        "_v6_click_tb_viol": click_tb_viol,
        "_v6_n_candidates": len(candidates),
        "_v6_n_strict_ok": len(strict_ok),
        "_v6_all_candidates": candidates,
        "v6_warning": (click_lr_viol >= 1) or (click_tb_viol >= 1) or (best["viol_sum"] > 0),
    }


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
    for i, p in enumerate(kps_2d[:9]):
        if p is None:
            continue
        c = (int(p[0]), int(p[1]))
        cv2.drawMarker(vis, c, KP_COLORS[i], cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(vis, f"c{i}", (c[0] + 8, c[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, KP_COLORS[i], 2)

    cv2.rectangle(vis, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    if pose is not None and pose.get("v6_warning"):
        cv2.rectangle(vis, (0, 28), (w, 50), (0, 0, 0), -1)
        msg = (f"[v6] viol(LR={pose['_v6_lr_viol']} TB={pose['_v6_tb_viol']} "
               f"FR={pose['_v6_fr_viol']}) click_LR={pose['_v6_click_lr_viol']} "
               f"TB={pose['_v6_click_tb_viol']}")
        cv2.putText(vis, msg, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0, 0, 255), 1)

    if pose is not None:
        line1 = (f"dims={pose['dims']}  reproj={pose['reproj_error_px']:.2f}px  "
                 f"strict_passed={pose['_v6_strict_passed']}  "
                 f"n_cand={pose['_v6_n_candidates']}  n_strict_ok={pose['_v6_n_strict_ok']}")
        line2 = (f"viol: LR={pose['_v6_lr_viol']}/{len(LR_PAIRS)}  "
                 f"TB={pose['_v6_tb_viol']}/{len(TB_PAIRS)}  "
                 f"FR={pose['_v6_fr_viol']}/{len(FR_PAIRS)}  "
                 f"sum={pose['_v6_viol_sum']}")
        line3 = (f"click viol: LR={pose['_v6_click_lr_viol']}  TB={pose['_v6_click_tb_viol']}  "
                 f"(>=1: strict disable, GUI red warning)")
        cv2.rectangle(vis, (0, h - 56), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, line1, (8, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)
        cv2.putText(vis, line2, (8, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 255, 200), 1)
        cv2.putText(vis, line3, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 200, 255), 1)

    cv2.imwrite(out_path, vis)
    print(f"[saved] {out_path}")


def _candidate_table_plot(pose, out_path):
    if pose is None or "_v6_all_candidates" not in pose:
        return
    cands = pose["_v6_all_candidates"]
    cands_sorted = sorted(cands, key=lambda c: (c["viol_sum"], c["err"]))
    seen = set()
    unique = []
    for c in cands_sorted:
        key = (c["lr_viol"], c["tb_viol"], c["fr_viol"], round(c["err"], 2),
               round(c["t"][2], 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    unique = unique[:30]

    n = len(unique)
    fig, ax = plt.subplots(figsize=(11, 0.32 * (n + 3) + 1.2))
    ax.axis("off")

    headers = ["#", "LR", "TB", "FR", "viol", "reproj(px)", "t.z(m)", "PASS"]
    rows = []
    for i, c in enumerate(unique):
        pass_str = "PASS" if c["viol_sum"] == 0 else "FAIL"
        rows.append([str(i),
                     f"{c['lr_viol']}/{len(LR_PAIRS)}",
                     f"{c['tb_viol']}/{len(TB_PAIRS)}",
                     f"{c['fr_viol']}/{len(FR_PAIRS)}",
                     str(c["viol_sum"]),
                     f"{c['err']:.2f}",
                     f"{c['t'][2]:.3f}",
                     pass_str])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                     loc="center",
                     colWidths=[0.05, 0.09, 0.09, 0.09, 0.08, 0.13, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for j, _h in enumerate(headers):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", weight="bold")
    for i, c in enumerate(unique):
        is_best = (i == 0)
        fc = "#c8e6c9" if c["viol_sum"] == 0 else "#ffcdd2"
        if is_best:
            fc = "#aed581" if c["viol_sum"] == 0 else "#ef9a9a"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(fc)

    title_str = (f"fix v6: Z-flipped cuboid + IPPE seed + 24 cube symmetry\n"
                 f"unique={n}  strict_ok={pose['_v6_n_strict_ok']}/{pose['_v6_n_candidates']}  "
                 f"selected viol={pose['_v6_viol_sum']}  "
                 f"strict_passed={pose['_v6_strict_passed']}")
    ax.set_title(title_str, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _print_per_pair(pose):
    if pose is None:
        return
    pp = pose.get("_v6_per_pair", {})
    print("    LR pairs (proj.u left < right):")
    for (a, b, ua, ub, ok) in pp.get("LR", []):
        mark = "OK" if ok else "FAIL"
        print(f"      ({a},{b}): u={ua:7.1f} vs {ub:7.1f}  [{mark}]")
    print("    TB pairs (proj.v top < bot):")
    for (a, b, va, vb, ok) in pp.get("TB", []):
        mark = "OK" if ok else "FAIL"
        print(f"      ({a},{b}): v={va:7.1f} vs {vb:7.1f}  [{mark}]")
    print("    FR pairs (cam.z near < far):")
    for (a, b, za, zb, ok) in pp.get("FR", []):
        mark = "OK" if ok else "FAIL"
        print(f"      ({a},{b}): z={za:6.3f} vs {zb:6.3f}  [{mark}]")


def main():
    with open(SRC_JSON) as f:
        d = json.load(f)
    cam = d["camera_data"]["intrinsics"]
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    manual = d["objects"][0]["manual_kps"]
    img0 = cv2.imread(SRC_IMG)
    if img0 is None:
        raise SystemExit(f"image not found: {SRC_IMG}")

    print(f"K=\n{K}\n")
    print("manual_kps:")
    for i, p in enumerate(manual[:8]):
        print(f"  {i}: ({p[0]:6.1f}, {p[1]:6.1f})")
    print()

    # Case A: 6 clicks
    print("=" * 78)
    print("[Case A] 6 clicks: 0~3 (FRONT vertical) + 4~5 (TOP rear edge, image up)")
    print("=" * 78)
    kps_6 = list(manual[:6]) + [None] * 3
    pose_6 = solve_pose_v6(kps_6, K)
    if pose_6:
        print(f"  reproj={pose_6['reproj_error_px']:.2f}px")
        print(f"  viol(LR={pose_6['_v6_lr_viol']}, TB={pose_6['_v6_tb_viol']}, "
              f"FR={pose_6['_v6_fr_viol']}, sum={pose_6['_v6_viol_sum']})")
        print(f"  click_viol(LR={pose_6['_v6_click_lr_viol']}, "
              f"TB={pose_6['_v6_click_tb_viol']})")
        print(f"  strict_passed={pose_6['_v6_strict_passed']}  "
              f"n_strict_ok={pose_6['_v6_n_strict_ok']}/{pose_6['_v6_n_candidates']}")
        _print_per_pair(pose_6)
    _draw_pose_overlay(img0, kps_6, pose_6,
                       "Case A: 6 clicks - fix v6 strict invariants",
                       os.path.join(OUT, "v6_strict_6points.png"))

    # Case B: 8 clicks
    print()
    print("=" * 78)
    print("[Case B] 8 clicks: 0~7 (manual_kps full)")
    print("=" * 78)
    kps_8 = list(manual[:8]) + [None]
    pose_8 = solve_pose_v6(kps_8, K)
    if pose_8:
        print(f"  reproj={pose_8['reproj_error_px']:.2f}px")
        print(f"  viol(LR={pose_8['_v6_lr_viol']}, TB={pose_8['_v6_tb_viol']}, "
              f"FR={pose_8['_v6_fr_viol']}, sum={pose_8['_v6_viol_sum']})")
        print(f"  click_viol(LR={pose_8['_v6_click_lr_viol']}, "
              f"TB={pose_8['_v6_click_tb_viol']})")
        print(f"  strict_passed={pose_8['_v6_strict_passed']}  "
              f"n_strict_ok={pose_8['_v6_n_strict_ok']}/{pose_8['_v6_n_candidates']}")
        _print_per_pair(pose_8)
    _draw_pose_overlay(img0, kps_8, pose_8,
                       "Case B: 8 clicks - fix v6 strict invariants",
                       os.path.join(OUT, "v6_strict_8points.png"))

    # 24 candidate table (Case A)
    print()
    print("=" * 78)
    print("[Candidate Table] unique (LR, TB, FR, reproj) for Case A (6 clicks)")
    print("=" * 78)
    _candidate_table_plot(pose_6, os.path.join(OUT, "v6_candidate_table.png"))

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, p in [("Case A (6 clicks)", pose_6), ("Case B (8 clicks)", pose_8)]:
        if p is None:
            print(f"  {name}: PnP FAILED")
            continue
        ok = (p["_v6_viol_sum"] == 0)
        verdict = "PASS (strict all invariants)" if ok else f"FAIL (viol={p['_v6_viol_sum']})"
        print(f"  {name}: reproj={p['reproj_error_px']:.2f}px  {verdict}")
    print(f"\n[done] outputs: {OUT}")


if __name__ == "__main__":
    main()
