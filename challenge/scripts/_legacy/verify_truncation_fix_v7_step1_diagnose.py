"""verify_truncation_fix_v7_step1_diagnose.py — 7 missing 케이스 candidate 진단.

저장된 JSON (`capturepallet03/1778651573855324672`) 에서 manual_kps 0~6 사용 (7 missing).
6-face IPPE × 24 cube symmetry × LM refine → 모든 candidate 의 reproj + viol_sum 출력.

목표:
  - "정상 해" 가 후보 안에 존재하는가? (사용자 직접 클릭한 1,2,4,5,6 의 reproj 가 작은
    candidate). 존재하면 selection logic 문제. 부재하면 IPPE seed/dim 문제.
  - dim a (110 정면) vs dim b (130 정면) 어느 쪽이 정답인가?
  - 외삽 점 (0, 3) 의 noise 가 어느 정도 영향을 주는가?

저장: data/pallet/results/annotate_truncation_v7/v7_candidate_table.png
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
    make_pallet_keypoints_3d, project_3d, PALLET_DIMS,
    _CUBOID_FACES, _CUBE_FLIPS_DEG, _rot_axis_angle,
    _seed_from_ippe_face, _refine_with_init,
    _eval_pair_invariants, _reproj_err_dict,
)


REPO = os.path.dirname(os.path.dirname(_HERE))
SEQ_DIR = os.path.join(REPO, "data", "outside", "capturepallet03")
GT_DIR  = os.path.join(REPO, "challenge", "data", "capturepallet03_manual_gt")
FRAME   = "1778651573855324672"
OUT_DIR = os.path.join(REPO, "data", "pallet", "results", "annotate_truncation_v7")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Load frame + saved JSON (user 시나리오) ───────────────────────────────
img = cv2.imread(os.path.join(SEQ_DIR, "rgb", f"{FRAME}.png"))
K = np.loadtxt(os.path.join(SEQ_DIR, "cam_K.txt"))
with open(os.path.join(GT_DIR, f"{FRAME}.json"), "r", encoding="utf-8") as f:
    saved = json.load(f)
saved_kps = saved["objects"][0]["manual_kps"]    # 9 pts (0..7 + centroid)

# 사용자 시나리오 재현 : 0~6 클릭 (centroid=8 도 saved 에 있음 → 외삽 한 7 missing 만)
# 0 = t 외삽 (image 좌측 밖), 1/2 = 직접 클릭, 3 = x parallelogram 외삽
# 4/5/6 = 직접 클릭, 7 = 미클릭, 8 = saved 에 있음 (PnP fill 결과)
clicks = [
    list(saved_kps[0]),    # 0 : t 외삽 (image 좌측 밖)
    list(saved_kps[1]),    # 1 : 직접 클릭
    list(saved_kps[2]),    # 2 : 직접 클릭
    list(saved_kps[3]),    # 3 : x 외삽 (image 안 좌측)
    list(saved_kps[4]),    # 4 : 직접 클릭
    list(saved_kps[5]),    # 5 : 직접 클릭
    list(saved_kps[6]),    # 6 : 직접 클릭
    None,                  # 7 : missing
    None,                  # centroid : not clicked
]
# 외삽인지 직접 클릭인지 메타 (extrapolated=True 면 PnP weight 낮춤)
extrapolated_mask = [True, False, False, True, False, False, False, False, False]


print("=" * 92)
print(f"frame: {FRAME}    image: {img.shape}")
print(f"clicks (7 점, 7 missing):")
for i, p in enumerate(clicks):
    if p is None:
        tag = "MISSING"
        coord = "----"
    else:
        tag = "EXTRAP" if extrapolated_mask[i] else "CLICK "
        coord = f"({p[0]:.1f}, {p[1]:.1f})"
    print(f"  {i}: {tag} {coord}")
print()


# ─── Enumerate candidates for both dims ─────────────────────────────────────
def enum_candidates(kps_2d, K, dims, dims_label):
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    if len(valid_idx) < 4:
        return []
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    inits = []
    seed_meta = []  # (init_idx, face_name)
    for fname, face in _CUBOID_FACES:
        seeds = _seed_from_ippe_face(kps_2d, K, kp3d, list(face))
        for s in seeds:
            inits.append(s)
            seed_meta.append(fname)

    for flag, lbl in ((cv2.SOLVEPNP_EPNP, "EPNP"),
                      (cv2.SOLVEPNP_SQPNP, "SQPNP")):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
                seed_meta.append(lbl)
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
                    seed_meta.append("IPPE-all")
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
    inits.append((Rx180.copy(), t_manual.copy())); seed_meta.append("Rx180+manual")
    inits.append((np.eye(3), t_manual.copy())); seed_meta.append("Iden+manual")

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

    # 실제 사용자 click 된 idx (extrapolated_mask=False) 만 weighted reproj 계산용
    real_click_idx = [i for i in valid_idx if not extrapolated_mask[i]]

    out = []
    for si, ((R0, t0), seed_name) in enumerate(zip(inits, seed_meta)):
        for fi, F in enumerate(flips):
            R_init = R0 @ F
            res = _refine_with_init(obj, img_pts, K, R_init, t0)
            if res is None:
                continue
            R, t = res
            if t[2] <= 0:
                continue
            if t[2] > z_far_limit:
                continue
            pts_cam_check = (R @ kp3d.T).T + t
            if (pts_cam_check[:, 2] <= 0).any():
                continue
            lrv, tbv, frv, proj_all, _ = _eval_pair_invariants(R, t, K, kp3d)
            # reproj (모든 valid idx)
            err_all = _reproj_err_dict(proj_all, valid_idx, kps_2d)
            # reproj (직접 클릭 만 — 외삽 점 제외)
            err_click = _reproj_err_dict(proj_all, real_click_idx, kps_2d) \
                        if real_click_idx else err_all
            # cuboid screen bbox area (degenerate 가드 진단)
            proj_8 = np.array(proj_all[:8], dtype=np.float64)
            bbox_w = float(proj_8[:, 0].max() - proj_8[:, 0].min())
            bbox_h = float(proj_8[:, 1].max() - proj_8[:, 1].min())
            bbox_area = bbox_w * bbox_h
            out.append({
                "dims_label": dims_label,
                "seed": seed_name,
                "flip": _CUBE_FLIPS_DEG[fi],
                "err_all": err_all,
                "err_click": err_click,
                "lr": lrv, "tb": tbv, "fr": frv,
                "viol_sum": lrv + tbv + frv,
                "z": float(t[2]),
                "bbox_w": bbox_w, "bbox_h": bbox_h,
                "bbox_area": bbox_area,
                "R": R, "t": t, "proj_all": proj_all,
            })
    return out


cands_a = enum_candidates(clicks, K, PALLET_DIMS, "A(1.1front)")
cands_b = enum_candidates(clicks, K, (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]),
                           "B(1.3front)")

print("=" * 92)
print(f"dim A (110 front, 130 depth, 11 h) : {len(cands_a)} candidates")
print(f"dim B (130 front, 110 depth, 11 h) : {len(cands_b)} candidates")
print()


def print_table(cands, title, top_n=15):
    print("─" * 92)
    print(f"[{title}]  TOP {top_n} by err_click + viol_sum*1000")
    print("─" * 92)
    print(f"  {'idx':>3s} {'dim':<11s} {'seed':<14s} {'flip(deg)':<13s} "
          f"{'err_all':>8s} {'err_clk':>8s} "
          f"{'lr':>2s} {'tb':>2s} {'fr':>2s} {'vio':>3s} "
          f"{'z':>5s} {'bbox_w':>7s} {'bbox_h':>7s}")
    s = sorted(cands, key=lambda c: c["err_click"] + 1000 * c["viol_sum"])
    for i, c in enumerate(s[:top_n]):
        print(f"  {i:>3d} {c['dims_label']:<11s} {c['seed']:<14s} "
              f"{str(c['flip']):<13s} "
              f"{c['err_all']:>8.2f} {c['err_click']:>8.2f} "
              f"{c['lr']:>2d} {c['tb']:>2d} {c['fr']:>2d} {c['viol_sum']:>3d} "
              f"{c['z']:>5.2f} {c['bbox_w']:>7.1f} {c['bbox_h']:>7.1f}")
    print()


print_table(cands_a, "DIM A 110-front  by err_click", top_n=15)
print_table(cands_b, "DIM B 130-front  by err_click", top_n=15)


# 전체 결합 후 strict-pass 만 top
all_cands = cands_a + cands_b
strict = [c for c in all_cands if c["viol_sum"] == 0]
print("=" * 92)
print(f"STRICT-PASS only : {len(strict)} / {len(all_cands)}")
print("=" * 92)
if strict:
    strict_sorted = sorted(strict, key=lambda c: c["err_click"])
    print(f"  {'idx':>3s} {'dim':<11s} {'seed':<14s} {'flip(deg)':<13s} "
          f"{'err_all':>8s} {'err_clk':>8s} "
          f"{'z':>5s} {'bbox_w':>7s} {'bbox_h':>7s}")
    for i, c in enumerate(strict_sorted[:10]):
        print(f"  {i:>3d} {c['dims_label']:<11s} {c['seed']:<14s} "
              f"{str(c['flip']):<13s} "
              f"{c['err_all']:>8.2f} {c['err_click']:>8.2f} "
              f"{c['z']:>5.2f} {c['bbox_w']:>7.1f} {c['bbox_h']:>7.1f}")
else:
    print("  (no strict-pass candidate)")
print()


# saved JSON 의 결과 비교
print("=" * 92)
print("saved JSON 의 결과 (v6 현재 선택):")
print("=" * 92)
saved_pose = saved["objects"][0]["pose_transform"]
saved_R = np.array(saved_pose)[:3, :3]
saved_t = np.array(saved_pose)[:3, 3]
print(f"  z = {saved_t[2]:.2f} m")
print(f"  reproj = {saved['objects'][0]['reproj_error_px']:.2f} px")
print(f"  dims = {saved['objects'][0]['dimensions_m']}")
# saved pose 의 invariants / err_click 계산
saved_dim = (
    saved["objects"][0]["dimensions_m"]["width"],
    saved["objects"][0]["dimensions_m"]["depth"],
    saved["objects"][0]["dimensions_m"]["height"],
)
kp3d_saved = make_pallet_keypoints_3d(*saved_dim)
proj_saved = project_3d(kp3d_saved, saved_R, saved_t, K)
proj_8 = np.array(proj_saved[:8], dtype=np.float64)
bbox_w = float(proj_8[:, 0].max() - proj_8[:, 0].min())
bbox_h = float(proj_8[:, 1].max() - proj_8[:, 1].min())
print(f"  cuboid bbox on image: {bbox_w:.1f} × {bbox_h:.1f} px  "
      f"(image: {img.shape[1]} × {img.shape[0]})")
lrv, tbv, frv, _, _ = _eval_pair_invariants(saved_R, saved_t, K, kp3d_saved)
print(f"  viol: lr={lrv} tb={tbv} fr={frv}")

# 사용자 직접 클릭 점만 reproj
real_idx = [i for i in range(8) if not extrapolated_mask[i] and clicks[i] is not None]
errs_click = []
for i in real_idx:
    du = proj_saved[i][0] - clicks[i][0]
    dv = proj_saved[i][1] - clicks[i][1]
    errs_click.append(float(np.hypot(du, dv)))
print(f"  사용자 직접 click 만 reproj: {np.mean(errs_click):.2f} px  (idx {real_idx})")
errs_extrap = []
for i in [0, 3]:
    if clicks[i] is None: continue
    du = proj_saved[i][0] - clicks[i][0]
    dv = proj_saved[i][1] - clicks[i][1]
    errs_extrap.append(float(np.hypot(du, dv)))
print(f"  외삽 점 (0, 3) reproj:      {np.mean(errs_extrap):.2f} px")
print()


# 진단 결론
print("=" * 92)
print("DIAGNOSIS")
print("=" * 92)
# best by err_click + strict
best_strict = sorted(strict, key=lambda c: c["err_click"])[0] if strict else None
best_overall = sorted(all_cands, key=lambda c: c["err_click"])[0]
print(f"best by err_click (overall): {best_overall['dims_label']} {best_overall['seed']} "
      f"err_click={best_overall['err_click']:.2f} err_all={best_overall['err_all']:.2f} "
      f"viol_sum={best_overall['viol_sum']} z={best_overall['z']:.2f}")
if best_strict is not None:
    print(f"best STRICT (err_click) : {best_strict['dims_label']} {best_strict['seed']} "
          f"err_click={best_strict['err_click']:.2f} err_all={best_strict['err_all']:.2f} "
          f"z={best_strict['z']:.2f} bbox={best_strict['bbox_w']:.0f}x{best_strict['bbox_h']:.0f}")
print()


# ─── Visualize : top-6 candidates side-by-side ───────────────────────────────
from annotate_draw import CUBOID_EDGES, KP_COLORS

def draw_pose(img_src, kps, proj_all, title, sub):
    vis = img_src.copy()
    pts = [(int(round(p[0])), int(round(p[1]))) for p in proj_all[:8]]
    for k, (a, b) in enumerate(CUBOID_EDGES):
        col = (0, 220, 0) if k < 4 else (0, 160, 0)
        thick = 3 if k < 4 else 1
        cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
    for i, p in enumerate(kps):
        if p is None:
            continue
        c = (int(round(p[0])), int(round(p[1])))
        if c[0] < 0 or c[0] >= vis.shape[1] or c[1] < 0 or c[1] >= vis.shape[0]:
            continue
        cv2.circle(vis, c, 5, KP_COLORS[i], -1)
        cv2.circle(vis, c, 7, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (c[0] + 7, c[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, KP_COLORS[i], 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(vis, title, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(vis, sub, (6, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
    return vis


# top-6 ranked by err_click (strict-pass 우선)
panel_cands = sorted(strict, key=lambda c: c["err_click"])[:3] + \
              sorted([c for c in all_cands if c["viol_sum"] > 0],
                     key=lambda c: c["err_click"])[:3]
if len(panel_cands) < 6:
    panel_cands += sorted(all_cands, key=lambda c: c["err_click"])[len(panel_cands):6]

panels = []
for c in panel_cands[:6]:
    vis = draw_pose(
        img, clicks, c["proj_all"],
        f"{c['dims_label']} {c['seed']} flip{c['flip']}",
        f"all={c['err_all']:.1f}  clk={c['err_click']:.1f}  "
        f"vio={c['viol_sum']}  z={c['z']:.2f}m  bbox={c['bbox_w']:.0f}x{c['bbox_h']:.0f}"
    )
    panels.append(vis)

# 3x2 grid
h, w = img.shape[:2]
grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
for i, p in enumerate(panels):
    r, c = i // 3, i % 3
    grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
out_path = os.path.join(OUT_DIR, "v7_candidate_table.png")
cv2.imwrite(out_path, grid)
print(f"saved: {out_path}")

# saved pose overlay (current v6 결과)
vis_saved = draw_pose(
    img, clicks, proj_saved,
    "SAVED (v6 current)",
    f"reproj_saved={saved['objects'][0]['reproj_error_px']:.2f}px  z={saved_t[2]:.2f}m  "
    f"bbox={bbox_w:.0f}x{bbox_h:.0f}"
)
cv2.imwrite(os.path.join(OUT_DIR, "v7_saved_pose.png"), vis_saved)
print(f"saved: {os.path.join(OUT_DIR, 'v7_saved_pose.png')}")
