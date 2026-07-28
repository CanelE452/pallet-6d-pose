"""diagnose_v8_night03.py — capturenight03 1779448848688752640 frame
v8 wireframe collapse 진단.

사용자 보고: 멀리 (작은) + oblique view, 클릭 0~5 (6 점), 6/7 missing.
wireframe 이 사용자 클릭에 fit 안 되고 더 작게/위쪽에 그려짐 (collapse).

screenshot 색 centroid 추출 (extract from `f9ef0d32...41.png`):
  0 red:    (353, 267)  near-top-LEFT  (image coords, after y-=27 offset)
  1 orange: (472, 266)  near-top-RIGHT
  2 yellow: (476, 281)  near-bot-RIGHT
  3 green:  (349, 284)  near-bot-LEFT
  4 cyan:   (401, 263)  far-top-LEFT
  5 blue:   (492, 260)  far-top-RIGHT
  6 magenta: MISSING
  7 white:   MISSING
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import (
    solve_pose, make_pallet_keypoints_3d, project_3d, PALLET_DIMS,
    LR_PAIRS, TB_PAIRS, FR_PAIRS, _CUBE_FLIPS_DEG, _rot_axis_angle,
    _seed_from_ippe_face, _CUBOID_FACES, _refine_with_init,
    _eval_pair_invariants, _eval_click_lr_viol, _eval_click_tb_viol,
    _reproj_err_dict, _eval_v8_tilt, V8_TILT_HARD_THR, V8_TILT_SOFT_THR,
    parallelogram_extrapolate,
)

REPO  = r"C:\Users\minjae\Documents\github\FoundationPose"
IMG   = os.path.join(REPO, "data/night/capturenight03/rgb/1779448848688752640.png")
K_TXT = os.path.join(REPO, "data/night/capturenight03/cam_K.txt")
OUT   = os.path.join(REPO, "data/pallet/results/annotate_v8_oblique")


def night03_clicks():
    """Screenshot 색 centroid 로부터 추출한 사용자 click (image coords)."""
    clicks = [None] * 9
    clicks[0] = [353.0, 267.0]
    clicks[1] = [472.0, 266.0]
    clicks[2] = [476.0, 281.0]
    clicks[3] = [349.0, 284.0]
    clicks[4] = [401.0, 263.0]
    clicks[5] = [492.0, 260.0]
    return clicks


def draw_overlay(img, proj_all, clicks, label, color=(0, 255, 0)):
    vis = img.copy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        pa, pb = proj_all[a], proj_all[b]
        if pa[0] == -1.0 and pa[1] == -1.0: continue
        if pb[0] == -1.0 and pb[1] == -1.0: continue
        cv2.line(vis, (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))), color, 2, cv2.LINE_AA)
    for i in range(8):
        p = proj_all[i]
        if p[0] == -1.0 and p[1] == -1.0: continue
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 4, color, -1)
        cv2.putText(vis, str(i), (int(p[0])+6, int(p[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    # User clicks (different markers)
    click_cols = [(0,0,255),(0,165,255),(0,255,255),(0,255,0),
                  (255,255,0),(255,0,0),(255,0,255),(255,255,255)]
    for i, c in enumerate(clicks[:8]):
        if c is None: continue
        cv2.circle(vis, (int(c[0]), int(c[1])), 7, click_cols[i], 2)
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2, cv2.LINE_AA)
    return vis


def enumerate_candidates(kps_2d, K, dims):
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    inits = []
    for name, face in _CUBOID_FACES:
        for k, (R, t) in enumerate(_seed_from_ippe_face(kps_2d, K, kp3d, list(face))):
            inits.append((f"IPPE_{name}_{k}", R, t))
    for flag, fn in [(cv2.SOLVEPNP_EPNP,"EPNP"),(cv2.SOLVEPNP_SQPNP,"SQPNP")]:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2,0]>0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((fn, R, tvec.flatten()))
        except cv2.error: pass
    try:
        ok_n, rvec_list, tvec_list, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for k,(rv,tv) in enumerate(zip(rvec_list,tvec_list)):
                if tv[2,0]>0:
                    R,_ = cv2.Rodrigues(rv)
                    inits.append((f"IPPE_all_{k}", R, tv.flatten()))
    except cv2.error: pass

    cx_K, cy_K = K[0,2], K[1,2]; fx_K = K[0,0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w_ = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w_, 50.0))
    t_man = np.array([(mean_u-cx_K)*z_guess/fx_K, (mean_v-cy_K)*z_guess/fx_K, z_guess])
    Rx180 = cv2.Rodrigues(np.array([np.pi,0,0]))[0]
    inits.append(("Rx180_manual", Rx180.copy(), t_man.copy()))
    inits.append(("Eye_manual", np.eye(3), t_man.copy()))

    flips = []
    for d in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1,0,0),d[0])
        ry = _rot_axis_angle((0,1,0),d[1])
        rz = _rot_axis_angle((0,0,1),d[2])
        flips.append((d, rz@ry@rx))

    click_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    click_span = max(click_pts[:,0].max()-click_pts[:,0].min(),
                     click_pts[:,1].max()-click_pts[:,1].min(), 50.0)
    z_far_limit = 50.0 * fx_K * max(dims) / click_span

    cands = []
    for init_name, R0, t0 in inits:
        for flip_deg, F in flips:
            R_init = R0 @ F
            res = _refine_with_init(obj, img, K, R_init, t0)
            if res is None: continue
            R, t = res
            if t[2]<=0 or t[2]>z_far_limit: continue
            pts_cam = (R@kp3d.T).T+t
            if (pts_cam[:,2]<=0).any(): continue
            lrv, tbv, frv, proj_all, _ = _eval_pair_invariants(R,t,K,kp3d)
            err = _reproj_err_dict(proj_all, valid_idx, kps_2d, weights=None)
            proj_8 = np.array(proj_all[:8], dtype=np.float64)
            bbox_w = float(proj_8[:,0].max()-proj_8[:,0].min())
            bbox_h = float(proj_8[:,1].max()-proj_8[:,1].min())
            tilt = _eval_v8_tilt(R)
            cands.append({
                "init": init_name, "flip": flip_deg, "R": R, "t": t, "err": err,
                "lr": lrv, "tb": tbv, "fr": frv, "sum": lrv+tbv+frv,
                "proj_all": proj_all, "bbox_w": bbox_w, "bbox_h": bbox_h,
                "bbox_area": bbox_w*bbox_h, "tz": float(t[2]), "tilt": tilt,
            })
    return cands


def click_bbox_area(clicks):
    pts = np.array([c for c in clicks[:8] if c is not None], dtype=np.float64)
    return float((pts[:,0].max()-pts[:,0].min()) * (pts[:,1].max()-pts[:,1].min()))


def main():
    os.makedirs(OUT, exist_ok=True)
    K = np.loadtxt(K_TXT).astype(np.float64)
    img = cv2.imread(IMG)
    print(f"IMG {img.shape[1]}x{img.shape[0]}, K fx={K[0,0]:.2f}")

    clicks = night03_clicks()
    print("Clicks:")
    for i, c in enumerate(clicks):
        print(f"  [{i}] {c}")

    # geometry analysis
    click_w = clicks[1][0] - clicks[0][0]   # ~119px (top edge horizontal)
    click_h = clicks[3][1] - clicks[0][1]   # ~17px (left edge vertical)
    click_45 = clicks[5][0] - clicks[4][0]  # ~91px (REAR top edge)
    click_04 = clicks[4][0] - clicks[0][0]  # ~48px (FRONT-REAR top-left shift)
    click_15 = clicks[5][0] - clicks[1][0]  # ~20px (FRONT-REAR top-right shift)
    print(f"\nClick geometry:")
    print(f"  width  (0->1)  = {click_w:.1f}px")
    print(f"  height (0->3)  = {click_h:.1f}px")
    print(f"  REAR_w (4->5)  = {click_45:.1f}px")
    print(f"  shift  (0->4)  = {click_04:.1f}px  (oblique to the LEFT)")
    print(f"  shift  (1->5)  = {click_15:.1f}px")
    print(f"  click_bbox_area = {click_bbox_area(clicks):.0f}px²")

    # expected z from width
    fx = K[0,0]
    z_from_w = fx * 1.1 / click_w
    z_from_h = fx * 0.11 / click_h
    z_from_d = fx * 1.3 / max(click_45, 1.0)   # REAR width if 130-front
    print(f"\nExpected z (Z=fx*X/u):")
    print(f"  from width(110)  = {z_from_w:.2f}m   (if 0->1 = 1.10m wide)")
    print(f"  from height(11)  = {z_from_h:.2f}m   (if 0->3 = 0.11m tall)")
    print(f"  from REAR_w(130) = {z_from_d:.2f}m   (if 4->5 = 1.30m wide, dim2)")

    # Solve pose (current v8)
    pose = solve_pose(clicks, K, img_shape=img.shape)
    if pose is None:
        print("\n!!! solve_pose returned None")
        return
    print(f"\nv8 solve_pose:")
    print(f"  dims = {pose['dims']}   reproj = {pose['reproj_error_px']:.2f}px")
    print(f"  tz = {pose['t'][2]:.2f}m")
    print(f"  v6_strict_passed = {pose['_v6_strict_passed']}")
    print(f"  viol_sum LR/TB/FR = {pose['_v6_lr_viol']}/{pose['_v6_tb_viol']}/{pose['_v6_fr_viol']}")
    print(f"  v8_tilt = {pose['_v8_tilt']:.3f} (hard={V8_TILT_HARD_THR}, soft={V8_TILT_SOFT_THR})")
    print(f"  n_candidates = {pose['_v6_n_candidates']}, n_strict_ok = {pose['_v6_n_strict_ok']}")

    proj_8 = np.array(pose["projected_all"][:8])
    pose_bbox_w = float(proj_8[:,0].max()-proj_8[:,0].min())
    pose_bbox_h = float(proj_8[:,1].max()-proj_8[:,1].min())
    pose_bbox_area = pose_bbox_w * pose_bbox_h
    click_area = click_bbox_area(clicks)
    print(f"  pose_bbox_area  = {pose_bbox_area:.0f}px²  ({pose_bbox_w:.1f}x{pose_bbox_h:.1f})")
    print(f"  click_bbox_area = {click_area:.0f}px²")
    print(f"  ratio (pose/click) = {pose_bbox_area/click_area:.2f}")

    # Per-click reproj
    print(f"\nPer-click distance:")
    for i in range(8):
        if clicks[i] is None: continue
        p = pose["projected_all"][i]
        d = float(np.hypot(p[0]-clicks[i][0], p[1]-clicks[i][1]))
        print(f"  [{i}] click=({clicks[i][0]:.0f},{clicks[i][1]:.0f}) proj=({p[0]:.0f},{p[1]:.0f}) dist={d:.1f}px")

    vis = draw_overlay(img, pose["projected_all"], clicks,
                       f"BEFORE-fix: dims={pose['dims'][:2]} reproj={pose['reproj_error_px']:.1f}px tilt={pose['_v8_tilt']:.2f}")
    cv2.imwrite(os.path.join(OUT, "v8_night03_before.png"), vis)

    # Enumerate candidates
    print(f"\n=== Enumerate candidates for both dims ===")
    all_cands = []
    for label, dims in [("110front", PALLET_DIMS),
                         ("130front", (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))]:
        cands = enumerate_candidates(clicks, K, dims)
        for c in cands:
            c["dims_label"] = label
            c["dims"] = dims
            # bbox-area-vs-click-ratio
            c["bbox_ratio"] = c["bbox_area"] / click_area
        all_cands.extend(cands)
    print(f"Total: {len(all_cands)}")

    # sort by viol, then err
    all_cands.sort(key=lambda c: (c["sum"], c["err"]))

    # Strict-pass only
    strict = [c for c in all_cands if c["sum"]==0]
    print(f"Strict-pass: {len(strict)}")

    # group: tilt-OK & bbox-ratio-OK
    bbox_ok = [c for c in strict if 0.5 <= c["bbox_ratio"] <= 2.5]
    bbox_ok.sort(key=lambda c: c["err"])
    print(f"Strict + bbox-ratio in [0.5,2.5]: {len(bbox_ok)}")
    print(f"\nTop 15 strict, sorted err:")
    strict_sorted = sorted(strict, key=lambda c: c["err"])
    for i, c in enumerate(strict_sorted[:15]):
        print(f"  [{i}] {c['dims_label']:>8} err={c['err']:6.2f} tz={c['tz']:5.2f} "
              f"tilt={c['tilt']:.2f} bbox={c['bbox_area']:5.0f} ratio={c['bbox_ratio']:.2f} "
              f"init={c['init']:<20} flip={str(c['flip'])}")

    # bbox-ratio sorted (best fit)
    print(f"\nTop 10 by |bbox_ratio - 1.0| (strict only):")
    by_ratio = sorted(strict, key=lambda c: abs(c["bbox_ratio"]-1.0))
    for i, c in enumerate(by_ratio[:10]):
        print(f"  [{i}] {c['dims_label']:>8} err={c['err']:6.2f} tz={c['tz']:5.2f} "
              f"tilt={c['tilt']:.2f} bbox_ratio={c['bbox_ratio']:.2f} "
              f"init={c['init']:<20} flip={str(c['flip'])}")

    # Save table
    table_path = os.path.join(OUT, "v8_night03_candidate_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"capturenight03 1779448848688752640  candidate table\n")
        f.write(f"clicks {[i for i,c in enumerate(clicks) if c is not None]}\n")
        f.write(f"click_bbox_area = {click_area:.0f}px²\n")
        f.write(f"Total cands: {len(all_cands)}  strict: {len(strict)}\n")
        f.write("="*150+"\n")
        f.write(f"{'#':>3} {'dims':>8} {'err':>7} {'tz':>6} {'tilt':>5} {'bbox':>7} {'ratio':>5} "
                f"{'LR':>2} {'TB':>2} {'FR':>2} {'init':<22} {'flip':<15}\n")
        for i, c in enumerate(all_cands[:80]):
            f.write(f"{i:3d} {c['dims_label']:>8} {c['err']:7.2f} {c['tz']:6.2f} "
                    f"{c['tilt']:5.2f} {c['bbox_area']:7.0f} {c['bbox_ratio']:5.2f} "
                    f"{c['lr']:2d} {c['tb']:2d} {c['fr']:2d} "
                    f"{c['init']:<22} {str(c['flip']):<15}\n")
    print(f"\nTable saved: {table_path}")

    # Save the best-bbox-ratio candidate overlay (proposed fix)
    if by_ratio:
        best_ratio = by_ratio[0]
        vis_after = draw_overlay(
            img, best_ratio["proj_all"], clicks,
            f"AFTER-bbox-ratio: dims={best_ratio['dims'][:2]} reproj={best_ratio['err']:.1f}px "
            f"ratio={best_ratio['bbox_ratio']:.2f} tilt={best_ratio['tilt']:.2f}")
        cv2.imwrite(os.path.join(OUT, "v8_night03_after_bbox_ratio.png"), vis_after)
        print(f"After-fix overlay saved")


if __name__ == "__main__":
    main()
