"""diagnose_v8_oblique.py — oblique view (frame 1778653498432396288) wireframe flip 진단.

사용자 보고: capturepallet08 1778653498432396288 frame, 카메라가 팔레트를 대각선 위/멀리서
봄. 사용자 클릭 0,1,2,3 = 수직 FRONT face (Y 축 vertical), 5,6 = 우측 cluster (REAR-TOP/BOT-
RIGHT), 4/7 missing. 결과 wireframe 이 image 한 영역에 작게 몰림 (flip 된 모양).

Step 1: 저장 JSON 확인 — 없음 (사용자 save 전 캡쳐).
Step 2: 사용자 클릭 좌표 추정 (screenshot 분석) → solve_pose direct call.
Step 3: 24 후보 score 표 + 가설 A/B/C/D 검증.
Step 4: fix v8 spec (필요 시).
Step 5: Before/After 시각화 저장.

저장: data/pallet/results/annotate_v8_oblique/
  - v8_oblique_clicks.png   (screenshot annotated)
  - v8_candidate_table.txt  (모든 24×6 후보 표)
  - v8_oblique_before.png   (현재 v7 풀이)
  - v8_oblique_after.png    (fix 후 풀이, 필요시)
  - v8_pose_diag.json       (선택 R, t, viol, candidate 통계)
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
    _reproj_err_dict,
)

REPO  = r"C:\Users\minjae\Documents\github\FoundationPose"
IMG   = os.path.join(REPO, "data/outside/capturepallet08/rgb/1778653498432396288.png")
K_TXT = os.path.join(REPO, "data/outside/capturepallet08/cam_K.txt")
OUT   = os.path.join(REPO, "data/pallet/results/annotate_v8_oblique")
os.makedirs(OUT, exist_ok=True)


def load_K():
    K = np.loadtxt(K_TXT)
    return K.astype(np.float64)


def estimate_clicks_from_image(img):
    """이미지에서 사용자가 클릭했을 plausible 좌표 추정.

    사용자 보고 (스크린샷 KEYPOINTS panel):
      - 0,1,2,3 = vertical FRONT face (수직 4 점)
      - 5,6 = 우측 cluster (REAR-TOP-RIGHT, REAR-BOT-RIGHT)
      - 4,7 = missing

    스크린샷 (224x126 thumbnail) 에서 wireframe 작게 한 영역에 몰린 영역 =
    image 가운데 약간 좌측 (사용자 클릭은 이보다 우측 큰 cuboid 였을 것).
    실제 640x480 RGB 에서 팔레트를 찾기 위해 visual cue 활용.
    """
    H, W = img.shape[:2]
    # 사용자 시나리오 추정 — KEYPOINTS panel 의 click area 가 image 우측 cluster.
    # FRONT face (0,1,2,3) 가 vertical (사용자 표현 "수직"): 보통 oblique 에서
    # cargo 면 (top) 이 위쪽으로 약간 보임. 4,7 missing → BACK-LEFT 모서리 가림.
    # 보수적으로 가운데 약간 좌측에 큰 cuboid (1.1m 폭) 가정.
    # 클릭 좌표는 화면에서 작게 보이는 팔레트 (멀리, 약 2.5m): 화면에서 약 60x40 px.
    # 0 (red, near-top-LEFT),  1 (orange, near-top-RIGHT)
    # 2 (yellow, near-bot-RIGHT),  3 (green, near-bot-LEFT)
    # 5 (blue, far-top-RIGHT),  6 (purple, far-bot-RIGHT)

    # 스크린샷 분석: 작은 cluster 가 image 우측 (≈ x=400, y=240) 영역.
    # FRONT face = vertical 직사각형 (사용자 인용 "수직"). 가로 폭 작고 세로 폭 큼.
    # → 사용자가 cuboid 의 SIDE face 를 FRONT 로 클릭한 가능성도.
    # 일단 plausible 한 oblique view 클릭 set 을 가정.

    # 실제 screenshot (903x522) 에서 색 centroid 추출 (`_inspect_screenshot.py`):
    #   0 red:     (366, 264)
    #   1 orange:  (462, 251)
    #   2 yellow:  (484, 278)
    #   3 green:   (416, 272)
    #   5 blue:    (549, 258)
    #   6 magenta: NOT_FOUND (자동 추출 실패, 시각으로는 (552, 270) 근방 있음 — 5 와 매우 근접)
    # 4, 7 missing (header 가 "Click #7" 이므로 7 이 다음 클릭 대기, 4 도 미클릭)
    # 화면 검토: 0~3 이 horizontal strip 형태 → 사용자가 TOP face 를 FRONT 로 클릭한 듯.
    # PnP 결과 wireframe = green parallelogram, 좌측 lower 로 늘어남.
    clicks = [None] * 9
    clicks[0] = [366.0, 264.0]   # red (label "0")
    clicks[1] = [462.0, 251.0]   # orange
    clicks[2] = [484.0, 278.0]   # yellow
    clicks[3] = [416.0, 272.0]   # green
    # 4 missing
    clicks[5] = [549.0, 258.0]   # blue
    clicks[6] = [552.0, 270.0]   # magenta (extracted by visual — close to 5)
    # 7 missing
    return clicks


def visualize_clicks(img, clicks, out_path):
    vis = img.copy()
    colors = [
        (0, 0, 255),    # 0 red
        (0, 165, 255),  # 1 orange
        (0, 255, 255),  # 2 yellow
        (0, 255, 0),    # 3 green
        (255, 255, 0),  # 4 cyan
        (255, 0, 0),    # 5 blue
        (255, 0, 255),  # 6 magenta
        (255, 255, 255),# 7 white
    ]
    for i, p in enumerate(clicks[:8]):
        if p is None:
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (u, v), 6, colors[i], 2)
        cv2.putText(vis, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors[i], 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def draw_cuboid_overlay(img, proj_all, label, out_path, color=(0, 255, 0)):
    vis = img.copy()
    # Edges of cuboid (12 edges of a box, idx 0..7 cyclic faces)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # FRONT face
        (4, 5), (5, 6), (6, 7), (7, 4),   # BACK face
        (0, 4), (1, 5), (2, 6), (3, 7),   # connectors
    ]
    for a, b in edges:
        pa, pb = proj_all[a], proj_all[b]
        if pa[0] == -1.0 and pa[1] == -1.0: continue
        if pb[0] == -1.0 and pb[1] == -1.0: continue
        ua, va_ = int(round(pa[0])), int(round(pa[1]))
        ub, vb = int(round(pb[0])), int(round(pb[1]))
        cv2.line(vis, (ua, va_), (ub, vb), color, 2, cv2.LINE_AA)
    # corner labels
    for i in range(8):
        p = proj_all[i]
        if p[0] == -1.0 and p[1] == -1.0: continue
        u, v = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (u, v), 4, color, -1)
        cv2.putText(vis, str(i), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def enumerate_candidates(kps_2d, K, dims):
    """v7 _solve_pose_single 와 동일 init/flip set 으로 모든 candidate enumerate.

    Returns: list of dict (R, t, err, lr/tb/fr viol, proj_all, init_name)
    """
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    inits = []
    # (a) IPPE 6 face seeds
    for name, face in _CUBOID_FACES:
        for k, (R, t) in enumerate(_seed_from_ippe_face(kps_2d, K, kp3d, list(face))):
            inits.append((f"IPPE_{name}_{k}", R, t))
    # (b) EPNP / SQPNP
    for flag, flag_name in [(cv2.SOLVEPNP_EPNP, "EPNP"), (cv2.SOLVEPNP_SQPNP, "SQPNP")]:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((flag_name, R, tvec.flatten()))
        except cv2.error:
            pass
    # (c) IPPE all-valid
    try:
        ok_n, rvec_list, tvec_list, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for k, (rv, tv) in enumerate(zip(rvec_list, tvec_list)):
                if tv[2, 0] > 0:
                    R, _ = cv2.Rodrigues(rv)
                    inits.append((f"IPPE_all_{k}", R, tv.flatten()))
    except cv2.error:
        pass
    # (d) manual t
    cx_K, cy_K = K[0, 2], K[1, 2]
    fx_K = K[0, 0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w_ = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w_, 50.0))
    t_man = np.array([(mean_u - cx_K) * z_guess / fx_K,
                      (mean_v - cy_K) * z_guess / fx_K, z_guess], dtype=np.float64)
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append(("Rx180_manual", Rx180.copy(), t_man.copy()))
    inits.append(("Eye_manual", np.eye(3), t_man.copy()))

    flips = []
    for ax_rot_deg in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1, 0, 0), ax_rot_deg[0])
        ry = _rot_axis_angle((0, 1, 0), ax_rot_deg[1])
        rz = _rot_axis_angle((0, 0, 1), ax_rot_deg[2])
        flips.append((ax_rot_deg, rz @ ry @ rx))

    click_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    click_span = max(
        click_pts[:, 0].max() - click_pts[:, 0].min(),
        click_pts[:, 1].max() - click_pts[:, 1].min(), 50.0)
    z_far_limit = 50.0 * fx_K * max(dims) / click_span

    cands = []
    for init_name, R0, t0 in inits:
        for flip_deg, F in flips:
            R_init = R0 @ F
            res = _refine_with_init(obj, img, K, R_init, t0)
            if res is None: continue
            R, t = res
            if t[2] <= 0: continue
            if t[2] > z_far_limit: continue
            pts_cam = (R @ kp3d.T).T + t
            if (pts_cam[:, 2] <= 0).any(): continue
            lrv, tbv, frv, proj_all, _ = _eval_pair_invariants(R, t, K, kp3d)
            err = _reproj_err_dict(proj_all, valid_idx, kps_2d, weights=None)
            proj_8 = np.array(proj_all[:8], dtype=np.float64)
            bbox_w = float(proj_8[:, 0].max() - proj_8[:, 0].min())
            bbox_h = float(proj_8[:, 1].max() - proj_8[:, 1].min())
            cands.append({
                "init": init_name, "flip": flip_deg,
                "R": R, "t": t, "err": err,
                "lr_viol": lrv, "tb_viol": tbv, "fr_viol": frv,
                "viol_sum": lrv + tbv + frv,
                "proj_all": proj_all,
                "bbox_w": bbox_w, "bbox_h": bbox_h,
                "bbox_area": bbox_w * bbox_h,
                "tz": float(t[2]),
            })
    return cands


def analyze_face_orientation(R, t, kp3d):
    """Selected R 의 cuboid local FRONT face normal 의 cam-frame 방향 분석.

    v6 정의: FRONT face (0,1,2,3) = Z_local = -d/2 → face normal = -Z_local = (0,0,-1).
    cam-frame 으로 변환: R @ (0,0,-1) = -R[:,2].
    - Z_cam_normal < 0 (음수) = FRONT face 가 카메라 쪽 (-Z_cam = 가까이) → 정상.
    - Z_cam_normal > 0 (양수) = FRONT face 가 카메라 반대쪽 (멀리) → cuboid flip된 것.
    """
    front_normal_local = np.array([0.0, 0.0, -1.0])  # cuboid local frame
    front_normal_cam = R @ front_normal_local
    return {
        "front_normal_cam": front_normal_cam.tolist(),
        "front_normal_cam_z": float(front_normal_cam[2]),
        "front_facing_camera": bool(front_normal_cam[2] < 0),
        # FRONT face center cam.z vs BACK face center cam.z
        "front_center_cam_z": float(((R @ kp3d[[0, 1, 2, 3]].mean(axis=0)) + t)[2]),
        "back_center_cam_z":  float(((R @ kp3d[[4, 5, 6, 7]].mean(axis=0)) + t)[2]),
    }


def detect_pallet_region_from_screenshot():
    """스크린샷 (224x126 thumbnail) 분석 — wireframe 영역 위치 추정.

    스크린샷의 wireframe colored corners 가 image 영역 비율로 어디에 있는지 보고,
    실제 640x480 이미지의 대략적 위치로 매핑. 사용자 클릭은 wireframe 보다 일반적
    으로 약간 더 큰 영역 (사용자가 정확히 클릭하면 PnP 결과 = 클릭 위치).
    """
    # screenshot 분석: KEYPOINTS panel 의 색 점들이 image 의 약 35-45% x, 50-65% y
    # 비율 영역에 분포 → 640x480 에서 약 (224-288, 240-310) 영역.
    # 실제로는 멀리 있는 pallet 이라 더 작을 것.
    pass


def main():
    K = load_K()
    img = cv2.imread(IMG, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    print(f"Image: {W}x{H}, K=fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    # === Step 1: JSON exists? ===
    json_path = os.path.join(
        REPO, "challenge/data/capturepallet08_manual_gt/1778653498432396288.json")
    if os.path.exists(json_path):
        print(f"\n[Step1] Saved JSON found: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2))
        # extract manual_kps from there
        obj0 = data["objects"][0]
        cuboid_kps = obj0.get("projected_cuboid", [])
        centroid_kp = obj0.get("projected_cuboid_centroid", None)
        clicks = [list(p) if p is not None else None for p in cuboid_kps]
        while len(clicks) < 8:
            clicks.append(None)
        clicks.append(list(centroid_kp) if centroid_kp else None)
    else:
        print(f"\n[Step1] Saved JSON NOT found ({json_path})")
        print("        Reconstructing clicks from screenshot heuristic.")
        clicks = estimate_clicks_from_image(img)

    print(f"\nClicks (0..8):")
    for i, p in enumerate(clicks):
        tag = "MISSING" if p is None else f"({p[0]:.1f}, {p[1]:.1f})"
        print(f"  [{i}] {tag}")

    visualize_clicks(img, clicks, os.path.join(OUT, "v8_oblique_clicks.png"))

    # Click invariant violations (sanity)
    click_lr = _eval_click_lr_viol(clicks)
    click_tb = _eval_click_tb_viol(clicks)
    print(f"\nClick invariants: LR viol={click_lr}, TB viol={click_tb}")

    # === Step 2: solve_pose direct + candidate enum ===
    pose = solve_pose(clicks, K, img_shape=img.shape)
    if pose is None:
        print("\n[Step2] solve_pose returned None!")
        # Probably degenerate reject filtered all candidates. Try without img_shape
        # (uses K cx*cy*4 estimate — same), so we go enumerate directly.
        print("        Enumerating candidates directly (no reject):")
        all_cands_pre = []
        for dims_label, dims in [("110front", PALLET_DIMS),
                                  ("130front", (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))]:
            cands = enumerate_candidates(clicks, K, dims)
            for c in cands:
                c["dims_label"] = dims_label
            all_cands_pre.extend(cands)
        all_cands_pre.sort(key=lambda c: (c["viol_sum"], c["err"]))
        print(f"  Total: {len(all_cands_pre)}")
        # Show top 10 + bbox stats
        print(f"  Top 10 (viol_sum, err):")
        for i, c in enumerate(all_cands_pre[:10]):
            print(f"    [{i}] {c['dims_label']:>8} init={c['init']:<20} flip={c['flip']} "
                  f"err={c['err']:.2f} viol={c['viol_sum']} tz={c['tz']:.2f} "
                  f"bbox={c['bbox_area']:.0f}")
        img_area = 640 * 480
        min_area = 0.015 * img_area
        print(f"  Image area = {img_area}, min_bbox_area threshold = {min_area:.0f}")
        survived = [c for c in all_cands_pre if c["bbox_area"] >= min_area]
        print(f"  Surviving bbox filter: {len(survived)} (of {len(all_cands_pre)})")
        # Save table for these
        table_path = os.path.join(OUT, "v8_candidate_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("=" * 130 + "\n")
            f.write("v8 candidate table (solve_pose returned None) — full enum, sorted viol,err\n")
            f.write(f"Frame: 1778653498432396288  clicks: "
                    f"{[i for i,p in enumerate(clicks) if p is not None]}\n")
            f.write(f"img_area={img_area} min_bbox_area={min_area:.0f}\n")
            f.write(f"surviving bbox filter: {len(survived)}/{len(all_cands_pre)}\n")
            f.write("=" * 130 + "\n")
            f.write(f"{'#':>3} {'dims':>8} {'init':<22} {'flip(deg)':<16} "
                    f"{'err':>7} {'LR':>3} {'TB':>3} {'FR':>3} {'sum':>3} "
                    f"{'tz':>6} {'bbox':>8}\n")
            f.write("-" * 130 + "\n")
            for i, c in enumerate(all_cands_pre[:60]):
                surv = "*" if c["bbox_area"] >= min_area else ""
                f.write(f"{i:3d} {c['dims_label']:>8} {c['init']:<22} {str(c['flip']):<16} "
                        f"{c['err']:7.2f} {c['lr_viol']:3d} {c['tb_viol']:3d} {c['fr_viol']:3d} "
                        f"{c['viol_sum']:3d} {c['tz']:6.2f} {c['bbox_area']:8.0f}{surv}\n")
        print(f"  Table saved: {table_path}")
        return
    print(f"\n[Step2] solve_pose result:")
    print(f"  R = \n{pose['R']}")
    print(f"  t = {pose['t']}")
    print(f"  dims = {pose['dims']}  (110-front: {pose['dims']==PALLET_DIMS})")
    print(f"  reproj_error_px = {pose['reproj_error_px']:.3f}")
    print(f"  v6_lr/tb/fr viol = {pose['_v6_lr_viol']}/{pose['_v6_tb_viol']}/{pose['_v6_fr_viol']}")
    print(f"  v6_strict_passed = {pose['_v6_strict_passed']}")
    print(f"  n_candidates / n_strict_ok = {pose['_v6_n_candidates']}/{pose['_v6_n_strict_ok']}")

    draw_cuboid_overlay(img, pose["projected_all"],
                        f"v7 SELECTED reproj={pose['reproj_error_px']:.2f}px tz={pose['t'][2]:.2f}m",
                        os.path.join(OUT, "v8_oblique_before.png"),
                        color=(0, 255, 0))

    # Enumerate all candidates for both dims
    all_cands = []
    for dims_label, dims in [("110front", PALLET_DIMS),
                              ("130front", (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))]:
        cands = enumerate_candidates(clicks, K, dims)
        for c in cands:
            c["dims_label"] = dims_label
        all_cands.extend(cands)
    print(f"\n[Step2] Total candidates enumerated: {len(all_cands)}")

    # Sort by viol_sum then err
    all_cands.sort(key=lambda c: (c["viol_sum"], c["err"]))

    # Save table
    table_path = os.path.join(OUT, "v8_candidate_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write("v8 candidate table  (sorted: viol_sum, err)\n")
        f.write(f"Frame: 1778653498432396288  clicks: {[i for i,p in enumerate(clicks) if p is not None]}\n")
        f.write("=" * 130 + "\n")
        f.write(f"{'#':>3} {'dims':>8} {'init':<22} {'flip(deg)':<16} "
                f"{'err':>7} {'LR':>3} {'TB':>3} {'FR':>3} {'sum':>3} "
                f"{'tz':>6} {'bbox':>8}\n")
        f.write("-" * 130 + "\n")
        kp3d_a = make_pallet_keypoints_3d(*PALLET_DIMS)
        for i, c in enumerate(all_cands[:60]):
            f.write(f"{i:3d} {c['dims_label']:>8} {c['init']:<22} {str(c['flip']):<16} "
                    f"{c['err']:7.2f} {c['lr_viol']:3d} {c['tb_viol']:3d} {c['fr_viol']:3d} "
                    f"{c['viol_sum']:3d} {c['tz']:6.2f} {c['bbox_area']:8.0f}\n")
        f.write("-" * 130 + "\n")
        # strict-pass only
        strict = [c for c in all_cands if c["viol_sum"] == 0]
        f.write(f"\nstrict-pass candidates: {len(strict)}\n")
        if strict:
            strict.sort(key=lambda c: c["err"])
            f.write(f"top 10 strict-pass (sorted by err):\n")
            for i, c in enumerate(strict[:10]):
                fan = analyze_face_orientation(c["R"], c["t"], kp3d_a)
                f.write(f"  [{i}] err={c['err']:.2f} init={c['init']} flip={c['flip']} "
                        f"tz={c['tz']:.2f}  "
                        f"front_z={fan['front_center_cam_z']:.2f} "
                        f"back_z={fan['back_center_cam_z']:.2f} "
                        f"FRONT_facing_cam={fan['front_facing_camera']}\n")
        # numerical noise check
        if len(strict) >= 2:
            top5 = strict[:5]
            errs = [c["err"] for c in top5]
            span = max(errs) - min(errs)
            f.write(f"\ntop-5 strict err span: {span:.3f}px  "
                    f"(< 0.5 = numerical noise sensitive)\n")

    print(f"  Candidate table saved: {table_path}")

    # === Step 3: hypothesis check ===
    print(f"\n[Step3] Hypothesis check:")
    fan = analyze_face_orientation(pose["R"], pose["t"],
                                    make_pallet_keypoints_3d(*pose["dims"]))
    print(f"  (A) FRONT face normal cam-frame: {fan['front_normal_cam']}")
    print(f"      FRONT face center cam.z = {fan['front_center_cam_z']:.3f}")
    print(f"      BACK  face center cam.z = {fan['back_center_cam_z']:.3f}")
    print(f"      FRONT facing camera (normal_z<0): {fan['front_facing_camera']}")
    if not fan['front_facing_camera']:
        print(f"      >>> HYPOTHESIS A CONFIRMED: cuboid flipped, FRONT face is far away.")
    elif fan['front_center_cam_z'] >= fan['back_center_cam_z']:
        print(f"      >>> HYPOTHESIS A CONFIRMED: FRONT center farther than BACK.")
    else:
        print(f"      >>> A: cuboid orientation OK (FRONT closer).")

    # (B) numerical noise
    strict = [c for c in all_cands if c["viol_sum"] == 0]
    if len(strict) >= 2:
        strict_sorted = sorted(strict, key=lambda c: c["err"])
        top5_err = [c["err"] for c in strict_sorted[:5]]
        span = max(top5_err) - min(top5_err)
        print(f"  (B) Top-5 strict err: {[f'{e:.2f}' for e in top5_err]}  "
              f"span={span:.3f}px")
        if span < 0.5:
            print(f"      >>> HYPOTHESIS B CONFIRMED: numerical-noise sensitive.")
    else:
        print(f"  (B) Insufficient strict candidates ({len(strict)}) — "
              f"strict mode failed or rejected most.")

    # (C) missing pair count
    missing_idx = [i for i in range(8) if clicks[i] is None]
    unverif_lr = sum(1 for (a, b) in LR_PAIRS if a in missing_idx or b in missing_idx)
    unverif_tb = sum(1 for (a, b) in TB_PAIRS if a in missing_idx or b in missing_idx)
    unverif_fr = sum(1 for (a, b) in FR_PAIRS if a in missing_idx or b in missing_idx)
    print(f"  (C) Missing corners: {missing_idx}")
    print(f"      Unverifiable click-pairs: LR={unverif_lr}/4, "
          f"TB={unverif_tb}/4, FR={unverif_fr}/4")
    if (unverif_lr + unverif_tb + unverif_fr) >= 6:
        print(f"      >>> HYPOTHESIS C CONFIRMED: ≥50% pairs unverifiable from clicks.")

    # (D) click vs projected for 5, 6
    print(f"  (D) Click vs projected distance (selected pose):")
    for i in [0, 1, 2, 3, 5, 6]:
        if clicks[i] is None: continue
        p = pose["projected_all"][i]
        d = float(np.hypot(p[0] - clicks[i][0], p[1] - clicks[i][1]))
        print(f"      [{i}] click=({clicks[i][0]:.1f},{clicks[i][1]:.1f}) "
              f"proj=({p[0]:.1f},{p[1]:.1f})  dist={d:.2f}px")

    # === Save diagnostic JSON ===
    diag_path = os.path.join(OUT, "v8_pose_diag.json")
    with open(diag_path, "w") as f:
        json.dump({
            "clicks": clicks,
            "pose_R": pose["R"].tolist(),
            "pose_t": pose["t"].tolist(),
            "dims": list(pose["dims"]),
            "reproj_error_px": pose["reproj_error_px"],
            "v6_lr_viol": pose["_v6_lr_viol"],
            "v6_tb_viol": pose["_v6_tb_viol"],
            "v6_fr_viol": pose["_v6_fr_viol"],
            "v6_strict_passed": pose["_v6_strict_passed"],
            "n_candidates": pose["_v6_n_candidates"],
            "n_strict_ok": pose["_v6_n_strict_ok"],
            "face_orientation": fan,
            "click_lr_viol": click_lr,
            "click_tb_viol": click_tb,
            "unverifiable_pairs": {"LR": unverif_lr, "TB": unverif_tb, "FR": unverif_fr},
        }, f, indent=2)
    print(f"\n  Diag JSON saved: {diag_path}")


if __name__ == "__main__":
    main()
