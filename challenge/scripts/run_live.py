"""challenge/scripts/run_live.py — main entry.

Challenge: Real-time Pallet 6D Pose (overtuned, low false positive).

본 스크립트는 scripts/dope/run_dope_live.py 의 분기 버전. 차이점:
  1. 기본 threshold 상향 (0.30) — config/task.yaml inference.belief.*
  2. PnP 결과 sanity gate (z range / kp count / reproj / depth-PnP / edge ratio)
  3. Temporal confirm — N프레임 연속 통과 시에만 "CONFIRMED"
  4. NOT DETECTED 시 어떤 gate 에서 실패했는지 화면 표시
  5. Sequence 재생 모드 (--seq) + 일시정지 / 점프 / fps 조절

분리된 모듈:
  run_live_io.py      NpDepthFrame / load_seq / sample_depth / scale_K /
                       run_forward / extract_peaks / build_belief_grid
  run_live_gates.py   sanity gate (kp_count / reproj / edge_ratio / evaluate_result)
  run_live_draw.py    draw_cuboid / build_live_panel / CUBOID_EDGES

Keys: q=종료  s=저장  b=belief 토글  r=auto-tune 리셋
      space=pause  n/p=±1 frame  ,/.=±10  ]/[ = fps up/down
"""
from __future__ import annotations
import argparse
import os
import sys

import cv2
import yaml
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run_live_* import

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from pyrr import Quaternion, matrix33

from run_live_io import (
    NpDepthFrame, load_seq, sample_depth, scale_K,
    run_forward, extract_peaks, build_belief_grid,
)
from run_live_gates import evaluate_result
from run_live_draw import draw_cuboid, build_live_panel, noop
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
from challenge.data_paths import get as _dp  # 경로 단일 출처


def enforce_camera_facing(result, pnp_solver):
    """Camera-facing convention 강제 (2026-05-22 결정).

    PnP 후 cuboid 8 corner 의 camera-frame z 비교 → 0~3 이 카메라 가까운 면 (near face)
    이 되도록 idx swap. 학습 데이터는 object-fixed (0~3 = SDG 의 z_max corner) 로
    생성됐지만, 추론 단에서 camera-facing 으로 통일.

    swap_map: Ry(180°) cuboid 회전에 대응. 0↔5, 1↔4, 2↔7, 3↔6.
       (LR 뒤집힘 동반 — plastic pallet 의 양면 시각 비슷해서 무방)
    R/t (location/quaternion) 은 model frame 기준으로 보존. user-facing idx 만 swap.
    forklift 응용에서 R 사용 시 별도 매핑 필요 (TODO).
    """
    loc = result.get("location")
    quat = result.get("quaternion")
    raw = result.get("raw_points")
    proj = result.get("projected_points")
    if loc is None or quat is None or pnp_solver._cuboid3d is None:
        return result

    q = Quaternion(quat)
    R = matrix33.create_from_quaternion(q)
    t = np.array(loc, dtype=np.float64)
    obj_pts = np.array(pnp_solver._cuboid3d.get_vertices()[:8], dtype=np.float64)
    pts_cam = (R @ obj_pts.T).T + t
    z_cam = pts_cam[:, 2]
    if z_cam[:4].mean() <= z_cam[4:].mean():
        return result   # 이미 0~3 이 가까이

    # Ry(180°) cuboid 회전 매핑 (LR 뒤집힘 동반)
    swap_map = [5, 4, 7, 6, 1, 0, 3, 2, 8]
    if raw is not None:
        result["raw_points"] = [raw[swap_map[i]] if swap_map[i] < len(raw) else None
                                for i in range(9)]
    if proj is not None:
        new_proj = [None] * 9
        for i in range(9):
            j = swap_map[i]
            if j < len(proj):
                new_proj[i] = proj[j]
        result["projected_points"] = new_proj
    return result


class State:
    """Belief click auto-tune state (DOPE detector 와 다른 사용자 GUI state)."""
    vertex2 = None
    sigma = 3
    cell_w = 0
    cell_h = 0
    grid_scale = 1.0
    clicks = []
    auto_threshold = None


def on_belief_click(event, x, y, flags, state):
    """Belief grid 클릭 시 그 채널의 peak 값을 sample → auto-threshold 추정."""
    if event != cv2.EVENT_LBUTTONDOWN or state.vertex2 is None:
        return
    if state.cell_w == 0 or state.cell_h == 0:
        return
    gx = int(x / state.grid_scale); gy = int(y / state.grid_scale)
    col = gx // state.cell_w; row = gy // state.cell_h
    ch = row * 3 + col
    if not (0 <= ch < 9):
        return
    bx = (gx - col * state.cell_w) // 8
    by = (gy - row * state.cell_h) // 8
    raw = state.vertex2[ch].cpu().numpy()
    sm = gaussian_filter(raw, sigma=state.sigma)
    bh, bw = sm.shape
    bx, by = np.clip(bx, 0, bw - 1), np.clip(by, 0, bh - 1)
    r = 2
    patch = sm[max(0, by-r):min(bh, by+r+1), max(0, bx-r):min(bw, bx+r+1)]
    val = float(patch.max())
    state.clicks.append((ch, val))
    name = f"corner{ch}" if ch < 8 else "center"
    print(f"\n[Click] {name}: peak={val:.6f}")
    min_val = min(v for _, v in state.clicks)
    state.auto_threshold = min_val * 1.0
    chs = sorted(set(c for c, _ in state.clicks))
    print(f"[Auto-tune] min={min_val:.6f} → threshold≈{state.auto_threshold:.6f} (clicked={chs})")


def _setup_input(args, C):
    """입력 소스 (seq / realsense / cam) 초기화. (K, seq_frames, pipeline, align, cap, use_depth)."""
    pipeline = align = cap = None
    seq_frames = None
    use_depth = (args.realsense or args.seq) and not args.no_depth

    if args.seq:
        seq_dir = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO_ROOT, args.seq)
        seq_frames, K_seq = load_seq(seq_dir)
        if K_seq is None:
            K = np.array([[C["camera"]["fx"], 0, C["camera"]["cx"]],
                          [0, C["camera"]["fy"], C["camera"]["cy"]],
                          [0, 0, 1]], dtype=np.float64)
        else:
            K = K_seq
        print(f"[Seq] {seq_dir} — {len(seq_frames)} frames, fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    elif args.realsense:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        rc = rs.config()
        rc.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
        if use_depth:
            rc.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
            align = rs.align(rs.stream.color)
        profile = pipeline.start(rc)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        print(f"[RealSense] fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    else:
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("카메라 열기 실패"); raise SystemExit(1)
        K = np.array([[C["camera"]["fx"], 0, args.width/2],
                      [0, C["camera"]["fy"], args.height/2],
                      [0, 0, 1]], dtype=np.float64)
    return K, seq_frames, pipeline, align, cap, use_depth


def _setup_controls(ctrl_win, bel, gates):
    """슬라이더 7개 생성."""
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 500, 250)
    cv2.createTrackbar("threshold(x1000)", ctrl_win, int(bel["threshold"] * 1000), 500, noop)
    cv2.createTrackbar("thresh_map(x1000)", ctrl_win, int(bel["thresh_map"] * 1000), 500, noop)
    cv2.createTrackbar("thresh_pts(x1000)", ctrl_win, int(bel["thresh_points"] * 1000), 500, noop)
    cv2.createTrackbar("thresh_ang(x100)", ctrl_win, int(bel["thresh_angle"] * 100), 100, noop)
    cv2.createTrackbar("sigma", ctrl_win, bel["sigma"], 10, noop)
    cv2.createTrackbar("min_kp", ctrl_win, gates["min_detected_keypoints"], 9, noop)
    cv2.createTrackbar("max_reproj_px", ctrl_win, int(gates["max_reproj_error_px"]), 30, noop)


def _read_trackbars(ctrl_win, cfg, gates):
    """슬라이더 → cfg / gates_live 반영."""
    cfg.threshold     = cv2.getTrackbarPos("threshold(x1000)", ctrl_win) / 1000.0
    cfg.thresh_map    = cv2.getTrackbarPos("thresh_map(x1000)", ctrl_win) / 1000.0
    cfg.thresh_points = cv2.getTrackbarPos("thresh_pts(x1000)", ctrl_win) / 1000.0
    cfg.thresh_angle  = cv2.getTrackbarPos("thresh_ang(x100)", ctrl_win) / 100.0
    cfg.sigma         = max(1, cv2.getTrackbarPos("sigma", ctrl_win))
    gates_live = dict(gates)
    gates_live["min_detected_keypoints"] = max(4, cv2.getTrackbarPos("min_kp", ctrl_win))
    gates_live["max_reproj_error_px"]    = max(2, cv2.getTrackbarPos("max_reproj_px", ctrl_win))
    return gates_live


def _draw_detection(img_draw, best, best_info, confirmed, pnp_solver, K_small, dist_coeffs):
    """검출된 result 를 image_draw 에 시각화 — wireframe + keypoint + yaw 화살표."""
    raw_points = best["raw_points"]
    proj_pts = best.get("projected_points")
    color_kp = (0, 255, 0) if confirmed else (0, 200, 200)
    if proj_pts is not None:
        c_front = (0, 255, 0) if confirmed else (0, 200, 200)
        c_back  = (0, 160, 0) if confirmed else (0, 130, 130)
        draw_cuboid(img_draw, proj_pts, c_front, c_back, thickness=2)
        for i, pt in enumerate(proj_pts):
            if i < 8 and pt is not None:
                cv2.drawMarker(img_draw, (int(pt[0]), int(pt[1])),
                               color_kp, cv2.MARKER_SQUARE, 10, 2)
    for i, pt in enumerate(raw_points):
        if pt is None:
            continue
        px, py = int(pt[0]), int(pt[1])
        if i == 8:
            cv2.circle(img_draw, (px, py), 7,
                       (0, 0, 255) if confirmed else (0, 150, 200), -1)
        else:
            cv2.circle(img_draw, (px, py), 5, color_kp, -1)
            cv2.putText(img_draw, str(i), (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_kp, 1)

    # Yaw 화살표
    loc = best["location"]; ori = best["quaternion"]
    q = Quaternion(ori)
    rot_mat = matrix33.create_from_quaternion(q)
    yaw_rad = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])
    yaw_deg = np.degrees(yaw_rad)
    rvec, _ = cv2.Rodrigues(rot_mat)
    tvec = np.array(loc, dtype=np.float32).reshape(3, 1)
    c3d = np.array(pnp_solver._cuboid3d.get_vertices()[8], dtype=np.float32)
    f3d = c3d + np.array([0, 0, 50], dtype=np.float32)
    pts, _ = cv2.projectPoints(np.array([c3d, f3d]), rvec, tvec, K_small, dist_coeffs)
    p1 = tuple(pts[0].ravel().astype(int))
    p2 = tuple(pts[1].ravel().astype(int))
    cv2.arrowedLine(img_draw, p1, p2, (0, 255, 255), 3, tipLength=0.15)
    cv2.putText(img_draw, f"Yaw: {yaw_deg:.1f}\xb0  z={best_info['z_m']:.2f}m",
                (p1[0], p1[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def _handle_seq_key(key, paused, seq_idx, seq_total, seq_fps_cur):
    """sequence 모드 키 처리. (paused, seq_idx, seq_fps_cur, quit_flag) 반환."""
    quit_flag = False
    if key == ord('q'):
        quit_flag = True
    elif key == ord(' '):
        paused = not paused
        print(f"[Pause] {'ON' if paused else 'OFF'}")
    elif key == ord('n'):
        if not paused:
            paused = True
        seq_idx = min(seq_idx + 1, seq_total - 1)
    elif key == ord('p'):
        if not paused:
            paused = True
            seq_idx = max(seq_idx - 2, 0)
        else:
            seq_idx = max(seq_idx - 1, 0)
    elif key == ord('.'):
        if not paused:
            paused = True
        seq_idx = min(seq_idx + 10, seq_total - 1)
    elif key == ord(','):
        if not paused:
            paused = True
            seq_idx = max(seq_idx - 11, 0)
        else:
            seq_idx = max(seq_idx - 10, 0)
    elif key in (ord(']'), ord('=')):
        seq_fps_cur = min(60.0, max(0.5, seq_fps_cur * 1.5))
        print(f"[fps] {seq_fps_cur:.2f}")
    elif key in (ord('['), ord('-')):
        seq_fps_cur = min(60.0, max(0.5, seq_fps_cur / 1.5))
        print(f"[fps] {seq_fps_cur:.2f}")
    return paused, seq_idx, seq_fps_cur, quit_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default=os.path.join(_REPO_ROOT, "challenge", "config", "task.yaml"))
    ap.add_argument("--weights",   default=None, help="override config baseline.weights")
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--seq",       default=None, help="data/outside/capture* 같은 시퀀스 폴더")
    ap.add_argument("--seq_fps",   type=float, default=15.0, help="시퀀스 재생 속도 (0=프레임당 1키)")
    ap.add_argument("--seq_loop",  action="store_true", help="시퀀스 끝나면 처음으로")
    ap.add_argument("--no_depth",  action="store_true")
    ap.add_argument("--cam_id",    type=int, default=0)
    ap.add_argument("--width",     type=int, default=1280)
    ap.add_argument("--height",    type=int, default=720)
    ap.add_argument("--out_dir",   default=_dp("real.live_captures_EMPTY"))
    ap.add_argument("--label",     default="", help="창 이름 접미사 (여러 인스턴스 비교용)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    weights = args.weights or os.path.join(_REPO_ROOT, C["baseline"]["weights"])
    gates = C["inference"]["gates"]
    bel = C["inference"]["belief"]
    temporal = C["inference"]["temporal"]

    dim_cm = [C["pallet"]["width"] * 100,
              C["pallet"]["height"] * 100,
              C["pallet"]["depth"] * 100]

    out_dir = os.path.join(_REPO_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 입력 소스 ─────────────────────────────────────────────────────────────
    K, seq_frames, pipeline, align, cap, use_depth = _setup_input(args, C)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ── DOPE 모델 ─────────────────────────────────────────────────────────────
    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = bel["threshold"]
        softmax = 1000
        thresh_angle = bel["thresh_angle"]
        thresh_map = bel["thresh_map"]
        sigma = bel["sigma"]
        thresh_points = bel["thresh_points"]
    cfg = Cfg()

    print(f"[모델] {weights} 로딩...")
    sd_keys = list(torch.load(weights, map_location="cpu").keys())
    parallel = sd_keys[0].startswith("module.") if sd_keys else False
    print(f"[모델] DataParallel checkpoint: {parallel}")
    model = ModelData(name="pallet", net_path=weights, parallel=parallel)
    model.load_net_model()
    pnp_solver = CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(dim_cm))
    pnp_solver.set_dist_coeffs(dist_coeffs)

    # ── 슬라이더 + GUI ─────────────────────────────────────────────────────────
    suffix = f" [{args.label}]" if args.label else ""
    ctrl = f"Controls{suffix}"
    win_live = f"Challenge Live{suffix}"
    win_belief = f"DOPE Belief Maps{suffix}"
    _setup_controls(ctrl, bel, gates)

    state = State()
    cv2.namedWindow(win_belief)
    cv2.setMouseCallback(win_belief, on_belief_click, state)

    print("[조작] q=종료  s=저장  b=belief 토글  r=auto-tune 리셋")
    print(f"[Gates] min_kp={gates['min_detected_keypoints']}  max_reproj={gates['max_reproj_error_px']}px  "
          f"z=[{gates['z_min_m']:.1f},{gates['z_max_m']:.1f}]m  depth_rel<{gates['depth_pnp_z_max_rel']:.2f}")

    frame_idx = 0
    seq_idx = 0
    show_belief = True
    consecutive_ok = 0
    paused = False
    seq_fps_cur = max(0.0, float(args.seq_fps))

    while True:
        depth_frame = None
        # ── 프레임 획득 ───────────────────────────────────────────────────────
        if seq_frames is not None:
            if seq_idx >= len(seq_frames):
                if args.seq_loop:
                    seq_idx = 0
                else:
                    print("[Seq] end of sequence")
                    break
            rgb_path, depth_path = seq_frames[seq_idx]
            img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if img is None:
                seq_idx += 1; continue
            if use_depth and depth_path is not None:
                d_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if d_u16 is not None and d_u16.dtype == np.uint16:
                    depth_frame = NpDepthFrame(d_u16)
            if not paused:
                seq_idx += 1
        elif args.realsense:
            frames = pipeline.wait_for_frames()
            if use_depth and align:
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame() or None
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
        else:
            ret, img = cap.read()
            if not ret:
                break

        # Auto-tune 클릭값 반영
        if state.auto_threshold is not None:
            t = state.auto_threshold
            cv2.setTrackbarPos("threshold(x1000)", ctrl, max(0, min(500, int(t * 1000))))
            cv2.setTrackbarPos("thresh_map(x1000)", ctrl, max(0, min(500, int(t * 1000))))
            cv2.setTrackbarPos("thresh_pts(x1000)", ctrl, max(0, min(500, int(t * 1000))))
            state.auto_threshold = None

        # 슬라이더 → cfg + gates
        gates_live = _read_trackbars(ctrl, cfg, gates)

        # ── 전처리 + Forward ──────────────────────────────────────────────────
        h, w = img.shape[:2]
        proc_scale = 400.0 / h
        new_w = int(w * proc_scale) & ~7
        img_small = cv2.resize(img, (new_w, 400))
        K_small = scale_K(K, proc_scale)
        pnp_solver.set_camera_intrinsic_matrix(K_small)
        img_rgb = img_small[..., ::-1].copy()

        vertex2, aff = run_forward(model.net, img_rgb)
        state.vertex2 = vertex2
        state.sigma = cfg.sigma

        peaks = extract_peaks(vertex2, sigma=cfg.sigma)
        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, cfg)
        except Exception:
            results = []

        img_draw = img_small.copy()

        # (1) Raw peak 항상 표시 (디버그용)
        for i, pk in enumerate(peaks):
            px, py = int(pk['x']), int(pk['y'])
            if not (0 <= px < img_draw.shape[1] and 0 <= py < img_draw.shape[0]):
                continue
            above = pk['val'] > cfg.threshold
            clr = ((0, 0, 200) if i == 8 else (0, 200, 0)) if above else (80, 80, 80)
            cv2.circle(img_draw, (px, py), 4 if i == 8 else 3, clr,
                       1 if not above else 2)
            cv2.putText(img_draw, f"{pk['val']:.3f}", (px + 6, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)

        # (2) PnP 결과 gate 평가
        best = None
        best_info = None
        last_reason = "no_result"
        for result in results:
            # Camera-facing convention 강제 — 0~3 이 카메라 가까운 면 되도록 swap
            result = enforce_camera_facing(result, pnp_solver)
            depth_cm = None
            raw = result.get("raw_points")
            if depth_frame is not None and raw is not None and raw[8] is not None:
                d_m = sample_depth(depth_frame,
                                   raw[8][0] / proc_scale,
                                   raw[8][1] / proc_scale)
                if d_m is not None:
                    depth_cm = d_m * 100.0
            ok, reason, info = evaluate_result(result, gates_live, depth_cm, K_small)
            last_reason = reason
            if ok:
                best = result
                best_info = info
                break

        consecutive_ok = consecutive_ok + 1 if best is not None else 0
        confirmed = consecutive_ok >= temporal["confirm_frames"]

        # (3) 검출 시각화
        if best is not None:
            _draw_detection(img_draw, best, best_info, confirmed,
                            pnp_solver, K_small, dist_coeffs)

        # (4) 상태 배너
        if best is None:
            cv2.putText(img_draw, f"NOT DETECTED ({last_reason})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
            status_text = "NOT DETECTED"
            status_sub = last_reason[:24]
            status_color = (0, 0, 220)
        elif not confirmed:
            cv2.putText(img_draw, f"PENDING {consecutive_ok}/{temporal['confirm_frames']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            status_text = f"PENDING {consecutive_ok}/{temporal['confirm_frames']}"
            status_sub = ""
            status_color = (0, 200, 200)
        else:
            cv2.putText(img_draw, "CONFIRMED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status_text = "CONFIRMED"
            status_sub = ""
            status_color = (0, 255, 0)

        # 우측 패널 hstack
        panel = build_live_panel(
            img_draw.shape[0],
            is_seq=(seq_frames is not None),
            paused=paused,
            seq_idx=seq_idx,
            seq_total=len(seq_frames) if seq_frames is not None else 0,
            seq_fps=seq_fps_cur,
            cfg_thr=cfg.threshold,
            gates_live=gates_live,
            status=status_text,
            status_color=status_color,
        )
        if status_sub:
            cv2.putText(panel, status_sub, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        img_draw = np.hstack([img_draw, panel])

        cv2.imshow(win_live, img_draw)

        if show_belief:
            grid = build_belief_grid(vertex2, img_small)
            state.cell_w = new_w
            state.cell_h = 400
            max_w = 900
            if grid.shape[1] > max_w:
                s = max_w / grid.shape[1]
                grid_show = cv2.resize(grid, None, fx=s, fy=s)
                state.grid_scale = s
            else:
                grid_show = grid
                state.grid_scale = 1.0
            cv2.imshow(win_belief, grid_show)

        if frame_idx % 60 == 0:
            print(f"[F{frame_idx}] thr={cfg.threshold:.3f} | "
                  f"results={len(results)} best={'Y' if best else 'N'} "
                  f"confirm={consecutive_ok}/{temporal['confirm_frames']} | "
                  f"last_reason={last_reason}")

        # 키 입력
        if seq_frames is not None and seq_fps_cur > 0 and not paused:
            wait_ms = max(1, int(1000.0 / seq_fps_cur))
        else:
            wait_ms = 1 if not paused else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if seq_frames is not None:
            paused, seq_idx, seq_fps_cur, quit_flag = _handle_seq_key(
                key, paused, seq_idx, len(seq_frames), seq_fps_cur)
            if quit_flag:
                break
        # 공통 키 (live + seq 모두)
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = os.path.join(out_dir, f"live_{frame_idx:04d}.png")
            cv2.imwrite(path, img_draw)
            print(f"[저장] {path}")
        elif key == ord('b'):
            show_belief = not show_belief
            if not show_belief:
                try: cv2.destroyWindow(win_belief)
                except Exception: pass
        elif key == ord('r'):
            state.clicks = []
            state.auto_threshold = None
            print("[Reset] Auto-tune 초기화")

        frame_idx += 1

    if pipeline: pipeline.stop()
    if cap: cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
