# main_rec.py (실행 즉시 자동 녹화 시작 버전: 첫 프레임 생성 시 VideoWriter 자동 초기화)

import argparse
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import math
from datetime import datetime
from collections import deque

from calib.config import (
    SAMPLE_STRIDE, Z_INLIER_THRESH, MIN_POINTS,
    EMA_ALPHA_OFFSET, EMA_ALPHA_YAW, EMA_ALPHA_WIDTH,
    COLOR_ALERT, COLOR_STATUS_OK, COLOR_META, COLOR_BOX, COLOR_CNT, COLOR_CENTER,
    COLOR_YAW, COLOR_OFFSET, COLOR_WIDTH,
    USE_6D_MODE, USE_PERCEPTION_YOLO, POSE_BACKEND, MODEL_PATH_6D_YOLO,
    MODEL_PATH_6D, DOPE_INPUT_HEIGHT, DOPE_PEAK_THRESHOLD, DOPE_PEAK_SIGMA,
    DOPE_THRESH_MAP, DOPE_THRESH_POINTS, DOPE_THRESH_ANGLE, DOPE_SOFTMAX,
    DOPE_GATE_MIN_KP, DOPE_GATE_MAX_REPROJ_PX, DOPE_GATE_Z_MIN_M, DOPE_GATE_Z_MAX_M,
    DOPE_GATE_DEPTH_REL, DOPE_GATE_EDGE_RATIO_TOL, DOPE_CONFIRM_FRAMES,
    PALLET_WIDTH_M, PALLET_HEIGHT_M, PALLET_DEPTH_M,
    CAM_TO_FORK_T, CAM_TO_FORK_RPY_DEG,
    DEPTH_CORRECT_Z_MIN_M, DEPTH_CORRECT_Z_MAX_M,
)
from calib.hud import draw_panel
from calib.fsm import CalibrationFSM
from calib.control import can_init, can_close, is_mock as can_is_mock
from calib.utils import fmt_deg, fmt_m
from calib.pose6d_adapter import pose6d_to_align_vars, depth_scale_correct, apply_cam_to_fork
from calib.dope_inference import _sample_depth
from ui.diagram import draw_fsm_diagram_panel

# YOLO segmentation + RGB-D RANSAC 은 USE_PERCEPTION_YOLO=True 일 때만 import 시도.
if USE_PERCEPTION_YOLO:
    from calib.perception import Perception
    from calib.geometry import robust_points_from_mask_or_roi, fit_plane_yaw_from_points


def setup_video_writer_filename():
    rec_dir = "./rec"
    os.makedirs(rec_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(rec_dir, f"forklift_recording_{ts}.mp4")


def setup_video_writer_filenames():
    """HUD + raw 두 파일명 페어 반환 (동일 timestamp 로 세션 페어링)."""
    rec_dir = "./rec"
    os.makedirs(rec_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        os.path.join(rec_dir, f"forklift_recording_{ts}.mp4"),  # HUD + FSM diagram
        os.path.join(rec_dir, f"forklift_raw_{ts}.mp4"),         # raw color (학습용)
    )


def realsense_check_or_exit():
    ctx = rs.context()
    devs = ctx.query_devices()
    if devs.size() == 0:
        print("❌ RealSense 카메라를 찾을 수 없습니다. USB3 포트/케이블을 확인하세요.")
        sys.exit(1)
    print(f"✅ RealSense 장치 {devs.size()}대 연결됨.")
    for d in devs:
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        sn   = d.get_info(rs.camera_info.serial_number) if d.supports(rs.camera_info.serial_number) else "N/A"
        print(f"  - {name} (S/N {sn})")
    # 시작 시 hardware_reset — 이전 process 가 USB 깨끗하게 release 안 했을 때
    # 발생하는 "Frame didn't arrive within 5000" 방지 (ESC 종료 후 재시작 안정성).
    try:
        for d in devs:
            d.hardware_reset()
        print("🔄 RealSense hardware_reset 완료 — 재인식 대기 (5s)...")
        import time as _time
        _time.sleep(5)
    except Exception as e:
        print(f"⚠️  hardware_reset 실패 (무시): {e}")


# === rel_yaw.py 로직 이식: gyro.Y 적분 기반 상대 yaw 추정기 ===
class RelYawEstimator:
    """
    rel_yaw 계산:
      - 자이로 y(rad/s) 적분 → yaw(deg)
      - 초기 기준 init_yaw = 0.0
      - [-180, +180]로 angle wrapping
    가속도는 roll/pitch 보정용 보조 신호로만 사용.
    """
    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.first = True
        self.last_ts_ms = None
        self.yaw_deg = 0.0     # 적분 누적 yaw(deg)
        self.init_yaw = None   # 기준 yaw(deg)

        # 보조: 가속도 기반 각도(roll/pitch)
        self.accel_angle_x = 0.0
        self.accel_angle_z = 0.0

        # 최근 rel_yaw 캐시(센서 드롭 시 유지)
        self.last_rel = 0.0

    def update_from_frames(self, accel, gyro, ts_ms: float) -> float:
        """
        accel: rs.motion_data (x,y,z) [m/s^2]
        gyro : rs.motion_data (x,y,z) [rad/s]
        ts_ms: 타임스탬프(ms)

        Returns:
            rel_yaw(deg) in [-180, +180]
        """
        if self.first:
            self.first = False
            self.last_ts_ms = ts_ms
            # 가속도 기반 각도(roll/pitch) 초기화
            self.accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            self.accel_angle_x = math.degrees(math.atan2(
                accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
            # 초기 yaw = 0 기준
            self.init_yaw = 0.0
            self.last_rel = 0.0
            return 0.0

        dt = max(0.0, (ts_ms - self.last_ts_ms) / 1000.0)
        self.last_ts_ms = ts_ms

        # gyro 적분 (rad/s * s -> rad -> deg)
        dangleY = math.degrees(gyro.y * dt)

        # accel 기반 roll/pitch(보조 정보) 계산 및 보정 (yaw에는 미적용)
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(math.atan2(
            accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
        totalgyroangleX = accel_angle_x
        totalgyroangleZ = accel_angle_z
        self.accel_angle_x = totalgyroangleX * self.alpha + accel_angle_x * (1 - self.alpha)
        self.accel_angle_z = totalgyroangleZ * self.alpha + accel_angle_z * (1 - self.alpha)

        # === 핵심: yaw는 "gyro.Y" 적분 ===
        self.yaw_deg += dangleY

        if self.init_yaw is None:
            self.init_yaw = self.yaw_deg

        rel_yaw = self.yaw_deg - self.init_yaw

        # [-180, +180]로 wrapping
        rel_yaw = (rel_yaw + 180.0) % 360.0 - 180.0

        self.last_rel = rel_yaw
        return rel_yaw


def main():
    # CLI: --no-raw 면 학습용 raw color 녹화 비활성. default = HUD + raw 둘 다 녹화.
    ap = argparse.ArgumentParser(
        description="Forklift main_rec — RealSense + DOPE 6D + FSM (default: HUD + raw 둘 다 녹화)",
    )
    ap.add_argument("--no-raw", action="store_true",
                    help="학습용 raw color 녹화 비활성 (default: 활성)")
    args = ap.parse_args()
    record_raw = not args.no_raw

    realsense_check_or_exit()

    # CAN 초기화 — canlib DLL 없는 환경은 mock 으로 silent 동작 (실제 송수신 없음)
    print("CAN 통신 초기화 중...")
    can_ok = can_init()
    if can_ok and not can_is_mock():
        print("✅ CAN 통신 초기화 성공(하트비트 자동)")
    elif can_ok and can_is_mock():
        print("⚠️  CAN 통신: canlib DLL 부재 — MOCK 모드 (실제 송수신 없음, FSM 시각화만 동작)")
        print("    상세 로그 보려면: $env:CAN_MOCK_VERBOSE=1 (PowerShell)  또는  CAN_MOCK_VERBOSE=1 (bash)")
    else:
        print("❌ CAN 통신 초기화 실패 — MOCK 모드로 fallback 합니다 (FSM 시각화만 동작)")

    # 모듈
    if USE_PERCEPTION_YOLO:
        perception = Perception()
    else:
        perception = None
    fsm = CalibrationFSM()

    # === 6D pose 추론기 (backend = dope | yolo) ===
    # 두 backend 모두 동일한 infer_pose(bgr, K, depth_frame) → dict 계약을 따르므로
    # 아래 추론/FSM 루프는 backend 와 무관하게 동일하게 동작한다. (dope_pose 변수 재사용)
    dope_pose = None
    _gates = {
        "min_kp": DOPE_GATE_MIN_KP,
        "max_reproj_px": DOPE_GATE_MAX_REPROJ_PX,
        "z_min_m": DOPE_GATE_Z_MIN_M,
        "z_max_m": DOPE_GATE_Z_MAX_M,
        "depth_rel": DOPE_GATE_DEPTH_REL,
        "edge_ratio_tol": DOPE_GATE_EDGE_RATIO_TOL,
    }
    if USE_6D_MODE and POSE_BACKEND == "yolo":
        try:
            from calib.yolo_inference import YoloPoseEstimator
            dope_pose = YoloPoseEstimator(
                weights_path=MODEL_PATH_6D_YOLO,
                pallet_width_m=PALLET_WIDTH_M,
                pallet_height_m=PALLET_HEIGHT_M,
                pallet_depth_m=PALLET_DEPTH_M,
                gates=_gates,
                confirm_frames=DOPE_CONFIRM_FRAMES,
            )
            print(f"✅ YOLO 6D pose estimator 초기화 완료 ({MODEL_PATH_6D_YOLO})")
        except Exception as e:
            print(f"❌ YOLO 6D pose estimator 초기화 실패: {e}")
            print("   기존 RGB-D RANSAC perception 모드로 fallback")
            dope_pose = None
    elif USE_6D_MODE:
        try:
            from calib.dope_inference import DopePoseEstimator
            dope_pose = DopePoseEstimator(
                weights_path=MODEL_PATH_6D,
                pallet_width_m=PALLET_WIDTH_M,
                pallet_height_m=PALLET_HEIGHT_M,
                pallet_depth_m=PALLET_DEPTH_M,
                input_height=DOPE_INPUT_HEIGHT,
                peak_threshold=DOPE_PEAK_THRESHOLD,
                peak_sigma=DOPE_PEAK_SIGMA,
                thresh_map=DOPE_THRESH_MAP,
                thresh_points=DOPE_THRESH_POINTS,
                thresh_angle=DOPE_THRESH_ANGLE,
                softmax=DOPE_SOFTMAX,
                gates=_gates,
                confirm_frames=DOPE_CONFIRM_FRAMES,
            )
            print("✅ DOPE 6D pose estimator 초기화 완료")
        except Exception as e:
            print(f"❌ DOPE 6D pose estimator 초기화 실패: {e}")
            print("   기존 RGB-D RANSAC perception 모드로 fallback")
            dope_pose = None

    # === RealSense 파이프라인 구성 ===
    pipeline = rs.pipeline()
    cfg = rs.config()
    # RGB-D: 640×480 @ 15fps — challenge/data/*.png 학습 데이터 해상도와 동일.
    # challenge/config/task.yaml: camera.width=640, height=480, fx=614.18, fy=614.31.
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    # IMU: rel_yaw 계산에 필수
    cfg.enable_stream(rs.stream.accel)
    cfg.enable_stream(rs.stream.gyro)

    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    # === 녹화: 실행 즉시 자동 녹화 시작 ===
    # HUD (FSM diagram 포함) + raw color 두 파일 (페어 timestamp). --no-raw 면 HUD 만.
    video_filename, raw_filename = setup_video_writer_filenames()
    video_writer = None
    raw_video_writer = None
    recording = True              # ← 기본값: True (자동 녹화)
    print(f"📹 자동 녹화 대기 (HUD): {video_filename}")
    if record_raw:
        print(f"📹 자동 녹화 대기 (raw 학습용): {raw_filename}")
    else:
        print(f"📹 raw 녹화 비활성 (--no-raw)")
    print("📹 'r' 녹화 토글, 'ESC' 종료")

    # EMA 상태
    offset_smooth = None
    yaw_smooth    = None
    width_smooth  = None

    # === rel_yaw 추정기 ===
    rel_yaw_est = RelYawEstimator(alpha=0.98)
    rel_yaw = 0.0
    last_accel_md = None  # gyro와 짝지어 사용할 최근 accel 저장

    # === FPS / DOPE timing 추적 ===
    fps_window = deque(maxlen=30)   # 최근 30 frame
    dope_fps_window = deque(maxlen=30)
    last_frame_ts = time.perf_counter()
    last_dope_ms = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # --- IMU 프레임 수집/적분 ---
            accel_md = None
            gyro_md = None

            for f in frames:
                if hasattr(f, "is_motion_frame") and f.is_motion_frame():
                    md = f.as_motion_frame().get_motion_data()
                    st = f.get_profile().stream_type()

                    if st == rs.stream.accel:
                        accel_md = md
                        last_accel_md = md

                    elif st == rs.stream.gyro:
                        gyro_md = md
                        # === rel_yaw 업데이트: "gyro.Y + 최근 accel" 사용 ===
                        use_accel = accel_md if accel_md is not None else last_accel_md
                        if use_accel is not None:
                            rel_yaw = rel_yaw_est.update_from_frames(
                                accel=use_accel, gyro=gyro_md, ts_ms=f.get_timestamp()
                            )
                        # accel 데이터가 전혀 없다면, 이전 rel_yaw 유지

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            # 프레임 누락 시 안전 처리 — 640×480 placeholder
            if not color_frame or not depth_frame:
                vis = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_panel(vis, [("프레임 없음", COLOR_ALERT)])
                diag = draw_fsm_diagram_panel(fsm, panel_size=(vis.shape[0], 720))
                show = cv2.hconcat([vis, diag])

                # ★ 자동 녹화: 첫 show가 생성되면 VideoWriter 자동 초기화
                if recording and video_writer is None:
                    h, w = show.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (w, h))
                    if video_writer.isOpened():
                        print(f"🔴 녹화 시작: {video_filename}")
                    else:
                        print("❌ VideoWriter 초기화 실패")
                        recording = False
                        video_writer = None

                if recording:
                    cv2.putText(show, "REC", (show.shape[1]-80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    if video_writer is not None:
                        video_writer.write(show)

                cv2.imshow("Forklift HUD + FSM", show)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('r'):
                    recording = not recording
                    print("🔴 녹화 시작" if recording else "⏹️ 녹화 중지")
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            # RealSense color intrinsic → DOPE PnP 용 3x3 K
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            camera_matrix = np.array([
                [color_intrin.fx, 0.0,             color_intrin.ppx],
                [0.0,             color_intrin.fy, color_intrin.ppy],
                [0.0,             0.0,             1.0],
            ], dtype=np.float64)
            vis = color_img.copy()
            H, W = vis.shape[:2]

            # 1) YOLO segmentation (선택) — USE_PERCEPTION_YOLO=False 면 skip
            det_ok = False
            mask_bin = None
            bbox_now = None
            ok_plane = False
            yaw_deg = None
            ex = ey = ez = None
            width_now = None
            dist_euclid = None
            dist_z = None
            pts_in = None

            if perception is not None:
                det_ok, mask_bin, bbox_now = perception.infer_front(color_img)
                # 2) 3D 포인트/기하 — YOLO mask 영역의 depth points 로 plane fit
                if det_ok and (mask_bin is not None) and (bbox_now is not None):
                    ok_pts, pts_in = robust_points_from_mask_or_roi(
                        depth_frame=depth_frame, depth_intrin=depth_intrin,
                        mask_or_roi=mask_bin, stride=SAMPLE_STRIDE,
                        z_inlier_thresh=Z_INLIER_THRESH, min_points=MIN_POINTS
                    )
                    if ok_pts:
                        ex = float(np.median(pts_in[:, 0]))
                        ey = float(np.mean(pts_in[:, 1]))
                        ez = float(np.mean(pts_in[:, 2]))
                        cur_off = np.array([ex, ey, ez], dtype=np.float32)

                        if offset_smooth is None:
                            offset_smooth = cur_off.copy()
                        else:
                            offset_smooth = (1 - EMA_ALPHA_OFFSET) * offset_smooth + EMA_ALPHA_OFFSET * cur_off

                        ok_plane, yaw_deg, a, b = fit_plane_yaw_from_points(pts_in)
                        if ok_plane:
                            if yaw_smooth is None:
                                yaw_smooth = yaw_deg
                            else:
                                yaw_smooth = (1 - EMA_ALPHA_YAW) * yaw_smooth + EMA_ALPHA_YAW * yaw_deg

                        width_now = float(np.max(pts_in[:, 0]) - np.min(pts_in[:, 0]))
                        if width_smooth is None:
                            width_smooth = width_now
                        else:
                            width_smooth = (1 - EMA_ALPHA_WIDTH) * width_smooth + EMA_ALPHA_WIDTH * width_now

                        c3d_mean = np.mean(pts_in, axis=0).astype(np.float32)
                        dist_euclid = float(np.linalg.norm(c3d_mean))
                        dist_z = float(c3d_mean[2])

            # 2.5) 6D pose (DOPE) — challenge run_live.py 와 동일 path
            #      (ObjectDetector + CuboidPNPSolver + camera-facing + gate + confirm).
            #      det_ok 와 무관하게 시도해 YOLO segmentation 이 놓친 케이스도 추론.
            #      gate 실패해도 dict 반환 (시각화용) — gate_passed=False 면 회색 표시.
            psi_pallet_deg: float = None
            d_lateral_m: float = None
            d_forward_m: float = None
            dope_result = None
            if dope_pose is not None:
                _dope_t0 = time.perf_counter()
                try:
                    dope_result = dope_pose.infer_pose(
                        color_img, camera_matrix, depth_frame=depth_frame,
                    )
                except Exception as e:
                    print(f"[DopePose] infer_pose error: {e}")
                    dope_result = None
                last_dope_ms = (time.perf_counter() - _dope_t0) * 1000.0
                dope_fps_window.append(last_dope_ms)
                # confirmed AND gate_passed 인 경우에만 FSM 으로 ψ/d_lat/d_fwd 전달
                # (한 번 보고 눈 감기 — gate 실패는 시각화만, FSM 으로 안 보냄)
                if (dope_result is not None
                        and dope_result.get("confirmed")
                        and dope_result.get("gate_passed")
                        and dope_result.get("R") is not None):
                    try:
                        # --- 단일 주입점: depth scale 보정 + cam→fork extrinsic ---
                        # DOPE·YOLO 두 backend 공통 (동일 dict 계약). monocular PnP 의
                        # t scale 모호성을 centroid 픽셀 RealSense depth 로 고정한다.
                        R6 = dope_result["R"]
                        t6 = dope_result["t_m"]
                        raw_o = dope_result.get("raw_points_orig") or []
                        proj_o = dope_result.get("proj_points_orig") or []
                        cuv = None
                        if len(raw_o) > 8 and raw_o[8] is not None:
                            cuv = raw_o[8]
                        elif len(proj_o) > 8 and proj_o[8] is not None:
                            cuv = proj_o[8]
                        d_cen = None
                        if cuv is not None:
                            d_cen = _sample_depth(depth_frame, cuv[0], cuv[1])
                        t6, _depth_ok = depth_scale_correct(
                            t6, cuv, d_cen,
                            z_min_m=DEPTH_CORRECT_Z_MIN_M,
                            z_max_m=DEPTH_CORRECT_Z_MAX_M,
                        )
                        R6, t6 = apply_cam_to_fork(
                            R6, t6, CAM_TO_FORK_T, CAM_TO_FORK_RPY_DEG,
                        )
                        psi_pallet_deg, d_lateral_m, d_forward_m = pose6d_to_align_vars(
                            R6, t6,
                        )
                    except Exception as e:
                        print(f"[Pose6D] pose6d_to_align_vars error: {e}")
                        psi_pallet_deg, d_lateral_m, d_forward_m = None, None, None

            # 3) 시각화 — YOLO bbox/contour 는 USE_PERCEPTION_YOLO 일 때만
            if perception is not None:
                if bbox_now is not None:
                    x1, y1, x2, y2 = bbox_now
                    cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BOX, 2)
                if mask_bin is not None:
                    mask_vis = (mask_bin * 255).astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, (120, 60, 120), 1)
                    if len(cnts) > 0:
                        largest = max(cnts, key=cv2.contourArea)
                        xs = largest[:, 0, 0]; ys = largest[:, 0, 1]
                        lx = int(xs.max()); ly = int(ys.min())
                        cv2.putText(vis, "YOLO_SEG", (lx + 4, ly + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 60, 120), 1)
            # 화면 중앙 십자 — 카메라 광축 표시
            cv2.drawMarker(vis, (W // 2, H // 2), COLOR_CENTER, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # FPS 측정 (이전 frame 끝 → 현재 frame 시작)
            _now = time.perf_counter()
            _dt = _now - last_frame_ts
            last_frame_ts = _now
            if _dt > 1e-4:
                fps_window.append(1.0 / _dt)
            fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0
            dope_ms_avg = sum(dope_fps_window) / len(dope_fps_window) if dope_fps_window else 0.0
            infer_fps = 1000.0 / dope_ms_avg if dope_ms_avg > 1e-3 else 0.0

            # 4) HUD 텍스트
            lines = []
            # 상단: FPS / DOPE timing — 사용자가 본 옛 화면 (image #24) 스타일
            lines.append((f"FPS: {fps_avg:.1f}", COLOR_META))
            lines.append((f"Infer FPS: {infer_fps:.1f}  | DOPE ms: {last_dope_ms:.0f} (avg {dope_ms_avg:.0f})", COLOR_META))

            # YOLO + RGB-D RANSAC 관련 라인 — USE_PERCEPTION_YOLO=True 일 때만
            if perception is not None:
                lines.append(("detected" if det_ok else "no detection", COLOR_STATUS_OK if det_ok else COLOR_ALERT))
                lines.append((f"pts: {len(pts_in) if pts_in is not None else 0}", COLOR_META))

                if ok_plane and (yaw_smooth is not None) and (yaw_deg is not None):
                    lines.append((f"yaw(now): {fmt_deg(yaw_deg)} deg", COLOR_YAW))
                    lines.append((f"yaw_smooth: {fmt_deg(yaw_smooth)} deg", COLOR_YAW))
                else:
                    lines.append(("yaw: N/A", COLOR_ALERT))

                if ex is not None and (offset_smooth is not None):
                    lines.append((f"offset_x(now): {fmt_m(ex)} m", COLOR_OFFSET))
                    lines.append((f"offset_smooth: ({fmt_m(offset_smooth[0])}, {fmt_m(offset_smooth[1])}, {fmt_m(offset_smooth[2])})", COLOR_OFFSET))
                else:
                    lines.append(("offset_smooth: N/A", COLOR_OFFSET))

                if (width_now is not None) and (width_smooth is not None):
                    lines.append((f"pallet width(now): {width_now:.3f} m", COLOR_WIDTH))
                    lines.append((f"pallet width_smooth: {width_smooth:.3f} m", COLOR_WIDTH))
                else:
                    lines.append(("pallet width_smooth: N/A", COLOR_WIDTH))

                if (dist_euclid is not None) and (dist_z is not None):
                    lines.append((f"distance: euclid {dist_euclid:.3f} m | z {dist_z:.3f} m", COLOR_META))
                else:
                    lines.append(("distance: N/A", COLOR_META))

            # === HUD: rel_yaw(gyro-Y) — IMU 회전 누적 (FSM 회전 종료 판정용, 두 모드 공통) ===
            lines.append((f"rel_yaw(gyro-Y): {fmt_deg(rel_yaw)} deg", COLOR_META))
            lines.append((f"|rel_yaw|(gyro-Y): {abs(rel_yaw):6.2f} deg", COLOR_META))

            # === HUD: 6D pose (DOPE) — 결과 요약 텍스트 ===
            # 상태:
            #   CONFIRMED : gate 통과 + N 연속 — FSM 으로 ψ/d_lat/d_fwd 전달
            #   PENDING   : gate 통과 but consecutive_ok < N
            #   GATE FAIL : DOPE 추론은 됐지만 gate 실패 — 시각화만, FSM 미전달
            #   NO DETECT : 추론 결과 자체 없음
            if psi_pallet_deg is not None:
                lines.append((f"[6D] psi_pallet: {fmt_deg(psi_pallet_deg)} deg", COLOR_YAW))
                lines.append((f"[6D] d_lateral: {fmt_m(d_lateral_m)} m", COLOR_OFFSET))
                lines.append((f"[6D] d_forward: {fmt_m(d_forward_m)} m", COLOR_META))
                lines.append((f"[6D] CONFIRMED", COLOR_STATUS_OK))
            elif dope_pose is not None:
                if dope_result is not None and dope_result.get("gate_passed") and not dope_result.get("confirmed"):
                    co = dope_result.get("consecutive_ok", 0)
                    lines.append((f"[6D] PENDING {co}/{DOPE_CONFIRM_FRAMES}", COLOR_META))
                elif dope_result is not None and not dope_result.get("gate_passed"):
                    fr = dope_result.get("info", {}).get("fail_reason", dope_pose.last_reason)
                    lines.append((f"[6D] GATE FAIL: {fr}", COLOR_ALERT))
                else:
                    lines.append((f"[6D] NO DET ({dope_pose.last_reason})", COLOR_ALERT))

            # === HUD: gate 진단 — 검출이 있을 때 모든 gate 값 한 줄로 표시 (디버깅) ===
            # 어느 gate 가 막혔는지 즉시 보임. ✓ / ✗ 마크로 통과/실패.
            from calib.config import (
                DOPE_GATE_MIN_KP as _G_MIN_KP,
                DOPE_GATE_MAX_REPROJ_PX as _G_REPROJ,
                DOPE_GATE_Z_MIN_M as _G_ZMIN,
                DOPE_GATE_Z_MAX_M as _G_ZMAX,
                DOPE_GATE_DEPTH_REL as _G_DREL,
            )
            if dope_result is not None:
                info = dope_result.get("info", {})
                n_kp = info.get("n_kp", 0)
                reproj = info.get("reproj")
                z_m = info.get("z_m")
                depth_rel = info.get("depth_rel")
                def _mk(label, val, ok):
                    return f"{label}={val}{'+' if ok else '!'}"
                parts = []
                parts.append(_mk("kp", n_kp, n_kp >= _G_MIN_KP))
                if reproj is not None:
                    parts.append(_mk("rep", f"{reproj:.1f}", reproj <= _G_REPROJ))
                if z_m is not None:
                    parts.append(_mk("z", f"{z_m:.2f}", _G_ZMIN <= z_m <= _G_ZMAX))
                if depth_rel is not None:
                    parts.append(_mk("dRel", f"{depth_rel:.2f}", depth_rel <= _G_DREL))
                lines.append(("[gate] " + " ".join(parts), COLOR_META))

            # === DOPE HUD 시각화 — 앞면(front face) polygon + 9 keypoint + centroid + yaw arrow ===
            # 사용자 요구 (image #24 스타일): 미니멀. back face / vertical edges 제거.
            # gate 실패해도 검출되면 점/arrow 회색으로 표시 (추론 동작 확인 용).
            if dope_result is not None:
                raw_orig = dope_result["raw_points_orig"]
                confirmed = dope_result.get("confirmed", False)
                gate_passed = dope_result.get("gate_passed", False)

                # 검출되면 항상 동일 색 (image #26 스타일) — gate fail 도 추론 결과는 정상 표시.
                # FSM 진행 여부는 별개로 confirmed/gate_passed 가 결정.
                clr_front = (0, 255, 0)      # 선명한 초록
                clr_kp    = (0, 255, 0)
                clr_ctr   = (0, 0, 255)      # 빨강 centroid
                clr_arrow = (0, 255, 255)    # 노랑 yaw arrow

                # front face polygon (0-1-2-3) — 가까운 면, 굵게 강조.
                # 4 점 다 있으면 closed polygon, 일부 missing 이면 연결 가능한 edge 만.
                front_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for a, b in front_edges:
                    if raw_orig[a] is not None and raw_orig[b] is not None:
                        pa = (int(raw_orig[a][0]), int(raw_orig[a][1]))
                        pb = (int(raw_orig[b][0]), int(raw_orig[b][1]))
                        cv2.line(vis, pa, pb, clr_front, 4, cv2.LINE_AA)

                # 9 keypoint + corner index (back face 4~7 도 점만 표시, 선 없음)
                for i, pt in enumerate(raw_orig):
                    if pt is None:
                        continue
                    px, py = int(pt[0]), int(pt[1])
                    if i == 8:
                        # centroid — 강조 빨간 원
                        cv2.circle(vis, (px, py), 9, clr_ctr, -1)
                    else:
                        cv2.circle(vis, (px, py), 6, clr_kp, -1)
                        cv2.putText(vis, str(i), (px + 5, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr_kp, 2)

                # yaw arrow — centroid (3D origin) + R[:,2] (entry face normal) 50cm
                # gate fail 이라도 PnP 가 성공해 R/t 있으면 그림 (회색)
                R6 = dope_result.get("R")
                t_m6 = dope_result.get("t_m")
                if R6 is not None and t_m6 is not None:
                    try:
                        rvec, _ = cv2.Rodrigues(R6)
                        tvec_cm = (np.asarray(t_m6, dtype=np.float64) * 100.0).reshape(3, 1)
                        f3d = np.array([[0, 0, 0], [0, 0, 50]], dtype=np.float64)
                        pts_proj, _ = cv2.projectPoints(
                            f3d, rvec, tvec_cm,
                            camera_matrix, np.zeros((4, 1), dtype=np.float64),
                        )
                        p1 = tuple(pts_proj[0].ravel().astype(int))
                        p2 = tuple(pts_proj[1].ravel().astype(int))
                        Hv, Wv = vis.shape[:2]
                        if (0 <= p1[0] < Wv and 0 <= p1[1] < Hv and
                                0 <= p2[0] < Wv and 0 <= p2[1] < Hv):
                            cv2.arrowedLine(vis, p1, p2, clr_arrow, 4, cv2.LINE_AA, tipLength=0.20)
                            cv2.putText(vis, "+Z(front)", (p2[0] + 8, p2[1] - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr_arrow, 2)
                    except Exception as _e:
                        pass

                # 상단 배너 — 상태 표시 (CONFIRMED 면 표시 안 함, HUD lines 에 detected (CONFIRMED) 로)
                if not gate_passed:
                    info = dope_result.get("info", {})
                    fr = info.get("fail_reason", "?")
                    cv2.putText(vis, f"DOPE: GATE FAIL ({fr})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 220), 2)
                elif not confirmed:
                    cv2.putText(vis, f"DOPE: PENDING {dope_result['consecutive_ok']}/{DOPE_CONFIRM_FRAMES}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 200), 2)
            elif dope_pose is not None:
                cv2.putText(vis, f"DOPE: NO DETECTION ({dope_pose.last_reason})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

            # 5) FSM 구동 — rel_yaw + 6D pose 입력 전달.
            # YOLO 끈 모드 (perception=None) 에선 fsm.step 의 det_ok / detected_length /
            # dist_z / yaw_smooth / offset_smooth 가 모두 None → SEARCH 단계 못 빠짐.
            # DOPE confirmed 일 때 이 값들을 DOPE 결과로 채워서 SEARCH→DETECTED→ALIGN 진행 가능하게.
            fsm_det_ok = det_ok
            fsm_width = width_smooth
            fsm_dist_z = dist_z
            fsm_yaw_smooth = yaw_smooth
            fsm_offset_smooth = offset_smooth
            if perception is None and psi_pallet_deg is not None:
                fsm_det_ok = True
                if fsm_width is None:
                    fsm_width = PALLET_WIDTH_M
                if fsm_dist_z is None:
                    fsm_dist_z = d_forward_m
                if fsm_yaw_smooth is None:
                    fsm_yaw_smooth = psi_pallet_deg
                if fsm_offset_smooth is None:
                    fsm_offset_smooth = np.array(
                        [d_lateral_m, 0.0, d_forward_m], dtype=np.float32
                    )
            guide_lines = fsm.step(
                det_ok=fsm_det_ok,
                detected_length=fsm_width,
                dist_z=fsm_dist_z,
                yaw_smooth=fsm_yaw_smooth,
                offset_smooth=fsm_offset_smooth,
                rel_yaw=rel_yaw,
                psi_pallet_deg=psi_pallet_deg,
                d_lateral_m=d_lateral_m,
                d_forward_m=d_forward_m,
            )
            lines.extend(guide_lines)

            # 6) HUD 렌더
            draw_panel(
                vis, lines, origin=(18, 22), pad=(7, 6),
                line_h=18, font_scale=0.45, thickness=1, cmd_status=fsm.cmd_status
            )

            # 7) FSM 다이어그램 패널 합성
            diag = draw_fsm_diagram_panel(fsm, panel_size=(vis.shape[0], 900))
            if diag.shape[0] != vis.shape[0]:
                diag = cv2.resize(diag, (diag.shape[1], vis.shape[0]), interpolation=cv2.INTER_AREA)
            show = cv2.hconcat([vis, diag])

            # ★ 자동 녹화: 첫 show 가 준비되면 VideoWriter 초기화 (HUD + raw)
            if recording and video_writer is None:
                h, w = show.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (w, h))
                if video_writer.isOpened():
                    print(f"🔴 녹화 시작 (HUD): {video_filename}")
                else:
                    print("❌ HUD VideoWriter 초기화 실패")
                    recording = False
                    video_writer = None
            if recording and record_raw and raw_video_writer is None:
                rh, rw = color_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                raw_video_writer = cv2.VideoWriter(raw_filename, fourcc, 15.0, (rw, rh))
                if raw_video_writer.isOpened():
                    print(f"🔴 녹화 시작 (raw): {raw_filename}")
                else:
                    print("❌ raw VideoWriter 초기화 실패 (HUD 녹화는 계속)")
                    raw_video_writer = None

            # 8) 녹화 표시/쓰기 — HUD 에는 REC 라벨 추가, raw 는 깨끗하게 저장
            if recording:
                if record_raw and raw_video_writer is not None:
                    raw_video_writer.write(color_img)   # raw 는 REC 라벨 없이 (학습용)
                cv2.putText(show, "REC", (show.shape[1]-80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if video_writer is not None:
                    video_writer.write(show)

            # 9) 표시 & 키 입력
            cv2.imshow("Forklift HUD + FSM", show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                recording = not recording
                print("🔴 녹화 시작" if recording else "⏹️ 녹화 중지")

    finally:
        print("프로그램 종료 처리 중...")
        try:
            if video_writer is not None:
                video_writer.release()
                print(f"✅ HUD 저장 완료: {video_filename}")
        except Exception:
            pass
        try:
            if raw_video_writer is not None:
                raw_video_writer.release()
                print(f"✅ raw 저장 완료: {raw_filename}")
        except Exception:
            pass
        try:
            can_close()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ 리소스 정리 완료")


if __name__ == "__main__":
    main()
