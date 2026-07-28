# -*- coding: utf-8 -*-
"""
YOLOv8 + RealSense D435i (IMU 기반 회전) 지게차 자율 제어 (오케스트레이션 포함)
실행 순서:
  1) load_base.py  선 실행 (서브프로세스)
  2) 본 스크립트의 비전/미션 실행
  3) loading.py    후 실행 (서브프로세스)

변경 사항
- YOLO/RealSense 초기화를 모듈 상단에서 제거하고, load_base.py 실행 이후에 진행
- 서브프로세스 실행 유틸 run_script() 추가
- 종료 처리 가드 강화
- 기존 로직/파라미터는 유지
"""

import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import time, math
import torch
import asyncio
import sys, os
from typing import Optional
from collections import deque

# =========================
# 외부 스크립트 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_BASE_PATH = "/home/hy-jang/25y_automatic_lifter/autolifter/load_base.py"   # 1) 선행 실행
LOADING_PATH   = "/home/hy-jang/25y_automatic_lifter/autolifter/loading.py"     # 3) 후행 실행
PYTHON_EXE = sys.executable

async def run_script(path: str, *args: str, timeout: Optional[float] = None) -> None:
    """외부 파이썬 스크립트를 서브프로세스로 실행하고 종료코드를 확인."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script not found: {path}")

    proc = await asyncio.create_subprocess_exec(
        PYTHON_EXE, path, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Timeout while running: {path}")

    if stdout:
        sys.stdout.write(stdout.decode(errors="ignore"))
    if stderr:
        sys.stderr.write(stderr.decode(errors="ignore"))

    if proc.returncode != 0:
        raise RuntimeError(f"Script failed ({proc.returncode}): {path}")

# =========================
# 사용자 제공 컨트롤러
# =========================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from forklift_ctrl import AutonomousForkliftController
except ImportError:
    print("[WARNING] 'forklift_ctrl.py' module not found. Using a Dummy Controller.")
    class AutonomousForkliftController:
        def __init__(self, can_channel, can_bitrate, use_extended_ids):
            print(f"[CTRL] Dummy Controller initialized.")
        async def start(self):
            print("[CTRL] Dummy Controller started.")
        async def stop(self):
            print("[CTRL] Dummy Controller stopped.")
        async def drive_forward(self, power, duration):
            print(f"[CTRL] Action: Drive Forward {power}% for {duration:.2f}s")
            await asyncio.sleep(duration)
        async def rotate_cw(self, power, duration):
            print(f"[CTRL] Action: Rotate CW {power}% for {duration:.2f}s")
            await asyncio.sleep(duration)
        async def rotate_ccw(self, power, duration):
            print(f"[CTRL] Action: Rotate CCW {power}% for {duration:.2f}s")
            await asyncio.sleep(duration)
        async def stop_all(self):
            print("[CTRL] Action: Stop All")
            await asyncio.sleep(0.01)

# ===== 설정 파라미터 =====
INSET_FRAC = 0.10
INSET_MIN_PX = 4
WHEEL_CONF_THRESH = 0.55
CENTER_Y_OFFSET_PX = 15.0  # ✅ 중심점을 위로 5px만 이동 (주석 정합성 주의)
CENTER_FRACTION = 0.5

LOCK_DELAY_S = 7.0
LOCK_DURATION_S = 3.0

# ===== 자율 정렬 제어 파라미터 =====
ALIGN_ROT_PCT = 50
ALIGN_ROT_THRESH_PX = 10.0
ALIGNMENT_THRESHOLD_PX_FINAL = 10

# ===== 전진 속도 추정 =====
DRIVE_MPS_50 = 0.250

def drive_seconds(distance_m: float, mps: float = DRIVE_MPS_50) -> float:
    if mps <= 0:
        raise ValueError("전진 속도(m/s)가 0 이하입니다.")
    return max(0.1, abs(float(distance_m)) / float(mps))

def rotation_seconds(angle_deg: float) -> float:
    t = (95 * float(angle_deg) + 1350) / 600
    return max(0.1, t)

# =========================
# 전역(지연) 초기화 리소스
# =========================
pipeline = None
align_to_color = None
intr = None
depth_scale = None
model = None

# ===== IMU 트래커 =====
class IMUTracker:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.first = True
        self.last_ts_gyro = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.init_yaw = None
        self.yaw_history = deque(maxlen=5)
    
    def reset_yaw(self):
        self.init_yaw = self.yaw
        self.yaw_history.clear()
        print(f"[IMU] Yaw reset at {self.yaw:.2f}°")
    
    def get_relative_yaw(self):
        if self.init_yaw is None:
            return 0.0
        rel = self.yaw - self.init_yaw
        rel = (rel + 180) % 360 - 180
        return rel
    
    def get_smoothed_yaw(self):
        if len(self.yaw_history) == 0:
            return self.get_relative_yaw()
        return np.mean(self.yaw_history)
    
    def update(self, frames):
        accel_frame = None
        gyro_frame = None
        
        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame()
                if motion.get_profile().stream_type() == rs.stream.accel:
                    accel_frame = motion
                elif motion.get_profile().stream_type() == rs.stream.gyro:
                    gyro_frame = motion
        
        if accel_frame is None or gyro_frame is None:
            return False
        
        accel = accel_frame.get_motion_data()
        gyro = gyro_frame.get_motion_data()
        ts = frames.get_timestamp()
        
        if self.first:
            self.first = False
            self.last_ts_gyro = ts
            self.roll = math.degrees(math.atan2(accel.y, accel.z))
            self.pitch = math.degrees(math.atan2(accel.x, math.sqrt(accel.y**2 + accel.z**2)))
            self.yaw = 0.0
            self.init_yaw = 0.0
            return True
        
        dt_gyro = (ts - self.last_ts_gyro) / 1000.0
        self.last_ts_gyro = ts
        
        dangleX = math.degrees(gyro.x * dt_gyro)
        dangleY = math.degrees(gyro.y * dt_gyro)
        dangleZ = math.degrees(gyro.z * dt_gyro)
        
        accel_roll = math.degrees(math.atan2(accel.y, accel.z))
        accel_pitch = math.degrees(math.atan2(accel.x, math.sqrt(accel.y**2 + accel.z**2)))
        
        self.roll = (self.roll + dangleX) * self.alpha + accel_roll * (1 - self.alpha)
        self.pitch = (self.pitch + dangleZ) * self.alpha + accel_pitch * (1 - self.alpha)
        self.yaw += dangleY
        
        self.yaw_history.append(self.get_relative_yaw())
        return True

imu_tracker = IMUTracker()

# ===== 깊이 및 3D 함수 =====
def depth_at_pixel(depth_frame, u, v, window=5):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    half = window // 2
    us, ue = max(0, int(u)-half), min(w, int(u)+half+1)
    vs, ve = max(0, int(v)-half), min(h, int(v)+half+1)
    depth = np.asanyarray(depth_frame.get_data())[vs:ve, us:ue]
    valid = depth[depth > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale

def deproject(u, v, z):
    return np.array(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z)),
                     dtype=np.float64)

def fit_plane(points):
    P = np.asarray(points, dtype=np.float64)
    centroid = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - centroid)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -np.dot(n, centroid)
    return n, d

def text_put(img, lines, org=(8, 20), line_h=22, color=(255,255,255)):
    for i, t in enumerate(lines):
        y = org[1] + i*line_h
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

# ===== 자율 구동 정렬 함수 =====
async def adjust_position(ctrl, offset_x, lines):
    """횡방향 오프셋(offset_x)에 따라 지게차를 회전하여 중앙의 세로선에 맞춥니다."""
    if abs(offset_x) > ALIGN_ROT_THRESH_PX:
        rotation_duration = 1.5
        if offset_x > 0:
            lines.append(f"ACTION: ROTATE RIGHT (CW) {rotation_duration:.2f}s")
            await ctrl.rotate_cw(ALIGN_ROT_PCT, rotation_duration)
        else:
            lines.append(f"ACTION: ROTATE LEFT (CCW) {rotation_duration:.2f}s")
            await ctrl.rotate_ccw(ALIGN_ROT_PCT, rotation_duration)
        await asyncio.sleep(0.3)
        return True
    await ctrl.stop_all()
    return False

# ===== 전역 상태 변수 =====
n_prev = None

# ===== 비전 루프 =====
async def run_vision_detection_autonomous(ctrl):
    """YOLO 바퀴 검출 및 각도/거리 측정 (자율 횡방향 정렬 포함)"""
    global n_prev
    
    window_on_time = None
    alignment_mode = True
    alignment_stable_count = 0
    ALIGNMENT_STABLE_FRAMES = 30
    
    alpha_locked = False
    alpha_deg_locked = None
    center_depth_locked = None
    wheel_distance_locked = None
    angle_samples_lock = []
    center_depth_samples_lock = []
    wheel_distance_samples_lock = []

    print("\n" + "="*70)
    print("[VISION] Starting autonomous alignment and measurement")
    print(f"[PHASE 1] Autonomously aligning to vertical center (±{ALIGNMENT_THRESHOLD_PX_FINAL}px)...")
    print("[PHASE 2] Measuring angle and distance (after alignment)")
    print("="*70 + "\n")
    
    try:
        while True:
            await asyncio.sleep(0)  # 이벤트 루프 양보
            
            try:
                frames = pipeline.wait_for_frames()
            except Exception as e:
                print(f"[ERROR] Could not get frames: {e}. Aborting vision.")
                return None
            
            await asyncio.sleep(0)
                
            imu_tracker.update(frames)
            frames = align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                await asyncio.sleep(0.001)
                continue

            frame = np.asanyarray(color_frame.get_data())
            h_img, w_img = frame.shape[:2]
            img_center_x = w_img / 2.0
            img_center_y = h_img / 2.0

            # YOLO 추론
            results = model.predict(source=frame, device=0 if torch.cuda.is_available() else 'cpu',
                                    half=torch.cuda.is_available(), verbose=False)
            await asyncio.sleep(0)
            
            r = results[0]
            boxes = r.boxes
            names = r.names
            annotated_frame = frame.copy()
            lines = []
            
            # 정렬 목표선
            cv2.line(annotated_frame, (int(img_center_x - ALIGNMENT_THRESHOLD_PX_FINAL), 0), 
                     (int(img_center_x - ALIGNMENT_THRESHOLD_PX_FINAL), h_img), (0, 255, 0), 1)
            cv2.line(annotated_frame, (int(img_center_x + ALIGNMENT_THRESHOLD_PX_FINAL), 0), 
                     (int(img_center_x + ALIGNMENT_THRESHOLD_PX_FINAL), h_img), (0, 255, 0), 1)
            
            if alignment_mode:
                lines.append("[PHASE 1] AUTONOMOUS ALIGNMENT")
            else:
                lines.append("[PHASE 2] MEASUREMENT MODE")
            lines.append(f"IMU Yaw: {imu_tracker.get_smoothed_yaw():.1f}°")

            wheel_ids = [k for k, v in names.items() if v.lower() == 'wheel']
            if len(wheel_ids) == 0:
                lines.append("'wheel' 클래스가 모델에 없습니다.")
                text_put(annotated_frame, lines, (8, 24))
                cv2.imshow("YOLOv8 Wheel Detection + IMU", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    await ctrl.stop_all()
                    return None
                await asyncio.sleep(0.001)
                continue
            
            star_cls = set(wheel_ids)
            wheel_dets = []
            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    if (cls_id in star_cls) and (conf >= WHEEL_CONF_THRESH):
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        wheel_dets.append({"xyxy": (x1, y1, x2, y2), "center": (cx, cy), "conf": conf})

            if len(wheel_dets) < 2:
                lines.append(f"Need 2+ wheels (conf>={WHEEL_CONF_THRESH:.2f})")
                text_put(annotated_frame, lines, (8, 24))
                cv2.imshow("YOLOv8 Wheel Detection + IMU", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    await ctrl.stop_all()
                    return None
                await asyncio.sleep(0.001)
                continue

            wheel_dets.sort(key=lambda d: d["center"][0])
            left, right = wheel_dets[0], wheel_dets[-1]

            for det, color in [(left,(0,255,0)), (right,(0,128,255))]:
                x1,y1,_,_ = map(int, det["xyxy"])
                x1,y1,x2,y2 = map(int, det["xyxy"])
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)
                label = f"wheel {det['conf']:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 센터 포인트 계산
            left_c = np.array(left["center"], dtype=float)
            right_c = np.array(right["center"], dtype=float)
            if left["conf"] <= right["conf"]:
                base, other = left_c, right_c
            else:
                base, other = right_c, left_c
            C_pix = base + CENTER_FRACTION * (other - base)
            C_pix = (float(np.clip(C_pix[0], 0.0, w_img - 1.0)), 
                     float(np.clip(C_pix[1] - CENTER_Y_OFFSET_PX, 0.0, h_img - 1.0)))
            cv2.drawMarker(annotated_frame, (int(C_pix[0]), int(C_pix[1])), (255,255,255), 
                          markerType=cv2.MARKER_STAR, markerSize=12, thickness=2)
            
            offset_x = C_pix[0] - img_center_x
            offset_y = C_pix[1] - img_center_y
            
            # ===== PHASE 1: 자율 정렬 모드 =====
            if alignment_mode:
                lines.append(f"Center offset X: {offset_x:+.1f}px")
                lines.append(f"Center offset Y: {offset_y:+.1f}px")
                is_aligned = abs(offset_x) <= ALIGNMENT_THRESHOLD_PX_FINAL
                
                if is_aligned:
                    alignment_stable_count += 1
                    if alignment_stable_count == 1:
                        await ctrl.stop_all()
                    lines.append(f"ALIGNED! Stable: {alignment_stable_count}/{ALIGNMENT_STABLE_FRAMES}")
                    
                    if alignment_stable_count >= ALIGNMENT_STABLE_FRAMES:
                        alignment_mode = False
                        window_on_time = None
                        print("\n" + "="*70)
                        print("[ALIGNMENT] ✓ Complete! Switching to measurement mode...")
                        print("="*70 + "\n")
                        lines.append("✓ Alignment complete!")
                    else:
                        lines.append("ACTION: HOLDING POSITION")
                else:
                    alignment_stable_count = 0
                    await adjust_position(ctrl, offset_x, lines)

                text_put(annotated_frame, lines, (8, 24))
                cv2.imshow("YOLOv8 Wheel Detection + IMU", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    await ctrl.stop_all()
                    return None
                continue
            
            # ===== PHASE 2: 측정 모드 =====
            lines.append(f"✓ Aligned (X-offset: {offset_x:+.1f}px)")
            
            # 중심점 기준 4포인트 평면 추정
            PLANE_OFFSET_VERTICAL = 5
            PLANE_OFFSET_HORIZONTAL = 25
            
            pts_pix = [
                (C_pix[0], C_pix[1]),                                # 중심
                (C_pix[0] - PLANE_OFFSET_HORIZONTAL, C_pix[1]),     # 좌
                (C_pix[0] + PLANE_OFFSET_HORIZONTAL, C_pix[1]),     # 우
                (C_pix[0], C_pix[1] - PLANE_OFFSET_VERTICAL)        # 상
            ]
            labels = ["Center", "Left", "Right", "Top"]

            pts3d, depths_m, valid_mask = [], [], []
            for (u,v) in pts_pix:
                u_clip = float(np.clip(u, 0, w_img - 1))
                v_clip = float(np.clip(v, 0, h_img - 1))
                z = depth_at_pixel(depth_frame, u_clip, v_clip, window=5)
                if z is None:
                    depths_m.append(None)
                    pts3d.append(None)
                    valid_mask.append(False)
                else:
                    P = deproject(u_clip, v_clip, z)
                    depths_m.append(z)
                    pts3d.append(P)
                    valid_mask.append(True)

            draw_colors = [(255,255,255), (0,255,0), (0,128,255), (255,0,255), (255,128,0)]
            for i, ((u,v), lab, ok) in enumerate(zip(pts_pix, labels, valid_mask)):
                u_clip = int(np.clip(u, 0, w_img - 1))
                v_clip = int(np.clip(v, 0, h_img - 1))
                c = draw_colors[i % len(draw_colors)]
                cv2.circle(annotated_frame, (u_clip, v_clip), 5, c, -1)

            valid_pts3d = [p for p in pts3d if p is not None]
            plane_ok = len(valid_pts3d) >= 3

            for lab, z in zip(labels, depths_m):
                lines.append(f"{lab}: {'N/A' if z is None else f'{z:.3f}m'}")

            if plane_ok:
                n, d = fit_plane(valid_pts3d)

                z_c = depth_at_pixel(depth_frame, C_pix[0], C_pix[1], window=7)
                P_c = None
                if z_c is not None:
                    P_c = deproject(C_pix[0], C_pix[1], z_c)
                else:
                    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
                    r_dir = np.array([(C_pix[0]-ppx)/fx, (C_pix[1]-ppy)/fy, 1.0], dtype=np.float64)
                    denom = float(np.dot(n, r_dir))
                    if abs(denom) > 1e-9:
                        s = float(-d / denom)
                        if s > 0:
                            P_c = s * r_dir

                if P_c is not None:
                    v_hat = P_c / (np.linalg.norm(P_c) + 1e-12)
                    
                    # 법선 방향 일관화
                    global n_prev
                    if n_prev is not None and np.dot(n, n_prev) < 0:
                        n = -n; d = -d
                    if np.dot(n, v_hat) < 0:
                        n = -n; d = -d
                    n_prev = n.copy()

                    # 평면과 카메라 Z축 각도
                    camera_forward = np.array([0, 0, 1], dtype=np.float64)
                    if n[2] < 0:
                        n = -n; d = -d
                    dot_nz = float(np.clip(np.dot(n, camera_forward), -1.0, 1.0))
                    alpha_deg = float(np.degrees(np.arccos(dot_nz)))
                    
                    lines.append(f"[DEBUG] Normal n: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
                    lines.append(f"[DEBUG] n·z: {dot_nz:.4f}")
                    lines.append(f"Alpha (Plane→CamZ): {alpha_deg:.1f}°")
                    
                    # 거리 계산 (기존 로직 유지)
                    CAMERA_HEIGHT_OFFSET = 1.30
                    raw_depth = math.sqrt(P_c[0]**2 + P_c[1]**2 + P_c[2]**2)
                    adjusted_depth = math.sqrt(max(0.0, raw_depth**2 - CAMERA_HEIGHT_OFFSET**2))
                    
                    lines.append(f"Raw depth: {raw_depth:.3f}m")
                    lines.append(f"Adjusted d: {adjusted_depth:.3f}m")

                    now = time.time()
                    if window_on_time is None:
                        window_on_time = now

                    elapsed = now - window_on_time
                    if not alpha_locked:
                        if elapsed < LOCK_DELAY_S:
                            lines.append(f"Warm-up: {LOCK_DELAY_S - elapsed:.1f}s")
                        elif elapsed < (LOCK_DELAY_S + LOCK_DURATION_S):
                            angle_samples_lock.append(alpha_deg)
                            if np.isfinite(adjusted_depth):
                                center_depth_samples_lock.append(adjusted_depth)
                                wheel_distance_samples_lock.append(adjusted_depth)
                            remain = (LOCK_DELAY_S + LOCK_DURATION_S) - elapsed
                            mean_a = np.mean(angle_samples_lock) if angle_samples_lock else 0.0
                            mean_d = np.mean(center_depth_samples_lock) if center_depth_samples_lock else 0.0
                            lines.append(f"LOCKING: α={mean_a:.1f}°, d={mean_d:.3f}m ({remain:.1f}s left)")
                        else:
                            if angle_samples_lock:
                                alpha_deg_locked = float(np.mean(angle_samples_lock))
                            if center_depth_samples_lock:
                                center_depth_locked = float(np.mean(center_depth_samples_lock))
                            if wheel_distance_samples_lock:
                                wheel_distance_locked = float(np.mean(wheel_distance_samples_lock))
                            alpha_locked = True
                    else:
                        lines.append(f"LOCKED: α={alpha_deg_locked:.1f}°, d={center_depth_locked:.3f}m")
                        if (alpha_deg_locked is not None) and (center_depth_locked is not None):
                            proj_len = center_depth_locked * math.cos(math.radians(alpha_deg_locked))
                            lines.append(f"Forward dist: {proj_len:.3f}m")
                else:
                    lines.append("Plane center 3D point N/A")
            else:
                lines.append("Plane: not enough valid depth points")

            text_put(annotated_frame, lines, (8, 24))
            cv2.imshow("YOLOv8 Wheel Detection + IMU", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                await ctrl.stop_all()
                return None

            if alpha_locked and (alpha_deg_locked is not None) and (center_depth_locked is not None):
                lcx, lcy = left["center"]
                rcx, rcy = right["center"]
                z_left = depth_at_pixel(depth_frame, lcx, lcy, window=5)
                z_right = depth_at_pixel(depth_frame, rcx, rcy, window=5)
                dir_cw = True
                if (z_left is not None) and (z_right is not None):
                    dir_cw = (z_left < z_right)

                proj_len_m = center_depth_locked * math.cos(math.radians(alpha_deg_locked))
                if wheel_distance_locked is None:
                    wheel_distance_locked = center_depth_locked
                
                print(f"\n{'='*70}")
                print(f"[LOCKED] α={alpha_deg_locked:.2f}°, depth={center_depth_locked:.3f}m")
                print(f"[LOCKED] Forward={proj_len_m:.3f}m, Direction={'CW' if dir_cw else 'CCW'}")
                print(f"{'='*70}\n")
                
                return (alpha_deg_locked, proj_len_m, dir_cw, wheel_distance_locked)
            
            await asyncio.sleep(0.001)

    finally:
        cv2.destroyAllWindows()

# ===== 미션 실행 함수 =====
async def run_four_step_mission(ctrl, alpha_deg: float, proj_len_m: float, dir_step1_cw: bool, wheel_distance_m: float = None):
    ROTATE_PCT = 50
    DRIVE_PCT = 50
    SAFETY_OFFSET = 1.20
    dur_step2 = drive_seconds(proj_len_m, DRIVE_MPS_50)
    final_distance_step4 = None
    vertical_distance = None
    
    if wheel_distance_m is not None:
        vertical_distance = wheel_distance_m * math.sin(math.radians(alpha_deg))
        final_distance_step4 = vertical_distance - SAFETY_OFFSET
        if final_distance_step4 < 0:
            print(f"[ERROR] ✗ Mission impossible - Insufficient distance! Shortage: {abs(final_distance_step4):.3f}m")
            return
    
    print(f"\n{'='*70}")
    print(f"[MISSION] IMU-based 4-Step Mission (Truck Alignment)")
    print(f"Step 1: Rotate {alpha_deg:.1f}° ({'CW' if dir_step1_cw else 'CCW'}) [IMU]")
    print(f"Step 2: Drive forward {proj_len_m:.3f}m")
    print(f"Step 3: Rotate 90° ({'CCW' if dir_step1_cw else 'CW'}) [IMU]")
    
    if final_distance_step4 is not None and final_distance_step4 >= 0.05:
        print(f"Step 4: Final alignment {final_distance_step4:.3f}m")
    else:
        print(f"Step 4: Already aligned (skipping final alignment)")
    print(f"{'='*70}\n")
    
    await ctrl.start()
    try:
        # Step 1: 초기 회전
        if dir_step1_cw:
            await ctrl.rotate_cw(ROTATE_PCT, rotation_seconds(alpha_deg))
        else:
            await ctrl.rotate_ccw(ROTATE_PCT, rotation_seconds(alpha_deg)-2.7)
        await asyncio.sleep(1.0)
        
        # Step 2: 전진
        await ctrl.drive_forward(DRIVE_PCT, dur_step2+5)
        await asyncio.sleep(1.0)
        
        # Step 3: 90도 회전
        if dir_step1_cw:
            await ctrl.rotate_ccw(ROTATE_PCT, rotation_seconds(90.0)-2.7)
        else:
            await ctrl.rotate_cw(ROTATE_PCT, rotation_seconds(90.0)-0.5)
        await asyncio.sleep(1.0)
        
        # Step 4: 최종 정렬 (필요 시)
        if final_distance_step4 is not None and final_distance_step4 >= 0.05:
            dur_step4 = drive_seconds(final_distance_step4, DRIVE_MPS_50)
            await ctrl.drive_forward(DRIVE_PCT, dur_step4)
            print(f"[MISSION] ✓ Step 4 완료: {final_distance_step4:.3f}m 전진")
        else:
            print(f"[MISSION] ✓ Step 4 스킵: 이미 정렬됨")
        
        print(f"[MISSION] ✓ All steps completed successfully!")
        
    finally:
        await ctrl.stop()

# =========================
# 지연 초기화: YOLO/RealSense
# =========================
def init_yolo_and_realsense():
    global pipeline, align_to_color, intr, depth_scale, model

    # YOLO
    model = YOLO("./model/finetuned_wheel.pt")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale_local = depth_sensor.get_depth_scale()
        align_local = rs.align(rs.stream.color)
        intr_local = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # 전역 바인딩
        align_to_color = align_local
        intr = intr_local
        globals()['depth_scale'] = depth_scale_local

    except Exception as e:
        print(f"[ERROR] RealSense initialization failed: {e}")
        class DummyIntrinsics:
            fx, fy, ppx, ppy = 640.0, 480.0, 320.0, 240.0
        intr = DummyIntrinsics()
        # pipeline이 시작 실패했을 수 있으므로 None으로 유지

# =========================
# 메인 시퀀스
# =========================
async def async_main():
    print("\n" + "="*70)
    print("YOLOv8 + D435i IMU Forklift Autonomous Control (Plane Fit Alpha)")
    print("="*70)

    # 1) load_base.py 선 실행
    print("\n=== [1/3] Running load_base.py ===")
    await run_script(LOAD_BASE_PATH)

    # 2) YOLO/RealSense 초기화
    print("\n=== Initialize YOLO & RealSense ===")
    init_yolo_and_realsense()

    ctrl = None
    try:
        ctrl = AutonomousForkliftController(can_channel=0, can_bitrate=500_000, use_extended_ids=False)
        await ctrl.start()

        # 2-1) 비전/계측
        mission_params = await run_vision_detection_autonomous(ctrl)
        if mission_params is None:
            print("[INFO] Vision detection aborted")
            return
        
        alpha_deg, proj_len_m, dir_cw, wheel_dist = mission_params

        print("\n미션 실행 준비 완료!")
        input("Enter를 눌러 4단계 미션을 시작하세요...")

        # 2-2) 미션
        await run_four_step_mission(ctrl, alpha_deg, proj_len_m, dir_cw, wheel_dist)

    except ValueError:
        print("[ERROR] 올바른 숫자를 입력하세요")
    except KeyboardInterrupt:
        print("\n\n[INFO] 프로그램 종료")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ctrl:
            print("[INFO] Stopping Forklift Controller...")
            try:
                await ctrl.stop()
            except Exception as e:
                print(f"[WARN] ctrl.stop() error: {e}")
        print("[INFO] Stopping RealSense Pipeline...")
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception as e:
            print(f"[WARN] pipeline.stop() error: {e}")

    # 3) loading.py 후 실행
    print("\n=== [3/3] Running loading.py ===")
    await run_script(LOADING_PATH)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
