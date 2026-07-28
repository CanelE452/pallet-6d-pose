# file: depth_cam_2/logger.py
# 목적
# - RealSense RGB-D로 '벽'까지의 거리(dist_z)를 기존 파이프라인과 동일한 개념으로 계산:
#     ROI/마스크 → Depth 샘플 → Deprojection(픽셀→3D) → z-중앙값 인라이어 → 인라이어 평균 → dist_z = mean_z
# - (전진) dist_z ≤ target_depth_m 도달 시 STOP 지속 / (후진) dist_z ≥ target_depth_m 도달 시 STOP 지속
# - UI: 좌측 RGB, 우측 Depth(컬러맵) 미리보기 + ROI/거리/상태/FPS 오버레이
# - CAN 없음/미연결이어도 실행(명령은 no-op). 있을 경우 calib/control.py API 자동연결.
#
# 실행 예:
#   # 전진(기본)
#   python depth_cam_2/logger.py --target-depth 2.0 --note forward_to_2m
#   # 후진
#   python depth_cam_2/logger.py --direction backward --target-depth 10.0 --note backward_to_10m
#   # ROI/스케일/풀스크린
#   python depth_cam_2/logger.py --roi-w 0.4 --roi-h 0.3 --roi-y 0.35 --ui-scale 1.8 --fullscreen
#
from __future__ import annotations
import argparse, csv, json, time, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import cv2
import pyrealsense2 as rs
import importlib

# ─────────────────────────────────────────
# 설정(프로젝트 기본값과 호환되는 범용 값)
# ─────────────────────────────────────────
SAMPLE_STRIDE   = 4       # ROI 내 픽셀 샘플링 간격(픽셀)
Z_INLIER_THRESH = 0.15    # z 중앙값 대비 인라이어 허용(m)
MIN_POINTS      = 80      # 인라이어 최소 개수

# ─────────────────────────────────────────
# 로그 경로/스키마 (project_root/logs/drive_calib/)
# ─────────────────────────────────────────
LOG_BASE = Path(__file__).resolve().parents[1] / "logs" / "drive_calib"
RAW_DIR = LOG_BASE / "raw"
META_DIR = LOG_BASE / "meta"
MODEL_DIR = LOG_BASE / "model"
SUMMARY_CSV = LOG_BASE / "summary.csv"
CSV_FIELDS = ["t_monotonic","sys_ts","state","cmd","dist_z","dist_euclid","vx_est","note"]

# ─────────────────────────────────────────
# 로거 구현
# ─────────────────────────────────────────
class DriveCalibLogger:
    def __init__(self, base_dir: Path = LOG_BASE):
        self.base = base_dir
        (self.base/"raw").mkdir(parents=True, exist_ok=True)
        (self.base/"meta").mkdir(parents=True, exist_ok=True)
        (self.base/"model").mkdir(parents=True, exist_ok=True)
        self.summary_path = self.base/"summary.csv"
        if not self.summary_path.exists():
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["run_id","start_time","travel_m","duration_s","vmax_est","a_est","notes"]
                )

    @staticmethod
    def _new_run_id(suffix: str = "run") -> str:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_{suffix}"

    # direction에 따라 run_id 접미사 분리(fwd/back)
    def start_run(self, stop_depth_m: float = 2.0, meta_extra: Optional[dict] = None, run_suffix: str = "run"):
        self.run_id = self._new_run_id(run_suffix)
        self.t0 = time.monotonic()
        self.rows = []
        self.stop_depth_m = float(stop_depth_m)
        self.meta = {
            "run_id": self.run_id,
            "start_time": dt.datetime.now().isoformat(timespec="seconds"),
            "stop_depth_m": self.stop_depth_m
        }
        if meta_extra: self.meta.update(meta_extra)

    def log_frame(self, state, cmd, dist_z, dist_euclid=None, vx_est=None, note=None):
        self.rows.append({
            "t_monotonic": round(time.monotonic() - self.t0, 6),
            "sys_ts": dt.datetime.now().isoformat(timespec="milliseconds"),
            "state": state,
            "cmd": cmd,
            "dist_z": ("" if dist_z is None else float(dist_z)),
            "dist_euclid": ("" if dist_euclid is None else float(dist_euclid)),
            "vx_est": ("" if vx_est is None else float(vx_est)),
            "note": (note or "")
        })

    def finish_run(self, notes="") -> dict:
        if not getattr(self, "rows", None):
            return {}
        raw_path = RAW_DIR / f"{self.run_id}.csv"
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader()
            for r in self.rows: w.writerow(r)

        dist_list = [float(r["dist_z"]) for r in self.rows if str(r["dist_z"]) not in ("","None")]
        t_list = [float(r["t_monotonic"]) for r in self.rows]
        start_d = dist_list[0] if dist_list else None
        min_d = min(dist_list) if dist_list else None
        max_d = max(dist_list) if dist_list else None
        travel = None
        if start_d is not None and (min_d is not None or max_d is not None):
            travel = (max_d - min_d) if (max_d is not None and min_d is not None) else None
        duration = (t_list[-1] - t_list[0]) if t_list else None

        vmax_est, a_est = None, None
        if len(dist_list) >= 5 and duration and travel:
            mid = len(dist_list)//2
            if len(dist_list)-mid > 3:
                dd = abs(dist_list[mid] - dist_list[-1]); dtm = t_list[-1] - t_list[mid]
                if dtm > 0: vmax_est = max(0.0, dd/dtm)
            k = max(3, int(len(dist_list)*0.2))
            if len(dist_list) > k:
                s0 = abs(dist_list[0] - dist_list[k]); t0 = t_list[k] - t_list[0]
                if s0 and t0 and t0>0: a_est = max(0.0, 2.0*s0/(t0*t0))

        meta_path = META_DIR / f"{self.run_id}.json"
        meta_out = dict(self.meta); meta_out.update({
            "raw_csv": str(raw_path.as_posix()),
            "start_depth_m": start_d,
            "min_depth_m": min_d,
            "max_depth_m": max_d,
            "travel_m": travel,
            "duration_s": duration,
            "vmax_est": vmax_est,
            "a_est": a_est,
            "notes": notes
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self.run_id, self.meta["start_time"], travel, duration, vmax_est, a_est, notes])

        return {"raw_csv": str(raw_path), "meta_json": str(meta_path)}

# ─────────────────────────────────────────
# RealSense 파이프라인 & UI 보조
# ─────────────────────────────────────────
def setup_rs():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()  # 깊이 컬러맵
    return pipeline, align, colorizer

def center_roi(W:int, H:int, frac_w:float=0.4, frac_h:float=0.3, top_frac:float=0.35):
    rw = max(0.05, min(0.95, float(frac_w)))
    rh = max(0.05, min(0.95, float(frac_h)))
    ty = max(0.0, min(1.0-rh, float(top_frac)))
    cx, cy = W//2, int(H*(ty + rh/2.0))
    w, h = int(W*rw), int(H*rh)
    x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
    x2, y2 = min(W-1, x1 + w), min(H-1, y1 + h)
    return x1, y1, x2, y2

def draw_overlay(img, text_lines, org=(10,25), scale: float = 1.0):
    """UI 스케일에 맞춘 텍스트 오버레이"""
    font_scale = 0.55 * max(0.5, scale)
    thickness  = max(1, int(2 * scale))
    line_step  = int(22 * scale)
    y = org[1]
    for s in text_lines:
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y += line_step

# ─────────────────────────────────────────
# 거리 계산(프로젝트와 동일 개념: inlier 평균 z)
# ─────────────────────────────────────────
MaskOrROI = Union[np.ndarray, Tuple[int,int,int,int]]

def _collect_pixels_from_mask_or_roi(h:int, w:int, mask_or_roi: MaskOrROI, stride:int) -> List[Tuple[int,int]]:
    pixels: List[Tuple[int,int]] = []
    if isinstance(mask_or_roi, tuple):
        x1,y1,x2,y2 = mask_or_roi
        x1=max(0,min(w-1,int(x1))); x2=max(0,min(w,int(x2)))
        y1=max(0,min(h-1,int(y1))); y2=max(0,min(h,int(y2)))
        for v in range(y1, y2, stride):
            for u in range(x1, x2, stride):
                pixels.append((u,v))
    else:
        mask = mask_or_roi
        assert mask.ndim==2 and mask.shape==(h,w), "mask size mismatch"
        ys, xs = np.where(mask>0)
        if len(xs)==0:
            return []
        xs = xs[::stride]; ys = ys[::stride]
        pixels = list(zip(xs, ys))
    return pixels

def robust_points_from_mask_or_roi(depth_frame,
                                   depth_intrin,
                                   mask_or_roi: MaskOrROI,
                                   stride: int = SAMPLE_STRIDE,
                                   z_inlier_thresh: float = Z_INLIER_THRESH,
                                   min_points: int = MIN_POINTS):
    """ROI/마스크 영역에서 stride 샘플 → get_distance → deproject → z중앙값 인라이어 → 인라이어 포인트 반환"""
    h = depth_frame.get_height(); w = depth_frame.get_width()
    get_distance = depth_frame.get_distance

    pixels = _collect_pixels_from_mask_or_roi(h, w, mask_or_roi, stride)
    if not pixels:
        return False, None

    pts = []
    for (u,v) in pixels:
        z = get_distance(int(u), int(v))
        if z and z>0:
            X,Y,Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(z))
            pts.append((X,Y,Z))
    if not pts:
        return False, None

    pts = np.asarray(pts, dtype=np.float32)
    z_med = np.median(pts[:,2])
    dz = np.abs(pts[:,2] - z_med)
    in_mask = dz <= float(z_inlier_thresh)
    pts_in = pts[in_mask]
    if pts_in.shape[0] < int(min_points):
        return False, None

    return True, pts_in

def roi_to_mask(depth_frame, roi_xyxy):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    x1, y1, x2, y2 = roi_xyxy
    x1 = max(0, min(w-1, int(x1))); x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h-1, int(y1))); y2 = max(0, min(h, int(y2)))
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask

def compute_wall_distance(depth_frame, depth_intrin, roi_xyxy):
    """ROI → (마스크 또는 튜플 그대로) → 인라이어 평균 → dist_z & dist_euclid"""
    mask = roi_to_mask(depth_frame, roi_xyxy)
    ok_pts, pts_in = robust_points_from_mask_or_roi(
        depth_frame=depth_frame, depth_intrin=depth_intrin,
        mask_or_roi=mask, stride=SAMPLE_STRIDE,
        z_inlier_thresh=Z_INLIER_THRESH, min_points=MIN_POINTS
    )
    if not ok_pts or pts_in is None or len(pts_in)==0:
        return None, None
    c3d_mean = pts_in.mean(axis=0).astype(np.float32)  # [mx,my,mz]
    dist_euclid = float(np.linalg.norm(c3d_mean))
    dist_z = float(c3d_mean[2])
    return dist_z, dist_euclid

# ─────────────────────────────────────────
# CAN 제어: calib/control.py가 있으면 우선 사용, 없으면 no-op
# ─────────────────────────────────────────
def _pick_first_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    return None

def _no_op(*args, **kwargs):
    return False

def resolve_can_api():
    """
    calib/control.py의 API를 우선 연결:
      can_init, can_close, start_heartbeat, stop_heartbeat,
      issue_command_forward, issue_command_backward, issue_command_stop
    없거나 실패하면 no-op으로 대체하여 실행이 끊기지 않도록 함.
    """
    mod = None
    for name in ("calib.control", "depth_cam_2.calib.control"):
        try:
            mod = importlib.import_module(name)
            break
        except Exception:
            continue

    if mod is None:
        return {
            "can_init": _no_op, "can_close": _no_op,
            "start_hb": _no_op, "stop_hb": _no_op,
            "cmd_fwd": _no_op, "cmd_back": _no_op, "cmd_stop": _no_op,
            "available": False,
        }

    CAN_INIT   = _pick_first_attr(mod, ["can_init", "init_can", "initialize", "setup"])
    CAN_CLOSE  = _pick_first_attr(mod, ["can_close", "close_can", "shutdown", "teardown"])
    HB_START   = _pick_first_attr(mod, ["start_heartbeat", "heartbeat_start", "startHB"])
    HB_STOP    = _pick_first_attr(mod, ["stop_heartbeat", "heartbeat_stop", "stopHB"])
    CMD_FWD    = _pick_first_attr(mod, ["issue_command_forward","forward","send_forward","cmd_forward","drive_forward"])
    CMD_BACK   = _pick_first_attr(mod, ["issue_command_backward","backward","send_backward","cmd_backward","drive_backward"])
    CMD_STOP   = _pick_first_attr(mod, ["issue_command_stop","stop","send_stop","cmd_stop","drive_stop","halt"])

    return {
        "can_init": CAN_INIT or _no_op,
        "can_close": CAN_CLOSE or _no_op,
        "start_hb": HB_START or _no_op,
        "stop_hb": HB_STOP or _no_op,
        "cmd_fwd": CMD_FWD or _no_op,
        "cmd_back": CMD_BACK or _no_op,
        "cmd_stop": CMD_STOP or _no_op,
        "available": True,
    }

# ─────────────────────────────────────────
# UI 포함 주행 & 로깅 루프 (direction별 임계 로직 + UI 스케일)
# ─────────────────────────────────────────
def run_with_wall_logging(logger: DriveCalibLogger,
                          direction: str = "forward",   # forward/backward
                          target_depth_m: float = 0.0,
                          sample_hz: float = 50.0,
                          note: str = "",
                          start_hold_s: float = 5.0,     # 기본 5초 안정화
                          stop_hold_s: float = 1.0,
                          hard_timeout_s: Optional[float] = 30.0,
                          roi_frac_w: float = 0.4,
                          roi_frac_h: float = 0.3,
                          roi_top_frac: float = 0.35,
                          dry_run: bool = False,
                          ui_scale: float = 1.0,
                          fullscreen: bool = True):
    assert direction in ("forward","backward"), "direction must be 'forward' or 'backward'"

    pipeline, align, colorizer = setup_rs()
    api = resolve_can_api()
    can_ok = False
    if not dry_run and api["available"]:
        try:
            can_ok = bool(api["can_init"]())
        except Exception as e:
            print(f"[WARN] CAN init failed: {e}")
            can_ok = False
        if can_ok:
            try: api["start_hb"]()
            except Exception: pass

    dt_s = 1.0/max(1.0, float(sample_hz))
    t0 = time.monotonic()
    stop_flag = False
    start_logged = False
    frame_count = 0
    t_fps = time.time()
    fps = 0.0
    run_started = False

    cv2.namedWindow("Wall Logger", cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty("Wall Logger", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def send_forward():
        if not dry_run and can_ok:
            try: api["cmd_fwd"]()
            except Exception: pass

    def send_backward():
        if not dry_run and can_ok:
            try: api["cmd_back"]()
            except Exception: pass

    def send_stop():
        if not dry_run and can_ok:
            try: api["cmd_stop"]()
            except Exception: pass

    rect_th = max(1, int(2 * ui_scale))  # ROI 박스 두께
    text_org = (int(10*ui_scale), int(25*ui_scale))

    # 메타 공통정보(안정화 이후 start_run에서 사용)
    run_suffix = "fwd" if direction=="forward" else "back"
    meta_extra = {
        "direction": direction,
        "command_profile": ("FORWARD_const" if direction=="forward" else "BACKWARD_const"),
        "sample_hz": sample_hz,
        "roi_frac_w": roi_frac_w, "roi_frac_h": roi_frac_h, "roi_top_frac": roi_top_frac,
        "ui_scale": ui_scale, "fullscreen": fullscreen
    }

    try:
        # ── (수정) 시작 안정화 구간: 로깅/명령 전송 없음
        if start_hold_s and start_hold_s > 0:
            t_hold = time.monotonic()
            while True:
                elapsed = time.monotonic() - t_hold
                if elapsed >= start_hold_s:
                    break
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth = aligned.get_depth_frame(); color = aligned.get_color_frame()
                if not depth or not color:
                    time.sleep(dt_s); continue

                W,H = color.get_width(), color.get_height()
                depth_intrin = depth.profile.as_video_stream_profile().intrinsics
                x1,y1,x2,y2 = center_roi(W,H,roi_frac_w,roi_frac_h,roi_top_frac)
                # 센서 파이프라인 워밍업을 위해 계산은 하지만, 기록은 하지 않음
                dZ, dE = compute_wall_distance(depth, depth_intrin, (x1,y1,x2,y2))

                # UI: 안정화 카운트다운
                color_np = np.asanyarray(color.get_data()).copy()
                depth_color = np.asanyarray(colorizer.colorize(depth).get_data())
                cv2.rectangle(color_np, (x1,y1), (x2,y2), (0,255,0), rect_th)
                remain = max(0.0, start_hold_s - elapsed)
                text = [
                    f"STATE: STABILIZING   CMD: IDLE   DIR: {direction}",
                    f"Stabilizing... {remain:0.1f}s remaining",
                    f"dist_z(prev): {dZ:.3f} m" if dZ is not None else "dist_z(prev): N/A",
                ]
                draw_overlay(color_np, text, text_org, scale=ui_scale)
                view = np.hstack([color_np, depth_color])
                if ui_scale != 1.0:
                    view = cv2.resize(view, None, fx=ui_scale, fy=ui_scale, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Wall Logger", view)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    raise KeyboardInterrupt
                time.sleep(dt_s)

        # ── 안정화 완료 후에 로깅 시작
        logger.start_run(stop_depth_m=target_depth_m, meta_extra=meta_extra, run_suffix=run_suffix)
        run_started = True

        # 루프 시작
        while True:
            if (hard_timeout_s is not None) and (time.monotonic()-t0 > hard_timeout_s):
                send_stop()
                logger.log_frame(state="STOP", cmd="STOP", dist_z=None, note="HARD_TIMEOUT")
                break

            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame(); color = aligned.get_color_frame()
            if not depth or not color: time.sleep(dt_s); continue

            W,H = color.get_width(), color.get_height()
            depth_intrin = depth.profile.as_video_stream_profile().intrinsics
            x1,y1,x2,y2 = center_roi(W,H,roi_frac_w,roi_frac_h,roi_top_frac)
            dZ, dE = compute_wall_distance(depth, depth_intrin, (x1,y1,x2,y2))

            if not start_logged and dZ is not None:
                logger.log_frame(state="IDLE", cmd="IDLE", dist_z=dZ, dist_euclid=dE, note="RUN_START")
                start_logged = True

            # ── 임계 도달 판단: forward(≤), backward(≥)
            hit_target = False
            if dZ is not None:
                if direction == "forward":
                    hit_target = (dZ <= target_depth_m)
                else:  # backward
                    hit_target = (dZ >= target_depth_m)

            # CAN/드라이런 가드
            if (not stop_flag) and hit_target and can_ok and (not dry_run):
                stop_flag = True
                send_stop()
                logger.log_frame(state="STOP", cmd="STOP", dist_z=dZ, dist_euclid=dE, note="HIT_TARGET_DEPTH")

            if stop_flag:
                send_stop()
                state_txt, cmd_txt = "STOP", "STOP"
                logger.log_frame(state=state_txt, cmd=cmd_txt, dist_z=dZ, dist_euclid=dE, note="STOPPING")
            else:
                if direction == "forward":
                    send_forward()
                    state_txt, cmd_txt = "FORWARD", "FORWARD"
                else:
                    send_backward()
                    state_txt, cmd_txt = "BACKWARD", "BACKWARD"
                logger.log_frame(state=state_txt, cmd=cmd_txt, dist_z=dZ, dist_euclid=dE)

            # UI
            color_np = np.asanyarray(color.get_data()).copy()
            depth_color = np.asanyarray(colorizer.colorize(depth).get_data())
            cv2.rectangle(color_np, (x1,y1), (x2,y2), (0,255,0), rect_th)

            frame_count += 1
            if time.time() - t_fps >= 0.5:
                fps = frame_count / (time.time() - t_fps)
                frame_count = 0; t_fps = time.time()

            text = [
                f"STATE: {state_txt:<8}  CMD: {cmd_txt:<8}  DIR: {direction:<8}  FPS: {fps:0.1f}",
                f"dist_z: {dZ:.3f} m" if dZ is not None else "dist_z: N/A",
                f"dist_euclid: {dE:.3f} m" if dE is not None else "dist_euclid: N/A",
                f"Target: {target_depth_m:.2f} m   ROI(w,h,y): {roi_frac_w:.2f},{roi_frac_h:.2f},{roi_top_frac:.2f}",
                "Press 'q' or 'ESC' to quit"
            ]
            draw_overlay(color_np, text, text_org, scale=ui_scale)
            view = np.hstack([color_np, depth_color])
            if ui_scale != 1.0:
                view = cv2.resize(view, None, fx=ui_scale, fy=ui_scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Wall Logger", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            time.sleep(dt_s)

            # STOP 모드로 전환된 뒤 stop_hold_s 유지 후 종료
            if stop_flag:
                t_hold0 = time.monotonic()
                while time.monotonic()-t_hold0 < stop_hold_s:
                    frames = pipeline.wait_for_frames()
                    aligned = align.process(frames)
                    depth = aligned.get_depth_frame(); color = aligned.get_color_frame()
                    if not depth or not color: time.sleep(dt_s); continue
                    W,H = color.get_width(), color.get_height()
                    depth_intrin = depth.profile.as_video_stream_profile().intrinsics
                    x1,y1,x2,y2 = center_roi(W,H,roi_frac_w,roi_frac_h,roi_top_frac)
                    dZ2, dE2 = compute_wall_distance(depth, depth_intrin, (x1,y1,x2,y2))
                    send_stop()
                    logger.log_frame(state="STOP", cmd="STOP", dist_z=dZ2, dist_euclid=dE2, note="STOP_HOLD")

                    # UI hold
                    color_np = np.asanyarray(color.get_data()).copy()
                    depth_color = np.asanyarray(colorizer.colorize(depth).get_data())
                    cv2.rectangle(color_np, (x1,y1), (x2,y2), (0,255,0), rect_th)
                    draw_overlay(color_np, [
                        f"STATE: STOP     CMD: STOP     DIR: {direction}  (HOLD)   FPS: {fps:0.1f}",
                        f"dist_z: {dZ2:.3f} m" if dZ2 is not None else "dist_z: N/A",
                        f"dist_euclid: {dE2:.3f} m" if dE2 is not None else "dist_euclid: N/A",
                        f"Target: {target_depth_m:.2f} m   ROI(w,h,y): {roi_frac_w:.2f},{roi_frac_h:.2f},{roi_top_frac:.2f}",
                        "Press 'q' or 'ESC' to quit"
                    ], text_org, scale=ui_scale)
                    view = np.hstack([color_np, depth_color])
                    if ui_scale != 1.0:
                        view = cv2.resize(view, None, fx=ui_scale, fy=ui_scale, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Wall Logger", view)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
                    time.sleep(dt_s)
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try: pipeline.stop()
        except: pass
        # CAN 종료 루틴(있을 때만)
        if not dry_run and can_ok:
            try: api["cmd_stop"]()
            except: pass
            try: api["stop_hb"]()
            except: pass
            try: api["can_close"]()
            except: pass

    default_note = (note or (("forward_until_wall_threshold") if direction=="forward"
                             else ("backward_until_wall_threshold")))
    paths = {}
    if run_started:
        paths = logger.finish_run(notes=default_note)
        print(f"[OK] Saved raw:  {paths.get('raw_csv')}")
        print(f"[OK] Saved meta: {paths.get('meta_json')}")
    else:
        print("[INFO] Stabilization phase aborted before run start. No logs saved.")

# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Drive forward/backward until wall distance reaches target, then STOP (continuous) and log. Shows RGB-D UI.")
    p.add_argument("--direction", type=str, choices=["forward","backward"], default="forward")
    p.add_argument("--target-depth", type=float, default=2.0)
    p.add_argument("--sample-hz", type=float, default=50.0)
    p.add_argument("--note", type=str, default="")
    p.add_argument("--start-hold-s", type=float, default=5.0, help="시작 안정화 시간(초). 이 시간 동안 로깅/제어하지 않음")
    p.add_argument("--stop-hold-s", type=float, default=1.0)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--roi-w", type=float, default=0.4)
    p.add_argument("--roi-h", type=float, default=0.3)
    p.add_argument("--roi-y", type=float, default=0.35)
    # UI 옵션 추가
    p.add_argument("--ui-scale", type=float, default=1.5, help="UI 화면 스케일 배율 (1.0=기본)")
    p.add_argument("--fullscreen", action="store_true", help="OpenCV 창을 전체 화면으로 표시")
    return p.parse_args()

def main():
    args = parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_BASE.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run_id","start_time","travel_m","duration_s","vmax_est","a_est","notes"])

    run_with_wall_logging(
        logger=DriveCalibLogger(),
        direction=str(args.direction),
        target_depth_m=float(args.target_depth),
        sample_hz=float(args.sample_hz),
        note=args.note,
        start_hold_s=float(args.start_hold_s),
        stop_hold_s=float(args.stop_hold_s),
        hard_timeout_s=(None if args.timeout_s is None else float(args.timeout_s)),
        roi_frac_w=float(args.roi_w),
        roi_frac_h=float(args.roi_h),
        roi_top_frac=float(args.roi_y),
        dry_run=bool(args.dry_run),
        ui_scale=float(args.ui_scale),
        fullscreen=bool(args.fullscreen),
    )

if __name__ == "__main__":
    main()
