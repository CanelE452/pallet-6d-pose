"""RealSense IMU 기반 rel_yaw 추정 (gyro.Y 적분).

RelYawEstimator : 순수 적분 로직 (depth_cam/main_rec.py RelYawEstimator 이식 + reset 추가).
RelYawReader    : RealSense 파이프라인을 백그라운드 스레드로 폴링 (Task 3).

로깅: RelYawReader 는 회전 종료 판정용 IMU 리더이며, 움직임 로직(명령/종료조건)과 무관하다.
start_recording()/stop_recording() 은 폴링 중 들어오는 IMU 샘플을 옆에서 버퍼에 베껴 적는
순수 관찰 기능 — rel_yaw 계산/타이밍에 영향을 주지 않는다(움직임 코드 불변).
"""
from __future__ import annotations
import math
import threading


class RelYawEstimator:
    """gyro.Y(rad/s) 적분 → rel_yaw(deg, [-180,180]). reset()으로 기준 재스냅."""

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.first = True
        self.last_ts_ms = None
        self.yaw_deg = 0.0
        self.init_yaw = 0.0
        self.last_rel = 0.0

    def reset(self):
        """현재 누적 yaw를 새 기준(0°)으로 — 회전 평가 시작 시 ref 스냅."""
        self.init_yaw = self.yaw_deg
        self.last_rel = 0.0

    def update_from_frames(self, accel, gyro, ts_ms: float) -> float:
        if self.first:
            self.first = False
            self.last_ts_ms = ts_ms
            self.init_yaw = self.yaw_deg
            self.last_rel = 0.0
            return 0.0
        dt = max(0.0, (ts_ms - self.last_ts_ms) / 1000.0)
        self.last_ts_ms = ts_ms
        self.yaw_deg += math.degrees(gyro.y * dt)         # 핵심: gyro.Y 적분
        rel = self.yaw_deg - self.init_yaw
        rel = (rel + 180.0) % 360.0 - 180.0               # [-180,180] wrap
        self.last_rel = rel
        return rel


def _rs():
    """pyrealsense2 lazy import. 없으면 None."""
    try:
        import pyrealsense2 as rs
        return rs
    except Exception:
        return None


class RelYawReader:
    """RealSense gyro/accel을 백그라운드로 폴링 → rel_yaw 제공.

    available() False면 RealSense 미설치/미연결 → 회전 평가 불가(거리 평가는 가능).
    """

    def __init__(self, alpha: float = 0.98):
        self._est = RelYawEstimator(alpha=alpha)
        self._lock = threading.Lock()
        self._rel = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._pipeline = None
        self._last_accel = None
        # ── 백그라운드 기록(관찰) 버퍼 — 움직임 로직과 무관 ──
        self._recording = False
        self._buffer = []
        self._rec_meta = {}
        self._t0 = None
        rs = _rs()
        self._rs = rs
        self._ok = False
        if rs is not None:
            try:
                self._pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.accel)
                cfg.enable_stream(rs.stream.gyro)
                self._cfg = cfg
                self._ok = True
            except Exception:
                self._ok = False

    def available(self) -> bool:
        return self._ok

    @property
    def rel_yaw(self) -> float:
        with self._lock:
            return self._rel

    def reset(self):
        with self._lock:
            self._est.reset()
            self._rel = 0.0

    # ── 관찰 기록 API (움직임 코드 밖에서 호출) ──
    def start_recording(self, **meta):
        """이 시점부터 들어오는 IMU 샘플을 버퍼에 기록 시작. meta(state/cmd/cmd_bytes)는 각 행에 박힘."""
        import time as _t
        with self._lock:
            self._buffer = []
            self._rec_meta = dict(meta)
            self._t0 = _t.monotonic()
            self._recording = True

    def stop_recording(self):
        """기록 중지하고 누적 행(list[dict]) 반환."""
        with self._lock:
            self._recording = False
            rows = self._buffer
            self._buffer = []
            return rows

    def _process_motion(self, accel, gyro, ts_ms: float):
        rel = self._est.update_from_frames(accel, gyro, ts_ms)
        with self._lock:
            self._rel = rel
            if getattr(self, "_recording", False):
                import time as _t
                t = (_t.monotonic() - self._t0) if self._t0 is not None else 0.0
                self._buffer.append({
                    "t_s": round(t, 4),
                    "gyro_x": gyro.x, "gyro_y": gyro.y, "gyro_z": gyro.z,
                    "accel_x": accel.x, "accel_y": accel.y, "accel_z": accel.z,
                    "rel_yaw_deg": round(rel, 4),
                    **self._rec_meta,
                })

    def _loop(self):
        rs = self._rs
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except Exception:
                continue
            accel = gyro = None
            ts = 0.0
            for f in frames:
                prof = f.get_profile()
                md = f.as_motion_frame().get_motion_data()
                ts = f.get_timestamp()
                if prof.stream_type() == rs.stream.accel:
                    accel = md
                    self._last_accel = md
                elif prof.stream_type() == rs.stream.gyro:
                    gyro = md
            use_accel = accel or self._last_accel
            if gyro is not None and use_accel is not None:
                self._process_motion(use_accel, gyro, ts)

    def start(self):
        if not self._ok:
            raise RuntimeError("RealSense 미가용 — 회전 평가 불가")
        self._pipeline.start(self._cfg)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="RelYawReader", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pipeline is not None and self._ok:
            try:
                self._pipeline.stop()
            except Exception:
                pass
