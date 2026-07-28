import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.imu_yaw import RelYawEstimator
from eval.imu_yaw import RelYawReader


class _MD:  # rs.motion_data 모방 (x,y,z)
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z


def test_first_frame_returns_zero():
    est = RelYawEstimator()
    assert est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 0.0, 0), 0.0) == 0.0


def test_integrates_gyro_y_to_degrees():
    # gyro.y = 1 rad/s 를 1초 적분 → 약 57.3deg
    est = RelYawEstimator()
    est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 0.0)      # first → 0
    rel = est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 1000.0)  # +1s
    assert abs(rel - math.degrees(1.0)) < 1e-6


def test_reset_resnaps_reference():
    est = RelYawEstimator()
    est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 0.0)
    est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 1000.0)   # rel≈57.3
    est.reset()
    rel = est.update_from_frames(_MD(0, 0, 9.8), _MD(0, 0.0, 0), 1500.0)
    assert abs(rel) < 1e-6      # reset 후 기준이 현재로 재설정 → 0 근처


def test_reader_processes_injected_frame():
    # RealSense 없이 estimator 경로만 검증: _process_motion 직접 호출
    reader = RelYawReader.__new__(RelYawReader)      # __init__ 우회(파이프라인 미생성)
    reader._est = RelYawEstimator()
    reader._lock = __import__("threading").Lock()
    reader._rel = 0.0
    reader._process_motion(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 0.0)
    reader._process_motion(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 1000.0)
    assert abs(reader.rel_yaw - math.degrees(1.0)) < 1e-6
    reader.reset()
    reader._process_motion(_MD(0, 0, 9.8), _MD(0, 0.0, 0), 1500.0)
    assert abs(reader.rel_yaw) < 1e-6


def _bare_reader():
    reader = RelYawReader.__new__(RelYawReader)      # __init__ 우회(파이프라인 미생성)
    reader._est = RelYawEstimator()
    reader._lock = __import__("threading").Lock()
    reader._rel = 0.0
    reader._recording = False
    reader._buffer = []
    reader._rec_meta = {}
    reader._t0 = None
    return reader


def test_recording_buffers_imu_samples():
    # 백그라운드 관찰: start_recording 후 _process_motion 샘플이 버퍼에 IMU 6축+meta로 쌓임
    reader = _bare_reader()
    reader.start_recording(state="ROTATE", cmd="ROTATE", cmd_bytes="127 127 127 127 118 127 127 127")
    reader._process_motion(_MD(0.1, 0.2, 9.8), _MD(0.3, 1.0, 0.4), 0.0)
    reader._process_motion(_MD(0.1, 0.2, 9.8), _MD(0.3, 1.0, 0.4), 1000.0)
    rows = reader.stop_recording()
    assert len(rows) == 2
    r = rows[0]
    for k in ("t_s", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
              "rel_yaw_deg", "state", "cmd", "cmd_bytes"):
        assert k in r
    assert r["gyro_y"] == 1.0 and r["accel_x"] == 0.1
    assert r["state"] == "ROTATE" and r["cmd_bytes"].endswith("127")


def test_recording_off_does_not_buffer():
    # 기록 안 켰으면 버퍼 안 쌓임 + rel_yaw 계산은 정상(움직임 영향 0)
    reader = _bare_reader()
    reader._process_motion(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 0.0)
    reader._process_motion(_MD(0, 0, 9.8), _MD(0, 1.0, 0), 1000.0)
    assert reader.stop_recording() == []
    assert abs(reader.rel_yaw - math.degrees(1.0)) < 1e-6
