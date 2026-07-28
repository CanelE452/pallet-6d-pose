import csv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.eval_motion import wrap_to_180, append_eval_row, append_calib_row, EVAL_HEADER


def test_wrap_to_180():
    assert wrap_to_180(190.0) == -170.0
    assert wrap_to_180(-190.0) == 170.0
    assert wrap_to_180(90.0) == 90.0


def test_append_eval_row_writes_header_once(tmp_path):
    p = tmp_path / "motion_eval.csv"
    append_eval_row(str(p), {"kind": "rotate", "target": 90.0, "direction": "cw",
                             "power": 118, "imu_stop_deg": 90.1, "imu_settled_deg": 92.4,
                             "elapsed_s": 3.2, "cmd_time_s": "", "measured": 91.8,
                             "reached": True, "note": ""})
    append_eval_row(str(p), {"kind": "drive_eval", "target": 2.0, "direction": "fwd",
                             "power": 60, "imu_stop_deg": "", "imu_settled_deg": "",
                             "elapsed_s": 7.1, "cmd_time_s": 7.0, "measured": 1.97,
                             "reached": True, "note": ""})
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == EVAL_HEADER                 # 헤더 1회
    assert len(rows) == 3                          # 헤더 + 2행
    # error 자동계산: rotate 91.8-90=1.8, drive 1.97-2.0=-0.03
    di = {h: i for i, h in enumerate(EVAL_HEADER)}
    assert abs(float(rows[1][di["error"]]) - 1.8) < 1e-9
    assert abs(float(rows[2][di["error"]]) - (-0.03)) < 1e-9


def test_append_calib_row(tmp_path):
    p = tmp_path / "calib_fwd.csv"
    append_calib_row(str(p), T_sec=5.0, d_measured=1.23, direction="fwd", power=60)
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["T_sec"] == "5.0" and rows[0]["d_measured"] == "1.23"


from eval.eval_motion import rotate_sequence, RotateResult


class _FakeReader:
    """호출마다 rel_yaw가 step씩 증가하는 가짜 IMU."""
    def __init__(self, step):
        self._v = 0.0
        self._step = step
    def reset(self):
        self._v = 0.0
    @property
    def rel_yaw(self):
        self._v += self._step
        return self._v


def test_rotate_sequence_stops_at_target():
    issued = []
    reader = _FakeReader(step=5.0)   # 매 폴링 +5°
    res = rotate_sequence(
        reader=reader, target_deg=90.0, turn_dir=-1,
        issue_rotate=lambda d: issued.append(("rot", d)),
        issue_stop=lambda: issued.append(("stop",)),
        period_s=0.0, settle_s=0.0, timeout_s=100.0,
        sleep=lambda s: None, now=_make_clock(),
    )
    assert isinstance(res, RotateResult)
    assert res.reached is True
    assert res.imu_stop_deg >= 90.0
    assert ("stop",) in issued                      # 종료 시 stop 호출


def test_rotate_sequence_timeout():
    reader = _FakeReader(step=0.0)                  # 영원히 0° → 타임아웃
    res = rotate_sequence(
        reader=reader, target_deg=90.0, turn_dir=-1,
        issue_rotate=lambda d: None, issue_stop=lambda: None,
        period_s=0.0, settle_s=0.0, timeout_s=0.05,
        sleep=lambda s: None, now=_make_clock(dt=0.02),
    )
    assert res.reached is False


def _make_clock(dt=0.01):
    t = {"v": 0.0}
    def now():
        t["v"] += dt
        return t["v"]
    return now


# drive_sequence 는 움직임 코드(원본) — 로깅 인자 없음. 로깅은 RelYawReader 백그라운드 버퍼 담당.
from eval.eval_motion import drive_sequence, write_raw_motion, RAW_HEADER, _move_bytes


def test_drive_sequence_is_original_signature():
    # 움직임 코드는 원본 그대로: reader/out_raw 없이 duration 만으로 동작, 반환 float
    issued = []
    elapsed = drive_sequence(
        duration_s=0.05, issue_drive=lambda: issued.append("d"),
        issue_stop=lambda: issued.append("s"),
        period_s=0.0, sleep=lambda s: None, now=_make_clock(dt=0.01),
    )
    assert elapsed > 0 and "s" in issued and "d" in issued


def test_move_bytes_matches_control():
    # 로깅에 박는 cmd_bytes 가 실배포 control 과 동일한지
    assert "67" in _move_bytes("forward")            # 전진 drive byte
    assert "118" in _move_bytes("rotate_ccw")        # 회전 byte


def test_write_raw_motion_schema(tmp_path, monkeypatch):
    import eval.eval_motion as em
    monkeypatch.setattr(em, "RAW_DIR", str(tmp_path))
    rows = [{"t_s": 0.0, "gyro_x": 0.1, "rel_yaw_deg": 1.0, "state": "ROTATE",
             "cmd": "ROTATE", "cmd_bytes": "127 127 127 127 118 127 127 127"},
            {"t_s": 0.02, "rel_yaw_deg": 2.0, "state": "ROTATE", "cmd": "ROTATE"}]
    path = em.write_raw_motion("20260608T000000", "rotate", rows)
    with open(path, newline="") as f:
        out = list(csv.reader(f))
    assert out[0] == RAW_HEADER                      # 헤더 (depth 컬럼 없음)
    di = {h: i for i, h in enumerate(RAW_HEADER)}
    assert out[1][di["gyro_x"]] == "0.1"
    assert out[2][di["gyro_x"]] == ""                # 누락 키 → 빈칸
    assert out[1][di["cmd_bytes"]].endswith("127")
