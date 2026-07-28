# 리프터 모션 캘리브레이션 평가 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FSM과 분리된 리프터 구동 프리미티브(회전·거리) 캘리브레이션 평가 도구를 만든다 — 회전은 IMU closed-loop로 멈추되 AprilTag 실측과 비교, 거리는 자(ruler) 실측으로 t(d) refit + held-out 검증.

**Architecture:** `eval/` 패키지에 순수 로직(적분·fit·헬퍼)과 하드웨어 의존(RealSense·CAN)을 분리한다. 순수 로직은 pytest로 TDD, 하드웨어 동작은 mock 주입으로 시퀀스 로직만 테스트하고 실기 동작은 수동 smoke. 기존 `depth_cam/calib/control.py`(CAN)·`depth_cam/calib/motion_models.py`(t(d))를 import 재사용하고, `RelYawEstimator` 적분 수식은 import 부작용 회피를 위해 `imu_yaw.py`에 이식한다.

**Tech Stack:** Python 3, conda env `pallet-pose`, pytest, scipy(`curve_fit`), numpy, pyrealsense2(런타임만, lazy import), Kvaser canlib(런타임만, control.py가 mock fallback 보유).

**작업 루트(이하 모든 경로의 기준):** `25y_automatic_lifter-master/25y_automatic_lifter-master/`

---

## File Structure

```
eval/
  __init__.py        # 빈 패키지 마커
  imu_yaw.py         # RelYawEstimator(순수 gyro.Y 적분+reset) + RelYawReader(RealSense 백그라운드 스레드)
  fit_fwd_model.py   # d=f(T) 적합 → t(d) 역산 파라미터(T0/T1/A) + R²/RMSE. CLI 진입점.
  eval_motion.py     # 순수 헬퍼(wrap/csv/error) + 동작 시퀀스(rotate/drive_calib/drive_eval) + CLI/메뉴/estop
  README.md          # 워크플로 순서, 실험 프로토콜, scope, 논문 서술 메모
  results/
    .gitkeep         # (런타임 산출: motion_eval.csv / calib_fwd.csv / raw/{ts}_rotate.csv)
tests/eval/
  test_fit_fwd_model.py
  test_imu_yaw.py
  test_eval_motion.py
```

**책임 경계:**
- `imu_yaw.py` — "지금 rel_yaw 몇 도?"만. 순수 적분(`RelYawEstimator`)은 합성 데이터로 테스트 가능. RealSense 폴링(`RelYawReader`)은 pyrealsense2를 lazy import해 의존 격리.
- `fit_fwd_model.py` — (T,d) → piecewise 적합 → 역산 파라미터. 완전 순수, CLI는 얇은 래퍼.
- `eval_motion.py` — CAN(control.py)·동작 시퀀스·로깅·메뉴. CAN/IMU는 인터페이스로 주입받아 시퀀스 로직 테스트 가능.

**기존 코드 재사용(import):**
- `depth_cam/calib/control.py`: `can_init, can_close, issue_command_forward, issue_command_backward, issue_command_rotate_in_place, issue_command_stop, is_mock`
- `depth_cam/calib/motion_models.py`: `fwd_sec_from_offset_piecewise` (scale=1,bias=0 기본이므로 `fwd_sec_from_offset_piecewise(d)` = t(d))
- 회전 방향 매핑(확인됨): `issue_command_rotate_in_place(+1)`=좌(CCW), `(-1)`=우(CW).

**import 경로 처리:** `depth_cam`은 `eval/`의 형제 디렉토리. 각 모듈 상단에서 작업 루트를 `sys.path`에 추가:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_cam.calib import control
```
단, `depth_cam/calib/__init__.py` 존재 확인 필요(없으면 Task 0에서 생성하지 말고 모듈 직접 경로 추가). control.py는 `from calib.control import ...` 식 상대 import를 안 쓰고 standalone이므로 `depth_cam/calib/control.py`를 `depth_cam.calib.control`로 import 가능한지 Task 0에서 검증한다.

---

### Task 0: 스캐폴드 + import 경로 검증

**Files:**
- Create: `eval/__init__.py`, `eval/results/.gitkeep`, `tests/eval/__init__.py`

- [ ] **Step 1: 디렉토리/빈 파일 생성**

```bash
cd 25y_automatic_lifter-master/25y_automatic_lifter-master
mkdir -p eval/results tests/eval
touch eval/__init__.py eval/results/.gitkeep tests/eval/__init__.py
```

- [ ] **Step 2: 기존 모듈 import 가능 여부 검증 (mock 환경)**

Run (작업 루트에서):
```bash
conda run -n pallet-pose python -c "
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from depth_cam.calib import control, motion_models
print('control mock?', control.is_mock())
print('t(0.5)=', motion_models.fwd_sec_from_offset_piecewise(0.5))
"
```
Expected: `control mock? True` (Kvaser DLL 없는 개발머신) + `t(0.5)= <float>` 출력.
만약 `ModuleNotFoundError: depth_cam` → `depth_cam/__init__.py` 부재. 그 경우:
```bash
ls depth_cam/__init__.py depth_cam/calib/__init__.py
```
없는 파일만 `touch`로 생성(빈 파일). 다시 Step 2 실행해 통과 확인.

- [ ] **Step 3: Commit**

```bash
git add eval/__init__.py eval/results/.gitkeep tests/eval/__init__.py
git commit -m "chore(eval): scaffold lifter motion eval package"
```

---

### Task 1: fit_fwd_model.py — d=f(T) 적합 + t(d) 역산 (★보정3·5)

**Files:**
- Create: `eval/fit_fwd_model.py`
- Test: `tests/eval/test_fit_fwd_model.py`

**핵심 수학 (스펙 ★보정5 — d를 노이즈 변수로 두고 d=f(T) 적합):**
```
vmax = a·t1,  d_acc = 0.5·a·t1²
d(T) = 0                          (T ≤ t0)
       0.5·a·(T−t0)²              (t0 < T ≤ t0+t1, 가속)
       d_acc + vmax·(T−t0−t1)     (T > t0+t1, 정속)
```

- [ ] **Step 1: 실패 테스트 작성 — 합성 파라미터 복원**

`tests/eval/test_fit_fwd_model.py`:
```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.fit_fwd_model import d_of_T, fit_fwd, t_of_d


def test_d_of_T_piecewise_shape():
    # t0=0, t1=4, a=0.075 → vmax=0.3, d_acc=0.6
    assert d_of_T(0.0, t0=0.0, t1=4.0, a=0.075) == 0.0          # 정지
    assert abs(d_of_T(2.0, t0=0.0, t1=4.0, a=0.075) - 0.15) < 1e-9   # 가속: 0.5*0.075*4
    assert abs(d_of_T(4.0, t0=0.0, t1=4.0, a=0.075) - 0.6) < 1e-9    # d_acc
    assert abs(d_of_T(6.0, t0=0.0, t1=4.0, a=0.075) - (0.6 + 0.3*2)) < 1e-9  # 정속


def test_fit_recovers_params():
    true = dict(t0=-0.02, t1=4.2, a=0.072)
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0])   # 가속~정속 분포
    d = np.array([d_of_T(t, **true) for t in T])
    rng = np.random.default_rng(0)
    d_noisy = d + rng.normal(0, 0.002, size=d.shape)           # mm급 노이즈
    res = fit_fwd(T, d_noisy)
    assert abs(res["a"] - true["a"]) < 0.01
    assert abs(res["t1"] - true["t1"]) < 0.3
    assert res["r2"] > 0.99


def test_t_of_d_is_inverse():
    # t(d)는 d(T)의 역함수여야 한다 (정속 구간 점)
    p = dict(t0=-0.02, t1=4.2, a=0.072)
    d = 1.5
    t = t_of_d(d, **p)
    assert abs(d_of_T(t, **p) - d) < 1e-6
```

- [ ] **Step 2: 실패 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_fit_fwd_model.py -v`
Expected: FAIL — `ModuleNotFoundError` 또는 `ImportError: cannot import name 'd_of_T'`.

- [ ] **Step 3: 구현**

`eval/fit_fwd_model.py`:
```python
"""전진 t(d) piecewise 모델 refit.

스펙 ★보정3: 논문 식(13-14) 형태 고정.
스펙 ★보정5: d를 노이즈 변수로 두고 d=f(T) 적합 후 t(d)로 역산.

사용:
  python eval/fit_fwd_model.py eval/results/calib_fwd.csv
  python eval/fit_fwd_model.py eval/results/calib_fwd.csv --dir fwd
"""
from __future__ import annotations
import argparse
import csv
import math
import numpy as np
from scipy.optimize import curve_fit


def d_of_T(T, t0, t1, a):
    """명령시간 T(s) → 이동거리 d(m). 가속→정속 piecewise."""
    T = np.asarray(T, dtype=float)
    a = max(a, 1e-9)
    d_acc = 0.5 * a * t1 * t1
    vmax = a * t1
    tau = T - t0
    d = np.where(
        tau <= 0.0, 0.0,
        np.where(tau <= t1, 0.5 * a * np.square(np.clip(tau, 0.0, None)),
                 d_acc + vmax * (tau - t1)),
    )
    return d if d.ndim else float(d)


def t_of_d(d, t0, t1, a):
    """이동거리 d(m) → 명령시간 t(s). d_of_T의 해석적 역함수."""
    a = max(a, 1e-9)
    d_acc = 0.5 * a * t1 * t1
    vmax = a * t1
    if d <= d_acc:
        return t0 + math.sqrt(max(0.0, 2.0 * d / a))
    return t0 + t1 + (d - d_acc) / vmax


def fit_fwd(T, d):
    """(T, d) 측정점에 d=f(T) 적합. 반환: {t0,t1,a,vmax,d_acc,r2,rmse,n}."""
    T = np.asarray(T, dtype=float)
    d = np.asarray(d, dtype=float)
    p0 = [0.0, 4.0, 0.075]                       # t0, t1, a 초기추정
    bounds = ([-2.0, 0.1, 1e-3], [2.0, 30.0, 5.0])
    popt, _ = curve_fit(d_of_T, T, d, p0=p0, bounds=bounds, maxfev=20000)
    t0, t1, a = (float(x) for x in popt)
    pred = d_of_T(T, t0, t1, a)
    ss_res = float(np.sum((d - pred) ** 2))
    ss_tot = float(np.sum((d - np.mean(d)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(d)))
    return {"t0": t0, "t1": t1, "a": a, "vmax": a * t1,
            "d_acc": 0.5 * a * t1 * t1, "r2": r2, "rmse": rmse, "n": len(d)}


def _load_calib(path, direction=None):
    T, d = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if direction and row.get("direction") != direction:
                continue
            T.append(float(row["T_sec"]))
            d.append(float(row["d_measured"]))
    return T, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="calib_fwd.csv 경로")
    ap.add_argument("--dir", default=None, help="fwd|back (생략 시 전체)")
    args = ap.parse_args()
    T, d = _load_calib(args.csv, args.dir)
    if len(T) < 3:
        raise SystemExit(f"적합에 최소 3점 필요 (현재 {len(T)}점). calib run 더 수집하세요.")
    r = fit_fwd(T, d)
    print(f"[fit dir={args.dir or 'all'}] n={r['n']}  R²={r['r2']:.4f}  RMSE={r['rmse']*1000:.1f}mm")
    print(f"  FWD_T0 = {r['t0']:.4f}")
    print(f"  FWD_T1 = {r['t1']:.4f}")
    print(f"  FWD_A  = {r['a']:.6f}   (vmax={r['vmax']:.4f} m/s, d_acc={r['d_acc']:.4f} m)")
    print("→ 위 값을 depth_cam/calib/config.py 의 FWD_T0/FWD_T1/FWD_A 에 반영 후 drive --eval 실행 (워크플로 c→d)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_fit_fwd_model.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/fit_fwd_model.py tests/eval/test_fit_fwd_model.py
git commit -m "feat(eval): add fit_fwd_model (d=f(T) fit + t(d) inverse, R2/RMSE)"
```

---

### Task 2: imu_yaw.py — RelYawEstimator (순수 적분 + reset)

**Files:**
- Create: `eval/imu_yaw.py`
- Test: `tests/eval/test_imu_yaw.py`

`RelYawEstimator`는 `depth_cam/main_rec.py:84` 로직과 **동일 수식**을 이식하되, 회전 평가용 `reset()`(ref 재스냅)을 추가한다. import 부작용(pyrealsense2 의존) 회피를 위해 main_rec를 import하지 않고 이식한다.

- [ ] **Step 1: 실패 테스트 작성**

`tests/eval/test_imu_yaw.py`:
```python
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.imu_yaw import RelYawEstimator


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
```

- [ ] **Step 2: 실패 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_imu_yaw.py -v`
Expected: FAIL — `ImportError: cannot import name 'RelYawEstimator'`.

- [ ] **Step 3: 구현 (RelYawEstimator만 — Reader는 Task 3)**

`eval/imu_yaw.py`:
```python
"""RealSense IMU 기반 rel_yaw 추정 (gyro.Y 적분).

RelYawEstimator : 순수 적분 로직 (depth_cam/main_rec.py RelYawEstimator 이식 + reset 추가).
RelYawReader    : RealSense 파이프라인을 백그라운드 스레드로 폴링 (Task 3).
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_imu_yaw.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/imu_yaw.py tests/eval/test_imu_yaw.py
git commit -m "feat(eval): add RelYawEstimator (gyro.Y integration + reset)"
```

---

### Task 3: imu_yaw.py — RelYawReader (RealSense 백그라운드 스레드)

**Files:**
- Modify: `eval/imu_yaw.py` (append `RelYawReader`)
- Test: `tests/eval/test_imu_yaw.py` (append)

RealSense 파이프라인 폴링을 백그라운드 스레드로 돌려 `reader.rel_yaw`로 최신값 노출. pyrealsense2는 lazy import(없으면 `available()=False`). 테스트는 프레임 소스를 주입해 스레드 없이 1프레임 처리만 검증.

- [ ] **Step 1: 실패 테스트 추가**

`tests/eval/test_imu_yaw.py` 끝에 추가:
```python
from eval.imu_yaw import RelYawReader


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
```

- [ ] **Step 2: 실패 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_imu_yaw.py::test_reader_processes_injected_frame -v`
Expected: FAIL — `ImportError: cannot import name 'RelYawReader'`.

- [ ] **Step 3: RelYawReader 구현 (imu_yaw.py 끝에 추가)**

```python
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

    def _process_motion(self, accel, gyro, ts_ms: float):
        rel = self._est.update_from_frames(accel, gyro, ts_ms)
        with self._lock:
            self._rel = rel

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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_imu_yaw.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/imu_yaw.py tests/eval/test_imu_yaw.py
git commit -m "feat(eval): add RelYawReader (RealSense background polling, lazy import)"
```

---

### Task 4: eval_motion.py — 순수 헬퍼 (wrap / CSV / error)

**Files:**
- Create: `eval/eval_motion.py`
- Test: `tests/eval/test_eval_motion.py`

먼저 하드웨어 무관 순수 함수만 작성: `wrap_to_180`, CSV append 2종, error 계산.

- [ ] **Step 1: 실패 테스트 작성**

`tests/eval/test_eval_motion.py`:
```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_eval_motion.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: 구현 (순수 헬퍼 + 상수만; 동작 시퀀스는 Task 5)**

`eval/eval_motion.py`:
```python
"""리프터 모션 캘리브레이션 평가 — 회전(IMU closed-loop) / 거리(calib·eval).

워크플로(거리, ★보정1): drive_calib 수집 → fit_fwd_model refit → config 갱신 → drive_eval.
스펙: docs/superpowers/specs/2026-06-04-lifter-motion-eval-design.md
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EVAL_CSV = os.path.join(RESULTS_DIR, "motion_eval.csv")
CALIB_CSV = os.path.join(RESULTS_DIR, "calib_fwd.csv")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

EVAL_HEADER = ["timestamp", "kind", "target", "direction", "power",
               "imu_stop_deg", "imu_settled_deg", "elapsed_s", "cmd_time_s",
               "measured", "reached", "error", "note"]
CALIB_HEADER = ["timestamp", "T_sec", "d_measured", "direction", "power"]


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _append(path, header, row_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row_list)


def append_eval_row(path, d: dict):
    """measured-target 으로 error 자동계산 후 1행 append."""
    try:
        error = float(d["measured"]) - float(d["target"])
    except (TypeError, ValueError):
        error = ""
    row = {**d, "timestamp": _ts(), "error": error}
    _append(path, EVAL_HEADER, [row.get(k, "") for k in EVAL_HEADER])


def append_calib_row(path, *, T_sec, d_measured, direction, power):
    _append(path, CALIB_HEADER, [_ts(), T_sec, d_measured, direction, power])


def write_raw_rotate(ts_label: str, samples):
    """회전 raw 시계열 (t, rel_yaw) 저장. samples: list[(t, rel_yaw)]."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{ts_label}_rotate.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "rel_yaw_deg"])
        w.writerows(samples)
    return path
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_eval_motion.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/eval_motion.py tests/eval/test_eval_motion.py
git commit -m "feat(eval): add eval_motion pure helpers (wrap, csv append, error calc)"
```

---

### Task 5: eval_motion.py — 동작 시퀀스 (rotate / drive_calib / drive_eval)

**Files:**
- Modify: `eval/eval_motion.py` (append 시퀀스 함수)
- Test: `tests/eval/test_eval_motion.py` (append)

CAN(`issue_*`)과 IMU(`reader`)를 인자로 주입받아 시퀀스 로직(종료판정·타임아웃·stop후 raw연장)을 mock으로 테스트한다. 실기 동작은 Task 6 smoke.

회전 종료(스펙): `|rel_yaw - ref| >= target` → stop → **stop 후 settle_s초 raw 계속 기록**(★보정2) → `imu_settled`.

- [ ] **Step 1: 실패 테스트 작성**

`tests/eval/test_eval_motion.py` 끝에 추가:
```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_eval_motion.py -k rotate_sequence -v`
Expected: FAIL — `ImportError: cannot import name 'rotate_sequence'`.

- [ ] **Step 3: 구현 (eval_motion.py 끝에 추가)**

```python
from dataclasses import dataclass


@dataclass
class RotateResult:
    target_deg: float
    turn_dir: int
    imu_stop_deg: float
    imu_settled_deg: float
    elapsed_s: float
    reached: bool
    raw: list          # [(t, rel_yaw)]


def rotate_sequence(*, reader, target_deg, turn_dir, issue_rotate, issue_stop,
                    period_s=0.02, settle_s=1.5, timeout_s=20.0,
                    sleep=time.sleep, now=time.monotonic):
    """회전 IMU closed-loop. stop 후 settle_s 동안 raw 연장 기록(★보정2).

    turn_dir: +1=좌(CCW), -1=우(CW). reader.rel_yaw 는 reset 기준 상대각.
    """
    reader.reset()
    t0 = now()
    raw = []
    imu_stop = 0.0
    reached = False
    while True:
        issue_rotate(turn_dir)
        delta = abs(reader.rel_yaw)
        t = now() - t0
        raw.append((round(t, 4), round(reader.rel_yaw, 4)))
        if delta >= target_deg:
            imu_stop = delta
            reached = True
            break
        if t > timeout_s:
            imu_stop = delta
            reached = False
            break
        sleep(period_s)
    issue_stop()
    # ★보정2: stop 후 settle_s 동안 over-rotate 정착 기록
    t_settle_end = now() + settle_s
    while now() < t_settle_end:
        raw.append((round(now() - t0, 4), round(reader.rel_yaw, 4)))
        sleep(period_s)
    imu_settled = abs(reader.rel_yaw)
    return RotateResult(target_deg, turn_dir, imu_stop, imu_settled,
                        now() - t0, reached, raw)


def drive_sequence(*, duration_s, issue_drive, issue_stop,
                   period_s=0.02, sleep=time.sleep, now=time.monotonic):
    """개루프 전진/후진 duration_s초 (raw 미기록·★보정3). 반환: 실제 경과초."""
    t0 = now()
    while now() - t0 < duration_s:
        issue_drive()
        sleep(period_s)
    issue_stop()
    return now() - t0
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `conda run -n pallet-pose pytest tests/eval/test_eval_motion.py -v`
Expected: 5 PASS (Task4의 3 + rotate 2).

- [ ] **Step 5: Commit**

```bash
git add eval/eval_motion.py tests/eval/test_eval_motion.py
git commit -m "feat(eval): add rotate/drive sequences (IMU closed-loop + settle logging)"
```

---

### Task 6: eval_motion.py — CLI + 대화형 메뉴 + estop 연결

**Files:**
- Modify: `eval/eval_motion.py` (CAN/IMU 연결, `run_rotate`/`run_drive_calib`/`run_drive_eval`, argparse, 메뉴, estop)

순수 시퀀스(Task5)를 실제 CAN/IMU에 연결. estop은 별도 스레드 키 감시. 이 Task는 하드웨어 의존이라 단위테스트 대신 import/argparse smoke + 수동 검증.

- [ ] **Step 1: 구현 (eval_motion.py 끝에 추가)**

```python
import threading

from depth_cam.calib import control
from depth_cam.calib import motion_models
from eval.imu_yaw import RelYawReader

# 배치 고정 power (확인됨): 전진 델타60(byte67), 회전 byte118
POWER_FWD = 60
POWER_ROT = 118

_ESTOP = threading.Event()


def _estop_watch():
    """[space] 입력 시 즉시 STOP. (간단 stdin 감시 — 현장 콘솔용)"""
    try:
        import keyboard  # 설치돼 있으면 사용
        keyboard.add_hotkey("space", lambda: (_ESTOP.set(), control.issue_command_stop()))
        keyboard.wait()
    except Exception:
        pass  # keyboard 미설치 환경: estop은 Ctrl-C 로 대체


def _dir_to_turn(direction: str) -> int:
    return +1 if direction == "ccw" else -1   # ccw=좌=+1, cw=우=-1


def run_rotate(reader, target_deg, direction, timeout_s, do_measure=True):
    if reader is None or not reader.available():
        print("[SKIP] RealSense 미가용 — 회전 평가 불가"); return
    input(f"▶ 회전 {target_deg}° {direction} 시작 — 엔터: ")
    res = rotate_sequence(
        reader=reader, target_deg=target_deg, turn_dir=_dir_to_turn(direction),
        issue_rotate=control.issue_command_rotate_in_place,
        issue_stop=control.issue_command_stop, timeout_s=timeout_s)
    label = time.strftime("%Y%m%dT%H%M%S")
    raw_path = write_raw_rotate(label, res.raw)
    print(f"  imu@stop={res.imu_stop_deg:.2f}° imu@settle={res.imu_settled_deg:.2f}° "
          f"t={res.elapsed_s:.2f}s reached={res.reached}  raw→{raw_path}")
    measured = ""
    if do_measure:
        measured = float(input("  AprilTag 실측각(°)? "))
    append_eval_row(EVAL_CSV, {
        "kind": "rotate", "target": target_deg, "direction": direction, "power": POWER_ROT,
        "imu_stop_deg": round(res.imu_stop_deg, 3), "imu_settled_deg": round(res.imu_settled_deg, 3),
        "elapsed_s": round(res.elapsed_s, 3), "cmd_time_s": "",
        "measured": measured, "reached": res.reached, "note": ""})


def run_drive_calib(T_sec, direction):
    issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward
    input(f"▶ calib 전진 {T_sec}s {direction} 시작 — 엔터: ")
    elapsed = drive_sequence(duration_s=T_sec, issue_drive=issue, issue_stop=control.issue_command_stop)
    d = float(input(f"  자 실측거리(m)? (T={elapsed:.2f}s) "))
    append_calib_row(CALIB_CSV, T_sec=round(elapsed, 3), d_measured=d, direction=direction, power=POWER_FWD)
    print(f"  → calib_fwd.csv 기록. refit: python eval/fit_fwd_model.py {CALIB_CSV} --dir {direction}")


def run_drive_eval(D_m, direction, do_measure=True):
    t_cmd = motion_models.fwd_sec_from_offset_piecewise(D_m)
    issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward
    input(f"▶ eval 전진 {D_m}m → {t_cmd:.2f}s {direction} (refit·config 갱신 후인지 확인!) — 엔터: ")
    elapsed = drive_sequence(duration_s=t_cmd, issue_drive=issue, issue_stop=control.issue_command_stop)
    measured = ""
    if do_measure:
        measured = float(input("  자 실측거리(m)? "))
    append_eval_row(EVAL_CSV, {
        "kind": "drive_eval", "target": D_m, "direction": direction, "power": POWER_FWD,
        "imu_stop_deg": "", "imu_settled_deg": "", "elapsed_s": round(elapsed, 3),
        "cmd_time_s": round(t_cmd, 3), "measured": measured, "reached": True, "note": ""})


def _menu(reader, args):
    target_deg, rot_dir, drv_dir = 90.0, "ccw", "fwd"
    while True:
        print(f"\n목표: 회전 {target_deg}°/{rot_dir}  전진 {drv_dir}")
        c = input("[r]회전 [c]calib(T초) [f]eval(거리) [t]각도 [d]방향 [q]종료 > ").strip()
        if c == "q":
            break
        elif c == "r":
            run_rotate(reader, target_deg, rot_dir, args.timeout)
        elif c == "c":
            T = float(input("  calib 전진 시간 T(s)? "))
            run_drive_calib(T, drv_dir)
        elif c == "f":
            D = float(input("  eval 거리 D(m)? "))
            run_drive_eval(D, drv_dir)
        elif c == "t":
            target_deg = float(input("  회전 목표각(°)? "))
        elif c == "d":
            rot_dir = "cw" if rot_dir == "ccw" else "ccw"
            drv_dir = "back" if drv_dir == "fwd" else "fwd"


def main():
    ap = argparse.ArgumentParser(description="리프터 모션 캘리브레이션 평가")
    ap.add_argument("--rotate", type=float, help="회전 목표각(°) 1회 실행")
    ap.add_argument("--drive", type=float, help="eval 거리(m) 1회 실행")
    ap.add_argument("--calib", type=float, help="calib 전진 시간(s) 1회 실행")
    ap.add_argument("--dir", default="fwd", help="cw|ccw (회전) / fwd|back (전진)")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--no-measure", action="store_true", help="실측 입력 생략(자동값만 기록)")
    args = ap.parse_args()

    control.can_init(channel=args.channel)
    print(f"[CAN] init (mock={control.is_mock()})")
    reader = None
    if args.rotate is not None or (args.drive is None and args.calib is None):
        reader = RelYawReader()
        if reader.available():
            reader.start(); print("[IMU] RealSense OK")
        else:
            print("[IMU] RealSense 미가용 — 거리 전용")
    threading.Thread(target=_estop_watch, daemon=True).start()
    try:
        if args.rotate is not None:
            run_rotate(reader, args.rotate, args.dir, args.timeout, not args.no_measure)
        elif args.calib is not None:
            run_drive_calib(args.calib, args.dir)
        elif args.drive is not None:
            run_drive_eval(args.drive, args.dir, not args.no_measure)
        else:
            _menu(reader, args)
    finally:
        control.issue_command_stop()
        if reader is not None and reader.available():
            reader.stop()
        control.can_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: import/argparse smoke (mock 환경, 하드웨어 없이 종료)**

Run (작업 루트):
```bash
conda run -n pallet-pose python -c "import eval.eval_motion as m; print('import OK'); m.main()" --help 2>&1 | head -5
```
Expected: `import OK` 후 argparse usage 출력(또는 `--help`로 정상 종료). `ImportError` 없을 것.

- [ ] **Step 3: calib 1회 mock 동작 확인 (CAN mock, 실측 자동입력)**

Run:
```bash
cd 25y_automatic_lifter-master/25y_automatic_lifter-master
echo -e "\n1.23" | conda run -n pallet-pose python eval/eval_motion.py --calib 2.0 --dir fwd
```
Expected: `[CAN] init (mock=True)` → 엔터 대기 통과 → calib_fwd.csv 기록 메시지. (mock이라 실제 전송 없음)
확인: `cat eval/results/calib_fwd.csv` → 헤더 + `..,2.0,1.23,fwd,60` 행.

- [ ] **Step 4: 전체 테스트 재실행 (회귀 없음 확인)**

Run: `conda run -n pallet-pose pytest tests/eval/ -v`
Expected: 모든 테스트 PASS (10).

- [ ] **Step 5: Commit**

```bash
git add eval/eval_motion.py
git commit -m "feat(eval): wire CAN/IMU, CLI + interactive menu + estop"
```

---

### Task 7: README + scope/논문 메모

**Files:**
- Create: `eval/README.md`

- [ ] **Step 1: README 작성**

`eval/README.md`:
```markdown
# 리프터 모션 캘리브레이션 평가 (eval/)

FSM과 분리된 구동 프리미티브 정확도 검증 도구.
설계: `docs/superpowers/specs/2026-06-04-lifter-motion-eval-design.md`

## 워크플로 (거리는 순서 의존 — ★중요)
1. `python eval/eval_motion.py --calib <T> --dir fwd` 여러 T(가속~정속) 반복 → `results/calib_fwd.csv`
2. `python eval/fit_fwd_model.py results/calib_fwd.csv --dir fwd` → 새 FWD_T0/T1/A + R²/RMSE
3. 출력값을 `depth_cam/calib/config.py` 의 FWD_T0/FWD_T1/FWD_A 에 반영
4. `python eval/eval_motion.py --drive <D> --dir fwd` (held-out) — ★ 3 완료 후에만
> refit 전에 --drive 실행 금지 (옛 파라미터로 검증하는 사고).

## 회전 평가
`python eval/eval_motion.py --rotate 90 --dir ccw` (RealSense 필요)
- IMU closed-loop로 멈춤. stop 후 1~2초 raw 연장 기록 → over-rotate 정착 캡처.
- imu@stop / imu@settle / AprilTag 실측 3값 → 오차 부분 분해.

## 실험 프로토콜 (통계 최소선)
- 거리 calib: 짧은 T(가속 d≲0.65m) 여러 + 긴 T(정속) 여러, 각 ≥10회, fwd/back
- 거리 eval: 각 D ≥10회 (calib과 disjoint run), fwd/back
- 회전: 90° CW·CCW 각 ≥10–15회, 고정 power(byte118) 주 결과

## Scope (이 도구로 대체 금지)
- 프리미티브(회전·거리) 정확도 검증 **전용**.
- closed-loop 정렬(직교 시퀀스 전후 d_lateral, 삽입 성공률)은 **별도 실험**.
- 단일 고정 구동 강도(전진 델타60/회전 byte118)에서만 t(d) 적합·검증.

## 논문 서술 메모
- 회전 error = 자이로 적분오차 + ECU 관성 합산. imu@settle로 부분 분해 가능.
- AprilTag 측정은 Olson(2011) 인용 — overhead + 상단 태그(in-plane yaw) + 카메라 캘리브 확인.
- refit한 FWD_T0/T1/A 가 논문 t(d) 를 대체. calib fit품질(R²/RMSE) + eval held-out 오차 둘 다 보고.

## Scope-out (별도 챙김)
- AprilTag yaw 산출 도구(실제 GT)의 캘리브레이션 신뢰성.
```

- [ ] **Step 2: Commit**

```bash
git add eval/README.md
git commit -m "docs(eval): add README (workflow, protocol, scope, paper notes)"
```

---

## Self-Review

**1. Spec coverage:**
- 회전 IMU closed-loop(C안) → Task 2/3/5 (rotate_sequence + reader). ✓
- stop후 raw 연장 + imu_settled 분해(★보정2) → Task 5 rotate_sequence settle 루프 + Task 4 CSV. ✓
- 거리 calib refit(★보정1·3·5) → Task 1 fit_fwd_model + Task 6 run_drive_calib. ✓
- 거리 held-out eval + 워크플로 순서(★보정1) → Task 6 run_drive_eval + Task 7 README 경고. ✓
- power 컬럼(★보정2-스키마) → Task 4 EVAL_HEADER/CALIB_HEADER. ✓
- 거리 raw 드롭(★보정3) → Task 5 drive_sequence(raw 미기록) + Task 7. ✓
- 안전(estop/타임아웃/시작확인) → Task 5 timeout + Task 6 estop/input 프롬프트. ✓
- CLI+대화형(C안) → Task 6 main/_menu. ✓
- 후진 방향별 적합 → Task 1 `--dir`, Task 6 direction 분기. ✓
- scope/Olson/논문메모 → Task 7. ✓

**2. Placeholder scan:** 모든 코드 step에 실제 코드 포함. "적절한 에러처리" 류 없음. ✓

**3. Type consistency:**
- `RelYawEstimator.update_from_frames/reset` — Task2 정의, Task3 reader가 사용. ✓
- `EVAL_HEADER`/`append_eval_row`/`append_calib_row` — Task4 정의, Task6 사용. 일치. ✓
- `rotate_sequence`/`drive_sequence`/`RotateResult` — Task5 정의, Task6 사용. 인자명(reader/target_deg/turn_dir/issue_rotate/issue_stop) 일치. ✓
- `write_raw_rotate` — Task4 정의, Task6 사용. ✓
- 방향 매핑 `_dir_to_turn(ccw→+1, cw→-1)` ↔ control.issue_command_rotate_in_place(+1=좌). 일치. ✓

미해결 리스크(구현 중 확인): Task0 Step2에서 `depth_cam.calib` import 가능 여부 — 실패 시 `__init__.py` 생성으로 해결(Step에 명시). RealSense `wait_for_frames` 인자/모션프레임 API는 실기에서만 검증(mock 단위테스트는 `_process_motion` 우회).
