#!/usr/bin/env python3
"""Jetson FSM 실행 전 환경 점검 — 무엇이 빠졌는지 먼저 알려준다.
실행: python jetson_fsm_deploy/preflight.py   (repo root 또는 어디서나)
실제 main_rec 실행 전 '될지 안 될지'를 미리 확인하는 용도. 통과 못해도 진행은 가능."""
import importlib, os, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "✅", "⚠️ ", "❌"
rows = []

def check(name, fn):
    try:
        ok, msg = fn()
        rows.append((OK if ok else WARN, name, msg))
    except Exception as e:
        rows.append((FAIL, name, f"{type(e).__name__}: {e}"))

# --- 파이썬 패키지 ---
def _mod(m):
    def f():
        mod = importlib.import_module(m)
        return True, getattr(mod, "__version__", "ok")
    return f
for m in ["numpy", "cv2", "torch", "torchvision", "ultralytics", "pyrr"]:
    check(f"import {m}", _mod(m))

# --- CUDA ---
def _cuda():
    import torch
    if not torch.cuda.is_available():
        return False, "torch.cuda 사용 불가 (CPU로만 동작 → 매우 느림)"
    return True, f"{torch.cuda.get_device_name(0)}"
check("CUDA", _cuda)

# --- RealSense ---
def _rs():
    import pyrealsense2 as rs
    ctx = rs.context()
    n = ctx.query_devices().size()
    if n == 0:
        return False, "pyrealsense2 OK 이나 카메라 미연결 (USB3 확인)"
    return True, f"{n}대 연결됨"
check("RealSense (pyrealsense2 + 장치)", _rs)

# --- CAN (Kvaser canlib) ---
def _can():
    sys.path.insert(0, str(_REPO / "25y_automatic_lifter-master" / "25y_automatic_lifter-master" / "depth_cam"))
    from calib.control import is_mock, can_init
    can_init()
    if is_mock():
        return False, "MOCK (canlib 없음) → 리프터 실제 안 움직임. CAN_PORTING.md 참고"
    return True, "실제 CAN 송신 가능"
check("CAN 제어", _can)

# --- 모델 파일 ---
def _yolo():
    p = os.environ.get("MODEL_PATH_6D_YOLO",
                       str(_REPO / "pallet_jetson_deploy" / "models" / "pallet_pose_cropaug_v2.pt"))
    if not os.path.isfile(p):
        return False, f"없음: {p}"
    return True, f"{os.path.basename(p)} ({'engine' if p.endswith('.engine') else 'pt/onnx'})"
check("YOLO weights", _yolo)

def _dope():
    p = os.environ.get("MODEL_PATH_6D", str(_REPO / "challenge" / "model" / "challengenight.pth"))
    return (os.path.isfile(p)), (os.path.basename(p) if os.path.isfile(p) else f"없음(DOPE 쓸때만 필요): {p}")
check("DOPE weights", _dope)

# --- config 핵심값 (실삽입 안전) ---
def _cfg():
    sys.path.insert(0, str(_REPO / "25y_automatic_lifter-master" / "25y_automatic_lifter-master" / "depth_cam"))
    from calib import config as c
    notes = []
    notes.append(f"POSE_BACKEND={c.POSE_BACKEND}")
    notes.append(f"GATE_PROFILE={c.GATE_PROFILE}")
    notes.append(f"pallet={c.PALLET_WIDTH_M}x{c.PALLET_DEPTH_M}x{c.PALLET_HEIGHT_M}m")
    cam_set = any(c.CAM_TO_FORK_T) or any(c.CAM_TO_FORK_RPY_DEG)
    notes.append("CAM_TO_FORK=실측됨" if cam_set else "CAM_TO_FORK=0(미실측)")
    ok = (c.GATE_PROFILE == "real") and cam_set
    return ok, " | ".join(notes) + ("" if ok else "  ← 실삽입엔 GATE_PROFILE=real + extrinsic 실측 권장")
check("config (안전 설정)", _cfg)

# --- 출력 ---
print("\n" + "=" * 70)
print(" Jetson FSM Preflight")
print("=" * 70)
w = max(len(n) for _, n, _ in rows)
for sym, name, msg in rows:
    print(f" {sym} {name.ljust(w)}  {msg}")
print("=" * 70)
nfail = sum(1 for s, _, _ in rows if s == FAIL)
nwarn = sum(1 for s, _, _ in rows if s == WARN)
print(f" {OK} 통과 외 — {FAIL} {nfail}개 / {WARN} {nwarn}개. ❌는 실행 막힘, ⚠️는 기능 제한.")
