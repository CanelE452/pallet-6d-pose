# calib/control.py
# =============================================================================
# CAN 통신 제어부 (fsm.py와의 연동 전용)
#  - 근거: control_forklift_v2.py (DirectFrameForkliftController)
#  - Kvaser CANlib + Frame 사용
#  - movement/control 템플릿, ID/플래그, Heartbeat 규격을 동일하게 유지
#  - fsm.py 퍼블릭 API 시그니처(확정):
#       issue_command_forward_and_turn(turn_dir), issue_command_rotate_in_place(turn_dir),
#       issue_command_backward(), issue_command_forward(), issue_command_stop()
#       (+) issue_command_backward_and_turn(turn_dir)
# =============================================================================
from __future__ import annotations
from typing import Optional, Dict
from dataclasses import dataclass
import os
import threading
import time

# Kvaser CANlib — DLL (canlib32.dll) 부재 시 자동 mock fallback.
# Kvaser SDK 가 설치된 머신 (사용자 운영 환경) 에서는 정상 import.
# SDK 없는 개발 머신 (CI / 검증용) 에서는 mock 으로 떨어져 main_rec.py 의
# import 및 init 단계 모두 통과 (실제 CAN 송수신은 silent skip).
try:
    from canlib import canlib, Frame  # Kvaser CANlib
    _CANLIB_DLL_OK = True
except (ImportError, FileNotFoundError, OSError) as _e:
    print(f"[CAN] canlib DLL load failed ({type(_e).__name__}: {_e}) — using mock canlib.")
    _CANLIB_DLL_OK = False

    class _MockCanlibError(Exception):
        pass

    class _MockChannel:
        def setBusOutputControl(self, *a, **k): pass
        def setBusParams(self, *a, **k): pass
        def busOn(self): pass
        def busOff(self): pass
        def close(self): pass
        def write(self, frame): pass
        def read(self, *a, **k): raise _MockCanlibError("mock: no frame")

    class _MockBitrate:
        """Kvaser canlib.Bitrate 인터페이스 모킹 (canlib 미설치 환경에서 can_init 통과용)."""
        BITRATE_1M   = -1
        BITRATE_500K = -2
        BITRATE_250K = -3
        BITRATE_125K = -4
        BITRATE_100K = -5
        BITRATE_62K  = -6
        BITRATE_50K  = -7

    class _MockCanlib:
        canERR_NOTFOUND = -3
        canOPEN_ACCEPT_VIRTUAL = 0
        canBITRATE_500K = -2
        Bitrate = _MockBitrate
        Driver = type("Driver", (), {"NORMAL": 4})()
        canERR_NOMSG = _MockCanlibError

        @staticmethod
        def openChannel(channel, flags=0):
            return _MockChannel()

        @staticmethod
        def getNumberOfChannels():
            return 0

    canlib = _MockCanlib()

    class Frame:
        def __init__(self, id_=0, data=b"", dlc=0, flags=0):
            self.id = id_
            self.data = data
            self.dlc = dlc
            self.flags = flags

# Mock send 의 [MOCK SEND] 로그 toggle. 기본 OFF (조용히 동작).
# 디버깅 시 PowerShell: $env:CAN_MOCK_VERBOSE=1 / bash: CAN_MOCK_VERBOSE=1
_MOCK_VERBOSE = os.environ.get("CAN_MOCK_VERBOSE", "0") == "1"


def is_mock() -> bool:
    """현재 CAN 통신이 mock (canlib DLL 없음) 인지 반환."""
    return not _CANLIB_DLL_OK


__all__ = [
    "can_init", "can_close", "start_heartbeat", "stop_heartbeat", "send_heartbeat",
    "issue_command_forward", "issue_command_backward",
    "issue_command_forward_and_turn", "issue_command_backward_and_turn",
    "issue_command_rotate_in_place", "issue_command_stop",
    "is_mock",
]

# =============================================================================
# 공용 상수/플래그
# =============================================================================
class MessageFlag:
    STD = 0x0000  # 표준 11-bit ID
    EXT = 0x0004  # 확장 29-bit ID

# ---- 설정 (필요시 외부에서 수정 가능) ----
CAN_CHANNEL = 0
CAN_BITRATE = 500_000
USE_EXTENDED_IDS = False  # False=표준11bit, True=확장29bit

# IDs (동일)
CAN_MOVEMENT_ID = 0x01E3
CAN_CONTROL_ID  = 0x02E3

# Heartbeat
HEARTBEAT_ID = 0x764
HEARTBEAT_PERIOD = 0.200
HEARTBEAT_DATA = [0x00]

# =============================================================================
# 조이스틱 강도/템플릿 (v2와 동일 계산)
# =============================================================================
AN_NEUTRAL = 127
JOYSTICK_FORWARD = 60
JOYSTICK_BACKWARD = 60
JOYSTICK_LEFT = 60
JOYSTICK_RIGHT = 60
JOYSTICK_ROTATE_CCW = 30
JOYSTICK_ROTATE_CW = 30

AN_FORWARD = min(255, AN_NEUTRAL - JOYSTICK_FORWARD)   # 67
AN_BACKWARD = max(0,   AN_NEUTRAL + JOYSTICK_BACKWARD) # 187
AN_LEFT =     min(255, AN_NEUTRAL + JOYSTICK_LEFT)     # 187
AN_RIGHT =    max(0,   AN_NEUTRAL - JOYSTICK_RIGHT)    # 67

# 회전 힘은 v2 기준으로 "118"을 명시 고정 (바이트 맵핑: CCW→data[4], CW→data[5])
AN_ROTATE_CCW = 118
AN_ROTATE_CW  = 118

AN_N = AN_NEUTRAL

# Byte 의미(예시):
# [0]??, [1]=steer(left/right), [2]=drive(fwd/back), [3]??, [4]=rot_CCW, [5]=rot_CW, [6]??, [7]??
MOVEMENT_TEMPLATES: Dict[str, list[int]] = {
    "stop":            [AN_N, AN_N, AN_N,        AN_N, AN_N, AN_N,         AN_N, AN_N],
    "forward":         [AN_N, AN_N, AN_FORWARD,  AN_N, AN_N, AN_N,         AN_N, AN_N],
    "backward":        [AN_N, AN_N, AN_BACKWARD, AN_N, AN_N, AN_N,         AN_N, AN_N],
    "turn_left":       [AN_N, AN_LEFT, AN_N,     AN_N, AN_N, AN_N,         AN_N, AN_N],
    "turn_right":      [AN_N, AN_RIGHT,AN_N,     AN_N, AN_N, AN_N,         AN_N, AN_N],

    # 제자리 회전: CCW는 data[4]=118, CW는 data[5]=118
    "rotate_ccw":      [AN_N, AN_N,   AN_N,      AN_N, AN_ROTATE_CCW,      AN_N,       AN_N, AN_N],
    "rotate_cw":       [AN_N, AN_N,   AN_N,      AN_N, AN_N,               AN_ROTATE_CW,AN_N, AN_N],

    # 전/후진 + 조향
    "forward_left":    [AN_N, AN_LEFT, AN_FORWARD,  AN_N, AN_N, AN_N,      AN_N, AN_N],
    "forward_right":   [AN_N, AN_RIGHT,AN_FORWARD,  AN_N, AN_N, AN_N,      AN_N, AN_N],
    "backward_left":   [AN_N, AN_LEFT,  AN_BACKWARD, AN_N, AN_N, AN_N,     AN_N, AN_N],
    "backward_right":  [AN_N, AN_RIGHT, AN_BACKWARD, AN_N, AN_N, AN_N,     AN_N, AN_N],
}

# control 템플릿 (5번째 바이트 sync 카운터는 runtime 갱신)
CONTROL_TEMPLATES: Dict[str, list[int]] = {
    "driving_mode":    [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "rotate_ccw_ctrl": [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "rotate_cw_ctrl":  [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "emergency":       [0x80, 0x00, 0x00, 0x00, 0x01, 0x40, 0x69, 0x93],  # sync 미적용
}

# =============================================================================
# 내부 상태/도우미
# =============================================================================
@dataclass
class _BusCtx:
    ch: Optional[canlib.Channel] = None
    tx_lock: threading.Lock = threading.Lock()
    sync_counter: int = 0x0A
    is_extended: bool = False

_CTX = _BusCtx()
_HB_THREAD = None
_HB_STOP = threading.Event()

def _flags():
    return MessageFlag.EXT if _CTX.is_extended else MessageFlag.STD

def _next_sync() -> int:
    _CTX.sync_counter = (_CTX.sync_counter + 1) & 0x0F
    return _CTX.sync_counter

def _write(frame: Frame):
    if _CTX.ch is None:
        if _MOCK_VERBOSE:
            print(f"[MOCK SEND] id=0x{frame.id:03X}, data={[hex(b) for b in frame.data]}, flags={'EXT' if _CTX.is_extended else 'STD'}")
        return
    with _CTX.tx_lock:
        _CTX.ch.write(frame)

def _mk_control(ctrl_type: str) -> Frame:
    data = CONTROL_TEMPLATES.get(ctrl_type, CONTROL_TEMPLATES["driving_mode"]).copy()
    if ctrl_type != "emergency":
        data[4] = _next_sync()
    return Frame(id_=CAN_CONTROL_ID, data=data, flags=_flags())

def _mk_movement(name: str) -> Frame:
    data = MOVEMENT_TEMPLATES.get(name, MOVEMENT_TEMPLATES["stop"])
    return Frame(id_=CAN_MOVEMENT_ID, data=data, flags=_flags())

def _mk_heartbeat() -> Frame:
    return Frame(id_=HEARTBEAT_ID, data=HEARTBEAT_DATA, flags=_flags())

# =============================================================================
# 초기화/종료/하트비트
# =============================================================================
def can_init(channel: int = CAN_CHANNEL, bitrate: int = CAN_BITRATE, is_extended_id: bool = USE_EXTENDED_IDS) -> bool:
    """
    Kvaser CAN 초기화 (성공 시 True)
    - bus on 이후 안정화 시퀀스(드라이빙 모드/정지/하트비트) 송신
    - Heartbeat 스레드를 자동 기동 (start_heartbeat)
    """
    _CTX.is_extended = bool(is_extended_id)
    try:
        _CTX.ch = canlib.openChannel(channel)
        br_map = {
            1_000_000: canlib.Bitrate.BITRATE_1M,
            500_000:   canlib.Bitrate.BITRATE_500K,
            250_000:   canlib.Bitrate.BITRATE_250K,
            125_000:   canlib.Bitrate.BITRATE_125K,
        }
        br = br_map.get(bitrate, canlib.Bitrate.BITRATE_500K)
        _CTX.ch.setBusParams(br)
        _CTX.ch.busOn()
        # 시작 안정화: driving_mode + stop + heartbeat (5회, v2의 버스트 요약)
        for _ in range(5):
            _write(_mk_control("driving_mode"))
            _write(_mk_movement("stop"))
            _write(_mk_heartbeat())
            time.sleep(0.005)
        # Heartbeat 자동 시작
        start_heartbeat()
        return True
    except Exception as e:
        print(f"[CAN INIT ERROR] {e}")
        _CTX.ch = None
        return False

def can_close():
    """STOP 송신 후 버스 해제"""
    try:
        stop_heartbeat()
    except Exception:
        pass
    try:
        # 안전 정지 프레임 1회
        _write(_mk_movement("stop"))
    except Exception:
        pass
    if _CTX.ch is not None:
        try:
            _CTX.ch.busOff()
            _CTX.ch.close()
        except Exception:
            pass
        _CTX.ch = None
    print("[CAN CLOSED]")

def send_heartbeat():
    """Heartbeat 1회 송신"""
    _write(_mk_heartbeat())

def _hb_loop():
    while not _HB_STOP.is_set():
        try:
            _write(_mk_heartbeat())
        except Exception:
            pass
        _HB_STOP.wait(HEARTBEAT_PERIOD)

def start_heartbeat():
    """200ms 주기 Heartbeat 시작"""
    global _HB_THREAD
    if _HB_THREAD is not None and _HB_THREAD.is_alive():
        return
    _HB_STOP.clear()
    _HB_THREAD = threading.Thread(target=_hb_loop, name="CANHeartbeat", daemon=True)
    _HB_THREAD.start()

def stop_heartbeat():
    global _HB_THREAD
    if _HB_THREAD is None:
        return
    _HB_STOP.set()
    _HB_THREAD = None

# =============================================================================
# fsm.py에서 호출하는 퍼블릭 API
#  - 각 명령은 movement 프레임 중심으로 즉시송신
#  - 진입 안정성을 위해 control(driving_mode)과 heartbeat를 소량 동반
# =============================================================================
def _prime_driving_channel(burst_n: int = 1):
    """수신기 활성 보장을 위해 소량의 control+heartbeat 동반 송신"""
    for _ in range(burst_n):
        _write(_mk_control("driving_mode"))
        _write(_mk_heartbeat())

def issue_command_forward_and_turn(turn_dir: int) -> None:
    """
    전진 + 회전 (turn_dir: +1=좌, -1=우)
    v2 매핑: forward_left / forward_right + driving_mode control 동반
    """
    _prime_driving_channel(burst_n=1)
    name = "forward_left" if turn_dir > 0 else "forward_right"
    _write(_mk_movement(name))

def issue_command_backward_and_turn(turn_dir: int) -> None:
    """
    후진 + 회전 (turn_dir: +1=좌, -1=우)
    - 좌회전: 왼쪽 바퀴 약화(= 좌로 조향) → backward_left
    - 우회전: 오른쪽 바퀴 약화(= 우로 조향) → backward_right
    """
    _prime_driving_channel(burst_n=1)
    name = "backward_left" if turn_dir > 0 else "backward_right"
    _write(_mk_movement(name))

def issue_command_rotate_in_place(turn_dir: int) -> None:
    """
    제자리 회전 (turn_dir: +1=좌(CCW), -1=우(CW))
    v2 매핑: rotate_ccw/rotate_cw + 해당 control(rotate_*_ctrl) 동반
    - 회전 힘은 data[4](CCW)/data[5](CW) = 118 고정
    """
    ctrl = "rotate_ccw_ctrl" if turn_dir > 0 else "rotate_cw_ctrl"
    _write(_mk_control(ctrl))
    _write(_mk_heartbeat())
    name = "rotate_ccw" if turn_dir > 0 else "rotate_cw"
    _write(_mk_movement(name))

def issue_command_backward() -> None:
    """후진 직진"""
    _prime_driving_channel(burst_n=1)
    _write(_mk_movement("backward"))

def issue_command_forward() -> None:
    """전진 직진"""
    _prime_driving_channel(burst_n=1)
    _write(_mk_movement("forward"))

def issue_command_stop() -> None:
    """정지 (driving_mode + stop + heartbeat)"""
    _write(_mk_control("driving_mode"))
    _write(_mk_movement("stop"))
    _write(_mk_heartbeat())
