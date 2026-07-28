# -*- coding: utf-8 -*-
"""
DirectFrameForkliftController (딸깍 방지/연속 주행 안정화 패치)
- Heartbeat 200ms + 시작/전환 버스트
- Idle에서도 movement/control 20ms keep-alive
- 표준/확장 ID 스위치 (USE_EXTENDED_IDS)
- 모든 CAN write에 asyncio.Lock 적용 (지터 감소)
"""

from canlib import canlib, Frame

class MessageFlag:
    STD = 0x0000  # 표준 11-bit ID  
    EXT = 0x0004  # 확장 29-bit ID
import asyncio
import keyboard
import logging
import time
from enum import Enum
from typing import Optional, Dict, Set

# =============================================================================
# 🔧 설정 영역
# =============================================================================
CAN_CHANNEL = 0
CAN_BITRATE = 500_000

# 📡 표준/확장 ID 스위치 (리모컨 캡처와 동일하게 맞추세요: False=표준11bit, True=확장29bit)
USE_EXTENDED_IDS = False

# CAN IDs
CAN_MOVEMENT_ID = 0x01E3
CAN_CONTROL_ID  = 0x02E3

# 🫀 Heartbeat (워치독) — 250ms 요구라면 여유있게 200ms로 가속
HEARTBEAT_ID = 0x764
HEARTBEAT_PERIOD = 0.200   # 200 ms
HEARTBEAT_DATA = [0x00]
HEARTBEAT_STARTUP_BURST_N = 5
HEARTBEAT_STARTUP_BURST_DT = 0.005  # 5 ms

KEY_SCAN_PERIOD = 0.02   # 50 Hz 키 스캔


# 딸깍 방지 최적화 주기
#ACTIVE: 지게차가 실제로 움직이거나 회전할 때의 전송 주기 (10ms = 100Hz)
#IDLE: 정지 상태이거나 다른 모드일 때의 전송 주기 (MOV: 10ms, CTRL: 20ms)
# 딸깍 방지/버스부하 최적화 주기(keep-alive 강화)
# ACTIVE: 주행 중 10ms, IDLE: 정지/비주행 시에도 20ms로 빠르게 유지
MOV_PERIOD_ACTIVE = 0.010
MOV_PERIOD_IDLE   = 0.010
#CTRL_PERIOD_ACTIVE = 0.010
#CTRL_PERIOD_IDLE   = 0.020

CTRL_PERIOD_ACTIVE = 0.005 #GOOD
CTRL_PERIOD_IDLE   = 0.005 #GOOD 떨림 있음

# 전송 주기를 더 여유롭게 조정
#CTRL_PERIOD_ACTIVE = 0.025   # 10ms → 25ms  진동 더 많이.
#CTRL_PERIOD_IDLE   = 0.050   # 20ms → 50ms진동 더 많이.

HOLD_GRACE = 0.20         # 키 래치 (키 떼도 200ms간 유지)
NEUTRAL_GUARD = 0.12      # Neutral Guard (중립 천이 지연)

LOG_LEVEL = "INFO"

# 🎮 조이스틱 강도 설정 (중립값 127 기준으로 ±값 설정)
AN_NEUTRAL = 127                  # 중립값 (고정값)
JOYSTICK_FORWARD = 60             # 전진 강도 (127-60=67) 
JOYSTICK_BACKWARD = 60            # 후진 강도 (127+60=187)
JOYSTICK_LEFT = 60                # 좌회전 강도 (127+60=187)
JOYSTICK_RIGHT = 60               # 우회전 강도 (127-60=67)
JOYSTICK_ROTATE_CCW = 30          # 반시계회전 강도 (127-30=97)
JOYSTICK_ROTATE_CW = 30           # 시계회전 강도 (127-30=97)

# 계산된 실제 조이스틱 값들
AN_FORWARD = min(255, AN_NEUTRAL - JOYSTICK_FORWARD)   # 67
AN_BACKWARD = max(0,   AN_NEUTRAL + JOYSTICK_BACKWARD) # 187
AN_LEFT =     min(255, AN_NEUTRAL + JOYSTICK_LEFT)     # 187
AN_RIGHT =    max(0,   AN_NEUTRAL - JOYSTICK_RIGHT)    # 67
AN_ROTATE_CCW = max(0, min(255, AN_NEUTRAL - JOYSTICK_ROTATE_CCW))  # 118
AN_ROTATE_CW  = max(0, min(255, AN_NEUTRAL - JOYSTICK_ROTATE_CW))   # 118

# =============================================================================
# 🎮 움직임/제어 프레임 템플릿
# =============================================================================
AN_N = AN_NEUTRAL

MOVEMENT_TEMPLATES: Dict[str, list[int]] = {
    "stop":            [AN_N, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N],
    "forward":         [AN_N, AN_N, AN_FORWARD,  AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward":        [AN_N, AN_N, AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "turn_left":       [AN_N, AN_LEFT, AN_N,     AN_N, AN_N, AN_N, AN_N, AN_N],
    "turn_right":      [AN_N, AN_RIGHT,AN_N,     AN_N, AN_N, AN_N, AN_N, AN_N],
    "rotate_ccw":      [AN_N, AN_N,   AN_N,      AN_N, AN_ROTATE_CCW, AN_N,        AN_N, AN_N],
    "rotate_cw":       [AN_N, AN_N,   AN_N,      AN_N, AN_N,          AN_ROTATE_CW,AN_N, AN_N],
    "forward_left":    [AN_N, AN_LEFT, AN_FORWARD,  AN_N, AN_N, AN_N, AN_N, AN_N],
    "forward_right":   [AN_N, AN_RIGHT,AN_FORWARD,  AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward_left":   [AN_N, AN_LEFT, AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
    "backward_right":  [AN_N, AN_RIGHT,AN_BACKWARD, AN_N, AN_N, AN_N, AN_N, AN_N],
}

# 제어 템플릿 — 5번째 바이트(data[4])는 동기 카운터로 run-time에 갱신
CONTROL_TEMPLATES: Dict[str, list[int]] = {
    "driving_mode":    [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "lift_mode":       [0x42, 0x00, 0x00, 0x05, 0x00, 0x40, 0x69, 0x93],
    "folding_mode":    [0x42, 0x00, 0x00, 0x06, 0x00, 0x40, 0x69, 0x93],
    "reach_mode":      [0x42, 0x00, 0x00, 0x09, 0x00, 0x40, 0x69, 0x93],

    "lift_up":         [0x42, 0x00, 0x00, 0x15, 0x00, 0x40, 0x69, 0x93],
    "lift_down":       [0x42, 0x00, 0x00, 0x25, 0x00, 0x40, 0x69, 0x93],
    "fold":            [0x42, 0x00, 0x00, 0x26, 0x00, 0x40, 0x69, 0x93],
    "unfold":          [0x42, 0x00, 0x00, 0x16, 0x00, 0x40, 0x69, 0x93],
    "reach_forward":   [0x42, 0x00, 0x00, 0x19, 0x00, 0x40, 0x69, 0x93],
    "reach_backward":  [0x42, 0x00, 0x00, 0x29, 0x00, 0x40, 0x69, 0x93],

    "rotate_ccw_ctrl": [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "rotate_cw_ctrl":  [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],

    "emergency":       [0x80, 0x00, 0x00, 0x00, 0x01, 0x40, 0x69, 0x93],  # sync 미적용
}

# =============================================================================
# 본체
# =============================================================================
class OperationMode(Enum):
    DRIVING = "driving"
    LIFT = "lift"
    FOLDING = "folding"
    REACH = "reach"

class DirectFrameForkliftController:
    def __init__(self):
        # CAN
        self.ch_a: Optional[canlib.Channel] = None

        # 상태
        self.is_running = False
        self.emergency_stop = False
        self.emergency_burst_sent = False
        self.sync_counter = 0x0A
        self.current_mode = OperationMode.DRIVING
        self.current_movement = "stop"
        self.current_control_type = "driving_mode"

        # Neutral Guard
        self.last_nonstop_cmd_ts = 0.0
        self.neutral_guard = NEUTRAL_GUARD

        # Heartbeat
        self.heartbeat_task = None
        self.last_heartbeat_time = 0.0
        self._heartbeat_count = 0

        # 키 상태 + 마지막 True 시각(래치용)
        self.key_states: Dict[str, bool] = {
            'w': False, 'a': False, 's': False, 'd': False,
            'q': False, 'e': False, 'k': False, 'l': False,
            'u': False, 'i': False, 'o': False, 'p': False,
            'space': False, 'z': False, 'm': False
        }
        self._last_true_ts: Dict[str, float] = {k: 0.0 for k in self.key_states}

        # 태스크
        self.key_monitor_task = None
        self.ctrl_tx_task = None
        self.mov_tx_task = None

        # TX 락 (모든 write 보호)
        self._tx_lock = asyncio.Lock()

        # 로깅
        logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger("forklift")

    # ---------- helpers ----------
    @property
    def _tx_flags(self) -> MessageFlag:
        return MessageFlag.EXT if USE_EXTENDED_IDS else MessageFlag.STD

    async def _write(self, frame: Frame):
        """모든 CAN write는 이 메서드를 통해 락 보호"""
        if not self.ch_a:
            return
        async with self._tx_lock:
            self.ch_a.write(frame)

    # ---------- CAN 연결 ----------
    def connect_can(self) -> bool:
        try:
            self.logger.info(f"🔌 CAN ch={CAN_CHANNEL}, bitrate={CAN_BITRATE}bps 연결 중... (ID:{'EXT' if USE_EXTENDED_IDS else 'STD'})")
            self.ch_a = canlib.openChannel(CAN_CHANNEL)
            br_map = {
                1_000_000: canlib.Bitrate.BITRATE_1M,
                500_000:   canlib.Bitrate.BITRATE_500K,
                250_000:   canlib.Bitrate.BITRATE_250K,
                125_000:   canlib.Bitrate.BITRATE_125K,
            }
            bitrate = br_map.get(CAN_BITRATE, canlib.Bitrate.BITRATE_500K)
            self.ch_a.setBusParams(bitrate)
            self.ch_a.busOn()
            self.logger.info("✅ CAN 연결 성공")
            return True
        except Exception as e:
            self.logger.error(f"❌ CAN 연결 실패: {e}")
            return False

    def disconnect_can(self):
        if self.ch_a:
            try:
                stop_data = MOVEMENT_TEMPLATES["stop"]
                stop_frame = Frame(id_=CAN_MOVEMENT_ID, data=stop_data, flags=self._tx_flags)
                self.ch_a.write(stop_frame)
                self.ch_a.busOff()
                self.ch_a.close()
                self.logger.info("🔌 CAN 해제 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ CAN 해제 중 오류: {e}")

    # ---------- 프레임 생성 ----------
    def _next_sync(self) -> int:
        self.sync_counter = (self.sync_counter + 1) & 0x0F
        return self.sync_counter

    def create_control_frame(self, ctrl_type: str) -> Frame:
        data = CONTROL_TEMPLATES.get(ctrl_type, CONTROL_TEMPLATES["driving_mode"]).copy()
        if ctrl_type != "emergency":
            data[4] = self._next_sync()
        return Frame(id_=CAN_CONTROL_ID, data=data, flags=self._tx_flags)

    def create_movement_frame(self, movement: str) -> Frame:
        data = MOVEMENT_TEMPLATES.get(movement, MOVEMENT_TEMPLATES["stop"])
        return Frame(id_=CAN_MOVEMENT_ID, data=data, flags=self._tx_flags)

    def create_heartbeat_frame(self) -> Frame:
        return Frame(id_=HEARTBEAT_ID, data=HEARTBEAT_DATA, flags=self._tx_flags)

    # ---------- 키 처리 ----------
    async def key_monitor_loop(self):
        self.logger.info(f"🎮 키 감시 시작 ({KEY_SCAN_PERIOD*1000:.1f}ms, HOLD_GRACE={HOLD_GRACE*1000:.0f}ms)")
        prev_states = self.key_states.copy()

        while self.is_running:
            try:
                now = time.time()
                changed: Set[str] = set()

                for key in self.key_states:
                    try:
                        raw = keyboard.is_pressed(key)
                    except Exception:
                        raw = False

                    if raw:
                        self._last_true_ts[key] = now

                    latched = raw or (now - self._last_true_ts[key] <= HOLD_GRACE)

                    if latched != self.key_states[key]:
                        self.key_states[key] = latched
                        changed.add(key)

                if changed:
                    await self._handle_mode_changes(changed, prev_states)
                    prev_states = self.key_states.copy()

                new_movement, new_control = self._determine_movement_from_keys()

                if new_movement != self.current_movement or new_control != self.current_control_type:
                    self.current_movement = new_movement
                    self.current_control_type = new_control
                    if new_movement != "stop":
                        self.last_nonstop_cmd_ts = time.time()
                        self.logger.info(f"🚀 동작 변경: {new_movement} [{self.current_mode.value}]")

                if new_movement == "quit":
                    self.logger.info("👋 종료 요청")
                    self.is_running = False
                    break
                elif new_movement == "emergency_stop":
                    if not self.emergency_stop:
                        self.emergency_stop = True
                        self.emergency_burst_sent = False
                        self.logger.warning("🚨 비상정지!")
                elif new_movement == "emergency_release":
                    if self.emergency_stop:
                        self.emergency_stop = False
                        self.emergency_burst_sent = False
                        self.logger.info("✅ 비상정지 해제")

                await asyncio.sleep(KEY_SCAN_PERIOD)

            except Exception as e:
                self.logger.error(f"❌ 키 감시 오류: {e}")
                await asyncio.sleep(0.1)

    async def _handle_mode_changes(self, changed: Set[str], prev: Dict[str, bool]):
        mode_keys = {'u': OperationMode.DRIVING, 'i': OperationMode.LIFT,
                     'o': OperationMode.FOLDING, 'p': OperationMode.REACH}
        switched = False
        for k in changed:
            if k in mode_keys and self.key_states[k] and not prev[k]:
                old = self.current_mode.value
                self.current_mode = mode_keys[k]
                self.current_control_type = f"{self.current_mode.value}_mode"
                self.logger.info(f"🔄 모드 전환: {old} → {self.current_mode.value}")
                switched = True

        if switched:
            try:
                # 모드 전환 버스트: control + heartbeat 함께 안정화(수신기 활성 보장)
                for _ in range(5):
                    await self._write(self.create_control_frame(self.current_control_type))
                    await self._write(self.create_heartbeat_frame())
                    await asyncio.sleep(0.005)
            except Exception as e:
                self.logger.debug(f"mode burst err: {e}")

    def _determine_movement_from_keys(self):
        if self.key_states['m']:
            return "emergency_release", "driving_mode"
        if self.key_states['space']:
            return "emergency_stop", "emergency"
        if self.key_states['z']:
            return "quit", "driving_mode"

        if self.key_states['q']:
            return "rotate_ccw", "rotate_ccw_ctrl"
        if self.key_states['e']:
            return "rotate_cw", "rotate_cw_ctrl"

        if self.current_mode == OperationMode.DRIVING:
            f = self.key_states['w']; b = self.key_states['s']
            l = self.key_states['a']; r = self.key_states['d']
            if f and l:   return "forward_left",  "driving_mode"
            if f and r:   return "forward_right", "driving_mode"
            if b and l:   return "backward_left", "driving_mode"
            if b and r:   return "backward_right","driving_mode"
            if f:         return "forward",       "driving_mode"
            if b:         return "backward",      "driving_mode"
            if l:         return "turn_left",     "driving_mode"
            if r:         return "turn_right",    "driving_mode"

        if self.key_states['k'] or self.key_states['l']:
            return self._get_fork_operation()

        # Neutral Guard
        if self.current_movement != "stop":
            now = time.time()
            if (now - self.last_nonstop_cmd_ts) <= self.neutral_guard:
                return self.current_movement, self.current_control_type

        return "stop", f"{self.current_mode.value}_mode"

    def _get_fork_operation(self):
        if self.current_mode == OperationMode.DRIVING:
            return "stop", "driving_mode"

        if self.current_mode == OperationMode.LIFT:
            if self.key_states['k']: return "stop", "lift_up"
            if self.key_states['l']: return "stop", "lift_down"

        if self.current_mode == OperationMode.FOLDING:
            if self.key_states['k']: return "stop", "unfold"
            if self.key_states['l']: return "stop", "fold"

        if self.current_mode == OperationMode.REACH:
            if self.key_states['k']: return "stop", "reach_forward"
            if self.key_states['l']: return "stop", "reach_backward"

        return "stop", f"{self.current_mode.value}_mode"

    # ---------- 전송 루프들 ----------
    async def control_tx_loop(self):
        self.logger.info(f"🔧 CONTROL TX 시작 (active={CTRL_PERIOD_ACTIVE*1000:.0f}ms / idle={CTRL_PERIOD_IDLE*1000:.0f}ms)")
        was_active = False
        while self.is_running:
            try:
                driving_active = (self.current_mode == OperationMode.DRIVING and self.current_movement != "stop")
                period = CTRL_PERIOD_ACTIVE if driving_active else CTRL_PERIOD_IDLE

                ctrl_type = "emergency" if self.emergency_stop else self.current_control_type
                frame = self.create_control_frame(ctrl_type)
                await self._write(frame)

                if self.emergency_stop and not self.emergency_burst_sent:
                    await self._handle_emergency_burst()
                    self.emergency_burst_sent = True

                # 주행 진입시 버스트 (control + heartbeat)
                if driving_active and not was_active:
                    try:
                        for _ in range(5):
                            await self._write(self.create_control_frame(self.current_control_type))
                            await self._write(self.create_heartbeat_frame())
                            await asyncio.sleep(0.005)
                    except Exception as e:
                        self.logger.debug(f"ctrl entry burst err: {e}")

                was_active = driving_active
                await asyncio.sleep(period)

            except Exception as e:
                self.logger.error(f"CONTROL TX 오류: {e}")
                await asyncio.sleep(0.001)

    async def movement_tx_loop(self):
        self.logger.info(f"🚚 MOVEMENT TX 시작 (active={MOV_PERIOD_ACTIVE*1000:.0f}ms / idle={MOV_PERIOD_IDLE*1000:.0f}ms)")
        msg_cnt = 0
        while self.is_running:
            try:
                mov_name = "stop" if self.emergency_stop else self.current_movement
                active = (self.current_mode == OperationMode.DRIVING and mov_name != "stop")

                mov = self.create_movement_frame(mov_name)
                await self._write(mov)

                msg_cnt += 1
                if msg_cnt % 200 == 0 and mov_name != "stop":
                    self.logger.debug(f"📊 MOV TX alive: {mov_name} (active={active})")

                await asyncio.sleep(MOV_PERIOD_ACTIVE if active else MOV_PERIOD_IDLE)

            except Exception as e:
                self.logger.error(f"MOVEMENT TX 오류: {e}")
                await asyncio.sleep(0.005)

    async def heartbeat_loop(self):
        """Heartbeat — 200ms 주기 keep-alive"""
        self.logger.info(f"🫀 Heartbeat 시작 (ID:0x{HEARTBEAT_ID:03X}, {HEARTBEAT_PERIOD*1000:.0f}ms)")
        while self.is_running:
            try:
                heartbeat_frame = self.create_heartbeat_frame()
                await self._write(heartbeat_frame)
                self.last_heartbeat_time = time.time()

                self._heartbeat_count += 1
                if self._heartbeat_count % 25 == 0:  # 5초마다(200ms*25=5s)
                    self.logger.debug(f"💓 Heartbeat alive (#{self._heartbeat_count})")

                await asyncio.sleep(HEARTBEAT_PERIOD)

            except Exception as e:
                self.logger.error(f"❌ Heartbeat 오류: {e}")
                await asyncio.sleep(0.1)

    async def _handle_emergency_burst(self):
        self.logger.warning("🚨 비상정지 CONTROL 버스트 전송...")
        for _ in range(10):
            await self._write(self.create_control_frame("emergency"))
            await asyncio.sleep(0.02)
        self.logger.warning("🛑 비상정지 버스트 완료(M로 해제)")

    # ---------- 시작 시 안정화 시퀀스 ----------
    async def _startup_burst_sequence(self):
        """프로그램 시작 직후: driving_mode + stop + heartbeat 버스트로 안정 진입"""
        self.logger.info("🚦 시작 안정화 버스트 수행 중...")
        for _ in range(HEARTBEAT_STARTUP_BURST_N):
            await self._write(self.create_control_frame("driving_mode"))
            await self._write(self.create_movement_frame("stop"))
            await self._write(self.create_heartbeat_frame())
            await asyncio.sleep(HEARTBEAT_STARTUP_BURST_DT)
        self.logger.info("✅ 시작 안정화 버스트 완료")

    # ---------- 입출력 ----------
    def print_frame_reference(self):
        print("\n" + "="*80)
        print("📋 CAN 프레임 참조표 (Heartbeat 포함)")
        print("="*80)
        print(f"\n🫀 Heartbeat(0x{HEARTBEAT_ID:03X}):")
        hx = [f"0x{b:02x}" for b in HEARTBEAT_DATA]
        print(f"  heartbeat       = Frame(id_=0x{HEARTBEAT_ID:03X}, data={hx}, flags={'EXT' if USE_EXTENDED_IDS else 'STD'})")
        print(f"\n🎮 조이스틱(0x{CAN_MOVEMENT_ID:03X}):")
        for name, data in MOVEMENT_TEMPLATES.items():
            hx = [f"0x{b:02x}" for b in data]
            print(f"  {name:15} = Frame(id_=0x{CAN_MOVEMENT_ID:03X}, data={hx}, flags={'EXT' if USE_EXTENDED_IDS else 'STD'})")
        print(f"\n🔧 제어(0x{CAN_CONTROL_ID:03X}):")
        for name, data in CONTROL_TEMPLATES.items():
            hx = [f"0x{b:02x}" for b in data]
            print(f"  {name:15} = Frame(id_=0x{CAN_CONTROL_ID:03X}, data={hx}, flags={'EXT' if USE_EXTENDED_IDS else 'STD'})")
        print("="*80)

    def print_instructions(self):
        print("\n" + "="*80)
        print("🚜 Heartbeat 포함 딸깍 방지 키보드 제어")
        print("="*80)
        print("W/A/S/D: 전진/좌/후진/우   | Q/E: 제자리 반시계/시계")
        print("U/I/O/P: 주행/리프트/폴딩/리치 모드")
        print("K/L    : 모드별 포크 동작")
        print("SPACE  : 비상정지  | M: 해제  | Z: 종료")
        print("="*80)
        print(f"CAN ch={CAN_CHANNEL}, bitrate={CAN_BITRATE}, ID type={'EXT(29bit)' if USE_EXTENDED_IDS else 'STD(11bit)'}")
        print(f"IDs: move=0x{CAN_MOVEMENT_ID:03X}, ctrl=0x{CAN_CONTROL_ID:03X}, heartbeat=0x{HEARTBEAT_ID:03X}")
        print(f"주기: scan={KEY_SCAN_PERIOD*1000:.0f}ms, MOV active={MOV_PERIOD_ACTIVE*1000:.0f}ms/idle={MOV_PERIOD_IDLE*1000:.0f}ms")
        print(f"     CTRL active={CTRL_PERIOD_ACTIVE*1000:.0f}ms/idle={CTRL_PERIOD_IDLE*1000:.0f}ms, HEARTBEAT={HEARTBEAT_PERIOD*1000:.0f}ms")
        print("="*80)

    # ---------- 실행 ----------
    async def run(self):
        self.print_frame_reference()
        if not self.connect_can():
            return
        self.is_running = True
        self.print_instructions()
        try:
            # 시작 안정화 버스트
            await self._startup_burst_sequence()

            # 태스크 실행
            self.key_monitor_task = asyncio.create_task(self.key_monitor_loop())
            self.ctrl_tx_task = asyncio.create_task(self.control_tx_loop())
            self.mov_tx_task = asyncio.create_task(self.movement_tx_loop())
            self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())

            await asyncio.gather(
                self.key_monitor_task,
                self.ctrl_tx_task,
                self.mov_tx_task,
                self.heartbeat_task,
                return_exceptions=True
            )
        finally:
            self.is_running = False
            for t in (self.key_monitor_task, self.ctrl_tx_task, self.mov_tx_task, self.heartbeat_task):
                try:
                    if t and not t.done():
                        t.cancel()
                except:
                    pass
            self.disconnect_can()
            self.logger.info("🏁 종료")

# =============================================================================
# 진입점
# =============================================================================
async def main():
    ctrl = DirectFrameForkliftController()
    await ctrl.run()

if __name__ == "__main__":
    #print("🎮 Heartbeat 포함 딸깍 방지 지게차 제어기")
    #print("🫀 특징: Heartbeat(0x764) 200ms 주기로 워치독 방지 + 시작 버스트")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 강제 종료")
