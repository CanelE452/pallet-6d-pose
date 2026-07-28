# -*- coding: utf-8 -*-
"""
AutonomousForkliftController (라이브러리용)
- 메인 실행부 없음. 외부 코드가 start()/stop()과 액션 API를 호출
- Heartbeat, Movement/Control TX 루프, 버스트/가드/락 포함
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional, List, Tuple, Callable, Any, Dict

from canlib import canlib, Frame
from .types import MessageFlag, OperationMode
from .exceptions import CANConnectionError

# ==============================
# 기본 설정(필요 시 import 후 재설정 가능)
# ==============================
CAN_CHANNEL = 0
CAN_BITRATE = 500_000
USE_EXTENDED_IDS = False

CAN_MOVEMENT_ID = 0x01E3
CAN_CONTROL_ID  = 0x02E3

HEARTBEAT_ID = 0x764
HEARTBEAT_PERIOD = 0.200
HEARTBEAT_DATA = [0x00]
HEARTBEAT_STARTUP_BURST_N = 5
HEARTBEAT_STARTUP_BURST_DT = 0.005

MOV_PERIOD_ACTIVE = 0.010
MOV_PERIOD_IDLE   = 0.010
CTRL_PERIOD_ACTIVE = 0.005
CTRL_PERIOD_IDLE   = 0.005
NEUTRAL_GUARD = 0.12

AN_NEUTRAL = 127
MAX_DELTA_THROTTLE = 60   # 안전 상한
MAX_DELTA_STEER    = 60
MAX_DELTA_ROTATE   = 30

# 정지/복귀 안정화 버스트 파라미터
STOP_BURST_N  = 3
STOP_BURST_DT = 0.010

CONTROL_TEMPLATES: Dict[str, List[int]] = {
    "driving_mode":    [0x42,0x00,0x00,0x0A,0x00,0x40,0x69,0x93],
    "lift_mode":       [0x42,0x00,0x00,0x05,0x00,0x40,0x69,0x93],
    "folding_mode":    [0x42,0x00,0x00,0x06,0x00,0x40,0x69,0x93],
    "reach_mode":      [0x42,0x00,0x00,0x09,0x00,0x40,0x69,0x93],
    "lift_down":         [0x42,0x00,0x00,0x15,0x00,0x40,0x69,0x93],
    "lift_up":       [0x42,0x00,0x00,0x25,0x00,0x40,0x69,0x93],
    "fold":            [0x42,0x00,0x00,0x26,0x00,0x40,0x69,0x93],
    "unfold":          [0x42,0x00,0x00,0x16,0x00,0x40,0x69,0x93],
    "reach_forward":   [0x42,0x00,0x00,0x19,0x00,0x40,0x69,0x93],
    "reach_backward":  [0x42,0x00,0x00,0x29,0x00,0x40,0x69,0x93],
    "rotate_ccw_ctrl": [0x42,0x00,0x00,0x0A,0x00,0x40,0x69,0x93],
    "rotate_cw_ctrl":  [0x42,0x00,0x00,0x0A,0x00,0x40,0x69,0x93],
    "emergency":       [0x80,0x00,0x00,0x00,0x01,0x40,0x69,0x93],
}

class AutonomousForkliftController:
    """
    비차단형 라이브러리 클래스:
      - start() : CAN 연결 + 내부 TX 루프 시작
      - stop()  : 루프 종료 + CAN 해제
      - 주행/회전/리프트 등 액션 API는 강도(%) + 시간(s) 기반
    """

    def __init__(
        self,
        *,
        can_channel: int = CAN_CHANNEL,
        can_bitrate: int = CAN_BITRATE,
        use_extended_ids: bool = USE_EXTENDED_IDS,
        logger: Optional[logging.Logger] = None,
    ):
        self.can_channel = can_channel
        self.can_bitrate = can_bitrate
        self.use_extended_ids = use_extended_ids

        self.logger = logger or logging.getLogger("forklift")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # CAN
        self.ch_a: Optional[canlib.Channel] = None

        # 상태
        self.is_running = False
        self.emergency_stop = False
        self.emergency_burst_sent = False
        self.sync_counter = 0x0A
        self.current_mode = OperationMode.DRIVING
        self.current_control_type = "driving_mode"

        self._movement_bytes: List[int] = [AN_NEUTRAL]*8
        self._movement_desc: str = "stop"
        self.last_nonneutral_ts = 0.0

        # 태스크
        self._tx_lock = asyncio.Lock()
        self.ctrl_tx_task: Optional[asyncio.Task] = None
        self.mov_tx_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

    # ========== 내부 유틸 ==========
    @property
    def _tx_flags(self) -> int:
        return MessageFlag.EXT if self.use_extended_ids else MessageFlag.STD

    async def _write(self, frame: Frame):
        if not self.ch_a:
            return
        async with self._tx_lock:
            self.ch_a.write(frame)

    def _next_sync(self) -> int:
        self.sync_counter = (self.sync_counter + 1) & 0x0F
        return self.sync_counter

    async def _burst_neutral_and_base_ctrl(self, base_ctrl: Optional[str] = None, n: int = STOP_BURST_N, dt: float = STOP_BURST_DT):
        """중립/기본 제어/하트비트를 짧게 여러 번 전송하여 해제 안정화"""
        ctrl = base_ctrl or f"{self.current_mode.value}_mode"
        for _ in range(max(1, n)):
            await self._write(self._ctrl_frame(ctrl))
            await self._write(self._mov_frame([AN_NEUTRAL]*8))
            await self._write(self._hb_frame())
            await asyncio.sleep(max(0.0, dt))

    # ========== 연결/해제 ==========
    def _connect_can(self):
        try:
            self.logger.info(f"🔌 CAN ch={self.can_channel}, bitrate={self.can_bitrate} (ID:{'EXT' if self.use_extended_ids else 'STD'})")
            self.ch_a = canlib.openChannel(self.can_channel)
            br_map = {
                1_000_000: canlib.Bitrate.BITRATE_1M,
                500_000:   canlib.Bitrate.BITRATE_500K,
                250_000:   canlib.Bitrate.BITRATE_250K,
                125_000:   canlib.Bitrate.BITRATE_125K,
            }
            bitrate = br_map.get(self.can_bitrate, canlib.Bitrate.BITRATE_500K)
            self.ch_a.setBusParams(bitrate)
            self.ch_a.busOn()
            self.logger.info("✅ CAN 연결 성공")
        except Exception as e:
            raise CANConnectionError(f"CAN 연결 실패: {e}") from e

    def _disconnect_can(self):
        if self.ch_a:
            try:
                self.ch_a.write(Frame(id_=CAN_MOVEMENT_ID, data=[AN_NEUTRAL]*8, flags=self._tx_flags))
            except Exception:
                pass
            try:
                self.ch_a.busOff()
                self.ch_a.close()
            except Exception:
                pass
            self.ch_a = None
            self.logger.info("🔌 CAN 해제 완료")

    # ========== 프레임 생성 ==========
    def _ctrl_frame(self, ctrl_type: str) -> Frame:
        data = CONTROL_TEMPLATES.get(ctrl_type, CONTROL_TEMPLATES["driving_mode"]).copy()
        if ctrl_type != "emergency":
            data[4] = self._next_sync()
        return Frame(id_=CAN_CONTROL_ID, data=data, flags=self._tx_flags)

    def _mov_frame(self, data8: List[int]) -> Frame:
        return Frame(id_=CAN_MOVEMENT_ID, data=data8, flags=self._tx_flags)

    def _hb_frame(self) -> Frame:
        return Frame(id_=HEARTBEAT_ID, data=HEARTBEAT_DATA, flags=self._tx_flags)

    # ========== 상태/설정 ==========
    def _set_mode(self, mode: OperationMode):
        if self.current_mode != mode:
            old = self.current_mode.value
            self.current_mode = mode
            self.current_control_type = f"{self.current_mode.value}_mode"
            self.logger.info(f"🔄 모드 전환: {old} → {self.current_mode.value}")

    def _set_control(self, ctrl_type: str):
        self.current_control_type = ctrl_type

    def _set_movement(self, data8: List[int], desc: str = "custom"):
        self._movement_bytes = data8
        self._movement_desc = desc
        if data8 != [AN_NEUTRAL]*8:
            self.last_nonneutral_ts = time.time()

    def _neutral_guard(self) -> List[int]:
        if (time.time() - self.last_nonneutral_ts) <= NEUTRAL_GUARD:
            return self._movement_bytes
        return [AN_NEUTRAL]*8

    # ========== 변환기 ==========
    @staticmethod
    def _clamp_byte(v: int) -> int:
        return 0 if v < 0 else (255 if v > 255 else v)

    @staticmethod
    def _pct_to_delta(pct: float, max_delta: int) -> int:
        pct = max(0.0, min(100.0, float(pct)))
        return int(round(max_delta * (pct / 100.0)))

    def _build_mov(
        self, *, throttle_pct: float = 0.0, steer_pct: float = 0.0, rotate_pct: float = 0.0
    ) -> List[int]:
        d_th = self._pct_to_delta(abs(throttle_pct), MAX_DELTA_THROTTLE)
        d_st = self._pct_to_delta(abs(steer_pct),    MAX_DELTA_STEER)
        d_rt = self._pct_to_delta(abs(rotate_pct),   MAX_DELTA_ROTATE)
        b = [AN_NEUTRAL]*8

        # steer: 좌(-)는 +delta, 우(+)는 -delta
        if steer_pct < 0: b[1] = self._clamp_byte(AN_NEUTRAL + d_st)
        elif steer_pct > 0: b[1] = self._clamp_byte(AN_NEUTRAL - d_st)

        # throttle: 전(+)=값 감소, 후(-)=값 증가
        if throttle_pct > 0: b[2] = self._clamp_byte(AN_NEUTRAL - d_th)
        elif throttle_pct < 0: b[2] = self._clamp_byte(AN_NEUTRAL + d_th)

        # rotate: CCW(-)=byte4 감소, CW(+)=byte5 감소
        if rotate_pct < 0: b[4] = self._clamp_byte(AN_NEUTRAL - d_rt)
        elif rotate_pct > 0: b[5] = self._clamp_byte(AN_NEUTRAL - d_rt)

        return b

    # ========== 내부 루프 ==========
    async def _ctrl_tx_loop(self):
        was_active = False
        while self.is_running:
            try:
                driving_active = (self.current_mode == OperationMode.DRIVING and self._movement_bytes != [AN_NEUTRAL]*8)
                period = CTRL_PERIOD_ACTIVE if driving_active else CTRL_PERIOD_IDLE

                ctrl_type = "emergency" if self.emergency_stop else self.current_control_type
                await self._write(self._ctrl_frame(ctrl_type))

                if self.emergency_stop and not self.emergency_burst_sent:
                    await self._emergency_burst()
                    self.emergency_burst_sent = True

                if driving_active and not was_active and not self.emergency_stop:
                    for _ in range(5):
                        await self._write(self._ctrl_frame(self.current_control_type))
                        await self._write(self._hb_frame())
                        await asyncio.sleep(0.005)

                was_active = driving_active
                await asyncio.sleep(period)
            except Exception as e:
                self.logger.error(f"CONTROL TX 오류: {e}")
                await asyncio.sleep(0.001)

    async def _mov_tx_loop(self):
        msg_cnt = 0
        while self.is_running:
            try:
                active = (self.current_mode == OperationMode.DRIVING and self._movement_bytes != [AN_NEUTRAL]*8 and not self.emergency_stop)
                data8 = self._movement_bytes if active else self._neutral_guard()
                if self.emergency_stop:
                    data8 = [AN_NEUTRAL]*8
                await self._write(self._mov_frame(data8))
                msg_cnt += 1
                if msg_cnt % 200 == 0 and active:
                    self.logger.debug(f"MOV alive: {self._movement_desc}")
                await asyncio.sleep(MOV_PERIOD_ACTIVE if active else MOV_PERIOD_IDLE)
            except Exception as e:
                self.logger.error(f"MOVEMENT TX 오류: {e}")
                await asyncio.sleep(0.005)

    async def _hb_loop(self):
        while self.is_running:
            try:
                await self._write(self._hb_frame())
                await asyncio.sleep(HEARTBEAT_PERIOD)
            except Exception as e:
                self.logger.error(f"Heartbeat 오류: {e}")
                await asyncio.sleep(0.1)

    async def _startup_burst(self):
        for _ in range(HEARTBEAT_STARTUP_BURST_N):
            await self._write(self._ctrl_frame("driving_mode"))
            await self._write(self._mov_frame([AN_NEUTRAL]*8))
            await self._write(self._hb_frame())
            await asyncio.sleep(HEARTBEAT_STARTUP_BURST_DT)

    async def _emergency_burst(self):
        for _ in range(10):
            await self._write(self._ctrl_frame("emergency"))
            await asyncio.sleep(0.02)

    # ========== 라이프사이클 ==========
    async def start(self):
        """CAN 연결 및 내부 루프 시작 (비차단)"""
        if self.is_running:
            return
        self._connect_can()
        self.is_running = True
        await self._startup_burst()
        self.ctrl_tx_task = asyncio.create_task(self._ctrl_tx_loop())
        self.mov_tx_task  = asyncio.create_task(self._mov_tx_loop())
        self.heartbeat_task = asyncio.create_task(self._hb_loop())

    async def stop(self):
        """내부 루프 종료 및 CAN 해제"""
        if not self.is_running:
            return
        # 해제 전에 강제 전체 정지
        try:
            await self.stop_all()
        except Exception:
            pass

        self.is_running = False
        for t in (self.ctrl_tx_task, self.mov_tx_task, self.heartbeat_task):
            try:
                if t and not t.done():
                    t.cancel()
            except Exception:
                pass
        self._disconnect_can()

    # ========== 공개 API: 모드/비상 ==========
    async def set_mode(self, mode: OperationMode):
        self._set_mode(mode)
        for _ in range(5):
            await self._write(self._ctrl_frame(f"{self.current_mode.value}_mode"))
            await self._write(self._hb_frame())
            await asyncio.sleep(0.005)

    async def emergency_stop_on(self):
        self.emergency_stop = True
        self.emergency_burst_sent = False
        # 버스트는 CTRL 루프에서 수행

    async def emergency_release(self):
        self.emergency_stop = False
        self.emergency_burst_sent = False
        # 안전: 해제 즉시 강제 중립
        self._set_movement([AN_NEUTRAL]*8, "stop")
        await self._burst_neutral_and_base_ctrl()

    # ========== 공개 API: 주행 ==========
    async def drive(self, *, throttle_pct: float, steer_pct: float, duration_s: float, desc: str = ""):
        self._set_mode(OperationMode.DRIVING)
        data8 = self._build_mov(throttle_pct=throttle_pct, steer_pct=steer_pct, rotate_pct=0.0)
        self._set_movement(data8, desc or f"drive th={throttle_pct} st={steer_pct}")
        self._set_control("driving_mode")
        try:
            await asyncio.sleep(max(0.0, float(duration_s)))
        finally:
            # ⛔️ 자동 정지
            self._set_movement([AN_NEUTRAL]*8, "stop")
            await self._burst_neutral_and_base_ctrl("driving_mode")

    async def rotate(self, *, rotate_pct: float, duration_s: float, desc: str = ""):
        self._set_mode(OperationMode.DRIVING)
        data8 = self._build_mov(throttle_pct=0.0, steer_pct=0.0, rotate_pct=rotate_pct)
        self._set_movement(data8, desc or f"rotate {rotate_pct}")
        self._set_control("driving_mode")
        try:
            await asyncio.sleep(max(0.0, float(duration_s)))
        finally:
            # ⛔️ 자동 정지
            self._set_movement([AN_NEUTRAL]*8, "stop")
            await self._burst_neutral_and_base_ctrl("driving_mode")

    # 편의 래퍼
    async def drive_forward(self, pct: float, duration_s: float):
        await self.drive(throttle_pct=abs(pct), steer_pct=0.0, duration_s=duration_s, desc=f"forward {pct}%")

    async def drive_backward(self, pct: float, duration_s: float):
        await self.drive(throttle_pct=-abs(pct), steer_pct=0.0, duration_s=duration_s, desc=f"backward {pct}%")

    async def turn_left(self, pct: float, duration_s: float):
        await self.drive(throttle_pct=0.0, steer_pct=-abs(pct), duration_s=duration_s, desc=f"turn_left {pct}%")

    async def turn_right(self, pct: float, duration_s: float):
        await self.drive(throttle_pct=0.0, steer_pct=+abs(pct), duration_s=duration_s, desc=f"turn_right {pct}%")

    async def forward_left(self, throttle_pct: float, steer_pct: float, duration_s: float):
        await self.drive(throttle_pct=abs(throttle_pct), steer_pct=-abs(steer_pct), duration_s=duration_s,
                         desc=f"fwd_left th{throttle_pct}% st{steer_pct}%")

    async def forward_right(self, throttle_pct: float, steer_pct: float, duration_s: float):
        await self.drive(throttle_pct=abs(throttle_pct), steer_pct=+abs(steer_pct), duration_s=duration_s,
                         desc=f"fwd_right th{throttle_pct}% st{steer_pct}%")

    async def rotate_cw(self, pct: float, duration_s: float):
        await self.rotate(rotate_pct=+abs(pct), duration_s=duration_s, desc=f"rotate_cw {pct}%")

    async def rotate_ccw(self, pct: float, duration_s: float):
        await self.rotate(rotate_pct=-abs(pct), duration_s=duration_s, desc=f"rotate_ccw {pct}%")

    # ========== 공개 API: 포크/리치 ==========
    async def _run_control_action(self, *, ctrl_type: str, mode_after: OperationMode, duration_s: float):
        """컨트롤 프레임 기반 액션을 duration 동안 유지 -> 종료 시 모드 복귀 + 중립 버스트"""
        self._set_mode(mode_after)
        self._set_control(ctrl_type)
        self._set_movement([AN_NEUTRAL]*8, "stop")
        try:
            await asyncio.sleep(max(0.0, float(duration_s)))
        finally:
            # 액션 해제: 모드 기본 제어로 복귀 + 안정화 버스트
            self._set_control(f"{mode_after.value}_mode")
            await self._burst_neutral_and_base_ctrl(f"{mode_after.value}_mode")

    async def lift_up(self, duration_s: float):
        await self._run_control_action(ctrl_type="lift_up", mode_after=OperationMode.LIFT, duration_s=duration_s)

    async def lift_down(self, duration_s: float):
        await self._run_control_action(ctrl_type="lift_down", mode_after=OperationMode.LIFT, duration_s=duration_s)

    async def fold(self, duration_s: float):
        await self._run_control_action(ctrl_type="fold", mode_after=OperationMode.FOLDING, duration_s=duration_s)

    async def unfold(self, duration_s: float):
        await self._run_control_action(ctrl_type="unfold", mode_after=OperationMode.FOLDING, duration_s=duration_s)

    async def reach_forward(self, duration_s: float):
        await self._run_control_action(ctrl_type="reach_forward", mode_after=OperationMode.REACH, duration_s=duration_s)

    async def reach_backward(self, duration_s: float):
        await self._run_control_action(ctrl_type="reach_backward", mode_after=OperationMode.REACH, duration_s=duration_s)

    # ========== 유틸: 즉시 정지 ==========
    async def stop_motion(self):
        """즉시 주행/회전 중립"""
        self._set_movement([AN_NEUTRAL]*8, "stop")
        for _ in range(STOP_BURST_N):
            await self._write(self._mov_frame([AN_NEUTRAL]*8))
            await asyncio.sleep(STOP_BURST_DT)

    async def stop_all(self):
        """모든 동작 완전 정지(모션 중립 + 현재 모드 기본 컨트롤 복귀 + 하트비트)"""
        self._set_movement([AN_NEUTRAL]*8, "stop")
        await self._burst_neutral_and_base_ctrl()
        # 이후 루프들이 계속 중립/기본제어를 유지 송신
        return

    # ========== 유틸: 스크립트 실행기(선택) ==========
    async def run_script(self, steps: List[Tuple[Callable[..., Any], tuple, dict]]):
        for fn, args, kwargs in steps:
            await fn(*args, **kwargs)
