# calib/fsm.py
# (ALIGN rework: YAW_CHECK uses yaw_smooth; 90° chains use rel_yaw baseline)
from __future__ import annotations
from typing import List, Tuple, Optional
import time
import math

from .config import (
    # thresholds
    YAW_TOL_DEG, OFF_TOL_M,
    ALIGN_DIST_M, ALIGN_BAND_M,
    WIDTH_MIN_FULL,
    CMD_STABLE_THR,

    # angle & stop
    REL_YAW_TARGET_DEG,
    STOP_SEC,

    # forward-time model switches & params (consumed via motion_models)
    USE_PIECEWISE_FWD_FIT,

    # HUD colors
    COLOR_STATUS_OK, COLOR_STATUS_TRK, COLOR_ALERT, COLOR_META,
)

from .control import (
    issue_command_forward,
    issue_command_backward,
    issue_command_rotate_in_place,
    issue_command_stop,
)

from .command_status import CommandStatus
from .motion_models import fwd_sec_from_offset_piecewise


# ------------------------------
# 작은 유틸
# ------------------------------
def _within_band(val: Optional[float], target: float, band: float) -> bool:
    return (val is not None) and (abs(val - target) <= band)

def _cmd_name_human(code: str) -> str:
    return {
        "FWD": "전진",
        "BACK": "후진",
        "ROT_LEFT": "제자리 좌회전",
        "ROT_RIGHT": "제자리 우회전",
        "STOP": "정지",
    }.get(code, code)

def _deg_delta(a: float, b: float) -> float:
    """절대 각도 차(도)."""
    return abs(a - b)


# ------------------------------
# FSM 본체
# ------------------------------
class CalibrationFSM:
    """
    Top-level:
      SEARCH → DETECTED → (ALIGN | RECOVER) → CHECK → (ALIGN | DONE)

    ALIGN substates (diagram-conform):
      DIST_CHECK
       ├─ dist_z > ALIGN_DIST_M          → ALIGN_FWD_ADJUST
       ├─ dist_z < ALIGN_DIST_M          → ALIGN_BWD_ADJUST
       └─ |dist_z-ALIGN_DIST_M| ≤ band   → YAW_CHECK
      YAW_CHECK
       ├─ yaw>0 → ROTATE_RIGHT_UNTIL_YAW_TOL
       ├─ yaw<0 → ROTATE_LEFT_UNTIL_YAW_TOL
       └─ |yaw|≤tol → OFFSET_CHECK   <-- yaw_smooth 기준
      OFFSET_CHECK
       ├─ offset>+tol → ALIGN_ROTATE_RIGHT (compute FWD_SEC)
       ├─ offset<-tol → ALIGN_ROTATE_LEFT  (compute FWD_SEC)
       ├─ |offset|≤tol & |yaw|>tol → DIST_CHECK
       └─ |offset|≤tol & |yaw|≤tol → READY_TO_DONE
      RIGHT branch (rel_yaw 기준):
         ALIGN_ROTATE_RIGHT(90) → [STOP] → FORWARD_AFTER_RIGHT(FWD_SEC) → [STOP] → ALIGN_ROTATE_LEFT_90(90) → YAW_CHECK
      LEFT branch  (rel_yaw 기준):
         ALIGN_ROTATE_LEFT(90) → [STOP] → FORWARD_AFTER_LEFT(FWD_SEC)  → [STOP] → ALIGN_ROTATE_RIGHT_90(90) → YAW_CHECK

    Global interlock:
      - 모든 제어 명령 상태 사이에 STOP_SEC 정지 적용
    """

    # ---------- 생성 ----------
    def __init__(self):
        self.state: str = "SEARCH"
        self.align_sub: str = "DIST_CHECK"
        self.recover_sub: str = "DECIDE_TURN"

        # 안정화 카운터
        self._stable_tag: Optional[str] = None
        self._stable_k: int = 0

        # 타이머
        self._timer_until: float = 0.0

        # STOP 인터록
        self._interlock_active: bool = False
        self._after_interlock_state: Optional[str] = None

        # 회전 체인용 rel_yaw 기준
        self._rel_yaw_ref: Optional[float] = None

        # 최근 회전 방향(+1 좌, -1 우) — 탐지 유실 시 스핀 방향 결정용
        self._last_dir_sign: int = +1

        # HUD 표기 상태
        self.cmd_status: CommandStatus = CommandStatus()
        self._last_exec_cmd: Optional[str] = None

        # 전진 시간 (OFFSET_CHECK에서 계산하여 분기 체인에서 사용)
        self._fwd_sec_cached: float = 0.0

    # ---------- 내부 헬퍼 ----------
    def _reset_stabilizer(self):
        self._stable_tag = None
        self._stable_k = 0

    def _stabilize(self, tag: str) -> bool:
        if self._stable_tag == tag:
            self._stable_k += 1
        else:
            self._stable_tag = tag
            self._stable_k = 1
        return self._stable_k >= CMD_STABLE_THR

    def _start_timer(self, sec: float):
        self._timer_until = time.time() + sec

    def _timer_active(self) -> bool:
        return time.time() < self._timer_until

    def _start_interlock_then(self, next_state: str):
        """전역 STOP 인터록 시작 후, 종료되면 next_state로 진입."""
        self._interlock_active = True
        self._after_interlock_state = next_state
        self._exec("STOP")
        self._start_timer(STOP_SEC)
        self.cmd_status.start_timed("STOP", STOP_SEC)

    def _maybe_finish_interlock(self) -> bool:
        """인터록 진행 중이면 처리하고 True 반환(즉, 본 로직 건너뜀)."""
        if self._interlock_active:
            self._exec("STOP")
            remain = max(0.0, self._timer_until - time.time())
            self.cmd_status.start_timed("STOP", remain)
            if not self._timer_active():
                self._interlock_active = False
                if self._after_interlock_state is not None:
                    self.align_sub = self._after_interlock_state
                    self._after_interlock_state = None
            return True
        return False

    def _exec(self, cmd: str):
        # 동일 명령 연속 송신 억제
        if cmd == self._last_exec_cmd:
            return
        self._last_exec_cmd = cmd

        if cmd == "FWD":
            issue_command_forward()
        elif cmd == "BACK":
            issue_command_backward()
        elif cmd == "ROT_LEFT":
            self._last_dir_sign = +1
            issue_command_rotate_in_place(+1)
        elif cmd == "ROT_RIGHT":
            self._last_dir_sign = -1
            issue_command_rotate_in_place(-1)
        elif cmd == "STOP":
            issue_command_stop()

    def _start_until_status(self, code: str, metric_name: str, current_value: float, target_value: float):
        self.cmd_status.start_until(code, metric_name, current_value, target_value)

    # ---------- 메인 스텝 ----------
    def step(self,
             *,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw_smooth: Optional[float],
             offset_smooth: Optional[tuple],
             # heading_deg는 더 이상 사용하지 않지만 하위 호환 위해 남겨둠
             heading_deg: Optional[float] = None,
             # ★ rel_yaw 인자 추가 (제자리 90° 체인 종료/진행률용)
             rel_yaw: Optional[float] = None) -> List[Tuple[str, tuple]]:
        """
        Args:
          det_ok           : 팔레트 탐지 여부
          detected_length  : front 폭 추정치(m)
          dist_z           : 리프터-파렛트 거리(m)
          yaw_smooth       : 평면 기반 yaw(deg) — YAW_CHECK에서 사용
          offset_smooth    : (offset_x, offset_y, offset_z) — 여기서는 offset_x만 사용
          heading_deg      : (옵션) IMU Z 적분 헤딩 — 사용 안 함(호환 유지용)
          rel_yaw          : (옵션) IMU 기반 상대 yaw(deg) — 회전 체인(±90°)에서 사용
        """
        lines: List[Tuple[str, tuple]] = []

        # 입력 정리
        ox = None
        if offset_smooth is not None:
            try:
                ox = float(offset_smooth[0])
            except Exception:
                ox = None

        yaw_ok = (yaw_smooth is not None) and (abs(yaw_smooth) <= YAW_TOL_DEG)
        off_ok = (ox is not None) and (abs(ox) <= OFF_TOL_M)
        band_ok = _within_band(dist_z, ALIGN_DIST_M, ALIGN_BAND_M)

        # 진행 중 'until' 상태 갱신
        if self.cmd_status.mode == "until":
            if self.cmd_status.metric_name == "|yaw|" and (yaw_smooth is not None):
                self.cmd_status.update_metric(abs(yaw_smooth))
            elif self.cmd_status.metric_name == "|offset_x|" and (ox is not None):
                self.cmd_status.update_metric(abs(ox))
            elif self.cmd_status.metric_name == "|rel_yaw|":
                if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                    cur = abs(rel_yaw - self._rel_yaw_ref)
                    self.cmd_status.update_metric(cur)

        # ------------------------------
        # SEARCH
        # ------------------------------
        if self.state == "SEARCH":
            self._reset_stabilizer()
            self.align_sub = "DIST_CHECK"
            self.recover_sub = "DECIDE_TURN"
            self._last_exec_cmd = None

            if det_ok:
                self.state = "DETECTED"
                lines.append(("[SEARCH→DETECTED]", COLOR_META))
            else:
                # 탐지 대기: 최근 방향으로 제자리 회전
                self._exec("ROT_LEFT" if self._last_dir_sign > 0 else "ROT_RIGHT")
                lines.append(("[SEARCH] 탐지 대기: 스핀", COLOR_STATUS_TRK))
            return lines

        # ------------------------------
        # DETECTED
        # ------------------------------
        if self.state == "DETECTED":
            if not det_ok:
                self.state = "SEARCH"
                self._exec("STOP")
                self.cmd_status.start_timed("STOP", 0.0)
                lines.append(("[DETECTED→SEARCH] 미탐지", COLOR_ALERT))
                return lines

            if (detected_length is not None) and (detected_length >= WIDTH_MIN_FULL):
                self.state = "ALIGN"
                self.align_sub = "DIST_CHECK"
                self._reset_stabilizer()
                lines.append(("[DETECTED→ALIGN]", COLOR_META))
            else:
                self.state = "RECOVER"
                self.recover_sub = "DECIDE_TURN"
                self._reset_stabilizer()
                lines.append(("[DETECTED→RECOVER] 전면 확보 필요", COLOR_ALERT))
            return lines

        # ------------------------------
        # RECOVER
        # ------------------------------
        if self.state == "RECOVER":
            # 폭 충분해지면 CHECK로
            if det_ok and (detected_length is not None) and (detected_length >= WIDTH_MIN_FULL):
                self.state = "CHECK"
                self._reset_stabilizer()
                lines.append(("[RECOVER→CHECK]", COLOR_META))
                return lines

            # 탐지/전면 확보 기다리며 스핀
            self._exec("ROT_LEFT" if self._last_dir_sign > 0 else "ROT_RIGHT")
            self.cmd_status.start_timed(self._last_exec_cmd or "ROT", 0.0)
            lines.append(("[RECOVER] 확보 대기", COLOR_STATUS_TRK))
            return lines

        # ------------------------------
        # CHECK
        # ------------------------------
        if self.state == "CHECK":
            both_ok = yaw_ok and off_ok
            tag = "CHECK_OK" if both_ok else "CHECK_NOK"
            if self._stabilize(tag):
                if both_ok:
                    self.state = "DONE"
                    self._exec("STOP")
                    self.cmd_status.start_timed("STOP", 0.0)
                    lines.append(("[CHECK→DONE] 정렬 완료", COLOR_STATUS_OK))
                else:
                    self.state = "ALIGN"
                    self.align_sub = "DIST_CHECK"
                    lines.append(("[CHECK→ALIGN] 추가 정렬", COLOR_STATUS_TRK))
            else:
                lines.append((f"[CHECK] 검수중 [{self._stable_k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
            return lines

        # ------------------------------
        # ALIGN
        # ------------------------------
        if self.state == "ALIGN":
            if not det_ok:
                # 미탐지 시 스핀 대기
                self.align_sub = f"SPIN_{'LEFT' if self._last_dir_sign>0 else 'RIGHT'}_UNTIL_DETECTED"
                lines.append(("[ALIGN] 미탐지 → 스핀", COLOR_ALERT))
                return lines

            # 인터록 처리 (있으면 우선 종료될 때까지 대기)
            if self._maybe_finish_interlock():
                lines.append(("[ALIGN] STOP 인터록", COLOR_META))
                return lines

            # ---- DIST_CHECK ----
            if self.align_sub == "DIST_CHECK":
                if (dist_z is not None) and (dist_z > ALIGN_DIST_M):
                    tag = "DIST_FWD"
                    if self._stabilize(tag):
                        self.align_sub = "ALIGN_FWD_ADJUST"
                        self._reset_stabilizer()
                        lines.append(("[DIST_CHECK→ALIGN_FWD_ADJUST]", COLOR_META))
                    else:
                        lines.append(("[DIST_CHECK] 전진 판정 대기", COLOR_STATUS_TRK))
                    return lines

                if (dist_z is not None) and (dist_z < ALIGN_DIST_M):
                    tag = "DIST_BACK"
                    if self._stabilize(tag):
                        self.align_sub = "ALIGN_BWD_ADJUST"
                        self._reset_stabilizer()
                        lines.append(("[DIST_CHECK→ALIGN_BWD_ADJUST]", COLOR_META))
                    else:
                        lines.append(("[DIST_CHECK] 후진 판정 대기", COLOR_STATUS_TRK))
                    return lines

                if band_ok:
                    tag = "DIST_BAND_OK"
                    if self._stabilize(tag):
                        self.align_sub = "YAW_CHECK"
                        self._reset_stabilizer()
                        lines.append(("[DIST_CHECK→YAW_CHECK]", COLOR_META))
                    else:
                        lines.append(("[DIST_CHECK] 밴드 OK 대기", COLOR_STATUS_TRK))
                    return lines

                # 거리 정보 없음
                self._exec("STOP")
                self.cmd_status.start_timed("STOP", 0.0)
                lines.append(("[DIST_CHECK] distance N/A", COLOR_ALERT))
                return lines

            # ---- ALIGN_FWD_ADJUST ----
            if self.align_sub == "ALIGN_FWD_ADJUST":
                if (dist_z is not None) and (abs(dist_z - ALIGN_DIST_M) > ALIGN_BAND_M):
                    self._exec("FWD")
                    self.cmd_status.start_timed("FWD", 0.0)
                    lines.append(("[ALIGN_FWD_ADJUST] 전진", COLOR_STATUS_TRK))
                else:
                    # 다음 전이 전 STOP 인터록
                    self._start_interlock_then("DIST_CHECK")
                    lines.append(("[ALIGN_FWD_ADJUST→DIST_CHECK] (STOP)", COLOR_META))
                return lines

            # ---- ALIGN_BWD_ADJUST ----
            if self.align_sub == "ALIGN_BWD_ADJUST":
                if (dist_z is not None) and (abs(dist_z - ALIGN_DIST_M) > ALIGN_BAND_M):
                    self._exec("BACK")
                    self.cmd_status.start_timed("BACK", 0.0)
                    lines.append(("[ALIGN_BWD_ADJUST] 후진", COLOR_STATUS_TRK))
                else:
                    self._start_interlock_then("DIST_CHECK")
                    lines.append(("[ALIGN_BWD_ADJUST→DIST_CHECK] (STOP)", COLOR_META))
                return lines

            # ---- YAW_CHECK (yaw_smooth 기준) ----
            if self.align_sub == "YAW_CHECK":
                if (yaw_smooth is not None) and (abs(yaw_smooth) > YAW_TOL_DEG):
                    if yaw_smooth > 0:
                        tag = "YAW_POS"
                        if self._stabilize(tag):
                            self.align_sub = "ROTATE_RIGHT_UNTIL_YAW_TOL"
                            self._reset_stabilizer()
                            self._exec("ROT_RIGHT")
                            self._start_until_status("ROT_RIGHT", "|yaw|", abs(yaw_smooth), YAW_TOL_DEG)
                            lines.append(("[YAW_CHECK→ROTATE_RIGHT_UNTIL_YAW_TOL]", COLOR_ALERT))
                        else:
                            lines.append(("[YAW_CHECK] yaw>+tol 대기", COLOR_STATUS_TRK))
                        return lines
                    else:
                        tag = "YAW_NEG"
                        if self._stabilize(tag):
                            self.align_sub = "ROTATE_LEFT_UNTIL_YAW_TOL"
                            self._reset_stabilizer()
                            self._exec("ROT_LEFT")
                            self._start_until_status("ROT_LEFT", "|yaw|", abs(yaw_smooth), YAW_TOL_DEG)
                            lines.append(("[YAW_CHECK→ROTATE_LEFT_UNTIL_YAW_TOL]", COLOR_ALERT))
                        else:
                            lines.append(("[YAW_CHECK] yaw<-tol 대기", COLOR_STATUS_TRK))
                        return lines
                else:
                    tag = "YAW_OK"
                    if self._stabilize(tag):
                        self.align_sub = "OFFSET_CHECK"
                        self._reset_stabilizer()
                        lines.append(("[YAW_CHECK→OFFSET_CHECK]", COLOR_META))
                    else:
                        lines.append(("[YAW_CHECK] tol 이내 대기", COLOR_STATUS_TRK))
                    return lines

            # ---- ROTATE_*_UNTIL_YAW_TOL (yaw_smooth 기준) ----
            if self.align_sub in ("ROTATE_RIGHT_UNTIL_YAW_TOL", "ROTATE_LEFT_UNTIL_YAW_TOL"):
                code = "ROT_RIGHT" if self.align_sub.startswith("ROTATE_RIGHT") else "ROT_LEFT"
                self._exec(code)
                # 진행바: |yaw| → YAW_TOL_DEG
                if yaw_smooth is not None:
                    self._start_until_status(code, "|yaw|", abs(yaw_smooth), YAW_TOL_DEG)
                if yaw_ok:
                    self._start_interlock_then("OFFSET_CHECK")
                    lines.append((f"[{self.align_sub}→OFFSET_CHECK] (STOP)", COLOR_META))
                else:
                    lines.append((f"[{self.align_sub}] 자세 보정 중", COLOR_STATUS_TRK))
                return lines

            # ---- OFFSET_CHECK ----
            if self.align_sub == "OFFSET_CHECK":
                if ox is None:
                    tag = "OFF_NA"
                    if self._stabilize(tag):
                        self.align_sub = "DIST_CHECK"
                        self._reset_stabilizer()
                        lines.append(("[OFFSET_CHECK→DIST_CHECK] offset N/A", COLOR_META))
                    else:
                        lines.append(("[OFFSET_CHECK] N/A 대기", COLOR_STATUS_TRK))
                    return lines

                if abs(ox) > OFF_TOL_M:
                    # 전진시간 산출 (피팅 함수)
                    self._fwd_sec_cached = fwd_sec_from_offset_piecewise(ox) if USE_PIECEWISE_FWD_FIT else 2.0
                    if ox > 0:
                        tag = "OFF_RIGHT"
                        if self._stabilize(tag):
                            # RIGHT(90) 체인 시작: rel_yaw 기준 설정
                            self._rel_yaw_ref = rel_yaw if rel_yaw is not None else 0.0
                            self.align_sub = "ALIGN_ROTATE_RIGHT"
                            self._reset_stabilizer()
                            self._exec("ROT_RIGHT")
                            # 진행: |rel_yaw|
                            cur = 0.0 if rel_yaw is None else abs(rel_yaw - self._rel_yaw_ref)
                            self._start_until_status("ROT_RIGHT", "|rel_yaw|", cur, REL_YAW_TARGET_DEG)
                            lines.append(("[OFFSET_CHECK→ALIGN_ROTATE_RIGHT] (FWD_SEC=%.2fs)" % self._fwd_sec_cached, COLOR_META))
                        else:
                            lines.append(("[OFFSET_CHECK] offset>+tol 대기", COLOR_STATUS_TRK))
                        return lines
                    else:
                        tag = "OFF_LEFT"
                        if self._stabilize(tag):
                            # LEFT(90) 체인 시작: rel_yaw 기준 설정
                            self._rel_yaw_ref = rel_yaw if rel_yaw is not None else 0.0
                            self.align_sub = "ALIGN_ROTATE_LEFT"
                            self._reset_stabilizer()
                            self._exec("ROT_LEFT")
                            cur = 0.0 if rel_yaw is None else abs(rel_yaw - self._rel_yaw_ref)
                            self._start_until_status("ROT_LEFT", "|rel_yaw|", cur, REL_YAW_TARGET_DEG)
                            lines.append(("[OFFSET_CHECK→ALIGN_ROTATE_LEFT] (FWD_SEC=%.2fs)" % self._fwd_sec_cached, COLOR_META))
                        else:
                            lines.append(("[OFFSET_CHECK] offset<-tol 대기", COLOR_STATUS_TRK))
                        return lines
                else:
                    # |offset| ≤ tol
                    if (yaw_smooth is not None) and (abs(yaw_smooth) > YAW_TOL_DEG):
                        tag = "OFF_OK_YAW_NOK"
                        if self._stabilize(tag):
                            self.align_sub = "DIST_CHECK"
                            self._reset_stabilizer()
                            lines.append(("[OFFSET_CHECK→DIST_CHECK] yaw 재보정", COLOR_META))
                        else:
                            lines.append(("[OFFSET_CHECK] yaw>tol 대기", COLOR_STATUS_TRK))
                        return lines
                    else:
                        tag = "READY_TO_DONE"
                        if self._stabilize(tag):
                            self.align_sub = "READY_TO_DONE"
                            self._reset_stabilizer()
                            lines.append(("[OFFSET_CHECK→READY_TO_DONE]", COLOR_STATUS_TRK))
                        else:
                            lines.append(("[OFFSET_CHECK] 완료 대기", COLOR_STATUS_TRK))
                        return lines

            # ---- RIGHT branch (rel_yaw 기준) ----
            if self.align_sub == "ALIGN_ROTATE_RIGHT":
                self._exec("ROT_RIGHT")
                if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                    rel = abs(rel_yaw - self._rel_yaw_ref)
                    self._start_until_status("ROT_RIGHT", "|rel_yaw|", rel, REL_YAW_TARGET_DEG)
                    if rel >= REL_YAW_TARGET_DEG:
                        # STOP → FORWARD_AFTER_RIGHT
                        self._start_interlock_then("FORWARD_AFTER_RIGHT")
                        lines.append(("[ALIGN_ROTATE_RIGHT→FORWARD_AFTER_RIGHT] (STOP)", COLOR_META))
                else:
                    lines.append(("[ALIGN_ROTATE_RIGHT] rel_yaw N/A", COLOR_ALERT))
                return lines

            if self.align_sub == "FORWARD_AFTER_RIGHT":
                self._exec("FWD")
                # 첫 진입 시 타이머 시작
                if not self._timer_active():
                    self._start_timer(self._fwd_sec_cached)
                remain = max(0.0, self._timer_until - time.time())
                self.cmd_status.start_timed("FWD", remain)
                lines.append((f"[FORWARD_AFTER_RIGHT] 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))
                if not self._timer_active():
                    # STOP → ALIGN_ROTATE_LEFT_90
                    self._start_interlock_then("ALIGN_ROTATE_LEFT_90")
                    lines.append(("[FORWARD_AFTER_RIGHT→ALIGN_ROTATE_LEFT_90] (STOP)", COLOR_META))
                return lines

            if self.align_sub == "ALIGN_ROTATE_LEFT_90":
                # 재기준: 첫 프레임에서만 설정(되도록 rel_yaw 기준)
                if self._rel_yaw_ref is None and (rel_yaw is not None):
                    self._rel_yaw_ref = rel_yaw
                self._exec("ROT_LEFT")
                if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                    rel = abs(rel_yaw - self._rel_yaw_ref)
                    self._start_until_status("ROT_LEFT", "|rel_yaw|", rel, REL_YAW_TARGET_DEG)
                    if rel >= REL_YAW_TARGET_DEG:
                        # 체인 종료 → YAW_CHECK  (수정)
                        self._start_interlock_then("YAW_CHECK")
                        # 다음 사이클을 위해 기준 초기화
                        self._rel_yaw_ref = None
                        lines.append(("[ALIGN_ROTATE_LEFT_90→YAW_CHECK] (STOP)", COLOR_META))
                else:
                    lines.append(("[ALIGN_ROTATE_LEFT_90] rel_yaw N/A", COLOR_ALERT))
                return lines

            # ---- LEFT branch (rel_yaw 기준) ----
            if self.align_sub == "ALIGN_ROTATE_LEFT":
                self._exec("ROT_LEFT")
                if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                    rel = abs(rel_yaw - self._rel_yaw_ref)
                    self._start_until_status("ROT_LEFT", "|rel_yaw|", rel, REL_YAW_TARGET_DEG)
                    if rel >= REL_YAW_TARGET_DEG:
                        self._start_interlock_then("FORWARD_AFTER_LEFT")
                        lines.append(("[ALIGN_ROTATE_LEFT→FORWARD_AFTER_LEFT] (STOP)", COLOR_META))
                else:
                    lines.append(("[ALIGN_ROTATE_LEFT] rel_yaw N/A", COLOR_ALERT))
                return lines

            if self.align_sub == "FORWARD_AFTER_LEFT":
                self._exec("FWD")
                if not self._timer_active():
                    self._start_timer(self._fwd_sec_cached)
                remain = max(0.0, self._timer_until - time.time())
                self.cmd_status.start_timed("FWD", remain)
                lines.append((f"[FORWARD_AFTER_LEFT] 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))
                if not self._timer_active():
                    self._start_interlock_then("ALIGN_ROTATE_RIGHT_90")
                    lines.append(("[FORWARD_AFTER_LEFT→ALIGN_ROTATE_RIGHT_90] (STOP)", COLOR_META))
                return lines

            if self.align_sub == "ALIGN_ROTATE_RIGHT_90":
                if self._rel_yaw_ref is None and (rel_yaw is not None):
                    self._rel_yaw_ref = rel_yaw
                self._exec("ROT_RIGHT")
                if (rel_yaw is not None) and (self._rel_yaw_ref is not None):
                    rel = abs(rel_yaw - self._rel_yaw_ref)
                    self._start_until_status("ROT_RIGHT", "|rel_yaw|", rel, REL_YAW_TARGET_DEG)
                    if rel >= REL_YAW_TARGET_DEG:
                        # 체인 종료 → YAW_CHECK  (수정)
                        self._start_interlock_then("YAW_CHECK")
                        self._rel_yaw_ref = None
                        lines.append(("[ALIGN_ROTATE_RIGHT_90→YAW_CHECK] (STOP)", COLOR_META))
                else:
                    lines.append(("[ALIGN_ROTATE_RIGHT_90] rel_yaw N/A", COLOR_ALERT))
                return lines

            # ---- SPIN_*_UNTIL_DETECTED (fallback) ----
            if self.align_sub in ("SPIN_LEFT_UNTIL_DETECTED", "SPIN_RIGHT_UNTIL_DETECTED"):
                code = "ROT_LEFT" if "LEFT" in self.align_sub else "ROT_RIGHT"
                self._exec(code)
                self.cmd_status.start_timed(code, 0.0)
                lines.append((f"[{self.align_sub}] 탐지 대기", COLOR_STATUS_TRK))
                return lines

            # ---- READY_TO_DONE ----
            if self.align_sub == "READY_TO_DONE":
                self.state = "DONE"
                self._exec("STOP")
                self.cmd_status.start_timed("STOP", 0.0)
                lines.append(("[ALIGN→DONE] 정렬 완료", COLOR_STATUS_OK))
                return lines

        # ------------------------------
        # DONE
        # ------------------------------
        if self.state == "DONE":
            self._exec("STOP")
            self.cmd_status.start_timed("STOP", 0.0)
            lines.append(("[DONE] 정렬 완료 유지", COLOR_STATUS_OK))
            return lines

        # fallback
        self._exec("STOP")
        self.cmd_status.start_timed("STOP", 0.0)
        lines.append((f"[{self.state}] 대기", COLOR_META))
        return lines
