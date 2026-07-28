# calib/fsm/top.py
# -----------------------------------------------------------------------------
# Top-level FSM (snapshot-based redesign).
#
# 상태 흐름:
#   SEARCH → ALIGN (composite) → INSERT → DONE
#
# - SEARCH: 탐지 전 대기(정지). det_ok 가 True 면 즉시 ALIGN 진입.
# - ALIGN:  AlignMachine (snapshot-based).
#           내부적으로 sub == "READY_TO_DONE" 이 되면 상위가 INSERT 로 전이.
# - INSERT: AlignMachine.enter_insert() 로 InsertPlan compute 후
#           내부 sub == "INSERT" 로 변경. fwd_sec 경과 후 READY_TO_DONE 복귀
#           → DONE.
# - DONE:   STOP 유지.
#
# 다이어그램에서 RECOVER 와 OUTER 의 DETECTED / CHECK 는 제거됨.
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Tuple
import time

from calib.config import (
    COLOR_META, COLOR_ALERT, COLOR_STATUS_OK, COLOR_STATUS_TRK,
)
from .commands import CommandExecutor
from .status_helper import StatusHelper
from .align import AlignMachine


class CalibrationFSM:
    """
    Top-level: SEARCH → ALIGN → INSERT → DONE.

    하위 호환 속성:
      - self.state                : 현재 top-level 상태 (str)
      - self.align                : AlignMachine 인스턴스
      - self.recover              : 호환용 더미 (sub=None)
      - self.cmd_status           : StatusHelper.cmd_status (HUD 진행바용)
    """
    def __init__(self):
        self.state: str = "SEARCH"

        self.execu = CommandExecutor()
        self.status = StatusHelper()
        self.align = AlignMachine(self.execu, self.status)

        # 다이어그램 패널(diagram.py) 호환용 — recover 그룹은 비활성 더미.
        class _RecoverStub:
            sub = None
        self.recover = _RecoverStub()

        self._last_cmd: Optional[str] = None
        self._last_ts: float = 0.0

    # ------------------------------------------------------------------ utils

    def _ensure_stop(self) -> None:
        now = time.time()
        if self._last_cmd != "STOP" or (now - self._last_ts) > 0.2:
            self.execu.exec("STOP")
            self.status.start_timed("STOP", 0.0)
            self._last_cmd, self._last_ts = "STOP", now

    def get_command_status(self):
        return self.status.cmd_status

    @property
    def cmd_status(self):
        return self.status.cmd_status

    # ------------------------------------------------------------------ step

    def step(self,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw_smooth: Optional[float],
             offset_smooth: Optional[tuple],
             rel_yaw: Optional[float] = None,
             # ---------- 6D pose 입력 모드 (선택적, 우선순위 높음) ----------
             # 어댑터를 거치지 않고 main_rec.py 에서 이미 변환된 값을 넘기는 경우.
             psi_pallet_deg: Optional[float] = None,
             d_lateral_m: Optional[float] = None,
             d_forward_m: Optional[float] = None,
             ) -> List[Tuple[str, tuple]]:

        lines: List[Tuple[str, tuple]] = []

        # ---- perception 어댑팅: 6D pose 직접 입력이 있으면 우선 사용 ----
        if psi_pallet_deg is not None:
            yaw_for_align = psi_pallet_deg
        else:
            yaw_for_align = yaw_smooth

        if d_lateral_m is not None:
            ox = d_lateral_m
        else:
            ox = None
            if offset_smooth is not None:
                try:
                    ox = float(offset_smooth[0])
                except Exception:
                    ox = None

        dist_for_align = d_forward_m if d_forward_m is not None else dist_z

        # HUD 진행도 갱신 — until 모드의 metric_name 에 따라 yaw / offset / rel_yaw 매핑
        self.status.update_until_metric(yaw_for_align, ox)
        try:
            cs = self.status.cmd_status
            if getattr(cs, "mode", None) == "until" and getattr(cs, "metric_name", "") == "rel_yaw":
                if rel_yaw is not None:
                    cs.update_metric(float(rel_yaw))
        except Exception:
            pass

        # ----------------------------- SEARCH -----------------------------
        if self.state == "SEARCH":
            self.align.reset("YAW_CHECK")
            self.execu.reset_last()

            if det_ok:
                self.state = "ALIGN"
                lines.append(("[SEARCH→ALIGN] 탐지 성공", COLOR_META))
            else:
                self._ensure_stop()
                lines.append(("[SEARCH] 탐지 대기: 정지", COLOR_STATUS_TRK))
            return lines

        # ------------------------------ ALIGN -----------------------------
        if self.state == "ALIGN":
            aln_lines = self.align.step(
                det_ok, detected_length, dist_for_align,
                yaw_for_align, ox, rel_yaw,
            )
            lines.extend(aln_lines)

            if self.align.sub == "READY_TO_DONE":
                # InsertPlan compute → INSERT 진입 (성공 시) 또는 곧장 DONE
                if self.align.enter_insert():
                    self.state = "INSERT"
                    lines.append(("[ALIGN→INSERT] 정렬 완료, 포켓 삽입 시작", COLOR_META))
                else:
                    self.state = "DONE"
                    self._ensure_stop()
                    lines.append(("[ALIGN→DONE] 정렬 완료(삽입 스킵)", COLOR_STATUS_OK))
            return lines

        # ------------------------------ INSERT ----------------------------
        if self.state == "INSERT":
            aln_lines = self.align.step(
                det_ok, detected_length, dist_for_align,
                yaw_for_align, ox, rel_yaw,
            )
            lines.extend(aln_lines)

            if self.align.sub == "READY_TO_DONE":
                self.state = "DONE"
                self._ensure_stop()
                lines.append(("[INSERT→DONE] 포켓 삽입 완료", COLOR_STATUS_OK))
            return lines

        # ------------------------------- DONE -----------------------------
        if self.state == "DONE":
            self._ensure_stop()
            lines.append(("[DONE] 정렬 완료 상태 유지", COLOR_STATUS_OK))
            return lines

        # ----------------------------- Fallback ---------------------------
        self._ensure_stop()
        lines.append((f"[{self.state}] 대기", COLOR_META))
        return lines
