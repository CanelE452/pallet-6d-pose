# calib/fsm/status_helper.py
from __future__ import annotations
from typing import List, Tuple, Optional

from calib.command_status import CommandStatus
from calib.config import COLOR_STATUS_TRK

class StatusHelper:
    """HUD 표시용 CommandStatus 연동 및 헬퍼."""
    def __init__(self):
        self.cmd_status = CommandStatus()

    def start_timed(self, code: str, duration_sec: float):
        self.cmd_status.start_timed(code, duration_sec)

    def start_until(self, code: str, metric_name: str, current_value: float, target_value: float):
        self.cmd_status.start_until(code, metric_name, current_value, target_value)

    def update_until_metric(self, yaw: Optional[float], ox: Optional[float]):
        if self.cmd_status.mode != "until":
            return
        if self.cmd_status.metric_name == "|yaw|" and (yaw is not None):
            self.cmd_status.update_metric(abs(yaw))
        elif self.cmd_status.metric_name == "|offset_x|" and (ox is not None):
            self.cmd_status.update_metric(abs(ox))

    def hud_wait(self, lines: List[Tuple[str, tuple]], tag: str, k: int, thr: int):
        lines.append((f"[대기] {tag} [{k}/{thr}]", COLOR_STATUS_TRK))
