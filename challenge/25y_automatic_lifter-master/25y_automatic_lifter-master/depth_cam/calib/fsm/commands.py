# calib/fsm/commands.py
from __future__ import annotations
from typing import Optional

from calib.control import (
    issue_command_forward,
    issue_command_backward,
    issue_command_forward_and_turn,
    issue_command_backward_and_turn,
    issue_command_rotate_in_place,
    issue_command_stop,
)

# HUD/로그에서 사람이 읽기 쉬운 라벨
HUMAN_NAME = {
    "FWD": "전진",
    "BACK": "후진",
    "FWD_LEFT": "전진+좌회전",
    "FWD_RIGHT": "전진+우회전",
    "BACK_LEFT": "후진+좌회전",
    "BACK_RIGHT": "후진+우회전",
    "ROT_LEFT": "제자리 좌회전",
    "ROT_RIGHT": "제자리 우회전",
    "STOP": "정지",
    # 새 플로우 가독성용(실제 실행은 ROT_LEFT/ROT_RIGHT로 이행)
    "SPIN_LEFT_UNTIL_DETECTED": "제자리 좌회전(탐지 대기)",
    "SPIN_RIGHT_UNTIL_DETECTED": "제자리 우회전(탐지 대기)",
}

def humanize(cmd: str) -> str:
    """명령 키를 사람이 읽기 쉬운 한글 라벨로 변환."""
    return HUMAN_NAME.get(cmd, cmd)

class CommandExecutor:
    """
    중복 송신 방지/최근 회전 방향 추적을 포함한 명령 실행기.

    - 동일 명령이 연속으로 들어오면 실제 CAN 송신을 억제(스팸 방지).
    - last_dir: 마지막 '회전' 의미의 방향을 유지(+1=좌, -1=우).
      ROT_* / FWD_LEFT/RIGHT / BACK_LEFT/RIGHT에서 갱신됨.
    """
    def __init__(self):
        self._last_exec_cmd: Optional[str] = None
        self.last_dir: int = +1  # +1 좌, -1 우

    # --- 유틸 ---

    @property
    def last_cmd(self) -> Optional[str]:
        """마지막으로 실제 실행된 명령 키(중복 억제 이후)."""
        return self._last_exec_cmd

    def reset_last(self) -> None:
        """중복 억제 상태 초기화(강제로 동일 명령을 다시 송신하고 싶을 때 사용)."""
        self._last_exec_cmd = None

    # --- 실행 ---

    def exec(self, cmd: str) -> str:
        """
        명령 실행(중복 억제 포함). 실제로 실행된 명령 키를 반환.
        SPIN_* 계열은 가독성용 라벨이며 실제 제어는 ROT_*로 매핑됨.
        """
        # SPIN_*은 내부적으로 ROT_*로 실행하되, 라벨은 유지 가능
        mapped_cmd = cmd
        if cmd == "SPIN_LEFT_UNTIL_DETECTED":
            mapped_cmd = "ROT_LEFT"
        elif cmd == "SPIN_RIGHT_UNTIL_DETECTED":
            mapped_cmd = "ROT_RIGHT"

        # 동일 명령 반복 송신 방지
        if mapped_cmd == self._last_exec_cmd:
            return mapped_cmd

        self._last_exec_cmd = mapped_cmd

        if mapped_cmd == "FWD":
            issue_command_forward()

        elif mapped_cmd == "BACK":
            issue_command_backward()

        elif mapped_cmd == "FWD_LEFT":
            self.last_dir = +1
            issue_command_forward_and_turn(+1)

        elif mapped_cmd == "FWD_RIGHT":
            self.last_dir = -1
            issue_command_forward_and_turn(-1)

        elif mapped_cmd == "BACK_LEFT":
            self.last_dir = +1
            issue_command_backward_and_turn(+1)

        elif mapped_cmd == "BACK_RIGHT":
            self.last_dir = -1
            issue_command_backward_and_turn(-1)

        elif mapped_cmd == "ROT_LEFT":
            self.last_dir = +1
            issue_command_rotate_in_place(+1)

        elif mapped_cmd == "ROT_RIGHT":
            self.last_dir = -1
            issue_command_rotate_in_place(-1)

        elif mapped_cmd == "STOP":
            issue_command_stop()

        return mapped_cmd
