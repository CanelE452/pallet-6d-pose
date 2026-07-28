# calib/utils.py
# 공통 유틸리티(숫자 포맷, 로깅, 방향 텍스트)

from typing import List, Tuple, Optional
from .config import COLOR_META

def fmt_m(v: float) -> str:
    return f"{v:+.3f}"

def fmt_deg(v: float) -> str:
    return f"{v:+.1f}"

def dir_to_text(turn_dir: int) -> str:
    if turn_dir > 0:
        return "좌회전"
    elif turn_dir < 0:
        return "우회전"
    return "정지"

def log_cmd(lines: List[Tuple[str, tuple]], cmd_text: str,
            yaw_smooth: Optional[float], offset_x: Optional[float], dist_z: Optional[float]):
    yaw_str = fmt_deg(yaw_smooth) if yaw_smooth is not None else "N/A"
    off_str = fmt_m(offset_x) if offset_x is not None else "N/A"
    z_str   = f"{dist_z:.2f}" if dist_z is not None else "N/A"
    msg = f"CMD: {cmd_text} | yaw={yaw_str} deg, offset_x={off_str} m, z={z_str} m"
    print(msg)
    lines.append((msg, COLOR_META))
