# calib/command_status.py
from dataclasses import dataclass, field
import time
from typing import Literal, Optional

Mode = Literal["timed", "until"]

# ── 한국어 라벨 매핑 ─────────────────────────────────────────────
# 기존 코드(ROT_CW_IN_PLACE 등)와 신규 코드(ROT_LEFT 등)를 모두 지원
KOR_LABELS = {
    # 기존 키 (하위 호환)
    "ROT_CW_IN_PLACE": "제자리우회전",
    "ROT_CCW_IN_PLACE": "제자리좌회전",
    "FORWARD": "전진",
    "BACKWARD": "후진",
    "FWD_TURN_CW": "전진+우회전",
    "FWD_TURN_CCW": "전진+좌회전",
    "BWD_TURN_CW": "후진+우회전",
    "BWD_TURN_CCW": "후진+좌회전",
    "STOP": "정지",

    # 신규 키 (표준 명령 키)
    "FWD": "전진",
    "BACK": "후진",
    "FWD_LEFT": "전진+좌회전",
    "FWD_RIGHT": "전진+우회전",
    "BACK_LEFT": "후진+좌회전",
    "BACK_RIGHT": "후진+우회전",
    "ROT_LEFT": "제자리좌회전",
    "ROT_RIGHT": "제자리우회전",

    # 탐지 대기 스핀(가독성용 표기)
    "SPIN_LEFT_UNTIL_DETECTED": "제자리 좌회전(탐지 대기)",
    "SPIN_RIGHT_UNTIL_DETECTED": "제자리 우회전(탐지 대기)",
}

@dataclass
class CommandStatus:
    # ── 공통 ─────────────────────────────────────────────────────
    code: str = "STOP"                    # 내부 코드(e.g., ROT_LEFT, FWD 등)
    label: str = "정지"                    # 표기용 한국어 라벨
    mode: Mode = "timed"                  # "timed" | "until"
    started_at: float = field(default_factory=time.time)

    # timed 모드
    duration_sec: Optional[float] = None  # 총 지속시간
    elapsed_sec: float = 0.0              # 내부 누적(매 조회 시 갱신 가능)

    # until 모드
    metric_name: Optional[str] = None     # 예: "|yaw|", "|offset_x|", "|rel_yaw|"
    current_value: Optional[float] = None
    target_value: Optional[float] = None

    # ── 체인(다단계) 진행 표기 ───────────────────────────────────
    # 예) ROTATE_LEFT (1/3) → FORWARD (2/3) → ROTATE_RIGHT (3/3)
    step_idx: Optional[int] = None        # 현재 단계 (1-base)
    total_steps: Optional[int] = None     # 전체 단계 수
    phase: Optional[str] = None           # 선택: "YAW_CHAIN", "RECOVER", 등

    # ── 초기화/리셋 유틸 ────────────────────────────────────────
    def reset(self, code: str = "STOP"):
        """상태를 초기화하고 STOP로 설정."""
        self._reset_common(code)
        self.mode = "timed"
        self.duration_sec = 0.0
        self._clear_until_fields()
        self.step_idx = None
        self.total_steps = None
        self.phase = None

    def _reset_common(self, code: str):
        self.code = code
        self.label = KOR_LABELS.get(code, code)
        self.started_at = time.time()
        self.elapsed_sec = 0.0

    def _clear_until_fields(self):
        self.metric_name = None
        self.current_value = None
        self.target_value = None

    # ── 외부 API: 시간 고정 동작 시작 ────────────────────────────
    def start_timed(self, code: str, duration_sec: float, *,
                    step_idx: Optional[int] = None,
                    total_steps: Optional[int] = None,
                    phase: Optional[str] = None):
        """고정 시간 동작 진행도: (경과/총시간)"""
        self._reset_common(code)
        self.mode = "timed"
        self.duration_sec = max(0.0, float(duration_sec))
        self._clear_until_fields()
        # 체인 정보
        self.step_idx = step_idx
        self.total_steps = total_steps
        self.phase = phase

    # ── 외부 API: 조건 만족형 동작 시작 ─────────────────────────
    def start_until(self, code: str, metric_name: str, current_value: float, target_value: float, *,
                    step_idx: Optional[int] = None,
                    total_steps: Optional[int] = None,
                    phase: Optional[str] = None):
        """조건 만족형 진행도: (현재 지표 / 목표 지표)"""
        self._reset_common(code)
        self.mode = "until"
        self.metric_name = metric_name
        self.current_value = float(current_value)
        self.target_value = float(target_value)
        self.duration_sec = None  # timed 아님
        # 체인 정보
        self.step_idx = step_idx
        self.total_steps = total_steps
        self.phase = phase

    # ── 진행 갱신 ───────────────────────────────────────────────
    def update_elapsed(self):
        """경과 시간을 now-started_at으로 갱신 (timed 모드에서 주로 사용)."""
        self.elapsed_sec = time.time() - self.started_at

    def update_metric(self, current_value: Optional[float] = None):
        """UNTIL 모드의 현재 측정값을 갱신."""
        if current_value is not None:
            self.current_value = float(current_value)

    # ── 타 UI/HUD 편의 프로퍼티(읽기 전용) ───────────────────────
    @property
    def t_total(self) -> Optional[float]:
        """총 시간(timed 전용)."""
        return self.duration_sec if self.mode == "timed" else None

    @property
    def t_elapsed(self) -> Optional[float]:
        """경과 시간(timed 전용). 조회 시 자동 갱신."""
        if self.mode != "timed":
            return None
        self.update_elapsed()
        return self.elapsed_sec

    @property
    def t_remain(self) -> Optional[float]:
        """잔여 시간(timed 전용)."""
        if self.mode != "timed":
            return None
        if self.duration_sec is None:
            return None
        self.update_elapsed()
        return max(0.0, float(self.duration_sec) - float(self.elapsed_sec))

    # ── 비율(0~1) 계산: HUD/Diagram 진행바용 ────────────────────
    def progress_ratio(self) -> Optional[float]:
        """
        0~1 사이의 진행 비율을 반환.
        - timed:   r = t_elapsed / t_total
        - until:   r = current / target
        """
        if self.mode == "timed":
            if not self.duration_sec or self.duration_sec <= 0:
                return None
            te = self.t_elapsed
            if te is None:
                return None
            r = te / float(self.duration_sec)
            return max(0.0, min(1.0, r))
        else:  # until
            if self.current_value is None or self.target_value in (None, 0):
                return None
            r = float(self.current_value) / float(self.target_value)
            return max(0.0, min(1.0, r))

    # ── HUD 표기 문자열 생성 ─────────────────────────────────────
    def _chain_prefix(self) -> str:
        """[1/3] 또는 [YAW_CHAIN 1/3] 같은 prefix 생성"""
        idx = self.step_idx
        tot = self.total_steps
        if idx is None or tot is None:
            return f"[{self.phase}] " if self.phase else ""
        if self.phase:
            return f"[{self.phase} {idx}/{tot}] "
        return f"[{idx}/{tot}] "

    def progress_text(self) -> str:
        """
        HUD에 표기할 진행도 문자열 생성
        - timed:   "(0.3s/2.0s)"
        - until:   "(현재 |yaw|=1.23 / 목표 |yaw|=2.00)"
        - prefix:  "[1/3] " 또는 "[YAW_CHAIN 1/3] "
        """
        prefix = self._chain_prefix()
        if self.mode == "timed" and self.duration_sec is not None:
            te = self.t_elapsed  # 내부적으로 update_elapsed 수행
            te = 0.0 if te is None else te
            return prefix + f"({te:.1f}s/{self.duration_sec:.1f}s)"
        elif (
            self.mode == "until"
            and self.metric_name is not None
            and self.current_value is not None
            and self.target_value is not None
        ):
            return prefix + (
                f"(현재 {self.metric_name}={self.current_value:.2f} "
                f"/ 목표 {self.metric_name}={self.target_value:.2f})"
            )
        return prefix.rstrip()  # prefix만이라도 표시

    def formatted_label(self) -> str:
        """라벨 + 진행 텍스트를 한 번에 반환 (HUD 한 줄 표기 용이)"""
        return f"{self.label} {self.progress_text()}".strip()
