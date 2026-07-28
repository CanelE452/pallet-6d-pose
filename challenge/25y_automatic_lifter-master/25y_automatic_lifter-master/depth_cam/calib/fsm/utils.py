# calib/fsm/utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union
import time

from calib.config import CMD_STABLE_THR


def within_band(val: Optional[float], target: float, band: float) -> bool:
    """값이 target±band 범위에 있는지 확인."""
    return (val is not None) and (abs(val - target) <= band)


class Stabilizer:
    """
    같은 '판정 태그(tag)'가 연속 thr 프레임 유지될 때 True를 반환하는 안정화기.
    - thr 미지정 시 config의 CMD_STABLE_THR을 사용.
    - stable(tag): 같은 tag가 연속 thr회 관측되면 True
    - reset(): 내부 카운터 초기화
    - k: 현재 연속 관측 프레임 수(읽기 전용)
    """
    def __init__(self, thr: Optional[int] = None):
        self._thr: int = int(thr) if thr is not None else int(CMD_STABLE_THR)
        self._tag: Optional[str] = None
        self._k: int = 0

    def reset(self) -> None:
        self._tag, self._k = None, 0

    @property
    def k(self) -> int:
        return self._k

    @property
    def thr(self) -> int:
        return self._thr

    def set_thr(self, thr: int) -> None:
        """런타임에 임계 프레임 수를 변경."""
        self._thr = int(thr)
        self.reset()

    def stable(self, tag: str) -> bool:
        """같은 tag가 연속 thr회 들어오면 True."""
        if self._tag == tag:
            self._k += 1
        else:
            self._tag, self._k = tag, 1
        return self._k >= self._thr


class SimpleTimer:
    """간단한 카운트다운 타이머 (초 기반)."""
    def __init__(self):
        self._until: float = 0.0

    def start(self, sec: float) -> None:
        self._until = time.time() + float(sec)

    def active(self) -> bool:
        return time.time() < self._until

    def remain(self) -> float:
        return max(0.0, self._until - time.time())


# ============================================================================
# Snapshot / SequencePlan 모델 — "한 번 보고 눈을 감고 동작"
# *_CHECK 진입 시점에 perception 값을 1회 캡처해 CheckSnapshot 으로 보존.
# 이후 cmd 시퀀스 진행 중에는 plan(LateralPlan/FwdAdjustPlan/...)만 사용.
# ============================================================================

@dataclass
class CheckSnapshot:
    """*_CHECK 진입 시 1회 캡처, 이후 cmd 시퀀스 내내 불변.

    Attributes:
        psi_pallet_deg: 팔레트 정면 yaw (deg). +면 우측, -면 좌측.
        d_lateral_m:    좌우 오프셋 (m). +면 우측 보정 필요, -면 좌측.
        d_forward_m:    전방 거리 (m). z방향 또는 평면 forward.
        ts:             캡처 시각 (time.time()).
    """
    psi_pallet_deg: float
    d_lateral_m: float
    d_forward_m: float
    ts: float


@dataclass
class YawCorrectPlan:
    """YAW_CHECK 에서 |ψ|>tol 일 때 산출. ψ 절댓값만큼 회전."""
    direction: Literal[-1, +1]      # -1=좌(ψ<0), +1=우(ψ>0)
    yaw_abs_deg: float              # 목표 회전량 |ψ_pallet|
    rel_yaw_ref: Optional[float]    # IMU rel_yaw 기준점 (진입 시 캐싱)


@dataclass
class FwdAdjustPlan:
    """ALIGN_FWD_ADJUST / ALIGN_BWD_ADJUST 용."""
    direction: Literal[-1, +1]      # +1=FWD, -1=BACK
    distance_m: float               # 목표 이동거리 |d_forward - ALIGN_DIST_M|
    fwd_sec: float                  # piecewise 모델로 환산한 시간


@dataclass
class LateralPlan:
    """OFFSET_NEED_CORRECT 진입 시 compute, *_BACK 단계까지 사용.

    체인: LATERAL_ROTATE_{dir} (|ψ|) → FORWARD_AFTER_{dir} (fwd_sec)
          → LATERAL_ROTATE_{opp}_BACK (back_yaw_deg) → YAW_CHECK
    """
    direction: Literal[-1, +1]      # -1=좌측 보정 (d_lateral>0), +1=우측 보정 (d_lateral<0)
    yaw_abs_deg: float              # 첫 회전 목표 (|ψ_pallet|)
    fwd_sec: float                  # FORWARD 단계 시간 (오프셋 보정용)
    back_yaw_deg: float             # 복귀 회전 목표 (보통 90° 또는 LATERAL_BACK_YAW_DEG)
    rel_yaw_ref: Optional[float]    # 현재 단계의 IMU rel_yaw 기준점


@dataclass
class InsertPlan:
    """READY_TO_DONE → INSERT 전이 시 compute."""
    fwd_sec: float                  # 포켓 삽입 전진 시간


SequencePlan = Union[YawCorrectPlan, FwdAdjustPlan, LateralPlan, InsertPlan]
