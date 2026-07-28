# calib/fsm/align.py
# -----------------------------------------------------------------------------
# Snapshot-based ALIGN micro-FSM.
#
# 핵심 원칙: "한 번 보고 눈을 감고 동작"
#   - *_CHECK 상태 진입 시점에 perception 값(ψ_pallet, d_lateral, d_forward)을
#     1회 snapshot 으로 캡처하고, 그 스냅샷에서 다음 cmd 시퀀스의 모든 파라미터를
#     사전 compute 해서 plan(LateralPlan/FwdAdjustPlan/...)에 캐싱.
#   - cmd 실행 상태 (YAW_CORRECT_*, LATERAL_ROTATE_*, FORWARD_AFTER_*, *_BACK,
#     ALIGN_FWD_ADJUST, ALIGN_BWD_ADJUST, INSERT) 진행 중에는 perception 무시.
#     IMU rel_yaw 누적 또는 timer 등 "종료 신호"만 점검.
#   - cmd 완료 시 다음 *_CHECK 상태로 복귀하여 다시 snapshot.
#
# 다이어그램 (사용자 확정):
#   [*] → YAW_CHECK
#   YAW_CHECK (snapshot ψ, d_lat):
#     ψ > +tol             → YAW_CORRECT_RIGHT
#     ψ < -tol             → YAW_CORRECT_LEFT
#     |ψ|≤tol ∧ |d_lat|≤tol → READY_TO_DONE   (직접 진입)
#     그 외 (|ψ|≤tol ∧ |d_lat|>tol) → OFFSET_CHECK
#   YAW_CORRECT_* → OFFSET_CHECK (IMU done)
#   OFFSET_CHECK (snapshot d_lat): |d_lat|>tol → DIST_CHECK; else → YAW_CHECK
#   DIST_CHECK (snapshot d_lat, d_fwd):
#     |d_lat|>tol                              → OFFSET_NEED_CORRECT
#     (d_fwd - ALIGN_DIST_M) > BAND            → ALIGN_FWD_ADJUST
#     (ALIGN_DIST_M - d_fwd) > BAND            → ALIGN_BWD_ADJUST
#     |d_fwd - ALIGN_DIST_M| ≤ BAND            → YAW_CHECK
#   ALIGN_FWD_ADJUST/ALIGN_BWD_ADJUST → DIST_CHECK
#   OFFSET_NEED_CORRECT → LATERAL_ROTATE_RIGHT/LEFT → FORWARD_AFTER_*
#                       → LATERAL_ROTATE_*_BACK → YAW_CHECK
#   READY_TO_DONE → (top.py 가 INSERT 로 전이)
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Tuple
import math
import time

import calib.config as cfg
from calib.config import (
    YAW_TOL_DEG, OFF_TOL_M,
    ALIGN_DIST_M, ALIGN_BAND_M,
    STOP_SEC,
    USE_PIECEWISE_FWD_FIT,
    COLOR_STATUS_TRK, COLOR_ALERT, COLOR_META, COLOR_STATUS_OK,
)
from calib.motion_models import fwd_sec_from_offset_piecewise
from .utils import (
    Stabilizer, SimpleTimer,
    CheckSnapshot, LateralPlan, FwdAdjustPlan, YawCorrectPlan, InsertPlan,
)
from .status_helper import StatusHelper
from .commands import CommandExecutor


def _wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _fwd_sec_for_distance(dist_m: float) -> float:
    """piecewise fit (offset→sec) 모델을 거리에도 재사용."""
    if not USE_PIECEWISE_FWD_FIT:
        return max(0.1, float(dist_m) * 4.0)
    return max(0.1, float(fwd_sec_from_offset_piecewise(abs(float(dist_m)))))


def _fwd_sec_for_insertion(dist_z: Optional[float]) -> Optional[float]:
    """파렛트 포켓 삽입용 전진 시간 — forklift body forward edge 기준 + safety margin.

    dist_z 는 fork center → pallet entry face 까지 거리 (snapshot 값).
    종료 시 forklift body forward edge (mast/fork carriage 전면) 가 pallet entry
    face 의 INSERT_BODY_SAFETY_M 앞에 멈춤. mast 가 pallet 정면 충돌 회피.
    fork tip 은 자연히 entry face 안쪽 forkLen 만큼 들어가 pallet 의 ~85% 침투.

    전진 거리 (fork center 이동량):
        total = dist_z - FORK_CENTER_TO_BODY_FRONT_M - INSERT_BODY_SAFETY_M

    config 의 FORK_CENTER_TO_BODY_FRONT_M / INSERT_BODY_SAFETY_M /
    INSERT_FWD_MPS / INS_FWD_{MIN,MAX}_SEC 사용.
    """
    if dist_z is None:
        return None
    body_offset   = float(getattr(cfg, "FORK_CENTER_TO_BODY_FRONT_M", 0.67))
    body_safety   = float(getattr(cfg, "INSERT_BODY_SAFETY_M", 0.05))
    v_mps         = float(getattr(cfg, "INSERT_FWD_MPS", 0.25))
    t_min         = float(getattr(cfg, "INS_FWD_MIN_SEC", 0.5))
    t_max         = float(getattr(cfg, "INS_FWD_MAX_SEC", 10.0))

    total = max(0.0, float(dist_z) - body_offset - body_safety)
    v = max(1e-3, v_mps)
    sec = total / v
    return max(t_min, min(t_max, sec))


class AlignMachine:
    """Snapshot-based align micro-FSM."""

    # cmd 실행 상태 (perception 무시, 종료 신호만 점검)
    CMD_STATES = frozenset({
        "YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT",
        "ALIGN_FWD_ADJUST", "ALIGN_BWD_ADJUST",
        "LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "LATERAL_ROTATE_LEFT_BACK",
        "LATERAL_ROTATE_LEFT",  "FORWARD_AFTER_LEFT",  "LATERAL_ROTATE_RIGHT_BACK",
        "INSERT",
    })

    # snapshot 캡처 상태
    CHECK_STATES = frozenset({"YAW_CHECK", "OFFSET_CHECK", "DIST_CHECK"})

    def __init__(self, execu: CommandExecutor, status: StatusHelper):
        self.execu = execu
        self.status = status

        # 진입 시 항상 YAW_CHECK 부터.
        self.sub: str = "YAW_CHECK"

        # 안정화 — *_CHECK 의 분기 결정 시 사용 (snapshot 한 번 캡처하기 전 노이즈 흡수)
        self._stb = Stabilizer()

        # 인터록(STOP) 타이머 — cmd 전환 사이 안전 STOP
        self._timer = SimpleTimer()
        self._interlock_active: bool = False
        self._after_interlock_sub: Optional[str] = None

        # snapshot / plan
        self._snapshot: Optional[CheckSnapshot] = None
        self._yaw_plan:     Optional[YawCorrectPlan] = None
        self._fwd_plan:     Optional[FwdAdjustPlan] = None
        self._lateral_plan: Optional[LateralPlan] = None
        self._insert_plan:  Optional[InsertPlan] = None

        # cmd 진행 상태(타이머/IMU 누적용)
        self._fwd_deadline_ts: Optional[float] = None
        self._insert_deadline_ts: Optional[float] = None

        # 최근 dist_z (READY_TO_DONE → INSERT 시 사용)
        self._latest_dist_z: Optional[float] = None

        # 명령 스팸 억제
        self._last_cmd: Optional[str] = None
        self._last_cmd_ts: float = 0.0

    # ------------------------------------------------------------------ utils

    def _exec(self, code: str):
        now = time.time()
        if self._last_cmd == code and (now - self._last_cmd_ts) < 0.10:
            return
        self.execu.exec(code)
        self._last_cmd, self._last_cmd_ts = code, now

    def _start_interlock_then(self, next_sub: str):
        self._interlock_active = True
        self._after_interlock_sub = next_sub
        self._exec("STOP")
        self._timer.start(STOP_SEC)
        self.status.start_timed("STOP", STOP_SEC)

    def _maybe_finish_interlock(self, lines) -> bool:
        if self._interlock_active:
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append(("[ALIGN] STOP 인터록 진행", COLOR_META))
            if not self._timer.active():
                self._interlock_active = False
                if self._after_interlock_sub:
                    self.sub = self._after_interlock_sub
                    self._after_interlock_sub = None
                    # CHECK 로 복귀할 때는 snapshot/plan 을 초기화하여 재 캡처 강제
                    if self.sub in self.CHECK_STATES:
                        self._snapshot = None
                        self._stb.reset()
            return True
        return False

    def reset(self, sub: str = "YAW_CHECK"):
        self.sub = sub
        self._stb.reset()
        self._interlock_active = False
        self._after_interlock_sub = None
        self._snapshot = None
        self._yaw_plan = None
        self._fwd_plan = None
        self._lateral_plan = None
        self._insert_plan = None
        self._fwd_deadline_ts = None
        self._insert_deadline_ts = None
        self._latest_dist_z = None

    # ---------------------------------------------------------- snapshot util

    def _take_snapshot(self,
                       psi_pallet_deg: Optional[float],
                       d_lateral_m: Optional[float],
                       d_forward_m: Optional[float]) -> Optional[CheckSnapshot]:
        """perception 값을 CheckSnapshot 으로 정규화. None 이면 캡처 실패."""
        if psi_pallet_deg is None or d_lateral_m is None:
            # forward 는 일부 CHECK 에서 None 허용 (YAW_CHECK 등)
            if psi_pallet_deg is None and d_lateral_m is None:
                return None
        snap = CheckSnapshot(
            psi_pallet_deg=float(psi_pallet_deg) if psi_pallet_deg is not None else 0.0,
            d_lateral_m=float(d_lateral_m) if d_lateral_m is not None else 0.0,
            d_forward_m=float(d_forward_m) if d_forward_m is not None else 0.0,
            ts=time.time(),
        )
        return snap

    def _rel_yaw_delta(self, rel_yaw: Optional[float], ref: Optional[float]) -> Optional[float]:
        if rel_yaw is None or ref is None:
            return None
        return _wrap_to_180(float(rel_yaw) - float(ref))

    # =========================================================== MAIN STEP

    def step(self,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw: Optional[float],   # = ψ_pallet (deg) — 팔레트 정면 yaw
             ox: Optional[float],    # = d_lateral (m)
             rel_yaw: Optional[float]  # IMU 상대 yaw (deg)
             ) -> List[Tuple[str, tuple]]:

        lines: List[Tuple[str, tuple]] = []

        # 최신 dist_z 캐시 (perception live — snapshot 캡처용)
        if dist_z is not None:
            self._latest_dist_z = dist_z

        # HUD 진행도 (only CHECK 상태에서만 의미가 있음 — cmd 상태는 plan 기반 caption)
        try:
            if self.sub in self.CHECK_STATES:
                self.status.update_until_metric(yaw, ox)
        except Exception:
            pass

        # 미탐지 처리: CHECK 상태일 때만 스핀 대기. cmd 상태는 perception 무시.
        if not det_ok and self.sub in self.CHECK_STATES:
            code = "ROT_LEFT" if self.execu.last_dir > 0 else "ROT_RIGHT"
            self._exec(code)
            self.status.start_timed(code, 0.0)
            lines.append(("[ALIGN] 미탐지 → 스핀 대기", COLOR_ALERT))
            return lines

        # 인터록 우선 처리
        if self._maybe_finish_interlock(lines):
            return lines

        # ------------------------------------------------------------------
        # ====== CHECK STATES (snapshot 캡처 → 분기 결정 → plan compute) =====
        # ------------------------------------------------------------------
        if self.sub == "YAW_CHECK":
            return self._step_yaw_check(yaw, ox, dist_z, rel_yaw, lines)

        if self.sub == "OFFSET_CHECK":
            return self._step_offset_check(yaw, ox, dist_z, rel_yaw, lines)

        if self.sub == "DIST_CHECK":
            return self._step_dist_check(yaw, ox, dist_z, rel_yaw, lines)

        # ------------------------------------------------------------------
        # ====== CMD STATES (perception 무시, 종료 신호만 점검) =============
        # ------------------------------------------------------------------
        if self.sub in ("YAW_CORRECT_RIGHT", "YAW_CORRECT_LEFT"):
            return self._step_yaw_correct(rel_yaw, lines)

        if self.sub == "ALIGN_FWD_ADJUST":
            return self._step_fwd_adjust(lines, direction=+1)

        if self.sub == "ALIGN_BWD_ADJUST":
            return self._step_fwd_adjust(lines, direction=-1)

        if self.sub in (
            "LATERAL_ROTATE_RIGHT", "FORWARD_AFTER_RIGHT", "LATERAL_ROTATE_LEFT_BACK",
            "LATERAL_ROTATE_LEFT",  "FORWARD_AFTER_LEFT",  "LATERAL_ROTATE_RIGHT_BACK",
        ):
            return self._step_lateral_chain(rel_yaw, lines)

        if self.sub == "INSERT":
            return self._step_insert(lines)

        if self.sub == "READY_TO_DONE":
            # top.py 가 ALIGN→INSERT 또는 ALIGN→DONE 전이시킬 상태.
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append(("[READY_TO_DONE] 정렬 완료(상위에서 INSERT/DONE 전이)", COLOR_STATUS_OK))
            return lines

        # fallback
        self._exec("STOP")
        self.status.start_timed("STOP", 0.0)
        lines.append((f"[{self.sub}] 대기", COLOR_META))
        return lines

    # =================================================== CHECK step 구현부

    def _step_yaw_check(self, yaw, ox, dist_z, rel_yaw, lines):
        """YAW_CHECK: snapshot 캡처 후 분기.

        ψ > +tol  → YAW_CORRECT_RIGHT
        ψ < -tol  → YAW_CORRECT_LEFT
        |ψ|≤tol ∧ |d_lat|≤tol → READY_TO_DONE (직접 진입)
        |ψ|≤tol ∧ |d_lat|>tol → OFFSET_CHECK
        """
        self._exec("STOP")
        self.status.start_timed("STOP", 0.0)

        if yaw is None:
            lines.append(("[YAW_CHECK] yaw N/A — perception 대기", COLOR_ALERT))
            return lines

        # snapshot 1회 캡처 (stable 후)
        if abs(yaw) > YAW_TOL_DEG:
            tag = "YAW_POS" if yaw > 0 else "YAW_NEG"
            if not self._stb.stable(tag):
                lines.append((f"[YAW_CHECK] {'우' if yaw>0 else '좌'} 회전 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
                return lines
            self._snapshot = self._take_snapshot(yaw, ox, dist_z)
            psi_abs = abs(self._snapshot.psi_pallet_deg) if self._snapshot else abs(yaw)
            psi_abs = max(float(getattr(cfg, "YAW_TURN_MIN_DEG", 1.0)), psi_abs)
            direction = +1 if yaw > 0 else -1
            self._yaw_plan = YawCorrectPlan(
                direction=direction,
                yaw_abs_deg=psi_abs,
                rel_yaw_ref=float(rel_yaw) if rel_yaw is not None else None,
            )
            self.sub = "YAW_CORRECT_RIGHT" if direction > 0 else "YAW_CORRECT_LEFT"
            self._stb.reset()
            code = "ROT_RIGHT" if direction > 0 else "ROT_LEFT"
            self._exec(code)
            self.status.start_until(code, "rel_yaw", 0.0, psi_abs * direction)
            lines.append((f"[YAW_CHECK→{self.sub}] ψ_snap={self._snapshot.psi_pallet_deg:+.2f}° |ψ|={psi_abs:.2f}°", COLOR_META))
            return lines

        # |ψ|≤tol — d_lateral 점검
        if ox is None:
            lines.append(("[YAW_CHECK] offset N/A — OFFSET_CHECK 로 위임", COLOR_STATUS_TRK))
            if self._stb.stable("YAW_OK_OFF_NA"):
                self._snapshot = self._take_snapshot(yaw, ox, dist_z)
                self.sub = "OFFSET_CHECK"
                self._stb.reset()
            return lines

        if abs(ox) <= OFF_TOL_M:
            # 직접 READY_TO_DONE 진입
            if self._stb.stable("YAW_OK_OFF_OK"):
                self._snapshot = self._take_snapshot(yaw, ox, dist_z)
                self.sub = "READY_TO_DONE"
                self._stb.reset()
                lines.append(("[YAW_CHECK→READY_TO_DONE] yaw OK ∧ offset OK", COLOR_STATUS_OK))
            else:
                lines.append((f"[YAW_CHECK] READY 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        # |ψ|≤tol ∧ |d_lat|>tol → OFFSET_CHECK
        if self._stb.stable("YAW_OK_OFF_NOK"):
            self._snapshot = self._take_snapshot(yaw, ox, dist_z)
            self.sub = "OFFSET_CHECK"
            self._stb.reset()
            lines.append(("[YAW_CHECK→OFFSET_CHECK] yaw OK, offset 보정 필요", COLOR_META))
        else:
            lines.append((f"[YAW_CHECK] OFFSET 분기 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
        return lines

    def _step_offset_check(self, yaw, ox, dist_z, rel_yaw, lines):
        """OFFSET_CHECK: snapshot 재캡처 — |d_lat|>tol 면 DIST_CHECK 로."""
        self._exec("STOP")
        self.status.start_timed("STOP", 0.0)

        if ox is None:
            if self._stb.stable("OFF_NA"):
                self.sub = "DIST_CHECK"
                self._stb.reset()
                self._snapshot = None
                lines.append(("[OFFSET_CHECK→DIST_CHECK] offset N/A", COLOR_META))
            else:
                lines.append((f"[OFFSET_CHECK] N/A 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        if abs(ox) > OFF_TOL_M:
            if self._stb.stable("OFF_NOK"):
                self._snapshot = self._take_snapshot(yaw, ox, dist_z)
                self.sub = "DIST_CHECK"
                self._stb.reset()
                lines.append((f"[OFFSET_CHECK→DIST_CHECK] |d_lat|={abs(ox):.3f}m > tol", COLOR_META))
            else:
                lines.append((f"[OFFSET_CHECK] 보정 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        # |d_lat|≤tol — 다이어그램상 YAW_CHECK 로 복귀 (yaw 재점검)
        if self._stb.stable("OFF_OK"):
            self.sub = "YAW_CHECK"
            self._stb.reset()
            self._snapshot = None
            lines.append(("[OFFSET_CHECK→YAW_CHECK] offset OK", COLOR_META))
        else:
            lines.append((f"[OFFSET_CHECK] OK 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
        return lines

    def _step_dist_check(self, yaw, ox, dist_z, rel_yaw, lines):
        """DIST_CHECK: snapshot d_lat & d_forward 기반으로 4 갈래 분기."""
        self._exec("STOP")
        self.status.start_timed("STOP", 0.0)

        # snapshot 갱신 (DIST_CHECK 진입 시 매번 d_lat 재평가 — 사용자 결정)
        snap = self._take_snapshot(yaw, ox, dist_z)
        if snap is None or dist_z is None:
            if self._stb.stable("DIST_NA"):
                lines.append(("[DIST_CHECK] perception N/A — 대기", COLOR_ALERT))
            else:
                lines.append((f"[DIST_CHECK] N/A 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        d_lat = snap.d_lateral_m
        d_fwd = snap.d_forward_m

        # 1) offset 재검 — 우선순위 최상
        if abs(d_lat) > OFF_TOL_M:
            if self._stb.stable("DIST→OFFSET_NEED"):
                self._snapshot = snap
                self._enter_offset_need_correct(rel_yaw, lines)
            else:
                lines.append((f"[DIST_CHECK] OFFSET 보정 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        # 2) forward 조정
        delta = d_fwd - ALIGN_DIST_M
        if delta > ALIGN_BAND_M:
            if self._stb.stable("DIST_FWD_ADJUST"):
                self._snapshot = snap
                self._fwd_plan = FwdAdjustPlan(
                    direction=+1,
                    distance_m=float(delta),
                    fwd_sec=_fwd_sec_for_distance(delta),
                )
                self.sub = "ALIGN_FWD_ADJUST"
                self._fwd_deadline_ts = None
                self._stb.reset()
                lines.append((f"[DIST_CHECK→ALIGN_FWD_ADJUST] d_fwd={d_fwd:.3f} Δ={delta:.3f} t={self._fwd_plan.fwd_sec:.2f}s", COLOR_META))
            else:
                lines.append((f"[DIST_CHECK] 전진 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        if -delta > ALIGN_BAND_M:
            if self._stb.stable("DIST_BWD_ADJUST"):
                self._snapshot = snap
                self._fwd_plan = FwdAdjustPlan(
                    direction=-1,
                    distance_m=float(-delta),
                    fwd_sec=_fwd_sec_for_distance(-delta),
                )
                self.sub = "ALIGN_BWD_ADJUST"
                self._fwd_deadline_ts = None
                self._stb.reset()
                lines.append((f"[DIST_CHECK→ALIGN_BWD_ADJUST] d_fwd={d_fwd:.3f} Δ={-delta:.3f} t={self._fwd_plan.fwd_sec:.2f}s", COLOR_META))
            else:
                lines.append((f"[DIST_CHECK] 후진 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
            return lines

        # 3) 거리 OK — YAW_CHECK 로 복귀
        if self._stb.stable("DIST_OK"):
            self.sub = "YAW_CHECK"
            self._stb.reset()
            self._snapshot = None
            lines.append((f"[DIST_CHECK→YAW_CHECK] dist band OK (Δ={delta:+.3f}m)", COLOR_META))
        else:
            lines.append((f"[DIST_CHECK] band OK 판정 대기 [{self._stb.k}/{self._stb.thr}]", COLOR_STATUS_TRK))
        return lines

    def _enter_offset_need_correct(self, rel_yaw, lines):
        """OFFSET_NEED_CORRECT 진입: snapshot 기반 LateralPlan compute → 회전 시작."""
        snap = self._snapshot
        if snap is None:
            lines.append(("[OFFSET_NEED_CORRECT] snapshot 없음 — abort", COLOR_ALERT))
            self.sub = "YAW_CHECK"
            return

        d_lat = snap.d_lateral_m
        psi_abs = max(float(getattr(cfg, "YAW_TURN_MIN_DEG", 1.0)), abs(snap.psi_pallet_deg))
        # 마진/언더로테이션 보정 후의 보장 회전량 (사용자가 ψ 가 너무 작으면 좌우 보정이 불가
        # 하므로 보정 시에는 최소 LATERAL_BACK_YAW_DEG 만큼은 회전하도록 floor 적용)
        lat_back = float(getattr(cfg, "LATERAL_BACK_YAW_DEG", 85.0))
        first_yaw = max(psi_abs, lat_back) if abs(psi_abs - lat_back) < 1e-3 else psi_abs

        fwd_sec = fwd_sec_from_offset_piecewise(d_lat) if USE_PIECEWISE_FWD_FIT else 2.0
        fwd_sec = max(0.1, float(fwd_sec))

        # d_lat < -tol (음수) → 우측 보정 → ROT_RIGHT 먼저
        # d_lat > +tol (양수) → 좌측 보정 → ROT_LEFT 먼저
        # 사용자 명세: "d_lateral < OFF_TOL_M (음수, 우측 보정)" → 회전 LEFT? Actually 다이어그램 그대로:
        #   d_lat<−tol → LATERAL_ROTATE_RIGHT (ROT_RIGHT)
        #   d_lat>+tol → LATERAL_ROTATE_LEFT  (ROT_LEFT)
        if d_lat < 0:
            direction = +1   # ROT_RIGHT
            self._lateral_plan = LateralPlan(
                direction=+1,
                yaw_abs_deg=first_yaw,
                fwd_sec=fwd_sec,
                back_yaw_deg=lat_back,
                rel_yaw_ref=float(rel_yaw) if rel_yaw is not None else None,
            )
            self.sub = "LATERAL_ROTATE_RIGHT"
            self._exec("ROT_RIGHT")
            self.status.start_until("ROT_RIGHT", "rel_yaw", 0.0, +first_yaw)
            lines.append((f"[DIST_CHECK→LATERAL_ROTATE_RIGHT] d_lat={d_lat:+.3f} ψ_snap={snap.psi_pallet_deg:+.2f} t_fwd={fwd_sec:.2f}s", COLOR_META))
        else:
            direction = -1   # ROT_LEFT
            self._lateral_plan = LateralPlan(
                direction=-1,
                yaw_abs_deg=first_yaw,
                fwd_sec=fwd_sec,
                back_yaw_deg=lat_back,
                rel_yaw_ref=float(rel_yaw) if rel_yaw is not None else None,
            )
            self.sub = "LATERAL_ROTATE_LEFT"
            self._exec("ROT_LEFT")
            self.status.start_until("ROT_LEFT", "rel_yaw", 0.0, -first_yaw)
            lines.append((f"[DIST_CHECK→LATERAL_ROTATE_LEFT] d_lat={d_lat:+.3f} ψ_snap={snap.psi_pallet_deg:+.2f} t_fwd={fwd_sec:.2f}s", COLOR_META))

    # =================================================== CMD step 구현부

    def _step_yaw_correct(self, rel_yaw, lines):
        """YAW_CORRECT_RIGHT/LEFT — IMU rel_yaw 누적이 plan.yaw_abs_deg 에 도달하면 OFFSET_CHECK."""
        plan = self._yaw_plan
        if plan is None:
            self._start_interlock_then("YAW_CHECK")
            lines.append((f"[{self.sub}] plan 없음 — 인터록", COLOR_ALERT))
            return lines

        code = "ROT_RIGHT" if plan.direction > 0 else "ROT_LEFT"
        self._exec(code)

        delta = self._rel_yaw_delta(rel_yaw, plan.rel_yaw_ref)
        target_signed = plan.yaw_abs_deg * plan.direction
        if delta is not None:
            self.status.start_until(code, "rel_yaw", delta, target_signed)
            # 종료 판정: 방향에 따라
            reached = (delta >= +plan.yaw_abs_deg) if plan.direction > 0 else (delta <= -plan.yaw_abs_deg)
            if reached:
                self._yaw_plan = None
                self._start_interlock_then("OFFSET_CHECK")
                lines.append((f"[{self.sub}→OFFSET_CHECK] Δ={delta:+.2f}°/{target_signed:+.2f}° (인터록)", COLOR_META))
                return lines
            lines.append((f"[{self.sub}] {delta:+.2f}°/{target_signed:+.2f}°", COLOR_STATUS_TRK))
        else:
            lines.append((f"[{self.sub}] rel_yaw N/A", COLOR_ALERT))
        return lines

    def _step_fwd_adjust(self, lines, direction: int):
        """ALIGN_FWD_ADJUST / ALIGN_BWD_ADJUST — plan.fwd_sec 만큼 전/후진."""
        plan = self._fwd_plan
        if plan is None or plan.direction != direction:
            self._start_interlock_then("DIST_CHECK")
            lines.append((f"[{self.sub}] plan 불일치 — 인터록", COLOR_ALERT))
            return lines

        code = "FWD" if direction > 0 else "BACK"
        self._exec(code)
        if self._fwd_deadline_ts is None:
            self._fwd_deadline_ts = time.time() + float(plan.fwd_sec)
        remain = max(0.0, self._fwd_deadline_ts - time.time())
        self.status.start_timed(code, remain)
        lines.append((f"[{self.sub}] {('전진' if direction>0 else '후진')} ({remain:.1f}s)", COLOR_STATUS_TRK))

        if remain <= 0.0:
            self._fwd_plan = None
            self._fwd_deadline_ts = None
            self._start_interlock_then("DIST_CHECK")
            lines.append((f"[{self.sub}→DIST_CHECK] (인터록)", COLOR_META))
        return lines

    def _step_lateral_chain(self, rel_yaw, lines):
        """LATERAL_* 체인 — plan.direction 기준 4단계 (회전 / 전진 / 복귀 회전)."""
        plan = self._lateral_plan
        if plan is None:
            self._start_interlock_then("YAW_CHECK")
            lines.append((f"[{self.sub}] plan 없음 — 인터록", COLOR_ALERT))
            return lines

        # RIGHT 체인: ROT_RIGHT(|ψ|) → FWD(fwd_sec) → ROT_LEFT(back_yaw)
        # LEFT  체인: ROT_LEFT(|ψ|)  → FWD(fwd_sec) → ROT_RIGHT(back_yaw)

        sub = self.sub

        if sub in ("LATERAL_ROTATE_RIGHT", "LATERAL_ROTATE_LEFT"):
            sign = +1 if sub == "LATERAL_ROTATE_RIGHT" else -1
            code = "ROT_RIGHT" if sign > 0 else "ROT_LEFT"
            self._exec(code)
            delta = self._rel_yaw_delta(rel_yaw, plan.rel_yaw_ref)
            target_signed = plan.yaw_abs_deg * sign
            if delta is not None:
                self.status.start_until(code, "rel_yaw", delta, target_signed)
                reached = (delta >= +plan.yaw_abs_deg) if sign > 0 else (delta <= -plan.yaw_abs_deg)
                if reached:
                    next_sub = "FORWARD_AFTER_RIGHT" if sign > 0 else "FORWARD_AFTER_LEFT"
                    self._fwd_deadline_ts = None
                    self._start_interlock_then(next_sub)
                    lines.append((f"[{sub}→{next_sub}] Δ={delta:+.2f}°/{target_signed:+.2f}° (인터록)", COLOR_META))
                    return lines
                lines.append((f"[{sub}] {delta:+.2f}°/{target_signed:+.2f}°", COLOR_STATUS_TRK))
            else:
                lines.append((f"[{sub}] rel_yaw N/A", COLOR_ALERT))
            return lines

        if sub in ("FORWARD_AFTER_RIGHT", "FORWARD_AFTER_LEFT"):
            self._exec("FWD")
            if self._fwd_deadline_ts is None:
                self._fwd_deadline_ts = time.time() + float(plan.fwd_sec)
            remain = max(0.0, self._fwd_deadline_ts - time.time())
            self.status.start_timed("FWD", remain)
            lines.append((f"[{sub}] 전진 ({remain:.1f}s)", COLOR_STATUS_TRK))
            if remain <= 0.0:
                # 복귀 회전 기준점 재설정
                self._lateral_plan = LateralPlan(
                    direction=plan.direction,
                    yaw_abs_deg=plan.yaw_abs_deg,
                    fwd_sec=plan.fwd_sec,
                    back_yaw_deg=plan.back_yaw_deg,
                    rel_yaw_ref=float(rel_yaw) if rel_yaw is not None else None,
                )
                next_sub = "LATERAL_ROTATE_LEFT_BACK" if sub == "FORWARD_AFTER_RIGHT" else "LATERAL_ROTATE_RIGHT_BACK"
                self._fwd_deadline_ts = None
                self._start_interlock_then(next_sub)
                lines.append((f"[{sub}→{next_sub}] (인터록)", COLOR_META))
            return lines

        if sub in ("LATERAL_ROTATE_LEFT_BACK", "LATERAL_ROTATE_RIGHT_BACK"):
            sign = -1 if sub == "LATERAL_ROTATE_LEFT_BACK" else +1
            code = "ROT_LEFT" if sign < 0 else "ROT_RIGHT"
            self._exec(code)
            delta = self._rel_yaw_delta(rel_yaw, plan.rel_yaw_ref)
            target_signed = plan.back_yaw_deg * sign
            if delta is not None:
                self.status.start_until(code, "rel_yaw", delta, target_signed)
                reached = (delta <= -plan.back_yaw_deg) if sign < 0 else (delta >= +plan.back_yaw_deg)
                if reached:
                    self._lateral_plan = None
                    self._start_interlock_then("YAW_CHECK")
                    lines.append((f"[{sub}→YAW_CHECK] Δ={delta:+.2f}°/{target_signed:+.2f}° (인터록)", COLOR_META))
                    return lines
                lines.append((f"[{sub}] {delta:+.2f}°/{target_signed:+.2f}°", COLOR_STATUS_TRK))
            else:
                lines.append((f"[{sub}] rel_yaw N/A", COLOR_ALERT))
            return lines

        # unreachable
        return lines

    def _step_insert(self, lines):
        """INSERT — 포켓 삽입 전진. top.py 가 직접 enter_insert() 로 호출."""
        plan = self._insert_plan
        if plan is None:
            self._start_interlock_then("READY_TO_DONE")
            lines.append(("[INSERT] plan 없음 — 인터록", COLOR_ALERT))
            return lines

        self._exec("FWD")
        if self._insert_deadline_ts is None:
            self._insert_deadline_ts = time.time() + float(plan.fwd_sec)
        remain = max(0.0, self._insert_deadline_ts - time.time())
        self.status.start_timed("FWD", remain)
        lines.append((f"[INSERT] 포켓 삽입 ({remain:.1f}s)", COLOR_STATUS_TRK))
        if remain <= 0.0:
            self._insert_plan = None
            self._insert_deadline_ts = None
            self.sub = "READY_TO_DONE"
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append(("[INSERT→READY_TO_DONE] 완료", COLOR_STATUS_OK))
        return lines

    # ----------------------------------------------- top.py 가 호출하는 API

    def enter_insert(self) -> bool:
        """top.py 가 ALIGN.READY_TO_DONE 도달 후 호출. InsertPlan compute 후 INSERT 진입."""
        sec = _fwd_sec_for_insertion(self._latest_dist_z)
        if sec is None or sec <= 0.0:
            return False
        self._insert_plan = InsertPlan(fwd_sec=float(sec))
        self._insert_deadline_ts = None
        self.sub = "INSERT"
        return True
