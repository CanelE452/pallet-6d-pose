"""리프터 모션 캘리브레이션 평가 — 회전(IMU closed-loop) / 거리(calib·eval).

워크플로(거리, ★보정1): drive_calib 수집 → fit_fwd_model refit → config 갱신 → drive_eval.
스펙: docs/superpowers/specs/2026-06-04-lifter-motion-eval-design.md
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EVAL_CSV = os.path.join(RESULTS_DIR, "motion_eval.csv")
CALIB_CSV = os.path.join(RESULTS_DIR, "calib_fwd.csv")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

EVAL_HEADER = ["timestamp", "kind", "target", "direction", "power",
               "imu_stop_deg", "imu_settled_deg", "elapsed_s", "cmd_time_s",
               "measured", "reached", "error", "note"]
CALIB_HEADER = ["timestamp", "T_sec", "d_measured", "direction", "power"]


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _append(path, header, row_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row_list)


def append_eval_row(path, d: dict):
    """measured-target 으로 error 자동계산 후 1행 append."""
    try:
        error = float(d["measured"]) - float(d["target"])
    except (TypeError, ValueError):
        error = ""
    row = {**d, "timestamp": _ts(), "error": error}
    _append(path, EVAL_HEADER, [row.get(k, "") for k in EVAL_HEADER])


def append_calib_row(path, *, T_sec, d_measured, direction, power):
    _append(path, CALIB_HEADER, [_ts(), T_sec, d_measured, direction, power])


# 백그라운드 관찰 raw 스키마 (회전·전진 공통). 움직임 코드와 분리된 RelYawReader 버퍼가 채움.
# depth 없음(회전 IMU 타이밍 보호). cmd_bytes = 실제 송신 CAN movement 8바이트(control과 동일).
RAW_HEADER = ["t_s", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
              "rel_yaw_deg", "state", "cmd", "cmd_bytes"]


def write_raw_motion(label: str, kind: str, rows):
    """관찰 시계열 raw 저장. rows: list[dict] (RAW_HEADER 키, reader.stop_recording() 반환). None → 빈칸.

    kind: "rotate" | "forward". 경로: results/raw/{label}_{kind}.csv
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{label}_{kind}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r.get(k, "") for k in RAW_HEADER])
    return path


from dataclasses import dataclass


@dataclass
class RotateResult:
    target_deg: float
    turn_dir: int
    imu_stop_deg: float
    imu_settled_deg: float
    elapsed_s: float
    reached: bool
    raw: list          # [(t, rel_yaw)]


def _move_bytes(name):
    """control.MOVEMENT_TEMPLATES(실배포 fsm과 동일)에서 movement 8바이트 → 공백구분 문자열."""
    from depth_cam.calib import control          # 실배포와 동일 모듈(읽기만)
    tpl = control.MOVEMENT_TEMPLATES.get(name)
    return " ".join(str(b) for b in tpl) if tpl else ""


def rotate_sequence(*, reader, target_deg, turn_dir, issue_rotate, issue_stop,
                    period_s=0.02, settle_s=1.5, timeout_s=20.0,
                    sleep=time.sleep, now=time.monotonic, stop_event=None):
    """회전 IMU closed-loop. stop 후 settle_s 동안 raw 연장 기록(★보정2).

    turn_dir: +1=좌(CCW), -1=우(CW). reader.rel_yaw 는 reset 기준 상대각.
    """
    reader.reset()
    t0 = now()
    raw = []
    imu_stop = 0.0
    reached = False
    while True:
        issue_rotate(turn_dir)
        delta = abs(reader.rel_yaw)
        t = now() - t0
        raw.append((round(t, 4), round(reader.rel_yaw, 4)))
        if delta >= target_deg:
            imu_stop = delta
            reached = True
            break
        if t > timeout_s:
            imu_stop = delta
            reached = False
            break
        if stop_event is not None and stop_event.is_set():
            imu_stop = delta
            reached = False
            break
        sleep(period_s)
    issue_stop()
    # ★보정2: stop 후 settle_s 동안 over-rotate 정착 기록
    t_settle_end = now() + settle_s
    while now() < t_settle_end:
        raw.append((round(now() - t0, 4), round(reader.rel_yaw, 4)))
        sleep(period_s)
    imu_settled = abs(reader.rel_yaw)
    return RotateResult(target_deg, turn_dir, imu_stop, imu_settled,
                        now() - t0, reached, raw)


def drive_sequence(*, duration_s, issue_drive, issue_stop,
                   period_s=0.02, sleep=time.sleep, now=time.monotonic, stop_event=None):
    """개루프 전진/후진 duration_s초 (raw 미기록·★보정3). 반환: 실제 경과초."""
    t0 = now()
    while now() - t0 < duration_s:
        if stop_event is not None and stop_event.is_set():
            break
        issue_drive()
        sleep(period_s)
    issue_stop()
    return now() - t0


import threading

from depth_cam.calib import control
from depth_cam.calib import motion_models
from eval.imu_yaw import RelYawReader

# 배치 고정 power (확인됨): 전진 델타60(byte67), 회전 byte118
POWER_FWD = 60
POWER_ROT = 118

_ESTOP = threading.Event()


def _estop_watch():
    """[space] 입력 시 즉시 STOP. (간단 stdin 감시 — 현장 콘솔용)"""
    try:
        import keyboard  # 설치돼 있으면 사용
        keyboard.add_hotkey("space", lambda: (_ESTOP.set(), control.issue_command_stop()))
        keyboard.wait()
    except Exception:
        pass  # keyboard 미설치 환경: estop은 Ctrl-C 로 대체


def _dir_to_turn(direction: str) -> int:
    return +1 if direction == "ccw" else -1   # ccw=좌=+1, cw=우=-1


def run_rotate(reader, target_deg, direction, timeout_s, do_measure=True):
    if reader is None or not reader.available():
        print("[SKIP] RealSense 미가용 — 회전 평가 불가"); return
    input(f"▶ 회전 {target_deg}° {direction} 시작 — 엔터: ")
    reader.start_recording(state="ROTATE", cmd="ROTATE",
                           cmd_bytes=_move_bytes("rotate_ccw" if direction == "ccw" else "rotate_cw"))
    res = rotate_sequence(                    # 움직임 코드(원본) — 손대지 않음
        reader=reader, target_deg=target_deg, turn_dir=_dir_to_turn(direction),
        issue_rotate=control.issue_command_rotate_in_place,
        issue_stop=control.issue_command_stop, timeout_s=timeout_s)
    rows = reader.stop_recording()            # 옆에서 베껴 적은 IMU 시계열
    label = time.strftime("%Y%m%dT%H%M%S")
    raw_path = write_raw_motion(label, "rotate", rows) if rows else None
    print(f"  imu@stop={res.imu_stop_deg:.2f}° imu@settle={res.imu_settled_deg:.2f}° "
          f"t={res.elapsed_s:.2f}s reached={res.reached}  raw→{raw_path}")
    measured = ""
    if do_measure:
        measured = float(input("  AprilTag 실측각(°)? "))
    append_eval_row(EVAL_CSV, {
        "kind": "rotate", "target": target_deg, "direction": direction, "power": POWER_ROT,
        "imu_stop_deg": round(res.imu_stop_deg, 3), "imu_settled_deg": round(res.imu_settled_deg, 3),
        "elapsed_s": round(res.elapsed_s, 3), "cmd_time_s": "",
        "measured": measured, "reached": res.reached, "note": ""})


def run_drive_calib(reader, T_sec, direction):
    issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward
    state = "FORWARD" if direction == "fwd" else "BACKWARD"
    input(f"▶ calib 전진 {T_sec}s {direction} 시작 — 엔터: ")
    rec = reader is not None and reader.available()
    if rec:
        reader.start_recording(state=state, cmd=state,
                               cmd_bytes=_move_bytes("forward" if direction == "fwd" else "backward"))
    elapsed = drive_sequence(duration_s=T_sec, issue_drive=issue,   # 움직임 코드(원본)
                             issue_stop=control.issue_command_stop)
    if rec:
        rows = reader.stop_recording()
        if rows:
            print(f"  raw→{write_raw_motion(time.strftime('%Y%m%dT%H%M%S'), 'forward', rows)}")
    d = float(input(f"  자 실측거리(m)? (T={elapsed:.2f}s) "))
    append_calib_row(CALIB_CSV, T_sec=round(elapsed, 3), d_measured=d, direction=direction, power=POWER_FWD)
    print(f"  → calib_fwd.csv 기록. refit: python eval/fit_fwd_model.py {CALIB_CSV} --dir {direction}")


def run_drive_eval(reader, D_m, direction, do_measure=True):
    t_cmd = motion_models.fwd_sec_from_offset_piecewise(D_m)
    issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward
    state = "FORWARD" if direction == "fwd" else "BACKWARD"
    input(f"▶ eval 전진 {D_m}m → {t_cmd:.2f}s {direction} (refit·config 갱신 후인지 확인!) — 엔터: ")
    rec = reader is not None and reader.available()
    if rec:
        reader.start_recording(state=state, cmd=state,
                               cmd_bytes=_move_bytes("forward" if direction == "fwd" else "backward"))
    elapsed = drive_sequence(duration_s=t_cmd, issue_drive=issue,   # 움직임 코드(원본)
                             issue_stop=control.issue_command_stop)
    if rec:
        rows = reader.stop_recording()
        if rows:
            print(f"  raw→{write_raw_motion(time.strftime('%Y%m%dT%H%M%S'), 'forward', rows)}")
    measured = ""
    if do_measure:
        measured = float(input("  자 실측거리(m)? "))
    append_eval_row(EVAL_CSV, {
        "kind": "drive_eval", "target": D_m, "direction": direction, "power": POWER_FWD,
        "imu_stop_deg": "", "imu_settled_deg": "", "elapsed_s": round(elapsed, 3),
        "cmd_time_s": round(t_cmd, 3), "measured": measured, "reached": True, "note": ""})


def _menu(reader, args):
    target_deg, rot_dir, drv_dir = 90.0, "ccw", "fwd"
    while True:
        print(f"\n목표: 회전 {target_deg}°/{rot_dir}  전진 {drv_dir}")
        c = input("[r]회전 [c]calib(T초) [f]eval(거리) [t]각도 [d]방향 [q]종료 > ").strip()
        if c == "q":
            break
        elif c == "r":
            run_rotate(reader, target_deg, rot_dir, args.timeout)
        elif c == "c":
            T = float(input("  calib 전진 시간 T(s)? "))
            run_drive_calib(reader, T, drv_dir)
        elif c == "f":
            D = float(input("  eval 거리 D(m)? "))
            run_drive_eval(reader, D, drv_dir)
        elif c == "t":
            target_deg = float(input("  회전 목표각(°)? "))
        elif c == "d":
            rot_dir = "cw" if rot_dir == "ccw" else "ccw"
            drv_dir = "back" if drv_dir == "fwd" else "fwd"


def main():
    ap = argparse.ArgumentParser(description="리프터 모션 캘리브레이션 평가")
    ap.add_argument("--rotate", type=float, help="회전 목표각(°) 1회 실행")
    ap.add_argument("--drive", type=float, help="eval 거리(m) 1회 실행")
    ap.add_argument("--calib", type=float, help="calib 전진 시간(s) 1회 실행")
    ap.add_argument("--dir", default="fwd", help="cw|ccw (회전) / fwd|back (전진)")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--no-measure", action="store_true", help="실측 입력 생략(자동값만 기록)")
    args = ap.parse_args()

    control.can_init(channel=args.channel)
    print(f"[CAN] init (mock={control.is_mock()})")
    # drive/calib 경로도 depth+IMU 시계열 로깅 위해 reader 항상 생성 (미가용이면 graceful)
    reader = RelYawReader()
    if reader.available():
        reader.start(); print("[IMU] RealSense OK")
    else:
        print("[IMU] RealSense 미가용 — 거리 전용(raw 미기록)")
    threading.Thread(target=_estop_watch, daemon=True).start()
    try:
        if args.rotate is not None:
            run_rotate(reader, args.rotate, args.dir, args.timeout, not args.no_measure)
        elif args.calib is not None:
            run_drive_calib(reader, args.calib, args.dir)
        elif args.drive is not None:
            run_drive_eval(reader, args.drive, args.dir, not args.no_measure)
        else:
            _menu(reader, args)
    finally:
        control.issue_command_stop()
        if reader is not None and reader.available():
            reader.stop()
        control.can_close()


if __name__ == "__main__":
    main()
