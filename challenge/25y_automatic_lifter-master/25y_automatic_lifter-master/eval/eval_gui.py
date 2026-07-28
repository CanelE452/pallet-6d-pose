"""리프터 모션 캘리브레이션 평가 — GUI (tkinter).

회전/거리(calib·eval)를 버튼으로 실행하고, 동작 후 실측값을 입력해 CSV에 기록.
AprilTag 검출/추론은 하지 않음 — 실측값(AprilTag 각·자 거리)은 사용자가 직접 입력.

실행: python eval/eval_gui.py
설계: docs/superpowers/specs/2026-06-04-lifter-motion-eval-design.md
"""
from __future__ import annotations
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_cam.calib import control, motion_models
from eval.imu_yaw import RelYawReader
from eval import eval_motion as em

HELP_TEXT = """\
■ 워크플로 (거리는 순서 중요)
  ① calib: 고정 시간(T초) 전진 → 자로 잰 거리 입력 (여러 T 반복)
  ② refit: 터미널에서
     python eval/fit_fwd_model.py eval/results/calib_fwd.csv --dir fwd
  ③ 위 출력값을 depth_cam/calib/config.py 의
     FWD_T0/FWD_T1/FWD_A 에 반영
  ④ eval(거리): refit·반영 후에만 실행 (held-out 검증)
  ※ refit 전에 eval 돌리면 옛 파라미터로 검증하는 사고.

■ 회전
  · IMU(gyro 적분)가 목표각에 도달했다고 "믿으면" 정지.
  · 정지 후 1~2초 더 기록 → 관성 over-rotate(imu@settle) 캡처.
  · imu@stop / imu@settle / AprilTag 실측 3값으로 오차 분해:
      settle − stop  = ECU 관성 over-rotate
      AprilTag − settle = 자이로 스케일/드리프트
  · RealSense(IMU) 없으면 회전 평가 불가(거리는 가능).

■ 실측 입력
  · 동작이 끝나면 [실측값]에 직접 입력 후 [기록 저장].
  · 회전=AprilTag 각(°), 거리=자로 잰 거리(m).

■ 비상정지
  · [■ 비상정지] 또는 Space 키 → 즉시 STOP + 동작 중단.

■ 구동 강도(power)는 단일 고정: 전진 60 / 회전 118.
"""


class MotionEvalGUI:
    def __init__(self, root):
        self.root = root
        root.title("리프터 모션 캘리브레이션 평가")
        self._stop_event = threading.Event()
        self._busy = False
        self._pending = None          # 저장 대기 컨텍스트(dict)

        control.can_init()
        self._can_mock = control.is_mock()
        self.reader = RelYawReader()
        self._imu_ok = self.reader.available()
        if self._imu_ok:
            self.reader.start()

        self._build()
        root.bind("<space>", lambda e: self.on_estop())
        self._poll_imu()

    # ---------- UI 구성 ----------
    def _build(self):
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=11)
        avail = set(tkfont.families())
        prefer = ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                  "Noto Sans CJK KR Regular", "Malgun Gothic", "UnDotum"]
        fam = next((p for p in prefer if p in avail), base.actual("family"))
        base.configure(family=fam)         # 한글 지원 폰트 우선 선택
        try:
            tkfont.nametofont("TkTextFont").configure(family=fam, size=11)
        except tk.TclError:
            pass
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.minsize(760, 540)

        left = ttk.Frame(self.root, padding=14)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=14)
        right.grid(row=0, column=1, sticky="nsew")

        # 상태
        status = ttk.Frame(left)
        status.pack(fill="x", pady=(0, 10))
        canlbl = "mock(개발)" if self._can_mock else "실기"
        imulbl = "OK" if self._imu_ok else "미가용(거리만)"
        ttk.Label(status, text=f"CAN: {canlbl}      IMU: {imulbl}",
                  font=(fam,11, "bold")).pack(anchor="w")
        self.imu_var = tk.StringVar(value="rel_yaw: --")
        ttk.Label(status, textvariable=self.imu_var, font=(fam,18, "bold"),
                  foreground="#2266cc").pack(anchor="w", pady=(2, 0))

        # ── 회전 ──
        rf = ttk.LabelFrame(left, text=" 회전 ", padding=10)
        rf.pack(fill="x", pady=6)
        ttk.Label(rf, text="목표각(°)").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.rot_deg = tk.StringVar(value="90")
        ttk.Entry(rf, textvariable=self.rot_deg, width=8).grid(row=0, column=1, padx=4)
        self.rot_dir = tk.StringVar(value="ccw")
        ttk.Radiobutton(rf, text="ccw(좌)", variable=self.rot_dir, value="ccw").grid(row=0, column=2, padx=4)
        ttk.Radiobutton(rf, text="cw(우)", variable=self.rot_dir, value="cw").grid(row=0, column=3, padx=4)
        self.btn_rot = ttk.Button(rf, text="회전 실행", command=self.on_rotate)
        self.btn_rot.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        rf.columnconfigure(1, weight=1)

        # ── 거리 calib ──
        cf = ttk.LabelFrame(left, text=" 거리 calib (시간 → 자 실측) ", padding=10)
        cf.pack(fill="x", pady=6)
        ttk.Label(cf, text="시간 T(s)").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.cal_t = tk.StringVar(value="3.0")
        ttk.Entry(cf, textvariable=self.cal_t, width=8).grid(row=0, column=1, padx=4)
        self.cal_dir = tk.StringVar(value="fwd")
        ttk.Radiobutton(cf, text="fwd", variable=self.cal_dir, value="fwd").grid(row=0, column=2, padx=4)
        ttk.Radiobutton(cf, text="back", variable=self.cal_dir, value="back").grid(row=0, column=3, padx=4)
        self.btn_cal = ttk.Button(cf, text="calib 실행", command=self.on_calib)
        self.btn_cal.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        cf.columnconfigure(1, weight=1)

        # ── 거리 eval ──
        ef = ttk.LabelFrame(left, text=" 거리 eval (refit 후 · held-out) ", padding=10)
        ef.pack(fill="x", pady=6)
        ttk.Label(ef, text="거리 D(m)").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.ev_d = tk.StringVar(value="2.0")
        ttk.Entry(ef, textvariable=self.ev_d, width=8).grid(row=0, column=1, padx=4)
        self.ev_dir = tk.StringVar(value="fwd")
        ttk.Radiobutton(ef, text="fwd", variable=self.ev_dir, value="fwd").grid(row=0, column=2, padx=4)
        ttk.Radiobutton(ef, text="back", variable=self.ev_dir, value="back").grid(row=0, column=3, padx=4)
        self.btn_ev = ttk.Button(ef, text="eval 실행", command=self.on_eval)
        self.btn_ev.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ef.columnconfigure(1, weight=1)

        # ── 결과 / 기록 ──
        gf = ttk.LabelFrame(left, text=" 결과 / 기록 ", padding=10)
        gf.pack(fill="x", pady=6)
        self.result_var = tk.StringVar(value="(동작 대기)")
        ttk.Label(gf, textvariable=self.result_var, wraplength=340, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(gf, text="실측값").grid(row=1, column=0, sticky="e", padx=4)
        self.measure = tk.StringVar(value="")
        self.ent_measure = ttk.Entry(gf, textvariable=self.measure, width=12)
        self.ent_measure.grid(row=1, column=1, padx=4)
        self.btn_save = ttk.Button(gf, text="기록 저장", command=self.on_save, state="disabled")
        self.btn_save.grid(row=1, column=2, sticky="ew", padx=4)
        gf.columnconfigure(1, weight=1)

        # ── 비상정지 ──
        self.btn_estop = tk.Button(left, text="■ 비상정지 (Space)", command=self.on_estop,
                                   bg="#cc3322", fg="white", font=(fam,13, "bold"),
                                   activebackground="#a82a1c", activeforeground="white")
        self.btn_estop.pack(fill="x", pady=(12, 0), ipady=6)

        # 설명 패널
        ttk.Label(right, text="설명", font=(fam,11, "bold")).pack(anchor="w")
        txt = tk.Text(right, width=44, height=32, wrap="word", relief="flat",
                      background="#f4f4f0", font=(fam,10), padx=8, pady=8)
        txt.pack(fill="both", expand=True, pady=(4, 0))
        txt.insert("1.0", HELP_TEXT)
        txt.config(state="disabled")

    # ---------- IMU 실시간 표시 ----------
    def _poll_imu(self):
        if self._imu_ok:
            self.imu_var.set(f"rel_yaw: {self.reader.rel_yaw:+7.2f}°")
        else:
            self.imu_var.set("rel_yaw: -- (IMU 미가용)")
        self.root.after(100, self._poll_imu)

    # ---------- 동작 공통 ----------
    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.btn_rot, self.btn_cal, self.btn_ev):
            b.config(state=state)

    def _run_async(self, fn):
        if self._busy:
            return
        self._stop_event.clear()
        self._set_busy(True)
        self.btn_save.config(state="disabled")
        self.result_var.set("동작 중...")
        threading.Thread(target=fn, daemon=True).start()

    def _done(self, result_text, pending):
        """워커 스레드 → 메인 스레드에서 호출."""
        self.result_var.set(result_text)
        self._pending = pending
        self.measure.set("")
        self.btn_save.config(state="normal")
        self._set_busy(False)

    # ---------- 회전 ----------
    def on_rotate(self):
        if not self._imu_ok:
            messagebox.showwarning("IMU 미가용", "RealSense(IMU)가 없어 회전 평가 불가.")
            return
        try:
            target = float(self.rot_deg.get())
        except ValueError:
            messagebox.showerror("입력 오류", "목표각이 숫자가 아닙니다.")
            return
        direction = self.rot_dir.get()

        def work():
            self.reader.start_recording(state="ROTATE", cmd="ROTATE",
                                        cmd_bytes=em._move_bytes("rotate_ccw" if direction == "ccw" else "rotate_cw"))
            res = em.rotate_sequence(           # 움직임 코드(원본) — 손대지 않음
                reader=self.reader, target_deg=target, turn_dir=em._dir_to_turn(direction),
                issue_rotate=control.issue_command_rotate_in_place,
                issue_stop=control.issue_command_stop, stop_event=self._stop_event)
            rows = self.reader.stop_recording()
            label = time.strftime("%Y%m%dT%H%M%S")
            raw_path = em.write_raw_motion(label, "rotate", rows) if rows else None
            txt = (f"결과(회전): imu@stop={res.imu_stop_deg:.2f}° "
                   f"imu@settle={res.imu_settled_deg:.2f}° t={res.elapsed_s:.2f}s "
                   f"reached={res.reached}\nraw→{raw_path}\n→ AprilTag 실측각(°) 입력 후 저장")
            pending = {"kind": "rotate", "target": target, "direction": direction,
                       "power": em.POWER_ROT, "imu_stop_deg": round(res.imu_stop_deg, 3),
                       "imu_settled_deg": round(res.imu_settled_deg, 3),
                       "elapsed_s": round(res.elapsed_s, 3), "cmd_time_s": "",
                       "reached": res.reached}
            self.root.after(0, lambda: self._done(txt, pending))
        self._run_async(work)

    # ---------- 거리 calib ----------
    def on_calib(self):
        try:
            T = float(self.cal_t.get())
        except ValueError:
            messagebox.showerror("입력 오류", "시간 T가 숫자가 아닙니다.")
            return
        direction = self.cal_dir.get()
        issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward

        state = "FORWARD" if direction == "fwd" else "BACKWARD"
        move = "forward" if direction == "fwd" else "backward"

        def work():
            if self._imu_ok:
                self.reader.start_recording(state=state, cmd=state, cmd_bytes=em._move_bytes(move))
            elapsed = em.drive_sequence(duration_s=T, issue_drive=issue,   # 움직임 코드(원본)
                                        issue_stop=control.issue_command_stop,
                                        stop_event=self._stop_event)
            rows = self.reader.stop_recording() if self._imu_ok else []
            raw_path = em.write_raw_motion(time.strftime("%Y%m%dT%H%M%S"), "forward", rows) if rows else None
            txt = (f"결과(calib): {direction} {elapsed:.2f}s 전진 완료\n"
                   + (f"raw→{raw_path}\n" if raw_path else "")
                   + "→ 자로 잰 거리(m) 입력 후 저장")
            pending = {"_calib": True, "T_sec": round(elapsed, 3),
                       "direction": direction, "power": em.POWER_FWD}
            self.root.after(0, lambda: self._done(txt, pending))
        self._run_async(work)

    # ---------- 거리 eval ----------
    def on_eval(self):
        try:
            D = float(self.ev_d.get())
        except ValueError:
            messagebox.showerror("입력 오류", "거리 D가 숫자가 아닙니다.")
            return
        direction = self.ev_dir.get()
        t_cmd = motion_models.fwd_sec_from_offset_piecewise(D)
        issue = control.issue_command_forward if direction == "fwd" else control.issue_command_backward

        state = "FORWARD" if direction == "fwd" else "BACKWARD"
        move = "forward" if direction == "fwd" else "backward"

        def work():
            if self._imu_ok:
                self.reader.start_recording(state=state, cmd=state, cmd_bytes=em._move_bytes(move))
            elapsed = em.drive_sequence(duration_s=t_cmd, issue_drive=issue,   # 움직임 코드(원본)
                                        issue_stop=control.issue_command_stop,
                                        stop_event=self._stop_event)
            rows = self.reader.stop_recording() if self._imu_ok else []
            raw_path = em.write_raw_motion(time.strftime("%Y%m%dT%H%M%S"), "forward", rows) if rows else None
            txt = (f"결과(eval): {direction} D={D}m → t_cmd={t_cmd:.2f}s "
                   f"(실제 {elapsed:.2f}s)\n"
                   + (f"raw→{raw_path}\n" if raw_path else "")
                   + "→ 자로 잰 거리(m) 입력 후 저장")
            pending = {"kind": "drive_eval", "target": D, "direction": direction,
                       "power": em.POWER_FWD, "imu_stop_deg": "", "imu_settled_deg": "",
                       "elapsed_s": round(elapsed, 3), "cmd_time_s": round(t_cmd, 3),
                       "reached": True}
            self.root.after(0, lambda: self._done(txt, pending))
        self._run_async(work)

    # ---------- 저장 ----------
    def on_save(self):
        if self._pending is None:
            return
        try:
            measured = float(self.measure.get())
        except ValueError:
            messagebox.showerror("입력 오류", "실측값이 숫자가 아닙니다.")
            return
        p = self._pending
        if p.get("_calib"):
            em.append_calib_row(em.CALIB_CSV, T_sec=p["T_sec"], d_measured=measured,
                                direction=p["direction"], power=p["power"])
            where = em.CALIB_CSV
        else:
            row = {**p, "measured": measured, "note": ""}
            em.append_eval_row(em.EVAL_CSV, row)
            where = em.EVAL_CSV
        self.result_var.set(f"저장됨 → {os.path.basename(where)}")
        self.btn_save.config(state="disabled")
        self._pending = None

    # ---------- 비상정지 ----------
    def on_estop(self):
        self._stop_event.set()
        control.issue_command_stop()
        self.result_var.set("■ 비상정지 — STOP 송신")

    def close(self):
        try:
            self.on_estop()
            if self._imu_ok:
                self.reader.stop()
            control.can_close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    gui = MotionEvalGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    root.mainloop()


if __name__ == "__main__":
    main()
