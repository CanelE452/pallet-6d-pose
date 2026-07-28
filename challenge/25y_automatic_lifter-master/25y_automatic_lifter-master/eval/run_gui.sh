#!/usr/bin/env bash
# 리프터 모션 평가 GUI 실행 (한글 렌더용 시스템 tkinter 사용).
#
# conda python의 Tk는 X11 코어 폰트만 봐서 한글이 깨짐.
# 시스템 python(/usr/bin/python3) + 시스템 tk(Xft) 는 나눔폰트를 인식한다.
# GUI 자체는 numpy/scipy 불필요 (control은 canlib 없으면 mock).
#
# ※ 실기(RealSense IMU 회전)까지 쓰려면 시스템 python에 pyrealsense2 설치 필요:
#     /usr/bin/python3 -m pip install pyrealsense2
#   (없으면 회전은 "IMU 미가용", 거리 calib/eval 은 동작)
cd "$(dirname "$0")/.." && DISPLAY="${DISPLAY:-:0}" /usr/bin/python3 eval/eval_gui.py "$@"
