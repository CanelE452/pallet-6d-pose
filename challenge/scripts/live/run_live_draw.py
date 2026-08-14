"""run_live.py — 시각화 모듈.

CUBOID_EDGES_*  : 8-corner cuboid 의 12 edge (front/back/vertical 별)
LIVE_PANEL_W    : 우측 키 안내 패널 폭
draw_cuboid     : projected 9-point (8 corner + center) → wireframe
build_live_panel: 우측 STATUS + 키 안내 패널
noop            : cv2.createTrackbar dummy callback
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import numpy as np
import cv2


# NDDS/DOPE 큐보이드 8-corner 컨벤션의 12 edge
# 0~3: 앞면 (front-top-left, front-top-right, front-bottom-right, front-bottom-left)
# 4~7: 뒷면 (back-top-left, ..., back-bottom-left)
CUBOID_EDGES_FRONT    = [(0, 1), (1, 2), (2, 3), (3, 0)]
CUBOID_EDGES_BACK     = [(4, 5), (5, 6), (6, 7), (7, 4)]
CUBOID_EDGES_VERTICAL = [(0, 4), (1, 5), (2, 6), (3, 7)]


LIVE_PANEL_W = 260


def noop(_x):
    """cv2.createTrackbar 의 callback dummy."""
    pass


def draw_cuboid(img, proj_pts, color_front, color_back, thickness=2):
    """proj_pts: 9개 (8 corner + center). 8개 중 None 이 아닌 점만 연결.
    뒷면 → 수직 → 앞면 순서로 그려 가까운 면 강조."""
    pts = []
    for i in range(8):
        if i < len(proj_pts) and proj_pts[i] is not None:
            pts.append((int(proj_pts[i][0]), int(proj_pts[i][1])))
        else:
            pts.append(None)
    # 뒷면 (가려지는 부분 먼저)
    for a, b in CUBOID_EDGES_BACK:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color_back, max(1, thickness - 1), cv2.LINE_AA)
    # 수직 (앞-뒤 연결)
    for a, b in CUBOID_EDGES_VERTICAL:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color_front, thickness, cv2.LINE_AA)
    # 앞면 (가장 강조)
    for a, b in CUBOID_EDGES_FRONT:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color_front, thickness + 1, cv2.LINE_AA)


def build_live_panel(h, is_seq, paused, seq_idx, seq_total, seq_fps,
                     cfg_thr, gates_live, status, status_color):
    """우측 키 안내 + 현재 상태 패널."""
    panel = np.full((h, LIVE_PANEL_W, 3), 25, dtype=np.uint8)

    def put(y, text, color=(220, 220, 220), scale=0.42, thick=1):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    y = 18
    put(y, "STATUS", (255, 255, 0), 0.5, 1); y += 22
    put(y, status, status_color, 0.5, 2); y += 22

    if is_seq:
        put(y, f"frame {seq_idx}/{seq_total}", (200, 200, 200)); y += 16
        put(y, f"fps {seq_fps:.1f}" + ("  PAUSED" if paused else ""),
            (0, 200, 255) if paused else (200, 200, 200)); y += 22
    else:
        put(y, "LIVE camera", (200, 200, 200)); y += 22

    put(y, f"thr {cfg_thr:.3f}", (200, 200, 200)); y += 16
    put(y, f"min_kp {gates_live['min_detected_keypoints']}", (200, 200, 200)); y += 16
    put(y, f"max_reproj {gates_live['max_reproj_error_px']:.1f}", (200, 200, 200)); y += 22

    put(y, "KEYS — COMMON", (255, 255, 0), 0.5, 1); y += 22
    put(y, "q       quit",          (200, 200, 200)); y += 16
    put(y, "s       save frame",    (200, 200, 200)); y += 16
    put(y, "b       belief toggle", (200, 200, 200)); y += 16
    put(y, "r       reset autotune",(200, 200, 200)); y += 22

    if is_seq:
        put(y, "KEYS — SEQUENCE", (255, 255, 0), 0.5, 1); y += 22
        put(y, "space   pause/play",  (200, 200, 200)); y += 16
        put(y, "n / p   +/- 1 frame", (200, 200, 200)); y += 16
        put(y, ", / .   +/- 10",      (200, 200, 200)); y += 16
        put(y, "] or =  speed up",    (200, 200, 200)); y += 16
        put(y, "[ or -  slow down",   (200, 200, 200)); y += 22

    put(y, "MARKERS", (255, 255, 0), 0.5, 1); y += 22
    put(y, "gray    raw peak",       (160, 160, 160), 0.4); y += 15
    put(y, "green   corner kp",      (0, 200, 0), 0.4);    y += 15
    put(y, "red     centroid",       (0, 0, 220), 0.4);    y += 15
    put(y, "yellow  yaw arrow",      (0, 200, 220), 0.4);  y += 15
    put(y, "wire    PnP cuboid",     (0, 220, 0), 0.4);    y += 15

    return panel
