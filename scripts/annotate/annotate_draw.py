"""annotate.py — 시각화 모듈.

상수: KP_NAMES / KP_COLORS / CUBOID_EDGES / PANEL_W
함수:
  draw_overlay(img, kps_2d, active, pose)        : 이미지 위 cuboid wireframe + 점
  draw_line_input(img, line_pts, mouse, zoom, pan): TWO-LINE 입력 진행 표시
  build_panel(h, active, kps_2d, pose, ...)       : 우측 키 안내 + 상태 패널
  render(state, frame_idx, total, frame_name)     : 전체 화면 합성 (image + zoom + overlay + panel)
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import numpy as np
import time as _time

import cv2


def _ascii_only(text):
    """cv2.putText 로 그릴 수 있는 문자만 남긴다.

    Hershey 폰트에는 한글 glyph 가 없어서, 한글을 그대로 넘기면 화면에 통째로
    '[??????] SEALED ??????' 처럼 물음표로 나온다(2026-08-16 실제로 겪음).
    호출부가 실수로 한글을 넘겨도 최소한 읽을 수 있는 게 남게 한다.
    """
    t = str(text)
    if t.isascii():
        return t
    kept = "".join(c for c in t if c.isascii())
    return kept.strip() or "(see terminal)"

from annotate_pnp import PALLET_DIMS


# Camera-facing convention (2026-05-22):
#   0~3 = 카메라에 가까운 near face (운용 시 = fork pocket 면)
#   4~7 = 반대편 far face
# 사용자가 "보이는 면" 에 0~3 클릭. 학습/추론 둘 다 동일 컨벤션.
KP_NAMES = [
    "NearTopLeft",     "NearTopRight",    "NearBottomRight",  "NearBottomLeft",
    "FarTopLeft",      "FarTopRight",     "FarBottomRight",   "FarBottomLeft",
    "Centroid",
]

# 색상 — 앞면(0~3) 따뜻한 색, 뒷면(4~7) 차가운 색, centroid 흰색
KP_COLORS = [
    (0,   0, 255),   # 0 red
    (0, 128, 255),   # 1 orange
    (0, 255, 255),   # 2 yellow
    (0, 255,   0),   # 3 green   (앞면 4개)
    (255, 255,   0), # 4 cyan
    (255,   0,   0), # 5 blue
    (255,   0, 128), # 6 magenta
    (128,  0, 255),  # 7 purple
    (255, 255, 255), # 8 white centroid
]

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # 앞면 — 두껍게
    (4, 5), (5, 6), (6, 7), (7, 4),   # 뒷면
    (0, 4), (1, 5), (2, 6), (3, 7),   # 수직
]

PANEL_W = 280  # 우측 키 안내 패널 폭


def annot_button_rect(panel_h):
    """패널 하단 'ANNOT-ONLY' 클릭 버튼의 panel-local 사각형 (x0,y0,x1,y1)."""
    x0, x1 = 10, PANEL_W - 10
    y1 = panel_h - 14
    y0 = y1 - 42
    return (x0, y0, x1, y1)


def session_button_rect(panel_h):
    """ANNOT-ONLY 버튼 바로 위의 'SESSION' 클릭 버튼 (panel-local)."""
    x0, x1 = 10, PANEL_W - 10
    y1 = panel_h - 14 - 42 - 8
    y0 = y1 - 34
    return (x0, y0, x1, y1)

# ─── Letterbox margin ─────────────────────────────────────────────────────────
# 캡처 이미지(640x480) 밖으로 projection 되는 keypoint/cuboid 코너(특히 화면 하단의
# far-bottom 6/7) 가 image 경계를 넘어가 안 보이는 문제 해결용 여백.
# 확장 캔버스 = margin(어두운 회색) + 원본 이미지를 (MARGIN_L, MARGIN_T) 위치에 배치.
# 캔버스 좌표 = image 좌표 + (MARGIN_L, MARGIN_T).  아래쪽(MARGIN_B)을 더 넉넉히.
MARGIN_L = 200
MARGIN_R = 200
MARGIN_T = 200
# 아래 여백은 320 이었는데 이미지 높이(480)의 67% 라 화면에서 검은 띠가 지나치게 길었다.
# 위쪽과 같은 200 으로 맞춘다 (2026-08-15 사용자 요청). far-bottom 이 이보다 더 아래로
# 벗어나는 프레임이 나오면 여기만 올리면 된다.
MARGIN_B = 200
MARGIN_BG = 40  # 여백 색 (어두운 회색)


def make_canvas(img):
    """원본 image 를 margin 으로 둘러싼 확장 캔버스 생성.
    반환 캔버스의 (MARGIN_L, MARGIN_T) 위치에 원본 image 가 들어간다.
    image 좌표 (u, v) → 캔버스 좌표 (u + MARGIN_L, v + MARGIN_T)."""
    h, w = img.shape[:2]
    cw = w + MARGIN_L + MARGIN_R
    ch = h + MARGIN_T + MARGIN_B
    canvas = np.full((ch, cw, 3), MARGIN_BG, dtype=np.uint8)
    canvas[MARGIN_T:MARGIN_T + h, MARGIN_L:MARGIN_L + w] = img
    # 원본 image 영역 경계선 (여백과 구분)
    cv2.rectangle(canvas, (MARGIN_L - 1, MARGIN_T - 1),
                  (MARGIN_L + w, MARGIN_T + h), (90, 90, 90), 1)
    return canvas


def draw_line_input(img, line_pts, mouse_xy, zoom, pan):
    """진행 중인 TWO-LINE input 을 확장 캔버스 위에 그린다.
    line_pts/mouse 는 image 좌표 → 캔버스 좌표로 offset 후 그림."""
    if not line_pts:
        if mouse_xy is not None:
            # mouse_xy 는 screen 좌표 → 캔버스 좌표 (zoom/pan 역변환).
            mu = (mouse_xy[0] / zoom) + pan[0]
            mv = (mouse_xy[1] / zoom) + pan[1]
            cv2.drawMarker(img, (int(mu), int(mv)), (0, 255, 255),
                           cv2.MARKER_CROSS, 14, 1)
        return
    # line_pts 는 image 좌표 → 캔버스 좌표로 offset.
    def cxy(p):
        return (int(p[0] + MARGIN_L), int(p[1] + MARGIN_T))
    colors = [(0, 255, 255), (0, 255, 255), (0, 200, 255), (0, 200, 255)]
    for i, p in enumerate(line_pts):
        cx, cy = cxy(p)
        cv2.circle(img, (cx, cy), 4, colors[i], -1)
        cv2.putText(img, f"L{i//2+1}-{['A','B'][i%2]}",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    if len(line_pts) >= 2:
        cv2.line(img, cxy(line_pts[0]), cxy(line_pts[1]),
                       (0, 255, 255), 2, cv2.LINE_AA)
    if len(line_pts) >= 4:
        cv2.line(img, cxy(line_pts[2]), cxy(line_pts[3]),
                       (0, 200, 255), 2, cv2.LINE_AA)
    if mouse_xy is not None and len(line_pts) in (1, 3):
        # mouse_xy 는 screen 좌표 → 캔버스 좌표 (zoom/pan 역변환).
        mu = (mouse_xy[0] / zoom) + pan[0]
        mv = (mouse_xy[1] / zoom) + pan[1]
        col = (0, 255, 255) if len(line_pts) == 1 else (0, 200, 255)
        cv2.line(img, cxy(line_pts[-1]), (int(mu), int(mv)),
                 col, 1, cv2.LINE_AA)


def draw_overlay(img, kps_2d, active_idx, pose=None, extrap_mask=None):
    """이미지에 cuboid wireframe + 사용자 클릭 점 그리기.

    v6 컨벤션 경고: pose.v4_warning=True 시 화면 상단에 빨간 경고 표시.
      - _v6_lr_viol / _v6_tb_viol / _v6_fr_viol: pose pair-wise invariant 위반 카운트
      - _v6_click_lr_viol / _v6_click_tb_viol: 사용자 클릭 LR/TB pair 부등호 위반

    v7: extrap_mask 가 주어지면 외삽 점은 outlined (속 빈 원) 으로 표시 →
    직접 click 과 시각 구분.
    """
    # 확장 캔버스 (margin) 에 그린다 — image 밖으로 나간 코너도 여백에 표시됨.
    # 모든 점: 캔버스 좌표 = image 좌표 + (MARGIN_L, MARGIN_T).
    vis = make_canvas(img)
    if pose is not None:
        proj = pose["projected_all"]
        # v7: project_3d sentinel = (-1, -1) — 그 외 음수 u/v 는 valid (image 밖).
        pts = [(int(p[0] + MARGIN_L), int(p[1] + MARGIN_T))
               if not (p[0] == -1.0 and p[1] == -1.0)
               else None for p in proj[:8]]
        for k, (a, b) in enumerate(CUBOID_EDGES):
            if pts[a] and pts[b]:
                col = (0, 220, 0) if k < 4 else (0, 160, 0)
                thick = 3 if k < 4 else 1
                cv2.line(vis, pts[a], pts[b], col, thick, cv2.LINE_AA)
    for i, p in enumerate(kps_2d):
        if p is None:
            continue
        c = (int(p[0] + MARGIN_L), int(p[1] + MARGIN_T))
        # 마커가 코너 픽셀을 덮어 정밀 클릭을 방해해서 반지름을 절반으로 줄였다
        # (2026-08-15 사용자 요청: 7/5 -> 4/3).
        r = 4 if i == active_idx else 3
        is_extrap = (extrap_mask is not None and i < len(extrap_mask)
                     and extrap_mask[i])
        if is_extrap:
            # 외삽 점: 속 빈 원 (외곽선 + 작은 중심 점)
            cv2.circle(vis, c, r, KP_COLORS[i], 1)
            cv2.circle(vis, c, 1, KP_COLORS[i], -1)
            cv2.circle(vis, c, r + 1, (0, 0, 0), 1)
            cv2.putText(vis, f"{i}*", (c[0] + 5, c[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, KP_COLORS[i], 1)
        else:
            cv2.circle(vis, c, r, KP_COLORS[i], -1)
            cv2.circle(vis, c, r + 1, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (c[0] + 5, c[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, KP_COLORS[i], 1)
    return vis


def _pose_dim_short(pose):
    if pose is None:
        return ""
    d = pose.get("dims", PALLET_DIMS)
    return f"front={d[0]*100:.0f}cm"


def build_panel(h, active_idx, kps_2d, pose, frame_idx, total, zoom, dirty,
                mode="click", trans_step=0.02, rot_step=5.0, split="eval",
                annot_only=False, sess_name=None):
    """우측 키 안내 + 현재 상태 패널."""
    panel = np.full((h, PANEL_W, 3), 25, dtype=np.uint8)

    def put(y, text, color=(220, 220, 220), scale=0.42, thick=1):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    y = 18
    mode_color = (0, 255, 0) if mode == "click" else (0, 200, 255)
    put(y, f"MODE: {mode.upper()}  [m=toggle]", mode_color, 0.5, 2); y += 22
    split_color = (0, 255, 0) if split == "eval" else (150, 150, 150)
    put(y, f"SPLIT: {split.upper()}  [v=toggle]", split_color, 0.5, 2); y += 22

    if mode == "click":
        put(y, "KEYBOARD - CLICK", (255, 255, 0), 0.5, 1); y += 22
        put(y, "L click  set point", (200, 200, 200)); y += 16
        put(y, "R click  delete", (200, 200, 200));    y += 16
        put(y, "0-8      select idx", (200, 200, 200)); y += 16
        put(y, "d        delete kp[active]", (200, 200, 200)); y += 16
        put(y, "t        TWO-LINE input *", (0, 255, 255)); y += 16
        put(y, "x        parallelogram extrap *", (0, 255, 255)); y += 16
        put(y, "  (* = extrap, PnP weight 0.3)", (140, 200, 255), 0.36); y += 14
        put(y, "s        save+next", (0, 255, 0));      y += 16
        put(y, "f        near-only save+next", (0, 255, 0)); y += 16
        put(y, "g        auto-fill save (4+pts)", (0, 255, 0)); y += 16
        put(y, "v        eval/train toggle", (0, 255, 0)); y += 16
        put(y, "n / p    next / prev", (200, 200, 200)); y += 16
        put(y, ", / .    -10 / +10",   (200, 200, 200)); y += 16
        put(y, "G / :    goto frame # (Enter)", (0, 255, 255)); y += 16
        put(y, "slider   click/drag to jump", (0, 255, 255)); y += 16
        put(y, "TAB      SESSION dropdown", (255, 160, 0), 0.45); y += 16
        put(y, "[ / ]    prev / next SESSION", (255, 160, 0), 0.45); y += 16
        put(y, "c        centroid auto", (200, 200, 200)); y += 16
        put(y, "z        undo last",   (200, 200, 200)); y += 16
        put(y, "r        reset all",   (200, 200, 200)); y += 16
        put(y, "+ / -    zoom in/out", (200, 200, 200)); y += 16
        put(y, "h j k l  pan (vim)",   (200, 200, 200)); y += 16
        put(y, "q        quit",        (180, 180, 180)); y += 22
    else:
        put(y, "KEYBOARD - MANIPULATE", (255, 255, 0), 0.5, 1); y += 22
        put(y, "translate (camera frame)", (160, 200, 255), 0.42); y += 16
        put(y, "  w/x   up/down  (Y)",   (200, 200, 200)); y += 16
        put(y, "  a/d   left/right (X)", (200, 200, 200)); y += 16
        put(y, "  q/e   near/far  (Z)",  (200, 200, 200)); y += 16
        put(y, "rotate (pallet local)",  (160, 200, 255), 0.42); y += 16
        put(y, "  j/l   yaw -/+",   (200, 200, 200)); y += 16
        put(y, "  i/k   pitch -/+", (200, 200, 200)); y += 16
        put(y, "  u/o   roll -/+",  (200, 200, 200)); y += 16
        put(y, "step", (160, 200, 255), 0.42); y += 16
        put(y, f"  1/2  trans x/2 x2 ({trans_step*100:.1f}cm)", (200, 200, 200)); y += 16
        put(y, f"  3/4  rot x/2 x2  ({rot_step:.1f}\xb0)", (200, 200, 200)); y += 16
        put(y, "save / quit", (160, 200, 255), 0.42); y += 16
        put(y, "  S    save+next", (0, 255, 0)); y += 16
        put(y, "  m    back to CLICK", (200, 200, 200)); y += 16
        put(y, "  Q    quit", (180, 180, 180)); y += 22

    put(y, "KEYPOINTS", (255, 255, 0), 0.5, 1); y += 22
    n_set = sum(1 for k in kps_2d if k is not None)
    put(y, f"set: {n_set}/9", (200, 200, 200)); y += 16
    for i in range(9):
        col = KP_COLORS[i]
        mark = ">" if i == active_idx else " "
        done = "[x]" if kps_2d[i] is not None else "[ ]"
        put(y, f"{mark} {i} {done} {KP_NAMES[i][:14]}", col, 0.4); y += 15
    y += 8

    put(y, "STATUS", (255, 255, 0), 0.5, 1); y += 22
    # 마지막 프레임임을 화면에 알린다. 콘솔에만 찍으면 어노 중에는 보이지 않아
    # "n 이 안 먹는다" 로 느껴진다(2026-08-15).
    _last = (total > 0 and frame_idx >= total - 1)
    put(y, f"frame {frame_idx+1}/{total}" + ("  <- LAST" if _last else ""),
        (0, 200, 255) if _last else (200, 200, 200)); y += 16
    if _last:
        put(y, "n = 다음 세션으로", (0, 200, 255), 0.38); y += 14
    put(y, f"zoom x{zoom:.1f}", (200, 200, 200)); y += 16
    if dirty:
        put(y, "*UNSAVED*", (0, 0, 255), 0.5, 2); y += 18
    if pose is not None:
        err = pose["reproj_error_px"]
        col = (0, 255, 0) if err < 5 else (0, 200, 255) if err < 10 else (0, 0, 255)
        put(y, f"reproj {err:.2f}px", col, 0.5, 2); y += 18
        dim_text = _pose_dim_short(pose)
        if dim_text:
            put(y, dim_text, (180, 220, 255), 0.45); y += 16

    # ── ANNOT-ONLY 클릭 버튼 (패널 하단) ──
    bx0, by0, bx1, by1 = annot_button_rect(h)
    bcol = (0, 150, 0) if annot_only else (60, 60, 60)
    cv2.rectangle(panel, (bx0, by0), (bx1, by1), bcol, -1)
    cv2.rectangle(panel, (bx0, by0), (bx1, by1), (210, 210, 210), 2)
    label = "ANNOT-ONLY: ON" if annot_only else "ANNOT-ONLY: OFF"
    cv2.putText(panel, label, (bx0 + 10, by0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "click to toggle (n/p=annot only)", (bx0 + 10, by1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    # ── SESSION 클릭 버튼 (그 위) ──
    sx0, sy0, sx1, sy1 = session_button_rect(h)
    cv2.rectangle(panel, (sx0, sy0), (sx1, sy1), (70, 45, 0), -1)
    cv2.rectangle(panel, (sx0, sy0), (sx1, sy1), (255, 170, 0), 2)
    cv2.putText(panel, f"SESSION: {(sess_name or '-')[:20]}", (sx0 + 8, sy0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(panel, "click = list   (slider 'session')", (sx0 + 8, sy1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def render(state, frame_idx, total_frames, frame_name):
    """State → 화면 합성 (image + zoom + overlay + 우측 패널)."""
    vis = draw_overlay(state.img, state.kps_2d, state.active, state.pose,
                       extrap_mask=getattr(state, "extrap_mask", None))
    if state.mode == "click" and state.line_mode:
        draw_line_input(vis, state.line_pts or [], state.last_mouse, state.zoom, state.pan)
    h, w = vis.shape[:2]
    # 클램프는 zoom 과 무관하게 항상 건다. zoom=1 에서만 클램프를 건너뛰면 화면은
    # 하나도 안 움직이는데 클릭 좌표에는 pan 이 더해져 점이 엉뚱한 곳에 찍힌다
    # (h/j/k/l 은 zoom 을 안 보고 pan 을 바꾼다). zoom=1 이면 crop=전체라 0 으로 클램프된다.
    crop_w = max(1, int(w / state.zoom))
    crop_h = max(1, int(h / state.zoom))
    state.pan[0] = max(0, min(w - crop_w, state.pan[0]))
    state.pan[1] = max(0, min(h - crop_h, state.pan[1]))
    if state.zoom > 1.001:
        crop = vis[state.pan[1]:state.pan[1] + crop_h,
                   state.pan[0]:state.pan[0] + crop_w]
        vis = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    name = KP_NAMES[state.active]
    col = KP_COLORS[state.active]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.3, overlay, 0.7, 0)
    toast = getattr(state, "toast", None)
    if toast and toast[2] > _time.time():
        cv2.rectangle(vis, (0, 28), (w, 60), (28, 28, 28), -1)
        cv2.rectangle(vis, (0, 28), (6, 60), toast[1], -1)
        cv2.putText(vis, _ascii_only(toast[0])[:62], (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, toast[1], 1, cv2.LINE_AA)
    cv2.putText(vis, f"Click #{state.active}: {name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(vis, frame_name[:20], (w - 220, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    # 현재 세션 (TAB/[ ] 로 갈아탈 수 있으므로 항상 보이게)
    _sess = getattr(state, "sess_name", None)
    if _sess:
        _sealed = getattr(state, "sess_sealed", False)
        cv2.putText(vis, f"[{_sess}]" + ("  SEALED" if _sealed else ""), (240, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 80, 255) if _sealed else (0, 200, 255), 1)
    # split 배지 (이미지 상단 중앙 — zoom 후에도 항상 보임)
    _split = getattr(state, "split", "eval")
    _sc = (0, 220, 0) if _split == "eval" else (150, 150, 150)
    cv2.putText(vis, _split.upper(), (w // 2 - 30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _sc, 2)
    # 세션 선택은 tkinter 드롭다운(annotate.pick_session_dialog)이 담당한다.
    # 예전엔 목록을 여기에 그렸는데, 진짜 위젯이 아니라 클릭 판정을 직접 해야 했고
    # 사용자가 "왜 화면에 그리냐"고 지적해 교체했다 (2026-08-15).

    # goto 번호 입력 중이면 하단 바에 버퍼 표시
    if getattr(state, "goto_mode", False):
        cv2.rectangle(vis, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, f"GOTO frame #: {state.goto_buf}_   (Enter=go, Esc=cancel)",
                    (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # v6 컨벤션 critical 경고 (zoom 후에도 항상 보이도록 상단 bar 아래에 표시).
    # fix v6 strict invariants (LR/TB/FR pair) 위반 또는 사용자 click LR/TB 모순.
    if state.pose is not None and state.pose.get("v4_warning"):
        msgs = []
        lrv = state.pose.get("_v6_lr_viol", 0)
        tbv = state.pose.get("_v6_tb_viol", 0)
        frv = state.pose.get("_v6_fr_viol", 0)
        if lrv > 0:
            msgs.append(f"LR-viol {lrv}/4")
        if tbv > 0:
            msgs.append(f"TB-viol {tbv}/4")
        if frv > 0:
            msgs.append(f"FR-viol {frv}/4")
        clrv = state.pose.get("_v6_click_lr_viol", 0)
        ctbv = state.pose.get("_v6_click_tb_viol", 0)
        if clrv > 0:
            msgs.append(f"CLICK-LR {clrv}/4")
        if ctbv > 0:
            msgs.append(f"CLICK-TB {ctbv}/4")
        warn_msg = "[v6] " + " | ".join(msgs) if msgs else "[v6] convention violation"
        cv2.rectangle(vis, (0, 30), (w, 52), (0, 0, 0), -1)
        cv2.putText(vis, warn_msg, (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2, cv2.LINE_AA)
    if state.mode == "manip":
        cv2.rectangle(vis, (1, 1), (w - 2, h - 2), (255, 180, 0), 3)
    panel = build_panel(h, state.active, state.kps_2d, state.pose,
                        frame_idx, total_frames, state.zoom, state.dirty,
                        mode=state.mode, trans_step=state.trans_step,
                        rot_step=state.rot_step_deg,
                        split=getattr(state, "split", "eval"),
                        annot_only=getattr(state, "annot_only", False),
                        sess_name=getattr(state, "sess_name", None))
    return np.hstack([vis, panel])
