"""challenge/scripts/annotate.py — main entry.

Manual annotation 도구. 시퀀스 frame 에 9 keypoint 클릭으로 라벨링 + PnP 자동 풀이 +
NDDS JSON GT 저장.

분리된 모듈:
  annotate_pnp.py    PnP / 3D model / projection / MANIPULATE / TWO-LINE
  annotate_draw.py   cuboid wireframe / overlay / UI panel / render
  annotate_io.py     State / make_annotation / save / load

Keypoint 순서 (**camera-facing convention, 2026-05-22 결정**):
  0: NearTopLeft      1: NearTopRight       ★ 카메라에 보이는 가까운 면 (near = fork pocket)
  2: NearBottomRight  3: NearBottomLeft     ★
  4: FarTopLeft       5: FarTopRight        (반대편 far face, 위쪽 corner)
  6: FarBottomRight   7: FarBottomLeft
  8: Centroid         (c 키로 자동)

→ 사용자는 "보이는 가까운 면" 에 0~3 을, 가능하면 보이는 far-top (4, 5) 도 클릭.
   far-bottom (6, 7) 은 가려질 경우가 많아 g 키로 자동 채움 가능 (0~5 → 6/7/8 PnP projection).

키:
  좌클릭        활성 keypoint 위치 → 다음 idx 자동
  0~8           특정 idx 활성 (수정)
  c             centroid 자동
  z             마지막 점 undo
  d             활성 idx 삭제
  r             전체 reset
  s             저장 + 다음 frame
  v             이 frame eval/train 토글 (평가용 표시, JSON split 필드에 저장)
  G 또는 :      frame 번호 입력 점프 (숫자 후 Enter, Esc 취소) — 상단 슬라이더 클릭/드래그도 가능
  ANNOT-ONLY 버튼 (우측 패널 하단 클릭) = ON 이면 n/p 가 어노된 frame 만 이동 (어노된 것만 보기)
  f             near-only 자동 저장 (0~3 만 클릭, 4~7 자동 PnP projection 채움)
  g             auto-fill 저장 (4+ 점 클릭, 미클릭 idx 자동 PnP projection 채움) ★ truncation/occlusion
  x             parallelogram 외삽 (active idx ← 같은 face 의 나머지 3 corner) ★ truncation
  m             CLICK ↔ MANIPULATE 모드 토글
  t             TWO-LINE input 토글
  n / p         다음 / 이전 frame
  , / .         -10 / +10 frame jump
  + / -         zoom in/out
  h j k l       pan (vim)
  q / Q         종료

사용:
  python challenge/scripts/annotate.py --seq data/outside/capturepallet07 --stride 15
  python challenge/scripts/annotate.py --seq data/outside/capturepallet09 --out_dir challenge/data/01_real/manual_gt/pallet09_manual_gt
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Windows DPI scaling 보정 — 모니터가 100% 가 아니면 cv2 윈도우 좌표 ↔ 마우스 좌표
# mismatch 로 클릭 위치보다 점이 어긋남. import 직후 한 번만 호출.
if os.name == "nt":
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor v2 (Win10+)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()        # legacy fallback
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)   # annotate_pnp / annotate_draw / annotate_io import

from annotate_pnp import (
    solve_pose, pose_from_locked, apply_manip, line_intersection,
    parallelogram_extrapolate,
    PALLET_DIMS,
)
from annotate_draw import (render, annot_button_rect,
                           MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B)
from annotate_io import (
    State, make_annotation, save_frame_json, load_existing_annotation,
)


# ─── Mouse callback ──────────────────────────────────────────────────────────

def on_mouse(event, x, y, flags, s: State):
    """L click = keypoint set + active advance.  R click = delete.
    TWO-LINE 모드 시 4 클릭으로 교점 계산해서 active kp 위치 결정."""
    s.last_mouse = (x, y)
    # panel 영역 (확장 캔버스 우측) 클릭은 무시.
    # render() 가 [확장캔버스 | panel] 을 hstack — 확장캔버스 폭 = image_w + MARGIN_L + MARGIN_R.
    # zoom 후에도 확장캔버스는 원래 폭으로 resize 되므로 panel 경계는 항상 canvas_w.
    if s.img is not None:
        canvas_w = s.img.shape[1] + MARGIN_L + MARGIN_R
        if x >= canvas_w:
            # 패널 영역 — ANNOT-ONLY 버튼 클릭만 처리, 나머지는 무시.
            if event == cv2.EVENT_LBUTTONDOWN:
                canvas_h = s.img.shape[0] + MARGIN_T + MARGIN_B
                px, py = x - canvas_w, y
                bx0, by0, bx1, by1 = annot_button_rect(canvas_h)
                if bx0 <= px <= bx1 and by0 <= py <= by1:
                    s.annot_only = not s.annot_only
                    print(f"[Annot-only] {'ON' if s.annot_only else 'OFF'}"
                          f"  (n/p 로 어노된 frame 만 이동)")
            return
    # MANIPULATE 모드에서는 마우스 클릭으로 점 안 찍음
    if s.mode != "click":
        return

    # screen → 확장캔버스 좌표 (zoom/pan 역변환) → image 좌표 (margin offset 제거).
    # image 밖 (u<0, v>480 등) 도 정상 — 여백에서 클릭한 코너의 실제 픽셀 좌표.
    cu = (x / s.zoom) + s.pan[0]
    cv = (y / s.zoom) + s.pan[1]
    u = cu - MARGIN_L
    v = cv - MARGIN_T

    # TWO-LINE sub-mode
    if s.line_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if s.line_pts is None:
                s.line_pts = []
            s.line_pts.append([float(u), float(v)])
            if len(s.line_pts) == 4:
                pt = line_intersection(s.line_pts[0], s.line_pts[1],
                                       s.line_pts[2], s.line_pts[3])
                if pt is not None:
                    s.kps_2d[s.active] = pt
                    if s.extrap_mask is not None:
                        s.extrap_mask[s.active] = True   # v7: t 외삽 표시
                    s.dirty = True
                    if s.active < 8:
                        s.active += 1
                    print(f"[Line] intersection → kp{s.active-1 if s.active>0 else 0}: "
                          f"({pt[0]:.1f}, {pt[1]:.1f})")
                else:
                    print("[Line] 평행선 — 교점 없음. 다시 시도하세요.")
                s.line_mode = False
                s.line_pts = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            if s.line_pts:
                s.line_pts.pop()
        return

    # 일반 CLICK 모드
    if event == cv2.EVENT_LBUTTONDOWN:
        s.kps_2d[s.active] = [float(u), float(v)]
        if s.extrap_mask is not None:
            s.extrap_mask[s.active] = False    # v7: 직접 클릭 = 외삽 아님
        s.dirty = True
        if s.active < 8:
            s.active += 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            if s.extrap_mask is not None:
                s.extrap_mask[s.active] = False
            s.pose = None
            s.dirty = True


def update_pose(s: State, K):
    """현재 mode 에 따라 pose 재계산. MANIPULATE 모드면 locked_pose 직접 사용."""
    if s.mode == "manip" and s.locked_pose is not None:
        s.pose = pose_from_locked(s, K)
    else:
        # v7: t/x 외삽 점 weight 0.3 + degenerate cuboid reject (img_shape 기반)
        s.pose = solve_pose(s.kps_2d, K,
                            extrapolated_mask=s.extrap_mask,
                            img_shape=s.img_shape)


# ─── Key dispatchers ──────────────────────────────────────────────────────────

def _handle_manip_key(key, s, out_json, out_png, src_png, K):
    """MANIPULATE 모드 키 처리. Returns: 'next' | 'quit' | None."""
    ts = s.trans_step
    rs = s.rot_step_deg
    if   key == ord('a'): apply_manip(s, dx=-ts)
    elif key == ord('d'): apply_manip(s, dx=+ts)
    elif key == ord('w'): apply_manip(s, dy=-ts)
    elif key == ord('x'): apply_manip(s, dy=+ts)
    elif key == ord('q'): apply_manip(s, dz=-ts)
    elif key == ord('e'): apply_manip(s, dz=+ts)
    elif key == ord('j'): apply_manip(s, dyaw=-rs)
    elif key == ord('l'): apply_manip(s, dyaw=+rs)
    elif key == ord('i'): apply_manip(s, dpitch=-rs)
    elif key == ord('k'): apply_manip(s, dpitch=+rs)
    elif key == ord('u'): apply_manip(s, droll=-rs)
    elif key == ord('o'): apply_manip(s, droll=+rs)
    elif key == ord('1'):
        s.trans_step = max(0.001, s.trans_step / 2.0)
        print(f"[step] trans={s.trans_step*100:.2f}cm")
    elif key == ord('2'):
        s.trans_step = min(0.5, s.trans_step * 2.0)
        print(f"[step] trans={s.trans_step*100:.2f}cm")
    elif key == ord('3'):
        s.rot_step_deg = max(0.5, s.rot_step_deg / 2.0)
        print(f"[step] rot={s.rot_step_deg:.2f}\xb0")
    elif key == ord('4'):
        s.rot_step_deg = min(45, s.rot_step_deg * 2.0)
        print(f"[step] rot={s.rot_step_deg:.2f}\xb0")
    elif key == ord('S'):
        if s.pose is None:
            return None
        # locked_pose 의 projected_cuboid 를 그대로 manual_kps 로 덮어쓰기
        proj = s.pose["projected_all"]
        s.kps_2d = [list(p) if (p[0] >= 0 or p[1] >= 0) else None for p in proj]
        ann = make_annotation(s.kps_2d, s.pose, s.img_shape, K, split=s.split)
        save_frame_json(out_json, out_png, src_png, ann)
        print(f"[Saved manip] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        s.mode = "click"
        s.locked_pose = None
        return 'save-next'
    elif key == ord('Q'):
        return 'quit'
    return None


def _handle_click_key(key, s, out_json, out_png, src_png, K):
    """CLICK 모드 키 처리. Returns: 'next' | 'prev' | 'quit' | None."""
    if key == ord('q'):
        return 'quit'

    if key == ord('v'):
        s.split = "train" if s.split == "eval" else "eval"
        s.dirty = True
        print(f"[Split] this frame -> {s.split.upper()}  (저장 시 JSON 에 반영)")
        return None

    if key == ord('s'):
        if s.pose is None:
            print("[WARN] PnP 실패 — 최소 4점 필요. 저장 안 됨.")
            return None
        # manual_kps 는 사용자 클릭 그대로 저장 (위치 안 옮김).
        # swap 보정은 라벨링 후 fix_manual_swap.py 후처리.
        ann = make_annotation(s.kps_2d, s.pose, s.img_shape, K, split=s.split)
        save_frame_json(out_json, out_png, src_png, ann)
        print(f"[Saved] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        return 'save-next'

    if key == ord('f'):
        # Front-only 자동 저장: 0~3 만 클릭한 상태에서 PnP projection 으로 4~7 채움.
        # cargo 가 rear face 가린 시퀀스용 단축키.
        if s.pose is None:
            print("[WARN] PnP 실패 — 0~3 4점 모두 필요")
            return None
        proj = s.pose["projected_all"]
        s.kps_2d = [list(p) if (p[0] >= 0 or p[1] >= 0) else None for p in proj]
        ann = make_annotation(s.kps_2d, s.pose, s.img_shape, K, split=s.split)
        save_frame_json(out_json, out_png, src_png, ann)
        print(f"[Saved front-only] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        return 'save-next'

    if key == ord('g'):
        # Auto-fill 저장: 사용자가 클릭한 점은 그대로 두고, 미클릭 0~7 점은 PnP
        # projection 으로 채워서 저장. 8 (centroid) 도 PnP projection 으로 채움.
        # 저장 후 frame 은 안 넘김 — 사용자가 시각적 확인 후 'n' 직접 누름.
        # ★ Truncation 시 (예: 0,3 image 밖 → 012456 만 클릭) 도 동작:
        #    4+점 클릭 + PnP 풀이 가능 → 미클릭 idx 자동 채움.
        n_clicked_07 = sum(1 for k in s.kps_2d[:8] if k is not None)
        if n_clicked_07 < 4:
            print(f"[WARN] g: 0~7 중 4 점 이상 필요. 현재 {n_clicked_07}/8 점.")
            return None
        if s.pose is None:
            print("[WARN] g: PnP 실패 — 위치 확인 (face-flip strict reject 가능). "
                  "위치 미세조정 후 재시도.")
            return None
        # 0~7 + 8(centroid) 중 미클릭만 PnP projection 으로 채움. 사용자 클릭은 그대로.
        proj = s.pose["projected_all"]
        n_auto = 0
        for i in range(9):
            if s.kps_2d[i] is None and proj[i][0] >= 0:
                s.kps_2d[i] = list(proj[i])
                n_auto += 1
        ann = make_annotation(s.kps_2d, s.pose, s.img_shape, K, split=s.split)
        save_frame_json(out_json, out_png, src_png, ann)
        print(f"[Saved auto-fill] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px "
              f"({n_clicked_07} manual + {n_auto} auto-fill) — 시각 확인 후 'n' 으로 다음 frame")
        s.dirty = False
        return None   # ★ frame 안 넘김. 사용자 확인 후 'n' 직접 누름.

    if key == ord('x'):
        # Parallelogram 외삽: 활성 idx (0~7) 의 위치를 그 idx 가 속한 face 의 나머지 3
        # corner 로부터 외삽. truncation 시 (image 밖이라 클릭 불가) 단축키.
        # 예: 012 클릭 + active=3 + 'x' → 3 = 0 + (2 - 1) 자동.
        # 후보 face 여러 개면 평균. centroid (idx=8) 는 'c' 로 처리, 'x' 는 무효.
        if s.active >= 8:
            print("[Parallelogram] active=8 (centroid) — 'c' 키 사용")
            return None
        pt, fname, finds = parallelogram_extrapolate(s.kps_2d, s.active)
        if pt is None:
            print(f"[Parallelogram] kp{s.active} 외삽 실패 — 같은 face 의 다른 3 corner "
                  f"중 미클릭 있음. (face 후보: FRONT/BACK/TOP/BOTTOM/LEFT/RIGHT 중 "
                  f"kp{s.active} 포함 face 의 나머지 3 점 모두 필요)")
            return None
        s.kps_2d[s.active] = pt
        if s.extrap_mask is not None:
            s.extrap_mask[s.active] = True   # v7: x parallelogram 외삽 표시
        s.dirty = True
        print(f"[Parallelogram] kp{s.active} ← face={fname} {finds} → "
              f"({pt[0]:.1f}, {pt[1]:.1f})")
        if s.active < 8:
            s.active += 1
        return None

    if key == ord('n'):
        if s.dirty:
            print("[WARN] 미저장 변경 있음. 다시 'n' 누르면 무시하고 다음.")
            s.dirty = False
            return None
        return 'next'
    if key == ord('p'):
        return 'prev'
    if key == ord(','):
        return 'jump-10'
    if key == ord('.'):
        return 'jump+10'

    if key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
               ord('5'), ord('6'), ord('7'), ord('8')):
        s.active = key - ord('0')
        return None

    if key == ord('c'):
        if s.pose is not None and s.pose["projected_all"][8][0] >= 0:
            s.kps_2d[8] = list(s.pose["projected_all"][8])
            s.dirty = True
            print(f"[Centroid] PnP projection: ({s.kps_2d[8][0]:.1f}, {s.kps_2d[8][1]:.1f})")
        else:
            pts = [k for k in s.kps_2d[:8] if k is not None]
            if len(pts) >= 4:
                s.kps_2d[8] = [float(np.mean([p[0] for p in pts])),
                               float(np.mean([p[1] for p in pts]))]
                s.dirty = True
                print("[Centroid] fallback (image corner mean) — PnP 풀린 후 c 권장")
        return None

    if key == ord('z'):
        if s.line_mode and s.line_pts:
            s.line_pts.pop()
        else:
            if s.active > 0:
                s.active -= 1
            s.kps_2d[s.active] = None
            s.dirty = True
        return None

    if key == ord('d'):
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            s.pose = None
            s.dirty = True
            print(f"[Delete] kp{s.active} 삭제")
        return None

    if key == ord('t'):
        if s.line_mode:
            s.line_mode = False
            s.line_pts = []
            print("[Line] 취소")
        else:
            s.line_mode = True
            s.line_pts = []
            print(f"[Line] kp{s.active} 위치 추정 — 4 번 클릭: line1-A, B, line2-A, B")
        return None

    if key == ord('r'):
        s.kps_2d = [None] * 9
        s.extrap_mask = [False] * 9
        s.active = 0
        s.line_mode = False
        s.line_pts = []
        s.dirty = True
        return None

    if key in (ord('+'), ord('=')):
        old_z = s.zoom
        s.zoom = min(4.0, s.zoom * 1.5)
        if s.last_mouse and s.img is not None:
            # pan/zoom 은 확장 캔버스 기준 (render 의 crop 도 확장 캔버스 = vis 에 작동).
            h = s.img.shape[0] + MARGIN_T + MARGIN_B
            w = s.img.shape[1] + MARGIN_L + MARGIN_R
            cx = s.pan[0] + (s.last_mouse[0] / old_z)
            cy = s.pan[1] + (s.last_mouse[1] / old_z)
            s.pan[0] = int(cx - (w / s.zoom) / 2)
            s.pan[1] = int(cy - (h / s.zoom) / 2)
        return None
    if key in (ord('-'), ord('_')):
        s.zoom = max(1.0, s.zoom / 1.5)
        if s.zoom <= 1.001:
            s.pan = [0, 0]
        return None

    if key == ord('h'): s.pan[0] -= 20
    elif key == ord('l'): s.pan[0] += 20
    elif key == ord('k'): s.pan[1] -= 20
    elif key == ord('j'): s.pan[1] += 20
    return None


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",     default="data/outside/capturepallet02")
    ap.add_argument("--out_dir", default=None,
                    help="기본: challenge/data/<seq_name>_manual_gt")
    ap.add_argument("--stride",  type=int, default=30, help="N frame 마다 1개 annotate")
    ap.add_argument("--start",   type=int, default=0, help="시작 frame idx")
    ap.add_argument("--default_split", choices=["eval", "train"], default="train",
                    help="새 frame 의 기본 split (v 키로 토글). 기본 train — eval 로 쓸 것만 v "
                         "로 표시. eval GT 대량 어노 시 --default_split eval.")
    args = ap.parse_args()

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO, args.seq)
    seq_name = os.path.basename(seq.rstrip("/\\"))
    out_dir = args.out_dir or os.path.join(_REPO, "challenge", "data", f"{seq_name}_manual_gt")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    K_path = os.path.join(seq, "cam_K.txt")
    K = np.loadtxt(K_path).reshape(3, 3) if os.path.isfile(K_path) else \
        np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]],
                 dtype=np.float64)

    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
    if not rgb_paths:
        print(f"[ERROR] no rgb frames in {seq}")
        return

    selected = list(range(args.start, len(rgb_paths), args.stride))
    print(f"[Annotate] {seq}")
    print(f"           {len(selected)} frames to annotate (stride={args.stride})")
    print(f"           Output: {out_dir}")
    print(f"           K = fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    win = "Annotate"
    cv2.namedWindow(win)
    s = State()
    cv2.setMouseCallback(win, on_mouse, s)
    # frame 점프 슬라이더 (클릭/드래그로 임의 frame 이동). 번호 입력은 G/: 키.
    if len(selected) > 1:
        cv2.createTrackbar("frame", win, 0, len(selected) - 1, lambda v: None)

    def _has_annot(ci):
        st = os.path.splitext(os.path.basename(rgb_paths[selected[ci]]))[0]
        return os.path.exists(os.path.join(out_dir, st + ".json"))

    def _step_annot(ci, d):
        """방향 d(+1/-1)로 어노된 다음 frame 인덱스. 없으면 제자리."""
        j = ci + d
        while 0 <= j < len(selected):
            if _has_annot(j):
                return j
            j += d
        return ci

    cur = 0
    while 0 <= cur < len(selected):
        frame_idx = selected[cur]
        path = rgb_paths[frame_idx]
        stem = os.path.splitext(os.path.basename(path))[0]
        out_json = os.path.join(out_dir, f"{stem}.json")
        out_png  = os.path.join(out_dir, f"{stem}.png")

        # 프레임 reset + 기존 라벨 로드
        s.img = cv2.imread(path)
        s.img_shape = s.img.shape
        s.kps_2d = [None] * 9
        s.extrap_mask = [False] * 9    # v7: 외삽 점 표시 (t/x 입력 시 True)
        s.active = 0
        s.pose = None
        s.zoom = 1.0
        s.pan = [0, 0]
        s.dirty = False
        s.split = args.default_split   # 기본 split; 기존 JSON 있으면 load 가 override
        if load_existing_annotation(s, out_json):
            update_pose(s, K)
        if len(selected) > 1:
            cv2.setTrackbarPos("frame", win, cur)   # 슬라이더를 현재 위치에 동기화

        # 메인 루프 (한 프레임)
        next_action = None
        prev_ao = s.annot_only
        while next_action is None:
            update_pose(s, K)
            vis = render(s, cur, len(selected), stem)
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF

            # 위젯(ANNOT-ONLY 버튼 / frame 슬라이더)은 키 입력이 없을 때(255)만 폴링한다.
            # 키 처리 前에 폴링하면 실제 키입력(예: 's' 저장)을 삼킬 수 있어 분리.
            if key == 255:
                # ANNOT-ONLY 버튼 토글 감지 (마우스 콜백이 s.annot_only 변경)
                if s.annot_only != prev_ao:
                    prev_ao = s.annot_only
                    if s.annot_only and not _has_annot(cur):   # 켜면 가까운 어노 frame 으로
                        nj = _step_annot(cur, +1)
                        if nj == cur:
                            nj = _step_annot(cur, -1)
                        if nj != cur:
                            s.goto = nj
                            next_action = 'goto'
                # frame 슬라이더 드래그/클릭 점프
                if next_action is None and len(selected) > 1:
                    tb = cv2.getTrackbarPos("frame", win)
                    if tb != cur:
                        s.goto = tb
                        next_action = 'goto'
                continue

            # ── Goto 번호 입력 모드 (G/: 로 진입, 숫자 타이핑 후 Enter) ──
            if s.goto_mode:
                if ord('0') <= key <= ord('9'):
                    s.goto_buf += chr(key)
                elif key in (13, 10):                       # Enter = 점프
                    if s.goto_buf:
                        s.goto = max(0, min(len(selected) - 1, int(s.goto_buf) - 1))
                        s.goto_mode = False
                        next_action = 'goto'
                        continue
                    s.goto_mode = False
                elif key == 27:                             # Esc = 취소
                    s.goto_mode = False
                elif key in (8, 127):                       # Backspace
                    s.goto_buf = s.goto_buf[:-1]
                continue
            if key in (ord('G'), ord(':')):
                s.goto_mode = True
                s.goto_buf = ""
                continue

            # ── Mode toggle ──
            if key == ord('m'):
                if s.mode == "click":
                    if s.pose is None:
                        print("[WARN] PnP 가 아직 안 풀려서 manipulate 진입 불가. 4점 이상 필요.")
                        continue
                    s.mode = "manip"
                    s.locked_pose = {"R": s.pose["R"].copy(), "t": s.pose["t"].copy()}
                    print("[Mode] CLICK → MANIPULATE")
                else:
                    s.mode = "click"
                    s.locked_pose = None
                    print("[Mode] MANIPULATE → CLICK")
                continue

            # ── Mode-specific dispatcher ──
            if s.mode == "manip":
                next_action = _handle_manip_key(key, s, out_json, out_png, path, K)
            else:
                next_action = _handle_click_key(key, s, out_json, out_png, path, K)

        # 다음 frame 결정
        if next_action == 'quit':
            break
        elif next_action == 'save-next':
            cur += 1                                    # 저장 후엔 항상 다음 sequential frame
        elif next_action == 'next':
            cur = _step_annot(cur, +1) if s.annot_only else cur + 1
        elif next_action == 'prev':
            cur = _step_annot(cur, -1) if s.annot_only else max(0, cur - 1)
        elif next_action == 'jump-10':
            cur = max(0, cur - 10)
        elif next_action == 'jump+10':
            cur = min(len(selected) - 1, cur + 10)
        elif next_action == 'goto':
            if s.goto is not None:
                cur = max(0, min(len(selected) - 1, s.goto))
            s.goto = None

    cv2.destroyAllWindows()
    saved = len(glob.glob(os.path.join(out_dir, "*.json")))
    print(f"\n[Done] saved={saved} JSON files in {out_dir}")


if __name__ == "__main__":
    main()
