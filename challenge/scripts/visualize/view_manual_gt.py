"""Manual annotation 만 인터랙티브로 표시 (모델 추론 X).

각 frame 에 표시:
  • manual_kps (사용자 클릭 9 점)        — 색 채운 원 + 번호
  • projected_cuboid (PnP reproject)     — 녹색 wireframe + 코너 십자
  • centroid                              — 흰 원 "C"
  • XYZ axis (pose_transform 기반)        — R/G/B 화살표
  • reproj_error_px (JSON 에 저장된 값)   — 좌상단
  • dimensions_m (PnP 풀 때 쓴 dims)      — 좌상단

키:
  SPACE  pause/resume
  n      다음 frame
  p      이전 frame
  r      처음 frame
  s      현재 frame 저장
  q      종료

사용:
  python challenge/scripts/visualize/view_manual_gt.py \\
      --gt_dir challenge/data/01_real/manual_gt/capturepallet07_manual_gt
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]


import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


# 0~7 corner colors (BGR), 8 = centroid
CORNER_COLORS = [
    (0, 0, 255),    # 0: FrontTopLeft   - red
    (0, 128, 255),  # 1: FrontTopRight  - orange
    (0, 255, 255),  # 2: FrontBottomLeft- yellow
    (0, 255, 0),    # 3: FrontBottomRight - green
    (255, 0, 0),    # 4: RearTopRight   - blue
    (255, 128, 0),  # 5: RearTopLeft    - teal
    (255, 0, 128),  # 6: RearBottomLeft - purple
    (255, 0, 255),  # 7: RearBottomRight- magenta
]
CENTROID_COLOR = (255, 255, 255)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),     # front face
    (4, 5), (5, 6), (6, 7), (7, 4),     # rear face
    (0, 4), (1, 5), (2, 6), (3, 7),     # connecting
]


def draw_cuboid_wireframe(img, pts):
    """projected_cuboid 8 corner -> wireframe + 코너 십자 + 번호."""
    h, w = img.shape[:2]
    valid = [None if p is None or p[0] < 0 else (int(p[0]), int(p[1])) for p in pts[:8]]
    for i0, i1 in CUBOID_EDGES:
        if valid[i0] is None or valid[i1] is None:
            continue
        cv2.line(img, valid[i0], valid[i1], (0, 220, 0), 2)
    for i, p in enumerate(valid):
        if p is None:
            continue
        cv2.drawMarker(img, p, (0, 220, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, str(i), (p[0] + 6, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)


def draw_manual_clicks(img, manual_kps):
    """사용자가 클릭한 9 점 (None / [-1,-1] 은 skip)."""
    if not manual_kps:
        return
    for i, kp in enumerate(manual_kps):
        if kp is None or kp[0] < 0:
            continue
        pt = (int(kp[0]), int(kp[1]))
        color = CORNER_COLORS[i] if i < 8 else CENTROID_COLOR
        cv2.circle(img, pt, 7, color, -1)
        cv2.circle(img, pt, 8, (0, 0, 0), 1)
        label = "C" if i == 8 else f"M{i}"
        cv2.putText(img, label, (pt[0] + 9, pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_pose_axes(img, pose_transform, K_mat, axis_len_m=0.3):
    """pose_transform (4x4) 의 XYZ 축을 이미지에 화살표로."""
    if pose_transform is None:
        return
    pose = np.array(pose_transform, dtype=np.float64)
    if pose.shape != (4, 4):
        return
    R = pose[:3, :3]
    t = pose[:3, 3]
    if t[2] <= 0:
        return

    origin_h = K_mat @ t
    origin_px = (origin_h[:2] / origin_h[2]).astype(int)

    h, w = img.shape[:2]
    if not (0 <= origin_px[0] < w and 0 <= origin_px[1] < h):
        return

    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # X red, Y green, Z blue
    labels = ["X", "Y", "Z"]
    # 시각 직관 (Y=UP) 을 위해 Y axis 만 부호 뒤집어 그림. 실제 6D pose 는 OpenCV (Y=DOWN) 그대로.
    sign = [1, -1, 1]
    for i in range(3):
        end_3d = t + R[:, i] * axis_len_m * sign[i]
        end_h = K_mat @ end_3d
        if end_h[2] <= 0:
            continue
        end_px = (end_h[:2] / end_h[2]).astype(int)
        cv2.arrowedLine(img, tuple(origin_px), tuple(end_px),
                        axis_colors[i], 3, tipLength=0.15)
        cv2.putText(img, labels[i], (end_px[0] + 5, end_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_colors[i], 2)


def render(img, obj, K_mat, idx, total, stem):
    vis = img.copy()
    pc = obj.get("projected_cuboid") or []
    centroid = obj.get("projected_cuboid_centroid")
    manual = obj.get("manual_kps")
    pose = obj.get("pose_transform")
    reproj = obj.get("reproj_error_px")
    dims = obj.get("dimensions_m")

    draw_cuboid_wireframe(vis, pc)
    draw_pose_axes(vis, pose, K_mat)
    draw_manual_clicks(vis, manual)
    if centroid:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(vis, (cx, cy), 9, CENTROID_COLOR, -1)
        cv2.circle(vis, (cx, cy), 10, (0, 0, 0), 2)

    # 정보 텍스트
    title = f"[{idx+1}/{total}] {stem}"
    info1 = ""
    if dims:
        info1 = f"dims (W H L) = {dims.get('width','?')} x {dims.get('height','?')} x {dims.get('depth','?')} m"
    info2 = f"reproj_error: {reproj:.2f} px" if reproj is not None else "reproj_error: n/a"

    for y, txt, color in [(25, title, (255, 255, 255)),
                          (50, info1, (255, 255, 0)),
                          (75, info2, (0, 255, 255))]:
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 범례
    legend = "green box = PnP reproject  |  color M# = manual click  |  X(red) Y(green) Z(blue) axes"
    cv2.putText(vis, legend, (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(vis, legend, (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--fx", type=float, default=614.18)
    ap.add_argument("--fy", type=float, default=614.31)
    ap.add_argument("--cx", type=float, default=329.28)
    ap.add_argument("--cy", type=float, default=234.53)
    ap.add_argument("--output_dir", default="data/pallet/eval_results/view_manual")
    ap.add_argument("--fps", type=float, default=1.5)
    args = ap.parse_args()

    pngs = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    pngs = [p for p in pngs if os.path.exists(os.path.splitext(p)[0] + ".json")]
    if not pngs:
        print(f"[ERROR] No .png+.json pairs in {args.gt_dir}")
        sys.exit(1)
    print(f"[View] {len(pngs)} frames in {args.gt_dir}")

    K_mat = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0, 0, 1]], dtype=np.float64)

    os.makedirs(args.output_dir, exist_ok=True)
    win = "manual_gt_view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    paused = False
    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))

    while True:
        png = pngs[idx]
        stem = os.path.splitext(os.path.basename(png))[0]
        with open(png.replace(".png", ".json")) as f:
            d = json.load(f)
        obj = d["objects"][0]
        img = cv2.imread(png)
        vis = render(img, obj, K_mat, idx, len(pngs), stem)
        if paused:
            cv2.putText(vis, "PAUSED", (vis.shape[1] - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            idx = (idx + 1) % len(pngs)
        elif key == ord('p'):
            idx = (idx - 1) % len(pngs)
        elif key == ord('r'):
            idx = 0
            paused = False
        elif key == ord('s'):
            out = os.path.join(args.output_dir, f"{stem}.jpg")
            cv2.imwrite(out, vis)
            print(f"  saved: {out}")
        else:
            if not paused:
                idx = (idx + 1) % len(pngs)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
