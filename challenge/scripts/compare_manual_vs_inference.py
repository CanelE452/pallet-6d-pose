"""Manual annotation 점 vs 모델 추론 점 인터랙티브 비교.

각 frame 에 표시:
  • DOPE 추론 keypoint        — 색깔 채운 원 + 번호 (KP_COLORS, 기존 visualize_inference 와 동일)
  • PnP cuboid wireframe       — 노란 선 (추론 → PnP)
  • Belief heatmap             — 30% opacity overlay
  • manual_kps (사용자 클릭)   — 파란 십자 + "M" 라벨
  • projected_cuboid (GT box)  — 녹색 wireframe + 녹색 십자 (corner)
  • reproj error (manual vs PnP) — 좌상단 표시

키 컨트롤 (GUI 활성화 후):
  SPACE  pause/resume (자동 재생 모드 → 멈춤)
  n      다음 frame
  p      이전 frame
  r      처음 frame 으로 (다시시작)
  s      현재 frame 저장 (output_dir)
  q      종료

사용:
  python challenge/scripts/compare_manual_vs_inference.py \\
      --gt_dir challenge/data/01_real/manual_gt/capturepallet07_manual_gt \\
      --weights weights/challenge/final_net_epoch_0060.pth
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

# 기존 shared lib 재사용
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "data_prep"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "self_training"))

from visualize_inference import (
    load_model, infer, extract_keypoints, draw_overlay, CUBOID_EDGES,
)
from pnp_solver import PalletPnPSolver, make_camera_matrix


def overlay_manual_and_gt_box(vis, manual_kps, proj_cuboid, centroid):
    """Manual click 점 (파란 십자) + GT cuboid wireframe (녹색)."""
    # GT cuboid wireframe (projected_cuboid 8 corner)
    if proj_cuboid and len(proj_cuboid) >= 8:
        pts = [tuple(map(int, p)) if p and len(p) >= 2 and p[0] >= 0 else None
               for p in proj_cuboid[:8]]
        for i0, i1 in CUBOID_EDGES:
            if pts[i0] is None or pts[i1] is None:
                continue
            cv2.line(vis, pts[i0], pts[i1], (0, 220, 0), 1)
        # corner 녹색 십자
        for i, p in enumerate(pts):
            if p is None:
                continue
            cv2.drawMarker(vis, p, (0, 220, 0), cv2.MARKER_CROSS, 10, 1)

    # Manual click 점 (파란 십자 + "M" 라벨)
    if manual_kps:
        for i, kp in enumerate(manual_kps):
            if kp is None:
                continue
            x, y = kp
            if x < 0 or y < 0:
                continue
            pt = (int(x), int(y))
            cv2.drawMarker(vis, pt, (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(vis, f"M{i}", (pt[0] + 6, pt[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)


def manual_vs_pred_reproj(manual_kps, pred_kps_orig):
    """Manual click 점들과 예측 점의 L2 거리 평균."""
    if not manual_kps or not pred_kps_orig:
        return None
    errs = []
    n = min(len(manual_kps), len(pred_kps_orig))
    for i in range(n):
        m = manual_kps[i]
        p = pred_kps_orig[i]
        if m is None or p is None or m[0] < 0:
            continue
        errs.append(((m[0] - p[0]) ** 2 + (m[1] - p[1]) ** 2) ** 0.5)
    if not errs:
        return None
    return float(np.mean(errs))


def render_frame(img, pred_kps_grid, belief, pnp, manual_kps, proj_cuboid, centroid, label):
    """draw_overlay 호출 + manual/GT 오버레이 추가."""
    vis = draw_overlay(img, pred_kps_grid, None, belief, pnp, label)
    overlay_manual_and_gt_box(vis, manual_kps, proj_cuboid, centroid)

    # manual vs pred 거리 계산 (원본 해상도)
    h, w = img.shape[:2]
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = w / bw, h / bh
    pred_orig = [(kp[0] * sx, kp[1] * sy) if kp else None for kp in pred_kps_grid]
    err = manual_vs_pred_reproj(manual_kps, pred_orig)

    # 우상단 정보
    y = 50
    if err is not None:
        cv2.putText(vis, f"manual vs pred (mean L2): {err:.1f} px",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, f"manual vs pred (mean L2): {err:.1f} px",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # 범례
    cv2.putText(vis, "color dot = inference  |  M cross = manual click  |  green box = GT cuboid",
                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(vis, "color dot = inference  |  M cross = manual click  |  green box = GT cuboid",
                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True,
                    help="manual_gt 폴더 (예: challenge/data/01_real/manual_gt/capturepallet07_manual_gt)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--fx", type=float, default=614.18)
    ap.add_argument("--fy", type=float, default=614.31)
    ap.add_argument("--cx", type=float, default=329.28)
    ap.add_argument("--cy", type=float, default=234.53)
    ap.add_argument("--output_dir", default="data/pallet/eval_results/compare_manual",
                    help="s 키로 저장될 폴더")
    ap.add_argument("--fps", type=float, default=2.0, help="auto-play fps")
    args = ap.parse_args()

    pngs = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    pngs = [p for p in pngs if os.path.exists(os.path.splitext(p)[0] + ".json")]
    if not pngs:
        print(f"[ERROR] No .png+.json pairs in {args.gt_dir}")
        sys.exit(1)
    print(f"[Compare] {len(pngs)} frames in {args.gt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)
    print(f"[Model] {args.weights} loaded on {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    win = "manual_vs_inference"
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
        manual_kps = obj.get("manual_kps")
        proj_cuboid = obj.get("projected_cuboid")
        centroid = obj.get("projected_cuboid_centroid")

        img = cv2.imread(png)
        belief = infer(model, img, device)
        pred_kps_grid = extract_keypoints(belief, args.threshold)

        label = f"[{idx+1}/{len(pngs)}] {stem}  (q/n/p/r/s/SPACE)"
        vis = render_frame(img, pred_kps_grid, belief, pnp,
                           manual_kps, proj_cuboid, centroid, label)
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
            out_path = os.path.join(args.output_dir, f"{stem}.jpg")
            cv2.imwrite(out_path, vis)
            print(f"  saved: {out_path}")
        else:
            # auto-play 모드: timeout 이면 다음 frame
            if not paused:
                idx = (idx + 1) % len(pngs)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
