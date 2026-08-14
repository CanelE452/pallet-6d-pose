"""
challenge/scripts/dataset/make_apriltag_gt.py

pallet11 시퀀스(또는 다른 AprilTag 포함 시퀀스)를 batch 처리해 NDDS GT JSON 생성.

전제:
  - Tag tag36h11, size=200mm, 팔레트 앞면 중앙(포크 들어가는 면) 부착
  - Tag 좌표축이 pallet 축과 일치 (X=right, Y=down, Z=forward, OpenCV convention)
  - Tag center pallet local 좌표: (0, 0, +depth/2) = (0, 0, +0.55m)

사용:
  python challenge/scripts/dataset/make_apriltag_gt.py
  python challenge/scripts/dataset/make_apriltag_gt.py --seq data/outside/capturepallet11 --vis_every 50
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]


import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))   # pnp_solver
sys.path.insert(0, os.path.join(_REPO, "scripts", "data_prep", "apriltag"))
sys.path.insert(0, _REPO)
from challenge.data_paths import get as _dp  # 경로 단일 출처

from apriltag_gt import (    create_detector, detect_tag_pose, tag_pose_to_pallet_pose,
    project_cuboid, save_gt_annotation, draw_overlay,
)


PALLET_DIMS = (1.1, 1.3, 0.11)    # (W=앞면폭, D=깊이, H=두께) — 실측 110×130×11cm
TAG_SIZE_M  = 0.200               # 사용자 실측

# Tag → Pallet 변환 (Pallet OpenCV convention)
# Pallet origin = centroid, 앞면 = Z_max
# Tag center 위치 (pallet local): (0, 0, +depth/2)
T_PALLET_FROM_TAG = np.eye(4, dtype=np.float64)
T_PALLET_FROM_TAG[2, 3] = PALLET_DIMS[1] / 2.0   # +0.65m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",      default="data/outside/capturepallet11")
    ap.add_argument("--out_dir",  default=_dp("manual.pallet11"))
    ap.add_argument("--vis_dir",  default=None, help="overlay 저장 경로 (기본: out_dir/_overlay)")
    ap.add_argument("--vis_every", type=int, default=50, help="N 프레임마다 overlay 저장")
    ap.add_argument("--min_margin", type=float, default=15.0,
                    help="AprilTag decision_margin 임계값 (낮을수록 noise 위험)")
    ap.add_argument("--max_z_m",    type=float, default=4.0, help="너무 멀어서 픽셀 부정확한 frame 제외")
    ap.add_argument("--copy_rgb",   action="store_true",
                    help="학습용 png를 GT 디렉토리에 복사 (기본: hardlink 시도, 실패시 copy)")
    args = ap.parse_args()

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO, args.seq)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_REPO, args.out_dir)
    vis_dir = args.vis_dir or os.path.join(out_dir, "_overlay")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    K_path = os.path.join(seq, "cam_K.txt")
    if not os.path.isfile(K_path):
        print(f"[ERROR] cam_K.txt not found: {K_path}")
        return
    K = np.loadtxt(K_path).reshape(3, 3)
    cam = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
    if not rgb_paths:
        print(f"[ERROR] No rgb frames in {seq}/rgb/")
        return

    print(f"[Seq] {seq}")
    print(f"[Out] {out_dir}")
    print(f"[Vis] {vis_dir} (every {args.vis_every} frames)")
    print(f"[Tag] family=tag36h11 size={TAG_SIZE_M*1000:.0f}mm  "
          f"min_margin={args.min_margin}  max_z={args.max_z_m}m")
    print(f"[Pallet] dims={PALLET_DIMS}  T_pallet_from_tag.t={T_PALLET_FROM_TAG[:3, 3]}")
    print(f"[Total] {len(rgb_paths)} frames\n")

    detector = create_detector()

    ok = 0
    skip_nodetect = 0
    skip_margin = 0
    skip_far = 0
    t0 = time.time()

    for i, p in enumerate(rgb_paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detect_tag_pose(gray, detector, cam, TAG_SIZE_M)
        if not dets:
            skip_nodetect += 1
            continue
        det = max(dets, key=lambda d: d["decision_margin"])
        if det["decision_margin"] < args.min_margin:
            skip_margin += 1
            continue

        T_cam_from_pallet = tag_pose_to_pallet_pose(det["T_cam_from_tag"], T_PALLET_FROM_TAG)
        z_m = float(T_cam_from_pallet[2, 3])
        if z_m > args.max_z_m or z_m < 0.2:
            skip_far += 1
            continue

        cuboid, centroid = project_cuboid(T_cam_from_pallet, K, PALLET_DIMS)

        # NDDS JSON 저장
        out_json = os.path.join(out_dir, f"{stem}.json")
        save_gt_annotation(out_json, img.shape, cam, T_cam_from_pallet,
                           cuboid, centroid, det, PALLET_DIMS)
        # 학습용 png 도 같이 (hardlink → 실패시 copy)
        out_img = os.path.join(out_dir, f"{stem}.png")
        if not os.path.exists(out_img):
            try:
                if args.copy_rgb:
                    shutil.copy2(p, out_img)
                else:
                    os.link(p, out_img)
            except (OSError, NotImplementedError):
                shutil.copy2(p, out_img)
        ok += 1

        # Overlay 검증
        if i % args.vis_every == 0:
            vis = draw_overlay(img, cuboid, centroid, det["corners"])
            cv2.putText(vis, f"frame {i}  z={z_m:.2f}m  margin={det['decision_margin']:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(vis_dir, f"{stem}_overlay.png"), vis)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{len(rgb_paths)}] ok={ok} skip(nodetect={skip_nodetect} "
                  f"margin={skip_margin} far={skip_far}) — {fps:.1f} fps")

    total = ok + skip_nodetect + skip_margin + skip_far
    print(f"\n=== Done ({time.time() - t0:.1f}s) ===")
    print(f"  GT frames:        {ok}/{total} ({ok/max(total,1):.1%})")
    print(f"  No-detect:        {skip_nodetect}")
    print(f"  Low margin:       {skip_margin}")
    print(f"  Out-of-z-range:   {skip_far}")
    print(f"  Saved to:         {out_dir}")
    print(f"  Overlay samples:  {vis_dir}")


if __name__ == "__main__":
    main()
