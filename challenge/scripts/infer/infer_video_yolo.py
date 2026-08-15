"""infer_video_yolo.py — YOLO26-pose 비디오 추론 → keypoint + PnP cuboid overlay.

cropaug 모델(100px reflect pad 학습) 영상 추론용. eval_ab_crop.py 의 검증된
convention 을 그대로 재사용한다:

  - pad+shift 추론   : eval_ab_crop.predict() (100px reflect pad → predict → (-pad) shift)
  - keypoint 순서    : annotate.py camera-facing (0~3 near, 4~7 far, 8 centroid)
  - PnP 3D 모델/dims : annotate_pnp.make_pallet_keypoints_3d / PALLET_DIMS=(1.1,1.3,0.11)
  - PnP solver       : scripts/self_training/pnp_solver (EPnP+RANSAC), eval_ab_crop.solve_pnp 경유
  - cuboid edges     : eval_ab_crop.EDGES, draw_cuboid()
  - keypoint 색상    : annotate_draw.KP_COLORS

프레임별:
  1. pad+shift 추론 → 9 keypoint + per-kp conf (없으면 NO DETECTION)
  2. conf>=kp_conf 점만 PnP. 6점 이상이면 풀이.
  3. overlay: keypoint(색+번호) + (성공시) cuboid wireframe(노란선) + HUD
  4. mp4 누적 + 대표 frame png

사용:
  /home/minjae/anaconda3/envs/pallet-yolo26/bin/python challenge/scripts/infer/infer_video_yolo.py \
      --weights runs/pose/challenge/weights/yolo26n_pose_v1_ft_cropaug/weights/best.pt \
      --source data/outside/forklift_raw_20260528_163408.mp4 \
      --pad 100 --out challenge/data/04_results/forklift_cropaug_infer.mp4
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-08-15 재편: 이 파일이 challenge/scripts/<계열>/ 로 한 단계 내려갔다.
# dirname 을 하나 더 감싸야 저장소 루트다 (계열 -> scripts -> challenge -> repo).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))

# 검증된 추론/PnP/그리기 로직 재사용 (A 모델 추론 경로와 동일)
import eval_ab_crop as ev  # predict, solve_pnp, draw_cuboid, EDGES, CAM_K, PAD
from annotate_pnp import PALLET_DIMS
from annotate_draw import KP_COLORS

CUBOID_YELLOW = (0, 255, 255)  # BGR


def load_K(path):
    if path and os.path.exists(path):
        K = np.loadtxt(path).reshape(3, 3).astype(np.float64)
        return K
    return ev.CAM_K.copy()


def draw_keypoints(img, kps, conf, kp_conf_thr):
    for i in range(9):
        u, v = int(round(kps[i, 0])), int(round(kps[i, 1]))
        if u < -50 or v < -50 or u > img.shape[1] + 50 or v > img.shape[0] + 50:
            continue
        col = KP_COLORS[i]
        used = conf[i] >= kp_conf_thr
        r = 5 if used else 3
        cv2.circle(img, (u, v), r, col, -1 if used else 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)


def draw_hud(img, lines, color=(255, 255, 255)):
    y = 18
    for txt, c in lines:
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    c, 1, cv2.LINE_AA)
        y += 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        _REPO, "runs/pose/challenge/weights/"
        "yolo26n_pose_v1_ft_cropaug/weights/best.pt"))
    ap.add_argument("--source", default=os.path.join(
        _REPO, "data/outside/forklift_raw_20260528_163408.mp4"))
    ap.add_argument("--cam_k", default=os.path.join(
        _REPO, "data/outside/forklift_raw_20260528_163408/cam_K.txt"))
    ap.add_argument("--pad", type=int, default=100,
                    help="reflect pad 폭 (cropaug 모델=100, no-pad 모델=0)")
    ap.add_argument("--out", default=os.path.join(
        _REPO, "challenge/data/04_results/forklift_cropaug_infer.mp4"))
    ap.add_argument("--frames_dir", default=None,
                    help="대표 frame png 디렉토리 (기본: <out>_frames)")
    ap.add_argument("--conf", type=float, default=0.1, help="detection conf")
    ap.add_argument("--kp_conf", type=float, default=0.5,
                    help="keypoint vis thr (PnP/그리기 기준)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--show", action="store_true",
                    help="저장 없이 cv2.imshow 실시간 재생 (q=종료, space=일시정지)")
    ap.add_argument("--n_sample_frames", type=int, default=6)
    ap.add_argument("--max_frames", type=int, default=0,
                    help=">0 이면 이 frame 수만 처리 (디버그)")
    args = ap.parse_args()

    frames_dir = args.frames_dir or (os.path.splitext(args.out)[0] + "_frames")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    K = load_K(args.cam_k)
    ev.CAM_K = K  # solve_pnp / reproj 가 모듈 전역 CAM_K 사용 → override
    ev.PAD = args.pad  # predict() 가 모듈 PAD 사용
    use_pad = args.pad > 0

    from ultralytics import YOLO
    print(f"[load] weights = {args.weights}")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[error] cannot open", args.source)
        sys.exit(1)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {W}x{H} total={total} pad={args.pad} K=\n{K}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    # 대표 frame 균등 샘플 인덱스
    n_proc = args.max_frames if args.max_frames > 0 else total
    sample_idx = set(np.linspace(0, max(0, n_proc - 1),
                                 args.n_sample_frames, dtype=int).tolist())

    n_det = n_pnp = 0
    used_counts = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and fi >= args.max_frames:
            break

        # 1) pad+shift 추론 (eval_ab_crop.predict — anchor 없이 box conf 최대 선택)
        kps, conf = ev.predict(model, frame, use_pad, args.kp_conf, args.conf)
        vis = frame.copy()

        if kps is None:
            draw_hud(vis, [(f"frame {fi}/{total}", (255, 255, 255)),
                           ("NO DETECTION", (0, 0, 255))])
        else:
            n_det += 1
            draw_keypoints(vis, kps, conf, args.kp_conf)
            n_used = int((conf >= args.kp_conf).sum())

            ok_pnp, R, t, n_pnp_pts = ev.solve_pnp(
                kps, conf, args.kp_conf, PALLET_DIMS)
            hud = [(f"frame {fi}/{total}  DET  kp>={args.kp_conf}:{n_used}/9",
                    (0, 255, 0))]
            if ok_pnp:
                n_pnp += 1
                used_counts.append(n_pnp_pts)
                proj = ev.reproj_from_pose(R, t, PALLET_DIMS)
                ev.draw_cuboid(vis, proj, CUBOID_YELLOW, thick=2)
                z = float(t[2])
                hud.append((f"PnP OK  pts={n_pnp_pts}  z={z:.2f}m",
                            (0, 255, 255)))
            else:
                hud.append((f"PnP FAIL  pts={n_pnp_pts}", (0, 0, 255)))
            draw_hud(vis, hud)

        writer.write(vis)
        if fi in sample_idx:
            p = os.path.join(frames_dir, f"frame_{fi:04d}.png")
            cv2.imwrite(p, vis)

        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{total}] det={n_det} pnp={n_pnp}")
        fi += 1

    cap.release()
    writer.release()

    proc = fi
    print("\n" + "=" * 56)
    print(f" Inference summary  ({proc} frames)")
    print("=" * 56)
    print(f"  detection rate : {n_det}/{proc} = "
          f"{100*n_det/max(1,proc):.1f}%")
    print(f"  PnP success    : {n_pnp}/{proc} = "
          f"{100*n_pnp/max(1,proc):.1f}%  "
          f"(of detected: {100*n_pnp/max(1,n_det):.1f}%)")
    if used_counts:
        print(f"  mean PnP kpts  : {np.mean(used_counts):.2f}  "
              f"(min {min(used_counts)}, max {max(used_counts)})")
    print("=" * 56)
    print(f"[out] video  -> {args.out}")
    print(f"[out] frames -> {frames_dir}")


if __name__ == "__main__":
    main()
