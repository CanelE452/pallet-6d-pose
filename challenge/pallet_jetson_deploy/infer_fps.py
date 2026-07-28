"""infer_fps.py — Jetson 배포용 단일 자체완결 YOLO26-pose 팔레트 추론 + FPS 측정.

standalone (repo 모듈 import 금지). 검증된 로직을 그대로 복제:
  - pad+shift 추론   : eval_ab_crop.predict (100px reflect pad → predict → (-pad) shift,
                       box conf 최대 instance 선택)
  - 9 keypoint 순서  : annotate camera-facing (0~3 near, 4~7 far, 8 centroid)
  - 3D 모델/dims     : annotate_pnp.make_pallet_keypoints_3d, PALLET_DIMS=(1.1,1.3,0.11)
  - PnP = SQPnP      : eval_ab_crop.solve_pnp (SOLVEPNP_SQPNP + RefineLM, conf>=kp_conf,
                       n>=6, median reproj>12px 실패)
  - cuboid edges     : EDGES / keypoint 색상 KP_COLORS / centroid·axes 그리기
  - reproj_from_pose : PnP 성공 시 cuboid 투영

좌표 convention: OpenCV (X=right, Y=down, Z=forward). 절대 변경 금지.

FPS 측정:
  (a) inference-only (model.predict 구간)  (b) full pipeline (pad+predict+pnp+draw)
  각각 rolling mean FPS + ms. 종료 시 표 형태 요약.

사용 (engine 추론 시 LD_LIBRARY_PATH 필요):
  ENV=/home/minjae/anaconda3/envs/pallet-yolo26
  export LD_LIBRARY_PATH=$ENV/lib/python3.10/site-packages/torch/lib:\
$ENV/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
  $ENV/bin/python pallet_jetson_deploy/infer_fps.py \
      --model runs/.../best_fp16_640.engine \
      --source data/outside/forklift_raw_20260528_163408.mp4 \
      --cam-k  data/outside/forklift_raw_20260528_163408/cam_K.txt \
      --max-frames 100
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ── 고정 상수 (eval_ab_crop / annotate_pnp / annotate_draw 복제) ──────────────
# cam_K fallback (--cam-k 미지정 시). RealSense D435i 640x480.
CAM_K_FALLBACK = np.array([[614.18, 0, 329.28],
                           [0, 614.31, 234.53],
                           [0, 0, 1]], dtype=np.float64)

# (width, depth, height) — 실측 plastic 팔레트 110 × 130 × 11 cm
PALLET_DIMS = (1.1, 1.3, 0.11)

# cuboid wireframe edges (annotate camera-facing face convention)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),   # near face
         (4, 5), (5, 6), (6, 7), (7, 4),   # far face
         (0, 4), (1, 5), (2, 6), (3, 7)]   # connectors

# keypoint 색상 (annotate_draw.KP_COLORS, BGR)
KP_COLORS = [
    (0,   0, 255),   # 0 red
    (0, 128, 255),   # 1 orange
    (0, 255, 255),   # 2 yellow
    (0, 255,   0),   # 3 green
    (255, 255,   0),  # 4 cyan
    (255,   0,   0),  # 5 blue
    (255,   0, 128),  # 6 magenta
    (128,  0, 255),   # 7 purple
    (255, 255, 255),  # 8 white centroid
]

CUBOID_YELLOW = (0, 255, 255)  # BGR
SQPNP_MAX_MED_REPROJ = 12.0    # px. SQPnP 후 median reproj 이보다 크면 실패 처리


# ── 3D 모델 (annotate_pnp.make_pallet_keypoints_3d 그대로) ────────────────────
def make_pallet_keypoints_3d(width=1.1, depth=1.3, height=0.11):
    """Camera-facing convention 9-keypoint 3D 모델.

    cuboid local frame: X=right(+), Y=down(OpenCV +y=bottom), Z=forward(+).
      0: near-top-LEFT       (-w/2, -h/2, -d/2)   near = Z_local 작은 쪽
      1: near-top-RIGHT      (+w/2, -h/2, -d/2)
      2: near-bottom-RIGHT   (+w/2, +h/2, -d/2)
      3: near-bottom-LEFT    (-w/2, +h/2, -d/2)
      4: far-top-LEFT        (-w/2, -h/2, +d/2)   far = Z_local 큰 쪽
      5: far-top-RIGHT       (+w/2, -h/2, +d/2)
      6: far-bottom-RIGHT    (+w/2, +h/2, +d/2)
      7: far-bottom-LEFT     (-w/2, +h/2, +d/2)
      8: centroid
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, -d],   # 0 near-top-LEFT
        [+w, -h, -d],   # 1 near-top-RIGHT
        [+w, +h, -d],   # 2 near-bottom-RIGHT
        [-w, +h, -d],   # 3 near-bottom-LEFT
        [-w, -h, +d],   # 4 far-top-LEFT
        [+w, -h, +d],   # 5 far-top-RIGHT
        [+w, +h, +d],   # 6 far-bottom-RIGHT
        [-w, +h, +d],   # 7 far-bottom-LEFT
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])


# ── 추론 (eval_ab_crop.predict 복제 — pad+shift, box conf 최대) ───────────────
def predict(model, img, pad, conf, imgsz):
    """반환: pred_kps(9,2), pred_conf(9,), predict_only_sec. 실패 시 (None,None,t)."""
    if pad > 0:
        inp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    else:
        inp = img
    t0 = time.perf_counter()
    r = model.predict(inp, verbose=False, conf=conf, imgsz=imgsz)[0]
    predict_sec = time.perf_counter() - t0
    if r.keypoints is None or len(r.keypoints) == 0:
        return None, None, predict_sec
    allkp = r.keypoints.data.cpu().numpy().astype(np.float64)  # (N,9,3)
    if pad > 0:
        allkp = allkp.copy()
        allkp[:, :, 0] -= pad
        allkp[:, :, 1] -= pad
    if allkp.shape[0] == 1:
        bi = 0
    elif r.boxes is not None:
        bi = int(np.argmax(r.boxes.conf.cpu().numpy()))  # box conf 최대 instance
    else:
        bi = 0
    kp = allkp[bi]
    return kp[:, :2].copy(), kp[:, 2].copy(), predict_sec


# ── PnP = SQPnP (eval_ab_crop.solve_pnp 복제) ─────────────────────────────────
def solve_pnp(kps_2d, kp_conf, kp_conf_thr, dims, K):
    """conf>=kp_conf 점만 SQPnP 직접 풀이. 반환 (ok, R, t, n_used).

    SQPnP(SOLVEPNP_SQPNP)는 n>=3 지원·비최소·전역최적이라 RANSAC 불필요.
    풀이 후 median reproj 임계로 outlier 안전장치. n<6 이면 실패.
    좌표 convention / 3D 모델 / dims / keypoint 순서는 일절 변경 안 함.
    """
    kp3d = make_pallet_keypoints_3d(*dims)        # (9,3)
    dist = np.zeros((5, 1), dtype=np.float64)     # distortion 없음

    obj_pts, img_pts = [], []
    for i in range(9):
        if kp_conf[i] >= kp_conf_thr:
            obj_pts.append(kp3d[i])
            img_pts.append([float(kps_2d[i, 0]), float(kps_2d[i, 1])])
    n = len(obj_pts)
    if n < 6:
        return False, None, None, n

    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return False, None, None, n

    try:  # 1-step LM 정제
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    except cv2.error:
        pass

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    med_reproj = float(np.median(
        np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
    if med_reproj > SQPNP_MAX_MED_REPROJ:
        return False, None, None, n

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    if t[2] < 0:  # 카메라 뒤로 풀리면 부호 뒤집기
        t, R = -t, -R
    return True, R, t, n


def reproj_from_pose(R, t, dims, K):
    """cuboid 9-keypoint 를 R,t 로 화면 투영 (eval_ab_crop._reproj/reproj_from_pose)."""
    kp3d = make_pallet_keypoints_3d(*dims)
    Pc = (R @ kp3d.T).T + t
    z = Pc[:, 2]
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], 1)


# ── 그리기 (eval_ab_crop.draw_cuboid + infer_video_yolo.draw_keypoints) ───────
def draw_cuboid(img, kps, color, thick=2):
    for a, b in EDGES:
        pa, pb = kps[a], kps[b]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 color, thick, cv2.LINE_AA)


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


def draw_axes(img, R, t, K, length=0.5):
    """centroid 기준 XYZ 축 (X=red, Y=green, Z=blue, OpenCV convention)."""
    origin = make_pallet_keypoints_3d(*PALLET_DIMS)[8]  # local centroid (=0,0,0)
    ends = origin + np.eye(3) * length
    pts3d = np.vstack([origin, ends])
    Pc = (R @ pts3d.T).T + t
    if (Pc[:, 2] <= 0).any():
        return
    uv = np.stack([K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2],
                   K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]], 1)
    o = (int(uv[0, 0]), int(uv[0, 1]))
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X red, Y green, Z blue
    for k in range(3):
        e = (int(uv[k + 1, 0]), int(uv[k + 1, 1]))
        cv2.arrowedLine(img, o, e, cols[k], 2, cv2.LINE_AA, tipLength=0.15)


def draw_hud(img, lines):
    y = 18
    for txt, c in lines:
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    c, 1, cv2.LINE_AA)
        y += 20


# ── source iterator (mp4 / 이미지 폴더 / 카메라 id) ───────────────────────────
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def open_source(source):
    """반환: (frame_generator, total_or_-1, label)."""
    # 정수 = 카메라
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {source}")

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        return gen(), -1, f"camera:{source}"

    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source)
                       if f.lower().endswith(_IMG_EXT))
        paths = [os.path.join(source, f) for f in files]

        def gen():
            for p in paths:
                f = cv2.imread(p)
                if f is not None:
                    yield f
        return gen(), len(paths), f"imgdir:{source}"

    # mp4 / video file
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def gen():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
    return gen(), total, f"video:{source}"


def load_K(path):
    if path and os.path.exists(path):
        return np.loadtxt(path).reshape(3, 3).astype(np.float64)
    return CAM_K_FALLBACK.copy()


def device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help=".pt / .onnx / .engine (ultralytics YOLO 자동 처리)")
    ap.add_argument("--source",
                    default="data/outside/forklift_raw_20260528_163408.mp4",
                    help="mp4 / 이미지 폴더 / 카메라 id(정수)")
    ap.add_argument("--cam-k", dest="cam_k", default=None,
                    help="3x3 cam_K txt (np.loadtxt). 미지정 시 fallback")
    ap.add_argument("--pad", type=int, default=100, help="reflect pad 폭")
    ap.add_argument("--conf", type=float, default=0.4, help="detection conf")
    ap.add_argument("--kp-conf", dest="kp_conf", type=float, default=0.5,
                    help="keypoint vis thr (PnP/그리기)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true",
                    help="cv2 창 실시간 표시 (없으면 headless, FPS만 측정)")
    ap.add_argument("--save", default=None, help="결과 mp4 저장 경로 (옵션)")
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                    help="0=전체")
    ap.add_argument("--warmup", type=int, default=5,
                    help="측정 제외 워밍업 프레임 수")
    args = ap.parse_args()

    K = load_K(args.cam_k)
    use_pad = args.pad
    dims = PALLET_DIMS

    from ultralytics import YOLO
    print(f"[load] model = {args.model}")
    model = YOLO(args.model)
    dev = device_name()
    print(f"[device] {dev}")
    print(f"[cam_K]\n{K}")

    frames, total, label = open_source(args.source)
    print(f"[source] {label}  total={total}  pad={args.pad} "
          f"conf={args.conf} kp_conf={args.kp_conf} imgsz={args.imgsz}")

    writer = None
    win = "infer_fps"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # rolling (워밍업 이후 측정값만 누적)
    roll_n = 60
    inf_ms_roll = deque(maxlen=roll_n)
    full_ms_roll = deque(maxlen=roll_n)
    # 전체 누적 (요약용, 워밍업 제외)
    inf_ms_all, full_ms_all = [], []
    pad_ms_all, pred_ms_all, pnp_ms_all, draw_ms_all = [], [], [], []

    n_det = n_pnp = 0
    fi = 0
    measured = 0
    for frame in frames:
        if args.max_frames > 0 and fi >= args.max_frames:
            break
        warming = fi < args.warmup
        t_full0 = time.perf_counter()

        # ── pad + predict ──
        t_pad0 = time.perf_counter()
        if use_pad > 0:
            _ = None  # pad 는 predict 내부에서 수행 (시간 분리 위해 별도 측정)
        pad_sec = time.perf_counter() - t_pad0  # (pad 는 predict 안에 있어 ~0)

        kps, conf, predict_sec = predict(model, frame, use_pad, args.conf, args.imgsz)

        vis = frame.copy() if (args.show or args.save) else frame

        # ── PnP ──
        t_pnp0 = time.perf_counter()
        ok_pnp, R, t, n_pts = (False, None, None, 0)
        if kps is not None:
            ok_pnp, R, t, n_pts = solve_pnp(kps, conf, args.kp_conf, dims, K)
        pnp_sec = time.perf_counter() - t_pnp0

        # ── draw ──
        t_draw0 = time.perf_counter()
        if args.show or args.save:
            if kps is None:
                draw_hud(vis, [(f"frame {fi}", (255, 255, 255)),
                               ("NO DETECTION", (0, 0, 255))])
            else:
                n_used = int((conf >= args.kp_conf).sum())
                draw_keypoints(vis, kps, conf, args.kp_conf)
                hud = [(f"frame {fi}  DET kp>={args.kp_conf}:{n_used}/9",
                        (0, 255, 0))]
                if ok_pnp:
                    proj = reproj_from_pose(R, t, dims, K)
                    draw_cuboid(vis, proj, CUBOID_YELLOW, thick=2)
                    draw_axes(vis, R, t, K)
                    hud.append((f"PnP OK pts={n_pts} z={float(t[2]):.2f}m",
                                (0, 255, 255)))
                else:
                    hud.append((f"PnP FAIL pts={n_pts}", (0, 0, 255)))
                # inference FPS HUD
                if inf_ms_roll:
                    fps_inf = 1000.0 / (sum(inf_ms_roll) / len(inf_ms_roll))
                    hud.append((f"infer {fps_inf:.1f} FPS", (255, 255, 0)))
                draw_hud(vis, hud)
        draw_sec = time.perf_counter() - t_draw0

        full_sec = time.perf_counter() - t_full0

        if kps is not None:
            n_det += 1
        if ok_pnp:
            n_pnp += 1

        if not warming:
            measured += 1
            inf_ms = predict_sec * 1000.0
            full_ms = full_sec * 1000.0
            inf_ms_roll.append(inf_ms)
            full_ms_roll.append(full_ms)
            inf_ms_all.append(inf_ms)
            full_ms_all.append(full_ms)
            pad_ms_all.append(pad_sec * 1000.0)
            pred_ms_all.append(inf_ms)
            pnp_ms_all.append(pnp_sec * 1000.0)
            draw_ms_all.append(draw_sec * 1000.0)

        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            writer.write(vis)

        if args.show:
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if (fi + 1) % 50 == 0:
            cur = (1000.0 / (sum(inf_ms_roll) / len(inf_ms_roll))) if inf_ms_roll else 0
            print(f"  [{fi+1}] det={n_det} pnp={n_pnp} infFPS={cur:.1f}")
        fi += 1

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ── 요약 ──
    proc = fi
    print_summary(proc, measured, n_det, n_pnp, dev, args,
                  inf_ms_all, full_ms_all,
                  pad_ms_all, pred_ms_all, pnp_ms_all, draw_ms_all)
    if args.save:
        print(f"[out] video -> {os.path.abspath(args.save)}")


def _stat(arr):
    if not arr:
        return 0.0, 0.0
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(np.median(a))


def print_summary(proc, measured, n_det, n_pnp, dev, args,
                  inf_ms, full_ms, pad_ms, pred_ms, pnp_ms, draw_ms):
    inf_mean, inf_med = _stat(inf_ms)
    full_mean, full_med = _stat(full_ms)

    def fps(ms):
        return 1000.0 / ms if ms > 0 else 0.0

    pad_m = _stat(pad_ms)[0]
    pred_m = _stat(pred_ms)[0]
    pnp_m = _stat(pnp_ms)[0]
    draw_m = _stat(draw_ms)[0]

    W = 64
    print("\n" + "=" * W)
    print(" Inference FPS Summary".center(W))
    print("=" * W)
    print(f" {'Device':<22}: {dev}")
    print(f" {'Model':<22}: {os.path.basename(args.model)}")
    print(f" {'imgsz / pad':<22}: {args.imgsz} / {args.pad}")
    print(f" {'Frames (proc/meas)':<22}: {proc} / {measured}  (warmup {args.warmup})")
    print("-" * W)
    print(f" {'Inference-only FPS':<22}: mean {fps(inf_mean):7.2f} | "
          f"median {fps(inf_med):7.2f}")
    print(f" {'Inference-only ms':<22}: mean {inf_mean:7.2f} | "
          f"median {inf_med:7.2f}")
    print(f" {'Full-pipeline FPS':<22}: mean {fps(full_mean):7.2f} | "
          f"median {fps(full_med):7.2f}")
    print(f" {'Full-pipeline ms':<22}: mean {full_mean:7.2f} | "
          f"median {full_med:7.2f}")
    print("-" * W)
    print(" ms breakdown (mean):")
    print(f"   {'pad':<10}: {pad_m:7.3f}")
    print(f"   {'predict':<10}: {pred_m:7.3f}")
    print(f"   {'pnp':<10}: {pnp_m:7.3f}")
    print(f"   {'draw':<10}: {draw_m:7.3f}")
    print("-" * W)
    print(f" {'Detection rate':<22}: {n_det}/{proc} = "
          f"{100*n_det/max(1,proc):.1f}%")
    print(f" {'PnP success rate':<22}: {n_pnp}/{proc} = "
          f"{100*n_pnp/max(1,proc):.1f}%  "
          f"(of detected {100*n_pnp/max(1,n_det):.1f}%)")
    print("=" * W)


if __name__ == "__main__":
    main()
