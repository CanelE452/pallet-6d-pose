"""DOPE mp4 inference WITH reflect-padding (train/infer consistency fix).

Problem this fixes
------------------
DOPE was trained on truncation crops that were reflect-padded so off-image
corners move back INSIDE the frame (pad_truncation_crops.py: pad symmetrically,
then resize the padded canvas back to 640x480). At inference the old
dope_predict_mp4.py fed the raw frame (only a 448 resize), so truncated corners
that fall outside the frame produce all-zero belief channels and are never
detected (measured det ~54%). YOLO solved the same problem by padding at
inference too (infer_video_yolo.py --pad 100, det 74.9%).

This script mirrors the YOLO inference padding for DOPE:

  raw frame (640x480)
    -> cv2.copyMakeBorder reflect, PAD px each side   (840x680 for PAD=100)
    -> resize back to 640x480                          (matches training canvas)
    -> DOPE infer (internal 448 resize)               (visualize_inference.infer)
    -> belief peaks in 448 belief space
    -> scale 448-belief -> 640x480 padded-resized canvas
    -> undo resize-back  (* (640+2P)/640 , * (480+2P)/480)
    -> undo pad          (- PAD)                       => ORIGINAL frame coords
    -> PnP (annotate_pnp.solve_pose: order-free 24-sym, ITERATIVE,
            dims 1.1/1.3/0.11, camera-facing v4) on original coords
    -> draw keypoints + cuboid wireframe on the ORIGINAL frame

PnP is order-free (solve_pose enumerates 24 cube symmetries + LM ITERATIVE
refine, picks the strict-invariant / min-reproj candidate), so the belief peak
ordering does not need to match the canonical corner ordering.

usage:
  python challenge/scripts/infer/dope_predict_mp4_pad.py \
      --weights weights/challenge_ft_otftrunc/final_net_epoch_0150.pth \
      --mp4 data/outside/forklift_raw_20260528_163408.mp4 \
      --out challenge/data/04_results/forklift_otftrunc_PAD_infer.mp4 --pad 100
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)  # annotate_pnp, convert_to_camera_facing_v4
sys.path.insert(0, os.path.join(_REPO, "scripts", "data_prep"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "train"))

from visualize_inference import load_model, infer, extract_keypoints, KP_COLORS
from annotate_pnp import solve_pose, make_pallet_keypoints_3d, project_3d, PALLET_DIMS
from cuboid_kp_refine import refine_keypoints  # K-free shape-prior refine (--refine)

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
CUBOID_YELLOW = (0, 255, 255)

# flip-TTA 평균 시 flip-back 점을 plain 점에 위치기반(Hungarian)으로 매칭할 때
# 같은 물리 코너로 인정하는 최대 거리(px). 정상 일관 코너는 ~15-35px, 잘못
# 매칭되는 코너는 ~250px+ 라 그 사이값. 초과하면 비일관으로 보고 평균에서 제외.
FLIP_MATCH_MAX_PX = 60.0


def infer_kps_orig(model, img, device, args):
    """원본 이미지 추론 → 9 keypoint (원본 px, None 보존)."""
    H, W = img.shape[:2]
    proc = pad_frame(img, args.pad, args.pad_mode)
    belief = infer(model, proc, device)
    peaks = extract_keypoints(belief, args.threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    return [belief_peak_to_orig(kp, (bh, bw), (H, W), args.pad)
            if kp is not None else None for kp in peaks]


def infer_kps_flip(model, img, device, args):
    """좌우 flip 이미지 추론 → x 만 되돌림(label swap 안 함) → 9 keypoint.

    camera-facing 0123 은 카메라기준 동적 convention 이라 yaw!=0 에서 flip
    하면 모델의 좌/우/상/하 index 배정 자체가 바뀐다 → 고정 대칭 swap 가정이
    대각선 포즈에서 깨진다(진단 확인: 000108 에서 swap 후 250px+ 오차).
    여기서는 swap 하지 않고, flip-back 점(같은 물체라 x 되돌리면 원위치 근처)
    을 그대로 반환. plain 과의 정합은 avg_consistent 가 위치기반으로 푼다.
    반환 index 는 flip 추론의 belief index (물리 코너 의미 아님)."""
    H, W = img.shape[:2]
    proc = pad_frame(cv2.flip(img, 1), args.pad, args.pad_mode)
    belief = infer(model, proc, device)
    peaks = extract_keypoints(belief, args.threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    flip_px = [None] * 9
    for i, kp in enumerate(peaks):
        if kp is None:
            continue
        ox, oy = belief_peak_to_orig(kp, (bh, bw), (H, W), args.pad)
        flip_px[i] = (W - 1 - ox, oy)  # flip 이미지 좌표 → 원본 방향으로 x 되돌림
    return flip_px


def avg_consistent(a, b, max_px=FLIP_MATCH_MAX_PX):
    """plain(a) 과 flip-back(b) 을 위치기반 Hungarian 으로 매칭 후 평균.

    a 의 index = 물리 코너(plain belief). b 의 index 는 flip belief 라 코너
    의미가 다를 수 있으므로(yaw!=0), 라벨이 아닌 2D 위치로 둘을 짝지운다.
    매칭 거리가 max_px 이하인 쌍만 평균(=좌우 flip 일관 코너), 초과 쌍/미검출은
    제외(None). 출력 index = plain(a) 의 물리 코너 index.

    frontal 포즈에서는 Hungarian 이 자동으로 옛 대칭 swap 과 동일한 순열을
    복원하므로 회귀 없음(진단 확인)."""
    from scipy.optimize import linear_sum_assignment
    out = [None] * 9
    ai = [i for i in range(9) if a[i] is not None]
    bi = [i for i in range(9) if b[i] is not None]
    if not ai or not bi:
        return out
    cost = np.zeros((len(ai), len(bi)))
    for r, ia in enumerate(ai):
        pa = np.asarray(a[ia], float)
        for c, ib in enumerate(bi):
            cost[r, c] = np.linalg.norm(pa - np.asarray(b[ib], float))
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] > max_px:
            continue
        ia, ib = ai[r], bi[c]
        out[ia] = ((a[ia][0] + b[ib][0]) / 2.0,
                   (a[ia][1] + b[ib][1]) / 2.0)
    return out


def draw_kps(vis, kps, label):
    """keypoint + index 그리기, 좌상단 라벨. 반환 detected 수."""
    for i, kp in enumerate(kps):
        if kp is None:
            continue
        pt = (int(round(kp[0])), int(round(kp[1])))
        cv2.circle(vis, pt, 6, KP_COLORS[i], -1)
        cv2.circle(vis, pt, 7, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (pt[0] + 8, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLORS[i], 1)
    detected = sum(1 for kp in kps if kp is not None)
    info = f"{label} | {detected}/9 kps"
    cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if detected >= 4 else (255, 255, 255), 1, cv2.LINE_AA)
    return detected


def run_flip_compare(model, device, args):
    """이미지 폴더 좌우 비교 이미지 저장.

    기본: [plain | flip-TTA 평균(AND)].
    --refine: [plain | shape-prior refine(K-free)]. refine 은 plain keypoint 를
    flat-cuboid 형상 prior(WIDTH/DEPTH 소실점 교점)로 보정한다. 게이트에서
    탈락하면(망가진 검출) plain 을 그대로 유지하고 'REFINE: gate=...' 표기."""
    import glob as _glob
    paths = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(_glob.glob(os.path.join(args.images, e)))
    paths = sorted(paths)
    if not paths:
        sys.exit(f"이미지를 찾지 못함: {args.images}")
    os.makedirs(args.out, exist_ok=True)
    mode = "refine" if args.refine else "flip-avg(AND)"
    print(f"[flip-compare:{mode}] {len(paths)}장 → {args.out}")
    t0 = time.time()
    nA = nB = 0
    for f, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        a = infer_kps_orig(model, img, device, args)
        if args.refine:
            H, W = img.shape[:2]
            res = refine_keypoints(a, img_size=(W, H))
            m = res.kps
            rlabel = (f"refine conf={res.confidence:.2f} "
                      f"maxmv={res.info.get('max_move', 0):.0f}px"
                      if res.applied else f"REFINE GATE: {res.info.get('gate')}")
        else:
            b = infer_kps_flip(model, img, device, args)
            m = avg_consistent(a, b)
            rlabel = "flip-avg (AND)"
        visA = img.copy(); dA = draw_kps(visA, a, "plain")
        visB = img.copy(); dB = draw_kps(visB, m, rlabel)
        if dA >= 4:
            nA += 1
        if dB >= 4:
            nB += 1
        combo = cv2.hconcat([visA, visB])
        cv2.imwrite(os.path.join(args.out, os.path.basename(p)), combo,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if f % 50 == 0:
            print(f"  {f}/{len(paths)}  plain={dA}/9  flip-avg={dB}/9")
    el = time.time() - t0
    print(f"\n[done] {len(paths)}장 in {el:.1f}s")
    print(f"[done] plain    det(>=4): {nA}/{len(paths)} ({100*nA/len(paths):.1f}%)")
    print(f"[done] flip-avg det(>=4): {nB}/{len(paths)} ({100*nB/len(paths):.1f}%)")
    print(f"[done] saved: {args.out}")


def pad_frame(img, pad, mode):
    """Reflect-pad then resize back to original size (mirrors training pad).

    Returns padded-resized image (same HxW as input)."""
    if pad <= 0:
        return img
    border = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "black": cv2.BORDER_CONSTANT,
        "white": cv2.BORDER_CONSTANT,
    }.get(mode, cv2.BORDER_REFLECT_101)
    val = (255, 255, 255) if mode == "white" else (0, 0, 0)
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=val)
    return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)


def belief_peak_to_orig(kp, belief_hw, frame_hw, pad):
    """Map a belief-space peak (bx, by) back to ORIGINAL (unpadded) frame coords.

    belief peak -> padded-resized 640x480 canvas -> padded canvas (w+2P) ->
    subtract pad => original coords.

      bx in [0, bw) ; canvas coord  cx = bx * W / bw   (W = frame width)
      undo resize-back: px = cx * (W + 2P) / W
      undo pad:         ox = px - P
    (the W/bw and *(W+2P)/W collapse, but kept explicit for clarity)
    """
    bh, bw = belief_hw
    H, W = frame_hw
    cx = kp[0] * (W / bw)
    cy = kp[1] * (H / bh)
    if pad > 0:
        px = cx * (W + 2 * pad) / W
        py = cy * (H + 2 * pad) / H
        ox = px - pad
        oy = py - pad
    else:
        ox, oy = cx, cy
    return ox, oy


def run_images(model, K, device, args):
    """이미지 폴더 추론 → overlay jpg 를 args.out 폴더에 같은 파일명으로 저장."""
    import glob as _glob
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for e in exts:
        paths.extend(_glob.glob(os.path.join(args.images, e)))
    paths = sorted(paths)
    if not paths:
        sys.exit(f"이미지를 찾지 못함: {args.images}")
    os.makedirs(args.out, exist_ok=True)
    print(f"[images] {len(paths)}장 → {args.out}")
    t0 = time.time()
    n_det = n_pnp = 0
    for f, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        vis, detected, pnp_ok = process_frame(model, img, K, device, args)
        if detected >= 4:
            n_det += 1
        if pnp_ok:
            n_pnp += 1
        info = f"{args.label} | {detected}/9 kps | pad={args.pad}"
        cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if detected >= 4 else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, os.path.basename(p)), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if f % 50 == 0:
            print(f"  {f}/{len(paths)}  det={detected}/9")
    el = time.time() - t0
    print(f"\n[done] {len(paths)}장 in {el:.1f}s")
    print(f"[done] det (>=4 kps): {n_det}/{len(paths)} ({100*n_det/len(paths):.1f}%)")
    if K is not None:
        print(f"[done] PnP success : {n_pnp}/{len(paths)} ({100*n_pnp/len(paths):.1f}%)")
    print(f"[done] saved: {args.out}")


def process_frame(model, img, K, device, args):
    """단일 프레임 추론 + overlay 그리기.

    K=None 이면 PnP/cuboid 생략 (카메라 intrinsics 모를 때 keypoint 만).
    Returns (vis, detected, pnp_ok)."""
    H, W = img.shape[:2]
    proc = pad_frame(img, args.pad, args.pad_mode)
    belief = infer(model, proc, device)
    peaks = extract_keypoints(belief, args.threshold)
    bh, bw = belief.shape[1], belief.shape[2]

    kps_orig = []
    for kp in peaks:
        if kp is None:
            kps_orig.append(None)
        else:
            kps_orig.append(belief_peak_to_orig(kp, (bh, bw), (H, W), args.pad))

    detected = sum(1 for kp in kps_orig if kp is not None)

    vis = img.copy()
    for i, kp in enumerate(kps_orig):
        if kp is None:
            continue
        pt = (int(round(kp[0])), int(round(kp[1])))
        cv2.circle(vis, pt, 6, KP_COLORS[i], -1)
        cv2.circle(vis, pt, 7, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (pt[0] + 8, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLORS[i], 1)

    pose = None
    if K is not None and detected >= 4:
        kps_2d = [list(kp) if kp is not None else None for kp in kps_orig]
        try:
            pose = solve_pose(kps_2d, K, dims=PALLET_DIMS, img_shape=(H, W))
        except Exception:
            pose = None
    pnp_ok = pose is not None
    if pnp_ok:
        R, t = pose["R"], pose["t"]
        kp3d = make_pallet_keypoints_3d(*pose["dims"])
        reproj = project_3d(kp3d, R, t, K)
        for i0, i1 in CUBOID_EDGES:
            p0 = reproj[i0]; p1 = reproj[i1]
            if p0[0] == -1.0 or p1[0] == -1.0:
                continue
            cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(p1[0])), int(round(p1[1]))),
                     CUBOID_YELLOW, 2)
    return vis, detected, pnp_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mp4", help="mp4 입력 (이 모드는 영상 출력)")
    ap.add_argument("--images", help="이미지 폴더 입력 (overlay jpg 를 --out 폴더에 저장)")
    ap.add_argument("--flip-compare", action="store_true",
                    help="images 모드: [plain | flip-TTA 평균(둘 다 검출된 점만)] 좌우 비교 저장")
    ap.add_argument("--refine", action="store_true",
                    help="flip-compare 모드 변형: 오른쪽을 flip-avg 대신 shape-prior "
                         "refine(K-free, cuboid_kp_refine) 결과로 (left=plain | right=refine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=int, default=100,
                    help="reflect pad px each side (YOLO parity = 100, 0 = off)")
    ap.add_argument("--pad-mode", default="reflect",
                    choices=["reflect", "replicate", "black"])
    ap.add_argument("--threshold", type=float, default=0.3, help="belief peak thr")
    ap.add_argument("--cam_k", default=None, help="3x3 cam_K.txt (overrides fx..cy)")
    ap.add_argument("--fx", type=float, default=614.18)
    ap.add_argument("--fy", type=float, default=614.31)
    ap.add_argument("--cx", type=float, default=329.28)
    ap.add_argument("--cy", type=float, default=234.53)
    ap.add_argument("--label", default="DOPE+PAD")
    args = ap.parse_args()

    device = __import__("torch").device(
        "cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[weights] {args.weights}")
    print(f"[pad] {args.pad}px ({args.pad_mode})")
    if (args.mp4 is None) == (args.images is None):
        sys.exit("--mp4 와 --images 중 정확히 하나를 지정하세요")
    model = load_model(args.weights, device)

    # K: cam_k 파일이 있으면 사용. mp4 모드는 기본 intrinsics(RealSense) 사용,
    # images 모드는 카메라가 미지일 수 있어 cam_k 없으면 K=None (PnP 생략).
    if args.cam_k and os.path.exists(args.cam_k):
        K = np.loadtxt(args.cam_k).reshape(3, 3).astype(np.float64)
    elif args.images is not None:
        K = None
        print("[K] none (images 모드 + cam_k 미지정 → PnP/cuboid 생략, keypoint 만)")
    else:
        K = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0, 0, 1]], dtype=np.float64)
    if K is not None:
        print(f"[K]\n{K}")

    if args.images is not None:
        if args.flip_compare:
            run_flip_compare(model, device, args)
        else:
            run_images(model, K, device, args)
        return

    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        print(f"failed to open: {args.mp4}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[mp4] {W}x{H} @ {fps:.1f}fps, {n} frames")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    t0 = time.time()
    f = n_det = n_pnp = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        proc = pad_frame(img, args.pad, args.pad_mode)
        belief = infer(model, proc, device)               # (9, bh, bw)
        peaks = extract_keypoints(belief, args.threshold)  # belief-space
        bh, bw = belief.shape[1], belief.shape[2]

        # map peaks -> ORIGINAL frame coords (None preserved)
        kps_orig = []
        for kp in peaks:
            if kp is None:
                kps_orig.append(None)
            else:
                kps_orig.append(belief_peak_to_orig(kp, (bh, bw), (H, W), args.pad))

        detected = sum(1 for kp in kps_orig if kp is not None)
        if detected >= 4:
            n_det += 1

        vis = img.copy()
        # draw keypoints (original coords)
        for i, kp in enumerate(kps_orig):
            if kp is None:
                continue
            pt = (int(round(kp[0])), int(round(kp[1])))
            cv2.circle(vis, pt, 6, KP_COLORS[i], -1)
            cv2.circle(vis, pt, 7, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLORS[i], 1)

        # order-free PnP (solve_pose: 24-sym, ITERATIVE refine, auto-dim, v4)
        kps_2d = [list(kp) if kp is not None else None for kp in kps_orig]
        pose = None
        if detected >= 4:
            try:
                pose = solve_pose(kps_2d, K, dims=PALLET_DIMS,
                                  img_shape=(H, W))
            except Exception as e:
                pose = None
        pnp_ok = pose is not None
        if pnp_ok:
            n_pnp += 1
            R, t = pose["R"], pose["t"]
            kp3d = make_pallet_keypoints_3d(*pose["dims"])
            reproj = project_3d(kp3d, R, t, K)
            for i0, i1 in CUBOID_EDGES:
                p0 = reproj[i0]; p1 = reproj[i1]
                if p0[0] == -1.0 or p1[0] == -1.0:
                    continue
                cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))),
                         (int(round(p1[0])), int(round(p1[1]))),
                         CUBOID_YELLOW, 2)

        z = pose["t"][2] if pnp_ok else 0.0
        info = (f"{args.label} f{f} | {detected}/9 kps | "
                f"PnP {'OK z=%.2fm' % z if pnp_ok else 'FAIL'} | pad={args.pad}")
        cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if pnp_ok else (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(vis)

        if f % 50 == 0:
            el = time.time() - t0
            fc = (f + 1) / max(1e-3, el)
            print(f"  frame {f:4d}/{n}  det={detected}/9  pnp={'Y' if pnp_ok else 'n'}"
                  f"  ({fc:.1f} FPS, ETA {(n-f-1)/max(1e-3,fc):.0f}s)")
        f += 1

    cap.release()
    writer.release()
    el = time.time() - t0
    print()
    print(f"[done] {f} frames in {el:.1f}s ({f/el:.1f} FPS)")
    print(f"[done] det (>=4 kps) : {n_det}/{f} ({100*n_det/max(1,f):.1f}%)")
    print(f"[done] PnP success   : {n_pnp}/{f} ({100*n_pnp/max(1,f):.1f}%) "
          f"(of det {100*n_pnp/max(1,n_det):.1f}%)")
    print(f"[done] saved: {args.out}")


if __name__ == "__main__":
    main()
