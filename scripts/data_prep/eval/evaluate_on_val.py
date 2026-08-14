"""Phase 5: 합성 검증셋에서 DOPE 추론 + 6D pose 종합 평가.

ORDER-FREE 평가 (2026-06-03 수정):
  과거 버전은 예측 keypoint index i 를 GT projected_cuboid index i 와 그대로
  비교했는데, 모델 학습 convention (camera-facing v4) 과 합성 GT 의 corner 순서
  (Isaac annotate 순서) 가 달라 same-index 비교가 항상 깨졌다 (reproj 130px+,
  PCK 과소평가). corner *위치* 는 맞고 *label/순서* 만 다른 문제 (same-idx 118px
  vs best-match 17px). 본 버전은 cuboid 의 48 corner automorphism (topology 보존
  relabeling) 중 GT 2D 와 가장 잘 맞는 permutation 을 골라 정합한 뒤 메트릭을
  계산한다. centroid(idx 8) 는 고정. 자유 Hungarian 이 아니라 cube 위상을
  보존하는 permutation 만 허용하므로 메트릭 의미가 왜곡되지 않는다.

  cuboid dims = 1.1 x 1.3 x 0.11 (GT self-consistency 최선값, PnP reproj 5px).

메트릭:
  - PCK@3px, PCK@5px, PCK@10px (order-free, belief map 해상도)
  - PnP success rate + 2D Reproj error (order-free)
  - 2D corner error (best-automorphism, PnP 무관 순수 keypoint 품질 지표)
  - 3D Volume Ratio (predicted cuboid volume / GT volume)
  - same-index reproj/PCK 도 reference 로 함께 출력 (convention 진단용)

사용법:
    python scripts/data_prep/eval/evaluate_on_val.py \
        --weights weights/pallet_category/final_net_epoch_0060.pth \
        --val_dir data/pallet/training_data/val \
        --output_dir data/pallet/eval_results
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "challenge", "scripts"))

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "..", "..", "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
try:
    from models import DopeNetwork
except ImportError:
    print("[ERROR] Cannot import DopeNetwork. Check Deep_Object_Pose path.")
    sys.exit(1)

from pnp_solver import (
    PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d,
    make_pallet_keypoints_3d_isaac,
)
from metrics import compute_reproj_error
import annotate_pnp as _apnp  # order-free PnP (auto-dim 110/130 + 24-sym + strict invariants)


# ── Order-free corner matching (2026-06-03) ──────────────────────────────────
# Cuboid 의 48 corner automorphism (graph automorphism of the cube = topology
# 보존 relabeling). 모델 출력 keypoint 순서가 GT projected_cuboid (Isaac) 순서와
# 다른 convention mismatch 를 흡수한다. 자유 point assignment (Hungarian) 이
# 아니라 cube 위상을 보존하는 48 permutation 만 허용 → 메트릭 의미 보존.
# (생성 근거: pnp_solver Isaac edge topology 의 graph automorphism, 별도 검증.)
CUBE_AUTOMORPHISMS = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 5, 4, 3, 2, 6, 7], [0, 3, 2, 1, 4, 7, 6, 5],
    [0, 3, 7, 4, 1, 2, 6, 5], [0, 4, 5, 1, 3, 7, 6, 2], [0, 4, 7, 3, 1, 5, 6, 2],
    [1, 0, 3, 2, 5, 4, 7, 6], [1, 0, 4, 5, 2, 3, 7, 6], [1, 2, 3, 0, 5, 6, 7, 4],
    [1, 2, 6, 5, 0, 3, 7, 4], [1, 5, 4, 0, 2, 6, 7, 3], [1, 5, 6, 2, 0, 4, 7, 3],
    [2, 1, 0, 3, 6, 5, 4, 7], [2, 1, 5, 6, 3, 0, 4, 7], [2, 3, 0, 1, 6, 7, 4, 5],
    [2, 3, 7, 6, 1, 0, 4, 5], [2, 6, 5, 1, 3, 7, 4, 0], [2, 6, 7, 3, 1, 5, 4, 0],
    [3, 0, 1, 2, 7, 4, 5, 6], [3, 0, 4, 7, 2, 1, 5, 6], [3, 2, 1, 0, 7, 6, 5, 4],
    [3, 2, 6, 7, 0, 1, 5, 4], [3, 7, 4, 0, 2, 6, 5, 1], [3, 7, 6, 2, 0, 4, 5, 1],
    [4, 0, 1, 5, 7, 3, 2, 6], [4, 0, 3, 7, 5, 1, 2, 6], [4, 5, 1, 0, 7, 6, 2, 3],
    [4, 5, 6, 7, 0, 1, 2, 3], [4, 7, 3, 0, 5, 6, 2, 1], [4, 7, 6, 5, 0, 3, 2, 1],
    [5, 1, 0, 4, 6, 2, 3, 7], [5, 1, 2, 6, 4, 0, 3, 7], [5, 4, 0, 1, 6, 7, 3, 2],
    [5, 4, 7, 6, 1, 0, 3, 2], [5, 6, 2, 1, 4, 7, 3, 0], [5, 6, 7, 4, 1, 2, 3, 0],
    [6, 2, 1, 5, 7, 3, 0, 4], [6, 2, 3, 7, 5, 1, 0, 4], [6, 5, 1, 2, 7, 4, 0, 3],
    [6, 5, 4, 7, 2, 1, 0, 3], [6, 7, 3, 2, 5, 4, 0, 1], [6, 7, 4, 5, 2, 3, 0, 1],
    [7, 3, 0, 4, 6, 2, 1, 5], [7, 3, 2, 6, 4, 0, 1, 5], [7, 4, 0, 3, 6, 5, 1, 2],
    [7, 4, 5, 6, 3, 0, 1, 2], [7, 6, 2, 3, 4, 5, 1, 0], [7, 6, 5, 4, 3, 2, 1, 0],
], dtype=np.int64)

CUBOID_DIMS = (1.1, 1.3, 0.11)  # (width, depth, height) — self-consistency 최선값


def best_corner_permutation(pred8, gt8, valid8):
    """48 cube automorphism 중 GT corner 와 가장 잘 맞는 permutation 선택.

    Args:
        pred8:  (8, 2) 예측 corner 픽셀좌표 (모델 출력 순서).
        gt8:    (8, 2) GT corner 픽셀좌표 (Isaac 순서).
        valid8: (8,) bool, 예측 corner 검출 여부.

    Returns:
        perm (8,) int — perm[k] 가 GT index k 에 대응하는 *예측* index, 또는 None.
        valid 가 4 개 미만이면 None.
    """
    if valid8.sum() < 4:
        return None
    best_perm = None
    best_err = np.inf
    for am in CUBE_AUTOMORPHISMS:
        vm = valid8[am]
        if vm.sum() < 4:
            continue
        err = np.linalg.norm(pred8[am][vm] - gt8[vm], axis=1).mean()
        if err < best_err:
            best_err = err
            best_perm = am
    return best_perm


# Isaac corner ordering (perm-aligned pred 와 일관) 의 cuboid edge groups.
# 0=(-w,-h,+d) 1=(+w,..) 3=(..,+h,..) 4=(..,-d): 0->1=W, 0->3=H, 0->4=D.
_ISAAC_W_EDGES = [(0, 1), (3, 2), (4, 5), (7, 6)]
_ISAAC_H_EDGES = [(0, 3), (1, 2), (4, 7), (5, 6)]
_ISAAC_D_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]


def solve_pnp_iterative(pred_re, valid_re, kp3d, K):
    """ITERATIVE PnP (flat-pallet 안전). pred_re 는 이미 Isaac 순서로 정합됨.

    EPnP/RANSAC 는 pallet 처럼 거의 평면(height 0.11m << 1.1/1.3) 인 물체에서
    심하게 발산한다 (GT self-consistency 43px). SOLVEPNP_ITERATIVE + IPPE seed 는
    동일 GT 에서 4.9px. 따라서 eval reproj 는 ITERATIVE 를 사용.

    Returns (R, t) or None.
    """
    idx = [i for i in range(9) if valid_re[i]]
    if len(idx) < 4:
        return None
    obj = kp3d[idx].astype(np.float64)
    img = np.array([pred_re[i] for i in idx], dtype=np.float64)
    # IPPE planar seed (top face 4점) → ITERATIVE refine. 실패 시 EPNP seed.
    R0 = t0 = None
    for seed_flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=seed_flag)
            if ok and tvec[2, 0] > 0:
                R0, t0 = rvec, tvec
                break
        except cv2.error:
            continue
    try:
        if R0 is not None:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, rvec=R0.copy(), tvec=t0.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    if t[2] <= 0:
        return None
    return R, t


def reproject_all(kp3d, R, t, K):
    """9 keypoint 전체 재투영 (9, 2)."""
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(kp3d, rvec, t.reshape(3, 1), K, None)
    return proj.reshape(-1, 2)


def best_match_corner_error(proj8, gt8, valid8):
    """proj8 (8,2) reprojection 을 GT corner 와 best-48-automorphism 으로 정합한
    뒤 평균 corner 거리. valid8 (8,) bool 은 GT corner 유효성."""
    best = np.inf
    for am in CUBE_AUTOMORPHISMS:
        d = np.linalg.norm(proj8[am][valid8] - gt8[valid8], axis=1).mean()
        if d < best:
            best = d
    return best


def load_model(weights_path, device):
    """학습된 DOPE 모델 로드."""
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_keypoints_from_belief(belief_maps, threshold=0.5, win=11):
    """belief map에서 keypoint 좌표 추출 (DOPE 공식 sub-pixel 방식).

    1. Gaussian filter로 노이즈 억제
    2. Non-maximum suppression으로 peak 탐색
    3. win x win 윈도우에서 weighted average로 sub-pixel 정밀도 확보 (기본 11)
    """
    OFFSET = 0.4395
    WIN = win
    RAN = WIN // 2
    keypoints = []

    for i in range(belief_maps.shape[0]):
        bmap_ori = belief_maps[i]
        max_val = bmap_ori.max()
        if max_val < threshold:
            keypoints.append((-1, -1, float(max_val)))
            continue

        bmap_smooth = gaussian_filter(bmap_ori, sigma=2)
        p = 1
        pad_l = np.zeros_like(bmap_smooth); pad_l[p:, :] = bmap_smooth[:-p, :]
        pad_r = np.zeros_like(bmap_smooth); pad_r[:-p, :] = bmap_smooth[p:, :]
        pad_u = np.zeros_like(bmap_smooth); pad_u[:, p:] = bmap_smooth[:, :-p]
        pad_d = np.zeros_like(bmap_smooth); pad_d[:, :-p] = bmap_smooth[:, p:]

        peaks_binary = (
            (bmap_smooth >= pad_l) & (bmap_smooth >= pad_r) &
            (bmap_smooth >= pad_u) & (bmap_smooth >= pad_d) &
            (bmap_smooth > threshold)
        )

        peak_ys, peak_xs = np.nonzero(peaks_binary)
        if len(peak_xs) == 0:
            keypoints.append((-1, -1, float(max_val)))
            continue

        peak_vals = [bmap_ori[py, px] for py, px in zip(peak_ys, peak_xs)]
        best_idx = np.argmax(peak_vals)
        px, py = int(peak_xs[best_idx]), int(peak_ys[best_idx])

        y_lo = max(0, py - RAN); y_hi = min(bmap_ori.shape[0], py + RAN + 1)
        x_lo = max(0, px - RAN); x_hi = min(bmap_ori.shape[1], px + RAN + 1)
        patch = bmap_ori[y_lo:y_hi, x_lo:x_hi]

        if patch.sum() > 0:
            ys = np.arange(y_lo, y_hi)
            xs = np.arange(x_lo, x_hi)
            xg, yg = np.meshgrid(xs, ys)
            wx = np.average(xg, weights=patch) + OFFSET
            wy = np.average(yg, weights=patch) + OFFSET
        else:
            wx, wy = float(px), float(py)

        keypoints.append((wx, wy, float(max_val)))
    return keypoints


def compute_pck(pred_kps, gt_kps, threshold_px=10):
    """PCK (Percentage of Correct Keypoints). Sub-pixel 좌표 지원."""
    correct = 0
    total = 0
    for kp, (gx, gy) in zip(pred_kps, gt_kps):
        px, py = kp[0], kp[1]
        if px < 0 or gx < 0:
            continue
        total += 1
        dist = np.sqrt((float(px) - float(gx)) ** 2 + (float(py) - float(gy)) ** 2)
        if dist <= threshold_px:
            correct += 1
    return correct, total


def compute_volume_from_keypoints(pred_kps_orig, R, t, camera_matrix):
    """예측 2D keypoint를 PnP depth로 back-project하여 3D 부피 계산.

    PnP는 고정 3D 모델을 사용하므로, 예측 keypoint의 실제 3D 위치를
    back-projection으로 복원하여 부피를 구해야 의미 있는 비교가 됨.

    Returns:
        volume: 추정된 부피 (m³), 또는 None (keypoint 부족 시)
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # 고정 3D 모델 keypoints의 PnP 복원 depth (per-corner)
    # Isaac corner ordering (perm-aligned pred 와 일관) + self-consistency dims.
    kp3d = make_pallet_keypoints_3d_isaac(*CUBOID_DIMS)
    corners_3d_cam = (R @ kp3d[:8].T).T + t  # (8, 3)

    # 예측 2D keypoint를 각 corner의 depth로 back-project
    bp = []
    for i in range(8):
        pt = pred_kps_orig[i] if i < len(pred_kps_orig) else None
        if pt is None:
            return None
        u, v = float(pt[0]), float(pt[1])
        if u < 0 or v < 0:
            return None
        Z = corners_3d_cam[i, 2]
        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        bp.append([X, Y, Z])

    bp = np.array(bp)

    # Cuboid edge lengths: 0→1 (width), 0→3 (height), 0→4 (depth)
    edge_w = np.linalg.norm(bp[1] - bp[0])
    edge_h = np.linalg.norm(bp[3] - bp[0])
    edge_d = np.linalg.norm(bp[4] - bp[0])

    return edge_w * edge_h * edge_d


GT_VOLUME = CUBOID_DIMS[0] * CUBOID_DIMS[1] * CUBOID_DIMS[2]  # 1.1*1.3*0.11


def main():
    parser = argparse.ArgumentParser(description="DOPE 6D Pose 종합 평가")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output_dir", default="data/pallet/eval_results")
    parser.add_argument("--preprocess", choices=["squash", "letterbox", "pad", "aspect"], default="aspect",
                        help="squash=resize(400,400) 비율왜곡(기존) / aspect=실배포(run_dope_live) 비율유지 height→400 width비례 / letterbox=비율보존+회색pad / pad=reflect-pad(truncation, dope_predict_mp4_pad)")
    parser.add_argument("--pad", type=int, default=100,
                        help="pad 모드 reflect-padding 픽셀 (each side)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Belief map peak threshold")
    parser.add_argument("--peak_win", type=int, default=11,
                        help="belief peak sub-pixel weighted-centroid 윈도우 크기 (홀수, 기본 11)")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.weights}")
    model = load_model(args.weights, device)

    # PnP setup — Isaac corner ordering + self-consistency dims.
    # order-free 정합 후 pred 는 Isaac 순서로 재배열되므로 isaac kp3d 사용.
    # EPnP/RANSAC 는 flat pallet 에서 발산 → solve_pnp_iterative (ITERATIVE) 사용.
    cam_matrix = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    kp3d_isaac = make_pallet_keypoints_3d_isaac(*CUBOID_DIMS)

    png_files = sorted(glob.glob(os.path.join(args.val_dir, "*.png")))[:args.max_frames]
    print(f"Evaluating {len(png_files)} frames...")

    # order-free (primary) counters
    pck_counters = {3: [0, 0], 5: [0, 0], 10: [0, 0]}
    pnp_success_count = 0
    pnp_total = 0
    reproj_errors = []        # PnP reproj (order-free)
    corner2d_errors = []      # best-automorphism 2D corner error (PnP 무관)
    volume_ratios = []
    # same-index reference counters (convention 진단용)
    pck_same = {3: [0, 0], 5: [0, 0], 10: [0, 0]}
    reproj_same = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, png_path in enumerate(png_files):
        basename = os.path.splitext(os.path.basename(png_path))[0]
        json_path = os.path.join(args.val_dir, basename + ".json")
        if not os.path.exists(json_path):
            continue

        # 이미지 로드 + 전처리 (squash | letterbox)
        img = cv2.imread(png_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        _mode = args.preprocess
        _P = args.pad
        if _mode == "letterbox":
            _s = min(400.0 / w_orig, 400.0 / h_orig)
            _nw, _nh = int(round(w_orig * _s)), int(round(h_orig * _s))
            _px, _py = (400 - _nw) // 2, (400 - _nh) // 2
            img_resized = np.zeros((400, 400, 3), dtype=img_rgb.dtype)
            img_resized[_py:_py + _nh, _px:_px + _nw] = cv2.resize(img_rgb, (_nw, _nh))
        elif _mode == "pad":  # reflect-pad P each side → resize(400) (truncation 정합)
            padded = cv2.copyMakeBorder(img_rgb, _P, _P, _P, _P, cv2.BORDER_REFLECT)
            img_resized = cv2.resize(padded, (400, 400))
        elif _mode == "aspect":  # 비율유지 short-side→400 (가로/세로 모두 객체크기 유지), 비정사각
            _sc = 400.0 / min(h_orig, w_orig)
            _nw = max(8, int(round(w_orig * _sc)) & ~7)
            _nh = max(8, int(round(h_orig * _sc)) & ~7)
            img_resized = cv2.resize(img_rgb, (_nw, _nh))
        else:  # squash (학습 RandomCrop 비율과 불일치 — 기존 호환/비교용)
            img_resized = cv2.resize(img_rgb, (400, 400))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out_bel, out_aff = model(tensor)

        belief = out_bel[-1][0].cpu().numpy()
        pred_kps = extract_keypoints_from_belief(belief, args.threshold, win=args.peak_win)

        # GT 로드
        with open(json_path) as f:
            data = json.load(f)
        obj = data["objects"][0]
        gt_cuboid = obj["projected_cuboid"]
        gt_centroid = obj["projected_cuboid_centroid"]

        bh, bw = belief.shape[1], belief.shape[2]

        def _belief_to_orig(bx, by):
            cx, cy = bx * 400.0 / bw, by * 400.0 / bh   # belief → 400 입력 캔버스
            if _mode == "letterbox":
                return (cx - _px) / _s, (cy - _py) / _s
            if _mode == "pad":  # 400→padded(W+2P,H+2P)→ pad 제거
                return cx * (w_orig + 2 * _P) / 400.0 - _P, cy * (h_orig + 2 * _P) / 400.0 - _P
            return cx * w_orig / 400.0, cy * h_orig / 400.0  # squash

        def _orig_to_belief(gx, gy):
            if _mode == "letterbox":
                cx, cy = gx * _s + _px, gy * _s + _py
            elif _mode == "pad":
                cx = (gx + _P) * 400.0 / (w_orig + 2 * _P)
                cy = (gy + _P) * 400.0 / (h_orig + 2 * _P)
            else:
                cx, cy = gx * 400.0 / w_orig, gy * 400.0 / h_orig
            return cx * bw / 400.0, cy * bh / 400.0

        # orig→belief 등방 스케일 (order-free PCK 거리용)
        if _mode == "letterbox":
            sx = sy = _s * bw / 400.0
        elif _mode == "pad":
            sx, sy = bw / (w_orig + 2 * _P), bh / (h_orig + 2 * _P)
        else:
            sx, sy = bw / w_orig, bh / h_orig

        # 예측 keypoint (orig 해상도) + validity (9점)
        pred9 = np.full((9, 2), -1.0, dtype=np.float64)
        valid9 = np.zeros(9, dtype=bool)
        for k in range(9):
            if pred_kps[k][0] >= 0:
                pred9[k] = list(_belief_to_orig(pred_kps[k][0], pred_kps[k][1]))
                valid9[k] = True
        gt9 = np.array(gt_cuboid + [gt_centroid], dtype=np.float64)  # orig 해상도

        pnp_total += 1

        # === same-index reference (convention 진단용) ===
        gt_scaled_same = [_orig_to_belief(gx, gy) for gx, gy in gt_cuboid]
        gt_scaled_same.append(_orig_to_belief(gt_centroid[0], gt_centroid[1]))
        for thr in pck_same:
            c, t = compute_pck(pred_kps, gt_scaled_same, threshold_px=thr)
            pck_same[thr][0] += c
            pck_same[thr][1] += t

        # === ORDER-FREE 정합 (48 cube automorphism, centroid 고정) ===
        perm = best_corner_permutation(pred9[:8], gt9[:8], valid9[:8])
        if perm is None:
            continue  # corner 4개 미만 — PnP/PCK 불가
        pred_re = pred9.copy()
        pred_re[:8] = pred9[:8][perm]
        valid_re = valid9.copy()
        valid_re[:8] = valid9[:8][perm]

        # PCK (order-free) — belief 해상도 거리. centroid(8) 포함.
        for k in range(9):
            if not valid_re[k]:
                continue
            dist_belief = np.hypot(
                (pred_re[k, 0] - gt9[k, 0]) * sx,
                (pred_re[k, 1] - gt9[k, 1]) * sy,
            )
            for thr in pck_counters:
                pck_counters[thr][1] += 1
                if dist_belief <= thr:
                    pck_counters[thr][0] += 1

        # 2D corner error (best-automorphism, orig 해상도) — PnP 무관 keypoint 품질
        c2d = np.linalg.norm(pred_re[valid_re] - gt9[valid_re], axis=1).mean()
        corner2d_errors.append(c2d)

        # === PnP (order-free) → reproj vs GT 2D ===
        # annotate_pnp.solve_pose 는 raw 예측 순서를 입력으로 받아 auto-dim
        # (110/130 정면) + 24 cube symmetry + strict invariant 로 *기하학적으로*
        # 올바른 pose 를 찾는다 (입력 label 순서 무관). 이후 reprojection 을
        # GT 2D 와 best-48-automorphism 으로 정합해 reproj error 측정.
        pred_raw = [tuple(pred9[k]) if valid9[k] else None for k in range(9)]
        pose = _apnp.solve_pose(pred_raw, cam_matrix)
        if pose is not None:
            pnp_success_count += 1
            proj_all = np.array(pose["projected_all"], dtype=np.float64)  # (9,2)
            reproj_err = best_match_corner_error(
                proj_all[:8], gt9[:8], np.ones(8, dtype=bool))
            reproj_errors.append(reproj_err)

            # 부피: 예측 2D keypoint(perm-aligned, Isaac 순서)를 solve_pose 의
            # per-corner depth 로 back-project 하여 복원된 cuboid 부피 추정.
            R_pred = pose["R"]
            t_pred = pose["t"]
            pred_re_list = [tuple(pred_re[k]) if valid_re[k] else None
                            for k in range(9)]
            pred_vol = compute_volume_from_keypoints(
                pred_re_list, R_pred, t_pred, cam_matrix)
            if pred_vol is not None and pred_vol > 0:
                volume_ratios.append(pred_vol / GT_VOLUME)

        if (i + 1) % 50 == 0:
            pck3 = pck_counters[3][0] / max(pck_counters[3][1], 1)
            print(f"  [{i+1}/{len(png_files)}] PCK@3px(of): {pck3:.3f}, "
                  f"PnP: {pnp_success_count}/{pnp_total}")

    # ========== 최종 결과 ==========
    print(f"\n{'='*60}")
    print(f" DOPE Evaluation Results ({len(png_files)} frames)")
    print(f"{'='*60}")

    # PCK (order-free)
    print("  [PCK — order-free (48 cube automorphism 정합)]")
    for thr in sorted(pck_counters):
        c, t = pck_counters[thr]
        pck = c / max(t, 1)
        print(f"  PCK@{thr}px:   {pck:.4f}  ({c}/{t})")

    # PCK same-index reference
    print("  [PCK — same-index reference (convention 진단용)]")
    for thr in sorted(pck_same):
        c, t = pck_same[thr]
        print(f"  PCK@{thr}px(si): {c / max(t, 1):.4f}  ({c}/{t})")

    # 2D corner error (PnP 무관)
    if corner2d_errors:
        c2d = np.array(corner2d_errors)
        print(f"\n  --- 2D Corner Error (order-free, {len(c2d)} frames) ---")
        print(f"  Corner err mean:    {c2d.mean():.2f} px")
        print(f"  Corner err med:     {np.median(c2d):.2f} px")
        print(f"  Corner <5px:        {np.mean(c2d < 5.0) * 100:.1f}%")
        print(f"  Corner <10px:       {np.mean(c2d < 10.0) * 100:.1f}%")

    # PnP + Reproj 메트릭
    print(f"\n  PnP success:  {pnp_success_count}/{pnp_total} ({pnp_success_count/max(pnp_total,1)*100:.1f}%)")

    if reproj_errors:
        reproj_arr = np.array(reproj_errors)
        reproj_5px = np.mean(reproj_arr < 5.0) * 100
        reproj_10px = np.mean(reproj_arr < 10.0) * 100
        print(f"\n  --- PnP Reproj Metrics (order-free, {len(reproj_errors)} frames) ---")
        print(f"  Reproj error mean:  {reproj_arr.mean():.2f} px")
        print(f"  Reproj error med:   {np.median(reproj_arr):.2f} px")
        print(f"  Reproj <5px:        {reproj_5px:.1f}%")
        print(f"  Reproj <10px:       {reproj_10px:.1f}%")
    else:
        print("  (No successful PnP for reproj metrics)")

    # 3D Volume
    if volume_ratios:
        vol_arr = np.array(volume_ratios)
        print(f"\n  --- 3D Volume Metrics ({len(volume_ratios)} frames) ---")
        print(f"  GT volume:          {GT_VOLUME:.4f} m³")
        print(f"  Pred volume mean:   {vol_arr.mean() * GT_VOLUME:.4f} m³")
        print(f"  Volume ratio mean:  {vol_arr.mean():.3f}  (1.0 = perfect)")
        print(f"  Volume ratio med:   {np.median(vol_arr):.3f}")
        print(f"  Volume ratio std:   {vol_arr.std():.3f}")
        print(f"  |ratio - 1| < 0.2: {np.mean(np.abs(vol_arr - 1.0) < 0.2) * 100:.1f}%")
        print(f"  |ratio - 1| < 0.5: {np.mean(np.abs(vol_arr - 1.0) < 0.5) * 100:.1f}%")
    else:
        print("\n  (No volume metrics available)")

    print(f"{'='*60}")

    # 결과 저장
    summary = {
        "weights": args.weights,
        "preprocess": args.preprocess,
        "val_dir": args.val_dir,
        "matching": "order-free (48 cube automorphism, centroid fixed)",
        "cuboid_dims": list(CUBOID_DIMS),
        "pck": {f"@{thr}px": pck_counters[thr][0] / max(pck_counters[thr][1], 1) for thr in pck_counters},
        "pck_same_index": {f"@{thr}px": pck_same[thr][0] / max(pck_same[thr][1], 1) for thr in pck_same},
        "corner2d_mean_px": float(np.mean(corner2d_errors)) if corner2d_errors else None,
        "corner2d_median_px": float(np.median(corner2d_errors)) if corner2d_errors else None,
        "pnp_success_rate": pnp_success_count / max(pnp_total, 1),
        "reproj_mean_px": float(np.mean(reproj_errors)) if reproj_errors else None,
        "reproj_median_px": float(np.median(reproj_errors)) if reproj_errors else None,
        "volume_ratio_mean": float(np.mean(volume_ratios)) if volume_ratios else None,
        "volume_ratio_median": float(np.median(volume_ratios)) if volume_ratios else None,
        "volume_ratio_std": float(np.std(volume_ratios)) if volume_ratios else None,
        "volume_within_20pct": float(np.mean(np.abs(np.array(volume_ratios) - 1.0) < 0.2)) if volume_ratios else None,
        "num_frames": len(png_files),
        # 검출 진단 (sigma↓ 정밀↑/검출↓ 트레이드오프 판별용): corner2d_count =
        # 유효 코너검출(≥4 corner + perm 성공) 프레임 수 = corner median 표본 수.
        "corner2d_count": len(corner2d_errors),
        "pnp_total": pnp_total,
        "detect_rate": len(corner2d_errors) / max(len(png_files), 1),
    }
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()
