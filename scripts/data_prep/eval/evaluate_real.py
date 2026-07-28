"""Real Test 평가 — AprilTag GT 기반 6D Pose 정량 평가.

Synthetic val 평가(evaluate_on_val.py)와 달리 full 6D GT가 있으므로
ADD, 5cm5°, Reproj error, Volume Ratio를 모두 계산.

사용법:
    python scripts/data_prep/eval/evaluate_real.py \
        --weights weights/mixed_v1/net_epoch_0060.pth \
        --test_dir data/pallet/real_data/real_test_seen \
        --output_dir data/pallet/eval_results/mixed_v1_real_seen

    python scripts/data_prep/eval/evaluate_real.py \
        --weights weights/mixed_v1/net_epoch_0060.pth \
        --test_dir data/pallet/real_data/real_test_unseen \
        --output_dir data/pallet/eval_results/mixed_v1_real_unseen
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

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d, make_pallet_keypoints_3d_isaac
from metrics import compute_ADD, compute_5cm_5deg, compute_reproj_error, PoseEvaluator


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_keypoints_from_belief(belief_maps, threshold=0.5):
    """belief map에서 keypoint 좌표 추출 (evaluate_on_val.py와 동일)."""
    OFFSET = 0.4395
    WIN = 11
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


def main():
    parser = argparse.ArgumentParser(description="Real Test 6D Pose 평가")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test_dir", required=True,
                        help="GT json 디렉토리 (*.json)")
    parser.add_argument("--image_dir", default=None,
                        help="이미지 디렉토리 (default: test_dir와 동일)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--cam_k", default=None,
                        help="cam_K.txt (3x3 intrinsic). 지정되면 fx/fy/cx/cy 무시")
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--visualize", action="store_true",
                        help="GT vs Pred overlay 저장")
    parser.add_argument("--pnp_mode", default="current",
                        choices=["current", "reproj_guided"],
                        help="PnP solver mode")
    parser.add_argument("--tau_huber", type=float, default=0.073)
    parser.add_argument("--tau_peak", type=float, default=0.41)
    parser.add_argument("--tau_w", type=float, default=0.27)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.test_dir, "eval_results")
    os.makedirs(args.output_dir, exist_ok=True)

    image_dir = args.image_dir or args.test_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.weights}")
    model = load_model(args.weights, device)

    if args.cam_k and os.path.exists(args.cam_k):
        K = np.loadtxt(args.cam_k)
        args.fx, args.fy = K[0, 0], K[1, 1]
        args.cx, args.cy = K[0, 2], K[1, 2]
        print(f"Loaded intrinsics from {args.cam_k}: fx={args.fx:.2f} fy={args.fy:.2f} cx={args.cx:.2f} cy={args.cy:.2f}")
    cam_matrix = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    # NOTE: 학습 시 가정한 pallet 크기 (1.1, 1.1, 0.15) 와 일치시킴
    # (config/default.yaml pallet: width 1.1, depth 1.1, height 0.15)
    kp3d = make_pallet_keypoints_3d_isaac(1.1, 1.1, 0.15)
    pnp_solver = PalletPnPSolver(cam_matrix, keypoints_3d=kp3d)
    evaluator = PoseEvaluator(kp3d[:8])  # 8 corners for ADD

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Find image-json pairs (tag-off images with GT json)
    json_files = sorted(glob.glob(os.path.join(args.test_dir, "*.json")))
    print(f"Found {len(json_files)} GT annotations in {args.test_dir}")

    pnp_total = 0
    pnp_success = 0

    for ji, json_path in enumerate(json_files):
        basename = os.path.splitext(os.path.basename(json_path))[0]

        # 이미지 찾기 (jpg 또는 png) — image_dir 우선, fallback test_dir
        img_path = None
        for ext in [".jpg", ".png"]:
            for d in [image_dir, args.test_dir]:
                candidate = os.path.join(d, basename + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path:
                break
        if img_path is None:
            continue

        # GT 로드
        with open(json_path) as f:
            gt_data = json.load(f)
        gt_obj = gt_data["objects"][0]
        T_gt = np.array(gt_obj["pose_transform"], dtype=np.float64)
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        gt_cuboid = gt_obj["projected_cuboid"]
        gt_centroid = gt_obj["projected_cuboid_centroid"]

        # 이미지 로드 + 추론
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _h0, _w0 = img_rgb.shape[:2]  # 비율유지(short-side→400): 가로/세로 모두 객체크기 유지. squash 금지
        _sc = 400.0 / min(_h0, _w0)
        _nw = max(8, int(round(_w0 * _sc)) & ~7); _nh = max(8, int(round(_h0 * _sc)) & ~7)
        img_resized = cv2.resize(img_rgb, (_nw, _nh))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out_bel, out_aff = model(tensor)

        belief = out_bel[-1][0].cpu().numpy()
        pred_kps = extract_keypoints_from_belief(belief, args.threshold)

        # Belief map → original image coords
        h_orig, w_orig = img.shape[:2]
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = bw / w_orig, bh / h_orig

        pred_kps_orig = []
        peak_confs = []
        for kp in pred_kps:
            if kp[0] < 0:
                pred_kps_orig.append(None)
                peak_confs.append(0.0)
            else:
                pred_kps_orig.append((float(kp[0]) / sx, float(kp[1]) / sy))
                peak_confs.append(float(kp[2]))

        # PnP
        pnp_total += 1
        if args.pnp_mode == "reproj_guided":
            success, R_pred, t_pred, _, _ = pnp_solver.solve_reproj_guided(
                pred_kps_orig, peak_confidences=peak_confs,
                tau_huber=args.tau_huber, tau_peak=args.tau_peak, tau_w=args.tau_w)
        else:
            success, R_pred, t_pred, _ = pnp_solver.solve(pred_kps_orig)

        if success:
            pnp_success += 1

            # GT 2D vs Pred 2D (reproj)
            reproj_pred = pnp_solver.reproject(R_pred, t_pred)
            gt_kps_2d = np.array(gt_cuboid + [gt_centroid], dtype=np.float64)

            evaluator.add_prediction(R_gt, t_gt, R_pred, t_pred,
                                     gt_kps_2d, reproj_pred)

        status = "OK" if success else "FAIL"
        if (ji + 1) % 10 == 0 or ji == 0:
            print(f"  [{ji+1}/{len(json_files)}] {basename}: PnP {status}")

        # Visualization
        if args.visualize and success:
            vis = img.copy()
            # GT cuboid (green)
            gt_pts = [(int(p[0]), int(p[1])) for p in gt_cuboid]
            EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i, j in EDGES:
                cv2.line(vis, gt_pts[i], gt_pts[j], (0, 255, 0), 2)

            # Pred cuboid (yellow)
            pred_pts = [(int(p[0]), int(p[1])) for p in reproj_pred[:8]]
            for i, j in EDGES:
                cv2.line(vis, pred_pts[i], pred_pts[j], (0, 255, 255), 2)

            # Legend
            cv2.putText(vis, "GT (green) | Pred (yellow)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            vis_path = os.path.join(args.output_dir, f"{basename}_compare.jpg")
            cv2.imwrite(vis_path, vis)

    # ========== 결과 ==========
    summary = evaluator.summarize()
    summary["pnp_success_rate"] = pnp_success / max(pnp_total, 1)
    summary["pnp_total"] = pnp_total

    print(f"\n{'='*60}")
    print(f" Real Test Results ({len(json_files)} frames)")
    print(f"{'='*60}")
    print(f"  PnP success:      {pnp_success}/{pnp_total} ({summary['pnp_success_rate']*100:.1f}%)")

    if summary["num_predictions"] > 0:
        print(f"\n  --- 6D Pose Metrics ({summary['num_predictions']} frames) ---")
        print(f"  ADD mean:          {summary['ADD_mean']*100:.2f} cm")
        print(f"  ADD median:        {summary['ADD_median']*100:.2f} cm")
        print(f"  ADD (<0.1d):       {summary['ADD_correct_rate']*100:.1f}%")
        print(f"  ADD AUC:           {summary['ADD_auc']:.4f}")
        print(f"  ADD-S mean:        {summary['ADD-S_mean']*100:.2f} cm")
        print(f"  ADD-S median:      {summary['ADD-S_median']*100:.2f} cm")
        print(f"  ADD-S (<0.1d):     {summary['ADD-S_correct_rate']*100:.1f}%")
        print(f"  ADD-S AUC:         {summary['ADD-S_auc']:.4f}")
        print(f"  5cm-5deg:          {summary['5cm5deg_rate']*100:.1f}%")
        print(f"  Trans error mean:  {summary['trans_error_mean_cm']:.2f} cm")
        print(f"  Rot error mean:    {summary['rot_error_mean_deg']:.2f} deg")
        if "reproj_error_mean_px" in summary:
            print(f"  Reproj error mean: {summary['reproj_error_mean_px']:.2f} px")
    else:
        print("  (No successful predictions)")

    print(f"{'='*60}")

    # 저장
    summary_path = os.path.join(args.output_dir, "real_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()
