"""
challenge/scripts/dataset/make_pseudo_gt.py

baseline (또는 ft된) DOPE 모델로 시퀀스를 추론하고, sanity gate 를 통과한 frame
에 대해서만 NDDS JSON GT 를 자동 생성한다 (pseudo-label).

전략:
  1. seq_stats.py / run_live.py 와 동일한 gate 사용 (config/task.yaml inference.gates)
  2. 통과한 frame: PnP 결과를 manual GT 와 같은 NDDS 포맷으로 저장
  3. gt_source: "pseudo" 로 표시, manual 과 구분 가능
  4. overlay 샘플도 N frame 마다 저장 (검증용)

사용:
  python challenge/scripts/dataset/make_pseudo_gt.py --seq data/outside/capturepallet09
  python challenge/scripts/dataset/make_pseudo_gt.py --seq data/outside/capturepallet07 \
      --weights challenge/weights/finetuned/final_net_epoch_NNNN.pth --min_kp 6 --max_reproj 10
"""

from __future__ import annotations
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
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import yaml
from torch.autograd import Variable
from torchvision import transforms
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))
sys.path.append(os.path.join(_REPO, "Deep_Object_Pose", "common"))

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector

from pnp_solver import make_pallet_keypoints_3d
from run_live import (
    NpDepthFrame, load_seq, scale_K,
    _evaluate_result, _transform,
    CUBOID_EDGES_FRONT, CUBOID_EDGES_BACK, CUBOID_EDGES_VERTICAL,
)


def run_forward(net, img_rgb):
    t = _transform(img_rgb)
    with torch.no_grad():
        out, seg = net(Variable(t).cuda().unsqueeze(0))
    return out[-1][0], seg[-1][0]


def sample_depth_at(depth_np, x, y, radius=3):
    if depth_np is None:
        return None
    h, w = depth_np.shape
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < w and 0 <= ny < h:
                v = depth_np[ny, nx]
                if v > 50:
                    vals.append(float(v) / 1000.0)
    return float(np.median(vals)) if vals else None


def reproject_full_cuboid(R, t, K, dims):
    """8 corner + centroid 모두 image plane 으로 투영."""
    kp3d = make_pallet_keypoints_3d(*dims)
    pts_cam = (R @ kp3d.T).T + t
    proj = []
    for p in pts_cam:
        if p[2] <= 0:
            proj.append([-1.0, -1.0])
        else:
            u = K[0, 0] * p[0] / p[2] + K[0, 2]
            v = K[1, 1] * p[1] / p[2] + K[1, 2]
            proj.append([float(u), float(v)])
    return proj


def draw_overlay(img, proj_all, raw_points, status_text):
    vis = img.copy()
    pts = []
    for p in proj_all[:8]:
        if p[0] >= 0:
            pts.append((int(p[0]), int(p[1])))
        else:
            pts.append(None)
    for a, b in CUBOID_EDGES_FRONT:
        if pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (0, 220, 0), 3, cv2.LINE_AA)
    for a, b in CUBOID_EDGES_BACK + CUBOID_EDGES_VERTICAL:
        if pts[a] and pts[b]:
            cv2.line(vis, pts[a], pts[b], (0, 160, 0), 1, cv2.LINE_AA)
    for i, pt in enumerate(raw_points):
        if pt is None:
            continue
        c = (int(pt[0]), int(pt[1]))
        cv2.circle(vis, c, 4, (0, 255, 255) if i == 8 else (0, 200, 200), -1)
    cv2.putText(vis, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO, "challenge", "config", "task.yaml"))
    ap.add_argument("--seq",    required=True)
    ap.add_argument("--out_dir", default=None,
                    help="기본: challenge/data/<seq_name>_pseudo_gt")
    ap.add_argument("--weights", default=None, help="기본은 task.yaml baseline.weights")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--vis_every", type=int, default=20)
    # Gate overrides
    ap.add_argument("--thr",       type=float, default=None)
    ap.add_argument("--min_kp",    type=int,   default=None)
    ap.add_argument("--max_reproj", type=float, default=None)
    ap.add_argument("--z_max",     type=float, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    weights = args.weights or os.path.join(_REPO, C["baseline"]["weights"])
    bel = C["inference"]["belief"]
    gates = dict(C["inference"]["gates"])
    if args.min_kp is not None:    gates["min_detected_keypoints"] = args.min_kp
    if args.max_reproj is not None: gates["max_reproj_error_px"]    = args.max_reproj
    if args.z_max is not None:      gates["z_max_m"]                = args.z_max

    dims = (C["pallet"]["width"], C["pallet"]["depth"], C["pallet"]["height"])
    dim_cm = [dims[0] * 100, dims[2] * 100, dims[1] * 100]  # [W, H, L] for Cuboid3d

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO, args.seq)
    seq_name = os.path.basename(seq.rstrip("/\\"))
    out_dir = args.out_dir or os.path.join(_REPO, "challenge", "data", f"{seq_name}_pseudo_gt")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO, out_dir)
    vis_dir = os.path.join(out_dir, "_overlay")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    frames, K_seq = load_seq(seq)
    K = K_seq if K_seq is not None else np.array(
        [[C["camera"]["fx"], 0, C["camera"]["cx"]],
         [0, C["camera"]["fy"], C["camera"]["cy"]],
         [0, 0, 1]], dtype=np.float64)
    if args.max:
        frames = frames[:args.max * args.stride]
    frames = frames[::args.stride]

    # Model
    print(f"[Model] {weights}")
    sd_keys = list(torch.load(weights, map_location="cpu").keys())
    parallel = sd_keys[0].startswith("module.") if sd_keys else False
    print(f"[Model] DataParallel: {parallel}")
    model = ModelData(name="pallet", net_path=weights, parallel=parallel)
    model.load_net_model()
    pnp_solver = CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(dim_cm))
    pnp_solver.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = args.thr if args.thr is not None else bel["threshold"]
        softmax = 1000
        thresh_angle = bel["thresh_angle"]
        thresh_map = args.thr if args.thr is not None else bel["thresh_map"]
        sigma = bel["sigma"]
        thresh_points = args.thr if args.thr is not None else bel["thresh_points"]
    cfg = Cfg()

    print(f"[Pseudo GT] {seq_name} → {out_dir}")
    print(f"[Gates] thr={cfg.threshold:.3f} min_kp={gates['min_detected_keypoints']} "
          f"max_reproj={gates['max_reproj_error_px']}px z=[{gates['z_min_m']},{gates['z_max_m']}]m")
    print(f"[Dims] {dims}\n")

    ok = 0
    reasons = {}
    t0 = time.time()

    for i, (rgb_path, depth_path) in enumerate(frames):
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        depth_np = None
        if depth_path:
            d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if d is not None and d.dtype == np.uint16:
                depth_np = d

        h, w = img.shape[:2]
        proc_scale = 400.0 / h
        new_w = int(w * proc_scale) & ~7
        img_small = cv2.resize(img, (new_w, 400))
        K_small = scale_K(K, proc_scale)
        pnp_solver.set_camera_intrinsic_matrix(K_small)
        img_rgb = img_small[..., ::-1].copy()

        vertex2, aff = run_forward(model.net, img_rgb)
        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, cfg)
        except Exception:
            results = []

        best = None
        info = None
        last_reason = "no_result"
        for r in results:
            depth_cm = None
            raw = r.get("raw_points")
            if depth_np is not None and raw is not None and raw[8] is not None:
                dm = sample_depth_at(depth_np,
                                     raw[8][0] / proc_scale,
                                     raw[8][1] / proc_scale)
                if dm is not None:
                    depth_cm = dm * 100.0
            okg, reason, gi = _evaluate_result(r, gates, depth_cm, K_small)
            last_reason = reason
            if okg:
                best = r
                info = gi
                break

        if best is None:
            reasons[last_reason] = reasons.get(last_reason, 0) + 1
            continue

        # PnP 결과로 풀 cuboid (proc_scale 좌표 → 원본 좌표)
        loc = np.array(best["location"], dtype=np.float64)
        quat = np.array(best["quaternion"], dtype=np.float64)   # xyzw
        # quaternion → R (DOPE 컨벤션: location은 cm)
        from pyrr import Quaternion, matrix33
        q = Quaternion(quat)
        R = np.asarray(matrix33.create_from_quaternion(q), dtype=np.float64)
        t_m = loc / 100.0   # cm → m

        # 원본 K 로 reprojection (proc_scale 원복)
        proj_all = reproject_full_cuboid(R, t_m, K, dims)

        # 원본 좌표로 reprojection error 재계산 (raw_points는 proc_scale 기준)
        raw_orig = []
        for rp in best["raw_points"]:
            if rp is None:
                raw_orig.append(None)
            else:
                raw_orig.append([float(rp[0]) / proc_scale, float(rp[1]) / proc_scale])

        errs = []
        for k in range(9):
            if raw_orig[k] is None:
                continue
            du = proj_all[k][0] - raw_orig[k][0]
            dv = proj_all[k][1] - raw_orig[k][1]
            errs.append(float(np.hypot(du, dv)))
        reproj_orig = float(np.mean(errs)) if errs else float("inf")

        # NDDS JSON 저장
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t_m
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        ann = {
            "camera_data": {
                "width": w, "height": h,
                "intrinsics": {
                    "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                },
            },
            "objects": [{
                "class": "pallet",
                "name": "real_pallet",
                "visibility": 1,
                "pose_transform": T.tolist(),
                "projected_cuboid": proj_all[:8],
                "projected_cuboid_centroid": proj_all[8],
                "dimensions_m": {"width": dims[0], "height": dims[2], "depth": dims[1]},
                "gt_source": "pseudo",
                "pseudo_kps": raw_orig,
                "reproj_error_px": reproj_orig,
                "gate_info": info,
            }],
        }
        with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=2)
        # rgb hardlink/copy
        out_img = os.path.join(out_dir, f"{stem}.png")
        if not os.path.exists(out_img):
            try: os.link(rgb_path, out_img)
            except (OSError, NotImplementedError): shutil.copy2(rgb_path, out_img)
        ok += 1

        # Overlay 검증 샘플
        if i % args.vis_every == 0:
            vis = draw_overlay(img, proj_all, raw_orig,
                               f"frame{i} z={info['z_m']:.2f}m reproj={reproj_orig:.1f}px")
            cv2.imwrite(os.path.join(vis_dir, f"{stem}_overlay.png"), vis)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(frames)}] ok={ok} ({ok/(i+1):.1%})  "
                  f"({(i+1)/(time.time()-t0):.1f} fps)")

    dt = time.time() - t0
    print(f"\n=== Done ({dt:.1f}s) ===")
    print(f"  Pseudo GT: {ok}/{len(frames)} ({ok/max(len(frames),1):.1%})")
    print(f"  Output: {out_dir}")
    print(f"  Top reasons (rejected):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {k:30s} {v:5d}")


if __name__ == "__main__":
    main()
