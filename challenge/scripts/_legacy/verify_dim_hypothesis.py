"""verify_dim_hypothesis.py — Q4 dim mismatch 가설 검증.

capturepallet09 frame N개에 challenge ep60 forward 1회 → 동일 belief 로 3 dim PnP 풀어서
z, reproj 비교. 학습 데이터의 정사각형 (mixed_v8 110×110×15) bias 가 실제 plastic 직사각형
(110×130×11) dim 추론 시 z 를 부풀리는지 확인.

기대:
  dim_real  z 가 비현실적으로 크고 (>5m)
  dim_sq11  z 가 정상 범위 (1~3m)        → dim mismatch 가설 확정
또는
  세 dim 모두 z 가 비슷하게 큼            → belief 자체가 부정확. 다른 원인 찾아야 함.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
from torchvision import transforms

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def scale_K(K, s):
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def run_forward(net, img_rgb):
    t = _transform(img_rgb)
    with torch.no_grad():
        out, seg = net(Variable(t).cuda().unsqueeze(0))
    return out[-1][0], seg[-1][0]


def reproj_error(raw_points, proj_points):
    if proj_points is None:
        return float("inf")
    errs = []
    for r, p in zip(raw_points, proj_points):
        if r is None or p is None:
            continue
        errs.append(np.hypot(r[0] - p[0], r[1] - p[1]))
    return float(np.mean(errs)) if errs else float("inf")


def kp_count(raw_points):
    return sum(1 for p in raw_points if p is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/outside/capturepallet09")
    ap.add_argument("--weights", default="weights/challenge/final_net_epoch_0060.pth")
    ap.add_argument("--n_frames", type=int, default=10, help="균등 sample 할 frame 수")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO_ROOT, args.seq)
    weights = args.weights if os.path.isabs(args.weights) else os.path.join(_REPO_ROOT, args.weights)

    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
    K_path = os.path.join(seq, "cam_K.txt")
    K = np.loadtxt(K_path, dtype=np.float64).reshape(3, 3)
    print(f"[Seq] {seq} — {len(rgb_paths)} frames")
    print(f"[K]   fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"[W]   {weights}")

    # 균등 sample
    stride = max(1, len(rgb_paths) // args.n_frames)
    sampled = rgb_paths[::stride][:args.n_frames]

    # DOPE 모델
    sd_keys = list(torch.load(weights, map_location="cpu").keys())
    parallel = sd_keys[0].startswith("module.") if sd_keys else False
    print(f"[Model] parallel={parallel}")
    model = ModelData(name="pallet", net_path=weights, parallel=parallel)
    model.load_net_model()

    # 3 dim 후보 (cm)
    dim_configs = [
        ("dim_real ", [110.0, 11.0, 130.0]),   # 실제 plastic (현재 task.yaml)
        ("dim_sq11 ", [110.0, 11.0, 110.0]),   # mixed_v8 정사각형 + 실제 두께
        ("dim_v8   ", [110.0, 15.0, 110.0]),   # mixed_v8 완전 일치
    ]
    solvers = {name: CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(d)) for name, d in dim_configs}
    for s in solvers.values():
        s.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = args.threshold
        softmax = 1000
        thresh_angle = 0.5
        thresh_map = 0.30
        sigma = 3
        thresh_points = 0.30
    cfg = Cfg()

    # 결과 누적
    print()
    print("─" * 96)
    header = f"{'frame':>5}  {'kp':>3}  "
    for name, _ in dim_configs:
        header += f"{name}z(m)  reproj  "
    print(header)
    print("─" * 96)

    z_collect = {name: [] for name, _ in dim_configs}
    reproj_collect = {name: [] for name, _ in dim_configs}

    for fi, path in enumerate(sampled):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[F{fi}] read fail")
            continue
        h, w = img.shape[:2]
        proc_scale = 400.0 / h
        new_w = int(w * proc_scale) & ~7
        img_small = cv2.resize(img, (new_w, 400))
        K_small = scale_K(K, proc_scale)
        img_rgb = img_small[..., ::-1].copy()

        vertex2, aff = run_forward(model.net, img_rgb)

        # 각 dim 으로 PnP 풀이 — solver 만 바꿔서 같은 belief 로
        row_per_dim = {}
        n_kp_best = 0
        for name, _ in dim_configs:
            solver = solvers[name]
            solver.set_camera_intrinsic_matrix(K_small)
            try:
                results = ObjectDetector.find_object_poses(vertex2, aff, solver, cfg)
            except Exception as e:
                results = []
            if not results:
                row_per_dim[name] = (None, None, 0)
                continue
            r = results[0]
            raw = r.get("raw_points")
            loc = r.get("location")
            proj = r.get("projected_points")
            n_kp = kp_count(raw) if raw else 0
            n_kp_best = max(n_kp_best, n_kp)
            if loc is None:
                row_per_dim[name] = (None, None, n_kp)
                continue
            z_m = float(loc[2]) / 100.0
            re = reproj_error(raw, proj)
            row_per_dim[name] = (z_m, re, n_kp)
            z_collect[name].append(z_m)
            reproj_collect[name].append(re)

        row_str = f"F{fi:>3}   {n_kp_best:>3}  "
        for name, _ in dim_configs:
            z, re, _ = row_per_dim[name]
            if z is None:
                row_str += f"{'  N/A':>9}  {'N/A':>6}  "
            else:
                row_str += f"{z:>8.2f}   {re:>5.1f}  "
        print(row_str)

    # 요약
    print("─" * 96)
    print()
    print("[요약] dim 별 median z + median reproj")
    print()
    for name, d in dim_configs:
        zs = z_collect[name]
        res = reproj_collect[name]
        if not zs:
            print(f"  {name} dim={d}  (PnP 모두 실패)")
            continue
        z_med = float(np.median(zs))
        z_min = float(np.min(zs))
        z_max = float(np.max(zs))
        re_med = float(np.median(res))
        print(f"  {name} dim={d}  z_median={z_med:5.2f}m  z_range=[{z_min:4.2f},{z_max:5.2f}]  "
              f"reproj_median={re_med:5.1f}px  ({len(zs)}/{len(sampled)} solved)")

    # 가설 판정 자동 출력
    print()
    z_real = z_collect.get("dim_real ", [])
    z_sq11 = z_collect.get("dim_sq11 ", [])
    if z_real and z_sq11:
        med_real = float(np.median(z_real))
        med_sq11 = float(np.median(z_sq11))
        ratio = med_real / med_sq11 if med_sq11 > 0 else float("inf")
        print(f"[가설 판정] med(dim_real)/med(dim_sq11) = {med_real:.2f} / {med_sq11:.2f} = {ratio:.2f}x")
        if ratio > 1.5:
            print(f"  → dim_real z 가 {ratio:.1f}x 부풀음. **dim mismatch 가설 확정 가능성 매우 높음**.")
            print(f"    학습 데이터가 정사각형 (mixed_v8) bias 가 강함.")
        elif med_real > 5.0 and med_sq11 > 5.0:
            print(f"  → 두 dim 모두 z>5m. belief 자체가 부정확할 가능성 (멀리 시퀀스, OOD).")
        else:
            print(f"  → ratio 가 작음. dim 영향 미미. 다른 원인 (belief 부정확, K 등) 의심.")


if __name__ == "__main__":
    main()
