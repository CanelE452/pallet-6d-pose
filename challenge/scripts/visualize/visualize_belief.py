"""visualize_belief.py — capturepallet09 belief map 디버그 시각화.

특정 frame 에 challenge ep60 forward → 한 장의 큰 PNG 로:
  좌측 상단  : 원본 RGB + raw peak overlay (9 channel 각 색깔)
  좌측 하단  : 원본 RGB + PnP wireframe + projected keypoints (3 dim 모두)
  우측       : 9 belief heatmap 3×3 grid (각 channel peak val 표시)
  하단 정보  : per-channel peak val + 3 dim 별 z + reproj

사용:
  python challenge/scripts/visualize/visualize_belief.py --frames 0,3,6,150,500
  python challenge/scripts/visualize/visualize_belief.py --n_frames 20  # 균등 sample
  python challenge/scripts/visualize/visualize_belief.py --range 0,2773,277  # start, end, stride
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
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
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


# 9 keypoint 색깔 (BGR) — 0~7 corner / 8 centroid
KP_COLORS = [
    (255,   0,   0),   # 0  blue       front-top-LEFT
    (255, 128,   0),   # 1  cyan       front-top-RIGHT
    (  0, 255, 255),   # 2  yellow     front-bottom-RIGHT
    (  0, 200,   0),   # 3  green      front-bottom-LEFT
    (255,   0, 255),   # 4  magenta    rear-top-LEFT
    (128,   0, 255),   # 5  pink       rear-top-RIGHT
    (  0,   0, 255),   # 6  red        rear-bottom-RIGHT
    ( 80,  80, 255),   # 7  salmon     rear-bottom-LEFT
    (255, 255, 255),   # 8  white      centroid
]

CUBOID_EDGES_FRONT    = [(0, 1), (1, 2), (2, 3), (3, 0)]
CUBOID_EDGES_BACK     = [(4, 5), (5, 6), (6, 7), (7, 4)]
CUBOID_EDGES_VERTICAL = [(0, 4), (1, 5), (2, 6), (3, 7)]


def scale_K(K, s):
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def run_forward(net, img_rgb):
    t = _transform(img_rgb)
    with torch.no_grad():
        out, seg = net(Variable(t).cuda().unsqueeze(0))
    return out[-1][0], seg[-1][0]


def extract_peaks(vertex2, sigma=3):
    peaks = []
    for ch in range(vertex2.size(0)):
        raw = vertex2[ch].cpu().numpy()
        sm = gaussian_filter(raw, sigma=sigma)
        val = float(sm.max())
        idx = np.unravel_index(sm.argmax(), sm.shape)
        peaks.append({'val': val, 'x': idx[1] * 8, 'y': idx[0] * 8})
    return peaks


def build_belief_grid(vertex2, img_bgr, peaks, threshold):
    """9 belief 채널을 3×3 grid heatmap + 각 cell 의 peak 표시."""
    upsampling = nn.UpsamplingNearest2d(scale_factor=8)
    h, w = img_bgr.shape[:2]
    cells = []
    for ch in range(min(vertex2.size(0), 9)):
        b = vertex2[ch].clone()
        bmin, bmax = float(b.min()), float(b.max())
        if bmax > bmin:
            b = (b - bmin) / (bmax - bmin)
        b_up = upsampling(b.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
        b_np = cv2.resize(b_up.cpu().numpy(), (w, h))
        hm = cv2.applyColorMap((b_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        bg = (img_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        ov = cv2.addWeighted(bg, 1, hm, 0.65, 0)
        # peak 표시
        pk = peaks[ch]
        px, py = pk['x'], pk['y']
        if 0 <= px < w and 0 <= py < h:
            color = (0, 255, 0) if pk['val'] > threshold else (100, 100, 200)
            cv2.drawMarker(ov, (px, py), color, cv2.MARKER_CROSS, 14, 2)
        # 라벨 + peak val
        lbl = f"c{ch}" if ch < 8 else "ctr"
        above = " HIT" if pk['val'] > threshold else "    "
        cv2.putText(ov, f"{lbl} {pk['val']:.3f}{above}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if pk['val'] > threshold else (180, 180, 180), 1)
        cells.append(ov)
    while len(cells) < 9:
        cells.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[r*3:r*3+3]) for r in range(3)]
    return np.vstack(rows)


def draw_raw_peaks(img, peaks, threshold):
    """원본 image 에 raw peak (9 채널 각 색깔) overlay."""
    for ch, pk in enumerate(peaks):
        px, py = pk['x'], pk['y']
        if not (0 <= px < img.shape[1] and 0 <= py < img.shape[0]):
            continue
        above = pk['val'] > threshold
        color = KP_COLORS[ch] if above else tuple(int(c * 0.4) for c in KP_COLORS[ch])
        r = 6 if ch == 8 else 4
        cv2.circle(img, (px, py), r, color, 2 if above else 1, cv2.LINE_AA)
        lbl = f"c{ch}" if ch < 8 else "ctr"
        cv2.putText(img, lbl, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"{pk['val']:.3f}", (px + 6, py + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color if above else (140, 140, 140), 1, cv2.LINE_AA)


def draw_wireframe(img, proj_pts, color, thickness=2):
    pts = []
    for i in range(8):
        if i < len(proj_pts) and proj_pts[i] is not None:
            p = proj_pts[i]
            pts.append((int(p[0]), int(p[1])))
        else:
            pts.append(None)
    for a, b in CUBOID_EDGES_BACK:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], tuple(int(c * 0.5) for c in color),
                     max(1, thickness - 1), cv2.LINE_AA)
    for a, b in CUBOID_EDGES_VERTICAL:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for a, b in CUBOID_EDGES_FRONT:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, thickness + 1, cv2.LINE_AA)


def reproj_error(raw, proj):
    if proj is None or raw is None:
        return float("inf")
    errs = []
    for r, p in zip(raw, proj):
        if r is None or p is None:
            continue
        errs.append(np.hypot(r[0] - p[0], r[1] - p[1]))
    return float(np.mean(errs)) if errs else float("inf")


def kp_count(raw):
    return sum(1 for p in raw if p is not None) if raw else 0


def parse_frames(args, total):
    """--frames / --n_frames / --range 중 하나로 frame index 리스트 결정."""
    if args.frames:
        return [int(x) for x in args.frames.split(",")]
    if args.range:
        parts = args.range.split(",")
        start = int(parts[0]); end = int(parts[1]); stride = int(parts[2]) if len(parts) > 2 else 1
        return list(range(start, min(end, total), stride))
    # n_frames 균등
    stride = max(1, total // args.n_frames)
    return list(range(0, total, stride))[:args.n_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/outside/capturepallet09")
    ap.add_argument("--weights", default="weights/challenge/final_net_epoch_0060.pth")
    ap.add_argument("--out_dir", default="debug/belief_viz")
    ap.add_argument("--frames", default=None, help="콤마 구분: 0,3,6,150")
    ap.add_argument("--n_frames", type=int, default=10, help="균등 sample 수")
    ap.add_argument("--range", default=None, help="start,end[,stride]")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO_ROOT, args.seq)
    weights = args.weights if os.path.isabs(args.weights) else os.path.join(_REPO_ROOT, args.weights)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
    K = np.loadtxt(os.path.join(seq, "cam_K.txt"), dtype=np.float64).reshape(3, 3)
    print(f"[Seq] {seq} — {len(rgb_paths)} frames")
    print(f"[K]   fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"[Out] {out_dir}")

    # 모델
    sd_keys = list(torch.load(weights, map_location="cpu").keys())
    parallel = sd_keys[0].startswith("module.") if sd_keys else False
    model = ModelData(name="pallet", net_path=weights, parallel=parallel)
    model.load_net_model()

    dim_configs = [
        ("real",  [110.0, 11.0, 130.0], (  0, 255, 255)),  # yellow
        ("sq11",  [110.0, 11.0, 110.0], (255,   0, 255)),  # magenta
        ("v8",    [110.0, 15.0, 110.0], (255, 255,   0)),  # cyan
    ]
    solvers = {n: CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(d)) for n, d, _ in dim_configs}
    for s in solvers.values():
        s.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = args.threshold
        softmax = 1000; thresh_angle = 0.5
        thresh_map = 0.30; sigma = 3; thresh_points = 0.30
    cfg = Cfg()

    selected = parse_frames(args, len(rgb_paths))
    print(f"[Frames] {len(selected)} frames: {selected[:10]}{'...' if len(selected) > 10 else ''}")

    saved = []
    for fi in selected:
        if fi < 0 or fi >= len(rgb_paths):
            continue
        path = rgb_paths[fi]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        proc_scale = 400.0 / h
        new_w = int(w * proc_scale) & ~7
        img_small = cv2.resize(img, (new_w, 400))
        K_small = scale_K(K, proc_scale)
        img_rgb = img_small[..., ::-1].copy()

        vertex2, aff = run_forward(model.net, img_rgb)
        peaks = extract_peaks(vertex2, sigma=cfg.sigma)

        # 3 dim PnP
        dim_results = {}
        for name, _, color in dim_configs:
            solver = solvers[name]
            solver.set_camera_intrinsic_matrix(K_small)
            try:
                results = ObjectDetector.find_object_poses(vertex2, aff, solver, cfg)
            except Exception:
                results = []
            if not results:
                dim_results[name] = None
                continue
            r = results[0]
            raw = r.get("raw_points")
            loc = r.get("location")
            proj = r.get("projected_points")
            re = reproj_error(raw, proj)
            n_kp = kp_count(raw)
            z = float(loc[2]) / 100.0 if loc is not None else None
            dim_results[name] = {
                "raw": raw, "proj": proj, "z": z, "reproj": re, "n_kp": n_kp, "color": color
            }

        # ── 좌측 상단: raw peak overlay ──
        img_peaks = img_small.copy()
        draw_raw_peaks(img_peaks, peaks, cfg.threshold)
        cv2.putText(img_peaks, f"F{fi} RAW PEAKS (thr={cfg.threshold:.2f})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2, cv2.LINE_AA)

        # ── 좌측 하단: 3 dim PnP wireframe ──
        img_pnp = img_small.copy()
        for name, _, color in dim_configs:
            r = dim_results[name]
            if r is None or r["proj"] is None:
                continue
            draw_wireframe(img_pnp, r["proj"], color, thickness=2)
        cv2.putText(img_pnp, f"F{fi} PnP wireframes (3 dim)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2, cv2.LINE_AA)
        # 범례
        ly = 50
        for name, dim, color in dim_configs:
            r = dim_results[name]
            if r and r["z"] is not None:
                txt = f"{name:5s} {dim}  z={r['z']:5.2f}m  reproj={r['reproj']:5.1f}px  kp={r['n_kp']}"
            else:
                txt = f"{name:5s} {dim}  PnP FAIL"
            cv2.putText(img_pnp, txt, (10, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            ly += 18

        # ── 좌측 패널 합치기 (peak overlay 위 + PnP wireframe 아래) ──
        left_panel = np.vstack([img_peaks, img_pnp])

        # ── 우측: 9 belief grid (3×3) ──
        belief_grid = build_belief_grid(vertex2, img_small, peaks, cfg.threshold)
        # belief_grid 크기 = (3h × 3w). left_panel = (2h × w). 크기 맞춰 resize
        target_h = left_panel.shape[0]
        scale = target_h / belief_grid.shape[0]
        new_bgw = int(belief_grid.shape[1] * scale)
        belief_grid_resized = cv2.resize(belief_grid, (new_bgw, target_h))

        # ── 합치기 + 하단 info bar ──
        full = np.hstack([left_panel, belief_grid_resized])

        # info bar
        info_h = 100
        info_bar = np.full((info_h, full.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(info_bar, f"capturepallet09 / F{fi} / {os.path.basename(path)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        # per-channel peak val
        peak_str = "peaks: " + " ".join(
            f"c{i}={p['val']:.2f}" + ("*" if p['val'] > cfg.threshold else "")
            for i, p in enumerate(peaks)
        )
        cv2.putText(info_bar, peak_str, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        n_above = sum(1 for p in peaks if p['val'] > cfg.threshold)
        verdict = "DETECT OK" if n_above >= 7 else f"INSUF KP ({n_above}/9 above thr)"
        cv2.putText(info_bar, verdict, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0) if n_above >= 7 else (0, 0, 220), 2, cv2.LINE_AA)

        full = np.vstack([full, info_bar])

        # 저장
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"F{fi:04d}_{stem}.png")
        cv2.imwrite(out_path, full)
        saved.append(out_path)
        z_real = dim_results["real"]["z"] if dim_results["real"] else None
        print(f"  F{fi:>4} n_above={n_above}/9 "
              f"z_real={z_real if z_real else 'NA':>5}{'m' if z_real else ''}  "
              f"-> {out_path}")

    print()
    print(f"[Done] {len(saved)} files saved")
    print(f"저장: {out_dir}/")


if __name__ == "__main__":
    main()
