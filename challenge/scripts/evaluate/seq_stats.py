"""
challenge/scripts/evaluate/seq_stats.py

시퀀스(들)에 대해 baseline 추론을 돌려 false positive/통과율을 측정한다.
GUI 없음 — headless. run_live.py 의 게이트 로직(_evaluate_result) 재사용.

사용법:
  python challenge/scripts/evaluate/seq_stats.py --seq data/outside/capturepallet01
  python challenge/scripts/evaluate/seq_stats.py --seq data/outside/capturepallet01 --max 60
  python challenge/scripts/evaluate/seq_stats.py --all
"""

from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import os
import sys
import time
import json
import argparse
import glob
from collections import Counter

import cv2
import yaml
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_REPO_ROOT, "Deep_Object_Pose", "common"))

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector

# run_live.py 함수 재사용
from run_live import (
    NpDepthFrame, load_seq, scale_K,
    extract_peaks, _evaluate_result, _transform,
)


def run_forward_local(net, img_rgb):
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
                if v > 50:  # mm
                    vals.append(float(v) / 1000.0)
    return float(np.median(vals)) if vals else None


def eval_seq(seq_dir, net, pnp_solver, cfg, cfg_gates, dim_cm, max_frames=None,
             stride=1, K_override=None):
    frames, K_seq = load_seq(seq_dir)
    K = K_override if K_override is not None else (K_seq if K_seq is not None else None)
    if K is None:
        raise RuntimeError(f"{seq_dir}: cam_K.txt 없음 + override 미지정")

    if max_frames:
        frames = frames[:max_frames * stride]
    frames = frames[::stride]

    reason_cnt = Counter()
    confirmed = 0
    pnp_success = 0      # gate 무시, find_object_poses 가 location 반환한 frame
    raw_detect = 0       # raw_points 중 1개 이상 → run_dope_live의 detected 로직
    total = 0
    consec = 0
    confirm_n = 2
    z_hist = []  # 통과한 frame의 z
    reproj_hist = []

    t0 = time.time()
    for rgb_path, depth_path in frames:
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            reason_cnt["read_fail"] += 1
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

        vertex2, aff = run_forward_local(net, img_rgb)
        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, cfg)
        except Exception:
            results = []

        best = None
        last_reason = "no_result"
        frame_pnp = False
        frame_raw = False
        for r in results:
            raw = r.get("raw_points")
            if raw is not None and any(p is not None for p in raw):
                frame_raw = True
            if r.get("location") is not None:
                frame_pnp = True
            depth_cm = None
            if depth_np is not None and raw is not None and raw[8] is not None:
                dm = sample_depth_at(depth_np,
                                     raw[8][0] / proc_scale,
                                     raw[8][1] / proc_scale)
                if dm is not None:
                    depth_cm = dm * 100.0
            ok, reason, info = _evaluate_result(r, cfg_gates, depth_cm, K_small)
            last_reason = reason
            if ok:
                best = (r, info)
                break

        if frame_raw:
            raw_detect += 1
        if frame_pnp:
            pnp_success += 1

        if best is not None:
            consec += 1
            z_hist.append(best[1]["z_m"])
            reproj_hist.append(best[1]["reproj"])
        else:
            consec = 0
        if consec >= confirm_n:
            confirmed += 1
        reason_cnt[last_reason] += 1
        total += 1

    dt = time.time() - t0
    return {
        "seq": os.path.basename(seq_dir.rstrip("/\\")),
        "frames_evaluated": total,
        "raw_detect_frames": raw_detect,        # raw_points >=1 (run_dope_live.py 로직)
        "raw_detect_rate": raw_detect / max(total, 1),
        "pnp_success_frames": pnp_success,      # location 반환 (gate 무시)
        "pnp_success_rate": pnp_success / max(total, 1),
        "confirmed_frames": confirmed,          # gate + temporal 통과
        "confirmed_rate": confirmed / max(total, 1),
        "reasons": dict(reason_cnt),
        "z_mean_m": float(np.mean(z_hist)) if z_hist else None,
        "z_std_m":  float(np.std(z_hist))  if z_hist else None,
        "reproj_mean_px": float(np.mean(reproj_hist)) if reproj_hist else None,
        "elapsed_s": dt,
        "fps": total / max(dt, 1e-6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO_ROOT, "challenge", "config", "task.yaml"))
    ap.add_argument("--seq",    default=None, help="단일 시퀀스 폴더")
    ap.add_argument("--all",    action="store_true", help="data/outside/capture* 전부")
    ap.add_argument("--max",    type=int, default=None, help="시퀀스당 최대 프레임")
    ap.add_argument("--stride", type=int, default=1, help="프레임 stride")
    ap.add_argument("--report", default=None, help="JSON 결과 저장 경로")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--thr",     type=float, default=None, help="belief threshold override")
    ap.add_argument("--min_kp",  type=int,   default=None, help="gate min_kp override")
    ap.add_argument("--thr_sweep", default=None, help="콤마 구분: 0.05,0.10,0.15,...")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    weights = args.weights or os.path.join(_REPO_ROOT, C["baseline"]["weights"])
    bel = C["inference"]["belief"]
    gates = C["inference"]["gates"]
    dim_cm = [C["pallet"]["width"] * 100,
              C["pallet"]["height"] * 100,
              C["pallet"]["depth"] * 100]

    class Cfg:
        mask_edges = 1; mask_faces = 1; vertex = 1
        threshold = bel["threshold"]; softmax = 1000
        thresh_angle = bel["thresh_angle"]
        thresh_map = bel["thresh_map"]
        sigma = bel["sigma"]
        thresh_points = bel["thresh_points"]
    cfg = Cfg()
    if args.min_kp is not None:
        gates = dict(gates); gates["min_detected_keypoints"] = args.min_kp

    print(f"[Gates] min_kp={gates['min_detected_keypoints']} max_reproj={gates['max_reproj_error_px']}px "
          f"z=[{gates['z_min_m']},{gates['z_max_m']}]m depth_rel<{gates['depth_pnp_z_max_rel']}")
    print(f"[Model] {weights}")
    # state_dict 키가 "module."으로 시작하면 DataParallel 저장본 → parallel=True
    sd_keys = list(torch.load(weights, map_location="cpu").keys())
    parallel = sd_keys[0].startswith("module.") if sd_keys else False
    print(f"[Model] DataParallel checkpoint: {parallel}")
    model = ModelData(name="pallet", net_path=weights, parallel=parallel)
    model.load_net_model()
    pnp_solver = CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(dim_cm))
    pnp_solver.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

    if args.all:
        seqs = sorted(glob.glob(os.path.join(_REPO_ROOT, "data", "outside", "capture*")))
        seqs = [s for s in seqs if os.path.isdir(s)]
    elif args.seq:
        seqs = [args.seq if os.path.isabs(args.seq) else os.path.join(_REPO_ROOT, args.seq)]
    else:
        ap.error("--seq or --all required")

    # threshold sweep 리스트 결정
    if args.thr_sweep:
        thr_list = [float(x) for x in args.thr_sweep.split(",")]
    elif args.thr is not None:
        thr_list = [args.thr]
    else:
        thr_list = [bel["threshold"]]

    all_results = []
    for thr in thr_list:
        cfg.threshold = thr
        cfg.thresh_map = thr
        cfg.thresh_points = thr
        print(f"\n############ thr={thr:.3f} ############")
        for s in seqs:
            if not glob.glob(os.path.join(s, "rgb", "*.png")):
                print(f"[Skip] {os.path.basename(s)} — no RGB")
                continue
            print(f"\n=== {os.path.basename(s)} (thr={thr:.3f}) ===")
            try:
                r = eval_seq(s, model.net, pnp_solver, cfg, gates, dim_cm,
                             max_frames=args.max, stride=args.stride)
            except Exception as e:
                print(f"[ERR] {e}")
                continue
            r["thr"] = thr
            all_results.append(r)
            print(f"  frames={r['frames_evaluated']}  "
                  f"raw_det={r['raw_detect_rate']:.1%}  "
                  f"pnp_ok={r['pnp_success_rate']:.1%}  "
                  f"confirmed={r['confirmed_rate']:.1%}  "
                  f"fps={r['fps']:.1f}")
            top = sorted(r["reasons"].items(), key=lambda kv: -kv[1])[:5]
            for k, v in top:
                print(f"    {k:30s} {v:5d}  ({v/max(r['frames_evaluated'],1):.1%})")
            if r["z_mean_m"] is not None:
                print(f"  passed: z={r['z_mean_m']:.2f}±{r['z_std_m']:.2f}m  "
                      f"reproj={r['reproj_mean_px']:.1f}px")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[Report] {args.report}")

    # Sweep 요약 표 (thr 별 raw/pnp/confirmed rate)
    if len(thr_list) > 1 or len(seqs) > 1:
        print("\n=========== SWEEP SUMMARY ===========")
        print(f"{'seq':<22} {'thr':>5}  {'raw':>6}  {'pnp':>6}  {'cfm':>6}  {'n':>5}")
        for r in all_results:
            print(f"{r['seq']:<22} {r['thr']:>5.3f}  "
                  f"{r['raw_detect_rate']:>5.1%}  "
                  f"{r['pnp_success_rate']:>5.1%}  "
                  f"{r['confirmed_rate']:>5.1%}  "
                  f"{r['frames_evaluated']:>5d}")


if __name__ == "__main__":
    main()
