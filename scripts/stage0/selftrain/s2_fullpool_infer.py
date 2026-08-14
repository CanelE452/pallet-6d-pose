"""s2_fullpool_infer.py — 6 도메인 full raw pool 에 s2 diffpnp(paper_s2_stageB ep0057)
추론 + GT-free 필터 입력(f1~f7) 을 한 번에 계산해 도메인별 jsonl 로 저장.

도메인 (manual=capturepallet11 제외):
  outside      outside/capturepallet02,03,04,05,08   (SEAL 07/09 제외)
  cad          outside/capturepalletcad
  noapril      capture0403noapril
  night        night/capturenight05,06,07            (SEAL 08/09 제외)
  wood_indoor  wood/_annotate_pallet_20260618_183705
  wood_outdoor wood/_annotate_pallet_20260618_184309

재사용: paper_s2_testset17_9filters(T) 의 infer_squash/flip_infer_squash/tta_stab_squash/M,
        paper_s2_filterval_9filters(F) 의 (W-1)-x flip 패치. n_det = 검출 코너 수(GT 무관).
필터 적용(FULL7 등)은 다음 단계에서 이 jsonl 을 읽어 처리 (이 스크립트는 추론만).

Usage: conda activate pallet-pose; python -u scripts/stage0/selftrain/s2_fullpool_infer.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401  ((W-1)-x flip 패치를 T 에 적용)
import paper_s2_testset17_9filters as T   # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import cv2                                # noqa: E402
import torch                              # noqa: E402

M, N_DET_MIN, E = T.M, T.N_DET_MIN, T.E
# argv: [weights] [out_subdir] [--skip-wood]
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".pth") else T.WEIGHTS
_OUT_NAME = next((a for a in sys.argv[1:] if not a.endswith(".pth") and not a.startswith("--")),
                 "paper_s2_fullpool_infer")
SKIP_WOOD = "--skip-wood" in sys.argv
RAW = os.path.join(ROOT, "data", "pallet", "raw_data")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "results", _OUT_NAME)

DOMAINS = {
    "outside": [os.path.join(RAW, "outside", s) for s in
                ("capturepallet02", "capturepallet03", "capturepallet04",
                 "capturepallet05", "capturepallet08")],
    "cad": [os.path.join(RAW, "outside", "capturepalletcad")],
    "noapril": [os.path.join(RAW, "capture0403noapril")],
    "night": [os.path.join(RAW, "night", s) for s in
              ("capturenight05", "capturenight06", "capturenight07")],
    "wood_indoor": [os.path.join(RAW, "wood", "_annotate_pallet_20260618_183705")],
    "wood_outdoor": [os.path.join(RAW, "wood", "_annotate_pallet_20260618_184309")],
}
DIMS = {"wood_indoor": (0.8, 0.59, 0.14), "wood_outdoor": (0.8, 0.59, 0.14)}
DIMS_DEFAULT = (1.1, 1.3, 0.12)
DEFAULT_K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]], float)


def load_K(seq_dir):
    kp = os.path.join(seq_dir, "cam_K.txt")
    return np.loadtxt(kp).reshape(3, 3) if os.path.isfile(kp) else DEFAULT_K


def _pt(p):
    return None if p is None or (isinstance(p, np.ndarray) and np.isnan(p[0])) else \
        [float(p[0]), float(p[1])]


def _f(x):
    """None-safe float (flip_score/frsep 등은 None 반환 가능)."""
    return None if x is None else float(x)


def build_rec_gtfree(model, img, K, device):
    """GT 없이 f1~f7 필터 입력 계산 (build_rec 의 GT-free 버전)."""
    _, pred8, pred_c, peaks, ratios = T.infer_squash(model, img, device)
    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    n_det = int(np.sum(~np.isnan(pred8[:, 0])))
    H, W = img.shape[:2]
    img_diag = float(np.hypot(W, H))
    scores = {}
    f7 = False
    pred_sr = pred_asp = None
    if n_det >= N_DET_MIN:
        scores["f1_peak"] = _f(min(peaks[i] for i in det8))
        scores["f2_peak_ratio"] = _f(min(ratios[i] for i in det8))
        kpB = T.flip_infer_squash(model, img, device)
        scores["f3_flip"] = _f(M.flip_score(pred8, pred_c, kpB))
        scores["f4_tta_stab"] = _f(T.tta_stab_squash(model, img, device))
        rear = [i for i in (4, 5, 6, 7) if not np.isnan(pred8[i, 0])]
        scores["f5_rear_conf"] = _f(min(peaks[i] for i in rear)) if len(rear) == 4 else None
        scores["f6_frsep"] = _f(M.frsep_frac(pred8))
        f7ok, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
        f7 = bool(f7ok)
        sr, asp = M.size_aspect(pred8, img_diag)
        pred_sr, pred_asp = _f(sr), _f(asp)
    return {
        "n_det": n_det,
        "scores": scores,
        "f7_posdepth": f7,
        "pred_sr": pred_sr,
        "pred_asp": pred_asp,
        "pred8": [_pt(pred8[i]) for i in range(8)],
        "pred_c": _pt(pred_c),
        "peaks": [float(x) for x in peaks],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(WEIGHTS, device)
    print(f"[init] weights={WEIGHTS} device={device}", flush=True)

    summary = []
    for dom, seqs in DOMAINS.items():
        if SKIP_WOOD and dom in ("wood_indoor", "wood_outdoor"):
            continue
        APNP.PALLET_DIMS = DIMS.get(dom, DIMS_DEFAULT)   # f7 posdepth 용 dims (wood 별도)
        out_jsonl = os.path.join(OUT_DIR, f"{dom}.jsonl")
        frames = []
        for seq in seqs:
            for p in sorted(glob.glob(os.path.join(seq, "rgb", "*.png"))):
                frames.append((os.path.basename(seq), p))
        n_tot = len(frames)
        n_det6 = 0
        print(f"\n[{dom}] {n_tot} frames  dims={APNP.PALLET_DIMS}  seqs={len(seqs)}", flush=True)
        with open(out_jsonl, "w") as f:
            for i, (sess, ip) in enumerate(frames):
                img = cv2.imread(ip)
                if img is None:
                    continue
                K = load_K(os.path.dirname(os.path.dirname(ip)))
                rec = build_rec_gtfree(model, img, K, device)
                rec["domain"] = dom
                rec["session"] = sess
                rec["fid"] = os.path.splitext(os.path.basename(ip))[0]
                if rec["n_det"] >= N_DET_MIN:
                    n_det6 += 1
                f.write(json.dumps(rec) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  [{dom}] {i+1}/{n_tot}  det>=6 so far={n_det6}", flush=True)
        print(f"[{dom}] done: {n_tot} frames, det>=6={n_det6}  -> {out_jsonl}", flush=True)
        summary.append((dom, n_tot, n_det6))

    L = ["# s2 diffpnp full-pool inference (6 domains, manual 제외)",
         "",
         "- weights: paper_s2_stageB net_epoch_0057 (diffpnp λ0.005, squash-parity)",
         "- n_det = 검출 코너 수(>=6 검출). scores f1~f7 는 GT-free 필터 입력.",
         "- wood dims=(0.8,0.59,0.14), 그 외 (1.1,1.3,0.12).",
         "",
         "```",
         f"{'domain':<14}{'frames':>8}{'det>=6':>9}{'det%':>8}",
         "-" * 40]
    for dom, n, d in summary:
        L.append(f"{dom:<14}{n:>8}{d:>9}{(100*d/n if n else 0):>7.1f}%")
    L.append("-" * 40)
    tn = sum(n for _, n, _ in summary)
    td = sum(d for _, _, d in summary)
    L.append(f"{'ALL':<14}{tn:>8}{td:>9}{(100*td/tn if tn else 0):>7.1f}%")
    L.append("```")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\n[save] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
