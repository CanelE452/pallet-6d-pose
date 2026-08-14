"""diag8_forgetting.py — B0/B1 real 검출붕괴 원인 진단 (학습 X, ~10분).
3모델(baseline challenge0123 / B0 v3-10k / B1 v3+addon-16k)에 대해:
 (1) real: threshold-free raw top-1 corner error vs thresholded det% vs corner peak score
 (2) old-synth(training_data/val): det% — forgetting이면 old도 붕괴 / 유지면 real-specific sim2real
판정: raw좌표 OK+det만↓=confidence / raw도↓=forgetting or domain / old유지+real만↓=sim2real.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
from eval_pvnet_heads import load_pvnet_model, collect_manual, load_gt8  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa
from diag2_raw_decode_stages import per_frame, classify, TAU0  # noqa

OUT = os.path.join(ROOT, "data/pallet/eval_results/stage11_16k")
MODELS = {
    "baseline": "weights/challenge0123/final_net_epoch_0060.pth",
    "B0_v3": "weights/stage11_16k_B0_v3/final_net_epoch_0075.pth",
    "B1_addon": "weights/stage11_16k_B1_v3addon/final_net_epoch_0075.pth",
}
VAL = os.path.join(ROOT, "data/pallet/training_data/val")


def run_set(model, dev, frames):
    """frames: list(jp,ip). returns raw_corner_med, det%, mean corner score(det/miss)."""
    raw_errs, dets, sc_det, sc_miss = [], [], [], []
    for jp, ip in frames:
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        ctr = np.array(o["projected_cuboid_centroid"], float)
        r = per_frame(model, dev, ip, gt8, ctr)
        if r is None:
            continue
        raw_errs.append(float(np.median(r["corner_err"])))
        cls, nfin = classify(r)
        det = nfin >= 6; dets.append(det)
        ms = float(np.mean(r["score"][:8]))
        (sc_det if det else sc_miss).append(ms)
    return {"n": len(dets),
            "raw_corner_med": float(np.median(raw_errs)) if raw_errs else float("nan"),
            "det_pct": 100*np.mean(dets) if dets else float("nan"),
            "score_det": float(np.mean(sc_det)) if sc_det else float("nan"),
            "score_miss": float(np.mean(sc_miss)) if sc_miss else float("nan")}


def main():
    import torch, glob
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = [(jp, ip) for _, _, jp, ip in collect_val_frames()] + \
           [(jp, ip) for _, _, jp, ip in collect_manual(0)]
    valjs = sorted(glob.glob(os.path.join(VAL, "*.json")))
    oldsyn = [(jp, jp[:-5]+".png") for jp in valjs if os.path.exists(jp[:-5]+".png")]
    idx = np.linspace(0, len(oldsyn)-1, 200).round().astype(int)
    oldsyn = [oldsyn[int(i)] for i in sorted(set(idx))]

    L = ["DIAG8 — B0/B1 real 검출붕괴 원인 (raw top-1 / score / old-synth)",
         f"real={len(real)}장  old-synth(training_data/val)={len(oldsyn)}장  thr={TAU0}"]
    L.append(f"\n  {'model':<12}{'set':<10}{'N':>4}{'rawCornMed':>11}{'det%':>7}"
             f"{'score(det)':>11}{'score(miss)':>12}")
    for name, w in MODELS.items():
        model, _, _ = load_pvnet_model(w, dev, numVec=0, numSeg=0)
        for sname, fr in (("real", real), ("old-synth", oldsyn)):
            r = run_set(model, dev, fr)
            L.append(f"  {name:<12}{sname:<10}{r['n']:>4}{r['raw_corner_med']:>11.1f}"
                     f"{r['det_pct']:>6.0f}%{r['score_det']:>11.3f}{r['score_miss']:>12.3f}")
    L.append("\n[판정 가이드] raw좌표 baseline급+det만↓ → confidence/threshold(혼합학습 불필요, recalib)")
    L.append("            raw좌표도↓ → forgetting or domain.  old-synth 유지+real만↓ → sim2real transfer")
    txt = "\n".join(L); print(txt)
    open(os.path.join(OUT, "diag8_forgetting.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag8_forgetting.txt")


if __name__ == "__main__":
    main()
