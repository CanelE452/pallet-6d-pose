"""stage0/pl_sanity_overlay.py — random PL-pass overlay dump for eyeball QA.

CPU-only. Picks a random 20–30 of the PASSED pseudo-labels (those satisfying a
gate built from the recorded scores) and draws the cuboid overlay so a human can
spot PL that fired on a non-pallet / hallucinated detection.

Reuses dump_pl_overlay.overlay (same edge/corner drawing as the existing Phase-1
PL overlays — no new viz geometry).

Gate (on the scores extract_pl_v1 already wrote into each PL json):
  n_detected >= 6  AND diag_pass  AND (flip_score <= flip_tau or flip off).

Usage:
  python scripts/stage0/filter_pl/pl_sanity_overlay.py \
      --pl_dir data/pallet/pl/stage0_paper_base \
      --out_dir data/pallet/eval_results/stage0_pl_sanity --n 30 --flip_tau 10
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import random
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))

sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
from dump_pl_overlay import overlay  # noqa: E402


def passes(scores, flip_tau):
    if scores.get("n_detected", 0) < 6:
        return False
    if not scores.get("diag_pass", False):
        return False
    fs = scores.get("flip_score")
    if flip_tau is not None and fs is not None and fs > flip_tau:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl_dir", required=True,
                    help="extract_pl_v1.py output dir")
    ap.add_argument("--out_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "stage0_pl_sanity"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--flip_tau", type=float, default=10.0,
                    help="flip cutoff for the gate; <0 disables flip in gate")
    ap.add_argument("--no_gate", action="store_true",
                    help="sample from ALL dumped PL, not just gate-passing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flip_tau = None if args.flip_tau < 0 else args.flip_tau
    jsons = [j for j in sorted(glob.glob(os.path.join(args.pl_dir, "*.json")))
             if not os.path.basename(j).startswith("_")]
    cand = []
    for jp in jsons:
        d = json.load(open(jp))
        scores = d.get("filter_scores", {})
        if args.no_gate or passes(scores, flip_tau):
            cand.append(jp)
    print(f"[sanity] {len(cand)} / {len(jsons)} PL "
          f"{'(all)' if args.no_gate else 'pass gate'}; sampling {args.n}")
    if not cand:
        print("  nothing to sample.")
        return

    random.seed(args.seed)
    pick = random.sample(cand, min(args.n, len(cand)))
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for jp in pick:
        base = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(args.pl_dir, base + ".png")
        if not os.path.exists(ip):
            continue
        d = json.load(open(jp))
        kps = d["objects"][0]["projected_cuboid"]
        img = cv2.imread(ip)
        if img is None:
            continue
        ov = overlay(img, kps)
        sc = d.get("filter_scores", {})
        tag = (f"det{sc.get('n_detected','?')}"
               f"_diag{int(sc.get('diag_pass', False))}"
               f"_flip{sc.get('flip_score','NA')}")
        cv2.imwrite(os.path.join(args.out_dir, f"{base}__{tag}.jpg"), ov)
        n += 1
    print(f"[save] {n} overlays -> {args.out_dir}")
    print("  EYEBALL: discard any PL whose cuboid is not on a real pallet.")


if __name__ == "__main__":
    main()
