"""paper_s2_testset17_passfail_withf3_overlays.py — same GT-free pass/fail/underdet
overlay split as paper_s2_filterval_passfail_withf3_overlays (f3 INCLUDED), but on
the hand-anno test set (cad11 + noapril6 = 17, stage22 manifest) instead of filterval.

Reuses the f3-included overlay module VERBATIM (O.accept / O.pnp_cuboid /
O.save_overlay / O.PASS_FILTERS / O.OUT_ROOT); importing it also applies the
(W-1)-x flip override (via its `import paper_s2_filterval_9filters`) used by f3.
Only the frame source (stage22 manifest) and domains (cad, noapril) change.
Adds cad/noapril subfolders under the existing filterval_passfail_withf3 root.

pass = f1 & f2 & f3 & f4 & f5 & f6 & f7 (f8/f9 excluded). underdet = n_det<6.
NO GT used for filtering or overlay (json read only for K).

Usage: conda activate pallet-pose;
       python scripts/stage0/paper_s2_testset17_passfail_withf3_overlays.py
"""
from __future__ import annotations
import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_passfail_withf3_overlays as O  # noqa: E402 (f3-incl; applies flip)
import paper_s2_testset17_9filters as T                  # noqa: E402
import cv2                                                # noqa: E402
import torch                                              # noqa: E402

DOMAINS = ["cad", "noapril"]
N_DET_MIN = O.N_DET_MIN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dm in DOMAINS:
        for v in ("pass", "fail", "underdet"):
            os.makedirs(os.path.join(O.OUT_ROOT, dm, v), exist_ok=True)

    frames = T.read_manifest(T.MANIFEST)
    print(f"[testset17] {len(frames)} frames; dims={O.DIMS}; weights=Stage B ep0057; "
          f"squash-parity; flip=(W-1)-x; pass=AND{O.PASS_FILTERS}")
    model = O.E.load_model(O.WEIGHTS, device)

    counts = {dm: {"pass": 0, "fail": 0, "underdet": 0} for dm in DOMAINS}
    saved = 0
    for dom, fid, jp, ip in frames:
        r = T.build_rec(model, dom, fid, jp, ip, device)
        if r is None:
            print(f"  {dom:<8}{fid}  build_rec=None (skip)")
            continue
        img = cv2.imread(ip)
        _, pred8, pred_c, _, _ = T.infer_squash(model, img, device)
        n_det = r["n_det"]

        if n_det < N_DET_MIN:
            verdict, sub, failed, proj = "UNDERDET", "underdet", [], None
        else:
            ok, failed = O.accept(r)
            verdict, sub = ("PASS", "pass") if ok else ("FAIL", "fail")
            K = O.E.K_from_json(json.load(open(jp)))
            proj = O.pnp_cuboid(pred8, pred_c, K, img.shape)
        counts[dom][sub] += 1
        path = os.path.join(O.OUT_ROOT, dom, sub, f"{fid}.jpg")
        if O.save_overlay(ip, pred8, pred_c, proj, verdict, fid, n_det, failed, path):
            saved += 1
        cm = f"{r['corner_med']:.1f}" if r["corner_med"] is not None else "-"
        print(f"  {dom:<8}{fid}  det={n_det} {verdict:<8} cm={cm} "
              f"fail={failed if verdict=='FAIL' else ''}")

    print("\n=== per-domain counts (WITH f3) ===")
    print(f"{'domain':<10}{'pass':>6}{'fail':>6}{'underdet':>10}{'total':>7}")
    print("-" * 39)
    tot = {"pass": 0, "fail": 0, "underdet": 0}
    for dm in DOMAINS:
        c = counts[dm]
        n = c["pass"] + c["fail"] + c["underdet"]
        for k in tot:
            tot[k] += c[k]
        print(f"{dm:<10}{c['pass']:>6}{c['fail']:>6}{c['underdet']:>10}{n:>7}")
    print("-" * 39)
    ntot = tot["pass"] + tot["fail"] + tot["underdet"]
    print(f"{'ALL':<10}{tot['pass']:>6}{tot['fail']:>6}{tot['underdet']:>10}{ntot:>7}")
    print(f"\n[save] {saved} overlays -> {O.OUT_ROOT} (cad, noapril)")


if __name__ == "__main__":
    main()
