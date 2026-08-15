"""s2_fullpool_full7_filter.py — full-pool 추론 jsonl 에 FULL7(f3 포함) 필터 적용.

입력:  data/pallet/results/paper_s2/paper_s2_fullpool_infer/{domain}.jsonl  (s2_fullpool_infer.py 산출)
필터:  FULL7 = f1_peak & f2_peak_ratio & f3_flip & f4_tta_stab & f5_rear_conf & f6_frsep & f7_posdepth
       (M.apply_filter + T.TAU, allfilters_5domains 와 동일 임계)
출력:  data/pallet/results/paper_s2/paper_s2_fullpool_full7/
         {domain}/pass/*.jpg   (pass 프레임 오버레이: pred8 cuboid + centroid)
         pass_manifest.tsv     (domain\tsession\tfid)
         summary.md            (도메인별 pass 수)

Usage: conda activate pallet-pose; python -u scripts/stage0/s2_fullpool_full7_filter.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
import s2_fullpool_infer as INF           # noqa: E402  (DOMAINS 재사용)
import cv2                                # noqa: E402
from annotate_draw import CUBOID_EDGES, KP_COLORS  # noqa: E402

M, TAU, N_DET_MIN = T.M, T.TAU, T.N_DET_MIN
FULL7 = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
         "f5_rear_conf", "f6_frsep", "f7_posdepth"]
ENV = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}
INFER_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2/paper_s2_fullpool_infer")
OUT_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2/paper_s2_fullpool_full7")

# session basename -> rgb dir (오버레이용 원본 이미지 경로)
SESS2DIR = {}
for _dom, _seqs in INF.DOMAINS.items():
    for _seq in _seqs:
        SESS2DIR[os.path.basename(_seq)] = _seq


def passes(rec):
    if rec["n_det"] < N_DET_MIN:
        return False
    row = {"n_det": rec["n_det"], "scores": rec["scores"],
           "f7_posdepth": rec["f7_posdepth"],
           "pred_sr": rec["pred_sr"], "pred_asp": rec["pred_asp"]}
    return all(M.apply_filter(f, row, TAU.get(f), ENV) for f in FULL7)


def draw_overlay(img, pred8, pred_c):
    vis = img.copy()
    pts = [None if p is None else (int(round(p[0])), int(round(p[1]))) for p in pred8]
    for a, b in CUBOID_EDGES:
        if a < 8 and b < 8 and pts[a] is not None and pts[b] is not None:
            cv2.line(vis, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p is not None:
            cv2.circle(vis, p, 4, KP_COLORS[i], -1)
            cv2.putText(vis, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 1, cv2.LINE_AA)
    if pred_c is not None:
        c = (int(round(pred_c[0])), int(round(pred_c[1])))
        cv2.circle(vis, c, 5, (0, 0, 255), -1)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    summary = []
    for jf in sorted(glob.glob(os.path.join(INFER_DIR, "*.jsonl"))):
        dom = os.path.splitext(os.path.basename(jf))[0]
        pass_dir = os.path.join(OUT_DIR, dom, "pass")
        os.makedirs(pass_dir, exist_ok=True)
        n_tot = n_det6 = n_pass = 0
        with open(jf) as f:
            for line in f:
                rec = json.loads(line)
                n_tot += 1
                if rec["n_det"] >= N_DET_MIN:
                    n_det6 += 1
                if not passes(rec):
                    continue
                n_pass += 1
                manifest.append(f"{dom}\t{rec['session']}\t{rec['fid']}")
                seq = SESS2DIR.get(rec["session"])
                ip = os.path.join(seq, "rgb", rec["fid"] + ".png") if seq else None
                if ip and os.path.isfile(ip):
                    img = cv2.imread(ip)
                    if img is not None:
                        vis = draw_overlay(img, rec["pred8"], rec["pred_c"])
                        cv2.imwrite(os.path.join(pass_dir, rec["fid"] + ".jpg"), vis)
        print(f"[{dom}] frames={n_tot} det>=6={n_det6} FULL7-pass={n_pass}", flush=True)
        summary.append((dom, n_tot, n_det6, n_pass))

    L = ["# s2 diffpnp full-pool + FULL7(f3 포함) 필터 결과",
         "",
         "- weights: paper_s2_stageB net_epoch_0057. FULL7 = f1&f2&f3&f4&f5&f6&f7.",
         f"- TAU: {{k: TAU.get(k) for k in FULL7[:6]}} = "
         f"{{'f1':0.5,'f2':1.5,'f3':10.0,'f4':5.0,'f5':0.5,'f6':0.06}}, f7=posdepth(bool).",
         "- pass 오버레이: {domain}/pass/*.jpg (pred cuboid). 원본은 raw_data.",
         "",
         "```",
         f"{'domain':<14}{'frames':>8}{'det>=6':>9}{'FULL7 pass':>12}{'pass%(det)':>11}",
         "-" * 55]
    for dom, n, d, p in summary:
        L.append(f"{dom:<14}{n:>8}{d:>9}{p:>12}{(100*p/d if d else 0):>10.1f}%")
    L.append("-" * 55)
    tn = sum(n for _, n, _, _ in summary)
    td = sum(d for _, _, d, _ in summary)
    tp = sum(p for _, _, _, p in summary)
    L.append(f"{'ALL':<14}{tn:>8}{td:>9}{tp:>12}{(100*tp/td if td else 0):>10.1f}%")
    L.append("```")
    with open(os.path.join(OUT_DIR, "pass_manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\n[save] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
