"""Overlay representative passed PLs for the best-viable 9kp combo per domain.
GT cuboid = magenta, PL(pred) cuboid = cyan, centroid = star.
Hungarian-matched corner pairs joined by thin yellow lines (shows 9kp registration).
Title shows the 9kp mean error.  Inference-free (reuses _full_s2.json).
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from filter_pr_camfacing import filt_diag, filt_ratio, filt_fullkp, filt_topbot  # noqa

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
# best viable combo per domain (from filter_combo_9kp run)
BEST = {"indoor": ["ratio"], "outside": ["diag"], "night": ["diag", "ratio"]}
PRED_C = (255, 255, 0)   # cyan
GT_C = (255, 0, 255)     # magenta


def assign(kp, gt8):
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    pairs, dists = [], []
    if valid.sum() < 6:
        return pairs, float("nan")
    idx = np.where(valid)[0]
    P, G = pred8[valid], np.asarray(gt8, float)
    cost = np.linalg.norm(P[:, None] - G[None], axis=2)
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        pairs.append((P[r], G[c]))
        dists.append(cost[r, c])
    ce = None
    if kp[8] is not None:
        gc = G.mean(0)
        ce = float(np.linalg.norm(np.asarray(kp[8], float) - gc))
        pairs.append((np.asarray(kp[8], float), gc))
    all9 = dists + ([ce] if ce is not None else [])
    return pairs, float(np.mean(all9))


def draw(img, pts8, color):
    pts = [None if p is None or (isinstance(p, float) and np.isnan(p)) or
           (hasattr(p, "__len__") and np.isnan(p[0]))
           else (int(round(p[0])), int(round(p[1]))) for p in pts8]
    for a, b in EDGES:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)


def main():
    fp = os.path.join(ROOT, "data/pallet/eval_results/filter_domain_analysis/_full_s2.json")
    data = json.load(open(fp))
    ex = set()
    exf = os.path.join(ROOT, "data/_eval_sets/_exclude.txt")
    for ln in open(exf):
        ln = ln.split("#")[0].strip()
        if ln:
            ex.add(ln)
    outdir = os.path.join(ROOT, "data/pallet/eval_results/filter_combo_9kp/overlays")
    os.makedirs(outdir, exist_ok=True)
    SINGLE = {"diag": filt_diag, "ratio": filt_ratio,
              "fullkp": filt_fullkp, "topbot": filt_topbot}
    saved = []
    for dom, combo in BEST.items():
        rows = [r for r in data[dom] if str(r["frame"]) not in ex
                and r["mean_match_px"] is not None]
        passed = []
        for r in rows:
            kp = [tuple(p) if p is not None else None for p in r["kp"]]
            ok = all(r["filters"].get("ransac_loo", False) if f == "ransac_loo"
                     else SINGLE[f](kp)[0] for f in combo)
            if not ok:
                continue
            pairs, e9 = assign(r["kp"], r["gt8"])
            if np.isfinite(e9):
                passed.append((e9, r, pairs))
        passed.sort(key=lambda x: x[0])
        pick = passed[:3]  # 3 best-aligned representatives
        for rank, (e9, r, pairs) in enumerate(pick):
            img = cv2.imread(r["img"])
            if img is None:
                continue
            draw(img, r["gt8"], GT_C)
            draw(img, r["kp"][:8], PRED_C)
            if r["kp"][8] is not None:
                c = (int(r["kp"][8][0]), int(r["kp"][8][1]))
                cv2.drawMarker(img, c, PRED_C, cv2.MARKER_STAR, 16, 2)
            for pp, gg in pairs:
                if not (np.isnan(pp[0]) or np.isnan(gg[0])):
                    cv2.line(img, (int(pp[0]), int(pp[1])),
                             (int(gg[0]), int(gg[1])), (0, 255, 255), 1, cv2.LINE_AA)
            txt = f"{dom} {'+'.join(combo)}  9kp={e9:.1f}px"
            cv2.rectangle(img, (0, 0), (470, 56), (0, 0, 0), -1)
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "GT=magenta  PL=cyan(star=ctr)", (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            op = os.path.join(outdir, f"{dom}_{'_'.join(combo)}_{rank}_{e9:.0f}px.jpg")
            cv2.imwrite(op, img)
            saved.append(op)
    print("\n".join(saved))


if __name__ == "__main__":
    main()
