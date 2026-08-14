"""filter_domain_overlay.py — representative overlays per domain.

For each domain picks:
  (a) pass+good   : combo (or diag if combo empty) passed AND good
  (b) caught-bad  : a structural filter (diag) correctly REJECTED a gross(>20px) frame
  (c) missed-bad  : diag PASSED but frame is gross(>20px) (filter slipped)
Draws 9 keypoints (pred) + GT cuboid + per-filter pass/fail badge.
Reads _full_{tag}.json produced by filter_domain_analysis.py.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
KP_COL = (0, 255, 255)
GT_COL = (0, 200, 0)
FILTERS = ["fullkp", "diag", "ratio", "ransac_loo", "combo"]


def draw(rec, thr=20.0):
    img = cv2.imread(rec["img"])
    if img is None:
        return None
    gt8 = np.array(rec["gt8"], float)
    for a, b in EDGES:
        cv2.line(img, tuple(gt8[a].astype(int)), tuple(gt8[b].astype(int)),
                 GT_COL, 1, cv2.LINE_AA)
    kp = rec["kp"]
    pts = [None if p is None else np.array(p, float) for p in kp]
    for a, b in EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     KP_COL, 2, cv2.LINE_AA)
    for i in range(9):
        if pts[i] is not None:
            c = (0, 0, 255) if i == 8 else KP_COL
            cv2.circle(img, tuple(pts[i].astype(int)), 4, c, -1)
            cv2.putText(img, str(i), tuple((pts[i] + 4).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    # badge
    y = 18
    mm = rec.get("mean_match_px")
    cv2.putText(img, f"reproj={mm}px good={rec['good']} ndet={rec['n_detected']}",
                (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for fid in FILTERS:
        y += 16
        ok = rec["filters"][fid]
        col = (0, 220, 0) if ok else (0, 0, 230)
        cv2.putText(img, f"{fid}: {'PASS' if ok else 'fail'}", (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img


def pick(rows, kind):
    gross = [r for r in rows if r["mean_match_px"] is not None
             and r["mean_match_px"] > 20]
    if kind == "good":
        c = [r for r in rows if r["filters"]["combo"] and r["good"]]
        if not c:
            c = [r for r in rows if r["filters"]["diag"] and r["good"]]
        if not c:
            c = [r for r in rows if r["filters"]["fullkp"] and r["good"]]
        return c[:3]
    if kind == "caught":  # diag rejected a gross frame
        return [r for r in gross if not r["filters"]["diag"]][:3]
    if kind == "missed":  # diag passed but gross
        return [r for r in gross if r["filters"]["diag"]][:3]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--in_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "filter_domain_analysis"))
    args = ap.parse_args()
    full = json.load(open(os.path.join(args.in_dir, f"_full_{args.tag}.json")))
    out_dir = os.path.join(args.in_dir, f"overlays_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for dom, rows in full.items():
        for kind in ("good", "caught", "missed"):
            for j, rec in enumerate(pick(rows, kind)):
                img = draw(rec)
                if img is None:
                    continue
                fn = os.path.join(out_dir, f"{dom}_{kind}_{j}_{rec['frame']}.jpg")
                cv2.imwrite(fn, img)
                saved.append(fn)
    print(f"[save] {len(saved)} overlays -> {out_dir}")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    main()
