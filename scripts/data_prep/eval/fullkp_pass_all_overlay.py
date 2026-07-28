"""fullkp_pass_all_overlay.py — overlay EVERY fullkp-pass frame per domain.

Reuses filter_domain_analysis/_full_s2.json (inference-free, ft_s2 — same basis as
diag_pass_all_overlay.py for consistency).
fullkp pass = filt_fullkp(kp)[0]  (all 9 keypoints detected).
Counts match filter_combo_9kp.py fullkp column: indoor 32 / outside 37 / night 29.

Per frame:
  - predicted 9 keypoints: GREEN dots + GREEN index 0..8 (cyan cuboid edges)
  - GT projected_cuboid 8 corners: MAGENTA cuboid
  - top badge: order-free 9kp reproj (px) + good(<10px) flag
Saves EVERY pass frame as an individual jpg (primary) + a per-domain contact sheet.
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from filter_pr_camfacing import filt_fullkp  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
GOOD_PX = 10.0
DOMAINS = ["indoor", "outside", "night"]


def nine_kp_err(kp, gt8):
    """order-free Hungarian 8 corners + centroid vs GT center. returns (e9, d8, ce)."""
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    d8 = np.full(8, np.nan)
    G = np.asarray(gt8, float)
    if valid.sum() >= 6:
        idx = np.where(valid)[0]
        P = pred8[valid]
        cost = np.linalg.norm(P[:, None] - G[None], axis=2)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            d8[idx[r]] = cost[r, c]
    ce = np.nan
    if kp[8] is not None:
        ce = float(np.linalg.norm(np.asarray(kp[8], float) - G.mean(0)))
    all9 = np.concatenate([d8, [ce]])
    e9 = float(np.nanmean(all9)) if not np.all(np.isnan(all9)) else float("nan")
    return e9, d8, ce


def draw_cuboid(img, pts8, color, thick=2):
    pts = []
    for p in pts8:
        if p is None or np.isnan(np.asarray(p, float)[0]):
            pts.append(None)
        else:
            pts.append((int(round(p[0])), int(round(p[1]))))
    for a, b in EDGES:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)
    return pts


def overlay(rec):
    img = cv2.imread(rec["img"])
    if img is None:
        return None, None
    e9, d8, ce = nine_kp_err(rec["kp"], rec["gt8"])
    # GT cuboid (magenta)
    draw_cuboid(img, rec["gt8"], MAGENTA, 2)
    # pred cuboid edges (cyan) so structure is visible, then green numbered pts
    draw_cuboid(img, rec["kp"][:8], CYAN, 1)
    for i in range(9):
        p = rec["kp"][i]
        if p is None:
            continue
        c = (int(round(p[0])), int(round(p[1])))
        cv2.circle(img, c, 4, GREEN, -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (c[0] + 4, c[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
    good = np.isfinite(e9) and e9 < GOOD_PX
    h, w = img.shape[:2]
    bw = min(w, 520)
    cv2.rectangle(img, (0, 0), (bw, 54), (0, 0, 0), -1)
    tag = "GOOD" if good else "BAD"
    tcol = (0, 255, 0) if good else (0, 0, 255)
    cv2.putText(img, f"9kp reproj={e9:.1f}px  [{tag}]", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcol, 2, cv2.LINE_AA)
    cv2.putText(img, "GT=magenta  pred=green 0-8", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img, (e9, good)


def contact_sheet(imgs, cols, cell=480):
    n = len(imgs)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        h, w = im.shape[:2]
        s = cell / max(h, w)
        rim = cv2.resize(im, (int(w * s), int(h * s)))
        rh, rw = rim.shape[:2]
        y0 = r * cell + (cell - rh) // 2
        x0 = c * cell + (cell - rw) // 2
        sheet[y0:y0 + rh, x0:x0 + rw] = rim
    return sheet


def main():
    fp = os.path.join(ROOT, "data/pallet/eval_results/filter_domain_analysis/_full_s2.json")
    data = json.load(open(fp))
    ex = set()
    for ln in open(os.path.join(ROOT, "data/_eval_sets/_exclude.txt")):
        ln = ln.split("#")[0].strip()
        if ln:
            ex.add(ln)
    base = os.path.join(ROOT, "data/pallet/eval_results/filter_combo_9kp/fullkp_pass_all")
    summary = {}
    for dom in DOMAINS:
        outdir = os.path.join(base, dom)
        os.makedirs(outdir, exist_ok=True)
        rows = [r for r in data[dom]
                if str(r["frame"]) not in ex and r["mean_match_px"] is not None]
        passed = []
        for r in rows:
            kp = [tuple(p) if p is not None else None for p in r["kp"]]
            if filt_fullkp(kp)[0]:
                passed.append(r)
        passed.sort(key=lambda r: nine_kp_err(r["kp"], r["gt8"])[0])
        imgs, n_good, n_bad = [], 0, 0
        for rank, r in enumerate(passed):
            img, info = overlay(r)
            if img is None:
                continue
            e9, good = info
            n_good += int(good)
            n_bad += int(not good)
            fn = os.path.join(outdir, f"{rank:02d}_{r['frame']}_{e9:.0f}px.jpg")
            cv2.imwrite(fn, img)
            imgs.append(img)
        cols = 3 if len(imgs) <= 9 else 6
        sheet = contact_sheet(imgs, cols) if imgs else None
        sheet_path = os.path.join(base, f"{dom}_contact_sheet.jpg")
        if sheet is not None:
            cv2.imwrite(sheet_path, sheet)
        summary[dom] = {"n": len(imgs), "good": n_good, "bad": n_bad,
                        "dir": outdir, "sheet": sheet_path}
        print(f"[{dom}] fullkp-pass={len(imgs)}  good(<{GOOD_PX:.0f}px)={n_good}"
              f"  bad={n_bad}  dir={outdir}")
    json.dump(summary, open(os.path.join(base, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
