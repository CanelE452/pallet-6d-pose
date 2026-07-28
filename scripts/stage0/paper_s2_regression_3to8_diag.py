"""paper_s2_regression_3to8_diag.py — PAPER_S2 follow-up DIAGNOSIS (NO retrain).

Q: In real filterval, Stage B (DiffPnP3D) improves rear at <3deg but REGRESSES
   corner/honest8 in the 3-8deg transition band vs paper_s1. Split the cause:
     raw heatmap keypoint  vs  PnP-reprojection  vs  transition-band geometry
     vs  front-sacrifice / rear-redistribution.

Reuses paper_s2_real_eval eval-parity harness EXACTLY (each model = own train
preprocess; downstream metric identical). Only 2 models, filterval only.

Per-frame keypoint error SPLIT (both order-free Hungarian, MEAN of matched dists
so raw and pnp are the SAME statistic and directly comparable):
  raw_corner = mean hungarian(pred8 raw-belief-peak, gt8)     [PRE-PnP]
  pnp_corner = mean hungarian(proj_all solve_pose-reproj, gt8) [POST-PnP] (== honest8)
  raw_pnp_disp = median index-aligned ||pred8[i]-proj_all[i]|| [how far PnP pulled raw]
Comparison uses the SAME frame subset (pnp_ok) to isolate the PnP effect.

Usage: conda activate pallet-pose; python scripts/stage0/paper_s2_regression_3to8_diag.py
"""
from __future__ import annotations
import json
import os

import numpy as np

import paper_s2_real_eval as P  # sets sys.path, imports S, E, APNP, eval_frame_squash
from paper_s2_real_eval import S, E, ROOT, THRESH, PAD_ASPECT, N_DET_MIN, eval_frame_squash

import cv2  # noqa: E402  (after P import so path is set)
import torch  # noqa: E402

OUT_DIR = P.OUT_DIR  # existing folder, no new dir
ELEV_BINS = P.ELEV_BINS      # [(-90,3),(3,8),(8,15),(15,90)]
ELEV_LBL = P.ELEV_LBL        # ["<3","3-8","8-15","15+"]
GOOD_PX, GROSS_PX = 10.0, 20.0

MODELS = [
    ("paper_s1",        "weights/paper_s1_maskaux/net_epoch_0065.pth", "aspect"),
    ("paper_s2_stageB", "weights/paper_s2_stageB/net_epoch_0057.pth", "squash"),
]
BASELINE, CHALLENGER = "paper_s1", "paper_s2_stageB"


def _clean(a):
    a = np.array(a, float)
    bad = (a[:, 0] == -1.0) & (a[:, 1] == -1.0)
    a[bad] = np.nan
    return a


def frame_split_errors(row):
    """Return dict with raw_corner, pnp_corner, disp, front, rear, good/gross,
    det, pnp_ok. raw/pnp both order-free mean; disp index-aligned median."""
    pred8 = np.array(row["pred8"], float)
    gt8 = np.array(row["gt8"], float)
    proj = None if row["proj_all"] is None else _clean(row["proj_all"])

    raw_d, _ = E.hungarian(pred8, gt8)
    raw_corner = float(np.mean(raw_d)) if raw_d is not None else np.inf

    pnp_corner = np.inf
    disp = np.inf
    if proj is not None:
        pnp_d, _ = E.hungarian(proj, gt8)
        if pnp_d is not None:
            pnp_corner = float(np.mean(pnp_d))
        # index-aligned raw<->pnp displacement (how far PnP moved each raw corner)
        both = (~np.isnan(pred8[:, 0])) & (~np.isnan(proj[:, 0]))
        if both.sum() >= 1:
            dd = np.linalg.norm(pred8[both] - proj[both], axis=1)
            disp = float(np.median(dd))

    # good/gross on raw corners (order-free), same convention as summarize
    good = gross = ncor = 0
    if raw_d is not None:
        for x in raw_d:
            ncor += 1
            good += int(x < GOOD_PX)
            gross += int(x > GROSS_PX)
    return {
        "fid": row["fid"], "v_geom": row["v_geom"],
        "raw_corner": raw_corner, "pnp_corner": pnp_corner, "disp": disp,
        "front": row["front"], "rear": row["back"],
        "det": row["det"], "pnp_ok": row["pnp_ok"],
        "good": good, "gross": gross, "ncor": ncor,
    }


def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return round(float(np.median(v)), 1) if v else None


def bin_stats(recs):
    """recs = frame_split_errors dicts. Raw/pnp/disp computed over pnp_ok subset
    (comparable); det/pnp% over all."""
    n = len(recs)
    pnp_rows = [r for r in recs if r["pnp_ok"]]
    good = sum(r["good"] for r in recs)
    gross = sum(r["gross"] for r in recs)
    ncor = sum(r["ncor"] for r in recs)
    return {
        "n": n,
        "n_pnp": len(pnp_rows),
        "raw_corner": med([r["raw_corner"] for r in pnp_rows]),
        "pnp_corner": med([r["pnp_corner"] for r in pnp_rows]),
        "disp": med([r["disp"] for r in pnp_rows]),
        "front": med([r["front"] for r in pnp_rows]),
        "rear": med([r["rear"] for r in pnp_rows]),
        "honest8_all": med([r["pnp_corner"] for r in pnp_rows]),
        "det_pct": round(100 * sum(r["det"] for r in recs) / n, 0) if n else None,
        "pnp_pct": round(100 * len(pnp_rows) / n, 0) if n else None,
        "good_pct": round(100 * good / ncor, 1) if ncor else None,
        "gross_pct": round(100 * gross / ncor, 1) if ncor else None,
    }


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fl = S.frames_filterval()
    elev = {fid: S.elev_of(jp) for _, fid, jp, _ in fl}
    by = {}
    for dom, *_ in fl:
        by[dom] = by.get(dom, 0) + 1
    print(f"[set] filterval N={len(fl)} {by}")

    rows = {m: [] for m, _, _ in MODELS}
    for mname, wrel, mode in MODELS:
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for dom, fid, jp, ip in fl:
            if mode == "squash":
                r = eval_frame_squash(mdl, jp, ip, device)
            else:
                r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD_ASPECT)
            if r is not None:
                r["dom"] = dom
                rows[mname].append(r)
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname} ({mode}) -> {len(rows[mname])} frames")

    # per-frame split errors keyed by fid
    split = {m: {r["fid"]: frame_split_errors(r) for r in rows[m]} for m in rows}

    # bin
    binned = {m: {lbl: [] for lbl in ELEV_LBL} for m in rows}
    for m in rows:
        for fid, rec in split[m].items():
            e = elev.get(fid)
            if e is None:
                continue
            for lbl, (lo, hi) in zip(ELEV_LBL, ELEV_BINS):
                if lo <= e < hi:
                    binned[m][lbl].append(rec)
                    break
    stats = {m: {lbl: bin_stats(binned[m][lbl]) for lbl in ELEV_LBL} for m in rows}

    write_report(stats, binned, split, elev, rows)
    montage(rows, split, elev)
    return stats


def write_report(stats, binned, split, elev, rows):
    L = []
    L.append("# PAPER_S2 — 3-8deg regression diagnosis (raw heatmap vs PnP)")
    L.append("")
    L.append("Q: Stage B improves rear at <3deg but regresses corner/honest8 at 3-8deg.")
    L.append("   Is the regression in the RAW heatmap keypoint, or in the PnP reprojection,")
    L.append("   or transition-band geometry, or front-sacrifice/rear-redistribution?")
    L.append("")
    L.append("## Method (eval-parity reused from paper_s2_real_eval)")
    L.append("```")
    L.append("each model = own train preprocess (s1 aspect no-pad / stageB squash);")
    L.append("downstream metric IDENTICAL. raw & pnp both order-free Hungarian MEAN of")
    L.append("matched dists (same statistic -> directly comparable). raw/pnp/disp/front/")
    L.append("rear computed over the SAME pnp_ok frame subset to isolate the PnP effect.")
    L.append("  raw_corner  = mean hungarian(raw belief-peak pred8, gt8)   [PRE-PnP]")
    L.append("  pnp_corner  = mean hungarian(solve_pose reproj proj_all, gt8) [POST-PnP=honest8]")
    L.append("  raw-pnp_disp= median index-aligned ||pred8[i]-proj_all[i]|| [PnP pull]")
    L.append("  front/rear  = split_metrics front(0-3)/back(4-7) order-free median.")
    L.append(f"  dims=(1.1,1.3,0.11) order-free W/D. good<{GOOD_PX} gross>{GROSS_PX}px per-corner.")
    L.append("  elevation = GT pose_transform. NO retrain (diagnosis only).")
    L.append("```")
    L.append("")
    # elev dist
    els = [e for e in elev.values() if e is not None]
    edist = ", ".join(f"{lbl}:{sum(1 for e in els if lo<=e<hi)}"
                      for lbl, (lo, hi) in zip(ELEV_LBL, ELEV_BINS))
    L.append(f"elev dist(deg): {edist}   (total with elev: {len(els)})")
    L.append("")
    L.append("## bin x model  (raw vs pnp split)")
    L.append("```")
    hdr = (f"{'bin':<7}{'model':<17}{'n':>4}{'n_pnp':>6}{'raw_cnr':>9}"
           f"{'pnp_cnr':>9}{'raw-pnp':>9}{'front':>7}{'rear':>7}"
           f"{'honest8':>9}{'gross%':>8}{'pnp%':>6}{'det%':>6}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for lbl in ELEV_LBL:
        for m, _, _ in MODELS:
            s = stats[m][lbl]
            L.append(
                f"{lbl:<7}{m:<17}{s['n']:>4}{s['n_pnp']:>6}"
                f"{str(s['raw_corner']):>9}{str(s['pnp_corner']):>9}"
                f"{str(s['disp']):>9}{str(s['front']):>7}{str(s['rear']):>7}"
                f"{str(s['honest8_all']):>9}{str(s['gross_pct']):>8}"
                f"{str(s['pnp_pct']):>6}{str(s['det_pct']):>6}")
        L.append("")
    L.append("```")
    L.append("")

    # paired deltas for 3-8 (challenger - baseline) on shared pnp_ok frames
    L.append("## 3-8deg PAIRED delta (StageB - s1), shared pnp_ok frames only")
    L.append("```")
    b_map = {r["fid"]: r for r in map(split[BASELINE].get, split[BASELINE])}
    # build fid-aligned on 3-8
    lbl = "3-8"
    b38 = {r["fid"]: r for r in binned[BASELINE][lbl]}
    c38 = {r["fid"]: r for r in binned[CHALLENGER][lbl]}
    shared = [f for f in b38 if f in c38 and b38[f]["pnp_ok"] and c38[f]["pnp_ok"]]
    L.append(f"shared pnp_ok frames in 3-8deg: {len(shared)}")
    for key, lab in [("raw_corner", "raw_corner"), ("pnp_corner", "pnp_corner(honest8)"),
                     ("disp", "raw-pnp_disp"), ("front", "front"), ("rear", "rear")]:
        bv = med([b38[f][key] for f in shared])
        cv = med([c38[f][key] for f in shared])
        dd = None if bv is None or cv is None else round(cv - bv, 2)
        # per-frame paired median delta
        pf = [c38[f][key] - b38[f][key] for f in shared
              if np.isfinite(c38[f][key]) and np.isfinite(b38[f][key])]
        pfm = round(float(np.median(pf)), 2) if pf else None
        L.append(f"  {lab:<22} s1={str(bv):>7}  stageB={str(cv):>7}  "
                 f"d(median-of-med)={str(dd):>7}  paired-median-d={str(pfm):>7}")
    L.append("```")
    L.append("")

    # VERDICT
    L.append("## VERDICT")
    L.append("```")
    s1_38 = stats[BASELINE]["3-8"]
    sb_38 = stats[CHALLENGER]["3-8"]
    s1_lt3 = stats[BASELINE]["<3"]
    sb_lt3 = stats[CHALLENGER]["<3"]

    def dnum(a, b):
        return None if a is None or b is None else round(b - a, 2)
    rawd = dnum(s1_38["raw_corner"], sb_38["raw_corner"])
    pnpd = dnum(s1_38["pnp_corner"], sb_38["pnp_corner"])
    dispd = dnum(s1_38["disp"], sb_38["disp"])
    frontd = dnum(s1_38["front"], sb_38["front"])
    reard = dnum(s1_38["rear"], sb_38["rear"])
    L.append("3-8deg band (StageB - s1, +=worse for error metrics):")
    L.append(f"  raw_corner  d={rawd}   pnp_corner d={pnpd}   raw-pnp_disp d={dispd}")
    L.append(f"  front d={frontd}   rear d={reard}")
    L.append("")
    L.append("<3deg band (for redistribution check):")
    L.append(f"  raw_corner d={dnum(s1_lt3['raw_corner'], sb_lt3['raw_corner'])}"
             f"   front d={dnum(s1_lt3['front'], sb_lt3['front'])}"
             f"   rear d={dnum(s1_lt3['rear'], sb_lt3['rear'])}")
    L.append("")
    # heuristic verdict text
    def big(x):
        return x is not None and abs(x) >= 1.5
    verdict = []
    if rawd is not None and pnpd is not None:
        if rawd >= 1.5 and pnpd >= 1.5:
            verdict.append("RAW+PnP both regress -> data/appearance-driven (raw heatmap worse).")
        elif rawd < 1.0 and pnpd >= 1.5:
            verdict.append("RAW ~flat but PnP regresses -> PnP/reprojection-driven regression.")
        elif rawd >= 1.5 and pnpd < 1.0:
            verdict.append("RAW regresses but PnP absorbs it -> heatmap worse, PnP compensates.")
        else:
            verdict.append(f"RAW d={rawd}, PnP d={pnpd} -> mixed/small, see numbers.")
    if big(dispd) and dispd > 0:
        verdict.append(f"raw-pnp_disp larger for StageB (d={dispd}) -> PnP fights raw more "
                       f"in transition band (geometry mismatch).")
    if frontd is not None and reard is not None and frontd > 0.5 and reard < -0.5:
        verdict.append(f"front worse (d={frontd}) while rear better (d={reard}) "
                       f"-> rear-redistribution / front-sacrifice pattern.")
    for v in verdict:
        L.append("* " + v)
    L.append("```")
    L.append("")
    L.append("CAVEAT: small samples per bin (see n / n_pnp). filterval domain-mixed low-angle.")
    L.append("ckpt = synthetic-val best (NOT real-selected). Diagnosis only, no retrain.")

    path = os.path.join(OUT_DIR, "regression_3to8_diagnosis.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {path}")


def draw_triplet(row, tag, elev):
    """raw pred8 = BLUE, PnP proj_all = RED, GT = GREEN edges+dots."""
    img = cv2.imread(row["ip"])
    if img is None:
        return None
    gt8 = np.array(row["gt8"], float)
    pr8 = np.array(row["pred8"], float)
    proj = None if row["proj_all"] is None else _clean(row["proj_all"])
    EDG = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in EDG:
        cv2.line(img, tuple(gt8[a].astype(int)), tuple(gt8[b].astype(int)),
                 (60, 200, 60), 1)
    if proj is not None:
        for a, b in EDG:
            if np.isnan(proj[a, 0]) or np.isnan(proj[b, 0]):
                continue
            cv2.line(img, tuple(proj[a].astype(int)), tuple(proj[b].astype(int)),
                     (0, 0, 220), 1)
    for i in range(8):
        if not np.isnan(pr8[i, 0]):
            cv2.circle(img, (int(pr8[i, 0]), int(pr8[i, 1])), 4, (255, 60, 0), -1)
        if proj is not None and not np.isnan(proj[i, 0]):
            cv2.circle(img, (int(proj[i, 0]), int(proj[i, 1])), 3, (0, 0, 220), -1)
    rear = row["back"] if np.isfinite(row["back"]) else -1
    h8 = row["pnp_honest8"] if np.isfinite(row["pnp_honest8"]) else -1
    txt = f"{tag} el={elev:.0f} V={row['v_geom']} rear={rear:.0f} h8={h8:.0f}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(img, "raw=blue proj=red GT=green", (4, img.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def montage(rows, split, elev):
    """3-8deg regression cases: StageB honest8 worse than s1 by margin."""
    ovdir = os.path.join(OUT_DIR, "regression_3to8_overlays")
    os.makedirs(ovdir, exist_ok=True)
    b_rows = {r["fid"]: r for r in rows[BASELINE]}
    c_rows = {r["fid"]: r for r in rows[CHALLENGER]}
    picks = []
    for fid in c_rows:
        e = elev.get(fid)
        if e is None or not (3 <= e < 8):
            continue
        if fid not in b_rows:
            continue
        rb, rc = split[BASELINE][fid], split[CHALLENGER][fid]
        if not (rb["pnp_ok"] and rc["pnp_ok"]):
            continue
        if not (np.isfinite(rb["pnp_corner"]) and np.isfinite(rc["pnp_corner"])):
            continue
        picks.append((rc["pnp_corner"] - rb["pnp_corner"], e, fid))
    picks.sort(reverse=True)  # most-regressed first
    n = 0
    for delta, e, fid in picks[:8]:
        a = draw_triplet(b_rows[fid], "s1", e)
        b = draw_triplet(c_rows[fid], "stageB", e)
        if a is None or b is None:
            continue
        h = max(a.shape[0], b.shape[0])
        canvas = np.zeros((h, a.shape[1] + b.shape[1] + 4, 3), np.uint8)
        canvas[:a.shape[0], :a.shape[1]] = a
        canvas[:b.shape[0], a.shape[1] + 4:] = b
        cv2.imwrite(os.path.join(ovdir, f"regr_dh8{delta:+.0f}_el{e:.0f}_{fid}.jpg"),
                    canvas)
        n += 1
    print(f"[overlays] {n} 3-8deg regression triplets -> {ovdir}")


if __name__ == "__main__":
    run()
