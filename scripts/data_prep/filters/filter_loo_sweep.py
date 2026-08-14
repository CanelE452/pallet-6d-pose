"""filter_loo_sweep.py — ransac_loo reproj-threshold sweep + diag/fullkp each.

Purpose: ransac_loo purity (good%) is near-perfect but quantity dies (N~1-3).
Sweep the RANSAC-consensus reproj threshold (and LOO tau) to trade purity for
quantity, and find a sweet spot that keeps full-9kp purity high while N grows.

Reads cached predictions (kp 9pts + gt8) from
  data/pallet/eval_results/filter_domain_analysis/_full_{tag}.json
Re-reads per-frame GT json (located via cached 'img' path) for camera K and
per-frame dimensions_m (avoids hardcoded W/D swap).

Metric (same as filter_combo_9kp.py, order-free):
  - 8 predicted corners Hungarian-matched to GT 8 corners -> 8 dists
  - centroid(idx8) vs GT cuboid center (mean of 8 GT corners) -> 1 dist
  - 9kp_err = mean over available of those 9
  good = 9kp_err < good_px (default 10).

ransac_loo(tau_px, tau_loo): RANSAC subset consensus (n_iter=50, subset=5,
consensus tau=tau_px, c>=6) AND leave-one-out PnP normalized-median < tau_loo.

Excludes frames in data/_eval_sets/_exclude.txt.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

from filter_pr_camfacing import (  # noqa: E402
    canonical_kp3d, filt_diag, filt_fullkp,
)

DOMAINS = ["indoor", "outside", "night"]

# where to find the GT json for K/dims, keyed by domain.
GT_DIR = {
    "indoor": os.path.join(ROOT, "data", "pallet", "raw_data",
                           "capture0403middle", "gt_final"),
    "outside": os.path.join(ROOT, "data", "_eval_sets", "outside_combined"),
    "night": os.path.join(ROOT, "data", "_eval_sets", "night_combined"),
}


# ── order-free 9kp metric (matches filter_combo_9kp) ──────────────────
def nine_kp_err(kp, gt8):
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    d8 = np.full(8, np.nan)
    if valid.sum() >= 6:
        P = pred8[valid]
        G = np.asarray(gt8, float)
        cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(cost)
        iv = np.where(valid)[0]
        for r, c in zip(ri, ci):
            d8[iv[r]] = cost[r, c]
    ce = np.nan
    if kp[8] is not None:
        gc = np.asarray(gt8, float).mean(axis=0)
        ce = float(np.linalg.norm(np.asarray(kp[8], float) - gc))
    all9 = np.concatenate([d8, [ce]])
    if np.all(np.isnan(all9)):
        return None
    return float(np.nanmean(all9))


# ── RANSAC consensus + LOO at configurable thresholds ─────────────────
def ransac_consensus(kp, kp3d, K, dist, tau, n_iter=50, subset=5, seed=0):
    det = [i for i in range(8) if kp[i] is not None]
    if len(det) < subset:
        return 0, None, None
    d2 = np.array([[float(kp[i][0]), float(kp[i][1])] for i in det])
    d3 = kp3d[det].astype(np.float64)
    rng = np.random.default_rng(seed)
    best_c, best_R, best_t = -1, None, None
    for _ in range(n_iter):
        sel = (np.arange(len(det)) if len(det) == subset
               else rng.choice(len(det), subset, replace=False))
        try:
            ok, rvec, tvec = cv2.solvePnP(d3[sel], d2[sel], K, dist,
                                          flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            continue
        if not ok or float(tvec[2, 0]) < 0:
            continue
        proj, _ = cv2.projectPoints(d3, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - d2, axis=1)
        c = int((err < tau).sum())
        if c > best_c:
            best_c = c
            R, _ = cv2.Rodrigues(rvec)
            best_R, best_t = R, tvec.flatten()
    return max(best_c, 0), best_R, best_t


def loo_stability(kp, kp3d, K, dist, R, t, tau_loo, min_pts=5):
    det = [i for i in range(8) if kp[i] is not None]
    if len(det) < min_pts:
        return False
    d2 = np.array([[float(kp[i][0]), float(kp[i][1])] for i in det])
    d3 = kp3d[det].astype(np.float64)
    rvec0, _ = cv2.Rodrigues(R)
    proj_all, _ = cv2.projectPoints(kp3d[:8], rvec0, t.reshape(3, 1), K, dist)
    pa = proj_all.reshape(-1, 2)
    diag = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            diag = max(diag, np.linalg.norm(pa[i] - pa[j]))
    if diag < 1e-6:
        return False
    errs = []
    for li in range(len(det)):
        mask = [m for m in range(len(det)) if m != li]
        if len(mask) < 4:
            continue
        ok, rv, tv = cv2.solvePnP(d3[mask], d2[mask], K, dist,
                                  flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        pr, _ = cv2.projectPoints(d3[li].reshape(1, 3), rv, tv, K, dist)
        errs.append(np.linalg.norm(pr.reshape(2) - d2[li]))
    if not errs:
        return False
    return (np.median(errs) / diag) < tau_loo


def load_exclude():
    fp = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
    ex = set()
    if os.path.exists(fp):
        for ln in open(fp):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def gt_K_dims(dom, frame):
    jp = os.path.join(GT_DIR[dom], f"{frame}.json")
    g = json.load(open(jp))
    o = g["objects"][0]
    dm = o.get("dimensions_m", {"width": 1.3, "depth": 1.1, "height": 0.11})
    intr = g["camera_data"]["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    return K, dm


def stats(errs):
    """errs = list of 9kp_err for passed frames."""
    if not errs:
        return 0, None, None
    a = np.array(errs)
    n = len(a)
    good = int(np.sum(a < GOOD_PX))
    return n, round(100 * good / n, 1), round(float(np.median(a)), 1)


GOOD_PX = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paper_base")
    ap.add_argument("--good_px", type=float, default=10.0)
    ap.add_argument("--taus", default="3,5,8,10,12,15")
    ap.add_argument("--loo_taus", default="0.05")
    ap.add_argument("--ransac_c", type=int, default=6)
    args = ap.parse_args()
    global GOOD_PX
    GOOD_PX = args.good_px

    taus = [float(x) for x in args.taus.split(",")]
    loo_taus = [float(x) for x in args.loo_taus.split(",")]

    fp = os.path.join(ROOT, "data", "pallet", "eval_results",
                      "filter_domain_analysis", f"_full_{args.tag}.json")
    data = json.load(open(fp))
    exclude = load_exclude()

    # Build per-frame: kp(9), gt8, 9kp_err, K, dims, kp3d. Cache PnP-able rows.
    rows = {d: [] for d in DOMAINS}
    for dom in DOMAINS:
        for r in data[dom]:
            if str(r["frame"]) in exclude:
                continue
            kp = [tuple(p) if p is not None else None for p in r["kp"]]
            e9 = nine_kp_err(kp, r["gt8"])
            if e9 is None:
                continue  # not detectable (<6 corners)
            K, dm = gt_K_dims(dom, r["frame"])
            kp3d = canonical_kp3d(dm["width"], dm["depth"], dm["height"])
            rows[dom].append({"kp": kp, "gt8": r["gt8"], "e9": e9,
                              "K": K, "kp3d": kp3d})
        print(f"[{dom}] detectable(>=6 corners, 9kp-able) = {len(rows[dom])}")

    dist = np.zeros((5, 1))

    # ── ransac_loo sweep ──────────────────────────────────────────────
    out = {"good_px": args.good_px, "ransac_c": args.ransac_c,
           "loo_taus": loo_taus, "sweep": {}}
    lines = []
    for loo_tau in loo_taus:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"[ransac_loo sweep]  LOO_tau={loo_tau}  consensus c>={args.ransac_c}"
                     f"  good=9kp<{args.good_px:.0f}px")
        lines.append("=" * 78)
        hdr = f"{'domain':<9}{'tau_px':>7}{'N':>5}{'good%':>8}{'9kp_med':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for dom in DOMAINS:
            for tau in taus:
                passed = []
                for r in rows[dom]:
                    c, R, t = ransac_consensus(r["kp"], r["kp3d"], r["K"],
                                               dist, tau)
                    if c < args.ransac_c or R is None:
                        continue
                    if loo_stability(r["kp"], r["kp3d"], r["K"], dist, R, t,
                                     loo_tau):
                        passed.append(r["e9"])
                n, gp, med = stats(passed)
                lines.append(f"{dom:<9}{tau:>7.0f}{n:>5}"
                             f"{(str(gp) if gp is not None else '--'):>8}"
                             f"{(str(med) if med is not None else '--'):>9}")
                out["sweep"].setdefault(f"loo{loo_tau}", {}).setdefault(
                    dom, {})[str(tau)] = {"N": n, "good_pct": gp, "med": med}
            lines.append("-" * len(hdr))

    # ── diag each / fullkp each (threshold-free) ──────────────────────
    for label, fn in [("diag", lambda kp: filt_diag(kp)[0]),
                      ("fullkp", lambda kp: filt_fullkp(kp)[0])]:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"[{label}]  good=9kp<{args.good_px:.0f}px")
        lines.append("=" * 78)
        hdr = f"{'domain':<9}{'N':>5}{'good%':>8}{'9kp_med':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        out[label] = {}
        for dom in DOMAINS:
            passed = [r["e9"] for r in rows[dom] if fn(r["kp"])]
            n, gp, med = stats(passed)
            lines.append(f"{dom:<9}{n:>5}"
                         f"{(str(gp) if gp is not None else '--'):>8}"
                         f"{(str(med) if med is not None else '--'):>9}")
            out[label][dom] = {"N": n, "good_pct": gp, "med": med}

    print("\n".join(lines))
    od = os.path.join(ROOT, "data", "pallet", "eval_results", "filter_loo_sweep")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"sweep_{args.tag}.json"), "w"),
              indent=2)
    open(os.path.join(od, f"sweep_{args.tag}.txt"), "w").write("\n".join(lines))
    print(f"\n[save] {od}/sweep_{args.tag}.json")


if __name__ == "__main__":
    main()
