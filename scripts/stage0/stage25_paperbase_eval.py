"""stage25_paperbase_eval.py — evaluate paper_base_v2 (논문 트랙 baseline) on the
SAME sets + SAME metrics as B2 / challenge0123.

목적: 논문 base(paper_base_v2, palletobj scan 미학습 = cad/noapril 진짜 unseen)의
      정직한 출발점 숫자. B2(v3/addon replay) 대비 gap = "v3/addon 없이 치른 비용".

모델 (heatmap only, pad100 추론):
  - paper_base_v2  weights/paper_base_v2/final_net_epoch_0060.pth (무패딩 학습→pad100 필수)
  - B2             weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth (internal best)
  - challenge0123  weights/challenge0123/final_net_epoch_0060.pth (최초 v4 base)

평가셋:
  - handannot17 : testset_full8_manifest.txt (cad11 + noapril6, 8/8 완전). 고앙각 편향→정성.
  - filterval   : collect_val_frames(outside+night, split-lock) + manual_gt(36). 주 신호(대표).
    (final-test sessions capturepallet09/07, capturenight09/08 는 SEAL 로 배제)

인프라 재사용: eval_capturecad_b2.eval_frame (order-free Hungarian corner, solve_pose
  order-free W/D, honest full-8 reproj, reflect-pad100, per-frame K). 모든 모델 동일 조건.

★ handannot17 N=17 극소·고앙각 편향, filter-val 도 도메인 편중 → 정성/방향성. 단정 금지.
"""
from __future__ import annotations
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/stage25_paperbase_eval")
MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")

_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

from stage18_elevation_threshold import elev_from_pose  # noqa: E402

GOOD_PX, GROSS_PX = 10.0, 20.0
PAD, THRESH = 100, 0.3

MODELS = {
    "paper_base_v2": "weights/paper_base_v2/final_net_epoch_0060.pth",
    "B2":            "weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth",
    "challenge0123": "weights/challenge0123/final_net_epoch_0060.pth",
}

# ---- filter-val split lock (mirror tau_calibrate.collect_val_frames) ----
FILTER_VAL_SESSIONS = {
    "outside": {"capturepallet08", "capturepallet02", "capturepallet03",
                "capturepallet04", "capturepallet05"},
    "night": {"capturenight06", "capturenight07", "capturenight05"},
}
FINAL_TEST_SESSIONS = {"capturepallet09", "capturepallet07",
                       "capturenight09", "capturenight08"}
EVAL_SET = {"outside": "outside_combined", "night": "night_combined"}
MANUAL_GT_DIR = os.path.join(
    ROOT, "data", "pallet", "eval_results", "stage0_gt_candidates", "manual_gt")
ELEV_BINS = [(-90, 5), (5, 10), (10, 15), (15, 25), (25, 90)]


def build_session_map():
    m = {}
    for base in ("outside", "night"):
        for d in glob.glob(os.path.join(ROOT, "data", "pallet", "raw_data",
                                        base, "capture*")):
            sess = os.path.basename(d)
            for p in glob.glob(os.path.join(d, "rgb", "*.png")):
                m[os.path.splitext(os.path.basename(p))[0]] = sess
    return m


def frames_handannot17():
    """(domain, fid, jp, ip) from the confirmed 8/8 manifest (cad11 + noapril6)."""
    out = []
    for ln in open(MANIFEST):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        dom, fid, jrel, irel = ln.split()
        out.append((dom, fid, os.path.join(ROOT, jrel), os.path.join(ROOT, irel)))
    return out


def frames_filterval():
    """collect_val_frames(outside+night split-lock) + manual_gt(36)."""
    sm = build_session_map()
    out = []
    for dom, valset in FILTER_VAL_SESSIONS.items():
        gt_dir = os.path.join(ROOT, "data", "_eval_sets", EVAL_SET[dom])
        for jp in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
            fid = os.path.splitext(os.path.basename(jp))[0]
            sess = sm.get(fid)
            if sess is None or sess not in valset:
                continue
            assert sess not in FINAL_TEST_SESSIONS, f"final-test leak: {fid}"
            ip = os.path.join(gt_dir, fid + ".png")
            if not os.path.exists(ip):
                ip = os.path.join(ROOT, "data", "pallet", "raw_data", dom,
                                  sess, "rgb", fid + ".png")
            out.append((dom, fid, jp, ip))
    for jp in sorted(glob.glob(os.path.join(MANUAL_GT_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(MANUAL_GT_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("manual", fid, jp, ip))
    return out


SETS = {"handannot17": frames_handannot17, "filterval": frames_filterval}


def per_corner_dists(row):
    pr = np.array(row["pred8"], float)
    gt = np.array(row["gt8"], float)
    return E.hungarian(pr, gt)


def med(a):
    a = [x for x in a if np.isfinite(x)]
    return round(float(np.median(a)), 1) if a else None


def summarize(rows):
    n = len(rows)
    det = [r for r in rows if r["det"]]
    front = [r["front"] for r in rows if np.isfinite(r["front"])]
    rear = [r["back"] for r in rows if np.isfinite(r["back"])]
    corner = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    worst2 = [r["worst2"] for r in rows if np.isfinite(r["worst2"])]
    honest = [r["pnp_honest8"] for r in rows
              if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
    good = gross = ncorner = 0
    for r in rows:
        d, _ = per_corner_dists(r)
        if d is None:
            continue
        for x in d:
            ncorner += 1
            good += int(x < GOOD_PX)
            gross += int(x > GROSS_PX)
    return {
        "n": n,
        "det_pct": round(100 * len(det) / n, 0) if n else None,
        "front_med": med(front),
        "rear_med": med(rear),
        "corner_med": med(corner),
        "worst2_med": med(worst2),
        "pnp_pct": round(100 * np.mean([r["pnp_ok"] for r in rows]), 0) if n else None,
        "honest8_med": med(honest),
        "good": good, "gross": gross, "ncorner": ncorner,
        "good_pct": round(100 * good / ncorner, 1) if ncorner else None,
        "gross_pct": round(100 * gross / ncorner, 1) if ncorner else None,
    }


def elev_of(jp):
    try:
        T = json.load(open(jp))["objects"][0].get("pose_transform")
        return elev_from_pose(T) if T is not None else None
    except Exception:
        return None


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # gather frame lists once (shared, so V_geom/elev meta identical across models)
    fl = {s: fn() for s, fn in SETS.items()}
    for s in fl:
        by = {}
        for dom, *_ in fl[s]:
            by[dom] = by.get(dom, 0) + 1
        print(f"[set] {s}: N={len(fl[s])} {by}")

    # elevation per fid (GT-only)
    elev = {s: {fid: elev_of(jp) for _, fid, jp, _ in fl[s]} for s in fl}

    results = {m: {s: [] for s in SETS} for m in MODELS}
    for mname, wrel in MODELS.items():
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for s in SETS:
            for dom, fid, jp, ip in fl[s]:
                r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD)
                if r is not None:
                    r["dom"] = dom
                    results[mname][s].append(r)
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname}")

    # V_geom + elevation dist (GT-only, from paper_base_v2 rows)
    vdist, edist = {}, {}
    ref = "paper_base_v2"
    for s in SETS:
        vdist[s], edist[s] = {}, {i: 0 for i in range(len(ELEV_BINS))}
        for r in results[ref][s]:
            vdist[s][r["v_geom"]] = vdist[s].get(r["v_geom"], 0) + 1
            e = elev[s].get(r["fid"])
            if e is not None:
                for i, (lo, hi) in enumerate(ELEV_BINS):
                    if lo <= e < hi:
                        edist[s][i] += 1
                        break

    out = {"models": MODELS, "pad": PAD, "thresh": THRESH,
           "sets": {}, "v_geom": {}, "elev_bins": {},
           "elev_bin_edges": ELEV_BINS}
    for s in SETS:
        out["v_geom"][s] = {str(k): v for k, v in sorted(vdist[s].items())}
        out["elev_bins"][s] = {f"{ELEV_BINS[i][0]}~{ELEV_BINS[i][1]}": edist[s][i]
                               for i in range(len(ELEV_BINS))}
        out["sets"][s] = {}
        for m in MODELS:
            out["sets"][s][m] = {"overall": summarize(results[m][s])}
            # per-domain breakdown
            doms = sorted(set(r["dom"] for r in results[m][s]))
            out["sets"][s][m]["by_domain"] = {
                dm: summarize([r for r in results[m][s] if r["dom"] == dm])
                for dm in doms}
            # V<8 vs V=8 (truncation split)
            out["sets"][s][m]["V8"] = summarize(
                [r for r in results[m][s] if r["v_geom"] == 8])
            out["sets"][s][m]["Vlt8"] = summarize(
                [r for r in results[m][s] if r["v_geom"] < 8])
            # elevation bins
            eb = {}
            for i, (lo, hi) in enumerate(ELEV_BINS):
                sel = [r for r in results[m][s]
                       if (elev[s].get(r["fid"]) is not None
                           and lo <= elev[s][r["fid"]] < hi)]
                if sel:
                    eb[f"{lo}~{hi}"] = summarize(sel)
            out["sets"][s][m]["elev_bins"] = eb

    with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"[save] {OUT_DIR}/eval.json")

    # ---- summary.md ----
    write_summary(out)


def _row(label, s):
    return (f"{label:<15}{str(s['n']):>4}{str(s['det_pct']):>6}"
            f"{str(s['front_med']):>7}{str(s['rear_med']):>7}"
            f"{str(s['corner_med']):>8}{str(s['worst2_med']):>8}"
            f"{str(s['pnp_pct']):>6}{str(s['honest8_med']):>9}"
            f"{str(s['good_pct']):>7}{str(s['gross_pct']):>7}")


def _hdr():
    return (f"{'model':<15}{'n':>4}{'det%':>6}{'front':>7}{'rear':>7}"
            f"{'corner':>8}{'worst2':>8}{'pnp%':>6}{'honest8':>9}"
            f"{'good%':>7}{'gross%':>7}")


def write_summary(out):
    L = ["# STAGE25 — paper_base_v2 vs B2 vs challenge0123 (same sets, same metrics)",
         "",
         "eval: order-free Hungarian corner + solve_pose(order-free W/D) + honest",
         "full-8 reproj(GT projected_cuboid) + reflect-pad100 + per-frame K.",
         "front/rear = matched-GT-idx bucket. good%/gross% = per-corner over all matched.",
         "",
         "★ handannot17 N=17 극소·고앙각 편향(정성). filter-val = 주 신호(대표).",
         "★ paper_base_v2 는 palletobj scan(v1/v2/v3/addon) 미학습 → cad/noapril 진짜 unseen.",
         ""]
    order = ["paper_base_v2", "B2", "challenge0123"]
    for s in out["sets"]:
        vd = out["v_geom"][s]
        ed = out["elev_bins"][s]
        L.append(f"## {s}")
        L.append("V_geom: " + ", ".join(f"V{k}:{v}" for k, v in vd.items()))
        L.append("elev bins(deg): " + ", ".join(f"{k}:{v}" for k, v in ed.items()))
        L.append("")
        L.append("### overall")
        L.append(_hdr()); L.append("-" * len(_hdr()))
        for m in order:
            L.append(_row(m, out["sets"][s][m]["overall"]))
        L.append("")
        # per-domain
        doms = sorted(set().union(*[set(out["sets"][s][m]["by_domain"])
                                    for m in order]))
        for dm in doms:
            L.append(f"### {s} / domain={dm}")
            L.append(_hdr()); L.append("-" * len(_hdr()))
            for m in order:
                bd = out["sets"][s][m]["by_domain"].get(dm)
                if bd:
                    L.append(_row(m, bd))
            L.append("")
        # truncation split
        L.append(f"### {s} / truncation split (V=8 full-view vs V<8 truncated)")
        L.append(_hdr()); L.append("-" * len(_hdr()))
        for m in order:
            L.append(_row(m + " V8", out["sets"][s][m]["V8"]))
            L.append(_row(m + " V<8", out["sets"][s][m]["Vlt8"]))
        L.append("")

    # gap vs B2 (filterval overall)
    L.append("## paper_base_v2 vs B2 GAP (filterval overall)")
    pb = out["sets"]["filterval"]["paper_base_v2"]["overall"]
    b2 = out["sets"]["filterval"]["B2"]["overall"]
    ch = out["sets"]["filterval"]["challenge0123"]["overall"]

    def d(a, b):
        return None if a is None or b is None else round(a - b, 1)
    L.append(f"  corner_med: paper={pb['corner_med']} B2={b2['corner_med']} "
             f"chal={ch['corner_med']}  Δ(paper-B2)={d(pb['corner_med'], b2['corner_med'])}")
    L.append(f"  rear_med  : paper={pb['rear_med']} B2={b2['rear_med']} "
             f"chal={ch['rear_med']}  Δ(paper-B2)={d(pb['rear_med'], b2['rear_med'])}")
    L.append(f"  honest8   : paper={pb['honest8_med']} B2={b2['honest8_med']} "
             f"chal={ch['honest8_med']}  Δ(paper-B2)={d(pb['honest8_med'], b2['honest8_med'])}")
    L.append(f"  det%      : paper={pb['det_pct']} B2={b2['det_pct']} chal={ch['det_pct']}")
    L.append(f"  good%     : paper={pb['good_pct']} B2={b2['good_pct']} chal={ch['good_pct']}")
    L.append("")

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
