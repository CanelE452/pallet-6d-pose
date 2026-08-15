"""eval_paper_s1_paired.py — PaperBase-v2 vs Paper-S1 (mask-aux), A_nopad, PAIRED.

Official protocol = A_nopad (no-pad / aspect), the train/infer-parity measurement for
no-pad-trained models (official_baseline_protocol.md). SAME frames for both models
(same-frame paired), SAME metrics: order-free Hungarian corner + solve_pose(order-free
W/D) + honest full-8 reproj + per-frame K (reused from eval_capturecad_b2 via stage25).

Sets: filterval (N=123, primary) + handannot17 (N=17, high-elev, qualitative).
Both models loaded strict=False -> Paper-S1's seg head is DROPPED at inference
(belief-peak decode, NO mask hard-gate). Judgement only; no training.

Usage:
  python scripts/stage0/eval_harness/eval_paper_s1_paired.py --s1_weights weights/paper_s1/paper_s1_maskaux/<ckpt>.pth
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
import stage25_paperbase_eval as S  # noqa: E402
from stage25_paperbase_eval import E, summarize, SETS, ELEV_BINS, elev_of  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s1")
BASE_W = os.path.join(ROOT, "weights/paper_base/paper_base_v2/final_net_epoch_0060.pth")
THRESH = 0.3
PAD = 0  # A_nopad (official)

METRICS = ["det_pct", "front_med", "rear_med", "corner_med", "honest8_med",
           "good_pct", "gross_pct", "pnp_pct"]
# direction of "better": +1 = higher is better, -1 = lower is better
BETTER = {"det_pct": +1, "front_med": -1, "rear_med": -1, "corner_med": -1,
          "honest8_med": -1, "good_pct": +1, "gross_pct": -1, "pnp_pct": +1}


def breakdown(rows, elev_map):
    out = {"overall": summarize(rows),
           "V8": summarize([r for r in rows if r["v_geom"] == 8]),
           "Vlt8": summarize([r for r in rows if r["v_geom"] < 8])}
    eb = {}
    for lo, hi in ELEV_BINS:
        sel = [r for r in rows if (elev_map.get(r["fid"]) is not None
                                   and lo <= elev_map[r["fid"]] < hi)]
        if sel:
            eb[f"{lo}~{hi}"] = summarize(sel)
    out["elev_bins"] = eb
    doms = sorted(set(r["dom"] for r in rows))
    out["by_domain"] = {dm: summarize([r for r in rows if r["dom"] == dm])
                        for dm in doms}
    return out


def _fmt(v):
    return "  -  " if v is None else f"{v}"


def paired_table(base_s, s1_s, title):
    L = [f"### {title}",
         f"{'metric':<11}{'PaperBase-v2':>14}{'Paper-S1':>12}{'Δ(S1-Base)':>13}"]
    L.append("-" * len(L[-1]))
    for m in METRICS:
        b, s = base_s.get(m), s1_s.get(m)
        d = ""
        if isinstance(b, (int, float)) and isinstance(s, (int, float)):
            delta = round(s - b, 2)
            arrow = ""
            if delta != 0:
                improved = (delta > 0) == (BETTER[m] > 0)
                arrow = " ↑good" if improved else " ↓bad"
            d = f"{delta:+g}{arrow}"
        L.append(f"{m:<11}{_fmt(b):>14}{_fmt(s):>12}{d:>13}")
    return "\n".join(L)


def verdict(base_s, s1_s):
    """Success criteria (spec): det held (±2%p) AND rear↓ AND corner↓ AND good↑
    AND gross↓ AND PnP↑."""
    def g(d, k):
        return d.get(k)
    checks = {}
    det_b, det_s = g(base_s, "det_pct"), g(s1_s, "det_pct")
    checks["det held (±2%p)"] = (det_b is not None and det_s is not None
                                 and abs(det_s - det_b) <= 2)
    for k, label in [("rear_med", "rear_med ↓"), ("corner_med", "corner_med ↓"),
                     ("honest8_med", "honest8 ↓")]:
        b, s = g(base_s, k), g(s1_s, k)
        checks[label] = (b is not None and s is not None and s < b)
    checks["good_pct ↑"] = (g(base_s, "good_pct") is not None
                            and g(s1_s, "good_pct") is not None
                            and g(s1_s, "good_pct") > g(base_s, "good_pct"))
    checks["gross_pct ↓"] = (g(base_s, "gross_pct") is not None
                             and g(s1_s, "gross_pct") is not None
                             and g(s1_s, "gross_pct") < g(base_s, "gross_pct"))
    checks["PnP% ↑"] = (g(base_s, "pnp_pct") is not None
                        and g(s1_s, "pnp_pct") is not None
                        and g(s1_s, "pnp_pct") > g(base_s, "pnp_pct"))
    return checks


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_weights", required=True)
    args = ap.parse_args()
    s1_w = args.s1_weights if os.path.isabs(args.s1_weights) \
        else os.path.join(ROOT, args.s1_weights)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fl = {s: fn() for s, fn in SETS.items()}
    elev = {s: {fid: elev_of(jp) for _, fid, jp, _ in fl[s]} for s in fl}
    for s in fl:
        by = {}
        for dom, *_ in fl[s]:
            by[dom] = by.get(dom, 0) + 1
        print(f"[set] {s}: N={len(fl[s])} {by}")

    models = {"paper_base_v2": BASE_W, "paper_s1": s1_w}
    results = {m: {s: [] for s in SETS} for m in models}
    for mname, wpath in models.items():
        mdl = E.load_model(wpath, device)
        for s in SETS:
            for dom, fid, jp, ip in fl[s]:
                r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD)
                if r is not None:
                    r["dom"] = dom
                    results[mname][s].append(r)
        print(f"[done] {mname}")

    out = {"protocol": "A_nopad (official)", "thresh": THRESH, "pad": PAD,
           "weights": {"paper_base_v2": BASE_W, "paper_s1": s1_w}, "sets": {}}
    for s in SETS:
        out["sets"][s] = {m: breakdown(results[m][s], elev[s]) for m in models}
    with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"[save] {OUT_DIR}/eval.json")
    write_summary(out, models, s1_w)


def write_summary(out, models, s1_w):
    L = ["# Paper-S1 (mask-aux) vs PaperBase-v2 — A_nopad paired eval", "",
         f"- protocol: A_nopad (official no-pad/aspect, train/infer parity). pad={PAD}, thresh={THRESH}.",
         "- metric: order-free Hungarian corner + solve_pose(order-free W/D) + honest full-8 reproj + per-frame K.",
         "- good%<10px, gross%>20px. same-frame paired (identical frame list per model).",
         f"- PaperBase-v2: {models['paper_base_v2']}",
         f"- Paper-S1:     {s1_w}",
         "- ★ data = V=8 100%. rear/corner improvement claims are FULL-VIEW only; NO V<8/truncation claim.",
         ""]
    for s in SETS:
        base = out["sets"][s]["paper_base_v2"]
        s1 = out["sets"][s]["paper_s1"]
        n = base["overall"]["n"]
        L.append(f"## {s}  (N={n})")
        L.append("")
        L.append("```")
        L.append(paired_table(base["overall"], s1["overall"], f"{s} overall"))
        L.append("")
        L.append(paired_table(base["V8"], s1["V8"], f"{s} V=8 (full-view)"))
        if base["Vlt8"]["n"]:
            L.append("")
            L.append(paired_table(base["Vlt8"], s1["Vlt8"],
                                  f"{s} V<8 (context only — not a claim)"))
        L.append("```")
        L.append("")
        # elevation bins (filterval primary)
        if s == "filterval":
            L.append("### elevation bins (overall)")
            L.append("```")
            for bk in sorted(set(base["elev_bins"]) | set(s1["elev_bins"]),
                             key=lambda x: float(x.split("~")[0])):
                if bk in base["elev_bins"] and bk in s1["elev_bins"]:
                    L.append(paired_table(base["elev_bins"][bk], s1["elev_bins"][bk],
                                          f"elev {bk} deg"))
                    L.append("")
            L.append("```")
            L.append("")
    # verdict on filterval overall (primary)
    L.append("## Success-criteria verdict (filterval overall, primary)")
    L.append("```")
    checks = verdict(out["sets"]["filterval"]["paper_base_v2"]["overall"],
                     out["sets"]["filterval"]["paper_s1"]["overall"])
    for k, v in checks.items():
        L.append(f"  [{'PASS' if v else 'FAIL'}] {k}")
    npass = sum(checks.values())
    L.append("")
    L.append(f"  {npass}/{len(checks)} criteria met.")
    if npass == len(checks):
        L.append("  => FULL SIGNAL: all criteria met — full-paper candidate.")
    elif npass >= len(checks) - 1:
        L.append("  => STRONG SIGNAL: most criteria met — full-paper candidate.")
    elif npass >= 3:
        L.append("  => PARTIAL/MIXED signal — inspect per-metric; not conclusive.")
    else:
        L.append("  => NO SIGNAL: mask-aux did not help on paper track (this data).")
    L.append("```")
    L.append("")
    L.append("★ Small-sample caveat: filterval N=123, handannot17 N=17. Real judgement only;")
    L.append("  ckpt selected on synthetic val. Paper purity: v3/addon/B2 unused. Do not over-conclude.")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
