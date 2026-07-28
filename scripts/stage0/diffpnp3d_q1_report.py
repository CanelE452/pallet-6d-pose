#!/usr/bin/env python
"""diffpnp3d_q1_report.py — PAPER_S2 Phase 3 (Q1) aggregate + GO/STOP verdict.

Reads q1_val_lam*.json (one per lambda, produced by diffpnp3d_q1_eval.py), builds
the lambda comparison table (overall / V8 / elev<10 low-angle), computes Δ vs the
lambda=0 baseline, renders a GO/STOP verdict against the user's criterion, and
draws a few rear-corner overlays (baseline vs best lambda).

GO (per user, 2026-07-08) = >=2 of {rear_med down, honest8 down, PnP% up, gross% down}
   AND guards hold: front_med not worse (Δ<=+1.0px), det% & good% no sharp drop (Δ>=-5%p).
Else STOP. Small-sample / undertrain caveats printed.

Usage: python scripts/stage0/diffpnp3d_q1_report.py --lambdas 0 0.003 0.005 0.008
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
EVAL_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp")
OUT_MD = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/quick_screen_results.md")
OV_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/q1_overlays")

METRICS = ["det_pct", "front_med", "rear_med", "corner_med", "worst2_med",
           "pnp_pct", "honest8_med", "good_pct", "gross_pct"]
BETTER = {"det_pct": +1, "front_med": -1, "rear_med": -1, "corner_med": -1,
          "worst2_med": -1, "pnp_pct": +1, "honest8_med": -1,
          "good_pct": +1, "gross_pct": -1}


def load(lams):
    data = {}
    for lam in lams:
        fp = os.path.join(EVAL_DIR, f"q1_val_lam{lam}.json")
        if os.path.exists(fp):
            data[lam] = json.load(open(fp))
    return data


def fmt(v):
    return " -" if v is None else f"{v:g}"


def table(data, lams, section):
    hdr = f"{'lambda':<9}" + "".join(f"{m:>11}" for m in METRICS)
    L = [hdr, "-" * len(hdr)]
    for lam in lams:
        if lam not in data:
            continue
        s = data[lam][section]
        L.append(f"{lam:<9}" + "".join(f"{fmt(s.get(m)):>11}" for m in METRICS))
    return "\n".join(L)


def delta_table(data, lams, section):
    base = data[lams[0]][section]
    hdr = f"{'lambda':<9}" + "".join(f"{m:>11}" for m in METRICS)
    L = ["Δ vs lambda=0 (↓=improve for err/gross, ↑=improve for det/pnp/good):",
         hdr, "-" * len(hdr)]
    for lam in lams[1:]:
        if lam not in data:
            continue
        s = data[lam][section]
        cells = []
        for m in METRICS:
            b, v = base.get(m), s.get(m)
            if b is None or v is None:
                cells.append(f"{'-':>11}")
                continue
            d = round(v - b, 1)
            good = (d != 0) and ((d > 0) == (BETTER[m] > 0))
            mark = "" if d == 0 else ("+" if good else "!")
            cells.append(f"{f'{d:+g}{mark}':>11}")
        L.append(f"{lam:<9}" + "".join(cells))
    return "\n".join(L)


def verdict(data, lams, section="overall"):
    if lams[0] not in data:
        return ["NO BASELINE (lambda=0 eval missing)"], None
    base = data[lams[0]][section]
    out = []
    best_lam, best_score = None, -1
    per_lam = {}
    for lam in lams[1:]:
        if lam not in data:
            continue
        s = data[lam][section]

        def d(m):
            b, v = base.get(m), s.get(m)
            return None if (b is None or v is None) else round(v - b, 2)
        crit = {
            "rear_med↓": (d("rear_med") is not None and d("rear_med") < 0),
            "honest8↓": (d("honest8_med") is not None and d("honest8_med") < 0),
            "PnP%↑": (d("pnp_pct") is not None and d("pnp_pct") > 0),
            "gross%↓": (d("gross_pct") is not None and d("gross_pct") < 0),
        }
        n_improve = sum(crit.values())
        # guards
        front_ok = (d("front_med") is None) or (d("front_med") <= 1.0)
        det_ok = (d("det_pct") is None) or (d("det_pct") >= -5.0)
        good_ok = (d("good_pct") is None) or (d("good_pct") >= -5.0)
        guards_ok = front_ok and det_ok and good_ok
        go = (n_improve >= 2) and guards_ok
        per_lam[lam] = {"crit": crit, "n_improve": n_improve,
                        "front_ok": front_ok, "det_ok": det_ok,
                        "good_ok": good_ok, "go": go,
                        "d_rear": d("rear_med"), "d_honest8": d("honest8_med"),
                        "d_front": d("front_med"), "d_det": d("det_pct"),
                        "d_good": d("good_pct"), "d_gross": d("gross_pct"),
                        "d_pnp": d("pnp_pct")}
        # composite score to pick "best" lambda (favor rear+honest8)
        score = n_improve + (1 if go else 0)
        if score > best_score:
            best_score, best_lam = score, lam

    out.append(f"criterion: GO = >=2 of {{rear↓, honest8↓, PnP%↑, gross%↓}} AND "
               f"guards(front Δ<=+1.0, det/good Δ>=-5%p). section={section}")
    for lam in lams[1:]:
        if lam not in per_lam:
            continue
        p = per_lam[lam]
        cs = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in p["crit"].items())
        g = (f"front_ok={'Y' if p['front_ok'] else 'n'} "
             f"det_ok={'Y' if p['det_ok'] else 'n'} "
             f"good_ok={'Y' if p['good_ok'] else 'n'}")
        out.append(f"  lam={lam}: improve={p['n_improve']}/4 [{cs}] | {g} "
                   f"=> {'GO' if p['go'] else 'STOP'}")
    any_go = any(per_lam[l]["go"] for l in per_lam)
    out.append("")
    out.append(f"VERDICT ({section}): {'GO' if any_go else 'STOP'}  "
               f"(best lambda={best_lam})")
    return out, (best_lam, any_go, per_lam)


def overlays(best_lam, lams):
    """rear-corner overlays: baseline lam0 vs best lam on a few V8 low-angle frames
    where lam-best most reduced rear error (needs re-inference)."""
    if best_lam is None:
        return None
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
    import diffpnp3d_q1_eval as EV
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = json.load(open(EV.VAL_LIST))
    # candidate frames: V8 & low elevation
    cand = [e for e in val]
    w0 = os.path.join(ROOT, "weights/paper_s2_q1/lam0/final_net_epoch_0008.pth")
    wb = os.path.join(ROOT, f"weights/paper_s2_q1/lam{best_lam}/final_net_epoch_0008.pth")
    if not (os.path.exists(w0) and os.path.exists(wb)):
        return None
    m0, mb = EV.load_model(w0, dev), EV.load_model(wb, dev)
    rows = []
    for e in cand:
        r0 = EV.eval_frame(m0, e, dev)
        rb = EV.eval_frame(mb, e, dev)
        if r0 is None or rb is None:
            continue
        if (r0["elev"] is not None and r0["elev"] < 12 and r0["v_geom"] == 8
                and np.isfinite(r0["back"]) and np.isfinite(rb["back"])):
            rows.append((r0["back"] - rb["back"], e, r0, rb))
    rows.sort(key=lambda x: -x[0])   # biggest rear improvement first
    os.makedirs(OV_DIR, exist_ok=True)
    picks = rows[:4]
    saved = []
    for imp, e, r0, rb in picks:
        img = cv2.imread(e["png"])
        if img is None:
            continue
        canvas = np.hstack([img.copy(), img.copy()])
        for tag, r, xoff in [("lam0", r0, 0), (f"lam{best_lam}", rb, img.shape[1])]:
            gt = np.array(r["gt8"], float)
            pr = np.array(r["pred8"], float)
            for j, p in enumerate(gt):
                col = (0, 200, 0) if j in EV.FRONT else (0, 140, 255)
                cv2.circle(canvas, (int(p[0]) + xoff, int(p[1])), 5, col, 2)
            for j, p in enumerate(pr):
                if np.isnan(p[0]):
                    continue
                col = (0, 0, 255) if j in EV.BACK else (255, 100, 0)
                cv2.circle(canvas, (int(p[0]) + xoff, int(p[1])), 3, col, -1)
            cv2.putText(canvas, f"{tag} rear={r['back']:.1f}",
                        (8 + xoff, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        fp = os.path.join(OV_DIR, f"rearimp_{e['fid']}.jpg")
        cv2.imwrite(fp, canvas)
        saved.append((fp, imp))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", nargs="+", default=["0", "0.003", "0.005", "0.008"])
    ap.add_argument("--no_overlay", action="store_true")
    args = ap.parse_args()
    lams = args.lambdas
    data = load(lams)
    if lams[0] not in data:
        print("ERROR: baseline lambda=0 eval json missing")
        sys.exit(1)

    n = data[lams[0]]["overall"]["n"]
    v8n = data[lams[0]]["V8"]["n"]
    lown = data[lams[0]]["elev_lt10"]["n"]

    L = ["# PAPER_S2 DiffPnP3D — Q1 quick-screen results", "",
         "Fixed train 3000 (2400 DiffPnP-eligible interior&V8 + 600 2D-aug) / val 500.",
         "scratch, 9 passes (ep0-8, --epochs 8), batch12, sigma2, aspect-squash, "
         "seed42 (identical init+shuffle all runs). ONLY variable = diffpnp_lambda "
         "(warmup0, ramp500 steps, temp0.1). Eval = squash-parity preprocess + "
         "anisotropic belief->orig, order-free Hungarian corner + per-frame-dims "
         "honest8. Final ckpt = ep8 for every lambda.", "",
         f"val N={n}  (V8={v8n}, elev<10={lown}).", ""]

    for sec, title in [("overall", "OVERALL (val 500)"),
                       ("V8", "V=8 full-view (DiffPnP-applicable regime)"),
                       ("elev_lt10", "low-angle elev<10deg (rear-collapse regime)")]:
        L.append(f"## {title}")
        L.append("```")
        L.append(table(data, lams, sec))
        L.append("")
        L.append(delta_table(data, lams, sec))
        L.append("```")
        L.append("")

    L.append("## GO / STOP verdict")
    L.append("```")
    vo, info_o = verdict(data, lams, "overall")
    L += vo
    L.append("")
    vv, info_v = verdict(data, lams, "V8")
    L += vv
    L.append("```")
    L.append("")

    best_lam = info_v[0] if info_v else None
    any_go = (info_o and info_o[1]) or (info_v and info_v[1])
    L.append("## Caveats (honest)")
    L.append("- 3000-frame / 9-pass scratch = heavily UNDERTRAINED; absolute px are "
             "high vs a full 60-epoch model. Screen reads RELATIVE Δ(lambda) only.")
    L.append("- val N=500 synthetic, 91% V=8 (full-view) — measures the regime the "
             "DiffPnP loss is APPLIED to; does NOT test the hard real rear low-angle "
             "sim2real regime (that is the STAGE22/23 lever). in-distribution check.")
    L.append("- honest8 uses per-frame dims (index) via order-free W/D solve; PnP "
             "success is low at this undertrain budget so honest8 median is a small "
             "subsample — treat as secondary to rear_med (2D, dims-free).")
    L.append("- medians on 500 frames carry noise; Δ smaller than ~0.5px is not "
             "meaningful. 'improvement' claims below note magnitude.")
    L.append("")

    ov_saved = None
    if not args.no_overlay:
        try:
            ov_saved = overlays(best_lam, lams)
        except Exception as ex:
            L.append(f"(overlay generation failed: {ex})")
    if ov_saved:
        L.append("## Overlays (baseline lam0 | best lam, rear corners)")
        for fp, imp in ov_saved:
            L.append(f"- {fp}  (rear improvement {imp:+.1f}px)")
        L.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_MD}")


if __name__ == "__main__":
    main()
