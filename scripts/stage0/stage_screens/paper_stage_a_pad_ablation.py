"""paper_stage_a_pad_ablation.py — PAPER_STAGE_A PART 1.

paper_base_v2 (무패딩 학습: aspect + truncation_aug_prob=0.0) 의 논문 정량 protocol
을 확정하기 위한 전처리 4-way ablation.

  A_nopad  : no-pad aspect (pad=0 -> pad_frame no-op -> preprocess 400/min aspect resize)
  B_pad50  : reflect-pad 50  (belief->orig 역매핑 *(W+2P)/W 후 -P)
  C_pad100 : reflect-pad 100 (= STAGE25 재현, cross-check)
  D_pad150 : reflect-pad 150 (optional)

train/infer parity 가설: 무패딩 학습 모델은 no-pad(aspect) 가 코너 정확도에 유리,
pad 는 검출↑이나 zoom-out 열화 + gross↑ (memory: dope-inference-needs-reflect-padding
6/22 정정 = "패딩 항상 좋음" 이 아니라 parity 가 본질).

인프라 재사용: stage25_paperbase_eval (frames_handannot17 / frames_filterval /
summarize / elev / SETS) + eval_capturecad_b2.eval_frame (order-free Hungarian corner,
solve_pose order-free W/D, honest full-8 reproj, per-frame K).

평가셋: filterval(주 신호, split-lock, final-test 봉인) + handannot17(N=17 고앙각 정성).
★ 학습 실행 없음. 판정만.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import stage25_paperbase_eval as S  # noqa: E402
from stage25_paperbase_eval import E, summarize, SETS, ELEV_BINS, elev_of  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_stage_a")
WEIGHTS = os.path.join(ROOT, "weights/paper_base/paper_base_v2/final_net_epoch_0060.pth")
THRESH = 0.3
PADS = {"A_nopad": 0, "B_pad50": 50, "C_pad100": 100, "D_pad150": 150}


def elev_dist(rows, elev_map):
    ed = {i: 0 for i in range(len(ELEV_BINS))}
    for r in rows:
        e = elev_map.get(r["fid"])
        if e is None:
            continue
        for i, (lo, hi) in enumerate(ELEV_BINS):
            if lo <= e < hi:
                ed[i] += 1
                break
    return {f"{ELEV_BINS[i][0]}~{ELEV_BINS[i][1]}": ed[i]
            for i in range(len(ELEV_BINS))}


def summarize_breakdown(rows, elev_map):
    out = {"overall": summarize(rows)}
    doms = sorted(set(r["dom"] for r in rows))
    out["by_domain"] = {dm: summarize([r for r in rows if r["dom"] == dm])
                        for dm in doms}
    out["V8"] = summarize([r for r in rows if r["v_geom"] == 8])
    out["Vlt8"] = summarize([r for r in rows if r["v_geom"] < 8])
    eb = {}
    for i, (lo, hi) in enumerate(ELEV_BINS):
        sel = [r for r in rows if (elev_map.get(r["fid"]) is not None
                                   and lo <= elev_map[r["fid"]] < hi)]
        if sel:
            eb[f"{lo}~{hi}"] = summarize(sel)
    out["elev_bins"] = eb
    return out


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fl = {s: fn() for s, fn in SETS.items()}
    elev = {s: {fid: elev_of(jp) for _, fid, jp, _ in fl[s]} for s in fl}
    for s in fl:
        by = {}
        for dom, *_ in fl[s]:
            by[dom] = by.get(dom, 0) + 1
        print(f"[set] {s}: N={len(fl[s])} {by}")

    mdl = E.load_model(WEIGHTS, device)
    # results[pad_name][set] = rows
    results = {pn: {s: [] for s in SETS} for pn in PADS}
    for pn, pad in PADS.items():
        for s in SETS:
            for dom, fid, jp, ip in fl[s]:
                r = E.eval_frame(mdl, jp, ip, device, THRESH, pad)
                if r is not None:
                    r["dom"] = dom
                    results[pn][s].append(r)
        print(f"[done] {pn} (pad={pad})")

    out = {"weights": WEIGHTS, "thresh": THRESH, "pads": PADS,
           "elev_bin_edges": ELEV_BINS, "meta": {}, "sets": {}}
    for s in SETS:
        vd = {}
        for r in results["A_nopad"][s]:
            vd[r["v_geom"]] = vd.get(r["v_geom"], 0) + 1
        out["meta"][s] = {
            "n": len(fl[s]),
            "v_geom": {str(k): v for k, v in sorted(vd.items())},
            "elev_bins": elev_dist(results["A_nopad"][s], elev[s]),
        }
        out["sets"][s] = {pn: summarize_breakdown(results[pn][s], elev[s])
                          for pn in PADS}

    with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"[save] {OUT_DIR}/eval.json")
    write_report(out)


def _hdr():
    return (f"{'variant':<11}{'n':>4}{'det%':>6}{'front':>7}{'rear':>7}"
            f"{'corner':>8}{'worst2':>8}{'pnp%':>6}{'honest8':>9}"
            f"{'good%':>7}{'gross%':>7}")


def _row(label, s):
    return (f"{label:<11}{str(s['n']):>4}{str(s['det_pct']):>6}"
            f"{str(s['front_med']):>7}{str(s['rear_med']):>7}"
            f"{str(s['corner_med']):>8}{str(s['worst2_med']):>8}"
            f"{str(s['pnp_pct']):>6}{str(s['honest8_med']):>9}"
            f"{str(s['good_pct']):>7}{str(s['gross_pct']):>7}")


def write_report(out):
    L = ["# PAPER_STAGE_A PART 1 — paper_base_v2 preprocess ablation (A/B/C/D)",
         "",
         "weights: weights/paper_base/paper_base_v2/final_net_epoch_0060.pth "
         "(무패딩 학습: aspect + truncation_aug_prob=0.0)",
         "eval: order-free Hungarian corner + solve_pose(order-free W/D) + honest",
         "full-8 reproj(GT projected_cuboid) + per-frame K. good%<10px, gross%>20px.",
         "A_nopad=aspect(pad0) B_pad50 C_pad100(=STAGE25 재현) D_pad150(reflect).",
         "",
         "★ filterval = 주 신호(split-lock, final-test 봉인). handannot17 N=17 고앙각 정성.",
         "★ 판정 기준: good%↑ / corner↓ / rear↓ / honest8↓ / PnP%. no-pad 우세 예상(예단 금지, 데이터로).",
         ""]
    order = list(PADS.keys())
    for s in SETS:
        m = out["meta"][s]
        L.append(f"## {s}  (N={m['n']})")
        L.append("V_geom: " + ", ".join(f"V{k}:{v}" for k, v in m["v_geom"].items()))
        L.append("elev bins(deg): " + ", ".join(f"{k}:{v}"
                                                 for k, v in m["elev_bins"].items()))
        L.append("")
        L.append("### overall")
        L.append(_hdr()); L.append("-" * len(_hdr()))
        for pn in order:
            L.append(_row(pn, out["sets"][s][pn]["overall"]))
        L.append("")
        # truncation split
        L.append("### truncation split (V=8 full-view vs V<8 truncated)")
        L.append(_hdr()); L.append("-" * len(_hdr()))
        for pn in order:
            L.append(_row(pn + " V8", out["sets"][s][pn]["V8"]))
            L.append(_row(pn + " V<8", out["sets"][s][pn]["Vlt8"]))
        L.append("")
        # per-domain
        doms = sorted(set().union(*[set(out["sets"][s][pn]["by_domain"])
                                    for pn in order]))
        for dm in doms:
            L.append(f"### domain={dm}")
            L.append(_hdr()); L.append("-" * len(_hdr()))
            for pn in order:
                bd = out["sets"][s][pn]["by_domain"].get(dm)
                if bd:
                    L.append(_row(pn, bd))
            L.append("")
        # elevation bins (overall per variant, filterval only for signal)
        if s == "filterval":
            allbins = []
            for pn in order:
                allbins += list(out["sets"][s][pn]["elev_bins"].keys())
            for bk in sorted(set(allbins),
                             key=lambda x: float(x.split("~")[0])):
                L.append(f"### elev bin {bk} deg")
                L.append(_hdr()); L.append("-" * len(_hdr()))
                for pn in order:
                    eb = out["sets"][s][pn]["elev_bins"].get(bk)
                    if eb:
                        L.append(_row(pn, eb))
                L.append("")

    with open(os.path.join(OUT_DIR, "pad_ablation_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_DIR}/pad_ablation_summary.md")


if __name__ == "__main__":
    main()
