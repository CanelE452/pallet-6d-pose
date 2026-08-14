"""paper_s2_filterval_9filters.py — apply the project's 9 PL-accept filters
(f1..f9 from s1_cad_9filters.py) to Stage B (paper_s2 net_epoch_0057) predictions
on the REAL filterval set (outside44 + night43 + manual36 = 123, LOW-ANGLE), with
per-domain FILTER SELECTIVITY (filter-accept vs GT-good confusion).

WHY: testset17_9filters ran the same 9 filters only on cad11+noapril6 (HIGH-ANGLE,
NOT real-representative; only 4/17 detected → selectivity untestable). The user asked
to also run the filters on the OTHER sets this model was inferred on. filterval is the
PRIMARY real low-angle regime (real_eval accuracy exists but NO filter pass/fail /
selectivity). Same model, same squash-parity decode, same filter helpers verbatim.

REUSE (no filter logic reimplemented):
  - T = paper_s2_testset17_9filters : build_rec / infer_squash / tta_stab_squash /
        env recipe / M(s1_cad_9filters helpers) / TAU / FILTER_ORDER / STRONG / WEAK.
  - S = stage25_paperbase_eval : frames_filterval() -> (dom, fid, jp, ip) with GT.
  - eval_frame_squash (via build_rec): corner_med (order-free Hungarian) / honest8 / GT.

dims = (1.1, 1.3, 0.12) m  (real KS T-11, same as testset17; f7 posdepth / honest8).
GT-good = corner_med < 10px. Selectivity = confusion(filter-accept, GT-good).

flip un-flip = (W-1)-x  (cv2.flip exact inverse; today's principled 1px fix, wood
diag 2026-07-10). f3's ~18px baseline is MODEL L-R non-equivariance (no flip aug),
NOT a coordinate offset — see wood_gt_eval.md flip 조사.

Usage: conda activate pallet-pose; python scripts/stage0/paper_s2/paper_s2_filterval_9filters.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_testset17_9filters as T  # noqa: E402  (build_rec, M, TAU, ...)
import stage25_paperbase_eval as S       # noqa: E402  (frames_filterval)
import cv2  # noqa: E402
import torch  # noqa: E402

M = T.M
TAU = T.TAU
FILTER_ORDER = T.FILTER_ORDER
FILTER_DESC = T.FILTER_DESC
STRONG = T.STRONG
WEAK = T.WEAK
GOOD_PX = T.GOOD_PX
N_DET_MIN = T.N_DET_MIN
E = T.E
WEIGHTS = T.WEIGHTS
OUT_DIR = T.OUT_DIR
OUT_MD = os.path.join(OUT_DIR, "filterval_9filter_check.md")
GROSS_PX = 20.0
DOMAINS = ["outside", "night", "manual"]


# ── principled (W-1) un-flip, consistent with 2026-07-10 wood fix ────────────
def _flip_infer_squash_wm1(model, img_bgr, device):
    H, W = img_bgr.shape[:2]
    _, pred8, pred_c, _, _ = T.infer_squash(model, cv2.flip(img_bgr, 1), device)
    kpf = [None] * 9
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            kpf[i] = ((W - 1) - pred8[i, 0], pred8[i, 1])
    if pred_c is not None:
        kpf[8] = ((W - 1) - pred_c[0], pred_c[1])
    swap = list(range(9))
    for a, b in T.FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return [kpf[swap[i]] for i in range(9)]


T.flip_infer_squash = _flip_infer_squash_wm1  # build_rec resolves at call-time


def confusion(recs, accept_fn):
    tp = fp = fn = tn = 0
    for r in recs:
        acc, good = accept_fn(r), r["good"]
        if acc and good:
            tp += 1
        elif acc and not good:
            fp += 1
        elif (not acc) and good:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, recall=rec, n_acc=tp + fp)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = S.frames_filterval()
    print(f"[filterval] {len(frames)} frames; dims={M.__dict__.get('APNP', '')}; "
          f"weights=Stage B (net_epoch_0057), squash-parity, flip=(W-1)-x")
    import annotate_pnp as APNP
    APNP.PALLET_DIMS = (1.1, 1.3, 0.12)

    model = E.load_model(WEIGHTS, device)
    recs = []
    for dom, fid, jp, ip in frames:
        r = T.build_rec(model, dom, fid, jp, ip, device)
        if r is not None:
            recs.append(r)
        cm = f"{r['corner_med']:.1f}" if (r and r['corner_med'] is not None) else "-"
        print(f"  {dom:<8}{fid}  det={r['n_det'] if r else '-'} cm={cm}")

    # f8 GT-envelope from ALL filterval GT bboxes (same pallet; model-independent)
    gsr = np.array([r["gt_sr"] for r in recs if r["gt_sr"] is not None])
    gasp = np.array([r["gt_asp"] for r in recs if r["gt_asp"] is not None])
    env = {"size_lo": float(np.percentile(gsr, 2.5)),
           "size_hi": float(np.percentile(gsr, 97.5)),
           "asp_lo": float(np.percentile(gasp, 2.5)),
           "asp_hi": float(np.percentile(gasp, 97.5))}

    def accept_filter(f):
        def fn(r):
            if r["n_det"] < N_DET_MIN:
                return False
            row = {"n_det": r["n_det"], "scores": r["scores"],
                   "f7_posdepth": r["f7_posdepth"],
                   "pred_sr": r["pred_sr"], "pred_asp": r["pred_asp"]}
            return bool(M.apply_filter(f, row, TAU.get(f), env))
        return fn

    def accept_combo(fs):
        fns = [accept_filter(f) for f in fs]
        return lambda r: all(g(r) for g in fns)

    ALL9 = FILTER_ORDER
    combos = {
        "COMBO strong(f1&f4&f5&f9)": accept_combo(STRONG),
        "COMBO all9": accept_combo(ALL9),
    }

    def block(rs, title):
        L = [f"### {title}  (N={len(rs)})"]
        det = [r for r in rs if r["n_det"] >= N_DET_MIN]
        good = [r for r in rs if r["good"]]
        gross = [r for r in rs if (r["corner_med"] is None or
                                   (r["corner_med"] is not None and r["corner_med"] > GROSS_PX))]
        L.append("```")
        L.append(f"det(n_det>=6) : {len(det)}/{len(rs)} ({100*len(det)/max(1,len(rs)):.0f}%)   "
                 f"corner_med median(det): {_c(_med([r['corner_med'] for r in det]),'{:.1f}')}px")
        L.append(f"GT-good(<{GOOD_PX:.0f}px): {len(good)}/{len(rs)} ({100*len(good)/max(1,len(rs)):.0f}%)   "
                 f"GT-gross(>{GROSS_PX:.0f}px): {len(gross)}/{len(rs)}")
        L.append("")
        L.append(f"{'filter/combo':<28}{'n_acc':>6}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
                 f"{'Prec':>7}{'Recall':>8}")
        L.append("-" * 65)
        for f in FILTER_ORDER:
            c = confusion(rs, accept_filter(f))
            L.append(f"{f:<28}{c['n_acc']:>6}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>4}"
                     f"{c['prec']:>7.3f}{c['recall']:>8.3f}")
        L.append("-" * 65)
        for name, fn in combos.items():
            c = confusion(rs, fn)
            L.append(f"{name:<28}{c['n_acc']:>6}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>4}"
                     f"{c['prec']:>7.3f}{c['recall']:>8.3f}")
        L.append("```")
        L.append(f"(total GT-good={len(good)}; TP=accept&good FP=accept&bad FN=reject&good TN=reject&bad)")
        L.append("")
        return L

    # ── markdown ─────────────────────────────────────────────────────────
    L = []
    L.append("# Stage B (paper_s2 net_epoch_0057) — 9-filter SELECTIVITY on REAL filterval")
    L.append("")
    L.append(f"- weights: `{os.path.relpath(WEIGHTS, ROOT)}` (squash-parity, THRESH={T.THRESH})")
    L.append("- set: **filterval = outside44 + night43 + manual36 = 123 real frames, LOW-ANGLE**")
    L.append("  (elev -7.7~10.4°; PRIMARY real rear/low-angle regime; final-test sessions SEALED).")
    L.append("- dims (W,D,H) = (1.1, 1.3, 0.12) m (real KS T-11). corner_med = order-free Hungarian.")
    L.append("- filters reused verbatim from `s1_cad_9filters.py` (M.apply_filter/TAU); flip=(W-1)-x.")
    L.append("- tau: " + ", ".join(f"{k}={v}" for k, v in TAU.items()) + "; f7 posdepth(bool); f8 GT size-env.")
    L.append(f"- f8 GT size-env(p2.5-97.5): size={env['size_lo']:.3f}-{env['size_hi']:.3f}, "
             f"asp={env['asp_lo']:.2f}-{env['asp_hi']:.2f}")
    L.append(f"- strong={STRONG}  weak={WEAK}")
    L.append("")
    L.append("## Selectivity: filter-accept vs GT-good (corner_med<10px)")
    L.append("")
    L += block(recs, "OVERALL filterval")
    for dm in DOMAINS:
        L += block([r for r in recs if r["dom"] == dm], f"domain = {dm}")

    # per-frame (detected only, compact)
    L.append("## per-frame (detected n_det>=6)")
    L.append("```")
    hdr = (f"{'dom':<8}{'fid':<14}{'V':>3}{'det':>4}{'cmed':>6}{'hon8':>6}"
           f"{'f1':>5}{'f3':>5}{'f4':>5}{'f9':>5}{'#p':>3} acc GTg")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in sorted([r for r in recs if r["n_det"] >= N_DET_MIN],
                    key=lambda r: (r["dom"], r["corner_med"] or 1e9)):
        row = {"n_det": r["n_det"], "scores": r["scores"], "f7_posdepth": r["f7_posdepth"],
               "pred_sr": r["pred_sr"], "pred_asp": r["pred_asp"]}
        npass = sum(1 for f in FILTER_ORDER if M.apply_filter(f, row, TAU.get(f), env))
        acc = accept_combo(STRONG)(r)
        s = r["scores"]
        L.append(f"{r['dom']:<8}{str(r['fid'])[:13]:<14}{r['v_geom']:>3}{r['n_det']:>4}"
                 f"{_c(r['corner_med'],'{:.1f}'):>6}{_c(r['honest8'],'{:.1f}'):>6}"
                 f"{_c(s.get('f1_peak'),'{:.2f}'):>5}{_c(s.get('f3_flip'),'{:.0f}'):>5}"
                 f"{_c(s.get('f4_tta_stab'),'{:.1f}'):>5}{_c(s.get('f9_bbox_iou'),'{:.2f}'):>5}"
                 f"{npass:>3}{'  Y' if acc else '  n'}{'  Y' if r['good'] else '  n'}")
    L.append("```")
    L.append("")
    L.append("★ caveat: real low-angle domain-mixed; heuristic tau; convention caveats. f3_flip "
             "baseline ~18px = model L-R non-equivariance (no flip aug), NOT coord bug (wood diag).")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_MD}")


def _c(v, fmt="{:.2f}"):
    return "  -  " if v is None else fmt.format(v)


if __name__ == "__main__":
    main()
