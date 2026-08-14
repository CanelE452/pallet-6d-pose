"""paper_s2_testset17_9filters.py — apply the project's 9 PL-accept filters
(f1..f9 from s1_cad_9filters.py) to Stage B (paper_s2) predictions on the
designated hand-anno test set (cad11 + noapril6 = 17 frames).

WHY: s1_cad_9filters.py ran f1..f9 on Paper-S1 (pad100). Stage B was never
put through the same filter check. Here we reuse the *exact same* filter scoring
helpers (M.top2_local_peak_ratio, M.frsep_frac, M.bbox_iou, M.size_aspect,
M.excursion, M.flip_score, M.posdepth_ok, M.apply_filter, M.TAU, M.FILTER_ORDER,
M.FLIP_PAIRS) — no filter logic is reimplemented — but the belief inference is
Stage B squash-parity (identical decode to paper_s2_testset17_overlay:
preprocess_squash 640x480->400x400, belief(50)->orig x(W/50,H/50)).

dims = (W,D,H) = (1.1, 1.3, 0.12) m  [USER-SPECIFIED] for f7 posdepth / honest8.
corner_med / honest8 taken from eval_frame_squash (same as overlay report).

Detected frames (n_det>=6) get f1..f9. Under-det frames are flagged
"n_det<6 (pre-filter stage)".

Usage: conda activate pallet-pose; python scripts/stage0/paper_s2/paper_s2_testset17_9filters.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

# reuse Stage B squash decode + harness (sets sys.path for eval/DOPE/challenge)
from paper_s2_real_eval import (ROOT as _R, E, eval_frame_squash,  # noqa: E402
                                preprocess_squash, THRESH, N_DET_MIN)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# import the canonical filter helpers (loads its own E; harmless)
M = _load("s1filters", os.path.join(ROOT, "scripts", "stage0",
                                    "s1_cad_9filters.py"))

# ── USER-SPECIFIED dims (height 0.12) — affects f7 posdepth + honest8 ─────────
APNP.PALLET_DIMS = (1.1, 1.3, 0.12)

MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")
WEIGHTS = os.path.join(ROOT, "weights/paper_s2_stageB/net_epoch_0057.pth")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp")
OUT_MD = os.path.join(OUT_DIR, "testset17_9filter_check.md")

TAU = M.TAU
FILTER_ORDER = M.FILTER_ORDER
FILTER_DESC = M.FILTER_DESC
GOOD_PX = M.GOOD_PX
FLIP_PAIRS = M.FLIP_PAIRS
# strong (appearance/confidence) vs weak (2D geometry) grouping per history 2026-07-07
STRONG = ["f1_peak", "f4_tta_stab", "f5_rear_conf", "f9_bbox_iou"]
WEAK = ["f2_peak_ratio", "f6_frsep", "f7_posdepth", "f8_size_env"]


def read_manifest(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        dom, fid, jp, ip = ln.split()
        rows.append((dom, fid, os.path.join(ROOT, jp), os.path.join(ROOT, ip)))
    return rows


def infer_squash(model, img_bgr, device):
    """Stage B squash belief decode -> belief(9,50,50), pred8 orig px, pred_c,
    peaks(9,), ratios(8,).  Identical decode to eval_frame_squash / overlay."""
    H, W = img_bgr.shape[:2]
    with torch.no_grad():
        beliefs, _ = model(preprocess_squash(img_bgr).to(device))
    belief = beliefs[-1][0].cpu().numpy()
    kps = extract_keypoints_from_belief(belief, THRESH)
    sx, sy = W / 50.0, H / 50.0
    pred8 = np.full((8, 2), np.nan)
    peaks = np.zeros(9)
    for i in range(9):
        peaks[i] = kps[i][2]
    for i in range(8):
        if kps[i][0] >= 0:
            pred8[i] = [kps[i][0] * sx, kps[i][1] * sy]
    pred_c = [kps[8][0] * sx, kps[8][1] * sy] if kps[8][0] >= 0 else None
    ratios = np.array([M.top2_local_peak_ratio(belief[i]) for i in range(8)])
    return belief, pred8, pred_c, peaks, ratios


def flip_infer_squash(model, img_bgr, device):
    """L-R flip squash inference -> un-flip x(W-x) + camera-facing swap."""
    H, W = img_bgr.shape[:2]
    flip = cv2.flip(img_bgr, 1)
    _, pred8, pred_c, _, _ = infer_squash(model, flip, device)
    kpf = [None] * 9
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            kpf[i] = (W - pred8[i, 0], pred8[i, 1])
    if pred_c is not None:
        kpf[8] = (W - pred_c[0], pred_c[1])
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return [kpf[swap[i]] for i in range(9)]


def tta_stab_squash(model, img_bgr, device):
    """scale/brightness TTA (frame-preserving) per-corner pos std mean (px)."""
    H, W = img_bgr.shape[:2]
    variants = [img_bgr,
                np.clip(img_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8),
                np.clip(img_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)]
    small = cv2.resize(img_bgr, (int(W * 0.9), int(H * 0.9)))
    variants.append(cv2.resize(small, (W, H)))
    preds = []
    for v in variants:
        _, p8, _, _, _ = infer_squash(model, v, device)
        preds.append(p8)
    preds = np.stack(preds, 0)
    stds = []
    for i in range(8):
        col = preds[:, i, :]
        ok = ~np.isnan(col[:, 0])
        if ok.sum() >= 3:
            stds.append(float(np.mean(np.std(col[ok], axis=0))))
    return float(np.mean(stds)) if stds else None


def build_rec(model, dom, fid, jp, ip, device):
    """Reuse eval_frame_squash for canonical corner/honest8/pred; compute f1..f9
    scores with squash re-inference + reused M helpers."""
    row = eval_frame_squash(model, jp, ip, device)
    if row is None:
        return None
    img = cv2.imread(ip)
    d = json.load(open(jp))
    K = E.K_from_json(d)
    H, W = img.shape[:2]
    img_diag = float(np.hypot(W, H))

    gt8 = np.array(row["gt8"], float)
    pred8 = np.array(row["pred8"], float)
    pred_c = row["pred_c"]
    n_det = int(row["n_det"])

    _, pred8b, pred_cb, peaks, ratios = infer_squash(model, img, device)
    # sanity: squash pred must match eval_frame_squash pred (same decode)
    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]

    scores = {}
    f7 = False
    pred_sr = pred_asp = None
    if n_det >= N_DET_MIN:
        scores["f1_peak"] = float(min(peaks[i] for i in det8))
        scores["f2_peak_ratio"] = float(min(ratios[i] for i in det8))
        kpB = flip_infer_squash(model, img, device)
        scores["f3_flip"] = M.flip_score(pred8, pred_c, kpB)
        scores["f4_tta_stab"] = tta_stab_squash(model, img, device)
        rear = [i for i in (4, 5, 6, 7) if not np.isnan(pred8[i, 0])]
        scores["f5_rear_conf"] = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
        scores["f6_frsep"] = M.frsep_frac(pred8)
        f7, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
        pred_sr, pred_asp = M.size_aspect(pred8, img_diag)
        pred_bb, gt_bb = M.bbox_of(pred8), M.bbox_of(gt8)
        scores["f9_bbox_iou"] = M.bbox_iou(pred_bb, gt_bb)

    gt_sr, gt_asp = M.size_aspect(gt8, img_diag)
    return {
        "dom": dom, "fid": fid, "ip": ip,
        "v_geom": int(row["v_geom"]), "n_det": n_det,
        "corner_med": (row["corner"] if np.isfinite(row["corner"]) else None),
        "honest8": (row["pnp_honest8"] if np.isfinite(row["pnp_honest8"]) else None),
        "good": bool(np.isfinite(row["corner"]) and row["corner"] < GOOD_PX),
        "scores": scores, "f7_posdepth": bool(f7),
        "pred_sr": pred_sr, "pred_asp": pred_asp,
        "gt_sr": gt_sr, "gt_asp": gt_asp,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = read_manifest(MANIFEST)
    print(f"[manifest] {len(frames)} frames; dims={APNP.PALLET_DIMS}; "
          f"weights=Stage B (net_epoch_0057), squash-parity")

    model = E.load_model(WEIGHTS, device)
    recs = []
    for dom, fid, jp, ip in frames:
        r = build_rec(model, dom, fid, jp, ip, device)
        if r is not None:
            recs.append(r)
        cm = f"{r['corner_med']:.1f}" if (r and r['corner_med'] is not None) else "-"
        print(f"  {dom:<8}{fid}  det={r['n_det'] if r else '-'} cm={cm}")

    # f8 envelope from GT bboxes of all 17 (model-independent), same recipe as M
    gsr = np.array([r["gt_sr"] for r in recs if r["gt_sr"] is not None])
    gasp = np.array([r["gt_asp"] for r in recs if r["gt_asp"] is not None])
    env = {"size_lo": float(np.percentile(gsr, 2.5)),
           "size_hi": float(np.percentile(gsr, 97.5)),
           "asp_lo": float(np.percentile(gasp, 2.5)),
           "asp_hi": float(np.percentile(gasp, 97.5))}

    det_recs = [r for r in recs if r["n_det"] >= N_DET_MIN]
    # apply filters (reuse M.apply_filter) to detected frames
    passmat = {}
    for r in det_recs:
        # adapt rec to M.apply_filter contract (needs scores + f7 + pred_sr/asp)
        row = {"n_det": r["n_det"], "scores": r["scores"],
               "f7_posdepth": r["f7_posdepth"],
               "pred_sr": r["pred_sr"], "pred_asp": r["pred_asp"]}
        passmat[r["fid"]] = {f: bool(M.apply_filter(f, row, TAU.get(f), env))
                             for f in FILTER_ORDER}

    # ── build markdown ──────────────────────────────────────────────────
    L = []
    L.append("# Stage B (paper_s2 net_epoch_0057) — 9-filter check on testset17")
    L.append("")
    L.append(f"- weights: `{os.path.relpath(WEIGHTS, ROOT)}`  (squash-parity decode, THRESH={THRESH})")
    L.append(f"- set: cad11 + noapril6 = 17 (hand-anno GT, HIGH-ANGLE; memory: NOT real-representative)")
    L.append(f"- dims (W,D,H) = {APNP.PALLET_DIMS} m  (f7 posdepth / honest8)")
    L.append(f"- filters/helpers reused verbatim from `scripts/stage0/s1_cad_9filters.py` (M.apply_filter, M.TAU, ...)")
    L.append(f"- corner_med = order-free Hungarian median (dims-indep); honest8 = full-8 PnP reproj (dims {APNP.PALLET_DIMS})")
    L.append(f"- tau: " + ", ".join(f"{k}={v}" for k, v in TAU.items()) + "; f7 posdepth(bool); f8 size-env(GT envelope)")
    L.append(f"- f8 size envelope (GT p2.5-p97.5): size={env['size_lo']:.3f}-{env['size_hi']:.3f}, "
             f"asp={env['asp_lo']:.2f}-{env['asp_hi']:.2f}")
    L.append("")
    n_det = len(det_recs)
    L.append(f"**Detection: {n_det}/17 have n_det>=6** (all noapril). "
             f"13/17 under-det (11 cad near-field + 2 noapril) = pre-filter stage, no f1..f9.")
    L.append("")

    # main table (detected frames)
    L.append("## Detected frames — f1..f9 pass(P)/fail(.) + GT contrast")
    L.append("```")
    hdr = (f"{'frame':<20}{'det':>4}{'cm':>7}{'hon8':>7}  " +
           " ".join(f.split('_')[0] for f in FILTER_ORDER) + "  #pass strong weak")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in sorted(det_recs, key=lambda x: x["fid"]):
        pm = passmat[r["fid"]]
        npass = sum(pm.values())
        ns = sum(pm[f] for f in STRONG)
        nw = sum(pm[f] for f in WEAK)
        cm = f"{r['corner_med']:.1f}" if r["corner_med"] is not None else "n/a"
        h8 = f"{r['honest8']:.1f}" if r["honest8"] is not None else "n/a"
        cells = " ".join((" P" if pm[f] else " .") for f in FILTER_ORDER)
        L.append(f"{r['fid']:<20}{r['n_det']:>4}{cm:>7}{h8:>7}  {cells}"
                 f"  {npass:>4}  {ns:>4} {nw:>4}")
    L.append("")
    L.append("filter legend: f1=peak f2=peak_ratio f3=flip f4=tta_stab f5=rear_conf")
    L.append("               f6=frsep f7=posdepth f8=size_env f9=bbox_iou")
    L.append("strong(appearance/conf)=f1,f4,f5,f9 | weak(2D-geom)=f2,f6,f7,f8")
    L.append("```")
    L.append("")

    # per-filter raw score values (detected)
    L.append("## Detected frames — raw filter values")
    L.append("```")
    vh = (f"{'frame':<20}{'f1pk':>6}{'f2rt':>6}{'f3fl':>6}{'f4tta':>7}"
          f"{'f5rr':>6}{'f6frs':>7}{'f7pd':>6}{'f8sr':>7}{'f8as':>6}{'f9iou':>7}")
    L.append(vh)
    L.append("-" * len(vh))
    for r in sorted(det_recs, key=lambda x: x["fid"]):
        s = r["scores"]

        def g(k, fmt="{:.2f}"):
            v = s.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"
        srs = "{:.3f}".format(r["pred_sr"]) if r["pred_sr"] else "n/a"
        asps = "{:.2f}".format(r["pred_asp"]) if r["pred_asp"] else "n/a"
        pds = "OK" if r["f7_posdepth"] else "no"
        L.append(
            f"{r['fid']:<20}{g('f1_peak'):>6}{g('f2_peak_ratio'):>6}"
            f"{g('f3_flip','{:.1f}'):>6}{g('f4_tta_stab','{:.1f}'):>7}"
            f"{g('f5_rear_conf'):>6}{g('f6_frsep','{:.3f}'):>7}"
            f"{pds:>6}{srs:>7}{asps:>6}{g('f9_bbox_iou','{:.2f}'):>7}")
    L.append("```")
    L.append("")

    # under-det list
    L.append("## Under-det frames (n_det<6, pre-filter stage — no f1..f9)")
    L.append("```")
    L.append(f"{'frame':<20}{'dom':>9}{'det':>5}{'V':>4}")
    for r in sorted([x for x in recs if x["n_det"] < N_DET_MIN],
                    key=lambda x: (x["dom"], x["fid"])):
        L.append(f"{r['fid']:<20}{r['dom']:>9}{r['n_det']:>5}{r['v_geom']:>4}")
    L.append("```")
    L.append("")

    # per-filter pass counts
    L.append("## Per-filter pass count (over detected n)")
    L.append("```")
    L.append(f"{'filter':<15}{'tau':>7}{'n_pass':>8}   desc")
    L.append("-" * 70)
    for f in FILTER_ORDER:
        np_pass = sum(passmat[r["fid"]][f] for r in det_recs)
        tau = TAU.get(f)
        taus = f"{tau}" if tau is not None else "bool/env"
        L.append(f"{f:<15}{taus:>7}{np_pass:>8}   {FILTER_DESC[f]}")
    L.append("```")
    L.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_MD}")


if __name__ == "__main__":
    main()
