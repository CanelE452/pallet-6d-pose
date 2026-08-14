"""stage19_partA_infer.py — B2 confidence/filter diagnostics (INFERENCE ONLY).

Records, per frame, PER-KEYPOINT signals for the B2 best model
(stage11_16k_B2_maskaux/final_net_epoch_0084) so a downstream CPU analyzer can
build calibration (signal<->GT-error Spearman), keypoint-level PR curves, and a
re-measured self-training funnel.  NO training.  final-test SEALED.

Signals recorded per keypoint (camera-facing 0123; front 0-3, rear 4-7, 8=centroid):
  peak      : belief-map peak value (top-1 local max)  [per-channel, 9]
  peak2     : 2nd-highest NMS local max                 [per-channel, 9]
  peakratio : peak / max(peak2, eps)                    [per-channel, 9]
  flipTTA   : L-R flip TTA per-index inconsistency (px, reflect-pad path, un-flip+swap) [9]
  loo       : leave-one-out SQPnP reprojection residual (px) [8, dims-known only]
Frame-level:
  diag_resid: centroid(8) point-to-line distance to the 4 SPACE_DIAGs,
              normalized by mean space-diagonal length (NO intersection calc —
              edge-on numeric blow-up avoided).  [scalar]
GT (dims-known sets only):
  gt_err9   : channel-aligned per-corner |pred-GT| px [8]
  hungarian : order-free 8-corner mean match px (frame good judgment)

Reuses (no new geometry / model code):
  eval_pvnet_heads.load_pvnet_model  (B2 aux seg head)
  dope_predict_mp4_pad.pad_frame     (reflect pad=100, matches B2 cad eval)
  filter_pr_camfacing.extract_keypoints_from_belief / canonical_kp3d / sqpnp
  filter_flip_consistency.FLIP_PAIRS (L-R symmetric corner swap)
  tau_calibrate.collect_val_frames (SEAL guard)  eval_pvnet_heads.collect_manual
  corner01_diagnosis.collect_cad

Frame sets:
  filter-val (GT, dims) 86 | manual (GT, dims) 36 | cad (GT, dims) 22
  pool_noapril (no GT, yield only) c0403noapril/rgb 188

Output: data/pallet/eval_results/stage19_conf_mixup/partA_records.json
CPU smoke: --list_only
GPU: conda run -n pallet-pose python scripts/stage0/stage19_partA_infer.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from tau_calibrate import collect_val_frames  # noqa: E402
from eval_pvnet_heads import collect_manual  # noqa: E402
from corner01_diagnosis import collect_cad  # noqa: E402
from filter_flip_consistency import FLIP_PAIRS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage19_conf_mixup")
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux",
                               "final_net_epoch_0084.pth")
NOAPRIL_RGB = os.path.join(ROOT, "data", "pallet", "raw_data",
                           "capture0403noapril", "rgb")
EXCLUDE_FP = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
PAD = 100
THRESHOLD = 0.3
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SPACE_DIAG = [(0, 6), (1, 7), (2, 4), (3, 5)]
FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]


def load_exclude():
    ex = set()
    if os.path.exists(EXCLUDE_FP):
        for ln in open(EXCLUDE_FP):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def build_swap():
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return swap


def collect_pool_noapril():
    out = []
    for ip in sorted(glob.glob(os.path.join(NOAPRIL_RGB, "*.png"))):
        fid = os.path.splitext(os.path.basename(ip))[0]
        out.append(("noapril", fid, None, ip))
    return out


# ── belief peaks: top-1 + top-2 NMS local maxima per channel ──────────────
def belief_peaks(bmap, threshold, sigma=2):
    from scipy.ndimage import gaussian_filter
    if bmap.max() < threshold:
        return 0.0, 0.0
    sm = gaussian_filter(bmap, sigma=sigma)
    p = 1
    pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
    pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
    pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
    pd = np.zeros_like(sm); pd[:, :-p] = sm[:, p:]
    peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return float(bmap.max()), 0.0
    vals = np.sort([bmap[y, x] for y, x in zip(ys, xs)])[::-1]
    top1 = float(vals[0])
    top2 = float(vals[1]) if len(vals) > 1 else 0.0
    return top1, top2


# ── point-to-line diag residual (no intersection) ─────────────────────────
def diag_residual(kp9):
    """centroid(8) mean normalized point-to-line dist to the 4 space diagonals."""
    if kp9[8] is None:
        return None
    c = np.asarray(kp9[8], float)
    dists, dlens = [], []
    for a, b in SPACE_DIAG:
        if kp9[a] is None or kp9[b] is None:
            continue
        pa = np.asarray(kp9[a], float); pb = np.asarray(kp9[b], float)
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        # perpendicular distance from c to line through pa,pb
        perp = abs(d[0] * (pa[1] - c[1]) - d[1] * (pa[0] - c[0])) / L
        dists.append(perp)
        dlens.append(L)
    if not dists:
        return None
    return float(np.mean(dists) / max(np.mean(dlens), 1e-6))


def load_gt(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    dm = o.get("dimensions_m", {"width": 1.3, "depth": 1.1, "height": 0.11})
    intr = d.get("camera_data", {}).get("intrinsics")
    return gt8, dm, intr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--pad", type=int, default=PAD)
    ap.add_argument("--list_only", action="store_true")
    args = ap.parse_args()

    exclude = load_exclude()
    fval = [f for f in collect_val_frames() if f[1] not in exclude]
    manual = collect_manual()
    cad = collect_cad()
    pool = collect_pool_noapril()
    # is_gt: filter-val/manual/cad have GT+dims; noapril pool does not
    frames = ([("outside" if d == "outside" else "night", f, j, i, True)
               for d, f, j, i in fval]
              + [(d, f, j, i, True) for d, f, j, i in manual]
              + [(d, f, j, i, True) for d, f, j, i in cad]
              + [(d, f, j, i, False) for d, f, j, i in pool])
    by = {}
    for d, *_ , g in frames:
        by[d] = by.get(d, 0) + 1
    print(f"[frames] total={len(frames)} {by}")
    gtn = sum(1 for *_, g in frames if g)
    print(f"[GT frames={gtn}  pool(no-GT)={len(frames) - gtn}]")
    if args.list_only:
        seal = {"capturepallet09", "capturepallet07", "capturenight09",
                "capturenight08"}
        leaks = [f for d, f, j, i, g in frames if j and any(s in j for s in seal)]
        print(f"[SEAL] final-test leaks: {len(leaks)}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    import cv2
    import torch
    from filter_pr_camfacing import (extract_keypoints_from_belief,
                                      canonical_kp3d, sqpnp)
    from eval_pvnet_heads import load_pvnet_model
    from dope_predict_mp4_pad import pad_frame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model, numVec, numSeg = load_pvnet_model(args.weights, device)
    print(f"[model] numVec={numVec} numSeg={numSeg}")
    swap = build_swap()

    def infer_full(img, pad, threshold):
        """reflect-pad inference. Returns kp9 (orig px, channel-aligned, None if
        undetected), peak9, peak2_9. Mapping == corner01_diagnosis.infer_one."""
        h0, w0 = img.shape[:2]
        proc = pad_frame(img, pad, "reflect")
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        ph, pw = proc.shape[:2]
        _sc = 400.0 / min(ph, pw)
        _nw = max(8, int(round(pw * _sc)) & ~7)
        _nh = max(8, int(round(ph * _sc)) & ~7)
        t = ((cv2.resize(rgb, (_nw, _nh)).astype(np.float32) / 255.0 - MEAN) / STD)
        tensor = (torch.from_numpy(t.transpose(2, 0, 1)).float()
                  .unsqueeze(0).to(device))
        with torch.no_grad():
            out = model(tensor)
        belief = out[0][-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, threshold)
        bh, bw = belief.shape[1], belief.shape[2]
        ux, uy = _nw / bw, _nh / bh
        kp9 = [None] * 9
        peak9 = [0.0] * 9
        peak2_9 = [0.0] * 9
        for i, k in enumerate(kps_bel):
            p1, p2 = belief_peaks(belief[i], threshold)
            peak9[i] = p1; peak2_9[i] = p2
            if k[0] < 0:
                continue
            cx = (k[0] * ux) / _sc
            cy = (k[1] * uy) / _sc
            ox = cx * (w0 + 2 * pad) / w0 - pad
            oy = cy * (h0 + 2 * pad) / h0 - pad
            kp9[i] = (float(ox), float(oy))
        return kp9, peak9, peak2_9

    def infer_flip9(img, pad, threshold):
        """L-R flip TTA through the SAME reflect-pad path; return kp9 un-flipped
        + label-swapped to canonical camera-facing indexing (orig px)."""
        h0, w0 = img.shape[:2]
        img_f = cv2.flip(img, 1)
        kpf, _, _ = infer_full(img_f, pad, threshold)   # px on flipped image
        unflip = [None] * 9
        for i in range(9):
            if kpf[i] is not None:
                unflip[i] = (w0 - kpf[i][0], kpf[i][1])
        out = [None] * 9
        for i in range(9):
            out[i] = unflip[swap[i]]
        return out

    def loo_reproj(kp9, kp3d, K):
        """Per-corner leave-one-out SQPnP reprojection residual (px)."""
        dist = np.zeros((5, 1))
        det = [i for i in range(8) if kp9[i] is not None]
        res = [None] * 8
        if len(det) < 6:
            return res
        import cv2 as _cv
        for li in det:
            keep = [i for i in det if i != li]
            if len(keep) < 5:
                continue
            obj = kp3d[keep].reshape(-1, 1, 3)
            img = np.array([[kp9[i][0], kp9[i][1]] for i in keep],
                           float).reshape(-1, 1, 2)
            ok, rvec, tvec = _cv.solvePnP(obj, img, K, dist,
                                          flags=_cv.SOLVEPNP_SQPNP)
            if not ok:
                continue
            proj, _ = _cv.projectPoints(kp3d[li].reshape(1, 3), rvec, tvec,
                                        K, dist)
            res[li] = float(np.linalg.norm(proj.reshape(2) -
                                           np.array(kp9[li], float)))
        return res

    from scipy.optimize import linear_sum_assignment
    records = []
    for n, (dom, fid, jp, ip, is_gt) in enumerate(frames):
        img = cv2.imread(ip)
        if img is None:
            continue
        kp9, peak9, peak2_9 = infer_full(img, args.pad, args.threshold)
        n_det = sum(1 for i in range(8) if kp9[i] is not None)
        if n_det < 6:
            continue
        kpB = infer_flip9(img, args.pad, args.threshold)
        flip9 = [None] * 9
        for i in range(9):
            if kp9[i] is not None and kpB[i] is not None:
                flip9[i] = float(np.linalg.norm(np.asarray(kp9[i], float) -
                                                np.asarray(kpB[i], float)))
        diag_r = diag_residual(kp9)
        peakratio9 = [round(peak9[i] / max(peak2_9[i], 1e-3), 3) for i in range(9)]

        rec = {
            "dom": dom, "fid": str(fid), "is_gt": bool(is_gt),
            "n_det": int(n_det),
            "kp9": [list(p) if p is not None else None for p in kp9],
            "peak9": [round(v, 4) for v in peak9],
            "peak2_9": [round(v, 4) for v in peak2_9],
            "peakratio9": peakratio9,
            "flip9": flip9, "diag_resid": diag_r,
        }
        if is_gt:
            gt8, dm, intr = load_gt(jp)
            kp3d = canonical_kp3d(dm["width"], dm["depth"], dm["height"])
            # channel-aligned per-corner GT err
            e = [None] * 8
            for i in range(8):
                if kp9[i] is not None:
                    e[i] = float(np.linalg.norm(np.asarray(kp9[i], float) - gt8[i]))
            rec["gt_err9"] = e
            # order-free hungarian frame good
            valid = [i for i in range(8) if kp9[i] is not None]
            P = np.array([kp9[i] for i in valid], float)
            cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            rec["hungarian"] = float(cost[ri, ci].mean())
            if intr is not None:
                K = np.array([[intr["fx"], 0, intr["cx"]],
                              [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
                rec["loo9"] = loo_reproj(kp9, kp3d, K)
        records.append(rec)
        if (n + 1) % 50 == 0:
            print(f"  [{n+1}/{len(frames)}] records={len(records)}")

    out_fp = os.path.join(OUT_DIR, "partA_records.json")
    json.dump({"weights": args.weights, "pad": args.pad,
               "threshold": args.threshold, "n": len(records),
               "records": records}, open(out_fp, "w"), indent=2)
    print(f"[save] {out_fp}  ({len(records)} records)")


if __name__ == "__main__":
    main()
