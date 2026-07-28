"""stage0/padding_edge_overlay.py — visual mechanism check for padding ablation.

Why the padding ablation hurts (paper_base_v2_s2/ep66: none >> reflect on
outside real). Two hypotheses we test BY EYE on edge frames:
  (H1) "fake pallet"  : reflect padding mirrors a pallet sliver at the image
                        border -> model fires a peak on the mirror -> wrong.
  (H2) "zoom-out"     : pad+resize shrinks the pallet in the model input
                        canvas -> predictions get globally smaller / shifted
                        -> wrong, with NO border-region peaks.

For each EDGE frame (GT cuboid bbox within EDGE_PX of an image border or
truncated outside) we save a side-by-side panel:
  A: orig + GT(green) + none pred (red, indexed)        "none gt_err n"
  B: reflect CANVAS (the padded-resized image the model ACTUALLY sees) +
     reflect belief peaks in CANVAS coords (red, indexed)  "reflect CANVAS"
     -> the border reflection sliver is visible here; if a peak lands on it
        that is a "fake pallet" detection (H1).
  C: orig + GT(green) + reflect pred (red, ORIGINAL coords)  "reflect gt_err n"
  D: orig + GT(green) + black   pred (red, ORIGINAL coords)  "black gt_err n"

Reuses verified pieces only (NO new geometry):
  padding_compare.infer_one math (reproduced inline so we can ALSO return the
    canvas (cx,cy) peaks = undo-pad-BEFORE coords, for panel B), self-checked
    against infer_one's gt_err/n_det per frame.
  dope_predict_mp4_pad.pad_frame  (reflect/black canvas)
  filter_pr_camfacing : load_model, extract_keypoints_from_belief,
                        hungarian_mean_dist
  tau_calibrate.collect_val_frames (filter-val GT frames)
  visualize_inference.KP_COLORS

GPU script. Usage:
  conda run -n pallet-pose python scripts/stage0/padding_edge_overlay.py
CPU-safe edge selection only (no model):
  python scripts/stage0/padding_edge_overlay.py --list_only
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

from tau_calibrate import collect_val_frames  # noqa: E402
from padding_compare import load_gt8, infer_one  # noqa: E402

EDGE_PX = 40.0
N_DET_MIN = 6
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "paper_base_v2_s2",
                               "net_epoch_0066.pth")


def edge_report(gt8, w0, h0, edge_px=EDGE_PX):
    """Return (is_edge, dist_to_nearest_border_px, sides_list).

    sides_list names borders that are within edge_px OR truncated (GT outside).
    dist = min over the 8 GT corners' distance to nearest image border; <0 means
    a corner sits outside the frame (truncated)."""
    xs, ys = gt8[:, 0], gt8[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    d_left, d_right = x0, (w0 - 1) - x1
    d_top, d_bot = y0, (h0 - 1) - y1
    dist = min(d_left, d_right, d_top, d_bot)
    sides = []
    for name, d in (("left", d_left), ("right", d_right),
                    ("top", d_top), ("bottom", d_bot)):
        if d < 0:
            sides.append(name + "(trunc)")
        elif d <= edge_px:
            sides.append(name + f"({d:.0f}px)")
    return (len(sides) > 0), float(dist), sides


def canvas_and_orig_peaks(model, extract_kp, hungarian, torch, cv2, img, gt8,
                          pad, mode, threshold, device):
    """Reproduce padding_compare.infer_one EXACTLY but ALSO return the canvas
    (pre-undo) peaks for panel B. Returns (gt_err, n_det, pred8_orig, canvas8,
    proc). canvas8/pred8_orig are (8,2) NaN-filled. canvas8 = (cx,cy) on proc
    (the model-input image); pred8_orig = (ox,oy) on the original frame.

    [확인] math is copy of padding_compare.infer_one lines 88-124."""
    from dope_predict_mp4_pad import pad_frame
    h0, w0 = img.shape[:2]
    use_pad = pad if mode != "none" else 0
    proc = pad_frame(img, use_pad, mode)  # padded-resized canvas, same HxW
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    ph, pw = proc.shape[:2]
    _sc = 400.0 / min(ph, pw)
    _nw = max(8, int(round(pw * _sc)) & ~7)
    _nh = max(8, int(round(ph * _sc)) & ~7)
    t = ((cv2.resize(rgb, (_nw, _nh)).astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps_bel = extract_kp(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    ux, uy = _nw / bw, _nh / bh

    pred8 = np.full((8, 2), np.nan)
    canvas8 = np.full((8, 2), np.nan)
    n_det = 0
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        cx = (k[0] * ux) / _sc  # belief -> proc(canvas) coords
        cy = (k[1] * uy) / _sc
        canvas8[i] = (cx, cy)
        if use_pad > 0:
            ox = cx * (w0 + 2 * use_pad) / w0 - use_pad
            oy = cy * (h0 + 2 * use_pad) / h0 - use_pad
        else:
            ox, oy = cx, cy
        pred8[i] = (ox, oy)
        n_det += 1
    gt_err, _ = hungarian(pred8, gt8)
    return gt_err, n_det, pred8, canvas8, proc


def draw_pred(vis, pred8, cv2, kp_colors, radius=6):
    for i in range(8):
        if np.isnan(pred8[i, 0]):
            continue
        pt = (int(round(pred8[i, 0])), int(round(pred8[i, 1])))
        cv2.circle(vis, pt, radius, kp_colors[i], -1)
        cv2.circle(vis, pt, radius + 1, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (pt[0] + 7, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, kp_colors[i], 1, cv2.LINE_AA)


def draw_gt(vis, gt8, cv2):
    for i in range(8):
        pt = (int(round(gt8[i, 0])), int(round(gt8[i, 1])))
        cv2.circle(vis, pt, 4, (0, 200, 0), -1)
        cv2.circle(vis, pt, 5, (0, 0, 0), 1)


def label(vis, text, cv2):
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--pad", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--edge_px", type=float, default=EDGE_PX)
    ap.add_argument("--max_frames", type=int, default=6)
    ap.add_argument("--domain", default="outside")
    ap.add_argument("--list_only", action="store_true")
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "stage0_padding_compare",
        "edge_overlays"))
    args = ap.parse_args()

    frames = [f for f in collect_val_frames() if f[0] == args.domain]
    print(f"[frames] {len(frames)} {args.domain} filter-val GT frames")

    # rank frames by edge status: edge frames first (sides!=[]), then by dist asc
    scored = []
    import cv2
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        gt8 = load_gt8(jp)
        is_edge, dist, sides = edge_report(gt8, w0, h0, args.edge_px)
        scored.append((dom, fid, jp, ip, is_edge, dist, sides))
    # edge frames (is_edge True) by smallest dist first; then non-edge by dist
    scored.sort(key=lambda r: (not r[4], r[5]))
    sel = scored[:args.max_frames]

    print(f"\n[edge selection] edge_px={args.edge_px}  picked {len(sel)}:")
    for dom, fid, jp, ip, is_edge, dist, sides in sel:
        tag = "EDGE" if is_edge else "near"
        print(f"  {tag}  {fid}  dist={dist:6.1f}px  sides={sides}")

    if args.list_only:
        return

    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, hungarian_mean_dist,
    )
    from visualize_inference import KP_COLORS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[load] {args.weights} ({device})")
    model = load_model(args.weights, device)

    os.makedirs(args.output_dir, exist_ok=True)
    saved = []
    print("\n" + "=" * 78)
    for dom, fid, jp, ip, is_edge, dist, sides in sel:
        img = cv2.imread(ip)
        gt8 = load_gt8(jp)

        # none (PAD=0): pred orig == canvas
        e_none, n_none, p_none, _, _ = canvas_and_orig_peaks(
            model, extract_keypoints_from_belief, hungarian_mean_dist,
            torch, cv2, img, gt8, args.pad, "none", args.threshold, device)
        # self-check vs infer_one (do not change infer_one)
        e_none_chk, n_none_chk, _ = infer_one(
            model, extract_keypoints_from_belief, hungarian_mean_dist,
            torch, cv2, img, gt8, args.pad, "none", args.threshold, device)
        # reflect: orig pred + canvas peaks + canvas image (proc)
        e_ref, n_ref, p_ref, c_ref, proc_ref = canvas_and_orig_peaks(
            model, extract_keypoints_from_belief, hungarian_mean_dist,
            torch, cv2, img, gt8, args.pad, "reflect", args.threshold, device)
        e_ref_chk, n_ref_chk, _ = infer_one(
            model, extract_keypoints_from_belief, hungarian_mean_dist,
            torch, cv2, img, gt8, args.pad, "reflect", args.threshold, device)
        # black
        e_blk, n_blk, p_blk, _, _ = canvas_and_orig_peaks(
            model, extract_keypoints_from_belief, hungarian_mean_dist,
            torch, cv2, img, gt8, args.pad, "black", args.threshold, device)

        # ---- panels ----
        A = img.copy(); draw_gt(A, gt8, cv2); draw_pred(A, p_none, cv2, KP_COLORS)
        label(A, f"none gt_err={e_none:.1f} n={n_none}", cv2)

        B = proc_ref.copy(); draw_pred(B, c_ref, cv2, KP_COLORS)
        label(B, "reflect CANVAS(model input)", cv2)

        C = img.copy(); draw_gt(C, gt8, cv2); draw_pred(C, p_ref, cv2, KP_COLORS)
        label(C, f"reflect gt_err={e_ref:.1f} n={n_ref}", cv2)

        D = img.copy(); draw_gt(D, gt8, cv2); draw_pred(D, p_blk, cv2, KP_COLORS)
        label(D, f"black gt_err={e_blk:.1f} n={n_blk}", cv2)

        combo = cv2.hconcat([A, B, C, D])
        out_path = os.path.join(args.output_dir, f"{fid}.jpg")
        cv2.imwrite(out_path, combo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(out_path)

        # ---- per-frame analysis ----
        h0, w0 = img.shape[:2]
        # fake-detection in panel B: canvas peak in border band [< pad_eff or
        # > size-pad_eff] of the canvas. pad in canvas units = pad*W/(W+2pad).
        pad_cx = args.pad * w0 / (w0 + 2 * args.pad)
        pad_cy = args.pad * h0 / (h0 + 2 * args.pad)
        border_hits = []
        for i in range(8):
            if np.isnan(c_ref[i, 0]):
                continue
            cx, cy = c_ref[i]
            in_border = (cx < pad_cx or cx > w0 - pad_cx or
                         cy < pad_cy or cy > h0 - pad_cy)
            if in_border:
                border_hits.append(i)
        # zoom-out signal: reflect orig-pred spread vs none orig-pred spread
        def spread(p):
            v = p[~np.isnan(p[:, 0])]
            return float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) >= 2 else np.nan
        sp_none, sp_ref = spread(p_none), spread(p_ref)

        print(f"\nframe {fid}  ({'EDGE' if is_edge else 'near'} "
              f"dist={dist:.0f}px sides={sides})")
        print(f"  self-check: none err {e_none:.4f}=={e_none_chk:.4f}? "
              f"{abs(e_none-e_none_chk)<1e-6}  n {n_none}=={n_none_chk}  | "
              f"reflect err {e_ref:.4f}=={e_ref_chk:.4f}? "
              f"{abs(e_ref-e_ref_chk)<1e-6}  n {n_ref}=={n_ref_chk}")
        print(f"  none    gt_err={e_none:6.1f}  n_det={n_none}")
        print(f"  reflect gt_err={e_ref:6.1f}  n_det={n_ref}")
        print(f"  black   gt_err={e_blk:6.1f}  n_det={n_blk}")
        print(f"  panelB border-band peaks (fake-reflect?): "
              f"{border_hits if border_hits else 'NONE'}  "
              f"(canvas band: x<{pad_cx:.0f} or >{w0-pad_cx:.0f}, "
              f"y<{pad_cy:.0f} or >{h0-pad_cy:.0f})")
        print(f"  bbox spread (diag px): none={sp_none:.1f}  reflect={sp_ref:.1f}"
              + (f"  -> reflect {100*(1-sp_ref/sp_none):.0f}% smaller (zoom-out?)"
                 if np.isfinite(sp_none) and np.isfinite(sp_ref) and sp_none > 0
                 else ""))

    print("\n" + "=" * 78)
    print(f"[save] {len(saved)} overlays -> {args.output_dir}")
    for s in saved:
        print(f"   {s}")


if __name__ == "__main__":
    main()
