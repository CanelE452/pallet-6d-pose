"""stage0/extract_pl_v1.py — paper_base PL extraction over the LOCKED pl_pool.

Track-1 (paper) Stage-0 entry. Runs paper_base DOPE inference on EXACTLY the
frames in the split-lock pl_pool (data/.../split_lock/pl_pool_frames.txt), dumps
one NDDS-format PL json per frame, and records diag/flip as SCORES (not gates) so
the τ / weight decision is made LATER in tau_calibrate.py.

★ GPU script — DO NOT RUN here; the user runs inference. CPU-safe parts (argparse,
   frame→path resolution, IO) are smoke-checkable with --dry_run.

Design contract (judged by the stage-0 spec):
  (1) reuses existing geometry — load_model / extract_keypoints_from_belief /
      filt_diag / canonical_kp3d from filter_pr_camfacing, and infer_flip_kp /
      flip_consistency_score from filter_flip_consistency. NO re-implemented geom.
  (2) input = split-lock pl_pool_frames.txt (pool sessions only).
  (3) every PL json carries  source_model='base_v2_sigma2_ep66'  to distinguish
      from the 4-arm base-v2 PL (extract_pl_v5.py).
  (4) diag pass + flip score stored per PL as scores, gate decided later.
  final-test sessions never appear (they are not in pl_pool_frames.txt).

Preprocessing (2026-06-19):
  inference uses ASPECT (ratio-preserving short-side->400, non-square input),
  NOT squash 400x400.  Predicted belief-map keypoints are remapped EXACTLY to
  ORIGINAL-image px:  orig = (b * _nw/bw) / _sc , (b * _nh/bh) / _sc.  This is
  the precise aspect inverse (avoids the ~1% &~7 systematic error in
  evaluate_on_val._belief_to_orig which reuses the squash formula).  The flip
  pass (infer_flip_kp) already returns ORIGINAL-px kp, so the flip score is on
  the same (original-px) scale.

Usage (user, GPU):
  conda run -n pallet-pose python scripts/stage0/filter_pl/extract_pl_v1.py \
      --weights weights/paper_base_v2_s2/net_epoch_0066.pth \
      --output_dir data/pallet/pl/stage0_paper_base

Smoke (CPU, no model):
  python scripts/stage0/filter_pl/extract_pl_v1.py --dry_run
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
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

SOURCE_MODEL = "base_v2_sigma2_ep66"
RAW = os.path.join(ROOT, "data", "pallet", "raw_data")
POOL_SESSION_DIRS = [
    os.path.join(RAW, "outside", "capturepallet01", "rgb"),
    os.path.join(RAW, "outside", "capturepallet10", "rgb"),
    os.path.join(RAW, "outside", "capturepallet11", "rgb"),
    os.path.join(RAW, "night", "capturenight01", "rgb"),
    os.path.join(RAW, "night", "capturenight02", "rgb"),
    os.path.join(RAW, "night", "capturenight03", "rgb"),
    os.path.join(RAW, "night", "capturenight04", "rgb"),
    os.path.join(RAW, "night", "capturenight10", "rgb"),
]
DEFAULT_POOL = os.path.join(ROOT, "data", "pallet", "eval_results",
                            "split_lock", "pl_pool_frames.txt")
# default RealSense intrinsics (same constants used by extract_pl_v5)
FX, FY, CX, CY = 605.906494140625, 605.9697875976562, 317.59619140625, 256.29229736328125


def domain_of(img_path):
    return "night" if (os.sep + "night" + os.sep) in img_path else "outside"


def resolve_pool_frames(pool_txt):
    """Map each frame id in pool_txt to its image path among POOL_SESSION_DIRS.

    Frame ids are unique filenames (verified: 2227 outside + 5804 night = 8031).
    Returns (list[(frame_id, img_path)], list[missing_ids]).
    """
    index = {}
    for d in POOL_SESSION_DIRS:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in glob.glob(os.path.join(d, ext)):
                index[os.path.splitext(os.path.basename(p))[0]] = p
    pairs, missing = [], []
    for ln in open(pool_txt):
        fid = ln.strip()
        if not fid:
            continue
        if fid in index:
            pairs.append((fid, index[fid]))
        else:
            missing.append(fid)
    return pairs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "paper_base_v2_s2", "net_epoch_0066.pth"))
    ap.add_argument("--pool_txt", default=DEFAULT_POOL)
    ap.add_argument("--image_dir", default=None,
                    help="if set, glob this dir for images directly (bypass pool_txt)")
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "pl", "stage0_paper_base"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--source_model", default=SOURCE_MODEL)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="limit (quick test)")
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU-safe: resolve frames + write a manifest, NO model")
    args = ap.parse_args()

    if args.image_dir:
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            imgs += glob.glob(os.path.join(args.image_dir, ext))
        pairs = [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(imgs)]
        missing = []
        print(f"[image_dir] {args.image_dir}: {len(pairs)} images")
    else:
        pairs, missing = resolve_pool_frames(args.pool_txt)
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    n_out = sum(1 for _, p in pairs if domain_of(p) == "outside")
    n_night = len(pairs) - n_out
    print(f"[pool] resolved {len(pairs)} frames "
          f"(outside={n_out} night={n_night}); missing={len(missing)}")
    if missing:
        print(f"  WARNING: {len(missing)} pool ids had no image file "
              f"(first: {missing[:3]})")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dry_run:
        manifest = {
            "source_model": args.source_model, "weights": args.weights,
            "pool_txt": args.pool_txt, "n_resolved": len(pairs),
            "n_outside": n_out, "n_night": n_night, "n_missing": len(missing),
            "dry_run": True,
        }
        json.dump(manifest, open(os.path.join(
            args.output_dir, "_manifest.json"), "w"), indent=2)
        print(f"[dry_run] no inference. manifest -> "
              f"{os.path.join(args.output_dir, '_manifest.json')}")
        return

    # ── GPU section (reused geometry only) ────────────────────────────────
    import cv2
    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, filt_diag, canonical_kp3d,
    )
    from filter_flip_consistency import infer_flip_kp, flip_consistency_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)
    dist = np.zeros((5, 1))
    # 3D model points only needed if a PnP pose is desired; PL filter is 2D so we
    # store keypoints + scores. canonical_kp3d kept for downstream PnP consumers.
    _ = canonical_kp3d(1.1, 1.3, 0.15)  # noqa: F841 (documents convention reuse)

    records = []
    t0 = time.time()
    for i, (fid, ip) in enumerate(pairs):
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ── ASPECT preprocess (ratio-preserving short-side->400, non-square) ──
        # 학습 입력은 RandomCrop400 (정사각 crop, 비율유지)이므로 short-side->400.
        # squash(400x400)는 종횡비 왜곡 → 좌표 왜곡 + N 과소집계 → 사용 금지.
        _sc = 400.0 / min(h0, w0)
        _nw = max(8, int(round(w0 * _sc)) & ~7)   # 8의 배수 (FCN stride 8)
        _nh = max(8, int(round(h0 * _sc)) & ~7)
        img_resized = cv2.resize(rgb, (_nw, _nh))
        t = ((img_resized.astype(np.float32) / 255.0 - mean) / std)
        tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, args.threshold)
        bh, bw = belief.shape[1], belief.shape[2]
        # belief(bh×bw) → input(_nh×_nw) [FCN upsample _nw/bw≈8] → orig(/ _sc).
        # 정확한 aspect 역매핑 (evaluate_on_val 의 squash재사용 ~1% 오차 회피).
        ux, uy = _nw / bw, _nh / bh

        kp = []  # 9 entries, original-image px or None
        confs = []
        for k in kps_bel:
            if k[0] < 0:
                kp.append(None); confs.append(0.0)
            else:
                kp.append(((k[0] * ux) / _sc, (k[1] * uy) / _sc)); confs.append(float(k[2]))
        n_det = sum(1 for x in kp[:8] if x is not None)

        # scores (NOT gates) — reused geometry
        diag_pass, diag_score = filt_diag(kp)
        kpB = infer_flip_kp(model, img, device, mean, std, args.threshold,
                            preprocess="aspect")
        flip_score, flip_ncmp = flip_consistency_score(kp, kpB)

        # NDDS dump — projected_cuboid 9 (None -> [-100,-100] so the loader
        # treats them as out-of-frame and skips their belief, same as v5).
        proj = [[float(p[0]), float(p[1])] if p is not None else [-100.0, -100.0]
                for p in kp]
        ndds = {
            "camera_data": {"width": w0, "height": h0,
                            "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY}},
            "source_model": args.source_model,
            "filter_scores": {
                "diag_pass": bool(diag_pass),
                "diag_score": (None if not np.isfinite(diag_score) else round(float(diag_score), 4)),
                "flip_score": (None if flip_score is None else round(float(flip_score), 3)),
                "flip_ncmp": int(flip_ncmp),
                "n_detected": int(n_det),
                "min_conf": round(float(min([c for x, c in zip(kp[:8], confs) if x is not None] or [0.0])), 4),
            },
            "objects": [{
                "class": "pallet", "name": "real_pallet", "visibility": 1,
                "source_model": args.source_model,
                "projected_cuboid": proj,
                "projected_cuboid_centroid": proj[8],
            }],
        }
        out_json = os.path.join(args.output_dir, fid + ".json")
        out_png = os.path.join(args.output_dir, fid + ".png")
        cv2.imwrite(out_png, img)
        json.dump(ndds, open(out_json, "w"), indent=2)

        records.append({
            "frame": fid, "domain": domain_of(ip), "src": ip,
            "n_detected": n_det, "diag_pass": bool(diag_pass),
            "diag_score": (None if not np.isfinite(diag_score) else round(float(diag_score), 4)),
            "flip_score": (None if flip_score is None else round(float(flip_score), 3)),
        })
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(pairs)}] dumped={len(records)} "
                  f"elapsed={time.time()-t0:.0f}s")

    summary = {
        "source_model": args.source_model, "weights": args.weights,
        "pool_txt": args.pool_txt, "n_input": len(pairs),
        "n_dumped": len(records), "n_missing": len(missing),
        "threshold": args.threshold, "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(args.output_dir, "_manifest.json"), "w"), indent=2)
    json.dump(records, open(os.path.join(args.output_dir, "_records.json"), "w"), indent=2)
    print(f"\n[done] {len(records)} PL -> {args.output_dir}")
    print(f"  manifest: {os.path.join(args.output_dir, '_manifest.json')}")
    print(f"  records : {os.path.join(args.output_dir, '_records.json')}  "
          f"(feeds pl_pass_count.py)")


if __name__ == "__main__":
    main()
