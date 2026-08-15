"""extract_pl_camfacing.py — self-training PL extractor (2D-geometric filter, NO PnP).

paper_base (camera-facing 0123) inference on unlabeled real pool -> 9 keypoints
(belief peaks) -> 2D-geometric domain filter -> NDDS json (CleanVisiiDopeLoader).

Filter selection (filter_combo_9kp, 2026-06-04):
    outside = diag
    night   = diag AND ratio

Reuses the AUDITED functions from filter_pr_camfacing.py
(extract_keypoints_from_belief, filt_diag, filt_ratio) so this matches exactly
the filter that the combo_9kp study was built on. No PnP (paper track =
generalize to unseen pallets, dims unknown).

Excludes eval-set frame IDs and _exclude.txt (so PL never contaminates eval).

Usage:
    python scripts/data_prep/inference/extract_pl_camfacing.py \
        --weights weights/paper_base/paper_base/final_net_epoch_0060.pth \
        --domain outside
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import (  # noqa: E402  (audited functions)
    extract_keypoints_from_belief, filt_diag, filt_ratio,
)

# domain -> (pool capture dirs glob, eval-set dir, intrinsics dict, filter list)
DOMAINS = {
    "outside": {
        "pool_glob": os.path.join(ROOT, "data", "outside", "capturepallet*"),
        "eval_dir": os.path.join(ROOT, "data", "_eval_sets", "outside_combined"),
        "intr": {"fx": 614.1838989257812, "fy": 614.3132934570312,
                 "cx": 329.2803955078125, "cy": 234.52880859375},
        "filters": ["diag"],
    },
    "night": {
        "pool_glob": os.path.join(ROOT, "data", "night", "capturenight*"),
        "eval_dir": os.path.join(ROOT, "data", "_eval_sets", "night_combined"),
        "intr": {"fx": 605.906494140625, "fy": 605.9697875976562,
                 "cx": 317.59619140625, "cy": 256.29229736328125},
        "filters": ["diag", "ratio"],
    },
    "indoor": {
        # indoor pool = capture02 + capture03 (pure unlabeled rgb, ~2023 frames).
        # capture0403noapril (188) = held-out eval/calib set, NOT a PL pool.
        # capture0403middle = AprilTag-labeled set, kept for eval. Both excluded.
        # noapril_eval + noapril_calib IDs excluded via _filelist_exclude as safety.
        "pool_dirs": [
            os.path.join(ROOT, "data", "pallet", "raw_data", "capture02"),
            os.path.join(ROOT, "data", "pallet", "raw_data", "capture03"),
        ],
        "eval_dir": None,  # uses _filelist_exclude instead of a json eval dir
        "_filelist_exclude": [
            os.path.join(ROOT, "data", "pallet", "raw_data",
                         "capture0403noapril", "noapril_eval", "filelist.txt"),
            os.path.join(ROOT, "data", "pallet", "raw_data",
                         "capture0403noapril", "noapril_calib", "filelist.txt"),
        ],
        "intr": {"fx": 614.1838989257812, "fy": 614.3132934570312,
                 "cx": 329.2803955078125, "cy": 234.52880859375},
        "filters": ["diag"],
    },
}

FILTER_FN = {"diag": filt_diag, "ratio": filt_ratio}


def load_exclude():
    fp = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
    ex = set()
    if os.path.exists(fp):
        for ln in open(fp):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def collect_pool(domain, exclude_ids):
    """All pool png paths minus eval-set frame IDs minus _exclude.txt."""
    d = DOMAINS[domain]
    if d.get("eval_dir"):
        eval_ids = {os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(d["eval_dir"], "*.json"))}
    else:
        eval_ids = set()
    for flp in d.get("_filelist_exclude", []):
        if os.path.exists(flp):
            for ln in open(flp):
                ln = ln.strip()
                if ln:
                    eval_ids.add(os.path.splitext(os.path.basename(ln))[0])
    drop = eval_ids | exclude_ids
    paths = []
    cap_dirs = d["pool_dirs"] if d.get("pool_dirs") else sorted(glob.glob(d["pool_glob"]))
    for cap in cap_dirs:
        for p in sorted(glob.glob(os.path.join(cap, "rgb", "*.png"))):
            fid = os.path.splitext(os.path.basename(p))[0]
            if fid in drop:
                continue
            paths.append(p)
    return paths, len(eval_ids)


def infer_kp(model, img, device, mean, std, threshold):
    """Return 9 kp as list of (u,v) in original px or None, + 9 confidences."""
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((cv2.resize(rgb, (400, 400)).astype(np.float32) / 255.0 - mean) / std)  # 학습 입력(400×400)과 일치
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w0, bh / h0
    kp, confs = [], []
    for k in kps_bel:
        if k[0] < 0:
            kp.append(None); confs.append(0.0)
        else:
            kp.append((k[0] / sx, k[1] / sy)); confs.append(float(k[2]))
    return kp, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "paper_base", "final_net_epoch_0060.pth"))
    ap.add_argument("--domain", required=True, choices=list(DOMAINS.keys()))
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth pool frame (temporal de-dup)")
    ap.add_argument("--n_overlay", type=int, default=12)
    args = ap.parse_args()

    dom = DOMAINS[args.domain]
    out_dir = args.output_dir or os.path.join(
        ROOT, "output", f"pl_paper_r1_{args.domain}")
    os.makedirs(out_dir, exist_ok=True)
    ov_dir = os.path.join(out_dir, "_overlays")
    os.makedirs(ov_dir, exist_ok=True)

    exclude = load_exclude()
    pool, n_eval = collect_pool(args.domain, exclude)
    pool = pool[::args.stride]
    if args.max_frames:
        pool = pool[:args.max_frames]
    intr = dom["intr"]
    flt = dom["filters"]
    print(f"[{args.domain}] pool={len(pool)} (eval excluded={n_eval}, "
          f"stride={args.stride})  filters={flt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])

    n_infer = n_det = n_pass = accepted = 0
    overlay_saved = 0
    t0 = time.time()
    for i, ip in enumerate(pool):
        img = cv2.imread(ip)
        if img is None:
            continue
        n_infer += 1
        kp, confs = infer_kp(model, img, device, mean, std, args.threshold)
        n_kp = sum(1 for x in kp if x is not None)
        if n_kp < 6:           # detection = >=6 of 9
            continue
        n_det += 1

        ok = True
        scores = {}
        for f in flt:
            passed, sc = FILTER_FN[f](kp)
            scores[f] = round(float(sc), 4) if np.isfinite(sc) else None
            ok = ok and bool(passed)
        if not ok:
            continue
        n_pass += 1

        # NDDS json: full-res px, None -> [-100,-100] (loader treats off-image)
        projected = [[float(p[0]), float(p[1])] if p is not None else [-100.0, -100.0]
                     for p in kp]
        base = f"{accepted:06d}"
        out_png = os.path.join(out_dir, base + ".png")
        out_json = os.path.join(out_dir, base + ".json")
        cv2.imwrite(out_png, img)
        ndds = {
            "camera_data": {"width": img.shape[1], "height": img.shape[0],
                            "intrinsics": dict(intr)},
            "objects": [{
                "class": "pallet", "name": "real_pallet", "visibility": 1,
                "projected_cuboid": projected[:8],
                "projected_cuboid_centroid": projected[8],
                "pl_meta": {"src_image": os.path.relpath(ip, ROOT),
                            "min_conf": round(min(c for x, c in zip(kp, confs)
                                                  if x is not None), 4),
                            "n_kp": n_kp, "filter_scores": scores},
            }],
        }
        json.dump(ndds, open(out_json, "w"), indent=2)

        # overlay (first n_overlay passed PL): 9kp + spatial diagonals
        if overlay_saved < args.n_overlay:
            ov = img.copy()
            SD = [(0, 6), (1, 7), (2, 4), (3, 5)]
            for a, b in SD:
                if kp[a] is not None and kp[b] is not None:
                    cv2.line(ov, tuple(map(int, kp[a])), tuple(map(int, kp[b])),
                             (0, 180, 255), 1)
            for idx in range(9):
                if kp[idx] is None:
                    continue
                c = (0, 0, 255) if idx < 4 else ((255, 100, 0) if idx < 8 else (0, 255, 0))
                p = tuple(map(int, kp[idx]))
                cv2.circle(ov, p, 4, c, -1)
                cv2.putText(ov, str(idx), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            cv2.imwrite(os.path.join(ov_dir, f"{base}.jpg"), ov)
            overlay_saved += 1

        accepted += 1
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(pool)}] det={n_det} pass={n_pass} "
                  f"acc={accepted}  {el:.0f}s  ({(i+1)/max(el,1):.1f}fps)")

    el = time.time() - t0
    summary = {
        "domain": args.domain, "weights": args.weights, "filters": flt,
        "pool_inferred": n_infer, "detected_ge6kp": n_det,
        "passed": n_pass, "accepted": accepted,
        "det_rate": round(n_det / max(n_infer, 1), 4),
        "pass_rate_of_det": round(n_pass / max(n_det, 1), 4),
        "pass_rate_of_pool": round(n_pass / max(n_infer, 1), 4),
        "elapsed_sec": round(el, 1), "stride": args.stride,
        "threshold": args.threshold,
    }
    json.dump(summary, open(os.path.join(out_dir, "_summary.json"), "w"), indent=2)
    print("\n" + "=" * 64)
    print(f"[{args.domain}] DONE -> {out_dir}")
    print(f"  inferred={n_infer} det(>=6kp)={n_det} "
          f"pass({'+'.join(flt)})={n_pass}  ({el:.0f}s)")
    print(f"  det_rate={summary['det_rate']:.1%}  "
          f"pass/det={summary['pass_rate_of_det']:.1%}  "
          f"pass/pool={summary['pass_rate_of_pool']:.1%}")
    print(f"  overlays: {ov_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
