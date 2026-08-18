"""Convert G/T frames to padded YOLO-pose images + labels.

Contract (identical to challenge/yolo_pose/scripts/convert_to_yolo_pose.py, which produced
the existing challenge YOLO sets - kept the same so the two are comparable):

  image   : cv2.copyMakeBorder(..., BORDER_REFLECT_101) with pad px on all four sides
  keypoint: x += pad, y += pad
  v       : 2 if the padded canvas contains the point, else 0 (and x=y=0 is written)
  bbox    : axis-aligned bbox of the v==2 keypoints, normalised by the PADDED size
  line    : "0 cx cy w h  x0 y0 v0 ... x8 y8 v8"

Synthetic GT is the renderer projection, so any point inside the padded canvas is v=2
(exact_synthetic). Frames whose 9 keypoints all fall outside the canvas are dropped and
counted.

Usage:
  python .../prepare_yolo_pose.py --manifest manifests/generic_train.txt \
      --out datasets/stage_a --split train --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"
PAD = 100
SENTINEL = -0.5          # renderer writes -1,-1 for "not projected"


def load_kps(ann_path):
    """Return the 9 keypoints in contract order, or None if unusable."""
    try:
        obj = json.load(open(ann_path, encoding="utf-8"))["objects"][0]
    except Exception:
        return None
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    cen = obj.get("projected_cuboid_centroid")
    kps = [tuple(map(float, p)) for p in proj[:8]]
    kps.append(tuple(map(float, cen)) if cen else (SENTINEL, SENTINEL))
    return kps


def to_line(w, h, kps):
    vis, out = [], []
    for x, y in kps:
        v = 2 if (0 <= x < w and 0 <= y < h) else 0
        vis.append(v)
    if sum(v == 2 for v in vis) == 0:
        return None
    inx = [k for k, v in zip(kps, vis) if v == 2]
    xs = [p[0] for p in inx]
    ys = [p[1] for p in inx]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx = min(1.0, max(0.0, (x0 + x1) / 2 / w))
    cy = min(1.0, max(0.0, (y0 + y1) / 2 / h))
    bw = min(1.0, max(0.0, max(1.0, x1 - x0) / w))
    bh = min(1.0, max(0.0, max(1.0, y1 - y0) / h))
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for (x, y), v in zip(kps, vis):
        if v == 2:
            parts += [f"{min(1.0, max(0.0, x / w)):.6f}", f"{min(1.0, max(0.0, y / h)):.6f}", "2"]
        else:
            parts += ["0.000000", "0.000000", "0"]
    out.append(" ".join(parts))
    return "\n".join(out)


def one(job):
    stem, img_rel, ann_rel, img_dst, lbl_dst = job
    kps = load_kps(REPO / ann_rel)
    if kps is None:
        return "no_annotation"
    img = cv2.imread(str(REPO / img_rel))
    if img is None:
        return "unreadable_image"
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    line = to_line(pw, ph, [(x + PAD, y + PAD) for x, y in kps])
    if line is None:
        return "all_kp_outside"
    cv2.imwrite(str(img_dst), padded)
    with open(lbl_dst, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="repo-relative txt of image paths")
    ap.add_argument("--out", required=True, help="dataset root, e.g. datasets/stage_a")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    reg = {r["image_path"]: r for r in
           csv.DictReader(open(OUT_ROOT / "manifests/all_samples.csv", encoding="utf-8"))}
    man = (OUT_ROOT / args.manifest) if not os.path.isabs(args.manifest) else Path(args.manifest)
    if not man.exists():
        man = REPO / args.manifest
    images = [l.strip() for l in open(man, encoding="utf-8") if l.strip()]
    if args.limit:
        images = images[:args.limit]

    img_dir = OUT_ROOT / args.out / "images" / args.split
    lbl_dir = OUT_ROOT / args.out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rel in images:
        r = reg.get(rel)
        if r is None:
            continue
        stem = r["sample_id"].replace("/", "__")
        jobs.append((stem, rel, r["annotation_path"],
                     str(img_dir / f"{stem}.png"), str(lbl_dir / f"{stem}.txt")))

    print(f"{args.manifest} -> {args.out}/{args.split}   {len(jobs)} frames, {args.workers} workers")
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(one, jobs, chunksize=64)):
            counts[res] = counts.get(res, 0) + 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)}  {counts}")
    print("  done:", counts)

    stat = OUT_ROOT / args.out / f"_prepare_{args.split}.json"
    json.dump({"manifest": args.manifest, "split": args.split, "pad": PAD,
               "border": "BORDER_REFLECT_101", "counts": counts, "n_input": len(jobs)},
              open(stat, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
