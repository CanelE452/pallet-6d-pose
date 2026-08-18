"""Draw label overlays so the keypoint order can be checked by eye.

Reads the generated YOLO labels (not the source JSON) so what you see is exactly what
the model will be trained on.

Colour code
  near face 0-3   green,  drawn as a closed quad
  far  face 4-7   blue,   drawn as a closed quad
  centroid 8      magenta
  bbox            yellow
  v=0 keypoints   not drawn; listed in the header instead

Usage:
  python .../render_overlays.py --dataset datasets/smoke --split train --n 100
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "challenge/yolo_pose_one_model"
NEAR = (0, 220, 0)
FAR = (255, 120, 0)
CEN = (255, 0, 255)
BOX = (0, 230, 230)
NAMES = ["nTL", "nTR", "nBR", "nBL", "fTL", "fTR", "fBR", "fBL", "C"]


def draw(img_path: Path, lbl_path: Path, dst: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    vals = [float(x) for x in lbl_path.read_text().split()]
    cx, cy, bw, bh = vals[1:5]
    x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    cv2.rectangle(img, (x0, y0), (x1, y1), BOX, 2)

    kps = [(vals[5 + 3 * i] * w, vals[6 + 3 * i] * h, int(vals[7 + 3 * i])) for i in range(9)]
    for face, col in ((range(0, 4), NEAR), (range(4, 8), FAR)):
        pts = [kps[i] for i in face]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            if a[2] == 2 and b[2] == 2:
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)
    for i, (x, y, v) in enumerate(kps):
        if v != 2:
            continue
        col = CEN if i == 8 else (NEAR if i < 4 else FAR)
        cv2.circle(img, (int(x), int(y)), 5, col, -1)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), 1)
        cv2.putText(img, f"{i}:{NAMES[i]}", (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, f"{i}:{NAMES[i]}", (int(x) + 7, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # padding boundary: the original image edge sits 100 px in from each side
    cv2.rectangle(img, (100, 100), (w - 100, h - 100), (160, 160, 160), 1)

    invis = [NAMES[i] for i, k in enumerate(kps) if k[2] != 2]
    hdr = [f"{img_path.stem}",
           f"padded {w}x{h}  (src {w-200}x{h-200})  v=2:{9-len(invis)}/9",
           f"v=0: {','.join(invis) if invis else 'none'}"]
    for j, t in enumerate(hdr):
        cv2.putText(img, t, (8, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, t, (8, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix", default="", help="only stems starting with this (G__ / T__)")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = OUT / args.dataset
    imgs = sorted((root / "images" / args.split).glob(f"{args.prefix}*.png"))
    picked = random.Random(args.seed).sample(imgs, min(args.n, len(imgs)))
    tag = args.outdir or f"{Path(args.dataset).name}_{args.split}{'_' + args.prefix.strip('_') if args.prefix else ''}"
    dstdir = OUT / "reports/overlays" / tag
    n = 0
    for p in picked:
        if draw(p, root / "labels" / args.split / f"{p.stem}.txt", dstdir / p.name):
            n += 1
    print(f"{n} overlays -> {dstdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
