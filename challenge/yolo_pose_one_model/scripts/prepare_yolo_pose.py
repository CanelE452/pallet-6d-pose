"""Convert G/T frames to padded YOLO-pose images + labels.

Contract (identical to challenge/yolo_pose/scripts/convert_to_yolo_pose.py, which produced
the existing challenge YOLO sets - kept the same so the two are comparable):

  image   : cv2.copyMakeBorder(..., BORDER_REFLECT_101) with pad px on all four sides
  keypoint: x += pad, y += pad
  v       : 2 if the point is known AND the padded canvas contains it, else 0
            (and x=y=0 is written).  "known" is carried as a third tuple element,
            never encoded as a magic coordinate.
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
    """Return the 9 keypoints in contract order, or None if unusable.

    ``keypoint_annotations`` 가 있으면 그것을 쓴다.  같은 파일 안의
    ``projected_cuboid`` 와 인덱스 배정이 다를 수 있고, 규약을 지키는 쪽이
    ``keypoint_annotations`` 이기 때문이다.  2026-09-06 `live_capture_gt` 851장 전수:

        camera-facing 0123 규약 (0 왼쪽 / 1 오른쪽, 0·1 위 / 3·2 아래)
          keypoint_annotations   위반   0 / 851
          projected_cuboid       위반 198 / 851  (23.3%)

    위반 198장 중 196장이 좌우가 **완전히** 뒤집힌 것이고(뒤집힘 폭 중앙값 35.9 px,
    정상 프레임의 여유폭 중앙값 180.1 px), 경계 잡음으로 볼 수 있는 5 px 미만은 13장뿐이다.
    근거: `_docs/audits/accuracy_root_cause_v1/REAL_LABEL_AUDIT.md`.

    합성 GT 에는 ``keypoint_annotations`` 가 없으므로(표본 전수 0/40) 기존과 똑같이
    ``projected_cuboid`` 로 떨어진다 — 합성 데이터셋 재빌드 결과는 바뀌지 않는다.
    """
    try:
        obj = json.load(open(ann_path, encoding="utf-8"))["objects"][0]
    except Exception:
        return None

    ann = obj.get("keypoint_annotations")
    if isinstance(ann, list) and len(ann) >= 9:
        kps = []
        for entry in ann[:9]:
            e = entry if isinstance(entry, dict) else {}
            xy = e.get("xy")
            # 계약 정본: scripts/annotate/real_gt_v2_schema.py
            # ``keypoint_annotations_to_ultralytics`` — visibility 0 은 좌표가
            # 남아 있어도 provenance 를 모른다는 뜻이라 학습 타깃이 [0,0,0] 이어야 한다.
            known = xy is not None and int(e.get("visibility", 0)) != 0
            kps.append((float(xy[0]), float(xy[1]), True) if known
                       else (0.0, 0.0, False))
        return kps

    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    cen = obj.get("projected_cuboid_centroid")
    kps = [(float(p[0]), float(p[1]), True) for p in proj[:8]]
    kps.append((float(cen[0]), float(cen[1]), True) if cen else (0.0, 0.0, False))
    return kps


def to_line(w, h, kps):
    """kps 는 ``(x, y)`` 또는 ``(x, y, known)``.

    ``known=False`` 는 좌표를 모른다는 뜻이라 padding 뒤에 캔버스 안으로 들어오더라도
    감독하지 않는다 — 상태를 좌표 값 하나로 표현하면 sentinel 이 실제 점과 구별되지
    않는다(-0.5 + PAD = 99.5 는 캔버스 안이라 v=2 가 됐고 bbox 까지 늘렸다).
    """
    vis, out = [], []
    for p in kps:
        x, y = p[0], p[1]
        known = p[2] if len(p) > 2 else True
        v = 2 if (known and 0 <= x < w and 0 <= y < h) else 0
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
    for p, v in zip(kps, vis):
        x, y = p[0], p[1]
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
    line = to_line(pw, ph, [(x + PAD, y + PAD, k) for x, y, k in kps])
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
