"""Audit generated YOLO-pose datasets. Training must not start on a failing dataset.

Checks (per the task spec):
  pair exists                    every image has a label and vice versa
  token count                    5 + 9*3 = 32 per line
  class id                       always 0
  keypoint count                 exactly 9
  bbox in [0,1]                  cx, cy, w, h
  bbox positive                  w > 0 and h > 0
  v=2 coords in [0,1]
  no NaN / Inf
  bbox contains visible kps      every v=2 point lies inside the bbox (+1px tolerance)
  centroid sanity                kp8 must sit near the mean of the visible corners
  no split duplication           an image sha256 may not appear in two splits
  no double padding              padded size must equal a known source size + 2*100

Usage:
  python .../audit_yolo_pose_labels.py --dataset datasets/smoke
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "challenge/yolo_pose_one_model"
PAD = 100
KNOWN_SRC = {(640, 480), (960, 540), (720, 480), (560, 560), (1280, 720)}
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]


def png_size(p):
    with open(p, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audit_split(root: Path, split: str, hash_all: bool):
    imgs = sorted((root / "images" / split).glob("*.png"))
    lbls = sorted((root / "labels" / split).glob("*.txt"))
    err = collections.Counter()
    samples = collections.Counter()
    hashes = {}
    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}
    err["image_without_label"] = len(img_stems - lbl_stems)
    err["label_without_image"] = len(lbl_stems - img_stems)

    for ip in imgs:
        lp = root / "labels" / split / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        size = png_size(ip)
        if size is None:
            err["unreadable_png"] += 1
            continue
        w, h = size
        if ((w - 2 * PAD), (h - 2 * PAD)) not in KNOWN_SRC:
            err["double_padding_or_unknown_size"] += 1
            samples["bad_size"] = f"{ip.name} {w}x{h}"
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        if len(lines) != 1:
            err["not_exactly_one_object"] += 1
        for line in lines:
            t = line.split()
            if len(t) != 32:
                err["token_count"] += 1
                continue
            try:
                vals = [float(x) for x in t]
            except ValueError:
                err["non_numeric"] += 1
                continue
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                err["nan_inf"] += 1
                continue
            if int(vals[0]) != 0:
                err["class_id"] += 1
            cx, cy, bw, bh = vals[1:5]
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                err["bbox_range"] += 1
            if bw <= 0 or bh <= 0:
                err["bbox_nonpositive"] += 1
            kps = [(vals[5 + 3 * i], vals[6 + 3 * i], int(vals[7 + 3 * i])) for i in range(9)]
            if len(kps) != 9:
                err["kp_count"] += 1
            x0, x1 = cx - bw / 2, cx + bw / 2
            y0, y1 = cy - bh / 2, cy + bh / 2
            tolx, toly = 1.0 / w, 1.0 / h
            vis = [(x, y) for x, y, v in kps if v == 2]
            for x, y, v in kps:
                if v not in (0, 2):
                    err["visibility_not_0_or_2"] += 1
                if v == 2 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    err["visible_kp_out_of_range"] += 1
                if v == 0 and (x != 0.0 or y != 0.0):
                    err["invisible_kp_not_zeroed"] += 1
                if v == 2 and not (x0 - tolx <= x <= x1 + tolx and y0 - toly <= y <= y1 + toly):
                    err["visible_kp_outside_bbox"] += 1
            if not vis:
                err["no_visible_kp"] += 1
            # Centroid sanity: only meaningful when ALL 8 corners are visible. With
            # partial visibility (truncation) the mean of the visible corners is
            # legitimately far from the centroid - e.g. seeing only the near face puts
            # the mean on that face while the centroid sits half a depth behind it.
            corners = [(x, y) for (x, y, v) in kps[:8] if v == 2]
            if kps[8][2] == 2 and len(corners) == 8:
                mx = sum(p[0] for p in corners) / len(corners)
                my = sum(p[1] for p in corners) / len(corners)
                d = math.hypot((kps[8][0] - mx) * w, (kps[8][1] - my) * h)
                diag = math.hypot(bw * w, bh * h)
                if diag > 0 and d > 0.35 * diag:
                    err["centroid_far_from_corners"] += 1
                    samples.setdefault("centroid", f"{ip.name} d={d:.1f}px diag={diag:.1f}px")
        if hash_all:
            hashes[sha(ip)] = ip.name
    return {"n_images": len(imgs), "n_labels": len(lbls), "errors": err,
            "samples": dict(samples), "hashes": hashes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hash", action="store_true", default=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    root = OUT / args.dataset

    print(f"auditing {args.dataset}")
    res = {}
    for split in ("train", "val"):
        if not (root / "images" / split).exists():
            continue
        res[split] = audit_split(root, split, args.hash)
        e = res[split]["errors"]
        total = sum(e.values())
        print(f"  {split}: images={res[split]['n_images']} labels={res[split]['n_labels']} "
              f"errors={total}")
        for k, v in e.most_common():
            if v:
                print(f"      {k}: {v}")
        for k, v in res[split]["samples"].items():
            print(f"      e.g. {k}: {v}")

    dup = 0
    if "train" in res and "val" in res and args.hash:
        both = set(res["train"]["hashes"]) & set(res["val"]["hashes"])
        dup = len(both)
        print(f"  train/val 동일 이미지 sha256 중복: {dup}")

    inv = all(FLIP_IDX[FLIP_IDX[i]] == i for i in range(9))
    print(f"  flip_idx involution: {'OK' if inv else 'BROKEN'}")

    fail = any(sum(res[s]["errors"].values()) for s in res) or dup or not inv
    if args.report:
        rp = OUT / args.report
        lines = ["# 04 — Label audit\n",
                 f"대상: `{args.dataset}`  ",
                 "생성: `python challenge/yolo_pose_one_model/scripts/audit_yolo_pose_labels.py "
                 f"--dataset {args.dataset}`\n", "```"]
        for s in res:
            lines.append(f"{s}: images={res[s]['n_images']} labels={res[s]['n_labels']} "
                         f"errors={sum(res[s]['errors'].values())}")
            for k, v in res[s]["errors"].most_common():
                if v:
                    lines.append(f"    {k}: {v}")
        lines += [f"train/val sha256 중복: {dup}",
                  f"flip_idx involution: {'OK' if inv else 'BROKEN'}", "```\n",
                  f"판정: **{'FAIL' if fail else 'PASS'}**"]
        rp.write_text("\n".join(lines), encoding="utf-8")
        print("wrote", rp.relative_to(REPO))
    print("VERDICT:", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
