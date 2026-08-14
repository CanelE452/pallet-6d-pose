"""DOPE NDDS JSON → YOLOv8-pose 포맷 변환.

입력 (둘 다 지원):
  1. Synthetic: challenge/data/02_synthetic/training/v1, v2 (재귀, part_NNN/train_palletobj_v*)
  2. Manual GT: challenge/data/01_real/manual_gt/capture*_manual_gt/
  3. mixed_v8: data/pallet/training_data/mixed_v8_train/

출력:
  challenge/data/03_derived/yolo_pose/
    images/{train,val}/{stem}.png   (symlink 또는 copy)
    labels/{train,val}/{stem}.txt   (YOLO pose 포맷)

YOLO pose 한 줄 포맷 (정규화 0~1):
  class cx cy w h kp1_x kp1_y kp1_v  kp2_x kp2_y kp2_v  ...  kp9_x kp9_y kp9_v

  class : 0 (pallet)
  cx,cy,w,h : 정규화된 bbox (image 안 keypoint 들의 axis-aligned bbox)
  kpN_x,y : 정규화된 keypoint 좌표 (0~1)
  kpN_v   : visibility (0=outside-image, 2=visible)
            YOLO 는 0/1/2 인데 1(가려짐) 정보 없으므로 0/2 만 사용

Keypoint 순서 (v4 convention):
  0: front-top-LEFT     1: front-top-RIGHT
  2: front-bot-RIGHT    3: front-bot-LEFT
  4: rear-top-LEFT      5: rear-top-RIGHT
  6: rear-bot-RIGHT     7: rear-bot-LEFT
  8: centroid

Usage:
  python convert_to_yolo_pose.py \
      --src challenge/data/02_synthetic/training/v1 challenge/data/02_synthetic/training/v2 \
      --src-glob "challenge/data/01_real/manual_gt/capture*_manual_gt" \
      --src data/pallet/training_data/mixed_v8_train \
      --out challenge/data/03_derived/yolo_pose \
      --val-ratio 0.05 \
      --link            # symlink (default: copy 안 함, png 는 image 폴더 안에 symlink)
"""

import argparse
import json
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def find_pairs(root: str):
    """root 아래에서 {stem}.json + {stem}.png 페어를 재귀 탐색."""
    pairs = []
    for json_path in glob(os.path.join(root, "**", "*.json"), recursive=True):
        if json_path.endswith(".json.orig"):
            continue
        png_path = json_path[:-5] + ".png"
        if os.path.exists(png_path):
            pairs.append((png_path, json_path))
    return pairs


def parse_json(json_path: str):
    """NDDS JSON → (img_w, img_h, list of objects with kps[(x,y), ...] x 9).

    합성/manual 모두 projected_cuboid (8) + projected_cuboid_centroid (1) 가 표준.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    cam = data.get("camera_data", {})
    intr = cam.get("intrinsics", {})
    img_w = cam.get("width") or intr.get("resolution", [640, 480])[0]
    img_h = cam.get("height") or intr.get("resolution", [640, 480])[1]

    objs = []
    for obj in data.get("objects", []):
        if obj.get("class", "").lower() != "pallet":
            continue
        cuboid = obj.get("projected_cuboid")
        centroid = obj.get("projected_cuboid_centroid")
        if cuboid is None or len(cuboid) != 8 or centroid is None:
            continue
        kps = [(float(p[0]), float(p[1])) for p in cuboid]
        kps.append((float(centroid[0]), float(centroid[1])))
        objs.append({"kps": kps})

    return img_w, img_h, objs


def to_yolo_line(img_w: int, img_h: int, kps: list) -> str | None:
    """9-keypoint → YOLO pose 한 줄. image 안 keypoint 가 0 이면 None 반환 (skip)."""
    visibility = []
    for x, y in kps:
        if 0 <= x < img_w and 0 <= y < img_h:
            visibility.append(2)
        else:
            visibility.append(0)
    if sum(v == 2 for v in visibility) == 0:
        return None

    # bbox = image 안 keypoint 들의 axis-aligned bbox
    in_kps = [(x, y) for (x, y), v in zip(kps, visibility) if v == 2]
    xs = [p[0] for p in in_kps]
    ys = [p[1] for p in in_kps]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = max(1.0, x1 - x0) / img_w
    h = max(1.0, y1 - y0) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for (x, y), v in zip(kps, visibility):
        if v == 2:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
        else:
            nx, ny = 0.0, 0.0
        parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(v)])
    return " ".join(parts)


def link_or_copy(src: str, dst: str, link: bool):
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        # Windows 에서 symlink 권한 없을 수 있어서 fallback: hard-copy
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copyfile(src, dst)
    else:
        shutil.copyfile(src, dst)


def pad_image_and_save(src_png: str, dst_png: str, pad: int, mode: str = "reflect"):
    """src image 에 pad px 사방 padding 추가 후 dst 에 저장. (new_w, new_h) 반환."""
    img = cv2.imread(src_png)
    if img is None:
        raise RuntimeError(f"failed to read {src_png}")
    h, w = img.shape[:2]
    border = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "black": cv2.BORDER_CONSTANT,
    }.get(mode, cv2.BORDER_REFLECT_101)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=(0, 0, 0))
    cv2.imwrite(dst_png, padded)
    return w + 2 * pad, h + 2 * pad


def shift_keypoints(kps: list, dx: int, dy: int) -> list:
    """모든 keypoint 좌표에 (dx, dy) 더해서 padded coord 로 변환."""
    return [(x + dx, y + dy) for (x, y) in kps]


def unique_stem(png_path: str, src_root: str) -> str:
    """source root 안에서의 상대 경로를 _ 로 join 해 unique stem 생성.
    예: part_003/train_palletobj_v1/000123.png → part_003_train_palletobj_v1_000123
    """
    rel = os.path.relpath(png_path, src_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return rel_no_ext.replace(os.sep, "_").replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[], help="입력 루트 (반복 가능)")
    ap.add_argument("--src-glob", action="append", default=[],
                    help="입력 루트 glob (예: 'challenge/data/01_real/manual_gt/capture*_manual_gt')")
    ap.add_argument("--out", required=True, help="출력 루트 (challenge/data/03_derived/yolo_pose)")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="png 를 symlink (default: copy)")
    ap.add_argument("--prefix", default="",
                    help="stem 앞에 붙일 prefix (multi-src 충돌 방지용, src 별로 따로 호출 권장)")
    ap.add_argument("--pad", type=int, default=0,
                    help="이미지 사방 padding (px). >0 이면 image 를 새로 저장하고 keypoint 좌표도 shift. "
                         "truncation case 의 image 밖 keypoint 를 padded 안에 들어오게 함. (default: 0 = padding 없음)")
    ap.add_argument("--pad-mode", default="reflect", choices=["reflect", "replicate", "black"],
                    help="padding 방식 (default: reflect)")
    args = ap.parse_args()

    src_roots = list(args.src)
    for g in args.src_glob:
        src_roots.extend(sorted(glob(g)))
    if not src_roots:
        print("ERROR: --src or --src-glob 필요", file=sys.stderr)
        sys.exit(1)

    out_root = args.out
    img_train = os.path.join(out_root, "images", "train")
    img_val = os.path.join(out_root, "images", "val")
    lbl_train = os.path.join(out_root, "labels", "train")
    lbl_val = os.path.join(out_root, "labels", "val")
    for d in (img_train, img_val, lbl_train, lbl_val):
        os.makedirs(d, exist_ok=True)

    rng = random.Random(args.seed)
    total = 0
    skipped_no_kp = 0
    skipped_no_pair = 0
    src_stats = {}

    for src in src_roots:
        if not os.path.isdir(src):
            print(f"  skip (not a dir): {src}")
            continue
        src_name = os.path.basename(os.path.normpath(src))
        prefix = args.prefix or src_name
        pairs = find_pairs(src)
        n_used = 0
        for png, jsn in pairs:
            try:
                img_w, img_h, objs = parse_json(jsn)
            except Exception as e:
                skipped_no_pair += 1
                continue
            if not objs:
                skipped_no_pair += 1
                continue

            # padding 적용 시 좌표/크기 변환
            if args.pad > 0:
                eff_w, eff_h = img_w + 2 * args.pad, img_h + 2 * args.pad
                shifted_objs = [{"kps": shift_keypoints(o["kps"], args.pad, args.pad)} for o in objs]
            else:
                eff_w, eff_h = img_w, img_h
                shifted_objs = objs

            lines = []
            for obj in shifted_objs:
                line = to_yolo_line(eff_w, eff_h, obj["kps"])
                if line:
                    lines.append(line)
            if not lines:
                skipped_no_kp += 1
                continue

            stem_unique = f"{prefix}__{unique_stem(png, src)}"
            split = "val" if rng.random() < args.val_ratio else "train"
            img_dst = os.path.join(out_root, "images", split, stem_unique + ".png")
            lbl_dst = os.path.join(out_root, "labels", split, stem_unique + ".txt")
            if args.pad > 0:
                # padding: 새 image 생성 (link 모드는 무시)
                pad_image_and_save(png, img_dst, args.pad, args.pad_mode)
            else:
                link_or_copy(png, img_dst, args.link)
            with open(lbl_dst, "w") as f:
                f.write("\n".join(lines) + "\n")
            n_used += 1
            total += 1

        src_stats[src] = n_used
        print(f"  {src_name:50s}  {n_used:6d} pairs")

    print("\n=== 변환 완료 ===")
    print(f"  총 변환 frame      : {total}")
    print(f"  skip (no in-img kp): {skipped_no_kp}")
    print(f"  skip (parse fail)  : {skipped_no_pair}")
    print(f"  출력 루트          : {out_root}")
    print(f"  train/val 비율     : {1-args.val_ratio:.2f} / {args.val_ratio:.2f}")


if __name__ == "__main__":
    main()
