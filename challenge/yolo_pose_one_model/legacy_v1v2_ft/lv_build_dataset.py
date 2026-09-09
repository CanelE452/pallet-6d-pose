"""LEGACY_V1V2_P0_10K -> YOLO-pose 데이터셋.

라벨 변환 계약은 새로 만들지 않는다. 기존 `scripts/prepare_yolo_pose.py` 의
`load_kps` / `to_line` / `PAD` 를 그대로 import 해서 쓴다 — G38 셋과 같은 계약이어야
FT 가 비교 가능하다 (reflect-pad 100, projected_cuboid[:8]+centroid, v=2 if in canvas).

split: sample_id 의 sha1 로 10% val. shard 로 자르면 생성 순서(=조명/배경 배치)와
       상관될 수 있어 쓰지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"
SRC = OUT_ROOT / "datasets/_raw_legacy_v1v2_p0_10k"
DST = OUT_ROOT / "datasets/legacy_v1v2_p0_10k"
VAL_FRAC = 0.10

_spec = importlib.util.spec_from_file_location(
    "prep", OUT_ROOT / "scripts/prepare_yolo_pose.py")
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep"] = prep
_spec.loader.exec_module(prep)
PAD = prep.PAD


def is_val(sample_id: str) -> bool:
    h = hashlib.sha1(sample_id.encode()).hexdigest()
    return (int(h[:8], 16) % 10000) < VAL_FRAC * 10000


def one(job):
    sid, img_src, ann_src, img_dst, lbl_dst = job
    kps = prep.load_kps(ann_src)
    if kps is None:
        return "no_annotation"
    img = cv2.imread(str(img_src))
    if img is None:
        return "unreadable_image"
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    line = prep.to_line(pw, ph, [(x + PAD, y + PAD) for x, y in kps])
    if line is None:
        return "all_kp_outside"
    cv2.imwrite(str(img_dst), padded)
    Path(lbl_dst).write_text(line + "\n", encoding="utf-8")
    return "ok"


def main():
    rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8")))
    assert len(rows) == 10000, f"manifest rows {len(rows)} != 10000"

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs, counts = [], {"train": 0, "val": 0}
    for r in rows:
        sid = r["sample_id"]
        split = "val" if is_val(sid) else "train"
        counts[split] += 1
        jobs.append((sid, SRC / r["rgb"], SRC / r["label"],
                     DST / "images" / split / f"{sid}.png",
                     DST / "labels" / split / f"{sid}.txt"))

    stats = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for k, res in enumerate(ex.map(one, jobs, chunksize=32), 1):
            stats[res] = stats.get(res, 0) + 1
            if k % 1000 == 0:
                print(f"  {k}/{len(jobs)}  {stats}", flush=True)

    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\nnc: 1\n"
        "kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        "names:\n  0: pallet\n", encoding="utf-8")

    n_img = {s: len(list((DST / "images" / s).glob("*.png"))) for s in ("train", "val")}
    n_lbl = {s: len(list((DST / "labels" / s).glob("*.txt"))) for s in ("train", "val")}
    report = {"planned": counts, "written_images": n_img, "written_labels": n_lbl,
              "convert_stats": stats, "pad": PAD, "val_frac": VAL_FRAC,
              "converter": "scripts/prepare_yolo_pose.py (imported)"}
    (Path(__file__).parent / "DATASET_BUILD.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = stats.get("ok", 0)
    if ok != 10000 or n_img != n_lbl:
        print("BUILD_INCOMPLETE", flush=True)
        sys.exit(1)
    print("BUILD_OK", flush=True)


if __name__ == "__main__":
    main()
