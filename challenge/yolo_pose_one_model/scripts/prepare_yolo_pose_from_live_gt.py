#!/usr/bin/env python3
"""수동 어노 GT(live_capture_gt) 를 YOLO-pose 데이터셋으로 바꾼다.

변환 로직은 새로 쓰지 않고 ``prepare_yolo_pose`` 의 것을 그대로 import 한다.
PAD=100 / BORDER_REFLECT_101 / visibility 판정 / bbox 계산이 synthetic 학습셋과
한 글자도 달라지면 안 되기 때문이다.  이 파일이 하는 일은 "어떤 GT 가 어떤 이미지에
대응하는가" 를 풀고 split 을 나누는 것뿐이다.

split 은 목적에 따라 고른다 (``--split-mode``).  과제 트랙은 "이 현장·이 팔레트에
맞추는" 것이 목적이라 기본이 ``interleave`` — 모든 세션에서 고르게 val 을 뺀다.
촬영 단위(``group``)로 가르면 FT 효과가 아니라 촬영 간 도메인 갭을 재게 된다.

사용 예::

    python challenge/yolo_pose_one_model/scripts/prepare_yolo_pose_from_live_gt.py \\
        --out datasets/live_gt_v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from prepare_yolo_pose import PAD, one  # noqa: E402  변환 규약을 공유한다

GT_ROOT = REPO / "challenge/data/01_real/live_capture_gt"
CAPTURE_ROOT = REPO / "challenge/data/01_real/_live_captures"

# GT 폴더 이름 → (이미지 rgb 폴더, 촬영 그룹).  촬영 그룹이 split 단위다.
SESSION_MAP = {
    "capture_20260902_manual_gt": (
        "handheld_20260902/sessions/capture_20260902", "handheld_20260902"),
    "capture_20260902_kimjihoon_manual_gt": (
        "handheld_20260902/sessions/capture_20260902_kimjihoon", "handheld_20260902"),
    "forklift_v4_173507_manual_gt": (
        "forklift_v4_20260901/sessions/forklift_v4_173507", "forklift_v4_20260901"),
    "forklift_v4_174126_manual_gt": (
        "forklift_v4_20260901/sessions/forklift_v4_174126", "forklift_v4_20260901"),
    "forklift_v4_174342_manual_gt": (
        "forklift_v4_20260901/sessions/forklift_v4_174342", "forklift_v4_20260901"),
    "forklift_v4_174925_manual_gt": (
        "forklift_v4_20260901/sessions/forklift_v4_174925", "forklift_v4_20260901"),
}


def collect_jobs(out_root: Path, val_groups: set[str], *,
                 mode: str = "group", every: int = 6):
    """(split, job) 목록과 그룹별 개수를 만든다.

    ``mode`` 가 split 의 성격을 정한다. 목적이 다르면 지표도 달라야 한다.

    * ``group``     — 촬영 단위로 가른다. 처음 본 촬영에 일반화되는지를 잰다.
    * ``interleave`` — 모든 세션에서 ``every`` 장마다 하나를 val 로 뺀다.  train 과
      val 이 같은 분포가 된다.  **과제 트랙처럼 "이 현장·이 팔레트에 맞추는" 것이
      목적일 때 이쪽을 쓴다.**  group split 으로 재면 도메인 갭이 섞여 들어와
      FT 효과가 아니라 촬영 차이를 재게 된다.
    """
    jobs = {"train": [], "val": []}
    counts: dict[str, int] = {}
    missing_image = 0
    for gt_name, (rgb_rel, group) in SESSION_MAP.items():
        gt_dir = GT_ROOT / gt_name
        if not gt_dir.is_dir():
            continue
        group_split = "val" if group in val_groups else "train"
        rgb_dir = CAPTURE_ROOT / rgb_rel / "rgb"
        for order, ann in enumerate(sorted(gt_dir.glob("*.json"))):
            split = (("val" if order % every == 0 else "train")
                     if mode == "interleave" else group_split)
            image = rgb_dir / f"{ann.stem}.png"
            if not image.is_file():
                missing_image += 1
                continue
            stem = f"{gt_name}__{ann.stem}"
            jobs[split].append((
                stem,
                os.path.relpath(image, REPO),
                os.path.relpath(ann, REPO),
                str(out_root / "images" / split / f"{stem}.png"),
                str(out_root / "labels" / split / f"{stem}.txt"),
            ))
            counts[group] = counts.get(group, 0) + 1
    return jobs, counts, missing_image


def add_crop_jobs(out_root: Path, crop_dir: Path, val_stems: set[str]):
    """truncation crop 을 train 에만 더한다.

    **val 프레임에서 파생된 crop 은 반드시 뺀다.**  crop 은 원본을 잘라 만든 것이라,
    val 원본의 crop 이 train 에 들어가면 val 이 train 을 외운 값을 재게 된다.
    crop 파일명은 ``<세션>_<프레임>_t<k>`` 이므로 ``_t<k>`` 를 떼면 원본을 찾는다.
    """
    jobs, dropped = [], 0
    for ann in sorted(crop_dir.glob("*.json")):
        image = ann.with_suffix(".png")
        if not image.is_file():
            continue
        origin = ann.stem.rsplit("_t", 1)[0]
        if origin in val_stems:
            dropped += 1
            continue
        stem = f"crop__{ann.stem}"
        jobs.append((
            stem,
            os.path.relpath(image, REPO),
            os.path.relpath(ann, REPO),
            str(out_root / "images" / "train" / f"{stem}.png"),
            str(out_root / "labels" / "train" / f"{stem}.txt"),
        ))
    return jobs, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="datasets/live_gt_v1",
                    help="yolo_pose_one_model 아래 데이터셋 루트")
    ap.add_argument("--val-group", default="forklift_v4_20260901",
                    help="group 모드에서 val 로 뺄 촬영 그룹")
    ap.add_argument("--split-mode", choices=["group", "interleave"], default="interleave",
                    help="interleave = 모든 세션에서 고르게 val 을 뺀다(과제 트랙 기본). "
                         "group = 촬영 단위로 갈라 일반화를 잰다")
    ap.add_argument("--val-every", type=int, default=6,
                    help="interleave 모드에서 몇 장마다 하나를 val 로 뺄지")
    ap.add_argument("--crop-dir", default=None,
                    help="truncation crop 폴더. train 에만 더하고 val 파생분은 뺀다")
    args = ap.parse_args(argv)

    out_root = REPO / "challenge/yolo_pose_one_model" / args.out
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs, counts, missing = collect_jobs(
        out_root, {args.val_group}, mode=args.split_mode, every=args.val_every)
    print(f"split 모드: {args.split_mode}"
          + (f" (매 {args.val_every}장마다 val)" if args.split_mode == "interleave"
             else f" (val group={args.val_group})"))
    print(f"촬영 그룹별 GT 수: {counts}")
    if missing:
        print(f"  ⚠️ 대응 이미지 없음 {missing}개 (건너뜀)")

    crop_dropped = 0
    if args.crop_dir:
        # val 원본의 stem 집합.  job[0] 은 "<gt_name>__<frame>" 이고 crop 은
        # "<gt_name 에서 _manual_gt 뺀 것>_<frame>" 이라 형태를 맞춰 준다.
        val_stems = set()
        for job in jobs["val"]:
            gt_name, frame = job[0].split("__", 1)
            val_stems.add(f"{gt_name.replace('_manual_gt', '')}_{frame}")
        crop_jobs, crop_dropped = add_crop_jobs(
            out_root, Path(args.crop_dir), val_stems)
        jobs["train"].extend(crop_jobs)
        print(f"crop 추가: {len(crop_jobs)}개 (val 파생 {crop_dropped}개 제외)")

    results = {}
    for split in ("train", "val"):
        tally: dict[str, int] = {}
        for job in jobs[split]:
            outcome = one(job)
            tally[outcome] = tally.get(outcome, 0) + 1
        results[split] = tally
        print(f"  {split:5} {len(jobs[split]):4d} frames → {tally}")

    if not results["train"].get("ok") or not results["val"].get("ok"):
        print("  ⚠️ train 또는 val 이 비었다 — split 을 다시 보라")
        return 1

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [9, 3]\n"
        "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        "names:\n"
        "  0: pallet\n", encoding="utf-8")

    (out_root / "_prepare_live_gt.json").write_text(json.dumps({
        "pad": PAD, "border": "BORDER_REFLECT_101",
        "split_mode": args.split_mode, "val_every": args.val_every,
        "val_group": args.val_group, "group_counts": counts,
        "results": results, "missing_image": missing,
        "crop_dir": args.crop_dir, "crop_dropped_from_val": crop_dropped,
        "source": "challenge/data/01_real/live_capture_gt (manual, 4-fold normalised)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  data.yaml: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
