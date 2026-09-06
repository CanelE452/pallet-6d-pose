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

def discover_sessions() -> dict[str, tuple[str, str]]:
    """GT 폴더 이름에서 (이미지 rgb 폴더, 촬영 그룹) 을 찾는다.

    예전에는 여기에 6 개를 손으로 적어 뒀는데, 세션이 28 개로 늘어난 뒤에도 그대로라
    새 촬영분이 **조용히 학습에서 빠졌다**.  그래서 이름 규약으로 찾는다 —
    ``<세션>_manual_gt`` 는 ``_live_captures/<그룹>/sessions/<세션>/rgb`` 에 대응한다.
    """
    found = {}
    for gt_dir in sorted(GT_ROOT.glob("*_manual_gt")):
        name = gt_dir.name[: -len("_manual_gt")]
        hits = list(CAPTURE_ROOT.glob(f"*/sessions/{name}/rgb"))
        if len(hits) != 1:
            print(f"  ⚠️ {name}: rgb 후보 {len(hits)}개 — 건너뜀")
            continue
        rgb = hits[0]
        found[gt_dir.name] = (
            str(rgb.parent.relative_to(CAPTURE_ROOT)),   # <그룹>/sessions/<세션>
            rgb.parents[2].name)                          # <그룹>
    return found


SESSION_MAP = None   # main() 에서 discover_sessions() 로 채운다


def collect_jobs(out_root: Path, val_groups: set[str], session_map: dict, *,
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
    for gt_name, (rgb_rel, group) in session_map.items():
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


def assert_derived_is_current(src_dir: Path, *, sample: int = 30) -> None:
    """파생 폴더가 정본 keypoint 필드를 들고 있고, 고친 생성기에서 나왔는지 본다.

    두 가지 실패를 막는다.  둘 다 **조용히** 틀린 라벨로 학습하게 만든다.

    1. `keypoint_annotations` 가 없다 -> `load_kps` 가 `projected_cuboid` fallback
       으로 내려간다.  그 필드는 `truncation_crops_livegt` 에서 318/1,203 이
       camera-facing 0123 규약을 어긴다.
    2. 필드는 있는데 provenance(`keypoint_source`)가 없다 -> 생성기를 고치기 **전에**
       만들어진 산출물이다.  낡은 `flip_noise_aug_livegt` 가 그 상태인데,
       851/851 이 이미지만 뒤집히고 라벨은 안 뒤집혔다.

    근거: `_docs/audits/next_accuracy_v2/DERIVED_DATA_AUDIT.md`
    """
    files = sorted(src_dir.glob("*.json"))[:sample]
    if not files:
        raise SystemExit(f"파생 폴더에 JSON 이 없다: {src_dir}")
    no_field, no_prov = [], []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))["objects"][0]
        except Exception:
            no_field.append(f.name)
            continue
        ann = obj.get("keypoint_annotations")
        if not (isinstance(ann, list) and len(ann) >= 9):
            no_field.append(f.name)
        elif not obj.get("keypoint_source"):
            no_prov.append(f.name)
    if no_field:
        raise SystemExit(
            f"거부: {src_dir} 의 {len(no_field)}/{len(files)} 이 keypoint_annotations 를\n"
            f"  갖고 있지 않다 -> projected_cuboid fallback 으로 내려간다.\n"
            f"  예: {no_field[:3]}\n"
            f"  생성기를 고쳤으니 이 폴더를 **다시 만들어라**.\n"
            f"  근거 _docs/audits/next_accuracy_v2/DERIVED_DATA_AUDIT.md")
    if no_prov:
        raise SystemExit(
            f"거부: {src_dir} 의 {len(no_prov)}/{len(files)} 에 keypoint_source 가 없다\n"
            f"  -> 생성기를 고치기 전에 만들어진 낡은 산출물이다.\n"
            f"  예: {no_prov[:3]}\n"
            f"  낡은 flip 산출물은 851/851 이 이미지만 뒤집히고 라벨은 안 뒤집혔다.\n"
            f"  이 폴더를 **다시 만들어라**.")


def add_derived_jobs(out_root: Path, src_dir: Path, val_stems: set[str],
                     *, sep: str, prefix: str):
    """원본에서 파생된 증강본을 train 에만 더한다.

    **val 프레임에서 나온 것은 반드시 뺀다.**  파생본은 원본을 변형한 것이라,
    val 원본의 파생본이 train 에 들어가면 val 이 train 을 외운 값을 재게 된다.
    파일명 끝의 ``sep`` 뒤를 떼면 원본 stem 이 된다 (crop ``_t<k>`` / flip ``_f``
    / noise ``_n``).
    """
    jobs, dropped = [], 0
    for ann in sorted(src_dir.glob("*.json")):
        image = ann.with_suffix(".png")
        if not image.is_file():
            continue
        if ann.stem.rsplit(sep, 1)[0] in val_stems:
            dropped += 1
            continue
        stem = f"{prefix}{ann.stem}"
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
    ap.add_argument("--aug-dir", default=None,
                    help="flip/noise 증강 폴더(_f/_n). train 에만 더한다")
    ap.add_argument("--pad", type=int, default=None,
                    help="reflect padding 픽셀. 기본은 prepare_yolo_pose.PAD(=100). "
                         "0 이면 무패딩 — Jetson 처럼 추론 예산이 빠듯할 때 쓴다. "
                         "★학습과 추론의 padding 은 반드시 같아야 한다(train/infer parity)")
    args = ap.parse_args(argv)

    if args.pad is not None:
        # ``one()`` 은 모듈 전역 PAD 를 조회하므로 여기서 갈아끼우면 반영된다.
        import prepare_yolo_pose as _pyp
        _pyp.PAD = args.pad
        print(f"padding = {args.pad} px" + (" (무패딩)" if args.pad == 0 else ""))

    out_root = REPO / "challenge/yolo_pose_one_model" / args.out
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    session_map = discover_sessions()
    print(f"세션 자동 탐색: {len(session_map)}개")
    jobs, counts, missing = collect_jobs(
        out_root, {args.val_group}, session_map,
        mode=args.split_mode, every=args.val_every)
    print(f"split 모드: {args.split_mode}"
          + (f" (매 {args.val_every}장마다 val)" if args.split_mode == "interleave"
             else f" (val group={args.val_group})"))
    print(f"촬영 그룹별 GT 수: {counts}")
    if missing:
        print(f"  ⚠️ 대응 이미지 없음 {missing}개 (건너뜀)")

    # val 원본의 stem 집합.  job[0] 은 "<gt_name>__<frame>" 이고 파생본은
    # "<gt_name 에서 _manual_gt 뺀 것>_<frame>" 이라 형태를 맞춰 준다.
    val_stems = set()
    for job in jobs["val"]:
        gt_name, frame = job[0].split("__", 1)
        val_stems.add(f"{gt_name.replace('_manual_gt', '')}_{frame}")

    derived = {}
    for label, path, sep, prefix in (
            ("crop", args.crop_dir, "_t", "crop__"),
            ("aug", args.aug_dir, "_", "aug__")):
        if not path:
            continue
        assert_derived_is_current(Path(path))
        add, dropped = add_derived_jobs(
            out_root, Path(path), val_stems, sep=sep, prefix=prefix)
        jobs["train"].extend(add)
        derived[label] = {"added": len(add), "dropped_from_val": dropped}
        print(f"{label} 추가: {len(add)}개 (val 파생 {dropped}개 제외)")

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
        # ★ import 된 PAD 는 --pad 로 갈아끼워도 안 바뀐다 — 모듈에서 실제 값을 읽는다.
        "pad": __import__("prepare_yolo_pose").PAD,
        "border": "BORDER_REFLECT_101",
        "split_mode": args.split_mode, "val_every": args.val_every,
        "val_group": args.val_group, "group_counts": counts,
        "results": results, "missing_image": missing,
        "crop_dir": args.crop_dir, "aug_dir": args.aug_dir, "derived": derived,
        "source": "challenge/data/01_real/live_capture_gt (manual, 4-fold normalised)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  data.yaml: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
