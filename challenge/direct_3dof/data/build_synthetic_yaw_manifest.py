#!/usr/bin/env python3
"""synthetic 라벨에서 direct-yaw 학습용 manifest 를 만든다.

``pose_transform`` 에 카메라 좌표계 6DoF 가 그대로 있으므로 PnP 를 거치지 않는다.
yaw 는 배포 코드와 같은 식으로만 뽑는다::

    yaw = atan2(R[0,2], R[2,2])          # geometry.py:_angles_from_R 과 동일

**형상비 필터가 이 스크립트의 핵심이다.**  현장 팔레트는 4방향 진입이라 90° 회전이
등가고, 그래서 타깃을 ``(sin 4ψ, cos 4ψ)`` 로 접는다.  그런데 synthetic 팔레트는
치수가 프레임마다 랜덤이라 수평 형상비 중앙값이 1.24 인 **직사각**이다.  직사각을
4-fold 로 접으면 실제로 다른 두 포즈에 같은 타깃을 주게 되어 라벨이 모순된다.
정사각에 가까운 프레임만 남기는 이유가 이것이다 (기본 1.10).

x, z 는 만들지 않는다.  그 값은 기존 keypoint → PnP 경로가 낸다.  synthetic 치수가
랜덤이라 ``실제크기 / z`` 만 관측되어 z 를 direct 로 배우는 것 자체가 불가능하다
(``3DOF_CONTRACT.md`` §7.2).

사용 예::

    python challenge/direct_3dof/data/build_synthetic_yaw_manifest.py \\
        --dataset challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k \\
        --split train --max-aspect-ratio 1.10 \\
        --out challenge/direct_3dof/data/manifests/synthetic_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))
from pose3dof import encode_yaw, wrap_yaw  # noqa: E402

REPO = TRACK.parents[1]

# dataset 파일명 접두어 → 원본 라벨 트리.  build_dataset.py 가 하드링크하며 붙인 이름이라
# 이 표가 곧 역매핑이다.  이름을 지어내지 않고 실제 접두어를 그대로 쓴다.
SOURCE_TREES = {
    "G38": REPO / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels",
    "P0": REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k",
    "TEX": REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_tex10k",
}


def resolve_label_path(stem: str) -> Path | None:
    """``G38__G__f2503`` / ``P0__shard_03_f0021`` → 원본 ``*_label.json``."""
    parts = stem.split("__")
    if not parts:
        return None
    prefix = parts[0]
    root = SOURCE_TREES.get(prefix)
    if root is None:
        return None
    if prefix == "G38":
        return root / f"{parts[-1]}_label.json"
    # P0 / TEX 는 shard 폴더가 한 단계 더 있다: shard_03_f0021 → shard_03/labels/f0021
    tail = parts[-1]
    marker = tail.rfind("_f")
    if marker < 0:
        return None
    return root / tail[:marker] / "labels" / f"{tail[marker + 1:]}_label.json"


def read_pose(label_path: Path) -> dict | None:
    """라벨 하나에서 yaw 와 형상비를 뽑는다.  필요한 키가 없으면 None."""
    try:
        with open(label_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        obj = payload["objects"][0]
        transform = np.asarray(obj["pose_transform"], dtype=np.float64)
        dims = obj["dimensions_m"]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    if transform.shape != (4, 4):
        return None
    rotation = transform[:3, :3]
    horizontal = (float(dims["width"]), float(dims["depth"]))
    if min(horizontal) <= 0.0:
        return None
    return {
        "yaw_rad": float(math.atan2(rotation[0, 2], rotation[2, 2])),
        "aspect_ratio": max(horizontal) / min(horizontal),
        "width_m": horizontal[0],
        "depth_m": horizontal[1],
        "height_m": float(dims["height"]),
        "z_m": float(transform[2, 3]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True,
                    help="YOLO 데이터셋 루트 (images/<split> 를 가진 폴더)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-aspect-ratio", type=float, default=1.10,
                    help="수평 형상비 상한. 4-fold 등가 가정이 성립할 만큼 정사각인 것만 남긴다")
    ap.add_argument("--out", required=True, help="출력 JSONL")
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    args = ap.parse_args(argv)

    images = sorted((Path(args.dataset) / "images" / args.split).glob("*.png"))
    if not images:
        ap.error(f"이미지가 없다: {Path(args.dataset) / 'images' / args.split}")
    if args.limit:
        images = images[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = {"label_missing": 0, "unreadable": 0, "aspect": 0}
    yaws: list[float] = []
    with open(out_path, "w", encoding="utf-8") as sink:
        for image in images:
            label = resolve_label_path(image.stem)
            if label is None or not label.is_file():
                dropped["label_missing"] += 1
                continue
            pose = read_pose(label)
            if pose is None:
                dropped["unreadable"] += 1
                continue
            if pose["aspect_ratio"] > args.max_aspect_ratio:
                dropped["aspect"] += 1
                continue
            folded = float(wrap_yaw(pose["yaw_rad"]))
            sin4, cos4 = encode_yaw(pose["yaw_rad"])
            sink.write(json.dumps({
                "image_path": os.path.relpath(image, REPO),
                "label_path": os.path.relpath(label, REPO),
                "yaw_rad": pose["yaw_rad"],
                "yaw_folded_rad": folded,
                "sin4yaw": float(sin4),
                "cos4yaw": float(cos4),
                "aspect_ratio": pose["aspect_ratio"],
                "width_m": pose["width_m"],
                "depth_m": pose["depth_m"],
                "height_m": pose["height_m"],
                "z_m": pose["z_m"],
                "source": image.stem.split("__")[0],
                "gt_source": "synthetic_exact_pose_transform",
                "symmetry_fold": 4,
            }, ensure_ascii=False) + "\n")
            yaws.append(folded)
            kept += 1

    print(f"입력 {len(images)}장 → 채택 {kept}장")
    print(f"  제외: 라벨없음 {dropped['label_missing']}  읽기실패 {dropped['unreadable']}  "
          f"형상비>{args.max_aspect_ratio} {dropped['aspect']}")
    if yaws:
        a = np.degrees(np.asarray(yaws))
        hist, _ = np.histogram(a, bins=9, range=(0, 90))
        print(f"  접은 yaw [0,90) 분포: {list(hist)}  (균일할수록 좋다)")
    print(f"  출력: {out_path}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
