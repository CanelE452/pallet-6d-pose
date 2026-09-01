"""arm 별 Ultralytics pose dataset 을 exposure lock 그대로 만든다.

핵심은 **한 epoch 의 물리적 크기를 모든 arm 에서 같게** 만드는 것이다.  accepted
unique PL 수가 arm 마다 다르므로 "같은 epoch" 은 공정 비교가 아니다.  그래서 epoch
당 슬롯 수를 고정하고, 작은 pool 은 with replacement 로 그 슬롯을 채운다.

    epoch = 1440 pseudo-real + 1440 synthetic replay = 2880 = 90 update(batch 32)
    R0-CONT 만 pseudo 슬롯을 synthetic 으로 대체한다 (pseudo exposure 0).

이미지는 복사하지 않고 symlink 한다.  중복 노출은 train 목록(.txt)에 같은 경로를
여러 번 적어서 만든다 — 디스크에 2,880 장을 복제할 이유가 없다.

pseudo label 로 저장하는 것은 2D box + 2D keypoint + visibility 뿐이다.
teacher 의 pose 는 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
CACHE = RESULTS / "teacher_cache/R0_TEACHER_CACHE.json"
LOCK = RESULTS / "SELFTRAIN_EXPOSURE_LOCK.json"
REPLAY_LIST = RESULTS / "SYNTHETIC_REPLAY_SUBSET.txt"
MANIFEST_DIR = RESULTS / "pseudo_manifests"
FILTER_LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
R0_DATASET = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
OUT_ROOT = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v1"

ARM_TO_FILTER = {
    "R0_CONT": None,
    "R1_NAIVE": "F0_NAIVE",
    "R2_CONF": "F1_CONF",
    "R3_CONF_REPROJ": "F2_CONF_REPROJ",
    "R4_CONF_REMOVE": "F3_CONF_REMOVE",
    "R5_PROPOSED": "F4_PROPOSED",
}
N_KEYPOINTS = 9
N_CORNERS = 8
VISIBLE = 2
NOT_LABELLED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pseudo_label_line(entry: dict, kp_conf_threshold: float) -> str | None:
    """teacher top1 을 Ultralytics pose 한 줄로 바꾼다.  실패하면 None."""

    top = entry["top1"]
    width = float(entry["image_width"])
    height = float(entry["image_height"])

    x1, y1, x2, y2 = (float(value) for value in top["box_xyxy"])
    x1, x2 = max(0.0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(height, max(y1, y2))
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 1.0 or box_h <= 1.0:
        return None
    tokens = [
        "0",
        f"{((x1 + x2) / 2) / width:.6f}", f"{((y1 + y2) / 2) / height:.6f}",
        f"{box_w / width:.6f}", f"{box_h / height:.6f}",
    ]

    keypoints = np.asarray(top["keypoints_xy"], dtype=float)
    confidences = np.nan_to_num(np.asarray(top["keypoints_conf"], dtype=float), nan=0.0)
    for index in range(N_KEYPOINTS):
        x, y = keypoints[index]
        inside = 0.0 <= x < width and 0.0 <= y < height
        # 신뢰도가 낮거나 화면 밖인 점을 억지로 visible GT 로 만들지 않는다.
        visible = VISIBLE if (confidences[index] >= kp_conf_threshold and inside) else NOT_LABELLED
        tokens += [
            f"{min(max(x / width, 0.0), 1.0):.6f}",
            f"{min(max(y / height, 0.0), 1.0):.6f}",
            str(visible),
        ]
    return " ".join(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=list(ARM_TO_FILTER))
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    filter_lock = json.loads(FILTER_LOCK.read_text())
    kp_conf = float(filter_lock["keypoint_validity"]["kp_conf_threshold"])
    pseudo_slots = int(lock["pseudo_exposures_per_epoch"])
    synthetic_slots = int(lock["synthetic_exposures_per_epoch"])
    seed = int(lock["augmentation"]["seed"])

    cache = json.loads(CACHE.read_text())
    by_sha = {entry["image_sha256"]: entry for entry in cache["entries"]}
    replay = [name for name in REPLAY_LIST.read_text().split("\n") if name]
    if len(replay) != synthetic_slots:
        raise SystemExit(f"REPLAY_SUBSET_SIZE_MISMATCH: {len(replay)} != {synthetic_slots}")

    source = json.loads((R0_DATASET / "data.yaml").read_text().replace(":", ": ", 0)) \
        if False else None  # data.yaml 은 아래에서 직접 쓴다

    report: dict[str, dict] = {}
    for arm in args.arms:
        filter_name = ARM_TO_FILTER[arm]
        dataset = OUT_ROOT / arm
        images = dataset / "images" / "train"
        labels = dataset / "labels" / "train"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)

        entries: list[str] = []
        unique_pseudo = 0

        # ── synthetic replay ────────────────────────────────────────────
        synthetic_count = synthetic_slots if filter_name else synthetic_slots + pseudo_slots
        for name in replay:
            link = images / f"replay__{name}"
            if not link.exists():
                link.symlink_to(R0_DATASET / "images" / "train" / name)
            label_source = (R0_DATASET / "labels" / "train" / name).with_suffix(".txt")
            target = (labels / f"replay__{name}").with_suffix(".txt")
            if not target.exists():
                target.write_text(label_source.read_text())
        # R0-CONT 는 replay 를 두 번 돌려 pseudo 슬롯을 메운다.
        repeats = synthetic_count // len(replay)
        entries += [f"replay__{name}" for name in replay] * repeats

        # ── pseudo-real ─────────────────────────────────────────────────
        if filter_name:
            rows = list(csv.DictReader((MANIFEST_DIR / f"{filter_name}.csv").open()))
            accepted: list[str] = []
            for row in rows:
                entry = by_sha[row["image_sha256"]]
                line = pseudo_label_line(entry, kp_conf)
                if line is None:
                    continue
                name = f"pl__{row['capture_session']}__{Path(row['image_path']).name}"
                link = images / name
                if not link.exists():
                    link.symlink_to(REPO_ROOT / row["image_path"])
                (labels / name).with_suffix(".txt").write_text(line + "\n")
                accepted.append(name)
            unique_pseudo = len(accepted)
            if not accepted:
                raise SystemExit(f"NO_PSEUDO_LABELS_FOR_ARM: {arm}")
            # 고정 슬롯을 결정적으로 채운다.  pool 이 작으면 with replacement.
            rng = random.Random(f"{seed}:{arm}")
            order = list(accepted)
            filled: list[str] = []
            while len(filled) < pseudo_slots:
                rng.shuffle(order)
                filled += order[: pseudo_slots - len(filled)]
            entries += filled

        rng = random.Random(f"{seed}:{arm}:order")
        rng.shuffle(entries)
        listing = dataset / "train.txt"
        listing.write_text(
            "\n".join(str(images / name) for name in entries) + "\n"
        )
        (dataset / "data.yaml").write_text(
            f"# generated by scripts/self_training_yolo/build_pseudo_datasets.py\n"
            f"path: {dataset}\n"
            f"train: train.txt\n"
            f"val: train.txt\n"
            f"nc: 1\n"
            f"kpt_shape: [9, 3]\n"
            f"flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
            f"names:\n  0: pallet\n"
        )

        report[arm] = {
            "filter": filter_name,
            "unique_pseudo_labels": unique_pseudo,
            "pseudo_exposures_per_epoch": pseudo_slots if filter_name else 0,
            "synthetic_exposures_per_epoch": synthetic_count,
            "pseudo_exposures_per_unique": (
                round(pseudo_slots / unique_pseudo, 2) if unique_pseudo else None
            ),
            "entries_per_epoch": len(entries),
            "dataset": str(dataset.relative_to(REPO_ROOT)),
            "train_list_sha256": sha256_file(listing),
        }
        print(f"{arm:16} filter={str(filter_name):16} unique_PL={unique_pseudo:<5} "
              f"epoch_entries={len(entries)}  "
              f"pseudo/unique={report[arm]['pseudo_exposures_per_unique']}")

    expected = pseudo_slots + synthetic_slots
    for arm, stats in report.items():
        if stats["entries_per_epoch"] != expected:
            raise SystemExit(
                f"EXPOSURE_MISMATCH: {arm} {stats['entries_per_epoch']} != {expected}"
            )
    (RESULTS / "PSEUDO_DATASET_REPORT.json").write_text(
        json.dumps({
            "epoch_entries": expected,
            "total_optimizer_updates": lock["total_optimizer_updates"],
            "arms": report,
        }, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nEXPOSURE OK — 모든 arm 이 epoch 당 {expected} 노출")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
