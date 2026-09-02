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
    # flip 단독.  R4(removal 단독)와 짝을 이뤄 두 제안 필터의 2x2 ablation 을 완성한다.
    "R6_CONF_FLIP": "F5_CONF_FLIP",
    # A8 cross-domain transfer.  Proposed 필터를 한 도메인 pool 에만 적용한다.
    # 노출 슬롯(1440)은 그대로라 optimizer update 는 다른 arm 과 같다.
    # §18-B geometry incremental control.  A2 와 다른 실험이다 — A2 는 Naive pool 에서
    # 뽑고, 이건 confidence 를 **이미 통과한** pool 에서 뽑아 geometry 가 추가로
    # 걷어낸 13 장의 고유 기여만 분리한다.
    "B_CONF_RANDOM_S1": "B_CONF_RANDOM_S1",
    "B_CONF_RANDOM_S2": "B_CONF_RANDOM_S2",
    "B_CONF_RANDOM_S3": "B_CONF_RANDOM_S3",
    "B_CONF_TOPN": "B_CONF_TOPN",
    "B_CONF_DECILE": "B_CONF_DECILE",
    "A8_DAY_ONLY": "F4_DAY_ONLY",
    "A8_NIGHT_ONLY": "F4_NIGHT_ONLY",
    # A2 — UNIQUE-QUANTITY-MATCHED control.  Proposed 와 **unique PL 개수**를 맞춘
    # Naive 무작위 표본이다.  MAIN 의 EXPOSURE-MATCHED 와 다른 실험이므로 섞지 않는다.
    "A2_NAIVE_MATCHED_S1": "A2_NAIVE_MATCHED_S1",
    "A2_NAIVE_MATCHED_S2": "A2_NAIVE_MATCHED_S2",
    "A2_NAIVE_MATCHED_S3": "A2_NAIVE_MATCHED_S3",
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
    # Ultralytics 의 `seed` 는 dataloader 에 도달하지 않아 학습이 비트 동일해진다
    # (seed 42/43/44 가중치 max|Δw| = 0 으로 확인).  진짜 replicate 를 만들려면
    # 우리가 통제하는 pseudo 샘플링을 바꿔야 한다 — 어떤 pseudo-label 이 몇 번
    # 노출되는지가 달라지고, 노출 **총량**은 그대로다.
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--suffix", default="")
    # A12 strength sensitivity.  총 슬롯(=optimizer update)은 그대로 두고
    # pseudo:synthetic 비율만 바꾼다.
    parser.add_argument("--pseudo-fraction", type=float, default=None)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    filter_lock = json.loads(FILTER_LOCK.read_text())
    kp_conf = float(filter_lock["keypoint_validity"]["kp_conf_threshold"])
    pseudo_slots = int(lock["pseudo_exposures_per_epoch"])
    synthetic_slots = int(lock["synthetic_exposures_per_epoch"])
    total_slots = pseudo_slots + synthetic_slots
    if args.pseudo_fraction is not None:
        pseudo_slots = int(round(total_slots * args.pseudo_fraction))
        synthetic_slots = total_slots - pseudo_slots
    seed = args.sampling_seed if args.sampling_seed is not None \
        else int(lock["augmentation"]["seed"])

    cache = json.loads(CACHE.read_text())
    by_sha = {entry["image_sha256"]: entry for entry in cache["entries"]}
    replay = [name for name in REPLAY_LIST.read_text().split("\n") if name]
    if not replay:
        raise SystemExit("REPLAY_SUBSET_EMPTY")

    source = json.loads((R0_DATASET / "data.yaml").read_text().replace(":", ": ", 0)) \
        if False else None  # data.yaml 은 아래에서 직접 쓴다

    report: dict[str, dict] = {}
    for arm in args.arms:
        filter_name = ARM_TO_FILTER[arm]
        dataset = OUT_ROOT / f"{arm}{args.suffix}"
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
        # 슬롯 수를 정확히 채운다.  R0-CONT 는 pseudo 슬롯까지 replay 로 메우고,
        # A12 는 비율이 달라 replay 수의 배수가 아닐 수 있다.
        replay_rng = random.Random(f"{seed}:{arm}:replay")
        replay_filled: list[str] = []
        pool = [f"replay__{name}" for name in replay]
        while len(replay_filled) < synthetic_count:
            replay_rng.shuffle(pool)
            replay_filled += pool[: synthetic_count - len(replay_filled)]
        entries += replay_filled

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

        report[f"{arm}{args.suffix}"] = {
            "sampling_seed": seed,
            "pseudo_fraction": pseudo_slots / total_slots,
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
        key = f"{arm}{args.suffix}"
        print(f"{key:20} filter={str(filter_name):16} unique_PL={unique_pseudo:<5} "
              f"epoch_entries={len(entries)}  seed={seed}  "
              f"pseudo/unique={report[key]['pseudo_exposures_per_unique']}")

    expected = total_slots
    for arm, stats in report.items():
        if stats["entries_per_epoch"] != expected:
            raise SystemExit(
                f"EXPOSURE_MISMATCH: {arm} {stats['entries_per_epoch']} != {expected}"
            )
    report_name = f"PSEUDO_DATASET_REPORT{args.suffix or ''}.json"
    (RESULTS / report_name).write_text(
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
