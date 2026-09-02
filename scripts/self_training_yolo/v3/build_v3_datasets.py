"""V3 — arm 별 dataset.  V2 와 **pseudo 프레임·좌표·신뢰도 선택이 동일**해야 한다.

다른 것은 딱 하나, loss mask semantics 다.

    V2   신뢰할 수 없는 코너 -> visibility 0  (stock 에서 '보이지 않음' 을 학습)
    V3   신뢰할 수 없는 코너 -> visibility 1  (sentinel, 어느 항에도 안 들어감)

V2 와 membership·좌표가 같은지 해시로 대조한다 (§18).  다르면 실패로 멈춘다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))

from keypoint_scores import per_keypoint_scores  # noqa: E402

V1_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
V2_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v2"
V3_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v3"
CACHE = V1_RESULTS / "teacher_cache/R0_TEACHER_CACHE.json"          # 읽기 전용
REPLAY_LIST = V1_RESULTS / "SYNTHETIC_REPLAY_SUBSET.txt"            # 읽기 전용
V2_DATASETS = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v2"
METHOD_LOCK = V3_RESULTS / "SELFTRAIN_V3_METHOD_LOCK.json"
R0_DATASET = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
OUT_ROOT = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v3"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
POOL_OBJECT_TYPE = "plastic_standard_110x130x11"

N_KEYPOINTS = 9
N_CORNERS = 8
SUPERVISED, TRUE_IGNORE = 2, 1

ARMS = {
    # arm                     ambiguity mask
    "V3A_TRUE_IGNORE": False,
    "V3B_TRUE_IGNORE_AMBIG": True,
}
# 비교 기준이 되는 V2 arm — 같은 pseudo membership 을 가져야 한다.
V2_COUNTERPART = {"V3A_TRUE_IGNORE": "V2B_KP_MASK", "V3B_TRUE_IGNORE_AMBIG": "V2C_AMBIG"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_dimensions() -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == POOL_OBJECT_TYPE:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {POOL_OBJECT_TYPE}")


def assert_cache_is_already_unflipped(cache: dict) -> None:
    contract = (cache.get("flip_contract") or {}).get("unflip", "")
    if "width - 1 - x" not in contract:
        raise SystemExit(f"UNEXPECTED_FLIP_CONTRACT: {contract!r}")
    residuals = []
    for entry in cache["entries"][:50]:
        top, flip = entry.get("top1"), entry.get("flip_top1")
        if not top or not flip:
            continue
        a = np.asarray(top["keypoints_xy"], dtype=float)
        b = np.asarray(flip["keypoints_xy"], dtype=float)
        residuals.append(float(np.median(np.linalg.norm(a - b, axis=1))))
    if not residuals or float(np.median(residuals)) > 20.0:
        raise SystemExit("FLIP_CACHE_LOOKS_STILL_FLIPPED")


def assert_synthetic_labels_never_use_the_sentinel() -> None:
    """sentinel 이 자유로운지 데이터로 확인한다.  synthetic 에 1 이 있으면 계약이 깨진다."""

    offenders = []
    for name in REPLAY_LIST.read_text().split():
        label = (R0_DATASET / "labels" / "train" / name).with_suffix(".txt")
        if not label.exists():
            continue
        parts = label.read_text().split()
        values = parts[5:5 + N_KEYPOINTS * 3]
        if any(float(values[index]) == TRUE_IGNORE for index in range(2, len(values), 3)):
            offenders.append(name)
        if len(offenders) > 3:
            break
    if offenders:
        raise SystemExit(
            f"SYNTHETIC_LABEL_USES_SENTINEL_1: {offenders[:3]} — sentinel 을 바꿔야 한다")


def box_tokens(entry: dict) -> list[str] | None:
    top = entry["top1"]
    width, height = float(entry["image_width"]), float(entry["image_height"])
    x1, y1, x2, y2 = (float(value) for value in top["box_xyxy"])
    x1, x2 = max(0.0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(height, max(y1, y2))
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return [
        "0",
        f"{((x1 + x2) / 2) / width:.6f}", f"{((y1 + y2) / 2) / height:.6f}",
        f"{(x2 - x1) / width:.6f}", f"{(y2 - y1) / height:.6f}",
    ]


def label_line(entry: dict, scores: dict, ambiguity_aware: bool) -> str | None:
    """신뢰할 수 없는 코너에 **0 이 아니라 1** 을 쓴다."""

    tokens = box_tokens(entry)
    if tokens is None:
        return None
    top = entry["top1"]
    width, height = float(entry["image_width"]), float(entry["image_height"])
    keypoints = np.asarray(top["keypoints_xy"], dtype=float)

    keep = list(scores["keep_corner"])
    if ambiguity_aware and scores["ambiguous_view"]:
        keep = [False] * N_CORNERS
    visibility = [SUPERVISED if flag else TRUE_IGNORE for flag in keep]
    visibility.append(SUPERVISED if scores["keep_centroid"] else TRUE_IGNORE)

    for index in range(N_KEYPOINTS):
        x, y = keypoints[index]
        inside = 0.0 <= x < width and 0.0 <= y < height
        # 화면 밖 점도 무시한다 — V2 는 여기에 0 을 써서 negative supervision 을 줬다.
        value = visibility[index] if inside else TRUE_IGNORE
        tokens += [
            f"{min(max(x / width, 0.0), 1.0):.6f}",
            f"{min(max(y / height, 0.0), 1.0):.6f}",
            str(int(value)),
        ]
    return " ".join(tokens)


def coordinate_signature(line: str) -> str:
    """box + keypoint 좌표만.  visibility 는 뺀다 — V2 와 대조할 부분이 그것이다."""

    parts = line.split()
    coordinates = parts[:5]
    for index in range(5, len(parts), 3):
        coordinates += parts[index:index + 2]
    return hashlib.sha256(" ".join(coordinates).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=sorted(ARMS))
    parser.add_argument("--sampling-seed", type=int, default=20260902)
    args = parser.parse_args()

    lock = json.loads(METHOD_LOCK.read_text())
    thresholds = lock["thresholds_reused_from_v2_without_research"]
    cache = json.loads(CACHE.read_text())
    assert_cache_is_already_unflipped(cache)
    assert_synthetic_labels_never_use_the_sentinel()
    dimensions = pool_dimensions()

    pseudo_slots = int(lock["pseudo_slots_per_epoch"])
    synthetic_slots = int(lock["synthetic_slots_per_epoch"])
    tau_box = float(thresholds["box_conf_threshold"])

    scored: dict[str, dict] = {}
    accepted: list[dict] = []
    for index, entry in enumerate(cache["entries"]):
        top = entry.get("top1")
        if not top or float(top["box_conf"]) < tau_box:
            continue
        accepted.append(entry)
        flip = entry.get("flip_top1") or {}
        scored[entry["image_path"]] = per_keypoint_scores(
            np.asarray(top["keypoints_xy"], dtype=float),
            np.asarray(top["keypoints_conf"], dtype=float),
            np.asarray(entry["camera_matrix"], dtype=float), dimensions,
            flip_keypoints_2d=(np.asarray(flip["keypoints_xy"], dtype=float)
                               if flip else None),
            flip_conf=(np.asarray(flip["keypoints_conf"], dtype=float)
                       if flip else None),
            kp_conf_threshold=float(thresholds["kp_conf_threshold"]),
            remove_threshold=float(thresholds["remove_threshold"]),
            flip_threshold=float(thresholds["flip_threshold"]),
            ambiguity_threshold=float(thresholds["ambiguity_threshold"]),
        )
        if (index + 1) % 200 == 0:
            print(f"  scored {index + 1}/{len(cache['entries'])}", flush=True)
    print(f"box-accepted frames {len(accepted)}", flush=True)

    replay = [name for name in REPLAY_LIST.read_text().split()
              if (R0_DATASET / "labels" / "train" / name).with_suffix(".txt").exists()]

    report: dict = {
        "schema_version": "v3_pseudo_dataset_report_v1",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "teacher_cache_sha256": sha256_file(CACHE),
        "box_accepted_frames": len(accepted),
        "balanced_replay": False,
        "arms": {},
    }

    for arm in args.arms:
        ambiguity_aware = ARMS[arm]
        rng = random.Random(args.sampling_seed)
        dataset = OUT_ROOT / arm
        images, labels = dataset / "images" / "train", dataset / "labels" / "train"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        listing: list[str] = []

        chosen: list[str] = []
        pool = list(replay)
        while len(chosen) < synthetic_slots:
            rng.shuffle(pool)
            chosen += pool[: synthetic_slots - len(chosen)]
        for name in set(chosen):
            link = images / f"replay__{name}"
            if not link.exists():
                link.symlink_to(R0_DATASET / "images" / "train" / name)
            source = (R0_DATASET / "labels" / "train" / name).with_suffix(".txt")
            (labels / f"replay__{name}").with_suffix(".txt").write_text(source.read_text())
        listing += [str(images / f"replay__{name}") for name in chosen]

        usable: list[str] = []
        stats = {"supervised_corners": 0, "ignored_corners": 0,
                 "supervised_centroid": 0, "ambiguous_frames": 0,
                 "zero_mask_frames": 0}
        signatures: dict[str, str] = {}
        for entry in accepted:
            scores = scored[entry["image_path"]]
            line = label_line(entry, scores, ambiguity_aware)
            if line is None:
                continue
            name = Path(entry["image_path"]).name
            link = images / f"pseudo__{name}"
            if not link.exists():
                link.symlink_to(REPO_ROOT / entry["image_path"])
            (labels / f"pseudo__{name}").with_suffix(".txt").write_text(line + "\n")
            usable.append(str(link))
            signatures[f"pseudo__{name}"] = coordinate_signature(line)
            visibility = [int(v) for v in line.split()[7::3]]
            corners = visibility[:N_CORNERS]
            stats["supervised_corners"] += sum(1 for v in corners if v == SUPERVISED)
            stats["ignored_corners"] += sum(1 for v in corners if v == TRUE_IGNORE)
            stats["supervised_centroid"] += int(visibility[N_CORNERS] == SUPERVISED)
            stats["ambiguous_frames"] += int(bool(scores["ambiguous_view"]))
            stats["zero_mask_frames"] += int(all(v == TRUE_IGNORE for v in corners))

        if not usable:
            raise SystemExit(f"NO_PSEUDO_LABELS: {arm}")
        exposures: list[str] = []
        while len(exposures) < pseudo_slots:
            rng.shuffle(usable)
            exposures += usable[: pseudo_slots - len(exposures)]
        listing += exposures
        rng.shuffle(listing)
        (dataset / "train.txt").write_text("\n".join(listing) + "\n")
        (dataset / "data.yaml").write_text(
            f"path: {dataset}\n"
            "train: train.txt\n"
            "val: train.txt\n"
            "kpt_shape: [9, 3]\n"
            "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
            "names:\n  0: item\n"
        )

        # ── §18 V2 와 membership·좌표가 같은지 ─────────────────────────
        counterpart = V2_DATASETS / V2_COUNTERPART[arm] / "labels" / "train"
        mismatch = {"missing_in_v2": 0, "coordinate_differs": 0}
        for name, signature in signatures.items():
            other = (counterpart / name).with_suffix(".txt")
            if not other.exists():
                mismatch["missing_in_v2"] += 1
                continue
            if coordinate_signature(other.read_text().strip()) != signature:
                mismatch["coordinate_differs"] += 1
        if mismatch["missing_in_v2"] or mismatch["coordinate_differs"]:
            raise SystemExit(
                f"PSEUDO_MEMBERSHIP_MISMATCH: {arm} vs {V2_COUNTERPART[arm]} {mismatch}")

        report["arms"][arm] = {
            "unique_pseudo_frames": len(signatures),
            "pseudo_exposures_per_epoch": len(exposures),
            "synthetic_exposures_per_epoch": len(chosen),
            "entries_per_epoch": len(listing),
            "ambiguity_mask": ambiguity_aware,
            "v2_counterpart": V2_COUNTERPART[arm],
            "v2_membership_match": True,
            **stats,
        }
        print(f"{arm:24} pseudo {len(signatures):4d}  supervised corners "
              f"{stats['supervised_corners']:5d} / ignored {stats['ignored_corners']:5d}"
              f"  ambiguous {stats['ambiguous_frames']:3d}"
              f"  zero-mask {stats['zero_mask_frames']:3d}", flush=True)

    out = V3_RESULTS / "V3_PSEUDO_DATASET_REPORT.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
