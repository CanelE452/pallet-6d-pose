"""V2 — arm 별 Ultralytics pose dataset 을 만든다.  V1 산출물은 건드리지 않는다.

V1 과의 차이는 오직 **supervision contract** 다 (아키텍처·손실·하이퍼파라미터 동일).

    V2-A CONF25      box confidence 만.  keypoint 는 V1 처럼 conf 기준 hard label.
    V2-B KP-MASK     + 코너별 conf/removal/flip mask
    V2-C AMBIG-MASK  + q >= 0.75 프레임의 semantic corner 전부 mask (box·centroid 유지)
    V2-D FULL        + synthetic replay 를 q bin B0/B1/B2 에 균등 배분

프레임은 keypoint 때문에 버려지지 않는다.  box 가 없거나 confidence 가 낮을 때만
빠진다 (§5).

이미지는 복사하지 않고 symlink 한다.  중복 노출은 train 목록(.txt)에 같은 경로를
여러 번 적어 만든다 — V1 과 같은 방식이라 노출 계약이 비교 가능하다.
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

from keypoint_scores import (  # noqa: E402
    ambiguity_q,
    per_keypoint_scores,
    visibility_vector,
)

V1_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
V2_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v2"
CACHE = V1_RESULTS / "teacher_cache/R0_TEACHER_CACHE.json"      # 읽기 전용
REPLAY_LIST = V1_RESULTS / "SYNTHETIC_REPLAY_SUBSET.txt"        # 읽기 전용
METHOD_LOCK = V2_RESULTS / "SELFTRAIN_V2_METHOD_LOCK.json"
R0_DATASET = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
OUT_ROOT = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v2"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
# adaptation pool 은 전부 내 plastic 팔레트다 (MAIN_UNLABELED_BALANCED.csv).
POOL_OBJECT_TYPE = "plastic_standard_110x130x11"

N_KEYPOINTS = 9
N_CORNERS = 8
VISIBLE, NOT_LABELLED = 2, 0

ARMS = {
    # arm            per-keypoint mask   ambiguity mask   balanced replay
    "V2A_CONF25":    (False, False, False),
    "V2B_KP_MASK":   (True,  False, False),
    "V2C_AMBIG":     (True,  True,  False),
    "V2D_FULL":      (True,  True,  True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_dimensions() -> dict:
    """adaptation pool 의 물체 치수.  registry 에서 온다 — GT 어노테이션이 아니다."""

    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == POOL_OBJECT_TYPE:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {POOL_OBJECT_TYPE}")


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


def flip_prediction(flip_top: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """teacher cache 의 flip 예측을 **그대로** 쓴다.

    `dump_teacher_predictions.py` 가 이미 `x -> width - 1 - x` 와 flip_idx 재배정을
    끝내고 저장한다 (cache 의 `flip_contract.unflip` 이 그렇게 선언한다).  여기서
    한 번 더 되돌리면 좌표가 반대편으로 날아간다 — 실제로 그렇게 해서 flip 잔차
    중앙값이 1.9 px 대신 127 px 로 나왔고 코너의 94% 가 잘못 masked 됐다.
    """

    if not flip_top:
        return None
    keypoints = np.asarray(flip_top["keypoints_xy"], dtype=float)
    confidence = np.nan_to_num(
        np.asarray(flip_top.get("keypoints_conf", []), dtype=float), nan=0.0)
    if keypoints.shape[0] < N_KEYPOINTS or confidence.shape[0] < N_KEYPOINTS:
        return None
    return keypoints, confidence


def assert_cache_is_already_unflipped(cache: dict) -> None:
    """계약을 문서가 아니라 데이터로 확인한다."""

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
    if not residuals:
        raise SystemExit("NO_FLIP_PREDICTIONS_IN_CACHE")
    median = float(np.median(residuals))
    if median > 20.0:
        raise SystemExit(
            f"FLIP_CACHE_LOOKS_STILL_FLIPPED: median residual {median:.1f} px. "
            "이중 unflip 이거나 cache 계약이 바뀌었다.")


def label_line(entry: dict, scores: dict | None, arm_flags: tuple[bool, bool, bool],
               kp_conf_threshold: float) -> str | None:
    """box 는 항상, keypoint 는 계약에 따라."""

    per_keypoint, ambiguity_aware, _ = arm_flags
    tokens = box_tokens(entry)
    if tokens is None:
        return None

    top = entry["top1"]
    width, height = float(entry["image_width"]), float(entry["image_height"])
    keypoints = np.asarray(top["keypoints_xy"], dtype=float)
    confidence = np.nan_to_num(
        np.asarray(top["keypoints_conf"], dtype=float), nan=0.0)

    if per_keypoint and scores is not None:
        visibility = visibility_vector(scores, ambiguity_aware)
    else:
        # V2-A 는 V1 과 같은 규칙 — conf 와 화면 안 여부만 본다.
        visibility = []
        for index in range(N_KEYPOINTS):
            x, y = keypoints[index]
            inside = 0.0 <= x < width and 0.0 <= y < height
            visibility.append(
                VISIBLE if (confidence[index] >= kp_conf_threshold and inside)
                else NOT_LABELLED)

    for index in range(N_KEYPOINTS):
        x, y = keypoints[index]
        inside = 0.0 <= x < width and 0.0 <= y < height
        value = visibility[index] if inside else NOT_LABELLED
        tokens += [
            f"{min(max(x / width, 0.0), 1.0):.6f}",
            f"{min(max(y / height, 0.0), 1.0):.6f}",
            str(int(value)),
        ]
    return " ".join(tokens)


def synthetic_bins() -> dict[str, list[str]]:
    """replay 후보를 GT projected corner 의 q 로 세 bin 에 나눈다.

    synthetic 은 GT 라벨이 있으므로 여기서는 GT 를 쓴다 — pseudo-label 선택이
    아니라 **replay 구성**이다.
    """

    bins = {"B0": [], "B1": [], "B2": []}
    for name in REPLAY_LIST.read_text().split():
        label = (R0_DATASET / "labels" / "train" / name).with_suffix(".txt")
        if not label.exists():
            continue
        parts = label.read_text().split()
        if len(parts) < 5 + N_KEYPOINTS * 3:
            continue
        values = np.asarray(parts[5:5 + N_KEYPOINTS * 3], dtype=float).reshape(-1, 3)
        q = ambiguity_q(values[:, :2])
        if not np.isfinite(q):
            continue
        bins["B0" if q < 0.50 else "B1" if q < 0.75 else "B2"].append(name)
    return bins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=sorted(ARMS))
    parser.add_argument("--sampling-seed", type=int, default=20260902)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    lock = json.loads(METHOD_LOCK.read_text())
    cache = json.loads(CACHE.read_text())
    assert_cache_is_already_unflipped(cache)
    dimensions = pool_dimensions()

    pseudo_slots = int(lock["pseudo_slots_per_epoch"])
    synthetic_slots = int(lock["synthetic_slots_per_epoch"])
    kp_conf = float(lock["kp_conf_threshold"])
    tau_box = float(lock["box_conf_threshold"])

    # ── 코너별 score 를 한 번만 계산해 arm 들이 공유한다 ────────────────
    scored: dict[str, dict] = {}
    accepted: list[dict] = []
    print(f"teacher entries {len(cache['entries'])}", flush=True)
    for index, entry in enumerate(cache["entries"]):
        top = entry.get("top1")
        if not top or float(top["box_conf"]) < tau_box:
            continue
        accepted.append(entry)
        flipped = flip_prediction(entry.get("flip_top1") or {})
        camera = np.asarray(entry["camera_matrix"], dtype=float)
        scored[entry["image_path"]] = per_keypoint_scores(
            np.asarray(top["keypoints_xy"], dtype=float),
            np.asarray(top["keypoints_conf"], dtype=float),
            camera, dimensions,
            flip_keypoints_2d=None if flipped is None else flipped[0],
            flip_conf=None if flipped is None else flipped[1],
            kp_conf_threshold=kp_conf,
            remove_threshold=float(lock["remove_threshold"]),
            flip_threshold=float(lock["flip_threshold"]),
            ambiguity_threshold=float(lock["ambiguity_threshold"]),
        )
        if (index + 1) % 200 == 0:
            print(f"  scored {index + 1}/{len(cache['entries'])}", flush=True)
    print(f"box-accepted frames {len(accepted)}", flush=True)

    bins = synthetic_bins()
    print("synthetic replay bins: "
          + "  ".join(f"{k} {len(v)}" for k, v in bins.items()), flush=True)

    report: dict = {
        "schema_version": "v2_pseudo_dataset_report_v1",
        "method_lock_sha256": sha256_file(METHOD_LOCK),
        "teacher_cache_sha256": sha256_file(CACHE),
        "box_accepted_frames": len(accepted),
        "synthetic_bin_sizes": {key: len(value) for key, value in bins.items()},
        "arms": {},
    }

    for arm in args.arms:
        flags = ARMS[arm]
        rng = random.Random(args.sampling_seed)
        dataset = OUT_ROOT / f"{arm}{args.suffix}"
        images = dataset / "images" / "train"
        labels = dataset / "labels" / "train"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        listing_entries: list[str] = []

        # ── synthetic replay ────────────────────────────────────────────
        if flags[2]:
            per_bin = synthetic_slots // 3
            chosen: list[str] = []
            for key in ("B0", "B1", "B2"):
                pool = bins[key]
                if not pool:
                    raise SystemExit(f"EMPTY_REPLAY_BIN: {key}")
                picked = []
                while len(picked) < per_bin:
                    rng.shuffle(pool)
                    picked += pool[: per_bin - len(picked)]
                chosen += picked
            chosen += chosen[: synthetic_slots - len(chosen)]
        else:
            pool = [name for group in bins.values() for name in group]
            chosen = []
            while len(chosen) < synthetic_slots:
                rng.shuffle(pool)
                chosen += pool[: synthetic_slots - len(chosen)]

        for name in set(chosen):
            link = images / f"replay__{name}"
            if not link.exists():
                link.symlink_to(R0_DATASET / "images" / "train" / name)
            source = (R0_DATASET / "labels" / "train" / name).with_suffix(".txt")
            (labels / f"replay__{name}").with_suffix(".txt").write_text(
                source.read_text())
        listing_entries += [str(images / f"replay__{name}") for name in chosen]

        # ── pseudo-real ────────────────────────────────────────────────
        written = 0
        usable: list[str] = []
        stats = {"ambiguous": 0, "kept_corners": 0, "masked_corners": 0,
                 "kept_centroid": 0}
        for entry in accepted:
            scores = scored[entry["image_path"]]
            line = label_line(entry, scores, flags, kp_conf)
            if line is None:
                continue
            name = Path(entry["image_path"]).name
            link = images / f"pseudo__{name}"
            if not link.exists():
                link.symlink_to(REPO_ROOT / entry["image_path"])
            (labels / f"pseudo__{name}").with_suffix(".txt").write_text(line + "\n")
            usable.append(str(link))
            written += 1
            visibility = [int(v) for v in line.split()[5 + 2::3]]
            stats["kept_corners"] += sum(1 for v in visibility[:N_CORNERS] if v)
            stats["masked_corners"] += sum(1 for v in visibility[:N_CORNERS] if not v)
            stats["kept_centroid"] += int(bool(visibility[N_CORNERS]))
            stats["ambiguous"] += int(bool(scores["ambiguous_view"]))

        if not usable:
            raise SystemExit(f"NO_PSEUDO_LABELS: {arm}")
        exposures: list[str] = []
        while len(exposures) < pseudo_slots:
            rng.shuffle(usable)
            exposures += usable[: pseudo_slots - len(exposures)]
        listing_entries += exposures

        rng.shuffle(listing_entries)
        (dataset / "train.txt").write_text("\n".join(listing_entries) + "\n")
        (dataset / "data.yaml").write_text(
            f"path: {dataset}\n"
            "train: train.txt\n"
            "val: train.txt\n"
            "kpt_shape: [9, 3]\n"
            "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
            "names:\n  0: item\n"
        )
        report["arms"][arm] = {
            "unique_pseudo_frames": written,
            "pseudo_exposures_per_epoch": len(exposures),
            "synthetic_exposures_per_epoch": len(chosen),
            "entries_per_epoch": len(listing_entries),
            "balanced_replay": flags[2],
            "per_keypoint_mask": flags[0],
            "ambiguity_mask": flags[1],
            **stats,
        }
        print(f"{arm:14} pseudo {written:4d} unique  "
              f"corners kept {stats['kept_corners']:5d} / masked "
              f"{stats['masked_corners']:5d}  ambiguous {stats['ambiguous']:3d}",
              flush=True)

    out = V2_RESULTS / f"V2_PSEUDO_DATASET_REPORT{args.suffix}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
