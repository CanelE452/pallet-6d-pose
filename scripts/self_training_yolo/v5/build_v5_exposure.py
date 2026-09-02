"""V5 — pool 에 reliability 점수를 매기고 노출 배분을 계산한다.  GT 를 쓰지 않는다.

V3-B 의 pseudo 라벨을 **그대로** 쓴다.  좌표도 true-ignore 마스크도 건드리지 않는다.
바뀌는 것은 train 목록에 각 프레임이 **몇 번 적히는가** 하나뿐이다.

    1  teacher cache 에서 frame·corner 신호를 모은다 (GT 없음)
    2  Day / Night 안에서 rank 정규화 -> R_total
    3  V3-B 의 Day/Night pseudo 슬롯 수를 그대로 읽는다
    4  프레임마다 1 회를 먼저 깔고, 남은 슬롯만 R_total 비례로 결정론적 배분
    5  V3-B 와 라벨 내용이 동일함을 해시로 증명한 뒤 dataset 을 쓴다
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))

from keypoint_scores import per_keypoint_scores  # noqa: E402
from reliability_score import (  # noqa: E402
    N_CORNERS, largest_remainder_allocation, score_pool,
)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
from pseudo_label_filters import geometry_scores  # noqa: E402

V1 = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
V5 = REPO_ROOT / "data/pallet/results/paper_selftrain_v5"
CACHE = V1 / "teacher_cache/R0_TEACHER_CACHE.json"
V3B = REPO_ROOT / ("challenge/yolo_pose_one_model/datasets/paper_selftrain_v3/"
                   "V3B_TRUE_IGNORE_AMBIG")
OUT_DATASET = REPO_ROOT / ("challenge/yolo_pose_one_model/datasets/"
                           "paper_selftrain_v5/V5_RELIABILITY_WEIGHTED")
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"

POOL_OBJECT_TYPE = "plastic_standard_110x130x11"
SCORES_CSV = V5 / "RELIABILITY_POOL_SCORES.csv"
SUMMARY = V5 / "RELIABILITY_POOL_SUMMARY.json"
RECEIPT = V5 / "V5_EXPOSURE_RECEIPT.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pool_dimensions() -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == POOL_OBJECT_TYPE:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {POOL_OBJECT_TYPE}")


def v3b_exposure() -> tuple[dict[str, int], list[str]]:
    """V3-B train 목록에서 프레임별 노출 횟수와 synthetic 항목을 읽는다."""

    entries = [line for line in (V3B / "train.txt").read_text().split() if line]
    pseudo = collections.Counter(e for e in entries if "/pseudo__" in e)
    synthetic = [e for e in entries if "/replay__" in e]
    return dict(pseudo), synthetic


def label_set_hash(directory: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(directory.glob("pseudo__*.txt"))
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(files)


def collect_signals() -> list[dict]:
    """teacher 예측만으로 frame·corner 신호를 모은다.  GT 는 열지 않는다."""

    lock = json.loads(LOCK.read_text())
    tau_box = float(lock["TAU_BOX"])
    validity = lock["keypoint_validity"]
    thresholds = lock["geometry_thresholds"]
    dimensions = pool_dimensions()
    cache = json.loads(CACHE.read_text())

    records: list[dict] = []
    for index, entry in enumerate(cache["entries"]):
        top = entry.get("top1")
        if not top or float(top["box_conf"]) < tau_box:
            continue
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.nan_to_num(
            np.asarray(top["keypoints_conf"], dtype=float), nan=0.0)
        camera = np.asarray(entry["camera_matrix"], dtype=float)
        flip = entry.get("flip_top1") or {}
        flip_xy = np.asarray(flip["keypoints_xy"], dtype=float) if flip else None
        flip_conf = (np.asarray(flip["keypoints_conf"], dtype=float)
                     if flip else None)
        corner = per_keypoint_scores(
            keypoints, confidence, camera, dimensions,
            flip_keypoints_2d=flip_xy, flip_conf=flip_conf,
            kp_conf_threshold=float(validity["kp_conf_threshold"]),
            remove_threshold=float(thresholds["tau_remove"]),
            flip_threshold=float(thresholds["tau_flip"]))
        frame = geometry_scores(
            keypoints, confidence >= float(validity["kp_conf_threshold"]),
            camera, dimensions, flip_xy,
            None if flip_conf is None
            else flip_conf >= float(validity["kp_conf_threshold"]))
        records.append({
            "frame_id": Path(entry["image_path"]).name,
            "image_path": entry["image_path"],
            "condition": entry["paper_condition"],
            "box_conf": float(top["box_conf"]),
            "s_reproj": float(frame["s_reproj"]),
            "s_remove": float(frame["s_remove"]),
            "s_flip": float(frame["s_flip"]) if frame["s_flip"] is not None
            else float("inf"),
            "kp_conf": [float(confidence[i]) for i in range(N_CORNERS)],
            "r_remove": [float(v) for v in corner["r_remove"]],
            "r_flip": [float(v) for v in corner["r_flip"]],
        })
        if (index + 1) % 200 == 0:
            print(f"  scored {index + 1}/{len(cache['entries'])}", flush=True)
    return records


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra = a[mask].argsort().argsort().astype(float)
    rb = b[mask].argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    scored = score_pool(collect_signals())
    by_frame = {r["frame_id"]: r for r in scored}
    print(f"box-accepted frames {len(scored)}", flush=True)

    v3_counts, synthetic = v3b_exposure()
    v3_by_name = {Path(k).name.replace("pseudo__", ""): v for k, v in v3_counts.items()}
    missing = set(v3_by_name) - set(by_frame)
    if missing:
        raise SystemExit(f"V3B_FRAME_NOT_SCORED: {len(missing)} 예: {sorted(missing)[:3]}")
    extra = set(by_frame) - set(v3_by_name)
    if extra:
        raise SystemExit(f"SCORED_FRAME_NOT_IN_V3B: {len(extra)}")

    # ── condition 별로 V3-B 슬롯 수를 그대로 유지하며 재배분 ─────────
    allocation: dict[str, int] = {}
    per_condition: dict[str, dict] = {}
    for condition in sorted({r["condition"] for r in scored}):
        names = sorted(n for n in v3_by_name if by_frame[n]["condition"] == condition)
        slots = sum(v3_by_name[n] for n in names)
        weights = [by_frame[n]["R_total"] for n in names]
        counts = largest_remainder_allocation(weights, slots)
        allocation.update(dict(zip(names, counts)))
        per_condition[condition] = {
            "unique_frames": len(names),
            "v3_slots": slots,
            "v5_slots": int(sum(counts)),
            "v3_exposure_range": [min(v3_by_name[n] for n in names),
                                  max(v3_by_name[n] for n in names)],
            "v5_exposure_range": [int(min(counts)), int(max(counts))],
            "R_total": {
                "min": float(np.min(weights)), "p05": float(np.percentile(weights, 5)),
                "p25": float(np.percentile(weights, 25)),
                "median": float(np.median(weights)),
                "p75": float(np.percentile(weights, 75)),
                "p95": float(np.percentile(weights, 95)),
                "max": float(np.max(weights)),
            },
            "spearman_vs": {
                name: spearman(weights, [by_frame[n][name] for n in names])
                for name in ("box_conf", "s_reproj", "s_remove", "s_flip")
            } | {"median_kp_conf": spearman(
                weights, [float(np.median(by_frame[n]["kp_conf"])) for n in names])},
        }

    # ── CSV ────────────────────────────────────────────────────────────
    V5.mkdir(parents=True, exist_ok=True)
    fields = ["frame_id", "condition", "box_conf", "s_reproj", "s_remove", "s_flip",
              "q_box_conf", "q_s_reproj", "q_s_remove", "q_s_flip",
              "R_kp_frame", "R_frame_geom", "R_total", "v3_exposures", "v5_exposures"]
    with SCORES_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(by_frame):
            row = by_frame[name]
            writer.writerow({
                "frame_id": name, "condition": row["condition"],
                "box_conf": f"{row['box_conf']:.6f}",
                "s_reproj": f"{row['s_reproj']:.6f}",
                "s_remove": f"{row['s_remove']:.6f}",
                "s_flip": f"{row['s_flip']:.6f}",
                "q_box_conf": f"{row['q_box_conf']:.6f}",
                "q_s_reproj": f"{row['q_s_reproj']:.6f}",
                "q_s_remove": f"{row['q_s_remove']:.6f}",
                "q_s_flip": f"{row['q_s_flip']:.6f}",
                "R_kp_frame": f"{row['R_kp_frame']:.6f}",
                "R_frame_geom": f"{row['R_frame_geom']:.6f}",
                "R_total": f"{row['R_total']:.6f}",
                "v3_exposures": v3_by_name[name],
                "v5_exposures": allocation[name],
            })

    SUMMARY.write_text(json.dumps({
        "schema_version": "v5_reliability_pool_summary_v1",
        "gt_used": False,
        "box_accepted_frames": len(scored),
        "by_condition": per_condition,
    }, indent=2, ensure_ascii=False) + "\n")

    # ── dataset — 라벨은 V3-B 것을 그대로 복사한다 ────────────────────
    images, labels = OUT_DATASET / "images" / "train", OUT_DATASET / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    listing: list[str] = []
    for name, count in sorted(allocation.items()):
        source_label = V3B / "labels" / "train" / f"pseudo__{name}"
        source_label = source_label.with_suffix(".txt")
        target_label = (labels / f"pseudo__{name}").with_suffix(".txt")
        target_label.write_text(source_label.read_text())
        link = images / f"pseudo__{name}"
        if not link.exists():
            link.symlink_to(REPO_ROOT / by_frame[name]["image_path"])
        listing += [str(link)] * count
    for entry in synthetic:
        name = Path(entry).name
        link = images / name
        if not link.exists():
            link.symlink_to((V3B / "images" / "train" / name).resolve())
        source = (V3B / "labels" / "train" / name).with_suffix(".txt")
        (labels / name).with_suffix(".txt").write_text(source.read_text())
        listing.append(str(link))

    import random
    random.Random(20260902).shuffle(listing)
    (OUT_DATASET / "train.txt").write_text("\n".join(listing) + "\n")
    (OUT_DATASET / "data.yaml").write_text(
        f"path: {OUT_DATASET}\ntrain: train.txt\nval: train.txt\n"
        "kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        "names:\n  0: item\n")

    # ── V3-B 와 라벨이 동일한지 증명 ──────────────────────────────────
    v3_hash, v3_n = label_set_hash(V3B / "labels" / "train")
    v5_hash, v5_n = label_set_hash(labels)
    if (v3_hash, v3_n) != (v5_hash, v5_n):
        raise SystemExit(
            f"PSEUDO_LABEL_CONTENT_DIFFERS: v3 {v3_n}/{v3_hash[:12]} "
            f"vs v5 {v5_n}/{v5_hash[:12]}")

    receipt = {
        "schema_version": "v5_exposure_receipt_v1",
        "pseudo_label_source": "V3B_TRUE_IGNORE_AMBIG (bit-identical)",
        "pseudo_label_set_sha256": v5_hash,
        "n_pseudo_labels": v5_n,
        "entries_per_epoch": len(listing),
        "pseudo_slots": int(sum(allocation.values())),
        "synthetic_slots": len(synthetic),
        "v3_entries_per_epoch": sum(v3_counts.values()) + len(synthetic),
        "v3_pseudo_slots": sum(v3_counts.values()),
        "by_condition": {c: {k: v for k, v in b.items()
                             if k not in ("R_total", "spearman_vs")}
                         for c, b in per_condition.items()},
        "most_repeated": sorted(allocation.items(), key=lambda kv: -kv[1])[:5],
        "least_repeated": sorted(allocation.items(), key=lambda kv: kv[1])[:5],
        "exposure_histogram": dict(sorted(
            collections.Counter(allocation.values()).items())),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'condition':11} {'unique':>7} {'v3 slots':>9} {'v5 slots':>9} "
          f"{'v3 range':>10} {'v5 range':>10}")
    for condition, block in per_condition.items():
        print(f"{condition:11} {block['unique_frames']:7d} {block['v3_slots']:9d} "
              f"{block['v5_slots']:9d} {str(block['v3_exposure_range']):>10} "
              f"{str(block['v5_exposure_range']):>10}")
    print(f"\nentries/epoch  V3-B {receipt['v3_entries_per_epoch']}  "
          f"V5 {receipt['entries_per_epoch']}")
    print(f"pseudo label set hash 일치: {v3_hash[:16]}  ({v5_n} labels)")
    print(f"exposure histogram: {receipt['exposure_histogram']}")
    for name in (SCORES_CSV, SUMMARY, RECEIPT):
        print(f"wrote {name.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
