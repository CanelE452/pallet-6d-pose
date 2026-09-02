"""M4 — filter 가 남기는 teacher prediction 의 품질을 GT 로 채점한다.

이 단계는 `PSEUDOLABEL_FILTER_LOCK.json` 이 commit 된 **뒤에만** 돌린다.
GT 는 여기서 처음 열리고, **평가에만** 쓴다.  TAU_BOX / tau_reproj / tau_remove /
tau_flip 을 이 결과를 보고 고치지 않는다.

묻는 것은 둘이다.

    1. confidence 가 0.7~0.8 을 넘으면 실제로 더 맞는가?
    2. geometry filter 가 confidence 위에 품질을 더 얹는가, 개수만 줄이는가?

정답 기준은 새로 만들지 않는다.  `metric_split_lock.md` §2.2 가 이미 동결한 값을 쓴다.

    gross          = keypoint error > 20 px
    catastrophic   = keypoint error > 40 px
    CORRECT_2D     = 검출됨 AND 감독 keypoint 중 gross 가 하나도 없음

primary 는 연속값(pass/reject median corner error 와 그 separation)이다.
binary precision/recall 은 위 frozen threshold 위에서만 보조로 낸다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo_label_filters import geometry_scores  # noqa: E402

MANIFEST = REPO_ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_PLASTIC_POS.json"
LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
TEACHER = (
    REPO_ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs"
    / "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
)
OUT = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json"
RECORDS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"

PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]
N_CORNERS = 8
GROSS_PX = 20.0          # metric_split_lock.md §2.2 [LOCKED]
CATASTROPHIC_PX = 40.0   # metric_split_lock.md §2.2 [LOCKED]

ARMS = ("F0_NAIVE", "F1_CONF", "F2_CONF_REPROJ", "F3_CONF_REMOVE",
        "F5_CONF_FLIP", "F4_PROPOSED")
READER = {
    "F0_NAIVE": "No filter",
    "F1_CONF": "Confidence",
    "F2_CONF_REPROJ": "Confidence + Reprojection",
    "F3_CONF_REMOVE": "Confidence + Keypoint-removal consistency",
    "F5_CONF_FLIP": "Confidence + Horizontal-flip consistency",
    "F4_PROPOSED": "Proposed",
}
CONF_BINS = ((0.0, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_dimensions(object_type: str) -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == object_type:
            return entry["physical_dimensions_m"]
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def collect(model, lock) -> list[dict]:
    kp_valid = float(lock["keypoint_validity"]["kp_conf_threshold"])
    min_corners = int(lock["keypoint_validity"]["min_valid_corners"])
    manifest = json.loads(MANIFEST.read_text())
    dimensions = registry_dimensions(manifest["object_types"][0])

    rows: list[dict] = []
    for index, item in enumerate(manifest["items"]):
        image_path = REPO_ROOT / item["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"UNREADABLE_IMAGE: {image_path}")
        height, width = image.shape[:2]

        payload = json.loads((REPO_ROOT / item["gt_v2_path"]).read_text())
        obj = payload["objects"][0]
        intrinsics = payload["camera_data"]["intrinsics"]
        K = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                      [0.0, intrinsics["fy"], intrinsics["cy"]],
                      [0.0, 0.0, 1.0]], dtype=float)
        annotations = obj["keypoint_annotations"]
        gt_xy = np.array([a["xy"] for a in annotations], dtype=float)
        supervised = np.array([
            bool(a.get("visibility", 0)) and a.get("in_frame", True)
            for a in annotations
        ])
        # legacy 프레임은 visibility 가 unknown 이라 supervision mask 가 통째로 빈다.
        # 그 경우에도 값을 낼 수 있게 all-annotated 를 함께 모은다 (진단이지
        # visible/occluded 주장이 아니다).
        present = np.array([
            a.get("xy") is not None and a.get("in_frame", True) for a in annotations
        ])

        record = {
            "frame_id": item["frame_id"],
            "domain": item.get("domain"),
            "detected": False,
            "box_conf": None,
            "valid_corners": 0,
            "corner_median_px": None,
            "corner_max_px": None,
            "gross_keypoints": None,
            "s_reproj": None, "s_remove": None, "s_flip": None,
        }

        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            keypoints = result.keypoints.xy.cpu().numpy()[best] - PAD
            confidences = (
                result.keypoints.conf.cpu().numpy()[best]
                if result.keypoints.conf is not None
                else np.zeros(9)
            )
            valid = np.nan_to_num(confidences, nan=0.0) >= kp_valid
            record["detected"] = True
            record["box_conf"] = float(result.boxes.conf.cpu().numpy()[best])
            record["valid_corners"] = int(np.count_nonzero(valid[:N_CORNERS]))

            distances = np.linalg.norm(keypoints - gt_xy, axis=1)
            annotated = distances[present]
            if annotated.size:
                record["errors_annotated_px"] = annotated.tolist()
            errors = distances[supervised]
            if errors.size:
                record["corner_median_px"] = float(np.median(errors))
                record["corner_max_px"] = float(errors.max())
                record["gross_keypoints"] = int(np.count_nonzero(errors > GROSS_PX))
                record["catastrophic_keypoints"] = int(
                    np.count_nonzero(errors > CATASTROPHIC_PX)
                )
                record["errors_px"] = errors.tolist()

            if record["valid_corners"] >= min_corners:
                flipped = cv2.flip(image, 1)
                flip_padded = cv2.copyMakeBorder(
                    flipped, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101
                )
                flip_result = model.predict(
                    flip_padded, imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False
                )[0]
                flip_keypoints = flip_valid = None
                if flip_result.boxes is not None and len(flip_result.boxes):
                    j = int(np.argmax(flip_result.boxes.conf.cpu().numpy()))
                    raw = flip_result.keypoints.xy.cpu().numpy()[j] - PAD
                    raw_conf = (
                        flip_result.keypoints.conf.cpu().numpy()[j]
                        if flip_result.keypoints.conf is not None
                        else np.zeros(9)
                    )
                    unflipped = np.stack([width - 1 - raw[:, 0], raw[:, 1]], axis=1)
                    flip_keypoints = unflipped[FLIP_IDX]
                    flip_valid = np.nan_to_num(raw_conf, nan=0.0)[FLIP_IDX] >= kp_valid
                scores = geometry_scores(
                    keypoints, valid, K, dimensions, flip_keypoints, flip_valid
                )
                record["s_reproj"] = scores["s_reproj"]
                record["s_remove"] = scores["s_remove"]
                record["s_flip"] = scores["s_flip"]
            record["keypoints_xy"] = keypoints.tolist()
            record["keypoint_valid"] = valid.tolist()
            record["image_path"] = item["image_path"]
            record["gt_xy"] = gt_xy.tolist()
            record["gt_supervised"] = supervised.tolist()
        rows.append(record)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(manifest['items'])}", flush=True)
    return rows


def passes(record: dict, arm: str, lock: dict) -> bool:
    min_corners = int(lock["keypoint_validity"]["min_valid_corners"])
    tau_box = float(lock["TAU_BOX"])
    thresholds = lock["geometry_thresholds"]
    if not record["detected"] or record["valid_corners"] < min_corners:
        return False
    if arm == "F0_NAIVE":
        return True
    if record["box_conf"] is None or record["box_conf"] < tau_box:
        return False
    if arm == "F1_CONF":
        return True
    if arm == "F2_CONF_REPROJ":
        return (record["s_reproj"] is not None
                and record["s_reproj"] <= float(thresholds["tau_reproj"]))
    flip = (record["s_flip"] is not None
            and record["s_flip"] <= float(thresholds["tau_flip"]))
    if arm == "F5_CONF_FLIP":
        return flip
    removal = (record["s_remove"] is not None
               and record["s_remove"] <= float(thresholds["tau_remove"]))
    if arm == "F3_CONF_REMOVE":
        return removal
    return removal and flip


def show(value, spec: str = ".3f") -> str:
    """0.0 은 falsy 다.  `value or nan` 을 쓰면 0 이 nan 으로 둔갑한다."""

    return "—" if value is None else format(value, spec)


def pooled(records: list[dict], key: str = "errors_px") -> dict:
    errors: list[float] = []
    for record in records:
        errors += record.get(key, [])
    if not errors:
        return {"n_frames": len(records), "n_keypoints": 0,
                "median_px": None, "p90_px": None, "gross_rate": None}
    values = np.asarray(errors)
    return {
        "n_frames": len(records),
        "n_keypoints": int(values.size),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "gross_rate": float(np.mean(values > GROSS_PX)),
        "catastrophic_rate": float(np.mean(values > CATASTROPHIC_PX)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    lock = json.loads(LOCK.read_text())
    model = YOLO(str(TEACHER), task="pose")
    print(f"teacher {TEACHER.name}  TAU_BOX {lock['TAU_BOX']}", flush=True)
    rows = collect(model, lock)

    correct = {
        record["frame_id"]: bool(
            record["detected"] and record.get("gross_keypoints") == 0
        )
        for record in rows
    }
    total_correct = sum(correct.values())

    table: dict[str, dict] = {}
    print(f"\n{'filter':44} {'pass':>5} {'pass~px':>8} {'rej~px':>8} "
          f"{'sep':>7} {'gross':>7} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("─" * 104)
    for arm in ARMS:
        accepted = [r for r in rows if passes(r, arm, lock)]
        rejected = [r for r in rows if not passes(r, arm, lock)]
        pass_stats, reject_stats = pooled(accepted), pooled(rejected)
        tp = sum(correct[r["frame_id"]] for r in accepted)
        fp = len(accepted) - tp
        fn = total_correct - tp
        precision = tp / len(accepted) if accepted else None
        recall = tp / total_correct if total_correct else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else None)
        separation = (
            reject_stats["median_px"] - pass_stats["median_px"]
            if pass_stats["median_px"] is not None and reject_stats["median_px"] is not None
            else None
        )
        table[arm] = {
            "reader_facing_name": READER[arm],
            "accepted": len(accepted),
            "retention": len(accepted) / len(rows),
            "pass": pass_stats,
            "reject": reject_stats,
            "separation_px": separation,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
        print(f"{READER[arm]:44} {len(accepted):>5} "
              f"{show(pass_stats['median_px'], '.2f'):>8} "
              f"{show(reject_stats['median_px'], '.2f'):>8} "
              f"{show(separation, '.2f'):>7} "
              f"{show(pass_stats['gross_rate']):>7} "
              f"{show(precision):>6} {show(recall):>6} {show(f1):>6}")

    bins = {}
    print(f"\n{'box_conf bin':16} {'N':>5} {'src':>6} {'n_kp':>6} {'corner~px':>10} "
          f"{'p90':>8} {'gross':>7}")
    print("─" * 66)
    for low, high in CONF_BINS:
        subset = [r for r in rows
                  if r["detected"] and low <= (r["box_conf"] or 0.0) < high]
        stats = pooled(subset)
        # supervision mask 가 빈 bin(전부 legacy)은 all-annotated 로 채우고 출처를 남긴다.
        stats["source"] = "strict"
        if stats["n_keypoints"] == 0:
            stats = pooled(subset, "errors_annotated_px")
            stats["source"] = "diag"
        bins[f"[{low:.2f},{high:.2f})"] = stats
        print(f"{f'[{low:.2f},{high:.2f})':16} {stats['n_frames']:>5} "
              f"{stats['source']:>6} {stats['n_keypoints']:>6} "
              f"{show(stats['median_px'], '.2f'):>10} "
              f"{show(stats['p90_px'], '.2f'):>8} {show(stats['gross_rate']):>7}")

    report = {
        "schema_version": "paper_m4_filter_quality_v1",
        "population": "PAPER_EVAL_PLASTIC_POS",
        "population_sha256": sha256_file(MANIFEST),
        "teacher_sha256": sha256_file(TEACHER),
        "filter_lock_sha256": sha256_file(LOCK),
        "gt_used_for": "evaluation only; thresholds were frozen before this ran",
        "criterion": {
            "CORRECT_2D": "detected AND no supervised keypoint error > 20 px",
            "gross_px": GROSS_PX,
            "catastrophic_px": CATASTROPHIC_PX,
            "source": "metric_split_lock.md §2.2 [LOCKED]",
            "primary": "continuous pass/reject median corner error and separation",
        },
        "n_frames": len(rows),
        "n_detected": sum(r["detected"] for r in rows),
        "n_correct_2d": total_correct,
        "filters": table,
        "confidence_bins": bins,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    # 프레임 단위 기록을 남긴다.  visual audit 이 같은 추론·같은 판정을 다시
    # 구현하면 두 벌이 갈라진다.
    RECORDS.write_text(json.dumps({
        "schema_version": "m4_frame_records_v1",
        "population": report["population"],
        "teacher_sha256": report["teacher_sha256"],
        "filter_lock_sha256": report["filter_lock_sha256"],
        "gross_px": GROSS_PX,
        "arms": list(ARMS),
        "frames": [
            {**row, "verdict": {arm: passes(row, arm, lock) for arm in ARMS}}
            for row in rows
        ],
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {RECORDS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
