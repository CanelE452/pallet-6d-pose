"""teacher cache 하나에서 F0~F4 pseudo-label manifest 를 만든다.

arm 마다 YOLO 를 다시 돌리지 않는다.  같은 cache 에 gate 만 다르게 건다.

    F0 NAIVE      top1 존재 AND valid corner >= 6              (quality gate 없음)
    F1 CONF       F0 AND box_conf >= TAU_BOX
    F2 CONF+REPROJ    F1 AND s_reproj <= tau_reproj
    F3 CONF+REMOVE    F1 AND s_remove <= tau_remove
    F4 PROPOSED       F1 AND s_remove <= tau_remove AND s_flip <= tau_flip

threshold 는 전부 `PSEUDOLABEL_FILTER_LOCK.json` 에서 읽는다.  여기서 새로 정하지
않는다.  GT 는 열지 않는다 — 이 단계는 아직 정답을 모른다.

pseudo supervision 으로 저장하는 것은 2D box + 2D keypoint + validity 뿐이다.
geometry score 를 계산하면서 푼 pose 는 filtering 을 위한 latent check 일 뿐이고
pseudo 6D GT 로 저장하지 않는다.

adaptation pool 의 object type
    pool 세션(capturepallet*, capturenight*)은 eval 쪽 같은 촬영 계열이 전부
    plastic 으로 라벨돼 있다.  그래서 registry 의 plastic 치수를 쓴다.  이건
    per-frame GT 가 아니라 촬영 계열 provenance 에서 오는 상수다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo_label_filters import geometry_scores  # noqa: E402

CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/pseudo_manifests"

POOL_OBJECT_TYPE = "plastic_standard_110x130x11"
N_CORNERS = 8

# F5 는 flip **단독** arm 이다.  F4(Proposed)는 removal 과 flip 을 함께 걸므로
# 누적 표만으로는 flip 고유 기여를 못 뽑는다 — F3(removal 단독)과 F5(flip 단독)를
# 나란히 둬야 두 필터를 각각 평가할 수 있다.
ARMS = ("F0_NAIVE", "F1_CONF", "F2_CONF_REPROJ", "F3_CONF_REMOVE",
        "F5_CONF_FLIP", "F4_PROPOSED")
READER_FACING = {
    "F0_NAIVE": "No filter",
    "F1_CONF": "Confidence",
    "F2_CONF_REPROJ": "Confidence + Reprojection",
    "F3_CONF_REMOVE": "Confidence + Keypoint-removal consistency",
    "F5_CONF_FLIP": "Confidence + Horizontal-flip consistency",
    "F4_PROPOSED": "Proposed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_dimensions(object_type: str) -> dict:
    registry = json.loads(REGISTRY.read_text())
    for entry in registry["objects"]:
        if entry["object_type"] == object_type:
            return entry["physical_dimensions_m"]
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def valid_mask(confidences, threshold: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(confidences, dtype=float), nan=0.0)
    return values >= threshold


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    tau_box = float(lock["TAU_BOX"])
    kp_valid = float(lock["keypoint_validity"]["kp_conf_threshold"])
    min_corners = int(lock["keypoint_validity"]["min_valid_corners"])
    tau_reproj = float(lock["geometry_thresholds"]["tau_reproj"])
    tau_remove = float(lock["geometry_thresholds"]["tau_remove"])
    tau_flip = float(lock["geometry_thresholds"]["tau_flip"])
    print(f"TAU_BOX {tau_box}  kp_conf {kp_valid}  min_corners {min_corners}  "
          f"tau_reproj {tau_reproj}  tau_remove {tau_remove}  tau_flip {tau_flip}")

    dimensions = registry_dimensions(POOL_OBJECT_TYPE)
    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text())

    scored: list[dict] = []
    started = time.time()
    for position, entry in enumerate(cache["entries"]):
        record = {
            "image_path": entry["image_path"],
            "image_sha256": entry["image_sha256"],
            "paper_condition": entry["paper_condition"],
            "capture_session": entry["capture_session"],
            "detected": entry["top1"] is not None,
            "valid_corners": 0,
            "box_conf": None,
            "kp_conf_median8": None,
            "s_reproj": None,
            "s_remove": None,
            "s_flip": None,
        }
        top = entry["top1"]
        if top is not None:
            keypoints = np.asarray(top["keypoints_xy"], dtype=float)
            valid = valid_mask(top["keypoints_conf"], kp_valid)
            record["valid_corners"] = int(np.count_nonzero(valid[:N_CORNERS]))
            record["box_conf"] = float(top["box_conf"])
            record["kp_conf_median8"] = float(top["kp_conf_median8"])
            record["box_xyxy"] = [float(v) for v in top["box_xyxy"]]
            record["keypoints_xy"] = keypoints.tolist()
            record["keypoint_valid"] = valid.tolist()

            camera_matrix = entry.get("camera_matrix")
            if camera_matrix is not None and record["valid_corners"] >= min_corners:
                flip = entry.get("flip_top1")
                flip_keypoints = flip_valid = None
                if flip is not None:
                    flip_keypoints = np.asarray(flip["keypoints_xy"], dtype=float)
                    flip_valid = valid_mask(flip["keypoints_conf"], kp_valid)
                scores = geometry_scores(
                    keypoints, valid, np.asarray(camera_matrix, dtype=float),
                    dimensions, flip_keypoints, flip_valid,
                )
                record["s_reproj"] = scores["s_reproj"]
                record["s_remove"] = scores["s_remove"]
                record["s_flip"] = scores["s_flip"]
        scored.append(record)
        if (position + 1) % 250 == 0:
            print(f"  scored {position + 1}/{len(cache['entries'])} "
                  f"({(position + 1) / (time.time() - started):.0f}/s)", flush=True)

    def passes(record: dict, arm: str) -> bool:
        if not record["detected"] or record["valid_corners"] < min_corners:
            return False
        if arm == "F0_NAIVE":
            return True
        if record["box_conf"] is None or record["box_conf"] < tau_box:
            return False
        if arm == "F1_CONF":
            return True
        if arm == "F2_CONF_REPROJ":
            return record["s_reproj"] is not None and record["s_reproj"] <= tau_reproj
        flip_ok = record["s_flip"] is not None and record["s_flip"] <= tau_flip
        if arm == "F5_CONF_FLIP":
            return flip_ok
        removal_ok = record["s_remove"] is not None and record["s_remove"] <= tau_remove
        if arm == "F3_CONF_REMOVE":
            return removal_ok
        if arm == "F4_PROPOSED":
            return removal_ok and flip_ok
        raise SystemExit(f"UNKNOWN_ARM: {arm}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    fields = ("image_path", "image_sha256", "paper_condition", "capture_session",
              "box_conf", "kp_conf_median8", "valid_corners",
              "s_reproj", "s_remove", "s_flip")

    print(f"\n{'arm':26} {'accepted':>8} {'day':>6} {'night':>6} "
          f"{'conf~':>7} {'reproj~':>8} {'remove~':>8} {'flip~':>7}")
    print("─" * 82)
    for arm in ARMS:
        accepted = [record for record in scored if passes(record, arm)]
        condition = Counter(record["paper_condition"] for record in accepted)
        target = OUT_DIR / f"{arm}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in accepted:
                writer.writerow({key: record.get(key) for key in fields})

        stats = {
            "reader_facing_name": READER_FACING[arm],
            "accepted": len(accepted),
            "daytime_accepted": condition.get("daytime", 0),
            "nighttime_accepted": condition.get("nighttime", 0),
            "median_box_conf": _median([r["box_conf"] for r in accepted]),
            "median_kp_conf": _median([r["kp_conf_median8"] for r in accepted]),
            "median_s_reproj": _median([r["s_reproj"] for r in accepted]),
            "median_s_remove": _median([r["s_remove"] for r in accepted]),
            "median_s_flip": _median([r["s_flip"] for r in accepted]),
            "manifest": str(target.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(target),
        }
        summary[arm] = stats
        print(f"{arm:26} {stats['accepted']:>8} {stats['daytime_accepted']:>6} "
              f"{stats['nighttime_accepted']:>6} "
              f"{(stats['median_box_conf'] or float('nan')):>7.3f} "
              f"{(stats['median_s_reproj'] or float('nan')):>8.4f} "
              f"{(stats['median_s_remove'] or float('nan')):>8.4f} "
              f"{(stats['median_s_flip'] or float('nan')):>7.4f}")

    total = len(scored)
    detected = sum(record["detected"] for record in scored)
    candidate = sum(
        record["detected"] and record["valid_corners"] >= min_corners for record in scored
    )
    funnel = {
        "total": total,
        "detected": detected,
        "candidate_min_valid_corners": candidate,
        "confidence": summary["F1_CONF"]["accepted"],
        "confidence_reprojection": summary["F2_CONF_REPROJ"]["accepted"],
        "confidence_keypoint_removal": summary["F3_CONF_REMOVE"]["accepted"],
        "confidence_flip": summary["F5_CONF_FLIP"]["accepted"],
        "proposed": summary["F4_PROPOSED"]["accepted"],
    }
    print("\nfunnel:", " -> ".join(f"{k} {v}" for k, v in funnel.items()))

    report = {
        "schema_version": "paper_pseudo_manifest_summary_v1",
        "teacher_cache": str(cache_path.relative_to(REPO_ROOT)),
        "teacher_cache_sha256": sha256_file(cache_path),
        "teacher_sha256": cache["teacher_sha256"],
        "filter_lock": str(LOCK.relative_to(REPO_ROOT)),
        "filter_lock_sha256": sha256_file(LOCK),
        "thresholds": {
            "TAU_BOX": tau_box, "kp_conf": kp_valid, "min_valid_corners": min_corners,
            "tau_reproj": tau_reproj, "tau_remove": tau_remove, "tau_flip": tau_flip,
        },
        "pool_object_type": POOL_OBJECT_TYPE,
        "pool_object_type_source": (
            "capture-series provenance: the eval subsets of the same capture series "
            "(capturepallet*, capturenight*) are all labelled plastic.  This is a "
            "constant from provenance, not per-frame ground truth."
        ),
        "registry_dimensions_m": dimensions,
        "gt_used": False,
        "funnel": funnel,
        "arms": summary,
    }
    (OUT_DIR / "PSEUDO_MANIFEST_SUMMARY.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    # 진단용 전체 score 덤프.  M4 filter-quality 단계가 이걸 읽는다.
    with (OUT_DIR / "ALL_SCORED.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("detected",) + fields)
        writer.writeheader()
        for record in scored:
            writer.writerow({key: record.get(key) for key in ("detected",) + fields})
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
