"""TAU_BOX 를 unlabeled pool 만 보고 고정한다.  PAPER_EVAL GT 를 열지 않는다.

"real label-free adaptation" 을 주장하려면 threshold 선택에 target GT 가 들어가면
안 된다.  그래서 여기서 쓰는 정보는 teacher cache 의 예측 분포뿐이고, 정답은 한 번도
읽지 않는다.

score 는 `result.boxes.conf` 하나다.  논문 표기는 **YOLO detection confidence**.
calibrated probability 라고 부르지 않는다.  `box_conf * kp_conf` 같은 합성 score 를
새로 만들지도 않는다 — keypoint confidence 는 valid corner 판정과 진단에만 쓴다.

candidate 분모 (gate 이전):

    top1 detection 존재  AND  valid corner >= 6 / 8      (kp_conf >= 0.5)

선택 규칙 (사전등록):

    두 조건(daytime, nighttime) 모두에서
        accepted >= max(100, 20% of that condition's candidate pool)
    를 만족하는 candidate threshold 중 **가장 높은 값**.

아무것도 만족 못 하면 TAU_BOX=0.70 으로 내려가고 CONF_RETENTION_FAIL=true 를 남긴다.
결과를 본 뒤 이 값을 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
OUT = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"

# repo 전반의 기존 규약.  infer_video_yolo.py / eval_ab_crop.py /
# pallet_jetson_deploy/infer_fps.py 가 모두 --kp_conf default 0.5 를 쓴다.
KP_CONF_VALID = 0.5
MIN_VALID_CORNERS = 6
N_CORNERS = 8

CANDIDATE_TAUS = (0.70, 0.75, 0.80, 0.85)
FALLBACK_TAU = 0.70
ABSOLUTE_MINIMUM = 100
RETENTION_FRACTION = 0.20

# 진단용 bin.  선택 규칙에는 쓰지 않는다.
CONF_BINS = ((0.0, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_corner_count(entry: dict) -> int:
    top = entry.get("top1")
    if not top:
        return 0
    conf = np.asarray(top["keypoints_conf"][:N_CORNERS], dtype=float)
    return int(np.count_nonzero(np.nan_to_num(conf, nan=0.0) >= KP_CONF_VALID))


def is_candidate(entry: dict) -> bool:
    return entry.get("top1") is not None and valid_corner_count(entry) >= MIN_VALID_CORNERS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text())
    entries = cache["entries"]

    conditions = ("daytime", "nighttime")
    by_condition = {
        condition: [e for e in entries if e["paper_condition"] == condition]
        for condition in conditions
    }

    candidates = {c: [e for e in by_condition[c] if is_candidate(e)] for c in conditions}
    pool_total = {c: len(by_condition[c]) for c in conditions}
    detected = {c: sum(e["top1"] is not None for e in by_condition[c]) for c in conditions}

    print("condition   pool  detected  candidate(>=6 valid corners)")
    print("─" * 58)
    for condition in conditions:
        print(f"{condition:11} {pool_total[condition]:<5} {detected[condition]:<9} "
              f"{len(candidates[condition])}")

    minimum_required = {
        condition: max(ABSOLUTE_MINIMUM,
                       int(np.ceil(RETENTION_FRACTION * len(candidates[condition]))))
        for condition in conditions
    }
    print(f"\nminimum required accepted: {minimum_required}")

    sweep = []
    print("\ntau     day  night   day_ok night_ok  verdict")
    print("─" * 52)
    for tau in CANDIDATE_TAUS:
        accepted = {
            condition: sum(e["top1"]["box_conf"] >= tau for e in candidates[condition])
            for condition in conditions
        }
        ok = {c: accepted[c] >= minimum_required[c] for c in conditions}
        passes = all(ok.values())
        sweep.append({
            "tau": tau,
            "accepted": accepted,
            "meets_minimum": ok,
            "passes": passes,
        })
        print(f"{tau:.2f}  {accepted['daytime']:5} {accepted['nighttime']:6}   "
              f"{str(ok['daytime']):6} {str(ok['nighttime']):8}  "
              f"{'PASS' if passes else 'reject'}")

    passing = [row["tau"] for row in sweep if row["passes"]]
    retention_fail = not passing
    tau_box = max(passing) if passing else FALLBACK_TAU
    print(f"\nTAU_BOX = {tau_box:.2f}   CONF_RETENTION_FAIL = {retention_fail}")

    # 진단: confidence bin 별 분포.  선택에는 쓰지 않는다.
    bins = {}
    for condition in conditions:
        values = np.array([e["top1"]["box_conf"] for e in candidates[condition]])
        bins[condition] = {
            f"[{low:.2f},{high:.2f})": int(np.count_nonzero((values >= low) & (values < high)))
            for low, high in CONF_BINS
        }

    lock = {
        "schema_version": "paper_pseudolabel_filter_lock_v1",
        "purpose": (
            "GT-free confidence threshold and frozen geometry-consistency "
            "thresholds for the MAIN self-training track."
        ),
        "GT_USED_FOR_SELECTION": False,
        "gt_free_statement": (
            "TAU_BOX was selected using the unlabeled adaptation pool only. "
            "No PAPER_EVAL ground truth was read at selection time."
        ),
        "teacher_checkpoint": cache["teacher_checkpoint"],
        "teacher_sha256": cache["teacher_sha256"],
        "pool_manifest": cache["pool_manifest"],
        "pool_manifest_sha256": cache["pool_manifest_sha256"],
        "teacher_cache": str(cache_path.relative_to(REPO_ROOT)),
        "teacher_cache_sha256": sha256_file(cache_path),
        "score": {
            "field": "result.boxes.conf",
            "paper_name": "YOLO detection confidence",
            "forbidden_names": ["calibrated probability", "pallet probability"],
            "composite_scores_forbidden": True,
        },
        "keypoint_validity": {
            "kp_conf_threshold": KP_CONF_VALID,
            "min_valid_corners": MIN_VALID_CORNERS,
            "corner_count": N_CORNERS,
            "centroid_excluded_from_denominator": True,
            "source": (
                "existing repo convention: challenge/scripts/infer/infer_video_yolo.py, "
                "challenge/scripts/evaluate/eval_ab_crop.py and "
                "challenge/pallet_jetson_deploy/infer_fps.py all default --kp_conf 0.5"
            ),
        },
        "selection_rule": {
            "candidates": CANDIDATE_TAUS,
            "denominator": "top1 detection exists AND valid corners >= 6 of 8",
            "rule": (
                "highest candidate tau where BOTH daytime and nighttime keep "
                "accepted >= max(100, 20% of that condition's candidate pool)"
            ),
            "absolute_minimum": ABSOLUTE_MINIMUM,
            "retention_fraction": RETENTION_FRACTION,
            "minimum_required": minimum_required,
        },
        "pool_counts": {
            "total": pool_total,
            "detected": detected,
            "candidate": {c: len(candidates[c]) for c in conditions},
        },
        "sweep": sweep,
        "TAU_BOX": tau_box,
        "CONF_RETENTION_FAIL": retention_fail,
        "geometry_thresholds": {
            "tau_reproj": 0.05,
            "tau_remove": 0.05,
            "tau_flip": 0.05,
            "normalization": "projected cuboid diagonal (dimensionless)",
            "source": (
                "scripts/data_prep/canonical_filters.py canonical defaults "
                "(filter_A tau_A, filter_C tau_C).  No new absolute pixel "
                "threshold is introduced for YOLO 640."
            ),
        },
        "confidence_bin_diagnostic": bins,
        "frozen_before_results": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
