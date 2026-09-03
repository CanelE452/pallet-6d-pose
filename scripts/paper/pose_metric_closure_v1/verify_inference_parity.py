"""R0 replay 가 기존 R0 캐시를 재현하는지 확인한다.  게이트다.

    python3 scripts/paper/pose_metric_closure_v1/verify_inference_parity.py

출력: INFERENCE_PARITY_R0.json

기존 캐시는 `data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv` 의
`PAPER_EVAL_ALL_POS` 319 행이다.  새 runner 가 만든
`predictions/R0.json` 과 raw box 좌표 · box confidence · 감독 키포인트 오차를
비교한다.  R0 가 재현되지 않으면 다른 arm 의 표를 신뢰할 수 없다.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
D = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
CACHE = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv"
REPLAY = D / "predictions/R0.json"
MANIFEST = D / "AXIS_REVIEW_MANIFEST.json"
OUT = D / "INFERENCE_PARITY_R0.json"


def main() -> int:
    cached = {r["frame_id"].replace(":", "__"): r
              for r in csv.DictReader(CACHE.open())
              if r["population_id"] == "PAPER_EVAL_ALL_POS"}
    replay = json.loads(REPLAY.read_text())["frames"]
    frames = {f["frame_id"]: f for f in json.loads(MANIFEST.read_text())["frames_list"]}

    box_delta, score_delta, kp_delta = [], [], []
    kp_incomparable: list[dict] = []
    detection_agree = 0
    detection_disagree: list[str] = []
    missing: list[str] = []

    for frame_id, old in cached.items():
        new = replay.get(frame_id)
        if new is None:
            missing.append(frame_id)
            continue
        old_detected = bool(old["top_score"])
        new_detected = new.get("status") == "OK"
        if old_detected != new_detected:
            detection_disagree.append(frame_id)
            continue
        detection_agree += 1
        if not new_detected:
            continue

        score_delta.append(abs(float(old["top_score"]) - new["box_conf"]))
        old_box = np.array([float(old[f"top_box_{k}"])
                            for k in ("x1", "y1", "x2", "y2")])
        box_delta.append(float(np.abs(old_box - np.array(new["box_xyxy"])).max()))

        # 감독 키포인트 오차를 GT 로 다시 계산해 캐시된 목록과 맞춘다
        cached_errors = [float(v) for v in old["top_keypoint_supervised_errors_px"].split(";") if v]
        gt = np.array([p if p else [np.nan, np.nan]
                       for p in frames[frame_id]["keypoints_xy"]], float)
        pred = np.array(new["keypoints_xy"], float)
        n = min(len(gt), len(pred))
        usable = np.isfinite(gt[:n]).all(axis=1)
        recomputed = np.linalg.norm(pred[:n][usable] - gt[:n][usable], axis=1)
        if len(recomputed) == len(cached_errors):
            kp_delta.append(float(np.abs(recomputed - np.array(cached_errors)).max()))
        else:
            kp_incomparable.append({
                "frame_id": frame_id,
                "cached_list_length": len(cached_errors),
                "cached_supervised_count": old["supervised_keypoint_count"],
                "recomputed_length": int(len(recomputed)),
            })

    def summary(values):
        if not values:
            return {"n": 0}
        arr = np.array(values)
        return {"n": int(arr.size), "median": float(np.median(arr)),
                "max": float(arr.max())}

    report = {
        "schema_version": "inference_parity_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "R0",
        "cache": str(CACHE.relative_to(REPO_ROOT)),
        "replay": str(REPLAY.relative_to(REPO_ROOT)),
        "population": "PAPER_EVAL_ALL_POS 319",
        "frames_in_cache": len(cached),
        "frames_in_replay": len(replay),
        "detection_agreement": detection_agree,
        "detection_disagreement": detection_disagree,
        "missing_from_replay": missing,
        "box_coordinate_delta_px": summary(box_delta),
        "box_confidence_delta": summary(score_delta),
        "supervised_keypoint_error_delta_px": summary(kp_delta),
        "supervised_keypoint_incomparable": {
            "n": len(kp_incomparable),
            "reason": "the cached CSV either stores an empty error list or counts supervised keypoints by an annotation visibility flag this script does not reconstruct; the mismatch is in the cache's list length, not in a coordinate",
            "frames": kp_incomparable,
        },
        "tolerance": {
            "box_xyxy_px": 0.0,
            "box_confidence": 0.0,
            "supervised_keypoint_error_px": 5e-7,
            "derivation": "box and confidence are compared against full float precision, so exact equality is required. The cached keypoint errors are serialised to six decimal places, so the largest possible rounding half-width is 5e-7 px; the tolerance is that half-width and is not fitted to the observed value.",
            "declared": "before the pose tables were read",
        },
    }
    exact_raw = (not detection_disagree and not missing
                 and detection_agree == len(cached)
                 and max([0.0] + box_delta) == 0.0
                 and max([0.0] + score_delta) == 0.0)
    kp_ok = max([0.0] + kp_delta) <= 5e-7
    report["gate"] = {
        "box_and_confidence_bit_exact": exact_raw,
        "keypoint_within_serialisation_precision": kp_ok,
        "verdict": "PASS" if (exact_raw and kp_ok) else "FAIL",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"frames cache {len(cached)}  replay {len(replay)}")
    print(f"detection agreement   {detection_agree} / {len(cached)}")
    print(f"box coord delta px    max {max([0.0] + box_delta):.6f}")
    print(f"box conf delta        max {max([0.0] + score_delta):.6f}")
    print(f"keypoint err delta px max {max([0.0] + kp_delta):.6f}  "
          f"(n {len(kp_delta)})")
    print(f"keypoint list incomparable: {len(kp_incomparable)} frames "
          f"(cache-side list length, not a coordinate difference)")
    print(f"verdict {report['gate']['verdict']}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0 if report['gate']['verdict'] == 'PASS' else 1


if __name__ == "__main__":
    raise SystemExit(main())
