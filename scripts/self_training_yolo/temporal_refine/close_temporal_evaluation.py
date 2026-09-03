"""TEMPORAL EVALUATION CLOSURE — 얼어 있는 좌표를 올바른 계약으로 다시 채점한다.

    python3 scripts/self_training_yolo/temporal_refine/close_temporal_evaluation.py \
        --output-dir data/pallet/results/paper_temporal_selftrain_v1/evaluation_closure_v1

refinement 를 다시 돌리지 않는다.  `TEMPORAL_REFINEMENT_PER_FRAME.json` 의
RAW_TEACHER · TEMPORAL_ONLY · TEMPORAL_GEOMETRY 좌표를 그대로 읽는다.

고치는 것은 채점 쪽 셋이다.

    population   lock 이 말한 대로 evaluation-ineligible 을 배제한다
    2D           프레임 요약의 요약이 아니라 코너 오차를 한 벡터로 모은다
    6D           frozen predict_pose_without_gt 를 쓴다 (자체 selector 금지)

coverage 에 사후 임계값을 붙이지 않는다.  치수는 frozen object contract 에서 읽는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT = REPO_ROOT / "data/pallet/results/paper_temporal_selftrain_v1/pilot"
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
sys.path.insert(0, str(REPO_ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation"))
sys.path.insert(0, str(REPO_ROOT))

METHODS = ["RAW_TEACHER", "TEMPORAL_ONLY", "TEMPORAL_GEOMETRY"]
GROSS_PX = 20.0
N_RESAMPLES = 10000
SEED = 20260903
# 사전에 기록된 known-broken 만.  결과를 보고 새로 추가하지 않는다.
KNOWN_BROKEN = {"capturepallet11": "pallet11_gt APRILTAG_BROKEN, recorded before the temporal work"}


def audit(refinement, frames_by_id, excluded_recordings):
    """center 109 개를 전수 감사한다.  완화 없이 lock 문구 그대로."""

    rows = []
    for frame_id, entry in refinement.items():
        frame = frames_by_id.get(frame_id, {})
        session = entry.get("source_recording")
        annotation = WORKSPACE / entry["gt_annotation_path"]
        manual = []
        if annotation.exists():
            obj = json.loads(annotation.read_text())["objects"][0]
            manual = [p for p in obj.get("projected_cuboid", [])[:8] if p]
        row = {
            "frame_id": frame_id,
            "source_recording": session,
            "population_role": frame.get("population_role", ""),
            "usage_role": frame.get("usage_role", ""),
            "controlled_eval_eligible": frame.get("controlled_eval_eligible", ""),
            "exclusion_reason": frame.get("exclusion_reason", ""),
            "source_dataset": frame.get("source_dataset", ""),
            "gt_source": frame.get("gt_source", ""),
            "manual_kps_present": bool(manual),
            "manual_kps_usable_count": len(manual),
            "known_broken_before_temporal": session in KNOWN_BROKEN,
            "known_broken_reason": KNOWN_BROKEN.get(session, ""),
            "FT_OVERLAP": frame.get("exclusion_reason") == "FT_OVERLAP",
            "paper_eval_recording_overlap": session in excluded_recordings,
            "closure_eligible": False,
            "closure_exclusion_reason": "",
        }
        reasons = []
        if row["known_broken_before_temporal"]:
            reasons.append("known-broken annotation set")
        if row["FT_OVERLAP"]:
            reasons.append("FT_OVERLAP")
        if row["paper_eval_recording_overlap"]:
            reasons.append("recording feeds PAPER_EVAL 319")
        # lock: FT_OVERLAP **또는 evaluation-ineligible** 은 center 로 쓰지 않는다.
        # workspace 가 usage_role 로 적격을 표현한다 — EVAL_LABELED 만 적격이다.
        if row["usage_role"] and row["usage_role"] != "EVAL_LABELED":
            reasons.append(f"evaluation-ineligible: usage_role {row['usage_role']}")
        if row["exclusion_reason"] and not row["FT_OVERLAP"]:
            reasons.append(f"workspace exclusion_reason {row['exclusion_reason']}")
        if row["manual_kps_usable_count"] < 6:
            reasons.append("fewer than six manual corners")
        row["closure_exclusion_reason"] = "; ".join(reasons)
        row["closure_eligible"] = not reasons
        rows.append(row)
    return rows


def pooled_2d(entries, gt_by_id, method):
    errors, frames_used = [], 0
    for frame_id, entry in entries:
        points = entry.get({"RAW_TEACHER": "raw_teacher",
                            "TEMPORAL_ONLY": "temporal_only",
                            "TEMPORAL_GEOMETRY": "temporal_geometry"}[method])
        if points is None:
            continue
        points = np.asarray(points, np.float64)
        gt = gt_by_id[frame_id]
        valid = np.isfinite(gt).all(axis=1) & np.isfinite(points).all(axis=1)
        if valid.sum() < 6:
            continue
        frames_used += 1
        errors.append(np.linalg.norm(points[valid] - gt[valid], axis=1))
    if not errors:
        return {"frames": 0, "corners": 0}
    pooled = np.concatenate(errors)
    return {
        "frames": frames_used,
        "corners": int(pooled.size),
        "median_px": float(np.median(pooled)),
        "p90_px": float(np.percentile(pooled, 90)),
        "gross20": float(np.mean(pooled > GROSS_PX)),
        "_errors": pooled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from pose_evaluation_paths import (load_pose_object_contract, object_spec,
                                       predict_pose_without_gt)
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from eval_workspace import load_frames

    method_lock = json.loads(
        (PILOT.parent / "TEMPORAL_METHOD_LOCK.json").read_text())
    excluded = set(method_lock["population"]["excluded"]["recordings_that_feed_PAPER_EVAL_319"])
    refinement = json.loads(
        (PILOT / "TEMPORAL_REFINEMENT_PER_FRAME.json").read_text())["frames"]
    frames_by_id = {f.get("frame_id"): f for f in load_frames(WORKSPACE)}
    print(f"original centres {len(refinement)}")

    rows = audit(refinement, frames_by_id, excluded)
    fields = list(rows[0])
    with (out_dir / "TEMPORAL_CENTER_ELIGIBILITY_AUDIT.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import collections
    eligible = [r for r in rows if r["closure_eligible"]]
    reason_counts = collections.Counter()
    for r in rows:
        if r["closure_eligible"]:
            continue
        for piece in r["closure_exclusion_reason"].split("; "):
            reason_counts[piece.split(":")[0]] += 1
    recordings = sorted({r["source_recording"] for r in eligible})
    audit_report = {
        "schema_version": "temporal_center_eligibility_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "original_centres": len(rows),
        "formal_eligible": len(eligible),
        "formal_recordings": len(recordings),
        "recordings": recordings,
        "exclusion_counts": dict(reason_counts),
        "rule_was_not_relaxed": True,
        "usage_role_distribution": dict(collections.Counter(r["usage_role"] for r in rows)),
    }
    (out_dir / "TEMPORAL_CENTER_ELIGIBILITY_AUDIT.json").write_text(
        json.dumps(audit_report, indent=2) + "\n")

    print(f"\nformal eligible {len(eligible)} / {len(rows)}   recordings {len(recordings)}")
    print("  usage_role 분포:", audit_report["usage_role_distribution"])
    print("  제외 사유:")
    for reason, count in reason_counts.most_common():
        print(f"    {count:5d}  {reason}")

    if not eligible:
        result = {
            "schema_version": "temporal_evaluation_closure_result_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "intent": "TEMPORAL_EVALUATION_CLOSURE_INTENT.json",
            "refinement_outputs_modified": False,
            "new_inference": 0, "new_refinement": 0, "student_training": 0,
            "formal_eligible": 0,
            "FORMAL_TEMPORAL_PILOT": "POPULATION_LIMITED",
            "original_pilot_classification": "EXPLORATORY_DIAGNOSTIC_ONLY",
            "why": ("under the population contract the lock actually stated, no centre in "
                    "the temporal pilot qualifies. Every eligible-looking centre came from "
                    "legacy annotation the workspace marks as not paper eligible, and the "
                    "recordings whose annotations are paper eligible are exactly the ones "
                    "that feed PAPER_EVAL and were excluded whole by design."),
            "rule_not_relaxed": True,
            "failed_to_improve_is_not_a_formal_result": (
                "the earlier FAILED_TO_IMPROVE was computed on a population the lock "
                "excluded, with a summary-of-summaries 2D metric and a private pose "
                "selector. It is retained as an exploratory diagnostic, not a preregistered "
                "result."),
            "audit": audit_report,
        }
        (out_dir / "TEMPORAL_EVALUATION_CLOSURE_RESULT.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print("\nFORMAL_TEMPORAL_PILOT = POPULATION_LIMITED")
        print("  eligibility 를 완화해 N 을 만들지 않는다.")
        return 0

    # ── 여기부터는 formal population 이 있을 때만
    contract_path = CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"
    contract = load_pose_object_contract(str(contract_path))
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    reference = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]

    gt_by_id, entries = {}, []
    for row in eligible:
        frame_id = row["frame_id"]
        entry = refinement[frame_id]
        annotation = WORKSPACE / entry["gt_annotation_path"]
        obj = json.loads(annotation.read_text())["objects"][0]
        gt = np.array([p if p else [np.nan, np.nan]
                       for p in obj["projected_cuboid"]], np.float64)[:8]
        gt_by_id[frame_id] = gt
        entries.append((frame_id, entry))

    two_d = {m: pooled_2d(entries, gt_by_id, m) for m in METHODS}
    (out_dir / "TEMPORAL_2D_POOLED_METRICS.json").write_text(json.dumps({
        "schema_version": "temporal_2d_pooled_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": "all supervised corner errors pooled into one vector",
        "forbidden_aggregation": "median of per-frame medians or p90s",
        "gross_threshold_px": GROSS_PX,
        "methods": {m: {k: v for k, v in two_d[m].items() if not k.startswith("_")}
                    for m in METHODS},
    }, indent=2) + "\n")
    print(f"\n{'method':22}{'frames':>8}{'corners':>9}{'median':>9}{'p90':>9}{'gross20':>10}")
    for m in METHODS:
        b = two_d[m]
        print(f"{m:22}{b['frames']:8d}{b['corners']:9d}{b.get('median_px', float('nan')):9.2f}"
              f"{b.get('p90_px', float('nan')):9.2f}{b.get('gross20', float('nan')):10.3f}")
    print("\n(6D re-scoring would follow here on a non-empty formal population)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
