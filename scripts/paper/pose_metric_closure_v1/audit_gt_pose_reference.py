"""GT reference 자체의 품질을 모델 결과와 결합하기 **전에** 확정한다.

    python3 scripts/paper/pose_metric_closure_v1/audit_gt_pose_reference.py

출력: GT_POSE_REFERENCE_AUDIT.json
      GT_POSE_REFERENCE_AUDIT.md

임계값을 새로 만들지 않는다.  `GT_AXIS_RESOLUTION_LOCK` 이 채택한 기존 어노테이션
품질 바(5.0 px, `scripts/annotate/_audit_annotate.py:200`)만 쓴다.  그 바를 넘는
프레임은 **제외하지 않고** review list 에 올린다 — 모델 결과를 보기 전에.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
GT = OUT_DIR / "GEOMETRY_RESOLVED_POSE_GT.json"
LOCK = OUT_DIR / "GT_AXIS_RESOLUTION_LOCK.json"
AUDIT_JSON = OUT_DIR / "GT_POSE_REFERENCE_AUDIT.json"
AUDIT_MD = REPO_ROOT / "_docs/paper/pose_metric_closure_v1/GT_POSE_REFERENCE_AUDIT.md"


def stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def main() -> int:
    payload = json.loads(GT.read_text())
    lock = json.loads(LOCK.read_text())
    quality_px = float(lock["quality_condition"]["threshold_px"])
    frames = payload["frames"]

    groups = {"ALL": list(frames.values())}
    for entry in frames.values():
        groups.setdefault(entry["object_type"], []).append(entry)
        groups.setdefault(f"domain:{entry['paper_domain']}", []).append(entry)

    summary = {}
    for name, rows in groups.items():
        chosen = np.array([r["chosen_reproj_px"] for r in rows])
        alternative = np.array([r["alternative_reproj_px"] for r in rows])
        margin = np.array([r["resolution_margin_px"] for r in rows])
        ratio = np.array([r["resolution_ratio"] for r in rows])
        summary[name] = {
            "n": len(rows),
            "chosen_reproj_px": stats(chosen),
            "alternative_reproj_px": stats(alternative),
            "resolution_margin_px": stats(margin),
            "resolution_ratio": stats(ratio),
            "long_axis": {
                "CF_WIDTH": sum(1 for r in rows if r["physical_long_axis"] == "CF_WIDTH"),
                "CF_DEPTH": sum(1 for r in rows if r["physical_long_axis"] == "CF_DEPTH"),
            },
        }

    # 기존 품질 바를 넘는 프레임 — 제외가 아니라 검토 목록
    over_bar = sorted(
        [r for r in frames.values() if r["chosen_reproj_px"] >= quality_px],
        key=lambda r: -r["chosen_reproj_px"])
    stored_over_bar = [r for r in frames.values()
                       if r["meets_annotation_quality_bar"] is False]
    # 가장 덜 분리된 프레임 (제외 기준 아님, 보고용)
    least_separated = sorted(frames.values(),
                             key=lambda r: r["resolution_margin_px"])[:10]

    report = {
        "schema_version": "gt_pose_reference_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_source": str(GT.relative_to(REPO_ROOT)),
        "rule_sha256": payload["rule_sha256"],
        "model_predictions_used": False,
        "total": payload["total"],
        "resolved": payload["resolved"],
        "unresolved": payload["unresolved"],
        "unresolved_detail": payload["unresolved_detail"],
        "quality_bar_px": quality_px,
        "quality_bar_source": lock["quality_condition"]["source"],
        "quality_bar_invented_here": False,
        "groups": summary,
        "refit_over_quality_bar": {
            "n": len(over_bar),
            "policy": "listed for review, NOT excluded — the lock forbids exclusion on any criterion other than solver failure",
            "frames": [{"frame_id": r["frame_id"], "session_id": r["session_id"],
                        "object_type": r["object_type"],
                        "chosen_reproj_px": r["chosen_reproj_px"],
                        "alternative_reproj_px": r["alternative_reproj_px"],
                        "stored_annotation_reproj_px": r["stored_annotation_reproj_px"],
                        "elevation_deg": r["elevation_deg"]} for r in over_bar],
        },
        "stored_annotation_over_quality_bar": {
            "n": len(stored_over_bar),
            "frames": [r["frame_id"] for r in stored_over_bar],
        },
        "least_separated_frames": [
            {"frame_id": r["frame_id"], "session_id": r["session_id"],
             "margin_px": r["resolution_margin_px"], "ratio": r["resolution_ratio"],
             "chosen_reproj_px": r["chosen_reproj_px"],
             "elevation_deg": r["elevation_deg"]} for r in least_separated],
        "gate": {
            "all_frames_resolved": payload["unresolved"] == 0,
            "counts_match": payload["resolved"] == 319,
            "ready_for_pose_evaluation": (payload["unresolved"] == 0
                                          and payload["resolved"] == 319),
        },
    }
    AUDIT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    all_stats = summary["ALL"]
    lines = [
        "# GT pose reference audit",
        "",
        "Run **before** any model pose result exists. The question here is only whether",
        "the ground-truth reference is trustworthy on its own terms.",
        "",
        "```text",
        f"total        {report['total']}",
        f"resolved     {report['resolved']}",
        f"unresolved   {report['unresolved']}",
        "```",
        "",
        "## Per population",
        "",
        "```text",
        f"{'population':32}{'n':>5}{'chosen med':>12}{'p90':>8}{'p95':>8}{'max':>8}"
        f"{'alt med':>10}{'margin med':>12}",
        "─" * 95,
    ]
    order = ["ALL"] + sorted(k for k in summary if k != "ALL")
    for name in order:
        s = summary[name]
        lines.append(
            f"{name:32}{s['n']:5d}{s['chosen_reproj_px']['median']:12.2f}"
            f"{s['chosen_reproj_px']['p90']:8.2f}{s['chosen_reproj_px']['p95']:8.2f}"
            f"{s['chosen_reproj_px']['max']:8.2f}"
            f"{s['alternative_reproj_px']['median']:10.2f}"
            f"{s['resolution_margin_px']['median']:12.2f}")
    lines += [
        "```",
        "",
        "`chosen` is the reprojection residual of the selected hypothesis; `alt` is the",
        "rejected one. The gap between them is what makes the axis identifiable.",
        "",
        "## Quality bar",
        "",
        "```text",
        f"bar            {quality_px} px",
        f"source         {report['quality_bar_source']}",
        "invented here  no",
        "```",
        "",
        f"Frames whose refit residual reaches the bar: **{len(over_bar)}**.",
        "They are listed for review and are **not** excluded — the resolution lock",
        "permits exclusion only on solver failure.",
        "",
    ]
    if over_bar:
        lines += ["```text",
                  f"{'frame':34}{'chosen':>9}{'alt':>9}{'stored':>9}{'elev':>8}",
                  "─" * 69]
        for r in over_bar[:20]:
            stored = r["stored_annotation_reproj_px"]
            lines.append(f"{r['frame_id'][:34]:34}{r['chosen_reproj_px']:9.2f}"
                         f"{r['alternative_reproj_px']:9.2f}"
                         f"{(stored if isinstance(stored,(int,float)) else float('nan')):9.2f}"
                         f"{r['elevation_deg']:8.1f}")
        lines += ["```", ""]
    lines += [
        "## Least separated frames",
        "",
        "Smallest margin between the two hypotheses. Reported so a reader can see how",
        "close the closest calls were; no margin threshold selects frames in or out.",
        "",
        "```text",
        f"{'frame':34}{'margin':>9}{'ratio':>8}{'chosen':>9}{'elev':>8}",
        "─" * 68,
    ]
    for r in report["least_separated_frames"]:
        lines.append(f"{r['frame_id'][:34]:34}{r['margin_px']:9.2f}{r['ratio']:8.1f}"
                     f"{r['chosen_reproj_px']:9.2f}{r['elevation_deg']:8.1f}")
    lines += [
        "```",
        "",
        "## Verdict",
        "",
        "```text",
        f"all frames resolved        {report['gate']['all_frames_resolved']}",
        f"counts match 319           {report['gate']['counts_match']}",
        f"ready for pose evaluation  {report['gate']['ready_for_pose_evaluation']}",
        "```",
        "",
        "No model prediction was read while producing this reference or this audit.",
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n")

    print(f"resolved {report['resolved']}/{report['total']}   unresolved {report['unresolved']}")
    print(f"chosen reproj  median {all_stats['chosen_reproj_px']['median']:.2f}  "
          f"p90 {all_stats['chosen_reproj_px']['p90']:.2f}  "
          f"p95 {all_stats['chosen_reproj_px']['p95']:.2f}  "
          f"max {all_stats['chosen_reproj_px']['max']:.2f}")
    print(f"alternative    median {all_stats['alternative_reproj_px']['median']:.2f}")
    print(f"margin         median {all_stats['resolution_margin_px']['median']:.2f}  "
          f"min {all_stats['resolution_margin_px']['min']:.2f}")
    print(f"over {quality_px} px bar (listed, not excluded): {len(over_bar)}")
    print(f"ready_for_pose_evaluation = {report['gate']['ready_for_pose_evaluation']}")
    return 0 if report["gate"]["ready_for_pose_evaluation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
