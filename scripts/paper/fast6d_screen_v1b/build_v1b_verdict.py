"""사전등록 gate 를 기계적으로 적용해 V1B 판정을 낸다.  결과를 보고 고치지 않는다.

    python3 scripts/paper/fast6d_screen_v1b/build_v1b_verdict.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATES = {"PROMOTABLE_SIGNAL_FOUND", "NO_PROMOTABLE_SIGNAL", "PARTIALLY_BLOCKED"}


def summarize(rows, arm):
    blocks = [r[arm] for r in rows if arm in r]
    if not blocks:
        return {"n": 0}
    arr = lambda k: np.array([b[k] for b in blocks], float)
    values = arr("add_sym_m")
    diameter = float(np.median(arr("diameter_m")))
    thresholds = np.linspace(0.0, 0.1 * diameter, 1001)
    accuracy = (values[None, :] <= thresholds[:, None]).mean(axis=1)
    return {"n": len(blocks),
            "rotation_median_deg": float(np.median(arr("rotation_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_deg"))),
            "translation_median_cm": float(np.median(arr("translation_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": float(np.trapz(accuracy, thresholds) / (0.1 * diameter))}


def gate_c1(base, arm, coverage_base, coverage_arm):
    d_iou = arm["iou3d_median"] - base["iou3d_median"]
    d_auc = arm["add_sym_auc"] - base["add_sym_auc"]
    checks = {
        "primary_delta_iou3d": d_iou, "primary_delta_add_sym_auc": d_auc,
        "primary_met": bool(d_iou >= 0.020 or d_auc >= 0.020),
        "other_primary_at_or_above_baseline": bool(
            (d_auc >= 0.0) if d_iou >= 0.020 else (d_iou >= 0.0)),
        "coverage_drop": coverage_base - coverage_arm,
        "coverage_ok": bool(coverage_base - coverage_arm <= 0.01),
        "rotation_not_worse_5pct": bool(
            arm["rotation_median_deg"] <= base["rotation_median_deg"] * 1.05),
        "translation_not_worse_5pct": bool(
            arm["translation_median_cm"] <= base["translation_median_cm"] * 1.05),
    }
    checks["VERDICT"] = "GO" if all(
        checks[k] for k in ("primary_met", "other_primary_at_or_above_baseline",
                            "coverage_ok", "rotation_not_worse_5pct",
                            "translation_not_worse_5pct")) else "STOP"
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    screen, report_dir = out_dir / "screen", out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads((out_dir / "FAST_6D_SCREEN_V1B_LOCK.json").read_text())
    bbox = json.loads((screen / "C1_OBSERVABLE_BBOX.json").read_text())
    bbox_rows = json.loads((screen / "C_PER_FRAME.json").read_text())["frames"]
    boot = json.loads((screen / "PAIRED_BOOTSTRAP.json").read_text())
    gate_geometry = json.loads((out_dir / "audit" / "GEOMETRY_CONVENTION_GATE.json").read_text())
    unit = json.loads((out_dir / "audit" / "LINE_FUSION_UNIT_TESTS.json").read_text())
    total = bbox["population_total"]

    # ---- C1
    c0 = bbox["MAIN_paired_on_resolved"]["C0"]
    c1 = bbox["MAIN_paired_on_resolved"]["C1"]
    c1_gate = gate_c1(c0, c1, c0["n"] / total, c1["n"] / total)

    # ---- line arms
    line = {}
    for s in (1, 2):
        block = json.loads((screen / f"LINE_ARMS_seed{s}.json").read_text())
        rows = json.loads((screen / f"LINE_PER_FRAME_seed{s}.json").read_text())["frames"]
        line[s] = {"summary": block, "rows": rows}

    l3 = {}
    for s in (1, 2):
        base, arm = line[s]["summary"]["ALL"]["L0"], line[s]["summary"]["ALL"]["L3"]
        l3[s] = {
            "delta_iou3d": arm["iou3d_median"] - base["iou3d_median"],
            "delta_add_sym_auc": arm["add_sym_auc"] - base["add_sym_auc"],
            "coverage_drop": base["pose_coverage"] - arm["pose_coverage"],
            "rotation_ratio": arm["rotation_median_deg"] / base["rotation_median_deg"],
            "translation_ratio": arm["translation_median_cm"] / base["translation_median_cm"],
        }
    condition_A = all(l3[s]["delta_iou3d"] >= 0 and l3[s]["delta_add_sym_auc"] >= 0
                      for s in (1, 2))
    median_iou = float(np.median([l3[s]["delta_iou3d"] for s in (1, 2)]))
    median_auc = float(np.median([l3[s]["delta_add_sym_auc"] for s in (1, 2)]))
    condition_B = bool(median_iou >= 0.020 or median_auc >= 0.020)
    condition_C = all(l3[s]["coverage_drop"] <= 0.01 for s in (1, 2))
    condition_D = all(l3[s]["rotation_ratio"] <= 1.05 and l3[s]["translation_ratio"] <= 1.05
                      for s in (1, 2))
    l3_verdict = "GO" if (condition_A and condition_B and condition_C and condition_D) else "STOP"

    both_seeds_positive_direction = all(
        l3[s]["delta_iou3d"] > 0 and l3[s]["delta_add_sym_auc"] > 0 for s in (1, 2))

    state = ("PROMOTABLE_SIGNAL_FOUND" if (c1_gate["VERDICT"] == "GO" or l3_verdict == "GO")
             else "NO_PROMOTABLE_SIGNAL")
    if gate_geometry["line_population"] == "NONE" or unit["LINE_FUSION_IMPLEMENTATION_GATE"] == "FAIL":
        state = "PARTIALLY_BLOCKED"
    assert state in STATES, state

    promoted = None
    if l3_verdict == "GO":
        promoted = "YOLO_POINT_PLUS_SPLITLATE_LINE_F3"
    elif c1_gate["VERDICT"] == "GO":
        promoted = "YOLO_POINT_PLUS_OBSERVABLE_BBOX_TRANSLATION"

    result = {
        "schema_version": "fast_6d_screen_v1b_result_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "POST_STOP_EXPLORATORY_CORRECTION",
        "claim_boundary": "DEVELOPMENT RESULT, NOT INDEPENDENTLY CONFIRMED",
        "population": "PAPER_EVAL positive 319, reused development data",
        "lock_commit": "6cb2b00",
        "new_training": 0, "new_checkpoint": 0, "depth": 0, "parameter_sweep": 0,
        "gates": {
            "GEOMETRY_CONVENTION_GATE": gate_geometry["object_status"],
            "line_population": gate_geometry["line_population"],
            "LINE_FUSION_IMPLEMENTATION_GATE": unit["LINE_FUSION_IMPLEMENTATION_GATE"],
        },
        "C1": {"summary_C0": c0, "summary_C1": c1,
               "unresolved_observable_box": bbox["C1_UNRESOLVED_OBSERVABLE_BOX"]["n"],
               "exceptions": bbox["exceptions"]["count"],
               "gate": c1_gate},
        "L3_two_seed_policy": {
            "per_seed": l3,
            "condition_A_both_seeds_non_negative": condition_A,
            "condition_B_median_delta": {"median_delta_iou3d": median_iou,
                                         "median_delta_add_sym_auc": median_auc,
                                         "met": condition_B},
            "condition_C_coverage": condition_C,
            "condition_D_no_5pct_degradation": condition_D,
            "VERDICT": l3_verdict,
            "both_seeds_positive_direction": both_seeds_positive_direction,
        },
        "arms_ALL": {
            "C0": c0, "C1": c1,
            "seed1": line[1]["summary"]["ALL"], "seed2": line[2]["summary"]["ALL"]},
        "by_material": {
            "C1_track": {material: {arm: summarize(
                [r for r in bbox_rows if r["object_type"] == material], arm)
                for arm in ("C0", "C1")}
                for material in sorted({r["object_type"] for r in bbox_rows})},
            "line_track": {f"seed{s}": line[s]["summary"]["by_material"] for s in (1, 2)}},
        "by_lighting_existing_paper_domain_field": {
            "C1_track": {domain: {arm: summarize(
                [r for r in bbox_rows if r["paper_domain"] == domain], arm)
                for arm in ("C0", "C1")}
                for domain in sorted({r["paper_domain"] for r in bbox_rows})},
            "line_track": {f"seed{s}": line[s]["summary"]["by_lighting_existing_paper_domain_field"]
                           for s in (1, 2)},
            "note": "the manifest labels only 120 of 319 frames; the remaining 199 are "
                    "reported as 'none' rather than guessed from session names"},
        "bootstrap": boot["contrasts"],
        "runtime_benchmark": {
            "triggered": bool(l3_verdict == "GO" or both_seeds_positive_direction),
            "reason": "the lock runs it only when L3 passes or both seeds point positive"},
        "CORRECTED_FAST6D": state,
        "PROMOTED_METHOD_CANDIDATE": promoted,
        "next_branch": ("candidate_closure" if promoted else "PAPER_FRAMING_CLOSURE_V1"),
        "next_action": "USER_REVIEW_OVERNIGHT_RESULT",
    }
    (report_dir / "FAST_6D_SCREEN_V1B_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"C1  = {c1_gate['VERDICT']}")
    print(f"L3  = {l3_verdict}  (seed1 dIoU {l3[1]['delta_iou3d']:+.4f} "
          f"dAUC {l3[1]['delta_add_sym_auc']:+.4f} | seed2 dIoU {l3[2]['delta_iou3d']:+.4f} "
          f"dAUC {l3[2]['delta_add_sym_auc']:+.4f})")
    print(f"CORRECTED_FAST6D = {state}")
    print(f"PROMOTED_METHOD_CANDIDATE = {promoted}")
    print(f"runtime triggered = {result['runtime_benchmark']['triggered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
