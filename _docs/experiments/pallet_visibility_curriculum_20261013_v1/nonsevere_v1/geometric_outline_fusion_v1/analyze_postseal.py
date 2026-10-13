"""CAL-only descriptive diagnostics after the five-arm inference seal.

No fitting, parameter selection, model changes, or candidate selection occur here.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, float)
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "P90": float(np.quantile(values, .9)) if len(values) else None,
    }


def main():
    destination = HERE / "POSTSEAL_DIAGNOSTICS.json"
    assert not destination.exists()
    seal = json.loads((HERE / "PREDICTION_SEAL.json").read_text())
    assert seal["before_scoring"]
    assert sha(HERE / "PREDICTIONS.jsonl.gz") == seal["sha256"]
    rows = [json.loads(line) for line in gzip.open(HERE / "SCORED_ROWS.jsonl.gz", "rt")]
    truth_path = HERE.parent / "OOF_pose_v1" / "ISOLATED_CAL_SCORING.json"
    truth = json.loads(truth_path.read_text())["rows"]
    assert len(rows) == 230 and len(truth) == 46
    lookup = {(row["id"], row["arm"]): row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        assert row["id"] in truth
        groups[row["arm"]].append(row)

    seed_report = {}
    for arm in ("RAW_GEOMETRY_INFO_STAGED", "FUSED_SIGMA2_GEOMETRY_INFO_STAGED"):
        first = Counter()
        human = Counter()
        seed_error = []
        seed_direct_error = []
        seed_condition = []
        protected_global_improvement = []
        per_frame = []
        for row in groups[arm]:
            stages = row["solution"]["stages"]
            initial, stage = stages[:2]
            flags = initial["initial_geometric_outline"]
            seed = stage["used_ids"]
            first[stage["reason"]] += 1
            metadata = truth[row["id"]]["metadata"]
            mapping = metadata["native_to_reference_corner_ids"]
            gt = np.asarray(truth[row["id"]]["target"]["gt"], float)[:8][mapping]
            q = np.asarray(row["observed_points"], float)[:8]
            errors = np.linalg.norm(gt - q, axis=1)
            states = metadata["human_states_native"]
            for key in seed:
                human[states[key]] += 1
                seed_error.append(float(errors[key]))
                if states[key] == "DIRECT_VISIBLE":
                    seed_direct_error.append(float(errors[key]))
            geometry = stage.get("geometry", {})
            condition = geometry.get("jacobian_condition")
            if condition is not None:
                seed_condition.append(condition)
            for candidate in stages[1:]:
                if candidate["reason"] == "previous_visible_support_worsened":
                    before = candidate.get("visible_score_before")
                    after = candidate.get("visible_score_after")
                    if before is not None and after is not None and after < before:
                        protected_global_improvement.append({
                            "id": row["id"], "stage": candidate["name"],
                            "added_ids": candidate["added_ids"],
                            "all_visible_cost_change": after - before,
                            "protected_subset_cost_change": candidate["protected_score_after"] - candidate["protected_score_before"],
                        })
            per_frame.append({
                "id": row["id"],
                "initial_geometric_outline_ids": [key for key, value in enumerate(flags) if value],
                "outline_admitted_visible_ids": [key for key in row["solution"]["visible_ids"] if flags[key]],
                "first_fit_ids": seed,
                "first_fit_human_states": [states[key] for key in seed],
                "first_fit_direct_corner_errors_px": {str(key): float(errors[key]) for key in seed if states[key] == "DIRECT_VISIBLE"},
                "first_fit_reason": stage["reason"],
                "first_fit_information": initial["geometric_outline_seed"],
                "interior_supplement_in_first_fit": any(not flags[key] for key in seed),
                "final_used_ids": row["solution"]["used_ids"],
            })
        seed_report[arm] = {
            "first_fit_reasons": dict(first),
            "human_state_count_in_first_fit": dict(human),
            "DIRECT_VISIBLE_seed_error_px": distribution(seed_direct_error),
            "DIRECT_VISIBLE_seed_error_gt8px": sum(value > 8 for value in seed_direct_error),
            "seed_condition": distribution(seed_condition),
            "outline_four_seed_available_frames": sum("minimum_information_eigenvalue" in row["solution"]["stages"][0]["geometric_outline_seed"] for row in groups[arm]),
            "interior_supplement_in_first_fit_frames": sum(row["interior_supplement_in_first_fit"] for row in per_frame),
            "rejected_for_protected_subset_despite_improved_all_visible_cost": protected_global_improvement,
            "per_frame": per_frame,
        }

    contrasts = {}
    for a, b in (
        ("RAW_GEOMETRY_INFO_STAGED", "RAW_VISIBLE_TOGETHER"),
        ("FUSED_SIGMA2_GEOMETRY_INFO_STAGED", "FUSED_SIGMA2_VISIBLE_TOGETHER"),
        ("FUSED_SIGMA2_VISIBLE_TOGETHER", "RAW_VISIBLE_TOGETHER"),
        ("FUSED_SIGMA2_GEOMETRY_INFO_STAGED", "FUSED_SIGMA2_LEGACY_STAGED"),
    ):
        cases = []
        for row in groups[a]:
            other = lookup[row["id"], b]
            pose_difference = max(
                float(np.max(np.abs(np.asarray(row["solution"]["R"]) - np.asarray(other["solution"]["R"])))),
                float(np.max(np.abs(np.asarray(row["solution"]["t"]) - np.asarray(other["solution"]["t"])))),
            )
            if pose_difference < 1e-8 and row["solution"]["used_ids"] == other["solution"]["used_ids"]:
                continue
            delta = {key: float(row["metric"][key]) - float(other["metric"][key]) for key in ("translation_cm", "rotation_deg", "ADDsym_cm", "success_5cm_5deg")}
            cases.append({
                "id": row["id"], "session": row["session"], "delta": delta,
                "a_used_ids": row["solution"]["used_ids"], "b_used_ids": other["solution"]["used_ids"],
                "a_stage_rejections": [stage["reason"] for stage in row["solution"]["stages"][1:] if not stage["accepted"]],
                "b_stage_rejections": [stage["reason"] for stage in other["solution"]["stages"][1:] if not stage["accepted"]],
            })
        contrasts[a + "-" + b] = {
            "different_cases": len(cases),
            "different_final_used_ids": sum(case["a_used_ids"] != case["b_used_ids"] for case in cases),
            "both_translation_rotation_improved": sum(case["delta"]["translation_cm"] < -1e-8 and case["delta"]["rotation_deg"] < -1e-8 for case in cases),
            "both_translation_rotation_worsened": sum(case["delta"]["translation_cm"] > 1e-8 and case["delta"]["rotation_deg"] > 1e-8 for case in cases),
            "success_recoveries": [case["id"] for case in cases if case["delta"]["success_5cm_5deg"] > 0],
            "success_losses": [case["id"] for case in cases if case["delta"]["success_5cm_5deg"] < 0],
            "cases": cases,
        }

    report = {
        "schema": "geometric_outline_fusion_postseal_diagnostics_v1",
        "scored_predictions_sha256": sha(HERE / "SCORED_ROWS.jsonl.gz"),
        "isolated_CAL_scoring_sha256": sha(truth_path),
        "CAL_only": True, "severe_rows": 0, "TEST_rows_read": 0, "FINAL_rows_read": 0,
        "parameter_or_model_retuning": 0,
        "seed_diagnostics": seed_report,
        "contrasts": contrasts,
        "interpretation": [
            "Expected geometric silhouette vertices are a difficulty prior; they are not independently observed physical corners.",
            "An outline seed can pass projection-Jacobian rank/condition gates and still disagree with held-out visible evidence.",
            "All observation states, weights, initial family, H, native order, and center were held fixed; this study does not solve initial family ambiguity.",
            "SAM line covariance measures fitted-line precision, not calibrated semantic ownership or virtual-corner bias.",
            "The translation benefit is present in both staged and together fusion controls; the ordering itself has no matched benefit here.",
            "Fresh FULL-CAL visibility confidence is in-sample exploratory, and four session bootstrap limits precision of inference.",
        ],
    }
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for arm, value in seed_report.items():
        print(arm, value["first_fit_reasons"], "direct_seed_error", value["DIRECT_VISIBLE_seed_error_px"], "protected_improves_all", len(value["rejected_for_protected_subset_despite_improved_all_visible_cost"]))
    for name, value in contrasts.items():
        print(name, {key: val for key, val in value.items() if key != "cases"})


if __name__ == "__main__":
    main()
