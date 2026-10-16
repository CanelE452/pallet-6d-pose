"""Read sealed results and build the final diagnosis summary; no fitting or tuning."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def read(relative):
    return json.loads((HERE / relative).read_text())


def binding(relative):
    path = HERE / relative
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    benchmark = read("comparator/BENCHMARK.json")
    diagnostic = read("counterfactual/POSTSEAL_SCORING.json")
    analysis = read("counterfactual/POSTSEAL_DIAGNOSTIC_ANALYSIS.json")
    audit = read("failure_audit/FAILURE_AUDIT.json")
    uncertainty = read("uncertainty_diagnostic/UNCERTAINTY_DIAGNOSTIC.json")
    selected_methods = ["ORIGINAL_N3_PNP", "CURRENT_BI_ONLY",
                        "CURRENT_JOINT_STATEWISE_V3", "OLD_RGB_SINGLE_FAMILY",
                        "BOUNDARY_PROGRESSIVE", "COUNT_FUSED"]
    fair = {}
    for name in selected_methods:
        values = benchmark["summary"][name]["TEST"]
        assert values["frames"] == 49
        fair[name] = {"label": benchmark["method_labels"][name],
                      "translation_cm": values["metrics"]["translation_cm"]["mean"],
                      "rotation_deg": values["metrics"]["rotation_deg"]["mean"],
                      "ADDsym_cm": values["metrics"]["ADDsym_cm"]["mean"],
                      "success_count": values["success_count"], "frames": 49}
    interventions = {}
    for name, partitions in diagnostic["summary"].items():
        values = partitions["TEST"]
        assert values["frames"] == 49
        interventions[name] = {
            "translation_cm": values["metrics"]["translation_cm"]["mean"],
            "rotation_deg": values["metrics"]["rotation_deg"]["mean"],
            "success_count": values["success_count"], "frames": 49}
    sparse_failures = [r for r in audit["rows"]
                       if r["cohort"] == "PRIMARY_NONSEVERE"
                       and r["baseline_success_to_current_failure"]]
    assert len(sparse_failures) == 4
    assert all(r["BI_count"] == 3 for r in sparse_failures)
    p07 = next(r for r in sparse_failures
               if r["id"] == "eval_pallet07:1778652170735118080")
    p07_measurement = next(p for p in p07["BI_measurements"] if p["native_id"] == 7)
    threshold_checks = {}
    for split in ["CAL", "TEST"]:
        table = uncertainty["threshold_tables"][split]
        threshold_checks[split] = {
            "compatible_frames": table["frames"],
            "all_constraints_at_budget3": next(r for r in table["threshold_table"]
                if r["threshold"] == 3.0)["all_constraints_including_geometry_canvas_depth_pass"],
            "all_constraints_at_budget6": next(r for r in table["threshold_table"]
                if r["threshold"] == 6.0)["all_constraints_including_geometry_canvas_depth_pass"]}
    bound_files = ["comparator/BENCHMARK.json", "comparator/INDEPENDENT_ARITHMETIC_CHECK.json",
                   "counterfactual/POSTSEAL_SCORING.json",
                   "counterfactual/POSTSEAL_DIAGNOSTIC_ANALYSIS.json",
                   "counterfactual/COMPLETION_RECEIPT.json",
                   "counterfactual/INDEPENDENT_CONTRACT_CHECK.json",
                   "failure_audit/FAILURE_AUDIT.json",
                   "uncertainty_diagnostic/UNCERTAINTY_DIAGNOSTIC.json",
                   "backbone_inventory/DIRECT_PHYSICAL_C2_CACHE_SUMMARY.json",
                   "COUNTERFACTUAL_PLAN_V1.json", "README_KO.md", "BUILD_SUMMARY.py"]
    result = {
        "schema": "FINAL_COMMON_INPUT_METHOD_COMPARISON_AND_CAUSAL_DIAGNOSIS_V1",
        "cohort": {"CAL": 46, "TEST": 49, "severe": 0, "crop_in_primary": 0,
                   "all_failures_and_fallbacks_kept": True},
        "same_input_TEST": fair,
        "diagnostic_TEST": interventions,
        "compatible_reference_constraint_checks": threshold_checks,
        "diagnostic_reference_feasibility": analysis["reference_feasibility"],
        "primary_original_success_to_current_failure": [
            {"id": r["id"], "BI_count": r["BI_count"], "metrics": r["metrics"],
             "current_equals_BI_only": r["final_equals_BI_only_pose"]}
            for r in sparse_failures],
        "actual_case": {"id": p07["id"], "metrics": p07["metrics"],
                        "native_7_measurement": p07_measurement,
                        "figure": "failure_audit/eval_pallet07__1778652170735118080.png"},
        "conclusion": {
            "current_superior_to_best_same_input_baselines": False,
            "BI_correction_and_OT_help_under_fixed_controls": True,
            "classification_or_budget_change_alone_fixes_gap": False,
            "reference_initialization_changes_current_metrics": False,
            "ideal_geometry_positive_control_is_real_image_or_OT_advantage_proof": False,
            "selector_only_can_close_old_RGB_translation_gap_in_current_found_pool": False,
            "global_search_completeness_proven": False,
            "independent_physical_ground_truth": False,
            "fresh_holdout_or_external_SOTA_comparison": False,
            "production_configuration_changed_or_promoted": False},
        "source_results": {relative: binding(relative) for relative in bound_files}}
    (HERE / "FINAL_FINDINGS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"same_input_methods": len(fair),
                      "diagnostic_arms": len(interventions),
                      "sparse_primary_failures": len(sparse_failures),
                      "source_bindings": len(bound_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
