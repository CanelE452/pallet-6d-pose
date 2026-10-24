"""Freeze V82 saved-evaluation bindings. Metadata/hash/generated only."""
from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent
V81 = P / "query_visibility_grounded_target_20261023_v81"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert bind(V / "EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json")["sha256"] == "ab89f3885627cdd319b93fb6ead6357950a10f7fa51d751798ddd7cabba5af28"
    generated = json.loads((R / "GENERATED_EVALUATOR_REUSE_TESTS.json").read_text())
    assert generated["PASS"] and generated["code"] == bind(R / "evaluate_saved_requested_masks.py")
    criteria = json.loads((V / "SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json").read_text())
    assert criteria["before_any_V82_forward"]
    assert criteria["criteria"] == {"C_end_to_end_recall_at_least": .8, "V_end_to_end_recall_at_least": .8,
        "C_precision_known_CV_at_least": .5, "hold_rate_at_most": .1}
    policy_path = R / "EVALUATION_POLICY_BEFORE_V82_OUTPUTS.json"
    write(policy_path, {"frozen_before_actual_V82_output_values_or_GT_mask_pixels_read_for_this_stage": True,
        "inherited_V81_source_GT_and_formulas": "기존GTpack/lineage/primary/strict535를그대로사용하며GT재생성·라벨변경·radius선택없음. nativeIoU/qpixel/referenceprimary/O/invalid/CVU/P/R/U는동일V81함수를그대로호출한다.",
        "source_information_contract_KO": "SYNTH는고정pallet, YCB는sourceGTmetadata로제공한requested_category추가입력조건. V81동일입력비교또는class-free/예측category라고주장하지않음.",
        "report_context_only": "기존V81reader의포괄적GTinput없음문구는최초저장전에분리한다. 요청category의GTmetadata제공True, GTmask/reference/pose/depth/visibility/instanceID모델입력False, GT기반출력재선택False를명시. 수식/선택/분모는변경하지않는다.",
        "source_component_screen": "사전criteria의같은strict535_C76/V459에서C recall>=.8,V recall>=.8,C조건부precision>=.5,hold<=.1. U는classrecall실패및hold분모. 모든숫자를PASS/FAIL과무관하게보존한다.",
        "matched_ID_comparison": "V81/V82의모든1035원ID·같은GTmask·qstate·primary/strict membership을검증. 한쪽nomatch/U를비교에서제거하지않음. 같은고정V79fourcases를유지하며새출력승패로교체하지않음.",
        "ownership_diagnostic": "V81정확integerIoU비교/bin/동률규칙과기존800 YCB sameRGB annotation mask lineage를재사용. 모든800 target과same strictC76의U를유지, 새CfalseV점수는새개수로그대로보고한다. bestother는진단값이지모델outputselector아님.",
        "no_new_forward_during_evaluation": True, "requires_root_authorization_after_new_output_seal": True,
        "no_GT_mask_pixels_RGB_models_current_reads_before_bound_outputs_exist_and_root_authorizes": True,
        "no_automatic_current223_admission_even_if_screen_passes": True,
        "source_VAL_development_pretraining_overlap_and_sameclass_scope_limits_preserved": True})
    bindings = {
        "decisions": bind(V / "EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json"),
        "component_criteria": bind(V / "SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json"),
        "V82_input_seal": bind(V / "input_preparation/SEAL.json"),
        "V82_input_manifest": bind(V / "input_preparation/INPUTS_RGB_ONLY.jsonl"),
        "V81_GT_pack_seal": bind(V81 / "source_evaluation_preparation/GT_PACK_SEAL.json"),
        "V81_GT_pack": bind(V81 / "source_evaluation_preparation/GT_EVALUATION_PACK_DO_NOT_FORWARD.npz"),
        "V81_GT_mask_lineage": bind(V81 / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl"),
        "V81_evaluation_seal": bind(V81 / "source_evaluation/GROUNDED_SAM21_FIXED/EVALUATION_SEAL.json"),
        "V81_evaluation_results": bind(V81 / "source_evaluation/GROUNDED_SAM21_FIXED/SOURCE_MASK_RESULTS.json"),
        "V81_reader": bind(V81 / "source_evaluation_preparation/evaluate_saved_grounded_masks.py"),
        "V81_math": bind(V81 / "source_evaluation_preparation/mask_evaluation_math.py"),
        "V81_evaluation_math_policy": bind(V81 / "source_evaluation_preparation/EVALUATION_POLICY_BEFORE_GT_AND_OUTPUTS.json"),
        "V81_ownership_policy": bind(V81 / "saved_target_attribution/DIAGNOSTIC_POLICY_BEFORE_TARGET_ATTRIBUTION_READS.json"),
        "V81_ownership_helpers": bind(V81 / "saved_target_attribution/analyze_saved_target_ownership.py"),
        "V81_same_RGB_GT_800_lineage": bind(V81 / "saved_target_attribution/SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl"),
        "V81_same_RGB_GT_availability": bind(V81 / "saved_target_attribution/SAME_RGB_GT_AVAILABILITY.json"),
        "V81_original_ownership_rows": bind(V81 / "saved_target_attribution/ALL1035_TARGET_ATTRIBUTION_GT_DIAGNOSTIC_ONLY.jsonl"),
        "same_fixed_four_selection": bind(P / "query_visibility_natural_occlusion_audit_20261023_v79/natural_failure_gallery/SELECTION_LOCK_BEFORE_RGB.json"),
        "same_fixed_four_V81_gallery_seal": bind(V81 / "fixed_natural_gallery/FINAL_REPORT_AND_GALLERY_SEAL.json"),
        "wrapper": bind(R / "evaluate_saved_requested_masks.py"),
        "ownership_wrapper": bind(R / "ownership_reuse_saved.py"),
        "generated": bind(R / "GENERATED_EVALUATOR_REUSE_TESTS.json"),
        "evaluation_policy": bind(policy_path), "entry_builder": bind(__file__)}
    write(R / "EVALUATOR_ENTRYLOCK_BEFORE_OUTPUTS.json", {"completed": True, "bindings": bindings,
        "requires_root_authorization_after_output_seal": True, "no_actual_evaluation_run": True,
        "no_automatic_current223_admission": True,
        "expected_output_directory": str(V / "source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED"),
        "expected_comparison_directory": str(V / "source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED_COMPARISON"),
        "actual_GT_mask_pixels_RGB_models_V82_outputs_current_forward_fit_PnP_reads": 0})
    print(json.dumps(bind(R / "EVALUATOR_ENTRYLOCK_BEFORE_OUTPUTS.json")))


if __name__ == "__main__":
    main()
