"""V82 wrapper: unchanged V81 math, explicit category-provided source context."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

R = Path(__file__).resolve().parent
V = R.parent
V81 = V.parent / "query_visibility_grounded_target_20261023_v81"
OLD_PREP = V81 / "source_evaluation_preparation"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(record):
    assert bind(record["path"]) == record


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def imported_reader():
    sys.path.insert(0, str(OLD_PREP))
    file = OLD_PREP / "evaluate_saved_grounded_masks.py"
    spec = importlib.util.spec_from_file_location("unchanged_V81_GT_only_reader", file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def screen(metrics, prospective):
    specifications = [
        ("C_end_to_end_recall_at_least", "C_end_to_end_recall_U_is_miss", "ge"),
        ("V_end_to_end_recall_at_least", "V_end_to_end_recall_U_is_miss", "ge"),
        ("C_precision_known_CV_at_least", "C_precision_known_CV_conditional", "ge"),
        ("hold_rate_at_most", "hold_rate", "le")]
    results = {}
    for criterion, key, direction in specifications:
        value, limit = metrics[key], prospective["criteria"][criterion]
        passed = value is not None and (value >= limit if direction == "ge" else value <= limit)
        results[criterion] = {"value": value, "limit": limit, "comparison": direction, "PASS": bool(passed)}
    return results


def generated():
    base = imported_reader()
    test = base.metrics_CVU(base.confusion_CVU(np.array([0, 0, 1, 1]), np.array([0, 2, 1, 2]), np.ones(4, bool)))
    assert test["C_end_to_end_recall_U_is_miss"] == test["V_end_to_end_recall_U_is_miss"] == .5
    assert test["hold_rate"] == .5
    criteria = read(V / "SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json")
    exact = {"C_end_to_end_recall_U_is_miss": .8, "V_end_to_end_recall_U_is_miss": .8,
        "C_precision_known_CV_conditional": .5, "hold_rate": .1}
    assert all(row["PASS"] for row in screen(exact, criteria).values())
    for key in exact:
        changed = dict(exact)
        changed[key] = None if key != "hold_rate" else np.nextafter(.1, 1.).item()
        assert not all(row["PASS"] for row in screen(changed, criteria).values())
    for key in ["C_end_to_end_recall_U_is_miss", "V_end_to_end_recall_U_is_miss", "C_precision_known_CV_conditional"]:
        changed = dict(exact); changed[key] = np.nextafter(exact[key], 0.).item()
        assert not all(row["PASS"] for row in screen(changed, criteria).values())
    write(R / "GENERATED_EVALUATOR_REUSE_TESTS.json", {"PASS": True, "generated_only": True,
        "code": bind(__file__), "V81_reader": bind(OLD_PREP / "evaluate_saved_grounded_masks.py"),
        "V81_math": bind(OLD_PREP / "mask_evaluation_math.py"), "same_CM_U_miss_formula": True,
        "prospective_screen_inclusive_endpoints_and_null_fail": True,
        "no_actual_GT_pixels_saved_V82_outputs_RGB_model_current_forward_fit_PnP": True})


def run(manifest, sam_seal, sam_sha, destination):
    entry = read(R / "EVALUATOR_ENTRYLOCK_BEFORE_OUTPUTS.json")
    assert entry["completed"] and entry["requires_root_authorization_after_output_seal"]
    for record in entry["bindings"].values():
        check(record)
    assert bind(manifest)  # Root supplies the completed mask manifest and exact seal SHA.
    assert bind(sam_seal)["sha256"] == sam_sha
    base = imported_reader()
    base.V = V  # Only the model-input manifest changes to the eight-field V82 packet.
    original_save = base.save
    def contextual_save(path, value):
        if Path(path).name == "SOURCE_MASK_RESULTS.json":
            value = dict(value)
            value["inherited_formula_code"] = value.pop("code")
            value["code"] = bind(__file__)
            value["inherited_formula_policy"] = value.pop("policy")
            value["policy"] = entry["bindings"]["evaluation_policy"]
            assert value.pop("GT_input_to_models_or_posthoc_association_mask_choice") is False
            value["source_GT_requested_category_metadata_provided_to_detector"] = True
            value["GT_masks_reference_pose_depth_visibility_or_instance_token_to_models"] = False
            value["GT_used_to_choose_detector_or_SAM_mask_after_prediction"] = False
            value["changed_information_contract_KO"] = "YCB는sourceGTmetadata로요청category를제공한추가입력조건이며, SYNTH는고정pallet요청이다. V81과동일입력비교나예측category로주장하지않는다."
            value["no_automatic_current223_admission"] = True
        original_save(path, value)
    base.save = contextual_save
    base.evaluate(manifest, sam_seal, sam_sha, destination)
    result = read(destination / "SOURCE_MASK_RESULTS.json")
    strict = result["benchmarks"][0]["strict535_same_frozen_IDs"]
    criteria = read(entry["bindings"]["component_criteria"]["path"])
    checks = screen(strict, criteria)
    comparison_dir = destination.parent / (destination.name + "_COMPARISON")
    comparison_dir.mkdir(exist_ok=False)
    npz_name = "SAVED_MASK_EVALUATION_PER_QUERY.npz"
    with np.load(V81 / "source_evaluation/GROUNDED_SAM21_FIXED" / npz_name, allow_pickle=False) as old, np.load(destination / npz_name, allow_pickle=False) as new:
        for key in ["input_id", "original_id", "source", "original_primary8", "reference_local_label8", "q_pixel_state8", "q_pixel_valid8", "strict535_mask8"]:
            assert np.array_equal(old[key], new[key]), key
        ids = new["original_id"]
        cases = read(entry["bindings"]["same_fixed_four_selection"]["path"])["cases"]
        id_index = {ident: i for i, ident in enumerate(ids)}
        fixed = []
        for case in cases:
            i, k = id_index[case["id"]], case["native_k"]
            assert new["strict535_mask8"][i, k]
            fixed.append({"original_id": case["id"], "native_k": k, "V79_fixed_cell": case["cell"],
                "GT_reference_label": int(new["reference_local_label8"][i, k]),
                "V81_prediction_CVU": int(old["prediction_CVU"][i, k]), "V82_prediction_CVU": int(new["prediction_CVU"][i, k]),
                "V81_whole_mask_IoU_no_mask0": float(old["IoU_allrows_nomask0"][i]),
                "V82_whole_mask_IoU_no_mask0": float(new["IoU_allrows_nomask0"][i])})
        availability = Counter((bool(a), bool(b)) for a, b in zip(old["mask_available"], new["mask_available"]))
        strict_transitions = []
        for label in [0, 1]:
            take = new["strict535_mask8"] & (new["reference_local_label8"] == label)
            counts = Counter(zip(old["prediction_CVU"][take].tolist(), new["prediction_CVU"][take].tolist()))
            strict_transitions.append({"GT_label": label, "N": int(take.sum()), "V81_to_V82_CVU_counts": [[counts[a, b] for b in [0, 1, 2]] for a in [0, 1, 2]]})
    write(comparison_dir / "SAME_IDS_V81_V82_COMPARISON_KO.json", {"completed": True, "PASS": True,
        "all1035_IDs_GT_primary_qstates_strict_masks_exact": True, "not_same_input_ablation": True,
        "V81_eval_seal": entry["bindings"]["V81_evaluation_seal"], "V82_eval_seal": bind(destination / "EVALUATION_SEAL.json"),
        "same_fixed_four_cases": fixed, "same_strict535_prediction_transitions": strict_transitions,
        "mask_availability_transitions_all1035": [dict(V81=a, V82=b, N=n) for (a,b),n in sorted(availability.items())],
        "기준": "모든기존ID를비교하며기존또는새mask보류를제거하지않음. 종류제공정보조건이달라졌음을명시한다."})
    write(comparison_dir / "PROSPECTIVE_COMPONENT_SCREEN_KO.json", {"completed": True,
        "criteria": entry["bindings"]["component_criteria"], "fixed_strict535_C76_V459": True,
        "checks": checks, "component_screen_PASS": all(row["PASS"] for row in checks.values()),
        "자동현재223평가승인": False, "해석": "통과해도종류제공source구성요소만검증한다. 인간DES·corner cue·실제initializer오차전이·동일종류다중instance는입증하지못한다."})
    from ownership_reuse_saved import run as ownership
    ownership(destination, manifest, comparison_dir)
    write(comparison_dir / "COMPARISON_AND_SCREEN_SEAL.json", {"completed": True, "PASS": True,
        "files": [bind(path) for path in sorted(comparison_dir.iterdir()) if path.is_file()],
        "source_evaluation_seal": bind(destination / "EVALUATION_SEAL.json"), "new_forward_fit_current_PnP": 0})
    print(json.dumps({"PASS": True, "evaluation": bind(destination / "EVALUATION_SEAL.json"),
        "comparison": bind(comparison_dir / "COMPARISON_AND_SCREEN_SEAL.json"), "component_screen": checks}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generated", "run"])
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--sam-seal", type=Path)
    parser.add_argument("--sam-seal-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "generated":
        generated()
    else:
        assert all(value is not None for value in [args.output_manifest, args.sam_seal, args.sam_seal_sha, args.output])
        run(args.output_manifest, args.sam_seal, args.sam_seal_sha, args.output)
