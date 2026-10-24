"""Run before V80 actual fit results; generated checks and prospectively fixed scope."""
import hashlib
import json
from pathlib import Path
from saved_dense_audit_math import generated_tests

R = Path(__file__).resolve().parents[1]
A = R.parent / "query_visibility_natural_occlusion_audit_20261023_v79"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


generated_path = R / "independent_preparation/GENERATED_SELECTED_SAVED_AUDIT_TESTS.json"
tests = generated_tests()
tests.update(math=bind(R / "independent_preparation/saved_dense_audit_math.py"), code=bind(__file__))
write(generated_path, tests)
decision_path = R / "EXPERIMENT_DECISIONS_BEFORE_GT_BUILD.json"
decision = json.loads(decision_path.read_text())
assert decision["training"]["updates"] == 2000 and decision["training"]["batch_size"] == 96
assert decision["training"]["threshold_visible"] == .5
assert decision["model"]["total_parameters"] == 41287
assert decision["admission_gate"] == {
    "natural_YCB_VAL_C_precision_min": .5, "natural_YCB_VAL_C_recall_min": .8,
    "natural_YCB_VAL_V_recall_min": .8, "controlled_YCB_VAL_pair_both_min": .7,
    "current_eval_only_if_all_gates_pass": True,
    "failure_action": "Preserve completed failed result; no current evaluation or loss/threshold/checkpoint sweep. Review auxiliary ownership maps and next explicit-target grounding candidate prospectively."}
write(R / "independent_preparation/SAVED_AUDIT_POLICY_BEFORE_RESULTS.json", {
    "frozen_before_actual_V80_fit_scores_or_model_results_read": True,
    "decisions": bind(decision_path), "generated_math_tests": bind(generated_path),
    "math": bind(R / "independent_preparation/saved_dense_audit_math.py"),
    "GT_independent_audit": bind(R / "independent_gt_audit/INDEPENDENT_DENSE_GT9_AUDIT.json"),
    "prior_strict535_IDs": bind(A.parent / "query_visibility_controlled_pairs_20261023_v77/source_failure_audit/FIXED_COHORT_IDS.csv"),
    "primary": "Same original primary labels/masks/IDs; binary logit>=0 isVISIBLE, unknown/invalid/refMISS excluded. Saved primary FP32 BCE only known+primary; U placeholders never summed.",
    "sampler": "Reconstruct all2000 deterministic V77 draws at seed20261023, 96each=32natural+32afterNonAnchor+32matchedPair; source/class24; TRAIN and known primary only; control-success required.",
    "selector": "All20 naturalVAL4source/class macroBCE checkpoints at100..2000, strict lower best earliest tie; baseline0 ineligible; pairedVAL/aux maps/strict535 never used for selection.",
    "bindings": "Hash-only checkpoint/V75initialization; entry/model/trainer/input/freeze consistency and generated lambda0/shared-gradient proof; unchanged41237shared+50aux=41287. No parameter deserialization or new model forward.",
    "auxiliary": "Saved FP16 logits all10002x8x2x9x9 finite; channel0=M0,channel1=M4givenM0; derive V=pA*ps,C=pA*(1-ps),O=1-pA in FP64 afterdequant, diagnostic only. Nominal9intersection valid; outsideM0 is Anegative and zeroVloss. Invalid never C/O. Threeclass harddiagnostic fixedargmax orderV,C,O; no primary change.",
    "auxiliary_FP16_scope": "F16-only saved map diagnostics cannot prove FP32 map harddecision parity near0/ties; no unavailable fullprecision comparison invented.",
    "numerics": "Exact discrete counts/ID/mask/sampler comparisons; FP64stableBCE vs savedFP32 primary maxdiff<1e-5 inherited previous saved-audit engineering tolerance, not a source performance threshold.",
    "source_gate": decision["admission_gate"], "strict535_diagnostic_only": True,
    "result_script_adapter": "May add schema-specific audit code after trainer freeze, bind additive code review before actual scores read; does not change this policy or model/protocol choices.",
    "current_new_forward_fit_PnP": 0})
print(json.dumps({"PASS": True, "policy": bind(R / "independent_preparation/SAVED_AUDIT_POLICY_BEFORE_RESULTS.json"),
                  "generated": bind(generated_path)}))
