"""Saved V80 source audit. Explicit execution only after root FIT_COMPLETE notice.

No model/torch/feature imports, no checkpoint deserialization, no source RGB.
Uses unchanged V79 audited masks/metadata and source saved score arrays.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from saved_dense_audit_math import map_confusion

R = Path(__file__).resolve().parents[1]
P = R.parent
V77 = P / "query_visibility_controlled_pairs_20261023_v77"
V79 = P / "query_visibility_natural_occlusion_audit_20261023_v79"
RUN = R / "runs/SOURCE_DENSE_AUX_V1"
sys.path.insert(0, str(V77 / "independent_preparation"))
from pair_audit_primitives import audit_natural_selector, binary_metric, paired_saved_metrics, stable_bce
from audit_completed_source_experiment import sampler as replay_sampler


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def check(binding):
    assert bind(binding["path"]) == binding, binding["path"]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in keys}


def metric(logit, nll, y, mask):
    m = binary_metric(logit, y, mask)
    return {"N": m["N"], "confusion_rows_GT_cols_pred_COVERED_VISIBLE": m["CM_rows_CV_cols_CV"],
            "visible_precision": m["V_precision"], "visible_recall": m["V_recall"],
            "covered_precision": m["C_precision"], "covered_recall": m["C_recall"],
            "mean_BCE_FP32": float(nll[mask].astype(float).mean()) if mask.any() else None}


def run(output):
    assert not output.exists(), "Preserve any previous successful or failed audit"
    output.mkdir(parents=True)
    policy_path = R / "independent_preparation/SAVED_AUDIT_POLICY_BEFORE_RESULTS.json"
    policy = read(policy_path)
    for key in ["decisions", "generated_math_tests", "math", "GT_independent_audit", "prior_strict535_IDs"]:
        check(policy[key])
    assert policy["frozen_before_actual_V80_fit_scores_or_model_results_read"]
    decision = read(policy["decisions"]["path"])
    assert bind(policy["decisions"]["path"])["sha256"] == "81ac5890ce9a6d210bce5403425d589351c6306c6379755f153182ed4fafd35a"
    complete, freeze, history, pub = (read(RUN / name) for name in ["FIT_COMPLETE.json", "MODEL_FREEZE.json", "HISTORY.json", "SOURCE_RESULTS.json"])
    entry_path = R / "TRAINER_ENTRYLOCK.json"
    entry = read(entry_path)
    assert complete["completed"] and freeze["completed"] and complete["optimizer_updates"] == freeze["optimizer_updates"] == 2000
    assert freeze["parameter_count"] == 41287 and freeze["added_auxiliary_parameters"] == 50
    assert freeze["threshold"] == .5 and freeze["aux_weight"] == .25
    assert not freeze["auxiliary_used_by_primary_selector"]
    assert not freeze["current_used_for_fit_or_checkpoint_or_threshold_selection"] and freeze["PnP"] == 0
    for key in ["checkpoint", "model_code", "loss_code", "decisions", "trainer", "entry"]:
        check(freeze[key])
    for key in ["model_freeze", "history", "scores", "auxiliary_logits", "source_results"]:
        check(complete[key])
    assert freeze["entry"] == bind(entry_path)
    assert entry["trainer"] == freeze["trainer"] == bind(R / "train_source_once.py")
    assert entry["decisions"] == freeze["decisions"] == policy["decisions"]
    assert entry["runtime"]["torch"] == "2.5.1+cu121" and entry["runtime"]["device"] == "cuda:0"
    assert not entry["restart_after_first_update"] and not entry["current_read_or_PnP"]
    for binding in entry["artifacts"]:
        check(binding)
    check(entry["generated"])
    for key in ["dense_GT", "dense_GT_seal", "independent_GT", "loss_review"]:
        check(entry[key])
    assert read(entry["independent_GT"]["path"])["PASS"] and read(entry["loss_review"]["path"])["PASS"]
    for binding in decision["bindings"].values():
        check(binding)
    old_inputs = read(decision["bindings"]["V77_inputs"]["path"])
    for binding in old_inputs["bindings"].values():
        check(binding)
    old_decision = read(decision["bindings"]["V77_decisions"]["path"])
    keys = ["id", "source", "split", "label8", "native_GTref8", "convex_self_proxy8_separate", "near_boundary8", "rule8"]
    g0 = npz(old_decision["bindings"]["teacher_GT"]["path"], keys)
    g1 = npz(old_inputs["bindings"]["after_teacher_GT"]["path"], keys + ["original_row_index", "anchor_k", "pair_anchor_success", "anchor_training_target", "original_primary8", "query_valid8", "reference_in_local_square8"])
    gt = {key: np.concatenate([g0[key], g1[key]]) for key in keys}
    assert len(g0["id"]) == 5203 and len(g1["id"]) == 4799
    original = g1["original_row_index"]
    anchor = g1["anchor_k"].astype(int)
    c = npz(V79 / "coverage/results/COVERAGE_PER_QUERY.npz", ["id", "source", "split", "domain", "original_row_index", "teacher_label8", "primary_loss_mask", "input_available", "reference_in_local_square_GT_only", "reference_offset_L_GT_only", "q_reference_error_px_GT_only"])
    for key in ["id", "source", "split"]:
        assert np.array_equal(c[key], gt[key])
    assert np.array_equal(c["teacher_label8"], gt["label8"])
    primary, available, square = (c[key] for key in ["primary_loss_mask", "input_available", "reference_in_local_square_GT_only"])
    assert primary.shape == (10002, 8) and not (primary & ~np.isin(gt["label8"], [0, 1])).any()
    assert not (primary & (~available | ~square)).any()
    assert np.array_equal(c["original_row_index"], np.r_[np.arange(5203), original])
    assert np.array_equal(c["domain"], np.r_[np.zeros(5203), np.ones(4799)])
    assert np.array_equal(g1["original_primary8"], primary[original])
    assert np.array_equal(g1["reference_in_local_square8"], square[5203:])
    assert np.array_equal(g1["anchor_training_target"], np.where(g1["pair_anchor_success"], 0, -1))
    good = g1["pair_anchor_success"] & (g1["anchor_training_target"] == 0) & primary[5203:][np.arange(4799), anchor] & (g1["label8"][np.arange(4799), anchor] == 0)
    data = (None, g0, g1, None, None, primary[:5203], primary[5203:], square[:5203], square[5203:], available[:5203], available[5203:], good)
    sampler = replay_sampler(data, pub)
    assert sampler["deterministic_draws_SHA256"] == freeze["training_sample_sequence_SHA256"]
    old_sampler = read(V77 / "independent_saved_audit/INPUT_ENTRY_AND_SAMPLER_AUDIT.json")["actual_sampler"]
    assert sampler["deterministic_draws_SHA256"] == old_sampler["deterministic_draws_SHA256"]
    score_keys = ["id", "source", "split", "domain", "primary_loss_mask", "input_available", "reference_in_local_square_GT_only", "after_original_row_index", "after_anchor_k", "pair_good", "support_logit", "NLL_FP32"]
    selected = npz(complete["scores"]["path"], score_keys)
    for key, value in [("id", gt["id"]), ("source", gt["source"]), ("split", gt["split"]), ("domain", c["domain"]), ("primary_loss_mask", primary), ("input_available", available), ("reference_in_local_square_GT_only", square), ("after_original_row_index", original), ("after_anchor_k", anchor), ("pair_good", good)]:
        assert np.array_equal(selected[key], value), key
    logit, nll, y = selected["support_logit"], selected["NLL_FP32"], gt["label8"]
    assert logit.shape == nll.shape == (10002, 8) and logit.dtype == nll.dtype == np.float32
    assert np.isfinite(logit).all() and np.isfinite(nll).all()
    bce_diff = np.abs(stable_bce(logit[primary].astype(float), y[primary]) - nll[primary])
    assert bce_diff.max() < 1.e-5
    selector = audit_natural_selector(history, metric_key="naturalVAL4strata_macroBCE", strata_key="stratumBCE")
    assert selector["best_update"] == freeze["best_update"] == complete["best_update"] == pub["best_update"]
    assert selector["natural4strata_macroBCE"] == freeze["best_naturalVAL4strata_macroBCE"] == pub["selection_naturalVALmacroBCE"]
    rows, val_strata = [], []
    for domain, domain_name in [(0, "NATURAL_ORIGINAL"), (1, "CONTROLLED_AFTER")]:
        for source in ["SYNTHETIC", "YCB_PUBLIC_REAL"]:
            for split in ["train", "val"]:
                base = np.broadcast_to(((c["domain"] == domain) & (gt["source"] == source) & (gt["split"] == split))[:, None], primary.shape)
                mask = base & primary
                m = metric(logit, nll, y, mask)
                published = pub["results"]["natural_and_after"][f"{domain_name}|{source}|{split}"]
                assert m == published["metrics"]
                counts = {"all_slots": int(base.sum()), "known": int((base & np.isin(y, [0, 1])).sum()), "admitted": int(mask.sum()), "U": int((base & (y == -1)).sum()), "known_ref_outside_square": int((base & np.isin(y, [0, 1]) & ~square).sum())}
                assert all(published[key] == value for key, value in counts.items())
                subgroups = {"strong": np.char.startswith(gt["rule8"].astype(str), "STRONG"), "fallback": ~np.char.startswith(gt["rule8"].astype(str), "STRONG"), "convex_self_hidden": gt["convex_self_proxy8_separate"] == 0, "convex_may_visible": gt["convex_self_proxy8_separate"] == 1}
                for key, condition in subgroups.items():
                    assert metric(logit, nll, y, mask & condition) == published["subgroups"][key]
                rows.append({"domain": domain_name, "source": source, "split": split, "metrics": m, "counts": counts})
                if domain == 0 and split == "val":
                    for label in [0, 1]:
                        use = mask & (y == label)
                        val_strata.append({"source": source, "label": label, "N": int(use.sum()), "BCE_FP32": float(nll[use].astype(float).mean())})
    saved_macro = float(np.mean([row["BCE_FP32"] for row in val_strata]))
    assert abs(saved_macro - selector["natural4strata_macroBCE"]) < 1.e-6
    pairs = paired_saved_metrics(logit[original], logit[5203:], g0["label8"][original], g1["label8"], primary[original], primary[5203:], anchor, g1["source"], g1["split"], g1["pair_anchor_success"])
    for row in pairs:
        pp = pub["results"]["paired_anchors"][row["source"] + "|" + row["split"]]
        for audit_key, pub_key in [("attempted_pairs", "attempted_rows"), ("failed_or_unknown_or_invalid_anchor_rows", "failed_or_unknown_or_invalid_anchor_rows"), ("known_admitted_V_to_C_pairs", "N"), ("both_correct_known_pair", "both_correct"), ("before_V_correct_known_pair", "before_VISIBLE_correct"), ("after_C_correct_known_pair", "after_COVERED_correct")]:
            assert row[audit_key] == pp[pub_key]
    baseline = npz(RUN / "BASELINE0_SOURCE_SCORES.npz", score_keys)
    for key in score_keys[:-2]:
        assert np.array_equal(baseline[key], selected[key])
    assert baseline["support_logit"].shape == (10002, 8) and not baseline["support_logit"].any()
    assert np.max(np.abs(baseline["NLL_FP32"][primary] - np.log(2.))) < 1.e-7
    assert read(RUN / "BASELINE0_RESULTS.json")["selection_eligible"] is False
    preflight = read(RUN / "SOURCE_GPU_PREFLIGHT.json")
    assert preflight["PASS"] and preflight["same_V77_initial_shared_state_and_primary_FP32"] and preflight["same_primary_admission"]
    assert preflight["optimizer_updates"] == 0 and not preflight["GTforward"] and preflight["current_PnP"] == 0
    cohort_rows = list(csv.DictReader(Path(policy["prior_strict535_IDs"]["path"]).open()))
    id_to_row = {ident: i for i, ident in enumerate(gt["id"])}
    strict = np.zeros_like(primary)
    for row in cohort_rows:
        if row["cohort"] == "STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL":
            strict[id_to_row[row["id"]], int(row["native_k"])] = True
    strict_recomputed = primary & ((c["domain"] == 0) & (gt["source"] == "YCB_PUBLIC_REAL") & (gt["split"] == "val"))[:, None]
    strict_recomputed &= np.char.startswith(gt["rule8"].astype(str), "STRONG") & (c["q_reference_error_px_GT_only"] <= 3.) & (np.max(np.abs(c["reference_offset_L_GT_only"]), axis=-1) <= 1. / 18.)
    assert np.array_equal(strict, strict_recomputed) and strict.sum() == 535
    strict_metric = metric(logit, nll, y, strict)
    public = next(row["metrics"] for row in rows if row["domain"] == "NATURAL_ORIGINAL" and row["source"] == "YCB_PUBLIC_REAL" and row["split"] == "val")
    pair = next(row for row in pairs if row["source"] == "YCB_PUBLIC_REAL" and row["split"] == "val")
    gate = {
        "natural_YCB_VAL_C_precision_min": {"value": public["covered_precision"], "minimum": .5},
        "natural_YCB_VAL_C_recall_min": {"value": public["covered_recall"], "minimum": .8},
        "natural_YCB_VAL_V_recall_min": {"value": public["visible_recall"], "minimum": .8},
        "controlled_YCB_VAL_pair_both_min": {"value": pair["both_correct_known_pair"] / pair["known_admitted_V_to_C_pairs"], "minimum": .7}}
    for value in gate.values():
        value["PASS"] = value["value"] is not None and value["value"] >= value["minimum"]
    for key, value in gate.items():
        assert decision["admission_gate"][key] == value["minimum"]

    # FP16-only map diagnostics; never used for primary threshold/selection.
    aux = npz(complete["auxiliary_logits"]["path"], ["aux_mask_logits", "id", "source", "split", "domain", "channels"])
    aux_logit = aux["aux_mask_logits"]
    assert aux_logit.shape == (10002, 8, 2, 9, 9) and aux_logit.dtype == np.float16 and np.isfinite(aux_logit).all()
    assert aux["channels"].tolist() == ["expected_support_M0_logit", "visible_given_M0_logit"]
    for key in ["id", "source", "split", "domain"]:
        assert np.array_equal(aux[key], c[key])
    dense = npz(entry["dense_GT"]["path"], ["valid9", "expected_support_M0", "visible_target_M4"])
    valid, full, visible = (dense[key] for key in ["valid9", "expected_support_M0", "visible_target_M4"])
    aux_rows = []
    for domain in [0, 1]:
        for source in ["SYNTHETIC", "YCB_PUBLIC_REAL"]:
            for split in ["train", "val"]:
                frame = (c["domain"] == domain) & (gt["source"] == source) & (gt["split"] == split)
                cm, probs = map_confusion(aux_logit[frame], valid[frame], full[frame], visible[frame])
                assert np.max(np.abs(probs.sum(-3) - 1.)) < 1.e-14
                a_pred = aux_logit[frame, :, 0] >= 0
                s_pred = aux_logit[frame, :, 1] >= 0
                a_truth, v_truth = full[frame], visible[frame]
                a_use, v_use = valid[frame], full[frame]
                binary = lambda target, prediction, use: [[int(np.sum(use & (target == g) & (prediction == k))) for k in [0, 1]] for g in [0, 1]]
                aux_rows.append({"domain": domain, "source": source, "split": split,
                    "nominal9_valid_cells": int(valid[frame].sum()), "CM_rows_GT_cols_pred_V_C_O": cm.tolist(),
                    "A_binary_CM_rows_GT_cols_pred_0_1": binary(a_truth, a_pred, a_use),
                    "visible_given_M0_binary_CM_rows_GT_cols_pred_0_1": binary(v_truth, s_pred, v_use),
                    "exact_zero_aux_logit_cells": int(np.count_nonzero(aux_logit[frame] == 0))})
    baseline_aux = npz(RUN / "BASELINE0_AUX_LOGITS.npz", ["aux_mask_logits"])["aux_mask_logits"]
    assert baseline_aux.shape == aux_logit.shape and baseline_aux.dtype == np.float16 and not baseline_aux.any()
    save(output / "AUXILIARY_MAP_DIAGNOSTICS.json", {"completed": True, "diagnostic_only": True, "saved_dtype": "float16",
        "shape": list(aux_logit.shape), "channels": aux["channels"].tolist(), "strata": aux_rows,
        "probability_rule": "V=sigmoid(A)*sigmoid(S),C=sigmoid(A)*(1-sigmoid(S)),O=1-sigmoid(A); fixedargmaxV,C,O",
        "FP16_limit": policy["auxiliary_FP16_scope"], "primary_checkpoint_threshold_changed": False})
    report = {"PASS": True, "code": bind(__file__), "frozen_policy": bind(policy_path), "checkpoint": freeze["checkpoint"],
        "parameter_count": 41287, "added_auxiliary_parameters": 50, "initialization_shared_V75_and_zero_aux_saved_preflight_PASS": True,
        "selection": selector, "saved_natural4strata_macroBCE": saved_macro,
        "saved_minus_history_macroBCE": saved_macro - selector["natural4strata_macroBCE"],
        "NLL64_vs_savedFP32_maxdiff_primary_only": float(bce_diff.max()),
        "all80016_ID_primary_inputavailability_square_teacher_exact": True,
        "all_published_metrics_subgroups_counts_exact": True, "source_metrics": rows, "naturalVAL_strata": val_strata,
        "actual_sampler": sampler, "paired_and_same_after_control_diagnostics": pairs,
        "strict535": {"fixed_IDs_and_geometry_exact": True, "counts_C76_V459": [int((strict & (y == label)).sum()) for label in [0, 1]], "metrics": strict_metric},
        "prospective_source_gate": gate, "prospective_source_gate_PASS": all(x["PASS"] for x in gate.values()),
        "auxiliary_saved_shape_finite_F16_channels_metadata_PASS": True,
        "U_NLL_placeholder_never_summed": True, "checkpoint_deserialization_RGB_features_model_forward_fit_current_PnP": 0,
        "bindings": [bind(RUN / name) for name in ["FIT_COMPLETE.json", "MODEL_FREEZE.json", "HISTORY.json", "SOURCE_RESULTS.json", "SELECTED_SOURCE_SCORES.npz", "SELECTED_AUX_LOGITS.npz", "SOURCE_GPU_PREFLIGHT.json"]]}
    save(output / "INDEPENDENT_SELECTED_SAVED_AUDIT.json", report)
    save(output / "AUDIT_SEAL.json", {"PASS": True, "bindings": [bind(path) for path in sorted(output.iterdir()) if path.is_file()], "existing_artifacts_modified": False})
    print(json.dumps({"PASS": True, "report": bind(output / "INDEPENDENT_SELECTED_SAVED_AUDIT.json"), "selection": selector, "source_gate": gate}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
