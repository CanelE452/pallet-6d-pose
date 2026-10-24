"""All-source saved GroundingDINO geometry audit; no model, GT or RGB imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from detector_audit_math import INPUT_FIELDS, associate_scalar, native_boxes_fp32

R = Path(__file__).resolve().parent
V = R.parent


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(record):
    assert bind(record["path"]) == record


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(output_seal, expected_seal_sha, destination):
    assert not destination.exists(), "Preserve prior successful or failed audit"
    destination.mkdir(parents=True)
    policy_path = R / "DETECTOR_AUDIT_POLICY_BEFORE_OUTPUTS.json"
    policy = read(policy_path)
    assert policy["frozen_before_actual_detector_output_values_read"]
    for key in ["decisions", "entry", "code", "guard_contract", "generated_producer", "inputseal", "independent_math", "generated_independent"]:
        check(policy[key])
    assert read(policy["generated_producer"]["path"])["PASS"] and read(policy["generated_independent"]["path"])["PASS"]
    guard = read(policy["guard_contract"]["path"])
    check(guard["official_modeling_source"])
    entry = read(policy["entry"]["path"])
    assert entry["completed"] and entry["code"] == policy["code"] and entry["decisions"] == policy["decisions"]
    assert not entry["current223"] and not entry["fit"] and entry["PnP"] == 0
    for binding in entry["source_inputs"].values():
        check(binding)
    assert bind(output_seal)["sha256"] == expected_seal_sha
    seal = read(output_seal)
    assert seal["completed"] and seal["PASS"] and seal["entrylock"] == policy["entry"]
    assert seal["code"] == policy["code"] and seal["decisions"] == policy["decisions"]
    assert seal["source_inputs"] == entry["source_inputs"] and seal["current223_fit_PnP"] == 0
    for key in ["entrylock", "decisions", "code", "assets", "runtime", "per_RGB", "associations", "summary", "caption_gate", "loading_info", "guard_contract", "first_repeat"]:
        check(seal[key])
    summary = read(seal["summary"]["path"])
    token_gate = read(seal["caption_gate"]["path"])
    loading = read(seal["loading_info"]["path"])
    repeat = read(seal["first_repeat"]["path"])
    for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
        assert key in loading["output_loading_info"] and loading["output_loading_info"][key] == []
    assert loading["PASS"] and loading["all_four_required_lists_empty"] and loading["before_first_source_RGB_decode"] and loading["config_custom_kernels_disabled"]
    decisions = read(policy["decisions"]["path"])
    assert token_gate["PASS"] and token_gate["caption"] == decisions["grounding"]["fixed_caption"]
    assert token_gate["token_count"] == len(token_gate["token_ids"]) == 77
    assert token_gate["root_preflight_exact"] and token_gate["use_fast"] is False
    caption_sha = hashlib.sha256(np.asarray(token_gate["token_ids"], np.int64).tobytes()).hexdigest()
    assert caption_sha == token_gate["token_ids_SHA256"]
    assert repeat["PASS"] and repeat["same_inputs"] and repeat["state_unchanged"]
    assert repeat["logits_maxdiff"] == repeat["boxes_maxdiff"] == 0.
    assert summary["state_unchanged"] and summary["state_SHA256_before"] == summary["state_SHA256_after"]
    assert summary["first_repeat_bitexact"] and summary["source_forward_calls"] == summary["expected_forward_calls"] == 436
    assert summary["NMS"] is False and summary["custom_kernels_disabled"]
    assert not summary["GT_or_per_image_class_forward"] and not summary["source_provenance_forward"]
    assert summary["current223"] == summary["fit"] == summary["PnP"] == 0
    targets = lines(entry["source_inputs"]["manifest"]["path"])
    unique = lines(entry["source_inputs"]["unique_manifest"]["path"])
    detections = lines(seal["per_RGB"]["path"])
    associations = lines(seal["associations"]["path"])
    assert len(unique) == len(detections) == 435 and len(targets) == len(associations) == 1035
    assert len({row["input_id"] for row in targets}) == 1035
    assert len({row["image_sha256"] for row in unique}) == 435
    output_by_sha = {}
    raw_query_count = total_kept = 0
    raw_prob_min, raw_prob_max = 1., 0.
    max_native_box_FP32_diff = 0.
    for original, record in zip(unique, detections):
        assert original["input_id"] == record["rgb_input_id"]
        for key in ["image", "image_sha256", "raw_hw"]:
            assert original[key] == record[key]
        assert record["caption_token_ids_SHA256"] == caption_sha
        assert len(record["decoded_RGB_SHA256"]) == 64
        check(record["detections"])
        with np.load(record["detections"]["path"], allow_pickle=False) as data:
            assert set(data.files) == {"boxes_xyxy", "scores", "detector_query_index", "text_labels", "raw_query_max_scores", "raw_normalized_boxes"}
            box, score, ordinal, labels, raw_score, raw_box = (data[key] for key in ["boxes_xyxy", "scores", "detector_query_index", "text_labels", "raw_query_max_scores", "raw_normalized_boxes"])
        assert raw_score.ndim == 1 and len(raw_score) > 0 and raw_score.dtype == np.float32
        assert raw_box.shape == (len(raw_score), 4) and raw_box.dtype == np.float32
        assert np.isfinite(raw_score).all() and np.isfinite(raw_box).all()
        assert ((raw_score >= 0) & (raw_score <= 1)).all()
        assert box.dtype == score.dtype == np.float32 and ordinal.dtype == np.int64
        assert box.shape == (len(score), 4) and labels.shape == ordinal.shape == score.shape
        assert labels.dtype.kind in "US" and np.isfinite(box).all() and np.isfinite(score).all()
        keep = np.flatnonzero(raw_score > np.float32(.4))
        assert np.array_equal(ordinal, keep)
        assert np.array_equal(score, raw_score[keep])
        native = native_boxes_fp32(raw_box, original["raw_hw"])[keep]
        max_native_box_FP32_diff = max(max_native_box_FP32_diff, float(np.max(np.abs(native - box))) if len(box) else 0.)
        assert np.array_equal(native, box), "FP32 conversion differs; preserve and inspect, no tolerance relaxation"
        assert record["detection_count"] == len(score)
        output_by_sha[original["image_sha256"]] = {"rgb_id": original["input_id"], "box": box, "score": score, "ordinal": ordinal}
        total_kept += len(score)
        raw_query_count += len(raw_score)
        raw_prob_min = min(raw_prob_min, float(raw_score.min()))
        raw_prob_max = max(raw_prob_max, float(raw_score.max()))
    reasons = Counter()
    for target, actual in zip(targets, associations):
        assert set(target) == INPUT_FIELDS
        data = output_by_sha[target["image_sha256"]]
        expected = associate_scalar(target, data["rgb_id"], data["box"], data["score"], data["ordinal"])
        assert actual == expected, target["input_id"]
        reasons[actual["reason"]] += 1
    assert summary["target_instances"] == 1035 and summary["query_slots"] == 8280 and summary["unique_RGB"] == 435
    assert summary["detections"] == total_kept and summary["association_reasons"] == dict(reasons)
    assert summary["associated"] == sum(row["status"] == "ASSOCIATED" for row in associations)
    assert summary["held_U"] == sum(row["status"] == "U" for row in associations)
    report = {"PASS": True, "completed": True, "audit_code": bind(__file__), "policy": bind(policy_path),
        "detector_output_seal": bind(output_seal), "original_RGB_rows": 435, "original_target_rows": 1035,
        "all_RGB_input_order_shape_SHA_caption_identity_exact": True,
        "raw_queries": raw_query_count, "kept_detections": total_kept,
        "all435_saved_raw_querymax_strict0p4_threshold_scores_ordinals_exact": True,
        "all435_normalizedFP32_boxes_to_nativeFP32_boxes_exact": True,
        "native_box_FP32_maxdiff": max_native_box_FP32_diff,
        "all1035_scalar_IoU_score_queryordinal_tie_clippedprompt_U_and_schema_exact": True,
        "all1035_q8_L_initialbox_original_input_unchanged": True,
        "association_reasons": dict(reasons), "saved_raw_querymax_score_range": [raw_prob_min, raw_prob_max],
        "caption77_gate_fixed22phrases_same435images_PASS": True,
        "pretrained_loading_four_required_lists_empty_PASS": True,
        "model_state_unchanged_and_first_repeat_certificate_PASS": True,
        "padding_validation_evidence": "Pinned official contrastiveembedding source + reviewed activefinite/inactive-inf guard + generated malformedfixtures + validate_output_arrays called for all435 unique-image results. Additional first repeat is checked bit-exact to the validated first result. Fullrawlogits notsaved; no independent numericpadding replay claimed.",
        "limits": [policy["saved_limits"],
            "Source RGB body/decodedpixel SHA and actual logits are not reread by this audit; prior sealed input hashes and execution certificates are verified.",
            "Exact association/geometry does not establish semantic detection correctness or target mask ownership."],
        "GT_category_reference_masks_sourceRGB_weights_model_forward_fit_current_PnP_reads": 0}
    write(destination / "INDEPENDENT_DETECTOR_OUTPUT_AUDIT.json", report)
    write(destination / "AUDIT_SEAL.json", {"PASS": True, "files": [bind(path) for path in sorted(destination.iterdir()) if path.is_file()], "existing_artifacts_modified": False})
    print(json.dumps({"PASS": True, "report": bind(destination / "INDEPENDENT_DETECTOR_OUTPUT_AUDIT.json"), "detections": total_kept, "reasons": dict(reasons)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-seal", type=Path, required=True)
    parser.add_argument("--detector-seal-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.detector_seal, args.detector_seal_sha, args.output)
