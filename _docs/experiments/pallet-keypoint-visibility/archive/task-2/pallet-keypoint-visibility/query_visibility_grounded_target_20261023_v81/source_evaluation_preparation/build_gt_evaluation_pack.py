"""GT-only benchmark join. No model, RGB, mask pixels or source feature vectors."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from mask_evaluation_math import generated_tests

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent
V76 = P / "query_visibility_reference_local_support_20261023_v76"
V79 = P / "query_visibility_natural_occlusion_audit_20261023_v79"


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(record):
    assert bind(record["path"]) == record


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    policy_path = R / "EVALUATION_POLICY_BEFORE_GT_AND_OUTPUTS.json"
    policy = json.loads(policy_path.read_text())
    assert policy["frozen_before_V81_source_GT_mask_or_output_reads"]
    for key in ["decisions", "input_seal"]:
        check(policy[key])
    test = generated_tests()
    test.update(code=bind(__file__), math=bind(R / "mask_evaluation_math.py"))
    write(R / "GENERATED_EVALUATION_MATH_TESTS.json", test)
    inputs_path = V / "input_preparation/INPUTS_RGB_ONLY.jsonl"
    provenance_path = V / "input_preparation/PROVENANCE_DO_NOT_FORWARD.jsonl"
    inputs, provenance = read_lines(inputs_path), read_lines(provenance_path)
    assert len(inputs) == len(provenance) == 1035
    gt_rows_path = V76 / "gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl"
    gt_index_path = V76 / "gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl"
    gt_rows, gt_index = read_lines(gt_rows_path), read_lines(gt_index_path)
    coverage_path = V79 / "coverage/results/COVERAGE_PER_QUERY.npz"
    keys = ["id", "source", "split", "domain", "query8_xy", "L", "raw_hw", "teacher_label8", "primary_loss_mask", "input_available", "reference_in_local_square_GT_only", "q_center_mask_valid", "q_center_M0", "q_center_M4", "q_center_support_state"]
    with np.load(coverage_path, allow_pickle=False) as data:
        coverage = {key: data[key] for key in keys}
    original = np.array([row["original_row_index"] for row in provenance], np.int64)
    assert len(np.unique(original)) == 1035 and (original < 5203).all()
    assert not coverage["domain"][original].any()
    assert (coverage["split"][original] == "val").all()
    query = np.array([row["initial_points"] for row in inputs], np.float64)
    side = np.array([row["roi_side"] for row in inputs], np.float64)
    hw = np.array([row["raw_hw"] for row in inputs], np.int32)
    assert np.array_equal(query, coverage["query8_xy"][original])
    assert np.array_equal(side, coverage["L"][original])
    assert np.array_equal(hw, coverage["raw_hw"][original])
    masks = []
    for i, (input_row, prov) in enumerate(zip(inputs, provenance)):
        row = int(original[i]); g, meta = gt_rows[row], gt_index[row]
        assert input_row["input_id"] == prov["input_id"]
        assert prov["original_id"] == g["id"] == meta["id"] == coverage["id"][row]
        assert prov["source"] == g["source"] == meta["source"] == coverage["source"][row]
        if "image_sha256_lineage_only" in meta:
            assert meta["image_sha256_lineage_only"] == input_row["image_sha256"]
            RGB_GT_join = "GT metadata imageSHA equals inputSHA"
        else:
            assert prov["source"] == "SYNTHETIC"
            frame_token = g["id"].split("__")[-1]
            assert Path(input_row["image"]).name == frame_token + "_rgb.png"
            assert Path(g["mask_pair"]["M0"]["path"]).stem == frame_token
            assert Path(g["mask_pair"]["M4"]["path"]).stem == frame_token
            RGB_GT_join = "Original frozen row/ID, nativeHW and RGB/M0/M4 frame-token lineage; SYNTH GTindex has no independent RGB SHA field"
        assert meta["raw_hw"] == input_row["raw_hw"]
        pair = g["mask_pair"]
        expected_threshold = 127 if prov["source"] == "SYNTHETIC" else 0
        assert pair["threshold_strict_gt"] == expected_threshold
        masks.append({"input_id": input_row["input_id"], "original_row_index": row,
            "original_id": g["id"], "source": g["source"], "split": "val",
            "image_sha256": input_row["image_sha256"], "raw_hw": input_row["raw_hw"],
            "M0": pair["M0"], "M4": pair["M4"], "mask_threshold_strict_gt": expected_threshold,
            "M0_target_area_prior_metadata": pair["M0_target_area"], "M4_target_area_prior_metadata": pair["M4_target_area"],
            "RGB_GT_join_evidence": RGB_GT_join})
    strict_path = P / "query_visibility_controlled_pairs_20261023_v77/source_failure_audit/FIXED_COHORT_IDS.csv"
    strict_rows = [row for row in csv.DictReader(strict_path.open()) if row["cohort"] == "STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL"]
    id_map = {ident: i for i, ident in enumerate(coverage["id"][original])}
    strict = np.zeros((1035, 8), bool)
    for row in strict_rows:
        strict[id_map[row["id"]], int(row["native_k"])] = True
    y = coverage["teacher_label8"][original]
    primary = coverage["primary_loss_mask"][original]
    qstate = coverage["q_center_support_state"][original]
    qvalid = coverage["q_center_mask_valid"][original]
    assert strict.sum() == 535 and int((strict & (y == 0)).sum()) == 76 and int((strict & (y == 1)).sum()) == 459
    assert not (strict & ~primary).any() and np.array_equal(qstate[strict], y[strict])
    gt_pack = {"input_id": np.array([row["input_id"] for row in inputs]),
        "original_id": coverage["id"][original], "original_row_index": original,
        "source": coverage["source"][original], "split": coverage["split"][original],
        "query8_xy": query, "L": side, "raw_hw": hw, "reference_local_label8": y,
        "original_primary8": primary, "input_available8": coverage["input_available"][original],
        "reference_in_local_square8": coverage["reference_in_local_square_GT_only"][original],
        "q_pixel_valid8": qvalid, "q_pixel_M08": coverage["q_center_M0"][original],
        "q_pixel_M48": coverage["q_center_M4"][original], "q_pixel_state8_CVO_invalid": qstate,
        "strict535_mask8": strict}
    np.savez_compressed(R / "GT_EVALUATION_PACK_DO_NOT_FORWARD.npz", **gt_pack)
    with (R / "GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl").open("x") as stream:
        for row in masks:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    counts = []
    for source in ["SYNTHETIC", "YCB_PUBLIC_REAL"]:
        f = coverage["source"][original] == source
        counts.append({"source": source, "targets": int(f.sum()), "all_query_slots": int(f.sum()) * 8,
            "original_primary_C": int((primary[f] & (y[f] == 0)).sum()),
            "original_primary_V": int((primary[f] & (y[f] == 1)).sum()),
            "q_pixel_states": {str(state): int(np.count_nonzero(qstate[f] == state)) for state in [-1, 0, 1, 2]},
            "all_reference_teacher_labels": {str(label): int(np.count_nonzero(y[f] == label)) for label in [-1, 0, 1]}})
    write(R / "GT_PACK_BUILD_RESULTS.json", {"completed": True, "PASS": True, "policy": bind(policy_path),
        "targets": 1035, "queries": 8280, "counts": counts, "strict535_C76_V459_and_qstate_same_as_reference": True,
        "original_primary_labels_masks_unchanged": True,
        "mask_and_RGB_pixels_model_weights_features_forward_fit_current_PnP_reads": 0,
        "bindings": [bind(path) for path in [inputs_path, provenance_path, gt_rows_path, gt_index_path, coverage_path, strict_path]],
        "scope": "GT-only evaluation join; never supplied to detector/SAM. PixelC/V/O and reference-local C/V/U are distinct benchmarks, neither human semanticD/E/S."})
    write(R / "GT_PACK_SEAL.json", {"completed": True, "PASS": True, "files": [bind(R / name) for name in [
        "EVALUATION_POLICY_BEFORE_GT_AND_OUTPUTS.json", "mask_evaluation_math.py", "build_gt_evaluation_pack.py",
        "GENERATED_EVALUATION_MATH_TESTS.json", "GT_EVALUATION_PACK_DO_NOT_FORWARD.npz", "GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl", "GT_PACK_BUILD_RESULTS.json"]]})
    print(json.dumps({"PASS": True, "seal": bind(R / "GT_PACK_SEAL.json"), "counts": counts}))


if __name__ == "__main__":
    main()
