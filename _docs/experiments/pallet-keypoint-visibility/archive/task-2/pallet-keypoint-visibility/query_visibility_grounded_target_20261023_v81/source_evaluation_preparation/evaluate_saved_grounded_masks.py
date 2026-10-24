"""Evaluate sealed source masks only. No detector/SAM/model imports or forward."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mask_evaluation_math import confusion_CVU, metrics_CVU, iou_or_zero, sample_native_mask

R = Path(__file__).resolve().parent
V = R.parent


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def check(record):
    assert bind(record["path"])["sha256"] == record["sha256"], record["path"]


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def contains_binding(value, binding):
    if isinstance(value, dict):
        if value.get("path") == binding["path"] and value.get("sha256") == binding["sha256"]:
            return True
        return any(contains_binding(child, binding) for child in value.values())
    if isinstance(value, list):
        return any(contains_binding(child, binding) for child in value)
    return False


def evaluate(manifest, sam_seal_path, sam_seal_sha, output):
    assert not output.exists(), "Preserve any prior output or failed attempt"
    output.mkdir(parents=True)
    policy_path = R / "EVALUATION_POLICY_BEFORE_GT_AND_OUTPUTS.json"
    policy = json.loads(policy_path.read_text())
    assert policy["frozen_before_V81_source_GT_mask_or_output_reads"]
    check(policy["decisions"])
    check(policy["input_seal"])
    gt_seal_path = R / "GT_PACK_SEAL.json"
    gt_seal = json.loads(gt_seal_path.read_text())
    assert gt_seal["completed"] and gt_seal["PASS"]
    for binding in gt_seal["files"]:
        check(binding)
    assert bind(sam_seal_path)["sha256"] == sam_seal_sha
    sam_seal = json.loads(Path(sam_seal_path).read_text())
    assert sam_seal.get("completed") is True
    assert contains_binding(sam_seal, bind(manifest)), "Output manifest must be bound by SAM seal"
    targets = lines(V / "input_preparation/INPUTS_RGB_ONLY.jsonl")
    predicted_rows = lines(manifest)
    gt_masks = lines(R / "GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl")
    assert len(targets) == len(predicted_rows) == len(gt_masks) == 1035
    with np.load(R / "GT_EVALUATION_PACK_DO_NOT_FORWARD.npz", allow_pickle=False) as data:
        gt = {key: data[key] for key in data.files}
    assert gt["input_id"].tolist() == [row["input_id"] for row in targets]
    predictions = np.full((1035, 8), 2, np.int8)
    mask_available = np.zeros(1035, bool)
    associated = np.zeros(1035, bool)
    iou = np.zeros(1035, np.float64)
    intersections, unions, gt_area, pred_area = (np.zeros(1035, np.int64) for _ in range(4))
    reasons, statuses = [], []
    decoded_GT_masks = decoded_prediction_masks = 0
    for i, (target, pred_row, mask_row) in enumerate(zip(targets, predicted_rows, gt_masks)):
        for key, value in target.items():
            assert pred_row[key] == value, (i, key)
        assert mask_row["input_id"] == target["input_id"]
        assert mask_row["image_sha256"] == target["image_sha256"]
        assert gt["query8_xy"][i].tolist() == target["initial_points"]
        assert gt["raw_hw"][i].tolist() == target["raw_hw"]
        assert pred_row["status"] in ["ASSOCIATED", "U"]
        associated[i] = pred_row["status"] == "ASSOCIATED"
        assert pred_row["mask_status"] in ["OK", "U"]
        mask_available[i] = pred_row["mask_status"] == "OK"
        if mask_available[i]:
            assert associated[i]
            assert pred_row["selected_mask_index"] == 0, "Frozen one-mask SAM, no GT selection"
            check(pred_row["mask"])
            with np.load(pred_row["mask"]["path"], allow_pickle=False) as data:
                predicted_mask = data["visible_mask"]
            assert predicted_mask.dtype == np.bool_ and predicted_mask.shape == tuple(target["raw_hw"])
            decoded_prediction_masks += 1
            pred_area[i] = int(predicted_mask.sum())
        else:
            assert pred_row.get("mask") is None
            predicted_mask = None
        reasons.append(pred_row.get("mask_reason"))
        statuses.append(pred_row["mask_status"])
        threshold = mask_row["mask_threshold_strict_gt"]
        assert threshold == (127 if gt["source"][i] == "SYNTHETIC" else 0)
        for key in ["M0", "M4"]:
            check(mask_row[key])
        full = np.asarray(Image.open(mask_row["M0"]["path"])) > threshold
        visible = np.asarray(Image.open(mask_row["M4"]["path"])) > threshold
        decoded_GT_masks += 2
        assert full.shape == visible.shape == tuple(target["raw_hw"])
        assert not (visible & ~full).any()
        assert int(full.sum()) == mask_row["M0_target_area_prior_metadata"]
        assert int(visible.sum()) == mask_row["M4_target_area_prior_metadata"]
        gt_area[i] = int(visible.sum())
        # Independently re-read query GT from native M0/M4; do not alter frozen GT.
        sampled_full = sample_native_mask(full, gt["query8_xy"][i], target["raw_hw"], True)
        sampled_visible = sample_native_mask(visible, gt["query8_xy"][i], target["raw_hw"], True)
        native_valid = sampled_full != 2
        qstate = np.where(native_valid, np.where(sampled_full == 0, 2, sampled_visible), -1).astype(np.int8)
        assert np.array_equal(native_valid, gt["q_pixel_valid8"][i])
        assert np.array_equal(qstate, gt["q_pixel_state8_CVO_invalid"][i])
        assert np.array_equal(np.where(native_valid, sampled_full, -1), gt["q_pixel_M08"][i])
        assert np.array_equal(np.where(native_valid, sampled_visible, -1), gt["q_pixel_M48"][i])
        predictions[i] = sample_native_mask(predicted_mask, gt["query8_xy"][i], target["raw_hw"], mask_available[i])
        iou[i], intersections[i], unions[i] = iou_or_zero(visible, predicted_mask)
    assert np.isin(predictions, [0, 1, 2]).all()
    assert (predictions[~gt["q_pixel_valid8"]] == 2).all()
    summaries = []
    for source in ["ALL", "SYNTHETIC", "YCB_PUBLIC_REAL"]:
        frame = np.ones(1035, bool) if source == "ALL" else gt["source"] == source
        query_frame = np.broadcast_to(frame[:, None], (1035, 8))
        primary = query_frame & gt["original_primary8"]
        qknown = query_frame & gt["q_pixel_valid8"] & np.isin(gt["q_pixel_state8_CVO_invalid"], [0, 1])
        qoutside = query_frame & (gt["q_pixel_state8_CVO_invalid"] == 2)
        qinvalid = query_frame & ~gt["q_pixel_valid8"]
        strict = query_frame & gt["strict535_mask8"]
        cm_reference = confusion_CVU(gt["reference_local_label8"], predictions, primary)
        cm_qpixel = confusion_CVU(gt["q_pixel_state8_CVO_invalid"], predictions, qknown)
        cm_strict = confusion_CVU(gt["reference_local_label8"], predictions, strict)
        available_frame = frame & mask_available
        associated_frame = frame & associated
        cross = np.zeros((3, 4), np.int64)
        qindex = {-1: 0, 0: 1, 1: 2, 2: 3}
        for ref in [-1, 0, 1]:
            for q in [-1, 0, 1, 2]:
                cross[ref + 1, qindex[q]] = int(np.sum(primary & (gt["reference_local_label8"] == ref) & (gt["q_pixel_state8_CVO_invalid"] == q)))
        summaries.append({"source": source, "target_instances": int(frame.sum()),
            "mask_IoU": {"all_rows_N": int(frame.sum()), "all_rows_no_mask0_mean": float(iou[frame].mean()),
                "usable_selected_mask_N": int(available_frame.sum()), "usable_selected_mask_conditional_mean": float(iou[available_frame].mean()) if available_frame.any() else None,
                "detector_associated_N": int(associated_frame.sum()), "detector_associated_with_failedmask0_mean": float(iou[associated_frame].mean()) if associated_frame.any() else None,
                "nomatch_N": int((frame & ~associated).sum()), "associated_missing_mask_N": int((frame & associated & ~mask_available).sum()),
                "mask_hold_rate": float((frame & ~mask_available).sum() / frame.sum()),
                "empty_GT_N": int((frame & (gt_area == 0)).sum()),
                "actual_both_empty_mask_N": int((available_frame & (gt_area == 0) & (pred_area == 0)).sum()),
                "all_rows_quantiles_0_25_50_75_100": np.quantile(iou[frame], [0, .25, .5, .75, 1]).tolist()},
            "REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component": metrics_CVU(cm_reference),
            "SEPARATE_QUERY_PIXEL_CV_given_GT_M0_and_native_valid": metrics_CVU(cm_qpixel),
            "query_pixel_O_separate_not_E_or_S": {"N": int(qoutside.sum()), "pred_CVU_counts": [int((qoutside & (predictions == k)).sum()) for k in [0, 1, 2]]},
            "query_pixel_invalid_separate": {"N": int(qinvalid.sum()), "pred_CVU_counts": [int((qinvalid & (predictions == k)).sum()) for k in [0, 1, 2]]},
            "strict535_same_frozen_IDs": metrics_CVU(cm_strict),
            "PRIMARY_reference_CVU_vs_query_GT_invalid_C_V_O": cross.tolist()})
    all_summary = summaries[0]
    assert all_summary["target_instances"] == 1035
    strict_cm = np.array(all_summary["strict535_same_frozen_IDs"]["CM_rows_GT_CV_cols_pred_CVU"])
    assert strict_cm.sum(1).tolist() == [76, 459]
    public = next(row for row in summaries if row["source"] == "YCB_PUBLIC_REAL")
    assert public["REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component"]["N"] == 2018
    np.savez_compressed(output / "SAVED_MASK_EVALUATION_PER_QUERY.npz", input_id=gt["input_id"], original_id=gt["original_id"],
        source=gt["source"], prediction_CVU=predictions, original_primary8=gt["original_primary8"],
        reference_local_label8=gt["reference_local_label8"], q_pixel_state8=gt["q_pixel_state8_CVO_invalid"],
        q_pixel_valid8=gt["q_pixel_valid8"], strict535_mask8=gt["strict535_mask8"],
        mask_available=mask_available, detector_associated=associated, IoU_allrows_nomask0=iou,
        intersection_pixels=intersections, union_pixels=unions, GT_visible_pixels=gt_area, predicted_visible_pixels=pred_area)
    with (output / "PER_QUERY.csv").open("x", newline="") as stream:
        fields = ["input_id", "original_id", "source", "native_k", "reference_label", "primary", "qGTstate_invalid_C_V_O", "qGT_valid", "pred_CVU", "strict535", "mask_status", "mask_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(1035):
            for k in range(8):
                writer.writerow(dict(input_id=gt["input_id"][i], original_id=gt["original_id"][i], source=gt["source"][i], native_k=k,
                    reference_label=int(gt["reference_local_label8"][i, k]), primary=bool(gt["original_primary8"][i, k]),
                    qGTstate_invalid_C_V_O=int(gt["q_pixel_state8_CVO_invalid"][i, k]), qGT_valid=bool(gt["q_pixel_valid8"][i, k]),
                    pred_CVU=int(predictions[i, k]), strict535=bool(gt["strict535_mask8"][i, k]), mask_status=statuses[i], mask_reason=reasons[i]))
    report = {"completed": True, "PASS": True, "code": bind(__file__), "policy": bind(policy_path),
        "GT_pack_seal": bind(gt_seal_path), "SAM_output_seal": bind(sam_seal_path), "SAM_output_manifest": bind(manifest),
        "same1035_order_and_input_fields": True, "all8280_native_query_GT_states_replayed_exact": True,
        "GT_masks_decoded": decoded_GT_masks, "prediction_masks_decoded": decoded_prediction_masks,
        "benchmarks": summaries, "empty_IoU_convention": policy["native_IoU"],
        "metric_scope": "Precision conditional on known GT C/V subset. End-to-end recall includesUmiss. No qO or GTunknown converted to externalE/selfS. Source reference-local labels and primary unchanged.",
        "mask_quality_used_to_choose_detection_mask_or_threshold": False,
        "GT_input_to_models_or_posthoc_association_mask_choice": False,
        "root_source_pilot_only_no_automatic_current_admission": True,
        "new_model_forward_fit_current_PnP": 0}
    save(output / "SOURCE_MASK_RESULTS.json", report)
    save(output / "EVALUATION_SEAL.json", {"completed": True, "PASS": True,
        "files": [bind(path) for path in sorted(output.iterdir()) if path.is_file()], "prior_artifacts_modified": False})
    print(json.dumps({"PASS": True, "results": bind(output / "SOURCE_MASK_RESULTS.json"),
                      "all1035_IoU": all_summary["mask_IoU"], "strict535": all_summary["strict535_same_frozen_IDs"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--sam-seal", type=Path, required=True)
    parser.add_argument("--sam-seal-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.output_manifest, args.sam_seal, args.sam_seal_sha, args.output)
