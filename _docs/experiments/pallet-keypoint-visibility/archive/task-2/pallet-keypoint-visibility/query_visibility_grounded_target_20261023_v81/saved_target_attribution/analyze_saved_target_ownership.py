"""Post-saved diagnostic IoU comparison. Never selects or changes model output."""
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
V = R.parent


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(binding):
    assert bind(binding["path"])["sha256"] == binding["sha256"], binding["path"]


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def overlap(predicted, truth):
    intersection = int(np.count_nonzero(predicted & truth))
    union = int(np.count_nonzero(predicted | truth))
    return {"intersection": intersection, "union": union, "IoU": intersection / union if union else 1.}


def rational(pair):
    return (pair["intersection"], pair["union"]) if pair["union"] else (1, 1)


def compare(a, b):
    an, ad = rational(a); bn, bd = rational(b)
    return (an * bd > bn * ad) - (an * bd < bn * ad)


def bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def rectangle_iou(a, b):
    if a is None or b is None:
        return None
    width = max(0., min(float(a[2]), b[2]) - max(float(a[0]), b[0]))
    height = max(0., min(float(a[3]), b[3]) - max(float(a[1]), b[1]))
    area = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) + (b[2] - b[0]) * (b[3] - b[1]) - width * height
    return width * height / area if area else None


def main():
    policy_path = R / "DIAGNOSTIC_POLICY_BEFORE_TARGET_ATTRIBUTION_READS.json"
    policy = json.loads(policy_path.read_text())
    for binding in policy["bindings"].values():
        check(binding)
    availability = json.loads((R / "SAME_RGB_GT_AVAILABILITY.json").read_text())
    assert availability["completed"] and availability["PASS"]
    assert availability["all_official_same_RGB_instance_M4_files_local"]
    same_rgb = {row["image_sha256"]: row for row in lines(R / "SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl")}
    gt_rows = lines(V / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl")
    masks = lines(V / "segmentation/outputs/OUTPUT_MASKS.jsonl")
    rgb_detections = {row["image_sha256"]: row for row in lines(V / "detector/outputs/RGB_DETECTIONS.jsonl")}
    with np.load(V / "source_evaluation/GROUNDED_SAM21_FIXED/SAVED_MASK_EVALUATION_PER_QUERY.npz", allow_pickle=False) as data:
        evaluation = {key: data[key] for key in data.files}
    with np.load(V / "source_evaluation_preparation/GT_EVALUATION_PACK_DO_NOT_FORWARD.npz", allow_pickle=False) as data:
        query, side = data["query8_xy"], data["L"]
    assert len(gt_rows) == len(masks) == 1035
    strict = evaluation["strict535_mask8"]
    strict_C = strict & (evaluation["reference_local_label8"] == 0)
    false_visible = strict_C & (evaluation["prediction_CVU"] == 1)
    strict_U = strict_C & (evaluation["prediction_CVU"] == 2)
    assert strict.sum() == 535 and strict_C.sum() == 76 and false_visible.sum() == 56 and strict_U.sum() == 5
    cache, decoded_gt, decoded_predictions = OrderedDict(), 0, 0

    def read_gt(binding, threshold):
        nonlocal decoded_gt
        key = (binding["path"], binding["sha256"], threshold)
        if key not in cache:
            check(binding)
            raw = np.asarray(Image.open(binding["path"]))
            assert raw.ndim == 2
            cache[key] = raw > threshold
            decoded_gt += 1
            if len(cache) > 32:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return cache[key]

    summaries, false_queries, held_queries = [], [], []
    for i, (gt, mask_row) in enumerate(zip(gt_rows, masks)):
        assert gt["input_id"] == mask_row["input_id"] == evaluation["input_id"][i]
        assert gt["original_id"] == evaluation["original_id"][i]
        assert gt["image_sha256"] == mask_row["image_sha256"]
        record = {"input_id": gt["input_id"], "original_id": gt["original_id"], "source": gt["source"],
            "image_sha256": gt["image_sha256"], "mask_status": mask_row["mask_status"],
            "association_status": mask_row["status"], "association_reason": mask_row["reason"],
            "association_iou_initial_querybox_vs_detection": mask_row["association_iou"],
            "selected_phrase_diagnostic_not_a_cause_label": None,
            "selected_box_xyxy_raw": mask_row["selected_box_xyxy_raw"], "selected_box_xyxy_prompt": mask_row["selected_box_xyxy_prompt"],
            "intended_overlap": None, "best_other_overlap": None, "best_other_original_id": None,
            "best_other_instance_id": None, "other_instance_count": None,
            "detbox_vs_intended_visible_bbox_IoU": None, "detbox_vs_bestother_visible_bbox_IoU": None,
            "predicted_mask_pixels": None, "intended_mask_pixels": gt["M4_target_area_prior_metadata"],
            "predicted_intended_pixel_precision": None, "intended_visible_recall": None}
        if mask_row["status"] != "ASSOCIATED":
            assert mask_row["mask_status"] == "U" and mask_row["mask"] is None
            record["attribution_bin"] = "NO_ASSOCIATION_U"
        elif mask_row["mask_status"] != "OK":
            record["attribution_bin"] = "ASSOCIATED_MASK_MISSING_U"
        else:
            check(mask_row["mask"])
            with np.load(mask_row["mask"]["path"], allow_pickle=False) as data:
                predicted = data["visible_mask"]
            assert predicted.dtype == np.bool_ and predicted.shape == tuple(gt["raw_hw"])
            decoded_predictions += 1
            intended = read_gt(gt["M4"], gt["mask_threshold_strict_gt"])
            assert intended.shape == predicted.shape and int(intended.sum()) == gt["M4_target_area_prior_metadata"]
            own = overlap(predicted, intended)
            assert own["IoU"] == evaluation["IoU_allrows_nomask0"][i]
            record["intended_overlap"] = own
            record["predicted_mask_pixels"] = int(predicted.sum())
            record["predicted_intended_pixel_precision"] = own["intersection"] / int(predicted.sum()) if predicted.any() else None
            record["intended_visible_recall"] = own["intersection"] / int(intended.sum()) if intended.any() else None
            record["detbox_vs_intended_visible_bbox_IoU"] = rectangle_iou(mask_row["selected_box_xyxy_raw"], bbox(intended))
            det = rgb_detections[gt["image_sha256"]]
            check(det["detections"])
            with np.load(det["detections"]["path"], allow_pickle=False) as data:
                selected = int(mask_row["selected_detection_index"])
                record["selected_phrase_diagnostic_not_a_cause_label"] = str(data["text_labels"][selected])
                assert data["boxes_xyxy"][selected].tolist() == mask_row["selected_box_xyxy_raw"]
            other_masks = []
            if gt["source"] == "YCB_PUBLIC_REAL":
                frame = same_rgb[gt["image_sha256"]]
                assert frame["all_official_instance_masks_local"]
                others = [candidate for candidate in frame["annotated_instances"] if candidate["original_id"] != gt["original_id"]]
                record["intended_instance_id"] = next(candidate["instance_id"] for candidate in frame["annotated_instances"] if candidate["original_id"] == gt["original_id"])
                record["intended_object_id_GT_diagnostic_only"] = next(candidate["object_id_GT_diagnostic_only"] for candidate in frame["annotated_instances"] if candidate["original_id"] == gt["original_id"])
            else:
                others = []
            record["other_instance_count"] = len(others)
            best = None
            for other in others:
                visible = read_gt(other["visible_mask"], other["threshold_strict_gt"])
                assert visible.shape == intended.shape
                candidate_overlap = overlap(predicted, visible)
                other_masks.append((other, visible, candidate_overlap))
                candidate_key = (int(other["instance_id"]), other["visible_mask"]["path"])
                if best is None or compare(candidate_overlap, best[2]) > 0 or (compare(candidate_overlap, best[2]) == 0 and candidate_key < best[3]):
                    best = (other, visible, candidate_overlap, candidate_key)
            if best is None:
                record["attribution_bin"] = "NO_OTHER_ANNOTATED_INSTANCE_LOCAL"
            else:
                record["best_other_overlap"] = best[2]
                record["best_other_original_id"] = best[0]["original_id"]
                record["best_other_instance_id"] = best[0]["instance_id"]
                record["best_other_object_id_GT_diagnostic_only"] = best[0]["object_id_GT_diagnostic_only"]
                record["best_other_mask_binding"] = best[0]["visible_mask"]
                record["detbox_vs_bestother_visible_bbox_IoU"] = rectangle_iou(mask_row["selected_box_xyxy_raw"], bbox(best[1]))
                comparison = compare(best[2], own)
                record["attribution_bin"] = "OTHER_IOU_GREATER" if comparison > 0 else "INTENDED_IOU_GREATER" if comparison < 0 else "EXACT_IOU_TIE"
            for k in np.flatnonzero(false_visible[i]):
                x, y = query[i, k]
                nx, ny = int(np.floor(x + .5)), int(np.floor(y + .5))
                assert predicted[ny, nx] and not intended[ny, nx]
                at_q = [other["original_id"] for other, visible, _ in other_masks if visible[ny, nx]]
                false_queries.append({"input_id": gt["input_id"], "original_id": gt["original_id"], "native_k": int(k),
                    "query_xy": query[i, k].tolist(), "L": float(side[i]), "attribution_bin": record["attribution_bin"],
                    "other_visible_instance_ids_at_query": at_q, "annotated_other_visible_at_query": bool(at_q),
                    "intended_overlap": own, "best_other_overlap": record["best_other_overlap"],
                    "best_other_original_id": record["best_other_original_id"], "best_other_mask_binding": record.get("best_other_mask_binding"),
                    "intended_mask_binding": gt["M4"], "prediction_mask_binding": mask_row["mask"],
                    "selected_phrase": record["selected_phrase_diagnostic_not_a_cause_label"],
                    "association_iou_initial_querybox_vs_detection": record["association_iou_initial_querybox_vs_detection"],
                    "detbox_vs_intended_visible_bbox_IoU": record["detbox_vs_intended_visible_bbox_IoU"],
                    "detbox_vs_bestother_visible_bbox_IoU": record["detbox_vs_bestother_visible_bbox_IoU"]})
        for k in np.flatnonzero(strict_U[i]):
            held_queries.append({"input_id": gt["input_id"], "original_id": gt["original_id"], "native_k": int(k),
                "query_xy": query[i, k].tolist(), "L": float(side[i]), "attribution_bin": record["attribution_bin"],
                "mask_status": mask_row["mask_status"], "association_reason": mask_row["reason"],
                "intended_mask_binding": gt["M4"], "prediction_mask_binding": None})
        summaries.append(record)
    assert len(false_queries) == 56 and len(held_queries) == 5
    salt = policy["fixed_examples"]["salt"]
    example_candidates = defaultdict(list)
    for row in false_queries + held_queries:
        example_candidates[row["attribution_bin"]].append(row)
    examples = []
    for bin_name in policy["attribution_bins"]:
        rows = example_candidates[bin_name]
        if not rows:
            continue
        def rank(row):
            payload = salt + "\n" + row["input_id"] + "\n" + str(row["native_k"])
            return hashlib.sha256(payload.encode()).hexdigest(), row["input_id"], row["native_k"]
        fixed = min(rows, key=rank)
        examples.append(dict(fixed, selection_SHA256=rank(fixed)[0], bin_population=len(rows)))
    counts = []
    for source in ["ALL", "SYNTHETIC", "YCB_PUBLIC_REAL"]:
        subset = [row for row in summaries if source == "ALL" or row["source"] == source]
        counts.append({"source": source, "target_rows": len(subset), "attribution_bins": dict(Counter(row["attribution_bin"] for row in subset))})
    false_bins = Counter(row["attribution_bin"] for row in false_queries)
    local_table = Counter((row["attribution_bin"], row["annotated_other_visible_at_query"]) for row in false_queries)
    write_lines(R / "ALL1035_TARGET_ATTRIBUTION_GT_DIAGNOSTIC_ONLY.jsonl", summaries)
    write_lines(R / "STRICT_C_FALSE_VISIBLE_56_GT_DIAGNOSTIC_ONLY.jsonl", false_queries)
    write_lines(R / "STRICT_C_HELD_U_5_GT_DIAGNOSTIC_ONLY.jsonl", held_queries)
    write(R / "FIXED_MIN_SHA_EXAMPLES_BEFORE_VISUALIZATION.json", {"frozen_before_new_visualization": True,
        "policy": bind(policy_path), "salt": salt, "examples": examples,
        "bins_without_fixed_strict_examples": [name for name in policy["attribution_bins"] if name not in example_candidates or not example_candidates[name]]})
    write(R / "TARGET_OWNERSHIP_ATTRIBUTION_RESULTS_KO.json", {"completed": True, "PASS": True,
        "정책": bind(policy_path), "availability": bind(R / "SAME_RGB_GT_AVAILABILITY.json"), "code": bind(__file__),
        "scope": "동일원래sourceVAL1035target, 고정strict535. sourceGT는조건부proxy. GT IoU최적상대는진단값일뿐출력selector가아님.",
        "전체대상수": counts, "고정strict_C76": {"pred_C": 15, "pred_V": 56, "pred_U": 5},
        "strict_C_falsevisible56_global_IoU_rank_bins": dict(false_bins),
        "strict_C_falsevisible56_unique_targets": len({row["input_id"] for row in false_queries}),
        "strict_C_falsevisible56_globalrank_by_other_visible_at_q": [dict(attribution_bin=key[0], annotated_other_visible_at_query=key[1], N=value) for key, value in sorted(local_table.items())],
        "해석한계": ["다른GT instance IoU가더크면오대상마스크와일관된증거이지만검출association과SAM중단일원인을확정하지않음.",
            "intended IoU가더크더라도가려진q를다른instance가차지하면부분적으로과포함된마스크와일관될수있음. global순위만으로정상소유권을보증하지않음.",
            "IoU가작거나0인동률은그자체로대상확정증거가아님. Other visibleGT가q를덮지않으면미표기물체/GT등록오차/배경등의가능성을남기며원라벨을고치지않음.",
            "선택phrase불일치를오대상원인으로사용하지않았음. 고정사례의숫자는원인확정이나전체성공률로과대해석하지않음.",
            "SYNTH no-other-local-GT는다른물체가실제로없다는의미가아님. 인간확인DES나팔레트native8gold로승격하지않음."],
        "decoded_saved_prediction_masks": decoded_predictions, "decoded_GT_visible_masks_with_LRU": decoded_gt,
        "prediction_label_query_threshold_model_or_prior_seal_changes": False,
        "RGBpixels_new_download_model_forward_fit_current223_PnP": 0})
    print(json.dumps({"PASS": True, "results": bind(R / "TARGET_OWNERSHIP_ATTRIBUTION_RESULTS_KO.json"),
        "targets": counts, "strict_falseV_bins": dict(false_bins), "local_q": [dict(bin=k[0], other_at_q=k[1], N=n) for k,n in sorted(local_table.items())]}))


if __name__ == "__main__":
    main()
