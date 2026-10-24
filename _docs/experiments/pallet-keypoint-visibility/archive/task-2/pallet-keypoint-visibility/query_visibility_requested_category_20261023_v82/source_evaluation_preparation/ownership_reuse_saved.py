"""Same V81 exact IoU bins on saved V82 masks; no inference or output choice."""
from collections import Counter, OrderedDict
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

V = Path(__file__).resolve().parents[1]
V81 = V.parent / "query_visibility_grounded_target_20261023_v81"


def load_helpers():
    file = V81 / "saved_target_attribution/analyze_saved_target_ownership.py"
    spec = importlib.util.spec_from_file_location("frozen_V81_attribution_math", file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run(new_eval, output_manifest, output):
    old = load_helpers()
    lineages = old.lines(V81 / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl")
    same_rgb = {row["image_sha256"]: row for row in old.lines(V81 / "saved_target_attribution/SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl")}
    predicted = old.lines(output_manifest)
    with np.load(new_eval / "SAVED_MASK_EVALUATION_PER_QUERY.npz", allow_pickle=False) as data:
        evaluation = {k: data[k] for k in data.files}
    with np.load(V81 / "source_evaluation_preparation/GT_EVALUATION_PACK_DO_NOT_FORWARD.npz", allow_pickle=False) as data:
        query = data["query8_xy"]
    old_attribution = {row["input_id"]: row for row in old.lines(V81 / "saved_target_attribution/ALL1035_TARGET_ATTRIBUTION_GT_DIAGNOSTIC_ONLY.jsonl")}
    strict_C = evaluation["strict535_mask8"] & (evaluation["reference_local_label8"] == 0)
    assert strict_C.sum() == 76
    cache = OrderedDict()
    def mask(binding, threshold):
        key = (binding["path"], binding["sha256"], threshold)
        if key not in cache:
            old.check(binding)
            raw = np.asarray(Image.open(binding["path"]))
            assert raw.ndim == 2
            cache[key] = raw > threshold
            if len(cache) > 32:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return cache[key]
    target_rows, strict_rows = [], []
    for i, (gt, prediction) in enumerate(zip(lineages, predicted)):
        assert gt["input_id"] == prediction["input_id"] == evaluation["input_id"][i]
        others = []
        if gt["source"] == "YCB_PUBLIC_REAL":
            frame = same_rgb[gt["image_sha256"]]
            assert frame["all_official_instance_masks_local"]
            others = [candidate for candidate in frame["annotated_instances"] if candidate["original_id"] != gt["original_id"]]
        row = {"input_id": gt["input_id"], "original_id": gt["original_id"], "source": gt["source"],
            "requested_category": prediction["requested_category"], "association_status": prediction["status"],
            "mask_status": prediction["mask_status"], "other_instance_count": len(others),
            "intended_overlap": None, "best_other_overlap": None, "best_other_original_id": None,
            "selected_box_xyxy_raw": prediction["selected_box_xyxy_raw"],
            "association_iou_initial_querybox_vs_detection": prediction["association_iou"]}
        if prediction["status"] != "ASSOCIATED":
            assert prediction["mask_status"] == "U"
            row["attribution_bin"] = "NO_ASSOCIATION_U"
        elif prediction["mask_status"] != "OK":
            row["attribution_bin"] = "ASSOCIATED_MASK_MISSING_U"
        else:
            old.check(prediction["mask"])
            with np.load(prediction["mask"]["path"], allow_pickle=False) as data:
                actual = data["visible_mask"]
            assert actual.dtype == np.bool_ and actual.shape == tuple(gt["raw_hw"])
            intended = mask(gt["M4"], gt["mask_threshold_strict_gt"])
            own = old.overlap(actual, intended)
            assert own["IoU"] == evaluation["IoU_allrows_nomask0"][i]
            row["intended_overlap"] = own
            best = None
            for other in others:
                visible = mask(other["visible_mask"], other["threshold_strict_gt"])
                pair = old.overlap(actual, visible)
                key = (int(other["instance_id"]), other["visible_mask"]["path"])
                if best is None or old.compare(pair, best[1]) > 0 or (old.compare(pair, best[1]) == 0 and key < best[2]):
                    best = (other, pair, key)
            if best is None:
                row["attribution_bin"] = "NO_OTHER_ANNOTATED_INSTANCE_LOCAL"
            else:
                comparison = old.compare(best[1], own)
                row["attribution_bin"] = "OTHER_IOU_GREATER" if comparison > 0 else "INTENDED_IOU_GREATER" if comparison < 0 else "EXACT_IOU_TIE"
                row["best_other_overlap"], row["best_other_original_id"] = best[1], best[0]["original_id"]
        row["V81_attribution_bin"] = old_attribution[row["input_id"]]["attribution_bin"]
        for k in np.flatnonzero(strict_C[i]):
            nx, ny = np.floor(query[i, k] + .5).astype(int)
            owners = [other["original_id"] for other in others if mask(other["visible_mask"], other["threshold_strict_gt"])[ny, nx]]
            strict_rows.append({"input_id": row["input_id"], "original_id": row["original_id"], "native_k": int(k),
                "prediction_CVU": int(evaluation["prediction_CVU"][i, k]), "attribution_bin": row["attribution_bin"],
                "other_visible_instance_ids_at_query": owners, "annotated_other_visible_at_query": bool(owners),
                "intended_overlap": row["intended_overlap"], "best_other_overlap": row["best_other_overlap"],
                "V81_attribution_bin": row["V81_attribution_bin"]})
        target_rows.append(row)
    assert len(target_rows) == 1035 and len(strict_rows) == 76
    public = [row for row in target_rows if row["source"] == "YCB_PUBLIC_REAL"]
    assert len(public) == 800
    false_visible = [row for row in strict_rows if row["prediction_CVU"] == 1]
    old.write_lines(output / "ALL1035_OWNERSHIP_DIAGNOSTIC_GT_ONLY.jsonl", target_rows)
    old.write_lines(output / "SAME_STRICT_C76_OWNERSHIP_GT_ONLY.jsonl", strict_rows)
    report = {"completed": True, "PASS": True, "same_V81_IoU_math_code": old.bind(V81 / "saved_target_attribution/analyze_saved_target_ownership.py"),
        "source_GT_same_RGB_800_lineage": old.bind(V81 / "saved_target_attribution/SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl"),
        "same_YCB_target_denominator": 800, "all800_bins": dict(Counter(row["attribution_bin"] for row in public)),
        "same_strict_C_denominator": 76, "strict_C76_prediction_CVU": [sum(row["prediction_CVU"] == k for row in strict_rows) for k in [0, 1, 2]],
        "new_strict_C_falsevisible_N": len(false_visible), "new_falsevisible_global_IoU_rank_bins": dict(Counter(row["attribution_bin"] for row in false_visible)),
        "new_falsevisible_other_visible_at_query_N": sum(row["annotated_other_visible_at_query"] for row in false_visible),
        "all800_V81_to_V82_bin_transitions": [dict(V81=k[0], V82=k[1], N=n) for k, n in sorted(Counter((row["V81_attribution_bin"], row["attribution_bin"]) for row in public).items())],
        "해석": "예측mask가어느GTinstance와더겹치는지진단한것이다. detector단독실패로확정하지않으며GTbestother로출력을고르지않았다. U·새falseV수·같은C76분모를모두보존했다.",
        "new_RGB_model_forward_fit_current_PnP": 0}
    old.write(output / "OWNERSHIP_DIAGNOSTIC_KO.json", report)
    return report
