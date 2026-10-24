"""Additional native M0/official box evidence on the same fixed56 errors."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from analyze_saved_target_ownership import bbox, rectangle_iou, bind, lines, write, write_lines

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent


def main():
    protocol_path = R / "FULL_BBOX_EVIDENCE_PROTOCOL_BEFORE_COMPUTE.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["frozen_before_new_full_bbox_diagnostic_values"]
    errors = lines(R / "STRICT_C_FALSE_VISIBLE_56_GT_DIAGNOSTIC_ONLY.jsonl")
    all_targets = {row["input_id"]: row for row in lines(R / "ALL1035_TARGET_ATTRIBUTION_GT_DIAGNOSTIC_ONLY.jsonl")}
    gt_lineage = {row["original_id"]: row for row in lines(V / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl")}
    instances_path = P / "query_visibility_public_real_20261022_v63/ycb/instances.jsonl"
    instances = {"YCB:%06d:%06d:%06d" % (row["scene_id"], row["frame_id"], row["instance_id"]): row for row in lines(instances_path) if row["split"] == "val"}
    cache = {}
    decoded = 0

    def full_bbox(identifier):
        nonlocal decoded
        if identifier not in cache:
            b = gt_lineage[identifier]["M0"]
            assert bind(b["path"])["sha256"] == b["sha256"]
            mask = np.asarray(Image.open(b["path"])) > 0
            assert mask.shape == tuple(gt_lineage[identifier]["raw_hw"])
            cache[identifier] = bbox(mask)
            decoded += 1
        return cache[identifier]

    def xyxy(rectangle):
        x, y, width, height = rectangle
        return [x, y, x + width, y + height]

    detail = []
    for row in errors:
        own, other = row["original_id"], row["best_other_original_id"]
        predicted_box = all_targets[row["input_id"]]["selected_box_xyxy_raw"]
        own_native, other_native = full_bbox(own), full_bbox(other)
        own_official, other_official = xyxy(instances[own]["bbox_obj"]), xyxy(instances[other]["bbox_obj"])
        own_native_iou, other_native_iou = rectangle_iou(predicted_box, own_native), rectangle_iou(predicted_box, other_native)
        own_official_iou, other_official_iou = rectangle_iou(predicted_box, own_official), rectangle_iou(predicted_box, other_official)
        detail.append({"input_id": row["input_id"], "original_id": own, "native_k": row["native_k"],
            "best_other_original_id": other, "selected_phrase": row["selected_phrase"],
            "selected_raw_box": predicted_box,
            "intended_native_M0_bbox": own_native, "best_other_native_M0_bbox": other_native,
            "selected_box_vs_intended_native_M0_bbox_IoU": own_native_iou,
            "selected_box_vs_best_other_native_M0_bbox_IoU": other_native_iou,
            "intended_official_bbox_obj_xyxy": own_official, "best_other_official_bbox_obj_xyxy": other_official,
            "selected_box_vs_intended_official_bbox_obj_IoU": own_official_iou,
            "selected_box_vs_best_other_official_bbox_obj_IoU": other_official_iou})
    assert len(detail) == 56
    def compare_bins(a, b):
        return dict(Counter("OTHER_GREATER" if row[b] > row[a] else "INTENDED_GREATER" if row[a] > row[b] else "EXACT_TIE" for row in detail))
    report = {"completed": True, "PASS": True, "code": bind(__file__), "protocol": bind(protocol_path),
        "fixed_strict_C_false_visible_points": 56, "unique_target_instances": len({row["input_id"] for row in detail}),
        "native_M0_bbox_comparison": compare_bins("selected_box_vs_intended_native_M0_bbox_IoU", "selected_box_vs_best_other_native_M0_bbox_IoU"),
        "official_bbox_obj_comparison": compare_bins("selected_box_vs_intended_official_bbox_obj_IoU", "selected_box_vs_best_other_official_bbox_obj_IoU"),
        "intended_M4_IoU_quantiles": np.quantile([row["intended_overlap"]["IoU"] for row in errors], [0, .25, .5, .75, 1]).tolist(),
        "best_other_M4_IoU_quantiles": np.quantile([row["best_other_overlap"]["IoU"] for row in errors], [0, .25, .5, .75, 1]).tolist(),
        "selected_phrase_counts_diagnostic_only": dict(Counter(row["selected_phrase"] for row in errors)),
        "추가설명": "다른instance와예측mask/선택box가더일치한다는증거를함께제공한다. 이작업은actualRGB를읽지않았으며detector단독실패로확정하거나phrase불일치를원인으로사용하지않는다.",
        "source_metadata": bind(instances_path), "extra_native_M0_masks_decoded": decoded,
        "model_output_label_threshold_or_original_bins_changed": False,
        "RGB_newdownload_model_forward_fit_current_PnP": 0}
    write_lines(R / "FIXED56_NATIVE_AND_OFFICIAL_BOX_CONTEXT_GT_ONLY.jsonl", detail)
    write(R / "FULL_BBOX_AND_PHRASE_CONTEXT_KO.json", report)
    print(json.dumps({"PASS": True, "report": bind(R / "FULL_BBOX_AND_PHRASE_CONTEXT_KO.json"),
        "native_M0": report["native_M0_bbox_comparison"], "official": report["official_bbox_obj_comparison"]}))


if __name__ == "__main__":
    main()
