"""Metadata/file-availability only; no prediction or GT mask pixel decoding."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    policy_path = R / "DIAGNOSTIC_POLICY_BEFORE_TARGET_ATTRIBUTION_READS.json"
    policy = json.loads(policy_path.read_text())
    assert policy["new_download_RGB_pixels_weights_model_forward_fit_current223_PnP"] == 0
    instances_path = P / "query_visibility_public_real_20261022_v63/ycb/instances.jsonl"
    all_instances = lines(instances_path)
    val = [row for row in all_instances if row["split"] == "val"]
    masks_path = V / "source_evaluation_preparation/GT_MASK_LINEAGE_DO_NOT_FORWARD.jsonl"
    masks = lines(masks_path)
    mask_by_id = {row["original_id"]: row for row in masks}
    frames = defaultdict(list)
    for row in val:
        frames[(int(row["scene_id"]), int(row["frame_id"]))].append(row)
    scene_metadata, scene_paths = {}, []
    output = []
    missing, uncovered, mismatches = [], [], []
    for (scene_id, frame_id), members in sorted(frames.items()):
        scene_root = Path(members[0]["mask_vis"]).parent.parent
        path = scene_root / "scene_gt.json"
        if scene_id not in scene_metadata:
            assert path.is_file()
            scene_metadata[scene_id] = json.loads(path.read_text())
            scene_paths.append(bind(path))
        official = scene_metadata[scene_id][str(frame_id)]
        by_instance = {int(row["instance_id"]): row for row in members}
        assert len(by_instance) == len(members)
        if set(by_instance) != set(range(len(official))):
            uncovered.append({"scene_id": scene_id, "frame_id": frame_id, "local_instance_ids": sorted(by_instance), "official_count": len(official)})
        annotated = []
        for instance_id, gt in enumerate(official):
            visible_path = scene_root / "mask_visib" / ("%06d_%06d.png" % (frame_id, instance_id))
            if not visible_path.is_file():
                missing.append(str(visible_path))
            row = by_instance.get(instance_id)
            if row is None:
                continue
            identifier = "YCB:%06d:%06d:%06d" % (scene_id, frame_id, instance_id)
            local = mask_by_id[identifier]
            if row["object_id"] != gt["obj_id"] or Path(row["mask_vis"]) != visible_path or local["M4"]["path"] != str(visible_path):
                mismatches.append(identifier)
            assert local["image_sha256"] == row["image_sha256"]
            annotated.append({"original_id": identifier, "scene_id": scene_id, "frame_id": frame_id,
                "instance_id": instance_id, "object_id_GT_diagnostic_only": int(gt["obj_id"]),
                "visible_mask": local["M4"], "full_mask": local["M0"], "threshold_strict_gt": 0,
                "visible_area_prior_metadata": local["M4_target_area_prior_metadata"],
                "raw_hw": local["raw_hw"], "input_id": local["input_id"]})
        output.append({"image_sha256": members[0]["image_sha256"], "scene_id": scene_id,
            "frame_id": frame_id, "official_instance_count": len(official), "local_annotation_count": len(annotated),
            "annotated_instances": annotated, "all_official_instance_masks_local": not any(str(scene_root) in name and ("%06d_" % frame_id) in name for name in missing) and len(annotated) == len(official)})
    assert len(val) == 800 and len(frames) == 200
    with (R / "SAME_RGB_ANNOTATED_MASK_INDEX_GT_ONLY.jsonl").open("x") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write(R / "SAME_RGB_GT_AVAILABILITY.json", {"completed": True, "PASS": not missing and not uncovered and not mismatches,
        "코드": bind(__file__), "정책": bind(policy_path), "YCB_VAL_targets": 800, "YCB_VAL_unique_RGB": 200,
        "official_instance_count_distribution_per_RGB": dict(Counter(row["official_instance_count"] for row in output)),
        "all_official_same_RGB_instance_M4_files_local": not missing and not uncovered,
        "official_total_instance_masks": sum(row["official_instance_count"] for row in output),
        "missing_files": missing, "not_in_local_intake_metadata": uncovered, "metadata_path_object_mismatches": mismatches,
        "SYNTH_target_rows": sum(row["source"] == "SYNTHETIC" for row in masks),
        "SYNTH_other_instance_status": "이자료의SYNTH GT는target별M0/M4만제공하며다른instance의M4를찾았다고주장하지않음. NO_OTHER_ANNOTATED_INSTANCE_LOCAL이지실제로다른물체가없다는의미가아님.",
        "source_bindings": [bind(instances_path), bind(masks_path), *scene_paths],
        "mask_pixels_RGB_prediction_outputs_model_forward_fit_current_PnP_reads": 0,
        "new_downloads": 0})
    print(json.dumps({"PASS": not missing and not uncovered and not mismatches,
        "availability": bind(R / "SAME_RGB_GT_AVAILABILITY.json"), "missing": len(missing), "uncovered": len(uncovered)}))


if __name__ == "__main__":
    main()
