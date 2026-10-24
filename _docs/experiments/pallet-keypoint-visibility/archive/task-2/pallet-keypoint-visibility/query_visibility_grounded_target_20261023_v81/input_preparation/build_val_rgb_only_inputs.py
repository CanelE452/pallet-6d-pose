"""Metadata/coordinate-only source VAL manifest, no RGB/GT/model imports."""
from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import json
import math
import numpy as np

R = Path(__file__).resolve().parent
P = R.parents[1]
V76 = P / "query_visibility_reference_local_support_20261023_v76"


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def check(record):
    assert bind(record["path"]) == record


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def clean(value):
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    policy_path = R / "SELECTION_POLICY_BEFORE_INPUT_READS.json"
    policy = read(policy_path)
    assert policy["frozen_before_new_manifest_cache_or_lineage_reads"]
    check(policy["bindings_to_original_input_lock"])
    lock = read(policy["bindings_to_original_input_lock"]["path"])
    write(R / "MANIFEST_BUILD_LOCK_BEFORE_GENERATION.json", {
        "frozen": True, "code": bind(__file__), "selection_policy": bind(policy_path),
        "source_input_lock": policy["bindings_to_original_input_lock"],
        "split_alias_interpretation": "Original literal validation and val are retained as the same frozen VAL partition; source spelling preserved in provenance.",
        "only_loaded_NPZ_arrays": ["query8_xy", "L", "query_box_xyxy"],
        "RGB_sha_source": "Prior sealed lineage/image inventory; raw image bodies not rehashed or decoded by this metadata preparation.",
        "caption_execution": "Root separate frozen forward protocol; same fixed all21YCB+pallet caption for every RGB, no per-image category supplied here.",
        "source_GT_RGB_weights_feature_vectors_forward_fit_current_PnP": 0})
    public_query_path = V76 / "public_query_emulation/INPUTS_RGB_ONLY.jsonl"
    public_query_binding = bind(public_query_path)
    public_queries = lines(public_query_path)
    public_inventory_summary = read(V76 / "public_input_preparation/PUBLIC_IMAGE_AND_P3_INVENTORY.json")
    check(public_inventory_summary["manifest"])
    public_inventory = {row["image_sha256"]: row for row in lines(public_inventory_summary["manifest"]["path"])}
    target_rows, provenance = [], []
    caches, input_bindings = OrderedDict(), []
    counts, q_nonfinite, box_nonfinite = Counter(), 0, 0
    raw_hw_counts = Counter()
    for pack in lock["packs"]:
        for key in ["manifest", "seal", "lineage"]:
            check(pack[key])
            input_bindings.append(pack[key])
        manifest = read(pack["manifest"]["path"])
        lineage = lines(pack["lineage"]["path"])
        assert manifest["completed"] and len(lineage) == manifest["frame_count"]
        selected = {i for i, row in enumerate(lineage) if row["split"].lower() in ["val", "validation"]}
        for chunk in manifest["chunks"]:
            start, end = int(chunk["start"]), int(chunk["end"])
            eligible = sorted(selected.intersection(range(start, end)))
            if not eligible:
                continue
            check(chunk["file"])
            caches[chunk["file"]["path"]] = chunk["file"]
            with np.load(chunk["file"]["path"], allow_pickle=False) as data:
                query, sides, boxes = (data[key] for key in ["query8_xy", "L", "query_box_xyxy"])
                assert query.shape == (end - start, 8, 2) and boxes.shape == (end - start, 4)
                for i in eligible:
                    row = lineage[i]
                    local = i - start
                    q, side, box = query[local], float(sides[local]), boxes[local]
                    assert np.isfinite(side) and side > 0
                    q_nonfinite += int((~np.isfinite(q)).sum())
                    box_nonfinite += int((~np.isfinite(box)).sum())
                    if pack["source"] == "SYNTHETIC":
                        image = row["raw_RGB"]["path"]
                        image_sha = row["raw_RGB"]["sha256"]
                        hw = row["raw_RGB"]["raw_hw"]
                        assert side == row["input_L"]
                        source_input = {"lineage": pack["lineage"], "line_index": i}
                    else:
                        assert pack["source"] == "YCB_PUBLIC_REAL"
                        original = public_queries[i]
                        assert original["id"] == row["id"] and original["split"] == row["split"]
                        assert np.array_equal(np.asarray(original["initial_points"], float), q, equal_nan=True)
                        assert original["roi_side"] == side and row["roi_side"] == side
                        assert np.array_equal(np.asarray(original["box"], float), box, equal_nan=True)
                        image, image_sha, hw = original["image"], original["image_sha256"], original["raw_hw"]
                        assert image == row["image"] and image_sha == row["image_sha256"]
                        inventory = public_inventory[image_sha]
                        assert inventory["image"] == image and inventory["raw_hw"] == hw and inventory["split"] == "val"
                        source_input = {"manifest": public_query_binding, "line_index": i}
                    assert len(hw) == 2 and all(int(d) == d and d > 0 for d in hw)
                    input_id = "V81_INPUT_%06d" % len(target_rows)
                    clean_q, clean_box = clean(q.tolist()), clean(box.tolist())
                    entry = {"input_id": input_id, "image": image, "image_sha256": image_sha,
                             "raw_hw": hw, "initial_points": clean_q, "roi_side": side,
                             "query_box_xyxy": clean_box}
                    assert set(entry) == set(policy["model_input_whitelist"])
                    # JSON nulls round-trip to NaN only; all finite native coordinates remain exact.
                    assert np.array_equal(np.asarray(clean_q, float), q, equal_nan=True)
                    assert np.array_equal(np.asarray(clean_box, float), box, equal_nan=True)
                    target_rows.append(entry)
                    provenance.append({"input_id": input_id, "original_row_index": i if pack["source"] == "SYNTHETIC" else 1003 + i,
                        "source_cache_row_index": i, "source": pack["source"], "original_id": row["id"],
                        "source_partition_literal": row["split"], "normalized_partition": "val",
                        "source_input_record": source_input, "lineage_record": {"file": pack["lineage"], "line_index": i},
                        "source_cache": {"file": chunk["file"], "array_row_index": local},
                        "native_query_L_qbox_copy_exact": True, "nonfinite_coordinate_serialization": "null means original nonfinite, never relocated",
                        "public_query_mode": "GT-derived query emulation using TRAIN-only synthetic initializer error bank" if pack["source"] == "YCB_PUBLIC_REAL" else "original frozen synthetic initializer query"})
                    counts[pack["source"]] += 1
                    raw_hw_counts[str(tuple(hw))] += 1
    assert counts == {"SYNTHETIC": 235, "YCB_PUBLIC_REAL": 800}
    assert len(target_rows) == len(provenance) == 1035
    rgb_unique = OrderedDict()
    for row in target_rows:
        key = row["image_sha256"]
        if key in rgb_unique:
            assert (row["image"], row["raw_hw"]) == (rgb_unique[key]["image"], rgb_unique[key]["raw_hw"])
        else:
            rgb_unique[key] = {"input_id": "V81_RGB_%06d" % len(rgb_unique), "image": row["image"],
                               "image_sha256": key, "raw_hw": row["raw_hw"]}
    unique_by_source = {source: len({row["image_sha256"] for row, prov in zip(target_rows, provenance) if prov["source"] == source}) for source in counts}
    assert unique_by_source == {"SYNTHETIC": 235, "YCB_PUBLIC_REAL": 200}
    assert len(rgb_unique) == 435
    write_lines(R / "INPUTS_RGB_ONLY.jsonl", target_rows)
    write_lines(R / "UNIQUE_RGB_INPUTS_ONLY.jsonl", list(rgb_unique.values()))
    write_lines(R / "PROVENANCE_DO_NOT_FORWARD.jsonl", provenance)
    report = {"completed": True, "PASS": True, "target_instances": 1035, "query_slots": 8280,
        "targets_by_source": dict(counts), "unique_RGB": len(rgb_unique), "unique_RGB_by_source": unique_by_source,
        "raw_HW_target_counts": dict(raw_hw_counts), "null_qbox_targets": sum(row["query_box_xyxy"] is None or any(x is None for x in row["query_box_xyxy"]) for row in target_rows),
        "nonfinite_query_coordinate_values": q_nonfinite, "nonfinite_box_coordinate_values": box_nonfinite,
        "all_VAL_rows_and_all8queries_retained": True, "all_native_query_L_qbox_coordinates_exact": True,
        "model_input_field_whitelist": policy["model_input_whitelist"], "GT_objectIDs_labels_masks_references_scores_in_model_input": False,
        "RGB_decode_freshRGB_body_hash_P3_vectors_weights_forward_fit_current_PnP": 0,
        "source_RGB_hash_evidence": "Prior immutable input lineage/inventory values; runner must verify actual RGB bytes before forward.",
        "incidental_schema_inspection": "Before builder generation, schema inspection materialized metadata arrays including context42 solely for shape/dtype; no values used in selection or manifests, no RGB/P3 feature vectors decoded. Builder loads only query8_xy/L/query_box_xyxy.",
        "public_query_caveat": policy["public_query_provenance"], "caption_and_model_execution_pending": True,
        "source_bindings": input_bindings + [public_query_binding, public_inventory_summary["manifest"]],
        "selected_cache_bindings": list(caches.values()), "code": bind(__file__), "policy": bind(policy_path)}
    write(R / "INPUT_MANIFEST_BUILD_RESULTS.json", report)
    files = [R / filename for filename in ["SELECTION_POLICY_BEFORE_INPUT_READS.json", "MANIFEST_BUILD_LOCK_BEFORE_GENERATION.json", "build_val_rgb_only_inputs.py", "INPUTS_RGB_ONLY.jsonl", "UNIQUE_RGB_INPUTS_ONLY.jsonl", "PROVENANCE_DO_NOT_FORWARD.jsonl", "INPUT_MANIFEST_BUILD_RESULTS.json"]]
    write(R / "SEAL.json", {"completed": True, "PASS": True, "files": [bind(path) for path in files],
        "no_model_or_RGB_decode": True, "current_PnP": 0})
    print(json.dumps({"PASS": True, "seal": bind(R / "SEAL.json"), "targets": len(target_rows), "RGB": len(rgb_unique), "targets_by_source": dict(counts)}))


if __name__ == "__main__":
    main()
