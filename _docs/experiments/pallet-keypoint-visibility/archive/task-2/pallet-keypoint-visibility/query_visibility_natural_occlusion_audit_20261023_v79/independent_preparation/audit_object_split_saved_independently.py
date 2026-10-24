"""Independent source-only recount; imports no evaluator/model/feature code."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parents[1]
P = HERE.parent


def bound(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_bound(record):
    assert bound(record["path"]) == record
    return Path(record["path"])


def main():
    directory = HERE / "object_split_diagnosis"
    protocol_path = directory / "PROTOCOL_BEFORE_CATEGORY_SCORES.json"
    results_path = directory / "OBJECT_SPLIT_SAVED_RESULTS.json"
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(results_path.read_text())
    seal = json.loads((directory / "SEAL.json").read_text())
    for binding in seal["files"]:
        read_bound(binding)
    read_bound(protocol["code"])
    assert result["protocol"] == bound(protocol_path)
    metadata_path = read_bound(protocol["metadata"])
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert len(metadata) == 5203
    ids = np.array([row["id"] for row in metadata])
    source = np.array([row["source"] for row in metadata])
    split = np.array([row["split"] for row in metadata])
    objects = np.array([row.get("object_id", -1) for row in metadata])
    public = source == "YCB_PUBLIC_REAL"
    assert public.sum() == 4200
    original_metadata_path = P / "query_visibility_public_real_20261022_v63/ycb/instances.jsonl"
    original = [json.loads(line) for line in original_metadata_path.read_text().splitlines()]
    by_key = {(int(r["scene_id"]), int(r["frame_id"]), int(r["instance_id"])): r for r in original}
    assert len(original) == len(by_key) == 4200
    for row in metadata:
        if row["source"] != "YCB_PUBLIC_REAL":
            continue
        key = tuple(int(row[k]) for k in ["scene_id", "frame_id", "instance_id"])
        baseline = by_key[key]
        assert row["object_id"] == baseline["object_id"]
        assert row["split"] == baseline["split"]
        assert row["image_sha256_lineage_only"] == baseline["image_sha256"]
        assert row["id"] == "YCB:%06d:%06d:%06d" % key
    train_objects = sorted(set(objects[public & (split == "train")].tolist()))
    val_objects = sorted(set(objects[public & (split == "val")].tolist()))
    assert train_objects == protocol["train_objectIDs"]
    assert val_objects == protocol["val_objectIDs"]
    assert sorted(set(val_objects) - set(train_objects)) == protocol["VAL_only_objectIDs"] == [6, 11]
    gt_path = read_bound(protocol["source_GT"])
    with np.load(gt_path, allow_pickle=False) as data:
        truth = data["label8"]
        assert np.array_equal(data["id"], ids)
        assert np.array_equal(data["source"], source)
        assert np.array_equal(data["split"], split)

    scores = {}
    primary = None
    for name, binding in protocol["source_scores"].items():
        path = read_bound(binding)
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["id"][:5203], ids)
            assert np.array_equal(data["source"][:5203], source)
            assert np.array_equal(data["split"][:5203], split)
            assert np.all(data["domain"][:5203] == 0)
            this_primary = data["primary_loss_mask"][:5203]
            assert not np.any(this_primary & (truth < 0))
            assert not np.any(this_primary & ~data["input_available"][:5203])
            assert not np.any(this_primary & ~data["reference_in_local_square_GT_only"][:5203])
            if primary is None:
                primary = this_primary.copy()
            else:
                assert np.array_equal(primary, this_primary)
            scores[name] = data["support_logit"][:5203]
            assert np.isfinite(scores[name][primary]).all()

    def metrics(logits, frame_filter):
        # Scalar count accumulation independent of producer vectorized confusion.
        cm = [[0, 0], [0, 0]]
        for row in np.flatnonzero(frame_filter):
            for k in range(8):
                if not primary[row, k]:
                    continue
                gt = int(truth[row, k])
                assert gt in [0, 1]
                prediction = int(float(logits[row, k]) >= 0.0)
                cm[gt][prediction] += 1
        cc, cv = cm[0]
        vc, vv = cm[1]
        def ratio(numerator, denominator):
            return None if denominator == 0 else numerator / denominator
        return {"N": cc + cv + vc + vv, "CM_rows_CV_cols_CV": cm,
                "C_recall": ratio(cc, cc + cv), "V_recall": ratio(vv, vc + vv),
                "C_precision": ratio(cc, cc + vc), "V_precision": ratio(vv, vv + cv)}

    per_object = []
    aggregate = []
    group_results = []
    expected_records = {(r["model"], r["split"], r["objectID"]): r for r in result["natural_class_metrics"]}
    assert len(expected_records) == len(result["natural_class_metrics"]) == 126
    groups_expected = {(r["model"], r["VAL_group"]): r for r in result["VAL_seen_unseen"]}
    assert len(groups_expected) == len(result["VAL_seen_unseen"]) == 6
    category_ids = sorted(set(objects[public].tolist()))
    assert category_ids == list(range(1, 22))
    for model, logits in scores.items():
        for partition in ["train", "val"]:
            subcm = np.zeros((2, 2), dtype=np.int64)
            for obj in category_ids:
                filt = public & (split == partition) & (objects == obj)
                item = {"model": model, "split": partition, "objectID": obj,
                        "category_present_in_TRAIN": obj in train_objects,
                        "frames": int(filt.sum()), "metrics": metrics(logits, filt)}
                assert item == expected_records[(model, partition, obj)]
                per_object.append(item)
                subcm += np.array(item["metrics"]["CM_rows_CV_cols_CV"])
            all_filter = public & (split == partition)
            item = {"model": model, "split": partition, "metrics": metrics(logits, all_filter)}
            assert np.array_equal(subcm, item["metrics"]["CM_rows_CV_cols_CV"])
            aggregate.append(item)
        for seen in [True, False]:
            filt = public & (split == "val") & (np.isin(objects, train_objects) == seen)
            key = "TRAIN_SEEN_CATEGORIES" if seen else "TRAIN_UNSEEN_CATEGORIES"
            item = {"model": model, "VAL_group": key, "frames": int(filt.sum()), "metrics": metrics(logits, filt)}
            assert item == groups_expected[(model, key)]
            group_results.append(item)
        a, b = group_results[-2:]
        all_val = aggregate[-1]["metrics"]
        assert a["metrics"]["N"] + b["metrics"]["N"] == all_val["N"] == 2018
        assert np.array_equal(np.array(a["metrics"]["CM_rows_CV_cols_CV"]) + np.array(b["metrics"]["CM_rows_CV_cols_CV"]), all_val["CM_rows_CV_cols_CV"])

    counts = {}
    for partition in ["train", "val"]:
        filt = public & (split == partition)
        rows = [metadata[i] for i in np.flatnonzero(filt)]
        counts[partition] = {"target_instance_rows": len(rows),
            "unique_RGB_frames": len({(r["scene_id"], r["frame_id"]) for r in rows}),
            "scene_ids": sorted({r["scene_id"] for r in rows}),
            "object_target_instance_rows": dict(sorted(Counter(r["object_id"] for r in rows).items())),
            "primary_N": int(primary[filt].sum()),
            "primary_C": int(np.count_nonzero(primary[filt] & (truth[filt] == 0))),
            "primary_V": int(np.count_nonzero(primary[filt] & (truth[filt] == 1)))}
    output = {"PASS": True, "completed": True,
        "audit_code": bound(__file__), "protocol": bound(protocol_path),
        "saved_results": bound(results_path), "original_intake_metadata": bound(original_metadata_path),
        "exact_metrics_reconciled": {"per_object_model_split_records": 126, "VAL_seen_unseen_records": 6,
            "ALL_YCB_aggregate_records": 6, "threshold": "fixed logit >= 0 (pVisible >= .5)"},
        "source_counts": counts, "train_objectIDs": train_objects, "VAL_objectIDs": val_objects,
        "VAL_only_objectIDs": [6, 11], "ALL_YCB": aggregate, "VAL_seen_unseen": group_results,
        "per_object_metrics": per_object,
        "scope_caveats": ["Producer field frames counts target-instance rows, not unique RGB frames. VAL800 target-instance rows share200 RGB frames; TRAIN3400 share800 RGB frames.",
            "TRAIN_SEEN/UNSEEN refers to presence as a natural target object in this frozen local intake; GT object ID is analysis only, never model input or target template.",
            "Category and capture scene co-vary. These stratified observations do not establish causal attribution to object novelty, semantics, ownership, or appearance.",
            "No relabeling, threshold or checkpoint change; no current/private data, RGB, model/feature arrays, new forward, training or PnP."],
        "current_forward_fit_PnP": 0}
    destination = HERE / "independent_audits/OBJECT_SPLIT_SAVED_RESULTS_INDEPENDENT_AUDIT.json"
    with destination.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"PASS": True, "output": bound(destination), "counts": counts,
                      "VAL_CM": {r["model"]: r["metrics"]["CM_rows_CV_cols_CV"] for r in aggregate if r["split"] == "val"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
