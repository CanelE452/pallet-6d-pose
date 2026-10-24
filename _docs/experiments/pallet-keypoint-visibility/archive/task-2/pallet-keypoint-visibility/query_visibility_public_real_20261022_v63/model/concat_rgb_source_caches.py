"""Combine frozen source-only SYNTH+YCB RGB caches with split/exclusion checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    start = time.monotonic()
    root, out_root = Path(args.source_dir), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = {split: out_root / f"MIXED_{split}_RGB.npz" for split in ("TRAIN", "VALIDATION")}
    provenance = out_root / "MIXED_RGB_SOURCE_PROVENANCE.json"
    if provenance.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite mixed RGB caches or provenance")
    denied = set(json.loads(Path(args.eval_denylist).read_text())["image_sha256"])
    if not denied:
        raise ValueError("empty evaluation-image denylist")
    inputs, combined, source_provenance = {}, {}, {}
    for split in ("TRAIN", "VALIDATION"):
        sources = []
        input_records = []
        for source in ("SYNTH", "YCB"):
            path = root / f"{source}_{split}_RGB.npz"
            with np.load(path, allow_pickle=False) as cache:
                data = {key: cache[key] for key in cache.files}
            count = len(data["x"])
            if set(data["image_sha256"].astype(str)) & denied:
                raise ValueError(f"evaluation image hash found: {path}")
            if data["rgb"].shape != (count, 3, 65, 65) or data["rgb"].dtype != np.uint8:
                raise ValueError(f"invalid raw RGB shape/dtype: {path}")
            if data["context"].shape != (count, 64):
                raise ValueError(f"invalid target context shape: {path}")
            if not np.isin(data["y"], (0, 1)).all():
                raise ValueError(f"invalid binary labels: {path}")
            input_records.append({"source": source, "path": str(path.resolve()), "sha256": sha256(path),
                                  "queries": count, "original_fields": sorted(data),
                                  "original_y_dtype": str(data["y"].dtype)})
            # The target values stay exactly 0/1; a common float32 dtype avoids
            # numpy's float32+int64 -> float64 promotion in the mixed cache.
            data["y"] = data["y"].astype(np.float32)
            if "sample_id" not in data:
                data["sample_id"] = np.asarray([f"SYNTH:{frame}:{int(corner)}" for frame, corner in
                                                zip(data["id"].astype(str), data["native_corner"])])
            if any(value.shape[:1] != (count,) for value in data.values()):
                raise ValueError(f"non-query-aligned field: {path}")
            sources.append(data)
            rgb_provenance_path = path.with_suffix(".RGB_PROVENANCE.json")
            rgb_provenance = json.loads(rgb_provenance_path.read_text())
            if not rgb_provenance["PASS"] or rgb_provenance["out_cache_sha256"] != sha256(path):
                raise ValueError(f"RGB preparation provenance mismatch: {path}")
            source_provenance[f"{source}_{split}"] = {"path": str(rgb_provenance_path.resolve()),
                                                       "sha256": sha256(rgb_provenance_path),
                                                       "unique_images": rgb_provenance["unique_images"],
                                                       "center_checks": rgb_provenance["center_checks_against_independent_PIL"]}
        if set(sources[0]) != set(sources[1]):
            raise ValueError(f"field mismatch after sample_id normalization: {set(sources[0]) ^ set(sources[1])}")
        combined[split] = {key: np.concatenate([data[key] for data in sources], axis=0) for key in sources[0]}
        inputs[split] = input_records
        offset = 0
        for data in sources:
            count = len(data["x"])
            for key, value in data.items():
                if not np.array_equal(combined[split][key][offset:offset + count], value):
                    raise AssertionError(f"concatenation changed source {split}:{key}")
            offset += count
    split_checks = {}
    for key in ("group_id", "sample_id", "image_sha256"):
        overlap = set(combined["TRAIN"][key].astype(str)) & set(combined["VALIDATION"][key].astype(str))
        if overlap:
            raise ValueError(f"TRAIN/VALIDATION {key} overlap: {sorted(overlap)[:5]}")
        split_checks[key] = {"checked": True, "overlap": 0}
    output_records = {}
    for split, data in combined.items():
        path = outputs[split]
        np.savez_compressed(path, **data)
        output_records[split] = {"path": str(path.resolve()), "sha256": sha256(path),
                                 "queries": len(data["x"]),
                                 "source_counts": {source: int((data["source"] == source).sum())
                                                   for source in np.unique(data["source"])},
                                 "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                            for key, value in data.items()}}
    report = {"PASS": True, "inputs": inputs, "outputs": output_records, "source_RGB_provenance": source_provenance,
              "normalization": {"y": "float32, exact unchanged binary values", "missing_SYNTH_sample_id": "SYNTH:{id}:{native_corner}"},
              "all_existing_fields_preserved": True, "all_existing_values_preserved": True,
              "split_checks": split_checks, "evaluation_image_overlap": 0, "denied_images": len(denied),
              "eval_denylist_sha256": sha256(args.eval_denylist), "code_sha256": sha256(__file__),
              "GT_masks_read": False, "RGB_images_decoded": False, "GPU_used": False,
              "seconds": time.monotonic() - start}
    provenance.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"PASS": True, "outputs": {key: {k: v for k, v in value.items() if k != "fields"}
                                                for key, value in output_records.items()},
                      "split_checks": split_checks, "evaluation_image_overlap": 0,
                      "provenance": str(provenance.resolve()), "seconds": report["seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-denylist", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
