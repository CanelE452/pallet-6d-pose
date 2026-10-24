"""Add exact query-pixel-centered, odd 65x65 RGB source crops to a cache.

All existing NPZ fields are preserved. RGB samples use OpenCV remap, while
every output center is independently checked against PIL's decoded RGB pixel.
The evaluation-image SHA denylist is enforced before either image decoder.
This helper reads source RGB and initial-query metadata only, never GT masks.
"""
import argparse
from collections import OrderedDict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf8") as stream:
            return json.load(stream)
    return json.loads(Path(path).read_text())


def unique_index(rows, key):
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate metadata {key}: {value}")
        result[value] = row
    return result


def build(args):
    start = time.monotonic()
    cache_path, out = Path(args.cache), Path(args.out)
    provenance_path = out.with_suffix(".RGB_PROVENANCE.json")
    temp = out.with_name(out.name + ".tmp.npz")
    if out.exists() or provenance_path.exists() or temp.exists():
        raise FileExistsError("refusing to overwrite RGB cache, provenance, or temporary output")
    denied = set(load_json(args.eval_denylist)["image_sha256"])
    if not denied:
        raise ValueError("empty evaluation-image denylist")
    with np.load(cache_path, allow_pickle=False) as stream:
        arrays = {key: stream[key] for key in stream.files}
    if "rgb" in arrays:
        raise ValueError("input already contains RGB; refusing to replace it")
    if arrays["x"].ndim != 4 or arrays["x"].shape[1:] != (65, 9, 9):
        raise ValueError("RGB source builder requires a flattened N,65,9,9 cache")
    count = len(arrays["x"])
    if "image_sha256" not in arrays or arrays["image_sha256"].shape != (count,):
        raise ValueError("per-query source image_sha256 is required")
    cache_hashes = arrays["image_sha256"].astype(str)
    if set(cache_hashes) & denied:
        raise ValueError("evaluation-image hash found in source cache metadata")
    allowed = {"SYNTH": {"SYNTH", "SYNTHETIC"}, "YCB": {"YCB", "YCBV", "YCB-V"}}[args.kind]
    if "source" not in arrays or not set(np.char.upper(arrays["source"].astype(str))) <= allowed:
        raise ValueError(f"cache source array must contain only {sorted(allowed)}")
    metadata_hashes = {}
    if args.kind == "SYNTH":
        if args.queries_csv:
            raise ValueError("--queries-csv is for YCB only")
        initial = unique_index(load_json(args.synth_inputs), "id")
        features = unique_index(load_json(args.synth_feature_metadata), "id")
        metadata_hashes = {"synth_inputs": sha256(args.synth_inputs),
                           "synth_feature_metadata": sha256(args.synth_feature_metadata)}
        if "id" not in arrays or "native_corner" not in arrays:
            raise ValueError("SYNTH requires id and native_corner NPZ arrays")
        cache_ids = arrays["id"].astype(str)
        records = []
        for index, frame_id in enumerate(cache_ids):
            frame, feature = initial[frame_id], features[frame_id]
            corner = int(arrays["native_corner"][index])
            if not 0 <= corner < 8:
                raise ValueError("native_corner must be 0..7")
            x, y = frame["initial_points"][corner]
            if str(frame["image_sha256"]) != cache_hashes[index]:
                raise ValueError(f"SYNTH metadata/cache image hash mismatch: {frame_id}")
            records.append((frame["image"], float(x), float(y), float(feature["side"])))
    else:
        if not args.queries_csv:
            raise ValueError("YCB requires --queries-csv")
        with Path(args.queries_csv).open(newline="") as stream:
            query = unique_index(list(csv.DictReader(stream)), "sample_id")
        metadata_hashes = {"queries_csv": sha256(args.queries_csv)}
        id_key = "sample_id" if "sample_id" in arrays else "id"
        if id_key not in arrays:
            raise ValueError("YCB cache requires sample_id or id")
        cache_ids = arrays[id_key].astype(str)
        records = []
        for sample_id in cache_ids:
            row = query[sample_id]
            records.append((row["image"], float(row["query_x"]), float(row["query_y"]), float(row["roi_side"])))
    if len(records) != count:
        raise AssertionError("metadata alignment failed")
    rgb = np.empty((count, 3, 65, 65), dtype=np.uint8)
    image_cache = OrderedDict()
    known_hash = {}
    decode_count, center_checks, out_of_bounds_centers = 0, 0, 0
    image_inventory = {}
    for index, (image_name, qx, qy, side) in enumerate(records):
        path = Path(image_name).resolve()
        if not np.isfinite([qx, qy, side]).all() or side <= 0:
            raise ValueError(f"nonfinite query or nonpositive ROI side at {index}")
        if path not in known_hash:
            actual_hash = sha256(path)
            if actual_hash in denied:
                raise ValueError(f"evaluation image denied before decode: {path}")
            known_hash[path] = actual_hash
            image_inventory[str(path)] = actual_hash
        if known_hash[path] != cache_hashes[index]:
            raise ValueError(f"actual image/cache SHA mismatch: {path}")
        if path in image_cache:
            cv_rgb, pil_rgb = image_cache.pop(path)
            image_cache[path] = (cv_rgb, pil_rgb)
        else:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"OpenCV could not decode {path}")
            cv_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with Image.open(path) as image:
                pil_rgb = np.array(image.convert("RGB"))
            if cv_rgb.shape != pil_rgb.shape:
                raise AssertionError(f"OpenCV/PIL dimensions disagree: {path}")
            image_cache[path] = (cv_rgb, pil_rgb)
            decode_count += 1
            while len(image_cache) > args.image_cache_size:
                image_cache.popitem(last=False)
        center_x, center_y = int(np.floor(qx + 0.5)), int(np.floor(qy + 0.5))
        offset = np.linspace(-side / 2.0, side / 2.0, 65, dtype=np.float32)
        map_x, map_y = np.meshgrid(offset + center_x, offset + center_y)
        crop = cv2.remap(cv_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        height, width = pil_rgb.shape[:2]
        inside = 0 <= center_x < width and 0 <= center_y < height
        expected = pil_rgb[center_y, center_x] if inside else np.zeros(3, np.uint8)
        if not np.array_equal(crop[32, 32], expected):
            raise AssertionError(f"RGB center disagrees with independent PIL pixel at {index}: {path}")
        center_checks += 1
        out_of_bounds_centers += int(not inside)
        rgb[index] = crop.transpose(2, 0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temp, **arrays, rgb=rgb)
    # Verify serialization preserves every source field byte for byte.
    with np.load(temp, allow_pickle=False) as stored:
        for key, value in arrays.items():
            same = np.array_equal(stored[key], value, equal_nan=True) if np.issubdtype(value.dtype, np.number) else np.array_equal(stored[key], value)
            if stored[key].dtype != value.dtype or not same:
                raise AssertionError(f"source field changed: {key}")
        if not np.array_equal(stored["rgb"], rgb):
            raise AssertionError("RGB serialization mismatch")
    temp.rename(out)
    provenance = {"PASS": True, "kind": args.kind, "source_cache": str(cache_path.resolve()),
                  "source_cache_sha256": sha256(cache_path), "out_cache": str(out.resolve()),
                  "out_cache_sha256": sha256(out), "code_sha256": sha256(__file__),
                  "metadata_sha256": metadata_hashes, "denylist_sha256": sha256(args.eval_denylist),
                  "denied_image_count": len(denied), "denied_image_overlap": 0,
                  "queries": count, "unique_images": len(known_hash), "dual_decoder_cache_misses": decode_count,
                  "RGB_shape": list(rgb.shape), "RGB_dtype": str(rgb.dtype),
                  "center_checks_against_independent_PIL": center_checks,
                  "out_of_bounds_centers": out_of_bounds_centers,
                  "recipe": {"center": "floor(query+0.5)", "offsets": "linspace(-roi_side/2,roi_side/2,65)",
                             "interpolation": "cv2.INTER_LINEAR", "border": "constant RGB zero",
                             "layout": "N,3,65,65 RGB uint8", "center_index": [32, 32]},
                  "GT_masks_read": False, "evaluation_RGB_read": False, "all_existing_fields_preserved": True,
                  "source_images_sha256": image_inventory, "seconds": time.monotonic() - start}
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in provenance.items() if key != "source_images_sha256"}))


def main():
    source_root = Path(__file__).resolve().parents[3] / "external_ray_proxy_head_20261022_v60"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--kind", choices=("SYNTH", "YCB"), required=True)
    parser.add_argument("--queries-csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-denylist", required=True)
    parser.add_argument("--synth-inputs", default=str(source_root / "PREPARED_INPUTS_SYNTH.json.gz"))
    parser.add_argument("--synth-feature-metadata", default=str(source_root / "SYNTH_FEATURE_METADATA.json"))
    parser.add_argument("--image-cache-size", type=int, default=16)
    args = parser.parse_args()
    if args.image_cache_size < 1:
        parser.error("--image-cache-size must be positive")
    build(args)


if __name__ == "__main__":
    main()
