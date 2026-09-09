"""Frozen populations for the synthetic attention / Direct-Hough comparison.

The training mask is decoded from JSON mask_rle, never thresholded at zero from
the renderer PNG (its background can contain values 1 and 2). Cuboid edge roles
use camera_dynamic_0123_v4 on both domains. They are cuboid support lines, not a
claim that every projected edge is a visible piece of pallet material.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import sys

import cv2
import numpy as np


CONVENTION = "camera_dynamic_0123_v4"
GENERATION_CHUNK_SIZE = 250
OPEN_REAL_KEYS = ("eval_outside", "eval_noapril", "eval_cad")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        head = stream.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or len(head) != 24:
        raise ValueError(f"Expected PNG image: {path}")
    return struct.unpack(">II", head[16:24])


def decode_rle(rle: dict) -> np.ndarray:
    """Uncompressed COCO RLE: column-major, first run is background."""
    height, width = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if not isinstance(counts, list) or any(not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("Expected nonnegative integer uncompressed mask_rle counts")
    if sum(counts) != height * width:
        raise ValueError("mask_rle does not cover the declared image")
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    for i, length in enumerate(counts):
        if i % 2:
            flat[cursor:cursor + length] = 255
        cursor += length
    return flat.reshape((height, width), order="F")


def _record(image: Path, annotation: Path, points, camera: dict,
            population: str, frame_id: str, group: str) -> dict:
    points = np.asarray(points, dtype=float)[:8]
    if points.shape != (8, 2):
        raise ValueError(f"Expected eight 2D corners: {annotation}")
    width, height = int(camera["width"]), int(camera["height"])
    if _image_size(image) != (width, height):
        raise ValueError(f"Image and annotation dimensions disagree: {image}")
    valid = np.isfinite(points).all(axis=1) & ~np.all(points == -1, axis=1)
    in_frame = valid & (points[:, 0] >= 0) & (points[:, 0] < width)
    in_frame &= (points[:, 1] >= 0) & (points[:, 1] < height)
    # Invalid entries remain a conventional finite sentinel for strict JSON.
    points[~valid] = -1
    return {
        "id": frame_id, "population": population,
        "image": str(image.resolve()), "annotation": str(annotation.resolve()),
        "gt_points": points.tolist(), "width": width, "height": height,
        "mask": None, "gt_valid": valid.tolist(), "gt_in_frame": in_frame.tolist(),
        "group": group, "convention": CONVENTION,
        "image_sha256": _sha(image), "annotation_sha256": _sha(annotation),
    }


def _generation_groups(source: Path, annotations: list[Path]) -> dict[str, list[Path]]:
    """Use actual renderer invocations, identified by their generation logs."""
    groups = {}
    for log in sorted((source / "logs").glob("chunk_*_*.log")):
        match = re.fullmatch(r"chunk_(\d+)_(\d+)\.log", log.name)
        if match is None:
            continue
        chunk, start = (int(v) for v in match.groups())
        if start != chunk * GENERATION_CHUNK_SIZE:
            raise ValueError(f"Unexpected generation chunk boundary: {log}")
        groups[f"paper4:render_chunk_{chunk:03d}"] = [
            p for p in annotations if start <= int(p.stem) < start + GENERATION_CHUNK_SIZE
        ]
    covered = [p for files in groups.values() for p in files]
    if len(covered) != len(annotations) or len(set(covered)) != len(covered):
        raise ValueError("Renderer log chunks do not partition the source annotations")
    return groups


def _allocate(groups: dict[str, list[Path]], counts: dict[str, int], rng: random.Random):
    remaining = sorted(groups)
    rng.shuffle(remaining)
    result, membership = {}, {}
    for population, count in counts.items():
        selected, assigned = [], []
        while len(selected) < count:
            if not remaining:
                raise ValueError("Not enough disjoint renderer groups for requested populations")
            group = remaining.pop()
            assigned.append(group)
            frames = list(groups[group])
            rng.shuffle(frames)
            selected.extend((group, frame) for frame in frames)
        result[population] = selected[:count]
        membership[population] = assigned
    return result, membership


def _synthetic_record(path: Path, population: str, group: str,
                      output: Path, require_mask: bool) -> dict:
    data = json.loads(path.read_text())
    if len(data.get("objects", [])) != 1:
        raise ValueError(f"Expected one target pallet: {path}")
    obj = data["objects"][0]
    if obj.get("keypoint_convention") != CONVENTION:
        raise ValueError(f"Unverified corner convention: {path}")
    record = _record(path.with_suffix(".png"), path, obj["projected_cuboid"],
                     data["camera_data"], population, f"{path.parent.name}__{path.stem}", group)
    record.update({
        "source_asset": obj.get("source_asset"),
        "source_kind": "synthetic",
        "conditions": {k: obj.get(k) for k in (
            "visibility", "raycast_visibility", "camera_elevation_deg", "camera_azimuth_deg",
        )},
        "physical_audit": obj.get("physical_audit", {}),
    })
    if require_mask:
        mask = decode_rle(obj["mask_rle"])
        if mask.shape != (record["height"], record["width"]):
            raise ValueError(f"RLE and camera dimensions disagree: {path}")
        area = int(np.count_nonzero(mask))
        if area == 0 or area != int(obj["mask_area_px"]):
            raise ValueError(f"RLE mask area mismatch: {path}")
        mask_path = output / "masks_rle" / f"{path.stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Failed to write decoded mask: {mask_path}")
        record["mask"] = str(mask_path.resolve())
        record["mask_source"] = "objects[0].mask_rle"
        record["mask_area_px"] = area
        # This is an independent serialization check, not the training target.
        source_png = path.parent / obj["visible_mask"]
        png = cv2.imread(str(source_png), cv2.IMREAD_GRAYSCALE)
        record["renderer_png_threshold127_equals_rle"] = bool(
            png is not None and png.shape == mask.shape and np.array_equal(png > 127, mask > 0)
        )
    return record


def _real_records(root: Path) -> list[dict]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from challenge.data_paths import EVAL_CANONICAL, REAL_DEV_OPEN_CLEAN_COUNT

    result = []
    for key in OPEN_REAL_KEYS:
        directory = root / EVAL_CANONICAL[key]
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            objects = data.get("objects", [])
            if not objects or objects[0].get("split") != "eval":
                continue
            obj = objects[0]
            if obj.get("gt_source") != "manual":
                raise ValueError(f"Non-manual canonical evaluation annotation: {path}")
            # Canonical legacy JSON lacks a convention string. Its migrated v2
            # copy explicitly defines camera-facing semantics; prove coordinate
            # parity instead of inferring a physical-axis permutation from pose.
            migrated = (root / "challenge/real_gt_v2/migrated_gt/eval_canonical"
                        / directory.name / path.name)
            migrated_obj = json.loads(migrated.read_text())["objects"][0]
            if migrated_obj.get("keypoint_frame") != CONVENTION:
                raise ValueError(f"Unverified real keypoint convention: {migrated}")
            points = np.asarray(obj["projected_cuboid"], float)[:8]
            mapped = np.asarray(migrated_obj["projected_cuboid"], float)[:8]
            if not np.allclose(points, mapped, atol=1e-6, rtol=0, equal_nan=True):
                raise ValueError(f"Canonical versus migrated corner disagreement: {path}")
            record = _record(path.with_suffix(".png"), path, points, data["camera_data"],
                             "real_dev", f"{key}__{path.stem}", key)
            record.update({
                "source_kind": "real_manual", "population_role": "DEV_OPEN",
                "convention_evidence": str(migrated.resolve()),
                "convention_evidence_sha256": _sha(migrated),
                "conditions": {
                    "session": key, "occlusion": migrated_obj.get("occlusion_level", "unknown"),
                    "truncation": migrated_obj.get("truncation", {}),
                },
                "keypoint_annotations": migrated_obj.get("keypoint_annotations", [])[:8],
            })
            result.append(record)
    if len(result) != REAL_DEV_OPEN_CLEAN_COUNT:
        raise ValueError(f"Canonical open DEV count changed: {len(result)}")
    return result


def build_manifest(root: Path, out: Path, seed=17, train_n=2048, val_n=256,
                   test_n=256, cross_n=128) -> dict[str, list[dict]]:
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts = {"synth_train": int(train_n), "synth_val": int(val_n), "synth_test": int(test_n)}
    if any(n <= 0 for n in counts.values()) or cross_n < 0:
        raise ValueError("Train/val/test counts must be positive; cross_n must be nonnegative")
    source = root / "data/pallet/training_data/paper_4pallet_mask_v1"
    annotations = sorted(p for p in source.glob("*.json") if p.stem.isdigit())
    groups = _generation_groups(source, annotations)
    rng = random.Random(seed)
    selected, group_membership = _allocate(groups, counts, rng)
    populations = {
        population: [_synthetic_record(path, population, group, out, True)
                     for group, path in entries]
        for population, entries in selected.items()
    }
    if cross_n:
        cross_root = root / "data/pallet/training_data/v4_split_base"
        cross_files = sorted(p for p in cross_root.glob("*.json")
                             if re.fullmatch(r"b\d+_\d+", p.stem))
        if len(cross_files) < cross_n:
            raise ValueError("Not enough v4 cross-synthetic images")
        rng.shuffle(cross_files)
        populations["cross_v4"] = [
            _synthetic_record(p, "cross_v4", f"v4_split_base:{p.stem.split('_')[0]}", out, False)
            for p in cross_files[:cross_n]
        ]
    populations["real_dev"] = _real_records(root)

    seen = {}
    for population, records in populations.items():
        for record in records:
            digest = record["image_sha256"]
            if digest in seen:
                raise ValueError(f"Duplicate image SHA across selected frames: {seen[digest]} / {record['id']}")
            seen[digest] = record["id"]
    synth_records = [r for k in counts for r in populations[k]]
    audit = {
        "source_root": str(source), "available_frames": len(annotations),
        "generation_groups": len(groups), "generation_chunk_size": GENERATION_CHUNK_SIZE,
        "split_method": "Seeded whole renderer-invocation chunks; within-chunk sampling to exact counts",
        "group_membership": group_membership,
        "group_disjoint": all(not set(group_membership[a]) & set(group_membership[b])
                              for a in counts for b in counts if a < b),
        "image_sha256_disjoint": True, "unique_selected_images": len(seen),
        "mask_semantics": "Visible pallet material from mask_rle; excludes gaps/background and occluding objects; not cuboid hull/amodal mask",
        "mask_rle_area_validated": len(synth_records),
        "renderer_png_threshold127_matches": sum(r["renderer_png_threshold127_equals_rle"] for r in synth_records),
        "role_convention": {"name": CONVENTION, "front": [0, 1, 2, 3],
                            "rear": [4, 5, 6, 7], "top": [0, 1, 4, 5], "bottom": [2, 3, 6, 7]},
        "gt_valid_semantics": "Finite annotated geometry excluding (-1,-1); offscreen coordinates retained; not physical visibility",
        "real_registry": "challenge.data_paths.EVAL_CANONICAL",
        "real_sessions": list(OPEN_REAL_KEYS), "real_expected_count": 52,
        "real_label_semantics": "Canonical manual projected_cuboid; coordinate parity to migrated camera_dynamic_0123_v4 verified; no pose reprojection",
        "real_role": "Already-open development diagnostic, not unseen held-out final",
        "omitted_cross_synthetic": {"aug_squash_v2": "Convention metadata absent; do not infer labels", "mixed_v8_train": "Excluded under CLAUDE v8 prohibition"},
        "selected_assets": dict(Counter(r["source_asset"] for r in synth_records)),
        "limitations": [
            "Renderer chunk isolation does not establish unseen mesh or fully independent scene assets; the same four assets recur",
            "Global seed/procedural scene identity is unavailable beyond recorded render invocation chunks",
            "Visible material attention target does not supply real pixelwise mask ground truth",
            "Image SHA audit covers these selected populations only; inherited backbone pretraining overlap requires separate provenance",
        ],
    }
    payload = {"schema_version": "hough_attention_transfer_manifest_v1", "seed": int(seed),
               "counts": {k: len(v) for k, v in populations.items()},
               "populations": populations, "audit": audit}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return populations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-n", type=int, default=2048)
    parser.add_argument("--val-n", type=int, default=256)
    parser.add_argument("--test-n", type=int, default=256)
    parser.add_argument("--cross-n", type=int, default=128)
    args = parser.parse_args()
    result = build_manifest(args.root, args.out, args.seed, args.train_n, args.val_n, args.test_n, args.cross_n)
    print(json.dumps({k: len(v) for k, v in result.items()}))


if __name__ == "__main__":
    main()
