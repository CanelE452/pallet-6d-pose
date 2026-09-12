"""Independent integrity audit for an exported local-fusion population."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .constants import POINT_SIGMA_DIAGONAL_FRACTION, R0_SHA256
from .data import OBSERVATION_KEYS, TARGET_KEYS, resolve_record_path
from .geometry import edge_graph_preserved
from .util import immutable_json, read_json, sha256


def run(export: Path, output: Path) -> dict:
    provenance = read_json(export / "EXPORT_PROVENANCE.json")
    stock_source = read_json(provenance["stock_source_manifest"])
    source_by_id = {record["id"]: record for record in stock_source["records"]}
    cache = Path(provenance["stock_cache_manifest"]).parent
    stock = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in (
        "points", "boxes", "point_valid", "gt_points", "gt_valid", "input_shape", "gain", "record_index",
    )}
    ids, image_hashes = set(), set()
    populations, symmetry_counts, asset_counts = {}, Counter(), Counter()
    failures = []
    source_image_bytes_verified = 0
    stock_point_exact = stock_box_exact = stock_target_exact = 0
    for split, expected in (("train", 1792), ("calibration", 256), ("synth_val", 512)):
        manifest_path = export / f"{split}.json"
        manifest = read_json(manifest_path)
        records = manifest["records"]
        populations[split] = len(records)
        if len(records) != expected:
            failures.append(f"{split} population {len(records)} != {expected}")
        for record in records:
            frame_id, image_sha = record["frame_id"], record["source_image_sha256"]
            if frame_id in ids: failures.append(f"duplicate frame id: {frame_id}")
            if image_sha in image_hashes: failures.append(f"duplicate image SHA: {image_sha}")
            ids.add(frame_id); image_hashes.add(image_sha)
            obs_path = resolve_record_path(manifest_path, record["observation"])
            target_path = resolve_record_path(manifest_path, record["supervision"])
            if sha256(obs_path) != record["observation_sha256"]:
                failures.append(f"observation SHA: {frame_id}"); continue
            if sha256(target_path) != record["supervision_sha256"]:
                failures.append(f"supervision SHA: {frame_id}"); continue
            obs = torch.load(obs_path, map_location="cpu")
            target = torch.load(target_path, map_location="cpu")
            if tuple(obs) != OBSERVATION_KEYS or tuple(target) != TARGET_KEYS:
                failures.append(f"key contract: {frame_id}"); continue
            expected_shapes = {
                "features": (1, 192, 24, 24), "content": (1, 1, 24, 24),
                "box": (1, 4), "image_hw": (1, 2), "dims": (1, 3),
                "base_points": (1, 9, 2), "point_valid": (1, 9), "point_sigma": (1, 9),
            }
            if any(tuple(obs[key].shape) != shape for key, shape in expected_shapes.items()):
                failures.append(f"observation shape: {frame_id}")
            if obs["features"].dtype != torch.float32 or obs["content"].dtype != torch.float32:
                failures.append(f"feature dtype: {frame_id}")
            source_record = source_by_id.get(frame_id)
            if source_record is None or source_record["image_sha256"] != image_sha:
                failures.append(f"source manifest binding: {frame_id}")
            else:
                source_image = Path(source_record["image"])
                if not source_image.is_file() or sha256(source_image) != image_sha:
                    failures.append(f"source image bytes: {frame_id}")
                else:
                    source_image_bytes_verified += 1
                row = int(record["stock_cache_index"])
                if int(stock["record_index"][row]) != row or source_record["index"] != row:
                    failures.append(f"stock row identity: {frame_id}")
                prepared_h, prepared_w = map(int, source_record["prepared_shape_hw"])
                input_h, input_w = map(int, stock["input_shape"][row])
                gain = min(input_h / prepared_h, input_w / prepared_w)
                resized_h, resized_w = round(prepared_h * gain), round(prepared_w * gain)
                offset = np.asarray((round((input_w-resized_w)/2-.1),
                                     round((input_h-resized_h)/2-.1)), np.float64)
                def recover(value):
                    shape = value.shape
                    return ((np.asarray(value).reshape(-1,2)-offset)/gain-100).reshape(shape).astype(np.float32)
                expected_points = torch.from_numpy(recover(stock["points"][row]))
                expected_box = torch.from_numpy(recover(stock["boxes"][row].reshape(2,2)).reshape(4))
                expected_target = torch.from_numpy(recover(stock["gt_points"][row]))
                if torch.equal(obs["base_points"][0], expected_points): stock_point_exact += 1
                else: failures.append(f"stock point parity: {frame_id}")
                if torch.equal(obs["box"][0], expected_box): stock_box_exact += 1
                else: failures.append(f"stock box parity: {frame_id}")
                target_equal = torch.allclose(target["points"][0], expected_target,
                                              rtol=0, atol=0, equal_nan=True)
                if target_equal and torch.equal(target["valid"][0], torch.from_numpy(np.array(stock["gt_valid"][row],copy=True))):
                    stock_target_exact += 1
                else: failures.append(f"stock target parity: {frame_id}")
                if not torch.equal(obs["point_valid"][0], torch.from_numpy(np.array(stock["point_valid"][row],copy=True))):
                    failures.append(f"stock point-valid parity: {frame_id}")
            diagonal = torch.linalg.vector_norm(obs["image_hw"].flip(-1), dim=-1)
            if not torch.allclose(obs["point_sigma"],
                                  torch.full((1, 9), POINT_SIGMA_DIAGONAL_FRACTION * float(diagonal)),
                                  rtol=1e-6, atol=1e-7):
                failures.append(f"point sigma: {frame_id}")
            permutations = record["symmetry_permutations"]
            if any(not edge_graph_preserved(permutation) for permutation in permutations):
                failures.append(f"symmetry graph: {frame_id}")
            symmetry_counts[f"C{record['symmetry_order']}"] += 1
            asset_counts[record["asset"]] += 1
    affine = read_json(export / "AFFINE_AUDIT.json")
    if provenance["stock_checkpoint_sha256"] != R0_SHA256:
        failures.append("R0 checkpoint SHA")
    if not affine["PASS"]:
        failures.append("affine audit")
    result = {
        "schema": "symdht_local_integrity_audit_v1",
        "PASS": not failures,
        "failures": failures,
        "populations": populations,
        "unique_frame_ids": len(ids),
        "unique_source_image_sha256": len(image_hashes),
        "source_image_bytes_verified": source_image_bytes_verified,
        "stock_point_exact_parity": stock_point_exact,
        "stock_box_exact_parity": stock_box_exact,
        "stock_target_exact_parity": stock_target_exact,
        "cross_split_frame_and_image_sha_disjoint": len(ids) == 2560 and len(image_hashes) == 2560,
        "symmetry_counts": dict(sorted(symmetry_counts.items())),
        "asset_counts": dict(sorted(asset_counts.items())),
        "c4_status": "C4_NOT_EVALUATED" if not symmetry_counts["C4"] else "EVALUATED",
        "affine_audit_sha256": sha256(export / "AFFINE_AUDIT.json"),
        "export_provenance_sha256": sha256(export / "EXPORT_PROVENANCE.json"),
    }
    immutable_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.export.resolve(), args.output.resolve())
    print("PASS" if result["PASS"] else "FAIL")
    if not result["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
