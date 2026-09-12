"""Export the fixed 2,560-frame pilot from the canonical stock-R0 cache."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .constants import (C2_YAW, IDENTITY, POINT_SIGMA_DIAGONAL_FRACTION,
                        R0_SHA256, permutations_for_order)
from .geometry import edge_graph_preserved
from .util import canonical_sha, immutable_json, read_json, sha256

REPO = Path(__file__).resolve().parents[4]
DEFAULT_STOCK = REPO / "data/pallet/results/pallet_line_pose_v1"
DEFAULT_SPLITS = REPO / "data/pallet/results/pallet_point_line_v4/export"
DEFAULT_OUTPUT = REPO / "data/pallet/results/pallet_symmetry_dht_local_v1/export"
LEGACY_METADATA = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/LEGACY_SPATIAL_METADATA.jsonl"
SYMMETRY_CONTRACT = REPO / "_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json"


def letterbox(prepared_hw: list[int], input_hw: np.ndarray) -> tuple[float, np.ndarray]:
    height, width = map(int, prepared_hw)
    input_h, input_w = map(int, input_hw)
    gain = min(input_h / height, input_w / width)
    resized_h, resized_w = round(height * gain), round(width * gain)
    left = round((input_w - resized_w) / 2 - .1)
    top = round((input_h - resized_h) / 2 - .1)
    if (resized_h + 2 * top not in {input_h - 1, input_h, input_h + 1}
            or resized_w + 2 * left not in {input_w - 1, input_w, input_w + 1}):
        raise ValueError("invalid cached letterbox geometry")
    return float(gain), np.asarray((left, top), np.float64)


def input_to_raw(value: np.ndarray, gain: float, offset: np.ndarray, pad: int = 100) -> np.ndarray:
    shape = value.shape
    points = value.reshape(-1, 2)
    return ((points - offset) / gain - pad).reshape(shape)


def raw_to_input(value: np.ndarray, gain: float, offset: np.ndarray, pad: int = 100) -> np.ndarray:
    shape = value.shape
    points = value.reshape(-1, 2)
    return ((points + pad) * gain + offset).reshape(shape)


def sample_roi(feature: np.ndarray, box_raw: np.ndarray, gain: float, offset: np.ndarray,
               stride: int, raw_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample native feature cells at 24x24 evenly spaced raw-ROI centres."""
    axis = (np.arange(24, dtype=np.float64) + .5) / 24
    raw_x = box_raw[0] + axis * (box_raw[2] - box_raw[0])
    raw_y = box_raw[1] + axis * (box_raw[3] - box_raw[1])
    yy, xx = np.meshgrid(raw_y, raw_x, indexing="ij")
    raw = np.stack((xx, yy), -1)
    network = raw_to_input(raw, gain, offset)
    # align_corners=False: feature cell j is centred at input stride*(j+.5).
    height, width = feature.shape[-2:]
    grid_x = 2 * (network[..., 0] / stride) / width - 1
    grid_y = 2 * (network[..., 1] / stride) / height - 1
    grid = torch.from_numpy(np.stack((grid_x, grid_y), -1)).float()[None]
    tensor = torch.from_numpy(np.asarray(feature, dtype=np.float32))[None]
    sampled = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    raw_h, raw_w = raw_hw
    content = ((raw[..., 0] >= 0) & (raw[..., 0] <= raw_w - 1)
               & (raw[..., 1] >= 0) & (raw[..., 1] <= raw_h - 1))
    return sampled, torch.from_numpy(content)[None, None]


def legacy_metadata() -> dict[str, dict]:
    result = {}
    with LEGACY_METADATA.open() as stream:
        for line in stream:
            row = json.loads(line)
            result[row["merged_stem"]] = row
    return result


def metadata(record: dict, legacy: dict[str, dict]) -> tuple[list[float], str, str]:
    """Return W,D,H, asset, and the deployable metadata source."""
    if record["source"] in {"P0", "TEX"}:
        row = legacy[record["id"]]
        x, y, z = map(float, row["fixed_renderer_dimensions_m_xyz_model_input"])
        return [x, z, y], "scene.usd", "LEGACY_SPATIAL_METADATA.fixed_renderer_dimensions_m_xyz_model_input"
    label = REPO / record["renderer_annotation_locator_provenance_only"]
    obj = read_json(label)["objects"][0]
    dims = obj["dimensions_m"]
    return [float(dims["width"]), float(dims["depth"]), float(dims["height"])], \
        obj["source_asset"], "renderer annotation objects[0].dimensions_m"


def symmetry(asset: str, contract: dict) -> tuple[int, str, list[list[int]]]:
    entry = contract["assets"].get(asset)
    if entry is None or entry["status"] != "CONFIRMED":
        return 1, "UNKNOWN->C1", permutations_for_order(1)
    order = int(entry["max_valid_order"])
    if order not in (1, 2):
        raise ValueError("no confirmed C4 training contract exists for this pilot")
    return order, entry["basis"], permutations_for_order(order)


def save_exclusive(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite tensor: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial tensor exists: {temporary}")
    torch.save(value, temporary)
    temporary.rename(path)


def affine_fixtures() -> dict:
    """Independent ramp/impulse checks for the sampling centre convention."""
    h = w = 80
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    feature = np.stack((xx, yy), 0)
    box = np.asarray((0., 0., 480., 480.))
    sampled, _ = sample_roi(feature, box, 1., np.zeros(2), 8, (640, 640))
    axis = box[0] + (np.arange(24) + .5) / 24 * (box[2] - box[0])
    expected = (axis + 100) / 8 - .5
    ramp_error = max(float(np.abs(sampled[0, 0].numpy() - expected[None]).max()),
                     float(np.abs(sampled[0, 1].numpy() - expected[:, None]).max()))
    impulse = np.zeros((1, h, w), np.float32)
    impulse[0, 31, 37] = 1
    # A one-cell ROI centred exactly on the impulse cell must sample exactly 1.
    centre_raw = np.asarray(((37.5 * 8 - 100), (31.5 * 8 - 100)))
    tiny = np.asarray((centre_raw[0] - 4, centre_raw[1] - 4,
                       centre_raw[0] + 4, centre_raw[1] + 4))
    hit, _ = sample_roi(impulse, tiny, 1., np.zeros(2), 8, (640, 640))
    impulse_peak = float(hit.max())
    return {"ramp_max_abs_cells": ramp_error, "impulse_peak": impulse_peak,
            "PASS": ramp_error < 1e-5 and impulse_peak > .95}


def run(stock: Path, split_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"inspect/remove or relocate existing incomplete export first: {output}")
    source_path = stock / "SOURCE_MANIFEST.json"
    cache = stock / "cache"
    source = read_json(source_path)
    cache_manifest = read_json(cache / "CACHE_MANIFEST.json")
    completion = read_json(cache / "CACHE_COMPLETE.json")
    if cache_manifest["backbone_sha256"] != R0_SHA256 or not completion["PASS"] or not completion["complete"]:
        raise RuntimeError("canonical stock-R0 cache binding/completion failed")
    records = source["records"]
    by_id = {record["id"]: record for record in records}
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in (
        "p3", "p4", "points", "boxes", "point_valid", "gt_points", "gt_valid",
        "input_shape", "gain", "record_index", "detected", "matched",
    )}
    legacy = legacy_metadata()
    contract = read_json(SYMMETRY_CONTRACT)
    if not edge_graph_preserved(list(C2_YAW)) or list(IDENTITY)[8] != 8:
        raise RuntimeError("explicit symmetry permutation failed the cuboid graph contract")
    output.mkdir(parents=True)
    (output / "observations").mkdir()
    (output / "supervision").mkdir()
    seen_id, seen_sha = set(), set()
    population, symmetry_counts, asset_counts = {}, Counter(), Counter()
    global_index = 0
    correspondence_error = 0.0
    split_hashes = {}
    for split in ("train", "calibration", "synth_val"):
        source_split = split_root / f"{split}.json"
        split_hashes[split] = sha256(source_split)
        requested = read_json(source_split)["records"]
        exported = []
        for requested_record in requested:
            frame_id = requested_record["frame_id"]
            record = by_id[frame_id]
            row = int(record["index"])
            if int(arrays["record_index"][row]) != row:
                raise RuntimeError("stock cache row mapping changed")
            if frame_id in seen_id or record["image_sha256"] in seen_sha:
                raise RuntimeError("cross-split frame/image-byte overlap")
            seen_id.add(frame_id); seen_sha.add(record["image_sha256"])
            gain, offset = letterbox(record["prepared_shape_hw"], arrays["input_shape"][row])
            if abs(gain - float(arrays["gain"][row])) > 1e-12:
                raise RuntimeError("stock gain and reconstructed affine differ")
            raw_hw = tuple(record["raw_shape_hw"])
            points = input_to_raw(np.asarray(arrays["points"][row]), gain, offset)
            box = input_to_raw(np.asarray(arrays["boxes"][row]).reshape(2, 2), gain, offset).reshape(4)
            gt = input_to_raw(np.asarray(arrays["gt_points"][row]), gain, offset)
            back = raw_to_input(points, gain, offset)
            correspondence_error = max(correspondence_error,
                                       float(np.nanmax(np.abs(back - arrays["points"][row]))))
            p3, c3 = sample_roi(arrays["p3"][row], box, gain, offset, 8, raw_hw)
            p4, c4 = sample_roi(arrays["p4"][row], box, gain, offset, 16, raw_hw)
            content = c3 & c4
            features = torch.cat((p3, p4), 1).float() * content
            diagonal = math.hypot(raw_hw[0], raw_hw[1])
            dims, asset, dims_source = metadata(record, legacy)
            order, basis, permutations = symmetry(asset, contract)
            symmetry_counts[f"C{order}"] += 1; asset_counts[asset] += 1
            observation = {
                "features": features,
                "content": content.float(),
                "box": torch.from_numpy(box.astype(np.float32))[None],
                "image_hw": torch.tensor(raw_hw, dtype=torch.float32)[None],
                "dims": torch.tensor(dims, dtype=torch.float32)[None],
                "base_points": torch.from_numpy(points.astype(np.float32))[None],
                "point_valid": torch.from_numpy(np.array(arrays["point_valid"][row], dtype=bool, copy=True))[None],
                "point_sigma": torch.full((1, 9), POINT_SIGMA_DIAGONAL_FRACTION * diagonal),
            }
            supervision = {
                "points": torch.from_numpy(gt.astype(np.float32))[None],
                "valid": torch.from_numpy(np.array(arrays["gt_valid"][row], dtype=bool, copy=True))[None],
            }
            name = f"{global_index:06d}.pt"
            obs_path, target_path = output / "observations" / name, output / "supervision" / name
            save_exclusive(observation, obs_path); save_exclusive(supervision, target_path)
            exported.append({
                "frame_id": frame_id,
                "session_id": requested_record["session_id"],
                "split": split,
                "source": record["source"],
                "source_image": record["image"],
                "source_image_sha256": record["image_sha256"],
                "stock_cache_index": row,
                "stock_detected": bool(arrays["detected"][row]),
                "stock_matched": bool(arrays["matched"][row]),
                "observation": f"observations/{name}",
                "observation_sha256": sha256(obs_path),
                "supervision": f"supervision/{name}",
                "supervision_sha256": sha256(target_path),
                "asset": asset,
                "dims_whd_m": dims,
                "dims_source": dims_source,
                "dims_deployable": True,
                "symmetry_order": order,
                "symmetry_basis": basis,
                "symmetry_permutations": permutations,
            })
            global_index += 1
        manifest = {
            "schema": "symdht_local_export_v1", "role": split,
            "coordinate_frame": "original raw pixels without the stored 100px reflect border",
            "channels": {"P3": 64, "P4": 128, "concat": 192},
            "records": exported,
        }
        immutable_json(output / f"{split}.json", manifest)
        population[split] = len(exported)
    affine = affine_fixtures()
    affine.update({
        "schema": "symdht_local_affine_audit_v1",
        "actual_frame_correspondence_count": global_index,
        "raw_input_raw_roundtrip_max_abs_px": correspondence_error,
        "actual_correspondence_PASS": correspondence_error <= 1e-5,
        "mapping": "raw -> (raw+100)*gain+integer_letterbox_left_top -> input/stride-0.5 native cell",
    })
    affine["PASS"] = bool(affine["PASS"] and affine["actual_correspondence_PASS"])
    immutable_json(output / "AFFINE_AUDIT.json", affine)
    provenance = {
        "schema": "symdht_local_export_provenance_v1",
        "population": population,
        "symmetry_counts": dict(sorted(symmetry_counts.items())),
        "asset_counts": dict(sorted(asset_counts.items())),
        "stock_source_manifest": str(source_path),
        "stock_source_manifest_sha256": sha256(source_path),
        "stock_cache_manifest": str(cache / "CACHE_MANIFEST.json"),
        "stock_cache_manifest_sha256": sha256(cache / "CACHE_MANIFEST.json"),
        "stock_checkpoint_sha256": cache_manifest["backbone_sha256"],
        "split_lineage_sha256": split_hashes,
        "symmetry_contract": str(SYMMETRY_CONTRACT),
        "symmetry_contract_sha256": sha256(SYMMETRY_CONTRACT),
        "legacy_metadata": str(LEGACY_METADATA),
        "legacy_metadata_sha256": sha256(LEGACY_METADATA),
        "feature_source_dtype": "float16 native cache, explicitly converted to float32 after bilinear ROI sampling",
        "feature_population": "P3/P4 and point outputs from the same frozen R0 forward/cache row",
        "point_sigma": "fixed 0.005 * original raw image diagonal",
        "export_contract_sha256": canonical_sha({"population": population, "ids": sorted(seen_id)}),
    }
    immutable_json(output / "EXPORT_PROVENANCE.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, default=DEFAULT_STOCK)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.stock.resolve(), args.split_root.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
