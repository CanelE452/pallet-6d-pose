"""Reference-free twenty-region RGB queries at unchanged actual poses.

Queries are not assertions that cuboid edges are material boundaries. The fixed
original detector box sets every candidate's patch size in a frame. Missing boxes
produce an explicitly absent visual block, never a GT/candidate-derived box.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

from scripts.research.pallet_boundary_action_repair_v1.action_data import actions
from scripts.research.pallet_boundary_action_repair_v1.gallery import EDGES
from .common import read, rows, save_rows, sha, write, measured

PATCH_SIZE = 16
PATCH_BOX_DIAGONAL_FRACTION = .05
CHUNK_ACTIONS = 8


def original_model_queries(native_xyz):
    xyz = np.asarray(native_xyz, np.float64)
    if xyz.shape != (8, 3):
        raise ValueError("Original eight-corner native model required")
    # Audit the read original adjacency against the registered cuboid geometry.
    if len(EDGES) != 12 or any(sum(i in edge for edge in EDGES) != 3 for i in range(8)):
        raise ValueError("Original cuboid adjacency must have twelve edges/degree three")
    for a, b in EDGES:
        if np.count_nonzero(xyz[a] != xyz[b]) != 1:
            raise ValueError("Source edge is not a native cuboid edge")
    return np.concatenate((xyz, np.asarray([(xyz[a] + xyz[b]) / 2 for a, b in EDGES])))


def project_queries(pose, geometry):
    native = np.asarray(geometry["native_xyz"], np.float64)
    query = original_model_queries(native)
    Q = np.asarray(geometry["physical_registration_Q"], np.float64)
    R = np.asarray(pose["R_physical"], np.float64) @ Q.T
    camera = (query - native.mean(0)) @ R.T + np.asarray(pose["centroid"], np.float64)
    homogeneous = camera @ np.asarray(geometry["K"], np.float64).T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    return pixels, camera[:, 2]


def sample(image, pixels, depths, side):
    """Exact float64 bilinear coordinate arithmetic; float32 RGB/mask output.

    Sample centers outside the original pixel-center canvas receive RGB=mask=0.
    No query center or sampling coordinate is clipped onto the image boundary.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Original RGB image required")
    h, w = image.shape[:2]
    output = np.zeros((20, 4, PATCH_SIZE, PATCH_SIZE), np.float32)
    if side is None or not np.isfinite(side) or side <= 0:
        return output, np.zeros(20, np.float32), np.zeros(20, bool)
    offset = ((np.arange(PATCH_SIZE, dtype=np.float64) + .5) / PATCH_SIZE - .5) * side
    dx, dy = np.meshgrid(offset, offset)
    pixels, depths = np.asarray(pixels), np.asarray(depths)
    query_valid = np.isfinite(pixels).all(1) & np.isfinite(depths) & (depths > 0)
    for region in np.flatnonzero(query_valid):
        x = pixels[region, 0] + dx
        y = pixels[region, 1] + dy
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        if not valid.any():
            continue
        xv, yv = x[valid], y[valid]
        x0, y0 = np.floor(xv).astype(np.int64), np.floor(yv).astype(np.int64)
        # Neighbor replication is used only for exactly valid last-pixel centers;
        # outside sample centers remain absent and are never clipped into view.
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        wx, wy = xv - x0, yv - y0
        value = ((1 - wx)[:, None] * (1 - wy)[:, None] * image[y0, x0]
                 + wx[:, None] * (1 - wy)[:, None] * image[y0, x1]
                 + (1 - wx)[:, None] * wy[:, None] * image[y1, x0]
                 + wx[:, None] * wy[:, None] * image[y1, x1]) / 255.
        for channel in range(3):
            output[region, channel][valid] = value[:, channel].astype(np.float32)
        output[region, 3] = valid.astype(np.float32)
    return output, output[:, 3].mean((1, 2)), query_valid


def make_inputs(old_run, box_path, output):
    """Join original non-reference geometry, RGB and detected boxes, no forward."""
    old_run, output = Path(old_run), Path(output)
    boxes = {r["id"]: r for r in rows(box_path)}
    output.mkdir(parents=True, exist_ok=False)
    values = []
    for domain, upstream_name in (("real", "REAL238_UPSTREAM.jsonl.gz"),
                                  ("synthetic", "SYNTHETIC1003_UPSTREAM.jsonl.gz")):
        upstream = list(rows(old_run / "d1_upstream_v1" / upstream_name))
        for index, row in enumerate(upstream):
            frame_dir = old_run / ("d1_" + domain + "_candidates_v1") / "frames" / f"{index:04d}"
            epath = frame_dir / "E_RUN.jsonl.gz"
            e = next(rows(epath))
            if e["id"] != row["id"]:
                raise ValueError("Original E_RUN/upstream identity join changed")
            family = e["families"]["ORIGINAL_WD"]
            native = np.asarray(family["native_xyz"], np.float64)
            if not np.array_equal(native, np.asarray(row["native_xyz"], np.float64)):
                raise ValueError("Original upstream native registry changed")
            original_model_queries(native)
            image = Path(row["image"])
            if sha(image) != row["image_sha256"]:
                raise ValueError("Original image bytes changed")
            with Image.open(image) as img:
                if [img.height, img.width] != row["raw_hw"]:
                    raise ValueError("Raw RGB canvas changed")
            box = boxes.get(row["id"])
            coordinates = box.get("detector_box_xyxy_raw") if box else None
            side = None
            if coordinates is not None:
                a = np.asarray(coordinates, np.float64)
                if a.shape != (4,) or not np.isfinite(a).all() or not (a[2] > a[0] and a[3] > a[1]):
                    raise ValueError("Bad original detector box cannot be silently replaced")
                side = float(PATCH_BOX_DIAGONAL_FRACTION * np.linalg.norm(a[2:] - a[:2]))
            values.append(dict(id=row["id"], domain=domain, partition=row["partition"],
                split=row["split"], image=str(image), image_sha256=row["image_sha256"], raw_hw=row["raw_hw"],
                K=row["K"], native_xyz=family["native_xyz"],
                physical_registration_Q=family["physical_registration_Q"],
                detector_box_xyxy_raw=coordinates, patch_side_raw_px=side,
                visual_status="AVAILABLE" if side is not None else "MISSING_ORIGINAL_DETECTOR_BOX",
                e_run_sha256=sha(epath), box_binding=box,
                image_inputs="original unedited RGB; no overlay or GT crop"))
    save_rows(output / "VISUAL_INPUTS.jsonl.gz", values)
    write(output / "INPUT_SEAL.json", dict(status="COMPLETE", reference_inputs=False,
        rows=len(values), real_n=sum(r["domain"] == "real" for r in values),
        synthetic_n=sum(r["domain"] == "synthetic" for r in values),
        available_visual_n=sum(r["visual_status"] == "AVAILABLE" for r in values),
        box_inputs_sha256=sha(box_path), adjacency=list(EDGES),
        adjacency_source_sha256=sha(Path(__file__).parents[1] / "pallet_boundary_action_repair_v1" / "gallery.py"),
        query_source_sha256=sha(__file__),
        artifacts={"VISUAL_INPUTS.jsonl.gz": sha(output / "VISUAL_INPUTS.jsonl.gz")}))


class PatchProvider:
    """Bounded RGB decode LRU and per-frame disk arrays, candidate chunks only."""
    def __init__(self, inputs_path, cache_root=None, *, rgb_lru_frames=2):
        self.geometry = {r["id"]: r for r in rows(inputs_path)}
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.images = OrderedDict()
        self.rgb_lru_frames = rgb_lru_frames
        self.decoded_images = 0
        self.region_count = 0
        self.open_cache = None
        self.open_cache_id = None

    def _image(self, item):
        key = item["id"]
        if key not in self.images:
            if sha(item["image"]) != item["image_sha256"]:
                raise ValueError("RGB input changed since geometry binding")
            with Image.open(item["image"]) as image:
                array = np.asarray(image.convert("RGB"), np.uint8)
            if list(array.shape[:2]) != item["raw_hw"]:
                raise ValueError("Raw image coordinate canvas changed")
            self.images[key] = array
            self.decoded_images += 1
            while len(self.images) > self.rgb_lru_frames:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        return self.images[key]

    def _prefix(self, fid):
        return self.cache_root / hashlib.sha256(fid.encode()).hexdigest()

    def __call__(self, reference_free_frame, actual_entries, indices):
        fid = reference_free_frame["id"]
        item = self.geometry[fid]
        index = np.asarray(indices, np.int64)
        if index.ndim != 1 or np.any(index < 0) or np.any(index >= len(actual_entries)):
            raise ValueError("Candidate chunk is not a subset of original actions")
        if len(index) > CHUNK_ACTIONS:
            raise ValueError("Visual query chunks must not exceed eight actual actions")
        if self.cache_root is not None:
            prefix = self._prefix(fid)
            if self.open_cache_id != fid:
                meta = read(str(prefix) + ".json")
                if meta["candidate_ids"] != [c["candidate_id"] for c in actual_entries]:
                    raise ValueError("Visual cache/action order mismatch")
                self.open_cache = np.load(str(prefix) + ".npy", mmap_mode="r", allow_pickle=False)
                self.open_cache_id = fid
            patches = np.asarray(self.open_cache[index], np.float32)
            return patches, patches[:, :, 3].mean((2, 3))
        patches = np.zeros((len(index), 20, 4, 16, 16), np.float32)
        validity = np.zeros((len(index), 20), np.float32)
        if item["patch_side_raw_px"] is not None:
            image = self._image(item)
            for out_index, action_index in enumerate(index):
                q, z = project_queries(actual_entries[int(action_index)]["actual_pose"], item)
                patches[out_index], validity[out_index], _ = sample(image, q, z, item["patch_side_raw_px"])
        self.region_count += len(index) * 20
        return patches, validity


def cache(inputs_path, features, output, domain):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provider = PatchProvider(inputs_path)
    records = []
    started = time.monotonic()
    for frame in rows(features):
        item = provider.geometry[frame["id"]]
        if item["domain"] != domain:
            raise ValueError("Visual domain join changed")
        entries = actions(frame)
        prefix = output / hashlib.sha256(frame["id"].encode()).hexdigest()
        array = np.lib.format.open_memmap(str(prefix) + ".npy", mode="w+", dtype=np.float16,
                                          shape=(len(entries), 20, 4, 16, 16))
        qrecords = []
        for first in range(0, len(entries), CHUNK_ACTIONS):
            indices = np.arange(first, min(first + CHUNK_ACTIONS, len(entries)))
            patches, validity = provider(frame, entries, indices)
            array[indices] = patches.astype(np.float16)
            for candidate_index in indices:
                xy, depth = project_queries(entries[int(candidate_index)]["actual_pose"], item)
                qrecords.append(dict(candidate_id=entries[int(candidate_index)]["candidate_id"],
                    query_xy_raw=[[float(x), float(y)] if np.isfinite([x, y]).all() else None for x, y in xy],
                    depth_m=[float(z) if np.isfinite(z) else None for z in depth]))
        array.flush()
        del array
        record = dict(id=frame["id"], candidate_ids=[c["candidate_id"] for c in entries],
            action_pose_sha256=hashlib.sha256(str([(c["candidate_id"], c["actual_pose"]) for c in entries]).encode()).hexdigest(),
            array_file=prefix.name + ".npy", array_sha256=sha(str(prefix) + ".npy"),
            shape=[len(entries), 20, 4, 16, 16], dtype="float16",
            queries=qrecords, patch_side_raw_px=item["patch_side_raw_px"], image_sha256=item["image_sha256"],
            raw_hw=item["raw_hw"], no_GT_inputs=True)
        write(str(prefix) + ".json", record)
        records.append(dict({k: record[k] for k in
            ("id", "array_file", "array_sha256", "shape", "dtype")},
            metadata_sha256=sha(str(prefix) + ".json")))
        if len(records) % 100 == 0:
            print("VISUAL_CACHE", domain, len(records), flush=True)
    write(output / "CACHE_SEAL.json", dict(status="COMPLETE", domain=domain,
        input_sha256=sha(inputs_path), features_sha256=sha(features),
        reference_inputs=False, records=records, new_model_forward_calls=0,
        query_source_sha256=sha(__file__), cached_precision="float16 from float32 RGB[0,1]; masks binary exact; models receive float32",
        allocated_all_candidate_patches_in_memory=False, max_candidate_chunk=CHUNK_ACTIONS,
        decoded_image_count=provider.decoded_images, queried_regions=provider.region_count,
        cache_bytes=sum(p.stat().st_size for p in output.iterdir()),
        wall_seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--boxes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("cache")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--domain", choices=("real", "synthetic"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inputs":
        make_inputs(args.old_run, args.boxes, args.output)
    else:
        cache(args.inputs, args.features, args.output, args.domain)


if __name__ == "__main__":
    main()
