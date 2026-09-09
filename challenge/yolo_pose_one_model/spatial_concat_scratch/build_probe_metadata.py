"""Build the complete 60k metadata manifest for post-hoc S0/S1 training.

Only fixed renderer-axis XYZ may enter the dimension branch.  The stored
camera-facing W/H/D is retained as provenance because it is needed to audit the
renderer convention, but it is an explicitly forbidden model input: W>D is the
parity label by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MERGED = REPO / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
G38_RAW = REPO / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
G38_MANIFEST = (
    REPO
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "ubuntu_cf_loss_queue_20260823T0930/G38_GENERIC_ONLY_MANIFEST.json"
)
LEGACY_METADATA = HERE / "LEGACY_SPATIAL_METADATA.jsonl"
PHASE_B_AUDIT = (
    REPO
    / "challenge/yolo_pose_one_model/dimension_conditioning_probe/"
    "phase_b_data/SYNTH_METADATA_AUDIT.json"
)
PHASE_B_SPLIT = (
    REPO
    / "challenge/yolo_pose_one_model/dimension_conditioning_probe/"
    "phase_b_data/SPLIT_MEMBERSHIP.json"
)
OUTPUT = HERE / "PROBE_METADATA_60K.jsonl"
AUDIT = HERE / "PROBE_METADATA_60K_AUDIT.json"

EXPECTED_SHA256 = {
    G38_MANIFEST: "a085075c86ad65ecdb6ff7a5961baa6424451ec7024ec330b9df10688b7aff48",
    LEGACY_METADATA: "1a37109f4789f6f574927f8862d15983fb5bd4657b0852dbbaa2ac43ab75ba29",
    PHASE_B_AUDIT: "32a54c669bd5f0f445cea28e9315be46a869cf296c487f85ecaf0062cc5a8161",
    PHASE_B_SPLIT: "fd8f39f59e229bd9bdd9172212ea3e52dcf8e77b80898b67905e809e90d8582e",
}
EXPECTED = {
    "G38:train": 38002,
    "G38:val": 1998,
    "P0:train": 8989,
    "P0:val": 1011,
    "TEX:train": 8989,
    "TEX:val": 1011,
}
EXPECTED_SPLIT_PARITY = {
    "train:long": 29503,
    "train:short": 26477,
    "val:long": 2243,
    "val:short": 1777,
}
EXPECTED_PHASE_B_ASSET_SPLIT = {"TRAIN": 20281, "DEV": 9624, "TEST": 10095}
EXPECTED_RAW_SHAPE_COUNTS = {
    "G38:480x640": 19315,
    "G38:480x720": 5834,
    "G38:540x960": 10962,
    "G38:560x560": 3889,
    "P0:480x640": 4758,
    "P0:480x720": 1494,
    "P0:540x960": 2865,
    "P0:560x560": 883,
    "TEX:480x640": 4732,
    "TEX:480x720": 1486,
    "TEX:540x960": 2901,
    "TEX:560x560": 881,
}
FRAME_RE = re.compile(r"^f(?P<index>[0-9]+)_label[.]json$")
FIXED_WIDTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 1), (2, 3), (4, 5), (6, 7))
)
FIXED_DEPTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 4), (1, 5), (2, 6), (3, 7))
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.absolute().relative_to(REPO.absolute()))


def publish_temporary_exclusive(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        publish_temporary_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        publish_temporary_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fixed_dimensions(
    obj: dict[str, Any], context: str
) -> tuple[list[float], list[float], list[int], str]:
    dims = obj.get("dimensions_m")
    permutation_value = obj.get("perm_v4")
    if not isinstance(dims, dict):
        raise RuntimeError(f"{context}: dimensions_m missing")
    if (
        not isinstance(permutation_value, list)
        or len(permutation_value) != 8
        or sorted(permutation_value) != list(range(8))
    ):
        raise RuntimeError(f"{context}: invalid perm_v4")
    camera = [float(dims[key]) for key in ("width", "height", "depth")]
    if min(camera) <= 0:
        raise RuntimeError(f"{context}: non-positive dimensions")
    permutation = [int(x) for x in permutation_value]
    mapped = frozenset((permutation[0], permutation[1]))
    if mapped in FIXED_WIDTH_EDGES:
        fixed = list(camera)
        axis_case = "CAMERA_WIDTH_IS_FIXED_X"
    elif mapped in FIXED_DEPTH_EDGES:
        fixed = [camera[2], camera[1], camera[0]]
        axis_case = "CAMERA_WIDTH_IS_FIXED_Z"
    else:
        raise RuntimeError(f"{context}: camera width edge not fixed X/Z")
    return camera, fixed, permutation, axis_case


def parity_from_object(obj: dict[str, Any], camera: list[float], context: str) -> str:
    parity = "long" if camera[0] > camera[2] else "short"
    explicit = obj.get("efront_kp12")
    if explicit is not None:
        if not isinstance(explicit, dict) or explicit.get("kp12_valid") is not True:
            raise RuntimeError(f"{context}: invalid explicit parity block")
        if explicit.get("front_face_type") != parity:
            raise RuntimeError(f"{context}: explicit/reconstructed parity mismatch")
    return parity


def decoded_raw_shape(image: Path, context: str) -> list[int]:
    try:
        with Image.open(image) as prepared:
            prepared_w, prepared_h = (int(value) for value in prepared.size)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"{context}: merged image header decode failed") from exc
    raw_shape = [prepared_h - 200, prepared_w - 200]
    if min(raw_shape) <= 0:
        raise RuntimeError(
            f"{context}: prepared image too small for the frozen 100px border: "
            f"{(prepared_h, prepared_w)}"
        )
    return raw_shape


def g38_rows() -> Iterable[dict[str, Any]]:
    membership = json.loads(G38_MANIFEST.read_text(encoding="utf-8"))
    split_by_id: dict[str, str] = {}
    for split in ("train", "val"):
        values = membership.get(split)
        if not isinstance(values, list):
            raise RuntimeError(f"G38 manifest missing {split}")
        for sample_id in values:
            if sample_id in split_by_id:
                raise RuntimeError(f"duplicate G38 membership: {sample_id}")
            split_by_id[str(sample_id)] = split

    labels: list[tuple[int, Path]] = []
    for path in (G38_RAW / "labels").glob("f*_label.json"):
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"noncanonical G38 label: {path.name}")
        labels.append((int(match.group("index")), path))
    labels.sort()
    if len(labels) != 40000 or len(split_by_id) != 40000:
        raise RuntimeError("G38 source/membership is not 40,000")
    phase_b = json.loads(PHASE_B_SPLIT.read_text(encoding="utf-8"))
    phase_b_split = phase_b.get("frame_to_split")
    phase_b_parity = phase_b.get("frame_to_parity")
    if not isinstance(phase_b_split, dict) or not isinstance(phase_b_parity, dict):
        raise RuntimeError("Phase-B frame mappings missing")

    for index, path in labels:
        frame_id = f"f{index:04d}" if index < 10000 else f"f{index}"
        sample_id = f"G__{frame_id}"
        split = split_by_id.get(sample_id)
        if split is None:
            raise RuntimeError(f"G38 membership missing {sample_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        objects = payload.get("objects")
        if not isinstance(objects, list) or len(objects) != 1:
            raise RuntimeError(f"{sample_id}: expected one object")
        obj = objects[0]
        camera, fixed, permutation, axis_case = fixed_dimensions(obj, sample_id)
        parity = parity_from_object(obj, camera, sample_id)
        if phase_b_parity.get(frame_id) != parity.upper() + "_FRONT":
            raise RuntimeError(f"{sample_id}: Phase-B parity mismatch")
        asset_split = phase_b_split.get(frame_id)
        if asset_split not in {"TRAIN", "DEV", "TEST"}:
            raise RuntimeError(f"{sample_id}: Phase-B asset split missing")
        camera_data = payload.get("camera_data") or {}
        raw_shape = [int(camera_data.get("height", 0)), int(camera_data.get("width", 0))]
        if min(raw_shape) <= 0:
            raise RuntimeError(f"{sample_id}: camera resolution missing")
        merged_stem = f"G38__{sample_id}"
        image = MERGED / "images" / split / f"{merged_stem}.png"
        label = MERGED / "labels" / split / f"{merged_stem}.txt"
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"{sample_id}: merged image/label missing")
        decoded_shape = decoded_raw_shape(image, sample_id)
        if decoded_shape != raw_shape:
            raise RuntimeError(
                f"{sample_id}: camera/prepared raw shape mismatch: "
                f"{raw_shape} != {decoded_shape}"
            )
        yield {
            "source": "G38",
            "sample_id": sample_id,
            "pair_group_id": sample_id,
            "merged_stem": merged_stem,
            "split": split,
            "dataset_split": split,
            "image_path": repo_relative(image),
            "yolo_label_path": repo_relative(label),
            "renderer_label_locator": repo_relative(path),
            "raw_shape_height_width": raw_shape,
            "prepared_pad_px": 100,
            "camera_facing_dimensions_m_width_height_depth_provenance_only": camera,
            "camera_facing_dimensions_are_forbidden_as_model_input": True,
            "fixed_renderer_dimensions_m_xyz_model_input": fixed,
            "perm_v4": permutation,
            "fixed_axis_reconstruction": axis_case,
            "front_face_type": parity,
            "front_face_class_short0_long1": int(parity == "long"),
            "source_asset": obj.get("source_asset"),
            "phase_b_asset_split_provenance_only": asset_split,
        }


def legacy_rows() -> Iterable[dict[str, Any]]:
    for line in LEGACY_METADATA.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = row["source"]
        source_map = {
            "legacy_v1v2_p0_10k": "P0",
            "legacy_v1v2_p0_tex10k": "TEX",
        }
        if source not in source_map:
            raise RuntimeError(f"unknown legacy source: {source}")
        short_source = source_map[source]
        split = row["split"]
        stem = row["merged_stem"]
        image = MERGED / "images" / split / f"{stem}.png"
        label = MERGED / "labels" / split / f"{stem}.txt"
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"{stem}: merged image/label missing")
        raw_shape = decoded_raw_shape(image, stem)
        parity = row["front_face_type"]
        if parity not in {"short", "long"}:
            raise RuntimeError(f"{stem}: invalid parity")
        output = dict(row)
        output.update(
            {
                "source": short_source,
                "dataset_split": split,
                "image_path": repo_relative(image),
                "yolo_label_path": repo_relative(label),
                "raw_shape_height_width": raw_shape,
                "raw_shape_provenance": (
                    "decoded already-padded merged image height/width minus "
                    "the frozen 100px border on each side"
                ),
                "prepared_pad_px": 100,
                "front_face_class_short0_long1": int(parity == "long"),
                "renderer_label_locator": (
                    repo_relative(
                        REPO
                        / "challenge/yolo_pose_one_model/datasets/"
                        "_raw_legacy_v1v2_p0_10k"
                        / row["label_member"]
                    )
                    if short_source == "P0"
                    else (
                        "/home/minjae/Downloads/legacy_v1v2_p0_tex10k.zip::"
                        + row["label_member"]
                    )
                ),
            }
        )
        yield output


def main() -> None:
    if OUTPUT.exists() or AUDIT.exists():
        raise RuntimeError("probe metadata outputs already exist; refusing overwrite")
    observed = {str(path): sha256(path) for path in EXPECTED_SHA256}
    for path, expected in EXPECTED_SHA256.items():
        if observed[str(path)] != expected:
            raise RuntimeError(f"source hash mismatch: {path}")

    rows = list(g38_rows()) + list(legacy_rows())
    source_order = {"G38": 0, "P0": 1, "TEX": 2}
    split_order = {"train": 0, "val": 1}
    rows.sort(
        key=lambda r: (
            split_order[r["split"]],
            source_order[r["source"]],
            r["sample_id"],
        )
    )
    if len(rows) != 60000:
        raise RuntimeError(f"metadata rows {len(rows)} != 60000")
    if len({row["merged_stem"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate merged stem")
    metadata_stems = {row["merged_stem"] for row in rows}
    merged_stems = {
        path.stem
        for split in ("train", "val")
        for path in (MERGED / "images" / split).glob("*.png")
    }
    if metadata_stems != merged_stems:
        raise RuntimeError(
            f"metadata/merged population mismatch: "
            f"metadata_only={len(metadata_stems-merged_stems)} "
            f"merged_only={len(merged_stems-metadata_stems)}"
        )

    counts = Counter(f"{row['source']}:{row['split']}" for row in rows)
    if dict(counts) != EXPECTED:
        raise RuntimeError(f"source/split counts {dict(counts)} != {EXPECTED}")
    parity = Counter(
        f"{row['source']}:{row['split']}:{row['front_face_type']}" for row in rows
    )
    split_parity = Counter(
        f"{row['split']}:{row['front_face_type']}" for row in rows
    )
    if dict(split_parity) != EXPECTED_SPLIT_PARITY:
        raise RuntimeError(
            f"split/parity counts {dict(split_parity)} != {EXPECTED_SPLIT_PARITY}"
        )
    phase_b_asset_split = Counter(
        row["phase_b_asset_split_provenance_only"]
        for row in rows
        if row["source"] == "G38"
    )
    if dict(phase_b_asset_split) != EXPECTED_PHASE_B_ASSET_SPLIT:
        raise RuntimeError(
            "Phase-B asset split provenance changed: "
            f"{dict(phase_b_asset_split)} != {EXPECTED_PHASE_B_ASSET_SPLIT}"
        )
    raw_shape_counts = Counter(
        f"{row['source']}:{row['raw_shape_height_width'][0]}x"
        f"{row['raw_shape_height_width'][1]}"
        for row in rows
    )
    if dict(raw_shape_counts) != EXPECTED_RAW_SHAPE_COUNTS:
        raise RuntimeError(
            f"decoded raw-shape counts {dict(raw_shape_counts)} "
            f"!= {EXPECTED_RAW_SHAPE_COUNTS}"
        )
    for row in rows:
        if row["dataset_split"] != row["split"]:
            raise RuntimeError("dataset split alias mismatch")
        dimensions = row["fixed_renderer_dimensions_m_xyz_model_input"]
        if (
            row.get("camera_facing_dimensions_are_forbidden_as_model_input") is not True
            or not isinstance(dimensions, list)
            or len(dimensions) != 3
            or min(float(value) for value in dimensions) <= 0.0
        ):
            raise RuntimeError(f"invalid model-input dimension contract: {row['merged_stem']}")
    leakage_agreement = sum(
        (
            row["camera_facing_dimensions_m_width_height_depth_provenance_only"][0]
            > row["camera_facing_dimensions_m_width_height_depth_provenance_only"][2]
        )
        == (row["front_face_type"] == "long")
        for row in rows
    )
    if leakage_agreement != len(rows):
        raise RuntimeError("camera-facing WHD parity audit changed")

    write_jsonl_exclusive(OUTPUT, rows)
    manifest_sha = sha256(OUTPUT)
    audit = {
        "schema_version": "spatial_concat_probe_metadata_60k_v1",
        "source_sha256": observed,
        "manifest": {
            "path": repo_relative(OUTPUT),
            "sha256": manifest_sha,
            "rows": len(rows),
        },
        "counts": dict(sorted(counts.items())),
        "parity_counts": dict(sorted(parity.items())),
        "split_parity_counts": dict(sorted(split_parity.items())),
        "phase_b_asset_split_counts_provenance_only": dict(
            sorted(phase_b_asset_split.items())
        ),
        "decoded_raw_shape_counts": dict(sorted(raw_shape_counts.items())),
        "prepared_image_shape_verified_against_raw_plus_200": len(rows),
        "merged_stem_unique": True,
        "merged_stem_set_exact": True,
        "merged_image_and_label_present": len(rows),
        "fixed_xyz_reconstructed_with_perm_v4": len(rows),
        "camera_facing_whd_parity_agreement": {
            "count": leakage_agreement,
            "total": len(rows),
            "model_input_forbidden": True,
        },
        "allowed_dimension_input": "fixed_renderer_dimensions_m_xyz_model_input",
        "split_role": "YOLO train/val reused as exploratory probe TRAIN/DEV; no TEST claim",
        "builder_sha256": sha256(Path(__file__)),
    }
    write_json_exclusive(AUDIT, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
