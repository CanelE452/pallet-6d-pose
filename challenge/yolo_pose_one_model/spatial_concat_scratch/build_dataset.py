"""Build the deduplicated G38 + legacy P0 + textured-P0 pose dataset.

The downloaded ``legacy_v1v2_p0_10k.zip`` is byte-for-byte the source that was
already converted into ``datasets/legacy_v1v2_p0_10k``.  It is therefore used
once.  Only the new textured archive is decoded here.  Both legacy variants
use the existing sample-id SHA1 split so matched scenario IDs cannot cross the
train/validation boundary.  The two renders are not label duplicates:
texture/height generation also changes sampled dimensions and can change the
facing axis/parity.

The RGB/keypoint conversion imports the original G38 converter.  This keeps
the 100 px reflect padding, cuboid ordering, centroid, visibility, and bbox
contract identical to the prior YOE run.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[3]
YOLO_ROOT = REPO / "challenge/yolo_pose_one_model"
EXPERIMENT = Path(__file__).resolve().parent
DATASETS = YOLO_ROOT / "datasets"

P0_ARCHIVE = Path("/home/minjae/Downloads/legacy_v1v2_p0_10k.zip")
TEX_ARCHIVE = Path("/home/minjae/Downloads/legacy_v1v2_p0_tex10k.zip")
P0_ARCHIVE_SHA256 = "9f944f26a95cd3c881fe9eaaff733f4a4c27bc21b5cfb191527c7abd6c1b14df"
TEX_ARCHIVE_SHA256 = "82299d550d3310efea3dcabafa3b6ffde4dcd37d6b66b692cca1148c9f2b8c0e"

G38 = DATASETS / "g38_generic_only"
P0 = DATASETS / "legacy_v1v2_p0_10k"
P0_RAW = DATASETS / "_raw_legacy_v1v2_p0_10k"
TEX = DATASETS / "legacy_v1v2_p0_tex10k"
MERGED = DATASETS / "g38_legacy_v1v2_p0_tex20k"
VAL_FRAC = 0.10

EXPECTED = {
    "g38": {"train": 38002, "val": 1998},
    "p0": {"train": 8989, "val": 1011},
    "tex": {"train": 8989, "val": 1011},
    "merged": {"train": 55980, "val": 4020},
}

FIXED_WIDTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 1), (2, 3), (4, 5), (6, 7))
)
FIXED_DEPTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 4), (1, 5), (2, 6), (3, 7))
)

_spec = importlib.util.spec_from_file_location(
    "prepare_yolo_pose", YOLO_ROOT / "scripts/prepare_yolo_pose.py"
)
prep = importlib.util.module_from_spec(_spec)
sys.modules["prepare_yolo_pose"] = prep
assert _spec.loader is not None
_spec.loader.exec_module(prep)
PAD = prep.PAD

_worker_zip: zipfile.ZipFile | None = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_val(sample_id: str) -> bool:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) < VAL_FRAC * 10000


def validate_zip(zf: zipfile.ZipFile) -> None:
    for info in zf.infolist():
        name = info.filename
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise RuntimeError(f"unsafe ZIP member: {name!r}")


def init_worker(zip_path: str) -> None:
    global _worker_zip
    _worker_zip = zipfile.ZipFile(zip_path)


def spatial_row(
    source: str,
    row: dict[str, str],
    ann: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    objects = ann.get("objects") or []
    if len(objects) != 1:
        raise RuntimeError(f"{source}/{row['sample_id']}: expected one object")
    obj = objects[0]
    dims = obj.get("dimensions_m") or {}
    efront = obj.get("efront_kp12") or {}
    xyz = [dims.get("width"), dims.get("height"), dims.get("depth")]
    if not all(isinstance(x, (int, float)) and x > 0 for x in xyz):
        raise RuntimeError(f"{source}/{row['sample_id']}: invalid dimensions {dims}")
    parity = efront.get("front_face_type")
    axis = efront.get("front_face_axis")
    if not efront.get("kp12_valid") or parity not in {"short", "long"}:
        raise RuntimeError(
            f"{source}/{row['sample_id']}: invalid efront parity {efront}"
        )
    if axis not in {"W", "D"}:
        raise RuntimeError(f"{source}/{row['sample_id']}: invalid front axis {axis}")
    permutation_value = obj.get("perm_v4")
    if (
        not isinstance(permutation_value, list)
        or len(permutation_value) != 8
        or sorted(permutation_value) != list(range(8))
    ):
        raise RuntimeError(f"{source}/{row['sample_id']}: invalid perm_v4")
    permutation = [int(x) for x in permutation_value]
    mapped_camera_width_edge = frozenset((permutation[0], permutation[1]))
    width, height, depth = map(float, xyz)
    if mapped_camera_width_edge in FIXED_WIDTH_EDGES:
        fixed_xyz = [width, height, depth]
        axis_case = "CAMERA_WIDTH_IS_FIXED_X"
    elif mapped_camera_width_edge in FIXED_DEPTH_EDGES:
        fixed_xyz = [depth, height, width]
        axis_case = "CAMERA_WIDTH_IS_FIXED_Z"
    else:
        raise RuntimeError(
            f"{source}/{row['sample_id']}: fixed dimensions not recoverable"
        )
    expected_parity = "long" if width > depth else "short"
    if parity != expected_parity:
        raise RuntimeError(
            f"{source}/{row['sample_id']}: parity does not match camera WHD"
        )
    prefix = "P0" if source == "legacy_v1v2_p0_10k" else "TEX"
    return {
        "source": source,
        "sample_id": row["sample_id"],
        "pair_group_id": row["sample_id"],
        "split": split,
        "rgb_member": row["rgb"],
        "label_member": row["label"],
        "merged_stem": f"{prefix}__{row['sample_id']}",
        "camera_facing_dimensions_m_width_height_depth_provenance_only": xyz,
        "camera_facing_dimensions_are_forbidden_as_model_input": True,
        "fixed_renderer_dimensions_m_xyz_model_input": fixed_xyz,
        "perm_v4": permutation,
        "fixed_axis_reconstruction": axis_case,
        "front_face_type": parity,
        "front_face_axis": axis,
        "kp12_valid": True,
        "pallet_type": row.get("pallet_type"),
        "diagnostic_mode": row.get("diagnostic_mode"),
        "scene_preset": row.get("scene_preset"),
        "material_variant": row.get("material_variant"),
    }


def convert_tex_one(row: dict[str, str]) -> tuple[str, dict[str, Any]]:
    assert _worker_zip is not None
    sid = row["sample_id"]
    split = "val" if is_val(sid) else "train"
    ann = json.loads(_worker_zip.read(row["label"]))
    meta = spatial_row("legacy_v1v2_p0_tex10k", row, ann, split)
    img_dst = TEX / "images" / split / f"{sid}.png"
    lbl_dst = TEX / "labels" / split / f"{sid}.txt"

    if img_dst.is_file() and lbl_dst.is_file():
        return "exists", meta

    obj = ann["objects"][0]
    proj = obj.get("projected_cuboid")
    cen = obj.get("projected_cuboid_centroid")
    if not proj or len(proj) < 8 or not cen:
        return "no_annotation", meta
    kps = [tuple(map(float, p)) for p in proj[:8]] + [tuple(map(float, cen))]

    encoded = np.frombuffer(_worker_zip.read(row["rgb"]), dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable_image", meta
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    line = prep.to_line(pw, ph, [(x + PAD, y + PAD) for x, y in kps])
    if line is None:
        return "all_kp_outside", meta
    if not cv2.imwrite(str(img_dst), padded):
        return "image_write_failed", meta
    lbl_dst.write_text(line + "\n", encoding="utf-8")
    return "ok", meta


def read_manifest_from_zip(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    text = zf.read("manifest.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 10000:
        raise RuntimeError(f"tex manifest rows {len(rows)} != 10000")
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError("duplicate tex sample_id")
    return rows


def build_tex(workers: int = 8) -> tuple[list[dict[str, Any]], dict[str, int]]:
    for split in ("train", "val"):
        (TEX / "images" / split).mkdir(parents=True, exist_ok=True)
        (TEX / "labels" / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(TEX_ARCHIVE) as zf:
        validate_zip(zf)
        rows = read_manifest_from_zip(zf)

    stats: dict[str, int] = {}
    metadata: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(TEX_ARCHIVE),)
    ) as ex:
        for i, (status, meta) in enumerate(
            ex.map(convert_tex_one, rows, chunksize=16), start=1
        ):
            stats[status] = stats.get(status, 0) + 1
            metadata.append(meta)
            if i % 1000 == 0:
                print(f"tex {i}/10000 {stats}", flush=True)

    successful = stats.get("ok", 0) + stats.get("exists", 0)
    if successful != 10000:
        raise RuntimeError(f"tex conversion incomplete: {stats}")
    return metadata, stats


def load_p0_metadata() -> list[dict[str, Any]]:
    rows = list(csv.DictReader((P0_RAW / "manifest.csv").open(encoding="utf-8")))
    if len(rows) != 10000:
        raise RuntimeError(f"P0 manifest rows {len(rows)} != 10000")
    out = []
    for i, row in enumerate(rows, start=1):
        split = "val" if is_val(row["sample_id"]) else "train"
        with (P0_RAW / row["label"]).open(encoding="utf-8") as f:
            ann = json.load(f)
        out.append(spatial_row("legacy_v1v2_p0_10k", row, ann, split))
        if i % 2000 == 0:
            print(f"p0 metadata {i}/10000", flush=True)
    return out


def count_pairs(root: Path, split: str) -> int:
    images = {p.stem for p in (root / "images" / split).glob("*.png")}
    labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
    if images != labels:
        raise RuntimeError(
            f"{root.name}/{split}: image-label mismatch "
            f"images_only={len(images-labels)} labels_only={len(labels-images)}"
        )
    return len(images)


def link_exact(src: Path, dst: Path) -> str:
    src_resolved = src.resolve(strict=True)
    if dst.is_symlink():
        if dst.resolve(strict=True) != src_resolved:
            raise RuntimeError(f"wrong existing link: {dst}")
        return "exists"
    if dst.exists():
        raise RuntimeError(f"refusing to overwrite non-symlink: {dst}")
    os.symlink(os.path.relpath(src_resolved, dst.parent), dst)
    return "symlink"


def merge_sources() -> dict[str, Any]:
    sources = [
        ("g38", "G38", G38),
        ("p0", "P0", P0),
        ("tex", "TEX", TEX),
    ]
    report: dict[str, Any] = {"sources": {}, "merged": {}}
    for split in ("train", "val"):
        (MERGED / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED / "labels" / split).mkdir(parents=True, exist_ok=True)
        for source, prefix, root in sources:
            n = count_pairs(root, split)
            if n != EXPECTED[source][split]:
                raise RuntimeError(
                    f"{source}/{split}: observed {n}, expected {EXPECTED[source][split]}"
                )
            modes: dict[str, int] = {}
            for img in sorted((root / "images" / split).glob("*.png")):
                label = root / "labels" / split / f"{img.stem}.txt"
                stem = f"{prefix}__{img.stem}"
                a = link_exact(img, MERGED / "images" / split / f"{stem}.png")
                b = link_exact(label, MERGED / "labels" / split / f"{stem}.txt")
                modes[a] = modes.get(a, 0) + 1
                modes[b] = modes.get(b, 0) + 1
            report["sources"].setdefault(source, {})[split] = {
                "pairs": n,
                "link_actions_image_plus_label": modes,
            }
        merged_n = count_pairs(MERGED, split)
        if merged_n != EXPECTED["merged"][split]:
            raise RuntimeError(
                f"merged/{split}: observed {merged_n}, expected {EXPECTED['merged'][split]}"
            )
        report["merged"][split] = merged_n
    return report


def verify_pair_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = {
        source: {r["sample_id"]: r for r in rows if r["source"] == source}
        for source in ("legacy_v1v2_p0_10k", "legacy_v1v2_p0_tex10k")
    }
    p0 = by_source["legacy_v1v2_p0_10k"]
    tex = by_source["legacy_v1v2_p0_tex10k"]
    if p0.keys() != tex.keys():
        raise RuntimeError("P0/TEX paired sample-id sets differ")
    split_mismatch = []
    parity_mismatch = []
    for sid in sorted(p0):
        if p0[sid]["split"] != tex[sid]["split"]:
            split_mismatch.append(sid)
        if (
            p0[sid]["front_face_type"],
            p0[sid]["front_face_axis"],
        ) != (
            tex[sid]["front_face_type"],
            tex[sid]["front_face_axis"],
        ):
            parity_mismatch.append(sid)
    if split_mismatch:
        raise RuntimeError(
            f"paired split invariant failed: split={len(split_mismatch)}"
        )
    return {
        "pair_groups": len(p0),
        "split_mismatch": 0,
        "front_parity_mismatch_expected_distinct_renders": len(parity_mismatch),
        "train_pair_groups": sum(r["split"] == "train" for r in p0.values()),
        "val_pair_groups": sum(r["split"] == "val" for r in p0.values()),
    }


def write_outputs(
    metadata: list[dict[str, Any]], tex_stats: dict[str, int], merge: dict[str, Any]
) -> None:
    metadata.sort(key=lambda r: (r["split"], r["sample_id"], r["source"]))
    with (EXPERIMENT / "LEGACY_SPATIAL_METADATA.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in metadata:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    data_yaml = (
        "# generated by spatial_concat_scratch/build_dataset.py\n"
        f"path: {MERGED.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "kpt_shape: [9, 3]\n"
        "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n"
        "names:\n  0: pallet\n"
    )
    (MERGED / "data.yaml").write_text(data_yaml, encoding="utf-8")

    parity_counts: dict[str, dict[str, int]] = {}
    for row in metadata:
        key = f"{row['source']}:{row['split']}"
        parity_counts.setdefault(key, {"short": 0, "long": 0})[
            row["front_face_type"]
        ] += 1
    report = {
        "schema_version": "g38_p0_tex20k_dataset_build_v1",
        "archive_sha256": {
            str(P0_ARCHIVE): P0_ARCHIVE_SHA256,
            str(TEX_ARCHIVE): TEX_ARCHIVE_SHA256,
        },
        "downloaded_p0_is_existing_p0_duplicate": True,
        "duplicate_p0_instances_added": 0,
        "pad_px": PAD,
        "split": "sha1(sample_id), 10%; paired p0/tex sample IDs stay together",
        "expected": EXPECTED,
        "tex_convert_stats": tex_stats,
        "pair_audit": verify_pair_groups(metadata),
        "legacy_parity_counts": parity_counts,
        "merge": merge,
        "metadata_rows": len(metadata),
        "dimension_input_contract": {
            "allowed": "fixed_renderer_dimensions_m_xyz_model_input",
            "forbidden": "camera_facing_dimensions_m_width_height_depth_provenance_only",
            "reason": "camera-facing W>D equals the parity target exactly",
            "reconstruction": "map camera width edge through perm_v4",
        },
        "data_yaml": str(MERGED / "data.yaml"),
    }
    (EXPERIMENT / "DATASET_BUILD.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    if not P0_ARCHIVE.is_file() or not TEX_ARCHIVE.is_file():
        raise FileNotFoundError("both downloaded legacy archives are required")
    observed = {P0_ARCHIVE: sha256(P0_ARCHIVE), TEX_ARCHIVE: sha256(TEX_ARCHIVE)}
    expected = {P0_ARCHIVE: P0_ARCHIVE_SHA256, TEX_ARCHIVE: TEX_ARCHIVE_SHA256}
    if observed != expected:
        raise RuntimeError(f"archive hash mismatch: {observed}")

    tex_meta, tex_stats = build_tex()
    p0_meta = load_p0_metadata()
    metadata = p0_meta + tex_meta
    verify_pair_groups(metadata)
    merge = merge_sources()
    write_outputs(metadata, tex_stats, merge)


if __name__ == "__main__":
    main()
