#!/usr/bin/env python3
"""Build the replayable Phase-B G38 metadata audit and asset-ID split.

This builder deliberately reads each renderer JSON.  It does not infer geometry,
parity, asset identity, background, or viewpoint from a filename.  G38 stores
``dimensions_m`` in camera-facing W/H/D order; ``perm_v4`` is therefore used to
recover the renderer's fixed-object X/Y/Z dimensions before any distribution or
coverage statistic is computed.

Outputs are written beside this script:

* SYNTH_METADATA_AUDIT.json
* SPLIT_MEMBERSHIP.json
* SPLIT_MEMBERSHIP_SHA256.txt
* DIMENSION_DISTRIBUTION.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "dimension_conditioning_phase_b_data_v1"
FRAME_RE = re.compile(r"^f(?P<index>[0-9]+)_label[.]json$")
CUBOID_TOLERANCE_M = 2.0e-6
ROTATION_TOLERANCE = 1.0e-6
PARITY_MIN_FRACTION = 0.40
PARITY_MAX_FRACTION = 0.60

# Fixed-object corner convention.  Edges are unordered because perm_v4 may
# reverse their orientation.
FIXED_WIDTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 1), (2, 3), (4, 5), (6, 7))
)
FIXED_DEPTH_EDGES = frozenset(
    frozenset(edge) for edge in ((0, 4), (1, 5), (2, 6), (3, 7))
)

# Frozen literal assignment requested for this probe.  source_asset is the only
# complete family identifier in all 40,000 source JSONs.  It is not promoted to
# a topology certificate: the two GLBs are unavailable on this machine.
ASSET_TO_SPLIT = {
    "eur_pallet_bk_cc0.glb": "TRAIN",
    "woodpallet_block_jtoastie_ccby.glb": "TRAIN",
    "scene.usd": "DEV",
    "scene_1.usd": "TEST",
}
ASSET_TO_TOPOLOGY_PROXY = {
    "eur_pallet_bk_cc0.glb": "Pallet_3",
    "woodpallet_block_jtoastie_ccby.glb": "Pallet_2",
    "scene.usd": "Pallet_0",
    "scene_1.usd": "Pallet_1",
}
SPLIT_ORDER = ("TRAIN", "DEV", "TEST")
PARITY_ORDER = ("SHORT_FRONT", "LONG_FRONT")


class AuditError(RuntimeError):
    """Raised when a frozen Phase-B contract is violated."""


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    frame_index: int
    sample_id: str
    current_g38_split: str
    split: str
    source_asset: str
    topology_proxy: str
    camera_width_m: float
    camera_height_m: float
    camera_depth_m: float
    fixed_x_m: float
    fixed_y_m: float
    fixed_z_m: float
    aspect_x_z: float
    parity: str
    explicit_parity: str | None
    axis_case: str
    background_id: str
    scene_preset: str
    elevation_deg_actual: float
    azimuth_deg_target: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_json_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(stable_json_bytes(payload, pretty=True))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{field}: expected numeric value, got {value!r}") from exc
    require(math.isfinite(result), f"{field}: value is not finite")
    return result


def finite_vector(value: Any, length: int, field: str) -> tuple[float, ...]:
    require(isinstance(value, list) and len(value) == length, f"{field}: invalid shape")
    return tuple(finite_float(item, f"{field}[{index}]") for index, item in enumerate(value))


def finite_matrix(
    value: Any, rows: int, columns: int, field: str
) -> tuple[tuple[float, ...], ...]:
    require(isinstance(value, list) and len(value) == rows, f"{field}: invalid row count")
    return tuple(
        finite_vector(row, columns, f"{field}[{index}]")
        for index, row in enumerate(value)
    )


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def orthogonality_error(matrix: Sequence[Sequence[float]]) -> float:
    return max(
        abs(
            sum(matrix[k][row] * matrix[k][column] for k in range(3))
            - (1.0 if row == column else 0.0)
        )
        for row in range(3)
        for column in range(3)
    )


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    require(bool(sorted_values), "quantile requested for an empty sequence")
    require(0.0 <= probability <= 1.0, "quantile probability is outside [0,1]")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "numeric summary requested for no values")
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": quantile(ordered, 0.05),
        "median": quantile(ordered, 0.50),
        "mean": mean,
        "population_std": math.sqrt(variance),
        "p95": quantile(ordered, 0.95),
        "max": ordered[-1],
        "unique_exact": len(set(ordered)),
        "unique_rounded_6dp": len({round(value, 6) for value in ordered}),
    }


def parity_summary(records: Sequence[FrameRecord]) -> dict[str, Any]:
    counts = Counter(record.parity for record in records)
    total = len(records)
    result = {
        "count": total,
        "counts": {label: counts[label] for label in PARITY_ORDER},
        "fractions": {
            label: counts[label] / total if total else None for label in PARITY_ORDER
        },
    }
    result["within_40_60_gate"] = bool(
        total
        and all(
            PARITY_MIN_FRACTION <= result["fractions"][label] <= PARITY_MAX_FRACTION
            for label in PARITY_ORDER
        )
    )
    return result


def dimension_summary(records: Sequence[FrameRecord]) -> dict[str, Any]:
    require(bool(records), "dimension summary requested for no frames")
    triplets = [
        (record.fixed_x_m, record.fixed_y_m, record.fixed_z_m) for record in records
    ]
    raw_triplets = [
        (record.camera_width_m, record.camera_height_m, record.camera_depth_m)
        for record in records
    ]
    return {
        "count": len(records),
        "fixed_renderer_dimensions_m": {
            "x": numeric_summary(record.fixed_x_m for record in records),
            "y": numeric_summary(record.fixed_y_m for record in records),
            "z": numeric_summary(record.fixed_z_m for record in records),
        },
        "fixed_x_over_z": numeric_summary(record.aspect_x_z for record in records),
        "unique_fixed_xyz_exact": len(set(triplets)),
        "unique_fixed_xyz_rounded_to_mm": len(
            {tuple(round(value, 3) for value in triplet) for triplet in triplets}
        ),
        "unique_camera_facing_whd_exact": len(set(raw_triplets)),
        "parity": parity_summary(records),
    }


def eta_squared(records: Sequence[FrameRecord], category: str, value: str) -> float:
    values = [float(getattr(record, value)) for record in records]
    overall = sum(values) / len(values)
    total_ss = sum((number - overall) ** 2 for number in values)
    if total_ss == 0.0:
        return 0.0
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, category))].append(float(getattr(record, value)))
    between_ss = sum(
        len(numbers) * ((sum(numbers) / len(numbers)) - overall) ** 2
        for numbers in groups.values()
    )
    return between_ss / total_ss


def grouped_summaries(
    records: Sequence[FrameRecord], attribute: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, attribute))].append(record)
    return {
        key: dimension_summary(grouped[key])
        for key in sorted(grouped)
    }


def range_overlap_coefficient(
    first: Sequence[FrameRecord], second: Sequence[FrameRecord], attribute: str
) -> float:
    values_a = [float(getattr(record, attribute)) for record in first]
    values_b = [float(getattr(record, attribute)) for record in second]
    intersection = max(0.0, min(max(values_a), max(values_b)) - max(min(values_a), min(values_b)))
    union = max(max(values_a), max(values_b)) - min(min(values_a), min(values_b))
    return intersection / union if union else 1.0


def split_overlap(records_by_split: Mapping[str, Sequence[FrameRecord]]) -> dict[str, Any]:
    train = records_by_split["TRAIN"]
    train_min = {
        axis: min(float(getattr(record, attribute)) for record in train)
        for axis, attribute in (("x", "fixed_x_m"), ("y", "fixed_y_m"), ("z", "fixed_z_m"))
    }
    train_max = {
        axis: max(float(getattr(record, attribute)) for record in train)
        for axis, attribute in (("x", "fixed_x_m"), ("y", "fixed_y_m"), ("z", "fixed_z_m"))
    }
    output: dict[str, Any] = {
        "train_axis_aligned_box_m": {"min": train_min, "max": train_max},
        "heldout_against_train": {},
    }
    for split in ("DEV", "TEST"):
        heldout = records_by_split[split]
        inside_axis = {
            axis: sum(
                train_min[axis] <= float(getattr(record, attribute)) <= train_max[axis]
                for record in heldout
            )
            for axis, attribute in (("x", "fixed_x_m"), ("y", "fixed_y_m"), ("z", "fixed_z_m"))
        }
        inside_box = sum(
            train_min["x"] <= record.fixed_x_m <= train_max["x"]
            and train_min["y"] <= record.fixed_y_m <= train_max["y"]
            and train_min["z"] <= record.fixed_z_m <= train_max["z"]
            for record in heldout
        )
        output["heldout_against_train"][split] = {
            "count": len(heldout),
            "inside_train_box_count": inside_box,
            "inside_train_box_fraction": inside_box / len(heldout),
            "inside_train_range_by_axis_count": inside_axis,
            "inside_train_range_by_axis_fraction": {
                axis: count / len(heldout) for axis, count in inside_axis.items()
            },
            "range_intersection_over_union": {
                "x": range_overlap_coefficient(train, heldout, "fixed_x_m"),
                "y": range_overlap_coefficient(train, heldout, "fixed_y_m"),
                "z": range_overlap_coefficient(train, heldout, "fixed_z_m"),
            },
        }
    return output


def registry_coverage(
    records: Sequence[FrameRecord], registry_payload: Mapping[str, Any]
) -> dict[str, Any]:
    tolerances = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
    global_ranges = {
        "x": (
            min(record.fixed_x_m for record in records),
            max(record.fixed_x_m for record in records),
        ),
        "y": (
            min(record.fixed_y_m for record in records),
            max(record.fixed_y_m for record in records),
        ),
        "z": (
            min(record.fixed_z_m for record in records),
            max(record.fixed_z_m for record in records),
        ),
    }
    output: dict[str, Any] = {
        "distance": "sqrt(sum(log(observed_axis / target_axis)^2)) over fixed x/y/z",
        "relative_tolerance_rule": "abs(observed_axis / target_axis - 1) <= tolerance on every fixed axis",
        "targets": {},
    }
    for item in registry_payload.get("objects", []):
        dimensions = item.get("physical_dimensions_m", {})
        target = {
            axis: finite_float(dimensions.get(axis), f"registry.{item.get('object_type')}.{axis}")
            for axis in ("x", "y", "z")
        }
        best_record: FrameRecord | None = None
        best_distance = math.inf
        counts = {tolerance: Counter() for tolerance in tolerances}
        exact_count = 0
        for record in records:
            observed = (record.fixed_x_m, record.fixed_y_m, record.fixed_z_m)
            target_tuple = (target["x"], target["y"], target["z"])
            current_distance = math.sqrt(
                sum(math.log(value / reference) ** 2 for value, reference in zip(observed, target_tuple))
            )
            if current_distance < best_distance:
                best_distance = current_distance
                best_record = record
            if observed == target_tuple:
                exact_count += 1
            relative_errors = tuple(
                abs(value / reference - 1.0) for value, reference in zip(observed, target_tuple)
            )
            for tolerance in tolerances:
                if all(error <= tolerance for error in relative_errors):
                    counts[tolerance]["ALL"] += 1
                    counts[tolerance][record.split] += 1
        require(best_record is not None, "registry coverage has no source records")
        nearest_dimensions = {
            "x": best_record.fixed_x_m,
            "y": best_record.fixed_y_m,
            "z": best_record.fixed_z_m,
        }
        output["targets"][str(item["object_type"])] = {
            "aliases": item.get("aliases", []),
            "target_physical_dimensions_m": target,
            "target_inside_observed_global_range_by_axis": {
                axis: global_ranges[axis][0] <= target[axis] <= global_ranges[axis][1]
                for axis in ("x", "y", "z")
            },
            "exact_triplet_count": exact_count,
            "within_relative_tolerance_counts": {
                f"{tolerance:.2f}": {
                    split: counts[tolerance][split]
                    for split in ("ALL", "TRAIN", "DEV", "TEST")
                }
                for tolerance in tolerances
            },
            "nearest": {
                "frame_id": best_record.frame_id,
                "split": best_record.split,
                "source_asset": best_record.source_asset,
                "fixed_renderer_dimensions_m": nearest_dimensions,
                "relative_error_by_axis": {
                    axis: abs(nearest_dimensions[axis] / target[axis] - 1.0)
                    for axis in ("x", "y", "z")
                },
                "log_l2_distance": best_distance,
            },
        }
    return output


def essential_record_payload(record: FrameRecord) -> dict[str, Any]:
    return {
        "frame_id": record.frame_id,
        "current_g38_split": record.current_g38_split,
        "new_split": record.split,
        "source_asset": record.source_asset,
        "topology_proxy": record.topology_proxy,
        "camera_facing_whd_m": [
            record.camera_width_m,
            record.camera_height_m,
            record.camera_depth_m,
        ],
        "fixed_xyz_m": [record.fixed_x_m, record.fixed_y_m, record.fixed_z_m],
        "parity": record.parity,
        "explicit_parity": record.explicit_parity,
        "background_id": record.background_id,
        "scene_preset": record.scene_preset,
        "elevation_deg_actual": record.elevation_deg_actual,
        "azimuth_deg_target": record.azimuth_deg_target,
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[4]
    source_root = repo / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
    labels_dir = source_root / "labels"
    rgb_dir = source_root / "rgb"
    g38_dataset = repo / "challenge/yolo_pose_one_model/datasets/g38_generic_only"
    g38_contract_path = repo / "challenge/yolo_pose_one_model/runs_posecls_g38/DATA_CONTRACT.json"
    g38_manifest_path = (
        repo
        / "challenge/yolo_pose_one_model/runs_camera_facing_loss"
        / "ubuntu_cf_loss_queue_20260823T0930/G38_GENERIC_ONLY_MANIFEST.json"
    )
    registry_path = repo / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
    asset_audit_path = (
        repo
        / "challenge/yolo_pose_one_model/broad_family_v2"
        / "TARGET_ASSET_EXCLUSION_AUDIT_V2.json"
    )
    asset_csv_path = (
        repo
        / "challenge/yolo_pose_one_model/broad_family_v2"
        / "CURRENT_ASSET_FAMILY_AUDIT.csv"
    )

    required_paths = (
        labels_dir,
        rgb_dir,
        g38_dataset,
        g38_contract_path,
        g38_manifest_path,
        registry_path,
        asset_audit_path,
        asset_csv_path,
    )
    for path in required_paths:
        require(path.exists(), f"required source does not exist: {path}")

    g38_contract = load_json(g38_contract_path)
    g38_manifest = load_json(g38_manifest_path)
    registry = load_json(registry_path)
    asset_audit = load_json(asset_audit_path)

    current_split_by_sample: dict[str, str] = {}
    for split, key in (("train", "train"), ("val", "val")):
        values = g38_manifest.get(key)
        require(isinstance(values, list), f"G38 manifest {key} is not a list")
        for sample_id in values:
            require(sample_id not in current_split_by_sample, f"duplicate G38 sample: {sample_id}")
            current_split_by_sample[str(sample_id)] = split

    label_paths: list[tuple[int, Path]] = []
    for path in labels_dir.glob("f*_label.json"):
        match = FRAME_RE.fullmatch(path.name)
        require(match is not None, f"unexpected renderer label filename: {path.name}")
        label_paths.append((int(match.group("index")), path))
    label_paths.sort(key=lambda item: item[0])

    declared_total = int(g38_contract["train_declared"]) + int(g38_contract["val_declared"])
    require(len(label_paths) == declared_total, "renderer JSON count does not match G38 contract")
    require(len(current_split_by_sample) == declared_total, "G38 manifest count mismatch")

    asset_csv_rows: dict[str, dict[str, str]] = {}
    with asset_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            asset_csv_rows[str(row["source_asset"])] = dict(row)

    availability = Counter()
    explicit_parity_counts = Counter()
    reconstructed_parity_counts = Counter()
    axis_case_counts = Counter()
    permutation_counts = Counter()
    asset_name_pairs = Counter()
    current_asset_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    maxima = Counter()
    minimum_abs_width_depth_margin = math.inf
    essential_digest = hashlib.sha256()
    records: list[FrameRecord] = []

    for frame_index, label_path in label_paths:
        frame_id = f"f{frame_index:04d}" if frame_index < 10000 else f"f{frame_index}"
        require(label_path.name == f"{frame_id}_label.json", f"noncanonical label name: {label_path.name}")
        sample_id = f"G__{frame_id}"
        require(sample_id in current_split_by_sample, f"{sample_id}: absent from G38 manifest")
        current_split = current_split_by_sample[sample_id]

        source_rgb_path = rgb_dir / f"{frame_id}_rgb.png"
        g38_image_path = g38_dataset / "images" / current_split / f"{sample_id}.png"
        g38_label_path = g38_dataset / "labels" / current_split / f"{sample_id}.txt"
        require(source_rgb_path.exists(), f"{frame_id}: source RGB is missing")
        require(g38_image_path.exists(), f"{frame_id}: G38 image is missing or broken")
        require(g38_label_path.exists(), f"{frame_id}: G38 YOLO label is missing or broken")
        availability["source_renderer_json"] += 1
        availability["source_rgb"] += 1
        availability["g38_image"] += 1
        availability["g38_yolo_label"] += 1

        payload = load_json(label_path)
        objects = payload.get("objects")
        require(isinstance(objects, list) and len(objects) == 1, f"{frame_id}: expected one object")
        obj = objects[0]
        require(isinstance(obj, dict), f"{frame_id}: object is not a JSON object")

        dimensions = obj.get("dimensions_m")
        require(isinstance(dimensions, dict), f"{frame_id}: dimensions_m is missing")
        width = finite_float(dimensions.get("width"), f"{frame_id}.dimensions_m.width")
        height = finite_float(dimensions.get("height"), f"{frame_id}.dimensions_m.height")
        depth = finite_float(dimensions.get("depth"), f"{frame_id}.dimensions_m.depth")
        require(min(width, height, depth) > 0.0, f"{frame_id}: dimensions are not positive")
        availability["camera_facing_dimensions_m"] += 1
        if obj.get("physical_dimensions_m") is not None:
            availability["explicit_physical_dimensions_m"] += 1

        permutation_value = obj.get("perm_v4")
        require(
            isinstance(permutation_value, list)
            and len(permutation_value) == 8
            and all(isinstance(value, int) and not isinstance(value, bool) for value in permutation_value),
            f"{frame_id}: perm_v4 is not eight integers",
        )
        permutation = tuple(int(value) for value in permutation_value)
        require(sorted(permutation) == list(range(8)), f"{frame_id}: perm_v4 is not bijective")
        availability["perm_v4_bijection"] += 1
        permutation_counts[str(permutation)] += 1

        mapped_width_edge = frozenset((permutation[0], permutation[1]))
        if mapped_width_edge in FIXED_WIDTH_EDGES:
            fixed_x, fixed_y, fixed_z = width, height, depth
            axis_case = "CAMERA_WIDTH_IS_FIXED_X"
        elif mapped_width_edge in FIXED_DEPTH_EDGES:
            fixed_x, fixed_y, fixed_z = depth, height, width
            axis_case = "CAMERA_WIDTH_IS_FIXED_Z"
        else:
            raise AuditError(f"{frame_id}: camera width edge does not map to fixed X or Z")
        axis_case_counts[axis_case] += 1
        availability["fixed_renderer_dimensions_reconstructed"] += 1

        cuboid = finite_matrix(obj.get("cuboid"), 8, 3, f"{frame_id}.cuboid")
        raw_width = distance(cuboid[0], cuboid[1])
        raw_depth = distance(cuboid[0], cuboid[4])
        raw_height = distance(cuboid[0], cuboid[3])
        maxima["camera_width_vs_cuboid_abs_m"] = max(
            maxima["camera_width_vs_cuboid_abs_m"], abs(width - raw_width)
        )
        maxima["camera_depth_vs_cuboid_abs_m"] = max(
            maxima["camera_depth_vs_cuboid_abs_m"], abs(depth - raw_depth)
        )
        maxima["camera_height_vs_cuboid_abs_m"] = max(
            maxima["camera_height_vs_cuboid_abs_m"], abs(height - raw_height)
        )

        fixed_cuboid: list[tuple[float, ...] | None] = [None] * 8
        for camera_index, fixed_index in enumerate(permutation):
            fixed_cuboid[fixed_index] = cuboid[camera_index]
        require(all(point is not None for point in fixed_cuboid), f"{frame_id}: inverse permutation failed")
        fixed_points = [point for point in fixed_cuboid if point is not None]
        fixed_width = distance(fixed_points[0], fixed_points[1])
        fixed_depth = distance(fixed_points[0], fixed_points[4])
        fixed_height = distance(fixed_points[0], fixed_points[3])
        maxima["fixed_x_vs_unpermuted_cuboid_abs_m"] = max(
            maxima["fixed_x_vs_unpermuted_cuboid_abs_m"], abs(fixed_x - fixed_width)
        )
        maxima["fixed_z_vs_unpermuted_cuboid_abs_m"] = max(
            maxima["fixed_z_vs_unpermuted_cuboid_abs_m"], abs(fixed_z - fixed_depth)
        )
        maxima["fixed_y_vs_unpermuted_cuboid_abs_m"] = max(
            maxima["fixed_y_vs_unpermuted_cuboid_abs_m"], abs(fixed_y - fixed_height)
        )

        parity = "LONG_FRONT" if width > depth else "SHORT_FRONT"
        minimum_abs_width_depth_margin = min(
            minimum_abs_width_depth_margin, abs(width - depth)
        )
        reconstructed_parity_counts[parity] += 1
        availability["parity_reconstructed_from_stored_dimensions"] += 1
        explicit_block = obj.get("efront_kp12")
        explicit_parity: str | None
        if explicit_block is None:
            explicit_parity = None
            availability["efront_kp12_null"] += 1
        else:
            require(isinstance(explicit_block, dict), f"{frame_id}: efront_kp12 is malformed")
            require(explicit_block.get("kp12_valid") is True, f"{frame_id}: explicit parity is not valid")
            front_type = explicit_block.get("front_face_type")
            require(front_type in ("short", "long"), f"{frame_id}: invalid front_face_type")
            explicit_parity = "LONG_FRONT" if front_type == "long" else "SHORT_FRONT"
            require(explicit_parity == parity, f"{frame_id}: reconstructed parity mismatch")
            availability["explicit_renderer_parity"] += 1
            availability["explicit_renderer_parity_agrees"] += 1
            explicit_parity_counts[explicit_parity] += 1

        source_asset = obj.get("source_asset")
        topology_proxy = obj.get("name")
        require(source_asset in ASSET_TO_SPLIT, f"{frame_id}: unknown source_asset {source_asset!r}")
        require(
            topology_proxy == ASSET_TO_TOPOLOGY_PROXY[source_asset],
            f"{frame_id}: source_asset/name mapping changed",
        )
        split = ASSET_TO_SPLIT[source_asset]
        availability["source_asset"] += 1
        availability["name_topology_proxy"] += 1
        if any(obj.get(field) is not None for field in ("topology_family", "mesh_family", "asset_family")):
            availability["explicit_topology_family"] += 1
        asset_name_pairs[(str(source_asset), str(topology_proxy))] += 1
        current_asset_by_split[current_split][str(source_asset)] += 1

        pose = finite_matrix(obj.get("pose_transform"), 4, 4, f"{frame_id}.pose_transform")
        require(
            max(abs(pose[3][index] - expected) for index, expected in enumerate((0.0, 0.0, 0.0, 1.0)))
            <= ROTATION_TOLERANCE,
            f"{frame_id}: pose_transform last row is invalid",
        )
        rotation = tuple(row[:3] for row in pose[:3])
        current_orthogonality = orthogonality_error(rotation)
        current_determinant_error = abs(determinant_3x3(rotation) - 1.0)
        require(current_orthogonality <= ROTATION_TOLERANCE, f"{frame_id}: rotation is not orthonormal")
        require(current_determinant_error <= ROTATION_TOLERANCE, f"{frame_id}: rotation determinant is not +1")
        maxima["pose_rotation_orthogonality_error"] = max(
            maxima["pose_rotation_orthogonality_error"], current_orthogonality
        )
        maxima["pose_rotation_det_abs_error"] = max(
            maxima["pose_rotation_det_abs_error"], current_determinant_error
        )
        finite_vector(obj.get("quaternion_xyzw"), 4, f"{frame_id}.quaternion_xyzw")
        euler = obj.get("euler_angles")
        require(isinstance(euler, dict), f"{frame_id}: euler_angles is missing")
        for axis in ("pitch", "yaw", "roll"):
            finite_float(euler.get(axis), f"{frame_id}.euler_angles.{axis}")
        availability["object_to_camera_pose_transform"] += 1
        availability["quaternion_xyzw"] += 1
        availability["euler_angles"] += 1

        camera = payload.get("camera_data")
        require(isinstance(camera, dict), f"{frame_id}: camera_data is missing")
        intrinsics = camera.get("intrinsics")
        require(isinstance(intrinsics, dict), f"{frame_id}: camera intrinsics are missing")
        for key in ("fx", "fy", "cx", "cy"):
            finite_float(intrinsics.get(key), f"{frame_id}.camera_data.intrinsics.{key}")
        background_id = camera.get("background_asset")
        scene_preset = camera.get("scene_preset")
        require(isinstance(background_id, str) and background_id, f"{frame_id}: background missing")
        require(isinstance(scene_preset, str) and scene_preset, f"{frame_id}: scene preset missing")
        availability["camera_intrinsics"] += 1
        availability["background_id"] += 1
        availability["scene_preset"] += 1

        v2_labels = obj.get("v2_labels")
        require(isinstance(v2_labels, dict), f"{frame_id}: v2_labels is missing")
        elevation = finite_float(
            v2_labels.get("elevation_deg_actual"), f"{frame_id}.elevation_deg_actual"
        )
        azimuth = finite_float(
            v2_labels.get("azimuth_deg_target"), f"{frame_id}.azimuth_deg_target"
        )
        availability["elevation_deg_actual"] += 1
        availability["azimuth_deg_target"] += 1
        require(
            obj.get("keypoint_convention") == "camera_dynamic_0123_v4",
            f"{frame_id}: unexpected keypoint convention",
        )
        availability["camera_dynamic_0123_v4"] += 1

        record = FrameRecord(
            frame_id=frame_id,
            frame_index=frame_index,
            sample_id=sample_id,
            current_g38_split=current_split,
            split=split,
            source_asset=str(source_asset),
            topology_proxy=str(topology_proxy),
            camera_width_m=width,
            camera_height_m=height,
            camera_depth_m=depth,
            fixed_x_m=fixed_x,
            fixed_y_m=fixed_y,
            fixed_z_m=fixed_z,
            aspect_x_z=fixed_x / fixed_z,
            parity=parity,
            explicit_parity=explicit_parity,
            axis_case=axis_case,
            background_id=background_id,
            scene_preset=scene_preset,
            elevation_deg_actual=elevation,
            azimuth_deg_target=azimuth,
        )
        records.append(record)
        essential_digest.update(stable_json_bytes(essential_record_payload(record)))

    require(len(records) == declared_total, "not every renderer frame was audited")
    require(len({record.frame_id for record in records}) == len(records), "duplicate frame IDs")
    require(set(current_split_by_sample) == {record.sample_id for record in records}, "manifest coverage mismatch")
    # Keep audited absences visible rather than allowing Counter's sparse JSON
    # representation to omit the field entirely.
    availability.setdefault("explicit_physical_dimensions_m", 0)
    availability.setdefault("explicit_topology_family", 0)
    require(
        max(maxima[key] for key in (
            "camera_width_vs_cuboid_abs_m",
            "camera_depth_vs_cuboid_abs_m",
            "camera_height_vs_cuboid_abs_m",
            "fixed_x_vs_unpermuted_cuboid_abs_m",
            "fixed_y_vs_unpermuted_cuboid_abs_m",
            "fixed_z_vs_unpermuted_cuboid_abs_m",
        )) <= CUBOID_TOLERANCE_M,
        "dimension/cuboid reconstruction exceeds tolerance",
    )

    records.sort(key=lambda record: record.frame_index)
    records_by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLIT_ORDER
    }
    assets_by_split = {
        split: sorted(asset for asset, assigned in ASSET_TO_SPLIT.items() if assigned == split)
        for split in SPLIT_ORDER
    }
    asset_sets = {split: set(assets) for split, assets in assets_by_split.items()}
    pairwise_asset_overlap = {
        f"{first}_{second}": sorted(asset_sets[first] & asset_sets[second])
        for index, first in enumerate(SPLIT_ORDER)
        for second in SPLIT_ORDER[index + 1 :]
    }
    require(not any(pairwise_asset_overlap.values()), "new split has source_asset overlap")
    require(
        set(ASSET_TO_SPLIT) == {record.source_asset for record in records},
        "asset assignment does not cover exactly the observed assets",
    )
    require(
        all(parity_summary(records_by_split[split])["within_40_60_gate"] for split in SPLIT_ORDER),
        "one or more new splits violates the 40-60 parity gate",
    )

    registry_result = registry_coverage(records, registry)
    uncovered_registry_targets: list[str] = []
    for object_type, coverage in registry_result["targets"].items():
        in_range = coverage["target_inside_observed_global_range_by_axis"]
        within_20 = coverage["within_relative_tolerance_counts"]["0.20"]["ALL"]
        if not all(in_range.values()) or within_20 == 0:
            uncovered_registry_targets.append(object_type)
    new_render_required = bool(uncovered_registry_targets)
    new_render_assessment = {
        "new_render_required": new_render_required,
        "verdict": (
            "DIM_PROBE_SYNTH_REQUIRED_FOR_MULTISHAPE_REGISTRY_COVERAGE"
            if new_render_required
            else "EXISTING_G38_COVERS_REGISTERED_OBJECT_NEIGHBORHOODS"
        ),
        "criterion": (
            "Each registered target must lie inside the observed fixed-axis range on x/y/z "
            "and have at least one G38 frame within 20% on every fixed axis."
        ),
        "uncovered_registry_object_types": uncovered_registry_targets,
        "interpretation": (
            "G38 has abundant nonconstant dimensions and exact parity supervision, but it "
            "does not cover every registered object neighborhood. A new render is required "
            "before a multishape/plastic+wood dimension-generalization claim."
            if new_render_required
            else "Existing G38 satisfies the registered-neighborhood coverage audit."
        ),
        "non_fabrication_note": (
            "No DIM_PROBE_SYNTH frames, asset IDs, backgrounds, viewpoints, or memberships "
            "are emitted here because such renderer metadata does not yet exist."
        ),
    }

    resolved_assets = asset_audit.get("resolved", {})
    unresolved_assets = asset_audit.get("unresolved", {})
    asset_details: dict[str, Any] = {}
    for asset in sorted(ASSET_TO_SPLIT):
        source_rows = [record for record in records if record.source_asset == asset]
        require(asset in asset_csv_rows, f"asset audit CSV is missing {asset}")
        resolved = asset in resolved_assets
        unresolved = asset in unresolved_assets
        require(resolved != unresolved, f"asset resolution status is not singular for {asset}")
        asset_details[asset] = {
            "split": ASSET_TO_SPLIT[asset],
            "topology_proxy": ASSET_TO_TOPOLOGY_PROXY[asset],
            "frame_count": len(source_rows),
            "mesh_resolved": resolved,
            "mesh_canonical_vertex_sha256": (
                resolved_assets[asset].get("canonical_vertex_sha256") if resolved else None
            ),
            "unresolved_reason": unresolved_assets[asset].get("reason") if unresolved else None,
            "parity": parity_summary(source_rows),
        }

    current_train_assets = set(current_asset_by_split["train"])
    current_val_assets = set(current_asset_by_split["val"])
    current_overlap = sorted(current_train_assets & current_val_assets)

    split_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_split_membership",
        "status": "PROVISIONAL_ASSET_ID_DISJOINT",
        "topology_certified": False,
        "policy": {
            "unit": "source_asset exact string",
            "assignment": dict(sorted(ASSET_TO_SPLIT.items())),
            "topology_proxy_field": "objects[0].name",
            "selection_rule": (
                "Frozen literal assignment. It is also the 2-family TRAIN pair maximizing "
                "the minimum heldout fraction inside TRAIN's fixed-xyz min/max box among "
                "the six 2/1/1 source_asset assignments; lexicographic heldout assignment "
                "gives DEV=scene.usd and TEST=scene_1.usd."
            ),
            "parity_rule": "LONG_FRONT iff stored dimensions_m.width > dimensions_m.depth; otherwise SHORT_FRONT",
            "dimension_rule": (
                "Map camera width edge through perm_v4; use (W,H,D) when it is a fixed-width "
                "edge and (D,H,W) when it is a fixed-depth edge."
            ),
            "random_frame_split": False,
        },
        "source": {
            "renderer_labels": relative_path(labels_dir, repo),
            "renderer_rgb": relative_path(rgb_dir, repo),
            "current_g38_dataset": relative_path(g38_dataset, repo),
            "current_g38_manifest": relative_path(g38_manifest_path, repo),
            "essential_metadata_sha256": essential_digest.hexdigest(),
        },
        "counts": {
            "all": len(records),
            **{split.lower(): len(records_by_split[split]) for split in SPLIT_ORDER},
        },
        "splits": {},
        "split_details": {},
        "frame_to_split": {
            record.frame_id: record.split for record in records
        },
        "frame_to_parity": {
            record.frame_id: record.parity for record in records
        },
        "integrity": {
            "all_source_frames_preserved": sum(len(value) for value in records_by_split.values()) == len(records),
            "frame_ids_unique": len({record.frame_id for record in records}) == len(records),
            "source_asset_pairwise_overlap": pairwise_asset_overlap,
            "source_asset_overlap_zero": not any(pairwise_asset_overlap.values()),
            "parity_40_60_each_split": all(
                parity_summary(records_by_split[split])["within_40_60_gate"]
                for split in SPLIT_ORDER
            ),
            "explicit_topology_family_available": availability["explicit_topology_family"] == len(records),
            "topology_certified": False,
        },
        "limitation": (
            "The split is exact source_asset-ID disjoint. It is not topology-certified: "
            "eur_pallet_bk_cc0.glb and woodpallet_block_jtoastie_ccby.glb are absent on this "
            "machine and have no audited mesh hash."
        ),
    }
    for split in SPLIT_ORDER:
        split_records = records_by_split[split]
        frame_ids = [record.frame_id for record in split_records]
        split_payload["splits"][split] = frame_ids
        split_payload["split_details"][split] = {
            "source_assets": assets_by_split[split],
            "topology_proxies": sorted({record.topology_proxy for record in split_records}),
            "count": len(split_records),
            "parity": parity_summary(split_records),
            "current_g38_origin_counts": dict(
                sorted(Counter(record.current_g38_split for record in split_records).items())
            ),
            "parity_frame_ids": {
                parity: [record.frame_id for record in split_records if record.parity == parity]
                for parity in PARITY_ORDER
            },
        }

    membership_path = output_dir / "SPLIT_MEMBERSHIP.json"
    write_json(membership_path, split_payload)

    sha_lines = [f"{sha256_file(membership_path)}  SPLIT_MEMBERSHIP.json"]
    for split in SPLIT_ORDER:
        ids = split_payload["splits"][split]
        sha_lines.append(
            f"{sha256_bytes((''.join(frame_id + chr(10) for frame_id in ids)).encode('utf-8'))}  {split}.frame_ids"
        )
        for parity in PARITY_ORDER:
            parity_ids = split_payload["split_details"][split]["parity_frame_ids"][parity]
            sha_lines.append(
                f"{sha256_bytes((''.join(frame_id + chr(10) for frame_id in parity_ids)).encode('utf-8'))}  {split}.{parity}.frame_ids"
            )
    sha_lines.append(f"{essential_digest.hexdigest()}  source.essential_metadata")
    (output_dir / "SPLIT_MEMBERSHIP_SHA256.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )

    background_by_split = {
        split: {
            background: parity_summary(
                [record for record in records_by_split[split] if record.background_id == background]
            )
            for background in sorted({record.background_id for record in records_by_split[split]})
        }
        for split in SPLIT_ORDER
    }
    scene_by_split = {
        split: dict(sorted(Counter(record.scene_preset for record in records_by_split[split]).items()))
        for split in SPLIT_ORDER
    }
    viewpoint_by_split = {
        split: {
            "elevation_deg_actual": numeric_summary(
                record.elevation_deg_actual for record in records_by_split[split]
            ),
            "azimuth_deg_target": numeric_summary(
                record.azimuth_deg_target for record in records_by_split[split]
            ),
        }
        for split in SPLIT_ORDER
    }

    distribution_payload = {
        "schema_version": f"{SCHEMA_VERSION}_dimension_distribution",
        "status": "PASS_NONCONSTANT_DIMENSIONS_WITH_WOOD_COVERAGE_GAP",
        "dimension_semantics": {
            "input_dimensions_m": "camera-facing W/H/D",
            "reported_dimensions": "renderer fixed-object x/y/z reconstructed with perm_v4",
            "fixed_axis_order": ["x_width", "y_height", "z_depth"],
            "raw_camera_dimensions_for_dims_only_forbidden": True,
            "reason": "Stored camera-facing W versus D determines the parity label exactly.",
        },
        "global": dimension_summary(records),
        "by_split": {
            split: dimension_summary(records_by_split[split]) for split in SPLIT_ORDER
        },
        "by_source_asset": grouped_summaries(records, "source_asset"),
        "by_background": grouped_summaries(records, "background_id"),
        "background_parity_by_split": background_by_split,
        "scene_preset_counts_by_split": scene_by_split,
        "viewpoint_by_split": viewpoint_by_split,
        "eta_squared_categorical_association": {
            "definition": "between-group sum of squares / total sum of squares",
            "source_asset": {
                "fixed_x": eta_squared(records, "source_asset", "fixed_x_m"),
                "fixed_y": eta_squared(records, "source_asset", "fixed_y_m"),
                "fixed_z": eta_squared(records, "source_asset", "fixed_z_m"),
                "fixed_x_over_z": eta_squared(records, "source_asset", "aspect_x_z"),
            },
            "background_id": {
                "fixed_x": eta_squared(records, "background_id", "fixed_x_m"),
                "fixed_y": eta_squared(records, "background_id", "fixed_y_m"),
                "fixed_z": eta_squared(records, "background_id", "fixed_z_m"),
                "fixed_x_over_z": eta_squared(records, "background_id", "aspect_x_z"),
            },
        },
        "split_dimension_overlap": split_overlap(records_by_split),
        "real_registry_coverage": registry_result,
        "new_render_assessment": new_render_assessment,
    }
    write_json(output_dir / "DIMENSION_DISTRIBUTION.json", distribution_payload)

    source_hashes = {
        relative_path(path, repo): sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            g38_contract_path,
            g38_manifest_path,
            registry_path,
            asset_audit_path,
            asset_csv_path,
        )
    }
    explicit_count = availability["explicit_renderer_parity"]
    hard_block = {
        "physical_dimensions_constant": dimension_summary(records)["unique_fixed_xyz_exact"] == 1,
        "parity_not_exactly_reconstructable": explicit_count == 0
        or availability["explicit_renderer_parity_agrees"] != explicit_count
        or availability["parity_reconstructed_from_stored_dimensions"] != len(records),
    }
    hard_block["dimension_supervision_not_auditable"] = bool(any(hard_block.values()))

    audit_payload = {
        "schema_version": f"{SCHEMA_VERSION}_synth_metadata_audit",
        "status": "PASS_PROVISIONAL_ASSET_ID_DISJOINT",
        "hard_block": hard_block,
        "population": {
            "source_renderer_frames": len(records),
            "objects": len(records),
            "current_g38_train": Counter(record.current_g38_split for record in records)["train"],
            "current_g38_val": Counter(record.current_g38_split for record in records)["val"],
            "new_train": len(records_by_split["TRAIN"]),
            "new_dev": len(records_by_split["DEV"]),
            "new_test": len(records_by_split["TEST"]),
            "all_frames_preserved": True,
        },
        "source": {
            "paths": {
                "renderer_labels": relative_path(labels_dir, repo),
                "renderer_rgb": relative_path(rgb_dir, repo),
                "current_g38_dataset": relative_path(g38_dataset, repo),
                "current_g38_contract": relative_path(g38_contract_path, repo),
                "current_g38_manifest": relative_path(g38_manifest_path, repo),
                "object_geometry_registry": relative_path(registry_path, repo),
                "asset_resolution_audit": relative_path(asset_audit_path, repo),
            },
            "sha256": source_hashes,
            "essential_metadata_sha256": essential_digest.hexdigest(),
            "records_jsonl_policy": (
                "Not used. Per-frame renderer JSON is authoritative; records.jsonl has "
                "non-unique idx values and inconsistent zero padding."
            ),
        },
        "metadata_availability_counts": dict(sorted(availability.items())),
        "metadata_semantics": {
            "physical_dimensions": {
                "explicit_physical_dimensions_m_count": availability["explicit_physical_dimensions_m"],
                "camera_facing_dimensions_m_count": availability["camera_facing_dimensions_m"],
                "fixed_renderer_dimensions_reconstructed_count": availability[
                    "fixed_renderer_dimensions_reconstructed"
                ],
                "note": (
                    "The renderer JSON does not carry a canonical physical_dimensions_m field. "
                    "Fixed renderer x/y/z is reconstructed exactly from stored camera-facing "
                    "dimensions_m and perm_v4; no real-object type is assigned to a G38 frame."
                ),
            },
            "asset": "objects[0].source_asset",
            "topology": {
                "explicit_topology_family_count": availability["explicit_topology_family"],
                "proxy_field": "objects[0].name",
                "proxy_count": availability["name_topology_proxy"],
                "certified": False,
            },
            "object_to_camera_rotation": {
                "field": "objects[0].pose_transform[:3][:3]",
                "count": availability["object_to_camera_pose_transform"],
                "evidence": "scripts/stage0/multihead/mh_cigm.py documents pose_transform as object-to-camera",
                "additional_fields": ["quaternion_xyzw", "euler_angles"],
            },
            "parity": {
                "explicit_field": "objects[0].efront_kp12.front_face_type",
                "explicit_count": explicit_count,
                "explicit_null_count": availability["efront_kp12_null"],
                "restoration_rule": "LONG_FRONT iff dimensions_m.width > dimensions_m.depth",
                "restored_count": availability["parity_reconstructed_from_stored_dimensions"],
                "explicit_agreement_count": availability["explicit_renderer_parity_agrees"],
                "minimum_abs_stored_width_minus_depth_m": minimum_abs_width_depth_margin,
                "rounding_forbidden": True,
            },
            "background": "camera_data.background_asset",
            "viewpoint": [
                "objects[0].v2_labels.elevation_deg_actual",
                "objects[0].v2_labels.azimuth_deg_target",
            ],
        },
        "dimension_reconstruction_contract": {
            "perm_direction": "perm_v4[new_camera_facing_index] = old_fixed_index",
            "inverse_assignment": "fixed[perm_v4[i]] = camera_facing[i] for i=0..7",
            "fixed_width_edges_unordered": [
                sorted(edge) for edge in sorted(FIXED_WIDTH_EDGES, key=lambda item: tuple(sorted(item)))
            ],
            "fixed_depth_edges_unordered": [
                sorted(edge) for edge in sorted(FIXED_DEPTH_EDGES, key=lambda item: tuple(sorted(item)))
            ],
            "formula": {
                "if_mapped_camera_width_is_fixed_width": "(x,y,z)=(W,H,D)",
                "if_mapped_camera_width_is_fixed_depth": "(x,y,z)=(D,H,W)",
                "otherwise": "HARD_BLOCK",
            },
            "axis_case_counts": dict(sorted(axis_case_counts.items())),
            "permutation_class_counts": dict(sorted(permutation_counts.items())),
        },
        "validation": {
            "cuboid_tolerance_m": CUBOID_TOLERANCE_M,
            "max_abs_errors": dict(sorted(maxima.items())),
            "perm_v4_bijection_count": availability["perm_v4_bijection"],
            "axis_assignment_classified_count": sum(axis_case_counts.values()),
            "explicit_parity_counts": {
                label: explicit_parity_counts[label] for label in PARITY_ORDER
            },
            "reconstructed_parity_counts": {
                label: reconstructed_parity_counts[label] for label in PARITY_ORDER
            },
            "explicit_parity_mismatch_count": explicit_count
            - availability["explicit_renderer_parity_agrees"],
        },
        "asset_and_topology": {
            "split_status": "PROVISIONAL_ASSET_ID_DISJOINT",
            "topology_certified": False,
            "asset_name_one_to_one": len(asset_name_pairs) == len(ASSET_TO_SPLIT),
            "asset_name_pair_counts": {
                f"{asset}|{name}": count
                for (asset, name), count in sorted(asset_name_pairs.items())
            },
            "assets": asset_details,
            "current_g38_train_val_source_asset_overlap": current_overlap,
            "current_g38_train_val_source_asset_overlap_count": len(current_overlap),
            "new_split_source_asset_pairwise_overlap": pairwise_asset_overlap,
            "new_split_source_asset_overlap_zero": not any(pairwise_asset_overlap.values()),
            "limitation": (
                "Two GLBs have no mesh hash on this machine. Exact source_asset separation is "
                "proven; topology-family separation is not."
            ),
        },
        "split": {
            "membership_path": relative_path(membership_path, repo),
            "membership_sha256": sha256_file(membership_path),
            "status": "PROVISIONAL_ASSET_ID_DISJOINT",
            "parity_40_60_each_split": True,
            "dimension_ranges_overlap": split_overlap(records_by_split),
        },
        "real_registry_coverage": registry_result,
        "new_render_assessment": new_render_assessment,
        "non_fabrication_guards": [
            "No parity was inferred from a filename or background.",
            "No explicit topology family was synthesized from source_asset or name.",
            "No G38 frame was assigned a PLASTIC_STANDARD or WOOD_SMALL object type.",
            "No DIM_PROBE_SYNTH frame metadata or membership was invented.",
        ],
    }
    write_json(output_dir / "SYNTH_METADATA_AUDIT.json", audit_payload)

    print(
        json.dumps(
            {
                "status": audit_payload["status"],
                "frames": len(records),
                "split_counts": {
                    split: len(records_by_split[split]) for split in SPLIT_ORDER
                },
                "membership_sha256": sha256_file(membership_path),
                "new_render": new_render_assessment["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
