"""CPU-only canonical DEV319 9-point and reconstructed-pose evaluation.

This stage performs no image/model forward.  It keeps all 319 population rows,
uses the frozen baseline box for every method, counts full-GT PCK failures, and
reports pose coverage instead of silently intersecting successful frames.
"""
from __future__ import annotations

import argparse
import csv
import io
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from . import common as C


SCHEMA = "resnet18_full_dim_refiner_dev_predictions_v1"
BASELINE = "FULL"
METHODS = (BASELINE,) + tuple(C.arm_key(arm, seed)
                             for arm in C.HEAD_ARMS for seed in C.SEEDS)


LEGACY_EVALUATION = C.ROOT / "scripts/research/pallet_resnet18_refiner_20261001_v1/evaluation.py"
RESAMPLES = 10000
BOOT_SEED = 20260914
OBJECT_GROUPS = {
    "plastic": "plastic_standard_110x130x11",
    "wood": "wood_small_80x59x14",
}


def _legacy():
    name = "pallet_full_dim_refiner_canonical_eval_primitives"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LEGACY_EVALUATION)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY_EVALUATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _symmetry_metrics():
    name = "pallet_full_dim_refiner_symmetry_metrics"
    if name in sys.modules:
        return sys.modules[name]
    path = C.POSE_CODE / "symmetry_aware_pose_metrics.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_key(path) -> str:
    value = Path(path)
    if not value.is_absolute():
        value = C.ROOT / value
    return str(value.resolve().relative_to(C.ROOT.resolve()))


def validate_points(value, expected_valid=None):
    points = np.asarray(value, np.float64)
    if points.shape != (9, 2) or np.isinf(points).any():
        raise ValueError("Points must be 9x2 and use paired NaNs for missing values")
    finite = np.isfinite(points)
    if not np.array_equal(finite[:, 0], finite[:, 1]):
        raise ValueError("Partially missing xy pair")
    valid = finite.all(-1)
    if expected_valid is not None and not np.array_equal(valid, np.asarray(expected_valid, bool)):
        raise ValueError("Missing mask drift")
    return points, valid


def method_points(row: dict, method: str):
    return row["base_points"] if method == BASELINE else row["refined"][method]


def validate_canonical_wdh(object_type: str, value, pose_spec: dict,
                           registry_path: Path | str = C.GEOMETRY_REGISTRY) -> np.ndarray:
    """Require the registry's fixed [x,z,y]=[W,D,H] order, not a sorted footprint."""
    registry = C.read_json(registry_path)
    matches = [row for row in registry["objects"] if row["object_type"] == object_type]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate registry object: {object_type}")
    raw = matches[0]["physical_dimensions_m"]
    expected = np.asarray([raw["x"], raw["z"], raw["y"]], np.float64)
    actual = np.asarray(value, np.float64)
    if actual.shape != (3,) or not np.isfinite(actual).all() or not np.array_equal(actual, expected):
        raise ValueError(f"Canonical fixed-axis W,D,H drift: {object_type}")
    pose = np.asarray([pose_spec["long_m"], pose_spec["short_m"], pose_spec["height_m"]],
                      np.float64)
    if (not np.array_equal(np.sort(expected[:2]), np.sort(pose[:2]))
            or expected[2] != pose[2]):
        raise ValueError(f"Registry/pose object geometry drift: {object_type}")
    return actual


def object_subgroups(rows, *, enforce_dev_counts=False) -> dict[str, list]:
    groups = {label: [row for row in rows if row.get("object_type") == object_type]
              for label, object_type in OBJECT_GROUPS.items()}
    known = {object_type for object_type in OBJECT_GROUPS.values()}
    unknown = [row.get("object_type") for row in rows if row.get("object_type") not in known]
    if unknown or sum(map(len, groups.values())) != len(rows):
        raise ValueError(f"Unexpected DEV object group: {sorted(set(unknown))}")
    if enforce_dev_counts and {name: len(value) for name, value in groups.items()} != {
            "plastic": 194, "wood": 125}:
        raise ValueError("DEV plastic/wood subgroup count drift")
    return groups


def dimension_support(rows, source=None) -> dict:
    """Disclose real fixed dimensions relative to source-TRAIN raw/context support."""
    from .data import SourceDimensions
    from .dimensions import FEATURE_NAMES, normalized_context
    source = SourceDimensions() if source is None else source
    train = source.partitions == "train"
    if int(train.sum()) != 55980:
        raise ValueError("Dimension support must use source TRAIN55980 only")
    train_dimensions = np.asarray(source.dimensions[train], np.float64)
    train_context = np.asarray(source.contexts[train], np.float64)
    raw_min, raw_max = train_dimensions.min(0), train_dimensions.max(0)
    context_min, context_max = train_context.min(0), train_context.max(0)
    groups = object_subgroups(rows)
    objects = {}
    for label, selected in groups.items():
        dimensions = {tuple(np.asarray(row["canonical_WDH_m"], np.float64)) for row in selected}
        if len(dimensions) != 1:
            raise ValueError(f"Nonconstant canonical dimensions inside {label}")
        value = np.asarray(next(iter(dimensions)), np.float64)
        context = normalized_context(value, source.normalization).astype(np.float64)
        raw_inside = (value >= raw_min) & (value <= raw_max)
        context_inside = (context >= context_min) & (context <= context_max)
        objects[label] = dict(
            object_type=OBJECT_GROUPS[label], frames=len(selected), canonical_WDH_m=value.tolist(),
            normalized_5D=context.tolist(),
            raw_axis_inside_train_range=dict(zip(("W", "D", "H"), raw_inside.tolist())),
            normalized_feature_inside_train_range=dict(zip(FEATURE_NAMES, context_inside.tolist())),
            outside_raw_axes=[name for name, inside in zip(("W", "D", "H"), raw_inside)
                              if not inside],
            outside_normalized_features=[name for name, inside in zip(FEATURE_NAMES, context_inside)
                                         if not inside],
            outside_train_axiswise_box=not bool(raw_inside.all()),
            outside_train_normalized_axiswise_box=not bool(context_inside.all()),
        )
    return dict(
        policy=("Axiswise support disclosure only; it does not establish density support or "
                "guarantee generalization."),
        source_train_rows=55980,
        source_manifest=C.binding(C.SOURCE_MANIFEST),
        dimension_sidecar=C.binding(C.DIMENSION_SIDECAR),
        normalization=C.binding(C.DIMENSION_NORMALIZATION),
        source_train_raw_WDH_min=raw_min.tolist(),
        source_train_raw_WDH_max=raw_max.tolist(),
        source_train_normalized_5D_min=context_min.tolist(),
        source_train_normalized_5D_max=context_max.tolist(),
        objects=objects,
    )


def pose_summary(values, denominator: int) -> dict:
    """Pose summary with per-object-diameter ADD and explicit failed-frame mass."""
    pose_auc = _symmetry_metrics().pose_auc
    if denominator < 0 or len(values) > denominator:
        raise ValueError("Invalid pose denominator")
    names = dict(rotation_median_deg="rotation_error_deg",
                 translation_median_cm="translation_error_cm",
                 yaw_median_deg="yaw_error_deg", iou3d_median="iou3d")
    normalized = []
    for row in values:
        add, diameter = float(row["add_sym_m"]), float(row["diameter_m"])
        if not np.isfinite(add) or add < 0 or not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("Invalid ADD/diameter value")
        normalized.append(add / diameter)
    out = dict(
        n=len(values), denominator=denominator, failure_count=denominator - len(values),
        coverage=len(values) / denominator if denominator else 0.,
        **{key: float(np.median([row[field] for row in values])) if values else None
           for key, field in names.items()},
    )
    for key, field in (("translation_p90_cm", "translation_error_cm"),
                       ("rotation_p90_deg", "rotation_error_deg"),
                       ("yaw_p90_deg", "yaw_error_deg")):
        out[key] = float(np.quantile([row[field] for row in values], .9)) if values else None
    out["add_sym_auc_conditional"] = pose_auc(normalized, 1.) if normalized else None
    full = normalized + [float("inf")] * (denominator - len(values))
    out["add_sym_auc_full_population"] = pose_auc(full, 1.) if full else None
    out["axis_accuracy"] = float(np.mean([row["axis_correct"] for row in values])) if values else None
    out["add_sym_policy"] = (
        "Each successful ADD-S is divided by its own model diameter; AUC integrates to "
        "0.1 diameter. Full-population AUC includes every pose failure as +inf (zero accuracy).")
    return out


def validate_rows(rows, expected_keys=None):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction rows are empty")
    keys = [row["image_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prediction image")
    if expected_keys is not None and keys != list(expected_keys):
        raise ValueError("DEV membership/order drift")
    for row in rows:
        if row["image_key"] != canonical_key(row["image_key"]):
            raise ValueError("Noncanonical image key")
        base, valid = validate_points(row["base_points"], row["point_valid"])
        if set(row["refined"]) != set(METHODS[1:]):
            raise ValueError("Exact twelve final seeded heads are required")
        if len(row["original_hw"]) != 2 or min(row["original_hw"]) <= 0:
            raise ValueError("Invalid original image shape")
        box = np.asarray(row["box_original"], np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
            raise ValueError("Invalid fixed FULL bbox")
        expected = np.r_[base[:8][valid[:8]].min(0), base[:8][valid[:8]].max(0)]
        if not np.allclose(box, expected, atol=1e-5, rtol=0):
            raise ValueError("Frozen baseline corner hull differs")
        if not np.isfinite(row["score"]):
            raise ValueError("Missing baseline score")
        for method in METHODS[1:]:
            points, _ = validate_points(row["refined"][method], valid)
            if not np.array_equal(points[8], base[8], equal_nan=True):
                raise ValueError("Center point changed")
    return rows


def _load_canonical_modules():
    # This reuses only frozen evaluator/pose/statistics primitives.  The old
    # ResNet schema, arm globals, input loader, and report writer are not used.
    return _legacy().canonical_modules()


def load_inputs(path: Path | str):
    prediction_path = Path(path)
    from .inference import validate_prediction_payload
    payload = validate_prediction_payload(C.read_json(prediction_path))
    if payload.get("schema") != SCHEMA or payload.get("complete") is not True:
        raise ValueError("Incomplete or wrong-schema prediction artifact")
    if payload.get("coordinate_system") != "original_unpadded_pixels":
        raise ValueError("Prediction coordinate-system drift")
    if payload.get("methods") != list(METHODS):
        raise ValueError("Prediction method order drift")
    for key in ("protocol", "cache", "selection", "training", "synthetic_heldout",
                "population", "registry"):
        C.verify_binding(payload[key])
    protocol_path = C.verify_binding(payload["protocol"])
    protocol = C.verify_protocol(protocol_path)
    if protocol.get("schema") != "resnet18_full_dim_local_refiner_protocol_v1":
        raise ValueError("Refiner protocol schema drift")
    for key in ("baseline", "baseline_protocol", "baseline_training_complete"):
        if payload[key] != protocol[key]:
            raise ValueError(f"Prediction {key}/protocol mismatch")
        C.verify_binding(payload[key])
    expected_checkpoints = {C.arm_key(arm, seed)
                            for arm in C.HEAD_ARMS for seed in C.SEEDS}
    if set(payload["checkpoints"]) != expected_checkpoints:
        raise ValueError("Prediction checkpoint set drift")
    for value in payload["checkpoints"].values():
        C.verify_binding(value)
    for value in payload["code"]:
        C.verify_binding(value)
    selected = C.read_json(C.verify_binding(payload["selection"]))
    if selected.get("complete") is not True or selected.get("no_real_selection") is not True:
        raise ValueError("Selection must be synthetic-only and frozen")
    E, M, S, G, B = _load_canonical_modules()
    pair = E.validate_evaluation_request(
        positive_manifest=C.DEV_POS,
        negative_manifest=C.DEV_NEG,
        population_role=E.PopulationRole.DEV,
        allow_unavailable_final=False,
    )
    items = pair.positive.items
    manifest_rows = C.read_json(C.DEV_POS)["items"]
    ids = {row["frame_id"]: row for row in manifest_rows}
    if len(items) != 319 or len(ids) != 319 or len({row["session_id"] for row in manifest_rows}) != 13:
        raise ValueError("Canonical DEV319 population drift")
    keys = [canonical_key(item.image) for item in items]
    rows = validate_rows(payload["frames"], keys)
    pose_manifest_path = C.POSE_RAW / "AXIS_REVIEW_MANIFEST.json"
    pose_manifest = C.read_json(pose_manifest_path)
    pose_by_key = {canonical_key(row["image"]): row for row in pose_manifest["frames_list"]}
    if len(pose_by_key) != 319 or set(pose_by_key) != set(keys):
        raise ValueError("Pose/2D population mismatch")
    pose_gt_path = C.POSE_RAW / "GEOMETRY_RESOLVED_POSE_GT.json"
    pose_gt = C.read_json(pose_gt_path)
    contract_path = C.POSE_RAW / "POSE_EVAL_OBJECT_CONTRACT.json"
    contract = G.load_pose_object_contract(str(contract_path))
    if pose_gt["pose_object_contract_sha256"] != C.sha256(contract_path):
        raise ValueError("Pose object contract drift")
    targets, pose_ids, mapping = {}, {}, []
    source_paths = [C.DEV_POS, C.DEV_NEG, LEGACY_EVALUATION,
                    pose_manifest_path, contract_path, pose_gt_path]
    for item, row in zip(items, rows):
        manifest = ids[item.frame_id]
        pose_meta = pose_by_key[row["image_key"]]
        if (row["frame_id"] != item.frame_id or row["session_id"] != manifest["session_id"]
                or row["object_type"] != manifest["object_type"]
                or pose_meta["object_type"] != manifest["object_type"]):
            raise ValueError("Prediction/manifest identity drift")
        if canonical_key(item.label) != canonical_key(pose_meta["annotation"]):
            raise ValueError("2D/pose annotation path mismatch")
        annotation_path = C.ROOT / pose_meta["annotation"]
        annotation = C.read_json(annotation_path)
        camera = annotation["camera_data"]
        if list(row["original_hw"]) != [camera["height"], camera["width"]]:
            raise ValueError("Image-size drift")
        target = E._legacy_forbidden_target(item)
        raw = camera["intrinsics"]
        K = np.asarray([[raw["fx"], 0., raw["cx"]],
                        [0., raw["fy"], raw["cy"]], [0., 0., 1.]], np.float64)
        if not np.array_equal(target.camera_intrinsics, K):
            raise ValueError("Canonical camera intrinsics drift")
        spec = G.object_spec(contract, pose_meta["object_type"])
        expected_dimensions = validate_canonical_wdh(
            pose_meta["object_type"], row["canonical_WDH_m"], spec)
        targets[item.frame_id] = target
        pose_ids[item.frame_id] = pose_meta["frame_id"]
        source_paths.append(annotation_path)
        mapping.append(dict(
            frame_id=item.frame_id,
            pose_frame_id=pose_meta["frame_id"],
            image_key=row["image_key"],
            session_id=row["session_id"],
            object_type=pose_meta["object_type"],
            camera_intrinsics=K.tolist(),
            dimensions_m=dict(long=spec["long_m"], short=spec["short_m"],
                              height=spec["height_m"]),
            canonical_WDH_m=expected_dimensions.tolist(),
            annotation=C.binding(annotation_path),
        ))
    if sum(int(target.keypoint_supervision_mask.sum()) for target in targets.values()) != 2818:
        raise ValueError("DEV supervised-landmark denominator drift")
    for module in (E, M, S, G):
        source_paths.append(Path(module.__file__))
    source_paths += [
        C.ROOT / "challenge/evaluation_v2/pnp_selector.py",
        C.ROOT / "challenge/evaluation_v2/oriented_iou3d.py",
        C.POSE_CODE / "symmetry_aware_pose_metrics.py",
        C.ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json",
    ]
    sources = [C.binding(value) for value in dict.fromkeys(source_paths)]
    return payload, rows, targets, pose_ids, mapping, sources, (E, M, S, G, B)


def summarize_2d(E, rows, targets, method):
    candidates, top, detailed = [], {}, []
    for row in rows:
        frame_id = row["frame_id"]
        target = targets[frame_id]
        points, valid = validate_points(method_points(row, method), row["point_valid"])
        box = np.asarray(row["box_original"], np.float64)
        overlap = float(E._box_iou(box, target.box_xyxy))
        candidate = E.DetectionCandidate(frame_id, True, float(row["score"]), box, points, overlap)
        candidates.append(candidate)
        top[frame_id] = candidate
        matched = overlap >= .5 and valid.all()
        mask = target.keypoint_supervision_mask
        distances = np.full(9, np.inf)
        distances[valid & mask] = np.linalg.norm(
            points[valid & mask] - target.keypoints_xy[valid & mask], axis=-1)
        errors = distances[mask] if matched else np.empty(0)
        detailed.append(dict(
            frame_id=frame_id,
            session_id=row["session_id"],
            object_type=row["object_type"],
            image_key=row["image_key"],
            box_iou=overlap,
            canonical_matched=bool(matched),
            gt_count=int(mask.sum()),
            finite_predicted_keypoints=int(valid.sum()),
            missing_supervised_keypoints=int((mask & ~valid).sum()),
            errors=errors,
            errors8=distances[:8][mask[:8]] if matched else np.empty(0),
            canonical_hits={str(threshold): int((errors <= threshold).sum())
                            for threshold in (5, 10, 20)},
            partial_point_hits={str(threshold): int(((distances <= threshold) & mask).sum())
                                if overlap >= .5 else 0 for threshold in (5, 10, 20)},
            point_errors=[float(value) if np.isfinite(value) else None for value in distances],
        ))
    pair = SimpleNamespace(positive=SimpleNamespace(items=rows),
                           negative=SimpleNamespace(items=[]))
    canonical = E._evaluate_2d_collected(pair, targets, candidates, top)
    canonical = {key: value for key, value in canonical.items()
                 if not key.startswith("box_ap") and key != "negative_count"}
    errors = np.concatenate([row["errors"] for row in detailed])
    denominator = sum(row["gt_count"] for row in detailed)
    summary = dict(
        median_px=canonical["keypoint_location_median_px"],
        p90_px=canonical["keypoint_location_p90_px"],
        matched_frames=sum(row["canonical_matched"] for row in detailed),
        full_frame_denominator=len(rows),
        failed_or_excluded_frames=sum(not row["canonical_matched"] for row in detailed),
        supervised_points=len(errors),
        gt_denominator=denominator,
        missing_supervised_keypoints=sum(row["missing_supervised_keypoints"] for row in detailed),
        ALL_GT_PCK={str(threshold): float(sum(
            row["canonical_hits"][str(threshold)] for row in detailed) / denominator)
            for threshold in (5, 10, 20)},
        partial_point_ALL_GT_PCK_SECONDARY={str(threshold): float(sum(
            row["partial_point_hits"][str(threshold)] for row in detailed) / denominator)
            for threshold in (5, 10, 20)},
        canonical=canonical,
        policy=("Conditional median/P90 require all9 finite and fixed baseline-box IoU>=0.5; "
                "ALL_GT_PCK counts every excluded supervised point as failure."),
    )
    return summary, detailed


def canonical_pose(rows, pose_ids, mapping, modules, prediction_binding):
    _, M, _, G, _ = modules
    from symmetry_aware_pose_metrics import cuboid_model_points
    truth = C.read_json(C.POSE_RAW / "GEOMETRY_RESOLVED_POSE_GT.json")["frames"]
    metadata = {row["frame_id"]: row for row in mapping}
    legacy = _legacy()
    groups = object_subgroups(rows, enforce_dev_counts=True)
    output = {}
    for method in METHODS:
        values, all_rows = [], []
        for row in rows:
            frame_id = row["frame_id"]
            item = metadata[frame_id]
            spec = dict(long_m=item["dimensions_m"]["long"],
                        short_m=item["dimensions_m"]["short"],
                        height_m=item["dimensions_m"]["height"])
            solved = legacy.solve_without_gt(
                method_points(row, method), np.asarray(item["camera_intrinsics"]), spec, M, G)
            record = dict(frame_id=frame_id, original_pose_frame_id=pose_ids[frame_id],
                          session_id=row["session_id"], object_type=item["object_type"],
                          image_key=row["image_key"], status=solved["status"])
            if solved["status"] != "OK":
                if "reason" in solved:
                    record["reason"] = solved["reason"]
                all_rows.append(record)
                continue
            target = truth[pose_ids[frame_id]]
            dimensions = target["physical_dimensions_m"]
            extents = (dimensions["across"], dimensions["height"], dimensions["along"])
            metrics = G.score_pose_against_gt(
                cuboid_model_points(extents), solved["rotation"], solved["translation"],
                np.asarray(target["R_gt_representative"]), np.asarray(target["t_gt"]), extents)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite pose metric: {method}/{frame_id}")
            record.update(
                translation_error_cm=metrics["translation_error_cm"],
                rotation_error_deg=metrics["rotation_error_deg"],
                yaw_error_deg=metrics["yaw_error_deg"],
                iou3d=metrics["iou3d"],
                add_sym_m=metrics["symmetry_aware_add_m"],
                diameter_m=metrics["model_diameter_m"],
                add_sym_normalized=(metrics["symmetry_aware_add_m"]
                                    / metrics["model_diameter_m"]),
                axis_correct=solved["chosen"] == target["physical_long_axis"],
                selected_axis=solved["chosen"],
                reprojection_mean_px=solved["reprojection_mean_px"],
            )
            values.append(record)
            all_rows.append(record)
        summary = pose_summary(values, len(rows))
        summary["excluded_frame_ids"] = [row["frame_id"] for row in all_rows
                                         if row["status"] != "OK"]
        sessions = sorted({row["session_id"] for row in rows})
        by_session = {session: pose_summary(
            [row for row in values if row["session_id"] == session],
            sum(row["session_id"] == session for row in rows)) for session in sessions}
        by_object = {label: pose_summary(
            [row for row in values if row["object_type"] == OBJECT_GROUPS[label]], len(selected))
            for label, selected in groups.items()}
        artifact = C.RAW / "evaluation" / f"POSE_EVALUATION_{method}.json"
        C.write_frozen_json(artifact, dict(
            complete=True, method=method, primary_path="MAIN", source=prediction_binding,
            canonical_functions=["pose_evaluation_paths.predict_pose_without_gt",
                                 "run_pose_evaluation.solve",
                                 "pose_evaluation_paths.score_pose_against_gt"],
            gt=C.binding(C.POSE_RAW / "GEOMETRY_RESOLVED_POSE_GT.json"),
            object_contract=C.binding(C.POSE_RAW / "POSE_EVAL_OBJECT_CONTRACT.json"),
            summary=summary, by_session=by_session, by_object=by_object,
            all_frame_rows=all_rows))
        output[method] = dict(summary=summary, rows=values, all_frame_rows=all_rows,
                              by_object=by_object, artifact=C.binding(artifact))
        print("FULL_DIM_REFINER_POSE", method, len(values), "/", len(rows), flush=True)
    return output


def comparison_groups():
    groups = {arm: [C.arm_key(arm, seed) for seed in C.SEEDS] for arm in C.HEAD_ARMS}
    return {
        "D0_minus_FULL": (groups["D0"], [BASELINE]),
        "P0_minus_FULL": (groups["P0"], [BASELINE]),
        "P5_minus_FULL": (groups["P5"], [BASELINE]),
        "P5_CONSTANT_minus_FULL": (groups["P5_CONSTANT"], [BASELINE]),
        "P5_minus_P0": (groups["P5"], groups["P0"]),
        "P5_minus_P5_CONSTANT": (groups["P5"], groups["P5_CONSTANT"]),
    }


def paired_results(stores, poses, rows, bootstrap, paired_stats):
    sessions = [row["session_id"] for row in rows]
    frame_ids = [row["frame_id"] for row in rows]
    output = {}
    for name, (left_names, right_names) in comparison_groups().items():
        left = [stores[method] for method in left_names]
        right = [stores[method] for method in right_names]
        entry = dict(conditional_keypoint_median={level: bootstrap.paired_medians(
            [[row["errors"] for row in method] for method in left],
            [[row["errors"] for row in method] for method in right], sessions,
            paired_stats, level=level)
            for level in ("session", "frame")})
        entry["ALL_GT_PCK"] = {str(threshold): bootstrap.paired_full_rates(
            left, right, sessions,
            lambda row, key=str(threshold): row["canonical_hits"][key],
            lambda row: row["gt_count"]) for threshold in (5, 10, 20)}
        entry["pose"] = {}
        for metric in ("translation_error_cm", "rotation_error_deg", "yaw_error_deg", "iou3d"):
            paired_left, paired_right = [], []
            for methods, destination in ((left_names, paired_left), (right_names, paired_right)):
                for method in methods:
                    lookup = {row["frame_id"]: row[metric] for row in poses[method]["rows"]}
                    destination.append([[lookup[frame_id]] if frame_id in lookup else []
                                        for frame_id in frame_ids])
            entry["pose"][metric] = bootstrap.paired_medians(
                paired_left, paired_right, sessions, paired_stats)
        def coverage(methods):
            result = []
            for method in methods:
                available = {row["frame_id"] for row in poses[method]["rows"]}
                result.append([dict(available=int(frame_id in available))
                               for frame_id in frame_ids])
            return result
        entry["pose"]["coverage"] = bootstrap.paired_full_rates(
            coverage(left_names), coverage(right_names), sessions,
            lambda row: row["available"], lambda row: 1)
        output[name] = entry
    return output


def seed_means(methods):
    result = {}
    for arm in C.HEAD_ARMS:
        values = [methods[C.arm_key(arm, seed)] for seed in C.SEEDS]
        mean = lambda rows: float(np.mean(rows)) if all(value is not None for value in rows) else None
        result[arm] = dict(
            seed_policy="mean of per-seed statistics; not an ensemble",
            **{key: mean([value[key] for value in values]) for key in (
                "median_px", "p90_px", "matched_frames", "failed_or_excluded_frames",
                "supervised_points", "missing_supervised_keypoints")},
            ALL_GT_PCK={str(threshold): mean([
                value["ALL_GT_PCK"][str(threshold)] for value in values])
                for threshold in (5, 10, 20)},
            pose={key: mean([value["pose"].get(key) for value in values]) for key in (
                "rotation_median_deg", "translation_median_cm", "yaw_median_deg",
                "translation_p90_cm", "rotation_p90_deg", "yaw_p90_deg",
                "iou3d_median", "add_sym_auc_conditional",
                "add_sym_auc_full_population", "coverage", "failure_count")},
        )
    return result


def write_frame_csv(rows, stores, poses, destination: Path | str | None = None):
    destination = C.RAW / "DEV_FRAME_RESULTS.csv" if destination is None else Path(destination)
    fields = ["method", "frame_id", "session_id", "object_type", "image_key",
              "canonical_matched",
              "box_iou", "gt_count", "finite_predicted_keypoints",
              "missing_supervised_keypoints", "canonical_hits5", "canonical_hits10",
              "canonical_hits20", "pose_available", "translation_error_cm",
              "rotation_error_deg", "yaw_error_deg", "iou3d", "add_sym_m",
              "diameter_m", "add_sym_normalized"]
    fields += [f"point{index}_error_px" for index in range(9)]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for method in METHODS:
        pose = {row["frame_id"]: row for row in poses[method]["rows"]}
        for record in stores[method]:
            output = {key: record[key] for key in fields if key in record}
            output["method"] = method
            output.update({f"canonical_hits{threshold}": record["canonical_hits"][str(threshold)]
                           for threshold in (5, 10, 20)})
            solved = pose.get(record["frame_id"])
            output["pose_available"] = solved is not None
            for key in ("translation_error_cm", "rotation_error_deg", "yaw_error_deg", "iou3d",
                        "add_sym_m", "diameter_m", "add_sym_normalized"):
                output[key] = solved[key] if solved is not None else None
            output.update({f"point{index}_error_px": value
                           for index, value in enumerate(record["point_errors"])})
            writer.writerow(output)
    text = buffer.getvalue()
    if destination.exists():
        if destination.read_text() != text:
            raise ValueError(f"Existing frame CSV differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    if temporary.exists():
        if temporary.read_text() == text:
            temporary.replace(destination)
            return destination
        temporary.unlink()
    temporary.write_text(text)
    temporary.replace(destination)
    return destination


def score(path: Path | str = C.RAW / "DEV_PREDICTIONS.json") -> dict:
    payload, rows, targets, pose_ids, mapping, sources, modules = load_inputs(path)
    E, _, _, _, _ = modules
    prediction_binding = C.binding(path)
    groups = object_subgroups(rows, enforce_dev_counts=True)
    support_disclosure = dimension_support(rows)
    C.write_frozen_json(C.DOC / "DIMENSION_SUPPORT.json", support_disclosure)
    C.write_frozen_json(C.DOC / "DEV_EVALUATION_LOCK.json", dict(
        complete=True,
        code=C.binding(__file__),
        predictions=prediction_binding,
        sources=sources,
        frame_mapping=mapping,
        positive_frames=319,
        sessions=13,
        supervised_keypoints=2818,
        object_subgroups={label: len(value) for label, value in groups.items()},
        dimension_support=C.binding(C.DOC / "DIMENSION_SUPPORT.json"),
        new_image_forwards=0,
        negative_images_evaluated=0,
        AP_claim=False,
        bootstrap=dict(resamples=RESAMPLES, seed=BOOT_SEED, secondary_unadjusted=True),
        pose_policy=("Canonical prediction-only selector and SQPnP/RefineLM; reconstructed "
                     "reference, denominator319, explicit failures."),
    ))
    methods, stores = {}, {}
    for method in METHODS:
        methods[method], stores[method] = summarize_2d(E, rows, targets, method)
        methods[method]["by_object"] = {}
        for label, selected in groups.items():
            selected_ids = {row["frame_id"] for row in selected}
            subgroup_targets = {key: value for key, value in targets.items() if key in selected_ids}
            methods[method]["by_object"][label], _ = summarize_2d(
                E, selected, subgroup_targets, method)
    support = [[row["canonical_matched"] for row in stores[method]] for method in METHODS]
    if not all(value == support[0] for value in support):
        raise ValueError("Fixed box/missing-mask contract failed to preserve conditional support")
    poses = canonical_pose(rows, pose_ids, mapping, modules, prediction_binding)
    for method in METHODS:
        methods[method]["pose"] = poses[method]["summary"]
        methods[method]["pose_by_object"] = poses[method]["by_object"]
    legacy = _legacy()
    paired = paired_results(stores, poses, rows, legacy, modules[-1])
    frame_csv = write_frame_csv(rows, stores, poses)
    precision = {method: [dict(row, errors=row["errors"].tolist(),
                                    errors8=row["errors8"].tolist())
                          for row in stores[method]] for method in METHODS}
    C.write_frozen_json(C.RAW / "DEV_FULL_PRECISION_ERRORS.json", precision)
    C.write_frozen_json(C.DOC / "DEV_PAIRED_RESULTS.json", dict(
        complete=True,
        results=paired,
        predictions=prediction_binding,
        code=C.binding(__file__),
        multiplicity="Unadjusted exploratory intervals on reused DEV",
        strict_support_no_silent_intersection=True,
    ))
    result = dict(
        complete=True,
        schema="resnet18_full_dim_refiner_dev_results_v1",
        role="REUSED_DEV",
        code=C.binding(__file__),
        evaluation_lock=C.binding(C.DOC / "DEV_EVALUATION_LOCK.json"),
        predictions=prediction_binding,
        methods=methods,
        seed_mean=seed_means(methods),
        positive_frames=319,
        sessions=13,
        gt_denominator=2818,
        object_subgroups={label: len(value) for label, value in groups.items()},
        dimension_support=C.binding(C.DOC / "DIMENSION_SUPPORT.json"),
        same_conditional_support=True,
        negative_images_evaluated=0,
        AP_claim=False,
        new_image_forwards=0,
        geometry_derived_reference_not_independent_physical_metrology=True,
        primary_pose_path="MAIN",
        diagnostic_oracle_evaluated=False,
        pose_artifacts={method: poses[method]["artifact"] for method in METHODS},
        frame_results=C.binding(frame_csv),
        full_precision_errors=C.binding(C.RAW / "DEV_FULL_PRECISION_ERRORS.json"),
        paired_results=C.binding(C.DOC / "DEV_PAIRED_RESULTS.json"),
        source_bindings=sources,
    )
    C.write_frozen_json(C.DOC / "DEV_RESULTS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=C.RAW / "DEV_PREDICTIONS.json")
    args = parser.parse_args()
    result = score(args.predictions)
    print({"PASS": True, "frames": result["positive_frames"], "methods": len(METHODS)})


if __name__ == "__main__":
    main()
