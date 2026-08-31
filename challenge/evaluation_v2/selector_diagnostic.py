"""Run the frozen prediction-only W/D selector on the exact DEV_POS140 set.

The run is deliberately split into two phases.  Phase 1 gives the selector only
predicted nine-point coordinates, camera intrinsics, and the fixed canonical
dimensions.  Phase 2 opens the GT parity and pose-equivalence records only after
all selections have been made, then computes diagnostic correctness and tail
attribution.  No threshold or selector weight is fitted from DEV results.

Tail dominance is fixed before inference: for each of restricted ADD-S,
rotation, translation, and yaw, rank all 140 frames by error (a missing or
invalid selector result is +infinity), take the worst ceil(10%), and fail when
selector failures are a strict majority in any tail.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from challenge.evaluation_v2.paper_real_eval import (  # type: ignore[import-not-found]
        INFERENCE_CONFIDENCE_FLOOR,
        INFERENCE_IMGSZ,
        INFERENCE_PAD,
        _UltralyticsPredictor,
    )
    from challenge.evaluation_v2.pnp_selector import (  # type: ignore[import-not-found]
        PnPSelectionResult,
        SelectorConfig,
        SelectorGateState,
        SelectorStatus,
        assess_selector_diagnostics,
        select_pnp_hypotheses,
    )
    from challenge.evaluation_v2.pose_metrics import (  # type: ignore[import-not-found]
        add_error_m,
        model_diameter_m,
        rotation_error_degrees,
        translation_error_m,
        yaw_error_degrees,
    )
    from challenge.evaluation_v2.real_dataset_contract import (  # type: ignore[import-not-found]
        ContractError,
        ManifestItem,
        PopulationId,
        REPO_ROOT,
        load_population_manifest,
        validate_registered_membership,
    )
    from scripts.annotate import pallet_geometry as geometry  # type: ignore[import-not-found]
    from scripts.annotate.real_gt_v2_schema import (  # type: ignore[import-not-found]
        SchemaValidationError,
        validate_gt_v2,
    )
else:
    from .paper_real_eval import (
        INFERENCE_CONFIDENCE_FLOOR,
        INFERENCE_IMGSZ,
        INFERENCE_PAD,
        _UltralyticsPredictor,
    )
    from .pnp_selector import (
        PnPSelectionResult,
        SelectorConfig,
        SelectorGateState,
        SelectorStatus,
        assess_selector_diagnostics,
        select_pnp_hypotheses,
    )
    from .pose_metrics import (
        add_error_m,
        model_diameter_m,
        rotation_error_degrees,
        translation_error_m,
        yaw_error_degrees,
    )
    from .real_dataset_contract import (
        ContractError,
        ManifestItem,
        PopulationId,
        REPO_ROOT,
        load_population_manifest,
        validate_registered_membership,
    )
    from scripts.annotate import pallet_geometry as geometry
    from scripts.annotate.real_gt_v2_schema import SchemaValidationError, validate_gt_v2


SCHEMA_VERSION = "pallet_pose_selector_diagnostic_v1"
TAIL_FRACTION = 0.10
TAIL_FAILURE_MAJORITY_MAX = 0.50
EXPECTED_SHORT_ASSIGNMENTS = ("YAW_0", "YAW_180")
EXPECTED_LONG_ASSIGNMENTS = ("YAW_90", "YAW_270")
ARTIFACT_NAMES = (
    "SELECTOR_DIAGNOSTIC.json",
    "SELECTOR_PER_FRAME.csv",
    "SELECTOR_FAILURES.md",
    "SELECTOR_REPORT.md",
)
DEFAULT_INTRINSICS_MANIFEST = (
    REPO_ROOT / "challenge/real_gt_v2/manifests/DEV_POS140_INTRINSICS.json"
)
INTRINSICS_SCHEMA_VERSION = "pallet_pose_camera_intrinsics_manifest_v1"


@dataclass(frozen=True)
class PredictionOnlyRecord:
    frame_id: str
    detection_count: int
    top_score: float | None
    selection: PnPSelectionResult | None
    prediction_failure: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--intrinsics-manifest",
        default=str(DEFAULT_INTRINSICS_MANIFEST),
        help="camera-only DEV_POS140 calibration manifest; must contain no GT pose/parity",
    )
    parser.add_argument("--device", default="0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path, purpose: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{purpose}_UNREADABLE: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{purpose}_ROOT_MUST_BE_OBJECT: {path}")
    return value


def _label_path(item: ManifestItem) -> Path:
    if item.label is None:
        raise ContractError(f"DEV_POS140_LABEL_REQUIRED: {item.frame_id}")
    path = (REPO_ROOT / item.label).resolve()
    if not path.is_file():
        raise ContractError(f"DEV_POS140_LABEL_NOT_FOUND: {item.frame_id}")
    return path


def _load_intrinsics_manifest(
    path: Path,
    manifest: Any,
) -> dict[str, np.ndarray]:
    """Load the narrow camera-only projection used before GT is opened."""

    payload = _load_json(path, "SELECTOR_INTRINSICS_MANIFEST")
    if payload.get("schema_version") != INTRINSICS_SCHEMA_VERSION:
        raise ContractError("SELECTOR_INTRINSICS_SCHEMA_INVALID")
    if (
        payload.get("population_id") != PopulationId.DEV_POS140.value
        or payload.get("population_membership_sha256") != manifest.membership_sha256
        or payload.get("count") != manifest.count
    ):
        raise ContractError("SELECTOR_INTRINSICS_POPULATION_MISMATCH")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != manifest.count:
        raise ContractError("SELECTOR_INTRINSICS_RECORD_COUNT_MISMATCH")
    allowed_fields = {
        "frame_id",
        "fx",
        "fy",
        "cx",
        "cy",
        "source_label_sha256",
    }
    out: dict[str, np.ndarray] = {}
    actual_order: list[str] = []
    for index, row in enumerate(raw_records):
        if not isinstance(row, dict) or set(row) != allowed_fields:
            raise ContractError(f"SELECTOR_INTRINSICS_RECORD_{index}_FIELDS_INVALID")
        frame_id = row.get("frame_id")
        source_sha = row.get("source_label_sha256")
        if not isinstance(frame_id, str) or not frame_id:
            raise ContractError(f"SELECTOR_INTRINSICS_RECORD_{index}_FRAME_INVALID")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise ContractError(f"SELECTOR_INTRINSICS_RECORD_{index}_SHA_INVALID")
        try:
            matrix = np.array(
                [
                    [float(row["fx"]), 0.0, float(row["cx"])],
                    [0.0, float(row["fy"]), float(row["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"SELECTOR_INTRINSICS_RECORD_{index}_VALUES_INVALID"
            ) from exc
        if (
            not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0.0
            or matrix[1, 1] <= 0.0
            or frame_id in out
        ):
            raise ContractError(f"SELECTOR_INTRINSICS_RECORD_{index}_VALUES_INVALID")
        out[frame_id] = matrix
        actual_order.append(frame_id)
    if actual_order != list(manifest.frame_ids):
        raise ContractError("SELECTOR_INTRINSICS_FRAME_ORDER_MISMATCH")
    return out


def _prediction_phase(
    items: Sequence[ManifestItem],
    predictor: Any,
    config: SelectorConfig,
    intrinsics_by_frame: Mapping[str, np.ndarray],
) -> tuple[PredictionOnlyRecord, ...]:
    """Finish every prediction-only choice before any GT parity is opened."""

    fixed_dimensions = geometry.canonical_dimensions()
    out: list[PredictionOnlyRecord] = []
    for item in items:
        image_path = (REPO_ROOT / item.image).resolve()
        predictions = predictor.predict(image_path)
        if not predictions:
            out.append(PredictionOnlyRecord(item.frame_id, 0, None, None, "NO_DETECTION"))
            continue
        score, _box, keypoints = max(predictions, key=lambda row: float(row[0]))
        points = np.asarray(keypoints, dtype=np.float64) if keypoints is not None else None
        if points is None or points.shape != (9, 2) or not np.isfinite(points).all():
            out.append(
                PredictionOnlyRecord(
                    item.frame_id,
                    len(predictions),
                    float(score),
                    None,
                    "INVALID_NINE_KEYPOINT_OUTPUT",
                )
            )
            continue
        selection = select_pnp_hypotheses(
            points,
            intrinsics_by_frame[item.frame_id],
            fixed_dimensions,
            config,
        )
        out.append(
            PredictionOnlyRecord(
                item.frame_id,
                len(predictions),
                float(score),
                selection,
                None,
            )
        )
    return tuple(out)


def _diagnostic_truth(item: ManifestItem) -> tuple[str, tuple[np.ndarray, ...]]:
    """Open parity and canonical equivalence class only in post-selection phase."""

    payload = _load_json(_label_path(item), "SELECTOR_DIAGNOSTIC_GT")
    try:
        validate_gt_v2(payload)
    except SchemaValidationError as exc:
        raise ContractError(f"SELECTOR_DIAGNOSTIC_GT_INVALID: {item.frame_id}: {exc}") from exc
    obj = payload["objects"][0]
    assignments = tuple(obj["camera_facing_pnp"]["axis_assignment_candidates"])
    if assignments == EXPECTED_SHORT_ASSIGNMENTS:
        expected = "short-face-front"
    elif assignments == EXPECTED_LONG_ASSIGNMENTS:
        expected = "long-face-front"
    else:
        raise ContractError(
            f"SELECTOR_DIAGNOSTIC_GT_PARITY_INVALID: {item.frame_id}: {assignments}"
        )
    transforms = tuple(
        np.asarray(candidate["pose_transform"], dtype=np.float64)
        for candidate in obj["canonical_pose_candidates"]
    )
    if len(transforms) != 2 or any(transform.shape != (4, 4) for transform in transforms):
        raise ContractError(f"SELECTOR_DIAGNOSTIC_GT_POSE_CLASS_INVALID: {item.frame_id}")
    return expected, transforms


def _hypothesis_payload(selection: PnPSelectionResult | None) -> dict[str, Any]:
    if selection is None:
        return {
            "short_score": None,
            "long_score": None,
            "score_margin": None,
            "short_success": False,
            "long_success": False,
        }
    by_name = {row.name: row for row in selection.hypotheses}
    short = by_name["short-face-front"]
    long = by_name["long-face-front"]
    margin = (
        abs(float(short.score) - float(long.score))
        if short.score is not None and long.score is not None
        else None
    )
    return {
        "short_score": short.score,
        "long_score": long.score,
        "score_margin": margin,
        "short_success": short.success,
        "long_success": long.success,
    }


def _pose_diagnostics(
    selection: PnPSelectionResult | None,
    target_transforms: Sequence[np.ndarray],
) -> dict[str, float | bool | None]:
    if (
        selection is None
        or selection.status is not SelectorStatus.SELECTED
        or not selection.canonical_candidates
    ):
        return {
            "pose_valid": False,
            "restricted_adds_error_m": None,
            "restricted_adds_normalized": None,
            "rotation_error_deg": None,
            "translation_error_m": None,
            "yaw_error_deg": None,
        }
    predicted = selection.canonical_candidates[0]
    model_points = geometry.canonical_keypoints_3d()[:8]
    diameter = model_diameter_m(model_points)
    add_values: list[float] = []
    rotation_values: list[float] = []
    translation_values: list[float] = []
    yaw_values: list[float] = []
    for transform in target_transforms:
        target_rotation = transform[:3, :3]
        target_translation = transform[:3, 3]
        add_values.append(
            add_error_m(
                model_points,
                predicted.rotation,
                predicted.translation,
                target_rotation,
                target_translation,
            )
        )
        rotation_values.append(rotation_error_degrees(predicted.rotation, target_rotation))
        translation_values.append(
            translation_error_m(predicted.translation, target_translation)
        )
        yaw_values.append(yaw_error_degrees(predicted.rotation, target_rotation))
    restricted_adds = min(add_values)
    return {
        "pose_valid": True,
        "restricted_adds_error_m": restricted_adds,
        "restricted_adds_normalized": restricted_adds / diameter,
        "rotation_error_deg": min(rotation_values),
        "translation_error_m": min(translation_values),
        "yaw_error_deg": min(yaw_values),
    }


def _post_selection_records(
    items: Sequence[ManifestItem],
    predictions: Sequence[PredictionOnlyRecord],
) -> list[dict[str, Any]]:
    if len(items) != len(predictions):
        raise ContractError("SELECTOR_PREDICTION_MEMBERSHIP_LENGTH_MISMATCH")
    by_id = {record.frame_id: record for record in predictions}
    if len(by_id) != len(predictions):
        raise ContractError("SELECTOR_PREDICTION_FRAME_IDS_NOT_UNIQUE")
    rows: list[dict[str, Any]] = []
    for item in items:
        record = by_id.get(item.frame_id)
        if record is None:
            raise ContractError(f"SELECTOR_PREDICTION_MISSING: {item.frame_id}")
        expected, target_transforms = _diagnostic_truth(item)
        selection = record.selection
        selected = selection.selected_hypothesis if selection is not None else None
        selector_status = (
            selection.status.value if selection is not None else record.prediction_failure
        )
        correct = bool(
            selection is not None
            and selection.status is SelectorStatus.SELECTED
            and selected == expected
        )
        row: dict[str, Any] = {
            "frame_id": item.frame_id,
            "domain": item.domain,
            "session": item.source_set,
            "detection_count": record.detection_count,
            "top_score": record.top_score,
            "selector_status": selector_status,
            "selected_hypothesis": selected,
            "expected_hypothesis": expected,
            "correct": correct,
            "prediction_failure": record.prediction_failure,
        }
        row.update(_hypothesis_payload(selection))
        row.update(_pose_diagnostics(selection, target_transforms))
        rows.append(row)
    return rows


def assess_tail_dominance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen worst-decile/strict-majority attribution rule."""

    if not records:
        raise ValueError("records must not be empty")
    tail_n = max(1, int(math.ceil(len(records) * TAIL_FRACTION)))
    metric_fields = (
        "restricted_adds_normalized",
        "rotation_error_deg",
        "translation_error_m",
        "yaw_error_deg",
    )
    metrics: dict[str, Any] = {}
    passed = True
    for field in metric_fields:
        ranked: list[tuple[float, str, bool]] = []
        for row in records:
            value = row.get(field)
            score = (
                float(value)
                if isinstance(value, (int, float)) and math.isfinite(float(value))
                else math.inf
            )
            ranked.append((score, str(row["frame_id"]), not bool(row["correct"])))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        tail = ranked[:tail_n]
        failures = sum(failure for _, _, failure in tail)
        share = failures / tail_n
        dominated = share > TAIL_FAILURE_MAJORITY_MAX
        passed = passed and not dominated
        metrics[field] = {
            "tail_n": tail_n,
            "selector_failure_count": failures,
            "selector_failure_fraction": share,
            "dominated": dominated,
            "frame_ids": [frame_id for _, frame_id, _ in tail],
        }
    return {
        "rule": "worst ceil(10%) over all 140; invalid/missing=+inf; fail if >50% are selector failures",
        "tail_fraction": TAIL_FRACTION,
        "strict_failure_majority_threshold": TAIL_FAILURE_MAJORITY_MAX,
        "assessed": True,
        "passed": passed,
        "metrics": metrics,
    }


def _session_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = sorted({str(row["session"]) for row in records})
    out: dict[str, Any] = {}
    for session in sessions:
        rows = [row for row in records if row["session"] == session]
        out[session] = {
            "n": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        }
    return out


def _checkpoint_provenance(weights: Path) -> dict[str, Any]:
    run_dir = weights.parent.parent
    args_path = run_dir / "args.yaml"
    data_path = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_generic_only/data.yaml"
    return {
        "path": _display_path(weights),
        "sha256": _sha256(weights),
        "size_bytes": weights.stat().st_size,
        "selection_policy": "pre-existing run source of truth uses last.pt; no DEV result selection",
        "run_args": (
            {"path": _display_path(args_path), "sha256": _sha256(args_path)}
            if args_path.is_file()
            else None
        ),
        "training_data_contract": (
            {"path": _display_path(data_path), "sha256": _sha256(data_path)}
            if data_path.is_file()
            else None
        ),
    }


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _csv_text(records: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "frame_id",
        "domain",
        "session",
        "detection_count",
        "top_score",
        "selector_status",
        "selected_hypothesis",
        "expected_hypothesis",
        "correct",
        "prediction_failure",
        "short_success",
        "long_success",
        "short_score",
        "long_score",
        "score_margin",
        "pose_valid",
        "restricted_adds_error_m",
        "restricted_adds_normalized",
        "rotation_error_deg",
        "translation_error_m",
        "yaw_error_deg",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def _report_markdown(payload: Mapping[str, Any]) -> str:
    gate = payload["gate"]
    sessions = payload["session_summary"]
    lines = [
        "# DEV_POS140 W/D selector diagnostic",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This is a development diagnostic, never a FINAL result. The frozen selector",
        "configuration was not changed after observing these values.",
        "",
        "## Gate",
        "",
        f"- overall: `{gate['overall_accuracy']}` (required >= 0.95)",
        f"- NIGHT: `{gate['night_accuracy']}` (required >= 0.90)",
        f"- minimum session: `{gate['minimum_session_accuracy']}` (required >= 0.85)",
        f"- tail dominance: `{gate['tail_dominance_passed']}`",
        f"- blocked reason: `{gate['blocked_reason']}`",
        "",
        "## Session accuracy",
        "",
        "| session | N | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sessions.items():
        lines.append(f"| `{name}` | {row['n']} | {row['correct']} | {row['accuracy']:.6f} |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "Selection used only predicted 9-keypoint coordinates, camera intrinsics,",
            "fixed physical dimensions (1.10/0.11/1.30 m), and frozen scoring constants.",
            "GT parity and canonical equivalence-class poses were opened only after all",
            "selection decisions had completed.",
            "",
            "A FAIL is not tuned away on DEV. Only source/synthetic-only redesign may be",
            "proposed before a new pre-registered diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def _failures_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    failures = [row for row in records if not bool(row["correct"])]
    lines = [
        "# Selector failures",
        "",
        f"Incorrect / ambiguous / missing: **{len(failures)} / {len(records)}**.",
        "",
        "No threshold or selector score was changed after this list was produced.",
        "",
        "| frame | domain | session | status | selected | expected |",
        "|---|---|---|---|---|---|",
    ]
    for row in failures:
        lines.append(
            f"| `{row['frame_id']}` | {row['domain']} | `{row['session']}` | "
            f"`{row['selector_status']}` | `{row['selected_hypothesis']}` | "
            f"`{row['expected_hypothesis']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_artifacts(out_dir: Path, payload: Mapping[str, Any]) -> None:
    if out_dir.exists():
        raise ContractError(f"OUTPUT_DIRECTORY_ALREADY_EXISTS: {out_dir}")
    if not out_dir.parent.is_dir():
        raise ContractError(f"OUTPUT_PARENT_NOT_FOUND: {out_dir.parent}")
    records = payload["records"]
    texts = {
        "SELECTOR_DIAGNOSTIC.json": json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        "SELECTOR_PER_FRAME.csv": _csv_text(records),
        "SELECTOR_FAILURES.md": _failures_markdown(records),
        "SELECTOR_REPORT.md": _report_markdown(payload),
    }
    out_dir.mkdir()
    for name in ARTIFACT_NAMES:
        with (out_dir / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(texts[name])


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_population_manifest(args.manifest, validate_files=True)
    validate_registered_membership(manifest, validate_files=True)
    if manifest.population_id is not PopulationId.DEV_POS140:
        raise ContractError(
            f"SELECTOR_DIAGNOSTIC_REQUIRES_DEV_POS140: {manifest.population_id.value}"
        )
    if manifest.count != 140 or not manifest.available or not manifest.frozen:
        raise ContractError("SELECTOR_DIAGNOSTIC_DEV_POS140_NOT_READY")

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise ContractError(f"WEIGHTS_NOT_FOUND: {weights}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise ContractError(f"OUTPUT_DIRECTORY_ALREADY_EXISTS: {out_dir}")

    config = SelectorConfig()
    intrinsics_path = Path(args.intrinsics_manifest).expanduser().resolve()
    intrinsics_by_frame = _load_intrinsics_manifest(intrinsics_path, manifest)
    predictor = _UltralyticsPredictor(weights, args.device)
    prediction_records = _prediction_phase(
        manifest.items,
        predictor,
        config,
        intrinsics_by_frame,
    )
    records = _post_selection_records(manifest.items, prediction_records)
    tail = assess_tail_dominance(records)
    tail_notes = (
        "Frozen worst-decile attribution rule passed: selector failures are not a strict "
        "majority in any metric tail."
        if tail["passed"]
        else "Frozen worst-decile attribution rule failed: selector failures dominate at "
        "least one metric tail."
    )
    gate = assess_selector_diagnostics(
        records,
        tail_dominance_assessed=True,
        tail_dominance_passed=bool(tail["passed"]),
        tail_dominance_notes=tail_notes,
    )
    status = (
        "SELECTOR_PASS"
        if gate.status is SelectorGateState.PASS
        else "POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR"
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(),
        "status": status,
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "population": {
            "population_id": manifest.population_id.value,
            "count": manifest.count,
            "membership_sha256": manifest.membership_sha256,
            "manifest": _display_path(manifest.source_path),
            "manifest_file_sha256": _sha256(manifest.source_path),
        },
        "checkpoint": _checkpoint_provenance(weights),
        "camera_intrinsics_manifest": {
            "path": _display_path(intrinsics_path),
            "sha256": _sha256(intrinsics_path),
            "schema_version": INTRINSICS_SCHEMA_VERSION,
            "contains_forbidden_gt_fields": False,
        },
        "inference_recipe": {
            "top_candidate_rule": "highest box confidence per frame",
            "pad": INFERENCE_PAD,
            "border": "BORDER_REFLECT_101",
            "imgsz": INFERENCE_IMGSZ,
            "confidence_floor": INFERENCE_CONFIDENCE_FLOOR,
            "device": str(args.device),
        },
        "selector_config": asdict(config),
        "gt_leakage_contract": {
            "selector_inputs": [
                "predicted_9_keypoints",
                "camera_intrinsics",
                "fixed_physical_dimensions",
                "frozen_selector_config",
            ],
            "forbidden": [
                "GT dimensions_m",
                "GT pose",
                "GT axis assignment",
                "GT keypoint error",
                "session prior",
            ],
            "comparison_phase": "GT parity read only after all selection decisions complete",
        },
        "gate": gate.to_dict(),
        "tail_dominance": tail,
        "tail_dominance_assessed": True,
        "tail_dominance_passed": bool(tail["passed"]),
        "tail_dominance_notes": tail_notes,
        "session_summary": _session_summary(records),
        "records": records,
    }
    _write_artifacts(out_dir, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "TAIL_FAILURE_MAJORITY_MAX",
    "TAIL_FRACTION",
    "assess_tail_dominance",
    "build_parser",
    "main",
    "run",
]
