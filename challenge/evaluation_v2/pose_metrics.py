"""Canonical pose metrics with a hard paper-contract gate.

No ADD/ADD-S, rotation, translation or yaw value is produced unless all four
prerequisites pass: canonical migration, the prediction-only axis selector,
the frozen symmetry specification and a frozen FINAL membership.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .pnp_selector import (
    NIGHT_AXIS_ACCURACY_MIN,
    OVERALL_AXIS_ACCURACY_MIN,
    SESSION_AXIS_ACCURACY_MIN,
    SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT,
    SelectorGateReport,
    SelectorGateState,
)


POSE_METRIC_FIELDS = (
    "add_or_adds_auc",
    "rotation_median_deg",
    "translation_median_m",
    "yaw_median_deg",
)


@dataclass(frozen=True)
class PoseMetricGate:
    canonical_migration_status: str
    selector_status: str
    symmetry_status: str
    final_manifest_status: str
    passed: bool
    blocked_reasons: tuple[str, ...]

    @property
    def blocked_reason(self) -> str | None:
        return ";".join(self.blocked_reasons) if self.blocked_reasons else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_migration": self.canonical_migration_status,
            "selector": self.selector_status,
            "symmetry": self.symmetry_status,
            "final_manifest": self.final_manifest_status,
            "passed": self.passed,
            "blocked_reason": self.blocked_reason,
            "blocked_reasons": list(self.blocked_reasons),
        }


@dataclass(frozen=True)
class PoseErrorRecord:
    """Per-frame canonical errors after a successful, fully gated evaluation."""

    add_error_m: float
    adds_error_m: float
    object_diameter_m: float
    rotation_error_deg: float
    translation_error_m: float
    yaw_error_deg: float
    object_type: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "add_error_m",
            "adds_error_m",
            "object_diameter_m",
            "rotation_error_deg",
            "translation_error_m",
            "yaw_error_deg",
        ):
            value = float(getattr(self, name))
            if math.isnan(value):
                raise ValueError(f"{name} may be finite or +inf, never NaN")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not math.isfinite(self.object_diameter_m) or self.object_diameter_m <= 0.0:
            raise ValueError("object_diameter_m must be positive and finite")
        if self.object_type is not None and (
            not isinstance(self.object_type, str) or not self.object_type.strip()
        ):
            raise ValueError("object_type must be null or a non-empty string")


def build_pose_metric_gate(
    *,
    canonical_migration_status: str,
    selector_report: SelectorGateReport,
    symmetry_status: str,
    final_manifest_frozen: bool,
) -> PoseMetricGate:
    """Build the strict four-condition gate using exact status values."""

    if not isinstance(selector_report, SelectorGateReport):
        raise TypeError("selector_report must be SelectorGateReport")
    reasons: list[str] = []
    if canonical_migration_status != "PASS":
        reasons.append("CANONICAL_MIGRATION_NOT_PASS")
    expected_sessions = dict(selector_report.expected_session_counts)
    expected_population_is_well_formed = bool(
        isinstance(selector_report.object_type, str)
        and selector_report.object_type.strip()
        and isinstance(selector_report.population_id, str)
        and selector_report.population_id.strip()
        and isinstance(selector_report.population_role, str)
        and selector_report.population_role.strip()
        and SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT.get(
            selector_report.object_type
        )
        == selector_report.population_id
        and type(selector_report.expected_sample_count) is int
        and selector_report.expected_sample_count > 0
        and type(selector_report.expected_night_count) is int
        and 0 <= selector_report.expected_night_count
        <= selector_report.expected_sample_count
        and expected_sessions
        and all(
            isinstance(session, str)
            and session
            and type(count) is int
            and count > 0
            for session, count in expected_sessions.items()
        )
        and sum(expected_sessions.values()) == selector_report.expected_sample_count
    )
    night_passed = bool(
        selector_report.expected_night_count == 0
        and selector_report.night_count == 0
        and selector_report.night_accuracy is None
    ) or bool(
        selector_report.expected_night_count > 0
        and selector_report.night_count == selector_report.expected_night_count
        and selector_report.night_accuracy is not None
        and selector_report.night_accuracy >= NIGHT_AXIS_ACCURACY_MIN
    )
    selector_passed = bool(
        selector_report.status is SelectorGateState.PASS
        and selector_report.population_validated is True
        and expected_population_is_well_formed
        and selector_report.sample_count == selector_report.expected_sample_count
        and selector_report.session_count == len(expected_sessions)
        and selector_report.overall_accuracy is not None
        and selector_report.overall_accuracy >= OVERALL_AXIS_ACCURACY_MIN
        and night_passed
        and selector_report.minimum_session_accuracy is not None
        and selector_report.minimum_session_accuracy >= SESSION_AXIS_ACCURACY_MIN
        and selector_report.tail_dominance_assessed is True
        and selector_report.tail_dominance_passed is True
    )
    if not selector_passed:
        reasons.append("POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR")
    if symmetry_status != "FROZEN":
        reasons.append("SYMMETRY_NOT_FROZEN")
    if final_manifest_frozen is not True:
        reasons.append("FINAL_MANIFEST_NOT_FROZEN")
    return PoseMetricGate(
        canonical_migration_status=canonical_migration_status,
        selector_status=selector_report.status.value,
        symmetry_status=symmetry_status,
        final_manifest_status="FROZEN" if final_manifest_frozen is True else "NOT_FROZEN",
        passed=not reasons,
        blocked_reasons=tuple(reasons),
    )


def blocked_pose_metrics(gate: PoseMetricGate) -> dict[str, Any]:
    """Return strict JSON-safe nulls for a blocked pose contract."""

    if gate.passed:
        raise ValueError("blocked_pose_metrics requires a blocked gate")
    out: dict[str, Any] = {
        "status": "BLOCKED",
        "metric_variant": None,
        "blocked_reason": gate.blocked_reason,
        "blocked_reasons": list(gate.blocked_reasons),
    }
    out.update({field: None for field in POSE_METRIC_FIELDS})
    return out


def _rotation(rotation: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite (3,3) rotation")
    if np.max(np.abs(matrix.T @ matrix - np.eye(3))) > 1e-6:
        raise ValueError(f"{name} is not orthonormal")
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-6:
        raise ValueError(f"{name} is not a proper rotation")
    return matrix


def rotation_error_degrees(
    predicted_rotation: Sequence[Sequence[float]] | np.ndarray,
    target_rotation: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    predicted = _rotation(predicted_rotation, "predicted_rotation")
    target = _rotation(target_rotation, "target_rotation")
    cosine = np.clip((np.trace(target.T @ predicted) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def yaw_error_degrees(
    predicted_rotation: Sequence[Sequence[float]] | np.ndarray,
    target_rotation: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Absolute relative yaw about the pallet's canonical local Y axis."""

    predicted = _rotation(predicted_rotation, "predicted_rotation")
    target = _rotation(target_rotation, "target_rotation")
    relative = target.T @ predicted
    return float(abs(np.degrees(np.arctan2(relative[0, 2], relative[2, 2]))))


def translation_error_m(
    predicted_translation: Sequence[float] | np.ndarray,
    target_translation: Sequence[float] | np.ndarray,
) -> float:
    predicted = np.asarray(predicted_translation, dtype=np.float64).reshape(-1)
    target = np.asarray(target_translation, dtype=np.float64).reshape(-1)
    if predicted.shape != (3,) or target.shape != (3,):
        raise ValueError("translations must contain exactly three values")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("translations must be finite")
    return float(np.linalg.norm(predicted - target))


def transformed_model_points(
    model_points: Sequence[Sequence[float]] | np.ndarray,
    rotation: Sequence[Sequence[float]] | np.ndarray,
    translation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("model_points must be a finite (N,3) array")
    matrix = _rotation(rotation, "rotation")
    offset = np.asarray(translation, dtype=np.float64).reshape(-1)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("translation must contain three finite values")
    return (matrix @ points.T).T + offset


def add_error_m(
    model_points: Sequence[Sequence[float]] | np.ndarray,
    predicted_rotation: Sequence[Sequence[float]] | np.ndarray,
    predicted_translation: Sequence[float] | np.ndarray,
    target_rotation: Sequence[Sequence[float]] | np.ndarray,
    target_translation: Sequence[float] | np.ndarray,
) -> float:
    predicted = transformed_model_points(model_points, predicted_rotation, predicted_translation)
    target = transformed_model_points(model_points, target_rotation, target_translation)
    return float(np.linalg.norm(predicted - target, axis=1).mean())


def adds_error_m(
    model_points: Sequence[Sequence[float]] | np.ndarray,
    predicted_rotation: Sequence[Sequence[float]] | np.ndarray,
    predicted_translation: Sequence[float] | np.ndarray,
    target_rotation: Sequence[Sequence[float]] | np.ndarray,
    target_translation: Sequence[float] | np.ndarray,
) -> float:
    """Unrestricted nearest-neighbour distance (not paper symmetry policy).

    The paper evaluator does not call this helper: on a cuboid keypoint set it
    could silently grant pitch/roll symmetries.  Paper ADD-S is instead the
    minimum corresponding-point ADD over the explicit proper rotations in the
    frozen benchmark symmetry contract.
    """

    predicted = transformed_model_points(model_points, predicted_rotation, predicted_translation)
    target = transformed_model_points(model_points, target_rotation, target_translation)
    pairwise = np.linalg.norm(predicted[:, None, :] - target[None, :, :], axis=2)
    return float(pairwise.min(axis=1).mean())


def model_diameter_m(model_points: Sequence[Sequence[float]] | np.ndarray) -> float:
    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("model_points must have shape (N>=2,3)")
    if not np.isfinite(points).all():
        raise ValueError("model_points must be finite")
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    diameter = float(pairwise.max())
    if diameter <= 0.0:
        raise ValueError("model diameter must be positive")
    return diameter


def pose_auc(normalized_errors: Iterable[float], *, max_fraction: float = 0.1) -> float:
    """Area under the unconditional accuracy curve over [0, max_fraction]."""

    errors = np.asarray(list(normalized_errors), dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("normalized_errors must be a non-empty 1-D sequence")
    if np.isnan(errors).any() or (errors < 0.0).any():
        raise ValueError("normalized_errors must be non-negative and not NaN")
    if not math.isfinite(max_fraction) or max_fraction <= 0.0:
        raise ValueError("max_fraction must be positive and finite")
    thresholds = np.linspace(0.0, max_fraction, 1001, dtype=np.float64)
    accuracy = np.array([(errors <= threshold).mean() for threshold in thresholds])
    area = float(
        np.sum((accuracy[:-1] + accuracy[1:]) * np.diff(thresholds) * 0.5)
        / max_fraction
    )
    return area


def summarize_pose_errors(
    records: Iterable[PoseErrorRecord],
    gate: PoseMetricGate,
    *,
    metric_variant: Literal["ADD", "ADD-S"],
) -> dict[str, Any]:
    """Summarize records only after the full contract passes.

    The blocked branch returns before iterating ``records``.  This is deliberate:
    callers cannot accidentally compute sensitive values and merely hide them
    during serialization.
    """

    if not isinstance(gate, PoseMetricGate):
        raise TypeError("gate must be PoseMetricGate")
    if not gate.passed:
        return blocked_pose_metrics(gate)
    if metric_variant not in {"ADD", "ADD-S"}:
        raise ValueError("metric_variant must be the frozen symmetry choice ADD or ADD-S")

    rows = list(records)
    if not rows:
        out: dict[str, Any] = {
            "status": "NOT_RUN",
            "metric_variant": metric_variant,
            "blocked_reason": "POSE_PREDICTIONS_NOT_PROVIDED",
            "blocked_reasons": ["POSE_PREDICTIONS_NOT_PROVIDED"],
        }
        out.update({field: None for field in POSE_METRIC_FIELDS})
        return out
    if not all(isinstance(row, PoseErrorRecord) for row in rows):
        raise TypeError("records must contain PoseErrorRecord values")

    normalized = [
        (row.add_error_m if metric_variant == "ADD" else row.adds_error_m)
        / row.object_diameter_m
        for row in rows
    ]
    rotation_values = [row.rotation_error_deg for row in rows if math.isfinite(row.rotation_error_deg)]
    translation_values = [
        row.translation_error_m for row in rows if math.isfinite(row.translation_error_m)
    ]
    yaw_values = [row.yaw_error_deg for row in rows if math.isfinite(row.yaw_error_deg)]
    successful = sum(
        math.isfinite(
            row.add_error_m if metric_variant == "ADD" else row.adds_error_m
        )
        and math.isfinite(row.rotation_error_deg)
        and math.isfinite(row.translation_error_m)
        and math.isfinite(row.yaw_error_deg)
        for row in rows
    )
    return {
        "status": "READY",
        "metric_variant": metric_variant,
        "blocked_reason": None,
        "blocked_reasons": [],
        "add_or_adds_auc": pose_auc(normalized),
        "rotation_median_deg": (
            float(np.median(rotation_values)) if rotation_values else None
        ),
        "translation_median_m": (
            float(np.median(translation_values)) if translation_values else None
        ),
        "yaw_median_deg": float(np.median(yaw_values)) if yaw_values else None,
        "pose_population_count": len(rows),
        "pose_success_count": successful,
        "pose_failure_count": len(rows) - successful,
    }


def summarize_multishape_pose_errors(
    records_by_object: Mapping[str, Iterable[PoseErrorRecord]],
    gates_by_object: Mapping[str, PoseMetricGate],
    metric_variants_by_object: Mapping[str, Literal["ADD", "ADD-S"]],
) -> dict[str, Any]:
    """Summarize object-specific pose rows and a fail-closed ``ALL`` row.

    A passed object subgroup may be reported even when another subgroup is
    blocked.  The ``ALL`` iterator is never constructed unless every required
    object gate passes.  This prevents a partial-population number from being
    mislabeled as an all-object paper result.
    """

    object_types = tuple(records_by_object)
    if not object_types:
        raise ValueError("records_by_object must contain at least one object type")
    if set(object_types) != set(gates_by_object) or set(object_types) != set(
        metric_variants_by_object
    ):
        raise ValueError("records, gates and metric variants must use identical object keys")
    if any(not isinstance(value, PoseMetricGate) for value in gates_by_object.values()):
        raise TypeError("gates_by_object must contain PoseMetricGate values")
    for object_type, variant in metric_variants_by_object.items():
        if variant not in {"ADD", "ADD-S"}:
            raise ValueError(f"{object_type}: metric variant must be ADD or ADD-S")

    materialized: dict[str, list[PoseErrorRecord]] = {}
    object_results: dict[str, dict[str, Any]] = {}
    for object_type in object_types:
        gate = gates_by_object[object_type]
        variant = metric_variants_by_object[object_type]
        if gate.passed:
            rows = list(records_by_object[object_type])
            materialized[object_type] = rows
            object_results[object_type] = summarize_pose_errors(
                rows, gate, metric_variant=variant
            )
        else:
            # ``summarize_pose_errors`` returns before touching the iterable.
            object_results[object_type] = summarize_pose_errors(
                records_by_object[object_type], gate, metric_variant=variant
            )

    blocked_objects = [
        object_type for object_type in object_types if not gates_by_object[object_type].passed
    ]
    if blocked_objects:
        reasons = [
            f"{object_type}:{reason}"
            for object_type in blocked_objects
            for reason in gates_by_object[object_type].blocked_reasons
        ]
        all_result: dict[str, Any] = {
            "status": "BLOCKED",
            "metric_variant": "OBJECT_SPECIFIC",
            "blocked_reason": ";".join(reasons),
            "blocked_reasons": reasons,
            "required_object_types": list(object_types),
        }
        all_result.update({field: None for field in POSE_METRIC_FIELDS})
    else:
        rows_with_variants = [
            (row, metric_variants_by_object[object_type])
            for object_type in object_types
            for row in materialized[object_type]
        ]
        if not rows_with_variants:
            all_result = {
                "status": "NOT_RUN",
                "metric_variant": "OBJECT_SPECIFIC",
                "blocked_reason": "POSE_PREDICTIONS_NOT_PROVIDED",
                "blocked_reasons": ["POSE_PREDICTIONS_NOT_PROVIDED"],
                "required_object_types": list(object_types),
            }
            all_result.update({field: None for field in POSE_METRIC_FIELDS})
        else:
            normalized = [
                (
                    row.add_error_m if variant == "ADD" else row.adds_error_m
                )
                / row.object_diameter_m
                for row, variant in rows_with_variants
            ]
            rotation = [
                row.rotation_error_deg
                for row, _ in rows_with_variants
                if math.isfinite(row.rotation_error_deg)
            ]
            translation = [
                row.translation_error_m
                for row, _ in rows_with_variants
                if math.isfinite(row.translation_error_m)
            ]
            yaw = [
                row.yaw_error_deg
                for row, _ in rows_with_variants
                if math.isfinite(row.yaw_error_deg)
            ]
            successful = sum(
                math.isfinite(row.add_error_m if variant == "ADD" else row.adds_error_m)
                and math.isfinite(row.rotation_error_deg)
                and math.isfinite(row.translation_error_m)
                and math.isfinite(row.yaw_error_deg)
                for row, variant in rows_with_variants
            )
            all_result = {
                "status": "READY",
                "metric_variant": "OBJECT_SPECIFIC",
                "blocked_reason": None,
                "blocked_reasons": [],
                "required_object_types": list(object_types),
                "add_or_adds_auc": pose_auc(normalized),
                "rotation_median_deg": float(np.median(rotation)) if rotation else None,
                "translation_median_m": (
                    float(np.median(translation)) if translation else None
                ),
                "yaw_median_deg": float(np.median(yaw)) if yaw else None,
                "pose_population_count": len(rows_with_variants),
                "pose_success_count": successful,
                "pose_failure_count": len(rows_with_variants) - successful,
            }

    return {"ALL": all_result, "objects": object_results}


__all__ = [
    "POSE_METRIC_FIELDS",
    "PoseErrorRecord",
    "PoseMetricGate",
    "add_error_m",
    "adds_error_m",
    "blocked_pose_metrics",
    "build_pose_metric_gate",
    "model_diameter_m",
    "pose_auc",
    "rotation_error_degrees",
    "summarize_pose_errors",
    "summarize_multishape_pose_errors",
    "transformed_model_points",
    "translation_error_m",
    "yaw_error_degrees",
]
