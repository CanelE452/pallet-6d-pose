"""Prediction-only W/D hypothesis selection for camera-facing pallet keypoints.

The public selector receives only nine predicted image points, camera
intrinsics, the fixed physical pallet dimensions and a fixed configuration.
It never receives a label JSON, a pose, an axis answer or a session prior.

A W/D parity decision cannot determine the sign of the canonical physical
axes.  Consequently each solved parity returns two signed canonical pose
candidates (0/180 or 90/270 degrees).  The API intentionally has no singular
``canonical_pose`` field.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.annotate import pallet_geometry as geometry


OVERALL_AXIS_ACCURACY_MIN = 0.95
NIGHT_AXIS_ACCURACY_MIN = 0.90
SESSION_AXIS_ACCURACY_MIN = 0.85

EXPECTED_DEV140_SESSION_COUNTS: Mapping[str, int] = {
    "eval_cad": 18,
    "eval_night08": 12,
    "eval_night09": 16,
    "eval_noapril": 12,
    "eval_outside": 22,
    "eval_pallet07": 27,
    "eval_pallet09": 33,
}
EXPECTED_DEV140_COUNT = sum(EXPECTED_DEV140_SESSION_COUNTS.values())
EXPECTED_DEV140_NIGHT_COUNT = 28

PLASTIC_OBJECT_TYPE = "plastic_standard_110x130x11"
WOOD_OBJECT_TYPE = "wood_small_80x59x14"
SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT: Mapping[str, str] = {
    PLASTIC_OBJECT_TYPE: "DEV_POS140",
    WOOD_OBJECT_TYPE: "DEV_WOOD_POS45",
}

LR_PAIRS = ((0, 1), (3, 2), (4, 5), (7, 6))
TB_PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6))
FR_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7))


class SelectorStatus(str, Enum):
    SELECTED = "SELECTED_PARITY_SIGN_AMBIGUOUS"
    AMBIGUOUS = "AMBIGUOUS_PARITY"
    FAILED = "FAILED"


class SelectorGateState(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class SelectorConfig:
    """Fixed, prediction-only scoring constants.

    All terms are penalties and lower total score is better.  The defaults are
    engineering priors, not values tuned against real pose labels.
    """

    reprojection_weight: float = 1.0
    cheirality_weight: float = 10_000.0
    invariant_weight: float = 100.0
    upright_weight: float = 25.0
    degeneracy_weight: float = 100.0
    upright_soft_min: float = 0.30
    depth_epsilon_m: float = 1e-6
    min_spread_ratio: float = 1e-3
    parity_tie_tolerance: float = 1e-6
    refine_lm: bool = True

    def __post_init__(self) -> None:
        numeric_nonnegative = (
            "reprojection_weight",
            "cheirality_weight",
            "invariant_weight",
            "upright_weight",
            "degeneracy_weight",
            "upright_soft_min",
            "depth_epsilon_m",
            "min_spread_ratio",
            "parity_tie_tolerance",
        )
        for name in numeric_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.upright_soft_min > 1.0:
            raise ValueError("upright_soft_min must be <= 1")
        if self.min_spread_ratio > 1.0:
            raise ValueError("min_spread_ratio must be <= 1")


@dataclass(frozen=True)
class CanonicalPoseCandidate:
    axis_assignment: geometry.AxisAssignment
    rotation: np.ndarray
    translation: np.ndarray
    pose_transform: np.ndarray
    keypoint_permutation: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_assignment": self.axis_assignment.value,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "pose_transform": self.pose_transform.tolist(),
            "canonical_to_camera_facing_keypoint_permutation": list(self.keypoint_permutation),
        }


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    camera_facing_dimensions: geometry.CameraFacingDimensionsWHD
    success: bool
    score: float | None
    score_components: Mapping[str, float | int | bool | None]
    rotation_camera_facing: np.ndarray | None
    translation_camera_facing: np.ndarray | None
    projected_keypoints: np.ndarray | None
    canonical_candidates: tuple[CanonicalPoseCandidate, ...]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "camera_facing_dimensions_m": self.camera_facing_dimensions.as_dict(),
            "success": self.success,
            "score": self.score,
            "score_components": dict(self.score_components),
            "rotation_camera_facing": (
                self.rotation_camera_facing.tolist()
                if self.rotation_camera_facing is not None
                else None
            ),
            "translation_camera_facing": (
                self.translation_camera_facing.tolist()
                if self.translation_camera_facing is not None
                else None
            ),
            "projected_keypoints": (
                self.projected_keypoints.tolist() if self.projected_keypoints is not None else None
            ),
            "canonical_candidates": [candidate.to_dict() for candidate in self.canonical_candidates],
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class PnPSelectionResult:
    status: SelectorStatus
    selected_hypothesis: str | None
    hypotheses: tuple[HypothesisResult, HypothesisResult]
    canonical_candidates: tuple[CanonicalPoseCandidate, ...]
    ambiguity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "selected_hypothesis": self.selected_hypothesis,
            "ambiguity": self.ambiguity,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "canonical_candidates": [candidate.to_dict() for candidate in self.canonical_candidates],
        }


@dataclass(frozen=True)
class SelectorGateReport:
    status: SelectorGateState
    overall_accuracy: float | None
    night_accuracy: float | None
    minimum_session_accuracy: float | None
    sample_count: int
    night_count: int
    session_count: int
    tail_dominance_assessed: bool
    tail_dominance_passed: bool | None
    tail_dominance_notes: str | None
    blocked_reason: str | None
    object_type: str = PLASTIC_OBJECT_TYPE
    population_id: str = "DEV_POS140"
    population_role: str = "DEV"
    population_membership_sha256: str | None = None
    expected_sample_count: int = EXPECTED_DEV140_COUNT
    expected_night_count: int = EXPECTED_DEV140_NIGHT_COUNT
    expected_session_counts: tuple[tuple[str, int], ...] = tuple(
        EXPECTED_DEV140_SESSION_COUNTS.items()
    )
    population_validated: bool = True

    @classmethod
    def not_run(
        cls,
        *,
        object_type: str = PLASTIC_OBJECT_TYPE,
        population_id: str = "DEV_POS140",
        population_role: str = "DEV",
        population_membership_sha256: str | None = None,
        expected_sample_count: int = EXPECTED_DEV140_COUNT,
        expected_night_count: int = EXPECTED_DEV140_NIGHT_COUNT,
        expected_session_counts: Mapping[str, int] = EXPECTED_DEV140_SESSION_COUNTS,
    ) -> "SelectorGateReport":
        return cls(
            status=SelectorGateState.NOT_RUN,
            overall_accuracy=None,
            night_accuracy=None,
            minimum_session_accuracy=None,
            sample_count=0,
            night_count=0,
            session_count=0,
            tail_dominance_assessed=False,
            tail_dominance_passed=None,
            tail_dominance_notes=None,
            blocked_reason="SELECTOR_DIAGNOSTIC_NOT_RUN",
            object_type=object_type,
            population_id=population_id,
            population_role=population_role,
            population_membership_sha256=population_membership_sha256,
            expected_sample_count=expected_sample_count,
            expected_night_count=expected_night_count,
            expected_session_counts=tuple(expected_session_counts.items()),
            population_validated=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "overall_accuracy": self.overall_accuracy,
            "night_accuracy": self.night_accuracy,
            "minimum_session_accuracy": self.minimum_session_accuracy,
            "sample_count": self.sample_count,
            "night_count": self.night_count,
            "session_count": self.session_count,
            "expected_population": {
                "object_type": self.object_type,
                "population_id": self.population_id,
                "population_role": self.population_role,
                "membership_sha256": self.population_membership_sha256,
                "total": self.expected_sample_count,
                "night": self.expected_night_count,
                "session_counts": dict(self.expected_session_counts),
            },
            "population_validated": self.population_validated,
            "tail_dominance_assessed": self.tail_dominance_assessed,
            "tail_dominance_passed": self.tail_dominance_passed,
            "tail_dominance_notes": self.tail_dominance_notes,
            "thresholds": {
                "overall_min": OVERALL_AXIS_ACCURACY_MIN,
                "night_min": (
                    NIGHT_AXIS_ACCURACY_MIN
                    if self.expected_night_count > 0
                    else None
                ),
                "minimum_session_min": SESSION_AXIS_ACCURACY_MIN,
            },
            "blocked_reason": self.blocked_reason,
        }


def _physical_dimensions(
    value: geometry.PhysicalDimensionsXYZ | Mapping[str, float],
) -> geometry.PhysicalDimensionsXYZ:
    if isinstance(value, geometry.PhysicalDimensionsXYZ):
        dimensions = value
    elif isinstance(value, Mapping):
        try:
            dimensions = geometry.PhysicalDimensionsXYZ(
                x_m=float(value["x"]), y_m=float(value["y"]), z_m=float(value["z"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("physical_dimensions must provide named x/y/z values") from exc
    else:
        raise TypeError(
            "physical_dimensions must be PhysicalDimensionsXYZ or a named x/y/z mapping; "
            "positional W/D/H tuples are forbidden"
        )

    return dimensions


def _inputs(
    predicted_keypoints: Sequence[Sequence[float]] | np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(predicted_keypoints, dtype=np.float64)
    camera = np.asarray(camera_intrinsics, dtype=np.float64)
    if points.shape != (9, 2) or not np.isfinite(points).all():
        raise ValueError("predicted_keypoints must be a finite (9,2) array")
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("camera_intrinsics must be a finite (3,3) matrix")
    if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
        raise ValueError("camera_intrinsics focal lengths must be positive")
    if not np.allclose(camera[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
        raise ValueError("camera_intrinsics last row must be [0,0,1]")
    return points, camera


def _spread_ratio(points: np.ndarray) -> float:
    centered = points[:8] - np.mean(points[:8], axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) < 2 or singular_values[0] <= 1e-12:
        return 0.0
    return float(singular_values[1] / singular_values[0])


def _canonical_candidates(
    rotation_cf: np.ndarray,
    translation_cf: np.ndarray,
    dimensions_cf: geometry.CameraFacingDimensionsWHD,
    physical_dimensions: geometry.PhysicalDimensionsXYZ,
) -> tuple[CanonicalPoseCandidate, CanonicalPoseCandidate]:
    assignments = geometry.axis_assignment_candidates_from_camera_facing_dimensions(
        dimensions_cf, physical_dimensions=physical_dimensions
    )
    out: list[CanonicalPoseCandidate] = []
    for assignment in assignments:
        rotation, translation = geometry.camera_facing_to_canonical_pose(
            rotation_cf, translation_cf, assignment
        )
        out.append(
            CanonicalPoseCandidate(
                axis_assignment=assignment,
                rotation=rotation,
                translation=translation,
                pose_transform=geometry.make_pose_transform(rotation, translation),
                keypoint_permutation=(
                    geometry.canonical_to_camera_facing_keypoint_permutation(
                        assignment, physical_dimensions
                    )
                ),
            )
        )
    return (out[0], out[1])


def _failed_hypothesis(
    name: str,
    dimensions: geometry.CameraFacingDimensionsWHD,
    reason: str,
    spread_ratio: float,
) -> HypothesisResult:
    return HypothesisResult(
        name=name,
        camera_facing_dimensions=dimensions,
        success=False,
        score=None,
        score_components={
            "reprojection_rmse_px": None,
            "cheirality_fraction": None,
            "invariant_violations": None,
            "upright_alignment": None,
            "finite": False,
            "spread_ratio": spread_ratio,
            "degenerate": True,
        },
        rotation_camera_facing=None,
        translation_camera_facing=None,
        projected_keypoints=None,
        canonical_candidates=(),
        failure_reason=reason,
    )


def _solve_hypothesis(
    name: str,
    representative_assignment: geometry.AxisAssignment,
    points: np.ndarray,
    camera: np.ndarray,
    config: SelectorConfig,
    physical_dimensions: geometry.PhysicalDimensionsXYZ,
) -> HypothesisResult:
    # OpenCV is intentionally lazy: contract/dry-run tests do not require the
    # inference environment, while actual PnP uses the pallet-pose environment.
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only outside the PnP env
        dimensions = geometry.camera_facing_dimensions(
            representative_assignment, physical_dimensions
        )
        return _failed_hypothesis(name, dimensions, f"OPENCV_UNAVAILABLE: {exc}", 0.0)

    object_points = geometry.camera_facing_keypoints_3d(
        representative_assignment, physical_dimensions
    )
    dimensions = geometry.camera_facing_dimensions(
        representative_assignment, physical_dimensions
    )
    spread_ratio = _spread_ratio(points)
    try:
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            object_points[:8],
            points[:8],
            camera,
            None,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return _failed_hypothesis(name, dimensions, "SQPNP_FAILED", spread_ratio)
        if config.refine_lm:
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                object_points[:8],
                points[:8],
                camera,
                None,
                rotation_vector,
                translation_vector,
            )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
        geometry.validate_proper_rotation(rotation, name=f"{name}.R_cf")
        projected, _ = cv2.projectPoints(
            object_points, rotation_vector, translation.reshape(3, 1), camera, None
        )
        projected = projected.reshape(9, 2)
    except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
        return _failed_hypothesis(name, dimensions, f"PNP_EXCEPTION: {exc}", spread_ratio)

    camera_points = (rotation @ object_points.T).T + translation
    finite = bool(
        np.isfinite(rotation).all()
        and np.isfinite(translation).all()
        and np.isfinite(projected).all()
        and np.isfinite(camera_points).all()
    )
    if not finite:
        return _failed_hypothesis(name, dimensions, "NONFINITE_PNP_RESULT", spread_ratio)

    residual = np.linalg.norm(projected - points, axis=1)
    reprojection_rmse = float(np.sqrt(np.mean(np.square(residual))))
    cheirality_fraction = float(np.mean(camera_points[:, 2] > config.depth_epsilon_m))
    lr_violations = sum(not (projected[a, 0] < projected[b, 0]) for a, b in LR_PAIRS)
    tb_violations = sum(not (projected[a, 1] < projected[b, 1]) for a, b in TB_PAIRS)
    fr_violations = sum(
        not (camera_points[a, 2] < camera_points[b, 2]) for a, b in FR_PAIRS
    )
    invariant_violations = int(lr_violations + tb_violations + fr_violations)
    upright_alignment = float(abs(rotation[1, 1]))
    upright_deficit = max(0.0, config.upright_soft_min - upright_alignment)
    degeneracy_deficit = max(0.0, config.min_spread_ratio - spread_ratio)
    degeneracy_fraction = (
        degeneracy_deficit / config.min_spread_ratio if config.min_spread_ratio > 0.0 else 0.0
    )
    score = float(
        config.reprojection_weight * reprojection_rmse
        + config.cheirality_weight * (1.0 - cheirality_fraction)
        + config.invariant_weight * invariant_violations
        + config.upright_weight * upright_deficit
        + config.degeneracy_weight * degeneracy_fraction
    )
    if not math.isfinite(score):
        return _failed_hypothesis(name, dimensions, "NONFINITE_SELECTOR_SCORE", spread_ratio)

    return HypothesisResult(
        name=name,
        camera_facing_dimensions=dimensions,
        success=True,
        score=score,
        score_components={
            "reprojection_rmse_px": reprojection_rmse,
            "cheirality_fraction": cheirality_fraction,
            "lr_violations": int(lr_violations),
            "tb_violations": int(tb_violations),
            "front_rear_violations": int(fr_violations),
            "invariant_violations": invariant_violations,
            "upright_alignment": upright_alignment,
            "finite": finite,
            "spread_ratio": spread_ratio,
            "degenerate": spread_ratio < config.min_spread_ratio,
        },
        rotation_camera_facing=rotation,
        translation_camera_facing=translation,
        projected_keypoints=projected,
        canonical_candidates=_canonical_candidates(
            rotation, translation, dimensions, physical_dimensions
        ),
        failure_reason=None,
    )


def select_pnp_hypotheses(
    predicted_keypoints: Sequence[Sequence[float]] | np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]] | np.ndarray,
    physical_dimensions: geometry.PhysicalDimensionsXYZ | Mapping[str, float],
    config: SelectorConfig | None = None,
) -> PnPSelectionResult:
    """Solve and score both physical W/D parities without evaluation labels.

    Returns both hypothesis records and every remaining signed canonical pose
    candidate.  A parity tie remains explicit instead of being broken with a
    hidden answer or session-specific prior.
    """

    points, camera = _inputs(predicted_keypoints, camera_intrinsics)
    physical = _physical_dimensions(physical_dimensions)
    config = config or SelectorConfig()
    if not isinstance(config, SelectorConfig):
        raise TypeError("config must be SelectorConfig")

    yaw0 = _solve_hypothesis(
        geometry.camera_facing_hypothesis_name(
            geometry.AxisAssignment.YAW_0, physical
        ),
        geometry.AxisAssignment.YAW_0,
        points,
        camera,
        config,
        physical,
    )
    yaw90 = _solve_hypothesis(
        geometry.camera_facing_hypothesis_name(
            geometry.AxisAssignment.YAW_90, physical
        ),
        geometry.AxisAssignment.YAW_90,
        points,
        camera,
        config,
        physical,
    )
    hypotheses = (yaw0, yaw90)
    valid = [hypothesis for hypothesis in hypotheses if hypothesis.success]
    if not valid:
        return PnPSelectionResult(
            status=SelectorStatus.FAILED,
            selected_hypothesis=None,
            hypotheses=hypotheses,
            canonical_candidates=(),
            ambiguity="NO_VALID_WD_HYPOTHESIS",
        )
    if len(valid) == 1:
        selected = valid[0]
        return PnPSelectionResult(
            status=SelectorStatus.SELECTED,
            selected_hypothesis=selected.name,
            hypotheses=hypotheses,
            canonical_candidates=selected.canonical_candidates,
            ambiguity="SIGNED_AXIS_UNRESOLVED_TWO_CANDIDATES",
        )

    assert valid[0].score is not None and valid[1].score is not None
    if abs(valid[0].score - valid[1].score) <= config.parity_tie_tolerance:
        return PnPSelectionResult(
            status=SelectorStatus.AMBIGUOUS,
            selected_hypothesis=None,
            hypotheses=hypotheses,
            canonical_candidates=tuple(
                candidate for hypothesis in valid for candidate in hypothesis.canonical_candidates
            ),
            ambiguity="WD_PARITY_TIED_AND_SIGNED_AXIS_UNRESOLVED",
        )
    selected = min(valid, key=lambda hypothesis: float(hypothesis.score))
    return PnPSelectionResult(
        status=SelectorStatus.SELECTED,
        selected_hypothesis=selected.name,
        hypotheses=hypotheses,
        canonical_candidates=selected.canonical_candidates,
        ambiguity="SIGNED_AXIS_UNRESOLVED_TWO_CANDIDATES",
    )


def assess_selector_diagnostics(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    tail_dominance_assessed: bool = False,
    tail_dominance_passed: bool | None = None,
    tail_dominance_notes: str | None = None,
    object_type: str = PLASTIC_OBJECT_TYPE,
    population_id: str | None = None,
) -> SelectorGateReport:
    """Apply an object-bound, pre-registered selector diagnostic gate.

    Each record must contain ``frame_id``, ``correct`` (bool), ``domain`` and
    ``session``.  Frame IDs and strata must match the object's frozen diagnostic
    population exactly.  These values are used only for post-hoc gate reporting
    and are not accepted by :func:`select_pnp_hypotheses`.
    """

    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError("object_type must be a non-empty string")
    expected_population_id = SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT.get(object_type)
    if expected_population_id is None:
        raise ValueError(f"no preregistered selector population for {object_type!r}")
    declared_population_id = (
        getattr(population_id, "value", population_id)
        if population_id is not None
        else expected_population_id
    )
    if declared_population_id != expected_population_id:
        raise ValueError(
            f"{object_type}: selector population must be {expected_population_id}, "
            f"got {declared_population_id!r}"
        )

    # Import only in the post-hoc diagnostic path.  The public inference
    # selector remains independent of manifests and ground truth.
    from .real_dataset_contract import load_repo_population

    manifest = load_repo_population(declared_population_id, validate_files=True)
    manifest_types = {
        item.object_type or PLASTIC_OBJECT_TYPE for item in manifest.items
    }
    if manifest_types != {object_type}:
        raise ValueError(
            f"{declared_population_id}: manifest object type does not match {object_type}"
        )
    expected_strata = {
        item.frame_id: (item.domain, item.session_id or item.source_set)
        for item in manifest.items
    }
    expected_session_counts = Counter(
        session for _, session in expected_strata.values()
    )
    expected_night_count = sum(
        domain == "NIGHT" for domain, _ in expected_strata.values()
    )
    binding = {
        "object_type": object_type,
        "population_id": manifest.population_id.value,
        "population_role": manifest.role.value,
        "population_membership_sha256": manifest.membership_sha256,
        "expected_sample_count": manifest.count,
        "expected_night_count": expected_night_count,
        "expected_session_counts": tuple(sorted(expected_session_counts.items())),
    }
    if not records:
        return SelectorGateReport.not_run(
            object_type=object_type,
            population_id=manifest.population_id.value,
            population_role=manifest.role.value,
            population_membership_sha256=manifest.membership_sha256,
            expected_sample_count=manifest.count,
            expected_night_count=expected_night_count,
            expected_session_counts=dict(sorted(expected_session_counts.items())),
        )
    if not isinstance(tail_dominance_assessed, bool):
        raise TypeError("tail_dominance_assessed must be bool")
    if tail_dominance_passed is not None and not isinstance(tail_dominance_passed, bool):
        raise TypeError("tail_dominance_passed must be bool or None")
    if tail_dominance_assessed and tail_dominance_passed is None:
        raise ValueError("tail_dominance_passed is required when tail dominance was assessed")
    if not tail_dominance_assessed and tail_dominance_passed is not None:
        raise ValueError("tail_dominance_passed must be None when assessment was not run")
    if tail_dominance_notes is not None and not isinstance(tail_dominance_notes, str):
        raise TypeError("tail_dominance_notes must be str or None")
    normalized: list[tuple[str, bool, str | None, str]] = []
    for index, record in enumerate(records):
        frame_id = record.get("frame_id")
        correct = record.get("correct")
        domain = record.get("domain")
        session = record.get("session")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"records[{index}].frame_id must be a non-empty string")
        if not isinstance(correct, bool):
            raise ValueError(f"records[{index}].correct must be bool")
        if domain not in {"DAY", "NIGHT", None}:
            raise ValueError(f"records[{index}].domain must be DAY, NIGHT or null")
        if not isinstance(session, str) or not session:
            raise ValueError(f"records[{index}].session must be a non-empty string")
        normalized.append((frame_id, correct, domain, session))

    overall = float(np.mean([correct for _, correct, _, _ in normalized]))
    night_rows = [
        correct for _, correct, domain, _ in normalized if domain == "NIGHT"
    ]
    sessions = sorted({session for _, _, _, session in normalized})
    session_accuracy = [
        float(
            np.mean(
                [
                    correct
                    for _, correct, _, row_session in normalized
                    if row_session == session
                ]
            )
        )
        for session in sessions
    ]
    night_accuracy = float(np.mean(night_rows)) if night_rows else None
    minimum_session = min(session_accuracy) if session_accuracy else None
    actual_session_counts = {
        session: sum(row_session == session for _, _, _, row_session in normalized)
        for session in sessions
    }
    actual_ids = [frame_id for frame_id, _, _, _ in normalized]
    exact_frame_membership = (
        len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == set(expected_strata)
        and all(
            expected_strata.get(frame_id) == (domain, session)
            for frame_id, _, domain, session in normalized
        )
    )
    population_matches = (
        len(normalized) == manifest.count
        and len(night_rows) == expected_night_count
        and actual_session_counts == dict(expected_session_counts)
        and exact_frame_membership
    )
    if not population_matches:
        return SelectorGateReport(
            status=SelectorGateState.FAIL,
            overall_accuracy=overall,
            night_accuracy=night_accuracy,
            minimum_session_accuracy=minimum_session,
            sample_count=len(normalized),
            night_count=len(night_rows),
            session_count=len(sessions),
            tail_dominance_assessed=tail_dominance_assessed,
            tail_dominance_passed=tail_dominance_passed,
            tail_dominance_notes=tail_dominance_notes,
            blocked_reason="SELECTOR_DIAGNOSTIC_POPULATION_MISMATCH",
            **binding,
            population_validated=False,
        )
    if not tail_dominance_assessed:
        return SelectorGateReport(
            status=SelectorGateState.FAIL,
            overall_accuracy=overall,
            night_accuracy=night_accuracy,
            minimum_session_accuracy=minimum_session,
            sample_count=len(normalized),
            night_count=len(night_rows),
            session_count=len(sessions),
            tail_dominance_assessed=False,
            tail_dominance_passed=None,
            tail_dominance_notes=tail_dominance_notes,
            blocked_reason="SELECTOR_TAIL_DOMINANCE_NOT_ASSESSED",
            **binding,
            population_validated=True,
        )
    night_passed = bool(
        expected_night_count == 0
        or (
            night_accuracy is not None
            and night_accuracy >= NIGHT_AXIS_ACCURACY_MIN
        )
    )
    passed = (
        overall >= OVERALL_AXIS_ACCURACY_MIN
        and night_passed
        and minimum_session is not None
        and minimum_session >= SESSION_AXIS_ACCURACY_MIN
        and tail_dominance_passed is True
    )
    return SelectorGateReport(
        status=SelectorGateState.PASS if passed else SelectorGateState.FAIL,
        overall_accuracy=overall,
        night_accuracy=night_accuracy,
        minimum_session_accuracy=minimum_session,
        sample_count=len(normalized),
        night_count=len(night_rows),
        session_count=len(sessions),
        tail_dominance_assessed=True,
        tail_dominance_passed=tail_dominance_passed,
        tail_dominance_notes=tail_dominance_notes,
        blocked_reason=(
            None
            if passed
            else (
                "SELECTOR_TAIL_DOMINANCE_GATE_FAILED"
                if tail_dominance_passed is False
                else "SELECTOR_ACCURACY_GATE_FAILED"
            )
        ),
        **binding,
        population_validated=True,
    )


__all__ = [
    "CanonicalPoseCandidate",
    "HypothesisResult",
    "EXPECTED_DEV140_COUNT",
    "EXPECTED_DEV140_NIGHT_COUNT",
    "EXPECTED_DEV140_SESSION_COUNTS",
    "NIGHT_AXIS_ACCURACY_MIN",
    "OVERALL_AXIS_ACCURACY_MIN",
    "PLASTIC_OBJECT_TYPE",
    "PnPSelectionResult",
    "SESSION_AXIS_ACCURACY_MIN",
    "SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT",
    "SelectorConfig",
    "SelectorGateReport",
    "SelectorGateState",
    "SelectorStatus",
    "WOOD_OBJECT_TYPE",
    "assess_selector_diagnostics",
    "select_pnp_hypotheses",
]
