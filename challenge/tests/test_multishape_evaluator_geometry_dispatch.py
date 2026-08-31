"""Manifest-driven geometry dispatch and fail-closed multi-shape metrics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from challenge.evaluation_v2 import paper_real_eval
from challenge.evaluation_v2.pnp_selector import PnPSelectionResult, SelectorStatus
from challenge.evaluation_v2.pose_metrics import (
    PoseErrorRecord,
    PoseMetricGate,
    summarize_multishape_pose_errors,
)
from challenge.evaluation_v2.real_dataset_contract import (
    ContractError,
    PopulationId,
    load_repo_population,
)
from scripts.annotate.object_geometry_registry import (
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


def _passed_gate() -> PoseMetricGate:
    return PoseMetricGate(
        canonical_migration_status="PASS",
        selector_status="PASS",
        symmetry_status="FROZEN",
        final_manifest_status="FROZEN",
        passed=True,
        blocked_reasons=(),
    )


def _target(object_type: str, dimensions) -> paper_real_eval.PositiveTarget:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.0, 0.0, 3.0]
    return paper_real_eval.PositiveTarget(
        frame_id=object_type,
        box_xyxy=np.array([0.0, 0.0, 10.0, 10.0]),
        keypoints_xy=np.zeros((9, 2), dtype=np.float64),
        keypoint_xy_present=np.ones(9, dtype=bool),
        keypoint_supervision_mask=np.ones(9, dtype=bool),
        visibility=np.full(9, 2, dtype=np.int64),
        camera_intrinsics=np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        canonical_pose_transform=transform,
        canonical_pose_candidate_transforms=(),
        object_type=object_type,
        physical_dimensions=dimensions,
    )


def test_actual_plastic_and_wood_labels_resolve_registry_geometry() -> None:
    registry = load_object_geometry_registry()
    plastic_item = load_repo_population(PopulationId.COMMON_DEV_PLASTIC_POS128).items[0]
    wood_item = load_repo_population(PopulationId.DEV_WOOD_POS45).items[0]
    plastic = paper_real_eval._legacy_forbidden_target(plastic_item, registry)
    wood = paper_real_eval._legacy_forbidden_target(wood_item, registry)
    assert plastic.object_type == PLASTIC_OBJECT_TYPE
    assert plastic.physical_dimensions == registry.resolve(PLASTIC_OBJECT_TYPE).physical_dimensions
    assert wood.object_type == WOOD_OBJECT_TYPE
    assert wood.physical_dimensions == registry.resolve(WOOD_OBJECT_TYPE).physical_dimensions
    assert plastic.physical_dimensions != wood.physical_dimensions


def test_label_dimension_poisoning_cannot_override_manifest_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_object_geometry_registry()
    item = load_repo_population(PopulationId.DEV_WOOD_POS45).items[0]
    payload = json.loads((paper_real_eval.REPO_ROOT / item.label).read_text("utf-8"))
    payload["objects"][0]["physical_dimensions_m"] = {
        "x": 1.10,
        "y": 0.11,
        "z": 1.30,
    }
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(payload), "utf-8")
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)
    with pytest.raises(ContractError, match="DIMENSIONS|INVALID_GT_V2_SCHEMA"):
        paper_real_eval._legacy_forbidden_target(
            replace(item, label=poisoned.name), registry
        )


def test_unknown_manifest_object_type_is_rejected_before_label_fallback() -> None:
    registry = load_object_geometry_registry()
    item = load_repo_population(PopulationId.DEV_WOOD_POS45).items[0]
    with pytest.raises(ContractError, match="MANIFEST_UNKNOWN_OBJECT_TYPE"):
        paper_real_eval._legacy_forbidden_target(
            replace(item, object_type="unknown_pallet"), registry
        )


def test_pose_selector_receives_each_objects_registry_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_object_geometry_registry()
    plastic_spec = registry.resolve(PLASTIC_OBJECT_TYPE)
    wood_spec = registry.resolve(WOOD_OBJECT_TYPE)
    targets = {
        PLASTIC_OBJECT_TYPE: _target(
            PLASTIC_OBJECT_TYPE, plastic_spec.physical_dimensions
        ),
        WOOD_OBJECT_TYPE: _target(WOOD_OBJECT_TYPE, wood_spec.physical_dimensions),
    }
    predictions = {
        frame_id: paper_real_eval.DetectionCandidate(
            frame_id=frame_id,
            is_positive=True,
            score=1.0,
            box_xyxy=target.box_xyxy,
            keypoints_xy=np.zeros((9, 2), dtype=np.float64),
            target_iou=1.0,
        )
        for frame_id, target in targets.items()
    }
    calls = []

    def fake_selector(_keypoints, _intrinsics, physical_dimensions):
        calls.append(physical_dimensions)
        return PnPSelectionResult(
            status=SelectorStatus.FAILED,
            selected_hypothesis=None,
            hypotheses=(),  # type: ignore[arg-type]
            canonical_candidates=(),
            ambiguity="TEST_FAILURE_AFTER_DIMENSION_CAPTURE",
        )

    monkeypatch.setattr(paper_real_eval, "select_pnp_hypotheses", fake_selector)
    rotations = (np.eye(3, dtype=np.float64),)
    children = {
        object_type: paper_real_eval.PoseContractContext(
            gate=_passed_gate(),
            metric_variant="ADD-S",
            equivalent_rotations=rotations,
            pose_resolution_mode=paper_real_eval.SIGNED_CANONICAL_POSE,
            evidence={},
        )
        for object_type in targets
    }
    context = paper_real_eval.PoseContractContext(
        gate=_passed_gate(),
        metric_variant="ADD-S",
        equivalent_rotations=(),
        pose_resolution_mode="OBJECT_SPECIFIC",
        evidence={},
        object_contracts=children,
        geometry_registry_sha256=registry.sha256,
    )
    records = paper_real_eval.evaluate_pose_records(targets, predictions, context)
    assert len(records) == 2
    assert calls == [plastic_spec.physical_dimensions, wood_spec.physical_dimensions]
    assert records[0].object_diameter_m != records[1].object_diameter_m


def test_all_pose_is_null_when_any_required_object_gate_is_blocked() -> None:
    plastic_record = PoseErrorRecord(
        add_error_m=0.0,
        adds_error_m=0.0,
        object_diameter_m=1.0,
        rotation_error_deg=0.0,
        translation_error_m=0.0,
        yaw_error_deg=0.0,
        object_type=PLASTIC_OBJECT_TYPE,
    )
    wood_gate = PoseMetricGate(
        canonical_migration_status="BLOCKED",
        selector_status="NOT_RUN",
        symmetry_status="UNREVIEWED",
        final_manifest_status="NOT_FROZEN",
        passed=False,
        blocked_reasons=("WOOD_SYMMETRY_UNREVIEWED",),
    )

    class Poison:
        def __iter__(self):
            raise AssertionError("blocked wood records must never be iterated")

    result = summarize_multishape_pose_errors(
        {
            PLASTIC_OBJECT_TYPE: (plastic_record,),
            WOOD_OBJECT_TYPE: Poison(),
        },
        {PLASTIC_OBJECT_TYPE: _passed_gate(), WOOD_OBJECT_TYPE: wood_gate},
        {PLASTIC_OBJECT_TYPE: "ADD-S", WOOD_OBJECT_TYPE: "ADD-S"},
    )
    assert result["objects"][PLASTIC_OBJECT_TYPE]["status"] == "READY"
    assert result["objects"][WOOD_OBJECT_TYPE]["status"] == "BLOCKED"
    assert result["ALL"]["status"] == "BLOCKED"
    assert all(result["ALL"][field] is None for field in paper_real_eval.POSE_METRIC_FIELDS)


def test_blocked_wood_skips_wood_pnp_but_keeps_passed_plastic_subgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_object_geometry_registry()
    plastic_spec = registry.resolve(PLASTIC_OBJECT_TYPE)
    wood_spec = registry.resolve(WOOD_OBJECT_TYPE)
    targets = {
        PLASTIC_OBJECT_TYPE: _target(
            PLASTIC_OBJECT_TYPE, plastic_spec.physical_dimensions
        ),
        WOOD_OBJECT_TYPE: _target(WOOD_OBJECT_TYPE, wood_spec.physical_dimensions),
    }
    predictions = {
        frame_id: paper_real_eval.DetectionCandidate(
            frame_id=frame_id,
            is_positive=True,
            score=1.0,
            box_xyxy=target.box_xyxy,
            keypoints_xy=np.zeros((9, 2), dtype=np.float64),
            target_iou=1.0,
        )
        for frame_id, target in targets.items()
    }
    calls = []

    def fake_selector(_keypoints, _intrinsics, physical_dimensions):
        calls.append(physical_dimensions)
        return PnPSelectionResult(
            status=SelectorStatus.FAILED,
            selected_hypothesis=None,
            hypotheses=(),  # type: ignore[arg-type]
            canonical_candidates=(),
            ambiguity="TEST_FAILURE_AFTER_DIMENSION_CAPTURE",
        )

    monkeypatch.setattr(paper_real_eval, "select_pnp_hypotheses", fake_selector)
    rotations = (np.eye(3, dtype=np.float64),)
    plastic_context = paper_real_eval.PoseContractContext(
        gate=_passed_gate(),
        metric_variant="ADD-S",
        equivalent_rotations=rotations,
        pose_resolution_mode=paper_real_eval.SIGNED_CANONICAL_POSE,
        evidence={},
    )
    wood_gate = PoseMetricGate(
        canonical_migration_status="NOT_RUN",
        selector_status="NOT_RUN",
        symmetry_status="UNREVIEWED",
        final_manifest_status="NOT_FROZEN",
        passed=False,
        blocked_reasons=("WOOD_SELECTOR_NOT_RUN", "WOOD_SYMMETRY_UNREVIEWED"),
    )
    wood_context = paper_real_eval.PoseContractContext(
        gate=wood_gate,
        metric_variant="ADD-S",
        equivalent_rotations=(),
        pose_resolution_mode=None,
        evidence={},
    )
    aggregate_gate = PoseMetricGate(
        canonical_migration_status="OBJECT_SPECIFIC",
        selector_status="OBJECT_SPECIFIC",
        symmetry_status="OBJECT_SPECIFIC",
        final_manifest_status="NOT_FROZEN",
        passed=False,
        blocked_reasons=("wood_small_80x59x14:WOOD_SELECTOR_NOT_RUN",),
    )
    context = paper_real_eval.PoseContractContext(
        gate=aggregate_gate,
        metric_variant="ADD-S",
        equivalent_rotations=(),
        pose_resolution_mode="OBJECT_SPECIFIC",
        evidence={},
        object_contracts={
            PLASTIC_OBJECT_TYPE: plastic_context,
            WOOD_OBJECT_TYPE: wood_context,
        },
        geometry_registry_sha256=registry.sha256,
    )

    records = paper_real_eval.evaluate_pose_records(targets, predictions, context)
    assert len(records) == 1
    assert records[0].object_type == PLASTIC_OBJECT_TYPE
    assert calls == [plastic_spec.physical_dimensions]

    metrics = paper_real_eval._pose_metrics_with_subgroups(records, targets, context)
    assert metrics["status"] == "BLOCKED"
    assert metrics["subgroups"]["PLASTIC"]["status"] == "READY"
    assert metrics["subgroups"]["PLASTIC"]["pose_population_count"] == 1
    assert metrics["subgroups"]["WOOD"]["status"] == "BLOCKED"
    assert metrics["subgroups"]["ALL"]["status"] == "BLOCKED"
    assert all(
        metrics["subgroups"]["ALL"][field] is None
        for field in paper_real_eval.POSE_METRIC_FIELDS
    )
