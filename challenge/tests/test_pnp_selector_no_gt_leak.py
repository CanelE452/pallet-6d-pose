"""GT-independent PnP selector and its fail-closed promotion gate."""

from __future__ import annotations

import ast
import inspect

import cv2
import numpy as np
import pytest

from challenge.evaluation_v2 import pnp_selector as selector
from challenge.evaluation_v2.real_dataset_contract import PopulationId, load_repo_population
from scripts.annotate import pallet_geometry as geometry


def _camera() -> np.ndarray:
    return np.array(
        [[610.0, 0.0, 320.0], [0.0, 612.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _project(assignment: geometry.AxisAssignment) -> np.ndarray:
    points = geometry.camera_facing_keypoints_3d(assignment)
    rotation_vector = np.array([0.18, -0.11, 0.04], dtype=np.float64)
    translation = np.array([0.12, 0.08, 3.2], dtype=np.float64)
    projected, _ = cv2.projectPoints(points, rotation_vector, translation, _camera(), None)
    return projected.reshape(9, 2)


def _dev140_records(*, correct: bool = True) -> list[dict[str, object]]:
    manifest = load_repo_population(PopulationId.DEV_POS140)
    return [
        {
            "frame_id": item.frame_id,
            "correct": correct,
            "domain": item.domain,
            "session": item.source_set,
        }
        for item in manifest.items
    ]


def test_public_selector_signature_has_no_gt_or_session_prior() -> None:
    signature = inspect.signature(selector.select_pnp_hypotheses)
    assert tuple(signature.parameters) == (
        "predicted_keypoints",
        "camera_intrinsics",
        "physical_dimensions",
        "config",
    )
    forbidden = {"gt", "label", "pose", "axis_assignment", "session", "prior"}
    for parameter in signature.parameters:
        tokens = set(parameter.lower().split("_"))
        assert tokens.isdisjoint(forbidden)

    # The selector body cannot perform file/JSON access.  Diagnostics are a
    # separate function and may consume session strata after inference.
    tree = ast.parse(inspect.getsource(selector.select_pnp_hypotheses))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called.isdisjoint({"open", "load", "loads", "read_text"})


def test_short_width_hypothesis_returns_both_signed_canonical_candidates() -> None:
    result = selector.select_pnp_hypotheses(
        _project(geometry.AxisAssignment.YAW_0),
        _camera(),
        geometry.canonical_dimensions(),
    )
    assert result.status is selector.SelectorStatus.SELECTED
    assert result.selected_hypothesis == "short-face-front"
    assert len(result.hypotheses) == 2
    assert {hypothesis.name for hypothesis in result.hypotheses} == {
        "short-face-front",
        "long-face-front",
    }
    assert {candidate.axis_assignment for candidate in result.canonical_candidates} == {
        geometry.AxisAssignment.YAW_0,
        geometry.AxisAssignment.YAW_180,
    }
    assert not hasattr(result, "canonical_pose")
    for candidate in result.canonical_candidates:
        assert np.linalg.det(candidate.rotation) == pytest.approx(1.0, abs=1e-9)
        assert np.max(np.abs(candidate.rotation.T @ candidate.rotation - np.eye(3))) <= 1e-9
        assert candidate.pose_transform.shape == (4, 4)


def test_long_width_hypothesis_returns_90_and_270_candidates() -> None:
    result = selector.select_pnp_hypotheses(
        _project(geometry.AxisAssignment.YAW_90),
        _camera(),
        {"x": 1.10, "y": 0.11, "z": 1.30},
    )
    assert result.status is selector.SelectorStatus.SELECTED
    assert result.selected_hypothesis == "long-face-front"
    assert {candidate.axis_assignment for candidate in result.canonical_candidates} == {
        geometry.AxisAssignment.YAW_90,
        geometry.AxisAssignment.YAW_270,
    }


def test_both_hypotheses_expose_prediction_only_score_components() -> None:
    result = selector.select_pnp_hypotheses(
        _project(geometry.AxisAssignment.YAW_0),
        _camera(),
        geometry.canonical_dimensions(),
    )
    required = {
        "reprojection_rmse_px",
        "cheirality_fraction",
        "invariant_violations",
        "upright_alignment",
        "finite",
        "spread_ratio",
        "degenerate",
    }
    for hypothesis in result.hypotheses:
        assert hypothesis.success
        assert hypothesis.score is not None and np.isfinite(hypothesis.score)
        assert required <= set(hypothesis.score_components)
        assert len(hypothesis.canonical_candidates) == 2


def test_positional_dimensions_are_rejected_but_named_object_geometry_is_allowed() -> None:
    points = _project(geometry.AxisAssignment.YAW_0)
    with pytest.raises(TypeError, match="positional W/D/H tuples are forbidden"):
        selector.select_pnp_hypotheses(points, _camera(), (1.1, 1.3, 0.11))
    # A prediction-only selector is generic over an already registry-resolved
    # named geometry.  It does not decide the object type from a frame or GT;
    # the paper evaluator performs and validates that manifest dispatch.
    result = selector.select_pnp_hypotheses(
        points, _camera(), {"x": 1.3, "y": 0.11, "z": 1.1}
    )
    assert isinstance(result, selector.PnPSelectionResult)


def test_invalid_prediction_or_camera_fails_before_pnp() -> None:
    with pytest.raises(ValueError, match=r"\(9,2\)"):
        selector.select_pnp_hypotheses(
            np.zeros((8, 2)), _camera(), geometry.canonical_dimensions()
        )
    bad_camera = _camera()
    bad_camera[0, 0] = 0.0
    with pytest.raises(ValueError, match="focal lengths"):
        selector.select_pnp_hypotheses(
            _project(geometry.AxisAssignment.YAW_0),
            bad_camera,
            geometry.canonical_dimensions(),
        )


def test_selector_gate_is_not_run_without_model_output_diagnostics() -> None:
    report = selector.assess_selector_diagnostics(None)
    assert report.status is selector.SelectorGateState.NOT_RUN
    assert report.blocked_reason == "SELECTOR_DIAGNOSTIC_NOT_RUN"
    assert report.overall_accuracy is None
    assert report.tail_dominance_assessed is False
    assert report.tail_dominance_passed is None


def test_small_or_wrong_population_cannot_pass_even_at_100_percent() -> None:
    report = selector.assess_selector_diagnostics(
        [
            {
                "frame_id": "not-a-dev140-member",
                "correct": True,
                "domain": "NIGHT",
                "session": "eval_night08",
            }
        ],
        tail_dominance_assessed=True,
        tail_dominance_passed=True,
    )
    assert report.status is selector.SelectorGateState.FAIL
    assert report.blocked_reason == "SELECTOR_DIAGNOSTIC_POPULATION_MISMATCH"


def test_selector_gate_rejects_duplicate_frame_at_same_stratum_counts() -> None:
    records = _dev140_records()
    first_session = records[0]["session"]
    same_session_indices = [
        index for index, row in enumerate(records) if row["session"] == first_session
    ]
    records[same_session_indices[1]] = dict(records[same_session_indices[0]])
    report = selector.assess_selector_diagnostics(
        records,
        tail_dominance_assessed=True,
        tail_dominance_passed=True,
    )
    assert report.sample_count == 140
    assert report.status is selector.SelectorGateState.FAIL
    assert report.blocked_reason == "SELECTOR_DIAGNOSTIC_POPULATION_MISMATCH"


def test_selector_gate_rejects_swapped_day_night_strata_at_same_counts() -> None:
    records = _dev140_records()
    night_index = next(i for i, row in enumerate(records) if row["domain"] == "NIGHT")
    day_index = next(i for i, row in enumerate(records) if row["domain"] == "DAY")
    records[night_index]["domain"] = "DAY"
    records[day_index]["domain"] = "NIGHT"
    report = selector.assess_selector_diagnostics(
        records,
        tail_dominance_assessed=True,
        tail_dominance_passed=True,
    )
    assert report.sample_count == 140
    assert report.night_count == 28
    assert report.status is selector.SelectorGateState.FAIL
    assert report.blocked_reason == "SELECTOR_DIAGNOSTIC_POPULATION_MISMATCH"


def test_dev140_still_cannot_pass_without_tail_dominance_assessment() -> None:
    report = selector.assess_selector_diagnostics(_dev140_records())
    assert report.sample_count == 140
    assert report.night_count == 28
    assert report.status is selector.SelectorGateState.FAIL
    assert report.blocked_reason == "SELECTOR_TAIL_DOMINANCE_NOT_ASSESSED"


def test_exact_dev140_and_tail_gate_can_pass_at_fixed_thresholds() -> None:
    report = selector.assess_selector_diagnostics(
        _dev140_records(),
        tail_dominance_assessed=True,
        tail_dominance_passed=True,
        tail_dominance_notes="rotation/yaw tail attribution completed",
    )
    assert report.status is selector.SelectorGateState.PASS
    assert report.overall_accuracy == 1.0
    assert report.night_accuracy == 1.0
    assert report.minimum_session_accuracy == 1.0
    assert report.tail_dominance_passed is True


def test_tail_dominance_failure_blocks_an_otherwise_perfect_gate() -> None:
    report = selector.assess_selector_diagnostics(
        _dev140_records(),
        tail_dominance_assessed=True,
        tail_dominance_passed=False,
        tail_dominance_notes="selector failures dominate the yaw tail",
    )
    assert report.status is selector.SelectorGateState.FAIL
    assert report.blocked_reason == "SELECTOR_TAIL_DOMINANCE_GATE_FAILED"
