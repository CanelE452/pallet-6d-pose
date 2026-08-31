"""Frozen DEV140 selector-diagnostic runner contract."""

from __future__ import annotations

import json

from challenge.evaluation_v2 import selector_diagnostic
from challenge.evaluation_v2.real_dataset_contract import PopulationId, load_repo_population


def _tail_rows(*, both_top_fail: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(20):
        # The top two errors are indices 18 and 19.  A 50% failure share is not
        # a strict majority and therefore passes; 100% fails.
        correct = not (index == 19 or (both_top_fail and index == 18))
        rows.append(
            {
                "frame_id": f"{index:03d}",
                "correct": correct,
                "restricted_adds_normalized": float(index),
                "rotation_error_deg": float(index),
                "translation_error_m": float(index),
                "yaw_error_deg": float(index),
            }
        )
    return rows


def test_tail_rule_is_frozen_worst_decile_strict_majority() -> None:
    boundary = selector_diagnostic.assess_tail_dominance(
        _tail_rows(both_top_fail=False)
    )
    assert boundary["tail_fraction"] == 0.10
    assert boundary["strict_failure_majority_threshold"] == 0.50
    assert boundary["passed"] is True
    assert all(row["tail_n"] == 2 for row in boundary["metrics"].values())
    assert all(
        row["selector_failure_fraction"] == 0.5
        for row in boundary["metrics"].values()
    )

    dominated = selector_diagnostic.assess_tail_dominance(
        _tail_rows(both_top_fail=True)
    )
    assert dominated["passed"] is False
    assert all(row["dominated"] is True for row in dominated["metrics"].values())


def test_invalid_selector_result_is_ranked_as_infinite_tail_failure() -> None:
    rows = _tail_rows(both_top_fail=False)
    rows[0]["correct"] = False
    rows[0]["restricted_adds_normalized"] = None
    rows[0]["rotation_error_deg"] = None
    rows[0]["translation_error_m"] = None
    rows[0]["yaw_error_deg"] = None
    report = selector_diagnostic.assess_tail_dominance(rows)
    assert report["passed"] is False
    assert all("000" in row["frame_ids"] for row in report["metrics"].values())


def test_current_dev140_truth_contains_only_two_frozen_parities() -> None:
    manifest = load_repo_population(PopulationId.DEV_POS140, validate_files=True)
    intrinsics = selector_diagnostic._load_intrinsics_manifest(
        selector_diagnostic.DEFAULT_INTRINSICS_MANIFEST,
        manifest,
    )
    assert tuple(intrinsics) == manifest.frame_ids
    assert all(matrix.shape == (3, 3) for matrix in intrinsics.values())
    camera_only = json.loads(
        selector_diagnostic.DEFAULT_INTRINSICS_MANIFEST.read_text("utf-8")
    )
    assert set(camera_only) == {
        "schema_version",
        "population_id",
        "population_membership_sha256",
        "count",
        "purpose",
        "records",
    }
    assert all(
        set(record)
        == {"frame_id", "fx", "fy", "cx", "cy", "source_label_sha256"}
        for record in camera_only["records"]
    )
    expected = {
        selector_diagnostic._diagnostic_truth(item)[0] for item in manifest.items
    }
    assert expected == {"short-face-front", "long-face-front"}


def test_selector_diagnostic_cli_requires_manifest_weights_and_output() -> None:
    parser = selector_diagnostic.build_parser()
    actions = {action.dest: action for action in parser._actions}
    assert actions["manifest"].required is True
    assert actions["weights"].required is True
    assert actions["out_dir"].required is True
