"""Checked-in selector and actual DEV evaluation artifact invariants."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from challenge.evaluation_v2 import paper_real_eval
from challenge.evaluation_v2.pnp_selector import SelectorStatus
from challenge.evaluation_v2.pose_metrics import build_pose_metric_gate
from challenge.evaluation_v2.real_dataset_contract import (
    ContractError,
    PopulationId,
    load_repo_population,
    manifest_path,
)
from scripts.annotate.object_geometry_registry import (
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
)


SELECTOR = (
    paper_real_eval.REPO_ROOT
    / "challenge/evaluation_v2/selector_diagnostic/SELECTOR_DIAGNOSTIC.json"
)
DEV_RESULT = (
    paper_real_eval.REPO_ROOT
    / "challenge/evaluation_v2/dev_results/YOLO26_G38_DEV.json"
)
DEV_CSV = (
    paper_real_eval.REPO_ROOT
    / "challenge/evaluation_v2/dev_results/YOLO26_G38_PER_FRAME.csv"
)
FORMAL_SELECTOR = (
    paper_real_eval.REPO_ROOT
    / "challenge/evaluation_v2/selector_diagnostic/PLASTIC_SELECTOR_DIAGNOSTIC.json"
)
FORMAL_SELECTOR_CSV = FORMAL_SELECTOR.with_name("PLASTIC_SELECTOR_PER_FRAME.csv")
WOOD_SELECTOR_STATUS = FORMAL_SELECTOR.with_name("WOOD_SELECTOR_STATUS.json")
MULTISHAPE_CONTRACT = (
    paper_real_eval.REPO_ROOT
    / "challenge/evaluation_v2/dev_results/YOLO26_G38_MULTISHAPE_DEV_CONTRACT_V3.json"
)


def _write_wood_selector_v2(path: Path) -> Path:
    population = load_repo_population(PopulationId.DEV_WOOD_POS45)
    records = [
        {
            "frame_id": item.frame_id,
            "domain": item.domain,
            "session": item.session_id or item.source_set,
            "selector_status": SelectorStatus.SELECTED.value,
            "selected_hypothesis": "short-face-front",
            "expected_hypothesis": "short-face-front",
            "correct": True,
            "restricted_adds_normalized": float(index),
            "rotation_error_deg": float(index),
            "translation_error_m": float(index),
            "yaw_error_deg": float(index),
        }
        for index, item in enumerate(population.items)
    ]
    tail_n = max(1, math.ceil(len(records) * 0.10))
    metrics = {}
    for field in (
        "restricted_adds_normalized",
        "rotation_error_deg",
        "translation_error_m",
        "yaw_error_deg",
    ):
        worst = sorted(
            records,
            key=lambda row: (float(row[field]), row["frame_id"]),
            reverse=True,
        )[:tail_n]
        metrics[field] = {
            "tail_n": tail_n,
            "selector_failure_count": 0,
            "selector_failure_fraction": 0.0,
            "dominated": False,
            "frame_ids": [row["frame_id"] for row in worst],
        }
    payload = {
        "schema_version": paper_real_eval.SELECTOR_DIAGNOSTIC_SCHEMA_V2,
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "object_type": WOOD_OBJECT_TYPE,
        "population_role": population.role.value,
        "population": {
            "population_id": population.population_id.value,
            "count": population.count,
            "membership_sha256": population.membership_sha256,
            "manifest": paper_real_eval._display_path(population.source_path),
            "manifest_file_sha256": paper_real_eval._sha256_file(
                population.source_path
            ),
        },
        "checkpoint": {"sha256": "0" * 64},
        "tail_dominance": {
            "assessed": True,
            "passed": True,
            "metrics": metrics,
        },
        "tail_dominance_assessed": True,
        "tail_dominance_passed": True,
        "tail_dominance_notes": "synthetic preregistered wood contract",
        "records": records,
    }
    path.write_text(json.dumps(payload), "utf-8")
    return path


def test_checked_in_selector_is_recomputed_and_remains_blocked() -> None:
    report, payload = paper_real_eval._selector_status(str(SELECTOR))
    assert payload is not None
    assert report.status.value == "FAIL"
    assert report.sample_count == 140
    assert report.night_count == 28
    assert report.overall_accuracy == pytest.approx(83 / 140)
    assert report.night_accuracy == pytest.approx(13 / 28)
    assert report.minimum_session_accuracy == pytest.approx(1 / 3)
    assert report.tail_dominance_assessed is True
    assert report.tail_dominance_passed is False


def test_formal_object_specific_selector_artifacts_are_fail_closed() -> None:
    report, payload = paper_real_eval._selector_status(str(FORMAL_SELECTOR))
    assert payload is not None
    assert report.status.value == "FAIL"
    assert report.sample_count == 140
    assert report.overall_accuracy == pytest.approx(83 / 140)
    assert report.night_accuracy == pytest.approx(13 / 28)
    with FORMAL_SELECTOR_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 140
    assert sum(row["correct"] == "True" for row in rows) == 83
    wood = json.loads(WOOD_SELECTOR_STATUS.read_text("utf-8"))
    assert wood["object_type"] == "wood_small_80x59x14"
    assert wood["status"] == "NOT_RUN"
    assert wood["accuracy"] is None


def test_legacy_plastic_selector_cannot_be_rebound_to_wood() -> None:
    with pytest.raises(ContractError, match="V1_IS_PLASTIC_DEV140_ONLY"):
        paper_real_eval._selector_status(
            str(FORMAL_SELECTOR), expected_object_type=WOOD_OBJECT_TYPE
        )


def test_wood_only_cli_rejects_legacy_plastic_selector_scalar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ContractError, match="LEGACY_SCALAR_IS_PLASTIC_ONLY"):
        paper_real_eval.main(
            [
                "--positive-manifest",
                str(manifest_path(PopulationId.DEV_WOOD_POS45)),
                "--negative-manifest",
                str(manifest_path(PopulationId.DEV_NEG2689)),
                "--population-role",
                "CROSS_SHAPE_DEV",
                "--weights",
                "not-loaded.pt",
                "--selector-diagnostic",
                str(FORMAL_SELECTOR),
                "--out",
                str(output),
                "--dry-run",
            ]
        )
    assert not output.exists()


def test_future_wood_selector_is_bound_to_exact_population_and_can_pass(
    tmp_path: Path,
) -> None:
    artifact = _write_wood_selector_v2(tmp_path / "wood-selector-v2.json")
    report, _ = paper_real_eval._selector_status(
        str(artifact), expected_object_type=WOOD_OBJECT_TYPE
    )
    assert report.status.value == "PASS"
    assert report.object_type == WOOD_OBJECT_TYPE
    assert report.population_id == PopulationId.DEV_WOOD_POS45.value
    assert report.population_role == "CROSS_SHAPE_DEV"
    assert report.population_validated is True
    assert report.sample_count == report.expected_sample_count == 45
    assert report.night_count == report.expected_night_count == 0
    assert report.night_accuracy is None
    assert dict(report.expected_session_counts) == {
        "wood_183705": 25,
        "wood_184309": 20,
    }
    gate = build_pose_metric_gate(
        canonical_migration_status="PASS",
        selector_report=report,
        symmetry_status="FROZEN",
        final_manifest_frozen=True,
    )
    assert gate.passed is True


def test_wood_selector_session_or_manifest_binding_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    artifact = _write_wood_selector_v2(tmp_path / "wood-selector-v2.json")
    payload = json.loads(artifact.read_text("utf-8"))
    payload["records"][0]["session"] = "wrong-session"
    artifact.write_text(json.dumps(payload), "utf-8")
    report, _ = paper_real_eval._selector_status(
        str(artifact), expected_object_type=WOOD_OBJECT_TYPE
    )
    assert report.status.value == "FAIL"
    assert report.population_validated is False
    assert report.blocked_reason == "SELECTOR_DIAGNOSTIC_POPULATION_MISMATCH"

    payload["population"]["membership_sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ContractError, match="MEMBERSHIP_SHA256_MISMATCH"):
        paper_real_eval._selector_status(
            str(artifact), expected_object_type=WOOD_OBJECT_TYPE
        )


def test_multishape_contract_artifact_dispatches_173_and_nulls_all_pose() -> None:
    report = json.loads(MULTISHAPE_CONTRACT.read_text("utf-8"))
    assert report["evaluation_mode"] == "DRY_RUN"
    assert report["population_contract"]["positive"]["population_id"] == (
        "COMMON_DEV_MULTISHAPE_POS"
    )
    assert report["population_contract"]["positive"]["count"] == 173
    assert report["population_contract"]["positive"]["object_types"] == [
        PLASTIC_OBJECT_TYPE,
        WOOD_OBJECT_TYPE,
    ]
    metadata = report["metrics_metadata"]
    assert metadata["object_subset"] == "ALL"
    assert metadata["object_type_counts"] == {
        "plastic_standard_110x130x11": 128,
        "wood_small_80x59x14": 45,
    }
    assert metadata["geometry_registry_sha256"] == (
        "0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627"
    )
    pose = report["metrics"]["pose"]
    assert pose["status"] == "BLOCKED"
    assert set(pose["subgroups"]) == {"ALL", "PLASTIC", "WOOD"}
    assert all(pose[field] is None for field in paper_real_eval.POSE_METRIC_FIELDS)
    assert pose["subgroups"]["PLASTIC"]["status"] == "BLOCKED"
    assert pose["subgroups"]["WOOD"]["status"] == "BLOCKED"
    object_contracts = report["gate_evidence"]["object_pose_contracts"]
    assert object_contracts[PLASTIC_OBJECT_TYPE]["selector_gate_report"][
        "expected_population"
    ]["population_id"] == "DEV_POS140"
    assert object_contracts[WOOD_OBJECT_TYPE]["selector_gate_report"][
        "expected_population"
    ]["population_id"] == "DEV_WOOD_POS45"


def test_selector_consumer_rejects_forged_correct_field(tmp_path: Path) -> None:
    payload = json.loads(SELECTOR.read_text("utf-8"))
    payload["records"][0]["correct"] = not payload["records"][0]["correct"]
    forged = tmp_path / "forged-selector.json"
    forged.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ContractError, match="CORRECT_0_MISMATCH"):
        paper_real_eval._selector_status(str(forged))


def test_actual_dev_artifacts_have_exact_population_checkpoint_and_null_pose() -> None:
    report = json.loads(
        DEV_RESULT.read_text("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    weights = (
        paper_real_eval.REPO_ROOT
        / report["weights"]["resolved_path"]
    )
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == report["weights"]["sha256"]
    assert report["evaluation_mode"] == "INFERENCE"
    assert report["metrics_metadata"]["population_ids"] == {
        "positive": "COMMON_DEV_POS128",
        "negative": "DEV_NEG2689",
    }
    assert report["metrics_metadata"]["N"] == 2817
    box_ap = report["metrics"]["box_and_keypoint_2d"]["box_ap50_95"]
    assert math.isfinite(box_ap) and 0.0 <= box_ap <= 1.0
    assert report["metrics"]["box_and_keypoint_2d"]["keypoint_nme"] is None
    pose = report["metrics"]["pose"]
    assert pose["status"] == "BLOCKED"
    for field in (
        "add_or_adds_auc",
        "rotation_median_deg",
        "translation_median_m",
        "yaw_median_deg",
    ):
        assert pose[field] is None
    with DEV_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2817
    assert sum(row["kind"] == "POSITIVE" for row in rows) == 128
    assert sum(row["kind"] == "NEGATIVE" for row in rows) == 2689
