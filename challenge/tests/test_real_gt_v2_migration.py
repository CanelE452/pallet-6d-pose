"""Migration gates for immutable legacy real-pallet labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.annotate import migrate_real_gt_v2 as migration
from scripts.annotate import pallet_geometry as geometry
from scripts.annotate.real_gt_v2_schema import validate_gt_v2


def _project(points: np.ndarray, transform: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    return np.column_stack([
        K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2],
        K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2],
    ])


def legacy_document(*, long_width: bool = False) -> dict:
    assignment = (
        geometry.AxisAssignment.YAW_90
        if long_width else geometry.AxisAssignment.YAW_0)
    dimensions = geometry.camera_facing_dimensions(assignment)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = [0.15, -0.05, 3.2]
    K = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 605.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    pixels = _project(
        geometry.camera_facing_keypoints_3d(assignment), transform, K)
    manual = pixels.tolist()
    # A missing historical manual click is preserved exactly; the effective
    # projected coordinate remains available with visibility=unknown.
    manual[4] = None
    return {
        "camera_data": {
            "width": 640,
            "height": 480,
            "intrinsics": {
                "fx": K[0, 0], "fy": K[1, 1],
                "cx": K[0, 2], "cy": K[1, 2],
            },
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "visibility": 1,
            "pose_transform": transform.tolist(),
            "projected_cuboid": pixels[:8].tolist(),
            "projected_cuboid_centroid": pixels[8].tolist(),
            "dimensions_m": dimensions.as_dict(),
            "gt_source": "manual",
            "split": "eval",
            "manual_kps": manual,
            "reproj_error_px": 0.5,
        }],
    }


def _write_symmetry_contract(repo: Path, *, valid: bool = True) -> Path:
    path = repo / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = geometry.canonical_to_camera_facing_transform("YAW_0")
    yaw180 = geometry.canonical_to_camera_facing_transform("YAW_180")
    payload = {
        "schema_version": "real_pallet_symmetry_contract_v2",
        "status": "FROZEN",
        "metric_variant": "ADD-S" if valid else "ADD",
        "canonical_axis": "+Y",
        "equivalent_yaw_degrees": [0, 180],
        "accepted_proper_rotations": [
            identity.tolist(), yaw180.tolist()],
        "equivalence_basis": {
            "kind": "DECLARED_BENCHMARK_ASSUMPTION",
            "statement": "Yaw 0 and 180 are the same benchmark pose.",
            "physical_inspection_claimed": False,
            "claim_boundary": "This is an evaluation convention, not inspection evidence.",
        },
        "reviewer_identity": "migration-test-reviewer",
        "review_date": "2020-01-01",
        "inclusion_exclusion_rules": ["Unladen pallet benchmark frames only."],
        "fixed_without_dev_or_final_pose_results": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("long_width", "expected_assignments"),
    [
        (False, ["YAW_0", "YAW_180"]),
        (True, ["YAW_90", "YAW_270"]),
    ],
)
def test_document_migration_preserves_legacy_and_keeps_signed_pose_unresolved(
        long_width: bool, expected_assignments: list[str]) -> None:
    legacy = legacy_document(long_width=long_width)
    legacy_snapshot = json.loads(json.dumps(legacy))
    state = migration.FileState(
        sha256="a" * 64, mtime_ns=123, size_bytes=456)
    migrated, diagnostics = migration.migrate_legacy_document(
        legacy,
        source_label="challenge/data/01_real/example/frame.json",
        source_state=state,
    )

    assert legacy == legacy_snapshot
    original_obj = legacy["objects"][0]
    obj = migrated["objects"][0]
    assert migrated["schema_version"] == "real_pallet_gt_v2"
    assert obj["physical_dimensions_m"] == {"x": 1.1, "y": 0.11, "z": 1.3}
    assert obj["canonical_pose"] is None
    assert obj["camera_facing_pnp"]["axis_assignment"] is None
    assert obj["camera_facing_pnp"]["axis_assignment_candidates"] == expected_assignments
    assert [item["axis_assignment"] for item in obj["canonical_pose_candidates"]] == (
        expected_assignments)
    assert obj["migration_status"] == "MANUAL_REVIEW_REQUIRED"
    assert obj["manual_kps"] == original_obj["manual_kps"]
    assert obj["dimensions_m"] == original_obj["dimensions_m"]
    assert obj["pose_transform"] == original_obj["pose_transform"]
    assert obj["legacy"] == {
        "dimensions_m": original_obj["dimensions_m"],
        "pose_transform": original_obj["pose_transform"],
        "fix_swap": None,
    }
    assert all(item["visibility"] == 0 for item in obj["keypoint_annotations"])
    assert all(item["source"] == "unknown" for item in obj["keypoint_annotations"])
    assert all(item["reason"] == "unknown" for item in obj["keypoint_annotations"])
    assert diagnostics.projection_parity_max_px <= 1e-4
    assert diagnostics.rotation_orthogonality_max_error <= 1e-6
    assert diagnostics.rotation_det_max_abs_error <= 1e-6
    assert diagnostics.reflection_count == 0
    assert diagnostics.manual_kps_preserved
    assert diagnostics.legacy_fields_preserved
    assert diagnostics.yaw180_equivalence_class_exact
    validate_gt_v2(migrated)


def test_valid_extrapolated_mask_transfers_source_and_manual_coordinates() -> None:
    legacy = legacy_document()
    obj = legacy["objects"][0]
    projected_zero = list(obj["projected_cuboid"][0])
    projected_one = list(obj["projected_cuboid"][1])
    obj["manual_kps"][0] = [101.25, 202.5]
    obj["manual_kps"][1] = [303.75, 404.0]
    obj["extrapolated_mask"] = [False, True] + [False] * 7

    migrated, _diagnostics = migration.migrate_legacy_document(
        legacy,
        source_label="challenge/data/01_real/example/frame.json",
        source_state=migration.FileState(
            sha256="a" * 64, mtime_ns=1, size_bytes=1),
    )
    annotations = migrated["objects"][0]["keypoint_annotations"]

    assert annotations[0]["xy"] == [101.25, 202.5]
    assert annotations[0]["xy"] != projected_zero
    assert annotations[0]["source"] == "manual_click"
    assert annotations[1]["xy"] == [303.75, 404.0]
    assert annotations[1]["xy"] != projected_one
    assert annotations[1]["source"] == "extrapolated"
    # A missing manual coordinate retains the stored projected fallback but
    # does not invent point provenance from the mask alone.
    assert annotations[4]["xy"] == obj["projected_cuboid"][4]
    assert annotations[4]["source"] == "unknown"
    assert all(entry["visibility"] == 0 for entry in annotations)
    assert all(entry["reason"] == "unknown" for entry in annotations)
    assert migrated["objects"][0]["occlusion_level"] == "unknown"


def test_absent_extrapolated_mask_keeps_legacy_projection_unknown() -> None:
    legacy = legacy_document()
    obj = legacy["objects"][0]
    projected_zero = list(obj["projected_cuboid"][0])
    obj["manual_kps"][0] = [101.25, 202.5]
    assert "extrapolated_mask" not in obj

    migrated, _diagnostics = migration.migrate_legacy_document(
        legacy,
        source_label="challenge/data/01_real/example/frame.json",
        source_state=migration.FileState(
            sha256="a" * 64, mtime_ns=1, size_bytes=1),
    )
    annotation = migrated["objects"][0]["keypoint_annotations"][0]

    assert annotation["xy"] == projected_zero
    assert annotation["source"] == "unknown"
    assert annotation["visibility"] == 0
    assert annotation["reason"] == "unknown"


def _write_fixture_repo(
        tmp_path: Path,
        *,
        wrong_sha: bool = False,
) -> tuple[Path, Path, Path, bytes, int]:
    repo = tmp_path / "repo"
    source = (
        repo / "challenge/data/01_real/eval_canonical/demo_session/frame.json")
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(legacy_document()), encoding="utf-8")
    original_bytes = source.read_bytes()
    original_mtime = source.stat().st_mtime_ns
    state = migration.file_state(source)
    audit = repo / "challenge/real_gt_v2/audit/LEGACY_GT_PER_FRAME.csv"
    audit.parent.mkdir(parents=True)
    fields = [
        "frame_id", "source_set", "label_path", "label_sha256",
        "label_mtime_ns", "label_size_bytes",
    ]
    with audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "frame_id": "frame",
            "source_set": "demo",
            "label_path": str(source.relative_to(repo)),
            "label_sha256": ("0" * 64 if wrong_sha else state.sha256),
            "label_mtime_ns": state.mtime_ns,
            "label_size_bytes": state.size_bytes,
        })
    return repo, audit, source, original_bytes, original_mtime


def test_audit_migration_preserves_session_structure_and_blocks_all_unresolved(
        tmp_path: Path) -> None:
    repo, audit, source, original_bytes, original_mtime = _write_fixture_repo(
        tmp_path)
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )

    output = output_root / "eval_canonical/demo_session/frame.json"
    assert output.is_file()
    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "UNCONFIRMED_SIGNED_CANONICAL_AXIS"
    assert gate["geometry_candidate_checks_pass"] is True
    assert gate["source_count"] == gate["migrated_count"] == 1
    assert gate["manual_review_required_count"] == 1
    assert gate["visibility_review_required_count"] == 1
    assert gate["canonical_pose_resolved_count"] == 0
    assert gate["reflection_transform_count"] == 0
    assert gate["source_audit_csv"] == (
        "challenge/real_gt_v2/audit/LEGACY_GT_PER_FRAME.csv")
    assert all(gate["checks"].values())

    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime_ns == original_mtime
    migrated = json.loads(output.read_text(encoding="utf-8"))
    assert migrated["objects"][0]["canonical_pose"] is None
    validate_gt_v2(migrated)

    report_rows = list(csv.DictReader(
        (report_root / "MIGRATION_REPORT.csv").open(encoding="utf-8")))
    axis_rows = list(csv.DictReader(
        (report_root / "MANUAL_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    visibility_rows = list(csv.DictReader(
        (report_root / "VISIBILITY_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    assert len(report_rows) == len(axis_rows) == len(visibility_rows) == 1
    assert report_rows[0]["source_label"] == (
        "challenge/data/01_real/eval_canonical/demo_session/frame.json")
    assert report_rows[0]["output_label"] == (
        "challenge/real_gt_v2/migrated_gt/"
        "eval_canonical/demo_session/frame.json")
    assert report_rows[0]["output_action"] == "CREATED"
    assert report_rows[0]["source_untouched"] == "True"
    assert json.loads(axis_rows[0]["axis_assignment_candidates"]) == [
        "YAW_0", "YAW_180"]
    assert json.loads(visibility_rows[0]["unknown_keypoint_indices"]) == list(range(9))
    assert visibility_rows[0]["unknown_keypoint_count"] == "9"


def test_rerun_skips_byte_identical_output_without_rewriting(
        tmp_path: Path) -> None:
    repo, audit, _source, _original_bytes, _original_mtime = (
        _write_fixture_repo(tmp_path))
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    first_gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )
    assert first_gate["geometry_candidate_checks_pass"] is True
    output = output_root / "eval_canonical/demo_session/frame.json"
    before = migration.file_state(output)

    second_gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )

    assert second_gate["geometry_candidate_checks_pass"] is True
    assert second_gate["blocked_reason"] == (
        "UNCONFIRMED_SIGNED_CANONICAL_AXIS")
    assert second_gate["existing_output_identical_skip_count"] == 1
    assert migration.file_state(output) == before
    report_rows = list(csv.DictReader(
        (report_root / "MIGRATION_REPORT.csv").open(encoding="utf-8")))
    assert report_rows[0]["output_action"] == "SKIPPED_IDENTICAL"


def test_frozen_yaw180_contract_promotes_equivalence_without_rewriting_gt(
        tmp_path: Path) -> None:
    repo, audit, source, original_bytes, original_mtime = _write_fixture_repo(
        tmp_path)
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"

    mechanical_gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )
    assert mechanical_gate["status"] == "BLOCKED"
    output = output_root / "eval_canonical/demo_session/frame.json"
    before_promotion = migration.file_state(output)
    before_bytes = output.read_bytes()
    contract = _write_symmetry_contract(repo)

    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
        symmetry_contract=contract,
    )

    assert gate["schema_version"] == migration.MIGRATION_GATE_SCHEMA_VERSION
    assert gate["status"] == "PASS"
    assert gate["blocked_reason"] is None
    assert gate["pose_resolution_mode"] == "YAW_180_EQUIVALENCE_CLASS"
    assert gate["canonical_pose_resolved_count"] == 0
    assert gate["canonical_pose_equivalence_resolved_count"] == 1
    assert gate["manual_review_required_count"] == 0
    assert gate["visibility_review_required_count"] == 1
    assert gate["symmetry_contract_path"] == (
        "challenge/real_gt_v2/SYMMETRY_CONTRACT.json")
    assert gate["symmetry_contract_sha256"] == migration.sha256_file(contract)
    assert gate["checks"]["yaw180_equivalence_class_exact"] is True
    assert all(gate["checks"].values())

    assert migration.file_state(output) == before_promotion
    assert output.read_bytes() == before_bytes
    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime_ns == original_mtime
    promoted = json.loads(output.read_text(encoding="utf-8"))
    assert promoted["objects"][0]["canonical_pose"] is None
    assert promoted["objects"][0]["camera_facing_pnp"]["axis_assignment"] is None

    report_rows = list(csv.DictReader(
        (report_root / "MIGRATION_REPORT.csv").open(encoding="utf-8")))
    manual_rows = list(csv.DictReader(
        (report_root / "MANUAL_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    visibility_rows = list(csv.DictReader(
        (report_root / "VISIBILITY_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    assert len(report_rows) == len(visibility_rows) == 1
    assert manual_rows == []
    assert report_rows[0]["status"] == (
        "CANONICAL_POSE_EQUIVALENCE_RESOLVED")
    assert report_rows[0]["output_action"] == "SKIPPED_IDENTICAL"
    assert report_rows[0]["yaw180_equivalence_class_exact"] == "True"
    assert report_rows[0]["source_size_bytes_before"] == str(len(original_bytes))
    assert report_rows[0]["source_size_bytes_after"] == str(len(original_bytes))


@pytest.mark.parametrize("contract_kind", ["invalid", "missing"])
def test_invalid_or_missing_symmetry_contract_fails_closed_before_writes(
        tmp_path: Path, contract_kind: str) -> None:
    repo, audit, source, original_bytes, original_mtime = _write_fixture_repo(
        tmp_path)
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    contract = (
        _write_symmetry_contract(repo, valid=False)
        if contract_kind == "invalid"
        else repo / "challenge/real_gt_v2/MISSING_SYMMETRY_CONTRACT.json"
    )

    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
        symmetry_contract=contract,
    )

    assert gate["schema_version"] == migration.MIGRATION_GATE_SCHEMA_VERSION
    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "SYMMETRY_CONTRACT_INVALID"
    assert gate["canonical_pose_resolved_count"] == 0
    assert gate["canonical_pose_equivalence_resolved_count"] == 0
    assert any("SYMMETRY_CONTRACT_INVALID" in error
               for error in gate["preflight_errors"])
    assert not list(output_root.rglob("*.json"))
    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime_ns == original_mtime
    manual_rows = list(csv.DictReader(
        (report_root / "MANUAL_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    visibility_rows = list(csv.DictReader(
        (report_root / "VISIBILITY_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    assert len(manual_rows) == len(visibility_rows) == 1


def test_rerun_fails_closed_and_preserves_human_reviewed_output(
        tmp_path: Path) -> None:
    repo, audit, _source, _original_bytes, _original_mtime = (
        _write_fixture_repo(tmp_path))
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )
    output = output_root / "eval_canonical/demo_session/frame.json"
    reviewed = json.loads(output.read_text(encoding="utf-8"))
    reviewed["objects"][0]["keypoint_annotations"][0].update({
        "visibility": 2,
        "source": "manual_click",
        "reason": "visible",
    })
    output.write_text(json.dumps(reviewed), encoding="utf-8")
    reviewed_state = migration.file_state(output)

    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )

    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "EXISTING_OUTPUT_PROTECTED"
    assert any("may contain human review" in error
               for error in gate["preflight_errors"])
    assert migration.file_state(output) == reviewed_state


def test_existing_output_error_policy_rejects_even_identical_file(
        tmp_path: Path) -> None:
    repo, audit, _source, _original_bytes, _original_mtime = (
        _write_fixture_repo(tmp_path))
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )
    output = output_root / "eval_canonical/demo_session/frame.json"
    before = migration.file_state(output)

    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
        existing_output_policy="error",
    )

    assert gate["blocked_reason"] == "EXISTING_OUTPUT_PROTECTED"
    assert migration.file_state(output) == before


def test_source_sha_mtime_baseline_mismatch_fails_before_writing_outputs(
        tmp_path: Path) -> None:
    repo, audit, source, original_bytes, original_mtime = _write_fixture_repo(
        tmp_path, wrong_sha=True)
    output_root = repo / "challenge/real_gt_v2/migrated_gt"
    report_root = repo / "challenge/real_gt_v2"
    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=output_root,
        report_root=report_root,
        repo_root=repo,
        expected_count=1,
    )
    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "SOURCE_BASELINE_MISMATCH"
    assert gate["migrated_count"] == 0
    assert gate["manual_review_required_count"] == 1
    assert gate["visibility_review_required_count"] == 1
    assert not list(output_root.rglob("*.json"))
    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime_ns == original_mtime
    visibility_rows = list(csv.DictReader(
        (report_root / "VISIBILITY_REVIEW_QUEUE.csv").open(encoding="utf-8")))
    assert len(visibility_rows) == 1


def test_output_root_cannot_alias_the_legacy_source_tree(tmp_path: Path) -> None:
    repo, audit, source, original_bytes, original_mtime = _write_fixture_repo(
        tmp_path)
    source_root = repo / migration.SOURCE_DATA_ROOT
    gate = migration.migrate_from_audit(
        audit_csv=audit,
        output_root=source_root,
        report_root=repo / "challenge/real_gt_v2",
        repo_root=repo,
        expected_count=1,
    )
    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "SOURCE_BASELINE_MISMATCH"
    assert any("refusing an in-place migration" in error
               for error in gate["preflight_errors"])
    assert source.read_bytes() == original_bytes
    assert source.stat().st_mtime_ns == original_mtime


def test_reflected_legacy_pose_is_never_promoted() -> None:
    legacy = legacy_document()
    legacy["objects"][0]["pose_transform"][0][0] = -1.0
    with pytest.raises(migration.MigrationError, match="reflection"):
        migration.migrate_legacy_document(
            legacy,
            source_label="challenge/data/01_real/example/frame.json",
            source_state=migration.FileState(
                sha256="a" * 64, mtime_ns=1, size_bytes=1),
        )


def test_real_phase_a_membership_passes_all_candidate_checks_dry_run(
        tmp_path: Path) -> None:
    gate = migration.migrate_from_audit(
        audit_csv=migration.DEFAULT_AUDIT_CSV,
        output_root=tmp_path / "migrated_gt",
        report_root=tmp_path / "reports",
        repo_root=migration.REPO_ROOT,
        expected_count=140,
        dry_run=True,
    )
    assert gate["status"] == "BLOCKED"
    assert gate["blocked_reason"] == "UNCONFIRMED_SIGNED_CANONICAL_AXIS"
    assert gate["geometry_candidate_checks_pass"] is True
    assert gate["source_count"] == gate["migrated_count"] == 140
    assert gate["manual_review_required_count"] == 140
    assert gate["visibility_review_required_count"] == 140
    assert gate["canonical_pose_resolved_count"] == 0
    assert gate["reflection_transform_count"] == 0
    assert gate["maxima"]["projection_parity_max_px"] <= 1e-4
    assert not list((tmp_path / "migrated_gt").rglob("*.json"))


def test_default_output_namespace_is_non_destructive() -> None:
    expected = migration.REPO_ROOT / "challenge/real_gt_v2/migrated_gt"
    assert migration.DEFAULT_OUTPUT_ROOT == expected
    assert migration.SOURCE_DATA_ROOT not in migration.DEFAULT_OUTPUT_ROOT.parents


def test_dataset_facing_canonical_path_is_one_relative_symlink() -> None:
    dataset_path = (
        migration.REPO_ROOT / "challenge/data/01_real/gt_v2_canonical")
    assert dataset_path.is_symlink()
    assert dataset_path.readlink() == Path("../../real_gt_v2/migrated_gt")
    assert dataset_path.resolve() == migration.DEFAULT_OUTPUT_ROOT.resolve()
