"""Population integrity and blocked paper-evaluation output."""

from __future__ import annotations

import argparse
import builtins
import copy
import csv
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from challenge.evaluation_v2 import paper_real_eval, real_dataset_contract as dataset_contract
from challenge.evaluation_v2.pnp_selector import (
    CanonicalPoseCandidate,
    PnPSelectionResult,
    SelectorGateReport,
    SelectorGateState,
    SelectorStatus,
)
from challenge.evaluation_v2.pose_metrics import (
    POSE_METRIC_FIELDS,
    build_pose_metric_gate,
    summarize_pose_errors,
)
from challenge.evaluation_v2.real_dataset_contract import (
    ContractError,
    ManifestItem,
    MembershipStatus,
    PopulationId,
    PopulationRole,
    load_population_manifest,
    load_repo_population,
    manifest_path,
    membership_sha256,
    validate_common_dev_membership,
    validate_evaluation_pair,
    validate_registered_membership,
    validate_repo_population_contract,
)
from scripts.annotate import pallet_geometry as geometry


def _manifest_json(population_id: PopulationId) -> dict:
    return json.loads(manifest_path(population_id).read_text("utf-8"))


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return path


def _rehash(payload: dict) -> None:
    items = tuple(
        ManifestItem(
            frame_id=item["frame_id"],
            image=item["image"],
            label=item.get("label"),
            source_set=item.get("source_set"),
            domain=item.get("domain"),
        )
        for item in payload["items"]
    )
    payload["membership_sha256"] = membership_sha256(items)


def _dry_run_args(out: Path, positive: PopulationId = PopulationId.COMMON_DEV_POS128) -> list[str]:
    return [
        "--positive-manifest",
        str(manifest_path(positive)),
        "--negative-manifest",
        str(manifest_path(PopulationId.DEV_NEG2689)),
        "--population-role",
        "DEV",
        "--weights",
        "intentionally-not-loaded.pt",
        "--out",
        str(out),
        "--dry-run",
    ]


def test_repo_population_counts_and_availability_are_exact() -> None:
    manifests = validate_repo_population_contract(validate_files=True)
    assert manifests[PopulationId.DEV_POS140].count == 140
    assert manifests[PopulationId.COMMON_DEV_POS128].count == 128
    assert manifests[PopulationId.DEV_NEG2689].count == 2689
    for population_id in (PopulationId.FINAL_POS, PopulationId.FINAL_NEG):
        manifest = manifests[population_id]
        assert manifest.count == 0
        assert manifest.expected_count == 0
        assert manifest.membership_status is MembershipStatus.UNAVAILABLE
        assert manifest.available is False
        assert manifest.frozen is False
        assert manifest.membership_sha256 is None
        assert "membership unavailable" in manifest.provenance["count_zero_semantics"]


def test_invalid_gt_quarantine_is_exact_and_disjoint_from_active_positive_gt() -> None:
    registry = json.loads(dataset_contract.INVALID_GT_QUARANTINE_PATH.read_text("utf-8"))
    entries = registry["entries"]
    assert registry["entry_count"] == len(entries) == 23
    assert registry["official_eval_exclusion_count"] == 21
    assert registry["stale_duplicate_count"] == 2

    forbidden_hashes = {entry["source_sha256"] for entry in entries}
    official_ids = {
        entry["frame_id"]
        for entry in entries
        if entry["classification"] != "STALE_DUPLICATE_INVALID"
    }
    stale_ids = {
        entry["frame_id"]
        for entry in entries
        if entry["classification"] == "STALE_DUPLICATE_INVALID"
    }
    assert len(forbidden_hashes) == 23
    assert len(official_ids) == 21
    assert stale_ids == {"1778653345465966336", "1778653498432396288"}

    for population_id in (PopulationId.DEV_POS140, PopulationId.COMMON_DEV_POS128):
        manifest = load_repo_population(population_id)
        assert official_ids.isdisjoint(manifest.frame_ids)
        for item in manifest.items:
            label = json.loads((paper_real_eval.REPO_ROOT / item.label).read_text("utf-8"))
            source_sha = label["real_gt_v2_migration"]["source_sha256"]
            assert source_sha not in forbidden_hashes

    dev = load_repo_population(PopulationId.DEV_POS140)
    assert stale_ids <= set(dev.frame_ids), "the two corrected canonical copies must remain"


def test_migrated_quarantined_source_sha_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = load_repo_population(PopulationId.DEV_POS140)
    first = dev.items[0]
    source_sha = json.loads(
        (paper_real_eval.REPO_ROOT / first.label).read_text("utf-8")
    )["real_gt_v2_migration"]["source_sha256"]
    poisoned_registry = {
        "schema_version": dataset_contract.INVALID_GT_QUARANTINE_SCHEMA,
        "status": "QUARANTINED",
        "entry_count": 1,
        "official_eval_exclusion_count": 0,
        "stale_duplicate_count": 1,
        "entries": [
            {
                "frame_id": first.frame_id,
                "source_path": "challenge/data/01_real/manual_gt/poisoned.json",
                "source_sha256": source_sha,
                "classification": "STALE_DUPLICATE_INVALID",
            }
        ],
    }
    path = tmp_path / "poisoned_quarantine.json"
    path.write_text(json.dumps(poisoned_registry), "utf-8")
    monkeypatch.setattr(dataset_contract, "INVALID_GT_QUARANTINE_PATH", path)
    with pytest.raises(ContractError, match="QUARANTINED_GT_SOURCE"):
        load_repo_population(PopulationId.DEV_POS140)


def test_migrated_quarantined_source_path_fails_closed_after_reserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = load_repo_population(PopulationId.DEV_POS140)
    first = dev.items[0]
    source_label = json.loads(
        (paper_real_eval.REPO_ROOT / first.label).read_text("utf-8")
    )["real_gt_v2_migration"]["source_label"]
    poisoned_registry = {
        "schema_version": dataset_contract.INVALID_GT_QUARANTINE_SCHEMA,
        "status": "QUARANTINED",
        "entry_count": 1,
        "official_eval_exclusion_count": 0,
        "stale_duplicate_count": 1,
        "entries": [
            {
                "frame_id": first.frame_id,
                "source_path": source_label,
                "source_sha256": "0" * 64,
                "classification": "STALE_DUPLICATE_INVALID",
            }
        ],
    }
    path = tmp_path / "poisoned_path_quarantine.json"
    path.write_text(json.dumps(poisoned_registry), "utf-8")
    monkeypatch.setattr(dataset_contract, "INVALID_GT_QUARANTINE_PATH", path)
    with pytest.raises(ContractError, match="QUARANTINED_GT_SOURCE_PATH"):
        load_repo_population(PopulationId.DEV_POS140)


def test_legacy_eval_artifacts_do_not_retain_quarantined_annotations() -> None:
    registry = json.loads(dataset_contract.INVALID_GT_QUARANTINE_PATH.read_text("utf-8"))
    forbidden_paths = {entry["source_path"] for entry in registry["entries"]}
    official_ids = {
        entry["frame_id"]
        for entry in registry["entries"]
        if entry["classification"] != "STALE_DUPLICATE_INVALID"
    }

    legacy_manifest_path = (
        paper_real_eval.REPO_ROOT
        / "challenge/yolo_pose_one_model/paper_generic_pipeline/eval_manifest.json"
    )
    if legacy_manifest_path.is_file():
        legacy = json.loads(legacy_manifest_path.read_text("utf-8"))
        assert legacy["manifest"] == "PAPER_YOLO_EVAL_DEV_140_GT_QA_CLEAN"
        assert legacy["n_total"] == legacy["checks"]["expected_total"] == 140
        assert legacy["MANIFEST_VALID"] is True
        assert {item["label"] for item in legacy["items"]}.isdisjoint(forbidden_paths)
        assert official_ids.isdisjoint(item["frame_id"] for item in legacy["items"])

    registry_csv = (
        paper_real_eval.REPO_ROOT
        / "challenge/yolo_pose_one_model/manifests/all_samples.csv"
    )
    rows = csv.DictReader(registry_csv.open(encoding="utf-8"))
    assert forbidden_paths.isdisjoint(row["annotation_path"] for row in rows)


def test_common_dev_is_exact_dev140_minus_recorded_leak12() -> None:
    dev = load_repo_population(PopulationId.DEV_POS140)
    common = load_repo_population(PopulationId.COMMON_DEV_POS128)
    validate_common_dev_membership(dev, common)
    excluded = set(common.provenance["excluded_frame_ids"])
    assert len(excluded) == 12
    assert common.frame_ids == tuple(item.frame_id for item in dev.items if item.frame_id not in excluded)
    assert sum(item.domain == "DAY" for item in common.items) == 100
    assert sum(item.domain == "NIGHT" for item in common.items) == 28


def test_negative_membership_is_explicit_and_sequential() -> None:
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    assert negative.frame_ids[0] == "000000"
    assert negative.frame_ids[-1] == "002688"
    assert negative.frame_ids == tuple(f"{index:06d}" for index in range(2689))
    assert all(item.image.endswith(f"/{item.frame_id}.png") for item in negative.items)


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.DEV_POS140)
    payload["membership_sha256"] = "0" * 64
    path = _write_manifest(tmp_path / "bad_hash.json", payload)
    with pytest.raises(ContractError, match="MEMBERSHIP_HASH_MISMATCH"):
        load_population_manifest(path)


def test_duplicate_frame_id_fails_before_hash_acceptance(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.DEV_POS140)
    payload["items"][1] = dict(payload["items"][0])
    _rehash(payload)
    path = _write_manifest(tmp_path / "duplicate.json", payload)
    with pytest.raises(ContractError, match="DUPLICATE_FRAME_IDS"):
        load_population_manifest(path)


def test_duplicate_image_with_distinct_frame_ids_fails_closed(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.DEV_POS140)
    payload["items"][1]["image"] = payload["items"][0]["image"]
    _rehash(payload)
    path = _write_manifest(tmp_path / "duplicate_image.json", payload)
    with pytest.raises(ContractError, match="DUPLICATE_IMAGE_PATHS"):
        load_population_manifest(path)


def test_missing_member_file_fails_instead_of_shortening_denominator(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.DEV_POS140)
    payload["items"][0]["image"] = "challenge/real_gt_v2/does-not-exist.png"
    _rehash(payload)
    path = _write_manifest(tmp_path / "missing.json", payload)
    with pytest.raises(ContractError, match="MISSING_DATA_FILE"):
        load_population_manifest(path)


def test_absolute_data_path_is_forbidden(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.DEV_POS140)
    payload["items"][0]["image"] = "/tmp/not-a-repo-member.png"
    _rehash(payload)
    path = _write_manifest(tmp_path / "absolute.json", payload)
    with pytest.raises(ContractError, match="ABSOLUTE_DATA_PATH_FORBIDDEN"):
        load_population_manifest(path, validate_files=False)


def test_recomputed_but_unregistered_membership_is_rejected(tmp_path: Path) -> None:
    payload = _manifest_json(PopulationId.COMMON_DEV_POS128)
    payload["items"][0]["source_set"] = "forged_set"
    _rehash(payload)
    manifest = load_population_manifest(
        _write_manifest(tmp_path / "forged.json", payload), validate_files=True
    )
    with pytest.raises(ContractError, match="UNREGISTERED_MEMBERSHIP"):
        validate_registered_membership(manifest)


def test_dev_comparison_accepts_only_common128_plus_neg2689() -> None:
    common = load_repo_population(PopulationId.COMMON_DEV_POS128)
    dev140 = load_repo_population(PopulationId.DEV_POS140)
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    pair = validate_evaluation_pair(common, negative, PopulationRole.DEV)
    assert pair.ready
    assert pair.pair_sha256 is not None
    with pytest.raises(ContractError, match="DEV_COMPARISON_REQUIRES_COMMON"):
        validate_evaluation_pair(dev140, negative, PopulationRole.DEV)


def test_final_zero_is_unavailable_not_a_valid_empty_evaluation() -> None:
    positive = load_repo_population(PopulationId.FINAL_POS)
    negative = load_repo_population(PopulationId.FINAL_NEG)
    with pytest.raises(ContractError, match="FINAL_MEMBERSHIP_UNAVAILABLE"):
        validate_evaluation_pair(positive, negative, PopulationRole.FINAL)
    pair = validate_evaluation_pair(
        positive,
        negative,
        PopulationRole.FINAL,
        allow_unavailable_final=True,
    )
    assert pair.ready is False
    assert pair.blocked_reason == "FINAL_MEMBERSHIP_UNAVAILABLE"


def test_positive_and_negative_frame_ids_must_be_disjoint() -> None:
    positive = load_repo_population(PopulationId.COMMON_DEV_POS128)
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    collided_item = replace(
        negative.items[0], frame_id=positive.items[0].frame_id
    )
    collided_negative = replace(negative, items=(collided_item,))
    with pytest.raises(ContractError, match="POSITIVE_NEGATIVE_FRAME_ID_OVERLAP"):
        validate_evaluation_pair(positive, collided_negative, PopulationRole.DEV)


def test_required_cli_arguments_are_all_required() -> None:
    parser = paper_real_eval.build_parser()
    actions = {action.dest: action for action in parser._actions}
    for name in (
        "positive_manifest",
        "negative_manifest",
        "population_role",
        "weights",
        "out",
    ):
        assert actions[name].required is True


def test_evaluator_source_has_no_machine_specific_external_manifest_path() -> None:
    source = inspect.getsource(paper_real_eval)
    assert "/home/minjae" not in source
    assert "pallet_worker_transfer" not in source


def test_migrated_visibility_zero_coordinates_are_not_2d_supervision() -> None:
    manifest = load_repo_population(PopulationId.COMMON_DEV_POS128)
    assert all(
        item.label is not None
        and item.label.startswith("challenge/real_gt_v2/migrated_gt/")
        for item in manifest.items
    )
    target = paper_real_eval._legacy_forbidden_target(manifest.items[0])
    assert target.keypoint_xy_present.all()
    assert (target.visibility == 0).all()
    assert not target.keypoint_supervision_mask.any()


def test_all_visibility_zero_keeps_keypoint_distribution_empty() -> None:
    positive = load_repo_population(PopulationId.COMMON_DEV_POS128)
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    item = positive.items[0]
    target = paper_real_eval._legacy_forbidden_target(item)
    one_frame_pair = replace(
        validate_evaluation_pair(positive, negative, PopulationRole.DEV),
        positive=replace(positive, items=(item,)),
        negative=replace(negative, items=()),
    )

    class PerfectLegacyCoordinatePrediction:
        def predict(self, _image_path: Path):
            return [(0.99, target.box_xyxy.copy(), target.keypoints_xy.copy())]

    metrics = paper_real_eval.evaluate_2d_with_predictor(
        one_frame_pair, PerfectLegacyCoordinatePrediction()
    )
    assert metrics["box_ap50"] == 1.0
    assert metrics["keypoint_all_labeled"] == {
        "count": 0,
        "median_px": None,
        "p90_px": None,
    }
    assert metrics["keypoint_visibility_1"]["count"] == 0
    assert metrics["keypoint_visibility_2"]["count"] == 0


def test_dry_run_imports_no_ultralytics_and_writes_strict_blocked_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("dry-run must not import Ultralytics")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    target = tmp_path / "report.json"
    assert paper_real_eval.main(_dry_run_args(target)) == 0
    report = json.loads(target.read_text("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert report["evaluation_mode"] == "DRY_RUN"
    assert report["inference"]["status"] == "NOT_RUN_DRY_RUN"
    assert report["population_contract"]["positive"]["count"] == 128
    assert report["population_contract"]["negative"]["count"] == 2689
    pose = report["metrics"]["pose"]
    assert pose["status"] == "BLOCKED"
    assert set(pose["blocked_reasons"]) == {
        "CANONICAL_MIGRATION_NOT_PASS",
        "POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR",
        "SYMMETRY_NOT_FROZEN",
        "FINAL_MANIFEST_NOT_FROZEN",
    }
    assert all(pose[field] is None for field in POSE_METRIC_FIELDS)
    assert report["metrics"]["box_and_keypoint_2d"]["box_ap50_95"] is None
    selector_gate = report["gate_evidence"]["selector_gate_report"]
    assert selector_gate["status"] == "NOT_RUN"
    assert selector_gate["tail_dominance_assessed"] is False
    assert selector_gate["tail_dominance_passed"] is None
    assert selector_gate["tail_dominance_notes"] is None


def test_dry_run_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    paper_real_eval.main(_dry_run_args(target))
    before = target.read_bytes()
    with pytest.raises(ContractError, match="OUTPUT_ALREADY_EXISTS"):
        paper_real_eval.main(_dry_run_args(target))
    assert target.read_bytes() == before


def test_evaluator_rejects_colliding_primary_and_sidecar_paths(tmp_path: Path) -> None:
    target = tmp_path / "collision.json"
    report = tmp_path / "report.md"
    with pytest.raises(ContractError, match="OUTPUT_PATHS_MUST_BE_DISTINCT"):
        paper_real_eval.main(
            _dry_run_args(target)
            + [
                "--per-frame-out",
                str(target),
                "--report-out",
                str(report),
            ]
        )
    assert not target.exists()
    assert not report.exists()


def test_evaluator_rejects_dev140_even_in_dry_run(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="DEV_COMPARISON_REQUIRES_COMMON"):
        paper_real_eval.main(
            _dry_run_args(tmp_path / "must-not-exist.json", PopulationId.DEV_POS140)
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_final_placeholders_can_emit_a_dry_run_blocked_report(tmp_path: Path) -> None:
    target = tmp_path / "final.json"
    args = [
        "--positive-manifest",
        str(manifest_path(PopulationId.FINAL_POS)),
        "--negative-manifest",
        str(manifest_path(PopulationId.FINAL_NEG)),
        "--population-role",
        "FINAL",
        "--weights",
        "not-loaded.pt",
        "--out",
        str(target),
        "--dry-run",
    ]
    paper_real_eval.main(args)
    report = json.loads(target.read_text("utf-8"))
    assert report["population_contract"]["ready"] is False
    assert report["population_contract"]["blocked_reason"] == "FINAL_MEMBERSHIP_UNAVAILABLE"
    assert all(report["metrics"]["pose"][field] is None for field in POSE_METRIC_FIELDS)


@pytest.mark.parametrize(
    ("population_id", "expected_subset", "expected_counts"),
    (
        (
            PopulationId.FINAL_WOOD_POS,
            "WOOD",
            {"wood_small_80x59x14": 0},
        ),
        (
            PopulationId.FINAL_ALL_POS,
            "ALL",
            {
                "plastic_standard_110x130x11": 0,
                "wood_small_80x59x14": 0,
            },
        ),
    ),
)
def test_empty_final_scope_comes_from_manifest_object_types(
    tmp_path: Path,
    population_id: PopulationId,
    expected_subset: str,
    expected_counts: dict[str, int],
) -> None:
    target = tmp_path / f"{population_id.value}.json"
    paper_real_eval.main(
        [
            "--positive-manifest",
            str(manifest_path(population_id)),
            "--negative-manifest",
            str(manifest_path(PopulationId.FINAL_NEG)),
            "--population-role",
            "FINAL",
            "--weights",
            "not-loaded.pt",
            "--out",
            str(target),
            "--dry-run",
        ]
    )
    report = json.loads(target.read_text("utf-8"))
    assert report["population_contract"]["positive"]["object_types"] == list(
        expected_counts
    )
    assert report["metrics_metadata"]["object_subset"] == expected_subset
    assert report["metrics_metadata"]["object_type_counts"] == expected_counts
    if population_id is PopulationId.FINAL_ALL_POS:
        assert set(report["metrics"]["pose"]["subgroups"]) == {
            "ALL",
            "PLASTIC",
            "WOOD",
        }


def test_blocked_pose_gate_does_not_even_iterate_metric_records() -> None:
    gate = build_pose_metric_gate(
        canonical_migration_status="NOT_RUN",
        selector_report=SelectorGateReport.not_run(),
        symmetry_status="NOT_FROZEN",
        final_manifest_frozen=False,
    )

    class Poison:
        def __iter__(self):
            raise AssertionError("blocked metric code must not inspect pose records")

    payload = summarize_pose_errors(Poison(), gate, metric_variant="ADD")
    assert payload["status"] == "BLOCKED"
    assert all(payload[field] is None for field in POSE_METRIC_FIELDS)


def test_forged_selector_pass_without_population_or_tail_evidence_stays_blocked() -> None:
    forged = replace(
        SelectorGateReport.not_run(),
        status=SelectorGateState.PASS,
        blocked_reason=None,
    )
    gate = build_pose_metric_gate(
        canonical_migration_status="PASS",
        selector_report=forged,
        symmetry_status="FROZEN",
        final_manifest_frozen=True,
    )
    assert gate.passed is False
    assert gate.blocked_reasons == (
        "POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR",
    )


def test_yaw180_symmetry_does_not_bypass_wd_parity_selector_gate() -> None:
    gate = build_pose_metric_gate(
        canonical_migration_status="PASS",
        selector_report=SelectorGateReport.not_run(),
        symmetry_status="FROZEN",
        final_manifest_frozen=True,
    )
    assert gate.passed is False
    assert gate.blocked_reasons == (
        "POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR",
    )


def _one_migrated_payload() -> dict:
    root = paper_real_eval.REPO_ROOT / "challenge" / "real_gt_v2" / "migrated_gt"
    path = next(root.rglob("*.json"))
    return json.loads(path.read_text("utf-8"))


def _write_temporary_gt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
) -> ManifestItem:
    label = tmp_path / "label.json"
    label.write_text(json.dumps(payload), "utf-8")
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)
    return ManifestItem(frame_id="synthetic", image="unused.png", label="label.json")


def test_paper_target_invokes_full_v2_validator_and_rejects_extra_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _one_migrated_payload()
    payload["objects"].append(copy.deepcopy(payload["objects"][0]))
    item = _write_temporary_gt(tmp_path, monkeypatch, payload)
    with pytest.raises(ContractError, match="EXACTLY_ONE_OBJECT"):
        paper_real_eval._legacy_forbidden_target(item)


@pytest.mark.parametrize("violation", ["dimensions", "reflection", "candidate_mismatch"])
def test_paper_target_rejects_geometry_schema_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    payload = _one_migrated_payload()
    obj = payload["objects"][0]
    if violation == "dimensions":
        obj["physical_dimensions_m"]["x"] = 9.9
    elif violation == "reflection":
        obj["canonical_pose_candidates"][0][
            "canonical_to_camera_facing_rotation"
        ] = np.diag([-1.0, 1.0, 1.0]).tolist()
    else:
        current = obj["camera_facing_pnp"]["axis_assignment_candidates"]
        obj["camera_facing_pnp"]["axis_assignment_candidates"] = (
            ["YAW_90", "YAW_270"]
            if current == ["YAW_0", "YAW_180"]
            else ["YAW_0", "YAW_180"]
        )
    item = _write_temporary_gt(tmp_path, monkeypatch, payload)
    with pytest.raises(ContractError, match="INVALID_GT_V2_SCHEMA"):
        paper_real_eval._legacy_forbidden_target(item)


def _valid_migration_pass(
    tmp_path: Path,
    *,
    resolution_mode: str = paper_real_eval.SIGNED_CANONICAL_POSE,
    symmetry_path: Path | None = None,
) -> dict:
    output_payload = _one_migrated_payload()
    output_object = output_payload["objects"][0]
    if resolution_mode == paper_real_eval.SIGNED_CANONICAL_POSE:
        selected_pose = copy.deepcopy(output_object["canonical_pose_candidates"][0])
        output_object["camera_facing_pnp"]["axis_assignment"] = selected_pose[
            "axis_assignment"
        ]
        output_object["canonical_pose"] = selected_pose
        output_object["migration_status"] = "CANONICAL_POSE_CONFIRMED"
    elif resolution_mode == paper_real_eval.YAW_180_EQUIVALENCE_CLASS:
        if symmetry_path is None:
            raise ValueError("equivalence migration fixture requires symmetry_path")
        output_object["camera_facing_pnp"]["axis_assignment"] = None
        output_object["canonical_pose"] = None
        output_object["migration_status"] = "CANONICAL_POSE_EQUIVALENCE_RESOLVED"
    else:
        raise ValueError(resolution_mode)

    sources = tmp_path / "sources"
    outputs = tmp_path / "outputs"
    sources.mkdir()
    outputs.mkdir()
    audit = tmp_path / "audit.csv"
    audit_fields = [
        "frame_id",
        "label_path",
        "label_sha256",
        "label_mtime_ns",
        "label_size_bytes",
    ]
    report_fields = [
        "frame_id",
        "source_label",
        "output_label",
        "status",
        "projection_parity_max_px",
        "rotation_orthogonality_max_error",
        "rotation_det_max_abs_error",
        "reflection_count",
        "manual_kps_preserved",
        "legacy_fields_preserved",
        "schema_valid",
        "source_sha_before",
        "source_sha_after",
        "source_mtime_ns_before",
        "source_mtime_ns_after",
        "source_size_bytes_before",
        "source_size_bytes_after",
        "source_untouched",
        "yaw180_equivalence_class_exact",
    ]
    audit_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    output_text = json.dumps(output_payload)
    for index in range(140):
        source_path = sources / f"{index}.json"
        output_path = outputs / f"{index}.json"
        source_path.write_text(json.dumps({"legacy_frame": index}), "utf-8")
        output_path.write_text(output_text, "utf-8")
        source_stat = source_path.stat()
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_relative = f"sources/{index}.json"
        output_relative = f"outputs/{index}.json"
        audit_rows.append(
            {
                "frame_id": str(index),
                "label_path": source_relative,
                "label_sha256": source_sha,
                "label_mtime_ns": str(source_stat.st_mtime_ns),
                "label_size_bytes": str(source_stat.st_size),
            }
        )
        report_rows.append(
            {
                "frame_id": str(index),
                "source_label": source_relative,
                "output_label": output_relative,
                "status": (
                    "CANONICAL_POSE_CONFIRMED"
                    if resolution_mode == paper_real_eval.SIGNED_CANONICAL_POSE
                    else "CANONICAL_POSE_EQUIVALENCE_RESOLVED"
                ),
                "projection_parity_max_px": "0.0",
                "rotation_orthogonality_max_error": "0.0",
                "rotation_det_max_abs_error": "0.0",
                "reflection_count": "0",
                "manual_kps_preserved": "True",
                "legacy_fields_preserved": "True",
                "schema_valid": "True",
                "source_sha_before": source_sha,
                "source_sha_after": source_sha,
                "source_mtime_ns_before": str(source_stat.st_mtime_ns),
                "source_mtime_ns_after": str(source_stat.st_mtime_ns),
                "source_size_bytes_before": str(source_stat.st_size),
                "source_size_bytes_after": str(source_stat.st_size),
                "source_untouched": "True",
                "yaw180_equivalence_class_exact": (
                    "True"
                    if resolution_mode
                    == paper_real_eval.YAW_180_EQUIVALENCE_CLASS
                    else ""
                ),
            }
        )
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    with (tmp_path / "MIGRATION_REPORT.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report_rows)
    (tmp_path / "MANUAL_REVIEW_QUEUE.csv").write_text("frame_id\n", "utf-8")
    checks = {name: True for name in paper_real_eval.MIGRATION_REQUIRED_CHECKS}
    if resolution_mode == paper_real_eval.YAW_180_EQUIVALENCE_CLASS:
        checks["yaw180_equivalence_class_exact"] = True
        checks["symmetry_contract_sha256_current"] = True
    return {
        "schema_version": paper_real_eval.MIGRATION_GATE_SCHEMA_VERSION,
        "status": "PASS",
        "blocked_reason": None,
        "dry_run": False,
        "source_audit_csv": str(audit),
        "source_count": 140,
        "migrated_count": 140,
        "output_json_count": 140,
        "manual_review_required_count": 0,
        "visibility_review_required_count": 0,
        "pose_resolution_mode": resolution_mode,
        "canonical_pose_resolved_count": (
            140 if resolution_mode == paper_real_eval.SIGNED_CANONICAL_POSE else 0
        ),
        "canonical_pose_equivalence_resolved_count": (
            140
            if resolution_mode == paper_real_eval.YAW_180_EQUIVALENCE_CLASS
            else 0
        ),
        "symmetry_contract_path": str(symmetry_path) if symmetry_path else None,
        "symmetry_contract_sha256": (
            hashlib.sha256(symmetry_path.read_bytes()).hexdigest()
            if symmetry_path
            else None
        ),
        "geometry_candidate_checks_pass": True,
        "checks": checks,
        "maxima": {
            "rotation_orthogonality_max_error": 0.0,
            "rotation_det_max_abs_error": 0.0,
            "projection_parity_max_px": 0.0,
        },
        "reflection_transform_count": 0,
        "failures": [],
        "thresholds": {
            "rotation_max_error": 1e-6,
            "projection_parity_max_px": 1e-4,
        },
    }


def test_migration_pass_is_recomputed_from_structured_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_migration_pass(tmp_path)
    path = _write_manifest(tmp_path / "migration.json", payload)
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)
    assert paper_real_eval._migration_status(str(path))[0] == "PASS"

    payload["checks"]["source_sha_and_mtime_unchanged"] = False
    path = _write_manifest(tmp_path / "forged.json", payload)
    with pytest.raises(ContractError, match="REQUIRED_CHECKS_NOT_TRUE"):
        paper_real_eval._migration_status(str(path))


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("source_changed", "SOURCE_SHA_MISMATCH"),
        ("canonical_null", "CANONICAL_POSE_REQUIRED"),
        ("row_threshold", "FRAME_MAXIMUM_EXCEEDS_THRESHOLD"),
        ("duplicate_output", "OUTPUT_LABEL_PATHS_MUST_BE_UNIQUE"),
    ],
)
def test_migration_pass_revalidates_current_sources_and_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
    message: str,
) -> None:
    payload = _valid_migration_pass(tmp_path)
    gate_path = _write_manifest(tmp_path / "migration.json", payload)
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)

    if violation == "source_changed":
        (tmp_path / "sources" / "0.json").write_text("changed", "utf-8")
    elif violation == "canonical_null":
        output_path = tmp_path / "outputs" / "0.json"
        document = json.loads(output_path.read_text("utf-8"))
        document["objects"][0]["camera_facing_pnp"]["axis_assignment"] = None
        document["objects"][0]["canonical_pose"] = None
        output_path.write_text(json.dumps(document), "utf-8")
    else:
        report_path = tmp_path / "MIGRATION_REPORT.csv"
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
        if violation == "row_threshold":
            rows[0]["projection_parity_max_px"] = "0.01"
        else:
            rows[1]["output_label"] = rows[0]["output_label"]
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(ContractError, match=message):
        paper_real_eval._migration_status(str(gate_path))


def test_migration_source_audit_relative_path_is_resolved_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_migration_pass(tmp_path)
    payload["source_audit_csv"] = "audit.csv"
    path = _write_manifest(tmp_path / "migration.json", payload)
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)
    assert paper_real_eval._migration_status(str(path))[0] == "PASS"


def _valid_symmetry_contract() -> dict:
    identity = geometry.canonical_to_camera_facing_transform("YAW_0")
    yaw_180 = geometry.canonical_to_camera_facing_transform("YAW_180")
    return {
        "schema_version": paper_real_eval.SYMMETRY_CONTRACT_SCHEMA_VERSION,
        "status": "FROZEN",
        "metric_variant": "ADD-S",
        "canonical_axis": "+Y",
        "equivalent_yaw_degrees": [0, 180],
        "accepted_proper_rotations": [identity.tolist(), yaw_180.tolist()],
        "equivalence_basis": {
            "kind": "DECLARED_BENCHMARK_ASSUMPTION",
            "statement": "Yaw rotations 0 and 180 are one benchmark pose class.",
            "physical_inspection_claimed": False,
            "claim_boundary": "Evaluation equivalence only; no signed-axis claim.",
        },
        "reviewer_identity": "benchmark-owner",
        "review_date": "2026-08-01",
        "inclusion_exclusion_rules": [
            "Unladen pallet instances are included.",
            "Cargo and directional fixtures are excluded.",
        ],
        "fixed_without_dev_or_final_pose_results": True,
    }


def test_frozen_symmetry_requires_proper_explicit_reviewed_rotation_set(
    tmp_path: Path,
) -> None:
    payload = _valid_symmetry_contract()
    path = _write_manifest(tmp_path / "symmetry.json", payload)
    status, variant, rotations, _, contract = paper_real_eval._symmetry_status(str(path))
    assert (status, variant, len(rotations)) == ("FROZEN", "ADD-S", 2)
    assert contract is not None
    assert contract.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    payload["accepted_proper_rotations"][1] = np.diag([-1.0, 1.0, 1.0]).tolist()
    path = _write_manifest(tmp_path / "reflection.json", payload)
    with pytest.raises(ContractError, match="SYMMETRY_CONTRACT_INVALID"):
        paper_real_eval._symmetry_status(str(path))


def test_repository_equivalence_gate_revalidates_all_140_current_labels() -> None:
    symmetry_path = (
        paper_real_eval.REPO_ROOT
        / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"
    )
    migration_path = (
        paper_real_eval.REPO_ROOT / "challenge/real_gt_v2/MIGRATION_GATE.json"
    )
    _, variant, rotations, _, symmetry = paper_real_eval._symmetry_status(
        str(symmetry_path)
    )
    assert symmetry is not None

    status, payload = paper_real_eval._migration_status(
        str(migration_path), symmetry_contract=symmetry
    )

    assert status == "PASS"
    assert variant == "ADD-S"
    assert len(rotations) == 2
    assert payload is not None
    assert payload["pose_resolution_mode"] == (
        paper_real_eval.YAW_180_EQUIVALENCE_CLASS
    )
    assert payload["canonical_pose_resolved_count"] == 0
    assert payload["canonical_pose_equivalence_resolved_count"] == 140


@pytest.mark.parametrize(
    "artifact_name",
    ["DEV_CONTRACT_DRY_RUN.json", "FINAL_CONTRACT_DRY_RUN.json"],
)
def test_checked_in_dry_runs_keep_only_the_two_real_remaining_pose_blockers(
    artifact_name: str,
) -> None:
    path = (
        paper_real_eval.REPO_ROOT
        / "challenge/evaluation_v2/dry_runs"
        / artifact_name
    )
    report = json.loads(path.read_text("utf-8"))

    assert report["pose_contract"]["canonical_migration"] == "PASS"
    assert report["pose_contract"]["symmetry"] == "FROZEN"
    assert report["gate_evidence"]["migration_pose_resolution_mode"] == (
        paper_real_eval.YAW_180_EQUIVALENCE_CLASS
    )
    assert report["gate_evidence"]["accepted_symmetry_rotation_count"] == 2
    assert report["metrics"]["pose"]["blocked_reasons"] == [
        "POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR",
        "FINAL_MANIFEST_NOT_FROZEN",
    ]
    assert all(
        report["metrics"]["pose"][field] is None for field in POSE_METRIC_FIELDS
    )


def test_equivalence_migration_pass_is_bound_to_exact_symmetry_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symmetry_path = _write_manifest(
        tmp_path / "symmetry.json", _valid_symmetry_contract()
    )
    _, _, _, _, symmetry = paper_real_eval._symmetry_status(str(symmetry_path))
    assert symmetry is not None
    payload = _valid_migration_pass(
        tmp_path,
        resolution_mode=paper_real_eval.YAW_180_EQUIVALENCE_CLASS,
        symmetry_path=symmetry_path,
    )
    gate_path = _write_manifest(tmp_path / "migration.json", payload)
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)

    assert paper_real_eval._migration_status(
        str(gate_path), symmetry_contract=symmetry
    )[0] == "PASS"

    payload["symmetry_contract_sha256"] = "0" * 64
    tampered = _write_manifest(tmp_path / "migration-tampered.json", payload)
    with pytest.raises(ContractError, match="SYMMETRY_CONTRACT_SHA256_MISMATCH"):
        paper_real_eval._migration_status(
            str(tampered), symmetry_contract=symmetry
        )


@pytest.mark.parametrize("violation", ["row_claim", "signed_pose_claim"])
def test_equivalence_migration_rejects_false_class_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    symmetry_path = _write_manifest(
        tmp_path / "symmetry.json", _valid_symmetry_contract()
    )
    _, _, _, _, symmetry = paper_real_eval._symmetry_status(str(symmetry_path))
    assert symmetry is not None
    payload = _valid_migration_pass(
        tmp_path,
        resolution_mode=paper_real_eval.YAW_180_EQUIVALENCE_CLASS,
        symmetry_path=symmetry_path,
    )
    gate_path = _write_manifest(tmp_path / "migration.json", payload)
    monkeypatch.setattr(paper_real_eval, "REPO_ROOT", tmp_path)

    if violation == "row_claim":
        report_path = tmp_path / "MIGRATION_REPORT.csv"
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
        rows[0]["yaw180_equivalence_class_exact"] = "False"
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        message = "YAW180_EQUIVALENCE_CLASS_NOT_EXACT"
    else:
        output_path = tmp_path / "outputs" / "0.json"
        document = json.loads(output_path.read_text("utf-8"))
        obj = document["objects"][0]
        selected = copy.deepcopy(obj["canonical_pose_candidates"][0])
        obj["camera_facing_pnp"]["axis_assignment"] = selected["axis_assignment"]
        obj["canonical_pose"] = selected
        output_path.write_text(json.dumps(document), "utf-8")
        message = "MUST_NOT_CLAIM_SIGNED_POSE"

    with pytest.raises(ContractError, match=message):
        paper_real_eval._migration_status(
            str(gate_path), symmetry_contract=symmetry
        )


def _passing_pose_context(
    *,
    add_s: bool = True,
    resolution_mode: str = paper_real_eval.SIGNED_CANONICAL_POSE,
) -> paper_real_eval.PoseContractContext:
    selector = SelectorGateReport(
        status=SelectorGateState.PASS,
        overall_accuracy=1.0,
        night_accuracy=1.0,
        minimum_session_accuracy=1.0,
        sample_count=140,
        night_count=28,
        session_count=7,
        tail_dominance_assessed=True,
        tail_dominance_passed=True,
        tail_dominance_notes="pre-registered synthetic test",
        blocked_reason=None,
    )
    gate = build_pose_metric_gate(
        canonical_migration_status="PASS",
        selector_report=selector,
        symmetry_status="FROZEN",
        final_manifest_frozen=True,
    )
    rotations = (geometry.canonical_to_camera_facing_transform("YAW_0"),)
    if add_s:
        rotations += (geometry.canonical_to_camera_facing_transform("YAW_180"),)
    return paper_real_eval.PoseContractContext(
        gate=gate,
        metric_variant="ADD-S" if add_s else "ADD",
        equivalent_rotations=rotations,
        pose_resolution_mode=resolution_mode,
        evidence={},
    )


def _synthetic_pose_inputs() -> tuple[
    dict[str, paper_real_eval.PositiveTarget],
    dict[str, paper_real_eval.DetectionCandidate],
    PnPSelectionResult,
]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.0, 0.0, 3.0]
    target = paper_real_eval.PositiveTarget(
        frame_id="frame",
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
    )
    prediction = paper_real_eval.DetectionCandidate(
        frame_id="frame",
        is_positive=True,
        score=1.0,
        box_xyxy=target.box_xyxy.copy(),
        keypoints_xy=np.zeros((9, 2), dtype=np.float64),
        target_iou=1.0,
    )
    candidates = _candidate_pair(
        (geometry.AxisAssignment.YAW_0, geometry.AxisAssignment.YAW_180),
        transform[:3, 3],
    )
    selection = PnPSelectionResult(
        status=SelectorStatus.SELECTED,
        selected_hypothesis="short-face-front",
        hypotheses=(),  # type: ignore[arg-type]
        canonical_candidates=candidates,
        ambiguity="SIGNED_AXIS_UNRESOLVED_TWO_CANDIDATES",
    )
    return {"frame": target}, {"frame": prediction}, selection


def _candidate_pair(
    assignments: tuple[geometry.AxisAssignment, geometry.AxisAssignment],
    translation: np.ndarray,
) -> tuple[CanonicalPoseCandidate, CanonicalPoseCandidate]:
    candidates = tuple(
        CanonicalPoseCandidate(
            axis_assignment=assignment,
            rotation=geometry.canonical_to_camera_facing_transform(assignment.value),
            translation=translation.copy(),
            pose_transform=np.block(
                [
                    [
                        geometry.canonical_to_camera_facing_transform(assignment.value),
                        translation[:, None],
                    ],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ),
            keypoint_permutation=geometry.canonical_to_camera_facing_keypoint_permutation(
                assignment
            ),
        )
        for assignment in assignments
    )
    return (candidates[0], candidates[1])


def test_blocked_evaluator_pose_path_never_invokes_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, _ = _synthetic_pose_inputs()
    blocked_gate = build_pose_metric_gate(
        canonical_migration_status="BLOCKED",
        selector_report=SelectorGateReport.not_run(),
        symmetry_status="NOT_FROZEN",
        final_manifest_frozen=False,
    )
    context = paper_real_eval.PoseContractContext(
        gate=blocked_gate,
        metric_variant="ADD",
        equivalent_rotations=(),
        pose_resolution_mode=None,
        evidence={},
    )
    monkeypatch.setattr(
        paper_real_eval,
        "select_pnp_hypotheses",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selector must not run before the gate")
        ),
    )
    with pytest.raises(ContractError, match="FORBIDDEN_BEFORE_ALL_GATES_PASS"):
        paper_real_eval.evaluate_pose_records(targets, predictions, context)


def test_valid_gates_enable_gt_free_selector_and_pose_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, selection = _synthetic_pose_inputs()
    calls: list[tuple] = []

    def fake_selector(*args):
        calls.append(args)
        return selection

    monkeypatch.setattr(paper_real_eval, "select_pnp_hypotheses", fake_selector)
    context = _passing_pose_context(add_s=True)
    records = paper_real_eval.evaluate_pose_records(targets, predictions, context)
    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert isinstance(calls[0][2], geometry.PhysicalDimensionsXYZ)
    assert len(records) == 1
    assert records[0].adds_error_m == pytest.approx(0.0)
    metrics = summarize_pose_errors(records, context.gate, metric_variant="ADD-S")
    assert metrics["status"] == "READY"
    assert metrics["add_or_adds_auc"] == pytest.approx(1.0)
    assert metrics["rotation_median_deg"] == pytest.approx(0.0)
    assert metrics["translation_median_m"] == pytest.approx(0.0)
    assert metrics["yaw_median_deg"] == pytest.approx(0.0)


def test_yaw180_equivalence_class_collapses_without_signed_gt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, selection = _synthetic_pose_inputs()
    target = targets["frame"]
    # Reverse the target representative order to prove that neither member is
    # treated as a privileged signed label.
    targets["frame"] = replace(
        target,
        canonical_pose_transform=None,
        canonical_pose_candidate_transforms=tuple(
            candidate.pose_transform for candidate in reversed(selection.canonical_candidates)
        ),
    )
    monkeypatch.setattr(
        paper_real_eval, "select_pnp_hypotheses", lambda *_args: selection
    )
    context = _passing_pose_context(
        add_s=True,
        resolution_mode=paper_real_eval.YAW_180_EQUIVALENCE_CLASS,
    )

    records = paper_real_eval.evaluate_pose_records(targets, predictions, context)
    assert len(records) == 1
    assert records[0].adds_error_m == pytest.approx(0.0)
    assert records[0].rotation_error_deg == pytest.approx(0.0)
    assert records[0].translation_error_m == pytest.approx(0.0)
    assert records[0].yaw_error_deg == pytest.approx(0.0)


def test_yaw180_equivalence_does_not_collapse_wrong_wd_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, short_selection = _synthetic_pose_inputs()
    target = targets["frame"]
    targets["frame"] = replace(
        target,
        canonical_pose_transform=None,
        canonical_pose_candidate_transforms=tuple(
            candidate.pose_transform for candidate in short_selection.canonical_candidates
        ),
    )
    long_candidates = _candidate_pair(
        (geometry.AxisAssignment.YAW_90, geometry.AxisAssignment.YAW_270),
        np.array([0.0, 0.0, 3.0]),
    )
    long_selection = replace(
        short_selection,
        selected_hypothesis="long-face-front",
        canonical_candidates=long_candidates,
    )
    monkeypatch.setattr(
        paper_real_eval, "select_pnp_hypotheses", lambda *_args: long_selection
    )
    context = _passing_pose_context(
        add_s=True,
        resolution_mode=paper_real_eval.YAW_180_EQUIVALENCE_CLASS,
    )

    records = paper_real_eval.evaluate_pose_records(targets, predictions, context)
    assert len(records) == 1
    assert records[0].adds_error_m > 0.0
    assert records[0].rotation_error_deg == pytest.approx(90.0)
    assert records[0].translation_error_m == pytest.approx(0.0)
    assert records[0].yaw_error_deg == pytest.approx(90.0)


def test_cross_parity_target_candidates_are_rejected_as_forged_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, selection = _synthetic_pose_inputs()
    target = targets["frame"]
    cross_parity = _candidate_pair(
        (geometry.AxisAssignment.YAW_0, geometry.AxisAssignment.YAW_90),
        np.array([0.0, 0.0, 3.0]),
    )
    targets["frame"] = replace(
        target,
        canonical_pose_transform=None,
        canonical_pose_candidate_transforms=tuple(
            candidate.pose_transform for candidate in cross_parity
        ),
    )
    monkeypatch.setattr(
        paper_real_eval, "select_pnp_hypotheses", lambda *_args: selection
    )
    context = _passing_pose_context(
        add_s=True,
        resolution_mode=paper_real_eval.YAW_180_EQUIVALENCE_CLASS,
    )

    with pytest.raises(ContractError, match="YAW180_EQUIVALENCE_CLASS_INVALID"):
        paper_real_eval.evaluate_pose_records(targets, predictions, context)


def test_cross_parity_prediction_candidates_cannot_be_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, selection = _synthetic_pose_inputs()
    cross_parity = _candidate_pair(
        (geometry.AxisAssignment.YAW_0, geometry.AxisAssignment.YAW_90),
        np.array([0.0, 0.0, 3.0]),
    )
    forged_selection = replace(selection, canonical_candidates=cross_parity)
    monkeypatch.setattr(
        paper_real_eval,
        "select_pnp_hypotheses",
        lambda *_args: forged_selection,
    )
    context = _passing_pose_context(add_s=True)

    with pytest.raises(
        paper_real_eval.PoseEvaluationNotRunnable,
        match="SIGNED_AXIS_AMBIGUITY_NOT_COVERED",
    ):
        paper_real_eval.evaluate_pose_records(targets, predictions, context)


def test_identity_add_contract_cannot_choose_one_signed_candidate_from_gt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, predictions, selection = _synthetic_pose_inputs()
    monkeypatch.setattr(
        paper_real_eval, "select_pnp_hypotheses", lambda *_args: selection
    )
    with pytest.raises(
        paper_real_eval.PoseEvaluationNotRunnable,
        match="SIGNED_AXIS_AMBIGUITY_NOT_COVERED",
    ):
        paper_real_eval.evaluate_pose_records(
            targets, predictions, _passing_pose_context(add_s=False)
        )
