"""Contract tests for the previously evaluated 45-frame wood DEV set."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import pytest

from challenge.evaluation_v2.real_dataset_contract import (
    ContractError,
    REPO_ROOT,
    WOOD_OBJECT_TYPE,
    PopulationId,
    PopulationRole,
    load_repo_population,
    validate_evaluation_pair,
    validate_wood_dev_membership,
)


def test_wood_population_is_exact_45_across_two_qualified_sessions() -> None:
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45, validate_files=True)
    validate_wood_dev_membership(wood, validate_files=True)

    assert PopulationRole.CROSS_SHAPE_DEV.value == "CROSS_SHAPE_DEV"
    assert wood.role is PopulationRole.CROSS_SHAPE_DEV
    assert wood.count == 45
    assert Counter(item.session_id for item in wood.items) == {
        "wood_183705": 25,
        "wood_184309": 20,
    }
    assert all(item.frame_id.startswith(f"{item.session_id}:") for item in wood.items)
    assert all(item.object_type == WOOD_OBJECT_TYPE for item in wood.items)
    assert all(item.population_role == "CROSS_SHAPE_DEV" for item in wood.items)
    assert all(
        item.source_population == PopulationId.DEV_WOOD_POS45.value
        for item in wood.items
    )
    assert wood.provenance["role"] == "CROSS_SHAPE_DEV"
    assert wood.provenance["previously_evaluated"] is True
    assert wood.provenance["final_eligible"] is False


def test_wood_manifest_uses_migrated_gt_without_changing_source_labels() -> None:
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45, validate_files=True)
    for item in wood.items:
        assert item.image_path == item.image
        assert item.gt_v2_path == item.label
        assert item.label is not None
        assert item.label.startswith("challenge/real_gt_v2/migrated_gt_wood/")
        migrated = json.loads((REPO_ROOT / item.label).read_text("utf-8"))
        source = migrated["real_gt_v2_migration"]["source_label"]
        assert source.startswith("challenge/data/01_real/manual_gt/wood_pallet_")
        assert (REPO_ROOT / source).is_file()
        assert migrated["object_type"] == WOOD_OBJECT_TYPE
        assert migrated["population_role"] == "DEV"


def test_wood_dev_can_pair_with_shared_dev_negative_population() -> None:
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    pair = validate_evaluation_pair(wood, negative, PopulationRole.CROSS_SHAPE_DEV)
    assert pair.ready is True
    assert pair.pair_sha256 is not None
    with pytest.raises(ContractError, match="CLI_ROLE_DOES_NOT_MATCH_MANIFEST_ROLE"):
        validate_evaluation_pair(wood, negative, PopulationRole.DEV)


def test_wood_manifest_binds_the_passed_audit_snapshot() -> None:
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    audit_path = REPO_ROOT / wood.provenance["source_audit"]
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    assert (
        hashlib.sha256(audit_bytes).hexdigest()
        == wood.provenance["source_audit_sha256"]
    )
    assert audit["status"] == "PASS"
    assert audit["count"] == 45
    assert audit["checks"]["negative_image_overlap_zero"] is True
    assert audit["checks"]["qualified_negative_frame_id_overlap_zero"] is True
    assert audit["bare_frame_id_collisions_with_dev_neg"] == 45
    assert audit["qualified_frame_id_collisions_with_dev_neg"] == 0
