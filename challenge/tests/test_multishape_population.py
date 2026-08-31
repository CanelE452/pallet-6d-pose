"""Object-aware alias, ordered-union and FINAL placeholder contract tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

from challenge.evaluation_v2.real_dataset_contract import (
    ContractError,
    MembershipStatus,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    PopulationId,
    PopulationRole,
    load_population_manifest,
    load_repo_population,
    manifest_path,
    validate_evaluation_pair,
    validate_multishape_dev_membership,
    validate_plastic_alias_membership,
    validate_repo_population_contract,
)
from scripts.annotate.build_multishape_manifests import (
    NEW_MANIFEST_IDS,
    assert_legacy_manifests_unchanged,
    build_payloads,
    materialize,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return path


def test_object_explicit_plastic_ids_are_exact_ordered_legacy_aliases() -> None:
    dev = load_repo_population(PopulationId.DEV_POS140)
    common = load_repo_population(PopulationId.COMMON_DEV_POS128)
    explicit_dev = load_repo_population(PopulationId.DEV_PLASTIC_POS140)
    explicit_common = load_repo_population(PopulationId.COMMON_DEV_PLASTIC_POS128)
    validate_plastic_alias_membership(dev, common, explicit_dev, explicit_common)

    def identity(item):
        return item.frame_id, item.image, item.label, item.source_set, item.domain

    assert tuple(map(identity, explicit_dev.items)) == tuple(map(identity, dev.items))
    assert tuple(map(identity, explicit_common.items)) == tuple(
        map(identity, common.items)
    )
    assert all(item.object_type == PLASTIC_OBJECT_TYPE for item in explicit_dev.items)
    assert all(
        item.object_type == PLASTIC_OBJECT_TYPE for item in explicit_common.items
    )


def test_multishape_is_literal_ordered_plastic128_plus_wood45() -> None:
    plastic = load_repo_population(PopulationId.COMMON_DEV_PLASTIC_POS128)
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    combined = load_repo_population(PopulationId.COMMON_DEV_MULTISHAPE_POS)
    validate_multishape_dev_membership(plastic, wood, combined)

    assert combined.count == 173
    assert combined.items == plastic.items + wood.items
    assert combined.frame_ids[:128] == plastic.frame_ids
    assert combined.frame_ids[128:] == wood.frame_ids
    assert Counter(item.object_type for item in combined.items) == {
        PLASTIC_OBJECT_TYPE: 128,
        WOOD_OBJECT_TYPE: 45,
    }


def test_multishape_relation_rejects_even_a_reordered_frozen_tuple() -> None:
    plastic = load_repo_population(PopulationId.COMMON_DEV_PLASTIC_POS128)
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    combined = load_repo_population(PopulationId.COMMON_DEV_MULTISHAPE_POS)
    reordered = replace(
        combined, items=(combined.items[1], combined.items[0], *combined.items[2:])
    )
    with pytest.raises(ContractError, match="MULTISHAPE_ORDERED_UNION_MISMATCH"):
        validate_multishape_dev_membership(plastic, wood, reordered)


def test_new_dev_pairs_and_legacy_common_pair_are_allowed_but_full140_is_not() -> None:
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    for population_id in (
        PopulationId.COMMON_DEV_POS128,
        PopulationId.COMMON_DEV_PLASTIC_POS128,
        PopulationId.COMMON_DEV_MULTISHAPE_POS,
    ):
        assert validate_evaluation_pair(
            load_repo_population(population_id), negative, PopulationRole.DEV
        ).ready
    for population_id in (PopulationId.DEV_POS140, PopulationId.DEV_PLASTIC_POS140):
        with pytest.raises(ContractError, match="DEV_COMPARISON_REQUIRES_COMMON"):
            validate_evaluation_pair(
                load_repo_population(population_id), negative, PopulationRole.DEV
            )
    assert validate_evaluation_pair(
        load_repo_population(PopulationId.DEV_WOOD_POS45),
        negative,
        PopulationRole.CROSS_SHAPE_DEV,
    ).ready


def test_all_new_final_populations_are_unavailable_not_empty_tests() -> None:
    final_negative = load_repo_population(PopulationId.FINAL_NEG)
    for population_id in (
        PopulationId.FINAL_PLASTIC_POS,
        PopulationId.FINAL_WOOD_POS,
        PopulationId.FINAL_ALL_POS,
    ):
        positive = load_repo_population(population_id)
        assert positive.count == 0
        assert positive.membership_status is MembershipStatus.UNAVAILABLE
        assert positive.frozen is False
        assert positive.membership_sha256 is None
        expected_scope = {
            PopulationId.FINAL_PLASTIC_POS: (PLASTIC_OBJECT_TYPE,),
            PopulationId.FINAL_WOOD_POS: (WOOD_OBJECT_TYPE,),
            PopulationId.FINAL_ALL_POS: (PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE),
        }[population_id]
        assert positive.object_types == expected_scope
        with pytest.raises(ContractError, match="FINAL_MEMBERSHIP_UNAVAILABLE"):
            validate_evaluation_pair(positive, final_negative, PopulationRole.FINAL)
        blocked = validate_evaluation_pair(
            positive,
            final_negative,
            PopulationRole.FINAL,
            allow_unavailable_final=True,
        )
        assert blocked.ready is False


def test_object_aware_item_fields_and_dispatch_metadata_are_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        manifest_path(PopulationId.COMMON_DEV_PLASTIC_POS128).read_text("utf-8")
    )
    assert {
        "frame_id",
        "object_type",
        "session_id",
        "image_path",
        "gt_v2_path",
        "population_role",
        "source_population",
    } <= set(payload["items"][0])

    bad_scope = json.loads(json.dumps(payload))
    bad_scope["object_types"] = [WOOD_OBJECT_TYPE]
    with pytest.raises(ContractError, match="OBJECT_TYPE_SCOPE_MISMATCH"):
        load_population_manifest(
            _write(tmp_path / "bad_object_scope.json", bad_scope),
            validate_files=False,
        )

    missing = json.loads(json.dumps(payload))
    del missing["items"][0]["object_type"]
    with pytest.raises(ContractError, match="OBJECT_AWARE_FIELDS_REQUIRED"):
        load_population_manifest(
            _write(tmp_path / "missing_object_type.json", missing), validate_files=False
        )

    tampered = json.loads(json.dumps(payload))
    tampered["items"][0]["domain"] = "NIGHT"
    with pytest.raises(ContractError, match="MEMBERSHIP_HASH_MISMATCH"):
        load_population_manifest(
            _write(tmp_path / "tampered_dispatch.json", tampered), validate_files=False
        )


def test_builder_is_deterministic_append_only_and_old_manifests_are_unchanged(
    tmp_path: Path,
) -> None:
    assert_legacy_manifests_unchanged()
    payloads = build_payloads()
    assert tuple(payloads) == NEW_MANIFEST_IDS
    created = materialize(payloads, output_dir=tmp_path)
    assert len(created) == len(NEW_MANIFEST_IDS) == 7
    assert materialize(payloads, output_dir=tmp_path, check_only=True) == created
    for population_id in NEW_MANIFEST_IDS:
        assert (tmp_path / f"{population_id.value}.json").read_bytes() == manifest_path(
            population_id
        ).read_bytes()


def test_full_repo_population_contract_includes_object_aware_populations() -> None:
    manifests = validate_repo_population_contract(validate_files=True)
    assert set(manifests) == set(PopulationId)
    assert manifests[PopulationId.COMMON_DEV_MULTISHAPE_POS].count == 173
