"""Cross-population identity checks for the 173-frame multishape DEV union."""

from __future__ import annotations

import json

from challenge.evaluation_v2.real_dataset_contract import (
    REPO_ROOT,
    PopulationId,
    load_repo_population,
)


def test_multishape_member_identities_are_unique_and_positive_negative_disjoint() -> (
    None
):
    positive = load_repo_population(PopulationId.COMMON_DEV_MULTISHAPE_POS)
    negative = load_repo_population(PopulationId.DEV_NEG2689)

    assert len(positive.frame_ids) == len(set(positive.frame_ids)) == 173
    assert len({item.image for item in positive.items}) == 173
    assert len({item.label for item in positive.items}) == 173
    assert set(positive.frame_ids).isdisjoint(negative.frame_ids)
    assert {item.image for item in positive.items}.isdisjoint(
        item.image for item in negative.items
    )


def test_session_qualification_resolves_all_45_wood_vs_negative_bare_id_collisions() -> (
    None
):
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    negative = load_repo_population(PopulationId.DEV_NEG2689)
    negative_ids = set(negative.frame_ids)
    bare_wood_ids = {item.frame_id.split(":", 1)[1] for item in wood.items}

    assert len(bare_wood_ids) == 45
    assert bare_wood_ids <= negative_ids
    assert set(wood.frame_ids).isdisjoint(negative_ids)


def test_wood_hash_duplicate_audit_is_bound_to_manifest_provenance() -> None:
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45)
    audit = json.loads((REPO_ROOT / wood.provenance["source_audit"]).read_text("utf-8"))
    checks = audit["checks"]
    assert checks["unique_image_sha256"] is True
    assert checks["unique_decoded_pixel_sha256"] is True
    assert checks["plastic_overlap_zero"] is True
    assert checks["common_plastic_overlap_zero"] is True
    assert checks["negative_image_overlap_zero"] is True
