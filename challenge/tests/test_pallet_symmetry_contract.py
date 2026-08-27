from __future__ import annotations

import copy
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.annotate import pallet_geometry as geometry
from scripts.annotate.pallet_symmetry import (
    EQUIVALENCE_BASIS_KIND,
    SYMMETRY_CONTRACT_SCHEMA_VERSION,
    SymmetryContractError,
    load_symmetry_contract,
    validate_symmetry_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"


def _payload() -> dict:
    return json.loads(CONTRACT_PATH.read_text("utf-8"))


def test_frozen_repository_contract_loads_with_file_identity() -> None:
    contract = load_symmetry_contract(
        CONTRACT_PATH,
        today=date(2026, 8, 27),
    )

    assert contract.status == "FROZEN"
    assert contract.metric_variant == "ADD-S"
    assert contract.equivalent_yaw_degrees == (0, 180)
    assert contract.source_path == CONTRACT_PATH.resolve()
    assert contract.sha256 == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert contract.payload["equivalence_basis"]["kind"] == (
        EQUIVALENCE_BASIS_KIND
    )
    assert contract.payload["equivalence_basis"]["physical_inspection_claimed"] is False
    assert len(contract.rotations) == 2
    assert np.array_equal(
        contract.rotations[0],
        geometry.canonical_to_camera_facing_transform("YAW_0"),
    )
    assert np.array_equal(
        contract.rotations[1],
        geometry.canonical_to_camera_facing_transform("YAW_180"),
    )


def test_declared_assumption_cannot_claim_physical_inspection() -> None:
    payload = _payload()
    payload["equivalence_basis"]["physical_inspection_claimed"] = True

    with pytest.raises(SymmetryContractError, match="physical_inspection_claimed=false"):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


@pytest.mark.parametrize("field", ["statement", "claim_boundary"])
def test_declared_assumption_requires_truthful_boundary_text(field: str) -> None:
    payload = _payload()
    payload["equivalence_basis"][field] = "   "

    with pytest.raises(SymmetryContractError, match=field):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


def test_v1_or_unstructured_physical_evidence_cannot_masquerade_as_v2() -> None:
    payload = _payload()
    payload["schema_version"] = "real_pallet_symmetry_contract_v1"
    payload["physical_evidence"] = "reviewed in person"

    with pytest.raises(SymmetryContractError, match="fields mismatch"):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


@pytest.mark.parametrize(
    "yaws",
    ([180, 0], [0], [0, 90, 180], [0, 180, 360]),
)
def test_adds_yaw_set_is_exact_and_ordered(yaws: list[int]) -> None:
    payload = _payload()
    payload["equivalent_yaw_degrees"] = yaws

    with pytest.raises(SymmetryContractError, match=r"exactly \[0, 180\]"):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


def test_reflection_is_not_an_accepted_symmetry_rotation() -> None:
    payload = _payload()
    payload["accepted_proper_rotations"][1] = np.diag([-1, 1, 1]).tolist()

    with pytest.raises(SymmetryContractError, match="reflection"):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


def test_other_proper_rotation_is_not_silently_added() -> None:
    payload = _payload()
    payload["accepted_proper_rotations"][1] = (
        geometry.canonical_to_camera_facing_transform("YAW_90").tolist()
    )

    with pytest.raises(SymmetryContractError, match="exact canonical yaw set"):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"reviewer_identity": ""}, "reviewer_identity"),
        ({"review_date": "2026-08-28"}, "future"),
        ({"inclusion_exclusion_rules": []}, "inclusion_exclusion_rules"),
        ({"fixed_without_dev_or_final_pose_results": False}, "must be true"),
    ),
)
def test_review_and_pre_result_evidence_are_required(
    mutation: dict,
    message: str,
) -> None:
    payload = _payload()
    payload.update(copy.deepcopy(mutation))

    with pytest.raises(SymmetryContractError, match=message):
        validate_symmetry_contract(payload, today=date(2026, 8, 27))


def test_schema_constant_is_v2() -> None:
    assert SYMMETRY_CONTRACT_SCHEMA_VERSION == "real_pallet_symmetry_contract_v2"
