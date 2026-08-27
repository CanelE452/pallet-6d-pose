"""Validation for the frozen real-pallet paper symmetry contract.

The contract records an evaluation equivalence, not an inferred per-frame axis
label.  Version 2 deliberately distinguishes a declared benchmark assumption
from physical-inspection evidence so callers cannot turn the former into the
latter by wording a generic ``physical_evidence`` string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Package import (tests/evaluation code).
    from .pallet_geometry import (
        canonical_to_camera_facing_transform,
        validate_proper_rotation,
    )
except ImportError:  # Direct ``python scripts/annotate/...`` execution.
    from pallet_geometry import (  # type: ignore[no-redef]
        canonical_to_camera_facing_transform,
        validate_proper_rotation,
    )


SYMMETRY_CONTRACT_SCHEMA_VERSION = "real_pallet_symmetry_contract_v2"
# Short alias retained for annotation-side callers that use the same naming as
# ``real_gt_v2_schema``.
SCHEMA_VERSION = SYMMETRY_CONTRACT_SCHEMA_VERSION
STATUS = "FROZEN"
METRIC_VARIANT = "ADD-S"
CANONICAL_AXIS = "+Y"
EQUIVALENCE_BASIS_KIND = "DECLARED_BENCHMARK_ASSUMPTION"
EQUIVALENT_YAW_DEGREES = (0, 180)

_TOP_LEVEL_FIELDS = frozenset({
    "schema_version",
    "status",
    "metric_variant",
    "canonical_axis",
    "equivalent_yaw_degrees",
    "accepted_proper_rotations",
    "equivalence_basis",
    "reviewer_identity",
    "review_date",
    "inclusion_exclusion_rules",
    "fixed_without_dev_or_final_pose_results",
})
_BASIS_FIELDS = frozenset({
    "kind",
    "statement",
    "physical_inspection_claimed",
    "claim_boundary",
})


class SymmetryContractError(ValueError):
    """Raised when a real-pallet symmetry artifact violates schema v2."""


@dataclass(frozen=True)
class ValidatedSymmetryContract:
    """Validated values consumed by evaluation and migration gates."""

    status: str
    metric_variant: str
    canonical_axis: str
    equivalent_yaw_degrees: tuple[int, ...]
    rotations: tuple[np.ndarray, ...]
    payload: dict[str, Any]
    sha256: str
    source_path: Path | None
    reviewer_identity: str
    review_date: date
    inclusion_exclusion_rules: tuple[str, ...]
    equivalence_basis_statement: str
    equivalence_claim_boundary: str

    @property
    def accepted_proper_rotations(self) -> tuple[np.ndarray, ...]:
        """Schema-field alias for callers that prefer the artifact name."""

        return self.rotations


# Backward-friendly semantic alias; the stable cross-module name is the more
# explicit ``ValidatedSymmetryContract``.
PalletSymmetryContract = ValidatedSymmetryContract


def _fail(message: str) -> None:
    raise SymmetryContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        _fail(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value.strip()


def _validate_yaws(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(type(item) is not int for item in value)
    ):
        _fail("equivalent_yaw_degrees must be an integer list")
    yaws = tuple(value)
    if yaws != EQUIVALENT_YAW_DEGREES:
        _fail("ADD-S requires equivalent_yaw_degrees exactly [0, 180]")
    return yaws


def _validate_rotations(
    value: Any,
    yaws: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(yaws)
    ):
        _fail("accepted_proper_rotations must match the explicit yaw set")

    rotations: list[np.ndarray] = []
    for index, item in enumerate(value):
        try:
            rotation = validate_proper_rotation(
                item,
                name=f"accepted_proper_rotations[{index}]",
                atol=1e-9,
            )
        except (TypeError, ValueError) as exc:
            raise SymmetryContractError(str(exc)) from exc
        expected = canonical_to_camera_facing_transform(f"YAW_{yaws[index]}")
        if not np.array_equal(rotation, expected):
            _fail(
                "accepted_proper_rotations do not match the exact canonical "
                "yaw set [0, 180]"
            )
        immutable = rotation.copy()
        immutable.setflags(write=False)
        rotations.append(immutable)
    return tuple(rotations)


def validate_symmetry_contract(
    payload: Any,
    *,
    today: date | None = None,
    source_path: Path | None = None,
    sha256: str | None = None,
) -> ValidatedSymmetryContract:
    """Validate one frozen ``real_pallet_symmetry_contract_v2`` object."""

    data = _mapping(payload, "contract")
    _exact_fields(data, _TOP_LEVEL_FIELDS, "contract")

    if data["schema_version"] != SYMMETRY_CONTRACT_SCHEMA_VERSION:
        _fail(
            "schema_version must equal "
            f"{SYMMETRY_CONTRACT_SCHEMA_VERSION!r}"
        )
    if data["status"] != STATUS:
        _fail("status must equal 'FROZEN'")
    if data["metric_variant"] != METRIC_VARIANT:
        _fail("metric_variant must equal 'ADD-S'")
    if data["canonical_axis"] != CANONICAL_AXIS:
        _fail("canonical_axis must equal '+Y'")

    yaws = _validate_yaws(data["equivalent_yaw_degrees"])
    rotations = _validate_rotations(data["accepted_proper_rotations"], yaws)

    basis = _mapping(data["equivalence_basis"], "equivalence_basis")
    _exact_fields(basis, _BASIS_FIELDS, "equivalence_basis")
    if basis["kind"] != EQUIVALENCE_BASIS_KIND:
        _fail(
            "equivalence_basis.kind must equal "
            f"{EQUIVALENCE_BASIS_KIND!r}"
        )
    if basis["physical_inspection_claimed"] is not False:
        _fail(
            "DECLARED_BENCHMARK_ASSUMPTION requires "
            "physical_inspection_claimed=false"
        )
    statement = _nonempty_string(
        basis["statement"], "equivalence_basis.statement"
    )
    claim_boundary = _nonempty_string(
        basis["claim_boundary"], "equivalence_basis.claim_boundary"
    )

    reviewer = _nonempty_string(data["reviewer_identity"], "reviewer_identity")
    raw_date = _nonempty_string(data["review_date"], "review_date")
    try:
        parsed_date = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise SymmetryContractError("review_date must be an ISO date") from exc
    reference_date = today or date.today()
    if parsed_date > reference_date:
        _fail("review_date cannot be in the future")

    raw_rules = data["inclusion_exclusion_rules"]
    if (
        not isinstance(raw_rules, Sequence)
        or isinstance(raw_rules, (str, bytes))
        or not raw_rules
    ):
        _fail("inclusion_exclusion_rules must be a non-empty string list")
    rules = tuple(
        _nonempty_string(rule, f"inclusion_exclusion_rules[{index}]")
        for index, rule in enumerate(raw_rules)
    )

    if data["fixed_without_dev_or_final_pose_results"] is not True:
        _fail("fixed_without_dev_or_final_pose_results must be true")

    payload_copy = copy.deepcopy(dict(data))
    if sha256 is None:
        canonical_bytes = json.dumps(
            payload_copy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    return ValidatedSymmetryContract(
        status=STATUS,
        metric_variant=METRIC_VARIANT,
        canonical_axis=CANONICAL_AXIS,
        equivalent_yaw_degrees=yaws,
        rotations=rotations,
        payload=payload_copy,
        sha256=sha256,
        source_path=source_path,
        reviewer_identity=reviewer,
        review_date=parsed_date,
        inclusion_exclusion_rules=rules,
        equivalence_basis_statement=statement,
        equivalence_claim_boundary=claim_boundary,
    )


def load_symmetry_contract(
    path: str | Path,
    *,
    today: date | None = None,
) -> ValidatedSymmetryContract:
    """Load and validate a frozen symmetry JSON artifact."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SymmetryContractError(
            f"symmetry contract is unreadable: {source}: {exc}"
        ) from exc
    return validate_symmetry_contract(
        payload,
        today=today,
        source_path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "CANONICAL_AXIS",
    "EQUIVALENCE_BASIS_KIND",
    "EQUIVALENT_YAW_DEGREES",
    "METRIC_VARIANT",
    "PalletSymmetryContract",
    "SCHEMA_VERSION",
    "SYMMETRY_CONTRACT_SCHEMA_VERSION",
    "STATUS",
    "SymmetryContractError",
    "ValidatedSymmetryContract",
    "load_symmetry_contract",
    "validate_symmetry_contract",
]
