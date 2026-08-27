"""Fail-closed population contract for paper-facing real evaluation.

The historical scripts use one external 140-frame manifest, with one script
silently removing twelve fine-tuning overlaps and the other retaining them.
This module replaces that implicit behaviour with repo-local manifests whose
membership, role and availability are machine checked.

``FINAL_POS == FINAL_NEG == 0`` currently means *membership unavailable*.  It
does not mean that no real data exists and it must never be evaluated as a
valid empty final test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "challenge" / "real_gt_v2" / "manifests"
MANIFEST_SCHEMA_VERSION = "pallet_pose_population_manifest_v1"
INVALID_GT_QUARANTINE_PATH = (
    REPO_ROOT / "challenge" / "real_gt_v2" / "INVALID_GT_QUARANTINE.json"
)
INVALID_GT_QUARANTINE_SCHEMA = "real_pallet_invalid_gt_quarantine_v1"


class ContractError(ValueError):
    """Raised when population membership cannot be trusted."""


class PopulationId(str, Enum):
    DEV_POS140 = "DEV_POS140"
    COMMON_DEV_POS128 = "COMMON_DEV_POS128"
    DEV_NEG2689 = "DEV_NEG2689"
    FINAL_POS = "FINAL_POS"
    FINAL_NEG = "FINAL_NEG"


class PopulationKind(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class PopulationRole(str, Enum):
    DEV = "DEV"
    FINAL = "FINAL"


class MembershipStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class PopulationExpectation:
    count: int
    kind: PopulationKind
    role: PopulationRole


EXPECTED_POPULATIONS: Mapping[PopulationId, PopulationExpectation] = {
    PopulationId.DEV_POS140: PopulationExpectation(140, PopulationKind.POSITIVE, PopulationRole.DEV),
    PopulationId.COMMON_DEV_POS128: PopulationExpectation(128, PopulationKind.POSITIVE, PopulationRole.DEV),
    PopulationId.DEV_NEG2689: PopulationExpectation(2689, PopulationKind.NEGATIVE, PopulationRole.DEV),
    PopulationId.FINAL_POS: PopulationExpectation(0, PopulationKind.POSITIVE, PopulationRole.FINAL),
    PopulationId.FINAL_NEG: PopulationExpectation(0, PopulationKind.NEGATIVE, PopulationRole.FINAL),
}


@dataclass(frozen=True)
class ManifestItem:
    """One immutable population member.

    Paths are always repository-relative.  Paper evaluation may be invoked
    with an absolute *manifest* path supplied by the caller, but a manifest may
    not smuggle machine-specific absolute data paths into the contract.
    """

    frame_id: str
    image: str
    label: str | None = None
    source_set: str | None = None
    domain: str | None = None

    def canonical_record(self) -> dict[str, str]:
        out = {"frame_id": self.frame_id, "image": self.image}
        if self.label is not None:
            out["label"] = self.label
        if self.source_set is not None:
            out["source_set"] = self.source_set
        if self.domain is not None:
            out["domain"] = self.domain
        return out


@dataclass(frozen=True)
class PopulationManifest:
    schema_version: str
    population_id: PopulationId
    kind: PopulationKind
    role: PopulationRole
    membership_status: MembershipStatus
    frozen: bool
    expected_count: int
    membership_sha256: str | None
    items: tuple[ManifestItem, ...]
    source_path: Path
    unavailable_reason: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.membership_status is MembershipStatus.AVAILABLE

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(item.frame_id for item in self.items)

    def summary(self) -> dict[str, Any]:
        return {
            "population_id": self.population_id.value,
            "kind": self.kind.value,
            "role": self.role.value,
            "membership_status": self.membership_status.value,
            "frozen": self.frozen,
            "count": self.count,
            "membership_sha256": self.membership_sha256,
            "unavailable_reason": self.unavailable_reason,
            "manifest": _display_path(self.source_path),
        }


@dataclass(frozen=True)
class EvaluationPopulationPair:
    positive: PopulationManifest
    negative: PopulationManifest
    role: PopulationRole
    ready: bool
    blocked_reason: str | None
    pair_sha256: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "ready": self.ready,
            "blocked_reason": self.blocked_reason,
            "pair_sha256": self.pair_sha256,
            "positive": self.positive.summary(),
            "negative": self.negative.summary(),
        }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def manifest_path(population_id: PopulationId | str) -> Path:
    """Return the repo-local manifest path for a named population."""

    population_id = _enum(PopulationId, population_id, "population_id")
    return MANIFEST_DIR / f"{population_id.value}.json"


def membership_sha256(items: Iterable[ManifestItem]) -> str:
    """Hash ordered membership identity, including data paths and strata."""

    digest = hashlib.sha256()
    for item in items:
        line = json.dumps(
            item.canonical_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _enum(enum_type: type[Enum], value: Any, field_name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractError(f"INVALID_{field_name.upper()}: {value!r}; expected one of {allowed}") from exc


def _repo_file(relative: str, field_name: str, validate_files: bool) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ContractError(f"ABSOLUTE_DATA_PATH_FORBIDDEN: {field_name}={relative!r}")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ContractError(f"DATA_PATH_ESCAPES_REPOSITORY: {field_name}={relative!r}") from exc
    if validate_files and not resolved.is_file():
        raise ContractError(f"MISSING_DATA_FILE: {field_name}={relative!r}")
    return resolved


def _load_invalid_gt_quarantine(
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return (forbidden source paths, SHA-256s, official excluded frame IDs)."""

    try:
        raw = json.loads(INVALID_GT_QUARANTINE_PATH.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"INVALID_GT_QUARANTINE_UNREADABLE: {INVALID_GT_QUARANTINE_PATH}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ContractError("INVALID_GT_QUARANTINE_ROOT")
    if raw.get("schema_version") != INVALID_GT_QUARANTINE_SCHEMA:
        raise ContractError("INVALID_GT_QUARANTINE_SCHEMA")
    if raw.get("status") != "QUARANTINED":
        raise ContractError("INVALID_GT_QUARANTINE_STATUS")
    entries = raw.get("entries")
    if not isinstance(entries, list) or raw.get("entry_count") != len(entries):
        raise ContractError("INVALID_GT_QUARANTINE_COUNT")

    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    official_frame_ids: set[str] = set()
    official_count = 0
    stale_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ContractError(f"INVALID_GT_QUARANTINE_ENTRY_{index}")
        frame_id = entry.get("frame_id")
        source_path = entry.get("source_path")
        source_sha = entry.get("source_sha256")
        classification = entry.get("classification")
        if not isinstance(frame_id, str) or not frame_id:
            raise ContractError(f"INVALID_GT_QUARANTINE_FRAME_ID_{index}")
        if (
            not isinstance(source_path, str)
            or not source_path.startswith("challenge/data/01_real/")
            or Path(source_path).is_absolute()
        ):
            raise ContractError(f"INVALID_GT_QUARANTINE_SOURCE_PATH_{index}")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)
        ):
            raise ContractError(f"INVALID_GT_QUARANTINE_SHA256_{index}")
        if source_path in source_paths or source_sha in source_hashes:
            raise ContractError(f"DUPLICATE_INVALID_GT_IDENTITY_{index}")
        source_paths.add(source_path)
        source_hashes.add(source_sha)
        if classification == "STALE_DUPLICATE_INVALID":
            stale_count += 1
        elif classification in {"RED_GT_QA_EXCLUDED", "AMBER_EXCLUDE"}:
            official_count += 1
            official_frame_ids.add(frame_id)
        else:
            raise ContractError(f"INVALID_GT_QUARANTINE_CLASSIFICATION_{index}")

    if raw.get("official_eval_exclusion_count") != official_count:
        raise ContractError("INVALID_GT_OFFICIAL_EXCLUSION_COUNT")
    if raw.get("stale_duplicate_count") != stale_count:
        raise ContractError("INVALID_GT_STALE_DUPLICATE_COUNT")
    return (
        frozenset(source_paths),
        frozenset(source_hashes),
        frozenset(official_frame_ids),
    )


def _assert_label_not_quarantined(
    label_path: Path,
    frame_id: str,
    index: int,
    forbidden_source_paths: frozenset[str],
    forbidden_source_hashes: frozenset[str],
    official_excluded_frame_ids: frozenset[str],
) -> None:
    if frame_id in official_excluded_frame_ids:
        raise ContractError(
            f"QUARANTINED_GT_FRAME_ID_AT_{index}: frame_id={frame_id}"
        )
    normalized_label = label_path.relative_to(REPO_ROOT).as_posix()
    if normalized_label in forbidden_source_paths:
        raise ContractError(
            f"QUARANTINED_GT_LABEL_PATH_AT_{index}: frame_id={frame_id} "
            f"label={normalized_label}"
        )
    try:
        label_bytes = label_path.read_bytes()
        label_payload = json.loads(label_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"POSITIVE_LABEL_UNREADABLE_AT_{index}: {label_path}: {exc}") from exc
    direct_sha = hashlib.sha256(label_bytes).hexdigest()
    if direct_sha in forbidden_source_hashes:
        raise ContractError(
            f"QUARANTINED_GT_LABEL_AT_{index}: frame_id={frame_id} sha256={direct_sha}"
        )
    if isinstance(label_payload, dict):
        migration = label_payload.get("real_gt_v2_migration")
        if isinstance(migration, dict):
            migrated_source_label = migration.get("source_label")
            normalized_source_label = migrated_source_label
            if isinstance(migrated_source_label, str) and not Path(
                migrated_source_label
            ).is_absolute():
                try:
                    normalized_source_label = (
                        (REPO_ROOT / migrated_source_label)
                        .resolve()
                        .relative_to(REPO_ROOT)
                        .as_posix()
                    )
                except ValueError:
                    pass
            if normalized_source_label in forbidden_source_paths:
                raise ContractError(
                    f"QUARANTINED_GT_SOURCE_PATH_AT_{index}: frame_id={frame_id} "
                    f"source_label={normalized_source_label}"
                )
            migrated_source_sha = migration.get("source_sha256")
            if migrated_source_sha in forbidden_source_hashes:
                raise ContractError(
                    f"QUARANTINED_GT_SOURCE_AT_{index}: frame_id={frame_id} "
                    f"source_sha256={migrated_source_sha}"
                )


def _parse_item(
    raw: Any,
    kind: PopulationKind,
    index: int,
    validate_files: bool,
    forbidden_source_paths: frozenset[str],
    forbidden_source_hashes: frozenset[str],
    official_excluded_frame_ids: frozenset[str],
) -> ManifestItem:
    if not isinstance(raw, dict):
        raise ContractError(f"INVALID_ITEM_{index}: expected object")
    frame_id = raw.get("frame_id")
    image = raw.get("image")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ContractError(f"INVALID_FRAME_ID_AT_{index}")
    if not isinstance(image, str) or not image.strip():
        raise ContractError(f"INVALID_IMAGE_PATH_AT_{index}")
    _repo_file(image, f"items[{index}].image", validate_files)

    label = raw.get("label")
    if kind is PopulationKind.POSITIVE:
        if not isinstance(label, str) or not label.strip():
            raise ContractError(f"POSITIVE_LABEL_REQUIRED_AT_{index}")
        label_path = _repo_file(label, f"items[{index}].label", validate_files)
        if validate_files:
            _assert_label_not_quarantined(
                label_path,
                frame_id,
                index,
                forbidden_source_paths,
                forbidden_source_hashes,
                official_excluded_frame_ids,
            )
    elif label is not None:
        if not isinstance(label, str) or not label.strip():
            raise ContractError(f"INVALID_LABEL_PATH_AT_{index}")
        _repo_file(label, f"items[{index}].label", validate_files)

    source_set = raw.get("source_set")
    domain = raw.get("domain")
    for name, value in (("source_set", source_set), ("domain", domain)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ContractError(f"INVALID_{name.upper()}_AT_{index}")
    return ManifestItem(
        frame_id=frame_id,
        image=image,
        label=label,
        source_set=source_set,
        domain=domain,
    )


def load_population_manifest(
    path: str | Path,
    *,
    validate_files: bool = True,
) -> PopulationManifest:
    """Load and fully validate a population manifest.

    Validation is intentionally eager: a missing member is an error rather
    than a reason to shorten the denominator.  Unavailable FINAL placeholders
    are valid contract documents, but they are not evaluation-ready.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ContractError(f"MANIFEST_NOT_FOUND: {source_path}")
    try:
        raw = json.loads(source_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"MANIFEST_UNREADABLE: {source_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("INVALID_MANIFEST_ROOT: expected object")

    schema = raw.get("schema_version")
    if schema != MANIFEST_SCHEMA_VERSION:
        raise ContractError(
            f"INVALID_SCHEMA_VERSION: {schema!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    population_id = _enum(PopulationId, raw.get("population_id"), "population_id")
    kind = _enum(PopulationKind, raw.get("kind"), "kind")
    role = _enum(PopulationRole, raw.get("role"), "role")
    status = _enum(MembershipStatus, raw.get("membership_status"), "membership_status")

    expectation = EXPECTED_POPULATIONS[population_id]
    if kind is not expectation.kind:
        raise ContractError(
            f"KIND_MISMATCH: {population_id.value} must be {expectation.kind.value}, got {kind.value}"
        )
    if role is not expectation.role:
        raise ContractError(
            f"ROLE_MISMATCH: {population_id.value} must be {expectation.role.value}, got {role.value}"
        )

    frozen = raw.get("frozen")
    if not isinstance(frozen, bool):
        raise ContractError("INVALID_FROZEN_FLAG")
    expected_count = raw.get("expected_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 0:
        raise ContractError("INVALID_EXPECTED_COUNT")
    if expected_count != expectation.count:
        raise ContractError(
            f"REGISTERED_COUNT_MISMATCH: {population_id.value} expected {expectation.count}, "
            f"manifest declares {expected_count}"
        )

    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise ContractError("INVALID_ITEMS: expected list")
    forbidden_source_paths: frozenset[str] = frozenset()
    forbidden_source_hashes: frozenset[str] = frozenset()
    official_excluded_frame_ids: frozenset[str] = frozenset()
    if validate_files and kind is PopulationKind.POSITIVE and status is MembershipStatus.AVAILABLE:
        (
            forbidden_source_paths,
            forbidden_source_hashes,
            official_excluded_frame_ids,
        ) = _load_invalid_gt_quarantine()
    items = tuple(
        _parse_item(
            item,
            kind,
            i,
            validate_files,
            forbidden_source_paths,
            forbidden_source_hashes,
            official_excluded_frame_ids,
        )
        for i, item in enumerate(raw_items)
    )
    ids = [item.frame_id for item in items]
    duplicate_ids = sorted(frame_id for frame_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ContractError(f"DUPLICATE_FRAME_IDS: {duplicate_ids[:10]}")
    resolved_images = [str((REPO_ROOT / item.image).resolve()) for item in items]
    duplicate_images = sorted(
        image for image, count in Counter(resolved_images).items() if count > 1
    )
    if duplicate_images:
        raise ContractError(f"DUPLICATE_IMAGE_PATHS: {duplicate_images[:10]}")
    if kind is PopulationKind.POSITIVE:
        resolved_labels = [
            str((REPO_ROOT / item.label).resolve())
            for item in items
            if item.label is not None
        ]
        duplicate_labels = sorted(
            label for label, count in Counter(resolved_labels).items() if count > 1
        )
        if duplicate_labels:
            raise ContractError(f"DUPLICATE_LABEL_PATHS: {duplicate_labels[:10]}")
    if len(items) != expected_count:
        raise ContractError(
            f"COUNT_MISMATCH: {population_id.value} declares {expected_count}, has {len(items)} items"
        )

    declared_hash = raw.get("membership_sha256")
    unavailable_reason = raw.get("unavailable_reason")
    if status is MembershipStatus.UNAVAILABLE:
        if population_id not in {PopulationId.FINAL_POS, PopulationId.FINAL_NEG}:
            raise ContractError("ONLY_FINAL_MEMBERSHIP_MAY_BE_UNAVAILABLE")
        if expected_count != 0 or items:
            raise ContractError("UNAVAILABLE_MEMBERSHIP_MUST_HAVE_ZERO_ITEMS")
        if frozen:
            raise ContractError("UNAVAILABLE_MEMBERSHIP_CANNOT_BE_FROZEN")
        if declared_hash is not None:
            raise ContractError("UNAVAILABLE_MEMBERSHIP_HASH_MUST_BE_NULL")
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            raise ContractError("UNAVAILABLE_REASON_REQUIRED")
    else:
        if expected_count == 0:
            raise ContractError("AVAILABLE_MEMBERSHIP_CANNOT_BE_EMPTY")
        if not frozen:
            raise ContractError("AVAILABLE_MEMBERSHIP_MUST_BE_FROZEN")
        if not isinstance(declared_hash, str) or len(declared_hash) != 64:
            raise ContractError("INVALID_MEMBERSHIP_SHA256")
        actual_hash = membership_sha256(items)
        if actual_hash != declared_hash:
            raise ContractError(
                f"MEMBERSHIP_HASH_MISMATCH: declared {declared_hash}, actual {actual_hash}"
            )
        unavailable_reason = None

    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ContractError("INVALID_PROVENANCE")
    return PopulationManifest(
        schema_version=schema,
        population_id=population_id,
        kind=kind,
        role=role,
        membership_status=status,
        frozen=frozen,
        expected_count=expected_count,
        membership_sha256=declared_hash,
        items=items,
        source_path=source_path,
        unavailable_reason=unavailable_reason,
        provenance=provenance,
    )


def load_repo_population(
    population_id: PopulationId | str,
    *,
    validate_files: bool = True,
) -> PopulationManifest:
    return load_population_manifest(manifest_path(population_id), validate_files=validate_files)


def validate_registered_membership(
    manifest: PopulationManifest,
    *,
    validate_files: bool = True,
) -> None:
    """Require a caller-supplied manifest to equal the repo-registered one."""

    if not isinstance(manifest, PopulationManifest):
        raise TypeError("manifest must be PopulationManifest")
    registered = load_repo_population(manifest.population_id, validate_files=validate_files)
    identity = (
        "kind",
        "role",
        "membership_status",
        "frozen",
        "expected_count",
        "membership_sha256",
        "items",
    )
    mismatched = [name for name in identity if getattr(manifest, name) != getattr(registered, name)]
    if mismatched:
        raise ContractError(
            f"UNREGISTERED_MEMBERSHIP_FOR_{manifest.population_id.value}: {mismatched}"
        )


def validate_common_dev_membership(
    dev_pos140: PopulationManifest | None = None,
    common_pos128: PopulationManifest | None = None,
    *,
    validate_files: bool = True,
) -> None:
    """Prove COMMON_DEV_POS128 is exactly DEV_POS140 minus the recorded 12."""

    dev_pos140 = dev_pos140 or load_repo_population(
        PopulationId.DEV_POS140, validate_files=validate_files
    )
    common_pos128 = common_pos128 or load_repo_population(
        PopulationId.COMMON_DEV_POS128, validate_files=validate_files
    )
    if dev_pos140.population_id is not PopulationId.DEV_POS140:
        raise ContractError("COMMON_RELATION_SOURCE_MUST_BE_DEV_POS140")
    if common_pos128.population_id is not PopulationId.COMMON_DEV_POS128:
        raise ContractError("COMMON_RELATION_TARGET_MUST_BE_COMMON_DEV_POS128")

    excluded = common_pos128.provenance.get("excluded_frame_ids")
    if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
        raise ContractError("COMMON_EXCLUDED_FRAME_IDS_REQUIRED")
    if len(excluded) != 12 or len(set(excluded)) != 12:
        raise ContractError("COMMON_EXCLUSION_MUST_CONTAIN_EXACTLY_12_UNIQUE_IDS")
    dev_ids = set(dev_pos140.frame_ids)
    common_ids = set(common_pos128.frame_ids)
    excluded_ids = set(excluded)
    if not excluded_ids <= dev_ids:
        raise ContractError("COMMON_EXCLUSION_NOT_SUBSET_OF_DEV_POS140")
    if common_ids != dev_ids - excluded_ids:
        raise ContractError("COMMON_DEV_POS128_IS_NOT_EXACT_DEV140_MINUS_LEAK12")
    expected_items = tuple(
        item for item in dev_pos140.items if item.frame_id not in excluded_ids
    )
    if tuple(item.frame_id for item in expected_items) != common_pos128.frame_ids:
        raise ContractError("COMMON_DEV_POS128_ORDER_MISMATCH")
    if common_pos128.items != expected_items:
        raise ContractError("COMMON_DEV_POS128_ITEM_IDENTITY_MISMATCH")


def _pair_digest(positive: PopulationManifest, negative: PopulationManifest) -> str:
    text = (
        f"{positive.population_id.value}:{positive.membership_sha256}\n"
        f"{negative.population_id.value}:{negative.membership_sha256}\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_evaluation_pair(
    positive: PopulationManifest,
    negative: PopulationManifest,
    role: PopulationRole | str,
    *,
    allow_unavailable_final: bool = False,
) -> EvaluationPopulationPair:
    """Validate the only paper-facing positive/negative pair for a role."""

    role = _enum(PopulationRole, role, "population_role")
    if positive.kind is not PopulationKind.POSITIVE or negative.kind is not PopulationKind.NEGATIVE:
        raise ContractError("POSITIVE_NEGATIVE_KIND_ORDER_REQUIRED")
    if positive.role is not role or negative.role is not role:
        raise ContractError("CLI_ROLE_DOES_NOT_MATCH_MANIFEST_ROLE")

    if role is PopulationRole.DEV:
        expected_pair = (PopulationId.COMMON_DEV_POS128, PopulationId.DEV_NEG2689)
        actual_pair = (positive.population_id, negative.population_id)
        if actual_pair != expected_pair:
            raise ContractError(
                "DEV_COMPARISON_REQUIRES_COMMON_DEV_POS128_AND_DEV_NEG2689; "
                f"got {actual_pair[0].value}+{actual_pair[1].value}"
            )
        if not positive.available or not negative.available:
            raise ContractError("DEV_MEMBERSHIP_UNAVAILABLE")
        return EvaluationPopulationPair(
            positive=positive,
            negative=negative,
            role=role,
            ready=True,
            blocked_reason=None,
            pair_sha256=_pair_digest(positive, negative),
        )

    expected_pair = (PopulationId.FINAL_POS, PopulationId.FINAL_NEG)
    actual_pair = (positive.population_id, negative.population_id)
    if actual_pair != expected_pair:
        raise ContractError(
            "FINAL_COMPARISON_REQUIRES_FINAL_POS_AND_FINAL_NEG; "
            f"got {actual_pair[0].value}+{actual_pair[1].value}"
        )
    if not positive.available or not negative.available:
        if not allow_unavailable_final:
            raise ContractError("FINAL_MEMBERSHIP_UNAVAILABLE")
        return EvaluationPopulationPair(
            positive=positive,
            negative=negative,
            role=role,
            ready=False,
            blocked_reason="FINAL_MEMBERSHIP_UNAVAILABLE",
            pair_sha256=None,
        )
    if not positive.frozen or not negative.frozen:
        raise ContractError("FINAL_MEMBERSHIP_NOT_FROZEN")
    return EvaluationPopulationPair(
        positive=positive,
        negative=negative,
        role=role,
        ready=True,
        blocked_reason=None,
        pair_sha256=_pair_digest(positive, negative),
    )


def validate_repo_population_contract(*, validate_files: bool = True) -> dict[PopulationId, PopulationManifest]:
    """Validate all five repo manifests and their cross-population relation."""

    manifests = {
        population_id: load_repo_population(population_id, validate_files=validate_files)
        for population_id in PopulationId
    }
    validate_common_dev_membership(
        manifests[PopulationId.DEV_POS140],
        manifests[PopulationId.COMMON_DEV_POS128],
        validate_files=validate_files,
    )
    return manifests


__all__ = [
    "ContractError",
    "EvaluationPopulationPair",
    "EXPECTED_POPULATIONS",
    "MANIFEST_DIR",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestItem",
    "MembershipStatus",
    "PopulationId",
    "PopulationKind",
    "PopulationManifest",
    "PopulationRole",
    "load_population_manifest",
    "load_repo_population",
    "manifest_path",
    "membership_sha256",
    "validate_common_dev_membership",
    "validate_evaluation_pair",
    "validate_registered_membership",
    "validate_repo_population_contract",
]
