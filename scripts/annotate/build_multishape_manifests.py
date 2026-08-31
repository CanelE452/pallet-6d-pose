#!/usr/bin/env python3
"""Build the immutable object-aware DEV manifests and FINAL placeholders.

The builder is intentionally append-only.  It verifies the byte hashes of the
five historical population manifests, never writes those files, and refuses to
replace a non-matching object-aware manifest.  ``--check`` performs the same
derivation without writing anything.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.real_dataset_contract import (  # noqa: E402
    MANIFEST_DIR,
    MANIFEST_SCHEMA_VERSION,
    PLASTIC_OBJECT_TYPE,
    POPULATION_OBJECT_TYPES,
    WOOD_OBJECT_TYPE,
    ManifestItem,
    PopulationId,
    load_repo_population,
    membership_sha256,
    validate_repo_population_contract,
)


WOOD_AUDIT_JSON = REPO_ROOT / "challenge/real_gt_v2/wood_audit/WOOD_GT_AUDIT.json"
WOOD_AUDIT_CSV = REPO_ROOT / "challenge/real_gt_v2/wood_audit/WOOD_GT_PER_FRAME.csv"
WOOD_MIGRATED_ROOT = REPO_ROOT / "challenge/real_gt_v2/migrated_gt_wood"

# These are the historical files protected by this builder.  Membership hashes
# alone are insufficient here: provenance and formatting bytes must also stay
# unchanged.
LEGACY_MANIFEST_BYTE_SHA256: Mapping[str, str] = {
    "DEV_POS140.json": "dfb7ed4f54fc17fb5a007b430bf88691fe1a418af155e1a3cf5c4b33806f0fd3",
    "COMMON_DEV_POS128.json": "06b0af912b65cdf4b7d2297ef5fbab1c75daa534b2a2ef73a81fd7f53fa40735",
    "DEV_NEG2689.json": "37ac706dd3fef77f110537e1033ae82fb152413ffc166dd6b03388143635f822",
    "FINAL_POS.json": "0a56774e6190d2d1cb394040f74f29ee8d5177638e5c22aad5ed5ac95138c37f",
    "FINAL_NEG.json": "96b3840698bb9def55f4e0c98df0604d74d80ce32041a8bb480da74a850f7f24",
}

NEW_MANIFEST_IDS = (
    PopulationId.DEV_PLASTIC_POS140,
    PopulationId.COMMON_DEV_PLASTIC_POS128,
    PopulationId.DEV_WOOD_POS45,
    PopulationId.COMMON_DEV_MULTISHAPE_POS,
    PopulationId.FINAL_PLASTIC_POS,
    PopulationId.FINAL_WOOD_POS,
    PopulationId.FINAL_ALL_POS,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {path}") from exc


def assert_legacy_manifests_unchanged() -> None:
    for filename, expected_sha256 in LEGACY_MANIFEST_BYTE_SHA256.items():
        path = MANIFEST_DIR / filename
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(
                f"historical manifest unreadable: {path}: {exc}"
            ) from exc
        if actual != expected_sha256:
            raise RuntimeError(
                f"historical manifest byte drift: {filename}: "
                f"expected {expected_sha256}, got {actual}"
            )


def _object_item(
    *,
    frame_id: str,
    object_type: str,
    session_id: str,
    image_path: str,
    gt_v2_path: str,
    source_population: PopulationId,
    population_role: str = "DEV",
    domain: str | None = None,
) -> dict[str, str]:
    item = {
        "frame_id": frame_id,
        "object_type": object_type,
        "session_id": session_id,
        "image_path": image_path,
        "gt_v2_path": gt_v2_path,
        "population_role": population_role,
        "source_population": source_population.value,
    }
    if domain is not None:
        item["domain"] = domain
    return item


def _manifest_item(raw: Mapping[str, str]) -> ManifestItem:
    return ManifestItem(
        frame_id=raw["frame_id"],
        image=raw["image_path"],
        label=raw["gt_v2_path"],
        source_set=raw["session_id"],
        domain=raw.get("domain"),
        object_type=raw["object_type"],
        session_id=raw["session_id"],
        population_role=raw["population_role"],
        source_population=raw["source_population"],
    )


def _available_manifest(
    population_id: PopulationId,
    items: list[dict[str, str]],
    provenance: Mapping[str, Any],
    *,
    role: str = "DEV",
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "population_id": population_id.value,
        "object_types": list(POPULATION_OBJECT_TYPES[population_id]),
        "kind": "POSITIVE",
        "role": role,
        "membership_status": "AVAILABLE",
        "frozen": True,
        "expected_count": len(items),
        "membership_sha256": membership_sha256(_manifest_item(item) for item in items),
        "items": items,
        "provenance": dict(provenance),
    }


def _final_placeholder(
    population_id: PopulationId,
    object_scope: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "population_id": population_id.value,
        "object_types": list(POPULATION_OBJECT_TYPES[population_id]),
        "kind": "POSITIVE",
        "role": "FINAL",
        "membership_status": "UNAVAILABLE",
        "frozen": False,
        "expected_count": 0,
        "membership_sha256": None,
        "items": [],
        "unavailable_reason": (
            f"Untouched frozen FINAL {object_scope} positive membership has not "
            "been captured and frozen."
        ),
        "provenance": {
            "count_zero_semantics": (
                "membership unavailable; not evidence that no real positive data exists"
            ),
            "development_populations_must_not_be_reused": True,
        },
    }


def _plastic_items(
    legacy_id: PopulationId,
    explicit_id: PopulationId,
) -> tuple[list[dict[str, str]], str]:
    legacy = load_repo_population(legacy_id, validate_files=True)
    items: list[dict[str, str]] = []
    for index, item in enumerate(legacy.items):
        if not item.source_set:
            raise RuntimeError(f"{legacy_id.value}[{index}] has no source_set")
        if not item.label:
            raise RuntimeError(f"{legacy_id.value}[{index}] has no GT v2 path")
        items.append(
            _object_item(
                frame_id=item.frame_id,
                object_type=PLASTIC_OBJECT_TYPE,
                session_id=item.source_set,
                image_path=item.image,
                gt_v2_path=item.label,
                source_population=explicit_id,
                domain=item.domain,
            )
        )
    return items, legacy.membership_sha256 or ""


def _wood_items() -> tuple[list[dict[str, str]], dict[str, Any]]:
    audit = _json(WOOD_AUDIT_JSON)
    if audit.get("status") != "PASS" or audit.get("population_id") != "DEV_WOOD_POS45":
        raise RuntimeError("wood audit is not a PASS for DEV_WOOD_POS45")
    if audit.get("count") != 45:
        raise RuntimeError(f"wood audit count is not 45: {audit.get('count')!r}")
    checks = audit.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RuntimeError("wood audit does not have an all-PASS checks block")

    try:
        with WOOD_AUDIT_CSV.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError(
            f"wood audit CSV unreadable: {WOOD_AUDIT_CSV}: {exc}"
        ) from exc
    if len(rows) != 45:
        raise RuntimeError(f"wood audit CSV has {len(rows)} rows, expected 45")
    if [row["frame_id"] for row in rows] != sorted(row["frame_id"] for row in rows):
        raise RuntimeError("wood audit CSV is not in deterministic frame_id order")

    items: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        frame_id = row["frame_id"]
        session_id = row["session_id"]
        if frame_id != f"{session_id}:{row['source_frame_id']}":
            raise RuntimeError(
                f"wood row {index} has an unqualified/mismatched frame_id"
            )
        if (
            row["population"] != PopulationId.DEV_WOOD_POS45.value
            or row["population_role"] != "DEV"
            or row["object_type"] != WOOD_OBJECT_TYPE
        ):
            raise RuntimeError(f"wood row {index} has invalid population metadata")

        source_label = Path(row["label_path"])
        try:
            source_tail = source_label.relative_to("challenge/data/01_real")
        except ValueError as exc:
            raise RuntimeError(
                f"wood row {index} source label is outside 01_real"
            ) from exc
        migrated = WOOD_MIGRATED_ROOT / source_tail
        migrated_payload = _json(migrated)
        if (
            migrated_payload.get("schema_version") != "real_pallet_gt_v2"
            or migrated_payload.get("object_type") != WOOD_OBJECT_TYPE
            or migrated_payload.get("population_role") != "DEV"
            or migrated_payload.get("real_gt_v2_migration", {}).get("source_label")
            != source_label.as_posix()
        ):
            raise RuntimeError(f"wood migrated GT metadata mismatch: {migrated}")

        image = REPO_ROOT / row["image_path"]
        if not image.is_file():
            raise RuntimeError(f"wood source image missing: {image}")
        items.append(
            _object_item(
                frame_id=frame_id,
                object_type=WOOD_OBJECT_TYPE,
                session_id=session_id,
                image_path=row["image_path"],
                gt_v2_path=_repo_relative(migrated),
                source_population=PopulationId.DEV_WOOD_POS45,
                population_role="CROSS_SHAPE_DEV",
            )
        )

    session_counts = Counter(item["session_id"] for item in items)
    if session_counts != Counter({"wood_183705": 25, "wood_184309": 20}):
        raise RuntimeError(f"unexpected wood session counts: {dict(session_counts)}")
    if len({item["frame_id"] for item in items}) != 45:
        raise RuntimeError("wood qualified frame IDs are not unique")
    return items, audit


def build_payloads() -> dict[PopulationId, dict[str, Any]]:
    """Derive all new manifests without writing to the repository."""

    assert_legacy_manifests_unchanged()
    plastic140, legacy140_hash = _plastic_items(
        PopulationId.DEV_POS140, PopulationId.DEV_PLASTIC_POS140
    )
    plastic128, legacy128_hash = _plastic_items(
        PopulationId.COMMON_DEV_POS128,
        PopulationId.COMMON_DEV_PLASTIC_POS128,
    )
    wood45, wood_audit = _wood_items()
    multishape173 = [dict(item) for item in plastic128 + wood45]

    if (len(plastic140), len(plastic128), len(wood45), len(multishape173)) != (
        140,
        128,
        45,
        173,
    ):
        raise RuntimeError("derived manifest counts are not 140/128/45/173")
    if len({item["frame_id"] for item in multishape173}) != 173:
        raise RuntimeError("multishape ordered union contains duplicate frame IDs")

    audit_sha = _sha256_bytes(WOOD_AUDIT_JSON.read_bytes())
    payloads = {
        PopulationId.DEV_PLASTIC_POS140: _available_manifest(
            PopulationId.DEV_PLASTIC_POS140,
            plastic140,
            {
                "alias_of_population": PopulationId.DEV_POS140.value,
                "legacy_membership_sha256": legacy140_hash,
                "object_type": PLASTIC_OBJECT_TYPE,
            },
        ),
        PopulationId.COMMON_DEV_PLASTIC_POS128: _available_manifest(
            PopulationId.COMMON_DEV_PLASTIC_POS128,
            plastic128,
            {
                "alias_of_population": PopulationId.COMMON_DEV_POS128.value,
                "legacy_membership_sha256": legacy128_hash,
                "object_type": PLASTIC_OBJECT_TYPE,
            },
        ),
        PopulationId.DEV_WOOD_POS45: _available_manifest(
            PopulationId.DEV_WOOD_POS45,
            wood45,
            {
                "role": "CROSS_SHAPE_DEV",
                "source_audit": _repo_relative(WOOD_AUDIT_JSON),
                "source_audit_sha256": audit_sha,
                "source_audit_membership_identity_sha256": wood_audit[
                    "membership_identity_sha256"
                ],
                "session_counts": dict(Counter(item["session_id"] for item in wood45)),
                "symmetry_status": wood_audit["symmetry_status"],
                "selector_status": wood_audit["selector_status"],
                "previously_evaluated": True,
                "final_eligible": False,
            },
            role="CROSS_SHAPE_DEV",
        ),
        PopulationId.COMMON_DEV_MULTISHAPE_POS: _available_manifest(
            PopulationId.COMMON_DEV_MULTISHAPE_POS,
            multishape173,
            {
                "ordered_union_of": [
                    PopulationId.COMMON_DEV_PLASTIC_POS128.value,
                    PopulationId.DEV_WOOD_POS45.value,
                ],
                "component_counts": {
                    PopulationId.COMMON_DEV_PLASTIC_POS128.value: 128,
                    PopulationId.DEV_WOOD_POS45.value: 45,
                },
                "order_contract": "plastic_128_then_wood_45",
            },
        ),
        PopulationId.FINAL_PLASTIC_POS: _final_placeholder(
            PopulationId.FINAL_PLASTIC_POS, "plastic"
        ),
        PopulationId.FINAL_WOOD_POS: _final_placeholder(
            PopulationId.FINAL_WOOD_POS, "wood"
        ),
        PopulationId.FINAL_ALL_POS: _final_placeholder(
            PopulationId.FINAL_ALL_POS, "multishape"
        ),
    }
    if tuple(payloads) != NEW_MANIFEST_IDS:
        raise RuntimeError("internal manifest ordering mismatch")
    return payloads


def _manifest_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def materialize(
    payloads: Mapping[PopulationId, Mapping[str, Any]],
    *,
    output_dir: Path = MANIFEST_DIR,
    check_only: bool = False,
) -> tuple[Path, ...]:
    """Create missing files exclusively, or prove existing files are exact."""

    output_dir = output_dir.resolve()
    expected = {
        output_dir / f"{population_id.value}.json": _manifest_text(payload)
        for population_id, payload in payloads.items()
    }
    for path, text in expected.items():
        if path.exists():
            try:
                existing = path.read_text("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise RuntimeError(
                    f"existing manifest unreadable: {path}: {exc}"
                ) from exc
            if existing != text:
                raise RuntimeError(f"refusing to replace non-matching manifest: {path}")
        elif check_only:
            raise RuntimeError(f"expected manifest is missing: {path}")

    if check_only:
        return tuple(expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in expected.items():
        if not path.exists():
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(text)
            except FileExistsError as exc:
                raise RuntimeError(f"manifest appeared concurrently: {path}") from exc
    return tuple(expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that all derived manifests already exist exactly; write nothing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="destination directory (default: registered repo manifest directory)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads = build_payloads()
    paths = materialize(payloads, output_dir=args.output_dir, check_only=args.check)
    if args.output_dir.resolve() == MANIFEST_DIR.resolve():
        validate_repo_population_contract(validate_files=True)
    action = "checked" if args.check else "materialized"
    print(f"{action} {len(paths)} object-aware manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
