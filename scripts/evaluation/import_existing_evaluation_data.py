#!/usr/bin/env python3
"""Audit and non-destructively import existing real evaluation data.

By default the command builds a complete staging workspace, verifies all source hashes,
sizes, mtimes and file counts again, and only then atomically publishes it.
An existing workspace is never overwritten.  ``--audit-only`` performs the
same source/membership audit without creating the destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import from tests.
    from .eval_workspace import (
        FRAME_COLUMNS,
        PROVENANCE_COLUMNS,
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        bool_text,
        copy2_verified,
        infer_annotation_tags,
        normalize_object_type,
        safe_component,
        scaffold_workspace,
        sha256_file,
        workspace_relative,
        write_manifest_views,
        write_reports,
    )
except ImportError:  # Direct script execution.
    from eval_workspace import (  # type: ignore[no-redef]
        FRAME_COLUMNS,
        PROVENANCE_COLUMNS,
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        bool_text,
        copy2_verified,
        infer_annotation_tags,
        normalize_object_type,
        safe_component,
        scaffold_workspace,
        sha256_file,
        workspace_relative,
        write_manifest_views,
        write_reports,
    )


MANIFEST_DIR = Path("challenge/real_gt_v2/manifests")
GEOMETRY_REGISTRY = Path("challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json")
LEGACY_ROOTS = (
    (3, "eval_canonical_legacy", Path("challenge/data/01_real/eval_canonical")),
    (4, "manual_gt_legacy", Path("challenge/data/01_real/manual_gt")),
)
SOURCE_COUNT_SCOPES = (
    Path("challenge/real_gt_v2/manifests"),
    Path("challenge/real_gt_v2/migrated_gt"),
    Path("challenge/real_gt_v2/migrated_gt_wood"),
    Path("challenge/data/01_real/eval_canonical"),
    Path("challenge/data/01_real/manual_gt"),
    Path("data/pallet/raw_data/negative_real_20260823/rgb"),
)


@dataclass(frozen=True)
class ExpectedCounts:
    plastic: int = 140
    controlled_plastic: int = 128
    plastic_excluded: int = 12
    wood: int = 45
    wood_sessions: tuple[tuple[str, int], ...] = (
        ("wood_183705", 25),
        ("wood_184309", 20),
    )
    multishape: int = 173
    negative: int = 2689


DEFAULT_EXPECTED = ExpectedCounts()


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    kind: str


@dataclass
class ImportRecord:
    source_priority: int
    source_dataset: str
    population_role: str
    paper_subset: str
    controlled_eval_eligible: bool
    cross_shape_eval_eligible: bool
    exclusion_reason: str
    session_id: str
    source_frame_id: str
    object_type: str
    lighting: str
    source_image: Path
    source_annotation: Path | None
    image_sha256: str
    annotation_sha256: str
    occlusion: str = "unknown"
    truncation: str = "unknown"
    active_frame_id: str = ""
    destination_stem: str = ""
    storage_mode: str = "independent_copy"
    notes: str = ""


@dataclass
class ImportPlan:
    repo_root: Path
    records: list[ImportRecord]
    provenance: list[dict[str, str]]
    snapshots: dict[Path, FileSnapshot]
    scope_counts_before: dict[str, int]
    unresolved_legacy: list[str]
    duplicate_positive_sha_groups: int
    duplicate_negative_sha_groups: int
    legacy_annotation_paths: int
    legacy_unique_active: int
    membership_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def _manifest_image(item: Mapping[str, Any]) -> str:
    value = item.get("image_path") or item.get("image")
    if not value:
        raise WorkspaceError(f"manifest item has no image path: {item}")
    return str(value)


def _manifest_annotation(item: Mapping[str, Any]) -> str | None:
    value = item.get("gt_v2_path") or item.get("label")
    return str(value) if value else None


def _membership_identity(item: Mapping[str, Any]) -> str:
    return Path(_manifest_image(item)).as_posix()


def validate_membership_contract(
    *,
    plastic_items: Sequence[Mapping[str, Any]],
    controlled_items: Sequence[Mapping[str, Any]],
    wood_items: Sequence[Mapping[str, Any]],
    negative_items: Sequence[Mapping[str, Any]],
    multishape_items: Sequence[Mapping[str, Any]] | None = None,
    expected: ExpectedCounts = DEFAULT_EXPECTED,
) -> None:
    """Fail closed on the frozen audited populations before any destination I/O."""

    actual_counts = {
        "plastic": len(plastic_items),
        "controlled_plastic": len(controlled_items),
        "wood": len(wood_items),
        "negative": len(negative_items),
    }
    expected_counts = {
        "plastic": expected.plastic,
        "controlled_plastic": expected.controlled_plastic,
        "wood": expected.wood,
        "negative": expected.negative,
    }
    mismatches = {
        name: (actual_counts[name], wanted)
        for name, wanted in expected_counts.items()
        if actual_counts[name] != wanted
    }
    if mismatches:
        detail = ", ".join(
            f"{name}: actual={actual}, expected={wanted}"
            for name, (actual, wanted) in mismatches.items()
        )
        raise WorkspaceError(f"audited membership count mismatch; import aborted: {detail}")

    plastic = {_membership_identity(item) for item in plastic_items}
    controlled = {_membership_identity(item) for item in controlled_items}
    if len(plastic) != len(plastic_items):
        raise WorkspaceError("DEV plastic manifest has duplicate image membership paths")
    if len(controlled) != len(controlled_items):
        raise WorkspaceError("controlled plastic manifest has duplicate image membership paths")
    if not controlled.issubset(plastic):
        extra = sorted(controlled - plastic)
        raise WorkspaceError(f"controlled plastic is not a subset of DEV plastic: {extra[:5]}")
    difference = plastic - controlled
    if len(difference) != expected.plastic_excluded:
        raise WorkspaceError(
            "plastic controlled exclusion mismatch: "
            f"actual={len(difference)}, expected={expected.plastic_excluded}"
        )

    expected_sessions = dict(expected.wood_sessions)
    actual_sessions = Counter(str(item.get("session_id", "")) for item in wood_items)
    if dict(actual_sessions) != expected_sessions:
        raise WorkspaceError(
            f"wood session membership mismatch: actual={dict(actual_sessions)}, "
            f"expected={expected_sessions}"
        )

    wood = {_membership_identity(item) for item in wood_items}
    if len(wood) != len(wood_items):
        raise WorkspaceError("DEV wood manifest has duplicate image membership paths")
    if plastic & wood:
        raise WorkspaceError("plastic and wood manifest paths overlap")

    if multishape_items is not None:
        multishape = {_membership_identity(item) for item in multishape_items}
        expected_union = controlled | wood
        if len(multishape_items) != expected.multishape or multishape != expected_union:
            raise WorkspaceError(
                "controlled multi-shape membership is not exactly controlled plastic + wood: "
                f"actual_items={len(multishape_items)}, actual_unique={len(multishape)}, "
                f"expected={expected.multishape}"
            )


def _canonical_manifest_record(item: Mapping[str, Any]) -> dict[str, str]:
    """Reproduce the frozen population membership-hash record exactly."""

    frame_id = str(item.get("frame_id", ""))
    image = _manifest_image(item)
    annotation = _manifest_annotation(item)
    if item.get("object_type") is not None:
        record = {
            "frame_id": frame_id,
            "object_type": str(item.get("object_type", "")),
            "session_id": str(item.get("session_id") or item.get("source_set") or ""),
            "image_path": image,
        }
        if annotation is not None:
            record["gt_v2_path"] = annotation
        for field in ("population_role", "source_population", "domain"):
            if item.get(field) is not None:
                record[field] = str(item[field])
        return record
    record = {"frame_id": frame_id, "image": image}
    if annotation is not None:
        record["label"] = annotation
    for field in ("source_set", "domain"):
        if item.get(field) is not None:
            record[field] = str(item[field])
    return record


def manifest_membership_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        line = json.dumps(
            _canonical_manifest_record(item),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_manifest_relative_path(repo_root: Path, value: str, field: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WorkspaceError(f"manifest {field} must be a safe repository-relative path: {value!r}")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"manifest {field} escapes repository: {value!r}") from exc
    return str(resolved)


def _read_manifest(repo_root: Path, name: str, expected_count: int) -> dict[str, Any]:
    path = repo_root / MANIFEST_DIR / f"{name}.json"
    if not path.is_file():
        raise WorkspaceError(f"required manifest missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkspaceError(f"invalid manifest JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkspaceError(f"manifest root must be an object: {path}")
    if document.get("schema_version") != "pallet_pose_population_manifest_v1":
        raise WorkspaceError(f"invalid manifest schema_version: {path}")
    if document.get("population_id") != name:
        raise WorkspaceError(
            f"manifest population_id mismatch: actual={document.get('population_id')!r}, expected={name!r}"
        )
    if document.get("membership_status") != "AVAILABLE" or document.get("frozen") is not True:
        raise WorkspaceError(f"import source manifest must be frozen and AVAILABLE: {path}")
    items = document.get("items")
    if not isinstance(items, list):
        raise WorkspaceError(f"manifest items must be a list: {path}")
    header_count = document.get("expected_count")
    if isinstance(header_count, bool) or not isinstance(header_count, int):
        raise WorkspaceError(f"manifest expected_count must be an integer: {path}")
    if header_count != expected_count or len(items) != expected_count:
        raise WorkspaceError(
            f"{name} count mismatch; import aborted: "
            f"header={header_count}, actual={len(items)}, expected={expected_count}"
        )
    declared_hash = document.get("membership_sha256")
    if not isinstance(declared_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
        raise WorkspaceError(f"invalid membership_sha256: {path}")

    frame_ids: list[str] = []
    image_paths: list[str] = []
    annotation_paths: list[str] = []
    positive = document.get("kind") == "POSITIVE"
    if document.get("kind") not in {"POSITIVE", "NEGATIVE"}:
        raise WorkspaceError(f"invalid manifest kind: {path}")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise WorkspaceError(f"{name}.items[{index}] must be an object")
        frame_id = item.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise WorkspaceError(f"{name}.items[{index}] has invalid frame_id")
        frame_ids.append(frame_id)
        legacy_image = item.get("image")
        aware_image = item.get("image_path")
        if legacy_image is not None and aware_image is not None and legacy_image != aware_image:
            raise WorkspaceError(f"{name}.items[{index}] has conflicting image path aliases")
        raw_image = aware_image if aware_image is not None else legacy_image
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise WorkspaceError(f"{name}.items[{index}] has invalid image path")
        image_paths.append(
            _validated_manifest_relative_path(
                repo_root, raw_image, f"{name}.items[{index}].image"
            )
        )
        legacy_annotation = item.get("label")
        aware_annotation = item.get("gt_v2_path")
        if (
            legacy_annotation is not None
            and aware_annotation is not None
            and legacy_annotation != aware_annotation
        ):
            raise WorkspaceError(f"{name}.items[{index}] has conflicting label path aliases")
        annotation = aware_annotation if aware_annotation is not None else legacy_annotation
        if positive:
            if not isinstance(annotation, str) or not annotation.strip():
                raise WorkspaceError(f"{name}.items[{index}] positive member has no label")
            annotation_paths.append(
                _validated_manifest_relative_path(
                    repo_root, annotation, f"{name}.items[{index}].label"
                )
            )
        elif annotation is not None:
            raise WorkspaceError(f"{name}.items[{index}] negative member must not have a label")
    duplicate_ids = sorted(key for key, count in Counter(frame_ids).items() if count > 1)
    duplicate_images = sorted(key for key, count in Counter(image_paths).items() if count > 1)
    duplicate_annotations = sorted(
        key for key, count in Counter(annotation_paths).items() if count > 1
    )
    if duplicate_ids:
        raise WorkspaceError(f"{name} duplicate frame_id membership: {duplicate_ids[:10]}")
    if duplicate_images:
        raise WorkspaceError(f"{name} duplicate image path membership: {duplicate_images[:10]}")
    if duplicate_annotations:
        raise WorkspaceError(
            f"{name} duplicate annotation path membership: {duplicate_annotations[:10]}"
        )
    actual_hash = manifest_membership_sha256(items)
    if actual_hash != declared_hash:
        raise WorkspaceError(
            f"{name} membership SHA mismatch; import aborted: "
            f"declared={declared_hash}, actual={actual_hash}"
        )
    return document


def _repo_source_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"source path escapes repository: {value}") from exc
    return path


def _repo_relative(repo_root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _snapshot(
    path: Path,
    kind: str,
    snapshots: dict[Path, FileSnapshot],
    hash_cache: dict[Path, str],
) -> FileSnapshot:
    path = path.resolve()
    if path in snapshots:
        return snapshots[path]
    if not path.is_file():
        raise WorkspaceError(f"source {kind} missing: {path}")
    stat = path.stat()
    digest = hash_cache.get(path)
    if digest is None:
        digest = sha256_file(path)
        hash_cache[path] = digest
    value = FileSnapshot(path=path, sha256=digest, size=stat.st_size, mtime_ns=stat.st_mtime_ns, kind=kind)
    snapshots[path] = value
    return value


def _source_file_counts(
    repo_root: Path,
    relative_scopes: Sequence[Path] | None = None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    scopes = set(relative_scopes or SOURCE_COUNT_SCOPES)
    for relative in sorted(scopes, key=lambda path: path.as_posix()):
        path = repo_root / relative
        count = sum(candidate.is_file() for candidate in path.rglob("*")) if path.is_dir() else int(path.is_file())
        result[relative.as_posix()] = count
    return result


def _manifest_session(item: Mapping[str, Any], object_type: str) -> str:
    value = item.get("session_id") or item.get("source_set")
    if value:
        return safe_component(str(value))
    image = Path(_manifest_image(item))
    return safe_component(image.parent.name or object_type)


def _source_frame_stem(item: Mapping[str, Any]) -> str:
    frame_id = str(item.get("frame_id") or Path(_manifest_image(item)).stem)
    if ":" in frame_id:
        frame_id = frame_id.rsplit(":", 1)[-1]
    return safe_component(frame_id, fallback=safe_component(Path(_manifest_image(item)).stem))


def _allocate_frame_identity(
    session_id: str,
    stem: str,
    digest: str,
    used: set[str],
) -> tuple[str, str]:
    clean_stem = safe_component(stem, fallback="frame")
    frame_id = f"{safe_component(session_id)}__{clean_stem}"
    destination_stem = clean_stem
    if frame_id in used:
        destination_stem = f"{clean_stem}__{digest[:10]}"
        frame_id = f"{safe_component(session_id)}__{destination_stem}"
    if frame_id in used:
        raise WorkspaceError(f"unable to allocate unique frame_id for {session_id}/{stem}")
    used.add(frame_id)
    return frame_id, destination_stem


def _legacy_mapped_rgb_dirs(repo_root: Path, annotation_dir_name: str) -> list[Path]:
    raw = repo_root / "data/pallet/raw_data"
    name = annotation_dir_name
    match = re.fullmatch(r"capturepallet(\d+)_manual_gt", name)
    if match:
        return [raw / "outside" / f"capturepallet{match.group(1)}" / "rgb"]
    match = re.fullmatch(r"capturenight(\d+)_manual_gt", name)
    if match:
        return [raw / "night" / f"capturenight{match.group(1)}" / "rgb"]
    match = re.fullmatch(r"wood_pallet_20260618_(\d+)_manual_gt", name)
    if match:
        stamp = match.group(1)
        return [
            raw / "wood/selected" / f"pallet_20260618_{stamp}",
            raw / "wood" / f"_annotate_pallet_20260618_{stamp}" / "rgb",
        ]
    mappings = {
        "capturepalletcad_manual_gt": [raw / "outside/capturepalletcad/rgb"],
        "capture0403noapril_manual_gt": [raw / "capture0403noapril/rgb"],
        "_outside_eval_manual_gt": [raw / "_outside_all/rgb"],
        "_night_eval_manual_gt": [raw / "_night_all/rgb"],
        "pallet11_gt": [raw / "outside/capturepallet11/rgb"],
        "forklift_20260528_manual_gt": [
            raw / "outside/forklift_raw_20260528_163408/rgb",
            raw / "outside/forklift_raw_20260528/rgb",
        ],
    }
    return mappings.get(name, [])


def resolve_legacy_image(
    repo_root: Path,
    annotation_path: Path,
    hash_cache: dict[Path, str],
) -> tuple[Path | None, str]:
    """Resolve by colocated image, then explicit session mapping; never basename-global."""

    suffixes = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    sibling = [annotation_path.with_suffix(suffix) for suffix in suffixes if annotation_path.with_suffix(suffix).is_file()]
    candidates = sibling
    resolution = "sibling"
    if not candidates:
        resolution = "session_mapping"
        for rgb_dir in _legacy_mapped_rgb_dirs(repo_root, annotation_path.parent.name):
            candidates.extend(rgb_dir / f"{annotation_path.stem}{suffix}" for suffix in suffixes)
        candidates = [path for path in candidates if path.is_file()]
    candidates = sorted({path.resolve() for path in candidates}, key=str)
    if not candidates:
        return None, "UNRESOLVED_NO_EXPLICIT_SESSION_IMAGE"
    if len(candidates) > 1:
        digests = set()
        for candidate in candidates:
            digest = hash_cache.get(candidate)
            if digest is None:
                digest = sha256_file(candidate)
                hash_cache[candidate] = digest
            digests.add(digest)
        if len(digests) != 1:
            return None, "UNRESOLVED_AMBIGUOUS_SESSION_IMAGES_DIFFER"
        resolution += "_identical_candidates"
    return candidates[0], resolution


def _provenance_row(
    *,
    record: ImportRecord | None,
    source_priority: int,
    source_dataset: str,
    image: FileSnapshot | None,
    annotation: FileSnapshot | None,
    disposition: str,
    duplicate_of: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "active_frame_id": record.active_frame_id if record else "",
        "source_priority": str(source_priority),
        "source_dataset": source_dataset,
        "source_image_path": str(image.path) if image else "",
        "source_annotation_path": str(annotation.path) if annotation else "",
        "source_image_sha256": image.sha256 if image else "",
        "source_annotation_sha256": annotation.sha256 if annotation else "",
        "source_image_mtime_ns_before": str(image.mtime_ns) if image else "",
        "source_annotation_mtime_ns_before": str(annotation.mtime_ns) if annotation else "",
        "source_image_size_before": str(image.size) if image else "",
        "source_annotation_size_before": str(annotation.size) if annotation else "",
        "disposition": disposition,
        "duplicate_of_frame_id": duplicate_of,
        "destination_image_path": "",
        "destination_annotation_path": "",
        "notes": notes,
    }


def build_import_plan(
    repo_root: Path,
    *,
    include_legacy: bool = True,
    expected: ExpectedCounts = DEFAULT_EXPECTED,
) -> ImportPlan:
    repo_root = repo_root.resolve()
    plastic_doc = _read_manifest(repo_root, "DEV_POS140", expected.plastic)
    controlled_doc = _read_manifest(repo_root, "COMMON_DEV_POS128", expected.controlled_plastic)
    wood_doc = _read_manifest(repo_root, "DEV_WOOD_POS45", expected.wood)
    negative_doc = _read_manifest(repo_root, "DEV_NEG2689", expected.negative)
    multishape_path = repo_root / MANIFEST_DIR / "COMMON_DEV_MULTISHAPE_POS.json"
    multishape_doc = (
        _read_manifest(repo_root, "COMMON_DEV_MULTISHAPE_POS", expected.multishape)
        if multishape_path.is_file()
        else None
    )
    plastic_items = plastic_doc["items"]
    controlled_items = controlled_doc["items"]
    wood_items = wood_doc["items"]
    negative_items = negative_doc["items"]
    validate_membership_contract(
        plastic_items=plastic_items,
        controlled_items=controlled_items,
        wood_items=wood_items,
        negative_items=negative_items,
        multishape_items=multishape_doc["items"] if multishape_doc else None,
        expected=expected,
    )

    snapshots: dict[Path, FileSnapshot] = {}
    hash_cache: dict[Path, str] = {}
    provenance: list[dict[str, str]] = []
    records: list[ImportRecord] = []
    active_by_sha: dict[str, ImportRecord] = {}
    used_frame_ids: set[str] = set()
    controlled_paths = {_membership_identity(item) for item in controlled_items}

    def audited_record(
        item: Mapping[str, Any],
        *,
        priority: int,
        source_dataset: str,
        object_type: str,
        paper_subset: str,
        controlled: bool,
        cross_shape: bool,
        exclusion_reason: str = "",
    ) -> ImportRecord:
        image_path = _repo_source_path(repo_root, _manifest_image(item))
        annotation_value = _manifest_annotation(item)
        if annotation_value is None:
            raise WorkspaceError(f"positive audited item has no annotation: {item}")
        annotation_path = _repo_source_path(repo_root, annotation_value)
        image = _snapshot(image_path, "image", snapshots, hash_cache)
        annotation = _snapshot(annotation_path, "annotation", snapshots, hash_cache)
        inferred = infer_annotation_tags(annotation_path)
        session = _manifest_session(item, object_type)
        stem = _source_frame_stem(item)
        frame_id, destination_stem = _allocate_frame_identity(session, stem, image.sha256, used_frame_ids)
        domain = str(item.get("domain", "unknown")).lower()
        lighting = domain if domain in {"day", "night"} else "unknown"
        record = ImportRecord(
            source_priority=priority,
            source_dataset=source_dataset,
            population_role="DEV",
            paper_subset=paper_subset,
            controlled_eval_eligible=controlled,
            cross_shape_eval_eligible=cross_shape,
            exclusion_reason=exclusion_reason,
            session_id=session,
            source_frame_id=str(item.get("frame_id", stem)),
            object_type=object_type,
            lighting=lighting,
            source_image=image_path,
            source_annotation=annotation_path,
            image_sha256=image.sha256,
            annotation_sha256=annotation.sha256,
            occlusion=inferred["occlusion"],
            truncation=inferred["truncation"],
            active_frame_id=frame_id,
            destination_stem=destination_stem,
        )
        if image.sha256 in active_by_sha:
            raise WorkspaceError(
                "audited positive image SHA is duplicated: "
                f"{active_by_sha[image.sha256].source_image} and {image_path}"
            )
        active_by_sha[image.sha256] = record
        records.append(record)
        provenance.append(
            _provenance_row(
                record=record,
                source_priority=priority,
                source_dataset=source_dataset,
                image=image,
                annotation=annotation,
                disposition="ACTIVE_INDEPENDENT_COPY",
            )
        )
        return record

    for item in plastic_items:
        annotation_value = _manifest_annotation(item) or ""
        if not annotation_value.startswith("challenge/real_gt_v2/migrated_gt/"):
            raise WorkspaceError(
                "DEV plastic annotation must come from migrated_gt source of truth: "
                f"{annotation_value}"
            )
        controlled = _membership_identity(item) in controlled_paths
        audited_record(
            item,
            priority=1,
            source_dataset="real_gt_v2_plastic_audited",
            object_type="plastic",
            paper_subset="COMMON_DEV_PLASTIC_POS128" if controlled else "DEV_PLASTIC_EXTRA",
            controlled=controlled,
            cross_shape=controlled,
            exclusion_reason="" if controlled else "FT_OVERLAP",
        )
    for item in wood_items:
        annotation_value = _manifest_annotation(item) or ""
        if not annotation_value.startswith("challenge/real_gt_v2/migrated_gt_wood/"):
            raise WorkspaceError(
                "DEV wood annotation must come from audited migrated_gt_wood: "
                f"{annotation_value}"
            )
        audited_record(
            item,
            priority=2,
            source_dataset="real_gt_v2_wood_audited",
            object_type="wood",
            paper_subset="DEV_WOOD_POS45",
            controlled=True,
            cross_shape=True,
        )

    controlled_sha = {
        record.image_sha256
        for record in records
        if record.paper_subset in {"COMMON_DEV_PLASTIC_POS128", "DEV_WOOD_POS45"}
    }
    if len(controlled_sha) != expected.multishape:
        raise WorkspaceError(
            "controlled multi-shape image SHA union mismatch: "
            f"actual={len(controlled_sha)}, expected={expected.multishape}"
        )

    negative_hash_groups: dict[str, list[str]] = defaultdict(list)
    for item in negative_items:
        image_path = _repo_source_path(repo_root, _manifest_image(item))
        image = _snapshot(image_path, "image", snapshots, hash_cache)
        session = "dev_negative"
        stem = _source_frame_stem(item)
        frame_id, destination_stem = _allocate_frame_identity(session, stem, image.sha256, used_frame_ids)
        record = ImportRecord(
            source_priority=1,
            source_dataset="real_gt_v2_negative_audited",
            population_role="DEV",
            paper_subset="DEV_NEG2689",
            controlled_eval_eligible=True,
            cross_shape_eval_eligible=False,
            exclusion_reason="",
            session_id=session,
            source_frame_id=str(item.get("frame_id", stem)),
            object_type="none",
            lighting="unknown",
            source_image=image_path,
            source_annotation=None,
            image_sha256=image.sha256,
            annotation_sha256="",
            active_frame_id=frame_id,
            destination_stem=destination_stem,
            storage_mode="independent_copy",
            notes="Frozen DEV negative membership; duplicate SHA membership is preserved.",
        )
        records.append(record)
        negative_hash_groups[image.sha256].append(frame_id)
        # There is no positive/negative overlap in the audited source today. If
        # it ever appears, failing is safer than silently changing membership.
        if image.sha256 in active_by_sha and active_by_sha[image.sha256].source_annotation is not None:
            raise WorkspaceError(f"audited positive/negative image SHA overlap: {image_path}")
        active_by_sha.setdefault(image.sha256, record)
        provenance.append(
            _provenance_row(
                record=record,
                source_priority=1,
                source_dataset=record.source_dataset,
                image=image,
                annotation=None,
                disposition="ACTIVE_INDEPENDENT_COPY",
                notes=record.notes,
            )
        )

    unresolved: list[str] = []
    legacy_annotation_paths = 0
    legacy_active = 0
    if include_legacy:
        for priority, source_dataset, relative_root in LEGACY_ROOTS:
            source_root = repo_root / relative_root
            if not source_root.is_dir():
                raise WorkspaceError(f"legacy source root missing: {source_root}")
            for annotation_path in sorted(source_root.rglob("*.json")):
                legacy_annotation_paths += 1
                annotation = _snapshot(annotation_path, "annotation", snapshots, hash_cache)
                image_path, resolution_note = resolve_legacy_image(repo_root, annotation_path, hash_cache)
                if image_path is None:
                    unresolved.append(_repo_relative(repo_root, annotation_path))
                    provenance.append(
                        _provenance_row(
                            record=None,
                            source_priority=priority,
                            source_dataset=source_dataset,
                            image=None,
                            annotation=annotation,
                            disposition="UNRESOLVED_IMAGE",
                            notes=resolution_note,
                        )
                    )
                    continue
                image = _snapshot(image_path, "image", snapshots, hash_cache)
                existing = active_by_sha.get(image.sha256)
                if existing is not None:
                    provenance.append(
                        _provenance_row(
                            record=None,
                            source_priority=priority,
                            source_dataset=source_dataset,
                            image=image,
                            annotation=annotation,
                            disposition="DUPLICATE_PROVENANCE_ONLY",
                            duplicate_of=existing.active_frame_id,
                            notes=resolution_note,
                        )
                    )
                    continue

                session = safe_component(annotation_path.parent.name)
                frame_id, destination_stem = _allocate_frame_identity(
                    session, annotation_path.stem, image.sha256, used_frame_ids
                )
                inferred = infer_annotation_tags(annotation_path)
                object_type = inferred["object_type"]
                if object_type == "unknown":
                    object_type = "wood" if session.startswith("wood_") else "plastic"
                lighting = "night" if session.startswith("capturenight") or session.startswith("night_eval") else "unknown"
                record = ImportRecord(
                    source_priority=priority,
                    source_dataset=source_dataset,
                    population_role="DEV_UNVERIFIED",
                    paper_subset="NONE",
                    controlled_eval_eligible=False,
                    cross_shape_eval_eligible=False,
                    exclusion_reason="UNVERIFIED_LEGACY",
                    session_id=session,
                    source_frame_id=annotation_path.stem,
                    object_type=object_type,
                    lighting=lighting,
                    source_image=image_path,
                    source_annotation=annotation_path,
                    image_sha256=image.sha256,
                    annotation_sha256=annotation.sha256,
                    occlusion=inferred["occlusion"],
                    truncation=inferred["truncation"],
                    active_frame_id=frame_id,
                    destination_stem=destination_stem,
                    notes=resolution_note,
                )
                records.append(record)
                active_by_sha[image.sha256] = record
                legacy_active += 1
                provenance.append(
                    _provenance_row(
                        record=record,
                        source_priority=priority,
                        source_dataset=source_dataset,
                        image=image,
                        annotation=annotation,
                        disposition="ACTIVE_INDEPENDENT_COPY",
                        notes=resolution_note,
                    )
                )

    warnings: list[str] = []
    if unresolved:
        raise WorkspaceError(
            "legacy annotation image resolution is incomplete; import aborted so annotations "
            f"are not silently dropped: count={len(unresolved)}, paths={unresolved[:10]}"
        )
    # 409 is current audit evidence, not a frozen membership contract. Report a
    # drift warning without preventing future legacy annotations from growing.
    if include_legacy and legacy_active != 409:
        warnings.append(f"legacy unique active drift: current={legacy_active}, 2026-08-30 audit=409")

    source_count_scopes = set(SOURCE_COUNT_SCOPES)
    for snapshot in snapshots.values():
        source_count_scopes.add(snapshot.path.parent.relative_to(repo_root))
    return ImportPlan(
        repo_root=repo_root,
        records=records,
        provenance=provenance,
        snapshots=snapshots,
        scope_counts_before=_source_file_counts(repo_root, tuple(source_count_scopes)),
        unresolved_legacy=unresolved,
        duplicate_positive_sha_groups=0,
        duplicate_negative_sha_groups=sum(1 for values in negative_hash_groups.values() if len(values) > 1),
        legacy_annotation_paths=legacy_annotation_paths,
        legacy_unique_active=legacy_active,
        membership_counts={
            "plastic": len(plastic_items),
            "controlled_plastic": len(controlled_items),
            "plastic_excluded": len(plastic_items) - len(controlled_items),
            "wood": len(wood_items),
            "multishape": len(controlled_sha),
            "negative": len(negative_items),
        },
        warnings=warnings,
    )


def _source_safety_changes(plan: ImportPlan) -> tuple[list[str], list[str], list[str]]:
    image_changes: list[str] = []
    annotation_changes: list[str] = []
    metadata_changes: list[str] = []
    for snapshot in plan.snapshots.values():
        if not snapshot.path.is_file():
            (image_changes if snapshot.kind == "image" else annotation_changes).append(
                f"REMOVED {snapshot.path}"
            )
            continue
        stat = snapshot.path.stat()
        digest = sha256_file(snapshot.path)
        bucket = image_changes if snapshot.kind == "image" else annotation_changes
        if digest != snapshot.sha256:
            bucket.append(f"SHA {snapshot.path}")
        if stat.st_size != snapshot.size or stat.st_mtime_ns != snapshot.mtime_ns:
            metadata_changes.append(f"SIZE_OR_MTIME {snapshot.path}")
    scope_after = _source_file_counts(
        plan.repo_root,
        tuple(Path(scope) for scope in plan.scope_counts_before),
    )
    count_changes = [
        f"{scope}: before={before}, after={scope_after.get(scope)}"
        for scope, before in plan.scope_counts_before.items()
        if scope_after.get(scope) != before
    ]
    metadata_changes.extend(f"FILE_COUNT {value}" for value in count_changes)
    return image_changes, annotation_changes, metadata_changes


def _session_document(records: Sequence[ImportRecord]) -> dict[str, Any]:
    first = records[0]
    lighting = {record.lighting for record in records}
    object_types = {record.object_type for record in records}
    occlusion = {record.occlusion for record in records}
    truncation = {record.truncation for record in records}
    resolution: list[int] | None = None
    if first.source_annotation:
        try:
            data = json.loads(first.source_annotation.read_text(encoding="utf-8"))
            camera = data.get("camera_data") or {}
            if camera.get("width") and camera.get("height"):
                resolution = [int(camera["width"]), int(camera["height"])]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            resolution = None
    return {
        "session_id": first.session_id,
        "population_role": first.population_role,
        "object_type": next(iter(object_types)) if len(object_types) == 1 else "unknown",
        "lighting": next(iter(lighting)) if len(lighting) == 1 else "unknown",
        "capture_protocol": "legacy_import",
        "camera": "unknown",
        "resolution": resolution,
        "default_tags": {
            "occlusion": next(iter(occlusion)) if len(occlusion) == 1 else "unknown",
            "truncation": next(iter(truncation)) if len(truncation) == 1 else "unknown",
        },
        "source_dataset": sorted({record.source_dataset for record in records}),
        "imported_read_only_source": True,
    }


def _build_workspace(stage_root: Path, plan: ImportPlan) -> list[dict[str, str]]:
    geometry = plan.repo_root / GEOMETRY_REGISTRY
    if not geometry.is_file():
        raise WorkspaceError(f"geometry registry missing: {geometry}")
    scaffold_workspace(stage_root, geometry_registry=geometry)
    frame_rows: list[dict[str, str]] = []
    records_by_session: dict[tuple[str, str], list[ImportRecord]] = defaultdict(list)
    provenance_by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in plan.provenance:
        key = (row["source_dataset"], row["source_image_path"], row["source_annotation_path"])
        provenance_by_identity[key] = row

    for record in plan.records:
        source_image_rel = _repo_relative(plan.repo_root, record.source_image)
        source_annotation_rel = _repo_relative(plan.repo_root, record.source_annotation)
        scope = "dev_existing" if record.population_role == "DEV" else "legacy_unverified"
        session_root = stage_root / scope / "sessions" / record.session_id
        destination_image = (
            session_root
            / "rgb"
            / f"{record.destination_stem}{record.source_image.suffix.lower()}"
        )
        copy2_verified(record.source_image, destination_image)
        image_path_value = workspace_relative(destination_image, stage_root)
        annotation_path_value = ""
        overlay_path_value = ""
        if record.source_annotation is not None:
            destination_annotation = (
                stage_root
                / scope
                / "annotations"
                / record.session_id
                / f"{record.destination_stem}.json"
            )
            destination_overlay = (
                destination_annotation.parent
                / "_overlays"
                / f"{record.destination_stem}.png"
            )
            copy2_verified(record.source_annotation, destination_annotation)
            annotation_path_value = workspace_relative(destination_annotation, stage_root)
            overlay_path_value = workspace_relative(destination_overlay, stage_root)
        records_by_session[(scope, record.session_id)].append(record)

        key = (
            record.source_dataset,
            str(record.source_image.resolve()),
            str(record.source_annotation.resolve()) if record.source_annotation else "",
        )
        provenance = provenance_by_identity.get(key)
        if provenance is not None:
            provenance["destination_image_path"] = image_path_value
            provenance["destination_annotation_path"] = annotation_path_value

        frame_rows.append(
            {
                "frame_id": record.active_frame_id,
                "population_role": record.population_role,
                "paper_subset": record.paper_subset,
                "controlled_eval_eligible": bool_text(record.controlled_eval_eligible),
                "cross_shape_eval_eligible": bool_text(record.cross_shape_eval_eligible),
                "exclusion_reason": record.exclusion_reason,
                "session_id": record.session_id,
                "object_type": record.object_type,
                "lighting": record.lighting,
                "occlusion": record.occlusion,
                "truncation": record.truncation,
                "distance_bin": "unknown",
                "size_bin": "unknown",
                "elevation_bin": "unknown",
                "view_bin": "unknown",
                "is_positive": bool_text(record.source_annotation is not None),
                "is_annotated": bool_text(record.source_annotation is not None),
                "image_path": image_path_value,
                "annotation_path": annotation_path_value,
                "overlay_path": overlay_path_value,
                "source_dataset": record.source_dataset,
                "source_image_path": source_image_rel,
                "source_annotation_path": source_annotation_rel,
                "image_sha256": record.image_sha256,
                "annotation_sha256": record.annotation_sha256,
                "source_image_sha256": record.image_sha256,
                "source_annotation_sha256": record.annotation_sha256,
                "storage_mode": record.storage_mode,
                "notes": record.notes,
            }
        )

    for (scope, session_id), records in records_by_session.items():
        session_path = stage_root / scope / "sessions" / session_id / "session.json"
        atomic_write_json(session_path, _session_document(records))

    frame_rows.sort(key=lambda row: row["frame_id"])
    atomic_write_csv(stage_root / "manifests/frames.csv", frame_rows, FRAME_COLUMNS)
    write_manifest_views(stage_root, frame_rows, enforce_frozen_dev=True)
    # Convert absolute audit paths to repository-relative paths in the published CSV.
    published_provenance: list[dict[str, str]] = []
    for raw in plan.provenance:
        row = dict(raw)
        for field in ("source_image_path", "source_annotation_path"):
            if row.get(field):
                row[field] = _repo_relative(plan.repo_root, Path(row[field]))
        published_provenance.append(row)
    atomic_write_csv(
        stage_root / "manifests/import_provenance.csv",
        published_provenance,
        PROVENANCE_COLUMNS,
    )
    write_reports(stage_root, frame_rows)
    return frame_rows


def _render_import_audit(
    plan: ImportPlan,
    image_changes: Sequence[str],
    annotation_changes: Sequence[str],
    metadata_changes: Sequence[str],
) -> str:
    counts = plan.membership_counts
    warning_lines = "\n".join(f"- {warning}" for warning in plan.warnings) or "- 없음"
    unresolved_lines = "\n".join(f"- {path}" for path in plan.unresolved_legacy[:200]) or "- 없음"
    return f"""# Import audit

```text
plastic total            {counts['plastic']} / 140
plastic controlled       {counts['controlled_plastic']} / 128
plastic excluded         {counts['plastic_excluded']} / 12
wood total               {counts['wood']} / 45
multi-shape controlled   {counts['multishape']} / 173
negative indexed         {counts['negative']} / 2689
legacy annotation paths  {plan.legacy_annotation_paths}
legacy unique active     {plan.legacy_unique_active}
legacy unresolved        {len(plan.unresolved_legacy)}
negative duplicate SHA groups  {plan.duplicate_negative_sha_groups}
```

Legacy `409`는 2026-08-30 현재 감사값일 뿐 frozen membership target이 아니다.

## Source safety

```text
source image changed        {len(image_changes)}
source annotation changed   {len(annotation_changes)}
source size/mtime/count changed  {len(metadata_changes)}
physical DEV -> FINAL copies  0
FINAL_EVAL execution alias  173 positive / 2689 negative rows (2688 unique)
duplicate active frame SHA groups  {plan.duplicate_positive_sha_groups + plan.duplicate_negative_sha_groups}
```

현재 active duplicate group은 frozen `DEV_NEG2689`의 기존 중복 membership이며
삭제하지 않고 `DUPLICATE_AUDIT.csv`에 보존한다.

## Warnings

{warning_lines}

## Unresolved legacy images

{unresolved_lines}
"""


def execute_import(
    plan: ImportPlan,
    root: Path,
) -> None:
    root = root.resolve()
    if root.exists():
        raise WorkspaceError(
            f"destination already exists; refusing to overwrite working annotations: {root}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.import-", dir=root.parent))
    published = False
    try:
        frame_rows = _build_workspace(stage, plan)
        overlay_script = plan.repo_root / "scripts/annotate/rebuild_annotation_overlays.py"
        if not overlay_script.is_file():
            raise WorkspaceError(f"required overlay backfill script not found: {overlay_script}")
        for scope in ("dev_existing", "legacy_unverified"):
            command = [
                sys.executable,
                str(overlay_script),
                "--dataset-root",
                str(stage),
                "--scope",
                scope,
                "--force",
            ]
            completed = subprocess.run(
                command,
                cwd=plan.repo_root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                raise WorkspaceError(
                    f"overlay backfill failed for {scope}: exit={completed.returncode}\n"
                    f"{completed.stdout[-4000:]}"
                )
        for role, scope in (("DEV", "dev_existing"), ("DEV_UNVERIFIED", "legacy_unverified")):
            members = [
                row
                for row in frame_rows
                if row["population_role"] == role
                and row["is_positive"] == "true"
                and row["is_annotated"] == "true"
            ]
            missing = [
                row["overlay_path"]
                for row in members
                if not (stage / row["overlay_path"]).is_file()
            ]
            if missing:
                raise WorkspaceError(
                    f"overlay completeness mismatch for {scope}: "
                    f"annotations={len(members)}, overlays={len(members) - len(missing)}, "
                    f"missing={missing[:10]}"
                )
        write_reports(stage, frame_rows)
        image_changes, annotation_changes, metadata_changes = _source_safety_changes(plan)
        if image_changes or annotation_changes or metadata_changes:
            detail = (image_changes + annotation_changes + metadata_changes)[:20]
            raise WorkspaceError(f"source safety verification failed; import aborted: {detail}")
        atomic_write_text(
            stage / "reports/IMPORT_AUDIT.md",
            _render_import_audit(plan, image_changes, annotation_changes, metadata_changes),
        )
        os.replace(stage, root)
        published = True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _print_plan(plan: ImportPlan) -> None:
    counts = plan.membership_counts
    print("[ROOT SOURCE AUDIT]")
    print(f"repo: {plan.repo_root}")
    print()
    print("[EXISTING IMPORT PLAN]")
    print(f"plastic total:          {counts['plastic']}")
    print(f"plastic controlled:     {counts['controlled_plastic']}")
    print(f"plastic excluded:       {counts['plastic_excluded']}")
    print(f"wood total:             {counts['wood']}")
    print(f"multi-shape controlled: {counts['multishape']}")
    print(f"negative indexed:       {counts['negative']}")
    print(f"legacy annotation paths:{plan.legacy_annotation_paths:>7}")
    print(f"legacy unique active:   {plan.legacy_unique_active:>7}")
    print(f"legacy unresolved:      {len(plan.unresolved_legacy):>7}")
    print(f"negative duplicate SHA groups: {plan.duplicate_negative_sha_groups}")
    if plan.warnings:
        print()
        print("[WARNINGS]")
        for warning in plan.warnings:
            print(f"- {warning}")


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "challenge").is_dir():
            return candidate
    raise WorkspaceError(f"could not find repository root from {start}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/evaluation/pallet_eval_v1"))
    parser.add_argument("--repo-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="perform a read-only audit without creating the workspace",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="compatibility alias; import is already the default",
    )
    parser.add_argument("--no-legacy", action="store_true", help="skip DEV_UNVERIFIED legacy scan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd())
        root = args.root if args.root.is_absolute() else repo_root / args.root
        plan = build_import_plan(repo_root, include_legacy=not args.no_legacy)
        _print_plan(plan)
        if args.audit_only:
            print()
            print("[AUDIT ONLY] destination was not created.")
            return 0
        execute_import(plan, root)
        print()
        print(f"[IMPORTED] {root.resolve()}")
        return 0
    except (WorkspaceError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
