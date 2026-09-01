"""Shared contracts and filesystem helpers for ``pallet_eval_v1``.

This module owns the portable CSV schemas, target definitions, safe atomic
writes, FINAL-session discovery, and report calculations.  Source-data import
is implemented separately so routine status refreshes never touch legacy data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pallet_eval_workspace_v1"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

FRAME_COLUMNS = (
    "frame_id",
    "population_role",
    "paper_subset",
    "controlled_eval_eligible",
    "cross_shape_eval_eligible",
    "exclusion_reason",
    "session_id",
    "object_type",
    "lighting",
    "acquisition_domain",
    "paper_domain",
    "usage_role",
    "occlusion",
    "truncation",
    "distance_bin",
    "size_bin",
    "elevation_bin",
    "view_bin",
    "is_positive",
    "is_annotated",
    "image_path",
    "annotation_path",
    "overlay_path",
    "source_dataset",
    "source_image_path",
    "source_annotation_path",
    "image_sha256",
    "annotation_sha256",
    "source_image_sha256",
    "source_annotation_sha256",
    "storage_mode",
    "notes",
)

SESSION_COLUMNS = (
    "session_id",
    "population_role",
    "object_type",
    "lighting",
    "is_positive",
    "frame_count",
    "annotated_count",
    "session_path",
    "session_metadata_path",
    "source_dataset",
    "notes",
)

PROVENANCE_COLUMNS = (
    "active_frame_id",
    "source_priority",
    "source_dataset",
    "source_image_path",
    "source_annotation_path",
    "source_image_sha256",
    "source_annotation_sha256",
    "source_image_mtime_ns_before",
    "source_annotation_mtime_ns_before",
    "source_image_size_before",
    "source_annotation_size_before",
    "disposition",
    "duplicate_of_frame_id",
    "destination_image_path",
    "destination_annotation_path",
    "notes",
)

ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "population_role": frozenset({"DEV", "DEV_UNVERIFIED", "FINAL"}),
    "object_type": frozenset({"plastic", "wood", "none", "unknown"}),
    "lighting": frozenset({"day", "night", "unknown"}),
    # acquisition_domain 은 환경 factor 가 아니라 **capture provenance category** 다.
    # indoor/outdoor x day/night 2x2 factorial 은 폐기했다 — 두 축을 독립으로
    # 가를 provenance 가 데이터에 없다.
    "acquisition_domain": frozenset({
        "outside", "night", "noapril", "cad", "forklift", "other", "unknown"}),
    # paper_domain 은 사람이 입력하는 값이 아니라 derive_paper_domain 이 정한다.
    "paper_domain": frozenset({"daytime", "nighttime", "none", "unknown"}),
    "usage_role": frozenset({
        "EVAL_LABELED", "ADAPT_UNLABELED", "DEV_SUPPORT", "NEGATIVE_EVAL",
        "unknown"}),
    "occlusion": frozenset({"none", "mild", "medium", "heavy", "unknown"}),
    "truncation": frozenset({"none", "mild", "medium", "heavy", "unknown"}),
    "distance_bin": frozenset({"near", "mid", "far", "unknown"}),
    "size_bin": frozenset({"small", "medium", "large", "unknown"}),
    "elevation_bin": frozenset({"low", "mid", "high", "unknown"}),
}

# 2026-09-01 에 추가된 도메인 축.  옛 manifest 에는 컬럼이 없으므로 빈 값을
# unknown 으로 받아들인다 (그 외 필드는 종전대로 빈 값을 거부한다).
NEW_DOMAIN_FIELDS = frozenset({"acquisition_domain", "paper_domain", "usage_role"})

PAPER_SUBSETS = frozenset({
    "COMMON_DEV_PLASTIC_POS128",
    "DEV_PLASTIC_EXTRA",
    "DEV_WOOD_POS45",
    "DEV_NEG2689",
    "NONE",
    "FINAL_POSITIVE",
    "FINAL_NEGATIVE",
})
DEV_EVAL_POSITIVE_SUBSETS = frozenset({
    "COMMON_DEV_PLASTIC_POS128",
    "DEV_WOOD_POS45",
})
DEV_PLASTIC_AUDITED_SUBSETS = frozenset({
    "COMMON_DEV_PLASTIC_POS128",
    "DEV_PLASTIC_EXTRA",
})
BOOLEAN_FIELDS = (
    "controlled_eval_eligible",
    "cross_shape_eval_eligible",
    "is_positive",
    "is_annotated",
)
FRAME_TAG_FIELDS = (
    "distance_bin",
    "size_bin",
    "elevation_bin",
    "view_bin",
    "occlusion",
    "truncation",
)
FRAME_TAG_COLUMNS = ("frame", *FRAME_TAG_FIELDS)
EFFECTIVE_FRAME_TAG_FIELDS = (
    "object_type",
    "lighting",
    "acquisition_domain",
    "occlusion",
    "truncation",
    "distance_bin",
    "size_bin",
    "elevation_bin",
    "view_bin",
)
FRAME_TAG_SOURCES = frozenset({"FRAME", "JSON", "SESSION", "UNSET"})
STORAGE_MODES = frozenset({
    "workspace_native",
    "independent_copy",
    "source_reference_read_only",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KNOWN_DEV_NEG_DUPLICATE_SHA256 = (
    "7e6359c1fc3dbaf6c0adfb7a9f935d53bc668c63cd1c3aab290c680ee26bb6e0"
)
KNOWN_DEV_NEG_DUPLICATE_PATHS = {
    "dev_negative__000238": "data/pallet/raw_data/negative_real_20260823/rgb/000238.png",
    "dev_negative__000239": "data/pallet/raw_data/negative_real_20260823/rgb/000239.png",
}
KNOWN_DEV_NEG_DUPLICATE_WORKSPACE_PATHS = {
    "dev_negative__000238": "dev_existing/sessions/dev_negative/rgb/000238.png",
    "dev_negative__000239": "dev_existing/sessions/dev_negative/rgb/000239.png",
}
REUSED_DEV_EVAL_ALIAS_NOTE = "REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV"

# DATASET_TARGETS.json 이 없을 때의 폴백.  그 파일과 값이 같아야 한다.
# 2026-09-01 domain 재설계: 300 은 논문 최소 완료선으로 남기고 400 을 권장으로 둔다.
# DATASET_TARGETS.json 이 없을 때의 폴백.  그 파일과 값이 같아야 한다.
# 2026-09-01: indoor/outdoor x day/night 2x2 폐기 -> acquisition_domain 기반.
DEFAULT_TARGETS: dict[str, Any] = {
    "progress_population": "PAPER_EVAL",
    "positive_minimum": 300,
    "positive_total": 300,
    "object_type": {"plastic": 180, "wood": 120},
    # MAIN 논문 조건은 Daytime / Nighttime 두 개뿐이다.
    "main_paper_domains": {
        "daytime": {"object_type": "plastic", "minimum_frames": 50,
                    "preferred_frames": 60, "minimum_sessions": 2,
                    "internal": "acquisition_domain=outside + lighting=day"},
        "nighttime": {"object_type": "plastic", "minimum_frames": 50,
                      "preferred_frames": 60, "minimum_sessions": 2,
                      "internal": "acquisition_domain=night + lighting=night"},
    },
    "appendix_only_internal_ids": {"noapril": True, "cad": True},
    "minimum_condition_coverage": {
        "clean": 80,
        "occlusion": 80,
        "truncation": 50,
        "far": 50,
    },
    "elevation_minimum": {"low": 60, "mid": 60, "high": 40},
    "elevation": {"low": 60, "mid": 60, "high": 40},
    "negative_unique_minimum": 1500,
    "negative_total": 1500,
}

DATASET_CONTRACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "dataset_id": "pallet_eval_v1",
    "identity": {
        "primary": "sha256(image bytes)",
        "frame_id": "globally unique session-qualified identifier",
        "filename_only_deduplication": False,
    },
    "roles": {
        "DEV": "previously used audited development data",
        "DEV_UNVERIFIED": "preserved legacy annotation, not paper eligible",
        "FINAL": "untouched final evaluation population",
    },
    "incoming_unreviewed": {
        "path": "incoming/sessions",
        "active_evaluation_member": False,
        "purpose": "raw continuous captures awaiting human positive/negative and object review",
        "activation": (
            "independently copy reviewed frames into final/positive or final/negative sessions"
        ),
        "forbidden": (
            "raw import must not assign FINAL membership or count toward combined "
            "evaluation collection targets before review"
        ),
    },
    "invariants": {
        "evaluation_target_progress_uses_all_available": True,
        "evaluation_target_progress_sha256_deduplicated": True,
        "source_data_is_read_only": True,
        "workspace_copies_are_not_hardlinks_or_symlinks": True,
        "conditions_are_tags_and_may_overlap": True,
        "unknown_metadata_is_not_inferred": True,
        "size_bin_is_legacy_compatibility_only": True,
        "final_rgb_is_rehashed_on_every_refresh": True,
        "active_image_sha_is_globally_unique": True,
        "sole_active_sha_exception": {
            "population": "DEV_NEG2689",
            "frame_ids": ["dev_negative__000238", "dev_negative__000239"],
            "sha256": (
                "7e6359c1fc3dbaf6c0adfb7a9f935d53bc668c63cd1c3aab290c680ee26bb6e0"
            ),
            "reason": "frozen source membership is preserved, not deduplicated",
        },
    },
    "final_eval_alias_policy": {
        "source_population": "DEV_EVAL",
        "physical_copy": False,
        "changes_active_frame_role": False,
        "row_provenance_note": REUSED_DEV_EVAL_ALIAS_NOTE,
        "held_out_final": False,
        "mandatory_new_annotation": False,
        "target_policy": (
            "DATASET_TARGETS.json is a rough collection goal; progress uses "
            "ALL_AVAILABLE (DEV_EVAL + physical FINAL)"
        ),
    },
    "path_contract": {
        "workspace_paths": "relative to dataset root when storage_mode is workspace_native or independent_copy",
        "source_reference_image_path": (
            "repository-root-relative when storage_mode is source_reference_read_only; "
            "source_image_path contains the same read-only identity"
        ),
        "source_paths": "relative to repository root when possible",
    },
    "allowed_values": {key: sorted(value) for key, value in ALLOWED_VALUES.items()},
    "condition_queries": {
        "plastic": "object_type == plastic",
        "wood": "object_type == wood",
        "day": "lighting == day",
        "night": "lighting == night",
        "occlusion": "occlusion in {mild, medium, heavy}",
        "truncation": "truncation in {mild, medium, heavy}",
        "far": "distance_bin == far",
        "clean": "occlusion == none AND truncation == none",
        "low": "elevation_bin == low",
        "mid": "elevation_bin == mid",
        "high": "elevation_bin == high",
    },
    "source_of_truth": {
        "frames": "manifests/frames.csv",
        "targets": "DATASET_TARGETS.json",
        "plastic_dev": "challenge/real_gt_v2/manifests/DEV_POS140.json",
        "plastic_controlled": "challenge/real_gt_v2/manifests/COMMON_DEV_POS128.json",
        "wood_dev": "challenge/real_gt_v2/manifests/DEV_WOOD_POS45.json",
        "negative_dev": "challenge/real_gt_v2/manifests/DEV_NEG2689.json",
    },
    "paper_evaluator_binding": {
        "population_role": "DEV",
        "positive_manifest": (
            "challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json"
        ),
        "negative_manifest": "challenge/real_gt_v2/manifests/DEV_NEG2689.json",
        "positive_rows": 173,
        "negative_rows": 2689,
        "negative_unique_images": 2688,
        "pair_sha256": (
            "2cfa7011d8ba3677b11019c103e2ccbaeeac53521c9291ed632f94c8d2c5c887"
        ),
    },
    "evaluation_populations": {
        "DEV_EVAL": {
            "positive": "COMMON_DEV_PLASTIC_POS128 + DEV_WOOD_POS45",
            "negative": "DEV_NEG2689 frozen membership",
            "held_out_final": False,
        },
        "FINAL_EVAL": {
            "positive": "frozen row-for-row alias of DEV_EVAL_POSITIVE",
            "negative": (
                "frozen row-for-row alias of DEV_EVAL_NEGATIVE; preserves the "
                "registered 2689-row membership and its one known duplicate image"
            ),
            "held_out_final": False,
            "reuses_dev_eval": True,
            "includes_physical_final": False,
            "alias_row_note": REUSED_DEV_EVAL_ALIAS_NOTE,
            "note": (
                "executable registered DEV pair alias; physical FINAL stays separate; "
                "never describe as held-out FINAL"
            ),
        },
        "ALL_AVAILABLE": {
            "positive": (
                "SHA256-deduplicated union(DEV_EVAL_POSITIVE, annotated "
                "QA-eligible physical FINAL positives)"
            ),
            "negative": (
                "SHA256-deduplicated union(DEV_EVAL_NEGATIVE, physical FINAL negatives)"
            ),
            "held_out_final": False,
            "note": "DEV may have been used for model selection",
        },
    },
}


class WorkspaceError(RuntimeError):
    """Raised when a workspace or session violates its explicit contract."""


@dataclass(frozen=True)
class Progress:
    positive_total: int
    plastic: int
    wood: int
    day: int
    night: int
    clean: int
    occlusion: int
    truncation: int
    far: int
    low: int
    mid: int
    high: int
    negative: int
    unknown_metadata: int
    evaluation_positive: int
    evaluation_negative: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy2_verified(source: Path, destination: Path) -> str:
    """Create an independent copy and verify content and inode separation."""

    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    shutil.copy2(source, destination)
    destination_hash = sha256_file(destination)
    if destination_hash != source_hash:
        raise WorkspaceError(f"copy SHA mismatch: {source} -> {destination}")
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        destination.unlink(missing_ok=True)
        raise WorkspaceError(f"copy unexpectedly shares source inode: {source}")
    if destination.is_symlink():
        destination.unlink(missing_ok=True)
        raise WorkspaceError(f"copy unexpectedly created a symlink: {destination}")
    return source_hash


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def bool_text(value: Any) -> str:
    if isinstance(value, str):
        return "true" if value.strip().lower() in {"1", "true", "yes", "y"} else "false"
    return "true" if bool(value) else "false"


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_component(value: str, *, fallback: str = "session") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned or fallback


def normalize_object_type(value: Any) -> str:
    raw = str(value or "unknown").strip().lower()
    if raw in {"plastic", "plastic_standard_110x130x11", "real_pallet"}:
        return "plastic"
    if raw in {"wood", "wood_small_80x59x14"}:
        return "wood"
    if raw in {"none", "negative"}:
        return "none"
    return "unknown"


def normalize_tag(field: str, value: Any, *, allow_view: bool = False) -> str:
    raw = str(value if value not in (None, "") else "unknown").strip().lower()
    aliases = {
        ("lighting", "daytime"): "day",
        ("lighting", "nighttime"): "night",
        ("occlusion", "partial"): "medium",
        ("truncation", "partial"): "medium",
    }
    raw = aliases.get((field, raw), raw)
    if field == "view_bin" and allow_view:
        return raw or "unknown"
    allowed = ALLOWED_VALUES.get(field)
    if allowed is not None and raw not in allowed:
        raise WorkspaceError(f"invalid {field}={value!r}; allowed={sorted(allowed)}")
    return raw


def workspace_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_workspace_path(root: Path, value: str) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def resolve_frame_image_path(
    root: Path,
    row: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Resolve either a workspace copy or an explicit read-only source reference."""

    value = row.get("image_path", "")
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if row.get("storage_mode") != "source_reference_read_only":
        return root / candidate
    if repo_root is None:
        for parent in (root.resolve(), *root.resolve().parents):
            if (parent / ".git").exists():
                repo_root = parent
                break
    if repo_root is None:
        raise WorkspaceError(
            f"repo_root is required to resolve read-only source reference: {value}"
        )
    return repo_root / candidate


def infer_annotation_document_tags(document: Mapping[str, Any]) -> dict[str, str]:
    """Infer only metadata explicitly represented in an annotation document."""

    inferred = {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
    objects = document.get("objects")
    if not isinstance(objects, list) or not objects or not isinstance(objects[0], Mapping):
        return inferred
    obj = objects[0]
    inferred["object_type"] = normalize_object_type(
        obj.get("object_type") or document.get("object_type") or obj.get("name")
    )

    level = str(obj.get("occlusion_level", "unknown")).strip().lower()
    reason_values = {
        str(entry.get("reason", "unknown")).strip().lower()
        for entry in obj.get("keypoint_annotations", [])
        if isinstance(entry, Mapping)
    }
    if level == "none":
        inferred["occlusion"] = "none"
    elif level in {"mild", "medium", "partial", "heavy"}:
        inferred["occlusion"] = "medium" if level == "partial" else level
    elif "occluded" in reason_values:
        inferred["occlusion"] = "medium"

    truncation = obj.get("truncation")
    fraction: float | None = None
    if isinstance(truncation, Mapping):
        raw_fraction = truncation.get("bbox_outside_fraction")
        if not isinstance(raw_fraction, bool):
            try:
                fraction = float(raw_fraction) if raw_fraction is not None else None
            except (TypeError, ValueError):
                fraction = None
    if (
        isinstance(truncation, Mapping)
        and truncation.get("is_truncated") is False
        and (fraction is None or fraction == 0.0)
    ):
        inferred["truncation"] = "none"
    if (
        isinstance(truncation, Mapping)
        and truncation.get("is_truncated") is True
    ) or (fraction is not None and fraction > 0.0) or "truncated" in reason_values:
        # The JSON supplies only a boolean, so no unrecorded severity threshold
        # is invented.  ``mild`` is the least-specific positive allowed tag.
        inferred["truncation"] = "mild"
    return inferred


def infer_annotation_tags(annotation_path: Path) -> dict[str, str]:
    """Infer explicit tags from saved JSON via the shared document resolver."""

    if not annotation_path.is_file():
        return {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
    if not isinstance(document, dict):
        return {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
    return infer_annotation_document_tags(document)


def load_session_metadata(session_dir: Path) -> dict[str, Any]:
    """Load one evaluation session's explicit metadata.

    The helper deliberately does not infer metadata from the directory name.
    Callers therefore share the same fail-closed JSON parsing and
    ``default_tags`` validation.
    """

    metadata_path = Path(session_dir) / "session.json"
    if not metadata_path.is_file():
        raise WorkspaceError(f"evaluation session is missing session.json: {session_dir}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceError(f"invalid session JSON {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise WorkspaceError(f"session JSON must be an object: {metadata_path}")
    default_tags = metadata.get("default_tags")
    if default_tags is not None and not isinstance(default_tags, dict):
        raise WorkspaceError(f"default_tags must be an object: {metadata_path}")
    return metadata


def canonical_frame_tag_identity(
    value: str | Path,
    *,
    session_id: str | None = None,
) -> str:
    """Return the basename/stem identity used to key ``frame_tags.csv`` rows.

    A session-qualified manifest ID is stripped only when its exact
    ``session_id__`` prefix is supplied.  Arbitrary filenames containing
    ``__`` remain distinct.
    """

    raw = str(value).strip()
    if not raw:
        raise WorkspaceError("empty frame tag identity")
    identity_path = Path(raw)
    if identity_path.is_absolute() or ".." in identity_path.parts:
        raise WorkspaceError(f"unsafe frame identity {raw!r}")
    stem = identity_path.stem.strip()
    prefix = f"{session_id}__" if session_id else None
    canonical = stem[len(prefix):] if prefix and stem.startswith(prefix) else stem
    if not canonical or canonical in {".", ".."}:
        raise WorkspaceError(f"invalid frame identity {raw!r}")
    return canonical


def load_frame_tag_overrides(session_dir: Path) -> dict[str, dict[str, str]]:
    """Load explicit per-frame tags, keyed by canonical frame identity.

    Multiple spellings such as ``000001.png`` and
    ``session__000001`` denote the same frame.  A file containing both is
    rejected instead of silently selecting one row.
    """

    tags_path = Path(session_dir) / "frame_tags.csv"
    if not tags_path.is_file():
        return {}
    result: dict[str, dict[str, str]] = {}
    declared_at: dict[str, int] = {}
    try:
        handle = tags_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise WorkspaceError(f"cannot read frame tags {tags_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        for line_number, raw_row in enumerate(reader, start=2):
            raw = {
                str(key): str(value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            identity = next(
                (
                    raw.get(key, "")
                    for key in ("frame", "filename", "image", "frame_id", "image_path")
                    if raw.get(key, "")
                ),
                "",
            )
            if not identity:
                raise WorkspaceError(f"{tags_path}:{line_number}: missing frame identity")
            try:
                canonical = canonical_frame_tag_identity(
                    identity, session_id=Path(session_dir).name)
            except WorkspaceError as exc:
                raise WorkspaceError(f"{tags_path}:{line_number}: {exc}") from exc
            if canonical in result:
                raise WorkspaceError(
                    f"{tags_path}:{line_number}: duplicate/conflicting frame tag alias "
                    f"for {canonical!r}; first declared on line {declared_at[canonical]}"
                )
            # Normalize legacy identity-column aliases before any atomic
            # rewrite so untouched rows cannot lose their identity.
            raw["frame"] = identity
            result[canonical] = raw
            declared_at[canonical] = line_number
    return result


def _normalized_effective_tag(field: str, value: Any) -> str:
    if field == "object_type":
        return normalize_object_type(value)
    return normalize_tag(field, value, allow_view=field == "view_bin")


def resolve_effective_frame_tags(
    session_metadata: Mapping[str, Any] | None,
    frame_override: Mapping[str, Any] | None = None,
    annotation_path: Path | None = None,
    *,
    annotation_document: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve effective tags and their UI/report provenance.

    Precedence is ``FRAME > JSON > SESSION > UNSET``.  Empty or ``unknown``
    frame values clear an override and fall through.  Annotation inference is
    limited to :func:`infer_annotation_tags`; in particular, distance, size,
    elevation, and view are never derived from pixels, geometry, or pose.
    """

    if annotation_path is not None and annotation_document is not None:
        raise WorkspaceError(
            "annotation_path and annotation_document are mutually exclusive"
        )
    metadata = dict(session_metadata or {})
    default_tags = metadata.get("default_tags") or {}
    if not isinstance(default_tags, Mapping):
        raise WorkspaceError("session default_tags must be an object")
    override = dict(frame_override or {})
    if annotation_document is not None:
        inferred = infer_annotation_document_tags(annotation_document)
    elif annotation_path is not None:
        inferred = infer_annotation_tags(Path(annotation_path))
    else:
        inferred = {"object_type": "unknown", "occlusion": "unknown", "truncation": "unknown"}

    effective: dict[str, str] = {}
    sources: dict[str, str] = {}
    for field in EFFECTIVE_FRAME_TAG_FIELDS:
        frame_value = _normalized_effective_tag(field, override.get(field, "unknown"))
        if frame_value != "unknown":
            effective[field] = frame_value
            sources[field] = "FRAME"
            continue

        json_value = _normalized_effective_tag(field, inferred.get(field, "unknown"))
        if json_value != "unknown":
            effective[field] = json_value
            sources[field] = "JSON"
            continue

        session_raw = (
            default_tags[field]
            if field in default_tags
            else metadata.get(field, "unknown")
        )
        session_value = _normalized_effective_tag(field, session_raw)
        if session_value != "unknown":
            effective[field] = session_value
            sources[field] = "SESSION"
            continue

        effective[field] = "unknown"
        sources[field] = "UNSET"
    return effective, sources


def _stored_frame_tag_identity(
    value: str | Path,
    *,
    session_id: str | None = None,
) -> str:
    raw = str(value).strip()
    canonical = canonical_frame_tag_identity(raw, session_id=session_id)
    suffix = Path(raw).suffix
    return f"{canonical}{suffix}" if suffix else canonical


def update_frame_tags_csv(
    session_dir: Path,
    frame_identity: str | Path,
    updates: Mapping[str, Any],
) -> Path:
    """Atomically merge explicit tags for one canonical frame.

    Fields omitted from ``updates`` are preserved.  Empty/``unknown`` values
    clear the corresponding override; a row with no remaining explicit tags
    is removed so resolution naturally falls through to JSON/session data.
    Existing duplicate canonical rows abort the operation before any write.
    """

    return update_frame_tags_csv_many(
        session_dir, {str(frame_identity): dict(updates)})


def update_frame_tags_csv_many(
    session_dir: Path,
    updates_by_frame: Mapping[str | Path, Mapping[str, Any]],
) -> Path:
    """Atomically merge tags for several frames in one session.

    The complete input is validated and merged in memory before the one CSV
    replacement.  A session-wide annotation UI action therefore cannot leave
    a half-updated file if one frame identity or value is invalid.
    """
    session_dir = Path(session_dir)
    tags_path = session_dir / "frame_tags.csv"
    rows_by_identity = load_frame_tag_overrides(session_dir)
    pending = []
    seen_canonical = set()
    for frame_identity, updates in updates_by_frame.items():
        if not isinstance(updates, Mapping):
            raise WorkspaceError(
                f"frame tag updates must be mappings: {frame_identity!s}")
        unsupported = sorted(set(updates) - set(FRAME_TAG_FIELDS))
        if unsupported:
            raise WorkspaceError(f"unsupported frame tag field(s): {unsupported}")
        canonical = canonical_frame_tag_identity(
            frame_identity, session_id=session_dir.name)
        if canonical in seen_canonical:
            raise WorkspaceError(
                "duplicate/conflicting batch frame tag alias: "
                f"{frame_identity!s} -> {canonical}")
        seen_canonical.add(canonical)
        normalized_updates = {
            field: _normalized_effective_tag(field, raw_value)
            for field, raw_value in updates.items()
        }
        pending.append((frame_identity, canonical, normalized_updates))

    for frame_identity, canonical, normalized_updates in pending:
        current = dict(rows_by_identity.get(canonical, {}))
        current["frame"] = current.get("frame") or _stored_frame_tag_identity(
            frame_identity, session_id=session_dir.name)
        for field, normalized in normalized_updates.items():
            current[field] = "" if normalized == "unknown" else normalized

        if any(str(current.get(field, "")).strip().lower() not in {"", "unknown"}
               for field in FRAME_TAG_FIELDS):
            rows_by_identity[canonical] = current
        else:
            rows_by_identity.pop(canonical, None)

    output_rows = [rows_by_identity[key] for key in sorted(rows_by_identity)]
    atomic_write_csv(tags_path, output_rows, FRAME_TAG_COLUMNS)
    return tags_path


def _workspace_readme() -> str:
    return """# Pallet real evaluation workspace v1

이 directory는 기존 source/raw/GT와 분리된 수정 가능한 evaluation working copy다.
active frame의 `DEV`, `DEV_UNVERIFIED`, `FINAL` role은 서로 섞거나 변경하지 않는다.
`FINAL_EVAL` manifest는 controlled `DEV_EVAL`을 물리 복사 없이 재사용하는 실행용
alias이며 held-out FINAL이 아니다.

## How does the tool know the condition?

- `SESSION.JSON`: object type, day/night lighting, session-wide defaults
- `ANNOTATION JSON`: annotation 완료 여부, occlusion evidence, truncation evidence
- `FRAME TAG UI`: occlusion/truncation, distance NEAR/MID/FAR, elevation override와
  annotated-session batch apply

우선순위는 `FRAME > JSON > SESSION > UNKNOWN`이다. elevation은 frame UI에서
LOW/MID/HIGH로 입력하고 distance는 NEAR/MID/FAR로 입력한다. 현재 contract에는
고정 meter threshold가 없다. 과거 CSV의 `size_bin`은 파일 호환을 위해 보존하지만
active UI, 진행률, 논문 condition 표에서는 사용하지 않는다. 그 외 view는 기존
metadata가 없으면 `unknown`으로 남는다.
조건은 dataset metadata 자체에 저장되므로 사용자가 매 frame마다 ChatGPT나 CLI에
"이 사진은 far다"라고 별도로 설명할 필요가 없다.

## Annotation SESSION selector

annotation UI의 `SESSION` 목록에는 수정 가능한 DEV 9개(플라스틱 7개 + 목재 2개)와
신규 촬영본의 zero-copy `STAGING EDIT` 4개가 함께 표시된다. DAY와 NIGHT capture가
각각 `PLASTIC`, `WOOD` 두 행으로 보인다. 각 행은 같은 raw capture를 복사하지 않고
참조하되 서로 겹치지 않는 실제 frame subset만 표시하며, object별 registry
geometry로 PnP를 푼다. raw session은 수정·이동하지 않는다.

분류 정본은 각 raw frame을 정확히 한 번 기록한
`incoming/sessions/<capture>/manifests/frame_review.csv`다. 실제 픽셀을 프레임
단위로 검수해 `plastic`, `wood`, `exclude`로 나눴으며, `exclude`(파렛트 없음,
카메라 이동, 심한 motion blur)는 두 객체 view에서 모두 숨긴다. 큰 재질 경계는
DAY source ordinal 기준 `WOOD=69..5480`, `PLASTIC=5481..24193`,
`WOOD=24241..29028`이고, NIGHT는 `WOOD=1..4849`,
`PLASTIC=4850..13583`이다. 경계 안의 검수 제외 구간까지 적용한 최종 수는 DAY
`PLASTIC 17,917 / WOOD 9,362 / EXCLUDE 1,749`, NIGHT
`PLASTIC 7,913 / WOOD 4,546 / EXCLUDE 1,124`이다.
partition view 안의 `frame N/M`과 goto는 view-local 번호다. 패널에는 원본 기준
`source ordinal N/raw_total`도 함께 표시하며, 경계 추적은 source ordinal 또는
filename을 사용한다.

PnP GT JSON, 호환 PNG, `frame_tags.csv`, `_overlays/<stem>.png`는 각각
`incoming/annotations/<capture>__plastic/` 또는
`incoming/annotations/<capture>__wood/` 아래에만 저장된다. 제공된 camera
intrinsics는 검증되지 않았으므로 GT의 `intrinsics_quality`는 `UNKNOWN`이고, 원래
`PROVIDED_UNVERIFIED` 품질과 `camera_info.json` 출처는 `intrinsics_source`에 보존한다.

staging save는 top-level `manifests/frames.csv`, DEV/FINAL 평가 manifest,
progress/report MD를 자동 갱신하거나 evaluation member를 만들지 않는다. 맞는 object
frame을 검수한 뒤 DEV/FINAL로 promotion하는 작업은 별도 절차다.

## Evaluation populations

- `DEV_EVAL_POSITIVE.csv`: controlled plastic 128 + wood 45 (173 images)
- `DEV_EVAL_NEGATIVE.csv`: frozen DEV membership 2689 rows
- `FINAL_EVAL_POSITIVE.csv`: registered DEV_EVAL positive 173행의 frozen 실행 alias
- `FINAL_EVAL_NEGATIVE.csv`: registered DEV_EVAL negative 2689행의 frozen 실행 alias
- `FINAL_{POSITIVE,NEGATIVE}.csv`: physical FINAL inventory만 포함하며 alias는 포함하지 않음
- `ALL_AVAILABLE_{POSITIVE,NEGATIVE}.csv`: DEV_EVAL + physical FINAL의 SHA256-deduplicated union
- `DEV_PLASTIC_AUDITED140.csv`: FT-overlap 12장을 포함한 review population

`FINAL_EVAL` alias row는 원래 `population_role=DEV`를 유지하고 notes에
`REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV`를 기록한다. `FINAL_EVAL`과
`ALL_AVAILABLE` 어느 것도 held-out FINAL로 부르지 않는다. 현재 evaluation은
이미 준비되었으며 새 annotation은 필수가 아니다. `DATASET_TARGETS.json`은
DEV_EVAL과 physical FINAL을 함께 세는 대략적 evaluation collection 목표다.
현재 DEV positive와 negative image는 모두 `dev_existing/sessions/` 아래의 독립
복사본이다. 원본 raw/GT와 source SHA provenance는 그대로 보존한다.

## Evaluator binding

현재 paper evaluator는 workspace CSV를 다시 `population_role == FINAL`로 거르지
않는다. 동일한 controlled membership의 기존 frozen DEV manifest를 직접 입력한다.

```bash
python challenge/evaluation_v2/paper_real_eval.py \\
  --positive-manifest challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json \\
  --negative-manifest challenge/real_gt_v2/manifests/DEV_NEG2689.json \\
  --population-role DEV \\
  --weights <checkpoint.pt> --out <result.json>
```

positive membership은 workspace 실행 alias와 같은 173행이다. negative도 같은
frozen 2,689행이며 그 안의 unique image는 2,688장이다. physical FINAL은 이 실행
alias에 자동으로 섞지 않는다. 이 평가에 `--population-role FINAL`을 사용하지 않는다.
등록된 pair SHA256은
`2cfa7011d8ba3677b11019c103e2ccbaeeac53521c9291ed632f94c8d2c5c887`이다.
AP/AUROC/FPR95 score pipeline도 같은 173/2,689 membership을 사용해야 한다.
현재 그 score pipeline과 workspace condition tag를 pair SHA에 묶은 통합 artifact는
아직 없으므로 해당 ranking/condition metric은 생성 전까지 `—`로 둔다.

## Workflow

```bash
# 선택: destination을 만들지 않는 source audit
python scripts/evaluation/import_existing_evaluation_data.py \\
  --root data/evaluation/pallet_eval_v1 --audit-only

# A. 기존 데이터 비파괴 import
python scripts/evaluation/import_existing_evaluation_data.py \\
  --root data/evaluation/pallet_eval_v1

# B. 기존 overlay 재생성
python scripts/annotate/rebuild_annotation_overlays.py \\
  --dataset-root data/evaluation/pallet_eval_v1 \\
  --scope dev_existing --force

# C. 상태/manifest/report 갱신
python scripts/evaluation/eval_dataset_status.py \\
  --root data/evaluation/pallet_eval_v1

# D/E. 새 FINAL image 배치 후 session.json 작성 (frame_tags.csv는 UI가 갱신)
# final/positive/sessions/<session>/rgb/*.png

# F. 기존 DEV 또는 plastic FINAL annotation
python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/dev_existing/sessions/<session> \\
  --out_dir data/evaluation/pallet_eval_v1/dev_existing/annotations/<session> \\
  --default_split eval \\
  --eval-root data/evaluation/pallet_eval_v1

python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<session> \\
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<session> \\
  --default_split eval \\
  --eval-root data/evaluation/pallet_eval_v1

# Wood geometry는 session에서 자동 결정하며, session에 없으면 intrinsics provenance만 명시한다.
python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<wood_session> \\
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<wood_session> \\
  --default_split eval \\
  --intrinsics-quality CALIBRATED --intrinsics-source '<calibration artifact>' \\
  --eval-root data/evaluation/pallet_eval_v1
```

G. 선택적으로 physical FINAL을 확장할 때 매 save마다 JSON,
`_overlays/<stem>.png`, `manifests/frames.csv`, report가 갱신된다.
`reports/NEXT_ANNOTATION_PRIORITY.md`는 새 annotation을 요구하지 않고 현재
DEV_EVAL과 physical FINAL을 합친 `ALL_AVAILABLE` 목표 진행률을 보여준다.
`0/300` 같은 수치는 DEV/FINAL로 나누지 않고 이 combined population에서 계산한다.

선택적으로 새 FINAL 촬영을 추가할 때만 `final/positive/sessions/<session>/rgb/` 또는
`final/negative/sessions/<session>/rgb/`에 둔다. 각 session에 `session.json`을
작성한다. frame별 수동 tag는 annotation UI에서 `/`를 눌러 전용 `CONDITIONS`
모드에 들어간 뒤 `1=occlusion ON/OFF`, `2=truncation ON/OFF`,
`3=LOW`, `4=MID`, `5=HIGH` elevation, `n=NEAR`, `m=MID`, `6=FAR`를
지정한다. 누를 때마다 화면 값이 즉시 바뀐다. `u`는 distance 수동 tag를
UNKNOWN/default로 되돌린다.
현재 frame에서 방금 바꾼 항목만 같은 session의 annotation JSON이 있는
frame에 일괄 적용하려면 `a`를 두 번 누른다. 미어노테이션 frame은 제외하고
다른 기존 tag는 보존한다. legacy size와 view는 annotation 화면에서 선택하도록
요구하지 않는다. `s`로 annotation과 tag를 함께 저장하므로
`frame_tags.csv`를 직접 편집할 필요가 없다.
현재 session의 annotated frame 전체에서 distance tag만 지우려면
`u`, `a`, `a` 순서로 누른다.

positive/negative 또는 plastic/wood가 섞인 연속 촬영본은 곧바로 FINAL에 넣지 않는다.
`scripts/evaluation/import_incoming_capture.py`로 `incoming/sessions/`에 먼저
비파괴 import한다. raw capture는 `INCOMING_UNREVIEWED`로 유지하면서 SESSION의
object별 zero-copy `STAGING EDIT` 행에서 annotation한다. staging output은
`incoming/annotations/`에만 쓰며 DEV/FINAL 평가와 combined 목표 수치에 자동으로
포함되지 않는다. 검수한 frame의 promotion과 평가 활성화는 별도로 수행한다.

`far`, `elevation`, `view`는 임의 threshold로 추정하지 않는다. 명시하지
않은 값은 `unknown`으로 남는다. 이 값은 DEV alias의 provenance를 그대로 설명할
뿐 새 annotation이나 metadata 보완을 의무화하지 않는다.
"""


def scaffold_workspace(
    root: Path,
    *,
    geometry_registry: Path | None = None,
    overwrite_contract_files: bool = False,
) -> None:
    root = root.resolve()
    directories = (
        "objects",
        "dev_existing/sessions",
        "dev_existing/annotations",
        "legacy_unverified/sessions",
        "legacy_unverified/annotations",
        "final/positive/sessions",
        "final/positive/annotations",
        "final/negative/sessions",
        "manifests",
        "reports",
    )
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)

    positive_templates = (
        ("plastic_day_01", "plastic", "day", "360deg_multi_height_multi_distance", "none", "none"),
        ("plastic_night_01", "plastic", "night", "360deg_multi_height_multi_distance", "none", "none"),
        ("plastic_occ_01", "plastic", "day", "partial_occlusion", "medium", "none"),
        ("plastic_trunc_01", "plastic", "day", "edge_truncation", "none", "medium"),
        ("wood_day_01", "wood", "day", "360deg_multi_height_multi_distance", "none", "none"),
        ("wood_night_01", "wood", "night", "360deg_multi_height_multi_distance", "none", "none"),
        ("wood_occ_01", "wood", "day", "partial_occlusion", "medium", "none"),
        ("wood_trunc_01", "wood", "day", "edge_truncation", "none", "medium"),
    )
    for session_id, object_type, lighting, protocol, occlusion, truncation in positive_templates:
        session = root / "final/positive/sessions" / session_id
        (session / "rgb").mkdir(parents=True, exist_ok=True)
        metadata = session / "session.json"
        if not metadata.exists():
            atomic_write_json(
                metadata,
                {
                    "session_id": session_id,
                    "population_role": "FINAL",
                    "object_type": object_type,
                    "lighting": lighting,
                    "capture_protocol": protocol,
                    "camera": "RealSense D435I",
                    "resolution": [1280, 720],
                    "default_tags": {
                        "occlusion": occlusion,
                        "truncation": truncation,
                    },
                },
            )
        tags = session / "frame_tags.csv"
        if not tags.exists():
            atomic_write_text(
                tags,
                "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n",
            )

    for session_id, lighting in (("negative_day_01", "day"), ("negative_night_01", "night")):
        session = root / "final/negative/sessions" / session_id
        (session / "rgb").mkdir(parents=True, exist_ok=True)
        metadata = session / "session.json"
        if not metadata.exists():
            atomic_write_json(
                metadata,
                {
                    "session_id": session_id,
                    "population_role": "FINAL",
                    "object_type": "none",
                    "lighting": lighting,
                    "capture_protocol": "negative_background",
                    "camera": "RealSense D435I",
                    "resolution": [1280, 720],
                    "default_tags": {},
                },
            )

    static_files: tuple[tuple[Path, str], ...] = (
        (root / "README.md", _workspace_readme()),
        (
            root / "DATASET_CONTRACT.json",
            json.dumps(DATASET_CONTRACT, ensure_ascii=False, indent=2) + "\n",
        ),
        (
            root / "DATASET_TARGETS.json",
            json.dumps(DEFAULT_TARGETS, ensure_ascii=False, indent=2) + "\n",
        ),
    )
    for path, content in static_files:
        if overwrite_contract_files or not path.exists():
            atomic_write_text(path, content)

    snapshot = root / "objects/OBJECT_GEOMETRY_REGISTRY.snapshot.json"
    if geometry_registry is not None and geometry_registry.is_file():
        if overwrite_contract_files or not snapshot.exists():
            copy2_verified(geometry_registry, snapshot)
    elif not snapshot.exists():
        atomic_write_json(
            snapshot,
            {
                "schema_version": "pallet_pose_object_geometry_registry_snapshot_v1",
                "status": "SOURCE_NOT_PROVIDED",
                "objects": [],
            },
        )

    for name, columns in (
        ("frames.csv", FRAME_COLUMNS),
        ("sessions.csv", SESSION_COLUMNS),
        ("import_provenance.csv", PROVENANCE_COLUMNS),
    ):
        path = root / "manifests" / name
        if not path.exists():
            initial_rows = _session_rows(root, []) if name == "sessions.csv" else []
            atomic_write_csv(path, initial_rows, columns)
    import_audit = root / "reports/IMPORT_AUDIT.md"
    if not import_audit.exists():
        atomic_write_text(
            import_audit,
            "# Import audit\n\nStatus: `NOT_RUN`\n\n"
            "`import_existing_evaluation_data.py`가 성공하면 이 파일을 교체한다.\n",
        )


def load_targets(root: Path) -> dict[str, Any]:
    path = root / "DATASET_TARGETS.json"
    if not path.is_file():
        return json.loads(json.dumps(DEFAULT_TARGETS))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkspaceError(f"targets must be a JSON object: {path}")
    # PAPER_EVAL 은 ALL_AVAILABLE 과 같은 집합의 paper-facing 이름이다
    # (SHA-deduplicated union(DEV_EVAL, NEW_EVAL), held_out_final=false).
    # 옛 이름도 계속 받는다.
    if value.get("progress_population") not in {"ALL_AVAILABLE", "PAPER_EVAL"}:
        raise WorkspaceError(
            "targets.progress_population must be PAPER_EVAL (or legacy "
            "ALL_AVAILABLE) so DEV_EVAL and physical FINAL are counted together"
        )
    coverage = value.get("minimum_condition_coverage")
    if not isinstance(coverage, dict):
        raise WorkspaceError(
            f"targets.minimum_condition_coverage must be an object: {path}")
    legacy_far = coverage.get("far_small")
    if "far" not in coverage and legacy_far is not None:
        coverage["far"] = legacy_far
        coverage.pop("far_small", None)
    elif "far" in coverage and legacy_far is not None:
        if coverage["far"] != legacy_far:
            raise WorkspaceError(
                "targets contain conflicting far and legacy far_small values")
        coverage.pop("far_small", None)
    if "far" not in coverage:
        raise WorkspaceError("targets.minimum_condition_coverage.far is required")
    return value


def load_frames(root: Path) -> list[dict[str, str]]:
    path = root / "manifests/frames.csv"
    if not path.is_file():
        raise WorkspaceError(f"frames manifest missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        missing_headers = [field for field in FRAME_COLUMNS if field not in headers]
        if missing_headers:
            raise WorkspaceError(f"frames.csv missing required columns: {missing_headers}")
        rows = [dict(row) for row in reader]
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        frame_id = row.get("frame_id", "")
        if not frame_id:
            raise WorkspaceError(f"frames.csv:{index}: empty frame_id")
        if "/" in frame_id or "\\" in frame_id:
            raise WorkspaceError(f"frames.csv:{index}: invalid frame_id {frame_id!r}")
        if frame_id in seen:
            raise WorkspaceError(f"frames.csv:{index}: duplicate frame_id {frame_id!r}")
        seen.add(frame_id)
        for field, allowed in ALLOWED_VALUES.items():
            value = row.get(field, "")
            # 새로 생긴 도메인 축은 옛 manifest 에 컬럼 자체가 없다.  빈 값을
            # unknown 으로 정규화한다 — 값을 추론하는 것이 아니라 "미상" 을 명시하는 것이다.
            if field in NEW_DOMAIN_FIELDS and value == "":
                value = "unknown"
                row[field] = value
            if value not in allowed:
                raise WorkspaceError(
                    f"frames.csv:{index}: invalid {field}={value!r}; allowed={sorted(allowed)}"
                )
        if row.get("paper_subset") not in PAPER_SUBSETS:
            raise WorkspaceError(
                f"frames.csv:{index}: invalid paper_subset={row.get('paper_subset')!r}"
            )
        for field in BOOLEAN_FIELDS:
            if row.get(field) not in {"true", "false"}:
                raise WorkspaceError(
                    f"frames.csv:{index}: {field} must be lowercase true or false"
                )
        if not row.get("session_id"):
            raise WorkspaceError(f"frames.csv:{index}: empty session_id")
        if not row.get("view_bin"):
            raise WorkspaceError(f"frames.csv:{index}: empty view_bin")
        if not row.get("source_dataset"):
            raise WorkspaceError(f"frames.csv:{index}: empty source_dataset")
        if row.get("storage_mode") not in STORAGE_MODES:
            raise WorkspaceError(
                f"frames.csv:{index}: invalid storage_mode={row.get('storage_mode')!r}"
            )
        for field in (
            "image_path",
            "annotation_path",
            "overlay_path",
            "source_image_path",
            "source_annotation_path",
        ):
            value = row.get(field, "")
            if not value:
                continue
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise WorkspaceError(
                    f"frames.csv:{index}: {field} must be a safe relative path: {value!r}"
                )
        if not row.get("image_path"):
            raise WorkspaceError(f"frames.csv:{index}: empty image_path")
        if not SHA256_RE.fullmatch(row.get("image_sha256", "")):
            raise WorkspaceError(f"frames.csv:{index}: invalid image_sha256")
        for field in (
            "annotation_sha256",
            "source_image_sha256",
            "source_annotation_sha256",
        ):
            value = row.get(field, "")
            if value and not SHA256_RE.fullmatch(value):
                raise WorkspaceError(f"frames.csv:{index}: invalid {field}")
        positive = row["is_positive"] == "true"
        annotated = row["is_annotated"] == "true"
        if positive and (not row.get("annotation_path") or not row.get("overlay_path")):
            raise WorkspaceError(
                f"frames.csv:{index}: positive frame requires annotation_path and overlay_path"
            )
        if annotated and (
            not row.get("annotation_path")
            or not SHA256_RE.fullmatch(row.get("annotation_sha256", ""))
        ):
            raise WorkspaceError(
                f"frames.csv:{index}: annotated frame requires annotation path and SHA"
            )
        if not annotated and row.get("annotation_sha256"):
            raise WorkspaceError(
                f"frames.csv:{index}: unannotated frame must not retain annotation_sha256"
            )
        role = row["population_role"]
        if role == "FINAL":
            expected_subset = "FINAL_POSITIVE" if positive else "FINAL_NEGATIVE"
            if row["paper_subset"] != expected_subset:
                raise WorkspaceError(
                    f"frames.csv:{index}: FINAL polarity/subset mismatch"
                )
        if role == "DEV_UNVERIFIED" and row["paper_subset"] != "NONE":
            raise WorkspaceError(
                f"frames.csv:{index}: DEV_UNVERIFIED must use paper_subset=NONE"
            )
    validate_active_image_sha_uniqueness(rows)
    return rows


def _is_exact_known_negative_duplicate(rows: Sequence[Mapping[str, str]]) -> bool:
    if len(rows) != 2:
        return False
    by_id = {row.get("frame_id", ""): row for row in rows}
    if set(by_id) != set(KNOWN_DEV_NEG_DUPLICATE_PATHS):
        return False
    for frame_id, expected_source_path in KNOWN_DEV_NEG_DUPLICATE_PATHS.items():
        row = by_id[frame_id]
        source_reference = (
            row.get("storage_mode") == "source_reference_read_only"
            and row.get("image_path") == expected_source_path
        )
        independent_copy = (
            row.get("storage_mode") == "independent_copy"
            and row.get("image_path")
            == KNOWN_DEV_NEG_DUPLICATE_WORKSPACE_PATHS[frame_id]
        )
        if (
            row.get("image_sha256") != KNOWN_DEV_NEG_DUPLICATE_SHA256
            or row.get("population_role") != "DEV"
            or row.get("paper_subset") != "DEV_NEG2689"
            or row.get("controlled_eval_eligible") != "true"
            or row.get("cross_shape_eval_eligible") != "false"
            or row.get("is_positive") != "false"
            or row.get("is_annotated") != "false"
            or row.get("object_type") != "none"
            or row.get("source_dataset") != "real_gt_v2_negative_audited"
            or not (source_reference or independent_copy)
            or row.get("source_image_path") != expected_source_path
            or row.get("source_image_sha256") != KNOWN_DEV_NEG_DUPLICATE_SHA256
            or row.get("annotation_path")
            or row.get("overlay_path")
        ):
            return False
    return True


def validate_active_image_sha_uniqueness(rows: Sequence[Mapping[str, str]]) -> None:
    """Reject copied/promoted active frames, except one frozen negative pair."""

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        digest = row.get("image_sha256", "")
        if digest:
            grouped[digest].append(row)
    invalid = []
    for digest, members in grouped.items():
        if len(members) < 2:
            continue
        if digest == KNOWN_DEV_NEG_DUPLICATE_SHA256 and _is_exact_known_negative_duplicate(members):
            continue
        invalid.append((digest, [row.get("frame_id", "") for row in members]))
    if invalid:
        digest, frame_ids = invalid[0]
        raise WorkspaceError(
            "duplicate active image SHA (DEV/FINAL copies or promotions are forbidden): "
            f"sha256={digest}, frame_ids={frame_ids}"
        )


def _read_session_tags(session_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Compatibility wrapper for callers predating the public tag helpers."""

    return load_session_metadata(session_dir), load_frame_tag_overrides(session_dir)


def _image_files(rgb_dir: Path) -> list[Path]:
    if not rgb_dir.is_dir():
        return []
    return sorted(
        path
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_final_rows(
    root: Path,
    existing_rows: Sequence[Mapping[str, str]] = (),
    *,
    rehash: bool = False,
) -> list[dict[str, str]]:
    """Discover FINAL frames without treating an unannotated positive as complete."""

    # Kept in the signature for API compatibility.  FINAL is intentionally
    # rehashed on every routine refresh: a changed RGB must update frames.csv
    # and participate in the global DEV/FINAL duplicate guard immediately.
    del existing_rows, rehash
    result: list[dict[str, str]] = []
    seen_frame_ids: set[str] = set()
    for is_positive, relative in (
        (True, Path("final/positive/sessions")),
        (False, Path("final/negative/sessions")),
    ):
        sessions_root = root / relative
        if not sessions_root.is_dir():
            continue
        for session_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            metadata, frame_tags = _read_session_tags(session_dir)
            session_id = safe_component(str(metadata.get("session_id") or session_dir.name))
            if session_id != safe_component(session_dir.name):
                raise WorkspaceError(
                    f"session_id {session_id!r} does not match directory {session_dir.name!r}"
                )
            role = str(metadata.get("population_role", "FINAL")).strip().upper()
            if role != "FINAL":
                raise WorkspaceError(f"FINAL session declares population_role={role!r}: {session_dir}")
            for image_path in _image_files(session_dir / "rgb"):
                stem = image_path.stem
                frame_id = f"{session_id}__{safe_component(stem, fallback='frame')}"
                if frame_id in seen_frame_ids:
                    raise WorkspaceError(f"duplicate FINAL frame_id {frame_id!r}")
                seen_frame_ids.add(frame_id)
                override = frame_tags.get(canonical_frame_tag_identity(
                    image_path.name, session_id=session_dir.name), {})

                annotation_path = (
                    root / "final/positive/annotations" / session_id / f"{stem}.json"
                    if is_positive
                    else None
                )
                overlay_path = (
                    root / "final/positive/annotations" / session_id / "_overlays" / f"{stem}.png"
                    if is_positive
                    else None
                )
                effective_tags, _tag_sources = resolve_effective_frame_tags(
                    metadata,
                    override,
                    annotation_path,
                )
                object_type = effective_tags["object_type"]
                if not is_positive:
                    object_type = "none"

                image_relative = workspace_relative(image_path, root)
                image_hash = sha256_file(image_path)
                annotation_hash = (
                    sha256_file(annotation_path)
                    if annotation_path is not None and annotation_path.is_file()
                    else ""
                )
                result.append(
                    {
                        "frame_id": frame_id,
                        "population_role": "FINAL",
                        "paper_subset": "FINAL_POSITIVE" if is_positive else "FINAL_NEGATIVE",
                        "controlled_eval_eligible": "true",
                        "cross_shape_eval_eligible": "true" if is_positive else "false",
                        "exclusion_reason": "",
                        "session_id": session_id,
                        "object_type": object_type,
                        "lighting": effective_tags["lighting"],
                        "occlusion": effective_tags["occlusion"] if is_positive else "unknown",
                        "truncation": effective_tags["truncation"] if is_positive else "unknown",
                        "distance_bin": effective_tags["distance_bin"],
                        "size_bin": effective_tags["size_bin"],
                        "elevation_bin": effective_tags["elevation_bin"],
                        "view_bin": effective_tags["view_bin"],
                        "is_positive": bool_text(is_positive),
                        "is_annotated": bool_text(bool(annotation_hash)),
                        "image_path": image_relative,
                        "annotation_path": workspace_relative(annotation_path, root) if annotation_path else "",
                        "overlay_path": workspace_relative(overlay_path, root) if overlay_path else "",
                        "source_dataset": "workspace_final",
                        "source_image_path": "",
                        "source_annotation_path": "",
                        "image_sha256": image_hash,
                        "annotation_sha256": annotation_hash,
                        "source_image_sha256": "",
                        "source_annotation_sha256": "",
                        "storage_mode": "workspace_native",
                        "notes": "",
                    }
                )
    return result


def reconcile_annotation_state(root: Path, row: Mapping[str, str]) -> dict[str, str]:
    result = {key: str(row.get(key, "")) for key in FRAME_COLUMNS}
    annotation = resolve_workspace_path(root, result.get("annotation_path", ""))
    if annotation is not None:
        exists = annotation.is_file()
        result["is_annotated"] = bool_text(exists)
        result["annotation_sha256"] = sha256_file(annotation) if exists else ""
    role = result.get("population_role", "")
    session_id = result.get("session_id", "")
    if safe_component(session_id) != session_id:
        raise WorkspaceError(f"unsafe session_id in frames.csv: {session_id!r}")
    if role == "DEV":
        session_dir = root / "dev_existing/sessions" / session_id
    elif role == "DEV_UNVERIFIED":
        session_dir = root / "legacy_unverified/sessions" / session_id
    else:
        session_dir = None
    if session_dir is not None and (session_dir / "session.json").is_file():
        metadata = load_session_metadata(session_dir)
        overrides = load_frame_tag_overrides(session_dir)
        image_value = result.get("image_path") or result.get("source_image_path")
        identity = Path(image_value).name if image_value else result.get("frame_id", "")
        override = overrides.get(canonical_frame_tag_identity(
            identity, session_id=session_dir.name), {})
        effective, _sources = resolve_effective_frame_tags(
            metadata,
            override,
            annotation if annotation is not None and annotation.is_file() else None,
        )
        for field in EFFECTIVE_FRAME_TAG_FIELDS:
            result[field] = effective[field]
    elif annotation is not None and annotation.is_file():
        # Read-only source-reference populations may not have workspace
        # session metadata.  Preserve their historical JSON-only refresh.
        inferred = infer_annotation_tags(annotation)
        for field in ("object_type", "occlusion", "truncation"):
            if inferred[field] != "unknown":
                result[field] = inferred[field]
    return result


def refresh_frame_index(root: Path, *, rehash_final: bool = False) -> list[dict[str, str]]:
    del rehash_final  # Compatibility only; FINAL RGB is always rehashed.
    existing = load_frames(root)
    non_final = [
        reconcile_annotation_state(root, row)
        for row in existing
        if row.get("population_role") != "FINAL"
    ]
    final_rows = discover_final_rows(root, existing, rehash=True)
    rows = sorted(non_final + final_rows, key=lambda row: row["frame_id"])
    ids = [row["frame_id"] for row in rows]
    if len(ids) != len(set(ids)):
        repeated = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise WorkspaceError(f"global duplicate frame_id(s): {repeated[:10]}")
    validate_active_image_sha_uniqueness(rows)
    # Imported workspaces carry provenance rows.  Once the frozen DEV import
    # has happened, every routine refresh continues to enforce its exact
    # audited membership; empty test/capture-only workspaces remain usable.
    imported_workspace = bool(read_csv(root / "manifests/import_provenance.csv"))
    populations = evaluation_population_views(rows)
    if imported_workspace:
        validate_frozen_dev_evaluation_population(populations)
    atomic_write_csv(root / "manifests/frames.csv", rows, FRAME_COLUMNS)
    write_manifest_views(root, rows, enforce_frozen_dev=imported_workspace)
    return rows


def _session_rows(root: Path, frames: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in frames:
        grouped[(row.get("population_role", ""), row.get("session_id", ""), row.get("is_positive", ""))].append(row)
    result: list[dict[str, str]] = []
    for (role, session, positive), members in sorted(grouped.items()):
        object_values = {row.get("object_type", "unknown") for row in members}
        lighting_values = {row.get("lighting", "unknown") for row in members}
        first = members[0]
        image_value = first.get("image_path", "")
        source_reference = all(
            row.get("storage_mode") == "source_reference_read_only" for row in members
        )
        if source_reference:
            session_path = str(Path(image_value).parent) if image_value else ""
            metadata_path = ""
        elif role == "FINAL":
            base = "final/positive/sessions" if is_true(positive) else "final/negative/sessions"
            session_path = f"{base}/{session}"
            metadata_path = f"{session_path}/session.json"
        elif role == "DEV_UNVERIFIED":
            base = "legacy_unverified/sessions"
            session_path = f"{base}/{session}"
            metadata_path = f"{session_path}/session.json"
        else:
            base = "dev_existing/sessions"
            session_path = f"{base}/{session}"
            metadata_path = f"{session_path}/session.json"
        result.append(
            {
                "session_id": session,
                "population_role": role,
                "object_type": next(iter(object_values)) if len(object_values) == 1 else "mixed",
                "lighting": next(iter(lighting_values)) if len(lighting_values) == 1 else "mixed",
                "is_positive": bool_text(positive),
                "frame_count": str(len(members)),
                "annotated_count": str(sum(is_true(row.get("is_annotated")) for row in members)),
                "session_path": session_path,
                "session_metadata_path": metadata_path,
                "source_dataset": first.get("source_dataset", ""),
                "notes": "" if image_value else "IMAGE_PATH_MISSING",
            }
        )
    existing = {
        (row["population_role"], row["session_id"], row["is_positive"])
        for row in result
    }
    for positive, relative in (
        (True, Path("final/positive/sessions")),
        (False, Path("final/negative/sessions")),
    ):
        sessions_root = root / relative
        if not sessions_root.is_dir():
            continue
        for session_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            metadata_path = session_dir / "session.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = safe_component(str(metadata.get("session_id") or session_dir.name))
            key = ("FINAL", session_id, bool_text(positive))
            if key in existing:
                continue
            result.append(
                {
                    "session_id": session_id,
                    "population_role": "FINAL",
                    "object_type": normalize_object_type(metadata.get("object_type")),
                    "lighting": str(metadata.get("lighting", "unknown")).lower(),
                    "is_positive": bool_text(positive),
                    "frame_count": "0",
                    "annotated_count": "0",
                    "session_path": workspace_relative(session_dir, root),
                    "session_metadata_path": workspace_relative(metadata_path, root),
                    "source_dataset": "workspace_final",
                    "notes": "EMPTY_SESSION_TEMPLATE",
                }
            )
    return sorted(
        result,
        key=lambda row: (row["population_role"], row["session_id"], row["is_positive"]),
    )


def _sorted_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return sorted(rows, key=lambda row: row.get("frame_id", ""))


def _require_population_sha(
    rows: Sequence[Mapping[str, str]],
    *,
    population: str,
) -> None:
    for row in rows:
        digest = row.get("image_sha256", "")
        if not SHA256_RE.fullmatch(digest):
            raise WorkspaceError(
                f"{population}: frame {row.get('frame_id', '')!r} has no valid image SHA256"
            )


def _require_unique_population_sha(
    rows: Sequence[Mapping[str, str]],
    *,
    population: str,
) -> None:
    _require_population_sha(rows, population=population)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["image_sha256"]].append(row.get("frame_id", ""))
    duplicate = next(
        ((digest, ids) for digest, ids in grouped.items() if len(ids) > 1),
        None,
    )
    if duplicate is not None:
        digest, frame_ids = duplicate
        raise WorkspaceError(
            f"{population}: duplicate image SHA256 is not allowed: "
            f"sha256={digest}, frame_ids={frame_ids}"
        )


def _sha_deduplicated_union(
    rows: Sequence[Mapping[str, str]],
    *,
    population: str,
) -> list[Mapping[str, str]]:
    """Return one deterministic row per image identity for a convenience union."""

    _require_population_sha(rows, population=population)
    by_sha: dict[str, Mapping[str, str]] = {}
    for row in _sorted_rows(rows):
        by_sha.setdefault(row["image_sha256"], row)
    return _sorted_rows(by_sha.values())


def _reused_dev_eval_alias_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    population: str,
) -> list[Mapping[str, str]]:
    """Create manifest-only DEV aliases without changing active membership.

    The copied dictionaries retain ``population_role=DEV`` and every physical
    path.  Only the provenance note changes.  Consequently these rows can be
    written to ``FINAL_EVAL_*.csv`` while ``frames.csv`` and the physical
    ``FINAL_{POSITIVE,NEGATIVE}.csv`` inventories remain untouched.
    """

    aliases: list[Mapping[str, str]] = []
    for row in rows:
        if row.get("population_role") != "DEV":
            raise WorkspaceError(
                f"{population}: reused alias source is not DEV: "
                f"{row.get('frame_id', '')}"
            )
        alias = dict(row)
        alias["notes"] = REUSED_DEV_EVAL_ALIAS_NOTE
        aliases.append(alias)
    return _sorted_rows(aliases)


def evaluation_population_views(
    frames: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    """Build DEV, manifest-only FINAL_EVAL aliases, and convenience views.

``controlled_eval_eligible`` is the existing frame-level QA eligibility
gate. ``FINAL_EVAL`` is the row-for-row, non-held-out execution alias of the
registered ``DEV_EVAL`` pair. Physical FINAL rows remain separate and enter
only their physical inventories plus ``ALL_AVAILABLE``. The frozen negative
execution membership retains its one known duplicate image; convenience
``ALL_AVAILABLE`` views are SHA-deduplicated.
    """

    audited_plastic = _sorted_rows(
        row
        for row in frames
        if row.get("paper_subset") in DEV_PLASTIC_AUDITED_SUBSETS
    )
    dev_positive_members = _sorted_rows(
        row for row in frames if row.get("paper_subset") in DEV_EVAL_POSITIVE_SUBSETS
    )
    dev_negative = _sorted_rows(
        row for row in frames if row.get("paper_subset") == "DEV_NEG2689"
    )

    # A malformed frozen row must not disappear silently merely because it no
    # longer satisfies a selector.  Validate the declared subsets first.
    for row in audited_plastic:
        if (
            row.get("population_role") != "DEV"
            or not is_true(row.get("is_positive"))
            or row.get("object_type") != "plastic"
        ):
            raise WorkspaceError(
                "DEV_PLASTIC_AUDITED140 contains a row with invalid role/polarity/object: "
                f"{row.get('frame_id', '')}"
            )
    for row in dev_positive_members:
        if (
            row.get("population_role") != "DEV"
            or not is_true(row.get("is_positive"))
            or not is_true(row.get("controlled_eval_eligible"))
        ):
            raise WorkspaceError(
                "DEV_EVAL_POSITIVE contains an ineligible declared member: "
                f"{row.get('frame_id', '')}"
            )
    for row in dev_negative:
        if (
            row.get("population_role") != "DEV"
            or is_true(row.get("is_positive"))
            or not is_true(row.get("controlled_eval_eligible"))
        ):
            raise WorkspaceError(
                "DEV_EVAL_NEGATIVE contains an ineligible declared member: "
                f"{row.get('frame_id', '')}"
            )

    _require_unique_population_sha(
        audited_plastic,
        population="DEV_PLASTIC_AUDITED140",
    )
    _require_unique_population_sha(
        dev_positive_members,
        population="DEV_EVAL_POSITIVE",
    )
    _require_population_sha(dev_negative, population="DEV_EVAL_NEGATIVE")

    physical_final_positive = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("is_annotated"))
        and is_true(row.get("controlled_eval_eligible"))
    )
    physical_final_negative = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and not is_true(row.get("is_positive"))
    )
    _require_unique_population_sha(
        physical_final_positive,
        population="PHYSICAL_FINAL_EVAL_POSITIVE",
    )
    _require_unique_population_sha(
        physical_final_negative,
        population="PHYSICAL_FINAL_EVAL_NEGATIVE",
    )

    reused_dev_positive = _reused_dev_eval_alias_rows(
        dev_positive_members,
        population="FINAL_EVAL_POSITIVE",
    )
    reused_dev_negative = _reused_dev_eval_alias_rows(
        dev_negative,
        population="FINAL_EVAL_NEGATIVE",
    )
    final_eval_positive = reused_dev_positive
    final_eval_negative = reused_dev_negative

    all_positive = _sha_deduplicated_union(
        dev_positive_members + physical_final_positive,
        population="ALL_AVAILABLE_POSITIVE",
    )
    all_negative = _sha_deduplicated_union(
        dev_negative + physical_final_negative,
        population="ALL_AVAILABLE_NEGATIVE",
    )
    positive_sha = {row["image_sha256"] for row in all_positive}
    negative_sha = {row["image_sha256"] for row in all_negative}
    overlap = sorted(positive_sha & negative_sha)
    if overlap:
        raise WorkspaceError(
            "evaluation positive/negative populations overlap by image SHA256: "
            f"{overlap[:5]}"
        )

    return {
        "DEV_PLASTIC_AUDITED140": audited_plastic,
        "DEV_EVAL_POSITIVE": dev_positive_members,
        "DEV_EVAL_NEGATIVE": dev_negative,
        # FINAL_EVAL 은 legacy 이름이다.  "held-out FINAL" 로 읽히지만 실제로는
        # DEV_EVAL 재사용이라 paper-facing 문서에서 혼동을 준다.  정식 이름은
        # PAPER_EVAL 이고, 아래 두 키는 manifest 파일명 하위호환으로만 남긴다.
        "FINAL_EVAL_POSITIVE": final_eval_positive,
        "FINAL_EVAL_NEGATIVE": final_eval_negative,
        # PAPER_EVAL = SHA-deduplicated union(DEV_EVAL, NEW_EVAL).
        # held_out_final = false.  진짜 untouched test 를 나중에 만들면
        # HELDOUT_EVAL 이라는 별도 이름을 쓴다.
        "PAPER_EVAL_POSITIVE": all_positive,
        "PAPER_EVAL_NEGATIVE": all_negative,
        "ALL_AVAILABLE_POSITIVE": all_positive,
        "ALL_AVAILABLE_NEGATIVE": all_negative,
    }


def validate_frozen_dev_evaluation_population(
    views: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Fail closed if the imported audited DEV contract has drifted."""

    expected = {
        "DEV_PLASTIC_AUDITED140": 140,
        "DEV_EVAL_POSITIVE": 173,
        "DEV_EVAL_NEGATIVE": 2689,
    }
    for name, count in expected.items():
        actual = len(views[name])
        if actual != count:
            raise WorkspaceError(f"{name}: expected {count} rows, found {actual}")

    controlled = sum(
        row.get("paper_subset") == "COMMON_DEV_PLASTIC_POS128"
        for row in views["DEV_EVAL_POSITIVE"]
    )
    wood = sum(
        row.get("paper_subset") == "DEV_WOOD_POS45"
        for row in views["DEV_EVAL_POSITIVE"]
    )
    excluded = sum(
        row.get("paper_subset") == "DEV_PLASTIC_EXTRA"
        for row in views["DEV_PLASTIC_AUDITED140"]
    )
    if (controlled, wood, excluded) != (128, 45, 12):
        raise WorkspaceError(
            "frozen DEV subset counts drifted: "
            f"plastic_controlled={controlled}, wood={wood}, plastic_excluded={excluded}"
        )

    negative_unique = len(
        {row["image_sha256"] for row in views["DEV_EVAL_NEGATIVE"]}
    )
    if negative_unique != 2688:
        raise WorkspaceError(
            "DEV_EVAL_NEGATIVE unique SHA count drifted: "
            f"expected 2688, found {negative_unique}"
        )

    for population in ("FINAL_EVAL_POSITIVE", "FINAL_EVAL_NEGATIVE"):
        aliases = [
            row
            for row in views[population]
            if row.get("notes") == REUSED_DEV_EVAL_ALIAS_NOTE
        ]
        expected_alias_count = 173 if population.endswith("POSITIVE") else 2689
        if len(aliases) != expected_alias_count:
            raise WorkspaceError(
                f"{population}: expected {expected_alias_count} reused DEV aliases, "
                f"found {len(aliases)}"
            )
        invalid_aliases = [
            row.get("frame_id", "")
            for row in aliases
            if row.get("population_role") != "DEV"
        ]
        if invalid_aliases:
            raise WorkspaceError(
                f"{population}: reused DEV alias provenance drifted: "
                f"{invalid_aliases[:5]}"
            )


def write_manifest_views(
    root: Path,
    frames: Sequence[Mapping[str, str]],
    *,
    enforce_frozen_dev: bool = False,
) -> None:
    manifests = root / "manifests"
    populations = evaluation_population_views(frames)
    if enforce_frozen_dev:
        validate_frozen_dev_evaluation_population(populations)

    final_positive_all = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL" and is_true(row.get("is_positive"))
    )
    final_negative_all = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL" and not is_true(row.get("is_positive"))
    )
    views: dict[str, list[Mapping[str, str]]] = {
        "DEV_EXISTING.csv": _sorted_rows(
            row for row in frames if row.get("population_role") == "DEV"
        ),
        "DEV_PLASTIC_AUDITED140.csv": populations["DEV_PLASTIC_AUDITED140"],
        "COMMON_DEV_PLASTIC_POS128.csv": _sorted_rows(
            row for row in frames if row.get("paper_subset") == "COMMON_DEV_PLASTIC_POS128"
        ),
        "DEV_WOOD_POS45.csv": _sorted_rows(
            row for row in frames if row.get("paper_subset") == "DEV_WOOD_POS45"
        ),
        "COMMON_DEV_MULTISHAPE_POS173.csv": populations["DEV_EVAL_POSITIVE"],
        "DEV_NEG2689.csv": populations["DEV_EVAL_NEGATIVE"],
        "DEV_EVAL_POSITIVE.csv": populations["DEV_EVAL_POSITIVE"],
        "DEV_EVAL_NEGATIVE.csv": populations["DEV_EVAL_NEGATIVE"],
        # Backward-compatible physical FINAL inventories.
        "FINAL_POSITIVE.csv": final_positive_all,
        "FINAL_NEGATIVE.csv": final_negative_all,
        # Explicit paper-evaluation populations.
        "FINAL_EVAL_POSITIVE.csv": populations["FINAL_EVAL_POSITIVE"],
        "FINAL_EVAL_NEGATIVE.csv": populations["FINAL_EVAL_NEGATIVE"],
        "ALL_AVAILABLE_POSITIVE.csv": populations["ALL_AVAILABLE_POSITIVE"],
        "ALL_AVAILABLE_NEGATIVE.csv": populations["ALL_AVAILABLE_NEGATIVE"],
        "ALL_AVAILABLE.csv": _sorted_rows(
            populations["ALL_AVAILABLE_POSITIVE"]
            + populations["ALL_AVAILABLE_NEGATIVE"]
        ),
    }
    for name, rows in views.items():
        atomic_write_csv(manifests / name, rows, FRAME_COLUMNS)
    atomic_write_csv(manifests / "sessions.csv", _session_rows(root, frames), SESSION_COLUMNS)


MAIN_DOMAINS = ("outside", "night")          # internal provenance 이름
CONDITIONAL_DOMAINS = ("noapril",)
APPENDIX_ONLY_DOMAINS = ("cad",)
ACQUISITION_DOMAINS = ("outside", "night", "noapril", "cad", "forklift",
                       "other", "unknown")

# ── paper-facing 표시 이름 ────────────────────────────────────────────────
# `acquisition_domain` 은 내부 capture provenance 다 (outside/noapril/cad...).
# 논문 독자는 그 이름을 모른다.  `paper_domain` 은 **이미 provenance 가 확인된**
# MAIN subset 을 사람이 읽을 이름으로 바꾼 alias 이지, 새 GT 정보를 만드는 것이
# 아니다.  매핑은 세 조건을 모두 만족할 때만 성립한다.
PAPER_DOMAINS = ("daytime", "nighttime", "none")
PAPER_DOMAIN_RULES: dict[str, dict[str, str]] = {
    "daytime": {"acquisition_domain": "outside", "lighting": "day",
                "object_type": "plastic"},
    "nighttime": {"acquisition_domain": "night", "lighting": "night",
                  "object_type": "plastic"},
}
PAPER_DOMAIN_LABEL = {"daytime": "Daytime", "nighttime": "Nighttime"}


def derive_paper_domain(row: Mapping[str, str]) -> str:
    """MAIN 논문 표에 쓰는 조건 이름.  규칙을 전부 만족할 때만 배정한다.

    noapril / cad 는 어떤 규칙에도 맞지 않으므로 항상 "none" 이다 — 구조적으로
    MAIN 표에 들어갈 수 없다.
    """
    for domain, rule in PAPER_DOMAIN_RULES.items():
        if all(str(row.get(k, "")).strip().lower() == v for k, v in rule.items()):
            return domain
    return "none"


def load_acquisition_domain_map(root: Path) -> dict[str, Any]:
    """세션 -> acquisition_domain.  근거가 있는 것만 담긴다.

    파일이 없으면 빈 매핑이다 — 그 경우 모든 세션이 unknown 이 된다.
    파일명으로 추론하지 않는다.
    """
    path = root / "ACQUISITION_DOMAIN_MAP.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        raise WorkspaceError("ACQUISITION_DOMAIN_MAP.json: sessions must be an object")
    out: dict[str, Any] = {}
    for session, spec in sessions.items():
        domain = str((spec or {}).get("domain", "unknown")).strip().lower()
        if domain not in ALLOWED_VALUES["acquisition_domain"]:
            raise WorkspaceError(
                f"ACQUISITION_DOMAIN_MAP.json: {session} has invalid domain "
                f"{domain!r}; allowed={sorted(ALLOWED_VALUES['acquisition_domain'])}"
            )
        out[session] = domain
    return out


def condition_membership(row: Mapping[str, str]) -> set[str]:
    matched: set[str] = set()
    if row.get("object_type") in {"plastic", "wood"}:
        matched.add(str(row["object_type"]))
    if row.get("lighting") in {"day", "night"}:
        matched.add(str(row["lighting"]))
    domain = str(row.get("acquisition_domain", "unknown")).strip().lower()
    if domain in ACQUISITION_DOMAINS and domain != "unknown":
        matched.add(f"domain:{domain}")
    if row.get("occlusion") in {"mild", "medium", "heavy"}:
        matched.add("occlusion")
    if row.get("truncation") in {"mild", "medium", "heavy"}:
        matched.add("truncation")
    if row.get("distance_bin") == "far":
        matched.add("far")
    if row.get("occlusion") == "none" and row.get("truncation") == "none":
        matched.add("clean")
    if row.get("elevation_bin") in {"low", "mid", "high"}:
        matched.add(str(row["elevation_bin"]))
    return matched


# UNKNOWN 을 한 덩어리로 세면 view_bin 하나 때문에 전 행이 UNKNOWN 이 되어
# domain readiness 를 읽을 수 없다.  축별로 나눈다.
CORE_DOMAIN_METADATA_FIELDS = ("object_type", "acquisition_domain")
ROBUSTNESS_METADATA_FIELDS = ("occlusion", "truncation", "distance_bin",
                              "elevation_bin")
AUX_METADATA_FIELDS = ("view_bin",)


def _any_unknown(row: Mapping[str, str], fields: Sequence[str]) -> bool:
    return any(
        str(row.get(field, "unknown")).strip().lower() in {"", "unknown"}
        for field in fields
    )


def core_domain_metadata_unknown(row: Mapping[str, str]) -> bool:
    """domain experiment(M2/M5) readiness 를 좌우하는 축만 본다."""
    return _any_unknown(row, CORE_DOMAIN_METADATA_FIELDS)


def robustness_metadata_unknown(row: Mapping[str, str]) -> bool:
    return _any_unknown(row, ROBUSTNESS_METADATA_FIELDS)


def aux_metadata_unknown(row: Mapping[str, str]) -> bool:
    """view 처럼 부가적인 것.  domain readiness 를 FAIL 시키지 않는다."""
    return _any_unknown(row, AUX_METADATA_FIELDS)


def metadata_unknown(row: Mapping[str, str]) -> bool:
    required = (
        "object_type",
        "lighting",
        "occlusion",
        "truncation",
        "distance_bin",
        "elevation_bin",
        "view_bin",
    )
    return any(str(row.get(field, "unknown")).strip().lower() in {"", "unknown"} for field in required)


def compute_progress(frames: Sequence[Mapping[str, str]]) -> Progress:
    populations = evaluation_population_views(frames)
    combined_positive = populations["ALL_AVAILABLE_POSITIVE"]
    combined_negative = populations["ALL_AVAILABLE_NEGATIVE"]
    counts = Counter(
        condition
        for row in combined_positive
        for condition in condition_membership(row)
    )
    return Progress(
        positive_total=len(combined_positive),
        plastic=counts["plastic"],
        wood=counts["wood"],
        day=counts["day"],
        night=counts["night"],
        clean=counts["clean"],
        occlusion=counts["occlusion"],
        truncation=counts["truncation"],
        far=counts["far"],
        low=counts["low"],
        mid=counts["mid"],
        high=counts["high"],
        negative=len(combined_negative),
        unknown_metadata=sum(metadata_unknown(row) for row in combined_positive),
        evaluation_positive=len(populations["FINAL_EVAL_POSITIVE"]),
        evaluation_negative=len(populations["FINAL_EVAL_NEGATIVE"]),
    )


def progress_line(progress: Progress, targets: Mapping[str, Any]) -> str:
    status = (
        "READY"
        if progress.evaluation_positive or progress.evaluation_negative
        else "EMPTY"
    )
    return (
        f"[{status}] FINAL_EVAL alias {progress.evaluation_positive} positive / "
        f"{progress.evaluation_negative} negative rows | combined target "
        f"{progress.positive_total}/{targets['positive_total']} positive, "
        f"{progress.negative}/{targets['negative_total']} negative "
        "(DEV_EVAL + physical FINAL, SHA256-deduplicated)"
    )


def _is_tagged(row: Mapping[str, str], field: str) -> bool:
    return str(row.get(field, "")).strip().lower() not in {"", "unknown"}


def _evaluation_counts(
    root: Path,
    frames: Sequence[Mapping[str, str]],
) -> tuple[dict[str, int], dict[str, list[Mapping[str, str]]]]:
    populations = evaluation_population_views(frames)
    audited = populations["DEV_PLASTIC_AUDITED140"]
    dev_positive = populations["DEV_EVAL_POSITIVE"]
    dev_negative = populations["DEV_EVAL_NEGATIVE"]
    final_positive = populations["FINAL_EVAL_POSITIVE"]
    final_negative = populations["FINAL_EVAL_NEGATIVE"]
    all_positive = populations["ALL_AVAILABLE_POSITIVE"]
    all_negative = populations["ALL_AVAILABLE_NEGATIVE"]

    overlays = 0
    for row in dev_positive:
        path = resolve_workspace_path(root, row.get("overlay_path", ""))
        overlays += bool(path and path.is_file())

    counts = {
        "plastic_audited": len(audited),
        "plastic_controlled": sum(
            row.get("paper_subset") == "COMMON_DEV_PLASTIC_POS128"
            for row in dev_positive
        ),
        "wood": sum(
            row.get("paper_subset") == "DEV_WOOD_POS45" for row in dev_positive
        ),
        "dev_positive": len(dev_positive),
        "dev_negative": len(dev_negative),
        "dev_negative_unique_sha": len({row["image_sha256"] for row in dev_negative}),
        "dev_annotated": sum(is_true(row.get("is_annotated")) for row in dev_positive),
        "dev_overlays": overlays,
        "final_positive": len(final_positive),
        "final_negative": len(final_negative),
        "final_negative_unique_sha": len(
            {row["image_sha256"] for row in final_negative}
        ),
        "physical_final_positive": sum(
            row.get("population_role") == "FINAL" and is_true(row.get("is_positive"))
            for row in frames
        ),
        "physical_final_negative": sum(
            row.get("population_role") == "FINAL" and not is_true(row.get("is_positive"))
            for row in frames
        ),
        "all_positive": len(all_positive),
        "all_negative": len(all_negative),
    }
    for field in (
        "lighting",
        "occlusion",
        "truncation",
        "distance_bin",
        "elevation_bin",
        "view_bin",
    ):
        counts[f"dev_{field}_tagged"] = sum(_is_tagged(row, field) for row in dev_positive)
    return counts, populations


def render_progress_report(
    root: Path,
    frames: Sequence[Mapping[str, str]],
    targets: Mapping[str, Any],
) -> str:
    value = compute_progress(frames)
    counts, _populations = _evaluation_counts(root, frames)
    _pos = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    unknown_core = sum(1 for r in _pos if core_domain_metadata_unknown(r))
    unknown_rob = sum(1 for r in _pos if robustness_metadata_unknown(r))
    unknown_aux = sum(1 for r in _pos if aux_metadata_unknown(r))
    _cells = paper_domain_rows(frames, targets)
    cell_table = "\n".join(
        [f"{'Condition':14}{'Object':10}{'Frames':>8}{'Minimum':>9}"
         f"{'Preferred':>11}{'Sessions':>10}{'MinSess':>9}   Status",
         "-" * 88]
        + [f"{c['label']:14}{c['object_type'].capitalize():10}{c['frames']:>8}"
           f"{c['minimum']:>9}{c['preferred']:>11}{c['sessions']:>10}"
           f"{c['minimum_sessions']:>9}   {c['status']}" for c in _cells]
    )
    _main_ready = main_domains_ready(_cells)
    _cov = _condition_counts(_pos)
    _elev = targets.get("elevation_minimum", targets.get("elevation", {}))
    _cond = targets["minimum_condition_coverage"]
    _morph_ok = (_cov["plastic"] >= targets["object_type"]["plastic"]
                 and _cov["wood"] >= targets["object_type"]["wood"])
    _rob_ok = all(_cov[k] >= v for k, v in _cond.items()) and all(
        _cov[k] >= v for k, v in _elev.items())
    _total_ok = len(_pos) >= targets.get("positive_minimum",
                                         targets["positive_total"])
    _dataset_ready = _total_ok and _main_ready and _morph_ok and _rob_ok
    _neg_min = targets.get("negative_unique_minimum", targets["negative_total"])
    _neg_status = ("SATISFIED"
                   if counts["all_negative"] >= _neg_min else "DEFICIT")
    if counts["final_positive"] or counts["final_negative"]:
        alias_status = "READY — REUSED DEV_EVAL, NOT HELD OUT"
        alias_explanation = (
            "이 evaluation은 registered controlled DEV pair를 row-for-row manifest view로\n"
            "재사용한다. 새 image나 annotation을 복사하지 않았고 active frame의\n"
            "`population_role=DEV`도 바꾸지 않았다. physical FINAL은 이 실행 alias에 섞지\n"
            "않는다. 따라서 `FINAL_EVAL` 이름은 held-out FINAL을 뜻하지 않는다."
        )
    else:
        alias_status = "EMPTY — NO REGISTERED FINAL_EVAL ALIAS"
        alias_explanation = (
            "현재 registered `FINAL_EVAL` alias는 비어 있다. physical FINAL inventory는\n"
            "combined evaluation target에는 포함되지만 이 evaluator alias에 자동으로 섞지\n"
            "않는다."
        )
    return f"""# DEV evaluation population

```text
Plastic audited      {counts['plastic_audited']:4d} / 140
Plastic controlled   {counts['plastic_controlled']:4d} / 128
Wood                 {counts['wood']:4d} / 45
Combined positive    {counts['dev_positive']:4d} / 173
Negative             {counts['dev_negative']:4d} / 2689

Annotated positive   {counts['dev_annotated']:4d} / 173
Review overlays      {counts['dev_overlays']:4d} / 173

Metadata availability
Lighting tagged      {counts['dev_lighting_tagged']:4d} / 173
Occlusion tagged     {counts['dev_occlusion_tagged']:4d} / 173
Truncation tagged    {counts['dev_truncation_tagged']:4d} / 173
Distance tagged      {counts['dev_distance_bin_tagged']:4d} / 173
Elevation tagged     {counts['dev_elevation_bin_tagged']:4d} / 173
View tagged          {counts['dev_view_bin_tagged']:4d} / 173
```

# FINAL_EVAL alias status

```text
Status               {alias_status}
Positive             {counts['final_positive']:4d}
Negative rows        {counts['final_negative']:4d}
Negative unique SHA  {counts['final_negative_unique_sha']:4d}
Alias provenance     {REUSED_DEV_EVAL_ALIAS_NOTE}

Physical FINAL inventory
Positive             {counts['physical_final_positive']:4d}
Negative             {counts['physical_final_negative']:4d}
```

{alias_explanation}

# Paper evaluation readiness

목표는 `PAPER_EVAL` 기준이다 — SHA256-deduplicated union(DEV_EVAL, NEW_EVAL).
`held_out_final = false` 다. 진짜 untouched test 를 나중에 만들면 `HELDOUT_EVAL`
이라는 별도 이름을 쓴다.

```text
Positive total       {value.positive_total:4d} / {targets.get('positive_minimum', targets['positive_total'])} minimum
DATASET_READY        {str(_dataset_ready).upper()}

DATASET_READY 는 네 조건을 **동시에** 만족해야 참이다
  total >= minimum                  {str(_total_ok).lower()}
  MAIN domain coverage              {str(_main_ready).lower()}
  morphology coverage               {str(_morph_ok).lower()}
  robustness minimum coverage       {str(_rob_ok).lower()}

Object
Plastic              {value.plastic:4d} / {targets['object_type']['plastic']}
Wood                 {value.wood:4d} / {targets['object_type']['wood']}

Lighting (descriptive — quota 없음)
DAY                  {value.day:4d}
NIGHT                {value.night:4d}

Condition coverage
Clean                {value.clean:4d} / {targets['minimum_condition_coverage']['clean']}
Occlusion            {value.occlusion:4d} / {targets['minimum_condition_coverage']['occlusion']}
Truncation           {value.truncation:4d} / {targets['minimum_condition_coverage']['truncation']}
Far                  {value.far:4d} / {targets['minimum_condition_coverage']['far']}

Elevation
Low                  {value.low:4d} / {_elev['low']}
Mid                  {value.mid:4d} / {_elev['mid']}
High                 {value.high:4d} / {_elev['high']}

Negative
Negative unique      {counts['all_negative']:4d} / {_neg_min}   {_neg_status}

UNKNOWN_METADATA     {value.unknown_metadata:4d}
```

`UNKNOWN_METADATA`는 combined positive 중 object/lighting/condition metadata가 하나라도
`unknown`인 frame 수다.

## Metadata unknown — 축별

한 덩어리로 세면 `view` 하나 때문에 전 행이 unknown 이 되어 domain readiness 를
읽을 수 없다. 축을 나눈다.

```text
CORE_DOMAIN_METADATA_UNKNOWN       {unknown_core:4d}   object_type · acquisition_domain
ROBUSTNESS_METADATA_UNKNOWN        {unknown_rob:4d}   occlusion · truncation · distance · elevation
AUX_METADATA_UNKNOWN               {unknown_aux:4d}   view
```

domain experiment(M2 / M5) readiness 는 AUX 때문에 FAIL 시키지 않는다.

## Main domain evaluation readiness

```text
{cell_table}
```

내부 provenance 대응은 `reports/PAPER_DOMAIN_COVERAGE.md` 를 본다.\n내부 capture id 별 집계는 `reports/DOMAIN_COVERAGE.md`(engineering audit).
`173 / 300` 한 줄만 보고 domain experiment 진척으로 읽지 말 것 —
DATASET_READY 는 위 네 조건을 모두 만족해야 참이다.

# All available evaluation

```text
DEV positive         {counts['dev_positive']:4d}
FINAL_EVAL positive  {counts['final_positive']:4d}  frozen reused DEV execution alias
Physical FINAL pos   {counts['physical_final_positive']:4d}
ALL positive         {counts['all_positive']:4d}

DEV negative         {counts['dev_negative']:4d}  frozen membership
DEV negative SHA     {counts['dev_negative_unique_sha']:4d}  unique images
FINAL_EVAL negative  {counts['final_negative']:4d}  frozen rows
FINAL_EVAL neg SHA   {counts['final_negative_unique_sha']:4d}  unique images
Physical FINAL neg   {counts['physical_final_negative']:4d}
ALL negative         {counts['all_negative']:4d}  SHA-deduplicated union
```

`FINAL_EVAL`은 registered DEV evaluator pair에 고정된 실행 alias다.
`ALL_AVAILABLE`만 physical FINAL을 포함할 수 있는 SHA-deduplicated convenience
view다. DEV는 model selection에 사용되었을 수 있으므로 어느 쪽도 held-out FINAL로
부르지 않는다.
"""


def _condition_counts(rows: Sequence[Mapping[str, str]]) -> Counter[str]:
    return Counter(condition for row in rows for condition in condition_membership(row))


def _registry_dimensions(root: Path) -> dict[str, str]:
    path = root / "objects/OBJECT_GEOMETRY_REGISTRY.snapshot.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, str] = {}
    for item in data.get("objects", []):
        if not isinstance(item, dict):
            continue
        object_type = normalize_object_type(item.get("object_type"))
        dimensions = item.get("physical_dimensions_m") or {}
        if all(axis in dimensions for axis in ("x", "y", "z")):
            result[object_type] = f"{dimensions['x']}×{dimensions['y']}×{dimensions['z']} m"
    return result


def render_composition_report(root: Path, frames: Sequence[Mapping[str, str]]) -> str:
    populations = evaluation_population_views(frames)
    positive_groups: dict[str, Sequence[Mapping[str, str]]] = {
        "FINAL_EVAL": populations["FINAL_EVAL_POSITIVE"],
    }
    negative_groups: dict[str, Sequence[Mapping[str, str]]] = {
        "FINAL_EVAL": populations["FINAL_EVAL_NEGATIVE"],
    }
    dimensions = _registry_dimensions(root)

    def tagged_count(rows: Sequence[Mapping[str, str]], field: str) -> int:
        return sum(_is_tagged(row, field) for row in rows)

    def observed_or_unavailable(
        rows: Sequence[Mapping[str, str]],
        condition: str,
        required_fields: Sequence[str],
    ) -> str:
        if required_fields and not all(
            tagged_count(rows, field) == len(rows) for field in required_fields
        ):
            return "—"
        return str(_condition_counts(rows)[condition])

    def known_count(
        rows: Sequence[Mapping[str, str]], condition: str, field: str
    ) -> str:
        return (
            str(_condition_counts(rows)[condition])
            if tagged_count(rows, field)
            else "—"
        )

    def composition_row(
        population: str,
        label: str,
        rows: Sequence[Mapping[str, str]],
        dim: str,
        *,
        negative: bool = False,
    ) -> str:
        sessions = len({row.get("session_id", "") for row in rows})
        if negative:
            day = night = occlusion = truncation = "—"
        else:
            day = known_count(rows, "day", "lighting")
            night = known_count(rows, "night", "lighting")
            occlusion = observed_or_unavailable(
                rows, "occlusion", ("occlusion",)
            )
            truncation = observed_or_unavailable(
                rows, "truncation", ("truncation",)
            )
        return (
            f"{population:<16}{label:<20}{len(rows):>8}{sessions:>11}"
            f"{day:>7}{night:>8}"
            f"  {dim:>12}{occlusion:>13}{truncation:>13}"
        )

    condition_labels = (
        ("Plastic", "plastic", ()),
        ("Wood", "wood", ()),
        ("DAY", "day", ()),
        ("NIGHT", "night", ()),
        ("Occlusion", "occlusion", ("occlusion",)),
        ("Truncation", "truncation", ("truncation",)),
        ("Far", "far", ("distance_bin",)),
    )
    condition_lines: list[str] = []
    for population, rows in positive_groups.items():
        for label, key, required_fields in condition_labels:
            count = observed_or_unavailable(rows, key, required_fields)
            condition_lines.append(
                f"{population:<16}{label:<14}{count:>5}      —        —"
                "       —         —       —       —        —"
            )

    composition_lines: list[str] = []
    coverage_lines: list[str] = []
    for population, positives in positive_groups.items():
        negatives = negative_groups[population]
        plastic = [row for row in positives if row.get("object_type") == "plastic"]
        wood = [row for row in positives if row.get("object_type") == "wood"]
        composition_lines.extend(
            (
                composition_row(
                    population,
                    "Plastic",
                    plastic,
                    dimensions.get("plastic", "registry"),
                ),
                composition_row(
                    population,
                    "Wood",
                    wood,
                    dimensions.get("wood", "registry"),
                ),
                composition_row(population, "Combined positive", positives, "—"),
                composition_row(
                    population, "Negative", negatives, "—", negative=True
                ),
            )
        )
        for label, key, required_fields in (
            ("Clean", "clean", ("occlusion", "truncation")),
            ("Occlusion", "occlusion", ("occlusion",)),
            ("Truncation", "truncation", ("truncation",)),
            ("Far", "far", ("distance_bin",)),
            ("Low angle", "low", ("elevation_bin",)),
            ("Mid angle", "mid", ("elevation_bin",)),
            ("High angle", "high", ("elevation_bin",)),
        ):
            count = observed_or_unavailable(positives, key, required_fields)
            coverage_lines.append(f"{population:<16}{label:<16}{count:>6}")

    positives = populations["FINAL_EVAL_POSITIVE"]
    metadata_lines = [
        f"Lighting tagged      {tagged_count(positives, 'lighting'):4d} / {len(positives)}",
        f"Occlusion tagged     {tagged_count(positives, 'occlusion'):4d} / {len(positives)}",
        f"Truncation tagged    {tagged_count(positives, 'truncation'):4d} / {len(positives)}",
        f"Distance tagged      {tagged_count(positives, 'distance_bin'):4d} / {len(positives)}",
        f"Elevation tagged     {tagged_count(positives, 'elevation_bin'):4d} / {len(positives)}",
    ]

    return f"""# Dataset composition

`FINAL_EVAL`은 registered controlled DEV pair를 그대로 재사용한 frozen 실행
alias다. physical FINAL은 이 population에 자동으로 합치지 않는다. 이 alias는
held-out FINAL이 아니며 조건은 서로 중복될 수 있다. metric은 evaluation 전까지
`—`다.

## Experiment 6 condition table

```text
Population      Condition         N   pnp↑  corner↓  R med↓  yaw med↓  t med↓  IoU3D↑  AUCall↑
──────────────────────────────────────────────────────────────────────────────────────────────
{os.linesep.join(condition_lines)}
```

## Experiment 7 split composition

```text
Population      Object                Frames   Sessions    DAY   NIGHT    Dimensions    Occlusion   Truncation
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
{os.linesep.join(composition_lines)}
```

```text
Population      Condition       Frames
──────────────────────────────────────
{os.linesep.join(coverage_lines)}
```

0과 `—`를 구분한다. `—`는 해당 조건이 없다는 뜻이 아니라 metadata가 부족해
판정할 수 없다는 뜻이다.

```text
{os.linesep.join(metadata_lines)}
```

FINAL_EVAL negative는 registered frozen membership 2689행을 유지하며 unique image는
2688장이다. `ALL_AVAILABLE_NEGATIVE.csv`만 known duplicate를 SHA256으로 합친
convenience view다. Alias provenance는 `{REUSED_DEV_EVAL_ALIAS_NOTE}`이고 held-out이
아니다.
"""


def _target_map(targets: Mapping[str, Any]) -> dict[str, int]:
    return {
        "plastic": int(targets["object_type"]["plastic"]),
        "wood": int(targets["object_type"]["wood"]),
        "day": int(targets["lighting"]["day"]),
        "night": int(targets["lighting"]["night"]),
        "clean": int(targets["minimum_condition_coverage"]["clean"]),
        "occlusion": int(targets["minimum_condition_coverage"]["occlusion"]),
        "truncation": int(targets["minimum_condition_coverage"]["truncation"]),
        "far": int(targets["minimum_condition_coverage"]["far"]),
        "low": int(targets["elevation"]["low"]),
        "mid": int(targets["elevation"]["mid"]),
        "high": int(targets["elevation"]["high"]),
    }


def render_priority_report(
    frames: Sequence[Mapping[str, str]],
    targets: Mapping[str, Any],
) -> str:
    progress = compute_progress(frames)
    populations = evaluation_population_views(frames)
    eval_positive = populations["FINAL_EVAL_POSITIVE"]
    eval_negative = populations["FINAL_EVAL_NEGATIVE"]
    status = "READY" if eval_positive or eval_negative else "EMPTY"
    return f"""# Combined evaluation target progress

```text
Positive                  {progress.positive_total:4d} / {targets['positive_total']}
Negative                  {progress.negative:4d} / {targets['negative_total']}
UNKNOWN_METADATA          {progress.unknown_metadata:4d}
Counting population       ALL_AVAILABLE
Counting policy           DEV_EVAL + physical FINAL; SHA256-deduplicated
New annotation required   NO
```

목표 진행률은 DEV와 FINAL을 따로 세지 않는다. 현재 controlled DEV_EVAL과 이후
추가되는 physical FINAL을 합친 `ALL_AVAILABLE` view 하나만 사용한다. 같은 image는
SHA256으로 한 번만 센다. 이 목표 미달은 새 annotation을 의무화하지 않는다.

## Registered evaluator population

```text
Status                    {status}
FINAL_EVAL positive       {len(eval_positive):4d}
FINAL_EVAL negative rows  {len(eval_negative):4d}
Negative unique images    {len({row['image_sha256'] for row in eval_negative}):4d}
FINAL_EVAL held-out       NO
Alias provenance          {REUSED_DEV_EVAL_ALIAS_NOTE}
```

등록된 2D/pose evaluator pair binding은 준비되어 있다. AP/AUROC/FPR95 score
pipeline과 workspace condition-tag subgroup evaluator의 통합 binding은 아직
보고되지 않았으므로 해당 metric cell은 `—`를 유지한다.
"""


def render_overlay_audit(root: Path, frames: Sequence[Mapping[str, str]]) -> str:
    populations = evaluation_population_views(frames)
    groups: dict[str, list[Mapping[str, str]]] = {
        "DEV": [
            row
            for row in frames
            if row.get("population_role") == "DEV"
            and is_true(row.get("is_positive"))
            and is_true(row.get("is_annotated"))
        ],
        "DEV_UNVERIFIED": [
            row
            for row in frames
            if row.get("population_role") == "DEV_UNVERIFIED"
            and is_true(row.get("is_annotated"))
        ],
        "FINAL_EVAL_ALIAS": [
            row
            for row in populations["FINAL_EVAL_POSITIVE"]
            if row.get("population_role") == "DEV"
        ],
        "PHYSICAL_FINAL": [
            row
            for row in frames
            if row.get("population_role") == "FINAL"
            and is_true(row.get("is_positive"))
            and is_true(row.get("is_annotated"))
        ],
    }
    lines = ["population          annotation JSON   overlays   missing", "──────────────────────────────────────────────────────"]
    missing_paths: list[str] = []
    for name, rows in groups.items():
        overlay_count = 0
        for row in rows:
            overlay = resolve_workspace_path(root, row.get("overlay_path", ""))
            if overlay and overlay.is_file():
                overlay_count += 1
            else:
                missing_paths.append(row.get("overlay_path", ""))
        lines.append(f"{name:<20}{len(rows):>15}{overlay_count:>11}{len(rows) - overlay_count:>10}")
    missing = "\n".join(path for path in missing_paths if path) or "(없음)"
    return f"""# Overlay audit

```text
{os.linesep.join(lines)}
```

## Missing overlays

```text
{missing}
```
"""


def write_duplicate_audit(root: Path, frames: Sequence[Mapping[str, str]]) -> int:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in frames:
        digest = row.get("image_sha256", "")
        if digest:
            grouped[digest].append(row)
    rows: list[dict[str, str]] = []
    duplicate_groups = 0
    for digest, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        duplicate_groups += 1
        group_id = f"DUP_{duplicate_groups:04d}"
        for row in members:
            rows.append(
                {
                    "duplicate_group": group_id,
                    "sha256": digest,
                    "frame_id": row.get("frame_id", ""),
                    "population_role": row.get("population_role", ""),
                    "paper_subset": row.get("paper_subset", ""),
                    "source_image_path": row.get("source_image_path", ""),
                    "active": "true",
                }
            )
    atomic_write_csv(
        root / "reports/DUPLICATE_AUDIT.csv",
        rows,
        (
            "duplicate_group",
            "sha256",
            "frame_id",
            "population_role",
            "paper_subset",
            "source_image_path",
            "active",
        ),
    )
    return duplicate_groups


def paper_domain_rows(
    frames: Sequence[Mapping[str, str]], targets: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """MAIN 논문 조건(Daytime / Nighttime)별 readiness.

    frame 수만 채우고 READY 로 넘기지 않는다 — 같은 영상에서 인접 50장을 뽑아도
    독립 표본 50개가 아니다.  세션 수를 따로 게이트한다.
    """
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    spec_all = targets.get("main_paper_domains", {})
    out: list[dict[str, Any]] = []
    for domain in ("daytime", "nighttime"):
        spec = spec_all.get(domain, {})
        minimum = int(spec.get("minimum_frames", 50))
        preferred = int(spec.get("preferred_frames", 60))
        min_sessions = int(spec.get("minimum_sessions", 2))
        rows = [r for r in positive if derive_paper_domain(r) == domain]
        sessions = {str(r.get("session_id", "")) for r in rows if r.get("session_id")}
        n, ns = len(rows), len(sessions)
        frame_ready, session_ready = n >= minimum, ns >= min_sessions
        if not frame_ready and not session_ready:
            status = "DEFICIT"
        elif not frame_ready:
            status = "FRAME_DEFICIT"
        elif not session_ready:
            status = "SESSION_DEFICIT"
        elif n >= preferred:
            status = "PREFERRED_READY"
        else:
            status = "READY"
        rule = PAPER_DOMAIN_RULES[domain]
        out.append({
            "paper_domain": domain, "label": PAPER_DOMAIN_LABEL[domain],
            "object_type": rule["object_type"],
            "internal": f"acquisition_domain={rule['acquisition_domain']} + "
                        f"lighting={rule['lighting']}",
            "frames": n, "sessions": ns, "session_ids": sorted(sessions),
            "minimum": minimum, "preferred": preferred,
            "minimum_sessions": min_sessions,
            "frame_ready": frame_ready, "session_ready": session_ready,
            "status": status,
            "deficit_to_minimum": max(0, minimum - n),
            "deficit_to_preferred": max(0, preferred - n),
        })
    return out


def main_domains_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Daytime · Nighttime 이 둘 다 frame+session 게이트를 통과했나."""
    return bool(rows) and all(
        r["frame_ready"] and r["session_ready"] for r in rows
    )


def render_paper_domain_coverage(
    frames: Sequence[Mapping[str, str]], targets: Mapping[str, Any]
) -> str:
    """PAPER_DOMAIN_COVERAGE.md — 논문-facing 조건 표."""
    rows = paper_domain_rows(frames, targets)
    lines = ["# Main domain evaluation readiness", ""]
    lines.append(
        "논문 MAIN 표가 쓰는 조건은 **Daytime / Nighttime** 두 개뿐이다.\n"
        "내부 capture id(`outside` · `noapril` · `cad`)는 여기에 나오지 않는다.\n"
    )
    lines.append("")
    lines.append("```text")
    lines.append(
        f"{'Condition':14}{'Object':10}{'Frames':>8}{'Minimum':>9}"
        f"{'Preferred':>11}{'Sessions':>10}{'MinSess':>9}   Status"
    )
    lines.append("-" * 88)
    for r in rows:
        lines.append(
            f"{r['label']:14}{r['object_type'].capitalize():10}{r['frames']:>8}"
            f"{r['minimum']:>9}{r['preferred']:>11}{r['sessions']:>10}"
            f"{r['minimum_sessions']:>9}   {r['status']}"
        )
    lines.append("```")
    lines.append("")

    short = [r for r in rows if r["deficit_to_minimum"] > 0]
    if short:
        lines.append("## Annotation priority")
        lines.append("")
        lines.append("```text")
        lines.append(f"{'Condition':14}{'Object':10}{'to minimum':>12}{'to preferred':>14}")
        lines.append("-" * 50)
        for r in sorted(short, key=lambda x: -x["deficit_to_minimum"]):
            lines.append(
                f"{r['label']:14}{r['object_type'].capitalize():10}"
                f"{r['deficit_to_minimum']:>12}{r['deficit_to_preferred']:>14}"
            )
        lines.append("```")
        lines.append("")

    lines.append("## Internal provenance")
    lines.append("")
    lines.append(
        "표시 이름과 내부 membership 의 대응이다. `paper_domain` 은 새 GT 를 만드는\n"
        "것이 아니라, 이미 provenance 가 확인된 subset 에 읽을 수 있는 이름을 붙인 것이다.\n"
    )
    lines.append("")
    lines.append("```text")
    for r in rows:
        lines.append(f"{r['label']:14}<- {r['internal']} + object_type={r['object_type']}")
        lines.append(f"{'':14}   sessions: {', '.join(r['session_ids']) or '(none)'}")
    lines.append("```")
    lines.append("")
    lines.append(
        "`noapril` 과 `cad` 는 어떤 규칙에도 맞지 않아 `paper_domain=none` 이다 —\n"
        "구조적으로 MAIN 표에 들어갈 수 없다. 데이터는 삭제하지 않고 provenance 로 남는다.\n"
    )
    lines.append("")
    lines.append("## M2 dataset gate")
    lines.append("")
    lines.append("```text")
    lines.append(f"MAIN_DOMAINS_READY   {str(main_domains_ready(rows)).lower()}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def domain_coverage_rows(
    frames: Sequence[Mapping[str, str]], targets: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """내부 acquisition domain 별 집계 — engineering provenance 감사용."""
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    out: list[dict[str, Any]] = []
    for domain in ACQUISITION_DOMAINS:
        rows = [r for r in positive
                if str(r.get("acquisition_domain", "unknown")).strip().lower() == domain]
        if not rows:
            continue
        sessions = {str(r.get("session_id", "")) for r in rows if r.get("session_id")}
        papers = Counter(derive_paper_domain(r) for r in rows)
        out.append({
            "domain": domain, "frames": len(rows), "sessions": len(sessions),
            "session_ids": sorted(sessions),
            "paper_domains": dict(papers),
        })
    return out


def render_domain_coverage(
    frames: Sequence[Mapping[str, str]], targets: Mapping[str, Any]
) -> str:
    rows = domain_coverage_rows(frames, targets)
    total = sum(r["frames"] for r in rows)
    lines = ["# Acquisition domain coverage", ""]
    lines.append("```text")
    lines.append("INTERNAL PROVENANCE REPORT")
    lines.append("NOT A PAPER-FACING DOMAIN TABLE")
    lines.append("```")
    lines.append("")
    lines.append(
        "여기 나오는 `outside` · `noapril` · `cad` 는 **내부 capture id** 다.\n"
        "논문 결과표의 조건 이름이 아니다 — 그건 `PAPER_DOMAIN_COVERAGE.md` 를 본다.\n"
    )
    lines.append("")
    lines.append("```text")
    lines.append(f"{'internal id':14}{'N':>6}{'Sessions':>10}   paper_domain 분포")
    lines.append("-" * 70)
    for r in rows:
        dist = ", ".join(f"{k}={v}" for k, v in sorted(r["paper_domains"].items()))
        lines.append(f"{r['domain']:14}{r['frames']:>6}{r['sessions']:>10}   {dist}")
    lines.append("-" * 70)
    lines.append(f"{'TOTAL':14}{total:>6}")
    lines.append("```")
    lines.append("")
    lines.append(
        "`paper_domain=none` 은 MAIN 표에 들어가지 않는다는 뜻이지 데이터가\n"
        "버려진다는 뜻이 아니다. provenance 는 그대로 유지된다.\n"
    )
    lines.append("")
    return "\n".join(lines)


def adaptation_leakage_rows(
    frames: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """ADAPT 와 EVAL 이 이미지·세션 수준에서 겹치는지.  삭제는 하지 않는다."""
    ev = [r for r in frames if r.get("usage_role") == "EVAL_LABELED"]
    ad = [r for r in frames if r.get("usage_role") == "ADAPT_UNLABELED"]
    ev_sha = {r.get("image_sha256", "") for r in ev if r.get("image_sha256")}
    ad_sha = {r.get("image_sha256", "") for r in ad if r.get("image_sha256")}
    ev_ses = {r.get("session_id", "") for r in ev if r.get("session_id")}
    ad_ses = {r.get("session_id", "") for r in ad if r.get("session_id")}
    sha_overlap = sorted(ev_sha & ad_sha)
    ses_overlap = sorted(ev_ses & ad_ses)
    return {
        "eval_frames": len(ev), "adapt_frames": len(ad),
        "sha_overlap": sha_overlap, "session_overlap": ses_overlap,
        "sha_overlap_count": len(sha_overlap),
        "session_overlap_count": len(ses_overlap),
        "pass": not sha_overlap and not ses_overlap,
    }


def write_reports(root: Path, frames: Sequence[Mapping[str, str]]) -> Progress:
    targets = load_targets(root)
    reports = root / "reports"
    atomic_write_text(reports / "ANNOTATION_PROGRESS.md", render_progress_report(root, frames, targets))
    atomic_write_text(reports / "DATASET_COMPOSITION.md", render_composition_report(root, frames))
    atomic_write_text(reports / "NEXT_ANNOTATION_PRIORITY.md", render_priority_report(frames, targets))
    atomic_write_text(reports / "OVERLAY_AUDIT.md", render_overlay_audit(root, frames))
    atomic_write_text(reports / "DOMAIN_COVERAGE.md", render_domain_coverage(frames, targets))
    atomic_write_text(reports / "PAPER_DOMAIN_COVERAGE.md",
                      render_paper_domain_coverage(frames, targets))
    write_duplicate_audit(root, frames)
    return compute_progress(frames)
