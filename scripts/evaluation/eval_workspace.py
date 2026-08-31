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
    "occlusion": frozenset({"none", "mild", "medium", "heavy", "unknown"}),
    "truncation": frozenset({"none", "mild", "medium", "heavy", "unknown"}),
    "distance_bin": frozenset({"near", "mid", "far", "unknown"}),
    "size_bin": frozenset({"small", "medium", "large", "unknown"}),
    "elevation_bin": frozenset({"low", "mid", "high", "unknown"}),
}

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

DEFAULT_TARGETS: dict[str, Any] = {
    "final_positive_total": 300,
    "object_type": {"plastic": 180, "wood": 120},
    "lighting": {"day": 220, "night": 80},
    "minimum_condition_coverage": {
        "clean": 100,
        "occlusion": 60,
        "truncation": 50,
        "far_small": 60,
    },
    "elevation": {"low": 90, "mid": 120, "high": 90},
    "final_negative_total": 1500,
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
    "invariants": {
        "dev_is_never_counted_toward_final_targets": True,
        "source_data_is_read_only": True,
        "workspace_copies_are_not_hardlinks_or_symlinks": True,
        "conditions_are_tags_and_may_overlap": True,
        "unknown_metadata_is_not_inferred": True,
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
        "far_small": "distance_bin == far OR size_bin == small",
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
    "evaluation_populations": {
        "DEV_EVAL": {
            "positive": "COMMON_DEV_PLASTIC_POS128 + DEV_WOOD_POS45",
            "negative": "DEV_NEG2689 frozen membership",
            "held_out_final": False,
        },
        "FINAL_EVAL": {
            "positive": (
                "population_role == FINAL AND is_positive == true AND "
                "is_annotated == true AND controlled_eval_eligible == true"
            ),
            "negative": "population_role == FINAL AND is_positive == false",
            "held_out_final": True,
        },
        "ALL_AVAILABLE": {
            "positive": "SHA256-deduplicated union(DEV_EVAL_POSITIVE, FINAL_EVAL_POSITIVE)",
            "negative": "SHA256-deduplicated union(DEV_EVAL_NEGATIVE, FINAL_EVAL_NEGATIVE)",
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
    far_small: int
    low: int
    mid: int
    high: int
    negative: int
    unknown_metadata: int


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


def infer_annotation_tags(annotation_path: Path) -> dict[str, str]:
    """Infer only metadata explicitly represented in saved annotation JSON."""

    inferred = {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
    if not annotation_path.is_file():
        return inferred
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return inferred
    objects = document.get("objects")
    if not isinstance(objects, list) or not objects or not isinstance(objects[0], dict):
        return inferred
    obj = objects[0]
    inferred["object_type"] = normalize_object_type(
        obj.get("object_type") or document.get("object_type") or obj.get("name")
    )

    level = str(obj.get("occlusion_level", "unknown")).strip().lower()
    reason_values = {
        str(entry.get("reason", "unknown")).strip().lower()
        for entry in obj.get("keypoint_annotations", [])
        if isinstance(entry, dict)
    }
    if level == "none":
        inferred["occlusion"] = "none"
    elif level in {"mild", "medium", "partial", "heavy"}:
        inferred["occlusion"] = "medium" if level == "partial" else level
    elif "occluded" in reason_values:
        inferred["occlusion"] = "medium"

    truncation = obj.get("truncation")
    fraction: float | None = None
    if isinstance(truncation, dict):
        raw_fraction = truncation.get("bbox_outside_fraction")
        if not isinstance(raw_fraction, bool):
            try:
                fraction = float(raw_fraction) if raw_fraction is not None else None
            except (TypeError, ValueError):
                fraction = None
    if (
        isinstance(truncation, dict)
        and truncation.get("is_truncated") is False
        and (fraction is None or fraction == 0.0)
    ):
        inferred["truncation"] = "none"
    if (
        isinstance(truncation, dict)
        and truncation.get("is_truncated") is True
    ) or (fraction is not None and fraction > 0.0) or "truncated" in reason_values:
        # The JSON supplies only a boolean, so no unrecorded severity threshold
        # is invented.  ``mild`` is the least-specific positive allowed tag.
        inferred["truncation"] = "mild"
    return inferred


def _workspace_readme() -> str:
    return """# Pallet real evaluation workspace v1

이 directory는 기존 source/raw/GT와 분리된 수정 가능한 evaluation working copy다.
`DEV`, `DEV_UNVERIFIED`, `FINAL`은 절대 합쳐서 FINAL 수치로 계산하지 않는다.

## Evaluation populations

- `DEV_EVAL_POSITIVE.csv`: controlled plastic 128 + wood 45 (173 images)
- `DEV_EVAL_NEGATIVE.csv`: frozen DEV membership 2689 rows
- `FINAL_EVAL_POSITIVE.csv`: annotated, QA-eligible FINAL positives only
- `FINAL_EVAL_NEGATIVE.csv`: FINAL negatives
- `ALL_AVAILABLE_{POSITIVE,NEGATIVE}.csv`: DEV/FINAL SHA256-deduplicated union
- `DEV_PLASTIC_AUDITED140.csv`: FT-overlap 12장을 포함한 review population

`ALL_AVAILABLE`은 편의/보조 evaluation이며 held-out FINAL로 부르지 않는다.

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

# D/E. 새 FINAL image 배치 후 session.json과 필요시 frame_tags.csv 작성
# final/positive/sessions/<session>/rgb/*.png

# F. 기존 DEV 또는 plastic FINAL annotation
python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/dev_existing/sessions/<session> \\
  --out_dir data/evaluation/pallet_eval_v1/dev_existing/annotations/<session> \\
  --population-role DEV --default_split eval \\
  --object-type plastic_standard_110x130x11 \\
  --eval-root data/evaluation/pallet_eval_v1

python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<session> \\
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<session> \\
  --population-role FINAL --default_split eval \\
  --object-type plastic_standard_110x130x11 \\
  --eval-root data/evaluation/pallet_eval_v1

# Wood는 geometry와 intrinsics provenance를 명시한다.
python scripts/annotate/annotate.py \\
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<wood_session> \\
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<wood_session> \\
  --population-role FINAL --default_split eval \\
  --object-type wood_small_80x59x14 \\
  --intrinsics-quality CALIBRATED --intrinsics-source '<calibration artifact>' \\
  --eval-root data/evaluation/pallet_eval_v1
```

G. 매 save마다 JSON, `_overlays/<stem>.png`, `manifests/frames.csv`, progress
report가 갱신된다. H. 부족 조건은 `reports/ANNOTATION_PROGRESS.md`와
`reports/NEXT_ANNOTATION_PRIORITY.md`에서 확인한다.

새 FINAL 촬영은 `final/positive/sessions/<session>/rgb/` 또는
`final/negative/sessions/<session>/rgb/`에 둔다. 각 session에 `session.json`을
작성하고 frame별 수동 tag는 `frame_tags.csv`로 override한다.

`far/small`, `elevation`, `view`는 임의 threshold로 추정하지 않는다. 명시하지
않은 값은 `unknown`으로 남아 `NEXT_ANNOTATION_PRIORITY.md`의 metadata queue에
표시된다.
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
    for frame_id, expected_path in KNOWN_DEV_NEG_DUPLICATE_PATHS.items():
        row = by_id[frame_id]
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
            or row.get("storage_mode") != "source_reference_read_only"
            or row.get("image_path") != expected_path
            or row.get("source_image_path") != expected_path
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
    metadata_path = session_dir / "session.json"
    if not metadata_path.is_file():
        raise WorkspaceError(f"FINAL session is missing session.json: {session_dir}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkspaceError(f"invalid session JSON {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise WorkspaceError(f"session JSON must be an object: {metadata_path}")

    frame_tags: dict[str, dict[str, str]] = {}
    canonical_tag_rows: dict[str, int] = {}
    tags_path = session_dir / "frame_tags.csv"
    if tags_path.is_file():
        with tags_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(csv.DictReader(handle), start=2):
                identity = next(
                    (
                        raw.get(key, "").strip()
                        for key in ("frame", "filename", "image", "frame_id", "image_path")
                        if raw.get(key, "").strip()
                    ),
                    "",
                )
                if not identity:
                    raise WorkspaceError(f"{tags_path}:{line_number}: missing frame identity")
                identity_path = Path(identity)
                if identity_path.is_absolute() or ".." in identity_path.parts:
                    raise WorkspaceError(
                        f"{tags_path}:{line_number}: unsafe frame identity {identity!r}"
                    )
                entry = {key: str(value or "").strip() for key, value in raw.items()}
                stem = identity_path.stem
                canonical = stem.rsplit("__", 1)[-1] if "__" in stem else stem
                if canonical in canonical_tag_rows:
                    raise WorkspaceError(
                        f"{tags_path}:{line_number}: duplicate/conflicting frame tag alias "
                        f"for {canonical!r}; first declared on line {canonical_tag_rows[canonical]}"
                    )
                canonical_tag_rows[canonical] = line_number
                for alias in {identity_path.name, stem, canonical}:
                    if alias in frame_tags:
                        raise WorkspaceError(
                            f"{tags_path}:{line_number}: conflicting frame tag alias {alias!r}"
                        )
                    frame_tags[alias] = entry
    return metadata, frame_tags


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
            default_tags = metadata.get("default_tags") or {}
            if not isinstance(default_tags, dict):
                raise WorkspaceError(f"default_tags must be an object: {session_dir / 'session.json'}")

            for image_path in _image_files(session_dir / "rgb"):
                stem = image_path.stem
                frame_id = f"{session_id}__{safe_component(stem, fallback='frame')}"
                if frame_id in seen_frame_ids:
                    raise WorkspaceError(f"duplicate FINAL frame_id {frame_id!r}")
                seen_frame_ids.add(frame_id)
                override = frame_tags.get(image_path.name) or frame_tags.get(stem) or {}

                def explicit(field: str, session_key: str | None = None) -> Any:
                    key = session_key or field
                    if override.get(field) not in (None, ""):
                        return override[field]
                    if key in default_tags:
                        return default_tags[key]
                    return metadata.get(key, "unknown")

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
                inferred = (
                    infer_annotation_tags(annotation_path)
                    if annotation_path is not None and annotation_path.is_file()
                    else {"occlusion": "unknown", "truncation": "unknown", "object_type": "unknown"}
                )
                object_override = override.get("object_type", "")
                object_type = (
                    normalize_object_type(object_override)
                    if object_override
                    else inferred["object_type"]
                )
                if object_type == "unknown":
                    object_type = normalize_object_type(explicit("object_type"))
                if not is_positive:
                    object_type = "none"

                # Explicit per-frame tags win.  Otherwise the current saved
                # annotation is more specific than a session-wide default.
                occlusion = (
                    normalize_tag("occlusion", override["occlusion"])
                    if override.get("occlusion")
                    else inferred["occlusion"]
                )
                truncation = (
                    normalize_tag("truncation", override["truncation"])
                    if override.get("truncation")
                    else inferred["truncation"]
                )
                if occlusion == "unknown":
                    occlusion = normalize_tag("occlusion", explicit("occlusion"))
                if truncation == "unknown":
                    truncation = normalize_tag("truncation", explicit("truncation"))

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
                        "lighting": normalize_tag("lighting", explicit("lighting")),
                        "occlusion": occlusion if is_positive else "unknown",
                        "truncation": truncation if is_positive else "unknown",
                        "distance_bin": normalize_tag("distance_bin", explicit("distance_bin")),
                        "size_bin": normalize_tag("size_bin", explicit("size_bin")),
                        "elevation_bin": normalize_tag("elevation_bin", explicit("elevation_bin")),
                        "view_bin": normalize_tag("view_bin", explicit("view_bin"), allow_view=True),
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
        if exists:
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


def evaluation_population_views(
    frames: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    """Build explicit DEV, FINAL, and SHA-unique convenience populations.

    ``controlled_eval_eligible`` is the existing frame-level QA eligibility
    gate.  FINAL positive images do not enter an evaluation population until
    their annotation JSON has been committed.  The frozen DEV negative view
    intentionally retains its known duplicate membership; only the
    ``ALL_AVAILABLE`` convenience union deduplicates it by image SHA256.
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

    final_positive = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("is_annotated"))
        and is_true(row.get("controlled_eval_eligible"))
    )
    final_negative = _sorted_rows(
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and not is_true(row.get("is_positive"))
    )
    _require_unique_population_sha(final_positive, population="FINAL_EVAL_POSITIVE")
    _require_unique_population_sha(final_negative, population="FINAL_EVAL_NEGATIVE")

    all_positive = _sha_deduplicated_union(
        dev_positive_members + final_positive,
        population="ALL_AVAILABLE_POSITIVE",
    )
    all_negative = _sha_deduplicated_union(
        dev_negative + final_negative,
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
        "FINAL_EVAL_POSITIVE": final_positive,
        "FINAL_EVAL_NEGATIVE": final_negative,
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


def condition_membership(row: Mapping[str, str]) -> set[str]:
    matched: set[str] = set()
    if row.get("object_type") in {"plastic", "wood"}:
        matched.add(str(row["object_type"]))
    if row.get("lighting") in {"day", "night"}:
        matched.add(str(row["lighting"]))
    if row.get("occlusion") in {"mild", "medium", "heavy"}:
        matched.add("occlusion")
    if row.get("truncation") in {"mild", "medium", "heavy"}:
        matched.add("truncation")
    if row.get("distance_bin") == "far" or row.get("size_bin") == "small":
        matched.add("far_small")
    if row.get("occlusion") == "none" and row.get("truncation") == "none":
        matched.add("clean")
    if row.get("elevation_bin") in {"low", "mid", "high"}:
        matched.add(str(row["elevation_bin"]))
    return matched


def metadata_unknown(row: Mapping[str, str]) -> bool:
    required = (
        "object_type",
        "lighting",
        "occlusion",
        "truncation",
        "distance_bin",
        "size_bin",
        "elevation_bin",
        "view_bin",
    )
    return any(str(row.get(field, "unknown")).strip().lower() in {"", "unknown"} for field in required)


def compute_progress(frames: Sequence[Mapping[str, str]]) -> Progress:
    final_positive_all = [
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("controlled_eval_eligible"))
    ]
    completed = [row for row in final_positive_all if is_true(row.get("is_annotated"))]
    counts = Counter(condition for row in completed for condition in condition_membership(row))
    negative = sum(
        row.get("population_role") == "FINAL" and not is_true(row.get("is_positive"))
        for row in frames
    )
    return Progress(
        positive_total=len(completed),
        plastic=counts["plastic"],
        wood=counts["wood"],
        day=counts["day"],
        night=counts["night"],
        clean=counts["clean"],
        occlusion=counts["occlusion"],
        truncation=counts["truncation"],
        far_small=counts["far_small"],
        low=counts["low"],
        mid=counts["mid"],
        high=counts["high"],
        negative=negative,
        unknown_metadata=sum(metadata_unknown(row) for row in final_positive_all),
    )


def progress_line(progress: Progress, targets: Mapping[str, Any]) -> str:
    return (
        f"[Progress] FINAL {progress.positive_total}/{targets['final_positive_total']} | "
        f"Plastic {progress.plastic}/{targets['object_type']['plastic']} | "
        f"Wood {progress.wood}/{targets['object_type']['wood']} | "
        f"Night {progress.night}/{targets['lighting']['night']} | "
        f"Occ {progress.occlusion}/{targets['minimum_condition_coverage']['occlusion']} | "
        f"Trunc {progress.truncation}/{targets['minimum_condition_coverage']['truncation']} | "
        f"FarSmall {progress.far_small}/{targets['minimum_condition_coverage']['far_small']}"
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
        "all_positive": len(all_positive),
        "all_negative": len(all_negative),
    }
    for field in (
        "lighting",
        "occlusion",
        "truncation",
        "distance_bin",
        "size_bin",
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
Size tagged          {counts['dev_size_bin_tagged']:4d} / 173
Elevation tagged     {counts['dev_elevation_bin_tagged']:4d} / 173
View tagged          {counts['dev_view_bin_tagged']:4d} / 173
```

# FINAL annotation progress

```text
Positive total       {value.positive_total:4d} / {targets['final_positive_total']}

Object
Plastic              {value.plastic:4d} / {targets['object_type']['plastic']}
Wood                 {value.wood:4d} / {targets['object_type']['wood']}

Lighting
DAY                  {value.day:4d} / {targets['lighting']['day']}
NIGHT                {value.night:4d} / {targets['lighting']['night']}

Condition coverage
Clean                {value.clean:4d} / {targets['minimum_condition_coverage']['clean']}
Occlusion            {value.occlusion:4d} / {targets['minimum_condition_coverage']['occlusion']}
Truncation           {value.truncation:4d} / {targets['minimum_condition_coverage']['truncation']}
Far / small          {value.far_small:4d} / {targets['minimum_condition_coverage']['far_small']}

Elevation
Low                  {value.low:4d} / {targets['elevation']['low']}
Mid                  {value.mid:4d} / {targets['elevation']['mid']}
High                 {value.high:4d} / {targets['elevation']['high']}

Negative
Negative             {value.negative:4d} / {targets['final_negative_total']}

UNKNOWN_METADATA     {value.unknown_metadata:4d}
```

DEV frame은 위 FINAL target에 포함하지 않는다.

# All available evaluation

```text
DEV positive         {counts['dev_positive']:4d}
FINAL positive       {counts['final_positive']:4d}
ALL positive         {counts['all_positive']:4d}

DEV negative         {counts['dev_negative']:4d}  frozen membership
DEV negative SHA     {counts['dev_negative_unique_sha']:4d}  unique images
FINAL negative       {counts['final_negative']:4d}
ALL negative         {counts['all_negative']:4d}  SHA-deduplicated union
```

`ALL_AVAILABLE`은 편의/보조 evaluation population이다. DEV는 model selection에
사용되었을 수 있으므로 held-out FINAL로 부르지 않는다.
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
        "DEV": populations["DEV_EVAL_POSITIVE"],
        "FINAL": populations["FINAL_EVAL_POSITIVE"],
        "ALL_AVAILABLE": populations["ALL_AVAILABLE_POSITIVE"],
    }
    negative_groups: dict[str, Sequence[Mapping[str, str]]] = {
        "DEV": populations["DEV_EVAL_NEGATIVE"],
        "FINAL": populations["FINAL_EVAL_NEGATIVE"],
        "ALL_AVAILABLE": populations["ALL_AVAILABLE_NEGATIVE"],
    }
    dimensions = _registry_dimensions(root)

    def composition_row(
        population: str,
        label: str,
        rows: Sequence[Mapping[str, str]],
        dim: str,
    ) -> str:
        local = _condition_counts(rows)
        sessions = len({row.get("session_id", "") for row in rows})
        return (
            f"{population:<16}{label:<20}{len(rows):>8}{sessions:>11}"
            f"{local['day']:>7}{local['night']:>8}"
            f"  {dim:>12}{local['occlusion']:>13}{local['truncation']:>13}"
        )

    condition_labels = (
        ("Plastic", "plastic"),
        ("Wood", "wood"),
        ("DAY", "day"),
        ("NIGHT", "night"),
        ("Occlusion", "occlusion"),
        ("Truncation", "truncation"),
        ("Far / small", "far_small"),
    )
    condition_lines: list[str] = []
    for population, rows in positive_groups.items():
        counts = _condition_counts(rows)
        for label, key in condition_labels:
            condition_lines.append(
                f"{population:<16}{label:<14}{counts[key]:>5}      —        —"
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
                composition_row(population, "Negative", negatives, "—"),
            )
        )
        counts = _condition_counts(positives)
        coverage_lines.extend(
            f"{population:<16}{label:<16}{counts[key]:>6}"
            for label, key in (
                ("Clean", "clean"),
                ("Occlusion", "occlusion"),
                ("Truncation", "truncation"),
                ("Far / small", "far_small"),
                ("Low angle", "low"),
                ("Mid angle", "mid"),
                ("High angle", "high"),
            )
        )

    return f"""# Dataset composition

DEV는 controlled 173장, FINAL positive는 annotation과 QA eligibility를 모두
충족한 frame만 포함한다. ALL_AVAILABLE은 각 DEV/FINAL union을 image SHA256으로
deduplicate한 보조 population이며 held-out FINAL이 아니다. 조건은 서로 중복될 수
있고 metric은 evaluation 전까지 `—`다.

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

DEV negative의 frozen membership은 2689행을 유지한다. ALL_AVAILABLE negative는
known duplicate image membership을 SHA256으로 합쳐 현재 2688 unique image다.
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
        "far_small": int(targets["minimum_condition_coverage"]["far_small"]),
        "low": int(targets["elevation"]["low"]),
        "mid": int(targets["elevation"]["mid"]),
        "high": int(targets["elevation"]["high"]),
    }


def render_priority_report(
    frames: Sequence[Mapping[str, str]],
    targets: Mapping[str, Any],
) -> str:
    completed = [
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("controlled_eval_eligible"))
        and is_true(row.get("is_annotated"))
    ]
    candidates = [
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("controlled_eval_eligible"))
        and not is_true(row.get("is_annotated"))
    ]
    current = _condition_counts(completed)
    target_map = _target_map(targets)
    ratios = {
        key: max(target - current[key], 0) / target if target else 0.0
        for key, target in target_map.items()
    }
    dimensions = {
        "plastic": "object",
        "wood": "object",
        "day": "lighting",
        "night": "lighting",
        "clean": "condition",
        "occlusion": "condition",
        "truncation": "condition",
        "far_small": "condition",
        "low": "elevation",
        "mid": "elevation",
        "high": "elevation",
    }
    labels = {
        "plastic": "Plastic",
        "wood": "Wood",
        "day": "DAY",
        "night": "NIGHT",
        "clean": "Clean",
        "occlusion": "Occlusion",
        "truncation": "Truncation",
        "far_small": "Far / small",
        "low": "Low",
        "mid": "Mid",
        "high": "High",
    }
    pair_kind_rank = {
        frozenset({"object", "lighting"}): 0,
        frozenset({"object", "condition"}): 1,
        frozenset({"lighting", "elevation"}): 2,
        frozenset({"object", "elevation"}): 3,
        frozenset({"lighting", "condition"}): 4,
        frozenset({"condition", "elevation"}): 5,
    }
    remaining = {
        key: max(target - current[key], 0)
        for key, target in target_map.items()
    }
    pairs: list[tuple[float, int, int, int, str, str]] = []
    keys = list(target_map)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if dimensions[left] == dimensions[right]:
                continue
            pairs.append(
                (
                    ratios[left] + ratios[right],
                    pair_kind_rank[frozenset({dimensions[left], dimensions[right]})],
                    remaining[left] + remaining[right],
                    index,
                    left,
                    right,
                )
            )
    # Normalized deficit remains the primary score.  At an all-zero start every
    # condition ties, so prefer actionable capture pairings (object+lighting),
    # then the larger absolute remaining coverage, instead of lexical labels.
    pairs.sort(key=lambda item: (-item[0], item[1], -item[2], item[3]))
    priority_lines = [
        f"{rank}. {labels[left]} + {labels[right]}  (score={score:.3f})"
        for rank, (score, _kind, _absolute, _index, left, right)
        in enumerate(pairs[:5], start=1)
    ]

    candidate_scores: list[tuple[float, Mapping[str, str], set[str]]] = []
    for row in candidates:
        matched = condition_membership(row)
        score = sum(ratios.get(key, 0.0) for key in matched)
        candidate_scores.append((score, row, matched))
    candidate_scores.sort(key=lambda item: (-item[0], item[1].get("frame_id", "")))
    candidate_lines = []
    for score, row, matched in candidate_scores[:30]:
        candidate_lines.append(
            f"{score:6.3f}  {row.get('session_id', ''):<24} "
            f"{row.get('frame_id', ''):<40} {row.get('image_path', '')}  "
            f"[{', '.join(sorted(labels[key] for key in matched if key in labels))}]"
        )
    if not candidate_lines:
        candidate_lines.append("(matching unannotated FINAL candidate 없음)")

    unknown = [
        row
        for row in frames
        if row.get("population_role") == "FINAL"
        and is_true(row.get("is_positive"))
        and is_true(row.get("controlled_eval_eligible"))
        and metadata_unknown(row)
    ]
    unknown_lines = [
        f"{row.get('session_id', ''):<24} {row.get('frame_id', ''):<40} {row.get('image_path', '')}"
        for row in unknown[:50]
    ] or ["(없음)"]
    return f"""# Next annotation priority

현재 FINAL annotation deficit 비율의 합으로 계산한다. UNKNOWN tag는 점수에
임의로 포함하지 않는다.

## NEXT PRIORITY

```text
{os.linesep.join(priority_lines)}
```

## Tagged unannotated candidates

```text
score   session                  frame                                    image  [matched]
{os.linesep.join(candidate_lines)}
```

## NEEDS_METADATA ({len(unknown)})

```text
{os.linesep.join(unknown_lines)}
```
"""


def render_overlay_audit(root: Path, frames: Sequence[Mapping[str, str]]) -> str:
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
        "FINAL": [
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


def write_reports(root: Path, frames: Sequence[Mapping[str, str]]) -> Progress:
    targets = load_targets(root)
    reports = root / "reports"
    atomic_write_text(reports / "ANNOTATION_PROGRESS.md", render_progress_report(root, frames, targets))
    atomic_write_text(reports / "DATASET_COMPOSITION.md", render_composition_report(root, frames))
    atomic_write_text(reports / "NEXT_ANNOTATION_PRIORITY.md", render_priority_report(frames, targets))
    atomic_write_text(reports / "OVERLAY_AUDIT.md", render_overlay_audit(root, frames))
    write_duplicate_audit(root, frames)
    return compute_progress(frames)
