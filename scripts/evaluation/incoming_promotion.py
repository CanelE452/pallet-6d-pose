"""Promote reviewed incoming annotations into the active evaluation workspace.

The editor promotes only the annotation just saved. Batch synchronization is
kept for repair/maintenance. Every promotion independently copies the image,
annotation, optional overlay, and explicit frame-tag row; raw captures remain
immutable.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:  # Package import from tests/editor.
    from . import eval_workspace as W
except ImportError:  # Direct script import from scripts/evaluation on sys.path.
    import eval_workspace as W  # type: ignore[no-redef]


MATERIALS = ("plastic", "wood")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _new_summary() -> dict[str, Any]:
    return {
        "promoted": 0,
        "updated": 0,
        "removed": 0,
        "skipped_existing": 0,
        "metadata_synced": 0,
        "session_metadata_synced": 0,
        "by_dest": {},
        "unresolved": [],
        "missing_source": [],
    }


def _merge_summary(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key in (
        "promoted", "updated", "removed", "skipped_existing",
        "metadata_synced", "session_metadata_synced",
    ):
        total[key] += int(part.get(key, 0))
    for key in ("unresolved", "missing_source"):
        total[key].extend(part.get(key, []))
    for destination, count in part.get("by_dest", {}).items():
        total["by_dest"][destination] = (
            total["by_dest"].get(destination, 0) + int(count)
        )


def _annotation_is_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or not objects or not isinstance(objects[0], dict):
        return False
    cuboid = objects[0].get("projected_cuboid") or []
    return isinstance(cuboid, list) and len(cuboid) >= 8


def _annotation_material(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        objects = payload.get("objects") if isinstance(payload, dict) else None
        first = objects[0] if isinstance(objects, list) and objects else {}
    except (OSError, json.JSONDecodeError):
        first = {}
    return W.normalize_object_type(
        first.get("object_type") if isinstance(first, dict) else None
    )


def annotated_stems(annotation_dir: Path) -> list[str]:
    """Return complete annotation stems from one object-specific staging view."""

    return [
        path.stem
        for path in sorted(Path(annotation_dir).glob("*.json"))
        if _annotation_is_complete(path)
    ]


def resolve_auto_destination(root: Path, annotation_dir: Path) -> tuple[str, str] | None:
    """Resolve ``(source_session, destination_session)`` from explicit metadata."""

    root = Path(root)
    name = Path(annotation_dir).name
    if "__" not in name:
        return None
    source_session, _, material = name.rpartition("__")
    if material not in MATERIALS:
        return None
    metadata_path = root / "incoming/sessions" / source_session / "session.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    lighting = str(metadata.get("lighting", "")).strip().lower()
    if lighting not in {"day", "night"}:
        return None
    destination = f"{material}_{lighting}_01"
    if not (root / "final/positive/sessions" / destination / "session.json").is_file():
        return None
    return source_session, destination


def _find_source_image(rgb_dir: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in Path(rgb_dir).iterdir()
        if path.is_file()
        and path.stem == stem
        and path.suffix.lower() in IMAGE_SUFFIXES
    ) if Path(rgb_dir).is_dir() else []
    if len(matches) > 1:
        raise W.WorkspaceError(
            f"incoming frame stem has multiple image files: {rgb_dir}/{stem}"
        )
    return matches[0] if matches else None


def _atomic_copy2(source: Path, destination: Path) -> bool:
    """Independently copy ``source`` when bytes differ; return whether changed."""

    source = Path(source)
    destination = Path(destination)
    if destination.is_file() and W.sha256_file(source) == W.sha256_file(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if W.sha256_file(source) != W.sha256_file(temporary):
            raise W.WorkspaceError(
                f"promotion copy verification failed: {source} -> {destination}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _copy_promoted_image(
    source: Path, destination_rgb: Path
) -> tuple[Path, bool]:
    """Copy a new image, but never overwrite a same-stem pixel collision."""

    existing = _find_source_image(destination_rgb, source.stem)
    if existing is not None:
        if W.sha256_file(source) != W.sha256_file(existing):
            raise W.WorkspaceError(
                "promotion image collision for the same frame stem: "
                f"source={source}, destination={existing}"
            )
        return existing, False
    destination = destination_rgb / source.name
    return destination, _atomic_copy2(source, destination)


def _normalized_resolution(metadata: dict[str, Any]) -> tuple[int, int] | None:
    value = metadata.get("resolution")
    try:
        if isinstance(value, dict):
            width, height = int(value["width"]), int(value["height"])
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            width, height = int(value[0]), int(value[1])
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _destination_images_come_from_source(destination_rgb: Path, source_rgb: Path) -> bool:
    for destination in destination_rgb.iterdir() if destination_rgb.is_dir() else ():
        if not destination.is_file() or destination.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        source = source_rgb / destination.name
        if not source.is_file() or W.sha256_file(source) != W.sha256_file(destination):
            return False
    return True


def _sync_destination_session_metadata(
    root: Path,
    source_session: str,
    destination_session: str,
) -> bool:
    """Copy explicit camera/resolution provenance required to reopen the session."""

    source_dir = root / "incoming/sessions" / source_session
    destination_dir = root / "final/positive/sessions" / destination_session
    source_metadata = W.load_session_metadata(source_dir)
    destination_metadata = W.load_session_metadata(destination_dir)

    expected_material = destination_session.split("_", 1)[0]
    actual_material = W.normalize_object_type(destination_metadata.get("object_type"))
    if actual_material != expected_material:
        raise W.WorkspaceError(
            "promotion destination object mismatch: "
            f"{destination_session} declares {actual_material!r}"
        )
    source_lighting = str(source_metadata.get("lighting", "")).strip().lower()
    destination_lighting = str(
        destination_metadata.get("lighting", "")
    ).strip().lower()
    if source_lighting != destination_lighting:
        raise W.WorkspaceError(
            "promotion destination lighting mismatch: "
            f"source={source_lighting!r}, destination={destination_lighting!r}"
        )
    if str(destination_metadata.get("population_role", "")).strip().upper() != "FINAL":
        raise W.WorkspaceError(
            f"promotion destination is not active FINAL: {destination_session}"
        )

    changed = False
    source_resolution = _normalized_resolution(source_metadata)
    destination_resolution = _normalized_resolution(destination_metadata)
    destination_rgb = destination_dir / "rgb"
    source_rgb = source_dir / "rgb"
    if source_resolution and source_resolution != destination_resolution:
        if not _destination_images_come_from_source(destination_rgb, source_rgb):
            raise W.WorkspaceError(
                "cannot mix incoming capture resolution into an existing destination: "
                f"source={source_resolution}, destination={destination_resolution}"
            )
        destination_metadata["resolution"] = list(source_resolution)
        changed = True

    camera = source_metadata.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    raw_quality = str(
        source_metadata.get("intrinsics_quality")
        or camera.get("intrinsics_quality")
        or "UNKNOWN"
    ).strip().upper()
    quality = raw_quality if raw_quality in {
        "CALIBRATED", "SENSOR_PROFILE_SCALED", "ESTIMATED_HFOV", "UNKNOWN"
    } else "UNKNOWN"
    raw_source = str(
        source_metadata.get("intrinsics_source")
        or camera.get("intrinsics_source")
        or "incoming capture metadata"
    ).strip()
    source_text = (
        f"{raw_source}; capture quality={raw_quality}; "
        f"source session={source_session}"
    )
    for key, value in (
        ("intrinsics_quality", quality),
        ("intrinsics_source", source_text),
    ):
        if destination_metadata.get(key) != value:
            destination_metadata[key] = value
            changed = True
    provenance = sorted(
        set(destination_metadata.get("promoted_from_sessions") or [])
        | {source_session}
    )
    if destination_metadata.get("promoted_from_sessions") != provenance:
        destination_metadata["promoted_from_sessions"] = provenance
        changed = True

    source_k = source_dir / "cam_K.txt"
    destination_k = destination_dir / "cam_K.txt"
    if source_k.is_file():
        changed = _atomic_copy2(source_k, destination_k) or changed
    if changed:
        W.atomic_write_json(destination_dir / "session.json", destination_metadata)
    return changed


def _tag_value(row: dict[str, str], field: str) -> str:
    value = str(row.get(field, "")).strip().lower()
    return "" if value in {"", "unknown"} else value


def _sync_frame_tags(
    source_annotation_dir: Path,
    destination_session_dir: Path,
    images_by_stem: dict[str, Path],
) -> int:
    """Make destination explicit frame tags equal to the staging tag rows."""

    source_rows = W.load_frame_tag_overrides(source_annotation_dir)
    destination_rows = W.load_frame_tag_overrides(destination_session_dir)
    updates: dict[str, dict[str, str]] = {}
    changed = 0
    for stem, image in images_by_stem.items():
        source = source_rows.get(stem, {})
        destination = destination_rows.get(stem, {})
        update = {
            field: source.get(field, "unknown")
            for field in W.FRAME_TAG_FIELDS
        }
        if any(
            _tag_value(source, field) != _tag_value(destination, field)
            for field in W.FRAME_TAG_FIELDS
        ):
            changed += 1
        updates[image.name] = update
    if updates:
        W.update_frame_tags_csv_many(destination_session_dir, updates)
    return changed


def promote_annotations(
    root: Path,
    annotation_dir: Path,
    source_session: str,
    destination_session: str,
    *,
    stems: Iterable[str] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Synchronize selected complete staging annotations into one destination."""

    root = Path(root).resolve()
    annotation_dir = Path(annotation_dir).resolve()
    incoming_annotation_root = (root / "incoming/annotations").resolve()
    try:
        annotation_dir.relative_to(incoming_annotation_root)
    except ValueError as exc:
        raise W.WorkspaceError(
            f"incoming annotation directory is outside workspace: {annotation_dir}"
        ) from exc
    for label, value in (
        ("source_session", source_session),
        ("destination_session", destination_session),
    ):
        if not value or Path(value).name != value or value in {".", ".."}:
            raise W.WorkspaceError(f"unsafe {label}: {value!r}")
    selected_stems = sorted(set(stems if stems is not None else annotated_stems(annotation_dir)))
    summary = _new_summary()
    if not selected_stems:
        return summary

    source_rgb = root / "incoming/sessions" / source_session / "rgb"
    destination_session_dir = root / "final/positive/sessions" / destination_session
    destination_rgb = destination_session_dir / "rgb"
    destination_annotations = root / "final/positive/annotations" / destination_session
    destination_overlays = destination_annotations / "_overlays"

    # Resolve every source and reject pixel collisions before changing any
    # destination metadata or files.  This keeps batch repair fail-closed.
    prepared: list[tuple[str, Path, Path, Path | None]] = []
    expected_material = destination_session.split("_", 1)[0]
    for stem in selected_stems:
        source_annotation = annotation_dir / f"{stem}.json"
        if not _annotation_is_complete(source_annotation):
            continue
        annotation_material = _annotation_material(source_annotation)
        if annotation_material != expected_material:
            raise W.WorkspaceError(
                "incoming annotation object does not match destination: "
                f"annotation={annotation_material!r}, destination={expected_material!r}, "
                f"path={source_annotation}"
            )
        source_image = _find_source_image(source_rgb, stem)
        if source_image is None:
            summary["missing_source"].append(f"{source_session}/{stem}")
            continue
        existing_image = _find_source_image(destination_rgb, stem)
        if (
            existing_image is not None
            and W.sha256_file(source_image) != W.sha256_file(existing_image)
        ):
            raise W.WorkspaceError(
                "promotion image collision for the same frame stem: "
                f"source={source_image}, destination={existing_image}"
            )
        prepared.append(
            (stem, source_annotation, source_image, existing_image)
        )
    if not prepared:
        return summary

    # Validate both CSVs before any independently copied artifact is changed.
    W.load_frame_tag_overrides(annotation_dir)
    W.load_frame_tag_overrides(destination_session_dir)
    if _sync_destination_session_metadata(root, source_session, destination_session):
        summary["session_metadata_synced"] = 1

    promoted_images: dict[str, Path] = {}
    for stem, source_annotation, source_image, existing_image in prepared:
        destination_image = existing_image or (destination_rgb / source_image.name)
        destination_annotation = destination_annotations / source_annotation.name
        was_complete = bool(existing_image) and destination_annotation.is_file()

        destination_image, image_changed = _copy_promoted_image(
            source_image, destination_rgb
        )
        annotation_changed = _atomic_copy2(source_annotation, destination_annotation)
        overlay_changed = False
        source_overlay = annotation_dir / "_overlays" / f"{stem}.png"
        if source_overlay.is_file():
            overlay_changed = _atomic_copy2(
                source_overlay, destination_overlays / source_overlay.name
            )
        else:
            destination_overlay = destination_overlays / f"{stem}.png"
            if destination_overlay.is_file():
                destination_overlay.unlink()
                overlay_changed = True

        if not was_complete:
            summary["promoted"] += 1
            summary["by_dest"][destination_session] = (
                summary["by_dest"].get(destination_session, 0) + 1
            )
        elif image_changed or annotation_changed or overlay_changed:
            summary["updated"] += 1
        else:
            summary["skipped_existing"] += 1
        promoted_images[stem] = destination_image

    summary["metadata_synced"] += _sync_frame_tags(
        annotation_dir, destination_session_dir, promoted_images
    )
    if refresh and (promoted_images or summary["metadata_synced"]):
        frames = W.refresh_frame_index(root, rehash_final=True)
        W.write_reports(root, frames)
    return summary


def promote_incoming_annotation(
    root: Path,
    annotation_path: Path,
    *,
    refresh: bool = False,
    deleted: bool = False,
    all_complete: bool = False,
) -> dict[str, Any]:
    """Synchronize one incoming annotation, or its complete staging session."""

    root = Path(root).resolve()
    annotation_path = Path(annotation_path).resolve()
    incoming_root = (root / "incoming/annotations").resolve()
    try:
        relative = annotation_path.relative_to(incoming_root)
    except ValueError:
        return _new_summary()
    if len(relative.parts) != 2 or annotation_path.suffix.lower() != ".json":
        raise W.WorkspaceError(
            f"incoming annotation must be one direct view JSON: {annotation_path}"
        )
    annotation_dir = annotation_path.parent
    resolved = resolve_auto_destination(root, annotation_dir)
    if resolved is None:
        summary = _new_summary()
        summary["unresolved"].append(annotation_dir.name)
        return summary
    source_session, destination_session = resolved
    if deleted:
        destination_annotations = (
            root / "final/positive/annotations" / destination_session
        )
        destination_annotation = destination_annotations / annotation_path.name
        destination_overlay = (
            destination_annotations / "_overlays" / f"{annotation_path.stem}.png"
        )
        summary = _new_summary()
        if destination_annotation.is_file():
            deleted_copy = destination_annotation.with_suffix(
                destination_annotation.suffix + ".deleted"
            )
            os.replace(destination_annotation, deleted_copy)
            summary["removed"] = 1
        destination_overlay.unlink(missing_ok=True)
        if refresh and summary["removed"]:
            frames = W.refresh_frame_index(root, rehash_final=True)
            W.write_reports(root, frames)
        return summary
    return promote_annotations(
        root,
        annotation_dir,
        source_session,
        destination_session,
        stems=None if all_complete else [annotation_path.stem],
        refresh=refresh,
    )


def promote_annotated_incoming(root: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Repair/synchronize all complete incoming annotations."""

    root = Path(root).resolve()
    annotation_root = root / "incoming/annotations"
    summary = _new_summary()
    if not annotation_root.is_dir():
        return summary
    for annotation_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        resolved = resolve_auto_destination(root, annotation_dir)
        if resolved is None:
            summary["unresolved"].append(annotation_dir.name)
            continue
        source_session, destination_session = resolved
        part = promote_annotations(
            root,
            annotation_dir,
            source_session,
            destination_session,
            refresh=False,
        )
        _merge_summary(summary, part)
    if refresh and (
        summary["promoted"]
        or summary["updated"]
        or summary["metadata_synced"]
        or summary["session_metadata_synced"]
    ):
        frames = W.refresh_frame_index(root, rehash_final=True)
        W.write_reports(root, frames)
    return summary
