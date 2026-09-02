#!/usr/bin/env python3
"""Import a raw capture ZIP into the evaluation workspace without activating it.

Incoming captures are intentionally kept outside ``final/``.  A raw continuous
capture may mix positive, negative, plastic, wood, and unusable frames; assigning
those labels during import would silently corrupt the paper evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import from tests/editors.
    from .eval_workspace import (
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        read_csv,
        safe_component,
        sha256_file,
    )
except ImportError:  # Direct ``python scripts/evaluation/...`` execution.
    from eval_workspace import (  # type: ignore[no-redef]
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        read_csv,
        safe_component,
        sha256_file,
    )


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COPY_CHUNK_BYTES = 1024 * 1024
FRAME_MANIFEST_COLUMNS = (
    "frame",
    "archive_entry",
    "image_path",
    "image_sha256",
    "byte_size",
    "width",
    "height",
    "duplicate_in_capture",
    "duplicate_in_other_incoming",
    "duplicate_in_active_evaluation",
    "duplicate_references",
)
SESSION_INDEX_COLUMNS = (
    "session_id",
    "lighting",
    "object_type",
    "active_evaluation_member",
    "review_status",
    "image_count",
    "frame_first",
    "frame_last",
    "resolution",
    "source_archive_name",
    "source_archive_sha256",
    "session_path",
)


def _validate_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise WorkspaceError(f"unsafe ZIP member path: {name!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0o170000
    return unix_mode == 0o120000


def _image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    names_seen: set[str] = set()
    for info in archive.infolist():
        path = _validate_member_path(info.filename)
        if info.is_dir():
            continue
        if info.flag_bits & 0x1:
            raise WorkspaceError(f"encrypted ZIP member is unsupported: {info.filename}")
        if _is_symlink(info):
            raise WorkspaceError(f"ZIP symlink is forbidden: {info.filename}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if len(path.parts) != 1:
            raise WorkspaceError(
                f"incoming image members must be flat, got {info.filename!r}"
            )
        normalized = path.name.casefold()
        if normalized in names_seen:
            raise WorkspaceError(f"duplicate/case-colliding ZIP image name: {path.name!r}")
        names_seen.add(normalized)
        members.append(info)
    if not members:
        raise WorkspaceError("archive contains no supported image members")
    return sorted(members, key=lambda item: item.filename)


def _read_camera_info(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, Any], bytes, str]:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == "camera_info.json"
    ]
    if len(matches) != 1:
        raise WorkspaceError(
            f"expected exactly one camera_info.json, found {len(matches)}"
        )
    info = matches[0]
    _validate_member_path(info.filename)
    if info.file_size > 1024 * 1024:
        raise WorkspaceError("camera_info.json is unexpectedly larger than 1 MiB")
    payload = archive.read(info)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceError(f"invalid camera_info.json: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkspaceError("camera_info.json root must be an object")
    return value, payload, hashlib.sha256(payload).hexdigest()


def _format_camera_matrix(camera_info: Mapping[str, Any]) -> str:
    matrix = camera_info.get("K")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise WorkspaceError("camera_info.json K must be a 3x3 matrix")
    try:
        return "".join(
            " ".join(f"{float(value):.18e}" for value in row) + "\n"
            for row in matrix
        )
    except (TypeError, ValueError) as exc:
        raise WorkspaceError("camera_info.json K contains a non-numeric value") from exc


def _png_dimensions(prefix: bytes) -> tuple[int | None, int | None]:
    if len(prefix) < 24 or prefix[:8] != PNG_SIGNATURE or prefix[12:16] != b"IHDR":
        return None, None
    return struct.unpack(">II", prefix[16:24])


def _copy_member_verified(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> tuple[str, int, int | None, int | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    prefix = bytearray()
    try:
        with archive.open(info, "r") as source, destination.open("xb") as target:
            while True:
                chunk = source.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if len(prefix) < 32:
                    prefix.extend(chunk[: 32 - len(prefix)])
                digest.update(chunk)
                target.write(chunk)
                byte_count += len(chunk)
        # Reading ZipExtFile to EOF verifies the member CRC.
        if byte_count != info.file_size:
            raise WorkspaceError(
                f"uncompressed size mismatch for {info.filename}: "
                f"expected={info.file_size}, copied={byte_count}"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    width, height = _png_dimensions(bytes(prefix))
    return digest.hexdigest(), byte_count, width, height


def _hash_references(rows: Iterable[Mapping[str, str]], id_field: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("image_sha256", "")).strip().lower()
        reference = str(row.get(id_field, "")).strip()
        if digest and reference:
            result[digest].append(reference)
    return dict(result)


def _active_hashes(root: Path) -> dict[str, list[str]]:
    return _hash_references(read_csv(root / "manifests/frames.csv"), "frame_id")


def _other_incoming_hashes(root: Path) -> dict[str, list[str]]:
    rows: list[dict[str, str]] = []
    sessions = root / "incoming/sessions"
    if sessions.is_dir():
        for manifest in sorted(sessions.glob("*/manifests/frames.csv")):
            session_id = manifest.parents[1].name
            for row in read_csv(manifest):
                enriched = dict(row)
                enriched["incoming_reference"] = (
                    f"{session_id}/{row.get('archive_entry', row.get('frame', ''))}"
                )
                rows.append(enriched)
    return _hash_references(rows, "incoming_reference")


def _camera_resolution(camera_info: Mapping[str, Any]) -> tuple[int | None, int | None]:
    try:
        width = int(camera_info["width"])
        height = int(camera_info["height"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return width, height


def _write_session_index(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    sessions_root = root / "incoming/sessions"
    if sessions_root.is_dir():
        for session_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            metadata_path = session_dir / "session.json"
            if not metadata_path.is_file():
                continue
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = value.get("resolution") or {}
            source = value.get("source_archive") or {}
            rows.append(
                {
                    "session_id": value.get("session_id", session_dir.name),
                    "lighting": value.get("lighting", "unknown"),
                    "object_type": value.get("object_type", "unknown"),
                    "active_evaluation_member": str(
                        bool(value.get("active_evaluation_member", False))
                    ).lower(),
                    "review_status": value.get("review_status", "unknown"),
                    "image_count": value.get("image_count", 0),
                    "frame_first": value.get("frame_first", ""),
                    "frame_last": value.get("frame_last", ""),
                    "resolution": (
                        f"{resolution.get('width')}x{resolution.get('height')}"
                        if resolution.get("width") and resolution.get("height")
                        else "unknown"
                    ),
                    "source_archive_name": source.get("name", ""),
                    "source_archive_sha256": source.get("sha256", ""),
                    "session_path": str(session_dir.relative_to(root)),
                }
            )
    atomic_write_csv(root / "incoming/manifests/sessions.csv", rows, SESSION_INDEX_COLUMNS)


def import_capture(
    root: Path,
    archive_path: Path,
    session_id: str,
    lighting: str,
    *,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract and hash one capture, then atomically publish it as unreviewed."""

    root = root.resolve()
    archive_path = archive_path.resolve()
    if not root.is_dir() or not (root / "DATASET_CONTRACT.json").is_file():
        raise WorkspaceError(f"evaluation workspace not found: {root}")
    if not archive_path.is_file():
        raise WorkspaceError(f"capture archive not found: {archive_path}")
    normalized_session_id = safe_component(session_id)
    if normalized_session_id != session_id:
        raise WorkspaceError(
            f"session_id must already be filesystem-safe: {session_id!r} -> "
            f"{normalized_session_id!r}"
        )
    lighting = lighting.strip().lower()
    if lighting not in {"day", "night", "unknown"}:
        raise WorkspaceError("lighting must be one of: day, night, unknown")

    sessions_root = root / "incoming/sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    destination = sessions_root / session_id
    if destination.exists():
        raise WorkspaceError(f"incoming session already exists; refusing overwrite: {destination}")

    source_before = archive_path.stat()
    archive_sha256 = sha256_file(archive_path)
    if expected_archive_sha256 is not None:
        expected = expected_archive_sha256.strip().lower()
        if archive_sha256 != expected:
            raise WorkspaceError(
                f"archive SHA mismatch: expected={expected}, actual={archive_sha256}"
            )

    active_hashes = _active_hashes(root)
    other_incoming_hashes = _other_incoming_hashes(root)
    stage = Path(tempfile.mkdtemp(prefix=f".{session_id}.importing.", dir=sessions_root))
    rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
            members = _image_members(archive)
            camera_info, camera_payload, camera_info_sha256 = _read_camera_info(archive)
            camera_width, camera_height = _camera_resolution(camera_info)
            current_hashes: dict[str, list[str]] = defaultdict(list)
            for index, info in enumerate(members, start=1):
                frame = PurePosixPath(info.filename).name
                output = stage / "rgb" / frame
                digest, byte_size, width, height = _copy_member_verified(
                    archive, info, output
                )
                if width is not None and height is not None:
                    if camera_width is not None and camera_height is not None:
                        if (width, height) != (camera_width, camera_height):
                            raise WorkspaceError(
                                f"image/camera resolution mismatch for {frame}: "
                                f"image={width}x{height}, camera={camera_width}x{camera_height}"
                            )
                current_hashes[digest].append(info.filename)
                rows.append(
                    {
                        "frame": Path(frame).stem,
                        "archive_entry": info.filename,
                        "image_path": f"incoming/sessions/{session_id}/rgb/{frame}",
                        "image_sha256": digest,
                        "byte_size": byte_size,
                        "width": width or "",
                        "height": height or "",
                    }
                )
                if index % 1000 == 0 or index == len(members):
                    print(f"[{session_id}] extracted {index}/{len(members)}", flush=True)

            for row in rows:
                digest = str(row["image_sha256"])
                same_capture = current_hashes[digest]
                other_refs = other_incoming_hashes.get(digest, [])
                active_refs = active_hashes.get(digest, [])
                references = [
                    *(f"same:{value}" for value in same_capture if value != row["archive_entry"]),
                    *(f"incoming:{value}" for value in other_refs),
                    *(f"active:{value}" for value in active_refs),
                ]
                row.update(
                    {
                        "duplicate_in_capture": str(len(same_capture) > 1).lower(),
                        "duplicate_in_other_incoming": str(bool(other_refs)).lower(),
                        "duplicate_in_active_evaluation": str(bool(active_refs)).lower(),
                        "duplicate_references": ";".join(references),
                    }
                )

            # Preserve the exact source camera payload as well as normalized provenance.
            (stage / "camera_info.json").write_bytes(camera_payload)
            atomic_write_text(stage / "cam_K.txt", _format_camera_matrix(camera_info))

        source_after = archive_path.stat()
        source_identity_before = (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
        )
        source_identity_after = (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
        )
        if source_identity_after != source_identity_before:
            raise WorkspaceError("source archive changed while it was being imported")

        widths = {str(row["width"]) for row in rows if row["width"] != ""}
        heights = {str(row["height"]) for row in rows if row["height"] != ""}
        if len(widths) > 1 or len(heights) > 1:
            raise WorkspaceError(f"capture has mixed resolutions: widths={widths}, heights={heights}")
        width = int(next(iter(widths))) if widths else camera_width
        height = int(next(iter(heights))) if heights else camera_height
        duplicate_in_capture = sum(row["duplicate_in_capture"] == "true" for row in rows)
        duplicate_other = sum(
            row["duplicate_in_other_incoming"] == "true" for row in rows
        )
        duplicate_active = sum(
            row["duplicate_in_active_evaluation"] == "true" for row in rows
        )
        metadata = {
            "schema_version": "pallet_eval_incoming_capture_v1",
            "session_id": session_id,
            "workspace_scope": "INCOMING_UNREVIEWED",
            "population_role": None,
            "active_evaluation_member": False,
            "review_status": "UNREVIEWED_MIXED_CAPTURE",
            "object_type": "unknown",
            "lighting": lighting,
            "capture_protocol": "raw_continuous_capture_imported_unreviewed",
            "default_tags": {
                "occlusion": "unknown",
                "truncation": "unknown",
                "distance_bin": "unknown",
                "size_bin": "unknown",
                "elevation_bin": "unknown",
                "view_bin": "unknown",
            },
            "image_count": len(rows),
            "frame_first": rows[0]["frame"],
            "frame_last": rows[-1]["frame"],
            "resolution": {"width": width, "height": height},
            "camera": {
                "intrinsics_quality": "PROVIDED_UNVERIFIED",
                "intrinsics_source": "source archive camera_info.json",
                "camera_info_sha256": camera_info_sha256,
                "K": camera_info.get("K"),
                "fx": camera_info.get("fx"),
                "fy": camera_info.get("fy"),
                "cx": camera_info.get("cx"),
                "cy": camera_info.get("cy"),
            },
            "source_archive": {
                "name": archive_path.name,
                "sha256": archive_sha256,
                "byte_size": source_before.st_size,
                "mtime_ns": source_before.st_mtime_ns,
            },
            "import": {
                "method": "independent_zip_extraction_with_per_entry_crc_and_sha256",
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_preserved": True,
                "destination_is_symlink": False,
            },
            "duplicate_audit": {
                "frames_duplicate_in_capture": duplicate_in_capture,
                "frames_duplicate_in_other_incoming": duplicate_other,
                "frames_duplicate_in_active_evaluation": duplicate_active,
            },
            "activation_rule": (
                "Human-review and independently copy selected frames into final/positive "
                "or final/negative; incoming frames are never evaluation members by import alone."
            ),
        }
        atomic_write_json(stage / "session.json", metadata)
        atomic_write_csv(stage / "manifests/frames.csv", rows, FRAME_MANIFEST_COLUMNS)
        os.rename(stage, destination)
        _write_session_index(root)
        return metadata
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/evaluation/pallet_eval_v1"),
        help="evaluation workspace root",
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lighting", choices=("day", "night", "unknown"), required=True)
    parser.add_argument(
        "--expected-archive-sha256",
        help="optional pre-audited digest; import fails if the archive does not match",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = import_capture(
            args.root,
            args.archive,
            args.session_id,
            args.lighting,
            expected_archive_sha256=args.expected_archive_sha256,
        )
    except (WorkspaceError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"[FAIL] incoming capture import: {exc}") from exc
    print(
        json.dumps(
            {
                "session_id": result["session_id"],
                "image_count": result["image_count"],
                "lighting": result["lighting"],
                "active_evaluation_member": result["active_evaluation_member"],
                "duplicate_audit": result["duplicate_audit"],
                "source_archive_sha256": result["source_archive"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
