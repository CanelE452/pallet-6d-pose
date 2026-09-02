#!/usr/bin/env python3
"""Materialize frozen DEV negative references as verified workspace copies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import from tests/editors.
    from .eval_workspace import (
        FRAME_COLUMNS,
        PROVENANCE_COLUMNS,
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        copy2_verified,
        evaluation_population_views,
        load_frames,
        read_csv,
        sha256_file,
        validate_active_image_sha_uniqueness,
        validate_frozen_dev_evaluation_population,
        workspace_relative,
        write_manifest_views,
        write_reports,
    )
except ImportError:  # Direct ``python scripts/evaluation/...`` execution.
    from eval_workspace import (  # type: ignore[no-redef]
        FRAME_COLUMNS,
        PROVENANCE_COLUMNS,
        WorkspaceError,
        atomic_write_csv,
        atomic_write_json,
        copy2_verified,
        evaluation_population_views,
        load_frames,
        read_csv,
        sha256_file,
        validate_active_image_sha_uniqueness,
        validate_frozen_dev_evaluation_population,
        workspace_relative,
        write_manifest_views,
        write_reports,
    )


EXPECTED_COUNT = 2689
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "challenge").is_dir():
            return candidate
    raise WorkspaceError(f"could not locate repository root from {start}")


def _png_resolution(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(24)
    if len(prefix) < 24 or prefix[:8] != PNG_SIGNATURE or prefix[12:16] != b"IHDR":
        return "unknown"
    width, height = struct.unpack(">II", prefix[16:24])
    return f"{width}x{height}"


def _negative_rows(frames: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = [
        dict(row)
        for row in frames
        if row.get("population_role") == "DEV"
        and row.get("paper_subset") == "DEV_NEG2689"
        and row.get("is_positive") == "false"
    ]
    if len(rows) != EXPECTED_COUNT:
        raise WorkspaceError(
            f"frozen DEV negative count mismatch: expected={EXPECTED_COUNT}, actual={len(rows)}"
        )
    return sorted(rows, key=lambda row: row["frame_id"])


def _target_name(row: Mapping[str, str]) -> str:
    source = Path(row.get("source_image_path", ""))
    if not source.name or source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise WorkspaceError(f"invalid DEV negative source path: {source}")
    return source.name


def _verify_published_session(
    root: Path, destination: Path, rows: Sequence[Mapping[str, str]]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    expected_names: set[str] = set()
    for index, row in enumerate(rows, start=1):
        name = _target_name(row)
        if name in expected_names:
            raise WorkspaceError(f"duplicate DEV negative destination name: {name}")
        expected_names.add(name)
        target = destination / "rgb" / name
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError(f"materialized DEV negative is missing/unsafe: {target}")
        if sha256_file(target) != row["image_sha256"]:
            raise WorkspaceError(f"materialized DEV negative SHA mismatch: {target}")
        mapping[row["frame_id"]] = workspace_relative(target, root)
        if index % 500 == 0 or index == len(rows):
            print(f"verified existing DEV negative {index}/{len(rows)}", flush=True)
    actual_names = {
        path.name
        for path in (destination / "rgb").iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if actual_names != expected_names:
        raise WorkspaceError(
            "materialized DEV negative membership differs from frozen rows: "
            f"expected={len(expected_names)}, actual={len(actual_names)}"
        )
    return mapping


def _session_metadata(rows: Sequence[Mapping[str, str]], resolution_counts: Counter[str]) -> dict[str, Any]:
    return {
        "session_id": "dev_negative",
        "population_role": "DEV",
        "paper_subset": "DEV_NEG2689",
        "object_type": "none",
        "lighting": "unknown",
        "capture_protocol": "legacy_audited_negative_import",
        "resolution": "mixed" if len(resolution_counts) > 1 else next(iter(resolution_counts)),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "camera": {
            "intrinsics_quality": "PROVIDED_UNVERIFIED",
            "intrinsics_source": "copied source camera_info.json/cam_K.txt",
            "note": "source camera metadata is not assumed valid for every mixed-resolution frame",
        },
        "default_tags": {
            "occlusion": "unknown",
            "truncation": "unknown",
            "distance_bin": "unknown",
            "size_bin": "unknown",
            "elevation_bin": "unknown",
            "view_bin": "unknown",
        },
        "source_dataset": "real_gt_v2_negative_audited",
        "source_preserved_read_only": True,
        "storage_mode": "independent_copy",
        "frame_count": len(rows),
        "known_duplicate_membership": ["dev_negative__000238", "dev_negative__000239"],
    }


def materialize_dev_negative(root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise WorkspaceError(f"evaluation workspace not found: {root}")
    repo_root = (repo_root or _find_repo_root(root)).resolve()
    frames = load_frames(root)
    rows = _negative_rows(frames)
    destination = root / "dev_existing/sessions/dev_negative"
    mapping: dict[str, str]

    if destination.is_symlink():
        raise WorkspaceError(f"DEV negative destination must not be a symlink: {destination}")
    if destination.exists():
        mapping = _verify_published_session(root, destination, rows)
        resolution_counts = Counter(
            _png_resolution(root / mapping[row["frame_id"]]) for row in rows
        )
    else:
        source_snapshots: dict[Path, tuple[int, int, str]] = {}
        target_names: set[str] = set()
        sources: dict[str, Path] = {}
        for row in rows:
            source_value = row.get("source_image_path", "")
            source_relative = Path(source_value)
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise WorkspaceError(f"unsafe DEV negative source path: {source_value}")
            source = (repo_root / source_relative).resolve()
            try:
                source.relative_to(repo_root)
            except ValueError as exc:
                raise WorkspaceError(
                    f"DEV negative source escapes repository: {source}"
                ) from exc
            if not source.is_file():
                raise WorkspaceError(f"DEV negative source missing: {source}")
            name = _target_name(row)
            if name in target_names:
                raise WorkspaceError(f"duplicate DEV negative destination name: {name}")
            target_names.add(name)
            digest = sha256_file(source)
            if digest != row.get("image_sha256") or digest != row.get("source_image_sha256"):
                raise WorkspaceError(f"DEV negative source SHA drift: {source}")
            stat = source.stat()
            source_snapshots[source] = (stat.st_size, stat.st_mtime_ns, digest)
            sources[row["frame_id"]] = source

        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".dev_negative.materializing.", dir=destination.parent)
        )
        mapping = {}
        resolution_counts: Counter[str] = Counter()
        try:
            for index, row in enumerate(rows, start=1):
                source = sources[row["frame_id"]]
                target = stage / "rgb" / _target_name(row)
                digest = copy2_verified(source, target)
                if digest != row["image_sha256"]:
                    raise WorkspaceError(f"DEV negative copy SHA drift: {source}")
                resolution_counts[_png_resolution(target)] += 1
                mapping[row["frame_id"]] = (
                    f"dev_existing/sessions/dev_negative/rgb/{target.name}"
                )
                if index % 500 == 0 or index == len(rows):
                    print(f"materialized DEV negative {index}/{len(rows)}", flush=True)

            source_root = next(iter(sources.values())).parent.parent
            if any(source.parent.parent != source_root for source in sources.values()):
                raise WorkspaceError("DEV negative sources unexpectedly span multiple roots")
            for metadata_name in ("camera_info.json", "cam_K.txt"):
                source_metadata = source_root / metadata_name
                if source_metadata.is_file():
                    copy2_verified(source_metadata, stage / metadata_name)
            atomic_write_json(stage / "session.json", _session_metadata(rows, resolution_counts))

            for source, before in source_snapshots.items():
                stat = source.stat()
                after = (stat.st_size, stat.st_mtime_ns, sha256_file(source))
                if after != before:
                    raise WorkspaceError(f"DEV negative source changed during copy: {source}")
            os.rename(stage, destination)
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise

    updated_frames = [dict(row) for row in frames]
    for row in updated_frames:
        if row.get("frame_id") not in mapping:
            continue
        row["image_path"] = mapping[row["frame_id"]]
        row["storage_mode"] = "independent_copy"
    validate_active_image_sha_uniqueness(updated_frames)
    validate_frozen_dev_evaluation_population(evaluation_population_views(updated_frames))

    provenance_path = root / "manifests/import_provenance.csv"
    provenance = read_csv(provenance_path)
    provenance_by_frame = {row.get("active_frame_id", ""): row for row in provenance}
    for frame_id, image_path in mapping.items():
        row = provenance_by_frame.get(frame_id)
        if row is None:
            raise WorkspaceError(f"missing DEV negative provenance row: {frame_id}")
        row["disposition"] = "ACTIVE_INDEPENDENT_COPY"
        row["destination_image_path"] = image_path

    atomic_write_csv(root / "manifests/frames.csv", updated_frames, FRAME_COLUMNS)
    atomic_write_csv(provenance_path, provenance, PROVENANCE_COLUMNS)
    write_manifest_views(root, updated_frames, enforce_frozen_dev=True)
    write_reports(root, updated_frames)
    return {
        "frame_count": len(rows),
        "destination": str(destination),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "storage_mode": "independent_copy",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/evaluation/pallet_eval_v1"),
        help="evaluation workspace root",
    )
    parser.add_argument("--repo-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = materialize_dev_negative(args.root, args.repo_root)
    except (WorkspaceError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[FAIL] DEV negative materialization: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
