"""Move confirmed-invalid real GT JSON files into a recoverable local archive.

The tracked registry is the authority.  Images are deliberately left in their
original directories; only the exact annotation bytes named by the registry
may move.  Running without ``--apply`` is a read-only validation/dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "challenge/real_gt_v2/INVALID_GT_QUARANTINE.json"
REGISTRY_SCHEMA = "real_pallet_invalid_gt_quarantine_v1"
REAL_DATA_ROOT = (REPO_ROOT / "challenge/data/01_real").resolve()


class QuarantineError(RuntimeError):
    """Raised before a move when the registry or filesystem state is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise QuarantineError(f"{field} must be a non-empty repo-relative path")
    resolved = (REPO_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise QuarantineError(f"{field} escapes the repository: {relative}") from exc
    return resolved


def _load_registry() -> tuple[Path, list[dict[str, Any]]]:
    try:
        registry = json.loads(REGISTRY_PATH.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuarantineError(f"cannot read registry: {REGISTRY_PATH}: {exc}") from exc
    if not isinstance(registry, dict):
        raise QuarantineError("registry root must be an object")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise QuarantineError("unexpected registry schema")
    if registry.get("status") != "QUARANTINED":
        raise QuarantineError("registry status must be QUARANTINED")
    if registry.get("images_retained_in_place") is not True:
        raise QuarantineError("registry must explicitly retain images in place")

    entries = registry.get("entries")
    if not isinstance(entries, list) or registry.get("entry_count") != len(entries):
        raise QuarantineError("registry entry_count mismatch")
    archive_root = _repo_path(registry.get("archive_root"), "archive_root")
    if archive_root.parent != (REPO_ROOT / "_archive").resolve():
        raise QuarantineError("archive_root must be one named child of repo/_archive")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise QuarantineError(f"entries[{index}] must be an object")
        source_rel = entry.get("source_path")
        expected_sha = entry.get("source_sha256")
        source = _repo_path(source_rel, f"entries[{index}].source_path")
        try:
            source.relative_to(REAL_DATA_ROOT)
        except ValueError as exc:
            raise QuarantineError(f"source is outside challenge/data/01_real: {source_rel}") from exc
        if source.suffix.lower() != ".json":
            raise QuarantineError(f"source is not JSON: {source_rel}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise QuarantineError(f"invalid SHA-256 at entries[{index}]")
        if source_rel in seen_paths or expected_sha in seen_hashes:
            raise QuarantineError(f"duplicate path or SHA-256 at entries[{index}]")
        seen_paths.add(source_rel)
        seen_hashes.add(expected_sha)
    return archive_root, entries


def _image_exists_beside(source: Path) -> bool:
    return any(source.with_suffix(ext).is_file() for ext in (".png", ".jpg", ".jpeg"))


def quarantine(*, apply: bool) -> tuple[int, int]:
    archive_root, entries = _load_registry()
    pending: list[tuple[Path, Path, str]] = []
    already = 0

    # Validate the complete transaction before moving the first file.
    for entry in entries:
        source_rel = entry["source_path"]
        expected_sha = entry["source_sha256"]
        source = _repo_path(source_rel, "source_path")
        destination = (archive_root / source_rel).resolve()
        try:
            destination.relative_to(archive_root)
        except ValueError as exc:
            raise QuarantineError(f"archive destination escapes archive root: {source_rel}") from exc

        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if source_exists and destination_exists:
            raise QuarantineError(f"source and archive both exist: {source_rel}")
        if not source_exists and not destination_exists:
            raise QuarantineError(f"source and archive are both missing: {source_rel}")
        actual_path = source if source_exists else destination
        actual_sha = _sha256(actual_path)
        if actual_sha != expected_sha:
            raise QuarantineError(
                f"SHA-256 mismatch for {source_rel}: expected {expected_sha}, got {actual_sha}"
            )
        if not _image_exists_beside(source):
            raise QuarantineError(f"source image is missing beside annotation: {source_rel}")
        if source_exists:
            pending.append((source, destination, source_rel))
        else:
            already += 1

    action = "MOVE" if apply else "WOULD_MOVE"
    for source, destination, source_rel in pending:
        print(f"{action} {source_rel} -> {destination.relative_to(REPO_ROOT)}")
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)

    if apply:
        for source, destination, source_rel in pending:
            if source.exists() or not destination.is_file():
                raise QuarantineError(f"post-move verification failed: {source_rel}")
    print(
        f"validated={len(entries)} moved={len(pending) if apply else 0} "
        f"pending={0 if apply else len(pending)} already_quarantined={already} "
        f"images_moved=0"
    )
    return (len(pending) if apply else 0), already


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the validated moves (default: dry-run only)",
    )
    args = parser.parse_args()
    quarantine(apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
