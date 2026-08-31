#!/usr/bin/env python3
"""Rebuild annotation review PNGs from the evaluation manifest.

Only active ``*.json`` annotations are visited.  Source images are resolved
from ``manifests/frames.csv`` and are opened read-only; review PNGs are always
written below the annotation session's ``_overlays`` directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

try:  # Package import from evaluation importer/tests.
    from .annotate_draw import (
        annotation_overlay_path,
        render_saved_annotation_overlay,
        validate_saved_annotation_overlay,
    )
except ImportError:  # Direct ``python scripts/annotate/...`` execution.
    from annotate_draw import (  # type: ignore[no-redef]
        annotation_overlay_path,
        render_saved_annotation_overlay,
        validate_saved_annotation_overlay,
    )


SCOPES = {
    "dev_existing": Path("dev_existing") / "annotations",
    "legacy_unverified": Path("legacy_unverified") / "annotations",
    "final": Path("final") / "positive" / "annotations",
}


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError(f"cannot find repository root above {start}")


def _path_candidates(raw: str, dataset_root: Path, repo_root: Path):
    value = Path(raw).expanduser()
    if value.is_absolute():
        yield value.resolve()
        return
    # Importers may store either workspace-relative or repository-relative
    # paths.  Supporting both is deterministic and does not guess a filename.
    seen = set()
    for candidate in (dataset_root / value, repo_root / value, Path.cwd() / value):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _manifest_annotation_rows(dataset_root: Path, repo_root: Path):
    manifest = dataset_root / "manifests" / "frames.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"frames manifest not found: {manifest}")
    rows = {}
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if "annotation_path" not in fields:
            raise ValueError(f"manifest has no annotation_path column: {manifest}")
        if not ({"image_path", "source_image_path"} & fields):
            raise ValueError(
                f"manifest has neither image_path nor source_image_path: {manifest}")
        for row_number, row in enumerate(reader, start=2):
            raw_annotation = (row.get("annotation_path") or "").strip()
            if not raw_annotation:
                continue
            for candidate in _path_candidates(
                    raw_annotation, dataset_root, repo_root):
                previous = rows.get(candidate)
                if previous is not None and previous != row:
                    raise ValueError(
                        "duplicate manifest annotation_path at row "
                        f"{row_number}: {raw_annotation}")
                rows[candidate] = row
    return rows


def _resolve_image(row, dataset_root: Path, repo_root: Path) -> Path:
    attempted = []
    for column in ("image_path", "source_image_path"):
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        for candidate in _path_candidates(raw, dataset_root, repo_root):
            attempted.append(str(candidate))
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "manifest image is missing; tried: " + ", ".join(attempted))


def rebuild(dataset_root: Path, scopes, *, force=False) -> dict:
    repo_root = _find_repo_root(Path(__file__).parent)
    dataset_root = dataset_root.resolve()
    manifest_rows = _manifest_annotation_rows(dataset_root, repo_root)
    annotation_paths = []
    annotation_roots = []
    for scope in scopes:
        annotation_root = dataset_root / SCOPES[scope]
        if annotation_root.is_dir():
            annotation_roots.append(annotation_root.resolve())
            annotation_paths.extend(sorted(annotation_root.rglob("*.json")))
    # An annotation can only belong to one physical scope, but de-duplicate in
    # case a caller repeats --scope.
    annotation_paths = sorted({path.resolve() for path in annotation_paths})
    expected_overlays = {
        Path(annotation_overlay_path(path)).resolve()
        for path in annotation_paths
    }
    cached_overlays = {
        path.resolve()
        for annotation_root in annotation_roots
        for path in annotation_root.rglob("*.png")
        if "_overlays" in path.relative_to(annotation_root).parts
    }
    orphan_overlays = sorted(cached_overlays - expected_overlays)

    generated = skipped = failed = 0
    errors = []
    for annotation_path in annotation_paths:
        overlay_path = Path(annotation_overlay_path(annotation_path))
        row = manifest_rows.get(annotation_path)
        if row is None:
            failed += 1
            errors.append(f"manifest row missing: {annotation_path}")
            continue
        try:
            source_image = _resolve_image(row, dataset_root, repo_root)
            # Membership and JSON validity are checked even for an existing
            # cache.  Cache presence cannot legitimize an orphan/malformed GT.
            validate_saved_annotation_overlay(annotation_path)
            cache_is_fresh = (
                overlay_path.is_file()
                and overlay_path.stat().st_mtime_ns >= max(
                    annotation_path.stat().st_mtime_ns,
                    source_image.stat().st_mtime_ns,
                )
            )
            if cache_is_fresh and not force:
                skipped += 1
                continue
            render_saved_annotation_overlay(
                source_image, annotation_path, overlay_path)
            generated += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{annotation_path}: {exc}")

    overlay_count = sum(
        Path(annotation_overlay_path(path)).is_file()
        for path in annotation_paths)
    return {
        "annotations": len(annotation_paths),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "overlays": overlay_count,
        "orphan_overlays": [str(path) for path in orphan_overlays],
        "errors": errors,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--scope", action="append", choices=tuple(SCOPES),
        help="repeatable; default: all scopes")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    repo_root = _find_repo_root(Path(__file__).parent)
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")
    scopes = list(dict.fromkeys(args.scope or SCOPES.keys()))

    try:
        result = rebuild(dataset_root, scopes, force=args.force)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for error in result["errors"]:
        print(f"[WARN] {error}", file=sys.stderr)
    print(f"annotation JSON : {result['annotations']}")
    print(f"generated       : {result['generated']}")
    print(f"skipped         : {result['skipped']}")
    print(f"failed          : {result['failed']}")
    print(f"overlays        : {result['overlays']}")
    print(f"orphan overlays : {len(result['orphan_overlays'])}")
    for path in result["orphan_overlays"]:
        print(f"[WARN] orphan overlay: {path}", file=sys.stderr)
    if (result["failed"]
            or result["orphan_overlays"]
            or result["annotations"] != result["overlays"]):
        print("[FAIL] active annotation JSON count != overlay count",
              file=sys.stderr)
        return 1
    print("[OK] active annotation JSON count == overlay count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
