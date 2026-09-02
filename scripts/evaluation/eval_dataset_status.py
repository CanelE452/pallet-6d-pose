#!/usr/bin/env python3
"""Refresh pallet evaluation manifests and human-readable status reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:  # Package import from editor/tests.
    from .eval_workspace import (
        WorkspaceError,
        load_targets,
        progress_line,
        refresh_frame_index,
        scaffold_workspace,
        write_reports,
    )
    from .incoming_promotion import promote_incoming_annotation
except ImportError:  # Direct ``python scripts/evaluation/...`` execution.
    from eval_workspace import (  # type: ignore[no-redef]
        WorkspaceError,
        load_targets,
        progress_line,
        refresh_frame_index,
        scaffold_workspace,
        write_reports,
    )
    from incoming_promotion import (  # type: ignore[no-redef]
        promote_incoming_annotation,
    )


def _assert_workspace_path(root: Path, path: str | Path | None, name: str) -> None:
    if path is None:
        return
    candidate = Path(path)
    if not candidate.is_absolute():
        cwd_candidate = candidate.resolve(strict=False)
        try:
            cwd_candidate.relative_to(root.resolve())
            candidate = cwd_candidate
        except ValueError:
            candidate = root / candidate
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise WorkspaceError(f"{name} must be inside evaluation workspace: {candidate}") from exc


def refresh_after_annotation(
    root: str | Path,
    annotation_path: str | Path,
    image_path: str | Path | None = None,
    deleted: bool = False,
    synchronize_session: bool = False,
) -> str:
    """Lightweight editor hook: refresh membership/reports and return one line.

    For an incoming annotation, synchronize only this frame's independent
    image/JSON/overlay/tag copy before rebuilding the combined report.  Direct
    evaluation annotations need no copy.  The function never reads or writes
    legacy source paths.
    """

    dataset_root = Path(root).resolve()
    if not dataset_root.is_dir():
        raise WorkspaceError(f"evaluation workspace does not exist: {dataset_root}")
    _assert_workspace_path(dataset_root, annotation_path, "annotation_path")
    _assert_workspace_path(dataset_root, image_path, "image_path")

    synchronized: dict = {}
    try:
        synchronized = promote_incoming_annotation(
            dataset_root,
            Path(annotation_path),
            deleted=bool(deleted),
            all_complete=bool(synchronize_session),
        )
    except Exception as exc:  # A derived-copy failure must not roll back valid GT.
        print(f"[WARN] incoming frame synchronization skipped: {exc}")

    frames = refresh_frame_index(dataset_root, rehash_final=True)
    progress = write_reports(dataset_root, frames)
    line = progress_line(progress, load_targets(dataset_root))
    changes = []
    for key, label in (
        ("promoted", "new"),
        ("updated", "updated"),
        ("removed", "removed"),
        ("metadata_synced", "tag rows synced"),
        ("session_metadata_synced", "session metadata synced"),
    ):
        count = int(synchronized.get(key, 0))
        if count:
            changes.append(f"{count} {label}")
    if changes:
        line = f"{line}  (incoming: {', '.join(changes)})"
    return line


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/evaluation/pallet_eval_v1"),
        help="evaluation workspace root",
    )
    parser.add_argument(
        "--verify-sha",
        action="store_true",
        help="compatibility flag; routine refresh now always rehashes FINAL images",
    )
    parser.add_argument(
        "--initialize-empty",
        action="store_true",
        help="create an empty workspace contract before reporting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = args.root.resolve()
    if args.initialize_empty:
        scaffold_workspace(root)
    if not root.is_dir():
        raise SystemExit(
            f"workspace not found: {root}\n"
            "Run import_existing_evaluation_data.py or pass --initialize-empty."
        )
    try:
        frames = refresh_frame_index(root, rehash_final=True)
        progress = write_reports(root, frames)
    except (WorkspaceError, OSError, ValueError) as exc:
        raise SystemExit(f"[FAIL] evaluation workspace refresh: {exc}") from exc
    report = (root / "reports/ANNOTATION_PROGRESS.md").read_text(encoding="utf-8")
    print(report.rstrip())
    print()
    print(progress_line(progress, load_targets(root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
