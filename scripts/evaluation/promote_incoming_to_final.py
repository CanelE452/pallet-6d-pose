#!/usr/bin/env python3
"""CLI for synchronizing reviewed incoming annotations into evaluation data.

The reusable implementation lives in :mod:`incoming_promotion`.  The default
is a true dry run; files are changed only when ``--apply`` is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:  # Package import from tests/other modules.
    from .incoming_promotion import (
        annotated_stems,
        promote_annotated_incoming,
        promote_annotations,
        promote_incoming_annotation,
        resolve_auto_destination,
    )
except ImportError:  # Direct ``python scripts/evaluation/...`` execution.
    from incoming_promotion import (  # type: ignore[no-redef]
        annotated_stems,
        promote_annotated_incoming,
        promote_annotations,
        promote_incoming_annotation,
        resolve_auto_destination,
    )


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "data/evaluation/pallet_eval_v1"


def _print_summary(summary: dict) -> None:
    print(
        "동기화 "
        f"신규 {summary['promoted']}장 · 갱신 {summary['updated']}장 · "
        f"제거 {summary['removed']}장 · "
        f"동일 {summary['skipped_existing']}장 · 태그 {summary['metadata_synced']}행 · "
        f"세션 메타데이터 {summary['session_metadata_synced']}건"
    )
    for destination, count in sorted(summary["by_dest"].items()):
        print(f"  -> {destination}: 신규 {count}장")
    if summary["missing_source"]:
        print(f"  원본 이미지 없음: {summary['missing_source'][:10]}")
    if summary["unresolved"]:
        print(f"  목적지 미해결: {summary['unresolved']}")


def _planned_auto_count(root: Path) -> tuple[int, list[str]]:
    total = 0
    unresolved: list[str] = []
    annotation_root = root / "incoming/annotations"
    if not annotation_root.is_dir():
        return total, unresolved
    for annotation_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        if resolve_auto_destination(root, annotation_dir) is None:
            unresolved.append(annotation_dir.name)
            continue
        total += len(annotated_stems(annotation_dir))
    return total, unresolved


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="모든 incoming annotation view를 메타데이터 기반 목적지에 동기화",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="root 기준 incoming/annotations/<session>__<material> 경로",
    )
    parser.add_argument("--source-session")
    parser.add_argument("--dest-session")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 복사·갱신한다. 생략하면 dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = args.root.resolve()
    if args.auto:
        if not args.apply:
            count, unresolved = _planned_auto_count(root)
            print(f"dry-run — 완성 주석 {count}장을 검사·동기화할 예정")
            if unresolved:
                print(f"  목적지 미해결: {unresolved}")
            print("실행하려면 --apply")
            return 0
        _print_summary(promote_annotated_incoming(root, refresh=True))
        return 0

    if not (args.annotations and args.source_session and args.dest_session):
        print("FAIL: --auto 또는 --annotations/--source-session/--dest-session 필요")
        return 1
    annotation_dir = args.annotations
    if not annotation_dir.is_absolute():
        annotation_dir = root / annotation_dir
    if not annotation_dir.is_dir():
        print(f"FAIL: annotation 폴더 없음: {annotation_dir}")
        return 1
    stems = annotated_stems(annotation_dir)
    if not stems:
        print("동기화할 완성 주석이 없습니다.")
        return 1
    if not args.apply:
        print(f"dry-run — 완성 주석 {len(stems)}장을 검사·동기화할 예정")
        print("실행하려면 --apply")
        return 0
    summary = promote_annotations(
        root,
        annotation_dir,
        args.source_session,
        args.dest_session,
        stems=stems,
        refresh=True,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "annotated_stems",
    "build_argument_parser",
    "main",
    "promote_annotated_incoming",
    "promote_annotations",
    "promote_incoming_annotation",
    "resolve_auto_destination",
]
