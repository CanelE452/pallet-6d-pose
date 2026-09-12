#!/usr/bin/env python3
"""Verify the reviewable code bundle against its immutable manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file(): failures.append(f"missing: {relative}")
        elif sha(path) != expected: failures.append(f"SHA mismatch: {relative}")
    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    print(f"PASS: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
