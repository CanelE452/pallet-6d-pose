"""Small immutable-artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def immutable_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)


def require_new(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
