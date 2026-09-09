from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.evaluation.eval_workspace import WorkspaceError, read_csv, scaffold_workspace
from scripts.evaluation.import_incoming_capture import import_capture


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_stub(width: int, height: int, marker: bytes) -> bytes:
    # The importer needs only the PNG signature/IHDR dimensions; ZIP CRC still
    # verifies every byte copied from the source member.
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + marker
    )


def _camera_info() -> dict[str, object]:
    return {
        "width": 640,
        "height": 480,
        "K": [[608.0, 0.0, 326.0], [0.0, 607.0, 239.0], [0.0, 0.0, 1.0]],
        "fx": 608.0,
        "fy": 607.0,
        "cx": 326.0,
        "cy": 239.0,
    }


def _write_archive(path: Path, images: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in images.items():
            archive.writestr(name, payload)
        archive.writestr("camera_info.json", json.dumps(_camera_info()))


def test_import_is_independent_inactive_and_source_preserving(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    archive = tmp_path / "morning.zip"
    _write_archive(
        archive,
        {
            "000000.png": _png_stub(640, 480, b"first"),
            "000001.png": _png_stub(640, 480, b"second"),
        },
    )
    source_before = (archive.stat().st_size, archive.stat().st_mtime_ns)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    metadata = import_capture(
        root,
        archive,
        "morning_capture",
        "day",
        expected_archive_sha256=archive_sha,
    )

    session = root / "incoming/sessions/morning_capture"
    assert metadata["active_evaluation_member"] is False
    assert metadata["population_role"] is None
    assert metadata["object_type"] == "unknown"
    assert metadata["lighting"] == "day"
    assert metadata["image_count"] == 2
    assert (session / "rgb/000000.png").read_bytes() == _png_stub(640, 480, b"first")
    assert not (session / "rgb/000000.png").is_symlink()
    assert (session / "cam_K.txt").read_text(encoding="utf-8").count("\n") == 3
    assert source_before == (archive.stat().st_size, archive.stat().st_mtime_ns)
    rows = read_csv(session / "manifests/frames.csv")
    assert len(rows) == 2
    assert {row["duplicate_in_capture"] for row in rows} == {"false"}
    # Incoming rows remain outside the active evaluation source of truth.
    assert all(
        "morning_capture" not in row.get("frame_id", "")
        for row in read_csv(root / "manifests/frames.csv")
    )
    incoming_sessions = read_csv(root / "incoming/manifests/sessions.csv")
    assert [row["session_id"] for row in incoming_sessions] == ["morning_capture"]


def test_duplicate_bytes_are_preserved_but_explicitly_audited(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    archive = tmp_path / "night.zip"
    duplicate = _png_stub(640, 480, b"same")
    _write_archive(archive, {"000010.png": duplicate, "000011.png": duplicate})

    metadata = import_capture(root, archive, "night_capture", "night")

    assert metadata["duplicate_audit"]["frames_duplicate_in_capture"] == 2
    rows = read_csv(root / "incoming/sessions/night_capture/manifests/frames.csv")
    assert [row["duplicate_in_capture"] for row in rows] == ["true", "true"]
    assert all(row["duplicate_references"].startswith("same:") for row in rows)


@pytest.mark.parametrize("bad_name", ["../escape.png", "/absolute.png", "nested/frame.png"])
def test_unsafe_or_nested_image_member_fails_without_publish(
    tmp_path: Path, bad_name: str
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    archive = tmp_path / "bad.zip"
    _write_archive(archive, {bad_name: _png_stub(640, 480, b"bad")})

    with pytest.raises(WorkspaceError):
        import_capture(root, archive, "bad_capture", "day")

    assert not (root / "incoming/sessions/bad_capture").exists()
    sessions_root = root / "incoming/sessions"
    assert not list(sessions_root.glob(".bad_capture.importing.*"))


def test_wrong_expected_archive_sha_fails_before_extraction(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    archive = tmp_path / "capture.zip"
    _write_archive(archive, {"000000.png": _png_stub(640, 480, b"frame")})

    with pytest.raises(WorkspaceError, match="archive SHA mismatch"):
        import_capture(root, archive, "capture", "day", expected_archive_sha256="0" * 64)

    assert not (root / "incoming/sessions/capture").exists()
