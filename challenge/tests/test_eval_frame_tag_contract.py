from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.evaluation.eval_workspace as workspace
from scripts.evaluation.eval_workspace import (
    FRAME_TAG_COLUMNS,
    WorkspaceError,
    canonical_frame_tag_identity,
    discover_final_rows,
    infer_annotation_document_tags,
    load_frame_tag_overrides,
    load_session_metadata,
    resolve_effective_frame_tags,
    update_frame_tags_csv,
)


def _annotation_document(
    *,
    object_type: str = "wood_small_80x59x14",
    occlusion_level: str = "unknown",
    reason: str = "occluded",
    truncated: bool = True,
) -> dict[str, object]:
    return {
        "objects": [
            {
                "object_type": object_type,
                "occlusion_level": occlusion_level,
                "truncation": {
                    "is_truncated": truncated,
                    "bbox_outside_fraction": 0.25 if truncated else 0.0,
                },
                "keypoint_annotations": [{"reason": reason}],
                # These values must never be used to invent condition bins.
                "projected_bbox": [-20, 10, 1400, 700],
                "pose": {"translation": [0.0, 0.0, 7.5]},
            }
        ]
    }


def _write_session(session: Path, metadata: dict[str, object]) -> None:
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_session_metadata_load_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = {
        "session_id": "plastic_night_01",
        "population_role": "FINAL",
        "object_type": "plastic",
        "lighting": "night",
        "default_tags": {"occlusion": "none"},
    }
    _write_session(session, metadata)
    assert load_session_metadata(session) == metadata

    (session / "session.json").write_text(
        json.dumps({"default_tags": ["not", "an", "object"]}),
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceError, match="default_tags must be an object"):
        load_session_metadata(session)

    with pytest.raises(WorkspaceError, match="missing session.json"):
        load_session_metadata(tmp_path / "missing")


def test_document_inference_and_effective_precedence_without_bin_guessing() -> None:
    document = _annotation_document()
    inferred = infer_annotation_document_tags(document)
    assert inferred == {
        "object_type": "wood",
        "occlusion": "medium",
        "truncation": "mild",
    }

    session = {
        "object_type": "plastic",
        "lighting": "night",
        "default_tags": {"occlusion": "none", "truncation": "none"},
    }
    effective, sources = resolve_effective_frame_tags(
        session,
        {
            "occlusion": "heavy",
            "truncation": "unknown",
            "distance_bin": "unknown",
        },
        annotation_document=document,
    )
    assert effective == {
        "object_type": "wood",
        "lighting": "night",
        # environment 는 2026-09-01 에 추가된 도메인 축이다. 근거가 없으므로 unknown.
        "environment": "unknown",
        "occlusion": "heavy",
        "truncation": "mild",
        "distance_bin": "unknown",
        "size_bin": "unknown",
        "elevation_bin": "unknown",
        "view_bin": "unknown",
    }
    assert sources == {
        "object_type": "JSON",
        "lighting": "SESSION",
        # environment 는 근거가 없어 어느 층에서도 값이 오지 않는다.
        "environment": "UNSET",
        "occlusion": "FRAME",
        "truncation": "JSON",
        "distance_bin": "UNSET",
        "size_bin": "UNSET",
        "elevation_bin": "UNSET",
        "view_bin": "UNSET",
    }

    # Explicit unknown means "clear this override", so JSON evidence wins.
    effective, sources = resolve_effective_frame_tags(
        session,
        {"occlusion": "unknown"},
        annotation_document=document,
    )
    assert effective["occlusion"] == "medium"
    assert sources["occlusion"] == "JSON"


def test_path_and_live_document_use_one_resolver_and_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "000123.json"
    document = _annotation_document(truncated=False, reason="visible")
    annotation.write_text(json.dumps(document), encoding="utf-8")
    from_path = resolve_effective_frame_tags({}, annotation_path=annotation)
    from_document = resolve_effective_frame_tags({}, annotation_document=document)
    assert from_path == from_document

    with pytest.raises(WorkspaceError, match="mutually exclusive"):
        resolve_effective_frame_tags(
            {},
            annotation_path=annotation,
            annotation_document=document,
        )


def test_atomic_update_roundtrip_and_unknown_clears_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _write_session(
        session,
        {
            "object_type": "plastic",
            "lighting": "day",
            "default_tags": {"distance_bin": "mid"},
        },
    )

    real_replace = workspace.os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        assert Path(source).parent == session
        real_replace(source, destination)

    monkeypatch.setattr(workspace.os, "replace", recording_replace)
    tags_path = update_frame_tags_csv(
        session,
        "rgb/000123.png",
        {
            "distance_bin": "far",
            "size_bin": "small",
            "elevation_bin": "high",
            "view_bin": "oblique",
            "occlusion": "medium",
            "truncation": "none",
        },
    )
    assert replacements[-1][1] == tags_path
    rows = list(csv.DictReader(tags_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert tuple(rows[0]) == FRAME_TAG_COLUMNS
    assert rows[0] == {
        "frame": "000123.png",
        "distance_bin": "far",
        "size_bin": "small",
        "elevation_bin": "high",
        "view_bin": "oblique",
        "occlusion": "medium",
        "truncation": "none",
    }

    update_frame_tags_csv(session, "session__000123", {"distance_bin": "unknown"})
    override = load_frame_tag_overrides(session)["000123"]
    effective, sources = resolve_effective_frame_tags(
        load_session_metadata(session),
        override,
    )
    assert effective["distance_bin"] == "mid"
    assert sources["distance_bin"] == "SESSION"
    assert effective["size_bin"] == "small"
    assert sources["size_bin"] == "FRAME"

    update_frame_tags_csv(
        session,
        "000123.png",
        {field: "unknown" for field in workspace.FRAME_TAG_FIELDS},
    )
    assert load_frame_tag_overrides(session) == {}
    assert tags_path.read_text(encoding="utf-8") == ",".join(FRAME_TAG_COLUMNS) + "\n"


def test_duplicate_canonical_rows_fail_before_atomic_update(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_session(session, {})
    tags_path = session / "frame_tags.csv"
    tags_path.write_text(
        ",".join(FRAME_TAG_COLUMNS)
        + "\n000123.png,far,small,high,front,medium,none"
        + "\nsession__000123,near,large,low,rear,none,none\n",
        encoding="utf-8",
    )
    before = tags_path.read_bytes()
    with pytest.raises(WorkspaceError, match="duplicate/conflicting frame tag alias"):
        update_frame_tags_csv(session, "000123.png", {"distance_bin": "mid"})
    assert tags_path.read_bytes() == before
    assert canonical_frame_tag_identity("rgb/000123.png") == "000123"
    assert canonical_frame_tag_identity("plastic_day_01__000123") == (
        "plastic_day_01__000123")
    assert canonical_frame_tag_identity(
        "plastic_day_01__000123", session_id="plastic_day_01") == "000123"


def test_final_discovery_uses_shared_precedence(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "final/positive/sessions/wood_night_01"
    _write_session(
        session,
        {
            "session_id": "wood_night_01",
            "population_role": "FINAL",
            "object_type": "plastic",
            "lighting": "night",
            "default_tags": {"occlusion": "none", "truncation": "none"},
        },
    )
    rgb = session / "rgb"
    rgb.mkdir()
    (rgb / "000123.png").write_bytes(b"image")
    annotation = root / "final/positive/annotations/wood_night_01/000123.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(_annotation_document()), encoding="utf-8")
    update_frame_tags_csv(session, "000123.png", {"occlusion": "heavy"})

    [row] = discover_final_rows(root)
    assert row["object_type"] == "wood"  # JSON beats session.
    assert row["lighting"] == "night"
    assert row["occlusion"] == "heavy"  # FRAME beats JSON.
    assert row["truncation"] == "mild"  # JSON beats session.
    assert row["distance_bin"] == "unknown"
    assert row["size_bin"] == "unknown"
    assert row["elevation_bin"] == "unknown"
    assert row["view_bin"] == "unknown"
