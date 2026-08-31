"""Focused tests for the non-destructive annotation review overlay cache."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ANNOTATE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "annotate"
sys.path.insert(0, str(ANNOTATE_DIR))

import annotate  # noqa: E402
import annotate_io  # noqa: E402
from annotate_draw import (  # noqa: E402
    MARGIN_B,
    MARGIN_L,
    MARGIN_R,
    MARGIN_T,
    _saved_overlay_points,
    annotation_overlay_path,
    render_saved_annotation_overlay,
)
from annotate_io import State  # noqa: E402
from rebuild_annotation_overlays import rebuild  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotation(offset=0.0):
    points = [
        [-25.0 + offset, 25.0], [130.0, 25.0], [130.0, 80.0],
        [-25.0, 80.0], [20.0, 5.0], [100.0, 5.0],
        [100.0, 60.0], [20.0, 60.0], [52.5, 42.5],
    ]
    reasons = [
        "truncated", "visible", "occluded", "unknown",
        "visible", "occluded", "truncated", "unknown", "visible",
    ]
    sources = [
        "manual_click", "extrapolated", "pnp_projected", "manual_click",
        "manual_click", "manual_click", "extrapolated", "unknown",
        "centroid_auto",
    ]
    return {
        "schema_version": "real_pallet_gt_v2",
        "objects": [{
            "projected_cuboid": points[:8],
            "projected_cuboid_centroid": points[8],
            # Deliberately stale: GT-v2 keypoint_annotations must win.
            "manual_kps": [[5.0, 5.0]] * 9,
            "extrapolated_mask": [False] * 9,
            "keypoint_annotations": [{
                "xy": point,
                "visibility": 2 if reason == "visible" else 1,
                "in_frame": 0.0 <= point[0] < 120.0 and 0.0 <= point[1] < 90.0,
                "source": source,
                "reason": reason,
            } for point, source, reason in zip(points, sources, reasons)],
        }],
    }


def _write_source(path: Path):
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(120, dtype=np.uint8)
    image[:, :, 1] = np.arange(90, dtype=np.uint8)[:, None]
    assert cv2.imwrite(str(path), image)


def test_saved_overlay_uses_v2_points_margin_and_atomic_refresh(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    annotation_path = tmp_path / "annotations" / "session" / "000001.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(json.dumps(_annotation()), encoding="utf-8")

    source_hash = _sha256(source)
    source_mtime = source.stat().st_mtime_ns
    annotation_hash = _sha256(annotation_path)
    stale_projection = _annotation()
    stale_projection["objects"][0]["projected_cuboid"][0] = [44.0, 44.0]
    _, points, projected, extrapolated, _ = _saved_overlay_points(stale_projection)
    assert points[0] == [-25.0, 25.0]
    assert points[0] != [5.0, 5.0]
    # Both saved representations remain visible: wireframe vs reviewed marker.
    assert projected[0] == [44.0, 44.0]
    assert projected[0] != points[0]
    assert extrapolated[1] is True

    overlay = Path(render_saved_annotation_overlay(source, annotation_path))
    assert overlay == annotation_path.parent / "_overlays" / "000001.png"
    rendered = cv2.imread(str(overlay))
    assert rendered.shape[:2] == (
        90 + MARGIN_T + MARGIN_B,
        120 + MARGIN_L + MARGIN_R,
    )
    before = _sha256(overlay)
    assert not (overlay.parent / "000001.tmp.png").exists()

    annotation_path.write_text(json.dumps(_annotation(offset=31.0)), encoding="utf-8")
    render_saved_annotation_overlay(source, annotation_path)
    assert _sha256(overlay) != before
    assert not (overlay.parent / "000001.tmp.png").exists()
    assert _sha256(source) == source_hash
    assert source.stat().st_mtime_ns == source_mtime
    assert annotation_hash != _sha256(annotation_path)  # only the test edit changed JSON

    with pytest.raises(ValueError, match="canonical annotation cache path"):
        render_saved_annotation_overlay(source, annotation_path, source)
    assert _sha256(source) == source_hash


def test_schema_dispatch_never_treats_partial_legacy_metadata_as_v2():
    legacy = _annotation()
    legacy.pop("schema_version")
    legacy["objects"][0]["manual_kps"][0] = [17.0, 19.0]
    legacy["objects"][0]["keypoint_annotations"] = [{
        "xy": [99.0, 101.0],
        "visibility": 2,
        "source": "manual_click",
        "reason": "visible",
    }]
    _, points, _, _, _ = _saved_overlay_points(legacy)
    assert points[0] == [17.0, 19.0]

    empty_v2 = _annotation()
    empty_v2["objects"][0]["keypoint_annotations"] = []
    with pytest.raises(ValueError, match="exactly 9"):
        _saved_overlay_points(empty_v2)

    malformed_v2 = _annotation()
    malformed_v2["objects"][0]["keypoint_annotations"][0]["xy"] = "bad"
    with pytest.raises(ValueError, match=r"keypoint_annotations\[0\]\.xy"):
        _saved_overlay_points(malformed_v2)


def test_editor_save_hook_and_delete_manage_overlay_without_touching_source(
        tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.png"
    _write_source(source)
    source_hash = _sha256(source)
    source_mtime = source.stat().st_mtime_ns
    out_json = tmp_path / "annotations" / "session" / "000002.json"
    out_png = out_json.with_suffix(".png")
    state = State()
    state.eval_root = None
    saved_annotation = _annotation()
    saved_annotation.pop("schema_version")
    monkeypatch.setattr(annotate, "_make_state_annotation",
                        lambda _state, _K: saved_annotation)

    assert annotate._save_state_annotation(
        state, np.eye(3), str(out_json), str(out_png), str(source))
    overlay = Path(annotation_overlay_path(out_json))
    assert out_json.is_file() and out_png.is_file() and overlay.is_file()
    assert overlay.stat().st_ino != source.stat().st_ino

    assert annotate._delete_annotation(
        state, str(out_json), str(out_png)) == "save-next"
    assert not out_json.exists()
    assert Path(str(out_json) + ".deleted").is_file()
    assert not overlay.exists()
    assert _sha256(source) == source_hash
    assert source.stat().st_mtime_ns == source_mtime

    # A derived-cache failure is a warning after the committed JSON save.
    failed_json = tmp_path / "annotations" / "session" / "000003.json"
    monkeypatch.setattr(
        annotate, "render_saved_annotation_overlay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))
    assert annotate._save_state_annotation(
        state, np.eye(3), str(failed_json),
        str(failed_json.with_suffix(".png")), str(source))
    assert failed_json.is_file()
    assert "[WARN] overlay failed" in capsys.readouterr().out


def test_json_commit_continues_overlay_and_refresh_when_compat_png_fails(
        tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.png"
    _write_source(source)
    annotation = tmp_path / "annotations" / "s1" / "000007.json"
    out_png = annotation.with_suffix(".png")
    saved_annotation = _annotation()
    saved_annotation.pop("schema_version")
    state = State()
    state.eval_root = None
    refresh_calls = []
    monkeypatch.setattr(annotate, "_make_state_annotation",
                        lambda _state, _K: saved_annotation)
    monkeypatch.setattr(
        annotate_io.os, "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no link")))
    monkeypatch.setattr(
        annotate_io.shutil, "copy2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no copy")))
    monkeypatch.setattr(
        annotate, "_refresh_evaluation_workspace",
        lambda _state, path, *, image_path=None, deleted=False:
            refresh_calls.append((path, image_path, deleted)),
    )

    assert annotate._save_state_annotation(
        state, np.eye(3), str(annotation), str(out_png), str(source))
    assert annotation.is_file()
    assert not out_png.exists()
    assert Path(annotation_overlay_path(annotation)).is_file()
    assert refresh_calls == [(str(annotation), str(source), False)]
    assert "[WARN] PNG compatibility copy failed" in capsys.readouterr().out


def test_eval_workspace_paths_are_role_matched_and_sources_are_protected(
        tmp_path):
    root = tmp_path / "pallet_eval_v1"
    dev_session = root / "dev_existing" / "sessions" / "dev_s1"
    final_session = root / "final" / "positive" / "sessions" / "final_s1"
    (dev_session / "rgb").mkdir(parents=True)
    (final_session / "rgb").mkdir(parents=True)
    dev_output = root / "dev_existing" / "annotations" / "dev_s1"
    final_output = root / "final" / "positive" / "annotations" / "final_s1"

    assert annotate._validate_evaluation_paths(
        str(root), str(dev_session), str(dev_output), "DEV") == (
            str(dev_session), str(dev_output))
    assert annotate._validate_evaluation_paths(
        str(root), str(final_session), str(final_output), "FINAL") == (
            str(final_session), str(final_output))
    with pytest.raises(ValueError, match="FINAL --seq"):
        annotate._validate_evaluation_paths(
            str(root), str(dev_session), str(dev_output), "FINAL")
    with pytest.raises(ValueError, match="DEV --seq"):
        annotate._validate_evaluation_paths(
            str(root), str(final_session), str(final_output), "DEV")
    with pytest.raises(ValueError, match="matching canonical namespace"):
        annotate._validate_evaluation_paths(
            str(root), str(dev_session), str(dev_session / "rgb"), "DEV")
    with pytest.raises(ValueError, match="matching canonical namespace"):
        annotate._validate_evaluation_paths(
            str(root), str(dev_session),
            str(root / "dev_existing" / "annotations" / "other"), "DEV")

    repo = Path(annotate._REPO)
    with pytest.raises(ValueError, match="audited real GT-v2"):
        annotate._require_nonlegacy_output_dir(
            str(repo / "challenge" / "real_gt_v2" / "migrated_gt" / "bad"),
            str(repo))
    with pytest.raises(ValueError, match="raw pallet data"):
        annotate._require_nonlegacy_output_dir(
            str(repo / "data" / "pallet" / "raw_data" / "outside" / "bad"),
            str(repo))


def test_final_workspace_delete_is_recoverable_and_refreshes_progress(
        tmp_path, monkeypatch):
    root = tmp_path / "pallet_eval_v1"
    source = (
        root / "final" / "positive" / "sessions" / "plastic_day_01" /
        "rgb" / "000005.png")
    source.parent.mkdir(parents=True)
    _write_source(source)
    annotation = (
        root / "final" / "positive" / "annotations" / "plastic_day_01" /
        "000005.json")
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(_annotation()), encoding="utf-8")
    out_png = annotation.with_suffix(".png")
    out_png.write_bytes(source.read_bytes())
    overlay = Path(render_saved_annotation_overlay(source, annotation))

    state = State()
    state.population_role = "FINAL"
    state.eval_root = str(root)
    refresh_calls = []
    monkeypatch.setattr(
        annotate,
        "_refresh_evaluation_workspace",
        lambda _state, path, *, image_path=None, deleted=False:
            refresh_calls.append((path, image_path, deleted)),
    )

    assert annotate._delete_annotation(
        state, str(annotation), str(out_png)) == "save-next"
    assert not annotation.exists()
    assert Path(str(annotation) + ".deleted").is_file()
    assert not overlay.exists()
    assert refresh_calls == [(str(annotation), None, True)]


def test_eval_root_save_and_delete_increment_then_decrement_final_progress(
        tmp_path, monkeypatch, capsys):
    from scripts.evaluation.eval_workspace import load_frames, scaffold_workspace

    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    session = root / "final" / "positive" / "sessions" / "plastic_day_01"
    source = session / "rgb" / "000006.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    _write_source(source)
    (session / "session.json").write_text(json.dumps({
        "session_id": "plastic_day_01",
        "population_role": "FINAL",
        "object_type": "plastic",
        "lighting": "day",
        "default_tags": {
            "occlusion": "none",
            "truncation": "none",
            "distance_bin": "near",
            "size_bin": "large",
            "elevation_bin": "low",
            "view_bin": "front",
        },
    }), encoding="utf-8")
    annotation = (
        root / "final" / "positive" / "annotations" / "plastic_day_01" /
        "000006.json")
    saved_annotation = _annotation()
    saved_annotation.pop("schema_version")
    monkeypatch.setattr(annotate, "_make_state_annotation",
                        lambda _state, _K: saved_annotation)
    state = State()
    state.population_role = "FINAL"
    state.eval_root = str(root)

    assert annotate._save_state_annotation(
        state, np.eye(3), str(annotation),
        str(annotation.with_suffix(".png")), str(source))
    assert "[Progress] FINAL 1/300" in capsys.readouterr().out
    rows = load_frames(root)
    assert len(rows) == 1 and rows[0]["is_annotated"] == "true"

    assert annotate._delete_annotation(
        state, str(annotation),
        str(annotation.with_suffix(".png"))) == "save-next"
    assert "[Progress] FINAL 0/300" in capsys.readouterr().out
    rows = load_frames(root)
    assert len(rows) == 1 and rows[0]["is_annotated"] == "false"


def test_backfill_resolves_manifest_and_reaches_one_overlay_per_json(tmp_path):
    root = tmp_path / "pallet_eval_v1"
    image = root / "dev_existing" / "sessions" / "s1" / "rgb" / "000004.png"
    annotation = root / "dev_existing" / "annotations" / "s1" / "000004.json"
    image.parent.mkdir(parents=True)
    annotation.parent.mkdir(parents=True)
    _write_source(image)
    valid_annotation = json.dumps(_annotation())
    annotation.write_text(valid_annotation, encoding="utf-8")
    image_hash = _sha256(image)
    annotation_hash = _sha256(annotation)

    manifest = root / "manifests" / "frames.csv"
    manifest.parent.mkdir(parents=True)
    def write_manifest(include_row=True):
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["frame_id", "image_path", "annotation_path"])
            writer.writeheader()
            if include_row:
                writer.writerow({
                    "frame_id": "s1__000004",
                    "image_path": "dev_existing/sessions/s1/rgb/000004.png",
                    "annotation_path": "dev_existing/annotations/s1/000004.json",
                })

    write_manifest()

    result = rebuild(root, ["dev_existing"], force=True)
    assert result == {
        "annotations": 1,
        "generated": 1,
        "skipped": 0,
        "failed": 0,
        "overlays": 1,
        "orphan_overlays": [],
        "errors": [],
    }
    overlay = Path(annotation_overlay_path(annotation))
    assert overlay.is_file()

    # Existing cache cannot hide malformed JSON.
    annotation.write_text("{", encoding="utf-8")
    malformed = rebuild(root, ["dev_existing"], force=False)
    assert malformed["failed"] == 1
    assert "Expecting property name" in malformed["errors"][0]

    # A valid but newer annotation refreshes without requiring --force.
    annotation.write_text(valid_annotation, encoding="utf-8")
    newer = max(annotation.stat().st_mtime_ns, overlay.stat().st_mtime_ns + 1)
    os.utime(annotation, ns=(newer, newer))
    stale = rebuild(root, ["dev_existing"], force=False)
    assert stale["generated"] == 1 and stale["skipped"] == 0

    # Membership is checked before the now-existing cache is skipped.
    write_manifest(include_row=False)
    missing = rebuild(root, ["dev_existing"], force=False)
    assert missing["failed"] == 1
    assert missing["errors"] == [f"manifest row missing: {annotation}"]

    # PNGs without an active sibling JSON are explicit contract failures.
    write_manifest()
    orphan = overlay.parent / "orphan.png"
    orphan.write_bytes(overlay.read_bytes())
    with_orphan = rebuild(root, ["dev_existing"], force=False)
    assert with_orphan["failed"] == 0
    assert with_orphan["orphan_overlays"] == [str(orphan)]
    assert _sha256(image) == image_hash
    assert _sha256(annotation) == annotation_hash
