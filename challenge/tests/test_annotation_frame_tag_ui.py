"""Regression tests for annotation-owned evaluation frame metadata."""

from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
ANNOTATE_DIR = REPO / "scripts/annotate"
sys.path.insert(0, str(ANNOTATE_DIR))

import annotate  # noqa: E402
import annotate_draw  # noqa: E402
from annotate_io import State  # noqa: E402
from object_geometry_registry import load_object_geometry_registry  # noqa: E402
from scripts.evaluation.eval_workspace import (  # noqa: E402
    DEFAULT_TARGETS,
    reconcile_annotation_state,
    render_priority_report,
)


def _session(path: Path, **updates: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "session_id": path.name,
        "population_role": "FINAL",
        "object_type": "plastic",
        "lighting": "night",
        "default_tags": {"occlusion": "none", "truncation": "none"},
    }
    metadata.update(updates)
    path.mkdir(parents=True, exist_ok=True)
    (path / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _args(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "object_type": None,
        "population_role": None,
        "lighting_condition": None,
        "intrinsics_quality": None,
        "intrinsics_source": None,
        "capture_session_id": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("total", [4546, 7913, 9362, 17917])
def test_frame_trackbar_does_not_rollback_sequential_navigation(total: int) -> None:
    # Programmatic slider alignment is lossy for these large sessions. Every
    # valid frame must nevertheless remain stable until the raw tick changes.
    for current in range(total):
        tick = annotate._frame_cur_to_tick(current, total)
        assert annotate._frame_trackbar_target(current, total, tick) is None

    # The original failure occurred immediately after save-next: frame 0 -> 1
    # still maps to tick 0, whose inverse is frame 0.
    assert annotate._frame_cur_to_tick(1, total) == 0
    assert annotate._frame_tick_to_cur(0, total) == 0


def test_frame_trackbar_follows_an_actual_tick_change() -> None:
    total = 4546
    current = 1
    requested_tick = 2
    assert requested_tick != annotate._frame_cur_to_tick(current, total)
    assert annotate._frame_trackbar_target(
        current, total, requested_tick
    ) == annotate._frame_tick_to_cur(requested_tick, total)


def _tag_state(session: Path, annotation: Path, frame: str = "000123.png") -> State:
    state = State()
    state.population_role = "DEV"
    state.eval_session_dir = str(session)
    state.session_metadata = json.loads(
        (session / "session.json").read_text(encoding="utf-8"))
    state.kps_2d = [[float(i), float(i)] for i in range(9)]
    state.extrap_mask = [False] * 9
    state.keypoint_annotations = [
        {
            "xy": [float(i), float(i)],
            "visibility": 2,
            "in_frame": True,
            "source": "manual_click",
            "reason": "visible",
        }
        for i in range(9)
    ]
    state.img_shape = (480, 640, 3)
    state.legacy_document = None
    state.occlusion_level = "unknown"
    state.loaded_annotation_path = str(annotation.resolve()) if annotation.exists() else None
    annotate._load_frame_tag_state(state, frame, annotation)
    return state


def _eval_pool_session(
    root: Path,
    name: str,
    *,
    object_type: str = "plastic",
    lighting: str = "day",
    role: str = "DEV",
    camera_matrix: np.ndarray | None = None,
    intrinsics_quality: str | None = None,
    intrinsics_source: str | None = None,
) -> Path:
    session = root / "dev_existing" / "sessions" / name
    metadata_updates: dict[str, object] = {}
    if intrinsics_quality is not None:
        metadata_updates["intrinsics_quality"] = intrinsics_quality
    if intrinsics_source is not None:
        metadata_updates["intrinsics_source"] = intrinsics_source
    _session(
        session,
        population_role=role,
        object_type=object_type,
        lighting=lighting,
        **metadata_updates,
    )
    rgb = session / "rgb"
    rgb.mkdir()
    (rgb / "000001.png").write_bytes(b"frame")
    if camera_matrix is not None:
        np.savetxt(session / "cam_K.txt", camera_matrix)
    return session


def _incoming_pool_session(
    root: Path,
    name: str,
    *,
    lighting: str,
    camera_matrix: np.ndarray,
    frame_count: int,
    first_frame_number: int,
) -> Path:
    session = root / "incoming" / "sessions" / name
    manifest_relative = "manifests/frame_review.csv"
    _session(
        session,
        workspace_scope="INCOMING_UNREVIEWED",
        population_role=None,
        active_evaluation_member=False,
        object_type="unknown",
        lighting=lighting,
        image_count=frame_count,
        frame_review_manifest=manifest_relative,
        camera={
            "intrinsics_quality": "PROVIDED_UNVERIFIED",
            "intrinsics_source": "source archive camera_info.json",
        },
    )
    rgb = session / "rgb"
    rgb.mkdir()
    review_rows = []
    for source_ordinal, frame_number in enumerate(range(
            first_frame_number, first_frame_number + frame_count), start=1):
        frame_name = f"{frame_number:06d}.png"
        (rgb / frame_name).write_bytes(b"raw frame")
        if source_ordinal == 1:
            review_label, exclude_reason = "exclude", "no_pallet"
        elif source_ordinal == frame_count:
            review_label, exclude_reason = "exclude", "motion_blur"
        elif source_ordinal % 2 == 0:
            review_label, exclude_reason = "plastic", ""
        else:
            review_label, exclude_reason = "wood", ""
        review_rows.append({
            "frame": frame_name,
            "source_ordinal": source_ordinal,
            "review_label": review_label,
            "exclude_reason": exclude_reason,
        })
    manifest = session / manifest_relative
    manifest.parent.mkdir()
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame", "source_ordinal", "review_label", "exclude_reason"),
        )
        writer.writeheader()
        writer.writerows(review_rows)
    np.savetxt(session / "cam_K.txt", camera_matrix)
    return session


def _review_rows(session: Path) -> list[dict[str, str]]:
    with (session / "manifests/frame_review.csv").open(
            encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_review_rows(session: Path, rows: list[dict[str, object]]) -> None:
    with (session / "manifests/frame_review.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame", "source_ordinal", "review_label", "exclude_reason"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_incoming_review_manifest_preserves_raw_order_and_hides_excludes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = _incoming_pool_session(
        root,
        "mixed",
        lighting="day",
        camera_matrix=np.eye(3),
        frame_count=6,
        first_frame_number=100,
    )
    rows = [
        {
            "frame": "000105.png", "source_ordinal": 6,
            "review_label": "plastic", "exclude_reason": "",
        },
        {
            "frame": "000104.png", "source_ordinal": 5,
            "review_label": "wood", "exclude_reason": "",
        },
        {
            "frame": "000103.png", "source_ordinal": 4,
            "review_label": "exclude", "exclude_reason": "motion_blur",
        },
        {
            "frame": "000102.png", "source_ordinal": 3,
            "review_label": "plastic", "exclude_reason": "",
        },
        {
            "frame": "000101.png", "source_ordinal": 2,
            "review_label": "wood", "exclude_reason": "",
        },
        {
            "frame": "000100.png", "source_ordinal": 1,
            "review_label": "exclude", "exclude_reason": "no_pallet",
        },
    ]
    _write_review_rows(session, rows)

    frame_paths = annotate._session_image_paths(str(session))
    registry = load_object_geometry_registry()
    partitions, review_info = annotate._incoming_reviewed_frame_partitions(
        json.loads((session / "session.json").read_text(encoding="utf-8")),
        str(session),
        frame_paths,
        registry,
    )

    assert [Path(path).name for path in partitions[
        registry.resolve("plastic").object_type]] == ["000102.png", "000105.png"]
    assert [Path(path).name for path in partitions[
        registry.resolve("wood").object_type]] == ["000101.png", "000104.png"]
    assert review_info["label_counts"] == {
        "exclude": 2, "plastic": 2, "wood": 2}
    assert review_info["exclude_reason_counts"] == {
        "motion_blur": 1, "no_pallet": 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "does not cover every raw RGB filename"),
        ("duplicate", "duplicate frame in review manifest"),
        ("unknown_label", "invalid review_label"),
        ("wrong_ordinal", "source_ordinal does not match sorted rgb"),
        ("missing_exclude_reason", "exclude row requires exclude_reason"),
        ("accepted_reason", "accepted review row must have an empty exclude_reason"),
        ("unknown_frame", "frame is not in raw rgb"),
    ],
)
def test_incoming_review_manifest_rejects_invalid_or_incomplete_rows(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    session = _incoming_pool_session(
        tmp_path / "pallet_eval_v1",
        "mixed",
        lighting="night",
        camera_matrix=np.eye(3),
        frame_count=6,
        first_frame_number=100,
    )
    rows = _review_rows(session)
    if mutation == "missing":
        rows.pop(2)
    elif mutation == "duplicate":
        rows.append(dict(rows[1]))
    elif mutation == "unknown_label":
        rows[1]["review_label"] = "metal"
    elif mutation == "wrong_ordinal":
        rows[1]["source_ordinal"] = "5"
    elif mutation == "missing_exclude_reason":
        rows[0]["exclude_reason"] = ""
    elif mutation == "accepted_reason":
        rows[1]["exclude_reason"] = "no_pallet"
    elif mutation == "unknown_frame":
        rows[1]["frame"] = "999999.png"
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    _write_review_rows(session, rows)

    with pytest.raises(ValueError, match=message):
        annotate._incoming_reviewed_frame_partitions(
            json.loads((session / "session.json").read_text(encoding="utf-8")),
            str(session),
            annotate._session_image_paths(str(session)),
            load_object_geometry_registry(),
        )


def test_incoming_ordinal_partition_metadata_no_longer_opens_views(
    tmp_path: Path,
) -> None:
    session = _incoming_pool_session(
        tmp_path / "pallet_eval_v1",
        "mixed",
        lighting="day",
        camera_matrix=np.eye(3),
        frame_count=6,
        first_frame_number=100,
    )
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    metadata.pop("frame_review_manifest")
    metadata["object_frame_partition"] = {
        "schema_version": "pallet_eval_object_frame_partition_v1",
        "index_base": 1,
        "end_inclusive": True,
        "ranges": [
            {"object_type": "wood", "start_ordinal": 1, "end_ordinal": 3},
            {"object_type": "plastic", "start_ordinal": 4, "end_ordinal": 6},
        ],
    }

    with pytest.raises(ValueError, match="frame_review_manifest is required"):
        annotate._incoming_reviewed_frame_partitions(
            metadata,
            str(session),
            annotate._session_image_paths(str(session)),
            load_object_geometry_registry(),
        )


def test_eval_session_pool_uses_per_session_output_metadata_and_intrinsics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    day_k = np.array(
        [[614.0, 0.0, 329.0], [0.0, 615.0, 235.0], [0.0, 0.0, 1.0]])
    night_k = np.array(
        [[606.0, 0.0, 318.0], [0.0, 607.0, 256.0], [0.0, 0.0, 1.0]])
    day = _eval_pool_session(root, "plastic_day", camera_matrix=day_k)
    night = _eval_pool_session(
        root, "plastic_night", lighting="night", camera_matrix=night_k)
    wood = _eval_pool_session(
        root,
        "wood_day",
        object_type="wood",
        camera_matrix=day_k,
        intrinsics_quality="SENSOR_PROFILE_SCALED",
        intrinsics_source="legacy annotation consensus",
    )
    _eval_pool_session(root, "dev_negative", object_type="none", camera_matrix=day_k)
    incoming_day = _incoming_pool_session(
        root,
        "real_unlabeled_day_20260830",
        lighting="day",
        camera_matrix=day_k,
        frame_count=420,
        first_frame_number=0,
    )
    incoming_night = _incoming_pool_session(
        root,
        "real_unlabeled_night_20260830",
        lighting="night",
        camera_matrix=night_k,
        frame_count=181,
        first_frame_number=29028,
    )

    day_out = root / "dev_existing" / "annotations" / day.name
    night_out = root / "dev_existing" / "annotations" / night.name
    day_out.mkdir(parents=True)
    night_out.mkdir(parents=True)
    (day_out / "000001.json").write_text("{}", encoding="utf-8")

    registry = load_object_geometry_registry()
    plastic_type = registry.resolve("plastic").object_type
    sessions, contexts = annotate._discover_evaluation_session_pool(
        str(root),
        str(day),
        str(day_out),
        _args(),
        registry,
        str(REPO),
        "DEV",
        plastic_type,
    )

    # Three existing DEV sessions plus two object-specific staging contexts
    # for each mixed DAY/NIGHT capture.  The negative session is never a
    # positive annotation choice.
    assert len(sessions) == 7
    assert len(contexts) == 7
    assert "dev_negative" not in {
        context["metadata"]["session_id"] for context in contexts.values()
    }

    def context_for(session_id: str) -> dict[str, object]:
        matches = [
            context for context in contexts.values()
            if context["metadata"]["session_id"] == session_id
        ]
        assert len(matches) == 1
        return matches[0]

    day_context = context_for("plastic_day")
    night_context = context_for("plastic_night")
    wood_context = context_for("wood_day")
    assert Path(day_context["out_dir"]) == day_out
    assert Path(night_context["out_dir"]) == night_out
    assert day_context["args"].lighting_condition == "day"
    assert night_context["args"].lighting_condition == "night"
    assert day_context["args"].capture_session_id == "plastic_day"
    assert night_context["args"].capture_session_id == "plastic_night"
    assert np.array_equal(day_context["K"], day_k)
    assert np.array_equal(night_context["K"], night_k)
    assert wood_context["geometry_spec"].object_type == registry.resolve("wood").object_type
    assert wood_context["args"].intrinsics_quality == "SENSOR_PROFILE_SCALED"
    assert wood_context["args"].intrinsics_source == "legacy annotation consensus"
    assert wood_context["writable"] is True
    assert Path(wood_context["out_dir"]) == (
        root / "dev_existing" / "annotations" / wood.name
    )
    assert np.array_equal(wood_context["K"], day_k)

    plastic_object = registry.resolve("plastic").object_type
    wood_object = registry.resolve("wood").object_type
    incoming_annotations_root = root / "incoming" / "annotations"
    all_staging_outputs: set[Path] = set()
    staging_example_stems: dict[Path, str] = {}
    expected_partition_counts: dict[tuple[str, str], int] = {}
    for incoming, expected_lighting, expected_k in (
        (incoming_day, "day", day_k),
        (incoming_night, "night", night_k),
    ):
        staging_contexts = [
            context for context in contexts.values()
            if context["metadata"].get("source_session_id") == incoming.name
        ]
        assert len(staging_contexts) == 2
        assert {
            context["geometry_spec"].object_type
            for context in staging_contexts
        } == {plastic_object, wood_object}

        contexts_by_object = {
            context["geometry_spec"].object_type: context
            for context in staging_contexts
        }
        expected_outputs = {
            plastic_object: incoming_annotations_root / f"{incoming.name}__plastic",
            wood_object: incoming_annotations_root / f"{incoming.name}__wood",
        }
        expected_view_ids = {
            plastic_object: f"{incoming.name}__plastic",
            wood_object: f"{incoming.name}__wood",
        }
        raw_frame_paths = sorted(
            path.resolve() for path in (incoming / "rgb").iterdir())
        review_manifest = incoming / "manifests" / "frame_review.csv"
        with review_manifest.open(encoding="utf-8", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        raw_by_name = {path.name: path for path in raw_frame_paths}
        object_for_label = {
            "plastic": plastic_object,
            "wood": wood_object,
        }
        expected_frame_paths = {
            plastic_object: [],
            wood_object: [],
        }
        excluded_paths = []
        expected_source_ordinals = {
            plastic_object: [],
            wood_object: [],
        }
        for row in review_rows:
            frame_path = raw_by_name[row["frame"]]
            if row["review_label"] == "exclude":
                excluded_paths.append(frame_path)
                continue
            object_type = object_for_label[row["review_label"]]
            expected_frame_paths[object_type].append(frame_path)
            expected_source_ordinals[object_type].append(
                int(row["source_ordinal"]))

        plastic_paths = set(expected_frame_paths[plastic_object])
        wood_paths = set(expected_frame_paths[wood_object])
        excluded_paths_set = set(excluded_paths)
        assert plastic_paths.isdisjoint(wood_paths)
        assert plastic_paths.isdisjoint(excluded_paths_set)
        assert wood_paths.isdisjoint(excluded_paths_set)
        assert plastic_paths | wood_paths | excluded_paths_set == set(raw_frame_paths)
        assert len(plastic_paths) + len(wood_paths) + len(excluded_paths_set) == (
            len(raw_frame_paths))
        for object_type, context in contexts_by_object.items():
            view_id = expected_view_ids[object_type]
            assert context["writable"] is True
            assert context["active_evaluation_member"] is False
            assert context["refresh_evaluation"] is True
            assert context["workspace_scope"] == "INCOMING_ANNOTATION"
            assert context["display_role"] == "STAGING"
            assert context["metadata"]["session_id"] == view_id
            assert context["metadata"]["source_session_id"] == incoming.name
            assert context["metadata"]["workspace_scope"] == "INCOMING_ANNOTATION"
            assert context["metadata"]["population_role"] == "FINAL"
            assert context["metadata"]["active_evaluation_member"] is False
            assert context["metadata"]["object_type"] == object_type
            assert context["metadata"]["intrinsics_quality"] == "UNKNOWN"
            assert "PROVIDED_UNVERIFIED" in context["metadata"]["intrinsics_source"]
            assert "source archive camera_info.json" in (
                context["metadata"]["intrinsics_source"])
            assert context["args"].population_role == "FINAL"
            assert context["args"].object_type == object_type
            assert context["args"].lighting_condition == expected_lighting
            assert context["args"].capture_session_id == view_id
            assert context["args"].intrinsics_quality == "UNKNOWN"
            assert context["args"].intrinsics_source == (
                context["metadata"]["intrinsics_source"])
            assert context["intrinsics_quality"] == "UNKNOWN"
            assert context["intrinsics_source"] == (
                context["metadata"]["intrinsics_source"])
            assert context["K_source"] == "cam_K.txt"
            assert np.array_equal(context["K"], expected_k)
            assert Path(context["out_dir"]) == expected_outputs[object_type]
            assert Path(context["tag_session_dir"]) == expected_outputs[object_type]
            assert Path(context["source_session_dir"]).resolve() == incoming.resolve()
            context_frame_paths = [
                Path(path).resolve() for path in context["frame_paths"]]
            assert context_frame_paths == expected_frame_paths[object_type]
            assert context["source_frame_count"] == len(raw_frame_paths)
            assert context["frame_count"] == len(expected_frame_paths[object_type])
            assert context["source_ordinals"] == expected_source_ordinals[object_type]
            assert Path(context["frame_review_manifest_path"]) == review_manifest
            assert context["frame_review_label"] in {"plastic", "wood"}
            assert context["frame_review_label_counts"] == {
                "exclude": 2,
                "plastic": len(expected_frame_paths[plastic_object]),
                "wood": len(expected_frame_paths[wood_object]),
            }
            assert context["frame_review_exclude_reason_counts"] == {
                "motion_blur": 1,
                "no_pallet": 1,
            }
            # Direct equality with the corresponding raw-list subsequence also
            # proves that raw source order is preserved even if CSV rows move.
            assert context_frame_paths == [
                path for path in raw_frame_paths
                if path in set(expected_frame_paths[object_type])
            ]
            assert set(context_frame_paths).isdisjoint(excluded_paths_set)
            assert not Path(context["out_dir"]).exists()
            output_path = Path(context["out_dir"])
            all_staging_outputs.add(output_path)
            staging_example_stems[output_path] = context_frame_paths[0].stem
            expected_partition_counts[(expected_lighting.upper(), object_type)] = (
                len(context_frame_paths))

        # The variants are disjoint zero-copy partitions of one raw list, and
        # all mutable annotation and frame-tag state has disjoint destinations.
        assert set(contexts_by_object[plastic_object]["frame_paths"]).isdisjoint(
            contexts_by_object[wood_object]["frame_paths"])
        assert (
            contexts_by_object[plastic_object]["tag_session_dir"]
            != contexts_by_object[wood_object]["tag_session_dir"]
        )

        # Partitioning never copies, moves or edits the raw RGB sequence.
        assert len(list((incoming / "rgb").iterdir())) == len(raw_frame_paths)
        raw_metadata = json.loads(
            (incoming / "session.json").read_text(encoding="utf-8"))
        assert raw_metadata["session_id"] == incoming.name
        assert raw_metadata["workspace_scope"] == "INCOMING_UNREVIEWED"
        assert raw_metadata["population_role"] is None
        assert raw_metadata["active_evaluation_member"] is False
        assert raw_metadata["object_type"] == "unknown"
        assert raw_metadata["image_count"] == len(raw_frame_paths)
        assert raw_metadata["frame_review_manifest"] == (
            "manifests/frame_review.csv")
        assert "object_frame_partition" not in raw_metadata
        assert raw_metadata["camera"]["intrinsics_quality"] == (
            "PROVIDED_UNVERIFIED")

    # Each staging view writes an assigned frame to its own output namespace;
    # no staging annotation is silently part of evaluation.
    assert len(all_staging_outputs) == 4
    for output in all_staging_outputs:
        output.mkdir(parents=True)
        (output / f"{staging_example_stems[output]}.json").write_text(
            "{}", encoding="utf-8")
    assert all(
        (output / f"{staging_example_stems[output]}.json").is_file()
        for output in all_staging_outputs)

    output_dirs = {
        key: context["out_dir"] for key, context in contexts.items()
    }
    # Context-aware chooser data must expose four separate editable rows.
    summary = annotate.session_summary(
        sessions, str(REPO), "DEV", output_dirs, contexts)
    assert len(summary) == 7
    staging_rows = [row for row in summary if row["role"] == "STAGING"]
    assert len(staging_rows) == 4
    assert {row["object"] for row in staging_rows} == {
        plastic_object, wood_object}
    assert [row["lighting"] for row in staging_rows].count("DAY") == 2
    assert [row["lighting"] for row in staging_rows].count("NIGHT") == 2
    assert {
        (row["lighting"], row["object"]): row["frames"]
        for row in staging_rows
    } == expected_partition_counts
    assert all(row["done"] == 1 for row in staging_rows)
    assert all(row["writable"] is True for row in staging_rows)
    assert all(row["status"] == "STAGING EDIT" for row in staging_rows)

    # Existing callers that do not pass contexts keep the historical tuple
    # contract, including a numeric done count and FINAL-role boolean.  This
    # compatibility check deliberately uses ordinary unique session paths;
    # object-specific zero-copy rows require context-aware lookup.
    legacy_sessions = [
        ("plastic_day", str(day)),
        ("plastic_night", str(night)),
        ("wood_day", str(wood)),
    ]
    legacy_output_dirs = {
        str(day.resolve()): str(day_out),
        str(night.resolve()): str(night_out),
        str(wood.resolve()): str(
            root / "dev_existing" / "annotations" / wood.name),
    }
    assert annotate.session_summary(
        legacy_sessions, str(REPO), "DEV", legacy_output_dirs
    ) == [
        ("plastic_day", 1, 1, False),
        ("plastic_night", 1, 0, False),
        ("wood_day", 1, 0, False),
    ]


def test_eval_session_pool_context_role_follows_each_candidate_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    camera_matrix = np.array(
        [[614.0, 0.0, 329.0], [0.0, 615.0, 235.0], [0.0, 0.0, 1.0]])
    dev = _eval_pool_session(
        root, "dev_plastic", camera_matrix=camera_matrix)
    final = root / "final" / "positive" / "sessions" / "final_plastic"
    _session(
        final,
        population_role="FINAL",
        object_type="plastic",
        lighting="night",
    )
    (final / "rgb").mkdir()
    (final / "rgb" / "000001.png").write_bytes(b"frame")
    np.savetxt(final / "cam_K.txt", camera_matrix)

    registry = load_object_geometry_registry()
    plastic_type = registry.resolve("plastic").object_type
    sessions, contexts = annotate._discover_evaluation_session_pool(
        str(root),
        str(dev),
        str(root / "dev_existing" / "annotations" / dev.name),
        _args(),
        registry,
        str(REPO),
        "DEV",
        plastic_type,
    )

    assert len(sessions) == 2
    by_session = {
        context["metadata"]["session_id"]: context
        for context in contexts.values()
    }
    assert by_session["dev_plastic"]["workspace_scope"] == "DEV"
    assert by_session["dev_plastic"]["display_role"] == "DEV"
    assert by_session["dev_plastic"]["args"].population_role == "DEV"
    assert by_session["final_plastic"]["workspace_scope"] == "FINAL"
    assert by_session["final_plastic"]["display_role"] == "FINAL"
    assert by_session["final_plastic"]["args"].population_role == "FINAL"


def test_review_only_click_keys_allow_browsing_but_block_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "incoming_session"
    _session(session)
    state = _tag_state(session, tmp_path / "uncreated.json")
    state.session_writable = False
    state.mode = "click"
    state.zoom = 1.0
    state.pan = [0, 0]
    state.condition_mode = False
    state.active = 0
    state.split = "eval"
    state.pose = {"reproj_error_px": 0.25}
    state.dirty = False
    state.annotation_dirty = False
    state.frame_tags_dirty = False
    unused = (
        str(tmp_path / "incoming/annotations/000001.json"),
        str(tmp_path / "incoming/annotations/000001.png"),
        str(tmp_path / "incoming/rgb/000001.png"),
        np.eye(3),
    )

    assert annotate._handle_click_key(ord("n"), state, *unused) == "next"
    assert annotate._handle_click_key(ord("p"), state, *unused) == "prev"
    assert annotate._handle_click_key(ord(","), state, *unused) == "jump-10"
    assert annotate._handle_click_key(ord("."), state, *unused) == "jump+10"

    annotate._handle_click_key(ord("+"), state, *unused)
    assert state.zoom == pytest.approx(1.5)
    annotate._handle_click_key(ord("h"), state, *unused)
    annotate._handle_click_key(ord("k"), state, *unused)
    assert state.pan == [-20, -20]
    annotate._handle_click_key(ord("l"), state, *unused)
    annotate._handle_click_key(ord("j"), state, *unused)
    assert state.pan == [0, 0]
    annotate._handle_click_key(ord("-"), state, *unused)
    assert state.zoom == pytest.approx(1.0)

    snapshot = {
        "active": state.active,
        "kps_2d": json.loads(json.dumps(state.kps_2d)),
        "keypoint_annotations": json.loads(json.dumps(state.keypoint_annotations)),
        "condition_mode": state.condition_mode,
        "split": state.split,
        "pose": dict(state.pose),
        "axis_assignment": state.axis_assignment,
        "frame_tags": dict(state.frame_tags),
        "frame_tag_pending_updates": dict(state.frame_tag_pending_updates),
        "dirty": state.dirty,
        "annotation_dirty": state.annotation_dirty,
        "frame_tags_dirty": state.frame_tags_dirty,
    }
    monkeypatch.setattr(
        annotate,
        "_save_state_annotation",
        lambda *_args, **_kwargs: pytest.fail("review-only key dispatched a save"),
    )
    monkeypatch.setattr(
        annotate,
        "_delete_annotation",
        lambda *_args, **_kwargs: pytest.fail("review-only key dispatched a delete"),
    )

    # These exercise keypoint selection, CONDITIONS entry, metadata edits,
    # visibility/axis edits, reset and every save shortcut family.
    for key in "7/vbwysfgxcr":
        assert annotate._handle_click_key(ord(key), state, *unused) is None

    assert state.active == snapshot["active"]
    assert state.kps_2d == snapshot["kps_2d"]
    assert state.keypoint_annotations == snapshot["keypoint_annotations"]
    assert state.condition_mode == snapshot["condition_mode"]
    assert state.split == snapshot["split"]
    assert state.pose == snapshot["pose"]
    assert state.axis_assignment == snapshot["axis_assignment"]
    assert state.frame_tags == snapshot["frame_tags"]
    assert state.frame_tag_pending_updates == snapshot["frame_tag_pending_updates"]
    assert state.dirty == snapshot["dirty"]
    assert state.annotation_dirty == snapshot["annotation_dirty"]
    assert state.frame_tags_dirty == snapshot["frame_tags_dirty"]


def test_review_only_save_never_creates_annotation_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "incoming_session"
    _session(session)
    state = _tag_state(session, tmp_path / "not-loaded.json")
    state.session_writable = False
    state.pose = {"reproj_error_px": 0.0}
    output = tmp_path / "incoming/annotations/incoming_session/000001.json"
    output_png = output.with_suffix(".png")
    source = tmp_path / "incoming/sessions/incoming_session/rgb/000001.png"
    monkeypatch.setattr(
        annotate,
        "_make_state_annotation",
        lambda *_args, **_kwargs: pytest.fail("review-only save built a GT document"),
    )

    assert not annotate._save_state_annotation(
        state, np.eye(3), str(output), str(output_png), str(source)
    )
    assert not output.exists()
    assert not output.parent.exists()


def test_review_only_update_pose_never_calls_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "incoming_session"
    _session(session)
    state = _tag_state(session, tmp_path / "not-loaded.json")
    state.session_writable = False
    state.pose = {"sentinel": True}
    state.locked_pose = {"sentinel": True}
    state._pose_key = ("stale",)
    monkeypatch.setattr(
        annotate,
        "solve_pose",
        lambda *_args, **_kwargs: pytest.fail("review-only frame invoked PnP"),
    )

    annotate.update_pose(state, np.eye(3), force=True)

    assert state.pose is None
    assert state.locked_pose is None
    assert state._pose_key is None


def test_review_only_image_clicks_do_not_change_keypoints(tmp_path: Path) -> None:
    session = tmp_path / "incoming_session"
    _session(session)
    state = _tag_state(session, tmp_path / "not-loaded.json")
    state.session_writable = False
    state.img = np.zeros((480, 640, 3), dtype=np.uint8)
    state.mode = "click"
    state.zoom = 1.0
    state.pan = [0, 0]
    state.active = 3
    state.dirty = False
    state.annotation_dirty = False
    before_kps = json.loads(json.dumps(state.kps_2d))
    before_annotations = json.loads(json.dumps(state.keypoint_annotations))

    for event in (
        annotate.cv2.EVENT_LBUTTONDOWN,
        annotate.cv2.EVENT_RBUTTONDOWN,
    ):
        annotate.on_mouse(
            event,
            annotate.MARGIN_L + 123,
            annotate.MARGIN_T + 234,
            0,
            state,
        )

    assert state.kps_2d == before_kps
    assert state.keypoint_annotations == before_annotations
    assert state.active == 3
    assert state.dirty is False
    assert state.annotation_dirty is False


def test_eval_intrinsics_use_annotation_consensus_and_reject_conflict(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    annotations = tmp_path / "annotations"
    session.mkdir()
    annotations.mkdir()
    intrinsics = {"fx": 605.0, "fy": 606.0, "cx": 317.0, "cy": 256.0}
    document = {"camera_data": {"intrinsics": intrinsics}}
    (annotations / "a.json").write_text(json.dumps(document), encoding="utf-8")
    (annotations / "b.json").write_text(json.dumps(document), encoding="utf-8")

    camera_matrix, source = annotate._resolve_session_intrinsics(
        str(session), str(annotations), evaluation=True)
    assert source == "canonical annotation consensus"
    assert np.array_equal(
        camera_matrix,
        np.array([[605.0, 0.0, 317.0], [0.0, 606.0, 256.0], [0.0, 0.0, 1.0]]),
    )

    conflict = {"camera_data": {"intrinsics": {**intrinsics, "fx": 700.0}}}
    (annotations / "b.json").write_text(json.dumps(conflict), encoding="utf-8")
    with pytest.raises(ValueError, match="intrinsics disagree"):
        annotate._resolve_session_intrinsics(
            str(session), str(annotations), evaluation=True)


def test_eval_session_auto_resolves_object_lighting_role_and_intrinsics(
    tmp_path: Path,
) -> None:
    session = tmp_path / "wood_night_01"
    metadata = _session(
        session,
        object_type="wood",
        intrinsics_quality="CALIBRATED",
        intrinsics_source="calibration/run-01.json",
    )
    args = _args()
    loaded, spec = annotate._resolve_annotation_configuration(
        args, str(session), load_object_geometry_registry(), str(tmp_path))
    assert loaded == metadata
    assert spec.object_type == "wood_small_80x59x14"
    assert args.population_role == "FINAL"
    assert args.lighting_condition == "night"
    assert args.intrinsics_quality == "CALIBRATED"
    assert args.intrinsics_source == "calibration/run-01.json"
    assert args.capture_session_id == "wood_night_01"


@pytest.mark.parametrize(
    ("session_updates", "cli_updates", "message"),
    [
        ({"object_type": "wood"}, {"object_type": "plastic"}, "object type mismatch"),
        ({"population_role": "FINAL"}, {"population_role": "DEV"}, "population role mismatch"),
        ({"lighting": "night"}, {"lighting_condition": "day"}, "lighting mismatch"),
        (
            {"intrinsics_quality": "SENSOR_PROFILE_SCALED"},
            {"intrinsics_quality": "CALIBRATED"},
            "intrinsics quality mismatch",
        ),
        (
            {"intrinsics_source": "session.json"},
            {"intrinsics_source": "cli.json"},
            "intrinsics source mismatch",
        ),
    ],
)
def test_cli_session_conflicts_fail_closed(
    tmp_path: Path,
    session_updates: dict[str, object],
    cli_updates: dict[str, object],
    message: str,
) -> None:
    session = tmp_path / "session"
    _session(session, **session_updates)
    with pytest.raises(ValueError, match=message):
        annotate._resolve_annotation_configuration(
            _args(**cli_updates),
            str(session),
            load_object_geometry_registry(),
            str(tmp_path),
        )


def test_condition_mode_isolates_number_keys_from_keypoint_selection(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    _session(session)
    annotation = tmp_path / "000123.json"
    state = _tag_state(session, annotation)

    unused = ("unused.json", "unused.png", "unused.png", np.eye(3))

    # Normal CLICK keeps all digit keys as keypoint selectors.  In particular,
    # numeric 0 cannot be confused with a condition shortcut.
    pending_before = dict(state.frame_tag_pending_updates)
    for index in range(9):
        annotate._handle_click_key(ord(str(index)), state, *unused)
        assert state.active == index
        assert state.frame_tag_pending_updates == pending_before

    # The former direct uppercase bindings are intentionally inactive.
    cycles_before = dict(state.frame_tag_cycle_values)
    for key in "OTDBEV":
        annotate._handle_click_key(ord(key), state, *unused)
    assert state.frame_tag_cycle_values == cycles_before

    # C/c is no longer shared by two commands.  The slash key is otherwise
    # unused throughout the editor and is the sole printable mode toggle.
    state.condition_mode = False
    annotate._handle_click_key(ord("c"), state, *unused)
    assert state.condition_mode is False
    annotate._handle_click_key(ord("C"), state, *unused)
    assert state.condition_mode is False
    annotate._handle_click_key(
        ord("/"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    assert state.condition_mode is True

    active_before = state.active
    expected = {
        "1": ("occlusion", "medium"),
        "2": ("truncation", "mild"),
    }
    for key, (field, value) in expected.items():
        annotate._handle_condition_key(ord(key), state, *unused)
        assert state.frame_tags[field] == value
        assert state.frame_tag_sources[field] == "FRAME"
        assert state.active == active_before

    for key, elevation in {"3": "low", "4": "mid", "5": "high"}.items():
        other_tags = {
            field: state.frame_tags[field]
            for field in ("occlusion", "truncation")
        }
        annotate._handle_condition_key(ord(key), state, *unused)
        assert state.frame_tags["elevation_bin"] == elevation
        assert state.frame_tag_sources["elevation_bin"] == "FRAME"
        assert state.active == active_before
        assert {
            field: state.frame_tags[field]
            for field in ("occlusion", "truncation")
        } == other_tags

    for key, distance in {"n": "near", "m": "mid", "6": "far"}.items():
        annotate._handle_condition_key(ord(key), state, *unused)
        assert state.frame_tags["distance_bin"] == distance
        assert state.frame_tag_sources["distance_bin"] == "FRAME"
        assert state.active == active_before

    # Legacy size storage remains readable, but key 7 no longer edits it and
    # cannot fall through to keypoint selection while CONDITIONS is modal.
    before_7 = (
        dict(state.frame_tags),
        dict(state.frame_tag_sources),
        dict(state.frame_tag_pending_updates),
        state.dirty,
    )
    annotate._handle_condition_key(ord("7"), state, *unused)
    assert before_7 == (
        state.frame_tags,
        state.frame_tag_sources,
        state.frame_tag_pending_updates,
        state.dirty,
    )
    assert state.active == active_before

    annotate._handle_condition_key(ord("u"), state, *unused)
    assert state.frame_tags["distance_bin"] == "unknown"
    assert state.frame_tag_sources["distance_bin"] == "UNSET"
    assert state.frame_tag_pending_updates["distance_bin"] == "unknown"
    assert "size_bin" not in state.frame_tag_pending_updates
    assert state.active == active_before

    # Every other digit is a complete no-op and cannot leak into the keypoint
    # selector or edit the advanced categorical metadata.
    tags_before = dict(state.frame_tags)
    pending_before = dict(state.frame_tag_pending_updates)
    for key in "0789":
        annotate._handle_condition_key(ord(key), state, *unused)
    assert state.active == active_before
    assert state.frame_tags == tags_before
    assert state.frame_tag_pending_updates == pending_before

    annotate._handle_condition_key(ord("/"), state, *unused)
    assert state.condition_mode is False


def test_global_mode_keys_are_unmodified_unique_ascii() -> None:
    assert annotate._CONDITION_MODE_KEY == ord("/")
    assert annotate._GOTO_MODE_KEY == ord(";")
    assert annotate._CONDITION_MODE_KEY != annotate._GOTO_MODE_KEY

    # The Qt5 HighGUI backend used by this tool folds Shift+letter to the
    # lowercase ASCII value.  An uppercase ord binding is therefore both
    # unreachable and liable to trigger the lowercase command instead.
    tree = ast.parse(inspect.getsource(annotate))
    shifted_letter_bindings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 1:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "ord":
            continue
        argument = node.args[0]
        if (
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and len(argument.value) == 1
            and argument.value.isupper()
        ):
            shifted_letter_bindings.append(argument.value)
    assert shifted_letter_bindings == []


def test_manipulate_save_uses_reachable_lowercase_s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    state.mode = "manip"
    state.locked_pose = {"sentinel": True}
    state.pose = {
        "projected_all": [[float(i), float(i)] for i in range(9)],
        "reproj_error_px": 0.25,
    }
    saved: list[str] = []
    monkeypatch.setattr(
        annotate,
        "_save_state_annotation",
        lambda _state, _K, out_json, *_args: saved.append(out_json) or True,
    )

    action = annotate._handle_manip_key(
        ord("s"), state, "reachable.json", "unused.png", "source.png", np.eye(3)
    )

    assert action == "save-next"
    assert saved == ["reachable.json"]
    assert state.mode == "click"
    assert state.locked_pose is None


def test_condition_mode_consumes_every_other_editor_shortcut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CLICK/MANIPULATE/navigation key may leak out of CONDITIONS."""

    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    unused = ("unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_click_key(ord("/"), state, *unused)

    active_before = state.active
    points_before = [None if point is None else list(point) for point in state.kps_2d]
    tags_before = dict(state.frame_tags)
    pending_before = dict(state.frame_tag_pending_updates)
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: None)

    # Includes keypoint, centroid, visibility, pose, save alternatives,
    # navigation, goto/session, mode-toggle and quit shortcuts.  CONDITIONS
    # owns none of them; only condition selectors, u, a, s, / and Esc have meaning.
    for key in "089cC'bwytzdrvfgp,.+-=_hjklxG:[]qQOTDBEV":
        annotate._handle_condition_key(ord(key), state, *unused)
    for key in (8, 9, 10, 13, 127):
        annotate._handle_condition_key(key, state, *unused)

    assert state.condition_mode is True
    assert state.active == active_before
    assert state.kps_2d == points_before
    assert state.frame_tags == tags_before
    assert state.frame_tag_pending_updates == pending_before

    annotate._handle_condition_key(27, state, *unused)
    assert state.condition_mode is False


def test_binary_condition_toggles_and_panel_updates_before_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    annotate._set_condition_mode(state, True)

    values = [
        annotate._handle_condition_key(
            ord("2"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
        or state.frame_tag_cycle_values["truncation"]
        for _ in range(4)
    ]
    assert values == ["mild", "none", "mild", "none"]
    assert state.frame_tags["truncation"] == "none"
    assert state.frame_tag_sources["truncation"] == "FRAME"

    # One more press turns the condition ON and is visible immediately, before
    # CSV/JSON save.
    annotate._handle_condition_key(
        ord("2"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    assert state.frame_tags["truncation"] == "mild"
    assert state.frame_tag_sources["truncation"] == "FRAME"
    annotate._handle_condition_key(
        ord("5"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    assert state.frame_tags["elevation_bin"] == "high"
    assert state.frame_tag_sources["elevation_bin"] == "FRAME"
    annotate._handle_condition_key(
        ord("6"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    assert state.frame_tags["distance_bin"] == "far"

    drawn: list[str] = []
    monkeypatch.setattr(
        annotate_draw.cv2,
        "putText",
        lambda image, text, *_args, **_kwargs: drawn.append(str(text)) or image,
    )
    annotate_draw.build_panel(
        1000,
        state.active,
        state.kps_2d,
        state.pose,
        0,
        1,
        1.0,
        state.dirty,
        mode="click",
        frame_tags=state.frame_tags,
        frame_tag_sources=state.frame_tag_sources,
        condition_mode=True,
    )
    assert any("MODE: CONDITIONS" in text for text in drawn)
    assert any("1=occlusion ON/OFF" in text for text in drawn)
    assert any("2=truncation ON/OFF" in text for text in drawn)
    assert any("3=LOW  4=MID  5=HIGH" in text for text in drawn)
    assert any("n=NEAR  m=MID  6=FAR" in text for text in drawn)
    assert any("u=DISTANCE UNKNOWN" in text for text in drawn)
    assert any("a,a=apply edits to annotated session" in text for text in drawn)
    assert any("2 Truncation: ON [FRAME]" in text for text in drawn)
    assert any("3/4/5 Elevation: HIGH [FRAME]" in text for text in drawn)
    assert any("n/m/6 Distance: FAR [FRAME]" in text for text in drawn)
    assert any("TAGS 4/4" in text for text in drawn)
    assert not any(
        forbidden in text
        for text in drawn
        for forbidden in (
            "3=distance", "Size:", "Image size", "SMALL", "View:", "MILD"
        )
    )
    header = next(text for text in drawn if text.startswith("MODE: CONDITIONS"))
    header_width = annotate_draw.cv2.getTextSize(
        header, annotate_draw.cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )[0][0]
    assert header_width <= annotate_draw.PANEL_W - 20


@pytest.mark.parametrize(
    ("key", "field", "expected"),
    [
        ("1", "occlusion", ("medium", "none", "medium", "none")),
        ("2", "truncation", ("mild", "none", "mild", "none")),
    ],
)
def test_each_condition_key_has_one_complete_isolated_binary_toggle(
    tmp_path: Path,
    key: str,
    field: str,
    expected: tuple[str, ...],
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    unused = ("unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_click_key(ord("/"), state, *unused)
    other_fields = {
        name: state.frame_tag_cycle_values[name]
        for name in state.frame_tag_cycle_values
        if name != field
    }

    observed = []
    for _ in expected:
        annotate._handle_condition_key(ord(key), state, *unused)
        observed.append(state.frame_tag_cycle_values[field])

    assert tuple(observed) == expected
    assert {
        name: state.frame_tag_cycle_values[name]
        for name in other_fields
    } == other_fields


def test_far_is_direct_and_key7_cannot_edit_legacy_size(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    _session(session)
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n"
        "000123.png,near,large,unknown,unknown,none,none\n",
        encoding="utf-8",
    )
    state = _tag_state(session, tmp_path / "000123.json")
    unused = ("unused.json", "unused.png", "unused.png", np.eye(3))
    active_before = state.active

    annotate._handle_condition_key(ord("6"), state, *unused)
    assert state.frame_tags["distance_bin"] == "far"
    assert state.frame_tags["size_bin"] == "large"
    assert state.frame_tag_sources["distance_bin"] == "FRAME"
    assert state.frame_tag_sources["size_bin"] == "FRAME"

    before_7 = (
        dict(state.frame_tags),
        dict(state.frame_tag_sources),
        dict(state.frame_tag_pending_updates),
        state.dirty,
    )
    annotate._handle_condition_key(ord("7"), state, *unused)
    assert before_7 == (
        state.frame_tags,
        state.frame_tag_sources,
        state.frame_tag_pending_updates,
        state.dirty,
    )

    # FAR is a direct category selection, not a toggle back to NEAR.
    annotate._handle_condition_key(ord("6"), state, *unused)
    assert state.frame_tags["distance_bin"] == "far"
    assert state.frame_tags["size_bin"] == "large"
    annotate._handle_condition_key(ord("u"), state, *unused)
    assert state.frame_tags["distance_bin"] == "unknown"
    assert state.frame_tags["size_bin"] == "large"
    assert "size_bin" not in state.frame_tag_pending_updates
    assert state.active == active_before


def test_batch_applies_only_pending_tags_to_annotated_session_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "dev_existing/sessions/eval_cad"
    _session(session, population_role="DEV", lighting="day")
    rgb = session / "rgb"
    rgb.mkdir()
    for stem in ("a", "b", "c"):
        (rgb / f"{stem}.png").write_bytes(stem.encode())

    annotation_dir = root / "dev_existing/annotations/eval_cad"
    annotation_dir.mkdir(parents=True)
    document = {"objects": [{"keypoint_annotations": []}]}
    for stem in ("a", "c"):
        (annotation_dir / f"{stem}.json").write_text(
            json.dumps(document), encoding="utf-8")
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n"
        "c.png,unknown,small,unknown,unknown,none,none\n",
        encoding="utf-8",
    )
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in annotation_dir.glob("*.json")
    }

    current_annotation = annotation_dir / "a.json"
    state = _tag_state(session, current_annotation, "a.png")
    state.eval_root = str(root)
    state.annotation_dirty = False
    refreshed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        annotate,
        "_refresh_evaluation_workspace",
        lambda _state, annotation, **kwargs: refreshed.append(
            (str(annotation), str(kwargs.get("image_path"))),
        ),
    )
    unused = (
        str(current_annotation),
        str(current_annotation.with_suffix(".png")),
        str(rgb / "a.png"),
        np.eye(3),
    )
    annotate._handle_condition_key(ord("n"), state, *unused)

    # First press only arms the material session-wide write.
    annotate._handle_condition_key(ord("a"), state, *unused)
    assert state.condition_batch_armed is not None
    rows_before = list(csv.DictReader((session / "frame_tags.csv").open()))
    assert [row["frame"] for row in rows_before] == ["c.png"]

    # Second identical press commits one atomic CSV update.  b.png has no JSON
    # annotation and is deliberately excluded; c.png keeps its legacy size tag.
    annotate._handle_condition_key(ord("a"), state, *unused)
    rows = {
        row["frame"]: row
        for row in csv.DictReader((session / "frame_tags.csv").open())
    }
    assert set(rows) == {"a.png", "c.png"}
    assert rows["a.png"]["distance_bin"] == "near"
    assert rows["c.png"]["distance_bin"] == "near"
    assert rows["c.png"]["size_bin"] == "small"
    assert state.frame_tags["distance_bin"] == "near"
    assert state.frame_tag_sources["distance_bin"] == "FRAME"
    assert state.frame_tag_pending_updates == {}
    assert refreshed == [(str(current_annotation), str(rgb / "a.png"))]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in annotation_dir.glob("*.json")
    } == before


def test_unknown_distance_batch_preserves_legacy_size_on_annotated_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "dev_existing/sessions/eval_cad"
    _session(session, population_role="DEV", lighting="day")
    rgb = session / "rgb"
    rgb.mkdir()
    for stem in ("a", "b", "c"):
        (rgb / f"{stem}.png").write_bytes(stem.encode())

    annotation_dir = root / "dev_existing/annotations/eval_cad"
    annotation_dir.mkdir(parents=True)
    document = {"objects": [{"keypoint_annotations": []}]}
    for stem in ("a", "c"):
        (annotation_dir / f"{stem}.json").write_text(
            json.dumps(document), encoding="utf-8")
    annotation_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in annotation_dir.glob("*.json")
    }
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n"
        "a.png,far,small,high,front,none,none\n"
        "b.png,far,small,low,side,none,none\n"
        "c.png,mid,medium,high,oblique,none,none\n",
        encoding="utf-8",
    )

    current_annotation = annotation_dir / "a.json"
    state = _tag_state(session, current_annotation, "a.png")
    state.eval_root = str(root)
    refreshed: list[str] = []
    monkeypatch.setattr(
        annotate,
        "_refresh_evaluation_workspace",
        lambda *_args, **_kwargs: refreshed.append("refresh"),
    )
    unused = (
        str(current_annotation),
        str(current_annotation.with_suffix(".png")),
        str(rgb / "a.png"),
        np.eye(3),
    )

    annotate._handle_condition_key(ord("u"), state, *unused)
    assert state.frame_tag_pending_updates == {
        "distance_bin": "unknown",
    }
    assert state.frame_tags["distance_bin"] == "unknown"
    assert state.frame_tags["size_bin"] == "small"

    # First a is confirmation only; the second atomically clears distance on
    # annotation-backed a/c.  Legacy size and unannotated b stay untouched.
    before_first_a = (session / "frame_tags.csv").read_bytes()
    annotate._handle_condition_key(ord("a"), state, *unused)
    assert (session / "frame_tags.csv").read_bytes() == before_first_a
    annotate._handle_condition_key(ord("a"), state, *unused)

    rows = {
        row["frame"]: row
        for row in csv.DictReader((session / "frame_tags.csv").open())
    }
    assert rows["a.png"]["distance_bin"] == ""
    assert rows["a.png"]["size_bin"] == "small"
    assert rows["c.png"]["distance_bin"] == ""
    assert rows["c.png"]["size_bin"] == "medium"
    for frame in ("a.png", "c.png"):
        assert rows[frame]["elevation_bin"] == "high"
    assert rows["a.png"]["view_bin"] == "front"
    assert rows["c.png"]["view_bin"] == "oblique"
    assert rows["b.png"]["distance_bin"] == "far"
    assert rows["b.png"]["size_bin"] == "small"
    assert rows["b.png"]["elevation_bin"] == "low"
    assert state.frame_tag_pending_updates == {}
    assert state.frame_tags["distance_bin"] == "unknown"
    assert state.frame_tags["size_bin"] == "small"
    assert refreshed == ["refresh"]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in annotation_dir.glob("*.json")
    } == annotation_hashes


def test_batch_rejects_another_sessions_annotation_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    session_a = root / "dev_existing/sessions/session_a"
    session_b = root / "dev_existing/sessions/session_b"
    _session(session_a, population_role="DEV")
    _session(session_b, population_role="DEV")
    for session in (session_a, session_b):
        rgb = session / "rgb"
        rgb.mkdir()
        (rgb / "same.png").write_bytes(b"frame")

    annotation_b = root / "dev_existing/annotations/session_b/same.json"
    annotation_b.parent.mkdir(parents=True)
    annotation_b.write_text("{}", encoding="utf-8")
    state = _tag_state(session_a, annotation_b, "same.png")
    state.eval_root = str(root)

    with pytest.raises(annotate.WorkspaceError, match="matching canonical namespace"):
        annotate._annotated_session_frames(state, str(annotation_b))


def test_batch_counts_only_annotation_json_files(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "dev_existing/sessions/eval_cad"
    _session(session, population_role="DEV")
    rgb = session / "rgb"
    rgb.mkdir()
    for stem in ("file", "directory"):
        (rgb / f"{stem}.png").write_bytes(b"frame")

    annotation_dir = root / "dev_existing/annotations/eval_cad"
    annotation_dir.mkdir(parents=True)
    annotation = annotation_dir / "file.json"
    annotation.write_text("{}", encoding="utf-8")
    (annotation_dir / "directory.json").mkdir()
    state = _tag_state(session, annotation, "file.png")
    state.eval_root = str(root)

    assert annotate._annotated_session_frames(state, str(annotation)) == [
        "file.png"
    ]


def test_batch_blocks_unsaved_annotation_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "dev_existing/sessions/eval_cad"
    _session(session, population_role="DEV")
    rgb = session / "rgb"
    rgb.mkdir()
    source = rgb / "a.png"
    source.write_bytes(b"frame")
    annotation = root / "dev_existing/annotations/eval_cad/a.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("{}", encoding="utf-8")
    state = _tag_state(session, annotation, "a.png")
    state.eval_root = str(root)
    annotate._set_frame_distance(state, "near")
    state.annotation_dirty = True
    monkeypatch.setattr(
        annotate,
        "update_frame_tags_csv_many",
        lambda *_args, **_kwargs: pytest.fail("batch write must be blocked"),
    )

    for _ in range(2):
        assert not annotate._apply_pending_tags_to_annotated_session(
            state, str(annotation), str(source))
        assert state.condition_batch_armed is None
    assert not (session / "frame_tags.csv").exists()


def test_render_receives_live_condition_mode_and_tag_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    state.img = np.zeros((480, 640, 3), dtype=np.uint8)
    state.condition_mode = True
    annotate._handle_condition_key(
        ord("2"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_condition_key(
        ord("3"), state, "unused.json", "unused.png", "unused.png", np.eye(3))

    received: dict[str, object] = {}

    def fake_panel(height: int, *_args: object, **kwargs: object) -> np.ndarray:
        received.update(kwargs)
        return np.zeros((height, annotate_draw.PANEL_W, 3), dtype=np.uint8)

    monkeypatch.setattr(annotate_draw, "build_panel", fake_panel)
    rendered = annotate_draw.render(state, 0, 1, "000123.png")

    assert received["condition_mode"] is True
    assert received["frame_tags"]["truncation"] == "mild"
    assert received["frame_tag_sources"]["truncation"] == "FRAME"
    assert received["frame_tags"]["elevation_bin"] == "low"
    assert received["frame_tag_sources"]["elevation_bin"] == "FRAME"
    assert rendered.shape[1] == (
        state.img.shape[1]
        + annotate_draw.MARGIN_L
        + annotate_draw.MARGIN_R
        + annotate_draw.PANEL_W
    )


def test_visibility_reason_updates_live_json_evidence_without_frame_override(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    state.active = 0
    annotate._cycle_visibility_reason(state, 0)
    assert state.keypoint_annotations[0]["reason"] == "occluded"
    assert state.frame_tags["occlusion"] == "medium"
    assert state.frame_tag_sources["occlusion"] == "JSON"


def test_live_projected_bbox_uses_saved_annotation_truncation_evidence(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    state.pose = {
        "projected_all": [
            [-5.0, 20.0], [100.0, 20.0], [100.0, 100.0], [10.0, 100.0],
            [20.0, 30.0], [90.0, 30.0], [90.0, 90.0], [20.0, 90.0],
            [50.0, 60.0],
        ]
    }
    annotate._refresh_effective_frame_tags(state, use_state_evidence=True)
    assert state.frame_tags["truncation"] == "mild"
    assert state.frame_tag_sources["truncation"] == "JSON"


def test_metadata_only_save_writes_one_row_roundtrips_and_preserves_gt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _session(session)
    annotation = tmp_path / "annotations/session/000123.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps({"objects": [{"keypoint_annotations": []}], "sentinel": 7}),
        encoding="utf-8",
    )
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n"
        "000123.png,near,large,high,oblique,none,none\n",
        encoding="utf-8",
    )
    before = hashlib.sha256(annotation.read_bytes()).hexdigest()
    state = _tag_state(session, annotation)
    state.pose = {"reproj_error_px": 0.0}
    state.condition_mode = True
    annotate._handle_condition_key(
        ord("1"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_condition_key(
        ord("2"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_condition_key(
        ord("3"), state, "unused.json", "unused.png", "unused.png", np.eye(3))
    annotate._handle_condition_key(
        ord("6"), state, "unused.json", "unused.png", "unused.png", np.eye(3))

    overlays: list[tuple[str, str]] = []
    monkeypatch.setattr(
        annotate,
        "render_saved_annotation_overlay",
        lambda source, target: overlays.append((source, target)) or "overlay.png",
    )
    monkeypatch.setattr(annotate, "_refresh_evaluation_workspace", lambda *_a, **_k: None)
    assert annotate._handle_condition_key(
        ord("s"),
        state,
        str(annotation),
        str(annotation.with_suffix(".png")),
        "source.png",
        np.eye(3),
    ) == "save-next"
    assert hashlib.sha256(annotation.read_bytes()).hexdigest() == before
    rows = list(csv.DictReader((session / "frame_tags.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0] == {
        "frame": "000123.png",
        "distance_bin": "far",
        "size_bin": "large",
        "elevation_bin": "low",
        "view_bin": "oblique",
        "occlusion": "medium",
        "truncation": "mild",
    }
    assert overlays == [("source.png", str(annotation))]

    reloaded = _tag_state(session, annotation)
    for field in (
        "distance_bin",
        "size_bin",
        "elevation_bin",
        "view_bin",
        "occlusion",
        "truncation",
    ):
        assert reloaded.frame_tag_sources[field] == "FRAME"
        assert reloaded.frame_tags[field] == rows[0][field]


def test_condition_mode_blank_save_cannot_delete_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "session"
    _session(session)
    annotation = tmp_path / "000123.json"
    annotation.write_text('{"sentinel": true}\n', encoding="utf-8")
    state = _tag_state(session, annotation)
    state.kps_2d = [None] * 9
    state.condition_mode = True
    annotate._handle_condition_key(
        ord("1"), state, str(annotation), "unused.png", "source.png", np.eye(3))
    monkeypatch.setattr(
        annotate,
        "_delete_annotation",
        lambda *_args, **_kwargs: pytest.fail("CONDITIONS must never delete GT"),
    )

    assert annotate._handle_condition_key(
        ord("s"), state, str(annotation), "unused.png", "source.png", np.eye(3)
    ) is None
    assert annotation.is_file()
    assert state.frame_tags_dirty is True


def test_condition_mode_blocks_image_clicks_from_moving_keypoints(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _session(session)
    state = _tag_state(session, tmp_path / "000123.json")
    state.img = np.zeros((480, 640, 3), dtype=np.uint8)
    state.active = 0
    before = list(state.kps_2d[0])
    state.condition_mode = True

    active_before = state.active
    dirty_before = state.dirty
    annot_only_before = state.annot_only
    sess_open_before = getattr(state, "sess_open", False)
    canvas_w = state.img.shape[1] + annotate.MARGIN_L + annotate.MARGIN_R
    canvas_h = state.img.shape[0] + annotate.MARGIN_T + annotate.MARGIN_B

    for event in (
        annotate.cv2.EVENT_LBUTTONDOWN,
        annotate.cv2.EVENT_RBUTTONDOWN,
    ):
        annotate.on_mouse(
            event,
            annotate.MARGIN_L + 100,
            annotate.MARGIN_T + 100,
            0,
            state,
        )

    # Panel buttons are modal too: neither ANNOT-ONLY nor SESSION may fire.
    for rect in (
        annotate.annot_button_rect(canvas_h),
        annotate.session_button_rect(canvas_h),
    ):
        x0, y0, x1, y1 = rect
        annotate.on_mouse(
            annotate.cv2.EVENT_LBUTTONDOWN,
            canvas_w + (x0 + x1) // 2,
            (y0 + y1) // 2,
            0,
            state,
        )

    assert state.kps_2d[0] == before
    assert state.active == active_before
    assert state.dirty == dirty_before
    assert state.annot_only == annot_only_before
    assert getattr(state, "sess_open", False) == sess_open_before


def test_csv_failure_after_gt_commit_is_partial_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = tmp_path / "session"
    _session(session)
    annotation = tmp_path / "annotations/000007.json"
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    state = _tag_state(session, annotation, "000007.png")
    state.pose = {"reproj_error_px": 0.0}
    annotate._handle_condition_key(
        ord("1"), state, str(annotation), "unused.png", str(source), np.eye(3))
    monkeypatch.setattr(annotate, "_make_state_annotation", lambda *_a: {"objects": [{}]})
    monkeypatch.setattr(
        annotate,
        "update_frame_tags_csv",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    assert not annotate._save_state_annotation(
        state, np.eye(3), str(annotation), str(annotation.with_suffix(".png")), str(source))
    assert annotation.is_file()
    assert state.frame_tags_dirty is True
    assert state.annotation_dirty is False
    assert "[SAVE PARTIAL]" in capsys.readouterr().out


def test_dev_reconcile_uses_frame_over_json_over_session(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    session = root / "dev_existing/sessions/dev_01"
    _session(session, population_role="DEV", lighting="day")
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin,occlusion,truncation\n"
        "000001.jpg,far,small,high,side,heavy,\n",
        encoding="utf-8",
    )
    annotation = root / "dev_existing/annotations/dev_01/000001.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps({"objects": [{
            "occlusion_level": "none",
            "truncation": {"is_truncated": True},
            "keypoint_annotations": [],
        }]}),
        encoding="utf-8",
    )
    row = reconcile_annotation_state(root, {
        "frame_id": "dev_01__000001",
        "population_role": "DEV",
        "session_id": "dev_01",
        "image_path": "dev_existing/sessions/dev_01/rgb/000001.jpg",
        "annotation_path": "dev_existing/annotations/dev_01/000001.json",
    })
    assert row["occlusion"] == "heavy"
    assert row["truncation"] == "mild"
    assert row["lighting"] == "day"
    assert row["distance_bin"] == "far"
    assert row["size_bin"] == "small"
    assert row["elevation_bin"] == "high"
    assert row["view_bin"] == "side"


def test_incomplete_optional_metadata_is_not_an_annotation_gate() -> None:
    row = {
        "frame_id": "final_01__000001",
        "session_id": "final_01",
        "population_role": "FINAL",
        "is_positive": "true",
        "is_annotated": "true",
        "controlled_eval_eligible": "true",
        "object_type": "plastic",
        "lighting": "day",
        "occlusion": "none",
        "truncation": "none",
        "distance_bin": "unknown",
        "size_bin": "large",
        "elevation_bin": "low",
        "view_bin": "front",
        "image_sha256": hashlib.sha256(b"final_01__000001").hexdigest(),
        "image_path": "final/positive/sessions/final_01/rgb/000001.png",
    }
    report = render_priority_report([row], DEFAULT_TARGETS)
    assert "New annotation required   NO" in report
    assert "final_01__000001" not in report


def test_legacy_size_is_not_an_active_annotation_missing_gate() -> None:
    state = State()
    state.frame_tags = {
        "occlusion": "none",
        "truncation": "none",
        "elevation_bin": "mid",
        "distance_bin": "near",
        "size_bin": "unknown",
        "view_bin": "unknown",
    }
    assert annotate._missing_evaluation_tags(state) == []


def test_session_frame_discovery_includes_png_jpg_and_jpeg_only(tmp_path: Path) -> None:
    rgb = tmp_path / "session/rgb"
    rgb.mkdir(parents=True)
    for name in ("a.png", "b.jpg", "c.JPEG", "ignore.txt"):
        (rgb / name).write_bytes(b"x")
    assert [Path(path).name for path in annotate._session_image_paths(tmp_path / "session")] == [
        "a.png", "b.jpg", "c.JPEG"
    ]
