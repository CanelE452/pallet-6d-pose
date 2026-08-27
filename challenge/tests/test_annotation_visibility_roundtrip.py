"""Regression tests for additive GT v2 annotation save/load semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ANNOTATE = Path(__file__).resolve().parents[2] / "scripts" / "annotate"
sys.path.insert(0, str(ANNOTATE))

from annotate import (  # noqa: E402
    _cycle_wd_parity,
    _delete_annotation,
    _require_nonlegacy_output_dir,
    _save_contract_error,
    resolve_out_dir,
)
from annotate_draw import draw_overlay  # noqa: E402
from annotate_io import (  # noqa: E402
    State,
    load_existing_annotation,
    make_annotation,
    save_frame_json,
)
from annotate_pnp import (  # noqa: E402
    make_pallet_keypoints_3d,
    project_3d,
    solve_pose,
)
from pallet_geometry import canonical_dimensions  # noqa: E402
from real_gt_v2_schema import validate_gt_v2  # noqa: E402
from fix_manual_swap import fix_one  # noqa: E402
from _repnp_with_new_dims import repnp_one  # noqa: E402


def _fixture():
    K = np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dims = (1.1, 1.3, 0.11)
    R, _ = cv2.Rodrigues(np.array([0.08, -0.12, 0.03], dtype=np.float64))
    t = np.array([0.05, 0.03, 4.5], dtype=np.float64)
    projected = project_3d(make_pallet_keypoints_3d(*dims), R, t, K)
    pose = {
        "R": R,
        "t": t,
        "dims": dims,
        "projected_all": projected,
        "reproj_error_px": 0.0,
        "_axis_assignment": "YAW_0",
        "_axis_assignment_candidates": ["YAW_0", "YAW_180"],
        "_wd_selection_reason": "geometry_rank_clear",
        "_wd_candidates": [],
    }
    return K, projected, pose


def _annotations(points):
    states = [
        (2, "manual_click", "visible"),
        (1, "extrapolated", "occluded"),
        (1, "pnp_projected", "truncated"),
        (0, "unknown", "unknown"),
        (2, "manual_click", "visible"),
        (1, "extrapolated", "occluded"),
        (1, "pnp_projected", "truncated"),
        (2, "manual_click", "visible"),
        (1, "centroid_auto", "unknown"),
    ]
    result = []
    for point, (visibility, source, reason) in zip(points, states):
        result.append({
            "xy": list(point),
            "visibility": visibility,
            "in_frame": True,
            "source": source,
            "reason": reason,
        })
    return result


def test_visibility_source_reason_roundtrip_and_atomic_save(tmp_path):
    K, points, pose = _fixture()
    annotations = _annotations(points)
    ann = make_annotation(
        [list(point) for point in points],
        pose,
        (480, 640, 3),
        K,
        keypoint_annotations=annotations,
        axis_assignment="YAW_0",
        axis_assignment_candidates=["YAW_0", "YAW_180"],
        axis_assignment_confirmed=True,
        population_role="DEV",
        metadata={
            "capture_session_id": "session-a",
            "camera_serial": "camera-1",
            "capture_timestamp": "2026-08-27T12:34:56+09:00",
            "lighting_condition": "day",
        },
    )
    validate_gt_v2(ann)

    canonical = canonical_dimensions()
    assert ann["objects"][0]["physical_dimensions_m"] == canonical.as_dict()
    assert ann["objects"][0]["canonical_pose"]["axis_assignment"] == "YAW_0"
    assert {item["axis_assignment"] for item in
            ann["objects"][0]["canonical_pose_candidates"]} == {
                "YAW_0", "YAW_180",
            }

    source_png = tmp_path / "source.png"
    assert cv2.imwrite(str(source_png), np.zeros((4, 4, 3), dtype=np.uint8))
    out_json = tmp_path / "frame.json"
    out_png = tmp_path / "frame.png"
    save_frame_json(str(out_json), str(out_png), str(source_png), ann)
    assert out_json.exists() and out_png.exists()
    assert not Path(str(out_json) + ".tmp").exists()

    state = State()
    state.kps_2d = [None] * 9
    state.extrap_mask = [False] * 9
    assert load_existing_annotation(state, str(out_json))
    assert state.keypoint_annotations == annotations
    assert state.axis_assignment == "YAW_0"
    assert state.axis_assignment_confirmed is True
    assert state.capture_metadata["camera_serial"] == "camera-1"


def test_legacy_load_preserves_coordinates_but_does_not_infer_visibility(tmp_path):
    _, points, pose = _fixture()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose["R"]
    transform[:3, 3] = pose["t"]
    legacy = {
        "camera_data": {"width": 640, "height": 480},
        "objects": [{
            "manual_kps": [list(point) for point in points],
            "extrapolated_mask": [False, True] + [False] * 7,
            "projected_cuboid": [list(point) for point in points[:8]],
            "projected_cuboid_centroid": list(points[8]),
            "dimensions_m": {"width": 1.1, "height": 0.11, "depth": 1.3},
            "pose_transform": transform.tolist(),
        }],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    state = State()
    state.kps_2d = [None] * 9
    state.extrap_mask = [False] * 9
    state.keypoint_annotations = None
    assert load_existing_annotation(state, str(path))
    assert state.kps_2d == legacy["objects"][0]["manual_kps"]
    assert all(entry["visibility"] == 0 for entry in state.keypoint_annotations)
    assert all(entry["reason"] == "unknown" for entry in state.keypoint_annotations)
    assert state.keypoint_annotations[1]["source"] == "extrapolated"
    assert state.keypoint_annotations[0]["source"] == "unknown"


def test_legacy_load_save_is_additive_and_preserves_all_compatibility_fields(
        tmp_path):
    K, points, pose = _fixture()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose["R"]
    transform[:3, 3] = pose["t"]
    original_object = {
        "class": "pallet",
        "name": "legacy-name-must-survive",
        "visibility": 0.375,
        "pose_transform": transform.tolist(),
        "projected_cuboid": [list(point) for point in points[:8]],
        "projected_cuboid_centroid": list(points[8]),
        "dimensions_m": {"width": 1.3, "height": 0.11, "depth": 1.1},
        "gt_source": "manual-reviewed-legacy",
        "manual_kps": [list(point) for point in points],
        "reproj_error_px": 17.25,
        "split": "eval",
        "tag_id": 42,
        "tag_decision_margin": 81.125,
        "sentinel_repaired": True,
        "fix_swap": {"legacy": "value"},
    }
    original = {
        "camera_data": {
            "width": 640,
            "height": 480,
            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
            "legacy_camera_note": "keep-me",
        },
        "legacy_root_note": {"nested": [1, 2, 3]},
        "objects": [original_object],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(original), encoding="utf-8")

    state = State()
    state.kps_2d = [None] * 9
    state.extrap_mask = [False] * 9
    state.keypoint_annotations = None
    assert load_existing_annotation(state, str(path))

    converted = make_annotation(
        state.kps_2d,
        pose,
        (480, 640, 3),
        K,
        keypoint_annotations=state.keypoint_annotations,
        axis_assignment_candidates=["YAW_0", "YAW_180"],
        legacy_object=state.legacy_object,
        legacy_document=state.legacy_document,
        population_role="DEV",
    )
    validate_gt_v2(converted)
    for key, value in original_object.items():
        assert converted["objects"][0][key] == value, key
    assert converted["camera_data"] == original["camera_data"]
    assert converted["legacy_root_note"] == original["legacy_root_note"]
    assert converted["objects"][0]["legacy"]["dimensions_m"] == (
        original_object["dimensions_m"])
    assert converted["objects"][0]["physical_dimensions_m"] == (
        canonical_dimensions().as_dict())

    # The frozen top-level manual_kps remains a legacy audit value, while the
    # v2 point is authoritative for editor reload/save.
    edited_xy = [float(points[0][0] + 13.0), float(points[0][1] - 7.0)]
    converted["objects"][0]["keypoint_annotations"][0].update({
        "xy": edited_xy,
        "visibility": 2,
        "source": "manual_click",
        "reason": "visible",
    })
    v2_path = tmp_path / "edited-v2.json"
    v2_path.write_text(json.dumps(converted), encoding="utf-8")
    reloaded = State()
    reloaded.kps_2d = [None] * 9
    reloaded.extrap_mask = [False] * 9
    reloaded.keypoint_annotations = None
    assert load_existing_annotation(reloaded, str(v2_path))
    assert reloaded.kps_2d[0] == edited_xy
    assert reloaded.legacy_object["manual_kps"][0] == original_object[
        "manual_kps"][0]

    saved_again = make_annotation(
        reloaded.kps_2d,
        pose,
        (480, 640, 3),
        K,
        extrap_mask=reloaded.extrap_mask,
        keypoint_annotations=reloaded.keypoint_annotations,
        axis_assignment=reloaded.axis_assignment,
        axis_assignment_candidates=reloaded.axis_assignment_candidates,
        axis_assignment_confirmed=reloaded.axis_assignment_confirmed,
        legacy_object=reloaded.legacy_object,
        legacy_document=reloaded.legacy_document,
        population_role="DEV",
    )
    validate_gt_v2(saved_again)
    assert saved_again["objects"][0]["keypoint_annotations"][0]["xy"] == edited_xy
    assert saved_again["objects"][0]["manual_kps"] == original_object["manual_kps"]
    assert saved_again["objects"][0]["dimensions_m"] == original_object[
        "dimensions_m"]
    assert saved_again["objects"][0]["pose_transform"] == original_object[
        "pose_transform"]


def test_final_population_blocks_unknown_corner_visibility():
    _, points, _ = _fixture()
    state = State()
    state.img_shape = (480, 640, 3)
    state.kps_2d = [list(point) for point in points]
    state.keypoint_annotations = _annotations(points)
    for entry in state.keypoint_annotations[:8]:
        if entry["visibility"] == 0:
            entry["visibility"] = 1
            entry["reason"] = "occluded"
    state.keypoint_annotations[6]["visibility"] = 0
    state.keypoint_annotations[6]["reason"] = "unknown"
    state.population_role = "FINAL"
    state.axis_assignment = "YAW_0"
    state.axis_assignment_candidates = ["YAW_0", "YAW_180"]
    state.axis_assignment_confirmed = True
    assert "kp0~7 visibility unknown at 6" in _save_contract_error(state)

    state.keypoint_annotations[6]["visibility"] = 1
    state.keypoint_annotations[6]["reason"] = "occluded"
    assert _save_contract_error(state) is None


def test_final_population_cannot_delete_all_points_or_use_legacy_session_hint(
        tmp_path):
    out_json = tmp_path / "frame.json"
    out_json.write_text('{"keep": true}', encoding="utf-8")
    state = State()
    state.population_role = "FINAL"
    state.sess_sealed = False
    state.kps_2d = [None] * 9
    state.keypoint_annotations = [{
        "xy": [1.0, 1.0], "visibility": 2, "in_frame": True,
        "source": "manual_click", "reason": "visible",
    } for _ in range(9)]
    state.img_shape = (480, 640, 3)
    state.axis_assignment_confirmed = True
    assert "deleting all points" in _save_contract_error(state)
    assert _delete_annotation(
        state, str(out_json), str(tmp_path / "frame.png")) is None
    assert out_json.read_text(encoding="utf-8") == '{"keep": true}'
    assert not Path(str(out_json) + ".deleted").exists()

    # The inverse combination confirms that the old session/path hint is not
    # consulted for role warnings or gates.
    dev_json = tmp_path / "dev.json"
    dev_json.write_text('{"dev": true}', encoding="utf-8")
    state.population_role = "DEV"
    state.sess_sealed = True
    assert _delete_annotation(
        state, str(dev_json), str(tmp_path / "dev.png")) == "save-next"
    assert Path(str(dev_json) + ".deleted").exists()


def test_default_output_is_v2_and_explicit_legacy_or_symlink_path_is_rejected(
        tmp_path):
    repo = tmp_path / "repo"
    legacy = (repo / "challenge" / "data" / "01_real" /
              "eval_canonical" / "capture_manual_gt")
    legacy.mkdir(parents=True)
    out_dir, eval_layout = resolve_out_dir("capture", str(repo))
    assert eval_layout is True
    assert Path(out_dir) == (repo / "challenge" / "data" / "01_real" /
                             "gt_v2_canonical" / "eval_canonical" /
                             "capture_manual_gt")
    assert _require_nonlegacy_output_dir(out_dir, str(repo)) == str(Path(out_dir))

    with pytest.raises(ValueError, match="legacy real GT is read-only"):
        _require_nonlegacy_output_dir(str(legacy), str(repo))
    alias = repo / "legacy-alias"
    alias.symlink_to(legacy.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="legacy real GT is read-only"):
        _require_nonlegacy_output_dir(str(alias / "capture_manual_gt"), str(repo))

    corrupt = legacy / "corrupt.json"
    corrupt.write_bytes(b'{"objects": [')
    before = corrupt.read_bytes()
    state = State()
    assert load_existing_annotation(state, str(corrupt), read_only=True) is False
    assert corrupt.read_bytes() == before
    assert not Path(str(corrupt) + ".corrupt").exists()


def test_ui_wd_parity_correction_resets_sign_and_selects_other_pnp_candidate():
    K, points, _ = _fixture()
    automatic = solve_pose(
        points,
        K,
        img_shape=(480, 640),
        physical_dimensions=canonical_dimensions(),
    )
    assert automatic is not None
    current = automatic["_camera_facing_hypothesis"]
    target = ({"short_face_front", "long_face_front"} - {current}).pop()

    state = State()
    state.pose = automatic
    state.axis_assignment = automatic["_axis_assignment_candidates"][0]
    state.axis_assignment_candidates = list(
        automatic["_axis_assignment_candidates"])
    state.axis_assignment_confirmed = True
    _cycle_wd_parity(state)
    assert state.camera_facing_hypothesis_override == target
    assert state.axis_assignment is None
    assert state.axis_assignment_candidates == []
    assert state.axis_assignment_confirmed is False

    corrected = solve_pose(
        points,
        K,
        img_shape=(480, 640),
        physical_dimensions=canonical_dimensions(),
        camera_facing_hypothesis_override=target,
    )
    assert corrected["_camera_facing_hypothesis"] == target
    assert corrected["_wd_selection_reason"] == "manual_camera_facing_override"
    assert corrected["_wd_manual_override_available"] is True
    expected_signs = ({"YAW_0", "YAW_180"} if target == "short_face_front"
                      else {"YAW_90", "YAW_270"})
    assert set(corrected["_axis_assignment_candidates"]) == expected_signs


def test_named_physical_pnp_keeps_both_parities_and_selection_reason():
    K, points, _ = _fixture()
    pose = solve_pose(
        points,
        K,
        img_shape=(480, 640),
        physical_dimensions=canonical_dimensions(),
    )
    assert pose is not None
    assert len(pose["_wd_candidates"]) == 2
    assert pose["_wd_selection_reason"] in {
        "geometry_rank_clear", "geometry_rank_ambiguous", "geometry_tie_prior",
    }
    assert pose["_axis_assignment_candidates"] in (
        ["YAW_0", "YAW_180"],
        ["YAW_90", "YAW_270"],
    )
    expected = canonical_dimensions().as_dict()
    assert all(item["physical_dimensions_m"] == expected
               for item in pose["_wd_candidates"])


def test_deprecated_mutators_are_dry_run_without_explicit_allow(tmp_path):
    K, points, pose = _fixture()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose["R"]
    transform[:3, 3] = pose["t"]
    document = {
        "camera_data": {
            "intrinsics": {
                "fx": K[0, 0], "fy": K[1, 1],
                "cx": K[0, 2], "cy": K[1, 2],
            },
        },
        "objects": [{
            "manual_kps": [list(point) for point in points],
            "pose_transform": transform.tolist(),
            "reproj_error_px": 99.0,
        }],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    before = path.read_bytes()

    result, error = fix_one(str(path), K, dry_run=False)
    assert error is None and result is not None
    assert path.read_bytes() == before

    success, _ = repnp_one(str(path))
    assert success
    assert path.read_bytes() == before


def test_visibility_marker_and_line_styles_render_without_collapsing():
    _, points, pose = _fixture()
    annotations = _annotations(points)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_overlay(
        image,
        [list(point) for point in points],
        0,
        pose,
        extrap_mask=[False, True, True] + [False] * 6,
        keypoint_annotations=annotations,
    )
    assert rendered.shape == (880, 1040, 3)
    # Each of visible/occluded/truncated/unknown markers contributes pixels.
    for index in range(4):
        x = int(points[index][0] + 200)
        y = int(points[index][1] + 200)
        patch = rendered[y - 6:y + 7, x - 6:x + 7]
        assert np.count_nonzero(patch) > 0
