"""Regression coverage for registry-backed plastic/wood annotation dispatch."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import pytest
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
ANNOTATE = REPO / "scripts" / "annotate"
sys.path.insert(0, str(ANNOTATE))

import annotate  # noqa: E402
import annotate_pnp  # noqa: E402
import annotate_wood  # noqa: E402
from annotate_io import (  # noqa: E402
    AnnotationGeometryMismatch,
    State,
    load_existing_annotation,
    make_annotation,
)
from object_geometry_registry import (  # noqa: E402
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)
from real_gt_v2_schema import validate_gt_v2  # noqa: E402


REGISTRY = load_object_geometry_registry()
PLASTIC = REGISTRY.resolve(PLASTIC_OBJECT_TYPE)
WOOD = REGISTRY.resolve(WOOD_OBJECT_TYPE)
K = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
R, _ = cv2.Rodrigues(np.array([0.08, -0.12, 0.03], dtype=np.float64))
T = np.array([0.05, 0.03, 4.5], dtype=np.float64)


def _pose(spec, *, swapped=False):
    base = spec.legacy_wdh_tuple
    dims = ((base[1], base[0], base[2]) if swapped else base)
    projected = annotate_pnp.project_3d(
        annotate_pnp.make_pallet_keypoints_3d(*dims), R, T, K)
    return projected, {
        "R": R,
        "t": T,
        "dims": dims,
        "projected_all": projected,
        "reproj_error_px": 0.0,
        "_physical_dimensions_m": spec.physical_dimensions_m,
        "_wd_candidates": [],
    }


def _annotation(spec, role="DEV", *, swapped=False):
    points, pose = _pose(spec, swapped=swapped)
    kwargs = {}
    if spec.object_type == WOOD_OBJECT_TYPE:
        kwargs.update({
            "intrinsics_quality": "SENSOR_PROFILE_SCALED",
            "intrinsics_source": "test sensor profile",
        })
    return make_annotation(
        points,
        pose,
        (480, 640, 3),
        K,
        geometry_spec=spec,
        population_role=role,
        **kwargs,
    )


def test_repo_root_registry_dims_and_canonical_default_output():
    assert annotate.find_repo_root(ANNOTATE) == str(REPO)
    assert annotate_wood.find_repo_root(ANNOTATE) == REPO
    assert Path(annotate_wood._REPO) == REPO
    assert annotate_wood.WOOD_DIMS == (0.8, 0.59, 0.14)
    assert annotate_wood.default_output_dir(
        "pallet_20260618_183705") == (
            REPO / "challenge" / "data" / "01_real" /
            "gt_v2_canonical" / "manual_gt" /
            "wood_pallet_20260618_183705_manual_gt")
    assert annotate_wood.legacy_read_dir(
        "pallet_20260618_183705") == (
            REPO / "challenge" / "data" / "01_real" / "manual_gt" /
            "wood_pallet_20260618_183705_manual_gt")
    with pytest.raises(ValueError, match="one folder name"):
        annotate_wood.default_output_dir("../plastic")


def test_wood_import_and_dispatch_never_mutate_global_pallet_dims(tmp_path):
    original = annotate_pnp.PALLET_DIMS
    importlib.reload(annotate_wood)
    assert annotate_pnp.PALLET_DIMS == original

    legacy = (tmp_path / "challenge" / "data" / "01_real" / "manual_gt" /
              "wood_pallet_20260618_183705_manual_gt")
    legacy.mkdir(parents=True)
    argv = annotate_wood.build_annotate_argv(
        video="pallet_20260618_183705",
        sequence=tmp_path / "stage",
        stride=5,
        start=0,
        out_dir=tmp_path / "canonical",
        population_role="FINAL",
        registry_path=REGISTRY.source_path,
        intrinsics_quality="CALIBRATED",
        intrinsics_source="per-session calibration",
        repo=tmp_path,
    )
    assert argv[argv.index("--object-type") + 1] == WOOD_OBJECT_TYPE
    assert argv[argv.index("--population-role") + 1] == "FINAL"
    assert argv[argv.index("--legacy-read-dir") + 1] == str(legacy)
    assert argv[argv.index("--capture-session-id") + 1] == "wood_183705"

    state = State()
    state.geometry_spec = WOOD
    state.kps_2d = [None] * 9
    state.extrap_mask = [False] * 9
    state.img_shape = (720, 1280, 3)
    with mock.patch.object(annotate, "solve_pose", return_value=None) as solve:
        annotate.update_pose(state, K, force=True)
    assert solve.call_args.kwargs["physical_dimensions"] == WOOD.physical_dimensions
    assert annotate_pnp.PALLET_DIMS == original


def test_plastic_and_wood_outputs_are_order_independent_and_axis_is_object_aware():
    original = annotate_pnp.PALLET_DIMS
    first_wood = _annotation(WOOD)
    plastic = _annotation(PLASTIC)
    second_wood = _annotation(WOOD)

    for document in (first_wood, plastic, second_wood):
        validate_gt_v2(document)
    assert first_wood == second_wood
    assert annotate_pnp.PALLET_DIMS == original

    # Plastic's legacy-compatible output remains implicit, while wood is
    # necessarily explicit at root and object level.
    assert "object_type" not in plastic
    assert "object_type" not in plastic["objects"][0]
    assert first_wood["object_type"] == WOOD_OBJECT_TYPE
    assert first_wood["objects"][0]["object_type"] == WOOD_OBJECT_TYPE
    assert first_wood["objects"][0]["physical_dimensions_m"] == {
        "x": 0.8, "y": 0.14, "z": 0.59,
    }

    # Wood has X>Z, the inverse magnitude ordering from plastic.  Axis parity
    # must come from named geometry, never from width > depth.
    assert [item["axis_assignment"] for item in
            first_wood["objects"][0]["canonical_pose_candidates"]] == [
                "YAW_0", "YAW_180",
            ]
    swapped = _annotation(WOOD, swapped=True)
    validate_gt_v2(swapped)
    assert [item["axis_assignment"] for item in
            swapped["objects"][0]["canonical_pose_candidates"]] == [
                "YAW_90", "YAW_270",
            ]


def test_pose_and_existing_label_cross_object_mix_fail_closed(tmp_path):
    points, pose = _pose(PLASTIC)
    with pytest.raises(ValueError, match="selected object_type"):
        make_annotation(
            points, pose, (480, 640, 3), K,
            geometry_spec=WOOD,
            intrinsics_quality="UNKNOWN",
        )

    plastic_legacy = {
        "camera_data": {"width": 640, "height": 480},
        "objects": [{
            "manual_kps": [[float(i), float(i)] for i in range(9)],
            "dimensions_m": {"width": 1.1, "height": 0.11, "depth": 1.3},
        }],
    }
    source = tmp_path / "plastic.json"
    source.write_text(json.dumps(plastic_legacy), encoding="utf-8")
    before = source.read_bytes()
    state = State()
    state.geometry_spec = WOOD
    state.geometry_registry = REGISTRY
    with pytest.raises(AnnotationGeometryMismatch, match="does not match requested"):
        load_existing_annotation(state, str(source), read_only=True)
    assert source.read_bytes() == before
    assert not Path(str(source) + ".corrupt").exists()


def test_legacy_wood_source_is_only_read_and_staging_is_separate(tmp_path):
    source_root = tmp_path / "source"
    video = "pallet_20260618_183705"
    source_dir = source_root / video
    source_dir.mkdir(parents=True)
    image_path = source_dir / "000000.jpg"
    Image.new("RGB", (16, 9), color=(20, 30, 40)).save(image_path)
    image_before = image_path.read_bytes()

    legacy = annotate_wood.legacy_read_dir(video, tmp_path)
    legacy.mkdir(parents=True)
    label = legacy / "000000.json"
    label.write_text('{"legacy": true}', encoding="utf-8")
    label_before = label.read_bytes()

    sequence, count = annotate_wood.prep_seq(
        video, np.eye(3), repo=tmp_path, wood_root=source_root)
    assert count == 1
    assert Path(sequence) != legacy
    assert (Path(sequence) / "rgb" / "000000.png").is_symlink()
    assert image_path.read_bytes() == image_before
    assert label.read_bytes() == label_before


@pytest.mark.parametrize("role", ["DEV", "FINAL"])
def test_wood_schema_accepts_current_dev_and_future_final(role):
    document = _annotation(WOOD, role=role)
    validate_gt_v2(document)
    assert document["population_role"] == role
