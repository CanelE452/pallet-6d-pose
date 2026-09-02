"""A square footprint (x == z) has one camera-facing hypothesis, not two."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATE = REPO_ROOT / "scripts" / "annotate"
sys.path.insert(0, str(ANNOTATE))

from annotate_pnp import _physical_wd_hypotheses  # noqa: E402
from pallet_geometry import (  # noqa: E402
    AxisAssignment,
    camera_facing_hypothesis_name,
)

CHALLENGE_REGISTRY = (
    REPO_ROOT / "challenge" / "config" / "CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
)

SQUARE = {"x": 1.10, "y": 0.15, "z": 1.10}
PAPER_PLASTIC = {"x": 1.10, "y": 0.11, "z": 1.30}
WOOD = {"x": 0.80, "y": 0.14, "z": 0.59}


def test_square_footprint_is_named_square_not_short_or_long() -> None:
    for axis in (AxisAssignment.YAW_0, AxisAssignment.YAW_90):
        assert camera_facing_hypothesis_name(axis, SQUARE) == "square-face-front"


def test_square_footprint_yields_a_single_hypothesis() -> None:
    """YAW_90 would be bit-identical to YAW_0, so scoring two is meaningless."""
    hypotheses = _physical_wd_hypotheses(SQUARE)
    assert len(hypotheses) == 1
    assert hypotheses[0]["camera_facing_hypothesis"] == "square_face_front"
    assert hypotheses[0]["axis_assignment"] == "YAW_0"
    assert hypotheses[0]["axis_assignment_candidates"] == ["YAW_0", "YAW_180"]


@pytest.mark.parametrize(
    "physical, expected",
    [
        (PAPER_PLASTIC, ["short_face_front", "long_face_front"]),
        (WOOD, ["long_face_front", "short_face_front"]),
    ],
)
def test_rectangular_objects_keep_both_hypotheses_and_their_names(
        physical, expected) -> None:
    """The square path must not disturb the two locked rectangular objects."""
    hypotheses = _physical_wd_hypotheses(physical)
    assert [h["camera_facing_hypothesis"] for h in hypotheses] == expected
    assert [h["axis_assignment"] for h in hypotheses] == ["YAW_0", "YAW_90"]


def _square_annotation():
    """Build a GT v2 document for the square field pallet, as the tool would."""
    import cv2
    import numpy as np

    import annotate_pnp
    from annotate_io import make_annotation
    from object_geometry_registry import (
        PLASTIC_SQUARE_OBJECT_TYPE,
        load_object_geometry_registry,
    )

    spec = load_object_geometry_registry(
        CHALLENGE_REGISTRY).resolve(PLASTIC_SQUARE_OBJECT_TYPE)
    K = np.array([[605.9064941406, 0.0, 317.5961914062],
                  [0.0, 605.9697875977, 256.2922973633],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.array([0.08, -0.12, 0.03], dtype=np.float64))
    T = np.array([0.05, 0.03, 3.0], dtype=np.float64)
    dims = spec.legacy_wdh_tuple
    points = annotate_pnp.project_3d(
        annotate_pnp.make_pallet_keypoints_3d(*dims), R, T, K)
    pose = {
        "R": R,
        "t": T,
        "dims": dims,
        "projected_all": points,
        "reproj_error_px": 0.0,
        "_physical_dimensions_m": spec.physical_dimensions_m,
        "_wd_candidates": [],
    }
    return make_annotation(
        points, pose, (480, 640, 3), K,
        geometry_spec=spec,
        population_role="DEV",
        intrinsics_quality="CALIBRATED",
        intrinsics_source="realsense_factory_intrinsics_meta_json",
    )


def test_saving_a_square_pallet_passes_schema_with_its_own_registry() -> None:
    """This is what '[SAVE BLOCKED] v2 schema error' was: the paper registry."""
    from real_gt_v2_schema import validate_gt_v2

    document = _square_annotation()
    validate_gt_v2(document, registry_path=CHALLENGE_REGISTRY)


def test_square_pallet_is_still_unknown_to_the_paper_registry() -> None:
    """The challenge object must not leak into paper validation."""
    from real_gt_v2_schema import validate_gt_v2

    document = _square_annotation()
    with pytest.raises(Exception, match="UNKNOWN_OBJECT_TYPE"):
        validate_gt_v2(document)
