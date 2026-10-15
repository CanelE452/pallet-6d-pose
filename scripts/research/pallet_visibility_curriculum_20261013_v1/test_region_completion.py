"""Finite three-point roots and non-fabricated region support."""
from copy import deepcopy

import numpy as np

from . import common as C
from .anchored_completion_api import AnchoredCompletion
from .region_completion import p3p_candidates, select_region_pose
from .test_boundary_rule_api import Provider, inputs
from .test_boundary_rules import scene


def test_p3p_enumeration_contains_the_exact_pose_without_hidden_coordinates():
    xyz, R, t, K = scene()
    q = C.project(xyz, R, t, K)
    before = q.copy()
    candidates, record = p3p_candidates(xyz, q, K, [0, 4, 7])
    assert 1 <= len(candidates) <= 4
    assert any(np.linalg.norm(np.asarray(c["R"])-R) < 1e-6
               and np.linalg.norm(np.asarray(c["t"])-t) < 1e-6 for c in candidates)
    for candidate in candidates:
        ids = candidate["used_ids"]
        np.testing.assert_allclose(C.project(xyz, candidate["R"], candidate["t"], K)[ids], q[ids],
                                   atol=1e-5, rtol=0)
    np.testing.assert_array_equal(q, before)
    assert record["full_pose_uniqueness_proved"] is False


def test_two_anchors_do_not_generate_pose_roots_from_hidden_predictions():
    xyz, R, t, K = scene()
    candidates, record = p3p_candidates(xyz, C.project(xyz, R, t, K), K, [0, 4])
    assert candidates == [] and record["reason"] == "fewer_than_three_image_anchors"


def test_p3p_uses_full_intrinsics_including_skew():
    xyz, R, t, K = scene()
    K[0, 1] = 51.
    q = C.project(xyz, R, t, K)
    candidates, _ = p3p_candidates(xyz, q, K, [0, 4, 7])
    assert any(np.linalg.norm(np.asarray(c["R"])-R) < 1e-6
               and np.linalg.norm(np.asarray(c["t"])-t) < 1e-6 for c in candidates)


def test_collinear_precise_correspondences_are_rejected_before_p3p():
    xyz, R, t, K = scene()
    q = C.project(xyz, R, t, K)
    q[[0, 4, 7]] = [[0., 0.], [1., 1.], [2., 2.]]
    candidates, record = p3p_candidates(xyz, q, K, [0, 4, 7])
    assert candidates == [] and record["reason"] == "degenerate_image_or_object_triangle"


def test_no_regions_and_no_candidates_preserve_the_original_solution_exactly():
    image, q, K, initial, observation = inputs(3)
    base = AnchoredCompletion(Provider(observation), {"supplement": False}).refine(
        image, q, K, initial, [1.3, .15, 1.1])["solution"]
    before = deepcopy(base)
    result = select_region_pose(base, C.cuboid(*initial["cf_extents"]), K, initial["R_cf"],
                                initial["centroid"], [], config={"include_p3p": False})
    assert result["R"] == base["R"] and result["t"] == base["t"]
    assert result["used_ids"] == base["used_ids"] and result["estimated_dof"] == 3
    assert result["region_audit"]["selected"] == "BASE"
    assert result["region_audit"]["regions_are_precise_correspondences"] is False
    assert base == before
