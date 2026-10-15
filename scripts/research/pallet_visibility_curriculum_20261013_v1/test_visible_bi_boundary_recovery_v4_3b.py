"""Observed interval guard tests, independent of all image/pose annotations."""
from copy import deepcopy

import numpy as np

from . import visible_bi_boundary_recovery_v4_3 as OLD
from . import visible_bi_boundary_recovery_v4_3b as NEW
from .test_visible_bi_boundary_recovery_v4_3 import displaced_scene, short_edge_scene


def rays():
    return [dict(supported=True, normal=[1., 0.], edge=10, native_neighbor=2,
                 observed_points=[[10., 4.], [10., 10.], [10., 20.]], inlier_mask=[True, True, True]),
            dict(supported=True, normal=[0., 1.], edge=13, native_neighbor=5,
                 observed_points=[[14., 0.], [22., 0.], [30., 0.]], inlier_mask=[True, True, True])]


def solve(args, module):
    image, mask, q, K, X, pose, states, _, base = args
    return module.recover_visible_boundary(image, q, K, pose, states,
        pallet_mask=mask, base_correction=base, native_xyz=X)


def test_near_corner_and_exact_existing_margin_pass_both_support_intervals():
    result = NEW._support_interval_guard([10., 0.], rays(), 4.+7.)
    assert result["passed"]
    assert [r["extrapolation_px"] for r in result["rays"]] == [4., 4.]
    assert NEW._support_interval_guard([10., -7.], rays(), 11.)["passed"]


def test_infinite_line_extrapolation_beyond_one_supported_segment_rejects():
    r = rays(); r[0]["observed_points"] = [[10., 21.27], [10., 25.], [10., 30.]]
    audit = NEW._support_interval_guard([10., 0.], r, 4.+7.)
    assert not audit["passed"]
    assert audit["rays"][0]["extrapolation_px"] == 21.27
    assert audit["rays"][1]["passed"]


def test_outlier_sample_cannot_extend_actual_inlier_interval():
    r = rays(); r[0]["observed_points"] = [[10., 0.], [10., 25.], [10., 30.]]
    r[0]["inlier_mask"] = [False, True, True]
    result = NEW._support_interval_guard([10., 0.], r, 11.)
    assert not result["passed"] and result["rays"][0]["extrapolation_px"] == 25.


def test_tangent_sign_and_image_rotation_do_not_change_interval_distance():
    first = NEW._support_interval_guard([10., 0.], rays(), 11.)
    angle = .67; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = rays()
    for ray in transformed:
        ray["normal"] = (-rotation@np.asarray(ray["normal"])).tolist()
        ray["observed_points"] = (np.asarray(ray["observed_points"])@rotation.T).tolist()
    second = NEW._support_interval_guard(rotation@np.array([10., 0.]), transformed, 11.)
    assert second["passed"]
    np.testing.assert_allclose([r["extrapolation_px"] for r in first["rays"]],
                               [r["extrapolation_px"] for r in second["rays"]], atol=1e-12)


def test_actual_synthetic_supported_corners_and_all_measurements_are_parent_exact():
    for args in (displaced_scene(), short_edge_scene()):
        old = solve(args, OLD); new = solve(args, NEW)
        assert new["recovered_native_ids"] == old["recovered_native_ids"]
        assert new["visible_measurements"] == old["visible_measurements"]
        assert new["records"] == old["records"]
        assert new["measured_points"] == old["measured_points"]
        assert new["boundary_recovery_support_guard"]["maximum_extrapolation_px"] == 11.
        assert new["boundary_recovery_support_guard"]["rejected_native_ids"] == []


def test_rejected_corner_restores_exact_frozen_base_and_preserves_other_BI(monkeypatch):
    args = displaced_scene(); old = solve(args, OLD); parent = deepcopy(old); base = deepcopy(args[8])
    k = old["recovered_native_ids"][0]
    row = parent["boundary_recovery_records"][k]
    point = np.asarray(row["intersection_xy"])
    # Keep the same intersection and normal; move only actual inlier support
    # along its line so the extrapolated corner is unsupported by the segment.
    ray = row["selected_rays"][0]; n = np.asarray(ray["normal"])
    tangent = np.array([-n[1], n[0]])
    ray["observed_points"] = (point + np.array([25., 30., 35.])[:, None]*tangent).tolist()
    ray["inlier_mask"] = [True, True, True]
    monkeypatch.setattr(NEW.ORIGINAL, "recover_visible_boundary", lambda *a, **kw: deepcopy(parent))
    new = solve(args, NEW)
    assert new["records"][k] == base["records"][k]
    old_measurement = next(m for m in base["visible_measurements"] if m["native_corner"] == k)
    assert next(m for m in new["visible_measurements"] if m["native_corner"] == k) == old_measurement
    assert new["measured_points"][k] == base["measured_points"][k]
    assert k not in new["recovered_native_ids"]
    assert new["boundary_recovery_records"][k]["rejected_intersection_xy"] == row["intersection_xy"]
    assert new["boundary_recovery_records"][k]["reason"] == "intersection_extrapolates_beyond_observed_inlier_support"
    for j in range(8):
        if j != k:
            assert new["records"][j] == old["records"][j]
            assert new["measured_points"][j] == old["measured_points"][j]
    assert new["measured_points"][8] == base["measured_points"][8]
    assert args[8] == base


def test_missing_observed_inliers_is_explicit_abstention():
    r = rays(); r[0]["inlier_mask"] = [False, False, False]
    result = NEW._support_interval_guard([10., 0.], r, 11.)
    assert not result["passed"]
    assert result["rays"][0]["reason"] == "missing_finite_observed_inlier_support"


def test_opt_in_RGB_adapter_equals_explicit_legacy_then_recovery_composition():
    args = displaced_scene()
    image, mask, q, K, X, pose, states, _, _ = args
    expected = solve(args, NEW)
    actual = NEW.correct_visible_bi(image, q, K, pose, states, pallet_mask=mask, native_xyz=X)
    assert actual == expected
