"""Independent trust-boundary and physical-frame checks without real labels."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from . import common as C
from .boundary_curriculum_api import BoundaryCurriculum
from .test_curriculum import fixture, appearance


class ReviewObserver:
    boundary_profile = {"maximum_corner_shift_px": 12.0}
    fusion = {"detector_sigma_px": 2.0, "maximum_alpha": .5}

    def __init__(self, evidence, action=None):
        self.evidence = evidence
        self.action = action

    def assess(self, image, raw, K, initial):
        H = C.self_hidden(C.cuboid(*initial["cf_extents"]), initial["R_cf"], initial["centroid"])
        if self.action == "mutate_image_in_place":
            image[0, 0] = [255, 255, 255]
        elif self.action == "mutate_raw_in_place":
            raw[0] += [1., 0.]
        elif self.action == "mutate_K_in_place":
            K[0, 0] *= 1.1
        elif self.action == "mutate_initial_in_place":
            initial["R_physical"] = (np.asarray(initial["R_physical"]) @ cv2.Rodrigues(np.array([0., .05, 0.]))[0]).tolist()
        fused = raw.copy()
        candidate = next(k for k in range(8) if not H[k] and np.isfinite(raw[k]).all()
                         and not np.all(raw[k] == -1))
        if self.action == "exceed_fused_bound":
            fused[candidate] += [7., 0.]
        elif self.action == "leave_image":
            fused[candidate, 0] = -.01
        elif self.action == "invalid_measurement":
            fused[candidate, 0] = np.nan
        return dict(raw_points=raw.tolist(), fused_points=fused.tolist(),
            self_hidden=H.tolist(), visibility_appearance=deepcopy(self.evidence),
            boundary_visibility={"states": ["NOT_ASSESSED"] * 8})


def review_inputs(center=True):
    _, K, R, t, R0, t0, _ = fixture()
    dimensions = [1.3, .15, 1.1]
    xyz = C.cuboid(*dimensions)
    q = C.project(xyz, R, t, K) + [1., -.5]
    if center:
        q = np.vstack([q, [321., 243.]])
    Q = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))[0]
    initial = dict(available=True, R_cf=R0.tolist(), R_physical=(R0 @ Q).tolist(),
                   centroid=t0.tolist(), cf_extents=dimensions)
    H = C.self_hidden(xyz, R0, t0)
    evidence = appearance(["SELF_OCCLUDED" if value else "VISIBLE_BOUNDARY" for value in H])
    return np.zeros((480, 640, 3), np.uint8), q, K, initial, evidence


def engine(evidence, action=None):
    return BoundaryCurriculum(ReviewObserver(evidence, action), dict(
        visible_threshold=.2, geometric_outline_priority=True, enable_unknown=False))


@pytest.mark.parametrize("action", ["mutate_image_in_place", "mutate_raw_in_place", "mutate_K_in_place", "mutate_initial_in_place"])
def test_provider_cannot_mutate_the_reference_that_wrapper_validates(action):
    image, q, K, initial, evidence = review_inputs()
    snapshots = image.copy(), q.copy(), K.copy(), deepcopy(initial)
    with pytest.raises(ValueError):
        engine(evidence, action).refine(image, q, K, initial, [1.1, .15, 1.3])
    np.testing.assert_array_equal(image, snapshots[0])
    np.testing.assert_array_equal(q, snapshots[1])
    np.testing.assert_array_equal(K, snapshots[2])
    assert initial == snapshots[3]


@pytest.mark.parametrize("action", ["exceed_fused_bound", "leave_image", "invalid_measurement"])
def test_provider_measurement_must_be_bounded_finite_and_inside_image(action):
    image, q, K, initial, evidence = review_inputs()
    with pytest.raises(ValueError):
        engine(evidence, action).refine(image, q, K, initial, [1.1, .15, 1.3])


@pytest.mark.parametrize("center", [False, True])
def test_nonidentity_physical_basis_asymmetric_model_and_readonly_inputs(center):
    image, q, K, initial, evidence = review_inputs(center)
    snapshots = image.copy(), q.copy(), K.copy(), deepcopy(initial)
    for value in (image, q, K):
        value.flags.writeable = False
    result = engine(evidence).refine(image, q, K, initial, [1.1, .15, 1.3])
    assert result["solution"]["accepted"]
    Q0 = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
    Rcf, Rphysical = map(np.asarray, (result["actual_pose"]["R_cf"], result["actual_pose"]["R_physical"]))
    np.testing.assert_allclose(Rcf.T @ Rphysical, Q0, rtol=0, atol=1e-12)
    assert result["actual_pose"]["cf_extents"] == [1.3, .15, 1.1]
    assert result["registered_dimensions"] == [1.1, .15, 1.3]
    assert len(result["observed_points"]) == (9 if center else 8)
    H = np.asarray(result["self_hidden"], bool)
    np.testing.assert_array_equal(np.asarray(result["corrected_measurements"])[:8][H], q[:8][H])
    for value, before in zip((image, q, K), snapshots[:3]):
        np.testing.assert_array_equal(value, before)
        assert not value.flags.writeable
    assert initial == snapshots[3]
    if center:
        assert result["completed_points"][8] == q[8].tolist()


def test_missing_and_outside_queries_remain_raw_and_do_not_enter_any_fit():
    image, q, K, initial, evidence = review_inputs()
    H = C.self_hidden(C.cuboid(*initial["cf_extents"]), initial["R_cf"], initial["centroid"])
    valid = np.flatnonzero(~H)
    q[valid[0]] = [-1., -1.]
    q[valid[1]] = [np.nan, np.nan]
    q[valid[2]] = [640., 100.]
    q[valid[3]] = [-2., 100.]
    excluded = set(valid[:4]) | set(np.flatnonzero(H))
    result = engine(evidence).refine(image, q, K, initial, [1.1, .15, 1.3])
    for stage in result["solution"]["stages"]:
        assert not excluded.intersection(stage["used_ids"])
    for key in valid[:4]:
        assert result["corrected_measurements"][key] == result["observed_points"][key]
    assert not result["solution"]["accepted"]
    assert result["actual_pose"] == initial
    assert result["completed_points"] == result["observed_points"]
