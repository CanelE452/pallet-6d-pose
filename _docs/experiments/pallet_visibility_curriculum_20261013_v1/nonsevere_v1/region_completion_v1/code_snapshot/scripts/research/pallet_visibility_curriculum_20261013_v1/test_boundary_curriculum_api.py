"""Semantic checks for measured points, conditional H and exact fallback."""
from copy import deepcopy

import numpy as np
import pytest

from . import common as C
from .boundary_curriculum_api import BoundaryCurriculum
from .test_curriculum import fixture, appearance


class Observer:
    def __init__(self, initial, evidence, shift=True, tamper=None):
        self.initial = initial
        self.evidence = evidence
        self.shift = shift
        self.tamper = tamper

    def assess(self, image, raw, K, initial):
        H = C.self_hidden(C.cuboid(*initial["cf_extents"]), initial["R_cf"], initial["centroid"])
        corrected = raw.copy()
        if self.shift:
            corrected[0] -= [1., -.5]
        if self.tamper == "hidden":
            corrected[np.flatnonzero(H)[0]] += [1., 0.]
        elif self.tamper == "centre":
            corrected[8] += [1., 0.]
        elif self.tamper == "raw":
            raw = raw + [1., 0.]
        return dict(raw_points=raw.tolist(), fused_points=corrected.tolist(),
                    visibility_appearance=deepcopy(self.evidence), self_hidden=H.tolist(),
                    boundary_visibility={"states": ["NOT_ASSESSED"] * 8})


def inputs():
    xyz, K, R, t, R0, t0, q = fixture()
    initial = dict(available=True, R_cf=R0.tolist(), R_physical=R0.tolist(),
                   centroid=t0.tolist(), cf_extents=[1.1, .15, 1.1])
    H = C.self_hidden(xyz, R0, t0)
    q = np.vstack([q + [2., -1.], [321., 243.]])
    q[np.flatnonzero(H)[0]] += [80., -60.]
    states = ["SELF_OCCLUDED" if h else "VISIBLE_BOUNDARY" for h in H]
    evidence = appearance(states)
    image = np.zeros((480, 640, 3), np.uint8)
    return image, q, K, initial, evidence


def test_image_measurements_and_hidden_completion_remain_separate():
    image, q, K, initial, evidence = inputs()
    before = q.copy()
    engine = BoundaryCurriculum(Observer(initial, evidence), dict(
        visible_threshold=.2, geometric_outline_priority=True, enable_unknown=False))
    result = engine.refine(image, q, K, initial, [1.1, .15, 1.1])
    assert result["solution"]["accepted"]
    assert result["solution"]["stages"][0]["RGB_states_and_weights_preserved"]
    H = np.asarray(result["self_hidden"], bool)
    assert all(not H[k] for stage in result["solution"]["stages"] for k in stage["used_ids"])
    np.testing.assert_array_equal(result["observed_points"], before)
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(result["corrected_measurements"][8], before[8])
    np.testing.assert_array_equal(np.asarray(result["corrected_measurements"])[:8][H], before[:8][H])
    np.testing.assert_allclose(np.asarray(result["completed_points"])[:8][H], np.asarray(result["projected_points"])[H])
    np.testing.assert_array_equal(result["completed_points"][8], before[8])
    Q = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
    np.testing.assert_allclose(result["actual_pose"]["R_physical"], np.asarray(result["actual_pose"]["R_cf"]) @ Q)
    assert not result["provenance"]["projections_used_for_fit"]


def test_insufficient_visible_support_preserves_initial_and_raw_completion_exactly():
    image, q, K, initial, evidence = inputs()
    allowed = set(np.flatnonzero(~C.self_hidden(C.cuboid(1.1, .15, 1.1), initial["R_cf"], initial["centroid"]))[:3])
    evidence["states"] = ["VISIBLE_BOUNDARY" if k in allowed else "UNKNOWN" for k in range(8)]
    engine = BoundaryCurriculum(Observer(initial, evidence), dict(
        visible_threshold=.2, geometric_outline_priority=True, enable_unknown=False))
    result = engine.refine(image, q, K, initial, [1.1, .15, 1.1])
    assert not result["solution"]["accepted"]
    assert result["actual_pose"] == initial
    np.testing.assert_array_equal(result["completed_points"], q)
    assert result["solution"]["used_ids"] == []


@pytest.mark.parametrize("tamper", ["hidden", "centre", "raw"])
def test_reject_observation_provider_that_moves_excluded_points_or_original_queries(tamper):
    image, q, K, initial, evidence = inputs()
    engine = BoundaryCurriculum(Observer(initial, evidence, tamper=tamper))
    with pytest.raises(ValueError, match="Observation provider"):
        engine.refine(image, q, K, initial, [1.1, .15, 1.1])

