"""Physical-basis, observation preservation and proposal input contracts."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from .anchored_completion_api import AnchoredCompletion
from .protected_completion_api import ProtectedCompletion, pose_proposal
from .test_boundary_rule_api import Provider, inputs


def test_physical_conversion_preserves_a_general_proper_model_basis():
    R0 = cv2.Rodrigues(np.array([.13, -.27, .19]))[0]
    Q = cv2.Rodrigues(np.array([-.31, .07, .11]))[0]
    R1 = cv2.Rodrigues(np.array([.15, -.21, .22]))[0]
    initial = dict(R_cf=R0.tolist(), R_physical=(R0 @ Q).tolist(),
                   cf_extents=[1.3, .15, 1.1], centroid=[0., 0., 3.])
    candidate = dict(initial, R_cf=R1.tolist(), R_physical=(R1 @ Q).tolist())
    proposal = pose_proposal("other", candidate, initial)
    np.testing.assert_allclose(proposal["R"], R1, atol=1e-14, rtol=0)
    assert proposal["native_order_preserved"]


@pytest.mark.parametrize("changed", ["dimensions", "basis"])
def test_alternate_families_or_native_correspondences_are_rejected(changed):
    _, _, _, initial, _ = inputs(3)
    candidate = deepcopy(initial)
    if changed == "dimensions":
        candidate["cf_extents"] = list(reversed(initial["cf_extents"]))
    else:
        candidate["R_physical"] = np.eye(3).tolist()
    with pytest.raises(ValueError):
        pose_proposal("unregistered", candidate, initial)


def test_state_readout_retains_image_anchors_and_separate_rigid_model():
    image, q, K, initial, observation = inputs(3)
    original = deepcopy(observation)
    output = ProtectedCompletion(AnchoredCompletion(Provider(observation),
                                                  {"supplement": False})).refine(
        image, q, K, initial, [1.3, .15, 1.1])
    for k in output["anchor_ids"]:
        np.testing.assert_array_equal(output["reported_points"][k], output["measured_points"][k])
    assert output["completed_points"][8] == q[8].tolist()
    assert output["provenance"]["reported_points_are_one_rigid_pose"] is False
    assert output["provenance"]["additional_proposals_are_measurements"] is False
    assert observation == original


def test_external_proposal_provider_cannot_change_original_queries():
    image, q, K, initial, observation = inputs(3)
    def mutate(image, query, camera, pose, dimensions, base):
        query[0] += [1., 0.]
        return []
    engine = ProtectedCompletion(AnchoredCompletion(Provider(observation), {"supplement": False}),
                                 proposal_provider=mutate)
    original = q.copy()
    with pytest.raises(ValueError, match="immutable inputs"):
        engine.refine(image, q, K, initial, [1.3, .15, 1.1])
    np.testing.assert_array_equal(q, original)
