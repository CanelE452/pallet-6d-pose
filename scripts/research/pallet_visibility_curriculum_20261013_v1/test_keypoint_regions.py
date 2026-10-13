"""Synthetic geometry/provenance canaries; no datasets, labels or models."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import keypoint_regions as R


def fixture():
    xyz = np.array([[x, y, z] for z in (-.1, .1) for y in (-.2, .2) for x in (-.3, .3)])
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    t = np.array([0., 0., 2.])
    camera = (xyz+t) @ K.T
    q = np.vstack([camera[:, :2]/camera[:, 2:], [50., 50.]])
    image = np.zeros((100, 100, 3), np.uint8)
    return xyz, K, q, image, dict(name="BASE", R=np.eye(3).tolist(), t=t.tolist())


def factory(q, image, K, xyz, **kw):
    return R.build_keypoint_regions(q, image, scales_px=kw.pop("scales_px", 5.), K=K, xyz=xyz, **kw)


def test_circle_inside_and_exact_pixel_and_normalized_outside_losses():
    xyz, K, q, image, pose = fixture()
    regions = factory(q, image, K, xyz, native_ids=[0])
    moved = deepcopy(pose)
    moved.update(name="MOVED", t=[.19, 0., 2.])  # exactly ten native pixels at point0
    result = R.evaluate_pose_regions([pose, moved], xyz, K, regions, raw_points=q,
                                     image_sha256=R.image_pixel_sha256(image))
    assert [r["name"] for r in result["candidates"]] == ["BASE", "MOVED"]
    a, b = result["candidates"]
    assert a["normalized_penalty"] == a["pixel_penalty_px2"] == 0.
    assert b["normalized_penalty"] == pytest.approx(1.)
    assert b["pixel_penalty_px2"] == pytest.approx(25.)
    assert b["hard_valid"] is True
    assert b["independent_precise_support_count"] == 0
    assert result["image_regions_statistically_independent"] is False


def test_ellipse_euclidean_gap_is_not_radial_distance_approximation():
    pixel, norm = R._distance(np.array([8., 8.]), np.array([4., 2.]), "ellipse")
    assert norm == pytest.approx((np.sqrt(20)-1)**2)
    angle = np.linspace(0., 2*np.pi, 200000)
    dense = np.stack([4*np.cos(angle), 2*np.sin(angle)], axis=1)
    oracle = np.min(np.sum((dense-[8., 8.])**2, axis=1))
    assert pixel == pytest.approx(oracle, abs=1e-7)


def test_axis_aligned_box_distance_and_native_weights():
    xyz, K, q, image, pose = fixture()
    regions = factory(q, image, K, xyz, native_ids=[0, 1], scales_px=np.tile([4., 2.], (8, 1)),
                      shape="axis_aligned_box")
    moved = deepcopy(pose)
    moved["t"] = [.19, .095, 2.]
    answer = R.evaluate_pose_regions([moved], xyz, K, regions, weights={0: 0., 1: 2.})["candidates"][0]
    assert answer["pixel_penalty_px2"] == pytest.approx(36.+9.)
    assert answer["normalized_penalty"] == pytest.approx(2.25+2.25)
    assert answer["image_region_count"] == 1


def test_pose_prior_cannot_rank_prune_or_increase_support():
    xyz, K, q, image, pose = fixture()
    regions = R.build_pose_prior_regions(q, scales_px=.01, K=K, xyz=xyz,
                                        pose_reference={"initial": True})
    moved = deepcopy(pose)
    moved["t"] = [.5, 0., 2.]
    answer = R.evaluate_pose_regions([moved], xyz, K, regions)["candidates"][0]
    assert answer["normalized_penalty"] == answer["pixel_penalty_px2"] == 0.
    assert answer["image_region_count"] == answer["independent_precise_support_count"] == 0
    assert answer["hard_valid"] is True
    assert answer["regions"][0]["inside"] is False
    assert answer["regions"][0]["contributes_to_penalty"] is False


def test_missing_nonfinite_and_centre_are_not_region_correspondences():
    xyz, K, q, image, pose = fixture()
    q[0] = [-1., -1.]
    q[1] = [np.nan, 10.]
    q[2] = [101., 30.]
    regions = factory(q, image, K, xyz)
    assert [r["native_corner"] for r in regions] == list(range(2, 8))
    assert regions[0]["raw_query_out_of_frame"] is True
    assert regions[0]["physical_truncation_certified"] is False
    with pytest.raises(ValueError, match="native_ids"):
        factory(q, image, K, xyz, native_ids=[8])


@pytest.mark.parametrize("change", ["centre", "native_order", "image", "K", "xyz", "record"])
def test_native_image_camera_and_geometry_tamper_rejected(change):
    xyz, K, q, image, pose = fixture()
    regions = factory(q, image, K, xyz)
    raw, model, camera = q.copy(), xyz.copy(), K.copy()
    image_sha = R.image_pixel_sha256(image)
    if change == "centre":
        raw[8, 0] += 1.
    elif change == "native_order":
        raw[[0, 1]] = raw[[1, 0]]
    elif change == "image":
        image_sha = "0"*64
    elif change == "K":
        camera[0, 0] += 1.
    elif change == "xyz":
        model[[0, 1]] = model[[1, 0]]
    else:
        regions[0]["half_axes_px"][0] += 1.
    with pytest.raises(ValueError, match="binding mismatch"):
        R.evaluate_pose_regions([pose], model, camera, regions, raw_points=raw, image_sha256=image_sha)


def test_cached_file_binding_is_explicit_and_not_pixel_verification():
    xyz, K, q, image, pose = fixture()
    regions = R.build_keypoint_regions(q, scales_px=np.full(8, 5.), image_sha256="a"*64,
                                      image_hw=[100, 100], K=K, xyz=xyz)
    assert regions[0]["image_binding"]["kind"] == "SEALED_FILE_SHA256"
    assert regions[0]["image_binding"]["verified_pixels"] is False
    assert R.evaluate_pose_regions([pose], xyz, K, regions, image_sha256="a"*64)["candidates"][0]["normalized_penalty"] == 0.
    with pytest.raises(ValueError, match="decoded BGR"):
        factory(q, image, K, xyz, image_sha256="a"*64)
    with pytest.raises(ValueError, match="integer"):
        R.build_keypoint_regions(q, scales_px=5., image_sha256="a"*64, image_hw=[100.5, 100])


def test_regions_from_different_frames_cannot_form_a_shared_candidate_score():
    xyz, K, q, image, pose = fixture()
    first = factory(q, image, K, xyz, native_ids=[0])
    image[0, 0] = [1, 2, 3]
    second = factory(q, image, K, xyz, native_ids=[1])
    with pytest.raises(ValueError, match="mixed frame/query"):
        R.evaluate_pose_regions([pose], xyz, K, first+second)


def test_hard_empirical_confidence_is_rejected_but_declared_joint_contract_can_prune():
    xyz, K, q, image, pose = fixture()
    with pytest.raises(ValueError, match="do not authorize"):
        factory(q, image, K, xyz, hard_policy={"kind": "EMPIRICALLY_CALIBRATED_EXPLORATORY", "coverage": .9})
    contract = dict(kind="CALIBRATED_JOINT_COVERAGE", coverage=.9, joint_coverage=True,
                    exchangeability_assumed=True, calibration_model_sha256="b"*64,
                    calibration_reference="separate held-out calibration", assumptions_reference="exchangeable deployment")
    regions = factory(q, image, K, xyz, hard_policy=contract, calibration_model_sha256="b"*64)
    moved = deepcopy(pose)
    moved["t"] = [.5, 0., 2.]
    answer = R.evaluate_pose_regions([pose, moved], xyz, K, regions)
    assert [r["hard_valid"] for r in answer["candidates"]] == [True, False]
    assert answer["hard_policy_contract_verified_statistically"] is False
    assert answer["finite_pool_contains_true_pose_guaranteed"] is False


def test_mask_alone_or_pose_projection_cannot_authorize_semantic_hard_region():
    xyz, K, q, image, pose = fixture()
    contract = dict(kind="INDEPENDENTLY_SPECIFIED_SEMANTIC_REGION", ownership_verified=True,
                    region_geometry_verified=False, independent_of_initial_projection=True,
                    source_reference="occluder mask")
    with pytest.raises(ValueError, match="mask alone"):
        factory(q, image, K, xyz, hard_policy=contract)
    contract.update(region_geometry_verified=True, independent_of_initial_projection=False)
    with pytest.raises(ValueError, match="independent geometry"):
        factory(q, image, K, xyz, hard_policy=contract)


def test_inputs_immutable_json_output_zero_weights_and_pose_aliases():
    xyz, K, q, image, pose = fixture()
    before = [a.copy() for a in (xyz, K, q, image)]
    regions = factory(q, image, K, xyz)
    records_before = deepcopy(regions)
    alias = dict(name="ALIAS", R_cf=pose["R"], centroid=pose["t"])
    answer = R.evaluate_pose_regions([alias], xyz, K, regions, weights=np.zeros(8), raw_points=q)
    assert answer["candidates"][0]["normalized_penalty"] == 0.
    assert answer["candidates"][0]["image_region_weight_sum"] == 0.
    json.dumps(answer, allow_nan=False)
    json.dumps(regions, allow_nan=False)
    for a, b in zip((xyz, K, q, image), before):
        np.testing.assert_array_equal(a, b)
    assert regions == records_before


@pytest.mark.parametrize("value", [0., -1., np.nan, [1., 2.]])
def test_invalid_scales_fail_closed(value):
    xyz, K, q, image, pose = fixture()
    with pytest.raises(ValueError, match="scales_px"):
        factory(q, image, K, xyz, scales_px=value)


def test_duplicate_regions_do_not_multiply_shared_evidence_and_invalid_poses_rejected():
    xyz, K, q, image, pose = fixture()
    regions = factory(q, image, K, xyz, native_ids=[0])
    with pytest.raises(ValueError, match="multiply evidence"):
        R.evaluate_pose_regions([pose], xyz, K, regions+regions)
    invalid = deepcopy(pose)
    invalid["R"][0][0] = -1.
    with pytest.raises(ValueError, match="proper"):
        R.evaluate_pose_regions([invalid], xyz, K, regions)
