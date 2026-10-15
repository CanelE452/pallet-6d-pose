"""Containment, missing silhouette support and exact piecewise derivatives."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import global_foreground_containment_v4_25 as FACTOR
from . import global_outline_analytic_factor_v4_11 as OLD
from . import joint_ot_jacobian_v2 as J
from . import statewise_global_outline_solver_v4_10 as OBS
from .common import cuboid
from .joint_ot_pnp import _pose_vector
from .test_global_outline_analytic_factor_v4_11 import scene


def square_scene():
    xyz = cuboid(2., 2., 2.)
    K = np.array([[400., 0., 200.], [0., 400., 200.], [0., 0., 1.]])
    scale = np.linalg.norm(np.ptp(xyz, axis=0))
    p = _pose_vector(np.eye(3), np.array([0., 0., 5.]), scale)
    # The nearest full model face projects exactly to [100,300]^2.
    return xyz, K, scale, p


@pytest.mark.parametrize('kind', ['full', 'subset', 'hole', 'occluded_notch', 'disconnected'])
def test_foreground_subsets_and_their_virtual_hull_bridges_have_zero_cost(kind):
    xyz, K, scale, p = square_scene()
    mask = np.zeros((400, 400), np.uint8)
    mask[100:301, 100:301] = 1
    if kind == 'subset':
        mask[:] = 0
        mask[150:251, 150:251] = 1
    elif kind == 'hole':
        mask[140:261, 140:261] = 0
    elif kind == 'occluded_notch':
        mask[145:256, 100:171] = 0
    elif kind == 'disconnected':
        mask[:] = 0
        mask[125:161, 125:161] = 1
        mask[240:276, 240:276] = 1
    observation = OBS.observe_global_outline(mask, mask.shape)
    value, gradient, audit = FACTOR.outline_value_and_gradient(p, xyz, K, scale, observation)
    assert value == 0.
    np.testing.assert_array_equal(gradient, np.zeros(6))
    assert audit['outside_sample_count'] == 0
    assert audit['model_to_observed_factor_weight'] == 0.
    assert audit['observed_to_model_factor_weight'] == .5


def test_no_model_to_observed_term_and_oversized_model_is_not_unique():
    xyz, K, scale, p = square_scene()
    mask = np.zeros((400, 400), np.uint8)
    mask[150:251, 150:251] = 1
    observation = OBS.observe_global_outline(mask, mask.shape)
    original, _, _ = OLD.outline_value_and_gradient(p, xyz, K, scale, observation)
    assert original > 0.  # Frozen full-outline forward coverage would pull inward.
    for geometry in (xyz, 1.25 * xyz):
        value, gradient, audit = FACTOR.outline_value_and_gradient(p, geometry, K, scale, observation)
        assert value == 0.
        np.testing.assert_array_equal(gradient, np.zeros(6))
        assert audit['model_expansion_or_depth_uniqueness_proved'] is False


def central_gradient(p, xyz, K, scale, observation):
    step = 1e-7
    return np.array([
        (FACTOR.outline_value_and_gradient(p + step * axis, xyz, K, scale, observation)[0]
         - FACTOR.outline_value_and_gradient(p - step * axis, xyz, K, scale, observation)[0]) / (2 * step)
        for axis in np.eye(6)])


@pytest.mark.parametrize('translation,rotation', [
    ([0., .3, 2.2], [.12, .65, .07]),
    ([-.9, .3, 2.2], [.12, .65, .07]),
    ([0., -.55, 2.2], [3.35, .4, -.06]),
])
def test_full_camera_all_pose_derivatives_with_frozen_and_clipped_observation(translation, rotation):
    xyz, K, scale, p, observation = scene(translation, rotation)
    p = p + np.array([.013, -.017, .011, .009, -.007, .019])
    value, gradient, audit = FACTOR.outline_value_and_gradient(p, xyz, K, scale, observation)
    assert value > 0.
    assert audit['outside_sample_count'] > 0
    np.testing.assert_allclose(gradient, central_gradient(p, xyz, K, scale, observation),
                               rtol=2e-5, atol=3e-5)
    points = np.asarray(observation['observed_boundary_samples_xy'])
    h, w = observation['image_hw']
    assert np.min(points) >= 7.
    assert np.max(points[:, 0]) <= w - 8.
    assert np.max(points[:, 1]) <= h - 8.


def test_vertex_nearest_distance_and_exact_old_reverse_coefficient():
    xyz, K, scale, p = square_scene()
    observation = dict(supported=True, observed_boundary_samples_xy=[[80., 85.], [75., 90.]])
    q, _, _, _ = J.projection_and_jacobian(p, xyz, K, scale)
    indices = OLD.ConvexHull(q).vertices
    polygon = q[indices]
    _, _, alpha = OLD._nearest(np.asarray(observation['observed_boundary_samples_xy']),
                               np.stack([polygon, np.roll(polygon, -1, axis=0)], axis=1))
    assert np.all((alpha == 0.) | (alpha == 1.))
    value, gradient, audit = FACTOR.outline_value_and_gradient(p, xyz, K, scale, observation)
    residual = np.array([[-20., -15.], [-25., -10.]])
    expected = .5 * OLD._loss_and_force(residual, 1., 1.25)[0].mean()
    assert value == pytest.approx(expected, abs=1e-13)
    assert audit['outside_sample_count'] == 2
    np.testing.assert_allclose(gradient, central_gradient(p, xyz, K, scale, observation),
                               rtol=2e-6, atol=3e-6)


def test_inside_zero_samples_stay_in_mean_denominator_and_only_frozen_samples_are_read():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Unrelated state/raw/observed polygon must not be accessed')

    xyz, K, scale, p = square_scene()
    outside = dict(supported=True, observed_boundary_samples_xy=[[80., 85.]])
    one = FACTOR.outline_value_and_gradient(p, xyz, K, scale, outside)
    mixed = dict(supported=True, observed_boundary_samples_xy=[[80., 85.], [200., 200.]],
                 original_raw_points=Poison(), states=Poison(),
                 polygon_xy=Poison(), observed_boundary_segments_xy=Poison())
    value, gradient, audit = FACTOR.outline_value_and_gradient(p, xyz, K, scale, mixed)
    assert value == pytest.approx(one[0] / 2.)
    np.testing.assert_allclose(gradient, one[1] / 2., rtol=0, atol=0)
    assert audit['inside_or_boundary_sample_count'] == 1
    assert audit['S_local_queries'] == audit['U_local_queries'] == audit['raw_coordinate_reads'] == 0


def test_unsupported_is_zero_without_projection_and_invalid_model_is_not_accepted(monkeypatch):
    xyz, K, scale, p = square_scene()
    with monkeypatch.context() as patch:
        patch.setattr(FACTOR.J, 'projection_and_jacobian',
                      lambda *args: (_ for _ in ()).throw(AssertionError('projection was called')))
        value, gradient, audit = FACTOR.outline_value_and_gradient(p, xyz, K, scale, {'supported': False})
        assert value == 0.
        np.testing.assert_array_equal(gradient, np.zeros(6))
        assert audit['enabled'] is False
    observation = dict(supported=True, observed_boundary_samples_xy=[[200., 200.]])
    bad = p.copy()
    bad[5] = 0.
    value, gradient, audit = FACTOR.outline_value_and_gradient(bad, xyz, K, scale, observation)
    assert value == 1e12 and audit['valid_projection'] is False
    np.testing.assert_array_equal(gradient, np.zeros(6))


def test_inputs_immutable_and_bad_observed_samples_rejected():
    xyz, K, scale, p, observation = scene([0., .3, 2.2], [.12, .65, .07])
    original = deepcopy(observation)
    arrays = [value.copy() for value in (xyz, K, p)]
    FACTOR.outline_value_and_gradient(p, xyz, K, scale, observation)
    assert observation == original
    for current, saved in zip((xyz, K, p), arrays):
        np.testing.assert_array_equal(current, saved)
    with pytest.raises(ValueError, match='frozen observed boundary samples'):
        FACTOR.outline_value_and_gradient(p, xyz, K, scale,
                                        dict(supported=True, observed_boundary_samples_xy=[[np.nan, 0.]]))
