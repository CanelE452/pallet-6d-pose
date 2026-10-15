from copy import deepcopy
import numpy as np
import pytest

from . import common as C
from . import projected_native_edge_factor_v4_17 as E


XYZ = C.cuboid(.8, .14, .59)
K = np.array([[610., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
SCALE = float(np.linalg.norm(np.ptp(XYZ, axis=0)))


def _edge(**updates):
    value = dict(native_corner=0, native_neighbor=1,
        normal_unit_xy=[.1, np.sqrt(.99)], offset_px=-245.,
        normal_variance_px2=4., angular_variance_rad2=.02**2,
        normal_angle_correlation=.3)
    normal = np.asarray(value['normal_unit_xy'])
    value['support_anchor_xy'] = (np.array([-normal[1], normal[0]]) * -300. + normal * 245.).tolist()
    value.update(updates)
    return value


def test_analytic_line_residual_and_robust_gradient_in_several_rotation_charts():
    for phi in ([.2, -.3, .1], [1e-10, -1e-9, 1e-8], [3., .2, .1], [-6.2, .1, .1]):
        p = np.r_[phi, [.15, -.2, 4.]]
        r, jacobian, _ = E.projected_edge_residual_and_jacobian(p, XYZ, K, SCALE, _edge())
        _, gradient, _ = E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, _edge(), weight=.7)
        numeric_jacobian = np.zeros((2, 6)); numeric_gradient = np.zeros(6)
        for j in range(6):
            step = np.zeros(6); step[j] = 1e-6
            numeric_jacobian[:, j] = (E.projected_edge_residual_and_jacobian(p+step, XYZ, K, SCALE, _edge())[0]
                - E.projected_edge_residual_and_jacobian(p-step, XYZ, K, SCALE, _edge())[0]) / 2e-6
            numeric_gradient[j] = (E.projected_edge_value_and_gradient(p+step, XYZ, K, SCALE, _edge(), weight=.7)[0]
                - E.projected_edge_value_and_gradient(p-step, XYZ, K, SCALE, _edge(), weight=.7)[0]) / 2e-6
        assert np.allclose(jacobian, numeric_jacobian, rtol=2e-5, atol=2e-5)
        assert np.allclose(gradient, numeric_gradient, rtol=3e-5, atol=3e-5)


def test_unoriented_cost_is_invariant_to_endpoint_and_observed_line_sign():
    p = np.r_[[.2, -.3, .1], [.15, -.2, 4.]]; edge = _edge(); original = deepcopy(edge)
    value, gradient, _ = E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, edge)
    for changed in [dict(native_corner=1, native_neighbor=0),
                    dict(normal_unit_xy=(-np.asarray(edge['normal_unit_xy'])).tolist(), offset_px=245.)]:
        candidate = {**edge, **changed}
        other, derivative, _ = E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, candidate)
        assert np.isclose(value, other, atol=1e-9)
        assert np.allclose(gradient, derivative, rtol=1e-10, atol=1e-9)
    assert edge == original


def test_one_measured_line_does_not_localize_a_corner_along_it():
    # Zero-thickness line test geometry with endpoints on x axis at equal z.
    xyz = XYZ.copy(); xyz[0] = [-.2, 0., 0.]; xyz[1] = [.2, 0., 0.]
    edge = _edge(normal_unit_xy=[0., 1.], offset_px=-240., support_anchor_xy=[320., 240.])
    for shift in (-.3, 0., .7):
        p = np.r_[[0., 0., 0.], [shift, 0., 4.]]
        value, gradient, audit = E.projected_edge_value_and_gradient(p, xyz, K, SCALE, edge)
        assert abs(value) < 1e-12 and np.linalg.norm(gradient) < 1e-8
        assert not audit['support_anchor_is_precise_keypoint']


def test_positive_noise_and_valid_projection_are_required_and_off_is_exact():
    p = np.r_[[.2, -.3, .1], [.15, -.2, 4.]]
    with pytest.raises(ValueError):
        E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, _edge(angular_variance_rad2=0.))
    with pytest.raises(ValueError):
        E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, _edge(normal_angle_correlation=1.))
    with pytest.raises(E.InvalidProjectedEdge):
        E.projected_edge_value_and_gradient(np.r_[[0., 0., 0.], [0., 0., -4.]], XYZ, K, SCALE, _edge())
    value, gradient, audit = E.projected_edge_value_and_gradient(p, None, None, None, None, weight=0.)
    assert value == 0. and np.array_equal(gradient, np.zeros(6)) and not audit['enabled']
