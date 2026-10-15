"""Finite model support differs from an infinite-line fit."""
import numpy as np

from . import common as C
from . import finite_native_edge_support_v4_17 as F


def test_collinear_displaced_or_short_model_segment_cannot_cover_real_inlier_support():
    xyz = C.cuboid(.8, .14, .59)
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    scale = np.linalg.norm(np.ptp(xyz, axis=0))
    edges = [dict(native_corner=0, native_neighbor=1, normal_unit_xy=[0., 1.],
        actual_support_tangent_interval_px=[-350., -290.], support_extrapolation_px=11.)]
    p = np.r_[[0., 0., 0.], [0., 0., 4./scale]]
    slack, _, _ = F.finite_support_slacks_and_jacobian(p, xyz, K, scale, edges)
    assert min(slack) >= 0
    shifted = p.copy(); shifted[3] += .8 / scale
    displaced, _, _ = F.finite_support_slacks_and_jacobian(shifted, xyz, K, scale, edges)
    assert min(displaced) < 0
    # Both poses retain exactly the same horizontal supporting image line.
    deeper = p.copy(); deeper[5] = 20./scale
    short, _, _ = F.finite_support_slacks_and_jacobian(deeper, xyz, K, scale, edges)
    assert min(short) < 0


def test_finite_guard_analytic_derivative_and_endpoint_sign_invariance():
    xyz = C.cuboid(.8, .14, .59)
    K = np.array([[610., 13., 320.], [0., 605., 240.], [0., 0., 1.]])
    scale = np.linalg.norm(np.ptp(xyz, axis=0)); p = np.r_[[.2, -.3, .1], [.1, -.2, 4.]]
    edge = dict(native_corner=0, native_neighbor=1, normal_unit_xy=[0., 1.],
        actual_support_tangent_interval_px=[-350., -290.], support_extrapolation_px=11.)
    values, jac, _ = F.finite_support_slacks_and_jacobian(p, xyz, K, scale, [edge])
    numeric = F.N.V._jacobian(lambda pp: F.finite_support_slacks_and_jacobian(pp, xyz, K, scale, [edge])[0], p)
    assert np.allclose(jac, numeric, rtol=2e-6, atol=2e-5)
    reversed_edge = {**edge, 'native_corner':1, 'native_neighbor':0}
    other, derivative, _ = F.finite_support_slacks_and_jacobian(p, xyz, K, scale, [reversed_edge])
    assert np.array_equal(values, other) and np.array_equal(jac, derivative)
    sign_reversed = {**edge, 'normal_unit_xy':[0., -1.],
        'actual_support_tangent_interval_px':[290., 350.]}
    other, derivative, _ = F.finite_support_slacks_and_jacobian(p, xyz, K, scale, [sign_reversed])
    assert np.array_equal(values, other[::-1]) and np.array_equal(jac, derivative[::-1])
