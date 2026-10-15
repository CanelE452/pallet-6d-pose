"""Two-coordinate image-line factor for a named native cuboid edge.

The old partial factor puts one projected vertex on a measured line.  This
factor instead compares the projected 3D edge's line with the measured RGB
line: offset at a saved support anchor, and unoriented angular discrepancy.
The anchor is a line statistic, never a measured keypoint.  Authentication,
S/U gates, deduplication and CAL-only noise fitting belong to the caller.

The angular residual is sin(theta), which is smooth for unoriented lines
and agrees with an angular residual locally.  Both residual coordinates
reverse sign together if the edge endpoints are reversed, so the complete
correlated quadratic/robust cost is unchanged.  A line cannot supply a
precise coordinate along itself, and this module never creates one.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .joint_ot_jacobian_v2 import projection_and_jacobian


class InvalidProjectedEdge(ValueError):
    """Nonpositive depth or a collapsed projected segment has no image line."""


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 8:
        raise ValueError(name + ' must be an integer native corner in [0,7]')
    return int(value)


def _parameters(edge):
    if not isinstance(edge, Mapping):
        raise ValueError('An authenticated edge mapping is required')
    first = _index(edge.get('native_corner'), 'native_corner')
    second = _index(edge.get('native_neighbor'), 'native_neighbor')
    if first == second:
        raise ValueError('Two distinct native endpoints are required')
    normal = np.asarray(edge['normal_unit_xy'], dtype=float)
    anchor = np.asarray(edge['support_anchor_xy'], dtype=float)
    offset = float(edge['offset_px'])
    if (normal.shape != (2,) or anchor.shape != (2,)
            or not np.isfinite(normal).all() or not np.isfinite(anchor).all()
            or not np.isfinite(offset) or not np.isclose(np.linalg.norm(normal), 1., atol=1e-7, rtol=0)):
        raise ValueError('Finite unit observed normal, offset and support anchor required')
    if abs(float(normal @ anchor) + offset) > 1e-6 * max(1., float(np.linalg.norm(anchor))):
        raise ValueError('Support anchor must lie on the saved measured line')
    # An observed line's algebraic sign is not physical information.
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    normal_variance = float(edge['normal_variance_px2'])
    angular_variance = float(edge['angular_variance_rad2'])
    correlation = float(edge.get('normal_angle_correlation', 0.))
    if (not np.isfinite([normal_variance, angular_variance, correlation]).all()
            or min(normal_variance, angular_variance) <= 0 or abs(correlation) >= 1. - 1e-8):
        raise ValueError('Positive fixed normal/angular variances and nonsingular correlation required')
    cross = correlation * np.sqrt(normal_variance * angular_variance)
    covariance = np.array([[normal_variance, cross], [cross, angular_variance]])
    return first, second, normal, anchor, covariance


def projected_edge_residual_and_jacobian(p, xyz, K, scale, edge):
    """Return whitened residual[2], analytic Jacobian[2,6], and an audit.

    ``angular_variance_rad2`` is a fixed small-angle engineering/CAL scale
    for the sine residual, not a coverage guarantee.  Zero TLS scatter is
    not a valid zero angular variance.  No noise term depends on the pose.
    """
    first, second, observed_normal, anchor, covariance = _parameters(edge)
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (8, 3) or not np.isfinite(xyz).all():
        raise ValueError('Finite registered native cuboid[8,3] required')
    q, depth, q_jacobian, _ = projection_and_jacobian(p, xyz[[first, second]], K, scale)
    if not np.isfinite(q).all() or np.min(depth) <= 1e-8:
        raise InvalidProjectedEdge('Positive finite projected edge endpoints required')
    direction = q[1] - q[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        raise InvalidProjectedEdge('Projected native edge collapsed')
    tangent = direction / length
    rotation90 = np.array([[0., -1.], [1., 0.]])
    normal = rotation90 @ tangent
    direction_jacobian = q_jacobian[1] - q_jacobian[0]
    normal_jacobian = rotation90 @ ((np.eye(2) - np.outer(tangent, tangent)) / length) @ direction_jacobian
    normal_error = float(normal @ (anchor - q[0]))
    angular_sine = float(np.cross(observed_normal, normal))
    raw = np.array([normal_error, angular_sine])
    raw_jacobian = np.vstack([
        (anchor - q[0]) @ normal_jacobian - normal @ q_jacobian[0],
        (rotation90 @ observed_normal) @ normal_jacobian,
    ])
    whitening = np.linalg.inv(np.linalg.cholesky(covariance))
    residual, jacobian = whitening @ raw, whitening @ raw_jacobian
    return residual, jacobian, dict(
        native_corner=first, native_neighbor=second,
        normal_residual_px=normal_error, angular_sine_residual=angular_sine,
        unoriented_angle_deg=float(np.degrees(np.arcsin(np.clip(abs(angular_sine), 0., 1.)))),
        projected_edge_length_px=length, projected_endpoints_xy=q.tolist(),
        support_anchor_is_precise_keypoint=False, observed_line_sign_is_not_information=True,
        covariance_px_and_small_angle_sine=covariance.tolist(),
        angular_residual_kind='sine of unoriented line angle',
        line_intercept_must_replace_old_normal_factor=True,
        S_U_image_queries=0, independent_measurement_or_coverage_claim=False,
    )


def projected_edge_value_and_gradient(p, xyz, K, scale, edge, *, weight=1., huber_scale=3.):
    """One radial robust factor per authenticated RGB edge, reliability once."""
    weight, delta = float(weight), float(huber_scale)
    if not np.isfinite([weight, delta]).all() or weight < 0 or delta <= 0:
        raise ValueError('Nonnegative reliability and positive robust scale required')
    if weight == 0:
        return 0., np.zeros(6), dict(enabled=False)
    residual, jacobian, audit = projected_edge_residual_and_jacobian(p, xyz, K, scale, edge)
    denominator = np.sqrt(1. + float(residual @ residual) / delta**2)
    value = weight * 2. * delta**2 * (denominator - 1.)
    gradient = (2. * weight * residual / denominator) @ jacobian
    return float(value), gradient, {**audit, 'enabled': True, 'weight_applied_once': weight,
        'whitened_line_residual': residual.tolist()}
