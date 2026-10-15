"""Independent scalar checks for v29; no consumer/producer/fitter imports.

Inputs are already parsed immutable image observations. This module does not
select observations, fit poses, read references or certify their provenance.
It recomputes energy and original-support geometry from projected vertices.
"""
from __future__ import annotations

import numpy as np


def radial_energy(residual, weight, delta):
    r = np.asarray(residual, float)
    z = float(r @ r)
    if not np.isfinite(z) or weight < 0 or delta <= 0:
        raise ValueError('Finite residual and nonnegative robust weight required')
    # Equivalent stable pseudo-Huber expression, independent of runtime code.
    return float(2. * weight * z / (np.sqrt(1. + z / delta**2) + 1.))


def point_energy(projected_xy, measurement, delta):
    k = measurement['native_corner']
    error = np.asarray(projected_xy[k]) - np.asarray(measurement['xy'])
    covariance = np.asarray(measurement['covariance_px2'])
    residual = np.linalg.solve(np.linalg.cholesky(covariance), error)
    return radial_energy(residual, float(measurement['fit_reliability_weight']), delta)


def legacy_normal_energy(projected_xy, boundary_record, delta):
    point = np.asarray(projected_xy[boundary_record['native_corner']])
    residual = np.asarray([(np.asarray(line['normal_unit_xy']) @ point + line['offset_px'])
                           / np.sqrt(line['normal_variance_px2'])
                           for line in boundary_record['lines']])
    return radial_energy(residual, float(boundary_record['fit_reliability_weight']), delta)


def tangent_energy(projected_xy, k, original_xy, tangent, variance, weight, delta):
    residual = float(np.asarray(tangent) @ (np.asarray(projected_xy[k])-np.asarray(original_xy)))
    return radial_energy(np.asarray([residual/np.sqrt(variance)]), weight, delta)


def actual_line_energy(projected_xy, edge, delta):
    """Recompute offset+sine covariance energy without the runtime EDGE API."""
    a, b = (np.asarray(projected_xy[edge[key]], float)
            for key in ('native_corner', 'native_neighbor'))
    direction = b-a
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError('Noncollapsed projected physical edge required')
    tangent = direction/length
    normal = np.asarray([-tangent[1], tangent[0]])
    observed = np.asarray(edge['normal_unit_xy'], float).copy()
    if observed[np.argmax(np.abs(observed))] < 0:
        observed *= -1.
    anchor = np.asarray(edge['support_anchor_xy'])
    offset_residual = float(normal @ (anchor-a))
    sine_residual = float(observed[0]*normal[1]-observed[1]*normal[0])
    vn, va = edge['normal_variance_px2'], edge['angular_variance_rad2']
    cross = float(edge.get('normal_angle_correlation', 0.))*np.sqrt(vn*va)
    covariance = np.asarray([[vn, cross], [cross, va]], float)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), [offset_residual, sine_residual])
    return dict(energy=radial_energy(whitened, float(edge['fit_reliability_weight']), delta),
                normal_residual_px=offset_residual, angular_sine_residual=sine_residual,
                projected_edge_length_px=length)


def original_support_slacks(projected_xy, group, *, margin=11., side_margin=1., fraction=.6):
    """Fixed observed endpoint order, with no direction inferred from a pose."""
    k, mate = group['native_corner'], group['native_neighbor']
    normal = np.asarray(group['normal_unit_xy'], float)
    tangent = np.asarray([-normal[1], normal[0]])
    actual = np.asarray(group['actual_inlier_xy']) @ tangent
    q = np.asarray(projected_xy)
    projected = q[[k, mate]] @ tangent
    coverage = [float(actual.min()-projected.min()+margin),
                float(projected.max()-actual.max()+margin)]
    directed = []
    for owner in group['activation_owners']:
        corner = owner['native_corner']
        sign = owner.get('fixed_tangent_sign')
        if sign is None:
            directed.append(dict(native_corner=corner, ambiguous=True, slacks=[]))
            continue
        if sign not in (-1., 1.):
            raise ValueError('A fixed image-evidence tangent sign must be +1 or -1')
        other = mate if corner == k else k
        distances = sign*(actual-float(tangent@q[corner]))
        # Directly count the required fraction, independently of runtime's
        # quantile implementation and its Jacobian branch.
        required = int(np.ceil(fraction*len(actual)-1e-12))
        sorted_distances = np.sort(distances)
        support = float(sorted_distances[-required]+side_margin)
        order = float(sign*tangent@(q[other]-q[corner]))
        directed.append(dict(native_corner=corner, ambiguous=False,
                             supported_fraction=float(np.mean(distances >= -side_margin)),
                             slacks=[support, order]))
    return dict(coverage_slacks=coverage, directed=directed,
                uses_original_support=True, sign_from_pose=False)
