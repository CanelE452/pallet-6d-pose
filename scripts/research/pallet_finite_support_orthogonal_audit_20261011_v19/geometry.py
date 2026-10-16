"""Observed-line-frame containment, with one block per actual semantic edge.

This replaces the projected-tangent containment surrogate only. It uses no
inferred 3D query ID or fabricated image point. This is a new objective;
historical V13/V16 residuals and failed checks remain historical evidence.
"""
import math


def observed_frame_block(endpoints, endpoint_jacobian, normal, offset, support):
    """Two endpoint normal errors plus support outside the observed interval.

    The tangent is perpendicular to the *observed* fixed normal. Endpoint
    order may reverse. At an endpoint or interval-order cusp the selected
    one-sided derivative is recorded; it is not a smooth rank certificate.
    """
    if len(endpoints) != 2 or any(len(p) != 2 for p in endpoints):
        raise ValueError('two 2D projected endpoints required')
    if len(support) < 3 or any(len(p) != 2 for p in support):
        raise ValueError('at least three actual 2D support points required')
    if len(normal) != 2 or any(not math.isfinite(v) for v in normal) or abs(math.hypot(*normal) - 1.) > 1e-10:
        raise ValueError('fixed unit observed normal required')
    if any(not math.isfinite(v) for p in endpoints + support for v in p):
        raise ValueError('finite observed/projected coordinates required')
    if not math.isfinite(offset):
        raise ValueError('finite observed offset required')
    if len(endpoint_jacobian) != 2 or any(len(p) != 2 for p in endpoint_jacobian):
        raise ValueError('two endpoint Jacobians required')
    width = len(endpoint_jacobian[0][0])
    if width == 0 or any(len(r) != width or any(not math.isfinite(v) for v in r)
                         for p in endpoint_jacobian for r in p):
        raise ValueError('finite consistent endpoint Jacobians required')
    dot = lambda a, b: math.fsum(x*y for x, y in zip(a, b))
    tangent = [-normal[1], normal[0]]
    endpoint_u = [dot(p, tangent) for p in endpoints]
    low, high = sorted(range(2), key=lambda k: (endpoint_u[k], k))
    signed = [dot(p, normal) - offset for p in endpoints]
    residual = [v/math.sqrt(2.) for v in signed]
    matrix = [[dot(normal, [p[i][j] for i in range(2)])/math.sqrt(2.)
               for j in range(width)] for p in endpoint_jacobian]
    endpoint_du = [[dot(tangent, [p[i][j] for i in range(2)])
                    for j in range(width)] for p in endpoint_jacobian]
    support_u = [dot(p, tangent) for p in support]
    branches, hinges = [], []
    for u in support_u:
        if u < endpoint_u[low]:
            value, row, branch = u-endpoint_u[low], [-v for v in endpoint_du[low]], 'BEFORE_OBSERVED_INTERVAL'
        elif u > endpoint_u[high]:
            value, row, branch = u-endpoint_u[high], [-v for v in endpoint_du[high]], 'AFTER_OBSERVED_INTERVAL'
        else:
            value, row, branch = 0., [0.]*width, 'INSIDE_OBSERVED_INTERVAL'
        hinges.append(value); branches.append(branch)
        residual.append(value/math.sqrt(len(support)))
        matrix.append([v/math.sqrt(len(support)) for v in row])
    return residual, matrix, dict(
        observed_tangent=tangent, endpoint_u=endpoint_u,
        interval_endpoint_indices=[low, high],
        interval_order_cusp=endpoint_u[low] == endpoint_u[high],
        support_u=support_u, outside_hinges_px=hinges,
        normal_signed_px=signed, hinge_branches=branches,
        endpoint_hit_cusps=[u in endpoint_u for u in support_u],
        block_factor_count=1, new_independent_point_correspondences=0,
        physical_semantic_line_correctness_certified=False,
        coordinate_frame='FIXED_OBSERVED_LINE',
        historical_projected_tangent_objective_reused=False)
