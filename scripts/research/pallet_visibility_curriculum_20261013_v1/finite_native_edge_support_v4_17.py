"""Opt-in finite image-support guards for already authenticated named edges.

An infinite image line cannot certify that the finite projected 3D edge
actually covers its observed inlier interval. These two inequalities express
that missing condition with the existing support-extrapolation margin. They
are conditional on a correct named-edge association, not a visibility or
confidence certificate. No point, image, label or new noise is inferred.
"""
from __future__ import annotations

import numpy as np

from . import native_edge_problem_v4_17 as N


def finite_support_slacks_and_jacobian(p, xyz, K, scale, edges):
    q, _, jac, _ = N.J.projection_and_jacobian(p, xyz, K, scale)
    values, rows, audits = [], [], []
    for edge in edges:
        k, mate = edge['native_corner'], edge['native_neighbor']
        normal = np.asarray(edge['normal_unit_xy'], float)
        tangent = np.array([-normal[1], normal[0]])
        support = np.asarray(edge['actual_support_tangent_interval_px'], float)
        margin = float(edge['support_extrapolation_px'])
        if (support.shape != (2,) or not np.isfinite(support).all() or support[1] <= support[0]
                or not np.isfinite(margin) or margin < 0):
            raise ValueError('Actual finite inlier support and original nonnegative margin required')
        endpoints = np.array([tangent @ q[k], tangent @ q[mate]])
        derivative = np.array([tangent @ jac[k], tangent @ jac[mate]])
        low, high = (0, 1) if endpoints[0] <= endpoints[1] else (1, 0)
        slack = np.array([support[0] + margin - endpoints[low],
                          endpoints[high] - support[1] + margin])
        values.extend(slack); rows.extend([-derivative[low], derivative[high]])
        audits.append(dict(native_corner=k, native_neighbor=mate,
            actual_inlier_support_tangent_interval_px=support.tolist(),
            projected_model_segment_tangent_interval_px=[float(endpoints[low]), float(endpoints[high])],
            original_support_extrapolation_px=margin, finite_support_slack_px=slack.tolist(),
            support_anchor_not_a_precise_keypoint=True,
            conditional_on_named_edge_association=True))
    return np.asarray(values), np.asarray(rows).reshape((-1, 6)), audits


class FiniteNativeEdgeProblem:
    """Compose the unchanged native problem with explicitly opt-in guards."""
    def __init__(self, native_problem):
        self.base = native_problem

    def __getattr__(self, name):
        return getattr(self.base, name)

    def constraints(self, p, sides):
        old, old_jacobian = self.base.constraints(p, sides)
        extra, derivative, _ = finite_support_slacks_and_jacobian(
            p, self.xyz, self.K, self.scale, self.edges)
        return np.r_[old, extra], np.vstack([old_jacobian, derivative])

    def evaluate(self, R, t):
        checked = self.base.evaluate(R, t)
        if checked.get('objective') is None:
            return {**checked, 'finite_native_support_guard_active': True}
        p = N.V._pose_vector(np.asarray(R), np.asarray(t), self.scale)
        slack, _, audit = finite_support_slacks_and_jacobian(
            p, self.xyz, self.K, self.scale, self.edges)
        feasible = bool(np.isfinite(slack).all()
            and (len(slack) == 0 or np.min(slack) >= -self.cfg['feasibility_tolerance']))
        return {**checked, 'accepted': bool(checked['accepted'] and feasible),
            'minimum_joint_slack': min(checked.get('minimum_joint_slack', float('inf')),
                float(np.min(slack)) if len(slack) else float('inf')),
            'finite_native_support_guard_active': True,
            'finite_native_support_guard_pass': feasible,
            'finite_native_support_audit': audit,
            'reason': checked.get('reason') if feasible else 'projected_finite_edge_does_not_cover_actual_support'}
