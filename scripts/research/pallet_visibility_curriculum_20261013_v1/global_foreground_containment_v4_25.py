"""Conditional global foreground containment, with the frozen outline API.

Observed foreground is assumed to be a subset of the projected full convex
cuboid. An observed sample inside that hull contributes zero; an outside
sample contributes the existing reverse finite-segment pseudo-Huber term.
There is no model-to-observed coverage term and no forced complete silhouette
match. The caller retains the existing hard constraints and owns factor weight.

This soft engineering cue does not certify SAM ownership, visibility, native
identity, calibrated probability or pose uniqueness. False-positive mask
foreground can still bias it, and an oversized projected model can have zero
cost. All eight model vertices, including S/O/U geometry, are permitted model
geometry; no state, raw keypoint, image, mask raster or reference is queried.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import global_outline_analytic_factor_v4_11 as OLD
from . import joint_ot_jacobian_v2 as J
from . import statewise_global_outline_solver_v4_10 as BASE


FACTOR_SCHEMA = 'CONDITIONAL_GLOBAL_FOREGROUND_CONTAINMENT_V4_25'
REVERSE_COEFFICIENT = .5


def _provenance():
    return dict(
        schema=FACTOR_SCHEMA,
        observed_foreground_subset_of_projected_full_convex_model_assumed=True,
        model_to_observed_factor_weight=0.,
        observed_to_model_factor_weight=REVERSE_COEFFICIENT,
        inside_model_hull_cost_zero=True,
        outside_distance_uses_nearest_finite_model_segment=True,
        observed_samples_reused_without_new_sampling=True,
        frame_border_closing_segments_not_created=True,
        full_model_geometry_not_S_or_U_image_observations=True,
        S_local_queries=0, U_local_queries=0, raw_coordinate_reads=0,
        image_reads=0, mask_raster_reads=0, GT_inputs=False,
        uncertainty_is_engineering_assumption=True,
        mask_ownership_or_visibility_certificate=False,
        model_expansion_or_depth_uniqueness_proved=False,
        global_uniqueness_proved=False)


def outline_value_and_gradient(p, xyz, K, scale, observation, configuration=None):
    """Return ``.5 * mean(rho(outside_distance / sigma))``, its Jacobian and audit.

    Configuration, default sigma=1px and delta=1.25, pose chart, full camera
    projection and positive-depth admission match the frozen outline factor.
    The fixed observed samples already carry the caller's canvas clipping and
    image/mask provenance. Their original polygon/segments are not used as
    nearest targets. Hull membership and nearest-segment identities are local
    piecewise associations; closest-point derivatives use the envelope theorem.
    """
    cfg = BASE._outline_config(configuration)
    audit = _provenance()
    if not observation.get('supported'):
        return 0., np.zeros(6), dict(audit, valid_projection=True, enabled=False,
                                    reason='unsupported_frozen_global_observation')
    observed = np.asarray(observation['observed_boundary_samples_xy'], dtype=float)
    if (observed.ndim != 2 or observed.shape[1] != 2 or not len(observed)
            or not np.isfinite(observed).all()):
        raise ValueError('Finite nonempty frozen observed boundary samples [N,2] required')
    q, depth, jq, _ = J.projection_and_jacobian(p, xyz, K, scale)
    if not np.isfinite(q).all() or np.min(depth) <= 1e-8:
        return 1e12, np.zeros(6), dict(audit, valid_projection=False,
                                      reason='invalid_or_nonpositive_model_projection')
    try:
        # scipy's 2D hull vertices are counterclockwise.
        indices = ConvexHull(q).vertices
    except QhullError:
        return 1e12, np.zeros(6), dict(audit, valid_projection=False,
                                      reason='degenerate_projected_model_hull')
    polygon, jp = q[indices], jq[indices]
    following, following_jp = np.roll(polygon, -1, axis=0), np.roll(jp, -1, axis=0)
    directions = following - polygon
    relative = observed[:, None, :] - polygon[None, :, :]
    halfplane_cross = (directions[None, :, 0] * relative[:, :, 1]
                       - directions[None, :, 1] * relative[:, :, 0])
    outside = np.any(halfplane_cross < 0., axis=1)
    number = len(observed)
    distances = np.zeros(number)
    value, gradient = 0., np.zeros(6)
    if np.any(outside):
        segments = np.stack([polygon, following], axis=1)
        residual, which, alpha = OLD._nearest(observed[outside], segments)
        loss, force = OLD._loss_and_force(residual, cfg['normal_sigma_px'], cfg['huber_delta'])
        foot_j = ((1. - alpha[:, None, None]) * jp[which]
                  + alpha[:, None, None] * following_jp[which])
        # Divide by every fixed observed sample, including exact zero interiors.
        value = REVERSE_COEFFICIENT * float(loss.sum()) / number
        gradient = (-REVERSE_COEFFICIENT
                    * np.einsum('ni,nij->j', force, foot_j) / number)
        distances[outside] = np.linalg.norm(residual, axis=1)
    audit.update(
        valid_projection=True, enabled=True,
        observed_sample_count=number,
        inside_or_boundary_sample_count=int((~outside).sum()),
        outside_sample_count=int(outside.sum()),
        projected_model_hull_native_ids=indices.tolist(),
        model_hull_not_canvas_clipped=True,
        model_to_observed_robust_mean=0.,
        observed_to_model_outside_robust_mean=value / REVERSE_COEFFICIENT,
        observed_to_model=dict(count=number, mean_px=float(distances.mean()),
                               median_px=float(np.median(distances)),
                               P90_px=float(np.quantile(distances, .9))),
        normal_sigma_px=float(cfg['normal_sigma_px']), huber_delta=float(cfg['huber_delta']),
        factor_weight_applied_by_caller_not_inside_kernel=True)
    return float(value), gradient, audit
