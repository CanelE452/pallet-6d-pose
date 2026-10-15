"""Explicit alternative-line likelihood for one associated native 3D edge.

All actual-image alternatives remain components. Their normalized prior
weights prevent a duplicated/ambiguous component list from acquiring extra
likelihood merely by being longer. Covariance volume enters each component
density, so a broad component receives no artificial peak-density advantage.
The result is -2log of the mixture of radial-Huber pseudo-likelihoods relative
to its fixed maximum peak, with an exact posterior-weighted derivative.
This is a conditional image factor, not a calibrated pose probability.
No input fetching, association selection, coordinate inference or fitting.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from . import projected_native_edge_factor_v4_17 as E


def mixture_edge_value_and_gradient(p, xyz, K, scale, components, *,
        component_weights=None, reliability_weight=1., huber_scale=3.):
    reliability_weight = float(reliability_weight)
    if not np.isfinite(reliability_weight) or reliability_weight < 0:
        raise ValueError('Finite nonnegative group reliability required')
    if reliability_weight == 0:
        return 0., np.zeros(6), dict(enabled=False, S_or_U_local_queries=0)
    if not isinstance(components, (tuple, list)) or not components:
        raise ValueError('At least one authenticated actual-image line component required')
    pair = frozenset((components[0]['native_corner'], components[0]['native_neighbor']))
    if len(pair) != 2 or any(frozenset((v['native_corner'], v['native_neighbor'])) != pair for v in components):
        raise ValueError('Mixture alternatives must describe the same named physical native edge')
    weights = np.ones(len(components)) if component_weights is None else np.asarray(component_weights, float)
    if weights.shape != (len(components),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Positive explicit mixture component weights required')
    log_prior = np.log(weights)
    log_prior -= logsumexp(log_prior)
    weights = np.exp(log_prior)
    log_volumes = []
    values, derivatives, component_audits = [], [], []
    for component in components:
        value, derivative, audit = E.projected_edge_value_and_gradient(p, xyz, K, scale,
            component, weight=1., huber_scale=huber_scale)
        values.append(value); derivatives.append(derivative); component_audits.append(audit)
        correlation = float(component.get('normal_angle_correlation', 0.))
        log_volumes.append(.5 * (np.log(float(component['normal_variance_px2']))
            + np.log(float(component['angular_variance_rad2'])) + np.log1p(-correlation**2)))
    values = np.asarray(values); derivatives = np.asarray(derivatives)
    log_density_weights = log_prior - np.asarray(log_volumes)
    fixed_log_peak = float(logsumexp(log_density_weights))
    logits = log_density_weights - .5 * values
    log_likelihood = float(logsumexp(logits))
    responsibilities = np.exp(logits - log_likelihood)
    value = -2. * reliability_weight * (log_likelihood - fixed_log_peak)
    # Harmless negative roundoff at identical zero-cost components is zero.
    if -1e-12 < value < 0:
        value = 0.
    return float(value), reliability_weight * responsibilities @ derivatives, dict(
        enabled=True, physical_native_edge=sorted(pair), component_count=len(components),
        normalized_component_priors=weights.tolist(),
        component_covariance_log_sqrt_determinants=log_volumes,
        fixed_mixture_log_peak=fixed_log_peak,
        covariance_volume_in_component_density=True,
        component_native_pseudo_losses=values.tolist(),
        posterior_component_responsibilities=responsibilities.tolist(),
        component_audits=component_audits,
        reliability_weight_applied_once=reliability_weight,
        all_parallel_alternatives_retained=True,
        best_RMS_component_selection=False, support_anchor_is_precise_KP=False,
        normalized_conditional_mixture=True, calibrated_pose_probability_claimed=False,
        independent_RGB_or_Gaussian_information_claimed=False, S_or_U_local_queries=0)
