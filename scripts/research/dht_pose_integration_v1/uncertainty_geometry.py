"""GT-free heteroscedastic corner/line geometry and DHT posterior moments.

Posterior spreads are model-distribution summaries, not calibrated geometric
uncertainties. Calibration/selection are deliberately outside this module.
"""
from __future__ import annotations

import numpy as np

from fusion import INCIDENT_ROLES, _image_geometry, _points_and_valid


def candidate_homogeneous(theta_deg, rho, width, height, pad=100, grid_size=50):
    """Original-image unit-normal h=(nx,ny,c), h@[x,y,1]=0.

    Uses the frozen VGG pixel-center inverse: feature center24.5 maps to the
    original image pixel center((W-1)/2,(H-1)/2). Supports broadcast arrays.
    """
    origin, _ = _image_geometry(width, height)
    theta, rho = np.broadcast_arrays(np.deg2rad(np.asarray(theta_deg, float)), np.asarray(rho, float))
    scale = np.array([(width + 2 * pad) / grid_size, (height + 2 * pad) / grid_size])
    raw_normal = np.stack([np.cos(theta), np.sin(theta)], -1) / scale
    norm = np.linalg.norm(raw_normal, axis=-1)
    normal = raw_normal / norm[..., None]
    offset = -np.sum(normal * origin, axis=-1) - rho / norm
    return np.concatenate([normal, offset[..., None]], -1)


def posterior_moments(probability, theta_deg, rho, mode_indices, width, height,
                      local_theta_deg=5., local_rho=2.):
    """Summarize [roles,K] probabilities over VALID bins only, without GT.

    Candidate homogeneous normals are aligned to each mode by original-normal
    dot-product sign. ``second_moment_about_mode`` is E[(h-hmode)(h-hmode)^T],
    not a covariance around the posterior mean. At q it gives mean squared
    residual DIFFERENCE from the argmax line: z M z^T. Full posterior modes are
    retained; the separate local chart summary is diagnostic only.
    """
    p = np.asarray(probability, np.float64)
    theta, rho = np.asarray(theta_deg, float), np.asarray(rho, float)
    indices = np.asarray(mode_indices, int)
    if p.ndim != 2 or theta.shape != (p.shape[1],) or rho.shape != theta.shape or indices.shape != (p.shape[0],):
        raise ValueError('Expected p[roles,K], theta/rho[K], mode_indices[roles]')
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError('Probabilities must be finite nonnegative with positive mass')
    if (indices < 0).any() or (indices >= p.shape[1]).any():
        raise ValueError('Invalid posterior mode index')
    p = p / p.sum(-1, keepdims=True)
    h = candidate_homogeneous(theta, rho, width, height)
    mode = h[indices]
    polarity = np.where(mode[:, :2] @ h[:, :2].T >= 0, 1., -1.)
    delta = polarity[..., None] * h[None] - mode[:, None]
    first = np.einsum('rk,rki->ri', p, delta, optimize=True)
    second = np.einsum('rk,rki,rkj->rij', p, delta, delta, optimize=True)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=-1)
    # Local feature-coordinate chart respects (theta+180,-rho) identity.
    raw_angle = theta[None] - theta[indices, None]
    delta_angle = (raw_angle + 90) % 180 - 90
    grid_polarity = np.where(np.cos(np.deg2rad(raw_angle)) >= 0, 1., -1.)
    delta_rho = grid_polarity * rho[None] - rho[indices, None]
    local = (np.abs(delta_angle) <= local_theta_deg) & (np.abs(delta_rho) <= local_rho)
    local_mass = np.sum(p * local, -1)
    if (local_mass <= 0).any():
        raise ValueError('Local chart must contain its positive-mass mode')
    local_probability = p * local / local_mass[:, None]
    chart = np.stack([np.deg2rad(delta_angle), delta_rho], -1)
    local_mean = np.einsum('rk,rki->ri', local_probability, chart, optimize=True)
    local_second = np.einsum('rk,rki,rkj->rij', local_probability, chart, chart, optimize=True)
    local_covariance = local_second - local_mean[:, :, None] * local_mean[:, None, :]
    return dict(mode_h=mode, mean_delta_h=first, second_moment_about_mode=second,
        entropy_nats=entropy, entropy_normalized=entropy / np.log(p.shape[1]) if p.shape[1] > 1 else np.zeros(len(p)),
        top5_mass=np.sort(p, axis=-1)[:, -min(5, p.shape[1]):].sum(-1),
        global_angular_rms_rad=np.sqrt(np.sum(p * np.deg2rad(delta_angle)**2, -1)),
        global_rho_rms_feature_px=np.sqrt(np.sum(p * delta_rho**2, -1)),
        local_mass=local_mass, local_mean_delta_theta_rho=local_mean,
        local_covariance_theta_rho=local_covariance,
        local_second_moment_theta_rho=local_second)


def line_variance_at_points(points, second_moment_about_mode):
    """Return [8,2] uncalibrated original-px² residual spreads at predicted q.

    Incident order matches fusion.INCIDENT_ROLES. A nonfinite input point stays
    NaN. Tiny numerical negative quadratic forms are clipped to zero; the
    registered one-pixel variance floor is applied by calibration/fusion later.
    """
    points, matrices = np.asarray(points, float), np.asarray(second_moment_about_mode, float)
    if points.shape not in ((8, 2), (9, 2)) or matrices.shape != (8, 3, 3):
        raise ValueError('Expected points[8/9,2], matrices[8,3,3]')
    z = np.column_stack([points[:8], np.ones(8)])
    incident = matrices[np.asarray(INCIDENT_ROLES)]
    raw = np.einsum('ki,krij,kj->kr', z, incident, z, optimize=True)
    return np.maximum(raw, 0)


def fuse(points, valid, lines, point_variance_xy, line_variance_per_corner_role,
         lam, width, height, max_move=None):
    """Weighted fixed-incidence fusion for one frame/seed; no GT accepted.

    Point variance[8/9,2] and incident-line variance[8,2] are ORIGINAL px².
    Finite nonnegative variances get floor1px². The baseline anchor is positive
    diagonal precision. Optional max_move is an ORIGINAL-pixel displacement
    norm cap, applied after solving. Missing corners and ninth center are copied.
    All-one variances with no cap recover the original unweighted fusion.
    """
    points, effective_valid = _points_and_valid(points, valid)
    _, diagonal = _image_geometry(width, height)
    lines = np.asarray(lines, float)
    vp, vl = np.asarray(point_variance_xy, float), np.asarray(line_variance_per_corner_role, float)
    if lines.shape != (8, 2, 2) or vp.shape not in ((8, 2), (len(points), 2)) or vl.shape != (8, 2):
        raise ValueError('Expected lines[8,2,2], point_variance[8/9,2], incident_variance[8,2]')
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lam must be finite and nonnegative')
    if max_move is not None and (not np.isfinite(max_move) or max_move < 0):
        raise ValueError('max_move must be a nonnegative original-pixel norm cap')
    result = points.copy()
    if lam == 0:
        return result, effective_valid
    direction = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(direction, axis=-1)
    line_valid = np.isfinite(lines).all((1, 2)) & (length > diagonal * 1e-12)
    normals = np.zeros((8, 2))
    normals[line_valid] = np.stack([-direction[line_valid, 1], direction[line_valid, 0]], -1) / length[line_valid, None]
    for k, roles in enumerate(INCIDENT_ROLES):
        if not effective_valid[k]:
            continue
        if not np.isfinite(vp[k]).all() or (vp[k] < 0).any():
            raise ValueError('Observed points require finite nonnegative point variances')
        if not np.isfinite(vl[k]).all() or (vl[k] < 0).any():
            raise ValueError('Observed points require finite nonnegative incident-line variances')
        available = [j for j, role in enumerate(roles) if line_valid[role]]
        if not available:
            continue
        used_roles = [roles[j] for j in available]
        n = normals[used_roles]
        # All terms use pixels and pixels²; common image-normalization cancels.
        point_precision = 1 / np.maximum(vp[k], 1.)
        line_precision = 1 / np.maximum(vl[k, available], 1.)
        residual = np.sum(n * (points[k] - lines[used_roles, 0]), axis=-1)
        system = np.diag(point_precision) + lam * (n.T * line_precision) @ n
        displacement = np.linalg.solve(system, -lam * n.T @ (line_precision * residual))
        norm = np.linalg.norm(displacement)
        if max_move is not None and norm > max_move:
            displacement *= max_move / norm
        result[k] = points[k] + displacement
    return result, effective_valid
