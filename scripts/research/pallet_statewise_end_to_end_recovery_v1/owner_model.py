"""A: one global proper upright cuboid owner, with per-group none mixtures.

Primitive NumPy API only: no file/reference/image/pose-solver inputs. This is
an opt-in conditional energy model, not a calibrated pose posterior. One
global permutation transports every active observation and every constraint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from scipy.special import kve, logsumexp


def pseudo_huber_squared(radius2, delta=3.):
    """Exactly the original e radial soft-L1/pseudo-Huber loss."""
    x = np.asarray(radius2, dtype=float)
    return 2. * delta * delta * (np.sqrt(1. + x / (delta * delta)) - 1.)


def log_radial_normalizer(dimension, delta=3.):
    """Integral of exp(-rho_delta(||z||)/2) over R^dimension.

    Z_d = 2 (2 pi)^((d-1)/2) delta exp(delta^2) K_((d+1)/2)(delta^2).
    kve evaluates exp(x) K_v(x) without underflow. Z_2 has a closed form.
    """
    d = int(dimension)
    delta = float(delta)
    if d < 1 or d != dimension or not np.isfinite(delta) or delta <= 0:
        raise ValueError('Positive integer dimension and positive delta required')
    if d == 2:
        return math.log(2. * math.pi * (1. + 1. / delta**2))
    return (math.log(2. * delta) + .5 * (d - 1) * math.log(2. * math.pi)
            + math.log(float(kve(.5 * (d + 1), delta**2))))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def upright_owner_permutations(xyz):
    """Four y-preserving proper C4 graph automorphisms, in fixed order.

    The coordinates themselves are NEVER rotated, rescaled or reordered.
    p[k] says that source channel k owns actual model vertex p[k]. Quarter
    turns swap x/z graph directions; they are correspondence alternatives,
    not metric symmetries of a rectangular box.
    """
    xyz = np.asarray(xyz, float)
    if xyz.shape != (8, 3) or not np.isfinite(xyz).all():
        raise ValueError('Exactly eight finite original cuboid vertices required')
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    if np.any(hi <= lo):
        raise ValueError('Three positive registered dimensions required')
    signs = (xyz - (lo + hi) / 2.) / ((hi - lo) / 2.)
    if not np.allclose(np.abs(signs), 1., atol=1e-10, rtol=0):
        raise ValueError('Each registered vertex must be a box corner')
    lookup = {tuple(np.rint(v).astype(int)): i for i, v in enumerate(signs)}
    if len(lookup) != 8:
        raise ValueError('Unique registered vertices required')
    quarter = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    result = []
    for turn in range(4):
        rotation = np.linalg.matrix_power(quarter, turn)
        p = tuple(lookup[tuple(np.rint(rotation @ v).astype(int))] for v in signs)
        result.append(p)
    return tuple(result)


def validate_owner_permutation(xyz, permutation):
    """Reject reflections, independent swaps, upside-down and nonbijective maps."""
    p = tuple(permutation)
    if p not in upright_owner_permutations(xyz):
        raise ValueError('Only one complete proper upright cyclic owner permutation is allowed')
    return p


def _field(record, name):
    if name not in record:
        raise ValueError('Required original observation field absent: ' + name)
    return record[name]


def _native_id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 8:
        raise ValueError('Strict native corner ID in [0,7] required')
    return int(value)


def _freeze_group(group, delta, none_prior):
    d = group['dimension']
    group['log_inlier_peak'] = -log_radial_normalizer(d, delta) - .5 * group['log_covariance_determinant']
    group['log_none_density'] = -float(np.log(group['none_support_widths']).sum())
    group['log_mixture_peak'] = float(np.logaddexp(math.log1p(-none_prior) + group['log_inlier_peak'],
                                                  math.log(none_prior) + group['log_none_density']))
    return group


def compile_owner_groups(xyz, K=None, measurements=None, boundary=None, image_hw=None, *,
                         none_prior, huber_scale=3., permutations=None, states=None):
    """Compile original COMPLETE and grouped PARTIAL observations once.

    ``xyz`` may be the existing fixed_cost.Context. Prepared Context lists can
    still archive full points revoked by partial: they are removed ONCE here,
    identically for every h. A model projection or S/O/T/U point is rejected.
    The required none_prior must be frozen from allowed synthetic validation.
    """
    source_records = {}
    if hasattr(xyz, 'xyz'):
        ctx = xyz
        xyz, K, measurements, boundary, image_hw = ctx.xyz, ctx.K, ctx.measurements, ctx.boundary, ctx.image_hw
        states = ctx.states
        huber_scale = ctx.cfg['huber_scale']
        source_records = {r['native_corner']: r for r in ctx.packet['solver_input']['visible']}
    xyz, K = np.asarray(xyz, float), np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('Original finite pixel-camera matrix required')
    if image_hw is None or len(image_hw) != 2 or min(image_hw) <= 0:
        raise ValueError('Original positive image canvas required')
    H, W = map(float, image_hw)
    eta, delta = float(none_prior), float(huber_scale)
    if not 0. < eta < 1. or not np.isfinite(delta) or delta <= 0:
        raise ValueError('Frozen nondegenerate none prior and positive inherited delta required')
    native_perms = upright_owner_permutations(xyz)
    given = native_perms if permutations is None else permutations
    # Duplicating a hypothesis never duplicates its prior mass.
    perms = tuple(dict.fromkeys(validate_owner_permutation(xyz, p) for p in given))
    if not perms:
        raise ValueError('At least one globally valid owner alternative required')
    measurements, boundary = list(measurements or []), list(boundary or [])
    revoked = {_native_id(_field(b, 'native_corner')) for b in boundary}
    groups, seen, by_owner = [], set(), {}
    # Continuous pixel area [0,W]x[0,H], so resizing every length by s also
    # scales its density support exactly by s^d (no added pixel pseudocount).
    canvas = np.array([[0., 0.], [W, 0.], [W, H], [0., H]])
    for m in measurements:
        k = _native_id(_field(m, 'native_corner'))
        source = m.get('source', source_records.get(k, {}).get('source'))
        if k in revoked:
            continue
        state = (states[k] if states is not None else m.get('state'))
        if state is not None and state not in ('B', 'I'):
            raise ValueError('Only original B/I complete image points may enter A')
        if (m.get('measurement_is_pose_projection') is True or m.get('image_measurement') is False
                or m.get('is_actual_image_observation') is False):
            raise ValueError('Virtual/model-projected coordinates cannot enter image likelihood')
        xy = np.asarray(_field(m, 'xy'), float)
        cov = np.asarray(m.get('fit_covariance_px2', m.get('covariance_px2')), float)
        if xy.shape != (2,) or cov.shape != (2, 2) or not np.isfinite([*xy, *cov.ravel()]).all():
            raise ValueError('Finite original pixel coordinate and px^2 covariance required')
        if not np.allclose(cov, cov.T, atol=1e-10, rtol=0) or np.linalg.eigvalsh(cov).min() <= 0:
            raise ValueError('Positive definite conditional covariance required')
        reliability = float(m.get('fit_reliability_weight', 1.))
        if not np.isfinite(reliability) or reliability < 0:
            raise ValueError('Original nonnegative reliability required')
        payload = dict(kind='COMPLETE_2D', native_corner=k, xy=xy.tolist(), covariance=cov.tolist(),
                       reliability=reliability, source=source)
        identity = _digest(payload)
        if identity in seen:
            continue
        if k in by_owner:
            raise ValueError('Conflicting complete observations require an explicit shared group')
        seen.add(identity); by_owner[k] = identity
        # Fixed finite support contains the canvas and this observed point;
        # neither its size nor none cost is adjusted for a candidate/h.
        support = np.vstack([canvas, xy])
        widths = support.max(axis=0) - support.min(axis=0)
        group = dict(group_id=f'full:{k}', kind='COMPLETE_2D', native_corner=k,
                     dimension=2, xy=xy, inverse_covariance=np.linalg.inv(cov),
                     covariance=cov, log_covariance_determinant=float(np.linalg.slogdet(cov)[1]),
                     none_support_widths=widths, reliability=reliability,
                     observation_fingerprint=identity, owner_status='set_of_possible_owners',
                     source=source or 'PREPARED_ORIGINAL_IMAGE_OBSERVATION')
        groups.append(_freeze_group(group, delta, eta))
    for b in boundary:
        k = _native_id(_field(b, 'native_corner'))
        state = states[k] if states is not None else b.get('state')
        if state is not None and state not in ('B', 'I'):
            raise ValueError('A partial factor must retain original B/I observation state')
        lines = list(_field(b, 'lines'))
        if not lines:
            raise ValueError('Partial needs its actual observed normal(s)')
        normals, offsets, variances, widths = [], [], [], []
        line_seen = set()
        for line in lines:
            n = np.asarray(_field(line, 'normal_unit_xy'), float)
            offset = float(_field(line, 'offset_px'))
            variance = float(_field(line, 'normal_variance_px2'))
            if (n.shape != (2,) or not np.isfinite([*n, offset, variance]).all()
                    or not math.isclose(float(n @ n), 1., abs_tol=1e-7) or variance <= 0):
                raise ValueError('Original normalized line and positive px^2 normal variance required')
            # Exact duplicate signed lines have one likelihood and one prior;
            # sign-equivalent duplicates are canonicalized too.
            sign = 1. if next((v for v in n if abs(v) > 1e-12), 1.) > 0 else -1.
            token = _digest(dict(n=(sign*n).tolist(), offset=sign*offset, variance=variance))
            if token in line_seen:
                continue
            line_seen.add(token)
            extent = np.r_[canvas @ n, -offset]
            widths.append(float(extent.max() - extent.min()))
            normals.append(n); offsets.append(offset); variances.append(variance)
        reliability = float(b.get('fit_reliability_weight', 1.))
        if not np.isfinite(reliability) or reliability < 0:
            raise ValueError('Original nonnegative grouped reliability required')
        payload = dict(kind='GROUPED_PARTIAL_NORMAL', native_corner=k,
                       normals=np.asarray(normals).tolist(), offsets=offsets, variances=variances,
                       reliability=reliability)
        identity = _digest(payload)
        if identity in seen:
            continue
        if k in by_owner:
            raise ValueError('Conflicting or full/partial duplicated likelihood for one native owner')
        seen.add(identity); by_owner[k] = identity
        group = dict(group_id=f'partial:{k}', kind='GROUPED_PARTIAL_NORMAL', native_corner=k,
                     dimension=len(normals), normals=np.asarray(normals), offsets=np.asarray(offsets),
                     variances=np.asarray(variances), none_support_widths=np.asarray(widths),
                     log_covariance_determinant=float(np.log(variances).sum()), reliability=reliability,
                     observation_fingerprint=identity, owner_status='set_of_possible_owners',
                     original_partial_tangent_constraints_preserved_by_caller=True)
        groups.append(_freeze_group(group, delta, eta))
    return dict(schema='GLOBAL_UPRIGHT_OWNER_CONDITIONAL_ENERGY_A_V1', xyz=xyz.copy(), K=K.copy(),
                image_hw=(int(H), int(W)), groups=tuple(groups), permutations=perms,
                normalized_global_priors=np.repeat(1. / len(perms), len(perms)),
                none_prior=eta, huber_scale=delta, original_revoked_full_ids=sorted(revoked),
                global_observation_ids=tuple(g['group_id'] for g in groups),
                fixed_common_observation_count=len(groups),
                uncertainty_is_calibrated_posterior=False, independent_RGB_information_claimed=False)


def _group_value_gradient(group, xy, eta, delta):
    if group['kind'] == 'COMPLETE_2D':
        residual = xy - group['xy']
        precision_residual = group['inverse_covariance'] @ residual
        r2 = float(residual @ precision_residual)
        derivative_half_loss = precision_residual / math.sqrt(1. + r2 / delta**2)
    else:
        residual = group['normals'] @ xy + group['offsets']
        r2 = float(np.sum(residual**2 / group['variances']))
        derivative_half_loss = (group['normals'].T @ (residual / group['variances'])
                                / math.sqrt(1. + r2 / delta**2))
    inlier_logit = math.log1p(-eta) + group['log_inlier_peak'] - .5 * float(pseudo_huber_squared(r2, delta))
    none_logit = math.log(eta) + group['log_none_density']
    log_mix = float(np.logaddexp(inlier_logit, none_logit))
    inlier_responsibility = math.exp(inlier_logit - log_mix)
    reliability = group['reliability']
    value = -2. * reliability * (log_mix - group['log_mixture_peak'])
    gradient = 2. * reliability * inlier_responsibility * derivative_half_loss
    return max(0., value), gradient, 1. - inlier_responsibility


def energies_per_global_h(compiled, projected_native_xy, *, exclude_group_ids=(), extra_h_energies=None,
                          extra_h_gradients=None, return_audit=False):
    """Every h explains the SAME groups. Exclusion is one fixed arm-level set.

    AB may replace B-consumed original groups once and pass its four B energies
    (and optional gradients w.r.t ORIGINAL projected_native_xy) before the global
    log mixture. Passing different exclusion sets per h is not supported.
    """
    q = np.asarray(projected_native_xy, float)
    if q.shape != (8, 2) or not np.isfinite(q).all():
        raise ValueError('Eight finite candidate model projections required')
    excluded = frozenset(exclude_group_ids)
    unknown = excluded - set(compiled['global_observation_ids'])
    if unknown:
        raise ValueError('Cannot remove an unknown image group: ' + str(sorted(unknown)))
    groups = [g for g in compiled['groups'] if g['group_id'] not in excluded]
    count = len(compiled['permutations'])
    extra = np.zeros(count) if extra_h_energies is None else np.asarray(extra_h_energies, float)
    if extra.shape != (count,) or np.any(np.isnan(extra)) or np.any(extra < 0):
        raise ValueError('One nonnegative or +inf replacement/constraint energy per global h required')
    gradients = np.zeros((count, 8, 2)) if extra_h_gradients is None else np.array(extra_h_gradients, float, copy=True)
    if gradients.shape != (count, 8, 2) or not np.isfinite(gradients).all():
        raise ValueError('One finite replacement projection gradient per h required')
    values = extra.copy(); none = np.zeros((count, len(groups))); per_group = np.zeros_like(none)
    for h, p in enumerate(compiled['permutations']):
        for i, group in enumerate(groups):
            owner = p[group['native_corner']]
            value, grad, none_mass = _group_value_gradient(group, q[owner], compiled['none_prior'], compiled['huber_scale'])
            values[h] += value; gradients[h, owner] += grad
            none[h, i] = none_mass; per_group[h, i] = value
    if return_audit:
        return values, gradients, dict(global_permutations=[list(p) for p in compiled['permutations']],
            active_group_ids=[g['group_id'] for g in groups], same_group_count_every_h=len(groups),
            excluded_once_for_entire_arm=sorted(excluded), per_h_group_energies=per_group.tolist(),
            per_h_none_responsibilities=none.tolist(),
            none_constant_energy_limit_per_h=sum(-2*g['reliability']*(math.log(compiled['none_prior'])+
                g['log_none_density']-g['log_mixture_peak']) for g in groups),
            hypothesis_specific_observation_deletion=False, S_image_queries=0)
    return values


def energy(compiled, projected_native_xy, *, exclude_group_ids=(), extra_h_energies=None,
           extra_h_gradients=None, return_audit=False, return_gradient=False):
    values, gradients, audit = energies_per_global_h(compiled, projected_native_xy,
        exclude_group_ids=exclude_group_ids, extra_h_energies=extra_h_energies,
        extra_h_gradients=extra_h_gradients, return_audit=True)
    if not compiled['groups'] and extra_h_energies is None:
        value, responsibility = 0., compiled['normalized_global_priors'].copy()
    elif np.isinf(values).all():
        value, responsibility = float('inf'), np.zeros(len(values))
    else:
        logits = np.log(compiled['normalized_global_priors']) - .5 * values
        normalization = float(logsumexp(logits))
        value, responsibility = -2. * normalization, np.exp(logits-normalization)
    derivative = np.einsum('h,hij->ij', responsibility, gradients)
    audit.update(global_owner_responsibilities=responsibility.tolist(),
                 per_h_total_energies=values.tolist(), normalized_global_priors=compiled['normalized_global_priors'].tolist(),
                 global_owner_argmax=int(np.argmax(responsibility)),
                 owner_is_one_global_assignment_not_independent_per_point=True,
                 calibrated_pose_probability_claimed=False)
    if return_gradient and return_audit:
        return float(value), derivative, audit
    if return_gradient:
        return float(value), derivative
    if return_audit:
        return float(value), audit
    return float(value)


def map_owner_records(records, permutation, xyz, *, state_vector=False):
    """Deep-copy original constraints and carry owner IDs with the SAME h.

    Pixel regions and observed lines/covariance/support are immutable. Derived
    object_xyz and incident_normals are REBUILT at the destination owner.
    Use for O/T/U domains and partial tangent guards; never keep old owner IDs
    while evaluating a different h. No record is removed or state set to S.
    """
    p = validate_owner_permutation(xyz, permutation)
    xyz = np.asarray(xyz, float)
    center = .5 * (xyz.min(axis=0) + xyz.max(axis=0))
    if state_vector:
        if len(records) != 8:
            raise ValueError('Eight original states required')
        mapped = [None] * 8
        for k, state in enumerate(records):
            mapped[p[k]] = state
        return mapped
    owner_fields = {'native_corner', 'native_neighbor', 'native_id', 'owner_native_id', 'physical_native_corner'}
    pair_fields = {'native_edge', 'physical_native_edge', 'physical_native_ids'}
    def visit(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in owner_fields:
                    result[key] = p[_native_id(item)]
                elif key in pair_fields:
                    result[key] = [p[_native_id(k)] for k in item]
                elif key == 'provenance':
                    result[key] = deepcopy(item)  # immutable producer record, not new association certification
                else:
                    result[key] = visit(item)
            corner = result.get('native_corner', result.get('native_id'))
            if corner is not None:
                if 'object_xyz' in value:
                    result['object_xyz'] = xyz[corner].copy()
                if 'incident_normals' in value:
                    # Original registered axis-aligned cuboid, all three
                    # outward incident faces. Ordering is immaterial to max
                    # incident cosine; these are not transported old normals.
                    result['incident_normals'] = np.diag(np.sign(xyz[corner] - center))
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return deepcopy(value)
    return visit(records)


def project_native(xyz, R, t, K):
    """Pure original-native projection, convenient for full solver integration."""
    xyz, R, t, K = map(lambda value: np.asarray(value, float), (xyz, R, t, K))
    camera = xyz @ R.T + t
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2, None], camera[:, 2]
