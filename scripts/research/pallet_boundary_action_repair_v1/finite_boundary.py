"""Fixed-universe finite-segment support for the original RGB boundary energy.

No observation, profile distribution, reliability, or RGB is changed. Unsupported
candidate geometry keeps its old finite diagnostic energy, but is ineligible.
The optimizer receives analytic, division-free finite-support inequalities.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import boundary_model as OLD

MACHINE_EPS = np.finfo(np.float64).eps
LAMBDA_EPS = 8 * np.sqrt(MACHINE_EPS)
PARALLEL_ANGLE_EPS = np.sqrt(MACHINE_EPS)


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def fixed_groups(bundle):
    """Duplicate serialization of the same support group adds no evidence."""
    seen = {}
    for group in (bundle or {}).get('groups', []):
        unique = {q['support_id']: q for q in group['queries'] if q['supported']}
        if not group['supported'] or not unique:
            continue
        key = int(group['native_corner'])
        value = dict(group=group, queries=[unique[k] for k in sorted(unique)])
        if key in seen:
            old = seen[key]
            if (set(unique) != {q['support_id'] for q in old['queries']} or
                    group['reliability_weight'] != old['group']['reliability_weight']):
                raise ValueError('Conflicting duplicate original boundary group')
            continue
        seen[key] = value
    return [seen[k] for k in sorted(seen)]


def fixed_queries(bundle):
    return [q for g in fixed_groups(bundle) for q in g['queries']]


def universe(bundle):
    gs = fixed_groups(bundle)
    return dict(group_count=len(gs), profile_count=sum(len(g['queries']) for g in gs),
        support_ids=[q['support_id'] for g in gs for q in g['queries']],
        consumed_native_ids=list((bundle or {}).get('consumed_native_ids', [])),
        consumed_group_ids=list((bundle or {}).get('consumed_group_ids', [])),
        original_fixed_bundle_sha256=hashlib.sha256(json.dumps(bundle, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest())


def numeric_tolerances(bundle):
    diag = float(np.hypot(*(bundle or {}).get('image_hw', [1., 1.])))
    return dict(lambda_epsilon=float(LAMBDA_EPS),
        projected_length_epsilon_px=float(np.sqrt(MACHINE_EPS)*max(1., diag)),
        parallel_angle_epsilon=float(PARALLEL_ANGLE_EPS),
        origin='float64 sqrt(machine epsilon), scaled only by original image diagonal; no accuracy fitting')


def intersection(query, projected, pixel_scale, *, image_hw=(1, 1), depth=None):
    """Return actual infinite-line intersection and independent finite predicate."""
    p = np.asarray(projected, float)
    k, m = query['native_corner'], query['native_neighbor']
    a, b = p[k], p[m]
    c, n = np.asarray(query['center_xy'], float), np.asarray(query['outward_normal_xy'], float)
    v, w = b-a, a-c
    D, N, M = cross(n, v), cross(w, n), cross(w, v)
    length = float(np.linalg.norm(v))
    eps_px = np.sqrt(MACHINE_EPS)*max(1., float(np.hypot(*image_hw)))
    status = 'SUPPORTED_FINITE_SEGMENT'
    if not np.isfinite(p).all() or not np.isfinite([D, N, M, length]).all():
        status = 'NONFINITE_PROJECTION'
    elif depth is not None and (not np.isfinite(depth).all() or np.min(depth) <= 0):
        status = 'NONPOSITIVE_DEPTH'
    elif length <= eps_px:
        status = 'DEGENERATE_PROJECTED_EDGE'
    elif abs(D) <= max(eps_px, PARALLEL_ANGLE_EPS*length):
        status = 'PARALLEL_NORMAL_AND_EDGE'
    lam, offset, point = None, None, None
    if status not in ('NONFINITE_PROJECTION', 'DEGENERATE_PROJECTED_EDGE', 'PARALLEL_NORMAL_AND_EDGE') and D != 0:
        lam, offset = float(N/D), float(M/D/pixel_scale)
        point = (a+lam*v).tolist()
        if status == 'SUPPORTED_FINITE_SEGMENT' and lam < -LAMBDA_EPS:
            status = 'BEFORE_FINITE_ENDPOINT'
        elif status == 'SUPPORTED_FINITE_SEGMENT' and lam > 1+LAMBDA_EPS:
            status = 'AFTER_FINITE_ENDPOINT'
    return dict(support_id=query['support_id'], native_corner=int(k), native_neighbor=int(m),
        projected_endpoints_xy=[a.tolist(), b.tolist()], projected_length_px=length,
        denominator_px=D, lambda_segment=lam, offset_base_px=offset,
        intersection_xy=point, finite_support=status == 'SUPPORTED_FINITE_SEGMENT',
        geometry_status=status, off_canvas_intersection=(point is not None and not
            (0 <= point[0] < image_hw[1] and 0 <= point[1] < image_hw[0])))


def support_audit(bundle, projected, depth=None):
    qs = fixed_queries(bundle)
    reports = [intersection(q, projected, bundle['pixel_scale'], image_hw=bundle['image_hw'], depth=depth) for q in qs]
    return dict(finite_support=all(q['finite_support'] for q in reports),
        status='FINITE_SUPPORT_PASS' if reports and all(q['finite_support'] for q in reports) else
            'NO_FIXED_B_INPUT_ORIGINAL_OBSERVATIONS_RETAINED' if not reports else 'FINITE_SUPPORT_REJECT',
        supported_profiles=sum(q['finite_support'] for q in reports), fixed_profile_count=len(reports),
        fixed_group_count=len(fixed_groups(bundle)), profiles=reports,
        numeric_tolerances=numeric_tolerances(bundle), actual_material_owner='UNVERIFIED',
        candidate_specific_observation_deletions=0, clipped_intersections=0)


def constraint_orientation(bundle, projected):
    init = np.asarray(bundle['initial_projection_xy'], float) if bundle else None
    result = []
    for q in fixed_queries(bundle):
        k, m = q['native_corner'], q['native_neighbor']
        n = np.asarray(q['outward_normal_xy'])
        D = cross(n, projected[m]-projected[k])
        if abs(D) <= numeric_tolerances(bundle)['projected_length_epsilon_px']:
            D = cross(n, init[m]-init[k])
        result.append(1. if D >= 0 else -1.)
    return np.asarray(result)


def constraints_and_jacobian(bundle, projected, projection_jacobian=None, orientations=None):
    """Smooth constraints within one nonparallel connected component.

    Signed D, signed N and signed(D-N) enforce the same finite segment as
    lambda=N/D without dividing near D=0. Every original fixed query remains.
    The orientation is fixed once per solve, never chosen to reduce its cost.
    """
    p = np.asarray(projected, float)
    jp = np.asarray(projection_jacobian) if projection_jacobian is not None else np.eye(16).reshape(8, 2, 16)
    qs = fixed_queries(bundle)
    if not qs:
        return np.empty(0), np.empty((0, jp.shape[-1]))
    signs = constraint_orientation(bundle, p) if orientations is None else np.asarray(orientations)
    init = np.asarray(bundle['initial_projection_xy'], float)
    eps_px = numeric_tolerances(bundle)['projected_length_epsilon_px']
    vals, jacs = [], []
    for q, sign in zip(qs, signs):
        k, m = q['native_corner'], q['native_neighbor']
        n = np.asarray(q['outward_normal_xy'], float)
        a, b = p[k], p[m]
        v, w = b-a, a-np.asarray(q['center_xy'])
        D, N = cross(n, v), cross(w, n)
        dj = n[0]*(jp[m,1]-jp[k,1])-n[1]*(jp[m,0]-jp[k,0])
        nj = n[1]*jp[k,0]-n[0]*jp[k,1]
        length = float(np.linalg.norm(v))
        lengthj = v@(jp[m]-jp[k])/max(length, eps_px)
        norm = max(float(np.linalg.norm(init[m]-init[k])), eps_px)
        threshold = max(eps_px, PARALLEL_ANGLE_EPS*length)
        thresholdj = PARALLEL_ANGLE_EPS*lengthj if PARALLEL_ANGLE_EPS*length > eps_px else np.zeros_like(lengthj)
        vals.extend([(sign*D-threshold)/norm,
                     sign*(N+LAMBDA_EPS*D)/norm,
                     sign*((1+LAMBDA_EPS)*D-N)/norm])
        jacs.extend([(sign*dj-thresholdj)/norm,
                     sign*(nj+LAMBDA_EPS*dj)/norm,
                     sign*((1+LAMBDA_EPS)*dj-nj)/norm])
    return np.asarray(vals), np.asarray(jacs)


def energy_and_gradient(bundle, projected, *, model='region', depth=None, return_audit=False):
    """Original full-universe energy unchanged; eligibility is a separate fact."""
    if bundle is None:
        value, grad = 0., np.zeros((8, 2))
    else:
        # Canonicalize repeated serialization; real original groups are unique.
        b = dict(bundle, groups=[g['group'] for g in fixed_groups(bundle)])
        value, grad = OLD.energy_and_gradient(b, projected, model=model)
    audit = support_audit(bundle, projected, depth)
    return (value, grad, audit) if return_audit else (value, grad)
