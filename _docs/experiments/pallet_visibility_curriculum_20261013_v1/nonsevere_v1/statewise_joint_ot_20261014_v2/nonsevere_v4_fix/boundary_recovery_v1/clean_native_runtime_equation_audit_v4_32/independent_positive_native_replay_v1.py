"""Pure positive-packet V32 replay using independent V29 scalar equations.

No V32 consumer, planner, AUTH, candidate generator or objective is imported.
The original V17 parser/guard and independent V29 energy equations are reused.
This module performs no IO or optimizer/PnP calls. Provenance and original
byte closure remain separate caller duties.
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.research.pallet_visibility_curriculum_20261013_v1 import audit_geometry_guided_edges_v4_29 as INDEPENDENT

WORKER_SCHEMA = 'POSITIVE_CLEAN_FRESH_BI_WORKER_PACKET_V32'
WORKER_KEYS = frozenset(('schema', 'identifiers', 'clean_input', 'physics',
                        'typed_bundles', 'regularizers', 'source_contract'))
FAMILY = 'INITIAL_ONLY_OBSERVED_BI_NATIVE_RAY_ACTUAL_EDGES_V32'
POPULATION = 'V32_INITIAL_ONLY_QUERY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
PROFILE = 'INITIAL_ONLY_ORIGINAL17C_OR_MEASURED_BI_QUERY_FIXED_NATIVE_RAY_V32'


def positive_task(packet):
    if set(packet) != WORKER_KEYS or packet['schema'] != WORKER_SCHEMA:
        raise ValueError('Exact current positive worker interface required')
    source, physics = packet['clean_input'], packet['physics']
    if packet['source_contract']['ready'] is not True:
        raise ValueError('Ready positive packet required for numeric replay; retain no-pose rows separately')
    return dict(spec=dict(K=deepcopy(source['K']), states=list(source['states']),
        visible=deepcopy(source['visible_measurements']), domains=deepcopy(physics['domains']),
        image_hw=list(physics['image_hw']), config=deepcopy(physics['config'])),
        states=list(source['states']), initial_self_hidden=list(source['initial_self_hidden']),
        boundary_merged=deepcopy(physics['boundary_merged']), u_regions=deepcopy(physics['u_regions']))


def _rotation(value):
    answer = np.asarray(value, float)
    if (answer.shape != (3, 3) or not np.isfinite(answer).all()
            or not np.allclose(answer.T@answer, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(answer), 1., atol=1e-8, rtol=0)):
        raise ValueError('Proper finite rotation required')
    return answer


def check_initial_physical_body(packet, family):
    source = packet['clean_input']
    xyz, body = np.asarray(source['native_xyz'], float), np.asarray(family['native_xyz'], float)
    if xyz.shape != (8, 3) or body.shape != (8, 3):
        raise ValueError('Eight known model vertices required')
    R, Q = _rotation(source['initial_coarse_pose']['R']), _rotation(source['physical_registration_Q'])
    Rf, Qf = _rotation(family['coarse_pose']['R']), _rotation(family['physical_registration_Q'])
    t, tf = np.asarray(source['initial_coarse_pose']['t'], float), np.asarray(family['coarse_pose']['t'], float)
    original, current = xyz@R.T+t, body@Rf.T+tf
    distance = np.linalg.norm(current[:, None]-original[None, :], axis=2)
    if (t.shape != (3,) or tf.shape != (3,) or not np.isfinite(distance).all()
            or not np.allclose(Rf@Qf, R@Q, atol=1e-10, rtol=0)
            or not np.allclose(current.mean(axis=0), original.mean(axis=0), atol=1e-10, rtol=0)
            or max(distance.min(axis=0).max(), distance.min(axis=1).max()) > 1e-9):
        raise ValueError('Fresh family must retain independent initial physical body and registration')
    if family['name'] == 'ORIGINAL_WD':
        if not (np.array_equal(xyz, body) and np.array_equal(R, Rf)
                and np.array_equal(Q, Qf) and np.array_equal(t, tf)):
            raise ValueError('Original family must retain original numeric physics')
    elif family['name'] != 'SWAPPED_WD':
        raise ValueError('Only the two declared WD families are supported')
    return dict(physical_registration_and_geometric_center_preserved=True,
        maximum_unordered_initial_body_gap=float(distance.min(axis=0).max()),
        native_index_correspondence_certified=False)


def replay(packet, family, plan, R, t, *, expected=None, expected_gradient=None):
    """Recompute factor identities/energy/guards and optional finite differences.

    All supplied poses are retained. An invalid/collapsed projection may raise
    and must be recorded by the caller alongside the original rejected row.
    Caller must not discard that row or replace it with another method.
    """
    task = positive_task(packet)
    family_check = check_initial_physical_body(packet, family)
    if (plan['observation_family'] != FAMILY or plan['population'] != POPULATION
            or plan['selection_profile'] != PROFILE):
        raise ValueError('Current initial-only V32 plan required')
    policy = plan['source_policy']
    contract = packet['source_contract']
    if contract['model_availability'][policy]:
        gamma = float(contract['tangent_parameters'][policy]['gamma'])
    else:
        if plan['active_groups'] or plan['new_owner_ids']:
            raise ValueError('Unavailable current model cannot carry line factors')
        gamma = 0.
    raw = INDEPENDENT.prepare_raw(task, family)
    before = INDEPENDENT._digest(dict(packet=packet, family=family, plan=plan))
    fresh = INDEPENDENT.recompute_native(task, family, R, t, plan, gamma, raw=raw)
    checks = list(fresh['checks'])
    R, t = np.asarray(R, float), np.asarray(t, float)
    camera = np.asarray(family['native_xyz'])@R.T+t
    homogeneous = camera@np.asarray(packet['clean_input']['K']).T
    manual = homogeneous[:, :2]/homogeneous[:, 2:3]
    checks.append((np.allclose(fresh['projected_native'], manual, atol=1e-8, rtol=1e-10),
                   'independent full-K projection'))
    checks.append((np.isclose(fresh['minimum_depth'], camera[:, 2].min(), atol=1e-12, rtol=0),
                   'independent model depth'))
    difference = None
    if expected is not None:
        difference = abs(fresh['objective']-float(expected['objective']))
        checks.append((difference <= 1e-8*max(1., abs(fresh['objective']), abs(expected['objective'])),
                       'independent native scalar energy'))
        original, stored = INDEPENDENT.cost_buckets(fresh['native_costs']), INDEPENDENT.cost_buckets(expected['native_costs'])
        checks.append((original.keys() == stored.keys() and all(np.isclose(original[k], stored[k], atol=1e-8, rtol=1e-8)
            for k in original), 'independent native factor identities and costs once'))
        checks.append((fresh['accepted'] == expected['accepted'], 'independent old/new hardguard acceptance'))
    finite_difference, gradient_maximum_difference = None, None
    if expected_gradient is not None:
        # The same declared parameterization, with independently perturbed
        # scalar energy rather than the runtime analytic derivative.
        p = np.r_[Rotation.from_matrix(R).as_rotvec(), t/raw.scale]
        finite_difference = np.empty(6)
        for axis in range(6):
            step = 1e-6*max(1., abs(p[axis]))
            sides = []
            for direction in (-1., 1.):
                candidate = p.copy(); candidate[axis] += direction*step
                rotation = Rotation.from_rotvec(candidate[:3]).as_matrix()
                translation = candidate[3:]*raw.scale
                sides.append(INDEPENDENT.recompute_native(task, family, rotation, translation,
                    plan, gamma, raw=raw)['objective'])
            finite_difference[axis] = (sides[1]-sides[0])/(2*step)
        saved = np.asarray(expected_gradient, float)
        if saved.shape != (6,):
            raise ValueError('Six-vector runtime analytic gradient required')
        gradient_maximum_difference = float(np.max(np.abs(saved-finite_difference)))
        checks.append((np.allclose(saved, finite_difference, atol=5e-4, rtol=3e-5),
                       'independent scalar central-difference gradient'))
    checks.append((INDEPENDENT._digest(dict(packet=packet, family=family, plan=plan)) == before,
                   'positive packet/family/plan immutable'))
    return dict(checks=[dict(passed=bool(ok), reason=reason) for ok, reason in checks],
        failure_count=sum(not bool(ok) for ok, _ in checks), family_check=family_check,
        objective=fresh['objective'], native_costs=fresh['native_costs'], accepted=fresh['accepted'],
        scalar_absolute_difference=difference, finite_difference_gradient=None if finite_difference is None else finite_difference.tolist(),
        gradient_maximum_absolute_difference=gradient_maximum_difference,
        actual_support=fresh['actual_support'], minimum_depth=fresh['minimum_depth'],
        source_provenance_or_physical_edge_identity_or_global_optimum_certified=False,
        V32_consumer_planner_or_objective_called_by_replay=False,
        old_V17_guard_primitives_and_independent_V29_scalar_kernel_reused=True)
