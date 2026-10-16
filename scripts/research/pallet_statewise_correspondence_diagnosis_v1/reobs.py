"""Same frozen reobservations with original multi-start or one E0 start.

No reference, annotation, image decoder, NN model loader, or selector training.
The SINGLE objective and scalar feasibility verification are the original
BI solver expressions. Only its proposal policy is replaced with one SLSQP
run from E0 in the unchanged registered native/physical basis.
"""
from copy import deepcopy
import time

import numpy as np
from scipy.optimize import minimize

from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as J
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_jacobian_v2 as A
from .common import digest, finite


def packet(observation, coarse_pose, K, image_hw):
    """Copy the actual E-after observations without refining their provenance."""
    return dict(observation=deepcopy(observation), coarse_pose=deepcopy(coarse_pose),
                K=deepcopy(K), image_hw=list(image_hw), uses_e_initialization=True)


def _problem(p):
    obs = p['observation']
    xyz = np.asarray(obs['initial_geometry']['native_xyz'], float)
    expected = RT.C.cuboid(*p['coarse_pose']['cf_extents'])
    if not np.array_equal(xyz, expected):
        raise ValueError('Original reobservation basis changed')
    states = ['U' if s in ('O', 'T') else s for s in obs['states']]
    cfg = deepcopy(RT.SOLVER_CONFIG)
    x, K, states, measured, domains, hw = J._inputs(
        xyz, p['K'], states, deepcopy(obs['frozen_visible_measurements']), [], p['image_hw'], cfg)
    if cfg['protect_seed_projection'] or domains:
        raise ValueError('Single wrapper requires the actual fixed no-seed-protection BI contract')
    scale = max(float(np.linalg.norm(np.ptp(x, axis=0))), 1e-3)
    initial = p['coarse_pose']
    R, t = np.asarray(initial['R_cf']), np.asarray(initial['centroid'])
    q, _ = J.project_native(x, R, t, K)
    objective = J._cached_value_jacobian(lambda v: A.visible_objective_and_gradient(
        v, x, K, measured, scale, cfg['huber_scale']))
    constraint = J._cached_value_jacobian(lambda v: A.constraint_values_and_jacobian(
        v, x, K, measured, domains, (), q, hw, scale, cfg))
    return dict(xyz=x, K=K, states=states, measurements=measured, domains=domains,
                image_hw=hw, cfg=cfg, scale=scale, reference=q,
                initial_vector=J._pose_vector(R, t, scale), objective_pair=objective,
                constraint_pair=constraint)


def active_values(p, vector=None):
    problem = _problem(p)
    v = problem['initial_vector'] if vector is None else np.asarray(vector, float)
    return finite(dict(objective=problem['objective_pair'](v)[0],
                       constraints=problem['constraint_pair'](v)[0]))


def solve_multi(p):
    start = time.monotonic()
    result = RT.solve(deepcopy(p['observation']), deepcopy(p['coarse_pose']),
                      deepcopy(p['K']), p['image_hw'])
    result.update(method='REOBS_MULTI', uses_e_initialization=True,
                  packet_sha256=digest(p), elapsed_seconds=time.monotonic()-start,
                  observation_package='E_AFTER_REOBSERVED_BI',
                  proposal_policy='UNCHANGED_ORIGINAL_BI_MULTISTART',
                  optimizer_calls=len(result['solution'].get('candidates', [])),
                  objective_contract='ORIGINAL_BI_ONLY_WITH_FIXED_COVARIANCE_AND_SUPPORT')
    return finite(result)


def solve_single(p):
    start = time.monotonic()
    before = digest(p)
    problem = _problem(p)
    cfg, measured = problem['cfg'], problem['measurements']
    answer = None
    outcome = None
    accepted = False
    reason = None
    calls = 0
    rank = None
    # Match the preexisting unresolved cardinality checks; no fabricated points.
    if len(measured) < 2:
        reason = 'fewer_than_two_corrected_BI_for_joint_pose'
    elif len(measured) == 2:
        reason = 'fewer_than_three_corrected_BI_for_pose_seed'
    else:
        calls = 1
        obj, con = problem['objective_pair'], problem['constraint_pair']
        try:
            answer = minimize(lambda v: obj(v)[0], problem['initial_vector'], method='SLSQP',
                              jac=lambda v: obj(v)[1].copy(),
                              constraints=[dict(type='ineq', fun=lambda v: con(v)[0].copy(),
                                                jac=lambda v: con(v)[1].copy())],
                              options=dict(maxiter=int(cfg['max_joint_iterations']),
                                           ftol=cfg['optimizer_ftol'], disp=False))
            R, t = J._pose(answer.x, problem['scale'])
            q, depth = J.project_native(problem['xyz'], R, t, problem['K'])
            values = J._constraint_values(answer.x, problem['xyz'], problem['K'], measured, [], (),
                                         problem['reference'], problem['image_hw'], problem['scale'], cfg)
            accepted = bool(np.isfinite(values).all() and np.isfinite(q).all()
                            and values.min() >= -cfg['feasibility_tolerance'] and J._valid_rotation(R)
                            and depth.min() >= cfg['positive_depth_m']-1e-10)
            if accepted:
                jac = A.visible_residual_and_jacobian(answer.x, problem['xyz'],
                    problem['K'], measured, problem['scale'])[1]
                active = np.flatnonzero(values <= 1e-5)
                joint = np.vstack([jac, con(answer.x)[1][active]]) if len(active) else jac
                rank = dict(visible=J._rank(jac), visible_and_active_joint=J._rank(joint),
                            active_constraint_count=len(active))
            outcome = dict(R=R, t=t, projected_native=q, objective=obj(answer.x)[0],
                           joint_feasible=accepted, minimum_constraint_slack=values.min(),
                           optimizer_success=bool(answer.success), optimizer_message=str(answer.message),
                           iterations=int(answer.nit), initialization='EXACT_E0_SINGLE_START', branch=0)
        except (ValueError, np.linalg.LinAlgError) as exc:
            outcome = dict(initialization='EXACT_E0_SINGLE_START', branch=0,
                           joint_feasible=False, error=str(exc))
        if not accepted:
            reason = 'all_searched_joint_branches_infeasible_or_not_converged'
    coarse = deepcopy(p['coarse_pose'])
    if accepted:
        Q = np.asarray(coarse['R_cf']).T @ np.asarray(coarse['R_physical'])
        actual = dict(available=True, R_cf=outcome['R'], R_physical=outcome['R'] @ Q,
                      centroid=outcome['t'], cf_extents=coarse['cf_extents'],
                      source='SINGLE_START_SAME_BI_OBJECTIVE')
    else:
        # RT._bi_only's existing no-corrected-seed branch, not a new guard.
        actual = dict(coarse, source='ORIGINAL_PNP_FALLBACK')
    assert digest(p) == before
    return finite(dict(method='REOBS_SINGLE', packet_sha256=before,
        uses_e_initialization=True, actual_pose=actual, accepted=accepted, fallback=not accepted,
        states=problem['states'], observed_states_preserved=p['observation']['states'],
        frozen_visible_measurements=p['observation']['frozen_visible_measurements'],
        solver_configuration=cfg, input_native_ids=[m['native_corner'] for m in measured],
        optimizer_calls=calls, elapsed_seconds=time.monotonic()-start,
        observation_package='E_AFTER_REOBSERVED_BI',
        objective_contract='ORIGINAL_BI_ONLY_WITH_FIXED_COVARIANCE_AND_SUPPORT',
        solution=dict(accepted=accepted, fallback_reason=reason,
                      candidates=[] if outcome is None else [outcome], rank=rank,
                      ambiguity=dict(detected=bool(rank and rank['visible_and_active_joint']['rank'] < 6),
                                     alternatives=[], global_uniqueness_proved=False,
                                     single_start_does_not_test_global_ambiguity=True),
                      search=dict(initializations=1 if calls else 0, global_search=False),
                      provenance=dict(S_image_terms=0, OT_used_as_precise_correspondences=False,
                                      original_objective_and_constraints=True,
                                      initial_pose_prior=False, arbitrary_motion_limits=False)),
        single_start_does_not_guarantee_same_basin=True,
        single_start_does_not_prove_global_uniqueness=True))
