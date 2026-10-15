"""Saved-start continuous SE3 fitting of authenticated native image factors.

The native observation problem owns all likelihoods and current hard guards.
There is no RGB/label IO, mask objective, coarse pose prior or new P3P search.
Old proposal acceptance and scores are never trusted. Bounded initializations
are chosen by current native energy and pose diversity, with the original
coarse proposal always included. Every endpoint is independently rechecked.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import native_edge_problem_v4_17 as N


DEFAULT_CONFIG = dict(maximum_native_ranked_starts=6,
    minimum_rotation_separation_deg=2., minimum_translation_separation_ratio=.01)


def _configuration(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if set(cfg) != set(DEFAULT_CONFIG):
        raise ValueError('Unknown native refinement initialization option')
    count = cfg['maximum_native_ranked_starts']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Positive integer native start budget required')
    for name in ('minimum_rotation_separation_deg', 'minimum_translation_separation_ratio'):
        cfg[name] = float(cfg[name])
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError('Finite nonnegative pose diversity threshold required')
    return cfg


def _same_basin(a, b, scale, cfg):
    angle = np.rad2deg(Rotation.from_matrix(np.asarray(a['R']) @ np.asarray(b['R']).T).magnitude())
    displacement = np.linalg.norm(np.asarray(a['t']) - np.asarray(b['t'])) / scale
    return (angle <= cfg['minimum_rotation_separation_deg']
        and displacement <= cfg['minimum_translation_separation_ratio'])


def choose_native_starts(problem, proposals, coarse_pose, *, configuration=None):
    """Fresh feasible native minima, separated basins, plus own coarse pose.

    The explicit cap is an engineering compute budget, not a global-optimality
    claim. Even rejected proposals can enter later through their own coarse
    initialization; old scores/acceptance/GT are never inspected.
    """
    cfg = _configuration(configuration)
    eligible = []; rejected = []; seen = set()
    for index, proposal in enumerate(proposals):
        checked = problem.evaluate(proposal['R'], proposal['t'])
        name = proposal.get('name', 'SAVED_' + str(index))
        if not checked['accepted']:
            rejected.append(dict(name=name, reason=checked.get('reason'),
                minimum_joint_slack=checked.get('minimum_joint_slack')))
            continue
        R, t = np.asarray(checked['R']), np.asarray(checked['t'])
        key = tuple(np.round(np.r_[R.ravel(), t], 12))
        if key in seen:
            continue
        seen.add(key)
        eligible.append(dict(name=name, R=R, t=t, native_objective=checked['objective']))
    eligible.sort(key=lambda s: (s['native_objective'], s['name']))
    starts = []
    for start in eligible:
        if not any(_same_basin(start, other, problem.scale, cfg) for other in starts):
            starts.append(start)
            if len(starts) == cfg['maximum_native_ranked_starts']:
                break
    # Coarse remains an optimization initialization, never an accepted result
    # unless its own fresh hard guards and objective actually pass.
    R = np.asarray(coarse_pose['R'], float); t = np.asarray(coarse_pose['t'], float)
    if R.shape != (3, 3) or t.shape != (3,) or not N.V._valid_rotation(R) or not np.isfinite(t).all():
        raise ValueError('Own finite proper initial coarse pose required')
    coarse = dict(name='OWN_INITIAL_COARSE', R=R, t=t, native_objective=None)
    if not any(N.V._same_pose((R, t), (s['R'], s['t']), problem.scale) for s in starts):
        starts.append(coarse)
    return starts, dict(configuration=cfg, fresh_feasible_saved_candidates=len(eligible),
        rejected_saved_candidates=rejected, selected_start_names=[s['name'] for s in starts],
        coarse_always_considered=True, no_old_acceptance_score_or_GT_filter=True,
        global_optimality_proved=False)


def refine_native_edge_pose(problem, proposals, coarse_pose, *, configuration=None):
    """Optimize every retained start and original T-union branch uniformly."""
    starts, selection = choose_native_starts(problem, proposals, coarse_pose,
        configuration=configuration)
    records = []; solutions = []; invalid_evaluations = 0

    def objective(p):
        nonlocal invalid_evaluations
        try:
            value, derivative, _ = problem.objective(p)
            if not np.isfinite(value) or not np.isfinite(derivative).all():
                raise FloatingPointError('Nonfinite native image objective')
            return value, derivative
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # Infeasible SLSQP trial points may cross the camera plane. Supply
            # a smooth depth-restoration penalty there; it is never reported
            # as a valid measurement likelihood or admitted as a candidate.
            invalid_evaluations += 1
            _, depth, _, depth_jacobian = N.J.projection_and_jacobian(
                p, problem.xyz, problem.K, problem.scale)
            violation = np.maximum(0., problem.cfg['positive_depth_m'] - depth)
            if not np.isfinite(depth).all() or not np.isfinite(depth_jacobian).all():
                return 1e20, np.zeros(6)
            return 1e12 + 1e9 * float(violation @ violation), -2e9 * violation @ depth_jacobian

    pair = N.V._cached_value_jacobian(objective)

    def consider(p, name, branch, optimizer=None):
        R, t = N.V._pose(p, problem.scale)
        checked = problem.evaluate(R, t)
        records.append(dict(name=name, branch=branch, R=R.tolist(), t=t.tolist(),
            accepted=checked['accepted'], objective=checked.get('objective'),
            minimum_joint_slack=checked.get('minimum_joint_slack'),
            optimizer_success=None if optimizer is None else bool(optimizer.success),
            iterations=0 if optimizer is None else int(optimizer.nit),
            message='fresh saved proposal' if optimizer is None else str(optimizer.message),
            reason=checked.get('reason')))
        if checked['accepted'] and not any(N.V._same_pose((R, t),
                (np.asarray(s['R']), np.asarray(s['t'])), problem.scale) for s in solutions):
            solutions.append(dict(**checked, initialization=name, branch=branch))

    for branch, sides in enumerate(problem.sides):
        constraint = N.V._cached_value_jacobian(lambda p: problem.constraints(p, sides))
        for start in starts:
            p = N.V._pose_vector(start['R'], start['t'], problem.scale)
            consider(p, start['name'] + '_UNREFINED', branch)
            try:
                answer = minimize(lambda p: pair(p)[0], p, jac=lambda p: pair(p)[1].copy(),
                    method='SLSQP', constraints=[dict(type='ineq',
                        fun=lambda p: constraint(p)[0].copy(), jac=lambda p: constraint(p)[1].copy())],
                    options=dict(maxiter=problem.cfg['max_joint_iterations'],
                        ftol=problem.cfg['optimizer_ftol'], disp=False))
                consider(answer.x, start['name'], branch, answer)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                records.append(dict(name=start['name'], branch=branch, accepted=False,
                    error=str(error), optimization_exception_does_not_drop_frame=True))
    solutions.sort(key=lambda s: (s['objective'], s['initialization'], s['branch']))
    chosen = deepcopy(solutions[0]) if solutions else dict(accepted=False, objective=None,
        native_costs=[], reason='no_current_feasible_saved_or_continuously_refined_native_candidate')
    return dict(chosen=chosen, solutions=solutions, candidates=records, start_selection=selection,
        search=dict(continuous_SE3=True, new_P3P_or_global_starts=0,
            starts_per_T_branch=len(starts), T_union_branches=len(problem.sides),
            invalid_trial_objective_evaluations=invalid_evaluations,
            invalid_trial_penalty_is_not_a_measurement_likelihood=True,
            all_saved_and_refined_endpoints_rechecked=True, mask_objective=False,
            pose_prior=False, S_or_U_point_costs=False, global_optimality_proved=False))
