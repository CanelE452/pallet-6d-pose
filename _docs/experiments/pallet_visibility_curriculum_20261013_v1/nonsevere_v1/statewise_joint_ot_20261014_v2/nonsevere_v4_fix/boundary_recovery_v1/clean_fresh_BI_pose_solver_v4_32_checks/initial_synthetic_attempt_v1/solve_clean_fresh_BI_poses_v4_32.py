"""Pure four-arm solver: typed V32 cores and fresh geometric seeds only.

There is no baseline/e/history/task argument and no IO. The caller must bind
independent original inputs, the new CAL models and all unchanged guards.
The existing scalar native/O/S/C math is retained; proposal and start sources
are replaced entirely. Comparison with e belongs to the parent after workers.
"""
from copy import deepcopy
import time

import numpy as np

from . import clean_fresh_BI_runtime_v4_32 as FRESH
from . import geometry_guided_edge_problem_v4_32 as CORE
from . import bounded_geometric_priors_problem_v4_22 as BOUNDED
from . import coarse_prior_containment_problem_v4_25 as CONTAIN
from . import evaluate_geometry_guided_edges_v4_29 as SCALAR

ARMS = ('CLEAN_FRESH_BI_ALL_BI_NATIVE', 'CLEAN_FRESH_BI_ALL_BI_CONTAINMENT',
        'CLEAN_FRESH_BI_SOURCE_NATIVE', 'CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
POLICY_ARMS = dict(zip(CORE.SOURCE_POLICIES, (ARMS[:2], ARMS[2:])))
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')


def _abstention(clean):
    return dict(schema='CLEAN_FRESH_BI_POSE_SOLVE_V32', identifiers=deepcopy(clean['identifiers']),
        arms={name: dict(accepted=False, pose_available=False, R=None, t=None,
            reason=clean['reason'], row_retained=True, old_method_or_baseline_fallback_used=False)
              for name in ARMS},
        row_retained=True, invalid_initial_coarse=True, baseline_inputs=0,
        cached_pose_or_external_guide_inputs=0, pose_accuracy_or_global_uniqueness_claim=False)


def _cores(clean, cores):
    if set(cores) != set(CORE.SOURCE_POLICIES):
        raise ValueError('Both separately typed V32 source policies required')
    for policy, by_family in cores.items():
        if set(by_family) != set(FAMILIES):
            raise ValueError('Both fresh W/D hypotheses required uniformly')
        for name, core in by_family.items():
            if not isinstance(core, CORE.GeometryGuidedEdgeProblem):
                raise ValueError('Authenticated typed V32 native core required; old core rejected')
            if (core.plan['source_policy'] != policy
                    or core.plan['observation_family'] != CORE.OBSERVATION_FAMILY
                    or core.plan['selection_profile'] != CORE.SELECTION_PROFILE):
                raise ValueError('Current V32 core policy/family/profile mismatch')
            FRESH._problem_binding(core, clean, name)


def solve_clean_fresh_BI_poses(clean, native_cores, *, o_regions, o_parameters,
        original_S_prepared, outline_observation, outline_configuration):
    """Evaluate every fresh contender and continuously refine fixed fresh starts.

    A source-policy/family's six diverse native starts plus independent coarse
    are fixed once from its new V32 native core and shared by native and C.
    Native endpoints also enter the corresponding C contenders. No old bank,
    posterior prior, old reference-start problem or old evaluator run is used.
    """
    FRESH._validate(clean)
    if not clean['ready']:
        return _abstention(clean)  # Do not inspect other supplied payloads.
    _cores(clean, native_cores)
    before = clean['clean_task_sha256']
    started = time.monotonic()
    cfg = CONTAIN.CONFIG._outline_config(outline_configuration)
    shape_weight = float(cfg['outline_factor_weight']) if outline_observation.get('supported') else 0.
    packets = {policy: {name: FRESH.select_clean_fresh_native_starts(core, clean, name)
                        for name, core in by_family.items()} for policy, by_family in native_cores.items()}
    arms = {}
    for policy, policy_arms in POLICY_ARMS.items():
        native_endpoints = {name: [] for name in FAMILIES}
        for containment, arm in zip((False, True), policy_arms):
            candidates, details, searches, plans = [], {}, {}, {}
            for name in FAMILIES:
                core = native_cores[policy][name]
                plans[name] = deepcopy(core.plan)
                family = clean['families'][name]
                s_prior = (None if original_S_prepared is None else
                    BOUNDED.S.rebind_bounded_visibility_trial_geometry(original_S_prepared, family['native_xyz']))
                bounded = BOUNDED.BoundedGeometricPriorsProblem(core, o_regions, o_parameters, s_prior)
                problem = CONTAIN.ContinuousContainmentProblem(bounded, outline_observation, cfg) if containment else bounded

                def add(R, t, token, origin):
                    checked = problem.evaluate(R, t)
                    shape = float(checked.get('global_containment_cost', 0.)) if containment else 0.
                    candidates.append(dict(name=token, family=name, accepted=checked['accepted'],
                        all_joint_constraints_satisfied=checked['accepted'],
                        native_costs=SCALAR._bridge(checked.get('native_costs', [])),
                        regularizer_costs=checked.get('regularizer_costs', []),
                        global_mask_cost=shape, origin=origin))
                    if checked['accepted']:
                        expected = checked['native_observation_objective']+checked['regularizer_objective']
                        expected += shape_weight*shape if containment else 0.
                        if not np.isclose(expected, checked['objective'], atol=1e-9, rtol=1e-10):
                            raise ValueError('Current scalar optimizer/selector cost mismatch')
                        details[token] = dict(evaluation=checked, family=family, origin=origin)

                for proposal in clean['fresh_seed_banks'][name]['proposals']:
                    add(proposal['R'], proposal['t'], name+'/'+proposal['name'], proposal['source'])
                if containment:
                    for i, endpoint in enumerate(native_endpoints[name]):
                        add(endpoint['R'], endpoint['t'], name+'/SAME_POLICY_NATIVE_'+str(i), 'SAME_POLICY_FRESH_NATIVE_ENDPOINT')
                refined = FRESH.refine_clean_fresh_native_problem(problem, clean, name, packets[policy][name])
                for i, solution in enumerate(refined['solutions']):
                    add(solution['R'], solution['t'], name+'/REFINED_'+str(i),
                        'CONTINUOUS_FRESH_CONTAINMENT_ENDPOINT' if containment else 'CONTINUOUS_FRESH_NATIVE_ENDPOINT')
                    if not containment:
                        native_endpoints[name].append(dict(R=solution['R'], t=solution['t']))
                searches[name] = dict(search=refined['search'], start_selection=refined['start_selection'],
                    candidates=refined['candidates'], start_packet_sha256=packets[policy][name]['start_packet_sha256'])
            selection = SCALAR.select_candidates(clean['states'], candidates,
                containment_weight=shape_weight if containment else 0.)
            token = selection['selected_name']
            if token is None:
                family = clean['families']['ORIGINAL_WD']
                coarse = family['coarse_pose']
                selected = dict(accepted=False, R=coarse['R'], t=coarse['t'], objective=None,
                    native_costs=[], regularizer_costs=[], reason='no_current_feasible_fresh_or_refined_candidate')
                origin = 'OWN_INDEPENDENT_INITIAL_COARSE_FALLBACK'
            else:
                detail = details[token]
                family, selected, origin = detail['family'], detail['evaluation'], detail['origin']
                cost = next(row for row in selection['candidates'] if row['name'] == token)['total_energy']
                if not np.isclose(cost, selected['objective'], atol=1e-9, rtol=1e-10):
                    raise ValueError('Selected total differs from its current checked objective')
            R = np.asarray(selected['R']); t = np.asarray(selected['t'])
            xyz = np.asarray(family['native_xyz']); Q = np.asarray(family['physical_registration_Q'])
            arms[arm] = dict(method=arm, accepted=bool(selected['accepted']), pose_available=True,
                R=R.tolist(), t=t.tolist(), R_physical=(R@Q).tolist(),
                centroid=(R@xyz.mean(axis=0)+t).tolist(), WD_family=family['name'],
                physical_registration_Q=Q.tolist(), source_policy=policy,
                solution=selected, candidate_selection=selection,
                selected_endpoint_origin=origin, new_geometry_edge_cost_plans=plans,
                family_refinements=searches,
                fallback_audit=dict(own_independent_original_coarse_only=not selected['accepted'],
                    old_method_or_baseline_fallback_used=False, cached_pose_inputs=0),
                same_policy_native_endpoints_retained=containment,
                no_new_full_2D_KPs=True, S_image_or_2D_queries=0, U_regions_only=True)
    FRESH._validate(clean)
    if clean['clean_task_sha256'] != before:
        raise ValueError('Clean original physics/seeds changed during solving')
    return dict(schema='CLEAN_FRESH_BI_POSE_SOLVE_V32', identifiers=deepcopy(clean['identifiers']),
        arms=arms, row_retained=True, elapsed_seconds=time.monotonic()-started,
        legacy_evaluator_run_or_refiner_wrapper_used=False,
        frozen_native_numerical_refinement_kernel_reused=True,
        old_reference_start_problem_used=False,
        cached_pose_or_external_guide_inputs=0, baseline_inputs=0,
        S_only_model_prior_and_PnP_projection=True, U_only_approximate_regions=True,
        pose_accuracy_or_global_uniqueness_claim=False)
