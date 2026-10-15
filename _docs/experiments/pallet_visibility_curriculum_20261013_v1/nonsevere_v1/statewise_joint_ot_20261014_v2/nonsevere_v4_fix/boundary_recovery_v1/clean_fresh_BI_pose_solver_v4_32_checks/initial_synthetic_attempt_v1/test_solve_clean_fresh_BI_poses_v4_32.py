"""Genuine clean family/factory/AUTH32/core32 integration; synthetic only."""
from copy import deepcopy
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from . import solve_clean_fresh_BI_poses_v4_32 as M
from . import prepare_geometry_guided_native_evidence_v4_32 as FACTORY
from . import geometry_guided_edge_authentication_v4_32 as AUTH
from . import common as C
from .test_prepare_geometry_guided_native_evidence_v4_32 import fixture as positive_fixture


class Poison:
    def __iter__(self): raise AssertionError('Forbidden payload iterated')
    def __getitem__(self, key): raise AssertionError('Forbidden payload indexed')
    def get(self, *args): raise AssertionError('Forbidden payload queried')
    def __array__(self, *args, **kwargs): raise AssertionError('Forbidden payload converted')


def fixture():
    _, inputs, metadata, _ = positive_fixture(partial=True)
    inputs['config'].update(max_joint_iterations=3, optimizer_ftol=1e-8)
    family = inputs['family']; F = M.FRESH
    clean = F.prepare_clean_fresh_BI_task(family['native_xyz'], inputs['K'], inputs['states'],
        inputs['initial_self_hidden'], inputs['visible_measurements'], family['coarse_pose'],
        family['physical_registration_Q'], initial_source_binding=inputs['initial_source_binding'],
        identifiers=dict(id='SYNTHETIC:CAL', partition='CAL', session='SYNTHETIC'))
    cores = {policy: {} for policy in M.CORE.SOURCE_POLICIES}
    for policy in cores:
        for name, family in clean['families'].items():
            base = FACTORY.native_baseline(**{**inputs, 'family': family})
            bundle = AUTH.authenticate_geometry_guided_edges(base, metadata['observation'],
                initial_geometry=metadata['initial_geometry'], image_binding=metadata['image_binding'],
                normal_variance_component_px2=0., angular_variance_component_sine2=0.,
                noise_provenance=dict(fit_partition='CAL', observation_family=M.CORE.OBSERVATION_FAMILY,
                    noise_population=M.CORE.POPULATION, selection_profile=M.CORE.SELECTION_PROFILE,
                    raw_observation_generation=M.CORE.RAW_OBSERVATION_GENERATION,
                    mode='ZERO_COMPONENT_PREFIT_SYNTHETIC_NOT_FITTED'), calibration_preparation=True)
            cores[policy][name] = M.CORE.GeometryGuidedEdgeProblem(base, bundle,
                source_policy=policy, tangent_gamma=.2)
    original = clean['families']['ORIGINAL_WD']
    R, t = np.asarray(original['coarse_pose']['R']), np.asarray(original['coarse_pose']['t'])
    model = dict(schema=M.BOUNDED.S.MODEL_SCHEMA, fit_partition='CAL',
        fixed_pseudocounts=M.BOUNDED.S.PSEUDOCOUNTS, geometry_tolerance_deg=2., epsilon=.3)
    model['model_digest_sha256'] = M.BOUNDED.S.model_digest(model)
    s_prior = M.BOUNDED.S.prepare_bounded_coarse_visibility_prior(original['native_xyz'],
        R, t, inputs['initial_self_hidden'], model)
    # A synthetic translated target mask makes the containment cost active;
    # no real RGB, stored mask, learned model or actual label is decoded.
    projection, _ = M.CORE.V.project_native(original['native_xyz'], R, t, inputs['K'])
    mask = np.zeros(inputs['image_hw'], np.uint8)
    hull = cv2.convexHull((np.asarray(projection) + [12., 0.]).astype(np.float32))
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    outline = M.CONTAIN.CONFIG.observe_global_outline(mask, inputs['image_hw'])
    arguments = dict(o_regions=[], o_parameters=dict(epsilon=.2, huber_scale=3.),
        original_S_prepared=s_prior, outline_observation=outline,
        outline_configuration=dict(outline_factor_weight=1.))
    return clean, cores, arguments


@pytest.fixture(scope='module')
def integrated():
    clean, cores, arguments = fixture()
    clean_before, arguments_before = deepcopy(clean), deepcopy(arguments)
    core_before = {p: {n: deepcopy((c.plan, c.base.prepartial_bound_source_archives,
        c.guard_boundary, c.cfg)) for n, c in by_family.items()} for p, by_family in cores.items()}
    modules = (M.FRESH, M.CORE, AUTH, FACTORY, M.CORE.PARENT, M.SCALAR)
    globals_before = {id(m): dict(vars(m)) for m in modules}
    sources = {m.__file__: Path(m.__file__).read_bytes() for m in modules}
    result = M.solve_clean_fresh_BI_poses(clean, cores, **arguments)
    assert clean == clean_before and arguments == arguments_before
    for p, by_family in cores.items():
        for n, c in by_family.items():
            assert C.finite((c.plan, c.base.prepartial_bound_source_archives, c.guard_boundary, c.cfg)) == C.finite(core_before[p][n])
    for m in modules:
        assert all(vars(m)[k] is v for k, v in globals_before[id(m)].items())
        assert sources[m.__file__] == Path(m.__file__).read_bytes()
    return clean, cores, arguments, result


def test_exact_four_uniform_arms_no_E_numeric_input_and_no_legacy_wrapper(integrated):
    clean, cores, args, result = integrated
    assert tuple(result['arms']) == M.ARMS and len(result['arms']) == 4
    assert not result['legacy_evaluator_run_or_refiner_wrapper_used']
    assert result['frozen_native_numerical_refinement_kernel_reused']
    assert not result['old_reference_start_problem_used']
    assert result['baseline_inputs'] == result['cached_pose_or_external_guide_inputs'] == 0
    assert result['row_retained'] and not result['pose_accuracy_or_global_uniqueness_claim']
    assert set(cores) == set(M.CORE.SOURCE_POLICIES)
    assert all(set(by_family) == set(M.FAMILIES) for by_family in cores.values())
    forbidden = {'task', 'baseline', 'e', 'proposals', 'initializations', 'pose_guides', 'initial_solutions'}
    assert not forbidden & set(inspect.signature(M.solve_clean_fresh_BI_poses).parameters)
    with pytest.raises(TypeError): M.solve_clean_fresh_BI_poses(clean, cores, **args, baseline=Poison())


def test_same_fresh_six_diverse_packet_shared_by_native_and_C_and_native_endpoints_retained(integrated):
    clean, cores, _, result = integrated
    for policy, (native_name, containment_name) in M.POLICY_ARMS.items():
        native, containment = result['arms'][native_name], result['arms'][containment_name]
        for family in M.FAMILIES:
            a, b = native['family_refinements'][family], containment['family_refinements'][family]
            assert a['start_packet_sha256'] == b['start_packet_sha256']
            assert a['start_selection'] == b['start_selection']
            assert len(a['start_selection']['selected_start_names']) <= 7
            assert a['start_selection']['configuration'] == M.FRESH.START_CONFIGURATION
            assert a['search']['same_clean_start_packet_sha256'] == b['search']['same_clean_start_packet_sha256']
        old_candidates = native['candidate_selection']['native_authentication_audit']['candidates']
        new_candidates = containment['candidate_selection']['native_authentication_audit']['candidates']
        endpoints = [r for r in old_candidates if '/REFINED_' in r['name']]
        retained = [r for r in new_candidates if '/SAME_POLICY_NATIVE_' in r['name']]
        assert len(retained) == len(endpoints)
        assert containment['same_policy_native_endpoints_retained']


def test_all_fresh_raw_contenders_are_rechecked_in_every_policy_arm(integrated):
    clean, _, _, result = integrated
    expected = {family+'/'+p['name'] for family in M.FAMILIES for p in clean['fresh_seed_banks'][family]['proposals']}
    for arm in result['arms'].values():
        audit = arm['candidate_selection']['native_authentication_audit']
        # Hierarchical authentication keeps rejected rows in excluded audit.
        names = {r['name'] for r in audit['candidates']}
        names |= {r['name'] for r in audit['excluded_candidates']}
        assert expected <= names
        assert arm['fallback_audit']['cached_pose_inputs'] == 0
        assert not arm['fallback_audit']['old_method_or_baseline_fallback_used']


def test_selected_current_guard_and_scalar_parts_sum_native_O_S_C_once(integrated):
    _, cores, args, result = integrated
    for policy, names in M.POLICY_ARMS.items():
        for is_containment, name in enumerate(names):
            arm = result['arms'][name]
            if not arm['accepted']:
                continue
            core = cores[policy][arm['WD_family']]
            family = arm['WD_family']
            s = M.BOUNDED.S.rebind_bounded_visibility_trial_geometry(args['original_S_prepared'], core.xyz)
            bounded = M.BOUNDED.BoundedGeometricPriorsProblem(core, args['o_regions'], args['o_parameters'], s)
            problem = M.CONTAIN.ContinuousContainmentProblem(bounded, args['outline_observation'], args['outline_configuration']) if is_containment else bounded
            checked = problem.evaluate(arm['R'], arm['t'])
            saved = arm['solution']
            assert checked['accepted'] and saved['all_original_hard_guards_recomputed']
            assert checked['objective'] == pytest.approx(saved['objective'], abs=1e-10)
            native = sum(r['weighted_robust_cost'] for r in checked['native_costs'])
            priors = sum(r['weighted_cost'] for r in checked['regularizer_costs'])
            assert [r['kind'] for r in checked['regularizer_costs']] == list(M.BOUNDED.REGULARIZER_KINDS)
            shape = checked.get('global_containment_cost', 0.)
            assert checked['objective'] == pytest.approx(native + priors + is_containment*shape, abs=1e-9)
            assert saved['native_observation_objective'] == pytest.approx(native, abs=1e-10)
            selected = next(r for r in arm['candidate_selection']['candidates'] if r['name'] == arm['candidate_selection']['selected_name'])
            assert selected['total_energy'] == pytest.approx(checked['objective'], abs=1e-10)
            assert not arm['candidate_selection']['native_Pareto_audit_not_prefilter'] is False
            assert not arm['candidate_selection']['old_full_mask_cost_or_numeric_tie_break_used']
            p = M.CORE.V._pose_vector(np.asarray(arm['R']), np.asarray(arm['t']), core.scale)
            for side in core.sides:
                actual = core.constraints(p, side); wrapped = problem.constraints(p, side)
                assert all(np.array_equal(a, b) for a, b in zip(actual, wrapped))


def test_invalid_independent_coarse_retains_four_abstentions_before_other_payload():
    clean, _, _ = fixture()
    original = clean['families']['ORIGINAL_WD']
    bad = deepcopy(original['coarse_pose']); bad['source'] = 'SAME_FROZEN_E'
    abstain = M.FRESH.prepare_clean_fresh_BI_task(original['native_xyz'], clean['spec']['K'], clean['states'],
        clean['initial_self_hidden'], [Poison()], bad, original['physical_registration_Q'],
        initial_source_binding=clean['initial_source_binding'])
    result = M.solve_clean_fresh_BI_poses(abstain, Poison(), o_regions=Poison(), o_parameters=Poison(),
        original_S_prepared=Poison(), outline_observation=Poison(), outline_configuration=Poison())
    assert len(result['arms']) == 4 and result['invalid_initial_coarse']
    assert all(not a['pose_available'] and not a['accepted'] and a['row_retained'] for a in result['arms'].values())


@pytest.mark.parametrize('bad', ['old_core', 'bad_physics', 'wrong_policy', 'missing_family'])
def test_old_core_or_wrong_positive_physics_rejects_before_start_selection(bad, monkeypatch):
    clean, cores, args = fixture()
    policy = M.CORE.SOURCE_POLICIES[0]; core = cores[policy]['ORIGINAL_WD']
    if bad == 'old_core': cores[policy]['ORIGINAL_WD'] = core.base
    elif bad == 'bad_physics': core.K = core.K.copy(); core.K[0, 0] += 1.
    elif bad == 'wrong_policy': core.plan['source_policy'] = 'RGB_NONFALLBACK_PRESERVE'
    else: del cores[policy]['SWAPPED_WD']
    def forbidden(*args, **kwargs): raise AssertionError('Invalid core reached fresh start selection')
    monkeypatch.setattr(M.FRESH, 'select_clean_fresh_native_starts', forbidden)
    with pytest.raises(ValueError): M.solve_clean_fresh_BI_poses(clean, cores, **args)


def test_no_feasible_contender_fallback_is_exact_original_independent_coarse(monkeypatch):
    clean, cores, args = fixture()
    original = deepcopy(clean['families']['ORIGINAL_WD']['coarse_pose'])
    def rejected(*a, **kw):
        return dict(accepted=False, objective=None, native_costs=[], reason='SYNTHETIC_CURRENT_GUARD_REJECTION')
    for by_family in cores.values():
        for core in by_family.values(): monkeypatch.setattr(core, 'evaluate', rejected)
    # Real continuous math was exercised above. This forced-rejection path
    # only verifies fallback bookkeeping, without pointless optimizer work.
    def none(problem, clean, name, packet):
        return dict(solutions=[], candidates=[], search=dict(same_clean_start_packet_sha256=packet['start_packet_sha256']),
                    start_selection=packet['selection'])
    monkeypatch.setattr(M.FRESH, 'refine_clean_fresh_native_problem', none)
    result = M.solve_clean_fresh_BI_poses(clean, cores, **args)
    for arm in result['arms'].values():
        assert not arm['accepted'] and arm['pose_available']
        assert arm['selected_endpoint_origin'] == 'OWN_INDEPENDENT_INITIAL_COARSE_FALLBACK'
        assert arm['WD_family'] == 'ORIGINAL_WD'
        assert arm['R'] == original['R'] and arm['t'] == original['t']
        assert arm['fallback_audit']['own_independent_original_coarse_only']
        assert not arm['fallback_audit']['old_method_or_baseline_fallback_used']
