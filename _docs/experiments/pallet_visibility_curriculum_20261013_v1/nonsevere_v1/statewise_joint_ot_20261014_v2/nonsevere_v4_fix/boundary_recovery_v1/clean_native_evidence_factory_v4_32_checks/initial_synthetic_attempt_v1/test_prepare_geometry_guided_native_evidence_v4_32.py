"""Explicit clean physics factory; synthetic only, no saved input/label IO."""
from copy import deepcopy
import inspect
import numpy as np
import pytest

from . import prepare_geometry_guided_native_evidence_v4_32 as M
from . import geometry_guided_edge_noise_calibration_v4_32 as NOISE
from . import clean_fresh_BI_runtime_v4_32 as CLEAN
from .test_geometry_guided_edge_authentication_v4_32 import fixture as auth_fixture


def binding():
    return dict(path='SYNTHETIC/initial_source_receipt.json',
        sha256=M.G.INITIAL_SOURCE_CLOSURE_SHA256, verified_by_caller=True,
        independent_initial_geometry_and_BI_q=True)


def fixture(partial=False, two=False):
    base, observation, kwargs, ids = auth_fixture(partial=partial, two=two)
    geometry = kwargs['initial_geometry']; xyz = np.asarray(base.xyz)
    R, t, Q = np.asarray(geometry['R_cf']), np.asarray(geometry['centroid']), np.asarray(base.Q)
    coarse = dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=Q.tolist(),
        R_physical=(R @ Q).tolist(), source=M.FRESH.COARSE_SOURCE,
        independent_of_cached_pose_bank=True)
    family = CLEAN._families(xyz, coarse, Q)['ORIGINAL_WD']
    visible = [deepcopy(a['source_record']) for a in base.prepartial_bound_source_archives.values()]
    # Full native boundary records are saved hardguard input, not new edges.
    native = dict(family=family, K=base.K.tolist(), states=list(base.states),
        visible_measurements=visible, domains=[], image_hw=list(base.image_hw),
        config=dict(prior_weight=5., analytic_jacobian=True),
        boundary_merged=deepcopy(base.boundary), u_regions=[],
        initial_self_hidden=geometry['initial_self_hidden'], initial_source_binding=binding())
    metadata = dict(identifier='SYNTHETIC:CAL', partition='CAL', session='SYNTHETIC',
        initial_geometry=geometry, image_binding=kwargs['image_binding'], observation=observation)
    return base, native, metadata, ids


@pytest.mark.parametrize('partial', [False, True])
def test_same_positive_inputs_preserve_native_value_gradient_all_guards_and_archives(partial):
    _, inputs, _, _ = fixture(partial=partial)
    before = deepcopy(inputs)
    family = inputs['family']
    old = M.G.N.prepare_native_edge_problem(family['native_xyz'], inputs['K'], inputs['states'],
        inputs['visible_measurements'], inputs['domains'], inputs['image_hw'], family['coarse_pose'],
        config={**inputs['config'], 'prior_weight': 0.}, boundary_observations=inputs['boundary_merged'],
        u_regions=inputs['u_regions'], edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    old = M.G.BASELINE.prepare_prepartial_native_problem(old,
        visible_measurements=inputs['visible_measurements'], initial_self_hidden=inputs['initial_self_hidden'])
    new = M.native_baseline(**inputs)
    p = np.array([.1, -.2, .01, .03, -.04, 2.])
    a, ja, ap = old.objective(p); b, jb, bp = new.objective(p)
    assert a == b and np.array_equal(ja, jb) and ap == bp
    for side in old.sides:
        c, jc = old.constraints(p, side); d, jd = new.constraints(p, side)
        assert np.array_equal(c, d) and np.array_equal(jc, jd)
    assert new.cfg['prior_weight'] == 0. and new.requested_prior_weight == 0.
    assert new.prepartial_bound_source_archives == old.prepartial_bound_source_archives
    assert inputs == before


def test_clean_family_matches_private_clean_runtime_and_fresh_starts_need_no_legacy_task(monkeypatch):
    _, inputs, _, _ = fixture()
    coarse = inputs['family']['coarse_pose']
    clean = CLEAN.prepare_clean_fresh_BI_task(inputs['family']['native_xyz'], inputs['K'], inputs['states'],
        inputs['initial_self_hidden'], inputs['visible_measurements'], coarse,
        inputs['family']['physical_registration_Q'], initial_source_binding=binding())
    for name, family in clean['families'].items():
        arguments = {**inputs, 'family': family}
        problem = M.native_baseline(**arguments)
        def forbidden(*args, **kw): raise AssertionError('No historical reference problem')
        monkeypatch.setattr(CLEAN.REFINE, 'reference_start_problem', forbidden, raising=False)
        packet = CLEAN.select_clean_fresh_native_starts(problem, clean, name)
        assert packet['cached_or_baseline_pose_inputs'] == 0
        assert packet['legacy_reference_start_problem_used'] is False


@pytest.mark.parametrize('change', ['family_extra', 'coarse_extra', 'coarse_source', 'imported', 'center', 'Q', 'proof'])
def test_opaque_or_inconsistent_source_fields_reject_before_BI_coordinates(change):
    _, inputs, _, _ = fixture()
    class Poison(dict):
        def __getitem__(self, key): raise AssertionError('BI payload read before source gate')
    inputs['visible_measurements'] = [Poison()]
    if change == 'family_extra': inputs['family']['initializations'] = object()
    elif change == 'coarse_extra': inputs['family']['coarse_pose']['baseline'] = object()
    elif change == 'coarse_source': inputs['family']['coarse_pose']['source'] = 'SAME_FROZEN_E'
    elif change == 'imported': inputs['family']['initializations_imported_from_legacy_task'] = True
    elif change == 'center': inputs['family']['coarse_model_center_camera_xyz'][2] += 1.
    elif change == 'Q': inputs['family']['coarse_pose']['physical_registration_Q'][0][0] += .1
    else: inputs['initial_source_binding']['verified_by_caller'] = False
    with pytest.raises(ValueError): M.native_baseline(**inputs)


def test_only_named_positive_inputs_no_opaque_task_bank_or_external_guide_api():
    forbidden = {'task', 'spec', 'baseline', 'proposals', 'initializations', 'pose_guides', 'initial_solutions'}
    for function in (M.native_baseline, M.prepare_one):
        parameters = inspect.signature(function).parameters
        assert not forbidden & set(parameters)
        assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    _, inputs, _, _ = fixture()
    with pytest.raises(TypeError): M.native_baseline(**inputs, proposals=object())


def test_S_or_unknown_observed_payload_is_skipped_before_copy_and_source_fields():
    _, inputs, _, _ = fixture()
    hidden = next(i for i, s in enumerate(inputs['states']) if s == 'S')
    class StateOnly(dict):
        def __getitem__(self, key):
            if key == 'native_corner': return super().__getitem__(key)
            raise AssertionError('S/U observed field read')
        def __deepcopy__(self, memo): raise AssertionError('S/U payload copied')
    inputs['visible_measurements'].append(StateOnly(native_corner=hidden))
    base = M.native_baseline(**inputs)
    assert hidden not in [m['native_corner'] for m in base.measurements]
    values = M._BI_records(['U'] * 8, [False] * 8, [StateOnly(native_corner=0)])
    assert values == []


def test_zero_component_two_policy_rows_exact_refs_tangents_floors_and_no_labels():
    _, inputs, metadata, _ = fixture(partial=True, two=True)
    before = deepcopy((inputs, metadata))
    row, audit = M.prepare_one(**inputs, **metadata)
    assert row['selection_profile'] == M.G.SELECTION_PROFILE
    assert row['observation_generation'] == M.G.OBSERVATION_FAMILY
    assert row['noise_population'] == M.G.POPULATION
    assert row['raw_observation_generation'] == M.G.RAW_OBSERVATION_GENERATION
    assert audit['authenticated_groups'] == 2
    for policy, profile in row['profiles'].items():
        assert profile['selection_profile'] == M.G.SELECTION_PROFILE
        for group in profile['groups']:
            component = group['components'][0]
            assert component['mixture_component_prior'] == 1.
            assert component['normal_variance_px2'] == component['normal_floor_px']**2
            assert component['angular_variance_rad2'] == component['angular_sine_floor']**2
            assert component['noise_provenance']['mode'] == 'ZERO_COMPONENT_PREFIT_METADATA_NOT_FITTED'
            assert component['selected_chain_ref'] == group['selected_chain_ref']
        for record in profile['tangent_records']:
            assert NOISE.prepared_tangent(record, row['states'], inputs['initial_self_hidden']) is not None
    assert (inputs, metadata) == before


def test_zero_supported_row_stays_in_both_profiles_and_S_projection_slot_not_copied():
    _, inputs, metadata, _ = fixture()
    metadata['observation']['selected_chain_refs'] = []
    metadata['observation']['joint_graph']['selected_line_names'] = []
    hidden = next(i for i, s in enumerate(inputs['states']) if s == 'S')
    class Poison:
        def __array__(self, *a, **kw): raise AssertionError('S projection read')
        def __deepcopy__(self, memo): raise AssertionError('S projection copied')
    metadata['initial_geometry']['initial_projection_xy'][hidden] = Poison()
    row, audit = M.prepare_one(**inputs, **metadata)
    assert audit['authenticated_groups'] == 0
    assert row['initial_geometry']['initial_projection_xy'][hidden] is None
    assert set(row['profiles']) == set(M.G.SOURCE_POLICIES)
    assert all(not profile['groups'] and not profile['tangent_records'] for profile in row['profiles'].values())


def test_CAL_gate_before_other_arguments_or_new_profile_numeric_payload():
    _, inputs, metadata, _ = fixture()
    metadata['partition'] = 'TEST'; inputs['family'] = object()
    with pytest.raises(ValueError, match='CAL-only'): M.prepare_one(**inputs, **metadata)
    _, inputs, metadata, _ = fixture()
    metadata['observation']['selection_profile'] = M.G.PREVIOUS.SELECTION_PROFILE
    metadata['observation']['selected_chain_refs'] = object()
    with pytest.raises(ValueError, match='V32 initial-only selection'):
        M.prepare_one(**inputs, **metadata)
