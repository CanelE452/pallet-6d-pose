"""Positive-only projection using toy AUTH32 data; no actual rows/labels/IO."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from . import clean_fresh_BI_input_projection_v4_32 as P
from .test_prepare_geometry_guided_native_metadata_v4_32 import rows, Poison
from .test_prepare_geometry_guided_native_evidence_v4_32 import binding


def exports(policy, *, tangent=False):
    value = dict(**P._tags(), fit_partition='CAL', source_policy=policy,
        model_digest_sha256=('b' if tangent else 'c')*64,
        newly_fitted_selection_population=True,
        prior_V29_fitted_model_relabelled=False, prior_V30_fitted_model_relabelled=False,
        prior_V31_fitted_model_relabelled=False)
    if tangent:
        value.update(gamma=.2, pixel_floor_variance_px2=1.,
                     tangent_axis='SELECTED_ACTUAL_MEASURED_LINE_TANGENT')
    return value


def project(partition='CAL', *, partial=True):
    task, row = rows(partial=partial)
    task['partition'] = row['partition'] = partition
    task['id'] = row['id'] = 'SYNTHETIC:'+partition
    return task, row, P.project_clean_input(task, row, initial_source_binding=binding())


def attached(projected, row, *, unavailable=False):
    ci = projected['clean_input']; coarse = ci['initial_coarse_pose']
    families = P.CLEAN._families(np.asarray(ci['native_xyz']), coarse, np.asarray(ci['physical_registration_Q']))
    bundles, tangents, lines = {}, {}, {}
    for policy in P.G.SOURCE_POLICIES:
        tangents[policy] = None if unavailable else exports(policy, tangent=True)
        lines[policy] = None if unavailable else exports(policy)
        bundles[policy] = {}
        for name, family in families.items():
            base = P.FACTORY.native_baseline(family=family, K=ci['K'], states=ci['states'],
                visible_measurements=ci['visible_measurements'], **projected['physics'],
                initial_source_binding=ci['initial_source_binding'])
            auth = projected['observation_auth_arguments']
            if unavailable:
                bundle = P.G.empty_geometry_guided_evidence(base, **auth,
                    association_configuration=row['observation']['association_configuration'], reason='SYNTHETIC new model unavailable')
            else:
                bundle = P.FACTORY.AUTH.authenticate_geometry_guided_edges(base, row['observation'], **auth,
                    normal_variance_component_px2=2., angular_variance_component_sine2=.001,
                    normal_angle_correlation=0., noise_provenance=lines[policy],
                    expected_source_policy=policy, calibration_preparation=False)
            bundles[policy][name] = bundle
    return dict(typed_bundles=bundles, regularizers={name:None for name in P.REGULARIZER_KEYS},
        model_availability={policy:not unavailable for policy in P.G.SOURCE_POLICIES},
        tangent_parameters=tangents, line_noise_provenance=lines)


def test_partitions_and_identifiers_do_not_route_physics_or_create_PnP(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Projector performed PnP/optimization')
    monkeypatch.setattr(P.CLEAN, 'prepare_clean_fresh_BI_task', forbidden)
    monkeypatch.setattr(P.FRESH, 'generate_fresh_observed_BI_pnp_proposals', forbidden)
    outputs = [project(partition)[2] for partition in P.PARTITIONS]
    digests = [o['source_contract']['positive_physics_sha256'] for o in outputs]
    assert len(set(digests)) == 1
    assert [o['identifiers']['partition'] for o in outputs] == list(P.PARTITIONS)
    for result in outputs:
        assert result['ready'] and not result['source_contract']['raw_observation_forwarded']
        assert set(result['observation_auth_arguments']) == {'initial_geometry', 'image_binding'}
        assert not {'observation','baseline','proposals','initializations','pose_guides'} & set(result)


def test_original_positive_projection_matches_frozen_CAL_extractor_numerically():
    from . import prepare_geometry_guided_native_metadata_v4_32 as CAL
    task, row, result = project()
    inputs, metadata = CAL.positive_inputs(task, row, initial_source_binding=binding())
    assert result['physics'] == {key:inputs[key] for key in result['physics']}
    ci = result['clean_input']
    for key in ('K','states','visible_measurements','initial_self_hidden','initial_source_binding'):
        assert ci[key] == inputs[key]
    assert ci['native_xyz'] == inputs['family']['native_xyz']
    assert ci['initial_coarse_pose'] == inputs['family']['coarse_pose']
    assert result['observation_auth_arguments']['image_binding'] == metadata['image_binding']


def test_old_bank_SWAPPED_initializations_and_raw_observation_are_not_read_or_copied():
    task, row = rows()
    row['observation'] = Poison(); row['extra_guides'] = Poison(); row['legacy_pose'] = Poison()
    result = P.project_clean_input(task, row, initial_source_binding=binding())
    assert result['ready']
    assert set(result['clean_input']) == {'native_xyz','K','states','initial_self_hidden',
        'visible_measurements','initial_coarse_pose','physical_registration_Q','initial_source_binding','identifiers'}
    assert set(result['physics']) == {'domains','image_hw','config','boundary_merged','u_regions','initial_self_hidden'}


def test_S_and_unknown_observed_payloads_never_read_and_S_model_slot_sanitized():
    task, row = rows(partial=False)
    class StateOnly(dict):
        def __getitem__(self, key):
            if key == 'native_corner': return super().__getitem__(key)
            raise AssertionError('Excluded q/source/covariance payload read')
        def __deepcopy__(self, memo): raise AssertionError('Excluded payload copied')
    s = next(k for k,state in enumerate(task['states']) if state == 'S')
    u = next(k for k,state in enumerate(task['states']) if state == 'B')
    task['states'][u] = 'U'; task['spec']['states'] = task['states']; row['states'] = task['states']
    for owner, key in ((task['spec'], 'visible'), (row, 'visible_measurements')):
        owner[key] = [m for m in owner[key] if m['native_corner'] != u]
        owner[key] += [StateOnly(native_corner=s), StateOnly(native_corner=u)]
    task['complete_BI_evidence']['initial_projection_xy'][s] = Poison()
    row['initial_geometry']['initial_projection_xy'][s] = Poison()
    result = P.project_clean_input(task, row, initial_source_binding=binding())
    assert result['observation_auth_arguments']['initial_geometry']['initial_projection_xy'][s] is None
    assert {s,u}.isdisjoint(m['native_corner'] for m in result['clean_input']['visible_measurements'])


@pytest.mark.parametrize('change', ['q','source','covariance','reliability','K','canvas','domain','U','R','center','projection','profile'])
def test_mismatched_original_physics_measurement_and_guards_failclosed(change):
    task, row = rows()
    if change == 'q': row['visible_measurements'][0]['xy'][0] += 1.
    elif change == 'source': row['visible_measurements'][0]['source'] = 'different'
    elif change == 'covariance': row['visible_measurements'][0]['fit_covariance_px2'][0][0] += 1.
    elif change == 'reliability': row['visible_measurements'][0]['fit_reliability_weight'] += .1
    elif change == 'K': row['initial_geometry']['K'][0][0] += 1.
    elif change == 'canvas': row['initial_geometry']['image_hw'] = [481,640]
    elif change == 'domain': row['domains'] = [dict(native_corner=0)]
    elif change == 'U': row['u_regions'] = [dict(native_corner=0)]
    elif change == 'R': row['initial_geometry']['R_cf'][0][0] += .01
    elif change == 'center': row['initial_geometry']['centroid'][2] += .01
    elif change == 'projection': row['initial_geometry']['initial_projection_xy'][0][0] += 1.
    else: row['selection_profile'] = P.G.PREVIOUS.SELECTION_PROFILE
    with pytest.raises(ValueError): P.project_clean_input(task, row, initial_source_binding=binding())


def test_binding_and_identity_gate_precede_physics_access():
    task, row = rows(); task['spec'] = Poison(); bad = binding(); bad['verified_by_caller'] = False
    with pytest.raises(ValueError, match='closure'):
        P.project_clean_input(task, row, initial_source_binding=bad)
    with pytest.raises(ValueError, match='partition'):
        P.project_clean_input(dict(partition='OTHER'), dict(partition='OTHER'), initial_source_binding=Poison())


@pytest.mark.parametrize('change', ['missing','rotation','behind'])
def test_invalid_coarse_retains_no_pose_row_before_all_measurement_guard_auth_payload(change):
    task, row = rows()
    coarse = task['families']['ORIGINAL_WD']['coarse_pose']
    if change == 'missing': del coarse['t']
    elif change == 'rotation': coarse['R'] = np.zeros((3,3)).tolist()
    else: coarse['t'] = [0.,0.,-100.]
    task['spec']['visible'] = task['spec']['domains'] = task['spec']['config'] = Poison()
    task['boundary_merged'] = task['u_regions'] = task['complete_BI_evidence'] = Poison()
    row['initial_geometry'] = row['visible_measurements'] = row['domains'] = row['u_regions'] = row['observation'] = Poison()
    projected = P.project_clean_input(task, row, initial_source_binding=binding())
    assert not projected['ready'] and projected['source_contract']['row_retained']
    assert projected['physics'] is projected['observation_auth_arguments'] is None
    assert projected['clean_input']['visible_measurements'] == []
    assert 'R' not in projected['clean_input']['initial_coarse_pose']
    packet = P.assemble_worker_input(projected, typed_bundles=Poison(), regularizers=Poison(),
        model_availability=Poison(), tangent_parameters=Poison(), line_noise_provenance=Poison())
    clean = P.CLEAN.prepare_clean_fresh_BI_task(**packet['clean_input'])
    assert not clean['ready'] and not clean['pose_available'] and clean['row_retained']
    assert packet['physics'] is packet['typed_bundles'] is packet['regularizers'] is None


def test_genuine_AUTH32_both_families_and_policies_worker_fresh_core_connection_and_immutability():
    task, row, projected = project(partial=True)
    arguments = attached(projected, row)
    # Unexpected top-level raw archives must never be copied to a worker.
    for families in arguments['typed_bundles'].values():
        for bundle in families.values():
            bundle['raw_observation'] = Poison(); bundle['guide_bank'] = Poison()
    before = deepcopy(projected)
    bundles_before = {p:{n:P.G._digest({k:v for k,v in b.items() if k not in ('raw_observation','guide_bank')})
        for n,b in families.items()} for p,families in arguments['typed_bundles'].items()}
    packet = P.assemble_worker_input(projected, **arguments)
    assert set(packet) == P.WORKER_KEYS
    assert projected == before
    clean = P.CLEAN.prepare_clean_fresh_BI_task(**packet['clean_input'])
    assert clean['ready'] and clean['cached_pose_inputs'] == 0
    for policy in P.G.SOURCE_POLICIES:
        for name, family in clean['families'].items():
            bundle = packet['typed_bundles'][policy][name]
            assert not {'raw_observation','guide_bank'} & set(bundle)
            for group in bundle['groups']:
                assert set(group['authenticated_raw_query_aliases']) == set(P.G.RAW_QUERY_ALIAS_KEYS)
            base = P.FACTORY.native_baseline(family=family, K=clean['spec']['K'], states=clean['states'],
                visible_measurements=packet['clean_input']['visible_measurements'], **packet['physics'],
                initial_source_binding=clean['initial_source_binding'])
            gamma = packet['source_contract']['tangent_parameters'][policy]['gamma']
            problem = P.G.GeometryGuidedEdgeProblem(base,bundle,source_policy=policy,tangent_gamma=gamma)
            p = P.G.V._pose_vector(np.asarray(family['coarse_pose']['R']),
                np.asarray(family['coarse_pose']['t']),problem.scale)
            value, gradient, _ = problem.objective(p)
            assert np.isfinite(value) and np.isfinite(gradient).all()
            assert base.cfg['prior_weight'] == 0.
    assert bundles_before == {p:{n:P.G._digest({k:v for k,v in b.items() if k not in ('raw_observation','guide_bank')})
        for n,b in families.items()} for p,families in arguments['typed_bundles'].items()}


def test_unavailable_is_explicit_newempty32_gamma0_and_never_oldmodel():
    _, row, projected = project()
    arguments = attached(projected, row, unavailable=True)
    packet = P.assemble_worker_input(projected, **arguments)
    assert packet['source_contract']['unavailable_policy_tangent_gamma'] == 0.
    assert not packet['source_contract']['old_model_substituted']
    for policy in P.G.SOURCE_POLICIES:
        assert packet['source_contract']['tangent_parameters'][policy] is None
        assert not packet['source_contract']['model_availability'][policy]
        for bundle in packet['typed_bundles'][policy].values():
            assert bundle['groups'] == [] and bundle['evidence_availability'] == 'UNAVAILABLE_UNIFORM_ABSTENTION'
    # Explicit new-model unavailability preserves the exact no-owner objective
    # and all original guard values/Jacobians; no old noise/gamma is inserted.
    ci = packet['clean_input']
    families = P.CLEAN._families(np.asarray(ci['native_xyz']), ci['initial_coarse_pose'],
                                 np.asarray(ci['physical_registration_Q']))
    for policy in P.G.SOURCE_POLICIES:
        for name, family in families.items():
            base = P.FACTORY.native_baseline(family=family, K=ci['K'], states=ci['states'],
                visible_measurements=ci['visible_measurements'], **packet['physics'],
                initial_source_binding=ci['initial_source_binding'])
            core = P.G.GeometryGuidedEdgeProblem(base,packet['typed_bundles'][policy][name],
                                                source_policy=policy,tangent_gamma=0.)
            p = np.array([.1,-.2,.01,.03,-.04,2.])
            a,ja,_ = core.objective(p); b,jb,_ = base.objective(p)
            assert a == b and np.array_equal(ja,jb)
            for side in base.sides:
                c,jc=core.constraints(p,side); d,jd=base.constraints(p,side)
                assert np.array_equal(c,d) and np.array_equal(jc,jd)
    policy = P.G.SOURCE_POLICIES[0]
    arguments['tangent_parameters'][policy] = exports(policy, tangent=True)
    with pytest.raises(ValueError, match='substitute'):
        P.assemble_worker_input(projected, **arguments)


@pytest.mark.parametrize('change', ['oldfamily','oldprofile','oldnoise','wrongpolicy','missingalias','rawref','wrongimage','extraedge','Sgate'])
def test_selected_auth_source_binding_and_model_policy_failclosed_before_worker(change):
    _, row, projected = project()
    arguments = attached(projected, row)
    policy = P.G.SOURCE_POLICIES[0]; bundle = arguments['typed_bundles'][policy]['ORIGINAL_WD']
    group = bundle['groups'][0]
    if change == 'oldfamily': bundle['observation_family'] = P.G.PREVIOUS.OBSERVATION_FAMILY
    elif change == 'oldprofile': arguments['tangent_parameters'][policy]['selection_profile'] = 'OLD'
    elif change == 'oldnoise': arguments['line_noise_provenance'][policy]['prior_V31_fitted_model_relabelled'] = True
    elif change == 'wrongpolicy': arguments['line_noise_provenance'][policy]['source_policy'] = P.G.SOURCE_POLICIES[1]
    elif change == 'missingalias': del group['authenticated_raw_query_aliases']['guide_reference_id']
    elif change == 'rawref': group['selected_chain_ref']['guide_reference']['guide_id'] = 'CACHED0'
    elif change == 'wrongimage': bundle['image_binding']['sha256'] = 'f'*64
    elif change == 'extraedge': bundle['groups'].append(deepcopy(group))
    else:
        group['native_neighbor'] = next(k for k,s in enumerate(projected['clean_input']['states']) if s == 'S')
        group['selected_chain_ref'] = group['authenticated_raw_query_aliases'] = group['noise_provenance'] = Poison()
        group['normal_unit_xy'] = group['actual_inlier_xy'] = Poison()
    with pytest.raises((ValueError, KeyError)):
        P.assemble_worker_input(projected, **arguments)


def test_worker_signature_no_raw_task_or_observation_and_same_source_binding_required():
    assert not {'task','row','observation','pose_guides','proposals'} & set(inspect.signature(P.assemble_worker_input).parameters)
    _, row, projected = project(); arguments = attached(projected, row)
    projected['source_contract']['initial_source_binding']['path'] += '-changed'
    with pytest.raises(ValueError, match='binding'):
        P.assemble_worker_input(projected, **arguments)


@pytest.mark.parametrize('change',['archive','physics','extra_worker_field'])
def test_parent_projection_integrity_rechecked_before_copying_packet(change):
    _, row, projected = project(); arguments = attached(projected,row)
    if change == 'archive': projected['clean_input']['visible_measurements'][0]['protection_px'] += 1.
    elif change == 'physics': projected['physics']['config']['prior_weight'] += 1.
    else: projected['clean_input']['proposals'] = Poison()
    with pytest.raises(ValueError,match='projection changed|binding'):
        P.assemble_worker_input(projected,**arguments)
