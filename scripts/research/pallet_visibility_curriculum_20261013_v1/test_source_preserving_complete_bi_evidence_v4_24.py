"""Focused provenance/owner algebra and analytic objective checks; synthetic."""
from copy import deepcopy

import numpy as np
import pytest

from . import source_preserving_complete_bi_evidence_v4_24 as V
from .test_complete_bi_image_chain_evidence_v4_20 import fixture as base_fixture, Poison
from .test_complete_bi_image_chain_evidence_v4_20d import fixture as old_fixture, ray


def fixture(*, fallback=True, pair=False):
    base, obs, kw, ids, R, t = old_fixture(pair)
    k = ids[0]
    if fallback:
        next(r for r in kw['visible_measurements'] if r['native_corner'] == k)['source'] = V.FALLBACK_SOURCE
    kw['noise_provenance']['observation_family'] = V.NOISE_OBSERVATION_FAMILY
    return base, obs, kw, ids, R, t


def auth(base, obs, kw):
    return V.authenticate_complete_bi_image_chain_groups(base, obs, **kw)


def plan(base, bundle):
    return V.plan_complete_bi_replacement(base, bundle, overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)


@pytest.mark.parametrize('source', ['FROZEN_RGB_COMPLETE_SOURCE', 'B_ACTUAL_BOUNDARY_RECOVERY',
    'VISIBLE_INTERIOR_RGB_CORRECTION', 'UNKNOWN_SOURCE', 'visible_backbone_fallback'])
def test_RGB_or_unknown_complete_cost_is_exactly_retained(source):
    base, obs, kw, (k, _, _), R, t = fixture(fallback=False)
    next(r for r in kw['visible_measurements'] if r['native_corner'] == k)['source'] = source
    before = deepcopy(kw['visible_measurements'])
    bundle = auth(base, obs, kw)
    old = V.OLD.plan_complete_bi_replacement(base, bundle, overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
    assert k in old['revoked_complete_ids']
    p = V.CompleteBIImageChainProblem(base, bundle, overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
    record = next(r for r in p.plan['corner_plans'] if r['native_corner'] == k)
    assert not record['point_cost_revoked'] and 'tangent_kind' not in record
    assert not p.groups and not p.revoked
    at = V.N.V._pose_vector(R, t, base.scale) + [.01, -.02, .005, .008, -.011, .025]
    original, original_gradient, _ = base.objective(at)
    value, gradient, parts = p.objective(at)
    assert value == original and np.array_equal(gradient, original_gradient)
    assert sum(c['kind'] == 'COMPLETE_BI' and c['native_corner'] == k for c in parts['native_costs']) == 1
    assert kw['visible_measurements'] == before


def test_fallback_single_line_revoke_once_tangent_once_FD_and_old_guards_exact():
    base, obs, kw, (k, _, _), R, t = fixture()
    bundle = auth(base, obs, kw)
    p = V.CompleteBIImageChainProblem(base, bundle, overlap_policy=V.PROTOTYPE_OVERLAP_POLICY)
    record = next(r for r in p.plan['corner_plans'] if r['native_corner'] == k)
    assert record['point_cost_revoked'] and record['tangent_kind'] == 'ARCHIVED_COMPLETE_BI_TANGENT'
    at = V.N.V._pose_vector(R, t, base.scale) + [.01, -.02, .005, .008, -.011, .025]
    value, gradient, parts = p.objective(at)
    numeric = V.N.V._jacobian(lambda x: np.array([p.objective(x)[0]]), at)[0]
    assert np.allclose(gradient, numeric, atol=4e-5, rtol=4e-6)
    assert value == pytest.approx(sum(c['weighted_robust_cost'] for c in parts['native_costs']))
    assert sum(c['kind'] == 'ARCHIVED_COMPLETE_BI_TANGENT' and c['native_corner'] == k
        for c in parts['native_costs']) == 1
    assert not any(c['kind'] == 'COMPLETE_BI' and c['native_corner'] == k for c in parts['native_costs'])
    oldv, oldj = base.constraints(at, base.sides[0])
    newv, newj = p.constraints(at, p.sides[0])
    assert np.array_equal(oldv, newv) and np.array_equal(oldj, newj)
    assert p.plan['ledger']['original_complete_q_factors_once'] == len(base.complete) - 1


def test_RGB_owner_removal_preserves_other_fallback_owner_and_shared_edge_once():
    base, obs, kw, (k, mate, _), _, _ = fixture(fallback=False)
    other = deepcopy(obs['edge_records'][0])
    other.update(native_corner=mate, native_neighbor=k)
    chain = other['chains'][0]
    chain.update(native_corner=mate, native_neighbor=k, native_edge=[mate, k],
        corrected_endpoint_normal_residual_px={str(mate): 0.})
    if base.states[mate] == 'I':
        chain['mask_context_kind'] = 'TARGET_TARGET_INTERNAL'
        chain['mask_minus'] = [.99] * len(chain['observed_points'])
    obs['edge_records'].append(other)
    next(r for r in kw['visible_measurements'] if r['native_corner'] == mate)['source'] = V.FALLBACK_SOURCE
    result = plan(base, auth(base, obs, kw))
    assert k not in result['revoked_complete_ids'] and mate in result['revoked_complete_ids']
    assert len(result['active_groups']) == 1
    assert result['active_groups'][0]['activation_owners'] == [
        dict(kind='ROLE_CORRECT_SELF_COMPLETE_ANCHOR', native_corner=mate)]
    assert result['source_owner_removals'] and result['ledger']['unique_physical_edge_factors_once'] == 1


def test_existing_partial_reference_survives_without_complete_fallback_owner():
    base, obs, kw, (k, mate, _), _, _ = base_fixture(partial=True)
    kw['noise_provenance']['observation_family'] = V.NOISE_OBSERVATION_FAMILY
    obs['configuration'] = deepcopy(V.OBS.DEFAULT_CONFIGURATION)
    obs['edge_records'] = [ray(base, kw['initial_projection_xy'], k, mate)]
    kw['junction_proposals'] = dict(schema='NATIVE_EDGE_TOPOLOGY_JUNCTION_PROPOSALS_V4_17B',
        configuration=deepcopy(V.OBS.DEFAULT_CONFIGURATION), image_binding=deepcopy(obs['image_binding']),
        mask_binding=dict(image_sha256=kw['image_binding']['sha256'], mask_pixel_sha256='e' * 64),
        records=[dict(native_corner=i, state=s, candidates=[]) for i, s in enumerate(base.states)])
    result = plan(base, auth(base, obs, kw))
    assert not result['revoked_complete_ids']
    assert len(result['active_groups']) == 1
    assert result['active_groups'][0]['activation_owners'] == [
        dict(kind='EXISTING_PARTIAL_NORMAL_REF', native_corner=k, line_index=0)]
    assert result['replaces_normal_factor_refs'] == [dict(native_corner=k, line_index=0)]


def test_unchanged_topology_failure_cannot_be_promoted_by_fallback_source():
    base, obs, kw, (k, _, _), _, _ = fixture(pair=True)
    kw['junction_proposals']['records'][k]['candidates'] = []
    result = plan(base, auth(base, obs, kw))
    assert k not in result['revoked_complete_ids'] and not result['active_groups']
    assert result['main_replacement_rejections']


@pytest.mark.parametrize('mode', ['source_mismatch', 'both_sources_changed', 'xy_changed', 'weight_changed'])
def test_frozen_source_and_measurement_archive_mutation_rejected(mode):
    base, obs, kw, (k, _, _), _, _ = fixture()
    bundle = auth(base, obs, kw)
    archive = bundle['frozen_measurement_archives'][str(k)]
    if mode == 'source_mismatch':
        archive['original_measurement_source'] = 'B_ACTUAL_BOUNDARY_RECOVERY'
    elif mode == 'both_sources_changed':
        archive['original_measurement_source'] = archive['source_record']['source'] = 'B_ACTUAL_BOUNDARY_RECOVERY'
    elif mode == 'xy_changed':
        archive['original_observed_xy'][0] += 1.
    else:
        archive['original_fit_reliability_weight'] += .01
    with pytest.raises(ValueError):
        plan(base, bundle)


def test_S_fields_untouched_and_old_noise_generation_cannot_be_reused():
    base, obs, kw, (k, _, _), _, _ = fixture()
    hidden = next(i for i, state in enumerate(base.states) if state == 'S')
    obs['edge_records'].append(dict(native_corner=hidden, native_neighbor=k, chains=Poison()))
    kw['junction_proposals']['records'][hidden]['candidates'] = Poison()
    bundle = auth(base, obs, kw)
    bundle['frozen_measurement_archives'][str(hidden)] = Poison()
    assert plan(base, bundle)['source_policy'] == V.SOURCE_POLICY
    kw['noise_provenance']['observation_family'] = V.OLD.NOISE_OBSERVATION_FAMILY
    with pytest.raises(ValueError, match='new-population'):
        auth(base, obs, kw)
