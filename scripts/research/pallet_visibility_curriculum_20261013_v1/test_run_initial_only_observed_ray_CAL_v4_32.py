"""Source/fixture-only CAL3 operator contracts; actual rows and models0."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import run_initial_only_observed_ray_CAL_v4_32 as R
from .test_initial_only_observed_ray_association_v4_32 import augmented_graph


class Poison:
    def __getitem__(self, key):
        raise AssertionError('Forbidden cached pose or primary payload read')

    def get(self, *args):
        raise AssertionError('Forbidden cached pose or primary payload read')


@pytest.mark.parametrize('stage', ['declare', 'run'])
def test_both_GO_gates_precede_IO(stage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('IO before GO')
    monkeypatch.setattr(R.C, 'sha', forbidden)
    monkeypatch.setattr(R.C, 'read', forbidden)
    monkeypatch.setattr(R, '_pins', forbidden)
    with pytest.raises(PermissionError, match='before IO'):
        R.declare() if stage == 'declare' else R.run(expected_plan_sha256='0'*64)


def test_fixed_cases_require_order_all_three_and_CAL_partition():
    good = [dict(id=k, partition='CAL') for k in R.CASES]
    R._cases(good)
    for bad in (good[::-1], good[:2], [*good[:2], good[0]], [dict(id=k, partition='TEST') for k in R.CASES]):
        with pytest.raises(ValueError, match='Same ordered'):
            R._cases(bad)


def test_primary_or_unlisted_row_gate_before_coordinate_payload():
    for row in (dict(partition='TEST', id=R.CASES[0]), dict(partition='CAL', id='unknown')):
        row.update(observation=Poison(), initial_geometry=Poison())
        with pytest.raises(ValueError, match='before coordinate'):
            R._rank_row(row, Poison(), Poison())


def synthetic_row():
    args, kwargs, chains = augmented_graph()
    image, mask, geometry, states, visible = args
    for ri, ray in enumerate(chains['edge_records']):
        for ci, chain in enumerate(ray['chains']):
            chain['chain_id'] = f'SYNTHETIC_ONLY_{ri}_{ci}'
    row = dict(id=R.CASES[0], partition='CAL', image_sha256=kwargs['image_binding']['sha256'],
        states=states, visible_measurements=visible, initial_geometry=geometry,
        mask_binding=dict(path='opaque_synthetic', sha256='b'*64),
        observation=dict(observation_generation=R.F.P.OBSERVATION_GENERATION,
            chain_observation=chains, guide_bank=dict(do_not_use=True), guide_geometries=dict(do_not_use=True),
            target_mask_pixel_sha256=R.F.P.VIS.mask_pixel_sha256(mask)),
        proposals=Poison(), baseline=Poison(), old_initializations=Poison())
    return row, mask, kwargs


def test_cached_pose_banks_and_guide_restore_are_not_connected_and_raw_unchanged(monkeypatch):
    row, mask, kwargs = synthetic_row()
    before = R.R._canonical_sha(R.OLD._immutable_observation_view(row))
    def forbidden(*args, **kwargs):
        raise AssertionError('Old guide reconstruction forbidden')
    monkeypatch.setattr(R.R, 'restore_pose_guides', forbidden)
    original_rank = R.F.rank_initial_only_observed_ray_graph
    calls = []
    def rank(*args, **kw):
        assert kw['pose_guides'] is None
        calls.append(1)
        return original_rank(*args, **kw)
    monkeypatch.setattr(R.F, 'rank_initial_only_observed_ray_graph', rank)
    graph, summary = R._rank_row(row, mask, None)
    assert calls == [1]
    assert summary['external_pose_guides_consumed'] == 0
    assert graph['graph_geometry_source'] == 'ORIGINAL_INITIAL_ONLY'
    assert R.R._canonical_sha(R.OLD._immutable_observation_view(row)) == before


def test_zero_selection_stays_an_explicit_summary(monkeypatch):
    row, mask, _ = synthetic_row()
    empty = dict(graph=dict(selected_line_names=[], actual_unique_line_candidates=0, line_candidates=[]),
        node_association_ledger=[], candidate_nodes_checked=0, rejected_node_count=0,
        external_pose_guides_consumed=0, graph_geometry_source='ORIGINAL_INITIAL_ONLY')
    monkeypatch.setattr(R.F, 'rank_initial_only_observed_ray_graph', lambda *a, **kw: deepcopy(empty))
    monkeypatch.setattr(R.R, 'decorate_graph', lambda *a, **kw: None)
    _, summary = R._rank_row(row, mask, None)
    assert summary['zero_selected_edges'] and summary['selected_count'] == 0
    assert summary['id'] == R.CASES[0]
    assert not summary['performance_or_noise_validity_claim']


def test_mask_pixel_binding_failure_precedes_rank(monkeypatch):
    row, mask, _ = synthetic_row()
    monkeypatch.setattr(R.F, 'rank_initial_only_observed_ray_graph', lambda *a, **kw: pytest.fail('Unbound mask rank'))
    changed = np.asarray(mask).copy()
    changed[0, 0] = 1-int(changed[0, 0])
    with pytest.raises(ValueError, match='pixel binding'):
        R._rank_row(row, changed, None)
