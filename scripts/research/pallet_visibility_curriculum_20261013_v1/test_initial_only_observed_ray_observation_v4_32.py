"""Uniform synthetic rows only; no actual observation, mask or model files."""
from copy import deepcopy

import pytest

from . import initial_only_observed_ray_observation_v4_32 as V
from .test_run_initial_only_observed_ray_CAL_v4_32 import synthetic_row, Poison


def fixture():
    row, mask, _ = synthetic_row()
    row['observation']['selected_physical_edges'] = 99  # Stale parent count must not survive.
    row['summary'] = dict(selected_physical_edges=99, selected_partial_edges=99,
                          zero_selected_edges=False, preserved_count=123)
    _, cfg = V.F.G._config(None)
    row['observation']['chain_observation']['configuration'] = V.F.G._config(None)[0]
    return row, mask


def test_same_numerical_graph_for_each_inventory_partition_and_output_ID(monkeypatch):
    row, mask = fixture()
    original = deepcopy(V.R._canonical_sha(row['observation']['chain_observation']))
    def forbidden(*a, **kw):
        raise AssertionError('Old guide/bank reconstruction forbidden')
    monkeypatch.setattr(V.R, 'restore_pose_guides', forbidden)
    graphs = []
    for partition in ('CAL', 'TEST', 'REMAINDER'):
        row['partition'] = partition
        row['id'] = 'OUTPUT_METADATA_'+partition
        new, summary = V.rerank_cached_row(row, None, mask)
        graphs.append(V.R._canonical_sha(new['observation']['joint_graph']))
        assert new['observation']['selected_physical_edges'] == len(new['observation']['selected_chain_refs'])
        assert new['summary']['selected_physical_edges'] == summary['selected']
        assert new['summary']['preserved_count'] == 123
        assert new['observation']['external_pose_guides_consumed'] == 0
        assert row['observation']['selected_physical_edges'] == 99
        assert V.R._canonical_sha(row['observation']['chain_observation']) == original
    assert len(set(graphs)) == 1


def test_inventory_gate_before_any_geometry_payload():
    with pytest.raises(ValueError, match='before payload'):
        V.rerank_cached_row(dict(partition='severe', observation=Poison()), Poison(), Poison())


def test_empty_current_selection_keeps_row_and_current_count(monkeypatch):
    row, mask = fixture()
    rank = V.F.rank_initial_only_observed_ray_graph
    def no_edges(*a, **kw):
        result = rank(*a, **kw)
        result['graph']['selected_line_names'] = []
        return result
    monkeypatch.setattr(V.F, 'rank_initial_only_observed_ray_graph', no_edges)
    new, summary = V.rerank_cached_row(row, None, mask)
    assert new['id'] == row['id'] and new['partition'] == row['partition']
    assert not new['observation']['selected_chain_refs']
    assert new['summary']['zero_selected_edges'] and summary['zero_selected']
    assert new['summary']['selected_physical_edges'] == 0
    assert new['observation']['parent_legacy_selected_physical_edges'] == 99
