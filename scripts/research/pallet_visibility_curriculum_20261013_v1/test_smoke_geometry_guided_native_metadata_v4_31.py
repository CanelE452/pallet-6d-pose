"""Synthetic metadata smoke composition/IO gates, no actual CAL decoding."""
from copy import deepcopy
import pytest

from . import smoke_geometry_guided_native_metadata_v4_31 as S
from .test_prepare_geometry_guided_native_evidence_v4_31 import rows


def synthetic():
    task, row = rows(); task['id'] = row['id'] = S.CASES[0]
    observation = row['observation']
    graph = deepcopy(observation['joint_graph'])
    for i, node in enumerate(graph['line_candidates']):
        reference = observation['selected_chain_refs'][i]
        node.update(native_edge=[reference['native_corner'], reference['native_neighbor']],
                    provenance=[str(i) + ':0'], proposal_grade='FULL_LENGTH_IMAGE_LINE_PROPOSAL')
        chain = observation['chain_observation']['edge_records'][i]['chains'][0]
        chain['chain_id'] = 'synthetic-chain-' + str(i)
    result = dict(selection_profile=S.G.SELECTION_PROFILE, graph=graph, node_association_ledger=[])
    return task, row, result


def test_GO_guard_precedes_all_actual_IO(monkeypatch):
    monkeypatch.setattr(S.C, 'read', lambda *args: pytest.fail('No actual read before GO'))
    with pytest.raises(PermissionError, match='before any IO'): S.run()


def test_compose_exact_saved_graph_then_prepare_both_policies_without_ranking(monkeypatch):
    task, row, result = synthetic(); before = deepcopy((row, result))
    monkeypatch.setattr(S.G.ASSOCIATION, 'rank_observed_anchor_native_ray_graph',
                        lambda *a, **kw: pytest.fail('No fresh rank in metadata smoke'))
    selected = S.compose_selected_row(row, result, [{'visibility': 'synthetic'}])
    prepared, meta = S.M.prepare_one(task, selected)
    assert selected['observation']['joint_graph'] is result['graph']
    assert selected['observation']['chain_observation'] is row['observation']['chain_observation']
    assert selected['observation']['selected_physical_edges'] == len(selected['observation']['selected_chain_refs'])
    assert selected['observation']['edge_visibility'] == [{'visibility': 'synthetic'}]
    assert set(prepared['profiles']) == set(S.G.SOURCE_POLICIES)
    assert meta['authenticated_groups'] == 1
    assert (row, result) == before


def test_wrong_profile_or_primary_partition_rejects_before_graph_payload():
    task, row, result = synthetic()
    for badrow, badresult in [(dict(partition='TEST'), {}),
                               (row, dict(selection_profile='OLD', graph=object()))]:
        with pytest.raises(ValueError, match='fixed CAL3'):
            S.compose_selected_row(badrow, badresult, [])


def test_empty_saved_selection_is_retained_without_new_evidence():
    task, row, result = synthetic(); result['graph']['selected_line_names'] = []
    selected = S.compose_selected_row(row, result, [])
    prepared, meta = S.M.prepare_one(task, selected)
    assert selected['summary']['zero_selected_edges']
    assert meta['authenticated_groups'] == 0
    assert all(not profile['groups'] and not profile['tangent_records'] for profile in prepared['profiles'].values())


def test_incompatible_source_union_is_rejected():
    target = {'frozen.py': 'a'*64}
    with pytest.raises(ValueError, match='ancestry conflict'):
        S._union(target, {'frozen.py': 'b'*64})
