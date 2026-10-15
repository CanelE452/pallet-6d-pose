import numpy as np
import pytest
from . import compact_edge_component_dag_v4_18 as P
from . import image_supported_edge_chains_v4_17c as OBS


def links(edges):return [dict(components=list(edge)) for edge in edges]


def test_complete_ordered_DAG_has_exponential_subsets_but_one_reduced_chain():
    r=P.graph_summary(links([(i,j) for i in range(10) for j in range(i+1,10)]))
    assert r['root_terminal_path_count_upper_bound']=='256'
    assert r['transitive_reduced_root_path_count']=='1'
    assert len(r['transitive_reduction_edges'])==9 and r['path_sets_enumerated']==0


def test_branch_alternatives_and_disconnected_components_are_preserved():
    r=P.graph_summary(links([(0,1),(0,2),(1,3),(2,3)]),[10])
    assert r['transitive_reduced_root_path_count']=='3'
    assert sorted(map(len,r['connected_groups']))==[1,4]
    with pytest.raises(ValueError,match='acyclic'):P.graph_summary(links([(0,1),(1,0)]))


def compact(points,components,edges):
    xy=np.asarray(points,float);components=np.asarray(components)
    return P.compact_actual_component_groups(xy,xy,np.ones(len(xy),bool),components,np.array([1.,0.]),
        np.array([0.,1.]),OBS._config(None),confidence=np.ones(len(xy)),positions=xy[:,0]/10,links=links(edges))


def test_collinear_union_preserves_only_actual_points_without_path_precision_boost():
    r=compact([[0,0],[1,0],[2,0],[7,0],[8,0],[9,0]],[1,1,1,2,2,2],[(1,2)])
    assert len(r['whole_component_line_families'])==1
    family=r['whole_component_line_families'][0]
    assert family['actual_point_count']==6 and family['invented_bridge_points']==0
    assert sum(len(x['actual_NMS_points_xy']) for x in r['raw_components'])==6


def test_overlapping_parallel_components_are_not_hidden_in_a_low_RMS_union():
    r=compact([[0,0],[1,0],[2,0],[1,.5],[2,.5],[3,.5]],[1,1,1,2,2,2],[(1,2)])
    assert not r['whole_component_line_families']
    assert r['unresolved_groups'][0]['reason']=='overlapping_actual_components_remain_parallel_alternatives'
