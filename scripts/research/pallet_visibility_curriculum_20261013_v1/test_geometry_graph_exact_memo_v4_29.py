"""Synthetic exact-score/cache/rank checks; no image searches or dataset IO."""
from copy import deepcopy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from . import geometry_graph_exact_memo_v4_29 as M
from . import bounded_geometry_graph_rank_v4_29 as B
from .test_joint_geometry_guided_lines_v4_28 import fixture,line


def encoded(value):
    return json.dumps(value,allow_nan=True,separators=(',',':'))


def score_fixture():
    edges=[(0,1),(2,3),(4,5),(0,3),(1,2),(4,7),(0,4),(1,5),(2,6)]
    states=['B','I']*4
    points={k:dict(record=dict(xy=[20.+k*13.,25.+k*11.]))for k in range(8)}
    cfg,graph_cfg=M.G._config(None)
    nodes=[];proposals=[];axes={};costs={}
    for i,edge in enumerate(edges):
        axis=i//3;theta=[.12,1.32,2.23][axis]+(i%3)*.031
        normal=np.array([np.cos(theta),np.sin(theta)]);tangent=np.array([-normal[1],normal[0]])
        anchor=np.array([45.+3*i,62.+2*i]);actual=anchor+np.arange(16)[:,None]*tangent
        name=f'{edge[0]}-{edge[1]}/{i}'
        nodes.append(dict(name=name,native_edge=list(edge),normal=normal.tolist(),offset=float(-normal@anchor),
            actual_inliers=actual.tolist(),support_anchor_xy=anchor.tolist(),
            image_quality_energy=float(i%3)/7.,actual_outline_fraction=.8,
            global_external_context=dict(available=i%2==0,external_support_fraction=.6)))
        proposals.append(dict(native_edge=list(edge),native_corner=edge[0],native_neighbor=edge[1],
            direction_xy=tangent.tolist(),vanishing_point_h=None,guide_id='INITIAL'))
        axes[edge]=axis
        costs[name]={'INITIAL':float(i%3)/17.,'SECOND':float(i%2)/19.}
    bank=dict(guide_proposals=proposals,image_binding=dict(image_hw=[160,180]),
        _cached_node_guide_costs=costs,_cached_edge_axes=axes,_cached_K=np.array([[100.,0,90],[0,100.,80],[0,0,1.]]),
        _cached_intersections={})
    return nodes,points,states,bank,cfg,graph_cfg


@pytest.mark.parametrize('limit',[1,2,4096])
def test_entire_ordered_score_bitexact_with_hits_evictions_and_axis_SVD(limit):
    nodes,points,states,bank,cfg,graph_cfg=score_fixture()
    old_bank=deepcopy(bank);memo=M.make_ordered_scorer(maximum_cache_entries=limit)
    fixed=deepcopy((nodes,points,states,cfg,graph_cfg))
    selections=[[],nodes,nodes[:3],nodes[3:6],nodes[6:],nodes[:6],nodes[::2]]
    selections.extend([[nodes[i]if i in ids else None for i in range(len(nodes))]
        for ids in combinations(range(len(nodes)),3)])
    for selected in selections+selections[:7]+[nodes,nodes]:
        expected=M.ORDER._ordered_score(selected,points,states,old_bank,cfg,graph_cfg)
        actual=memo(selected,points,states,bank,cfg,graph_cfg)
        assert encoded(actual)==encoded(expected)
    assert encoded((nodes,points,states,cfg,graph_cfg))==encoded(fixed)
    stats=memo.cache_info()
    assert all(x['entries']<=limit for x in stats.values())
    assert all(x['misses']>0 for x in stats.values())
    # A tiny LRU may legitimately thrash when one score traverses many
    # corners/axes. The large-cache case demonstrates reuse of all three.
    if limit==4096:assert all(x['hits']>0 for x in stats.values())
    assert actual['independently_measured_axis_ray_count']==3
    assert actual['ordered_support_unavailable'] and np.isinf(actual['total_engineering_energy'])


def test_single_entry_caches_reuse_each_term_when_working_set_fits():
    nodes,points,states,bank,cfg,graph_cfg=score_fixture()
    points={0:points[0]};selected=nodes[:4]
    memo=M.make_ordered_scorer(maximum_cache_entries=1)
    first=memo(selected,points,states,bank,cfg,graph_cfg)
    assert encoded(memo(selected,points,states,bank,cfg,graph_cfg))==encoded(first)
    assert all(x['hits']>0 for x in memo.cache_info().values())


def test_rank_context_cannot_be_reused_with_other_frame_or_node_identity():
    nodes,points,states,bank,cfg,graph_cfg=score_fixture();memo=M.make_ordered_scorer()
    memo(nodes,points,states,bank,cfg,graph_cfg)
    with pytest.raises(AssertionError,match='one immutable rank context'):
        memo(nodes,deepcopy(points),states,bank,cfg,graph_cfg)
    with pytest.raises(AssertionError,match='node identity'):
        memo(deepcopy(nodes),points,states,bank,cfg,graph_cfg)
    changed=states[0];states[0]='I'
    with pytest.raises(AssertionError,match='one immutable rank context'):
        memo(nodes,points,states,bank,cfg,graph_cfg)
    states[0]=changed


def test_returned_JSON_cannot_mutate_cached_corner_or_ordered_support():
    nodes,points,states,bank,cfg,graph_cfg=score_fixture();memo=M.make_ordered_scorer()
    first=memo(nodes,points,states,bank,cfg,graph_cfg);expected=encoded(first)
    first['junctions'][0]['line_pair_intersections_xy'][0][0]=999999.
    first['ordered_support_checks'][0]['reason']='external mutation'
    assert encoded(memo(nodes,points,states,bank,cfg,graph_cfg))==expected


def graph_fixture():
    args,kw=fixture();geometry=args[2]
    geometry['initial_self_hidden']=[False]*8
    geometry['initial_projection_xy']=[[30.,55.],[130.,55.],[130.,125.],[30.,125.],
        [30.,85.],[130.,85.],[130.,145.],[30.,145.]]
    args[3][:]=['I']*8
    args[4][:]=[dict(native_corner=k,state='I',xy=xy,source='VISIBLE_BACKBONE_FALLBACK',
        covariance_px2=[[3.,0],[0,3.]],fit_covariance_px2=[[25.,0],[0,25.]],fit_reliability_weight=.8)
        for k,xy in enumerate(geometry['initial_projection_xy'])]
    rays=[]
    for k,mate,normal,offset,actual in [
            (0,1,[0.,1.],-60.,[[x,60.]for x in range(35,75)]),
            (0,3,[1.,0.],-35.,[[35.,y]for y in range(65,105)]),
            (4,5,[0.,1.],-90.,[[x,90.]for x in range(35,75)]),
            (4,7,[1.,0.],-37.,[[37.,y]for y in range(95,135)])]:
        base=line(actual,normal,offset);alternatives=[]
        for i in range(3):
            item=deepcopy(base);item['offset']+=i*.01;item['inlier_rms_px']=i*.01
            alternatives.append(item)
        rays.append(dict(native_corner=k,native_neighbor=mate,supported=True,chains=alternatives,rejected_chains=[]))
    return args,kw,dict(edge_records=rays)


@pytest.mark.parametrize('limit',[1,3,4096])
def test_full_fresh_graph_bitexact_ties_and_input_source_immutability(limit):
    args,kw,obs=graph_fixture();cfg=dict(graph_configuration=dict(maximum_beam_states=5,reported_joint_alternatives=5))
    fixed=deepcopy((args,kw,obs));sources=[Path(M.G.__file__),Path(M.ORDER.__file__),Path(B.PRODUCER.__file__)]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
    expected=B.PRODUCER._rank_fresh(obs,args[2],args[3],args[4],**kw,configuration=cfg,
        pose_guides=None,visibility_context=None,pallet_mask=args[1])
    memo=M.make_ordered_scorer(maximum_cache_entries=limit)
    actual=B.rank_joint_geometry_line_graph_bounded(obs,args[2],args[3],args[4],**kw,
        configuration=cfg,pose_guides=None,visibility_context=None,pallet_mask=args[1],
        intersection_cache_limit=1,score_graph=memo,node_factory=B.PRODUCER._fresh_node)
    actual['joint_alternatives']=[a for a in actual['joint_alternatives']if not a['ordered_support_unavailable']]
    assert encoded(actual)==encoded(expected)
    assert M.G.NMS.G.C.finite((args,kw,obs))==M.G.NMS.G.C.finite(fixed)
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
    assert actual['beam_truncated']


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_invalid_cache_bound_rejected_without_context_or_data(limit):
    with pytest.raises(ValueError):M.make_ordered_scorer(maximum_cache_entries=limit)
