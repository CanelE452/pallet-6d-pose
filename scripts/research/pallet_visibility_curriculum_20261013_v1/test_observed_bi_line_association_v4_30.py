from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from . import observed_bi_line_association_v4_30 as P
from .test_geometry_graph_exact_memo_v4_29 import graph_fixture


class Poison:
    def __array__(self,*args,**kwargs):raise AssertionError('Excluded coordinate read')


def data(first=(0.,0.),second=(1.,0.),states=('B','I')):
    return (dict(native_edge=[0,1],normal=[0.,1.],offset=0.),
        {0:dict(record=dict(xy=list(first),source='RGB')),1:dict(record=dict(xy=list(second),source='VISIBLE_BACKBONE_FALLBACK'))},
        list(states)+['S']*6,[False,False]+[True]*6,dict(normal_search_radius_px=43.))


def test_remote_model_anchor_cannot_admit_a_line_far_from_original_BI_q():
    values=data((1.,100.),(10.,100.));before=deepcopy(values)
    result=P.assess_observed_BI_line_association(*values)
    assert not result['passes']and all(e['absolute_distance_px']==100. for e in result['endpoints'])
    assert values==before


def test_both_eligible_BI_endpoints_must_pass_and_exact43_boundary_is_closed():
    values=data((1.,19.),(10.,44.));result=P.assess_observed_BI_line_association(*values)
    assert not result['passes']and [e['passes']for e in result['endpoints']]==[True,False]
    values[1][1]['record']['xy'][1]=43.
    assert P.assess_observed_BI_line_association(*values)['passes']


def test_large_alongline_move_and19px_normal_correction_have_no12px_corner_cap():
    values=data((1e6,19.),(-1e6,-19.))
    assert P.assess_observed_BI_line_association(*values)['passes']


def test_unknown_mate_and_initialS_gate_before_any_excluded_coordinate_or_line_fields():
    values=data(states=('B','U'));values[1][1]['record']=dict(xy=Poison())
    result=P.assess_observed_BI_line_association(*values)
    assert result['passes']and len(result['endpoints'])==1
    node,points,states,hidden,cfg=values;hidden[1]=True;node['normal']=Poison();node['offset']=Poison()
    assert not P.assess_observed_BI_line_association(node,points,states,hidden,cfg)['passes']


def test_no_eligible_anchor_abstains_before_line_fields_and_normal_sign_is_irrelevant():
    values=data(states=('U','U'));values[0]['normal']=Poison()
    assert not P.assess_observed_BI_line_association(*values)['passes']
    values=data((1.,19.),(2.,-43.));a=P.assess_observed_BI_line_association(*values)
    values[0]['normal']=[0.,-1.];b=P.assess_observed_BI_line_association(*values)
    assert a['passes']==b['passes']and [e['absolute_distance_px']for e in a['endpoints']]==[e['absolute_distance_px']for e in b['endpoints']]


def test_all_admitted_full_graph_exact_and_original_globals_sources_inputs_immutable():
    args,kw,obs=graph_fixture();before=deepcopy((args,kw,obs))
    paths=[Path(m.__file__)for m in(P.P,P.G,P.P.ORDER,P.B,P.M)]
    source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    globals_before={k:id(v)for k,v in P.G.rank_joint_geometry_line_graph.__globals__.items()}
    config=dict(graph_configuration=dict(maximum_beam_states=5,reported_joint_alternatives=5))
    expected=P.P._rank_fresh(obs,args[2],args[3],args[4],**kw,configuration=config,
        pose_guides=None,visibility_context=None,pallet_mask=args[1])
    actual=P.rank_observed_BI_associated_graph(obs,args[2],args[3],args[4],**kw,configuration=config,pallet_mask=args[1])
    assert actual['rejected_node_count']==0
    assert json.dumps(actual['graph'],sort_keys=True)==json.dumps(expected,sort_keys=True)
    assert P.G.NMS.G.C.finite((args,kw,obs))==P.G.NMS.G.C.finite(before)
    assert globals_before=={k:id(v)for k,v in P.G.rank_joint_geometry_line_graph.__globals__.items()}
    assert source_before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}


def test_filtered_chain_keeps_original_refs_rawarchive_and_new_population_label():
    args,kw,obs=graph_fixture();before=deepcopy(obs)
    # Keep ray0 chain0 but move its other variants beyond the same43px
    # normal reach. Their original indices must never be renumbered.
    for chain in obs['edge_records'][0]['chains'][1:]:
        chain['observed_points']=(np.asarray(chain['observed_points'])+[0.,70.]).tolist();chain['offset']-=70.
    before=deepcopy(obs)
    actual=P.rank_observed_BI_associated_graph(obs,args[2],args[3],args[4],**kw,pallet_mask=args[1])
    rejected=[r for r in actual['node_association_ledger']if not r['passes']]
    assert len(rejected)==2 and {r['node_provenance'][0]for r in rejected}=={'0:1','0:2'}
    assert any(n['name'].endswith('/0:0')for n in actual['graph']['line_candidates'])
    assert obs==before and actual['selection_profile']==P.SELECTION_PROFILE
    assert not actual['old_V29_noise_validated_for_this_population']
