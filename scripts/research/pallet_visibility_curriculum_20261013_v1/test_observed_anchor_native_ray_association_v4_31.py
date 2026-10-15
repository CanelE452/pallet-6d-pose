"""Synthetic admission/rank-only tests; no cached rows or actual measurements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from . import observed_anchor_native_ray_association_v4_31 as V
from .test_geometry_graph_exact_memo_v4_29 import graph_fixture


class Poison:
    def __array__(self,*args,**kwargs):raise AssertionError('Excluded coordinate converted')
    def __getitem__(self,key):raise AssertionError('Excluded payload accessed')
    def get(self,*args,**kwargs):raise AssertionError('Excluded payload accessed')


def fixture(*,mate_state='I',kind='FROZEN17C_ORIGINAL_MODEL_SEED',anchor_kind=None):
    q=[10.,19.];other=[40.,19.]
    ref=dict(guide_id='ORIGINAL_INITIAL',guide_kind=kind,
        actual_search_anchor_xy=q.copy(),actual_search_direction_xy=[1.,0.])
    if anchor_kind is not None:ref['search_anchor_kind']=anchor_kind
    node=dict(name='0-1/7:3',native_edge=[0,1],native_corner=0,native_neighbor=1,
        normal=[0.,1.],offset=0.,provenance=['7:3'],
        line_record=dict(guide_reference=ref,generating_guide_direction_xy=[1.,0.]))
    points={0:dict(record=dict(xy=q.copy(),source='RGB_CORRECTED')),
        1:dict(record=dict(xy=other,source='VISIBLE_BACKBONE_FALLBACK'))}
    states=['B',mate_state]+['S']*6;hidden=[False,False]+[True]*6
    cfg=dict(normal_search_radius_px=43.,maximum_corner_shift_px=12.)
    geometry=dict(initial_projection_xy=[[110.,35.],[150.,35.]]+[Poison()for _ in range(6)])
    return node,points,states,hidden,cfg,geometry


def assess(values):
    node,points,states,hidden,cfg,geometry=values
    return V.assess_observed_anchor_native_ray_association(node,points,states,hidden,cfg,geometry=geometry)


def test_initialS_and_nonBI_source_gate_before_all_payload():
    values=fixture();node,points,states,hidden,cfg,geometry=values
    node['normal']=node['line_record']=Poison();points[0]=Poison();geometry['initial_projection_xy']=Poison()
    hidden[1]=True
    assert assess(values)['reason']=='initial_S_gate_before_line_q_or_reference_payload'
    hidden[1]=False;states[0]='U'
    assert assess(values)['reason']=='source_anchor_must_be_original_eligible_BI_before_payload'


def test_original_initial_and_measured_BI_query_paths_are_retained():
    original=fixture();a=assess(original)
    assert a['passes']and a['reference_source']=='IMMUTABLE_OBSERVED_BI_MATE_MINUS_SOURCE'
    measured=fixture(kind='MEASURED_BI_ENDPOINT_DIRECTION',anchor_kind='OBSERVED_BI_REFERENCE_BAND')
    assert assess(measured)['passes']
    vp=fixture(kind='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR',anchor_kind='OBSERVED_BI_REFERENCE_BAND')
    assert assess(vp)['passes']


def test_model_tag_rejected_even_when_query_anchor_equals_q():
    values=fixture(kind='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR',anchor_kind='MODEL_PROJECTION_SEARCH_ONLY_BAND')
    assert assess(values)['reason']=='model_projection_query_is_not_an_observed_q_query'


def test_actual_anchor_numeric_match_is_metadata_tolerance_not_spatial_correction_cap():
    values=fixture();ref=values[0]['line_record']['guide_reference']
    ref['actual_search_anchor_xy'][0]+=5e-7
    assert assess(values)['passes']
    ref['actual_search_anchor_xy'][0]+=2e-6
    assert assess(values)['reason']=='actual_query_anchor_does_not_equal_original_source_BI_q'
    values=fixture();result=assess(values)
    # A19px normal correction remains within the SAME43px normal reach. The
    # anchor metadata equality test does not reinstate a12px correction cap.
    assert result['passes']and result['normal_association']['endpoints'][0]['absolute_distance_px']==19.
    assert result['metadata_anchor_max_absolute_difference_px']==0.


def test_same_parent43_normal_gate_applies_to_both_BI_endpoints():
    values=fixture();values[1][1]['record']['xy'][1]=44.
    assert assess(values)['reason']=='parent_V30_original_BI_normal_gate_failed'
    values[1][1]['record']['xy'][1]=43.
    assert assess(values)['passes']


def test_bothBI_opposite_query_rejects_without_trial_pose_sign_rescue():
    values=fixture();line=values[0]['line_record'];line['guide_reference']['actual_search_direction_xy']=[-1.,0.]
    line['generating_guide_direction_xy']=[-1.,0.]
    values[5]['initial_projection_xy']=Poison()
    result=assess(values)
    assert result['reason']=='actual_query_direction_opposes_fixed_native_ray'
    assert result['signed_direction_cosine']==-1. and not result['pose_selected_orientation']


def test_short_BI_ray_and_missing_BI_mate_abstain_not_model_fallback():
    values=fixture();values[1][1]['record']['xy']=[19.,19.]
    values[5]['initial_projection_xy']=Poison()
    assert assess(values)['reason']=='fixed_native_reference_ray_shorter_than_existing10px'
    values=fixture();del values[1][1]
    values[0]['line_record']=values[0]['normal']=Poison();values[5]['initial_projection_xy']=Poison()
    assert assess(values)['reason']=='both_BI_mate_missing_immutable_measurement'


@pytest.mark.parametrize('mate_state',['O','T','U'])
def test_unknownmate_only_original_model_sign_and_no_unknown_q_payload(mate_state):
    values=fixture(mate_state=mate_state);values[1][1]=Poison()
    result=assess(values)
    assert result['passes']and result['reference_source']=='ORIGINAL_INITIAL_MODEL_NATIVE_RAY_FOR_UNKNOWN_MATE'
    assert result['fixed_reference_delta_xy']==[40.,0.]
    # Reversing the frozen original model ray changes eligibility, even if a
    # different pose guide could point the other way. No pose selection occurs.
    values[5]['initial_projection_xy'][1]=[70.,35.]
    assert assess(values)['reason']=='actual_query_direction_opposes_fixed_native_ray'


def test_original_unknown_model_reference_short_and_orthogonal_ambiguity_abstain():
    values=fixture(mate_state='U');values[1][1]=Poison()
    values[5]['initial_projection_xy'][1]=[115.,35.]
    assert assess(values)['reason']=='fixed_native_reference_ray_shorter_than_existing10px'
    values=fixture();line=values[0]['line_record'];line['guide_reference']['actual_search_direction_xy']=[0.,1.]
    line['generating_guide_direction_xy']=[0.,1.]
    assert assess(values)['reason']=='fixed_native_direction_orthogonal_ambiguity'


def test_unoriented_TLS_sign_reversal_does_not_change_native_query_sign():
    values=fixture();before=assess(values)
    values[0]['normal']=[-x for x in values[0]['normal']];values[0]['offset']=-values[0]['offset']
    after=assess(values)
    assert before['passes']==after['passes']is True
    assert before['signed_direction_cosine']==after['signed_direction_cosine']==1.
    assert [r['absolute_distance_px']for r in before['normal_association']['endpoints']]==[
        r['absolute_distance_px']for r in after['normal_association']['endpoints']]


def test_missing_or_disagreeing_reference_is_not_fabricated_or_relocated():
    values=fixture();before=deepcopy(values[0]);values[0]['line_record'].pop('guide_reference')
    assert assess(values)['reason']=='missing_original_actual_query_reference'
    values=fixture();values[0]['line_record']['generating_guide_direction_xy']=[0.,1.]
    assert assess(values)['reason']=='actual_query_direction_alias_disagrees'
    assert before['line_record']['guide_reference']['actual_search_anchor_xy']==[10.,19.]


def augmented_graph():
    args,kw,obs=graph_fixture();measurements={r['native_corner']:r for r in args[4]}
    for ray in obs['edge_records']:
        k,mate=ray['native_corner'],ray['native_neighbor'];q=np.asarray(measurements[k]['xy']);delta=np.asarray(measurements[mate]['xy'])-q
        direction=(delta/np.linalg.norm(delta)).tolist()
        for chain in ray['chains']:
            chain['guide_reference']=dict(guide_id='ORIGINAL_INITIAL',guide_kind='FROZEN17C_ORIGINAL_MODEL_SEED',
                actual_search_anchor_xy=q.tolist(),actual_search_direction_xy=direction)
            chain['generating_guide_direction_xy']=direction
    return args,kw,obs


@pytest.mark.parametrize('cache',[1,4096])
def test_all_admitted_entire_graph_exact_to_R30_and_sources_globals_archive_unchanged(cache):
    args,kw,obs=augmented_graph();before=deepcopy((args,kw,obs))
    paths=[Path(m.__file__)for m in(V.F,V.P,V.G,V.P.ORDER,V.B,V.M)]
    source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    globals_before={k:id(v)for k,v in V.G.rank_joint_geometry_line_graph.__globals__.items()}
    cfg=dict(graph_configuration=dict(maximum_beam_states=5,reported_joint_alternatives=5))
    expected=V.F.rank_observed_BI_associated_graph(obs,args[2],args[3],args[4],**kw,configuration=cfg,pallet_mask=args[1],
        maximum_cache_entries=cache,intersection_cache_limit=cache)
    result=V.rank_observed_anchor_native_ray_graph(obs,args[2],args[3],args[4],**kw,configuration=cfg,pallet_mask=args[1],
        maximum_cache_entries=cache,intersection_cache_limit=cache)
    assert result['rejected_node_count']==0 and result['candidate_nodes_checked']==12
    assert json.dumps(expected['graph'],sort_keys=True)==json.dumps(result['graph'],sort_keys=True)
    assert V.G.NMS.G.C.finite((args,kw,obs))==V.G.NMS.G.C.finite(before)
    assert globals_before=={k:id(v)for k,v in V.G.rank_joint_geometry_line_graph.__globals__.items()}
    assert source_before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}


def test_reject_before_dedup_keeps_good_original_index_alias_and_raw_provenance():
    args,kw,obs=augmented_graph();good=deepcopy(obs['edge_records'][0]['chains'][0]);bad=deepcopy(good)
    bad['guide_reference']['search_anchor_kind']='MODEL_PROJECTION_SEARCH_ONLY_BAND'
    bad['guide_reference']['guide_kind']='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR'
    obs['edge_records'][0]['chains']=[bad,good]
    before=deepcopy(obs)
    result=V.rank_observed_anchor_native_ray_graph(obs,args[2],args[3],args[4],**kw,pallet_mask=args[1])
    rejected=[r for r in result['node_association_ledger']if not r['passes']]
    assert len(rejected)==1 and rejected[0]['node_provenance']==['0:0']
    assert any(n['provenance']==['0:1']for n in result['graph']['line_candidates'])
    assert obs==before
    assert result['selection_profile']==V.SELECTION_PROFILE and result['parent_selection_profile']==V.F.SELECTION_PROFILE
    assert result['new_CAL46_noise_and_uniform_validation_required']and not result['old_V29_or_V30_noise_validated_for_this_population']
    assert result['new_RGB_NMS_TLS_posefit_labels_or_precise_KPs']==0
