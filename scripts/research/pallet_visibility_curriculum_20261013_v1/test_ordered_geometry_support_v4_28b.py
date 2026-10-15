"""Cached support ordering/partial-role fixtures, no image or dataset reads."""
from copy import deepcopy
import numpy as np
import pytest

from . import ordered_geometry_support_v4_28b as P
from .test_joint_geometry_guided_lines_v4_28 import fixture,line,Poison


def support_fixture():
    points={2:dict(record=dict(xy=[510.,269.])),1:dict(record=dict(xy=[507.,246.]))}
    node=dict(name='vertical',normal=[1.,0.],actual_inliers=[[493.,float(y)]for y in range(248,268)])
    bank=dict(guide_proposals=[dict(native_edge=[1,2],native_corner=2,guide_id='WRONG_COARSE',
        direction_xy=[0.,1.],vanishing_point_h=None)])
    return points,node,bank,P.OLD.NMS._config(None)


def test_wrong_end_rejected_despite_parallel_geometry_and_opposite_coarse_cannot_rescue():
    points,node,bank,cfg=support_fixture()
    correct=P._support(2,1,np.array([493.,269.]),node,points,bank,cfg)
    wrong=P._support(2,1,np.array([493.,246.]),node,points,bank,cfg)
    assert correct['passes'] and correct['best_supported_fraction']==1
    assert not wrong['passes'] and wrong['best_supported_fraction']==0
    assert wrong['measured_BI_mate_order_used']
    assert not any(a['source']=='WRONG_COARSE'for a in wrong['angle_compatibility'])


def test_existing_fraction_and_one_pixel_side_margin_are_exact():
    points,node,bank,cfg=support_fixture()
    node['actual_inliers']=[[493.,0.],[493.,-1.],[493.,-2.],[493.,2.],[493.,3.]]
    # Ordered direction is up: y<=1 passes exactly3/5, including the1px margin.
    out=P._support(2,1,np.array([493.,0.]),node,points,bank,cfg)
    assert out['passes'] and out['best_supported_fraction']==pytest.approx(.6)
    node['actual_inliers'][1]=[493.,2.]
    assert not P._support(2,1,np.array([493.,0.]),node,points,bank,cfg)['passes']


def test_authoritative_S_U_anchor_gate_precedes_poison_fields():
    args,kw=fixture()
    for k in (1,2):
        out=P.assess_ordered_incident_support(k,0,Poison(),Poison(),geometry=args[2],states=args[3],
            visible_measurements=Poison(),**kw)
        assert out['excluded']


def partial_fixture():
    parent=dict(native_corner=0,native_neighbor=1,measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',
        mask_context_kind='TARGET_TARGET_INTERNAL')
    candidate=line([[x,55.]for x in range(30,48)],[0.,1.],-55.)
    for field in ('RGB_contrast','mask_plus','mask_minus'):candidate.pop(field)
    candidate.update(reason='insufficient_along_edge_span',inlier_count=18,inlier_span_fraction=.535,
        angle_change_deg=4.,observation_confidence=[.5]*18)
    return parent,candidate,P.OLD.NMS._config(None)


def test_partial_saved_TLS_unchanged_but_never_claims_complete_provenance_or_endpoint_guard_pass():
    parent,candidate,cfg=partial_fixture();before=deepcopy(candidate)
    promoted=P._promote_span_only(parent,candidate,cfg)
    assert promoted['normal']==candidate['normal'] and promoted['observed_points']==candidate['observed_points']
    assert candidate==before and promoted['proposal_grade']=='PARTIAL_LINE_PROPOSAL'
    assert promoted['actual_finite_inlier_span_px']==17.
    assert 'RGB_contrast'not in promoted and 'mask_plus'not in promoted
    node=P._graded_node(parent,promoted,'0:0',cfg)
    assert node['old_corrected_endpoint_normal_guard_pass']is None
    assert not node['production_17c_accepted'] and not node['RGB_gradient_lattice_provenance_complete']


@pytest.mark.parametrize('field,value',[('inlier_fraction',.59),('inlier_rms_px',2.),
    ('angle_change_deg',13.),('reason','not_enough_inliers')])
def test_partial_rechecks_all_other_guards_and_only_span_reason(field,value):
    parent,candidate,cfg=partial_fixture();candidate[field]=value
    assert P._promote_span_only(parent,candidate,cfg)is None


def test_tiny_accidental_point_cloud_is_not_promoted():
    parent,candidate,cfg=partial_fixture()
    candidate['observed_points']=[[30+i*.1,55.]for i in range(18)]
    assert P._promote_span_only(parent,candidate,cfg)is None


def test_partial_stored_fraction_matches_actual_inlier_mask():
    parent,candidate,cfg=partial_fixture();candidate['inlier_fraction']=.75
    assert P._promote_span_only(parent,candidate,cfg)is None


@pytest.mark.parametrize('value',[True,'0',.5])
def test_public_native_ID_validation_precedes_poison_fields(value):
    args,kw=fixture()
    with pytest.raises(ValueError,match='Strict native'):
        P.assess_ordered_incident_support(value,1,Poison(),Poison(),geometry=args[2],states=args[3],
            visible_measurements=Poison(),**kw)


def test_public_nonincident_BI_pair_rejected_before_poison_fields():
    args,kw=fixture();args[3][2]='I';args[2]['initial_self_hidden'][2]=False
    with pytest.raises(ValueError,match='incident'):
        P.assess_ordered_incident_support(0,2,Poison(),Poison(),geometry=args[2],states=args[3],
            visible_measurements=Poison(),**kw)


def graph_fixture():
    args,kw=fixture();g,states,fit=args[2:]
    g['centroid']=[4.,4.,4.];g['initial_self_hidden'][3]=False;states[3]='I'
    fit.append(dict(native_corner=3,state='I',xy=[30.,25.],source='VISIBLE_BACKBONE_FALLBACK',
        covariance_px2=[[3.,0.],[0.,3.]],fit_covariance_px2=[[25.,0.],[0.,25.]],fit_reliability_weight=.8))
    top=line([[x,25.]for x in range(30,71)],[0.,1.],-25.)
    vertical=line([[30.,y]for y in range(30,51)],[1.,0.],-30.)
    parent,bottom,_=partial_fixture()
    observation=dict(edge_records=[dict(parent,supported=True,chains=[top],rejected_chains=[bottom]),
        dict(native_corner=0,native_neighbor=3,supported=True,chains=[vertical],rejected_chains=[])])
    return observation,g,states,fit,kw


def test_cached_graph_selects_directed_partial_alternative_over_wrong_end_without_new_pixels():
    observation,g,states,fit,kw=graph_fixture();before=deepcopy(observation)
    old=P.OLD.rank_joint_geometry_line_graph(observation,g,states,fit,**kw)
    new=P.rank_ordered_geometry_line_graph(observation,g,states,fit,**kw)
    chosen={n['name']:n for n in new['line_candidates']if n['name']in new['selected_line_names']}
    assert any(n['proposal_grade']==P.PARTIAL_GRADE for n in chosen.values())
    assert new['partial_cached_TLS_promotions'] and observation==before
    assert all(not a['ordered_support_unavailable']for a in new['joint_alternatives'])
    assert new['U_junctions_or_measurements_created']==0 and not new['full_KP_or_PnP_adoption']
    old_selected=[n for n in old['line_candidates']if n['name']in old['selected_line_names']]
    assert any(n['offset']==-25 for n in old_selected)


def test_single_line_parent_ranking_energy_unchanged_when_ordering_not_applicable():
    observation,g,states,fit,kw=graph_fixture();observation['edge_records']=observation['edge_records'][1:]
    a=P.OLD.rank_joint_geometry_line_graph(observation,g,states,fit,**kw)
    b=P.rank_ordered_geometry_line_graph(observation,g,states,fit,**kw)
    assert a['selected_line_names']==b['selected_line_names']
    assert a['joint_alternatives'][0]['total_engineering_energy']==b['joint_alternatives'][0]['total_engineering_energy']
