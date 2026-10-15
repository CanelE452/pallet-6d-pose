"""Focused synthetic image/geometry contracts; no dataset, GT, model or pose fit."""
from copy import deepcopy
import cv2
import numpy as np
import pytest

from . import joint_geometry_guided_lines_v4_28 as P
from .test_face_aware_partial_bi_observations_v4_13d import fixture as old_fixture, Poison


def fixture():
    args,kw=old_fixture()
    args[2].update(K=[[100.,0,80.],[0,100.,80.],[0,0,1.]],physical_registration_Q=np.eye(3).tolist())
    return args,dict(image_binding=kw['image_binding'])


def test_finite_infinite_homogeneous_scale_sign_and_singular_vp():
    anchor=np.array([21.,37.])
    for vp in ([1.,0,0],[0,1.,0],[120.,30.,2.]):
        first=P.vanishing_direction_at(vp,anchor)
        for scale in (-1e7,-.1,.2,1e12):
            assert np.allclose(P.vanishing_direction_at(np.asarray(vp)*scale,anchor),first,atol=1e-12)
    with pytest.raises(ValueError,match='no line direction'):
        P.vanishing_direction_at([21.,37.,1.],anchor)


def test_native_basis_reexpression_physical_axis_and_nonzero_center_invariant():
    args,_=fixture();g=args[2]
    g['native_xyz']=(np.asarray(g['native_xyz'])+[.2,-.4,.1]).tolist()
    Q=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    alias={k:deepcopy(v) for k,v in g.items() if k!='initial_projection_xy'}
    alias['initial_projection_xy']=g['initial_projection_xy']
    alias['native_xyz']=(np.asarray(g['native_xyz'])@Q.T).tolist()
    alias['R_cf']=(np.asarray(g['R_cf'])@Q.T).tolist()
    alias['physical_registration_Q']=Q.tolist()
    a,b=P._geometry(g),P._geometry(alias)
    for k in range(8):assert np.allclose(a['projection'][k],b['projection'][k],atol=1e-12)
    assert P._edge_axis(a,0,1)==P._edge_axis(b,0,1)==0
    for u,v in zip(a['vanishing'],b['vanishing']):assert np.allclose(u,v,atol=1e-12)
    guides=P.prepare_joint_geometry_guides(g,args[3],args[4],image_binding=dict(sha256='0'*64,image_hw=[160,160]),
        pose_guides=[dict(guide_id='Q_ALIAS',source_kind='ORIGINAL_TASK_COARSE',geometry=alias)])
    assert guides['same_physical_pose_aliases']==[['ORIGINAL_INITIAL','Q_ALIAS']]


def test_S_U_gates_before_observation_coordinate_access_and_selected_pose_rejected():
    args,kw=fixture()
    for k in range(1,8):args[-1].append(dict(native_corner=k,state=args[3][k],xy=Poison(),covariance_px2=Poison()))
    out=P.prepare_joint_geometry_guides(args[2],args[3],args[4],**kw)
    assert out['guide_proposals'] and out['S_or_U_observation_coordinate_reads']==0
    assert all(p['native_corner']==0 and p['native_neighbor']==1 for p in out['guide_proposals'])
    with pytest.raises(ValueError,match='no selected e'):
        P.prepare_joint_geometry_guides(args[2],args[3],args[4],**kw,
            pose_guides=[dict(guide_id='e',source_kind='SELECTED_E',geometry=args[2])])


def test_actual_new_strip_direction_differs_from_BI_endpoint_direction_without_q_replacement():
    args,kw=fixture();args[3][1]='I'
    args[4].append(dict(native_corner=1,state='I',xy=[130.,75.],source='VISIBLE_BACKBONE_FALLBACK',
        covariance_px2=[[3.,0],[0,3.]],fit_covariance_px2=[[25.,0],[0,25.]],fit_reliability_weight=.8))
    guides=P.prepare_joint_geometry_guides(args[2],args[3],args[4],**kw)
    proposal=next(p for p in guides['guide_proposals'] if p['native_corner']==0
        and p['guide_kind']=='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR'
        and p['search_anchor_kind']=='OBSERVED_BI_REFERENCE_BAND')
    image,mask=args[:2];lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=(cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8)
    points=P.NMS.G._eligible_points(args[-1],args[3],args[2]['initial_self_hidden'],image.shape[:2])
    before=deepcopy(args[-1])
    out=P._guided_edge(image,mask,lab,gradients,proposal,points,P.NMS._config(None))
    assert out['actual_query_strip_audit']
    assert np.allclose(out['actual_query_strip_audit'][0]['search_direction_xy'],[1.,0.])
    assert args[-1]==before
    assert out['chains'] or out['rejected_chains']
    line=next(c for c in [*out['chains'],*out['rejected_chains']] if 'normal' in c)
    assert abs(line['normal'][0])<1e-5
    assert 'query_strip_actual_support' in line and 'legacy_observed_endpoint_reference_support' in line


def test_unknown_mate_protection_and_visibility_context_precede_every_additional_RGB_probe(monkeypatch):
    args,kw=fixture();seen=[];original=P.NMS._sample
    guides=P.prepare_joint_geometry_guides(args[2],args[3],args[4],**kw)
    proposal=next(p for p in guides['guide_proposals'] if p['search_anchor_kind']=='OBSERVED_BI_REFERENCE_BAND')
    def sample(array,xy):seen.append(np.asarray(xy).copy());return original(array,xy)
    monkeypatch.setattr(P.NMS,'_sample',sample)
    class Context:
        calls=0
        def query_allowed(self,k,mate,xy,context_radius_px):
            self.calls+=1
            return np.asarray(xy)[:,0]<75-context_radius_px
    context=Context();image,mask=args[:2]
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    gradients=(cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8)
    points=P.NMS.G._eligible_points(args[-1],args[3],args[2]['initial_self_hidden'],image.shape[:2])
    P._guided_edge(image,mask,lab,gradients,proposal,points,P.NMS._config(None),context)
    assert context.calls and seen
    assert max(x[:,0].max() for x in seen)<75


def test_global_external_contour_distinguishes_internal_mask_hole_from_exterior():
    mask=np.zeros((120,160));mask[15:105,15:145]=1;mask[45:75,55:105]=0
    cfg=P.NMS._config(None);context=P._external_context(mask,mask.shape,cfg)
    exterior=dict(normal=[1.,0],actual_inliers=np.column_stack([np.ones(20)*15,np.arange(30,50)]).tolist())
    hole=dict(normal=[1.,0],actual_inliers=np.column_stack([np.ones(20)*55,np.arange(47,67)]).tolist())
    a,b=(P._line_external_context(node,context,cfg) for node in (exterior,hole))
    assert a['external_support_fraction']>.9 and b['external_support_fraction']==0
    assert a['mask_holes_not_external'] and b['same_foreground_component_fraction']==1


def line(points,normal,offset):
    points=np.asarray(points,float)
    return dict(observed_points=points.tolist(),inlier_mask=[True]*len(points),normal=normal,offset=offset,
        inlier_fraction=1.,inlier_span_fraction=1.,inlier_rms_px=0.,RGB_contrast=[30.]*len(points),
        mask_plus=[1.]*len(points),mask_minus=[0.]*len(points))


def test_rounded_line_extension_reports_finite_support_not_a_measured_corner():
    a=line([[x,40] for x in range(40,81)],[0,1.],-40.)
    b=line([[40,y] for y in range(40,81)],[1.,0],-40.)
    out=P.rounded_BI_junction([a,b],P.NMS._config(None),[120,160])
    assert out['available'] and np.allclose(out['candidate_extrapolated_intersection_xy'],[40,40])
    assert not out['adopted'] and not out['full_2D_corner_measurement']
    assert out['within_canvas'] and out['available_means_computable_not_production_guard_pass']
    assert all(r['excluded_actual_points']>0 for r in out['straight_actual_support'])
    assert not P.rounded_BI_junction([a,a],P.NMS._config(None))['available']


def test_same_support_different_TLS_fits_preserved_and_old_guard_failure_archived():
    args,kw=fixture();args[3][0]='B';args[4][0]['state']='B'
    a=line([[x,60] for x in range(35,65)],[0,1.],-60.)
    b=deepcopy(a);b['normal']=[1e-3,np.sqrt(1-1e-6)];b['offset']=-60.
    b.update(reason='existing_corrected_endpoint_normal_guard_failed',corrected_endpoint_normal_guard_pass=False)
    observations=dict(edge_records=[dict(native_corner=0,native_neighbor=1,supported=True,chains=[a],rejected_chains=[b])])
    out=P.rank_joint_geometry_line_graph(observations,args[2],args[3],args[4],**kw,pallet_mask=args[1])
    assert out['actual_unique_line_candidates']==2
    assert len({n['support_id'] for n in out['line_candidates']})==1
    assert len({n['exact_fit_id'] for n in out['line_candidates']})==2
    assert any(n['production_guard_bypassed_only_for_unadopted_line_candidate'] for n in out['line_candidates'])
    assert out['U_junctions_or_measurements_created']==0


def test_blank_actual_RGB_abstains_and_does_not_mutate_visible_or_create_U_junction():
    args,kw=fixture();args[0][:]=50;before=deepcopy(args[-1])
    out=P.observe_joint_geometry_guided_lines(*args,**kw)
    assert not out['joint_graph']['selected_line_names']
    assert args[-1]==before and out['old_points_changed']==0
    assert out['joint_graph']['U_junctions_or_measurements_created']==0


def test_measured_named_axis_orthogonality_with_skew_K_finite_and_infinite_VP():
    K=np.array([[500.,17.,83.],[0,490.,61.],[0,0,1.]])
    edges=[[(0,1),(4,5)],[(0,3),(1,2)],[(0,4),(1,5)]]
    nodes=[];axis_map={}
    for axis in range(3):
        vp=K[:,axis]
        for i,edge in enumerate(edges[axis]):
            anchor=np.array([25.+axis*17+i*48,35.+i*61+axis*7,1.])
            line=np.cross(vp,anchor);line=line/np.linalg.norm(line[:2])
            name=f'{axis}/{i}'
            nodes.append(dict(name=name,native_edge=list(edge),normal=line[:2].tolist(),offset=float(line[2]),
                support_anchor_xy=anchor[:2].tolist(),image_quality_energy=0.))
            axis_map[edge]=axis
    guide=dict(_cached_node_guide_costs={n['name']:{} for n in nodes},_cached_edge_axes=axis_map,
        _cached_K=K,_cached_intersections={},image_binding=dict(image_hw=[180,240]))
    score=P._score_graph(nodes,{},['S']*8,guide,P.NMS._config(None),P.DEFAULT_GRAPH_CONFIGURATION)
    assert score['independently_measured_axis_ray_count']==3
    assert score['image_measured_axis_orthogonality_energy']<1e-20
    bad=deepcopy(nodes)
    # Same actual parallel family falsely assigned to two physical axes.
    for i in range(2):
        for key in ('normal','offset','support_anchor_xy'):bad[i+2][key]=deepcopy(bad[i][key])
    score=P._score_graph(bad,{},['S']*8,guide,P.NMS._config(None),P.DEFAULT_GRAPH_CONFIGURATION)
    assert score['image_measured_axis_orthogonality_energy']>.30
    guide['_cached_K']=K*1e9
    again=P._score_graph(bad,{},['S']*8,guide,P.NMS._config(None),P.DEFAULT_GRAPH_CONFIGURATION)
    assert again['image_measured_axis_orthogonality_energy']==pytest.approx(score['image_measured_axis_orthogonality_energy'])


def test_independent_occluder_context_blocks_strong_base_boundary_before_query_and_archived_activation(monkeypatch):
    args,kw=fixture()
    historical=P.NMS.observe_image_supported_edge_chains(*args,**kw)
    assert any(r.get('supported') for r in historical['edge_records'])
    class IndependentOccluder:
        def query_allowed(self,k,mate,xy,context_radius_px):
            # Its boundary is 5px from the strong transition, inside the7px
            # context but outside the smaller RGB-only footprint.
            yy=np.asarray(xy)[:,1]
            return (yy<65-context_radius_px)|(yy>75+context_radius_px)
    seen=[];original=P.NMS._sample
    def sample(array,xy):seen.append(np.asarray(xy).copy());return original(array,xy)
    monkeypatch.setattr(P.NMS,'_sample',sample)
    out=P.observe_joint_geometry_guided_lines(*args,**kw,visibility_context=IndependentOccluder())
    assert not out['joint_graph']['selected_line_names']
    assert out['current_visibility_context_applied_before_base_and_additional_queries']
    assert seen and not any(np.any((a[:,1]>=65)&(a[:,1]<=75)) for a in seen)
    out=P.observe_joint_geometry_guided_lines(*args,**kw,base_observation=historical,
        visibility_context=IndependentOccluder())
    assert out['base_observation']==historical
    assert not out['joint_graph']['selected_line_names']
    assert out['joint_graph']['historical_line_candidates_excluded_by_visibility']


def test_historical_visibility_rejection_precedes_local_mask_context_lookup(monkeypatch):
    args,kw=fixture()
    historical=P.NMS.observe_image_supported_edge_chains(*args,**kw)
    class Reject:
        def query_allowed(self,k,mate,xy,context_radius_px):return np.zeros(len(xy),bool)
    def forbidden(*args,**kwargs):raise AssertionError('Unsafe local mask lookup')
    monkeypatch.setattr(P,'_line_external_context',forbidden)
    graph=P.rank_joint_geometry_line_graph(historical,args[2],args[3],args[-1],**kw,
        pallet_mask=args[1],visibility_context=Reject())
    assert not graph['line_candidates'] and graph['historical_line_candidates_excluded_by_visibility']
