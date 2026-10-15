"""Synthetic mask/actual-chain topology contracts; no image/model/pose/GT run."""
from copy import deepcopy
import numpy as np
import pytest

from . import common as C
from . import native_edge_topology_junctions_v4_17b as P
from .test_face_aware_partial_bi_observations_v4_13d import Poison


def chain(mask,k,mate,xy,normal,internal=False):
    xy=np.asarray(xy,float);normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    offset=-float(normal@xy.mean(axis=0));tangent=np.array([-normal[1],normal[0]])
    mp=P._sample(mask,xy+normal*1.5);mm=P._sample(mask,xy-normal*1.5)
    return dict(native_corner=k,native_neighbor=mate,native_edge=[k,mate],
        measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',
        mask_context_kind='TARGET_TARGET_INTERNAL' if internal else 'TARGET_BACKGROUND_OUTLINE',
        observed_points=xy.tolist(),inlier_mask=[True]*len(xy),normal=normal.tolist(),offset=offset,
        inlier_count=len(xy),inlier_fraction=1.,inlier_span_fraction=1.,
        inlier_rms_px=float(np.sqrt(np.mean((xy@normal+offset)**2))),angle_change_deg=0.,
        corrected_endpoint_normal_guard_pass=True,
        RGB_gradient_energy=[8.]*len(xy),RGB_alignment=[1.]*len(xy),RGB_contrast=[30.]*len(xy),
        actual_inlier_tangent_interval_px=[float((xy@tangent).min()),float((xy@tangent).max())],
        image_gradient_normal_xy=np.tile(normal,(len(xy),1)).tolist(),mask_plus=mp.tolist(),mask_minus=mm.tolist())


def fixture(internal=False):
    mask=np.zeros((160,160));mask[60:130,60:130]=1.
    if internal:mask[:]=1.
    geometry=dict(native_xyz=C.cuboid(1.,.1,1.).tolist(),
        R_cf=(np.eye(3) if internal else np.diag([-1.,1.,-1.])).tolist(),
        centroid=[4.,4.,4.] if internal else [-4.,4.,4.],image_hw=[160,160],
        initial_self_hidden=[False,False,True,False,True,True,True,True])
    states=['I' if internal else 'B','U','S','U','S','S','S','S']
    visible=[dict(native_corner=0,state=states[0],xy=[60.,60.],source='VISIBLE_BACKBONE_FALLBACK',
                  covariance_px2=[[4.,0.],[0.,4.]])]
    xy1=np.column_stack([np.arange(64.,91.),np.ones(27)*59.5])
    xy3=np.column_stack([np.ones(27)*59.5,np.arange(64.,91.)])
    c1=chain(mask,0,1,xy1,[0,1],internal);c3=chain(mask,0,3,xy3,[1,0],internal)
    role='INTERNAL_TWO_VISIBLE_FACES' if internal else 'OUTLINE_VISIBLE_HIDDEN_FACES'
    binding=dict(sha256='0'*64,image_hw=[160,160])
    obs=dict(schema='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C',configuration=P.CHAINS._config(None),image_binding=binding,
             edge_records=[dict(native_corner=0,native_neighbor=1,supported=True,face_association={'role':role},chains=[c1]),
                           dict(native_corner=0,native_neighbor=3,supported=True,face_association={'role':role},chains=[c3])])
    kw=dict(image_binding=binding,mask_binding=dict(image_sha256='0'*64,mask_pixel_sha256=P.mask_pixel_sha256(mask)))
    return [obs,mask,geometry,states,visible],kw


def test_two_actual_external_chains_same_component_arc_and_wedge_propose_only():
    args,kw=fixture();before=deepcopy(args[-1])
    out=P.propose_native_edge_junctions(*args,**kw);record=out['records'][0]
    assert record['candidate_count']==1 and record['geometrically_unique_within_existing_3px_rule']
    item=record['candidates'][0]
    assert np.allclose(item['intersection_xy'],[59.5,59.5])
    assert item['local_topology']['passed']
    assert item['requires_separate_conditional_bootstrap_before_any_full_point_adoption']
    assert args[-1]==before and out['precise_KP_adoptions']==out['old_q_covariance_state_changes']==0


def test_all_admissible_alternatives_remain_without_best_RMS_selection():
    args,kw=fixture()
    for ray in args[0]['edge_records']:ray['chains'].append(deepcopy(ray['chains'][0]))
    record=P.propose_native_edge_junctions(*args,**kw)['records'][0]
    assert record['candidate_count']==4 and not record['precise_point_selected']
    assert record['candidate_disagreement_px']==0


def test_one_actual_external_chain_remains_partial_not_full_junction():
    args,kw=fixture();args[0]['edge_records']=args[0]['edge_records'][:1]
    record=P.propose_native_edge_junctions(*args,**kw)['records'][0]
    assert record['candidate_count']==0 and len(record['external_chain_refs'])==1


def test_I_internal_creases_are_preserved_separately_never_mixed_with_B():
    args,kw=fixture(True)
    out=P.propose_native_edge_junctions(*args,**kw);record=out['records'][0]
    assert record['candidates']==[] and len(record['internal_chain_refs'])==2
    assert not out['B_outline_and_I_crease_mixed']


def test_S_and_U_anchor_gates_precede_poisoned_observation_and_coordinate_fields():
    args,kw=fixture()
    args[0]['edge_records'].append(dict(native_corner=2,native_neighbor=Poison(),chains=Poison()))
    args[-1].append(dict(native_corner=2,state='S',xy=Poison(),covariance_px2=Poison()))
    out=P.propose_native_edge_junctions(*args,**kw)
    assert out['S_coordinate_reads']==out['U_coordinate_reads']==out['U_corner_queries']==0
    args[3][0]='U';args[-1][0]['xy']=Poison()
    out=P.propose_native_edge_junctions(*args,**kw)
    assert all(not r['candidates'] for r in out['records'])


def test_old_finite_11px_support_limit_is_required():
    args,kw=fixture();ray=args[0]['edge_records'][0]
    xy=np.column_stack([np.arange(76.,103.),np.ones(27)*59.5])
    ray['chains']=[chain(args[1],0,1,xy,[0,1])]
    record=P.propose_native_edge_junctions(*args,**kw)['records'][0]
    assert not record['candidates']
    assert any(x['reason']=='existing_finite_support_shift_or_canvas_guard_failed' for x in record['rejected_pairs'])


def test_nearly_parallel_named_lines_do_not_create_a_precise_junction():
    args,kw=fixture();x=np.arange(64.,91.);xy=np.column_stack([x,59.5+.04*(x-77)])
    args[0]['edge_records'][1]['chains']=[chain(args[1],0,3,xy,[-.04,1.])]
    record=P.propose_native_edge_junctions(*args,**kw)['records'][0]
    assert not record['candidates']
    assert record['rejected_pairs'][0]['reason']=='nearly_parallel_external_lines'


def test_hole_boundary_is_not_an_external_silhouette_junction():
    args,kw=fixture();mask=args[1];mask[:]=0;mask[20:140,20:140]=1;mask[60:100,60:100]=0
    for ray in args[0]['edge_records']:
        old=ray['chains'][0];ray['chains']=[chain(mask,0,ray['native_neighbor'],old['observed_points'],old['normal'])]
    kw['mask_binding']['mask_pixel_sha256']=P.mask_pixel_sha256(mask)
    record=P.propose_native_edge_junctions(*args,**kw)['records'][0]
    assert not record['candidates'] and not record['external_chain_refs']
    assert all(x['audit']['reason']=='target_background_edge_is_not_supported_by_actual_external_contour'
               for x in record['rejected_chain_topology'])


def test_mask_binding_and_archived_actual_context_are_checked():
    args,kw=fixture();kw['mask_binding']['mask_pixel_sha256']='f'*64
    with pytest.raises(ValueError,match='binding'):P.propose_native_edge_junctions(*args,**kw)
    args,kw=fixture();args[0]['edge_records'][0]['chains'][0]['mask_plus']=[0.]*27
    with pytest.raises(ValueError,match='occupancy'):P.propose_native_edge_junctions(*args,**kw)


def test_actual_RGB_support_cannot_be_replaced_by_a_model_or_mask_line():
    args,kw=fixture();args[0]['edge_records'][0]['chains'][0]['RGB_gradient_energy']=[0.]*27
    with pytest.raises(ValueError,match='positive RGB'):P.propose_native_edge_junctions(*args,**kw)
