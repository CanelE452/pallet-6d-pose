"""Two-family geometry and fixed-observation contracts, without real data."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import common as C
from . import statewise_two_wd_family_v4_9 as M


def inputs():
    xyz=C.cuboid(1.1,.11,1.3)
    R=Rotation.from_euler("xyz",[2.,-35.,1.],degrees=True).as_matrix()
    Q=M.RY90.copy()
    pose=dict(R=R.tolist(),t=[.1,.15,3.5],physical_registration_Q=Q.tolist(),R_physical=(R@Q).tolist())
    return xyz,pose


def test_swapping_family_preserves_physical_shape_rendered_box_and_pose_including_Q90():
    xyz,pose=inputs();families=M.make_wd_families(xyz,pose)
    assert len(families)==2
    second=families[1]
    np.testing.assert_allclose(second['cf_extents'],[1.3,.11,1.1])
    np.testing.assert_allclose(second['physical_registration_Q'],M.RY90@pose['physical_registration_Q'])
    np.testing.assert_allclose(second['coarse_pose']['R_physical'],pose['R_physical'],atol=1e-12)
    assert second['initial_rendered_vertex_set_gap_m']<1e-12
    assert second['registered_physical_vertex_set_gap_m']<1e-12
    np.testing.assert_allclose(families[0]['extent_diagonal_m'],second['extent_diagonal_m'],atol=1e-15,rtol=0)


def test_supplied_initialization_basis_changes_and_inputs_do_not_mutate():
    xyz,pose=inputs();start={'R':pose['R'],'t':pose['t'],'name':'KNOWN_INITIAL'}
    original=deepcopy(start);families=M.make_wd_families(xyz,pose,(s for s in [start]))
    assert start==original
    np.testing.assert_allclose(families[1]['initializations'][0]['R'],np.asarray(start['R'])@M.RY90.T)


def test_square_model_does_not_create_duplicate_family():
    _,pose=inputs()
    assert len(M.make_wd_families(C.cuboid(1.,.1,1.),pose))==1


def test_same_queries_covariances_states_ranges_partial_and_prior0_reach_both_fits():
    xyz,pose=inputs();states=['B','U','O','B','B','T','S','I']
    visible=[{'native_corner':0,'xy':[40.,50.],'covariance_px2':[[4.,0.],[0.,4.]]}]
    domains=[{'native_corner':2,'state':'O','center_xy':[80.,90.]}]
    partial=[{'native_corner':0,'lines':'opaque frozen evidence'}];regions=[{'native_corner':1,'center_xy':[500.,600.]}]
    saved=deepcopy((states,visible,domains,partial,regions));calls=[]
    def solver(X,K,S,V,D,hw,P,**kw):
        calls.append(deepcopy((S,V,D,kw)))
        Q=np.asarray(P['physical_registration_Q']);R=np.asarray(P['R'])
        return {'accepted':True,'objective':float(np.ptp(np.asarray(X),axis=0)[0]),
            'R':R.tolist(),'t':P['t'],'R_physical':(R@Q).tolist()}
    result=M.solve_two_wd_families(xyz,np.eye(3),states,visible,domains,[480,640],pose,
        config={'prior_weight':4.},boundary_observations=partial,u_regions=regions,solver_callable=solver)
    assert result['effective_prior_weight']==0 and result['requested_prior_weight']==4
    assert len(calls)==2 and calls[0]==calls[1]
    assert calls[0][3]['config']['prior_weight']==0
    assert (states,visible,domains,partial,regions)==saved


def test_lower_objective_selects_family_and_complete_tie_uses_original_without_GT():
    xyz,pose=inputs();states=['U']*8
    def solver(X,K,S,V,D,hw,P,**kw):
        R=np.asarray(P['R']);Q=np.asarray(P['physical_registration_Q'])
        return dict(accepted=True,objective=0. if np.ptp(np.asarray(X),axis=0)[0]>1.2 else 1.,
                    R=R.tolist(),t=P['t'],R_physical=(R@Q).tolist())
    result=M.solve_two_wd_families(xyz,np.eye(3),states,[],[],[480,640],pose,solver_callable=solver)
    assert result['selected_family']=='SWAPPED_WD'
    def tie(*a,**kw):
        result=solver(*a,**kw);result['objective']=1.;return result
    assert M.solve_two_wd_families(xyz,np.eye(3),states,[],[],[480,640],pose,solver_callable=tie)['selected_family']=='ORIGINAL_WD'


def test_bad_native_order_or_physical_registration_binding_reject():
    xyz,pose=inputs()
    with pytest.raises(ValueError,match='ordered'):
        M.make_wd_families(xyz[::-1],pose)
    pose['R_physical']=np.eye(3).tolist()
    with pytest.raises(ValueError,match='binding'):
        M.make_wd_families(xyz,pose)


def test_true_swapped_visible_rectangle_recovered_without_reference_input():
    original_xyz=C.cuboid(1.1,.11,1.3)
    true_native_xyz=C.cuboid(1.3,.11,1.1)
    R=Rotation.from_euler('xyz',[3.,-35.,2.],degrees=True).as_matrix()
    t=np.array([.1,.15,3.5]);K=np.array([[614.,0.,320.],[0.,614.,240.],[0.,0.,1.]])
    points=C.project(true_native_xyz,R,t,K)
    states=['U']*8
    visible=[]
    for k in [0,3,4,7]:
        states[k]='B'
        visible.append(dict(native_corner=k,state='B',xy=points[k].tolist(),
            covariance_px2=[[1.,0.],[0.,1.]],fit_covariance_px2=[[1.,0.],[0.,1.]],
            fit_reliability_weight=1.,protection_px=2.,source='VISIBLE_BACKBONE_FALLBACK'))
    states[6]='S'
    pose=dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist())
    result=M.solve_two_wd_families(original_xyz,K,states,visible,[],[480,640],pose,
        config=dict(max_p3p_triples=4,global_rotation_starts=2,max_joint_iterations=100,max_seed_nfev=100))
    assert result['accepted'] and result['selected_family']=='SWAPPED_WD'
    assert result['families'][1]['objective']<1e-8
    assert result['families'][0]['objective']>result['families'][1]['objective']+1e-3
    q=np.asarray(result['solution']['projected_native'])
    np.testing.assert_allclose(q[[0,3,4,7]],points[[0,3,4,7]],atol=1e-4,rtol=0)
    assert result['states_preserved'] and result['original_S_gate_preserved']

