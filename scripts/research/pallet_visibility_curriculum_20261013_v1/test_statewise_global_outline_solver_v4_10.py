from copy import deepcopy
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as V
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_global_outline_solver_v4_10 as M


def fixture():
    xyz=C.cuboid(1.1,.11,1.3);K=np.array([[500.,3.,160.],[0.,500.,120.],[0.,0.,1.]])
    R=Rotation.from_euler('xyz',[.12,-.25,.03]).as_matrix();t=np.array([.01,.08,3.])
    q,_=V.project_native(xyz,R,t,K);camera=-R.T@t
    hidden=[np.max(n@(camera-x)/np.linalg.norm(camera-x))<0 for x,n in zip(xyz,V.registered_incident_normals(xyz))]
    visible_ids=[i for i,s in enumerate(hidden) if not s][:4]
    states=['S' if hidden[k] else 'U' for k in range(8)]
    visible=[]
    for k in visible_ids:
        states[k]='B';visible.append(dict(native_corner=k,state='B',xy=q[k].tolist(),covariance_px2=np.eye(2).tolist(),
            fit_covariance_px2=(np.eye(2)*16).tolist(),fit_reliability_weight=1.,protection_px=2.))
    mask=np.zeros((240,320),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(q.astype('float32')).round().astype('int32'),1)
    return xyz,K,R,t,q,states,visible,mask


def saved(R,t,xyz,K):
    q,_=V.project_native(xyz,R,t,K)
    return dict(accepted=True,R=R.tolist(),t=t.tolist(),R_physical=R.tolist(),prior_audit={},
        projected_native=q.tolist(),rigid_projection8=q.tolist(),feasible_pose_samples=[],search={'full_original_search':True},
        BI_prior_seed=None,fallback_pose=dict(R=R.tolist(),t=t.tolist(),projected_native=q.tolist(),S_projection_only=True))


def solve(data,*,domains=None,regions=None):
    xyz,K,R,t,q,states,visible,mask=data
    initial=saved(R,t+np.array([.014,-.008,.035]),xyz,K)
    return M.refine_global_outline_joint_pose(xyz,K,states,visible,domains or [],mask.shape,
        dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist()),
        initial_solutions=[initial],outline_observation=M.observe_global_outline(mask,mask.shape),
        u_regions=regions,config=dict(prior_weight=0.,analytic_jacobian=True,max_joint_iterations=100))


def test_continuous_shape_moves_pose_and_preserves_measurements():
    data=fixture();before=deepcopy(data[6]);out=solve(data)
    assert out['accepted']
    assert np.linalg.norm(np.asarray(out['t'])-data[3])<.01
    assert np.linalg.norm(np.asarray(out['t'])-(data[3]+[.014,-.008,.035]))>.02
    assert data[6]==before
    assert out['search']['continuous_SE3_refinement'] and out['search']['new_P3P_or_global_starts']==0
    for m in before:assert out['reported_native'][m['native_corner']]==m['xy']
    assert out['rank']['BI_partial_and_global_outline']['rank']==6


def test_all_individual_OT_and_U_regions_final_feasible():
    data=list(fixture());xyz,K,R,t,q,states,visible,mask=data
    remaining=[k for k,s in enumerate(states) if s=='U'];assert len(remaining)>=2
    first,second=remaining[:2];states[first]='O'
    domains=[dict(native_corner=first,state='O',shape='ellipse',center_xy=q[first].tolist(),half_axes_px=[2.,2.],angle_deg=0.)]
    regions=[dict(native_corner=second,state='U',shape='ellipse',center_xy=q[second].tolist(),half_axes_px=[2.,2.],angle_deg=0.)]
    out=solve(data,domains=domains,regions=regions)
    assert out['accepted']
    assert out['joint_constraint_audit'][0]['feasible'] and out['U_region_constraint_audit'][0]['feasible']
    assert out['reported_native'][second] is None


def test_numeric_shape_derivative_agrees_at_different_step():
    xyz,K,R,t,_,_,_,mask=fixture();scale=np.linalg.norm(np.ptp(xyz,axis=0));p=V._pose_vector(R,t+[.02,.01,.03],scale)
    observation=M.observe_global_outline(mask,mask.shape);_,gradient,_=M.outline_value_and_gradient(p,xyz,K,scale,observation)
    other=[]
    for k in range(6):
        delta=np.zeros(6);delta[k]=2e-5
        other.append((M._outline_value(p+delta,xyz,K,scale,observation,M.DEFAULT_OUTLINE_CONFIGURATION)[0]
                     -M._outline_value(p-delta,xyz,K,scale,observation,M.DEFAULT_OUTLINE_CONFIGURATION)[0])/(4e-5))
    np.testing.assert_allclose(gradient,other,rtol=3e-4,atol=3e-3)


def test_zero_and_unsupported_global_outline_return_exact_saved_solution():
    xyz,K,R,t,_,states,visible,mask=fixture();initial=saved(R,t,xyz,K)
    args=(xyz,K,states,visible,[],mask.shape,dict(R=R,t=t))
    empty=M.observe_global_outline(np.zeros_like(mask),mask.shape)
    assert M.refine_global_outline_joint_pose(*args,initial_solutions=[initial],outline_observation=empty)==initial
    observed=M.observe_global_outline(mask,mask.shape)
    assert M.refine_global_outline_joint_pose(*args,initial_solutions=[initial],outline_observation=observed,
        outline_configuration={'outline_factor_weight':0.})==initial


def test_canvas_edges_not_reconstructed_as_observed_pallet_edges():
    mask=np.zeros((80,100),np.uint8);mask[20:80,30:100]=1
    observation=M.observe_global_outline(mask,mask.shape)
    points=np.asarray(observation['observed_boundary_samples_xy'])
    assert np.max(points[:,0])<=92 and np.max(points[:,1])<=72
    assert not np.any(np.isclose(points[:,0],99)) and not np.any(np.isclose(points[:,1],79))
    assert M._clip(np.array([99.,20.]),np.array([99.,79.]),((7.,92.),(7.,72.))) is None
    # The artificial x=99/y=79 closure cannot be a nearest forward target.
    point=np.array([[90.,60.]])
    wrong=M._nearest_residual(point,np.asarray(observation['polygon_xy']))
    actual=M._nearest_segment_residual(point,np.asarray(observation['observed_boundary_segments_xy']))
    assert np.linalg.norm(wrong)<20 and np.linalg.norm(actual)>=40


def test_registration_mismatch_rejects_instead_of_reusing_other_basis():
    xyz,K,R,t,_,states,visible,mask=fixture();initial=saved(R,t,xyz,K)
    initial['R_physical']=Rotation.from_euler('y',90,degrees=True).as_matrix().tolist()
    with pytest.raises(ValueError,match='another native registration'):
        M.refine_global_outline_joint_pose(xyz,K,states,visible,[],mask.shape,dict(R=R,t=t),
            initial_solutions=[initial],outline_observation=M.observe_global_outline(mask,mask.shape))
