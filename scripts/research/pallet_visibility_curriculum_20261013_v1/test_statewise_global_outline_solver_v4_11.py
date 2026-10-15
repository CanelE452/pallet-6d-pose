"""Exercise full pose fitting and final range guards after gradient replacement."""
import numpy as np
import pytest

from .test_statewise_global_outline_solver_v4_10 import fixture,saved
from . import statewise_global_outline_solver_v4_11 as M


def test_analytic_pose_refinement_preserves_joint_OT_U_and_point_readout():
    xyz,K,R,t,q,states,visible,mask=fixture()
    remaining=[k for k,s in enumerate(states) if s=='U']
    first,second=remaining[:2];states[first]='O'
    domains=[dict(native_corner=first,state='O',shape='ellipse',center_xy=q[first].tolist(),half_axes_px=[2.,2.],angle_deg=0.)]
    regions=[dict(native_corner=second,state='U',shape='ellipse',center_xy=q[second].tolist(),half_axes_px=[2.,2.],angle_deg=0.)]
    before=[m.copy() for m in visible]
    initial=saved(R,t+np.array([.014,-.008,.035]),xyz,K)
    out=M.refine_global_outline_joint_pose(xyz,K,states,visible,domains,mask.shape,
        dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist()),
        initial_solutions=[initial],outline_observation=M.observe_global_outline(mask,mask.shape),
        u_regions=regions,config=dict(prior_weight=0.,analytic_jacobian=True,max_joint_iterations=100))
    assert out['accepted'] and np.linalg.norm(np.asarray(out['t'])-t)<.01
    assert out['joint_constraint_audit'][0]['feasible'] and out['U_region_constraint_audit'][0]['feasible']
    assert out['reported_native'][second] is None and visible==before
    assert out['rank']['BI_partial_and_global_outline']['rank']==6
    assert out['global_outline_factor_audit']['outline_gradient'].startswith('analytic')


def test_new_gradient_solver_rejects_other_native_registration():
    xyz,K,R,t,q,states,visible,mask=fixture();initial=saved(R,t,xyz,K)
    initial['R_physical']=np.eye(3).tolist()
    with pytest.raises(ValueError,match='another native registration'):
        M.refine_global_outline_joint_pose(xyz,K,states,visible,[],mask.shape,dict(R=R,t=t),
            initial_solutions=[initial],outline_observation=M.observe_global_outline(mask,mask.shape))
