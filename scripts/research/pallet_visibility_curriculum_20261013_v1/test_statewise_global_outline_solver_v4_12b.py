import numpy as np
from .test_statewise_global_outline_solver_v4_10 import fixture,saved
from . import statewise_global_outline_solver_v4_12b as M


def test_remote_small_and_collapsed_saved_candidates_do_not_abort_valid_joint_fit():
    xyz,K,R,t,q,states,visible,mask=fixture()
    observation=M.observe_global_outline(mask,mask.shape)
    result=M.refine_global_outline_joint_pose(xyz,K,states,visible,[],mask.shape,
        dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist()),
        initial_solutions=[saved(R,t,xyz,K)],outline_observation={**observation,'normal_bias_px':.3},
        initializations=[dict(R=R,t=[0.,0.,1e8],name='TINY_PROJECTED_HULL'),
                         dict(R=R,t=[0.,0.,1e16],name='COLLAPSED_PROJECTED_HULL')],
        config=dict(prior_weight=0.,analytic_jacobian=True,max_joint_iterations=5))
    assert result['accepted'] and np.isfinite(result['objective'])
    assert any(r['initialization']=='TINY_PROJECTED_HULL_UNREFINED' for r in result['candidates'])
    assert any(r['initialization']=='COLLAPSED_PROJECTED_HULL_UNREFINED' for r in result['candidates'])
    assert all(a['visible_support_guards_pass'] for a in result['visible_protection_audit'])
    value,audit=M._outline_value(M.V._pose_vector(result['R'],result['t'],
        np.linalg.norm(np.ptp(xyz,axis=0))),xyz,K,np.linalg.norm(np.ptp(xyz,axis=0)),
        {**observation,'normal_bias_px':.3},M.DEFAULT_OUTLINE_CONFIGURATION)
    assert audit['valid_projection'] and np.isfinite(value)
