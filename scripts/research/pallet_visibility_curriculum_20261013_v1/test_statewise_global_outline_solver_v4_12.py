import cv2
import numpy as np

from .test_statewise_global_outline_solver_v4_10 import fixture,saved
from . import statewise_global_outline_solver_v4_12 as M


def test_known_mask_expansion_bias_improves_physical_pose_against_zero_mean():
    xyz,K,R,t,q,states,visible,mask=fixture()
    expanded=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    observation=M.observe_global_outline(expanded,expanded.shape)
    outputs=[]
    for bias in (0.,2.):
        out=M.refine_global_outline_joint_pose(xyz,K,states,visible,[],mask.shape,
            dict(R=R.tolist(),t=t.tolist(),physical_registration_Q=np.eye(3).tolist()),
            initial_solutions=[saved(R,t,xyz,K)],outline_observation={**observation,'normal_bias_px':bias},
            config=dict(prior_weight=0.,analytic_jacobian=True,max_joint_iterations=100))
        assert out['accepted'] and all(a['visible_support_guards_pass'] for a in out['visible_protection_audit'])
        outputs.append(out)
    before,after=[np.linalg.norm(np.asarray(out['t'])-t) for out in outputs]
    assert after<before and after<.01
    assert outputs[1]['objective_components']['global_outline_audit']['expected_observed_minus_model_normal_offset_px']==2.
