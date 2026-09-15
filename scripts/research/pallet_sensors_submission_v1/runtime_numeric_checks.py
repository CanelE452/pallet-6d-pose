"""Numerical prior-coordinate checks; preserved outputs remain bit-exact."""
import numpy as np
from env import *
from posefix_contract_math import axis_aligned_crop_matrix,transform_points
CROP_ATOL=3e-4  # Existing validate_prior.py absolute component; no relative allowance.
def coordinates(c,r,label):
    er=old('evaluate_real');er.exact(c['keypoints_xy'][8],r['keypoints_xy'][8],label+'/center8')
    p=np.asarray(c['keypoints_xy']);q=np.asarray(r['keypoints_xy'])
    assert p.shape==q.shape==(9,2) and np.isfinite(p).all() and np.isfinite(q).all()
    matrix=axis_aligned_crop_matrix(np.asarray(r['box_xyxy']))
    pc=transform_points(p,matrix);qc=transform_points(q,matrix)
    np.testing.assert_allclose(pc,qc,atol=CROP_ATOL,rtol=0,err_msg=label)
    return dict(original_px_max=float(np.max(np.abs(p-q))),crop_px_max=float(np.max(np.abs(pc-qc))))
def choice(candidates,camera,dimensions):
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    p=np.asarray(max(candidates,key=lambda c:c['score'])['keypoints_xy'],float)
    r=select_pnp_hypotheses(p,camera,dimensions,None)
    h=next((h for h in r.hypotheses if h.name==r.selected_hypothesis and h.success),None)
    return None if h is None else dict(name=h.name,dimensions=h.camera_facing_dimensions.as_dict())
def self_test():
    r=dict(keypoints_xy=np.ones((9,2))*50,box_xyxy=[0,0,100,100]);c=dict(r,keypoints_xy=r['keypoints_xy'].copy())
    assert coordinates(c,r,'same')['crop_px_max']==0
    c['keypoints_xy'][0,0]+=.01
    try:coordinates(c,r,'deliberate_error')
    except AssertionError:return True
    raise AssertionError('Numerical gate accepted a material corner perturbation')
