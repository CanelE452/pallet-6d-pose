"""CPU-only changed-interface regression checks; no additional main/smoke fits."""
import numpy as np,torch
from env import *
from prior_inference import correct,PriorInference
from posefix_contract_math import self_test,axis_aligned_crop_matrix,transform_points
from prior_model import target_distribution
def run():
    original=(torch.backends.cudnn.enabled,torch.backends.cudnn.allow_tf32)
    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
        assert torch.backends.cudnn.enabled and not torch.backends.cudnn.allow_tf32
    assert (torch.backends.cudnn.enabled,torch.backends.cudnn.allow_tf32)==original
    p=np.arange(18,dtype=float).reshape(9,2);q=p+2;v=np.ones(9,bool);rule=dict(lam=0.,max_move_image_diagonal_fraction=.01)
    assert np.array_equal(correct(p,q,v,rule,(480,720)),p)
    rule=dict(lam=1.,max_move_image_diagonal_fraction=0.);assert np.array_equal(correct(p,q,v,rule,(480,720)),p)
    rule=dict(lam=.5,max_move_image_diagonal_fraction=None);out=correct(p,q,v,rule,(480,720));assert np.array_equal(out[:8],p[:8]+1) and np.array_equal(out[8],p[8])
    rule=dict(lam=1.,max_move_image_diagonal_fraction=.001);out=correct(p,q,v,rule,(480,720));assert np.max(np.linalg.norm(out-p,axis=1))<=np.hypot(480,720)*.001+1e-12
    p[1]=np.nan;v[1]=False;out=correct(p,q,v,rule,(480,720));assert np.isnan(out[1]).all() and np.array_equal(out[8],p[8])
    class Empty:
        def predict(self,bgr):return dict(selected_index=None,candidates=[])
    obj=object.__new__(PriorInference);obj.extractor=Empty();obj.rule=rule;empty=obj.predict(np.zeros((16,16,3),np.uint8));assert empty['candidates']==[] and not empty['head_used']
    class NoPose:
        def predict(self,bgr):return dict(selected_index=0,candidates=[dict(keypoints_xy=None,box_xyxy=[0,0,10,10],score=.9)])
    obj.extractor=NoPose();missing=obj.predict(np.zeros((16,16,3),np.uint8));assert missing['candidates'][0]['keypoints_xy'] is None and not missing['head_used']
    t=torch.tensor([[[287.999,383.999],[288.,383.],[100.,384.],[float('nan'),0.]]]);dist,valid=target_distribution(t,torch.ones(1,4,dtype=torch.bool))
    assert valid.tolist()==[[True,False,False,False]] and torch.allclose(dist.sum(-1),torch.tensor([[1.,0.,0.,0.]]))
    for b in ([0,0,0,1],[0,0,float('nan'),1]):
        rejected=False
        try:axis_aligned_crop_matrix(b)
        except ValueError:rejected=True
        assert rejected
    write(DOC/'REGRESSION_TESTS.json',dict(complete=True,math=self_test(),lambda0_exact=True,cap0_exact=True,lambda_once=True,cap_original_diagonal=True,center8_exact=True,invalid_point_preserved=True,invalid_box_rejected=True,empty_detection_preserved=True,missing_pose_preserved=True,bilinear_boundary_safe=True,cudnn_context_enabled_and_restored=True,no_GPU_or_additional_training=True,inputs=[bound(HERE/'prior_inference.py'),bound(HERE/'test_contracts.py')]))
    print('CHANGED_INTERFACE_REGRESSIONS_PASS')
if __name__=='__main__':run()
