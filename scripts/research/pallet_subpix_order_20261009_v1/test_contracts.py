"""Synthetic input handoff and preservation checks; no real head or pose calls."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import common as C
from .reverse import reverse_captured


def run():
    gray=np.zeros((90,120),np.uint8)
    q0=np.array([[20+10*k,35] for k in range(8)]+[[60,45]],np.float64)
    captured=dict(p3=object(),p4=object(),selected_index=0,input_shape=(90,120),canvas_shape=(90,120),added_border=0,
        candidates=[dict(candidate_index=0,score=.9,box_xyxy=np.array([10,10,110,80]),keypoints_xy=q0.copy(),keypoints_conf=np.ones(9))])
    seen=[]
    def prediction(head,arm,cap,dims,order,T,rule,hw,norm):
        assert cap['p3'] is captured['p3'] and cap['p4'] is captured['p4']
        seen.append(cap['candidates'][0]['keypoints_xy'].copy())
        candidates=copy.deepcopy(cap['candidates']); points=candidates[0]['keypoints_xy']
        points[:8,0]+=.9
        return dict(candidates=candidates,selected_index=0,head_used=True),{}
    def preservation(before,after,index):
        assert before[0]['score']==after[0]['score']
        assert np.array_equal(before[0]['box_xyxy'],after[0]['box_xyxy'])
        assert np.array_equal(before[0]['keypoints_xy'][8],after[0]['keypoints_xy'][8])
    inf=SimpleNamespace(predict_captured=prediction,preservation=preservation)
    rule=dict(lam=1.,max_move_image_diagonal_fraction=.01)
    def call():return reverse_captured(inf,None,captured,np.array([1.,1.3,.11]),2,1.,rule,gray.shape,{},gray)
    with patch.object(C,'correct',return_value=(q0.copy(),{'algorithm_corner_calls':0})):
        pred,diag,qSN=call()
    assert np.array_equal(seen[-1],q0)
    assert np.array_equal(qSN[:8],q0[:8]+[.9,0])
    shifted=q0.copy();shifted[:8,0]+=1.4
    with patch.object(C,'correct',return_value=(shifted,{'algorithm_corner_calls':8})):
        pred,diag,qSN=call()
    assert np.array_equal(seen[-1],shifted)
    assert np.allclose(np.linalg.norm(diag['qFinal'][:8]-q0[:8],axis=-1),1.5,rtol=0,atol=1e-12)
    assert np.array_equal(captured['candidates'][0]['keypoints_xy'],q0)
    q0[2]=[-1,-1];q0[3]=[np.nan,np.nan];captured['candidates'][0]['keypoints_xy']=q0.copy()
    shifted=q0.copy();shifted[:8,0]+=.2
    with patch.object(C,'correct',return_value=(shifted,{'algorithm_corner_calls':6})):
        _,diag,_=call()
    assert np.array_equal(diag['qFinal'][[2,3,8]],q0[[2,3,8]],equal_nan=True)
    def no_head(head,arm,cap,*args):return dict(candidates=copy.deepcopy(cap['candidates']),selected_index=0,head_used=False),None
    inf.predict_captured=no_head
    with patch.object(C,'correct',return_value=(shifted,{'algorithm_corner_calls':6})):
        _,diag,qSN=call()
    assert np.array_equal(diag['qFinal'],diag['qS'],equal_nan=True)
    return dict(status='PASS',model_calls=0,F_calls=0,identity_SUBPIX_input_handoff=True,
        changed_SUBPIX_coordinates_used_by_N3=True,features_same_objects=True,
        input_capture_unmodified=True,final_cap_anchored_Base=True,center_missing_preserved=True,
        N3_not_applied_preserves_SUBPIX=True,GT_inputs=False)


if __name__=='__main__':print(C.finite(run()))
