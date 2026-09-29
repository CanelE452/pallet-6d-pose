import numpy as np
import pytest
from .train_cross_input import point_residuals,paired


def row():
    return dict(id='fixed',name='train.png',recording='TRAIN',support=[True,False,True,False,False,False,False,False],
        raw=[[0.,0.]]*8,ref=[[1.,0.]]*8,canonical_REF_planned_covered=[0])


def prediction(offset):
    return {'fixed':dict(selected_index=0,candidates=[dict(keypoints_xy=[[offset,0.]]*9),dict(keypoints_xy=[[1.,0.]]*9)])}


def test_selected_candidate_only_not_target_nearest_and_true_ignore():
    raw=point_residuals(prediction(3.),[row()],'raw')
    ref=point_residuals(prediction(3.),[row()],'ref')
    assert [p['corner'] for p in raw]==[0,2]
    assert all(p['l2_px']==3 for p in raw)
    assert all(p['l2_px']==2 for p in ref)
    assert raw[0]['split']=='CANONICAL_REF_PLANNED_COVERED'
    assert raw[1]['split']=='CANONICAL_REF_UNMASKED'


def test_paired_differences_use_same_points_and_failure_penalty():
    before=point_residuals(prediction(3.),[row()],'ref')
    after=point_residuals(prediction(2.),[row()],'ref')
    p=paired(before,after)
    assert p['improved']==2 and p['worsened']==0 and p['paired_L2_delta_mean_px']==-1
    missing=point_residuals({'fixed':dict(selected_index=None,candidates=[])},[row()],'ref')
    p=paired(before,missing)
    assert p['common_observed']==0 and p['point_occurrences']==2
    assert p['full_missing_penalty_delta_mean_px']==pytest.approx(640*np.sqrt(2)-2)
    with pytest.raises(AssertionError):paired(before,list(reversed(after)))
