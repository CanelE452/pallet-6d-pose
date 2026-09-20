import copy
import inspect
import numpy as np
from . import large_corner_selector_features as F


def fixture():
    q=np.array([[20,20],[70,20],[70,40],[20,40],[30,10],[80,10],[80,30],[30,30],[50,25]],float)
    p=dict(keypoints_xy=q.tolist(),keypoints_conf=[.95]*9,box_xyxy=[10,5,90,50],score=.9)
    views={name:copy.deepcopy(p) for name in F.VARIANTS}
    return np.zeros((60,100,3),np.uint8),views


def test_hypotheses_change_channels_not_gt_equivalence():
    im,views=fixture();before=copy.deepcopy(views);h=F.hypotheses(views)
    assert len(h)==12 and h[0]['name']=='R0:native' and h[1]['name']=='R0:reindex90'
    assert F.REINDEX90 not in F.C2
    assert views==before and h[1]['candidate']['keypoints_xy'][8]==views['R0']['keypoints_xy'][8]


def test_feature_contract_and_no_supervision_input():
    im,views=fixture();h=F.hypotheses(views)
    a=F.features(im,views,h[0]);b=F.features(im,views,h[1]);assert a.shape==b.shape==(183,) and not np.array_equal(a,b)
    assert set(inspect.signature(F.features).parameters)=={'image','views','hypothesis','evidence'}


def test_stress_is_deterministic_rgb_only_and_does_not_mutate_input():
    im,views=fixture();old=im.copy();box=views['R0']['box_xyxy']
    a=F.stress_image(im,box,17);b=F.stress_image(im,box,17)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(im,old);assert not np.array_equal(a,im)
