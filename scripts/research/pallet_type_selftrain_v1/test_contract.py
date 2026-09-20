import copy
import numpy as np
from .train import label
from .pseudo import passed


def test_pseudo_label_padding_and_no_mutation():
    candidate=dict(box_xyxy=[10,20,110,80],keypoints_xy=[[50,40]]*9,keypoints_conf=[.9]*9)
    before=copy.deepcopy(candidate);row=np.array(label(candidate,[100,200]).split(),float)
    np.testing.assert_allclose(row[1:5],[.4,.5,.25,.2])
    np.testing.assert_allclose(row[5:].reshape(9,3)[0],[.375,140/300,2],atol=1e-9)
    assert candidate==before


def test_uncertain_keypoint_true_ignore():
    candidate=dict(box_xyxy=[10,20,110,80],keypoints_xy=[[50,40]]*9,keypoints_conf=[.9]*8+[.1])
    row=np.array(label(candidate,[100,200]).split(),float)[5:].reshape(9,3)
    assert row[8,2]==1 and (row[:8,2]==2).all()


def test_no_threshold_rescue():
    assert passed(dict(s_remove=.05,s_flip=.01),['s_remove','s_flip'])
    assert not passed(dict(s_remove=.050001,s_flip=.01),['s_remove','s_flip'])
    assert not passed(dict(s_remove=None),['s_remove'])
    assert not passed(dict(s_remove=float('nan')),['s_remove'])


def test_degenerate_box_does_not_make_training_example():
    assert label(dict(box_xyxy=[0,0,0,0]),[100,200]) is None
