import numpy as np
from .iou_review import iou,rejected


def square(x=0.,y=0.,size=10.):
    return np.array([[x,y],[x+size,y],[x+size,y+size],[x,y+size]]*2,float)


def test_exact_iou_and_nested():
    a=square();before=a.copy()
    assert abs(iou(a,a)-1)<1e-7
    assert iou(a,square(20))==0
    assert abs(iou(a,square(5))-1/3)<1e-7
    assert abs(iou(a,square(2,2,2))-.04)<1e-7
    np.testing.assert_array_equal(a,before)


def test_missing_and_degenerate():
    assert iou(None,square())==0
    assert iou(np.zeros((8,2)),square())==0


def test_hull_does_not_detect_index_swap_or_interior_motion():
    a=np.r_[square()[:4],[[2,2],[3,3],[4,4],[5,5]]]
    b=a.copy();b[4]=[8,8]
    assert abs(iou(a,b)-1)<1e-7
    assert abs(iou(a,a[::-1])-1)<1e-7


def test_threshold_boundary():
    assert not rejected(dict(either=.9),'either',.9)
    assert rejected(dict(either=.899),'either',.9)
