import numpy as np
from .identity_rank_hard import mine_arrays
from .identity_rank_features import QUARTER


def test_only_natural_recoverable_c2_mistakes_are_mined():
    gt=np.array([[0,0],[100,0],[100,20],[0,20],[10,50],[90,50],[90,60],[10,60],[50,30]],float)
    q=np.stack([gt[QUARTER],gt,gt[QUARTER]+300,gt[QUARTER]])
    a=dict(points=q,gt_points=np.tile(gt,(4,1,1)),gt_valid=np.ones((4,9),bool),
        boxes=np.tile([0,0,110,70],(4,1)),matched=np.ones(4,bool))
    side=dict(order=np.array([2,2,2,4]),group_valid=np.ones((4,4),bool))
    eligible,positive,normal,_,_=mine_arrays(a,side)
    np.testing.assert_array_equal(eligible,[True,True,True,False])
    np.testing.assert_array_equal(positive,[True,False,False,False])
    np.testing.assert_array_equal(normal,[False,True,False,False])
    np.testing.assert_array_equal(a['points'],q)
