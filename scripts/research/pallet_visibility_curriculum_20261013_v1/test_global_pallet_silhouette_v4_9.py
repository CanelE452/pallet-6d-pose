import cv2
import numpy as np
import pytest

from .common import cuboid, project
from .global_pallet_silhouette_v4_9 import observe_mask_outline, score_projected_outline, choose_by_outline


def scene(t=None):
    xyz=cuboid(1.3,.11,1.1);K=np.array([[600.,0,320.],[0,600.,240.],[0,0,1.]])
    angle=.6;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t=np.array([0.,.3,2.]) if t is None else np.asarray(t,float)
    mask=np.zeros((480,640),np.uint8)
    xy=project(xyz,R,t,K)
    cv2.fillConvexPoly(mask,np.rint(cv2.convexHull(xy.astype(np.float32))).astype(np.int32),1)
    return xyz,K,R,t,mask


def test_exact_projected_shape_beats_shift_and_depth_error():
    x,k,r,t,m=scene();o=observe_mask_outline(m,m.shape)
    matched=score_projected_outline(o,x,k,r,t)
    shifted=score_projected_outline(o,x,k,r,t+np.array([.1,0,0]))
    farther=score_projected_outline(o,x,k,r,t+np.array([0,0,.2]))
    assert matched['outline_loss']==0 and matched['hull_IoU']==1
    assert shifted['outline_loss']>.03 and farther['outline_loss']>.03


def test_clipped_canvas_border_is_not_counted_as_observed_pallet_edge():
    x,k,r,t,m=scene([-.8,.3,2.]);o=observe_mask_outline(m,m.shape)
    assert m[:,0].sum()>0
    p=o['interior_contour_xy']
    assert np.all(p[:,0]>=7) and np.all(p[:,1]>=7)
    assert score_projected_outline(o,x,k,r,t)['outline_loss']==0


def test_stable_selection_reads_only_observed_mask_and_pose_geometry():
    x,k,r,t,m=scene();o=observe_mask_outline(m,m.shape)
    candidates=[dict(name='bad',xyz=x,K=k,R=r,t=t+np.array([.1,0,0])),
                dict(name='good',xyz=x,K=k,R=r,t=t),dict(name='tie',xyz=x,K=k,R=r,t=t)]
    before=m.copy();choice=choose_by_outline(o,candidates)
    assert choice['selected_name']=='good' and np.array_equal(m,before)
    assert all(c['outline']['S_local_queries']==0 and c['outline']['U_local_queries']==0 for c in choice['candidates'])


def test_empty_or_degenerate_masks_supply_no_candidate_evidence():
    assert not observe_mask_outline(np.zeros((480,640)),(480,640))['supported']
    m=np.zeros((480,640));m[20,30:32]=1
    assert not observe_mask_outline(m,m.shape)['supported']


def test_invalid_mask_or_rotation_and_nonpositive_depth():
    x,k,r,t,m=scene();o=observe_mask_outline(m,m.shape)
    bad=m.astype(float);bad[0,0]=np.nan
    with pytest.raises(ValueError):observe_mask_outline(bad,bad.shape)
    with pytest.raises(ValueError):score_projected_outline(o,x,k,r*2,t)
    assert not score_projected_outline(o,x,k,r,[0,0,-2])['supported']
