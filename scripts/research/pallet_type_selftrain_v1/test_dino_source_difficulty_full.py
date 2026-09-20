import numpy as np
from .dino_source_difficulty_full import measure


def arrays():
    q=np.full((1,8,2),10.);gt=q.copy();gt[0,:3]=[[100,10],[50,10],[18,10]]
    valid=np.zeros((1,8),bool);valid[0,:3]=True
    return dict(boxes=np.array([[0.,0.,200.,200.]]),points=q,gt_points=gt,
        gain=np.array([2.]),point_valid=np.ones((1,8),bool),gt_valid=valid)


def test_original_pixel_units_and_strict_thresholds():
    r=measure(arrays(),np.array([0]));assert {k:int(v[0]) for k,v in r.items()}==dict(corners=3,good5=1,hard20=1,far40=1)


def test_affine_translation_and_letterbox_scale_invariance():
    a=arrays();b={k:v.copy() for k,v in a.items()}
    for k in ['boxes','points','gt_points']:b[k]=b[k]*3+77
    b['gain']*=3
    for k,v in measure(a,np.array([0])).items():np.testing.assert_array_equal(v,measure(b,np.array([0]))[k])


def test_invalid_nearby_input_cannot_hide_a_far_corner():
    a=arrays();a['points'][0,0]=a['gt_points'][0,0];a['point_valid'][0,0]=False
    assert measure(a,np.array([0]))['far40'][0]==1


def test_outside_wide_crop_targets_are_not_counted():
    a=arrays();a['gt_points'][0,0]=[3000,3000]
    r=measure(a,np.array([0]));assert r['corners'][0]==2 and r['far40'][0]==0
