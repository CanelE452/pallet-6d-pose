import numpy as np
import pytest
from .augmentation_stability import VARIANTS, transform, map_points, instability, rank_keep


@pytest.mark.parametrize('variant',VARIANTS)
def test_inverse_map_and_preserve(variant):
    im=np.full((401,599,3),123,np.uint8); before=im.copy()
    aug,m=transform(im,variant)
    points=np.array([[0.,0.],[43.2,34.3],[598.,400.]])
    np.testing.assert_allclose(map_points(map_points(points,m),np.linalg.inv(m)),points,atol=1e-10)
    np.testing.assert_array_equal(im,before)
    assert aug.dtype==np.uint8


def test_pixel_center_resize():
    im=np.zeros((100,200,3),np.uint8)
    _,m=transform(im,'resize090')
    np.testing.assert_allclose(map_points([[0,0],[199,99]],m),[[-.05,-.05],[179.05,89.05]])


def test_one_bad_corner_is_not_averaged_away():
    p=np.zeros((8,2)); q=np.zeros((6,8,2));q[:,3,0]=20
    s=instability(p,q,100)
    assert s['score']==.2 and s['max_corner_rms_px']==20
    assert instability(p,np.zeros_like(q),100)['score']==0


def test_ranking_missing_and_ties():
    rows=[dict(id='b',score=1),dict(id='a',score=1),dict(id='c',score=None),dict(id='d',score=0)]
    assert rank_keep(rows,.5)=={'a','d'}
    assert rank_keep(rows,.8)=={'a','b','d'}


def test_crop_origin():
    im=np.zeros((100,200,3),np.uint8)
    aug,m=transform(im,'crop_left_top')
    assert aug.shape==(98,196,3)
    np.testing.assert_array_equal(map_points([[4,2]],m),[[0,0]])
