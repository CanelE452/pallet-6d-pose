import cv2
import numpy as np
import pytest
from . import physical_shape_filter as S


def cube():
    return S.F.cuboid_keypoints_3d(1.1,1.3,.15)


def projection(rvec=(.2,.4,.1),tvec=(.1,.2,3.)):
    xyz=cube();r=np.array(rvec,float);t=np.array(tvec,float)
    K=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
    return xyz,r,t,S.F._project(xyz,r,t,K)


@pytest.mark.parametrize('rotation',[(.2,.4,.1),(.5,-.7,.2),(-.6,.9,-.2),(0,0,0),(0,np.pi,0)])
def test_valid_perspective_cuboid_keeps_all_faces(rotation):
    xyz,r,t,q=projection(rotation)
    assert S.depths_check(xyz,r,t)['passed']
    assert S.face_checks(q,q)['passed']


def test_positive_translation_not_enough_if_corner_behind_camera():
    xyz=cube()
    assert not S.depths_check(xyz,np.zeros(3),np.array([0.,0.,.1]))['passed']
    assert not S.depths_check(xyz,np.zeros(3),np.array([0.,0.,-3.]))['passed']


def test_bowtie_rejected():
    d=S.polygon_details(np.array([[0,0],[1,1],[0,1],[1,0]],float))
    assert d['self_crossing']


def test_concave_face_rejected():
    d=S.polygon_details(np.array([[0,0],[1,0],[.2,.2],[0,1]],float))
    assert d['concave']


def test_collapsed_corner_detected():
    _,_,_,q=projection();wrong=q.copy();wrong[1]=wrong[0]
    check=S.face_checks(wrong,q)
    assert not check['passed']
    assert any('collapsed_face_or_corner' in f['reasons'] for f in check['faces'])


def test_winding_compared_to_projection_not_fixed_screen_sign():
    _,_,_,q=projection();wrong=q.copy();wrong[:,0]*=-1
    assert S.face_checks(wrong,wrong)['passed']
    check=S.face_checks(wrong,q)
    assert not check['passed']
    assert any('winding_mismatch_with_PnP' in f['reasons'] for f in check['faces'])


def test_physically_edge_on_face_not_automatically_rejected():
    xyz=cube();r=np.zeros(3);t=np.array([.55,0.,3.])
    K=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
    q=S.F._project(xyz,r,t,K);d=S.face_checks(q,q)
    assert d['passed']
    assert any(f['projected']['degenerate'] for f in d['faces'])


def test_translation_scale_invariance_and_no_mutation():
    _,_,_,q=projection();copy=q.copy()
    original=S.face_checks(q,q)['passed']
    assert S.face_checks(q*2+37,q*2+37)['passed']==original
    assert np.array_equal(q,copy)


def test_missing_coordinate_fails():
    _,_,_,q=projection();wrong=q.copy();wrong[0,0]=np.nan
    assert not S.face_checks(wrong,q)['passed']
