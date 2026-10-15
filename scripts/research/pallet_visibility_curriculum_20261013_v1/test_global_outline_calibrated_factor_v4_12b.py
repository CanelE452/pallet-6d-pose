import numpy as np
import pytest
from . import global_outline_calibrated_factor_v4_12b as M
from .test_global_outline_analytic_factor_v4_11 import scene
from .global_outline_analytic_factor_v4_11 import outline_value_and_gradient as old


def test_orientation_avoids_large_coordinate_area_cancellation():
    tiny=np.array([[0.,0.],[1e-7,0.],[1e-7,1e-7],[0.,1e-7]])
    for offset in ([0.,0.],[500.,500.],[1e5,-1e5]):
        polygon=tiny+offset
        assert M._polygon_orientation(polygon)==1.
        assert M._polygon_orientation(polygon[::-1])==-1.
    assert M._polygon_orientation(np.zeros((4,2))) is None


def test_tiny_noncollapsed_polygon_signed_halfplane_has_pixel_units():
    polygon=np.array([[500.,500.],[500.0000001,500.],[500.0000001,500.0000001],[500.,500.0000001]])
    segments=np.stack([polygon,np.roll(polygon,-1,axis=0)],axis=1)
    points=np.array([[500.00000005,499.]])
    loss,force=M._signed_loss_and_force(np.array([[0.,-1.]]),points,segments,np.array([0]),polygon,1.,1.,1.25,reverse=True)
    assert loss[0]==pytest.approx(0.,abs=1e-12)
    np.testing.assert_allclose(force,0.,atol=1e-12)


@pytest.mark.parametrize('bias',[0.,.8,-.6])
def test_bias_gradients_and_zero_mean_analytic_equivalence(bias):
    xyz,K,scale,p,observation=scene([-.9,.3,2.2],[.12,.65,.07])
    p=p+[.013,-.017,.011,.009,-.007,.019]
    observation={**observation,'normal_bias_px':bias}
    value,gradient,_=M.outline_value_and_gradient(p,xyz,K,scale,observation)
    step=1e-7;eye=np.eye(6)
    numeric=np.array([(M.outline_value_and_gradient(p+eye[j]*step,xyz,K,scale,observation)[0]-
                       M.outline_value_and_gradient(p-eye[j]*step,xyz,K,scale,observation)[0])/(2*step) for j in range(6)])
    np.testing.assert_allclose(gradient,numeric,rtol=1e-5,atol=2e-4)
    if bias==0.:
        expected=old(p,xyz,K,scale,observation)
        assert value==pytest.approx(expected[0],rel=1e-12,abs=1e-12)
        np.testing.assert_allclose(gradient,expected[1],rtol=1e-12,atol=1e-9)
