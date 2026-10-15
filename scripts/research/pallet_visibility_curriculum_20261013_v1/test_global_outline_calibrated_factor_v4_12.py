import numpy as np
import pytest

from .test_global_outline_analytic_factor_v4_11 import scene
from .global_outline_analytic_factor_v4_11 import outline_value_and_gradient as old
from .global_outline_calibrated_factor_v4_12 import outline_value_and_gradient as calibrated,_signed_loss_and_force


@pytest.mark.parametrize('translation',[[0.,.3,2.2],[-.9,.3,2.2],[.9,.3,2.2]])
def test_zero_bias_recovers_value_and_exact_gradient(translation):
    x,k,s,p,o=scene(translation,[.12,.65,.07]);p=p+[.013,-.017,.011,.009,-.007,.019]
    before=old(p,x,k,s,o);after=calibrated(p,x,k,s,o)
    assert before[0]==pytest.approx(after[0],rel=1e-12,abs=1e-12)
    np.testing.assert_allclose(before[1],after[1],rtol=1e-12,atol=1e-9)


@pytest.mark.parametrize('bias',[.8,-.6])
def test_nonzero_signed_bias_gradient_against_central_difference(bias):
    x,k,s,p,o=scene([-.9,.3,2.2],[.12,.65,.07]);p=p+[.013,-.017,.011,.009,-.007,.019]
    o={**o,'normal_bias_px':bias};v,g,_=calibrated(p,x,k,s,o)
    step=1e-7;eye=np.eye(6)
    finite=np.array([(calibrated(p+eye[j]*step,x,k,s,o)[0]-calibrated(p-eye[j]*step,x,k,s,o)[0])/(2*step) for j in range(6)])
    np.testing.assert_allclose(g,finite,rtol=1e-5,atol=2e-4)


def test_outward_observation_bias_prefers_inset_physical_boundary_and_zero_distance_gradient():
    polygon=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    segments=np.stack([polygon,np.roll(polygon,-1,axis=0)],axis=1)
    points=np.array([[5.,1.],[5.,0.],[5.,-1.]])
    residual=np.array([[0.,1.],[0.,0.],[0.,-1.]])
    loss,force=_signed_loss_and_force(residual,points,segments,np.array([0,0,0]),polygon,1.,1.,1.25,reverse=False)
    assert loss[0]==0 and loss[1]>0 and loss[2]>loss[1]
    assert np.isfinite(force).all() and force[1,1]<0
