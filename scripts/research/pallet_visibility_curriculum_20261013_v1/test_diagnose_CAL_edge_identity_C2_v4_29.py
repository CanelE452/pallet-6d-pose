import numpy as np
from . import diagnose_CAL_edge_identity_C2_v4_29 as P


def test_C2_native_transport_preserves_center_and_unordered_registered_cuboid():
    xyz=P.C.cuboid(1.1,.17,.8)+[.2,-.1,.3]
    Q=np.array([[0.,0,1],[0,1,0],[-1,0,0]]);R=Q;t=np.array([.2,.1,4.])
    # Native translation origin may be nonzero; rotation symmetry is about
    # the model center rather than the native origin.
    centered=xyz-xyz.mean(0)
    r2,t2=P.alternate_reference(xyz,Q,R,t)
    assert np.allclose(R@xyz.mean(0)+t,r2@xyz.mean(0)+t2)
    first=xyz@R.T+t;second=xyz@r2.T+t2
    assert np.linalg.norm(first[:,None,:]-second[None,:,:],axis=2).min(1).max()<1e-12


def test_line_normal_is_distance_to_infinite_named_line_and_sign_independent_angle():
    q=np.zeros((8,2));q[0]=[1.,2.];q[1]=[11.,2.]
    c=dict(native_corner=0,native_neighbor=1,normal_unit_xy=[0.,1.],support_anchor_xy=[30.,7.])
    value=P.line_errors(q,c)
    assert value==dict(normal_px=5.,angle_deg=0.)
    c['native_corner'],c['native_neighbor']=1,0
    reversed_value=P.line_errors(q,c)
    assert reversed_value==dict(normal_px=-5.,angle_deg=0.)
