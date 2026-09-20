import numpy as np
from .finalize_renderer_front_audit import center_visibility
from .renderer_front_visibility_audit import visibility
from .camera_facing_contract_audit import ORIGIN


def test_independent_normals_and_rigid_transform():
    q = ORIGIN*np.array([2., 1., .1]); cam = np.array([3., -5., 2.])
    center, face = center_visibility(q, cam)
    old, _ = visibility(q, cam)
    np.testing.assert_allclose(face, old, atol=1e-12)
    r = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    c2,f2 = center_visibility(q@r.T+[4,8,1], cam@r.T+[4,8,1])
    np.testing.assert_allclose(c2, center, atol=1e-12)
    np.testing.assert_allclose(f2, face, atol=1e-12)


def test_face_center_max_and_object_center_max_can_differ():
    q = ORIGIN*np.array([2., 1., .1]); cam = np.array([4., -3.8, 1.])
    center, face = center_visibility(q, cam)
    assert center.argmax() == 3
    assert face.argmax() == 0
