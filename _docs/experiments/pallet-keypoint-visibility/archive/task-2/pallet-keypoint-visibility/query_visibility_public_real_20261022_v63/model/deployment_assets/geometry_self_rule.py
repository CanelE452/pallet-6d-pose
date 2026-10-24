"""Pure conditional cuboid self-visibility; no RGB, fitting, model, or pose solve.

This is the common.py self_hidden rule, guarded by registered_geometry's native
cuboid and valid-pose assumptions. It is a coarse-pose geometric proxy, not a
learned score and not proof of human semantic self visibility.
"""
import numpy as np


def native_cuboid(extents):
    dimensions=np.asarray(extents,dtype=float)
    if dimensions.shape!=(3,) or not np.isfinite(dimensions).all() or (dimensions<=0).any():
        raise ValueError('Finite positive [width,height,depth] required')
    a,b,c=dimensions/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def conditional_self_hidden(extents,R,t,*,native_xyz=None):
    xyz=native_cuboid(extents)
    if native_xyz is not None:
        supplied=np.asarray(native_xyz,dtype=float)
        if supplied.shape!=(8,3) or not np.allclose(supplied,xyz,atol=1e-10,rtol=0):
            raise ValueError('Original registered native cuboid basis/order required')
    R=np.asarray(R,dtype=float);t=np.asarray(t,dtype=float)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Finite3x3 R and3vector t required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or not np.isclose(np.linalg.det(R),1,atol=1e-7):
        raise ValueError('Proper rotation required')
    if np.any((xyz@R.T+t)[:,2]<=0):
        raise ValueError('Original coarse cuboid must have positive camera depth')
    rays=(-R.T@t)[None]-xyz
    cosines=np.sign(xyz)*rays/np.linalg.norm(rays,axis=1)[:,None]
    return cosines.max(1)<-np.sin(np.deg2rad(2))
