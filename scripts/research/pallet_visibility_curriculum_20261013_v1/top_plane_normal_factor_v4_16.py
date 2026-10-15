"""Pure two-DOF oriented top-plane normal prior, with analytic derivatives.

Only R_cf*[0,-1,0] is regularized.  Native yaw, position and depth have no
direct cost.  The S2 log is nonsmooth at the exact antipode; this is reported
explicitly, never replaced by an unoriented/zero-residual normal.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .joint_ot_jacobian_v2 import skew,so3_right_jacobian

TOP_NORMAL_CF=np.array([0.,-1.,0.])


class NormalCutLocusError(ValueError):
    """An exact antipodal oriented normal has no unique tangent direction."""


def _rotation(R):
    R=np.asarray(R,dtype=float)
    if (R.shape!=(3,3) or not np.isfinite(R).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0)
        or not np.isclose(np.linalg.det(R),1.,atol=1e-7,rtol=0)):
        raise ValueError('Proper finite camera-facing rotation required')
    return R


def tangent_basis(normal):
    n=np.asarray(normal,float)
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n),1.,atol=1e-7):
        raise ValueError('Unit camera normal required')
    axis=np.eye(3)[int(np.argmin(np.abs(n)))]
    first=axis-n*float(axis@n);first/=np.linalg.norm(first)
    return np.column_stack([first,np.cross(n,first)])


def normal_log_and_jacobian(candidate_R_cf,coarse_R_cf,*,candidate_rotvec=None):
    """S2 log[2] derivative in the supplied (else principal) rotvec chart."""
    R=_rotation(candidate_R_cf);R0=_rotation(coarse_R_cf)
    n=R@TOP_NORMAL_CF;n0=R0@TOP_NORMAL_CF;B=tangent_basis(n0)
    phi=Rotation.from_matrix(R).as_rotvec() if candidate_rotvec is None else np.asarray(candidate_rotvec,float)
    if (phi.shape!=(3,) or not np.isfinite(phi).all()
        or not np.allclose(Rotation.from_rotvec(phi.copy()).as_matrix(),R,atol=1e-9,rtol=0)):
        raise ValueError('Candidate rotation and supplied rotvec chart differ')
    dn=-R@skew(TOP_NORMAL_CF)@so3_right_jacobian(phi)
    c=float(np.clip(n0@n,-1.,1.));v=n-c*n0;s=float(np.linalg.norm(v))
    theta=float(np.arctan2(s,c))
    if c<0 and s<=1e-12:
        raise NormalCutLocusError('Exact oriented-normal antipode: no unique S2 logarithm')
    projection=np.eye(3)-np.outer(n0,n0);dc=n0@dn
    if 1-c<1e-6:
        epsilon=1-c
        coefficient=1+epsilon/3+2*epsilon**2/15+2*epsilon**3/35
        derivative=-1/3-4*epsilon/15-6*epsilon**2/35
    else:
        coefficient=theta/s
        derivative=(theta*c-s)/s**3
    residual=B.T@(coefficient*v)
    jacobian=B.T@(coefficient*projection@dn+np.outer(v,derivative*dc))
    return residual,jacobian,dict(angle_rad=theta,angle_deg=float(np.degrees(theta)),
        candidate_top_normal_camera=n.tolist(),coarse_top_normal_camera=n0.tolist(),
        residual_dimension=2,oriented_top_normal=True,full_rotation_or_position_prior=False)


def normal_residual_and_jacobian(p,coarse_R_cf,sigma_angle_rad):
    p=np.asarray(p,float)
    if p.shape!=(6,) or not np.isfinite(p).all():raise ValueError('Finite rotvec plus scaled translation pose[6] required')
    sigma=float(sigma_angle_rad)
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive angular scale required')
    residual,jacobian,audit=normal_log_and_jacobian(Rotation.from_rotvec(p[:3].copy()).as_matrix(),coarse_R_cf,candidate_rotvec=p[:3])
    full=np.column_stack([jacobian,np.zeros((2,3))])/sigma
    return residual/sigma,full,audit


def normal_value_and_gradient(p,coarse_R_cf,sigma_angle_rad,weight=1.,huber_scale=3.):
    """Radial pseudo-Huber of two normal tangent coordinates, as in BI cost."""
    weight=float(weight);delta=float(huber_scale)
    if not np.isfinite(weight) or weight<0 or not np.isfinite(delta) or delta<=0:
        raise ValueError('Nonnegative weight and positive Huber scale required')
    if weight==0:return 0.,np.zeros(6),dict(enabled=False,normal_prior_weight=0.)
    residual,jacobian,audit=normal_residual_and_jacobian(p,coarse_R_cf,sigma_angle_rad)
    squared=float(residual@residual);denominator=np.sqrt(1+squared/delta**2)
    value=weight*2*delta**2*(denominator-1)
    gradient=weight*2*residual@jacobian/denominator
    return float(value),gradient,{**audit, 'enabled':True,'normal_prior_weight':weight,
        'normal_angular_sigma_rad':float(sigma_angle_rad),'normal_whitened_residual':residual.tolist(),
        'loss_kind':'two-coordinate radial pseudo-Huber','direct_translation_or_depth_penalty':False,
        'native_yaw_direct_penalty':False,'S_U_or_image_queries':0,
        'backbone_correlated_regularizer_not_independent_image_measurement':True,
        'does_not_establish_pose_uniqueness':True}
