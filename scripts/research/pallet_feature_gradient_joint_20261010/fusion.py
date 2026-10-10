"""Frozen inference-only N3 distribution/image-gradient joint location solver.

Inputs contain no targets, pose references or human visibility. Native integer
(x,y) denotes the center of image pixel [y,x]. Sobel and INTER_LINEAR remap use
BORDER_REFLECT_101. INTER_LINEAR retains OpenCV's interpolation-table precision.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

METHODS = ('FG_JOINT_POSTERIOR', 'JOINT_FIXED_ISOTROPIC')
WINDOW_RADIUS = 5
SPATIAL_SIGMA = 2.5
FLAT_S_MIN = 1e-4
PRIOR_FLOOR = 1.0
IMAGE_SCALE_SQUARED = 16.0
MAX_CONDITION = 1e12
CAP_FRACTION = .01
OFFSETS = np.stack(np.meshgrid(np.arange(-5,6,dtype=np.float64),
                               np.arange(-5,6,dtype=np.float64)), -1).reshape(-1,2)
SPATIAL_WEIGHTS = np.exp(-np.sum(OFFSETS**2,axis=1)/(2*SPATIAL_SIGMA**2))


def prepare_gradients(gray):
    """One original-image Sobel computation, shareable between only the two arms."""
    gray=np.asarray(gray)
    if (gray.ndim!=2 or gray.dtype!=np.float32 or not gray.size or
            not np.isfinite(gray).all() or np.min(gray)<0 or np.max(gray)>1):
        raise ValueError('Expected nonempty finite grayscale float32 in [0,1]')
    return dict(raw_hw=gray.shape,
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REFLECT_101),
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REFLECT_101))


def image_normal_equation(gradients, center):
    """Fixed qN-centered 11x11 native sample coordinates; no iterative recentering."""
    center=np.asarray(center,dtype=np.float64)
    h,w=gradients['raw_hw']
    if center.shape!=(2,) or not np.isfinite(center).all():
        raise ValueError('invalid_window_center')
    if not (0<=center[0]<w and 0<=center[1]<h):
        raise ValueError('outside_initial')
    locations=center[None]+OFFSETS
    mapx=locations[:,0].astype(np.float32).reshape(11,11)
    mapy=locations[:,1].astype(np.float32).reshape(11,11)
    g=np.stack([cv2.remap(gradients[key],mapx,mapy,cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
                for key in ('gx','gy')],axis=1).astype(np.float64)
    if not np.isfinite(g).all():
        raise ValueError('nonfinite_sampled_gradient')
    S=float(np.sum(SPATIAL_WEIGHTS*np.sum(g*g,axis=1),dtype=np.float64))
    if not np.isfinite(S):
        raise ValueError('nonfinite_gradient_energy')
    if S<FLAT_S_MIN:
        return np.zeros((2,2),np.float64),np.zeros(2,np.float64),S
    tensor=SPATIAL_WEIGHTS[:,None,None]*g[:,:,None]*g[:,None,:]
    A=np.sum(tensor,axis=0,dtype=np.float64)/S
    b=np.einsum('iab,ib->a',tensor,locations,dtype=np.float64)/S
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite_image_normal_equation')
    return A,b,S


def posterior_covariance(logits, displacements_native, temperature, cap_px):
    """Uncapped all222-candidate covariance proxy; prior center remains saved qN."""
    logits=np.asarray(logits,dtype=np.float64)
    d=np.asarray(displacements_native,dtype=np.float64)
    if logits.shape!=(222,) or d.shape!=(222,2):
        raise ValueError('invalid_posterior_shape')
    if not np.isfinite(logits).all() or not np.isfinite(d).all():
        raise ValueError('nonfinite_posterior')
    if not np.array_equal(d[-1],np.zeros(2)):
        raise ValueError('invalid_null_displacement')
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid_temperature')
    if not np.isfinite(cap_px) or cap_px**2<PRIOR_FLOOR:
        raise ValueError('cap_below_prior_floor')
    z=logits/temperature;z-=np.max(z)
    p=np.exp(z);p/=np.sum(p,dtype=np.float64)
    m=np.sum(p[:,None]*d,axis=0,dtype=np.float64)
    centered=d-m
    C=centered.T@(p[:,None]*centered)
    C=(C+C.T)*.5
    if not np.isfinite(C).all():
        raise ValueError('nonfinite_covariance')
    ce,cv=np.linalg.eigh(C)
    if np.min(ce)<-1e-10:
        raise ValueError('invalid_covariance_spectrum')
    se=np.clip(ce+PRIOR_FLOOR,PRIOR_FLOOR,cap_px**2)
    Sigma=(cv*se[None,:])@cv.T
    Sigma=(Sigma+Sigma.T)*.5
    positive=p>0
    return Sigma,dict(candidate_count=222,last_null_included=True,
        probability_sum=float(p.sum()),null_probability=float(p[-1]),
        posterior_entropy=float(-np.sum(p[positive]*np.log(p[positive]))),
        candidate_mean_native_px=m,C=C,C_eigenvalues_px2=ce,
        Sigma_posterior=Sigma,Sigma_posterior_eigenvalues_px2=se,
        posterior_mean_is_not_prior_center=True)


def joint_solve(mu,Sigma,A,b):
    """The only position decision equation; no inv() or finished-coordinate mixing."""
    mu=np.asarray(mu,np.float64);Sigma=np.asarray(Sigma,np.float64)
    A=np.asarray(A,np.float64);b=np.asarray(b,np.float64)
    if mu.shape!=(2,) or Sigma.shape!=(2,2) or A.shape!=(2,2) or b.shape!=(2,):
        raise ValueError('invalid_joint_shape')
    if not all(np.isfinite(x).all() for x in (mu,Sigma,A,b)):
        raise ValueError('nonfinite_joint_inputs')
    eig=np.linalg.eigvalsh(Sigma)
    if np.min(eig)<=0 or np.max(eig)/np.min(eig)>MAX_CONDITION:
        raise ValueError('invalid_prior_spd')
    precision=np.linalg.solve(Sigma,np.eye(2,dtype=np.float64))
    system=precision+A/IMAGE_SCALE_SQUARED
    if not np.allclose(system,system.T,atol=1e-12,rtol=0):
        raise ValueError('nonsymmetric_joint_system')
    ev=np.linalg.eigvalsh(system)
    condition=float(np.max(ev)/np.min(ev)) if np.min(ev)>0 else np.inf
    if not np.isfinite(condition) or condition>MAX_CONDITION or np.min(ev)<=0:
        raise ValueError('invalid_joint_spd_or_condition')
    # Flat information must not introduce a floating-point perturbation of mu.
    if not np.any(A):
        if np.any(b):
            raise ValueError('inconsistent_zero_image_equation')
        q=mu.copy()
    else:
        q=np.linalg.solve(system,precision@mu+b/IMAGE_SCALE_SQUARED)
    if not np.isfinite(q).all():
        raise ValueError('nonfinite_joint_output')
    return q,dict(prior_precision=precision,joint_system=system,
                  joint_system_eigenvalues=ev,joint_system_condition=condition)


def _cap(q0,q,cap):
    delta=q-q0;length=float(np.linalg.norm(delta))
    return q0+delta*(cap/length) if length>cap else q.copy()


def joint_refine(gray,q0,qN,logits,candidate_displacements_network,gain,prediction_support,
                 *,temperature=1.0,method='FG_JOINT_POSTERIOR',point_support=None,
                 gradient_cache=None):
    """Return (9x2 final points, diagnostics) without reading labels or metadata.

    Invalid distribution/image/SPD retains qN before the final q0-based cap.
    Unsupported/sentinel/nonfinite input corners and center8 retain q0 exactly.
    The isotropic arm replaces only Sigma; posterior and image diagnostics are
    computed identically. No cornerSubPix call exists in this module.
    """
    if method not in METHODS:
        raise ValueError('Unknown frozen joint method')
    q0=np.asarray(q0,np.float64).copy();qN=np.asarray(qN,np.float64).copy()
    support=np.asarray(prediction_support,bool)
    if q0.shape!=(9,2) or qN.shape!=(9,2) or support.shape!=(9,):
        raise ValueError('Expected q0/qN[9,2], prediction_support[9]')
    model_support=np.ones(8,bool) if point_support is None else np.asarray(point_support,bool)
    if model_support.shape!=(8,):
        raise ValueError('Expected point_support[8]')
    gray=np.asarray(gray);h,w=gray.shape if gray.ndim==2 else (0,0)
    cap=CAP_FRACTION*float(np.hypot(w,h))
    image_error=None
    try:
        gradients=prepare_gradients(gray) if gradient_cache is None else gradient_cache
        if tuple(gradients['raw_hw'])!=(h,w):
            raise ValueError('gradient_cache_shape_mismatch')
    except (ValueError,cv2.error,KeyError) as e:
        gradients=None;image_error='invalid_image_or_gradient_cache'
    logits=np.asarray(logits,np.float64)
    d=np.asarray(candidate_displacements_network,np.float64)
    distribution_error=None
    if logits.shape!=(8,222) or d.shape!=(222,2):
        distribution_error='invalid_posterior_shape'
    elif not np.isfinite(gain) or gain<=0:
        distribution_error='invalid_network_native_gain'
    else:
        d=d/float(gain)
    output=q0.copy();unconstrained=q0.copy();records=[]
    before=np.zeros(8);final=np.zeros(8);active=np.zeros(8,bool)
    for k in range(8):
        record=dict(corner=k,prediction_support=bool(support[k]),point_support=bool(model_support[k]),
                    status=None,fallback_reason=None,cap_active=False)
        if not support[k] or not model_support[k]:
            reason='unsupported_prediction'
        elif not np.isfinite(q0[k]).all() or not np.isfinite(qN[k]).all():
            reason='nonfinite_initial'
        elif (q0[k]==-1).all() or (qN[k]==-1).all():
            reason='missing_initial_sentinel'
        else:
            reason=None
        if reason is not None:
            record.update(status=reason,fallback_reason=reason)
            records.append(record);continue
        q=qN[k].copy()
        try:
            if distribution_error:
                raise ValueError(distribution_error)
            Sigma,posterior=posterior_covariance(logits[k],d,float(temperature),cap)
            record.update(posterior)
            if method=='JOINT_FIXED_ISOTROPIC':
                Sigma=np.eye(2,dtype=np.float64)*IMAGE_SCALE_SQUARED
            record.update(Sigma=Sigma,Sigma_eigenvalues_px2=np.linalg.eigvalsh(Sigma),
                          Sigma_source='fixed16I' if method=='JOINT_FIXED_ISOTROPIC' else '222candidate_covariance')
            if image_error:
                raise ValueError(image_error)
            A,b,S=image_normal_equation(gradients,qN[k])
            record.update(A=A,b=b,S=S,A_eigenvalues=np.linalg.eigvalsh(A),
                          fixed_window_center=qN[k].copy())
            if S<FLAT_S_MIN:
                record.update(status='flat_gradient',fallback_reason='flat_gradient')
            else:
                q,solver=joint_solve(qN[k],Sigma,A,b)
                record.update(solver,status='joint_solved')
        except (ValueError,np.linalg.LinAlgError,cv2.error) as e:
            reason=str(e) if isinstance(e,ValueError) else 'linear_algebra_or_opencv_error'
            record.update(status='fallback',fallback_reason=reason)
            q=qN[k].copy()
        unconstrained[k]=q
        before[k]=np.linalg.norm(q-q0[k]);active[k]=before[k]>cap
        output[k]=_cap(q0[k],q,cap)
        final[k]=np.linalg.norm(output[k]-q0[k])
        record.update(q_unconstrained=q.copy(),q_final=output[k].copy(),
                      total_before_cap_px=float(before[k]),total_final_px=float(final[k]),
                      cap_active=bool(active[k]))
        records.append(record)
    assert np.array_equal(output[8],q0[8],equal_nan=True)
    usable=support[:8]&model_support&np.isfinite(q0[:8]).all(1)&np.isfinite(qN[:8]).all(1)&~(q0[:8]==-1).all(1)&~(qN[:8]==-1).all(1)
    assert np.array_equal(output[:8][~usable],q0[:8][~usable],equal_nan=True)
    assert np.max(final[usable],initial=0)<=cap+1e-10
    return output,dict(method=method,cap_px=cap,q_unconstrained=unconstrained,
        total_before_cap_px=before,total_final_px=final,cap_active=active,
        corner_records=records,status_counts=dict(Counter(r['status'] for r in records)),
        fallback_counts=dict(Counter(r['fallback_reason'] for r in records if r['fallback_reason'])),
        gradient_border='BORDER_REFLECT_101',sample_border='BORDER_REFLECT_101',
        sample_interpolation='cv2.INTER_LINEAR',pixel_centers='integer native x/y',
        GT_inputs=False,cornerSubPix_calls=0,center_preserved=True,
        missing_and_unsupported_preserved=True,
        covariance_note='uncapped native222-candidate covariance proxy; prior mean is frozen qN')
