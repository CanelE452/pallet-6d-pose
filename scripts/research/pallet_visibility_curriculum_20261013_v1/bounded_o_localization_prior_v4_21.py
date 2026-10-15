"""Bounded coarse O-localization prior, never a precise O correspondence.

The old predictor-centered ellipse is a dependent localization assumption.
Its outside-region residual is zero everywhere inside the ellipse, with a
single frame-level contamination cap outside. Canvas and other genuine
geometric constraints belong to the caller. No image or label is read here.
"""
from copy import deepcopy
import math

import numpy as np

from . import joint_ot_jacobian_v2 as J


def prepare_o_localization_regions(domains, states, initial_self_hidden):
    """Archive only eligible O ellipses; reject S/non-O before geometry reads."""
    if len(states)!=8 or len(initial_self_hidden)!=8:
        raise ValueError('Eight authoritative states and initial S flags required')
    if any(bool(h)!=(s=='S')for s,h in zip(states,initial_self_hidden)):
        raise ValueError('Initial S mask must remain fixed')
    result=[];seen=set()
    for record in domains:
        k=record.get('native_corner',record.get('native_id'))
        if isinstance(k,(bool,np.bool_))or not isinstance(k,(int,np.integer))or not 0<=k<8:
            raise ValueError('Valid native O ID required')
        # Do not touch coordinates/provenance on forbidden source records.
        if initial_self_hidden[k]or states[k]!='O' or record.get('state')!='O':
            raise ValueError('Only non-S external O localization may enter this prior')
        if k in seen:raise ValueError('Each O assumption may enter only once')
        seen.add(k)
        if record.get('shape')!='ellipse' or not record.get('localization_available'):
            continue
        if record.get('independent_precise_support')is not False:
            raise ValueError('This prior is only for explicitly nonprecise predictor localization')
        center=np.asarray(record['center_xy'],float);axes=np.asarray(record['half_axes_px'],float)
        angle=float(record.get('angle_deg',0.))
        if (center.shape!=(2,)or axes.shape!=(2,)or not np.isfinite(center).all()
                or not np.isfinite(axes).all()or np.any(axes<=0)or not np.isfinite(angle)):
            raise ValueError('Finite positive archived localization ellipse required')
        a=np.deg2rad(angle);basis=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        result.append(dict(native_corner=int(k),center_xy=center,axes_px=axes,basis=basis,
            original_domain=deepcopy(record),source='ARCHIVED_COARSE_O_LOCALIZATION_ASSUMPTION'))
    return result


def bounded_o_localization_value_and_gradient(p,xyz,K,scale,regions,*,epsilon,huber_scale=3.):
    """Frame-level capped outside-ellipse energy and analytic SE3 derivative.

    rho is the existing scalar pseudo-Huber loss on max(ellipse_radius-1,0).
    H=exp(-sum(rho)/2), L=-2 log(epsilon+(1-epsilon)*H). Consequently
    0<=L<=-2log(epsilon), regardless of the number of O records. This is a
    coarse correlated regularizer, not an independent image likelihood or
    a calibrated pose probability. An empty set has exactly zero cost.
    """
    eps,delta=float(epsilon),float(huber_scale)
    if not np.isfinite([eps,delta]).all()or not 0<eps<1 or delta<=0:
        raise ValueError('Finite 0<epsilon<1 and positive existing Huber scale required')
    if not regions:
        return 0.,np.zeros(6),dict(region_count=0,raw_outside_energy=0.,frame_cap=-2*math.log(eps),
            O_point_terms=0,S_raw_or_image_terms=0,coarse_dependent_region_prior=True)
    ids=[r['native_corner']for r in regions]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate O localization assumptions')
    # Only model projections for eligible O are needed, not image O/S points.
    points=np.asarray(xyz,float)[ids]
    q,_,jac,_=J.projection_and_jacobian(p,points,K,scale)
    value=0.;gradient=np.zeros(6);audit=[]
    for i,region in enumerate(regions):
        center=np.asarray(region['center_xy'],float);axes=np.asarray(region['axes_px'],float)
        basis=np.asarray(region['basis'],float)
        local=(basis.T@(q[i]-center))/axes
        radius=float(np.linalg.norm(local));outside=max(0.,radius-1.)
        denominator=math.sqrt(1.+(outside/delta)**2)
        extra=2.*outside*outside/(denominator+1.)
        value+=extra
        if radius>1.:
            dr_dxy=(local/radius/axes)@basis.T
            gradient+=(2.*outside/denominator)*(dr_dxy@jac[i])
        audit.append(dict(native_corner=ids[i],ellipse_radius=radius,outside_residual=outside,
            unbounded_regional_energy=extra,inside_old_ellipse=radius<=1.))
    logH=-.5*value
    logtotal=float(np.logaddexp(math.log(eps),math.log1p(-eps)+logH))
    responsibility=math.exp(math.log1p(-eps)+logH-logtotal)
    bounded=max(0.,-2.*logtotal)
    return bounded,responsibility*gradient,dict(region_count=len(regions),regions=audit,
        raw_outside_energy=value,frame_cap=-2.*math.log(eps),epsilon=eps,
        coarse_component_responsibility=responsibility,O_point_terms=0,
        S_raw_or_image_terms=0,coarse_dependent_region_prior=True,
        localization_window_is_not_physical_occluder_support=True,
        inside_ellipse_penalty_exactly_zero=value==0.,
        independence_or_pose_probability_claim=False)
