"""Pure saved-pose tangent budget; no normal duplication or new pose fit.

The remaining tangent of the archived old B/I measurement is uncertain data,
not a newly measured exact corner. Its source can be an old RGB correction or
a backbone fallback; neither is relabelled. Lambda scales inverse covariance
inside the same radial pseudo-Huber, so .25 means variance x4. No S/U/O/T
coordinate, image, GT or selector parameter is consumed here.
"""
from collections.abc import Mapping
import numpy as np


def typed_tangent_supplement(projected_native, states, boundary_observations, *,
                             inverse_variance_weight=1., huber_scale=3.):
    strength=float(inverse_variance_weight);delta=float(huber_scale)
    if not np.isfinite(strength) or strength<0 or not np.isfinite(delta) or delta<=0:
        raise ValueError('Finite nonnegative inverse variance and positive Huber scale required')
    if len(states)!=8:
        raise ValueError('Eight authoritative states required')
    total=0.;audits=[];seen=set()
    for record in boundary_observations:
        if not isinstance(record,Mapping):raise ValueError('Typed record mapping required')
        k=record.get('native_corner')
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<8:
            raise ValueError('Native corner ID required')
        # State and explicit typed gate precede line/archive/projected fields.
        if states[k] not in ('B','I') or record.get('state')!=states[k]:
            raise ValueError('Tangent supplement accepts authoritative B/I only')
        if record.get('explicit_complete_BI_likelihood_revocation') is not True:
            raise ValueError('Only explicitly revoked typed B/I gets this supplement')
        if k in seen:raise ValueError('One tangent budget per typed native point')
        seen.add(k)
        if strength==0:
            audits.append(dict(native_corner=int(k),state=states[k],cost=0.,inactive_exact_zero=True))
            continue
        source=record['association_provenance'];lines=record['lines']
        if len(lines)!=1:
            raise ValueError('Current one-normal primitive required; no ambiguous tangent for multiple lines')
        n=np.asarray(lines[0]['normal_unit_xy'],float)
        if n.shape!=(2,) or not np.isfinite(n).all() or not np.isclose(n@n,1.,atol=1e-8,rtol=0):
            raise ValueError('Unit actual line normal required')
        tangent=np.array([-n[1],n[0]])
        q=np.asarray(projected_native[k],float)
        old=np.asarray(source['original_observed_xy'],float)
        covariance=np.asarray(source['original_effective_fit_covariance_px2'],float)
        reliability=float(source['original_fit_reliability_weight'])
        if (q.shape!=(2,) or old.shape!=(2,) or covariance.shape!=(2,2)
                or not all(np.isfinite(a).all() for a in (q,old,covariance))
                or not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0)
                or not np.isfinite(reliability) or reliability<=0):
            raise ValueError('Archived immutable B/I point/covariance/reliability required')
        np.linalg.cholesky(covariance)
        variance=float(tangent@covariance@tangent);residual=float(tangent@(q-old))
        standardized=np.sqrt(strength)*residual/np.sqrt(variance)
        loss=float(reliability*2*delta**2*(np.sqrt(1+(standardized/delta)**2)-1))
        total+=loss
        audits.append(dict(native_corner=int(k),state=states[k],tangent_unit_xy=tangent.tolist(),
            old_measurement_source=source['original_measurement_source'],
            old_archived_BI_xy=old.tolist(),model_projection_xy=q.tolist(),tangent_residual_px=residual,
            original_marginal_tangent_variance_px2=variance,effective_tangent_variance_px2=variance/strength,
            inverse_variance_weight=strength,standardized_tangent_residual=standardized,
            fixed_fit_reliability_weight=reliability,huber_scale=delta,cost=loss,
            normal_term_added=False,point_is_new_precise_RGB_measurement=False))
    return dict(cost=float(total),records=audits,inverse_variance_weight=strength,
        covariance_scaling_inside_pseudo_Huber=True,normal_duplication=False,
        U_S_or_OT_observation_terms=0,new_confidence_calibration=False)
