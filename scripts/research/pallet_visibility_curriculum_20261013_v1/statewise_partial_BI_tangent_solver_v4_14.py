"""Opt-in uncertain archived-BI tangent prediction beside partial RGB normals.

One typed selected-line tangent uses its explicitly archived original q and
fixed total fit covariance. It is an uncertain prediction, never a precise
RGB junction, and is not assumed independent from the line/image. Variance
inflation is an engineering assumption (default4); reliability stays fixed.
Legacy B factors, normal variance, all OT/U guards, old full-BI seeds and Q
are untouched. U stays range-only and S projection-only. No GT/RGB/I/O/fit.

The immutable v13 search is reused in a fresh function globals dictionary;
only its final objective callback changes. No parent globals/files mutate.
Weight0, no typed records, and normal-factor-weight0 delegate exactly v13.
"""
from copy import deepcopy
from types import FunctionType

import numpy as np

from . import statewise_partial_BI_u_solver_v4_13 as OLD
from . import joint_ot_pnp as V
from . import joint_ot_jacobian_v2 as J


def _parameters(weight,variance_inflation):
    if (isinstance(weight,(bool,np.bool_)) or not np.isfinite(weight) or weight<0
            or isinstance(variance_inflation,(bool,np.bool_)) or not np.isfinite(variance_inflation)
            or variance_inflation<1):
        raise ValueError('Finite nonnegative tangent weight and variance inflation >=1 required')
    return float(weight),float(variance_inflation)


def _boundary_inputs(records,states,image_hw,xyz=None):
    normalized=OLD._boundary_inputs(records,states,image_hw,xyz)
    for target,source in zip(normalized,records or []):
        if 'tangent_prediction_line_index' in source:
            target['tangent_prediction_line_index']=source['tangent_prediction_line_index']
    return normalized


def _tangent_records(boundary,variance_inflation):
    result=[]
    for b in boundary:
        if b.get('state') not in ('B','I'):
            raise ValueError('Tangent prediction accepts B/I before archived coordinate fields')
        if b.get('explicit_complete_BI_likelihood_revocation') is not True:continue
        lines=b['lines'];index=b.get('tangent_prediction_line_index')
        if index is None:
            if len(lines)!=1:raise ValueError('Multiple typed lines require explicit tangent_prediction_line_index')
            index=0
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(lines):
            raise ValueError('Tangent prediction line index must identify one selected line')
        n=V._finite_array(lines[index]['normal_unit_xy'],(2,),'selected normal')
        t=np.array([-n[1],n[0]]);archive=b['association_provenance']
        q=V._finite_array(archive.get('original_observed_xy'),(2,),'archived original BI xy')
        covariance=V._finite_array(archive.get('original_effective_fit_covariance_px2'),(2,2),'archived total fit covariance')
        if not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0):raise ValueError('Symmetric fixed covariance required')
        np.linalg.cholesky(covariance);variance=float(t@covariance@t)
        reliability=float(b['fit_reliability_weight'])
        if variance<=0 or not np.isfinite(reliability) or reliability<=0:raise ValueError('Positive tangent variance/reliability required')
        result.append(dict(native_corner=b['native_corner'],state=b['state'],tangent=t,original_xy=q,
            variance_px2=variance*variance_inflation,original_variance_px2=variance,
            reliability=reliability,line_index=int(index),original_measurement_source=archive.get('original_measurement_source')))
    return result


def _tangent_residual_and_jacobian(p,xyz,K,boundary,scale,variance_inflation=4.):
    _parameters(1.,variance_inflation);records=_tangent_records(boundary,variance_inflation)
    if not records:return np.empty(0),np.empty((0,6))
    xy,_,jacobian,_=J.projection_and_jacobian(p,xyz,K,scale)
    residual=[];rows=[]
    for r in records:
        k=r['native_corner'];t=r['tangent'];std=np.sqrt(r['variance_px2'])
        residual.append(float(t@(xy[k]-r['original_xy']))/std)
        rows.append((t/std)@jacobian[k])
    return np.asarray(residual),np.asarray(rows).reshape((-1,6))


def _tangent_objective_pair(p,xyz,K,boundary,scale,weight,huber_scale,variance_inflation=4.):
    weight,variance_inflation=_parameters(weight,variance_inflation)
    if weight==0:return 0.,np.zeros(6)
    records=_tangent_records(boundary,variance_inflation)
    residual,jacobian=_tangent_residual_and_jacobian(p,xyz,K,boundary,scale,variance_inflation)
    value=0.;gradient=np.zeros(6)
    for r,e,j in zip(records,residual,jacobian):
        denominator=np.sqrt(1+e*e/huber_scale**2);w=weight*r['reliability']
        value+=w*2*huber_scale**2*(denominator-1)
        gradient+=2*w*e/denominator*j
    return float(value),gradient


def objective_with_tangent(p,xyz,K,measurements,scale,prior,cfg,boundary,normal_weight,
                           *,prediction_tangent_weight=1.,tangent_variance_inflation=4.):
    value,gradient,parts=OLD._joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,normal_weight)
    extra,eg=_tangent_objective_pair(p,xyz,K,boundary,scale,
        prediction_tangent_weight*normal_weight,cfg['huber_scale'],tangent_variance_inflation)
    return value+extra,gradient+eg,{**parts,'uncertain_archived_BI_tangent_robust_loss':extra,
        'prediction_tangent_weight':float(prediction_tangent_weight),
        'tangent_variance_inflation':float(tangent_variance_inflation),
        'tangent_prediction_is_precise_RGB_coordinate':False,'tangent_normal_independence_claimed':False,
        'S_U_tangent_prediction_terms':0}


def _clone(function,overrides):
    copied=FunctionType(function.__code__,{**function.__globals__,**overrides},
        name=function.__name__,argdefs=function.__defaults__,closure=function.__closure__)
    copied.__kwdefaults__=deepcopy(function.__kwdefaults__)
    return copied


def augment_tangent_audit(result,xyz,K,boundary,weight,inflation,schema):
    records=_tangent_records(boundary,inflation);rows=[];tangent_rank=None
    if result.get('accepted'):
        xy=np.asarray(result['projected_native'],float)
        for r in records:
            k=r['native_corner'];residual=float(r['tangent']@(xy[k]-r['original_xy']))
            rows.append(dict(native_corner=k,state=r['state'],line_index=r['line_index'],
                source=r['original_measurement_source'],archived_xy=r['original_xy'].tolist(),
                tangent_unit_xy=r['tangent'].tolist(),original_directional_variance_px2=r['original_variance_px2'],
                assumed_variance_px2=r['variance_px2'],residual_px=residual,
                standardized_residual=residual/np.sqrt(r['variance_px2']),
                fixed_CAL_reliability=r['reliability'],full_RGB_corner_measurement=False))
        shape=V._finite_array(xyz,(8,3),'registered xyz');intrinsics=V._finite_array(K,(3,3),'K')
        scale=max(float(np.linalg.norm(np.ptp(shape,axis=0))),1e-3)
        p=V._pose_vector(np.asarray(result['R']),np.asarray(result['t']),scale)
        _,jacobian=_tangent_residual_and_jacobian(p,shape,intrinsics,boundary,scale,inflation)
        weighted=jacobian*np.sqrt(weight*np.asarray([r['reliability'] for r in records]))[:,None]
        tangent_rank=V._rank(weighted)
    result['schema']=schema
    result['tangent_prediction_factor_audit']=dict(weight=weight,variance_inflation=inflation,
        active_native_ids=[r['native_corner'] for r in records],rows=rows,
        outside_factor_weight_and_reliability_not_double_scaled=True,
        normal_covariance_and_all_OT_U_constraints_unchanged=True,old_complete_BI_cost_not_restored=True,
        coordinate_is_uncertain_archived_prediction_not_precise_RGB_measurement=True,
        correlation_with_normal_not_modelled_and_independence_not_claimed=True,
        tangent_information_rank=tangent_rank,
        parent_combined_rank_excludes_optional_tangent_rows=True,
        parent_ambiguity_is_kept_conservatively_no_uniqueness_proof=True,
        new_total_error_calibration=False,S_terms=0,U_terms=0)
    result.setdefault('provenance',{}).update(uncertain_BI_tangent_prediction_terms=len(records),
        S_tangent_terms=0,U_tangent_terms=0)
    return result


def solve_partial_BI_tangent_u_region_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,
        initializations=None,config=None,*,boundary_observations=None,boundary_factor_weight=1.,u_regions=None,
        prediction_tangent_weight=1.,tangent_variance_inflation=4.):
    weight,inflation=_parameters(prediction_tangent_weight,tangent_variance_inflation)
    args=(xyz,K,states,visible,domains,image_hw,coarse_pose)
    kw=dict(initializations=initializations,config=config,boundary_observations=boundary_observations,
        boundary_factor_weight=boundary_factor_weight,u_regions=u_regions)
    if weight==0 or boundary_factor_weight==0 or not boundary_observations:
        return OLD.solve_partial_BI_u_region_joint_ot_pnp(*args,**kw)
    boundary=_boundary_inputs(boundary_observations,states,image_hw)
    if not _tangent_records(boundary,inflation):return OLD.solve_partial_BI_u_region_joint_ot_pnp(*args,**kw)
    def objective(*a):return objective_with_tangent(*a,prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    solve=_clone(OLD.solve_partial_BI_u_region_joint_ot_pnp,{'_joint_objective_pair':objective,
        '_boundary_inputs':_boundary_inputs})
    result=solve(*args,**kw)
    return augment_tangent_audit(result,xyz,K,boundary,weight,inflation,'statewise_partial_BI_tangent_solver_v4_14')


solve_boundary_u_region_joint_ot_pnp=solve_partial_BI_tangent_u_region_joint_ot_pnp
