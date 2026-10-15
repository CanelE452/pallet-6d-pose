"""Uncertain archived typed-BI and legacy-fallback-B tangents, opt in only.

Legacy records keep their original raw-B revocation and RGB-normal/range
semantics. A distinct exact archive authorizes one uncertain old prediction
tangent; it never upgrades them to typed junctions or new full2D measurements.
The normal/tangent noises are not independent or newly calibrated. No GT,
image, model fit or file I/O occurs here; all immutable parent modules remain.
"""
from copy import deepcopy
from collections.abc import Mapping
import numpy as np

from . import statewise_partial_BI_tangent_solver_v4_14 as T
from . import statewise_partial_BI_u_solver_v4_13 as P

ARCHIVE_KEY='legacy_B_tangent_prediction_archive'
_parameters=T._parameters


def archive_legacy_B_tangent_predictions(records,visible,states,config=None):
    """Copy records and archive exact fallbackB prediction/noise, after gates.

One tangent uses explicit line0 in the already validated observed-line order.
It is one archived predictor factor per vertex, not one copy per incident ray.
The caller does not fit/select its direction, covariance or weight from GT.
"""
    if len(states)!=8:raise ValueError('Eight authoritative states required')
    source=P.V3._records(visible);by_id={P.V3._id(m):m for m in source}
    cfg=P._configuration(config);result=[]
    for record in records or []:
        k=P.V3._id(record);state=P.V3._state(states[k])
        if state not in ('B','I') or record.get('state',state)!=state:
            raise ValueError('Archive gate rejects S/U/O/T before coordinate fields')
        if record.get('explicit_complete_BI_likelihood_revocation') is True:
            result.append(deepcopy(record));continue
        if state!='B' or record.get('replaces_raw_B_likelihood') is not True:
            raise ValueError('Legacy prediction archive requires fallbackB revocation')
        m=by_id.get(k)
        if m is None or m.get('state',state)!='B' or m.get('source')!='VISIBLE_BACKBONE_FALLBACK':
            raise ValueError('Legacy archive requires its exact original fallbackB source')
        q=P.V3._finite_array(m['xy'],(2,),'legacy original observed xy')
        observed=P.V3._finite_array(m['covariance_px2'],(2,2),'legacy observed covariance')
        fit=np.asarray(m.get('fit_covariance_px2',cfg['covariance_scale']*observed+
                            np.asarray(cfg['state_BI_total_error_floor_px2']['B'])),float)
        weight=float(m.get('fit_reliability_weight',cfg['state_loss_weights']['B']))
        if not np.isfinite(weight) or weight<=0:raise ValueError('Positive fixed reliability required')
        copied=deepcopy(record)
        copied[ARCHIVE_KEY]=dict(native_corner=k,state='B',original_measurement_source=m['source'],
            original_observed_xy=q.tolist(),original_observed_covariance_px2=observed.tolist(),
            original_effective_fit_covariance_px2=fit.tolist(),original_fit_reliability_weight=weight,
            tangent_prediction_line_index=0,uncertain_original_prediction_not_new_RGB_point=True,
            line_index_rule='first existing validated observed line; one tangent per native vertex',
            legacy_raw_B_revocation_is_not_relabelled_as_typed_BI=True)
        result.append(copied)
    return result


def _boundary_inputs(records,states,image_hw,xyz=None):
    normalized=T._boundary_inputs(records,states,image_hw,xyz)
    for target,source in zip(normalized,records or []):
        if ARCHIVE_KEY not in source:continue
        if target['state']!='B' or not target['legacy_fallback_B_record']:
            raise ValueError('Dedicated legacy archive accepts only non-typed fallbackB')
        target[ARCHIVE_KEY]=deepcopy(source[ARCHIVE_KEY])
    return normalized


def validate_archived_predictions(boundary,visible,config=None):
    source_records=P.V3._records(visible)
    P.validate_complete_BI_likelihood_revocations(boundary,source_records,config)
    cfg=P._configuration(config);source={P.V3._id(m):m for m in source_records}
    for b in boundary:
        if ARCHIVE_KEY not in b:continue
        k=b['native_corner']
        if b['state']!='B' or not b['legacy_fallback_B_record']:
            raise ValueError('Legacy archive state gate precedes coordinate fields')
        m=source[k];a=b[ARCHIVE_KEY]
        if (not isinstance(a,Mapping) or a.get('native_corner')!=k or a.get('state')!='B'
                or m.get('source')!='VISIBLE_BACKBONE_FALLBACK'
                or a.get('original_measurement_source')!=m.get('source')
                or a.get('uncertain_original_prediction_not_new_RGB_point') is not True
                or a.get('legacy_raw_B_revocation_is_not_relabelled_as_typed_BI') is not True):
            raise ValueError('Exact fallbackB source/archive semantics required')
        observed=np.asarray(m['covariance_px2'],float)
        fit=np.asarray(m.get('fit_covariance_px2',cfg['covariance_scale']*observed+
                            np.asarray(cfg['state_BI_total_error_floor_px2']['B'])),float)
        for key,value,shape in (('original_observed_xy',m['xy'],(2,)),
                ('original_observed_covariance_px2',observed,(2,2)),
                ('original_effective_fit_covariance_px2',fit,(2,2))):
            array=P.V3._finite_array(a.get(key),shape,key)
            if not np.array_equal(array,np.asarray(value,float)):
                raise ValueError('Legacy archive differs from immutable '+key)
        weight=float(m.get('fit_reliability_weight',cfg['state_loss_weights']['B']))
        if a.get('original_fit_reliability_weight')!=weight or b['fit_reliability_weight']!=weight:
            raise ValueError('Legacy archive/factor reliability must equal fixed original')
        index=a.get('tangent_prediction_line_index')
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or index!=0:
            raise ValueError('Fixed first-line rule required')


def _tangent_records(boundary,inflation):
    result=T._tangent_records(boundary,inflation)
    for b in boundary:
        if ARCHIVE_KEY not in b:continue
        if b['state']!='B' or not b['legacy_fallback_B_record']:
            raise ValueError('Legacy tangent accepts B only before archive coordinates')
        a=b[ARCHIVE_KEY];index=a['tangent_prediction_line_index']
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or index!=0 or not b['lines']:
            raise ValueError('Existing first line required')
        n=P.V3._finite_array(b['lines'][index]['normal_unit_xy'],(2,),'legacy selected normal')
        tangent=np.array([-n[1],n[0]])
        q=P.V3._finite_array(a['original_observed_xy'],(2,),'legacy archived q')
        covariance=P.V3._finite_array(a['original_effective_fit_covariance_px2'],(2,2),'legacy fixed total covariance')
        if not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0):raise ValueError('Symmetric covariance required')
        np.linalg.cholesky(covariance);variance=float(tangent@covariance@tangent)
        if variance<=0:raise ValueError('Positive tangent variance required')
        result.append(dict(native_corner=b['native_corner'],state='B',tangent=tangent,original_xy=q,
            variance_px2=variance*inflation,original_variance_px2=variance,reliability=b['fit_reliability_weight'],
            line_index=index,original_measurement_source=a['original_measurement_source'],archive_kind='LEGACY_FALLBACK_B'))
    return result


_tangent_residual_and_jacobian=T._clone(T._tangent_residual_and_jacobian,{'_tangent_records':_tangent_records})
_tangent_objective_pair=T._clone(T._tangent_objective_pair,{'_tangent_records':_tangent_records,
    '_tangent_residual_and_jacobian':_tangent_residual_and_jacobian})
objective_with_tangent=T._clone(T.objective_with_tangent,{'_tangent_objective_pair':_tangent_objective_pair})
_augment=T._clone(T.augment_tangent_audit,{'_tangent_records':_tangent_records,
    '_tangent_residual_and_jacobian':_tangent_residual_and_jacobian})


def augment(result,xyz,K,boundary,weight,inflation,schema):
    result=_augment(result,xyz,K,boundary,weight,inflation,schema)
    result['tangent_prediction_factor_audit'].update(
        legacy_fallback_B_archive_ids=[b['native_corner']for b in boundary if ARCHIVE_KEY in b],
        typed_revocation_ids=[b['native_corner']for b in boundary if b['explicit_complete_BI_likelihood_revocation']],
        legacy_archive_does_not_fake_typed_BI_revocation=True,one_tangent_per_archived_native_vertex=True)
    return result


def solve_archived_BI_tangent_u_region_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,
        initializations=None,config=None,*,boundary_observations=None,boundary_factor_weight=1.,u_regions=None,
        prediction_tangent_weight=1.,tangent_variance_inflation=4.):
    weight,inflation=_parameters(prediction_tangent_weight,tangent_variance_inflation)
    args=(xyz,K,states,visible,domains,image_hw,coarse_pose)
    kw=dict(initializations=initializations,config=config,boundary_observations=boundary_observations,
            boundary_factor_weight=boundary_factor_weight,u_regions=u_regions)
    if weight==0 or boundary_factor_weight==0 or not boundary_observations:
        return P.solve_partial_BI_u_region_joint_ot_pnp(*args,**kw)
    boundary=_boundary_inputs(boundary_observations,states,image_hw)
    if not _tangent_records(boundary,inflation):return P.solve_partial_BI_u_region_joint_ot_pnp(*args,**kw)
    def objective(*a):return objective_with_tangent(*a,prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    operation=T._clone(P.solve_partial_BI_u_region_joint_ot_pnp,{'_joint_objective_pair':objective,
        '_boundary_inputs':_boundary_inputs,'validate_complete_BI_likelihood_revocations':validate_archived_predictions})
    return augment(operation(*args,**kw),xyz,K,boundary,weight,inflation,'statewise_archived_BI_tangent_solver_v4_15')
