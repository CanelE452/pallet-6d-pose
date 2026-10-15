"""CAL-only zero-mean scale for exact archived complete-BI tangent errors.

Preparation seals prediction metadata, prior CAL diagnostic and label bytes.
Fitting reuses that diagnostic's fixed-native errors, without reopening numeric
CAL labels. All30 main one-direction records remain; no point/pose/image fit.
The runtime helper changes only a tangent variance, never q, state or guide.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import minimize

from . import common as C
from . import calibrate_complete_bi_edge_mixture_v4_20d as CAL
from . import audit_cal_complete_tangent_v4_23 as DIAG

OUTPUT=CAL.OUTPUT.parent/'CAL_archived_complete_tangent_v4_24'
SCHEMA='CAL_ARCHIVED_COMPLETE_BI_TANGENT_SCALE_V4_24'
OBSERVATION_FAMILY='EXACT_ARCHIVED_COMPLETE_BI_TANGENT_MAIN_V20D_ORIGINAL_GUIDE'
PIXEL_FLOOR_VARIANCE_PX2=float(CAL.AUTH.OBS.PIXEL_LOCALIZATION_FLOOR_PX)**2
RECIPE=dict(fit_partition='CAL',inventory='all original CAL46; fixed-native compatible44 only',
    population='ALL existing20d MAIN one-direction revoked complete B/I; no source/error/role trimming',
    observation_family=OBSERVATION_FAMILY,reference='sealed CAL23 fixed-native model reprojection; same ALIGN44 phase',
    q='exact archived complete measurement, actual RGB or backbone source; not renamed raw',
    tangent='immutable ORIGINAL observer model guide; same across both WD families',
    covariance='exact original effective fit covariance and reliability archives unchanged',
    variance='existing_pixel_floor1px^2 + gamma * (t_ORIGINAL @ effective_fit_cov @ t_ORIGINAL)',
    pixel_floor_variance_px2=PIXEL_FLOOR_VARIANCE_PX2,gamma_nonnegative=True,
    bias_or_mean_correction=0.,estimator='uncentered scalar pseudo-Huber delta3 plus logvariance NLL',huber_scale=3.,
    weighting='equal supported sessions; equal supported frames within session; equal tangent records within frame',
    optimizer='L-BFGS-B scalar gamma >=0',optimizer_gamma_starts=[0.,1.,4.],optimizer_maximum_iterations=1000,
    optimizer_ftol=1e-12,optimizer_gtol=1e-8,global_minimum_guaranteed=False,
    matched_old_control='EXACT saved variance4*a; no added1px floor; differs from recipe gamma4',
    independent_information_or_future_coverage_claim=False,reference_independent_metrology=False,
    sample_count_precision_boost=False,old_reliability_used_as_extra_CAL_weight=False,
    CAL_labels_previously_exposed=True,runtime_frame_session_state_or_source_routing=False,
    TEST_or_remaining_labels_used=False,S_U_or_other_raw_coordinate_queries=0,image_NN_or_pose_fits=0)


def _digest(value):return CAL.OLD._digest(value)


def _number(value,name,*,positive=False):return CAL.OLD.OLD_CAL._number(value,name,positive=positive)


def _variance_axis(tangent_unit_xy,original_effective_fit_covariance_px2):
    tangent=np.asarray(tangent_unit_xy,float);cov=np.asarray(original_effective_fit_covariance_px2,float)
    if (tangent.shape!=(2,)or cov.shape!=(2,2)or not np.isfinite(tangent).all()or not np.isfinite(cov).all()
            or not np.isclose(tangent@tangent,1.,atol=1e-10,rtol=0)
            or not np.allclose(cov,cov.T,atol=1e-10,rtol=0)or np.linalg.eigvalsh(cov).min()<=0):
        raise ValueError('Immutable original-guide unit tangent and positive symmetric effective covariance required')
    return float(tangent@cov@tangent)


def _prepared_tangent(plan,hidden):
    k=plan['native_corner']
    if not isinstance(k,int)or isinstance(k,bool)or not 0<=k<8:raise ValueError('Explicit native corner required')
    if hidden[k]or plan['state']not in ('B','I'):return None
    if not plan['point_cost_revoked']or plan.get('tangent_kind')!='ARCHIVED_COMPLETE_BI_TANGENT':return None
    if len(plan['active_physical_edges'])!=1:raise ValueError('Exact main one-direction population required')
    archive=plan['original_measurement_archive']
    if plan['tangent_axis_source']!='IMMUTABLE_ORIGINAL_OBSERVER_MODEL_GUIDE_SAME_ACROSS_WD':
        raise ValueError('Original observer guide required')
    a=_variance_axis(plan['tangent_unit_xy'],archive['original_effective_fit_covariance_px2'])
    old=_number(plan['tangent_variance_px2'],'saved old tangent variance',positive=True)
    if not np.isclose(old,4.*a,atol=1e-10,rtol=1e-10):raise ValueError('Exact saved variance x4 required')
    return dict(native_corner=k,state=plan['state'],source=archive['original_measurement_source'],
        source_group=DIAG._source_group(archive['original_measurement_source']),
        archived_q_xy=deepcopy(archive['original_observed_xy']),
        observed_covariance_px2=deepcopy(archive['original_observed_covariance_px2']),
        effective_fit_covariance_px2=deepcopy(archive['original_effective_fit_covariance_px2']),
        fit_reliability_weight=archive['original_fit_reliability_weight'],
        original_guide_tangent_unit_xy=deepcopy(plan['tangent_unit_xy']),original_variance_axis_px2=a,
        exact_saved_old_tangent_variance_px2=old,active_physical_edge=plan['active_physical_edges'][0])


def scalar_tangent_nll_and_gradient(gamma,errors,variance_axes,weights,*,pixel_floor_variance_px2=PIXEL_FLOOR_VARIANCE_PX2,huber_scale=3.):
    """-2 log robust scalar density, including variance volume; derivative d/dγ."""
    gamma=_number(gamma,'gamma');floor=_number(pixel_floor_variance_px2,'fixed pixel floor',positive=True)
    delta=_number(huber_scale,'Huber delta',positive=True)
    errors=np.asarray(errors,float);a=np.asarray(variance_axes,float);weights=np.asarray(weights,float)
    if (gamma<0 or errors.ndim!=1 or not len(errors)or a.shape!=errors.shape or weights.shape!=errors.shape
            or not np.isfinite(errors).all()or not np.isfinite(a).all()or not np.isfinite(weights).all()
            or np.any(a<=0)or np.any(weights<=0)):raise ValueError('Finite errors, positive original variance axes and weights required')
    variance=floor+gamma*a
    denominator=np.sqrt(1.+errors*errors/(delta*delta*variance))
    robust=2.*delta*delta*(denominator-1.)
    value=float(weights@(np.log(variance)+robust))
    gradient=float(weights@(a*(1./variance-errors*errors/(variance*variance*denominator))))
    return value,gradient


def _training(records,*,fit_ids=None,inventory_ids=None,heldout_session=None):
    records=list(records)
    if any(r.get('partition')!='CAL'for r in records):raise ValueError('CAL only; TEST and remainder forbidden')
    ids=[r['id']for r in records];inventory=list(ids if inventory_ids is None else inventory_ids)
    if len(set(ids))!=len(ids)or len(set(inventory))!=len(inventory)or set(ids)!=set(inventory):raise ValueError('Unique complete CAL inventory required')
    allowed=set(ids if fit_ids is None else fit_ids)
    if not allowed<=set(ids):raise ValueError('Training IDs must belong to CAL inventory')
    eligible=[];train=[]
    for row in records:
        if row['id']not in allowed or row.get('compatible')is not True or row['session']==heldout_session:continue
        eligible.append(row['id']);points=[]
        for point in row['tangents']:
            if point['state']not in ('B','I'):raise ValueError('Only original nonS B/I main tangents')
            e=_number(point['signed_tangent_error_px'],'fixed-native signed tangent error')
            a=_number(point['original_variance_axis_px2'],'original variance axis',positive=True)
            if not np.isclose(point['exact_saved_old_tangent_variance_px2'],4.*a,atol=1e-10,rtol=1e-10):raise ValueError('Original x4 control changed')
            points.append({**point,'signed_tangent_error_px':e,'original_variance_axis_px2':a})
        if points:train.append(dict(id=row['id'],session=row['session'],tangents=points))
    if not train:raise ValueError('At least one applicable compatible CAL training frame required')
    sessions=sorted({r['session']for r in train});counts=Counter(r['session']for r in train)
    points=[];weights=[]
    for row in train:
        w=1./len(sessions)/counts[row['session']]/len(row['tangents'])
        points.extend({**p,'id':row['id'],'session':row['session']}for p in row['tangents']);weights.extend([w]*len(row['tangents']))
    return records,inventory,eligible,train,sessions,points,np.asarray(weights)


def _descriptive(errors,axes,weights,variance):
    errors=np.asarray(errors);variance=np.asarray(variance);weights=np.array(weights,float,copy=True);weights/=weights.sum()
    denominator=np.sqrt(1.+errors*errors/(RECIPE['huber_scale']**2*variance))
    value=float(weights@(np.log(variance)+2.*RECIPE['huber_scale']**2*(denominator-1.)))
    z=np.abs(errors)/np.sqrt(variance)
    return dict(uncentered_NLL=value,empirical_within_sigma={str(k):float(weights@(z<=k))for k in (1,2,3)},
        normalized_absolute_error_max=float(z.max()),weighted_mean_absolute_error_px=float(weights@np.abs(errors)),
        weighted_mean_sigma_px=float(weights@np.sqrt(variance)),normal_or_future_coverage_certificate=False)


def fit_archived_complete_tangent_model(records,*,fit_ids=None,inventory_ids=None,heldout_session=None,input_bindings=None):
    records,inventory,eligible,train,sessions,points,weights=_training(records,fit_ids=fit_ids,inventory_ids=inventory_ids,heldout_session=heldout_session)
    errors=np.array([p['signed_tangent_error_px']for p in points]);a=np.array([p['original_variance_axis_px2']for p in points])
    def objective(x):
        value,gradient=scalar_tangent_nll_and_gradient(float(x[0]),errors,a,weights,huber_scale=RECIPE['huber_scale'])
        return value,np.array([gradient])
    searches=[]
    for start in RECIPE['optimizer_gamma_starts']:
        result=minimize(objective,np.array([start]),method='L-BFGS-B',jac=True,bounds=[(0.,None)],
            options=dict(maxiter=RECIPE['optimizer_maximum_iterations'],ftol=RECIPE['optimizer_ftol'],gtol=RECIPE['optimizer_gtol']))
        gamma=max(float(result.x[0]),0.);value,gradient=scalar_tangent_nll_and_gradient(gamma,errors,a,weights)
        if not np.isfinite([gamma,value,gradient]).all():raise ValueError('Finite scalar scale result required')
        searches.append(dict(gamma_start=start,gamma=gamma,training_NLL=value,gradient=gradient,
            success=bool(result.success),message=str(result.message),iterations=int(result.nit)))
    winner=min(range(len(searches)),key=lambda j:(searches[j]['training_NLL'],j));gamma=searches[winner]['gamma']
    model=dict(schema=SCHEMA,fit_partition='CAL',observation_family=OBSERVATION_FAMILY,recipe=deepcopy(RECIPE),
        gamma=gamma,pixel_floor_variance_px2=PIXEL_FLOOR_VARIANCE_PX2,bias_correction_px=0.,
        inventory_ids=inventory,inventory_frame_count=len(records),compatible_inventory_count=sum(r.get('compatible')is True for r in records),
        compatible_training_ids=eligible,fit_ids=[r['id']for r in train],fit_sessions=sessions,heldout_session=heldout_session,
        excluded_heldout_ids=[r['id']for r in records if r['session']==heldout_session],
        incompatible_inventory_ids=[r['id']for r in records if r.get('compatible')is not True],
        unsupported_training_ids=[fid for fid in eligible if fid not in {r['id']for r in train}],
        fit_frame_count=len(train),fit_tangent_count=len(points),source_counts=dict(Counter(p['source']for p in points)),
        source_group_counts=dict(Counter(p['source_group']for p in points)),state_counts=dict(Counter(p['state']for p in points)),
        session_weights={s:1./len(sessions)for s in sessions},fit_weight_sum=float(weights.sum()),
        optimizer_searches=searches,selected_optimizer_start_index=winner,training_NLL=searches[winner]['training_NLL'],
        fitted_variance_descriptive=_descriptive(errors,a,weights,PIXEL_FLOOR_VARIANCE_PX2+gamma*a),
        exact_old_x4_descriptive=_descriptive(errors,a,weights,4.*a),
        input_bindings=deepcopy(input_bindings or {}),CAL_reused_exploratory=True,default_adoption=False,
        covariance_calibration_not_independent_information=True)
    model['model_digest_sha256']=_digest(model)
    return model


def tangent_variance_parameters(model):
    if (model.get('schema')!=SCHEMA or model.get('fit_partition')!='CAL'or model.get('recipe')!=RECIPE
            or model.get('observation_family')!=OBSERVATION_FAMILY):raise ValueError('Separate declared archived-complete tangent CAL model required')
    if _digest({k:v for k,v in model.items()if k!='model_digest_sha256'})!=model.get('model_digest_sha256'):raise ValueError('Tangent CAL model digest changed')
    gamma=_number(model['gamma'],'gamma');floor=_number(model['pixel_floor_variance_px2'],'fixed pixel floor',positive=True)
    if gamma<0 or floor!=PIXEL_FLOOR_VARIANCE_PX2 or model['bias_correction_px']!=0.:raise ValueError('Fixed floor, nonnegative scale and zero mean required')
    return dict(gamma=gamma,pixel_floor_variance_px2=floor,model_digest_sha256=model['model_digest_sha256'],
        fit_partition='CAL',observation_family=OBSERVATION_FAMILY,
        q_covariance_guide_state_and_reliability_changed=False,independent_or_unique_information_claimed=False)


def calibrated_archived_tangent_variance(model,tangent_unit_xy,original_effective_fit_covariance_px2):
    """Pure variance-only export; no raw coordinates, image, state or file API."""
    parameters=tangent_variance_parameters(model)
    a=_variance_axis(tangent_unit_xy,original_effective_fit_covariance_px2)
    return dict(tangent_variance_px2=parameters['pixel_floor_variance_px2']+parameters['gamma']*a,
        original_variance_axis_px2=a,exact_old_x4_variance_px2=4.*a,**parameters)


def _verify_preparation(output):
    protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if (protocol['recipe']!=RECIPE or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(output/'CAL_TANGENT_INPUTS.jsonl.gz')!=seal['CAL_inputs_sha256']):raise ValueError('Prepared tangent model source/input seal changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):DIAG._verify(protocol[key])
    return protocol


def prepare(output=OUTPUT):
    """Compile main one-dir metadata only. No CAL numerical errors decoded."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier tangent preparations')
    parent=C.read(DIAG.OUTPUT/'PROTOCOL.json');dseal=C.read(DIAG.OUTPUT/'AUDIT_SEAL.json')
    pseal=C.read(CAL.PREPARED/'CAL_PREFLIGHT_SEAL_V1.json')
    if (C.sha(DIAG.OUTPUT/'AUDIT.json')!=dseal['audit_sha256']
            or C.sha(DIAG.OUTPUT/'PROTOCOL.json')!=dseal['protocol_sha256']
            or C.sha(CAL.PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz')!=pseal['prepared_groups_sha256']):raise ValueError('Frozen CAL metadata/error audit required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):DIAG._verify(parent[key])
    ids=parent['CAL46_inventory_ids'];sources=dict(parent['source_sha256'])
    for module in (sys.modules[__name__],C,CAL,DIAG):
        p=Path(module.__file__).resolve();sources[str(p.relative_to(C.ROOT))]=C.sha(p)
    paths=(CAL.PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz',CAL.PREPARED/'CAL_PREFLIGHT_SEAL_V1.json',
        DIAG.OUTPUT/'AUDIT.json',DIAG.OUTPUT/'AUDIT_SEAL.json',DIAG.OUTPUT/'PROTOCOL.json',DIAG.OUTPUT/'PREPARATION_SEAL.json',CAL.LABELS)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    for mapping in (sources,inputs,parent['protected_user_sha256']):DIAG._verify(mapping)
    output.mkdir(parents=True)
    protocol=dict(schema='CAL_ARCHIVED_COMPLETE_TANGENT_PREFIT_V24',recipe=RECIPE,ids=ids,rows=46,
        prepared_before_new_scale_fit_or_error_decode_UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=parent['protected_user_sha256'],
        CAL_label_path=str(CAL.LABELS.relative_to(C.ROOT)),CAL_error_path=str((DIAG.OUTPUT/'AUDIT.json').relative_to(C.ROOT)),
        CAL_errors_previously_exposed=True,no_direct_numeric_label_reopen_planned=True,source_binding_scope='conservative inherited union plus actual scalar fitter/runtime helper',
        no_TEST_or_REMAINDER_scores_predictions_or_labels=True,new_pose_image_NN_calls=0,current23_modified=False)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    prepared=[]
    for row in C.rows(CAL.PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz'):
        if row['partition']!='CAL':raise ValueError('CAL only')
        hidden=row['initial_geometry']['initial_self_hidden'];points=[]
        for plan in row['replacement_plan']['corner_plans']:
            point=_prepared_tangent(plan,hidden)
            if point is not None:points.append(point)
        prepared.append(dict(id=row['id'],partition='CAL',session=row['session'],tangents=points))
    if len(prepared)!=46 or [r['id']for r in prepared]!=ids or sum(len(r['tangents'])for r in prepared)!=30:raise ValueError('Exact complete CAL46 / all30 main one-dir metadata required')
    C.save_rows(output/'CAL_TANGENT_INPUTS.jsonl.gz',prepared)
    for mapping in (sources,inputs,parent['protected_user_sha256']):DIAG._verify(mapping)
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),CAL_inputs_sha256=C.sha(output/'CAL_TANGENT_INPUTS.jsonl.gz'),
        source_sha256=sources,input_sha256=inputs,rows=46,main_tangent_inventory=30,before_new_scale_fit_or_error_decode=True,
        source_counts=dict(Counter(p['source']for r in prepared for p in r['tangents'])),
        state_counts=dict(Counter(p['state']for r in prepared for p in r['tangents'])),numeric_error_or_label_decodes=0))
    print(dict(prepared_CAL46=True,main_tangents=30,numeric_error_decodes=0,new_scale_fit=False),flush=True)


def _joined_error_records(prepared,diagnostic):
    if diagnostic['inventory_frames']!=46 or diagnostic['compatible_frames']!=44:raise ValueError('Frozen original CAL46/fixed44 diagnostic required')
    known={r['id']:r for r in diagnostic['frames']};records=[]
    if len(known)!=46 or [r['id']for r in prepared]!=[r['id']for r in diagnostic['frames']]:raise ValueError('Exact ordered CAL46 diagnostic join required')
    for row in prepared:
        audit=known[row['id']]
        if row['session']!=audit['session']or audit['partition']!='CAL':raise ValueError('Fixed session/partition required')
        record=dict(id=row['id'],partition='CAL',session=row['session'],compatible=audit['compatible'],
            fixed_native_reference_phase=audit['fixed_native_reference_phase'],tangents=[])
        if not audit['compatible']:records.append(record);continue
        points={p['native_corner']:p for p in audit['points']if p['mode']=='ONE_DIRECTION_ARCHIVED_COMPLETE_TANGENT'}
        if set(points)!={p['native_corner']for p in row['tangents']}:raise ValueError('Every eligible main tangent retained without source trimming')
        for p in row['tangents']:
            prior=points[p['native_corner']];t=prior['tangent']
            if (p['state']!=prior['state']or p['source']!=prior['exact_measurement_source']or p['source_group']!=prior['source_group']
                    or p['archived_q_xy']!=prior['archived_q_xy']or p['effective_fit_covariance_px2']!=prior['immutable_effective_fit_covariance_px2']
                    or p['observed_covariance_px2']!=prior['immutable_observed_covariance_px2']
                    or p['fit_reliability_weight']!=prior['immutable_reliability_weight']
                    or p['original_guide_tangent_unit_xy']!=t['original_observer_tangent_unit_xy']
                    or p['exact_saved_old_tangent_variance_px2']!=t['current_variance_px2']
                    or not np.isclose(p['original_variance_axis_px2'],t['original_fit_variance_px2'],atol=1e-12,rtol=1e-12)):
                raise ValueError('Exact archived q/source/covariance/ORIGINALguide and x4 audit required')
            e=float(np.asarray(p['original_guide_tangent_unit_xy'])@np.asarray(prior['error_xy_px']))
            if not np.isclose(e,t['signed_error_px'],atol=1e-12,rtol=1e-12):raise ValueError('Same zero-mean fixed-native signed tangent error required')
            record['tangents'].append({**p,'signed_tangent_error_px':e})
        records.append(record)
    if sum(r['compatible']for r in records)!=44 or sum(len(r['tangents'])for r in records)!=30:raise ValueError('All30 compatible CAL main tangents retained')
    return records


def fit(output=OUTPUT):
    output=Path(output).resolve();protocol=_verify_preparation(output)
    if (output/'MODEL.json').exists():raise FileExistsError('Preserve fitted tangent model')
    prepared=list(C.rows(output/'CAL_TANGENT_INPUTS.jsonl.gz'))
    records=_joined_error_records(prepared,C.read(C.ROOT/protocol['CAL_error_path']))
    model=fit_archived_complete_tangent_model(records,input_bindings=protocol['input_sha256']);folds=[]
    for session in sorted({r['session']for r in records}):
        local=fit_archived_complete_tangent_model(records,heldout_session=session,input_bindings=protocol['input_sha256'])
        held=[r for r in records if r['session']==session and r['compatible']and r['tangents']]
        # One held session: equal held frames, then equal points within frame.
        hp=[];hw=[]
        for row in held:
            hp.extend(row['tangents']);hw.extend([1./len(held)/len(row['tangents'])]*len(row['tangents']))
        comparisons=None
        if hp:
            e=np.array([p['signed_tangent_error_px']for p in hp]);a=np.array([p['original_variance_axis_px2']for p in hp])
            comparisons=dict(fitted=_descriptive(e,a,hw,PIXEL_FLOOR_VARIANCE_PX2+local['gamma']*a),
                exact_old_x4=_descriptive(e,a,hw,4.*a))
        folds.append(dict(heldout_session=session,model=local,heldout_ids=[r['id']for r in records if r['session']==session],
            heldout_frames=len(held),heldout_tangents=len(hp),heldout_frame_balanced_comparisons=comparisons))
    C.write(output/'MODEL.json',model);C.save_rows(output/'CAL_TANGENT_ERRORS.jsonl.gz',records)
    C.write(output/'CAL_LOGO_REPORT.json',dict(inventory_frames=46,compatible_frames=44,all_main_tangents=30,LOGO_folds=folds,
        fit_frames=model['fit_frame_count'],source_counts=model['source_counts'],state_counts=model['state_counts'],
        matched_control_is_exact_old4a_not_recipe1plus4a=True,exploratory_CAL_reuse=True,new_reference_label_decodes=0,
        source_or_error_trimming=False,default_adoption=False))
    _verify_preparation(output)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),CAL_errors_sha256=C.sha(output/'CAL_TANGENT_ERRORS.jsonl.gz'),
        LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],fit_partition='CAL',inventory46=True,compatible44=True,
        new_pose_image_NN_or_direct_reference_label_decodes=0,current23_sources_parameters_modified=False,default_adoption=False))
    print(dict(gamma=model['gamma'],fit_frames=model['fit_frame_count'],fit_tangents=model['fit_tangent_count'],LOGO=len(folds),
        old_NLL=model['exact_old_x4_descriptive']['uncentered_NLL'],new_NLL=model['training_NLL']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.mode=='prepare'else fit)(args.output)
