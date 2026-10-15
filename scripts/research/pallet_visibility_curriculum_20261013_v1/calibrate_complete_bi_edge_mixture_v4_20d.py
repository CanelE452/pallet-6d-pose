"""Separate CAL noise for typed complete-BI/partial support-balanced mixtures.

Preparation reuses only the sealed main v20d replacement-plan active groups.
The support/fit identities and normalized hierarchical priors are fixed before
reference decode. CAL44 fixed native alignment supplies residuals; all CAL46
remain in the inventory. No image, pose fit, TEST or remaining label is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import types

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import common as C
from . import calibrate_native_edge_mixture_v4_18 as OLD
from . import complete_bi_image_chain_evidence_v4_20d as AUTH

PREPARED = OLD.OUTPUT.parent/'complete_bi_image_chain_evidence_v4_20d_checks'
OUTPUT = OLD.OUTPUT.parent/'CAL_complete_bi_edge_mixture_v4_20d'
LABELS = OLD.LABELS
SCHEMA = 'CAL_TYPED_COMPLETE_PARTIAL_SUPPORT_BALANCED_NMS_MIXTURE_NOISE_V20D'
RECIPE = {**deepcopy(OLD.RECIPE),
    'observation_generation':AUTH.NOISE_OBSERVATION_FAMILY,
    'applicability':'ONLY main self-typed v20d plan active groups; all CAL46 authentication inventory kept separately',
    'group':'one physical native edge; distinct inlier support sets; exact TLS fit variants within each support',
    'component_prior':'equal total mass per distinct support; equal split among its distinct exact fitted variants',
    'prior_used_in':'uncentered NLL, variance gradient, optimizer conditioning weightedmedians, LOGO heldout NLL, runtime mixture',
    'replacement':'self B exterior / I internal; all alternatives finite11/shift12/sine.25; B exact-chain topology and3px consistency',
    'overlap_policy':AUTH.PROTOTYPE_OVERLAP_POLICY,
    'one_direction_complete_tangent':'exact archived complete measurement and original observer guide; variance x4 engineering',
    'tangent_total_error_calibrated_by_this_model':False,
    'full_q_and_line_RGB_correlation':'possible; no independent information increase',
    'old_guard_calibration_validates_new_RGB_localization':False,
    'full_KP_or_projected_finite_3D_segment_guard_added':False}
conditional_covariance_floors = OLD.conditional_covariance_floors


def _component_priors(components):
    if any('mixture_component_prior'not in c for c in components):
        raise ValueError('Explicit new-generation hierarchical prior required')
    prior=np.asarray([c['mixture_component_prior']for c in components],float)
    if (not len(prior)or not np.isfinite(prior).all()or np.any(prior<=0)
            or not np.isclose(prior.sum(),1.,atol=1e-12,rtol=0)):
        raise ValueError('Explicit finite positive normalized support-balanced component priors required')
    return prior/prior.sum()


def _validate_support_priors(components):
    prior=_component_priors(components)
    counts=Counter(c['support_id']for c in components)
    keys=[(c['support_id'],c['fit_variant_id'])for c in components]
    if len(set(keys))!=len(keys):raise ValueError('Exact duplicate fit variants must already be collapsed')
    expected=np.asarray([1./len(counts)/counts[c['support_id']]for c in components])
    if not np.allclose(prior,expected,atol=1e-14,rtol=1e-12):
        raise ValueError('Equal support total prior split over its exact fit variants required')
    for c in components:
        if c.get('component_id')!=c['support_id']+':'+c['fit_variant_id']:
            raise ValueError('Exact support/fit/component identity required')
    return prior


def _training_groups(records, *, fit_ids=None, inventory_ids=None, heldout_session=None):
    records=list(records)
    if any(r.get('partition')!='CAL'for r in records):
        raise ValueError('CAL only; TEST and remaining reference inputs are forbidden')
    ids=[r['id']for r in records];inventory=list(ids if inventory_ids is None else inventory_ids)
    if len(set(ids))!=len(ids)or len(set(inventory))!=len(inventory)or set(ids)!=set(inventory):
        raise ValueError('Complete unique declared CAL inventory required')
    allowed=set(ids if fit_ids is None else fit_ids)
    if not allowed<=set(ids):raise ValueError('Fit IDs must belong to the CAL inventory')
    train=[];eligible=[]
    for row in records:
        if row['id']not in allowed or row.get('compatible')is not True or row['session']==heldout_session:
            continue  # Before reading excluded reference errors or group fields.
        eligible.append(row['id']);groups=[]
        for group in row['groups']:
            _validate_support_priors(group['components']);components=[]
            for component in group['components']:
                error=np.array([OLD._finite(component['normal_error_px'],'normal error'),
                    OLD._finite(component['angular_sine_error'],'angular sine error')])
                if abs(error[1])>1.+1e-12:raise ValueError('Angular sine must lie in [-1,1]')
                floor=conditional_covariance_floors(component['actual_inlier_span_px'],component['normal_floor_px'])
                supplied=np.asarray(component['conditional_variance_floor'],float)
                if supplied.shape!=(2,)or not np.allclose(floor,supplied,atol=1e-14,rtol=1e-12):
                    raise ValueError('CAL/runtime conditional covariance floors differ')
                components.append(dict(error=error,floor=floor,
                    mixture_component_prior=component['mixture_component_prior']))
            groups.append(components)
        if groups:train.append(dict(id=row['id'],session=row['session'],groups=groups))
    if not train:raise ValueError('At least one applicable compatible CAL training frame required')
    sessions=sorted({r['session']for r in train});counts=Counter(r['session']for r in train)
    groups=[];weights=[]
    for row in train:
        weight=1./(len(sessions)*counts[row['session']]*len(row['groups']))
        groups.extend(row['groups']);weights.extend([weight]*len(row['groups']))
    return records,inventory,eligible,train,sessions,groups,np.asarray(weights)


def mixture_noise_nll_and_gradient(variance_components, groups, weights, *, huber_scale=3.):
    """Uncentered cov-volume NLL, with the same support priors as runtime."""
    variance=np.asarray(variance_components,float);weights=np.asarray(weights,float);delta=float(huber_scale)
    if (variance.shape!=(2,)or not np.isfinite(variance).all()or np.any(variance<0)
            or weights.shape!=(len(groups),)or not np.isfinite(weights).all()or np.any(weights<=0)
            or not np.isfinite(delta)or delta<=0):raise ValueError('Finite nonnegative noise/positive weights required')
    value=0.;gradient=np.zeros(2)
    for components,weight in zip(groups,weights):
        prior=_component_priors(components)
        errors=np.asarray([c['error']for c in components],float)
        floors=np.asarray([c['floor']for c in components],float)
        covariance=variance+floors
        if (errors.shape!=(len(components),2)or not np.isfinite(errors).all()
                or floors.shape!=errors.shape or np.any(covariance<=0)or not np.isfinite(covariance).all()):
            raise ValueError('Finite positive total component covariances required')
        denominator=np.sqrt(1.+(errors*errors/covariance).sum(axis=1)/(delta*delta))
        robust=2.*delta*delta*(denominator-1.)
        logits=np.log(prior)-.5*np.log(covariance).sum(axis=1)-.5*robust
        log_density=float(logsumexp(logits));posterior=np.exp(logits-log_density)
        derivative=-.5/covariance+.5*errors*errors/(covariance*covariance*denominator[:,None])
        value+=-2.*weight*log_density;gradient+=-2.*weight*(posterior@derivative)
    return float(value),gradient


def fit_native_edge_mixture_model(records, *, fit_ids=None, inventory_ids=None,
        heldout_session=None, input_bindings=None):
    records,inventory,eligible,train,sessions,groups,weights=_training_groups(records,
        fit_ids=fit_ids,inventory_ids=inventory_ids,heldout_session=heldout_session)
    errors=[];floors=[];component_weights=[]
    for group,weight in zip(groups,weights):
        errors.extend(np.abs([c['error']for c in group]));floors.extend(c['floor']for c in group)
        component_weights.extend(weight*_component_priors(group))
    errors=np.asarray(errors);floors=np.asarray(floors)
    scales=np.array([max(OLD.OLD_CAL._weighted_median(errors[:,j],component_weights)/.67449,
        np.sqrt(OLD.OLD_CAL._weighted_median(floors[:,j],component_weights)))**2 for j in range(2)])
    def objective(scaled):
        value,gradient=mixture_noise_nll_and_gradient(np.asarray(scaled)*scales,groups,weights,
            huber_scale=RECIPE['huber_scale'])
        return value,gradient*scales
    searches=[]
    for start in RECIPE['optimizer_scaled_variance_starts']:
        result=minimize(objective,np.array(start),method='L-BFGS-B',jac=True,bounds=[(0.,None),(0.,None)],
            options=dict(maxiter=RECIPE['optimizer_maximum_iterations'],ftol=RECIPE['optimizer_ftol'],gtol=RECIPE['optimizer_gtol']))
        variance=np.maximum(np.asarray(result.x)*scales,0.)
        value,gradient=mixture_noise_nll_and_gradient(variance,groups,weights,huber_scale=RECIPE['huber_scale'])
        if not np.isfinite([*variance,value,*gradient]).all():raise ValueError('Finite noise fit result required')
        searches.append(dict(scaled_start=start,variance_components=variance.tolist(),training_mixture_nll=value,
            success=bool(result.success),message=str(result.message),iterations=int(result.nit),gradient_wrt_variance=gradient.tolist()))
    winner=min(range(len(searches)),key=lambda j:(searches[j]['training_mixture_nll'],j))
    variance=np.asarray(searches[winner]['variance_components'])
    model=dict(schema=SCHEMA,fit_partition='CAL',recipe=deepcopy(RECIPE),heldout_session=heldout_session,
        inventory_ids=inventory,inventory_frame_count=len(records),compatible_inventory_count=sum(r.get('compatible')is True for r in records),
        compatible_training_ids=eligible,fit_ids=[r['id']for r in train],fit_sessions=sessions,fit_frame_count=len(train),
        fit_physical_edge_count=len(groups),fit_alternative_count=sum(len(g)for g in groups),
        fit_distinct_support_count=sum(len({c['support_id']for c in g['components']})
            for r in records if r['id']in {x['id']for x in train}for g in r['groups']),
        unsupported_training_ids=[fid for fid in eligible if fid not in {r['id']for r in train}],
        excluded_heldout_ids=[r['id']for r in records if r['session']==heldout_session],
        incompatible_inventory_ids=[r['id']for r in records if r.get('compatible')is not True],
        sigma_normal_px=float(np.sqrt(variance[0])),sigma_sine=float(np.sqrt(variance[1])),
        normal_variance_component_px2=float(variance[0]),angular_variance_component_sine2=float(variance[1]),
        normal_angle_correlation=0.,normal_bias_correction_px=0.,angular_bias_correction=0.,fit_weight_sum=float(weights.sum()),
        session_weights={s:1./len(sessions)for s in sessions},optimizer_conditioning_variance_scales=scales.tolist(),
        optimizer_searches=searches,selected_optimizer_start_index=winner,training_mixture_nll=searches[winner]['training_mixture_nll'],
        input_bindings=deepcopy(input_bindings or {}),independent_generalization_or_probability_claim=False,
        model_is_for_new_image_NMS_only=True,archived_complete_tangent_total_error_not_calibrated=True)
    model['model_digest_sha256']=OLD._digest(model)
    return model


_EXPORT=types.FunctionType(OLD.mixture_noise_parameters.__code__,
    {**OLD.mixture_noise_parameters.__globals__,'SCHEMA':SCHEMA,'RECIPE':RECIPE,
        'MIX':types.SimpleNamespace(NOISE_OBSERVATION_FAMILY=AUTH.NOISE_OBSERVATION_FAMILY)},'export_complete_bi_noise')


def mixture_noise_parameters(model):
    result=_EXPORT(model)
    result['noise_provenance'].update(applicability_profile=RECIPE['applicability'],
        support_prior_policy=RECIPE['component_prior'],archived_complete_tangent_calibrated=False)
    return result


def _verify_preparation(output):
    protocol=OLD._verify_preparation(output)
    if protocol['recipe']!=RECIPE or protocol['schema']!='TYPED_COMPLETE_BI_SUPPORT_BALANCED_PREFIT_PROTOCOL_V20D':
        raise ValueError('Separate new-population support-balanced CAL preparation required')
    return protocol


def prepare(output=OUTPUT):
    """No labels decoded. Sealed main-plan IDs/priors copied, never reselected."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier CAL preparations')
    prep=C.read(PREPARED/'PREPARATION_SEAL.json');seal=C.read(PREPARED/'CAL_PREFLIGHT_SEAL_V1.json')
    report=C.read(PREPARED/'CAL_PREFLIGHT_V1.json');journal=PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz'
    if (C.sha(PREPARED/'PREPARATION_SEAL.json')!=seal['preparation_sha256']
            or C.sha(PREPARED/'CAL_PREFLIGHT_V1.json')!=seal['report_sha256']
            or C.sha(journal)!=seal['prepared_groups_sha256']or report['error_count']!=0
            or report['successful_metadata_rows']!=46):raise ValueError('Successful sealed main CAL46 preflight required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):OLD.E._verify_sources(prep[key])
    rows=list(C.rows(journal));ids=prep['ids']
    if len(rows)!=46 or len(set(ids))!=46 or [r['id']for r in rows]!=ids or any(r['partition']!='CAL'for r in rows):
        raise ValueError('Exact CAL46 only, no other partition decode')
    sources=dict(prep['source_sha256'])
    # Bind the conservative frozen dependency union, without claiming that
    # every inherited module executes during this small CAL fit.
    ancestry_protocols=[]
    for relative in prep['input_sha256']:
        if relative.endswith('/PROTOCOL.json'):
            ancestor=C.read(C.ROOT/relative)
            for key in ('source_sha256','input_sha256','protected_user_sha256'):
                if key in ancestor:OLD.E._verify_sources(ancestor[key])
            for path,digest in ancestor.get('source_sha256',{}).items():
                if path in sources and sources[path]!=digest:raise ValueError('Frozen source ancestry conflict')
                sources[path]=digest
            ancestry_protocols.append(relative)
    for module in (sys.modules[__name__],C,OLD,OLD.ALIGN,OLD.FACTOR,AUTH):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (journal,PREPARED/'PREPARATION_SEAL.json',
        PREPARED/'CAL_PREFLIGHT_SEAL_V1.json',PREPARED/'CAL_PREFLIGHT_V1.json',LABELS)}
    output.mkdir(parents=True)
    protocol=dict(schema='TYPED_COMPLETE_BI_SUPPORT_BALANCED_PREFIT_PROTOCOL_V20D',recipe=RECIPE,
        prepared_before_CAL_reference_decode_UTC=datetime.now(timezone.utc).isoformat(),ids=ids,rows=46,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=prep['protected_user_sha256'],
        CAL_label_path=str(LABELS.relative_to(C.ROOT)),other_partition_prediction_rows_decoded=0,
        TEST_or_remaining_label_paths=False,new_pose_fits=0,new_RGB_or_model_forwards=0,
        prior_identity_binding='support/variant/component and normalized weights sealed before CAL label decode',
        source_binding_scope='conservative inherited frozen source union plus explicit fitter/auth/kernel; not all union modules execute',
        ancestor_protocols=ancestry_protocols)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    for row in rows:
        groups=row['authenticated_group_bundle']['groups']
        if groups!=row['replacement_plan']['active_groups']:raise ValueError('Main runtime active-group fit population required')
        for group in groups:
            prior=_validate_support_priors(group['components'])
            if not np.array_equal(prior,np.asarray(group['component_weights'])):
                # Equal arithmetic normalization can differ at machine roundoff.
                if not np.allclose(prior,group['component_weights'],atol=1e-14,rtol=1e-12):raise ValueError('Runtime priors changed')
        row.update(all_NMS_physical_edges=len(row['authentication_inventory_groups']),
            all_NMS_alternatives=sum(len(g['components'])for g in row['authentication_inventory_groups']),
            all_NMS_source_chain_alternatives=len(row['authenticated_group_bundle']['actual_lattice_gradient_subpixel_relation']),
            all_NMS_alternatives_are_unique_fit_variants_not_source_directed_chain_count=True)
    C.save_rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz',rows)
    for mapping in (sources,inputs,prep['protected_user_sha256']):OLD.E._verify_sources(mapping)
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'),rows=46,
        source_sha256=sources,before_CAL_reference_decode=True,no_TEST_or_remaining_label_decode=True,
        runtime_applicable_group_count=sum(len(r['authenticated_group_bundle']['groups'])for r in rows),
        support_and_fit_variant_priors_fixed_before_labels=True))
    print(dict(prepared_CAL=46,active_groups=sum(len(r['authenticated_group_bundle']['groups'])for r in rows),
        active_variants=sum(len(g['components'])for r in rows for g in r['authenticated_group_bundle']['groups']),
        label_decodes=0),flush=True)


def fit(output=OUTPUT):
    """Isolated CAL44 fixed alignment; same signed anchor/gauge as runtime."""
    output=Path(output).resolve();protocol=_verify_preparation(output)
    if (output/'MODEL.json').exists():raise FileExistsError('Preserve fitted CAL model')
    prepared=list(C.rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'))
    labels=C.read(C.ROOT/protocol['CAL_label_path'])['rows'];records=[]
    for row in prepared:
        label=labels[row['id']]
        if label['metadata'].get('partition')!='CAL':raise ValueError('CAL reference only')
        alignment=OLD.ALIGN._reference_alignment({'runtime':{'initial_geometry':row['initial_geometry']}},label)
        record=dict(id=row['id'],partition='CAL',session=row['session'],compatible=bool(alignment['compatible']),
            fixed_3D_phase=int(alignment['phase']),geometry_gap_m=float(alignment['geometry_gap_m']),
            all_NMS_physical_edges=row['all_NMS_physical_edges'],all_NMS_alternatives=row['all_NMS_alternatives'],
            all_NMS_source_chain_alternatives=row['all_NMS_source_chain_alternatives'],
            runtime_applicable_group_count=len(row['authenticated_group_bundle']['groups']),groups=[])
        if alignment['compatible']:
            geometry=row['initial_geometry'];xyz=np.asarray(geometry['native_xyz']);K=np.asarray(geometry['K'])
            scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
            p=OLD.N.V._pose_vector(alignment['R_reference_native'],alignment['t_reference'],scale)
            for group in row['authenticated_group_bundle']['groups']:
                components=[];_validate_support_priors(group['components'])
                for edge in group['components']:
                    _,_,audit=OLD.FACTOR.projected_edge_residual_and_jacobian(p,xyz,K,scale,edge)
                    floor=conditional_covariance_floors(edge['actual_inlier_span_px'],edge['normal_floor_px'])
                    if not np.allclose(floor,[edge['normal_variance_px2'],edge['angular_variance_rad2']],atol=1e-14,rtol=1e-12):
                        raise ValueError('Preparation/runtime covariance floors differ')
                    components.append(dict(normal_error_px=audit['normal_residual_px'],angular_sine_error=audit['angular_sine_residual'],
                        actual_inlier_span_px=edge['actual_inlier_span_px'],normal_floor_px=edge['normal_floor_px'],
                        conditional_variance_floor=floor.tolist(),source_component=deepcopy(edge),
                        support_id=edge['support_id'],fit_variant_id=edge['fit_variant_id'],component_id=edge['component_id'],
                        mixture_component_prior=edge['mixture_component_prior']))
                record['groups'].append(dict(physical_native_edge=group['physical_native_edge'],components=components,
                    old_reliability_preserved_but_not_extra_CAL_weight=group['reliability_weight']))
        records.append(record)
    if len(records)!=46 or sum(r['compatible']for r in records)!=44:raise ValueError('Expected original CAL46 / fixed44')
    model=fit_native_edge_mixture_model(records,input_bindings=protocol['input_sha256']);folds=[]
    for session in sorted({r['session']for r in records}):
        local=fit_native_edge_mixture_model(records,heldout_session=session,input_bindings=protocol['input_sha256'])
        held=[r for r in records if r['session']==session and r['compatible']and r['groups']]
        held_groups=[[dict(error=np.array([c['normal_error_px'],c['angular_sine_error']]),
            floor=np.asarray(c['conditional_variance_floor']),mixture_component_prior=c['mixture_component_prior'])
            for c in g['components']]for r in held for g in r['groups']]
        held_nll=None
        if held_groups:
            held_nll=mixture_noise_nll_and_gradient([local['normal_variance_component_px2'],local['angular_variance_component_sine2']],
                held_groups,np.full(len(held_groups),1./len(held_groups)))[0]
        folds.append(dict(heldout_session=session,model=local,heldout_ids=[r['id']for r in records if r['session']==session],
            heldout_supported_frames=len(held),heldout_applicable_edge_count=len(held_groups),
            descriptive_heldout_equal_edge_mixture_nll=held_nll,same_component_support_priors_as_training_and_runtime=True,
            independent_probability_or_native_identity_guarantee=False))
    C.write(output/'MODEL.json',model);C.save_rows(output/'CAL_MIXTURE_ERRORS.jsonl.gz',records)
    C.write(output/'CAL_LOGO_REPORT.json',dict(inventory_frame_count=46,compatible_frame_count=44,LOGO_folds=folds,
        supported_fit_frames=model['fit_frame_count'],applicable_fit_edges=model['fit_physical_edge_count'],
        fit_alternatives=model['fit_alternative_count'],fit_supports=model['fit_distinct_support_count'],
        all_NMS_physical_edges=sum(r['all_NMS_physical_edges']for r in records),
        all_NMS_alternatives=sum(r['all_NMS_alternatives']for r in records),
        all_NMS_source_chain_alternatives=sum(r['all_NMS_source_chain_alternatives']for r in records),
        no_error_or_geometry_size_trimming=True,
        no_TEST_or_remaining_labels_read=True,calibrated_pose_probability_or_default_adoption=False))
    _verify_preparation(output)
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),CAL_errors_sha256=C.sha(output/'CAL_MIXTURE_ERRORS.jsonl.gz'),
        LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],fit_partition='CAL',CAL_inventory=46,
        compatible_inventory=44,new_pose_or_image_fits=0,default_adoption=False))
    print(dict(sigma_normal_px=model['sigma_normal_px'],sigma_sine=model['sigma_sine'],fit_frames=model['fit_frame_count'],
        physical_edges=model['fit_physical_edge_count'],variants=model['fit_alternative_count'],supports=model['fit_distinct_support_count']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.mode=='prepare'else fit)(args.output)
