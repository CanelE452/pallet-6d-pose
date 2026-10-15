"""Post-CAL exploratory complete-q / archived-tangent audit; no new fit.

A separate protocol and source snapshots precede isolated CAL reference decode.
All CAL46 are inventoried; only fixed-native compatible44 nonS B/I records are
quantified. The current23 source, noise model and all pose outputs are untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np

from . import common as C
from . import calibrate_complete_bi_edge_mixture_v4_20d as CAL

OUTPUT=CAL.OUTPUT.parent/'CAL_complete_tangent_audit_v4_23'


def _verify(mapping):
    CAL.OLD.E._verify_sources(mapping)


def _mode(plan):
    return ('RETAINED_COMPLETE_Q'if not plan['point_cost_revoked']else
        'ONE_DIRECTION_ARCHIVED_COMPLETE_TANGENT'if 'tangent_kind'in plan else 'RANK2_TYPED_LINES_ONLY')


def _source_group(source):
    if source=='VISIBLE_BACKBONE_FALLBACK':return 'BACKBONE_FALLBACK_COMPLETE'
    if source in ('B_TARGET_BACKGROUND_LINES','B_ACTUAL_BOUNDARY_RECOVERY','I_INTERNAL_RGB_JUNCTION'):
        return 'RGB_COMPLETE'
    return 'OTHER_EXACT_SOURCE_DISCLOSED'


def point_record(plan, reference_xy):
    """Eligible state gate precedes archived coordinate/covariance access."""
    if plan['state']not in ('B','I'):raise ValueError('Only nonS B/I archived complete records may be read')
    archive=plan['original_measurement_archive'];q=np.asarray(archive['original_observed_xy'],float)
    target=np.asarray(reference_xy,float);cov=np.asarray(archive['original_effective_fit_covariance_px2'],float)
    observed_cov=np.asarray(archive['original_observed_covariance_px2'],float)
    if q.shape!=(2,)or target.shape!=(2,)or cov.shape!=(2,2)or observed_cov.shape!=(2,2):raise ValueError('Finite 2D measurement/noise required')
    if not np.isfinite([*q,*target,*cov.ravel(),*observed_cov.ravel()]).all():raise ValueError('Finite measurement/noise required')
    if min(np.linalg.eigvalsh(cov).min(),np.linalg.eigvalsh(observed_cov).min())<=0:raise ValueError('Positive definite immutable covariances required')
    error=q-target;source=archive['original_measurement_source']
    result=dict(native_corner=plan['native_corner'],state=plan['state'],mode=_mode(plan),
        exact_measurement_source=source,source_group=_source_group(source),archived_q_xy=q.tolist(),
        fixed_native_reference_projection_xy=target.tolist(),error_xy_px=error.tolist(),radial_error_px=float(np.linalg.norm(error)),
        immutable_observed_covariance_px2=observed_cov.tolist(),immutable_effective_fit_covariance_px2=cov.tolist(),
        immutable_reliability_weight=archive['original_fit_reliability_weight'],
        effective_sigma_principal_px=np.sqrt(np.linalg.eigvalsh(cov)).tolist(),
        observed_sigma_principal_px=np.sqrt(np.linalg.eigvalsh(observed_cov)).tolist(),
        effective_point_mahalanobis_radius=float(np.sqrt(error@np.linalg.solve(cov,error))),
        observed_point_mahalanobis_radius=float(np.sqrt(error@np.linalg.solve(observed_cov,error))),
        point_q_is_exact_frozen_source_not_renamed_backbone=True,
        active_physical_edges=plan['active_physical_edges'])
    if 'tangent_kind'in plan:
        tangent=np.asarray(plan['tangent_unit_xy'],float)
        if tangent.shape!=(2,)or not np.isclose(tangent@tangent,1.,atol=1e-10,rtol=0):raise ValueError('Immutable original-guide unit tangent required')
        variance=float(plan['tangent_variance_px2']);original=float(tangent@cov@tangent)
        if not np.isclose(variance,4.*original,atol=1e-10,rtol=1e-10):raise ValueError('Current unchanged variance x4 required')
        signed=float(tangent@error);sigma=np.sqrt(variance)
        result['tangent']=dict(kind=plan['tangent_kind'],axis_source=plan['tangent_axis_source'],
            original_observer_tangent_unit_xy=tangent.tolist(),signed_error_px=signed,absolute_error_px=abs(signed),
            original_fit_variance_px2=original,current_variance_px2=variance,current_sigma_px=float(sigma),
            variance_inflation=4.,standard_deviation_inflation=2.,absolute_normalized_error=abs(signed)/sigma,
            signed_normalized_error=signed/sigma,original_fit_absolute_normalized_error=abs(signed)/np.sqrt(original),
            current_variance_or_parameter_changed=False,dependent_archived_prediction_not_new_RGB=True)
    return result


def _weighted_quantile(values,weights,fraction):
    values=np.asarray(values,float);weights=np.asarray(weights,float);order=np.argsort(values,kind='stable')
    pos=min(int(np.searchsorted(np.cumsum(weights[order]),weights.sum()*fraction)),len(order)-1)
    return float(values[order[pos]])


def _weights(rows,kind):
    if kind=='POINT_POOLED':return np.full(len(rows),1./len(rows))
    counts=Counter(r['id']for r in rows)
    if kind=='FRAME_BALANCED':return np.asarray([1./len(counts)/counts[r['id']]for r in rows])
    sessions=defaultdict(set)
    for r in rows:sessions[r['session']].add(r['id'])
    return np.asarray([1./len(sessions)/len(sessions[r['session']])/counts[r['id']]for r in rows])


def _stats(values,weights):
    x=np.asarray(values,float);w=np.asarray(weights,float);w=w/w.sum()
    return dict(mean=float(w@x),median=_weighted_quantile(x,w,.5),P90=_weighted_quantile(x,w,.9),
        P95=_weighted_quantile(x,w,.95),maximum=float(x.max()),minimum=float(x.min()))


def summarize(rows):
    if not rows:return dict(points=0,frames=0,sessions=0)
    result=dict(points=len(rows),frames=len({r['id']for r in rows}),sessions=len({r['session']for r in rows}),weightings={})
    for kind in ('POINT_POOLED','FRAME_BALANCED','SESSION_FRAME_BALANCED'):
        w=_weights(rows,kind)
        entry=dict(radial_error_px=_stats([r['radial_error_px']for r in rows],w),
            effective_point_mahalanobis_radius=_stats([r['effective_point_mahalanobis_radius']for r in rows],w),
            effective_point_mahalanobis_within_radius={str(v):float(w@np.array([r['effective_point_mahalanobis_radius']<=v for r in rows]))for v in (1,2,3)})
        if all('tangent'in r for r in rows):
            entry['tangent_absolute_error_px']=_stats([r['tangent']['absolute_error_px']for r in rows],w)
            entry['current_tangent_sigma_px']=_stats([r['tangent']['current_sigma_px']for r in rows],w)
            entry['tangent_absolute_normalized_error']=_stats([r['tangent']['absolute_normalized_error']for r in rows],w)
            entry['current_tangent_empirical_within_sigma']={str(v):float(w@np.array([r['tangent']['absolute_normalized_error']<=v for r in rows]))for v in (1,2,3)}
            entry['original_fit_tangent_empirical_within_sigma']={str(v):float(w@np.array([r['tangent']['original_fit_absolute_normalized_error']<=v for r in rows]))for v in (1,2,3)}
        result['weightings'][kind]=entry
    comparable=[r for r in rows if r.get('active_line_noise_comparisons')]
    if comparable:
        # Prior means are descriptive; never choose a best-fit component.
        scalars=[]
        for r in comparable:
            cc=r['active_line_noise_comparisons'];prior=np.array([c['mixture_component_prior']for c in cc]);prior/=prior.sum()
            scalars.append(dict(q_normal_absolute_error=float(prior@np.abs([c['archived_q_normal_error_px']for c in cc])),
                q_effective_normal_sigma=float(prior@np.array([c['archived_q_effective_normal_sigma_px']for c in cc])),
                line_total_normal_sigma=float(prior@np.array([c['new_line_total_normal_sigma_px']for c in cc])),
                line_anchor_absolute_error=float(prior@np.abs([c['reference_line_support_anchor_normal_error_px']for c in cc])),
                line_sigma_exceeds_q_effective_normal_sigma=float(prior@np.array([c['new_line_total_normal_sigma_px']>c['archived_q_effective_normal_sigma_px']for c in cc]))))
        result['different_gauge_line_comparison']=dict(points=len(comparable),
            support_prior_means_not_component_selection=True,
            likelihood_or_independent_precision_ordering_not_claimed=True,weightings={})
        for kind in ('POINT_POOLED','FRAME_BALANCED','SESSION_FRAME_BALANCED'):
            w=_weights(comparable,kind)
            result['different_gauge_line_comparison']['weightings'][kind]={key:_stats([r[key]for r in scalars],w)for key in scalars[0]}
    return result


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Append-only CAL diagnostic protocol')
    parent=CAL._verify_preparation(CAL.OUTPUT);model_seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    if C.sha(CAL.OUTPUT/'MODEL.json')!=model_seal['model_sha256']or C.sha(CAL.OUTPUT/'CAL_MIXTURE_ERRORS.jsonl.gz')!=model_seal['CAL_errors_sha256']:
        raise ValueError('Frozen CAL20d model/errors required')
    sources=dict(parent['source_sha256'])
    for module in (sys.modules[__name__],C,CAL,CAL.AUTH,CAL.OLD.ALIGN):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    paths=(CAL.PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz',CAL.PREPARED/'CAL_PREFLIGHT_SEAL_V1.json',
        CAL.OUTPUT/'MODEL.json',CAL.OUTPUT/'MODEL_SEAL.json',CAL.OUTPUT/'CAL_MIXTURE_ERRORS.jsonl.gz',
        CAL.OUTPUT/'PROTOCOL.json',CAL.OUTPUT/'PREPARATION_SEAL.json',CAL.LABELS)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    for mapping in (sources,inputs,parent['protected_user_sha256']):_verify(mapping)
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='CAL_COMPLETE_Q_TANGENT_EXPLORATORY_AUDIT_V23',
        created_before_this_audit_CAL_numeric_decode_UTC=datetime.now(timezone.utc).isoformat(),
        CAL_labels_previously_exposed_in_training=True,diagnostic_not_holdout_or_parameter_selection=True,
        CAL46_inventory_ids=parent['ids'],rows=46,source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],CAL_label_path=str(CAL.LABELS.relative_to(C.ROOT)),
        new_parameter_fits=0,new_pose_or_image_or_NN_calls=0,TEST_or_REMAINDER_scores_predictions_or_labels_read=False,
        measurements='fixed-native compatible44; every archived complete nonS B/I; no source/error trimming',
        partitions='CAL only; no full238 journal decode',current23_sources_or_parameters_modified=False,
        source_binding_scope='conservative inherited source union plus actual CAL audit functions; not all union modules execute'))
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=parent['protected_user_sha256'],
        before_this_audit_CAL_numeric_decode=True,not_a_new_model_or_performance_run=True))
    print(dict(prepared=True,rows=46,new_fits=0),flush=True)


def audit(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']:raise ValueError('Protocol changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    if (output/'AUDIT.json').exists():raise FileExistsError('Preserve exploratory result')
    prepared=list(C.rows(CAL.PREPARED/'CAL_PREPARED_GROUPS.jsonl.gz'))
    errors={r['id']:r for r in C.rows(CAL.OUTPUT/'CAL_MIXTURE_ERRORS.jsonl.gz')}
    if len(prepared)!=46 or [r['id']for r in prepared]!=protocol['CAL46_inventory_ids']or any(r['partition']!='CAL'for r in prepared):
        raise ValueError('Only exact CAL46 prepared inventory allowed')
    labels=C.read(C.ROOT/protocol['CAL_label_path'])['rows'];model=C.read(CAL.OUTPUT/'MODEL.json')
    CAL.mixture_noise_parameters(model)
    frames=[];points=[]
    for row in prepared:
        label=labels[row['id']]
        if label['metadata'].get('partition')!='CAL':raise ValueError('CAL reference required')
        align=CAL.OLD.ALIGN._reference_alignment({'runtime':{'initial_geometry':row['initial_geometry']}},label)
        known=errors[row['id']]
        if bool(align['compatible'])!=known['compatible']or int(align['phase'])!=known['fixed_3D_phase']:
            raise ValueError('Same immutable fixed-native CAL alignment required')
        frame=dict(id=row['id'],partition='CAL',session=row['session'],compatible=bool(align['compatible']),
            fixed_native_reference_phase=int(align['phase']),geometry_gap_m=float(align['geometry_gap_m']),points=[])
        if not align['compatible']:
            frame['quantitative_point_fields_read']=False;frames.append(frame);continue
        geometry=row['initial_geometry'];hidden=geometry['initial_self_hidden']
        xy=C.project(geometry['native_xyz'],align['R_reference_native'],align['t_reference'],geometry['K'])
        line_errors={}
        for group in known['groups']:
            pair=tuple(group['physical_native_edge'])
            for component in group['components']:line_errors[(pair,component['component_id'])]=component
        active={tuple(g['physical_native_edge']):g for g in row['replacement_plan']['active_groups']}
        for plan in row['replacement_plan']['corner_plans']:
            k=plan['native_corner']
            if hidden[k]or plan['state']not in ('B','I'):
                raise ValueError('S/nonBI never enters complete point audit')
            point=point_record(plan,xy[k]);point.update(id=row['id'],session=row['session'])
            comparisons=[]
            cov=np.asarray(point['immutable_effective_fit_covariance_px2']);err=np.asarray(point['error_xy_px'])
            for edge in plan['active_physical_edges']:
                group=active[tuple(edge)]
                for component in group['components']:
                    old=line_errors[(tuple(edge),component['component_id'])];normal=np.asarray(component['normal_unit_xy'])
                    normal_sigma=np.sqrt(model['normal_variance_component_px2']+component['normal_floor_px']**2)
                    comparisons.append(dict(physical_edge=edge,support_id=component['support_id'],fit_variant_id=component['fit_variant_id'],
                        component_id=component['component_id'],mixture_component_prior=component['mixture_component_prior'],
                        archived_q_normal_error_px=float(normal@err),archived_q_effective_normal_sigma_px=float(np.sqrt(normal@cov@normal)),
                        new_line_total_normal_sigma_px=float(normal_sigma),
                        reference_line_support_anchor_normal_error_px=old['normal_error_px'],
                        reference_line_angular_sine_error=old['angular_sine_error'],
                        q_and_line_have_different_residual_gauges_and_may_share_RGB=True,
                        line_has_angular_and_shared_edge_information_not_a_2D_point_variance_comparison=True))
            point['active_line_noise_comparisons']=comparisons;points.append(point);frame['points'].append(point)
        frames.append(frame)
    if len(frames)!=46 or sum(f['compatible']for f in frames)!=44:raise ValueError('CAL46 inventory / compatible44 retained')
    groups=defaultdict(list)
    for point in points:
        for dims in [('ALL',),('MODE',point['mode']),('SOURCE',point['source_group']),('EXACT_SOURCE',point['exact_measurement_source']),('STATE',point['state']),
                ('SOURCE_STATE_MODE',point['source_group'],point['state'],point['mode']),('SESSION',point['session']),
                ('SESSION_SOURCE_STATE_MODE',point['session'],point['source_group'],point['state'],point['mode'])]:groups['|'.join(dims)].append(point)
    result=dict(schema='CAL_ARCHIVED_COMPLETE_Q_AND_TANGENT_AUDIT_V23',inventory_frames=46,compatible_frames=44,
        all_compatible_complete_points=len(points),revoked_complete_points=sum(p['mode']!='RETAINED_COMPLETE_Q'for p in points),
        replacement_modes=dict(Counter(p['mode']for p in points)),source_groups=dict(Counter(p['source_group']for p in points)),
        exact_sources=dict(Counter(p['exact_measurement_source']for p in points)),summaries={key:summarize(value)for key,value in sorted(groups.items())},
        frames=frames,new_line_model=dict(sigma_normal_px=model['sigma_normal_px'],sigma_sine=model['sigma_sine'],model_sha256=C.sha(CAL.OUTPUT/'MODEL.json')),
        new_parameter_or_model_fits=0,pose_optimizations=0,image_or_NN_calls=0,TEST_or_REMAINDER_scores_predictions_or_labels_read=False,
        current23_sources_parameters_and_mainrun_modified=False,initial_S_coordinates_read=0,
        exploratory_CAL_reuse=True,normal_coverage_or_information_independence_certified=False,
        interpretation='Empirical CAL error sensitivity only. 2D point Mahalanobis radii and 1D tangent sigma coverage differ; old covariance, new line anchor noise and angular/shared-edge information are not equivalent likelihoods.')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    C.write(output/'AUDIT.json',result)
    C.write(output/'AUDIT_SEAL.json',dict(audit_sha256=C.sha(output/'AUDIT.json'),preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=protocol['source_sha256'],input_sha256=protocol['input_sha256'],
        protected_user_sha256=protocol['protected_user_sha256'],all_sources_inputs_protected_reverified=True))
    print(dict(complete=len(points),modes=result['replacement_modes'],sources=result['source_groups'],audit_sha256=C.sha(output/'AUDIT.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','audit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.mode=='prepare'else audit)(args.output)
