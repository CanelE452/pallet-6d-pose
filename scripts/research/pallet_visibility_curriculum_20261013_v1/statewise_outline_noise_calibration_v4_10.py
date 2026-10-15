"""CAL-only total signed global-mask envelope bias/noise, no pose fitting.

Unordered body vertices and derived body pose define the CAL silhouette.
No semantic keypoint targets or S/U-local image/coordinate query enters here.
Frames receive equal total weight regardless of boundary pixel count. The
scalar bias/scatter is an opt-in engineering model, not calibrated physical
confidence; correlated silhouettes and four sessions limit LOGO conclusions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import shutil

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import common as C
from . import statewise_uncertainty_calibration_v4 as OLD
from . import statewise_global_outline_solver_v4_10 as OUTLINE

OUTPUT=OLD.N/'nonsevere_v4_fix/boundary_recovery_v1/CAL_global_outline_noise_v4_10'
PARENT=OUTPUT.parent/'shared_edge_partial_b_eval_v4_7'
MASKS=OUTPUT.parent/'image_evidence_v4_6'
LABELS=OLD.BASE/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
RECIPE=dict(schema='CAL_WHOLE_OUTLINE_NORMAL_NOISE_RECIPE_V4_10',
    population='all original CAL46; no native-compatible or human-visibility filter',
    target='unordered cuboid(body_xyz) @ body_R.T + t from existing derived reference',
    sign='observed mask outside reference silhouette positive; inside negative',
    sample_scope='observed convex-mask real segments clipped inside existing frame margin7, no canvas closing segment',
    normalization='observed frozen-mask hull bounding-box diagonal',
    frame_weight='equal total frame weight; within-frame arc samples average only',
    bias_estimator='median of per-frame median signed normalized distances',
    noise_estimator='1.4826 times frame-equal weighted median absolute signed residual about global bias',
    minimum_runtime_sigma_px=1.,minimum_normalized_sigma=1e-8,
    material_hierarchy='global shared model only; single wood session cannot validate independent wood generalization',
    LOGO='all bias/noise/normalization fitting stats exclude held-out capture session',
    independent_pixel_count_likelihood_claimed=False,physical_probability_guarantee=False,
    GT_is_independently_measured=False,TEST_labels_read=False,pose_fits=0,
    model_adoption='candidate only; current defaults unchanged')


def _digest(value):
    import json
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _polygon(vertices):
    points=np.asarray(vertices,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():raise ValueError('Finite vertices required')
    polygon=cv2.convexHull(points.astype('float32')).reshape(-1,2).astype(float)
    if len(polygon)<3:raise ValueError('Degenerate silhouette')
    area=np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))
    if abs(area)<1e-8:raise ValueError('Degenerate silhouette area')
    if area<0:polygon=polygon[::-1]
    return polygon


def signed_nearest_segment_distances(points,polygon):
    """Exact nearest finite segment distance; geometric inside/outside sign."""
    points=np.asarray(points,float);poly=_polygon(polygon)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Finite sample points required')
    a=poly;b=np.roll(poly,-1,axis=0);direction=b-a;squared=np.sum(direction**2,axis=1)
    alpha=np.sum((points[:,None,:]-a)*direction,axis=-1)/squared
    foot=a+np.clip(alpha,0,1)[...,None]*direction;residual=points[:,None,:]-foot
    index=np.argmin(np.sum(residual**2,axis=-1),axis=1);nearest=foot[np.arange(len(points)),index]
    distance=np.linalg.norm(points-nearest,axis=1)
    relative=points[:,None,:]-a
    cross=direction[None,:,0]*relative[:,:,1]-direction[None,:,1]*relative[:,:,0]
    inside=np.min(cross,axis=1)>=-1e-8
    signed=np.where(inside,-distance,distance)
    normals=np.column_stack([direction[:,1],-direction[:,0]])/np.sqrt(squared)[:,None]
    return dict(signed_distances_px=signed,nearest_xy=nearest,outward_normals_xy=normals[index],segment_indices=index,
                nearest_distance_is_finite_segment_not_infinite_line=True)


def fit_outline_noise_model(records,*,heldout_session=None,input_bindings=None):
    """Pure frame-equal CAL fitter, no individual semantic corner target."""
    records=list(records)
    if any(r.get('partition')!='CAL' for r in records):raise ValueError('CAL only; TEST forbidden')
    train=[r for r in records if r['supported'] and r['session']!=heldout_session]
    if len(train)<3 or len({r['session'] for r in train})<2:raise ValueError('At least three frames/two CAL sessions required')
    medians=np.asarray([np.median(r['signed_distances_normalized']) for r in train])
    bias=float(np.median(medians));errors=[];weights=[]
    for r in train:
        e=np.abs(np.asarray(r['signed_distances_normalized'])-bias)
        errors.extend(e);weights.extend([1./(len(train)*len(e))]*len(e))
    mad=OLD._quantile(errors,weights,.5);sigma=max(float(1.4826*mad),RECIPE['minimum_normalized_sigma'])
    model=dict(schema='CAL_WHOLE_OUTLINE_NORMAL_NOISE_MODEL_V4_10',fit_partition='CAL',recipe=deepcopy(RECIPE),
        heldout_session=heldout_session,bias_ratio=bias,total_normal_sigma_ratio=sigma,
        fit_ids=[r['id'] for r in train],fit_sessions=sorted({r['session'] for r in train}),
        unsupported_CAL_ids=[r['id'] for r in records if not r['supported']],
        excluded_heldout_ids=[r['id'] for r in records if r['session']==heldout_session],
        fit_frame_count=len(train),boundary_pixel_count_is_not_effective_sample_count=True,
        material_model='shared_global',material_dependent_model_fit=False,input_bindings=input_bindings or {})
    model['model_sha256']=_digest(model)
    return model


def outline_noise_parameters(model,mask_diagonal_px):
    """Pure generic runtime parameters; no GT, image/KP query or inference."""
    if model.get('fit_partition')!='CAL' or model.get('schema')!='CAL_WHOLE_OUTLINE_NORMAL_NOISE_MODEL_V4_10':raise ValueError('CAL outline model required')
    if _digest({k:v for k,v in model.items() if k!='model_sha256'})!=model['model_sha256']:raise ValueError('Model digest changed')
    diagonal=float(mask_diagonal_px)
    if not np.isfinite(diagonal) or diagonal<=0:raise ValueError('Positive frozen mask diagonal required')
    return dict(outward_mask_bias_px=diagonal*model['bias_ratio'],
                total_normal_sigma_px=max(model['recipe']['minimum_runtime_sigma_px'],diagonal*model['total_normal_sigma_ratio']),
                model_sha256=model['model_sha256'],bias_applies_to='observed-minus-model signed residual',
                per_pixel_independence_claim=False,calibrated_physical_probability=False)


def _summary(values):
    a=np.asarray(values,float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),P10=float(np.quantile(a,.1)),
                P90=float(np.quantile(a,.9)),minimum=float(a.min()),maximum=float(a.max()))


def _evaluate(records,model):
    result=[]
    for record in records:
        if not record['supported']:continue
        parameters=outline_noise_parameters(model,record['mask_diagonal_px'])
        x=np.asarray(record['signed_distances_px']);centered=x-parameters['outward_mask_bias_px'];sigma=parameters['total_normal_sigma_px']
        result.append(dict(id=record['id'],session=record['session'],material=record['material'],
            mask_diagonal_px=record['mask_diagonal_px'],bias_px=parameters['outward_mask_bias_px'],sigma_px=sigma,
            old_median_signed_px=float(np.median(x)),new_median_signed_px=float(np.median(centered)),
            old_mean_abs_px=float(np.mean(np.abs(x))),new_mean_abs_px=float(np.mean(np.abs(centered))),
            old_P90_abs_px=float(np.quantile(np.abs(x),.9)),new_P90_abs_px=float(np.quantile(np.abs(centered),.9)),
            conditional1px_point_fraction_within_1sigma=float(np.mean(np.abs(x)<=1.)),
            empirical_total_point_fraction_within_1sigma=float(np.mean(np.abs(centered)<=sigma)),
            empirical_total_point_fraction_within_2sigma=float(np.mean(np.abs(centered)<=2*sigma)),
            all_boundary_within_2sigma=bool(np.max(np.abs(centered))<=2*sigma)))
    return result


def _evaluation_summary(records):
    return {key:_summary([r[key] for r in records]) for key in (
        'old_median_signed_px','new_median_signed_px','old_mean_abs_px','new_mean_abs_px','old_P90_abs_px','new_P90_abs_px',
        'bias_px','sigma_px','conditional1px_point_fraction_within_1sigma','empirical_total_point_fraction_within_1sigma',
        'empirical_total_point_fraction_within_2sigma','all_boundary_within_2sigma')}


def _dataset(rows,labels,metadata):
    records=[]
    if len(rows)!=46 or any(r['partition']!='CAL' for r in rows):raise ValueError('Exact CAL46 required')
    for row in rows:
        label=labels[row['id']]
        if label['metadata']['partition']!='CAL':raise ValueError('CAL label metadata required')
        truth=label['target']['truth'];g=row['runtime']['initial_geometry'];meta=metadata[row['id']]
        if meta['partition']!='CAL' or meta['image_sha256']!=row['image_sha256']:raise ValueError('CAL image/mask binding mismatch')
        maskpath=MASKS/meta['mask_npz']
        if C.sha(maskpath)!=meta['mask_file_sha256']:raise ValueError('Frozen mask file mismatch')
        with np.load(maskpath,allow_pickle=False) as arrays:mask=arrays['pallet_mask_clean'].copy()
        observation=OUTLINE.observe_global_outline(mask,g['image_hw'],image_binding=dict(image_sha256=row['image_sha256'],mask_file_sha256=meta['mask_file_sha256']))
        body=OLD._cuboid(truth['body_xyz']);bodyR=np.asarray(truth['body_R']);t=np.asarray(truth['t'])
        camera=body@bodyR.T+t
        if not np.isfinite(camera).all() or camera[:,2].min()<=0:raise ValueError('CAL body reference has invalid projected depth')
        ref_projection=OLD._project(body,bodyR,t,g['K']);polygon=_polygon(ref_projection)
        # Unordered vertex-set check removes native corner phase/numbering.
        Qref=bodyR.T@np.asarray(truth['R']);reference_object=body@Qref
        registered_model=np.asarray(g['native_xyz'])@np.asarray(g['physical_registration_Q'])
        distances=np.linalg.norm(registered_model[:,None,:]-reference_object[None,:,:],axis=-1)
        ii,jj=linear_sum_assignment(distances);unordered_gap=float(distances[ii,jj].max())
        semantic_alignment=OLD._reference_alignment(row,label)
        record=dict(id=row['id'],partition='CAL',session=row['session'],material='wood' if row['session'].startswith('wood') else 'plastic',
            supported=bool(observation['supported']),mask_binding=observation['image_binding'],mask_array_sha256=observation['mask_array_sha256'],
            mask_npz=meta['mask_npz'],image_hw=g['image_hw'],configuration=observation['configuration'],
            unordered_registered_model_body_max_matching_gap_m=unordered_gap,
            fixed_native_compatible=semantic_alignment['compatible'],fixed_native_phase=semantic_alignment['phase'],
            fixed_native_matching_gap_m=semantic_alignment['geometry_gap_m'],
            reference_is_derived_not_independent_physical_GT=True,S_or_U_local_coordinates_or_queries_read=0,
            no_native_corner_correspondence_used_for_shape_error=True)
        if not observation['supported']:
            record['reason']=observation['reason'];records.append(record);continue
        observed=np.asarray(observation['observed_boundary_samples_xy']);diagonal=float(np.linalg.norm(np.ptp(np.asarray(observation['polygon_xy']),axis=0)))
        signed=signed_nearest_segment_distances(observed,polygon)['signed_distances_px']
        reverse=signed_nearest_segment_distances(ref_projection,observation['polygon_xy'])['signed_distances_px']
        record.update(mask_diagonal_px=diagonal,mask_area_px=observation['mask_area_px'],hull_area_px=observation['foreground_hull_area_px'],
            observed_polygon_xy=observation['polygon_xy'],reference_body_polygon_xy=polygon.tolist(),
            observed_boundary_sample_count=len(observed),signed_distances_px=signed.tolist(),signed_distances_normalized=(signed/diagonal).tolist(),
            signed_summary_px=_summary(signed),absolute_summary_px=_summary(np.abs(signed)),
            frame_median_signed_px=float(np.median(signed)),frame_median_signed_ratio=float(np.median(signed/diagonal)),
            outside_fraction=float(np.mean(signed>0)),inside_fraction=float(np.mean(signed<0)),
            reverse_reference_vertices_mask_signed_px=reverse.tolist(),mask_boundary_pixels_not_independent_measurements=True)
        records.append(record)
    return records


def calibrate(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve CAL outline evidence '+str(output))
    journal=PARENT/'CAL_PREDICTIONS.jsonl.gz';seal=C.read(PARENT/'CAL_PREDICTION_SEAL.json')
    if C.sha(journal)!=seal['predictions_sha256']:raise ValueError('CAL observation journal changed')
    maskseal=C.read(MASKS/'SEAL.json');metapath=MASKS/'RECORDS.json'
    if maskseal['records']!=95 or maskseal['GT_or_pose_labels_read'] or C.sha(metapath)!=maskseal['records_sha256']:raise ValueError('Frozen GT-free masks required')
    metadata={r['id']:r for r in C.read(metapath)};rows=list(C.rows(journal));paths=[journal,PARENT/'CAL_PREDICTION_SEAL.json',LABELS,MASKS/'SEAL.json',metapath]
    paths.extend(MASKS/metadata[r['id']]['mask_npz'] for r in rows)
    bindings={str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}
    sources={str(Path(p).relative_to(C.ROOT)):C.sha(p) for p in (__file__,OLD.__file__,OUTLINE.__file__)}
    output.mkdir(parents=True);C.write(output/'PROTOCOL.json',dict(schema='CAL_GLOBAL_MASK_ENVELOPE_TOTAL_NOISE_PROTOCOL_V4_10',recipe=RECIPE,
        source_sha256=sources,input_sha256=bindings,CAL=46,TEST_label_files_read=False,
        model_candidate_not_default=True,all_CAL_including_native_incompatible_kept=True,no_new_RGB_forward_or_pose_solves=True))
    for relative in sources:
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest)
    labels=C.read(LABELS)['rows'];records=_dataset(rows,labels,metadata)
    model=fit_outline_noise_model(records,input_bindings=bindings);C.write(output/'MODEL.json',model)
    full=_evaluate(records,model);folds=[]
    for session in sorted({r['session'] for r in records}):
        fold=fit_outline_noise_model(records,heldout_session=session,input_bindings=bindings)
        evaluation=_evaluate([r for r in records if r['session']==session],fold)
        C.write(output/'folds'/session/'MODEL.json',fold)
        folds.append(dict(heldout_session=session,train_ids=fold['fit_ids'],heldout_ids=[r['id'] for r in records if r['session']==session],
            bias_ratio=fold['bias_ratio'],sigma_ratio=fold['total_normal_sigma_ratio'],summary=_evaluation_summary(evaluation),rows=evaluation))
    logo=[r for f in folds for r in f['rows']]
    report=dict(schema='CAL_WHOLE_MASK_BIAS_NOISE_EVIDENCE_V4_10',CAL=46,supported=sum(r['supported'] for r in records),
        unsupported=[dict(id=r['id'],reason=r['reason']) for r in records if not r['supported']],
        global_model=model,full_CAL_in_sample_summary=_evaluation_summary(full),CAL_LOGO_summary=_evaluation_summary(logo),
        by_session={s:dict(frame_count=sum(r['session']==s for r in records),
            signed_frame_medians_px=_summary([r['frame_median_signed_px'] for r in records if r['supported'] and r['session']==s])) for s in sorted({r['session'] for r in records})},
        unordered_vertex_matching_gaps=[{k:r[k] for k in ('id','unordered_registered_model_body_max_matching_gap_m','fixed_native_compatible','fixed_native_phase','fixed_native_matching_gap_m')} for r in records],
        all_CAL_native_incompatible_frames_kept=True,frame_equal_not_pixel_independent=True,
        one_wood_session_cannot_validate_material_generalization=True,reference_derived_not_independent_physical_truth=True,
        no_pose_accuracy_or_TEST_effect_claim=True)
    C.write(output/'EVIDENCE_REPORT.json',report);C.write(output/'CAL_LOGO_REPORT.json',dict(folds=folds,summary=_evaluation_summary(logo),
        conditional_upstream_all_CAL_fit=True,all_fit_stats_exclude_held_session=True,physical_probability_guarantee=False))
    C.write(output/'FULL_CAL_FIT_DIAGNOSTICS.json',dict(rows=full,summary=_evaluation_summary(full)))
    C.save_rows(output/'CAL_FRAME_MASK_RESIDUALS.jsonl.gz',records)
    for relative,digest in {**sources,**bindings}.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('CAL source/input changed')
    C.write(output/'MODEL_SEAL.json',dict(schema='CAL_WHOLE_OUTLINE_MODEL_SEAL_V4_10',model_sha256=C.sha(output/'MODEL.json'),
        internal_model_sha256=model['model_sha256'],report_sha256=C.sha(output/'EVIDENCE_REPORT.json'),LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=sources,input_sha256=bindings,
        CAL_only=True,TEST_labels_read=False,pose_solves=0,RGB_forward=0,model_adopted=False))
    print(C.finite(dict(output=str(output),supported=report['supported'],bias_ratio=model['bias_ratio'],sigma_ratio=model['total_normal_sigma_ratio'],
        full_CAL_summary=report['full_CAL_in_sample_summary'],LOGO_summary=report['CAL_LOGO_summary'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();calibrate(args.output)
