"""CAL-only radial/tangent total-error diagnosis and opt-in measurement model.

Directions come from fixed initial native geometry, never target coordinates.
This empirical scatter/bias model is not a calibrated physical probability.
It has no RGB queries and does not change partial line observations.  Original
q, conditional covariance, effective covariance and reliability stay archived.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import statewise_uncertainty_calibration_v4 as OLD

OUTPUT = OLD.N / 'nonsevere_v4_fix/boundary_recovery_v1/CAL_radial_total_error_v4_9'
PARENT = OUTPUT.parent / 'shared_edge_partial_b_eval_v4_7'
LABELS = OLD.BASE / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
RECIPE = dict(
    schema='CAL_RADIAL_TANGENT_TOTAL_ERROR_RECIPE_V4_9',
    target='q minus fixed-native-compatible derived reference reprojection',
    human_DIRECT_annotation='separate audit, never substitutes reference fit target',
    fit_complete_BI_only=True, partial_B_reference_only_not_2D_fit=True,
    bias_estimator='session/frame/point-balanced coordinate weighted median',
    scatter_estimator='all untrimmed weighted residual outer products about robust bias',
    diagonal_shrinkage=0.75, minimum_points=20, minimum_frames=10,
    minimum_sessions=3, group_shrinkage_points=20., minimum_std_at_CAL_scale_px=1.,
    hierarchy='B/I corrected/fallback -> B/I -> all complete B/I',
    normalization='frozen nonS prediction scale, no clipping or S coordinate reads',
    basis='initial projected native corner minus projected native model center; outward radial then CCW tangent',
    conditional_covariance='Loewner lower bound only; no blind image-covariance-as-total assumption',
    reliability='frozen original reliability unchanged',
    TEST_labels_read=False, physical_probability_guarantee=False,
    LOGO_scope='only this noise/bias fit; upstream observation models are frozen all-CAL fitted inputs',
    adoption='candidate only; no pose accuracy evaluated by this module')


def _digest(x):
    import hashlib, json
    return hashlib.sha256(json.dumps(C.finite(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _basis(geometry, k):
    """Read only eligible nonS model vertex; no S 2D coordinate is formed."""
    xyz=np.asarray(geometry['native_xyz'],float)
    if xyz.shape!=(8,3) or not np.isfinite(xyz).all(): raise ValueError('Finite native XYZ8 required')
    center=OLD._project(xyz.mean(axis=0)[None],geometry['R_cf'],geometry['centroid'],geometry['K'])[0]
    corner=OLD._project(xyz[[k]],geometry['R_cf'],geometry['centroid'],geometry['K'])[0]
    radial=corner-center;length=float(np.linalg.norm(radial))
    if not np.isfinite(length) or length<=1e-8: raise ValueError('Projected corner-center radial axis is undefined')
    radial/=length
    return np.asarray([radial,[-radial[1],radial[0]]]),corner,center


def _weights(points):
    sessions={}
    for p in points:
        frames=sessions.setdefault(p['session'],{});frames[p['id']]=frames.get(p['id'],0)+1
    return np.asarray([1./(len(sessions)*len(sessions[p['session']])*sessions[p['session']][p['id']]) for p in points])


def _median(x,w):
    return np.asarray([OLD._quantile(x[:,j],w,.5) for j in range(x.shape[1])])


def _eligible(points):
    return (len(points)>=RECIPE['minimum_points'] and len({p['id'] for p in points})>=RECIPE['minimum_frames']
            and len({p['session'] for p in points})>=RECIPE['minimum_sessions'])


def _moments(points,minimum_variance):
    if not points: raise ValueError('Empty fit population')
    x=np.asarray([p['reference_error_rt_normalized'] for p in points],float);w=_weights(points);w/=w.sum()
    bias=_median(x,w);delta=x-bias
    scatter=np.einsum('n,ni,nj->ij',w,delta,delta)
    a=RECIPE['diagonal_shrinkage'];scatter=(1-a)*scatter+a*np.diag(np.diag(scatter))
    eig,V=np.linalg.eigh(scatter);scatter=(V*np.maximum(eig,minimum_variance))@V.T
    return dict(bias_rt_normalized=bias.tolist(),total_scatter_rt_normalized=scatter.tolist(),
                point_count=len(points),frame_count=len({p['id'] for p in points}),
                sessions=sorted({p['session'] for p in points}),eligible_independent_group=_eligible(points))


def fit_radial_total_error_model(points,cases,*,heldout_session=None,input_bindings=None):
    """Pure CAL fitter. Complete observations only; held-session excludes all fit stats."""
    points=list(points);cases=list(cases)
    if any(p.get('partition')!='CAL' for p in points) or any(c.get('partition')!='CAL' for c in cases):
        raise ValueError('CAL records only; TEST/FINAL cannot enter fitting')
    train=[p for p in points if p['compatible'] and not p['partial_B'] and p['session']!=heldout_session]
    train_cases=[c for c in cases if c['session']!=heldout_session]
    if not _eligible(train): raise ValueError('Insufficient complete CAL population for pooled model')
    scale_ref=float(np.median([c['nonself_scale_px'] for c in train_cases]))
    minvar=(RECIPE['minimum_std_at_CAL_scale_px']/scale_ref)**2
    pooled=_moments(train,minvar);states={};groups={}
    for state in ('B','I'):
        state_points=[p for p in train if p['state']==state]
        if _eligible(state_points):
            state_fit=_moments(state_points,minvar);state_fit['backoff']='state'
        else:
            state_fit={**deepcopy(pooled),'backoff':'pooled','requested_point_count':len(state_points),
                       'requested_sessions':sorted({p['session'] for p in state_points})}
        states[state]=state_fit
        for kind in ('corrected','fallback'):
            group=f'{state}:{kind}';subset=[p for p in train if p['group']==group]
            if _eligible(subset):
                fitted=_moments(subset,minvar);alpha=len(subset)/(len(subset)+RECIPE['group_shrinkage_points'])
                for key in ('bias_rt_normalized','total_scatter_rt_normalized'):
                    fitted[key]=(alpha*np.asarray(fitted[key])+(1-alpha)*np.asarray(state_fit[key])).tolist()
                fitted.update(backoff='group_shrunk_to_state',group_alpha=alpha)
            else:
                fitted={**deepcopy(state_fit),'backoff':f"{state_fit['backoff']}_insufficient_group",
                        'requested_point_count':len(subset),'requested_frame_count':len({p['id'] for p in subset}),
                        'requested_sessions':sorted({p['session'] for p in subset})}
            groups[group]=fitted
    model=dict(schema='CAL_RADIAL_TANGENT_TOTAL_ERROR_MODEL_V4_9',fit_partition='CAL',recipe=deepcopy(RECIPE),
        heldout_session=heldout_session,fit_ids=[c['id'] for c in train_cases],
        compatible_ids=[c['id'] for c in train_cases if c['compatible']],
        incompatible_ids=[c['id'] for c in train_cases if not c['compatible']],
        complete_fit_point_count=len(train),CAL_scale_reference_px=scale_ref,
        pooled=pooled,state_models=states,group_models=groups,input_bindings=input_bindings or {})
    model['model_sha256']=_digest(model)
    return model


def _effective_covariance(model_total,conditional):
    """PSD upper envelope in the conditional-covariance whitened basis."""
    conditional=np.asarray(conditional,float);model_total=np.asarray(model_total,float)
    if conditional.shape!=(2,2) or model_total.shape!=(2,2): raise ValueError('Covariance2x2 required')
    if not np.isfinite(conditional).all() or np.linalg.eigvalsh(conditional).min()<=0:
        raise ValueError('Conditional covariance must be finite SPD')
    e,V=np.linalg.eigh(conditional);root=(V*np.sqrt(e))@V.T;invroot=(V/np.sqrt(e))@V.T
    vals,axes=np.linalg.eigh(invroot@model_total@invroot)
    total=root@(axes*np.maximum(vals,1.))@axes.T@root
    return (total+total.T)/2


def apply_radial_total_error_model(visible,geometry,states,model,*,nonself_scale_px,partial_B_ids=()):
    """Pure runtime application, no targets/GT/RGB. Partial line reference records preserved."""
    if model.get('fit_partition')!='CAL' or model.get('schema')!='CAL_RADIAL_TANGENT_TOTAL_ERROR_MODEL_V4_9':
        raise ValueError('CAL radial model required')
    if _digest({k:v for k,v in model.items() if k!='model_sha256'})!=model['model_sha256']:
        raise ValueError('CAL model digest changed')
    scale=float(nonself_scale_px)
    if not np.isfinite(scale) or scale<=0: raise ValueError('Positive frozen nonS scale required')
    hidden=geometry['initial_self_hidden'];partial=set(map(int,partial_B_ids));result=[];audit=[]
    for source in visible:
        k=int(source['native_corner'])
        if not 0<=k<8 or hidden[k] or states[k]=='S':
            raise ValueError('Initial S prohibited before coordinate/covariance reads')
        if states[k] not in ('B','I') or source.get('state')!=states[k]:raise ValueError('Only current B/I observations')
        if k in partial:
            result.append(deepcopy(source));audit.append(dict(native_corner=k,applied=False,reason='partial_B_line_reference_exactly_preserved'));continue
        basis,_,_=_basis(geometry,k);q=np.asarray(source['xy'],float)
        group=OLD._group(source);fit=model['group_models'][group]
        bias_rt=scale*np.asarray(fit['bias_rt_normalized']);bias_xy=basis.T@bias_rt
        total=basis.T@(scale**2*np.asarray(fit['total_scatter_rt_normalized']))@basis
        effective=_effective_covariance(total,source['covariance_px2'])
        measurement=deepcopy(source)
        measurement.update(xy=(q-bias_xy).tolist(),fit_covariance_px2=effective.tolist(),
            original_xy_before_CAL_bias=source['xy'],original_conditional_covariance_px2=source['covariance_px2'],
            original_effective_covariance_px2=source.get('fit_covariance_px2'),
            CAL_radial_bias_adjustment_xy_px=bias_xy.tolist(),CAL_radial_model_sha256=model['model_sha256'],
            effective_noise_source='CAL_UNTRIMMED_TOTAL_SCATTER_WITH_CONDITIONAL_PSD_LOWER_BOUND',
            original_source=source.get('source'),bias_is_image_oracle=False)
        result.append(measurement);audit.append(dict(native_corner=k,applied=True,group=group,backoff=fit['backoff'],bias_rt_px=bias_rt.tolist()))
    return dict(visible_measurements=result,records=audit,S_coordinate_reads=0,RGB_queries=0,GT_inputs=False,
                partial_line_normal_offsets_or_variance_changed=False,old_reliability_unchanged=True)


def _dataset(rows,labels):
    if len(rows)!=46 or any(r['partition']!='CAL' for r in rows):raise ValueError('Exact primary CAL46 only')
    points=[];cases=[]
    for row in rows:
        label=labels[row['id']]
        if label['metadata']['partition']!='CAL':raise ValueError('CAL labels required')
        g=row['runtime']['initial_geometry'];alignment=OLD._reference_alignment(row,label)
        scale=float(row['partial_boundary_observation']['unchanged_nonself_prediction_scale_px'])
        partial={p['native_corner'] for p in row['partial_boundary_observation']['boundary_observations']}
        cases.append(dict(id=row['id'],partition='CAL',session=row['session'],compatible=alignment['compatible'],
            phase=alignment['phase'],phase_gaps_m=alignment['phase_gaps_m'],geometry_gap_m=alignment['geometry_gap_m'],
            nonself_scale_px=scale,initial_S_ids=[k for k,h in enumerate(g['initial_self_hidden']) if h],
            independent_physical_pose_truth=label['metadata'].get('independent_physical_pose_truth',False)))
        for measurement in row['frozen_BI_fit']:
            k=measurement['native_corner']
            if g['initial_self_hidden'][k] or row['states'][k]=='S':raise ValueError('S excluded before measurement/coordinate reads')
            if row['states'][k] not in ('B','I') or row['states'][k]!=measurement['state']:raise ValueError('B/I only')
            basis,coarse,center=_basis(g,k);q=np.asarray(measurement['xy'],float)
            ref=None;reference_error=None
            if alignment['compatible']:
                ref=OLD._project(alignment['xyz'][[k]],alignment['R_reference_native'],alignment['t_reference'],g['K'])[0]
                reference_error=basis@(q-ref)
            hstate=label['metadata']['human_states_native'][k];refid=int(alignment['mapping'][k]);ann=None;ann_error=None
            if hstate=='DIRECT_VISIBLE' and label['target']['valid'][refid]:
                ann=np.asarray(label['target']['gt'][refid],float);ann_error=basis@(q-ann)
            points.append(dict(id=row['id'],partition='CAL',session=row['session'],native_corner=k,state=measurement['state'],
                group=OLD._group(measurement),source=measurement.get('source'),partial_B=k in partial,
                compatible=alignment['compatible'],human_state=hstate,annotation_valid=ann is not None,
                nonself_scale_px=scale,basis_rt_from_xy=basis.tolist(),coarse_xy=coarse.tolist(),model_center_xy=center.tolist(),
                xy=q.tolist(),reference_xy=None if ref is None else ref.tolist(),annotation_xy=None if ann is None else ann.tolist(),
                reference_error_rt_px=None if ref is None else reference_error.tolist(),
                reference_error_rt_normalized=None if ref is None else (reference_error/scale).tolist(),
                annotation_error_rt_px=None if ann is None else ann_error.tolist(),
                measurement=deepcopy(measurement),geometry=g,states=row['states']))
    return points,cases


def _distribution(x):
    x=np.asarray(x,float)
    if not len(x):return dict(n=0)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),P90=float(np.percentile(x,90)),max=float(x.max()))


def _error_summary(points,key):
    valid=[p for p in points if p.get(key) is not None]
    if not valid:return dict(points=0)
    e=np.asarray([p[key] for p in valid]);w=_weights(valid);w/=w.sum()
    return dict(points=len(valid),frames=len({p['id'] for p in valid}),sessions=sorted({p['session'] for p in valid}),
        radial_px=_distribution(e[:,0]),tangent_px=_distribution(e[:,1]),norm_px=_distribution(np.linalg.norm(e,axis=1)),
        session_balanced_radial_mean_px=float(w@e[:,0]),session_balanced_tangent_mean_px=float(w@e[:,1]),
        session_balanced_radial_median_px=float(_median(e,w)[0]),session_balanced_tangent_median_px=float(_median(e,w)[1]))


def _evaluate(points,model):
    result=[]
    for p in points:
        if not p['compatible'] or p['partial_B']:continue
        applied=apply_radial_total_error_model([p['measurement']],p['geometry'],p['states'],model,nonself_scale_px=p['nonself_scale_px'])
        new=applied['visible_measurements'][0];basis=np.asarray(p['basis_rt_from_xy']);old_error=np.asarray(p['xy'])-p['reference_xy'];new_error=np.asarray(new['xy'])-p['reference_xy']
        old_cov=np.asarray(p['measurement']['fit_covariance_px2']);new_cov=np.asarray(new['fit_covariance_px2'])
        v=dict(id=p['id'],session=p['session'],native_corner=p['native_corner'],group=p['group'],human_state=p['human_state'],
            old_reference_norm_px=float(np.linalg.norm(old_error)),new_reference_norm_px=float(np.linalg.norm(new_error)),
            old_reference_mah2=float(old_error@np.linalg.solve(old_cov,old_error)),new_reference_mah2=float(new_error@np.linalg.solve(new_cov,new_error)),
            bias_rt_px=applied['records'][0]['bias_rt_px'],model_backoff=applied['records'][0]['backoff'],old_effective_trace_px2=float(np.trace(old_cov)),new_effective_trace_px2=float(np.trace(new_cov)))
        if p['annotation_xy'] is not None:
            oldann=np.asarray(p['xy'])-p['annotation_xy'];newann=np.asarray(new['xy'])-p['annotation_xy']
            v.update(old_annotation_norm_px=float(np.linalg.norm(oldann)),new_annotation_norm_px=float(np.linalg.norm(newann)))
        result.append(v)
    return result


def _evaluation_summary(rows):
    report={key:_distribution([r[key] for r in rows if key in r]) for key in
        ('old_reference_norm_px','new_reference_norm_px','old_annotation_norm_px','new_annotation_norm_px','old_reference_mah2','new_reference_mah2','old_effective_trace_px2','new_effective_trace_px2')}
    cap=float(-2*np.log(.2))
    report['descriptive_80pct_chi2_2_cap']=cap
    for kind in ('old','new'):
        covered=[r[f'{kind}_reference_mah2']<=cap for r in rows]
        frames={}
        for r,c in zip(rows,covered):frames.setdefault(r['id'],[]).append(c)
        report[f'{kind}_point_coverage_at_cap']=float(np.mean(covered)) if covered else None
        report[f'{kind}_frame_all_points_coverage_at_cap']=float(np.mean([all(c) for c in frames.values()])) if frames else None
    return report


def diagnose(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing CAL diagnostic: '+str(output))
    # Never open a TEST label or prediction journal. All source/data bindings are
    # pinned before opening the CAL coordinate labels and fitting candidates.
    seal=C.read(PARENT/'CAL_PREDICTION_SEAL.json');journal=PARENT/'CAL_PREDICTIONS.jsonl.gz'
    if C.sha(journal)!=seal['predictions_sha256'] or C.sha(PARENT/'PROTOCOL.json')!=seal['protocol_sha256']:raise ValueError('Sealed CAL observations required')
    bindings={str(p.relative_to(C.ROOT)):C.sha(p) for p in (journal,PARENT/'CAL_PREDICTION_SEAL.json',PARENT/'PROTOCOL.json',LABELS)}
    sources={str(Path(__file__).relative_to(C.ROOT)):C.sha(__file__),str(Path(OLD.__file__).relative_to(C.ROOT)):C.sha(OLD.__file__)}
    output.mkdir(parents=True)
    protocol=dict(schema='CAL_ONLY_RADIAL_TOTAL_ERROR_DIAGNOSIS_V4_9',recipe=RECIPE,input_sha256=bindings,source_sha256=sources,
        CAL=46,TEST_labels_read=False,TEST_predictions_read=False,GT_used_only_for_CAL_fit_and_postseal_diagnosis=True,
        no_pose_solver_or_model_forward=True,no_parameter_grid=True,prior_configuration_unchanged=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative in sources:
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest)
    rows=list(C.rows(journal));labels=C.read(LABELS)['rows'];points,cases=_dataset(rows,labels)
    model=fit_radial_total_error_model(points,cases,input_bindings=bindings);C.write(output/'MODEL.json',model)
    groups={}
    for group in sorted({p['group'] for p in points}):
        subset=[p for p in points if p['group']==group]
        groups[group]={part:dict(reference=_error_summary([p for p in subset if pred(p)],'reference_error_rt_px'),
            DIRECT_annotation=_error_summary([p for p in subset if pred(p)],'annotation_error_rt_px')) for part,pred in
            (('all_references',lambda p:True),('complete_2D',lambda p:not p['partial_B']),('partial_B_reference_only',lambda p:p['partial_B']),
             ('human_DIRECT_compatible',lambda p:p['compatible'] and p['human_state']=='DIRECT_VISIBLE'),
             ('human_SELF_compatible',lambda p:p['compatible'] and p['human_state']=='SELF_OCCLUDED'),
             ('native_incompatible_annotation_only',lambda p:not p['compatible']))}
    full_eval=_evaluate(points,model);logo=[]
    for session in sorted({c['session'] for c in cases}):
        fitted=fit_radial_total_error_model(points,cases,heldout_session=session,input_bindings=bindings)
        fold_points=[p for p in points if p['session']==session];evaluation=_evaluate(fold_points,fitted)
        C.write(output/'folds'/session/'MODEL.json',fitted)
        logo.append(dict(heldout_session=session,train_ids=fitted['fit_ids'],compatible_train_ids=fitted['compatible_ids'],
            heldout_ids=[c['id'] for c in cases if c['session']==session],model_sha256=fitted['model_sha256'],
            summary=_evaluation_summary(evaluation),rows=evaluation))
    logo_rows=[v for fold in logo for v in fold['rows']]
    C.write(output/'CAL_LOGO_REPORT.json',dict(schema='CAL_RADIAL_TOTAL_ERROR_LOGO_V4_9',folds=logo,summary=_evaluation_summary(logo_rows),
        conditional_on_all_CAL_fitted_upstream=True,fit_bias_scatter_minimums_and_hierarchy_exclude_held_session=True,physical_probability_guarantee=False))
    report=dict(schema='CAL_RADIAL_TOTAL_ERROR_EVIDENCE_V4_9',CAL=46,compatible_reference_frames=sum(c['compatible'] for c in cases),
        incompatible_frames=[c for c in cases if not c['compatible']],fixed_native_phase_counts=dict(Counter(str(c['phase']) for c in cases)),
        all_BI_references=len(points),compatible_BI_references=sum(p['compatible'] for p in points),
        complete_compatible_fit_points=model['complete_fit_point_count'],partial_B_reference_points=sum(p['partial_B'] for p in points),
        human_states=dict(Counter(p['human_state'] for p in points)),groups=groups,
        full_CAL_in_sample_summary=_evaluation_summary(full_eval),LOGO_summary=_evaluation_summary(logo_rows),
        coverage_is_descriptive_not_calibrated_physical_probability=True,untrimmed_scatter_tail_included=True,
        annotation_and_derived_reference_do_not_share_truth_certificate=True,model_candidate_not_adopted=True,
        no_pose_accuracy_claim=True,TEST_labels_or_predictions_read=False,S_coordinates_or_RGB_queries_read=False)
    C.write(output/'EVIDENCE_REPORT.json',report)
    # Point archive contains CAL coordinates only and explicitly separates
    # historical partial raw references from full active 2D measurements.
    C.save_rows(output/'CAL_POINT_DIAGNOSTICS.jsonl.gz',points)
    C.write(output/'FULL_CAL_FIT_DIAGNOSTICS.json',dict(summary=_evaluation_summary(full_eval),rows=full_eval))
    for relative,digest in {**bindings,**sources}.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Bound CAL source/input changed')
    C.write(output/'MODEL_SEAL.json',dict(model_sha256=C.sha(output/'MODEL.json'),LOGO_sha256=C.sha(output/'CAL_LOGO_REPORT.json'),
        evidence_sha256=C.sha(output/'EVIDENCE_REPORT.json'),protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=sources,input_sha256=bindings,
        CAL_only=True,TEST_labels_read=False,pose_solves=0,no_production_adoption=True))
    print(C.finite(dict(output=str(output),complete_fit_points=model['complete_fit_point_count'],groups={k:v['backoff'] for k,v in model['group_models'].items()},
        LOGO_summary=report['LOGO_summary'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();diagnose(args.output)
