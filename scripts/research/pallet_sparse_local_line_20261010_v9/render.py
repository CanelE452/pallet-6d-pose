"""Fixed local C2 plots and original six cases; no model/pose/GT scoring.

Stored unused semantic line support is overlaid only on the LOCAL panel.
Local initialization, diagnostic inlier support and numerical NEW are labelled;
none is represented as independent global pose or physical-truth validation.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from . import common as C

CASE_PROTOCOL = C.REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'
SHOWN = ('BASE','N3_SUBPIX',C.COMPARATOR,C.PRIMARY)


def display_row(row):
    solver = row.get('solver') or {}
    new = row['new_pose_estimated']
    pool = solver.get('factor_pool')
    return dict(id=row['id'],method=row['method'],native_points=row['native_points'],pose=row['pose'],
        output_status=row['output_status'],new_pose_estimated=new,input_points=row.get('input_points'),
        reprojected_ids=row.get('reprojected_ids',[]),selected_corner_ids=row.get('selected_corner_ids',[]),
        evaluation_reference=row.get('evaluation_reference',{}),head_arm=row.get('head_arm','FIXED_CONTROL'),
        actual_used_U=solver.get('used',[]),applied_H=row.get('hidden_initial',[]),
        fallback_used=row['fallback_used'],pose_available=row['pose_available'],
        accepted_fit_ids=solver.get('fit_input_ids',[]) if new else [],
        accepted_inlier_ids=solver.get('final_inliers',[]) if new else [],
        raw_candidate_diagnostic_inliers=solver.get('final_inliers',[]),
        original_H=row.get('original_self_hidden_initial',[]),
        actual_unused_line_edges=solver.get('line_edges',[]),
        accepted_fit_line_edges=solver.get('fit_line_edges',[]) if new else [],
        diagnostic_line_inliers=solver.get('final_line_inliers',[]) if new else [],
        factor_pool=pool,initial_pose_start_used=solver.get('initial_pose_start_used'),
        diagnostic_inlier_scalar_residuals=solver.get('diagnostic_inlier_scalar_residuals'),
        diagnostic_inlier_support_rank6=solver.get('diagnostic_inlier_support_rank6'),
        diagnostic_inlier_geometry_unavailable=solver.get('diagnostic_inlier_geometry_unavailable'),
        numeric_LOCAL_NEW_is_not_accuracy_success=row['method']==C.PRIMARY,
        local_global_uniqueness_proven=False if row['method']==C.PRIMARY else None)


def _render(args):
    C.verify_protocol(args);C.protect(args)
    folder = Path(args.output)
    scoring = C.read(folder/'SCORING_RECEIPT.json')
    C.require(scoring['complete'] is True and scoring['actual_total_scored_rows'] == 980 and
        scoring['new_local_optimizers'] == 0,'complete actual local/control scores required')
    C.bound(args.protocol,scoring['protocol'],'scoring protocol')
    for key,name in (('predictions','PREDICTIONS.jsonl.gz'),('comparator_predictions','COMPARATOR_PREDICTIONS.jsonl.gz'),
        ('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz')):
        C.bound(folder/name,scoring[key],'scored display '+key)
    for name in ('FIGURE_BINDINGS.json','FIGURE_CASE_ROWS.jsonl.gz'):
        C.output_path(args,name)
    figures = folder/'figures'
    C.require(not figures.exists(),'preserve previous figures')
    figures.mkdir()
    records = [display_row(row) for filename in ('PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz',
        'FIXED_PREDICTIONS.jsonl.gz') for row in C.rows(folder/filename)]
    data = {(r['method'],r['id']):r for r in records}
    ids = C.cohort_ids(args)
    C.require(len(data) == len(records) == 980 and
        all((method,fid) in data for fid in ids for method in SHOWN),'complete four-method980 display population')
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for axis,field,unit in zip(axes,('translation_cm','rotation_deg'),('cm','degrees')):
        for index,method in enumerate(SHOWN):
            values = np.array([data[method,fid]['pose'][field] for fid in ids if data[method,fid]['pose']['available']])
            axis.scatter(values,np.full(len(values),index),s=8,alpha=.24)
            if len(values):
                low,high = np.quantile(values,[.1,.9])
                axis.plot([low,high],[index,index],linewidth=3)
                axis.scatter([values.mean()],[index],marker='D',facecolor='white',edgecolor='black',zorder=5)
        axis.set_yticks(range(len(SHOWN)),SHOWN if axis is axes[0] else [])
        axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1)
        axis.set_xlabel(field+' ('+unit+')');axis.grid(alpha=.2)
    fig.suptitle('Fixed sparse ROLE local C2: Clean153 + Moderate92\nAll operational outputs including fallback; diamond=mean, segment=P10-P90; geometric proxy reference')
    fig.tight_layout();fig.savefig(figures/'01_all_operational.png',dpi=140);plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(11,12))
    for i,comparator in enumerate(('BASE','N3_SUBPIX',C.COMPARATOR)):
        for axis,field in zip(axes[i],('translation_cm','rotation_deg')):
            pairs = np.array([(data[comparator,fid]['pose'][field],data[C.PRIMARY,fid]['pose'][field]) for fid in ids
                if data[comparator,fid]['pose']['available'] and data[C.PRIMARY,fid]['pose']['available']],dtype=float).reshape(-1,2)
            upper = max(1.,pairs.max()) if len(pairs) else 1.
            if len(pairs):
                axis.scatter(pairs[:,0],pairs[:,1],s=12,alpha=.5)
            axis.plot([0,upper],[0,upper],'k--');axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
            axis.set_xlabel(comparator+' '+field);axis.set_ylabel('LOCAL_POINT_LINE\n'+field);axis.grid(alpha=.2)
    fig.suptitle('Same245 operational outputs: below diagonal favors LOCAL\nNumerical LOCAL NEW and fallback remain separate; known geometric proxy DEV')
    fig.tight_layout();fig.savefig(figures/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
    C.bound(CASE_PROTOCOL,C.read(args.protocol)['inputs']['visual_case_protocol'],'prespecified original visual cases')
    case_protocol = C.read(CASE_PROTOCOL)
    C.require(len(case_protocol['cases']) == 6,'same six prespecified cases required')
    case_ids = {x['id'] for x in case_protocol['cases']}
    C.bound(Path(args.parent)/'OBSERVATIONS.jsonl.gz',C.read(args.protocol)['inputs']['parent:OBSERVATIONS.jsonl.gz'],
        'stored original ROLE line observations')
    observations = {r['id']:r for r in C.rows(Path(args.parent)/'OBSERVATIONS.jsonl.gz')
        if r['head_arm']=='IMAGE_ROLE' and r['id'] in case_ids}
    C.require(set(observations) == case_ids and all(r['GT_input'] is False for r in observations.values()),
        'same six GT-free stored ROLE observations')
    case_rows = []
    for index,case in enumerate(case_protocol['cases']):
        path = Path(args.source_root)/case['image']['path']
        C.bound(path,case['image'],'unchanged illustrative RGB')
        with Image.open(path) as source:
            rgb = np.asarray(source.convert('RGB'))
        fig,axes = plt.subplots(2,2,figsize=(14,11))
        for axis,method in zip(axes.flat,SHOWN):
            row = data[method,case['id']]
            axis.imshow(rgb)
            output = np.asarray(row['native_points'],float)
            valid = np.isfinite(output[:8]).all(1) & (output[:8] != [-1,-1]).any(1)
            axis.scatter(output[:8][valid,0],output[:8][valid,1],s=32,facecolor='none',edgecolor='#00dce6',linewidth=1.4)
            for k in np.flatnonzero(valid):
                axis.text(output[k,0]+3,output[k,1]+3,str(k),color='cyan',fontsize=8)
            for k in row['reprojected_ids']:
                axis.scatter(*output[k],s=68,marker='s',facecolor='none',edgecolor='#ff8c24')
            if method in (C.COMPARATOR,C.PRIMARY):
                points = np.asarray(row['input_points'],float)
                active = np.isfinite(points[:8]).all(1) & (points[:8] != [-1,-1]).any(1)
                axis.scatter(points[:8][active,0],points[:8][active,1],s=35,marker='+',color='#42ff50')
            displayed_lines = []
            if method == C.PRIMARY:
                for edge in row['actual_unused_line_edges']:
                    candidates = [line for line in observations[case['id']]['lines'] if int(line['edge'])==edge]
                    C.require(candidates,'actual saved unused line has no original observation')
                    line = candidates[0]
                    support = np.asarray(line['support_points'],float)
                    C.require(support.ndim == 2 and support.shape[1] == 2 and len(support) >= 2 and
                        np.isfinite(support).all(),'finite actual stored line support required')
                    axis.plot(support[:,0],support[:,1],color='#fff340',linestyle='--',linewidth=1.5,marker='.',markersize=4)
                    axis.text(support[0,0]+4,support[0,1]-4,'e'+str(edge),color='#fff340',fontsize=8)
                    displayed_lines.append(dict(edge=edge,support_points=support.tolist(),support_queries=line['queries'],
                        accepted_fit=edge in row['accepted_fit_line_edges'],diagnostic_inlier=edge in row['diagnostic_line_inliers']))
            reference = row['evaluation_reference'];gt = np.asarray(reference.get('native_points_px',[]),float)
            if len(gt):
                known = reference.get('valid_native_ids',[])
                axis.scatter(gt[known,0],gt[known,1],s=19,marker='.',color='white',alpha=.65)
            T,R = row['pose'].get('translation_cm'),row['pose'].get('rotation_deg')
            title = (method+'\n'+row['output_status']+' / T='+str(round(T,4) if T is not None else None)+
                'cm / R='+str(round(R,4) if R is not None else None)+'deg\nfit points='+str(row['accepted_fit_ids'])+
                '; diagnostic points='+str(row['accepted_inlier_ids']))
            if method == C.PRIMARY:
                title += ('\nunused lines='+str(row['actual_unused_line_edges'])+'; fit lines='+str(row['accepted_fit_line_edges'])+
                    '; 8px support rank6='+str(row['diagnostic_inlier_support_rank6']))
            axis.set_title(title,fontsize=8);axis.axis('off')
            displayed = {k:v for k,v in row.items() if k not in ('native_points','input_points','evaluation_reference')}
            displayed['displayed_original_unused_line_support'] = displayed_lines
            case_rows.append(displayed)
        fig.suptitle(case['label']+' / '+case['id']+'\ncyan=output; green+=sparse points; yellow dashed=stored unused line support; orange square=H projection; white=proxy\n'
            'LOCAL uses sealed pose starts only, no residual prior/global uniqueness. Missing native coordinates are not fit inputs; 8px support is diagnostic.',fontsize=8)
        fig.tight_layout();fig.savefig(figures/('case_%02d.png'%(index+1)),dpi=140,bbox_inches='tight');plt.close(fig)
    C.require(len(case_rows)==24,'same six cases times four panels')
    target = C.output_path(args,'FIGURE_CASE_ROWS.jsonl.gz')
    with target.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
        for row in case_rows:
            zipped.write((json.dumps(C.finite(row),ensure_ascii=False,allow_nan=False)+'\n').encode())
    C.verify_protocol(args);C.protect(args)
    C.require(len(list(figures.glob('*.png')))==8,'eight actual PNG files')
    C.write_new(C.output_path(args,'FIGURE_BINDINGS.json'),dict(complete=True,
        figures=[C.binding(path) for path in sorted(figures.glob('*.png'))],fixed_case_protocol=C.binding(CASE_PROTOCOL),
        case_rows=C.binding(target),renderer_code=C.binding(__file__),scored_methods=C.binding(folder/'PREDICTIONS.jsonl.gz'),
        scored_comparator=C.binding(folder/'COMPARATOR_PREDICTIONS.jsonl.gz'),fixed_controls=C.binding(folder/'FIXED_PREDICTIONS.jsonl.gz'),
        stored_ROLE_observations=C.binding(Path(args.parent)/'OBSERVATIONS.jsonl.gz'),
        scoring_receipt=C.binding(folder/'SCORING_RECEIPT.json'),protocol=C.binding(args.protocol),
        panels=24,PNG_files=8,selection_uses_new_scores=False,only_display_fields_retained_in_memory=True,
        original_raw_witnesses_unchanged=True,numeric_LOCAL_NEW_is_not_accuracy_success=True,
        local_initialization_is_not_residual_prior=True,line_zero_point_LOCAL_is_not_line_effect=True,
        new_detector_head_PnP_local_optimizer_GT_score_training_RGB_calls=0,physical_truth_certified=False))
    print('SPARSE_LOCAL_FIGURES',8,flush=True)


def run(args):
    C.verify_protocol(args);C.protect(args)
    for name in ('FIGURES_STARTED.json','FIGURES_FAILURE.json'):
        C.output_path(args,name)
    folder = Path(args.output)
    C.write_new(C.output_path(args,'FIGURES_STARTED.json'),dict(protocol=C.binding(args.protocol),renderer_code=C.binding(__file__),
        scoring_receipt=C.binding(folder/'SCORING_RECEIPT.json'),configured_PNG_files=8,configured_case_panels=24,automatic_retry=False))
    try:
        _render(args)
    except BaseException as error:
        figures,case_rows = folder/'figures',folder/'FIGURE_CASE_ROWS.jsonl.gz'
        C.write_new(C.output_path(args,'FIGURES_FAILURE.json'),dict(complete=False,
            error=dict(type=type(error).__name__,message=str(error)),
            preserved_PNG_prefix=[C.binding(p) for p in sorted(figures.glob('*.png'))] if figures.is_dir() else [],
            preserved_case_rows=C.binding(case_rows) if case_rows.is_file() else None,
            new_detector_head_PnP_local_optimizer_GT_score_training_RGB_calls=0,automatic_retry=False))
        raise


if __name__ == '__main__':
    run(C.parser(__doc__,stages=None).parse_args())
