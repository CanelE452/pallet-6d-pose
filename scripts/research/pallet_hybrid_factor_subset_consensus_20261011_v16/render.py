"""Six prespecified Base/N3/hybrid cases and all six saved operating methods.

Plots read saved metrics and poses. No model, solver, GT scoring, bootstrap or
case selection runs here. The same-q point-only arm remains a line ablation.
"""
from pathlib import Path
import gzip
import json
import time
import traceback

from . import runner as R
from . import evaluation as E
from . import pipeline_ready as P
from ..pallet_sparse_local_line_20261010_v9 import common as C

CASE_PROTOCOL=C.REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'
SOURCE_RENDER=C.REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/render.py'
METHODS=E.METHODS
LABELS=('Base','N3→cornerSubPix','V7 hybrid point robust','V15 boundary-only subset',
    'Hybrid point + finite edge subset','Same hybrid point-only subset')
SHOWN=(METHODS[0],METHODS[1],METHODS[4])
SHOWN_LABELS=(LABELS[0],LABELS[1],LABELS[4])
NAMES=('FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURES_FAILURE.json','FIGURE_CASE_ROWS.jsonl.gz')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def display_row(row):
    solver=row.get('solver') or {};new=row['new_pose_estimated'];best=solver.get('best_candidate') or {}
    return dict(id=row['id'],method=row['method'],native_points=row['native_points'],pose=row['pose'],
        output_status=row['output_status'],new_pose_estimated=new,input_points=row.get('input_points'),
        reprojected_ids=row.get('reprojected_ids',[]),evaluation_reference=row.get('evaluation_reference',{}),
        actual_used_U=solver.get('used',[]),applied_H=row.get('hidden_initial',[]),fallback_used=row['fallback_used'],
        pose_available=row['pose_available'],accepted_fit_ids=solver.get('fit_input_ids',[]) if new else [],
        accepted_inlier_ids=solver.get('final_inliers',[]) if new else [],
        actual_unused_line_edges=solver.get('line_edges',[]),accepted_fit_line_edges=solver.get('fit_line_edges',[]) if new else [],
        diagnostic_line_inliers=solver.get('final_line_inliers',[]),solver_state=solver.get('state'),
        accepted_projected=solver.get('projected') if new else None,
        rejected_endpoint_edges=(row.get('observation_filter') or {}).get('rejected_edge_ids',[]),
        saved_best_candidate_support=best.get('whole_pool_score'),selected_proposal_kind=best.get('proposal_kind'),
        stable_guard_state=(best.get('stable_incidence_guard') or {}).get('guard_state'),
        selected_boundary_ids=row.get('selected_corner_ids',[]),parent_method=row.get('parent_method'),
        physical_ownership_truth='UNKNOWN',numerical_NEW_is_not_accuracy_success=True,global_uniqueness_proven=False)

def inputs(args):
    _,receipt=E.score_gate(args)
    cases=C.read(CASE_PROTOCOL)['cases'];cohort=C.read(args.cohort);ids=cohort['ids'];by_id={f['id']:f for f in cohort['frames']}
    C.bound(args.cohort,C.read(CASE_PROTOCOL)['cohort'],'original fixed case cohort')
    C.require(len(ids)==len(by_id)==245 and len(cases)==len({c['id'] for c in cases})==6
        and sum(c['label']=='clean' for c in cases)==sum(c['label']=='moderate' for c in cases)==3,'prespecified3clean+3moderate/full245')
    paths=dict(renderer=Path(__file__),source_renderer=SOURCE_RENDER,case_protocol=CASE_PROTOCOL,cohort=Path(args.cohort),
        new_scores=R.DOC/'PREDICTIONS.jsonl.gz',hybrid_scores=Path(args.parent)/'PREDICTIONS.jsonl.gz',
        boundary_scores=E.E15.R.DOC/'PREDICTIONS.jsonl.gz',fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',
        hybrid_scoring=Path(args.parent)/'SCORING_RECEIPT.json',boundary_scoring=E.E15.R.DOC/'SCORING_RECEIPT.json',
        fixed_scoring=Path(args.v9_input)/'SCORING_RECEIPT.json',metrics=R.DOC/'METRICS.json',
        observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',scoring_receipt=R.DOC/'SCORING_RECEIPT.json',
        geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry_validation=R.DOC/'VALIDATION_CHECKS.json',
        validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        geometry_checker=R.CODE/'validation_checks.py',evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',
        scoring_started=R.DOC/'SCORING_STARTED.json',scoring_authority=R.DOC/'SCORING_AUTHORITY.json',
        statistics_started=R.DOC/'STATISTICS_STARTED.json',verification=R.DOC/'VERIFICATION.json',
        verification_protocol=R.DOC/'STATISTICS_VERIFICATION_PROTOCOL.json',verification_started=R.DOC/'STATISTICS_VERIFICATION_STARTED.json',
        verification_checker=R.CODE/'verify.py',authority=Path(E.__file__),protocol=Path(args.protocol),
        original_parent_figure_bindings=E.E15.R.DOC/'FIGURE_BINDINGS.json',original_parent_visual_review=E.E15.R.DOC/'ROOT_VISUAL_REVIEW.json')
    C.bound(paths['new_scores'],receipt['predictions'],'paired490 display scores')
    for authority,stream,field,method,expected in (('hybrid_scoring','hybrid_scores','predictions',None,1960),
        ('boundary_scoring','boundary_scores','predictions',METHODS[3],245),('fixed_scoring','fixed_scores','fixed_predictions',None,490)):
        stored=C.read(paths[authority]);C.require(stored['complete'] is True,'complete immutable comparator scoring '+authority)
        if authority=='hybrid_scoring':C.require(stored['scored_method_rows']==expected and stored['methods']==list(E.C7.METHODS),'all8 original V7 score population')
        if method is not None:C.require(stored['scored_rows']==expected and stored['methods']==[method],'immutable boundary comparator')
        C.bound(paths[stream],stored[field],'unchanged display '+stream)
    metrics=C.read(paths['metrics']);v=C.read(paths['verification']);own=C.read(paths['verification_protocol']);started=C.read(paths['verification_started'])
    C.require(metrics['schema']=='hybrid_factor_subset_raw_row_metrics_v16' and metrics['complete'] is True
        and metrics['methods']==list(METHODS) and metrics['actual_arithmetic']==E.ARITHMETIC,'all6/1470/972/45/135 saved statistics')
    C.require(v['schema']=='independent_hybrid_factor_subset_statistics_checks_v16' and v['complete'] is True and v['passed'] is True
        and v['failure_count']==0 and v['inputs']==own['inputs']==started['inputs']
        and v['actual_counts']['saved_scored_rows_streamed']==1470 and v['actual_counts']['moment_scalars_checked']==972
        and v['actual_counts']['pair_groups_checked']==45 and v['actual_counts']['metric_CI_slots_checked']==135,
        'actual independent six-method saved statistics PASS')
    C.require(own['schema']=='independent_hybrid_factor_subset_statistics_protocol_v16','own saved statistics protocol')
    C.bound(paths['verification_protocol'],v['protocol'],'actual independent statistics protocol')
    C.bound(paths['verification_protocol'],started['protocol'],'actual independent STARTED')
    C.bound(paths['verification_checker'],v['checker'],'actual current verifier')
    for key in ('metrics','new_scores','hybrid_scores','boundary_scores','fixed_scores','geometry_seal'):
        C.bound(paths[key],v['inputs']['METRICS' if key=='metrics' else key],'independently checked display '+key)
    parent_fig=C.read(paths['original_parent_figure_bindings']);parent_visual=C.read(paths['original_parent_visual_review'])
    C.require(parent_fig['complete'] is True and parent_fig['PNG_files']==8 and parent_fig['panels']==24
        and parent_visual['passed'] is True and parent_visual['actual_image_tool_views']==8,'original parent fixed-case visual authority')
    for i,c in enumerate(cases):
        C.require(c['id'] in by_id and all(c[k]==by_id[c['id']][k] for k in ('session','label','image')),'unchanged case identity')
        image=Path(args.source_root)/c['image']['path'];C.bound(image,c['image'],'original RGB');paths['image:'+str(i)]=image
    return paths,{k:C.binding(p) for k,p in paths.items()}

def freeze(args):
    R.output(args,NAMES[0]);_,b=inputs(args)
    C.write_new(R.output(args,NAMES[0]),dict(schema='prespecified_six_case_hybrid_display_v16',inputs=b,
        methods=list(METHODS),labels=list(LABELS),display_methods=list(SHOWN),cases=6,panels=18,PNG_files=8,
        frames=245,scored_rows=1470,paired_plot_comparators=['N3_SUBPIX',P.METHODS[1]],
        selection_uses_new_scores=False,new_model_fit_GT_score_training_RGB_calls=0,
        new_statistic_or_CI_arithmetic_calls=0,saved_aggregate_markers_only=True,
        physical_reference_is_GEOMETRIC_PROXY=True,source_RGB_is_existing_only=True))

def run(args):
    lock=C.read(R.DOC/NAMES[0]);paths,b=inputs(args)
    C.require(lock['schema']=='prespecified_six_case_hybrid_display_v16' and lock['inputs']==b
        and lock['methods']==list(METHODS) and lock['display_methods']==list(SHOWN)
        and lock['frames']==245 and lock['scored_rows']==1470 and lock['panels']==18 and lock['PNG_files']==8,
        'own figure freeze differs')
    for n in NAMES[1:]:R.output(args,n)
    C.require(not (R.DOC/'figures').exists(),'preserve old figures')
    C.write_new(R.output(args,NAMES[1]),dict(protocol=C.binding(R.DOC/NAMES[0]),configured_PNGs=8,configured_panels=18,scored_rows=1470))
    folder=R.DOC/'figures';began=time.monotonic();case_rows=[]
    try:
        folder.mkdir()
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        data={};parent_seen={m:set() for m in E.C7.METHODS};ids=C.read(args.cohort)['ids'];known=set(ids)
        for key in ('new_scores','hybrid_scores','boundary_scores','fixed_scores'):
            for row in C.rows(paths[key]):
                if key=='hybrid_scores':
                    C.require(row['method'] in parent_seen and row['id'] in known and row['id'] not in parent_seen[row['method']],'original8method population')
                    parent_seen[row['method']].add(row['id'])
                    if row['method']!=P.PARENT_METHOD:continue
                identity=(row['method'],row['id']);C.require(row['method'] in METHODS and row['id'] in known and identity not in data,'unique same-population display')
                data[identity]=display_row(row)
        C.require(len(data)==1470 and all((m,fid) in data for fid in ids for m in METHODS) and all(v==known for v in parent_seen.values()),'all1470 and original1960 retained')
        cases=C.read(CASE_PROTOCOL)['cases'];case_ids={c['id'] for c in cases};metrics=C.read(paths['metrics'])
        obs={r['id']:r for r in C.rows(paths['observations']) if r['head_arm']=='IMAGE_ROLE' and r['id'] in case_ids}
        C.require(set(obs)==case_ids,'original sameROLE case observations')
        fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for i,m in enumerate(METHODS):
                values=[data[m,f]['pose'][field] for f in ids if data[m,f]['pose']['available']]
                axis.scatter(values,[i]*len(values),s=7,alpha=.22)
                saved=metrics['strata']['combined']['methods'][m]['metrics']['operational'][field]
                if saved['n']:
                    axis.plot([saved['median'],saved['P90']],[i,i],linewidth=2)
                    axis.scatter([saved['mean']],[i],marker='D',facecolor='white',edgecolor='black')
            labels=[label+' (n='+str(metrics['strata']['combined']['methods'][m]['operational'])+'/245)' for m,label in zip(METHODS,LABELS)]
            axis.set_yticks(range(len(METHODS)),labels);axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1);axis.grid(alpha=.2);axis.set_xlabel(field)
        fig.suptitle('All245 operating outputs including fallback, all six fixed methods\nSaved statistics: diamond=mean; segment=median–P90. Proxy DEV; unavailable counts stay explicit.')
        fig.savefig(folder/'01_all_operational.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(14,11),layout='constrained')
        for ri,control in enumerate(('N3_SUBPIX',P.METHODS[1])):
            for axis,field in zip(axes[ri],('translation_cm','rotation_deg')):
                for new,label in ((True,'Numerical NEW'),(False,'Fallback')):
                    pairs=[(data[control,f]['pose'][field],data[P.PRIMARY,f]['pose'][field]) for f in ids
                        if data[P.PRIMARY,f]['new_pose_estimated']==new and data[P.PRIMARY,f]['pose']['available'] and data[control,f]['pose']['available']]
                    if pairs:x,y=zip(*pairs);axis.scatter(x,y,s=11,alpha=.4,label=label+' n='+str(len(pairs)))
                values=[data[m,f]['pose'][field] for m in (P.PRIMARY,control) for f in ids if data[m,f]['pose']['available']]
                upper=max([1.,*values]);axis.plot([0,upper],[0,upper],'k--');axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
                axis.set_xlabel(control+' '+field);axis.set_ylabel('Hybrid point + finite edge '+field);axis.legend(fontsize=8);axis.grid(alpha=.2)
        fig.suptitle('Same-frame operational comparisons: N3 and the same-q point-only line ablation\nBelow diagonal favors primary. NEW and fallback separate; no case or policy selection from these plots.')
        fig.savefig(folder/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
        for ci,c in enumerate(cases):
            with Image.open(paths['image:'+str(ci)]) as source:rgb=np.asarray(source.convert('RGB'))
            fig,axes=plt.subplots(1,3,figsize=(22,9));fig.subplots_adjust(wspace=.04,top=.68)
            for axis,m,label in zip(axes,SHOWN,SHOWN_LABELS):
                r=data[m,c['id']];axis.imshow(rgb);p=np.asarray(r['native_points'],float)
                valid=np.isfinite(p[:8]).all(1)&(p[:8]!=[-1,-1]).any(1)
                axis.scatter(p[:8][valid,0],p[:8][valid,1],s=28,facecolor='none',edgecolor='cyan')
                for k in np.flatnonzero(valid):axis.text(p[k,0]+3,p[k,1]+3,str(k),fontsize=8,color='cyan')
                for k in r['reprojected_ids']:axis.scatter(*p[k],s=65,marker='s',facecolor='none',edgecolor='orange')
                if r['input_points'] is not None:
                    q=np.asarray(r['input_points'],float);v=np.isfinite(q[:8]).all(1)&(q[:8]!=[-1,-1]).any(1)
                    axis.scatter(q[:8][v,0],q[:8][v,1],s=28,marker='+',color='#43ff50')
                displayed=[]
                for edge in r['actual_unused_line_edges']:
                    line=next(x for x in obs[c['id']]['lines'] if x['edge']==edge);support=np.asarray(line['support_points'],float);fitted=edge in r['accepted_fit_line_edges']
                    axis.plot(support[:,0],support[:,1],'-' if fitted else '--',color='yellow',linewidth=1.5 if fitted else .8)
                    axis.text(*support[0],str(edge),color='yellow',fontsize=8);segment=None
                    if fitted and r['accepted_projected'] is not None:
                        edge_a,edge_b=EDGES[edge];projected=np.asarray(r['accepted_projected'],float);segment=projected[[edge_a,edge_b]].tolist()
                        axis.plot(projected[[edge_a,edge_b],0],projected[[edge_a,edge_b],1],color='#f55bff',linewidth=1.2)
                    displayed.append(dict(edge=edge,support_points=support.tolist(),source_queries=line['queries'],query_radii_px=line['query_radii_px'],
                        accepted_fit=fitted,accepted_projected_segment=segment,physical_ownership_truth='UNKNOWN'))
                ref=r['evaluation_reference'];g=np.asarray(ref.get('native_points_px',[]),float);valid_ref=ref.get('valid_native_ids',[])
                if len(g):axis.scatter(g[valid_ref,0],g[valid_ref,1],s=14,marker='.',color='white',alpha=.7)
                T,rot=r['pose'].get('translation_cm'),r['pose'].get('rotation_deg')
                text=label+'\n'+r['output_status']+' / '+str(r['solver_state'])+'\nT='+str(round(T,4) if T is not None else None)+'cm; R='+str(round(rot,4) if rot is not None else None)+'deg'
                text+='\nfit points='+str(r['accepted_fit_ids'])+'; fit edges='+str(r['accepted_fit_line_edges'])+'; H='+str(r['applied_H'])
                text+='\nadopted boundary='+str(r['selected_boundary_ids'])+'; proposal='+str(r['selected_proposal_kind'])
                axis.set_title(text,fontsize=9,pad=8);axis.axis('off');case_rows.append(dict(case_index=ci,id=c['id'],method=m,display=r,displayed_lines=displayed))
            fig.suptitle(c['label']+' / '+c['id']+'\ncyan=output; green+=same selected hybrid RGB observations; yellow solid=fit / dashed=input-only; magenta=accepted finite edge\norange=NEW H reprojection; white=proxy. NEW experimental fit excludes initial H; fallback retains original all8 N3.\nActual edge ownership remains UNKNOWN. Same fixed cases, no new images or selected best examples.',fontsize=10)
            fig.savefig(folder/('case_%02d.png'%(ci+1)),dpi=140);plt.close(fig)
        C.require(len(case_rows)==18 and len(list(folder.glob('*.png')))==8,'all fixed figures complete')
        with R.output(args,NAMES[4]).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
            for row in case_rows:z.write((json.dumps(C.finite(row),ensure_ascii=False,allow_nan=False)+'\n').encode())
        C.require(inputs(args)[1]==b,'display input bytes changed')
        C.write_new(R.output(args,NAMES[2]),dict(complete=True,protocol=C.binding(R.DOC/NAMES[0]),inputs=b,panels=18,PNG_files=8,
            display_records=1470,display_methods=list(SHOWN),figures=[C.binding(p) for p in sorted(folder.glob('*.png'))],case_rows=C.binding(R.DOC/NAMES[4]),
            elapsed_seconds=time.monotonic()-began,new_model_fit_GT_score_training_RGB_calls=0,new_statistic_or_CI_arithmetic_calls=0,selection_uses_new_scores=False))
    except BaseException as exc:
        C.write_new(R.output(args,NAMES[3]),dict(complete=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),
            preserved_PNG_prefix=[C.binding(p) for p in sorted(folder.glob('*.png'))],completed_case_panels=len(case_rows),automatic_retry=False));raise

def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','run');args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
