"""Show six original cases and all 245 saved operating results; no new fits.

The four case panels are Base/N3/point robust/finite-edge consensus. Whole-frame
plots also preserve original LOCAL. Cases predate this repair and its scores.
"""
from pathlib import Path
import gzip
import json
import time
import traceback

from . import runner as R
from . import evaluation_review as E
from ..pallet_sparse_local_line_20261010_v9 import common as C

CASE_PROTOCOL=C.REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'
SOURCE_RENDER=C.REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/render.py'
METHODS=('BASE','N3_SUBPIX',C.COMPARATOR,C.PRIMARY,'FINITE_EDGE_CONSENSUS')
LABELS=('Base','N3→cornerSubPix','Sparse point robust','Original LOCAL','Finite edge consensus')
SHOWN=(METHODS[0],METHODS[1],METHODS[2],METHODS[4])
SHOWN_LABELS=(LABELS[0],LABELS[1],LABELS[2],LABELS[4])
NAMES=('FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURES_FAILURE.json','FIGURE_CASE_ROWS.jsonl.gz')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def display_row(row):
    """Copy saved display fields only; acceptance never comes from a picture."""
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
        factor_pool=solver.get('factor_pool'),accepted_point_line_geometry=solver.get('point_line_geometry') if new else None,
        saved_best_candidate_support=best.get('whole_pool_score'),saved_candidate_count=solver.get('candidate_count',0),
        numerical_NEW_is_not_accuracy_success=True,global_uniqueness_proven=False)

def inputs(args):
    receipt=C.read(R.DOC/'SCORING_RECEIPT.json');C.require(receipt['complete'] is True and receipt['scored_rows']==245,'complete scoring required')
    C.bound(R.DOC/'PREDICTIONS.jsonl.gz',receipt['predictions'],'new display scores')
    case=C.read(CASE_PROTOCOL);cohort=C.read(args.cohort);cases=case['cases'];ids=cohort['ids']
    C.bound(args.cohort,case['cohort'],'original six-case cohort')
    by_id={f['id']:f for f in cohort['frames']}
    C.require(len(ids)==len(by_id)==245 and len(cases)==6 and len({c['id'] for c in cases})==6,'original6/full245 required')
    C.require(sum(c['label']=='clean' for c in cases)==sum(c['label']=='moderate' for c in cases)==3,'fixed3clean+3moderate cases')
    paths=dict(renderer=Path(__file__),source_renderer=SOURCE_RENDER,case_protocol=CASE_PROTOCOL,
        new_scores=R.DOC/'PREDICTIONS.jsonl.gz',old_local=Path(args.v9_input)/'PREDICTIONS.jsonl.gz',
        point=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',fixed=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',
        metrics=R.DOC/'METRICS.json',observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
        scoring_receipt=R.DOC/'SCORING_RECEIPT.json',geometry_seal=R.DOC/'GEOMETRY_SEAL.json',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',
        original_validation_started=R.DOC/'VALIDATION_STARTED.json',
        review_validation=R.DOC/'REVIEW_VALIDATION_CHECKS.json',review_protocol=R.DOC/'REVIEW_VALIDATION_PROTOCOL.json',
        review_started=R.DOC/'REVIEW_VALIDATION_STARTED.json',review_checker=R.CODE/'validation_review.py',
        scoring_authority=R.DOC/'SCORING_AUTHORITY_SUPPLEMENT.json',
        statistics_review=R.CODE/'statistics_review.py',statistics_authority=R.DOC/'STATISTICS_AUTHORITY_SUPPLEMENT.json',
        original_statistics_started=R.DOC/'STATISTICS_STARTED.json',original_statistics_failure=R.DOC/'STATISTICS_FIRST_FAILURE.json',
        review_statistics_started=R.DOC/'REVIEW_STATISTICS_STARTED.json',
        authority=Path(E.__file__),protocol=Path(args.protocol),cohort=Path(args.cohort),
        old_scoring_receipt=Path(args.v9_input)/'SCORING_RECEIPT.json')
    old=C.read(paths['old_scoring_receipt']);C.require(old['complete'] is True,'preserved V9 scores required')
    for alias,key in (('old_local','predictions'),('point','comparator_predictions'),('fixed','fixed_predictions')):
        C.bound(paths[alias],old[key],'unchanged display '+alias)
    metrics=C.read(paths['metrics']);C.require(metrics['complete'] is True and metrics['methods']==list(METHODS),'full five-method statistics required')
    C.require(metrics['actual_arithmetic']['scored_rows']==1225,'full1225 scored display population required')
    for i,c in enumerate(cases):
        C.require(c['id'] in by_id and all(c[k]==by_id[c['id']][k] for k in ('session','label','image')),'original case metadata differs')
        image=Path(args.source_root)/c['image']['path'];C.bound(image,c['image'],'original RGB');paths['image:'+str(i)]=image
    return paths,{k:C.binding(p) for k,p in paths.items()}

def freeze(args):
    E.geometry_gate(args);R.output(args,NAMES[0]);_,bindings=inputs(args)
    C.write_new(R.output(args,NAMES[0]),dict(schema='prespecified_six_case_display_v13',inputs=bindings,
        methods=list(METHODS),labels=list(LABELS),display_methods=list(SHOWN),display_labels=list(SHOWN_LABELS),
        cases=6,panels=24,PNG_files=8,frames=245,scored_rows=1225,selection_uses_new_scores=False,
        new_model_fit_GT_score_training_RGB_calls=0,frozen_after_complete_scores_before_render=True,
        physical_reference_is_GEOMETRIC_PROXY=True,source_RGB_is_existing_only=True))
    print('FINITE_EDGE_FIGURES_FROZEN',flush=True)

def run(args):
    E.geometry_gate(args);lock=C.read(R.DOC/NAMES[0]);paths,b=inputs(args)
    C.require(lock['inputs']==b and lock['methods']==list(METHODS) and lock['display_methods']==list(SHOWN),'own figure freeze differs')
    for n in NAMES[1:]:R.output(args,n)
    C.require(not (R.DOC/'figures').exists(),'preserve old figures')
    C.write_new(R.output(args,NAMES[1]),dict(protocol=C.binding(R.DOC/NAMES[0]),configured_PNGs=8,configured_panels=24,scored_rows=1225))
    folder=R.DOC/'figures';began=time.monotonic();case_rows=[]
    try:
        folder.mkdir()
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        data={}
        for key in ('new_scores','old_local','point','fixed'):
            for row in C.rows(paths[key]):
                identity=(row['method'],row['id']);C.require(row['method'] in METHODS and identity not in data,'unknown/duplicate display row')
                data[identity]=display_row(row)
        ids=C.read(args.cohort)['ids'];C.require(len(data)==1225 and all((m,fid) in data for fid in ids for m in METHODS),'all1225 display records required')
        cases=C.read(CASE_PROTOCOL)['cases'];case_ids={c['id'] for c in cases}
        obs={r['id']:r for r in C.rows(paths['observations']) if r['head_arm']=='IMAGE_ROLE' and r['id'] in case_ids}
        C.require(set(obs)==case_ids,'original case observation join')
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for i,m in enumerate(METHODS):
                a=np.asarray([data[m,f]['pose'][field] for f in ids if data[m,f]['pose']['available']])
                axis.scatter(a,np.full(len(a),i),s=7,alpha=.25)
                if len(a):axis.plot(np.quantile(a,[.1,.9]),[i,i],linewidth=3);axis.scatter([a.mean()],[i],marker='D',facecolor='white',edgecolor='black')
            labels=[label+' (n='+str(sum(data[m,f]['pose']['available'] for f in ids))+'/245)' for m,label in zip(METHODS,LABELS)]
            axis.set_yticks(range(len(METHODS)),labels);axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1);axis.grid(alpha=.2);axis.set_xlabel(field)
        fig.suptitle('All245 operating outputs, including fallbacks; unavailable counts remain explicit\nGeometric proxy DEV; diamond=mean, segment=P10–P90')
        fig.savefig(folder/'01_all_operational.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for new,label in ((True,'Numerical NEW'),(False,'Fallback output')):
                pairs=np.asarray([(data['N3_SUBPIX',f]['pose'][field],data[METHODS[-1],f]['pose'][field]) for f in ids
                    if data[METHODS[-1],f]['new_pose_estimated']==new and data[METHODS[-1],f]['pose']['available'] and data['N3_SUBPIX',f]['pose']['available']])
                if len(pairs):axis.scatter(pairs[:,0],pairs[:,1],s=12,alpha=.45,label=label+' n='+str(len(pairs)))
            values=[data[m,f]['pose'][field] for m in (METHODS[-1],'N3_SUBPIX') for f in ids if data[m,f]['pose']['available']]
            upper=max([1.,*values]);axis.plot([0,upper],[0,upper],'k--')
            axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
            axis.set_xlabel('N3 '+field);axis.set_ylabel('Finite consensus '+field);axis.legend(fontsize=8);axis.grid(alpha=.2)
        fig.suptitle('Same245 population; below diagonal favors repair; NEW and fallback are separate\nGeometric proxy reference; numerical availability is not pose accuracy success')
        fig.savefig(folder/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
        for ci,c in enumerate(cases):
            with Image.open(paths['image:'+str(ci)]) as source:rgb=np.asarray(source.convert('RGB'))
            fig,axes=plt.subplots(2,2,figsize=(16,12));fig.subplots_adjust(hspace=.31,wspace=.04,top=.84)
            for axis,m,label in zip(axes.flat,SHOWN,SHOWN_LABELS):
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
                    line=next(x for x in obs[c['id']]['lines'] if x['edge']==edge);support=np.asarray(line['support_points'],float)
                    fitted=edge in r['accepted_fit_line_edges']
                    axis.plot(support[:,0],support[:,1],'-' if fitted else '--',color='yellow',linewidth=1.5 if fitted else .8)
                    axis.text(*support[0],str(edge),color='yellow',fontsize=8)
                    segment=None
                    if fitted and r['accepted_projected'] is not None:
                        a,b=EDGES[edge];projected=np.asarray(r['accepted_projected'],float);segment=projected[[a,b]].tolist()
                        axis.plot(projected[[a,b],0],projected[[a,b],1],color='#f55bff',linewidth=1.2)
                    displayed.append(dict(edge=edge,support_points=support.tolist(),source_queries=line['queries'],
                        query_radii_px=line['query_radii_px'],accepted_fit=fitted,accepted_projected_segment=segment,
                        physical_ownership_truth='UNKNOWN'))
                ref=r['evaluation_reference'];g=np.asarray(ref.get('native_points_px',[]),float);known=ref.get('valid_native_ids',[])
                if len(g):axis.scatter(g[known,0],g[known,1],s=14,marker='.',color='white',alpha=.7)
                T,rot=r['pose'].get('translation_cm'),r['pose'].get('rotation_deg')
                text=label+'\n'+r['output_status']+' / '+str(r['solver_state'])+'\nT='+str(round(T,4) if T is not None else None)+'cm; R='+str(round(rot,4) if rot is not None else None)+'deg'
                text+='\nfit points='+str(r['accepted_fit_ids'])+'; fit edges='+str(r['accepted_fit_line_edges'])+'; H='+str(r['applied_H'])
                axis.set_title(text,fontsize=9,pad=8);axis.axis('off')
                case_rows.append(dict(case_index=ci,id=c['id'],method=m,display=r,displayed_lines=displayed))
            fig.suptitle(c['label']+' / '+c['id']+'\ncyan=output; green+=supplied points; yellow solid=fit support / dashed=input-only; magenta=accepted finite edge\norange=H reprojection; white=proxy. H initial xy never fit; finite support is not an ownership certificate.',fontsize=10)
            fig.savefig(folder/('case_%02d.png'%(ci+1)),dpi=140);plt.close(fig)
        C.require(len(case_rows)==24 and len(list(folder.glob('*.png')))==8,'all fixed figures complete')
        with R.output(args,NAMES[4]).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
            for row in case_rows:z.write((json.dumps(C.finite(row),ensure_ascii=False,allow_nan=False)+'\n').encode())
        C.require(inputs(args)[1]==b,'display inputs changed')
        C.write_new(R.output(args,NAMES[2]),dict(complete=True,protocol=C.binding(R.DOC/NAMES[0]),inputs=b,
            panels=24,PNG_files=8,display_records=1225,display_methods=list(SHOWN),
            figures=[C.binding(p) for p in sorted(folder.glob('*.png'))],case_rows=C.binding(R.DOC/NAMES[4]),
            elapsed_seconds=time.monotonic()-began,new_model_fit_GT_score_training_RGB_calls=0,selection_uses_new_scores=False))
        print('FINITE_EDGE_FIGURES_COMPLETE',8,24,flush=True)
    except BaseException as exc:
        C.write_new(R.output(args,NAMES[3]),dict(complete=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),
            preserved_PNG_prefix=[C.binding(p) for p in sorted(folder.glob('*.png'))],completed_case_panels=len(case_rows),automatic_retry=False));raise

def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','run');args=p.parse_args()
    (freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
