"""Display the original six prespecified cases and all 245 saved metrics.

No new model, pose fit or GT score. All six methods are drawn together;
fallbacks and rejected candidates remain labelled. Selection predates V11.
"""
from pathlib import Path
import gzip
import json
import time
import traceback

from . import runner as R
from . import evaluation_review as E
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9.render import display_row,CASE_PROTOCOL

METHODS=('BASE','N3_SUBPIX',C.COMPARATOR,C.PRIMARY,'FILTERED_LOCAL','FILTERED_LOCAL_WEAK_SAFE')
LABELS=('Base','N3→cornerSubPix','Sparse point robust','Original LOCAL','Filtered LOCAL','Filtered LOCAL + weak gate')
NAMES=('FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURES_FAILURE.json','FIGURE_CASE_ROWS.jsonl.gz')


def inputs(args):
    receipt=C.read(R.DOC/'SCORING_RECEIPT.json');C.require(receipt['complete'] and receipt['scored_rows']==490,'complete scoring required')
    C.bound(R.DOC/'PREDICTIONS.jsonl.gz',receipt['predictions'],'new display scores')
    cases=C.read(CASE_PROTOCOL)['cases'];ids=C.read(args.cohort)['ids']
    C.require(len(cases)==6 and all(c['id'] in ids for c in cases),'original six cases within245 required')
    paths=dict(renderer=Path(__file__),case_protocol=CASE_PROTOCOL,new_scores=R.DOC/'PREDICTIONS.jsonl.gz',
        old_local=Path(args.v9_input)/'PREDICTIONS.jsonl.gz',point=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',
        fixed=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',metrics=R.DOC/'METRICS.json',
        observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',scoring_receipt=R.DOC/'SCORING_RECEIPT.json',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',review_validation=R.DOC/'REVIEW_VALIDATION_CHECKS.json',
        review_protocol=R.DOC/'REVIEW_VALIDATION_PROTOCOL.json',authority=Path(E.__file__),
        protocol=Path(args.protocol),cohort=Path(args.cohort))
    paths['old_scoring_receipt']=Path(args.v9_input)/'SCORING_RECEIPT.json'
    old=C.read(paths['old_scoring_receipt']);C.require(old['complete'] is True,'preserved V9 scores required')
    for alias,key in (('old_local','predictions'),('point','comparator_predictions'),('fixed','fixed_predictions')):
        C.bound(paths[alias],old[key],'unchanged scored display '+alias)
    for i,c in enumerate(cases):
        p=Path(args.source_root)/c['image']['path'];C.bound(p,c['image'],'original RGB');paths['image:'+str(i)]=p
    return paths,{k:C.binding(p) for k,p in paths.items()}


def freeze(args):
    E.geometry_gate(args);R.output(args,NAMES[0]);paths,bindings=inputs(args)
    C.write_new(R.output(args,NAMES[0]),dict(schema='prespecified_six_case_display_v11',inputs=bindings,
        methods=list(METHODS),labels=list(LABELS),cases=6,panels=36,PNG_files=8,frames=245,
        selection_uses_new_scores=False,new_model_fit_GT_score_training_RGB_calls=0,
        frozen_after_complete_scores_before_render=True,physical_reference_is_GEOMETRIC_PROXY=True))
    print('REPAIR_FIGURES_FROZEN',flush=True)


def run(args):
    E.geometry_gate(args);lock=C.read(R.DOC/NAMES[0]);paths,b=inputs(args)
    C.require(lock['inputs']==b and lock['methods']==list(METHODS),'own figure freeze differs')
    for n in NAMES[1:]:R.output(args,n)
    folder=R.DOC/'figures';C.require(not folder.exists(),'preserve old figures');folder.mkdir()
    C.write_new(R.output(args,NAMES[1]),dict(protocol=C.binding(R.DOC/NAMES[0]),configured_PNGs=8,configured_panels=36))
    began=time.monotonic();case_rows=[]
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        data={}
        for key in ('new_scores','old_local','point','fixed'):
            for row in C.rows(paths[key]):
                item=display_row(row);item['solver_state']=(row.get('solver') or {}).get('state')
                if row['method'] in METHODS[3:]:
                    item['numeric_LOCAL_NEW_is_not_accuracy_success']=True;item['local_global_uniqueness_proven']=False
                item['rejected_edges']=(row.get('observation_filter') or {}).get('rejected_edge_ids',[])
                identity=(row['method'],row['id']);C.require(identity not in data,'duplicate display row');data[identity]=item
        ids=C.read(args.cohort)['ids'];C.require(len(data)==1470 and all((m,i) in data for i in ids for m in METHODS),'all1470 display records required')
        cases=C.read(CASE_PROTOCOL)['cases'];case_ids={c['id'] for c in cases}
        obs={r['id']:r for r in C.rows(paths['observations']) if r['head_arm']=='IMAGE_ROLE' and r['id'] in case_ids}
        C.require(set(obs)==case_ids,'original case observation join')
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for i,m in enumerate(METHODS):
                a=np.asarray([data[m,f]['pose'][field] for f in ids if data[m,f]['pose']['available']])
                axis.scatter(a,np.full(len(a),i),s=7,alpha=.25)
                if len(a):axis.plot(np.quantile(a,[.1,.9]),[i,i],linewidth=3);axis.scatter([a.mean()],[i],marker='D',facecolor='white',edgecolor='black')
            axis.set_yticks(range(6),LABELS);axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1);axis.grid(alpha=.2);axis.set_xlabel(field)
        fig.suptitle('All245 operating outputs; failures and fallbacks retain their original denominator\nGeometric proxy DEV; diamond=mean, segment=P10–P90')
        fig.savefig(folder/'01_all_operational.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for m,label in zip(METHODS[-2:],LABELS[-2:]):
                a=np.asarray([(data['N3_SUBPIX',f]['pose'][field],data[m,f]['pose'][field]) for f in ids if data[m,f]['pose']['available'] and data['N3_SUBPIX',f]['pose']['available']])
                axis.scatter(a[:,0],a[:,1],s=10,alpha=.4,label=label)
            upper=max(1.,max(data[m,f]['pose'][field] for m in METHODS[-2:]+('N3_SUBPIX',) for f in ids if data[m,f]['pose']['available']))
            axis.plot([0,upper],[0,upper],'k--');axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1);axis.set_xlabel('N3 '+field);axis.set_ylabel('Repair '+field);axis.legend(fontsize=8);axis.grid(alpha=.2)
        fig.suptitle('Same245 frames; below diagonal favors repair; geometric proxy reference')
        fig.savefig(folder/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
        for ci,c in enumerate(cases):
            with Image.open(paths['image:'+str(ci)]) as source:rgb=np.asarray(source.convert('RGB'))
            fig,axes=plt.subplots(2,3,figsize=(18,12),layout='constrained')
            for axis,m,label in zip(axes.flat,METHODS,LABELS):
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
                    axis.plot(support[:,0],support[:,1],'--',color='yellow',linewidth=1.3);axis.text(*support[0],str(edge),color='yellow',fontsize=8)
                    displayed.append(dict(edge=edge,support_points=support.tolist(),fit=edge in r['accepted_fit_line_edges']))
                ref=r['evaluation_reference'];g=np.asarray(ref.get('native_points_px',[]),float);known=ref.get('valid_native_ids',[])
                if len(g):axis.scatter(g[known,0],g[known,1],s=14,marker='.',color='white',alpha=.7)
                T,rot=r['pose'].get('translation_cm'),r['pose'].get('rotation_deg')
                axis.set_title(label+'\n'+r['output_status']+' / '+str(r['solver_state'])+'\nT='+str(round(T,4) if T is not None else None)+'cm; R='+str(round(rot,4) if rot is not None else None)+'deg\nfit='+str(r['accepted_fit_ids'])+'; lines='+str(r['accepted_fit_line_edges'])+'; removed='+str(r['rejected_edges']),fontsize=8)
                axis.axis('off');case_rows.append(dict(case_index=ci,id=c['id'],method=m,display=r,displayed_lines=displayed))
            fig.suptitle(c['label']+' / '+c['id']+'\ncyan=output; green+=actual sparse points; yellow=retained line support; orange=H reprojection; white=proxy\nNumerical NEW differs from accuracy success; missing points and reprojections are never new fit observations.',fontsize=9)
            fig.savefig(folder/('case_%02d.png'%(ci+1)),dpi=140);plt.close(fig)
        C.require(len(case_rows)==36 and len(list(folder.glob('*.png')))==8,'all fixed figures complete')
        with R.output(args,NAMES[4]).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
            for r in case_rows:z.write((json.dumps(C.finite(r),ensure_ascii=False,allow_nan=False)+'\n').encode())
        C.require(inputs(args)[1]==b,'display inputs changed')
        C.write_new(R.output(args,NAMES[2]),dict(complete=True,protocol=C.binding(R.DOC/NAMES[0]),inputs=b,panels=36,PNG_files=8,
            figures=[C.binding(p) for p in sorted(folder.glob('*.png'))],case_rows=C.binding(R.DOC/NAMES[4]),
            elapsed_seconds=time.monotonic()-began,new_model_fit_GT_score_training_RGB_calls=0,selection_uses_new_scores=False))
        print('REPAIR_FIGURES_COMPLETE',8,36,flush=True)
    except BaseException as exc:
        C.write_new(R.output(args,NAMES[3]),dict(complete=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),
            preserved_PNG_prefix=[C.binding(p) for p in sorted(folder.glob('*.png'))],completed_case_panels=len(case_rows),automatic_retry=False));raise


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','run');args=p.parse_args()
    (freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
