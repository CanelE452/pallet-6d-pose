"""Show fixed cases and saved point-consensus decisions without fitting a pose.

The six cases predate these results. Numerical NEW, a lower proxy error, and
an independently checked decision do not certify physical correspondence.
"""
from pathlib import Path
import gzip
import json
import os
import time
import traceback

from . import runner as R
from . import evaluation as E
from . import pipeline as P
from ..pallet_sparse_local_line_20261010_v9 import common as C

CASE_PROTOCOL=C.REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'
SOURCE_RENDER=C.REPO/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/render.py'
METHODS=('N3_SUBPIX','NATIVE_STANDARD_PNP',*P.METHODS)
LABELS=('N3→cornerSubPix','Native N3 + ordinary PnP','Original point consensus','Restored count→SSE decision')
NAMES=('FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURES_FAILURE.json','FIGURE_CASE_ROWS.jsonl.gz')

def display_row(row):
    s=row.get('solver') or {};new=row['new_pose_estimated'];decision=row.get('decision_restoration') or {}
    return dict(id=row['id'],method=row['method'],native_points=row['native_points'],pose=row['pose'],
        output_status=row['output_status'],new_pose_estimated=new,input_points=row.get('input_points'),
        reprojected_ids=row.get('reprojected_ids',[]),evaluation_reference=row.get('evaluation_reference',{}),
        actual_used_U=s.get('used'),applied_H=row.get('hidden_initial'),fallback_used=row['fallback_used'],
        pose_available=row['pose_available'],accepted_fit_ids=s.get('fit_input_ids') if new else [] if 'fit_input_ids' in s else None,
        diagnostic8px_inlier_ids=s.get('final_inliers'),solver_state=s.get('state'),
        selected_boundary_ids=row.get('selected_corner_ids'),observation_supply=row.get('observation_supply'),
        decision_restored=decision.get('restored'),decision_execution_kind=row.get('decision_execution_kind'),
        inherited_multiple_solutions=s.get('multiple_solutions'),global_uniqueness_proven=False,
        line_factor_count=0,numerical_NEW_is_not_accuracy_success=True,physical_ownership_truth='UNKNOWN')

def recorded(value):return 'unrecorded' if value is None else str(value)

def inputs(args):
    _,score=E.score_gate(args);case=C.read(CASE_PROTOCOL);cohort=C.read(args.cohort)
    C.bound(args.cohort,case['cohort'],'original six-case cohort')
    ids=cohort['ids'];by_id={f['id']:f for f in cohort['frames']};cases=case['cases']
    C.require(len(ids)==len(by_id)==245 and len(cases)==6 and len({c['id'] for c in cases})==6,'original6/full245')
    C.require(sum(c['label']=='clean' for c in cases)==sum(c['label']=='moderate' for c in cases)==3,'fixed3clean+3moderate')
    paths=dict(renderer=Path(__file__),source_renderer=SOURCE_RENDER,case_protocol=CASE_PROTOCOL,
        new_scores=R.DOC/'PREDICTIONS.jsonl.gz',ordinary_scores=Path(args.ordinary_parent)/'PREDICTIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',fixed_scoring=Path(args.v9_input)/'SCORING_RECEIPT.json',
        metrics=R.DOC/'METRICS.json',scoring_receipt=R.DOC/'SCORING_RECEIPT.json',scoring_started=R.DOC/'SCORING_STARTED.json',
        scoring_authority=R.DOC/'SCORING_AUTHORITY.json',evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',
        geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry_validation=R.DOC/'VALIDATION_CHECKS.json',
        validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        geometry_checker=R.CODE/'validation_checks.py',statistics_started=R.DOC/'STATISTICS_STARTED.json',
        verification=R.DOC/'VERIFICATION.json',verification_protocol=R.DOC/'STATISTICS_VERIFICATION_PROTOCOL.json',
        verification_started=R.DOC/'STATISTICS_VERIFICATION_STARTED.json',verification_checker=R.CODE/'verify.py',
        evaluation_code=Path(E.__file__),pipeline_code=Path(P.__file__),protocol=Path(args.protocol),cohort=Path(args.cohort),
        original_V16_full_scalar_FAIL=Path(args.decision_parent)/'VALIDATION_READY_CHECKS.json',
        complete_all_failure_trace=Path(args.decision_parent)/'TRACE_VALIDATION_CHECKS.json',
        point_only_failure_scope=Path(args.decision_parent)/'FAILURE_WITNESS_SUMMARY.json')
    C.bound(paths['new_scores'],score['predictions'],'actual paired490 decision scores')
    ordinary=E.ordinary_authority(args);C.require(ordinary['complete'] is True and ordinary['passed'] is True,'immutable ordinary control authority')
    C.bound(paths['ordinary_scores'],ordinary['bindings']['PREDICTIONS.jsonl.gz'],'actual ordinary490 scores')
    fixed=C.read(paths['fixed_scoring']);C.require(fixed['complete'] is True and fixed['fixed_control_rows']==490,'fixed Base/N3 scores')
    C.bound(paths['fixed_scores'],fixed['fixed_predictions'],'unchanged Base/N3 score bytes')
    metrics=C.read(paths['metrics']);C.require(metrics['complete'] is True and metrics['methods']==list(E.METHODS)
        and metrics['actual_arithmetic']==E.ARITHMETIC,'six methods retain all245/1470')
    check=C.read(paths['verification']);own=C.read(paths['verification_protocol']);start=C.read(paths['verification_started'])
    C.require(check['schema']=='independent_point_consensus_decision_statistics_checks_v18'
        and own['schema']=='independent_point_consensus_decision_statistics_protocol_v18'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['error'] is None
        and check['inputs']==own['inputs']==start['inputs']
        and check['actual_counts']['saved_scored_rows_streamed']==1470
        and check['actual_counts']['moment_scalars_checked']==972
        and check['actual_counts']['distributions_checked']==162
        and check['actual_counts']['pair_groups_checked']==45
        and check['actual_counts']['metric_CI_slots_checked']==135,'actual independent972/135 verification PASS')
    C.bound(paths['verification_protocol'],check['protocol'],'statistics own protocol')
    C.bound(paths['verification_protocol'],start['protocol'],'statistics STARTED protocol')
    C.bound(paths['verification_checker'],check['checker'],'current statistical checker')
    C.require(check['source_topology_adapter']==own['source_topology_adapter']==start['source_topology_adapter']
        and check['source_topology_adapter']['scalar_formula_AST_changes']==0
        and all(v['V16_original_full_scalar_validation_passed'] is False for v in (check,own,start))
        and check['geometry_gate_scope']==own['geometry_gate_scope']==start['geometry_gate_scope'],
        'unchanged statistical arithmetic and narrow point-only authority; full V16 stays FAIL')
    for key,alias in (('METRICS','metrics'),('new_scores','new_scores'),('ordinary_scores','ordinary_scores'),('fixed_scores','fixed_scores'),
        ('scoring_receipt','scoring_receipt'),('scoring_started','scoring_started'),('statistics_started','statistics_started'),
        ('geometry_seal','geometry_seal'),('geometry_validation','geometry_validation'),
        ('geometry_validation_protocol','validation_protocol'),('geometry_validation_started','validation_started'),
        ('evaluation_protocol','evaluation_protocol'),('scoring_authority','scoring_authority'),('protocol','protocol')):
        C.bound(paths[alias],check['inputs'][key],'independently verified figure input '+key)
    for i,c in enumerate(cases):
        C.require(c['id'] in by_id and all(c[k]==by_id[c['id']][k] for k in ('session','label','image')),'original case metadata')
        image=Path(args.source_root)/c['image']['path'];C.bound(image,c['image'],'original RGB');paths['image:'+str(i)]=image
    return paths,{k:C.binding(p) for k,p in paths.items()}

def freeze(args):
    _,b=inputs(args)
    C.write_new(R.output(args,NAMES[0]),dict(schema='prespecified_point_consensus_decision_display_v18',inputs=b,
        methods=list(METHODS),source_methods=list(E.METHODS),labels=list(LABELS),cases=6,panels=24,PNG_files=8,frames=245,
        source_scored_rows=1470,display_records=980,selection_uses_new_scores=False,new_model_fit_GT_score_training_calls=0,
        frozen_after_complete_scores_and_independent_statistics_before_render=True,source_RGB_is_existing_only=True,
        physical_reference_is_GEOMETRIC_PROXY=True,partial_line_factors=0,missing_H_or_fit_metadata_displayed_as='unrecorded'))
    print('POINT_CONSENSUS_DECISION_FIGURES_FROZEN',flush=True)

def run(args):
    lock=C.read(R.DOC/NAMES[0]);paths,b=inputs(args)
    C.require(lock['inputs']==b and lock['methods']==list(METHODS) and lock['source_methods']==list(E.METHODS),'own figure freeze differs')
    for name in NAMES[1:]:R.output(args,name)
    folder=R.DOC/'figures';C.require(not folder.exists(),'preserve original figures')
    C.write_new(R.output(args,NAMES[1]),dict(protocol=C.binding(R.DOC/NAMES[0]),configured_PNGs=8,configured_panels=24,source_scored_rows=1470))
    began=time.monotonic();case_rows=[]
    try:
        folder.mkdir()
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        data={}
        for key,allowed in (('fixed_scores',E.METHODS[:2]),('ordinary_scores',E.ORDINARY_METHODS),('new_scores',P.METHODS)):
            count=0
            for row in C.rows(paths[key]):
                identity=(row['method'],row['id']);C.require(row['method'] in allowed and identity not in data,'unknown/duplicate display source row')
                data[identity]=display_row(row);count+=1
            C.require(count==490,'complete immutable paired score stream')
        ids=C.read(args.cohort)['ids'];C.require(len(data)==1470 and all((m,fid) in data for fid in ids for m in E.METHODS),'all1470 source rows')
        cases=C.read(CASE_PROTOCOL)['cases'];metrics=C.read(paths['metrics'])
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for i,m in enumerate(METHODS):
                values=np.asarray([data[m,f]['pose'][field] for f in ids if data[m,f]['pose']['available']])
                saved=metrics['strata']['combined']['methods'][m]['metrics']['operational'][field]
                C.require(len(values)==saved['n'],'saved operational denominator')
                axis.scatter(values,np.full(len(values),i),s=7,alpha=.25)
                if len(values):axis.plot([saved['median'],saved['P90']],[i,i],linewidth=3);axis.scatter([saved['mean']],[i],marker='D',facecolor='white',edgecolor='black')
            axis.set_yticks(range(len(METHODS)),[label+' (n='+str(metrics['strata']['combined']['methods'][m]['operational'])+'/245)' for m,label in zip(METHODS,LABELS)])
            axis.invert_yaxis();axis.set_xscale('symlog',linthresh=1);axis.grid(alpha=.2);axis.set_xlabel(field)
        fig.suptitle('All245 operating outputs, including fallback; diamond=saved mean, segment=saved median–P90\nKnown DEV geometric proxy; numerical NEW is not pose accuracy success')
        fig.savefig(folder/'01_all_operational.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for axis,field in zip(axes,('translation_cm','rotation_deg')):
            for restored,label in ((True,'Extra support veto removed'),(False,'Original decision retained')):
                pairs=np.asarray([(data[P.METHODS[0],f]['pose'][field],data[P.PRIMARY,f]['pose'][field]) for f in ids
                    if data[P.PRIMARY,f]['decision_restored']==restored and data[P.PRIMARY,f]['pose']['available'] and data[P.METHODS[0],f]['pose']['available']])
                if len(pairs):axis.scatter(pairs[:,0],pairs[:,1],s=12,alpha=.45,label=label+' n='+str(len(pairs)))
            values=[data[m,f]['pose'][field] for m in P.METHODS for f in ids if data[m,f]['pose']['available']]
            upper=max([1.,*values]);axis.plot([0,upper],[0,upper],'k--');axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
            axis.set_xlabel('Original point consensus '+field);axis.set_ylabel('Restored same best '+field);axis.legend(fontsize=8);axis.grid(alpha=.2)
        fig.suptitle('Same245 q/H and fitted candidate pools; below diagonal favors the restored decision\nTrue non-equivalent numerical ties remain fallback. No pose re-fit or candidate reselection.')
        fig.savefig(folder/'02_same_frame_pairs.png',dpi=140);plt.close(fig)
        for ci,c in enumerate(cases):
            with Image.open(paths['image:'+str(ci)]) as source:rgb=np.asarray(source.convert('RGB'))
            fig,axes=plt.subplots(2,2,figsize=(16,12));fig.subplots_adjust(hspace=.41,wspace=.04,top=.79)
            for axis,m,label in zip(axes.flat,METHODS,LABELS):
                r=data[m,c['id']];axis.imshow(rgb);points=np.asarray(r['native_points'],float)
                valid=np.isfinite(points[:8]).all(1)&(points[:8]!=[-1,-1]).any(1)
                axis.scatter(points[:8][valid,0],points[:8][valid,1],s=28,facecolor='none',edgecolor='cyan')
                for k in np.flatnonzero(valid):axis.text(points[k,0]+3,points[k,1]+3,str(k),fontsize=8,color='cyan')
                for k in r['reprojected_ids']:axis.scatter(*points[k],s=65,marker='s',facecolor='none',edgecolor='orange')
                if r['input_points'] is not None:
                    supplied=np.asarray(r['input_points'],float);v=np.isfinite(supplied[:8]).all(1)&(supplied[:8]!=[-1,-1]).any(1)
                    axis.scatter(supplied[:8][v,0],supplied[:8][v,1],s=28,marker='+',color='#43ff50')
                reference=r['evaluation_reference'];truth=np.asarray(reference.get('native_points_px',[]),float);known=reference.get('valid_native_ids',[])
                if len(truth):axis.scatter(truth[known,0],truth[known,1],s=14,marker='.',color='white',alpha=.7)
                T,rot=r['pose'].get('translation_cm'),r['pose'].get('rotation_deg')
                text=label+'\n'+r['output_status']+' / '+recorded(r['solver_state'])+'\nT='+str(round(T,4) if T is not None else None)+'cm; R='+str(round(rot,4) if rot is not None else None)+'deg'
                text+='\nfit U='+recorded(r['accepted_fit_ids'])+'; 8px support='+recorded(r['diagnostic8px_inlier_ids'])
                text+='\nH(pred)='+recorded(r['applied_H'])+'; boundary adopted='+recorded(r['selected_boundary_ids'])
                text+='\nrestored decision='+recorded(r['decision_restored'])+'; multiple solutions='+recorded(r['inherited_multiple_solutions'])
                axis.set_title(text,fontsize=9,pad=8);axis.axis('off')
                case_rows.append(dict(case_index=ci,id=c['id'],method=m,display=r,displayed_lines=[]))
            fig.suptitle(c['label']+' / '+c['id']+'\ncyan=output; green+=actual supplied q; orange=NEW H projection; white=evaluation proxy\nNative ordinary fits whole U; point consensus fits its approved actual inliers. Fallback retains original all8 N3.\nAbsent H/fit metadata is unrecorded. No line factor, new q selection, physical ownership or accuracy guarantee.',fontsize=10)
            fig.savefig(folder/('case_%02d.png'%(ci+1)),dpi=140);plt.close(fig)
        C.require(len(case_rows)==24 and len(list(folder.glob('*.png')))==8,'all fixed figures complete')
        with R.output(args,NAMES[4]).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
                for row in case_rows:z.write((json.dumps(C.finite(row),ensure_ascii=False,allow_nan=False)+'\n').encode())
            raw.flush();os.fsync(raw.fileno())
        C.require(inputs(args)[1]==b,'display inputs changed')
        C.write_new(R.output(args,NAMES[2]),dict(complete=True,protocol=C.binding(R.DOC/NAMES[0]),inputs=b,panels=24,PNG_files=8,
            source_display_records=1470,display_records=980,display_methods=list(METHODS),figures=[C.binding(p) for p in sorted(folder.glob('*.png'))],
            case_rows=C.binding(R.DOC/NAMES[4]),elapsed_seconds=time.monotonic()-began,new_model_fit_GT_score_training_calls=0,
            existing_RGB_images_read=6,selection_uses_new_scores=False,statistical_moments_recomputed=False))
        print('POINT_CONSENSUS_DECISION_FIGURES_COMPLETE',8,24,flush=True)
    except BaseException as exc:
        C.write_new(R.output(args,NAMES[3]),dict(complete=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),
            preserved_PNG_prefix=[C.binding(p) for p in sorted(folder.glob('*.png'))],completed_case_panels=len(case_rows),automatic_retry=False));raise

def main():
    p=E.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','run');args=p.parse_args()
    (freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
