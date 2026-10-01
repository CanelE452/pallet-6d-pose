"""Export the fixed253-plus-direction18 input intervention only after root authorizes completed results.

No fitting, candidate selection, PnP or metric calculation. Certified training,
source validation and actual real evaluation remain separate stages. A failed
source gate permits historical R0 input illustrations only.
"""
from . import common as C
import argparse
import csv
import io
import itertools
import json
import numpy as np

SPECS=[('translation_cm','median','T median (cm)'),('rotation_deg','median','R median (deg)'),
       ('translation_cm','P90','T P90 (cm)'),('rotation_deg','P90','R P90 (deg)')]


def publish(path,value):
    payload=value if isinstance(value,bytes) else (value if isinstance(value,str) else
        json.dumps(C.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():assert path.read_bytes()==payload,('EXISTING_PUBLICATION_DIFFERS',str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(payload)


def csv_export(name,rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);publish(C.DOC/name,out.getvalue());return len(rows)


def finite_text(value,digits=6):
    return 'inf (failure retained)' if value is None or np.isposinf(value) else f'{value:.{digits}f}'


def metric_row(model,summary):
    full=summary['full_population']
    parts=[finite_text(full[axis][quantile]) for axis,quantile,_ in SPECS]
    return '| '+model+' | '+' | '.join(parts)+f" | {summary['failed_pose']} |"


def plots(source,previous,training,traces,receipts,real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    folder=C.DOC/'figures';folder.mkdir(parents=True,exist_ok=True)
    artifacts,values=[],{};labels=['R0_ONLY','UNION s1','UNION s2','UNION s3'];x=np.arange(4)
    def finish(fig,name):
        path=folder/name;fig.savefig(path,dpi=145);plt.close(fig);artifacts.append(C.bind(path))
    fig,axes=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
    for col,model in enumerate(C.MODEL_NAMES):
        rows=[r for r in traces if r['model']==model]
        calls=[r['call'] for r in rows];j=[r['objective'] for r in rows];h=[r['Huber'] for r in rows];s=[r['Sign_logistic'] for r in rows];penalty=[r['L2_penalty'] for r in rows]
        gap=[r['gradient_gap_upper_bound']/1e-6 for r in rows];linf=[r['gradient_linf']/1e-8 for r in rows]
        values[f'train_{model}']=dict(calls=calls,objective=j,Huber=h,Sign_logistic=s,L2_penalty=penalty,previous_weight_same_new_J=training['models'][model]['previous_fixed_sign']['same_new_objective']['objective'],gap_ratio_to_limit=gap,linf_ratio_to_limit=linf,
            armijo_accepted=[r['armijo_accepted'] for r in rows],final_accepted_call=receipts[model]['final_accepted_call'])
        ax=axes[0,col];ax.plot(calls,j,label='Huber + sign logistic + ridge',color='#0d9488');ax.plot(calls,h,label='Huber data term',color='#d97706',linestyle='--')
        ax.plot(calls,s,label='Sign logistic term',color='#8b5cf6',linestyle='-.')
        ax.axhline(training['models'][model]['previous_fixed_sign']['same_new_objective']['objective'],color='#64748b',linestyle=':',label='Previous253 weights + zero18 under SAME J')
        ax.set_title(model);ax.set_yscale('log');ax.set_ylabel('Full TRAIN objective /2598 (log scale)');ax.grid(alpha=.2)
        ax=axes[1,col];ax.plot(calls,np.maximum(gap,1e-30),label='Gap bound /1e-6',color='#6366f1');ax.plot(calls,np.maximum(linf,1e-30),label='Gradient Linf /1e-8',color='#d97706')
        ax.axhline(1.,color='#dc2626',linestyle='--',label='Both must be <=1')
        ax.set_yscale('log');ax.set_xlabel('Every objective call, including rejected trials');ax.set_ylabel('Ratio to fixed certificate limit (log)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Fixed block Newton + Armijo | only fixed direction18 added to253, same loss/targets/runtime, fresh zeros\nEvery initial/trial evaluation retained; final checkpoint is the last accepted certified point. No previous-weight warm start; the old253 model is embedded with eighteen zero rows.')
    finish(fig,'training_huber_and_certificate.png')
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,(axis,q,title) in zip(axes.flat,SPECS):
        old=[previous['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        new=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        fixed=source['summaries']['R0_GEO']['full_population'][axis][q]
        values[f'source_{axis}_{q}']=dict(previous_sign253=old,current_direction271=new,fixed_R0_GEO=fixed)
        for offset,arr,color,label in [(-.2,old,'#64748b','Previous253 / same Huber + sign'),(.2,new,'#0d9488','Current271 / same Huber + sign')]:
            shown=[v if v is not None and np.isfinite(v) else 0. for v in arr]
            ax.bar(x+offset,shown,.4,color=color,label=label)
            for j,v in enumerate(arr):ax.text(j+offset,shown[j],finite_text(v,3),ha='center',va='bottom',fontsize=8)
        ax.axhline(fixed,color='#a855f7',linestyle=':',label='Fixed R0 GEO')
        if q=='P90':ax.axhline(fixed*1.05,color='#dc2626',linestyle='--',label='Fixed R0 GEO x1.05')
        vmax=max(v for v in old+new+[fixed*1.05] if v is not None and np.isfinite(v))
        ax.set_ylim(0,max(vmax,1e-9)*1.28);ax.set_xticks(x,labels);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle(f"Source VAL1024 | {source['checks_passed']}/{source['checks_total']} checks pass\nSame frozen candidates and unchanged45 checks. Fixed direction18 is the only input addition; loss and all evaluation conditions are unchanged.")
    finish(fig,'source_val_method_comparison.png')
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,axis in zip(axes[:2],['T','R']):
        mae=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['MAE'] for m in C.MODEL_NAMES]
        rmse=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['RMSE'] for m in C.MODEL_NAMES]
        values[f'train_nonanchor_{axis}']=dict(MAE=mae,RMSE=rmse)
        ax.bar(x-.2,mae,.4,label='MAE',color='#0d9488');ax.bar(x+.2,rmse,.4,label='RMSE',color='#d97706')
        ax.set_title(f'{axis}: signed-log1p residual\nValid nonanchor candidates only',fontsize=10);ax.set_ylabel('Transformed normalized error, not cm/deg');ax.set_xticks(x,labels);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    before=[training['models'][m]['previous_fixed_sign']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    after=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    values['train_unsafe_count']=dict(previous_sign253=before,current_direction271=after)
    axes[2].bar(x-.2,before,.4,label='Previous253 / Huber + sign',color='#64748b');axes[2].bar(x+.2,after,.4,label='Current271 / Huber + sign',color='#0d9488')
    axes[2].set_xticks(x,labels);axes[2].set_title('Actual selected pose: either-axis violation\nAvailable2597; failed1 retained separately',fontsize=10);axes[2].legend(fontsize=8);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('TRAIN only | exact0 anchor predictions are a construction, not learned accuracy\nReference-free runtime uses all original valid candidates; physical nonregression is not guaranteed.')
    finish(fig,'train_axis_regression_and_violations.png')
    if real is not None:
        models=real['models'];fig,axes=plt.subplots(3,4,figsize=(21,15),constrained_layout=True)
        for row,pop in enumerate(['NATURAL99','CLEAN29','WOOD45']):
            for col,(axis,q,title) in enumerate(SPECS):
                ax=axes[row,col];vals=[real['summaries'][pop][m]['full_population'][axis][q] for m in models]
                values[f'real_{pop}_{axis}_{q}']=dict(models=models,values=vals)
                shown=[v if v is not None and np.isfinite(v) else 0. for v in vals]
                ax.barh(np.arange(len(models)),shown,color=['#0d9488' if m.startswith('UNION') else '#d97706' if m=='R0_ONLY' else '#64748b' for m in models])
                ax.set_yticks(np.arange(len(models)),models,fontsize=8);ax.invert_yaxis();ax.set_title(f'{pop} | {title}',fontsize=11);ax.set_xlim(0,max(max(shown),1e-9)*1.25);ax.grid(axis='x',alpha=.15)
                for j,v in enumerate(vals):ax.text(shown[j],j,' '+finite_text(v,2),va='center',fontsize=7)
        fig.suptitle('Actual current real evaluation | all13 models and all173 frames retained\nOriginal controls retain their identities; no best seed; pose availability does not certify accuracy.',fontsize=14)
        finish(fig,'real_all_models.png')
        keys=list(real['stability']['gates']);matrix=np.asarray([[real['matched_intervention_stability']['gates'][k]['PASS'],real['original_goal_stability']['gates'][k]['PASS'],real['stability']['gates'][k]['PASS']] for k in keys],int)
        values['real_gate_matrix']=dict(rows=keys,columns=['matched','original_SINGLE251','combined_AND'],values=matrix.tolist())
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.imshow(matrix,cmap=ListedColormap(['#fee2e2','#ccfbf1']),vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(3),['Matched intervention','Original SINGLE251 contract','Required AND']);ax.set_yticks(range(5),keys)
        for i in range(5):
            for j in range(3):ax.text(j,i,'PASS' if matrix[i,j] else 'FAIL',ha='center',va='center')
        ax.set_title(f"Stable joint T/R {'established' if real['stability']['PASS'] else 'NOT established'} | {int(matrix[:,2].sum())}/5 categories PASS")
        finish(fig,'real_stability_gates.png')
    return artifacts,values



def gallery_baseline(not_run_path):
    """Use historical R0 panels only; do not open any new real result/cache."""
    import matplotlib.pyplot as plt
    from PIL import Image
    prior_path=C.SIGN_DOC/'REPORT_DATA.json';prior=C.read(prior_path)
    selection_path=C.ANCHOR_DOC/'REPORT_DATA.json';old_selection=C.read(selection_path)
    selected=[r['id'] for r in old_selection['illustrations']]
    assert len(selected)==len(set(selected))==6
    prior_rows={r['id']:r for r in prior['illustrations']}
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(signs[i]!=signs[j])==1]
    details,artifacts=[],[]
    for page in range(3):
        fig,axes=plt.subplots(2,1,figsize=(11,14),squeeze=False)
        fig.subplots_adjust(top=.84,bottom=.02,left=.04,right=.96,hspace=.45)
        for rowidx,fid in enumerate(selected[2*page:2*page+2]):
            old=prior_rows[fid];saved=next(p for p in old['panels'] if p['model']=='R0');pose=saved['pose']
            C.verify(old['image'])
            with Image.open(C.ROOT/old['image']['path']) as im:rgb=np.asarray(im.convert('RGB'))
            assert list(rgb.shape[:2])==old['image_hw'] and pose['available']
            corners=signs*np.asarray(pose['cf_extents'])/2
            xyz=corners@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid'])
            camera=xyz@np.asarray(old['K']).T
            assert np.isfinite(camera).all() and (camera[:,2]!=0).all()
            uv=camera[:,:2]/camera[:,2:3]
            np.testing.assert_allclose(corners,saved['corners_cf'],rtol=0,atol=0)
            np.testing.assert_allclose(xyz,saved['corners_camera'],rtol=0,atol=0)
            np.testing.assert_allclose(uv,saved['projected_uv'],rtol=0,atol=0)
            ax=axes[rowidx,0];ax.imshow(rgb);drawn=[]
            for i,j in edges:
                if xyz[[i,j],2].min()>0:ax.plot(uv[[i,j],0],uv[[i,j],1],color='#ef4444',lw=2);drawn.append([i,j])
            ax.set_xlim(-.5,rgb.shape[1]-.5);ax.set_ylim(rgb.shape[0]-.5,-.5)
            dims=' / '.join(f'{100*d:g}' for d in old['dimensions_m'])
            ax.set_title(f"{old['recording']} | HISTORICAL R0 INPUT BASELINE ONLY\nObject Width / Height / Depth: {dims} cm | {saved['hypothesis']}\nHistorical R0 T={saved['T_cm']:.2f} cm, R={saved['R_deg']:.2f} deg\nCurrent271 signed-axis direction model: NOT EVALUATED",fontsize=10)
            ax.axis('off')
            details.append(dict(id=fid,recording=old['recording'],image=old['image'],image_hw=old['image_hw'],K=old['K'],dimensions_m=old['dimensions_m'],
                corner_signs=signs.tolist(),edges=edges,current_learned_real_evaluated=False,
                display_selection='Fixed previous six IDs; largest historical R0 T per natural recording, not a representative sample.',
                panels=[dict(model='R0',parent='R0',hypothesis=saved['hypothesis'],pose=pose,corners_cf=corners,corners_camera=xyz,
                    projected_uv=uv,T_cm=saved['T_cm'],R_deg=saved['R_deg'],drawn_edges=drawn,
                    pose_source='Historical operational R0 input illustration only; not the current271 signed-axis direction model or a new performance calculation.')]))
        fig.suptitle('Historical operational R0 input examples ONLY\nCurrent271 signed-axis direction model real evaluation NOT RUN\nNo current learned real routes or T/R evaluation\nSingle RGB + pallet dimensions + existing calibrated K',fontsize=13,y=.985)
        path=C.DOC/'figures'/f'baseline_input_rgb_dimensions_{page+1}.jpg';fig.savefig(path,dpi=115);plt.close(fig);artifacts.append(C.bind(path))
    assert any(np.array_equal(np.asarray(r['dimensions_m']),np.asarray([1.1,.11,1.3])) for r in details)
    selection=dict(complete=True,status='HISTORICAL_R0_INPUT_ILLUSTRATIONS_ONLY',source_selection_report=C.bind(selection_path),historical_baseline_report=C.bind(prior_path),
        current_learned_real_evaluated=False,not_run=C.bind(not_run_path),actual_RGB_images=6,panels=6,illustrations=details,
        new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0)
    publish(C.DOC/'GALLERY_SELECTION.json',selection)
    return artifacts,details



def gallery_learned(real_protocol, real, routing):
    import matplotlib.pyplot as plt
    from PIL import Image
    for key in ('metadata', 'poses'):
        C.verify(real_protocol['inputs'][key])
    C.verify(routing['choices'])
    metadata = C.read(C.ROOT / real_protocol['inputs']['metadata']['path'])
    meta = {r['id']: r for r in metadata}
    poses = C.read(C.ROOT / real_protocol['inputs']['poses']['path'])
    choices = C.read(C.ROOT / routing['choices']['path'])
    mb = next(b for b in real['artifacts'] if b['path'].endswith('/POSE_METRICS.json'))
    C.verify(mb)
    metrics = C.read(C.ROOT / mb['path'])
    old_report = C.read(C.ANCHOR_DOC / 'REPORT_DATA.json')
    selected = [r['id'] for r in old_report['illustrations']]
    assert len(selected) == len(set(selected)) == 6
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(signs[i] != signs[j]) == 1]
    panels_models = ['R0', 'UNION_s1', 'UNION_s2', 'UNION_s3']
    details, artifacts = [], []
    for page in range(3):
        fig, axes = plt.subplots(2,4,figsize=(24,12),constrained_layout=False)
        fig.subplots_adjust(top=.82,bottom=.03,left=.012,right=.988,wspace=.05,hspace=.46)
        for rowidx,fid in enumerate(selected[2*page:2*page+2]):
            row = meta[fid]
            C.verify(row['image'])
            with Image.open(C.ROOT / row['image']['path']) as im: rgb = np.asarray(im.convert('RGB'))
            assert list(rgb.shape[:2]) == row['hw']
            detail = dict(id=fid,recording=row['recording'],image=row['image'],image_hw=row['hw'],K=row['K'],
                          dimensions_m=row['xyz'],corner_signs=signs.tolist(),edges=edges,
                          display_selection='Same six IDs as prior pareto report: previous largest R0 T per natural recording; no selection on current learned outcomes.',panels=[])
            for col,model in enumerate(panels_models):
                if model == 'R0':
                    pose = poses['R0'][fid]['GEO_pose']
                    parent,hyp = 'R0',poses['R0'][fid]['GEO_name']
                else:
                    choice = choices['records'][model][fid]
                    pose,parent,hyp = choice['pose'],choice['parent'],choice['hypothesis']
                metric = metrics[model][fid]
                assert bool(pose['available'])==bool(metric['available'])
                corners=xyz=uv=None
                if pose['available']:
                    corners = signs * np.asarray(pose['cf_extents'])/2
                    xyz = corners @ np.asarray(pose['R_cf']).T + np.asarray(pose['centroid'])
                    camera = xyz @ np.asarray(row['K']).T
                    assert np.isfinite(camera).all() and (camera[:,2] != 0).all()
                    uv = camera[:,:2]/camera[:,2:3]
                ax = axes[rowidx,col]
                ax.imshow(rgb)
                drawn = []
                if pose['available']:
                    for i,j in edges:
                        if xyz[[i,j],2].min() > 0:
                            ax.plot(uv[[i,j],0],uv[[i,j],1],color='#ef4444' if model=='R0' else '#22d3ee',lw=2.1)
                            drawn.append([i,j])
                else:
                    ax.text(.5,.5,'POSE UNAVAILABLE\nFailure retained',transform=ax.transAxes,ha='center',va='center',
                            color='red',bbox=dict(facecolor='white',alpha=.85))
                ax.set_xlim(-.5,rgb.shape[1]-.5)
                ax.set_ylim(rgb.shape[0]-.5,-.5)
                dims=' x '.join(f'{100*d:g}' for d in row['xyz'])
                ax.set_title(f"{row['recording']} | {model}\nT={finite_text(metric['translation_cm'],2)} cm, R={finite_text(metric['rotation_deg'],2)} deg\nDimensions: {dims} cm\n{parent} / {hyp}",fontsize=10)
                ax.axis('off')
                detail['panels'].append(dict(model=model,parent=parent,hypothesis=hyp,pose=pose,
                    corners_cf=None if corners is None else corners.tolist(),corners_camera=None if xyz is None else xyz.tolist(),projected_uv=None if uv is None else uv.tolist(),
                    T_cm=metric['translation_cm'],R_deg=metric['rotation_deg'],drawn_edges=drawn,
                    pose_source='Frozen operational R0 or current learned whole-pose choice; direct K projection; no new PnP.'))
            details.append(detail)
        fig.suptitle('Actual current271 direction model real-image results | same RGB + dimensions + calibrated K\nR0 and all three UNION seeds; no reference-selected oracle or ground-truth outline\nFixed prior six-frame diagnostic examples, not a representative performance sample',fontsize=15,y=.98)
        p = C.DOC/'figures'/f'learned_rgb_dimensions_{page+1}.jpg'
        fig.savefig(p,dpi=115)
        plt.close(fig)
        artifacts.append(C.bind(p))
    selection = dict(complete=True,source_selection_report=C.bind(C.ANCHOR_DOC/'REPORT_DATA.json'),
                     routing=C.bind(C.DOC/'REAL_ROUTING_LOCK.json'),metrics=mb,metadata=real_protocol['inputs']['metadata'],
                     poses=real_protocol['inputs']['poses'],actual_RGB_images=6,panels=24,illustrations=details,
                     new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0,current_learned_real_evaluated=True)
    publish(C.DOC/'GALLERY_SELECTION.json',selection)
    return artifacts,details


def main():
    """Export only completed, independently verified actual outcomes."""
    protocol=C.protocol('TRAIN_PROTOCOL')
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json');training=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    source=C.read(C.DOC/'SOURCE_VAL_GATE.json');source_review=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    diagnostic_path=C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json'
    diagnostic=C.read(diagnostic_path) if diagnostic_path.exists() else None
    if diagnostic is not None:
        assert diagnostic['complete'] and diagnostic['PASS'] and diagnostic['no_new_selector_policy_evaluated']
        assert diagnostic['new_argmin_computations']==diagnostic['new_fits']==diagnostic['real_reference_reads']==diagnostic['new_real_routes']==0
        for binding in diagnostic['inputs'].values():C.verify(binding)
        assert C.bind(C.DOC/'SOURCE_VAL_GATE.json')==diagnostic['inputs']['current_source_val_gate']
    previous=C.read(C.SIGN_DOC/'SOURCE_VAL_GATE.json')
    assert protocol['feature_dim']==271 and protocol['base_feature_dim']==253 and protocol['direction_dim']==18
    assert complete['complete'] and complete['fit_count']==4 and complete['all_certified']
    baseline_names=set(source['summaries'])-set(C.MODEL_NAMES)
    assert len(baseline_names)==4
    for name in baseline_names:assert source['summaries'][name]==previous['summaries'][name]
    # Direction18 is fixed by the completed TRAIN-only input audit; no basis,
    # normalization, target or source/real condition is selected in this report.
    assert protocol['direction_receipt_binding']==protocol['inputs']['direction_receipt']
    assert training['complete'] and training['PASS'] and training['independent_check_PASS']
    assert source_review['complete'] and source_review['PASS']
    assert source_review['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    assert source['complete'] and source['checks_total']==45
    assert source['protocol']==training['protocol']==complete['protocol']==C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    assert source['loss_rule']=='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC'
    assert source['sign_rule']=='NONZERO_SIGNED_TARGET_AXES' and source['sign_coefficient']==1.
    assert source['real_routing_authorized']==source['PASS']
    real=real_review=real_protocol=real_routing=not_run=None
    not_run_path=C.DOC/'REAL_EVALUATION_NOT_RUN.json'
    if source['PASS']:
        assert not not_run_path.exists()
        real=C.read(C.DOC/'REAL_RESULTS.json');real_review=C.read(C.DOC/'REAL_VERIFICATION.json')
        assert real['complete'] and real_review['complete'] and real_review['PASS']
        C.verify(C.read(C.DOC/'REAL_PROTOCOL_SHA.json'))
        real_protocol=C.read(C.DOC/'REAL_PROTOCOL.json');real_routing=C.read(C.DOC/'REAL_ROUTING_LOCK.json')
        assert real['protocol']==C.bind(C.DOC/'REAL_PROTOCOL.json')
        assert real['routing_lock']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
        assert real['populations']==dict(NATURAL99=99,CLEAN29=29,WOOD45=45)
        assert len(real['models'])==13 and real['full_frame_rows']==13*173
        assert real['real_stability_contract']=='matched_intervention_AND_original_SINGLE251_stability'
        for b in real['artifacts']:C.verify(b)
    else:
        not_run=C.read(not_run_path)
        assert not_run['complete'] and not_run['status']=='NOT_RUN_SOURCE_GATE_FAILED'
        assert not_run['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
        assert not_run['learned_real_routes']==not_run['real_metric_calls']==not_run['raw_real_reference_reads']==0
        for path,absent in not_run['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
    C.verify(source['metrics'])
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        ids,models=z['ids'].tolist(),z['models'].tolist();errors={m:z[m].copy() for m in models}
    lock=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');C.verify(lock['choices']);choices=C.read(C.ROOT/lock['choices']['path'])
    assert choices['ids']==ids and len(ids)==1024
    val_rows=[]
    for model in models:
        assert errors[model].shape==(1024,2)
        for j,fid in enumerate(ids):
            error=errors[model][j];available=bool(np.isfinite(error).all());assert available or np.isposinf(error).all()
            choice=choices['records'].get(model,{}).get(fid,{})
            val_rows.append(dict(model=model,id=fid,split='VAL',available=available,T_cm=float(error[0]),R_deg=float(error[1]),
                candidate=choice.get('candidate_name','fixed_GEO'),fallback=choice.get('fallback',False)))
    checks=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,group in source['comparisons'].items() for b,comp in group.items() for k,v in comp['checks'].items()]
    receipts,traces,iteration_rows,exports={},[],[],{}
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');receipts[model]=fit
        for key in ('START','trace','checkpoint'):C.verify(fit[key])
        assert fit['complete'] and fit['fits_executed']==1 and fit['certificate']['PASS']
        assert fit['sign_rule']=='NONZERO_SIGNED_TARGET_AXES' and fit['sign_coefficient']==1.
        ck=C.read(C.ROOT/fit['checkpoint']['path'])
        assert ck['schema']=='pallet_pose_signed_axes_direction_linear271x2_v1'
        assert np.asarray(ck['weight']).shape==(271,2) and np.isfinite(ck['weight']).all()
        for artifact in (fit,ck,complete):
            assert artifact['feature_dim']==271 and artifact['base_feature_dim']==253 and artifact['direction_dim']==18
            assert artifact['direction_receipt_binding']==protocol['direction_receipt_binding']
            assert artifact['direction_normalization_sha']==protocol['direction_normalization_sha']
        log=[json.loads(s) for s in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        calls=[dict(model=model,**r) for r in log if r['event']=='objective'];its=[dict(model=model,**r) for r in log if r['event']=='iteration']
        assert len(calls)==fit['objective_calls'] and len(its)==fit['iterations']
        assert calls[-1]['call']==fit['final_accepted_call'] and calls[-1]['armijo_accepted']
        assert calls[-1]['Sign_logistic']==fit['certificate']['Sign_logistic']
        for row in calls:
            assert abs(row['objective']-row['Huber']-row['Sign_logistic']-row['L2_penalty'])<=1e-12
        traces.extend(calls);iteration_rows.extend(its)
        path=C.DOC/'model_parameters'/f'{model}.json';publish(path,(C.ROOT/fit['checkpoint']['path']).read_bytes())
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(path))
    assert len(traces)==complete['total_objective_calls'] and len(iteration_rows)==complete['total_iterations']
    real_status='실사 결과까지 평가했습니다. 안정성 판정은 아래 결과 문서를 확인해야 합니다.' if real is not None else 'source 조건 실패로 실사 평가는 실행하지 않았습니다.'
    publish(C.DOC/'model_parameters'/'README.md',
        '# 수렴 인증된 271×2 최종 학습 가중치4개\n\n네 JSON은 새 zero 초기화에서 실제 학습한 마지막 승인 checkpoint의 byte 동일 사본입니다. Huber+sign logistic+ridge의 수치 수렴 인증은 실제 T/R 성능 인증과 다릅니다. 이전 checkpoint에서 이어 학습하지 않았습니다.\n\n'+
        f"source {source['checks_passed']}/45. {real_status}\n\n[상세 결과](../REPORT_KO.md), [TRAIN 검산](../TRAIN_CONVERGENCE_KO.md), [source 판정](../SOURCE_VAL_GATE.json)\n")
    csv_counts={name:csv_export(name,rows) for name,rows in [('SOURCE_VAL_FRAME_RESULTS.csv',val_rows),('SOURCE_VAL_CHECKS.csv',checks),
        ('TRAINING_OBJECTIVE_LOG.csv',traces),('TRAINING_ITERATION_LOG.csv',iteration_rows)]}
    assert csv_counts['SOURCE_VAL_FRAME_RESULTS.csv']==8192 and csv_counts['SOURCE_VAL_CHECKS.csv']==45
    artifacts,figure_values=plots(source,previous,training,traces,receipts,real)
    if real is None:galleries,illustrations=gallery_baseline(not_run_path)
    else:galleries,illustrations=gallery_learned(real_protocol,real,real_routing)
    artifacts.extend(galleries)
    objective_comparison={m:dict(old_weight=training['models'][m]['previous_fixed_sign']['same_new_objective']['objective'],
        new_weight=training['models'][m]['independent_recompute']['objective'],
        decrease=training['models'][m]['same_new_objective_decrease']) for m in C.MODEL_NAMES}
    stable=bool(real is not None and real['stability']['PASS'])
    data=dict(complete=True,status='REAL_STABILITY_PASS' if stable else 'REAL_STABILITY_FAIL' if real is not None else 'CERTIFIED_TRAIN_SOURCE_GATE_FAILED',
        actual_new_fits=4,actual_training_attempts=4,certified_models=4,
        feature_dim=271,base_feature_dim=253,direction_dim=18,output_dim=2,
        source_VAL_checks_passed=source['checks_passed'],source_VAL_checks_total=source['checks_total'],source_VAL_evaluated=True,
        learned_real_evaluated=real is not None,current_real_TR_effect_measured=real is not None,
        stable_joint_improvement_achieved=stable,goal_complete=False,method_success=stable,
        code=C.bind(C.HERE/'report.py'),train_protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),source_gate=C.bind(C.DOC/'SOURCE_VAL_GATE.json'),
        previous_source_gate=C.bind(C.SIGN_DOC/'SOURCE_VAL_GATE.json'),training_verification=C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
        previous_training_verification=C.bind(C.SIGN_DOC/'TRAIN_CONVERGENCE.json'),
        source_verification=C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),
        real_not_run=C.bind(not_run_path) if not_run is not None else None,
        real_results=C.bind(C.DOC/'REAL_RESULTS.json') if real is not None else None,
        real_verification=C.bind(C.DOC/'REAL_VERIFICATION.json') if real is not None else None,
        prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),design=C.bind(C.DOC/'DESIGN_KO.md'),
        direction_receipt=protocol['inputs']['direction_receipt'],direction_representation=protocol['inputs']['direction_representation'],
        direction_verification=protocol['inputs']['direction_verification'],direction_normalization_sha=protocol['direction_normalization_sha'],
        previous_sign_report=C.bind(C.SIGN_DOC/'REPORT_DATA.json'),
        prior_RBF_report=C.bind(C.RBF_DOC/'REPORT_DATA.json'),basis=C.bind(C.RBF_DOC/'RBF_BASIS.json'),exports=exports,csv_rows=csv_counts,
        objective_calls=len(traces),iterations=len(iteration_rows),fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        rejected_Armijo_trials=sum(1 for r in traces if r['phase']=='trial' and not r['armijo_accepted']),
        same_new_objective_by_model=objective_comparison,
        same_objective_comparison_scope='Same Huber+sign+ridge objective and targets; old253 weights embedded with eighteen zero rows. Compare old and new full TRAIN J without claiming that the old542-dimensional gradient is the original506-parameter certificate.',
        zero_initialization_verified=True,previous_weight_warmstart=False,
        artifacts=artifacts,figure_values=figure_values,actual_RGB_images=6,illustrations=illustrations,
        gallery=C.bind(C.DOC/'GALLERY_SELECTION.json'),baseline_panels=6,current_learned_real_panels=18 if real is not None else 0,
        new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0,
        report_scope='Export frozen certified TRAIN/source/real outcomes and directly project already frozen poses; no optimizer, routing, quality or PnP recomputation.')
    if diagnostic is not None:data['source_fixed_choice_diagnostic']=C.bind(diagnostic_path)
    body=render_report(source,previous,training,receipts,traces,iteration_rows,real,data,diagnostic)
    publish(C.DOC/'REPORT_KO.md',body)
    publish(C.DOC/'PUBLIC_REPORT_KO.md',body)
    publish(C.DOC/'REPORT_DATA.json',data)
    verdict='고정 실사 안정성 조건 통과' if stable else '실사 안정성 조건 미달' if real is not None else 'source 조건 미달·실사 미실행'
    publish(C.DOC/'README.md',f"# 고정 방향18 추가: 253→271 — {verdict}\n\n[상세 결과·실제 RGB와 치수](REPORT_KO.md)\n\n새 zero 초기화4fit은 수렴 인증을 통과했습니다. source {source['checks_passed']}/45. {real_status}\n\n[TRAIN 검산](TRAIN_CONVERGENCE_KO.md), [source 검산](SOURCE_VAL_VERIFICATION_KO.md), [공개 검증](PUBLIC_REVIEW_KO.md)\n")
    print(json.dumps(dict(PASS=True,csv_rows=csv_counts,objective_calls=len(traces),iterations=len(iteration_rows),
        figures=len(artifacts),source_checks_passed=source['checks_passed'],source_gate_PASS=source['PASS'],
        learned_real_evaluated=real is not None,stable_joint_improvement_achieved=stable)))


def render_report(source,previous,training,receipts,traces,iterations,real,data,diagnostic):
    stable=data['stable_joint_improvement_achieved']
    if real is None:
        headline=f"4개 학습의 수렴 인증은 통과했지만 source {source['checks_passed']}/45로 실패했습니다. 사전 조건에 따라 이번 모델의 실사 선택·T/R 평가는 미실행이며, 안정적 공동 개선 목표는 달성하지 못했습니다."
    else:
        count=sum(bool(row['PASS']) for row in real['stability']['gates'].values())
        headline=f"4개 수렴 인증과 source 45/45를 통과한 뒤 실사 173장을 평가했습니다. 실사 안정성 {count}/5범주가 통과했으며, 안정적 공동 개선은 {'확인됐습니다' if stable else '확립되지 않았습니다'}."
    lines=['# 고정 잔차 방향18 추가: 253→271 입력 실험','',f'**{headline}**','',
        f"새 zero 초기화로 4개 모델을 학습했습니다. objective 호출 {len(traces)}회와 승인 Newton 반복 {len(iterations)}회를 모두 보존했습니다. 바꾼 요인은 입력 18개의 추가뿐이며, 이전 가중치를 이어 학습하지 않았습니다.",'',
        '운영 입력은 **단일 RGB + 팔레트 치수 + 기존 카메라 보정 K**입니다. 시간 정보, 추가 센서, 새 실사 정답을 넣지 않았습니다. 수렴 인증·source 조건·실사 안정성은 서로 다른 판정입니다.','',
        '## 바꾼 입력과 유지한 조건','',
        '[직전 253차원 Huber+sign 단계](../pallet_pose_signed_axes_sign_20261001_v1/REPORT_KO.md)는 수렴했으나 source 조건을 통과하지 못했습니다. 뒤이어 실시한 [TRAIN 입력 감사](../pallet_pose_residual_direction_audit_20261001_v1/REPORT_KO.md)는 고정 pose의 투영점과 관측 q9 사이의 x/y 방향 18개 열이 기존 입력의 선형 열공간과 중복되지 않음을 확인했습니다. 그 감사는 성능 개선이나 표현 충분성의 증거가 아니며 새 학습도 하지 않았습니다.','',
        '이번에는 그 감사에서 고정한 방향18과 정규화만 추가했습니다. 투영값에서 관측값을 빼고 bbox 대각선으로 나누며, 기존 R0 TRAIN 유효5,194후보의 float32 평균·표준편차를 그대로 사용합니다. 다시 정규화를 선택하거나 RBF를 학습하지 않았습니다. 이미 준비된 source 좌표에 추가 padding도 하지 않습니다.','',
        '```text',
        'base253_c = phi253(candidate_c) − phi253(R0 GEO anchor)',
        'd18_c = flatten((project(frozen_pose_c, K) − observed_q9) / bbox_diagonal)',
        'extra18_c = float64(normalize_FP32(d18_c)) − float64(normalize_FP32(d18_anchor))',
        'x271_c = concatenate(base253_c, extra18_c)',
        'e_c = ((T_c − T_anchor)/sT, (R_c − R_anchor)/sR)',
        'y_c = sign(e_c) * log1p(abs(e_c)); prediction_c = x271_c @ W271x2',
        'J = mean_all_2598_frames(mean_original_valid_candidates_and_2_axes(',
        '      Huber(prediction−y, delta=1) + 1{y != 0} softplus(−sign(y)*prediction)))',
        '    + (1e−4/2) * ||W||F²',
        'runtime_score = max(predicted_T_axis, predicted_R_axis)',
        'whole_pose_choice = old_tie_argmin(runtime_score over original valid candidates)',
        '```','',
        '기존 raw94·context189·고정RBF64·원래 253차원 차분·타깃·scale·valid mask·TRAIN2,598행은 유지했습니다. 이전 6개 해시는 원래 253차원 의미와 값을 보존하고, 방향 원값·방향 차분·확장 입력의 해시3개를 추가했습니다. invalid 후보와 전체 실패1행은 제거하지 않으며 anchor 입력·타깃은 정확히0입니다. sign 계수1, bias0, λ1e−4, zero 초기화, 최대1,000승인 반복/2,000objective 호출, Armijo 및 두 수렴 조건도 바꾸지 않았습니다.','',
        '추론은 정답·margin·safe mask를 읽지 않습니다. 양 축 예측의 최댓값으로 원래 whole pose 하나를 선택하며 anchor에 새 동률 우선권을 주지 않습니다. 예측상 비악화는 실제 T/R 비악화 보장이 아닙니다.','',
        '## 실제 학습과 수렴','',
        '| 모델 | objective 호출 | 승인 반복 | 최종 J | Huber | Sign logistic | L2 penalty | gradient Linf | gap 상계 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        fit=receipts[model];r=training['models'][model]['independent_recompute']
        lines.append(f"| {model} | {fit['objective_calls']} | {fit['iterations']} | {r['objective']:.12f} | {r['Huber']:.12f} | {r['Sign_logistic']:.12f} | {r['L2_penalty']:.12f} | {r['gradient_linf']:.9g} | {r['certified_gap_upper_bound']:.9g} |")
    lines+=['','마지막 승인점에서 `||gradient||∞ ≤ 1e−8`와 `||gradient||²/(2λ) ≤ 1e−6`를 함께 요구했습니다. 모든 초기·trial 호출을 기록했고 거절 trial을 마지막 checkpoint로 쓰지 않았습니다. 이 인증은 고정 목적식의 float64 수치 검산이며 구간 연산의 절대 증명이나 T/R 성능 인증은 아닙니다.','',
        '| 모델 | 직전253 가중치+zero18의 동일 J | 현재271 가중치의 동일 J | 이전−현재 |',
        '|---|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        row=data['same_new_objective_by_model'][model]
        lines.append(f"| {model} | {row['old_weight']:.12f} | {row['new_weight']:.12f} | {row['decrease']:.12f} |")
    lines+=['','이 표는 동일한 loss·타깃·frame 분모에 직전253 가중치와 zero18을 넣은 값과 현재 값을 비교합니다. 이전 네이티브253의 수렴 인증과 확장271의 gradient는 구분합니다. 이전 가중치는 독립 사후 비교에만 사용했고 학습 초기화나 새 정책 탐색에는 쓰지 않았습니다.','',
        '![전체 objective 호출과 수렴 인증](figures/training_huber_and_certificate.png)','',
        '## TRAIN 회귀와 실제 선택 진단','',
        '| 모델 | nonanchor T MAE | T RMSE | T 부호 정확도 | nonanchor R MAE | R RMSE | R 부호 정확도 | nonanchor 후보 수 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        r=training['models'][model]['regression_statistics']['nonanchor_valid_candidates'];t,r0=r['axes']['T'],r['axes']['R']
        lines.append(f"| {model} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {t['sign_accuracy']:.6f} | {r0['MAE']:.6f} | {r0['RMSE']:.6f} | {r0['sign_accuracy']:.6f} | {r['candidate_pairs']} |")
    lines+=['','MAE/RMSE는 signed-log1p 정규화 공간의 값이며 cm/도 단위가 아닙니다. 정의상 정확히0인 anchor는 위 회귀 표에서 제외했습니다. 전체 후보·실제 선택 후보의 별도 통계와 부호 confusion matrix는 [TRAIN_CONVERGENCE.json](TRAIN_CONVERGENCE.json)에 있습니다.','',
        '| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | unsafe 이전→현재 | 안전 개선 이전→현재 | anchor 이전→현재 | 최대 정규화 초과 이전→현재 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        item=training['models'][model];old=item['previous_fixed_sign'];cells=[]
        for axis in ('T','R','either'):
            cells.append(f"{old['statistics']['anchor_violations'][axis]['count']}→{item['statistics']['anchor_violations'][axis]['count']}")
        for group in ('safe_improvement','anchor'):
            cells.append(f"{old['risk_statistics']['classes'][group]['count']}→{item['risk_statistics']['classes'][group]['count']}")
        cells.append(f"{old['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}→{item['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}")
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    lines+=['','unsafe는 실제 선택의 T 또는 R가 원래 anchor보다 커진 행입니다. 유효2,597행과 실패1행을 구분하며, TRAIN 개선이나 평균 loss 감소만으로 source·실사 안정성을 대신하지 않습니다.','',
        '![TRAIN 두 축 회귀와 anchor 위반](figures/train_axis_regression_and_violations.png)','',
        '## source VAL 전체 결과','',
        f"현재 **{source['checks_passed']}/{source['checks_total']}**, 직전253은 **{previous['checks_passed']}/{previous['checks_total']}**입니다. 각 모델의 원래1,024행을 유지했습니다. 서로 다른 입력으로 새로 학습한 R0_ONLY도 비교군으로 바뀌므로 pass 개수만으로 전체 성능 우열을 정하지 않습니다. 고정 R0_GEO와 세 DIVERSE_GEO의 수치는 직전과 동일함을 확인했습니다.",'',
        '| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |',
        '|---|---:|---:|---:|---:|---:|']
    for model,summary in source['summaries'].items():lines.append(metric_row(model,summary))
    lines+=['','각 UNION seed는 새 R0_ONLY·고정 R0_GEO·paired DIVERSE_GEO와 비교합니다. 두 중앙값은 엄격히 감소, 각 P90은1.05배 이하, 실패는 증가하지 않아야 하며 총45조건 모두를 요구합니다. 평균 또는 가장 좋은 seed로 실패 조건을 대체하지 않습니다.','',
        '![직전253과 현재271의 source T/R](figures/source_val_method_comparison.png)','']
    if source['failed_checks']:
        lines+=['실패 조건은 다음과 같습니다.','']
        for item in source['failed_checks']:lines.append('- `'+(item if isinstance(item,str) else json.dumps(item,ensure_ascii=False,sort_keys=True))+'`')
    else:lines+=['source 45조건을 모두 통과했습니다. 이 사실만으로 실사 공동 개선 성공을 선언하지 않습니다.']
    if diagnostic is not None:
        models=[diagnostic['models'][f'UNION_s{s}'] for s in (1,2,3)]
        changed='/'.join(str(m['changed_identity_from_previous_sign']) for m in models)
        oldsafe='/'.join(str(m['previous_sign_actual_selection']['classes']['safe_improvement']) for m in models)
        newsafe='/'.join(str(m['actual_source_selection']['classes']['safe_improvement']) for m in models)
        oldunsafe='/'.join(str(m['previous_sign_actual_selection']['classes']['unsafe']) for m in models)
        newunsafe='/'.join(str(m['actual_source_selection']['classes']['unsafe']) for m in models)
        riskold=models[2]['previous_sign_actual_selection']['unsafe_normalized_excess']['maximum']
        risknew=models[2]['actual_source_selection']['unsafe_normalized_excess']['maximum']
        lines+=['',f"[고정 source 선택 진단](SOURCE_FIXED_CHOICE_DIAGNOSTIC_KO.md)에서 UNION seed1/2/3의 선택은 직전 대비 {changed}행 바뀌었습니다. anchor 대비 두 축이 모두 비악화이고 한 축 이상이 개선된 safe 선택은 {oldsafe}→{newsafe}, 한 축 이상 악화된 unsafe 선택은 {oldunsafe}→{newunsafe}였습니다. unsafe 수는 세 seed에서 줄었지만 seed2의 safe도 줄었고, seed3의 unsafe 정규화 초과 최댓값은 {riskold:.6f}→{risknew:.6f}로 커졌습니다. 이는 안정적 공동 개선 성공이 아닙니다.",'',
            '이 비교는 이미 고정한 선택·오류·축 예측의 집계입니다. 모든 계수를 함께 다시 학습했으므로 바뀐 선택을 추가18계수만의 인과효과로 분리하지 않습니다. 부분 후보에서 찾은 기회 하한을 전체 후보 oracle로 해석하지 않으며, 진단의 후속 목적식 제안은 미실행입니다. 진단 산출물 저장 후 마지막 출력 로그가 파일 접근 제한으로 중단된 경위와 산출물 검증은 [실행 기록](EXECUTION_KO.md)에 보존합니다.']
    if real is None:
        lines+=['','## 실사 미실행과 역사적 R0 입력 사진','',
            '**현재271 모델의 실사 T/R 효과는 미측정입니다.** [미실행 영수증](REAL_EVALUATION_NOT_RUN_KO.md)은 current learned route·실사 metric 산출물의 부재를 확인합니다.','',
            '아래 실제 RGB6장은 이전에 고정했던 동일한 입력 예시이며 **빨간 pose와 T/R는 역사적 R0 결과만** 표시합니다. 현재 모델의 성능 그림이 아닙니다. 과거 촬영별 R0 T가 가장 컸던 사례이므로 대표 표본도 아닙니다.','',
            '**110 × 11 × 130 cm는 물리 Width110 / Height11 / Depth130 cm**입니다. 원본 이미지·물리 치수·기존 K를 유지하고, W/D 가설의 camera-facing `cf_extents`와 원래 치수를 구분했습니다.']
        prefix='baseline_input_rgb_dimensions'
    else:
        lines+=['','## 실제 실사 결과와 고정 안정성 조건','',
            '자연99·clean29·wood45를 합한173행과13모델을 모두 유지했습니다. 학습4모델의692개 whole-pose 선택을 참조 접근 전에 잠갔으며, 원래 SINGLE251/R0/PRIOR1/FULL125 기준과 matched intervention 기준을 범주별 AND로 요구합니다. wood45는 별도 stress 집합입니다.','',
            '| 안정성 범주 | matched | 원래 goal | 필수 AND |','|---|---|---|---|']
        for key,gate in real['stability']['gates'].items():
            flags=[real['matched_intervention_stability']['gates'][key]['PASS'],real['original_goal_stability']['gates'][key]['PASS'],gate['PASS']]
            lines.append('| '+key+' | '+' | '.join('PASS' if flag else 'FAIL' for flag in flags)+' |')
        for pop in ('NATURAL99','CLEAN29','WOOD45'):
            lines+=['',f'### {pop}','','| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
            for model in real['models']:lines.append(metric_row(model,real['summaries'][pop][model]))
        lines+=['','![실사 전체13모델](figures/real_all_models.png)','',
            '![원래 조건과 matched 조건의 AND](figures/real_stability_gates.png)','',
            '촬영과 training seed를 교차한 paired bootstrap2,000회, 촬영별 제외 민감도와 CI는 [REAL_RESULTS.json](REAL_RESULTS.json), [REAL_DETAILED_COMPARISONS.json](REAL_DETAILED_COMPARISONS.json)에 보존했습니다. 아래는 이전과 동일한6장의 R0와 현재 UNION 세 seed입니다. 예시를 현재 오차로 다시 선정하지 않았습니다. [전체2,249행 CSV](REAL_FRAME_RESULTS.csv)도 함께 확인할 수 있습니다.']
        prefix='learned_rgb_dimensions'
    for page in range(1,4):lines+=['',f'![실제 RGB와 원본 치수 {page}](figures/{prefix}_{page}.jpg)']
    lines+=['','[GALLERY_SELECTION.json](GALLERY_SELECTION.json)에 원본 RGB SHA·크기·치수·K·pose·투영좌표를 저장했습니다. 보고서는 저장된 pose를 표시하며 새 PnP·이미지 추론·물리 오차 계산을 하지 않습니다.','',
        '## 검산·자료·한계','',
        '- [설계](DESIGN_KO.md) · [학습 계약](TRAIN_PROTOCOL.json) · [사전 검산](PREFIT_REVIEW_KO.md) · [실행 기록](EXECUTION_KO.md)',
        '- [독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [독립 source 검산](SOURCE_VAL_VERIFICATION_KO.md)',
        f"- [source 전체8,192행](SOURCE_VAL_FRAME_RESULTS.csv) · [45개 조건](SOURCE_VAL_CHECKS.csv) · [objective {len(traces)}행](TRAINING_OBJECTIVE_LOG.csv) · [승인 반복 {len(iterations)}행](TRAINING_ITERATION_LOG.csv)",
        '- [271×2 최종 파라미터4개](model_parameters/) · [그림·수치·SHA](REPORT_DATA.json) · [공개 검산](PUBLIC_REVIEW_KO.md) · [공개 manifest](PUBLICATION_MANIFEST.json)',
        '- [학습 코드](../../../scripts/research/pallet_pose_signed_axes_direction_20261001_v1/convex_train.py) · [방향 입력 코드](../../../scripts/research/pallet_pose_signed_axes_direction_20261001_v1/direction_features.py) · [source 평가 코드](../../../scripts/research/pallet_pose_signed_axes_direction_20261001_v1/evaluate_source.py) · [실사 평가 코드](../../../scripts/research/pallet_pose_signed_axes_direction_20261001_v1/evaluate_real.py) · [보고서 코드](../../../scripts/research/pallet_pose_signed_axes_direction_20261001_v1/report.py)','',
        'source VAL과 실사 DEV는 이전 방법에서도 반복 사용했습니다. refiner의 source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건, 교사 계보의 기존 이미지9장·수동 코너38개를 독립 시험 결과로 바꾸지 않습니다. 실사 pose 참조도2D 주석·K·치수에서 유도했으며 독립 장비로 측정한6D GT가 아닙니다.','',
        '방향18은 기존 예측과 pose로 만든 입력이며 독립적인 새 RGB 관측이 아닙니다. 입력 감사의 비중복성, 강볼록 목적식의 수렴, TRAIN 회귀 변화가 전이 성공을 보장하지 않습니다. 이번 단일 비교의 결과로 표현·데이터 지원범위·감독 가운데 유일한 원인을 확정하지 않습니다.','']
    return '\n'.join(lines)



def refresh_text():
    """Attach the later fixed-choice diagnostic without regenerating any numeric artifact."""
    data=C.read(C.DOC/'REPORT_DATA.json')
    assert data['complete'] and data['feature_dim']==271
    numeric_bindings=[C.bind(C.DOC/name) for name in data['csv_rows']]
    for key in ('train_protocol','source_gate','previous_source_gate','training_verification',
                'previous_training_verification','source_verification','prefit_review','gallery'):
        C.verify(data[key])
    for binding in data['artifacts']:C.verify(binding)
    for model,pair in data['exports'].items():
        for binding in pair.values():C.verify(binding)
    source=C.read(C.ROOT/data['source_gate']['path']);previous=C.read(C.ROOT/data['previous_source_gate']['path'])
    training=C.read(C.ROOT/data['training_verification']['path'])
    receipts={m:C.read(C.DOC/f'FIT_{m}.json') for m in C.MODEL_NAMES}
    real=None
    if data['learned_real_evaluated']:
        C.verify(data['real_results']);C.verify(data['real_verification']);real=C.read(C.ROOT/data['real_results']['path'])
    else:C.verify(data['real_not_run'])
    diagnostic_path=C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json';diagnostic=C.read(diagnostic_path)
    assert diagnostic['complete'] and diagnostic['PASS'] and diagnostic['no_new_selector_policy_evaluated']
    assert diagnostic['new_argmin_computations']==diagnostic['new_fits']==diagnostic['real_reference_reads']==diagnostic['new_real_routes']==0
    for binding in diagnostic['inputs'].values():C.verify(binding)
    assert C.bind(C.DOC/'SOURCE_VAL_GATE.json')==diagnostic['inputs']['current_source_val_gate']
    oldbody=(C.DOC/'REPORT_KO.md').read_text()
    assert (C.DOC/'PUBLIC_REPORT_KO.md').read_text()==oldbody
    body=render_report(source,previous,training,receipts,[None]*data['objective_calls'],[None]*data['iterations'],real,data,diagnostic)
    assert [s for s in body.splitlines() if s.startswith('|')]==[s for s in oldbody.splitlines() if s.startswith('|')],'NUMERIC_TABLE_DRIFT'
    data['source_fixed_choice_diagnostic']=C.bind(diagnostic_path)
    data['code']=C.bind(__file__)
    data['text_refresh_reason']='Attach the subsequently completed fixed-choice diagnostic; original figures, numeric CSV exports and checkpoints stay byte-identical.'
    for name in ('REPORT_KO.md','PUBLIC_REPORT_KO.md'):(C.DOC/name).write_text(body)
    (C.DOC/'REPORT_DATA.json').write_text(json.dumps(C.clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for binding in data['artifacts']:C.verify(binding)
    for binding in numeric_bindings:C.verify(binding)
    for pair in data['exports'].values():
        for binding in pair.values():C.verify(binding)
    print(json.dumps(dict(PASS=True,report=C.bind(C.DOC/'REPORT_KO.md'),report_data=C.bind(C.DOC/'REPORT_DATA.json'),
        numerical_tables_unchanged=True,numeric_exports_unchanged=True,figures_unchanged=True,checkpoints_unchanged=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-text',action='store_true');args=parser.parse_args()
    refresh_text() if args.refresh_text else main()
