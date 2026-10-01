"""Export the fixed-Newton control only after root authorizes completed results.

No fitting, candidate selection, PnP or metric calculation. Certified training,
source validation and actual real evaluation remain separate stages. A failed
source gate permits historical R0 input illustrations only.
"""
from . import common as C
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


def plots(source,previous,training,traces,receipts,old_rejected,real):
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
        calls=[r['call'] for r in rows];j=[r['objective'] for r in rows];h=[r['Huber'] for r in rows]
        gap=[r['gradient_gap_upper_bound']/1e-6 for r in rows];linf=[r['gradient_linf']/1e-8 for r in rows]
        values[f'train_{model}']=dict(calls=calls,objective=j,Huber=h,gap_ratio_to_limit=gap,linf_ratio_to_limit=linf,
            armijo_accepted=[r['armijo_accepted'] for r in rows],final_accepted_call=receipts[model]['final_accepted_call'])
        ax=axes[0,col];ax.plot(calls,j,label='Same signed-axis Huber + ridge',color='#0d9488');ax.plot(calls,h,label='Huber data term',color='#d97706',linestyle='--')
        if model=='R0_ONLY':ax.axhline(old_rejected['certificate']['objective_value'],color='#64748b',linestyle=':',label='Previous rejected R0 J (same loss)')
        ax.set_title(model);ax.set_yscale('log');ax.set_ylabel('Full TRAIN objective /2598 (log scale)');ax.grid(alpha=.2)
        ax=axes[1,col];ax.plot(calls,np.maximum(gap,1e-30),label='Gap bound /1e-6',color='#6366f1');ax.plot(calls,np.maximum(linf,1e-30),label='Gradient Linf /1e-8',color='#d97706')
        ax.axhline(1.,color='#dc2626',linestyle='--',label='Both must be <=1')
        ax.set_yscale('log');ax.set_xlabel('Every objective call, including rejected trials');ax.set_ylabel('Ratio to fixed certificate limit (log)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Fixed block Newton + Armijo | same targets/features/Huber/ridge/runtime, fresh zero initialization\nEvery initial/trial evaluation retained; final checkpoint is the last accepted certified point. No previous rejected-weight warm start.')
    finish(fig,'training_huber_and_certificate.png')
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,(axis,q,title) in zip(axes.flat,SPECS):
        old=[previous['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        new=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        fixed=source['summaries']['R0_GEO']['full_population'][axis][q]
        values[f'source_{axis}_{q}']=dict(previous_rbf_margin=old,current_signed_newton=new,fixed_R0_GEO=fixed)
        for offset,arr,color,label in [(-.2,old,'#64748b','Previous certified RBF margin CE'),(.2,new,'#0d9488','Current signed-axis Huber / Newton')]:
            shown=[v if v is not None and np.isfinite(v) else 0. for v in arr]
            ax.bar(x+offset,shown,.4,color=color,label=label)
            for j,v in enumerate(arr):ax.text(j+offset,shown[j],finite_text(v,3),ha='center',va='bottom',fontsize=8)
        ax.axhline(fixed,color='#a855f7',linestyle=':',label='Fixed R0 GEO')
        if q=='P90':ax.axhline(fixed*1.05,color='#dc2626',linestyle='--',label='Fixed R0 GEO x1.05')
        vmax=max(v for v in old+new+[fixed*1.05] if v is not None and np.isfinite(v))
        ax.set_ylim(0,max(vmax,1e-9)*1.28);ax.set_xticks(x,labels);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle(f"Source VAL1024 | {source['checks_passed']}/{source['checks_total']} checks pass\nPrevious L-BFGS signed-axis attempt had no source evaluation. Prior RBF is a different objective/decision method.")
    finish(fig,'source_val_method_comparison.png')
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,axis in zip(axes[:2],['T','R']):
        mae=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['MAE'] for m in C.MODEL_NAMES]
        rmse=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['RMSE'] for m in C.MODEL_NAMES]
        values[f'train_nonanchor_{axis}']=dict(MAE=mae,RMSE=rmse)
        ax.bar(x-.2,mae,.4,label='MAE',color='#0d9488');ax.bar(x+.2,rmse,.4,label='RMSE',color='#d97706')
        ax.set_title(f'{axis}: signed-log1p residual\nValid nonanchor candidates only',fontsize=10);ax.set_ylabel('Transformed normalized error, not cm/deg');ax.set_xticks(x,labels);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    before=[training['models'][m]['previous_fixed_rbf']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    after=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    values['train_unsafe_count']=dict(previous_rbf_margin=before,current_signed_newton=after)
    axes[2].bar(x-.2,before,.4,label='Previous RBF margin',color='#64748b');axes[2].bar(x+.2,after,.4,label='Signed Huber / Newton',color='#0d9488')
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
    prior_path=C.RBF_DOC/'REPORT_DATA.json';prior=C.read(prior_path)
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
            dims=' x '.join(f'{100*d:g}' for d in old['dimensions_m'])
            ax.set_title(f"{old['recording']} | HISTORICAL R0 INPUT BASELINE ONLY\nDimensions: {dims} cm | {saved['hypothesis']}\nHistorical R0 T={saved['T_cm']:.2f} cm, R={saved['R_deg']:.2f} deg\nCurrent signed-axis Newton model: NOT EVALUATED",fontsize=10)
            ax.axis('off')
            details.append(dict(id=fid,recording=old['recording'],image=old['image'],image_hw=old['image_hw'],K=old['K'],dimensions_m=old['dimensions_m'],
                corner_signs=signs.tolist(),edges=edges,current_learned_real_evaluated=False,
                display_selection='Fixed previous six IDs; largest historical R0 T per natural recording, not a representative sample.',
                panels=[dict(model='R0',parent='R0',hypothesis=saved['hypothesis'],pose=pose,corners_cf=corners,corners_camera=xyz,
                    projected_uv=uv,T_cm=saved['T_cm'],R_deg=saved['R_deg'],drawn_edges=drawn,
                    pose_source='Historical operational R0 input illustration only; not the current signed-axis Newton model or a new performance calculation.')]))
        fig.suptitle('Historical operational R0 input examples ONLY\nCurrent signed-axis Newton model real evaluation NOT RUN\nNo current learned real routes or T/R evaluation\nSingle RGB + pallet dimensions + existing calibrated K',fontsize=13,y=.985)
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
                assert pose['available'] and metric['available']
                corners = signs * np.asarray(pose['cf_extents'])/2
                xyz = corners @ np.asarray(pose['R_cf']).T + np.asarray(pose['centroid'])
                camera = xyz @ np.asarray(row['K']).T
                assert np.isfinite(camera).all() and (camera[:,2] != 0).all()
                uv = camera[:,:2]/camera[:,2:3]
                ax = axes[rowidx,col]
                ax.imshow(rgb)
                drawn = []
                for i,j in edges:
                    if xyz[[i,j],2].min() > 0:
                        ax.plot(uv[[i,j],0],uv[[i,j],1],color='#ef4444' if model=='R0' else '#22d3ee',lw=2.1)
                        drawn.append([i,j])
                ax.set_xlim(-.5,rgb.shape[1]-.5)
                ax.set_ylim(rgb.shape[0]-.5,-.5)
                dims=' x '.join(f'{100*d:g}' for d in row['xyz'])
                ax.set_title(f"{row['recording']} | {model}\nT={metric['translation_cm']:.2f} cm, R={metric['rotation_deg']:.2f} deg\nDimensions: {dims} cm\n{parent} / {hyp}",fontsize=10)
                ax.axis('off')
                detail['panels'].append(dict(model=model,parent=parent,hypothesis=hyp,pose=pose,
                    corners_cf=corners.tolist(),corners_camera=xyz.tolist(),projected_uv=uv.tolist(),
                    T_cm=metric['translation_cm'],R_deg=metric['rotation_deg'],drawn_edges=drawn,
                    pose_source='Frozen operational R0 or current learned whole-pose choice; direct K projection; no new PnP.'))
            details.append(detail)
        fig.suptitle('Actual current signed-axis Newton real-image results | same RGB + dimensions + calibrated K\nR0 and all three UNION seeds; no reference-selected oracle or ground-truth outline\nFixed prior six-frame diagnostic examples, not a representative performance sample',fontsize=15,y=.98)
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
    """This completed phase has four certified fits and an actual source FAIL."""
    protocol=C.protocol('TRAIN_PROTOCOL')
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json');training=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    source=C.read(C.DOC/'SOURCE_VAL_GATE.json');source_review=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    diagnostic=C.read(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json')
    not_run_path=C.DOC/'REAL_EVALUATION_NOT_RUN.json';not_run=C.read(not_run_path)
    previous=C.read(C.RBF_DOC/'SOURCE_VAL_GATE.json');old_rejected=C.read(C.SIGNED_DOC/'REJECTED_R0_ONLY.json')
    assert complete['complete'] and complete['fit_count']==4 and complete['all_certified']
    assert training['complete'] and training['PASS'] and training['independent_check_PASS']
    assert source_review['complete'] and source_review['PASS']
    assert diagnostic['complete'] and diagnostic['PASS'] and diagnostic['no_new_selector_policy_evaluated']
    assert diagnostic['new_argmin_computations']==diagnostic['new_fits']==diagnostic['real_reference_reads']==diagnostic['new_real_routes']==0
    assert C.bind(C.DOC/'SOURCE_VAL_GATE.json') in diagnostic['bindings']
    assert diagnostic['source_gate']==dict(PASS=source['PASS'],passed=source['checks_passed'],total=source['checks_total'],failed_checks=source['failed_checks'])
    assert source['complete'] and not source['PASS'] and source['checks_total']==45
    assert not source['real_routing_authorized'] and not source['real_reference_accessed']
    assert not_run['complete'] and not_run['status']=='NOT_RUN_SOURCE_GATE_FAILED' and not_run['learned_real_routes']==0
    for path,absent in not_run['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
    assert not (C.DOC/'REAL_RESULTS.json').exists() and not (C.DOC/'REAL_FRAME_RESULTS.csv').exists()
    C.verify(source['metrics'])
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        ids,models=z['ids'].tolist(),z['models'].tolist();errors={m:z[m].copy() for m in models}
    lock=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');C.verify(lock['choices']);choices=C.read(C.ROOT/lock['choices']['path'])
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
        log=[json.loads(s) for s in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        calls=[dict(model=model,**r) for r in log if r['event']=='objective'];its=[dict(model=model,**r) for r in log if r['event']=='iteration']
        assert len(calls)==fit['objective_calls'] and len(its)==fit['iterations']
        assert calls[-1]['call']==fit['final_accepted_call'] and calls[-1]['armijo_accepted']
        traces.extend(calls);iteration_rows.extend(its)
        path=C.DOC/'model_parameters'/f'{model}.json';publish(path,(C.ROOT/fit['checkpoint']['path']).read_bytes())
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(path))
    assert len(traces)==complete['total_objective_calls'] and len(iteration_rows)==complete['total_iterations']
    publish(C.DOC/'model_parameters'/'README.md','# 수렴 인증된 최종 학습 가중치4개\n\n네 JSON은 새 zero 초기화에서 실제 학습한 마지막 승인 checkpoint의 byte 동일 사본입니다. Huber 목적함수의 수치 수렴 인증은 통과했습니다. 직전 단계의 거부 optimizer 상태와는 다릅니다.\n\nsource 검증은43/45로 실패하여 실사 평가는 실행하지 않았습니다. 이 파일은 운영 성능 검증이 완료된 모델이라는 뜻이 아닙니다.\n\n[상세 결과](../REPORT_KO.md), [TRAIN 검산](../TRAIN_CONVERGENCE_KO.md), [source 판정](../SOURCE_VAL_GATE.json), [실사 미실행](../REAL_EVALUATION_NOT_RUN_KO.md)\n')
    csv_counts={name:csv_export(name,rows) for name,rows in [('SOURCE_VAL_FRAME_RESULTS.csv',val_rows),('SOURCE_VAL_CHECKS.csv',checks),
        ('TRAINING_OBJECTIVE_LOG.csv',traces),('TRAINING_ITERATION_LOG.csv',iteration_rows)]}
    assert csv_counts['SOURCE_VAL_FRAME_RESULTS.csv']==8192 and csv_counts['SOURCE_VAL_CHECKS.csv']==45
    artifacts,figure_values=plots(source,previous,training,traces,receipts,old_rejected,None)
    galleries,illustrations=gallery_baseline(not_run_path);artifacts.extend(galleries)
    current_r0=receipts['R0_ONLY']['certificate'];old_r0=old_rejected['certificate'];objective_delta=old_r0['objective_value']-current_r0['objective_value']
    old_start=C.read(C.ROOT/old_rejected['START']['path']);new_start=C.read(C.ROOT/receipts['R0_ONLY']['START']['path'])
    assert old_start['initial_weight_sha']==new_start['initial_weight_sha']
    assert old_rejected['final_weight_sha']!=new_start['initial_weight_sha']
    data=dict(complete=True,status='CERTIFIED_TRAIN_SOURCE_GATE_FAILED',actual_new_fits=4,actual_training_attempts=4,certified_models=4,
        source_VAL_checks_passed=source['checks_passed'],source_VAL_checks_total=source['checks_total'],source_VAL_evaluated=True,
        learned_real_evaluated=False,current_real_TR_effect_measured=False,stable_joint_improvement_achieved=False,goal_complete=False,method_success=False,
        code=C.bind(C.HERE/'report.py'),train_protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),source_gate=C.bind(C.DOC/'SOURCE_VAL_GATE.json'),
        previous_source_gate=C.bind(C.RBF_DOC/'SOURCE_VAL_GATE.json'),training_verification=C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
        source_verification=C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),real_not_run=C.bind(not_run_path),
        source_transfer_diagnostic=C.bind(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json'),
        prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),design=C.bind(C.DOC/'DESIGN_KO.md'),
        previous_rejected_fit=C.bind(C.SIGNED_DOC/'REJECTED_R0_ONLY.json'),previous_rejected_verification=C.bind(C.SIGNED_DOC/'REJECTED_FIT_VERIFICATION.json'),
        prior_RBF_report=C.bind(C.RBF_DOC/'REPORT_DATA.json'),basis=C.bind(C.RBF_DOC/'RBF_BASIS.json'),exports=exports,csv_rows=csv_counts,
        objective_calls=len(traces),iterations=len(iteration_rows),fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        rejected_Armijo_trials=sum(1 for r in traces if r['phase']=='trial' and not r['armijo_accepted']),
        previous_rejected_R0_J=old_r0['objective_value'],current_certified_R0_J=current_r0['objective_value'],R0_same_objective_decrease=objective_delta,
        zero_initialization_verified=True,previous_rejected_weight_warmstart=False,
        artifacts=artifacts,figure_values=figure_values,actual_RGB_images=6,illustrations=illustrations,
        gallery=C.bind(C.DOC/'GALLERY_SELECTION.json'),baseline_panels=6,current_learned_real_panels=0,
        new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0,
        report_scope='Only export frozen certified TRAIN/source diagnostics and project historical R0 poses; no optimizer, routing, quality or PnP recomputation.')
    fail_count=source['checks_total']-source['checks_passed']
    lines=['# Signed T/R Newton: 수렴은 통과, source 검증은 실패','',
        f"**네 모델의 수렴 인증은 통과했지만, source 검증은 {source['checks_total']}개 중 {source['checks_passed']}개 통과·{fail_count}개 실패입니다.** 사전 조건을 모두 만족하지 않아 현재 모델의 실사 선택·T/R 평가를 실행하지 않았습니다. 안정적인 T·R 공동 개선 목표는 여전히 미달성입니다.",'',
        f"이번 실제 학습은 **새로운 zero 초기화4회**, objective 호출{len(traces)}회·승인된 Newton 반복{len(iteration_rows)}회입니다. [직전 L-BFGS 시도](../pallet_pose_signed_axes_20261001_v1/REPORT_KO.md)의 거부 파라미터로 이어서 학습하지 않았습니다. 최적화기의 수렴 문제를 해결했다는 증거와, source·실사 정확도 검증은 분리합니다.",'',
        '운영 입력은 **단일 RGB + 팔레트 치수 + 기존 카메라 보정 K**입니다. 시간 정보·새 센서·새 실사 정답은 추가하지 않았습니다. 아래6장은 동일한 원본 실제 입력을 보여주는 이전 R0 예시이며 이번 모델의 실사 결과가 아닙니다.','',
        '## 변경 범위: 목적함수는 같고 최적화기만 변경','',
        '직전 signed-axis 구현의 objective·Hessian·anchor 차분·signed target·추론8개 함수가 AST 기준으로 동일함을 사전 검산했습니다. raw94 정규화, context189+고정RBF64, 이전 TRAIN 입력으로 정한 센터64개와 폭, 후보·valid mask·기존 TRAIN2598행·물리적 T/R 참조·scale·λ를 유지했습니다. 변경은 두 축의 SPD Hessian block을 푸는 generalized Newton과 고정 Armijo backtracking입니다. [설계](DESIGN_KO.md)와 [프로토콜](TRAIN_PROTOCOL.json)은 실제 fit 전에 봉인했습니다.','',
        '```text','x_c = phi253(candidate_c) − phi253(R0 GEO anchor)',
        'e_c = [(T_c − T_anchor)/sT, (R_c − R_anchor)/sR]',
        'target_c = sign(e_c) * log1p(abs(e_c))',
        'prediction_c = x_c @ W; W shape=(253,2); bias=0',
        'J = mean_all2598[mean_valid_candidates_and_2axes Huber(prediction_c−target_c, delta=1)]',
        '    + (lambda/2)||W||_F²',
        'sT=2.4636887551191258 cm; sR=1.113474019956766 deg; lambda=1e-4',
        'runtime: choose whole pose minimizing max(predicted T change, predicted R change)','```','',
        '모든 후보가 실패한1행은 데이터 손실0으로 전체2598행 분모에 남습니다. 유효·유한 쌍에만 차분을 계산하며 anchor의 target/feature difference는0입니다. 추론은 원래 모든 유효 후보를 비교하고 실제 T/R 참조·TRAIN margin·GT-safe mask를 사용하지 않습니다. anchor 예측0과 선택 점수≤0은 예측 공간의 성질이며 실제 양축 비악화를 보장하지 않습니다. 동률은 원래 R0→가설명 순서를 유지합니다.','',
        '각 Newton 방향은 두 개의253×253 SPD block을 Cholesky로 풀었습니다. Huber 잔차가 정확히±1이면 데이터 곡률0에 기존λI만 더하는 고정 generalized Hessian 규칙을 씁니다. alpha1에서 시작해0.5씩 줄이며 c1=1e−4 Armijo 조건을 적용했습니다. damping·warm start·재시작·step 탐색 정책 변경·예산 확대는 없습니다. 초기 평가와 거부된 모든 trial도2000회 한도에 포함하고 승인된 반복은1000회 이내로 제한했습니다.','',
        '## 실제 학습과 수렴 인증','',
        '| 모델 | objective 호출 | 승인 반복 | 거부 trial | Huber | J | gradient Linf | gap 상한 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        fit=receipts[model];r=training['models'][model];q=r['independent_recompute'];trace=r['newton_trace']
        lines.append(f"| {model} | {fit['objective_calls']} | {fit['iterations']} | {trace['rejected_Armijo_trials']} | {q['Huber']:.9f} | {q['objective']:.9f} | {q['gradient_linf']:.3e} | {q['certified_gap_upper_bound']:.3e} |")
    lines += ['', '모든 최종 모델은 optimizer 성공과 `gradient Linf≤1e−8`, `||gradient||²/(2λ)≤1e−6`을 만족했습니다. 최종 checkpoint는 마지막 승인 지점이며, 거부된 탐색 trial이나 가장 좋아 보이는 중간 반복을 고른 것이 아닙니다. 독립 검산은 초기/최종 목적함수·기울기·506×506 Hessian과 모든 trace의 count·Armijo 산술·선택된 지점 연결을 확인했습니다. 저장하지 않은 중간 가중치와 Newton 방향까지 독립 재계산했다는 주장은 하지 않습니다.','',
        '| 같은 R0_ONLY Huber 목적함수 비교 | 직전 L-BFGS 거부 상태 | 이번 zero-init Newton 인증 상태 |','|---|---:|---:|',
        f"| J | {old_r0['objective_value']:.15g} | {current_r0['objective_value']:.15g} |",
        f"| gradient-gap 상한 | {old_r0['gradient_l2_squared_over_2lambda']:.15g} | {current_r0['gradient_l2_squared_over_2lambda']:.15g} |",
        f"| 호출 / 반복 | {old_r0['objective_calls']} / {old_r0['iterations']} | {current_r0['objective_calls']} / {current_r0['iterations']} |",'',
        f"같은 J의 감소량은 약 **{objective_delta:.12g}**입니다. 직전 gap 상한1.81e−4가 실패했다는 사실은 실제 최적값과의 차이가1e−6보다 컸다는 증명이 아니며, 이번 작은 J 차이와 모순되지 않습니다. 수치 인증은 float64로 계산한 gradient에 근거하며, 아주 작은 보고 상한을 부동소수점 오차까지 포함하는 엄밀한 interval 최적성 증명으로 해석하지 않습니다.",'',
        '![Huber 이력과 두 인증 기준](figures/training_huber_and_certificate.png)','',
        '위 그래프는 초기 평가와 모든 Armijo trial을 포함합니다. 위 행은 큰 거부 trial과 작은 최종 J를 함께 볼 수 있도록 로그 축을 사용합니다. 아래 행은 gap과 gradient Linf를 각각 자신의 한계로 나눈 비율이므로 두 곡선 모두1 이하이어야 합니다. **이전 RBF margin CE와 이번 Huber loss의 값은 직접 비교하지 않습니다.** R0에만 표시한 이전 수평선은 동일한 signed Huber 목적함수의 거부 상태입니다.','',
        '## TRAIN의 두 축 회귀와 실제 선택','',
        '| 모델 | 비-anchor 후보 수 | T MAE | T RMSE | T 부호 정확도 | R MAE | R RMSE | R 부호 정확도 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        r=training['models'][model]['regression_statistics']['nonanchor_valid_candidates'];t,rot=r['axes']['T'],r['axes']['R']
        lines.append(f"| {model} | {r['candidate_pairs']} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {100*t['sign_accuracy']:.3f}% | {rot['MAE']:.6f} | {rot['RMSE']:.6f} | {100*rot['sign_accuracy']:.3f}% |")
    lines += ['', 'MAE/RMSE는 signed-log1p 정규화 target 공간의 값이며 cm/degree가 아닙니다. anchor는 구성상 target과 예측이 항상0이므로 이 표에서는 제외했습니다. 부호 정확도의 높은 수치만으로 드문 안전 개선 후보를 잘 찾는다고 결론내릴 수 없습니다. true negative/zero/positive별 혼동행렬과 실제 선택 후보만의 회귀 오차도 [TRAIN 원자료](TRAIN_CONVERGENCE.json)에 보존했습니다.','',
        '| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 안전 개선 이전→현재 | anchor 선택 이전→현재 |','|---|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        new=training['models'][model];old=new['previous_fixed_rbf'];parts=[]
        for axis in ('T','R','either'):parts.append(f"{old['statistics']['anchor_violations'][axis]['count']} → {new['statistics']['anchor_violations'][axis]['count']}")
        for group in ('safe_improvement','anchor'):parts.append(f"{old['risk_statistics']['classes'][group]['count']} → {new['risk_statistics']['classes'][group]['count']}")
        lines.append('| '+model+' | '+' | '.join(parts)+' |')
    lines += ['', '이전 비교 대상은 실제 인증됐던 RBF margin 모델이며, 실패한 signed L-BFGS 상태의 선택/T/R는 계산하지 않았습니다. 위반은 해당 행의 실제 선택 오차가 운영 R0 anchor보다 커진 경우입니다. 유효2597행과 실패1행을 구분하고, 전체2598행 분모도 유지합니다. 안전 개선과 한 축 이상 악화가 함께 늘 수 있으므로 개선 수 하나만 성공으로 해석하지 않습니다.','',
        '![TRAIN 두 축 회귀와 위반](figures/train_axis_regression_and_violations.png)','',
        f"## source VAL: {source['checks_passed']}/{source['checks_total']}, 전체 gate FAIL",'',
        '고정된1024행을 모두 평가했습니다. UNION 세 seed 각각을 학습된 R0_ONLY·고정 R0 GEO·짝지은 DIVERSE GEO와 비교하며 양축 중앙값 엄격 개선·P90의5% 이내 보존·실패 수 비증가를 요구합니다. 모든45개가 통과해야 실사 routing을 허용하는 기존 규칙을 유지했습니다.','',
        '| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
    for model,summary in source['summaries'].items():lines.append(metric_row(model,summary))
    lines+=['','실패 조건은 다음과 같습니다.','']
    lines.extend('- `'+item+'`' for item in source['failed_checks'])
    u3=source['summaries']['UNION_s3']['full_population'];r0=source['summaries']['R0_GEO']['full_population'];learned0=source['summaries']['R0_ONLY']['full_population']
    lines += ['',f"seed3 T 중앙값은 {u3['translation_cm']['median']:.9f}cm이며, R0_ONLY {learned0['translation_cm']['median']:.9f}cm와 R0_GEO {r0['translation_cm']['median']:.9f}cm보다 엄격히 작지 않아 두 비교가 실패했습니다. 나머지 조건이 통과해도 이 gate를 성공으로 간주하지 않습니다.",'',
        '![이전 RBF와 현재 source T/R](figures/source_val_method_comparison.png)','',
        '이전 signed L-BFGS 단계는 source 미평가였으므로 그 단계와 정확도 비교값은 없습니다. 그림의 이전 수치는 별도의 RBF margin 방법입니다. R0_ONLY도 새 목표로 다시 학습되므로 pass 수나 학습된 control 변화만으로 전체 우월성을 주장하지 않고 고정 R0_GEO를 함께 표시했습니다.','',
        '## 고정 source 선택에서 확인된 한계','',
        '[실패 원인 진단](SOURCE_TRANSFER_DIAGNOSTIC_KO.md)은 기존 고정 선택·예측·오차만 비교했으며 새 argmin 정책이나 실사 평가를 실행하지 않았습니다. 연속 회귀의 수렴과 실제 양축 보존이 같지 않다는 점을 다음 관측에서 확인할 수 있습니다.','',
        '| 모델 | 안전 개선 이전 RBF→현재 | unsafe 이전 RBF→현재 | 현재 anchor 유지 | 알려진 기회 | 놓친 기회 하한 |','|---|---:|---:|---:|---:|---:|']
    for model in ('UNION_s1','UNION_s2','UNION_s3'):
        row=diagnostic['models'][model];old=row['previous_RBF_actual_selection']['classes'];new=row['actual_source_selection']['classes'];known=row['known_safe_opportunities']
        lines.append(f"| {model} | {old['safe_improvement']} → {new['safe_improvement']} | {old['unsafe']} → {new['unsafe']} | {new['anchor']} | {known['frames_with_demonstrated_opportunity']} | {known['miss_lower_bound']} |")
    lines += ['', 'safe 개선은 양축 모두 anchor 이하이며 최소 한 축이 엄격히 좋아진 경우입니다. 개선 수와 함께 unsafe 선택도 증가했습니다. 알려진 기회·miss는 **이미 채점된 일부 후보 캐시가 증명하는 하한**이며 전체 네 후보 pool의 recall이나 전체 false-negative 수가 아닙니다. 새로운 oracle나 누락 후보의 오차를 계산하지 않았습니다.','',
        '현재 R0_ONLY는 source1024행에서 anchor1023개와 안전 개선1개를 선택했습니다. R0_GEO와 중앙값이 같아도 전체 오차 배열이 동일한 것은 아닙니다. UNION에서 W/D 분기를 바꾼 선택은 seed별2/0/1개였고 대부분 같은 분기에서 expert를 바꾼 경우입니다. 표현·감독 손실·합성 지원범위 가운데 유일한 원인을 이 관측만으로 확정할 수 없습니다.','',
        '진단 문서에는 같은 두 축 Huber에 물리 변화 sign의 logistic 보조항을 추가하는 단일 TRAIN 비교를 **미실행 제안**으로 남겼습니다. 이번 보고서의 성과가 아니며 구현·학습·새 routing을 실행하지 않았습니다. 과거 분류·회귀 혼합 및 상대 이득 학습의 음성 선행도 유지했습니다.','',
        '## 실사 평가 미실행과 같은 실제 입력 사진','',
        '**현재 Newton 모델의 실사 learned 선택·T/R 채점·5범주 안정성 판정은 미실행입니다.** [실사 미실행 영수증](REAL_EVALUATION_NOT_RUN_KO.md)에 새 REAL_PROTOCOL·선택·실사 결과의 부재를 기록했습니다. 과거 oracle나 RBF 실사 결과를 이번 모델 성능으로 재사용하지 않았습니다.','',
        '아래6장은 이전에 고정한 같은 실제 RGB와 원본 치수입니다. `110 × 11 × 130 cm`를 포함한 치수·기존 K를 보여주며, 빨간 선과 T/R는 **이전 운영 R0 baseline만** 표시합니다. 현재 모델의 실사 그림이 아닙니다. 각 자연 촬영에서 과거 R0 T 오차가 가장 컸던 예시여서 대표 표본이나 이번 성능의 근거로 볼 수 없습니다.']
    for page in range(1,4):lines+=['',f'![기존 R0 입력 RGB와 원본 치수 {page}](figures/baseline_input_rgb_dimensions_{page}.jpg)']
    lines += ['', '이미지6장·K·치수·R0 pose6개·투영좌표·이전 수치 출처는 [갤러리 원자료](GALLERY_SELECTION.json)에 연결했습니다. 새 이미지 forward·PnP·실사 참조 계산은0회입니다. 직전 [실제로 측정된 RBF 실사 결과](../pallet_pose_anchor_rbf_20261001_v1/REPORT_KO.md)도 안정적인 T·R 공동 개선은 미달성이었으며, 이번 source 실패와는 별도 기록입니다.','',
        '## 검산 자료와 해석의 범위','',
        '- [사전 설계](DESIGN_KO.md), [학습 계약](TRAIN_PROTOCOL.json), [입력·목적식·solver 독립 사전 검산](PREFIT_REVIEW_KO.md)',
        '- [TRAIN 독립 수렴·선택·회귀 검산](TRAIN_CONVERGENCE_KO.md), [source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md), [실사 미실행 확인](REAL_EVALUATION_NOT_RUN_KO.md)',
        f"- [source8192행 CSV](SOURCE_VAL_FRAME_RESULTS.csv), [45개 조건 CSV](SOURCE_VAL_CHECKS.csv), [objective{len(traces)}행 CSV](TRAINING_OBJECTIVE_LOG.csv), [승인 반복{len(iteration_rows)}행 CSV](TRAINING_ITERATION_LOG.csv)",
        '- [인증 checkpoint4개](model_parameters/), [그림·데이터 SHA 연결](REPORT_DATA.json), [공개물 독립 검산](PUBLIC_REVIEW_KO.md), [공개 SHA 목록](PUBLICATION_MANIFEST.json)',
        '- [Newton 학습 코드](../../../scripts/research/pallet_pose_signed_axes_newton_20261001_v1/convex_train.py), [source 평가 코드](../../../scripts/research/pallet_pose_signed_axes_newton_20261001_v1/evaluate_source.py), [미실행 실사 평가 코드](../../../scripts/research/pallet_pose_signed_axes_newton_20261001_v1/evaluate_real.py), [보고서 생성 코드](../../../scripts/research/pallet_pose_signed_axes_newton_20261001_v1/report.py)',
        '', 'source VAL과 실사 DEV는 이전 방법들에서 반복 사용됐고, refiner의 기존 source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건이 있어 독립 일반화 시험이라고 주장하지 않습니다. 교사 계보에는 기존 이미지9장의 수동 코너38개가 포함됩니다. 이번 새 실사 정답 학습은0개이며, 기존 실사 pose 참조는2D 주석·K·치수로 만든 것으로 독립 장비의6D 실측 정답이 아닙니다.','',
        'R0_ONLY는 결정론적 fit 하나이고 UNION 세 모델은 서로 다른 기존 frozen refiner seed의 후보를 사용합니다. 네 fit을 네 독립 데이터 반복 시험으로 해석하지 않습니다. 이번에는 동일 loss의 수렴 문제를 해결했지만 source의 모든 사전 조건을 만족하지 못했습니다. 현재 모델의 실사 T/R 효과는 측정되지 않았으며 원래 안정성 목표는 미달성입니다.','']
    publish(C.DOC/'REPORT_KO.md','\n'.join(lines));publish(C.DOC/'REPORT_DATA.json',data)
    publish(C.DOC/'README.md',f"# Signed T/R Newton — 수렴 통과, source gate 실패\n\n[상세 보고서·실제 입력 사진과 치수](REPORT_KO.md)\n\nzero 초기화4fit은 모두 수렴 인증을 통과했습니다. source는{source['checks_passed']}/45로 실패했고 현재 실사 평가는 미실행입니다. 안정적 T·R 공동 개선은 미달성입니다.\n\n[TRAIN 검산](TRAIN_CONVERGENCE_KO.md), [source 검산](SOURCE_VAL_VERIFICATION_KO.md), [실사 미실행](REAL_EVALUATION_NOT_RUN_KO.md), [공개 검증](PUBLIC_REVIEW_KO.md)\n")
    print(json.dumps(dict(PASS=True,csv_rows=csv_counts,objective_calls=len(traces),iterations=len(iteration_rows),figures=len(artifacts),
        source_checks_passed=source['checks_passed'],source_gate_PASS=False,learned_real_evaluated=False,goal_complete=False)))


if __name__=='__main__':main()
