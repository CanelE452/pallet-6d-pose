"""Export the fixed Huber-plus-sign intervention only after root authorizes completed results.

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
        values[f'train_{model}']=dict(calls=calls,objective=j,Huber=h,Sign_logistic=s,L2_penalty=penalty,previous_weight_same_new_J=training['models'][model]['previous_fixed_newton']['same_new_objective']['objective'],gap_ratio_to_limit=gap,linf_ratio_to_limit=linf,
            armijo_accepted=[r['armijo_accepted'] for r in rows],final_accepted_call=receipts[model]['final_accepted_call'])
        ax=axes[0,col];ax.plot(calls,j,label='Huber + sign logistic + ridge',color='#0d9488');ax.plot(calls,h,label='Huber data term',color='#d97706',linestyle='--')
        ax.plot(calls,s,label='Sign logistic term',color='#8b5cf6',linestyle='-.')
        ax.axhline(training['models'][model]['previous_fixed_newton']['same_new_objective']['objective'],color='#64748b',linestyle=':',label='Previous Newton weights under SAME new J')
        ax.set_title(model);ax.set_yscale('log');ax.set_ylabel('Full TRAIN objective /2598 (log scale)');ax.grid(alpha=.2)
        ax=axes[1,col];ax.plot(calls,np.maximum(gap,1e-30),label='Gap bound /1e-6',color='#6366f1');ax.plot(calls,np.maximum(linf,1e-30),label='Gradient Linf /1e-8',color='#d97706')
        ax.axhline(1.,color='#dc2626',linestyle='--',label='Both must be <=1')
        ax.set_yscale('log');ax.set_xlabel('Every objective call, including rejected trials');ax.set_ylabel('Ratio to fixed certificate limit (log)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Fixed block Newton + Armijo | one added sign-logistic term, same targets/features/runtime, fresh zeros\nEvery initial/trial evaluation retained; final checkpoint is the last accepted certified point. No previous-weight warm start; previous Huber-only J is not overlaid.')
    finish(fig,'training_huber_and_certificate.png')
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,(axis,q,title) in zip(axes.flat,SPECS):
        old=[previous['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        new=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        fixed=source['summaries']['R0_GEO']['full_population'][axis][q]
        values[f'source_{axis}_{q}']=dict(previous_newton_huber=old,current_huber_sign=new,fixed_R0_GEO=fixed)
        for offset,arr,color,label in [(-.2,old,'#64748b','Previous certified Newton / Huber only'),(.2,new,'#0d9488','Current Huber + sign / same Newton')]:
            shown=[v if v is not None and np.isfinite(v) else 0. for v in arr]
            ax.bar(x+offset,shown,.4,color=color,label=label)
            for j,v in enumerate(arr):ax.text(j+offset,shown[j],finite_text(v,3),ha='center',va='bottom',fontsize=8)
        ax.axhline(fixed,color='#a855f7',linestyle=':',label='Fixed R0 GEO')
        if q=='P90':ax.axhline(fixed*1.05,color='#dc2626',linestyle='--',label='Fixed R0 GEO x1.05')
        vmax=max(v for v in old+new+[fixed*1.05] if v is not None and np.isfinite(v))
        ax.set_ylim(0,max(vmax,1e-9)*1.28);ax.set_xticks(x,labels);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle(f"Source VAL1024 | {source['checks_passed']}/{source['checks_total']} checks pass\nSame frozen candidates and unchanged45 checks. Prior Newton used Huber only; current training adds sign logistic.")
    finish(fig,'source_val_method_comparison.png')
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,axis in zip(axes[:2],['T','R']):
        mae=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['MAE'] for m in C.MODEL_NAMES]
        rmse=[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis]['RMSE'] for m in C.MODEL_NAMES]
        values[f'train_nonanchor_{axis}']=dict(MAE=mae,RMSE=rmse)
        ax.bar(x-.2,mae,.4,label='MAE',color='#0d9488');ax.bar(x+.2,rmse,.4,label='RMSE',color='#d97706')
        ax.set_title(f'{axis}: signed-log1p residual\nValid nonanchor candidates only',fontsize=10);ax.set_ylabel('Transformed normalized error, not cm/deg');ax.set_xticks(x,labels);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    before=[training['models'][m]['previous_fixed_newton']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    after=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES]
    values['train_unsafe_count']=dict(previous_newton_huber=before,current_huber_sign=after)
    axes[2].bar(x-.2,before,.4,label='Previous Newton / Huber',color='#64748b');axes[2].bar(x+.2,after,.4,label='Huber + sign / Newton',color='#0d9488')
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
            dims=' / '.join(f'{100*d:g}' for d in old['dimensions_m'])
            ax.set_title(f"{old['recording']} | HISTORICAL R0 INPUT BASELINE ONLY\nObject Width / Height / Depth: {dims} cm | {saved['hypothesis']}\nHistorical R0 T={saved['T_cm']:.2f} cm, R={saved['R_deg']:.2f} deg\nCurrent signed-axis Huber + sign model: NOT EVALUATED",fontsize=10)
            ax.axis('off')
            details.append(dict(id=fid,recording=old['recording'],image=old['image'],image_hw=old['image_hw'],K=old['K'],dimensions_m=old['dimensions_m'],
                corner_signs=signs.tolist(),edges=edges,current_learned_real_evaluated=False,
                display_selection='Fixed previous six IDs; largest historical R0 T per natural recording, not a representative sample.',
                panels=[dict(model='R0',parent='R0',hypothesis=saved['hypothesis'],pose=pose,corners_cf=corners,corners_camera=xyz,
                    projected_uv=uv,T_cm=saved['T_cm'],R_deg=saved['R_deg'],drawn_edges=drawn,
                    pose_source='Historical operational R0 input illustration only; not the current signed-axis Huber + sign model or a new performance calculation.')]))
        fig.suptitle('Historical operational R0 input examples ONLY\nCurrent signed-axis Huber + sign model real evaluation NOT RUN\nNo current learned real routes or T/R evaluation\nSingle RGB + pallet dimensions + existing calibrated K',fontsize=13,y=.985)
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
        fig.suptitle('Actual current signed-axis Huber + sign real-image results | same RGB + dimensions + calibrated K\nR0 and all three UNION seeds; no reference-selected oracle or ground-truth outline\nFixed prior six-frame diagnostic examples, not a representative performance sample',fontsize=15,y=.98)
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
    diagnostic_path=C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json'
    diagnostic=C.read(diagnostic_path)
    assert diagnostic['complete'] and diagnostic['PASS'] and diagnostic['no_new_selector_policy_evaluated']
    assert diagnostic['new_argmin_computations']==diagnostic['new_fits']==diagnostic['real_reference_reads']==diagnostic['new_real_routes']==0
    assert C.bind(C.DOC/'SOURCE_VAL_GATE.json') in diagnostic['bindings']
    assert diagnostic['source_gate']==dict(PASS=source['PASS'],passed=source['checks_passed'],total=source['checks_total'],failed_checks=source['failed_checks'])
    previous=C.read(C.NEWTON_DOC/'SOURCE_VAL_GATE.json')
    assert complete['complete'] and complete['fit_count']==4 and complete['all_certified']
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
        '# 수렴 인증된 최종 학습 가중치4개\n\n네 JSON은 새 zero 초기화에서 실제 학습한 마지막 승인 checkpoint의 byte 동일 사본입니다. Huber+sign logistic+ridge의 수치 수렴 인증은 실제 T/R 성능 인증과 다릅니다. 이전 checkpoint에서 이어 학습하지 않았습니다.\n\n'+
        f"source {source['checks_passed']}/45. {real_status}\n\n[상세 결과](../REPORT_KO.md), [TRAIN 검산](../TRAIN_CONVERGENCE_KO.md), [source 판정](../SOURCE_VAL_GATE.json)\n")
    csv_counts={name:csv_export(name,rows) for name,rows in [('SOURCE_VAL_FRAME_RESULTS.csv',val_rows),('SOURCE_VAL_CHECKS.csv',checks),
        ('TRAINING_OBJECTIVE_LOG.csv',traces),('TRAINING_ITERATION_LOG.csv',iteration_rows)]}
    assert csv_counts['SOURCE_VAL_FRAME_RESULTS.csv']==8192 and csv_counts['SOURCE_VAL_CHECKS.csv']==45
    artifacts,figure_values=plots(source,previous,training,traces,receipts,real)
    if real is None:galleries,illustrations=gallery_baseline(not_run_path)
    else:galleries,illustrations=gallery_learned(real_protocol,real,real_routing)
    artifacts.extend(galleries)
    objective_comparison={m:dict(old_weight=training['models'][m]['previous_fixed_newton']['same_new_objective']['objective'],
        new_weight=training['models'][m]['independent_recompute']['objective'],
        decrease=training['models'][m]['same_new_objective_decrease']) for m in C.MODEL_NAMES}
    stable=bool(real is not None and real['stability']['PASS'])
    data=dict(complete=True,status='REAL_STABILITY_PASS' if stable else 'REAL_STABILITY_FAIL' if real is not None else 'CERTIFIED_TRAIN_SOURCE_GATE_FAILED',
        actual_new_fits=4,actual_training_attempts=4,certified_models=4,
        source_VAL_checks_passed=source['checks_passed'],source_VAL_checks_total=source['checks_total'],source_VAL_evaluated=True,
        learned_real_evaluated=real is not None,current_real_TR_effect_measured=real is not None,
        stable_joint_improvement_achieved=stable,goal_complete=False,method_success=stable,
        code=C.bind(C.HERE/'report.py'),train_protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),source_gate=C.bind(C.DOC/'SOURCE_VAL_GATE.json'),
        previous_source_gate=C.bind(C.NEWTON_DOC/'SOURCE_VAL_GATE.json'),training_verification=C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
        previous_training_verification=C.bind(C.NEWTON_DOC/'TRAIN_CONVERGENCE.json'),
        source_verification=C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),
        real_not_run=C.bind(not_run_path) if not_run is not None else None,
        real_results=C.bind(C.DOC/'REAL_RESULTS.json') if real is not None else None,
        real_verification=C.bind(C.DOC/'REAL_VERIFICATION.json') if real is not None else None,
        prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),design=C.bind(C.DOC/'DESIGN_KO.md'),
        prior_RBF_report=C.bind(C.RBF_DOC/'REPORT_DATA.json'),basis=C.bind(C.RBF_DOC/'RBF_BASIS.json'),exports=exports,csv_rows=csv_counts,
        objective_calls=len(traces),iterations=len(iteration_rows),fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        rejected_Armijo_trials=sum(1 for r in traces if r['phase']=='trial' and not r['armijo_accepted']),
        same_new_objective_by_model=objective_comparison,
        same_objective_comparison_scope='Both fixed weights evaluated under the new Huber+sign+ridge objective. Previous recorded Huber-only J is separate provenance.',
        zero_initialization_verified=True,previous_weight_warmstart=False,
        artifacts=artifacts,figure_values=figure_values,actual_RGB_images=6,illustrations=illustrations,
        gallery=C.bind(C.DOC/'GALLERY_SELECTION.json'),baseline_panels=6,current_learned_real_panels=18 if real is not None else 0,
        new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0,
        report_scope='Export frozen certified TRAIN/source/real outcomes and directly project already frozen poses; no optimizer, routing, quality or PnP recomputation.')
    data['source_transfer_diagnostic']=C.bind(diagnostic_path)
    body=render_report(source,previous,training,receipts,traces,iteration_rows,real,data,diagnostic)
    publish(C.DOC/'REPORT_KO.md',body)
    publish(C.DOC/'PUBLIC_REPORT_KO.md',body)
    publish(C.DOC/'REPORT_DATA.json',data)
    verdict='고정 실사 안정성 조건 통과' if stable else '실사 안정성 조건 미달' if real is not None else 'source 조건 미달·실사 미실행'
    publish(C.DOC/'README.md',f"# Huber + sign logistic — {verdict}\n\n[상세 결과·실제 RGB와 치수](REPORT_KO.md)\n\n새 zero 초기화4fit은 수렴 인증을 통과했습니다. source {source['checks_passed']}/45. {real_status}\n\n[TRAIN 검산](TRAIN_CONVERGENCE_KO.md), [source 검산](SOURCE_VAL_VERIFICATION_KO.md), [공개 검증](PUBLIC_REVIEW_KO.md)\n")
    print(json.dumps(dict(PASS=True,csv_rows=csv_counts,objective_calls=len(traces),iterations=len(iteration_rows),
        figures=len(artifacts),source_checks_passed=source['checks_passed'],source_gate_PASS=source['PASS'],
        learned_real_evaluated=real is not None,stable_joint_improvement_achieved=stable)))


def render_report(source,previous,training,receipts,traces,iterations,real,data,diagnostic):
    source_pass=source['PASS'];stable=data['stable_joint_improvement_achieved']
    if real is None:
        headline=f"수렴 인증4개는 통과했지만 source는{source['checks_passed']}/45로 실패했습니다. 사전 규칙에 따라 이번 모델의 실사 선택·T/R 평가를 실행하지 않았고, 안정적 공동 개선 목표는 미달성입니다."
    else:
        n=sum(bool(v['PASS']) for v in real['stability']['gates'].values())
        headline=f"수렴 인증4개와 source45/45를 통과한 뒤 실사173장을 평가했습니다. 실사 안정성은{n}/5범주 {'통과' if stable else '통과에 그쳐 안정적 공동 개선은 미달성'}입니다."
    lines=['# 두 축 Huber + sign logistic: 실제 학습·평가 결과','',f'**{headline}**','',
        f"실제 학습은 새 zero 초기화4회, objective 호출{len(traces)}회, 승인된 Newton 반복{len(iterations)}회입니다. 모든 초기·Armijo trial을 기록했고 이전 가중치를 warm start로 쓰지 않았습니다. 이번에 바꾼 것은 TRAIN sign 보조 손실 한 항입니다.",'',
        '운영 입력은 **단일 RGB 이미지 + 팔레트 치수 + 기존 카메라 보정 K**입니다. 시간 정보·새 센서·새 실사 정답은 추가하지 않았습니다. 수렴, source 검증, 실사 안정성은 각각 별도 판정입니다.','',
        '## 바꾼 것과 유지한 것','',
        '직전 [signed-axis Newton](../pallet_pose_signed_axes_newton_20261001_v1/REPORT_KO.md)은 네 목적식의 수렴을 인증했지만 source43/45로 실패했습니다. 고정 선택 진단에서 안전 개선과 unsafe 선택이 함께 늘었고, TRAIN에서도 두 물리 축의 부호 오류가 남았습니다. 이 관측만으로 표현력·합성 지원범위·손실 가운데 유일 원인을 확정하지 않습니다. 이번에는 연속 크기 감독을 유지하면서 운영의0 경계에 부호 감독을 더하는 한 비교를 사전에 고정했습니다.','',
        'raw94 정규화, context189+고정RBF64, 센터64개와 폭, 후보 pose·valid mask·TRAIN2598행·anchor·물리 T/R 참조와 scale, 두 축253×2 가중치·bias0·λ1e−4, Newton/Armijo 예산과 runtime은 유지했습니다. [설계](DESIGN_KO.md)·[프로토콜](TRAIN_PROTOCOL.json)은 실제 fit 전에 봉인했습니다.','',
        '```text','x_c = phi253(candidate_c) − phi253(R0 GEO anchor)',
        'e_c = [(T_c−T_anchor)/sT, (R_c−R_anchor)/sR]',
        'y_c = sign(e_c) * log1p(abs(e_c)); p_c = x_c @ W',
        'L_c,axis = Huber(p_c,axis−y_c,axis, delta=1)',
        '         + 1{y_c,axis!=0} * softplus(−sign(y_c,axis)*p_c,axis)',
        'J = mean_all2598[mean_original_valid_candidates_and_2axes L] + (1e−4/2)||W||²',
        'sign coefficient=1; W shape=(253,2); bias=0',
        'sT=2.4636887551191258 cm; sR=1.113474019956766 deg',
        'runtime: choose whole pose minimizing max(predicted T change,predicted R change)','```','',
        'target가 정확히0이면 logistic 항 전체를 제외하므로 log2 상수도 남기지 않습니다. 원래 유효 후보×2축 분모를 그대로 사용하며 nonzero sign 개수로 재정규화하지 않습니다. 모든 후보가 실패한1행은 데이터 손실0으로 전체2598행 분모에 남습니다. 계수·threshold·margin·seed를 source VAL에서 탐색하지 않았습니다.','',
        '추론 출력은 혼합 손실로 학습한 signed-axis 점수이며 cm/degree 오차나 보정된 확률이 아닙니다. 추론에는 실제 T/R 오차·sign 정답·TRAIN margin·GT-safe mask를 넣지 않습니다. anchor 예측0과 선택 점수≤0은 예측 공간의 성질이며 실제 물리오차 비증가의 증명이 아닙니다. 정확 동률은 원래 R0→가설명 순서를 유지합니다.','',
        '## 실제 수렴 인증과 손실 성분','',
        'solver는 이전과 같은 block generalized Newton + Armijo(alpha1, 반감0.5, c1=1e−4)이며 한 fit당 초기 포함 최대2000호출·승인1000반복입니다. Huber kink에서는 Huber 곡률만0이고 logistic 곡률과λI는 유지합니다. 봉인 후 목적식·규제·후보를 바꾸거나 예산을 확장·재시작하지 않고 마지막 승인 지점을 인증합니다.','',
        '| 모델 | 호출 | 승인 반복 | Huber | Sign logistic | L2 | 새 J | gradient Linf | gap 상한 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        r=training['models'][model];q=r['independent_recompute'];fit=receipts[model]
        lines.append(f"| {model} | {fit['objective_calls']} | {fit['iterations']} | {q['Huber']:.9f} | {q['Sign_logistic']:.9f} | {q['L2_penalty']:.9f} | {q['objective']:.9f} | {q['gradient_linf']:.3e} | {q['certified_gap_upper_bound']:.3e} |")
    lines+=['','모든 최종 모델에 optimizer 성공, `gradient Linf≤1e−8`, `||gradient||²/(2λ)≤1e−6`을 요구했습니다. 독립 Torch64 autograd·506×506 Hessian·scalar target 재구성은 실행 코드와 일치했습니다. 초기/모든 trial의 count·Armijo 산술·accepted 상태 연결을 확인했으며, 저장하지 않은 모든 중간 가중치까지 독립 재계산했다는 주장은 하지 않습니다. 매우 작은 float64 gap 값을 반올림 오차를 포함한 interval 인증으로 해석하지 않습니다.','',
        '| 모델 | 이전 Newton 가중치의 같은 새 J | 현재 가중치의 새 J | 감소 |','|---|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        r=data['same_new_objective_by_model'][model]
        lines.append(f"| {model} | {r['old_weight']:.9f} | {r['new_weight']:.9f} | {r['decrease']:.9f} |")
    lines+=['','위 비교는 **두 고정 가중치에 동일한 새 Huber+sign+ridge**를 평가한 값입니다. 과거 보고서에 적힌 Huber-only J와 새 J를 직접 비교하지 않습니다. 원래 Huber-only 목적식과 인증값은 TRAIN JSON에서 별도 provenance로 남겼습니다. 동일 J가 줄어도 T/R 안정성의 보장은 아닙니다.','',
        '![손실 성분과 고정 수렴 기준](figures/training_huber_and_certificate.png)','',
        '위쪽 곡선은 초기·거부된 trial까지 모두 포함하며 로그 축입니다. 회색 선은 이전 Newton 가중치를 **새 J**로 평가한 값입니다. 아래쪽은 gap과 gradient Linf를 각 한계로 나눈 비율로, 두 곡선 모두1 이하여야 합니다.','',
        '## TRAIN의 회귀 부호와 실제 선택','',
        '| 모델 | 비-anchor 후보 수 | T MAE | T RMSE | T 부호 정확도 | R MAE | R RMSE | R 부호 정확도 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        r=training['models'][model]['regression_statistics']['nonanchor_valid_candidates'];t,rot=r['axes']['T'],r['axes']['R']
        lines.append(f"| {model} | {r['candidate_pairs']} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {100*t['sign_accuracy']:.3f}% | {rot['MAE']:.6f} | {rot['RMSE']:.6f} | {100*rot['sign_accuracy']:.3f}% |")
    lines+=['','MAE/RMSE는 signed-log1p target 공간의 값으로 cm/degree가 아닙니다. 구성상 항상 정답·예측0인 anchor는 이 표에서 제외했습니다. exact0 class와 부호별 confusion·실제 선택만의 별도 회귀 통계는 JSON에 있습니다.','',
        '| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 안전 개선 이전→현재 | anchor 선택 이전→현재 |','|---|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        new=training['models'][model];old=new['previous_fixed_newton'];parts=[]
        for axis in ('T','R','either'):parts.append(f"{old['statistics']['anchor_violations'][axis]['count']} → {new['statistics']['anchor_violations'][axis]['count']}")
        for group in ('safe_improvement','anchor'):parts.append(f"{old['risk_statistics']['classes'][group]['count']} → {new['risk_statistics']['classes'][group]['count']}")
        lines.append('| '+model+' | '+' | '.join(parts)+' |')
    lines+=['','이전은 인증된 Newton Huber-only 모델이며 후보·TRAIN 데이터는 같습니다. 위반은 실제 선택 오차가 해당 frame의 운영 R0 anchor보다 커진 경우입니다. 유효2597행/실패1행을 구분하고 전체2598행 분모를 유지합니다. 안전 개선은 양축 비증가와 최소 한 축 엄격 개선을 모두 만족한 실제 선택입니다.','',
        '![TRAIN 두 축 회귀와 안전 위반](figures/train_axis_regression_and_violations.png)','',
        f"## source VAL: {source['checks_passed']}/45, 전체 {'PASS' if source_pass else 'FAIL'}",'',
        '고정1024행을 모두 유지합니다. UNION 세 seed 각각을 새 R0_ONLY·고정 R0 GEO·짝지은 DIVERSE GEO와 비교하고 양축 중앙값 엄격 개선·각축P90의5% 이내 보존·실패 수 비증가를 요구합니다. 좋은 seed를 따로 선택하지 않으며45개 모두 통과해야 실사로 진행합니다.','',
        '| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
    for model,summary in source['summaries'].items():lines.append(metric_row(model,summary))
    lines+=['','실패한 원래 조건:','']
    lines += ['- `'+item+'`' for item in source['failed_checks']] if source['failed_checks'] else ['실패 조건은0개입니다. 이것만으로 실사 안정성을 주장하지 않습니다.']
    lines+=['','![이전 Newton과 현재 source T/R](figures/source_val_method_comparison.png)','',
        f"직전 Newton은 {previous['checks_passed']}/45였고 이번은 {source['checks_passed']}/45입니다. 비교 그림은 이전 **Huber-only Newton**과 현재 Huber+sign의 실제 고정 source 결과이며, 이전 RBF나 거부 L-BFGS의 값으로 대체하지 않았습니다. R0_ONLY도 새 목적식으로 학습하므로 고정 R0 GEO를 함께 보존합니다.",'']
    if source['failed_checks']==previous['failed_checks'] and source['failed_checks']:
        old_t=previous['summaries']['UNION_s3']['full_population']['translation_cm']['median']
        new_t=source['summaries']['UNION_s3']['full_population']['translation_cm']['median']
        direction='더 커졌습니다' if new_t>old_t else '같습니다' if new_t==old_t else '작아졌지만 실패 조건은 유지됐습니다'
        lines+=[f"실패한 조건은 직전과 동일합니다. UNION_s3의 T 중앙값은 {old_t:.9f}→{new_t:.9f}cm로 {direction}. 동일한43/45를 향상으로 표현하지 않으며, source 공동 개선 조건은 여전히 미달성입니다.",'']
    lines+=['## 고정 source 선택 진단','',
        '[추가 진단](SOURCE_TRANSFER_DIAGNOSTIC_KO.md)은 이미 동결된 선택·예측·채점값만 분석했다. 새 argmin·threshold·후보 오차 재계산이나 실사 routing은 없다.','',
        '| 모델 | safe 개선 이전 Newton→현재 | unsafe 이전 Newton→현재 | 현재 anchor | 알려진 기회 | 놓친 기회 하한 |','|---|---:|---:|---:|---:|---:|']
    for model in ('UNION_s1','UNION_s2','UNION_s3'):
        row=diagnostic['models'][model];old=row['previous_Newton_actual_selection']['classes'];new=row['actual_source_selection']['classes'];known=row['known_safe_opportunities']
        lines.append(f"| {model} | {old['safe_improvement']} → {new['safe_improvement']} | {old['unsafe']} → {new['unsafe']} | {new['anchor']} | {known['frames_with_demonstrated_opportunity']} | {known['miss_lower_bound']} |")
    lines+=['','안전 개선은 두 축이 모두 anchor 이하이면서 적어도 한 축이 엄격 개선된 실제 선택이다. 개선 수와 함께 unsafe 선택도 증가하므로 개선 수 하나만 성공으로 읽지 않는다. 알려진 기회·miss는 이미 채점된 부분 후보 캐시가 증명하는 하한이며 전체 pool recall이나 전체 false-negative 수가 아니다. 데이터·표현·감독 가운데 유일한 원인을 이 관측으로 확정하지 않는다.','']
    if real is None:
        lines+=['## 실사 미실행과 실제 RGB 입력 예시','',
            '**이번 모델의 실사 learned 선택·T/R 평가·5범주 안정성은 미실행입니다.** [미실행 영수증](REAL_EVALUATION_NOT_RUN_KO.md)은 금지된 실사 산출물의 부재를 확인합니다. 과거 oracle와 다른 방법의 실사 결과를 이번 성능으로 재사용하지 않습니다.','',
            '아래6장은 이전에 고정한 실제 RGB입니다. 원본 물체의 x/y/z는 Width/Height/Depth 순서입니다. 치수 예시 **110 × 11 × 130 cm는 Width 110 / Height 11 / Depth 130 cm**를 뜻하며, 원본 dimensions와 기존 K를 사용합니다. 투영용 camera-facing `cf_extents`는 W/D 가설에 따른 축 재배열로 원본 physical dimensions와 구분됩니다. 빨간 선과 T/R는 이전 운영 **R0 입력 예시만** 표시하며 현재 모델 결과가 아닙니다. 자연 촬영별 과거 R0 T가 가장 컸던 예시여서 대표 표본이나 개선 근거가 아닙니다.']
        image_prefix='baseline_input_rgb_dimensions'
    else:
        lines+=['## 실제 실사 결과와 원래 안정성 조건','',
            '자연99·clean29·wood45의 모든173행과13개 모델을 유지했습니다. 학습된4개 모델의692개 선택은 참조 접근 전에 잠갔습니다. GT로 후보를 골라 표시하지 않으며 raw baseline DIVERSE와 SINGLE251의 정체성을 보존했습니다. 원래 SINGLE251/R0/PRIOR1/FULL125 조건과 matched intervention 조건을 범주별 AND로 모두 요구합니다. wood45는 별도 stress이고 주 목표를 대체하지 않습니다.','',
            '| 안정성 범주 | matched | 원래 goal | 필수 AND |','|---|---|---|---|']
        for key,v in real['stability']['gates'].items():
            flags=[real['matched_intervention_stability']['gates'][key]['PASS'],real['original_goal_stability']['gates'][key]['PASS'],v['PASS']]
            lines.append('| '+key+' | '+' | '.join('PASS' if f else 'FAIL' for f in flags)+' |')
        for pop in ('NATURAL99','CLEAN29','WOOD45'):
            lines+=['',f'### {pop}','','| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
            for model in real['models']:lines.append(metric_row(model,real['summaries'][pop][model]))
        lines+=['','![모든 실사 모델](figures/real_all_models.png)','',
            '![두 조건 가족과 AND](figures/real_stability_gates.png)','',
            '촬영+training seed의 paired bootstrap2000회, 촬영별 제외 민감도, 전체 실패 처리와 CI는 [원본 결과](REAL_RESULTS.json) 및 [상세 비교](REAL_DETAILED_COMPARISONS.json)에 남겼습니다. 아래는 같은 고정6장의 R0와 세 UNION 결과입니다. `110 × 11 × 130 cm`를 포함한 원본 치수·K를 사용하며 새 성능에 맞춰 예시를 고르지 않았습니다. 대표 성능은 위 전체 모집단 표로 확인해야 합니다.']
        image_prefix='learned_rgb_dimensions'
    for page in range(1,4):lines+=['',f'![실제 RGB와 원본 치수 {page}](figures/{image_prefix}_{page}.jpg)']
    lines+=['','원본 이미지 SHA·크기·치수·K·저장된 pose와 투영좌표는 [갤러리 자료](GALLERY_SELECTION.json)에 연결했습니다. 보고서는 기존 pose를 직접 투영할 뿐 새 image forward·PnP·물리 오차 계산을 하지 않았습니다.','',
        '## 검산과 한계','',
        '- [사전 설계](DESIGN_KO.md), [학습 계약](TRAIN_PROTOCOL.json), [사전 검산](PREFIT_REVIEW_KO.md)',
        '- [실행 기록과 재현 범위](EXECUTION_KO.md)',
        '- [독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md), [독립 source 검산](SOURCE_VAL_VERIFICATION_KO.md)',
        f"- [source8192행](SOURCE_VAL_FRAME_RESULTS.csv), [45개 조건](SOURCE_VAL_CHECKS.csv), [objective{len(traces)}행](TRAINING_OBJECTIVE_LOG.csv), [승인반복{len(iterations)}행](TRAINING_ITERATION_LOG.csv)",
        '- [최종 checkpoint4개](model_parameters/), [그림·원자료 연결](REPORT_DATA.json), [공개 검산](PUBLIC_REVIEW_KO.md), [SHA 목록](PUBLICATION_MANIFEST.json)',
        '- [학습 코드](../../../scripts/research/pallet_pose_signed_axes_sign_20261001_v1/convex_train.py), [source 코드](../../../scripts/research/pallet_pose_signed_axes_sign_20261001_v1/evaluate_source.py), [실사 코드](../../../scripts/research/pallet_pose_signed_axes_sign_20261001_v1/evaluate_real.py), [보고서 코드](../../../scripts/research/pallet_pose_signed_axes_sign_20261001_v1/report.py)','',
        'source VAL·실사 DEV는 이전 방법에서 반복 사용했습니다. 기존 refiner source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건이 있어 새로운 독립 일반화 시험이 아닙니다. 교사 계보에는 기존 이미지9장의 수동 코너38개가 포함됩니다. 이번 새 실사 GT 학습은0이며, 기존 실사 pose 참조는2D 주석·K·치수로 만든 것으로 독립 장비의6D 실측 GT가 아닙니다.','',
        '분류+회귀 혼합, signed2D gain, RGB MLP의 음성 선행을 유지하며 단순 loss 변경이 전이를 해결한다고 주장하지 않습니다. 네 fit은 공유 R0_ONLY 하나와 서로 다른 기존 refiner seed의 UNION 세 개이며 네 독립 촬영 반복 시험이 아닙니다. 수렴이나 TRAIN 부호 변화만으로 source·실사 안정성을 대체하지 않습니다.','']
    return '\n'.join(lines)


if __name__=='__main__':main()
