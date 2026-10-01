"""Publish completed risk-margin learning and real evaluation; no fits or metrics.

Only frozen JSON/NPZ/trace summaries are exported. RGB panels project frozen
whole poses with the existing K; no evaluator/trainer/reference parser imports.
"""
from . import common as C
import csv
import io
import itertools
import json
import numpy as np

SPECS = [('translation_cm', 'median', 'T median (cm)'),
         ('rotation_deg', 'median', 'R median (deg)'),
         ('translation_cm', 'P90', 'T P90 (cm)'),
         ('rotation_deg', 'P90', 'R P90 (deg)')]


def publish(path, value):
    payload = value if isinstance(value, bytes) else (value if isinstance(value, str) else
        json.dumps(C.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == payload, f'Existing publication differs: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(payload)


def csv_export(name, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    publish(C.DOC / name, out.getvalue())
    return len(rows)


def plots(source, previous, training, traces, real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.DOC / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    artifacts, values = [], {}
    labels, x = ['R0_ONLY', 'UNION s1', 'UNION s2', 'UNION s3'], np.arange(4)

    def finish(fig, name):
        path = folder / name
        fig.savefig(path, dpi=145)
        plt.close(fig)
        artifacts.append(C.bind(path))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (axis, q, title) in zip(axes.flat, SPECS):
        old = [previous['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        new = [source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        fixed = source['summaries']['R0_GEO']['full_population'][axis][q]
        values[f'source_{axis}_{q}'] = dict(previous_context189=old, current_margin189=new, fixed_R0_GEO=fixed)
        ax.bar(x-.2, old, .4, color='#64748b', label='Previous context189 / ordinary CE')
        ax.bar(x+.2, new, .4, color='#0d9488', label='Current context189 / risk-margin CE')
        ax.axhline(fixed, color='#a855f7', linestyle=':', label='Fixed R0 GEO')
        if axis == 'translation_cm' and q == 'P90':
            ax.axhline(1.05*fixed, color='#dc2626', linestyle='--', label='Fixed R0 GEO x 1.05')
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylim(0, max(old + new + [fixed*1.05])*1.28)
        ax.grid(axis='y', alpha=.2)
        for j in range(4):
            for dx, v in [(-.2, old[j]), (.2, new[j])]:
                ax.text(j+dx, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    axes[1, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Source VAL1024 | actual learned selectors | 45/45 checks pass\nSame features and targets; R0_ONLY is also retrained. This is not a real-image improvement claim.')
    finish(fig, 'source_val_loss_comparison.png')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, model in zip(axes.flat, C.MODEL_NAMES):
        rows = [r for r in traces if r['model'] == model]
        ax.plot([r['call'] for r in rows], [r['objective'] for r in rows], label='Margin CE + ridge', color='#0d9488')
        ax.plot([r['call'] for r in rows], [r['unadjusted_CE'] for r in rows], label='Ordinary CE (diagnostic)', color='#d97706')
        old = training['models'][model]['previous_converged_same_target']['native_unadjusted_objective']['CE']
        ax.axhline(old, color='#64748b', linestyle='--', label='Previous fit ordinary CE')
        ax.set_title(model)
        ax.set_xlabel('Objective evaluations (solver calls)')
        ax.set_ylabel('Full TRAIN loss; 2598 rows')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Four certified fits | 1453 objective calls, 1313 iterations\nMargin objective improves; ordinary CE worsens against the previous fit in every model')
    finish(fig, 'training_margin_and_unadjusted_ce.png')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    old = [training['models'][m]['previous_converged_same_target'] for m in C.MODEL_NAMES]
    new = [training['models'][m] for m in C.MODEL_NAMES]
    specs = [('target_accuracy', [d['statistics']['target_accuracy_full_population']*100 for d in old],
              [d['statistics']['target_accuracy_full_population']*100 for d in new], 'Exact target selected (%) / all2598'),
             ('unsafe_count', [d['statistics']['anchor_violations']['either']['count'] for d in old],
              [d['statistics']['anchor_violations']['either']['count'] for d in new], 'Either-axis anchor violation (frames)'),
             ('max_risk', [d['risk_statistics']['normalized_max_excess_all_available']['maximum'] for d in old],
              [d['risk_statistics']['normalized_max_excess_all_available']['maximum'] for d in new], 'Maximum normalized excess / available2597')]
    for ax, (name, before, after, title) in zip(axes, specs):
        values[f'train_{name}'] = dict(previous_context189=before, current_margin189=after)
        ax.bar(x-.2, before, .4, color='#64748b', label='Previous ordinary CE')
        ax.bar(x+.2, after, .4, color='#0d9488', label='Current risk-margin CE')
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(before+after)*1.24)
        ax.grid(axis='y', alpha=.2)
        for j in range(4):
            for dx, v in [(-.2,before[j]),(.2,after[j])]:
                ax.text(j+dx,v,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle('TRAIN only | one all-invalid failure retained separately\nFewer unsafe selections do not guarantee both-axis gains; UNION s1 R violations rise from 27 to 29')
    finish(fig, 'train_target_and_risk.png')

    models = real['models']
    fig, axes = plt.subplots(3, 4, figsize=(21, 15), constrained_layout=True)
    for row, population in enumerate(['NATURAL99', 'CLEAN29', 'WOOD45']):
        for col, (axis, q, title) in enumerate(SPECS):
            ax = axes[row, col]
            vals = [real['summaries'][population][m]['full_population'][axis][q] for m in models]
            values[f'real_{population}_{axis}_{q}'] = dict(models=models, values=vals)
            colors = ['#d97706' if m == 'R0_ONLY' else '#0d9488' if m.startswith('UNION') else '#64748b' for m in models]
            ax.barh(np.arange(len(models)), vals, color=colors)
            ax.set_yticks(np.arange(len(models)), models, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f'{population} | {title}', fontsize=11)
            ax.set_xlim(0, max(vals)*1.24)
            ax.grid(axis='x', alpha=.15)
            for j,v in enumerate(vals):
                ax.text(v,j,f' {v:.2f}',va='center',fontsize=7)
    fig.suptitle('Actual real evaluation | all13 models, all173 frames retained | lower is better\nOriginal baselines retain their identities. No best-seed selection; 0 pose failures is not 0 error.', fontsize=14)
    finish(fig, 'real_all_models.png')

    keys = list(real['stability']['gates'])
    matrix = np.asarray([[real['matched_intervention_stability']['gates'][k]['PASS'],
                          real['original_goal_stability']['gates'][k]['PASS'],
                          real['stability']['gates'][k]['PASS']] for k in keys], dtype=int)
    values['real_gate_matrix'] = dict(rows=keys, columns=['matched', 'original_SINGLE251', 'combined_AND'], values=matrix.tolist())
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(['#fee2e2', '#ccfbf1']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(3), ['Matched intervention', 'Original SINGLE251 contract', 'Required AND'])
    ax.set_yticks(range(5), keys)
    for i in range(5):
        for j in range(3): ax.text(j,i,'PASS' if matrix[i,j] else 'FAIL',ha='center',va='center')
    ax.set_title('Stable joint T/R improvement NOT established\nThree failed categories; tail and clean-preservation guards pass')
    finish(fig, 'real_stability_gates.png')
    return artifacts, values


def gallery(real_protocol, real, routing):
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
        fig.suptitle('Actual learned real-image results | same RGB + dimensions + calibrated K\nR0 and all three UNION seeds; no reference-selected oracle or ground-truth outline\nFixed prior six-frame diagnostic examples, not a representative performance sample',fontsize=15,y=.98)
        p = C.DOC/'figures'/f'learned_rgb_dimensions_{page+1}.jpg'
        fig.savefig(p,dpi=115)
        plt.close(fig)
        artifacts.append(C.bind(p))
    selection = dict(complete=True,source_selection_report=C.bind(C.ANCHOR_DOC/'REPORT_DATA.json'),
                     routing=C.bind(C.DOC/'REAL_ROUTING_LOCK.json'),metrics=mb,metadata=real_protocol['inputs']['metadata'],
                     poses=real_protocol['inputs']['poses'],actual_RGB_images=6,panels=24,illustrations=details,
                     new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0)
    publish(C.DOC/'GALLERY_SELECTION.json',selection)
    return artifacts,details


def main():
    protocol=C.protocol('TRAIN_PROTOCOL')
    real_protocol=C.protocol('REAL_PROTOCOL')
    source=C.read(C.DOC/'SOURCE_VAL_GATE.json')
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    training=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    prefit=C.read(C.DOC/'PREFIT_REVIEW.json')
    source_review=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    real_review=C.read(C.DOC/'REAL_VERIFICATION.json')
    transfer=C.read(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC.json')
    real=C.read(C.DOC/'REAL_RESULTS.json')
    routing=C.read(C.DOC/'REAL_ROUTING_LOCK.json')
    previous=C.read(C.CONTEXT_DOC/'SOURCE_VAL_GATE.json')
    assert prefit['PASS'] and training['PASS'] and training['independent_check_PASS'] and source_review['PASS']
    assert real_review['complete'] and real_review['PASS'] and transfer['complete'] and transfer['PASS']
    assert transfer['diagnostic_only'] and transfer['new_fits']==transfer['new_metric_calls']==0
    assert complete['all_certified'] and complete['fit_count']==4
    assert source['PASS'] and source['checks_passed']==source['checks_total']==45
    assert previous['checks_passed']==43 and previous['checks_total']==45
    assert real['complete'] and not real['stability']['PASS'] and real['full_frame_rows']==2249
    assert real['baseline_metric_parity_checks']==1557 and len(real['models'])==13
    assert real['routing_lock']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    assert not routing['real_reference_values_read'] and not routing['runtime_uses_margin']
    for b in real['artifacts']: C.verify(b)
    assert sum(g['PASS'] for g in real['stability']['gates'].values())==2
    assert all(s['failed_pose']==0 for pop in real['summaries'].values() for s in pop.values())
    C.verify(source['metrics'])
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        ids,models=z['ids'].tolist(),z['models'].tolist()
        errors={m:z[m].copy() for m in models}
    sr=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json'); C.verify(sr['choices'])
    choices=C.read(C.ROOT/sr['choices']['path'])
    val_rows=[]
    for model in models:
        assert errors[model].shape==(1024,2) and np.isfinite(errors[model]).all()
        for j,fid in enumerate(ids):
            pick=choices['records'].get(model,{}).get(fid,{})
            val_rows.append(dict(model=model,id=fid,split='VAL',available=True,T_cm=float(errors[model][j,0]),
                R_deg=float(errors[model][j,1]),candidate=pick.get('candidate_name','fixed_GEO'),fallback=pick.get('fallback',False)))
    checks=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,group in source['comparisons'].items()
            for b,comp in group.items() for k,v in comp['checks'].items()]
    receipts,traces,exports={},[],{}
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');receipts[model]=fit
        for key in ('START','trace','checkpoint'):C.verify(fit[key])
        assert fit['certificate']['PASS'] and fit['fits_executed']==1 and not fit['runtime_uses_margin']
        log=[json.loads(s) for s in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        traces.extend(dict(model=model,**row) for row in log if row['event']=='objective')
        path=C.DOC/'model_parameters'/f'{model}.json'
        publish(path,(C.ROOT/fit['checkpoint']['path']).read_bytes())
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(path))
        old=training['models'][model]['previous_converged_same_target'];new=training['models'][model]
        assert new['independent_recompute']['objective'] < old['objective']['objective']
        assert new['independent_recompute']['unadjusted_CE'] > old['native_unadjusted_objective']['CE']
        assert new['statistics']['target_accuracy_full_population'] < old['statistics']['target_accuracy_full_population']
    assert len(traces)==complete['total_objective_calls']==1453
    iterations=sum(r['iterations'] for r in receipts.values());assert iterations==1313
    csv_counts={name:csv_export(name,rows) for name,rows in [('SOURCE_VAL_FRAME_RESULTS.csv',val_rows),
        ('SOURCE_VAL_CHECKS.csv',checks),('TRAINING_OBJECTIVE_LOG.csv',traces)]}
    csv_counts['REAL_FRAME_RESULTS.csv']=sum(1 for _ in csv.DictReader((C.DOC/'REAL_FRAME_RESULTS.csv').open()))
    assert csv_counts=={'SOURCE_VAL_FRAME_RESULTS.csv':8192,'SOURCE_VAL_CHECKS.csv':45,'TRAINING_OBJECTIVE_LOG.csv':1453,'REAL_FRAME_RESULTS.csv':2249}
    artifacts,figure_values=plots(source,previous,training,traces,real)
    galleries,illustrations=gallery(real_protocol,real,routing);artifacts.extend(galleries)
    h=real['hierarchy']['NATURAL99']['R0']
    source_limit=source['summaries']['R0_GEO']['full_population']['translation_cm']['P90']*1.05
    data=dict(complete=True,code=C.bind(C.HERE/'report.py'),train_protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),
        real_protocol=C.bind(C.DOC/'REAL_PROTOCOL.json'),source_gate=C.bind(C.DOC/'SOURCE_VAL_GATE.json'),
        previous_source_gate=C.bind(C.CONTEXT_DOC/'SOURCE_VAL_GATE.json'),training_verification=C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
        source_verification=C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),prefit_verification=C.bind(C.DOC/'PREFIT_REVIEW.json'),
        real_verification=C.bind(C.DOC/'REAL_VERIFICATION.json'),selector_transfer_diagnostic=C.bind(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC.json'),
        real_results=C.bind(C.DOC/'REAL_RESULTS.json'),real_routing=C.bind(C.DOC/'REAL_ROUTING_LOCK.json'),
        actual_new_fits=4,objective_calls=len(traces),iterations=iterations,fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        source_VAL_checks_passed=45,source_VAL_checks_total=45,previous_source_VAL_checks_passed=43,
        failed_checks=source['failed_checks'],learned_real_evaluated=True,learned_real_choices=692,real_metric_rows=2249,
        real_models=13,real_frames=173,real_baseline_parity_rows=1557,real_oracle_diagnostic_evaluated=False,
        stable_joint_improvement_achieved=False,method_success=False,goal_complete=False,
        real_gate_PASS={k:g['PASS'] for k,g in real['stability']['gates'].items()},
        feature_map=protocol['feature_map'],raw_feature_dim=94,feature_dim=189,loss_rule=protocol['loss_rule'],runtime_uses_margin=False,
        csv_rows=csv_counts,exports=exports,artifacts=artifacts,figure_values=figure_values,illustrations=illustrations,
        gallery_selection=C.bind(C.DOC/'GALLERY_SELECTION.json'),actual_RGB_images=6,actual_learned_RGB_panels=24,
        figure_scope='All current learned results; gallery uses prior fixed IDs, never oracle routing.',
        natural_R0_hierarchy=h,seed3_source_T_P90=dict(value_cm=source['summaries']['UNION_s3']['full_population']['translation_cm']['P90'],limit_cm=source_limit),
        new_reference_metric_calculations_by_report=0,new_fits_by_report=0,new_image_forwards_by_report=0,new_PnP_solves_by_report=0)
    lines=['# TRAIN 위험 margin 학습과 실제 실사 T·R 결과','',
        '**합성 source 45/45 조건은 통과했지만, 실제 실사에서 안정적인 T·R 동시 개선은 달성하지 못했습니다.**','',
        '새 선택기 네 개를 실제 학습했고 수렴 검산을 통과했습니다. source VAL1024의 모든 조건을 통과한 뒤, 정답을 보지 않고 실사173장의 네 모델 선택692개를 먼저 동결했습니다. 이후 전체13모델×173장=2249행을 채점했습니다. 원래 목표와 추가 대조 조건을 함께 적용한 최종5개 범주 중 **2개 통과·3개 실패**입니다. 좋은 seed만 골라 성공으로 해석하지 않습니다.','',
        'T는 위치 오차(cm), R은 회전 오차(°)이며 작을수록 좋습니다. 실행 입력은 **RGB 이미지 한 장 + 팔레트 치수 + 기존 카메라 보정 K**입니다. 시간 정보·추가 센서·새 실사 정답을 넣지 않았습니다.','',
        '## 이번에 바꾼 학습과 바꾸지 않은 추론','',
        '직전 [context189 방법](../pallet_pose_anchor_context_20261001_v1/REPORT_KO.md)의 특징189개, 정규화, R0 기준 양축 보존 정답, 후보 pool, TRAIN2598행, λ=1e−4, 0초깃값, solver를 유지했습니다. 후보별 TRAIN 위험 초과를 학습 loss의 margin으로 추가한 것이 이번 변경입니다.','',
        '```text','r_c = max((T_c − T_anchor)/sT, (R_c − R_anchor)/sR, 0)','m_c = log1p(r_c)','J = mean_all2598[logsumexp_c(−score_c + m_c) + score_target] + (λ/2)||w||²',
        'sT = 2.4636887551191258 cm; sR = 1.113474019956766 deg','```','',
        '정답 후보의 margin은0이며 원래 유효 경쟁자는 제거하지 않습니다. 모든 후보가 실패한1행은 loss0/target−1로 전체2598행 분모에 남습니다. 유효 후보와 유한 anchor에만 뺄셈을 적용해 inf−inf를 피했습니다. **margin·참조 오차·safe mask는 추론에 사용하지 않습니다.** 실제 출력은 기존189차원 입력 점수의 argmin으로 고른 전체 R,t pose 하나입니다.','',
        '## 실제 학습과 TRAIN에서 확인된 대가','',
        '| 모델 | 반복 | objective 호출 | gap 상한 | 인증 |','|---|---:|---:|---:|---|']
    for m in C.MODEL_NAMES:
        r=receipts[m];g=training['models'][m]['independent_recompute']['certified_gap_upper_bound']
        lines.append(f"| {m} | {r['iterations']} | {r['objective_calls']} | {g:.3e} | PASS |")
    lines += ['',f'네 fit 합계는 **{iterations}회 반복·{len(traces)}회 objective 호출**입니다. 모든 fit은 같은 상한(maxiter1000/maxfun2000), optimizer 성공과 `||gradient||²/(2λ)≤1e−6`을 만족했습니다. 수렴 인증은 이 loss를 충분히 최소화했다는 뜻이며 실사 T/R 개선 인증이 아닙니다.','',
        '| 모델 | 같은 margin objective 이전→현재 | 일반 CE 이전→현재 | 정답 선택률 이전→현재 |','|---|---:|---:|---:|']
    for m in C.MODEL_NAMES:
        n=training['models'][m];o=n['previous_converged_same_target'];a=n['independent_recompute']
        lines.append(f"| {m} | {o['objective']['objective']:.9f} → {a['objective']:.9f} | {o['native_unadjusted_objective']['CE']:.9f} → {a['unadjusted_CE']:.9f} | {100*o['statistics']['target_accuracy_full_population']:.3f}% → {100*n['statistics']['target_accuracy_full_population']:.3f}% |")
    lines += ['', '동일한 새 margin 목적식으로 이전 weight와 현재 weight를 비교했습니다. **새 목적함수는 네 모델 모두 낮아졌지만, margin 없는 일반 CE와 target 정확도는 모두 악화했습니다.** 정확도 분모는 전체2598장이며 실패1장을 정답으로 세지 않습니다.','',
        '![학습 objective와 일반 CE](figures/training_margin_and_unadjusted_ce.png)','',
        '| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 최대 정규화 초과 이전→현재 |','|---|---:|---:|---:|---:|']
    for m in C.MODEL_NAMES:
        n=training['models'][m];o=n['previous_converged_same_target'];parts=[]
        for a in ('T','R','either'):parts.append(f"{o['statistics']['anchor_violations'][a]['count']} → {n['statistics']['anchor_violations'][a]['count']}")
        parts.append(f"{o['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f} → {n['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}")
        lines.append('| '+m+' | '+' | '.join(parts)+' |')
    lines += ['', '위반은 실제 unrestricted 선택이 같은 행 R0 anchor보다 나빠진 경우입니다. 위험 크기의 분모는 유효2597장이며 실패1장의 크기는 미정으로 별도 유지합니다. UNION의 한 축 이상 위반 수와 최대 위험은 줄었지만 **seed1의 R 위반은27→29로 증가**했습니다. 전체 정답 선택률이나 각 축의 개선을 함께 보아야 합니다.','',
        '![TRAIN 정확도와 위험](figures/train_target_and_risk.png)','', '## source VAL: 45/45 통과','',
        '직전43/45에서 이번45/45로 모든 사전 조건을 통과했습니다. UNION 세 seed 각각이 학습된 R0_ONLY·기존 R0 GEO·해당 DIVERSE GEO와 비교되며, 양축 중앙값 엄격 개선·각축 P90의5% 이내 보존·실패 수 비증가를 요구합니다. R0_ONLY도 새 loss로 다시 학습했으므로 pass 수만으로 모든 성능의 우열을 주장하지 않습니다. 고정 R0_GEO 수치를 함께 표시합니다.','',
        '| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
    for m,s in source['summaries'].items():
        v=s['full_population'];lines.append(f"| {m} | {v['translation_cm']['median']:.6f} | {v['rotation_deg']['median']:.6f} | {v['translation_cm']['P90']:.6f} | {v['rotation_deg']['P90']:.6f} | {s['failed_pose']} |")
    lines += ['',f"이전에 실패했던 seed3의 T P90은 {source['summaries']['UNION_s3']['full_population']['translation_cm']['P90']:.9f}cm로 고정 R0 한계 {source_limit:.9f}cm 안에 들어왔습니다. 이것은 합성 source에서의 통과이며 실사 개선의 증거를 대신하지 않습니다.",'',
        '![직전과 현재 source T/R](figures/source_val_loss_comparison.png)','', '## 실제 실사: 원래 목표와 대조 조건을 모두 적용','',
        '새 learned 선택을 참조 오차 접근 전에 동결한 뒤 기존 참조로 채점했습니다. 원래9모델의173행씩1557개 metric이 이전 값과 일치함도 확인했습니다. 새 이미지 forward·새 PnP·추가 fit은0회이며, 이번 실사 metric2249행은 동결 이후 실제 계산한 결과입니다. 보고서 생성은 이를 재계산하지 않습니다.','',
        '**평가 계약 보완:** 이전의 미실행 learned-real 평가 코드에는 matched 대조만 있고 원래 SINGLE251 비교가 빠져 있었습니다. 이번에는 실사 routing과 채점 전에 그 원래5개 조건을 복구해 고정했으며, 원래 조건과 matched 조건이 모두 통과해야 성공하도록 AND했습니다. 기준 완화나 기존 기록 덮어쓰기는 없습니다.','',
        '| 판정 범주 | matched 대조 | 원래 SINGLE251 포함 목표 | 최종 AND |','|---|---|---|---|']
    for k,g in real['stability']['gates'].items():
        fmt=lambda v:'PASS' if v else 'FAIL'
        lines.append(f"| {k} | {fmt(real['matched_intervention_stability']['gates'][k]['PASS'])} | {fmt(real['original_goal_stability']['gates'][k]['PASS'])} | {fmt(g['PASS'])} |")
    lines += ['', '![실사 판정](figures/real_stability_gates.png)','',
        '자연 가림99장에서 UNION seed1은 R0의 T/R 중앙값과 같고, seed2는 T가 같으며 R은 악화했습니다. seed3은 T만 개선하고 R은 같습니다. 따라서 **세 seed의 T와 R이 함께 엄격히 개선되지 않았습니다.** tail과 clean 보존의 통과는 악화를 제한했다는 결과이며 새 정확도 향상과 같지 않습니다.','',
        '| R0 대비 자연99 | 평균 seed 중앙값 차이 | recording×seed bootstrap95% CI |','|---|---:|---:|']
    for a in ('translation_cm','rotation_deg'):
        v=h['hierarchical_bootstrap']['metrics'][a];lines.append(f"| {a} | {v['point_estimate']['value']:.9f} | [{v['CI95'][0]:.9f}, {v['CI95'][1]:.9f}] |")
    lines += ['', '차이는 현재−R0이며 음수가 개선입니다. 촬영6개와 frozen refiner seed3개를 함께 재표본한2000회(seed20261001)의 동일 draw를 사용했습니다. 두 축 신뢰구간 상한이 모두0미만이어야 하는 조건을 통과하지 못했고, 촬영 하나씩 제외하는 검사에서도 공동 개선이 유지되지 않았습니다.','',
        '아래는 원래 baseline 이름을 그대로 보존한 **13개 모델 전체**입니다. 전부 pose 반환 실패는0이므로 conditional과 full-population T/R가 일치합니다. 실패0은 올바른 pose0오차라는 뜻이 아닙니다.']
    for pop,label in [('NATURAL99','자연 가림99'),('CLEAN29','clean29'),('WOOD45','wood45 반복 DEV stress')]:
        lines+=['',f'### {label}','','| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
        for m in real['models']:
            s=real['summaries'][pop][m];v=s['full_population']
            lines.append(f"| {m} | {v['translation_cm']['median']:.6f} | {v['rotation_deg']['median']:.6f} | {v['translation_cm']['P90']:.6f} | {v['rotation_deg']['P90']:.6f} | {s['failed_pose']} |")
    lines += ['', '![실사 전체13모델 T/R](figures/real_all_models.png)','',
        'FULL128=자연 가림99+clean29이며 wood45는 별도 반복 DEV stress입니다. FULL128 및 severity별 나머지 집계도 [원본 결과 JSON](REAL_RESULTS.json)에 보존했습니다. wood 결과나 특정 seed를 기본 성공 판정으로 대체하지 않습니다.','',
        '## 같은 실제 이미지와 치수에서 본 learned 결과','',
        '**이번 네 열은 기존 R0와 실제 학습된 UNION seed1·2·3의 선택 결과입니다. oracle 선택 그림이 아닙니다.** 이전 보고서에서 이미 고정한 자연 촬영별6개 이미지 ID를 그대로 사용했습니다. 선정 규칙은 각 촬영의 기존 R0 T 오차 최대 예시였으므로 대표 표본이나 평균 개선의 증거로 해석하지 않습니다.','',
        '선은 저장된 전체 pose의 R_cf·centroid·camera-facing extents를 기존 K로 투영했습니다. 정답 윤곽선은 표시하지 않았고 새 PnP를 실행하지 않았습니다. 각 칸에는 원본 입력 치수·실제 선택 expert/hypothesis·현재 T/R를 넣었습니다. 크게 잘못된 pose도 축척이나 프레임을 바꾸어 숨기지 않았습니다.']
    for page in range(1,4):lines+=['',f'![실제 learned RGB와 치수 {page}](figures/learned_rgb_dimensions_{page}.jpg)']
    lines += ['', '6개 이미지·K·치수·24개 실제 pose·투영좌표·선택 ID와 SHA는 [갤러리 원자료](GALLERY_SELECTION.json)에 있습니다. 이전 [참조 기반 oracle 가능성](../pallet_pose_pareto_anchor_20261001_v1/REAL_VERIFICATION_KO.md)은 이번에 재계산하지 않았으며 현재 학습 모델의 성공으로 사용하지 않습니다.','',
        '## 남은 원인과 검증 자료','',
        'TRAIN에서 큰 위험의 크기를 줄이고 source 조건을 모두 통과해도 실사에서 필요한 공동 개선으로 이어지지 않았습니다. [기존 선택의 source→실사 이전 진단](SELECTOR_TRANSFER_DIAGNOSTIC_KO.md)은 현재 동결 결과의 설명용 분석이며 추가 fit이나 새 성공 기준이 아닙니다. 현 결과만으로 입력 정보 부족이나 Linear189 표현력 하나를 원인으로 확정하지 않습니다.','',
        '자연99장에서 UNION의 실제 anchor 유지 수는 seed별93/90/89장이고 안전 개선은3/4/6장뿐입니다. 기존 참조 oracle이 보여 준51/51/53개의 안전 개선 기회 중48/47/47개를 놓쳤으며, 실제 UNION의 W/D 가설 전환은 세 seed 모두0개였습니다. 이는 많은 교정으로 tail이 무너진 실패가 아니라, 대부분 기존 선택에 머물며 필요한 개선을 충분히 회수하지 못한 양상입니다. 이 사후 관측만으로 가설 전환을 늘리는 정책을 정당화하지 않습니다. 후속 고정 RBF 특징 제안은 별도 문서의 미실행 설계이며 이번 결과가 아닙니다.','',
        '- [학습 전 독립 검산](PREFIT_REVIEW_KO.md), [학습 계약](TRAIN_PROTOCOL.json), [TRAIN 독립 수렴·위험 검산](TRAIN_CONVERGENCE_KO.md)','- [source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md), [source8192행 CSV](SOURCE_VAL_FRAME_RESULTS.csv), [45개 조건 CSV](SOURCE_VAL_CHECKS.csv)',
        '- [실사 계약](REAL_PROTOCOL.json), [정답 접근 전 선택 잠금](REAL_ROUTING_LOCK.json), [실사 독립 검산](REAL_VERIFICATION_KO.md), [실사2249행 CSV](REAL_FRAME_RESULTS.csv), [전체 비교와 recording 분석](REAL_DETAILED_COMPARISONS.json)',
        '- [학습1453행 CSV](TRAINING_OBJECTIVE_LOG.csv), [공개 checkpoint4개](model_parameters/), [그림·원자료 연결](REPORT_DATA.json)',
        '- [공개 파일 검산](PUBLIC_REVIEW_KO.md), [공개 SHA 목록](PUBLICATION_MANIFEST.json)',
        '- [학습 코드](../../../scripts/research/pallet_pose_anchor_risk_20261001_v1/convex_train.py), [source 평가 코드](../../../scripts/research/pallet_pose_anchor_risk_20261001_v1/evaluate_source.py), [실사 평가 코드](../../../scripts/research/pallet_pose_anchor_risk_20261001_v1/evaluate_real.py), [보고서 생성 코드](../../../scripts/research/pallet_pose_anchor_risk_20261001_v1/report.py)','',
        'source VAL과 실사 DEV는 여러 방법에서 반복 사용했습니다. refiner의 기존 source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건이 있어 독립적인 일반화 시험으로 주장하지 않습니다. 전체 교사 계보에는 기존 수동 코너38개/이미지9장이 포함됩니다. 이번 학습에 새 실사 정답은0개이며, 실사 참조는2D 주석·K·치수에서 만든 pose로 독립 장비 실측6D 정답이 아닙니다.','',
        'R0_ONLY는 결정론적 fit 하나이고 UNION_s1/2/3은 서로 다른 기존 frozen refiner seed를 사용합니다. 네 fit을 네 독립 반복 실험으로 해석하지 않습니다. 이전 상대 관계·상대 이득 선택기의 선행 실패도 있으므로 새 loss 하나로 transfer가 해결됐다고 주장하지 않습니다. 원래 안정성 목표는 **미달성**으로 유지합니다.','',
        '공개 자료는 판정·전 행 오차·가중치·코드·실제 이미지 결과를 검토할 수 있게 구성했습니다. 정확한 재실행에는 SHA로 연결된 로컬 원본 이미지·특징·pose 캐시가 필요합니다.','']
    publish(C.DOC/'REPORT_KO.md','\n'.join(lines))
    publish(C.DOC/'REPORT_DATA.json',data)
    publish(C.DOC/'README.md','# TRAIN 위험 margin 선택기\n\n[상세 결과·실제 RGB와 치수](REPORT_KO.md)\n\nsource45/45 통과 후 실제 실사173장을 평가했으나 안정적인 T·R 동시 개선은 미달성입니다. 원래 목표와 matched 대조의 최종 AND는5범주 중2PASS/3FAIL입니다.\n\n[전체 실사 CSV](REAL_FRAME_RESULTS.csv), [검증과 SHA](PUBLIC_REVIEW_KO.md), [학습 가중치](model_parameters/)\n')
    print(json.dumps(dict(PASS=True,csv_rows=csv_counts,objective_calls=len(traces),iterations=iterations,figures=len(artifacts),actual_RGB_images=6,goal_complete=False)))


if __name__=='__main__':
    main()
