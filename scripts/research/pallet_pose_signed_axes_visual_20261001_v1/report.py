"""Export independently checked experiment results and fixed-order VAL RGBs.

No training, selection, PnP or metric recomputation. Saved values only.
"""
import csv
import io
import itertools
import json
import numpy as np
from . import common as C

SPECS=[('translation_cm','median','T median (cm)'),('rotation_deg','median','R median (deg)'),
       ('translation_cm','P90','T P90 (cm)'),('rotation_deg','P90','R P90 (deg)')]

def save(path,value):
    payload=value if isinstance(value,bytes) else (value if isinstance(value,str) else
        json.dumps(C.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    assert not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(payload)

def csv_save(name,rows):
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);save(C.DOC/name,buf.getvalue());return C.bind(C.DOC/name)

def num(x):return 'inf (failure retained)' if x is None or np.isposinf(x) else f'{x:.6f}'

def metric_row(model,s):
    return '| '+model+' | '+' | '.join(num(s['full_population'][a][q]) for a,q,_ in SPECS)+f" | {s['failed_pose']} |"

def main():
    import matplotlib.pyplot as plt
    from PIL import Image
    required=('TRAIN_PROTOCOL','TRAINING_COMPLETE','TRAIN_CONVERGENCE','SOURCE_VAL_ROUTING_LOCK',
              'SOURCE_VAL_APPEARANCE_VERIFICATION','SOURCE_VAL_GATE','SOURCE_VAL_VERIFICATION')
    bound={name:C.bind(C.DOC/(name+'.json')) for name in required}
    docs={name:C.read(C.ROOT/b['path']) for name,b in bound.items()}
    p,train,source=docs['TRAIN_PROTOCOL'],docs['TRAIN_CONVERGENCE'],docs['SOURCE_VAL_GATE']
    assert train['complete'] and train['PASS'] and docs['SOURCE_VAL_VERIFICATION']['PASS']
    assert docs['SOURCE_VAL_APPEARANCE_VERIFICATION']['PASS'] and source['complete']
    assert source['checks_total']==45 and docs['TRAINING_COMPLETE']['fit_count']==4
    assert source['protocol']==train['protocol']==docs['TRAINING_COMPLETE']['protocol']==bound['TRAIN_PROTOCOL']
    assert source['routing_lock']==bound['SOURCE_VAL_ROUTING_LOCK']
    assert docs['SOURCE_VAL_VERIFICATION']['source_gate']==bound['SOURCE_VAL_GATE']
    assert docs['SOURCE_VAL_VERIFICATION']['routing_lock']==bound['SOURCE_VAL_ROUTING_LOCK']
    assert source['visual_input_verification']==bound['SOURCE_VAL_APPEARANCE_VERIFICATION']
    previous=C.read(C.Q_DOC/'SOURCE_VAL_GATE.json');bound['previous_Q_source']=C.bind(C.Q_DOC/'SOURCE_VAL_GATE.json')
    for m in source['baselines']:assert source['summaries'][m]==previous['summaries'][m]
    real=None
    if source['PASS']:
        real=C.read(C.DOC/'REAL_RESULTS.json')
        rv=C.read(C.DOC/'REAL_VERIFICATION.json');assert rv['complete'] and rv['PASS']
        bound.update(real=C.bind(C.DOC/'REAL_RESULTS.json'),real_verification=C.bind(C.DOC/'REAL_VERIFICATION.json'))
    else:
        stopped=C.read(C.DOC/'REAL_EVALUATION_NOT_RUN.json');assert stopped['complete']
        bound['real_not_run']=C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
    route=docs['SOURCE_VAL_ROUTING_LOCK'];C.verify(route['choices']);C.verify(source['metrics'])
    choices=C.read(C.ROOT/route['choices']['path'])
    C.verify(route['metadata']);C.verify(route['poses'])
    metadata=C.read(C.ROOT/route['metadata']['path']);rows=[r for r in metadata if r['split']=='VAL']
    poses=C.read(C.ROOT/route['poses']['path']);assert len(rows)==1024 and choices['ids']==[r['id'] for r in rows]
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        assert z['ids'].tolist()==choices['ids'];errors={m:z[m].copy() for m in source['summaries']}
    traces=[];iterations=[];certs=[];exports=[]
    for model in C.MODEL_NAMES:
        receipt=C.read(C.DOC/f'FIT_{model}.json')
        for key in ('checkpoint','trace','START'):C.verify(receipt[key])
        ck=C.read(C.ROOT/receipt['checkpoint']['path'])
        save(C.DOC/'model_parameters'/f'{model}.json',(C.ROOT/receipt['checkpoint']['path']).read_bytes())
        exports.append(C.bind(C.DOC/'model_parameters'/f'{model}.json'))
        for line in (C.ROOT/receipt['trace']['path']).read_text().splitlines():
            t=json.loads(line)
            if t['event']=='objective':
                traces.append(dict(model=model,**{k:t[k] for k in ('call','objective','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty','gradient_linf','gradient_gap_upper_bound','armijo_accepted')}))
            elif t['event']=='iteration':
                iterations.append(dict(model=model,**t))
        certs.append(dict(model=model,objective=receipt['certificate']['objective_value'],gradient_linf=receipt['certificate']['gradient_linf'],
            gap_bound=receipt['certificate']['gradient_l2_squared_over_2lambda'],calls=receipt['objective_calls'],iterations=receipt['iterations'],
            embedded_Q_objective=train['models'][model]['previous_fixed_asymmetric']['same_new_objective']['objective'],
            previous_safe=train['models'][model]['previous_fixed_asymmetric']['risk_statistics']['classes']['safe_improvement']['count'],
            current_safe=train['models'][model]['risk_statistics']['classes']['safe_improvement']['count'],
            previous_unsafe=train['models'][model]['previous_fixed_asymmetric']['risk_statistics']['classes']['unsafe']['count'],
            current_unsafe=train['models'][model]['risk_statistics']['classes']['unsafe']['count']))
    exports += [csv_save('TRAIN_TRACE.csv',traces),csv_save('TRAIN_CERTIFICATES.csv',certs),csv_save('TRAIN_ITERATIONS.csv',iterations)]
    metric_rows=[dict(id=row['id'],model=m,T_cm=errors[m][i,0],R_deg=errors[m][i,1],
        W_cm=100*row['dims'][0],H_cm=100*row['dims'][1],D_cm=100*row['dims'][2],
        available=bool(np.isfinite(errors[m][i]).all()),
        candidate=choices['records'].get(m,{}).get(row['id'],{}).get('candidate_name','fixed_GEO'),
        fallback=choices['records'].get(m,{}).get(row['id'],{}).get('fallback',False))
        for i,row in enumerate(rows) for m in errors]
    exports.append(csv_save('SOURCE_VAL_METRICS.csv',metric_rows))
    checks=[dict(model=m,baseline=b,criterion=k,PASS=value) for m,group in source['comparisons'].items()
        for b,comparison in group.items() for k,value in comparison['checks'].items()]
    assert len(checks)==45 and len(metric_rows)==8192 and len(traces)==312 and len(iterations)==84
    exports.append(csv_save('SOURCE_VAL_CHECKS.csv',checks))
    folder=C.DOC/'figures';folder.mkdir(exist_ok=True)
    figures=[];plot_data={}
    def finish(fig,name):
        path=folder/name;assert not path.exists();fig.savefig(path,dpi=125);plt.close(fig);figures.append(C.bind(path))
    fig,axes=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
    for j,model in enumerate(C.MODEL_NAMES):
        t=[r for r in traces if r['model']==model];calls=[r['call'] for r in t];ax=axes[0,j]
        for key,label in [('objective','Total J'),('Huber_symmetric','Symmetric Huber'),('Huber_underprediction','Underprediction Huber'),('Sign_logistic','Sign term')]:
            ax.plot(calls,[r[key] for r in t],label=label)
        ax.axhline(certs[j]['embedded_Q_objective'],ls=':',color='#64748b',label='Q weights + zero385')
        ax.set_title(model);ax.set_yscale('log');ax.grid(alpha=.2)
        ax=axes[1,j]
        ax.plot(calls,[max(r['gradient_linf']/1e-8,1e-30) for r in t],label='Gradient /1e-8')
        ax.plot(calls,[max(r['gradient_gap_upper_bound']/1e-6,1e-30) for r in t],label='Gap /1e-6')
        ax.axhline(1,color='#dc2626',ls='--');ax.set_yscale('log');ax.set_xlabel('All objective calls, including rejected trials');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('656 inputs = previous271 + frozen native-image DINO385\nUnchanged asymmetric2:1 Huber + sign + ridge; zero initialization / fixed Newton')
    finish(fig,'training_and_certificate.png')
    x=np.arange(4);fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,(axis,q,title) in zip(axes.flat,SPECS):
        a=[previous['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        b=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES]
        plot_data[axis+'_'+q]=dict(previous_Q271=a,current656=b)
        for offset,values,label,color in [(-.2,a,'Previous271','#64748b'),(.2,b,'Current656','#0d9488')]:
            ax.bar(x+offset,values,.4,label=label,color=color)
            for i,val in enumerate(values):ax.text(i+offset,val,f'{val:.3f}',ha='center',va='bottom',fontsize=8)
        ref=source['summaries']['R0_GEO']['full_population'][axis][q]
        ax.axhline(ref,color='#a855f7',ls=':',label='Fixed R0 GEO')
        if q=='P90':ax.axhline(1.05*ref,color='#dc2626',ls='--',label='R0 GEO x1.05')
        ax.set_xticks(x,['R0_ONLY','UNION s1','UNION s2','UNION s3']);ax.set_title(title);ax.set_ylim(0,1.25*max(a+b+[1.05*ref]));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle(f"Source VAL1024 | {source['checks_passed']}/45 fixed checks pass\nSame candidates, targets, loss and gate; image385 inputs added")
    finish(fig,'source_val_comparison.png')
    signs=np.array(list(itertools.product((-1.,1.),repeat=3)));edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(signs[i]!=signs[j])==1]
    illustrations=[];display=['R0_GEO','UNION_s1','UNION_s2','UNION_s3']
    for page in range(3):
        fig,axes=plt.subplots(2,4,figsize=(24,12));fig.subplots_adjust(top=.82,bottom=.03,left=.015,right=.99,wspace=.08,hspace=.48)
        for rr,idx in enumerate(range(2*page,2*page+2)):
            row=rows[idx];fid=row['id'];C.verify(row['image'])
            with Image.open(C.ROOT/row['image']['path']) as im:rgb=np.asarray(im.convert('RGB'))
            item=dict(id=fid,source_VAL_index=idx,image=row['image'],dimensions_m=row['dims'],K=row['K'],image_hw=row['hw'],corner_signs=signs,edges=edges,panels=[])
            for cc,m in enumerate(display):
                if m=='R0_GEO':pose=poses['records']['R0'][fid]['GEO_pose'];parent='R0';hyp=poses['records']['R0'][fid]['GEO_name']
                else:
                    choice=choices['records'][m][fid];parent=choice['parent'];hyp=choice['hypothesis']
                    pose=poses['records']['R0'][fid]['GEO_pose'] if choice['fallback'] else next(h['pose'] for h in poses['records'][parent][fid]['hypotheses'] if h['name']==hyp)
                ax=axes[rr,cc];ax.imshow(rgb);uv=None;drawn=[]
                if pose['available']:
                    xyz=signs*(np.asarray(pose['cf_extents'])/2)@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid'])
                    camera=xyz@np.asarray(row['K']).T;uv=camera[:,:2]/camera[:,2,None]
                    for a,b in edges:
                        if xyz[[a,b],2].min()>0:
                            ax.plot(uv[[a,b],0],uv[[a,b],1],color='#ef4444' if m=='R0_GEO' else '#06b6d4',lw=2);drawn.append([a,b])
                dims=' / '.join(f'{100*d:.2f}' for d in row['dims']);t,r=errors[m][idx]
                ax.set_title(f'{fid}\n{m} -> {parent} / {hyp}\nWidth / Height / Depth: {dims} cm\nT={num(t)} cm | R={num(r)} deg',fontsize=9)
                ax.set_xlim(-.5,rgb.shape[1]-.5);ax.set_ylim(rgb.shape[0]-.5,-.5);ax.axis('off')
                item['panels'].append(dict(model=m,parent=parent,hypothesis=hyp,pose=pose,projected_uv=uv,drawn_edges=drawn,T_cm=t,R_deg=r))
            illustrations.append(item)
        fig.suptitle('Actual SOURCE VAL outputs | first6 fixed-order frames, no outcome-based selection\nSynthetic images + physical W/H/D + existing calibrated K\nCurrent656 learned choices and saved physical T/R errors; these are NOT real-image evaluation results',fontsize=15)
        finish(fig,f'source_val_rgb_dimensions_{page+1}.jpg')
    lines=['# 이미지·치수 기반 T/R 공동 개선 실험','',
        f"**source VAL {source['checks_passed']}/45 조건 통과. 실사 안정적 T/R 공동 개선은 아직 입증되지 않았다.**" if real is None or not real['stability']['PASS'] else '**고정된 기존 DEV 평가의 공동 개선 기준 통과. 독립 TEST 결과는 아니다.**','',
        '입력은 RGB 한 장과 팔레트 실제 W/H/D 치수, 기존에 보정된 K다. 이미 존재하는 R0/DIVERSE의 whole-pose 후보 중 하나를 선택한다. 새 주석·시간 정보·센서를 받지 않는다.', '',
        '## 무엇을 바꿨는가', '',
        '직전 Q의 271개 기하/잔차 입력에 고정 DINO의 후보별 이미지 특징385개만 추가했다. 8개 투영 꼭짓점의 원본 영상 영역 token384 평균과 지원 비율1이며, 모든 후보는 같은 R0 crop을 공유한다. 이전 271개 입력과 9개 해시를 보존했다. source의 반사 padding100을 표본 위치에서 제외하고 실제 사진은 pad0을 사용한다.', '',
        '가중치는 656×2=1,312개다. bias0, ridge1e−4, 비대칭 Huber 과소:과대 비용2:1, sign계수1, 타깃 스케일·유효 후보·동률 규칙·평가 조건을 유지했다. 4개 모델 모두 0부터 학습했다. 모델 용량 증가도 포함된 비교이며, backbone은 학습하지 않았다.', '',
        '[상세 설계](DESIGN_KO.md) · [원본 영상 입력 검산](../pallet_pose_dino_native_inputs_20261001_v1/REPORT_KO.md) · [이전 DINO 시도 및 한계](../pallet_pose_dino_input_audit_20261001_v1/PRIOR_METHODS_KO.md)', '',
        '## 학습과 독립 검산', '',
        'TRAIN 2,598행(실패1행 포함), 고정4fits. 수렴 기준은 gradient∞≤1e−8, gradient²/(2λ)≤1e−6이다. 이것은 학습 목적식의 수렴이며 T/R 일반화 성능 보장이 아니다.', '',
        '| 모델 | 최종 J | Q 가중치+추가385=0의 동일 J | gradient∞ | gap 상계 | objective 호출 | 승인 반복 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in certs:lines.append(f"| {c['model']} | {c['objective']:.9f} | {c['embedded_Q_objective']:.9f} | {c['gradient_linf']:.3g} | {c['gap_bound']:.3g} | {c['calls']} | {c['iterations']} |")
    lines += ['', 'TRAIN에서 R0 anchor 대비 안전 개선 선택과 어느 한 축이라도 악화된 선택을 함께 확인했다. 안전 개선이 늘어도 위험한 선택이 동시에 늘 수 있으므로 목적식 감소를 T/R 공동 개선으로 해석하지 않는다.', '',
        '| 모델 | 안전 개선 Q→656 | 어느 축이든 악화 Q→656 |','|---|---:|---:|']
    for c in certs:lines.append(f"| {c['model']} | {c['previous_safe']} → {c['current_safe']} | {c['previous_unsafe']} → {c['current_unsafe']} |")
    lines += ['', '![학습과 수렴](figures/training_and_certificate.png)', '',
        '[학습 독립 검산](TRAIN_CONVERGENCE_KO.md) · [전체 학습 trace](TRAIN_TRACE.csv) · [최종 인증 수치](TRAIN_CERTIFICATES.csv)', '',
        '## Source VAL 전체 결과', '',
        '원래 1,024행을 전부 유지하며 실패는 T/R 모두 +∞다. UNION 각 seed를 shared R0_ONLY, 고정 R0_GEO, 대응 DIVERSE_GEO와 비교한다. T/R 중앙값 각각 엄격 감소, P90 각각 1.05배 이내, 실패 비증가의 45조건을 모두 요구한다.', '',
        '| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
    lines += [metric_row(m,s) for m,s in source['summaries'].items()]
    lines += ['', '직전 Q와 비교한 새 UNION의 중앙값 변화:', '',
        '| 모델 | T 중앙값 이전→현재 cm | R 중앙값 이전→현재 ° |','|---|---:|---:|']
    for model in C.MODEL_NAMES[1:]:
        a=previous['summaries'][model]['full_population'];b=source['summaries'][model]['full_population']
        lines.append(f"| {model} | {a['translation_cm']['median']:.6f} → {b['translation_cm']['median']:.6f} | {a['rotation_deg']['median']:.6f} → {b['rotation_deg']['median']:.6f} |")
    lines += ['', '세 UNION 모두 T 중앙값은 기준 R0보다 낮다. 그러나 seed1의 R 중앙값은 R0와 동일하여 엄격한 감소 조건을 충족하지 못했다. 직전의 seed3 T 미달은 해소됐지만 seed1 R 조건이 새로 미달했다. 같은 43/45라도 실패 원인은 바뀌었다. 이미지 특징 추가는 이번 합성 검증의 T에는 도움이 됐지만, 안정적인 T/R 동시 개선은 확인되지 않았다.']
    selection=docs['SOURCE_VAL_VERIFICATION']['source_selection_comparison']
    lines += ['', '고정된 선택을 사후 진단한 결과다. 정답 기반 안전 판정을 추론의 필터로 사용하지 않았다. 한 축이라도 R0 anchor보다 나빠진 선택은 그대로 포함했다.', '',
        '| 모델 | VAL 안전 개선 Q→656 | VAL 어느 축이든 악화 Q→656 | 바뀐 선택 |','|---|---:|---:|---:|']
    for model,s in selection['models'].items():
        a=s['previous']['classes'];b=s['current']['classes']
        lines.append(f"| {model} | {a['safe_improvement']} → {b['safe_improvement']} | {a['unsafe']} → {b['unsafe']} | {s['changed_identity_count']} |")
    lines += ['', '![이전271과 이미지추가656 비교](figures/source_val_comparison.png)', '',
        f"실패한 조건 {len(source['failed_checks'])}개:", '', '```json',json.dumps(source['failed_checks'],ensure_ascii=False,indent=2),'```','',
        '[전체 판정](SOURCE_VAL_GATE.json) · [입력 독립 검산](SOURCE_VAL_APPEARANCE_VERIFICATION.json) · [route/오차/판정 독립 검산](SOURCE_VAL_VERIFICATION_KO.md) · [전체8모델×1,024행 수치·치수](SOURCE_VAL_METRICS.csv)', '',
        '## 이미지와 실제 치수', '',
        '기존 VAL 순서의 첫 6행이다. 결과로 선별하지 않았다. 빨강은 R0_GEO, 청록은 각 새 UNION이 고정 선택한 pose다. 각 패널에 W/H/D(cm), 선택 후보, 저장된 T/R를 표시했다. 합성 VAL이며 실사 성능 예시가 아니다.', '',
        *[f'![VAL 이미지·치수·추정자세 {i}](figures/source_val_rgb_dimensions_{i}.jpg)' for i in range(1,4)], '',
        '## 실사 평가와 결론', '']
    if real is None:lines += ['source 조건을 모두 통과하지 못해 이번 모델의 실사 입력 추출·선택·T/R 평가는 실행하지 않았다. 기존 실사 수치를 이번 모델의 결과로 옮겨 쓰지 않았다. 이미지 특징 추가의 학습 수렴만으로 개선됐다고 결론낼 수 없다.', '', '[실사 미실행 기록](REAL_EVALUATION_NOT_RUN.json) · [기존 실제 사진6장·팔레트 치수110×11×130cm](../pallet_pose_signed_axes_asymmetric_20261001_v1/REPORT_KO.md) — 해당 사진은 이전 R0 입력 예시이며 이번 모델의 실사 결과가 아니다.']
    else:
        lines += [f"기존173프레임의 실제 평가에서 결합 안정성 판정은 **{real['stability']['PASS']}**다. 원래 기준과 matched 비교를 각각 적용하고 AND로 결합했다.",'','[전체 실사 결과](REAL_RESULTS.json) · [독립 검산](REAL_VERIFICATION.json)']
        for pop in ('NATURAL99','CLEAN29','WOOD45'):
            lines += ['',f'### {pop}','','| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |','|---|---:|---:|---:|---:|---:|']
            lines += [metric_row(m,real['summaries'][pop][m]) for m in real['models']]
    lines += ['', 'source VAL과 실사 DEV는 기존 시도에서 반복 사용했다. 실사 참조 pose는 기존2D 주석/K/치수에서 도출된 것으로 새 물리 계측 정답이 아니다. 기존 teacher 계보에 실사9장/수동 코너38개가 있으므로 전체 시스템을 real-GT-free라고 표현하지 않는다. 꼭짓점 평균은 순서 정보를 잃으며, 원본 영역 token에도 전역 문맥의 패딩 영향이 남을 수 있다.', '',
        '## 코드·파라미터·재현', '',
        '[봉인 프로토콜](TRAIN_PROTOCOL.json) · [PREFIT](PREFIT_REVIEW_KO.md) · [656×2 파라미터4개](model_parameters/) · [그림·수치 출처](REPORT_DATA.json) · [실행 기록](EXECUTION_KO.md)', '',
        '[학습 코드](../../../scripts/research/pallet_pose_signed_axes_visual_20261001_v1/convex_train.py) · [이미지 입력](../../../scripts/research/pallet_pose_signed_axes_visual_20261001_v1/appearance_features.py) · [source 평가](../../../scripts/research/pallet_pose_signed_axes_visual_20261001_v1/evaluate_source.py) · [실사 평가](../../../scripts/research/pallet_pose_signed_axes_visual_20261001_v1/evaluate_real.py)', '',
        '원본 RGB·대용량 token/NPZ·사전학습 backbone은 로컬에 보존한다. GitHub에는 코드·최종 선형 파라미터·수치·이미지·SHA 영수증을 게시한다. 전체 재실행에는 해시가 일치하는 로컬 원본 자료가 필요하다.','']
    save(C.DOC/'REPORT_KO.md','\n'.join(lines))
    save(C.DOC/'REPORT_DATA.json',dict(complete=True,code=C.bind(__file__),inputs=bound,figures=figures,exports=exports,
        training_certificates=certs,source_comparison=plot_data,source_summaries=source['summaries'],source_selection_comparison=selection,
        illustrations=illustrations,illustration_selection='First6 fixed-order SOURCE VAL rows; no outcome-based selection',
        source_checks_passed=source['checks_passed'],source_checks_total=45,current_real_evaluated=real is not None,
        stable_joint_improvement_achieved=bool(real and real['stability']['PASS']),new_fits=0,new_metric_calls=0,new_PnP_solves=0))
    print('VISUAL656_REPORT_COMPLETE',C.bind(C.DOC/'REPORT_KO.md'),flush=True)

if __name__=='__main__':main()
