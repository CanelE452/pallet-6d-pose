"""Independent coordinate/matrix audit and actual point-refinement report."""
from pathlib import Path
import hashlib
import html
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from data import RUN,ROOT,KIT,PARENT,EXPORT,Split,read,write,sha


def bootstrap(a,b):
    difference=np.array(b)-np.array(a);x=difference.mean(0);rng=np.random.default_rng(10101)
    sample=x[rng.integers(0,len(x),(5000,len(x)))].mean(-1)
    return dict(mean_difference_px=float(x.mean()),per_seed_difference_px=difference.mean(-1).tolist(),
                paired_frame_ci95=np.quantile(sample,[.025,.975]).tolist(),
                scope='Seeds averaged within each frame; descriptive interval on reused synthetic evaluation.')


def main():
    torch.set_num_threads(2);s=read(RUN/'SUMMARY.json');p=read(RUN/'PROTOCOL.json');training=read(RUN/'TRAINING_COMPLETE.json')
    for name,digest in p['source_sha256'].items():assert sha(KIT/name)==digest
    assert sha(KIT/'evaluate.py')==read(RUN/'EVALUATION_PROTOCOL.json')['source_sha256']
    for split,digest in p['point_sha256'].items():assert sha(PARENT/f'points_{split}.json')==digest
    for key,digest in p['line_cache_sha256'].items():assert sha(RUN/f'lines_{key}.npz')==digest
    targets={split:read(RUN/f'targets_{split}.json')['rows'] for split in ('train','calibration','synth_val')}
    for split,rows in targets.items():
        assert sha(RUN/f'targets_{split}.json')==p['target_sha256'][split]
        for row in rows:assert sha(ROOT/row['label'])==row['label_sha256']
    predictions={};metrics={};maxerr=0.;maxrecon=0.;maxmatrix=0.;count=0
    for name in s['summaries']:
        pr=read(RUN/'predictions'/f'{name}.json');mt=read(RUN/'metrics'/f'{name}.json');split=pr['split']
        assert sha(RUN/'predictions'/f'{name}.json')==mt['prediction_sha256']
        original=read(PARENT/f'points_{split}.json')['rows']
        assert len(pr['rows'])==len(mt['rows'])==len(targets[split])==len(original)
        if pr['arm']!='stock':assert sha(RUN/'models'/f'{pr["arm"]}_seed{pr["seed"]}'/'final.pt')==pr['checkpoint_sha256']
        recalculated=[];norms=[];corners=[];bad=0
        for row,met,gt,stock in zip(pr['rows'],mt['rows'],targets[split],original):
            assert row['id']==met['id']==gt['id']==stock['id']
            old=np.array(row['original_points']);out=np.array(row['points']);truth=np.array(gt['points']);valid=np.array(gt['valid'])
            assert np.array_equal(old,np.array(stock['points'],dtype=np.float32))
            err=np.sqrt(((out-truth)**2).sum(-1));before=np.sqrt(((old-truth)**2).sum(-1))
            delta=max(abs(err-np.array(met['errors'])).max(),abs(before-np.array(met['baseline_errors'])).max())
            maxerr=max(maxerr,float(delta));assert delta<1e-9
            recalculated.append(float(err[valid].mean()));norms.append(float(np.minimum(err[valid]/row['image_diagonal'],1).mean()));corners.extend(err[valid])
            bad+=int(((before<=5)&(err>20)&valid).sum())
            if 'A' in row:
                matrix=np.array(row['A']);proposal=np.array(row['proposal']);new=old+np.einsum('pij,pj->pi',matrix,proposal)
                residual=float(abs(new-out).max());maxrecon=max(maxrecon,residual);assert residual<.0003
                if pr['arm']=='directional':
                    lines=np.array(row['lines']);n=lines[:,:2]/np.linalg.norm(lines[:,:2],axis=-1)[:,None]
                    weights=np.array(row['weights']);expected=np.einsum('pm,mi,mj->pij',weights[:,:-1],n,n)
                    if pr.get('variant')=='isotropic_projection':expected=np.trace(expected,axis1=-2,axis2=-1)[:,None,None]*np.eye(2)[None]*.5
                    d=float(abs(matrix-expected).max());maxmatrix=max(maxmatrix,d);assert d<1e-6
                    assert np.linalg.eigvalsh(matrix).min()>-1e-6 and np.linalg.eigvalsh(matrix).max()<1.00001
            count+=1
        sm=mt['summary'];full=512 if split=='synth_val' else 256;corners=np.array(corners)
        check=dict(raw_frame_mean=np.mean(recalculated),raw_frame_median=np.median(recalculated),raw_frame_p90=np.quantile(recalculated,.9),
                   corner_mean=corners.mean(),pck5=(corners<=5).mean(),pck10=(corners<=10).mean(),
                   full_normalized_mean=(sum(norms)+full-len(norms))/full,new_corner_catastrophes=bad)
        for k,v in check.items():assert abs(v-sm[k])<1e-9
        predictions[name]=pr['rows'];metrics[name]=mt['rows']
    for seed in (1,2,3):
        a=predictions[f'directional_seed{seed}_calibration_actual'];b=predictions[f'directional_seed{seed}_calibration_isotropic_projection']
        for x,y in zip(a,b):
            assert x['proposal']==y['proposal'] and x['weights']==y['weights']
        rows=[training['bindings'][f'{arm}_seed{seed}'] for arm in ('local','isotropic','directional','donor')]
        for key in ('initial_state_sha256','plan_sha256','parameters'):assert len({r[key] for r in rows})==1
    audit=dict(status='PASS',frame_conditions=count,max_point_error_replay_delta=maxerr,max_coordinate_reconstruction_px=maxrecon,
        max_direction_matrix_delta=maxmatrix,GT_camera_order_no_reassignment=True,identical_isotropic_intervention_inputs=True,
        source_targets_checkpoints_verified=True,scope='Numerical/source consistency, not independent real-GT or real-generalization certification.')
    write(RUN/'AUDIT.json',audit)
    def values(pattern):return [[r['mean_px'] for r in metrics[pattern.format(seed=seed)]] for seed in (1,2,3)]
    comparisons={
        'directional_vs_stock':bootstrap(values('stock_synth_val'),values('directional_seed{seed}_synth_val_actual')),
        'directional_vs_local':bootstrap(values('local_seed{seed}_synth_val_actual'),values('directional_seed{seed}_synth_val_actual')),
        'directional_vs_isotropic':bootstrap(values('isotropic_seed{seed}_synth_val_actual'),values('directional_seed{seed}_synth_val_actual')),
        'actual_lines_vs_donor_input':bootstrap(values('directional_seed{seed}_calibration_donor_input'),values('directional_seed{seed}_calibration_actual')),
        'directional_vs_isotropic_intervention':bootstrap(values('directional_seed{seed}_calibration_isotropic_projection'),values('directional_seed{seed}_calibration_actual')),
    }
    analysis={}
    for seed in (1,2,3):
        analysis[str(seed)]={}
        for arm in ('local','isotropic','directional','donor'):
            rows=metrics[f'{arm}_seed{seed}_synth_val_actual'];old=[];new=[]
            for r in rows:
                v=np.array(r['valid']);old.extend(np.array(r['baseline_errors'])[v]);new.extend(np.array(r['errors'])[v])
            old=np.array(old);new=np.array(new)
            analysis[str(seed)][arm]=dict(worsened_fraction=float((new>old).mean()),
                easy_count=int((old<=3).sum()),easy_mean_change=float((new-old)[old<=3].mean()),
                hard_count=int((old>10).sum()),hard_mean_change=float((new-old)[old>10].mean()))
        rows=predictions[f'directional_seed{seed}_synth_val_actual'];A=np.array([r['A'] for r in rows]);eig=np.linalg.eigvalsh(A)
        analysis[str(seed)]['directional'].update(anisotropy_median=float(np.median((eig[...,1]-eig[...,0])/(eig.sum(-1)+1e-9))),
            null_weight_median=float(np.median([r['null_weight'] for r in rows])))
    write(RUN/'DIAGNOSTICS.json',dict(comparisons=comparisons,conditions=analysis,
        scope='Post-hoc explanation only; baseline-error groups use GT and are not runtime selection rules.'))
    decision=dict(status='POINT_REFINEMENT_IMPROVES_STOCK_NO_INCREMENTAL_HOUGH_GAIN',pilot_gate=s['pilot_gate'],gates=s['gates'],
                  comparisons=comparisons,real_or_6d_pose_established=False,production_yolo_modified=False)
    write(RUN/'CONCLUSION.json',decision)
    sm=s['summaries'];base=sm['stock_synth_val']
    body=['# 점별 선 후보와 방향별 점 보정 — 실험 결과',
        '**이번에는 실제 점 좌표가 기존 YOLO보다 세 seed 모두 좋아졌다. 그러나 선을 쓰지 않는 주변 영상 보정이 더 좋았으므로, Hough 추가의 이득은 입증하지 못했다.**',
        '## 실험한 아키텍처',
        '각 YOLO 점마다 작은/넓은 주변 영상 패치를 부동소수 좌표로 추출해 별도 token을 만든다. 8개 점을 하나의 heatmap으로 합치지 않는다. 각 token은 36개 DHT 선 후보의 방향·거리·영상 특징·mass를 보고 후보별 관련도와 무시(null) 선택을 학습한다. 후보 선을 따라 모은 영상 특징도 함께 사용하며 점 사이 연결선은 만들지 않는다.',
        '보정은 δ=A·v, A=Σw·n·nᵀ로 계산한다. n은 각 선의 단위 법선이고 v는 학습한 작은 보정 제안이다. 따라서 선별 기여는 법선 방향이며, 수평선 하나만 있으면 가로로 보정할 수 없다. 여러 방향의 선을 쓰면 두 좌표 모두 보정될 수 있다. null은 A에 기여하지 않는다. w는 학습 관련도이며 보정된 정답확률은 아니다.',
        '원래 YOLO와 v9 DHT는 고정하고 결합부만 학습했다. 원래 YOLO 번호 체계의 합성 점 GT로 최종 좌표 손실을 학습했으며, 추론/주평가에서 GT로 번호를 맞추지 않았다. 보정 head를0으로 초기화해 시작은 원래 점과 정확히 같다. 정답점으로 어려운 예측을 교체하지 않았고 self-training이나 실행 시 GT필터도 없다.',
        '## 합성 평가 505장 — 실제 점 오차(px)',
        '모든 조건은 동일 seed내 같은 초기화·이미지 순서·2,000update·batch16·학습률3e-4로 학습했다. 총12개 모델/24,000update다. 전체 모듈58,787파라미터는 같으나 영상만 쓰는 local 조건은 선 관련 파라미터가 비활성이다. isotropic/directional/donor는 같은 학습 모듈을 사용한다.',
        '| 조건 | seed1 평균 | seed2 평균 | seed3 평균 |','|---|---:|---:|---:|',
        '| 원래 YOLO | '+ ' | '.join([f'{base["raw_frame_mean"]:.4f}']*3)+' |']
    labels={'local':'주변 영상만으로 점 보정','isotropic':'선 사용, 방향 구분 없는 보정','directional':'점별 선 후보 + 방향별 보정','donor':'다른 이미지 선 + 방향별 보정'}
    for arm,label in labels.items():body.append('| '+label+' | '+' | '.join(f'{sm[f"{arm}_seed{seed}_synth_val_actual"]["raw_frame_mean"]:.4f}' for seed in (1,2,3))+' |')
    body+=['방향별 결합은 원래 YOLO 대비 **4.16% / 3.91% / 3.81% 개선**했다. 중앙값과 P90도 세 seed 모두 개선했고, 원래 코너 오차≤5px였던 점이>20px로 바뀐 새 큰 실패는0건이었다. 이것은 모든 점이 좋아졌다는 뜻은 아니며, 작은 악화가 생긴 점도 약40~42% 있다.',
        '그러나 local과 isotropic 조건보다 세 seed 모두 평균오차가 높았다. 따라서 사전 Hough의 추가 이득 조건은 통과하지 못했다. 좋아진 전체 결과를 전부 선 덕분이라고 해석할 수 없다.',
        '## 선과 방향을 실제로 사용했는가',
        '같은 directional 모델에서 calibration253장의 선 입력만 바꾸거나, 같은 관련도·같은 보정제안 v에서 A만 trace(A)/2·I로 바꿨다. 후자는 총 가중치 크기를 유지하면서 방향 제약만 제거하는 비교다.',
        '| 동일 모델 입력/계산 변경 | seed1 | seed2 | seed3 |','|---|---:|---:|---:|']
    for variant,label in [('actual','실제 선 + 방향별 계산'),('isotropic_projection','실제 선, 방향 제약만 제거'),('donor_input','다른 이미지 선 입력')]:
        body.append('| '+label+' | '+' | '.join(f'{sm[f"directional_seed{seed}_calibration_{variant}"]["raw_frame_mean"]:.4f}' for seed in (1,2,3))+' |')
    body+=['실제 선은 다른 이미지 선보다 세 모델 모두 낫다. A의 방향성도 0이 아니므로 방향 정보가 아예 무시된 것은 아니다. 하지만 같은 모델에서 방향 제약만 제거해도 결과가 거의 같거나 좋아졌다. 이 조건에서 방향 제약의 추가 이득은 확인되지 않았다.',
        '## 남는 문제와 해석',
        '이번 결과는 점과 선이 함께 있으면 반드시 성능이 망가진다는 설명을 반박한다. 보정 크기를 제한하고 점별 주변 정보를 유지하면 원래 YOLO보다 좋은 결과를 만들었다. 동시에 가장 강한 대조군은 선 없이 영상만 보는 보정이므로 현재 Hough 분기의 필요성은 입증되지 않았다.',
        '확인된 병목은 현재 선 후보와 방향 제약이 주변 영상에서 얻는 보정보다 추가로 더 좋은 결정을 만들지 못한다는 점이다. 부정확한 선 방향이 유용한 보정 일부를 줄이거나, 이미 점 주변 영상에 있는 정보를 반복 제공할 가능성이 있다. 두 설명의 원인 비율은 확정하지 않았다. 합성에서 학습하지 않은 실사 배경·가림·GT오차의 영향은 이번에 평가하지 않았다.',
        'GT로 사후 분류한 원래 오차>10px 코너253개에서도 영상만 쓴 보정의 개선폭이 대체로 더 컸다. “어려운 점이면 선이 항상 더 좋다”는 규칙은 지지되지 않는다. 이 분류는 설명용이며 실행 시 사용하는 필터가 아니다.',
        '## 검증 범위',
        f'원본 YOLO 라벨과 렌더러 투영의 최대 차이는0.001px 미만이었다. 독립 점 오차·행렬·보정좌표 검산 {count:,}개 프레임·조건을 통과했다. 초기 YOLO출력 일치, 단일 수평선의 가로보정0, 선의 부호/순서 불변, 선 제거 시 원래 점 복원 검사도 통과했다.',
        '평가는 원래 번호 순서 그대로 v>0인 코너3,997개, 매칭 성공505장이다. 전체512장 중 누락7장은 정규화오차1로 함께 기록했다. calibration도253/256으로 같은 처리다. 이전 실험에서 재사용한 합성 평가이며 새로운 독립 holdout이나 실사 개선 증거는 아니다. 최종6D 자세는 평가하지 않았다.',
        '**판정: 점 보정의 합성 개선은 확인. Hough 추가 이득은 미확인.** 기존 생산 YOLO는 수정하지 않았다. 다음 후보의 기준은 이번 local 보정이며, 선 결합은 이 기준보다 좋아져야 추가할 근거가 있다.',
        '전체 통계/paired bootstrap: CONCLUSION.json. 코너 난이도·방향성 진단: DIAGNOSTICS.json. 검산: AUDIT.json. 기존41.49px 실사점 지표나 과거 역할면제GT정렬 수치와 직접 비교하지 않는다.']
    text=body[0]
    for prev,block in zip(body,body[1:]):text+=('\n' if prev.startswith('|') and block.startswith('|') else '\n\n')+block
    (RUN/'REPORT_KO.md').write_text(text+'\n')
    # Show unchanged input vs actual refined coordinates and per-point line association.
    records=Split(EXPORT/'synth_val.json').records;idx={r['id']:i for i,r in enumerate(records)}
    d=metrics['directional_seed1_synth_val_actual'];delta={r['id']:r['mean_px']-r['baseline_mean_px'] for r in d}
    fixed=sorted(delta,key=lambda k:hashlib.sha256(('v10-gallery:'+k).encode()).digest())[:2]
    chosen=list(dict.fromkeys(fixed+[min(delta,key=delta.get),max(delta,key=delta.get)]));gallery=[];(RUN/'figures').mkdir(exist_ok=True)
    for fid in chosen:
        i=idx[fid];rec=records[i];row=predictions['directional_seed1_synth_val_actual'][i];local=predictions['local_seed1_synth_val_actual'][i]
        gt=targets['synth_val'][i];truth=np.array(gt['points']);valid=np.array(gt['valid']);old=np.array(row['original_points']);new=np.array(row['points']);lp=np.array(local['points'])
        image=np.asarray(Image.open(ROOT/rec['original_image']).convert('RGB'))[100:-100,100:-100]
        change=np.array(d[i]['baseline_errors'])-np.array(d[i]['errors']);change[~valid]=0;point=int(np.argmax(abs(change)))
        center=np.mean(np.stack((truth[point],old[point],new[point],lp[point])),0);span=max(25.,np.max(abs(np.stack((truth[point],old[point],new[point],lp[point]))-center))+12)
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for ax in axes:ax.imshow(image);ax.axis('off')
        axes[0].set_xlim(0,image.shape[1]);axes[0].set_ylim(image.shape[0],0)
        axes[0].scatter(truth[valid,0],truth[valid,1],s=18,c='#49eaff',label='GT')
        axes[0].scatter(old[:,0],old[:,1],s=20,facecolors='none',edgecolors='#ffdc45',label='YOLO')
        axes[0].scatter(new[:,0],new[:,1],s=12,c='#5aff79',label='directional')
        axes[0].set_title(f'All point coordinates\nYOLO {d[i]["baseline_mean_px"]:.2f}px -> directional {d[i]["mean_px"]:.2f}px',fontsize=10)
        axes[0].legend(fontsize=8,loc='lower right')
        for ax in axes[1:]:ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]+span,center[1]-span)
        for pp,color,label in [(truth[point],'#49eaff','GT'),(old[point],'#ffdc45','YOLO'),(lp[point],'#ec77ff','local'),(new[point],'#5aff79','directional')]:
            axes[1].scatter(*pp,c=color,s=45,label=label)
        axes[1].legend(fontsize=8);axes[1].set_title(f'Point {point}: actual pixel coordinates\nNo GT reassignment, no arrow magnification',fontsize=10)
        weights=np.array(row['weights'])[point];top=np.argsort(weights[:-1])[-3:][::-1];lines=np.array(row['lines'])
        for order,j in enumerate(top):
            a,b,c=lines[j];xx=np.array([center[0]-span,center[0]+span])
            if abs(b)>abs(a)*.1:axes[2].plot(xx,-(a*xx+c)/b,lw=1.5,label=f'candidate {j}: {weights[j]:.3f}')
            else:
                yy=np.array([center[1]-span,center[1]+span]);axes[2].plot(-(b*yy+c)/a,yy,lw=1.5,label=f'candidate {j}: {weights[j]:.3f}')
        axes[2].scatter(*old[point],c='#ffdc45',s=45);axes[2].scatter(*new[point],c='#5aff79',s=35)
        axes[2].set_title(f'Point-specific line association\nnull weight {weights[-1]:.3f}; uncalibrated scores',fontsize=10);axes[2].legend(fontsize=7)
        reason='fixed SHA sample' if fid in fixed else ('largest gain' if delta[fid]<0 else 'largest harm')
        fig.suptitle(f'{fid} | {reason} | synthetic validation, same camera-order IDs',fontsize=11);fig.tight_layout()
        path=RUN/'figures'/f'{fid}.png';fig.savefig(path,dpi=120);plt.close(fig)
        gallery.append(dict(id=fid,path=str(path.relative_to(RUN)),selection=reason,delta_px=delta[fid]))
    write(RUN/'GALLERY.json',gallery)
    table='<table><tr><th>실제 점 평균오차(px)</th><th>seed1</th><th>seed2</th><th>seed3</th></tr><tr><td>원래 YOLO</td>'+f'<td>{base["raw_frame_mean"]:.4f}</td>'*3+'</tr>'
    for arm,label in labels.items():table+='<tr><td>'+label+'</td>'+''.join(f'<td>{sm[f"{arm}_seed{seed}_synth_val_actual"]["raw_frame_mean"]:.4f}</td>' for seed in (1,2,3))+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>점별 선 결합과 방향별 보정 — 결과</title><style>body{font:16px/1.65 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;background:#111827;color:#eef3fc}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #53657d;text-align:left}img{width:100%}a{color:#84c9ff}pre{white-space:pre-wrap;font:15px/1.7 system-ui}select{font:inherit;padding:8px}.note{background:#25344b;padding:15px;border-left:4px solid #65dcb2}</style><h1>점별 선 후보와 방향별 점 보정</h1><p class="note">기존 YOLO 대비 점 오차가 세 seed 모두 개선됐습니다. 그러나 주변 영상만 사용하는 보정이 더 좋아 Hough 추가의 이득은 입증하지 못했습니다. GT 번호 맞춤 없이 실제 점 좌표를 평가했습니다.</p>'''+table
    page+='<h2>실제 좌표 변화와 점별 선 관련도</h2><select id="choose">'+''.join(f'<option value="{g["path"]}">{g["id"]} | {g["selection"]}</option>' for g in gallery)+'</select><p><img id="panel" src="'+gallery[0]['path']+'" alt="전체 점, 점 좌표 확대, 점별 선 후보"></p><p><a href="REPORT_KO.md">전체 결과</a> · <a href="CONCLUSION.json">통계와 판정</a> · <a href="AUDIT.json">독립 검산</a></p><pre>'+html.escape(text)+'</pre><script>document.getElementById("choose").addEventListener("change",e=>document.getElementById("panel").src=e.target.value)</script></html>'
    (RUN/'index.html').write_text(page)
    print('AUDIT',audit);print('COMPARISONS',comparisons)


if __name__=='__main__':main()
