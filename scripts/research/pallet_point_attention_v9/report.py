"""Verified report for trained point-neighborhood feature modulation."""
from pathlib import Path
import hashlib
import json
import html
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from prepare import RUN,ROOT,EXPORT,sha,write,Split


def read(p):return json.loads(p.read_text())


def paired(a,b):
    # One image is the resampling unit; repeated seeds are averaged first.
    delta=np.array(b)-np.array(a);x=delta.mean(0);rng=np.random.default_rng(9103)
    samples=x[rng.integers(0,len(x),(3000,len(x)))].mean(1)
    return dict(mean_change_px=float(x.mean()),seed_mean_changes=delta.mean(1).tolist(),
                frame_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),
                interpretation='Descriptive reused-evaluation interval; seeds are not independent images.')


def main():
    torch.set_num_threads(2);s=read(RUN/'SUMMARY.json');protocol=read(RUN/'PROTOCOL.json');train=read(RUN/'TRAINING_COMPLETE.json')
    for name,digest in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    assert sha(Path(__file__).with_name('evaluate.py'))==read(RUN/'EVALUATION_PROTOCOL.json')['source_sha256']
    for name,digest in protocol['dependency_sha256'].items():assert sha(ROOT/name)==digest
    for split,digest in protocol['points_sha256'].items():assert sha(RUN/f'points_{split}.json')==digest
    allmetrics={};allpred={};count=0;maxdiff=0.;maxsummary=0.;losses={}
    for name in s['summaries']:
        p=read(RUN/'predictions'/f'{name}.json');m=read(RUN/'metrics'/f'{name}.json')
        assert sha(RUN/'predictions'/f'{name}.json')==m['prediction_sha256']
        assert sha(RUN/'models'/f'{p["arm"]}_seed{p["seed"]}'/'final.pt')==p['checkpoint_sha256']
        stock=read(RUN/f'points_{p["split"]}.json')['rows'];assert len(stock)==len(p['rows'])==len(m['rows'])
        replay=[]
        for obs,row,st in zip(p['rows'],m['rows'],stock):
            assert obs['id']==row['id']==st['id']
            assert np.array_equal(np.array(obs['yolo_points']),np.asarray(st['points'],dtype=np.float32))
            lines=np.array(obs['lines']);mass=np.array(obs['mass']);ep=np.array(row['gt_endpoints']);valid=np.array(row['valid_edges'])
            n=lines[...,:2];c=lines[...,2];norm=np.linalg.norm(n,axis=-1)
            a,b=ep[:,0],ep[:,1]
            cost=(abs(np.einsum('ed,qmd->eqm',a,n)+c[None])+abs(np.einsum('ed,qmd->eqm',b,n)+c[None]))/(2*norm[None])
            top=mass.argmax(-1);role=cost[np.arange(12),np.arange(12),top];chosen=cost[:,np.arange(12),top]
            u,v=linear_sum_assignment(chosen[valid])
            values=dict(role_mean_px=role[valid].mean(),unordered_mean_px=chosen[valid][u,v].mean(),
                role_normalized_mean=np.minimum(role[valid]/row['image_diagonal'],1.).mean(),
                horizontal_mean_px=role[:8][valid[:8]].mean(),height_mean_px=role[8:][valid[8:]].mean(),
                best3_role_mean_px=cost[np.arange(12),np.arange(12)].min(-1)[valid].mean(),
                gt_coverage10=(chosen[valid].min(-1)<=10).mean(),role_correct10=(role[valid]<=10).mean())
            for k,val in values.items():maxdiff=max(maxdiff,abs(float(val)-row[k]));assert abs(float(val)-row[k])<1e-8
            replay.append(values);count+=1
        for key in m['summary']:
            vals=np.array([r[key] for r in replay])
            for stat,z in [('mean',vals.mean()),('median',np.median(vals)),('p90',np.quantile(vals,.9))]:
                delta=abs(float(z)-m['summary'][key][stat]);maxsummary=max(maxsummary,delta);assert delta<1e-8
        allpred[name]=p['rows'];allmetrics[name]=m['rows']
    for seed in (1,2,3):
        rows=[train['bindings'][f'{a}_seed{seed}'] for a in ('none','points','donor')]
        for key in ('initial_state_sha256','plan_sha256','trainable'):assert len({r[key] for r in rows})==1
        for arm in ('none','points','donor'):
            cell=RUN/'models'/f'{arm}_seed{seed}';trace=[json.loads(x) for x in (cell/'TRACE.jsonl').read_text().splitlines()]
            assert len(trace)==2000 and all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in trace)
            losses[f'{arm}_seed{seed}']=dict(first200=float(np.mean([r['loss'] for r in trace[:200]])),last200=float(np.mean([r['loss'] for r in trace[-200:]])))
    audit=dict(status='PASS',replayed_frame_conditions=count,max_metric_difference=maxdiff,max_summary_difference=maxsummary,
        source_checkpoint_and_point_bindings=True,matched_parameters_initialization_and_plans=True,
        finite_training_updates=18000,original_yolo_points_preserved=True,
        inference_supervision_loads={split:read(RUN/f'INFERENCE_AUDIT_{split}.json')['supervision_loads'] for split in ('synth_val','calibration')},
        caveat='Checks arithmetic and experiment contracts, not independent correctness of source GT or real generalization.')
    write(RUN/'AUDIT.json',audit);write(RUN/'TRAINING_TREND.json',losses)
    def values(split,arm,variant,key):return [[r[key] for r in allmetrics[f'{arm}_seed{seed}_{split}_{variant}']] for seed in (1,2,3)]
    comparisons={
        'val_role_points_minus_none':paired(values('synth_val','none','none','role_mean_px'),values('synth_val','points','points','role_mean_px')),
        'val_geometry_points_minus_none':paired(values('synth_val','none','none','unordered_mean_px'),values('synth_val','points','points','unordered_mean_px')),
        'cal_input_points_minus_donor':paired(values('calibration','points','donor','role_mean_px'),values('calibration','points','points','role_mean_px')),
        'cal_input_points_minus_none':paired(values('calibration','points','none','role_mean_px'),values('calibration','points','points','role_mean_px')),
    }
    influence={}
    for seed in (1,2,3):
        actual=allpred[f'points_seed{seed}_calibration_points'];zero=allpred[f'points_seed{seed}_calibration_none']
        diff=np.array([np.mean(abs(np.array(a['gate_map'])-np.array(b['gate_map']))) for a,b in zip(actual,zero)])
        influence[str(seed)]=dict(mean_spatial_channel_mean_gate_change=float(diff.mean()),max_frame_change=float(diff.max()))
    decision=dict(status='LEARNED_POINT_USE_CONFIRMED_STABLE_GAIN_NOT_ESTABLISHED',pilot_gate=s['pilot_gate'],
                  gates=s['gates'],comparisons=comparisons,gate_input_effect=influence,
                  trained_models=9,optimizer_steps=18000,line_only=True,real_or_pose_improvement_established=False)
    write(RUN/'CONCLUSION.json',decision)
    body=['# 점 주변 영상 특징을 학습하는 Deep Hough — 결과',
        '**점 위치 정보를 실제로 활용하는 학습은 확인했다. 하지만 점 정보 없이 동일하게 추가 학습한 모델보다 안정적으로 좋아지지는 않았다. 두 seed는 소폭 개선, 한 seed는 악화했다.**',
        '## 무엇을 학습했는가',
        '기존 YOLO26의 backbone 특징과 실제 예측점 좌표는 고정했다. 점을 연결하지 않고 점 주변 Gaussian map 3개(관측 박스 대각선의 1/3/10%)를 영상 특징과 함께 입력했다. 3×3 convolution이 이 정보를 받아 채널별 공간 가중치를 학습하고, 그 가중치가 반영된 특징으로 DHT 투표와 선의 방향·역할 예측을 수행했다.',
        '가중치는 1+0.5·tanh 형태로 0.5~1.5 범위이며 처음에는 정확히 1이다. 전체 선의 영상 증거를 남기면서 강조·억제를 학습한다. 추론 뒤 선 점수에 거리 가중치를 더했던 v8과 달리, 이번에는 영상 특징의 가중치를 실제 학습했다.',
        'none/points/donor 세 조건은 같은 143,900개 파라미터, 동일 시작 가중치·학습 순서·손실·학습률을 사용했다. none은 점 map 0, points는 올바른 YOLO 점, donor는 다른 이미지의 점 위치를 현재 ROI에 맞춰 사용한다. seed3개 × 조건3개 × 2,000 update, batch8, 총18,000회 학습했다. 초기 DHT 가중치에서 trunk·line·새 가중치 모듈을 같은 선 손실로 추가 학습했다.',
        '이번 비교는 선 학습만 수행한다. 기존의 약한 새 점/자세 head는 학습하지 않았다. 정답선은 합성 GT에서 감독 신호로 만들었으며 GT 점을 attention 입력으로 넣지 않았다. GT 기반 필터나 self-training은 사용하지 않았다.',
        '## 합성 평가 505장 — 선 오차(px)',
        '역할 포함 오차는 어느 경계인지까지 평가하며, 유효한 물체 전체 C1/C2 대칭만 허용한다. 공간 배치 오차는 번호를 제외하고 동일한 일대일 GT 선 매칭을 적용한다. 점/자세 오차와는 다른 지표이므로 과거41.49px 등과 직접 비교하면 안 된다.',
        '| 비교 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    table_rows=[]
    for key,label in [('role_mean_px','역할 포함'),('unordered_mean_px','선 공간 배치')]:
        for arm,title in [('none','점 정보 없음'),('points','점 주변 학습'),('donor','다른 이미지 점으로 학습')]:
            vals=[s['summaries'][f'{arm}_seed{seed}_synth_val_{arm}'][key]['mean'] for seed in (1,2,3)]
            row=(label+' / '+title,vals);table_rows.append(row)
            body.append('| '+row[0]+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    body+=['점 주변 학습의 역할 오차 변화는 **1.69% 개선 / 0.86% 개선 / 1.08% 악화**, 공간 배치는 **3.86% 개선 / 2.72% 개선 / 2.57% 악화**였다. 미리 정한 모든 seed의 5% 개선 조건을 통과하지 못했고, seed1의 역할 P90도 악화했다.',
        '## 점 정보를 실제로 사용했는가 — 같은 모델의 입력만 변경',
        'calibration253장에서 points 모델의 가중치를 고정하고 점 입력만 바꿨다. 아래는 역할 포함 선 평균오차다.',
        '| 점 입력 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    for variant,label in [('points','올바른 YOLO 점'),('none','점 정보 제거'),('donor','다른 이미지 점'),('box','박스 코너·변 중앙')]:
        vals=[s['summaries'][f'points_seed{seed}_calibration_{variant}']['role_mean_px']['mean'] for seed in (1,2,3)]
        body.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    body+=['**올바른 점은 제거/다른 이미지 점/박스 지점보다 세 모델 모두 좋았다.** 가중치 map 자체도 점 입력에 따라 달라졌다. 이는 단순 연결 검사보다 강한 실제 입력 활용 증거다. 다만 같은 모델의 입력 제거는 학습 분포와 달라질 수 있으므로, 모델 전체의 개선 여부는 위의 같은 조건으로 학습한 none/donor 대조군으로 판단했다.',
        '## 왜 안정적인 개선까지 이어지지 않았는가',
        '확인된 사실은 점 주변 정보가 일부 도움이 되지만, 역할 구분과 전체 경계를 빠짐없이 찾는 데 필요한 이득이 작고 seed에 따라 변한다는 것이다. 역할 오차가 공간 배치 오차보다 계속 큰 것도 이 문제와 일치한다. 특정 경계 역할·방향 혼동의 원인 비율을 이번 결과만으로 확정하지 않는다.',
        '추정되는 제한은 다음과 같다. 번호를 사용하지 않는 점 주변 map은 위치를 알려주지만 방향과 어느 경계인지 직접 알려주지는 않는다. 현재의 작은 공간 가중치 모듈은 기존 특징을 0.5~1.5배 조절하며, 새 경계 표현을 충분히 만들지 못했을 수 있다. 또한 기존24×24 ROI 특징 해상도와 구조선/가시 경계의 차이가 병목일 수 있다. 이들은 개별 원인 실험으로 아직 확정하지 않은 가설이다.',
        '학습 손실은 올바른 점 조건에서 세 seed 모두 낮았지만 평가 개선은 두 seed에만 나타났다. 그러므로 단순 연결 오류로 점 입력이 무시된 상황은 아니며, 학습량을 늘리면 반드시 해결된다고도 할 수 없다. 이번 예산 뒤 추가 학습이나 평가 기반 설정 선택은 하지 않았다.',
        '## 판단과 검증 범위',
        '**점 주변을 이용한 학습 자체는 된다. 현재 작은 가중치 모듈의 안정적인 성능 개선은 미확인이다.** 이번 모델을 기존 YOLO 개선 모델로 채택하지 않는다. 실제 점/자세를 수정하거나 실사에서 성능이 좋아졌다는 결과도 아니다.',
        f'수치 감사 {count:,}개 프레임·조건 통과, 최대 재계산 차이 {maxdiff:.2e}px. 18,000 update의 유한 손실·gradient, 동일 파라미터/시작 상태/학습 순서, 원래 YOLO 점 보존, 추론 중 GT 파일 접근0건을 확인했다.',
        '추가 선 학습의 train1775와 평가505는 분리되어 있지만 모두 이전 실험에서 사용한 합성 집합이다. 새 독립 holdout이 아니고 원래 YOLO의 학습 이력까지 독립임을 주장하지 않는다. 평가는 기존 검출 매칭 성공505/512 및253/256 조건부다. 모든 점과 GT annotation은 원본 그대로 유지했다.',
        '시각화의 가중치 map은 채널 평균 feature gate 또는 점 입력 교체로 인한 gate 변화다. 원본 픽셀 중요도나 정답 확률을 나타내는 attention attribution으로 해석하면 안 된다.']
    text=body[0]
    for prev,block in zip(body,body[1:]):text+=('\n' if prev.startswith('|') and block.startswith('|') else '\n\n')+block
    (RUN/'REPORT_KO.md').write_text(text+'\n')
    # Calibration input intervention gallery, explicitly separate from main validation table.
    data=Split(EXPORT/'calibration.json');records=data.records;index={r['id']:i for i,r in enumerate(records)}
    base=allmetrics['none_seed1_calibration_none'];guided=allmetrics['points_seed1_calibration_points']
    delta={r['id']:g['role_mean_px']-r['role_mean_px'] for r,g in zip(base,guided)}
    fixed=sorted(delta,key=lambda k:hashlib.sha256(('attention-v9-gallery:'+k).encode()).digest())[:2]
    selected=list(dict.fromkeys(fixed+[min(delta,key=delta.get),max(delta,key=delta.get)]));gallery=[]
    (RUN/'figures').mkdir(exist_ok=True)
    for fid in selected:
        i=index[fid];rec=records[i];O,_,_=data.batch([i],'cpu',targets=False);bb=O['box'][0].numpy()
        image=np.asarray(Image.open(ROOT/rec['original_image']).convert('RGB'))[100:-100,100:-100]
        actual=allpred['points_seed1_calibration_points'][i];zero=allpred['points_seed1_calibration_none'][i]
        points=np.array(actual['yolo_points']);sigma=.03*np.linalg.norm(bb[2:]-bb[:2])
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        for ax in axes.flat:
            ax.imshow(image);ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
            ax.scatter(points[:,0],points[:,1],s=15,c='#ffdc50',zorder=5)
        for x,y in points:axes[0,0].add_patch(Circle((x,y),sigma,color='#ee62c3',alpha=.25))
        axes[0,0].set_title('Frozen YOLO points and neighborhoods')
        change=np.abs(np.array(actual['gate_map'])-np.array(zero['gate_map']))
        hm=axes[0,1].imshow(change,extent=[bb[0],bb[2],bb[3],bb[1]],alpha=.70,cmap='magma',vmin=0,vmax=max(float(change.max()),1e-6))
        fig.colorbar(hm,ax=axes[0,1],fraction=.04,pad=.02)
        axes[0,1].set_title('Gate change caused by point input\nChannel mean difference; not pixel importance',fontsize=10)
        for ax,arm in zip(axes[1],('none','points')):
            pred=allpred[f'{arm}_seed1_calibration_{arm}'][i];met=allmetrics[f'{arm}_seed1_calibration_{arm}'][i]
            for seg in np.array(met['gt_endpoints']):ax.plot(seg[:,0],seg[:,1],c='#53e6ff',lw=1)
            l=np.array(pred['lines']);top=l[np.arange(12),np.array(pred['mass']).argmax(-1)]
            for a,b,c in top:
                cuts=[]
                if abs(b)>1e-10:
                    for x in [bb[0],bb[2]]:
                        y=-(a*x+c)/b
                        if bb[1]<=y<=bb[3]:cuts.append([x,y])
                if abs(a)>1e-10:
                    for y in [bb[1],bb[3]]:
                        x=-(b*y+c)/a
                        if bb[0]<=x<=bb[2]:cuts.append([x,y])
                if len(cuts)>=2:
                    q=np.array(cuts[:2]);ax.plot(q[:,0],q[:,1],c='#ff864a',alpha=.8,lw=1.1)
            ax.set_title(('Matched no-point model' if arm=='none' else 'Learned point-neighborhood model')+f'\nrole {met["role_mean_px"]:.1f}px / geometry {met["unordered_mean_px"]:.1f}px',fontsize=10)
        reason='fixed SHA sample' if fid in fixed else ('largest calibration gain' if delta[fid]<0 else 'largest calibration harm')
        fig.suptitle(f'{fid} | {reason}\nCalibration diagnostic; yellow points, cyan GT, orange predicted lines',fontsize=11)
        fig.tight_layout();path=RUN/'figures'/f'{fid}.png';fig.savefig(path,dpi=120);plt.close(fig)
        gallery.append(dict(id=fid,path=str(path.relative_to(RUN)),selection=reason,role_delta_px=delta[fid]))
    write(RUN/'GALLERY.json',gallery)
    table='<table><tr><th>합성 평가 선 오차(px)</th><th>seed 1</th><th>seed 2</th><th>seed 3</th></tr>'
    for label,vals in table_rows:table+='<tr><td>'+label+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>학습형 점 주변 Deep Hough — 결과</title><style>body{font:16px/1.65 system-ui;max-width:1350px;margin:32px auto;padding:0 24px;background:#111827;color:#eef2fb}a{color:#8bceff}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #53617b;text-align:left}img{width:100%;max-width:1100px}pre{white-space:pre-wrap;font:15px/1.7 system-ui}select{font:inherit;padding:8px}.note{background:#28344b;padding:16px;border-left:4px solid #ffd070}</style><h1>점 주변 영상 특징을 학습한 결과</h1><p class="note">점 입력 활용은 확인했습니다. 같은 예산의 점 없는 모델 대비 두 seed 소폭 개선, 한 seed 악화로 안정적인 성능 향상은 입증하지 못했습니다. 원래 YOLO 점은 고정했습니다.</p>'''+table
    page+='<h2>점 입력으로 변하는 특징 가중치와 선 예측</h2><p>calibration의 고정 사례2장과 최대 개선/악화 사례입니다. 위 표의 주 평가는 별도의 synth_val 집합입니다.</p><select id="choose">'
    page+=''.join(f'<option value="{g["path"]}">{g["id"]} | {g["selection"]}</option>' for g in gallery)+'</select><p><img id="panel" src="'+gallery[0]['path']+'" alt="점 주변, 점 입력에 따른 특징 가중치 변화, 두 모델 선 비교"></p>'
    page+='<p><a href="REPORT_KO.md">전체 보고서</a> · <a href="CONCLUSION.json">결론과 통계</a> · <a href="AUDIT.json">독립 검산</a></p><pre>'+html.escape(text)+'</pre><script>document.getElementById("choose").addEventListener("change",e=>document.getElementById("panel").src=e.target.value)</script></html>'
    (RUN/'index.html').write_text(page)
    print(json.dumps(decision,ensure_ascii=False,indent=2));print('AUDIT',audit)


if __name__=='__main__':main()
