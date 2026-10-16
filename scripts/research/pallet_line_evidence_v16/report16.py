"""Measured outcomes, actual sampled line evidence, and reusable local HTML."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v16-mpl')
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context16 import *
from model16 import EvidenceEvaluator
from eval16 import prediction_path

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','amodal':'V12 전체 복원','v14':'V14 융합','balanced_old':'V15 보수적·균형','risk_old':'V15 보수적','balanced_rich':'V15 큰 선 후보','control':'V16 동일후보 대조군','strip':'V16 선 구간 특징','role':'V16 특징 + 경계 역할','balanced_role':'V16 역할 + 사례 균형','median_role':'V16 역할 + 적극적 복원','role_off':'역할 모델 · 근거 끄기','role_donor':'역할 모델 · 다른 영상 근거'}
COLORS={'control':'#64748b','strip':'#2563eb','role':'#f97316','balanced_role':'#16a34a','median_role':'#a855f7'}
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(rows,arm,typ=None):
    r=[rows['stock']] if arm=='stock' else [rows[f'{arm}_seed{s}'] for s in SEEDS]
    return float(np.mean([x['strata'][typ]['mean_px'] if typ else x['frame_mean_px'] for x in r]))

def tradeoff(s):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,(condition,typ) in zip(axes,[('corner20','hidden'),('crop40','outside')]):
        ax.axvline(1,color='#16a34a',ls='--');ax.axhline(80,color='#16a34a',ls='--')
        for arm in ARMS:
            for seed,marker in zip(SEEDS,('o','s','^')):
                mild=s['results']['real/crop20'];hard=s['results']['real/'+condition];n=f'{arm}_seed{seed}';b=f'base_seed{seed}';a=f'amodal_seed{seed}'
                x=100*(mild[n]['strata']['outside']['mean_px']/mild[b]['strata']['outside']['mean_px']-1)
                y=100*(hard[b]['strata'][typ]['mean_px']-hard[n]['strata'][typ]['mean_px'])/(hard[b]['strata'][typ]['mean_px']-hard[a]['strata'][typ]['mean_px'])
                ax.scatter(x,y,color=COLORS[arm],marker=marker,s=55,label=NAMES[arm].replace('V16 ','') if seed==1 else None)
        ax.set_xlim(-2,40);ax.set_ylim(-10,110);ax.set_xlabel('잘림20% 악화율(%) · 1% 이내 필요');ax.set_ylabel('기존 복원 이득 유지율(%) · 80% 이상 필요');ax.set_title(LABELS[condition]);ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    fig.suptitle('선 구간 특징의 부분 이득과 전체 안정성은 별개\n원/사각/삼각은 seed1/2/3 · 손실별 고정 모델, 평가 후 임계값 탐색 없음')
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(RUN/'tradeoff.png',dpi=130,bbox_inches='tight');plt.close(fig)

def gallery():
    d=RUN/'figures';d.mkdir(exist_ok=True);rows=[];model=EvidenceEvaluator(1).eval();model.load_state_dict(torch.load(RUN/'models/role_seed1/final.pt',weights_only=True)['state_dict'])
    for condition in ('corner20','crop20','crop40','crop30'):
        m,O=load('real',condition);rep=repcache('real',condition,1);t=targets('real',condition);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);typ='hidden' if condition=='corner20' else 'outside'
        ps={a:np.load(prediction_path('real',condition,f'{a}_seed1')) for a in ('control','strip','role')}
        det=ps['role']['detected'];obs=np.full(len(det),-1);obs[det]=np.arange(det.sum());subgroup=valid&np.array([x[typ] for x in t]);available=np.flatnonzero(det&subgroup.any(-1))
        errors={a:np.linalg.norm(p['points']-gt,axis=-1) for a,p in ps.items()};diff=np.where(valid,errors['strip']-errors['control'],0).sum(-1)/valid.sum(-1)
        examples=[(int(sorted(available,key=lambda i:digest('v16-gallery:'+m['rows'][i]['id']))[0]),'SHA 고정'),(int(available[np.argmin(diff[available])]),'선 특징이 더 좋은 예'),(int(available[np.argmax(diff[available])]),'선 특징이 더 나쁜 예')]
        for i,why in examples:
            r=m['rows'][i];raw=PREP.raw_image(r);canvas=raw.copy()
            if r['mask'] is not None:
                a,u,b,v=r['mask'];canvas[u:v,a:b]=r['fill_bgr']
            if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
            possible=np.flatnonzero(subgroup[i]);j=int(possible[np.argmax(np.abs(errors['strip'][i,possible]-errors['control'][i,possible]))]);oi=int(obs[i])
            with torch.no_grad():out=model(subset(O,slice(oi,oi+1)),subset(rep,slice(oi,oi+1)),'role')
            assert np.max(np.abs(out['points'][0].numpy()-ps['role']['points'][i]))<.001
            selected=int(out['choice'][0,j]);centers=out['sample_center'][0,j,selected].numpy();probs=out['role_logits'][0,j,selected].softmax(-1).numpy();support=probs[...,1:3].sum(-1)
            fig,axes=plt.subplots(2,2,figsize=(14,10));ax=axes[0,0];ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));orig=gt[i]+np.array(r['origin']);ax.scatter(*orig[valid[i]].T,marker='+',s=75,color='#16a34a')
            for n in np.flatnonzero(valid[i]):ax.annotate(str(n),orig[n],color='#16a34a')
            ax.set_title('원본과 직접 클릭한 GT · 번호 유지')
            ax=axes[0,1];ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946');box=np.array(r['box']);span=np.linalg.norm(box[2:]-box[:2])
            for l in range(3):
                ax.plot(*centers[l].T,color='#64748b',alpha=.6,lw=1);h=ax.scatter(*centers[l].T,c=support[l],vmin=0,vmax=1,cmap='viridis',s=28)
            ax.scatter(*gt[i,j],marker='+',color='#16a34a',s=150,label=f'점{j} GT')
            for a in ('control','strip','role'):ax.scatter(*ps[a]['points'][i,j],facecolors='none',edgecolors=COLORS[a],s=100,label=NAMES[a].replace('V16 ',''))
            pts=np.concatenate((gt[i,valid[i]],ps['role']['candidates'][i,j],box.reshape(2,2)));lo=pts.min(0)-.12*span;hi=pts.max(0)+.12*span;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_title('역할 모델이 실제 읽은 선 샘플 위치 · 선택 후보 주변')
            ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2);fig.colorbar(h,ax=ax,fraction=.03,pad=.02,label='해당 점의 경계 역할 확률')
            ax=axes[1,0];hm=ax.imshow(support,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(7),[f'{x:+.2f}' for x in np.linspace(-.35,.35,7)]);ax.set_yticks(range(3),['연관 선 A','연관 선 B','연관 선 C']);ax.set_xlabel('선 방향 샘플 위치 / 상자 대각선');ax.set_title('긴 경계 + 높이 경계 역할 확률 · 정답 확률 보장 아님')
            for l in range(3):
                for z in range(7):ax.text(z,l,f'{support[l,z]:.2f}',ha='center',va='center',color='white' if support[l,z]<.5 else 'black',fontsize=9)
            fig.colorbar(hm,ax=ax,fraction=.03,pad=.02)
            ax=axes[1,1];p=ps['role'];ec=np.linalg.norm(p['candidates'][i,j]-gt[i,j],axis=-1);truth=ec[0]-ec;x=np.arange(8)
            ax.bar(x-.2,p['gain'][i,j],.4,color='#f97316',label='역할 모델 예측 점수');ax.bar(x+.2,truth,.4,color='#14b8a6',label='GT 사후 개선량');ax.axhline(0,color='#475569',lw=1)
            ax.set_xticks(x,['유지','복원0.5','복원1','작은선0.5','작은선1','큰선A','큰선B','큰선C'],rotation=30,ha='right');ax.set_ylabel('기준 대비 오차 감소(px)');ax.set_title(f'역할 모델의 후보 선택: {selected}');ax.legend(fontsize=8)
            fig.suptitle(f'{LABELS[condition]} · {why} · {r["id"]} · seed1\n선 특징 - 대조군: {diff[i]:+.3f}px (프레임 평균) · 샘플/역할 시각화이며 saliency map 아님',fontsize=12)
            fig.tight_layout(rect=(0,.02,1,.93),h_pad=4);name=f'{condition}_{len(rows):02d}.png';fig.savefig(d/name,dpi=115,bbox_inches='tight',pad_inches=.15);plt.close(fig)
            rows.append(dict(condition=condition,id=r['id'],point=j,path='figures/'+name,selection=why,difference_px=float(diff[i]),role_candidate=selected,scope='Posthoc GT selects illustrative examples/points only. Actual sample locations/role probabilities, not causal attention explanation.'))
    write(RUN/'GALLERY.json',dict(rows=rows));return rows

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    s=read(RUN/'SUMMARY.json');audit=read(RUN/'AUDIT.json');integ=read(RUN/'INTEGRATION_CHECK.json');diag=read(RUN/'DIAGNOSIS.json');post=read(RUN/'POSTHOC_INTERVENTIONS.json')
    assert audit['status']==integ['status']=='PASS' and not s['stable_arms'];tradeoff(s);g=gallery()
    measured=[(NAMES[a],[avg(s['results']['real/'+c],a,t) for c,t in FOCUS]) for a in ('stock','base','amodal','v14','balanced_old')+ARMS]
    lines=['# 후보별 선 특징: 심한 잘림은 개선, 전체 안정성은 미달','',
        '선 구간을 읽는 구조(strip)는 같은 후보를 쓰는 대조군보다 잘림40% 화면 밖 점 오차를 27.696→26.186px로 줄였다. 세 시드 개선율은 6.02%, 7.69%, 2.66%이며, 평균 차이 -1.511px의 실제 9세션 paired bootstrap 95% 구간은 -2.778~-0.349px였다. 해당 조건의 추가 개선 기준은 통과했다.', '',
        '그러나 잘림20%는 12.515→13.538px로 악화했다. 이번 5구조 모두 원본·약한 잘림 보존과 어려운 점의 기존 복원 이득80% 유지를 동시에 충족하지 못했다. “선과 점은 결합 불가능”이라는 결론은 아니다. 현재 고정 특징·후보 선택 구조의 부분 효과와 남은 실패를 구분한다.', '',
        '## 실제 주요 결과','',
        '3시드 평균px, 작을수록 좋음. 원본은 검출된 프레임의 평균, 다른 열은 해당 조건에서 가려진 점/화면 밖 점의 평균이다. 조건 사이 평가 대상이 다르다.', '',
        '| 방법 | 원본 | 코너 가림 | 잘림20% 밖 | 잘림40% 밖 | 잘림10% 밖 | 잘림30% 밖 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in measured:lines.append('| '+name+' | '+' | '.join(f'{x:.3f}' for x in v)+' |')
    lines+=['','## 실제로 선의 영상 근거를 썼는가','',
        '사전에 정한 role 모델의 선 특징 끄기/다른 영상 특징 개입을 수행했다. role은 잘림40%에서 다른 영상 특징 대비 확실한 차이를 보이지 않았다. 개선이 더 컸던 strip도 사후에 같은 개입으로 검사했다. 이 추가 진단은 결과를 본 뒤 정한 것으로, 학습이나 모델 선택에는 쓰지 않았다.', '',
        'strip의 실제 선 주변 영상 특징은 다른 영상 특징보다 잘림40%에서 평균0.382px 더 좋았다. 세 시드 모두1% 이상 좋았고, 차이의 세션95%구간은 -0.802~-0.044px였다. 특징을 완전히 끄는 것보다도1.526px 좋았다. 이 조건에서는 이미지별 선 주변 정보를 활용했다는 근거가 생겼다. 단, 다른 영상 교체는 분포 밖 개입이며, 특정 파렛트 외곽선을 정확히 인식했음을 보장하지 않는다.', '',
        '잘림20%에서는 실제 특징이 다른 영상 특징보다 오히려 평균0.385px 나빴다(95%구간 +0.025~+0.914px). 모델이 영상에 반응하는 것과 올바르게 반응하는 것은 별개였다. 단순히 선 정보가 무시되는 문제 하나로 현재 실패를 설명할 수 없다.', '',
        '## 무엇이 점을 개선했는가','',
        '잘림40%에서 strip이 기준점 대비 얻은 평균 이득의 대부분은 기존 복원 후보 선택에서 나왔다. 후보 종류별 합산은 GAIN_DECOMPOSITION.json에 있다. 선 후보 자체가 정답점으로 많이 이동시킨 결과로 해석하지 않는다. 점 주변의 선 특징을 복원 선택에 사용하는 경로는 부분적으로 작동했다.', '',
        '## 경계 역할 학습을 붙이면 왜 더 좋아지지 않았나','',
        'role은 잘림20% 악화를 줄였지만 잘림40%는 strip의26.186px에서27.447px로 나빠졌다. strip 대비 악화 +1.262px의 세션95%구간은 +0.239~+2.466px였다. 역할 분류 손실이 전체 보정 선택에 필요한 표현과 자동으로 일치하지 않았다.', '',
        '합성 원본 평가에서 role의 해당 꼭짓점 긴 경계 F1은0.541, 짧은 높이 경계 F1은0.327이었다. 높이 경계 precision은0.243으로, 보조 역할을 신뢰할 만큼 정확하게 구별하지 못했다. 실제 이미지에는 신뢰할 수 있는 선 가시성 정답이 없어 이 F1을 실사 의미 분류 성능으로 확장하지 않는다.', '',
        '가능한 원인은 세 가지다. 첫째, 원래 점 GT로 만든 cuboid 경계는 실제로 보이는 목재 외곽선과 다를 수 있다. 둘째, 짧은 높이선은 24×24 고정 특징에서 구별하기 어려울 수 있다. 셋째, “경계처럼 보임”과 “이 복원 후보가 정답에 가까워짐”은 서로 다른 학습 목표다. 앞의 분류 정확도·복원 선택 변화는 측정했고, 실제 mesh 가시성 정답이나 고해상도 특징 재학습으로 세 가설을 각각 분리하는 실험은 이번에 하지 않았다.', '',
        '공격적으로 복원하는 median_role은 모든 시드에서 어려운 점의 복원 이득80%를 유지했지만, 약한 잘림을 크게 악화시켰다. 사례 균형을 적용해도 전체 안정성을 확보하지 못했다. 따라서 gradient surgery, 추가 학습 시간, 보수성 조절 중 하나가 해결책이라고 이번 결과로 단정할 수 없다.', '',
        '## 최종 구조와 다음 판단','',
        '구현한 EvidencePointLineNetwork는 고정 YOLO/점/복원/DHT 부모에 후보별 선 구간 토큰 평가기를 연결한 구조다. 평가기56,686파라미터를 학습했으며 단순 추론 필터가 아니다. 원래 점들을 연결해 선을 만드는 방식도 아니다. 다만 이번은 부모 고정 실험이며 전체 end-to-end 아키텍처 학습은 아니다.', '',
        '재사용 평가에서 부분 개선은 확인했으므로 결합 자체를 포기할 근거는 없다. 그러나 현재 구조를 안정적인 최종 모델로 채택할 근거도 부족하다. 추가 투자를 한다면 동일 손실의 strip 대조군을 유지하면서 실제 가시 경계의 감독과 더 세밀한 선 특징이 후보 선택 오류를 줄이는지 분리해 검사하는 것이 타당하다. 새 현장/자연 가림 데이터가 없으면 배포 안정성은 최종 확인할 수 없다.', '',
        '## 검증과 범위','',
        f'총15모델·30,000업데이트의 손실/gradient가 유한함을 확인했다. 독립 검산 {audit["prediction_frame_rows"]:,}예측 프레임 행, {audit["selection_frame_rows"]:,}선택 프레임 행을 검사했다. 후보 재구성 최대 차이 {audit["candidate_replay_max_px"]:.8f}px, 지표 재계산 최대 차이 {audit["metric_replay_max"]:.3g}. 보조 정답2400샘플은 float64 별도 구현과 일치했다.', '',
        f'통합15모델×6조건×4프레임(360프레임)의 선택 ID는 캐시 평가와 일치했고 최대 좌표 차이는 {integ["max_error_px"]:.8f}px였다. 이전 모델/학습 입력 해시를 확인했고, 원래 수동 GT ID와 crop 좌표 이동을 보존했다. 사후 strip 개입은 별도로 후보 불변·선택 재현을 검산했다.', '',
        '실제297장·9세션·수동1251점과 합성505장의 재사용 DEV이다. 자연 가림 라벨은unknown이다. 수동 재클릭이나 선 가시성 신규 주석은 하지 않았다. 선 역할 정답은 기존 합성 GT에서 유도한 대리 정답이다. 실험의 실제 GT는 추론·학습·시드 선택에 들어가지 않았다. 검출 실패 포함 정규화 오차, PCK5/10, P90, 모든 조건·시드, 악화 적용 건수와 세션별 차이를 함께 공개한다. 95%구간은 다중 비교 보정이 없는 탐색적 구간이다. 이전41.49px와는 평가 대상/지표가 달라 직접 비교하지 않는다.', '',
        '이전 파일과 배포 모델은 교체하지 않았다. export/에 모든 시드의 재현용 통합 모델을 저장했다.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n');(RUN/'DESIGN_KO.md').write_text((ROOT/'_docs/notes/pallet_line_evidence_v16_design.md').read_text())
    (ROOT/'_docs/notes/pallet_line_evidence_v16_results.md').write_text('\n'.join(lines)+'\n')
    write(RUN/'CONCLUSION.json',dict(status='CANDIDATE_LINE_EVIDENCE_PARTIAL_GAIN_FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],strip_crop40_pass=s['line_gates']['strip']['crop40'],
        auxiliary_role_does_not_improve_stability=True,posthoc_actual_vs_donor_crop40=post['comparisons']['real/crop40']['strip_donor'],adoption=False,no_real_tuning=True))
    table='<table><tr><th>방법</th>'+''.join('<th>'+LABELS[c]+('</th>' if t is None else ' 대상점</th>') for c,t in FOCUS)+'</tr>'+''.join('<tr><td>'+n+'</td>'+''.join(f'<td>{x:.3f}</td>' for x in v)+'</tr>' for n,v in measured)+'</table>'
    body=[]
    for line in lines[1:]:
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line and not line.startswith('|'):body.append('<p>'+html.escape(line)+'</p>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V16 후보별 선 근거 — 부분 개선과 안정성 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}h1{font-size:29px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#0369a1}.note{color:#596579}.verdict{border-left:5px solid #b45309}.focus{background:#eff6ff}.scroll{overflow-x:auto}h2{margin-top:28px}</style>
<h1>선 구간 정보는 부분적으로 유효 · 전체 안정성은 미달</h1><p>5구조 × 3시드 · 30,000업데이트 · 합성만 학습 · 실제297장/합성505장 × 8조건</p><section class="verdict"><b>심한 잘림 27.70 → 26.19px, 약한 잘림 12.52 → 13.54px.</b><p>동일후보 대조군 대비 선 구간 특징의 효과입니다. 경계 역할 학습은 악화와 유용한 복원을 함께 줄였습니다. 사전 안정성 기준을 모두 통과한 구조는 없습니다.</p><div class="scroll">'''+table+'''</div><p class="note">3시드 평균px. 원본은 프레임 평균, 나머지는 해당 가림/잘림 대상점 평균입니다.</p></section>
<section><h2>어떤 정보를 어떻게 합쳤나</h2><p>고정 YOLO → 점 보정·복원 + Deep Hough → 8개 점 후보 → 후보 주변 선 위/양옆 특징 → 후보별 경계 역할과 보정 점수 → 유지/복원 선택</p><p>점 주변 선의 증거를 복원 선택에 사용합니다. GT를 넣어 선에 강제로 맞추는 필터가 아닙니다. 이번은 부모 고정 실험이며 선의 보조 정답은 실제 가시 경계가 아닌 기하적 대리 정답입니다.</p><a href="DESIGN_KO.md">구조·통제·학습·사전 기준</a><img src="tradeoff.png"><p class="note">초록 세로선 왼쪽과 가로선 위를 동시에 만족해야 합니다. 원/사각/삼각은 별도 학습 시드입니다.</p></section>
<section><h2>전체 평가 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p></section>
<section><h2>실제로 어디를 읽었는가</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">실제 샘플 위치와 학습한 경계 역할 확률입니다. 원인 설명을 보장하는 attention/saliency map은 아닙니다. GT는 예시/점 선정과 사후 진단에만 사용했습니다. 회색 영역은 입력에서 잘린 부분입니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서</a> · <a href="SUMMARY.json">전체 수치·판정</a> · <a href="DIAGNOSIS.json">역할 정확도·악화·세션 진단</a> · <a href="POSTHOC_INTERVENTIONS.json">다른 영상 개입</a> · <a href="GAIN_DECOMPOSITION.json">후보 종류별 개선</a> · <a href="AUDIT.json">독립 검산</a> · <a href="INTEGRATION_CHECK.json">통합 검증</a> · <a href="PROTOCOL.json">사전 등록</a></p></section><script>
const data='''+json.dumps(s['results'],ensure_ascii=False)+''', labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''',gallery='''+json.dumps(g,ensure_ascii=False)+''';
const params=new URLSearchParams(location.search);for(const id of ['split','seed','metric'])if(params.has(id))document.getElementById(id).value=params.get(id);
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>방법</th>'+Object.values(labels).map(x=>'<th>'+x+'</th>').join('')+'</tr>';for(const a of Object.keys(names)){out+='<tr'+(a==='strip'?' class="focus"':'')+'><td>'+names[a]+'</td>';for(const c of Object.keys(labels)){const r=data[split+'/'+c],rs=a==='stock'?[r.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>r[a+'_seed'+s]),v=rs.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:metric.startsWith('pck')?x.strata.all[metric]:x[metric]);out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;document.getElementById('counts').textContent='검출: '+Object.keys(labels).map(c=>labels[c]+' '+data[split+'/'+c].stock.detected_frames+'/'+data[split+'/'+c].stock.frames).join(' · ');}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · seed1 · 선 특징 - 대조군 '+r.difference_px.toFixed(3)+'px';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page);write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),figures=len(g),source_sha256=sha(Path(__file__))))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
