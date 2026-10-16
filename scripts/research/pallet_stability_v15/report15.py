"""Safety/usefulness tradeoff report; no post-evaluation retuning or adoption."""
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context15 import *

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'새 잘림10%','crop30':'새 잘림30%'}
NAMES={'stock':'기존 YOLO','base':'V12 보정 억제 기준','amodal':'V12 전체 복원','v14':'V14 점·선 융합','median_old':'V15 중앙값 기준','risk_old':'V15 보수적 선택','balanced_old':'V15 보수적 + 사례 균형','balanced_rich':'V15 균형 + 큰 선 후보','rich_off':'큰 후보 모델 · 선 끄기','rich_donor':'큰 후보 모델 · 다른 영상 선'}
COLORS={'median_old':'#f97316','risk_old':'#2563eb','balanced_old':'#16a34a','balanced_rich':'#a855f7'}

def avg(rows,arm,typ=None):
    r=[rows['stock']] if arm=='stock' else [rows[f'{arm}_seed{s}'] for s in SEEDS]
    return float(np.mean([x['strata'][typ]['mean_px'] if typ else x['frame_mean_px'] for x in r]))

def tradeoff(s):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,(condition,typ) in zip(axes,[('corner20','hidden'),('crop40','outside')]):
        ax.axvspan(-2,1,ymin=.75,ymax=1,alpha=.08,color='green');ax.axvline(1,c='#16a34a',ls='--');ax.axhline(80,c='#16a34a',ls='--')
        for arm in ARMS:
            for seed,marker in zip(SEEDS,('o','s','^')):
                mild=s['results']['real/crop20'];hard=s['results']['real/'+condition];n=f'{arm}_seed{seed}';b=f'base_seed{seed}';a=f'amodal_seed{seed}'
                x=100*(mild[n]['strata']['outside']['mean_px']/mild[b]['strata']['outside']['mean_px']-1)
                y=100*(hard[b]['strata'][typ]['mean_px']-hard[n]['strata'][typ]['mean_px'])/(hard[b]['strata'][typ]['mean_px']-hard[a]['strata'][typ]['mean_px'])
                ax.scatter(x,y,c=COLORS[arm],marker=marker,s=60,label=NAMES[arm] if seed==1 else None)
        ax.scatter(0,0,c='black',marker='x',s=70);ax.annotate('유지만 할 때',(0,0),xytext=(4,-7),textcoords='offset points',fontsize=8)
        ax.set_xlim(-2,37);ax.set_ylim(-10,110);ax.set_xlabel('약한 잘림의 기준 대비 악화(%) · 1% 이내 필요');ax.set_ylabel('기존 복원 이득 유지율(%) · 80% 이상 필요')
        ax.set_title(LABELS[condition]+'의 복원 이득과 악화 억제');ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    fig.suptitle('보수적으로 선택하면 악화는 줄지만 유용한 복원도 줄어듦\n원/사각/삼각: seed1/2/3 · 각 점은 별도 학습한 모델, 임계값 탐색 아님',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(RUN/'tradeoff.png',dpi=135,bbox_inches='tight');plt.close(fig)

def gallery():
    directory=RUN/'figures';directory.mkdir(exist_ok=True);g=[]
    for condition in ('corner20','crop20','crop40','crop10'):
        m=manifest('real',condition);t=targets('real',condition);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=C.P.target_type(t).numpy()
        preds={a:np.load(RUN/'predictions/real'/condition/f'{a}_seed1.npz') for a in ARMS};p=preds['balanced_rich'];det=p['detected'];rep=repcache('real',condition,1);ix=np.full(len(det),-1);ix[det]=np.arange(det.sum())
        err={a:np.linalg.norm(x['points']-gt,axis=-1) for a,x in preds.items()};diff=np.where(valid,err['balanced_old']-err['median_old'],0).sum(-1)/valid.sum(-1)
        subgroup=valid&(kind==(1 if condition=='corner20' else 2));available=np.flatnonzero(det&subgroup.any(-1))
        examples=[(int(sorted(available,key=lambda i:digest('v15-gallery:'+m['rows'][i]['id']))[0]),'SHA 고정'),
            (int(available[np.argmin(diff[available])]),'보수적 균형 선택이 더 좋은 예'),(int(available[np.argmax(diff[available])]),'보수적 균형 선택이 더 나쁜 예')]
        for i,why in examples:
            r=m['rows'][i];raw=PREP.raw_image(r);canvas=raw.copy()
            if r['mask'] is not None:
                a,u,b,v=r['mask'];canvas[u:v,a:b]=r['fill_bgr']
            if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
            poss=np.flatnonzero(subgroup[i]);j=int(poss[np.argmax(np.abs(err['balanced_old'][i,poss]-err['median_old'][i,poss]))]);oi=ix[i]
            fig,axes=plt.subplots(1,3,figsize=(19,6));orig=gt[i]+np.array(r['origin']);axes[0].imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));axes[0].scatter(*orig[valid[i]].T,marker='+',s=80,c='#22c55e');axes[0].set_title('원본 · 직접 클릭한 GT')
            for n in np.flatnonzero(valid[i]):axes[0].annotate(str(n),orig[n],c='#22c55e')
            ax=axes[1];ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946');box=np.array(r['box']);span=np.linalg.norm(box[2:]-box[:2]);center=(box[:2]+box[2:])/2
            lines=rep['lines'][oi].numpy();w=rep['line_weights'][oi,j].numpy();top=np.argsort(-w[:-1],kind='stable')[:3]
            for k in top:
                l=lines[k];norm=max(np.linalg.norm(l[:2]),1e-8);n=l[:2]/norm;c=l[2]/norm;foot=center-(center@n+c)*n;tangent=np.array([-n[1],n[0]]);ends=foot+np.array([-3,3])[:,None]*span*tangent;ax.plot(*ends.T,c='#0891b2',alpha=.45,lw=1)
            ax.scatter(*gt[i,j],marker='+',c='#22c55e',s=130,label=f'점 {j} GT')
            ax.scatter(*p['candidates'][i,j,0],marker='x',c='black',s=70,label='기준점')
            for a in ARMS:ax.scatter(*preds[a]['points'][i,j],s=65,facecolors='none',edgecolors=COLORS[a],label=NAMES[a].replace('V15 ',''))
            ax.scatter(*p['candidates'][i,j,5:].T,marker='.',s=90,c='#64748b',label='큰 선 후보3개')
            pts=np.concatenate((gt[i,valid[i]],p['candidates'][i,j],box.reshape(2,2)));lo=pts.min(0)-.1*span;hi=pts.max(0)+.1*span;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_title('ROI 확대 · 선 후보와 각 모델의 최종 점');ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
            ec=np.linalg.norm(p['candidates'][i,j]-gt[i,j],axis=-1);truth=ec[0]-ec;x=np.arange(8)
            axes[2].bar(x-.2,p['gain'][i,j],.4,color='#a855f7',label='보수적 모델 점수');axes[2].bar(x+.2,truth,.4,color='#14b8a6',label='GT 사후 개선량');axes[2].axhline(0,c='#475569',lw=1)
            axes[2].set_xticks(x,['유지','복원0.5','복원1','작은선0.5','작은선1','큰선A','큰선B','큰선C'],rotation=35,ha='right');axes[2].set_ylabel('기준 대비 오차 감소(px), 양수가 좋음');axes[2].legend(fontsize=8)
            axes[2].set_title(f'큰 선 후보 모델: 선택 {int(p["choice"][i,j])}')
            fig.suptitle(f'{LABELS[condition]} · {why} · {r["id"]}\n균형 선택 - 중앙값 선택: {diff[i]:+.3f}px (프레임 평균) · seed1',fontsize=12)
            fig.tight_layout(rect=(0,.03,1,.89));name=f'{condition}_{len(g):02d}.png';fig.savefig(directory/name,dpi=125,bbox_inches='tight',pad_inches=.15);plt.close(fig)
            g.append(dict(condition=condition,id=r['id'],point=j,selection=why,path='figures/'+name,frame_difference_px=float(diff[i]),seed=1,scope='GT selects illustrative example/point only; predicted lines and model scores shown without amplified point movement.'))
    write(RUN/'GALLERY.json',dict(rows=g));return g

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});s=read(RUN/'SUMMARY.json');audit=read(RUN/'AUDIT.json');integ=read(RUN/'INTEGRATION_CHECK.json');d=read(RUN/'STABILITY_DIAGNOSIS.json')
    assert audit['status']==integ['status']=='PASS';tradeoff(s);g=gallery()
    lines=['# 안정성 실험: 악화 억제는 가능, 큰 복원 이득 유지는 미달','',
        '보수적인 선택과 악화 사례 균형을 적용하면 원본·약한 잘림의 악화를 거의 제거할 수 있었다. 같은 모델은 코너 가림과 심한 잘림에서 기준보다 작지만 세 시드 모두 일관된 개선을 보였다. 다만 기존 복원 이득의80%를 유지하는 사전 기준에는 못 미쳤다. 큰 선 후보를 추가해도 전체 안정성 기준을 통과하지 못했다.', '',
        '## 무엇이 안정성에 필요한가','',
        '이번 결과는 악화를 피하는 것과 큰 개선을 유지하는 것을 함께 확인해야 한다는 점을 보여준다. 무조건 유지하는 모델은 악화가 적지만 유용한 복원도 못 한다. 그래서 원본·약한 잘림의 오차를 기준 대비1% 이내로 유지하면서, 코너 가림·심한 잘림의 기존 복원 이득80% 이상을 남겨야 통과하도록 사전에 정했다.', '',
        '실험에서는 평균에 가까운 후보 선호를 보수적으로 바꾸는 경로와, 잘못된 보정 사례에 충분한 학습 가중치를 주는 경로, 더 큰 선 후보를 생성하는 경로를 차례로 비교했다.12개 모델 중 이 두 요구와 새 잘림 비율 조건을 모두 통과한 모델은 없다. 결과에 맞춰 기준이나 임계값을 바꾸지 않았다.', '',
        '## 통제한 학습 실험','',
        '중앙값 기준(tau0.5) → 보수적 선택(tau0.2) → 악화·중립·개선 사례 균형 → 큰 선 후보3개 추가. 같은8슬롯 평가기 구조·초기화·batch순서를 사용했고, 기존후보 대조군은 새3슬롯을 비활성화했다. 각3시드·2000업데이트·batch16, 총24000업데이트. V14와 달리 후보선호 CE 없이 pinball 손실만 사용했다.', '',
        '학습은 V14의 분리된 합성253장,6조건에서 검출된1462관측만 사용했다. 이 자료는 학습용이며 평가로 사용하지 않는다. 악화<-0.25px / 중립 / 개선>0.25px을 후보별·합성 노출 상태별로 역빈도 가중치(최대10)로 균형화했고 유효 학습점을 버리지 않았다. YOLO·점 보정·복원·DHT 부모는 고정했다.', '',
        '큰 선 후보는 DHT의 상위3개 점별 연관 선으로 향하는 법선 이동이며, 최대 이동량을 상자 대각선5%로 제한했다. 유지와 함께 선택할 후보로 제공한다. GT로 거르는 추론 필터·self-training·최종점의 강제 선 투영은 아니다.', '',
        '## 실제 주요 결과','',
        '3개 시드 평균px, 작을수록 좋음. 원본은 프레임 평균, 나머지는 가려진 점 또는 화면 밖 점 평균이다. 행 간 서로 다른 대상을 직접 비교하지 않는다.', '',
        '| 방법 | 원본 | 코너 가림280점 | 잘림20% 화면 밖558점 | 잘림40% 화면 밖571점 |',
        '|---|---:|---:|---:|---:|']
    summary_table=[]
    for arm in ('base','amodal','v14')+ARMS:
        values=[avg(s['results']['real/'+c],arm,typ) for c,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]]
        lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |');summary_table.append((NAMES[arm],values))
    lines+=['',
        '보수적+사례 균형 모델은 원본과 잘림10/20/30%의 악화1% 이내 기준을 모든 시드에서 통과했다. 약한 잘림의 기준 대비 평균 차이는 +0.0009px,9세션95%구간 -0.0011~+0.0029px였다. 원래 점을 대부분 유지해 이전의 큰 악화를 억제했다.', '',
        '동시에 코너 가림 오차는 기준16.539→16.288px, 심한 잘림은27.771→27.536px로 세 시드 모두 개선했다. 기준 대비 차이의9세션95%구간은 각각 -0.612~-0.039px, -0.482~-0.116px였다. 이 작고 일관된 개선은 관찰된 결과다. 하지만 기존 전체 복원의13.554px,19.921px에 크게 못 미쳐 큰 개선까지 유지한 안정적 성공으로 판정하지 않았다.', '',
        '보수적 선택만 적용하면 코너 가림15.577px, 심한 잘림26.775px로 약1px의 기준 대비 개선을 남겼다. 그러나 기존 복원 이득80% 기준은 실패했고, 새 잘림10%에서도 seed3가 악화1% 상한을 넘었다. 평균이 좋아진다는 이유로 모든 구도에서 안정적이라고 판단하지 않는다.', '',
        '## 학습하지 않은 잘림 비율','',
        '원래 합성505장·실제297장에 잘림10%,30%를 추가해 새로 YOLO 추론했다. 실제는 두 비율 모두293장이 검출됐고 화면 밖 유효점은 각각489개,601개다. 새 비율은 사전 등록했지만 같은 원본 장면을 재사용하므로 새로운 현장 검증은 아니다.', '',
        '| 방법 | 새 잘림10% 화면 밖 | 새 잘림30% 화면 밖 |','|---|---:|---:|']
    for arm in ('base','amodal','v14')+ARMS:
        lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{avg(s["results"]["real/"+c],arm,"outside"):.3f}' for c in NEW_CONDITIONS)+' |')
    lines+=['',
        '## 큰 선 후보를 만들면 해결되나','',
        '새 선 후보의 평균 이동량은 어려운 점에서 약5.5~6.2px로 커졌다. 정답을 알고 최선의 후보를 고르는 특권 진단에서는 기존 작은 선 후보보다 추가 개선 여지가 커졌다. 다음 수치는 실제 추론 성능이 아니라 후보 집합의 사후 가능성이다.', '',
        '| 대상 | 기존5후보 GT 최선 | 새8후보 GT 최선 | 추가 개선 여지 |','|---|---:|---:|---:|']
    for c,typ in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        v=[d['rows']['real/'+c][f'balanced_rich_seed{i}'][typ] for i in SEEDS];means=[np.mean([r[k] for r in v]) for k in ('oracle_old5_px','oracle_rich8_px','oracle_extra_gain_px')]
        lines.append('| '+LABELS[c]+' | '+' | '.join(f'{x:.3f}' for x in means)+' |')
    lines+=['',
        '하지만 학습한 큰 후보 모델은 약한 잘림12.576px, 심한 잘림27.765px로 보수적+균형 기존후보 모델보다 나빴다. 약한 잘림의 큰 선 후보 적용 건수는9/78/43점이었고 그중4/67/35점이 악화했다. 나머지에서는 좋은 후보가 있어도 유지하는 경우가 많았다. 후보의 가능한 이득과 실제로 올바른 후보를 골라내는 능력 사이에 차이가 남았다.', '',
        '## 결과가 가리키는 다음 설계','',
        '작은 보정을 보수적으로 사용하는 경로는 악화 억제와 소폭 개선에 유효했다. 큰 복원을 안정적으로 살리려면 각 후보가 실제 파렛트 외곽선에 근거하는지, 정답 방향으로 이동하는지를 구별하는 능력이 필요하다. 보수성만 높이는 것으로 큰 복원 이득까지 확보되지는 않았다. 복원이 줄어드는 현상은 합성 검증에서도 나타나므로 실사 전이 문제 하나만으로 설명하지 않는다.', '',
        '현재 평가기는 후보 주변 영상과 공통 선 연관 요약을 읽지만, 후보를 만든 개별 선의 연속적인 영상 지지나 외곽선 역할을 직접 감독하지 않는다. 다음에는 후보별 선 위·주변 영상 특징, 관측 가능한 선 구간, 외곽선 역할을 명시적으로 학습하는 방향을 검증할 가치가 있다. 이는 아직 학습해 확인하지 않은 후속 가설이며 성능 보장이 아니다. 정답 line association을 추론 때 주는 것으로 성공을 주장하지 않는다.', '',
        '## 검증과 해석 범위','',
        f'독립 검산 PASS: 예측 프레임 행{audit["prediction_frame_rows"]:,}개, 선택 프레임 행{audit["selection_frame_rows"]:,}개. 후보 재구성 최대 차이{audit["candidate_replay_max_px"]:.8f}px, 지표 재계산 최대 차이{audit["metric_replay_max"]:.3g}. 같은 초기화·batch순서, 모든24000유한 업데이트, 부모/학습소스 해시와 새 GT 좌표 이동을 확인했다.', '',
        f'통합 모델 StablePointLineNetwork는 기존 V14의 불필요한 평가기를 제거하고 부모 특징·후보와 V15 평가기를 연결했다.12개 모델×6조건×4프레임에서 캐시 평가와 선택 ID가 일치했고 최대 좌표 차이는{integ["max_error_px"]:.8f}px였다. 모든 실험 체크포인트를 export/에 저장했으며 실제 평가로 시드를 고르지 않았다.', '',
        '실제297장·9세션·직접 클릭1251점, 합성505장의 재사용 평가다. 자연 가림 라벨은unknown이며 인공 가림·잘림 결과다. 수동 클릭오차를 새로 측정하지 않았다. 기존 reflect100 padding과 원래 점 ID, 검출 실패 포함 지표를 유지했다. 하위 분위수 점수의 실사 정확도/coverage는 보장되지 않으며 사례 균형은 학습 목표 분포도 바꾼다. 자연 가림·새 현장·최종6D 성능이나 이전41.49px와 직접 비교하지 않는다.', '',
        '사전 정의한 전체 안정성 기준은 네 구조 모두 실패다. 보수적+균형 모델의 소폭 개선은 별도로 기록하되 이를 기준을 완화한 최종 합격으로 바꾸지 않았다. 기존 배포 모델을 교체하지 않았다.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n');(RUN/'DESIGN_KO.md').write_text((ROOT/'_docs/notes/pallet_stability_v15_design.md').read_text())
    write(RUN/'CONCLUSION.json',dict(status='PARTIAL_SMALL_CONSISTENT_GAIN_FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],gates=s['gates'],line_gates=s['line_gates'],
        descriptive_finding='balanced_old suppresses mild/new-crop regressions and improves hard strata modestly in all seeds; does not retain80% previous recovery gains.',adoption=False,real_tuning=False))
    table='<table><tr><th>방법</th><th>원본</th><th>코너 가림</th><th>잘림20% 밖</th><th>잘림40% 밖</th></tr>'+''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+'</tr>' for name,vals in summary_table)+'</table>'
    body=[]
    for line in lines[1:]:
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line and not line.startswith('|'):body.append('<p>'+html.escape(line)+'</p>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>안정성 실험 — 악화 억제와 복원 이득</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}h1{font-size:29px}h2{margin-top:26px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#0369a1}.note{color:#596579}.verdict{border-left:5px solid #b45309}.focus{background:#ecfdf5}.scroll{overflow-x:auto}</style>
<h1>악화 억제와 소폭 개선은 확인 · 큰 복원 이득 유지는 미달</h1><p>4구조 × 3시드 · 24000업데이트 · 실제 학습0장 · 새로운 잘림10%·30% 추가 평가</p><section class="verdict"><b>전체 안정성 기준을 통과한 구조는 없습니다.</b><p>보수적+사례 균형 모델은 원본·약한 잘림을 유지하고 어려운 점에서도 소폭 개선했습니다. 다만 기존 큰 복원 이득은 대부분 잃었습니다. 큰 선 후보는 선택 가능한 이득을 늘렸지만 모델이 안정적으로 활용하지 못했습니다.</p><div class="scroll">'''+table+'''</div><p class="note">3시드 평균px, 작을수록 좋음. 원본은 프레임 평균, 나머지는 가려진/화면 밖 대상 점 평균입니다.</p></section>
<section><h2>악화 억제와 유용한 복원 사이의 상충</h2><a href="tradeoff.png" target="_blank"><img src="tradeoff.png"></a><p>초록 점선의 왼쪽(약한 잘림 악화1% 이내)이면서 위쪽(복원 이득80% 이상 유지)에 있어야 합니다. 보정을 모두 끄면 왼쪽 아래에 머뭅니다.</p><a href="DESIGN_KO.md">구조·손실·사전 기준 설계</a></section>
<section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3개 시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 유효 점PCK5</option><option value="pck10">전체 유효 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p><p class="note">같은 입력·검출·원래 GT ID를 공유합니다. 새 비율에서도 원래 장면은 재사용했습니다. 새 비율의 다른 영상 선 진단은 수행하지 않아 —로 표시합니다.</p></section>
<section><h2>악화를 피한 예와 필요한 복원을 놓친 예</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">회색은 잘려 입력에 없는 영역입니다. 가운데 ROI는 확대했지만 이동량을 과장하지 않았습니다. GT는 예시/점 선정과 사후 개선량 진단에만 사용했습니다. 실제 추론은 모델 점수로 선택합니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서와 전체 표</a> · <a href="SUMMARY.json">모든 수치·기준 판정</a> · <a href="STABILITY_DIAGNOSIS.json">후보·세션·유지 진단</a> · <a href="AUDIT.json">독립 검산</a> · <a href="INTEGRATION_CHECK.json">통합 추론 검증</a> · <a href="PROTOCOL.json">사전 등록</a></p></section><script>
const data='''+json.dumps(s['results'],ensure_ascii=False)+''',labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''',gallery='''+json.dumps(g,ensure_ascii=False)+''';
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>방법</th>'+Object.values(labels).map(n=>'<th>'+n+'</th>').join('')+'</tr>';for(const a of Object.keys(names)){out+='<tr'+(a==='balanced_old'?' class="focus"':'')+'><td>'+names[a]+'</td>';for(const c of Object.keys(labels)){const rows=data[split+'/'+c],r=a==='stock'?[rows.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>rows[a+'_seed'+s]);if(r.some(x=>!x)){out+='<td>—</td>';continue;}const v=r.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:metric.startsWith('pck')?x.strata.all[metric]:x[metric]);out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;document.getElementById('counts').textContent='검출: '+Object.keys(labels).map(c=>labels[c]+' '+data[split+'/'+c].stock.detected_frames+'/'+data[split+'/'+c].stock.frames).join(' · ');}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · seed1 · 균형 선택 - 중앙값 선택 '+r.frame_difference_px.toFixed(3)+'px (음수 개선)';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page);write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),figures=len(g)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
