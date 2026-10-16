"""Korean architecture/comparison report with exact saved line and candidate evidence."""
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context14 import *
from eval14 import path_for

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림'}
NAMES={'stock':'기존 YOLO','v10_local':'V10 점 보정','v10_directional':'V10 선 방향 보정','v11_local_extra':'V11 추가 점 보정','v11_joint':'V11 DHT 공동학습','v12_gate':'V12 보정 억제','v12_amodal':'V12 억제 + 복원','v13_scorer':'V13 후보 평가','insample_fusion':'V14 기존 학습 이미지 융합','heldout_points':'V14 분리 학습 · 점','heldout_fusion':'V14 최종 점·선 후보','fusion_off':'V14 같은 모델 · 선 끄기','fusion_donor':'V14 같은 모델 · 다른 영상 선'}

def mean(rows,arm,typ=None,metric='frame_mean_px'):
    cells=[rows['stock']] if arm=='stock' else [rows[f'{arm}_seed{s}'] for s in SEEDS]
    return float(np.mean([r['strata'][typ]['mean_px'] if typ else r[metric] for r in cells]))

def figures():
    directory=RUN/'figures';directory.mkdir(exist_ok=True);gallery=[]
    for condition in ('clean','corner20','crop20','crop40'):
        m=read(V11/'cache/real'/condition/'manifest.json');t=target_rows('real',condition)
        p=np.load(path_for('real',condition,'heldout_fusion',1));reference=np.load(path_for('real',condition,'heldout_points',1))
        gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=P.target_type(t).numpy();det=p['detected']
        rep=repcache('real',condition,1);obsindex=np.full(len(det),-1);obsindex[det]=np.arange(det.sum())
        e=np.linalg.norm(p['points']-gt,axis=-1);er=np.linalg.norm(reference['points']-gt,axis=-1);ec=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1)
        diff=np.where(valid,e-er,0).sum(-1)/valid.sum(-1)
        subgroup=valid if condition=='clean' else valid&(kind==(1 if condition=='corner20' else 2))
        eligible=np.flatnonzero(det&subgroup.any(-1));examples=[(int(sorted(eligible,key=lambda i:digest('v14-gallery:'+m['rows'][i]['id']))[0]),'SHA 고정'),
            (int(eligible[np.argmin(diff[eligible])]),'점 대조군 대비 최대 개선'),(int(eligible[np.argmax(diff[eligible])]),'점 대조군 대비 최대 악화')]
        for i,why in examples:
            r=m['rows'][i];raw=PREP.raw_image(r);canvas=raw.copy()
            if r['mask'] is not None:
                a,u,b,v=r['mask'];canvas[u:v,a:b]=r['fill_bgr']
            if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
            possible=np.flatnonzero(subgroup[i]);j=int(possible[np.argmax(np.abs(e[i,possible]-er[i,possible]))]);oi=obsindex[i]
            lines=rep['lines'][oi].numpy();weights=rep['line_weights'][oi,j].numpy();top=np.argsort(weights[:-1])[-3:][::-1]
            fig,axes=plt.subplots(1,3,figsize=(19,6));original=gt[i]+np.array(r['origin'])
            axes[0].imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));axes[0].scatter(*original[valid[i]].T,marker='+',c='#22c55e',s=80);axes[0].set_title('원본 · 직접 클릭한 GT (초록 +)')
            for n in np.flatnonzero(valid[i]):axes[0].annotate(str(n),original[n],c='#22c55e')
            ax=axes[1];ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946')
            box=np.array(r['box']);span=np.linalg.norm(box[2:]-box[:2]);center=(box[:2]+box[2:])/2
            colors=('#0891b2','#0d9488','#84cc16')
            for rank,(lineid,color) in enumerate(zip(top,colors)):
                line=lines[lineid];n=line[:2]/max(np.linalg.norm(line[:2]),1e-8);c=line[2]/max(np.linalg.norm(line[:2]),1e-8)
                foot=center-(center@n+c)*n;tangent=np.array([-n[1],n[0]]);ends=foot+np.array([-3,3])[:,None]*span*tangent
                ax.plot(*ends.T,c=color,lw=1+3*weights[lineid],alpha=.7,label=f'선 {lineid}: 가중치 {weights[lineid]:.3f}')
            ax.scatter(*gt[i,j],marker='+',s=130,c='#22c55e',label=f'점 {j} GT')
            for k,(name,color) in enumerate(zip(('유지','복원 절반','복원 전체','선 절반','선 전체'),('#2563eb','#fb923c','#ef4444','#a3a3a3','#111827'))):
                ax.scatter(*p['candidates'][i,j,k],s=65,facecolors='none',edgecolors=color,label=name)
            ax.scatter(*p['points'][i,j],marker='*',s=170,c='#a855f7',label='최종 선택')
            pts=np.concatenate((gt[i,valid[i]],p['candidates'][i,valid[i]].reshape(-1,2),box.reshape(2,2)))
            lo=pts.min(0)-.1*span;hi=pts.max(0)+.1*span;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
            ax.set_title(f'입력 ROI 확대 · 점 {j}의 상위 선과 후보\n선 미선택(null) 가중치 {weights[-1]:.3f}')
            x=np.arange(5);truth=ec[i,j,0]-ec[i,j]
            axes[2].bar(x-.2,p['gain'][i,j],.4,label='모델 예상',color='#a855f7');axes[2].bar(x+.2,truth,.4,label='GT 사후 계산',color='#14b8a6')
            axes[2].axhline(0,c='#475569',lw=1);axes[2].set_xticks(x,['유지','복원0.5','복원1','선0.5','선1']);axes[2].set_ylabel('기준 대비 오차 감소(px), 양수가 좋음');axes[2].legend(fontsize=8)
            axes[2].set_title(f'예상 개선량과 실제 개선량 · 선택 {int(p["choice"][i,j])}')
            fig.suptitle(f'{LABELS[condition]} · {why} · {r["id"]}\n최종 융합 - 점 대조군: {diff[i]:+.3f}px (프레임 평균) · seed1',fontsize=12)
            fig.tight_layout(rect=(0,.03,1,.89));name=f'{condition}_{len(gallery):02d}.png';fig.savefig(directory/name,dpi=125,bbox_inches='tight',pad_inches=.15);plt.close(fig)
            gallery.append(dict(condition=condition,id=r['id'],selection=why,point=j,seed=1,path='figures/'+name,frame_difference_px=float(diff[i]),
                top_line_ids=top.tolist(),top_line_weights=weights[top].tolist(),null_weight=float(weights[-1]),
                scope='GT selects examples and point only for posthoc illustration. Line weights are actual parent attention, not causal attribution/calibrated correctness. Lines are predicted, not GT. No amplified coordinate displacement.'))
    write(RUN/'GALLERY.json',dict(rows=gallery));return gallery

def diagram():
    # Repo-native SVG; no external assets or network dependencies.
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="460" viewBox="0 0 1120 460" role="img" aria-label="고정 YOLO에서 점 보정, 복원, DHT 선 후보를 만든 뒤 학습한 평가기로 선택하는 구조"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#64748b"/></marker></defs><rect width="1120" height="460" fill="#fff"/>
<g font-family="sans-serif" text-anchor="middle" font-size="18" fill="#172334"><rect x="340" y="15" width="440" height="60" rx="12" fill="#e2e8f0"/><text x="560" y="42">고정 YOLO</text><text x="560" y="64" font-size="15">검출 상자 · 8개 점 · 영상 특징</text>
<rect x="30" y="120" width="320" height="80" rx="12" fill="#dbeafe"/><text x="190" y="150">점 주변 보정 + 보정 억제</text><text x="190" y="179" font-size="15">기준점 p0 · 고정 V10 / V12</text>
<rect x="400" y="120" width="320" height="80" rx="12" fill="#ffedd5"/><text x="560" y="150">전체 ROI · 점 관계로 복원</text><text x="560" y="179" font-size="15">복원점 pa · 고정 V12</text>
<rect x="770" y="120" width="320" height="80" rx="12" fill="#dcfce7"/><text x="930" y="150">DHT 선 · 점별 연관 + null</text><text x="930" y="179" font-size="15">기준점 주변 추가 보정 dl · 고정 V11</text>
<rect x="230" y="260" width="660" height="60" rx="12" fill="#eef2ff"/><text x="560" y="284">유지 / 복원 절반·전체 / 선 보정 절반·전체</text><text x="560" y="308" font-size="15">p0, (p0+pa)/2, pa, p0+dl/2, p0+dl</text>
<rect x="270" y="360" width="580" height="80" rx="12" fill="#f3e8ff"/><text x="560" y="389">학습한 후보 평가 → 최대 예상 개선량 후보 선택</text><text x="560" y="416" font-size="15">기준 점수0 · 동점이면 유지 · 최종 점 출력</text></g>
<g stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M450 75L190 120"/><path d="M560 75V120"/><path d="M670 75L930 120"/><path d="M190 200L400 260"/><path d="M560 200V260"/><path d="M930 200L720 260"/><path d="M560 320V360"/><path d="M350 158H400"/><path d="M190 200V225H930V200"/></g><text x="920" y="242" font-family="sans-serif" font-size="13" fill="#64748b">선 보정도 기준점 주변에서 계산</text></svg>'''
    (RUN/'architecture.svg').write_text(svg)

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    s=read(RUN/'SUMMARY.json');audit=read(RUN/'AUDIT.json');integration=read(RUN/'INTEGRATION_CHECK.json');d=read(RUN/'FAILURE_DIAGNOSIS.json')
    assert audit['status']==integration['status']=='PASS';g=figures();diagram()
    architecture=(ROOT/'_docs/notes/pallet_final_fusion_v14_architecture.md').read_text();(RUN/'ARCHITECTURE_KO.md').write_text(architecture)
    lines=['# 최종 점·선 아키텍처 구현 및 기존 방법 비교','',
        '구현과 평가를 완료했다. 최종 후보 V14는 원래 YOLO보다 원본·가림·심한 잘림에서 좋아졌지만, 기존 복원 방법을 안정적으로 넘지 못했고 약한 잘림 악화도 해결하지 못했다. 따라서 구현은 완료했지만 검증을 통과한 최종 배포 모델로 채택하지 않는다.', '',
        '## 최종 후보 구조','',
        '고정 YOLO → 점 주변 보정·보정 억제 / 전체 ROI 기반 복원 / DHT 선 보정 → 유지·복원 절반·복원 전체·선 절반·선 전체의5후보 → 학습한 예상 개선량 평가 → 최종 점. YOLO 내부는 그대로이며 이후 모델을 통합했으므로 전체 예측 아키텍처를 바꾼 것이다.', '',
        'DHT는 영상과 예측 점 주변 prior로 선을 예측한다. 점을 연결해서 선을 만들거나 점을 무조건 선 위에 올려놓지 않는다. 후보별 주변 영상과 기준 후보와의 차이, 전체 ROI 문맥, 점 ID·후보 ID, 좌표와 선 연관 요약을 보고 선택한다. 유지 후보의 예상 개선량은0이며 동점이면 유지한다. GT 필터나 self-training은 아니다.', '',
        '새 평가기는46425파라미터다. YOLO 이후 전체 통합 구조는473804파라미터이며 부모 보정기·복원기·DHT는 고정했다. 이번에는 YOLO부터 전체를 공동 학습하지 않았다.', '',
        '## 학습과 평가 데이터 분리','',
        '보정·복원기의 학습 이미지와 원본 ID·이미지 SHA가 겹치지 않는 기존 합성 calibration253장을 평가기 학습에 사용했다. 따라서 이253장은 이제 학습 데이터이며 평가에 사용하지 않는다. 같은 수의 기존 학습 이미지253장도 별도 대조군으로 새로 추론했다. 이는 보정·복원기 학습에서의 분리이며 YOLO 사전학습 미노출이나 새로운 장면 독립성을 보장하지 않는다.', '',
        '기존 학습 이미지 융합 / 분리 학습 점 대조군 / 분리 학습 최종 점·선 융합의3구조를 각3개 시드,2000업데이트씩 학습했다. 총9모델·18000업데이트. 검출 성공 관측은 각각1464개,1462개다. 두 분리 학습 대조군은 초기화·batch 순서가 정확히 같다. 모든 부모 체크포인트를 유지하고 실제 이미지로 학습하거나 모델을 고르지 않았다.', '',
        '평가는 기존 합성505장과 실제297장·9세션·직접 클릭한1251점의6조건이다. 모든 방법에 같은 검출과 원래 점 ID를 사용했다. 검출 실패를 포함하는 정규화 지표도 보고한다. 이전 방법들 사이의 과거 학습량이 동일한 것은 아니다.', '',
        '## 주요 조건 비교','',
        '3개 시드 평균 오차(px). 원본은 검출 프레임별 평균이고, 가림·잘림 행은 지정 대상 점의 평균이다. 작을수록 좋으며 행 간 서로 다른 대상의 수치를 직접 비교하지 않는다.', '',
        '| 조건 / 대상 | YOLO | V10 점 보정 | V11 DHT | V12 억제 | V12 복원 | V13 평가 | V14 점 | V14 최종 점·선 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    compact=[];arms=('stock','v10_local','v11_joint','v12_gate','v12_amodal','v13_scorer','heldout_points','heldout_fusion')
    for c,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        r=s['results']['real/'+c];name=LABELS[c]+(' / 프레임 평균' if typ is None else ' / 가려진280점' if typ=='hidden' else ' / 화면 밖'+str(r['stock']['strata'][typ]['detected_points'])+'점')
        values=[mean(r,a,typ) for a in arms];lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |');compact.append((name,values))
    lines+=['',
        'V14 최종 후보는 원본 오차 유지 기준을 세 시드 모두 통과했고 코너 가림의 기존 복원 이득80%도 유지했다. 약한 잘림은 세 시드 모두 기준보다1% 넘게 나빴고, 심한 잘림은 seed3에서 기존 복원 이득80% 유지 기준을 통과하지 못했다. 전체 채택 기준은 실패다.', '',
        '## 선과 분리 학습의 추가 효과','',
        '같은 분리 학습 자료를 쓴 점 대조군 대비 최종 융합의 원본 평균은0.025px 좋아졌다. 9세션95% 구간은 -0.030~-0.018px이지만 한 시드는 악화했고 시드별1% 개선 기준도 통과하지 못했다. 가림·잘림 대상 점에서는 세 시드의 방향이 섞이며 선의 추가 이득 기준을 통과한 주요 조건이 없다.', '',
        '같은 최종 모델에서 선을 끄면 코너 가림 대상 점은 오히려0.026px 좋아졌다. 정상 선이 다른 영상의 선보다0.017px 좋기는 했지만, 선을 안 쓰는 것보다 좋은 결과는 아니었다. 정상 선과 잘못된 선의 차이 하나만으로 최종 성능 향상을 주장하지 않는다.', '',
        '분리 학습 최종 융합은 기존 학습 이미지 융합보다 약한 잘림에서 평균0.205px 좋아졌지만, 세션95% 구간 -0.595~+0.188px에0이 포함되고 한 시드는 악화했다. 원본과 심한 잘림 평균은 나빠졌다. 학습 이미지의 분리만으로 V13 문제가 해결된다는 가설은 이번 결과가 지지하지 않는다. 두 그룹의 원본 영상 분포도 다르므로 분리 여부만의 순수 인과 효과로 단정하지 않는다.', '',
        '## 무엇이 아직 막고 있나','',
        '최종 후보의 실제 약한 잘림에서 전체 복원의 예상 개선량은 시드별 +0.853~+1.123px였지만 실제 개선량은 -4.554~-3.613px였다. 이미지 분리 후에도 도움이 되는 보정과 해로운 보정을 구별하는 오차가 남았다.', '',
        '이번 선 보정 후보는 어려운 점을 크게 복원하기보다 작은 위치 조정을 제안한다. 선 후보를 추가한 GT 특권 상한의 이득도 작다. 다음 표는 정답을 알고 후보를 고르는 사후 진단이며 실제 달성 성능을 뜻하지 않는다.', '',
        '| 실제 대상 점 | 선 보정 길이 평균(px), 시드 평균 | 점 후보 GT 최선 오차 | 선까지 포함한 GT 최선 오차 | 선 후보 추가 이득 |',
        '|---|---:|---:|---:|---:|']
    for c,typ in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        vals=[r['strata'][typ] for r in d['rows']['real/'+c]]
        fields=('line_delta_mean_px','points_candidate_oracle_px','fusion_candidate_oracle_px','line_candidate_oracle_additional_gain_px')
        means=[float(np.mean([r[f] for r in vals])) for f in fields]
        lines.append('| '+LABELS[c]+' | '+' | '.join(f'{v:.4f}' for v in means)+' |')
    times=[r['post_yolo_batch1_mean_ms'] for r in integration['rows'] if 'post_yolo_batch1_mean_ms' in r]
    lines+=['',
        '따라서 현재 후보 집합에서 선을 더 잘 선택하는 것만으로 큰 추가 개선을 얻기는 어렵다. 이 결론은 이번 고정 DHT와 선 보정 후보에 한정한다. 선 자체의 원리적 불가능성이나 다른 선 기반 복원 아키텍처의 실패를 입증한 것은 아니다.', '',
        '## 통합 구현과 검증','',
        'FinalPointLineNetwork는 YOLO 관측을 받아 부모 점·복원·DHT 경로와 평가기를 모두 실행한다. 각 시드·4조건·8프레임에서 저장된 평가 예측과 비교했고 최대 차이는0.00003052px였다. 실제 최대 점수 후보 선택도 일치했다. 통합 체크포인트3개를 export/final_seed1.pt~final_seed3.pt로 저장했다. 실제 성적에 따라 그중 하나를 고르지 않았다.', '',
        f'YOLO 이후 전체 구조의 batch1 추가 추론시간은 약{min(times):.2f}~{max(times):.2f}ms였다.10회 예열 후30회 GPU 동기화 측정이며 YOLO, 이미지 로딩, 전송 시간을 제외한다. 전체 시스템 FPS나 다른 방법과 동일 조건의 속도 우위는 주장하지 않는다.', '',
        f'독립 검산 PASS: 예측 프레임 행{audit["prediction_frame_rows"]:,}개와 선택 프레임 행{audit["selection_frame_rows"]:,}개. 후보 재구성 최대 차이{audit["candidate_replay_max_px"]:.8f}px, 지표 재계산 최대 차이{audit["metric_replay_max"]:.3g}. 모든18000학습 업데이트의 유한 손실·gradient, 부모 체크포인트와 학습 소스 해시, 실제 수동 GT 해시·좌표 이동, 합성 GT 이동을 확인했다.', '',
        '실제297장 전체의 자연 가림 라벨은 unknown이며 인공 스트레스 평가다. 수동 GT를 새로 찍거나 클릭 오차의 크기를 추정한 실험은 아니다. 기존 reflect100 padding, 검출 실패, original-ID 평가를 유지했다. 재사용 DEV 결과이므로 새 현장·자연 가림·최종6D 일반화를 입증하지 않는다. 예전41.49px와도 다른 모집단·지표다.', '',
        '## 최종 판단','',
        '요청한 점·선 구조를 하나의 모델로 구현하고 기존 방법들과 비교했다. 현재 채택할 수 있는 안정적 개선 구조로는 확인되지 않았으므로 기존 배포 모델을 교체하지 않는다. 큰 가림·잘림에서 확인된 이득은 주로 기존 V12 복원 경로에서 오며, 이번 선 후보의 추가 개선은 실용적 기준 미달이다.', '',
        '이후 연구를 진행한다면 후보 선택기만 반복 수정하기보다 선 후보가 화면 밖·가려진 점에 충분한 새로운 정보를 주도록 후보 생성부터 바꿔야 한다. 이번 결과를 근거로 아직 검증하지 않은 구조의 성능을 보장하지 않는다.', '',
        '시각화는 저장된 DHT 선·점별 연관 가중치와5후보·예상 개선량을 보여준다. 상위 선 가중치는 인과적 설명이나 보정된 정확도 확률이 아니다. 최대 개선·악화 예시와 대표 점은 설명용으로 GT를 사후 사용해 골랐다. 정상 추론에는 GT를 넣지 않았다.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    write(RUN/'CONCLUSION.json',dict(status='IMPLEMENTED_COMPARED_NOT_ACCEPTED',robustness_pass=s['robustness_pass'],line_gates=s['line_gates'],
        deployment_changed=False,real_tuning=False,implemented_architecture='FinalPointLineNetwork',final_candidate='heldout_fusion',all_seeds_retained=True,
        reason='Mild crop regresses in all seeds and severe crop retention fails seed3. No major stratum passes incremental line criterion.'))
    body=[]
    for line in lines[1:]:
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line and not line.startswith('|'):body.append('<p>'+html.escape(line)+'</p>')
    table='<table><tr><th>조건 / 대상</th>'+''.join('<th>'+NAMES[a]+'</th>' for a in arms)+'</tr>'+''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>' for name,values in compact)+'</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>최종 점·선 아키텍처 — 기존 방법 비교</title>
<style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}h1{font-size:29px}h2{margin-top:26px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#0369a1}.note{color:#596579}.verdict{border-left:5px solid #b91c1c}.final{background:#f3e8ff;font-weight:600}.scroll{overflow-x:auto}</style>
<h1>최종 점·선 구조 구현 완료 · 안정적 추가 개선 기준은 미달</h1><p>3구조 × 3시드 · 총18000업데이트 · 합성 학습만 사용 · 기존 방법과 동일 평가</p>
<section class="verdict"><b>구현·검산 완료, 배포 모델 교체하지 않음</b><p>원래 YOLO 대비 개선은 있으나 기존 복원을 안정적으로 넘지 못했습니다. 약한 잘림 악화가 남고, 선의 추가 이득은 주요 조건의 사전 기준을 통과하지 못했습니다.</p><div class="scroll">'''+table+'''</div><p class="note">단위px, 작을수록 좋음. 원본은 프레임 평균, 다른 행은 지정 대상 점 평균입니다. 3개 시드 평균이며 실제 이미지는 직접 클릭한 GT만 평가합니다.</p></section>
<section><h2>최종 아키텍처</h2><a href="architecture.svg" target="_blank"><img src="architecture.svg" alt="최종 후보 구조도"></a><p>YOLO 내부와 기존 보정·복원·DHT는 고정하고, 46425파라미터 평가기를 학습했습니다. 모듈 추가로 전체 예측 구조를 바꿨으며 GT 필터·self-training은 아닙니다.</p><a href="ARCHITECTURE_KO.md">상세 설계와 학습·평가 계약</a></section>
<section><h2>기존 방법 전체 비교</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3개 시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균 오차px</option><option value="frame_p90_px">프레임 오차P90 px</option><option value="hidden">가려진 점 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="pck5">전체 유효 점PCK5 (높을수록 좋음)</option><option value="pck10">전체 유효 점PCK10 (높을수록 좋음)</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p><p class="note">같은 검출·원래 점 ID·GT를 공유합니다. 정규화 지표는 원본 대각선 기준, 검출 실패1입니다. 기존 방법들의 과거 학습량까지 동일한 대조군은 아닙니다.</p></section>
<section><h2>실제로 참고한 선과 후보 선택</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">회색은 잘려 입력에서 보이지 않는 영역입니다. 가운데는 ROI 확대이며 좌표 이동을 과장하지 않았습니다. 선은 모델 예측, 가중치는 실제 선 연관 값입니다. GT는 사후 설명·진단에만 사용했습니다.</p></section>
<section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서와 전체 표</a> · <a href="SUMMARY.json">모든 수치·시드·구간</a> · <a href="AUDIT.json">독립 검산</a> · <a href="FAILURE_DIAGNOSIS.json">후보 정보량 진단</a> · <a href="INTEGRATION_CHECK.json">통합 추론·시간</a> · <a href="PROTOCOL.json">사전 기준</a></p><p>통합 후보 체크포인트: <a href="export/final_seed1.pt">seed1</a> · <a href="export/final_seed2.pt">seed2</a> · <a href="export/final_seed3.pt">seed3</a>. 실제 성적에 따라 시드를 고르지 않았으며 실험용입니다.</p></section><script>
const data='''+json.dumps(s['results'],ensure_ascii=False)+''',labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''',gallery='''+json.dumps(g,ensure_ascii=False)+''';
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>방법</th>'+Object.values(labels).map(n=>'<th>'+n+'</th>').join('')+'</tr>';for(const a of Object.keys(names)){out+='<tr'+(a==='heldout_fusion'?' class="final"':'')+'><td>'+names[a]+'</td>';for(const c of Object.keys(labels)){const rows=data[split+'/'+c],r=a==='stock'?[rows.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>rows[a+'_seed'+s]);const v=r.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:(metric.startsWith('pck')?x.strata.all[metric]:x[metric]));out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;document.getElementById('counts').textContent='검출: '+Object.keys(labels).map(c=>labels[c]+' '+data[split+'/'+c].stock.detected_frames+'/'+data[split+'/'+c].stock.frames).join(' · ');}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · seed1 · 점 대조군 대비 '+r.frame_difference_px.toFixed(3)+'px (음수 개선)';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page)
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),html_sha256=sha(RUN/'index.html'),source_sha256=sha(Path(__file__)),figures=len(g)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
