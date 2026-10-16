"""Posthoc diagnosis and Korean delivery. Never selects or retrains checkpoints."""
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context13 import *

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 주변 가림'}
NAMES={'stock':'기존 YOLO','base':'보정 억제 기준','full':'복원 전체 적용','constant':'공통 점수','geometry':'좌표 후보 평가','image':'영상 후보 평가'}

def avg(rows,arm,typ=None):
    cells=[rows['stock']] if arm=='stock' else [rows[f'{arm}_seed{s}'] for s in SEEDS]
    return float(np.mean([r['strata'][typ]['mean_px'] if typ else r['frame_mean_px'] for r in cells]))

def diagnostic():
    result={};counts={a:np.zeros(3,dtype=int) for a in ARMS}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            targets=read(V11/'targets'/f'{split}_{condition}.json')['rows']
            gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets]);kind=P.target_type(targets).numpy()
            rows={}
            for seed in SEEDS:
                for arm in ARMS:
                    p=np.load(RUN/'predictions'/split/condition/f'{arm}_seed{seed}.npz')
                    error=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1)
                    gain=error[...,:1]-error;chosen=p['choice'];actual=np.linalg.norm(p['points']-gt,axis=-1)
                    eligible=valid&p['detected'][:,None];counts[arm]+=np.array([((chosen==k)&eligible).sum() for k in range(3)])
                    d={}
                    for typ,k in [('all',None),('intact',0),('hidden',1),('outside',2)]:
                        mask=eligible if k is None else eligible&(kind==k)
                        if not mask.any():continue
                        d[typ]=dict(points=int(mask.sum()),true_full_gain_mean_px=float(gain[...,2][mask].mean()),
                            predicted_full_gain_mean_px=float(p['gain'][...,2][mask].mean()),
                            full_beneficial_fraction=float((gain[...,2][mask]>0).mean()),
                            full_predicted_positive_fraction=float((p['gain'][...,2][mask]>0).mean()),
                            kept_despite_helpful_candidate=int((mask&(chosen==0)&(gain.max(-1)>0)).sum()),
                            helpful_candidate_points=int((mask&(gain.max(-1)>0)).sum()),
                            new_catastrophes_base5_to_selected20=int((mask&(error[...,0]<=5)&(actual>20)).sum()),
                            new_catastrophes_full5_to_selected20=int((mask&(error[...,2]<=5)&(actual>20)).sum()))
                    rows[f'{arm}_seed{seed}']=d
            result[f'{split}/{condition}']=rows
    # Parent error targets seen by training: in-sample, not an independent evaluation.
    targets=read(P.V10/'targets_train.json')['rows'];train_rows=list(targets);conditions=['clean']*len(targets)
    for condition in CONDITIONS[1:]:
        m=read(V12/'cache/train_corner20/manifest.json') if condition=='corner20' else read(V11/'cache/train'/condition/'manifest.json')
        t=read(V12/'targets_train_corner20.json')['rows'] if condition=='corner20' else read(V11/'targets'/f'train_{condition}.json')['rows']
        part=[r for r,c in zip(t,m['rows']) if c['detected']];train_rows.extend(part);conditions.extend([condition]*len(part))
    gt=np.array([r['points'] for r in train_rows]);v=np.array([r['valid'] for r in train_rows]);kind=P.target_type(train_rows).numpy();conditions=np.array(conditions)
    train={}
    for seed in SEEDS:
        rep=parent_cache('train','mixed',seed);c=candidates(rep).numpy();e=np.linalg.norm(c-gt[:,:,None],axis=-1);gain=e[...,0]-e[...,2];rows={}
        for condition in CONDITIONS:
            for typ,k in [('all',None),('hidden',1),('outside',2)]:
                mask=v&(conditions==condition)[:,None]
                if k is not None:mask=mask&(kind==k)
                if mask.any():rows[condition+'/'+typ]=dict(points=int(mask.sum()),true_full_gain_mean_px=float(gain[mask].mean()),full_beneficial_fraction=float((gain[mask]>0).mean()))
        train[str(seed)]=rows
    d=dict(posthoc=True,evaluation=result,training_parent_targets=train,
        valid_evaluation_choice_counts={a:c.tolist() for a,c in counts.items()},
        scope='Frozen saved predictions only. No retraining, thresholds, checkpoint selection or inference GT. Training parent errors are in-sample. Differences support a transfer hypothesis, not causal identification.')
    write(RUN/'FAILURE_DIAGNOSIS.json',d);return d

def gallery():
    directory=RUN/'figures';directory.mkdir(exist_ok=True);gallery=[]
    for condition in ('clean','corner20','crop20','crop40'):
        m=read(V11/'cache/real'/condition/'manifest.json');t=read(V11/'targets'/f'real_{condition}.json')['rows']
        p=np.load(RUN/'predictions/real'/condition/'image_seed1.npz');gt=np.array([r['points'] for r in t]);v=np.array([r['valid'] for r in t]);kind=P.target_type(t).numpy()
        error=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1);actual=np.linalg.norm(p['points']-gt,axis=-1)
        subgroup=v if condition=='clean' else v&(kind==(1 if condition=='corner20' else 2))
        eligible=np.flatnonzero(p['detected']&subgroup.any(-1));diff=np.where(v,actual-error[...,2],0).sum(-1)/v.sum(-1)
        examples=[(int(sorted(eligible,key=lambda i:digest('v13-gallery:'+m['rows'][i]['id']))[0]),'SHA 고정'),
                  (int(eligible[np.argmin(diff[eligible])]),'전체 적용 대비 최대 개선'),(int(eligible[np.argmax(diff[eligible])]),'전체 적용 대비 최대 악화')]
        for i,why in examples:
            r=m['rows'][i];im=cv2.imread(str(ROOT/r['image']));assert im is not None;canvas=im.copy()
            if r['mask'] is not None:
                a,u,b,w=r['mask'];canvas[u:w,a:b]=r['fill_bgr']
            if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
            possible=np.flatnonzero(subgroup[i]);j=int(possible[np.argmax(np.abs(actual[i,possible]-error[i,possible,2]))])
            fig,axes=plt.subplots(1,3,figsize=(18,6));origin=np.array(r['origin']);original=gt[i]+origin
            axes[0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));axes[0].scatter(*original[v[i]].T,marker='+',c='#22c55e',s=80)
            axes[0].set_title('원본과 수동 GT (초록 +)')
            for n in np.flatnonzero(v[i]):axes[0].annotate(str(n),original[n],color='#22c55e')
            ax=axes[1];ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946');ax.set_title(f'변형 입력 · 점 {j} 후보와 선택')
            ax.scatter(*gt[i,v[i]].T,marker='+',c='#22c55e',s=80,label='수동 GT')
            for k,(name,color) in enumerate(zip(('유지','절반','전체'),('#2563eb','#d97706','#ef4444'))):
                ax.scatter(*p['candidates'][i,v[i],k].T,s=45,facecolors='none',edgecolors=color,label=name)
            ax.scatter(*p['points'][i,j],marker='*',s=150,c='#a855f7',label='선택한 후보');ax.legend(fontsize=8)
            pts=np.concatenate([gt[i,v[i]],p['candidates'][i,v[i]].reshape(-1,2)])
            lo=np.minimum(pts.min(0)-15,[0,0]);hi=np.maximum(pts.max(0)+15,canvas.shape[:2][::-1]);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
            x=np.arange(3);truth=error[i,j,0]-error[i,j]
            axes[2].bar(x-.19,p['gain'][i,j],.38,label='모델이 예상한 개선량',color='#a855f7')
            axes[2].bar(x+.19,truth,.38,label='GT로 사후 계산한 개선량',color='#14b8a6')
            axes[2].axhline(0,color='#475569',lw=1);axes[2].set_xticks(x,['유지','절반','전체']);axes[2].set_ylabel('기준 대비 오차 감소(px), 양수가 좋음')
            axes[2].set_title(f'점 {j}의 후보 평가 · 선택: {int(p["choice"][i,j])}');axes[2].legend(fontsize=8)
            fig.suptitle(f'{LABELS[condition]} · {why} · {r["id"]}\n영상 후보 평가 - 전체 적용: {diff[i]:+.3f}px (프레임 평균) · seed1',fontsize=12)
            fig.tight_layout(rect=(0,.04,1,.88));name=f'{condition}_{len(gallery):02d}.png';fig.savefig(directory/name,dpi=125,bbox_inches='tight',pad_inches=.15);plt.close(fig)
            gallery.append(dict(condition=condition,id=r['id'],selection=why,path='figures/'+name,point=j,seed=1,frame_difference_px=float(diff[i]),
                scope='Example and point selection use GT only for posthoc illustration. Coordinates are unamplified. This is a decision visualization, not an attention/saliency map.'))
    write(RUN/'GALLERY.json',dict(rows=gallery));return gallery

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    s=read(RUN/'SUMMARY.json');a=read(RUN/'AUDIT.json');assert a['status']=='PASS';d=diagnostic();g=gallery()
    lines=['# 후보 개선량 평가 모듈: 이번 구조는 합격하지 못함','',
        '모듈을 추가해 전체 예측 아키텍처를 바꿨다. YOLO 본체와 기존 점 보정·복원기는 고정하고, 유지·절반·전체 복원 후보 중 예상 오차 감소가 가장 큰 후보를 고르는 학습 모듈을 추가했다. 후보 생성과 선택을 모델 안에서 수행하며, GT로 거르는 추론 필터나 self-training은 아니다.', '',
        '다만 이번에는 후보 평가만 학습했다. YOLO 내부 구조를 변경하거나 전체를 공동 학습하지 않았다. 1단계 사전 기준을 통과하지 못했으므로 약속한 조건에 따라 공동 학습으로 확장하지 않았다.', '',
        '## 실제 평가 결과','',
        '3개 시드 평균 오차(px), 작을수록 좋음. 원본은 프레임별 평균이고 다른 행은 명시한 대상 점의 평균이다. 행 간 오차를 직접 비교하지 않는다. 모든 모델은 동일한 검출·대상 점을 사용한다.','',
        '| 조건 / 대상 | 기준: 보정 억제 | 복원 전체 적용 | 새 영상 후보 평가 |','|---|---:|---:|---:|']
    table=[]
    for c,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        r=s['results']['real/'+c];name=LABELS[c]+(' / 프레임 평균' if typ is None else ' / 가려진280점' if typ=='hidden' else ' / 화면 밖'+str(r['stock']['strata'][typ]['detected_points'])+'점')
        vals=[avg(r,arm,typ) for arm in ('base','full','image')];lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |');table.append((name,vals))
    lines+=['',
        '세 시드 모두 원본 오차 유지와 가림·심한 잘림 개선량80% 유지 기준은 통과했다. 그러나 약한 잘림에서 기준 성능을 유지하지 못했고, 전체 적용 및 좌표 대조군보다도 나빠져 최종 판정은 실패다. 이번 모듈의 추가적인 성능 향상은 확인되지 않았다.', '',
        '약한 잘림의 화면 밖 점에서 새 모듈은 전체 적용보다 평균0.252px 더 나빴다. 9세션을 묶어 계산한 대응95% 구간은 +0.070~+0.410px였다. 심한 잘림도 전체 적용보다 +0.301px, 구간 +0.122~+0.594px였다. 코너 가림은 +0.083px, 구간 -0.045~+0.221px로 추가 효과가 뚜렷하지 않았다.', '',
        '## 무엇을 학습했나','',
        '합성 관측3609개로 공통 점수 / 좌표 후보 평가 / 영상 후보 평가를 각3개 시드, 2000업데이트씩 학습했다. 총9모델·18000업데이트이며 실제 학습은0장이다. 모델마다 같은 초기화와 학습 batch 순서를 맞췄다. 영상 평가 모듈은 총47300파라미터이고 공통 점수 대조군의 활성 파라미터는3개여서 용량이 같은 대조군은 아니다.', '',
        '기준 후보 p0는 V12 learned_gate, 전체 후보 p1은 V12 gated_amodal 출력이다. p0, (p0+p1)/2, p1 중 하나를 고른다. 영상 모델은 후보 주변3×3 영상 특징, 기존 복원기의 점 문맥, 좌표·이동량·분기 가중치를 받는다. 좌표 대조군도 상위 모델이 영상으로 예측한 좌표와 가중치를 사용하므로 영상과 완전히 독립적이지 않다.', '',
        '합성 GT로 error(p0)-error(candidate)를 학습했다. SmoothL1 개선량 회귀와 후보 선호 손실을 사용하며, 기준 후보 점수는 정확히0이다. 추론은 최대 예상 개선량 후보를 고르고 동점이면 유지한다. 실제 GT, 조건 이름, 가림 마스크를 모델 입력에 주지 않았다.', '',
        '## 실패를 확인한 근거','',
        '약한 잘림에서 실제로 보정을 적용한 점 중66.3~78.0%가 악화했다. 이 점들은 모델이 개선량을 양수로 예상해 선택한 경우다. 최종 좌표 조합이나 argmax 구현 오류가 아니라 예상 개선량이 실제와 맞지 않는 문제가 남았다.', '',
        '약한 잘림의 동일 후보에서 GT로 가장 좋은 후보를 고르는 특권 진단 오차는 평균11.566px였다. 기준12.372px보다 낮으므로 후보 집합에 활용할 여지는 있지만, 이는 정답을 아는 선택의 결과이며 실제로 달성 가능한 성능이나 선 정보의 효과를 증명하지 않는다.', '',
        '세 후보는 같은 이동 방향의 크기만 바꾼다. 원래 복원 방향이 틀리면 유지로 피할 수는 있어도 올바른 새 방향을 만들어내지는 못한다. 그리고 공통 점수 대조군은 평가 유효 점91200개에서 모두 전체 적용을 골랐다. 좌표 대조군도91199개에서 전체를, 1개에서 절반을 골라 사실상 입력에 따른 선택을 배우지 못했다. 영상 모델도 유익함과 해로움을 충분히 구별하지 못했다.', '',
        '합성 검증의 약한 잘림에서는 전체 복원이 기준보다 평균적으로 좋아졌지만, 실제의 동일 조건에서는 악화했다. 이 차이는 합성에서 배운 보정 이득이 실제에 옮겨지지 않는다는 가설을 뒷받침한다. 구도·세션·초기 점 오차·GT 차이가 섞여 있으므로 도메인 차이 하나가 유일한 원인이라고 단정하지 않는다.','',
        '| 분할: 약한 잘림의 화면 밖 점 | 전체 복원의 실제 평균 개선량(px), seed1/2/3 |','|---|---|']
    train=[d['training_parent_targets'][str(seed)]['crop20/outside']['true_full_gain_mean_px'] for seed in SEEDS]
    lines.append('| 합성 학습: 부모 모델이 이미 본 데이터 | '+' / '.join(f'{v:+.3f}' for v in train)+' |')
    for split,name in [('synth_val','합성 검증'),('real','실제 재사용 평가')]:
        vals=[d['evaluation'][split+'/crop20'][f'image_seed{seed}']['outside']['true_full_gain_mean_px'] for seed in SEEDS]
        lines.append('| '+name+' | '+' / '.join(f'{v:+.3f}' for v in vals)+' |')
    lines+=['',
        '실제 약한 잘림의 화면 밖 점에서 모델은 전체 적용의 평균 개선량을 +0.651~+1.087px로 예상했지만 실제 개선량은 -4.554~-3.613px였다. 합성 학습에서 부모 복원의 이득은 +2.810~+3.140px였고, 합성 검증에서는 +0.374~+0.578px로 줄었다. 학습 데이터에서 벗어나며 이득이 줄고 실제에서는 부호까지 바뀌는 현상은 확인했다.', '',
        '부모 복원기가 이미 학습한 데이터의 오차로 평가 모듈을 학습했다는 한계도 있다. 쉬워진 학습 오차가 새로운 이미지의 실패를 대표하지 못했을 가능성은 있으나, 이를 단독 원인으로 확인한 교차 학습 비교는 하지 않았다.', '',
        '## 다음 설계에 주는 결론','',
        '이 결과로 점과 선을 함께 쓸 수 없다고 결론내릴 수는 없다. 이번 V13은 복원 후보 선택의 병목을 분리한 실험으로 Hough 분기를 넣지 않았다. 따라서 Hough의 추가 이득도 입증하지 않았다.', '',
        '추가 실험을 한다면 먼저 부모 모델이 보지 않은 합성 이미지의 예측으로 후보 평가 학습 데이터를 만들어야 한다. 유지가 이기는 실패 사례와 복원이 이기는 사례를 실제 GT를 쓰지 않고 충분히 포함하고, 구도·잘림·초기 오차별 검증을 분리하는 것이 우선이다. 현재 구조를 실제 평가에 맞춰 임계값 조정하는 것으로 해결됐다고 보고하지 않는다.', '',
        '선 분기를 다시 넣을 때는 기존 점을 선에 무조건 끌어당기기보다, 점 주변 영상에서 독립적으로 얻은 선 근거로 다른 방향의 후보를 만들고 유지 후보와 비교하는 설계가 맞다. 선도 틀릴 수 있으므로 새로운 후보가 맞는 경우와 틀린 경우를 평가기가 학습해야 한다. 이는 아직 실행하지 않은 설계 제안이다.', '',
        '## 검증과 범위','',
        f'독립 검산 PASS: 예측 프레임 행{a["prediction_frame_rows"]:,}개, 선택 프레임 행{a["selection_frame_rows"]:,}개. 후보 재구성 최대 차이{a["maxima"]["candidate_replay_px"]:.8f}px, 지표 재계산 최대 차이{a["maxima"]["metric_replay"]:.3g}. 부모 체크포인트와 학습 소스는 변경되지 않았다.', '',
        '실제297장·9세션·직접 클릭한1251점과 합성505장을 재사용했다. 자연 가림의 정답 라벨은 unknown이므로 인공 가림·잘림 결과다. 수동 GT를 재배정하거나 자동 투영 GT로 바꾸지 않았지만, 수동 클릭 오차 자체가 없음을 입증한 실험은 아니다. 이전41.49px 및 최종6D 성능과 직접 비교하지 않는다.', '',
        '기존 검출과 reflect100 padding을 공유한다. 검출 실패 포함 정규화 지표는 HTML에서 따로 볼 수 있다. 새 모듈을 실제 배포 모델로 교체하지 않았다. 심한 악화(기준5px 이내에서 선택 후20px 초과) 건수는 FAILURE_DIAGNOSIS.json에 조건·시드별 기록했다.', '',
        '시각화의 최대 개선·악화 예시와 대표 점은 GT로 사후 선정했다. 그래프의 예상 개선량은 모델 출력이고 실제 개선량은 진단용 GT 계산이다. Attention 또는 인과적 saliency 시각화가 아니다.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    body=[]
    for line in lines[1:]:
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line and not line.startswith('|'):body.append('<p>'+html.escape(line)+'</p>')
    static='<table><tr><th>조건</th><th>보정 억제 기준</th><th>복원 전체 적용</th><th>새 영상 후보 평가</th></tr>'+''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+'</tr>' for name,vals in table)+'</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>후보 개선량 평가 — 실험 결과</title>
<style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1250px;margin:auto;padding:28px}h1{font-size:29px}h2{margin-top:28px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#0369a1}.note{color:#596579}.verdict{border-left:5px solid #b91c1c} .flow{padding:15px;background:#eaf0ff;font-weight:600}</style>
<h1>후보 평가 모듈 추가: 약한 잘림 악화를 막지 못함</h1><p>YOLO 본체 고정 · 후보 평가 모듈만 학습 · 9모델 / 3시드 · 실제 학습0장</p>
<section class="verdict"><b>사전 기준 실패 — 공동 학습 확장하지 않음</b><p>가림·심한 잘림의 이전 개선은 대부분 유지했지만, 새 모듈의 추가 이득은 확인되지 않았습니다.</p>'''+static+'''<p class="note">단위px, 작을수록 좋음. 원본은 프레임 평균, 나머지는 지정된 점의 평균입니다.</p></section>
<section><h2>어디를 바꿨나</h2><div class="flow">고정 YOLO와 보정·복원 → 유지 / 절반 / 전체 후보 → 학습한 개선량 평가 → 최종 점</div><p>모듈 추가로 전체 예측 아키텍처를 변경했습니다. YOLO 내부 구조는 그대로입니다. 이번 병목 분리 실험에는 Hough를 넣지 않았습니다.</p></section>
<section><h2>조건별 수치</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3개 시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임별 평균 오차px</option><option value="hidden">가려진 점 오차px</option><option value="outside">화면 밖 점 오차px</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div style="overflow-x:auto"><table id="table"></table></div><p class="note">동일 검출·동일 대상 점. 검출 실패 포함 지표는 실패1, 원본 대각선으로 정규화합니다.</p></section>
<section><h2>어떤 후보를 왜 골랐나</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">회색은 잘려 모델이 보지 못한 영역입니다. 모델 예상 개선량과 GT 사후 개선량을 비교합니다. GT는 정상 추론에 쓰지 않았습니다. 예시/점 선정은 사후 설명용이며, attention 지도는 아닙니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">전체 보고서와 표</a> · <a href="SUMMARY.json">수치와 기준 판정</a> · <a href="AUDIT.json">독립 검산</a> · <a href="FAILURE_DIAGNOSIS.json">실패 진단</a> · <a href="PROTOCOL.json">고정 프로토콜</a></p></section><script>
const data='''+json.dumps(s['results'],ensure_ascii=False)+''',labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''',gallery='''+json.dumps(g,ensure_ascii=False)+''';
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>조건</th><th>검출</th>'+Object.values(names).map(n=>'<th>'+n+'</th>').join('')+'</tr>';for(const c of Object.keys(labels)){const rows=data[split+'/'+c];out+='<tr><td>'+labels[c]+'</td><td>'+rows.stock.detected_frames+'/'+rows.stock.frames+'</td>';for(const a of Object.keys(names)){const r=a==='stock'?[rows.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>rows[a+'_seed'+s]);const v=r.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:x[metric]);out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · seed1 · 전체 적용 대비 '+r.frame_difference_px.toFixed(3)+'px (음수 개선)';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page)
    write(RUN/'CONCLUSION.json',dict(status='CANDIDATE_SCORER_FAIL',stage1_pass=s['stage1_pass'],joint_training_run=False,
        hough_used=False,real_training=False,production_modified=False,diagnostic='Candidate oracle improves but learned scoring does not transfer; not proof that fusion is impossible.'))
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),gallery_images=len(g)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)
    print('COUNTS',d['valid_evaluation_choice_counts'],flush=True)
    print('TRAIN CROP20 OUTSIDE',train,flush=True)

if __name__=='__main__':main()
