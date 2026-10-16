from context39 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v39-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
from report37 import clipped
from evaluate39 import METHODS
TITLE='V39 큰 점 오류의 보정 방향·크기 식별 검증'
LABELS=dict(anchor='V34 점 유지',standard='기존 V38 중심 예측',balanced='큰 오류 학습 비중 증가',hard_geometry='큰 오류 전용·영상+기하',hard_appearance='큰 오류 전용·영상+법선',point_gate='점 특징으로 큰 오류 선택',line_gate='점+선 특징으로 큰 오류 선택 [주판정]',GT_hard_geometry='정답 큰 오류 선택·전용 기하 [진단]',GT_hard_appearance='정답 큰 오류 선택·전용 영상 [진단]',GT_safe_hard_geometry='정답 큰 오류+해로운 보정 거절 [진단]',candidate_oracle='정답으로 최선 후보 선택 [상한]')
def ci_text(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'

def gallery(p):
    seed=1;a=dict(np.load(V37/'arrays/v34_both_seed1.npz'));v=dict(np.load(RUN/'arrays/both_seed1.npz'));g=p['g'];base=np.linalg.norm(v['anchor']-g['gt'],axis=-1);after=np.linalg.norm(v['points_line_gate']-g['gt'],axis=-1);gain=base-after;check=(~p['train'])[:,None];chosen=v['selected_line_gate']>=0;cases=[]
    selections=[('hard_help',check&v['hard']&chosen&(gain>1),gain,'Median >1px improvement among correctly detected hard check points, seed1.'),('hard_wrong_direction',check&v['hard']&chosen&(gain<-1),-gain,'Median >1px worsening among correctly detected hard check points, seed1.'),('easy_false_positive',check&~v['hard']&chosen&(gain<-1),-gain,'Median >1px worsening among easy check points incorrectly routed to hard expert, seed1.')]
    for name,mask,value,rule in selections:
        if not mask.any():continue
        middle=np.median(value[mask]);i,j=map(int,np.unravel_index(np.where(mask,np.abs(value-middle),np.inf).argmin(),mask.shape));k=int(v['selected_line_gate'][i,j]);row=p['rows'][i];im=actual_image(row);h,w=im.shape[:2];n=a['candidate_normal'][i,j,k];rho=a['candidate_rho'][i,j,k];anchor=v['anchor'][i,j];gt=g['gt'][i,j];point=v['points_line_gate'][i,j];mu=v['mu_hard_geometry'][i,j,k]*p['diag'][i];truth=v['target'][i,j,k]*p['diag'][i];amount=a['candidate_amount'][i,j,k];coords=np.array([anchor,gt,point]);fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained',gridspec_kw=dict(width_ratios=[1.2,1.2,1]))
        for ax,zoom in zip(axes[:2],(False,True)):
            bounds=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e7ebef');seg=clipped(n,rho,bounds)
            if seg is not None:ax.plot(*seg.T,c='#e58c29',ls='--',label='Selected Hough candidate')
            for xy,label,col,m in [(gt,'GT corner','#22a150','x'),(anchor,'V34 anchor','#337fc4','o'),(point,'Routed correction','#dc6262','+')]:ax.scatter(*xy,c=col,marker=m,s=60,label=label,zorder=5)
            ax.annotate('',xy=point,xytext=anchor,arrowprops=dict(arrowstyle='->',color='#dc6262',lw=2));ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[3],bounds[1]);ax.set_aspect('equal');ax.legend(fontsize=7)
        axes[2].scatter([amount,truth,mu],[0,1,2],c=['#e58c29','#22a150','#29a8b0'],s=80);axes[2].axvline(0,c='gray',ls=':');axes[2].set(yticks=[0,1,2],yticklabels=['Applied amount','True normal target','Predicted target'],xlabel='Signed correction (px)',ylim=(-.6,2.6),title='Correct hard detection does not\nguarantee correct correction');axes[2].grid(axis='x',alpha=.2)
        info=dict(name=name,seed=seed,id=row['id'],index=i,point=j,candidate=k,selection=rule,hard=bool(v['hard'][i,j]),hard_probability=float(v['hard_probability_line'][i,j]),base_px=float(base[i,j]),after_px=float(after[i,j]),actual_normal_px=float(truth),predicted_normal_px=float(mu),amount_px=float(amount));fig.suptitle(f'{name} / seed{seed} / {row["id"]} / point{j}\nP(hard)={info["hard_probability"]:.2f}; anchor {base[i,j]:.2f}px -> correction {after[i,j]:.2f}px',fontsize=10);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);cases.append(info)
    return cases

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';p=population();s=read(RUN/'SUMMARY.json');ph=read(RUN/'PHENOTYPES.json');r=s['results']['check/all/all_detected']['methods']['both'];df=s['direction']['fit/all/hard']['both'];dc=s['direction']['check/all/hard']['both'];(RUN/'figures').mkdir(exist_ok=True);table=[dict(method=m,label=LABELS[m],mean_px=r[m]['mean_px'],ci=r[m]['difference']['ci95'],gains=r[m]['gain_percent'],selected=r[m]['selected_points']) for m in METHODS]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');names=['standard','balanced','hard_geometry','hard_appearance'];x=np.arange(4)
    axes[0].bar(x-.18,[df[n]['mean_px'] for n in names],.36,label='Fit groups / in sample');axes[0].bar(x+.18,[dc[n]['mean_px'] for n in names],.36,label='Check groups');axes[0].axhline(np.mean(dc['standard']['zero_target_MAE_px']),c='gray',ls='--',label='Check zero correction');axes[0].set(xticks=x,xticklabels=['Standard','Balanced','Hard + geom','Hard + image'],ylabel='Hard-point normal target MAE (px)',title='Hard-example training fits but does not transfer');axes[0].legend(fontsize=7)
    axes[1].bar(['Fit / in sample','Fit / group OOF','Check'],[np.mean(df['hard_geometry']['sign_accuracy'])*100,np.mean(df['oof_hard_geometry']['sign_accuracy'])*100,np.mean(dc['hard_geometry']['sign_accuracy'])*100],color=['#68a777','#e7b567','#dc726b']);axes[1].set(ylabel='Normal sign accuracy (%), |true target|>=1px',ylim=(0,100),title='OOF fits fewer groups; no claim of chance baseline');fig.savefig(RUN/'figures/direction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(3)
    for mode,col in [('point','#809aad'),('line','#68a777')]:
        hs=[s['hardness'][f'both_seed{k}']['check'][mode] for k in SEEDS];off=-.18 if mode=='point' else .18;axes[0].bar(x+off,[q['auroc'] for q in hs],.36,label=mode,color=col);axes[1].plot([q['recall'] for q in hs],[q['precision'] for q in hs],'o-',label=mode,color=col)
        for k,q in zip(SEEDS,hs):axes[1].annotate(str(k),(q['recall'],q['precision']))
    axes[0].set(xticks=x,xticklabels=['Seed1','Seed2','Seed3'],ylim=(0,1),ylabel='Hardness AUROC',title='Large errors can be ranked from observed features');axes[0].legend();axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Thresholds fixed on calibration groups');axes[1].axvline(.5,c='gray',ls=':');axes[1].axhline(.5,c='gray',ls=':');axes[1].legend();fig.savefig(RUN/'figures/hardness.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');y=[r[m]['mean_px'] for m in METHODS];ax.barh(METHODS,y,color=['#b294cc' if 'GT' in m or 'oracle' in m else '#68a777' for m in METHODS]);ax.axvline(y[0],c='gray',ls='--');ax.invert_yaxis();ax.set(xlabel='Point error (px), same1040checkpoints /3seeds',title='GT routing/oracles are diagnostics, not deployed performance')
    for i,v in enumerate(y):ax.text(v+.12,i,f'{v:.2f}',va='center')
    fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig);cases=gallery(p);hg=[s['hardness'][f'both_seed{k}']['check']['line'] for k in SEEDS];checkph=[ph['results'][str(k)]['check'] for k in SEEDS];tablelines=['| 방법 | 전체1040점 오차px | 시드별 개선율% | 기준 대비 차이95%구간px |','|---|---:|---|---|']
    for q in table:tablelines.append('| '+q['label']+' | '+f'{q["mean_px"]:.3f}'+' | '+', '.join(f'{x:+.2f}' for x in q['gains'])+' | '+ci_text(q['ci'])+' |')
    body=['# '+TITLE,'','**큰 오류를 찾아내는 신호는 확인됐다. 그러나 그 점을 어느 방향으로 얼마나 고칠지는 학습 구간 밖에서 제대로 맞추지 못했다. 어려운 점의 학습 비중만 늘리거나 전용 예측기를 붙이는 것으로는 안정적인 점·선 융합이 되지 않았다.**','',
    '고정 V36 선 후보(grid/both)와 V34 점,3시드에서48개 작은 gradient boosting 모델을 학습했다. 모델당60라운드,총2880라운드이며 신경망 optimizer 업데이트는0이다. 기존 V38의6개 보정 중심 예측기는 그대로 대조군으로 재사용했다. 감사에서3개 모델은 별도로 재학습했다.','',
    '## 무엇을 바꿨나','',
    '큰 오류는 원래 V34 점의 GT 오차가10px 이상인 경우로 고정했다. 새 학습12그룹·기준 선택용4그룹·확인8그룹은 V38과 같다. 큰 오류 여부는 학습 정답이나 평가층을 만드는 데만 사용하고 실제 입력에는 넣지 않았다. 기존 고정 부모와 offset은 과거16그룹을 이미 학습한 이력이 있다.','',
    '1. **학습 비중 대조:** 그룹별 비중을 유지하면서 큰 오류/나머지 오류의 총 학습 비중을 절반씩 맞췄다.\n2. **큰 오류 전용 대조:** 큰 오류 점만 학습했다. 영상+기하와 영상+법선 좌표축을 비교했다. 후자는 선 위치·support·거리 특징을 빼지만 선에서 읽은 영상 특징과 역할·점수는 있으므로 순수 점 신경망과 같지는 않다.\n3. **실제 큰 오류 선택:** 점/query 특징만 쓰는 분류기와 여기에 선 특징의 요약을 더한 분류기를 비교했다. 기준은 calibration4그룹에서 F1을 최대화하는0.10~0.90 고정 격자로 선택하고 확인셋으로 다시 맞추지 않았다.','',
    '큰 오류 전용 예측기를 모든 점에 적용하는 대조, 실제 예측으로 큰 오류를 골라 적용하는 대조, 정답으로 큰 오류를 알려주는 진단을 분리했다. 전용 예측기는 필요한 법선 보정량을 예측하고 기존24개 후보 중 양의 예상 이득을 가진 후보를 고른다. 보정량은 기존처럼 상자 대각선15% 이내다. L1 손실의 출력은 보정 중심 예측값이며 통계적인 조건부 평균 보장은 아니다.','',
    '아래는 **재사용 합성 확인셋**의 수치다. 확인200관측 중130검출·1040점이며 실사297 성능과 섞지 않는다.','',
    '## 1. 큰 오류를 알아볼 수 있었나','',
    '| 점+선 특징 분류기 | 시드1 | 시드2 | 시드3 |','|---|---:|---:|---:|',
    '| 실제 큰 오류 점 수 /1040 | '+' | '.join(str(q['hard_points']) for q in hg)+' |',
    '| AUROC | '+' | '.join(f'{q["auroc"]:.3f}' for q in hg)+' |',
    '| Precision | '+' | '.join(f'{100*q["precision"]:.1f}%' for q in hg)+' |',
    '| Recall | '+' | '.join(f'{100*q["recall"]:.1f}%' for q in hg)+' |','',
    '점+선 특징의 큰 오류 순위 성능은 약0.93으로, 큰 오류를 전혀 알아볼 수 없는 상태는 아니었다. 하지만 한 시드 recall47.5%로 사전등록한 precision/recall 각각50% 기준을 모두 통과하지 못했다. 높은 AUROC가 현재 기준에서의 정확한 선택이나 올바른 보정 방향을 뜻하지는 않는다.','',
    '![큰 오류 식별](figures/hardness.png)','',
    '## 2. 어려운 점만 학습하면 보정 방향도 좋아졌나','',
    '| 법선 보정량 예측 | 학습 큰 오류 MAEpx | 확인 큰 오류 MAEpx |','|---|---:|---:|',
    *['| '+label+' | '+f'{df[k]["mean_px"]:.3f} | {dc[k]["mean_px"]:.3f} |' for k,label in [('standard','기존 V38 중심'),('balanced','큰 오류 비중 증가'),('hard_geometry','큰 오류 전용·영상+기하'),('hard_appearance','큰 오류 전용·영상+법선')]],'',
    f'큰 오류 전용 기하는 학습 오차18.214→11.040px로 줄었지만, 확인 구간에서는15.442→16.073px로 세 시드 모두 나빠졌다. 큰 오류 비중만 늘려도 확인15.725px로 개선되지 않았다. 아무 보정도 하지 않는0예측의 확인 법선 MAE는 {np.mean(dc["standard"]["zero_target_MAE_px"]):.3f}px였다.','',
    '전용 기하 모델의 법선 부호 정확도는 학습87.9~88.8%였으나 확인55.2~56.6%로 떨어졌다. 정답 법선 보정량이1px 이상인 후보에서 측정한 값이며, 부호 분포를 모르는 상태에서 단순히50%를 우연 기준으로 가정하지 않는다.','',
    f'학습12그룹에서 그룹을 통째로 빼고 예측한 OOF MAE도 {df["oof_hard_geometry"]["mean_px"]:.3f}px로 학습 내부11.040px보다 나빴다. OOF 모델은8그룹으로 학습하므로 데이터 수 차이도 포함한다. 그래도 **쉬운 점 비중만 줄이면 해결된다는 설명은 지지되지 않으며, 현재 특징·전용 예측기의 그룹 간 전달 문제가 남는다.** 이것만으로 정보 이론적인 불가능성을 증명한 것은 아니다.','',
    '![방향 예측의 전달](figures/direction.png)','',
    f'영상에 선 기하를 더하면 확인 MAE16.670→16.073px로 줄었고 차이95%구간은 {ci_text(dc["geometry_vs_appearance"]["difference"]["ci95"])}px였다. 선 특징의 부분 정보는 있지만 기존 중심이나0보정 기준을 넘을 정도의 효과는 아니었다.','',
    '## 3. 실제로 점을 보정하면','',*tablelines,'',
    f'주판정은 원래10.463px→{r["line_gate"]["mean_px"]:.3f}px로 개선되지 않았다. 시드별 개선율은 '+', '.join(f'{x:+.2f}%' for x in r['line_gate']['gain_percent'])+'이며 두 시드에서 악화됐다. 점 특징만으로 선택한10.868px보다 선 특징을 더한10.484px가 좋았지만, 기존 점 대비 안정적인 개선은 아니다.','',
    '큰 오류 전용 모델을 모든 점에 적용하면16.272px로 악화됐다. 특히 쉬운 점은 원래2.543px→9.807px였다. 전용 모델의 역할 범위를 넘어 적용하는 문제가 있으므로 이를 선 자체의 한계로 해석하지 않는다. 실제 큰 오류 선택이 그 악화를 크게 줄였지만 좋은 점의 손상과 어려운 점의 잘못된 보정은 남았다.','',
    '![전체 점 성능](figures/performance.png)','',
    '## 4. 큰 오류를 완벽하게 알려주면 해결되나','',
    f'정답 큰 오류만 골라 전용 모델을 적용하면10.204px였지만 차이95%구간은 {ci_text(r["GT_hard_geometry"]["difference"]["ci95"])}px로0을 포함했다. 시드별로1px 넘게 악화된 점도39/45/36개였다. **큰 오류를 완벽하게 골라도 전용 보정이 항상 맞지는 않았다.**','',
    '여기에 정답으로 해로운 보정까지 거절하면9.606px가 됐다. 이 값은 실제 적용 성능이 아니라 남아 있는 보정 선택 손실을 보여준다. 정답으로 기존 후보 중 최선을 고르는7.802px 역시 학습 가능성을 보장하지 않는 상한이다.','',
    '## 5. 큰 오류 사례의 형태 — 후속 기술 통계','',
    '다른 GT 꼭짓점5px 이내에 잘못 가까워진 사례는 시드별9/9/1개(큰 오류159/189/179개 중)였다. 이 기준에서는 대부분의 큰 오류를 단순 점 번호 바뀜으로 설명할 수 없다. 가까운 GT점만으로 실제 의미적 혼동을 확정한 것은 아니다.','',
    '큰 오류 중 정답 가시 물리선에서 독립2방향을 확보하지 못한 점은 약67~70%였다. 정답 선/offset으로 관측 가능한 성분을 모두 보정하는 진단에서도 큰 오류 평균 약26~30px가 남았다. 2D 전체를 복원할 관측이 부족한 사례가 있지만, 이번 법선 방향 예측 실패를 그것 하나로 설명할 수는 없다.','',
    '이 통계는 실행 후 원인 가설을 기술한 것이며 모델이나 기준을 다시 고르지 않았다. 새 실사GT 검수나 annotation 수정은 없다.','',
    '## 실제 점·선과 보정 방향 사례','',
    '세 사례는 모두 고정 seed1의 조건부 중앙값 사례다. 좋은 결과만 고르지 않고, 올바르게 큰 오류를 찾았지만 보정이 틀린 경우와 쉬운 점을 잘못 선택한 경우를 함께 보였다. 선은 실제 선택한 Hough 후보이며 attention 지도가 아니다. 점 오차가 줄어도 후보 선이 물리 경계에 정확히 맞았다는 뜻은 아니다.']
    for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, point{q["point"]}','',q['selection'],'',f'큰 오류 예측 확률 {q["hard_probability"]:.2f}; 점 오차 {q["base_px"]:.2f}→{q["after_px"]:.2f}px. 실제 법선 목표 {q["actual_normal_px"]:+.2f}px, 예측 {q["predicted_normal_px"]:+.2f}px.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 결론과 다음 판단','',
    '**현재 가장 구체적으로 남은 문제는 어려운 점을 알아보는 것보다, 그 점의 올바른 보정 방향·크기를 그룹 밖에서도 맞추는 것이다.** 큰 오류 선택 신호와 일부 선 기하 정보는 측정됐으나 운영 모델을 교체할 안정적인 보정 구조는 확보하지 못했다.','',
    '현재 데이터/특징을 그대로 두고 쉬운 점 가중치, gate, 불확실성 또는 gradient surgery만 반복하는 우선순위는 낮아졌다. 다음 검증에서는 큰 오류의 역할 대응과 실제 관측을 바꾸지 않은 상태에서 식별 가능한지부터 봐야 한다. 특히 관측 가능한 물리 선과 가상 꼭짓점의 대응을 명확히 하는 추가 증거 또는 다른 시점/시간 정보를 사용해, 새로운 정보가 들어왔을 때 방향 예측이 실제로 달라지는지 비교하는 것이 타당하다. 이 추가 관측·구조 학습은 이번에 실행하지 않았다.','',
    '이는 Deep Hough와 점의 결합이 원천적으로 불가능하다는 결론이 아니다. 이번 고정 특징과 작은 회귀/분류 모델, 같은2CAD의 재사용 합성 그룹 범위에서 확인된 실패다.','',
    '## 감사와 재현','',
    '새48개 모델의 예측을 재현했고60개 점 보정 배열의 모든 조건별 수치를 독립 계산했다. calibration/check 특징·GT를 크게 바꾸어도3개 재학습 모델이 동일했고, checkGT/점수 교란은 calibration 기준 선택에 영향을 주지 않았다. 같은 점 입력의 분류기는 두 선 후보 대조에서 정확히 같았다. 이전 완료 결과 파일은 모두 그대로 보존했다.','',
    '확인셋의 큰 오류 점 수는 시드별159/189/179개로 다르므로 조건부 평균과 bootstrap도 각 시드의 분모를 유지했다. 같은 source그룹을 함께 재표집하고 시드 평균을 구했다. 확인200관측 중 검출 실패70은1040점의 조건부 오차에 들어가지 않으며 새 모듈로 복구하지 않았다.','',
    '재현: `scripts/research/pallet_hard_direction_v39/README.md`. 사전PROTOCOL과일차SUMMARY,후속PHENOTYPE_PROTOCOL/PHENOTYPES를분리보존했다. 실제 신경망 학습/실사 평가/PCGrad는 수행하지 않았다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_hard_direction_v39/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'CASES.json',dict(rows=cases));write(RUN/'FIGURE_DATA.json',dict(table=table));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),phenotypes_sha256=sha(RUN/'PHENOTYPES.json')));print('REPORT39 COMPLETE',flush=True)
if __name__=='__main__':main()
