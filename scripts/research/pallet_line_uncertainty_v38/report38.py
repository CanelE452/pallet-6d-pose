from context38 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v38-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
from report37 import clipped
from evaluate38 import METHODS
TITLE='V38 점 위치의 선 보정 불확실성 검증'
LABELS=dict(anchor='V34 점 유지',mean_only='보정량 중심만',constant_marginal='상수 불확실성·후보별',constant_point='상수 불확실성·점별',appearance_marginal='영상 불확실성·후보별',appearance_point='영상 불확실성·점별',geometry_marginal='영상+기하 불확실성·후보별',geometry_point='영상+기하 불확실성·점별 [주판정]')

def pct(v):return ', '.join(f'{100*x:.2f}%' for x in v)
def fmtci(v):return '['+', '.join(f'{x:+.3f}' for x in v)+']'

def cases(p,probes):
    primary=[r for r in probes['all_primary_selected_cases'] if r['arm']=='both'];bad=max(primary,key=lambda r:r['after_px']-r['base_px']);choices=[('wrong_direction',bad['seed'],bad['index'],bad['point'],bad['candidate'],'Largest worsening among the four primary selected point-seeds; a selected-tail extreme, not typical performance.')]
    a=dict(np.load(V37/'arrays/v34_both_seed1.npz'));v=dict(np.load(RUN/'arrays/both_seed1.npz'));g=p['g'];check=(~p['train'])[:,None]&p['supported'];be=np.linalg.norm(a['anchor']-g['gt'],axis=-1);ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);benefit=be-ce.min(-1);meanerr=np.linalg.norm(v['points_mean_only']-g['gt'],axis=-1);used=v['selected_geometry_point']>=0
    for name,mask,value,description in [('missed_gain',check&~used&(benefit>3),benefit,'Median oracle improvement among supported rejected check points with >3px attainable gain.'),('preserved_point',check&~used&(be<2)&(meanerr-be>1),meanerr-be,'Median avoided worsening among supported accurate anchors (<2px) harmed >1px by the mean-only selector.')]:
        if not mask.any():continue
        target=np.median(value[mask]);i,j=np.unravel_index(np.where(mask,np.abs(value-target),np.inf).argmin(),mask.shape);candidate=int(ce[i,j].argmin()) if name=='missed_gain' else int(v['selected_mean_only'][i,j]);choices.append((name,1,int(i),int(j),candidate,description))
    out=[]
    for name,seed,i,j,k,description in choices:
        a=dict(np.load(V37/'arrays'/f'v34_both_seed{seed}.npz'));v=dict(np.load(RUN/'arrays'/f'both_seed{seed}.npz'));row=p['rows'][i];im=actual_image(row);h,w=im.shape[:2];n=a['candidate_normal'][i,j,k];rho=a['candidate_rho'][i,j,k];anchor=a['anchor'][i,j];gt=g['gt'][i,j];proposal=a['candidate'][i,j,k];actual=v['points_geometry_point'][i,j];mu=v['mu'][i,j,k]*p['diag'][i];width=v['width_geometry_point'][i,j,k]*p['diag'][i];truth=v['target'][i,j,k]*p['diag'][i];interval=np.stack((anchor+n*(mu-width),anchor+n*(mu+width)));role=int(a['candidate_role'][j,k]);coord=np.array([anchor,gt,proposal,actual]);fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained',gridspec_kw=dict(width_ratios=[1.2,1.2,1]))
        for ax,zoom in zip(axes[:2],(False,True)):
            bounds=(coord[:,0].min()-20,coord[:,1].min()-20,coord[:,0].max()+20,coord[:,1].max()+20) if zoom else (min(0,coord[:,0].min())-10,min(0,coord[:,1].min())-10,max(w,coord[:,0].max())+10,max(h,coord[:,1].max())+10)
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e7ebef');seg=clipped(n,rho,bounds)
            if seg is not None:ax.plot(*seg.T,c='#e58c29',ls='--',label='Candidate Hough line')
            ax.plot(*interval.T,c='#29a8b0',lw=6,alpha=.45,label='Predicted normal-target interval')
            for xy,label,color,marker in [(gt,'GT corner','#22a150','x'),(anchor,'V34 anchor','#337fc4','o'),(proposal,'Candidate projection','#e58c29','^'),(actual,'Conservative result','#dc6262','+')]:ax.scatter(*xy,c=color,marker=marker,s=50,label=label,zorder=5)
            ax.annotate('',xy=proposal,xytext=anchor,arrowprops=dict(arrowstyle='->',color='#e58c29',lw=1.5));ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[3],bounds[1]);ax.set_aspect('equal');ax.legend(fontsize=6)
        ax=axes[2];ax.errorbar(mu,2,xerr=width,fmt='o',c='#29a8b0',capsize=6,label='Predicted mean +/- interval');ax.scatter(truth,1,c='#22a150',marker='x',s=80);ax.scatter(a['candidate_amount'][i,j,k],0,c='#e58c29',marker='^',s=70);ax.axvline(0,c='gray',ls=':');ax.set(yticks=[0,1,2],yticklabels=['Proposed correction','True normal target','Predicted target'],xlabel='Signed normal correction (px)',title='One-dimensional uncertainty\nNot a 2D corner confidence region',ylim=(-.7,2.7));ax.grid(axis='x',alpha=.2)
        info=dict(name=name,seed=seed,id=row['id'],index=i,point=j,candidate=k,role=role,selection=description,base_px=float(np.linalg.norm(anchor-gt)),proposal_px=float(np.linalg.norm(proposal-gt)),after_px=float(np.linalg.norm(actual-gt)),actual_normal_px=float(truth),mean_normal_px=float(mu),halfwidth_px=float(width),selected=bool(v['selected_geometry_point'][i,j]>=0),covered=bool(abs(mu-truth)<=width))
        fig.suptitle(f'{name} / seed{seed} / {row["id"]} / point{j}\nAnchor {info["base_px"]:.2f}px, proposal {info["proposal_px"]:.2f}px, conservative result {info["after_px"]:.2f}px',fontsize=10);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);out.append(info)
    return out

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS' and read(RUN/'PROBE_AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');probe=read(RUN/'PROBES.json');reliability=read(RUN/'RELIABILITY.json');p=population();part=partition(p);key='check/all/all_detected';r=s['results'][key]['methods']['both'];uq=s['uncertainty'][key]['both'];(RUN/'figures').mkdir(exist_ok=True);table=[]
    for method in ('anchor',*METHODS):table.append(dict(method=method,label=LABELS[method],mean_px=r[method]['mean_px'],selected=r[method]['selected_points'],gains=r[method]['gain_percent'],ci=r[method]['difference']['ci95']))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');names=['mean_only','constant_point','appearance_point','geometry_point'];x=np.arange(len(names))
    for k in range(3):axes[0].bar(x+(k-1)*.25,[r[n]['gain_percent'][k] for n in names],width=.25,label=f'Seed{k+1}')
    axes[0].axhline(0,c='black',lw=1);axes[0].set(xticks=x,xticklabels=['Mean only','Constant UQ','Appearance UQ','Geometry UQ'],ylabel='Improvement vs anchor (%)',title='Average gain is not stable across seeds');axes[0].legend()
    for k in range(3):axes[1].bar(x+(k-1)*.25,[r[n]['selected_points'][k] for n in names],width=.25,label=f'Seed{k+1}')
    axes[1].set(xticks=x,xticklabels=['Mean only','Constant UQ','Appearance UQ','Geometry UQ'],ylabel='Selected points /1040',title='Pointwise-calibrated geometry uncertainty mostly abstains');fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');modes=['constant_point','appearance_point','geometry_point'];x=np.arange(3)
    for k in range(3):
        axes[0].bar(x+(k-1)*.25,[uq[m][k]['simultaneous_point_coverage']*100 for m in modes],width=.25,label=f'Seed{k+1}');axes[1].bar(x+(k-1)*.25,[uq[m][k]['mean_halfwidth_px'] for m in modes],width=.25)
    axes[0].axhline(90,c='gray',ls='--',label='Calibration target 90%');axes[0].set(xticks=x,xticklabels=['Constant','Appearance','Geometry'],ylim=(0,100),ylabel='All24 candidate coverage at each point (%)',title='Check coverage drops from calibration');axes[0].legend(fontsize=7);axes[1].set(xticks=x,xticklabels=['Constant','Appearance','Geometry'],ylabel='Mean interval half-width (px)',title='Sharper intervals alone do not establish safety')
    for mode,col in [('appearance','#809aad'),('geometry','#68a777')]:
        rows=reliability['reliability']['both_seed1'][mode]['bins'];axes[2].plot([q['mean_scale'] for q in rows],[q['mean_absolute_normal_error'] for q in rows],'-o',label=mode,c=col)
    axes[2].set(xlabel='Predicted error scale (normalized)',ylabel='Actual absolute error (normalized)',title='Seed1 check bins fixed by calibration');axes[2].legend();fig.savefig(RUN/'figures/calibration.png',dpi=150);plt.close(fig)
    funnels=[probe['oracles'][f'both_seed{s}']['loss_funnel'] for s in SEEDS];fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');labels=['Possible >1px gain','Also positive mean gain','Also positive lower gain'];x=np.arange(3)
    for k,f in enumerate(funnels):ax.bar(x+(k-1)*.25,[f[n] for n in ('possible_over1px','possible_with_positive_predicted_gain','possible_with_positive_lower_gain')],width=.25,label=f'Seed{k+1}')
    ax.set(xticks=x,xticklabels=labels,ylabel='Point count /1040',title='Where useful correction opportunities disappear');ax.legend();fig.savefig(RUN/'figures/funnel.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for seed in SEEDS:
        rows=[q for q in probe['curves'][f'both_seed{seed}'] if q['family']=='point'];axes[0].plot([q['calibration_level'] for q in rows],[q['check_selected_points'] for q in rows],'-o',label=f'Seed{seed}');base=r['anchor']['per_seed_px'][seed-1];axes[1].plot([q['calibration_level'] for q in rows],[q['check_mean_px']-base for q in rows],'-o',label=f'Seed{seed}')
    axes[0].set(xlabel='Calibration quantile (not selected on check)',ylabel='Selected check points',title='Post-primary fixed-quantile sensitivity');axes[1].set(xlabel='Calibration quantile',ylabel='Error difference vs anchor (px)',title='More conservative does not imply useful correction');axes[1].axhline(0,c='gray',ls='--');axes[0].legend();axes[1].legend();fig.savefig(RUN/'figures/curve.png',dpi=150);plt.close(fig);gallery=cases(p,probe)
    primary=uq['geometry_point'];const=uq['constant_point'];allsel=[q for q in probe['all_primary_selected_cases'] if q['arm']=='both'];oracle=probe['oracle_summary']['both'];dims={k:int(part[k].sum()) for k in ('fit','calibration','check')}
    body=['# '+TITLE,'','**불확실성으로 오차가 큰 후보를 어느 정도 구분할 수 있었다. 그러나 실제 보정할 점에서의 신뢰도와 안정적인 성능 개선은 확보하지 못했다. 보정량 중심을 틀리게 예측하는 문제도 함께 남아 있다.**','',
    '기존 선·점 모델을 고정하고 작은 불확실성 모듈을 붙여 검증했다. 총36개 gradient boosting 모델을 학습했다(60라운드씩2160라운드; 신경망 optimizer 업데이트0). 이 가운데18개는 그룹을 빼고 예측하는 OOF용,6개는 최종 보정량 보정 중심 예측기,12개는 불확실성 예측기다. 감사에서3개 모델은 별도로 다시 학습해 같은 결과인지 검사했다.','',
    '## 실험 설계와 비교 범위','',
    f'기존 합성600관측 중436검출을 사용했다. 기존 학습16그룹을 새 예측기 학습12그룹({dims["fit"]}검출)과 오차 범위 보정4그룹({dims["calibration"]}검출)으로 나누고, 기존 확인8그룹130검출·1040점에서 검사했다. 세 시드 모두 같은 그룹 분리와 고정 설정을 사용했다. **앞서 보고한 실사297 성능과 다른 합성 진단**이다.','',
    '보정 중심 예측기는 점/query·선 영상·기하·예측 support 정보를 통해 필요한 법선 보정량을 예측한다. 불확실성은 학습그룹을 통째로 뺀3fold 예측의 잔차로 학습했다. 같은 보정 중심 예측기에 상수/영상/영상+기하 불확실성을 붙였으므로 불확실성 입력의 효과를 분리한다. 코드에서 mean으로 부르는 분기는 절대오차(L1) 손실로 학습한 위치 예측값이며, 통계적인 조건부 평균을 보장하지 않는다. 보수적 하한은 중심이 평균인지와 무관하게 정답이 구간 안에 있다는 조건에서 성립한다.','',
    '추가 기하 입력은 예측 선의 가중 support 중심·퍼짐·질량, 점과 support 사이 접선 거리, 다른 후보와의 불일치다. 실제 정답 가시 구간·정답 역할은 입력에 넣지 않았다. 가중 support는 기존 격자 후보에서 추출한 영상 특징의 요약으로, 정밀한 물리 끝점이나 실제 가시 길이와 같지 않다.','',
    '오차 범위 보정4그룹에서90% 경험적 분위수를 계산했다. 후보별 방식은 각 후보의 잔차를, 점별 방식은 한 점의24후보 중 최대 잔차를 사용한다. 점별 방식이 후보 선택에 따른 위험을 일부 고려하지만, **이후 실제로 보정하는 작은 부분집합에 대한90% 보장을 주지는 않는다.**','',
    '새 평균/불확실성 학습에는 보정용4그룹을 넣지 않았지만, 기존 고정 V36 모델과 V35 offset은 과거 학습16그룹을 이미 사용했다. 같은2CAD와 재사용 확인셋이며, 엄밀한 독립 conformal 보장·미지자산·실사 최종 성능을 주장하지 않는다.','',
    '## 1. 불확실성은 실제 오차와 관련이 있었나','',
    '| 주판정: 영상+기하·점별 범위 | 시드1 | 시드2 | 시드3 |','|---|---:|---:|---:|',
    '| 오차 크기 순위 상관(Spearman) | '+' | '.join(f'{q["scale_spearman"]:.3f}' for q in primary)+' |',
    '| 한 점의24후보가 모두 범위에 들어온 비율 | '+' | '.join(f'{100*q["simultaneous_point_coverage"]:.2f}%' for q in primary)+' |',
    '| 예측 범위의 평균 반폭px | '+' | '.join(f'{q["mean_halfwidth_px"]:.3f}' for q in primary)+' |',
    '| 상수 불확실성의 평균 반폭px | '+' | '.join(f'{q["mean_halfwidth_px"]:.3f}' for q in const)+' |','',
    '오차 크기의 순위 관계와 상수 대비 좁은 범위는 확인됐다. 하지만 목표90% 범위의 확인셋 적용률이 떨어졌고, 사전등록한85% 최소 기준도 한 시드에서 미달했다. 순위를 잘 매기는 것과 정확한 오차 범위를 주는 것은 다르다. 또한 좁은 범위일수록 유용하다는 판정은 필요한 적용률이 충족될 때에만 가능하다.','',
    '![불확실성 검증](figures/calibration.png)','',
    '## 2. 불확실한 선을 거절하면 점이 좋아졌나','',
    '같은24개 선 후보와 최대 상자 대각선15% 보정량을 유지했다. 필요한 법선 보정량의 예측 중심 μ와 범위 반폭w로 가능한 점 오차 감소량의 하한을 계산하고, 하한이 양수인 후보만 선택했다. 없으면 원래 V34 점을 그대로 둔다.','',
    '`하한 = 2aμ − a²||n||² − 2|a|w` (a는 제안 법선 보정량, n은 저장된 선 법선). 정답 보정량이 범위 안에 있다면 이 하한이 양수인 보정은 실제 제곱 점 오차도 줄여야 한다. 이 항등식은 별도 검증했다.','',
    '| 방법 / V36 both 후보 | 전체1040점 평균 오차px | 시드별 보정 점 수 | 기준 대비 차이95%구간px |','|---|---:|---|---|']
    for q in table:body.append('| '+q['label']+' | '+f'{q["mean_px"]:.3f}'+' | '+str(q['selected'])+' | '+fmtci(q['ci'])+' |')
    body+=['','### 가림·잘림 조건별 주판정','','| 조건 | 대상 점 수 | V34 기준px | 불확실성 보정px |','|---|---:|---:|---:|']
    for condition,stratum in [('clean','all_detected'),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside')]:
        ck='check/'+condition+'/'+stratum
        if ck in s['results']:
            cc=s['results'][ck];mm=cc['methods']['both'];body.append('| '+condition+'/'+stratum+' | '+str(cc['points'])+' | '+f'{mm["anchor"]["mean_px"]:.3f} | {mm["geometry_point"]["mean_px"]:.3f} |')
    body+=['',
    f'주판정은 원래 점 {r["anchor"]["mean_px"]:.3f}px → {r["geometry_point"]["mean_px"]:.3f}px로 개선하지 못했다. 시드별 보정은1/3/0점,1040점씩 총3120점×시드 중4개(0.13%)뿐이다. 이4개 모두 정답 보정량이 예측 범위 밖이었다. 그중3개는1px 넘게 악화됐다. **표본4개로 일반 위험률을 추정할 수는 없지만, 전체 범위 적용률을 실제 선택점의 안전성으로 사용할 수 없다는 사례다.**','',
    f'영상 불확실성·점별 대조는 평균 {r["appearance_point"]["mean_px"]:.3f}px로 좋아 보였지만 시드별 개선율은 '+', '.join(f'{x:+.2f}%' for x in r['appearance_point']['gain_percent'])+'였다. 한 시드 악화와 조건별 실패가 있어 안정적 개선으로 채택하지 않았다. 이 대조를 확인 결과를 보고 주판정으로 바꾸지 않았다.','',
    '![성능과 보정률](figures/performance.png)','',
    '## 3. 보정량 중심과 불확실성 중 어디서 막혔나','',
    '후속 산술 진단은 새 학습 없이 정답 정보의 한 요소만 넣었다. 다음은 실사용 성능이 아닌 oracle 결과다.','',
    '| 후속 진단 / both | 점 오차px | 뜻 |','|---|---:|---|',
    f'| 보정 보정 중심만 정답, 현재 불확실성 유지 | {oracle["oracle_mean"]["mean_px"]:.3f} | 평균 보정 방향·크기를 맞췄을 때 남는 여지 |',
    f'| 중심은 현재 예측, 오차 반폭만 정확히 제공 | {oracle["oracle_radius"]["mean_px"]:.3f} | 현재 평균에서 불확실성을 정확히 알 때 가능한 보정 |',
    f'| 중심 정답·반폭0 | {oracle["oracle_both"]["mean_px"]:.3f} | 현재 후보 중 점 정답으로 최선을 고르는 상한 |','',
    '보정 중심만 정답으로 바꾸면9.360px, 반폭만 정확히 주면10.161px로 줄었다. 둘 다 오차 요인이 남아 있다는 진단이다. 서로 다른 oracle의 이득 크기를 곧바로 원인의 기여율로 해석하지 않는다. 정답으로 고른 선은 실제 물리 경계가 틀려도 점 오차를 우연히 줄일 수 있다.','',
    '| 개선 기회가 사라지는 단계 | 시드1 | 시드2 | 시드3 |','|---|---:|---:|---:|']
    for label,k in [('원래 후보 중1px 이상 좋아질 수 있는 점','possible_over1px'),('그중 평균 예측도 양의 이득을 주는 후보가 있는 점','possible_with_positive_predicted_gain'),('그중 불확실성 하한까지 양수인 점','possible_with_positive_lower_gain')]:body.append('| '+label+' | '+' | '.join(str(q[k]) for q in funnels)+' |')
    body+=['','![개선 기회의 손실](figures/funnel.png)','',
    '따라서 단순히 기준을 더 보수적으로 바꾸면 해결된다고 보기 어렵다. 필요한 보정의 평균 방향·크기가 틀리거나 작게 예측되어 대부분의 좋은 후보가 불확실성 적용 전에 이미 제외된다. 반대로 큰 이득을 자신 있게 예측한 극단적 오류 일부는 범위 보정 후에도 선택된다. 단순한 전체 후보 신뢰도 대신 실제 선택될 사례의 잔차를 검증해야 한다.','',
    '## 4. 보수성만 조절하면 되나','',
    '일차 결과를 보존한 뒤50/70/80/90/95% 분위수를 보정용4그룹에서만 계산해 민감도를 확인했다. 확인셋에서 최선 기준을 골라 성능으로 보고하지 않았다. 범위를 좁히면 적용률은 늘지만 오보정이 남았고, 넓히면 대부분 거절하는 양상이 남았다.','',
    '![보수성 대조](figures/curve.png)','',
    '## 실제 이미지와 보정량 범위','',
    '아래는 원본 합성 이미지와 후보 선, 점, 법선 보정량의1차원 범위다. 청록 구간은 목표 보정량의 예측 범위로 실제 선 가시 구간이나2D 꼭짓점 신뢰영역이 아니다. 주황 삼각형은 검토한 후보가 제안하는 점이며, 거절한 경우 빨간 최종점은 원래 파란 점에 겹친다.']
    for q in gallery:body+=['',f'### {q["name"]}: seed{q["seed"]} / {q["id"]} / point{q["point"]}','',q['selection'],'',f'원래 점 {q["base_px"]:.2f}px → 최종 {q["after_px"]:.2f}px. 검토 후보 {q["proposal_px"]:.2f}px. 실제 법선 보정량 {q["actual_normal_px"]:+.2f}px, 예측 {q["mean_normal_px"]:+.2f}±{q["halfwidth_px"]:.2f}px.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 이번 결론과 다음 판단','',
    '**사용할 수 있는 부분은 오차 크기를 구분하는 불확실성 신호다. 그러나 이번 모듈을 그대로 붙여 안정적으로 개선되는 구조는 확보하지 못했다.** 보정 거절로 대부분의 점을 유지한 것을 성능 개선으로 세지 않았으며 운영 모델을 교체하지 않는다.','',
    '다음에 검증할 대상은 큰 점 오류에서 필요한 법선 보정의 부호·크기를 현재 특징으로 식별할 수 있는지, 그리고 실제 선택되는 극단 사례에서 잔차 범위가 맞는지다. 전체 작은 오차가 많은 표본의 평균이나 신뢰도 순위만 좋아지는 것으로는 충분하지 않다. V27/V37에서도 방향 목표를 시도했으므로 자유 head나 손실만 다시 바꾸기 전에, 어려운 점의 식별 가능성과 선택 후 조건부 잔차를 분리해 확인해야 한다. 이 후속 학습은 이번에 실행하지 않았다.','',
    '이번 실험은 Hough 분기를 다시 학습하거나 end-to-end 아키텍처를 교체한 것이 아니라, 고정 후보에 평균/불확실성 예측 모듈을 붙인 빠른 진단이다. Gradient surgery는 사용하지 않았다. 부호가 틀린 평균과 실제 선택 사례의 범위 실패가 점·선 손실 경사 충돌 때문이라는 근거는 이번 검사에서 나오지 않았다.','',
    '## 감사와 재현','',
    '모든36개 모델의 저장 예측을 재현했다(OOF18·최종평균6·scale12).42개 점 보정 배열과 모든 조건별 수치를 독립 재계산했고, 새 학습에 쓰지 않는 calibration/check 특징·정답을 크게 바꿔도3개 재학습 모델이 동일했다. checkGT를 바꿔도 범위 보정 분위수는 같았다. 정답이 범위 안에 있고 선택된 사례에서는 실제 오차가 증가하지 않는 항등식도 확인했다. 후속18개 oracle과60개 분위수 대조를 재계산했다. 이전 완료 결과와 원본 라벨은 그대로 보존했다.','',
    '확인200관측 중 검출 실패70은 조건부1040점 오차에서 제외되어 있고 이번 모듈이 복구하지 않았다. 수동 실사GT를 새로 평가하거나 수정하지 않았다. 자세한 재현은 `scripts/research/pallet_line_uncertainty_v38/README.md`; 사전 기준은PROTOCOL, 일차결과SUMMARY, 후속진단PROBE_PROTOCOL/PROBES에 따로 저장했다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_line_uncertainty_v38/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}code{background:#eef3f8;overflow-wrap:anywhere}</style>'+html+'</html>');write(RUN/'CASES.json',dict(rows=gallery));write(RUN/'FIGURE_DATA.json',dict(table=table));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),probes_sha256=sha(RUN/'PROBES.json'),reliability_sha256=sha(RUN/'RELIABILITY.json')));print('REPORT38 COMPLETE',flush=True)

if __name__=='__main__':main()
