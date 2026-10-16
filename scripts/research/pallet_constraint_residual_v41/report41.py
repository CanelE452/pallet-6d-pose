from context41 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v41-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
TITLE='V41 물리 제약 계산과 잔차 학습 검증'
LABELS=dict(pred_segment='실제 영상 선분·예측 대응',GT_match_segment='영상 선분·GT 대응 [진단]',GT_line='정확한 GT 물리선 [진단]')
def ci_text(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'

def gallery(p):
    v=dict(np.load(RUN/'arrays/seed1.npz'));quality=dict(np.load(RUN/'arrays/quality_oracle_seed1.npz'));match=dict(np.load(V40/'arrays/matching_seed1.npz'));g=p['g'];base=np.linalg.norm(v['anchor']-g['gt'],axis=-1);actualkey='pred_segment_physical3_residual_gate';actual=np.linalg.norm(v['points_'+actualkey]-g['gt'],axis=-1);ideal=np.linalg.norm(v['points_GT_line_physical3_residual']-g['gt'],axis=-1);direct=np.linalg.norm(v['points_pred_segment_physical3_direct']-g['gt'],axis=-1);check=(~p['train'])[:,None];selected=v['selected_'+actualkey];role=np.array(INC)[np.arange(8)[None],np.maximum(selected,0)];visible=g['length'][np.arange(p['N'])[:,None],role]>0;cases=[]
    configs=[('actual_false_support',check&(selected>=0)&~visible&(actual-base>1),actual-base,actualkey,'Median >1px harm with no GT-visible support for selected role, seed1 check.'),('actual_direct_help',check&(base-direct>1),base-direct,'pred_segment_physical3_direct','Median >1px improvement of actual physical-normal direct model, seed1 check.'),('GT_constraint_help',check&(base-ideal>1),base-ideal,'GT_line_physical3_residual','Median >1px improvement with ideal GT physical line, seed1 check; diagnostic only.')]
    for name,mask,value,key,rule in configs:
        if not mask.any():continue
        i,j=map(int,np.unravel_index(np.where(mask,np.abs(value-np.median(value[mask])),np.inf).argmin(),mask.shape));k=int(v['selected_'+key][i,j]);r=int(INC[j][k]);chosen=v['points_'+key][i,j];source='GT_line' if key.startswith('GT_line') else 'pred_segment';seg=g['support'][i,r] if source=='GT_line' else match['pred_segment'][i,r];im=actual_image(p['rows'][i]);h,w=im.shape[:2];anchor=v['anchor'][i,j];gt=g['gt'][i,j];coords=np.array([anchor,gt,chosen]);fig,axes=plt.subplots(1,3,figsize=(16,5.7),layout='constrained',gridspec_kw=dict(width_ratios=[1.1,1.1,1]))
        for ax,zoom in zip(axes[:2],(False,True)):
            bounds=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#edf0f3');ax.plot(*seg.T,c='#e0a24a',marker='.',lw=2,label='Constraint segment'+(' [GT]' if source=='GT_line' else ' [observed]'))
            if g['length'][i,r]>0:ax.plot(*g['support'][i,r].T,c='#888888',ls='--',label='GT visible support')
            for q,label,color,marker in [(gt,'GT virtual corner','#28a04e','x'),(anchor,'V34 anchor','#357ec0','o'),(chosen,'Applied correction','#d56b6b','+')]:ax.scatter(*q,c=color,marker=marker,s=60,label=label,zorder=6)
            ax.annotate('',xy=chosen,xytext=anchor,arrowprops=dict(arrowstyle='->',color='#d56b6b'));ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]));ax.set_aspect('equal');ax.legend(fontsize=7)
        prefix=source+'_physical3';values=[float(v[prefix+'_target'][i,j,k]*p['diag'][i]),float(v[prefix+'_prior'][i,j,k]*p['diag'][i]),float(v['mu_'+prefix+'_direct'][i,j,k]*p['diag'][i]),float(v['mu_'+prefix+'_residual'][i,j,k]*p['diag'][i])];axes[2].scatter(values,np.arange(4),c=['#28a04e','#e0a24a','#357ec0','#a17fbb'],s=65);axes[2].set(yticks=np.arange(4),yticklabels=['True normal target','Analytic constraint','Direct regression','Constraint + residual'],xlabel='Signed normal correction (px)',ylim=(-.6,3.6));axes[2].tick_params(axis='y',labelsize=8);axes[2].axvline(0,c='gray',ls=':');axes[2].grid(axis='x',alpha=.2);after=float(np.linalg.norm(chosen-gt));fig.suptitle(f'{name} / {p["rows"][i]["id"]} / point{j}, role{r}\nAnchor {base[i,j]:.2f}px -> {after:.2f}px',fontsize=12);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);cases.append(dict(name=name,id=p['rows'][i]['id'],index=i,point=j,role=r,method=key,rule=rule,base_px=float(base[i,j]),after_px=after,normal_values_px=values))
    return cases

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'PROBE_AUDIT.json')['status']=='PASS';p=population();s=read(RUN/'SUMMARY.json');probe=read(RUN/'PROBE.json');r=s['results']['check/all/all_detected'];d=s['direction']['check/all/hard'];(RUN/'figures').mkdir(exist_ok=True);table=[];normal=[]
    for source in SOURCES:
        for family in FAMILIES:
            prefix=source+'_'+family;table.append(dict(source=source,family=family,label=LABELS[source]+(' / 기존24방향' if family=='legacy24' else ' / 물리선3방향'),direct=r[prefix+'_direct']['mean_px'],analytic=r[prefix+'_analytic']['mean_px'],residual=r[prefix+'_residual']['mean_px'],gate=r[prefix+'_residual_gate']['mean_px']))
        q=d[source+'_physical3'];normal.append(dict(source=source,label=LABELS[source],counts=q['direct']['per_seed_points'],direct=q['direct']['mean_px'],analytic=q['analytic']['mean_px'],residual=q['residual']['mean_px']))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');x=np.arange(3);labels=['Observed match','GT match','GT physical line']
    for ax,values,title,ylabel in [(axes[0],[q for q in table if q['family']=='physical3'],'Same physical-normal proposals','All-point error (px)'),(axes[1],normal,'Valid hard-point roles only; populations differ','Normal target MAE (px)')]:
        for offset,mode,color in [(-.25,'direct','#6d97b9'),(0,'analytic','#dba755'),(.25,'residual','#a082b3')]:ax.bar(x+offset,[q[mode] for q in values],width=.25,label=mode,color=color)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,title=title);ax.legend(fontsize=8)
    axes[0].axhline(r['anchor']['mean_px'],c='gray',ls='--');fig.savefig(RUN/'figures/comparison.png',dpi=150);plt.close(fig)
    cats=['no_GT_visible_support','physical_mismatch','physical_close'];counts=[sum(row['residual_gate'][cat]['harmed_over1px'] for row in probe['selected_categories']) for cat in cats];fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.bar(['No GT-visible\nrole support','Visible but\nmisaligned','Close physical\nsupport'],counts,color=['#cb7474','#daa258','#72a58b']);ax.set(ylabel='Harmed >1px (point x seed)',title='Actual constraint + residual + frozen hardness gate')
    for i,y in enumerate(counts):ax.text(i,y+.3,str(y),ha='center')
    ax.set_ylim(0,max(counts)*1.15);fig.savefig(RUN/'figures/harm.png',dpi=150);plt.close(fig);cases=gallery(p)
    pointrows=['| 输入/방향 | 직접 회귀 | 계산만 | 계산+잔차 | 계산+잔차+기존 선택기 |','|---|---:|---:|---:|---:|']+[f'| {q["label"]} | {q["direct"]:.3f} | {q["analytic"]:.3f} | {q["residual"]:.3f} | {q["gate"]:.3f} |' for q in table];pointrows[0]=pointrows[0].replace('输入','입력')
    normalrows=['| 물리선 입력 | 큰 오류 역할 수: 시드별 | 직접 회귀 | 계산만 | 계산+잔차 |','|---|---|---:|---:|---:|']+[f'| {q["label"]} | {q["counts"]} | {q["direct"]:.3f} | {q["analytic"]:.3f} | {q["residual"]:.3f} |' for q in normal]
    cond=[]
    for condition in ('clean/all_detected','corner20/hidden','crop20/outside','crop40/outside'):
        rr=s['results']['check/'+condition];cond.append(dict(condition=condition,anchor=rr['anchor']['mean_px'],direct=rr['pred_segment_physical3_direct']['mean_px'],gate=rr['pred_segment_physical3_residual_gate']['mean_px'],GT=rr['GT_line_physical3_residual']['mean_px']))
    condrows=['| 조건 | 기존 점 | 실제 직접 회귀 | 실제 잔차+선택 | GT선 잔차 [진단] |','|---|---:|---:|---:|---:|']+[f'| {q["condition"]} | {q["anchor"]:.3f} | {q["direct"]:.3f} | {q["gate"]:.3f} | {q["GT"]:.3f} |' for q in cond]
    body=['# '+TITLE,'','**정확한 물리선을 올바른 법선 방향으로 연결하면 점 보정에 도움이 됐다. 그러나 실제 대응 선에 같은 계산·잔차 구조를 적용하면 악화됐다. 실제 입력에서는 물리선 방향의 직접 회귀가 작지만 세 시드에서 일관된 개선을 보였다.**','',
    '새27개 HGB모델×60라운드=1620라운드, 감사재학습3개 별도, 신경망 업데이트0. 기존 V34점/V40선분을 고정한 합성 진단이다. 같은2CAD·학습12/기준선택4/재사용확인8그룹, 확인200관측 중130검출·1040점. 미검출70개는 점 오차에 포함하지 않는다. V35상수offset/상류모델은 과거16그룹 학습 이력이 있다. 새실사평가/수동GT수정/렌더는 없다.','',
    '## 이전 투영 실험과 이번 대조의 차이','',
    'V30/V35에서도 선 법선 방향 투영을 실험했다. 이번에는 V40의 같은 영상 선분/GT대응/정확물리선 입력을 유지하면서 직접 보정량 회귀와 기하 계산값+잔차 회귀를 같은 특징·같은 학습 표본으로 비교했다. 두 모델 모두 기하 계산값을 입력으로 받는다. 단순 투영 자체를 새 아이디어로 주장하지 않는다.','',
    '기존24방향에서는 물리선의 보정량을 기존 후보 법선에 사영했다. 물리선3방향에서는 점에 연결된 세 역할의 실제 대응 선 법선을 이동 방향으로 사용했다. 각 보정은 상자 대각선15% 이내이고 선택한 물리선의 접선 성분은 유지한다. 두 후보군은 입력 요약·수·방향도 달라지므로 후보군 사이의 차이를 방향 하나만의 효과라고 하지 않는다.','',
    '물리선 잔차 목표는 `선 법선·GT점 − 선 거리 − 기존 역할 offset`이다. 같은 선이 주어졌다면 기존 점 위치와 무관한 선/꼭짓점 제약 오차만 학습한다. 실제 추론에는 GT점을 사용하지 않는다. 선 역할/가시성을 정답으로 알려주는 대조는 모두 비배포 진단이다.','',
    '## 전체 점 결과 — 단위px, 낮을수록 좋음','',
    '기존 점10.463px, V40실제선분보정10.329px. 아래는 모두 같은1040점·3시드 평균이다. 선택기는 V39 큰 오류 예측과 calibration 기준을 고정 재사용했으며 새로 튜닝하지 않았다.','',*pointrows,'',
    '정확한 GT물리선+물리방향 잔차는8.772px로 기존 점보다15.61/16.59/16.26% 개선, 차이95%구간[-3.639,−.254]px였다. 이 수치는 **GT선이 주어지는 조건부 진단**이며 실사용8.772px를 달성한 것이 아니다. 직접 회귀9.078px도 좋아졌고, 잔차의 추가 차이−.306px 구간[-.987,+.233]은0을 포함했다. 점 성능에서 잔차만의 추가 우월성이 확정된 것은 아니다.','',
    '실제 선분+물리방향에서는 계산만16.799px, 계산+잔차12.564px였다. 큰 오류 선택기로 제한해도10.821px, 세 시드 모두2.05~4.97% 악화됐다. 정확하지 않은 제약을 계산 경로가 강하게 전달하는 문제를 잔차 모델이 충분히 상쇄하지 못했다.','',
    '반면 실제 물리방향의 직접 회귀는10.410px로0.39/0.61/0.52% 개선, 차이95%구간[-.076,−.028]px였다. 모든 확인 조건/층에서 최대 악화는약0.045%로 보존 기준1% 이내였다. 다만 등록한 매시드1% 성능 개선에는 미달이고 V40평균10.329px보다도 좋지 않다. **작고 일관된 효과라는 단서는 남기되 독립 검증된 구조로 채택하지 않는다.**','',
    '![전체점과 법선 목표 비교](figures/comparison.png)','',
    '## 보정 방향·크기 예측과 적용 결과가 다른 이유','',*normalrows,'',
    '이 표는 각 입력에서 선이 있는 큰 오류 역할만 평가한다. 입력별 가시 범위와 법선 방향이 달라서 서로 다른 행의 절대MAE를 직접 비교하면 안 된다. 같은 행의 직접/계산/잔차 비교가 대조다. 기존24방향의 수치는 SUMMARY에 별도로 보존했다.','',
    '정확한 GT선에서는 직접14.169px→계산1.456px/잔차1.582px로 크게 줄었다. 잔차 부호 정확도도98~99%였지만 계산만보다 법선MAE가 낮지는 않았다. 반대로 실제 대응 선에서는 직접6.981px→잔차9.369px로 악화됐다. “회귀가 기하를 못 배운다”는 문제는 정확한 제약 조건에서 완화됐지만, 실제 선의 제약 자체가 틀린 경우에는 강한 계산 경로가 해로웠다.','',
    '정답 대응으로 같은 LSD선분을 골라주면 법선 잔차1.472px였지만 전체 점10.303px, 차이구간[-.429,+.061]로0을 포함했다. V40에서 정답 대응으로 찾을 수 있었던 것은841가시역할 중532개뿐이며 짧은 구간의 방향/거리 오차도 남는다. GT대응과 정확한 GT선의 차이에는 관측범위·정밀도·구간 정보가 함께 포함된다.','',
    '## 실제 적용 피해는 어디서 생겼나','',
    '주판정인 실제 잔차+기존 선택기의1px초과 피해45건(점×시드)을 후속 분해했다. 해당 역할에 GT가시지원이 없는 경우34건, 가시지원은 있지만3px/10도 품질 기준에서 벗어난 경우8건, 가까운 물리지원3건이었다. GT가시지원 없음은 캐시에 해당 역할의 가시 선이 없다는 뜻이며 영상 선분 자체가 가짜라는 증명은 아니다.','',
    '![피해 사례의 물리 지원](figures/harm.png)','',
    '정답 물리선 품질로 나쁜 선택을 거절하는 비배포 진단은10.821→10.423px로 완화됐다. 기존 점 대비 개선은0.24/0.25/0.69%로 작았다. 잘못된 선을 거절하는 것만으로 정확한GT선의8.772px 수준을 회복하지는 못했다. 이 품질 거절에는 점 정답 이득을 사용하지 않았지만 선GT/가시성을 사용하므로 실제 선택기가 아니다.','',
    '## 가림·잘림','',*condrows,'',
    '정확한 GT선의 잔차 보정은 표의 가림·화면밖 조건에서 세 시드 모두 개선됐다. 그러나 잘림20/40의 쉬운 점 일부층은 최대8.07/13.87% 악화돼 모든 조건 보존을 만족한 것은 아니다. 실제 잔차+선택기는 화면밖 잘림에서 세 시드 모두 악화했다. 전체 평균이나 GT조건 개선을 안정적인 실사 복원 성공으로 해석하지 않는다.','',
    '## 실제 사례','',
    '고정seed1의 조건부 중앙값 사례다. 주황은 이번에 실제 사용한 제약 선분(정답 진단은GT표시), 파랑은기존점, 빨강은적용후점, 초록은GT다. 오른쪽은 같은 후보에서의 정답 보정량/계산값/직접회귀/잔차 예측이다. attention 그림이 아니다.']
    for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, point{q["point"]}','',q['rule'],'',f'점 오차 {q["base_px"]:.2f}→{q["after_px"]:.2f}px. 방법 `{q["method"]}`.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 현재 판단','',
    '현재 근거는 “선과 점이 공존할 수 없다”보다 **정확한 역할·방향의 제약은 도움이 되지만, 실제로 틀린 제약을 적용하는 손실이 이득을 압도한다**는 설명을 더 지지한다. GT조건이 좋아졌다고 실제 선이 그 수준으로 예측된다는 보장은 없다.','',
    '강한 기하 계산+잔차 모듈은 이번 실제입력 기준으로 채택하지 않는다. 실제 물리법선 후보+직접 회귀의 작은 일관된 개선은 보존하되, 같은2CAD/재사용8그룹이므로 독립 장면에서 확인하기 전 안정적 향상이라고 확정하지 않는다. 다음 범위는 단순 점 어려움보다 해당 역할의 물리 제약이 실제로 유효한지를 예측할 수 있는지와 새 장면에서 작은 이득이 재현되는지다. 새 선택기/독립렌더/장기신경망학습은 이번에 실행하지 않았다.','',
    '## 검증','',
    '27모델 예측과186개 점 배열을 독립 재계산했다. 3개 모델에서 비학습 영역 특징/GT/유효성 교란에도 재학습 결과가 같았다. 실제 입력의GT독립성,기존24직접모델의V40완전재현,접선보존·이동제한·잔차offset항등식을 검사했다. 후속 물리품질은 외적 기반 거리로 별도 계산했고 GT품질거절 출력을 재검사했다. 이전 완료 결과는 모두 보존했다.','',
    '재현: `scripts/research/pallet_constraint_residual_v41/README.md`. 일차PROTOCOL/SUMMARY와후속PROBE_PROTOCOL/PROBE분리. 동일3시드 고정 후 source그룹5000bootstrap이므로 시드모집단의불확실성까지표현한구간은아니다. 실제신경망/실사/수동GT변경없음.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_constraint_residual_v41/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',dict(table=table,normal=normal,conditions=cond,harm_counts=counts));write(RUN/'CASES.json',dict(rows=cases));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),probe_sha256=sha(RUN/'PROBE.json')));print('REPORT41 COMPLETE',flush=True)
if __name__=='__main__':main()
