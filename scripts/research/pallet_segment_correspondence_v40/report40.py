from context40 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v40-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown,cv2
from features25 import actual_image
from report37 import clipped
TITLE='V40 선분·끝점 대응 정보가 보정에 주는 효과'
LABELS=dict(baseline='기존 특징',pred_segment='영상 선분 + 예측 대응',GT_match_segment='영상 선분 + 정답 대응 [진단]',GT_line='정확한 물리 직선 [진단]',GT_segment='정확한 가시 구간·끝점 [진단]',GT_shuffled_segment='역할을 바꾼 가시 구간 [대조]')
ENGLISH=['Baseline','Image segments\nPredicted match','Image segments\nGT match','GT physical\ninfinite line','GT visible\nsupport extent','GT extent\nwrong role']
def ci_text(values):return '['+', '.join(f'{x:+.3f}' for x in values)+']'
def gains(row):return ', '.join(f'{x:+.2f}' for x in row['gain_percent'])

def gallery(p):
    v=dict(np.load(RUN/'arrays/seed1.npz'));a=dict(np.load(V37/'arrays/v34_both_seed1.npz'));m=dict(np.load(RUN/'arrays/matching_seed1.npz'));gt=p['g']['gt'];base=np.linalg.norm(v['anchor']-gt,axis=-1);actual=np.linalg.norm(v['points_pred_segment_all']-gt,axis=-1);ideal=np.linalg.norm(v['points_GT_segment_all']-gt,axis=-1);gain=base-actual;check=(~p['train'])[:,None];cases=[]
    selected=v['selected_pred_segment_all'];roles=np.repeat(np.array(INC),8,axis=1)[np.arange(8)[None],np.maximum(selected,0)];has_observation=(selected>=0)&m['pred_valid'][np.arange(p['N'])[:,None],roles]
    for name,mask,value,rule in [('actual_help',check&(gain>1)&has_observation,gain,'Seed1 check: median >1px improvement, selected role has an observed image segment.'),('actual_harm',check&(gain<-1)&has_observation,-gain,'Seed1 check: median >1px worsening, selected role has an observed image segment.'),('GT_support_gap',check&v['hard']&(actual-ideal>1),actual-ideal,'Seed1 hard check: median >1px advantage of privileged GT support over actual inputs.')]:
        if not mask.any():continue
        i,j=map(int,np.unravel_index(np.where(mask,np.abs(value-np.median(value[mask])),np.inf).argmin(),mask.shape));k=int(v['selected_pred_segment_all'][i,j]);selection='actual predicted correction'
        if k<0:k=int(v['selected_GT_segment_all'][i,j]);selection='GT-support diagnostic correction (actual abstained)'
        if k<0:k=int(a['candidate_score'][i,j].argmax());selection='top-score diagnostic (both abstained)'
        role=int(np.repeat(np.array(INC),8,axis=1)[j,k]);im=actual_image(p['rows'][i]);h,w=im.shape[:2];anchor=v['anchor'][i,j];ap=v['points_pred_segment_all'][i,j];gp=v['points_GT_segment_all'][i,j];coords=np.array([anchor,ap,gp,gt[i,j]]);fig,axes=plt.subplots(1,3,figsize=(16,5.7),layout='constrained',gridspec_kw=dict(width_ratios=[1.1,1.1,1]))
        for ax,zoom in zip(axes[:2],(False,True)):
            bounds=(coords[:,0].min()-25,coords[:,1].min()-25,coords[:,0].max()+25,coords[:,1].max()+25) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#edf0f3')
            seg=clipped(a['candidate_normal'][i,j,k],a['candidate_rho'][i,j,k],bounds)
            if seg is not None:ax.plot(*seg.T,c='#999999',ls=':',lw=1,label='Hough candidate line')
            for q,ok,label,color,ls in [(m['pred_segment'][i,role],m['pred_valid'][i,role],'Image segment / predicted role','#e09c25','-'),(m['GT_match_segment'][i,role],m['GT_match_valid'][i,role],'Image segment / GT role','#1dabb3','--'),(p['g']['support'][i,role],p['g']['length'][i,role]>0,'GT visible support envelope','#555555','--')]:
                if ok:ax.plot(*q.T,c=color,ls=ls,marker='.',lw=2,label=label)
            for xy,label,color,marker in [(gt[i,j],'GT virtual corner','#2d9b51','x'),(anchor,'V34 anchor','#377cc0','o'),(ap,'Actual correction','#d66767','+'),(gp,'GT-support correction [oracle]','#a077ba','+')]:ax.scatter(*xy,c=color,marker=marker,s=60,label=label,zorder=6)
            ax.annotate('',xy=ap,xytext=anchor,arrowprops=dict(arrowstyle='->',color='#d66767'));ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]));ax.set_aspect('equal');ax.legend(fontsize=6,loc='best')
        true=float(v['target'][i,j,k]*p['diag'][i]);vals=[float(v['mu_baseline_all'][i,j,k]*p['diag'][i]),float(v['mu_pred_segment_all'][i,j,k]*p['diag'][i]),float(v['mu_GT_segment_all'][i,j,k]*p['diag'][i]),true];axes[2].scatter(vals,np.arange(4),c=['#8e9cab','#e09c25','#a077ba','#2d9b51'],s=70);axes[2].set(yticks=np.arange(4),yticklabels=['Baseline target','Actual segment target','GT-support target [oracle]','True normal target'],xlabel='Signed normal correction (px)',ylim=(-.6,3.6));axes[2].tick_params(axis='y',labelsize=8);axes[2].axvline(0,c='gray',ls=':');axes[2].grid(axis='x',alpha=.2);fig.suptitle(f'{name} / seed1 / {p["rows"][i]["id"]} / point{j}, role{role}\nAnchor {base[i,j]:.2f}px -> Actual {actual[i,j]:.2f}px; GT-support diagnostic {ideal[i,j]:.2f}px',fontsize=12);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig)
        cases.append(dict(name=name,index=i,point=j,role=role,candidate=k,candidate_selection=selection,id=p['rows'][i]['id'],rule=rule,base_px=float(base[i,j]),actual_px=float(actual[i,j]),GT_support_px=float(ideal[i,j]),normal_targets_px=vals))
    return cases

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'PROBE_AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');probe=read(RUN/'PROBE.json');p=population();(RUN/'figures').mkdir(exist_ok=True);sources=('baseline',*SOURCES);r=s['results']['check/all/all_detected'];d=s['direction']['check/all/hard'];table=[]
    for source in sources:
        method='standard' if source=='baseline' else source+'_all';table.append(dict(source=source,label=LABELS[source],normal_all=d['all'][source]['mean_px'],normal_hard=d['hard'][source]['mean_px'],point=r[method]['mean_px'],gains=r[method]['gain_percent'],ci=r[method]['difference']['ci95']))
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained');x=np.arange(6)
    for ax,family,title in zip(axes,FAMILIES,['All-point fitting','Hard-point-only fitting']):
        values=[d[family][source]['mean_px'] for source in sources];ax.bar(x,values,color=['#8899aa','#e0a65b','#86bcc6','#ab8bc0','#ab8bc0','#bdb7c3']);ax.set(xticks=x,xticklabels=ENGLISH,ylabel='Hard-point normal target MAE (px)',title=title+' / reused check');ax.tick_params(axis='x',labelsize=8);ax.axhline(values[0],c='gray',ls='--')
        for i,y in enumerate(values):ax.text(i,y+.08,f'{y:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)*1.12)
    fig.savefig(RUN/'figures/direction.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.barh(ENGLISH,[q['point'] for q in table],color=['#8899aa','#e0a65b','#86bcc6','#ab8bc0','#ab8bc0','#bdb7c3']);ax.axvline(r['anchor']['mean_px'],c='gray',ls='--',label='V34 anchor');ax.invert_yaxis();ax.set(xlabel='Point error (px), same1040checkpoints /3seed means',title='GT observations are diagnostics, not deployed inputs');ax.legend()
    for i,q in enumerate(table):ax.text(q['point']+.03,i,f'{q["point"]:.3f}',va='center')
    ax.set_xlim(0,r['anchor']['mean_px']*1.08);fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    cov=s['coverage']['check'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(3);axes[0].bar(x,[q['pred_segment']['assigned_visible_roles'] for q in cov],label='Predicted match / GT-visible');axes[0].bar(x,[q['pred_segment']['assigned_invisible_roles'] for q in cov],bottom=[q['pred_segment']['assigned_visible_roles'] for q in cov],label='Predicted match / GT-invisible');axes[0].axhline(532,c='#3a9b70',ls='--',label='GT-matched LSD segments');axes[0].set(xticks=x,xticklabels=['Seed1','Seed2','Seed3'],ylabel='Role assignments',title='841 GT-visible roles /130 detected observations');axes[0].legend(fontsize=8);axes[1].bar(['Predicted\nmatch','GT\nmatch'],[np.mean([q['pred_segment']['physical_support_line_distance_px'] for q in cov]),cov[0]['GT_match_segment']['physical_support_line_distance_px']]);axes[1].set(ylabel='Physical support-to-line distance (px)',title='Conditional assigned-visible populations differ');fig.savefig(RUN/'figures/matching.png',dpi=150);plt.close(fig)
    cases=gallery(p);rows=['| 입력 정보 | 큰 오류 보정 MAE: 전체 학습 | 큰 오류 보정 MAE: 전용 학습 |','|---|---:|---:|']+[f'| {q["label"]} | {q["normal_all"]:.3f} | {q["normal_hard"]:.3f} |' for q in table];pointrows=['| 입력 정보 | 전체 점 오차px | 시드별 개선율% | 기준 대비 차이95%구간px |','|---|---:|---|---|']+[f'| {q["label"]} | {q["point"]:.3f} | '+', '.join(f'{x:+.2f}' for x in q['gains'])+' | '+ci_text(q['ci'])+' |' for q in table];conditions=[]
    for c,st in [('clean','all_detected'),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        key='check/'+c+'/'+st
        if key in s['results']:
            rr=s['results'][key];conditions.append(dict(condition=c+'/'+st,anchor=rr['anchor']['mean_px'],pred=rr['pred_segment_all']['mean_px'],GTline=rr['GT_line_all']['mean_px'],GTsegment=rr['GT_segment_all']['mean_px']))
    condrows=['| 조건 | 기존 점 | 실제 영상 선분 | 정확한 물리 직선 [진단] | 정확한 가시 구간 [진단] |','|---|---:|---:|---:|---:|']+[f'| {q["condition"]} | {q["anchor"]:.3f} | {q["pred"]:.3f} | {q["GTline"]:.3f} | {q["GTsegment"]:.3f} |' for q in conditions]
    body=['# '+TITLE,'','**영상 선분·끝점을 추가하는 이번 구조는 안정성 기준을 통과하지 못했다. 정확한 물리 직선의 위치·방향에는 부분 정보가 있었지만, 보이는 구간의 끝점을 추가한 이득은 확인되지 않았다.**','',
    '기존 V34 점과 V36 both의24개 보정 후보를 고정하고30개 작은 HGB 회귀 모델을 각60라운드 학습했다(총1800라운드, 감사 재학습3개 별도). 신경망 업데이트는0이다. 기존 전체/큰 오류 전용 예측기를 대조군으로 재사용했다.','',
    '## 실험 범위와 대응 정보','',
    '기존 합성436검출/600요청, 동일2CAD에서 학습12그룹·기준 선택용4그룹·재사용 확인8그룹을 유지했다. 아래 점 오차는 확인200관측 중130검출·1040점이며 검출 실패70개는 포함하지 않는다. 기존 상류 모델은 과거16그룹 학습 이력이 있다. 새 실사 평가나 수동GT 수정은 없으며 독립 최종 성능 검증이 아니다.','',
    '실제 변환된 영상에서 OpenCV LSD로8px 이상 선분17,952개를 뽑았다. 각 영상에서 검출 상자 주변의 긴128개 이내를 유지했다. 기존 Hough의 예측 역할·방향·거리·지원 중심으로 선분을 대응시켰다. 실제 입력에는 정답 선/점이 없다.','',
    '대조는 같은 영상 선분에 정답 물리 역할/가시 구간으로 대응을 알려주기, 정확한 물리 직선만 알려주기, 정확한 가시 구간 양 끝까지 알려주기, 그 가시 구간의 역할 슬롯을4개 이동시키기다. 정답 입력 대조는 모두 비배포 진단이다. GT 대응 대조는 역할뿐 아니라 가시성/선 위치로 후보를 찾으므로 순수 역할 분류 효과로 해석하지 않는다.','',
    '전체 점 학습과 큰 오류(기존 점GT오차≥10px) 전용 학습을 비교했다. 전용 모델의 실제 적용은 V39의 점+선 큰 오류 선택기와 기준을 그대로 사용한다. 새 선택기나 endpoint detector를 학습한 것은 아니다. 선분은 보정 입력에만 추가했고 보정 후보는 그대로라 새로운 접선 이동 후보가 생기지 않는다.','',
    '## 선분 대응은 얼마나 정확했나','',
    '확인셋에는 가시 물리 역할841개가 있다. 예측 대응은 가시 역할545~557개와 비가시 역할185~212개에 선분을 배정했다. 비가시 역할 배정은 목표 물리 경계의 가시성에 맞지 않는다는 뜻이며 영상 선분 자체가 가짜라는 뜻은 아니다. 정답 대응으로도 같은 LSD 풀에서 찾은 것은532개(63.3%)였다.','',
    '가시 역할에 배정된 선의 물리 경계 거리는 예측 대응2.91~3.07px, 정답 대응0.95px였다. 서로 다른 조건부 모집단이므로 이 숫자의 차이를 순수 대응 개선량으로 보지 않는다. 정답 대응이 배경/다른 경계 배정을 줄였어도 다음 보정 결과의 향상은 작았다.','',
    '![선분 대응의 범위](figures/matching.png)','',
    '## 필요한 보정 방향·크기가 더 잘 예측됐나','',*rows,'',
    '주대조인 전체 점 학습에서 실제 선분은15.442→15.385px로 작은 평균 개선이었다. 시드별−0.22/−0.99/+2.56%로 두 시드는 악화됐으며 등록한 매시드10% 개선 기준 미달이다. 평균 차이구간은[-.108,−.007]px로0을 제외하지만 이는 세 시드를 고정하고 장면 그룹을 재표집한 값이다. 시드별 일관성을 대신하지 않는다.','',
    '정확한 물리 직선은14.741px로 세 시드 모두2.29~7.83% 좋아졌다. 정확한 가시 구간·끝점은14.761px로 직선만의14.741px보다 추가 이득이 없었다(차이95%구간[-.030,+.076]px). 반면 역할을 바꾼 가시 구간15.456px보다 정확한 역할14.761px는 좋았다(차이[-.993,−.289]px). **역할과 물리 선 정보의 부분 효과는 있으나 모든 등록 효과크기 기준을 통과한 것은 아니다.**','',
    '![법선 보정량 예측](figures/direction.png)','',
    '## 실제 점과 가림·잘림 결과','',*pointrows,'',
    '실제 영상 선분을 추가하면 기존 점10.463→10.329px로 평균은 줄었지만 시드별−3.54/+0.48/+7.18%이며 차이구간이0을 포함한다. 기존 특징 보정기의10.290px보다도 좋지 않다. 큰 오류 전용 학습+기존 선택기는10.420px로 V39의10.484px보다 작게 줄었지만 세 시드 안정성은 실패했다.','',
    '정답 물리 직선10.070px도 기존 점 대비 차이구간[-.875,+.081]px로0을 포함한다. 정확한 선을 제공한 실험조차 현재 회귀기·보정 후보·선택 구조 전체를 안정화하지 못했다. 정확한 가시 구간10.143px는 실사용 수치가 아니다.','',*condrows,'',
    '가림은 시드별 방향이 달랐고, 화면 밖 잘림20/40에서는 실제 선분 추가로 세 시드 모두 악화됐다. 쉬운 조건의 개선을 가림·잘림 복원 성공으로 확장하지 않는다.','',
    '![전체 점 결과](figures/performance.png)','',
    '## 끝점 추가가 해법으로 확인되지 않은 이유','',
    '후속 기술 통계에서 큰 오류 점과 관련된 가시 구간 끝점은 목표 가상 꼭짓점에서 중앙54~56px 떨어져 있었다. 10px 이내인 역할은약20%,3px 이내는0~0.9%였다. **이 끝점은 물리 모서리 접합점 GT가 아니라 보이는 지원 구간의 양 끝이다.** 가림/잘림 때문에 중간에서 끝나거나 내부 구조 경계일 수 있다. V35의 캐시는 첫/마지막 가시 샘플을 잇기 때문에 중간 가림을 건너뛸 수도 있다.','',
    '정답 물리선이 있는 후보만 보아도 기존 보정 MAE 약15.61~18.62px 대비 정답 직선 입력12.68~16.79px였다. 그 정확한 선에서 기존 학습 상수offset을 더해 보정량을 직접 계산하는 진단은7.44~8.06px였다. 현재 회귀기가 알려준 제약을 충분히 사용하지 못하는 여지도 있다. 다만 이 계산은 정답 선과 과거16그룹offset을 사용하며 배포 가능한 새 모델 결과가 아니다.','',
    '정답offset으로 물리선의 관측 성분을 정확히 계산해도 고정 Hough 후보 방향에 대한 법선 목표 오차7.03~7.64px가 남았다. 물리선과 후보의 방향이 다르면 후보 법선 목표에 물리선의 미관측 접선 오차가 섞이기 때문이다. 두 선이10도 이내인 부분집합에서는3.13~4.52px였다. 이 분해는 수치적으로 독립 검증했다.','',
    '따라서 현재 확인된 것은 대응/가시성·선 정밀도의 부분 손실, 가시 끝점과 목표 꼭짓점의 차이, 물리 제약을 회귀와 고정 후보로 전달하는 손실이다. 하나만의 원인으로 확정하지 않는다. GT 정보가 있어도 현재 모델이 실패했다는 이유로 일반적인 점·선 융합 불가능성을 주장하지 않는다.','',
    '## 실제 영상 사례','',
    '고정seed1에서 조건부 중앙값으로 선택한 개선·악화·정답 지원과의 차이 사례다. 주황은 실제 영상에서 찾은 선분, 청록은 정답 대응으로 찾은 같은 영상 선분, 회색 점선 구간은 정답 가시 지원이다. 파랑은 기존 점, 빨강은 실제 보정, 보라는 정답 지원을 입력한 진단 보정, 초록은 목표GT다. 선분은 attention이 아니다. 개선/악화 사례는 선택된 역할에 실제 영상 선분이 있는 경우로 제한했지만 개별 개선만으로 그 선분의 인과적 기여를 증명하지는 않는다.']
    for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, point{q["point"]}','',q['rule'],'',f'점 오차 {q["base_px"]:.2f}→{q["actual_px"]:.2f}px. 정답 지원 진단 {q["GT_support_px"]:.2f}px. 선 표시는 {q["candidate_selection"]}에 대응하는 역할{q["role"]}이다.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 채택 판단과 다음 범위','',
    '이번 단순 선분/끝점 특징 추가 모듈은 채택하지 않는다. 실제 대응을 사용하는 안정적 성능 개선은 확보하지 못했다. 정확한 물리 직선의 역할·위치가 부분 정보를 준다는 근거는 남았지만, 가시 끝점 추가를 우선할 근거는 약해졌다.','',
    '후속 구조를 검토한다면 실제 접합점과 가림/화면 절단점을 구분하는 관측, 목표 점과 같은 역할·방향의 물리 제약, 그 제약을 계산으로 전달하는 경로를 구분해야 한다. 기존 V30/V35 투영이나 gate를 다시 붙이기만 하면 해결된다는 뜻은 아니다. 현재 결과로 새 장기학습을 정당화하지 않으며 새로운 장면 그룹과 실제 대응 신뢰도 검증이 필요하다. 이 후속 학습은 실행하지 않았다.','',
    '## 검증과 재현','',
    '30모델 예측·60점 배열의 모든 조건별 오차를 재계산했다. 비학습 영역 특징/GT 교란3재학습, 실제 대응의GT독립성, 정답 대응의가상점GT독립성, 영상 선분 재추출 일치, 끝점순서·좌표 이동/크기변환 불변성, 물리선잔차·접선분해 검사를 통과했다. 이전 완료 파일을 보존했다. 후속 감사 코드의 NumPy 고급 인덱싱 축 오류는 수정 후 재검사했으며 일차 모델/결과는 바꾸지 않았다.','',
    '재현: `scripts/research/pallet_segment_correspondence_v40/README.md`. 일차PROTOCOL/SUMMARY와 후속PROBE_PROTOCOL/PROBE는 분리했다. 확인셋은 재사용 소규모 합성이며 수동GT 검수나 실사 평가를 새로 한 것이 아니다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_segment_correspondence_v40/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',dict(table=table,conditions=conditions));write(RUN/'CASES.json',dict(rows=cases));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),probe_sha256=sha(RUN/'PROBE.json')));print('REPORT40 COMPLETE',flush=True)
if __name__=='__main__':main()
