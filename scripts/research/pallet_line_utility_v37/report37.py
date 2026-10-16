from context37 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v37-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image

TITLE='V37 선이 점의 보정 방향·크기를 알려주는가'
HEADERS=['선 후보','물리 선 거리px','점 직접 보정px','점 정답 선택 상한px','실제 입력 선택px']


def clipped(n,r,bounds):
    x0,y0,x1,y1=bounds;pts=[]
    if abs(n[1])>1e-9:
        for x in (x0,x1):
            y=(r-n[0]*x)/n[1]
            if y0-1e-6<=y<=y1+1e-6:pts.append((x,y))
    if abs(n[0])>1e-9:
        for y in (y0,y1):
            x=(r-n[1]*y)/n[0]
            if x0-1e-6<=x<=x1+1e-6:pts.append((x,y))
    return np.array(pts[:2]) if len(pts)>=2 else None


def gallery(p):
    a=dict(np.load(RUN/'arrays/v34_both_seed1.npz'));b=dict(np.load(RUN/'arrays/v34_grid_seed1.npz'));g=p['g'];valid=(~p['train'])[:,None]&p['supported'];gt=g['gt'];err=lambda q:np.linalg.norm(q-gt,axis=-1);gain=err(a['anchor'])-err(a['selector_visual_geometry']);oracle=err(a['anchor'])-err(a['free_point_oracle']);linegain=b['physical_distance']-a['physical_distance'];pointloss=err(a['matched_top'])-err(b['matched_top']);rows=[]
    def median(mask,value):
        if not mask.any():return None
        v=np.median(value[mask]);return np.unravel_index(np.where(mask,np.abs(value-v),np.inf).argmin(),mask.shape)
    choices=[('line_better_point_worse',median(valid&(linegain>1)&(pointloss>1),pointloss),'Median point worsening among supported check points whose physical line improved >1px but matched projection worsened >1px.'),('missed_opportunity',median(valid&(oracle>3)&(gain<-1),oracle-gain),'Median lost gain among >3px oracle opportunities where actual selector worsens >1px.'),('selected_help',median(valid&(gain>1),gain),'Median improvement among supported check points helped >1px by the observable selector.'),('crop20',median(valid&g['outside']&(p['condition']=='crop20')[:,None],err(a['anchor'])),'Median anchor error among supported outside check crop20 points.')]
    for name,where,selection in choices:
        if where is None:continue
        i,j=map(int,where);row=p['rows'][i];im=actual_image(row);h,w=im.shape[:2];role=int(p['roles'][i,j]);ir=a['candidate_role'][j];sel=a['selected_selector_visual_geometry'][i,j];ok=a['selected_free_point_oracle'][i,j];cand=np.where(ir==role)[0];top=int(cand[a['candidate_score'][i,j,cand].argmax()]);coord=np.array([gt[i,j],a['anchor'][i,j],a['selector_visual_geometry'][i,j],a['free_point_oracle'][i,j],a['matched_top'][i,j]]);fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        lines=[(g['normal'][i,role],g['rho'][i,role],'GT physical role','#159e9c'),(a['candidate_normal'][i,j,top],a['candidate_rho'][i,j,top],'Role top score','#e58c29')]
        if name=='line_better_point_worse':
            oldtop=int(cand[b['candidate_score'][i,j,cand].argmax()]);lines.append((b['candidate_normal'][i,j,oldtop],b['candidate_rho'][i,j,oldtop],'Grid role top score','#555555'))
        if sel>=0:lines.append((a['candidate_normal'][i,j,sel],a['candidate_rho'][i,j,sel],'Observable selected','#dc6262'))
        if ok>=0:lines.append((a['candidate_normal'][i,j,ok],a['candidate_rho'][i,j,ok],'Point-GT oracle line','#b05dbf'))
        for ax,zoom in zip(axes,(False,True)):
            bounds=(coord[:,0].min()-20,coord[:,1].min()-20,coord[:,0].max()+20,coord[:,1].max()+20) if zoom else (min(0,coord[:,0].min())-10,min(0,coord[:,1].min())-10,max(w,coord[:,0].max())+10,max(h,coord[:,1].max())+10)
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e7ebef')
            for n,r,label,color in lines:
                segment=clipped(n,r,bounds)
                if segment is not None:ax.plot(*segment.T,c=color,ls='--',lw=1.3,label=label)
            ax.plot(*g['support'][i,role].T,c='#159e9c',lw=4,label='Visible GT support')
            for xy,label,color,marker in [(gt[i,j],'GT corner','#22a150','x'),(a['anchor'][i,j],'V34 anchor','#337fc4','o'),(a['selector_visual_geometry'][i,j],'Observable result','#dc6262','+'),(a['free_point_oracle'][i,j],'Point-GT oracle','#b05dbf','D')]:ax.scatter(*xy,c=color,marker=marker,s=65,label=label,zorder=5)
            if name=='line_better_point_worse':
                ax.scatter(*a['matched_top'][i,j],c='#e58c29',marker='^',s=70,label='Role top projection',zorder=5)
                ax.annotate('',xy=a['matched_top'][i,j],xytext=a['anchor'][i,j],arrowprops=dict(arrowstyle='->',color='#e58c29',lw=1.5))
            for xy,color in [(a['selector_visual_geometry'][i,j],'#dc6262'),(a['free_point_oracle'][i,j],'#b05dbf')]:ax.annotate('',xy=xy,xytext=a['anchor'][i,j],arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[3],bounds[1]);ax.set_aspect('equal');ax.legend(fontsize=6)
        info=dict(name=name,id=row['id'],point=j,matched_role=role,selected_candidate=int(sel),selected_role=int(ir[sel]) if sel>=0 else None,oracle_candidate=int(ok),oracle_role=int(ir[ok]) if ok>=0 else None,selection=selection,anchor_px=float(err(a['anchor'])[i,j]),selected_px=float(err(a['selector_visual_geometry'])[i,j]),oracle_px=float(err(a['free_point_oracle'])[i,j]),grid_line_px=float(b['physical_distance'][i,j]),both_line_px=float(a['physical_distance'][i,j]),grid_matched_point_px=float(err(b['matched_top'])[i,j]),both_matched_point_px=float(err(a['matched_top'])[i,j]))
        fig.suptitle(f'{name} / {row["id"]} / point{j}\nV34 {info["anchor_px"]:.2f}px -> selected {info["selected_px"]:.2f}px; point-GT oracle {info["oracle_px"]:.2f}px',fontsize=10);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);rows.append(info)
    return rows


def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS' and read(RUN/'EXTRA_AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');r=read(RUN/'RESIDUAL.json');o=read(RUN/'OBJECTIVES.json');p=population();(RUN/'figures').mkdir(exist_ok=True);supported=s['results']['check/all/supported']['methods'];allr=s['results']['check/all/all_detected']['methods'];table=[]
    for arm in ARMS:table.append([arm,s['diagnostics']['check/all/supported']['v34/'+arm]['physical_distance_px'],supported['v34/'+arm]['matched_top']['mean_px'],supported['v34/'+arm]['free_point_oracle']['mean_px'],supported['v34/'+arm]['selector_visual_geometry']['mean_px']])
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');labels=['V34 anchor','GT-role projection','GT-point oracle','Point/query selector','+ visual selector','+ geometry selector','+ both selector'];keys=['anchor','matched_top','free_point_oracle','selector_point','selector_visual','selector_geometry','selector_visual_geometry'];v=[allr['v34/both'][k]['mean_px'] for k in keys];ax.barh(labels,v,color=['#3b7ea1','#e7b567','#b294cc','#809aad','#809aad','#68a777','#68a777']);ax.axvline(v[0],c='gray',ls='--');ax.invert_yaxis();ax.set(xlabel='Euclidean point error (px), lower is better',title='V37 same 1040 detected check points / 3-seed means\nOracle and GT-role methods use unavailable targets; other selectors use train fits only')
    for i,x in enumerate(v):ax.text(x+.15,i,f'{x:.2f}',va='center')
    fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(4);axes[0].bar(x-.18,[q[1] for q in table],width=.36,label='Physical support distance');axes[0].bar(x+.18,[s['diagnostics']['check/all/supported']['v34/'+a]['normal_mae_px'] for a in ARMS],width=.36,label='Point normal correction MAE');axes[0].set(xticks=x,xticklabels=ARMS,ylabel='px',title='Physical line accuracy != point correction accuracy');axes[0].legend(fontsize=8)
    comps=r['components']['supported'];names=['local','angular','offset'];xx=np.arange(3)
    for k,arm in enumerate(('grid','both')):axes[1].bar(xx+(k-.5)*.35,[comps[arm]['mae_px'][n] for n in names],width=.35,label=arm)
    axes[1].set(xticks=xx,xticklabels=['Support position','Angle x distance','Physical offset'],ylabel='Component MAE (px)',title='Signed components are correlated; MAEs do not add');axes[1].legend();fig.savefig(RUN/'figures/residual.png',dpi=150);plt.close(fig)
    conditions=[('clean','all_detected'),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside')];matrix=[];ticks=[]
    for c,st in conditions:
        k='check/'+c+'/'+st
        if k not in s['results']:continue
        ticks.append(c+' / '+st+' / n='+str(s['results'][k]['points']));rr=s['results'][k]['methods']['v34/both'];matrix.append([rr[q]['mean_px'] for q in ('anchor','free_point_oracle','selector_visual_geometry')])
    mat=np.array(matrix);fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.imshow(mat,cmap='YlOrRd',aspect='auto');ax.set(xticks=np.arange(3),xticklabels=['V34 anchor','Point-GT oracle','Observable selector'],yticks=np.arange(len(ticks)),yticklabels=ticks,title='V36 both candidates; condition-specific point errors (px)')
    for (i,j),v0 in np.ndenumerate(mat):ax.text(j,i,f'{v0:.2f}',ha='center',va='center',color='black')
    fig.savefig(RUN/'figures/conditions.png',dpi=150);plt.close(fig);cases=gallery(p)
    primary=allr['v34/both'];oraclecomp=s['comparisons']['check/all/all_detected']['v34/both/free_point_oracle_vs_grid'];selcomp=s['comparisons']['check/all/all_detected']['v34/both/geometry_vs_visual'];normal=s['diagnostics']['check/all/supported']['v34/both'];cr=r['components']['outside']['both'];selection=[r['selection'][f'v34/both/{seed}/visual_geometry/check'] for seed in SEEDS]
    body=['# '+TITLE,'', '**선 후보에는 조건부 점 보정 여지가 남아 있지만, V36의 선 정확도 개선이 그 여지를 늘리지는 못했다. 정답 없이 고르는 작은 보정 모델도 기존 점을 넘지 못했다. 선 융합이 원천적으로 불가능하다는 결과는 아니다.**','',
    '기존 V36 선 후보 4종×3시드를 고정했다. 신경망 학습은 0회이고, 학습 구간으로 계산한 작은 회귀식은 일차 96개와 후속 목표 대조 48개, 총 144개다. 아래 수치는 **재사용 합성 확인셋**이며 앞서 보고한 실사297 수치와 다르다.','',
    '## 같은 점과 선 후보로 무엇을 확인했나','',
    'V12 기준점과 V34 점을 각각 고정하여 선 종류만 바꿨다. 주판정은 V34 기준과 V36 both 후보로 사전 고정했고, 다른 후보·기준 결과는 대조로 보존했다. V36에서 재학습한 점 결과는 별도 참고값이다.','',
    '1. 정답 역할을 알려주고 선 점수로 고른 뒤, 학습 구간에서 구한 물리선↔가상점 상수 offset을 사용해 법선 방향으로 보정했다.\n2. 같은 후보에서 정답 물리 경계에 가장 가까운 선과, 정답 점 오차를 가장 많이 줄이는 선을 따로 골랐다. **두 oracle은 다르며 실제 추론이 아니다.**\n3. 정답 역할·가시성 없이 점마다 관련 3역할×8후보를 놓고, 실제 입력만으로 이득을 예측해 하나를 선택하거나 원래 점을 유지했다.','',
    '직접 투영은 V35와 같은 전체 보정량을 사용한다. 실제 입력 선택과 그에 대응하는 점 oracle은 기존 구조의 범위에 맞춰 최대 이동을 상자 대각선의15%로 제한했다. 따라서 표의 직접 투영과 점 oracle은 이동 범위 및 정답 사용 조건이 다르다. 같은 범위의 격자↔개선 후보 비교를 주판정으로 사용했다.','',
    '## 1. 선을 더 잘 맞추면 점 보정 정보도 늘었나','',
    '확인200관측 중130검출,1040점 중 관련 정답 물리 경계가 보이는820점이 아래 공통 분모다. 각 값은3시드 평균이다. 물리 선 거리는 실제 보이는 구간에서 선까지의 거리이며 나머지는 점 오차다.','',
    '| '+' | '.join(HEADERS)+' |','|---|---:|---:|---:|---:|',*['| '+q[0]+' | '+' | '.join(f'{v:.3f}' for v in q[1:])+' |' for q in table],'',
    f'V34 원래 점은 지원820점에서 {supported["v34/both"]["anchor"]["mean_px"]:.3f}px다. 선 거리7.571→6.222px 개선과 달리, 선 직접 투영은14.788→14.828px였다. 점 정답으로 고르는 동일 후보 상한도7.054→7.075px로 좋아지지 않았다.','',
    f'전체 검출1040점에서도 원래 점 {primary["anchor"]["mean_px"]:.3f}px → 개선 후보의 점 oracle {primary["free_point_oracle"]["mean_px"]:.3f}px로 조건부 여지는 있다. 하지만 격자 후보 oracle {allr["v34/grid"]["free_point_oracle"]["mean_px"]:.3f}px 대비 차이는 {oraclecomp["difference"]["mean"]:+.3f}px,8그룹95%구간 {oraclecomp["difference"]["ci95"]}로 **추가 후보 이득 기준 실패**다. 점 정답으로 잘못된 역할의 선을 우연히 이용할 수도 있어 이 상한을 학습 가능성의 증명으로 보지 않는다.','',
    '참고로 같은 합성1040점에서 V36 기존 재학습 점은 격자9.854px/좌표만9.848px/both10.014px다. 이번의 V34 기준10.463px보다 좋은 기존 결과를 숨기지 않으며, 진단용 투영을 새 최선 모델로 부르지 않는다.','',
    '![후보별 오차](figures/residual.png)','',
    '## 2. 왜 선 거리 개선이 보정량 개선으로 연결되지 않았나','',
    f'개선 후보의 선 힌트로 예측한 법선 보정량 오차는 {normal["normal_mae_px"]:.3f}px였다. 같은 후보 방향에서 **아무 보정도 하지 않을 때** 보정량 오차는 {normal["no_correction_mae_px"]:.3f}px다. 정답 보정량이1px 이상인 표본의 방향 일치율은 {100*normal["sign_agreement_when_target_ge1px"]:.1f}%지만, 방향이 어느 정도 맞는 것만으로 크기까지 믿을 수는 없다. 이는 전체2D 방향 정확도가 아닌 단일 법선 부호다.','',
    f'후속 산술 분해에서 선 지지구간 중앙의 위치 오차 항은 {comps["both"]["mae_px"]["local"]:.3f}px, 방향 오차와 목표까지 거리로 생기는 항은 {comps["both"]["mae_px"]["angular"]:.3f}px, 물리선↔가상점 offset 항은 {comps["both"]["mae_px"]["offset"]:.3f}px였다. 이 항들은 부호와 상관관계가 있어 MAE를 더해 원인 비율로 해석하면 안 된다.','',
    f'선 거리가 개선된1369개 점×시드 중428개({100*r["physical_better_normal_worse"]["fraction"]:.1f}%)에서 점 보정량 오차는 커졌다. 물리 support 구간의 위치·각도 오차가 점 위치에서는 서로 상쇄되거나 증폭될 수 있다. 이번 support 정의에서 지원820점 모두 해당 구간 밖으로1px 넘게 연장해야 했고, 연장 거리 중앙값은25.93px였다. 선 위 가상 꼭짓점과 보이는 짧은 경계 구간을 같은 것으로 취급하면 안 된다.','',
    f'화면 밖 지원102점에서는 연장 거리 중앙값 {cr["extrapolation_median_px"]:.2f}px, 각도×거리 항 {cr["mae_px"]["angular"]:.2f}px였다. 물리 offset 항 {cr["mae_px"]["offset"]:.2f}px보다 크다. **잘림에서는 보이는 선 구간의 정확도뿐 아니라 목표 점 위치까지 연장한 제약의 불확실성을 다뤄야 한다**는 가설을 뒷받침한다. 전체 실패의 유일한 원인으로 확정한 것은 아니다.','',
    '## 3. 실제 입력으로 유리한 후보를 골랐나','',
    '학습16그룹에서만 ridge 이득 회귀를 계산하고, 확인8그룹으로 설정을 고르지 않았다. 모든 선택기는 같은24개 선 보정 후보와 no-op을 사용한다. 점/query 대조에도 역할·선 점수는 있고, visual은 같은 후보의 영상 특징을 추가한다. geometry는 선 법선·점까지의 거리·예정 보정량을 추가한다. 따라서 점/query 대조라는 이름은 **선 없이 점을 새로 예측하는 모델**이라는 뜻이 아니다.','',
    '| 선택 입력 | 전체1040점 오차px |','|---|---:|',*['| '+label+' | '+f'{primary[k]["mean_px"]:.3f}'+' |' for label,k in [('원래 V34 점 유지','anchor'),('점/query·역할·선 점수','selector_point'),('위 입력+선 영상 특징','selector_visual'),('위 기본 입력+선 기하','selector_geometry'),('영상+기하 모두','selector_visual_geometry')]],'',
    f'기하를 추가하면 영상 선택기보다 {abs(selcomp["difference"]["mean"]):.3f}px 좋아졌고 차이 구간은 {selcomp["difference"]["ci95"]}였다. **기하가 전혀 정보가 없다는 결과는 아니다.** 하지만 영상+기하 {primary["selector_visual_geometry"]["mean_px"]:.3f}px는 원래 점 {primary["anchor"]["mean_px"]:.3f}px보다 나빴다. 선택한 점 중1px 넘게 악화된 비율은 시드별 '+', '.join(f'{100*q["selected_harm_over1px_fraction"]:.1f}%' for q in selection)+'였다.','',
    '![전체 비교](figures/performance.png)','',
    '일차 선택기는 학습 데이터에서도 악화됐으므로 실패를 도메인 차이로만 설명하지 않는다. 작은 선형식의 표현력, 후보 이득의 불균형, 선택 후 과대평가, 손실과 최종 지표의 차이가 후보 원인이다. 그중 목표 차이와 보정 방향 표현을 다음의 짧은 추가 대조로 확인했다.','',
    '## 4. 보정 방향 직접 학습·픽셀 이득 목표 추가 대조','',
    '일차 결과를 보존하고 V34기준·격자/both·3시드에48개 회귀식을 추가했다. 첫 방식은 정답 법선 보정량을 직접 학습하고 `a(2t−a)`로 제안 보정a의 이득을 계산한다. 둘째는 최종 평가와 같은 픽셀 Euclidean 오차 감소량을 학습한다. 특징·후보·학습그룹·정규화·lambda·0기준 no-op은 고정했다.','',
    '| 후속 목표 / both 후보 | 점/query | +영상 | +기하 | 영상+기하 |','|---|---:|---:|---:|---:|']
    for target,label in [('normal','법선 보정량 직접 학습'),('pixel','픽셀 점 오차 이득 직접 학습')]:body.append('| '+label+' | '+' | '.join(f'{o["results"]["check/all/all_detected"][target+"/both/"+mode]["mean_px"]:.3f}' for mode in MODES)+' |')
    body+=['','후속 통과 여부: `'+json.dumps(o['gates'],ensure_ascii=False)+'`. 픽셀 이득 목표는 영상+기하 결과를 일차12.043→10.711px로 줄였지만, 원래 점10.463px를 넘지 못했다. 보정량 직접 학습은15.086px로 악화됐다. 따라서 손실 목표 차이가 일차 실패에 기여했지만 그것만 고쳐서 안정화되지는 않았다. 결과를 보고 최초 목표나 기준을 바꾸지 않았다. 이 대조도 작은 선형모델 범위이며 큰 비선형 아키텍처 전체의 가능성을 판정하지 않는다.','',
    '## 가림·잘림과 관측 가능한 방향','',
    '![조건별](figures/conditions.png)','',
    '가림·잘림 모두 point-GT oracle과 실제 선택 사이의 차이가 남는다. 조건 이름 자체를 신뢰도 대신 사용하지 않았다. 정답 관련 선 여러 개와 정답 offset을 모두 주면 지원820점에서3.555px까지 줄지만, 단일 정답 법선만으로는8.751px가 남았다(V34기준9.796px). 이는 관측 방향에 따라 접선 오차가 남는다는 진단이며 실제 예측 선의 성능이 아니다.','',
    '## 실제 합성 이미지 사례','',
    '모두 고정 seed1이며 선택 규칙을 함께 저장했다. 선은 후보 선의 연장이고 attention 시각화가 아니다. 굵은 청록색은 정답 가시 support의 양 끝을 이은 구간이며, 가림이 있으면 구간 전체가 연속 가시라는 뜻은 아니다. 분홍색은 점 정답으로 고른 비실사용 oracle이다. 회색은 이미지 바깥이며 새 영상 정보가 아니다. selected_help에서도 픽셀 오차가 줄었다는 사실만으로 선택한 선이 실제 경계에 정확히 맞는다고 판단하지 않는다.']
    for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, point{q["point"]}','',q['selection'],'',f'원래 점 {q["anchor_px"]:.2f}px → 실제 선택 {q["selected_px"]:.2f}px, 점 정답 선택 {q["oracle_px"]:.2f}px. 물리선 거리는 grid {q["grid_line_px"]:.2f} → both {q["both_line_px"]:.2f}px이고 해당 역할 직접 투영 점 오차는 {q["grid_matched_point_px"]:.2f} → {q["both_matched_point_px"]:.2f}px다.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 이번 판단과 다음 설계 조건','',
    '**이번 개선 선 후보를 기존 자유 좌표 head에 다시 넣어 장기 학습하는 근거는 확보하지 못했다. 운영 모델을 교체하지 않는다.** 선 거리 개선이 점 위치에서의 잔차 오차를 충분히 줄이지 못했고, 작은 실제 입력 선택기도 안정성 기준을 통과하지 못했다.','',
    '다음 설계에서 우선 확인할 대상은 물리 구간 전체의 분류 점수보다 **각 목표 점에서의 선 제약 잔차와 그 불확실성**이다. 정답 역할을 아는 합성에서 먼저, 가시 support의 길이·범위와 θ/ρ 오차를 목표 점까지 전달해 법선 보정량 오차를 제대로 예측하는지 검사할 수 있다. 예측 uncertainty가 학습그룹 밖에서도 보정 오차와 맞고, 원래 점보다 불확실한 제약을 거절할 수 있어야 한다. 이 uncertainty 학습·검증은 이번에 실행하지 않았다.','',
    '그 검사가 성립한 뒤에야 역할/offset이 맞는 여러 비평행 제약을 제한된 복원 경로로 넣는 대조가 타당하다. V30의 단순 법선 투영이나 기존 gain gate를 새 아이디어로 반복하는 제안이 아니다. 목적은 선 분류가 좋아졌다는 이유로 점을 무조건 움직이지 않고, 실제 점 위치에서 신뢰할 수 있는 제약이 추가됐음을 먼저 보이는 것이다.','',
    'Gradient surgery는 실행하지 않았다. 이번의 짧은 경계→먼 목표 연장 오차, 접선 미관측, 잘못된 이득 선택은 경사 충돌만의 문제로 볼 수 없다. PCGrad가 이 문제들을 해결한다는 증거는 이번 진단에서 나오지 않았다.','',
    '## 범위·감사·재현','',
    '기존 합성600관측/436검출, 학습306검출·16그룹, 확인130검출·8그룹이다. 확인 요청200 중 검출 실패70은 조건부px에서 제외되며 이번 모듈이 복구하지 않았다. 같은2CAD와 재사용 확인셋으로 독립 실사·미지자산 최종 성능을 주장하지 않는다. upstream 모델은 이미 학습16그룹을 사용했으므로 학습 잔차는 낙관적일 수 있다. 수동 실사GT를 읽어 새로 평가하거나 수정하지 않았다.','',
    '일차384개 점 예측 배열을 독립 재계산했고96개 선택기의 저장 계수로 선택을 재현했다. 선 거리 수치는V36과 일치했고, 선 표현 부호 반전·법선/접선 항등식·check GT/특징 교란 시 train fit 불변·기존 결과 해시 보존을 검사했다. V35 미사용 ridge계수의1.36e-15 재계산 차이는1e-12 허용오차로 비교했고 실제 쓰는 median은 정확히 같았다. 정답support 내부에 있는 대상이0개인 층은 분모0으로 명시했다. 후속48개 선택기도 계수·선택·수치를 재현했고,8개 추가 fit에서 확인GT/특징 교란에도 같은 계수가 나왔다. FP32 법선의 미세한 단위길이 차이 때문에 이상적 이득 항등식은 최대5.85e-9 정규화 제곱 단위 차이가 있었으며, 저장된 실제 법선 길이를 포함한 식은6.1e-16 이내로 일치했다. 후속 보정목표/잔차 분해는 별도 프로토콜과 감사로 보존한다.','',
    '재현: `scripts/research/pallet_line_utility_v37/README.md`. 일차 `PROTOCOL/SUMMARY`, 산술 분해 `RESIDUAL_PROTOCOL/RESIDUAL`, 후속48식 `OBJECTIVE_PROTOCOL/OBJECTIVES`. 총144회 ridge fit, 신경망 optimizer 업데이트0.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_line_utility_v37/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}code{background:#eef3f8;overflow-wrap:anywhere}</style>'+html+'</html>');write(RUN/'CASES.json',dict(rows=cases));write(RUN/'FIGURE_DATA.json',dict(table=table,headers=HEADERS));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),residual_sha256=sha(RUN/'RESIDUAL.json'),objectives_sha256=sha(RUN/'OBJECTIVES.json')));print('REPORT37 COMPLETE',flush=True)

if __name__=='__main__':main()
