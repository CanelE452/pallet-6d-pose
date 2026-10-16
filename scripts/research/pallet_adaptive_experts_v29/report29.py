"""Source-driven Korean report, expert comparisons, routing diagnosis and real cases."""
from context29 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v29-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
import markdown,cv2
from eval27 import population
from features25 import actual_image
from model29 import INCIDENT
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

NAMES={'stock':'기존 YOLO 점','base':'V12 점 기준','line_vector':'V18 기존 융합','line_role_utility':'V27 기존 융합','point':'새 점 분기','guided':'새 점 가이드 선','independent':'새 점 독립 선','point_router':'유지/점 선택기','adaptive':'새 적응형 선택기','no_state':'상태 감독 없는 선택기','predicted_state':'예측 상태로 고정 배정','oracle_expert_GT':'GT 최선 분기 (진단)','oracle_state_GT':'GT 상태로 고정 배정 (진단)'}
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
ORDER=tuple(NAMES)

def value(s,a,c,t=None,split='real'):
    rr=s['results'][split+'/'+c]
    return float(np.mean([rr[f'{a}_seed{i}']['frame_mean_px'] if t is None else rr[f'{a}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))

def dm(s,c,typ,key):return np.mean([v[typ][key] for v in s['diagnostics']['real/'+c].values()],axis=0)

def table(s,arms=ORDER,split='real'):
    cols=[(c,t) for c,t in COLS if split+'/'+c in s['results']];labels=[LABELS[COLS.index(x)] for x in cols]
    out=['| 방법 | '+' | '.join(labels)+' |','|---|'+'---:|'*len(cols)]
    out += ['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in arms]
    return out

def figures(s):
    fd=RUN/'figures';fd.mkdir(exist_ok=True);order=('stock','point','guided','independent','point_router','adaptive','oracle_expert_GT');matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in order]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-15,vmax=15,aspect='auto')
    for i in range(len(order)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(order)),order);ax.set_title('V29: separately trained experts and per-point adaptive selection\nReal297 reused DEV / 3-seed means; GT oracle is privileged');fig.colorbar(im,ax=ax,label='Error change vs stock YOLO (px)');fig.savefig(fd/'performance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for ax,(c,t) in zip(axes,[('corner20','hidden'),('crop20','outside'),('crop40','outside')]):
        real=dm(s,c,t,'route_counts');oracle=dm(s,c,t,'oracle_route_counts');x=np.arange(4)
        ax.bar(x-.18,100*real/real.sum(),.36,label='Actual router');ax.bar(x+.18,100*oracle/oracle.sum(),.36,label='GT best expert (diagnostic)',alpha=.6,hatch='//');ax.set_xticks(x,['Keep','Point','Guided','Independent'],rotation=18);ax.set_ylim(0,105);ax.set_title(c+' / state accuracy '+f'{100*dm(s,c,t,"state_accuracy"):.1f}%');ax.set_ylabel('Chosen fraction (%)');ax.legend(fontsize=8)
    fig.suptitle('Correctly recognizing truncation does not guarantee correct expert selection');fig.savefig(fd/'routing.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.2),layout='constrained');ax.axis('off');ax.set(xlim=(0,13),ylim=(0,4))
    boxes=[(.1,1.3,2.2,1.1,'Frozen YOLO\nimage features + points'),(3,2.7,3,.9,'Point correction expert\nstock point + local/global image'),(3,1.45,3,.9,'Guided DHT + line decoder\npoint maps guide voting'),(3,.2,3,.9,'Independent DHT + decoder\nNO point coordinates'),(7,1.1,2.3,1.5,'Per-point predicted state\n+ expert quality/risk\nkeep or select an expert'),(10,1.1,2.7,1.5,'8 output points\nincluding hidden / outside\nGT-free forward')]
    for x,y,w,h,text0 in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='#edf3fa',ec='#426f94'));ax.text(x+w/2,y+h/2,text0,ha='center',va='center',fontsize=9)
    for y in (3.1,1.9,.65):ax.annotate('',(3,y),(2.3,1.9),arrowprops=dict(arrowstyle='->'));ax.annotate('',(7,1.9),(6,y),arrowprops=dict(arrowstyle='->'))
    ax.annotate('',(10,1.9),(9.3,1.9),arrowprops=dict(arrowstyle='->'));ax.set_title('New expert fine-tuning on 80% synthetic originals; frozen-expert router training on disjoint20%\nDHT structural-line supervision -> point-free intersection decoders -> router (separate stages)');fig.savefig(fd/'architecture.png',dpi=170);plt.close(fig)
    return [dict(arm=a,values=[value(s,a,c,t) for c,t in COLS]) for a in ORDER]

def cases():
    specs=[('case3','crop20','fixed previous case3'),('clean_gain','clean','largest guided gain on a valid point'),('hidden_gain','corner20','largest guided gain on a hidden point'),('truncation_failure','crop40','largest independent loss on an outside point')];out=[]
    for key,c,selection in specs:
        m,ts=population('real',c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);b=np.load(RUN/'predictions/real'/c/'bundle_seed1.npz');a=np.load(RUN/'predictions/real'/c/'adaptive_seed1.npz');e=np.linalg.norm(b['candidates']-gt[:,:,None],axis=-1);mask=valid&b['detected'][:,None]
        if key=='case3':i=b['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');p=0
        else:
            if c=='corner20':mask&=np.array([t['hidden'] for t in ts])
            if c=='crop40':mask&=np.array([t['outside'] for t in ts])
            change=e[:,:,3]-e[:,:,0] if c=='crop40' else e[:,:,0]-e[:,:,2];index=np.where(mask,change,-np.inf).argmax();i,p=np.unravel_index(index,mask.shape)
        raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];coords=np.concatenate((gt[i,p][None],b['candidates'][i,p],a['points'][i,p][None]));rect=(min(0,coords[:,0].min())-20,min(0,coords[:,1].min())-20,max(w,coords[:,0].max())+20,max(h,coords[:,1].max())+20)
        fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained');errors={}
        for ax,arm,idx in zip(axes,('guided','independent'),(2,3)):
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#dfe3e8');ax.add_patch(Rectangle((0,0),w,h,fill=False,ec='black',ls='--',lw=.7))
            # Show all 12 top modes; emphasize the 3 roles incident on this corner.
            for role in range(12):
                ll=b[arm+'_lines'][i,role,0];color=['#e89d00','#ce53a3','#008f8c'][INCIDENT[p].index(role)] if role in INCIDENT[p] else '#d1d1d1'
                for region,ls,alpha in [(rect,'--',.5),((0,0,w,h),'-',.9)]:
                    seg=clipline(ll,region)
                    if seg is not None:ax.plot(*np.array(seg).T,color=color,ls=ls,lw=1.7 if role in INCIDENT[p] else .65,alpha=alpha)
            for label,pos,color,marker in [('GT',gt[i,p],'#00bf50','x'),('YOLO',b['candidates'][i,p,0],'#168bff','o'),(arm,b['candidates'][i,p,idx],'#ec7a00','s'),('Adaptive',a['points'][i,p],'#842baf','D')]:
                ax.scatter(*pos,c=color,marker=marker,s=65,label=label,zorder=5,edgecolors='white' if marker!='x' else None)
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.set_title(arm+f' line expert: {e[i,p,idx]:.2f}px\npoint ID{p}; colored lines = incident structural roles',fontsize=10);ax.legend(fontsize=8,loc='upper right')
        fig.suptitle(f'{key} / frame {m["rows"][i]["id"].split("/")[-1]} / seed1\nYOLO {e[i,p,0]:.2f}px; adaptive {np.linalg.norm(a["points"][i,p]-gt[i,p]):.2f}px | grey = outside model input',fontsize=11);fig.savefig(RUN/'figures'/f'{key}.png',dpi=170);plt.close(fig)
        perseed=[]
        for seed in SEEDS:
            sb=np.load(RUN/'predictions/real'/c/f'bundle_seed{seed}.npz');sa=np.load(RUN/'predictions/real'/c/f'adaptive_seed{seed}.npz');assert sb['ids'][i]==b['ids'][i]
            ee=np.linalg.norm(sb['candidates'][i,p]-gt[i,p],axis=-1)
            perseed.append(dict(seed=seed,**{name:float(ee[j]) for j,name in enumerate(('stock','point','guided','independent'))},adaptive=float(np.linalg.norm(sa['points'][i,p]-gt[i,p]))))
        out.append(dict(key=key,condition=c,seed=1,id=m['rows'][i]['id'],point=int(p),selection=selection,gt=gt[i,p].tolist(),errors={name:float(e[i,p,j]) for j,name in enumerate(('stock','point','guided','independent'))},adaptive_error=float(np.linalg.norm(a['points'][i,p]-gt[i,p])),per_seed=perseed,row=m['rows'][i]))
    return out

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');risk=read(RUN/'RISK_DIAGNOSIS.json');td=figures(s);gallery=cases();case=gallery[0]
    lines=['# V29: 점·점 가이드 선·점 독립 선의 적응형 선택 검증','',
    '**24모델·25,200업데이트·3시드 학습과 평가를 완료했다. 선 분기는 일부 조건에서 크게 좋아졌지만, 적응형 선택기는 안정성 기준을 통과하지 못했다.**','',
    '사용자의 세 조건 가설을 그대로 학습한 분기와 점별 선택기로 검증했다. 원본/가림에서는 새 선 분기가 강했고, 큰 잘림에서는 선 전용이 약했다. 따라서 clean→점 / occlusion→점 가이드 선 / truncation→독립 선이라는 고정 배정은 지지되지 않았다. 개별 사례의 보완 가능성과 전체 선택 실패는 동시에 관측됐다.','',
    '## 실제로 바꾼 구조','', '![학습한 구조](figures/architecture.png)','',
    '- 점 분기: frozen stock YOLO의 점·영상 특징에서 새 보정. 기존 V12/V25의 보정 출력은 입력하지 않았다. 따라서 기존 강한 가림 복원기를 그대로 보존한 설계는 아니다.','- 선 분기: 같은 V6 시작·같은 구조·같은 배치에서 DHT를 각각1200업데이트 재학습. guided는 예측 점 거리맵, independent는 항상 영맵이다. 독립 분기는 영상/박스/카메라/치수는 사용하지만 예측 점 좌표는 읽지 않는다.','- 선 디코더: 점 좌표 없이 구조적 역할에 맞는 선 교차 후보27개를 학습 선택한다. 비평행/범위 검사를 통과하지 못하면 최소노름 추정으로 대체하며, 이를 충분한 관측 근거로 간주하지 않는다. 두 분기의 디코더 초기값·파라미터 수·학습 순서가 같다.','- 선택기: 유지 / 새 점 / 점 가이드 선 / 점 독립 선 중 점별 예상 오차가 가장 작은 후보를 선택한다. 상태 감독 포함·상태 감독 없음·유지/점만 보는 대조를 같은 예산으로 학습했다.','',
    '합성764원본을 해시로80/20 분리했다. 새 전문가·디코더 미세학습 원본과 선택기 원본은 겹치지 않는다. 단, 시작 YOLO/V6의 과거 사전학습 원본과까지 완전히 독립인 것은 아니다. 실제 GT는 학습·체크포인트·추론 선택에 넣지 않았다.','',
    '**가시성 감독의 한계:** 상태 정답은 기존 인공 마스킹/화면 밖 표시다. 자연 가림이나 자기 가림의 완전한 가시성 GT가 아니다. 선 감독도 cuboid 구조선이며, 실제 보이는 외곽 경계/끝점의 새 물리 GT는 확보하지 않았다.','',
    '## 같은 실사 조건의 전체 비교','',
    '실사297장·9세션 재사용DEV의3시드 평균px. 원본은 검출 영상 평균, 가림은 가려진 유효점, 잘림은 화면 밖 유효점 평균이다. 원본의 모든 파렛트/점이 물리적으로 전부 보인다는 뜻은 아니다. 검출실패 포함 정규화 오차와 전체8조건은 SUMMARY.json에 보존했다.','',*table(s),'',
    '![분기와 선택기 성능](figures/performance.png)','',
    f'새 선 분기는 원본에서 점 분기 {value(s,"point","clean"):.2f}→{value(s,"guided","clean"):.2f}px, 코너 가림에서 {value(s,"point","corner20","hidden"):.2f}→{value(s,"guided","corner20","hidden"):.2f}px였다. 반면 독립 선의 잘림40 오차는 {value(s,"independent","crop40","outside"):.2f}px로 점 분기 {value(s,"point","crop40","outside"):.2f}px보다 컸다. **선이 무조건 잘림 전용이고 점이 원본 전용이라는 역할 배정은 맞지 않았다.**','',
    '중앙 가림도 코너 가림과 분리했다. 가려진 수동 점 수가 적은 조건은 해석 범위가 작다.','',
    '| 중앙 가림·숨겨진 점 | 검출 대상점/전체 대상점 | 점 | 점 가이드 선 | 점 독립 선 |','|---|---:|---:|---:|---:|']
    for c in ('occlude20','occlude40'):
        rr=s['results']['real/'+c]['stock_seed1']['strata']['hidden'];lines.append(f'| {c} | {rr["detected_points"]}/{rr["total_points"]} | {value(s,"point",c,"hidden"):.3f} | {value(s,"guided",c,"hidden"):.3f} | {value(s,"independent",c,"hidden"):.3f} |')
    lines+=['','## 사용자가 지목한3번 사례','',f'동일 frame000546·pointID0·crop20·seed1: 기존 점 **{case["errors"]["stock"]:.2f}px**, 새 점 **{case["errors"]["point"]:.2f}px**, 점 가이드 선 **{case["errors"]["guided"]:.2f}px**, **점 독립 선 {case["errors"]["independent"]:.2f}px**. 적응형 선택기는 {case["adaptive_error"]:.2f}px였다. 이 시드에서는 독립 선이 도움이 됐지만 선택기가 골라내지 못했다. V28의 추론 gate 우회와 달리 이번은 DHT와 선 전용 디코더를 실제로 재학습한 결과다.','',
    '같은 사례의 독립 선 오차는 **6.18 / 11.16 / 23.87px**로, 시드3에서는 기존 점보다 나빴다. 두 시드의 개선을 반복적으로 안정된 성공으로 해석하지 않는다.','',
    '| 같은 사례·시드 | 기존 YOLO | 새 점 | 점 가이드 선 | 점 독립 선 | 적응형 |','|---|---:|---:|---:|---:|---:|',
    *['| '+str(r['seed'])+' | '+' | '.join(f'{r[a]:.3f}' for a in ('stock','point','guided','independent','adaptive'))+' |' for r in case['per_seed']],
    '', '![3번 독립 선과 가이드 선 비교](figures/case3.png)','',
    '## 선택기가 왜 실패했는가','', '![실제 선택과 GT 최선 선택](figures/routing.png)','',
    f'1. **좋은 선을 거의 선택하지 않았다.** 코너 가림에서 두 선 중 하나가 유지/점 중 최선보다1px 이상 좋은 비율은 약68.69%인데 선 선택은 약0.12%였다. 잘림20도 GT 최선 분기는 {value(s,"oracle_expert_GT","crop20","outside"):.2f}px까지 가능하지만 실제 적응형은 {value(s,"adaptive","crop20","outside"):.2f}px다. GT 최선은 추론 성능이 아니라 선택 여지의 진단이다.',
    f'2. **상태 오분류도 있지만 그것만의 문제는 아니다.** 코너 가림 대상 상태 정확도는 {100*dm(s,"corner20","hidden","state_accuracy"):.1f}%, 잘림20 {100*dm(s,"crop20","outside","state_accuracy"):.1f}%, 잘림40 {100*dm(s,"crop40","outside","state_accuracy"):.1f}%였다. 큰 잘림을 맞게 인식해도 독립 선 자체가 더 나쁘다. GT 상태로 고정 배정해도 잘림40은 {value(s,"oracle_state_GT","crop40","outside"):.2f}px이므로 상태 분류 정확도만 높이는 것으로 해결되지 않는다.',
    '3. **합성과 실사에서 전문가 순위가 달랐다.** 선택기 학습 원본의 가림에서는 점 약13.58px, 가이드 선22.66px였다. 실사 코너 가림에서는 점16.16px, 가이드 선11.82px로 순위가 반대였다. 선택기는 실사에서도 선의 예상 오차를 더 크게 주었다. 이는 관측/오차 분포 차이라는 해석을 지지하지만, 카메라·재질·라벨 중 단독 원인을 확정하는 실험은 아니다.',
    '4. **학습집합에서도 선택 능력이 충분하지 않았다.** 선택기 원본의 전체 대상점에서도 선이 유지/점보다1px 이상 좋은 비율은 약21.80%인데 선 선택은 약1.49%였다. 따라서 실패를 실사 도메인 차이만으로 설명할 수도 없다. 현재 위험도 회귀·후보 순위 손실·입력 특징이 개별 성공을 구별하기에 부족하다는 가능성이 남는다.',
    '5. **강한 점 전문가 보존도 미완료다.** 새 점 분기는 기존 V18/V27의 큰 가림·잘림 이득을 재현하지 못했다. 적응형의 기존 대비 열세에는 선택 오류뿐 아니라 점 분기 자체의 약화도 포함된다. 기존 강한 점 복원기와 새 선 분기를 결합한 재학습은 이번 고정 대조에 포함하지 않았다.','',
    '## 점 힌트가 선을 좋게 한 것인가','',
    '| 조건 | guided | 같은 모델·점맵0 | 같은 모델·다른 영상 점맵 | 별도로 학습한 independent |','|---|---:|---:|---:|---:|']
    for c,t in COLS:lines.append('| '+LABELS[COLS.index((c,t))]+' | '+' | '.join(f'{value(s,a,c,t):.3f}' for a in ('guided','guided_zero','guided_donor','independent'))+' |')
    lines+=['','점맵을0으로 바꾸면 악화되는 조건이 있지만, 다른 영상의 정규화 점맵이 코너 가림에서 원래 점맵보다 좋았다. 따라서 선 분기 자체의 이득을 “정확한 자기 점 힌트 덕분”으로 확정하지 않는다. 영맵 개입은 학습 분포 밖 입력이며, donor도 비슷한 파렛트 형상을 가질 수 있다. 별도 학습한 독립 선과의 차이 및 실제 점맵 개입을 함께 본다.','',
    '## 큰 잘림의 선 분기 자체는 왜 약한가','',
    f'실사 잘림20은 독립 선 디코더 {value(s,"independent","crop20","outside"):.2f}px에 비해 같은 선 쌍 중 GT 최선 후보가 {dm(s,"crop20","outside","independent_pair_oracle_px"):.2f}px였다. 반면 잘림40은 디코더 {value(s,"independent","crop40","outside"):.2f}px, GT 최선 선 쌍도 {dm(s,"crop40","outside","independent_pair_oracle_px"):.2f}px다. 큰 잘림에는 전문가 선택 이전의 선 위치/역할/교차 후보 문제도 있다. 이 선 쌍 진단은 이산 후보 최선이며 학습한 후보 가중 평균의 엄밀한 하한은 아니다.','',
    '정답 구조선을 넣어 현재 Hough 표현 범위·교차각 조건으로 복원 가능한지도 검사했다. 인접3역할의 끝점 GT가 모두 있는 실사 잘림20 대상96점에서는100%, 잘림40 대상91점에서는78.0%만 충분한 정답 선 쌍이 있었다. 전체 잘림40 대상571점 중91점에 한정한 진단이며, 나머지를 실패로 계산하지 않았다. 별도 합성 확인의 큰 잘림도456점 중86.4%였다. 따라서 현재 ROI 기준 Hough 범위와 교차 조건에는 일부 큰 잘림을 표현하기 어려운 한계가 있다. 이는 선의 실제 가시성 검사가 아니며, 이 제한 하나가 모든 실패를 설명하지는 않는다.','',
    '## 별도 합성 원본에서의 일반화','',*table(s,('stock','point','guided','independent','adaptive','oracle_expert_GT'),'prospective'),'','합성 확인 원본에서는 선 디코더가 실사 원본보다 상대적으로 약했다. 이 순위 차이를 무시하고 합성 평균 성능만으로 실사 분기를 배정하면 좋은 선을 놓칠 수 있다. 상세 예상/실제 오차는 RISK_DIAGNOSIS.json에 조건과 시드별로 기록했다.','',
    '## 실제 이미지 추가 사례','',
    '다음은 개선/실패를 설명하기 위해 seed1 결과에서 해당 변화가 가장 큰 유효점을 골랐다. 대표 평균으로 해석하지 않으며 모든 원본은 정량 평가에 포함했다. 색선은 해당 코너에 인접한3개 구조 역할의 최상위 모드다. 디코더는 전체3모드씩을 사용하므로 그림의 최상위 선만 사용한 추론은 아니다.','']
    for case0 in gallery[1:]:lines+=['![실제 점과 선 비교](figures/'+case0['key']+'.png)','']
    lines+=['## 판정과 다음 해결 범위','',
    '**이번 고정 구조의 안정적 개선: 미통과. 점과 선의 보완 가능성: 확인. 세 상태만으로 정해진 전문가를 배정하는 방식: 현재 분기로는 불충분.**','',
    '각 시드 원본/잘림10·20·30의 V12 대비 악화1% 이내와, 가림/잘림40의 V18 이득90% 보존을 동시에 요구했다. 적응형은 이를 통과하지 못했다. 잘림20에서 유지/점 선택기 대비 세 시드 개선율도 '+', '.join(f'{x:.2f}%' for x in s['line_evidence']['per_seed_gain_percent'])+'로 모두 악화였다. 임계값이나 체크포인트를 결과에 맞춰 바꾸지 않았다.','',
    '다음에는 (1) 기존 강한 점 복원기를 보존하고, (2) 선 분기는 보이는 경계의 역할·길이·교차각·영상 지지를 명시적으로 평가하며, (3) 평균 예상 오차만 회귀하기보다 점 후보 대비 선 후보의 실제 우열을 구별하는 학습을 검증하는 것이 우선이다. 기존 V27에도 효용 학습이 있었으므로 손실 이름만 바꾸는 해결책이 아니라, 새 독립 선 후보와 원본 분리·관측 근거가 함께 달라져야 한다. 잘림40은 선택 이전에 선 후보의 복원 성능부터 개선해야 한다. 새 자연 가시 경계/끝점 감독과 독립 실사 관측은 아직 해결 범위 밖이다.','',
    '이번에는 위 추가 재학습까지 실행한 것이 아니다. 사용자 요청의 세 전문가 실제 학습·점별 적응형 대조·GT 최선 여지·조건별 평가·오류 원인 분석을 완료했다.','',
    '## 재현·GT·감사','',
    '합성 GT 유효점만 손실에 사용했고, 원본 ID 분리·동일 시작/배치·모든 학습 gradient 유한성·점 독립 분기의 좌표 변경 불변성24회·모든 결과 유한성·GT oracle·선택 결과의 후보 일치·지표 독립 재계산을 검사했다. 기존 V25/V27/V28 완료 파일951개는 그대로였다. 수동 실사 GT는 변경하지 않았으며 클릭 오차를 새로 측정한 실험은 아니다. 평가 포맷의 original_points 메타데이터 누락은 저장된 YOLO 결과로 보충했고 예측이나 GT는 바꾸지 않았다.','',
    '[사전 계약](PROTOCOL.json) · [학습 완료](TRAINING_COMPLETE.json) · [전체 수치](SUMMARY.json) · [위험도 진단](RISK_DIAGNOSIS.json) · [정답 선 표현 범위 검사](SUPPORT_DIAGNOSIS.json) · [검사](AUDIT.json) · [사례 좌표/출처](CASES.json)']
    body='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(body);NOTE.parent.mkdir(parents=True,exist_ok=True)
    nb=body.replace('(figures/','(../../../../data/pallet/results/pallet_adaptive_experts_v29/figures/')
    for name in ('PROTOCOL.json','TRAINING_COMPLETE.json','SUMMARY.json','RISK_DIAGNOSIS.json','SUPPORT_DIAGNOSIS.json','AUDIT.json','CASES.json'):nb=nb.replace('('+name+')','(../../../../data/pallet/results/pallet_adaptive_experts_v29/'+name+')')
    NOTE.write_text(nb);html=markdown.markdown(body,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V29 점과 선 적응형 실험 결과</title><style>body{font-family:Arial,sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;line-height:1.7;color:#203040}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccd7df;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf3fa}h2{margin-top:44px;border-bottom:2px solid #d8e5ef}a{color:#176aaf}strong{color:#123f63}</style><body>'+html+'</body></html>')
    write(RUN/'CASES.json',dict(cases=gallery));write(RUN/'FIGURE_DATA.json',dict(table=td,case3=gallery[0]));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),note_sha256=sha(NOTE),images=[str(p.relative_to(ROOT)) for p in sorted((RUN/'figures').glob('*.png'))]));print('REPORT29 COMPLETE',flush=True)

if __name__=='__main__':main()
