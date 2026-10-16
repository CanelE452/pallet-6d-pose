from context31 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v31-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
import cv2,markdown
from eval31 import population
from features25 import actual_image
from model29 import INCIDENT
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

NAMES=dict(stock='기존 YOLO',base='V12 기준점',strong='V25 강한 점 복원',v18='V18 기존 융합',v30='V30 기존 적응형',joint_old='기존 데이터·공동 이득',joint_new='새 데이터·공동 이득',selector_old='기존 데이터·점 선택만',selector_new='새 데이터·점 선택만',split_old='기존 데이터·분리 보정',split_new='새 데이터·분리 선 보정',generic_new='새 데이터·일반 방향 보정',split_line_off='분리 모델·선 끄기',split_donor='분리 모델·다른 이미지 선',split_rank_only='분리 모델·하한 제한 제거',split_mean_only='분리 모델·평균 이득만 선택')
ORDER=('base','strong','v18','v30','joint_old','joint_new','selector_old','selector_new','split_old','split_new','generic_new')
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
def value(s,arm,c,t,split='real'):
    rows=s['results'][split+'/'+c]
    return float(np.mean([rows[f'{arm}_seed{i}']['frame_mean_px'] if t is None else rows[f'{arm}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))
def table(s,arms=ORDER,split='real'):
    cols=[(c,t) for c,t in COLS if split+'/'+c in s['results']]
    return ['| 방법 | '+' | '.join(LABELS[COLS.index(x)] for x in cols)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in arms]
def avg(s,c,t,key,arm='split_new',split='real'):
    vals=[s['diagnostics'][split+'/'+c][f'{arm}_seed{i}'][t].get(key) for i in SEEDS]
    vals=[v for v in vals if v is not None];return float(np.mean(vals)) if vals else None
def fmt(x):return '해당 없음' if x is None else f'{x:.3f}'

def plots(s):
    folder=RUN/'figures';folder.mkdir(exist_ok=True)
    matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in ORDER]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(ORDER)),ORDER);ax.set_title('V31 separated point choice and incremental correction gain\nReal297 reused DEV / 3-seed means (px), change vs base');fig.colorbar(im,ax=ax,label='Error change (px)');fig.savefig(folder/'performance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained');ax.axis('off');ax.set(xlim=(0,13),ylim=(0,4))
    boxes=[(.1,1.2,2.6,1.5,'Frozen V12 + V25\noriginal / restored points\nactual image features'),(3.2,1.2,2.4,1.5,'Stage 1: point choice\nbounded relative gain\nfreeze after training'),(6.2,1.2,3,1.5,'Stage 2: directional correction\nsigned amount + gain mean\nlower quantile + separate rank'),(9.8,1.2,3,1.5,'Use positive lower gain\nand positive rank\notherwise keep selected point')]
    for x,y,w,h,label in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc='#eaf2fb',ec='#567695'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for x,u in [(2.7,3.2),(5.6,6.2),(9.2,9.8)]:ax.annotate('',(u,1.95),(x,1.95),arrowprops=dict(arrowstyle='->'))
    ax.text(7.7,.55,'Frozen DHT normals / matched generic directions\nNo clean / occlusion / truncation state label',ha='center');ax.set_title('Separate parent benefit from incremental line benefit; keep path is exact');fig.savefig(folder/'architecture.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    cc=[('clean','all'),('corner20','hidden'),('crop20','outside'),('crop40','outside')];labels=[c for c,t in cc]
    for a in ('split_old','split_new','generic_new'):
        axes[0].plot(labels,[100*avg(s,c,t,'applied_fraction',a) for c,t in cc],marker='o',label=a)
        axes[1].plot(labels,[avg(s,c,t,'parent_regret_px',a) for c,t in cc],marker='o',label=a)
    axes[0].set(ylabel='Applied corrections (%)',title='Does the model stop or correct?');axes[1].set(ylabel='Error above GT best parent (px)',title='Point selection loss before correction')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(folder/'selection.png',dpi=160);plt.close(fig)
    return matrix

def cases():
    gallery=[]
    for key,c in [('case3','crop20'),('gain','crop20'),('harm','crop20'),('hidden','corner20')]:
        m,ts=population('real',c);g=np.array([t['points'] for t in ts]);v=np.array([t['valid'] for t in ts]);p=np.load(RUN/'predictions/real'/c/'split_new_seed1.npz');base=np.load(RUN/'predictions/real'/c/'base_seed1.npz');strong=np.load(RUN/'predictions/real'/c/'strong_seed1.npz')
        ef=np.linalg.norm(p['points']-g,axis=-1);eb=np.linalg.norm(base['points']-g,axis=-1);ea=np.linalg.norm(p['parent_points']-g,axis=-1);mask=v&p['detected'][:,None]&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts])
        if key=='case3':i=p['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');j=0;selection='Fixed user case3'
        else:
            change=ef-eb if key=='harm' else ea-ef
            i,j=np.unravel_index(np.where(mask,change,-np.inf).argmax(),mask.shape);selection='Max harm vs base' if key=='harm' else 'Max correction gain vs selected parent (ties possible)'
        aid=int(p['parent_id'][i,j]);allowed=np.where(p['allowed'][i,j]==1)[0];k=int(allowed[p['rank'][i,j,allowed].argmax()]);lineid=(k-8)%18;source=('guided','independent')[lineid//9];role=INCIDENT[j][(lineid%9)//3];mode=lineid%3
        bundle=np.load(V29.RUN/'predictions/real'/c/'bundle_seed1.npz');line=bundle[source+'_lines'][i,role,mode]
        raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];coords=np.array([g[i,j],base['points'][i,j],strong['points'][i,j],p['points'][i,j],p['candidate_points'][i,j,k]])
        view=(min(0,coords[:,0].min())-15,min(0,coords[:,1].min())-15,max(w,coords[:,0].max())+15,max(h,coords[:,1].max())+15)
        fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            rect=(coords[:,0].min()-25,coords[:,1].min()-25,coords[:,0].max()+25,coords[:,1].max()+25) if zoom else view
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e1e5e9');ax.add_patch(Rectangle((0,0),w,h,fill=False,ls='--',ec='black',lw=.7))
            seg=clipline(line,rect)
            if seg is not None:ax.plot(*np.array(seg).T,c='#159a9a',ls='--',label='Highest-rank DHT candidate')
            for name,pos,color,marker in [('GT',g[i,j],'#15a54b','x'),('Base',base['points'][i,j],'#2988ec','o'),('Restored',strong['points'][i,j],'#8547b3','s'),('Final',p['points'][i,j],'#ed8a00','D')]:ax.scatter(*pos,c=color,marker=marker,s=65,label=name,zorder=5)
            ax.annotate('',p['candidate_points'][i,j,k],p['parent_points'][i,j],arrowprops=dict(arrowstyle='->',color='#888888',ls='--'))
            if p['applied'][i,j]==1:ax.annotate('',p['points'][i,j],p['parent_points'][i,j],arrowprops=dict(arrowstyle='->',color='#e84639',lw=2))
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=8)
        applied=bool(p['applied'][i,j]);fig.suptitle(f'{key} / seed1 / point{j}: parent={aid}, applied={applied}, error={ef[i,j]:.2f}px\nDashed grey arrow = candidate, red arrow = applied; mean gain={p["gain"][i,j,k]:+.2f}, lower={p["lower"][i,j,k]:+.2f}px')
        fig.savefig(RUN/'figures'/f'{key}.png',dpi=150);plt.close(fig)
        per=[]
        for seed in SEEDS:
            rr=dict(seed=seed)
            for a in ('base','strong','selector_new','split_new','generic_new'):
                z=np.load(RUN/'predictions/real'/c/f'{a}_seed{seed}.npz');rr[a]=float(np.linalg.norm(z['points'][i,j]-g[i,j]))
            per.append(rr)
        gallery.append(dict(key=key,condition=c,id=m['rows'][i]['id'],point=int(j),selection=selection,applied=applied,rank_candidate=k,per_seed=per))
    return gallery

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');matrix=plots(s);gallery=cases();tr=read(RUN/'TRAIN_DIAGNOSIS.json')
    body=['# V31: 점 선택과 선의 추가 보정 이득을 분리한 학습','',f'**실제27,000업데이트·21개 체크포인트·3시드 완료. 안정성+선 추가 이득 동시 기준: {"통과" if s["accepted"] else "미통과"}.**','',
    '기존 데이터/새 원본과 공동/분리 구조를 교차 비교했다. 점 선택기만의 결과와 같은 수의 일반 방향 후보 대조를 함께 보고한다. 가림/잘림 상태로 방법을 고정 배정하지 않는다.','',
    '## 구현한 변화','', '![구조](figures/architecture.png)','',
    '1단계는 V12 기준점과 V25 복원점의 상대 오차 감소를 학습하고 점을 선택한다. 이 선택기를 고정한 뒤2단계에서 선 법선 방향 이동량, 선택한 점에 대한 순수 추가 이득 평균, 하위10% 분위수, 별도의 후보 순위를 학습한다. 하한이 양수이며 순위도 유지 후보보다 높은 후보만 적용한다. 그 외에는 선택한 점을 정확히 유지한다.','',
    '예상 평균/하한 이득은 실제 이동 거리의 ±범위로 제한했다. 하한은 평균 이하로 강제한다. 이는 수치적 모순을 방지하지만 보정의 정답성이나 실사 불확실성 보장을 뜻하지 않는다. 별도64합성 원본에서 하한의 경험적 포괄률을 진단했고 평가를 보고 임계값을 바꾸지 않았다.','',
    '분리 구조는 점 선택1000업데이트와 보정1000업데이트, 공동 구조는2000업데이트다. 기존/새 데이터마다 선택기를 한 번 학습해 일반 방향 대조에 그대로 공유했다. 실제 optimizer 업데이트는27,000이며 arm별2000을 중복 계산하면30,000이다. 공동·분리 비교는 학습 목표/일정/모듈 용량이 함께 바뀌는 패키지 대조이지 하나의 손실만의 인과 검증이 아니다.','',
    '일반 방향 대조는 각 부모마다18개 고정 각도와 동일15% box 대각선 최대 반경을 사용한다. 점 선택과 학습 예산·파라미터 수는 선 대조와 같다. 방향 후보 수는 맞췄지만 DHT의 실제 선까지 거리와 일반 방향의 전체 반경은 분포가 다르다. 모든 비교 차이를 선 의미 정보만의 효과로 단정하지 않는다.','',
    '## 데이터와 GT','',
    '새281원본: 학습153, 하한 진단64, 최종 합성 비교64. YOLO train과 기존 V6/V21/V23 요청 모집단의 pair_group 및 정확한 이미지 해시 중복을 제외했다. V25/V29의 학습 원본도 그 기존 모집단에 포함된다. 기존 데이터 대조는 V30의153원본 중 동일6조건만 사용했다. 원본 수는 같지만 검출된 관측 수는 다를 수 있다.','',
    '실제 변형 이미지를 YOLO와 고정 점/DHT 부모에 다시 입력했다. GT 주변에 예측 좌표만 옮기는 잔차 생성은 하지 않았다. 새로운 합성 GT와 renderer 투영 집합 최대 차이는 '+f'{read(DATA/"TARGET_AUDIT.json")["renderer_max_px"]:.6f}px'+ '였다. 집합 투영 검사는 모든 점의 의미 ID나 물리적 가시 경계를 독립 인증하지 않는다. 실사 수동 클릭 GT는 수정하지 않았다.','',
    '**새 원본도 과거 YOLO 검증 풀 출신이므로 전체 파이프라인이 한 번도 보지 않은 최종 평가라고 부르지 않는다.** 새 asset/scene 독립성이나 근접 중복까지 인증하지 않았다. 실사297장·9세션은 명시적으로 이어온 재사용DEV이며 정본140/논문/봉인 평가와 혼합하지 않는다.','',
    '## 동일 실사 비교','',
    '3시드 평균px. 원본은 검출 프레임 평균, 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 검출 실패 포함 정규화 지표·P90·중앙 가림20/40·분모는 SUMMARY.json에 포함한다.','',*table(s),'','![비교](figures/performance.png)','',
    '## 등록 기준 판정','',
    '각 시드 원본/잘림10/20/30에서 V12 대비 악화1% 이내, 가림/잘림40에서 V18 이득90% 이상 유지. 추가 선 이득은 잘림20에서 점 선택만 및 일반 방향 대조 각각 대비 모든 시드1% 이상과9세션 쌍대95%구간 상한<0를 요구했다.','',
    '| 방법 | 안정성 | 실패 조건 |','|---|---|---|']
    for a,gate in s['gates'].items():
        bad=['seed'+seed+': '+','.join(k for k,v in r.items() if not v) for seed,r in gate['per_seed'].items() if not all(r.values())]
        body.append('| '+NAMES[a]+' | '+('통과' if gate['pass_gate'] else '미통과')+' | '+'; '.join(bad)+' |')
    for a,r in s['line_evidence'].items():body+=['',f'선 분리 vs {NAMES[a]}: 시드별 개선율 '+str([round(v,3) for v in r['per_seed_gain_percent']])+', 세션95%차이 구간 '+str([round(v,4) for v in r['ci95']])+'px. 통과='+str(r['pass_gate'])+'. 다중 비교 보정 없는 재사용DEV 분석이다.']
    body+=['','## 보정이 실제로 실행됐는가','', '![선택](figures/selection.png)','', '| 대상 | 선 적용률 | 출발점 선택 손실px | 후보 이득 AUROC | 하한 포괄률 | 적용된 선 실제 이득px |','|---|---:|---:|---:|---:|---:|']
    for c,t in [('clean','all'),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:body.append('| '+c+' | '+' | '.join(fmt(avg(s,c,t,k)) for k in ('applied_fraction','parent_regret_px','candidate_gain_auc','lower_coverage','actual_applied_gain_px'))+' |')
    body+=['','선 적용률0은 성능 안정화의 증명이 아니다. 점 선택기 수준으로 돌아간 것이므로 어려운 조건의 기존 이득 보존과 따로 확인한다. 하한 포괄률 목표는0.9이지만 조건부/실사 보장이나 독립 표본의 신뢰구간으로 해석하지 않는다.','',
    '## 하한을 제거하면 달라지는가','',*table(s,('selector_new','split_new','split_rank_only','split_mean_only','split_donor')),'',
    '하한 제거/평균 이득만/다른 이미지 선은 고정 체크포인트의 사후 개입이다. 성능이 좋아 보이는 개입을 새 최종 모델로 선택하거나 실사에 맞춰 임계값을 조정하지 않았다.','',
    '## 학습집합에서부터 판별되는가','', '| 집합 | 부모 우열 AUROC | 후보 이득 AUROC | 하한 포괄률 | 보정 적용률 |','|---|---:|---:|---:|---:|']
    for dataset in ('old','new'):
        rr=[r['stats']['all'] for r in tr['rows'] if r['dataset']==dataset]
        body.append('| train '+dataset+' | '+' | '.join(fmt(float(np.mean([r[k] for r in rr]))) for k in ('parent_help_auc','candidate_gain_auc','lower_coverage','applied_fraction'))+' |')
    for split in ('fresh_calibration','fresh_check','real'):
        body.append('| '+split+' crop20 outside | '+' | '.join(fmt(avg(s,'crop20','outside',k,split=split)) for k in ('parent_help_auc','candidate_gain_auc','lower_coverage','applied_fraction'))+' |')
    body+=['','## 분리한 새 합성 원본 비교','',*table(s,('base','strong','joint_new','selector_new','split_new','generic_new'),split='fresh_check'),'','## 실제 사례와 모든 시드','', '회색은 영상 밖이다. 초록GT·파랑기준점·보라복원점·주황최종점. 점선 회색 화살표/청록 선은 후보이며 빨간 화살표만 실제 적용이다. 후보를 그린 것을 모델이 사용했다는 뜻으로 해석하지 않는다.']
    for case in gallery:
        body+=['',f'### {case["key"]}: {case["id"]}, point{case["point"]}','',case['selection']+f'; seed1 correction applied={case["applied"]}.','',f'![{case["key"]}](figures/{case["key"]}.png)','', '| seed | base | strong | point selector | separated lines | generic |','|---|---:|---:|---:|---:|---:|']
        for r in case['per_seed']:body.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[a]:.3f}' for a in ('base','strong','selector_new','split_new','generic_new'))+' |')
    body+=['','## 실행 범위와 남은 문제','',
    '완료: 부모 미학습 원본 확보·실제 재추론·원본 분리, 점 선택/선 추가 이득 분리 학습, 수치 범위 제약, 별도 분위수/순위 출력, 일반 방향 대조,3시드 전체 평가와 이전 파일 무변경 감사.','',
    '미완료: 전체 조건의 안정적 개선(판정표 참조), 실제 가시 경계/끝점·역할 GT 감독, 새로운 자연 가림/잘림 실사 봉인 평가. PCGrad는 이번에 실행하지 않았다. 하한이 지나치게 보수적이면 수치 범위 수정만으로 선택 정보가 좋아진 것이 아니다. 학습집합부터 후보 판별이 약한지, 별도 원본에서만 무너지는지를 위 진단과 함께 해석해야 한다.','',
    '재현: `python scripts/research/pallet_separated_gain_v31/run31.py`. 데이터 전환 중 캐시 경로 범위 오류는 seed2 시작 전 assert에서 중단됐으며 수정 후 재개했다. 완료된 seed1의 입력·학습식은 바꾸지 않았다. CACHE_SCOPE_FIX.json에 원래 코드 해시와 수정 범위를 보존했다.']
    interpretation=['## 이번 결과로 수정한 판단','',
    f'**현재 분리 구조는 선 보정을 적용하지 않는 쪽으로 수렴했다.** 실사 잘림20에서 점 선택만/분리 선/일반 방향이 모두 {value(s,"split_new","crop20","outside"):.3f}px다. V30의 {value(s,"v30","crop20","outside"):.3f}px 및 V12의 {value(s,"base","crop20","outside"):.3f}px보다 나쁘다. 큰 잘림은 강한 점 복원기의 이득을 거의 보존하지만 약한 잘림·가림까지 동시에 개선하지 못했다.','',
    f'하한 제한을 제거해도 잘림20은 {value(s,"split_rank_only","crop20","outside"):.3f}px, 평균 이득만 선택하면 {value(s,"split_mean_only","crop20","outside"):.3f}px다. 멈춤 기준만 완화하면 해결된다는 가설을 지지하지 않는다. 이는 고정 모델 추론 개입이며 하한 없이 재학습한 대조는 아니다.','',
    '점 선택 우열 AUROC는 신규 학습집합에서 평균0.643이지만 별도 합성 잘림20에서0.500, 실사0.429로 무너졌다. 조건별 모집단이 다르므로 숫자 차이 전체를 단일 도메인 효과로 해석하지 않지만, 별도 원본의 어려운 조건에서 선택 판단이 부족하다는 근거다. 선 후보 이득 AUROC는 신규 학습집합에서도0.534, 별도 합성 잘림20에서0.487이었다. 점 선택의 일반화 문제와 선 후보 판별 학습 자체의 약함을 분리해 봐야 한다.','',
    f'실사 잘림20에서 현재 선택한 출발점은 {avg(s,"crop20","outside","parent_px"):.3f}px, GT 최적 출발점은 {avg(s,"crop20","outside","best_parent_GT_px"):.3f}px다. 현재 이동량의 후보 중 GT로 최선을 골라도 {avg(s,"crop20","outside","oracle_candidates_GT_px"):.3f}px다. 이는 선택한 출발점을 고정한 진단이며, 다른 출발점까지 허용한 V30 oracle과 직접 비교하지 않는다. 선 적용 여부만 잘 고르는 것으로 이미 잘못 선택한 출발점을 충분히 복구할 후보가 없는 경우가 있다.','',
    '**해결된 것:** 예상 이득의 기하적 범위 위반, 점 이득/선 추가 이득 및 순위 출력 혼합, 이번 보정 학습에서 부모 학습 원본의 오차만 보던 데이터 범위. **해결되지 않은 것:** 좋은 점 선택과 유용한 선 이동량/후보를 별도 원본에서도 구분하는 능력. 수치 모순 제거와 데이터 분리만으로 안정화된다는 가설은 이번 예산에서 지지되지 않았다.','',
    '따라서 동일 특징으로 선택기·손실을 계속 바꾸는 실험의 우선순위를 낮춘다. 다음에는 실제 가시 경계/끝점과 점 역할 대응, 남아 있는 면의 전체 관측 근거가 후보 우열 판별을 개선하는지를 먼저 대조해야 한다. 이 정보 보강이 반드시 성공한다는 뜻은 아니며, 이번 실험은 새 물리적 선 GT를 만들어 학습하지 않았다.','']
    body=body[:6]+interpretation+body[6:]
    body='\n'.join(body)+'\n';(RUN/'REPORT_KO.md').write_text(body)
    NOTE.parent.mkdir(parents=True,exist_ok=True);NOTE.write_text(body.replace('](figures/','](../../../../data/pallet/results/pallet_separated_gain_v31/figures/'))
    html=markdown.markdown(body,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V31 점 선택과 선 이득 분리 실험</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1250px;padding:0 20px;color:#213044}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #cbd4dd;padding:8px}th{background:#eaf2fa}h1,h2,h3{line-height:1.35}code{background:#edf1f5}</style>'+html+'</html>')
    write(RUN/'FIGURE_DATA.json',dict(table=[dict(arm=a,values=matrix[i].tolist()) for i,a in enumerate(ORDER)]));write(RUN/'CASES.json',dict(rows=gallery));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json')));print('REPORT31 COMPLETE',flush=True)

if __name__=='__main__':main()
