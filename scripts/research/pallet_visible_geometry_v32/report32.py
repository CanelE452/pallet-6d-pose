from context32 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v32-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
import cv2,markdown
from eval32 import population,render_data
from features25 import actual_image
from plpose_local.geometry import EDGES
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

TITLE='V32 실제 경계 감독과 점·선 융합'
NAMES=dict(base='V12 기준점',strong='V25 강한 점 복원',v18='V18 기존 융합',v30='V30 적응형',v31='V31 분리 선택',point='새 점 대조',latent_hough='점 손실만 Hough',virtual_hough='가상 경계 감독+융합',visible_hough='실제 경계 감독+융합',visible_aux_only='실제 경계 감독만*',tokens_off='두 특징 끄기*',hough_off='Hough만 끄기*',endpoints_off='끝점만 끄기*',donor='인접 프레임 특징*',cross_session_donor='다른 세션 특징*')
ORDER=('base','strong','v18','v30','v31','point','latent_hough','virtual_hough','visible_hough','visible_aux_only')
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
def value(s,a,c,t,split='real'):
    rr=s['results'][split+'/'+c]
    return float(np.mean([rr[f'{a}_seed{i}']['frame_mean_px'] if t is None else rr[f'{a}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))
def table(s,arms=ORDER,split='real'):
    cols=[(c,t) for c,t in COLS if split+'/'+c in s['results']]
    return ['| 방법 | '+' | '.join(LABELS[COLS.index(x)] for x in cols)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in arms]
def plots(s):
    (RUN/'figures').mkdir(exist_ok=True)
    matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in ORDER]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(ORDER)),ORDER);ax.set_title('V32 visible CAD creases and endpoints / reused real DEV\n3-seed mean px, parentheses: change vs V12 base');fig.colorbar(im,ax=ax,label='Error change (px)');fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained');ax.axis('off');ax.set(xlim=(0,13),ylim=(0,5))
    boxes=[(.1,2.2,2.8,1.4,'Frozen image / pixel features\nBase + restored points\nKeep both until final output'),(3.5,2.2,2.5,1.4,'Trainable spatial trunk\n24 role probability maps\n12 edges + 12 endpoints'),(6.6,2.2,2.7,1.4,'Bilinear Hough + pooling\nSpatial endpoint pooling\n64 + 64 feature tokens'),(10,2.2,2.8,1.4,'Point-specific prediction\nSigned point mixture\nBounded 2D residual')]
    for x,y,w,h,label in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc='#e8f1fa',ec='#5b7899'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for a,b in [(2.9,3.5),(6,6.6),(9.3,10)]:ax.annotate('',(b,2.9),(a,2.9),arrowprops=dict(arrowstyle='->'))
    ax.annotate('',(11.4,2.15),(4.8,1.5),arrowprops=dict(arrowstyle='->'));ax.text(6.8,1.2,'Local patches at BOTH point estimates + global visual features',ha='center',fontsize=10)
    ax.text(6.5,.3,'Physical supervision only during training. No GT / condition label in forward.\nAux-only control zeros both tokens but retains identical physical supervision.',ha='center',fontsize=10)
    ax.set_title('V32 tested architecture (not an accepted stable replacement)');fig.savefig(RUN/'figures/architecture.png',dpi=160);plt.close(fig)
    return matrix

def overlay(ax,im,row,prob,title,end=None):
    h,w=im.shape[:2];ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));a,u,b,v=row['box']
    rgba=np.zeros((*prob.shape,4));rgba[...,:3]=[.05,1,.3];rgba[...,3]=.62*np.clip(prob,0,1)
    ax.imshow(rgba,extent=(a,b,v,u),interpolation='nearest')
    if end is not None:
        y,x=np.where(end>.5);ax.scatter(a+x/47*(b-a),u+y/47*(v-u),s=7,c='#f23848',alpha=.65)
    ax.set_xlim(max(0,a-15),min(w,b+15));ax.set_ylim(min(h,v+25),max(0,u-25));ax.set_title(title,fontsize=10);ax.set_aspect('equal')

def labels_gallery():
    cases=[];datasets={c:render_data(1,'check',c) for c in ('clean','physical_panel','crop20')}
    groups=set.intersection(*[set(r['camera_group'] for r in d[0]['rows'] if r['detected']) for d in datasets.values()])
    chosen=sorted(groups,key=lambda x:(not x.startswith('a1_'),not x.endswith('_v2'),x))[0]
    for c in ('clean','physical_panel','crop20'):
        m,O,S,ts=datasets[c];detrows=[r for r in m['rows'] if r['detected']]
        row=next(r for r in detrows if r['camera_group']==chosen);i=detrows.index(row);full=m['rows'].index(row);p=np.load(RUN/'predictions/render_check'/c/'visible_hough_seed1.npz')
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');im=actual_image(row)
        for ax,target,title in zip(axes,(O['virtual_target'][i].numpy(),O['physical_target'][i].numpy(),p['role_maps'][full].astype(float)),('Virtual cuboid target','Visible CAD crease target','Learned physical prediction')):
            overlay(ax,im,row,target[:12].max(0),title,target[12:].max(0))
        fig.suptitle(c+' / '+row['id']+' / green: edge map; red: endpoint map > 0.5\nPhysical endpoints are CAD chain ends, not virtual cuboid corners or crop cuts',fontsize=10)
        fig.savefig(RUN/'figures'/f'labels_{c}.png',dpi=160);plt.close(fig);cases.append(dict(condition=c,id=row['id']))
    return cases

def cases():
    gallery=[]
    for key,c in [('case3','crop20'),('gain','crop20'),('harm','crop20'),('hidden','corner20')]:
        m,ts=population('real',c);g=np.array([t['points'] for t in ts]);v=np.array([t['valid'] for t in ts]);p=np.load(RUN/'predictions/real'/c/'visible_hough_seed1.npz');base=np.load(RUN/'predictions/real'/c/'base_seed1.npz');point=np.load(RUN/'predictions/real'/c/'point_seed1.npz');strong=np.load(RUN/'predictions/real'/c/'strong_seed1.npz')
        ef=np.linalg.norm(p['points']-g,axis=-1);eb=np.linalg.norm(base['points']-g,axis=-1);ep=np.linalg.norm(point['points']-g,axis=-1);mask=v&p['detected'][:,None]&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts])
        if key=='case3':i=p['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');j=0;selection='User-fixed case3'
        else:
            change=ef-eb if key=='harm' else ep-ef;i,j=np.unravel_index(np.where(mask,change,-np.inf).argmax(),mask.shape);selection='Max harm vs base (seed1)' if key=='harm' else 'Max gain vs matched point (seed1)'
        raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];coords=np.array([g[i,j],base['points'][i,j],strong['points'][i,j],point['points'][i,j],p['points'][i,j]])
        view=(min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10)
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,zoom in zip(axes[:2],(False,True)):
            rect=(coords[:,0].min()-25,coords[:,1].min()-25,coords[:,0].max()+25,coords[:,1].max()+25) if zoom else view
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e1e5e9');ax.add_patch(Rectangle((0,0),w,h,fill=False,ls='--',ec='black',lw=.7))
            for role,pair in enumerate(EDGES):
                if j not in pair:continue
                seg=clipline(p['lines'][i,role],rect)
                if seg is not None:ax.plot(*np.array(seg).T,c='#12a4a4',ls='--',alpha=.75,lw=1)
            for name,pos,color,marker in [('GT',g[i,j],'#15a54b','x'),('Base',base['points'][i,j],'#2988ec','o'),('Restored',strong['points'][i,j],'#8547b3','s'),('Point ctrl',point['points'][i,j],'#232323','+'),('Final',p['points'][i,j],'#ed8a00','D')]:ax.scatter(*pos,c=color,marker=marker,s=55,label=name,zorder=5)
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=7)
        maps=p['role_maps'][i].astype(float);overlay(axes[2],raw,m['rows'][i],maps[:12].max(0),'Predicted roles (not attribution)',maps[12:].max(0))
        fig.suptitle(f'{key} / seed1 / point{j}: base={eb[i,j]:.2f}, point ctrl={ep[i,j]:.2f}, final={ef[i,j]:.2f}px\nTeal dashed lines: top Hough bin for incident cuboid roles; candidates, not proof of use',fontsize=10);fig.savefig(RUN/'figures'/f'{key}.png',dpi=150);plt.close(fig)
        per=[]
        for seed in SEEDS:
            rr=dict(seed=seed)
            for a in ('base','strong','point','visible_hough','visible_aux_only'):
                z=np.load(RUN/'predictions/real'/c/f'{a}_seed{seed}.npz');rr[a]=float(np.linalg.norm(z['points'][i,j]-g[i,j]))
            per.append(rr)
        gallery.append(dict(key=key,condition=c,id=m['rows'][i]['id'],point=int(j),selection=selection,per_seed=per))
    return gallery

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');matrix=plots(s);lc=labels_gallery();gallery=cases();tr=read(RUN/'TRAIN_DIAGNOSIS.json');roles=read(RUN/'ROLE_METRICS.json')['rows'];aux=read(RUN/'AUX_COMPARISONS.json')
    body=['# '+TITLE,'','**15모델·3시드·실제30,000업데이트 완료. 물리적 경계 감독의 부분 효과는 확인했지만, 전체 조건의 안정성 기준은 미통과했다.**','',
    '잘림20에서 새 점 대조16.331→실제 경계 융합14.343px로 세 시드 모두 개선했다. 가상 경계 감독17.030px보다도 좋았다. 하지만 기존 V12 기준점12.372px와 V30의13.385px에는 못 미쳤고, 코너 가림에서 기존 강한 점 복원의 이득을 보존하지 못했다. 따라서 운영 모델을 교체할 결과는 아니다.','',
    '추가 대조: 같은 실제 경계 감독을 주되 선·끝점 특징을 전달하지 않으면15.297px였다. 융합의 추가 이득0.954px는 단순한 보조 감독 효과만으로 설명되지 않는다. 반면 Hough만 끄기14.359px, 끝점만 끄기15.537px여서 현재 모델은 Hough보다 끝점 경로에 더 의존하는 것으로 보인다. 이는 고정 모델의 개입 분석이며, 각 경로를 따로 재학습한 완전한 인과 분해는 아니다.','',
    '## 동일 실사 비교','',
    '3시드 평균px, 작을수록 좋다. 원본은 검출 프레임 평균, 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 실사297장·9세션은 반복 사용한 DEV다. 정본140/논문/봉인 평가가 아니다. 별표는 첫12모델 결과를 본 뒤 추가한 탐색 대조다.','',*table(s),'','![성능](figures/performance.png)','',
    '## 구현한 구조','', '![구조](figures/architecture.png)','',
    'YOLO와 기존 V12/V25 점 부모는 고정하고, 영상 특징과 실제 gray/Sobel ROI를 받아48×48 공간 특징을 학습했다.12개 경계 역할과12개 끝점 역할 확률맵을 만든다. 선 맵은 미분 가능한24각도×48ρ Hough 투표와 pooling을 거치고, 끝점 맵은 공간 pooling을 거쳐 각각64차원 특징이 된다. Hough는 인접ρ칸에 선형 분배해 칸 사이 정보의 학습 경로를 유지한다. 원 논문의 전체 네트워크를 그대로 재현한 모델은 아니다.','',
    '두 점 주변 패치·전역 영상·선/끝점 특징을 함께 사용해 최종점 = 기준점 + tanh(mix)×(복원점−기준점) + 잔차를 출력한다. 잔차는 box 대각선15% 이내의2D벡터다. 원래 점/복원점을 미리 하나로 버리지 않지만, 실제 선택이 항상 정확하다는 보장은 없다. 법선 방향만으로 강제하지 않으며 접선 방향도 끝점·영상 근거로 학습할 수 있다. 상태 이름이나 GT는 추론 입력에 없다.','',
    '첫4대조는 같은 초기값·배치·모델 파라미터 수·2000업데이트를 사용한다. 점 대조는 선/끝점 특징0, 보조손실 없음. latent는 점 손실만, virtual은 가상 cuboid 경계/끝점 감독, visible은 실제 가시 CAD 경계/끝점 감독이다. 물리 감독만 대조는 visible과 같은 손실을 쓰되 두 특징을0으로 둔다. 사용하지 않는 분기 파라미터도 있어 동일 총 파라미터가 동일 유효 용량을 뜻하지는 않는다. 여기서 점 대조의 고정 부모에는 과거 선 학습의 영향이 있을 수 있다.','',
    '## GT 정의와 검증','',
    '기존2개 USD 자산에서 삼각분할 정점을 용접하고,35도 이상 꺾인 모서리/열린 경계를 추출했다. 일직선으로 닿은 조각을 합쳐 삼각형 분할 끝점이 라벨이 되지 않게 했다. 축 정렬 직선·길이15mm 이상만 유지한다. 평행한 cuboid 경계에서3D대각선4% 이내인 물리 선에만 해당 역할을 부여한다. 따라서 정확히 같은 가상 모서리나 사람의 의미 라벨이 아니라 기하 근접 관계다. 내부 리브·단차 일부가 포함되고 곡선/미배정 경계는 빠진다.','',
    '물체 단독 depth와 가림 물체 포함 scene depth 양쪽에 일치하는 표본만 가시 선으로 썼다. 허용 오차는3mm+깊이0.1%, 중심/상하좌우5픽셀이다. 끝점은 물리적 직선 체인의 끝이며 crop 절단점이나 가림 경계에 새 끝점을 만들지 않는다. 수평/수직 교차와 표면 법선 등 모든 물리 의미를 독립 인증한 GT는 아니다.','',
    '검사: 용접 cube/세분화 cube 모두12체인, 별도 OpenCV 투영 차이2.27e-13px, crop에서 끝점이 새로 생기지 않음, 물리 가림 가시 표본이 원본의 부분집합임을 확인했다. 실사 수동 클릭의 valid 여부·원본 파일 해시·crop 좌표 변환을 감사했고 수정하지 않았다. 점 평가 GT는 기존 가상 cuboid 모서리이고 실제 CAD 체인 끝점과 동일하지 않다. 이 대응 차이는 여전히 중요한 한계다.','',
    '아래 초록은 경계맵, 빨강은 끝점맵이다. 오른쪽은 학습한 예측이며 GT나 모델 기여도 그림이 아니다. 분리한 렌더 확인 집합의 같은 원본/가림/잘림을 보여준다.']
    for c in lc:body+=['',f'![{c["condition"]}](figures/labels_{c["condition"]}.png)']
    body+=['','## 데이터와 예산','',
    '2자산×12외관/조명 그룹×5시점×물리 가림 유무 =240 RGB를 렌더했다. 원본마다 clean/crop20/crop40/corner20, 물리 가림에는 physical_panel을 만들어600관측을 요청했다. YOLO 검출436/600만 후속 특징을 가진다. 자산별 외관 그룹0~7학습,8~11확인으로 같은 카메라/변형을 함께 묶었다. 학습16그룹400요청, 확인8그룹200요청이다. 두 자산은 양쪽에 공유하므로 미지 자산/자연 실사 일반화 검증이 아니다.','',
    '배치마다 V31의 부모 학습과 분리한153개 합성 원본 관측8개와 새 렌더 관측8개를 섞었다. 이전 관측에는 선 라벨을 만들지 않고 렌더만 보조 감독했다. 손실은 대각선 정규화 점 거리 +0.05×양/음 픽셀 균형 BCE, FP32 AdamW3e-4다. 별도128원본 prospective와 V31 fresh_check64도 평가했다. 이 자료 역시 전체 파이프라인의 미사용 최종 평가로 주장하지 않는다.','',
    '첫12모델24,000업데이트 이후, 경계 감독만의 효과를 분리하려고3모델6,000업데이트를 추가했다. 최종 체크포인트를 고정했으며 실사에 맞춰 임계값/체크포인트를 선택하지 않았다. 물리 가림의 가상 corner hidden 여부는 알 수 없어 hidden 성능으로 해석하지 않는다.','',
    '## 등록 기준 판정','',
    '각 시드 원본/잘림10/20/30에서 기준점 대비 악화1% 이내, 코너 가림/잘림40에서 V18 이득90% 이상 유지. 선 추가 이득은 잘림20에서 점 및 가상 감독 대조 각각 대비 시드별1% 이상과9세션 쌍대95%구간 상한<0를 요구했다.','',
    '| 방법 | 안정성 | 실패 조건 |','|---|---|---|']
    for a,gate in s['gates'].items():
        bad=['seed'+seed+': '+','.join(k for k,v in r.items() if not v) for seed,r in gate['per_seed'].items() if not all(r.values())]
        body.append('| '+NAMES[a]+' | '+('통과' if gate['pass_gate'] else '미통과')+' | '+'; '.join(bad)+' |')
    for a,r in s['line_evidence'].items():body+=['',f'실제 경계 vs {NAMES[a]}: 시드별 개선율 '+str([round(v,3) for v in r['per_seed_gain_percent']])+', 오차 차이95%구간 '+str([round(v,4) for v in r['ci95']])+'px. 대응 대조 통과='+str(r['pass_gate'])+'.']
    r=aux['comparisons']['real/crop20']['visible_hough_vs_visible_aux_only']['subgroups']['outside']
    body+=['',f'추가 탐색 대조(융합 vs 물리 감독만): 시드별 개선율 {[round(v,3) for v in r["per_seed_gain_percent"]]}%, 차이95%구간 {[round(v,4) for v in r["ci95"]]}px. 반복DEV·9세션 부트스트랩이며 시드 평균을 먼저 취한다. 시드/인접 프레임을 독립 표본으로 부풀리지 않았지만 다중 비교 보정/봉인 평가도 아니다.','',
    '## 어떤 정보를 이용했나','',*table(s,('visible_hough','visible_aux_only','tokens_off','hough_off','endpoints_off','donor','cross_session_donor')),'',
    '인접 donor는 각32배치에서 역할맵을 한 칸 옮겼다. 같은 세션의 가까운 프레임이 많아 차이가 작다는 것만으로 이미지별 정보가 없다고 결론 내릴 수 없다. 그래서 GT를 보지 않고 서로 다른 세션의 동일 조건 역할맵으로 바꾸는 추가 검사를 했다. 수신 이미지의 공간 특징·box·점 부모는 유지한다. 다른 세션 교체 시 잘림20 14.343→14.627px로 악화되어 일부 입력별 경계 정보 활용을 지지한다. 단, 강제로 어긋난 특징을 주는 사후 개입은 실제 재학습 대조와 다르다.','',
    '두 특징을 끄기14.343→15.593px, 끝점만 끄기15.537px에 비해 Hough만 끄기14.359px의 변화는 매우 작다. 현재의 추가 성능을 순수 Hough 선의 성과로 부르는 것은 부정확하다. Hough 제거 시 끝점/영상이 보완할 수 있어 선 정보 자체가 원천적으로 쓸모없다는 증명도 아니다.','',
    '## 경계/끝점 자체는 학습됐나','',
    '분리한 렌더의 역할별48×48 픽셀, 임계값0.5 고정. 아래는 조건/시드별 TP/FP/FN 합산 micro F1이며 픽셀/시드 간 상관을 독립 표본 수로 해석하지 않는다. 실사 선 GT 정확도가 아니다.','',
    '| 방법 | 가시 경계 F1 | 물리 끝점 F1 |','|---|---:|---:|']
    pooled={}
    for a in ('virtual_hough','visible_hough','visible_aux_only'):
        rr=[v for k,v in roles.items() if k.split('/')[-1].startswith(a+'_seed')];vals=[]
        for name in ('edges','endpoints'):
            tp=sum(r[name]['tp'] for r in rr);fp=sum(r[name]['fp'] for r in rr);fn=sum(r[name]['fn'] for r in rr);vals.append(2*tp/max(2*tp+fp+fn,1))
        pooled[a]=vals;body.append('| '+NAMES[a]+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    body+=['','점/latent 대조는 해당 역할 감독이 없어 그 확률맵의 의미별 F1을 공정한 분류 성능 비교로 쓰지 않는다. 보조 감독 모델의 학습집합 점 오차는 아래와 같다.','', '| 방법 | 기존 합성 관측 점 오차px | 새 렌더 점 오차px |','|---|---:|---:|']
    for a in ARMS:
        rr=[r for r in tr['rows'] if r['arm']==a];body.append('| '+NAMES[a]+' | '+' | '.join(f'{np.mean([r[k] for r in rr]):.3f}' for k in ('legacy_mean_px','render_mean_px'))+' |')
    body+=['','물리 감독을 추가한 모델이 점 전용보다 학습집합의 점 오차도 크다. 보조 과제와 점 과제의 예산/표현 경쟁 가능성이 있지만, 이것만으로 gradient 충돌을 확정할 수 없다. 이번 V32에서 PCGrad는 실행하지 않았다. 경계→끝점/점 역할 대응과 Hough 사용의 약함은 gradient surgery만으로 해결된다고 가정할 수 없다.','']
    for split,title in [('prospective','기존 prospective128'),('fresh_check','V31 분리 합성64'),('render_check','새 렌더 확인8그룹')]:body+=['## '+title,'',*table(s,('base','strong','point','latent_hough','virtual_hough','visible_hough','visible_aux_only'),split),'']
    body+=['## 검출 실패와 전체 분모','', '| 집합/조건 | 검출/전체 프레임 | 화면 밖 검출/전체 유효점 | 최종 full normalized error |','|---|---:|---:|---:|']
    for split in ('real','prospective','fresh_check','render_check'):
        for c in ('clean','crop20','crop40'):
            rr=s['results'][split+'/'+c];r=rr['visible_hough_seed1'];outside=r['strata']['outside'];body.append(f'| {split}/{c} | {r["detected_frames"]}/{r["frames"]} | {outside["detected_points"]}/{outside["total_points"]} | {np.mean([rr[f"visible_hough_seed{i}"]["full_normalized_mean"] for i in SEEDS]):.6f} |')
    body+=['','full normalized error는 원본 이미지 대각선으로 나눈 프레임 오차이며 검출 실패에1을 부여한다. 픽셀 평균만의 개선과 혼동하지 않는다. 모든 조건/시드의 P90·PCK·검출 분모·새 catastrophe는 SUMMARY.json에 저장했다.','',
    '## 실제 사례와 모든 시드','',
    '회색은 영상 밖이다. 초록GT·파랑기준점·보라복원점·검정새 점 대조·주황최종점. 청록 점선은 해당 점과 연결된 cuboid 역할의 Hough 최대 bin 후보다. 후보를 그렸다고 최종 모델이 그 선을 사용한 것은 아니다. 확률맵도 attention/기여도 지도와 다르다. gain/harm은 seed1 극단 사례로 뽑았으며 대표적인 평균 사례로 주장하지 않는다.']
    for case in gallery:
        body+=['',f'### {case["key"]}: {case["id"]}, point{case["point"]}','',case['selection'],'',f'![{case["key"]}](figures/{case["key"]}.png)','', '| seed | base | strong | point control | visible fusion | aux only |','|---|---:|---:|---:|---:|---:|']
        for r in case['per_seed']:body.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[a]:.3f}' for a in ('base','strong','point','visible_hough','visible_aux_only'))+' |')
    body+=['','## 원인 판단과 다음 단계','',
    '**확인한 사실:** 실제 경계 감독은 정의한 물리 선/끝점 예측을 학습시켰고, 가상 경계 감독 및 동일 점 대조보다 잘림20을 개선했다. 보조 감독만의 효과 외에 특징 전달의 이득도 있다. 하지만 Hough 고유 기여는 작고, 기존 좋은 점을 보존하는 안정성은 아직 해결되지 않았다.','',
    '**가능성이 높은 원인:** (1) 실제 선 체인 끝점과 가상 cuboid 점의 역할/위치가 다르다. 가까운 평행 선이라는 라벨만으로 어느 방향으로 얼마를 움직일지는 결정되지 않는다. (2) Hough/끝점 전체를 전역 벡터로 압축해8점에 반복 전달한다. 점별로 어떤 선의 어느 구간이 근거인지 직접 대응시키지 않아 Hough 위치 정보의 효용을 배우기 어려울 수 있다. (3) 기존 좋은 부모를 입력에 남기는 것과 실제로 그 부모의 이득을 지키는 것은 다르다. 감독을 붙인 공유 공간 특징에서 점 적합 오차도 늘어 표현/최적화 경쟁 가능성이 남았다. 각각은 다음 대조로 분리할 가설이며 단독 원인으로 확정한 결과는 아니다.','',
    '**다음 실험의 우선순위:** 전체 재학습이나 게이트 임계값 탐색 전에, 같은 물리 감독 아래 전역 pooling과 점 역할별 선 구간/끝점 cross-attention을 하나의 대조로 비교한다. 기준점·강한 복원점 둘 다 query로 유지하고, 선 법선 거리/방향·가시 support 구간·실제 체인 끝점의 위치를 따로 전달해야 한다. 선/끝점의 오류가 작아도 이동량을 설명하지 못하면 구조를 더 크게 만드는 우선순위를 낮춘다.','',
    '보존 문제는 추가 정보 실험과 분리해 고정한 강한 부모에 대한 잔차 경로 및 유지 출력을 비교한다. 경계 감독이 점 손실을 방해하는지 확인하려면 먼저 공유 특징에서 두 손실의 gradient 내적과 경계 전용 trunk 분리 대조를 측정한다. 그때 충돌이 재현될 경우에만 PCGrad 대조를 넣는 편이 원인을 구분하기 쉽다. 이번 결과만으로 PCGrad 개선 가능성을 수치로 주장하지 않는다.','',
    '새 자산·실사 자연 가림/잘림에서의 별도 검증과 실제 경계↔가상 점의 역할 검수가 남았다. 이전 좋은 결과를 전부 운으로 지울 근거도, Hough 아키텍처가 안정적으로 해결됐다고 할 근거도 없다. 현재 보존할 결과는 강한 점 복원기의 조건부 이득과 실제 경계/끝점 감독의 부분 효과다.','',
    '## 재현과 감사','',
    '`scripts/research/pallet_visible_geometry_v32/README.md`에 환경별 실행 순서를 기록했다. AUDIT.json: 15모델·30,000업데이트·660예측 파일, 수동GT/좌표 변환·독립 지표 재계산·GPU 출력 재현·이전 완료 파일 무변경 통과. 보조 catastrophe 기준의 original_points 메타데이터를 stock YOLO 기준으로 바로잡고 재집계했다. 실제 예측점·GT·체크포인트는 바꾸지 않았다.','',
    '학습 중 추가한 Hough/끝점 끄기 옵션은 추론 전용이다. 원래 학습 소스 스냅샷 및 해시를 보존했고, 정상 forward가 현재 소스와 완전히 같은지 실사/렌더에서 재검증했다. 원래4대조와 사후 aux-only/세션 donor의 판정 범위를 구분한다.','']
    body='\n'.join(body)+'\n';(RUN/'REPORT_KO.md').write_text(body);NOTE.parent.mkdir(parents=True,exist_ok=True);NOTE.write_text(body.replace('](figures/','](../../../../data/pallet/results/pallet_visible_geometry_v32/figures/'))
    html=markdown.markdown(body,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+TITLE+'</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1350px;padding:0 20px;color:#213044}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #cbd4dd;padding:8px}th{background:#eaf2fa}h1,h2,h3{line-height:1.35}code{background:#edf1f5}</style>'+html+'</html>')
    write(RUN/'FIGURE_DATA.json',dict(table=[dict(arm=a,values=matrix[i].tolist()) for i,a in enumerate(ORDER)],role_micro_f1=pooled));write(RUN/'CASES.json',dict(rows=gallery,render_labels=lc));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json')));print('REPORT32 COMPLETE',flush=True)

if __name__=='__main__':main()
