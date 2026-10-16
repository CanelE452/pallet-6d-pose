from context33 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v33-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
import cv2,markdown
from eval33 import population
from features25 import actual_image
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

TITLE='V33 점별 선 구간·끝점 대응 실험'
NAMES=dict(base='V12 기준점',strong='V25 강한 점 복원',v18='V18 기존 융합',v30='V30 적응형',v31='V31 분리 선택',visible_hough='V32 실제 경계 융합',pooled='동일 전역 pooling',query='점별 선+끝점 대응',no_geometry='명시적 위치 관계 제거',line_only='점별 선만 학습',end_only='점별 끝점만 학습',point_aux='경계 감독만·점 보정',line_off='융합 모델·선 끄기*',end_off='융합 모델·끝점 끄기*',both_off='융합 모델·둘 다 끄기*',cross_donor='다른 세션 경계맵*')
ORDER=('base','strong','v30','visible_hough')+ARMS
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
def value(s,a,c,t,split='real'):
    rr=s['results'][split+'/'+c];return float(np.mean([rr[f'{a}_seed{i}']['frame_mean_px'] if t is None else rr[f'{a}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))
def table(s,arms=ORDER,split='real'):
    cols=[(c,t) for c,t in COLS if split+'/'+c in s['results']]
    return ['| 방법 | '+' | '.join(LABELS[COLS.index(x)] for x in cols)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in arms]

def plots(s):
    (RUN/'figures').mkdir(exist_ok=True);matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in ORDER]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(ORDER)),ORDER);ax.set_title('V33 point-specific line support / endpoint correspondence\nReused real DEV, 3-seed mean px; parentheses: change vs base');fig.colorbar(im,ax=ax,label='Error change (px)');fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),layout='constrained');ax.axis('off');ax.set(xlim=(0,13),ylim=(0,4))
    boxes=[(.1,1.5,2.7,1.4,'Same physical supervision\nSame visual trunk / parents\nSame data and point head'),(3.4,1.5,2.5,1.4,'Separate queries\nBase AND restored point\nLocal patches + identity'),(6.5,1.5,2.9,1.4,'Incident-role attention\nHough support / endpoints\nRelative position + null'),(10,1.5,2.8,1.4,'Both queried contexts\nPoint mixture + residual\nNo hard condition routing')]
    for x,y,w,h,text in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc='#e8f1fa',ec='#5b7899'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    for a,b in [(2.8,3.4),(5.9,6.5),(9.4,10)]:ax.annotate('',(b,2.2),(a,2.2),arrowprops=dict(arrowstyle='->'))
    ax.text(6.5,.5,'Controls: global pooling / no explicit geometry / line only / endpoint only / aux only\nAll 6 trained with identical physical loss, plans, initialization and 2,000 updates per seed.',ha='center',fontsize=10);fig.savefig(RUN/'figures/architecture.png',dpi=160);plt.close(fig);return matrix

def cases():
    gallery=[]
    for key,c in [('case3','crop20'),('gain','crop20'),('harm','crop20'),('hidden','corner20')]:
        m,ts=population('real',c);g=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);p=np.load(RUN/'predictions/real'/c/'query_seed1.npz');pool=np.load(RUN/'predictions/real'/c/'pooled_seed1.npz');b=p['base'];strong=p['strong'];ef=np.linalg.norm(p['points']-g,axis=-1);eb=np.linalg.norm(b-g,axis=-1);ep=np.linalg.norm(pool['points']-g,axis=-1);mask=valid&p['detected'][:,None]&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts])
        if key=='case3':i=p['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');j=0;selection='Fixed user case3'
        else:i,j=np.unravel_index(np.where(mask,ef-eb if key=='harm' else ep-ef,-np.inf).argmax(),mask.shape);selection='Max harm vs base, seed1' if key=='harm' else 'Max gain vs pooled, seed1'
        row=m['rows'][i];raw=actual_image(row);h,w=raw.shape[:2];bb=np.array(row['box']);center=(bb[:2]+bb[2:])/2;diag=np.linalg.norm(np.maximum(bb[2:]-bb[:2],1));coords=np.array([g[i,j],b[i,j],strong[i,j],p['points'][i,j],pool['points'][i,j]]);la=p['line_attention'][i,2*j:2*j+2].astype(float);ea=p['end_attention'][i,2*j:2*j+2].astype(float)
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,zoom in zip(axes[:2],(False,True)):
            rect=(coords[:,0].min()-25,coords[:,1].min()-25,coords[:,0].max()+25,coords[:,1].max()+25) if zoom else (min(0,coords[:,0].min())-10,0,max(w,coords[:,0].max())+10,h)
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e1e5e9');ax.add_patch(Rectangle((0,0),w,h,fill=False,ec='#777777',ls='--'))
            for parent in (0,1):
                k=int(la[parent,:-1].argmax());normal=p['line_normal'][i,k];offset=p['line_offset'][i,k];line=np.r_[normal,-offset*diag-normal@center];seg=clipline(line,rect)
                if seg is not None:ax.plot(*np.array(seg).T,c='#00a6a6',ls='--',alpha=.7,lw=1)
            for name,pos,color,marker in [('GT',g[i,j],'#15a54b','x'),('Base',b[i,j],'#2988ec','o'),('Restored',strong[i,j],'#8547b3','s'),('Pooled',pool['points'][i,j],'#242424','+'),('Query',p['points'][i,j],'#ed8a00','D')]:ax.scatter(*pos,c=color,marker=marker,s=60,label=name,zorder=5)
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=7)
        ax=axes[2];ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));heat=ea[:,:-1].mean(0).reshape(12,8,8).sum(0);rgba=np.zeros((8,8,4));rgba[...,:3]=[1,.1,.3];rgba[...,3]=.6*heat/max(heat.max(),1e-12);ax.imshow(rgba,extent=(bb[0],bb[2],bb[3],bb[1]),interpolation='nearest')
        for parent in (0,1):
            k=int(la[parent,:-1].argmax());normal=p['line_normal'][i,k];tangent=np.array([-normal[1],normal[0]]);loc=p['line_support'][i,k]*diag+center;spread=p['line_spread'][i,k]*diag;ends=np.array([loc-2*spread*tangent,loc+2*spread*tangent]);ax.plot(*ends.T,c='#00ffff',lw=3);ax.scatter(*loc,c='#00ffff',s=20)
            k=int(ea[parent,:-1].argmax());loc=p['endpoint_locations'][i,k]*diag+center;ax.scatter(*loc,c='#ffff00',marker='*',s=90)
        ax.set_xlim(max(0,bb[0]-10),min(w,bb[2]+10));ax.set_ylim(min(h,bb[3]+20),max(0,bb[1]-20));ax.set_title('Actual endpoint attention (rescaled per case)\nCyan: top line support mean +/- 2 SD; yellow: top endpoint',fontsize=8)
        fig.suptitle(f'{key} / seed1 / point{j}: base {eb[i,j]:.2f}, pooled {ep[i,j]:.2f}, query {ef[i,j]:.2f}px\nAttention non-null, base/restored query: line {1-la[:,-1]}, endpoint {1-ea[:,-1]}',fontsize=9);fig.savefig(RUN/'figures'/f'{key}.png',dpi=160);plt.close(fig)
        per=[]
        for seed in SEEDS:
            rr=dict(seed=seed)
            for a in ('pooled','query','line_only','end_only'):
                z=np.load(RUN/'predictions/real'/c/f'{a}_seed{seed}.npz');rr[a]=float(np.linalg.norm(z['points'][i,j]-g[i,j]))
            z=np.load(V32.RUN/'predictions/real'/c/f'base_seed{seed}.npz');rr['base']=float(np.linalg.norm(z['points'][i,j]-g[i,j]));per.append(rr)
        gallery.append(dict(key=key,condition=c,id=row['id'],point=int(j),selection=selection,line_nonnull=(1-la[:,-1]).tolist(),end_nonnull=(1-ea[:,-1]).tolist(),per_seed=per))
    return gallery

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');matrix=plots(s);gallery=cases();tr=read(RUN/'TRAIN_DIAGNOSIS.json');audit=read(RUN/'AUDIT.json');v=lambda a,c='crop20',t='outside':value(s,a,c,t)
    body=['# '+TITLE,'',f'**6대조×3시드·18모델·실제36,000업데이트 완료. 전체 안정성+추가 선 이득 동시 기준: {"통과" if s["accepted"] else "미통과"}.**','',
    f'실사 잘림20: 전역 pooling {v("pooled"):.3f}px → 점별 선·끝점 대응 {v("query"):.3f}px. 점별 선만 {v("line_only"):.3f}px, 끝점만 {v("end_only"):.3f}px, 경계 감독만 {v("point_aux"):.3f}px다. 기존 기준점은 {v("base"):.3f}px다. 평균의 부분 개선과 모든 시드/조건의 안정성을 아래 판정표에서 구분한다.','',
    '## 동일 실사 비교','',
    '3시드 평균px, 작을수록 좋다. 원본은 검출 프레임 평균, 코너 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 실사297장·9세션은 반복 사용한 DEV이며 정본140/논문/봉인 평가와 다르다. GT를 학습하거나 실사 결과에 맞춰 임계값/체크포인트를 고르지 않았다.','',*table(s),'','![성능](figures/performance.png)','',
    '## 이번에 바꾼 것','', '![구조](figures/architecture.png)','',
    'V32와 같은 물리 경계/끝점 감독, 공간 특징, 최종 점 head를 유지했다. 바꾼 부분은 선·끝점 특징을 점에 전달하는 방식이다. 원래 점과 강한 복원점 각각의 영상 패치·좌표·점 번호로16개 query를 만든다. 각 query는 해당 점에 연결된3개 cuboid 역할의 선/끝점 token에만 attention을 줄 수 있다. 상태 이름이나 GT는 forward에 없다.','',
    '선은 역할별 Hough 상위4bin, 총48후보다. 원래24각도×48ρ를 유지했으며 top4는 이웃 중복을 제거한 NMS가 아니다. 각 선 주변 실제 관측 영역의 예측 경계 확률을 적분해 support 중심·접선 표준편차·영상 특징을 얻는다. 법선·거리·중심·범위·확률과 역할을 token에 보존하고, 각 점과의 법선/접선 거리로 attention을 조정한다. 그림의 ±2표준편차는 관측 support 요약이지 실제 선의 정확한 양끝이 아니다.','',
    '끝점은 역할별8×8 공간 셀, 총768개다. 각 셀 안의48×48 예측 끝점 확률로 중심좌표를 계산해 작은 위치 차이를 보존한다. query와의 상대 변위·거리도 사용한다. 두 부모의 attention 결과를 합쳐 기존128차원 선/끝점 입력을 대체한다. 근거를 사용하지 않는 null token도 넣었다. 최종점은 V32와 같은 signed mix+대각선15% 이내2D잔차이며, 정확한 부모 보존이나 법선 이동만을 강제하지 않는다.','',
    '6대조는 전역 pooling / 점별 전체 대응 / 명시적 위치 관계 제거 / 선만 / 끝점만 / 경계 보조 감독만이다. 선만·끝점만은 반대 특징 전달 경로를 끈 상태로 처음부터 학습했다. 이 대조도 기존 점·영상 입력과24채널 경계/끝점 보조 감독을 모두 유지한다. 즉 선만 대조는 점 보정에 선 특징만 추가하는 모델이다. 모든 대조는 같은 파라미터 집합396,925개·초기값·배치·손실·2000업데이트를 사용하지만 실제 활성 용량과 계산량은 다르다. 위치 관계 제거는 명시적인 좌표 token/상대 거리 입력을0으로 만들며, 역할과 support 기반 영상 특징은 남아 있어 모든 암묵적 기하를 제거한 대조는 아니다.','',
    'Hough bin 선택 위치는 이산적이다. 선택된 vote·support/끝점 중심·attention을 통한 gradient는 전달되지만 top-k 인덱스 자체는 미분하지 않는다. 사전검사에서 점 손실이 각 대응 모듈까지 전달되고, 한 끝점 픽셀의 셀 중심이 정확히 복구되며, pooled 경로가 원래 V32 forward와 동일함을 확인했다.','',
    '## 등록한 판정 기준','',
    '각 시드 원본/잘림10/20/30에서 기준점 대비 악화1% 이내, 코너 가림/잘림40에서 V18 이득90% 이상 유지. query는 pooled 및 end-only 각각 대비 잘림20에서 모든 시드1% 이상 개선하고9세션 쌍대95%오차 차이 구간 상한이0보다 작아야 한다. 최종2000step 고정이며 결과를 본 뒤 기준을 바꾸지 않았다.','',
    '| 방법 | 안정성 | 실패 조건 |','|---|---|---|']
    for a,gate in s['gates'].items():
        bad=['seed'+seed+': '+','.join(k for k,v in r.items() if not v) for seed,r in gate['per_seed'].items() if not all(r.values())];body.append('| '+NAMES[a]+' | '+('통과' if gate['pass_gate'] else '미통과')+' | '+'; '.join(bad)+' |')
    for a,r in s['line_evidence'].items():body+=['',f'query vs {NAMES[a]}: 시드별 개선율 {[round(x,3) for x in r["per_seed_gain_percent"]]}%, 오차 차이95%구간 {[round(x,4) for x in r["ci95"]]}px, 통과={r["pass_gate"]}.']
    body+=['','위 신뢰구간은3시드를 먼저 평균한 뒤9세션을 묶어 재표집한다. 시드/인접 프레임을 독립 표본으로 부풀리지 않았지만, 반복DEV의 적응적 연구와 다중 비교 때문에 독립 최종 검증의 확률 보장으로 해석하지 않는다.','',
    '## 선과 끝점 중 무엇이 기여했나','',*table(s,('query','line_only','end_only','point_aux','line_off','end_off','both_off','cross_donor')),'',
    f'고정 query 모델의 선 끄기 {v("line_off"):.3f}px, 끝점 끄기 {v("end_off"):.3f}px, 둘 다 끄기 {v("both_off"):.3f}px, 다른 세션 경계맵 {v("cross_donor"):.3f}px. 별표는 재학습이 아닌 추론 개입이다. 다른 세션 교체는 동일 조건에서 GT 없이 고르고, 수신 이미지 특징·점·box를 유지한다. 학습된 line-only/end-only와 추론 때만 끄는 결과를 혼동하지 않는다.','',
    'attention이 높다고 실제 오차 감소에 기여한 것은 아니다. null 이외의 attention 질량은 단순한 경로 활성도로만 보고, 고유 이득은 제거 재학습·위치 대조·다른 세션 개입과 함께 판단한다.','',
    '## 학습집합 적합과 최적화 진단','', '| 방법 | 기존 관측 오차px | 렌더 오차px | 고정1배치 gradient cosine 평균 |','|---|---:|---:|---:|']
    for a in ARMS:
        rr=[r for r in tr['rows'] if r['arm']==a];body.append('| '+NAMES[a]+' | '+' | '.join(f'{np.mean([r[k] for r in rr]):.4f}' for k in ('legacy_mean_px','render_mean_px','one_batch_gradient_cosine'))+' |')
    body+=['','gradient cosine은 최종 모델에서 미리 정한 첫 혼합 배치의 점 손실과 경계 손실을 공유 trunk에 대해 비교한 값이다. 단 한 배치이므로 지속적인 gradient 충돌의 원인 증명이나 PCGrad 효과 예측으로 쓰지 않는다. PCGrad는 이번에 실행하지 않았다.','',
    '## 경계 정확도와 실제 참조량','',
    '분리 렌더의 역할별48×48 픽셀·임계값0.5에서 TP/FP/FN을 조건/시드에 걸쳐 합산한 micro F1이다. 동일 CAD 라벨 규칙에 대한 지표이며 실사 선 정확도가 아니다. 픽셀 상관과 같은 원본 변형을 독립 표본 수로 해석하지 않는다.','',
    '| 방법 | 경계 F1 | 끝점 F1 |','|---|---:|---:|']
    role=read(RUN/'ROLE_METRICS.json')['rows']
    for a in ARMS:
        rr=[v for k,v in role.items() if k.split('/')[-1].startswith(a+'_seed')];vals=[]
        for typ in ('edges','endpoints'):
            tp=sum(r[typ]['tp'] for r in rr);fp=sum(r[typ]['fp'] for r in rr);fn=sum(r[typ]['fn'] for r in rr);vals.append(2*tp/max(2*tp+fp+fn,1))
        body.append('| '+NAMES[a]+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    body+=['','아래는 query 모델에서 유효점의 두 부모 query를 평균한 null 이외 attention 질량이다. 실제 보정 이득이나 선이 정확할 확률이 아니다.','', '| 조건 | 선 attention 질량 | 끝점 attention 질량 |','|---|---:|---:|']
    for c in ('clean','corner20','crop20','crop40'):
        m,ts=population('real',c);valid=np.array([t['valid'] for t in ts]);vals=[]
        for k in ('line_use','end_use'):
            vv=[]
            for seed in SEEDS:
                z=np.load(RUN/'predictions/real'/c/f'query_seed{seed}.npz');mask=valid&z['detected'][:,None];vv.append(z[k][mask].mean())
            vals.append(np.mean(vv))
        body.append('| '+c+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    body+=['',
    '## 기존 좋은 점의 이득을 보존했나','', '| 조건 | 기준점 | 강한 복원점 | query | query−강한 복원점 |','|---|---:|---:|---:|---:|']
    for c,t in [('corner20','hidden'),('crop40','outside')]:body.append(f'| {c} | {v("base",c,t):.3f} | {v("strong",c,t):.3f} | {v("query",c,t):.3f} | {v("query",c,t)-v("strong",c,t):+.3f} |')
    for split,title in [('prospective','기존 prospective128'),('fresh_check','V31 분리 합성64'),('render_check','V32 렌더 확인8그룹')]:body+=['','## '+title,'',*table(s,('base','strong')+ARMS,split)]
    body+=['','## 검출 실패를 포함한 평가','', '| 집합/조건 | 검출/전체 프레임 | 화면 밖 검출/전체 유효점 | query full normalized error |','|---|---:|---:|---:|']
    for split in ('real','prospective','fresh_check','render_check'):
        for c in ('clean','crop20','crop40'):
            rr=s['results'][split+'/'+c];r=rr['query_seed1'];o=r['strata']['outside'];body.append(f'| {split}/{c} | {r["detected_frames"]}/{r["frames"]} | {o["detected_points"]}/{o["total_points"]} | {np.mean([rr[f"query_seed{i}"]["full_normalized_mean"] for i in SEEDS]):.6f} |')
    body+=['','full normalized error는 원본 영상 대각선 정규화, 검출 실패에1을 부여한 프레임 평균이다. P90/PCK/중앙 가림 등 모든 조건·시드 지표는 SUMMARY.json에 있다. 픽셀 평균은 검출된 유효점 기준이며 검출률 자체를 개선한 모델은 아니다.','',
    '## GT와 데이터의 범위','',
    'V32의 원본·실제 변형 영상·고정 부모 출력·물리 경계/끝점 라벨을 그대로 사용했다.153합성 원본의 기존 관측8개와 새 렌더 관측8개를 배치마다 섞었다.2CAD자산240RGB·600요청 관측 중436검출, 외관16학습/8확인 그룹이며 같은 자산이 양쪽에 있다. 물리적 CAD 체인은 실제지만, 가까운 평행 cuboid 경계에 역할을 붙이는 규칙이므로 완전한 인간 의미 경계 GT는 아니다. 체인 끝점과 평가하는 가상 cuboid 점도 동일하지 않다.','',
    '수동 클릭의 valid 여부·원본 해시·crop 좌표 변환을 다시 감사했고 GT를 수정하지 않았다. 이전 결과/입력 해시도 확인했다. 새 자산/자연 실사 가림·잘림에 대한 독립 평가라고 주장하지 않는다. 물리 패널 가림에서는 가상 corner의 hidden GT가 없어 가림점 성능으로 해석하지 않는다.','',
    '## 실제 이미지와 대응 attention','',
    '회색은 영상 밖이다. 초록GT·파랑기준점·보라강한 복원점·검정pooled·주황query. 청록 점선은 각 부모 query가 가장 큰 가중치를 준 선 후보다. 모든 attention을 가중합하므로 이 선만 선택해 적용한 것은 아니다. 오른쪽은 실제 끝점 attention을 역할/부모에 대해 합산·평균하고 사례별 최대값으로 색을 정규화했다. 그림 간 색 농도로 활성도를 비교하지 않는다. 청록은 최상위 선의 support 중심±2표준편차, 노랑은 최상위 끝점 셀의 예측 중심이다.','',
    'case3은 사용자가 지목한 고정 사례다. gain/harm은seed1 극단 사례를 명시적으로 선택했으며 대표 평균 사례가 아니다. 모든 시드를 함께 보고한다.']
    for case in gallery:
        body+=['',f'### {case["key"]}: {case["id"]}, point{case["point"]}','',case['selection'],'',f'![{case["key"]}](figures/{case["key"]}.png)','', '| seed | base | pooled | query | line only | endpoint only |','|---|---:|---:|---:|---:|---:|']
        for r in case['per_seed']:body.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[a]:.3f}' for a in ('base','pooled','query','line_only','end_only'))+' |')
    if (RUN/'INTERPRETATION.md').exists():body+=['', (RUN/'INTERPRETATION.md').read_text()]
    body+=['','## 감사·재현','',f'AUDIT.json: {audit["models"]}모델·{audit["updates"]:,}업데이트·{audit["prediction_files"]}예측 파일, 지표 재계산·GT/좌표·동일 초기화/배치·기존 입력 무변경·36회 GPU 출력 재현 검사 통과. Attention 역할 마스크·정규화와 두 특징 제거 시 donor 불변성도 검사했다. 기존 pooled/aux-only 대조 재학습의 V32 차이는 감사 JSON에 기록했다.','',
    '`python scripts/research/pallet_correspondence_v33/run33.py`로 준비 완료된 V32 입력에서 실행한다. Python 환경은 pallet-yolo26이며 GPU가 필요하다. 이전 완료 결과는 수정하지 않는다.','']
    text='\n'.join(body)+'\n';(RUN/'REPORT_KO.md').write_text(text);NOTE.parent.mkdir(parents=True,exist_ok=True);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_correspondence_v33/figures/'))
    html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+TITLE+'</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1350px;padding:0 20px;color:#213044}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #cbd4dd;padding:8px}th{background:#eaf2fa}h1,h2,h3{line-height:1.35}code{background:#edf1f5}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',dict(table=[dict(arm=a,values=matrix[i].tolist()) for i,a in enumerate(ORDER)]));write(RUN/'CASES.json',dict(rows=gallery));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json')));print('REPORT33 COMPLETE',flush=True)

if __name__=='__main__':main()
