from context36 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v36-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
from data32 import population
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline
TITLE='V36 연속 선 좌표·후보 선택과 점 융합 학습 검증'
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
HEADERS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
NAMES=dict(base='V12 기준점',strong='V25 강한 점',v30='V30 적응형',v34='V34 점+선',grid='격자 대조',refine='좌표 보정만',rank='후보 선택만',both='좌표+후보 선택',adapt_grid='인코더 적응·격자',adapt_refine='인코더 적응·좌표',adapt_both='인코더 적응·좌표+선택')
ORDER=tuple(NAMES)

def values(s,a,arm,split='real'):
    data=a if arm.startswith('adapt_') else s;name=arm.removeprefix('adapt_');return [float(np.mean([data['results'][split+'/'+c][f'{name}_seed{i}']['frame_mean_px'] if t is None else data['results'][split+'/'+c][f'{name}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS])) for c,t in COLS if split+'/'+c in data['results']]

def point_table(s,a,arms=ORDER,split='real'):
    cols=[x for x in COLS if split+'/'+x[0] in s['results']];return ['| 방법 | '+' | '.join(HEADERS[COLS.index(x)] for x in cols)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[k]+' | '+' | '.join(f'{v:.3f}' for v in values(s,a,k,split))+' |' for k in arms]

def plots(s,a,l):
    (RUN/'figures').mkdir(exist_ok=True);matrix=np.array([values(s,a,k) for k in ORDER]);change=matrix-matrix[0];fig,ax=plt.subplots(figsize=(13,8),layout='constrained');im=ax.imshow(change,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
    ax.set_yticks(range(len(ORDER)),ORDER);ax.set_xticks(range(6),['Clean','Corner hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_title('V36 real reused DEV / 3-seed means\nMatched primary4-arm study and separate downstream adaptation');fig.colorbar(im,ax=ax,label='Error change vs base (px)');fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    r=l['results']['check/all']['methods'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in [(axes[0],'mean_support_distance_px','Actually quality-selected line'),(axes[1],'best8_distance_px','Best-of8 GT physical-line oracle')]:
        v=[r[k][key] for k in ARMS];ax.bar(ARMS,v,color=['#819bb9','#64a185','#d1a660','#a283b9']);ax.set(title=title,ylabel='Physical support distance (px)')
        for i,x in enumerate(v):ax.text(i,x+.05,f'{x:.3f}',ha='center')
    fig.suptitle('Same 820 supported rendered-check points; not real line-label accuracy');fig.savefig(RUN/'figures/lines.png',dpi=160);plt.close(fig);return matrix

def galleries():
    saved=[];meta=read(V35.RUN/'POPULATION.json')['rows'];g=np.load(V35.RUN/'geometry.npz');e=np.load(RUN/'LINE_ERRORS.npz');check=np.array([r['split']=='check' for r in meta])[:,None]&e['supported'];gain=e['top_grid_1']-e['top_both_1'];good=check&(gain>1);med=np.median(gain[good]);i1,j1=np.unravel_index(np.where(good,np.abs(gain-med),np.inf).argmin(),gain.shape);i2,j2=np.unravel_index(np.where(check,-gain,-np.inf).argmax(),gain.shape)
    for key,i,j,selection in [('line_help',i1,j1,'Median gain among >1px line improvements'),('line_harm',i2,j2,'Largest line-distance regression, seed1')]:
        row=meta[i];role=int(e['roles'][i,j]);raw=actual_image(row);h,w=raw.shape[:2];bb=np.array(row['box']);diag=np.linalg.norm(np.maximum(bb[2:]-bb[:2],1));center=(bb[:2]+bb[2:])/2;support=g['support'][i,role];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            rect=(support[:,0].min()-25,support[:,1].min()-25,support[:,0].max()+25,support[:,1].max()+25) if zoom else (0,0,w,h);ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.plot(*support.T,c='#11a57a',lw=4,label='Visible GT support extent')
            for arm,color in [('grid','#3c79bd'),('refine','#9364be'),('both','#d57b35')]:
                p=np.load(RUN/'predictions/render_check'/row['condition']/f'{arm}_seed1.npz');ri=p['ids'].tolist().index(row['id']);k=role*8+int(p['score'][ri].reshape(12,8)[role].argmax());n=p['normal'][ri,k];rho=p['offset'][ri,k]*diag+n@center;seg=clipline(np.r_[n,-rho],rect)
                if seg is not None:ax.plot(*np.array(seg).T,c=color,ls='--',label=arm+' selected line')
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=7)
        fig.suptitle(f'{key}: {row["id"]} / point{j} role{role}\nGrid distance={e["top_grid_1"][i,j]:.2f}px -> both={e["top_both_1"][i,j]:.2f}px; quality-selected, not attention attribution',fontsize=10);fig.savefig(RUN/'figures'/f'{key}.png',dpi=160);plt.close(fig);saved.append(dict(key=key,id=row['id'],point=int(j),selection=selection))
    m,ts=population('real','crop20');grid=np.load(RUN/'predictions/real/crop20/grid_seed1.npz');ref=np.load(RUN/'predictions/real/crop20/refine_seed1.npz');ad=np.load(RUN/'adapt/predictions/real/crop20/refine_seed1.npz');gt=np.array([t['points'] for t in ts]);mask=np.array([t['valid'] for t in ts])&np.array([t['outside'] for t in ts])&grid['detected'][:,None];gain=np.linalg.norm(grid['points']-gt,axis=-1)-np.linalg.norm(ref['points']-gt,axis=-1);eligible=mask&(gain>1);median=np.median(gain[eligible]);hi,hj=np.unravel_index(np.where(eligible,np.abs(gain-median),np.inf).argmin(),gain.shape);ci=grid['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546')
    for key,i,j,selection in [('point_help',hi,hj,'Median gain among >1px outside-point improvements, seed1'),('case3',ci,0,'Previously specified fixed case3, point0')]:
        raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];coords=np.array([gt[i,j],grid['points'][i,j],ref['points'][i,j],ad['points'][i,j]]);fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            rect=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e7ebef')
            for point,label,color,marker in [(coords[0],'GT','#20a35d','x'),(coords[1],'Grid control','#3d81c0','o'),(coords[2],'Refine only','#925cbf','D'),(coords[3],'Adapt encoder + refine','#df8e32','+')]:ax.scatter(*point,c=color,marker=marker,s=60,label=label)
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=8)
        err=np.linalg.norm(coords[1:]-coords[0],axis=-1);fig.suptitle(f'{key}: {m["rows"][i]["id"]} / point{j}\nGrid / refine / adapted refine errors = '+', '.join(f'{x:.2f}' for x in err)+' px',fontsize=10);fig.savefig(RUN/'figures'/f'{key}.png',dpi=160);plt.close(fig);saved.append(dict(key=key,id=m['rows'][i]['id'],point=int(j),selection=selection,errors_px=err.tolist()))
    return saved

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'POINT_SUMMARY.json');a=read(RUN/'ADAPT_SUMMARY.json');l=read(RUN/'LINE_SUMMARY.json');p=read(RUN/'PROBES.json');matrix=plots(s,a,l);cases=galleries();lr=l['results']['check/all']['methods']['both'];body=['# '+TITLE,'',
    '**21모델·42,000업데이트를 완료했다. 선택된 선의 합성 정확도는 개선됐으나, 실사의 점 보정이 모든 조건에서 안정적으로 좋아지는 구조는 아직 확인하지 못했다. 일차4대조와 기존 선 인코더를 적응시킨 추가3대조를 분리해 보존했다.**','',
    '## 바꾼 구조','',
    '동일 V34 영상 trunk·점 부모·query 특징을 고정했다.12역할마다 같은8개 Hough 후보를 뽑고, 작은 MLP가 θ/ρ의 격자 한 칸 이내 연속 잔차와 후보 품질을 예측한다. 격자/좌표만/후보선택만/둘다 네 대조가 같은 초기 점 head와 같은2000배치를 사용했다. 선 좌표는 실제 물리 경계 정답으로, 후보 품질은 경계까지의 거리로, 점 head는 점 정답으로 학습했다.','',
    '점 손실에서 선 예측기로 흐르는 경사는 차단했다. 선 예측기 내부의 좌표/품질 두 손실은 공유 MLP를 학습한다. 이것은 PCGrad가 아니며 점/선 목표 충돌과 선 예측기 내부 목표 경쟁을 구분하기 위한 고정 설계다. 새 좌표는 선 token의 위치·법선·support 중심에 반영하지만 영상 특징은 같은 원래 후보의 캐시를 유지했다. 실제 선 위치로 영상 특징을 다시 샘플링한 구조는 아니다.','',
    '일차 실험에서는 기존 선 token 인코더와 관계 인코더를 고정하고 점 head만 함께 학습했다. 추론 좌표 개입에 대한 반응이 작아서, 후속에서는 선 예측기를 고정하고 **선 token/관계 인코더와 점 head를 동일 V34 초기값에서 함께 학습**했다. 후속9모델18,000업데이트는 첫12모델24,000업데이트에 추가한 탐색 대조다.','',
    '## 선 자체는 더 정확해졌나','',
    '기존 합성 확인200관측 중 검출130관측, 실제 관련 역할의 가시 경계가 있는820점을 같은 분모로 비교했다. 아래는 점 오차가 아니라 실제 물리 support에서 선까지의 거리다. Best8은 정답 경계를 이용해 후보를 골랐을 때의 가능성이고 실제 선택 성능과 구분한다.','',
    '| 방법 | 실제 품질 선택 선 거리px | Best8 정답선 oracle 거리px | 선택선5px 이내 비율 |','|---|---:|---:|---:|']
    for arm in ARMS:
        r=l['results']['check/all']['methods'][arm];body.append(f'| {NAMES[arm]} | {r["mean_support_distance_px"]:.3f} | {r["best8_distance_px"]:.3f} | {r["selected_within5px"]:.3f} |')
    body+=['','![선 정확도](figures/lines.png)','',
    f'둘다 적용한 선은 격자 대비 시드별 {[round(x,2) for x in lr["per_seed_gain_percent"]]}% 개선,8그룹 차이95%구간 {[round(x,4) for x in lr["difference"]["ci95"]]}px로 등록한 선택선 개선 기준을 통과했다. 구간 상한이0에 가깝고 재사용된 작은 확인셋이라는 한계를 유지한다.','',
    '다만 좌표만 보정한 경우 선택선7.571→7.249px, Best8 3.137→2.892px로 작은 개선이었다. 둘다의 Best8은3.077px로 개선이 작아 별도 좌표 잠재력10% 기준은 실패했다. 후보 선택만의 평균6.037px가 둘다6.222px보다 좋지만, seed2의 개선은3.17%로 시드 간 차이가 컸다. 전체 격자 내 개선 여지가 전부 해결됐다는 결과는 아니다.','',
    f'격자 한 칸 보정 범위 안에 정답 선이 들어오는 후보가 있는 지원점 비율은 {l["results"]["check/all"]["reachable_fraction"]*100:.2f}%였다. 나머지는 이번 좌표 잔차만으로 정답에 도달할 수 없다. GT 역할·가시 support를 이용한 선 정확도 평가이며 실사 선 라벨 정확도를 측정한 것은 아니다.','',
    '## 실사 점 성능','',
    '재사용 실사297/9세션,3시드 평균px. 원본은 검출 프레임 평균, 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 정본140/논문/봉인 평가와 구분한다. 최초/후속 대조를 섞어 동일 학습 예산의 단일 승자 비교로 부르지 않는다.','',*point_table(s,a),'','![실사 점 성능](figures/performance.png)','',
    '일차 좌표만은 잘림20에서 격자13.485→12.720px로 세 시드9.38/4.56/2.75% 개선했고9세션 차이95%구간[-1.412,-0.263]px였다. 그러나 기준점12.372px에는 못 미치며 원래 점·가림·다른 잘림을 동시에 보존하지 못했다. 일차 주 비교인 둘다는14.394px로 격자보다 악화됐다.','',
    '## 실제로 정확한 선 좌표를 사용하는가','',
    '일차 좌표만 모델의 최종 가중치를 유지한 채 연속 좌표 보정을 끄면 잘림20 오차 변화는+0.000412px였다. 다른 세션 geometry만 넣으면+0.004759px이고 구간이0을 포함했다. 선 token 전체 제거는+1.482px 악화된다. 따라서 좌표만 모델의 재학습 이득을 추론 시 정밀한 좌표 활용의 이득으로 곧바로 해석할 수 없다. 작은 입력 변화로 학습 경로가 달라진 효과와 선 token의 영상/역할 정보 활용도 함께 남는다.','',
    '후속에서는 고정되어 있던 선 token/관계 인코더를 점 head와 함께 적응시켰다. 새 선 예측기와 영상 trunk는 바뀌지 않았고 선 좌표/품질이 원래와 정확히 같다는 검사를 통과했다. 아래는 후속 학습 후 실제 geometry 의존성이다.','',
    '| 후속 구조 | 개입 | 잘림20 화면밖 점 오차 변화px | 9세션95%구간 |','|---|---|---:|---|']
    for arm in ('refine','both'):
        for mode in ('geometry_donor','line_off'):
            r=a['comparisons']['real/crop20'][arm+'_'+mode];body.append(f'| {NAMES["adapt_"+arm]} | {mode} | {r["mean_px"]:+.5f} | {[round(x,5) for x in r["ci95"]]} |')
    body+=['','기하 교체는 점/영상/품질 점수는 유지하고 다른 세션의 정규화된 법선·위치·support 중심/범위만 바꾼 특징 수준 개입이다. 실제 영상과 불일치할 수 있으므로 물리적인 인과 정확도 보장과 구분한다. 선 전체 제거는 영상 요약도 제거하므로 순수 기하의 기여로 해석하지 않는다.','',
    '## Gradient surgery와 남은 원인','',
    '최종 둘다 모델의 공유 선 MLP에서 고정20배치/시드의 좌표 손실과 품질 손실 경사를 측정했다. Cosine 중앙값은+0.03~+0.057, 음수 배치는40~45%였고 품질 경사 norm이 좌표보다 중앙5.4~9.5배 컸다. 점 손실은 이 MLP로 들어가지 않는다.','',
    '선 예측기 내부에서 두 목표가 경쟁하거나 학습 신호 크기가 불균형할 가능성은 있다. 그러나 이 검사는 최종 모델의 일부 배치로, 전체 학습 내내 충돌이 주원인이었다는 증명은 아니다. PCGrad는 이번에 사용하지 않았다. 품질/좌표 head 분리나 균형 대조의 근거로는 쓸 수 있지만, 실사의 점 보존 실패와 선 기하 무반응까지 자동으로 해결한다고 주장하지 않는다.','',
    '## 고정한 안정성 판정','',
    '모든 시드에서 원본/잘림10/20/30은 기준점 대비 악화1% 이내, 코너 가림/잘림40은 V18 이득90% 이상 유지해야 한다. 주 모델은 잘림20에서 동일 격자 대조 대비1% 이상 개선과9세션 구간 상한<0도 요구했다. 결과를 보고 기준을 낮추지 않았다.','',
    '| 단계/구조 | 전체 보존 통과 | 대응 격자 대비 잘림20 추가 이득 |','|---|---|---|']
    for arm,v in s['gates'].items():body.append(f'| 일차 {NAMES[arm]} | {v["pass_gate"]} | '+('주 비교 통과' if arm=='both' and s['added_point_effect'] else '주 비교 미통과' if arm=='both' else '별도 탐색 비교')+' |')
    for arm,v in a['gates'].items():body.append(f'| 후속 {NAMES[arm]} | {v["stable"]} | {v["added"]} |')
    body+=['','## 실제 이미지','',
    'line_help/line_harm은 합성에서 선 거리의 개선과 악화를 보여준다. 품질 점수로 고른 역할 후보이며 점 head의 attention 설명이나 실사 edge GT가 아니다. point_help는 잘림20 화면 밖 점의1px 초과 개선 중 중앙 개선량 사례, case3은 사용자가 이전에 지목한 고정 사례다. 극단 사례와 조건부 사례를 전체 평균의 대표로 부르지 않는다.']
    for c in cases:body+=['',f'### {c["key"]}: {c["id"]}','',c['selection'],'',f'![{c["key"]}](figures/{c["key"]}.png)']
    for split,title in [('prospective','기존 prospective'),('fresh_check','분리 합성 확인'),('render_check','물리 렌더 확인')]:body+=['','## '+title,'',*point_table(s,a,('base','strong','v34','grid','refine','both','adapt_grid','adapt_refine','adapt_both'),split)]
    if (RUN/'INTERPRETATION.md').exists():body+=['',(RUN/'INTERPRETATION.md').read_text()]
    body+=['','## 범위·GT·감사','',
    '21체크포인트·42,000optimizer업데이트, 일차12모델과 추가9모델을 모두 보존했다. 합성 train/check source group은 분리됐지만 동일2CAD이고, 실사297와 합성 확인셋은 이전 연구에서 재사용된DEV다. 미지 자산·새 자연 가림/잘림 실사의 최종 일반화 성공으로 해석하지 않는다. 검출 실패 포함 정규화 오차/P90/PCK/분모는 POINT_SUMMARY·ADAPT_SUMMARY에 있다.','',
    '사전검사에서 격자 사이 이동, 유한 경사, 점→선 경사 차단, 선 정답 학습 신호, 같은 입력 forward/backward 재현을 확인했다. 이번 전체21학습을 각각 두 번 반복한 것은 아니다. 감사에서는 초기값·배치·고정 가중치·추가단계 선 예측기 불변·독립 거리/점 오차·수동GT valid/crop/hash·이전 완료파일 불변을 확인했다. 클릭 자체의 정밀도를 새로 측정하거나 GT를 수정하지 않았다.','',
    '재현: `scripts/research/pallet_continuous_lines_v36/README.md`. 최초 PROTOCOL/POINT_SUMMARY/LINE_SUMMARY와 추가 PROBE_PROTOCOL/ADAPT_PROTOCOL/ADAPT_SUMMARY를 구분한다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_continuous_lines_v36/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#26384c}img{max-width:100%}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #c3cedc}th{background:#eaf2f8}</style>'+html+'</html>');write(RUN/'CASES.json',dict(rows=cases));write(RUN/'FIGURE_DATA.json',dict(table=[dict(method=NAMES[k],values=matrix[i].tolist()) for i,k in enumerate(ORDER)]));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,point_sha256=sha(RUN/'POINT_SUMMARY.json'),line_sha256=sha(RUN/'LINE_SUMMARY.json'),adapt_sha256=sha(RUN/'ADAPT_SUMMARY.json')));print('REPORT36 COMPLETE',flush=True)
if __name__=='__main__':main()
