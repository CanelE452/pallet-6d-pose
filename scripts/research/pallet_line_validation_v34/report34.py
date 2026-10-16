from context34 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v34-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2,markdown
from data32 import population
from features25 import actual_image
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

TITLE='V34 반복 재현성과 선 위치 반응 검증'
NAMES=dict(base='V12 기준점',strong='V25 강한 점 복원',v30='V30 적응형',v33_line_only='V33 점+선',v33_pooled='V33 전역 pooling',line_only_r0='수정 연산·점+선',line_only_r1='점+선 동일 반복2',pooled_r0='수정 연산·전역 pooling',pooled_r1='전역 동일 반복2')
ORDER=('base','strong','v30','v33_line_only','v33_pooled','line_only_r0','pooled_r0')
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']
def value(s,a,c,t,split='real'):
    r=s['results'][split+'/'+c];return float(np.mean([r[f'{a}_seed{i}']['frame_mean_px'] if t is None else r[f'{a}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))
def table(s,arms=ORDER,split='real'):
    cols=[(c,t) for c,t in COLS if split+'/'+c in s['results']]
    return ['| 방법 | '+' | '.join(LABELS[COLS.index(x)] for x in cols)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in arms]

def plots(s,cf):
    (RUN/'figures').mkdir(exist_ok=True);matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in ORDER]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(ORDER)),ORDER);ax.set_title('V34 same architecture, verified equivalent pooling / fixed-grid sampling\n3-seed means; repeated runs do not increase sample size');fig.colorbar(im,ax=ax,label='Error change vs base (px)');fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for source,label in [('v33','V33 original weights'),('deterministic','V34 corrected run')]:
        rr=[r for r in cf['results'][source+'/real/crop20']['response'] if r['stratum']=='active']
        for h,style in [(4.,'-'),(8.,'--')]:
            rows=[r for r in rr if r['shift_px']==h];axes[0].plot([r['seed'] for r in rows],[r['median_signed_ratio'] for r in rows],marker='o',ls=style,label=label+f' +/-{h:g}px');axes[1].plot([r['seed'] for r in rows],[r['positive_fraction'] for r in rows],marker='o',ls=style)
    axes[0].axhline(.02,c='gray',ls=':');axes[1].axhline(.6,c='gray',ls=':');axes[0].set(title='Median signed response / input line shift',xlabel='Seed',ylabel='Response ratio');axes[1].set(title='Positive directional response fraction',xlabel='Seed',ylabel='Fraction');axes[0].legend(fontsize=8)
    for ax in axes:ax.set_xticks(SEEDS);ax.grid(alpha=.2)
    fig.savefig(RUN/'figures/response.png',dpi=170);plt.close(fig);return matrix

def cases():
    source='deterministic';m,ts=population('real','crop20');rr=[r for r in m['rows'] if r['detected']];ts=[t for t,r in zip(ts,m['rows']) if r['detected']];p=np.load(RUN/'counterfactual'/source/'real_crop20_seed1.npz');g=np.array([t['points'] for t in ts]);v=np.array([t['valid'] for t in ts]);active=v&(p['top_attention']>=.01);delta=p['shifted'][:,:,1,1]-p['shifted'][:,:,1,0];ratio=(delta*p['normal']).sum(-1)/16;gallery=[]
    basefile=np.load(V32.RUN/'predictions/real/crop20/base_seed1.npz');base=basefile['points'][basefile['detected']]
    for key in ('case3','positive','negative'):
        if key=='case3':i=p['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');j=0;selection='Fixed user case3'
        else:i,j=np.unravel_index(np.where(active,ratio if key=='positive' else -ratio,-np.inf).argmax(),active.shape);selection='Max signed sensitivity, seed1' if key=='positive' else 'Min signed sensitivity, seed1'
        row=rr[i];raw=actual_image(row);h,w=raw.shape[:2];bb=np.array(row['box']);center=(bb[:2]+bb[2:])/2;diag=np.linalg.norm(np.maximum(bb[2:]-bb[:2],1));normal=p['normal'][i,j];offset=p['line_offset'][i,j]*diag+normal@center;coords=np.array([g[i,j],base[i,j],p['baseline'][i,j],p['shifted'][i,j,1,0],p['shifted'][i,j,1,1]])
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,zoom in zip(axes[:2],(False,True)):
            rect=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,0,max(w,coords[:,0].max())+10,h)
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e1e5e9');ax.add_patch(Rectangle((0,0),w,h,fill=False,ec='#777777',ls='--'))
            for shift,color in [(-8,'#568bce'),(0,'#079a9a'),(8,'#db5959')]:
                line=np.r_[normal,-offset-shift];seg=clipline(line,rect)
                if seg is not None:ax.plot(*np.array(seg).T,c=color,ls='--',lw=1)
            for label,pos,color,marker in [('GT',g[i,j],'#14a950','x'),('Base',base[i,j],'#347fca','o'),('Line model',p['baseline'][i,j],'#eb9000','D'),('Line -8px',p['shifted'][i,j,1,0],'#568bce','+'),('Line +8px',p['shifted'][i,j,1,1],'#db5959','+')]:ax.scatter(*pos,c=color,marker=marker,s=60,label=label,zorder=5)
            ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.legend(fontsize=7)
        shifts=np.array([-8,-4,0,4,8]);points=np.array([p['shifted'][i,j,1,0],p['shifted'][i,j,0,0],p['baseline'][i,j],p['shifted'][i,j,0,1],p['shifted'][i,j,1,1]]);motion=(points-p['baseline'][i,j])@normal;axes[2].plot(shifts,motion,'o-',label='Measured point response');axes[2].axhline(0,c='gray',lw=1);axes[2].set(xlabel='Input line-token shift (px)',ylabel='Predicted point normal movement (px)',title='Same image / parents / token appearance\nMax |normal motion| = '+f'{np.abs(motion).max():.6f} px');axes[2].grid(alpha=.2)
        fig.suptitle(f'{key} / seed1 / point{j}: baseline error={np.linalg.norm(p["baseline"][i,j]-g[i,j]):.2f}px, signed ratio={ratio[i,j]:+.5f}\nDashed lines are feature interventions; actual image and GT are unchanged. Top attention={p["top_attention"][i,j]:.4f}',fontsize=10);fig.savefig(RUN/'figures'/f'{key}.png',dpi=160);plt.close(fig)
        gallery.append(dict(key=key,id=row['id'],point=int(j),selection=selection,signed_ratio=float(ratio[i,j]),top_attention=float(p['top_attention'][i,j]),line_shifts_px=shifts.tolist(),normal_response_px=motion.tolist(),error_px=float(np.linalg.norm(p['baseline'][i,j]-g[i,j]))))
    return gallery

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');cf=read(RUN/'COUNTERFACTUAL.json');rep=read(RUN/'DENSE_REPLICATION.json');first=read(RUN/'REPLICATION.json');audit=read(RUN/'AUDIT.json');matrix=plots(s,cf);gallery=cases();val=lambda a:value(s,a,'crop20','outside')
    body=['# '+TITLE,'',f'**총24모델·48,000업데이트 검증 완료. 수정 경로 반복 가중치/trace 일치={rep["all_equal"]}, 예측 일치={rep["all_predictions_equal"]}. 전체 성능 안정성+추가 선 이득={s["accepted"]}. 선 위치 반응/효용 기준={cf["gates"]["deterministic"]["pass_gate"]}.**','',
    f'같은 구조를 유지하고 재현성을 방해하는 구현을 분리했다. 최종 잘림20 점+선은 {val("line_only_r0"):.3f}px, 같은 수정 연산의 전역 대조는 {val("pooled_r0"):.3f}px, 기존 기준점은 {val("base"):.3f}px다. 재현성 성공과 정확도 개선은 별개 판정이다.','',
    '## 같은 학습이 정말 반복됐나','',
    '초기12모델은 strict deterministic algorithms, cuDNN deterministic, CUBLAS workspace :4096:8, TF32 off, FP32를 사용했다. CUDA adaptive_avg_pool2d backward가 지원되지 않아 첫 업데이트 전에 멈췄다. 정확히 나누어지는 격자만 사용하는 것을 확인하고 같은 평균을 계산하는 고정 avg_pool2d로 바꿨다. 첫 실패는 업데이트0회이며 원본 코드·로그를 보존했다.','',
    '그러나 같은 가중치/입력의 반복 backward에서도 grid_sample 입력 gradient가 달라졌다. 개별 연산 검사에서 최대1.19e-7 차이였고, forward는 같았다. 결정적 옵션이 켜져 있다는 사실만으로 실제 재현성이 검증된 것은 아니었다. 다른 연산들의 동시 원인이 전혀 없다는 주장과 구분한다.','',
    '두 번째12모델은 고정된 샘플링 좌표의 bilinear/zero-padding/align_corners=True 연산을 dense 선형 연산자로 구성해 bmm으로 계산했다. 출력과 입력 gradient의 float64 동등성, float32 반복 gradient 일치를 확인했다. 기존 점/영상/라벨/초기값/배치/손실/2000업데이트를 유지했다. 학습 아키텍처를 새로 설계한 비교가 아니라 동등한 수학 연산의 구현을 수정한 검증이다. Grid 좌표에 대한 학습은 지원하지 않으며 현재 고정 부모 좌표에서만 사용한다. 이 dense 구현의 메모리·속도는 별도 최적화하지 않았으므로 운영용 교체로 채택한 것이 아니다.','',
    '| 구조 | 시드 | 설정+pooling 수정 가중치/trace | dense sampler 추가 가중치/trace |','|---|---:|---|---|']
    for a,b in zip(first['rows'],rep['rows']):body.append(f'| {a["arm"]} | {a["seed"]} | {a["state_equal"]}/{a["trace_equal"]} | {b["state_equal"]}/{b["trace_equal"]} |')
    body+=['','반복은 서로 다른 프로세스에서 실행했다. 최종 가중치 tensor 해시, 전체2000step trace, 모든 평가 예측을 비교했다. 두 반복을 독립적인6시드로 부풀리지 않고, 성능은 반복0의3시드로 보고한다. 이 장비/라이브러리/입력에서의 재현이며 다른 하드웨어/버전까지 비트 일치를 보장하지 않는다.','',
    '## 동일 실사 비교','',
    '3시드 평균px. 원본은 검출 프레임 평균, 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 실사297/9세션은 사용자 승인한 재사용DEV 계보이며 정본140/논문/봉인 평가와 구분한다.','',*table(s),'','![성능](figures/performance.png)','',
    '점+선 모델도 원래 점/복원점과 영상 특징, 물리 경계/끝점 보조 감독을 유지한다. 점 보정에 전달하는 추가 경계 특징이 선뿐이라는 뜻이다. 선만으로8점 포즈를 추정한 모델과 구분한다.','',
    '## 고정한 성능 기준','',
    '각 시드 원본/잘림10/20/30은 기준점 대비 악화1% 이내, 코너 가림/잘림40은 V18 이득90% 이상 유지. 점+선은 전역 대조 대비 잘림20에서 시드별1% 이상 개선하고9세션 쌍대95%구간 상한<0여야 한다. 연산 수정 중에도 이 기준은 바꾸지 않았다.','',
    '| 구조 | 안정성 | 실패 조건 |','|---|---|---|']
    for a,g in s['gates'].items():
        bad=['seed'+seed+': '+','.join(k for k,v in row.items() if not v) for seed,row in g['per_seed'].items() if not all(row.values())];body.append('| '+a+' | '+str(g['pass_gate'])+' | '+'; '.join(bad)+' |')
    added=s['added_line_effect'];body+=['',f'점+선 vs 전역: 시드별 개선율 {[round(x,3) for x in added["per_seed_gain_percent"]]}%, 차이95%구간 {[round(x,4) for x in added["ci95"]]}px, 통과={added["pass_gate"]}. 반복DEV·9세션 분석이고 다중 비교/적응적 연구의 영향을 제거한 독립 최종 평가가 아니다.','',
    '## 선 위치를 바꾸면 보정 방향이 따라가나','',
    '각 점의 두 부모 query attention을 평균해 가장 큰 선 token을 GT 없이 골랐다. 그 선의 법선 방향으로 offset과 support 중심만 ±4/±8px 이동했다. 법선, 역할, confidence/질량/범위, 영상 특징과 다른 token, 점 부모, 원본 이미지는 고정했다. 입력 선을 옮긴 만큼 점이 반드시1:1로 따라야 한다는 가정은 하지 않는다.','',
    '응답률은 `(점(+h)−점(−h))·선 법선/(2h)`다. 양수는 선 이동 방향으로 반응,0은 무반응, 음수는 반대다. 사전 기준은 활성 query(top-token attention≥.01)의 시드별 중앙 응답률>.02 및 양의 반응 비율>.60을4/8px 모두 만족하는 것이다. 1e-4 이하의 양수는 수치 미세변화로 제외했다. 전체/활성/화면 밖·가림 하위군을 모두 저장했으며 이 수치는 실무 판정 기준이고 물리 법칙이 아니다.','',
    '![응답](figures/response.png)','', '| 원천 | 시드 | 이동px | 활성 유효점 수 | 중앙 응답률 | 양의 반응 비율 | 중앙 실제 이동px |','|---|---:|---:|---:|---:|---:|---:|']
    for source in ('v33','deterministic'):
        for r in cf['results'][source+'/real/crop20']['response']:
            if r['stratum']=='active':body.append(f'| {source} | {r["seed"]} | ±{r["shift_px"]:.0f} | {r["points"]} | {r["median_signed_ratio"]:.5f} | {r["positive_fraction"]:.3f} | {r["median_absolute_motion_px"]:.4f} |')
    body+=['','## 선의 위치와 선에서 읽은 영상 특징을 분리','',
    '다른 촬영 세션의 token으로 (a) 명시적 geometry만, (b) 샘플링한32차원 영상 특징만, (c) 둘 다 교체했다. 수신 이미지의 점 패치/전역 특징/부모 점은 유지했다. 역할 embedding과 확률/질량은 그대로다. 형상만 교체할 때 선 위치·법선·support 중심/범위를 함께 바꿨다. 이 개입은 실제 영상에 없는 조합일 수 있으므로 효과를 순수 인과적 정확도 보장으로 해석하지 않는다.','',
    '| 모델 원천 | 개입 | 잘림20 화면 밖 오차 | 오차 변화 | 95%구간 | 평균 출력 이동px |','|---|---|---:|---:|---|---:|']
    for source in ('v33','deterministic'):
        for mode,vv in cf['results'][source+'/real/crop20']['variants'].items():
            r=vv['target_stratum'];body.append(f'| {source} | {mode} | {r["changed_px"]:.3f} | {r["mean"]:+.3f} | {[round(x,4) for x in r["ci95"]]} | {r["movement_px"]:.3f} |')
    body+=['','정확한 geometry의 효용 판정은 geometry donor가 오차를 증가시키고 세션95%구간 하한>0인지를 요구했다. 선 끄기로 나빠졌다는 사실만으로 이 기준을 대신하지 않는다. Geometry 위치 반응과 유용성 판정은 COUNTERFACTUAL.json의 gates에 따로 있다.','',
    '## 다른 조건의 선 위치 반응','', '| 집합/조건 | 원천 | 활성점 중앙 응답률(3시드 평균, ±4px) | 양의 반응 비율 |','|---|---|---:|---:|']
    for key,r in cf['results'].items():
        rows=[x for x in r['response'] if x['stratum']=='active' and x['shift_px']==4.];source,split,c=key.split('/');body.append(f'| {split}/{c} | {source} | {np.mean([x["median_signed_ratio"] for x in rows]):.5f} | {np.mean([x["positive_fraction"] for x in rows]):.3f} |')
    for split,title in [('prospective','기존 prospective128'),('fresh_check','분리 합성64'),('render_check','렌더 확인8그룹')]:body+=['','## '+title,'',*table(s,('base','strong','v33_line_only','line_only_r0','pooled_r0'),split)]
    body+=['','## 분모·GT·해석 범위','',
    '검출 실패 포함 원본 대각선 정규화 오차, P90/PCK 및 모든 조건의 분모는 SUMMARY.json에 저장했다. 점 평균은 검출된 유효점에 대한 값이다. 원래 수동 클릭 GT의 valid·파일 해시·crop 좌표를 재감사했고 수정하지 않았다. 이는 라벨 전달 오류가 없다는 검사이며 손으로 찍은 모든 점이 실제 정답에 정확히 놓였다는 증명은 아니다. 클릭 오차의 크기를 이번에 새로 측정하거나 교정하지 않았다. V32의153원본+물리 렌더 관측/라벨을 그대로 사용했으며2CAD 자산이 렌더 학습/확인에 공유된다. 물리 체인 끝점과 가상 cuboid 점 역할의 의미 차이는 이번 연산 수정으로 해결되지 않는다.','',
    '신뢰구간은3시드를 먼저 평균하고 세션/합성 그룹을 묶어 재표집했다. fresh_check는2개 원천 그룹뿐이므로 해당 그룹 구간을 강한 일반화 근거로 보지 않는다. 이미지별 선 feature 반사실은 실제 이미지 재렌더와 다르고, 방향 반응을 보이는 것과 올바른 점을 복원하는 것은 별개다.','',
    '## 실제 이미지와 선 이동 반응','',
    'case3은 이전에 지목한 고정 사례다. positive/negative는 seed1 활성점의 가장 큰/작은 방향 반응을 골랐고 대표 평균 사례가 아니다. 점선은 실제 이미지를 수정한 것이 아니라 선 token에 준 위치 개입을 표시한다. 오른쪽은 입력 선 이동 대비 실제 예측점의 법선 이동 곡선이다. 세로축은 미세한 반응을 확대하므로 제목의 실제 최대 이동px를 함께 읽는다. Case3 point0의 top attention은0.0046으로 사전 활성 기준0.01보다 낮아 주 판정 모집단에서 제외되며 고정 사례 확인용으로만 표시했다.']
    for case in gallery:body+=['',f'### {case["key"]}: {case["id"]}, point{case["point"]}','',case['selection'],'',f'![{case["key"]}](figures/{case["key"]}.png)']
    if (RUN/'INTERPRETATION.md').exists():body+=['',(RUN/'INTERPRETATION.md').read_text()]
    body+=['','## 완료 감사','',f'{audit["models"]}모델·{audit["updates"]:,}업데이트, {audit["prediction_files"]}평가 예측과 {audit["counterfactual_files"]}반사실 파일 감사. 출력 재현, geometry/appearance 분리, 선택한 token 외의 불변성, GT·기존 완료 파일·입력 해시를 확인했다. 재현: scripts/research/pallet_line_validation_v34/README.md.','']
    body='\n'.join(body)+'\n';(RUN/'REPORT_KO.md').write_text(body);NOTE.parent.mkdir(parents=True,exist_ok=True);NOTE.write_text(body.replace('](figures/','](../../../../data/pallet/results/pallet_line_validation_v34/figures/'));html=markdown.markdown(body,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+TITLE+'</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1350px;padding:0 20px;color:#213044}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #cbd4dd;padding:8px}th{background:#eaf2fa}h1,h2,h3{line-height:1.35}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',dict(table=[dict(arm=a,values=matrix[i].tolist()) for i,a in enumerate(ORDER)]));write(RUN/'CASES.json',dict(rows=gallery));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),counterfactual_sha256=sha(RUN/'COUNTERFACTUAL.json')));print('REPORT34 COMPLETE',flush=True)

if __name__=='__main__':main()
