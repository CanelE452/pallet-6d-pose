from context35 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v35-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline

TITLE='V35 실제 경계·점 대응과 선 좌표 정밀도 검증'
ORDER=('base','strong','v34','gt_zero','gt_median','grid_oracle_median','top4_oracle_median','pred_median','pred_ridge_bounded','normal_oracle','gt_joint_median','pred_joint_median','joint_oracle')
LABELS=dict(base='V12 기준점',strong='V25 강한 점',v34='V34 점+선',gt_zero='정답 물리 선·offset0',gt_median='정답 물리 선·학습 상수offset',grid_oracle_median='전체1152bin 중 GT선 최선·상수offset',top4_oracle_median='예측top4 중 GT선 최선·상수offset',pred_median='예측top1·상수offset',pred_ridge_bounded='예측top1·범위 제한 offset',normal_oracle='정답 선·정답offset [상한]',gt_joint_median='여러 정답 선·상수offset*',pred_joint_median='여러 예측 선·상수offset*',joint_oracle='여러 정답 선·정답offset [상한]')

def merged(s,f,q,key='check/all/supported'):
    out=s['results'][key]['methods']|f['results'][key]['methods']
    if key=='check/all/supported':out['grid_oracle_median']=q['results']['check/all']
    return out

def table(rows):
    return ['| 방법 | 평균 점 오차px | 기준점 대비 차이95%구간 |','|---|---:|---|']+[f'| {LABELS[k]} | {rows[k]["mean_px"]:.3f} | '+str([round(x,3) for x in rows[k]['difference']['ci95']])+' |' for k in ORDER if k in rows]

def plots(rows,s,q):
    (RUN/'figures').mkdir(exist_ok=True);v=[rows[k]['mean_px'] for k in ORDER];fig,ax=plt.subplots(figsize=(12,6),layout='constrained');ax.barh(range(len(v)),v,color=['#3b7ea1' if 'oracle' not in k and not k.startswith('gt') else '#b294cc' for k in ORDER]);ax.set_yticks(range(len(v)),ORDER);ax.invert_yaxis();ax.axvline(v[0],c='gray',ls='--');ax.set(xlabel='Point error (px), lower is better',title='V35 same 820 supported check points / 3-seed means\nGT lines, role selection and oracle offsets are diagnostics, not deployable inputs')
    for i,x in enumerate(v):ax.text(x+.2,i,f'{x:.2f}',va='center')
    fig.savefig(RUN/'figures/performance.png',dpi=150);plt.close(fig)
    c=s['coverage']['check/all'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');axes[0].bar(['All bins\nGT oracle','Pred top4\nGT oracle','Pred top1'],[q['results']['check/all']['support_distance_mean_px'],c['best4_oracle_distance_mean_px'],c['prediction_distance_mean_px']],color=['#68a777','#e7b567','#db726b']);axes[0].set(ylabel='Physical support-to-line distance (px)',title='Finite grid and candidate extraction both leave error');axes[1].bar(['Zero offset','Train median','Train ridge'],[c['normal_mae_px'][k] for k in ['zero','median','ridge']],color=['#8a99b0','#68a777','#e7b567']);axes[1].set(ylabel='Normal offset MAE (px)',title='Physical line versus virtual corner bias');fig.savefig(RUN/'figures/decomposition.png',dpi=160);plt.close(fig)

def gallery():
    rows=read(RUN/'POPULATION.json')['rows'];g=np.load(RUN/'geometry.npz');a=np.load(RUN/'analysis_seed1.npz');b=np.load(RUN/'followup_seed1.npz');check=np.array([r['split']=='check' for r in rows]);valid=check[:,None]&a['supported'];cases=[]
    mask=valid&g['outside']&np.array([r['condition']=='crop20' for r in rows])[:,None];ii,jj=np.where(mask);first=(int(ii[0]),int(jj[0]));offset=np.unravel_index(np.where(valid,np.abs(a['gt_bias_px']),-np.inf).argmax(),valid.shape);line=np.unravel_index(np.where(valid,a['support_distance_top1'],-np.inf).argmax(),valid.shape)
    gain=np.linalg.norm(a['base']-g['gt'],axis=-1)-np.linalg.norm(a['gt_median']-g['gt'],axis=-1);good=valid&(gain>1);target=np.median(gain[good]);helpful=np.unravel_index(np.where(good,np.abs(gain-target),np.inf).argmin(),gain.shape)
    for key,(i,j),selection in [('helpful',helpful,'Closest to median gain among check points improved >1px with GT line'),('crop20',first,'First supported outside point in check crop20'),('offset',offset,'Largest physical-to-virtual normal offset in check'),('line_error',line,'Largest predicted support distance in check seed1')]:
        row=rows[i];im=actual_image(row);h,w=im.shape[:2];role=int(a['selected_role'][i,j]);normal=a['gt_normal'][i,j];rho=a['gt_rho'][i,j];pn=a['pred_normal'][i,j];pr=a['pred_rho'][i,j];gt=g['gt'][i,j];base=a['base'][i,j];gp=a['gt_median'][i,j];pp=a['pred_median'][i,j];coords=np.array([gt,base,gp,pp]);support=g['support'][i,role];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            bounds=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10)
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#e7ebef')
            for n,r,label,color in [(normal,rho,'Physical GT line','#159e9c'),(pn,pr,'Predicted Hough line','#e58c29')]:
                seg=clipline(np.r_[n,-r],bounds)
                if seg is not None:ax.plot(*np.array(seg).T,c=color,ls='--',lw=1.5,label=label)
            aa,bb=EDGES[role];uv=g['gt'][i,[aa,bb]];ax.plot(*uv.T,c='#9c65c4',ls=':',label='Virtual GT role edge');ax.plot(*support.T,c='#159e9c',lw=3,label='Visible support extent')
            for xy,label,color,marker in [(gt,'GT corner','#22a150','x'),(base,'Base','#337fc4','o'),(gp,'GT line + median offset','#9c65c4','D'),(pp,'Pred line + median offset','#dc6262','+')]:ax.scatter(*xy,c=color,marker=marker,s=60,label=label,zorder=5)
            ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[3],bounds[1]);ax.set_aspect('equal');ax.legend(fontsize=6)
        fig.suptitle(f'{key} / {row["id"]} / point{j} role{role}\nPhysical bias={a["gt_bias_px"][i,j]:+.2f}px; base={np.linalg.norm(base-gt):.2f}, GT line={np.linalg.norm(gp-gt):.2f}, pred line={np.linalg.norm(pp-gt):.2f}px',fontsize=10);fig.savefig(RUN/'figures'/f'{key}.png',dpi=160);plt.close(fig);cases.append(dict(key=key,id=row['id'],point=int(j),role=role,selection=selection,bias_px=float(a['gt_bias_px'][i,j]),base_px=float(np.linalg.norm(base-gt)),gt_line_px=float(np.linalg.norm(gp-gt)),pred_line_px=float(np.linalg.norm(pp-gt))))
    return cases

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS' and read(RUN/'GRID_AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');f=read(RUN/'FOLLOWUP.json');q=read(RUN/'QUANTIZATION.json');r=merged(s,f,q);plots(r,s,q);cases=gallery();c=s['coverage']['check/all'];body=['# '+TITLE,'',
    '**현재 예측 선을 그대로 점 보정에 넣는 방법은 통과하지 못했다. 정답 선에는 조건부 보정 정보가 있지만, 현재 Hough 격자의 좌표 정밀도와 후보 추출 오차가 그 이득을 잃게 한다. 실제 경계와 가상 점 사이 거리 차이도 일부 기여하지만 주된 원인을 그것 하나로 설명할 수 없다.**','',
    'V34 후속으로 기존 합성600관측을 점검했다. 고정 모델3시드 추론과 학습 구간에서만24개(point,role) offset 보정식을 계산했다. 새 신경망 학습·실사 평가·수동GT 수정은 하지 않았다. 아래 px는 V34 실사297 수치와 다른 **분리 합성 확인셋**의 값이다.','',
    '## 무엇을 분리했나','',
    '단위 법선 n, 물리 선 위치 ρ, 원래 점 p, 정답 가상 모서리 g에 대해 `n·(g−p) = (ρ−n·p) + (n·g−ρ)`다. 선까지의 거리만 사용하면 마지막 물리선↔가상점 offset이 빠진다. 정답 선·역할을 알고 이 항을 학습한 경우와 예측 선으로 교체한 경우를 비교했다. 선 하나는 법선 방향만 보정하고 접선 방향은 그대로 남긴다.','',
    '역할별로 실제 보이는 CAD 체인 중 support 길이가 가장 긴 선을 고르고(8px 이상), 각 점의 관련 역할 중 가장 긴 선을 사용했다. 이는 점 오차로 가장 좋은 선을 고른 것은 아니지만 **정답 선의 역할·가시성을 사용하는 진단**이다. 실제 추론에서는 얻을 수 없는 대응이다. 별도로 정답 없이 예측 점수로 역할을 고르는 결과도 아래에 보고한다.','',
    '## 같은 지원점에서 비교','',
    '확인용200관측 중 검출130관측,1040점 가운데 정답 관련 선이 보이는820점이 공통 분모다.3시드는 먼저 평균하고8개 source group을 묶어95%구간을 계산했다. 별표는 최초 결과 후 추가한 탐색 대조이며, oracle은 정답을 쓰는 한계 진단이다.','',*table(r),'','![점 오차](figures/performance.png)','',
    'GT선이나 GT로 후보를 고르는 결과를 실사용 아키텍처 성능으로 부르지 않는다. 기존 V34도 이 지원점에서는9.796px로, 단일 정답 선+상수 offset11.499px보다 좋다. 단일 선의 부분 이득이 곧 기존 최선 모델보다 좋아진다는 뜻은 아니다. 여러 정답 선+상수 offset8.296px는 평균 개선이 크지만 기준점 대비8그룹95%구간[-10.944,+0.999]px가0을 포함하므로 안정적 개선으로 채택하지 않는다.','',
    '## 1. 실제 경계와 가상 모서리가 다른가','',
    f'차이가 있는 점은 있다. 선택한 정답 선에서 물리선↔가상점 거리의 절댓값 중앙값은 {c["gt_abs_bias_median_px"]:.6f}px, P90은 {c["gt_abs_bias_p90_px"]:.3f}px였다. 평균 법선 offset 오차는0보정 {c["normal_mae_px"]["zero"]:.3f}px → 학습 상수 {c["normal_mae_px"]["median"]:.3f}px → 방향/aspect ridge {c["normal_mae_px"]["ridge"]:.3f}px였다. 많은 최장 경계가 가상 모서리 선과 거의 일치하므로 역할/offset 불일치가 모든 실패의 원인이라는 가설은 지지되지 않는다.','',
    '상수 offset의0보정 대비 차이는−0.448px,95%구간[-0.944,-0.004]px로 부분 효과가 있다. 다만 방향별 ridge의 사전등록 개선 기준은 구간이0을 포함해 실패했다. 상수 보정의 탐색 결과를 등록한 ridge 성공으로 바꾸지 않았다. 실제 물리 끝점과 가상 꼭짓점의 동일성까지 검증한 것은 아니다.','',
    '## 2. 선이 틀리는 원인은 후보인가, 격자인가','',
    f'같은 실제 경계 support에서 예측top1의 평균 선 거리 오차는 {c["prediction_distance_mean_px"]:.3f}px였다. top4 중 정답 경계로 가장 좋은 후보를 골라도 {c["best4_oracle_distance_mean_px"]:.3f}px다. 전체24×48=1152개 격자 중 정답 경계에 가장 가까운 bin을 골랐을 때는 {q["results"]["check/all"]["support_distance_mean_px"]:.3f}px까지 낮아진다.','',
    '![오차 분해](figures/decomposition.png)','',
    '상수 offset을 동일하게 사용하면 연속 정답 선11.499px → 전체격자 GT최선13.126px → 예측top4 GT최선14.446px → 예측top1 16.903px였다. 이 비교에서 **격자 근사와 후보 추출이 모두 손실을 만든다.** 전체격자 최선은 물리 support 거리 기준이며 최종 점 오차를 최적화한 oracle은 아니다.','',
    '현재 model33.py는 topk로 고른 정수 bin에서 θ/ρ를 계산하고 그 bin 내부의 연속 좌표 잔차를 별도로 예측하지 않는다. Support 중심은 연속 특징이지만 명시적인 선 좌표와 구분된다. 따라서 작은 좌표 오차를 줄이는 학습 경로와 후보/역할 판별을 각각 확인해야 한다. 이 제한은 현재 구현의 것이며 Deep Hough 전체의 불가능성을 뜻하지 않는다.','',
    '## 3. offset 보정식의 별도 실패도 분리했다','',
    '방향/aspect ridge는 예측 선 방향에 적용하면 확인 지원점에서23~24%가 학습 특징 범위를 벗어났고, 일부 offset은1200~1400px까지 커졌다. 이는 작은 학습 분산으로 정규화된 특징에서의 외삽 문제다. 큰 수치를 선 자체의 한계 근거로 사용하지 않는다.','',
    '첫 결과를 보존하고 학습 offset 최솟값/최댓값으로만 제한하는 탐색 대조를 추가했다. 예측 선+제한 offset은41.820→17.556px로 폭주가 줄었지만, 단순 상수16.903px나 기준점12.373px보다 좋지 않았다. 따라서 이 보정식의 폭주를 해결해도 현재 예측 선의 보정 실패는 남는다. 상한 설정이나 보정식은 확인GT로 조정하지 않았다.','',
    '정답 역할·가시성 없이 예측 점수로 선을 고르면 전체 검출1040점에서 상수 offset31.752px, 제한 offset31.880px였다(기준점13.393px). 경계가 실제 관측되는지와 역할이 맞는지도 별도 문제다. GT 대응 조건부820점 성능을 전체 동작 성능으로 바꿔 부르지 않는다.','',
    '## 4. 선 하나로 충분한가','',
    f'지원820점 중 하나의 독립 방향만 관측되는 점은 {c["rank1"]}개, 두 방향은 {c["rank2"]}개다. 정확한 offset까지 알려준 단일 법선 상한에서도 오차11.146px가 남았으며, 여러 방향 상한은5.488px였다. 이 값은 정답 잔차를 알고 계산한 직교투영이므로 구현/관측 가능성 검사이지 예측 모델이 아니다.','',
    '잘림20 화면 밖 지원점53개에서는 기준21.700→단일 정답offset20.488→여러 방향 정답offset19.348px로 여지가 작았다. 가림/잘림이라는 상태 이름만으로 선을 선택하기보다, 실제로 보이는 독립 방향과 역할 대응을 확인해야 한다. 접선 정보가 없는 점은 정답 선을 넣어도 완전히 복원할 수 없다.','',
    '## 조건별 분모','', '| 조건 | 요청 | 검출 | 선 지원점 | 독립1방향 | 독립2방향 |','|---|---:|---:|---:|---:|---:|']
    for key,v in s['coverage'].items():
        if key.startswith('check/'):body.append(f'| {key} | {v["requested_frames"]} | {v["detected_frames"]} | {v["supported_points"]} | {v["rank1"]} | {v["rank2"]} |')
    body+=['','학습16그룹/확인8그룹은 분리됐지만 같은2CAD를 공유하며, 이전 연구에서 사용한 렌더 확인셋이다. 자연 실사나 미지 자산의 최종 평가가 아니다. 검출 실패70/200은 위 조건부px에서 제외되어 있고 새 모듈로 복구하지 않았다. 학습 offset은 V34가 이미 학습한 렌더 train 구간으로 계산했으므로 부모 미학습 잔차에 대한 새 일반화 실험도 아니다.','',
    '## 실제 합성 이미지에서 확인','',
    'helpful은 정답 선으로1px 넘게 좋아진 지원점 중 개선량 중앙값에 가장 가까운 사례다. 전체 평균 사례가 아닌 조건부 개선 예시다. crop20은 확인셋에서 처음 나오는 지원되는 화면 밖 점이다. offset/line_error는 각각 큰 불일치를 보여주도록 선택한 극단 사례이며 대표 평균 사례가 아니다. 점선은 선 연장, 굵은 청록 선은 가시 support의 양 끝을 잇는 구간이며 중간에 가림이 있으면 전체가 연속 가시라는 뜻은 아니다.']
    for case in cases:body+=['',f'### {case["key"]}: {case["id"]}, point{case["point"]}','',case['selection'],'',f'![{case["key"]}](figures/{case["key"]}.png)']
    body+=['','## 다음에 바꿀 부분과 이번 판단','',
    '**현재 예측 선의 직접 투영/offset 보정은 채택하지 않는다. 다음 우선순위는 선 좌표를 실제 경계에 더 정확히 맞추는 경로와 올바른 역할 후보를 찾는 경로다.** 같은 특징/감독 아래 기존 정수 Hough bin 대조와 bin 주변 연속 θ/ρ 보정 또는 실제 support를 이용한 선 맞춤을 비교해야 한다. 새로운 실험에서는 점 오차만으로 두루뭉술하게 학습하지 말고 정확한 물리 경계 거리도 확인한다. 후보 top4의 GT최선과 전체격자 최선의 차이를 줄이는 것이 후보/역할 개선의 별도 목표다.','',
    '관측 가능한 비평행 선이 충분할 때에만 여러 방향을 사용하고, 그렇지 않으면 기존 좋은 점을 보존하는 별도 대조가 필요하다. 이미 V30에 법선 제한이 있었으므로 단순 투영이나 게이트 추가를 새로운 해법으로 반복하는 제안은 아니다. 그 연속 선/후보 개선 모델의 새 학습은 V35에 포함하지 않았고 실사 개선 가능성은 아직 미검증이다.','',
    'Gradient surgery는 이번에 사용하지 않았다. bin 내부 좌표 표현의 제한, 선 후보 선택 오류, 관측되지 않는 접선 성분은 점/선 손실 gradient 충돌과 다른 문제다. 이번 결과만으로 PCGrad가 해결할 가능성을 높게 볼 근거는 없다.','',
    '## 감사와 재현','',
    '600관측/436검출, 확인200/130. 물리 역할·선2764개를3D 최근접 관계와 독립 OpenCV 투영으로 검수했고 최대오차2.28e-13px였다. 확인GT를 크게 바꿔도 학습 offset계수가 그대로인 검사,63개 방법×시드 수치 재계산,1152bin 좌표식,기존파일 hash 불변 검사를 통과했다. 기하 일관성을 확인한 것이며 사람의 의미 라벨 인증은 아니다. 수동 실사GT는 이번에 수정하거나 새로 검수하지 않았다.','',
    '재현: `scripts/research/pallet_role_offset_v35/README.md`. 최초 기준과 실패 결과는 PROTOCOL/SUMMARY, 후속 범위 제한·여러 선 대조는 FOLLOWUP, 전체격자 oracle은 QUANTIZATION에 별도로 보존했다. 새 신경망 optimizer 업데이트는0회다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_role_offset_v35/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'CASES.json',dict(rows=cases));write(RUN/'FIGURE_DATA.json',dict(table=[dict(method=LABELS[k],value=r[k]['mean_px'],ci=r[k]['difference']['ci95']) for k in ORDER]));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),followup_sha256=sha(RUN/'FOLLOWUP.json'),quantization_sha256=sha(RUN/'QUANTIZATION.json')));print('REPORT35 COMPLETE',flush=True)
if __name__=='__main__':main()
