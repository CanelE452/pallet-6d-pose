"""Publish all registered arms, failure diagnosis and source-driven visuals."""
from context27 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v27-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import markdown,cv2,re
from features25 import actual_image
from eval27 import population
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','line_vector':'V18 기존 점+선 보정','v25_point':'V25 점+픽셀 보정','v25_line':'V25 점+선+픽셀 보정','point_utility':'V27 점 후보·효용','line_distance':'V27 선 후보·거리 손실','line_utility':'V27 선 후보·효용','line_role_utility':'V27 선 후보·효용·역할','role_line_off':'V27 역할 모델·선 경로 끄기'}
ORDER=tuple(NAMES)
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
def value(s,arm,c,typ=None,split='real'):
    rr=s['results'][split+'/'+c];names=['stock'] if arm=='stock' else [f'{arm}_seed{i}' for i in SEEDS];return float(np.mean([rr[n]['frame_mean_px'] if typ is None else rr[n]['strata'][typ]['mean_px'] for n in names]))

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');td=read(RUN/'TRAIN_DIAGNOSIS.json');ud=read(RUN/'UTILITY_DIAGNOSIS.json');figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
    table=[dict(arm=a,values=[value(s,a,c,t) for c,t in COLS]) for a in ORDER];matrix=np.array([x['values'] for x in table]);gain=matrix-matrix[1];fig,ax=plt.subplots(figsize=(11,6.3),layout='constrained');im=ax.imshow(gain,cmap='RdYlGn_r',vmin=-8,vmax=8,aspect='auto')
    for i in range(len(ORDER)):
        for j in range(len(COLS)):ax.text(j,i,f'{matrix[i,j]:.2f}\n({gain[i,j]:+.2f})',ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Clean','Corner20\nhidden','Crop10\noutside','Crop20\noutside','Crop30\noutside','Crop40\noutside']);ax.set_yticks(range(len(ORDER)),ORDER);ax.set_title('Real297 reused DEV / 3-seed means\nError px (difference from V12 base); no new arm passed all stability gates');fig.colorbar(im,ax=ax,label='Error change vs base (px)');fig.savefig(figdir/'performance.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.6),layout='constrained');ax.set(xlim=(0,13),ylim=(0,4));ax.axis('off')
    boxes=[(.1,2.3,2.8,1.2,'Frozen YOLO + DHT\npoint / image / line features'),(.1,.4,2.8,1.2,'Frozen V25 point proposal\n(existing hard-case gains)'),(3.4,1.1,3.1,1.7,'10 bounded candidates\nkeep / point / x / y\n3 line normals\n3 nonparallel intersections'),(7.1,1.1,2.7,1.7,'New learned module\nper-candidate amount\nexpected point-error gain\noptional incident role'),(10.4,1.1,2.4,1.7,'Soft candidate mixture\ncorrected8points\nGT-free inference')]
    for x,y,w,h,label in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='#e8eff6',ec='#436d90'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for x,y,u,v in [(2.9,2.9,3.4,2.2),(2.9,1,3.4,1.7),(6.5,2,7.1,2),(9.8,2,10.4,2)]:ax.annotate('',(u,v),(x,y),arrowprops=dict(arrowstyle='->',lw=1.7))
    ax.set_title('V27 trained architecture:65,443 parameters per new module / frozen upstream\nSynthetic GT supervises training losses only; physical edge visibility GT not provided');fig.savefig(figdir/'architecture.png',dpi=160);plt.close(fig)
    diag=[s['diagnostics']['real/crop20'][f'line_role_utility_seed{i}']['outside'] for i in SEEDS];dm=lambda key:float(np.mean([x[key] for x in diag]));fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');vals=[dm('base_px'),dm('after_px'),dm('oracle_candidates_px')];bars=axes[0].bar(['Base','Learned mixture','GT best candidate\n+ optimal amount'],vals,color=['#357bb0','#c55a47','#999']);bars[2].set_hatch('//')
    for i,v in enumerate(vals):axes[0].text(i,v+.25,f'{v:.2f}',ha='center')
    axes[0].set(ylabel='Outside point error (px)',ylim=(0,19));mass=[dm('keep_weight'),dm('point_weight'),dm('line_normal_weight'),dm('line_pair_weight')];axes[1].bar(['Keep','Point / x / y','Line normal','Line pair'],100*np.array(mass));axes[1].set(ylabel='Mean soft weight (%)',ylim=(0,105))
    for i,v in enumerate(mass):axes[1].text(i,100*v+2,f'{100*v:.2f}%',ha='center')
    fig.suptitle('Role + utility model: good candidates exist, but routing stays near point proposals\nOracle uses GT and is not an achieved prediction; weights are not calibrated confidence');fig.savefig(figdir/'selection.png',dpi=170);plt.close(fig)
    gen=[]
    for split in ('train','prospective','synth_val','real'):
        if split=='train':rr=[r for r in td['rows'] if r['condition']=='crop20' and r['arm']=='line_role_utility']
        else:rr=[s['diagnostics'][split+'/crop20'][f'line_role_utility_seed{i}']['outside'] for i in SEEDS]
        gen.append(dict(split=split,points_per_seed=rr[0]['points'],base=float(np.mean([r['base_px'] for r in rr])),after=float(np.mean([r['after_px'] for r in rr])),oracle=float(np.mean([r['oracle_candidates_px'] for r in rr]))))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(4)
    for i,(key,label) in enumerate([('base','Base'),('after','Learned'),('oracle','GT candidate oracle')]):ax.bar(x+(i-1)*.24,[r[key] for r in gen],.24,label=label,hatch='//' if key=='oracle' else None)
    ax.set_xticks(x,[r['split'] for r in gen]);ax.set_ylabel('Crop20 outside point error (px)');ax.legend();ax.set_title('Different populations: within-population before/after only\nSynthetic improvement does not transfer to real mild cropping');fig.savefig(figdir/'generalization.png',dpi=170);plt.close(fig)
    # Explicitly labelled extrema, not a representative sample or tuning set.
    m,ts=population('real','crop20');p=np.load(RUN/'predictions/real/crop20/line_role_utility_seed1.npz');gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);outside=np.array([t['outside'] for t in ts]);mask=valid&outside&p['detected'][:,None];difference=np.linalg.norm(p['points']-gt,axis=-1)-np.linalg.norm(p['base']-gt,axis=-1);score=np.where(mask,difference,0).sum(-1)/np.maximum(mask.sum(-1),1);eligible=np.flatnonzero(mask.any(-1));order=eligible[np.argsort(score[eligible])];chosen=[order[0],order[len(order)//2],order[-1]];gallery=[];fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,i,label in zip(axes,chosen,['Largest improvement','Median change','Largest worsening']):
        raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));g=gt[i,valid[i]];b=p['base'][i,valid[i]];a=p['points'][i,valid[i]];ax.scatter(g[:,0],g[:,1],marker='x',s=55,c='#00c943',label='GT');ax.scatter(b[:,0],b[:,1],s=25,c='#168bff',label='V12 base');ax.scatter(a[:,0],a[:,1],s=25,c='#ff3434',label='V27 role utility');coords=np.concatenate((g,b,a));ax.set_xlim(min(-25,coords[:,0].min()-15),max(w+25,coords[:,0].max()+15));ax.set_ylim(max(h+10,coords[:,1].max()+15),min(-10,coords[:,1].min()-15));ax.set_facecolor('#d9dde2');ax.axvline(0,color='black',ls='--');ax.set_title(f'{label}\nframe {m["rows"][i]["id"].split("/")[-1]} / outside change {score[i]:+.2f}px',fontsize=10);ax.legend(loc='upper right',fontsize=8);gallery.append(dict(id=m['rows'][i]['id'],condition='crop20',seed=1,selection=label,outside_change_px=float(score[i]),raw_image=m['rows'][i].get('image')))
    fig.suptitle('Real crop20 examples; grey canvas is outside observed image\nOutcome-selected illustrations only; all297frames remain in quantitative evaluation');fig.savefig(figdir/'examples.png',dpi=160);plt.close(fig)
    lines=['# V27 새 방향별 효용 구조: 학습·안정성 검증 결과','', '**12개 모델·24,000업데이트 완료. 사전 등록한 전체 안정성 및 선 고유 이득 동시 통과 모델은 '+('있음' if s['accepted_line_arms'] else '**없음**')+'.**','',
    '새 구조가 학습되지 않거나 선 경로가 끊겨서 생긴 결과는 아니다. 합성에서는 평균 보정 이득이 있었지만, 실사 약한 잘림에서는 기존 점 보정을 계속 과신했고 이전의 악화를 충분히 줄이지 못했다. 이 결론은 이번 고정 부모+보정 모듈에 대한 것이며 모든 점·선 아키텍처의 불가능을 뜻하지 않는다.','',
    '## 실제로 학습한 구조','', '![V27 구조](figures/architecture.png)','',
    'YOLO·DHT·V12 표현과 V25 점 전용 보정기는 고정하고, 그 뒤에 **후보별 이동량·기대 오차 감소량·역할 대응을 예측하는65,443파라미터 모듈**을 학습했다. 따라서 추론 아키텍처에 새 보정 모듈을 추가한 실험이다. self-training 필터나 GT로 후보를 고르는 추론 규칙이 아니다.','',
    '각 점에는 유지/점 벡터/x/y와 선 법선3개·비평행 선 쌍3개의 후보가 있다. 이동은 box 대각선15% 이내다. 평행에 가까운 선 쌍은 제외하고, 선의 위치만으로 부족한 방향은 두 번째 선의 교점 후보로 보완한다. 끝점 검출기는 이번에 추가하지 않았다.','',
    '모든 대조는 같은 배치 순서·2000업데이트·3개 시드·AdamW 3e-4·FP32를 사용했다. 점 전용 대조도 후보10개와 같은 파라미터 수를 갖지만 선 좌표/토큰/association을 입력받지 않는다. 공유 부모에 남아 있는 과거 선 학습의 영향까지 제거한 순수 YOLO 대조는 아니다.','',
    '| 새 대조 | 후보 | 학습 감독 |','|---|---|---|',
    '| point_utility | 점 벡터와 방향/크기 변형 | 점 오차+후보 효용+이동량 |','| line_distance | 점+선 법선+선 쌍 | 점 오차만 |','| line_utility | 점+선 법선+선 쌍 | 점 오차+후보 효용+이동량 |','| line_role_utility | 점+선 법선+선 쌍 | 위 감독+인접 cuboid 역할 |','',
    '역할 감독은 합성 cuboid GT 인접 선분과의 기하 대응이다. **실제 가시 물리 경계/끝점 GT 감독과 DHT 본체 재학습은 미포함**이다. 학습 관측은 V25와 같고 frozen V25 점 보정기도 이 원본으로 학습했으므로 in-sample 잔차 학습이라는 제약이 있다. 별도 원본 합성 평가를 분리해 보고했다.','',
    '## 기존 방법과 동일 실사 비교','',
    '실사297장·9세션 재사용 DEV,3시드 평균. clean은 검출 영상별 평균, 나머지는 가려진/화면 밖 유효점 평균px다. 조건별 대상점이 다르므로 한 평균으로 합치지 않는다. 검출 실패도 전체 분모 지표에 유지했다.','',
    '| 방법 | clean | 코너 가림 | 잘림10 | 잘림20 | 잘림30 | 잘림40 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in table:lines.append('| '+NAMES[row['arm']]+' | '+' | '.join(f'{v:.3f}' for v in row['values'])+' |')
    lines+=['','![전체 대조 결과](figures/performance.png)','',
    f'주 후보의 잘림20%는 **{value(s,"point_utility","crop20","outside"):.3f}→{value(s,"line_role_utility","crop20","outside"):.3f}px**로 새 점 후보 대조보다 좋아졌다. 하지만 기준점 **{value(s,"base","crop20","outside"):.3f}px**보다 악화됐고 기존 V25 점+선 보정 **{value(s,"v25_line","crop20","outside"):.3f}px** 대비 변화도 작다. 코너 가림은 V25 점+선보다 평균이 나빠졌으므로 전 조건 개선으로 해석할 수 없다.','',
    '## 안정성 사전 기준의 판정','',
    '각 시드에서 clean·잘림10/20/30은 점 기준의101% 이내, 코너 가림·잘림40은 기존 V18 보정 이득90% 이상 유지가 필요하다. 추가 선 효과는 주 분석 잘림20에서 새 점 대조 대비 모든 시드1% 이상 개선 및 세션 bootstrap95% 구간 상한<0이어야 하며 전체 안정성도 함께 통과해야 한다.','',
    '| 구조 | 전체 안정성 | 잘림20 선 이득: 시드1/2/3 | 차이의 세션95% 구간px | 선 고유 이득 기준 |','|---|---|---:|---:|---|']
    for a in ARMS:
        e=s['line_evidence'].get(a);lines.append('| '+NAMES[a]+' | '+('통과' if s['gates'][a]['all_conditions_pass'] else '미통과')+' | '+('/'.join(f'{v:.2f}%' for v in e['per_seed_gain_percent']) if e else '대조군')+' | '+(f'[{e["ci95"][0]:.3f}, {e["ci95"][1]:.3f}]' if e else '—')+' | '+('통과' if e and e['pass_gate'] else '미통과' if e else '대조군')+' |')
    lines+=['','역할+효용 모델이 실패한 조건:','']
    for seed,g in s['gates']['line_role_utility']['per_seed'].items():lines.append('- 시드'+seed+': '+', '.join(k for k,v in g.items() if not v)+'.')
    lines+=['','세션 bootstrap은 동일9세션을 단위로5000번 재표집했다. 평균 차이 구간이0 아래여도 시드별1%와 전체 보존이 충족되지 않으면 통과가 아니다. 재사용DEV·다중비교 보정 없음이며 새 봉인 실사 검증 주장은 하지 않는다.','',
    '## 왜 안정화되지 않았나: 확인된 사실과 해석','',
    '![후보 여지와 실제 선택](figures/selection.png)','',
    f'**후보 부재만의 문제는 아니다.** 잘림20 검출 밖점558개/시드에서 GT로 후보와 이동량을 최적으로 고르는 oracle는 **{dm("base_px"):.2f}→{dm("oracle_candidates_px"):.2f}px**였다. 실제 출력은 **{dm("after_px"):.2f}px**다. Oracle는 유지·점 후보·선/교점·연속 이동량을 포함하며 GT를 쓰므로 실현된 추론 성능이 아니다. V26의 상위3선 투영 oracle와 후보 집합도 달라 직접 같은 지표로 비교하면 안 된다.','',
    f'**선 경로는 작동하지만 기여가 작다.** 역할 모델은 실사 잘림20에서 점/x/y 후보에 평균 **{100*dm("point_weight"):.2f}%**, 선 법선에 **{100*dm("line_normal_weight"):.2f}%**, 선 쌍에 **{100*dm("line_pair_weight"):.2f}%**, 유지에 **{100*dm("keep_weight"):.2f}%**의 혼합 가중치를 부여했다. 이 가중치는 보정량의 기여도나 검증된 확률 신뢰도와 동일하지 않다. 선 경로를 끄면 오차가 '+f'{value(s,"line_role_utility","crop20","outside"):.3f}→{value(s,"role_line_off","crop20","outside"):.3f}px'+'로 바뀌었다. 즉 평균 기여는 약0.03px이며, 큰 복원 이득 대부분을 새 선 경로로 설명할 수 없다.','',
    '![합성에서 실사로의 변화](figures/generalization.png)','',
    '| 모집단·잘림20 밖점 | 점 수/시드 | 점 기준 | 새 역할+효용 | GT 후보 oracle |','|---|---:|---:|---:|---:|']
    for r in gen:lines.append(f'| {r["split"]} | {r["points_per_seed"]} | {r["base"]:.3f} | {r["after"]:.3f} | {r["oracle"]:.3f} |')
    lines+=['','**합성에서 유리한 보정이 실사 약한 잘림에서 불리해지는 문제가 남았다.** 학습뿐 아니라 별도 합성 원본에서도 평균이 좋아졌지만 실사에서는 악화됐다. 이는 관측/잔차 분포 차이와 효용 일반화 부족이라는 해석을 지지한다. 이 실험만으로 카메라, 재질, 원본 중복, 손실 비중 중 어느 하나를 단독 원인으로 확정하지는 않는다.','',
    '같은 잘림20 밖점·후보 집합에서 후보 보정 이득을 예측하는 AUROC:','', '| 모집단 | 시드1 | 시드2 | 시드3 |','|---|---:|---:|---:|']
    for split in ('train','prospective','synth_val','real'):
        rr=[r for r in (td['rows'] if split=='train' else ud['rows']) if r['arm']=='line_role_utility' and r['condition']=='crop20' and (split=='train' or r['split']==split)];lines.append('| '+split+' | '+' | '.join(f'{r["utility_help_auc"]:.3f}' for r in rr)+' |')
    lines+=['','후보 종류를 함께 모은 상관된 표본의 진단 AUROC이며 확률 보정이나 독립 표본 검정이 아니다. 일부 시드는 학습집합에서도 효용 구분이 약하다. 따라서 실패 전부를 실사 도메인 차이로만 돌릴 수도 없고, 역할 감독 하나로 효용 학습이 안정화됐다고 할 수도 없다.','',
    '## 실제 사례와 카메라 관측','', '![실사 개선·중간·악화 사례](figures/examples.png)','',
    '그림은 시드1·잘림20의 최대 개선/중앙 변화/최대 악화 영상을 명시적으로 고른 설명용 예다. 정량 결과는 전체 고정 모집단을 사용했다. 회색은 관측 영상 밖이며 녹색GT·파랑점 기준·빨강새 모델이다. 수동 실사 GT는 재사용했으며 새 재어노테이션은 수행하지 않았다.','',
    'V26 카메라360관측도 평가했다. 모든 방법이 같은 검출 결과172/360을 공유하므로 새 보정 모듈이 검출 실패를 해결한 것은 아니다. 새 OpenGL 재질/조명 분포가 기존 학습과 달라 이 표를 실사 구도별 성능으로 일반화하지 않는다. 아래는 각 조건의 검출 영상별 전체8점 평균px다.','', '| 방법 | clean | 좌측잘림20 | 우측잘림20 | 가림판 |','|---|---:|---:|---:|---:|']
    for a in ('base','v25_point','v25_line')+ARMS:lines.append('| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,None,"camera"):.3f}' for c in ('clean','crop_left20','crop_right20','physical_panel'))+' |')
    lines+=['','검출 실패 포함 전체 정규화 지표(실사 잘림20, 실패1·검출점 오차/원본 대각선 최대1):','', '| 방법 | 전체 영상 수 | 검출 영상 수 | 전체 정규화 오차 |','|---|---:|---:|---:|']
    for a in ('base','v25_line')+ARMS:
        rr=[s['results']['real/crop20'][f'{a}_seed{i}'] for i in SEEDS];lines.append(f'| {NAMES[a]} | {rr[0]["frames"]} | {rr[0]["detected_frames"]} | {np.mean([r["full_normalized_mean"] for r in rr]):.5f} |')
    lines+=['','## 완료 범위와 다음 판단','',
    '이번 요청의 **새 모듈 구현→실제 합성 학습→12모델 고정 평가→안정성 판정→원인 진단**을 완료했다. 수치/기하 검사,24개 통합 네트워크 재현, 전체 지표 독립 재계산, V25/V26 완료 파일 불변 검사를 통과했다. [감사 결과](AUDIT.json).','',
    '**현재 채택 판단: 운영 모델 교체 근거 부족.** 이 설정의 효용·역할 감독만으로는 약한 잘림의 과보정과 시드 변동을 해결하지 못했다. 점·선 결합 자체가 불가능하다는 판정은 아니다.','',
    '다음 원인 대조에서 바꿔야 할 것은 후보 수나 게이트를 다시 늘리는 일이 아니라, (1) 부모가 학습하지 않은 합성 원본으로 보정 잔차를 만들고, (2) 실사와 다른 점 정확도/필요 보정 방향의 분포를 실제 관측 생성으로 맞춘 뒤, (3) 점 오차 최적화와 후보 효용 감독을 분리해 효용 학습 자체가 시드별로 성립하는지 확인하는 것이다. 실제 가시 경계·끝점 감독과 DHT 본체 개선도 미검증으로 남는다. 이 제안은 이번에 추가 실행한 결과가 아니다.','',
    '[사전 등록](PROTOCOL.json) · [전체 결과·시드별 통계](SUMMARY.json) · [학습 완료 기록](TRAINING_COMPLETE.json) · [학습집합 진단](TRAIN_DIAGNOSIS.json) · [동일 조건 효용 진단](UTILITY_DIAGNOSIS.json) · [그림 입력](FIGURE_DATA.json)','']
    report='\n'.join(lines);(RUN/'REPORT_KO.md').write_text(report);html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V27 새 구조 학습·안정성 검증</title><style>body{font:17px/1.75 sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#183048}h1,h2{line-height:1.4}h2{margin-top:2.5em}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd5de;text-align:left}th{background:#eaf0f6}a{color:#096ba9}strong{color:#122e53}</style>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'</html>';(RUN/'index.html').write_text(html)
    initial=RUN/'DESIGN_REGISTERED.md'
    if not initial.exists():initial.write_text(NOTE.read_text())
    assert sha(initial)==read(RUN/'PROTOCOL.json')['sources'][str(NOTE.relative_to(ROOT))]
    prefix='../../../../data/pallet/results/pallet_direction_utility_v27/';NOTE.write_text(initial.read_text().replace('상태: 사전 등록·구현 중.','상태: 학습·검증 완료, 안정성 기준 미통과.')+'\n\n---\n\n[HTML 보고서]('+prefix+'index.html)\n\n'+re.sub(r'\]\(([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',report))
    write(RUN/'FIGURE_DATA.json',dict(table=table,generalization=gen,role_mixture_weights=mass,real_crop20_role_oracle=vals,gallery=gallery,sources={n:sha(RUN/n) for n in ('SUMMARY.json','TRAIN_DIAGNOSIS.json','UTILITY_DIAGNOSIS.json','AUDIT.json')}));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,source_sha256=sha(Path(__file__)),report_sha256=sha(RUN/'REPORT_KO.md'),figures=len(list(figdir.glob('*.png'))),stable_arms=s['stable_arms'],accepted_line_arms=s['accepted_line_arms']));record('V27 전체 성능/안정성·실패진단·학습구조·실제 사례 그림과 MD/HTML 완료. 실패 결과와 미실행 후속을 분리.');print('REPORT27',len(table),flush=True)

if __name__=='__main__':main()
