"""Source-linked report of every registered arm; no winner selection."""
from context25 import *
from prepare25 import ARMS
from eval25 import population
from features25 import actual_image
import os,html,cv2
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v25-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','line_vector':'V18 전체 보정','gate_original':'기존 관측·적용률','gate_paired':'짝지은 관측·적용률','gate_paired_local':'짝지은 관측·국소 적용률','vector_local':'방향 보정·점','vector_line':'방향 보정·점+선','vector_local_edge':'방향 보정·점+픽셀 미분','vector_line_edge':'방향 보정·점+선+픽셀 미분'}
SHORT={'stock':'YOLO','base':'Point baseline','line_vector':'Frozen V18','gate_original':'Gate / original','gate_paired':'Gate / paired','gate_paired_local':'Local gate / paired','vector_local':'Vector / point','vector_line':'Vector / line','vector_local_edge':'Vector / point + pixels','vector_line_edge':'Vector / line + pixels'}
CORE=(('corner20','hidden','Corner occlusion'),('crop20','outside','Crop 20%'),('crop40','outside','Crop 40%'))

def mean(s,arm,c,typ=None,split='real'):
    r=s['results'][split+'/'+c];rr=[r['stock']] if arm=='stock' else [r[f'{arm}_seed{i}'] for i in SEEDS]
    return float(np.mean([x['frame_mean_px'] if typ is None else x['strata'][typ]['mean_px'] for x in rr]))

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');diag=read(RUN/'DIAGNOSIS.json');figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
    methods=('stock','base','line_vector')+ARMS;vals=np.array([[mean(s,a,c,t) for c,t,_ in CORE] for a in methods]);delta=vals-vals[1];lim=max(abs(delta.min()),abs(delta.max()));fig,ax=plt.subplots(figsize=(11,7));im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_yticks(range(len(methods)),[SHORT[a] for a in methods]);ax.set_xticks(range(3),[x[2] for x in CORE]);ax.set_title('Final checkpoint results: improvement and harm\nMean target-point error (px); parentheses: change from point baseline')
    for i in range(len(methods)):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',color='white' if abs(delta[i,j])>.75*lim else 'black')
    fig.colorbar(im,ax=ax,label='Error change (px); negative is better');fig.tight_layout();fig.savefig(figdir/'performance.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(2)
    for off,split,color in [(-.17,'new_train','#2563eb'),(.17,'real','#d97706')]:
        cells=[next(r for r in diag['rows'] if r['split']==split and r['condition']=='crop20' and r['axis']=='side' and r['level']==side) for side in ('left','right')];v=np.array([[p['wrong']*100 for p in r['per_seed']] for r in cells]);ax.bar(x+off,v.mean(1),width=.3,color=color,label='Synthetic new512' if split=='new_train' else 'Real297');ax.scatter(np.repeat(x+off,3)+np.tile([-.04,0,.04],2),v.flatten(),s=20,color='black')
    ax.set_xticks(x,['Left crop','Right crop']);ax.set_ylabel('Wrong proposal direction (%)');ax.set_ylim(0,100);ax.set_title('Existing observations: right-side real crops are especially harmful');ax.legend();fig.tight_layout();fig.savefig(figdir/'direction_gap.png',dpi=160);plt.close(fig)
    m,ts=population('real','crop20');pred={a:np.load(RUN/'predictions/real/crop20'/f'{a}_seed1.npz') for a in ('base','gate_paired','vector_local','vector_line_edge')};gallery=[]
    chosen=[]
    for side in ('left','right'):
        candidates=[i for i,(r,t) in enumerate(zip(m['rows'],ts)) if r['detected'] and r['crop_side']==side and any(np.asarray(t['valid'])&np.asarray(t['outside']))];chosen+=sorted(candidates,key=lambda i:digest('v25-fixed-gallery:'+m['rows'][i]['id']))[:2]
    for j,i in enumerate(chosen):
        r=m['rows'][i];t=ts[i];raw=PREP.raw_image(r);origin=np.array(r['origin']);h,w=r['image_hw'];canvas=np.full_like(raw,185);x,y=origin.astype(int);canvas[y:y+h,x:x+w]=actual_image(r);valid=np.array(t['valid']);mask=valid&np.array(t['outside']);gt=np.array(t['original_points']);fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,a in zip(axes,('gate_paired','vector_local','vector_line_edge')):
            ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));b=pred['base']['points'][i]+origin;p=pred[a]['points'][i]+origin;ax.scatter(gt[valid,0],gt[valid,1],c='#16a34a',marker='x',s=45,label='GT');ax.scatter(b[valid,0],b[valid,1],facecolors='none',edgecolors='#2563eb',s=35,label='Baseline')
            for k in np.where(valid)[0]:ax.annotate('',p[k],b[k],arrowprops=dict(arrowstyle='->',color='#ea580c',lw=1.5))
            eb=np.linalg.norm(b-gt,axis=-1)[mask].mean();ep=np.linalg.norm(p-gt,axis=-1)[mask].mean();ax.set_title(f'{SHORT[a]}\noutside error: {eb:.2f} -> {ep:.2f}px');ax.set_xlim(0,raw.shape[1]);ax.set_ylim(raw.shape[0],0);ax.axis('off')
        axes[0].legend(loc='upper left',fontsize=8);fig.suptitle(f'Fixed hash-selected case / {r["crop_side"]} crop / gray = unavailable input');fig.tight_layout();name=f'case_{j}.png';fig.savefig(figdir/name,dpi=150);plt.close(fig);gallery.append(dict(id=r['id'],side=r['crop_side'],seed=1,file='figures/'+name,selection='First two detected valid outside-point frames per side by fixed v25 hash, not ranked by improvement.'))
    intervention=read(RUN/'INTERVENTION_DIAGNOSIS.json');iv={v:np.mean([r['need_inward'] for r in intervention['rows'] if r['split']=='new_train' and r['variant']==v])*100 for v in ('original','opposite')}
    stable=s['stable_arms'];stage2=s['evidence_gates']['gate_paired_vs_gate_original'];stage3=s['evidence_gates']['vector_line_vs_vector_local'];edge=s['evidence_gates']['vector_line_edge_vs_vector_local_edge'];lines=['# V25 관측 조건 보강과 보정 방향 변경 결과','',f'실험 완료: 21개 최종 모델, 42,000업데이트. 전체 안정성 통과: {", ".join(NAMES[a] for a in stable) if stable else "없음"}.','',
        '## 무엇을 실행했는가','',
        '1. 기존 캐시에서 잘리는 쪽·화면 밖 거리·투영비·잔존율별 방향 오류를 분해했다. 신규 합성512/확인128과 실사297을 사용했다. 원래253은 이 분해에서는 제외했으며 학습에는 모두 유지했다.',
        '2. 동일 원본을 반대쪽에서20% 잘라640개 입력을 만들고, YOLO·Hough·점 부모를 세 시드 재추론했다. 기존6조건은 전부 유지하고 추가 잘림 슬롯의 기존/짝지은 관측을 비교했다.',
        '3. 같은 짝지은 데이터에서2D 방향 보정의 점/선 대조와 실제 픽셀 미분 특징 유무를 비교했다. 새 가시 경계·끝점 GT를 만든 실험은 아니다.','',
        '## 기존 원인의 확인','',
        '약한 잘림에서 안쪽 보정이 필요한 비율이 신규 합성 약40%, 실사 약79%로 달랐다. 오른쪽 잘림의 실사 보정 방향 오류가 더 컸고, 점ID·5px 오차구간·잘리는 쪽을 맞춰도 차이가 남았다. 카메라 자세나 실제 가시 경계가 단독 원인이라고 확정한 것은 아니다.','',
        '![기존 방향 차이](figures/direction_gap.png)','',
        f'반대쪽 입력을 실제로 만든 뒤에도, 합성에서 안쪽 이동이 필요한 비율은 기존 {iv["original"]:.2f}% → 반대쪽 {iv["opposite"]:.2f}%였다. 실사의 약79%와 가까워지지 않았다. 따라서 양쪽 잘림을 더 관측하는 것만으로 필요한 방향 분포 차이가 해결됐다고 볼 수 없다. 원본과 반대쪽은 화면 밖 대상점이 달라 전체 분모와 동일점 비교를 [개입 진단](INTERVENTION_DIAGNOSIS.json)에 분리했다.','',
        '## 같은 평가의 비교','',
        '실사297장·9세션·3시드 평균. 대상점 오차px, 작을수록 좋다. 코너 가림280점 / 잘림20%558점 / 잘림40%571점. 검출 실패를 포함한 정규화 지표는 SUMMARY.json에 함께 기록했다.','',
        '![전체 성능 비교](figures/performance.png)','',
        '| 방법 | 코너 가림 | 잘림20% | 잘림40% | 전체 안정성 |','|---|---:|---:|---:|---|']
    for a,values in zip(methods,vals):lines.append('| '+NAMES[a]+' | '+' | '.join(f'{v:.3f}' for v in values)+' | '+('통과' if a in stable else '미통과' if a in ARMS else '대조군')+' |')
    lines+=['','## 단계별 판정','', '| 대조 | 잘림20%의 시드별 개선율 | 세션95%구간px | 이 조건의 추가 이득 통과 |','|---|---|---|---|']
    for label,v in [('짝지은 관측−기존 관측',stage2),('선−점 방향 보정',stage3),('픽셀 미분 포함 선−점',edge)]:
        row=v['crop20'];lines.append(f'| {label} | '+', '.join(f'{x:.2f}%' for x in row['per_seed_gain_percent'])+' | '+str([round(x,3) for x in row['ci95']])+' | '+('예' if row['pass_gate'] else '아니오')+' |')
    small=edge['crop20'];direction=[s['diagnostics']['real/crop20'][f'vector_line_edge_seed{i}']['outside'] for i in SEEDS]
    lines+=['',f'픽셀 미분을 포함한 점+선은 같은 점 전용 대조보다 약한 잘림에서 {mean(s,"vector_local_edge","crop20","outside"):.3f} → {mean(s,"vector_line_edge","crop20","outside"):.3f}px였다. 세 시드 개선율은 '+', '.join(f'{x:.2f}%' for x in small['per_seed_gain_percent'])+'이며 세션95%구간은 '+str([round(x,3) for x in small['ci95']])+'px다. 세 시드에서 같은 방향의 작은 추가 신호가 있지만, 사전 기준인 각 시드1%에는 못 미쳤고 점 기준12.372px의 악화도 해소하지 못했다. 다중 비교 보정은 하지 않은 재사용DEV 결과다.',
        '', '방향을 자유롭게 한 점+선+픽셀 모델은 기존 벡터의 방향이 잘못됐던 점 중 '+', '.join(f'{d["old_wrong_corrected_direction"]}/{d["old_wrong_points"]}' for d in direction)+'개를 각 시드에서 GT 쪽 방향으로 바꿨다. 따라서 방향 변경 경로가 무효였던 것은 아니다. 하지만 그 기존 실패 집합의 평균 오차 변화는 '+', '.join(f'+{d["old_wrong_mean_change_px"]:.2f}px' for d in direction)+'로 여전히 악화됐다. 일부 방향을 바로잡는 것과 전체 오차를 줄이는 것은 다르다.',
        '', '방향 보정이 고정 적용률보다 좋아져도 출력 공간과 손실이 함께 달라진 결과다. 방향 자유도 하나만의 원인 효과라고 해석하지 않는다. 반면 방향 보정 안의 점/선 대조는 같은 초기 파라미터·학습 배치·손실을 사용한다. 기존/짝지은 관측 대조는 같은2,000업데이트와 원본 그룹 순서를 쓰지만 새 관측의 검출 여부 때문에 실제 관측 노출 수가 시드별48개(약0.12%) 더 많다. 자세한 분모는 AUDIT.json에 있다.','',
        '## 실제 이미지의 고정 표본','',
        '세 시드 중 시드1. 좌/우 각각 해시 순서의2장을 선택했으며 개선량으로 고르지 않았다. 녹색은 수동GT, 파란색은 기준점, 주황 화살표는 모델이 실제로 이동한 방향이다. 회색 영역의 원본 픽셀은 모델에 입력되지 않았다.']
    for r in gallery:lines+=['',f'![{r["side"]} 잘림 고정 사례]({r["file"]})']
    lines+=['','## 판정 범위와 다음 결정','',
        ('전체 안정성 통과 대조가 있다. 점 전용 대비 선의 추가 이득은 조건별 검증표를 별도로 확인해야 하며, 새 촬영 세션을 봉인해 최종 재현하기 전 배포 결론을 내리지 않는다.' if stable else '등록한 대조 중 쉬운 조건 보존과 어려운 조건 이득90% 유지를 모든 시드에서 동시에 통과한 구조는 없다. 이번 관측 변화·방향 보정·픽셀 미분 조합만으로 안정적인 융합이 해결됐다고 말할 수 없다. 이것이 모든 점·선 아키텍처의 불가능성을 뜻하지는 않는다.'),
        '', '이번 개입은 잘리는 쪽의 관측 보강이다. 실제 카메라 이동·정면 자세의 신규 렌더링, 실제 가시 경계·끝점 감독, 독립 실사 촬영은 실행하지 않았다. 기존 박스 투영비를 실제 yaw나 가시성 정답으로 취급하지 않았다. 픽셀 미분은 질감·배경 경계도 포함할 수 있다.',
        '', '다음 원인 판단에서는 조건부 추가 이득과 전체 안정성을 분리하고, 반대쪽 관측에서도 남는 오차를 먼저 보아야 한다. 정답 감독이 가능한 가시 경계/끝점의 관측 유무 또는 새 촬영 조건을 바꾸는 실험은 이 결과와 별도로 등록해야 한다.',
        '', '## 재현과 검증','', '[프로토콜](PROTOCOL.json) · [조건 분해](DIAGNOSIS.json) · [전체 결과·시드별 판정](SUMMARY.json) · [GT·지표 감사](AUDIT.json) · [통합 경로 재생](INTEGRATION_CHECK.json) · [원본별 학습 배치](GROUPS_seed1.json)',
        '', '실사 GT 학습·수정, self-training 필터, 최선 체크포인트 선택은 수행하지 않았다. 모든 실사 평가와 기존 합성 확인 집합은 재사용DEV다. 유효 클릭 좌표 사용을 검사했으며 수동 주석의 물리적 오차가0이라는 보장은 아니다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report)
    # Embed the exact same report as HTML, with tables kept selectable.
    import markdown
    body=markdown.markdown(report,extensions=['tables','fenced_code']);page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V25 관측 조건과 보정 방향</title><style>body{font:16px/1.75 sans-serif;max-width:1240px;margin:40px auto;padding:0 24px;color:#182230}h1,h2{line-height:1.4}h2{margin-top:48px}img{max-width:100%;height:auto;border:1px solid #dde4eb;border-radius:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}th{background:#eef3f8}a{color:#175db5}code{background:#eef3f8}p,li{max-width:1100px}</style><main>'+body+'</main></html>'
    (RUN/'index.html').write_text(page);write(RUN/'GALLERY.json',dict(rows=gallery));write(RUN/'FIGURE_DATA.json',dict(methods=methods,core_conditions=CORE,error_px=vals.tolist(),summary_sha256=sha(RUN/'SUMMARY.json'),diagnosis_sha256=sha(RUN/'DIAGNOSIS.json')))
    # Put the result in the organized topic folder, rebasing every local result link.
    note=NOTE.read_text();marker='\n## 3. 실행 완료 결과\n'
    if marker in note:note=note.split(marker)[0]
    import re
    rebased=re.sub(r'\]\(([^)]+)\)',lambda m:'](../../../../data/pallet/results/pallet_observation_v25/'+m[1]+')',report)
    NOTE.write_text(note.replace('상태: 실행 중.','상태: 실행 완료.').replace('진행 중. [실험 산출물]','[실험 산출물]')+marker+'\n'+rebased)
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'REPORT_KO.md',RUN/'index.html',NOTE,*figdir.glob('*.png')]}));record('V25 결과 문서·성능 비교 그림·좌/우 고정 실패 표본·HTML 생성. 전체 안정성 통과='+str(stable)+'. 저장: data/pallet/results/pallet_observation_v25/index.html 및 정리된 notes의 pallet_observation_v25.md.');print('REPORT25 COMPLETE',stable,flush=True)

if __name__=='__main__':main()
