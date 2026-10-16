from features44 import *
from evaluate44 import METHODS
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v44-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2,markdown
from features25 import actual_image
TITLE='V44 새 장면·부모 미학습 예측으로 내린 종료 판정'
LABELS=dict(base='기존 base',strong='기존 strong',anchor='고정 V34 기준점',frozen_v41='고정 V41 직접 회귀',reused_point='기존 장면 감독 / 점 특징',reused_line='기존 장면 감독 / 점+선 특징',fresh_point='부모 미학습 장면 감독 / 점 특징',fresh_line='부모 미학습 장면 감독 / 점+선 특징')

def table_data(s):
 r=s['results']['check/all/all_detected'];rows=[dict(method=m,label=LABELS[m],mean_px=r[m]['mean_px'],per_seed=r[m]['per_seed_px'],gain=r[m]['gain_percent'],ci=r[m]['difference']['ci95']) for m in METHODS];conds=[]
 for key in ('check/clean/all_detected','check/corner20/hidden','check/crop20/outside','check/crop40/outside','check/physical_panel/all_detected'):
  if key in s['results']:conds.append(dict(key=key,values={m:s['results'][key][m]['mean_px'] for m in ('anchor','frozen_v41','fresh_point','fresh_line')}))
 return dict(rows=rows,conditions=conds)

def gallery(p):
 q=dict(np.load(RUN/'arrays/parents_seed1.npz'));c=dict(np.load(RUN/'arrays/corrections_seed1.npz'));point=c['points_fresh_point'];line=c['points_fresh_line'];gt=p['g']['gt'];pe=np.linalg.norm(point-gt,axis=-1);le=np.linalg.norm(line-gt,axis=-1);gain=pe-le;check=p['check'][:,None];cases=[]
 for name,mask,value in [('line_help',check&(gain>1),gain),('line_harm',check&(gain<-1),-gain),('outside_case',check&p['g']['outside'],le)]:
  if not mask.any():continue
  i,j=map(int,np.unravel_index(np.where(mask,abs(value-np.median(value[mask])),np.inf).argmin(),mask.shape));coords=np.array([gt[i,j],q['anchor'][i,j],q['frozen_v41'][i,j],point[i,j],line[i,j]]);im=actual_image(p['rows'][i]);h,w=im.shape[:2];fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
  for ax,zoom in zip(axes,(False,True)):
   bounds=(coords[:,0].min()-25,coords[:,1].min()-25,coords[:,0].max()+25,coords[:,1].max()+25) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#edf0f3')
   for k,role in enumerate(L.INC[j]):
    best=int(q['score'][i,role].argmax());normal=q['normal'][i,role,best];rho=q['rho'][i,role,best];tangent=np.array([-normal[1],normal[0]]);ends=normal[None]*rho+np.array([-3000.,3000.])[:,None]*tangent[None];ax.plot(*ends.T,c='#559b91',ls='--',lw=1,alpha=.6,label='DHT top-score lines / input roles' if k==0 else None)
    if q['match_valid'][i,role]:ax.plot(*q['match_segments'][i,role].T,c=['#d59a33','#c58344','#d4b671'][k],lw=2,label='Observed incident-role segment' if k==0 else None)
   for xy,label,color,mark in [(gt[i,j],'GT','#29964c','x'),(q['anchor'][i,j],'V34 anchor','#8a8d91','s'),(q['frozen_v41'][i,j],'Frozen V41','#8569a0','+'),(point[i,j],'New point-feature correction','#347cbb','o'),(line[i,j],'New point+line correction','#c55e72','+')]:ax.scatter(*xy,c=color,marker=mark,s=55,label=label,zorder=6)
   ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]));ax.set_aspect('equal');ax.legend(fontsize=7)
  fig.suptitle(f'{name} / {p["rows"][i]["id"]} / point{j}\nPoint {pe[i,j]:.2f}px, point+line {le[i,j]:.2f}px; observed inputs, not attention or GT-selected output',fontsize=10);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);cases.append(dict(name=name,index=i,id=p['rows'][i]['id'],point=j,point_px=float(pe[i,j]),line_px=float(le[i,j]),observed_segments=int(q['match_valid'][i,list(L.INC[j])].sum()),rule='Seed1 finalcheck, median within prespecified help>1px/harm>1px/outside condition; no model or threshold selection.'))
 return cases

def main():
 assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');p=population44();t=table_data(s);(RUN/'figures').mkdir(exist_ok=True);methods=['frozen_v41',*ARMS];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for m,label in zip(methods,['Frozen V41','Reused / point','Reused / point+line','New / point','New / point+line']):
  r=s['results']['check/all/all_detected'][m];axes[0].plot(SEEDS,r['gain_percent'],marker='o',label=label)
 for origin in ('reused','fresh'):
  r=s['comparisons']['check/all/all_detected'][origin+'_line_vs_'+origin+'_point'];axes[1].plot(SEEDS,r['gain_percent'],marker='o',label=origin+' supervision: line vs point')
 for ax in axes:ax.axhline(0,c='gray',ls=':');ax.axhline(1,c='#aaa',ls='--');ax.set(xticks=SEEDS,xlabel='Seed',ylabel='Improvement (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axes[0].set_title('New scene groups: improvement vs frozen V34');axes[1].set_title('Incremental line information: matched output heads');fig.savefig(RUN/'figures/decision.png',dpi=150);plt.close(fig);cases=gallery(p)
 table=['| 모델 | 평균px | 시드별px | V34 대비 개선율% | 차이95%구간px |','|---|---:|---|---|---|']+[f'| {r["label"]} | {r["mean_px"]:.3f} | '+', '.join(f'{x:.3f}' for x in r['per_seed'])+' | '+', '.join(f'{x:+.2f}' for x in r['gain'])+' | '+', '.join(f'{x:+.3f}' for x in r['ci'])+' |' for r in t['rows']]
 condition=['| 새 평가 조건 | V34 | 고정 V41 | 새 점 특징 | 새 점+선 특징 |','|---|---:|---:|---:|---:|']+['| '+r['key'].removeprefix('check/')+' | '+' | '.join(f'{r["values"][m]:.3f}' for m in ('anchor','frozen_v41','fresh_point','fresh_line'))+' |' for r in t['conditions']]
 frozen=s['gates']['frozen'];new=s['gates']['parent_unseen'];decision='현재 고정 특징·선 후보를 쓰는 보정 연구 계열을 종료한다.' if not new['passed'] else '이번 합성 화면 검증은 통과했다. 실제 적용은 별도 독립 실사 검증이 필요하다.';r=s['results']['check/all/all_detected'];comp=s['comparisons']['check/all/all_detected']['fresh_line_vs_fresh_point'];coverage=s['coverage'];body=['# '+TITLE,'','**'+decision+'**','',
 '이번에는 기존 확인셋에서 선택기·가중치·문턱을 다시 조정하지 않았다. 새 원본 장면을 만든 뒤, 고정한 두 질문과 종료 기준으로 판정했다.','',
 '## 두 검증과 데이터 분리','',
 '1. 보존 후보인 V41 직접 회귀의 작은 효과가 새 장면에서 재현되는지 고정 V34 기준점과 비교했다.\n2. 부모 모델들이 보지 못한 새 장면의 실제 예측으로 보정기를 학습하고, 같은 보정 방식에서 점 특징만 사용한 경우와 선 특징을 추가한 경우를 비교했다. 기존 장면으로 학습한 동일 대조도 함께 두었다.','',
 '같은 CAD 2개로 새로운 외관·조명·카메라 난수의 24개 그룹, 원본 RGB 240장과 파생 관측 600개를 생성했다. 각 원본의 구도·가림판·crop20/crop40·corner20은 같은 그룹에 묶었다. 12그룹은 새 학습, 4그룹은 별도 확인, 8그룹은 최종 평가다. 어떤 보정기도 새 최종 평가나 별도 확인 그룹으로 학습·문턱 선택하지 않았다.','',
 f'검출 수: 새 학습 {coverage["train/all"]["detected"]}/{coverage["train/all"]["requested"]}, 별도 확인 {coverage["calibration/all"]["detected"]}/{coverage["calibration/all"]["requested"]}, 최종 평가 {coverage["check/all"]["detected"]}/{coverage["check/all"]["requested"]}. 아래 점 오차는 검출된 관측의 8점을 대상으로 한다. 미검출은 복구하지 못했으며 점 오차에서 제외했다.','',
 '모든 고정 부모는 새 학습 장면도 보지 않았다. 따라서 부모의 학습 밖 예측으로 보정기를 학습한 것은 맞지만, 동일 장면으로 부모를 교차 재학습한 OOF 실험은 아니다. 기존/새 감독 비교에는 원본 장면 분포 변화도 포함돼 순수한 학습 노출 효과로 단정하지 않는다. 같은 CAD·동일 계열 렌더러의 새 장면 검증이며 새 CAD 또는 실사 일반화 검증이 아니다.','',
 '## 모델과 고정 기준','',
 '신경망 부모와 V41 보존 후보는 변경하지 않았다. 기존 특징 생성기를 새 관측에 연결하고, 예전 436관측에서 세 시드 모두 특징·V41 출력의 최대 차이 0을 확인했다. 실제 특징 구성 함수에는 GT 묶음을 전달하지 않았다.','',
 '새 보정기는 정규화된 x/y 잔차를 각각 HGB 절대오차 회귀로 학습한다. 이동 벡터는 bbox 대각선의 15%로 제한한다. 점 특징은 base/strong/V34 위치, 점 query, 종횡비, 점 ID의 47차원이다. 선 대조는 같은 점 특징에 기존 물리 3역할 후보의 특징 평균·표준편차·관측 여부를 추가한다. 알고리즘과 출력·학습 예산은 같으며 입력 차원은 다르다. 점 대조의 공통 V34 기준점·query도 선을 학습한 상류 모델에서 나오므로 순수 YOLO 점 전용 아키텍처와의 비교로 해석하지 않는다.','',
 '기존/새 학습 원본 × 점/점+선 입력 × 3시드 × x/y 2회귀 = 24모델, 각 60라운드로 총 1440라운드다. 신경망 업데이트는 0이다. 문턱·선택기·양성 가중치 sweep은 없다. 두 비교 모두 매시드 1% 개선, 그룹 단위 paired bootstrap 95% 구간 상단<0, 조건별 악화 1% 이내를 요구했다. 두 번째는 점 대조 대비 선 추가 이득도 매시드 1% 이상이어야 한다.','',
 '## 새 최종 평가 결과','',*table,'',
 '고정 V41의 평균 개선은 새 장면에서도 재현됐다. 9.058→9.025px, 시드별 .19/.61/.30% 개선, 차이 95% 구간 [-.0508,-.0129]px다. 다만 등록한 매시드 1% 개선에는 미달하고 crop40 쉬운 점에서 한 시드가 1.24% 악화하여 전체 안정성 기준은 실패했다. 작은 평균 효과와 완전한 안정성은 구분한다.', '', '새 점+선 보정은 9.058→8.674px로 세 시드 모두 3.92/4.84/3.97% 개선했고 평균 차이 구간도 0 미만이다. 그러나 새 점 특징 대조도 8.767px로 좋아졌다. 선을 추가한 순수 이득은 .43/.96/1.79%이고 차이 95% 구간 [-.1798,+.0026]px로 0을 포함한다. 선 추가 기준과 조건 보존 기준이 모두 실패했다.','',
 f'새 점+선 대조와 새 점 대조의 직접 비교: 평균 차이 {comp["difference"]["mean"]:+.4f}px, 95% 구간 {comp["difference"]["ci95"]}. 시드별 추가 개선율 {comp["gain_percent"]}%.','',
 '![두 검증의 새 장면 결과](figures/decision.png)','',*condition,'',
 '피해를 구체적으로 보면, seed3의 crop20 쉬운 점 135개 평균은 V34 3.011→새 점+선 4.018px, crop40 쉬운 점 105개는 3.087→4.152px다. 상대 악화율만 큰 극소수 사례가 아니라 해당 층 평균에서 약 1px 손해가 남았다. crop40 화면 밖 점 64개씩에서는 새 점 대조가 세 시드 모두 새 점+선보다 좋았다. 선이 잘림에 항상 더 유리하다는 가정도 이 결과가 지지하지 않는다.', '', '새 감독과 기존 감독의 직접 비교 역시 신뢰구간이 0을 포함한다. 따라서 부모 미학습 감독이 모든 문제를 해결했다거나 감독 출처만이 원인이었다고 결론 내리지 않는다.', '', '확인 데이터의 성능을 보고 모델이나 기준을 바꾸지 않았다. 기준 실패를 단순히 “학습이 더 필요하다”로 넘기지 않고 사전에 정한 종료 판정에 반영했다. 실험군이 모두 나쁘거나 점 대조와 차이가 없으면 선의 추가 효과를 주장하지 않는다.','',
 '## 실제 새 장면 사례','',
 '아래는 최종 평가 seed1의 조건별 중앙 사례다. 파랑은 새 점 특징 보정, 빨강은 새 점+선 특징 보정, 초록은 GT다. 초록빛 점선은 해당 점의 3역할별 DHT 최고 점수 직선, 주황 실선은 영상에서 매칭된 관측 선분이다. 둘 다 실제 입력에서 나온 것이며 attention 또는 정답으로 선택한 선이 아니다. 관측 선분이 없는 역할도 고정 DHT·영상 특징은 남으므로 보정 출력이 생길 수 있다. 새로운 보정기는 여러 선 특징을 함께 사용하므로 특정 선 하나가 이동을 결정했다고 해석하지 않는다.']
 for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, 점 {q["point"]}','',f'점 특징 {q["point_px"]:.2f}px → 점+선 특징 {q["line_px"]:.2f}px. 영상 매칭 선분 {q["observed_segments"]}개. {q["rule"]}','',f'![{q["name"]}](figures/{q["name"]}.png)']
 body+=['','## 최종 범위와 중단','',decision,'',
 '이 판정은 현재 고정된 표현·관측 후보와 보정 학습 방식에 대한 연구 우선순위 결정이다. 모든 Deep Hough/YOLO 융합이 수학적으로 불가능하다는 증명은 아니다. 다만 이 계열에서 추가 gate·가중치·문턱을 바꾸며 성공을 기대하는 실험은 더 진행하지 않는다. 선 관측 자체나 표현을 바꾸는 연구는 다른 가설과 새 예산을 가진 별도 과제로 취급한다.','',
 '## 검증·실행 환경','',
 '새 원본/기존 원본 이미지 해시와 그룹의 중복 0, 12/4/8 그룹 분리, 독립 GT 투영, 보존 특징·예측 일치, 24모델 재현, 실제 점 오차 독립 재계산, 비학습 영역 교란 재학습을 검사했다. 원래 완료 결과는 변경하지 않았다. 수동 실사 GT 수정·봉인 실사셋 접근은 없다.','',
 '실행 전 NVIDIA 커널580.173.02와 설치 라이브러리580.178.04가 달라 CUDA가 실패했다. 공식580.173.02 패키지에서 사용자 공간 라이브러리만 /tmp에 추출하여 해당 프로세스의 LD_LIBRARY_PATH에 지정했다. 실제 CUDA 텐서 연산과 NVIDIA EGL 렌더러를 확인했다. 시스템 드라이버 설치·교체·재부팅은 하지 않았다.','',
 '재현: `scripts/research/pallet_fresh_decision_v44/README.md`. 모델 학습 전에 PROTOCOL을 고정했고, 종료 후 결과·문서·코드 해시와 실제 Chrome 전달을 COMPLETION에 기록한다.','']
 text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_fresh_decision_v44/figures/'));html=markdown.markdown(text,extensions=['tables']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',t);write(RUN/'CASES.json',dict(rows=cases));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json')));print('REPORT44 COMPLETE',flush=True)
if __name__=='__main__':main()
