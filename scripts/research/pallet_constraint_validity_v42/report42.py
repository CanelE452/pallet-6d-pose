from context42 import *
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v42-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown,cv2
from features25 import actual_image
from experiment42 import load_inputs
TITLE='V42 목표점별 물리 제약 유효성 선택 검증'
LABELS=dict(visible='역할 경계 가시성',physical='물리 경계 위치·방향',constraint='목표점의 제약 정확도')
def ci_text(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'

def gallery(p):
    a,v,X=load_inputs(p,1);q=dict(np.load(RUN/'arrays/seed1.npz'));g=p['g'];m=dict(np.load(V40/'arrays/matching_seed1.npz'));key='constraint_visual_pre_all';base=np.linalg.norm(q['fallback']-g['gt'],axis=-1);after=np.linalg.norm(q['points_'+key]-g['gt'],axis=-1);gain=base-after;check=(~p['train'])[:,None];ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);valid=q['label_constraint']&a['valid'];best=np.where(valid,ce,np.inf).argmin(-1);besterr=np.where(valid,ce,np.inf).min(-1);bestaccepted=q['accepted_constraint_visual'][np.arange(p['N'])[:,None],np.arange(8)[None],best];cases=[]
    configs=[('accepted_help',check&(gain>.25),gain,'Median >0.25px incremental improvement over V41direct, seed1 check.'),('accepted_harm',check&(gain<-1),-gain,'Median >1px incremental harm versus V41direct, seed1 check.'),('missed_valid_constraint',check&(base-besterr>1)&~bestaccepted,base-besterr,'Median >1px GT-valid candidate opportunity rejected by learned classifier, seed1 check; candidate chosen withGT for diagnosis.')]
    for name,mask,value,rule in configs:
        if not mask.any():continue
        i,j=map(int,np.unravel_index(np.where(mask,np.abs(value-np.median(value[mask])),np.inf).argmin(),mask.shape));k=int(best[i,j] if name=='missed_valid_constraint' else q['selected_'+key][i,j]);role=int(INC[j][k]);candidate=a['candidate'][i,j,k];coords=np.array([q['fallback'][i,j],q['points_'+key][i,j],g['gt'][i,j],candidate]);im=actual_image(p['rows'][i]);h,w=im.shape[:2];fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            bounds=(coords[:,0].min()-20,coords[:,1].min()-20,coords[:,0].max()+20,coords[:,1].max()+20) if zoom else (min(0,coords[:,0].min())-10,min(0,coords[:,1].min())-10,max(w,coords[:,0].max())+10,max(h,coords[:,1].max())+10);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#edf0f3');ax.plot(*m['pred_segment'][i,role].T,c='#dc9c38',lw=2,marker='.',label='Observed segment / predicted role')
            if g['length'][i,role]>0:ax.plot(*g['support'][i,role].T,c='#666666',ls='--',label='GT physical support [diagnostic]')
            for xy,label,color,mark in [(g['gt'][i,j],'GT corner','#2b9d4f','x'),(q['fallback'][i,j],'V41 direct fallback','#347cbb','o'),(q['points_'+key][i,j],'V42 applied','#d26565','+'),(candidate,'Inspected candidate','#9873b2','+')]:ax.scatter(*xy,c=color,marker=mark,s=65,label=label,zorder=6)
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]));ax.set_aspect('equal');ax.legend(fontsize=7)
        prob=float(q['prob_constraint_visual'][i,j,k]);truth=bool(q['label_constraint'][i,j,k]);th=read(RUN/'models/seed1.json')['thresholds']['constraint_visual']['threshold'];fig.suptitle(f'{name} / {p["rows"][i]["id"]} / point{j}, role{role}\nP(valid)={prob:.3f}, threshold={th:.2f}, GT valid={truth}; fallback {base[i,j]:.2f}px -> applied {after[i,j]:.2f}px; candidate {ce[i,j,k]:.2f}px',fontsize=11);fig.savefig(RUN/'figures'/f'{name}.png',dpi=150);plt.close(fig);cases.append(dict(name=name,id=p['rows'][i]['id'],index=i,point=j,role=role,candidate=k,rule=rule,probability=prob,threshold=th,GT_valid=truth,fallback_px=float(base[i,j]),applied_px=float(after[i,j]),candidate_px=float(ce[i,j,k])))
    return cases

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'PROBE_AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');probe=read(RUN/'PROBE.json');p=population();(RUN/'figures').mkdir(exist_ok=True);r=s['results']['check/all/all_detected'];classrows=[];rows=[]
    for target in TARGETS:
        name=target+'_visual';classrows.append(dict(target=target,label=LABELS[target],auroc=[s['classification'][str(k)][name]['check']['auroc'] for k in SEEDS],precision=[s['classification'][str(k)][name]['check']['precision'] for k in SEEDS],recall=[s['classification'][str(k)][name]['check']['recall'] for k in SEEDS]))
        for feature in FEATURES:
            name=target+'_'+feature;rows.append(dict(method=name,label=LABELS[target]+(' / 영상+기하' if feature=='visual' else ' / 기하'),pre_all=r[name+'_pre_all']['mean_px'],pre_hard=r[name+'_pre_hard']['mean_px'],post_hard=r[name+'_post_hard']['mean_px']))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(3)
    for row,label in zip(classrows,['Visible role','Physical alignment','Target constraint']):axes[0].plot(x,row['precision'],marker='o',label=label);axes[1].plot(x,row['recall'],marker='o',label=label)
    for ax,ylabel in zip(axes,['Precision','Recall']):ax.set(xticks=x,xticklabels=['Seed1','Seed2','Seed3'],ylim=(0,1.05),ylabel=ylabel,title='Actual valid segment-role slots / reused check');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].axhline(.9,c='gray',ls=':');fig.savefig(RUN/'figures/classification.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');xx=np.arange(3)
    for dx,key,label,color in [(-.25,'all_candidates','Any candidate / pointGT','#95a9bc'),(0,'constraint','GT-valid pool / pointGT','#a68bb9'),(.25,'accepted','Learned accepted pool / pointGT','#ddb363')]:ax.bar(xx+dx,[q['pools'][key]['opportunities_over1px'] for q in probe['rows']],width=.25,label=label,color=color)
    ax.set(xticks=xx,xticklabels=['Seed1','Seed2','Seed3'],ylabel='Points with >1px potential gain over fallback',title='Candidate coverage is lost before residual ranking');ax.legend(fontsize=8);fig.savefig(RUN/'figures/opportunities.png',dpi=150);plt.close(fig);cases=gallery(p)
    classmd=['| 영상+기하 분류 목표 | AUROC: 시드별 | Precision: 시드별 | Recall: 시드별 |','|---|---|---|---|']+[f'| {q["label"]} | '+', '.join(f'{x:.3f}' for x in q['auroc'])+' | '+', '.join(f'{100*x:.1f}%' for x in q['precision'])+' | '+', '.join(f'{100*x:.1f}%' for x in q['recall'])+' |' for q in classrows]
    tablemd=['| 유효성 입력·목표 | 후보 전 선별 | 후보 전 선별+점 어려움 | 기존 선택 후 거절+점 어려움 |','|---|---:|---:|---:|']+[f'| {q["label"]} | {q["pre_all"]:.3f} | {q["pre_hard"]:.3f} | {q["post_hard"]:.3f} |' for q in rows]
    oracle=[dict(key=k,label=label,value=value) for k,label,value in [('fallback','기존 직접 회귀',r['fallback']['mean_px']),('primary','실제 유효성 선택',r['constraint_visual_pre_all']['mean_px']),('GT_labels','GT유효성 + 기존 잔차 순위 [진단]',r['GT_constraint_pre_all']['mean_px']),('GT_valid_best','GT유효성 + 점GT 최선 후보 [진단]',probe['oracle_results']['constraint']['check/all/all_detected']['mean_px']),('accepted_best','실제 통과 후보 + 점GT 최선 선택 [진단]',probe['oracle_results']['accepted']['check/all/all_detected']['mean_px']),('all_best','전체3후보+fallback 중 점GT 최선 [진단]',r['GT_candidate']['mean_px'])]]
    oraclemd=['| 대조 | 전체 점 오차px |','|---|---:|']+[f'| {q["label"]} | {q["value"]:.3f} |' for q in oracle]
    body=['# '+TITLE,'','**유효한 선을 높은 정밀도로 고르는 것은 가능했다. 하지만 기존 직접 회귀보다 점 성능을 추가로 개선하지는 못했다. 선택기가 남긴 선은 이미 잘 맞던 점에 치우쳤고, 어려운 점을 고칠 유효 후보 대부분이 탈락했다.**','',
    '18개 HGB 분류기×60라운드=1080라운드, 감사 재학습 3개 별도. 신경망 업데이트 0. V41의 실제 선분3후보와 직접회귀/잔차/점어려움 선택기를 고정했다. 같은 2 CAD·12학습/4기준선택/8재사용확인 그룹이다. 확인 200관측 중 130검출·1040점이며 미검출 70관측은 점 오차에서 제외한다. 상류 모델과 offset은 과거 16그룹 노출 이력이 있다. 새 실사 평가, 수동 GT 수정, 렌더는 없다.','',
    '## 이번에 학습한 것','',
    '1. **가시성:** 해당 역할의 물리 경계가 보이는가.\n2. **물리 경계 정확도:** 가시 경계이고 예측선의 GT지원거리≤3px, 방향차≤10도인가.\n3. **목표점 제약 정확도:** 위 조건에 더해 실제 선+역할offset의 목표점 법선 제약 오차≤3px인가.','',
    '세 목표는 포함 관계다. 세 번째도 점을 실제로 개선한다는 정답은 아니다. 원래 오차가1px인 점에3px 허용오차의 제약을 적용하면 나빠질 수 있다. GT는 학습 감독/진단으로만 사용하고 실제 추론 입력에는 넣지 않았다. 가시지원 없음은 캐시의 역할 가시성 정의이며 모든 영상 선분이 가짜라는 뜻은 아니다.','',
    '같은 입력에서 영상 특징을 뺀 기하 대조와 영상+기하를 비교했다. 기준 선택 4그룹에서 경험적 precision≥90%, 통과 10역할 슬롯 이상·2그룹 이상인 고정 0.10~0.95 격자 중 recall이 가장 큰 기준을 선택했다. 확인셋으로 기준을 고르지 않았다. 표본 수와 상류 학습 이력 때문에 90%를 보장하는 절차는 아니다.','',
    '통과 후보를 먼저 선별해 잔차의 양의 예상이득 순으로 고르거나, 기존 선택 후 거절하는 대조를 분리했다. 거절/미선택 시 V41직접회귀 결과로 돌아간다. 이는 원래 점보다 이미 작게 개선된 10.410px이므로 단순 복귀를 새 선택기의 이득으로 세지 않는다.','',
    '## 분류는 학습 밖에서 작동했나','',*classmd,'',
    '주대조인 목표점 제약 분류기의 확인 AUROC는 .925~.939, 역할 슬롯 precision 93.7~96.7%, recall 50.1~54.1%였다. 실제 잔차 순위로 최종 선택한 점에서의 제약 precision은 96.3/91.6/98.4%로 등록한 90% 기준을 통과했다. 역할 슬롯과 최종 선택점은 서로 다른 모집단이다.','',
    '![분류 정밀도와 재현율](figures/classification.png)','',
    '## 점 성능의 추가 이득은 있었나','',
    '아래 단위는 px이며 낮을수록 좋다. 기존 점 10.463px, 복귀 기준인 V41 직접 회귀 10.410px와 비교한다.','',*tablemd,'',
    '주판정은 10.410014→10.408929px, 차이 −.001085px·95% 구간 [-.008859,+.007855]였다. 시드별 추가 개선 +.0766/−.0150/−.0301%로 두 시드가 아주 조금 악화했다. 실질적인 추가 성능 개선은 확인하지 못했다. 기존 점 대비 약 .5% 개선은 대부분 복귀 모델에서 이미 있던 효과다.','',
    '복귀 결과에서 바뀐 점은 69/57/39개다. 복귀 모델 대비 1px 넘게 악화된 점은 2/4/1개였고 조건 보존 기준은 통과했다. 분류 정밀도·조건 보존·최소 적용 수는 통과했으나, 등록한 매시드 1% 추가 성능 기준 실패로 최종 accepted=false다.','',
    '가시성만 고르면 10.593px로 악화했다. 물리 정확도 선택 10.404px는 주대조보다 평균이 조금 좋지만 추가 이득 구간이 0을 포함하고 한 시드가 악화했다. 확인 결과를 보고 이 대조를 최종 모델로 바꾸지 않는다.','',
    '## 높은 정밀도인데 개선되지 않은 이유','',
    '선택점의 복귀 오차 중앙값은 1.50~1.57px였다. 총 선택 162/167/125점 중 원래 큰 오류점은 11/17/7개였고, 큰 오류점에서 복귀 결과와 실제로 달라진 것은 2/7/1개뿐이었다. 새 선택기의 변화는 대부분 쉬운 점에서 일어났다.','',*oraclemd,'',
    '후속 정답 진단에서 1px 초과 개선 가능한 후보가 있는 점은 전체 후보에서 110/106/104개, GT 유효 제약만 남기면 78/72/67개, 학습한 선택기가 통과시킨 후보에서는 15/11/6개였다. 통과 후보 중 큰 오류를 1px 넘게 고칠 가능성은 1/0/1점뿐이었다. **주된 추가 손실은 잔차 순위를 매기기 전에 개선 가능한 후보를 탈락시키는 부분이었다.**','',
    '![개선 기회가 줄어드는 위치](figures/opportunities.png)','',
    '정답 유효성을 알려줘도 현재 잔차 순위로는 10.328px다. 같은 유효 후보 중 점 정답으로 최선을 고르면 10.148px이므로 후보 순위·이득 예측 손실도 남아 있다. 실제 통과 후보에서 점 정답으로 최선을 골라도 10.377px여서 순위만 고쳐 얻을 효과는 제한적이다. 모든 GT 수치는 비배포 진단이며 실제 달성 성능이 아니다.','',
    '전체 후보의 정답 최선 9.929px와 GT 유효 후보 10.148px의 차이를 물리 정보로 달성 가능한 이득으로 해석하지 않는다. 전자는 물리적으로 틀린 선이 우연히 점을 가까이 움직이는 후보도 고를 수 있다. 이 실험은 직접 회귀로 복귀하므로, V41의 원래 점을 포함한 oracle과 비교 기준·후보 구성이 다르다.','',
    '## 실제 영상 사례','',
    '고정 seed1에서 각 사례 조건의 중앙값을 골랐다. 파랑은 V41 직접 회귀 복귀점, 빨강은 이번 출력, 보라는 검사한 후보, 초록은 GT다. 주황은 합성 영상에서 실제 추출한 선분을 표시한다. attention 지도는 아니다. 누락 사례의 최선 후보 선택은 점 GT 진단이다.']
    for q in cases:body+=['',f'### {q["name"]}: {q["id"]}, point{q["point"]}','',q['rule'],'',f'유효확률{q["probability"]:.3f}/기준{q["threshold"]:.2f},GT유효{q["GT_valid"]}. 복귀{q["fallback_px"]:.2f}→실제{q["applied_px"]:.2f}px,검사후보{q["candidate_px"]:.2f}px.','',f'![{q["name"]}](figures/{q["name"]}.png)']
    body+=['','## 현재 판단과 다음 범위','',
    '이번 유효성 분류는 배경·다른 역할의 제약을 구분할 신호가 있음을 보였으며, 좋은 점을 보존할 구성 요소의 가능성이 있다. 그러나 추가 점 정확도 향상을 만드는 모듈로 채택하지 않는다. 신뢰할 수 있으면서 현재 점을 개선할 수 있는 제약을 놓치지 않는 문제가 남았다.','',
    '다음에는 이미 GT상 유효하고 개선 가능한 후보가 왜 탈락하는지, 큰 오류 조건에서의 재현율·확률 보정과 기대 피해·이득을 분리해 검증해야 한다. 현재 확인셋을 보고 단순히 기준을 낮추면 새 성능 검증이 아니다. 새 장면에서 기준을 고정한 재현과 효용 검증이 필요하다. 이 후속 학습·독립 렌더는 이번에 실행하지 않았다.','',
    '## 검증과 재현','',
    '18분류기 예측 재현, 93점 배열의 조건별 재계산, 3개 비학습 영역 교란 재학습을 통과했다. check 정답·점수를 교란해도 calibration 기준이 바뀌지 않았고, 실제 입력·선택은 GT와 독립이다. 세 정답의 포함 관계와 V41 기하식 일치를 검증했으며 후속 정답 후보 최소 오차·개선 기회 수도 별도로 재계산했다. 이전 완료 파일은 보존했다.','',
    '재현: `scripts/research/pallet_constraint_validity_v42/README.md`. 일차 PROTOCOL/SUMMARY와 후속 PROBE_PROTOCOL/PROBE를 분리했다. 3시드를 고정하고 원본 장면 그룹 단위 5000회 bootstrap으로 구간을 계산했다. 독립 실사·새 CAD·신경망 향상 검증은 아니다.','']
    text='\n'.join(body);(RUN/'REPORT_KO.md').write_text(text);NOTE.write_text(text.replace('](figures/','](../../../../data/pallet/results/pallet_constraint_validity_v42/figures/'));html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px/1.75 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#24364b}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #bccada;padding:8px}th{background:#edf3f8}</style>'+html+'</html>');write(RUN/'FIGURE_DATA.json',dict(classification=classrows,table=rows,oracle=oracle));write(RUN/'CASES.json',dict(rows=cases));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(RUN/'SUMMARY.json'),probe_sha256=sha(RUN/'PROBE.json')));print('REPORT42 COMPLETE',flush=True)
if __name__=='__main__':main()
