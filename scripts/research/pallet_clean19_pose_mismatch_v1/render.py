"""Posthoc figures and Korean report from immutable diagnostics only."""
from collections import Counter,defaultdict
import html
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import diagnose as D

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]

def f(v,n=4):return '—' if v is None else f'{v:.{n}f}'

def png(name):
    plt.tight_layout();plt.savefig(D.DOC/'figures'/name,dpi=160);plt.close()

def plot_all(fr,tt,oracle,rep,pair):
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,arm in zip(axs,['S1','S2']):
        rr=tt['MODERATE_OCCLUSION'][arm]['rows'];ok=[r for r in rr if r['ADD_delta'] is not None]
        ax.axhline(0,c='gray');ax.axvline(0,c='gray')
        ax.scatter([r['PCK10_count_delta'] for r in ok],[r['ADD_delta'] for r in ok],c=['crimson' if r['PCK_UP_AND_POSE_DOWN'] else 'steelblue' for r in ok],alpha=.7)
        ax.set(xlabel='PCK10 correct-corner count change',ylabel='Normalized ADD change (lower is better)',title=f'S0 -> {arm}; red = PCK up / ADD worse')
    png('moderate_pck_vs_add_scatter.png')
    for sev,filename in [(D.SEVS[1],'axis_transition_moderate.png'),(D.SEVS[2],'axis_transition_severe.png')]:
        fig,axs=plt.subplots(1,2,figsize=(9,3.8))
        for ax,arm in zip(axs,['S1','S2']):
            c=tt[sev][arm]['axis_counts'];matrix=np.array([[c.get(f'{a}->{b}',0) for b in [True,False]] for a in [True,False]])
            ax.imshow(matrix,cmap='Blues',vmin=0)
            for i in range(2):
                for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='darkred',fontsize=20)
            ax.set(xticks=[0,1],xticklabels=['Correct','Wrong'],yticks=[0,1],yticklabels=['Correct','Wrong'],xlabel=arm,ylabel='S0',title=sev)
        png(filename)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,sev in zip(axs,D.SEVS[1:]):
        xx=np.arange(3)
        for i,k in enumerate(['CURRENT','ORACLE_WD','ORACLE_AXIS']):ax.bar(xx+(i-1)*.25,[oracle['groups'][sev][a][k]['ADDsym_AUC'] for a in D.STUDENTS],width=.25,label=k)
        ax.set(xticks=xx,xticklabels=D.STUDENTS,ylabel='ADDsym AUC',title=sev);ax.legend(fontsize=8)
    fig.suptitle('POSTHOC GT ORACLE — NONDEPLOYABLE',fontsize=12);png('selector_oracle_headroom.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,sev in zip(axs,D.SEVS[1:]):
        values=[];labels=[]
        for a in D.STUDENTS:
            for b in [True,False]:
                values.append([r['score_delta_normalized'] for r in fr[a].values() if r['severity']==sev and r['current_pose']['metric'].get('axis_correct')==b and r['score_delta_normalized'] is not None]);labels.append(a+(' correct' if b else ' wrong'))
        ax.boxplot(values,labels=labels,showfliers=False);ax.tick_params(axis='x',rotation=35);ax.set(title=sev,ylabel='Normalized score margin (no new threshold)')
    png('selector_score_margin.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for i,arm in enumerate(D.STUDENTS):
        rr=[r for r in rep['corner_summary'] if r['model']==arm]
        axs[0].bar(np.arange(8)+(i-1)*.25,[r['axis_recovered'] for r in rr],width=.25,label=arm)
        axs[1].plot(range(8),[r['ADD_improves']/max(r['trials'],1) for r in rr],marker='o',label=arm)
    axs[0].set(title='GT replacement: wrong -> correct axis',xlabel='Corner',ylabel='Recovery count');axs[1].set(title='GT replacement: ADD improves / trials',xlabel='Corner',ylabel='Fraction')
    for ax in axs:ax.legend()
    fig.suptitle('NONDEPLOYABLE; LOO unavailable: production selector rejects missing corners');png('corner_influence.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,sev in zip(axs,D.SEVS[1:]):
        for i,arm in enumerate(D.STUDENTS):
            vals=[next(r['angle_median'] for r in pair['summary'] if r['severity']==sev and r['model']==arm and r['family']==fam and r['axis_subset'] is None) for fam in ['LR','TB_vertical','FR']]
            ax.bar(np.arange(3)+(i-1)*.25,vals,width=.25,label=arm)
        ax.set(xticks=range(3),xticklabels=['LR','Vertical','FR'],title=sev,ylabel='Native edge direction error median (deg)');ax.legend()
    png('pair_geometry.png')

def draw_edges(im,xy,color,dashed=False):
    if xy is None:return
    for a,b in D.EDGES:
        if not np.isfinite(xy[[a,b]]).all():continue
        u,v=xy[a],xy[b]
        # clip line endpoints to prevent overflow in pathological projections
        u=np.clip(u,-10000,10000);v=np.clip(v,-10000,10000)
        if dashed:
            for t in np.arange(0,1,.1):cv2.line(im,tuple((u+(v-u)*t).astype(int)),tuple((u+(v-u)*min(t+.05,1)).astype(int)),color,1,cv2.LINE_AA)
        else:cv2.line(im,tuple(u.astype(int)),tuple(v.astype(int)),color,2,cv2.LINE_AA)

def projection(p,K):
    if not p['available']:return None
    R=np.array(p['R_cf']);rv,_=cv2.Rodrigues(R)
    return cv2.projectPoints(D.Pose.cuboid(*p['cf_extents']),rv,np.array(p['centroid']),K,None)[0].reshape(8,2)

def montage(item,frames,data):
    fid=item['frame_id'];record=next(r for r in data['records'] if r['id']==fid)
    im=cv2.imread(str(D.ROOT/record['image']['path']));h,w=im.shape[:2];scale=640/w;height=round(h*scale)
    image=cv2.resize(im,(640,height));K=np.array(data['meta'][fid]['K']);target=np.array(data['truth'][fid]['gt']);panels=[]
    for arm in ['RGB','S0','S1','S2']:
        view=image.copy();lines=[arm, fid,record['severity']+' / '+record['object_type']]
        if arm=='RGB':lines+=['Original RGB (no prediction)','Yellow = predicted 2D corners','Red = selected pose projection','Cyan dashed = alternate pose projection','Green crosses = evaluation reference','Axis = W/D extents parity only','ORACLE is NONDEPLOYABLE']
        else:
            r=frames[arm][fid];p=r['current_pose'];m=p['metric'];q=D.points(data['preds'][arm][fid]);sr=r['selector']
            if q is not None:draw_edges(view,q*scale,(0,220,255))
            if m['available']:draw_edges(view,projection(p,K)*scale,(40,60,255))
            alt=next((a for a in r['hypotheses'] if a['name']!=sr['selected_hypothesis']),None)
            if alt and alt['pose']['available']:draw_edges(view,projection(alt['pose'],K)*scale,(255,240,0),True)
            for j in range(8):
                if data['truth'][fid]['valid'][j]:cv2.drawMarker(view,tuple((target[j]*scale).astype(int)),(20,255,20),cv2.MARKER_CROSS,9,1)
            lines += [f"PCK10: {r['PCK_counts']['10']}/{r['valid_corner_count']} | axis: {m.get('axis_correct')}",
                'ADDnorm '+f(m.get('ADDsym_normalized'))+' | IoU '+f(m.get('IoU3D')),
                'R/yaw/t(cm): '+ '/'.join(f(m.get(k),2) for k in ['rotation_deg','yaw_deg','translation_cm']),
                'Selected: '+str(sr['selected_hypothesis']),
                'Score selected/alternate: '+f(next((a['selector_score'] for a in r['hypotheses'] if a['name']==sr['selected_hypothesis']),None),3)+' / '+f(alt['selector_score'] if alt else None,3),
                'Margin norm: '+f(r['score_delta_normalized'],4),
                'POSTHOC ORACLE ONLY: alternate ADD better' if r['alternate_ADD_better'] else 'Alternate shown, not selected using GT']
            # Same reference-derived display ROI for all models, never used in inference.
            box=np.array(data['truth'][fid]['box'])*scale
            margin=max(25,.2*max(box[2]-box[0],box[3]-box[1]))
            x0,y0=np.maximum(0,np.floor(box[:2]-margin)).astype(int)
            x1,y1=np.minimum([640,height],np.ceil(box[2:]+margin)).astype(int)
            crop=view[y0:y1,x0:x1]
            if crop.size:
                zoom=min(640/crop.shape[1],height/crop.shape[0]);cw=round(crop.shape[1]*zoom);ch=round(crop.shape[0]*zoom)
                view=np.full_like(view,24);view[(height-ch)//2:(height-ch)//2+ch,(640-cw)//2:(640-cw)//2+cw]=cv2.resize(crop,(cw,ch))
            cv2.putText(view,'Shared display ROI zoom',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            if not r['matched']:cv2.putText(view,'MATCH FAILURE (2D penalty)',(12,48),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
        header=np.full((240,640,3),(28,37,44),np.uint8)
        for j,line in enumerate(lines):cv2.putText(header,line,(10,20+22*j),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1,cv2.LINE_AA)
        panels.append(np.vstack([header,view]))
    dest=D.DOC/'figures'/item['file'];assert cv2.imwrite(str(dest),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))

def case_selection(tt):
    cases=[]
    bad={}
    for arm in ['S1','S2']:
        for r in tt['MODERATE_OCCLUSION'][arm]['rows']:
            if r['PCK_UP_AND_POSE_DOWN']:bad[r['frame_id']]=max(bad.get(r['frame_id'],0),r['ADD_delta'])
    # Include all mismatches, not only the most visually persuasive examples.
    for i,(fid,delta) in enumerate(sorted(bad.items(),key=lambda x:(-x[1],x[0]))):cases.append(dict(frame_id=fid,kind='moderate_pck_up_pose_down',file=f'case_moderate_pck_up_pose_down_{i:02d}.jpg',selection='all unique moderate PCK-up ADD-down',delta=delta))
    recovered={}
    for arm in ['S1','S2']:
        for r in tt['MODERATE_OCCLUSION'][arm]['rows']:
            if r['ADD_delta'] is not None and r['ADD_delta']<0:recovered[r['frame_id']]=min(recovered.get(r['frame_id'],0),r['ADD_delta'])
    for i,(fid,delta) in enumerate(sorted(recovered.items(),key=lambda x:(x[1],x[0]))[:4]):cases.append(dict(frame_id=fid,kind='moderate_pose_recovered',file=f'case_moderate_pose_recovered_{i:02d}.jpg',selection='largest ADD improvements, posthoc',delta=delta))
    flips=sorted({r['frame_id'] for a in ['S1','S2'] for r in tt['SEVERE_OCCLUSION'][a]['rows'] if r['axis_delta'] in [-1,1]})
    for i,fid in enumerate(flips):cases.append(dict(frame_id=fid,kind='severe_axis_flip',file=f'case_severe_axis_flip_{i:02d}.jpg',selection='all severe axis transitions, includes harm and recovery'))
    return cases

def supplementary(fr,tt,rep,pair):
    # Separate change while selected axis remains correct from parity flips.
    stable=[];reference=[];edge=[]
    for sev in D.SEVS:
        for arm in ['S1','S2']:
            for subset in ['True->True','False->False','True->False','False->True']:
                rr=[r for r in tt[sev][arm]['rows'] if r['axis_transition']==subset]
                stable.append(dict(severity=sev,model=arm,axis_transition=subset,n=len(rr),ADD_improves=sum(r['ADD_delta']<0 for r in rr),ADD_worsens=sum(r['ADD_delta']>0 for r in rr),ADD_delta_median=float(np.median([r['ADD_delta'] for r in rr])) if rr else None))
            for subset in ['manual_only','mixed_documented','has_unknown']:
                rr=[r for r in tt[sev][arm]['rows'] if fr[arm][r['frame_id']]['provenance']['subset']==subset]
                reference.append(dict(severity=sev,model=arm,subset=subset,frames=len(rr),mismatch=sum(r['PCK_UP_AND_POSE_DOWN'] for r in rr),axis_flips=sum(r['axis_delta'] in [-1,1] for r in rr)))
            by={(r['model'],r['frame_id'],r['edge']):r for r in pair['rows'] if r['severity']==sev}
            for a,b in D.EDGES:
                name=f'{a}-{b}';deltas=[];joint=0
                for r in tt[sev][arm]['rows']:
                    x=by.get(('S0',r['frame_id'],name));y=by.get((arm,r['frame_id'],name))
                    if x and y and x['direction_error_deg'] is not None and y['direction_error_deg'] is not None:
                        delta=y['direction_error_deg']-x['direction_error_deg'];deltas.append(delta);joint+=r['PCK10_count_delta']>0 and delta>0
                edge.append(dict(severity=sev,model=arm,edge=name,n=len(deltas),direction_delta_median=float(np.median(deltas)) if deltas else None,PCK_up_direction_worse=int(joint)))
    recovery=[]
    for arm in D.STUDENTS:
        rr=[r for r in rep['rows'] if r['model']==arm];wrong={r['frame_id'] for r in rr if r['base'].get('axis_correct') is False}
        recovered={r['frame_id'] for r in rr if r['axis_recovered']}
        recovery.append(dict(model=arm,wrong_frames_with_trials=len(wrong),recoverable_by_any_single_reference_corner=len(recovered),not_recovered=len(wrong-recovered)))
    return dict(stable_axis=stable,reference_subsets=reference,pair_transitions=edge,any_corner_recovery=recovery)

def main():
    assert D.read(D.DOC/'ANALYSIS_COMPLETE.json')['new_training']==0
    (D.DOC/'figures').mkdir(exist_ok=False)
    data=D.load();fr=D.read(D.DOC/'FRAME_DIAGNOSTICS.json');tt=D.read(D.DOC/'TRANSITIONS.json');oracle=D.read(D.DOC/'ORACLE_WD_AUDIT.json');rep=D.read(D.DOC/'GT_REPLACEMENT_ORACLE.json');pair=D.read(D.DOC/'PAIR_GEOMETRY_AUDIT.json')
    sup=supplementary(fr,tt,rep,pair);D.save(D.DOC/'SUPPLEMENTARY.json',sup)
    plot_all(fr,tt,oracle,rep,pair)
    cases=case_selection(tt)
    for i,fid in enumerate(rep['random_controls']):cases.append(dict(frame_id=fid,kind='random_control',file=f'case_random_control_{i:02d}.jpg',selection='seed20260922 six moderate frames before perturbation'))
    for case in cases:montage(case,fr,data)
    D.save(D.DOC/'CASE_SELECTION.json',cases)
    write_report(data,fr,tt,oracle,rep,pair,sup,cases)
    for b in D.read(D.DOC/'INPUT_BINDINGS.json')['files']:D.C.verify(b)
    print('REPORT_READY',len(cases),'case images + 7 charts',flush=True)

def write_report(data,fr,tt,oracle,rep,pair,sup,cases):
    hs=D.read(D.DOC/'HYPOTHESIS_AUDIT.json');moderate=oracle['groups'][D.SEVS[1]];severe=oracle['groups'][D.SEVS[2]]
    assert moderate['S2']['ORACLE_WD']['ADDsym_AUC']<moderate['S0']['ORACLE_WD']['ADDsym_AUC']
    corner=Counter()
    for r in rep['corner_summary']:corner[r['corner']]+=r['axis_recovered']
    topcorners=corner.most_common(3)
    topedges=sorted([r for r in sup['pair_transitions'] if r['severity']==D.SEVS[1]],key=lambda r:-r['PCK_up_direction_worse'])[:4]
    decision=dict(PRIMARY_BOTTLENECK='CASE B — KEYPOINT GEOMETRY BOTTLENECK (MODERATE; not candidate-absence claim)',
        SECONDARY_BOTTLENECK='CASE A — W/D SELECTOR BOTTLENECK (especially SEVERE)',
        qualifier='MIXED; local corner sensitivity observed, true LOO blocked; geometry-derived reference limits physical causality',
        NEXT_ONE_EXPERIMENT='Independent-session paired audit with existing checkpoints: validate corner and W/D references independently, then compare S0/S1/S2 on the same frozen frames; no training/selector tuning. Not executed.',
        NEW_TRAINING=0,OPTIMIZER_STEPS=0,LOO='BLOCKED_CURRENT_SELECTOR_REQUIRES_ALL9_FINITE',
        MODERATE=moderate,SEVERE=severe,TOP_INFLUENTIAL_CORNERS=topcorners,TOP_INFLUENTIAL_EDGES=topedges,
        PCK_UP_POSE_DOWN={s:{a:tt[s][a]['PCK_UP_AND_POSE_DOWN'] for a in ['S1','S2']} for s in D.SEVS})
    D.save(D.DOC/'DECISION.json',decision)
    lines=['# CLEAN19 EASY→HARD: 2D 개선이 6D로 연결되지 않는 이유','',
        '## 1. 한 줄 결론','',
        '**중간 가림의 하락은 W/D 선택 오류만으로 설명되지 않는다. 두 후보에서 정답 기반 최선을 골라도 S1/S2의 ADD가 S0보다 낮아, 예측점의 상대 배치·pose fit 문제가 남는다. 심한 가림에서는 W/D 선택 오류의 영향이 더 크다.**','',
        'PRIMARY_BOTTLENECK: **KEYPOINT GEOMETRY (MODERATE)**. SECONDARY_BOTTLENECK: **W/D SELECTOR (특히 SEVERE)**. 원인 개입으로 증명한 인과가 아니라 동결 예측의 진단이다.','',
        '**새 학습 0 / optimizer step 0 / checkpoint·GT·선택기 수정 0.** LOO는 기존 selector가 finite `(9,2)`만 받아 1점 제거를 거부하므로 수행 불가로 기록했다. 0회 영향이라고 해석하지 않는다. 나머지 후보·정답교체 분석은 완료했다.','',
        '## 2. 기존 결과 요약','',
        '공식 RESULTS.json을 읽어 재현. 6모델×300장의 2D·pose 및 모든 material/session group과 common-matched supplement가 허용오차 1e-7 이내 일치했다. PCK10은 전체 유효 코너 분모, Axis는 pose-available 조건부, ADD AUC는 실패 포함 전체 이미지 분모, IoU는 pose-available 중앙값이다.','']
    rows=[]
    for s in D.SEVS:
        for a in D.STUDENTS:
            g=data['results']['groups'][s][a];two=g['twoD'];six=g['sixD']
            rows.append([s,a,f(100*two['PCK']['10'],2),f(100*six['axis_accuracy'],2),f(six['ADDsym_AUC']),f(six['IoU3D']['median'])])
    lines+=table(['난도','모델','PCK10 %','Axis %','ADD AUC','IoU3D med'],rows)
    lines+=['','[전체 6모델·종류·세션·common-matched 원본 수치](OFFICIAL_RESULTS_SNAPSHOT.json) · [재현 검사](PARITY.json)','',
        '## 3. 왜 이 분석이 필요한가','',
        'PCK는 각 점이 10px 안에 드는지 세며 2D 대칭을 최소화한다. PnP는 동일 점들의 상대 배치 전체와 native camera-facing ID를 사용한다. 일부 점이 11→9px가 되어 PCK가 올라가도 다른 점·중심 P8·변 방향의 변화로 6D 오차는 커질 수 있다. 원근 투영과 치수 결합 때문에 독립적인 점 오차와 6D 오차는 단조 관계가 아니다.','',
        '**중요한 정의 한계:** 현재 AxisAcc는 `predicted cf width == reference body width`라는 W/D parity 검사다. 전체 회전이 맞다는 뜻이 아니다. 두 W/D 후보가 모두 풀리면 그중 하나가 이 기준을 만족하는 것은 구조상 당연하다. 따라서 `alternate axis-correct 존재`를 `좋은 6D 후보 존재`로 부르지 않고 ADD·회전·IoU를 함께 보았다.','',
        '## 4. W/D hypothesis 분해','', '**POSTHOC GT ORACLE — NONDEPLOYABLE**. Oracle W/D는 ADD 최소 후보, Oracle Axis는 parity-correct 후보 중 ADD 최소(없으면 전체 ADD 최소), 동률은 후보 이름순이다. GT는 지표·oracle에만 쓰며 실제 selector에는 넣지 않았다.','']
    rows=[]
    for s in D.SEVS[1:]:
        for a in D.STUDENTS:
            x=oracle['groups'][s][a]
            rows.append([s,a,x['axis_wrong'],x['category_counts'].get('ALTERNATE_GOOD_SELECTOR_WRONG',0),x['alternate_ADD_better'],x['category_counts'].get('BOTH_HYPOTHESES_BAD_AXIS',0),f(x['CURRENT']['ADDsym_AUC']),f(x['ORACLE_WD']['ADDsym_AUC']),f(x['ORACLE_AXIS']['ADDsym_AUC'])])
    lines+=table(['난도','모델','현재 parity 오류','대안 parity 정답','그중 ADD도 개선','둘 다 parity 오답','현재 ADD AUC','W/D oracle AUC','Axis oracle AUC'],rows)
    lines+=['','![후보 선택 oracle 상한](figures/selector_oracle_headroom.png)','',
        '둘 다 parity 오답 0은 geometry 문제가 없다는 증거가 아니다. 중간 가림의 **S2 oracle ADD '+f(moderate['S2']['ORACLE_WD']['ADDsym_AUC'])+'는 S0 oracle '+f(moderate['S0']['ORACLE_WD']['ADDsym_AUC'])+'보다 낮다.** 축만 항상 맞게 해도 S2 ADD는 '+f(moderate['S2']['ORACLE_AXIS']['ADDsym_AUC'])+'에 그친다. 현재 폭/깊이 선택만 완벽하게 해서는 이 결과를 충분히 복구하지 못한다.','',
        '정상/비정상 pose를 새 임의 임계값으로 나누지 않았다. SELECTED_AXIS_CORRECT_DISTRIBUTION_REQUIRED 범주에서 ADD·R·t·IoU 분포를 별도로 제공하며, S0 대비 악화 여부도 계산했다.','',
        '## 5. MODERATE에서 어디서 깨지는가','',
        'PCK_UP_AND_POSE_DOWN은 S0보다 PCK10 정답 코너 수가 늘고 normalized ADD가 증가한 경우로 고정했다. S1 '+str(tt[D.SEVS[1]]['S1']['PCK_UP_AND_POSE_DOWN'])+'건, S2 '+str(tt[D.SEVS[1]]['S2']['PCK_UP_AND_POSE_DOWN'])+'건이다. 중복 프레임은 모델별 횟수와 구분했다.','',
        '![PCK와 ADD 변화](figures/moderate_pck_vs_add_scatter.png)','',
        '![중간 가림 축 전이](figures/axis_transition_moderate.png)','']
    rows=[[r['model'],r['axis_transition'],r['n'],r['ADD_improves'],r['ADD_worsens'],f(r['ADD_delta_median'])] for r in sup['stable_axis'] if r['severity']==D.SEVS[1]]
    lines+=table(['모델','S0→학생 Axis','건수','ADD 개선','ADD 악화','ADD 변화 med'],rows)
    lines+=['','이미지: 좌상 RGB, 우상 S0, 좌하 S1, 우하 S2. **노랑=예측 2D / 빨강=선택 pose 재투영 / 청록 점선=대안 pose / 초록 십자=evaluation reference**. Oracle 표시가 있어도 실제 선택 결과를 바꾸지 않았다. MATCH FAILURE의 800px는 벌점이며 800px짜리 이동선을 그리지 않았다.']
    for c in cases:
        if c['kind'].startswith('moderate'):lines+=['',f"### {c['kind']} · {c['frame_id']}",'',f"![{c['frame_id']}](figures/{c['file']})"]
    lines+=['','## 6. SEVERE와 무엇이 다른가','',
        '심한 가림에서는 alternate가 parity뿐 아니라 ADD도 개선하는 비율이 더 높다. W/D oracle headroom이 중간 가림보다 크다. S2의 oracle ADD AUC는 '+f(severe['S2']['ORACLE_WD']['ADDsym_AUC'])+'로, 현재 '+f(severe['S2']['CURRENT']['ADDsym_AUC'])+'보다 높다. 이는 정답을 사용하는 진단 상한이며 실제 성능이 아니다. 심한 가림81장은 모두 플라스틱이어서 난도 차이에 재질 차이도 섞인다.','',
        '![심한 가림 축 전이](figures/axis_transition_severe.png)','']
    for c in cases:
        if c['kind']=='severe_axis_flip':lines+=['',f"![{c['frame_id']}](figures/{c['file']})"]
    lines+=['','## 7. 어떤 corner/pair가 pose에 민감한가','',
        '**LOO 미수행:** 중간 가림87×3×8 입력에서 한 점을 NaN으로 제거해 기존 selector와 pose.infer를 호출했다. 남은 코너 수가 6 이상이어도 finite9 입력 계약에 막힌다. 제거점을 추정해서 채우거나 mask-aware selector를 새로 만들지 않았다. 따라서 LOO 영향 count는 null이다.','',
        '**GT_REPLACEMENT_ORACLE_NONDEPLOYABLE:** axis wrong / PCK-up ADD-down과 deterministic random6개를 포함한 subset에서 유효 reference 코너 한 개만 native ID로 치환했다. 2D symmetry best-branch로 재배열하지 않았다. 정답교체 '+str(len(rep['rows']))+'회. projected/unknown 점은 물리적 원인 확정 근거가 아니다.','',
        '![정답교체 코너 민감도](figures/corner_influence.png)','']
    lines+=table(['모델','오답 프레임(교체 가능)','한 코너로 parity 복구 가능','어느 한 코너로도 미복구'],[[r['model'],r['wrong_frames_with_trials'],r['recoverable_by_any_single_reference_corner'],r['not_recovered']] for r in sup['any_corner_recovery']])
    lines+=['','P5/P6 등에서 민감도 신호가 있지만 코너마다 유효 trial 수가 다르고 같은 프레임·모델의 반복 측정이다. 복구 횟수를 독립 표본이나 그 점만 학습하면 해결된다는 증거로 쓰지 않는다.','',
        '![edge family 방향 오차](figures/pair_geometry.png)','',
        '12 cuboid edges를 LR / vertical(TB) / front–rear(FR)로 나눴다. native ID 방향을 유지해 반전도 오차로 남겼다. [PAIR_GEOMETRY_AUDIT](PAIR_GEOMETRY_AUDIT.json)에 벡터·길이·각도 및 front/rear/vertical 분류, [SUPPLEMENTARY](SUPPLEMENTARY.json)에 PCK 상승과 변 방향 악화가 함께 난 횟수를 기록했다.','',
        '학습의 코너·edge별 실제 가림 횟수는 [고정 augmentation 빈도](AUGMENTATION_FREQUENCY_SNAPSHOT.json)에서 확인할 수 있다. 학습 및 평가의 그룹 빈도를 병치하는 기술적 연관일 뿐 인과를 확정하지 않는다.','',
        '## 8. selector score가 무엇을 보고 틀리는가','',
        '현재 점수는 reprojection + cheirality + projected ordering invariants + upright + spread penalty다. 두 후보는 같은 입력이므로 spread는 동일하며, 둘 다 invariant=0인 경우 이 항이 후보 구별에 기여하지 못한다. 가중치·동률 허용오차는 수정하지 않았다.','']
    rows=[]
    for s in D.SEVS[1:]:
        for a in D.STUDENTS:
            x=hs[s][a]['False'];rows.append([s,a,x['n'],x['wrong_lower_reprojection'],x['both_zero_invariants'],f(x['components']['score_delta_normalized']['median'])])
    lines+=table(['난도','모델','parity 오답','선택 후보의 재투영오차 더 낮음','두 후보 invariant=0','오답 margin med'],rows)
    lines+=['','![축 정오답별 점수 간격](figures/selector_score_margin.png)','',
        '오답 후보가 더 낮은 재투영오차를 내므로 낮은 residual은 실제 자세 정답의 증거가 아니다. 다만 더 낮아서 선택됐다는 인과는 다른 penalty까지 같이 확인해야 하며, 숫자 비교만으로 가중치 튜닝 근거를 만들지 않았다. 현재 selector는 fit은8코너로 하지만 residual score는 **중심 P8까지 9점**을 사용한다. 중심 영향의 별도 개입은 이번 0..7 코너 분석 범위 밖이다.','',
        '## 9. reference 한계','',
        'pose reference는 기존 annotation+치수의 기하 복원이며 독립 측정이 아니다. 구체적 corner source를 그대로 부착했다. unknown은 GT가 틀렸다는 뜻이 아니고 object-level manual 표기를 corner-level manual로 승격하지 않았다.','']
    counts=Counter(r['provenance']['subset'] for r in fr['S0'].values())
    lines+=table(['출처 그룹','300장 중 수'],[[k,counts.get(k,0)] for k in ['manual_only','mixed_documented','has_unknown']])
    lines+=['','8코너 manual-only 프레임이 없으므로 독립 고신뢰 pose subset 성능은 제공할 수 없다. mixed_documented도 독립 pose GT가 아니며 PnP 보완점이 섞여 있다.','']
    lines+=table(['난도','모델','출처 subset','프레임','PCK↑ ADD↓','axis flip'],[[r['severity'],r['model'],r['subset'],r['frames'],r['mismatch'],r['axis_flips']] for r in sup['reference_subsets'] if r['severity']==D.SEVS[1]])
    lines+=['','## 10. 객관적 판정','',
        '- 중간 가림: 선택 오류는 존재하지만 oracle로도 S0 대비 하락이 남는다. 예측점 배치/pose-fit 병목을 우선한다. 두 가설 모두 parity 오답이라는 근거로 내린 판정은 아니다.','- 심한 가림: 좋은 parity+ADD 대안의 존재와 oracle 여유가 더 뚜렷해 선택기 병목이 중요하다.','- 코너 교체로 선택이 달라지는 민감도는 확인됐지만 LOO 효과와 물리적 정답 복구는 미확정이다.','- 전체를 global ambiguity 하나나 reference 오류 하나로 환원할 증거는 없다. 단일 seed·같은 세션 재사용 DEV이다.','',
        '## 11. 다음 딱 한 실험','',
        '**기존 모델을 동결한 새 세션 독립 reference audit 1회**를 제안한다. 새 세션에서 clean/moderate/severe를 미리 정한 동일 수로 수집하고, 예측을 가린 상태에서 두 검토자가 corner 정의·W/D parity를 확인한다. 판단이 안 되는 사례는 따로 표시하되 삭제하지 않는다. S0/S1/S2와 고정 selector를 동일 프레임에 적용해 2D·W/D·ADD의 전이를 재확인한다. 기하 복원 6D는 독립 실측으로 부르지 않는다. 모델·selector 튜닝 없음. **이번 실행에서는 수행하지 않았다.**','',
        '## 12. 재현 정보','',
        'HEAD_BEFORE: `'+D.read(D.DOC/'INPUT_BINDINGS.json')['head']+'`. branch main. 이 보고서가 포함된 최종 commit SHA는 push 완료 stdout에서 제공한다.','',
        '[입력 hash](INPUT_BINDINGS.json) · [현재 metric parity](PARITY.json) · [전체 프레임 진단](FRAME_DIAGNOSTICS.json) · [전이](TRANSITIONS.json) · [선택기 성분](HYPOTHESIS_AUDIT.json) · [oracle](ORACLE_WD_AUDIT.json) · [LOO 제약](LOO_CORNER_INFLUENCE.json) · [정답교체](GT_REPLACEMENT_ORACLE.json) · [검사](AUDIT.json)','',
        '처음 parity 비교 helper가 JSON null을 np.allclose에 전달해 TypeError로 중단됐다. 숫자 불일치가 아니라 자료형 처리 오류이며 재귀 비교로 정정했다. STOP.json을 보존했고 입력 hash 확인 후 재개했다. 학습·원본·공식 결과 변경은 없다.','',
        '## 고정 무작위 대조 이미지','', '중간 가림에서 seed20260922로 선택한 6개. 좋고 나쁜 사례의 편향을 줄이기 위해 극단 사례와 함께 표시한다.']
    for c in cases:
        if c['kind']=='random_control':lines+=['',f"![{c['frame_id']}](figures/{c['file']})"]
    lines+=['','## 보충: 현재·oracle의 회전/이동/IoU 전체 비교','',
        '**POSTHOC GT ORACLE — NONDEPLOYABLE**. 아래 모든 oracle은 실제 추론 성능이 아니다.','']
    lines+=table(['난도','모델','선택 규칙','Axis %','ADD AUC','R med°','Yaw med°','t med cm','IoU med'],[
        [s,a,k,f(100*x[k]['axis_accuracy'],2),f(x[k]['ADDsym_AUC']),f(x[k]['rotation_deg']['median'],2),f(x[k]['yaw_deg']['median'],2),f(x[k]['translation_cm']['median'],2),f(x[k]['IoU3D']['median'])]
        for s in D.SEVS[1:] for a,x in oracle['groups'][s].items() for k in ['CURRENT','ORACLE_WD','ORACLE_AXIS']])
    lines+=['','## 보충: 점수 성분·학습 가림 빈도','',
        '아래는 선택된 후보의 성분 중앙값이다. selector margin 분포 q10/q90와 pose 분포는 HYPOTHESIS_AUDIT에 있다. spread는 두 후보에 공통이고, upright는 soft-min 아래로 내려갈 때만 실제 penalty가 붙는다.','']
    keys=['reprojection_rmse_px','cheirality_fraction','invariant_violations','upright_alignment','spread_ratio']
    lines+=table(['난도','모델','Axis','n','RMSE med','cheirality','invariant','upright','spread'],[
        [s,a,b,hs[s][a][b]['n']]+[f(hs[s][a][b]['components'][k]['median']) for k in keys]
        for s in D.SEVS[1:] for a in D.STUDENTS for b in ['True','False']])
    freq=D.read(D.DOC/'AUGMENTATION_FREQUENCY_SNAPSHOT.json')['material']
    lines+=['','S1/S2의 실제 학습 가림 occurrence 빈도(고유 이미지 수 아님):','']
    lines+=table(['재질','모델']+[f'P{j}' for j in range(8)],[[mat,a]+[freq[mat][a]['masked_channels'].get(str(j),0) for j in range(8)] for mat in freq for a in ['S1','S2']])
    lines+=['','S2에서 P5/P6 등의 가림이 잦고 한 점 교체에 따른 축 선택 변화도 나타난다. 그러나 감독 가능한 코너 분포·재질·같은 이미지 반복이 함께 섞여 있어 이 병치를 학습 가림이 오류를 만들었다는 인과로 해석하지 않는다.','',
        '아래는 중간 가림에서 PCK가 늘면서 각 변의 방향 오차가 늘어난 횟수다. 큰 수는 민감도 실험 결과가 아니라 기술적 동시발생 빈도다.','']
    lines+=table(['모델','edge','평가가능','PCK↑·방향오차↑','방향오차 변화 med°'],[[r['model'],r['edge'],r['n'],r['PCK_up_direction_worse'],f(r['direction_delta_median'],2)] for r in sup['pair_transitions'] if r['severity']==D.SEVS[1]])
    lines+=['','그림의 모델 패널은 공통 reference bbox를 **표시 목적으로만** 확대했다. RGB 패널은 전체 프레임이며 추론·metric·selector에는 이 crop을 사용하지 않았다.','']
    D.save(D.DOC/'REPORT_KO.md','\n'.join(lines)+'\n')
    summary=['# 진단 요약','',lines[4],'', 'PRIMARY: '+decision['PRIMARY_BOTTLENECK'],'','SECONDARY: '+decision['SECONDARY_BOTTLENECK'],'', 'NEW_TRAINING: 0. LOO: blocked (finite9 input contract).','']
    summary+=table(['난도','모델','PCK10 %','Axis %','ADD AUC','wrong','alternate parity-correct','both parity-bad','oracle Axis Δpp','oracle ADD Δ'],[
        [s,a,f(100*data['results']['groups'][s][a]['twoD']['PCK']['10'],2),f(100*x['CURRENT']['axis_accuracy'],2),f(x['CURRENT']['ADDsym_AUC']),x['axis_wrong'],x['category_counts'].get('ALTERNATE_GOOD_SELECTOR_WRONG',0),x['category_counts'].get('BOTH_HYPOTHESES_BAD_AXIS',0),f(100*(x['ORACLE_WD']['axis_accuracy']-x['CURRENT']['axis_accuracy']),2),f(x['ORACLE_WD']['ADDsym_AUC']-x['CURRENT']['ADDsym_AUC'])]
        for s in D.SEVS[1:] for a,x in oracle['groups'][s].items()])
    summary+=['','PCK_UP_POSE_DOWN: '+str(decision['PCK_UP_POSE_DOWN']),'','TOP_INFLUENTIAL_CORNERS (GT oracle count): '+str(topcorners),'','TOP_INFLUENTIAL_EDGES (descriptive frequency, not causality): '+str(topedges),'','NEXT_ONE_EXPERIMENT: '+decision['NEXT_ONE_EXPERIMENT'],'','[이미지 포함 전체 보고서](REPORT_KO.md)']
    D.save(D.DOC/'SUMMARY_KO.md','\n'.join(summary)+'\n')
    D.save(D.DOC/'REFERENCE_PROVENANCE.md','# Reference provenance\n\n'+ '\n'.join(table(['subset','count'],counts.items()))+'\n\nAll pose references are geometry reconstructed, not independent measurements. Unknown does not imply incorrect. Per-corner sources are attached in FRAME_DIAGNOSTICS.json; subset transitions in SUPPLEMENTARY.json.\n')
    page='<meta charset="utf-8"><title>2D to 6D diagnosis</title><style>body{background:#16232b;color:white;font:16px sans-serif;margin:25px}img{max-width:1400px;width:100%}</style><h1>Frozen S0/S1/S2 pose mismatch</h1>'
    for c in cases:page+=f'<h2>{html.escape(c["kind"]+" / "+c["frame_id"])}</h2><img loading="lazy" src="figures/{c["file"]}">'
    D.save(D.DOC/'GALLERY.html',page)

if __name__=='__main__':main()
