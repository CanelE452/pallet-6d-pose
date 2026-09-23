"""Fixed decision rules and transparent posthoc case selection; no new fitting."""
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as E
from scripts.research.pallet_clean19_pose_mismatch_v1.render import table,f,draw_edges,projection
C=E.C;D=E.D

def plot(name,title,labels,series,ylabel):
    fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(labels));width=.8/len(series)
    for j,(label,values) in enumerate(series.items()):ax.bar(xx+(j-(len(series)-1)/2)*width,values,width,label=label)
    ax.set(xticks=xx,xticklabels=labels,title=title,ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(E.DOC/'figures'/name,dpi=150);plt.close(fig)

def montage(record,label,number,preds,poses,pm,fm,truth,meta):
    fid=record['id'];im=cv2.imread(str(E.ROOT/record['image']['path']));h,w=im.shape[:2];scale=600/w;hh=round(h*scale);original=cv2.resize(im,(600,hh));K=np.array(meta[fid]['K']);panels=[]
    for arm in ('RGB','M0','M1'):
        view=original.copy();lines=[arm+' | '+label,fid,record['severity']]
        if arm=='RGB':lines+=['Yellow: predicted 2D cuboid','Red: CURRENT selected pose','Cyan dashed: alternate W/D pose','Green: evaluation reference','ORACLE: POSTHOC ONLY / NONDEPLOYABLE']
        else:
            q=D.points(preds[arm][fid]);row=poses[arm][fid];metric=pm[arm][fid]['current'];two=fm[arm][fid]
            if q is not None:draw_edges(view,q*scale,(0,220,255))
            if row['current']['available']:draw_edges(view,projection(row['current'],K)*scale,(30,30,255))
            for hyp in row['hypotheses']:
                if hyp['name']!=row['current'].get('selected_hypothesis') and hyp['pose']['available']:draw_edges(view,projection(hyp['pose'],K)*scale,(255,230,0),True)
            for j in range(8):
                if truth[fid]['valid'][j]:cv2.drawMarker(view,tuple((np.array(truth[fid]['gt'][j])*scale).astype(int)),(0,255,0),cv2.MARKER_CROSS,11,1)
            lines += [f"PCK10 {sum(e<=10 for e in two['errors'])}/{two['corners']} | Axis {metric.get('axis_correct')}",
                'CURRENT ADDnorm '+f(metric.get('ADDsym_normalized')),
                'R/yaw/t(cm) '+ '/'.join(f(metric.get(k),2) for k in ('rotation_deg','yaw_deg','translation_cm')),
                'Selected '+str(row['current'].get('selected_hypothesis')),
                'ORACLE ADDnorm '+f(pm[arm][fid]['oracle'].get('ADDsym_normalized'))+' POSTHOC ONLY']
        header=np.full((205,600,3),25,np.uint8)
        for i,line in enumerate(lines):cv2.putText(header,line,(8,20+23*i),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,245,245),1,cv2.LINE_AA)
        panels.append(np.vstack([header,view]))
    filename=f'case_{number:02d}_{label}.jpg';assert cv2.imwrite(str(E.DOC/'figures'/filename),np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,86]);return filename

def main():
    E.immutable();results=E.read(E.DOC/'RESULTS.json')['groups'];train=E.read(E.DOC/'TRAIN_DIAGNOSTICS.json');source=E.read(E.DOC/'SYNTH_HELDOUT_RESULTS.json')['materials'];gate=E.read(E.DOC/'GPU_GRADIENT_PARITY.json');weights=E.read(E.DOC/'POSE_SENSITIVITY_WEIGHTS.json');cal=E.read(E.DOC/'LAMBDA_CALIBRATION.json')
    mod=results['MODERATE_OCCLUSION'];clean=results['CLEAN'];severe=results['SEVERE_OCCLUSION']
    candidate=mod['M1']['oracle']['ADDsym_AUC']>mod['M0']['oracle']['ADDsym_AUC'];current=mod['M1']['current']['ADDsym_AUC']>=mod['M0']['current']['ADDsym_AUC']
    cleanok=clean['M1']['twoD']['PCK']['10']>=clean['M0']['twoD']['PCK']['10']-.01
    sourceok=all(source[m]['M1']['PCK']['10']['fraction']>=source[m]['M0']['PCK']['10']['fraction']-.01 for m in E.MATS)
    synthbetter=all(train['diag_reduced'].values())
    if candidate and current and cleanok and sourceok:primary='DIAGONAL_POSE_SENSITIVITY_SIGNAL';route='POSE_SENSITIVE_POSITIVE_SIGNAL';nextone='고정 모델을 독립 새 세션에 그대로 적용하는 confirmation; 재학습·튜닝 없이 같은 지표 확인.'
    elif candidate and not current:primary='SELECTOR_BOTTLENECK_AFTER_CANDIDATE_GAIN';route='CANDIDATE_ONLY_POSITIVE';nextone='독립 TRAIN 후보 쌍으로 RGB/geometry candidate scorer를 설계하고, 고정 후보 품질과 선택 정확도를 분리 평가. 이번에는 구현하지 않음.'
    elif candidate and current:primary='RECOVERY_PRESERVATION_TRADEOFF_REMAINS';route='PRESERVATION_FAILED';nextone='같은 고정 M1 목표에 별도 기능 보존 목적을 추가하는 단일 통제 실험을 설계; 지금 실행하지 않음.'
    elif synthbetter:primary='SYNTH_REAL_GEOMETRY_TRANSFER_GAP';route='SYNTH_ONLY';nextone='평가300 밖 실사에서 신뢰 가능한 pose/relative geometry 감독을 확보·검증하는 단일 provenance 실험. 기존 DEV를 학습에 추가하지 않음.'
    else:primary='LOCAL_DIAGONAL_SENSITIVITY_INSUFFICIENT';route='HARMFUL' if not cleanok or not sourceok else 'NO_SIGNAL';nextone='고정 S1과 differentiable-PnP 목적의 단일 비교를 설계하되, 먼저 TRAIN-only 수치·기하 gradient 검증. 이번에는 실행하지 않음.'
    decision=dict(primary=primary,secondary='RECOVERY_PRESERVATION_TRADEOFF' if not(cleanok and sourceok) else 'PRESERVATION_WITHIN_PREDECLARED_1PP',route=route,candidate_improved=candidate,current_nonworse=current,clean_within1pp=cleanok,source_both_materials_within1pp=sourceok,synthetic_diag_reduced_both=synthbetter,next_one_experiment=nextone,statistical_significance_claim=False)
    E.save(E.DOC/'DECISION.json',decision)
    (E.DOC/'figures').mkdir(exist_ok=False)
    plot('01_gradient_repeatability.png','Independent fresh R0 GPU gradients',['OLD / OLD','OLD / NEW lambda0'],{'max absolute difference':[max(r.get('max_abs',0) for r in gate[k]['parameters']) for k in ('old_old','old_new')]},'Gradient difference (both zero)')
    plot('02_pose_weight_by_corner.png','Frozen diagonal weights: valid scalar mean = 1',[str(i) for i in range(8)],{m+' '+xy:[next(r['median'] for r in weights['corner_xy'] if r['material']==m and r['coordinate']==xy and r['corner']==i) for i in range(8)] for m in E.MATS for xy in 'xy'},'Median weight')
    plot('03_m0_m1_moderate_candidate.png','MODERATE candidate quality — ORACLE POSTHOC ONLY',['MODERATE87'],{a:[mod[a]['oracle']['ADDsym_AUC']] for a in E.ARMS},'Oracle ADDsym AUC — NONDEPLOYABLE')
    plot('04_m0_m1_moderate_current.png','MODERATE production D9',['ADDsym AUC','Axis accuracy'],{a:[mod[a]['current']['ADDsym_AUC'],mod[a]['current']['axis_accuracy']] for a in E.ARMS},'Actual production metric')
    plot('05_m0_m1_severe.png','SEVERE81; oracle is NONDEPLOYABLE',['CURRENT ADD AUC','ORACLE ADD AUC','PCK10'],{a:[severe[a]['current']['ADDsym_AUC'],severe[a]['oracle']['ADDsym_AUC'],severe[a]['twoD']['PCK']['10']] for a in E.ARMS},'Fraction / AUC')
    plot('06_clean_preservation.png','CLEAN132 preservation',['PCK10','CURRENT ADD AUC'],{a:[clean[a]['twoD']['PCK']['10'],clean[a]['current']['ADDsym_AUC']] for a in E.ARMS},'Fraction / AUC')
    plot('07_source_heldout.png','Same synthetic heldout256; material-specific students',list(E.MATS),{a:[source[m][a]['PCK']['10']['fraction'] for m in E.MATS] for a in ('R0','old_S1','M0','M1')},'PCK10')
    plot('08_train_diag_loss.png','Fixed TRAIN synthetic microbatch; NOT generalization',list(E.MATS),{a:[train['synthetic_fixed_probe'][m][a]['L_diag_effective'] for m in E.MATS] for a in E.ARMS},'E2E / batch-scaled L_diag (no lambda)')
    plot('09_candidate_vs_selection_loss.png','Candidate quality versus production selection loss',['Moderate M0','Moderate M1','Severe M0','Severe M1'],{'candidate oracle AUC':[g[a]['oracle']['ADDsym_AUC'] for g in (mod,severe) for a in E.ARMS],'oracle minus CURRENT':[g[a]['selection_loss'] for g in (mod,severe) for a in E.ARMS]},'AUC (oracle POSTHOC ONLY)')
    records=E.read(C.H.P.DOC/'SPLIT.json')['evaluation'];pm=E.read(E.RAW/'POSE_METRICS.json');fm=E.read(E.RAW/'FRAME_METRICS.json');poses=E.read(E.RAW/'POSE_PREDICTIONS.json');truth=E.read(C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');meta={r['id']:r for r in E.read(C.H.V.RAW/'INFERENCE_METADATA.json')};preds={a:{} for a in E.ARMS}
    for m in E.MATS:
        for a in E.ARMS:preds[a].update(E.read(E.RAW/f'EVAL_{m}_{a}.json')['predictions'])
    chosen=[]
    def delta(r,kind):
        vals=[pm[a][r['id']][kind] for a in E.ARMS]
        return vals[1]['ADDsym_normalized']-vals[0]['ADDsym_normalized'] if all(v['available'] for v in vals) else None
    for sev,label,kind,better in [('MODERATE_OCCLUSION','moderate_candidate_improved','oracle',True),('MODERATE_OCCLUSION','moderate_current_improved','current',True),('MODERATE_OCCLUSION','moderate_harmed','current',False),('SEVERE_OCCLUSION','severe_improved','current',True),('SEVERE_OCCLUSION','severe_harmed','current',False)]:
        rr=[r for r in records if r['severity']==sev and delta(r,kind) is not None and (delta(r,kind)<0 if better else delta(r,kind)>0)]
        for r in sorted(rr,key=lambda r:((delta(r,kind) if better else -delta(r,kind)),r['id']))[:5]:chosen.append((r,label))
    rng=np.random.default_rng(20260923)
    chosen.extend((records[i],'random_control') for i in sorted(rng.choice(len(records),6,replace=False)))
    cases=[]
    for i,(r,label) in enumerate(chosen):cases.append(dict(id=r['id'],label=label,file=montage(r,label,i+1,preds,poses,pm,fm,truth,meta)))
    E.save(E.DOC/'CASE_MANIFEST.json',dict(selection='posthoc top5 signed ADD changes per requested category; random6 seed20260923; images are display only',cases=cases))
    rows=[]
    for group in ['ALL300','CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION','PLASTIC','WOOD']:
        for a in E.ARMS:
            r=results[group][a];two=r['twoD'];p=r['current']
            rows.append([group,a,f(two['PCK']['10']),str(two['correct']['10'])+'/'+str(two['corners']),f(p['ADDsym_AUC']),f(r['oracle']['ADDsym_AUC']),f(r['selection_loss']),f(p['axis_accuracy'])])
    text=['# Diagonal pose-sensitive EASY→HARD pilot','','## 1. 결론','',f'4 fits × 320 update = 1,280 update 완료. **{primary}**. routing: `{route}`.',f"MODERATE CURRENT ADD AUC: {mod['M0']['current']['ADDsym_AUC']:.6f} → {mod['M1']['current']['ADDsym_AUC']:.6f}. 후보 oracle AUC: {mod['M0']['oracle']['ADDsym_AUC']:.6f} → {mod['M1']['oracle']['ADDsym_AUC']:.6f}. Oracle는 GT를 사용하는 사후 진단이며 배포 성능이 아니다.",'','## 2. 왜 v1 수식을 바꿨는가','','기존 `rᵀHr`는 좌표 간 교차항으로 상쇄된다. H=[[1,1],[1,1]], r=[1,-1]이면 기존 값0, 대각 값2. 이번에는 `sum(diag(H)*r²*mask)/Nvalid`만 사용한다. 양의 가중치에서 각 residual의 직접 gradient는 0 방향이다. 공유 파라미터 때문에 다른 출력까지 보존된다는 뜻은 아니다. 기존 v1 파일은 변경하지 않았다.','','## 3. gradient integrity','','실제 Trainer seed42/deterministic 설정, TF32/AMP off, 새 R0 3개에서 독립 forward/backward 1회씩. 사전 허용오차 atol=1e-5, rtol=1e-4. OLD→OLD와 OLD→NEW λ0 모두 max_abs=0, max_rel=0, loss/component exact. optimizer0. 이전 v1의 bit-exact STOP을 방법 성능 실패로 해석하지 않는다.','','![gradient](figures/01_gradient_repeatability.png)','','## 4. synthetic geometry / weights','','기존512/512 renderer binding, occurrence projection 최대0.00055742px를 재사용. 두 material 각각 synthetic2560 활성, real2560 비활성, 비활성 synthetic0. valid scalar 평균1 오차 최대1.79e-7. 실사·무시점 가중치/gradient는0, center8은 제외.','','![weights](figures/02_pose_weight_by_corner.png)','','## 5. lambda / training','',f"λ={cal['lambda_diag']:.12g}; 기존 TOTAL loss coordinate-head gradient norm={cal['g_base']:.6f}, 대각={cal['g_diag']:.6f}; ratio=.25. 실제 head의 pose_head+kpts_head 모듈 객체에서 파라미터를 추출했다.",'','모두 원래 R0, seed42, 5epoch, batch/nbs16, 640, AdamW, lr1e-4, lrf.1, cosine. material별 기존 S1 RGB/order/target/mask와 정확히 같은5120 occurrences; 각 실사/합성2560. M0는 기존 criterion 그대로, M1만 대각 손실. 평가 중 학습·pseudo refresh·rescue 없음. final last만 사용; 학습 중 validation 호출은 무연산으로 차단했다.','','## 6. MODERATE 및 전체 요약','']
    text+=table(['집단','arm','PCK10','맞은 코너/전체','CURRENT ADD AUC','ORACLE AUC (진단)','선택손실','AxisAcc'],rows)
    text+=['','### 고정 분모 2D 및 6D 상세','']
    text+=table(['집단','arm','PCK5','PCK20','median/P90 px','gross20','detected/matched'],[[g,a,f(results[g][a]['twoD']['PCK']['5']),f(results[g][a]['twoD']['PCK']['20']),f(results[g][a]['twoD']['full_penalty_median_px'],2)+' / '+f(results[g][a]['twoD']['full_penalty_P90_px'],2),f(results[g][a]['twoD']['gross20']),str(results[g][a]['twoD']['detected'])+'/'+str(results[g][a]['twoD']['matched'])] for g in ['ALL300','CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION'] for a in E.ARMS])
    text+=['']
    text+=table(['집단','arm','R median/P90 °','yaw median/P90 °','t median/P90 cm','IoU3D median/P90','pose coverage'],[[g,a,*[f(results[g][a]['current'][k]['median'],3)+' / '+f(results[g][a]['current'][k]['P90'],3) for k in ('rotation_deg','yaw_deg','translation_cm','IoU3D')],f(results[g][a]['current']['coverage'])] for g in ['ALL300','CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION'] for a in E.ARMS])
    text+=['','M0 두 재질은 이전 S1 final 가중치와 bit-exact 일치한다. 모델 로딩 시 실행 모듈 `__main__.DiagModel` 이름 해석만 보완했으며, checkpoint 수정·학습 재시작은 없었다.']
    text+=['','![moderate candidate](figures/03_m0_m1_moderate_candidate.png)','','![moderate current](figures/04_m0_m1_moderate_current.png)','','## 7. SEVERE','','![severe](figures/05_m0_m1_severe.png)','','## 8. CLEAN/source 및 TRAIN','','![clean](figures/06_clean_preservation.png)','','![source](figures/07_source_heldout.png)','','합성 heldout은 기존256장 그대로이며 두 material 학생을 각각 같은256장에 평가했다. 보존기준1pp는 pilot 분기 기준이지 통계적 유의성 기준이 아니다.','']
    text+=table(['material','arm','heldout PCK10','맞은점/전체','TRAIN Ldiag','TRAIN weighted error','TRAIN pose normalized'],[[m,a,f(source[m][a]['PCK']['10']['fraction']),str(source[m][a]['PCK']['10']['correct'])+'/'+str(source[m][a]['points']),f(train['synthetic_fixed_probe'][m][a]['L_diag_effective']),f(train['synthetic_fixed_probe'][m][a]['weighted_coordinate_error_mean']),f(train['synthetic_fixed_probe'][m][a]['pose_mean'])] for m in E.MATS for a in E.ARMS])
    text+=['','![train](figures/08_train_diag_loss.png)','',f"고정 synthetic TRAIN Ldiag 감소: {train['diag_reduced']}. 이는 TRAIN fit 진단이며 일반화 성능이 아니다.",'','![candidate selection](figures/09_candidate_vs_selection_loss.png)','','## 9. 성공/실패 사례','','노란선=학생 2D, 빨간선=현재 선택 pose, 하늘 점선=대안 W/D, 초록십자=평가 정답. 원본 RGB와 M0/M1 전체 영상을 동일 배율로 표시. 개선/손상은 사후 ADD로 선택했으며 랜덤6장을 별도 표시한다.']
    for case in cases:text+=['',f"### {case['label']} — {case['id']}",'',f"![{case['label']}](figures/{case['file']})"]
    source_pose=E.read(E.DOC/'SYNTH_HELDOUT_POSE_RESULTS.json')['materials']
    text+=['','### 합성 heldout pose와 해석 보완','','합성 heldout256은 renderer pose/K/dimensions와 저장된 2D target 투영을 직접 대조했다. 새 pose GT를 추정하지 않았다. `SYNTH_ONLY`는 고정 synthetic TRAIN 대각 loss 감소를 뜻하며 합성 heldout pose 개선을 뜻하지 않는다. [SOURCE_GEOMETRY_BINDING.json](SOURCE_GEOMETRY_BINDING.json), [SYNTH_HELDOUT_POSE_RESULTS.json](SYNTH_HELDOUT_POSE_RESULTS.json).','']
    text+=table(['material','R0 ADD AUC','old S1 ADD AUC','M0 ADD AUC','M1 ADD AUC'],[[m,*[f(source_pose[m][a]['ADDsym_AUC'],6) for a in ('R0','old_S1','M0','M1')]] for m in E.MATS])
    text+=['','## 10. 객관적 판정','',f"PRIMARY_BOTTLENECK_AFTER: `{primary}`",'',f"SECONDARY_BOTTLENECK_AFTER: `{decision['secondary']}`",'','## 11. 다음 딱 한 실험','',nextone,'','## 12. 한계','','single seed; 같은 세션에서 재사용한 DEV300; 독립 물리적 pose GT가 아닌 geometry-resolved reference; diagonal local approximation; Linear-Covariance 완전 재현 아님; 좌표 분리 loss는 공유 신경망 출력 보존을 보장하지 않음. PCK 상승을 pose 성공으로 치환하지 않으며, oracle을 실제 성능으로 보고하지 않는다.','','## 13. 재현','',f"HEAD_BEFORE: `{E.read(E.DOC/'INPUT_BINDINGS.json')['head']}`",'','입력 hash: [INPUT_BINDINGS.json](INPUT_BINDINGS.json). 손실/gradient: [GPU_GRADIENT_PARITY.json](GPU_GRADIENT_PARITY.json). 전체 material×severity/session/common-matched 및 2D/6D 분포: [RESULTS.json](RESULTS.json). 코너 유지/손실: [TRANSITIONS.json](TRANSITIONS.json). 학습: FIT_*.json 및 [TRAINING_PARITY.json](TRAINING_PARITY.json). 이 문서를 포함하는 Git 커밋이 재현 코드 버전이다. 대용량 checkpoint/cache는 push 대상에서 제외했다.']
    E.save(E.DOC/'REPORT_KO.md','\n'.join(text)+'\n')
    E.save(E.DOC/'SUMMARY_KO.md','\n'.join(text[:8])+'\n\n전체: [REPORT_KO.md](REPORT_KO.md)\n')
    E.immutable();assert all((E.DOC/'figures'/case['file']).exists() for case in cases)
    E.save(E.DOC/'AUDIT.json',dict(complete=True,fits=4,optimizer_steps=1280,original_artifacts_unchanged=True,input_hashes_verified=True,predictions_frozen_before_scoring=True,production_D9_unchanged=True,denominators=[300,132,87,81],figures=len(list((E.DOC/'figures').iterdir())),image_links_exist=True,decision=decision))
    print('REPORT_COMPLETE',decision,flush=True)

if __name__=='__main__':main()
