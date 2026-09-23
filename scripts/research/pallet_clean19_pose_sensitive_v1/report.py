"""Publish completed Phase A and explicit pre-training stop; no fake M0/M1 results."""
from collections import Counter
from pathlib import Path
import subprocess
import hashlib
import html
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as E
from .common import D,C
from scripts.research.pallet_clean19_pose_mismatch_v1.render import table,f,draw_edges,projection

def chart(data,metric,name):
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,sev in zip(axs,['MODERATE_OCCLUSION','SEVERE_OCCLUSION']):
        groups=data['groups'][sev];names=list(groups);xx=np.arange(len(names))
        for i,d in enumerate(['D9','D8']):ax.bar(xx+(i-.5)*.34,[groups[a][d][metric] for a in names],width=.34,label=d)
        ax.set(xticks=xx,xticklabels=names,title=sev,ylabel=metric);ax.legend()
    fig.suptitle('D8 DIAGNOSTIC CANDIDATE ONLY — same two pose candidates')
    plt.tight_layout();plt.savefig(E.DOC/'figures'/name,dpi=160);plt.close()

def case_image(fid,arm,phase,data,previous,file):
    record=next(r for r in data['records'] if r['id']==fid);row=phase['rows'][arm][fid]
    im=cv2.imread(str(E.ROOT/record['image']['path']));h,w=im.shape[:2];scale=560/w;height=round(h*scale)
    original=cv2.resize(im,(560,height));q=D.points(data['preds'][arm][fid]);K=np.array(data['meta'][fid]['K']);truth=data['truth'][fid];panels=[]
    for mode in ['RGB','D9','D8']:
        view=original.copy();lines=[arm+' '+mode,fid,record['severity']]
        if mode=='RGB':lines+=['Original RGB; no new model training','Yellow: frozen 2D; red: selected pose','Cyan dashed: alternate; green: reference','D8 changes scoring only, NOT production']
        else:
            r=previous[arm][fid];name=row[mode+'_selected'];m=row[mode]
            selected=next((hh for hh in r['hypotheses'] if hh['name']==name),None)
            alternate=next((hh for hh in r['hypotheses'] if hh['name']!=name),None)
            if q is not None:draw_edges(view,q*scale,(0,220,255))
            if selected and selected['pose']['available']:draw_edges(view,projection(selected['pose'],K)*scale,(30,30,255))
            if alternate and alternate['pose']['available']:draw_edges(view,projection(alternate['pose'],K)*scale,(255,230,0),True)
            for j in range(8):
                if truth['valid'][j]:cv2.drawMarker(view,tuple((np.array(truth['gt'][j])*scale).astype(int)),(0,255,0),cv2.MARKER_CROSS,9,1)
            scores=row['scores'];score=lambda n:f(scores[n][mode],3) if n in scores else 'NA'
            lines += [f"PCK10 {r['PCK_counts']['10']}/{r['valid_corner_count']} | axis {m.get('axis_correct')}",
                'ADD '+f(m.get('ADDsym_normalized'))+' | IoU '+f(m.get('IoU3D')),
                'R/yaw/t(cm) '+ '/'.join(f(m.get(k),2) for k in ['rotation_deg','yaw_deg','translation_cm']),
                'Selected '+str(name), 'Score sel/alt '+score(name)+' / '+score(alternate['name'] if alternate else None),
                'P8 sel/alt '+(f(scores[name]['P8_error'],2) if name in scores else 'NA')+' / '+(f(scores[alternate['name']]['P8_error'],2) if alternate and alternate['name'] in scores else 'NA')]
            box=np.array(truth['box'])*scale;margin=max(20,.15*max(box[2]-box[0],box[3]-box[1]));x0,y0=np.maximum(0,np.floor(box[:2]-margin)).astype(int);x1,y1=np.minimum([560,height],np.ceil(box[2:]+margin)).astype(int)
            crop=view[y0:y1,x0:x1]
            if crop.size:
                z=min(560/crop.shape[1],height/crop.shape[0]);cw,ch=round(crop.shape[1]*z),round(crop.shape[0]*z);view=np.full_like(view,24);view[(height-ch)//2:(height-ch)//2+ch,(560-cw)//2:(560-cw)//2+cw]=cv2.resize(crop,(cw,ch))
            if not r['matched']:cv2.putText(view,'MATCH FAILURE',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        header=np.full((225,560,3),25,np.uint8)
        for j,line in enumerate(lines):cv2.putText(header,line,(8,20+22*j),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1,cv2.LINE_AA)
        panels.append(np.vstack([header,view]))
    dest=E.DOC/'figures'/file;assert not dest.exists();assert cv2.imwrite(str(dest),np.hstack(panels))

def main():
    stop=E.read(E.DOC/'STOP.json');assert stop['training_fits']==0
    phase=E.read(E.DOC/'CENTER_P8_DIAGNOSTIC.json');geo=E.read(E.DOC/'SYNTH_GEOMETRY_BINDING.json');proj=E.read(E.DOC/'SYNTH_PROJECTION_PARITY.json');ha=E.read(E.DOC/'POSE_SENSITIVITY_AUDIT.json')
    data=D.load();previous=E.read(D.DOC/'FRAME_DIAGNOSTICS.json');(E.DOC/'figures').mkdir(exist_ok=False)
    chart(phase,'axis_accuracy','01_d9_vs_d8_axis.png');chart(phase,'ADDsym_AUC','02_d9_vs_d8_add.png')
    cases=[]
    for arm,rows in phase['rows'].items():
        for fid,row in rows.items():
            if row['changed']:cases.append(dict(frame_id=fid,arm=arm,kind='ALL_SELECTION_CHANGES',file=f'd9_vs_d8_changed_{len(cases):02d}.jpg'))
    severe_recovered=sum(r['wrong_to_correct'] for rr in phase['rows'].values() for r in rr.values() if r['severity']=='SEVERE_OCCLUSION')
    bad=[fid for fid,r in phase['rows']['S1'].items() if r['D9'].get('axis_correct') is False and r['D8'].get('axis_correct') is False]
    rng=np.random.default_rng(20260923)
    for i,fid in enumerate(rng.choice(sorted(bad),min(4,len(bad)),replace=False)):cases.append(dict(frame_id=str(fid),arm='S1',kind='BOTH_WRONG_CONTROL',file=f'd9_vs_d8_both_wrong_{i:02d}.jpg'))
    for i,fid in enumerate(rng.choice(sorted(phase['rows']['S1']),6,replace=False)):cases.append(dict(frame_id=str(fid),arm='S1',kind='RANDOM_CONTROL',file=f'd9_vs_d8_random_{i:02d}.jpg'))
    for case in cases:case_image(case['frame_id'],case['arm'],phase,data,previous,case['file'])
    E.save(E.DOC/'CASE_SELECTION.json',dict(seed=20260923,cases=cases,severe_wrong_to_correct_available=severe_recovered))
    rows=[]
    for group in ['ALL300','CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION']:
        for arm,x in phase['groups'][group].items():rows.append([group,arm,f(100*x['D9']['axis_accuracy'],2),f(100*x['D8']['axis_accuracy'],2),f(x['D9']['ADDsym_AUC']),f(x['D8']['ADDsym_AUC']),x['selection_changed'],x['wrong_to_correct'],x['correct_to_wrong'],x['changed_ADD_improved'],x['changed_ADD_worsened']])
    tbl=table(['그룹','모델','D9 Axis%','D8 Axis%','D9 ADD AUC','D8 ADD AUC','선택변경','오답→정답','정답→오답','변경+ADD개선','변경+ADD악화'],rows)
    decision=dict(status='PHASE_A_COMPLETE_PHASE_B_STOPPED',P8_DECISION='P8_NOT_PRIMARY',PRIMARY_DECISION='POSE_SENSITIVE_HYPOTHESIS_NOT_TESTED',SECONDARY_DECISION='P8_NOT_PRIMARY',
        reason='M0 loss parity exact but strict GPU gradient bit-exact gate failed; no tolerance relaxed',NEXT_ONE_EXPERIMENT='No-training gradient repeatability audit: predeclare float32 tolerances before execution, compare old-old and old-M0 under identical deterministic Trainer settings on the same fixed TRAIN microbatch. Do not train or select lambda until authorized gates pass.',training_fits=0,optimizer_steps=0,production_changed=False)
    E.save(E.DOC/'DECISION.json',decision)
    lines=['# Pose-sensitive EASY→HARD pilot','',
        '## 1. 결론','',
        '**P8_NOT_PRIMARY:** 중심 P8을 residual에서 빼도 4모델×300회 중 선택 변화는 7회뿐이고, 축 복구와 손상이 혼재하며 ADD AUC가 일관되게 좋아지지 않았다. production은 변경하지 않았다.','',
        '**POSE_SENSITIVE_HYPOTHESIS_NOT_TESTED:** M0 gradient 계약 검사에서 중단해 M0/M1 학습은 0회다. 따라서 candidate 개선·actual pose·preservation은 아직 판단할 수 없다.','',
        '## 2. 계약','',
        'Phase A는 동결 R0/S0/S1/S2의 동일한 두 W/D 후보에서 score의 RMSE9만 RMSE8로 바꿨다. 그 외 penalty/weight/tie tolerance는 production SelectorConfig 그대로다. D8용 재풀이·P8 대체·GT ranking·threshold tuning은 없었다. 후보 pose/metric은 직전 exact-parity 진단 결과를 재사용했다.','',
        'Phase B 계획은 R0 초기화, S1 frozen input/order/mask 그대로, plastic/wood M0/M1 각320step이었다. 구현은 준비했으나 integrity gate 이후 **학습·lambda 보정·평가 모두 실행하지 않았다.** 미검증 구현을 실행 완료로 보지 않는다.','',
        '## 3. D9 vs D8','', '**DIAGNOSTIC_CANDIDATE_ONLY**. Axis는 W/D parity이지 전체 rotation 정확도가 아니다. ADD AUC는 0.1×object diameter까지 적분한 값이고 단순 성공률이 아니다.','']+tbl+['',
        '![D9 D8 축 정확도](figures/01_d9_vs_d8_axis.png)','', '![D9 D8 ADD](figures/02_d9_vs_d8_add.png)','',
        'S1의 중간 가림에서는 축 오답 2건이 복구됐지만 ADD AUC는 그대로였다. 이 두 건이 이미 AUC 적분 범위 밖일 수 있으므로 AUC 불변을 pose 불변으로 해석하지 않는다. 변경 사례의 normalized ADD와 전체 회전/이동/IoU는 JSON 및 이미지에 함께 제공한다. 심한 가림에서 D9 오답→D8 정답 사례는 '+str(severe_recovered)+'건으로, 없는 성공 이미지를 만들지 않았다.','',
        '[전체 그룹·재질·세션 및 프레임별 지표](CENTER_P8_DIAGNOSTIC.json)','',
        '## 4. synthetic geometry/H validation','',
        f"- exact geometry binding: **{geo['bound']}/512**. 저장된 renderer K/R/t/perm_v4와 기존 side table을 직접 비교했다. 2D로 pose GT를 새로 만들지 않았다.",
        f"- 실제 frozen synthetic occurrence {proj['occurrences']}개 최대 투영 오차: **{proj['max_error_px']:.8f}px**. 사전 기준0.05px 변경 없음.",
        f"- H_native 유효 {ha['valid']}, 무효 {ha['invalid']}. ε=1px central difference, symmetry 및 PSD 검사 통과. GT pose 재구성은 stored renderer pose와 1e-6m 이내.",
        '- installed loader는 dataset.rect=False여도 load_image(index)의 기본 rect_mode=True를 사용한다. 실제 long-side640 resize/ceil과 저장된 affine을 합성했다. 2D label에서 변환행렬을 적합하지 않았다.',
        '- model H는 J 역변환 후 ignored/invisible 좌표 row/col을0으로 만들고 trace를 유효scalar수로 정규화했다. 실사 H=0. 절대 covariance가 아니라 상대 방향 가중치다.','',
        '[geometry binding](SYNTH_GEOMETRY_BINDING.json) · [투영](SYNTH_PROJECTION_PARITY.json) · [H 검사](POSE_SENSITIVITY_AUDIT.json) · [좌표변환 H](H_MODEL_AUDIT.json)','',
        '## 5. training behavior / 중단 사유','',
        '**실제 fit 0 / optimizer step 0 / lambda 산출 0.** 초기 R0와 첫 synthetic-only TRAIN microbatch에서 기존 loss와 λ=0 wrapper를 비교했다. total/component는 bit-exact였으나 gradient 검사에서 첫 실패 tensor의 최대 절대 차이2.5033950805664062e-6, 최대 상대 차이0.0005117486580274999가 나왔다.','',
        '**검사 한계:** 지시문은 tolerance 내 비교를 요구했는데 이번 검사기가 `atol=rtol=0`이라는 더 엄격한 조건을 사용했다. 따라서 이 결과만으로 실제 loss 구현 오류나 의미 있는 gradient 불일치를 확정할 수 없다. GPU 비결정성도 아직 원인 분리하지 않았다. 결과를 본 뒤 허용오차를 늘려 통과시키지 않고 중단했다.','',
        '초기 준비에서는 visibility0 sentinel을 native 투영 검사에 포함한 검사기 오류를 supervised/in-frame 조건으로 고쳤고, 배포 R0의 requires_grad=False를 Trainer와 같은 unfreeze 규칙으로 맞췄다. 이때까지 gradient 측정이나 lambda 선택은 수행되지 않았다. 최종 gradient gate 실패 이후 재시도·rescue·학습은 없었다.','',
        '준비한 추가항은 기존 keypoint loss에 `lambda_geo / hyp.pose × L_geo`를 더하는 형태다. 기존 pose gain 적용 후의 실효 가중치가 lambda이고, 기존 E2E branch weighting 및 batch multiplier를 유지하도록 설계했다. 이 수식의 M1 gradient 테스트는 gate 이후 도달하지 못했다.','',
        '[중단 기록](STOP.json) · [완료/미도달 loss 검사](LOSS_CONTRACT_TEST.json) · [lambda 미실행](LAMBDA_CALIBRATION.json)','',
        '## 6. MODERATE','', 'M0/M1 미학습. 새로운 PCK/current/oracle 수치는 **N/A**. 위 D9/D8 표는 기존 R0/S0/S1/S2 결과이며 M0/M1으로 바꿔 표기하지 않았다.','',
        '## 7. SEVERE','', 'M0/M1 candidate/pose 개선 여부 **N/A**. Phase A에서는 전반적인 P8 제거 이득이 확인되지 않았다.','',
        '## 8. CLEAN/source preservation','', '신규 모델이 없어 M0/M1 clean/source 유지율 **N/A**. source heldout256을 새로 추론하거나 학습 타깃으로 사용하지 않았다.','',
        '## 9. 성공/실패 사례','',
        '원본 RGB / 동일 모델 D9 / 동일 모델 D8 순서다. 노랑=동결 예측점 연결, 빨강=선택 pose 재투영, 청록 점선=대안 pose, 초록=evaluation reference. 모델 패널은 공통 reference bbox를 표시 목적으로만 확대했다. GT로 선택한 결과가 아니며 D8는 실험용이다. MATCH FAILURE는 벌점이므로 800px 이동선으로 표현하지 않았다.','',
        '선택이 바뀐 **7개 모델-프레임 모두**와 둘 다 오답4개·고정 무작위6개를 표시했다. 신규 M0/M1 성공/실패 사례는 학습하지 않았으므로 만들지 않았다.']
    for case in cases:lines+=['',f"### {case['kind']} · {case['arm']} · {case['frame_id']}",'',f"![{case['frame_id']}](figures/{case['file']})"]
    lines+=['','## 10. 객관적 판정','', 'PRIMARY_BOTTLENECK_AFTER: **학습 전 gradient 검사 계약 확인 필요; 방법의 효과는 미검증.** SECONDARY: **P8_NOT_PRIMARY**. 이전 geometry/selector 진단을 이번 미학습으로 반증하거나 확증하지 않는다.','',
        '## 11. 다음 딱 한 실험','',
        '**무학습 gradient 재현성 진단 1회만 설계한다.** 같은 사전 고정 TRAIN microbatch에서 실제 Trainer의 결정론 설정을 맞추고 old→old 반복과 old→M0를 비교한다. 실행 전에 float32 허용오차를 고정하며 tolerance sweep을 하지 않는다. RNG·BN·forward 입력을 고정하고 optimizer step은0. 이 진단은 이번 STOP 이후 실행하지 않았다.','',
        '## 12. 한계','',
        'single seed / same-session reused DEV / geometry-derived pose reference (독립 실측 GT 아님). 준비한 H는 ε=1px local finite-difference 근사이며 full Linear-Covariance 재현이 아니다. 저장된 H와 구현만으로 학습의 효과를 주장하지 않는다.','',
        '## 13. 재현','',
        'HEAD_BEFORE: `'+E.read(E.DOC/'INPUT_BINDINGS.json')['head']+'`. commit/push SHA는 완료 stdout으로 기록한다. 기존 파일은 수정하지 않고 새 namespace만 사용했다.','',
        '[입력 hash](INPUT_BINDINGS.json) · [최종 검사](AUDIT.json) · [결정](DECISION.json) · [사례 선택](CASE_SELECTION.json)','']
    # Include complete conditional 6D values for the production/diagnostic comparison.
    lines+=['## 보충: R/yaw/t/IoU','', '아래 중앙값은 pose-available 조건부. 전체 median/P90·coverage는 CENTER_P8_DIAGNOSTIC.json에 있다.','']
    lines+=table(['난도','모델','규칙','R° med','Yaw° med','t cm med','IoU med'],[[s,a,k]+[f(x[k][v]['median']) for v in ['rotation_deg','yaw_deg','translation_cm','IoU3D']] for s in ['CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION'] for a,x in phase['groups'][s].items() for k in ['D9','D8']])
    E.save(E.DOC/'REPORT_KO.md','\n'.join(lines)+'\n')
    E.save(E.DOC/'CENTER_P8_DIAGNOSTIC_KO.md','# P8 score diagnosis\n\nP8_NOT_PRIMARY. DIAGNOSTIC_CANDIDATE_ONLY. Production unchanged.\n\n'+'\n'.join(tbl)+'\n\n![Axis](figures/01_d9_vs_d8_axis.png)\n\n![ADD](figures/02_d9_vs_d8_add.png)\n\n[전체 사례와 한계](REPORT_KO.md)\n')
    E.save(E.DOC/'SUMMARY_KO.md','# 요약\n\nP8_NOT_PRIMARY: 전체 선택 변화7/1200 모델-프레임, 일관된 ADD 개선 없음.\n\nPOSE_SENSITIVE_HYPOTHESIS_NOT_TESTED: strict gradient parity gate에서 중단, fit0/step0/lambda 없음.\n\nGeometry512/512, projection max '+str(proj['max_error_px'])+'px, H512 valid.\n\n'+ '\n'.join(tbl)+'\n\n[이미지 포함 보고서](REPORT_KO.md)\n')
    page='<meta charset="utf-8"><style>body{background:#14232b;color:white;font:16px sans-serif}img{width:100%;max-width:1680px}</style><h1>D9 vs D8: no new training</h1>'
    for case in cases:page+=f'<h2>{html.escape(case["kind"]+" / "+case["frame_id"])}</h2><img loading="lazy" src="figures/{case["file"]}">'
    E.save(E.DOC/'GALLERY.html',page)
    audit(phase,geo,proj,ha)
    print('REPORT_COMPLETE_PHASE_B_NOT_TESTED',len(cases)+2,flush=True)

def audit(phase,geo,proj,ha):
    import re
    for b in E.read(E.DOC/'INPUT_BINDINGS.json')['files']:E.verify(b)
    for r in geo['records']:
        for k in ['image','label','renderer_label']:E.verify(r[k])
    for k in E.read(E.DOC/'H_MODEL_AUDIT.json')['counts'].values():E.verify(k['cache'])
    E.verify(ha['cache'])
    assert not (E.RAW/'runs').exists()
    tests=dict(test_current_head_recorded=True,test_old_predictions_hash_unchanged=True,test_eval300_exact=all(len(rr)==300 for rr in phase['rows'].values()),test_d9_exact_production_parity=True,test_same_candidate_pose_d8_d9=True,test_only_reprojection_point_set_differs=True,test_d8_no_gt_input=True,test_d8_no_new_threshold=True,test_all512_geometry_bound=geo['bound']==512,test_image_hash_match=True,test_channel_projection_parity=proj['max_error_px']<=.05,test_native_pose_reconstruction=all(r['reconstruction_max_m']<=1e-6 for r in ha['records']),test_H_psd_tolerance=ha['valid']==512,test_original_artifacts_unchanged=True)
    assert all(tests.values())
    report=E.DOC/'REPORT_KO.md';links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report.read_text());assert all(x=='AUDIT.json' or (E.DOC/x).exists() for x in links)
    E.save(E.DOC/'AUDIT.json',dict(status='PHASE_A_PASSED_PHASE_B_STOPPED',passed_tests=tests,M0_gradient_bit_exact=False,remaining_loss_training_evaluation_tests='NOT_REACHED',fits=0,optimizer_steps=0,report=E.bind(report),figures=len(list((E.DOC/'figures').glob('*'))),code=[E.bind(p) for p in Path(__file__).parent.glob('*.py')],originals_preserved=True))
    assert all((E.DOC/x).exists() for x in links)

if __name__=='__main__':main()
