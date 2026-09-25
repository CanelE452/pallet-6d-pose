"""Stage reports from completed locked results, never used for fitting."""
import argparse
from . import common as C

def stage1():
    j=C.read(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json');m=j['groups']['MODERATE']
    lines=['# Stage1 — Moderate W/D 후보 선택 진단','',
        '**Q1: YES. 현재 선택기가 더 좋은 후보를 실제로 버리는 사례가 있다.**','',
        f"고정 Moderate21, S1 PCK10={m['PCK10']:.6f}. CURRENT ADD AUC={m['CURRENT']['ADDsym_AUC']:.8f}, ORACLE={m['ORACLE']['ADDsym_AUC']:.8f}, gap={m['selection_loss']:.8f}. 기존 값을1e-7 이내 재현했다.",'',
        f"W/D parity {m['CURRENT']['axis_correct_count']}/21. 현재 axis wrong {m['axis_wrong']}장, alternate ADD-better {m['alternate_ADD_better']}장, 둘의 교집합 {m['current_wrong_alternate_better']}장. 현재 wrong인데 재투영오차는 오히려 더 낮은 사례 {m['wrong_lower_reprojection']}장. 두 후보의 invariant violation이 모두0인 사례 {m['both_invariant_zero']}/21장.",'',
        '재투영 잔차·좌우/상하 규칙만으로 올바른 W/D를 고르기 어려운 사례다. 이것이 synthetic 학습으로 해결된다는 뜻은 아니며 Stage2/3에서 따로 검증한다. real GT를 보고 특징을 고르지 않았고, Stage0 목록을 고정한 뒤 이 결과를 읽었다.','',
        '|frame|recording|current|best (사후)|best ADDnorm|ADD gap|category|','|---|---|---|---|---:|---:|---|']
    for r in j['moderate']:lines.append(f"|{r['id']}|{r['recording']}|{r['current']}|{r['best']}|{r['best_ADD_actual']:.5f}|{r['oracle_current_ADD_gap']:.5f}|{r['category']}|")
    for name in ('01_current_vs_oracle.png','02_selector_margin.png','03_component_distributions.png','04_wrong_candidate_cases.jpg','05_recording_breakdown.png'):
        lines+=['',f'![{name}](../figures/stage1/{name})']
    lines+=['','H10 role warning은 유지하지만 C4 remapping으로 점수를 만들거나 primary에서 사례를 제외하지 않았다. GT oracle은 비배포용이다. 이미 열람한 recording-disjoint DEV이며 독립 TEST가 아니다. 작은 ROI만 공개하고 원본/좌표캐시는 로컬에 보존한다.','']
    C.save(C.sdoc(1)/'STAGE1_REPORT_KO.md','\n'.join(lines))
    C.save(C.DOC/'REPORT_KO.md','# Selector recovery + frozen expert routing\n\nStage1 완료, Stage2–4 진행 중.\n\n[Stage1 보고서·이미지](stage1_diagnostic/STAGE1_REPORT_KO.md)\n')

def stage2():
    v=C.read(C.sdoc(2)/'SCORER_VAL_RESULTS.json');t=C.read(C.sdoc(2)/'SCORER_SYNTH_TEST.json');lab=C.read(C.sdoc(2)/'EXACT_LABEL_AUDIT.json')
    lines=['# Stage2 — 합성 전용 W/D shared candidate scorer','',f"선택 모델: **{t['winner']}**. VAL 동률이면 LINEAR라는 사전 규칙으로 선택했다. TEST는 선택 완료 후 한 번만 평가했다.",'',
        'TRAIN 4096 / VAL 1024 / TEST 1024 프레임. 같은 프레임의 S0/S1은 같은 split이다. renderer 그룹 분리, 기존 replay512와 그 파생 이미지까지 제외했다. exact Xcf의 metric width/depth로 라벨을 만들었으며 면적 휴리스틱을 쓰지 않았다.','',
        f"정답 좌표와 renderer pose 재투영의 최대 차이 {lab['projection_max_px']:.6f}px. GEO + head GAP448의 두 모델만 학습했다. base weight/gradient는 그대로이며 hook 전후 예측은 bit-exact다.",'',
        '|variant|VAL accuracy|best epoch|epochs|','|---|---:|---:|---:|']
    for a,r in v['variants'].items():lines.append(f"|{a}|{r['best_val_accuracy']:.6f}|{r['best_epoch']}|{r['epochs']}|")
    lines+=['','|expert / TEST strata|N|D9 accuracy|scorer accuracy|Brier|','|---|---:|---:|---:|---:|']
    for a,groups in t['by_expert'].items():
        for g,r in groups.items():lines.append(f"|{a} / {g}|{r['n']}|{r['current_accuracy']}|{r['learned_accuracy']}|{r['brier']}|")
    lines+=['','합성 TEST 전체 D9 {:.4%} → scorer {:.4%}. Q2는 개선 신호 YES이며 실사 일반화의 증거는 아직 아니다.'.format(t['aggregate']['current_accuracy'],t['aggregate']['learned_accuracy']),
        '', '주의: 원래 R0는 더 넓은 합성 원천으로 학습되었다. 이 TEST는 새 선택기/router 학습에서만 보류된 TEST이지 base detector가 처음 보는 합성 원천이라는 뜻이 아니다. 그룹 분리로 VAL/TEST는 P0/TEX 저양각 위주이며 HIGH 계층은 0장(N/A)이다.','',
        '![VAL](../figures/stage2/01_validation.png)','![TEST](../figures/stage2/02_test.png)','',
        '실사 GT 경로 접근을 runtime guard로 차단한 별도 학습 프로세스. real GT/axis/ADD/oracle로 checkpoint를 고르지 않았다.']
    C.save(C.sdoc(2)/'STAGE2_REPORT_KO.md','\n'.join(lines)+'\n')

def stage3():
    j=C.read(C.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];d=C.read(C.sdoc(3)/'STAGE3_DECISION.json');t=C.read(C.sdoc(3)/'REAL_SCORER_TRANSITIONS.json')
    lines=['# Stage3 — frozen 합성 선택기의 실사 적용','',f"판정: **{d['primary']}** / {d['secondary']}",'',
        '실사128장의 두 expert별 결정을 전부 저장/hash lock한 다음에만 참조값을 읽었다. 같은 scorer, 같은 tie 규칙을 모든 난도·expert에 적용했다. raw2D를 바꾸지 않으므로 기존 PCK와 verified-anchor PCK는 동일하다.','',
        '|group|expert|current AUC|scorer AUC|oracle AUC|axis current→scorer|gap recovery|','|---|---|---:|---:|---:|---|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        for a in C.ARMS:
            r=j[g][a];lines.append(f"|{g}|{a}|{r['current']['ADDsym_AUC']:.6f}|{r['scorer']['ADDsym_AUC']:.6f}|{r['oracle']['ADDsym_AUC']:.6f}|{r['current']['axis_correct_count']}→{r['scorer']['axis_correct_count']}|{r['gap_recovery']}|")
    lines+=['',f"Moderate S1 transitions: {t['MODERATE']['S1']}",'','|group|arm/method|R med °|yaw med °|t med cm|IoU3D med|','|---|---|---:|---:|---:|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        for a in C.ARMS:
            for k in ('current','scorer','oracle'):
                r=j[g][a][k];lines.append(f"|{g}|{a}/{k}|{r['rotation_deg']['median']:.4f}|{r['yaw_deg']['median']:.4f}|{r['translation_cm']['median']:.4f}|{r['IoU3D']['median']:.4f}|")
    lines+=['', 'Stage4의 S0 유효성은 사전 구현한 보수적 부호 규칙(Clean/Moderate/Severe AUC가 모두 비감소)을 사용한다. 효과크기 임계값을 결과에 맞춰 만들지 않았다. 이미 열람한 DEV이며 독립 검증이 아니다.']
    for f in sorted((C.DOC/'figures/stage3').glob('*.png')):lines+=['',f'![{f.stem}](../figures/stage3/{f.name})']
    C.save(C.sdoc(3)/'STAGE3_REPORT_KO.md','\n'.join(lines)+'\n')

def stage4():
    r=C.read(C.sdoc(4)/'REAL_ROUTER_RESULTS.json');d=C.read(C.sdoc(4)/'STAGE4_DECISION.json');syn=C.read(C.sdoc(4)/'ROUTER_SYNTH_RESULTS.json')['groups'];fit=C.read(C.sdoc(4)/'ROUTER_VAL_RESULTS.json');cost=C.read(C.sdoc(4)/'COMPUTE_COST.json');oc=C.read(C.sdoc(4)/'OCCLUSION_PLAN_LOCK.json')
    lines=['# Stage4 — frozen S0/S1 clean-hard router','',f"판정: **{d['primary']}** / {d['secondary']}",'',
        f"공통 selector: **{r['base_selector']}**. Stage3의 S1 회복은 확인됐지만 S0 Moderate에는 새 선택기가 손해여서, 지시문 fallback대로 양쪽 모두 D9를 썼다. 따라서 이 표는 S1+새 scorer의 Stage3 이득까지 결합했다는 뜻이 아니다.",'',
        f"같은 합성 프레임 split을 재사용하여 clean/occluded 쌍 TRAIN8192 / VAL2048 / TEST2048 sample. 원래 S1의 size/fill/coverage/paired-feasibility 조건과 scheduled .5를 유지하여 실제 가림은 {oc['applied']}/{oc['frames']}개다. skip은 clean과 동일하게 남기고 applied-only도 따로 보고한다. placement는 GT오차/모델 결과를 사용하지 않았다. 원래 real 학습 policy를 synthetic에 적용했으며 random affine까지 복제한 실험은 아니다.",'',
        f"고정 expert 두 개의 exact synthetic ADDnorm 중 더 작은 쪽을 라벨로 사용했다. MLP64/32 한 개, seed42, best VAL={fit['best_val_accuracy']:.6f}, epoch={fit['best_epoch']}. 실사 GT 입력0, TEST1회, real decision 전부 lock 후 평가.",'',
        '|synthetic TEST|N|expert-choice acc|route S0/S1|S0 mean ADDnorm|S1|routed|oracle|','|---|---:|---:|---|---:|---:|---:|---:|']
    for g,x in syn.items():lines.append(f"|{g}|{x['n']}|{x['accuracy']:.4f}|{x['route_S0']}/{x['route_S1']}|{x['S0']['mean_ADDnorm']}|{x['S1']['mean_ADDnorm']}|{x['ROUTED']['mean_ADDnorm']}|{x['ORACLE']['mean_ADDnorm']}|")
    lines+=['','|real group|arm|PCK5|PCK10|PCK20|ADD AUC|axis correct|R med°|yaw med°|t med cm|IoU3D med|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        for a,x in r['groups'][g].items():
            k=x['twoD']['PCK'];p=x['sixD'];lines.append(f"|{g}|{a}|{k['5']:.4f}|{k['10']:.4f}|{k['20']:.4f}|{p['ADDsym_AUC']:.6f}|{p['axis_correct_count']}/{p['frames']}|{p['rotation_deg']['median']:.3f}|{p['yaw_deg']['median']:.3f}|{p['translation_cm']['median']:.3f}|{p['IoU3D']['median']:.4f}|")
    lines+=['','|group|S0 routes|S1 routes|routed AUC − S1 AUC|','|---|---:|---:|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        counts=r['routing'][g]['counts'];lines.append(f"|{g}|{counts['S0']}|{counts['S1']}|{d['actual_delta_vs_S1'][g]:+.6f}|")
    lines+=['',f"Dual median {cost['timings_ms']['dual_total_ms']['median']:.2f}ms / mean {cost['timings_ms']['dual_total_ms']['mean']:.2f}ms, peak allocated {cost['GPU_peak_allocated_MiB']:.1f}MiB, benchmark wall {cost['wall_seconds']:.2f}s. RTX3080, batch1, dual resident, decode/load 제외, PnP+context+CPU router 포함. Jetson 측정도 최종 배포 승격도 아니다.",'',
        '아래 사례는 routed−S1 ADD 차이의 양 끝에서 사후 선택한 설명용 이미지이다. 개선과 실패를 함께 보여주며, 극단 사례가 전체 빈도를 대표하지 않는다. GT best expert는 사후 진단용이고 추론 경로에 들어가지 않는다.']
    for f in sorted((C.DOC/'figures/stage4').glob('*')):lines+=['',f'![{f.stem}](../figures/stage4/{f.name})']
    C.save(C.sdoc(4)/'STAGE4_REPORT_KO.md','\n'.join(lines)+'\n')

def stage0():
    s1=C.read(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json')['groups']['MODERATE'];s2=C.read(C.sdoc(2)/'SCORER_SYNTH_TEST.json');s3=C.read(C.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];tr=C.read(C.sdoc(3)/'REAL_SCORER_TRANSITIONS.json');d3=C.read(C.sdoc(3)/'STAGE3_DECISION.json');s4=C.read(C.sdoc(4)/'REAL_ROUTER_RESULTS.json');d4=C.read(C.sdoc(4)/'STAGE4_DECISION.json');sy4=C.read(C.sdoc(4)/'ROUTER_SYNTH_RESULTS.json')['groups']['ALL'];cost=C.read(C.sdoc(4)/'COMPUTE_COST.json');audit=C.read(C.DOC/'FINAL_AUDIT.json');m=s3['MODERATE']['S1']
    q=dict(Q1_SELECTOR_BOTTLENECK_REAL='YES' if s1['alternate_ADD_better'] else 'NO',Q2_SYNTH_SELECTOR_LEARNS='YES' if s2['aggregate']['learned_accuracy']>s2['aggregate']['current_accuracy'] else 'NO',Q3_REAL_SELECTOR_RECOVERY='YES' if d3['moderate_gain'] else 'NO',Q4_CLEAN_HARD_TRADEOFF_SEPARABLE='YES' if d4['primary']=='CLEAN_HARD_TRADEOFF_SEPARABLE_SIGNAL' else 'NO')
    route='B' if q['Q3_REAL_SELECTOR_RECOVERY']=='YES' and q['Q4_CLEAN_HARD_TRADEOFF_SEPARABLE']=='NO' else 'A' if q['Q3_REAL_SELECTOR_RECOVERY']=='YES' else 'C' if q['Q2_SYNTH_SELECTOR_LEARNS']=='YES' else 'D'
    nextstep='zero-init single-model preservation 1개: easy/clean 출력을 보존하면서 hard 적응하는 제한된 residual/gate. 이번 실행에서는 학습하지 않음.' if route=='B' else 'Refer to frozen interpretation branch in the user directive.'
    C.save(C.DOC/'FINAL_DECISION.json',dict(status='COMPLETED',questions=q,route=route,primary_bottleneck='Clean preservation without losing hard gain',secondary_bottleneck='Remaining candidate selection and keypoint/role quality; scorer not uniformly transferable to S0',next_one_step=nextstep,independent_final_test=False,production_promoted=False))
    lines=['# W/D selector recovery + frozen expert routing — 통합 보고서','',
        '## 1. 결론 — Q1~Q4','',
        '**선택기 학습은 S1 실사 성능을 개선했다. 하지만 두 모델을 고르는 router는 Clean과 Severe의 장점을 동시에 보존하지 못했다.** 따라서 이번 결과에서 우선 남길 것은 S1+합성 선형 선택기 파일럿이며, router는 최종안으로 채택하지 않는다. base 모델이나 기존 논문 최종 경로를 자동 교체하지 않았다.','',
        '|질문|판정|근거|','|---|---|---|',
        f"|Q1 좋은 후보를 놓치는가?|{q['Q1_SELECTOR_BOTTLENECK_REAL']}|Moderate21 중 alternate ADD-better {s1['alternate_ADD_better']}장|",
        f"|Q2 합성 정답만으로 선택을 배웠나?|{q['Q2_SYNTH_SELECTOR_LEARNS']}|TEST {s2['aggregate']['current_accuracy']:.2%} → {s2['aggregate']['learned_accuracy']:.2%}|",
        f"|Q3 실사 Moderate gap을 회수했나?|{q['Q3_REAL_SELECTOR_RECOVERY']}|AUC {m['current']['ADDsym_AUC']:.6f} → {m['scorer']['ADDsym_AUC']:.6f}, gap recovery {m['gap_recovery']:.2%}|",
        f"|Q4 Clean 회복과 hard 이득 보존을 분리했나?|{q['Q4_CLEAN_HARD_TRADEOFF_SEPARABLE']}|Clean +{d4['actual_delta_vs_S1']['CLEAN']:.6f}, Severe {d4['actual_delta_vs_S1']['SEVERE']:+.6f}|",'',
        'Q3 YES는 완전 해결이 아니라 **부분 회복**이다. 21장 중 axis recovery 2장과 regression 1장이 있고 oracle gap은 아직 남는다. Q4 NO 역시 모든 router의 불가능성이 아니라 이번 고정 feature/합성 감독/단일 파일럿의 실패다.','',
        '## 2. 출발점 — recording-disjoint S0/S1','',
        '동일 frozen S0/S1, 학습 recording과 분리된 HELDOUT128을 그대로 사용했다. Clean29 / Moderate21 / Severe78, 일반 플라스틱만 포함한다. 전체 YOLO 재학습·새 라벨·추가 seed/하이퍼파라미터 탐색·개별 점 remapping은 하지 않았다. R0를 포함한 기존 결과는 hash로 보호했다.','',
        '|group|S0 PCK10|S1 PCK10|S0 D9 AUC|S1 D9 AUC|S1 new scorer AUC|','|---|---:|---:|---:|---:|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        z=s4['groups'][g];lines.append(f"|{g}|{z['S0']['twoD']['PCK']['10']:.4f}|{z['S1']['twoD']['PCK']['10']:.4f}|{s3[g]['S0']['current']['ADDsym_AUC']:.6f}|{s3[g]['S1']['current']['ADDsym_AUC']:.6f}|{s3[g]['S1']['scorer']['ADDsym_AUC']:.6f}|")
    lines+=['','## 3. Stage1 — selector diagnosis','',
        f"현재 axis wrong {s1['axis_wrong']}/21장, alternate axis-correct {s1['alternate_axis_correct']}장, wrong∩alternate ADD-better {s1['current_wrong_alternate_better']}장. wrong인데 current reprojection이 더 낮은 사례 {s1['wrong_lower_reprojection']}장. 두 후보의 invariant violation이 모두0인 사례 {s1['both_invariant_zero']}/21장이다. 기하학적 검사 통과나 작은 잔차만으로 올바른 W/D를 보장하지 못한다.",'',
        '기존 PCK/AUC/axis 통계를 1e-7 이내 재현했다. real GT를 읽기 전 feature 목록을 고정했고, Stage1 결과로 특징을 골라내지 않았다. [전체 프레임 표·분해](stage1_diagnostic/STAGE1_REPORT_KO.md)','',
        '![selector headroom](figures/stage1/01_current_vs_oracle.png)','',
        '## 4. Stage2 — synthetic-only scorer','',
        'TRAIN4096 / VAL1024 / TEST1024 프레임. 기존 replay512의 SHA와 같은 renderer 파생 프레임을 제외하고 renderer 그룹을 분리했다. S0/S1 sample은 프레임별 같은 split이다. exact K/R/t/Xcf/치수로 W/D 라벨을 만들며 historical area heuristic은 사용하지 않았다. frozen prediction lock 뒤에 synthetic label을 연결했다.','',
        'GEO_LINEAR와 GEO_IMG_MLP(input→64→GELU→32→GELU→1) 두 개만 학습했다. 동일 후보 scorer가 두 후보에 적용되고 작은 score를 택한다. AdamW .001, wd .0001, batch256, max30, seed42, VAL patience5. 두 모델의 best VAL 정확도는 0.9404297로 같아 사전 tie 규칙으로 GEO_LINEAR를 선택했다. 그 후 TEST 한 번만 평가했다.','',
        f"TEST S0 {s2['by_expert']['S0']['TEST']['current_accuracy']:.2%} → {s2['by_expert']['S0']['TEST']['learned_accuracy']:.2%}, S1 {s2['by_expert']['S1']['TEST']['current_accuracy']:.2%} → {s2['by_expert']['S1']['TEST']['learned_accuracy']:.2%}. Brier·margin 및 elevation/size 계층은 [Stage2 상세표](stage2_synth_scorer/STAGE2_REPORT_KO.md)에 있다.",'',
        '![synthetic test](figures/stage2/02_test.png)','',
        '## 5. Stage3 — frozen scorer real recovery','',
        f"판정 `{d3['primary']}`. S1 Moderate axis {m['current']['axis_correct_count']}/21 → {m['scorer']['axis_correct_count']}/21, 선택 변경 {tr['MODERATE']['S1']['selected_changed']}장, axis 회복 {tr['MODERATE']['S1']['recoveries']}장 / 퇴행 {tr['MODERATE']['S1']['regressions']}장. oracle AUC {m['oracle']['ADDsym_AUC']:.6f}. raw keypoint를 건드리지 않아 PCK와 verified visible-anchor PCK는 동일하다.",'',
        'S1 Clean AUC는 동일하고 Severe도 개선했다. 반면 S0 Moderate는 0.464119→0.401452로 나빠졌다. 따라서 새 scorer가 모든 expert에 통하는 것은 아니며, Stage4에서는 공통 D9 fallback을 사용했다. [전체 R/yaw/t/IoU3D·recording·전이 표](stage3_real_recovery/STAGE3_REPORT_KO.md)','',
        '![real selector](figures/stage3/01_moderate_auc_current_scorer_oracle.png)','',
        '아래는 Moderate에서 선택이 바뀐 **세 장 전부**다. GT는 표시·사후 평가에만 사용했다. 노란 raw keypoints는 그대로이고, 빨간 PnP 후보만 바뀐다. 회복뿐 아니라 잘못 바뀐 사례도 포함한다.','',
        '![all changed moderate cases](figures/stage3/06_scorer_changed_cases.jpg)','',
        '## 6. Stage4 — clean-preservation router','',
        f"판정 `{d4['primary']}`. 두 expert 모두 {s4['base_selector']}로 통일했다. 합성 clean/occluded 쌍과 exact ADDnorm 우열로 router MLP 한 개만 학습했다. TEST accuracy {sy4['accuracy']:.2%}, routed AUC {sy4['ROUTED']['ADDsym_AUC']:.6f}는 S0 {sy4['S0']['ADDsym_AUC']:.6f}보다 낮다. 즉 synthetic에서부터 expert 선택이 충분히 학습되지 않았다. 실사 실패를 synthetic→real gap 하나로 단정하지 않는다.",'',
        '|group|S0 AUC|S1 AUC|routed AUC|GT best expert AUC|route S0/S1|','|---|---:|---:|---:|---:|---|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        z=s4['groups'][g];ct=s4['routing'][g]['counts'];lines.append('|'+g+'|'+ '|'.join(f"{z[a]['sixD']['ADDsym_AUC']:.6f}" for a in ('S0','S1','ROUTED','POSTHOC_BEST_EXPERT'))+f"|{ct['S0']}/{ct['S1']}|")
    lines+=['', '전체 AUC가 +0.000445라는 작은 양수여도 Severe 손실과 전체 PCK10 감소가 있어 성공으로 판정하지 않는다. Stage4는 D9 기반이므로 Stage3 새 scorer의 이득을 합친 최종 수치가 아니다. [PCK5/10/20·6D 전체 표·가림 정책·비용](stage4_clean_preservation/STAGE4_REPORT_KO.md)','',
        '![routing cases](figures/stage4/07_routed_examples.jpg)','',
        '## 7. 병목 계층','',
        '|계층|이번에 확인한 것|남은 문제|','|---|---|---|',
        '|keypoint|raw2D는 Stage1–3 불변|큰 raw corner 오차는 이 selector가 직접 복구하지 않음|',
        '|candidate|좋은 W/D 후보가 이미 있는 사례 존재|oracle 자체가 완벽한 pose/정확한 GT를 뜻하지 않음|',
        '|selector|합성 선형 scorer가 S1 Moderate gap 37.8% 회수|남은 gap과 S0 악화; 모든 expert에 일반화 못함|',
        '|preservation|일부 clean 이득을 입력별로 고를 수 있음|Severe를 손상하지 않는 보존에는 실패 — 다음 주 병목|','',
        '## 8. Compute cost','',
        f"RTX3080 dual expert median {cost['timings_ms']['dual_total_ms']['median']:.2f}ms, P90 {cost['timings_ms']['dual_total_ms']['P90']:.2f}ms, peak allocated {cost['GPU_peak_allocated_MiB']:.1f}MiB / reserved {cost['GPU_peak_reserved_MiB']:.1f}MiB. batch1, 5회 warmup, 30장, 두 모델 sequential. RGB decode·checkpoint load 제외; GAP transfer·PnP·CPU router 포함. tiny CPU model wrapper 재구성 overhead가 포함된 보수적 pilot이며 Jetson/최적화 배포 수치가 아니다.",'',
        f"합성 clean feature extraction {C.read(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json')['seconds']:.2f}s, real context extraction {C.read(C.DOC/'REAL_FEATURE_LOCK.json')['seconds']:.2f}s, occluded synthetic extraction {C.read(C.sdoc(4)/'ROUTER_SYNTH_PREDICTION_LOCK.json')['seconds']:.2f}s. base optimizer step0. GPU temperature/process snapshots와 실측은 [COMPUTE_COST](stage4_clean_preservation/COMPUTE_COST.json)에 보존했다.",'',
        '![latency](figures/stage4/06_latency_cost.png)','',
        '## 9. 다음 딱 한 단계','',f"분기 **{route}**: {nextstep}",'',
        'S1의 새 selector 부분 회복은 보존하되, 다음에는 clean 출력 보존을 목표로 잡는다. 이번 실패 router를 즉시 distill하거나 새로운 seed/특징/학습량 sweep을 추가하지 않는다. 별도 지시 전 다음 학습은 시작하지 않았다.','',
        '## 10. 한계','',
        '- HELDOUT128은 recording-disjoint이지만 이미 열람한 DEV이며 독립 final TEST가 아니다. Stage1/3은 같은 Moderate21을 사용했다.',
        '- single seed42 base / small learners, 작은 실제 표본. 통계적 유의성이나 전범위 일반화를 주장하지 않는다.',
        '- real6D reference는 geometry-derived이며 독립 계측 GT가 아니다. camera-facing/180° role-contract warning은 유지했다. C4 remap이나 점별 matching으로 성능을 만들지 않았다.',
        '- 일반 플라스틱만 평가했다. 정사각형·목재·다른 도메인 성능은 이 결과로 판단하지 않는다.',
        '- synthetic TEST는 새 scorer/router의 보류 데이터지만 기존 base R0는 더 넓은 원천 합성을 학습했다. base-unseen TEST라고 부르지 않는다.',
        '- 그룹 분리 결과 합성 VAL/TEST는 P0/TEX 저양각 위주이며 HIGH elevation0장이다. 이 분포 차이를 보고 split을 다시 고르지 않았다.',
        '- 원래 S1 가림 policy는 real 학습용이었다. 이번에는 synthetic에 적용하고 기존 .5 schedule/coverage/paired feasibility를 유지했다. rectangle이 실제 적용되지 않은 쌍도 포함하며 applied-only를 따로 보고했다. random affine 전체 학습 pipeline의 완전 복제는 아니다.',
        '- GT best candidate/expert는 사후 비배포 진단일 뿐이다. real GT는 고정 결정 뒤 scoring/시각화에만 썼고, synthetic 학습 프로세스에는 읽기 금지 guard가 있다.','',
        '## 11. 재현·해시·커밋','',
        f"감사 **{audit['status']} / {audit['n_tests']}개**. 기존 입력 {audit['immutable_inputs']}개와 선택 합성 이미지 {audit['selected_synthetic_images_checked']}개 hash 확인. 가중치·gradient 불변, hook bit-exact, 순서 swap/tie 불변, split/파생 이미지 배제, VAL-only/TEST-once, decision-before-GT, exact population, 모든 그림을 검사했다.",'',
        '[실행 순서](../../../scripts/research/pallet_selector_recovery_v1/README.md) · [전체 입력 hash](INPUT_BINDINGS.json) · [자동 감사](FINAL_AUDIT.json) · [최종 판정](FINAL_DECISION.json)','',
        '|stage|commit|push|','|---|---|---|']
    for st in range(1,5):
        gr=C.read(C.DOC/f'STAGE{st}_GIT.json');lines.append(f"|{st}|`{gr['commit']}`|{gr['push']} / remote SHA 일치|")
    lines+=['','각 push 후 SHA 영수증은 다음 커밋에 넣어 self-reference를 피했다. 통합 보고서 커밋은 별도이며 최종 live remote SHA는 CLI에서 재확인한다. 대형 tensor/checkpoint/원본은 로컬 private namespace에, 작은 보고서·그래프·ROI 이미지는 Git에 보존했다. 다른 기존 작업 파일은 stage/수정하지 않았다.','']
    C.save(C.DOC/'REPORT_KO.md','\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args();globals()[f'stage{a.stage}']()
