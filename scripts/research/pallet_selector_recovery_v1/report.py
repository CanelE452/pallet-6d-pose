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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args();globals()[f'stage{a.stage}']()
