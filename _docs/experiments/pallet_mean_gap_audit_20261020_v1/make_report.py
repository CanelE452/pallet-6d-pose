"""Build Korean analysis handoff from the published scalar ledgers."""
import argparse,csv,json,math,platform,sys
from pathlib import Path
from collections import Counter
import numpy as np
from analyze import read,rows,index,sha,write,csvwrite,SEEDS,PF

def rr(p):return list(csv.DictReader(p.open()))
def table(headers,values):
    return '\n| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in values)+'\n'

def run(root,data):
    a=root/'results_attempt2';p=root/'public';p.mkdir(exist_ok=False)
    import shutil
    shutil.copytree(a,p/'tables');shutil.copytree(root/'figures',p/'figures')
    for name in ('analyze.py','visualize_results.py','make_report.py'):
        shutil.copyfile(root/name,p/name)
    pair=rr(a/'image_paired.csv');cause=rr(a/'failure_coverage.csv');bank=rr(a/'exact_bank_coverage.csv')
    summaries=rr(a/'all_saved_methods_summary.csv');sm={(r['population'],r['arm']):r for r in summaries}
    co=rr(a/'gap_concentration.csv');pairs=rr(a/'paired_seed_summary.csv');priors=rr(a/'prior_trials_reaggregated.csv')
    cidx={(r['id'],r['seed']):r for r in cause};bankidx={r['id']:r for r in bank}
    contributions=[]
    for seed in SEEDS:
        for comparator in PF:
            rs=[r for r in pair if r['seed']==str(seed) and r['comparator']==comparator]
            for group in sorted({cidx[r['id'],str(seed)]['candidate_coverage_class'] for r in rs}):
                ss=[r for r in rs if cidx[r['id'],str(seed)]['candidate_coverage_class']==group]
                contributions.append(dict(seed=seed,comparator=comparator,cause_group=group,n=len(ss),delta_T_sum=math.fsum(float(r['delta_T_cm']) for r in ss),delta_R_sum=math.fsum(float(r['delta_R_deg']) for r in ss),positive_T_sum=math.fsum(max(0,float(r['delta_T_cm'])) for r in ss),negative_T_sum=math.fsum(min(0,float(r['delta_T_cm'])) for r in ss),positive_R_sum=math.fsum(max(0,float(r['delta_R_deg'])) for r in ss),negative_R_sum=math.fsum(min(0,float(r['delta_R_deg'])) for r in ss),bank_eligible_success_possible=sum(bankidx[r['id']]['HEAD_eligible_good']=='True' for r in ss)))
    csvwrite(p/'tables/cause_contributions.csv',contributions)
    top=[];s18=[r for r in pair if r['seed']=='20261018' and r['comparator']==PF[0]]
    for tag in ('T_cm','R_deg'):
        for rank,r in enumerate(sorted(s18,key=lambda x:(-float(x['delta_'+tag]),x['id']))[:20],1):
            top.append(dict(metric=tag,rank=rank,**r,**{k:v for k,v in cidx[r['id'],'20261018'].items() if k not in r},raw_generated_good=bankidx[r['id']]['raw_generated_good']))
    csvwrite(p/'tables/top_error_diagnosis.csv',top)
    oracle=[]
    for group in ['ALL_PRIMARY','GOOD_CANDIDATE_NOT_SELECTED','NO_SUCCESS_CANDIDATE_IN_EXACT_BANK','SUCCESS_CONTROL']:
        ss=[r for r in bank if r['partition']!='CAL' and (group=='ALL_PRIMARY' or cidx[r['id'],'20261018']['candidate_coverage_class']==group)]
        oracle.append(dict(group=group,n=len(ss),bank='exact original D1 eligible HEAD candidates',success_possible=sum(r['HEAD_eligible_good']=='True' for r in ss),same_pose_minimax_T_mean_cm=float(np.mean([float(r['oracle_same_pose_T_cm']) for r in ss])),same_pose_minimax_R_mean_deg=float(np.mean([float(r['oracle_same_pose_R_deg']) for r in ss])),realized_performance=False))
    csvwrite(p/'tables/oracle_by_cause.csv',oracle)
    # Only two hypotheses considered; no evaluation-tuned intervention is fitted.
    raw=rows(data/'pallet-bi-full-execution-20261019-v1/P0_BASELINE_REAGGREGATION/SCORED_ROWS.jsonl.gz')
    refs=read(data/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']
    cal=[]
    for s in SEEDS:
        er=np.array([np.array(r['poses'][f'D1_H1_MLP_s{s}']['centroid'])-refs['CAL'][r['id']]['t'] for r in raw if r['partition']=='CAL'])*100
        cal.append(dict(seed=s,CAL_n=len(er),depth_positive_n=int(sum(er[:,2]>0)),depth_negative_n=int(sum(er[:,2]<0)),depth_mean_error_cm=float(np.mean(er[:,2])),depth_median_error_cm=float(np.median(er[:,2])),new_calibration_fit=False))
    csvwrite(p/'tables/calibration_bias_diagnostic.csv',cal)
    hypotheses=[dict(hypothesis='H1_global_scale_or_depth_correction',status='NOT_RUN_INSUFFICIENT_CAUSAL_SUPPORT',evidence='Top5 depth-error signs split:2 positive/3 negative; CAL46 also mixed. A scalar scale change cannot fix orientation or missing good candidates.',proposed_minimal_action='Only if coherent bias were established: fixed global correction fit on permitted calibration and gate on synthetic validation.',side_effect='May move120 successes across5cm boundary and worsen opposite-sign depth failures; rotation unchanged.',new_fit=0,new_inference=0,reason_for_stop='No coherent global bias or independent camera/geometry registration error identified; do not tune on PRIMARY tail.'),dict(hypothesis='H2_physical_owner_bound_finite_observation_search',status='BLOCKED_MISSING_INDEPENDENT_CORRESPONDENCE',evidence='41 no-good bank frames; observations/geometry/reference causal separation unresolved. Prior outline deletion, covariance, RGB, union and selectors already tested.',proposed_minimal_action='Range-limited search bound to actual finite material edge/owner, preserving ambiguity when uncertified.',side_effect='A false owner/range guard can reject correct candidates and remove successful outline protections.',new_fit=0,new_inference=0,reason_for_stop='No independent real physical correspondence/owner labels; new manual labels/data prohibited by scope. Do not fabricate labels or use reference projection as observation.')]
    csvwrite(p/'MINIMAL_INTERVENTIONS.csv',hypotheses)
    checks=read(a/'VALIDATION.json')
    # Report values are machine-bound for an independent documentation check.
    claims=[]
    def claim(name,v):claims.append(dict(name=name,value=v));return v
    basearms=[*PF,*[f'D1_H1_MLP_s{s}' for s in SEEDS]]
    metrics=[]
    for arm in basearms:
        r=sm['PRIMARY192',arm]
        metrics.append([arm,r['successes']+'/192',r['T_n']+'/'+r['R_n'],f"{float(r['T_mean_cm']):.6f}",f"{float(r['R_mean_deg']):.6f}",f"{float(r['T_median_cm']):.3f}/{float(r['R_median_deg']):.3f}",f"{float(r['T_P90_cm']):.3f}/{float(r['R_P90_deg']):.3f}",f"{float(r['T_P95_cm']):.3f}/{float(r['R_P95_deg']):.3f}",f"{r['T_gt50cm']}/{r['T_gt100cm']}/{r['R_gt45deg']}"])
        for key in ('successes','T_mean_cm','R_mean_deg'):claim(arm+'_'+key,float(r[key]))
    ct=[x for x in co if x['seed']=='20261018' and x['comparator']==PF[0]]
    for r in ct:
        for key in ('positive_sum','negative_sum','net_sum','top_fraction_of_positive'):claim(r['metric']+'_top'+r['top_k']+'_'+key,float(r[key]))
    casefiles={r['id']:r['file'] for r in read(root/'figures/PANEL_MANIFEST.json')['cases']}
    top5=sorted(s18,key=lambda x:-float(x['delta_T_cm']))[:5]
    cases_table=[]
    for r in top5:
        c=cidx[r['id'],'20261018'];cases_table.append([f"[{r['id']}](figures/{casefiles[r['id']]})",f"{float(r['delta_T_cm']):+.3f}",f"{float(r['method_T_cm']):.3f}/{float(r['method_R_deg']):.3f}",f"{float(r['comparator_T_cm']):.3f}/{float(r['comparator_R_deg']):.3f}",c['candidate_coverage_class'],f"{100*float(c['depth_error_energy_fraction']):.1f}%",f"{float(c['signed_depth_error_cm']):+.3f}"])
    now='2026-10-20';head='93c24d143fa3901de1fa127b3156e8250d81dba8'
    text=f'''# PRIMARY192 평균 격차 분해와 소규모 개입 판단 ({now})

**목표한 평균 우위는 확보하지 못했다.** D1의 위치 평균은 세 seed 모두 두 PoseFix+SubPix 경로보다 높다. seed20의 회전 평균은 두 경로보다 낮지만 같은 최종 자세의 위치 평균이 높으므로 양 평균 개선 성공이 아니다. 이번 작업은 저장 예측의 독립 재평가·원인 진단을 완료했고, 근거 없는 새 보정·학습은 실행하지 않았다. 기존 원고·슬라이드·숫자 원장·운영 기본값은 보존했다.

## 1. 검증한 평균 격차

동일 image ID/hash·원녹화 recording·카메라 K·알려진 물리 치수·기존 physical 축·같은 고정 참조를 사용한다. PRIMARY192=TEST49+REMAINDER143, CAL46은 별도다. 성공은 **한 최종 자세**의 T≤5cm AND C2 회전≤5°다. C2는 I/Y180°만 허용하며90°는 동치가 아니다. 참조는 기존 기하 재구성값이며 **독립 물리 GT가 아니다**. 실사는 모두 반복 개발 자료이고 독립 신규 test는0장이다.

아래5경로는 전체192/산출192/no-pose0/오차결측0이다. 따라서 이번5경로에서 available-only 평균과 공통 산출192 paired 평균의 분모가 우연히 같으며, 이를 일반적인 no-pose 처리 규칙으로 확대하지 않는다. [영상별 계약](tables/identity_contract.csv), [전체74경로×4집합](tables/all_saved_methods_summary.csv).
'''
    text+=table(['경로','성공','T/R 산출','T 평균 cm','R 평균 °','중앙 T/R','P90 T/R','P95 T/R','T>50/>100/R>45'],metrics)
    text+='\n[6개 seed×비교군 paired 요약](tables/paired_seed_summary.csv): Δ는 D1−비교군, 양수는 악화. 회복/손상은 비교군→D1이다.\n'
    text+=table(['seed','비교군','공통 산출','ΔT cm','ΔR °','회복/손상','공동성공/공동실패'],[[r['seed'],r['comparator'],r['common_available'],f"{float(r['paired_delta_T_cm']):+.6f}",f"{float(r['paired_delta_R_deg']):+.6f}",r['RECOVER']+'/'+r['DAMAGE'],r['BOTH_SUCCESS']+'/'+r['BOTH_FAILURE']] for r in pairs])
    text+='\n**재현 검산:** 238개 ID/hash 모두 고유, 중복·누락0. 저장 scalar29,988개를 원 scorer import 없이 반환 R/t에서 다시 계산하여 T/R 최대차0, 성공 불일치0. 공식40군×4집합의3,200개 값도 최대차0이다. 서로 다른 저장 평가 간12,376개 중복 경로 값도 차0이다. [검산 JSON](tables/VALIDATION.json), [공식 값 대조 CSV](tables/baseline_official_checks.csv). 원 float 행렬을 정규직교화하거나 단위를 조정하지 않았다.\n\n![전체192 산점도](figures/all192_scatter.png)\n'
    text+='''
## 2. 원인 근거와 평균 기여

대표 seed18/PF-anchor 대비 위치 양의 악화 합은1031.496918cm, 개선 합은−604.430753cm, 순차는427.066165cm(192로 나누면+2.224303cm)다. 회전은+680.951416°−486.999284°=+193.952132°(+1.010167°). net을 분모로 한 기여율은 상쇄 때문에100%를 넘을 수 있어 **양의 악화 총합 대비 비율**도 별도로 표시했다.
'''
    text+=table(['항목','상위 k','양의 악화 기여율','net 대비 기여율','나머지 영상의 signed 합'],[[r['metric'],r['top_k'],f"{100*float(r['top_fraction_of_positive']):.2f}%",f"{100*float(r['top_fraction_of_net']):.2f}%",f"{float(r['remainder_signed_sum']):+.6f}"] for r in ct])
    text+='''
상위10%는 ceil(192×0.1)=20장이다. **위치85장, 회전83장에서 악화**하므로 모든 문제가5장에만 있다는 설명은 틀리다. 동시에 상위5장의 양의 기여가 위치58.72%/회전72.59%, 상위20장은87.06%/94.05%로 tail이 평균 격차를 강하게 지배한다. 중앙 오차는 D1이 더 낮고, 나머지 signed 합은 음수다. 이는 사후 기술적 분해이며 어떤 영상도 평가에서 제거하지 않았다. 순수한 균일 전역 오차보다 **촬영군에 집중된 큰 손상과 광범위한 소오차 변화가 섞인 상태**로 해석한다.

![signed 평균 격차](figures/signed_gap_budget.png)

REC_002의77장은 위치 차합+614.437798cm/회전+166.839937°이며 나머지 촬영군의 개선과 상쇄된다. REC_001의66장은 위치−117.675934cm이면서 회전+23.971057°다. 단일 촬영군이나 단일 지표로 전체 결론을 대체하지 않는다. [recording별 전량](tables/recording_summary.csv), [session별](tables/session_summary.csv), [성공/실패 층별](tables/success_failure_strata.csv).

![recording별 기여](figures/recording_contributions.png)

**동일 원 D1 은행의 성공 가능151/192**를 재확인했다. 대표 seed18은151 중120 성공/31 미선택, 나머지41은 은행에 공동 성공 후보가 없다. seed19/20의 선택손실은34/30이다. 원 generated bank와 eligible HEAD bank에서 각각151이고, PF 두 완성 pose를 준입 전 단순 union하면154다. union154는 원 D1의 상한이나 실현된 성능이 아니다. [정확 은행별 coverage](tables/exact_bank_coverage.csv), [원인별 같은-pose minimax oracle](tables/oracle_by_cause.csv). oracle은 같은 한 후보의 T/R를 쓰며 두 독립 최저값을 결합하지 않았다.
'''
    text+=table(['대표 seed18/PF-anchor 층','n','ΔT 합 cm','ΔR 합 °'],[[r['cause_group'],r['n'],f"{r['delta_T_sum']:+.6f}",f"{r['delta_R_sum']:+.6f}"] for r in contributions if r['seed']==20261018 and r['comparator']==PF[0]])
    text+='\n상위 위치 악화5장은 다음과 같다. depth 비중은 번역오차 제곱에서 카메라 z축 성분이 차지하는 비율이며 원인 확정이나 개입 문턱이 아니다.\n'
    text+=table(['영상/그림','ΔT cm','D1 T/R','PF T/R','은행 진단','depth 제곱 비중','signed Δz cm'],cases_table)
    text+='''
상위5장 중4장은 후보 부재, `031571`은 성공 후보 미선택이다. `031571`의 E0는0.671cm/1.872°인데 H1은156.064cm/83.341°를 반환했다. `031426`은176.535cm/86.697°, `031671`은171.880cm/88.794°로 depth와 orientation이 함께 틀린다. 단지 회전이90° 근처라는 이유만으로 순수 W/D 전환 오류라 부르지 않았다. `005249`는164.209°여서90° 전환만으로 설명되지 않는다. [영상별 전체 분류](tables/failure_coverage.csv)의 `rotation90_compatible_diagnostic`은 C4를 진단용으로 적용했을 때5° 이내가 되는지 표시하며 공식 C2 점수에는 영향을 주지 않는다. 대표 seed18은 이 엄격 조건4장뿐이다. 나머지는 원 수치와 UNKNOWN을 유지한다.

physical owner mismatch와 참조 정확성은 **UNKNOWN**이다. saved B/I 관측·선과 투영이 충돌할 수 있어도 mask 오류, 실제 재료 edge 불일치, 등록 축/치수, 재구성 참조 한계를 현재 자료로 인과 분리하지 못했다. 원 비용 진단은 G13 같은-family22장에서 outline이22/22 잘못된 선택을 지지했지만 성공114장에서는114/114 올바른 선택을 보호했다. 그 원장과 이번 D1의31/41 집합은 서로 다른 정확 은행/단계이므로 합치지 않았다. 기존 [선택 점수 진단](../pallet_selection_score_diagnosis_v1/RESULTS_KO.md)을 따른다.

## 3. 이미 한 것과 구분되는 개입 검토

두 가설만 검토했다. 기존 순위/연속손실·F1 기하·RGB·guarded switch·outline 제외·후보 union·covariance·image/contour/PL은 새 아이디어로 반복하지 않았다. 새 실행 대신74개 경로를 동일 scalar 계약에서 재집계했다. [기존 trial 재집계](tables/prior_trials_reaggregated.csv)에 불리한 seed와 미채택 trial도 유지했다.

1. **전역 scale/depth 보정 — 근거 부족으로 미실행.** 상위5장의 z오차는 양2/음3으로 갈린다. CAL46의 seed18도 양16/음30, signed depth 평균−1.506cm/중앙−0.392cm다. 이 사실만으로 전역 scale 오류를 식별하지 못하고 회전 오류도 고칠 수 없다. 이 가설은 실패 영상만 맞추는 문턱을 만들지 않고 중단했다. [CAL 진단](tables/calibration_bias_diagnostic.csv).
2. **물리 owner에 묶인 유한 관측 검색 — BLOCKED.** 원인에 맞는 최소 개입 후보는 실제 대응 edge/owner가 확인된 범위에서만 관측을 질의하는 것이다. 기존 finite guard/outline 삭제와 구별되는 핵심은 독립적인 물리 대응 근거다. 그러나 해당 실사 레이블은 없고 신규 수작업 레이블·수집은 이번 범위에서 금지되어 있다. 재구성 참조의 투영을 가짜 대응 정답으로 만들지 않았다. 잘못된 owner guard가 성공 후보와 보호 관측까지 제거할 부작용도 있다.

[최소 개입 검토/중단표](MINIMAL_INTERVENTIONS.csv). 이번 작업의 새 fitting/학습/selector forward/PnP/자세 최적화/영상 모델 호출은 모두0이다. 실사 GT를 추론 입력으로 사용하거나 frame별 방법을 선택하지 않았다. 개선이 확인되지 않았으므로 새 개입의 세 seed 확대 실행도0이다. GPU가 없어서 중단한 것이 아니라 과학적 근거·독립 대응이 부족해서 중단했다.

## 4. 실제 결과와 trade-off

기존 방법 중 작은 선택기 변경으로 평균이 내려간 예는 있다. 대표 `D1_H1_PAIR_MLP`는117/192, T10.391721cm/R8.891290°로 원 D1보다 평균이 낮지만 PF 두 경로의 위치 평균보다 높고 성공3장을 잃는다. 기존 `R_CONT_F0`는117/119/120성공, T≈10.97–10.98cm로 역시 위치 평균 격차를 닫지 못한다. 기존 guarded seed18은116/192,11.385cm/10.246°이며 원120보다4장 감소한다. PL123/123/122는 별도 확장이고 T11.488cm로 평균 우위가 아니다. F_BI109/117/107도 원 D1과 다른 방법이며 대표11.733cm/9.605°다.

74경로 중 두 PF보다 T/R 평균이 동시에 낮은 경로는0개다. 이 표에서 유리한 회전 seed와 유리한 위치 변형을 합쳐 하나의 출력이라고 보고하지 않았다. 두 PF 설정도 결과에 따라 하나로 대표 선정하지 않았다. 이번 결과로 성공률만 높여 평균 목표를 달성했다고 주장하지 않는다.

## 5. 남은 한계와 다음 행동

우선순위는 REC_002의 후보 부재41집합과 선택손실31집합을 분리해 물리 대응·등록 축/instance geometry를 독립적으로 확인하는 것이다. 현재 범위에서는 레이블을 만들거나 새 대규모 학습을 시작하지 않는다. 독립 증거가 확보되기 전에는 물리 owner 교정의 성능을 약속할 수 없다. 허용 train/calibration/synthetic validation에만 선택·튜닝하고 실제 평균·성공·tail 비악화를 보인 최소 pilot만 향후 세 seed로 확대해야 한다. 독립 신규 실사 확인은 별도 범위 결정이 필요하다.

이 보고서는 연구 실험 handoff이며 논문/발표자료를 수정하지 않는다. 추론 성능은 세션 안내상 GPT-6 기반 Codex이며 실제 Pro 옵션·정확한 런타임 모델/추론 등급은 확인할 수 없다. high/xhigh를 ChatGPT Pro라고 부르지 않았다. 추가 유료 API/서비스 구매는0이다.

## 시각화 읽는 법

[23개 사례 HTML](figures/index.html), [선정 기준/ID](figures/CASE_SELECTION.csv), [그림 표기 원수치69개](figures/PANEL_VALUES.csv). 위쪽은 원본 RGB/빨강 D1/보라 PF-anchor, 아래쪽은 파랑 YOLO-anchor/초록 점선 GRef/저장 관측이다. GRef는 **기하 재구성 참조의 모델 투영**이다. 영상 좌표는 원해상도 왼쪽 위 원점, x오른쪽/y아래, 패널별 동일 축척이다. B주황/I청록 원은 저장 이미지 측정이며 fallback이 포함될 수 있다. 자홍 선은 저장된 실제 inlier 지지 구간이다. 철회된 full2D와 S를 이미지 측정으로 그리지 않았다. cuboid 선은 예측·참조 투영이며 실제 material edge의 인증이 아니다. 개인정보 redaction은 display 복사본에만 적용된 기존 검토 픽셀을 재사용했고 원추론·수치·참조는 바꾸지 않았다.

23개 source 화면 contact sheet와 대표 오류/성공/비식별 패널5개 및 전체 산점도/기여 그래프3개를 실제 픽셀로 확인했다. 나머지 패널은 생성·수치/해시 검산 대상이며 전량 pixel/물리 owner 검토를 했다고 주장하지 않는다. 공개 범위는 기존 연구 결과 패널 권한과 이번 사용자 게시 승인이다. 원 RGB bulk/전체 영상·체크포인트·후보은행·참조 R/t·개인정보·논문은 게시하지 않는다.
'''
    (p/'FINAL_REPORT_KO.md').write_text(text)
    csvwrite(p/'REPORT_VALUES.csv',claims)
    readme=f'''# 팔레트 평균 오차 격차 감사 — 2026-10-20

**평균 우위 확보 실패.** PRIMARY192에서 D1 seed18은12.072574cm/11.548887°, PF-anchor는9.848271cm/10.538719°, YOLO-anchor는9.377412cm/10.593944°다. 모든 경로192산출이며 누락 제외로 만든 차이가 아니다. 상위 tail, recording 분포와 후보 부재/선택손실을 분리했고 근거 없는 새 보정은 중단했다.

- [한국어 최종 보고서](FINAL_REPORT_KO.md): 평균 격차 → 원인 → 개입 구분 → 실제 결과 → 한계
- [baseline 독립 검산](tables/VALIDATION.json), [공식 수치 대조3,200개](tables/baseline_official_checks.csv)
- [영상별 paired1152행](tables/image_paired.csv), [seed별](tables/paired_seed_summary.csv), [recording별](tables/recording_summary.csv)
- [원인 분류714행](tables/failure_coverage.csv), [정확 은행coverage](tables/exact_bank_coverage.csv), [top20 진단](tables/top_error_diagnosis.csv)
- [기존74경로×4집합](tables/all_saved_methods_summary.csv), [기존 trial 재집계](tables/prior_trials_reaggregated.csv), [두 개입 검토/미실행](MINIMAL_INTERVENTIONS.csv)
- [대표6패널 이미지23개](figures/index.html), [입력/출력 SHA](tables/SOURCE_MANIFEST.json), [재현 명령](REPRODUCE.md)

![평균 격차의 양/음 기여](figures/signed_gap_budget.png)

![좋은 후보가 있지만 미선택된 사례](figures/cases/case_03.jpg)

그림은 빨강 D1/보라 PF-anchor/파랑 YOLO-anchor/초록 점선 기하 참조다. 같은 RGB·K·물리 치수와 기존 physical C2 계약이다. 참조는 독립 물리 GT가 아니다. 전체192 평가는 유지하며 위 사례를 제외하거나 정답으로 방법을 선택하지 않았다. HTML은 내려받아 열면 되고 GitHub에서는 각 JPEG를 직접 볼 수 있다.

소스 기준 commit `{head}`. 원 사용자 checkout `{read(root/'PREFLIGHT.json')['repositories']['/home/minjae/Documents/github/pallet-pose']['head']}`의 dirty 상태는 보존했다. 새 학습/forward/PnP0; 원고·슬라이드 수정0. 최종 게시 commit은 이 파일을 포함한 GitHub commit permalink로 고정한다.
'''
    (p/'README.md').write_text(readme)
    repro=f'''# 재현 및 환경

Ubuntu22.04.5 LTS, minjae-MS-7D17, i9-11900(16 logical CPU),31GiB RAM. 호스트 RTX3080 10GiB/driver580.178.04/CUDA13.0 확인. 확인 당시 GPU≈992MiB/7%는 데스크톱·rustdesk 사용이며 학습 작업 점유는 보이지 않았다. sandbox ps/GPU는 격리되므로 호스트 읽기 전용 확인을 따로 수행했다. 작업 종료·GPU 선점 없음.

Python `{sys.version.split()[0]}`, NumPy `{np.__version__}`. 정확한 dependency 버전은 ENVIRONMENT.json. 모든 계산CPU1, BLAS/OMP1, GPU0. 분석CLI wall={checks['wall_seconds']:.6f}s이며 전체 작업/시각 검토 시간이나 영상추론 latency/FPS가 아니다.

기준 저장소 remote `https://github.com/CanelE452/pallet-6d-pose.git`, 공개 저장소, 원격 main 확인SHA `{head}`. 사용자 원 checkout은 다른 오래된 main과 dirty 변경을 보존했다. 최신 worktree는 위SHA와 기존 untracked draft를 보존했다. 현재 작업의 새 파일만 별도 게시 checkout에서 commit한다.

아래 DATA_ROOT는 기존 local 연구 캐시 디렉터리다. 원 RGB·참조·체크포인트·후보은행은 공개 패키지에 포함하지 않으며 없는 입력을 재구성/다운로드하지 않는다. 관련9개 보고서는 source repo `_docs/experiments`에서 모두 찾았다. 다른 모집단으로 대체하지 않았다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/pallet-mean-gap-mpl
PY=/home/minjae/anaconda3/envs/pallet-pose/bin/python
DATA_ROOT=/home/minjae/Documents/pallet-research
SOURCE_REPO=/tmp/pallet-bi-full-execution-20261019-v1
NEW_OUT=/tmp/pallet-mean-gap-independent-reproduction
$PY -B analyze.py --data-root "$DATA_ROOT" --source-repo "$SOURCE_REPO" --out "$NEW_OUT"
$PY -B visualize_results.py --data-root "$DATA_ROOT" --analysis "$NEW_OUT" --out /tmp/pallet-mean-gap-new-figures
$PY -B verify.py --public . --data-root "$DATA_ROOT" --output /tmp/pallet-mean-gap-check.json
```

analyze.py/visualize_results.py는 새 출력만 생성하고 기존 출력 overwrite를 거부한다. make_report.py는 실제 local 실행 폴더(results_attempt2, figures, PREFLIGHT.json)를 받아 문서·공개 subset을 조립하는 packaging 전용 도구다. 표는 원정밀 CSV, 문서는 반올림 요약이다. 공식 연속 지표는 available-only; paired 공통 산출은 별도 표다. 실패/no-pose가0오차로 대입되지 않는다. median/linear P90/P95·sample variance(ddof1),5cm/5° 공동성공·엄격한 큰오차 횟수를 사용한다. Δ는 method−comparator이다.

입력 체인: 원 D1 real238 upstream/HEAD bank/봉인 predictions → P0/S1/S2/S3S4/F와 기존risk/observation evaluation scalar. 고정 reference seal SHA를 검증한 후 saved pose를 재채점한다. 238RGB hash·K·치수·partition·session·recording을 조인하고 원 saved returned pose5경로와 bitwise 일치를 확인한다. 그림은 같은 reference와 saved observation·pose만 표시한다. SOURCE_MANIFEST는 상대 alias/기존 basename/bytes/SHA를 보존하고 local 절대경로 mapping은 비공개로 남긴다.

기술 실패1회: 처음 reader는 모든 seal에 complete=true가 있다고 가정해 과거 risk/observation의 status=COMPLETE 형식에서 조기중단했다. 그 초기 소스·빈 출력은 local 보존했다. 실제7개 seal과 scored hash를 확인하고 exact COMPLETE 형식도 허용하도록 reader만 수정했다. 과학 설정·원 데이터 변경은0이다. 최종 분석은 results_attempt2 한 번 완료했다.
'''
    (p/'REPRODUCE.md').write_text(repro)
    import PIL,matplotlib
    write(p/'ENVIRONMENT.json',dict(os='Ubuntu22.04.5 LTS',host='minjae-MS-7D17',cpu='Intel i9-11900',logical_cpus=16,memory='31GiB',python=sys.version,numpy=np.__version__,pillow=PIL.__version__,matplotlib=matplotlib.__version__,gpu_used=False,threads=1,source_commit=head,source_original_head=read(root/'PREFLIGHT.json')['repositories']['/home/minjae/Documents/github/pallet-pose']['head'],model='GPT-6-based Codex per session metadata; exact runtime model/reasoning and Pro choice unavailable',paid_services_purchased=0))
    print(json.dumps(dict(public=str(p),files=len(list(p.rglob('*'))),new_methods_run=0,prior_methods=checks['all_method_count'],hypotheses=2)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();run(a.root,a.data_root)
