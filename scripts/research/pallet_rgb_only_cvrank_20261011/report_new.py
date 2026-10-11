"""Korean paper decisions from sealed observations; no model or input access."""
import argparse, json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def n(v,d=3):return 'NA' if v is None else f'{v:.{d}f}'
def run(args):
    doc=Path(args.public_dir);metrics=read(doc/'METRICS.json');paired=read(doc/'PAIRED_RESULTS.json')
    groups={k:v['common_seed12'] for k,v in metrics['groups'].items()}
    selection=read(doc/'SYNTHETIC_SELECTION.json');train=read(doc/'RGB_ONLY_TRAIN_RECEIPTS.json')
    transitions=read(doc/'PAIRED_TRANSITION_SUMMARY.json')['pairs'];cost=read(doc/'LATENCY.json')['methods']
    compute=read(doc/'COMPUTE_RECEIPT.json')
    strict=lambda m:groups[m]['strict_success']['successes']
    strong=lambda p:p['effect_pp']>0 and p['CI95'][0]>0 and p['p_Holm']<.05
    combined=all(strong(p) for p in paired['primary'][:2])
    simplified='RGB_ONLY' if strict('RGB_ONLY_SUBPIX')>strict('M1_SUBPIX') else 'M1'
    recommendation='M3' if combined else simplified
    decision= '결합 M3의 두 정보 증분이 확인됐다' if combined else '특징·RGB 결합의 두 단독 입력 대비 증분을 입증하지 못해 결합의 필수성 주장을 제외한다'
    lines=[f"RGB_ONLY 두 시드 총 {train['actual_updates']:,} updates와 CVRANK 동일 110장 재평가를 완료했으며, {decision}.",'',
        '문제는 기존 특징과 원영상 RGB를 함께 쓰는 M3가 각 단독 입력보다 실제 6D에서 나은지, 동일 후처리를 허용한 비학습 보정과 비교해 추가 복잡성을 남길 근거가 있는지다. 이 문서는 고정된 입력 절제와 비교 조건에서 그 두 질문에 답한다.','',
        '[확인] 사용자 범위대로 Severe를 제외한 **Clean 68 + Moderate 42 = 110장, 9세션**만 평가했다. 동일 두 seed의 220개 예측을 독립 영상 220장으로 세지 않는다. 수동 2D 코너로 재구성한 기하 참조를 사용했으며 독립 물리 센서 참조가 아니다. 이전 245장/319장 평가와 합치지 않는다([PLAN_AND_CONTRACT.json](PLAN_AND_CONTRACT.json), [입력 대조](INPUT_REUSE_CONTRACT.json)).','',
        '[확인] 기존 FP32/TF32-off Base의 초기 코너·예측 상자·선택 객체·support·K·등록 치수를 모든 방법에 공통 사용한다. 구형 TF32 캐시의 q0를 새 입력으로 쓰지 않았다. 기존 완료 결과 2,200행은 byte contract를 검산해 재사용했고, RGB_ONLY 440행과 CVRANK 660행을 추가해 **15경로 × 110장 × 2seed = 3,300행**을 공개한다([원행](ROW_METRICS.jsonl.gz), [재사용 검산](EVALUATION_REUSE_VERIFICATION.json)).','',
        '[확인] RGB_ONLY는 후단에 p3/p4를 넣지 않는 입력 절제다. frozen YOLO가 준 초기점·상자·support와 role/context/변위/coverage는 유지한다. OLD_P의 동일 seed last6000에서 warm-start하며 M3와 공통 초기 파라미터 값 및 RNG 생성 순서를 맞췄다. 동일 TRAIN 배치 순서, AdamW, batch16, 각 3,000 updates, CE + native 기대좌표 SmoothL1(beta1), COORD=0.15214983019356332, GEO=0이다. 기존 모델 재학습·새 geometry cache·추가 이미지·추가 seed·성능 기반 설정 변경은 0이다([학습 영수증](RGB_ONLY_TRAIN_RECEIPTS.json)).','',
        '[확인] base regular(64), base null(61), RGB residual(81) 입력의 앞 48차원을 모두 명시적 0으로 만들고 adapt3/adapt4/patch_body 호출을 생략했다. pre/post-fit 실제 TRAIN probe에서 원본·0·다른 영상 특징·특징 누락에 대해 base/residual/final logits·확률·decode 좌표가 불변이었다. 학습 후 RGB 변경 반응, 실제 좌표 gradient와 유한차분도 확인했다([FEATURE_EXCLUSION_AUDIT.json](FEATURE_EXCLUSION_AUDIT.json), [구현](../../../scripts/research/pallet_rgb_only_cvrank_20261011/rgb_model.py#L32)). 단순 zero-image adapter나 M3 logits 재사용이 아니다. probe의 decode 좌표는 network pixel frame이며 실사 native 좌표/6D 지표와 구분한다.','',
        '[확인] 공통 가중치 초기값은 일치하지만 특징을 제거한 step0 logits는 원래 full-feature 출력과 같지 않다. 동일 가중치 control 대비 최대 logits 차이는 seed1 15.7368, seed2 12.6822다. 특징 치환 불변 검산은 RGB_ONLY 내부의 독립성을 뜻한다.', '',
        '[확인] 파라미터는 serialized 24,387, 실행되지 않는 전체 특징 모듈 10,640, 학습/실행 tensor 원소 13,747, 0 입력 scorer 열 7,680, 입력에 반응하는 원소 6,067로 구분한다. 0 입력 열도 같은 Linear tensor에 남고 AdamW decay로 값이 바뀔 수 있다. 같은 serialized shape를 같은 유효 표현력이라고 주장하지 않는다. 검출기 전체와 기존 특징 capture는 그대로이며 후단 특징 사용만 제거했다.','',
        f"[확인] calibration 1,004장에 기존 fixed-index Gaussian CE와 T∈{{0.5,1,2,4}}를 적용했다. RGB_ONLY 선택 T는 seed1={read(doc/'CALIBRATION.json')['temperatures']['RGB_ONLY_seed1']['temperature']}, seed2={read(doc/'CALIBRATION.json')['temperatures']['RGB_ONLY_seed2']['temperature']}다. 고정 C2 selection 782장×2seed에서 좁은 후보 M1/M3/RGB_ONLY 중 **{selection['selected_arm']}**를 실사 RGB_ONLY 추론 전에 선택했다. signed-front C1 249장 제외 계약은 그대로이며 새 solver 수선/기하 캐시를 만들지 않았다. 이전 전체 screen의 사전 선택 M4 기록은 유지한다([합성 선택](SYNTHETIC_SELECTION.json)).",'',
        '| 합성 후보 | 성공/1564 | T 중앙값 cm | 활성 원소 | 결과 출처 |','|---|---:|---:|---:|---|']
    for m,p in selection['arms'].items():lines.append(f"| {m} | {p['success_count']} | {n(p['T_median_cm'])} | {p['active_parameter_elements']} | {p['source']} |")
    lines += ['', '[확인] CVRANK는 기존 반경12px와 source calibration 설정을 유지했다. Shi–Tomasi/Harris/선분 교점/기울기 교점 후보를 하나의 selector로 합친 한 방법이며 독립 방법 4개로 세지 않는다. 과거 native 좌표는 현재 q0와 **0/110 exact, 최대 0.28955078125px 차이**여서 재사용하지 않았다. 현재 Base에서 110장을 CPU로 한 번씩 재계산하고 동일한 두 seed에 정렬했다. GT나 참조 가시성으로 후보를 고르지 않았다([CVRANK 영수증](CVRANK_COMPUTE_RECEIPT.json), [cap/순서 검산](CVRANK_COORDINATE_VERIFICATION.json)).','',
        '[확인] 학습형 최종 경로는 **uncapped 기대좌표 → SubPix → 원래 Base 기준 1% 영상 대각선 cap 한 번**이다. CVRANK_NATIVE 무상한, CVRANK_CAP1 최종 cap, CVRANK_THEN_SUBPIX 무상한 → SubPix → 최종 cap을 모두 남긴다. SubPix는 win(5,5), zeroZone(-1,-1), EPS|COUNT 40, 0.001이며 자세는 동일 production F/SQPnP→RefineLM이다. K·등록 치수는 RGB_ONLY에서도 PnP에 계속 필요하다.','',
        '[확인] 성공은 T<5cm **AND** R<5°이며 미산출은 실패로 분모 유지한다. 연속 수치는 각 영상 안 산출된 두 seed 값을 평균한 뒤 110개 영상의 평균·표본 SD(ddof1)·중앙값·P90·P95를 계산한다. seed별 수치도 METRICS에 모두 있다. 아래 CDF는 220개 예측 원행을 쓰므로 요약 표와 단위를 구분한다.','',
        '| 방법 | 성공/220 (%) | T cm 평균±SD | T 중앙/P90/P95 | R ° 평균±SD | R 중앙/P90/P95 | 미산출 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,p in groups.items():
        t,r=p['metrics']['Tcm'],p['metrics']['Rdeg']
        lines.append(f"| {m} | {strict(m)}/220 ({n(100*p['strict_success']['rate'],2)}) | {n(t['mean'])}±{n(t['SD'])} | {n(t['median'])}/{n(t['P90'])}/{n(t['P95'])} | {n(r['mean'])}±{n(r['SD'])} | {n(r['median'])}/{n(r['P90'])}/{n(r['P95'])} | {220-p['pose_availability']['available']} |")
    lines += ['', '[확인] 모든 평균·분산·SD·중앙값·P90·P95·세션 bootstrap 평균 95% CI는 [METRICS.json](METRICS.json)와 [ACCURACY_TABLE.csv](ACCURACY_TABLE.csv)에 있다. ADDsym 및 IoU3D도 기존 구현을 유지했다. 직접 long/short 축 오류의 참조는 109장(UNCLEAR 1장 제외)이며 실제 산출된 selected hypothesis 분모를 별도로 기록한다. 이 제외는 전체 성공 분모 110장에 적용하지 않는다. 기존 R>45°·|yaw|≥60° confusion과 직접 축 오류를 다른 열로 보고한다.','',
        '| 방법 | 직접 축 오류/실제 분모 | 기존 R/yaw W/D % | ADDsym m 평균±SD | IoU3D 평균±SD | PCK@1/2/3/5/10 px % |','|---|---:|---:|---:|---:|---|']
    for m,p in groups.items():
        add,iou=p['metrics']['ADDsym_m'],p['metrics']['IoU3D']
        pck='/'.join(n(p['PCK'][str(t)]['rate']*100,2) for t in (1,2,3,5,10))
        lines.append(f"| {m} | {p['axis_wrong']['errors']}/{p['axis_wrong']['denominator']} | {n(100*p['legacy_WD_confusion']['rate'],2)} | {n(add['mean'])}±{n(add['SD'])} | {n(iou['mean'])}±{n(iou['SD'])} | {pck} |")
    lines += ['', '[확인] PCK는 동일 고정 참조 코너 분모에 대한 native 2D 오류이며 PnP 실패 때문에 0으로 바꾸지 않는다. 정확도 회복과 손상은 각각 세며 순효과만 제시하지 않는다.','',
        '| after − before | 회복 | 손상 | 코너 개선·자세 악화 | 손상 코너 수 | 개선 코너 수 |','|---|---:|---:|---:|---:|---:|']
    for p in transitions:
        lines.append(f"| {p['after']} − {p['before']} | {p['recovery']} | {p['damage']} | {p['corner_better_pose_worse']} | {p['corners_damaged']} | {p['corners_improved']} |")
    lines += ['', '[확인] 위 손상/회복은 각 대비 220개 정렬 예측에서 센다. 단독→SubPix의 모든 전이도 포함했다. 전체 3,520 전이는 [ALL_PAIRED_TRANSITIONS.csv](ALL_PAIRED_TRANSITIONS.csv), 모든 성공·미산출 전환 및 코너 개선·자세 악화 ID와 seed·T/R은 [CHANGED_CASES.csv](CHANGED_CASES.csv)에 남긴다. 손상과 큰 회전 꼬리를 숨기지 않는다.','',
        '| 사전 고정 strict 성공률 대비 | 효과 %p | 세션 95% CI %p | 원 p | Holm p (3검정) |','|---|---:|---:|---:|---:|']
    for p in paired['primary']:lines.append(f"| {p['after']} − {p['before']} | {n(p['effect_pp'])} | [{n(p['CI95'][0])}, {n(p['CI95'][1])}] | {n(p['p_raw'],6)} | {n(p['p_Holm'],6)} |")
    lines += ['',f"[확인] 합성 선택에서는 RGB_ONLY {selection['arms']['RGB_ONLY']['success_count']}/1564가 M3 {selection['arms']['M3']['success_count']}/1564보다 높았으나, 실사 +SubPix 성공은 RGB_ONLY {strict('RGB_ONLY_SUBPIX')}/220와 M3 {strict('M3_SUBPIX')}/220로 순서가 바뀌었다. 좁은 합성 선택은 실사 개선을 보장하지 않았다. 같은 실사에서 M3의 RGB_ONLY 대비 증분은 관측됐지만 M1 대비 증분은 family3 기준으로 입증되지 않았다.",'',
        '[추정] 이 차이는 합성에서 학습한 RGB 반응과 실사 정보 사이의 차이, 특징을 제거한 공통 warm-start 조건, 공통 PnP의 축 선택에 대한 국소 좌표 민감성 등으로 설명될 수 있다. 이번 실험은 그 원인을 각각 분리하지 않았으므로 어느 하나를 확정 원인으로 단정하지 않는다. 설정을 바꾸거나 추가 fit을 수행하지 않았다.']
    lines += ['', '[확인] 실제 9세션 단위 paired cluster bootstrap 10,000회(seed20260917), 같은 영상의 두 seed와 두 방법을 함께 재표집했다. 기존 draw의 ID/세션 순서·SHA가 일치해 그대로 사용했다. 원 p는 중심화한 양측 bootstrap의 exact integer 동률 및 plus-one 규칙이며 CI로 역산하지 않는다. M3−M1은 이미 관측한 개발 대비의 재사용이다. 새 family3는 이전 family5/4를 덮어쓰지 않으며 다른 비교는 기술 통계다. **비유의는 동등성/비열등성 증명이 아니다**([PAIRED_RESULTS.json](PAIRED_RESULTS.json)).','',
        f"[확인] 합성 사전 선택은 {selection['selected_arm']}로 고정한다. 아래 논문 구성 추천 **{recommendation}+SubPix**는 공개된 개발 결과를 보고 하는 사후 구성 판단이며 합성 선택 결과를 바꾸지 않는다. 지금까지 반복 노출된 110장이므로 독립 최종 시험이나 신규 검증 집합이라고 표현하지 않는다.",'',
        '| 구성/주장 | 과학적 관찰 | 이번 논문 구성 결정 |','|---|---|---|',
        '| frozen Base, 등록 K·치수, 동일 PnP | 모든 경로가 공통 사용 | 유지 |',
        '| 순정 SubPix와 각 +SubPix 절제 | 단독/결합의 성공·손상 모두 공개 | 유지: 후처리와 필수 비교군 |',
        f"| M1 기존 특징 보정 | 최종 {strict('M1_SUBPIX')}/220 | {'유지: 단순 구성 후보' if recommendation=='M1' else '비교군으로 보존'} |",
        f"| RGB_ONLY native RGB 보정 | 최종 {strict('RGB_ONLY_SUBPIX')}/220; 동일 OLD_P warm-start 조건의 실제 두 학습 | {'유지: 단순 구성 후보' if recommendation=='RGB_ONLY' else '비교군으로 보존; 핵심 구성에서 제외'} |",
        f"| M3 특징+RGB 결합 | 최종 {strict('M3_SUBPIX')}/220; 양 단독 정보 대비 family3 | {'유지: 양쪽 증분 근거 확인' if combined else '결합의 필수성·상보성 핵심 주장 제외; 전체 실험 보존'} |",
        f"| CVRANK + 동일 SubPix | {strict('CVRANK_THEN_SUBPIX')}/220; 기존 calibration된 비학습 대조 | 유지: 학습 없는 비교군; 모든 고전법의 최적법으로 일반화 금지 |",
        f"| PoseFix-style +SubPix | {strict('POSEFIX_SUBPIX')}/220; 같은 성공률도 동등성 증명 아님 | 유지: 외부 스타일 비교군; 이기지 못한 결과 그대로 |",
        '| 기존 N3/M0/M2/M4 | 이전 완료 결과 및 M4 선택 이력 보존 | 이번 주표에서 제외; 이전 기록/보충 자료로 연결 |',
        '| Efficient/Pareto-optimal·새 치수/여러 backbone·독립 실사 시험 | 이 실험이 검증하지 않음 | 주장 제외 |','',
        '[확인] 논문에 쓸 문장: “동일 frozen 기반 검출기와 공통 warm-start/replay 조건에서 특징-only, RGB-only, 결합 및 calibration된 고전 보정을 동일 비가림/중간가림 개발 집단과 PnP로 비교하였다. 전체 회복·손상과 paired session 불확실성을 공개하였다.”','',
        ('[확인] 두 단독 정보 대비 성공률 증분을 확인한 범위에서만 결합의 관측 이점을 기술한다.' if combined else '[확인] 논문에서 뺄 문장: “기존 특징 재사용과 RGB 결합이 반드시 필요하다”, “두 정보를 합쳤기 때문에 상보성이 입증됐다”. 관측된 작은 성공률 차이만으로 필수성을 주장하지 않는다.'),'',
        '[확인] 학습형의 필수성, 처음부터 학습한 모든 RGB-only의 불가능성, 모든 고전법 대비 우월, 독립 최종 정확도·비열등성, 효율/Pareto 주장은 쓰지 않는다. 비용을 주된 연구 목표로 바꾸지 않는다. 큰 회전은 raw RGB 정확도만의 원인이라고 단정하지 않으며 축 오류·국소 보정·공통 solver의 관측 자료 범위로 제한한다.','',
        '![공통 입력과 특징 제거](FIGURES/01_pipeline.png)','',
        '![동일 15경로 성공률](FIGURES/02_success.png)','',
        '![전체 범위 위치와 회전 누적 분포](FIGURES/03_full_pose_CDF.png)','',
        '![세 대비 효과와 회복 손상](FIGURES/04_paired_effects.png)','',
        '![실제 영상 ID의 회복 손상 및 코너 자세 상충](FIGURES/05_actual_cases.png)','',
        '[확인] 공개 사례는 실제 평가 행의 T/R 및 코너 오류 그래프다. 원본 RGB와 정답 원좌표는 공개하지 않는다. 사례 선택 규칙·모든 ID·그림 해시는 [FIGURE_RECEIPT.json](FIGURE_RECEIPT.json)에 있다.','',
        '| 경로 | 전체/추가 params | 전체 latency 중앙/P90 ms | peak allocated/reserved MiB | 출처 |','|---|---:|---:|---:|---|']
    for m,p in cost.items():lines.append(f"| {m} | {p['total_params']}/{p['additional_params']} | {n(p['latency_ms']['median'])}/{n(p['latency_ms']['P90'])} | {n(p['peak_memory_MiB'])}/{n(p['peak_reserved_MiB'])} | {p['source']} |")
    lines += ['', '[확인] 비용은 같은 18장 패널, batch1, 준비20회, 영상당5회, 경로별 독립 프로세스다. 새 RGB_ONLY 2경로·CVRANK 3경로만 측정하고 이전 18경로 결과 중 필요한 10경로를 재사용했다. native ndarray→검출·tile·보정·SubPix·PnP 전체를 동기화해 측정하고 파일 decode/모델 load는 latency에서 제외한다. serialized·active·추가 파라미터와 실제 상주 메모리를 구분한다([LATENCY.json](LATENCY.json)).','',
        '[확인] model 연산 FP32, AMP/TF32 off를 유지한다. 기존 p3/p4 FP16 capture 반올림은 reused feature 경로에 남으며 RGB_ONLY 후단은 이를 읽지 않는다. 이전 PoseFix+SubPix 독립 실행의 최대 약 **0.94px** accuracy 좌표 차이는 잔여 제한으로 유지한다. 새 경로의 실측 parity도 각 비용 영수증에 남기며 유리한 재실행을 고르지 않는다.','',
        f"[확인] 새 GPU 예약 wall은 {n(compute['GPU_device_reserved_seconds'],2)}초/2700초, 계측된 CPU 분석 wall은 {n(compute['CPU_analysis_reserved_seconds'],2)}초다. 계측하지 못한 준비 오류와 병행 CPU 분석까지 포함하기 위해 초기화보다 이른 시작부터 전체 경과 시간에 계측 CPU 시간을 더한 보수적 상한 {n(compute['CPU_analysis_conservative_upper_seconds'],2)}초/3600초도 기록했다. 최종 독립 검산 시간은 별도 영수증에 기록해 이 상한에 합산 확인한다. GPU 숫자는 실제 kernel occupancy가 아닌 imports/load/CPU 준비를 포함한 process wall이다. 첫 입력 복원 및 비용 패널 준비 실패를 숨기지 않고 별도 stage와 수정 전후 소스 해시를 보존했다([COMPUTE_RECEIPT.json](COMPUTE_RECEIPT.json), [구현 오류 복구](IMPLEMENTATION_REPAIRS.json)).",'',
        '[확인] 기존 dirty Git 변경·기존 실험·입력/checkpoint·원고/LaTeX/PDF를 보존했다([SOURCE_PRESERVATION.json](SOURCE_PRESERVATION.json)). 공개물은 새 namespace의 코드·작은 파생 결과뿐이다. 독립 계산·공개 필드·문서 링크 검산은 [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json), 실제 실행 기록과 별도 clean 재현 recipe는 [REPRODUCE.md](REPRODUCE.md)에 있다.','',
        '[확인] 영상과 초기점을 쓰는 후단 보정의 선례는 [PoseFix CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html)다. 여기서는 팔레트용 입력 절제와 PoseFix-style 비교군을 사용하며 원 논문 전체를 재현한 것으로 쓰지 않는다. Shi–Tomasi/Harris와 cornerSubPix의 API 의미는 [OpenCV 공식 문서](https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html)로 확인했고 실제 실험 설정은 기존 저장소 설정을 유지했다.']
    with (doc/'RESULTS_KO.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    with (doc/'PAPER_DECISIONS.json').open('x') as f:json.dump(dict(status='DECIDED_ON_EXPOSED_DEV',synthetic_selected_arm=selection['selected_arm'],
        recommended_simple_configuration=recommendation+'_SUBPIX',combined_increment_confirmed_both_single_inputs=combined,
        core_claim_removed=[] if combined else ['feature-and-RGB joint necessity','complementarity proven'],
        old_records_deleted=False,independent_test_claim=False),f,indent=2)
    print('KOREAN_REPORT_WRITTEN',recommendation,combined,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-dir',required=True);run(p.parse_args())
