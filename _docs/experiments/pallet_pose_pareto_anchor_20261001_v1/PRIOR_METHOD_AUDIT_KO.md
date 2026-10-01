# R0 양축 비악화 target의 선행 감사

**조사한 저장소 구현에서 정확히 같은 학습 전례는 찾지 못했다.** 이는 제한된 코드·보고서 검색 결과이며 학술적 신규성이나 성공 예측이 아니다. 보존 손실·상대 utility·rollback 자체는 이미 여러 번 실행됐으므로 미시도 개념으로 주장하지 않는다. 근거의 정확한 SHA는 [PRIOR_METHOD_AUDIT.json](PRIOR_METHOD_AUDIT.json)에 있다.

찾은 대상은 고정 **R0 operational GEO**를 앵커로, source에서 `T≤T_R0 AND R≤R_R0`인 후보만 target 후보로 허용하고 그 안에서 기존 TRAIN `max(T/sT,R/sR)`를 최소화하는 규칙이다. 기존 물리 pose union/converged/pairwise는 모든 유효 후보를 대상으로 minmax target 또는 순위를 정하며 Pareto는 cost 동률 처리다.

| 가까운 선행 | 실제 차이와 완료된 음성 증거 |
|---|---|
| N2/Replay utility | 2D point gain SmoothL1와 수직 pair 양 끝점 양수 선택. clean PCK10 95.233→93.618%, GREEN81.351→78.267%. whole-pose 물리 T/R 적격조건이 아니다. |
| Corner confidence·geometry rollback | confidence·제거 안정성·반전·heatmap spread 및 구조 재검사 후 기존 점으로 복귀. GREEN81.351→79.736%, DEV59.459→58.919%. 학습된 T/R target이 아니다. |
| DHT local no-harm | 연속 점 보정의 `relu(new2Derror−base2Derror)` 보존 손실. 세 Hough seed 모두 median/P90 악화, track 종료. |
| Hough gain selector | 전체 P/Q 2D layout의 상대 gain 회귀와 aggregate calibration guard. 세 seed mean gain +0.016280/0/−0.019861px, 합성 gate 실패. |
| Stage4 router | W/D 선택 후 두 expert의 C2 ADDnorm 비교 BCE. routed0.09768815가 S0 0.09621724보다 나빴다. T/R 각각 비증가인 target과 다르다. |
| 1% displacement cap | 2D 이동량 상한이다. 640×480의8px cap으로 같은 대응의 >20→≤10px 복구를 보장할 수 없다. cap은 실패한 주 결과를 대체하지 않았다. |
| 현재 pose selector | source VAL의 수렴 CE4/45, pairwise16/45 조건 실패. 실사 learned routing 미승인. 수렴·작은 비교 정확도가 최종 목표를 보장하지 않았다. |

기존 상세 목적함수 감사는 `pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md`에 있다. 직접 치수 좌표헤드의 squared regret와 PoseFix source-clean preservation도 이미 실행된 보존 손실이며 현재 고정 후보 score target과 같지 않다.

## 최소 개입의 경계

새로 바꾸는 것은 **source target 적격조건 하나**다. feature94, 고정 후보·scale·tie, 정규화, CE+명시적 L2, 수렴 solver와 평가 gate를 유지해야 해석할 수 있다. 적격 safe mask를 CE 경쟁 후보나 runtime valid mask에 적용하면 위험 후보를 억제하는 학습 신호가 사라진다. 전체 원래 geometric-valid 후보는 계속 경쟁하고, GT mask는 source target 생성에만 사용한다.

R0_ONLY에도 anchored target을 적용하면 target이 달라질 수 있으므로 이전 R0_ONLY weight를 같은 목표의 exact reuse로 주장할 수 없다. 후보의 GT 비악화 성질은 설명용 oracle/target의 성질이며 학습된 모델의 안전성 보장이 아니다. source45조건과 기존 실사 안정성 판정이 여전히 필요하다.

## 데이터 범위 유지

TRAIN2598/VAL1024의 기존 선언 C2·proper-rigid 자격을 유지한다. 전체5,120 추론·feature 모집단 및 C1/improper 진단행을 보존하고, TRAIN 후보 없는1행도 실패/+∞·전체 분모로 유지한다. TRAIN feature 모양은 R0_ONLY2598×2×94, UNION2598×4×94다.

기존 split의 ID·이미지SHA·scenario·renderer group 교집합은0이다. refiner source1412고유 이미지와 **원래 selector TRAIN4096/VAL1024/TEST1024**의 겹침은105/26/26이며, 이를 eligible TRAIN2598의 새 겹침 수로 바꾸어 말하지 않는다. VAL은 반복 사용 개발 모집단이자 renderer group1개이며 refiner까지 미노출인 독립 검증이 아니다. split 재선정·이미지 추가·원본 label 수정은 하지 않는다.

입력은 이미지 한 장+팔레트 치수+기존 K다. 시간축·새 센서를 추가하지 않는다. 실제 source target 계산은 root의 실사 anchored 진단 확인과 SOURCE_PROTOCOL 봉인 뒤에만 수행한다. 이번 선행 감사는 새 수치·GT 읽기·fit·이미지 forward를 실행하지 않았다.
