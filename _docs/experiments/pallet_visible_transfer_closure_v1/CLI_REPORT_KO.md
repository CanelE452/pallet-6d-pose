# 최종 실행 보고

STATUS: COMPLETED — 진단·단일 짝 실험·해석·원고 마감 완료

HEAD_START: 9238735e2702bf817c6325811eb0d48ed2c0ade3
BRANCH: main
SCOPE: bounded visible-transfer closure

진단 commit/push: dc83b6341d4629eae75eb71a5f295bedcd91eb46 (origin/main 일치 확인).
최종 commit/HEAD_END와 push 후 remote SHA는 이 파일을 포함한 최종 Git 커밋 및 CLI 응답에서 확인한다. 파일 자체에 자기 커밋 SHA를 넣는 순환 참조는 만들지 않는다.

## DIAGNOSIS

- TRAIN_UNIQUE_IMAGES: 217; epoch당 실사512+합성512 슬롯, 유효 코너1,627개.
- TRAIN_TARGET_FOLLOWING: 기존 RAW의 corrected-target 잔차4.078px, 기존 REF3.070px. 보정은 일부 전달됐다. 추가320 update 후 REF2.994px로 제한적인 추가 감소.
- VISIBLE_PCK10_CANCELLATION: 기존 진입3/이탈3, Clean+2·Moderate−2·Severe0. 신규 진입2/이탈2로 동률 유지.
- REFERENCE_SENSITIVITY: 동일66점 legacy RAW45/REF44, verified43/43. 같은 subset은 참조변경 전에도 REF 우세가 없었다. 참조 차이만으로 전체128장과의 차이를 설명할 수 없다.
- PRIMARY_EXPLANATION: 부분 타깃 추종, 제한적인 추가 최적화 효과, 표본 구성 및 문턱에서의 상쇄가 함께 관찰됐다.
- COMPETING_EXPLANATION: 모순되거나 틀린 pseudo, 증강/합성 replay, 동결 표현, 학습 밖 전이 문제는 분리 확정되지 않았다.
- UNRESOLVED: unlabeled TRAIN 타깃의 실제 정확도, 전체 증강 tensor 이력, 독립 세션/물리6D 검증.

## INTERVENTION

- SELECTED: A_budget. B는 실행하지 않음.
- WHY_THIS_ONE: TRAIN 잔차와 마지막 학습 구간의 감소를 근거로 제한된 학습량 가설을 시험. 66점에서 올릴 점을 골라 설계하지 않음.
- STANDARD_SOURCE_AND_DEVIATION: 기존 학습 구현/목적함수 재사용. 첫5epoch cosine 및 E2ELoss 가중치 경로 보존 후 실제 종료 학습률1.859423525e−6 고정 연장. Soft Teacher/Unbiased Teacher v2 위치 신뢰도 원리는 검토했지만 keypoint calibration 증거 없이 도입하지 않음.
- NEW_FITS: 2. RAW_UPDATES / REF_UPDATES: 640 / 640.
- PAIR_INTEGRITY: PASS. 각 원래320 prefix의 loss/LR CSV 및 저장EMA model bit-exact; RGB·박스·기본support·초기값·학습조건 동일, 동결747 tensor exact.

## VISIBLE66

- OLD_RAW_PCK10 / OLD_REF_PCK10: 43/66 / 43/66.
- NEW_RAW_PCK10 / NEW_REF_PCK10: 44/66 / 44/66.
- PAIRED_GAINS / PAIRED_LOSSES: 신규RAW→신규REF 2/2; 기존REF→신규REF 1/0.
- PCK5/20: 신규RAW22/60, 신규REF26/63 (분모각66).
- 평균 오차: 신규RAW10.486px / 신규REF10.040px.
- 중앙값/P90: 신규RAW6.966/19.577px, 신규REF7.120/17.005px.

## FULL128

| 모델 | PCK10 | D9 AUC | matched P90 px | 검출/pose |
| --- | --- | --- | --- | --- |
| R0 | 484/985 (49.14%) | 0.33796 | 40.902 | 128/128 |
| OLD_RAW | 468/985 (47.51%) | 0.33472 | 41.958 | 128/128 |
| OLD_REF | 507/985 (51.47%) | 0.35902 | 42.085 | 128/128 |
| NEW_RAW | 469/985 (47.61%) | 0.33681 | 41.917 | 128/128 |
| NEW_REF | 504/985 (51.17%) | 0.36189 | 42.160 | 128/128 |

## DECISION / PAPER_UPDATE

- EXPERIMENT_STATUS: COMPLETED.
- VISIBLE_TRANSFER_RESULT: 혼합; 검수 PCK10의 corrected 우위는 확인 못함.
- CORRECTED_TARGET_ADDED_VALUE: 같은640 update에서 전체128장 PCK10 +3.553pp/AUC +0.02509. 기존320 비교를 일관되게 능가하지는 않음.
- EVIDENCE_STATUS: REUSED_DEV_ONLY.
- METHOD_DEVELOPMENT_STOPPED: TRUE.
- SUPPORTED_WORDING: 명시한 감독 예산과 반복DEV 범위에서 보정 타깃의 추가 가치가 일부 지표에 존재하며, 타깃 감독은 부분적으로 전달된다.
- UNSUPPORTED_WORDING: 가시점 병목 해결, 모든 난도/지표 개선, 독립 일반화 확인, 물리6D 정확도 검증.
- REPORT_PATH: `_docs/experiments/pallet_visible_transfer_closure_v1/REPORT_KO.md`.
- MANUSCRIPT_PATH: `_docs/paper/selftraining_submission_v1/manuscript.tex` 및 `manuscript.pdf`.
- BUILD_STATUS: PASS, 9페이지; unresolved reference/citation 또는 Overfull 없음. Underfull 조판 경고만 남음.
- USER_ACTION_REQUIRED: NO. 추가 좌표/촬영/레이블링 요청 없음.

왜 동률이었나: 개선점과 악화점이10px 문턱에서 상쇄됐으며, 작은 가시점 subset은 legacy 참조에서도 전체128장의 이득을 재현하지 않았다.

무엇을 확인했나: 보정 감독은 부분 전달되며, 한 번의 동등조건 학습량 연장으로 타깃 추종은 소폭 늘지만 corrected의 가시점 PCK10 추가 우위는 생기지 않는다.

무엇이 미확정인가: 남은 원인 하나의 확정, 실제 unlabeled 타깃의 정확도, 독립 세션/숨은점/물리6D 일반화다. 이를 해결할 때까지 새 방법을 반복하지 않고 기존 범위로 원고를 마무리했다.
