# A — 정의 감사 완료, 본학습 의사결정 대기

## 실제 완료

첨부 core SHA 두 개 일치, 25/25 수학 검사 통과. 실제 source 60,000개 모두 3D 코너로 C1/C2 순열을 유도했다.
합성 train 55,980장: C1 19,268 / C2 36,712. val 4,020장: C1 1,013 / C2 3,007.
승인된 정사각형 C4 task train 696 / DEV 155장은 별도 cohort다. 원영상 픽셀·parent 중복 0, interleaved session 개발 평가라는 한계는 유지한다.

기존 square F0/F1/F2/clean stage1/stage2 각각 155장, 총 **775회 실제 추론**을 다시 수행했다.
기존 prepared label 기준 유효점 mask가 C4에 불변하지 않은 DEV 프레임은 1장이지만, 다섯 모델의 최종 최소 오차가 mask 버그 수정으로 바뀐 프레임은 각각 0장이었다.
이는 버그가 없다는 뜻이 아니라 이 실제 출력의 선택된 최솟값에 영향이 없었다는 뜻이다.
좌표·mask 공동 순열이 달라질 때 실패하는 첨부 fixture도 유지한다.

기존 R0/P1–3/D1–3/L1–3의 DEV319 저장 출력도 새 corner8/full-denominator 지표로 재채점했다.
square155와 혼합하지 않았고 기존 논문 표·모델은 수정하지 않았다. 재채점 차이는 모델 개선이 아니다.

## 원본 annotation 대비 추가 정정

851장 모두 원본 영상의 reflect100 padding과 prepared 영상 픽셀이 정확히 일치했다. 그러나 기존 prepared label **2개**는 원래 미표시 점을 padding된 sentinel 위치의 보이는 점으로 잘못 저장했다.

- train `capture_20260902_manual_gt__010583`: zero-based 6, 7, 8번 미표시.
- val `capture_20260902_kimjihoon_manual_gt__004969`: zero-based 4, 7, 8번 미표시.

원본 annotation·영상·기존 label은 수정하지 않았다. `keypoint_annotations`와 기존 준비 코드의 순수 변환 함수를 사용해 별도 851장 target view를 만들었다. 위 2장만 전체 tuple 및 그에 따른 bbox가 정정된다. 기존 데이터 전체를 PASS로 바꾸지 않고 `PROVENANCE_QA.json`의 불일치 기록을 유지했다.

저장된 다섯 모델의 동일 예측 775개를 공통 정정 target으로 다시 채점한 결과는 `A0_ANNOTATION_CORRECTED.json`과 `annotation_corrected_per_frame.csv`에 별도 보존한다. 기존 `A0_SQUARE.json` 및 v4/v6별 역사적 결과는 그대로 둔다. 추가 추론·학습은 0회이며, 이 변화도 모델 개선이 아니다. 향후 두 학습군은 동일한 새 target view를 사용해야 한다(`ANNOTATION_TARGET_AMENDMENT.json`).

## 학습을 시작하지 않은 이유

실제 설치된 `PoseLoss26`은 RLE 손실을 **전체 positive/annotated point에 대해 평균한 뒤 0으로 clamp**한다.
이는 물체별로 분해 가능한 손실이 아니다. 실제 설치 RLE 함수의 fixture에서 배치 공통 clamp 값은
`0.6137056351`, 물체별 clamp 뒤 동일 reduction을 적용하면 `2.3068528175`로 달랐다.

따라서 지시문의 다음 두 조건은 일반적으로 동시에 성립하지 않는다.

1. 물체마다 독립적으로 complete keypoint-cost 최소 branch를 선택한다.
2. 원래 stock RLE reduction/clamp를 바꾸지 않는다.

위치비용만으로 branch를 고르거나 RLE를 제외하면 요청한 실험을 실행한 것이 아니다.
원래 손실을 보존하면서 두 head가 공유하는 물체별 대칭 **조합을 공동 선택**하는 변경을 허용할지 사용자에게 질문했고, 답변 전 본학습은 보류했다.

## 배선 검증과 미완료 범위

합성 실제 16장 + square 실제 16장에서 R0 전체 검출기 forward를 실행했다. Identity adapter의 손실과 실제 head-output/flow-parameter gradient는 stock과 **bit-exact**였다.
두 head `.8/.2` 경로가 존재함을 확인했고, GT 행 순서를 뒤섞어도 안정 정렬 adapter가 원래 assignment/목표를 복구했다.
이는 identity 검사이며 EQUIV 학습 성공 검사로 대체하지 않는다.

- 본학습 fits: **0/12**, optimizer updates: **0/24,000**.
- Smoke optimizer updates: **0/12**. Identity 검사는 forward/gradient 검사일 뿐 optimizer를 만들지 않았다.
- EQUIV adapter, 공유 branch 선택 검사, main training, 새 모델 평가·통계: **미실행**.
- `definition_integrity`: 수학/geometry/mask/분모 및 identity 배선 PASS. EQUIV 학습 배선은 PENDING.
- `geometry_gain`: **NOT_EVALUATED**. 대칭 지도 효과가 없다는 판정이 아니다.

다음 판단: stock RLE를 보존한 공동 branch 선택을 승인받은 뒤, 동일 seed·데이터·2,000-step 예산을 변경하지 않고 A를 실행한다.
