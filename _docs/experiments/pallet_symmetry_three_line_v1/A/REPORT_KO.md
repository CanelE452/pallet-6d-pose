# A 완료 — 대칭 정의와 성능 효과의 분리

12 fits × 2,000 actual updates = **24,000 main updates** 완료. Smoke는 별도 12 updates이며 R0에서 본학습을 다시 초기화했다.
초기 R0 및 각 cohort의 INDEXED/EQUIV seed1/2/3을 마지막 raw weights로만 평가했다. 새 A 모델에 기존 P 또는 DHT를 붙이지 않았다.

## 목적함수와 무결성

사용자가 승인한 **공동 대칭 조합 최소화**를 사용했다. 물체별 독립 min의 원래 제안과 동일하다고 주장하지 않는다. 두 head의 위치·visibility·batch-global RLE clamp와 stock reduction을 보존하고, GT 객체당 하나의 전체 tuple 순열을 두 head가 공유한다.
전체 손실 최적 조합은 선형 하한 인증 또는 exact one-hot MILP로 선택하고, 선택된 target의 실제 loss/gradient는 stock 코드로 계산했다. 원본 라이브러리·학습 source 파일은 바꾸지 않았다.
CPU 회귀검사 43/43 PASS. 실제 GPU identity 및 C4/C2·누락점·GT 행 정렬·complete-loss/gradient 검사를 통과했다. `definition_integrity=PASS`는 아래 성능 개선의 증명이 아니다.
R0 초기화·샘플 순서·전체 trainable parameter set·AdamW 1e-4·batch16·FP32·clip10·BN 통계 고정 조건을 검증했다. AMP/EMA/TF32/증강은 없고, 두 head 가중치는 양 군 0.8/0.2로 일정하다.

## 2D 결과

R0는 단일 값, INDEXED/EQUIV는 seed별 지표의 3-seed 평균이다. 아래 corner8/full-denominator 지표는 기존 논문의 matched9 표를 대체하지 않는다.

| 모집단 | 모델 | E_sym ↓ | E_fixed ↓ | sym pooled median px | sym pooled P90 px | 검출 매칭률 |
|---|---|---:|---:|---:|---:|---:|
| RECT 합성 val | R0 | 0.00898044 | 0.00898339 | 1.9833 | 7.1763 | 0.9965 |
| RECT 합성 val | INDEXED | 0.01243308 | 0.01244258 | 2.6330 | 9.2042 | 0.9944 |
| RECT 합성 val | EQUIV | 0.01177769 | 0.01178544 | 2.6330 | 9.4994 | 0.9954 |
| SQUARE 실사 DEV | R0 | 0.01219286 | 0.02160639 | 3.9155 | 8.9109 | 0.9935 |
| SQUARE 실사 DEV | INDEXED | 0.01562774 | 0.02235683 | 1.7417 | 4.3359 | 0.9871 |
| SQUARE 실사 DEV | EQUIV | 0.00475481 | 0.01580568 | 1.6836 | 3.8651 | 0.9978 |
| RECT 실사 DEV | R0 | 0.04952304 | 0.04965589 | 6.9663 | 61.6855 | 0.9749 |
| RECT 실사 DEV | INDEXED | 0.07501036 | 0.07509210 | 7.8485 | 110.2498 | 0.9498 |
| RECT 실사 DEV | EQUIV | 0.09095305 | 0.09101877 | 8.0479 | 351.0285 | 0.9321 |

## 주 비교와 판단

- **RECT 합성 val: UNRESOLVED**. EQUIV−INDEXED Δ=-0.00065538552, 95% CI [-0.00167098668, 0.000320451983], 개선 seed 2/3. 예측 4020장, 주지표 4020장, bootstrap frame 단위 4020개.
  corner 무주석 비평가 프레임은 0장이다(해당 ID 목록 보존). 검출 실패는 제거하지 않고 raw diagonal penalty를 유지했다.
- **SQUARE 실사 DEV: UNRESOLVED**. EQUIV−INDEXED Δ=-0.0108729309, 95% CI [-0.0221194744, 3.50065565e-05], 개선 seed 3/3. 예측 155장, 주지표 155장, bootstrap session 단위 28개.
  corner 무주석 비평가 프레임은 0장이다(해당 ID 목록 보존). 검출 실패는 제거하지 않고 raw diagonal penalty를 유지했다.
- **RECT 실사 DEV: UNRESOLVED**. EQUIV−INDEXED Δ=0.015942689, 95% CI [-0.00349904959, 0.0388641343], 개선 seed 1/3. 예측 319장, 주지표 319장, bootstrap session 단위 13개.
  corner 무주석 비평가 프레임은 0장이다(해당 ID 목록 보존). 검출 실패는 제거하지 않고 raw diagonal penalty를 유지했다.

CI가 0을 포함하는 결과는 동등성/비열등성 증명이 아니다. 합성 frame bootstrap은 실제 시나리오 의존성을 과소평가할 수 있다. 실사는 재사용 DEV이며 square는 같은 세션의 interleave split이다. 세 모집단을 합친 단일 효과나 독립 confirmatory 결과를 주장하지 않는다.
C1/C2/C4별, fixed/sym, PCK5/10/20, true-collapse, center 오차와 branch 분포는 `results_and_intervals.json`에 별도 보존했다. R0 대비 비교는 부 비교이며 primary는 EQUIV−INDEXED다.

## 검출 변화와 점 위치 변화의 분해

주지표를 바꾸지 않고, 전체 분모에서 두 모델 모두 매칭/한쪽만 매칭/둘 다 실패의 기여를 분리했다. 두 모델 모두 매칭된 부분집합은 사후 설명용이며 primary를 대체하지 않는다.
- RECT 합성 val: common-matched frame 평균 오차 차이의 seed 평균 0.30546px, EQUIV만 매칭 [10, 3, 14]장 / INDEXED만 매칭 [7, 4, 4]장. 기여 합산 검산은 `MATCH_AND_GEOMETRY_DECOMPOSITION.json`에 있다.
- SQUARE 실사 DEV: common-matched frame 평균 오차 차이의 seed 평균 -0.11451px, EQUIV만 매칭 [2, 2, 1]장 / INDEXED만 매칭 [0, 0, 0]장. 기여 합산 검산은 `MATCH_AND_GEOMETRY_DECOMPOSITION.json`에 있다.
- RECT 실사 DEV: common-matched frame 평균 오차 차이의 seed 평균 -0.92432px, EQUIV만 매칭 [2, 4, 5]장 / INDEXED만 매칭 [20, 7, 1]장. 기여 합산 검산은 `MATCH_AND_GEOMETRY_DECOMPOSITION.json`에 있다.

## 손해와 6D 보조 결과

| 모집단 | INDEXED 중심 이동 median cm | EQUIV 중심 이동 median cm | INDEXED 10cm·10deg 성공률 | EQUIV 10cm·10deg 성공률 |
|---|---:|---:|---:|---:|
| RECT 합성 val | 3.7194 | 3.8310 | 0.6617 | 0.6511 |
| RECT 실사 DEV | 7.9387 | 8.3452 | 0.5580 | 0.5214 |
| SQUARE 실사 DEV | 2.2219 | 2.0828 | 0.9376 | 0.9419 |

- RECT 합성 val: frame 평균 오차 악화 [1958, 2056, 2003]장, INDEXED <5px → EQUIV >10px 점 [434, 299, 413]개(seed1/2/3).
- SQUARE 실사 DEV: frame 평균 오차 악화 [69, 73, 71]장, INDEXED <5px → EQUIV >10px 점 [0, 0, 0]개(seed1/2/3).
- RECT 실사 DEV: frame 평균 오차 악화 [179, 149, 139]장, INDEXED <5px → EQUIV >10px 점 [55, 56, 49]개(seed1/2/3).

6D는 RECT에서 동일 prediction-only 축 선택 + SQPnP/RefineLM이다. SQUARE는 x=z인 동일 W/D 기하 가설을 하나의 대표로 합친 후 같은 solver를 쓴다. 그렇지 않으면 기존 직사각형 selector가 완전 동률을 모호성 실패로 처리하기 때문이다. 이 처리와 C4 회전 fixture는 A pose 성능 열람 전에 고정했다. 예측을 GT bbox match로 gate하지 않는다. median/P90은 유효 pose 쌍에 조건부이고 coverage·성공률은 전체 reference 분모로 보고한다. 물리 축의 전역 C1/C2/C4만 허용하며 unrestricted nearest-neighbour ADD-S로 바꾸지 않았다.
실사 DEV GT와 square reference는 geometry-reconstructed이며 독립 계측이 아니다. Square는 정정된 annotated corners와 등록된 1.1×0.15×1.1m cuboid로 별도 기준 pose를 구했다. 원 annotation의 저장 pose를 번호·원점 확인 없이 재사용하지 않았다. Source C1에서 camera-facing label만으로 물리적 앞면을 복구할 수 없다는 한계도 유지한다.
합성 전체 val의 G38__G__f11070 한 장은 CF index basis의 determinant가 −1이었다. 기존 matched-subset용 assertion이 A pose 추론 전에 이를 검출했다. Source body GT는 저장된 proper renderer R·physical extents로 직접 표현해 같은 박스 부피를 유지했다. 프레임을 빼거나 reflection을 허용 그룹에 넣지 않았고 학습/예측도 재실행하지 않았다. `SOURCE_POSE_FRAME_AUDIT.json`에 정정 경위를 보존한다.
A에서 음성 2,689장 검출을 새로 재검증한 것은 아니므로 false-positive 개선/안전성을 주장하지 않는다.

합성 회전 보조 지표를 열람한 뒤, renderer의 +Y up과 PnP의 +Y down 사이 고정 proper basis 변환 누락을 발견했다. 모든 합성 예측에 동일하게 R_source = R_previous × diag(1,−1,−1)을 적용해 회전만 재계산했다. 사전 등록 변경으로 가장하지 않으며 GT별 위상 선택이나 학습·추론 재실행은 없다. 정정 전 A 수치와 B 원본 전체는 보존했다. B의 합성 회전 보조 지표만 `../B/POSE_SOURCE_ROTATION_CORRECTION.json`으로 정정한다. 중심·body IoU·2D·DEV 결과는 불변이다. 자세한 증거는 `SOURCE_ORIENTATION_CORRECTION.json`에 있다.

## 데이터·기록·다음 판단

합성 train55,980과 square train696은 분리했다. Square 기존 prepared label 2행의 미표시 점 오류만 별도 target view로 정정했고 두 군이 같은 view를 사용했다. 원본 label·영상·기존 A0·B·논문 결과는 보존했다.
실제 새 평가 forward는 **31,458 images**(RECT 합성4020+DEV319, SQUARE155 각각7모델)다. 학습 forward384,000, 고정 train probe768, smoke192, 이전 A0/배선 감사는 별도 기록이다.
수학적으로 올바른 task-equivalent 정의와 실측 gain을 분리해 판단한다. 이번 고정 데이터·초기화·2,000-step 조건 밖으로 일반화하지 않으며, 결과를 이유로 새 seed/계수/학습량 탐색을 자동 추가하지 않는다.
