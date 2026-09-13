# Task-Space Tail-Risk Active Adaptation v1 — 완료 결과

과학적 판정: **`TASK_RISK_AL_NO_SIGNAL`**.

위험도가 어려운 포즈를 예측하는 Stage0는 통과했지만, 그 위험도로 선택한
30장으로 학습한 학생은 위치 tail 및 검출 안전 기준을 통과하지 못했다.
좋은 error predictor와 좋은 acquisition/training 결과는 동일하지 않았다.
추가 score/perturbation/budget/seed 탐색 없이 이 track을 종료한다.

## 기존 결과 계승과 실행 범위

기존 `RETROSPECTIVE_AL_NO_SIGNAL`과 과거 파일은 보존했다. 기존 P90 관찰을
새 성공으로 재판정하지 않았다. 실제 원본 JSON에서 재검증한 숫자는
`PREVIOUS_RESULT_BINDING.json`에 있다. 기존 common-kp median/P90은
random4.3502/25.9490, diversity4.2481/24.1507, old geometry4.3452/20.6280px다.

고정 pool174, 별도 평가145, 음성2689를 그대로 사용했다. Stage0는 정확히
174×8=1392 R0 image forwards, 학습은 proposed seeds1/2/3와 full174 seed1의
4fits×300=1200updates다. 기존 대조군은 재학습하지 않았다. 각 fit은
synthetic7200/real2400 nominal batch slots를 사용했다. Mosaic의 추가 source
참조를 nominal slots와 혼동하지 않는다.

모든 새 fit의 초기 R0 state SHA가 같았고, 각300step의 실제 합성
image/box/keypoint tensor SHA가 기존 같은 seed와 일치했다. Stock loss,
optimizer/warmup/normal BN/증강/last-checkpoint 규칙을 바꾸지 않았다.
모든 학습이 끝난 뒤에만 평가145 GT와 mixed319 GT container를 채점에 열었다.

## Stage0 — 위험도 자체는 유효했는가?

GT를 읽지 않는 session/raw `cam_K.txt`와 이미 존재한 category 고정 규격을
사용했다. GT에서 frame별 치수나 실제 축을 선택하지 않았다. 기존 wood45의
scaled intrinsics profile을 더 정확한 calibration으로 승격하지 않았다.

수식·perturbation·metric·gate·핵심 코드 SHA를 pool GT 오차를 읽기 전에
고정했다. Phase1 pool-only GT 진단과 Phase2 추론은 별도 프로세스다.
추론에 GT read denial을 적용했다. P0의 box/keypoint/score 및 dense grid
index는 기존 stock R0 cache와 bit-exact였다.

- 원본 유효 포즈174/174. 8개 중6개 이상 유효174/174.
- R_task min/Q25/median/Q75/P90/max:
  0.0374 / 0.4009 / 0.6063 / 0.8247 / 0.9322 / 0.9971.
- 각 target hard20은35장. Session 검사는 finite8장 이상인 세션에서 수행.

| Target / score | Spearman | hard20 AUROC | precision@30 | recall@30 | 양의 세션 상관 |
|---|---:|---:|---:|---:|---:|
| 위치 / old brightness | 0.3065 | 0.6460 | 0.4000 | 0.3429 | — |
| 위치 / task risk | 0.6490 | 0.8295 | 0.5333 | 0.4571 | 8/9 |
| yaw / old brightness | 0.3914 | 0.6479 | 0.4333 | 0.3714 | — |
| yaw / task risk | 0.5361 | 0.7667 | 0.5333 | 0.4571 | 9/9 |
| kp / old brightness | 0.4909 | 0.6623 | 0.4000 | 0.3429 | — |
| kp / task risk | 0.2502 | 0.7085 | 0.3667 | 0.3143 | 8/9 |

G0–G4는 위치/yaw 모두 충족했다. Task minus brightness AUROC의
10,000-draw session-cluster95% 구간은 위치[0.0372,0.3305],
yaw[0.0089,0.2258]이다. Paired frame bootstrap도 별도 저장했다.

무료 inverse-confidence baseline도 위치/yaw AUROC0.8080/0.7566으로 강했고,
precision@30은 둘 다0.5667로 task risk보다 높았다. 모든 uncertainty
baseline에 대한 우월성을 주장하지 않는다. Point sigma용 새 모델은 학습하지 않았다.

## 선택 메커니즘

Old diversity30와 old geometry30의 겹침은13장, 각 고유 집합은17장이다.
Proposed30은 diversity와18장, old geometry와18장 겹친다. 정확한 선택
순서와 시간 간격을 감사했고, 재랭크하지 않은 `1+R_task`를 사용했다.

| 선택30 | kp hard35 포함 | 위치 hard35 포함 | yaw hard35 포함 |
|---|---:|---:|---:|
| old diversity | 6 | 5 | 6 |
| old geometry | 10 | 8 | 11 |
| proposed | 8 | 9 | 9 |

Proposed는9개 pool 세션에서 plastic19/wood11을 선택했다. 고정 metadata의
주야 구분은 DAY12/NIGHT9/UNKNOWN9이며 UNKNOWN을 임의 보완하지 않았다.
평균 candidate/axis switch rate는0.1083/0.0667이다.

Geometry-only17의 위치/yaw hard 수는7/9, diversity-only17은4/4였다.
고유 집합의 bootstrap과 분포는 `OLD_SELECTION_MECHANISM_REPORT.md`,
새 선택과 학생 tail의 연결은 `SELECTION_MECHANISM_LINK.json`에 있다.
Hard-frame enrichment가 학생 개선을 보장하거나 인과적 매개를 입증하지는 않는다.
단순 risk 상위30과 diversity를 포함한 실제 학습 선택30도 구분한다.

## 학생 비교 — 세 seed 평균

Primary는 **전체145장 중 가장 나쁜15장의 위치 오차 평균(CVaR90)**이다.
포즈가 산출된 것을 정확한 포즈라는 뜻으로 해석하지 않는다.
2D는 seed별 random/diversity/old geometry/proposed 공통143/142/142장에서
supervised corners를 pooled 계산했다. 이 집합은 과거 four-arm 집합과 달라
아래 old kp 값이 과거 정본과 다르다. 과거 결과를 수정한 것이 아니다.

| 방법 | 위치 median / CVaR90 cm | yaw median / CVaR90 ° | 공통 kp median / P90 px | AP50–95 | Det |
|---|---:|---:|---:|---:|---:|
| random | 10.254 / 110.588 | 1.332 / 52.399 | 4.375 / 28.729 | 0.782465 | 0.993103 |
| diversity | 8.814 / 109.527 | 1.394 / 41.236 | 4.299 / 25.755 | 0.769148 | 0.997701 |
| old geometry | 9.742 / 112.009 | 1.292 / 49.131 | 4.408 / 22.033 | 0.785874 | 0.993103 |
| proposed | 10.219 / 284.419 | 1.444 / 43.813 | 4.272 / 23.517 | 0.781537 | 0.986207 |

모든17개 모델의 pose coverage는145/145다. AUROC/FPR95, IoU3D, ADDsym AUC,
frame-mean-kp CVaR90와 개별 집계는 `PER_SEED_RESULTS.json`에 보존했다.

| Seed / 방법 | 위치 median / CVaR90 cm | yaw median / CVaR90 ° | 공통 kp median / P90 px | AP50–95 | Det |
|---|---:|---:|---:|---:|---:|
| 1 random | 9.965 / 114.350 | 1.382 / 55.304 | 4.472 / 29.484 | 0.789597 | 0.993103 |
| 2 random | 10.691 / 98.085 | 1.290 / 46.725 | 4.363 / 25.312 | 0.796765 | 0.993103 |
| 3 random | 10.105 / 119.330 | 1.322 / 55.167 | 4.291 / 31.391 | 0.761033 | 0.993103 |
| 1 diversity | 9.046 / 114.912 | 1.590 / 46.264 | 4.480 / 27.962 | 0.766142 | 1.000000 |
| 2 diversity | 8.509 / 103.538 | 1.277 / 36.348 | 4.170 / 24.320 | 0.793828 | 1.000000 |
| 3 diversity | 8.887 / 110.132 | 1.315 / 41.096 | 4.249 / 24.984 | 0.747475 | 0.993103 |
| 1 old geometry | 8.712 / 123.350 | 1.293 / 50.471 | 4.440 / 22.737 | 0.781286 | 0.993103 |
| 2 old geometry | 10.389 / 107.989 | 1.354 / 46.526 | 4.464 / 22.589 | 0.797373 | 0.993103 |
| 3 old geometry | 10.126 / 104.689 | 1.228 / 50.396 | 4.319 / 20.774 | 0.778964 | 0.993103 |
| 1 proposed | 9.879 / 293.077 | 1.446 / 42.877 | 4.187 / 20.272 | 0.780993 | 0.986207 |
| 2 proposed | 10.543 / 285.672 | 1.466 / 42.869 | 4.284 / 26.546 | 0.781538 | 0.986207 |
| 3 proposed | 10.234 / 274.509 | 1.421 / 45.693 | 4.344 / 23.734 | 0.782080 | 0.986207 |

## 판정과 tail 해석

- A/B FAIL: 위치 CVaR90이 diversity/old geometry보다 나쁘며 개선 seed는
  각각0/3. 차이는+174.892/+172.410cm다.
- C PASS: yaw CVaR90은 두 대조군 중 더 나쁜49.131°보다 낮다.
  Diversity41.236°보다도 낮다는 주장은 하지 않는다.
- D PASS: 공통 kp P90이 random와 diversity보다 낮다.
- E FAIL: Proposed 검출은 각143/145, 최고 대조군 평균144.667/145로
  평균1.667프레임 손실이다. 허용1프레임을 넘었다. Pose coverage는 유지됐다.
- F PASS: AP나 kp median/P90 개선으로 primary 실패를 덮지 않았다.

같은 평가 프레임 `eval_pallet09:1778653804674198784`에서 세 seed 모두
26.61–27.82m 위치 오차가 났다. 같은 프레임의 diversity 오차는0.25–1.14m,
old geometry는0.94–1.11m였다. 유한한 PnP 해가 나와도 큰 오차를 제거하지
않고 primary에 포함했다. 이 프레임을 제외하거나 새 PnP filter를 도입하지 않았다.

평가 세션은4개뿐이다. Paired session bootstrap10,000회의 seed-mean
CVaR 차이95% 구간은 diversity 대비[-1.833,576.872]cm,
old geometry 대비[-11.386,582.893]cm로 넓고0을 포함한다.
모집단의 통계적 열등성 확정이 아니라, 사전 고정 개발 gate의 미충족이다.
Seed를 독립 세션으로 세지 않았다.

## Full174 fixed-compute reference

Seed1 한 번: 위치 median/CVaR90=8.957/103.362cm,
yaw median/CVaR90=1.597/48.727°, 개별 matched kp median/P90=4.011/22.442px,
AP50–95=0.784065, Det=144/145, pose coverage=145/145.

같은300updates에서174 labels를 사용한 descriptive reference이지 upper bound가
아니다. Nominal real slots/label은174장 모델13.79, 선택30장 모델80으로 다르다.
이 결과로 추가 학습이나 프로토콜 수정은 하지 않았다.
R0 참고 위치/yaw CVaR90은116.972cm/55.283°다.

## 감사, 환경, Git

35개 contract 항목과14개 synthetic regression이 통과했다. 원본508개 binding,
319개 원본 라벨·이미지, 기존 체크포인트/결과 및 paper/final을 보존했다.
정본 pose aggregate와17개 모델의 per-frame 재계산이 일치했다. Phase1
pool-only GT도 모든 학습 후 정본과174장 bit-exact임을 확인했다.
환경 경고와 read-guard의 bytes 경로 한계는 `RUNTIME_NOTES.md`에 명시했다.

실제 RTX3080을 사용했고 학습 중 기록한 최고 온도는67°C다. 시스템/드라이버
설정 변경, 재부팅, 외부 프로세스 변경은 없었다. 이전 GPU-busy 중단은
`RESOURCE_PAUSE_REPORT_KO.md`와 불변 RESOURCE_BLOCKED JSON으로 보존했다.

시작 SHA: `d653dce26c43db3fd60c387ed1936bea5751aa24`.
Main에서만 작업하고 이번 scripts/docs만 commit 대상으로 삼았다.
대형 cache/checkpoint는 ignore 정책을 따른다. Commit 후 실제 local/origin SHA와
push 확인은 최종 CLI 답변 및 raw `GIT_PUSH_RECEIPT.json`을 기준으로 한다.

## 주장 범위와 종료

**Current development data do not support a robust task-risk active acquisition advantage.**

Retrospective development 결과다. 독립 일반화·SOTA·novelty를 주장하지 않는다.
하나의 acquisition 집합에 대한 세 student seeds이며 selection variance를 측정하지 않았다.
Human target labels30개를 사용하는 방법이지 zero-target-label 방법이 아니다.
사용자가 새 질문을 열기 전에는 다른 ML/Hough/DHT/self-training 탐색을 재개하지 않는다.
