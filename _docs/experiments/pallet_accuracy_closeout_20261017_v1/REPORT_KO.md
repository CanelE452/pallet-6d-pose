# 동일192장 PoseFix+SubPix 비교와 E0 정보 계약 마무리

## 1. 실행 SHA·입력 계약·실행 범위

최신 원격 `origin/main`을 fetch하여 확인한 기준 SHA는 **`b843726ee2a78d7d29b79dafecdfc9d982b266fd`**다. 지시문 작성 기준 이후 관련 diff는 없었다. 지시문 SHA256은 `c99493d8b3d17024f95290d4579e53ecbf18b488d1ec87ef7a477e2624477cce`이며 [실행 계약](PLAN_AND_CONTRACT.md)에 범위와 중단 조건을 고정했다. 게시 커밋은 이 문서가 포함된 원격 커밋이며 최종 사용자 응답에서 실제 SHA를 확인한다.

**실행 완료:** A의 원성능·교차 집계 재현, B의 원95장 점/pose parity와 두 고정 PoseFix+SubPix의 PRIMARY192 확장·비교, C1의 238장 관측/공분산/owner 계약 감사, 단일 수정 가설의 반례 검사, 분리된 정확도 계산·독립 검산·코드/결과 보존이다. **C2의 새 opt-in 선택 경로·새192장 수정안 평가는 실행하지 않았다.** 현재 후보인 covariance 재적용을 실제 결함 수정으로 정당화할 증거가 없었기 때문이다. 결과를 높이기 위한 대체 가중치나 임계값을 만들지 않았다. 판정은 `NO_SUPPORTED_CHANGE`다.

PRIMARY192는 원 TEST49+추가 REMAINDER143이고 CAL46은 별도로 유지했다. 같은 영상·중심·물리 치수·K·Q와 고정 physical/C2 참조를 사용한다. 성공은 **하나의 최종 pose에서 위치≤5cm AND 회전≤5°**다. 직사각형의90° W/D 전환은 허용 대칭이 아니다. 불가용을 실패 분모에 남기며 연속 오차는 가용 자세만 집계한다. 이번 여섯 경로 모두192장 자세가 가용했다.

## 2. 원결과와 두 PoseFix+SubPix의 동일192장 결과

**E0 114/192, PF-anchor 112/192, YOLO-anchor 111/192**다. E0는 각각 **2장(+1.0417%p), 3장(+1.5625%p)** 많았다. 두 설정을 모두 보고하며 `TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER`를 유지한다. 숫자상 더 높은 비교군과의 격차가2장이라는 기술은 배치할 PF 설정을 사후 선발했다는 뜻이 아니다. E0의 작은 성공 수 우세와 더 나쁜 평균·꼬리 오차를 함께 보존한다.

| 경로 | 성공 | 위치 평균±표본SD(cm) | 위치 P90/P95/최대(cm) | 회전 평균±표본SD(°) | 회전 P90/P95/최대(°) | T>50/T>100/R>45 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 66/192 | 14.536±26.443 | 28.193/47.890/183.080 | 14.282±28.965 | 81.794/84.482/88.800 | 10/6/29 |
| N0 | 108/192 | 11.306±26.877 | 20.210/37.201/174.758 | 10.631±25.131 | 69.549/84.228/88.905 | 8/5/20 |
| E0 | 114/192 | 14.112±30.077 | 41.325/60.282/260.412 | 13.001±28.388 | 74.442/84.159/159.137 | 15/3/26 |
| POSEFIX_RAW | 80/192 | 13.836±27.578 | 23.465/55.654/185.783 | 11.640±26.525 | 80.309/84.237/88.945 | 12/5/23 |
| POSEFIX_SUBPIX_PF_ANCHOR | 112/192 | 9.848±20.592 | 20.907/46.417/173.997 | 10.539±25.128 | 61.246/84.978/89.649 | 10/3/20 |
| POSEFIX_SUBPIX_YOLO_ANCHOR | 111/192 | 9.377±19.375 | 16.948/36.818/173.997 | 10.594±25.247 | 69.538/84.932/89.649 | 9/2/20 |

표본SD는 ddof=1이고 이상치를 제외하지 않았다. 중앙값·분산·세션별 결과·모든 행과 실패·오류/fallback/unknown은 [전체 비교표](FAIR_192_COMPARISON.csv), [행별 오차](FAIR_ROW_METRICS.csv), [결과 원장](metrics/FAIR_RESULTS.json)에 있다. 245장 원고,110장,다른 seed의 수치를 이 표에 섞지 않았다.

원성능 R0=66/N0=108/E0=114/PF RAW=80은 저장 pose에서 직접 재계산했다. 952개 원 metric과 최대 차이는 위치2.84e-14cm/회전0°다. 원 COMMON95에서는 점 배열475/475와 최종 pose 전체 dictionary475/475가 정확히 일치했고 최대 절대 차이는0이었다. 그 뒤 별도 채점에서 PF RAW66/PF-anchor71/YOLO-anchor70/N074/E086 /95를 재현했다. 추가143 PF RAW도 기존 cache F와 최대차이0으로 일치했다.

PF RAW는 synthetic-only PRIOR1 seed1 last6000의 첫 uncapped 저장 출력이다. PF-anchor는 SubPix의 추가 이동만 PF 출력 기준 원영상 대각선1%로 제한하고, YOLO-anchor는 PF+SubPix 총 이동을 원 YOLO 기준1%로 제한한다. 기존 창5×5/40회/epsilon.001 및 같은 F의 W/D 선택→SQPnP→RefineLM을 그대로 사용했다. 중요한 입력 함정인 추가143의 `raw_points`=N3+SubPix는 YOLO anchor로 사용하지 않고 원 prediction-only YOLO cache를 연결했다. [정의·parity·실행비용](POSEFIX_METHOD_AND_PARITY_KO.md)을 참조한다.

| 분리 집합 | R0 | N0 | E0 | PF RAW | PF-anchor | YOLO-anchor |
|---|---:|---:|---:|---:|---:|---:|
| CAL46 | 31/46 | 34/46 | 37/46 | 32/46 | 32/46 | 32/46 |
| TEST49 | 30/49 | 40/49 | 49/49 | 34/49 | 39/49 | 38/49 |
| REMAINDER143 | 36/143 | 68/143 | 65/143 | 46/143 | 73/143 | 73/143 |
| COMMON95 | 61/95 | 74/95 | 86/95 | 66/95 | 71/95 | 70/95 |

숫자상 PF-anchor 대비 전체 +2장은 원49장에서 E0 49 대 PF39의 **+10장**, 추가143장에서 E0 65 대 PF73의 **−8장**이 합쳐진 결과다. 모집단 내부의 차이를 숨기고 전체 +2장만으로 일반화를 주장할 수 없다. [성공 수와 전체 tail ECDF 갤러리](gallery/index.html)는 여섯 경로를 함께 보여준다.

## 3. E0의 paired 회복·손상

| 기준→E0 | 공동 성공 | 실패→성공 | 성공→실패 | 공동 실패 | 순차이 | 기록 군집95% CI(%p) |
|---|---:|---:|---:|---:|---:|---:|
| N0→E0 | 89 | 25 | 19 | 59 | +6 | [-7.451, 23.333] |
| POSEFIX_RAW→E0 | 65 | 49 | 15 | 63 | +34 | [-2.518, 46.018] |
| POSEFIX_SUBPIX_PF_ANCHOR→E0 | 93 | 21 | 19 | 59 | +2 | [-9.449, 25.926] |
| POSEFIX_SUBPIX_YOLO_ANCHOR→E0 | 92 | 22 | 19 | 59 | +3 | [-9.020, 27.419] |

E0−N0의 +6장은25회복−19손상이고, PF-anchor 대비 +2장은21회복−19손상이다. PF-YOLO-anchor 대비 +3장은22회복−19손상이다. [전량 전이](PAIRED_TRANSITIONS.csv)에 이미지별 ID·양쪽 오차를 보존했다. 공동실패의 연속 tail도 결과 JSON에서 별도 집계했다.

기존 원녹화6개 `cluster_id`를 통째로 재표집한 paired frame-weighted percentile bootstrap(seed20260917,10,000회)을 결과 해독 전에 고정했다. 이름상 session8개를 단위로 한 민감도 결과도 별도 보존했다. 두 PF 대비 성공률 차이 구간은 모두0을 포함한다. **통계적 우월성이 입증되지 않았으며, 비유의가 동등성의 증거도 아니다.** 군집 수가 적고 반복 개발 자료이므로 확증적 검정으로 해석하지 않는다.

## 4. CAL의 point/partial/outline 공분산·owner·출처 감사

상태는 **`OBSERVATION_MODEL_MISMATCH`**이며 우연한 구현 오류가 확인된 `CONFIRMED_BUG`가 아니다. 물리 native vertex의 기하적 운반은 맞지만 영상 semantic owner의 정답 인증에는 **`INSUFFICIENT_EVIDENCE`**가 남는다. [정보 계약 감사](E0_INFORMATION_CONTRACT_AUDIT_KO.md), [단일 수정 제안과 반례](MECHANISM_AND_EDIT_PROPOSAL.md)에 코드 위치와 실제 저장 관측을 연결한다.

238장1,252개 보존 관측(CAL267,PRIMARY985)을 검사했다. 현재 complete likelihood는 CAL192점/PRIMARY644점이며 나머지75/341점은 partial로 철회된 원점 archive다. full과 partial의 동시 정밀2D point 사용, px/정규화 혼동, Q 제곱 적용, covariance 역행렬 부호 같은 우발적 오류는 확인되지 않았다. native 물리 꼭짓점 대응의 최대 수치 차이는1.11e-16m, whitening 항등오차는4.16e-17, 원 adapter 재현 차이는0이었다. 이것은 owner가 영상에서 물리적으로 옳다는 인증이 아니다.

원 CAL fit 입력267점 중 기하 호환44/46장과254점이 교정에 사용됐다. 원 입력의 인간 상태는 DIRECT_VISIBLE254/SELF_OCCLUDED13이다. fit의 기하 호환254점에는 DIRECT_VISIBLE242/SELF_OCCLUDED12가 섞여 있다. fit에서 제외된13점은 두 프레임의 물리/native 등록 불호환 때문에 제외됐으며 인간 SELF13점과 같은 집합이 아니다. 원 세션 수는 eval_cad18/eval_night08 3/eval_outside5/wood_18430920으로 불균형하다. 그룹별 실제 fit 수는 B:corrected145/B:fallback88/I:corrected6/I:fallback15다. I:corrected는2개 세션뿐이다. 공분산 단위는 px², 변환된 fit covariance와 reliability는 기존 CAL total-error model에서 전달되며 단순히 '불확실성 모듈이 없다'고 설명할 수 없다.

새 B_ACTUAL_BOUNDARY_RECOVERY로 관측 출처·조건부 covariance가 바뀐39/1,252점(CAL10점/9장,PRIMARY29점/22장)에서는 옛 fit covariance/reliability가 그대로 보존된다. PRIMARY29점 중27점은 full,2점은 partial archive다. 이는 원 frozen-old-noise 실험의 명시적 설계여서 전달 버그로 단정할 수 없다. 새 출처를 기존 B:corrected에 억지로 넣어 현재 adapter를 재적용하면 일부 관측을 더 강하게 믿게 된다. point/partial/outline에는 같은 RGB 근거가 재사용될 수 있어 독립 정보량으로 단순 합산하는 근거도 부족하다. 출처·세션·상태별 표본과 원/effective scale·reliability 분포를 전량 CSV로 보존했다.

## 5. 수정 가설의 반례와 미실행 결정

검사한 한 가지는 **후단에서 출처가 바뀐 관측에 기존 CAL covariance adapter를 다시 적용**하는 안이다. 현재 CAL에 새 B_ACTUAL_BOUNDARY_RECOVERY 전용 그룹은 없다. 한 실제 CAL 예에서 sigma10.968px→7.560px로 줄어 더 강한 점 likelihood가 된다. 그런데 큰 고정참조 잔차416.470px는 독립 물리 GT 오차가 아니다. RGB에서 해당 새 점은 보이는 앞아래 코너에 가깝고, 고정 k3→ref6/branch/phase 매핑과의 잔차가 크다. **잘못된 점을 발견했다는 결론도, 더 정밀하다고 믿어도 된다는 결론도 지지되지 않는다.** 원 N3/SubPix→ORACLE→fresh CAL의 점·pose·branch 매핑은 정확히 일치하여 RAW/N3 입력 혼합 오류는 아니었다. 단순한 C2 phase 뒤집기도 해당 영상 관측과 맞지 않아 임의로 바꿀 수 없다. 먼저 native W/D semantic owner를 판별할 독립 근거가 필요하다.

수정안을 떠올리게 한 `wood_night_01:030898`의 full_BI/외곽선 충돌도 직접 확인했다. 활성 full 다섯 점은 이미 실제 RGB 출처(B 네 점+I 한 점)이며 현재 CAL adapter를 다시 적용해도 **다섯 점의 covariance와 reliability가 모두 그대로**다. 따라서 이 재적용안은 원래 동기가 된 비용 충돌 자체를 바꾸지 못한다. 새 출처와 native owner의 불확실성을 보여주는 [CAL 실제 영상·매핑 증거](audit_gallery/CAL_OWNER_PHASE_WITNESS.png)는 큰 잔차를 잘못된 물리점의 정답 증거로 읽지 않도록 분리해 제시했다.

기존 CAL radial LOGO 시도에서도 평균 ref 잔차17.4896→17.5322px, 직접 annotation 잔차15.4759→15.5846px로 악화했다. 이 과거 시도를 새 개선안으로 다시 이름 붙이지 않았다. 새로운 신뢰도 숫자·global weight·특정 여섯 장 규칙·A/B guard 탐색이나 수동 경계 재작성을 실행하지 않았다.

따라서 **`NO_SUPPORTED_CHANGE`**로 종료했다. E0 source·선택·예측은 그대로다. 새 수정안의192 성공 수나 회복/손상은 **미실행(N/A)**이며 E0를 복사해 새 실험0회복/0손상으로 가장하지 않는다. 새로운 opt-in 경로가 다른 방법을 이겼다는 근거를 만들지 않았다. 현재 확보한 정확도 결과는 기존 E0가 숫자상114로 가장 높으나 최강 PF+SubPix112와의 차이는 작고 tail은 더 나쁘다는 것이다.

## 6. 비교 비대칭·검산·남은 문제

E0는 실사 CAL에서 상태별 교정을 사용했고 PF는 synthetic-only 사전학습이다. N0·E0·PF는 앞단과 후단 구성이 다른 전체 시스템 경로여서 upstream 보정기만의 인과 효과로 해석할 수 없다. 모든 실사는 이미 반복 개발에 사용됐다. 참조는 기하 재구성 자세로 실제 물리 corner owner와 카메라/축 확정에 한계가 남는다. 현재 결과로 독립 세션 일반화나 실사용 정확도를 주장하지 않는다.

예측기는 참조·오라클·성공 라벨 접근을 차단하고 점·pose를 먼저 봉인했으며 평가기는 별도 프로세스에서 고정참조를 읽었다. 독립 scalar C2/표준 라이브러리 통계 재집계는 새 metric 함수를 import하지 않고1,428행·30모집단/경로 요약·20 paired 대비·120 CI 배열·476 anchor 점 배열을 검사했다. metric/요약/CI 최대 차이는1.68e-11, cap 점 차이는0px였으며3.144초였다. [독립 영수증](INDEPENDENT_ARITHMETIC.json)과 종합 [VALIDATION.json](VALIDATION.json)에 실제 검산·분모·SHA·보존 증거가 있다. [재현 명령](REPRODUCE.md), 원 protocol 두 개와 [기술 수정 이력](EVALUATOR_TECHNICAL_ATTEMPTS.json)을 함께 보존했다. 기존143 저장 순서 차이 때문에 최초 evaluator가 numeric PF192 채점 전에 중단됐고 immutable ID join만 수정했다. 과학적 공식·숫자·분모·점·pose는 바꾸지 않았다.

실제 새 PF238 후처리는2.630초(CPU2.569초),475행95 기술 재현은1.424초, 원952metric 재집계는0.139초, 공통238×6 평가0.492초, 정보 계약 감사20.016초였다. NN forward/학습0이며 PF 후처리에는714회 F 호출이 있다. 이는 neural cache를 재사용한 비용으로 end-to-end FPS가 아니다. 각 프로세스 high-water RSS와 환경/오류는 seal에 보존한다. 운영 기본값·기존 N3/SubPix·E0·가중치·참조·논문·사용자 수정은 교체하지 않았다.

바로 다음 필요한 한 가지는 **기존 CAL과 분리된 원녹화에서 B 실제 회복점의 물리 corner owner를 독립 확인한 소규모 관측 묶음**이다. 한 점과 주변 선의 공동 출처·오차를 함께 확인해야 같은 RGB의 중복 근거와 새 출처의 uncertainty를 구별할 수 있다. 이번에는 새 데이터·학습·모델을 도입하지 않았다.

## 7. 최종 의사결정

**REJECT — 근거 없는 covariance 재적용 수정안은 채택하지 않는다. 기존 E0는114/192로 두 고정 PF+SubPix보다2/3장 많지만 독립 우월성은 미확인이고 tail은 더 나빠 추가 검증이 필요하다.**
