# 상태별 결과와 논문 주장 범위

각 CAL46/TEST49의 prediction hash·protocol hash·observation hash와 scoring seal을 먼저 확인한 뒤, 이미 생성된 좌표 결과와 해당 격리 정답만 사용했다. 새로운 예측·학습·파라미터 선택·TEST 기반 튜닝은 없다. 전체 자세 평가는 46/49장을 유지하며, 고정된 native/body model compatibility 검사 후 좌표 진단은 CAL44/46, TEST42/49장이다. 제외된 frame ID와 모든 수치는 `PER_STATE_FINDINGS_V1.json`에 있다. 원 예측과의 비교는 정확히 같은 유효 corner, 초기 rigid/이전 N3 진행 보정과의 비교는 같은 frame/native slot을 사용했다.

## 어디서 좋아졌고 어디서 실패했는가

아래는 TEST의 같은 좌표들에 대한 평균 pixel error다. `DIRECT_OUTLINE`/`DIRECT_MODEL_INTERIOR`는 초기 기하학 외곽 여부로 나눈 DIRECT_VISIBLE이며, 실제 배경 접촉을 인증한 분류는 아니다.

| 상류 | 상태·점 수 | 원 예측 | 초기 rigid | 새 selected rigid | 이전 N3 진행 rigid |
|---|---|---:|---:|---:|---:|
| N3 | self-occluded 65점 | 8.634 | 5.071 | 5.020 | 3.987 |
| N3 | externally occluded 8점 | 12.388 | 9.551 | 10.259 | 9.272 |
| N3 | visible model interior 19점 | 6.798 | 5.851 | 6.843 | 7.897 |
| N3 | visible outline 231점 | 6.466 | 6.118 | 6.136 | 5.108 |
| PoseFix+Sub | self-occluded 65점 | 8.693 | 4.793 | 4.867 | — |
| PoseFix+Sub | externally occluded 8점 | 12.667 | 10.413 | 11.211 | — |
| PoseFix+Sub | visible model interior 20점 | 4.710 | 4.564 | 5.504 | — |
| PoseFix+Sub | visible outline 230점 | 6.118 | 5.827 | 5.928 | — |

원 예측 대비 self-occluded 좌표 개선은 N3 −3.614px, PoseFix+Sub −3.826px다. 하지만 초기 rigid completion 대비 변화는 각각 −0.051px, +0.074px이며 세션 bootstrap 구간은 0을 포함한다. 따라서 숨은 좌표가 원 예측보다 좋아졌다는 사실을 새 refinement만의 기여로 설명하면 안 된다. 초기 rigid projection이 이미 얻은 이익이 대부분이다.

같은 N3 상류에서 새 selected의 self error는 이전 진행 보정보다 +1.033px 높았다. 3개 촬영 세션을 묶은 paired bootstrap 구간은 [+0.277, +1.593]px였다. visible outline도 +1.028px 높았다. 이는 재사용 TEST의 탐색적 비교이며 독립 일반화 판정은 아니지만, 현재 결과로 “이전 방법보다 전반적으로 더 좋다”는 주장은 지지되지 않는다.

외부 가림은 8개 점·8장 모두 한 촬영 세션에 있다. 새 방법은 raw보다 낫지만 초기 rigid 대비 N3 +0.709px, PoseFix+Sub +0.798px로 악화됐다. 세션 하나를 반복 추출한 구간은 불확실성을 추정하지 못하므로 의미 있는 95% CI로 표기하지 않았다. human TRUNCATED와 별도 FULLY_HIDDEN label의 유효 좌표는 이 범위에서 0개다. 해당 상태에 대한 성능 향상을 주장할 근거는 없다. UNANNOTATED는 별도 진단이며 숨은 점 개선 증거에 합치지 않았다.

## 실제 국소 보정의 이익과 rigid 출력의 차이

실제로 좌표가 바뀐 predicted boundary anchors의 동일점 비교는 N3 TEST 100점에서 raw 4.236px → 측정 3.408px, PoseFix+Sub 101점에서 4.473px → 3.447px였다. 개선/악화 점 수는 각각 66/34와 67/34다. 세션 단위 평균 차이 구간은 N3 [−1.327, −0.525]px, PoseFix+Sub [−1.178, −0.839]px였다. 이 관측은 **국소 이미지 측정이 일부 좌표를 개선했다**는 제한된 근거다.

같은 점들의 최종 rigid projection은 각각 4.178px, 4.350px다. 보정 측정의 이익 일부가 하나의 rigid pose와 초기 prior를 맞추는 과정에서 사라진다. anchor 보호는 seed의 projection과 loss를 기준으로 하며, GT pixel error를 제한하거나 모든 측정에 정확히 맞추는 규칙이 아니다. 등록 모델·virtual/rounded corner·카메라·초기 자세의 오차도 숨은 모서리로 전달될 수 있다. 이 설명은 구현 계약과 관측된 패턴에 따른 해석이며 개별 실패 원인의 인과 인증은 아니다.

실제 RGB 내부 junction은 CAL에서 6점 중 2점이 좌표 오차 개선, 4점이 악화되어 평균 raw 4.426 → 측정 5.210px였다. N3 TEST에서는 5점 중 4/1 개선/악화로 6.566 → 5.616px였지만 최종 pose fit에 쓰인 frame은 0장이었다. PoseFix+Sub TEST는 8점 중 6/2로 5.181 → 5.296px였고 최종 fit 사용은 1장뿐이었다. CAL 최종 fit 사용도 2장이다. 따라서 “실제 내부 교점 보충이 숨은 점 개선을 일관되게 만든다”는 결과는 없다. 교점 관측 자체와 최종 fit 사용을 구분해야 한다.

봉인된 stage 기록을 보면 N3의 RGB junction은 fixed-seed anchor budget을 넘겨 거절되거나 충분한 support가 없어 보류됐다. PoseFix+Sub에는 RGB 관측을 포함한 bootstrap이 일단 수락되었다가 최종 immutable-pool 검사에서 탈락한 사례도 있다. 따라서 내부점이 최종 fit에 없다는 결과는 RGB 측정이 생성되지 않았다는 뜻이 아니다. 실제 추가 관측과 기존 anchor 보호·전체 pool 검사가 충돌하는 보수적 설계의 tradeoff다. 해당 trace는 JSON에 포함했고, 이를 보고 guard를 완화하거나 TEST 재선택을 수행하지 않았다.

## 조건부 상태와 uncertainty의 실측 한계

최종 사용점의 human state를 확인하면 N3 TEST에 SELF_OCCLUDED 2점, PoseFix+Sub에 4점이 포함되었다. CAL에도 2점이 있었다. 코드가 제외하는 것은 **초기 모델에서 self-hidden으로 추정된 점**이며 실제 human self-occluded point를 항상 제외한다는 명제는 성립하지 않는다. 이 점들은 initial-H/가시성/모서리 대응의 조건부 성격을 보여준다. human EXTERNAL_OCCLUDED 점은 최종 selected fit에서 0점이었지만, 외부 물체 판별기가 그 물체의 존재를 확정한 것은 아니다.

유한 positive projected covariance가 있고 model-compatible인 accepted corner에만 2차원 nominal 95% ellipse를 적용했다. 실제 포함률은 CAL 60.16%(128점), N3 TEST 67.5%(160점), PoseFix+Sub 63.13%(160점)였다. N3 TEST self subset은 66.67%(33점), 외부 가림은 25%(4점)였다. PoseFix+Sub self는 59.38%(32점), 외부 가림은 0%(3점)였다. 이는 local L2 surrogate·가정한 pixel noise·prior·고정 초기 회전·미모델링 bias의 한계에 대한 진단이며 calibrated confidence가 아니다. fallback frame에는 covariance가 없으므로 전체 frame coverage로 부르면 안 된다.

## 방법 완성과 성능 주장

새 selected의 전체 TEST N3 자세는 이동 3.038cm·회전 6.488°·성공 40/49로 초기 성공 40/49를 늘리지 못했고, 이전 진행 보정의 43/49에도 미치지 못했다. PoseFix+Sub에서는 2.949cm·6.313°·42/49로 자체 초기의 3.216cm·6.350°·39/49보다 개선됐지만 boundary-only의 2.797cm·6.317°·42/49보다 이동 오차가 높았다. 같은 관측의 UNPROTECTED는 43/49였으므로 protection의 관측 보호와 성공 수 사이에 tradeoff도 있다. 이 결과로 TEST를 보고 보호를 끄거나 frame별 winner를 선택하지 않았다.

현재 구현과 실험은 재현 가능한 방법론 연구 산출물로 완성되어 있다. 논문은 실측/투영 분리, 보호 불변량, 관측에 따른 DOF, 실패·fallback 공개와 국소 좌표 개선을 중심으로 주장할 수 있다. **새 방법이 기존 rigid completion보다 숨은 점을 더 정확하게 복원하거나 최고 refiner라는 주장은 아직 뒷받침되지 않는다.** 추가적인 unseen 촬영 검증 없이 strong generalization/SOTA 주장을 하지 않는다. 현 결과와 실패 설명을 숨기지 않고 제공하는 것이 paper-ready와 최고 성능을 구분하는 기준이다.

분석 스크립트는 `per_state_analysis.py`다. 원본 inference·seal·규칙·weights는 수정하지 않았고, 모든 비교 및 uncertainty 진단은 예측 봉인 이후 수행했다.
