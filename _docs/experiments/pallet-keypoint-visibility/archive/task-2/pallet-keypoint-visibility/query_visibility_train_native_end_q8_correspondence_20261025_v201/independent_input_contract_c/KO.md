# V201 원 q8 입력 계약의 독립 검토

원래 함께 저장된 q8는 사용할 수 있는 문맥이다. 다만 **배포 시 이용 가능한 문맥**과 **active-only 증강 offset을 쉽게 알려주는 source 단서**는 별개다. 이 검토는 q8를 금지하거나 새 입력·head를 선택하지 않는다. V200 결과는 설계 근거로 쓰지 않았다.

V183 producer의 `transform`은 한 active row에만 `q′=q+Δ`, `L′=L·s`를 적용한다. 원 scene의 `original_input_metadata8`은 변하지 않으며 SELF15는 새 simultaneous initializer8이 아니다. BEFORE4도 원 BEFORE anchor 문맥을 네 image variant에 고정한다. V181 public/CF loader는 원 frame 또는 pair/variant native8의 q/L/availability를 검사한다. 같은 `original_global_index`가 여러 occurrence에 반복되므로 인접8행이나 그 ID 하나로 q8를 만들 수 없다.

## 두 가지 작은 scalar 계약의 tradeoff

| 계약 | 모델에 줄 수 있는 scalar | 보존되는 정보 | 잃거나 노출되는 정보 |
|---|---|---|---|
| active-crop relative other7 | `(q_j_original − q_active)/L_active`, 명시 mask8, native slot; self coordinate/mask0 | 다른7점과 실제 crop의 공간적 정렬 | active 이동은 모든 점에 공통 `−Δ/L`로 남고 scale도 점 간 간격에 남는다. Source-only jitter를 간접 복원할 수 있다. |
| other7 layout-only | 최저 usable 다른 native 기준점, 다른 점들 최대 거리로 정규화한 좌표·mask·native; active q/L 인자 없음 | 원 다른7점의 상대 배치·방향과 missing pattern | active q/L jitter에 불변인 대신 의도한 점이 active crop 안 어디에 있는지의 연결도 제거한다. 더 좋은 계약으로 자동 채택할 근거가 아니다. |

`original_other7_scalar_contract.py`는 두 계약을 **별도 함수**로 구현했다. 파일·NPZ·pixel·모델·GT·label을 읽지 않는 Python scalar 함수다. 같은 origin/role의 native0..7 정확8개를 순서와 무관하게 join한다. 사용 가능한 다른 점만 명시 mask로 내보내고, self의 원 xy는 읽지 않는다. nested GT/reference/pose/end metadata는 record whitelist에서 거부한다. 제공 active availability가 bound 원 selected flag와 다르면 거부한다. 원 active inputU는 forward 금지로 유지하며 missing context를 E로 바꾸지 않는다. Layout normalization이 불가능하면 context unavailable로 남긴다. 이 구현은 배포 입력 adapter의 작은 계약 예시이며 실제 cache/head에 연결하지 않았다.

## self0만으로 offset 정보가 사라지지 않는 가상 증거

이상적인 사각형의 세 다른 점이 `q0=q1+q3−q2`를 만족한다고 하자. 원 self q0를 넣지 않고 각 다른 점만 active-relative로 만들더라도,

`r1+r3−r2 = (q0_original − q_active)/L_active = −Δ/L_active`

가 정확히 남는다. Fixture의 Δ=(1.5,−0.5), L=20에서는 (−0.075,0.025)가 복원됐다. 다른 모든 점의 relative 좌표도 동일하게 이동한다. Pairwise difference는 공통 translation을 없애지만 L_active로 나누는 한 scale 신호가 남는다.

이는 **선언된 식에 어떤 정보가 들어 있는가**를 증명한다. 실제 TRAIN q8가 이상적 사각형이라는 주장도, 현재 모델이 이를 이용했다는 증거도, q8의 보편적 무효성도 아니다. RGB+q8가 항상 불충분하다는 결론도 아니다. 원 initializer q8가 정당하게 함께 제공되면 active-relative other7은 배포 가능한 입력이다. Synthetic active-only jitter의 규칙성과 실제 initializer의 동시 오차 구조가 다른지 별도로 확인할 문제다. Forbidden evaluation/GT leakage와 이런 source shortcut 위험을 혼동하지 않는다.

## q8 provenance를 같은 것으로 취급하지 않기

- 독립적으로 읽은 producer 코드는 원 scene q8를 복사하고 active crop만 변환함을 보여준다. q8를 원래 만든 upstream estimator의 물리 정확성까지 인증하지 않는다.
- root가 전달한 첫 bound V180 metadata 추적은 synthetic 원 q8에도 **V132 frozen_C16_actual_initializer_v1** 계보가 있음을 가리킨다. A의 상세 추적이 진행 중이므로 이 검토는 실제 synthetic q8를 GT projection으로 부르지 않는다. 위 사각형은 이상적 invented fixture일 뿐이다.
- Public은 원 SOURCE_DEVELOPMENT initializer 입력8·`native_continuous_valid` 계보를 유지한다. CF/edited RGB에서 initializer를 다시 돌린 값이라고 부르지 않는다. 원 context와 image-original/BEFORE 역할을 명시해야 한다.
- Input availability는 가시성·selected-instance·intended end의 인증이 아니다. 같은 수치 q8가 서로 다른 semantic 상황과 양립할 수 있다는 scalar 예는 새로운 실사 label을 만들지 않는다. Native slot도 배포 식별자이며 물리 vertex/end gold로 승격하지 않는다.

## 다음에 확인할 최소 항목

1. 각 TRAIN row의 context가 원 scene/원 BEFORE/원 initializer 중 무엇인지, active image origin과 anchor origin이 무엇인지 보존된 binding으로 exact join한다. 원 group8·native·availability·q/L을 검산하고 repeated ID/인접행·GT-nearest·다른 증강 mate fallback을 쓰지 않는다.
2. 비교하려는 입력이 crop 정렬 정보를 유지해야 하는지 prospectively 명시한다. Active-relative는 그 정렬과 offset 신호를 함께 준다. Layout-only는 둘을 함께 없애므로 자동 대안이 아니다. 원 q/L·RGB sampler·target은 바꾸지 않는다.
3. 실제 source q8 upstream provenance 및 active-only jitter 조건과 배포의 simultaneous initializer 조건 차이를 문서화한다. 원 self0만으로 모든 간접 단서가 제거됐다고 주장하지 않는다. 추가 mask/centering·head 설계·학습을 이 검토에서 고르지 않는다.

## 검증과 범위

24개 invented scalar/schema fixture가 통과했다. 공통 active 이동, 간접 anchor 복원, active scale, 공통 image 좌표 이동/scale 불변성, layout-only의 crop 연결 손실, self 원xy 제외, all native8, inputU/missing, 순서 변경 및 mixed origin/role/중복native/nested GT/nonfinite 거부를 확인했다. 이 결과는 TRAIN 전수집계나 모델 효과가 아니다. 원 NPZ/header/pixel/weights/평가/score/HOLDOUT/COMMON/VAL·V200 본문 신규 읽기와 model/forward/fit/PnP는0이다.

[scalar 계약 코드](original_other7_scalar_contract.py) · [가상 검사 코드](check_invented_scalar_contract.py) · [24개 결과](INVENTED_FIXTURES.json) · [검토 JSON](REVIEW.json)
