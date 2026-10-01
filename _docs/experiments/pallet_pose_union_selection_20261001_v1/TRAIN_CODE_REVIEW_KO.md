# Source TRAIN gate와 전체 pose ranker 코드 검토

2026-10-01. 아래 세 파일을 읽어 후보·라벨·수학·분할·예산의 일치를 확인했다. 이 최종 검토에서는 기존의 analytic 검사를 반복하지 않았고, 실제 source/VAL/실사 품질값을 열거나 학습·추론을 실행하지 않았다. 결과 수치나 학습 성공을 판정하는 문서가 아니다.

**검토한 범위에서 gate와 학습 target을 바꾸거나 허용오차를 완화해야 할 새로운 차단 결함은 발견하지 못했다.** 이전에 발견한 중앙값의1ulp 재검증 문제는 저장된 scale을 수정하지 않고 같은 연산 순서로 확인하도록 고쳐졌다. 아래의 surrogate loss와 fallback 차이는 공개해야 할 해석 범위다.

| 코드 | SHA256 | 검토 범위 |
|---|---|---|
| [source_features.py](../../../scripts/research/pallet_pose_union_selection_20261001_v1/source_features.py) | `f36a26ebeb88e1cd36697c8be92c4080860de437407c67423c848baf03d07cfc` | 특징/pose 동결, 적격 TRAIN 라벨, scale·cost·gate |
| [train.py](../../../scripts/research/pallet_pose_union_selection_20261001_v1/train.py) | `26cf86cd3b2d3ede9c5d9dbe01ecd80508072b8ba36f6a800648f0dbf1b19e30` | 학습 입력·target·loss·정규화·paired 반복·receipt |
| [seal_training.py](../../../scripts/research/pallet_pose_union_selection_20261001_v1/seal_training.py) | `897a52c84b827fb25e46d34608a88da4af02e6d7ee71c12915988952c55e3f5e` | TRAIN gate 통과 조건과 후속 학습/VAL 계약 고정 |

## 후보와 target이 일치하는가

`source_features.choose_cost()`와 `train.targets()` 모두 후보 순서를 R0 long/short, 해당 DIVERSE251_s long/short로 만든다. R0_ONLY는 앞의 두 후보만 사용한다. 세 refiner seed를 한 pool에 합치지 않는다. 각 후보의 T와 R는 같은 완전한 pose의 값이다.

둘 다 float64에서 `max(T/sT, R/sR)`를 계산하고 유효 후보의 최소 cost를 target으로 고른다. exact cost 동률의 집합에서만 Pareto 지배 후보를 제거하고, R0 우선·가설 이름 순으로 결정한다. 학습 코드의 마지막 expert 이름 정렬은 현재 하나의 R0와 하나의 DIVERSE pool에서 앞선 기준과 중복되므로 두 구현의 선택을 바꾸지 않는다. 근접 동률 epsilon이나 GT 기준의 inference gate는 없다.

GT-free candidate validity는 finite94-feature와 기존 pose available의 교집합이다. TRAIN 라벨 컨테이너는 valid 위치의 두 축이 finite/nonnegative이고 invalid 위치는 두 축 모두+∞인지 확인한다. raw feature의 invalid 자리에는0 또는 무시할 finite vector가 있을 수 있지만 mask=false이면 softmax 선택 대상에 들어가지 않는다.

## Loss와 실제 목표의 관계

학습은 target 후보의 **음수 scalar score에 대한 masked cross entropy**다. 따라서 target pose의 예측 score를 낮추는 방향이며 inference의 `argmin(score)`와 부호가 맞는다. 유효 후보가0개면 target−1,1개면 해당 후보를 선택할 수 있지만 ranking loss는0이다. 둘 이상인 행의 CE를 합쳐 원래 minibatch 행 수로 나누므로 실패/단일후보 행을 분모에서 몰래 삭제하지 않는다.

이 모델은 T/R cost 숫자를 회귀하지 않고 **cost-best 후보의 분류·순위**를 학습한다. 큰 손상과 작은 손상의 cost margin을 직접 가중하지 않으며, CE 감소가 T/R 중앙값 또는 P90 개선을 보장하지 않는다. Source TRAIN의 scalar target이 공동 개선 방향을 갖는지 먼저 검사하고, 학습 후 별도 VAL gate에서 다시 축별 결과를 확인하는 이유다. Shared scalar score를 물리 오차의 보정된 예측값이나 확률로 설명하면 안 된다.

한 프레임에 후보가 네 개여도 optimizer는 그 frame loss 하나를 사용한다. UNION이 R0_ONLY보다 후보가 많아서 source frame이나 업데이트 수를 두 배 사용하는 구현은 아니다. 두 분류 문제의 softmax 난도와 정답 분포는 달라지므로, 동일 예산이 동일한 학습 난도를 뜻하지는 않는다.

## 적격성·라벨 연결·읽기 분리

[최종 source 계약](SOURCE_CONTRACT.json)의 적격 ID와 [TRAIN 가능성 검사 계약](TRAIN_FEASIBILITY_PROTOCOL.json)의 TRAIN2598/VAL1024를 대조한다. source_features는 전체5120행의 feature/pose를 먼저 잠근 후, declared C2이면서 메타데이터의 canonical cuboid→Xcf 대응이 proper rigid인 TRAIN만 채점한다. C1 1497행과 반사 형상 C2 한 행은 전체 입력에서 보존하지만 이번 학습 target에는 들어가지 않는다. 실제 모델 T/R에 따라 제외한 규칙이 아니다.

Renderer 참조는 historical sidetable/records의 binding을 검증한 뒤 적격 TRAIN table_index로 가져온다. ID·prepared K·physical dims가 일치하는지 확인하며, runtime 회전 참조는 `R_table @ diag(1,-1,-1)`, translation은 원래 t다. C2 full-rotation을 비교하고 source/runtime basis 변환의 동일 거리도 확인한다. 새로운 solver,90° 동치,GT box 또는 corner 재배정은 추가하지 않는다.

`freeze()`는 별도 strict source-label 접근 차단 hook을 설치한다. `train_gate`는 다른 프로세스에서 특징 lock 이후에만 source 참조를 연결한다. 학습 단계는 다시 source raw-label/VAL-quality/real-reference 접근을 금지하고 허용된 두 NPZ만 읽는다. feature 컨테이너에는5120행이 있지만 fit tensor와 정규화는 `source_index`로2,598행을 고른 뒤 만든다. VAL feature를 컨테이너에서 읽는 것과 VAL 값을 학습/정규화에 쓰는 것을 구분해야 하며, 현재 코드에서는 후자를 하지 않는다. TRAIN 라벨 NPZ에는 VAL/C1 quality 배열 자체가 없다.

## Scale과 정규화의 기준이 공통인가

T/R scale은 **동결 R0+기존 GEO의 적격 TRAIN full-population 중앙값** 두 개다. 새로운 ranker가 고른 값이나 UNION 최선값으로 scale을 다시 맞추지 않는다. 두 값이 finite·양수가 아니면 SOURCE_TRAIN_GATE가 STOP이며 학습이 시작되지 않는다.

학습 입력 검증은 NPZ의 저장된 scale이 원래 operational R0 T/R에서 다시 계산한 중앙값과 같은지 확인한다. 처음의 `np.median` 검증은 source gate의 `lo+0.5*(hi−lo)`와 float64 연산 순서가 달라1ulp로 중단될 여지가 있었다. 최종 `train.source_median()`은 source gate와 같은 정렬·rank·연산 순서를 사용한다. **이 수정은 scale 값이나 target·허용오차 변경이 아니다.** 해당 반례는 이미 완료한 analytic selfcheck에 들어 있고 이번 검토에서는 재실행하지 않았다.

94-feature mean/std는 적격 TRAIN의 유효 **R0 두 후보**에서만 계산하고 std floor1e−6을 사용한다. DIVERSE 출력·VAL·실사 통계는 섞지 않는다. 동일 mean/std를 여섯 학습과 모든 후보에 사용한다. UNION의 새 특징이 R0 기준의 범위를 크게 벗어날 가능성은 남지만, 이를 보고 정규화를 재조정하는 절차는 없다. normalization SHA를 paired receipt에서 비교한다.

## TRAIN gate가 검사하는 것

세 UNION의 source cost-best whole-pose 선택 각각을 R0_ONLY cost-best 선택과 비교한다. 두 축의 **full-population 중앙값이 모두 엄격히 작고**, 두 P90이 각각1.05배 이내이고, 실패 수가 늘지 않아야 PASS다. +∞를 포함한 extended-real quantile로 검사하며 conditional 통계는 별도로 보고한다. 어떤 frame도 새 quality 기준으로 삭제하지 않는다. 저장된 세 seed의 PASS를 모두 요구하므로 좋은 seed만 골라 fit하는 경로가 없다.

이는 source의 같은 scalar cost에 대한 학습 기회 검사이지 후보 합집합 전체의 가능성을 증명하는 bound가 아니다. 실패는 이 고정 objective/데이터 설계의 중단 근거이며, 모든 다른 selector도 불가능하다는 뜻이 아니다. 통과도 Linear94가 좋은 후보를 식별할 수 있거나 실사에서 일반화한다는 증거가 아니다.

Pareto 기회 count는 R0 두 후보와 refiner 두 후보의 모든 유효 쌍을 비교하는 관측치다. 서로 다른 W/D branch 비교도 포함되므로 이 count만으로 “refiner 좌표 자체가 개선된 프레임 수”를 주장할 수 없다. 실제 gate는 scalar cost가 선택한 한 whole pose를 사용한다.

## Fallback의 명시적 차이

TRAIN feasibility의 사용 가능한 후보가0개이면 두 오류를+∞로 기록한다. 학습의 해당 행은 label−1/loss0이다. 이후 `seal_training.py`가 고정하는 VAL/runtime 정책은 **양 arm 모두 동일한 saved R0 GEO pose로 fallback**, 그마저 unavailable이면 failure다. 이는 특정 arm에 유리한 GT 선택이 아니며 학습 target을 만들어내는 경로도 아니다.

다만 feasibility의0후보 실패 통계와 후속 operational fallback 통계를 숫자 그대로 같은 것으로 부르면 안 된다. 둘의 역할을 분리해 보고해야 한다. 이 fallback 선언의 실제 VAL 적용·최종 실사 구현까지 세 파일의 코드 검토로 확인했다고 주장하지 않는다. 그 단계의 lock·receipt·평가 코드 검토가 따로 필요하다.

## Paired 비교와 실행 예산

각 seed의 두 arm은 같은 초기 Linear94 상태, 같은 TRAIN 정규화, 같은2,598행의 epoch별 permutation과 minibatch 경계를 사용한다.30epoch×ceil(2598/256)=**330 update/fit**, 최대6fit=**1,980 update**다. 마지막 batch는38행이며 양 arm에 같은 frame이 들어간다. batch 평균 CE를 쓰는 일반적인 SGD 가중치 구조이며, 서로 다른 arm에 추가 sampling이나 별도의 epoch 수를 주지 않는다.

코드는 초기 상태·normalization·order·각 step의 row batch SHA를 보존하고 paired 완료 검사에서 대조한다. AdamW의 lr0.001/weight_decay0.0001, CPU 한 thread, final epoch30만 사용한다. 조기 종료·best-VAL epoch·extra seed가 없다. 중간 fit이 receipt 없이 남으면 자동 재시도하지 않고 명시적 실패로 중단한다.

`seal_training`은 complete/PASS인 TRAIN gate 없이는 학습 protocol을 만들지 않는다. 실제 loader는 gate/feature/label/source contract/prediction lock/runtime amendment의 cross-binding과 코드 SHA를 다시 확인한다. Seal 문서에 값이 있다는 사실만으로 학습을 한 것은 아니며, 여섯 final checkpoint와330-step receipt가 모두 있어야 완료다.

두 arm의 데이터·초기값·예산은 짝지었지만 UNION의 refiner training seed와 selector optimizer seed는 함께 바뀐다. 이 세 반복을 두 변동 원인의 독립 요인 실험으로 해석하지 않는다. 또한 원래 source/refiner 노출 이력과 반복 실사 DEV의 한계는 그대로다.

## 검토 범위의 결론

현재 구현은 사전에 고정한 후보·scale·target·loss 부호·동률·TRAIN-only 정규화와 최대6fit 예산을 일관되게 연결한다. 기존 analytic 검사 결과와 수정 이력은 [학습 설계 검토](TRAINING_DESIGN_REVIEW_KO.md), 입력/runtime 실패 및 개정은 [실행 계약 검토](RUNTIME_AND_SOURCE_METHOD_KO.md)에 설명돼 있다.

이번 검토는 실제 TRAIN gate 결과, 최적화 수렴, VAL 통과, 실사 T/R 향상 또는 목표 완료를 판정하지 않았다. 실제 실행 결과와 full failure·세 seed·clean/tail·recording 기준은 후속 잠금 자료와 최종 보고서에서 별도로 확인해야 한다.
