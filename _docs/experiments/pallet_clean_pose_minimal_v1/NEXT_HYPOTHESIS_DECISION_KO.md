# 추가 TRAIN 교차 진단 후 남은 학습 예산 판단

## 결론

**누락된 CLEAR43까지 대조를 완성하고, 남은 학생2회는 근거 없는 새 레시피에 사용하지 않는다.** 이것은 새 GEO 한 번이 실패했기 때문만이 아니다. 같은 입력의 TRAIN 교차 진단까지 추가한 결과, 가림에 대한 작은 학습 반응은 있지만 이를 더 늘리면 자연 가림 T/R가 좋아진다는 원인 근거는 아직 없다.

이 문서는 결과 전의 [가설 감사](NEXT_HYPOTHESIS_AUDIT.md)를 덮어쓰지 않는 후속 판단이다. 기존 예측/후보를 재사용한 강한 단순 대조, 단 한 번의 공통 GEO 학습, 실제 두 흐름의 2×2, 새 fit0의 같은-input TRAIN 진단을 함께 반영한다. 모든 연구 가설이 반박됐다고 주장하지 않는다.

## 1. 같은 masked TRAIN 입력에서도 가림학습의 이득은 작고 국소적이다

[사전 입력 잠금](TRAIN_CROSS_INPUT_PROTOCOL.json)은 기존 seed42 K8의 실제32 batch trace 재현 후62 real occurrence/45 unique,476지원코너를 고정했다. 이 중 실제 mask가 적용된 입력은9 occurrence이며 canonical REF 기준 가려진 점은21 occurrence다. 모든8학생은 완전히 같은 CLEAR/OCC RGB·타깃·support를 받는다. seed43 모델도 이 동일 seed42 진단 입력에서 비교했다.

기존 seed42 own-input4 context는 고정 예측을 재사용하고, 나머지12 context만 새로 추론했다. GPU 계측5.673917초, 새 fit0, optimizer 생성/step0, 평가 참조 읽기0이다. 최고 confidence 검출을 사용했으며 GT·타깃에 가장 가까운 detection이나 TAL anchor로 바꾸지 않았다. inference setup/fusion 이후 상태 hash 불변과 grad 없음, 동일 입력의 모든모델 detector parity를 검사했고 checkpoint 파일 SHA도 보존했다.

아래는 **동일 OCC 입력에서 OCC-trained−CLEAR-trained**의 own RAW/REF 의사 타깃 L2 평균 차이다. 음수는 타깃 추종 잔차 감소이며 물리 GT 정확도가 아니다.

| 학습 seed/target | covered21 Δmean px | unmasked455 Δmean px | 전체476 Δmean px |
|---|---:|---:|---:|
| 42 / RAW | −0.077645 | +0.019699 | +0.015405 |
| 42 / REF | −0.054194 | +0.020698 | +0.017394 |
| 43 / RAW | −0.065783 | +0.018567 | +0.014845 |
| 43 / REF | −0.024666 | +0.011304 | +0.009717 |

missing은 모든 context/타깃에서0이다. covered 평균의 감소 방향은 네 조건에 있으나 모든 점이 개선된 것은 아니다. covered 개선/악화는42 RAW15/6, REF16/5,43 RAW10/11, REF12/9다. 평균 감소와 개선점 개수는 다른 요약이다. 각 모델의 covered 평균 잔차는 여전히 약10.8–11.3px다. [전체 결과](TRAIN_CROSS_INPUT_RESULTS.json)에는 같은 CLEAR 입력 대조, 양쪽 RAW/REF 타깃, paired 평균/중앙값, 실패 패널티도 남겼다.

따라서 “가린 좌표를 전혀 배우지 못한다”는 설명은 지지하지 않는다. 동시에 이 작은 변화가 실제 pose 개선이나 충분한 가림 복원 능력을 뜻하지도 않는다. 표본은 작은 TRAIN prefix이고21점은 독립21영상이 아니다. 실제 attention을 계측하지 않았으므로 low-attention의 증명/반박이라고 부르지 않는다.

## 2. 마지막 두 fit로 ROI-mask를 바로 바꾸지 않는 이유

이전 감사에서 확인한 actual exposure418/2560(16.33%), REF covered632/19017(3.32%)는 남아 있는 구체적 경고다. 그러나 이번 교차 진단은 다음을 보여준다.

- 현재 mask를 본 모델은 같은 가린 점의 의사 타깃을 조금 더 따랐다. 학습 경로가 완전히 차단된 상태는 아니다.
- 같은 입력의 unmasked 다수 점과 전체 평균에서는 작은 손해가 있다. mask 노출 증가가 이 손익을 어느 방향으로 바꿀지 아직 모른다.
- exposure 부족, target 오차, 표현력/최적화, instance 선택을 이 소표본만으로 분리할 수 없다. 높은 residual 자체는 원인 선택 기준이 아니다.
- 과거640 update·높은 LR·보조 SmoothL1·agreement gating·빈도 증가의 유효한 결과를 숨기고 새 계수를 찾는 방식으로 이어가면 적응적 sweep이 된다.

ROI/지원점 기반 placement, partial unfreeze, 새로운 독립 TRAIN reliability cue, geometric loss는 여전히 **미검증 가설**이다. 이 중 어떤 하나도 현재 증거에서 자연 가림 T/R 개선 가능성이 구체적으로 확인된 원인 수정으로 올라오지 않았다.2회로 하나의 고정 변경을 두 seed에서 확인하는 설계는 가능하지만, 지금은 그 변경을 선택할 충분한 근거가 없다. 따라서 잔여 예산을 남기는 것이 정당하다. 이후 새 TRAIN 모순·신뢰성 근거·입력분포 진단이 생기면 같은 자료와 원장에서 재개할 수 있다.

## 3. 최종33개 구성의 독립 수치 점검

[FINAL_RESULTS.json](FINAL_RESULTS.json)과 별도 private frame metrics에서 자연99의 T/R 중앙값·P90을33개 구성 모두 다시 집계했다. 모두 valid99/99였고 저장 결과와 일치했다. **OLD_REF217+old GEO의 T/R 중앙값을 동시에 엄격하게 낮춘 구성은0개다.** 이는 엄격한 수치 비교이며 통계적/실용적 성공 기준을 새로 만든 것이 아니다.

OLD_REF217+old GEO는 T11.410110cm/R4.955259°의 **고정된 강한 참조**로 남길 수 있다. 그러나 전 지표에서 최적이거나 유일한 Pareto 해라고 주장하면 안 된다. T/R 중앙값만의 Pareto 집합에는 다음 네 구성이 있다.

| 구성 | T 중앙값 cm | R 중앙값 ° | T P90 cm |
|---|---:|---:|---:|
| OLD_REF217 + old GEO | 11.410110 | 4.955259 | 127.143068 |
| REF CLEAR42 + new GEO | 11.987881 | 4.891090 | 128.303501 |
| REF OCC42 + new GEO | 11.427597 | 4.946747 | 128.644978 |
| REF CLEAR43 + old GEO | 11.184092 | 4.985203 | 128.533569 |

이 Pareto 집합도 tail·Clean 보존·촬영 기록·복잡도·반복 안정성을 포함하지 않은 기술적 요약이다. 예컨대 RAW 계열의 T P90이 더 낮은 대조도 있으므로 fixed reference의 tail 한계를 숨기지 않는다.

REF CLEAR43+old GEO−고정 참조는 T 중앙값 −0.226018cm(−2.26018mm), R +0.029944°다. 공동 개선이 아니다. 더구나 **프레임별 차이의 중앙값**은 T +0.135431cm/R +0.094472°로 둘 다 악화이며,99장 중 공동 개선16/공동 악화43/T만 개선25/R만 개선15다. 모집단 중앙값의 차이를 typical-frame 변화와 혼동하지 않는다. 이 한 seed의 낮은 T만 골라 robust upgrade로 승격하지 않는다.

마지막 판단은 가장 좋은 seed/selector를 프레임이나 상황별로 섞지 않고, 실제 고정 구성의 모든 비교·난도·기록·큰오류를 보존해야 한다. 반복 사용 DEV와 기하 재구성 pose 참조의 한계는 이번 추가 검사로 없어지지 않는다.

## 근거 잠금

다음 hash는 이 판단이 사용한 정확한 결과를 연결한다. 학생/선택기 비용은 별도 최종 원장이 담당하며 이 문서는 원장을 수정하지 않았다.

| Artifact | SHA256 |
|---|---|
| TRAIN_CROSS_INPUT_PROTOCOL.json | 398cf15d906ee8948bbf8ae8c4a1d473a54d54bc4918bd33460db36b43cc8881 |
| TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json | a956af28e3457a62b7918ef0ba6113a8e646f8899060e9def4f913b27a6f5c20 |
| TRAIN_CROSS_INPUT_RESULTS.json | 8a2fec7730f297da95d283a15060f380291593b7ba1e94605bdbc83aa5241c95 |
| FINAL_RESULTS.json | a1c1b75dead0d6320f14e25fdb4f65ac20ca8c763de044f63083057176ef7e23 |
| NEXT_HYPOTHESIS_AUDIT.md | 56ed3dfbeb366f765fcb92c1c5ef52445f8ade3606e9618804645fab18e35593 |
