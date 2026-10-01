# 189차원 anchor 문맥의 학습 전 독립 검토

**구현·입력 계약 검산 PASS. 성능 개선을 뜻하지 않는다.** 실제 학습이나 source VAL/실사 성능 평가는 실행하지 않았다.

고정 map은 `[z94, abs(z−z_R0_GEO)94, anchor_identity1]`이다. z는 기존 float32 정규화를 끝낸 뒤 float64로 변환한다. 추가 정규화나 학습되는 feature extractor는 없다. anchor는 입력만으로 저장된 R0 GEO의 0/1 index이며 참조 T/R에서 선택하지 않는다. identity는 숫자 특징이 같은 다른 후보가 아니라 그 index 하나에만 1이다.

독립 NumPy 행 반복 구현과 합성/실제 TRAIN map이 bit 단위로 일치했다. invalid 후보의 189열은 모두 0이며 score는 +∞다. anchor가 없는 all-invalid 한 행은 그대로 남고, 유효 후보가 있는데 anchor가 없거나 잘못된 index인 4종 계약 위반은 STOP한다. 기존 94 scorer를 추가 95가중치 0으로 넣을 수 있음을 확인했다. 점수 차이는 수치 합산 순서에 따른 최대 1.78e-15였으며, 절대 차이 map이 실제로 순위를 바꿀 수 있는 합성 예도 확인했다.

CE+ridge와 anchored target 함수 AST는 이전 구현과 같다. 독립 행별 CE/기울기 및 189개 좌표 중앙차분으로 검산했으며 최대 차분 오차는 2.5e-10다. 원래 valid 후보가 모두 CE 경쟁자로 남고 all-invalid 행도 전체 분모에 포함된다. 고정 map에 대한 선형 점수이므로 가중치에 대한 볼록성은 유지된다. 이는 새 특징이 일반화한다는 증거는 아니다.

기존 source eligibility로 TRAIN **2,598개**의 순서·ID·source-index SHA를 복원했다. anchor 유효 **2,597개**, anchor=-1은 기존 `TEX__shard_04_f0110` 한 행으로, 모든 모델에 그대로 포함된다. 4모델 원본 valid SHA, anchor index SHA, 기존 float32 정규화 SHA 및 94열 보존을 확인했다. 이전 SOURCE_FEASIBILITY와 4개 fit 영수증의 target/safe-mask SHA를 연결했다. **TRAIN label NPZ와 새 target 값은 읽거나 계산하지 않았다.** VAL은 컨테이너에 있지만 선택·변환한 행은 TRAIN만이다.

선행과의 차이:

- **Stage4 expert router**: 이미 expert box/keypoint/confidence/pose 차이와 선택 margin, 선택적 GAP448 문맥을 사용했다. 두 expert의 W/D를 먼저 선택한 뒤 ADDnorm 기반 MLP routing이며, 4 whole-pose 후보의 고정 189 선형 map이 아니다.
- **Model-conditioned selector**: model별 기존 94차원 GEO_LINEAR를 따로 학습한다. 공유 per-candidate 선형 점수, source parity BCE, VAL early-stop이며 고정 R0 GEO 기준 abs94+identity1은 없다.
- **N2/Replay utility**: 후보 좌표 차이와 corner/partner/global RGB·수치 문맥을 사용한다. corner utility로 vertical pair를 교체하는 비선형 CNN이며 whole-pose T/R anchored CE와 다르다.
- **DHT local no-harm**: baseline 대비 2D 오차의 ReLU no-harm 및 line utility를 학습하는 연속 좌표 교정 목적식이다. 이 목적식의 baseline 비교와 이번 입력-only abs-normalized pose-feature context는 같은 연산이 아니다.

상대 후보 문맥과 baseline 대비 학습의 선행은 이미 있다. 이번 좁은 변경은 고정 operational R0 anchor의 abs94+identity1을 이전 anchored-target 볼록 선형 문제에 넣는 것이다. 새 정보/후보/이미지 forward/GT를 더하지 않으며, 과거 실패가 해결됐다는 주장은 별도 고정 source VAL gate 이후에만 평가할 수 있다. 입력은 이미지 한 장·치수·기존 K 계약을 유지한다.

[기계 검산과 SHA](PREFIT_REVIEW.json). 선행 코드의 정확한 경로·SHA와 범위 한계도 JSON에 기록했다.
