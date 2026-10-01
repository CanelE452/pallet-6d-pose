# 부호 있는 재투영 잔차 18개를 추가하는 단일 입력 대조

목표는 자연 가림에서 위치 T와 회전 R을 안정적으로 함께 개선하는 것이다. 입력은 RGB 이미지 한 장과 팔레트의 물리 치수이며 기존 카메라 보정 K를 사용한다. 이번 비교는 기존 253차원 입력에 고정된 방향 성분 18개만 추가한다. 입력 진단 통과를 성능 성공으로 취급하지 않는다.

## 근거와 반대 근거

직전 Huber+sign 회귀는 네 모델 모두 수렴했지만 합성 VAL의 45개 조건 중 43개만 통과했다. seed 3의 T 중앙값은 R0 기준 1.674594cm보다 큰 1.683872cm였다. 손실 최적화의 완료가 올바른 후보 선택으로 이어지지 않았다. [직전 결과](../pallet_pose_signed_axes_sign_20261001_v1/REPORT_KO.md)

별도 TRAIN 입력 감사에서 기존 253차원과 새 방향 18차원의 선형 중복을 계산했다. 기존 입력 밖에 남는 추가 성분의 Frobenius 비율은 R0 0.189749, UNION 3개에서 0.422551/0.433444/0.436207이었다. 모든 모델의 수치 rank가 18 증가했다. 이는 방향 성분을 추가할 입력상 근거이며, 정답 예측 가능성이나 실사 전이의 증거는 아니다. 정확히 같은 입력에서 양·음 타깃이 충돌하는 그룹은 없었다. 따라서 기존 실패의 원인을 정보 소실이라고 확정하지 않는다. [입력 감사](../pallet_pose_residual_direction_audit_20261001_v1/REPORT_KO.md)

과거 RGB MLP·structured DHT·signed utility 입력도 전이에 실패했다. 새 18개는 독립적인 RGB 관측이 아니라 기존 예측과 고정 pose에서 얻는 값이다. 기하적으로 일관된 오검출이나 데이터 분포 차이를 해결하지 못할 수 있다. 이 비교가 실패하면 기준을 바꾸거나 추가 seed·임계값을 탐색하지 않는다.

## 변경하는 한 가지 요인

기존 후보의 camera-facing `R_cf`, 중심 t, `cf_extents`로 코너 0–7과 중심 8을 만들고 같은 좌표계의 K로 투영한다. `d18 = flatten((투영점 − 관측 q9) / bbox 대각선)`을 float32로 저장한다. bbox 폭·높이의 최소값 1e−6, 점 순서, 중심 포함 여부와 부호를 유지한다. 새 PnP·이미지 추론·참조 pose 계산은 없다.

기존 94차원 정규화와 RBF 64개의 중심·폭은 그대로 사용한다. 추가 18개는 앞선 입력 감사에서 고정한 R0 TRAIN 유효 후보 5,194개의 float32 평균·표준편차(최소 1e−6)만 사용한다. 정규화 후 float64로 변환해 같은 프레임의 R0 운영 GEO anchor를 뺀다. 기존 253차원 차분 뒤에 이 18차원 차분을 붙여 271차원 입력을 만든다. 기존 253차원과 타깃의 6개 해시를 보존하고, 원래 방향·방향 차분·확장 입력의 3개 해시를 별도로 기록한다.

source의 q/K/이미지는 이미 padding을 반영한 좌표이며 다시 100픽셀을 더하지 않는다. real은 기존 raw 좌표를 유지한다. 원래 물리 치수와 camera-facing extents를 혼동하거나 후보 좌표에 C2 대칭 변환을 새로 적용하지 않는다. source/real 추출은 기존 잔차 크기와 `atol=1e−4px, rtol=1e−6`으로 대조하며 불일치하면 중단한다.

## 유지하는 학습·추론 계약

- 모델은 R0_ONLY 하나와 UNION_s1/s2/s3 세 개다. 네 fit을 모두 271×2 영가중치로 시작하며 최고 seed를 고르지 않는다.
- 적격 TRAIN 2,598행, 전체 후보 실패 1행, 원래 valid mask와 R0 anchor, 물리 T/R signed-log1p 타깃과 scale을 유지한다. invalid와 anchor의 차분 입력은 정확히 0이다.
- 각 프레임의 원래 유효 후보와 두 축에 대해 `Huber(p−y, δ=1) + 1{y≠0} softplus(−sign(y)p)`를 평균하고 전체 2,598행으로 평균한다. 전부 실패한 행은 0으로 기여하되 분모에 남는다. sign 계수는 1, bias는 0, ridge는 모든 542개 파라미터에 λ/2×제곱합, λ=1e−4다.
- 직전의 Newton/Armijo, 초기 step 1, backtrack 0.5, c1=1e−4, damping 0을 유지한다. 각 fit은 최대 1,000 accepted iterations·2,000 objective calls다. 재시작·budget 연장·tolerance 탐색은 없다.
- 수렴 인증은 optimizer 성공, gradient L∞≤1e−8와 `||g||²/(2λ)≤1e−6`을 모두 요구한다. 이는 FP64 계산의 수렴 검산이며 실수 구간 증명이나 T/R 보장은 아니다.
- runtime은 모든 원래 유효한 전체 pose 후보에서 두 예측 축의 max가 가장 작은 후보를 고른다. 원래 동률 규칙과 실패 fallback을 유지하며 margin·safe mask·GT 기반 선택은 추가하지 않는다.

## 실행 순서와 원래 목표의 판정

코드와 입력을 사전 검산한 뒤 TRAIN_PROTOCOL에 고정한다. 순서는 네 fit 수렴 인증 → 독립 학습 검산 → GT를 읽지 않는 source 후보 선택 고정 → source VAL 1,024행 전체 평가다. source 45개 조건을 모두 통과해야 실사 후보 선택·평가를 진행한다. 하나라도 실패하면 실제 미실행 기록과 전체 실패 조건을 공개한다.

source는 각 seed의 T/R 중앙값 strict 개선, P90 5% 이내, 실패 수 비증가를 R0_ONLY·R0_GEO·paired DIVERSE_GEO에 그대로 적용한다. 실사는 natural99·clean29·wood45 전부를 유지하고 IoU로 제외하지 않는다. 자연 가림의 세 seed 공동 개선, paired recording×seed bootstrap 2,000회(seed 20261001), 6개 recording 각각 제외, natural 꼬리·실패 및 clean 보존 조건을 유지한다. **원래 paired SINGLE251 비교와 matched-control 비교를 항목마다 AND**한다. 반복 사용한 DEV에서 통과하더라도 새 recording의 일반화를 입증한 것으로 확대하지 않는다.

후속 보고서는 실제 수렴/선택/성능과 입력 진단을 구분한다. 전체 CSV·학습 파라미터·실행 기록과 고정 이미지·물리 치수를 GitHub에 공개하고 원격 파일을 검증한다. 원래 dirty 작업 트리와 기존 보고서·체크포인트·GT는 덮어쓰지 않는다.
