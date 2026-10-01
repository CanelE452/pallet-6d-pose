# TRAIN 위험 가산항 손실의 학습 전 독립 검산

**계약·수학 검산 PASS. 새 fit과 성능 평가는 실행하지 않았다.** 이번 변경은 후보별 `risk=max((T−Ta)/sT,(R−Ra)/sR,0)`의 `log1p(risk)`를 TRAIN logit에 더하는 하나의 손실 변경이다. margin은 참조를 쓰는 TRAIN 상수이며 runtime 입력이나 선택 mask가 아니다.

고정된 target y의 margin은 정확히0이다. 새 손실은 `CE(-score+margin,y)+(λ/2)||w||²`, λ=1e−4다. 후보별 비정규화 softmax 질량을 기존 값의 `1+risk`배로 만드는 것과 같고, 원래 유효한 unsafe 후보도 경쟁자로 남는다. `risk=0`이면 이전 CE+ridge의 값·gradient·probability를 정확히 재현한다.

독립 scalar 구현으로 합성 risk·margin·target·189 map을 검산했다. 189개 중앙차분의 최대 gradient 오차는 7.18e-11, Hessian 오차는 8.56e-10였다. 별도 covariance Hessian과도 일치했으며 최소 고유값은 0.0001였다. 고정 margin에서 Hessian은 확률가중 특징 covariance+λI이므로 가중치에 대한 λ-strong convexity를 유지한다. 이는 수정한 목적함수의 수렴 인증 근거이며 T/R 성능 인증이 아니다.

전체 실패행은 `inf−inf`를 계산하지 않는다. 먼저0 배열을 만들고 valid 후보만 차분하며, NumPy 부동소수 예외를 raise로 둔 합성/실제 검산을 통과했다. invalid와 all-invalid risk/margin은0이지만 그 행을 안전한 예측으로 세지 않는다. target=-1인 행은 CE0으로 전체 분모에 남으며, ridge는 그대로다. 잘못된 anchor와 nonzero target margin은 거부한다.

기존 TRAIN guard 아래 허용된 cached TRAIN label NPZ만 사용해 **2,598행**을 재구성했다. 유효 anchor는2,597개, 기존 실패 `TEX__shard_04_f0110` 한 행은 유지했다. 네 모델의 risk/margin은 독립 scalar 계산과 bit 단위로 같고, 모델별 SHA를 JSON에 고정했다. 기존 SOURCE_FEASIBILITY의 target/safe/원본 error/valid SHA, 직전189 prefit의 context/raw feature SHA, 정규화·ID 순서·source index·고정 scale 연결이 모두 같다. 새로운 target 선택 규칙이나 후보/분모를 도입하지 않았다.

runtime score 함수의 API에는 margin이 없으며 context/normalization/target/Hessian helper AST가 직전 구현과 같다. runtime 점수는 같은 합성 입력·가중치의 이전 함수와 정확히 일치한다. 새 loss 식별자와 runtime margin 금지 검사를 추가했고, 수렴 인증은 변경한 margin 목적함수의 gradient를 같은 λ bound로 계산함을 별도 확인했다. 학습 가산항을 VAL이나 실사에서 참조값으로 다시 계산할 경로를 추가하지 않았다.

별도로 root가 원래 실사 계약의 SINGLE251 세 seed 비교를 matched R0_ONLY/pairedD 기준과 함께 복구하고 있다. 이는 실제 실사 실행 전 평가 범위를 원래 요구와 맞추는 작업이며 이번 TRAIN 손실 변경과 구분한다. 본 검산에서는 해당 실사 기준을 채점하거나 실사 참조를 읽지 않았다.

실제 TRAIN 배열에서는 고정 supervision SHA만 계산했다. 새 모델의 objective/정책 시험·오차 요약·oracle 비교는 하지 않았다. VAL 품질·실사 GT·원본 source 참조를 읽지 않았고 fit/optimizer/image forward/PnP는0회다. 이후 고정 protocol에 따른 학습과 기존 source45개 조건을 통과해야 별도의 실사 단계 자격을 평가할 수 있다.

[검산 JSON 및 모델별 SHA](PREFIT_REVIEW.json)
