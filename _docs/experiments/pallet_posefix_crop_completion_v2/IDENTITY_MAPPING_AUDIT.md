# Identity 계약 감사

Source normalized native9 label→cache→prepared/crop affine 순서를 유지. 실사 frozen target/pnp hidden index 순서 그대로 재사용. 모든 center8 pass-through, crop실험 channel swap없음. evaluator는 whole-object branch 하나만 사용하고 canonical[idx]=native error 역대응. 기전진단은 과거FP branch로 고정.

구현상mapping불일치 발견없음. 다른채널 candidate는 가능한전체순열과 실제모델전체branch 일치로 LEGIT_SYMMETRY_EQUIVALENT / CHANNEL_CONFUSION / UNKNOWN 분류. 호환성은 물리정체검증이나 자동swap 근거가 아님. source/real label의 물리적 정의까지 새로 독립검증한것은 아니며 REVIEW_PENDING 유지.
