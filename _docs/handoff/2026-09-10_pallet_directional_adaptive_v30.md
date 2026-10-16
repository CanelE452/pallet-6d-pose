# V30 점별 방향 보정과 이득 학습

사용자는 V29의 세 상황 예시는 핵심이 아니며 adaptive 점/선 활용이 핵심이라고 정정했고, 점이 좋으면 유지·선의 유효한 방향만 부분 보정하는 실행을 승인했다.

- 새 코드 `scripts/research/pallet_directional_adaptive_v30/`, 결과 `data/pallet/results/pallet_directional_adaptive_v30/`.
- 기존 완료 V25/V27/V28/V29 코드·결과·해시된 노트는 변경 금지.
- V12 원래 점과 V25 `vector_local_edge` 강한 점 복원기를 고정한다. 둘 다 정확히 유지 가능한 후보다. V29 guided/independent DHT도 고정한다.
- 상태 분류/condition 이름/가시성 GT는 새 forward와 손실에 들어가지 않는다.
-44경로: 유지2 + 두 점 부모 사이 full/x/y 부분 보정6 + 두 기준점에서2DHT×3incident역할×3mode 선 법선 부분 보정36.
-15%boxdiag 범위의 방향과 signed alpha[-1,1]를 학습하므로 선에 수직인 방향으로만 이동한다. 선에서 멀어지는 부호도 허용하며 이는 선에 붙이는 모델이 아니라 방향 유용성 모델이다. 법선 후보는 선택한 기준점의 접선 좌표를 보존한다.
- 금번4대조: point_gain / whole_line / normal_no_evidence / normal_gain. 모두41,630파라미터·동일초기값/배치. whole_line은 V29 전체 코너로 향하는4개 선 경로만 허용. normal_no_evidence는 선을 따라 읽는 영상/미분/coverage만 제거한다.
- 합성153원본은 V29 router sources를 그대로 사용한다. V29 DHT 미세학습 원본과 분리된다. V25강한점 부모는 이미 이 원본으로 학습했으므로 잔차 in-sample 한계는 그대로이며 전체 out-of-fold라고 주장하지 않는다.
- 각시드2000업데이트×4대조×3시드=24,000. batch16 FP32 AdamW3e-4 weightdecay1e-4 clip5. signedamount 정답투영 + 현재amount에서의 GT오차감소량 감독, groupbalanced Huber+rank. 최종 checkpoint만.
- GT oracle는 선택만 / amount까지 / 점후보만 / 부모2개만으로 분해해 실제출력과 분리한다.
- 평가계보는 사용자와 계속한 real297/9session reusedDEV, prospective128/synth_val505. 정본140/논문계보와 혼합하지 않으며 새 봉인평가 아님.
- 각시드 clean/crop10/20/30 V12대비1% 이내, corner20/crop40 V18이득90% 보존. normal_gain vs point_gain crop20 각시드1%+세션95%CI상한<0 동시에 필요.
- 완료까지 학습→GT없는추론→평가→합성/실사선택원인→수치/기하/부모재현감사→시각화MD/HTML→Chrome실제창확인→기존Discordhelper완료문장만. 민감한 훅 내용을 읽거나 데이터/이미지를 외부로 보내지 않는다.

상태 확인은 PROTOCOL/PRECHECK/TRAINING_COMPLETE/INFERENCE_COMPLETE/SUMMARY/AUDIT/COMPLETION 및driver.log를 읽는다. 진행 중 현재분기의 결과를 보고 손실/임계값을 바꾸지 않는다.
