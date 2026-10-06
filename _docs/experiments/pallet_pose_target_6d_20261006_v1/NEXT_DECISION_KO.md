# 다음 판단

[확인] 이번 판정은 POSE_TARGET_NOT_SUPPORTED야. source TRAIN final ADDsym hard target만 바꾼 고정 실험이고, 다음 실험은 실행하지 않아.

[확인] matched target 교체는 soft2D→hard6D 전체야. 비용 정렬과 label hardness 각각의 기여는 별도로 식별되지 않았어.

[확인] seed1 synthetic screen: STOP — all three paired pose means worsen and RAW canonical good5-to-bad10 damage increases REAL_DEV는 screen 조건에 포함되지 않았어.

[추정·미검증] 원인 후보는 서로 구분해야 해. 기존 pre-update forward에서 관측한 모델의 TRAIN action 적중률이 낮으면 후보 간 구별/표현·최적화의 가능성이 남고, 그 관측 적중률이 높지만 synthetic oracle 회수가 낮으면 일반화 가능성이 남아. 아래 모델 관측과 별개인 static 2D/6D teacher index 일치율 21.2358%를 모델 적중률로 해석하지 않아. synthetic 개선과 REAL만의 악화는 synthetic-to-real transfer 가능성과도 맞지만 독립 실사 참조가 없어 원인을 확정하지 않아. 6D 개선과 canonical 2D 손상이 함께 늘면 preservation 목적의 충돌 가능성이 있어. 현재 관찰만으로 모델 크기·최적화·특징 부재를 원인으로 단정하지 않아.

[확인] TRAIN target/기존2Dteacher의 index 일치·NoOp/이동 구분은 TRAIN_2D_6D_TARGET_COMPARISON의 실제 요약에 있고, 이번 loss·checkpoint를 선택하는 데 쓰지 않았어. TRAIN 새로운 추가 점수 추론이 없으면 학습 전후 exact-match의 전 모집단 평가를 완료했다고 말하지 않아.

[추정·미검증] 후속 후보는 이 원행을 바탕으로 사전 고정할 수 있지만 현재 실행하지 않아. 새 backbone·cap·LR·temperature·seed·candidate 탐색은 이번 결과에서 정당화되지 않았어.

[확인] 기존 학습 forward에서 optimizer update 직전에 관측한 action 적중/CE야. 누적과 마지막100 update(5901–6000)의 eligible/제외·이동/NoOp exposure 분모를 구분해 아래 실제 receipt 값을 옮겼어. 추가 model forward/optimizer update는0이야. checkpoint가 변하며 같은 TRAIN 행을 반복 노출한 관측이고 최종 checkpoint의 고유 TRAIN 전체 정확도가 아니야.

```json
[
  {
    "seed": 1,
    "cumulative": {
      "observed_exposures": 96000,
      "eligible_target_exposures": 96000,
      "excluded_target_exposures": 0,
      "matching_eligible_exposures": 6245,
      "moving_target_exposures": 90589,
      "matching_moving_target_exposures": 1194,
      "NoOp_target_exposures": 5411,
      "matching_NoOp_target_exposures": 5051,
      "eligible_CE_sum": 487603.3127593994,
      "eligible_action_accuracy": 0.06505208333333333,
      "moving_target_action_accuracy": 0.01318040821733323,
      "NoOp_target_action_accuracy": 0.9334688597301792,
      "eligible_mean_CE": 5.0792011745770775,
      "scope": "existing pre-update training forwards; ordered repeated exposures under changing checkpoints; not final whole-TRAIN accuracy or independent underfitting evidence",
      "additional_model_forwards": 0,
      "additional_optimizer_updates": 0
    },
    "last_window": {
      "first_step": 5901,
      "last_step": 6000,
      "observed_exposures": 1600,
      "eligible_target_exposures": 1600,
      "excluded_target_exposures": 0,
      "matching_eligible_exposures": 92,
      "moving_target_exposures": 1522,
      "matching_moving_target_exposures": 23,
      "NoOp_target_exposures": 78,
      "matching_NoOp_target_exposures": 69,
      "eligible_CE_sum": 8014.591384887695,
      "eligible_action_accuracy": 0.0575,
      "moving_target_action_accuracy": 0.015111695137976347,
      "NoOp_target_action_accuracy": 0.8846153846153846,
      "eligible_mean_CE": 5.00911961555481,
      "scope": "existing pre-update training forwards; ordered repeated exposures under changing checkpoints; not final whole-TRAIN accuracy or independent underfitting evidence",
      "additional_model_forwards": 0,
      "additional_optimizer_updates": 0
    }
  }
]
```

[추정·미검증] 이 관측 pre-update exposure 적중률과 heldout oracle-action/gap 회수는 범위가 달라. 관측 적중이 낮으면 표현·최적화 가능성, 관측 적중이 높고 heldout 회수가 낮으면 일반화 가능성이 남지만, 해당 수치만으로 underfitting 또는 어느 원인을 확정하지 않아.
