# V176 — query_visibility_direct_owner_public_ycb_control_20261024_v176

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `931f20784cf7db9797e044227d584e9729715b24ceab07f11ce32f7506c5dfdd`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `6917d6c16db6959a8ed5d65d0c7f1e5b5f02537db47808f2f96354c83eb81aeb`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `5dee59416106d6ea8d60278f82bf0af2fa6b27dfd5af369d1d36dd464153dc49`

## 방법·목표의 원문 필드

`$.schema`

```text
V176_ROOT_C49_SOURCE_OWNER_READOUT_KO_v1
```

`$.phase_complete_scope`

```text
C49 actual fixed8000 fit, final freeze, saved source evaluation, root main-metric/log/reference audit, fixed32 source comparison with B actual32/root actual16 image QA, and Korean report; full goal remains unsolved.
```

`$.truth_and_metrics`

```text
Original-q point-owner soft mass: T=incident+other target, F=foreign, BG=background. Known-mass CM and precision; recall includes known truth predicted U. Ambiguous/INVALID/missing unknown mass remains separate. No hard GT cutoff or renormalization.
```


## 입력·학습·평가 분모의 원문 필드

`$.primary`

```text
FULL_INVENTORY/VALIDATION
```

`$.cohort_counts`

```text
{
  "ALL": {
    "rows": 6144,
    "known_mass": 6056.0,
    "predictionU_rows": 88,
    "ambiguous_mass": 0.0,
    "invalid_mass": 32.0,
    "missing_q_inventory_unknown_rows": 56
  },
  "TRAIN": {
    "rows": 4096,
    "known_mass": 4016.0,
    "predictionU_rows": 80,
    "ambiguous_mass": 0.0,
    "invalid_mass": 32.0,
    "missing_q_inventory_unknown_rows": 48
  },
  "VALIDATION": {
    "rows": 2048,
    "known_mass": 2040.0,
    "predictionU_rows": 8,
    "ambiguous_mass": 0.0,
    "invalid_mass": 0.0,
    "missing_q_inventory_unknown_rows": 8
  }
}
```

`$.paired_rows6056`

```text
6056
```

`$.original_rows6144_preserved`

```text
true
```

`$.original_inputU88_preserved`

```text
true
```

`$.execution`

```text
{
  "actual_new_heads_trained": 1,
  "actual_fit_runs": 1,
  "updates": 8000,
  "fit_seconds": 39.21434958593454,
  "actual_saved_evaluation_runs": 1,
  "fixed_final_frozen_before_outputs": true,
  "same_C48_initial_and_original_draws": true,
  "old_checkpoint_loads": 0,
  "training_VALIDATION_soft_read_or_hash_calls": 0,
  "new_initializer_encoder_forward": 0,
  "new_current223_private_V169_V112_reads": 0,
  "PnP": 0
}
```

`$.primary_validation`

```text
{
  "excerpt": "{\"C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314\": {\"owner3\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[705.1387997688726, 73.73103980533779, 85.99555678013712, 0.0], [2.0089762695133686, 72.26556253619492, 0.980072021484375, 0.0], [122.85222396161407, 17.003397658467293, 960.0243711983785, 0.0]], \"truth_known_mass\": [864.8653963543475, 75.25461082719266, 1099.8799928184599], \"expected_precision_on_known_truth_mass\": [0.8495648189986417, 0.4433470094245087, 0.9169287212974007], \"expected_recall_including_predictionU\": [0.8153162361926286, 0.9602808617552816, 0.8728446534774225], \"prediction_counts\": [830, 163, 1047, 8], \"predictionU_rows\": 8, \"ambiguous_mass\": 0.0, \"invalid_mass\": 0.0, \"missing_q_inventory_unknown_rows\": 8, \"unknown_mass_by_prediction\": [0.0, 0.0, 0.0, 8.0]}, \"binary_target_presence\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[705.1387997688726, 159.7265965854749, 0.0], [124.86120023112744, 1050.273403414525, 0.0]], \"truth_known_mass\": [864.8653963543475, 1175.1346036456525], \"expected_precision_on_known_truth_mass\": [0.8495648189986417, 0.867994548276467], \"expected_recall_including_predictionU\": [0.8153162361926286, 0.8937473206526579], \"prediction_counts\": [830, 1210, 8], \"predictionU_rows\": 8, \"ambiguous_mass\": 0.0, \"invalid_mass\": 0.0, \"missing_q_inventory_unknown_rows\": 8, \"unknown_mass_by_prediction\": [0.0, 0.0, 8.0]}}, \"C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC\": {\"owner3\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[726.7267075185664, 49.302076627500355, 88.83661220828071, 0.0], [2.0089762695133686, 72.2",
  "truncated": true
}
```

`$.groups_per_cohort`

```text
63
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
public YCB CF314를 추가한 C49는 C48보다 source VAL T recall과 FOREIGN precision이 낮았다. nonT recall은 소폭 높고 FOREIGN recall은 같아, 이 추가가 source 진단을 전반 개선했다는 증거는 없다. 모델 채택과 전체 과제 해결은 하지 않는다.
```

`$.validation_history`

```text
Historically reused original public synthetic VALIDATION2048; not a new blind split and not natural-real transfer.
```

`$.remaining_limits`

```text
[
  "C49 source VAL T recall81.53% is lower than C48 84.03%; FOREIGN precision44.33% is lower than53.53%, with FOREIGN recall96.03% unchanged.",
  "C49 nonT recall89.37% is0.72pp above C48; this tradeoff does not establish overall improvement.",
  "public253 and CF314 diagonal TRAIN apparent results reflect exposed training supervision, not held natural-real transfer.",
  "CF COVER central overwrite and assistant-weak SHAM evidence are controlled compositions, not physical native external/self visibility or correct-coordinate gold.",
  "Soft boundary mass and class-balanced losses do not produce calibrated probabilities; related variants/native slots share source base geometry.",
  "Physical initializer native identity is not certified by literal owner supervision; original q/L/native labels and qualifications remain unchanged."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_direct_owner_public_ycb_control_20261024_v176/runs/C49_PUBLIC_YCB_CF314_FIXED8000/COMPLETE.json](runs/C49_PUBLIC_YCB_CF314_FIXED8000/COMPLETE.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/saved_C49_owner_evaluation_v1/COMPLETE.json](saved_C49_owner_evaluation_v1/COMPLETE.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_ACTUAL_PRIMARY_METRICS_LOGS_AND_REFERENCE_AUDIT.json](ROOT_ACTUAL_PRIMARY_METRICS_LOGS_AND_REFERENCE_AUDIT.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/independent_source_fit_review_b/REVIEW.json](independent_source_fit_review_b/REVIEW.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/fixed32_source_crop_review_v1/ACTUAL_VISUAL_QA.json](fixed32_source_crop_review_v1/ACTUAL_VISUAL_QA.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_ACTUAL_C49_FIXED_SOURCE_CROP_VISUAL_QA.json](ROOT_ACTUAL_C49_FIXED_SOURCE_CROP_VISUAL_QA.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/saved_C49_owner_evaluation_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json](saved_C49_owner_evaluation_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/saved_C49_owner_evaluation_v1/ALL6144_C49_OWNER_AND_EXACT_POINT_MASS.csv](saved_C49_owner_evaluation_v1/ALL6144_C49_OWNER_AND_EXACT_POINT_MASS.csv)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_READOUT_KO.html](../query_visibility_direct_query_owner_control_20261024_v175/ROOT_READOUT_KO.html)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/fixed32_source_crop_review_v1/index.html](fixed32_source_crop_review_v1/index.html)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_SCOPE_BEFORE_PUBLIC_YCB_FEATURE_OR_NEW_C49_SCORE_BODIES.json](ROOT_SCOPE_BEFORE_PUBLIC_YCB_FEATURE_OR_NEW_C49_SCORE_BODIES.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_ACTUAL_C49_FIT_TERMINAL.json](ROOT_ACTUAL_C49_FIT_TERMINAL.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_ACTUAL_SAVED_C49_EVALUATION_TERMINAL.json](ROOT_ACTUAL_SAVED_C49_EVALUATION_TERMINAL.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/runs/C49_PUBLIC_YCB_CF314_FIXED8000/C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314/MR1840_FINAL_APPARENT_ONLY.json](runs/C49_PUBLIC_YCB_CF314_FIXED8000/C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314/MR1840_FINAL_APPARENT_ONLY.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/runs/C49_PUBLIC_YCB_CF314_FIXED8000/C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314/CF3816_FINAL_APPARENT_ONLY.json](runs/C49_PUBLIC_YCB_CF314_FIXED8000/C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314/CF3816_FINAL_APPARENT_ONLY.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/runs/C49_PUBLIC_YCB_CF314_FIXED8000/FROZEN_BEFORE_SAVED_OUTPUTS.json](runs/C49_PUBLIC_YCB_CF314_FIXED8000/FROZEN_BEFORE_SAVED_OUTPUTS.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_STATIC_C49_SOURCE_TRAINING_REVIEW.json](ROOT_STATIC_C49_SOURCE_TRAINING_REVIEW.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/ROOT_STATIC_C49_SAVED_EVALUATOR_REVIEW.json](ROOT_STATIC_C49_SAVED_EVALUATOR_REVIEW.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_RESULTS_KO.json](../query_visibility_direct_query_owner_control_20261024_v175/ROOT_RESULTS_KO.json)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/report_preparation/write_root_readout.py](report_preparation/write_root_readout.py)
- [query_visibility_direct_owner_public_ycb_control_20261024_v176/fixed32_source_crop_review_v1/COMPLETE.json](fixed32_source_crop_review_v1/COMPLETE.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
