# V175 — query_visibility_direct_query_owner_control_20261024_v175

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `6de40613b34af1168abe852940007841e6bb2223167b91808412801982ee24b9`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `6cf626b977dd4e724d88f14c6418fb56d2f1c940936cd1b37f7a5631fea2362c`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `8b8c98ccda698b840da01802f98c7ea98af269aeb25d7ae3203bea6bf249f342`

## 방법·목표의 원문 필드

`$.schema`

```text
V175_ROOT_SOURCE_OWNER_READOUT_KO_v1
```

`$.truth_and_metrics`

```text
Exact original-q point owner6; T=incident+other target, F=foreign, BG=background. Known soft mass CM and P/R; unknown ambiguity/INVALID/missing separately retained, predictionU column included in known-mass recall denominator. No hard GT cutoff or probability normalization.
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
    "missing_q_rows": 56
  },
  "TRAIN": {
    "rows": 4096,
    "known_mass": 4016.0,
    "predictionU_rows": 80,
    "ambiguous_mass": 0.0,
    "invalid_mass": 32.0,
    "missing_q_rows": 48
  },
  "VALIDATION": {
    "rows": 2048,
    "known_mass": 2040.0,
    "predictionU_rows": 8,
    "ambiguous_mass": 0.0,
    "invalid_mass": 0.0,
    "missing_q_rows": 8
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
  "actual_fit_runs": 1,
  "actual_saved_evaluation_runs": 1,
  "updates_each": 8000,
  "total_updates": 16000,
  "all2_frozen_before_outputs": true,
  "same_initial_state_sha": "80e2093d6a9d71a30c4cc9d976eb0347c96126cee6fb998642ca25a26e8dd073",
  "same_draw_sha": "6eae0c449469d39d67d6bcbceb3f965495ed50c1154553d5420bc2dfd6ac0411",
  "initializer_encoder_new_forward": 0,
  "prior_checkpoint_loads": 0,
  "private_current223_V169_V112_access": 0
}
```

`$.primary_validation`

```text
{
  "excerpt": "{\"C47_PUBLIC_WEAK253_ONLY\": {\"owner3\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[183.03291481919587, 438.17636695085093, 243.65611458430067, 0.0], [7.0, 36.753513623028994, 31.50109720416367, 0.0], [32.96708518080413, 309.0701194261201, 757.8427882115357, 0.0]], \"truth_known_mass\": [864.8653963543475, 75.25461082719266, 1099.8799928184599], \"expected_precision_on_known_truth_mass\": [0.820775402776663, 0.046879481662026774, 0.7336329024313026], \"expected_recall_including_predictionU\": [0.21163167770468264, 0.48838886041715335, 0.6890231599445241], \"prediction_counts\": [223, 784, 1033, 8], \"predictionU_rows\": 8, \"ambiguous_mass\": 0.0, \"invalid_mass\": 0.0, \"missing_q_inventory_unknown_rows\": 8, \"unknown_mass_by_prediction\": [0.0, 0.0, 0.0, 8.0]}, \"binary_target_presence\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[183.03291481919587, 681.8324815351516, 0.0], [39.96708518080413, 1135.1675184648484, 0.0]], \"truth_known_mass\": [864.8653963543475, 1175.1346036456525], \"expected_precision_on_known_truth_mass\": [0.820775402776663, 0.6247482214996414], \"expected_recall_including_predictionU\": [0.21163167770468264, 0.9659893555539824], \"prediction_counts\": [223, 1817, 8], \"predictionU_rows\": 8, \"ambiguous_mass\": 0.0, \"invalid_mass\": 0.0, \"missing_q_inventory_unknown_rows\": 8, \"unknown_mass_by_prediction\": [0.0, 0.0, 8.0]}}, \"C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC\": {\"owner3\": {\"rows\": 2048, \"known_mass\": 2040.0, \"expected_soft_CM\": [[726.7267075185664, 49.302076627500355, 88.83661220828071, 0.0], [2.0089762695133686, 72.26556253619492, 0.98007202148",
  "truncated": true
}
```

`$.groups_per_cohort`

```text
63
```

`$.public253_training_apparent_only`

```text
{
  "C47_PUBLIC_WEAK253_ONLY": {
    "completed": true,
    "weak_TRAIN_apparent_only": true,
    "rows": 253,
    "class_counts": [
      53,
      52,
      148
    ],
    "CM_T_F_BG": [
      [
        53,
        0,
        0
      ],
      [
        0,
        52,
        0
      ],
      [
        0,
        0,
        148
      ]
    ],
    "held_owner_metric": null,
    "no_native_or_quality_success_claim": true
  },
  "C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC": {
    "completed": true,
    "weak_TRAIN_apparent_only": true,
    "rows": 253,
    "class_counts": [
      53,
      52,
      148
    ],
    "CM_T_F_BG": [
      [
        53,
        0,
        0
      ],
      [
        0,
        52,
        0
      ],
      [
        0,
        0,
        148
      ]
    ],
    "held_owner_metric": null,
    "no_native_or_quality_success_claim": true
  }
}
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
동일조건 C47/C48 source control에서 직접 point-owner 감독 추가의 효과가 관찰됐다. C48의 source VAL T recall은84.03%이며 nonT recall88.66%, F precision53.53%의 남은 오류가 있다. 실사 전이·native 외부가림·전체 과제 해결이나 모델 채택을 확정하지 않는다.
```

`$.validation_history`

```text
Historically reused original public synthetic VALIDATION2048; not new blind/external data.
```

`$.remaining_limits`

```text
[
  "C48 nonT recall88.66% falls below C47 96.60% and v61 89.14%.",
  "C48 FOREIGN precision53.53% leaves substantial false-positive known mass despite FOREIGN recall96.03%.",
  "Class balancing changes loss costs; raw softmax values are not calibrated probabilities.",
  "Public253 assistant-weak centre2x2 observations differ in footprint/authority from synthetic exact-point raster mass.",
  "Source four variants/native slots share base geometry and are correlated.",
  "Point target/foreign/BG membership does not certify physical native identity, correct xy, native external/self visibility, quality or entity validity. Original q/L and D/E/S/U/source qualifications remain unchanged."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_direct_query_owner_control_20261024_v175/runs/C47_C48_DIRECT_OWNER_FIXED8000/COMPLETE.json](runs/C47_C48_DIRECT_OWNER_FIXED8000/COMPLETE.json)
- [query_visibility_direct_query_owner_control_20261024_v175/saved_direct_owner_evaluation_v1/COMPLETE.json](saved_direct_owner_evaluation_v1/COMPLETE.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_ACTUAL_PRIMARY_METRICS_AND_TRAINING_LOG_AUDIT.json](ROOT_ACTUAL_PRIMARY_METRICS_AND_TRAINING_LOG_AUDIT.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_ACTUAL_PLOT_AND_FIXED32_CROP_VISUAL_REVIEW.json](ROOT_ACTUAL_PLOT_AND_FIXED32_CROP_VISUAL_REVIEW.json)
- [query_visibility_direct_query_owner_control_20261024_v175/independent_source_fit_review_a/REVIEW.json](independent_source_fit_review_a/REVIEW.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_direct_query_owner_control_20261024_v175/saved_direct_owner_evaluation_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json](saved_direct_owner_evaluation_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json)
- [query_visibility_direct_query_owner_control_20261024_v175/saved_direct_owner_evaluation_v1/ALL6144_SAVED_DIRECT_OWNER_AND_EXACT_POINT_MASS.csv](saved_direct_owner_evaluation_v1/ALL6144_SAVED_DIRECT_OWNER_AND_EXACT_POINT_MASS.csv)
- [query_visibility_direct_query_owner_control_20261024_v175/fixed32_source_crop_review_v2/index.html](fixed32_source_crop_review_v2/index.html)
- [query_visibility_direct_query_owner_control_20261024_v175/fixed32_source_crop_review_v2/ACTUAL_VISUAL_QA.json](fixed32_source_crop_review_v2/ACTUAL_VISUAL_QA.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_ACTUAL_DIRECT_OWNER_FIT_TERMINAL.json](ROOT_ACTUAL_DIRECT_OWNER_FIT_TERMINAL.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_ACTUAL_SAVED_DIRECT_OWNER_EVALUATION_TERMINAL.json](ROOT_ACTUAL_SAVED_DIRECT_OWNER_EVALUATION_TERMINAL.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_SCOPE_BEFORE_DIRECT_OWNER_SOURCE_FIT.json](ROOT_SCOPE_BEFORE_DIRECT_OWNER_SOURCE_FIT.json)
- [query_visibility_direct_query_owner_control_20261024_v175/runs/C47_C48_DIRECT_OWNER_FIXED8000/ALL2_FROZEN_BEFORE_SAVED_OUTPUTS.json](runs/C47_C48_DIRECT_OWNER_FIXED8000/ALL2_FROZEN_BEFORE_SAVED_OUTPUTS.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_STATIC_DIRECT_OWNER_TRAINING_REVIEW.json](ROOT_STATIC_DIRECT_OWNER_TRAINING_REVIEW.json)
- [query_visibility_direct_query_owner_control_20261024_v175/ROOT_STATIC_SAVED_DIRECT_OWNER_EVALUATOR_REVIEW.json](ROOT_STATIC_SAVED_DIRECT_OWNER_EVALUATOR_REVIEW.json)
- [query_visibility_direct_query_owner_control_20261024_v175/fixed32_source_crop_review_v2/COMPLETE.json](fixed32_source_crop_review_v2/COMPLETE.json)
- [query_visibility_direct_query_owner_control_20261024_v175/scientific_readout_v1/PLOT_NUMBERS.json](scientific_readout_v1/PLOT_NUMBERS.json)
- [query_visibility_direct_query_owner_control_20261024_v175/report_preparation/write_root_readout.py](report_preparation/write_root_readout.py)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_owner_source_benchmark_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json](../query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_owner_source_benchmark_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_direct_query_owner_control_20261024_v175/scientific_readout_v1/OWNER_VALIDATION_COMPARISON.png](scientific_readout_v1/OWNER_VALIDATION_COMPARISON.png)

![원 보고서 진단 그림](scientific_readout_v1/OWNER_VALIDATION_COMPARISON.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
