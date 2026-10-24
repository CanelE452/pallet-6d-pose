# V169 — query_visibility_frozen_public_threshold_current_benchmark_20261024_v169

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `3015f54fb2c0d6760f86f9ee90c6e2d9c011ce62efbdae8ac7198285ce2e4fbb`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `86316ee8a8d1cfe3f60656dd0f1733f0a53bf90718cbe163ada19c0de8f8159d`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `08bbcb2d20bb323cda800dc601150ca8f4d444fdab57c3067f11fcc14af356c7`

## 방법·목표의 원문 필드

`$.schema`

```text
V169_FROZEN_PUBLIC_THRESHOLD_CURRENT_TRANSFER_BENCHMARK_UNSOLVED
```

`$.phase_complete_scope`

```text
frozen diagnostic prediction and saved evaluation plus direct row audit; crop final QA publication remains separate
```


## 입력·학습·평가 분모의 원문 필드

`$.cohorts`

```text
{
  "frames": 223,
  "all_original_slots": 1784,
  "valid": 1739,
  "original_inputU": 45,
  "additional_U": 0,
  "primary_common_paired": 1237,
  "primary_D": 1147,
  "primary_E": 90,
  "secondary_same_slot": 1248,
  "WOOD_frames": 105,
  "SQUARE_frames": 118,
  "primary_new_four_U": 0,
  "manual_reference_near_D_proxy": 376,
  "near_proxy_WOOD": 153,
  "near_proxy_SQUARE": 223
}
```

`$.primary_tables`

```text
{
  "excerpt": "[{\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"v61\", \"CM_rows_D_E_columns_D_E_U\": [[260, 887, 0], [2, 88, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9923664122137404, \"D_recall\": 0.22667829119442023, \"E_precision\": 0.09025641025641026, \"E_recall\": 0.9777777777777777, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"legacy\", \"CM_rows_D_E_columns_D_E_U\": [[655, 319, 173], [3, 83, 4]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9954407294832827, \"D_recall\": 0.5710549258936356, \"E_precision\": 0.2064676616915423, \"E_recall\": 0.9222222222222223, \"U\": 177, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C16\", \"CM_rows_D_E_columns_D_E_U\": [[1137, 10, 0], [55, 35, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9538590604026845, \"D_recall\": 0.99128160418483, \"E_precision\": 0.7777777777777778, \"E_recall\": 0.3888888888888889, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C45_ORIGINAL0\", \"CM_rows_D_E_columns_D_E_U\": [[995, 152, 0], [35, 55, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9660194174757282, \"D_recall\": 0.8674803836094158, \"E_precision\": 0.26570048309178745, \"E_recall\": 0.6111111111111112, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C45_PUBLICCAL_FULL\", \"CM_rows_D_E_columns_D_E_U\": [[359, 788, 0], [2, 88, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9944598337950139, \"D_recall\": 0.3129904097646033, \"E_precision\": 0.1004566210045662, \"E_recall\": 0.9777777777777777, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\":",
  "truncated": true
}
```

`$.all_primary_and_secondary_tables`

```text
{
  "excerpt": "[{\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"v61\", \"CM_rows_D_E_columns_D_E_U\": [[260, 887, 0], [2, 88, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9923664122137404, \"D_recall\": 0.22667829119442023, \"E_precision\": 0.09025641025641026, \"E_recall\": 0.9777777777777777, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"legacy\", \"CM_rows_D_E_columns_D_E_U\": [[655, 319, 173], [3, 83, 4]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9954407294832827, \"D_recall\": 0.5710549258936356, \"E_precision\": 0.2064676616915423, \"E_recall\": 0.9222222222222223, \"U\": 177, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C16\", \"CM_rows_D_E_columns_D_E_U\": [[1137, 10, 0], [55, 35, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9538590604026845, \"D_recall\": 0.99128160418483, \"E_precision\": 0.7777777777777778, \"E_recall\": 0.3888888888888889, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C45_ORIGINAL0\", \"CM_rows_D_E_columns_D_E_U\": [[995, 152, 0], [35, 55, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9660194174757282, \"D_recall\": 0.8674803836094158, \"E_precision\": 0.26570048309178745, \"E_recall\": 0.6111111111111112, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"ALL\", \"model\": \"C45_PUBLICCAL_FULL\", \"CM_rows_D_E_columns_D_E_U\": [[359, 788, 0], [2, 88, 0]], \"D_N\": 1147, \"E_N\": 90, \"D_precision\": 0.9944598337950139, \"D_recall\": 0.3129904097646033, \"E_precision\": 0.1004566210045662, \"E_recall\": 0.9777777777777777, \"U\": 0, \"N\": 1237}, {\"cohort\": \"paired_same_query_RGB\", \"group\":",
  "truncated": true
}
```

`$.execution`

```text
{
  "frozen_current_prediction_executions": 1,
  "C16_repeat_calls": 2,
  "candidate_two_heads_total_calls": 4,
  "new_head_training": 0,
  "new_threshold_fit_or_choice": 0,
  "saved_evaluation_attempts": 2,
  "first_attempt": "Evaluation-tool-only NameError: require missing from transplanted transitions namespace; outputs/code/protocol/entry/review preserved before minimal repair.",
  "repair": "Added require namespace plus actual transplanted192-panel invented regression; repaired root entry then saved evaluation exit0.",
  "model_inference_repeated_due_to_eval_failure": false
}
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
공개 한 영상에서 맞춘 full CAL 문턱은 current 전이에서 보임 D를 크게 잃었다. 고정0 C46도 C45보다 E는 늘지만 D가 줄었으며 C16과 공동 개선이나 전체 과제 해결을 입증하지 못했다. 모델은 채택하지 않는다.
```

`$.scope_and_limits_KO`

```text
[
  "current223은 평가 전용이며 training/threshold/normalizer/checkpoint/architecture/arm/model 선택에 사용하지 않았다. 공개에서 동결한 두 head×두문턱 네조건을 모두 기록했다.",
  "과거 legacy CAL46 겹침과 current223 반복 개발 노출이 있다. fresh blind holdout이나 독립 장면 일반화로 주장하지 않는다.",
  "SELF는 기존 사람S 부분집합에서의 이진 hidden 반응으로 별도 보고한다. 학습된 external 점수로 자기 가림 원인을 식별했다고 주장하지 않으며 기존 geometry gate와 구분한다.",
  "<=3px376은 기존 manual-reference의 고정 proxy 부분집합이다. physical native/좌표 gold가 아니며 primary 가시성 분모를 줄이는 gate나 새 학습 정답으로 사용하지 않았다.",
  "원래 mapping/semantic unknown/out-of-frame/S 및 모든U와 source qualification 실패를 유지했다. 원 q/L/xy와 crop native 순서는 바꾸지 않았으며 PnP는 실행하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/frozen_two_head_four_condition_predictions_v1/PREDICTION_SEAL.json](frozen_two_head_four_condition_predictions_v1/PREDICTION_SEAL.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/EVALUATION_SEAL.json](saved_seven_condition_evaluation_v1/EVALUATION_SEAL.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_row_audit/COMPLETE.json](saved_row_audit/COMPLETE.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/fixed_current_crop_review_v1/caption_aligned_v2/CAPTION_ALIGNMENT_COMPLETE.json](fixed_current_crop_review_v1/caption_aligned_v2/CAPTION_ALIGNMENT_COMPLETE.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/crop_visual_review_a/ACTUAL_VISUAL_QA_COMPLETE.json](crop_visual_review_a/ACTUAL_VISUAL_QA_COMPLETE.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/fixed_current_crop_review_v1/caption_aligned_v2/index.html](fixed_current_crop_review_v1/caption_aligned_v2/index.html)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/fixed_current_crop_review_v1/index.html](fixed_current_crop_review_v1/index.html)
- [query_visibility_frozen_threshold_full_source_20261024_v168/ROOT_READOUT_KO.html](../query_visibility_frozen_threshold_full_source_20261024_v168/ROOT_READOUT_KO.html)
- [query_visibility_source_threshold_feasibility_20261024_v170/SUMMARY_KO.json](../query_visibility_source_threshold_feasibility_20261024_v170/SUMMARY_KO.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/CONFUSION.csv](saved_seven_condition_evaluation_v1/CONFUSION.csv)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/HUMAN_SELF_BINARY_RESPONSE.csv](saved_seven_condition_evaluation_v1/HUMAN_SELF_BINARY_RESPONSE.csv)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/MANUAL_NEAR_D.csv](saved_seven_condition_evaluation_v1/MANUAL_NEAR_D.csv)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/COORDINATE_ERROR_BINS.csv](saved_seven_condition_evaluation_v1/COORDINATE_ERROR_BINS.csv)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/PAIRED_RECOVERY_AND_DAMAGE.csv](saved_seven_condition_evaluation_v1/PAIRED_RECOVERY_AND_DAMAGE.csv)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/ALL1784_NATIVE_LEDGER.csv](saved_seven_condition_evaluation_v1/ALL1784_NATIVE_LEDGER.csv.parts.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/provenance/FIRST_ACTUAL_SAVED_EVAL_MISSING_REQUIRE_NAMESPACE/PRESERVATION_MANIFEST.json](provenance/FIRST_ACTUAL_SAVED_EVAL_MISSING_REQUIRE_NAMESPACE/PRESERVATION_MANIFEST.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/ROOT_REQUIREMENTS_AND_DIAGNOSTIC_POLICY.json](ROOT_REQUIREMENTS_AND_DIAGNOSTIC_POLICY.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/frozen_two_head_four_condition_predictions_v1/C16_ORIGINAL_ALL1784_BYTE_PARITY_SEAL.json](frozen_two_head_four_condition_predictions_v1/C16_ORIGINAL_ALL1784_BYTE_PARITY_SEAL.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json](saved_seven_condition_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_row_audit/audit_rows_without_evaluator_helpers.py](saved_row_audit/audit_rows_without_evaluator_helpers.py)
- [query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/saved_seven_condition_evaluation_v1/METRICS.csv](saved_seven_condition_evaluation_v1/METRICS.csv)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
