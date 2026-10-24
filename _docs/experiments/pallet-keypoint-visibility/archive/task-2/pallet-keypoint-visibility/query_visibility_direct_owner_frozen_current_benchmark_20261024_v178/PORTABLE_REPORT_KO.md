# V178 — query_visibility_direct_owner_frozen_current_benchmark_20261024_v178

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `9c260d7b0fa5d0ad63f74d5432814637ef9bbe4316ad059f71ec8cdaceb5c1ce`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `853aadddff948d43be9c306fb472722eb35ddbdf41f0ce59808dde28e4cdb697`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `404d98b4cf878930adf4a75059f498dbcc312511e8bfc71b3f65af59f28e0b36`

## 방법·목표의 원문 필드

`$.schema`

```text
V178_ROOT_FROZEN_DIRECT_OWNER_CURRENT_READOUT_v1
```


## 입력·학습·평가 분모의 원문 필드

`$.original_inputU`

```text
45
```

`$.primary_cohort`

```text
paired_same_query_RGB_D_vs_E
```

`$.primary_N`

```text
1237
```

`$.primary_aggregate`

```text
{
  "excerpt": "[{\"model\": \"v61\", \"N\": 1237, \"D_N\": 1147, \"E_N\": 90, \"CM_rows_D_E_columns_D_E_U\": [[260, 887, 0], [2, 88, 0]], \"D_precision\": 0.9923664122137404, \"D_recall\": 0.22667829119442023, \"E_response_precision\": 0.09025641025641026, \"E_response_recall\": 0.9777777777777777, \"U_rate\": 0.0, \"near376_visible\": 85, \"near376_recall\": 0.22606382978723405}, {\"model\": \"legacy\", \"N\": 1237, \"D_N\": 1147, \"E_N\": 90, \"CM_rows_D_E_columns_D_E_U\": [[655, 319, 173], [3, 83, 4]], \"D_precision\": 0.9954407294832827, \"D_recall\": 0.5710549258936356, \"E_response_precision\": 0.2064676616915423, \"E_response_recall\": 0.9222222222222223, \"U_rate\": 0.14308811641067098, \"near376_visible\": 205, \"near376_recall\": 0.5452127659574468}, {\"model\": \"C47\", \"N\": 1237, \"D_N\": 1147, \"E_N\": 90, \"CM_rows_D_E_columns_D_E_U\": [[80, 1067, 0], [0, 90, 0]], \"D_precision\": 1.0, \"D_recall\": 0.06974716652136007, \"E_response_precision\": 0.07778738115816768, \"E_response_recall\": 1.0, \"U_rate\": 0.0, \"near376_visible\": 5, \"near376_recall\": 0.013297872340425532}, {\"model\": \"C48\", \"N\": 1237, \"D_N\": 1147, \"E_N\": 90, \"CM_rows_D_E_columns_D_E_U\": [[181, 966, 0], [1, 89, 0]], \"D_precision\": 0.9945054945054945, \"D_recall\": 0.15780296425457715, \"E_response_precision\": 0.08436018957345971, \"E_response_recall\": 0.9888888888888889, \"U_rate\": 0.0, \"near376_visible\": 35, \"near376_recall\": 0.09308510638297872}, {\"model\": \"C49\", \"N\": 1237, \"D_N\": 1147, \"E_N\": 90, \"CM_rows_D_E_columns_D_E_U\": [[193, 954, 0], [1, 89, 0]], \"D_precision\": 0.9948453608247423, \"D_recall\": 0.16826503923278116, \"E_response_precision\": 0.08533077660594439, \"E_response_recall\"",
  "truncated": true
}
```

`$.primary_material`

```text
{
  "excerpt": "[{\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"v61\", \"human_D_N\": 616, \"human_external_E_N\": 29, \"D_pred_D\": 145, \"D_pred_hidden\": 471, \"D_pred_U\": 0, \"E_pred_D\": 0, \"E_pred_hidden\": 29, \"E_pred_U\": 0, \"D_precision\": 1.0, \"D_recall\": 0.2353896103896104, \"external_E_response_precision\": 0.058, \"external_E_response_recall\": 1.0, \"unresolved_rate\": 0.0, \"hidden_response_not_learned_S\": true}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"legacy\", \"human_D_N\": 616, \"human_external_E_N\": 29, \"D_pred_D\": 448, \"D_pred_hidden\": 65, \"D_pred_U\": 103, \"E_pred_D\": 2, \"E_pred_hidden\": 23, \"E_pred_U\": 4, \"D_precision\": 0.9955555555555555, \"D_recall\": 0.7272727272727273, \"external_E_response_precision\": 0.26136363636363635, \"external_E_response_recall\": 0.7931034482758621, \"unresolved_rate\": 0.16589147286821707, \"hidden_response_not_learned_S\": true}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"C47\", \"human_D_N\": 616, \"human_external_E_N\": 29, \"D_pred_D\": 80, \"D_pred_hidden\": 536, \"D_pred_U\": 0, \"E_pred_D\": 0, \"E_pred_hidden\": 29, \"E_pred_U\": 0, \"D_precision\": 1.0, \"D_recall\": 0.12987012987012986, \"external_E_response_precision\": 0.05132743362831858, \"external_E_response_recall\": 1.0, \"unresolved_rate\": 0.0, \"hidden_response_not_learned_S\": true}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"C48\", \"human_D_N\": 616, \"human_external_E_N\": 29, \"D_pred_D\": 155, \"D_pred_hidden\": 461, \"D_pred_U\": 0, \"E_pred_D\": 1, \"E_pred_hidden\": 28, \"E_pred_U\": 0, \"D_precision\": 0.993589743589743",
  "truncated": true
}
```

`$.current_data_used_for_training_or_calibration_or_selection`

```text
false
```

`$.new_training_updates`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.conclusion_KO`

```text
C47/C48/C49의 동결 사용자 평가를 완료했다. C48/C49는 기존 보임 주석 유지율이15.8%/16.8%로 낮아 v61/legacy 대비 실사용 개선을 입증하지 못했다. 목표 차이와 남은 인식 실패가 함께 관찰되며 모델 채택과 전체 해결은 하지 않는다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json](frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/saved_five_model_evaluation_v1/EVALUATION_SEAL.json](saved_five_model_evaluation_v1/EVALUATION_SEAL.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/independent_actual_saved_evaluation_review_a/REVIEW.json](independent_actual_saved_evaluation_review_a/REVIEW.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/final_report_review_a/REVIEW.json](final_report_review_a/REVIEW.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1/ACTUAL_VISUAL_QA.json](fixed19_current_crop_review_v1/ACTUAL_VISUAL_QA.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/ROOT_FINAL_REPORT_REVIEW.json](ROOT_FINAL_REPORT_REVIEW.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/saved_five_model_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json](saved_five_model_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/saved_five_model_evaluation_v1/ALL1784_NATIVE_LEDGER.csv](saved_five_model_evaluation_v1/ALL1784_NATIVE_LEDGER.csv.parts.json)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1/index.html](fixed19_current_crop_review_v1/index.html)
- [query_visibility_public_held_blind_center_owner_review_20261024_v177/ROOT_READOUT_KO.html](../query_visibility_public_held_blind_center_owner_review_20261024_v177/ROOT_READOUT_KO.html)
- [query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/ROOT_ACTUAL_PLOT_AND_FIXED19_VISUAL_REVIEW.json](ROOT_ACTUAL_PLOT_AND_FIXED19_VISUAL_REVIEW.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/scientific_readout_v1/CURRENT223_DIRECT_OWNER_DIAGNOSTIC_V2.png](scientific_readout_v1/CURRENT223_DIRECT_OWNER_DIAGNOSTIC_V2.png)

![원 보고서 진단 그림](scientific_readout_v1/CURRENT223_DIRECT_OWNER_DIAGNOSTIC_V2.png)

[query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1/CASE_0256_CARD.png](fixed19_current_crop_review_v1/CASE_0256_CARD.png)

![원 보고서 진단 그림](fixed19_current_crop_review_v1/CASE_0256_CARD.png)

[query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1/CASE_0840_CARD.png](fixed19_current_crop_review_v1/CASE_0840_CARD.png)

![원 보고서 진단 그림](fixed19_current_crop_review_v1/CASE_0840_CARD.png)

[query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/fixed19_current_crop_review_v1/CASE_0401_CARD.png](fixed19_current_crop_review_v1/CASE_0401_CARD.png)

![원 보고서 진단 그림](fixed19_current_crop_review_v1/CASE_0401_CARD.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
