# V118 — query_visibility_same_material_query_head_20261023_v118

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `9a243fb31441fa6e6700162e4258820dc1747d2f932d7a792b95fc02c5dd3a45`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `3c5c074606ba7ad30fda7be4cb7cdeace8e72cae34085c3f90c47cf6b990b573`

## 방법·목표의 원문 필드

자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.

## 입력·학습·평가 분모의 원문 필드

`$.current_primary_comparison`

```text
{
  "excerpt": "[{\"group\": \"WOOD_PRIMARY105\", \"model\": \"v61\", \"display_model\": \"v61\", \"human_D_N\": 616, \"human_E_N\": 29, \"confusion_rows_D_E_cols_D_hidden_U\": [[145, 471, 0], [0, 29, 0]], \"D_precision\": 1.0, \"D_recall\": 0.2353896103896104, \"E_precision\": 0.058, \"E_recall\": 1.0, \"U_rate\": 0.0}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"legacy\", \"display_model\": \"기존 실사 모델\", \"human_D_N\": 616, \"human_E_N\": 29, \"confusion_rows_D_E_cols_D_hidden_U\": [[448, 65, 103], [2, 23, 4]], \"D_precision\": 0.9955555555555555, \"D_recall\": 0.7272727272727273, \"E_precision\": 0.26136363636363635, \"E_recall\": 0.7931034482758621, \"U_rate\": 0.16589147286821707}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V111_C11_RAW\", \"display_model\": \"C11\", \"human_D_N\": 616, \"human_E_N\": 29, \"confusion_rows_D_E_cols_D_hidden_U\": [[598, 18, 0], [11, 18, 0]], \"D_precision\": 0.9819376026272578, \"D_recall\": 0.9707792207792207, \"E_precision\": 0.5, \"E_recall\": 0.6206896551724138, \"U_rate\": 0.0}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V114_C12_RAW\", \"display_model\": \"C12\", \"human_D_N\": 616, \"human_E_N\": 29, \"confusion_rows_D_E_cols_D_hidden_U\": [[601, 15, 0], [21, 8, 0]], \"D_precision\": 0.9662379421221865, \"D_recall\": 0.9756493506493507, \"E_precision\": 0.34782608695652173, \"E_recall\": 0.27586206896551724, \"U_rate\": 0.0}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V118_C13_RAW\", \"display_model\": \"C13\", \"human_D_N\": 616, \"human_E_N\": 29, \"confusion_rows_D_E_cols_D_hidden_U\": [[599, 17, 0], [21, 8, 0]], \"D_precision\": 0.9661290322580646, \"D_recall\": 0.9724025974025974, \"E_precision\": 0.32, \"E_recall\": 0.27586206896551724, \"U_rate\": 0.0}, {\"group\": \"S",
  "truncated": true
}
```

`$.manual_near_D_rows`

```text
{
  "excerpt": "[{\"group\": \"WOOD_PRIMARY105\", \"model\": \"v61\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"42\", \"pred_binary_hidden\": \"111\", \"unresolved\": \"0\", \"semantic_D_recall\": \"0.27450980392156865\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"legacy\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"116\", \"pred_binary_hidden\": \"11\", \"unresolved\": \"26\", \"semantic_D_recall\": \"0.7581699346405228\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V91_RAW\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"145\", \"pred_binary_hidden\": \"8\", \"unresolved\": \"0\", \"semantic_D_recall\": \"0.9477124183006536\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V91_PLUS_OLD_GEOMETRY\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"145\", \"pred_binary_hidden\": \"8\", \"unresolved\": \"0\", \"semantic_D_recall\": \"0.9477124183006536\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V111_C11_RAW\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"150\", \"pred_binary_hidden\": \"3\", \"unresolved\": \"0\", \"semantic_D_recall\": \"0.9803921568627451\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V111_C11_PLUS_OLD_GEOMETRY\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"150\", \"pred_binary_hidden\": \"3\", \"unresolved\": \"0\", \"semantic_D_recall\": \"0.9803921568627451\"}, {\"group\": \"WOOD_PRIMARY105\", \"model\": \"V118_C13_RAW\", \"scope\": \"MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3\", \"N\": \"153\", \"pred_visible\": \"153\", \"pred_binary_hidden\": \"0\", \"unresolved\": \"0\", \"sema",
  "truncated": true
}
```

`$.self_binary_response_rows`

```text
{
  "excerpt": "[{\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"v61\", \"human_S_support\": \"21\", \"binary_hidden_response\": \"12\", \"visible_response\": \"9\", \"unresolved\": \"0\", \"human_S_binary_hidden_response_rate\": \"0.5714285714285714\", \"learned_self_prediction\": \"False\", \"external_vs_self_identified\": \"False\", \"scope\": \"HumanS subset apparentvisibility binary response; oldgeometry route separately named. ZeroS support is no gold evidence.\"}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"legacy\", \"human_S_support\": \"21\", \"binary_hidden_response\": \"3\", \"visible_response\": \"13\", \"unresolved\": \"5\", \"human_S_binary_hidden_response_rate\": \"0.14285714285714285\", \"learned_self_prediction\": \"False\", \"external_vs_self_identified\": \"False\", \"scope\": \"HumanS subset apparentvisibility binary response; oldgeometry route separately named. ZeroS support is no gold evidence.\"}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"V91_RAW\", \"human_S_support\": \"21\", \"binary_hidden_response\": \"2\", \"visible_response\": \"19\", \"unresolved\": \"0\", \"human_S_binary_hidden_response_rate\": \"0.09523809523809523\", \"learned_self_prediction\": \"False\", \"external_vs_self_identified\": \"False\", \"scope\": \"HumanS subset apparentvisibility binary response; oldgeometry route separately named. ZeroS support is no gold evidence.\"}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"model\": \"V91_PLUS_OLD_GEOMETRY\", \"human_S_support\": \"21\", \"binary_hidden_response\": \"2\", \"visible_response\": \"19\", \"unresolved\": \"0\", \"human_S_binary_hidden_response_rat",
  "truncated": true
}
```

`$.coordinate_error_rows`

```text
{
  "excerpt": "[{\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"task\": \"D_vs_E\", \"error_bin\": \"LE3\", \"model\": \"v61\", \"manual_click_reference_N\": \"153\", \"other_or_unspecified_reference_N\": \"92\", \"scope\": \"Existing general q-reference error; not coordinate correction or verified query-miss truth\", \"N\": \"245\", \"frames\": \"89\", \"visible_support\": \"238\", \"occluded_support\": \"7\", \"TP\": \"52\", \"FN_decided\": \"186\", \"FP\": \"0\", \"TN\": \"7\", \"U_visible\": \"0\", \"U_occluded\": \"0\", \"out_of_task_S_visible\": \"0\", \"out_of_task_S_occluded\": \"0\", \"other_unresolved_visible\": \"0\", \"other_unresolved_occluded\": \"0\", \"visible_precision\": \"1.0\", \"visible_recall_all_GT\": \"0.2184873949579832\", \"occluded_precision\": \"0.03626943005181347\", \"occluded_recall_all_GT\": \"1.0\", \"abstention_rate\": \"0.0\", \"out_of_task_S_rate\": \"0.0\", \"unresolved_rate\": \"0.0\", \"decision_coverage\": \"1.0\", \"correct_fraction_all_GT\": \"0.24081632653061225\", \"selective_accuracy\": \"0.24081632653061225\"}, {\"cohort\": \"paired_same_query_RGB\", \"group\": \"WOOD_PRIMARY105\", \"task\": \"D_vs_E\", \"error_bin\": \"LE3\", \"model\": \"legacy\", \"manual_click_reference_N\": \"153\", \"other_or_unspecified_reference_N\": \"92\", \"scope\": \"Existing general q-reference error; not coordinate correction or verified query-miss truth\", \"N\": \"245\", \"frames\": \"89\", \"visible_support\": \"238\", \"occluded_support\": \"7\", \"TP\": \"181\", \"FN_decided\": \"21\", \"FP\": \"1\", \"TN\": \"6\", \"U_visible\": \"36\", \"U_occluded\": \"0\", \"out_of_task_S_visible\": \"0\", \"out_of_task_S_occluded\": \"0\", \"other_unresolved_visible\": \"0\", \"other_unresolved_occluded\": \"0\", \"visible_precision\": \"0.9945054945054945\", ",
  "truncated": true
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
동일 재질 공개 대조를 추가한 C13에서 공개 목재 VAL 가림은27/35→34/35로 개선됐으나 현재 촬영 paired 외부가림은목재8/29·정사6/61로 C12와 같았다. 목재 D는601/616→599/616, 정사 D531/531은 유지. 실사 외부가림 해결을 입증하지 못했다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_same_material_query_head_20261023_v118/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_same_material_query_head_20261023_v118/source_visual_followup/RESULT.json](source_visual_followup/RESULT.json)
- [query_visibility_same_material_query_head_20261023_v118/saved_current_review/REVIEW.json](saved_current_review/REVIEW.json)
- [query_visibility_same_material_query_head_20261023_v118/public_query_grounding_probe/CONCISE_INTERPRETATION_AND_EXECUTION_CLOSURE.json](public_query_grounding_probe/CONCISE_INTERPRETATION_AND_EXECUTION_CLOSURE.json)
- [query_visibility_same_material_query_head_20261023_v118/public_query_grounding_probe/RESULT.json](public_query_grounding_probe/RESULT.json)
- [query_visibility_same_material_query_head_20261023_v118/ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json](ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json)
- [query_visibility_same_material_foreground_source_20261023_v116/spatial_features_v1/FEATURE_COMPLETE.json](../query_visibility_same_material_foreground_source_20261023_v116/spatial_features_v1/FEATURE_COMPLETE.json)
- [query_visibility_same_material_foreground_source_20261023_v116/feature_integrity_review/REVIEW.json](../query_visibility_same_material_foreground_source_20261023_v116/feature_integrity_review/REVIEW.json)
- [query_visibility_same_material_query_head_20261023_v118/runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY/FIT_COMPLETE.json](runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY/FIT_COMPLETE.json)
- [query_visibility_same_material_query_head_20261023_v118/runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY/MODEL_FREEZE.json](runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY/MODEL_FREEZE.json)
- [query_visibility_same_material_query_head_20261023_v118/saved_source_review/REVIEW.json](saved_source_review/REVIEW.json)
- [query_visibility_same_material_query_head_20261023_v118/public_source_comparison/RESULT.json](public_source_comparison/RESULT.json)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/INFERENCE_SEAL.json](private_evaluation/C13_FROZEN_CURRENT223/INFERENCE_SEAL.json)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json](private_evaluation/C13_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/AVAILABILITY_BY_GROUP.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/AVAILABILITY_BY_GROUP.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/CONFUSION.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/CONFUSION.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/COORDINATE_ERROR_BINS.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/COORDINATE_ERROR_BINS.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/D_VS_EXTERNAL_E_2X3.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/D_VS_EXTERNAL_E_2X3.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/HUMAN_SELF_BINARY_RESPONSE.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/HUMAN_SELF_BINARY_RESPONSE.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/MANUAL_NEAR_D.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/MANUAL_NEAR_D.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/METRICS.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/METRICS.csv)
- [query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223/evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv](private_evaluation/C13_FROZEN_CURRENT223/evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_same_material_query_head_20261023_v118/source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png](source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png)

![원 보고서 진단 그림](source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png)

[query_visibility_same_material_query_head_20261023_v118/source_visual_followup/CASE_00.png](source_visual_followup/CASE_00.png)

![원 보고서 진단 그림](source_visual_followup/CASE_00.png)

[query_visibility_same_material_query_head_20261023_v118/source_visual_followup/CASE_01.png](source_visual_followup/CASE_01.png)

![원 보고서 진단 그림](source_visual_followup/CASE_01.png)

[query_visibility_same_material_query_head_20261023_v118/source_visual_followup/CASE_02.png](source_visual_followup/CASE_02.png)

![원 보고서 진단 그림](source_visual_followup/CASE_02.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
