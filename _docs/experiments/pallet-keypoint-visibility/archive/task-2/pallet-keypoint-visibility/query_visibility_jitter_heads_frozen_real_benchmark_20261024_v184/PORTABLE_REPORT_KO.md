# V184 — query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `ed0d3a563a42894dabeb1bf51c71602c55505877decb94052f9acb1b4532ca9c`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `07f374d4194a6e5aece6cfc9df1b5246bd613921f5e8bab90ca7ec25a8c206c3`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `f77962db57317103b1d703edfca6a37c5de72f4f8d9485cc7033741db4e7819e`

## 방법·목표의 원문 필드

자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.

## 입력·학습·평가 분모의 원문 필드

`$.primary_all`

```text
[
  {
    "model": "v61",
    "group": "ALL",
    "N": 1237,
    "D": 1147,
    "E": 90,
    "CM_rows_D_E_cols_D_E_U": [
      [
        260,
        887,
        0
      ],
      [
        2,
        88,
        0
      ]
    ],
    "D_precision": 0.9923664122137404,
    "D_recall": 0.22667829119442023,
    "E_precision": 0.09025641025641026,
    "E_recall": 0.9777777777777777,
    "U_rate": 0.0,
    "E_wrongly_passed_as_visible": 2,
    "E_predicted_U": 0
  },
  {
    "model": "legacy",
    "group": "ALL",
    "N": 1237,
    "D": 1147,
    "E": 90,
    "CM_rows_D_E_cols_D_E_U": [
      [
        655,
        319,
        173
      ],
      [
        3,
        83,
        4
      ]
    ],
    "D_precision": 0.9954407294832827,
    "D_recall": 0.5710549258936356,
    "E_precision": 0.2064676616915423,
    "E_recall": 0.9222222222222223,
    "U_rate": 0.14308811641067098,
    "E_wrongly_passed_as_visible": 3,
    "E_predicted_U": 4
  },
  {
    "model": "C52",
    "group": "ALL",
    "N": 1237,
    "D": 1147,
    "E": 90,
    "CM_rows_D_E_cols_D_E_U": [
      [
        821,
        302,
        24
      ],
      [
        52,
        33,
        5
      ]
    ],
    "D_precision": 0.9404352806414662,
    "D_recall": 0.7157802964254577,
    "E_precision": 0.09850746268656717,
    "E_recall": 0.36666666666666664,
    "U_rate": 0.023443815683104285,
    "E_wrongly_passed_as_visible": 52,
    "E_predicted_U": 5
  },
  {
    "model": "C53",
    "group": "ALL",
    "N": 1237,
    "D": 1147,
    "E": 90,
    "CM_rows_D_E_cols_D_E_U": [
      [
        868,
        273,
        6
      ],
      [
        70,
        20,
        0
      ]
    ],
    "D_precision": 0.9253731343283582,
    "D_recall": 0.7567567567567568,
    "E_precision": 0.06825938566552901,
    "E_recall": 0.2222222222222222,
    "U_rate": 0.004850444624090542,
    "E_wrongly_passed_as_visible": 70,
    "E_predicted_U": 0
  }
]
```

`$.primary_by_material`

```text
{
  "excerpt": "[{\"model\": \"v61\", \"group\": \"WOOD_PRIMARY105\", \"N\": 645, \"D\": 616, \"E\": 29, \"CM_rows_D_E_cols_D_E_U\": [[145, 471, 0], [0, 29, 0]], \"D_precision\": 1.0, \"D_recall\": 0.2353896103896104, \"E_precision\": 0.058, \"E_recall\": 1.0, \"U_rate\": 0.0, \"E_wrongly_passed_as_visible\": 0, \"E_predicted_U\": 0}, {\"model\": \"legacy\", \"group\": \"WOOD_PRIMARY105\", \"N\": 645, \"D\": 616, \"E\": 29, \"CM_rows_D_E_cols_D_E_U\": [[448, 65, 103], [2, 23, 4]], \"D_precision\": 0.9955555555555555, \"D_recall\": 0.7272727272727273, \"E_precision\": 0.26136363636363635, \"E_recall\": 0.7931034482758621, \"U_rate\": 0.16589147286821707, \"E_wrongly_passed_as_visible\": 2, \"E_predicted_U\": 4}, {\"model\": \"C52\", \"group\": \"WOOD_PRIMARY105\", \"N\": 645, \"D\": 616, \"E\": 29, \"CM_rows_D_E_cols_D_E_U\": [[475, 120, 21], [21, 5, 3]], \"D_precision\": 0.9576612903225806, \"D_recall\": 0.7711038961038961, \"E_precision\": 0.04, \"E_recall\": 0.1724137931034483, \"U_rate\": 0.037209302325581395, \"E_wrongly_passed_as_visible\": 21, \"E_predicted_U\": 3}, {\"model\": \"C53\", \"group\": \"WOOD_PRIMARY105\", \"N\": 645, \"D\": 616, \"E\": 29, \"CM_rows_D_E_cols_D_E_U\": [[471, 144, 1], [24, 5, 0]], \"D_precision\": 0.9515151515151515, \"D_recall\": 0.7646103896103896, \"E_precision\": 0.03355704697986577, \"E_recall\": 0.1724137931034483, \"U_rate\": 0.0015503875968992248, \"E_wrongly_passed_as_visible\": 24, \"E_predicted_U\": 0}, {\"model\": \"v61\", \"group\": \"SQUARE118\", \"N\": 592, \"D\": 531, \"E\": 61, \"CM_rows_D_E_cols_D_E_U\": [[115, 416, 0], [2, 59, 0]], \"D_precision\": 0.9829059829059829, \"D_recall\": 0.21657250470809794, \"E_precision\": 0.12421052631578948, \"E_recall\": 0.9672131147540983, \"U_ra",
  "truncated": true
}
```

`$.table_primary`

```text
[
  [
    "v61",
    "99.24%",
    "22.67%",
    "9.03%",
    "97.78%",
    "0.00%",
    "[[260, 887, 0], [2, 88, 0]]"
  ],
  [
    "기존 실사 학습 모델",
    "99.54%",
    "57.11%",
    "20.65%",
    "92.22%",
    "14.31%",
    "[[655, 319, 173], [3, 83, 4]]"
  ],
  [
    "새 RGB",
    "94.04%",
    "71.58%",
    "9.85%",
    "36.67%",
    "2.34%",
    "[[821, 302, 24], [52, 33, 5]]"
  ],
  [
    "새 RGB+SAM",
    "92.54%",
    "75.68%",
    "6.83%",
    "22.22%",
    "0.49%",
    "[[868, 273, 6], [70, 20, 0]]"
  ]
]
```

`$.original_current_inputU`

```text
45
```

`$.primary_D1147_E90_N1237`

```text
true
```

`$.public_coverage`

```text
{
  "original_input_U": 108,
  "source_available": 252,
  "center4_invalid": 112,
  "otherwise_available_center4_invalid": 4,
  "no_constructed_crop": 24,
  "known_center204": 204,
  "unknown_center156": 156,
  "v61_own_available": 251,
  "v61_own_U": 109,
  "new_owner_U": {
    "C52": 108,
    "C53": 108
  },
  "new_regional_U": {
    "C52": 116,
    "C53": 110
  }
}
```

`$.threshold_model_architecture_or_training_data_choice`

```text
false
```

`$.training_definition_diagnosis`

```text
{
  "source_rows_audited": 70832,
  "public_CF_support_truth_all_UNKNOWN": true,
  "definitions": {
    "end": "Fixed selected-end geometric quads/camera projection, classmap registered against target-onlyBEFORE, projected through exact96fullcropbilinear samples. Visible means same targetpart/triangle/depth as BEFORE. Foreignfront means sourceforeign999 depth in front of an otherwise registered target-supported endpatch. Any positive sampled visible/foreignmass gives independent YES; BOTH permitted; self/ambiguity/registration/out-of-frame remainUNKNOWN. No sampled endmass means visibleNO/foreignUNKNOWN, never nativeE.",
    "literal_owner": "Separate central2x2 of fixed96 grid at [47:49]^2 with fourtap raster owner rawmass. Synthetic T aggregates anytargetpart, F999, BG, AMBIG/INVALID separate. Public253 assistantweak literal-center owner and CF314 selected-anchor opaque-cover/sham owner-only; not nativevisibility/qualitygold.",
    "training": "Five equal mean losses, supportBCE only synthetic; public253/CF314 ownerCE only. Shared trunk may couple representation, but ownerFOREIGN is not a direct foreign-support target. No new model or architecture conclusion."
  },
  "causal_real_failure_explanation_established": false
}
```


## 결론·실패·한계의 원문 필드

`$.conclusion`

```text
Visible recall improved while true external occlusion rejection degraded sharply. Neither new model is adopted; reliable visibility/external occlusion remains unsolved.
```


## 원문 표: table_primary

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 모델 | D precision | D recall | E precision | E recall | U rate | CM: truth D/E × pred D/E/U |
| --- | --- | --- | --- | --- | --- | --- |
| v61 | 99.24% | 22.67% | 9.03% | 97.78% | 0.00% | [[260, 887, 0], [2, 88, 0]] |
| 기존 실사 학습 모델 | 99.54% | 57.11% | 20.65% | 92.22% | 14.31% | [[655, 319, 173], [3, 83, 4]] |
| 새 RGB | 94.04% | 71.58% | 9.85% | 36.67% | 2.34% | [[821, 302, 24], [52, 33, 5]] |
| 새 RGB+SAM | 92.54% | 75.68% | 6.83% | 22.22% | 0.49% | [[868, 273, 6], [70, 20, 0]] |

## 원문 표: table_material

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 재질 그룹 | 모델 | D/E 분모 | D recall | E precision | E recall | U rate | E→D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WOOD_PRIMARY105 | v61 | 616/29 | 23.54% | 5.80% | 100.00% | 0.00% | 0 |
| WOOD_PRIMARY105 | 기존 실사 학습 모델 | 616/29 | 72.73% | 26.14% | 79.31% | 16.59% | 2 |
| WOOD_PRIMARY105 | 새 RGB | 616/29 | 77.11% | 4.00% | 17.24% | 3.72% | 21 |
| WOOD_PRIMARY105 | 새 RGB+SAM | 616/29 | 76.46% | 3.36% | 17.24% | 0.16% | 24 |
| SQUARE118 | v61 | 531/61 | 21.66% | 12.42% | 96.72% | 0.00% | 2 |
| SQUARE118 | 기존 실사 학습 모델 | 531/61 | 38.98% | 19.11% | 98.36% | 11.82% | 1 |
| SQUARE118 | 새 RGB | 531/61 | 65.16% | 13.33% | 45.90% | 0.84% | 31 |
| SQUARE118 | 새 RGB+SAM | 531/61 | 74.76% | 10.42% | 24.59% | 0.84% | 46 |

## 원문 표: table_manual_near376

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 모델 | N | D 응답 | D 응답률 | E 응답 | U 응답 |
| --- | --- | --- | --- | --- | --- |
| v61 | 376 | 85 | 22.61% | 291 | 0 |
| 기존 실사 학습 모델 | 376 | 205 | 54.52% | 108 | 63 |
| 새 RGB | 376 | 260 | 69.15% | 111 | 5 |
| 새 RGB+SAM | 376 | 272 | 72.34% | 100 | 4 |

## 원문 표: table_public_weak_center

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 모델 | weak truth T/nonT × pred T/nonT/U CM | T precision | T recall | 전체 prediction U | known weak N | unknown weak N |
| --- | --- | --- | --- | --- | --- | --- |
| v61 | [[7, 17, 0], [16, 164, 0]] | 30.43% | 29.17% | 109 | 204 | 156 |
| 새 RGB | [[17, 7, 0], [26, 154, 0]] | 39.53% | 70.83% | 108 | 204 | 156 |
| 새 RGB+SAM | [[16, 8, 0], [14, 166, 0]] | 53.33% | 66.67% | 108 | 204 | 156 |

## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_FINAL_REPORT_QA.json](ROOT_FINAL_REPORT_QA.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/frozen_C52_C53_real_predictions_v1/PREDICTION_SEAL.json](frozen_C52_C53_real_predictions_v1/PREDICTION_SEAL.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/saved_real_head_evaluation_v1/COMPLETE.json](saved_real_head_evaluation_v1/COMPLETE.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/independent_actual_saved_real_evaluation_review_b/REVIEW_V2_COHORT_JOIN.json](independent_actual_saved_real_evaluation_review_b/REVIEW_V2_COHORT_JOIN.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/source_target_alignment_audit_c/REVIEW.json](source_target_alignment_audit_c/REVIEW.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/index.html](fixed19_real_visual_v1/index.html)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/independent_actual_saved_real_evaluation_review_b/FAILED_ACTUAL_SAVED_REVIEW.json](independent_actual_saved_real_evaluation_review_b/FAILED_ACTUAL_SAVED_REVIEW.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_MINIMAL_INDEPENDENT_REAL_AUDITOR_COHORT_SCHEMA_REPAIR.json](ROOT_MINIMAL_INDEPENDENT_REAL_AUDITOR_COHORT_SCHEMA_REPAIR.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/saved_real_head_evaluation_v1/current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json](saved_real_head_evaluation_v1/current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/saved_real_head_evaluation_v1/publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json](saved_real_head_evaluation_v1/publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json](ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/COMPLETE.json](fixed19_real_visual_v1/COMPLETE.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/independent_original_input_review_a/REVIEW.json](independent_original_input_review_a/REVIEW.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_SAVED_SAM_PACKET_REVIEW.json](ROOT_SAVED_SAM_PACKET_REVIEW.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/independent_saved_prediction_review_a/REVIEW.json](independent_saved_prediction_review_a/REVIEW.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_ACTUAL_IMAGE_QA.json](ROOT_ACTUAL_IMAGE_QA.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/independent_actual_visual_review_a/REVIEW.json](independent_actual_visual_review_a/REVIEW.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/FIXED_PRIMARY_REAL_METRICS.png](fixed19_real_visual_v1/FIXED_PRIMARY_REAL_METRICS.png)

![원 보고서 진단 그림](fixed19_real_visual_v1/FIXED_PRIMARY_REAL_METRICS.png)

[query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_01.png](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_01.png)

![원 보고서 진단 그림](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_01.png)

[query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_02.png](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_02.png)

![원 보고서 진단 그림](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_02.png)

[query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_03.png](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_03.png)

![원 보고서 진단 그림](fixed19_real_visual_v1/ALL90_EXTERNAL_E_CONTACT_03.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
