# V183 — query_visibility_query_jitter_paired_source_training_20261024_v183

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `d1faad0586fb222c3043f750b1be4ade87f2fc16e8c668ee780f48e3a934d774`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `6ed9769b3d9f22fa44730a93f8c57d4652d2485d0f0a1d82a106e5472abfa90b`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `7e5eb2a9304f0c4aa72355fcec11a46d0ecfcd4f12f1db3fe9e9f5e438d29ce9`

## 방법·목표의 원문 필드

`$.fixed_decoder`

```text
Two independent support logits strictly >0; original inputU first
```

`$.private_publicheld_new_pixel_prediction_annotation_body_access_this_phase`

```text
0
```


## 입력·학습·평가 분모의 원문 필드

`$.table_training_rows`

```text
[
  [
    "SYNTH_AUG65536",
    65536,
    64272,
    1264,
    48282,
    26335,
    0
  ],
  [
    "PUBLIC_TRAIN1480",
    1480,
    1286,
    194,
    0,
    0,
    253
  ],
  [
    "CF3816",
    3816,
    3456,
    360,
    0,
    0,
    314
  ]
]
```

`$.metric_rows_order`

```text
Binary truth NO/YES; prediction NO/YES/U
```

`$.source_training_apparent`

```text
true
```

`$.new_training_updates`

```text
16000
```


## 결론·실패·한계의 원문 필드

`$.full_goal_remaining`

```text
Verify frozen models on untouched current/publicheld inputs and references; reliable natural-real visibility/external occlusion remains unproven.
```


## 원문 표: table_fixed_pairs_and_old401

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 원문 열1 | 원문 열2 | 원문 열3 | 원문 열4 | 원문 열5 | 원문 열6 | 원문 열7 | 원문 열8 | 원문 열9 | 원문 열10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C50 기존 RGB | 84/95 | 401 | 401 | 390 | 11 | 0 | 0 | 0 | 0 |
| C51 기존 RGB+SAM | 84/95 | 401 | 401 | 390 | 11 | 0 | 0 | 0 | 0 |
| C52 증강 RGB | 90/95 | 401 | 400 | 396 | 4 | 0 | 0 | 0 | 0 |
| C53 증강 RGB+SAM | 92/95 | 401 | 401 | 398 | 3 | 0 | 0 | 0 | 0 |

## 원문 표: table_known_available_support

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 원문 열1 | 원문 열2 | 원문 열3 | 원문 열4 | 원문 열5 | 원문 열6 | 원문 열7 | 원문 열8 | 원문 열9 | 원문 열10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMMON4096 | C50 기존 RGB | 부위 보임 | [121, 2886] | 100.00% | 98.27% | [[121, 0, 0], [50, 2836, 0]] | 993 | 96 | 0 |
| COMMON4096 | C50 기존 RGB | 외부 물체가 앞에 있음 | [1495, 138] | 76.84% | 98.55% | [[1454, 41, 0], [2, 136, 0]] | 2367 | 96 | 0 |
| COMMON4096 | C51 기존 RGB+SAM | 부위 보임 | [121, 2886] | 100.00% | 98.93% | [[121, 0, 0], [31, 2855, 0]] | 993 | 96 | 0 |
| COMMON4096 | C51 기존 RGB+SAM | 외부 물체가 앞에 있음 | [1495, 138] | 80.47% | 98.55% | [[1462, 33, 0], [2, 136, 0]] | 2367 | 96 | 0 |
| COMMON4096 | C52 증강 RGB | 부위 보임 | [121, 2886] | 100.00% | 99.38% | [[121, 0, 0], [18, 2868, 0]] | 993 | 96 | 0 |
| COMMON4096 | C52 증강 RGB | 외부 물체가 앞에 있음 | [1495, 138] | 91.33% | 99.28% | [[1482, 13, 0], [1, 137, 0]] | 2367 | 96 | 0 |
| COMMON4096 | C53 증강 RGB+SAM | 부위 보임 | [121, 2886] | 100.00% | 99.45% | [[121, 0, 0], [16, 2870, 0]] | 993 | 96 | 0 |
| COMMON4096 | C53 증강 RGB+SAM | 외부 물체가 앞에 있음 | [1495, 138] | 90.79% | 100.00% | [[1481, 14, 0], [0, 138, 0]] | 2367 | 96 | 0 |
| SYNTH_VAL2048 | C50 기존 RGB | 부위 보임 | [91, 1328] | 97.95% | 93.37% | [[65, 26, 0], [88, 1240, 0]] | 621 | 8 | 0 |
| SYNTH_VAL2048 | C50 기존 RGB | 외부 물체가 앞에 있음 | [750, 73] | 43.12% | 64.38% | [[688, 62, 0], [26, 47, 0]] | 1217 | 8 | 0 |
| SYNTH_VAL2048 | C51 기존 RGB+SAM | 부위 보임 | [91, 1328] | 97.71% | 96.31% | [[61, 30, 0], [49, 1279, 0]] | 621 | 8 | 0 |
| SYNTH_VAL2048 | C51 기존 RGB+SAM | 외부 물체가 앞에 있음 | [750, 73] | 46.34% | 52.05% | [[706, 44, 0], [35, 38, 0]] | 1217 | 8 | 0 |
| SYNTH_VAL2048 | C52 증강 RGB | 부위 보임 | [91, 1328] | 98.23% | 95.86% | [[68, 23, 0], [55, 1273, 0]] | 621 | 8 | 0 |
| SYNTH_VAL2048 | C52 증강 RGB | 외부 물체가 앞에 있음 | [750, 73] | 42.24% | 67.12% | [[683, 67, 0], [24, 49, 0]] | 1217 | 8 | 0 |
| SYNTH_VAL2048 | C53 증강 RGB+SAM | 부위 보임 | [91, 1328] | 98.01% | 96.39% | [[65, 26, 0], [48, 1280, 0]] | 621 | 8 | 0 |
| SYNTH_VAL2048 | C53 증강 RGB+SAM | 외부 물체가 앞에 있음 | [750, 73] | 42.24% | 67.12% | [[683, 67, 0], [24, 49, 0]] | 1217 | 8 | 0 |

## 원문 표: table_training_rows

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 원문 열1 | 원문 열2 | 원문 열3 | 원문 열4 | 원문 열5 | 원문 열6 | 원문 열7 |
| --- | --- | --- | --- | --- | --- | --- |
| SYNTH_AUG65536 | 65536 | 64272 | 1264 | 48282 | 26335 | 0 |
| PUBLIC_TRAIN1480 | 1480 | 1286 | 194 | 0 | 0 | 253 |
| CF3816 | 3816 | 3456 | 360 | 0 | 0 | 314 |

## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json](ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000/COMPLETE.json](runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000/COMPLETE.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/independent_actual_fit_review_a/REVIEW.json](independent_actual_fit_review_a/REVIEW.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/saved_jitter_source_evaluation_v2_IDENTITY_GUARD_REPAIR/COMPLETE.json](saved_jitter_source_evaluation_v2_IDENTITY_GUARD_REPAIR/COMPLETE.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/independent_saved_evaluation_review_b/REVIEW.json](independent_saved_evaluation_review_b/REVIEW.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_FINAL_REPORT_QA.json](ROOT_FINAL_REPORT_QA.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_CF_LINEAGE_GUARD_REPAIR_BEFORE_CORRECTED_SAVED_EVALUATION.json](ROOT_CF_LINEAGE_GUARD_REPAIR_BEFORE_CORRECTED_SAVED_EVALUATION.json)
- [query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json](../query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184/ROOT_SCOPE_BEFORE_NEW_REAL_OUTPUTS_OR_V183_SOURCE_RESULTS.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/saved_jitter_source_evaluation_v2_IDENTITY_GUARD_REPAIR/ALL_FIXED_METRICS_STRATA_PAIRS_AND_FRAGILITY.json](saved_jitter_source_evaluation_v2_IDENTITY_GUARD_REPAIR/ALL_FIXED_METRICS_STRATA_PAIRS_AND_FRAGILITY.json.parts.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/fixed_source_case_and_plot_v1/index.html](fixed_source_case_and_plot_v1/index.html)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/fixed_source_case_and_plot_v1/COMPLETE.json](fixed_source_case_and_plot_v1/COMPLETE.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/independent_saved_augmentation_review_a/REVIEW.json](independent_saved_augmentation_review_a/REVIEW.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/independent_actual_sam_input_review_b/COMPOSER_SAM_REVIEW.json](independent_actual_sam_input_review_b/COMPOSER_SAM_REVIEW.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/independent_saved_evaluation_review_b/CF_LINEAGE_GUARD_REPAIR_REVIEW.json](independent_saved_evaluation_review_b/CF_LINEAGE_GUARD_REPAIR_REVIEW.json)
- [query_visibility_query_jitter_paired_source_training_20261024_v183/ROOT_ACTUAL_SAVED_IMAGE_QA.json](ROOT_ACTUAL_SAVED_IMAGE_QA.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_query_jitter_paired_source_training_20261024_v183/fixed_source_case_and_plot_v1/CARD.png](fixed_source_case_and_plot_v1/CARD.png)

![원 보고서 진단 그림](fixed_source_case_and_plot_v1/CARD.png)

[query_visibility_query_jitter_paired_source_training_20261024_v183/fixed_source_case_and_plot_v1/SCIENTIFIC_SOURCE_COMPARISON.png](fixed_source_case_and_plot_v1/SCIENTIFIC_SOURCE_COMPARISON.png)

![원 보고서 진단 그림](fixed_source_case_and_plot_v1/SCIENTIFIC_SOURCE_COMPARISON.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
