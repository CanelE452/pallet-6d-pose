# V157 — query_visibility_source_context_label_generalization_20261024_v157

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `9b02f6251d61d77f5ad0ca871e0234f26c046bdda0e63b9565d5f8620824bd18`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `84eb8399d9575493fbeb37a62c20903f6edc9561e2d7ac07edc14319d6628f76`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `bc2039c7478c51d61a028dd1e0eb1f57e4072893b2beb796a6f5cb2fe2df3a73`

## 방법·목표의 원문 필드

`$.schema`

```text
V157_SOURCE_CONTEXT_LABEL_GENERALIZATION_DIAGNOSIS_COMPLETE_GOAL_UNSOLVED
```

`$.previous_goal_turn_classification`

```text
progress
```


## 입력·학습·평가 분모의 원문 필드

`$.analyzed_rows`

```text
{
  "MR1840": 1840,
  "PALLOX512": 512,
  "OLD6144": 6144,
  "NEW20480": 20480
}
```

`$.all_source_rows_labels_splits_qL_preserved`

```text
true
```

`$.occluder_coverage`

```text
{
  "SOURCE_VALIDATION::WICKER_BASKET": {
    "rows": 28,
    "frames": 28,
    "native_counts": {
      "0": 0,
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 25,
      "5": 3,
      "6": 0,
      "7": 0
    }
  },
  "SOURCE_VALIDATION::SLATTED_WOOD_CRATE": {
    "rows": 24,
    "frames": 24,
    "native_counts": {
      "0": 0,
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 0,
      "5": 24,
      "6": 0,
      "7": 0
    }
  },
  "SOURCE_DEVELOPMENT::GRAY_MOULDED_TUB": {
    "rows": 23,
    "frames": 23,
    "native_counts": {
      "0": 0,
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 1,
      "5": 22,
      "6": 0,
      "7": 0
    }
  },
  "SOURCE_DEVELOPMENT::BLUE_PLASTIC_KLT": {
    "rows": 8,
    "frames": 8,
    "native_counts": {
      "0": 0,
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 4,
      "5": 4,
      "6": 0,
      "7": 0
    }
  },
  "SOURCE_DEVELOPMENT::GRAY_CRATE_AND_WHITE_PACKAGED_BLOCK": {
    "rows": 18,
    "frames": 18,
    "native_counts": {
      "0": 0,
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 6,
      "5": 12,
      "6": 0,
      "7": 0
    }
  }
}
```

`$.new_model_training_encoder_R0_private_V112_PnP_qL_calls`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.limits_KO`

```text
[
  "public E가5개native에없지만 heldE도native4/5라 그것만으로held실패를설명하지않습니다.",
  "TRAIN 가림물은회색/파란플라스틱계열, held는바구니/슬랫목재계열입니다. texture/scene 차이는관찰됐지만단독인과증명은아닙니다.",
  "target-context descriptor는같은frame의D/E에서동일합니다. 그러나현모델은query-token/local96/central32도사용하므로context만사용하는모델이라고주장하지않습니다.",
  "같은frame제외TRAIN의이웃은시간/장면이유사할수있어거리차이를독립물리샘플의일반화증명으로보지않습니다.",
  "명목참조는cm오차가능pose/CAD이며pixel/nativegold가아닙니다. 기존역할라벨도assistant약한감독으로유지합니다.",
  "기존E/위치오류/자기가림을합치거나보조owner음수를외부가림정답으로만들지않습니다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_source_context_label_generalization_20261024_v157/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_source_context_label_generalization_20261024_v157/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_source_context_label_generalization_20261024_v157/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_source_context_label_generalization_20261024_v157/ROOT_NEXT_SOURCE_ACTION_BEFORE_ANY_NEW_FIT.json](ROOT_NEXT_SOURCE_ACTION_BEFORE_ANY_NEW_FIT.json)
- [query_visibility_source_context_label_generalization_20261024_v157/source_crop_review_agent/FINAL_VISUAL_QA.json](source_crop_review_agent/FINAL_VISUAL_QA.json)
- [query_visibility_source_context_label_generalization_20261024_v157/source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json](source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json)
- [query_visibility_source_context_label_generalization_20261024_v157/ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json](ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json)
- [query_visibility_source_context_label_generalization_20261024_v157/root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json](root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json)
- [query_visibility_source_context_label_generalization_20261024_v157/root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json](root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json)
- [query_visibility_source_context_label_generalization_20261024_v157/source_crop_review_agent/SOURCE_CROP_REVIEW_FINDINGS.json](source_crop_review_agent/SOURCE_CROP_REVIEW_FINDINGS.json)
- [query_visibility_same_q_counterfactual_20261023_v95/ROOT_C2_ACTUAL_CORPUS_ADMISSION_BEFORE_FIT.json](../query_visibility_same_q_counterfactual_20261023_v95/ROOT_C2_ACTUAL_CORPUS_ADMISSION_BEFORE_FIT.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_source_context_label_generalization_20261024_v157/SOURCE_LABEL_AND_QUERY_ALIGNMENT_KO.png](SOURCE_LABEL_AND_QUERY_ALIGNMENT_KO.png)

![원 보고서 진단 그림](SOURCE_LABEL_AND_QUERY_ALIGNMENT_KO.png)

[query_visibility_source_context_label_generalization_20261024_v157/source_crop_review_agent/contact_sheets/CATEGORY_FIRST_ROWS_04.png](source_crop_review_agent/contact_sheets/CATEGORY_FIRST_ROWS_04.png)

![원 보고서 진단 그림](source_crop_review_agent/contact_sheets/CATEGORY_FIRST_ROWS_04.png)

[query_visibility_source_context_label_generalization_20261024_v157/source_feature_coverage_agent/SOURCE_COVERAGE_SAVED_ERRORS.png](source_feature_coverage_agent/SOURCE_COVERAGE_SAVED_ERRORS.png)

![원 보고서 진단 그림](source_feature_coverage_agent/SOURCE_COVERAGE_SAVED_ERRORS.png)

[query_visibility_source_context_label_generalization_20261024_v157/source_crop_review_agent/contact_sheets/FIRST12TRAIN_01.png](source_crop_review_agent/contact_sheets/FIRST12TRAIN_01.png)

![원 보고서 진단 그림](source_crop_review_agent/contact_sheets/FIRST12TRAIN_01.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
