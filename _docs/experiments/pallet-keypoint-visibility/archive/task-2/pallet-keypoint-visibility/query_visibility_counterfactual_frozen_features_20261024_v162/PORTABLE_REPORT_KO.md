# V162 — query_visibility_counterfactual_frozen_features_20261024_v162

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `928a46b1f9bea1b88ea0e4004608b50e4bcfc98e95f9008d882e85bcd18f7e69`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `bb7e71ea97213631ff0fa65f6eec03e79fae00bde47471deb77e1de8df5c8624`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `c0932355c3acb7a7841e31923af0639ba29aab00b2eeb292515354dea8409fcf`

## 방법·목표의 원문 필드

`$.schema`

```text
V162_ROOT_SOURCE_CONTROLLED_OWNER_EXTENSION_RESULT
```


## 입력·학습·평가 분모의 원문 필드

`$.original_source_rows_each`

```text
28976
```

`$.original_source_inputU_each`

```text
535
```

`$.CF_rows_preserved`

```text
3816
```

`$.CF_inputU`

```text
360
```

`$.updates_each`

```text
8000
```

`$.owner_rows`

```text
[
  253,
  567
]
```

`$.additional_owner_rows`

```text
314
```


## 결론·실패·한계의 원문 필드

`$.limitations_KO`

```text
[
  "공개 실사 가림은 기존 구조적 약라벨이며 공식 사람 키포인트 가시성 gold가 아니다.",
  "한 seed의 고정 대조실험이다. owner TRAIN 적합과 주 가림 일반화는 별도로 평가한다.",
  "CF는 53개 query/46개 TRAIN 프레임을 재사용했다. 477variant를 독립 실사 사진으로 세지 않는다.",
  "추가 CF anchor에는 native2/3이 없다. 원래 전체 WOOD/SQUARE/native8 범위와 평가 분모는 유지한다.",
  "별도 entity 한계와 실사 좌표품질 gold 부재가 남아 있다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_counterfactual_frozen_features_20261024_v162/feature_preparation/PREPARATION_COMPLETE.json](feature_preparation/PREPARATION_COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_ACTUAL_FEATURE_COMPLETED.json](ROOT_ACTUAL_FEATURE_COMPLETED.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/frozen_cf_features_v1/FEATURE_COMPLETE.json](frozen_cf_features_v1/FEATURE_COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/independent_saved_feature_review/REVIEW_COMPLETE.json](independent_saved_feature_review/REVIEW_COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_ACTUAL_FIT_COMPLETED.json](ROOT_ACTUAL_FIT_COMPLETED.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/runs/C43_C44_OWNER_CF_FIXED8000/CONTROL_COMPLETE.json](runs/C43_C44_OWNER_CF_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/independent_saved_fit_review_preparation/SAVED_C43_C44_EXECUTION_REVIEW.json](independent_saved_fit_review_preparation/SAVED_C43_C44_EXECUTION_REVIEW.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/source_result_readout/READOUT_COMPLETE.json](source_result_readout/READOUT_COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json](source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json.parts.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/source_result_readout/MR_HELD_KNOWN_DE_CHANGE_CENSUS.json](source_result_readout/MR_HELD_KNOWN_DE_CHANGE_CENSUS.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_source_visual_v2/COMPLETE.json](controlled_source_visual_v2/COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_source_visual_all_changed_mr_v1/COMPLETE.json](controlled_source_visual_all_changed_mr_v1/COMPLETE.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_source_visual_preparation/ACTUAL_VISUAL_QA_CAPTION_V2.json](controlled_source_visual_preparation/ACTUAL_VISUAL_QA_CAPTION_V2.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_source_visual_preparation/READOUT_COMPLETE_CAPTION_V2.json](controlled_source_visual_preparation/READOUT_COMPLETE_CAPTION_V2.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_ACTUAL_VISUAL_QA.json](ROOT_ACTUAL_VISUAL_QA.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_fit_independent_review/FINAL_PUBLICATION_REVIEW.json](controlled_fit_independent_review/FINAL_PUBLICATION_REVIEW.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/next_native_part_supervision_readiness/READINESS_MEMO.json](next_native_part_supervision_readiness/READINESS_MEMO.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/controlled_source_visual_v2/index_with_all_changed_appendix.html](controlled_source_visual_v2/index_with_all_changed_appendix.html)
- [query_visibility_counterfactual_frozen_features_20261024_v162/source_result_readout/ARM_DATASET_SUMMARY.json](source_result_readout/ARM_DATASET_SUMMARY.json)
- [query_visibility_counterfactual_frozen_features_20261024_v162/source_result_readout/C44_MINUS_C43_DIAGNOSTIC.json](source_result_readout/C44_MINUS_C43_DIAGNOSTIC.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_counterfactual_frozen_features_20261024_v162/ROOT_SOURCE_CONTROL_RESULT.png](ROOT_SOURCE_CONTROL_RESULT.png)

![원 보고서 진단 그림](ROOT_SOURCE_CONTROL_RESULT.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
