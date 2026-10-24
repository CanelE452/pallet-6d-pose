# V180 — query_visibility_frozen_sam_source_baseline_20261024_v180

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `5a51c8ff00f038a8a79f8dc58818bf56bab32ade65f27cda05f5d84367e20639`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `50d9612eb9888f6918dec7a5a94acd75d002e3a143a63f70ccf57a0128245e94`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `de61c1b23e670a173002e4127746427f75e136207b34bf729ad38a4e869c2ed6`

## 방법·목표의 원문 필드

`$.schema`

```text
V180_SOURCE_ONLY_SAM_AND_LOCAL_SUPPORT_READOUT_v1
```


## 입력·학습·평가 분모의 원문 필드

`$.source_TRAIN_rows`

```text
4096
```

`$.new_training_updates`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.limits`

```text
[
  "Synthetic TRAIN diagnostics are neither real transfer nor independent generalization evidence.",
  "A class-agnostic SAM box mask has no selected-pallet/native identity certificate.",
  "Nonmask is not external occlusion; foreign and background are kept separate.",
  "Literal q owner, whole-image target mask, crop-any-pallet and fixed native-end support are different questions.",
  "New end supports are non-exclusive and connect source geometry to the fixed crop; they do not relabel original source native D/E/S or private human annotations.",
  "Whole image IoU can hide very small foreign covers; the fixed16 foreign union contains58 pixels.",
  "Predicted mask quality/logits are uncalibrated; no threshold was selected.",
  "Source supervision training_admission remains false; no new head was fitted in V180."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_supervision_train4096_v1/COMPLETE.json](source_end_supervision_train4096_v1/COMPLETE.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json](frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json.parts.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/saved_sam_allTRAIN4096_evaluation_c_v3/COMPLETE.json](saved_sam_allTRAIN4096_evaluation_c_v3/COMPLETE.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/independent_actual_sam_evaluation_review_b/REVIEW.json](independent_actual_sam_evaluation_review_b/REVIEW.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/independent_actual_end_support_review_c/REVIEW.json](independent_actual_end_support_review_c/REVIEW.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/ROOT_ACTUAL_VISUAL_AND_LINK_QA.json](ROOT_ACTUAL_VISUAL_AND_LINK_QA.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/ROOT_ACTUAL_EXECUTION_TERMINALS.json](ROOT_ACTUAL_EXECUTION_TERMINALS.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/final_report_review_c/REVIEW.json](final_report_review_c/REVIEW.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/saved_sam_allTRAIN4096_evaluation_c_v3/index.html](saved_sam_allTRAIN4096_evaluation_c_v3/index.html)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_spatial_visual128_v1/index.html](source_end_spatial_visual128_v1/index.html)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/saved_sam_allTRAIN4096_evaluation_c_v3/ACTUAL_VISUAL_QA_A.json](saved_sam_allTRAIN4096_evaluation_c_v3/ACTUAL_VISUAL_QA_A.json)
- [query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_spatial_visual128_v1/FIXED16_CONFIRMED_END_FOREIGN_RAW_RGB_CONTRAST.json](source_end_spatial_visual128_v1/FIXED16_CONFIRMED_END_FOREIGN_RAW_RGB_CONTRAST.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_frozen_sam_source_baseline_20261024_v180/scientific_readout_v1/SOURCE_SAM_AND_END_SUPPORT_V2.png](scientific_readout_v1/SOURCE_SAM_AND_END_SUPPORT_V2.png)

![원 보고서 진단 그림](scientific_readout_v1/SOURCE_SAM_AND_END_SUPPORT_V2.png)

[query_visibility_frozen_sam_source_baseline_20261024_v180/scientific_readout_v1/FIXED4_COVER_INPUT_SAM_TARGET_END.png](scientific_readout_v1/FIXED4_COVER_INPUT_SAM_TARGET_END.png)

![원 보고서 진단 그림](scientific_readout_v1/FIXED4_COVER_INPUT_SAM_TARGET_END.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
