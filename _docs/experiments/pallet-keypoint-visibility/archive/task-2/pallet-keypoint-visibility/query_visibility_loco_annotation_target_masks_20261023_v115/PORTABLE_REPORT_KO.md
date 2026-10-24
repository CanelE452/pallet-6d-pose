# V115 — query_visibility_loco_annotation_target_masks_20261023_v115

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `8ab5c53e062e8d2133efc54cf20e39032771b2a4282aaba4101f255f54ccf71d`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `ed24ae09a903d634a1ab2fb0de37b43f47bbcf96faac56fad53cd535e574e143`

## 방법·목표의 원문 필드

`$.schema`

```text
V115_PUBLIC_SAM_PILOT_ROOT_READOUT_KO_V1
```


## 입력·학습·평가 분모의 원문 필드

`$.execution_seconds`

```text
5.917535969056189
```

`$.training_admission`

```text
false
```

`$.new_model_training`

```text
false
```


## 결론·실패·한계의 원문 필드

`$.decision_KO`

```text
초기 추론 대상과 대표 annotation의 대응이 미확인이고, 고정 사례에서 서로 다른 팔레트를 가리킵니다. 분할의 경계·구멍·소유권도 불확실하므로 candidate mask로만 보존합니다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_material_augmented_query_head_20261023_v114/public_source_regression_inventory/COMPACT_RESULT.json](../query_visibility_material_augmented_query_head_20261023_v114/public_source_regression_inventory/COMPACT_RESULT.json)
- [query_visibility_material_augmented_query_head_20261023_v114/next_wood_foreground_source_feasibility/RESULT.json](../query_visibility_material_augmented_query_head_20261023_v114/next_wood_foreground_source_feasibility/RESULT.json)
- [query_visibility_material_augmented_query_head_20261023_v114/ROOT_READOUT_KO.html](../query_visibility_material_augmented_query_head_20261023_v114/ROOT_READOUT_KO.html)
- [query_visibility_loco_annotation_target_masks_20261023_v115/ROOT_ACTUAL_SAM_CANDIDATE_EXECUTION.json](ROOT_ACTUAL_SAM_CANDIDATE_EXECUTION.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/source_sam_candidates_v1/SAM_CANDIDATE_OUTPUT_SEAL.json](source_sam_candidates_v1/SAM_CANDIDATE_OUTPUT_SEAL.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/independent_saved_mask_audit/completed_audit_v1/REVIEW.json](independent_saved_mask_audit/completed_audit_v1/REVIEW.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/independent_metadata_review/REVIEW.json](independent_metadata_review/REVIEW.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/ROOT_ACTUAL_FIXED_VISUAL_REVIEW_AND_DECISION.json](ROOT_ACTUAL_FIXED_VISUAL_REVIEW_AND_DECISION.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/source_sam_candidates_v1/SUMMARY.json](source_sam_candidates_v1/SUMMARY.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/independent_saved_mask_audit/completed_audit_v1/SEAL.json](independent_saved_mask_audit/completed_audit_v1/SEAL.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/independent_visual_observations/ACTUAL_VISUAL_OBSERVATIONS.json](independent_visual_observations/ACTUAL_VISUAL_OBSERVATIONS.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json](../query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json)
- [query_visibility_loco_annotation_target_masks_20261023_v115/report_preparation/build_readout.py](report_preparation/build_readout.py)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_loco_annotation_target_masks_20261023_v115/fixed_visual_review/OVERVIEW_01_FRAMES_00_09.png](fixed_visual_review/OVERVIEW_01_FRAMES_00_09.png)

![원 보고서 진단 그림](fixed_visual_review/OVERVIEW_01_FRAMES_00_09.png)

[query_visibility_loco_annotation_target_masks_20261023_v115/fixed_visual_review/OVERVIEW_02_FRAMES_10_19.png](fixed_visual_review/OVERVIEW_02_FRAMES_10_19.png)

![원 보고서 진단 그림](fixed_visual_review/OVERVIEW_02_FRAMES_10_19.png)

[query_visibility_loco_annotation_target_masks_20261023_v115/fixed_visual_review/detail_cards/FRAME_00_DETAIL.png](fixed_visual_review/detail_cards/FRAME_00_DETAIL.png)

![원 보고서 진단 그림](fixed_visual_review/detail_cards/FRAME_00_DETAIL.png)

[query_visibility_loco_annotation_target_masks_20261023_v115/fixed_visual_review/detail_cards/FRAME_01_DETAIL.png](fixed_visual_review/detail_cards/FRAME_01_DETAIL.png)

![원 보고서 진단 그림](fixed_visual_review/detail_cards/FRAME_01_DETAIL.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
