# V185 — query_visibility_public_part_supervision_readiness_20261024_v185

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `2c44dea6ea9c6e4d149ed2fb9eb57cfc97881695ba2fcdc15dab3d8adc7af370`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `a0eef55a9166eaaaaa7201805702c9a6915d8c3415b3fabf57c9ed72549ba069`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `ad4c3f6e7e65939408dfde3d6b4491a4dd27858b2f941ef67c9bb17ba127233e`

## 방법·목표의 원문 필드

`$.full_goal_preserved`

```text
{
  "materials": [
    "WOOD",
    "SQUARE"
  ],
  "native_slots": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "self_upstream_geometry": true,
  "visibility_external_location_error_separate": true,
  "real_SQUARE_direct_part_supervision_missing": true,
  "runner_part_evidence_incomplete": true
}
```


## 입력·학습·평가 분모의 원문 필드

`$.source_assets`

```text
{
  "MR6D_source_TRAIN_frames": 185,
  "original_native_rows": 1480,
  "inputU": 194,
  "MR_GT_instance_observations": 258,
  "depth_frames": 185,
  "instance_full_masks": 258,
  "instance_visible_masks": 258,
  "GT_camera_scenes": 4,
  "nominal_Euro_mesh": 1,
  "YCB_donors": 59,
  "CF_payloads": 477,
  "CF_rows": 3816,
  "CF_inputU": 360
}
```

`$.new_training_admission`

```text
false
```

`$.new_training_or_classifier_forward`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.part_intervention_history`

```text
{
  "attempts": 66,
  "weak_observed_supported_cases": 17,
  "unknown_cases": 5,
  "passed_pairs": 46,
  "placement_failures": 5,
  "unknown_attempts": 15,
  "actual_variants": 143,
  "potential_statuses": 198,
  "weak_native_targets": 92,
  "weak_E_COVER": 46,
  "weak_D_SHAM": 46,
  "all_remaining_native_targets_UNKNOWN": 1052,
  "observed_native_slots": [
    0,
    1,
    4,
    5
  ],
  "all_original_labels_unchanged": true
}
```

`$.decision`

```text
Existing assets are sufficient to attempt a source-only nominal geometry registration probe, but neither old owner/weak-native labels nor rectangle overlap automatically defines V180 support2. No labels admitted or fit launched.
```


## 원문 표: grid_bbox_overlap_table

행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.

| 원문 열1 | 원문 열2 | 원문 열3 | 원문 열4 | 원문 열5 | 원문 열6 |
| --- | --- | --- | --- | --- | --- |
| 전체 96×96 | 16 | 17 | 5 | 44 | 46 |
| 중앙 32×32 | 12 | 17 | 5 | 35 | 46 |
| 중심 2×2 | 3 | 17 | 5 | 9 | 46 |

## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_part_supervision_readiness_20261024_v185/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_part_supervision_readiness_20261024_v185/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/report_review_a/REVIEW.json](report_review_a/REVIEW.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/ROOT_SCOPE_BEFORE_PUBLIC_SOURCE_TARGET_OR_ASSET_BODIES.json](ROOT_SCOPE_BEFORE_PUBLIC_SOURCE_TARGET_OR_ASSET_BODIES.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/public_semantics_review_a/REVIEW.json](public_semantics_review_a/REVIEW.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/local_train_asset_inventory_b/REVIEW.json](local_train_asset_inventory_b/REVIEW.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/counterfactual_part_readiness_c/REVIEW.json](counterfactual_part_readiness_c/REVIEW.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/local96_region_intersection_b/actual_intersections_v1/COMPLETE.json](local96_region_intersection_b/actual_intersections_v1/COMPLETE.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/ROOT_ACTUAL_SOURCE_DIAGNOSTIC_VIEW.json](ROOT_ACTUAL_SOURCE_DIAGNOSTIC_VIEW.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/ROOT_HISTORICAL_REPORT_PROSE_EXPOSURE_NOTE.json](ROOT_HISTORICAL_REPORT_PROSE_EXPOSURE_NOTE.json)
- [query_visibility_public_source_registration_probe_20261024_v186/ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json](../query_visibility_public_source_registration_probe_20261024_v186/ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json)
- [query_visibility_public_part_supervision_readiness_20261024_v185/local96_region_intersection_b/actual_intersections_v1/SUMMARY.json](local96_region_intersection_b/actual_intersections_v1/SUMMARY.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_native_part_counterfactual_20261024_v164/visual_readout_v1/Q0057_D1/CARD.png](../query_visibility_native_part_counterfactual_20261024_v164/visual_readout_v1/Q0057_D1/CARD.png)

![원 보고서 진단 그림](../query_visibility_native_part_counterfactual_20261024_v164/visual_readout_v1/Q0057_D1/CARD.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
