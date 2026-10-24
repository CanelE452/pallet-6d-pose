# V114 — query_visibility_material_augmented_query_head_20261023_v114

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false}

원본:

- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `ae2092656fd4d55a24a06af9bd35f0a0859af7df2cfda7bbf6c123edfa0e4da3`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `7ff805b273e5ff45b3459543607b24fd581731ce27a4a37c129c93fa45b9c112`

## 방법·목표의 원문 필드

`$.phase`

```text
V114_C12_MATERIAL_AUGMENTED_SPATIAL_QUERY
```


## 입력·학습·평가 분모의 원문 필드

`$.execution_boundary`

```text
{
  "report_saved_metadata_only": true,
  "report_new_model_calls": 0,
  "report_new_RGB_or_NPZ_array_reads": 0,
  "new_private_or_holdout_fit": 0,
  "V112_new_inference": 0,
  "threshold_or_arm_choice_from_current": false,
  "PnP": 0,
  "parent_pointer_writes": 0,
  "sealed_result_edits": 0
}
```


## 결론·실패·한계의 원문 필드

`$.limits_KO`

```text
[
  "공개 material 추가 후 같은 실사 D/E 비교에서 외부 E 재현율은 C11보다 낮아졌다. 범용 대체 모델이나 목표 완료를 선언하지 않는다.",
  "C11과 head/feature 정의는 같지만 자료 구성, 배치64→96, 선택 기준8→10코호트가 함께 바뀌었다. 단일 원인의 효과로 해석할 수 없다.",
  "공개 source teacher는 제공 pose/depth/owner 기반 조건부 표면 AUX와 AI 약한 물리 단서이다. 실제 initializer native8 대응이나 인간 팔레트 모서리 gold가 아니다.",
  "material VAL120은 개발/체크포인트 선택에 노출되었다. Foam의 E 및 VAL은 없으므로 foam 외부가림 전이를 평가하지 못했다.",
  "실사223도 과거 개발 과정에 노출된 자료이다. 이번 점수로 재학습·체크포인트·문턱·arm을 선택하지 않았다.",
  "D/E 가시성 반응과 좌표 정확성은 다르다. q/L은 그대로이며 좌표 수정·PnP·좌표 정답 검증기는 없다.",
  "초기 기하 self 경로와 raw binary hidden을 구분한다. raw hidden은 학습된 S/가림 원인 분류가 아니며 SQUARE의 인간 S 정답은 없다.",
  "C12 동결 뒤 실행한 고정 C11 material baseline은 추가 진단이다. C12 선택이나 결과 수정에 사용하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/D_VS_EXTERNAL_E_2X3.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/D_VS_EXTERNAL_E_2X3.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/METRICS.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/METRICS.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/HUMAN_SELF_BINARY_RESPONSE.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/HUMAN_SELF_BINARY_RESPONSE.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/MANUAL_NEAR_D.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/MANUAL_NEAR_D.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/COORDINATE_ERROR_BINS.csv](private_evaluation/C12_FROZEN_CURRENT223/evaluation/COORDINATE_ERROR_BINS.csv)
- [query_visibility_material_augmented_query_head_20261023_v114/runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SOURCE_RESULTS.json](runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SOURCE_RESULTS.json)
- [query_visibility_material_augmented_query_head_20261023_v114/runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SOURCE_GATE_RESULTS.json](runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/SOURCE_GATE_RESULTS.json)
- [query_visibility_material_augmented_query_head_20261023_v114/saved_source_review/REVIEW.json](saved_source_review/REVIEW.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/c11_frozen_material_baseline/SOURCE_RESULTS.json](../query_visibility_ycb_wood_foam_source_20261023_v113/c11_frozen_material_baseline/SOURCE_RESULTS.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/ROOT_ACTUAL_FIXED7_MATERIAL_RGB_OBSERVATIONS.json](../query_visibility_ycb_wood_foam_source_20261023_v113/ROOT_ACTUAL_FIXED7_MATERIAL_RGB_OBSERVATIONS.json)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/INFERENCE_SEAL.json](private_evaluation/C12_FROZEN_CURRENT223/INFERENCE_SEAL.json)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json](private_evaluation/C12_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json)
- [query_visibility_material_augmented_query_head_20261023_v114/saved_current_review/REVIEW.json](saved_current_review/REVIEW.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json](../query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json)
- [query_visibility_material_augmented_query_head_20261023_v114/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_material_augmented_query_head_20261023_v114/ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json](ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json)
- [query_visibility_material_augmented_query_head_20261023_v114/runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/MODEL_FREEZE.json](runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/MODEL_FREEZE.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/independent_actual_teacher_review/REVIEW.json](../query_visibility_ycb_wood_foam_source_20261023_v113/independent_actual_teacher_review/REVIEW.json)
- [query_visibility_spatial_target_query_features_20261023_v111/ROOT_READOUT_KO.html](../query_visibility_spatial_target_query_features_20261023_v111/ROOT_READOUT_KO.html)
- [query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223/evaluation/CURRENT_PRIMARY_RESULTS.json](private_evaluation/C12_FROZEN_CURRENT223/evaluation/CURRENT_PRIMARY_RESULTS.json)
- [query_visibility_material_augmented_query_head_20261023_v114/runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/FIT_COMPLETE.json](runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY/FIT_COMPLETE.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/corpus_build_v1/CORPUS_SEAL.json](../query_visibility_ycb_wood_foam_source_20261023_v113/corpus_build_v1/CORPUS_SEAL.json)
- [query_visibility_ycb_wood_foam_source_20261023_v113/ROOT_CONDITIONAL_MATERIAL_SOURCE_TRAINING_ADMISSION.json](../query_visibility_ycb_wood_foam_source_20261023_v113/ROOT_CONDITIONAL_MATERIAL_SOURCE_TRAINING_ADMISSION.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_ycb_wood_foam_source_20261023_v113/fixed_material_rgb_review/CONTACT_01_FIXED_MATERIAL_CASES.png](../query_visibility_ycb_wood_foam_source_20261023_v113/fixed_material_rgb_review/CONTACT_01_FIXED_MATERIAL_CASES.png)

![원 보고서 진단 그림](../query_visibility_ycb_wood_foam_source_20261023_v113/fixed_material_rgb_review/CONTACT_01_FIXED_MATERIAL_CASES.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
