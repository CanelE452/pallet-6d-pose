# 게시 이미지 전체 파일 카탈로그

동결 manifest에 `PUBLISHED_RAW`로 등록된 이미지 **16,727개** 전부를 나열합니다. 결과나 품질에 따른 선택은 없습니다. 이미지 stat·open·decode는 수행하지 않았고, 경로·크기·SHA는 게시 manifest의 메타데이터를 복사했습니다.

- [전체 CSV](publication/REVIEW_IMAGE_CATALOG.csv) — 버전, 상대 경로, 원문 bytes, SHA256, GitHub blob URL
- [실험 색인](EXPERIMENT_INDEX_KO.md)
- [그림·갤러리 색인](GALLERY_INDEX_KO.md)

기준 manifest SHA256: `f5f4c2d819c7533492e24d2238baa8a6f638438c4458492d01c0321a34130fc7`. CSV는 publication root 기준 상대 경로를 유지합니다. URL은 `https://github.com/CanelE452/pallet-6d-pose`의 `main`을 가리키며 향후 갱신될 수 있습니다. 각 행 SHA와 동결 manifest가 원문 식별 근거이며, main URL 자체가 불변 검산 인증서는 아닙니다.

`버전`은 경로의 `query_visibility_*_v숫자` 실험 디렉터리로 구분했습니다. 해당 디렉터리가 없는 파일도 버전 미지정/공유 자산으로 모두 보존했습니다. 아래 링크는 파일 탐색용이며 실제 이미지 판독이나 성능 평가 완료를 뜻하지 않습니다.

| 버전/분류 | 파일 수 | 경로 찾기 |
|---|---:|---|
| v63 | 12 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_real_20261022_v63) |
| v64 | 3 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_multitarget_20261022_v64) |
| v65 | 8 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_amodal_global_20261022_v65) |
| v66 | 72 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_controlled_real_20261022_v66) |
| v67 | 15 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_visible_preservation_20261023_v67) |
| v68 | 7 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_sam21_frozen_20261023_v68) |
| v69 | 44 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_reference_semantic_20261023_v69) |
| v70 | 8 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_target_ownership_20261023_v70) |
| v71 | 639 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_reference_jitter_20261023_v71) |
| v72 | 147 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_cue_occlusion_20261023_v72) |
| v73 | 47 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_pallet_loco_20261023_v73) |
| v74 | 453 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_loco_native_context_20261023_v74) |
| v75 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_context_pretrain_20261023_v75) |
| v76 | 54 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_reference_local_support_20261023_v76) |
| v77 | 25 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_controlled_pairs_20261023_v77) |
| v78 | 7 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_dino_query_support_20261023_v78) |
| v79 | 31 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_natural_occlusion_audit_20261023_v79) |
| v80 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_dense_support_20261023_v80) |
| v81 | 8 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_grounded_target_20261023_v81) |
| v82 | 7 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_requested_category_20261023_v82) |
| v83 | 20 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_cue_audit_20261023_v83) |
| v84 | 101 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_paired_cue_20261023_v84) |
| v85 | 557 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_cue_corpus_20261023_v85) |
| v86 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_cue_evidence_20261023_v86) |
| v87 | 33 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_external_ray_20261023_v87) |
| v88 | 30 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_external_ray_head_20261023_v88) |
| v91 | 19 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_physical_cue_aux_20261023_v91) |
| v92 | 43 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_flat_pallet_cue_bridge_20261023_v92) |
| v93 | 9 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_synthetic_part_teacher_20261023_v93) |
| v94 | 84 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_matched_query_scalar_20261023_v94) |
| v95 | 53 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_same_q_counterfactual_20261023_v95) |
| v96 | 104 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_query_anchored_cue_20261023_v96) |
| v97 | 31 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_expected_cue_triplets_20261023_v97) |
| v98 | 4 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_C1_warmstart_20261023_v98) |
| v99 | 290 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_diverse_expected_cue_source_20261023_v99) |
| v100 | 72 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_box_cue_20261023_v100) |
| v101 | 21 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_local_residual_head_20261023_v101) |
| v102 | 11 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_external_only_bounded_prior_20261023_v102) |
| v103 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_flat_pallet_pilot_20261023_v103) |
| v104 | 17 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_flat_pallet_queries_20261023_v104) |
| v105 | 7 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_frozen_rgb_head_20261023_v105) |
| v106 | 80 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_ycb_surface_cues_20261023_v106) |
| v107 | 32 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_target_context_rgb_20261023_v107) |
| v108 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_target_relative_rgb_head_20261023_v108) |
| v109 | 47 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_ycb_real_surface_source_20261023_v109) |
| v110 | 45 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_natural_surface_rgb_head_20261023_v110) |
| v111 | 17 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_spatial_target_query_features_20261023_v111) |
| v113 | 57 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_ycb_wood_foam_source_20261023_v113) |
| v114 | 8 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_material_augmented_query_head_20261023_v114) |
| v115 | 35 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_loco_annotation_target_masks_20261023_v115) |
| v116 | 32 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_same_material_foreground_source_20261023_v116) |
| v117 | 44 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_success_failure_visual_review_20261023_v117) |
| v118 | 29 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_same_material_query_head_20261023_v118) |
| v119 | 26 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_consistent_query_context_20261023_v119) |
| v120 | 196 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_natural_pallet_cue_intake_20261023_v120) |
| v121 | 5 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_natural_cue_query_baseline_20261023_v121) |
| v122 | 91 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_flat_pallet_cue_source_20261023_v122) |
| v123 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_flat_pallet_augmented_head_20261023_v123) |
| v124 | 15 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_training_duration_control_20261023_v124) |
| v125 | 31 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_boundary_query_source_20261023_v125) |
| v126 | 1 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_wide_only_loss_diagnostic_20261023_v126) |
| v127 | 221 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_spatial_stencil_control_20261023_v127) |
| v128 | 505 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_registered_native_render_20261023_v128) |
| v129 | 18 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_registered_native_diversity_20261023_v129) |
| v130 | 4 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_coverage_duration_20261024_v130) |
| v131 | 5 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_small_query_jitter_20261024_v131) |
| v132 | 9 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_actual_initializer_source_20261024_v132) |
| v134 | 60 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_witness_diagnosis_20261024_v134) |
| v136 | 40 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_continuous_native_source_20261024_v136) |
| v137 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_continuous_source_coverage_control_20261024_v137) |
| v138 | 1 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_saved_query_reliability_20261024_v138) |
| v139 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_broad_source_geometry_control_20261024_v139) |
| v140 | 36 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_reference_feasibility_20261024_v140) |
| v141 | 16 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_initializer_association_20261024_v141) |
| v142 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_target_gradient_control_20261024_v142) |
| v143 | 1 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_frozen_native_current_benchmark_20261024_v143) |
| v144 | 25 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_selected_native_labels_20261024_v144) |
| v145 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_fine_rgb_control_20261024_v145) |
| v146 | 76 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_source_expansion_20261024_v146) |
| v148 | 4 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_supervision_control_20261024_v148) |
| v149 | 1 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_frozen_current_benchmark_20261024_v149) |
| v150 | 2 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_native_saved_mechanism_20261024_v150) |
| v151 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_public_native_intake_20261024_v151) |
| v152 | 48 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_actual_query_bridge_20261024_v152) |
| v153 | 503 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_full_public_source_20261024_v153) |
| v154 | 47 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_mr6d_source_supervision_bridge_20261024_v154) |
| v155 | 6 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_independent_auxiliary_source_control_20261024_v155) |
| v156 | 30 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_separated_frozen_current_benchmark_20261024_v156) |
| v157 | 23 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_context_label_generalization_20261024_v157) |
| v158 | 2,090 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_train_query_owner_witness_20261024_v158) |
| v159 | 15 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_part_support_feasibility_20261024_v159) |
| v160 | 29 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_slot_owner_factorial_20261024_v160) |
| v161 | 21 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_train_owner_counterfactual_20261024_v161) |
| v162 | 75 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_counterfactual_frozen_features_20261024_v162) |
| v163 | 2,952 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_train_native_part_witness_20261024_v163) |
| v164 | 768 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_part_counterfactual_20261024_v164) |
| v166 | 733 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_native_part_supervision_control_20261024_v166) |
| v169 | 95 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_frozen_public_threshold_current_benchmark_20261024_v169) |
| v171 | 3 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_constraint_witness_crops_20261024_v171) |
| v172 | 1 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_mask_semantics_audit_20261024_v172) |
| v173 | 3 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_full_source_query_causal_census_20261024_v173) |
| v175 | 73 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_query_owner_control_20261024_v175) |
| v176 | 68 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_owner_public_ycb_control_20261024_v176) |
| v177 | 1,975 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_held_blind_center_owner_review_20261024_v177) |
| v178 | 21 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_direct_owner_frozen_current_benchmark_20261024_v178) |
| v180 | 676 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_frozen_sam_source_baseline_20261024_v180) |
| v181 | 344 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_local_support_raw_head_20261024_v181) |
| v182 | 874 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_source_same_query_causal_diagnosis_20261024_v182) |
| v183 | 8 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_query_jitter_paired_source_training_20261024_v183) |
| v184 | 178 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_jitter_heads_frozen_real_benchmark_20261024_v184) |
| v186 | 18 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/query_visibility_public_source_registration_probe_20261024_v186) |
| 공유 gallery_assets | 33 | [해당 디렉터리](https://github.com/CanelE452/pallet-6d-pose/tree/main/_docs/experiments/pallet-keypoint-visibility/gallery_assets) |
| 버전 미지정 | 67 | [전체 CSV에서 버전 필터](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet-keypoint-visibility/publication/REVIEW_IMAGE_CATALOG.csv) |

합계 **16,727개**. 확장자 png/jpg/jpeg/svg/webp/gif/bmp/tif/tiff를 포함합니다. 원본문서·원 payload manifest·라벨·모델·평가값은 변경하지 않았습니다.

## 작은 CSV로 전체 파일 확인

GitHub에서 표를 열어 보기 쉽도록 전체 CSV의 원래 순서를 유지해 아래 조각을 추가했습니다. 각 조각은 같은 열 제목과 500행 이하, 300,000 bytes 미만입니다. 행 범위는 전체 CSV의 데이터 행 번호(제목 제외, 1부터)입니다. 전체 CSV도 그대로 보존합니다.

| 조각 | 전체 데이터 행 범위 | 행 수 | bytes |
|---|---:|---:|---:|
| [REVIEW_IMAGE_CATALOG_part001.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part001.csv) | 1–500 | 500 | 246,703 |
| [REVIEW_IMAGE_CATALOG_part002.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part002.csv) | 501–1,000 | 500 | 248,393 |
| [REVIEW_IMAGE_CATALOG_part003.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part003.csv) | 1,001–1,500 | 500 | 247,025 |
| [REVIEW_IMAGE_CATALOG_part004.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part004.csv) | 1,501–2,000 | 500 | 240,974 |
| [REVIEW_IMAGE_CATALOG_part005.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part005.csv) | 2,001–2,500 | 500 | 243,276 |
| [REVIEW_IMAGE_CATALOG_part006.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part006.csv) | 2,501–3,000 | 500 | 244,621 |
| [REVIEW_IMAGE_CATALOG_part007.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part007.csv) | 3,001–3,500 | 500 | 275,148 |
| [REVIEW_IMAGE_CATALOG_part008.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part008.csv) | 3,501–4,000 | 500 | 245,530 |
| [REVIEW_IMAGE_CATALOG_part009.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part009.csv) | 4,001–4,500 | 500 | 244,841 |
| [REVIEW_IMAGE_CATALOG_part010.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part010.csv) | 4,501–5,000 | 500 | 258,110 |
| [REVIEW_IMAGE_CATALOG_part011.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part011.csv) | 5,001–5,500 | 500 | 245,164 |
| [REVIEW_IMAGE_CATALOG_part012.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part012.csv) | 5,501–6,000 | 500 | 241,354 |
| [REVIEW_IMAGE_CATALOG_part013.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part013.csv) | 6,001–6,500 | 500 | 252,780 |
| [REVIEW_IMAGE_CATALOG_part014.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part014.csv) | 6,501–7,000 | 500 | 252,775 |
| [REVIEW_IMAGE_CATALOG_part015.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part015.csv) | 7,001–7,500 | 500 | 252,812 |
| [REVIEW_IMAGE_CATALOG_part016.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part016.csv) | 7,501–8,000 | 500 | 252,266 |
| [REVIEW_IMAGE_CATALOG_part017.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part017.csv) | 8,001–8,500 | 500 | 254,171 |
| [REVIEW_IMAGE_CATALOG_part018.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part018.csv) | 8,501–9,000 | 500 | 248,423 |
| [REVIEW_IMAGE_CATALOG_part019.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part019.csv) | 9,001–9,500 | 500 | 225,646 |
| [REVIEW_IMAGE_CATALOG_part020.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part020.csv) | 9,501–10,000 | 500 | 252,698 |
| [REVIEW_IMAGE_CATALOG_part021.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part021.csv) | 10,001–10,500 | 500 | 260,564 |
| [REVIEW_IMAGE_CATALOG_part022.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part022.csv) | 10,501–11,000 | 500 | 260,456 |
| [REVIEW_IMAGE_CATALOG_part023.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part023.csv) | 11,001–11,500 | 500 | 239,639 |
| [REVIEW_IMAGE_CATALOG_part024.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part024.csv) | 11,501–12,000 | 500 | 231,119 |
| [REVIEW_IMAGE_CATALOG_part025.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part025.csv) | 12,001–12,500 | 500 | 230,378 |
| [REVIEW_IMAGE_CATALOG_part026.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part026.csv) | 12,501–13,000 | 500 | 230,293 |
| [REVIEW_IMAGE_CATALOG_part027.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part027.csv) | 13,001–13,500 | 500 | 230,365 |
| [REVIEW_IMAGE_CATALOG_part028.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part028.csv) | 13,501–14,000 | 500 | 230,259 |
| [REVIEW_IMAGE_CATALOG_part029.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part029.csv) | 14,001–14,500 | 500 | 235,315 |
| [REVIEW_IMAGE_CATALOG_part030.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part030.csv) | 14,501–15,000 | 500 | 245,952 |
| [REVIEW_IMAGE_CATALOG_part031.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part031.csv) | 15,001–15,500 | 500 | 245,957 |
| [REVIEW_IMAGE_CATALOG_part032.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part032.csv) | 15,501–16,000 | 500 | 245,947 |
| [REVIEW_IMAGE_CATALOG_part033.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part033.csv) | 16,001–16,500 | 500 | 246,659 |
| [REVIEW_IMAGE_CATALOG_part034.csv](publication/review_image_catalog_parts/REVIEW_IMAGE_CATALOG_part034.csv) | 16,501–16,727 | 227 | 102,066 |

분할 합계 **16,727행**. 각 조각의 제목을 한 번만 남기고 순서대로 연결한 bytes의 SHA256은 원 전체 CSV와 동일한 `41efe713c58c400500c56795b3fb52ca5fe8fce1b3e468cd9b1f1b0443edcd64`입니다. 새 이미지 열람·선택·평가·원 manifest 변경은 없습니다.
