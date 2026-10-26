# Batch1 최종 요구사항 검증

[확인] 분석 산출물 검사 통과. T1(b)와 녹화 단위 CI는 근거/조건 부족으로 미실행·미계산 상태를 보존했다. main 공개는 별도 remote SHA 검증 대상이다.

| 검사 | 결과 |
|---|---|
| initial_file_and_summary_gate | True |
| Gate5_existing_derived_scores_only_user_authorized | True |
| quantitative_CSV_unchanged_since_authorization | 18 |
| T1a_21rows_14252_coordinate_errors_reproduced | True |
| T1b_status | SKIPPED_UNPROVEN_ORIGINAL_INFERENCE |
| T2_frame_CI_calculated | 150 |
| T2_recording_CI_not_calculated | 150 |
| T3_equivalence | {"main": {"frames": 876, "nine_channel_comparisons": 7884, "frame_mismatches": 0, "mask_mismatches": 0, "channel_mismatches": 0, "candidate_frames": 1395, "candidate_mismatches": 0}, "noflip": {"frames": 876, "nine_channel_comparisons": 7884, "frame_mismatches": 0, "mask_mismatches": 0, "channel_mismatches": 0, "candidate_frames": 1395, "candidate_mismatches": 0}} |
| T3_stage_rows_primary_and_manual | 132 |
| T4_primary_21_auxiliary_24_no_new_forward | True |
| T5_292_deltas_9_fixed_distinct_cases_PNG_and_PDF | True |
| T6_existing_score_rows_total | 72 |
| T6_actual_cross_kind_rows | 36 |
| neural_forwards_by_task | {"T1b": 0, "T3": 1752, "T4": 0, "others": 0} |
| retraining_threshold_hyperparameter_changes | 0 |
| original_experiment_files_unchanged | 598 |
| R0_unchanged | True |
| independent_full_scope_T0_T6_review_passed | True |
| current_report_and_publication_safety_review_passed | True |
