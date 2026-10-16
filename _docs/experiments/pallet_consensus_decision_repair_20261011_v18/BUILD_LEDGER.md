# V18 실행량과 게시 상태


```json
{
  "geometry_execution": {
    "parent_geometry_rows_visited": 490,
    "ambiguity_rows_visited": 490,
    "parent_ledger_rows_visited": 245,
    "original_point_only_rows_selected": 245,
    "pure_restoration_decisions": 245,
    "restored_rows": 161,
    "true_tie_rows_retained": 1,
    "replay_display_H_assignments": 185,
    "scalar_pose_to_mask_diagnostic_calls": 161,
    "geometry_rows": 490,
    "ledger_rows": 245,
    "frames_complete": 245,
    "actual_new_solver_calls": 0,
    "actual_new_optimizer_calls": 0,
    "actual_new_projection_calls": 0,
    "actual_new_J_calls": 0,
    "actual_new_SVD_calls": 0,
    "actual_new_model_GT_RGB_training_calls": 0
  },
  "geometry_scope": {
    "original_full_scalar_validation_passed": false,
    "original_full_scalar_checks": 198134007,
    "original_full_scalar_failures": 564,
    "all_failed_predicates_retained": true,
    "point_only_original_failed_predicates": 0,
    "independent_parent_point_check_count_not_claimed": true,
    "certified_scope": "unchanged POINT_ONLY candidates and fields covered by the completed original paired audit; no failed predicate in this arm",
    "full_geometry_PASS_not_claimed": true,
    "new_geometry_calls": 0
  },
  "geometry_checks": 626482,
  "statistics": {
    "saved_scored_rows_streamed": 1470,
    "moment_scalars_checked": 972,
    "distributions_checked": 162,
    "nonempty_CI_slots": 117,
    "metric_CI_slots_checked": 135,
    "paired_delta_scalars_computed": 16938,
    "nonempty_resampled_means_computed": 1170000,
    "pair_groups_checked": 45,
    "resample_denominators_computed": 450000,
    "None_CI_slots": 18
  },
  "runtime": {
    "rows": 150,
    "measured": 130,
    "checks": 677122,
    "summary_slots": 56
  },
  "bundle_files": 529,
  "bundle_bytes": 5073312474,
  "publication_complete": false,
  "goal_completion_not_inferred_from_audit": true
}
```

증거는 [CONTROL_RECEIPT.json](CONTROL_RECEIPT.json) / [VALIDATION_CHECKS.json](VALIDATION_CHECKS.json) / [VERIFICATION.json](VERIFICATION.json) / [RUNTIME_RECEIPT.json](RUNTIME_RECEIPT.json) / [RUNTIME_CHECKS.json](RUNTIME_CHECKS.json) / [FIGURE_BINDINGS.json](FIGURE_BINDINGS.json) / [PUBLIC_READER_CHECKS.json](PUBLIC_READER_CHECKS.json)이다. 부모 full FAIL와 기존 실패/원행을 삭제·PASS 치환하지 않는다. archive는 이미 frozen generic helper의 실제 manifest이며 새 학습/성능 재실험이 아니다. 후기 보고서는 원 bundle 뒤 생성된다. 게시 전용 브랜치 remote SHA 확인은 root의 후속 실제 영수증에 남긴다.
