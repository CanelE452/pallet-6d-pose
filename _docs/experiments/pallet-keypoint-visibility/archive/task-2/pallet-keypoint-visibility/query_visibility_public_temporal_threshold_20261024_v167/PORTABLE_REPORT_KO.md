# V167 — query_visibility_public_temporal_threshold_20261024_v167

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `de5f6051a74f201dc0158646a9cb8e16a1135b08a14f9026201176da773f8210`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `60685b47e806ac3efb62117c4f9a15c5169aa9adefdb4a159e087649c64d063e`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `163642d52d6603db76cebf9506c62ef78c5c42c2d631e404795d78869cad840f`

## 방법·목표의 원문 필드

`$.schema`

```text
V167_PUBLIC_ONLY_SCALAR_THRESHOLD_SAME_VIDEO_STABILITY_RESULT_KO
```

`$.phase_complete_scope`

```text
public MR-only scalar fitting and same-video purged threshold stability; no other-source/current qualification
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "completed": true,
  "actual_exec_exit_code": 0,
  "automatic_retry": false,
  "fixed_models": [
    "C45_REPLAY_C43_OWNER253_NATIVE0",
    "C46_PART_NATIVE92_OWNER253_NATIVE0"
  ],
  "new_head_training_calls": 0,
  "new_model_or_encoder_forward_calls": 0,
  "inputs": "already saved public MR logits and existing assistant weak labels only",
  "actual_scalar_fit_results": "18 purged folds plus2 full PUBLIC_CAL scalars",
  "independent_saved_audit_B_PASS": true,
  "independent_audit_checks_passed": 8,
  "both_full_scalars_frozen": true
}
```

`$.cohort`

```text
{
  "public_sequence": "val/000001",
  "public_sequences": 1,
  "frames": 45,
  "all_original_rows": 360,
  "full_known_D_E_rows": 154,
  "full_known_D": 102,
  "full_known_E": 52,
  "threshold_fit_eligible_rows": 145,
  "fit_D": 93,
  "fit_E": 52,
  "known_inputU9_all_D": true,
  "all360_inputU": 108,
  "unknown_truth": 206,
  "all_original_status_and_U_retained": true,
  "original_V166_split_labels_and_fixed0_results_unchanged": true
}
```


## 결론·실패·한계의 원문 필드

`$.authority_and_limits_KO`

```text
[
  "공식 공개 영상 val/000001 하나의45프레임이다. 9fold는9개 독립 장면이 아니고154query는154개 독립 표본이 아니다.",
  "V153~V166에서 공개 metric·crop·ranking을 반복 관찰하고 설계를 발전시켰다. 이 OOF는 문턱 fitting의 영상 내부 안정성만 다루며 fresh independent fullmodel validation이나 보지 않은 장면의 일반화가 아니다.",
  "D/E는 기존 assistant 약라벨이며 사람 native 가시성·물리 가림·좌표 품질 gold가 아니다.",
  "full PUBLIC_CAL은 같은145행에 문턱을 맞추고 적용한 apparent 수치다. C45 E47/52,D74/102 및 C46 E45/52,D65/102를 held-test 성공으로 해석하지 않는다.",
  "결정 logit 문턱만 맞췄다. p(E)가 확률적으로 보정됐다는 주장은 하지 않는다. 원래0문턱의 부정적 결과도 유지한다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_temporal_threshold_20261024_v167/public_calibration_v1/COMPLETE.json](public_calibration_v1/COMPLETE.json)
- [query_visibility_public_temporal_threshold_20261024_v167/public_calibration_v1/ALL2_THRESHOLDS_FROZEN_BEFORE_ANY_PRIVATE_READ.json](public_calibration_v1/ALL2_THRESHOLDS_FROZEN_BEFORE_ANY_PRIVATE_READ.json)
- [query_visibility_public_temporal_threshold_20261024_v167/independent_saved_calibration_review_b/REVIEW.json](independent_saved_calibration_review_b/REVIEW.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_READOUT_KO.html](../query_visibility_native_part_supervision_control_20261024_v166/ROOT_READOUT_KO.html)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/index.html](../query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/index.html)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/diagnostic_representatives.html](../query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/diagnostic_representatives.html)
- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_public_temporal_threshold_20261024_v167/public_calibration_v1/C45_REPLAY_C43_OWNER253_NATIVE0_PURGED9_FOLDS.json](public_calibration_v1/C45_REPLAY_C43_OWNER253_NATIVE0_PURGED9_FOLDS.json)
- [query_visibility_public_temporal_threshold_20261024_v167/public_calibration_v1/C46_PART_NATIVE92_OWNER253_NATIVE0_PURGED9_FOLDS.json](public_calibration_v1/C46_PART_NATIVE92_OWNER253_NATIVE0_PURGED9_FOLDS.json)
- [query_visibility_public_temporal_threshold_20261024_v167/independent_calibration_code_review_a/REVIEW.json](independent_calibration_code_review_a/REVIEW.json)
- [ACTIVE_HANDOFF_KO.json](../ACTIVE_HANDOFF_KO.json)
- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_ACTUAL_PUBLIC_THRESHOLD_ENTRY.json](ROOT_ACTUAL_PUBLIC_THRESHOLD_ENTRY.json)
- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_ACTUAL_PUBLIC_THRESHOLD_EXECUTION_TERMINAL.json](ROOT_ACTUAL_PUBLIC_THRESHOLD_EXECUTION_TERMINAL.json)
- [query_visibility_public_temporal_threshold_20261024_v167/ROOT_DESIGN_BEFORE_PUBLIC_THRESHOLD_CALIBRATION.json](ROOT_DESIGN_BEFORE_PUBLIC_THRESHOLD_CALIBRATION.json)
- [query_visibility_public_temporal_threshold_20261024_v167/calibration_preparation/PROTOCOL_BEFORE_PUBLIC_LOGIT_OR_CALIBRATION_LABEL_BODIES.json](calibration_preparation/PROTOCOL_BEFORE_PUBLIC_LOGIT_OR_CALIBRATION_LABEL_BODIES.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_RESULTS_KO.json](../query_visibility_native_part_supervision_control_20261024_v166/ROOT_RESULTS_KO.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
