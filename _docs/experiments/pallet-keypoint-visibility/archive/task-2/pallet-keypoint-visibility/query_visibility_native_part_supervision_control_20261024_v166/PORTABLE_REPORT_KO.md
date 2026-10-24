# V166 — query_visibility_native_part_supervision_control_20261024_v166

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"phase_complete":true,"selected_model":null,"adopted":false,"fully_solved":false,"goal_status":"active_unsolved"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `52cd8d8461771f86a9476329ed971d090f07d069a0db75b9195da11288ced9f1`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `38bc8bda70293d5b3e92a5a7e28e0ee6196528356bc095d61771507033e48f8d`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `e5b568cabbbae8521a549af293f60200e1ef50d46e12507227c4d260ff88aced`

## 방법·목표의 원문 필드

`$.schema`

```text
V166_ACTUAL_FIXED_NATIVE_PART92_CONTROL_RESULT_KO
```

`$.phase_complete_does_not_mean_full_task_success`

```text
true
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "actual_fits": 2,
  "optimizer_updates_each": 8000,
  "total_optimizer_updates": 16000,
  "terminal_exit_code": 0,
  "automatic_retry": false,
  "both_models_frozen_before_source_comparison": true,
  "common_original_draws_and_initial_state_equal": true,
  "C45_exact_C43_replay": true,
  "both_entity_quality_outputs_and_states_exact": true,
  "independent_saved_review_checks_passed": 6,
  "original_source_rows_each": 28976,
  "original_inputU_each": 535,
  "PART_rows_preserved": 1144,
  "PART_inputU_preserved": 178,
  "PART_all198_statuses_preserved": true,
  "PART_missing55_null": true
}
```


## 결론·실패·한계의 원문 필드

`$.visual_interpretation_KO`

```text
[
  "A가 확인한 FN에는 자연 바구니·나무상자와 팔레트의 비슷한 색·질감, 선택 부위와 literal q crop의 차이가 있었다. V164 합성 donor의 선명한 색·경계·그림자 차이는 가능한 설명 가설이며 실패 원인으로 확정하지 않는다.",
  "같은 갈색 바구니/나무상자 유형에서도 TP와 FN이 모두 있었다. 대표5개의 약한 관찰을 전체52E의 원인·대표성이나 native/좌표 gold로 확대하지 않는다.",
  "전체154 갤러리는 빠짐없는 저장 시각 증거다. 실제 직접 본 수는 A의5 CARD이며, 나머지149는 해시·배열·디코드·링크 검증 범위다."
]
```

`$.limitations_KO`

```text
[
  "공개 held는 공식 공개 시퀀스 val/000001의45프레임 한 묶음이며 시간적으로 인접한 프레임이다. 독립적인 여러 장면·데이터셋 일반화나 안정성 증거가 아니다.",
  "기존 D/E는 assistant 구조적 약라벨이다. 사람 native 가시성·물리 가림·3px 좌표 품질 gold가 아니다.",
  "한 seed의 고정 8,000회 대조이며 사후 문턱·epoch·조건 선택을 하지 않았다. AUROC는 동결 후 공개 입력 유효145행의 진단이다.",
  "전체 WOOD/SQUARE/native8 목표, 실사 SQUARE/runner/native7 부위 witness와 실사 좌표 품질 gold 부족, 별도 entity 한계가 남아 있다.",
  "이 공개 영상의 crop·점수는 V153–V166 과정에서 반복 관찰했다. 후속 purged time-block 결과는 문턱 적합의 영상 내 안정성 진단이며 독립적인 전체 모델 검증이 아니다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/CONTROL_COMPLETE.json](runs/C45_C46_PART_NATIVE_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_native_part_supervision_control_20261024_v166/independent_saved_fit_review_preparation/SAVED_C45_C46_EXECUTION_REVIEW.json](independent_saved_fit_review_preparation/SAVED_C45_C46_EXECUTION_REVIEW.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_POST_FREEZE_PUBLIC_RANKING_DIAGNOSIS.json](ROOT_POST_FREEZE_PUBLIC_RANKING_DIAGNOSIS.json)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/ACTUAL_VISUAL_QA_COMPLETE.json](source_validation_full154_visual_v1/ACTUAL_VISUAL_QA_COMPLETE.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json](ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/index.html](source_validation_full154_visual_v1/index.html)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/diagnostic_representatives.html](source_validation_full154_visual_v1/diagnostic_representatives.html)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/all_changed_appendix.html](source_validation_full154_visual_v1/all_changed_appendix.html)
- [ACTIVE_HANDOFF_KO.json](../ACTIVE_HANDOFF_KO.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/COMPLETE.json](source_validation_full154_visual_v1/COMPLETE.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_DESIGN_BEFORE_NATIVE_PART_SUPERVISION_FIT.json](ROOT_DESIGN_BEFORE_NATIVE_PART_SUPERVISION_FIT.json)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json](runs/C45_C46_PART_NATIVE_FIXED8000/ALL2_FROZEN_BEFORE_SOURCE_COMPARISON.json)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/BASELINE_C43_EXACT_REPLAY_PARITY.json](runs/C45_C46_PART_NATIVE_FIXED8000/BASELINE_C43_EXACT_REPLAY_PARITY.json)
- [query_visibility_native_part_supervision_control_20261024_v166/ROOT_ACTUAL_FIT_EXECUTION_TERMINAL.json](ROOT_ACTUAL_FIT_EXECUTION_TERMINAL.json)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/ALL360_SOURCE_VALIDATION_CENSUS.jsonl](source_validation_full154_visual_v1/ALL360_SOURCE_VALIDATION_CENSUS.jsonl)
- [query_visibility_native_part_supervision_control_20261024_v166/source_validation_full154_visual_v1/ALL154_GALLERY_MANIFEST.jsonl](source_validation_full154_visual_v1/ALL154_GALLERY_MANIFEST.jsonl)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/C45_REPLAY_C43_OWNER253_NATIVE0/FIT_COMPLETE.json](runs/C45_C46_PART_NATIVE_FIXED8000/C45_REPLAY_C43_OWNER253_NATIVE0/FIT_COMPLETE.json)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/C45_REPLAY_C43_OWNER253_NATIVE0/SOURCE_RESULTS.json](runs/C45_C46_PART_NATIVE_FIXED8000/C45_REPLAY_C43_OWNER253_NATIVE0/SOURCE_RESULTS.json)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/C46_PART_NATIVE92_OWNER253_NATIVE0/FIT_COMPLETE.json](runs/C45_C46_PART_NATIVE_FIXED8000/C46_PART_NATIVE92_OWNER253_NATIVE0/FIT_COMPLETE.json)
- [query_visibility_native_part_supervision_control_20261024_v166/runs/C45_C46_PART_NATIVE_FIXED8000/C46_PART_NATIVE92_OWNER253_NATIVE0/SOURCE_RESULTS.json](runs/C45_C46_PART_NATIVE_FIXED8000/C46_PART_NATIVE92_OWNER253_NATIVE0/SOURCE_RESULTS.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
