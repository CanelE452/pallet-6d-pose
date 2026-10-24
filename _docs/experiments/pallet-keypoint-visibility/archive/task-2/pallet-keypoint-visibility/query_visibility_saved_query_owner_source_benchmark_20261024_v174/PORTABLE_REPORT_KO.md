# V174 — query_visibility_saved_query_owner_source_benchmark_20261024_v174

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"adopted":false,"fully_solved":false}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `af5765a88c67c82b275f3e81633c4866c897bf8e0f0b8318e1f43042cdb95e6d`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `b380f8573b0721b16afa423e36d644027da15c71777df8dc7642276a868bf5b8`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `0f7b6e6eb47eb3901820ec578eb28f2c5db11d144323a227ada9988777517941`

## 방법·목표의 원문 필드

`$.schema`

```text
V174_SAVED_SOURCE_OWNER_REPORT_KO_v1
```

`$.phase`

```text
V174
```

`$.soft_truth_definition`

```text
원 q exactpoint owner6 float64 질량에서 INCIDENT+OTHER_TARGET=T, FOREIGN=F, BG=B; ambiguity/invalid은unknown, null q는 inventory unknown으로 별도 보존. categorical GT cutoff 없음.
```

`$.metrics_definition`

```text
Expected known-mass confusion, precision and recall; integer human-label counts or calibrated probabilities are not claimed. U prediction column remains in confusion/recall denominator.
```


## 입력·학습·평가 분모의 원문 필드

`$.primary`

```text
original VALIDATION2048 FULL_INVENTORY
```

`$.cohort_denominators`

```text
{
  "ALL": {
    "rows": 6144,
    "known_mass": 6056.0,
    "target_mass": 2400.7019595522434,
    "non_target_mass": 3655.2980404477566,
    "inputU_rows": 88,
    "ambiguous_mass": 0.0,
    "invalid_mass": 32.0,
    "null_query_unknown_rows": 56,
    "paired_matched_available_rows": 6056,
    "TRAIN_apparent_only": false
  },
  "TRAIN": {
    "rows": 4096,
    "known_mass": 4016.0,
    "target_mass": 1535.8365631978959,
    "non_target_mass": 2480.163436802104,
    "inputU_rows": 80,
    "ambiguous_mass": 0.0,
    "invalid_mass": 32.0,
    "null_query_unknown_rows": 48,
    "paired_matched_available_rows": 4016,
    "TRAIN_apparent_only": true
  },
  "VALIDATION": {
    "rows": 2048,
    "known_mass": 2040.0,
    "target_mass": 864.8653963543475,
    "non_target_mass": 1175.1346036456525,
    "inputU_rows": 8,
    "ambiguous_mass": 0.0,
    "invalid_mass": 0.0,
    "null_query_unknown_rows": 8,
    "paired_matched_available_rows": 2040,
    "TRAIN_apparent_only": false
  }
}
```

`$.new_training_forward_encoder_thresholdfit_PnP_qL_or_label_change`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.limitations_KO`

```text
[
  "Exactpoint owner는 crop 안에 부위가 있음·physical native/keypoint 정확도·원 native 외부가림/자체가림 정답과 같지 않다.",
  "TRAIN4096은 apparent 진단이며 primary는 고정 VALIDATION2048이다. 전체6144 및 paired6056을 구분한다.",
  "C45/C46과 v61의 수치는 서로 다른 출력 의미의 제한된 source 진단이다. 모델 승자 선택·채택·전체 목표 해결을 뜻하지 않는다.",
  "고정 VALIDATION은 과거 반복 사용된 공개 source 분할이며 새 blind/external 검증이 아니다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_owner_source_benchmark_v1/COMPLETE.json](saved_owner_source_benchmark_v1/COMPLETE.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/actual_saved_owner_evaluation_review_a/REVIEW.json](actual_saved_owner_evaluation_review_a/REVIEW.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/ROOT_ACTUAL_SAVED_OWNER_EVALUATION_TERMINAL.json](ROOT_ACTUAL_SAVED_OWNER_EVALUATION_TERMINAL.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_owner_source_benchmark_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json](saved_owner_source_benchmark_v1/EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_owner_source_benchmark_v1/ALL6144_SAVED_OWNER_AND_EXACT_POINT_MASS.csv](saved_owner_source_benchmark_v1/ALL6144_SAVED_OWNER_AND_EXACT_POINT_MASS.csv)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json)
- [query_visibility_saved_query_owner_source_benchmark_20261024_v174/saved_evaluation_preparation/PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json](saved_evaluation_preparation/PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
