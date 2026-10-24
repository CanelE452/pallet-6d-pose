# V147 — query_visibility_public_native_features_20261024_v147

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `343c92c3f332f9683da971e4d402084b0c255e20273d15436a471ea0d199ec15`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `6e6716f342ea1493d138ffb802fe1a631c51d655c64f92df331b804875584316`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `e6b9ee05eb1616cb1851383e3d3380d6feb764ec77d2ff2511cd24183479daf2`

## 방법·목표의 원문 필드

`$.phase`

```text
V147_PUBLIC_NATIVE_INPUT_ONLY_DINO_FEATURES
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "session_id": 17800,
  "terminal_exit_code": 0,
  "primary_encoder_calls": 64,
  "repeat_first_calls": 1,
  "views_per_call": 17,
  "rows": 512,
  "features_shape": [
    512,
    1928
  ],
  "elapsed_seconds": 4.374753409996629
}
```

`$.new_training`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
공개 원512개 입력의 동일 DINO1928 특징 추출과 전수 독립검산을 마쳤습니다. 가림 분류기 성능이 개선됐다는 결과가 아닙니다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_native_features_20261024_v147/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_native_features_20261024_v147/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_native_features_20261024_v147/ROOT_READOUT_QA.json](ROOT_READOUT_QA.json)
- [query_visibility_public_native_features_20261024_v147/public_native_DINO512_v1/FEATURE_COMPLETE.json](public_native_DINO512_v1/FEATURE_COMPLETE.json)
- [query_visibility_public_native_features_20261024_v147/independent_saved_feature_review/REVIEW_COMPLETE.json](independent_saved_feature_review/REVIEW_COMPLETE.json)
- [query_visibility_public_native_source_expansion_20261024_v146/ROOT_READOUT_KO.html](../query_visibility_public_native_source_expansion_20261024_v146/ROOT_READOUT_KO.html)
- [query_visibility_public_native_features_20261024_v147/independent_feature_review/FINAL_METADATA_PRECISION_FOLLOWUP.json](independent_feature_review/FINAL_METADATA_PRECISION_FOLLOWUP.json)
- [query_visibility_public_native_source_expansion_20261024_v146/ROOT_PHASE_RESULT.json](../query_visibility_public_native_source_expansion_20261024_v146/ROOT_PHASE_RESULT.json)
- [query_visibility_public_native_source_expansion_20261024_v146/source_reference_offset_diagnosis_agent/DIAGNOSIS_COMPLETE.json](../query_visibility_public_native_source_expansion_20261024_v146/source_reference_offset_diagnosis_agent/DIAGNOSIS_COMPLETE.json)

현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
