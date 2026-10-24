# V131 — query_visibility_public_small_query_jitter_20261024_v131

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `df9bedbb98aec93026576cf0fa4adfae7515d55c1e9b647f2b673f2f18f77437`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `d82f89ced61c77864b628c4be8fbd07778a4d2224308d781e5446686b228149d`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `a8609f451547460d276d8676cac0c727deabd940417133ae35995df64eeb600b`

## 방법·목표의 원문 필드

`$.phase`

```text
V131_FROZEN_SMALL_QUERY_DIAGNOSIS_COMPLETE
```

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

`$.new_fit_updates`

```text
0
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
공개 약한 cue의 작은 좌표 이동에서는 C16이 안정적이다. 실제 외부 가림 성능 개선은 아직 입증하지 못했다.
```

`$.interpretation_KO`

```text
[
  "VAL306은 원본 2개 프레임의34개합성장면을9회씩 변형한 상관된 입력이다. 306개의 독립 실사 사례가 아니다.",
  "LE3 VAL170의 C16 보임90/90·외부가림80/80은 약한 cue에 대한 작은 이동 안정성이다. 실제 키포인트 정확도 인증이나 자연가림 전이 성공을 뜻하지 않는다.",
  "이전 큰 q 이동(약36~53px)에서 C16 보임100/162·외부가림87/144였으므로 query 위치와 crop/context 변화가 이 source 실패의 큰 요인이다. 모든 실사 실패의 단일 원인으로 확정하지 않는다.",
  "C23은 작은 이동에서도 C16보다 낫지 않고 기존 전체 source8개기준중4개만 통과했다. V131 진단으로 기존 채택기준을 바꾸거나 C23을 채택하지 않는다.",
  "v61의 query-pixel presence와 의미상 키포인트 가림은 목표가 다르다. 높은 가림 recall만으로 유용성을 판단하지 않는다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_public_small_query_jitter_20261024_v131/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_public_small_query_jitter_20261024_v131/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_public_small_query_jitter_20261024_v131/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/EVALUATION_SEAL.json](saved_three_model_evaluation_v1/EVALUATION_SEAL.json)
- [query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/READOUT_KO.html](saved_three_model_evaluation_v1/READOUT_KO.html)
- [query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html](../query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html)
- [query_visibility_source_coverage_duration_20261024_v130/ROOT_READOUT_KO.html](../query_visibility_source_coverage_duration_20261024_v130/ROOT_READOUT_KO.html)
- [query_visibility_public_small_query_jitter_20261024_v131/ROOT_SMALL_QUERY_DIAGNOSIS_POLICY_BEFORE_INPUTS.json](ROOT_SMALL_QUERY_DIAGNOSIS_POLICY_BEFORE_INPUTS.json)
- [query_visibility_public_small_query_jitter_20261024_v131/source_crops_v1/CROP_COMPLETE.json](source_crops_v1/CROP_COMPLETE.json)
- [query_visibility_public_small_query_jitter_20261024_v131/frozen_C16_C23_small_jitter_v1/PREDICTION_SEAL.json](frozen_C16_C23_small_jitter_v1/PREDICTION_SEAL.json)
- [query_visibility_public_small_query_jitter_20261024_v131/v61_frozen_source_baseline_v1/PREDICTION_SEAL.json](v61_frozen_source_baseline_v1/PREDICTION_SEAL.json)
- [query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/SOURCE_RESULTS.json](saved_three_model_evaluation_v1/SOURCE_RESULTS.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/figures/THREE_MODELS_SMALL_JITTER_SUMMARY.png](saved_three_model_evaluation_v1/figures/THREE_MODELS_SMALL_JITTER_SUMMARY.png)

![원 보고서 진단 그림](saved_three_model_evaluation_v1/figures/THREE_MODELS_SMALL_JITTER_SUMMARY.png)

[query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/figures/SCENE_038_ALL9_RAW_AND_MARKED.png](saved_three_model_evaluation_v1/figures/SCENE_038_ALL9_RAW_AND_MARKED.png)

![원 보고서 진단 그림](saved_three_model_evaluation_v1/figures/SCENE_038_ALL9_RAW_AND_MARKED.png)

[query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/figures/SCENE_039_ALL9_RAW_AND_MARKED.png](saved_three_model_evaluation_v1/figures/SCENE_039_ALL9_RAW_AND_MARKED.png)

![원 보고서 진단 그림](saved_three_model_evaluation_v1/figures/SCENE_039_ALL9_RAW_AND_MARKED.png)

[query_visibility_public_small_query_jitter_20261024_v131/saved_three_model_evaluation_v1/figures/SCENE_042_ALL9_RAW_AND_MARKED.png](saved_three_model_evaluation_v1/figures/SCENE_042_ALL9_RAW_AND_MARKED.png)

![원 보고서 진단 그림](saved_three_model_evaluation_v1/figures/SCENE_042_ALL9_RAW_AND_MARKED.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
