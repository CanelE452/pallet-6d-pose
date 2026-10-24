# V135 — query_visibility_native_pair_supervision_control_20261024_v135

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `9288f4cf58a5ea9384b487b014fe34e53404e9983f825e84760f884dc320b7c3`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `085b12eb075a32be73f76eb55069ff1fd43c322467ba3ff934e8d4947b7242b0`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `f879db4cf269316be6885a765645b2e6478c0b3ee449daf951daa3bb9c9470e9`

## 방법·목표의 원문 필드

`$.phase`

```text
V135_NATIVE_PAIR_SAMPLING_MARGIN_CONTROL_COMPLETE_BOTH_SOURCE_INELIGIBLE
```

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.

## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
같은 코너의 가림/근접 비가림 짝을 더 학습시키면 보임 오판은 줄지만 가림 검출도 줄었다. margin 추가 효과도 확인하지 못했다. 두 후보 모두 미채택이며 현재 실사 결과는 그대로다.
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_native_pair_supervision_control_20261024_v135/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_native_pair_supervision_control_20261024_v135/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/source_visual_review/READOUT_KO.html](../query_visibility_native_witness_diagnosis_20261024_v134/source_visual_review/READOUT_KO.html)
- [query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/C26_NATIVE_PAIR_SAMPLER/SOURCE_RESULTS.json](runs/C26_C27_NATIVE_PAIR_FIXED8000/C26_NATIVE_PAIR_SAMPLER/SOURCE_RESULTS.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/C26_NATIVE_PAIR_SAMPLER/MODEL_FREEZE.json](runs/C26_C27_NATIVE_PAIR_FIXED8000/C26_NATIVE_PAIR_SAMPLER/MODEL_FREEZE.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/C27_NATIVE_PAIR_SAMPLER_AND_MARGIN/SOURCE_RESULTS.json](runs/C26_C27_NATIVE_PAIR_FIXED8000/C27_NATIVE_PAIR_SAMPLER_AND_MARGIN/SOURCE_RESULTS.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/C27_NATIVE_PAIR_SAMPLER_AND_MARGIN/MODEL_FREEZE.json](runs/C26_C27_NATIVE_PAIR_FIXED8000/C27_NATIVE_PAIR_SAMPLER_AND_MARGIN/MODEL_FREEZE.json)
- [query_visibility_native_pair_supervision_control_20261024_v135/runs/C26_C27_NATIVE_PAIR_FIXED8000/CONTROL_COMPLETE.json](runs/C26_C27_NATIVE_PAIR_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/ROOT_DIAGNOSIS_BEFORE_CONTROLS.json](../query_visibility_native_witness_diagnosis_20261024_v134/ROOT_DIAGNOSIS_BEFORE_CONTROLS.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/same_before_query_control_v1/PREDICTION_SEAL.json](../query_visibility_native_witness_diagnosis_20261024_v134/same_before_query_control_v1/PREDICTION_SEAL.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/saved_fixed_query_evaluation_v1/SOURCE_FIXED_QUERY_DIAGNOSIS.json](../query_visibility_native_witness_diagnosis_20261024_v134/saved_fixed_query_evaluation_v1/SOURCE_FIXED_QUERY_DIAGNOSIS.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/REVIEW_COMPLETE_SEAL.json](../query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/REVIEW_COMPLETE_SEAL.json)
- [query_visibility_native_witness_diagnosis_20261024_v134/source_contract_review/SOURCE_CONTRACT_SUMMARY_KO.json](../query_visibility_native_witness_diagnosis_20261024_v134/source_contract_review/SOURCE_CONTRACT_SUMMARY_KO.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE01_BASE136_SLOT3_FROZEN_BEFORE_QUERY.png](../query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE01_BASE136_SLOT3_FROZEN_BEFORE_QUERY.png)

![원 보고서 진단 그림](../query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE01_BASE136_SLOT3_FROZEN_BEFORE_QUERY.png)

[query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE02_BASE134_SLOT0_FROZEN_BEFORE_QUERY.png](../query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE02_BASE134_SLOT0_FROZEN_BEFORE_QUERY.png)

![원 보고서 진단 그림](../query_visibility_native_witness_diagnosis_20261024_v134/fixed_query_saved_review/figures/CASE02_BASE134_SLOT0_FROZEN_BEFORE_QUERY.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
