# V143 — query_visibility_frozen_native_current_benchmark_20261024_v143

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"selected_model":null,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `0ca3b87e32df19702708caf919e0004656ed570fadf96db3f8b2b0b2875cc66c`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `dccac687d99ef102ec7f765aa30c71ffd5ca8063924fbd7868477aa331ad8a84`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `45e07f7847904c99831b304526c08627c36a95c1ea0a4973dc9477d33220c15c`

## 방법·목표의 원문 필드

자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.

## 입력·학습·평가 분모의 원문 필드

`$.aggregate_primary`

```text
{
  "excerpt": "[{\"model\": \"v61\", \"CM_D_E_by_pred_D_E_U\": [[260, 887, 0], [2, 88, 0]], \"D_precision\": 0.9923664122137404, \"D_recall\": 0.22667829119442023, \"E_precision\": 0.09025641025641026, \"E_recall\": 0.9777777777777777, \"U\": 0.0, \"counts\": {\"N\": 1237, \"visible_support\": 1147, \"occluded_support\": 90, \"TP\": 260, \"FN_decided\": 887, \"FP\": 2, \"TN\": 88, \"U_visible\": 0, \"U_occluded\": 0, \"out_of_task_S_visible\": 0, \"out_of_task_S_occluded\": 0, \"other_unresolved_visible\": 0, \"other_unresolved_occluded\": 0}}, {\"model\": \"legacy\", \"CM_D_E_by_pred_D_E_U\": [[655, 319, 173], [3, 83, 4]], \"D_precision\": 0.9954407294832827, \"D_recall\": 0.5710549258936356, \"E_precision\": 0.2064676616915423, \"E_recall\": 0.9222222222222223, \"U\": 0.14308811641067098, \"counts\": {\"N\": 1237, \"visible_support\": 1147, \"occluded_support\": 90, \"TP\": 655, \"FN_decided\": 319, \"FP\": 3, \"TN\": 83, \"U_visible\": 173, \"U_occluded\": 4, \"out_of_task_S_visible\": 0, \"out_of_task_S_occluded\": 0, \"other_unresolved_visible\": 0, \"other_unresolved_occluded\": 0}}, {\"model\": \"C16\", \"CM_D_E_by_pred_D_E_U\": [[1137, 10, 0], [55, 35, 0]], \"D_precision\": 0.9538590604026845, \"D_recall\": 0.99128160418483, \"E_precision\": 0.7777777777777778, \"E_recall\": 0.3888888888888889, \"U\": 0.0, \"counts\": {\"N\": 1237, \"visible_support\": 1147, \"occluded_support\": 90, \"TP\": 1137, \"FN_decided\": 10, \"FP\": 55, \"TN\": 35, \"U_visible\": 0, \"U_occluded\": 0, \"out_of_task_S_visible\": 0, \"out_of_task_S_occluded\": 0, \"other_unresolved_visible\": 0, \"other_unresolved_occluded\": 0}}, {\"model\": \"C29\", \"CM_D_E_by_pred_D_E_U\": [[1113, 34, 0], [57, 33, 0]], \"D_precision\": 0.9512820512820512, \"",
  "truncated": true
}
```

`$.actual_execution`

```text
{
  "head_session": 44629,
  "head_exit_code": 0,
  "saved_evaluation_exit_code": 0,
  "new_training": 0,
  "new_encoder_RGB_initializer": 0,
  "PnP_or_q_L_correction": 0,
  "C16_original_byteparity": true
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
새 합성 검증의 작은 개선은 실사 전체 개선으로 이어지지 않았다. C32는 C16 대비 E 정답 +3/90이지만 D 정답 −67/1147이며, 수동 D·오차≤3px에서도 19/376 오판이 생겼다. 해결 또는 모델 채택 근거가 아니다.
```

`$.limitations_KO`

```text
[
  "현재223은 개발 노출 자료이며 독립 blind test가 아닙니다.",
  "사람 S는 별도 응답 진단이며 binary hidden이 학습된 self/외부 원인 인증을 뜻하지 않습니다.",
  "원래 reference 오차와 manualD376은 진단이며 primary 필터·좌표 수정·모델 선택에 사용하지 않았습니다.",
  "C29/C32 source qualification 실패와 current_allowedfalse는 불변이며 두 모델을 모두 보고하고 채택하지 않았습니다.",
  "기존 semantic reference D/E는 실사 단일 픽셀 presence의 gold가 아니다.",
  "실사 CAL46은 현재 자료 일부와 학습이 겹친 과거 참고이며 독립 일반화 비교가 아니다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_frozen_native_current_benchmark_20261024_v143/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/independent_saved_evaluation_review/REVIEW.json](independent_saved_evaluation_review/REVIEW.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/frozen_current_head_benchmark_v1/PREDICTION_SEAL.json](frozen_current_head_benchmark_v1/PREDICTION_SEAL.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/saved_current_benchmark_evaluation_v1/EVALUATION_SEAL.json](saved_current_benchmark_evaluation_v1/EVALUATION_SEAL.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/saved_current_benchmark_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json](saved_current_benchmark_evaluation_v1/CURRENT_FROZEN_BENCHMARK_RESULTS.json)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/saved_current_benchmark_evaluation_v1/COORDINATE_ERROR_BINS.csv](saved_current_benchmark_evaluation_v1/COORDINATE_ERROR_BINS.csv)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/saved_current_benchmark_evaluation_v1/ALL1784_NATIVE_LEDGER.csv](saved_current_benchmark_evaluation_v1/ALL1784_NATIVE_LEDGER.csv)
- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_READOUT_KO.html](../query_visibility_native_target_gradient_control_20261024_v142/ROOT_READOUT_KO.html)
- [query_visibility_frozen_native_current_benchmark_20261024_v143/independent_saved_prediction_review/REVIEW.json](independent_saved_prediction_review/REVIEW.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_frozen_native_current_benchmark_20261024_v143/REAL_TRANSFER_READOUT_KO.png](REAL_TRANSFER_READOUT_KO.png)

![원 보고서 진단 그림](REAL_TRANSFER_READOUT_KO.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
