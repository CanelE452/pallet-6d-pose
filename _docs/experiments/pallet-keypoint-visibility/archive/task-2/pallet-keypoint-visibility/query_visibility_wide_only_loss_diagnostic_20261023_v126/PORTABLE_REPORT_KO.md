# V126 — query_visibility_wide_only_loss_diagnostic_20261023_v126

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `2d59ddd1be853ea93af08dafb6b178bf95238cee74e1ea552347804b4340d749`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `36f2f656caef4e531050b83d405b424c67f4d920402aa65f83c42d3216b4b0ff`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `b39627563fb833c70cae590ee969b85c09ec5cf1e84a86d0cf4c1e1e8164e041`

## 방법·목표의 원문 필드

`$.phase`

```text
V126_C18_MATCHED_OLDLOSS_ZERO_CONTROL_AND_TRAIN_CAPACITY_DIAGNOSIS
```

`$.goal_assessment_KO`

```text
계속 검증할 근거는 있으나 성공을 예측할 근거는 아직 부족하다. 목표가 불가능하다고 판정할 결과도 없다. 학습 자료 적합과 새 장면 일반화를 분리하는 단계다.
```


## 입력·학습·평가 분모의 원문 필드

`$.primary_fixed1000_control`

```text
{
  "excerpt": "{\"C17_FINAL1000\": {\"TRAIN795\": {\"CM\": [[368, 55, 0], [67, 305, 0]], \"E_qOUT\": {\"known_N\": 156, \"available_N\": 156, \"correct\": 109}, \"E_qIN\": {\"known_N\": 216, \"available_N\": 216, \"correct\": 196}, \"SHAM_D_qIN\": {\"known_N\": 77, \"available_N\": 77, \"correct\": 44}, \"SHAM_D_qOUT\": {\"known_N\": 295, \"available_N\": 295, \"correct\": 273}}, \"LOCO_VALIDATION\": {\"CM\": [[198, 126, 0], [86, 202, 0]], \"E_qOUT\": {\"known_N\": 144, \"available_N\": 144, \"correct\": 70}, \"E_qIN\": {\"known_N\": 144, \"available_N\": 144, \"correct\": 132}, \"SHAM_D_qIN\": {\"known_N\": 69, \"available_N\": 69, \"correct\": 7}, \"SHAM_D_qOUT\": {\"known_N\": 219, \"available_N\": 219, \"correct\": 161}}, \"TOMIE_DEVELOPMENT\": {\"CM\": [[1179, 405, 0], [400, 986, 0]], \"E_qOUT\": {\"known_N\": 339, \"available_N\": 339, \"correct\": 153}, \"E_qIN\": {\"known_N\": 1047, \"available_N\": 1047, \"correct\": 833}, \"SHAM_D_qIN\": {\"known_N\": 325, \"available_N\": 325, \"correct\": 114}, \"SHAM_D_qOUT\": {\"known_N\": 1061, \"available_N\": 1061, \"correct\": 882}}, \"VALIDATION_COMBINED\": {\"CM\": [[1377, 531, 0], [486, 1188, 0]], \"E_qOUT\": {\"known_N\": 483, \"available_N\": 483, \"correct\": 223}, \"E_qIN\": {\"known_N\": 1191, \"available_N\": 1191, \"correct\": 965}, \"SHAM_D_qIN\": {\"known_N\": 394, \"available_N\": 394, \"correct\": 121}, \"SHAM_D_qOUT\": {\"known_N\": 1280, \"available_N\": 1280, \"correct\": 1043}}}, \"C18_FINAL1000\": {\"TRAIN795\": {\"CM\": [[369, 54, 0], [70, 302, 0]], \"E_qOUT\": {\"known_N\": 156, \"available_N\": 156, \"correct\": 108}, \"E_qIN\": {\"known_N\": 216, \"available_N\": 216, \"correct\": 194}, \"SHAM_D_qIN\": {\"known_N\": 77, \"available_N\": 77, \"correct\": 49}, \"SHAM_D_qOUT\": {\"known_N\": 29",
  "truncated": true
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
원인 진단은 진전됐지만 새 장면의 외부 가림 판단은 해결되지 않았다.
```

`$.limits_KO`

```text
[
  "공개 합성가림은 기록된 특정 구조 cue를 기준으로 한 조건부 AUX 라벨이다. native8 물리 키포인트의 인간 정답과 동일하다고 주장하지 않는다.",
  "현재223은 이전 논의에서 이미 본 개발 평가셋으로 새 blind holdout이 아니다. 이번 실험의 학습/문턱/체크포인트 선택에는 사용하지 않았다.",
  "기존 손실의 영향이 모든 설정에서 없다고 증명한 실험이 아니다. 특징이나 head의 일반적인 충분성도 증명하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/saved_controlled_review/REVIEW.json](saved_controlled_review/REVIEW.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/capacity_probe_review/REVIEW.json](capacity_probe_review/REVIEW.json)
- [query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html](../query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_C18_DESIGN_BEFORE_SOURCE_FIT.json](ROOT_C18_DESIGN_BEFORE_SOURCE_FIT.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_ACTUAL_C18_SOURCE_FIT_ENTRY.json](ROOT_ACTUAL_C18_SOURCE_FIT_ENTRY.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_CAPACITY_PROBE_POLICY_BEFORE_FIT.json](ROOT_CAPACITY_PROBE_POLICY_BEFORE_FIT.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/ROOT_ACTUAL_CAPACITY_PROBE_ENTRY.json](ROOT_ACTUAL_CAPACITY_PROBE_ENTRY.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/capacity_probe/FULL_BATCH_LBFGS_MAX400/RESULTS.json](capacity_probe/FULL_BATCH_LBFGS_MAX400/RESULTS.json)
- [query_visibility_wide_only_loss_diagnostic_20261023_v126/representation_contract_review/REVIEW.json](representation_contract_review/REVIEW.json)
- [query_visibility_training_duration_control_20261023_v124/ROOT_RESULTS_KO.json](../query_visibility_training_duration_control_20261023_v124/ROOT_RESULTS_KO.json)
- [query_visibility_training_duration_control_20261023_v124/ROOT_REPORT_VISUAL_QA.json](../query_visibility_training_duration_control_20261023_v124/ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_training_duration_control_20261023_v124/runs/C16_EXACT_C15_DURATION_CONTROL_8000/MODEL_FREEZE.json](../query_visibility_training_duration_control_20261023_v124/runs/C16_EXACT_C15_DURATION_CONTROL_8000/MODEL_FREEZE.json)
- [query_visibility_training_duration_control_20261023_v124/runs/C16_EXACT_C15_DURATION_CONTROL_8000/FIT_COMPLETE.json](../query_visibility_training_duration_control_20261023_v124/runs/C16_EXACT_C15_DURATION_CONTROL_8000/FIT_COMPLETE.json)
- [query_visibility_training_duration_control_20261023_v124/private_evaluation/C16_FROZEN_CURRENT223/INFERENCE_SEAL.json](../query_visibility_training_duration_control_20261023_v124/private_evaluation/C16_FROZEN_CURRENT223/INFERENCE_SEAL.json)
- [query_visibility_training_duration_control_20261023_v124/private_evaluation/C16_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json](../query_visibility_training_duration_control_20261023_v124/private_evaluation/C16_FROZEN_CURRENT223/evaluation/EVALUATION_SEAL.json)
- [query_visibility_training_duration_control_20261023_v124/saved_current_review/REVIEW.json](../query_visibility_training_duration_control_20261023_v124/saved_current_review/REVIEW.json)
- [query_visibility_training_duration_control_20261023_v124/saved_source_review/REVIEW.json](../query_visibility_training_duration_control_20261023_v124/saved_source_review/REVIEW.json)
- [query_visibility_training_duration_control_20261023_v124/ROOT_PHASE_RESULT.json](../query_visibility_training_duration_control_20261023_v124/ROOT_PHASE_RESULT.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_wide_only_loss_diagnostic_20261023_v126/visual_followup/STATUS_AND_REMAINING_GAP_KO.png](visual_followup/STATUS_AND_REMAINING_GAP_KO.png)

![원 보고서 진단 그림](visual_followup/STATUS_AND_REMAINING_GAP_KO.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
