# V139 — query_visibility_broad_source_geometry_control_20261024_v139

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"selected_model":null,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `457e318f11aa1ef59a455718c9a01b6c5b9bda3c1839b7584a50bda8e2116b70`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `066f3e90117a5c8ad01bf80a0b2da7dc84198196ecc7d5f84f535d1d09b15529`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `75333d3ea753aabc5148c12856628e47abdb939f33cdc47e925445c6b3d60958`

## 방법·목표의 원문 필드

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "session_id": 83370,
  "exit_code": 0,
  "arms": 2,
  "updates_each": 8000,
  "parameters_each": 224513,
  "all_jobs_terminal": true,
  "current_forward": 0,
  "PnP": 0,
  "q_L_correction": 0
}
```

`$.C30_all_outputs_training_and_final_state_exact_C29`

```text
true
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
넓힌 source에서도 G42 추가는 보임 일부만 개선하고 외부 가림은 개선하지 못했다. 동일 예산 G0 대조는 이전 C29를 완전히 재현했다. 두 모델 모두 전체 기준 미달로 미채택이다.
```

`$.limitations_KO`

```text
[
  "이번 결과는 이 head와 학습 조건에서의 G42 비교다. 모든 구조 정보의 무용성이나 다른 모델의 불가능성을 증명하지 않는다.",
  "좌표 오차3px 이하 등 GT 구간은 진단용이다. 입력 좌표를 옮기거나 평가에서 실패를 제거하지 않았다.",
  "이 합성자료의 D/E는 등록된 코너의 보임/외부 가림이다. D 출력이 초기 query 좌표의 정확성을 보증하지 않는다.",
  "정답 자기 가림 S와 불확실 U는 외부 가림 이진 학습의 음성이 아니다. 예측 보류 U는 전체 known D/E 분모에 남겼다.",
  "같은 기본 장면과 여러 slot/variant는 상관된다. 작은 수의 개선을 독립 표본 유의성이나 실사 전이로 해석하지 않는다.",
  "현재223으로 학습·문턱·체크포인트·모델을 선택하지 않았고 새 실사 추론도 하지 않았다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_broad_source_geometry_control_20261024_v139/runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/CONTROL_COMPLETE.json](runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_ACTUAL_SOURCE_FIT_ENTRY.json](ROOT_ACTUAL_SOURCE_FIT_ENTRY.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_diagnosis_v1/DIAGNOSIS_SEAL.json](saved_geometry_control_diagnosis_v1/DIAGNOSIS_SEAL.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_visual_report_v1/VISUAL_REPORT_COMPLETE.json](saved_geometry_control_visual_report_v1/VISUAL_REPORT_COMPLETE.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_NEXT_PUBLIC_NATIVE_FEASIBILITY_SCOPE.json](ROOT_NEXT_PUBLIC_NATIVE_FEASIBILITY_SCOPE.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_diagnosis_v1/INDEX.html](saved_geometry_control_diagnosis_v1/INDEX.html)
- [query_visibility_broad_source_geometry_control_20261024_v139/ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json](ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL/C30_EXACT_C29_REPLAY.json](runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/C30_BROAD_SOURCE_ZERO_GEOMETRY_CONTROL/C30_EXACT_C29_REPLAY.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/independent_saved_review/PAIRED_C30_C31_FULL_COHORT_TRANSITIONS.json](independent_saved_review/PAIRED_C30_C31_FULL_COHORT_TRANSITIONS.json.parts.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_diagnosis_v1/SOURCE_RESULTS.json](saved_geometry_control_diagnosis_v1/SOURCE_RESULTS.json.parts.json)
- [query_visibility_broad_source_geometry_control_20261024_v139/public_native_source_readiness/PUBLIC_NATIVE_READINESS_EXISTING_METADATA_ONLY.json](public_native_source_readiness/PUBLIC_NATIVE_READINESS_EXISTING_METADATA_ONLY.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C30_C31_RGB96.png](saved_geometry_control_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C30_C31_RGB96.png)

![원 보고서 진단 그림](saved_geometry_control_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C30_C31_RGB96.png)

[query_visibility_broad_source_geometry_control_20261024_v139/saved_geometry_control_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C30_C31_RGB96.png](saved_geometry_control_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C30_C31_RGB96.png)

![원 보고서 진단 그림](saved_geometry_control_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C30_C31_RGB96.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.

큰 원문 링크는 무손실 gzip의 `.parts.json` descriptor로 연결합니다. 원문 SHA와 복원 방법은 게시 안내를 따릅니다.
