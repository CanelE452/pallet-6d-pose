# V142 — query_visibility_native_target_gradient_control_20261024_v142

기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.

**상태:** {"completed":true,"PASS":true,"selected_model":null,"fully_solved":false,"goal_status":"active"}

원본:

- [ROOT_PHASE_RESULT.json](ROOT_PHASE_RESULT.json) — SHA256 `ce41ec4e30455ab761d0aea8f70b01ce6070af7719b492679ca4d37f1ee30145`
- [ROOT_READOUT_KO.html](ROOT_READOUT_KO.html) — SHA256 `5cf05188b4d21370478020e7ca0f2c55674e4ebd0e5285df6a90a5c35979b672`
- [ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json) — SHA256 `ec076e7d050ecc4d5df8548954d8458ebb82263cae78b0468e7a9ee7ea1c4563`

## 방법·목표의 원문 필드

`$.goal_turn_classification`

```text
PROGRESS
```


## 입력·학습·평가 분모의 원문 필드

`$.actual_execution`

```text
{
  "fits": 1,
  "new_baseline_fits": 0,
  "session_id": 19045,
  "exit_code": 0,
  "updates": 8000,
  "seconds": 60.13757442496717,
  "parameters": 224513,
  "seed": 1330,
  "C16_mixed_source_initialization_update": 4500,
  "optimizer": "Fresh AdamW lr1e-4 wd.001 clip5; FP32/TF32false",
  "new_encoder_forward": 0,
  "new_private_body_or_forward": 0,
  "V112_access": 0,
  "PnP_pose_q_L_correction": 0
}
```

`$.paired_all_cohorts_binding`

```text
{
  "path": "query_visibility_native_target_gradient_control_20261024_v142/independent_saved_review/PAIRED_C29_C30_TO_C32_TRANSITIONS.json",
  "sha256": "23d663fdd22d67e1c61152466e7572562d49e34f38f60009fedc9cf9c8ff6068"
}
```


## 결론·실패·한계의 원문 필드

`$.outcome_KO`

```text
AUX 일반cue/surface 학습 신호를 native scalar에서 제거하는 matched8000 대조를 실제 완료했다. 새 native VAL은7건 회복했지만 기존native VAL의D2건은 악화했고, 고정wood NEAR(2.54px)의 falseE도 남았다. 단일원인 해결이나 새 실사 성능 개선을 증명하지 못했다.
```

`$.interpretation_limits_KO`

```text
[
  "새VAL D1714→1717/1772, E112→116/130은 이 조건에서 AUXgradient제거가 준 작은이득이다. 기존VAL D808→806/828, E48/60변화없음도 함께보고한다.",
  "대상과같은재질 foreignfamily는목재와polymer를합친범주다. 실제목재∩같은재질교집합과목재∩같은재질∩LE3를별도로집계했다.",
  "C32 native35중20통과로C29의18보다2늘었지만15기준실패가남았다. AUX29는26→22로낮아졌다. 전체64조건은그대로이며후보채택하지않았다.",
  "부분개선은 label목표혼합이유일실패원인이라는증거가아니다. 보이는틈/올바른part를읽는표현, 실제reference역할과domain전이 문제는남는다.",
  "같은base의여러slot/variant는상관된다. 7건회복을독립표본의유의성이나실사전이로해석하지않는다.",
  "초기좌표오차GT구간은진단용이다. 입력보정이나실패제외에사용하지않았다. S/U정답은외부가림negative가아니고, 예측U는knownDE분모에남겼다."
]
```


## 코드·검증·전체 표·갤러리

아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.

- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_RESULTS_KO.json](ROOT_RESULTS_KO.json)
- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_READOUT_KO.html](ROOT_READOUT_KO.html)
- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_REPORT_VISUAL_QA.json](ROOT_REPORT_VISUAL_QA.json)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_READOUT_KO.html](../query_visibility_public_native_initializer_association_20261024_v141/ROOT_READOUT_KO.html)
- [query_visibility_native_target_gradient_control_20261024_v142/independent_saved_review/PAIRED_C29_C30_TO_C32_TRANSITIONS.json](independent_saved_review/PAIRED_C29_C30_TO_C32_TRANSITIONS.json)
- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_NATIVE_GRADIENT_CONTROL_POLICY_BEFORE_FIT.json](ROOT_NATIVE_GRADIENT_CONTROL_POLICY_BEFORE_FIT.json)
- [query_visibility_native_target_gradient_control_20261024_v142/ROOT_ACTUAL_C32_SOURCE_FIT_ENTRY.json](ROOT_ACTUAL_C32_SOURCE_FIT_ENTRY.json)
- [query_visibility_native_target_gradient_control_20261024_v142/runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/CONTROL_COMPLETE.json](runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/CONTROL_COMPLETE.json)
- [query_visibility_native_target_gradient_control_20261024_v142/runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/FIT_COMPLETE.json](runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/FIT_COMPLETE.json)
- [query_visibility_native_target_gradient_control_20261024_v142/runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/MODEL_FREEZE.json](runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/C32_AUX_TO_NATIVE_GRAD0_FIXED8000/MODEL_FREEZE.json)
- [query_visibility_native_target_gradient_control_20261024_v142/independent_saved_review/REVIEW.json](independent_saved_review/REVIEW.json)
- [query_visibility_native_target_gradient_control_20261024_v142/EXISTING_METHOD_AND_TARGET_ONTOLOGY_AUDIT.json](EXISTING_METHOD_AND_TARGET_ONTOLOGY_AUDIT.json)
- [query_visibility_native_target_gradient_control_20261024_v142/saved_native_gradient_fixed_visual_report_v1/VISUAL_REPORT_COMPLETE.json](saved_native_gradient_fixed_visual_report_v1/VISUAL_REPORT_COMPLETE.json)
- [query_visibility_public_native_initializer_association_20261024_v141/ROOT_PHASE_RESULT.json](../query_visibility_public_native_initializer_association_20261024_v141/ROOT_PHASE_RESULT.json)

## 원 보고서의 그림

원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.

[query_visibility_native_target_gradient_control_20261024_v142/saved_native_gradient_fixed_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C32_RGB96.png](saved_native_gradient_fixed_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C32_RGB96.png)

![원 보고서 진단 그림](saved_native_gradient_fixed_visual_report_v1/BASE_0519_NATIVE_0_C16_V61_C29_C32_RGB96.png)

[query_visibility_native_target_gradient_control_20261024_v142/saved_native_gradient_fixed_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C32_RGB96.png](saved_native_gradient_fixed_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C32_RGB96.png)

![원 보고서 진단 그림](saved_native_gradient_fixed_visual_report_v1/BASE_0590_NATIVE_1_C16_V61_C29_C32_RGB96.png)


현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.
